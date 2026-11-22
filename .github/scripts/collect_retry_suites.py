#!/usr/bin/env python3
"""Builds the pod-level-retry matrix for _test_matrix.yaml from the failed-suite artifacts.

Called twice per run: the early wave polls (--wait-for-failure) and returns at the first
failure; the final wave runs after fan-in and retries what the early wave did not claim.

Usage (called by the GHA workflow; the flag options need --opt=value, see build_parser):
    python3 collect_retry_suites.py \
        --repo "$GITHUB_REPOSITORY" --run-id "$GITHUB_RUN_ID" \
        --call-id "$CALL_ID" --run-attempt "$GITHUB_RUN_ATTEMPT" \
        --config-root tests/configs --test-type regression \
        --extra-test-flags="-v" --extra-test-flags-multi="" \
        --out-dir failed_suites [--wait-for-failure]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Deliberately stdlib-only: the collector jobs run bare `python3` with no venv.

# Single-card suites land on this pool unless the descriptor names another; multi-card
# suites always need the two-card pool. Mirrors the `runs-on` of `test`/`test_multi_spyre`.
DEFAULT_SINGLE_RUNNER = "spyre_pf_x1"
MULTI_RUNNER = "spyre_pf_x2"

# The multi-card matrix spells its configs relative to tests/configs/ and does not use
# generate_matrix's config_root (which can point at a subdir, e.g. model_ops_tests).
MULTI_CONFIG_ROOT = "tests/configs"


def _report_stem(config: str) -> str:
    """`basename "${CONFIG%.yaml}"`, the shell the test jobs use to name a report."""
    return Path(config[: -len(".yaml")] if config.endswith(".yaml") else config).name


def normalize_single(suite: dict, config_root: str, extra_flags: str, test_type: str):
    """A `test` leg's descriptor -> a retry entry. Mirrors that job's Derive report paths."""
    config = suite.get("config", "")
    # A bucket (several short configs in one job) names its report after the bucket and
    # runs every member in one invocation; anything else runs its single config.
    members = (suite.get("configs") or config).split()
    return {
        "name": suite.get("name", ""),
        "kind": "single",
        "runner": suite.get("runner") or DEFAULT_SINGLE_RUNNER,
        "config_paths": " ".join(f"{config_root}/{m}" for m in members),
        "report_name": f"{suite.get('report') or _report_stem(config)}.xml",
        "extra_test_flags": suite.get("extra_test_flags") or extra_flags,
        "parallel": "true",
        "test_type": test_type,
    }


def normalize_multi(suite: dict, extra_flags: str):
    """A `test_multi_spyre` leg's descriptor -> a retry entry.

    No TORCH_SPYRE_TEST_TYPE: that job does not set one either, and the OOT framework
    reads it with a "" default, so an empty value is the same as unset.
    """
    config = suite.get("config", "")
    return {
        "name": suite.get("name", ""),
        "kind": "multi",
        "runner": MULTI_RUNNER,
        "config_paths": f"{MULTI_CONFIG_ROOT}/{config}",
        "report_name": f"{_report_stem(config)}.xml",
        "extra_test_flags": extra_flags,
        # The distributed suites drive torchrun themselves and must not be xdist-split.
        "parallel": "false",
        "test_type": "",
    }


def entry_key(entry: dict) -> str:
    """Dedup/exclusion key: the whole normalized entry, so a wave can never re-run what an
    earlier wave claimed, nor skip a suite that merely shares a name with one."""
    return json.dumps(entry, sort_keys=True)


def build_matrix(
    singles, multis, config_root, extra_flags, extra_flags_multi, test_type, exclude=()
):
    """Normalized retry entries for the given raw descriptors, minus `exclude`."""
    normalized = [
        normalize_single(s, config_root, extra_flags, test_type) for s in singles
    ] + [normalize_multi(s, extra_flags_multi) for s in multis]

    seen = set(exclude)
    entries = []
    for entry in normalized:
        if entry_key(entry) in seen:
            continue
        seen.add(entry_key(entry))
        # A missing runner label would leave `runs-on` unsatisfiable and the leg would
        # hang until the job timeout instead of failing loudly.
        assert entry["runner"], f"no runner label for {entry['name']!r}"
        entries.append(entry)
    return entries


def list_artifacts(repo: str, run_id: str):
    """Every artifact name in the run, as {name: id}.

    --paginate: the endpoint defaults to 30 per page and a regression run uploads one XML
    per suite plus the wheel, well over one page. An un-paginated query silently drops
    everything past page 1, so a suite whose descriptor landed later would never be
    retried and the gate would report green. --paginate applies --jq per page and
    concatenates (gh rejects --slurp with --jq), so parse a per-object JSON stream.
    """
    listing = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            "--jq",
            ".artifacts[] | {id: .id, name: .name}",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return {
        obj["name"]: obj["id"]
        for obj in (json.loads(ln) for ln in listing.stdout.splitlines() if ln.strip())
    }


def download_suites(repo: str, names_to_ids: dict, out_dir: Path):
    """Each named descriptor artifact's suite.json, in sorted-by-name order."""
    suites = []
    for name in sorted(names_to_ids):
        result = subprocess.run(
            ["gh", "api", f"/repos/{repo}/actions/artifacts/{names_to_ids[name]}/zip"],
            capture_output=True,
            check=True,
        )
        dest = out_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / f"{name}.zip"
        zip_path.write_bytes(result.stdout)
        subprocess.run(
            ["unzip", "-q", "-o", str(zip_path), "-d", str(dest)], check=True
        )
        suites.append(json.loads((dest / "suite.json").read_text()))
    return suites


def select(artifacts: dict, prefix: str):
    """This call's and attempt's single- and multi-card descriptors.

    call_id and attempt are in the artifact names because the artifacts endpoint is
    run-scoped: it returns descriptors from every call of this reusable workflow in the run
    (runtests_nightly.yaml calls it twice) and from every attempt. Filtering on both stops
    one call from retrying another's suites with its own wrong flags, and a re-run from
    retrying a suite that failed on an earlier attempt but passed on this one.

    `failed-suite-<call_id>-` cannot match a multi-card descriptor: the segment after
    `failed-suite-` is then the literal `multi`, never a call_id.
    """
    single = {
        n: i for n, i in artifacts.items() if n.startswith(f"failed-suite-{prefix}")
    }
    multi = {
        n: i
        for n, i in artifacts.items()
        if n.startswith(f"failed-suite-multi-{prefix}")
    }
    return single, multi


def wait_for_failure(repo, run_id, prefix, marker, poll_secs, deadline_secs):
    """Poll until the first descriptor appears, the matrix fans in, or the deadline passes.

    Returning at the first failure is the point of the early wave. Returning on `marker`
    (the fan-in job's artifact) is what keeps a green run fast: without it this would poll
    to the deadline on every run that never fails.

    A visible marker means every leg has finished, and a leg uploads its descriptor before
    it finishes, so no failure is still in flight -- at worst an upload is not yet visible
    through the eventually-consistent artifacts API, and the final wave, which lists again
    after the fan-in, picks it up.
    """
    deadline = time.monotonic() + deadline_secs
    while True:
        artifacts = list_artifacts(repo, run_id)
        single, multi = select(artifacts, prefix)
        if single or multi:
            print(
                f"First failure(s) visible: {len(single)} single, {len(multi)} multi."
            )
            return artifacts
        if marker in artifacts:
            print(
                "Test matrix fanned in with no failed-suite descriptor; nothing to do."
            )
            return artifacts
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Safety valve only: the final wave still retries everything that failed.
            print(
                f"::warning::Gave up waiting for the test matrix after {deadline_secs}s; "
                "leaving every retry to the final wave."
            )
            return artifacts
        time.sleep(min(poll_secs, remaining))


def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    The flag-carrying options (--extra-test-flags, --extra-test-flags-multi) hold values
    that start with a dash -- "-v" on a PR, "--cpu-compile -vvv -s" on nightly -- which
    argparse reads as another option unless the caller uses the --opt=value form. The
    workflow spells those two that way; tests/test_collect_retry_suites.py pins it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--run-id", required=True, help="Workflow run ID")
    parser.add_argument("--call-id", required=True, help="generate_matrix's call_id")
    parser.add_argument("--run-attempt", required=True, help="GITHUB_RUN_ATTEMPT")
    parser.add_argument(
        "--config-root", required=True, help="generate_matrix's config_root"
    )
    parser.add_argument(
        "--test-type", default="", help="resolved_test_type, for the retried leg's env"
    )
    parser.add_argument(
        "--extra-test-flags", default="", help="inputs.extra_test_flags"
    )
    parser.add_argument(
        "--extra-test-flags-multi",
        default="",
        help="inputs.extra_test_flags_multi_spyre",
    )
    parser.add_argument(
        "--out-dir", default="failed_suites", help="Where to unzip them"
    )
    parser.add_argument(
        "--wait-for-failure",
        action="store_true",
        help="Early wave: poll until the first failure, the fan-in marker, or the deadline",
    )
    parser.add_argument(
        "--poll-secs",
        type=int,
        default=60,
        help=(
            "Seconds between artifact listings. A minute is deliberately unhurried: the "
            "retry it starts takes many minutes, so finer polling buys noise, while "
            "GITHUB_TOKEN's REST budget is per repository per hour and nightly runs two "
            "of these pollers at once alongside the rest of CI."
        ),
    )
    parser.add_argument(
        "--deadline-secs",
        type=int,
        default=4 * 60 * 60,
        help="Cap on --wait-for-failure polling; the final wave covers whatever is missed",
    )
    parser.add_argument(
        "--exclude-matrix",
        default="",
        help="Final wave: the early wave's `matrix` output, whose suites it already claimed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    prefix = f"{args.call_id}-{args.run_attempt}-"
    marker = f"legs-complete-{args.call_id}-{args.run_attempt}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_for_failure:
        artifacts = wait_for_failure(
            args.repo, args.run_id, prefix, marker, args.poll_secs, args.deadline_secs
        )
    else:
        artifacts = list_artifacts(args.repo, args.run_id)

    single_arts, multi_arts = select(artifacts, prefix)
    print(f"Descriptors: {len(single_arts)} single-card, {len(multi_arts)} multi-card.")

    # An empty --exclude-matrix (the early wave was skipped, or claimed nothing) excludes
    # nothing, so a wave that did not run can never cost a failed suite its retry.
    exclude = []
    if args.exclude_matrix.strip():
        exclude = [
            entry_key(e) for e in json.loads(args.exclude_matrix).get("suite", [])
        ]
        print(f"Already claimed by the early wave: {len(exclude)} suite(s).")

    suites = build_matrix(
        download_suites(args.repo, single_arts, out_dir),
        download_suites(args.repo, multi_arts, out_dir),
        args.config_root,
        args.extra_test_flags,
        args.extra_test_flags_multi,
        args.test_type,
        exclude=exclude,
    )
    for entry in suites:
        print(f"  retry: {entry['name']} [{entry['runner']}] {entry['config_paths']}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"matrix={json.dumps({'suite': suites})}\n")
            f.write(f"has_failed={'true' if suites else 'false'}\n")
    else:
        print(json.dumps({"suite": suites}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
