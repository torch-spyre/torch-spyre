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

# _test_matrix.yaml's legs_complete job: `name:` there must keep matching this, or the
# early wave polls to its deadline instead of stopping when the matrix finishes.
FANIN_JOB_NAME = "Test matrix fan-in ({call_id})"


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
        entries.append(entry)
    return entries


TRANSIENT_HTTP_CODES = ("429", "502", "503", "504")


def _gh(args: list, max_attempts: int = 5, base_delay: int = 2, text: bool = False):
    """`gh api`, retrying only a transient 5xx or a rate-limit 429.

    Same idiom as download_run_xml_artifacts.py. Retrying a permanent 404 would just burn
    the budget a real 502 needs, so only these codes are retried.
    """
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(["gh", "api", *args], capture_output=True, text=text)
        if result.returncode == 0:
            return result
        err = result.stderr if text else result.stderr.decode("utf-8", "replace")
        if not any(c in err for c in TRANSIENT_HTTP_CODES) or attempt == max_attempts:
            raise RuntimeError(f"gh api {args[0]} failed: {err.strip()}")
        delay = base_delay * (2 ** (attempt - 1))
        print(f"[retry] gh api attempt {attempt}/{max_attempts}, again in {delay}s")
        time.sleep(delay)
    raise AssertionError("unreachable")


def list_artifacts(repo: str, run_id: str):
    """Every artifact name in the run, as {name: id}.

    --paginate: the endpoint defaults to 30 per page and a regression run uploads one XML
    per suite plus the wheel, well over one page. An un-paginated query silently drops
    everything past page 1, so a suite whose descriptor landed later would never be
    retried and the gate would report green. --paginate applies --jq per page and
    concatenates (gh rejects --slurp with --jq), so parse a per-object JSON stream.
    """
    listing = _gh(
        [
            "--paginate",
            f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            "--jq",
            ".artifacts[] | {id: .id, name: .name}",
        ],
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
        result = _gh([f"/repos/{repo}/actions/artifacts/{names_to_ids[name]}/zip"])
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


def fanin_complete(repo: str, run_id: str, attempt: str, job_name: str) -> bool:
    """Has _test_matrix.yaml's fan-in job finished, i.e. is every test leg done?

    Read off the job's own completion rather than anything it uploads: an artifact is one
    more thing that can fail to appear, and a poller that misses the end of the matrix
    stalls the final wave behind it. The name carries generate_matrix's call_id, so a run
    holding two calls of this workflow (runtests_nightly.yaml) matches only its own; the
    attempts endpoint scopes it to this attempt. A rename here is caught by
    tests/test_collect_retry_suites.py, and bounded by --deadline-secs regardless.
    """
    jobs = _gh(
        [
            "--paginate",
            f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100",
            "--jq",
            ".jobs[] | {name: .name, status: .status}",
        ],
        text=True,
    )
    return any(
        job_name in obj["name"] and obj["status"] == "completed"
        for obj in (json.loads(ln) for ln in jobs.stdout.splitlines() if ln.strip())
    )


def wait_for_failure(
    repo, run_id, attempt, prefix, fanin_job, poll_secs, deadline_secs
):
    """Poll until the first descriptor appears, the matrix fans in, or the deadline passes.

    Returning at the first failure is the point of the early wave. Returning on the fan-in
    is what keeps a green run fast: without it this would poll to the deadline on every
    run that never fails, and the final wave waits on this job.

    Fan-in means every leg has finished, and a leg uploads its descriptor before it
    finishes, so no failure is still in flight -- at worst an upload is not yet visible
    through the eventually-consistent artifacts API, and the final wave, which lists again
    after the fan-in, picks it up.

    An API call that fails outright is logged and polled again rather than raised: the
    whole of this wave is a head start, so losing it to one 502 an hour into a run would
    be a poor trade. Only the deadline gives up, and the final wave still retries every
    failed suite.
    """
    deadline = time.monotonic() + deadline_secs
    artifacts: dict = {}
    while True:
        try:
            artifacts = list_artifacts(repo, run_id)
            single, multi = select(artifacts, prefix)
            if single or multi:
                print(
                    f"First failure(s) visible: {len(single)} single, {len(multi)} multi."
                )
                return artifacts
            if fanin_complete(repo, run_id, attempt, fanin_job):
                print(
                    "Test matrix fanned in with no failed-suite descriptor; nothing to do."
                )
                return artifacts
        except (RuntimeError, OSError, ValueError) as err:
            print(f"::warning::Could not read the run's state, retrying: {err}")
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

    --extra-test-flags and --extra-test-flags-multi carry values that start with a dash
    ("-v", "--cpu-compile -vvv -s"), which argparse reads as another option unless the
    caller uses the --opt=value form. Every caller must spell those two that way.
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
        help="Early wave: poll until the first failure, the fan-in, or the deadline",
    )
    parser.add_argument(
        "--fanin-job-name",
        default=FANIN_JOB_NAME,
        help=(
            "Display name of _test_matrix.yaml's fan-in job, whose completion means every "
            "test leg is done. '{call_id}' is substituted"
        ),
    )
    parser.add_argument(
        "--poll-secs",
        type=int,
        default=60,
        help=(
            "Seconds between artifact listings. A minute: the retry it starts takes "
            "many minutes, and GITHUB_TOKEN's REST budget is per repository per hour."
        ),
    )
    parser.add_argument(
        "--deadline-secs",
        type=int,
        default=90 * 60,
        help=(
            "Cap on --wait-for-failure polling. Bounds how long the poller can idle if "
            "it never sees the fan-in; the final wave covers whatever is missed"
        ),
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_for_failure:
        artifacts = wait_for_failure(
            args.repo,
            args.run_id,
            args.run_attempt,
            prefix,
            args.fanin_job_name.format(call_id=args.call_id),
            args.poll_secs,
            args.deadline_secs,
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
