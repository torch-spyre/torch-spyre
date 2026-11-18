#!/usr/bin/env python3
# Copyright 2026 Anubhav Jana (Anubhav.Jana97@ibm.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Filter OOT test suite configs by TEST_TYPE label or suite-prefix group.

Selection rules
---------------
  "smoke"            Configs whose test_suite_config.labels contains "smoke".
  "unit"             Configs whose test_suite_config.labels contains "unit".
  "integration"      Configs whose test_suite_config.labels contains
                     "integration" -- the device-layer surfaces flex and
                     deeptools exercise most (streams, job
                     launch plans, codegen, LX/scratchpad planning, tensor
                     layout, allocator/GC, D2D copies). Used as the default
                     test_type for integration-tests.yaml.
  "regression"       Configs whose test_suite_config.labels contains
                     "regression" -- the full functional-coverage tier.
  "trunk"            Configs whose test_suite_config.labels contains
                     "trunk" -- everything torch-spyre's push-to-main
                     workflows cover.
  "suite_<group>"    Configs residing inside a directory named "<group>", or
                     whose filename starts with "<group>_".  This lets the
                     existing <group>/<name>_config.yaml layout act as a
                     coarse grouping without re-tagging every config.
  <other>            Treated as an arbitrary label name; matches configs
                     whose labels array contains the value.

These tier names (smoke/unit/integration/regression/trunk) ARE the label
vocabulary tests/configs/**/*.yaml declares -- there is no separate alias
layer translating human-facing names to internal label names.

Only explicitly declared labels are respected: a config's
test_suite_config.labels array is matched literally against --test-type, with
no catch-all value that matches regardless of labels. Configs with no labels
field match nothing -- they are excluded from every test_type, including
regression and trunk. Every config must carry an explicit labels list; an
unlabeled config is a gap to close by adding one (see
tests/scripts/check_oot_configs.py, which fails CI on missing labels), not a
signal to widen matching.

Output formats
--------------
  paths         Space-separated list of absolute paths (Makefile / bash use).
  matrix-json   JSON object {"suite": [{name, config, runner}, ...]} for
                GitHub Actions dynamic-matrix consumption.  "config" is the
                path relative to --config-dir.  A suite named in --flags-map
                also carries "extra_test_flags".

Runner overrides
----------------
  CI runner requirements (which suites need spyre_pf_x2 or spyre_pf_x4) are
  kept separate from the test configs.  Pass --runner-map <yaml-file> to apply
  a mapping of {config-relative-path: runner-label}.  Suites not listed in the
  map use the default runner (spyre_pf_x1).  This argument is optional and only
  relevant for matrix-json output.

  --flags-map <yaml-file> likewise maps a config path, or a directory prefix
  ending in "/", to the pytest flags that suite runs with instead of the
  caller's extra_test_flags.

Usage
-----
  # Makefile / local dev
  python3 tests/oot_framework/utils/filter_configs.py \\
      --config-dir tests/configs/torch_spyre_tests \\
      --test-type smoke \\
      --format paths

  # GitHub Actions generate_matrix job
  python3 tests/oot_framework/utils/filter_configs.py \\
      --config-dir tests/configs/torch_spyre_tests \\
      --test-type unit \\
      --runner-map .github/runner_overrides.yaml \\
      --format matrix-json
"""

import argparse
import json
import sys
from pathlib import Path


try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required by filter_configs.py.  Install with: pip install pyyaml"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _display_name(config_path: Path, config_dir: Path) -> str:
    """Derive a human-readable display name from the config file path.

    Examples (config_dir = tests/configs/torch_spyre_tests):
      test_spyre_config.yaml           --> "Test Spyre"
      inductor/test_building_blocks_config.yaml -> "Inductor / Test Building Blocks"
    """
    rel = config_path.relative_to(config_dir)
    stem = rel.stem
    if stem.endswith("_config"):
        stem = stem[:-7]

    def _title(s: str) -> str:
        return " ".join(w.capitalize() for w in s.replace("_", " ").split())

    parts = list(rel.parts[:-1]) + [stem]
    if len(parts) >= 2:
        return _title(parts[-2]) + " / " + _title(parts[-1])
    return _title(parts[-1])


def _load_labels(path: Path) -> list:
    """Read test_suite_config.labels from a YAML config; no labels means no match."""
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    tsc = raw.get("test_suite_config") or {}
    return list(tsc.get("labels") or [])


def _load_path_map(map_path: str) -> dict:
    """Load a {config-relative-path: value} YAML file (--runner-map, --flags-map)."""
    path = Path(map_path)
    if not path.is_file():
        sys.exit(f"ERROR: map file not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return {k: str(v) for k, v in data.items()}


def _suite_flags(flags_map: dict, rel: str) -> str:
    """The flags --flags-map gives this config: an exact path, else the longest prefix."""
    if rel in flags_map:
        return flags_map[rel]
    prefixes = [k for k in flags_map if k.endswith("/") and rel.startswith(k)]
    return flags_map[max(prefixes, key=len)] if prefixes else ""


# The grouping policy and timings key configs relative to tests/configs/torch_spyre_tests/, but trunk
# scans tests/configs, where the same file is torch_spyre_tests/<key>. Normalise to one root so a
# bucket is the same on every tier.
LAYOUT_ROOT = "torch_spyre_tests/"


def _layout_key(rel: str) -> str:
    return rel[len(LAYOUT_ROOT) :] if rel.startswith(LAYOUT_ROOT) else rel


def _config_facts(path: Path) -> tuple:
    """(global block, test file paths) of a config, for the bucket safety checks."""
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    tsc = raw.get("test_suite_config") or {}
    return tsc.get("global"), [str(f.get("path", "")) for f in tsc.get("files") or []]


def _bucket_name(members: list) -> str:
    """Job name for a bucket that says what is in it: "Inductor / Cache, Codegen +14 more (16)".

    Members' own display names, with the shared directory and "Test" prefix said once.
    """
    head, _, _ = members[0]["name"].rpartition(" / ")
    short = []
    for m in members:
        n = m["name"].rpartition(" / ")[2]
        short.append(n[len("Test ") :] if n.startswith("Test ") else n)
    shown = ", ".join(short[:3])
    if len(short) > 3:
        shown += f" +{len(short) - 3} more"
    return f"{head + ' / ' if head else ''}{shown} ({len(members)})"


def _fold_buckets(results: list) -> list:
    """Fold configs that share a bucket into one matrix entry, after tier filtering.

    Re-checks what test_layout.plan promised, so a hand-edited timings or policy file can never
    produce a bucket that behaves differently from its members run alone: members must share
    runner, flags and `global:` block (run_test.sh unions global lists across merged configs), and
    must not name the same test file (the merge keeps only the first config's mode for that file).
    """
    out, buckets = [], {}
    for r in results:
        b = r.get("bucket")
        if not b:
            out.append(r)
            continue
        if b not in buckets:
            buckets[b] = dict(r, members=[r])
            out.append(buckets[b])
            continue
        g = buckets[b]
        for field in ("runner", "flags", "global"):
            if r[field] != g[field]:
                sys.exit(
                    f"ERROR: bucket {b!r}: {r['config']} and {g['config']} differ in {field}"
                )
        shared = set(r["files"]) & {f for m in g["members"] for f in m["files"]}
        if shared:
            sys.exit(
                f"ERROR: bucket {b!r}: {r['config']} shares test file(s) {sorted(shared)} with another member"
            )
        g["members"].append(r)
    for g in out:
        members = g.pop("members", None)
        if members and len(members) > 1:
            g["configs"] = " ".join(m["config"] for m in members)
            g["report"] = f"bucket_{g['bucket']}"
            g["name"] = _bucket_name(members)
    return out


# The tier ladder. A config's labels also carry suite groups and one-off markers;
# those are not tiers and must never make a config look already-covered.
TIER_LABELS = ("smoke", "unit", "integration", "regression", "trunk")


def _matches(labels: list, config_path: Path, test_type: str) -> bool:
    """Return True if *config_path* should be included for *test_type*.

    No catch-all: every test_type (including regression/trunk) must appear
    literally in the config's labels array. An empty test_type matches
    everything (used only when a caller deliberately omits filtering).
    """
    if not test_type:
        return True

    if test_type.startswith("suite_"):
        group = test_type[len("suite_") :].lower()
        path_parts = [p.lower() for p in config_path.parts]
        if group in path_parts:
            return True
        stem = config_path.stem.lower()
        return stem.startswith(group + "_") or stem == group

    return test_type in labels


def _already_covered(labels: list, exclude_tiers: list) -> bool:
    """True when this config was already executed by one of the covered tiers.

    The test is whether any ALREADY-COVERED tier selects this config -- not whether
    every tier it declares is covered. A config labeled
    [unit, regression, integration, trunk] was already run by the integration run,
    so a later regression run re-executes identical work; its `trunk` label is
    irrelevant because no trunk run happened.

    Set difference over DECLARED labels, never a ladder. Measured on the 213 live
    configs: all 59 integration configs also declare regression, so integration is
    an exact subset here and the regression delta is 73 configs. That is a property
    of these files, NOT a rule -- prod showed 616 case-level
    integration-not-in-trunk violations, so inferring the ladder instead of reading
    the labels would silently drop real tests.
    """
    if not exclude_tiers:
        return False
    return any(t in labels for t in exclude_tiers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter OOT test configs by TEST_TYPE label or suite prefix."
    )
    ap.add_argument(
        "--config-dir",
        required=True,
        help="Root directory to search for *.yaml config files.",
    )
    ap.add_argument(
        "--test-type",
        default="regression",
        metavar="TYPE",
        help=(
            "Selection type: smoke | unit | integration | regression | "
            "trunk | suite_<group> | <label>. Matches configs whose "
            "test_suite_config.labels array contains this value literally "
            "-- no catch-all. Default: regression."
        ),
    )
    ap.add_argument(
        "--runner-map",
        default=None,
        metavar="FILE",
        help=(
            "Optional YAML file mapping config-relative paths to CI runner labels. "
            "Only used with --format matrix-json. "
            "Suites not in the map use the default runner (spyre_pf_x1)."
        ),
    )
    ap.add_argument(
        "--flags-map",
        default=None,
        metavar="FILE",
        help=(
            "Optional YAML file mapping config-relative paths (or 'dir/' prefixes) "
            "to the pytest flags that suite runs with. Only used with --format "
            "matrix-json."
        ),
    )
    ap.add_argument(
        "--exclude-tiers",
        default="",
        help=(
            "Comma-separated tiers whose results already exist for this artifact. "
            "A config is dropped when ANY tier it declares is in this list: that tier "
            "already covered the config, so re-running it adds nothing. Empty (the "
            "default) runs the full tier -- which is what an unreachable ClickHouse "
            "degrades to."
        ),
    )
    ap.add_argument(
        "--format",
        choices=["paths", "matrix-json"],
        default="paths",
        help=(
            "Output format.  'paths': space-separated absolute paths. "
            "'matrix-json': GitHub Actions dynamic-matrix JSON."
        ),
    )
    ap.add_argument(
        "--layout-policy",
        metavar="JSON_FILE",
        default=None,
        help="Optional .github/test_layout_policy.json: group short configs into shared jobs.",
    )
    ap.add_argument(
        "--timings",
        metavar="JSON_FILE",
        default=None,
        help="Optional .github/test_timings.json: per-config runtimes the grouping packs by.",
    )
    args = ap.parse_args()

    config_dir = Path(args.config_dir).resolve()
    if not config_dir.is_dir():
        sys.exit(f"ERROR: --config-dir does not exist: {config_dir}")

    test_type = args.test_type.strip()
    exclude_tiers = [
        t.strip() for t in (args.exclude_tiers or "").split(",") if t.strip()
    ]
    # Never let a tier suppress itself: an explicit rerun of a tier must re-execute.
    exclude_tiers = [t for t in exclude_tiers if t != test_type]

    runner_map: dict = {}
    if args.runner_map:
        runner_map = _load_path_map(args.runner_map)
    flags_map: dict = _load_path_map(args.flags_map) if args.flags_map else {}

    # Test files more than one config on disk lists (e.g. test_inductor_ops.py, split across many
    # configs by test class). A bucket's JUnit report is named after the bucket, so the timings
    # refresh credits its cases to a config by test file; a shared file makes that ambiguous, and
    # those configs therefore always run alone (test_layout.plan) to keep their timings fresh.
    owners: dict = {}
    for cfg in config_dir.rglob("*.yaml"):
        try:
            for f in _config_facts(cfg)[1]:
                owners.setdefault(Path(f).stem, set()).add(cfg)
        except Exception:  # noqa: BLE001  (unreadable configs are reported below)
            continue

    results = []
    skipped_covered = 0
    for cfg in sorted(config_dir.rglob("*.yaml")):
        try:
            labels = _load_labels(cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: skipping {cfg} ({exc})", file=sys.stderr)
            continue
        if not _matches(labels, cfg, test_type):
            continue
        if _already_covered(labels, exclude_tiers):
            skipped_covered += 1
            continue
        rel = str(cfg.relative_to(config_dir))
        glob, files = _config_facts(cfg)
        results.append(
            {
                "name": _display_name(cfg, config_dir),
                "config": rel,
                "runner": runner_map.get(rel, "spyre_pf_x1"),
                "flags": _suite_flags(flags_map, rel),
                "path": str(cfg),
                "key": _layout_key(rel),
                "dir": str(Path(_layout_key(rel)).parent),
                "global": glob,
                "files": files,
                "shared_file": any(
                    len(owners.get(Path(f).stem, ())) > 1 for f in files
                ),
            }
        )

    if skipped_covered:
        print(
            f"delta: skipped {skipped_covered} config(s) whose tiers are all already "
            f"covered ({','.join(exclude_tiers)})",
            file=sys.stderr,
        )

    if not results:
        print(
            f"WARNING: no configs matched TEST_TYPE={test_type!r} under {config_dir}",
            file=sys.stderr,
        )

    if args.format == "paths":
        print(" ".join(r["path"] for r in results))
    else:
        # Imported only when grouping is asked for: callers that sparse-check-out filter_configs.py
        # alone (upstream_tests.yaml, integration-tests.yaml) never pass --layout-policy.
        if args.layout_policy:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import test_layout
        else:
            test_layout = None
        try:
            policy = (
                test_layout.load_policy(args.layout_policy) if test_layout else None
            )
            timings = test_layout.load_timings(args.timings) if test_layout else {}
        except ValueError as exc:  # test_layout.LayoutError
            sys.exit(f"ERROR: {exc}")
        if args.layout_policy and policy is None:
            print(
                f"WARNING: {args.layout_policy} not found; every config runs as its own job",
                file=sys.stderr,
            )
        buckets = test_layout.plan(results, policy, timings) if test_layout else {}
        for r in results:
            r["bucket"] = buckets.get(r["key"], "")
        results = _fold_buckets(results)
        matrix = [
            {
                "name": r["name"],
                "config": r["config"],
                "runner": r["runner"],
                **({"extra_test_flags": r["flags"]} if r["flags"] else {}),
                **(
                    {"configs": r["configs"], "report": r["report"]}
                    if r.get("configs")
                    else {}
                ),
            }
            for r in results
        ]
        print(json.dumps({"suite": matrix}))


if __name__ == "__main__":
    main()
