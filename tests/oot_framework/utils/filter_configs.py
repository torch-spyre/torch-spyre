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
  (empty) / "full"   All configs (backward compatible by default).
  "smoke"            Configs whose test_suite_config.labels contains "smoke".
  "core"             Configs whose test_suite_config.labels contains "core".
  "device_critical"  Configs whose test_suite_config.labels contains
                     "device_critical" -- the device-layer surfaces flex and
                     deeptools/dxp_standalone exercise most (streams, job
                     launch plans, codegen, LX/scratchpad planning, tensor
                     layout, allocator/GC, D2D copies). Used as the default
                     test_type for the tests.yml CI workflow.
  "suite_<group>"    Configs residing inside a directory named "<group>", or
                     whose filename starts with "<group>_".  This lets the
                     existing <group>/<name>_config.yaml layout act as a
                     coarse grouping without re-tagging every config.
  <other>            Treated as an arbitrary label name; matches configs
                     whose labels array contains the value.

Configs with no labels field default to ["full"] (backward compatible) --
they run under "full" but are excluded from every narrower test_type
(smoke, core, device_critical, suite_<group>, custom labels). Every config
should carry an explicit labels list; an unlabeled config is a gap to close
by adding one, not a signal to widen matching.

Output formats
--------------
  paths         Space-separated list of absolute paths (Makefile / bash use).
  matrix-json   JSON object {"suite": [{name, config, runner}, ...]} for
                GitHub Actions dynamic-matrix consumption.  "config" is the
                path relative to --config-dir.

Runner overrides
----------------
  CI runner requirements (which suites need spyre_pf_x2 or spyre_pf_x4) are
  kept separate from the test configs.  Pass --runner-map <yaml-file> to apply
  a mapping of {config-relative-path: runner-label}.  Suites not listed in the
  map use the default runner (spyre_pf_x1).  This argument is optional and only
  relevant for matrix-json output.

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
      --test-type core \\
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
    """Read test_suite_config.labels from a YAML config; default to ["full"]."""
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    tsc = raw.get("test_suite_config") or {}
    return list(tsc.get("labels", ["full"]))


def _load_runner_map(runner_map_path: str) -> dict:
    """Load {config-relative-path: runner-label} from a YAML file."""
    path = Path(runner_map_path)
    if not path.is_file():
        sys.exit(f"ERROR: --runner-map file not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return {k: str(v) for k, v in data.items()}


def _matches(labels: list, config_path: Path, test_type: str) -> bool:
    """Return True if *config_path* should be included for *test_type*."""
    if not test_type or test_type == "full":
        return True

    if test_type.startswith("suite_"):
        group = test_type[len("suite_") :].lower()
        path_parts = [p.lower() for p in config_path.parts]
        if group in path_parts:
            return True
        stem = config_path.stem.lower()
        return stem.startswith(group + "_") or stem == group

    return test_type in labels


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
        default="full",
        metavar="TYPE",
        help=(
            "Selection type: smoke | core | full | suite_<group> | <label>. "
            "Default: full (all configs)."
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
        "--format",
        choices=["paths", "matrix-json"],
        default="paths",
        help=(
            "Output format.  'paths': space-separated absolute paths. "
            "'matrix-json': GitHub Actions dynamic-matrix JSON."
        ),
    )
    args = ap.parse_args()

    config_dir = Path(args.config_dir).resolve()
    if not config_dir.is_dir():
        sys.exit(f"ERROR: --config-dir does not exist: {config_dir}")

    test_type = args.test_type.strip()

    runner_map: dict = {}
    if args.runner_map:
        runner_map = _load_runner_map(args.runner_map)

    results = []
    for cfg in sorted(config_dir.rglob("*.yaml")):
        try:
            labels = _load_labels(cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: skipping {cfg} ({exc})", file=sys.stderr)
            continue
        if _matches(labels, cfg, test_type):
            rel = str(cfg.relative_to(config_dir))
            results.append(
                {
                    "name": _display_name(cfg, config_dir),
                    "config": rel,
                    "runner": runner_map.get(rel, "spyre_pf_x1"),
                    "path": str(cfg),
                }
            )

    if not results:
        print(
            f"WARNING: no configs matched TEST_TYPE={test_type!r} under {config_dir}",
            file=sys.stderr,
        )

    if args.format == "paths":
        print(" ".join(r["path"] for r in results))
    else:
        matrix = [
            {"name": r["name"], "config": r["config"], "runner": r["runner"]}
            for r in results
        ]
        print(json.dumps({"suite": matrix}))


if __name__ == "__main__":
    main()
