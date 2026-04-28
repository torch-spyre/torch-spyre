"""
spyre_test_utilities.py -- Utility functions for the torch-spyre OOT test framework.

"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Optional YAML import
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError as _yaml_err:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for spyre_test_utilities. "
        "Install it with: pip install pyyaml"
    ) from _yaml_err


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Provides YAML config merging for multi-config test runs.

# Usage (Python):
#     from spyre_test_utilities import merge_yaml_configs

#     merged_path = merge_yaml_configs(["config_a.yaml", "config_b.yaml"])
#     # ... run tests ...
#     os.unlink(merged_path)   # caller is responsible for cleanup

# Usage (bash, via the CLI entry-point at the bottom):
#     python3 spyre_test_utilities.py config_a.yaml config_b.yaml
#     # prints the path of the merged (temp) YAML to stdout


def _deep_merge_globals(
    base: Dict[str, Any], incoming: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two `global:` dicts according to the superset rules.

    Rules (per key):
    - Key absent in base                 -> copy incoming value as-is.
    - Key present in both, values equal  -> keep as-is (no duplication).
    - Key present in both, values differ:
        * If both values are lists       -> append unique incoming items

        * Otherwise (scalar)             -> raise ValueError; callers should
                                           not have conflicting scalar globals.


    Each list element is typically a dict such as ``{name: float16}`` or
    ``{name: add, dtypes: [...], force_xfail: true}``.  Two elements are
    considered *the same* when they serialise to identical YAML (i.e. all
    their fields match), so a re-listed identical op is deduplicated while
    an op with the same name but different sub-fields is appended as a new
    entry (superset semantics).
    """
    result: Dict[str, Any] = dict(base)

    for key, incoming_val in incoming.items():
        if key not in result:
            result[key] = incoming_val
            continue

        base_val = result[key]

        if base_val == incoming_val:
            # Identical — nothing to do.
            continue

        # Values differ — only lists are mergeable under superset rules.
        if isinstance(base_val, list) and isinstance(incoming_val, list):
            # Use serialised YAML as the deduplication key so dict elements
            # are compared by value, not by Python object identity.
            existing_keys = {
                yaml.dump(item, default_flow_style=True) for item in base_val
            }
            merged_list = list(base_val)
            for item in incoming_val:
                serialised = yaml.dump(item, default_flow_style=True)
                if serialised not in existing_keys:
                    merged_list.append(item)
                    existing_keys.add(serialised)
            result[key] = merged_list
        else:
            # Scalar conflict - not automatically resolvable.
            raise ValueError(
                f"Conflicting scalar values for global key '{key}': "
                f"{base_val!r} vs {incoming_val!r}. "
                "Resolve the conflict manually before merging."
            )

    return result


def _merge_file_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge per-file entries from all configs into a deduplicated list.

    Two entries with the same ``path`` value are merged: their ``tests``
    lists are concatenated (with deduplication by name) and their
    ``unlisted_test_mode`` is kept from the first occurrence (a warning is
    emitted if configs disagree).

    Entries with distinct paths are appended in the order they appear.
    """
    # Preserve insertion order; key = resolved path string.
    merged: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        path = entry.get("path", "")
        if path not in merged:
            # First time we see this path — deep-copy the entry.
            merged[path] = {
                "path": path,
                "unlisted_test_mode": entry.get("unlisted_test_mode", "xfail"),
                "tests": list(entry.get("tests") or []),
            }
        else:
            existing = merged[path]

            # Warn on unlisted_test_mode conflict; keep first value.
            incoming_mode = entry.get("unlisted_test_mode", "xfail")
            if existing["unlisted_test_mode"] != incoming_mode:
                print(
                    f"[spyre_merge] WARNING: conflicting unlisted_test_mode for "
                    f"path '{path}': keeping '{existing['unlisted_test_mode']}', "
                    f"ignoring '{incoming_mode}'.",
                    file=sys.stderr,
                )

            # Merge tests: deduplicate by the set of names in each test block.
            existing_test_names: set = set()
            for t in existing["tests"]:
                for n in t.get("names") or []:
                    existing_test_names.add(n.strip())

            for test_block in entry.get("tests") or []:
                block_names = {n.strip() for n in (test_block.get("names") or [])}
                # Append the whole block if ANY of its names is new.
                # (Partial overlaps are very unlikely in practice but if they
                # occur the block is still added so no test is silently lost.)
                if not block_names.issubset(existing_test_names):
                    existing["tests"].append(test_block)
                    existing_test_names.update(block_names)

    return list(merged.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_yaml_configs(
    config_paths: Sequence[str | os.PathLike],
    *,
    output_dir: Optional[str | os.PathLike] = None,
    prefix: str = "_spyre_merged_config_",
    suffix: str = ".yaml",
) -> str:
    """Merge multiple YAML test-suite configs into one temporary file.

    Parameters
    ----------
    config_paths:
        Ordered sequence of paths to YAML config files.  At least one path
        must be provided.  A single path is accepted for convenience (the
        file is copied to a temp path so the caller's cleanup flow is
        uniform).
    output_dir:
        Directory in which to create the temp file.  Defaults to the
        directory of the first config file so that ``${TORCH_ROOT}`` /
        ``${TORCH_DEVICE_ROOT}`` relative paths resolve correctly when the
        YAML is loaded from the same location.
    prefix / suffix:
        Passed to ``tempfile.mkstemp``.

    Returns
    -------
    str
        Absolute path to the merged temporary YAML file.

    Raises
    ------
    ValueError
        If no paths are provided, any path does not exist, or irreconcilable
        scalar conflicts are found in the ``global:`` section.
    """
    paths = [Path(p) for p in config_paths]

    if not paths:
        raise ValueError("At least one config path must be provided.")

    for p in paths:
        if not p.is_file():
            raise ValueError(f"Config file not found: {p}")

    # single config - just create a temp copy so the caller always
    # gets a temp file it can safely delete.
    if len(paths) == 1:
        raw = paths[0].read_text(encoding="utf-8")
        dest_dir = output_dir or paths[0].parent
        fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(dest_dir))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(raw)
        return tmp_path

    # ------------------------------------------------------------------
    # Multi-config merge
    # ------------------------------------------------------------------
    loaded: List[Dict[str, Any]] = []
    for p in paths:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Top-level YAML value must be a mapping in: {p}")
        loaded.append(data)

    # All configs must have a `test_suite_config` top-level key.
    suites = []
    for i, (data, p) in enumerate(zip(loaded, paths)):
        suite = data.get("test_suite_config")
        if not isinstance(suite, dict):
            raise ValueError(f"Missing or invalid 'test_suite_config' key in: {p}")
        suites.append(suite)

    # ---- Merge `files` entries ----------------------------------------
    all_file_entries: List[Dict[str, Any]] = []
    for suite in suites:
        all_file_entries.extend(suite.get("files") or [])

    merged_files = _merge_file_entries(all_file_entries)

    # ---- Merge `global` sections --------------------------------------
    merged_global: Dict[str, Any] = {}
    for suite in suites:
        g = suite.get("global")
        if isinstance(g, dict):
            merged_global = _deep_merge_globals(merged_global, g)

    # ---- Assemble final document --------------------------------------
    merged_doc: Dict[str, Any] = {
        "test_suite_config": {
            "files": merged_files,
        }
    }
    if merged_global:
        merged_doc["test_suite_config"]["global"] = merged_global

    # ---- Write to temp file ------------------------------------------
    dest_dir = output_dir or paths[0].parent
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(dest_dir))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        yaml.dump(
            merged_doc,
            fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    return tmp_path


# ---------------------------------------------------------------------------
# CLI entry-point (used by run_test.sh)
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Print the merged temp-file path to stdout; all other output goes to stderr."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="spyre_test_utilities",
        description=(
            "Merge multiple torch-spyre YAML test configs into one temporary file.\n"
            "Prints the merged file path to stdout. The caller must delete it."
        ),
    )
    parser.add_argument(
        "configs",
        nargs="+",
        metavar="CONFIG",
        help="Two or more YAML config file paths to merge.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory for the merged temp file "
            "(default: directory of the first config)."
        ),
    )
    args = parser.parse_args()

    if len(args.configs) < 2:
        # Single config — still honour the call so run_test.sh needn't
        # special-case it; we just echo back a temp copy.
        pass

    merged = merge_yaml_configs(args.configs, output_dir=args.output_dir)
    # Emit a human-readable note to stderr so it doesn't pollute the path.
    print(
        f"[spyre_merge] Merged {len(args.configs)} config(s) -> {merged}",
        file=sys.stderr,
    )
    # The path goes to stdout for shell capture.
    print(merged)


if __name__ == "__main__":
    _cli()
