"""
oot_test_utilities.py -- Utility functions for the OOT PyTorch test framework.
# Copyright Author: Anubhav Jana (Anubhav.Jana97@ibm.com)

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
        "PyYAML is required for oot_test_utilities. Install it with: pip install pyyaml"
    ) from _yaml_err

import torch


# ---------------------------------------------------------------------------
# Device type helpers
# ---------------------------------------------------------------------------


def _get_privateuse1_device_type() -> str:
    """Return the backend name registered for the privateuse1 device slot.

    torch._C._get_privateuse1_backend_name() returns e.g. "spyre" or whatever
    name was passed to torch._register_device_module().  This is what
    cls.device_type will be at test runtime inside PrivateUse1TestBase.

    Falls back to "privateuse1" if no backend has been registered yet (e.g.
    during import before the backend module is loaded).
    """
    try:
        return torch._C._get_privateuse1_backend_name()
    except Exception:
        return "privateuse1"  # fallback if not registered yet


"""
Utility for printing per-test tags at run time alongside PASS/FAIL output.
"""

# To store method_name -> full tag list set during test execution
_RUNTIME_TAGS: Dict[str, List[str]] = {}


def print_test_tags_oot(test_instance, op_tags: List[str] = []) -> None:
    """Print [TAGS = ...] for a test method at run time.

    Combines method-level tags (test-level + dynamic op__/dtype__/module__) stored
    at collection time with per-op tags available only at run time.

    Usage in a test method:
        from oot_test_utilities import print_test_tags_oot
        print_test_tags_oot(self, op_tags=op.op_tags)
    """
    method_name = test_instance._testMethodName
    _method_fn = getattr(test_instance.__class__, method_name, None)
    _method_tags = getattr(_method_fn, "_oot_method_tags", [])
    _per_op_tags = [t for t in op_tags if t not in set(_method_tags)]
    _all_tags = _method_tags + _per_op_tags
    # Store for pytest_runtest_makereport hook to work without -s
    _RUNTIME_TAGS[method_name] = _all_tags
    # Also write directly to stderr (visible with -s)
    os.write(2, f"[TAGS = {' '.join(_all_tags)}]\n".encode())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Provides YAML config merging for multi-config test runs.

# Usage (Python):
#     from oot_test_utilities import merge_yaml_configs

#     merged_path = merge_yaml_configs(["config_a.yaml", "config_b.yaml"])
#     # ... run tests ...
#     os.unlink(merged_path)   # caller is responsible for cleanup

# Usage (bash, via the CLI entry-point at the bottom):
#     python3 oot_test_utilities.py config_a.yaml config_b.yaml
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
    lists are combined and their ``unlisted_test_mode`` is kept from the
    first occurrence.

    Test block deduplication within a path:
    - A block is a TRUE duplicate and dropped only when its ``names``,
      ``tags``, AND ``edits`` all match an already-seen block exactly.
    - Blocks that share the same ``names`` but differ in ``tags`` or
      ``edits`` (e.g. the same test op run for different models/dtypes)
      are kept as SEPARATE entries so each produces its own tagged variant.
    - Entries with distinct paths are appended in the order they appear.
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
                    f"[oot_merge] WARNING: conflicting unlisted_test_mode for "
                    f"path '{path}': keeping '{existing['unlisted_test_mode']}', "
                    f"ignoring '{incoming_mode}'.",
                    file=sys.stderr,
                )

            for test_block in entry.get("tests") or []:
                block_names = frozenset(
                    n.strip() for n in (test_block.get("names") or [])
                )

                # A block is a TRUE duplicate only when names + tags + edits
                # all match an already-present block exactly.  Blocks with
                # the same names but different tags/edits represent distinct
                # configurations (e.g. same op for different models) and must
                # be kept as separate entries.
                is_true_duplicate = any(
                    frozenset(n.strip() for n in (t.get("names") or [])) == block_names
                    and t.get("tags") == test_block.get("tags")
                    and t.get("edits") == test_block.get("edits")
                    for t in existing["tests"]
                )

                if not is_true_duplicate:
                    existing["tests"].append(test_block)
                    # Note: we intentionally do NOT track block_names in a
                    # global "seen names" set here, because the same name is
                    # reused across configs with different tags.

    return list(merged.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_yaml_configs(
    config_paths: Sequence[str | os.PathLike],
    *,
    output_dir: Optional[str | os.PathLike] = None,
    prefix: str = "_oot_merged_config_",
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
        prog="oot_test_utilities",
        description=(
            "Merge multiple OOT PyTorch YAML test configs into one temporary file.\n"
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
        f"[oot_merge] Merged {len(args.configs)} config(s) -> {merged}",
        file=sys.stderr,
    )
    # The path goes to stdout for shell capture.
    print(merged)


if __name__ == "__main__":
    _cli()
