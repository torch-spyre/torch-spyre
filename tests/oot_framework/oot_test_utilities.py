"""
oot_test_utilities.py -- Utility functions for the OOT PyTorch test framework.
# Copyright Author: Anubhav Jana (Anubhav.Jana97@ibm.com)

"""

from __future__ import annotations

import ast
import platform as _platform
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import regex as re
import torch

from .oot_test_constants import (
    DTYPE_STR_MAP,
    REL_PATH_TOKENS,
)
from .oot_test_matching import parse_dtype


# ---------------------------------------------------------------------------
# Optional YAML import
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError as _yaml_err:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for oot_test_utilities. Install it with: pip install pyyaml"
    ) from _yaml_err


# Normalise platform.machine() into a safe marker string once at import time.
_OOT_PLATFORM_ARCH: str = (
    re.sub(r"[^a-zA-Z0-9_]", "_", _platform.machine() or "unknown").strip("_")
    or "unknown"
)


def _extract_base_module_name(name: str) -> str:
    """Extract base module name by stripping YAML-generated suffixes.

    Strips suffixes like:
    - _93b52f93 (8-char hex hash)
    - _4096 (numeric identifier)
    - _layer0 (layer identifier)

    Examples:
        GraniteRotaryEmbedding_93b52f93 -> GraniteRotaryEmbedding
        GraniteRMSNorm_4096 -> GraniteRMSNorm
        GraniteDecoderLayer_layer0 -> GraniteDecoderLayer

    Args:
        name: YAML module name with potential suffix

    Returns:
        Base module name without suffix
    """
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        base_name, suffix = parts
        # Check if suffix looks like a hash (hex) or number or "layerN"
        if suffix.replace("layer", "").isdigit() or all(
            c in "0123456789abcdef" for c in suffix
        ):
            return base_name
    return name


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


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------


def _log_warning(msg: str) -> None:
    """Write warning message to stderr for visibility during test runs."""
    os.write(2, f"[OOTDeviceTestBase WARNING] {msg}\n".encode())


def _log_error(msg: str) -> None:
    """Write error message to stderr for visibility during test runs."""
    os.write(2, f"[OOTDeviceTestBase ERROR] {msg}\n".encode())


# ---------------------------------------------------------------------------
# Python literal evaluator
# ---------------------------------------------------------------------------

_PY_ALLOWED_NODES = {
    ast.Expression,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Name,
    ast.Call,
    ast.Load,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
}
_PY_ALLOWED_NAMES: Dict[str, Any] = {
    "None": None,
    "Ellipsis": Ellipsis,
    "slice": slice,
    "inf": float("inf"),
    "nan": float("nan"),
}

_TOKEN_RE = re.compile(r"\$\{([^}]+)\}")


def _eval_py_literal(expr: str) -> Any:
    """Safely evaluate a restricted Python literal (slice, tuple, Ellipsis, etc.)."""
    node = ast.parse(expr, mode="eval")
    for n in ast.walk(node):
        if type(n) not in _PY_ALLOWED_NODES:
            raise ValueError(
                f"Node type {type(n).__name__!r} not allowed in py: {expr!r}"
            )
        if isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id == "slice"):
                raise ValueError(f"Only slice(...) calls are allowed in py: {expr!r}")
        if isinstance(n, ast.Name) and n.id not in _PY_ALLOWED_NAMES:
            raise ValueError(f"Name {n.id!r} not allowed in py: {expr!r}")
    return eval(compile(node, "<py>", "eval"), {"__builtins__": {}}, _PY_ALLOWED_NAMES)


# ---------------------------------------------------------------------------
# Dtype resolution
# ---------------------------------------------------------------------------


def _resolve_dtype_str(spec: str) -> torch.dtype:
    """Resolve a dtype string using DTYPE_STR_MAP. Accepts 'float16' or 'torch.float16'."""
    bare = spec.removeprefix("torch.")
    if bare in DTYPE_STR_MAP:
        return DTYPE_STR_MAP[bare]
    try:
        return parse_dtype(bare)
    except ValueError:
        pass
    raise ValueError(
        f"Unsupported dtype: {spec!r}. "
        f"Supported aliases: {sorted(DTYPE_STR_MAP)} and torch.<dtype>"
    )


# ---------------------------------------------------------------------------
# Tensor path resolution
# ---------------------------------------------------------------------------


def _resolve_tensor_path(raw_path: str) -> str:
    """Expand ``${TOKEN}`` placeholders in a tensor init_args.path and return
    an absolute path.

    Resolution order:
    1. Replace every ``${TOKEN}`` using the env-var declared in REL_PATH_TOKENS.
    2. If the result is already absolute, return it.
    3. Otherwise resolve relative to the process working directory.

    Raises:
        ValueError:       Unknown token or its env-var is unset.
        FileNotFoundError: Resolved path does not exist on disk.
    """
    token_map: dict[str, str] = {
        tok.strip("${}") if tok.startswith("${") else tok: env_var
        for tok, env_var in REL_PATH_TOKENS
    }

    def _replace(m: re.Match) -> str:
        name = m.group(1)
        if name not in token_map:
            raise ValueError(
                f"Unknown path token '${{{name}}}' in init_args.path={raw_path!r}. "
                f"Known tokens: {sorted(token_map)}"
            )
        value = os.environ.get(token_map[name])
        if value is None:
            raise ValueError(
                f"Environment variable '{token_map[name]}' (for token '${{{name}}}') "
                f"is not set. Export it before running tests."
            )
        return value

    expanded = _TOKEN_RE.sub(_replace, raw_path)
    resolved = str(Path(expanded).resolve())

    if not Path(resolved).exists():
        raise FileNotFoundError(
            f"Tensor file not found: {resolved!r}  (from init_args.path={raw_path!r})"
        )
    return resolved


# ---------------------------------------------------------------------------
# Regex pattern helpers
# ---------------------------------------------------------------------------


def _is_regex_pattern(name: str) -> bool:
    """Return True if *name* contains any regex metacharacter."""
    return any(c in name for c in r"\.^$*+?{}[]|()")


def _regex_entries_for_name(
    name: str, regex_entries: List[Tuple[str, Any]]
) -> List[Any]:
    """Return all TestEntry objects whose stored regex pattern matches *name*.

    *name* is the base method name as seen by ``instantiate_test`` (e.g.
    ``"test_rope_fms_prefill_bs1"``).  Matching uses ``re.fullmatch`` so
    the pattern must match the entire method name, not just a substring.

    Examples of patterns that will match ``test_rope_fms_prefill_bs1``:
      - ``test_rope_fms_.*``
      - ``test_rope_fms_prefill_.*``
      - ``test_rope_.*``
      - ``test_(rope|qkv)_.*``
    """
    return [entry for pattern, entry in regex_entries if re.fullmatch(pattern, name)]


# ---------------------------------------------------------------------------
# Multi-entry test map helpers
# ---------------------------------------------------------------------------


def _build_test_entry_map(
    file_entry: Any,
) -> Tuple[Dict[str, List[Any]], List[Tuple[str, Any]]]:
    """Build exact-name map and regex list from file_entry.tests.

    A single TestEntry can cover multiple test ids via name: [list].
    Each method_name in the list gets its own entry in the map.

    This supports multiple TestEntry objects per method_name.
    This is needed when two configs target the same test name with different
    tags/dtypes (e.g. the same op tested for two different models).
    The correct entry for a given variant is resolved later by
    ``_select_entry_for_variant`` once the dtype is known from the
    instantiated method name.

    Each method_name in the list is routed to either the exact map or the
    regex list depending on whether it contains regex metacharacters.

    Returns
    -------
    exact_map : Dict[str, List[TestEntry]]
        Keyed by exact base method names (no regex metacharacters).
    regex_entries : List[Tuple[str, TestEntry]]
        Ordered list of ``(regex_pattern, entry)`` pairs for names that
        contain regex metacharacters.  Patterns are bare method names —
        the ``ClassName::`` prefix has already been stripped by
        ``entry.method_names()``.

    Usage
    -----
    ::

        cls.TEST_ENTRIES, cls.REGEX_ENTRIES = _build_test_entry_map(file_entry)
    """
    exact_map: Dict[str, List[Any]] = {}
    regex_entries: List[Tuple[str, Any]] = []

    for entry in file_entry.tests:
        for method_name in entry.method_names():
            if _is_regex_pattern(method_name):
                regex_entries.append((method_name, entry))
            else:
                exact_map.setdefault(method_name, []).append(entry)

    return exact_map, regex_entries


def _entry_dtype_set(
    entry: Any,
    global_supported_dtypes: Optional[Set[torch.dtype]],
) -> Optional[Set[torch.dtype]]:
    """Return the effective dtype set for *entry*.

    Priority (highest to lowest):
      1. entry.edits.dtypes.include  -- explicit per-entry dtype list
      2. global_supported_dtypes     -- global filter from the YAML global section
      3. None                        -- no filtering; all dtypes match

    Returns None when neither the entry nor the global config restricts dtypes,
    meaning the entry is considered compatible with any dtype.
    """
    included = entry.edits.dtypes.resolved_include()
    if included:
        return included
    return global_supported_dtypes  # may itself be None


# Matches "test_model_ops_db_<unique>__<idx>_<device>_<dtype>", capturing
# the op unique_name key into model_ops_entry_by_unique_name.
_MODEL_OPS_VARIANT_RE = re.compile(
    r"^test_model_ops_db_(?P<unique>.+?__\d+)_[A-Za-z0-9]+_\w+$"
)


def _select_entry_by_op_index(method_name: str) -> Optional[Any]:
    """Resolve the TestEntry for a test_model_ops_db variant via the
    authoritative unique_name mapping; returns None to let callers fall
    back to the dtype heuristic."""
    m = _MODEL_OPS_VARIANT_RE.match(method_name)
    if not m:
        return None
    try:
        from models.test_model_ops_v2 import (  # type: ignore
            model_ops_entry_by_unique_name,
        )
    except ImportError:
        return None
    return model_ops_entry_by_unique_name.get(m.group("unique"))


def _select_entry_for_variant(
    entries: List[Any],
    method_name: str,
    global_supported_dtypes: Optional[Set[torch.dtype]],
) -> Any:
    """Pick the best-matching TestEntry for a concrete variant method name.

    When only one entry exists the choice is trivial.  When multiple entries
    share the same base test name merged from different configs we select
    by matching the dtype embedded in *method_name* against each entry's
    effective dtype set.

    Selection rules:
      1. Entry whose effective dtype set contains the variant's dtype.
      2. Entry with no dtype restriction (effective set is None) acts as
         a wildcard / fallback.
      3. First entry in the list (last-resort fallback to old behaviour).

    The list order reflects YAML insertion order so config-A entries take
    precedence over config-B entries for identical dtype sets.
    """
    # Deferred imports to avoid circular dependency at module level.
    from .oot_test_matching import extract_dtype_from_name, parse_dtype

    if len(entries) == 1:
        return entries[0]

    dtype_str = extract_dtype_from_name(method_name)
    variant_dtype: Optional[torch.dtype] = None
    if dtype_str:
        try:
            variant_dtype = parse_dtype(dtype_str)
        except ValueError:
            pass

    # Pass 1 - strict dtype match
    if variant_dtype is not None:
        for entry in entries:
            eset = _entry_dtype_set(entry, global_supported_dtypes)
            if eset is not None and variant_dtype in eset:
                return entry

    # Pass 2 - wildcard entry (no dtype restriction)
    for entry in entries:
        eset = _entry_dtype_set(entry, global_supported_dtypes)
        if eset is None:
            return entry

    # Pass 3 - fallback: return first entry
    return entries[0]


def _extract_op_name_from_method(
    method_name: str, base_test_name: str, oot_device_type: str
) -> Optional[str]:
    """Extract the op name from a parametrized method name.

    method_name: test_scalar_support_add_<device>_float16
    base_test_name: test_scalar_support
    oot_device_type: the registered privateuse1 backend name (e.g. "spyre")
    returns: "add"

    Returns None if the op name cannot be determined.
    """
    if not method_name.startswith(base_test_name + "_"):
        return None
    remainder = method_name[len(base_test_name) + 1 :]  # "add_<device>_float16"
    # op name is the first segment before the device suffix
    if f"_{oot_device_type}_" in remainder:
        return remainder.split(f"_{oot_device_type}_")[0]
    return None


# ---------------------------------------------------------------------------
# Per-test tag printing
# ---------------------------------------------------------------------------

"""
Utility for printing per-test tags at run time alongside PASS/FAIL output.
"""

# To store method_name -> full tag list set during test execution
_RUNTIME_TAGS: Dict[str, List[str]] = {}
# To store method_name -> formatted shape/stride string set during test execution
_RUNTIME_SHAPES: Dict[str, str] = {}


def _format_input_args_shapes(input_args: List[Any]) -> str:
    """Return a human-readable string of shape/stride info for each InputArg.

    Handles InputArgTensor, InputArgTensorList, InputArgConfig, InputArgValue,
    and InputArgPy. Falls back to a type label for unknown arg types.

    Args:
        input_args: List of InputArg objects from InputsEdits.args.

    Returns:
        A formatted multi-line string, or an empty string if input_args is empty.
    """
    if not input_args:
        return ""

    try:
        from .oot_test_config_models import (
            InputArgTensor,
            InputArgTensorList,
            InputArgConfig,
            InputArgValue,
            InputArgPy,
        )
    except ImportError:
        return ""

    lines: List[str] = []
    for i, arg in enumerate(input_args):
        if isinstance(arg, InputArgTensor):
            spec = arg.tensor
            stride_str = f", stride={spec.stride}" if spec.stride is not None else ""
            lines.append(
                f"  arg[{i}]: Tensor(shape={spec.shape}, dtype={spec.dtype}{stride_str})"
            )
        elif isinstance(arg, InputArgTensorList):
            inner = ", ".join(
                (
                    f"Tensor(shape={s.shape}, dtype={s.dtype}"
                    + (f", stride={s.stride}" if s.stride is not None else "")
                    + ")"
                )
                for s in arg.tensor_list
            )
            lines.append(f"  arg[{i}]: TensorList[{inner}]")
        elif isinstance(arg, InputArgConfig):
            lines.append(
                f"  arg[{i}]: Config(config_path={arg.config_path!r}, "
                f"config_kwargs={arg.config_kwargs!r})"
            )
        elif isinstance(arg, InputArgValue):
            lines.append(f"  arg[{i}]: value={arg.value!r}")
        elif isinstance(arg, InputArgPy):
            lines.append(f"  arg[{i}]: py={arg.py!r}")
        else:
            lines.append(f"  arg[{i}]: {type(arg).__name__}")

    return "\n".join(lines)


def print_test_tags_oot(
    test_instance: Any,
    op_tags: List[str] = [],
    input_args: Optional[List[Any]] = None,
) -> None:
    """Print [TAGS = ...] and optional per-arg shape/stride info at run time.

    Combines method-level tags (test-level + dynamic op__/dtype__/module__) stored
    at collection time with per-op tags available only at run time.  When
    *input_args* is supplied (``op.ops_item.sample_inputs_func.args``), each
    argument's shape and stride are stored for the pytest reporting hook and
    also written to stderr (visible with -s / --capture=no).

    Usage in a test method:
        from oot_framework.oot_test_utilities import print_test_tags_oot
        print_test_tags_oot(self, op_tags=op.op_tags,
                            input_args=op.ops_item.sample_inputs_func.args)
    """
    method_name = test_instance._testMethodName
    _method_fn = getattr(test_instance.__class__, method_name, None)
    _method_tags = getattr(_method_fn, "_oot_method_tags", [])
    _per_op_tags = [t for t in op_tags if t not in set(_method_tags)]
    _all_tags = _method_tags + _per_op_tags
    # Store for pytest_runtest_makereport / pytest_runtest_logreport hooks
    _RUNTIME_TAGS[method_name] = _all_tags
    if input_args is not None:
        shapes_str = _format_input_args_shapes(input_args)
        if shapes_str:
            _RUNTIME_SHAPES[method_name] = shapes_str
    # Also write directly to stderr (visible with -s / --capture=no)
    os.write(2, f"[TAGS = {' '.join(_all_tags)}]\n".encode())
    if method_name in _RUNTIME_SHAPES:
        os.write(2, f"[INPUT SHAPES]\n{_RUNTIME_SHAPES[method_name]}\n".encode())


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
