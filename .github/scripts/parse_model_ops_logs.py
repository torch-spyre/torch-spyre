#!/usr/bin/env python3
"""
Parse raw GHA job logs produced by the model-ops-tests workflow and produce:

  1. model_ops_log.txt  – cleaned, concatenated plain-text log (one file per run)
  2. <out>.json         – structured per-variant records ready for ClickHouse
                          ingest (see ingest_model_ops.py)

Each GHA job log file corresponds to one model suite (e.g. "GPT OSS 20B Spyre").
The parser extracts every individual pytest test-case result (XPASS / XFAIL /
FallbackWarning) using the same logic as run_all_test.py::TestLogAnalyzer,
producing one JSON record per variant.

JSON output schema (array of objects)
--------------------------------------
  run_id              : GHA run ID string
  suite_name          : human-readable model suite (from filename)
  model_name          : config stem, e.g. "gpt-oss-20b"
  yaml_file           : config filename, e.g. "gpt-oss-20b_spyre.yaml"
  operation           : torch op name, e.g. "torch.mul"
  classification      : "spyre_enabled" | "not_implemented" | "cpu_fallback"
  test_name           : pytest node id, e.g. "test_model_ops_db_torch_mul__1_spyre_float16"
  status              : "XPASS" | "XFAIL" | "FALLBACK"
  input_shapes        : list[str] of per-tensor shape strings, e.g. ["[1,12,4096]"]
  input_strides       : list[str] of per-tensor stride strings
  input_dtypes        : list[str] of per-tensor dtype strings, e.g. ["torch.float16"]
  arg_values          : list[str] of non-tensor argument values (may be empty)
  target_shape        : reshaped target shape string (for reshape/view ops, else "")
  triggered_at        : ISO-8601 timestamp of first GHA log line in this file
  ingested_at         : ISO-8601 timestamp (now)

  # Suite-level fields (same for every variant in a suite)
  suite_outcome       : "passed" | "failed" | "error" | "unknown"
  suite_exit_code     : int or null
  suite_tests_total   : int
  suite_tests_passed  : int
  suite_tests_failed  : int
  suite_tests_skipped : int
  suite_tests_error   : int
  suite_tests_xfail   : int
  suite_tests_xpass   : int
  suite_duration_s    : float

Usage
-----
  python3 parse_model_ops_logs.py \\
      --log-dir  raw_logs/ \\
      --run-id   <GHA_RUN_ID> \\
      --out      model_ops_<run_id>.json \\
      --log-out  model_ops_log.txt
"""

import argparse
import json
import regex as re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# GHA / pytest line patterns
# ---------------------------------------------------------------------------

# GHA step timestamp prefix:  2025-01-15T10:23:45.1234567Z  text…
RE_GHA_TS = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(?P<rest>.*)$"
)

# Pytest summary line detector — presence only (last line with "= ... in Xs =" wins)
# Example: "================== 5 passed, 1 xfailed, 2 xpassed in 9.00s =================="
RE_PYTEST_SUMMARY = re.compile(r"={3,}.*\bin (?P<secs>[\d.]+)s")

# Individual count patterns extracted separately from a summary line
_RE_SUMM_FAIL = re.compile(r"(\d+) failed")
_RE_SUMM_PASS = re.compile(r"(\d+) passed")
_RE_SUMM_SKIP = re.compile(r"(\d+) skipped")
_RE_SUMM_ERR = re.compile(r"(\d+) error")
_RE_SUMM_XFAIL = re.compile(r"(\d+) xfailed")
_RE_SUMM_XPASS = re.compile(r"(\d+) xpassed")
RE_COLLECTED = re.compile(r"collected (?P<n>\d+) item")
RE_GHA_EXIT = re.compile(
    r"Error: Process completed with exit code (?P<code>\d+)", re.IGNORECASE
)
RE_COLLECT_ERROR = re.compile(r"ERROR collecting", re.IGNORECASE)
RE_TIMEOUT = re.compile(
    r"(The job running on runner .+ exceeded the maximum execution time"
    r"|No new output for \d+s.*stall)",
    re.IGNORECASE,
)

# ANSI / control chars
_RE_ANSI = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")

# ---------------------------------------------------------------------------
# TestLogAnalyzer patterns
# ---------------------------------------------------------------------------

# GHA compact format (used in CI logs):
#   "test_model_ops_v2.py::Class::test_name XPASS [TAGS...]  [N%]"
#   Followed on subsequent lines by the [INPUT SHAPES] block.
RE_TEST_INLINE = re.compile(
    r"test_model_ops_v2\.py::[^:]+::"
    r"(?P<test>test_model_ops_db_[\w]+)"
    r"\s+(?P<status>XPASS|XFAIL)"
)

# GHA stall-watcher variant — the stall message interrupts the line so the
# status appears on the NEXT line by itself:
#   "…::test_name [stall-watcher] No new output for 30s …"
#   "XPASS [TAGS …]"
# We capture the test name only (no status); the deferred-commit path picks
# up the XPASS/XFAIL that follows.
RE_TEST_STALL = re.compile(
    r"test_model_ops_v2\.py::[^:]+::"
    r"(?P<test>test_model_ops_db_[\w]+)"
    r"\s+\[stall-watcher\]"
)

# Legacy verbose format: status appears on a separate line AFTER Op:/Input: lines.
# Separator line (no inline status):
RE_TEST_SEP_ONLY = re.compile(
    r"test_model_ops_v2\.py::[^:]+::"
    r"(?P<test>test_model_ops_db_[\w]+)"
    r"(?!\s+(?:XPASS|XFAIL))"  # negative lookahead — no inline status
)

# Standalone XPASS/XFAIL line (stall-watcher or legacy verbose)
RE_XPASS_ALONE = re.compile(r"^XPASS\b")
RE_XFAIL_ALONE = re.compile(r"^XFAIL\b")

# Legacy: "Op: torch.mul | Test: test_model_ops_db_torch_mul__1_spyre_float16"
RE_OP_LINE = re.compile(
    r"Op:\s+(?P<op>[\w.]+)\s+\|\s+Test:\s+(?P<test>test_model_ops_db_[\w]+)"
)

# Legacy: XPASS / XFAIL on their own line
RE_XPASS_LINE = re.compile(r"^\s*XPASS")
RE_XFAIL_LINE = re.compile(r"^\s*XFAIL")

# [INPUT SHAPES] section header (GHA format, appears AFTER the test+status line)
RE_INPUT_SHAPES_HDR = re.compile(r"^\[INPUT SHAPES\]$")

# arg[N]: Tensor(shape=[…], dtype=torch.x, stride=[…])
RE_ARG_TENSOR = re.compile(
    r"arg\[\d+\]:\s+Tensor\("
    r"shape=(?P<shape>\[[\d,\s\-]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+),\s+"
    r"stride=(?P<stride>\[[\d,\s\-]+\])\)"
)

# arg[N]: TensorList[Tensor(…), …]
RE_ARG_TENSORLIST = re.compile(r"arg\[\d+\]:\s+TensorList\[(?P<contents>.+)\]")

# Individual Tensor inside TensorList
RE_TENSOR_IN_LIST = re.compile(
    r"Tensor\("
    r"shape=(?P<shape>\[[\d,\s\-]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+),\s+"
    r"stride=(?P<stride>\[[\d,\s\-]+\])\)"
)

# arg[N]: value=<scalar or quoted string>
RE_ARG_VALUE = re.compile(r"arg\[\d+\]:\s+value=(?P<val>.+)")

# arg[N]: py='<python repr>'  (slice objects etc. — treated as arg_values)
RE_ARG_PY = re.compile(r"arg\[\d+\]:\s+py=(?P<val>.+)")

# Legacy single-tensor input line:
#   "Input: shape=[1, 41, 4096], stride=[167936, 4096, 1], dtype=torch.bfloat16"
RE_INPUT_SINGLE = re.compile(
    r"Input:\s+shape=(?P<shape>\[[\d,\s]+\]),\s+"
    r"stride=(?P<stride>\[[\d,\s]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+)"
)

# Legacy tensor-in-list item:
#   "  [0]: shape=[…], stride=[…], dtype=torch.x"
RE_INPUT_LIST_ITEM = re.compile(
    r"^\s*\[\d+\]:\s+shape=(?P<shape>\[[\d,\s]+\]),\s+"
    r"stride=(?P<stride>\[[\d,\s]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+)"
)

# Legacy tensor-in-Args block:
#   "  [0]: Tensor(shape=[…], stride=[…], dtype=torch.x)"
RE_ARGS_TENSOR = re.compile(
    r"^\s*\[\d+\]:\s+Tensor\(shape=(?P<shape>\[[\d,\s]+\]),\s+"
    r"stride=(?P<stride>\[[\d,\s]+\]),\s+"
    r"dtype=(?P<dtype>torch\.\w+)\)"
)

# Legacy non-tensor arg value:  "  [0]: 1e-05"
RE_LEGACY_ARG_VALUE = re.compile(r"^\s*\[\d+\]:\s+(?!Tensor\()(?P<val>.+)$")

# Legacy Target shape:  "Target shape: (1, 12, -1, 128)"
RE_TARGET_SHAPE = re.compile(r"Target shape:\s+(?P<shape>\([^)]+\))")

# Legacy Input: List of N tensors header
RE_INPUT_LIST_HDR = re.compile(r"Input:\s+List of \d+ tensors:")

# FallbackWarning:
#   "FallbackWarning: aten.cos.default is falling back to cpu"
RE_FALLBACK_ATEN = re.compile(
    r"FallbackWarning:\s+(?P<op>aten\.[\w.]+)\s+is falling back"
)
#   "FallbackWarning: conversion from torch.int64 to torch.float32 is falling back"
RE_FALLBACK_CONV = re.compile(
    r"FallbackWarning:\s+conversion from\s+(?P<src>torch\.\w+)\s+to\s+"
    r"(?P<dst>torch\.\w+)\s+is falling back"
)


def _normalize_op_name(op_name: str) -> str:
    """
    Normalize operation names to treat similar ops as the same.
    Implements the logic from run_all_test.py

    Examples:
    - torch.embedding → torch.nn.functional.embedding
    - aten.embedding.default → torch.nn.functional.embedding
    - aten.index_copy.out → torch.index_copy_
    - torch.index_copy.out → torch.index_copy_
    - torch.nn_functional_embedding → torch.nn.functional.embedding
    """
    if not op_name:
        return op_name

    # Convert underscores after "nn" and "functional" keywords to dots
    # This handles cases like torch.nn_functional_embedding → torch.nn.functional.embedding
    if "nn_" in op_name:
        op_name = op_name.replace("nn_", "nn.")
    if "functional_" in op_name:
        op_name = op_name.replace("functional_", "functional.")

    # Convert aten ops to torch ops
    if op_name.startswith("aten."):
        op_name = (
            op_name.replace("aten.", "torch.").split(".default")[0].split(".out")[0]
        )

    # Normalize embedding operations
    if "embedding" in op_name.lower():
        return "torch.nn.functional.embedding"

    # Normalize cos/sin operations
    if op_name in ["aten.cos", "torch.cos"]:
        return "torch.cos"
    if op_name in ["aten.sin", "torch.sin"]:
        return "torch.sin"

    # Normalize index_copy operations
    if "index_copy" in op_name.lower() or "index.copy" in op_name.lower():
        return "torch.index_copy_"

    return op_name


def _op_from_test_name(test_name: str) -> str:
    """
    Derive the torch op name from a pytest test node name.

    Examples
    --------
    test_model_ops_db_torch_mul__1_spyre_float16
        → torch.mul
    test_model_ops_db_torch_Tensor_contiguous__49_spyre_float16
        → torch.Tensor.contiguous
    test_model_ops_db_torch_nn_functional_linear__23_spyre_float16
        → torch.nn.functional.linear
    test_model_ops_db_torch___eq____43_spyre_int64
        → torch.__eq__
    test_model_ops_db_torch__C__log_api_usage_once__16_spyre_float16
        → torch._C._log_api_usage_once
    test_model_ops_db_torch_index_copy_out__43_spyre_float16
        → torch.index_copy_
    """
    # Strip known prefix
    s = re.sub(r"^test_model_ops_db_", "", test_name)
    # Strip trailing __<number>... (variant index + dtype suffix)
    s = re.sub(r"__\d+.*$", "", s)

    # Replace underscores with dots in specific positions:
    # 1. First underscore after "torch"
    # 2. Underscore after "Tensor" (when it appears as a word)
    # 3. Underscore after single capital letters (like _C_)
    # All other underscores remain as underscores

    if s.startswith("torch_"):
        # Replace the first underscore after "torch"
        s = "torch." + s[6:]

    # Replace underscore after "Tensor" when it's a complete word
    s = re.sub(r"\bTensor_", "Tensor.", s)

    # Replace underscore after single capital letter (like _C_)
    # Pattern: underscore + single capital letter + underscore → underscore + capital + dot
    s = re.sub(r"_([A-Z])_", r"_\1.", s)

    # Apply normalization (handles embedding, index_copy, cos, sin)
    s = _normalize_op_name(s)

    return s


def _clean(s: str) -> str:
    s = _RE_ANSI.sub("", s)
    s = _RE_CTRL.sub("", s)
    return s.strip()


def _strip_gha_prefix(line: str):
    """Return (iso_ts_or_None, bare_line)."""
    m = RE_GHA_TS.match(line)
    if m:
        return m.group("ts"), m.group("rest")
    return None, line


def _parse_ts(ts_str):
    if not ts_str:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def _compact_shape(raw: str) -> str:
    """Normalise shape string → '[1,12,4096]' (no spaces)."""
    return re.sub(r"\s+", "", raw)


# ---------------------------------------------------------------------------
# Suite-name / model-name extraction from filename
# ---------------------------------------------------------------------------

_SKIP_NAMES = re.compile(
    r"^(detect changed|run spyre unit|ingest|push.*(clickhouse|diagnostics)|"
    r"checkout|install|derive|upload|build|gather|set up)",
    re.IGNORECASE,
)


def _suite_from_filename(filename: str):
    """Return (suite_name_or_None, model_name_or_None).

    Only files whose suite name ends with "Spyre" (case-insensitive) are
    accepted.  Non-Spyre jobs (e.g. "GPT OSS 20B.txt") are skipped.
    """
    stem = re.sub(r"\.txt$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"^\d+_", "", stem).strip()
    if _SKIP_NAMES.match(stem):
        return None, None

    # Strip "run-tests _ " prefix if present
    m = re.match(r"^run-tests\s*[_|]\s*(.+)$", stem, re.IGNORECASE)
    if m:
        suite_name = m.group(1).strip()
    elif re.search(r"[A-Z]", stem) and (" " in stem or "-" in stem):
        suite_name = stem
    else:
        return None, None

    # Only process Spyre jobs
    if not re.search(r"\bSpyre\b", suite_name, re.IGNORECASE):
        return None, None

    # Derive model_name: lower-case, spaces→hyphens, collapse repeated hyphens
    model_name = re.sub(r"\s+", "-", suite_name.lower())
    model_name = re.sub(r"-+", "-", model_name).strip("-")

    return suite_name, model_name


def _yaml_file_from_model_name(model_name: str) -> str:
    """Best-effort guess at the yaml filename."""
    # e.g. "gpt-oss-20b-spyre" → "gpt-oss-20b_spyre.yaml"
    # Suite names ending in " Spyre" map to _spyre.yaml
    slug = re.sub(r"-spyre$", "_spyre", model_name)
    return f"{slug}.yaml"


def _pick_log_files(log_dir: Path):
    """Return (path, suite_name, model_name) for every recognisable file."""
    candidates = []
    for fpath in sorted(log_dir.iterdir()):
        if not fpath.is_file() or fpath.name.startswith("."):
            continue
        if fpath.suffix not in (".txt", ".log", ""):
            continue
        suite, model = _suite_from_filename(fpath.name)
        if suite is None:
            continue
        candidates.append((fpath, suite, model))

    # Deduplicate: prefer .txt over extension-less
    seen: dict = {}
    for fpath, suite, model in candidates:
        if suite not in seen:
            seen[suite] = (fpath, model)
        elif fpath.suffix in (".txt", ".log") and seen[suite][0].suffix == "":
            seen[suite] = (fpath, model)

    return [(path, suite, model) for suite, (path, model) in sorted(seen.items())]


# ---------------------------------------------------------------------------
# Core per-file parser
# ---------------------------------------------------------------------------


class _TestLogAnalyzer:
    """
    Stateful line-by-line parser for a single model-ops GHA job log.

    Supports two log formats:

    (A) GHA compact format (CI logs downloaded from GitHub Actions):
        test_model_ops_v2.py::Class::test_name XPASS [TAGS...]  [N%]
          [INPUT SHAPES]
          arg[0]: Tensor(shape=[1, 12, 4096], dtype=torch.float16, stride=[49152, 4096, 1])
          arg[1]: value=1e-05
          arg[2]: value='(1, 12, -1, 128)'
          arg[3]: TensorList[Tensor(shape=…), …]
          arg[4]: py='(None, slice(None, None, None), None)'

        XPASS/XFAIL is on the same line as the test id; [INPUT SHAPES] follows.
        The record is only committed when the [INPUT SHAPES] block ends
        (empty line or next test separator) so shapes are captured correctly.

    (B) Legacy verbose (-v -s) format:
        test_model_ops_v2.py::Class::test_name
          Op: torch.mul | Test: test_name
          Input: shape=…, stride=…, dtype=…
          Args:
            [0]: 0.5
          Target shape: (…)
        XPASS […]
    """

    def __init__(self):
        # Completed variant records
        self.xpass_variants: dict[str, dict] = {}
        self.xfail_variants: dict[str, dict] = {}
        self.fallback_ops: set = set()

        # Pending record being built (test seen, shapes not yet collected)
        self._pending_test: str | None = None
        self._pending_op: str | None = None
        self._pending_status: str | None = None  # "XPASS" | "XFAIL"
        self._in_shapes_block: bool = False  # inside [INPUT SHAPES]

        # Accumulated shape/stride/dtype/arg data for the pending record
        self._shapes: list[str] = []
        self._strides: list[str] = []
        self._dtypes: list[str] = []
        self._args: list[str] = []
        self._target_shape: str = ""

        # Legacy verbose-format state
        self._legacy_op: str | None = None
        self._legacy_test: str | None = None
        self._legacy_in_block: bool = False

    # ── public ────────────────────────────────────────────────────────────

    def feed(self, line: str):
        line = _clean(line)

        # FallbackWarning can appear anywhere — handle first
        if "FallbackWarning" in line:
            m = RE_FALLBACK_ATEN.search(line)
            if m:
                aten = m.group("op")
                torch_op = aten.replace("aten.", "torch.").split(".default")[0]
                # Apply normalization (handles embedding, index_copy, cos, sin)
                torch_op = _normalize_op_name(torch_op)
                self.fallback_ops.add(torch_op)
                return
            m = RE_FALLBACK_CONV.search(line)
            if m:
                # Skip type_conversion operations - don't add them to fallback_ops
                return

        # ── (A) GHA compact: test + inline XPASS/XFAIL ───────────────────
        m = RE_TEST_INLINE.search(line)
        if m:
            # Commit any previous pending record before starting a new one
            self._flush_pending()
            self._pending_test = m.group("test")
            self._pending_op = _op_from_test_name(self._pending_test)
            self._pending_status = m.group("status")
            self._in_shapes_block = False
            self._clear_shape_state()
            return

        # ── (A) Stall-watcher: test on this line, XPASS/XFAIL on next line ─
        m = RE_TEST_STALL.search(line)
        if m:
            self._flush_pending()
            self._pending_test = m.group("test")
            self._pending_op = _op_from_test_name(self._pending_test)
            self._pending_status = None  # will be set by standalone XPASS/XFAIL
            self._in_shapes_block = False
            self._clear_shape_state()
            return

        # ── (A/stall) Standalone XPASS/XFAIL → set status on pending record ─
        # Handles stall-watcher split lines and legacy verbose fallback.
        if RE_XPASS_ALONE.match(line):
            if self._pending_test and self._pending_status is None:
                self._pending_status = "XPASS"
            elif self._legacy_op and self._legacy_test:
                self._commit_legacy("XPASS")
            return
        if RE_XFAIL_ALONE.match(line):
            if self._pending_test and self._pending_status is None:
                self._pending_status = "XFAIL"
            elif self._legacy_op and self._legacy_test:
                self._commit_legacy("XFAIL")
            return

        # ── (A) [INPUT SHAPES] header ─────────────────────────────────────
        if RE_INPUT_SHAPES_HDR.match(line):
            if self._pending_test and self._pending_status:
                self._in_shapes_block = True
            return

        # ── (A) arg[N] lines inside [INPUT SHAPES] ────────────────────────
        if self._in_shapes_block and self._pending_test:
            # Empty line ends the block → commit
            if not line:
                self._flush_pending()
                return

            # arg[N]: Tensor(shape=…, dtype=…, stride=…)
            m = RE_ARG_TENSOR.search(line)
            if m:
                self._shapes.append(_compact_shape(m.group("shape")))
                self._strides.append(_compact_shape(m.group("stride")))
                self._dtypes.append(m.group("dtype"))
                return

            # arg[N]: TensorList[Tensor(…), …]
            m = RE_ARG_TENSORLIST.search(line)
            if m:
                for tm in RE_TENSOR_IN_LIST.finditer(m.group("contents")):
                    self._shapes.append(_compact_shape(tm.group("shape")))
                    self._strides.append(_compact_shape(tm.group("stride")))
                    self._dtypes.append(tm.group("dtype"))
                return

            # arg[N]: value=…
            m = RE_ARG_VALUE.search(line)
            if m:
                raw = m.group("val").strip().strip("'\"")
                op = self._pending_op or ""

                # Check if this is a shape (tuple or list format)
                is_shape = False
                shape_content = ""

                # Handle tuple format: (1, 2, 3)
                if raw.startswith("(") and raw.endswith(")"):
                    shape_content = raw[1:-1].strip()
                    is_shape = True
                # Handle list format: [1, 2, 3]
                elif raw.startswith("[") and raw.endswith("]"):
                    shape_content = raw[1:-1].strip()
                    is_shape = True

                if is_shape and shape_content:
                    is_numeric = all(c in "0123456789,- " for c in shape_content)

                    # For reshape/view: record as target_shape only, not input_shapes
                    if "view" in op or "reshape" in op:
                        self._target_shape = raw
                        return

                    # For size-creating ops (zeros/ones/empty/full/rand/randn/sym_sum):
                    # the tuple/list IS the output shape — reformat as bracket shape string
                    # e.g. "(1, 8, 2048, 64)" → "[1,8,2048,64]"
                    # e.g. "[1, 65]" → "[1,65]"
                    _SIZE_OPS = {
                        "torch.zeros",
                        "torch.ones",
                        "torch.empty",
                        "torch.full",
                        "torch.rand",
                        "torch.randn",
                        "torch.sym_sum",
                    }
                    if is_numeric and op in _SIZE_OPS:
                        dims = [
                            d.strip() for d in shape_content.split(",") if d.strip()
                        ]
                        self._shapes.append("[" + ",".join(dims) + "]")
                        return

                    # Everything else: keep as arg_value
                self._args.append(raw)
                return

            # arg[N]: py=… (slice objects, index expressions)
            m = RE_ARG_PY.search(line)
            if m:
                self._args.append(m.group("val").strip().strip("'\""))
                return

            # Another test line starting — flush first, then re-process
            if "test_model_ops_v2.py::" in line:
                self._flush_pending()
                self.feed(line)  # re-enter to handle as new test
            return

        # ── (B) Legacy verbose format ─────────────────────────────────────

        # Legacy separator (no inline XPASS/XFAIL)
        m = RE_TEST_SEP_ONLY.search(line)
        if m and "::" in line:
            self._flush_pending()
            self._legacy_test = m.group("test")
            self._legacy_op = _op_from_test_name(self._legacy_test)
            self._legacy_in_block = True
            return

        # Legacy Op: line overrides op name
        m = RE_OP_LINE.search(line)
        if m:
            self._legacy_op = m.group("op")
            self._legacy_test = m.group("test")
            self._legacy_in_block = True
            return

        # Legacy XPASS/XFAIL on its own line (already handled above for
        # stall-watcher pending; if we reach here it's pure legacy verbose)
        if RE_XPASS_LINE.match(line):
            if self._legacy_op and self._legacy_test:
                self._commit_legacy("XPASS")
            return
        if RE_XFAIL_LINE.match(line):
            if self._legacy_op and self._legacy_test:
                self._commit_legacy("XFAIL")
            return

        if not self._legacy_in_block:
            return

        # Legacy Target shape
        m = RE_TARGET_SHAPE.search(line)
        if m:
            self._target_shape = m.group("shape")
            return

        # Legacy single-tensor Input: line
        m = RE_INPUT_SINGLE.search(line)
        if m:
            self._shapes.append(_compact_shape(m.group("shape")))
            self._strides.append(_compact_shape(m.group("stride")))
            self._dtypes.append(m.group("dtype"))
            return

        if RE_INPUT_LIST_HDR.search(line):
            return

        # Legacy [0]: shape=…
        m = RE_INPUT_LIST_ITEM.search(line)
        if m:
            self._shapes.append(_compact_shape(m.group("shape")))
            self._strides.append(_compact_shape(m.group("stride")))
            self._dtypes.append(m.group("dtype"))
            return

        # Legacy [0]: Tensor(…)
        m = RE_ARGS_TENSOR.search(line)
        if m:
            self._shapes.append(_compact_shape(m.group("shape")))
            self._strides.append(_compact_shape(m.group("stride")))
            self._dtypes.append(m.group("dtype"))
            return

        # Legacy [0]: non-tensor value
        m = RE_LEGACY_ARG_VALUE.match(line)
        if m:
            val = m.group("val").strip()
            if not re.match(r"^(shape|stride|dtype|Tensor|Args)\b", val, re.IGNORECASE):
                self._args.append(val)

    def finish(self):
        """Flush any pending record at end of file."""
        self._flush_pending()

    # ── internal ─────────────────────────────────────────────────────────

    def _clear_shape_state(self):
        self._shapes = []
        self._strides = []
        self._dtypes = []
        self._args = []
        self._target_shape = ""

    def _flush_pending(self):
        """Commit the pending GHA-format record (if any) and reset pending state."""
        if self._pending_test and self._pending_status and self._pending_op:
            self._store(
                op=self._pending_op,
                test=self._pending_test,
                status=self._pending_status,
            )
        self._pending_test = None
        self._pending_op = None
        self._pending_status = None
        self._in_shapes_block = False
        self._clear_shape_state()

    def _commit_legacy(self, status: str):
        """Commit a legacy verbose-format record."""
        if self._legacy_op and self._legacy_test:
            self._store(
                op=self._legacy_op,
                test=self._legacy_test,
                status=status,
            )
        self._legacy_op = None
        self._legacy_test = None
        self._legacy_in_block = False
        self._clear_shape_state()

    def _store(self, op: str, test: str, status: str):
        """Write the accumulated record into xpass_variants or xfail_variants."""
        # Apply normalization to operation name
        op = _normalize_op_name(op)

        # Skip operations that don't start with "torch." (e.g. type_conversion,
        # aten. residuals, or any other non-torch namespace)
        if not op.startswith("torch."):
            return

        # For ops with no tensor args but scalar value args that look like
        # shapes (e.g. torch.zeros, torch.full), promote values → input_shapes
        shapes = list(self._shapes)
        strides = list(self._strides)
        dtypes = list(self._dtypes)
        args = list(self._args)
        target_shape = self._target_shape

        if not shapes and args:
            # Heuristic: if the only args look like dimension lists, use as shapes
            shape_like = [a for a in args if re.match(r"^[\d,\s]+$", a)]
            if shape_like:
                shapes = [f"[{a.replace(' ', '')}]" for a in shape_like]
                args = [a for a in args if a not in shape_like]

        key = f"{op}|{test}"
        record = {
            "operation": op,
            "classification": "spyre_enabled"
            if status == "XPASS"
            else "not_implemented",
            "test_name": test,
            "input_shapes": shapes,
            "input_strides": strides,
            "input_dtypes": dtypes,
            "arg_values": args,
            "target_shape": target_shape,
            "status": status,
        }
        if status == "XPASS":
            self.xpass_variants[key] = record
        else:
            self.xfail_variants[key] = record


def _group_by_operation(variants: list[dict]) -> list[dict]:
    """
    Group a flat list of per-variant dicts by operation name,
    matching the output schema shown in the dashboard JSON.
    """
    grouped: dict[str, list] = defaultdict(list)
    for v in variants:
        grouped[v["operation"]].append(v)
    result = []
    for op_name in sorted(grouped):
        group_variants = grouped[op_name]
        result.append(
            {
                "operation": op_name,
                "variant_count": len(group_variants),
                "variants": group_variants,
            }
        )
    return result


def _build_spyre_failed(
    xpass_variants: list[dict], xfail_variants: list[dict]
) -> list[dict]:
    """
    Build the 'spyre_failed' section: operations that have BOTH xpass AND xfail
    variants (partial support — some shapes work, some don't).

    Special handling: If an operation appears in both spyre_enabled (xpass) and
    cpu_fallback, it should be moved entirely to cpu_fallback with all details.
    """
    xpass_by_op: dict[str, list] = defaultdict(list)
    xfail_by_op: dict[str, list] = defaultdict(list)
    for v in xpass_variants:
        xpass_by_op[v["operation"]].append(v)
    for v in xfail_variants:
        xfail_by_op[v["operation"]].append(v)

    mixed_ops = set(xpass_by_op) & set(xfail_by_op)
    result = []
    for op_name in sorted(mixed_ops):
        xp = xpass_by_op[op_name]
        xf = xfail_by_op[op_name]
        result.append(
            {
                "operation": op_name,
                "xpass_count": len(xp),
                "xfail_count": len(xf),
                "xpass_variants": xp,
                "xfail_variants": xf,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Suite-level stats from log text
# ---------------------------------------------------------------------------


def _parse_suite_stats(lines: list[str]) -> dict:
    stats = {
        "outcome": "unknown",
        "exit_code": None,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests_error": 0,
        "tests_xfail": 0,
        "tests_xpass": 0,
        "duration_s": 0.0,
    }
    chunk = "\n".join(lines)

    m_col = RE_COLLECTED.search(chunk)
    if m_col:
        stats["tests_total"] = int(m_col.group("n"))

    # Parse last pytest summary line using individual patterns
    def _first_int(pattern, text, default=0):
        mm = pattern.search(text)
        return int(mm.group(1)) if mm else default

    for line in reversed(lines):
        m = RE_PYTEST_SUMMARY.search(line)
        if m:
            stats["tests_failed"] = _first_int(_RE_SUMM_FAIL, line)
            stats["tests_passed"] = _first_int(_RE_SUMM_PASS, line)
            stats["tests_skipped"] = _first_int(_RE_SUMM_SKIP, line)
            stats["tests_error"] = _first_int(_RE_SUMM_ERR, line)
            stats["tests_xfail"] = _first_int(_RE_SUMM_XFAIL, line)
            stats["tests_xpass"] = _first_int(_RE_SUMM_XPASS, line)
            try:
                stats["duration_s"] = float(m.group("secs"))
            except (TypeError, ValueError):
                pass
            if stats["tests_total"] == 0:
                stats["tests_total"] = (
                    stats["tests_passed"]
                    + stats["tests_failed"]
                    + stats["tests_skipped"]
                    + stats["tests_error"]
                )
            break

    # Outcome
    m_exit = RE_GHA_EXIT.search(chunk)
    if m_exit:
        stats["exit_code"] = int(m_exit.group("code"))
        stats["outcome"] = "failed" if stats["exit_code"] != 0 else "passed"
    elif stats["tests_failed"] > 0 or stats["tests_error"] > 0:
        stats["outcome"] = "failed"
        stats["exit_code"] = 1
    elif stats["tests_passed"] > 0 or stats["tests_xpass"] > 0:
        stats["outcome"] = "passed"
        stats["exit_code"] = 0
    elif RE_COLLECT_ERROR.search(chunk):
        stats["outcome"] = "error"
        stats["exit_code"] = 2

    return stats


# ---------------------------------------------------------------------------
# Per-file entry point
# ---------------------------------------------------------------------------


def parse_log_file(
    text: str,
    run_id: str,
    suite_name: str,
    model_name: str,
) -> dict:
    """
    Parse one GHA job log and return a single suite-level record that contains
    every per-variant result plus the suite summary stats.

    Implements the same priority logic as run_all_test.py:
    1. CPU fallback takes precedence - ops with fallback warnings are moved from
       spyre_enabled to cpu_fallback with full variant details
    2. Operations in both XPASS and XFAIL are classified as spyre_failed
    3. Remaining XPASS ops are spyre_enabled, XFAIL ops are not_implemented
    """
    raw_lines = text.splitlines()

    # Strip GHA timestamps, collect first timestamp
    lines: list[str] = []
    first_ts: str | None = None
    for raw in raw_lines:
        ts, bare = _strip_gha_prefix(raw)
        if ts and first_ts is None:
            first_ts = ts
        lines.append(_clean(bare))

    # Suite-level stats
    stats = _parse_suite_stats(lines)

    # Per-variant analysis
    analyzer = _TestLogAnalyzer()
    for line in lines:
        analyzer.feed(line)
    analyzer.finish()

    xpass_list = list(analyzer.xpass_variants.values())
    xfail_list = list(analyzer.xfail_variants.values())

    # PRIORITY 1: Move operations with CPU fallback warnings from spyre_enabled to cpu_fallback
    # CPU fallback takes precedence over spyre_enabled (matching run_all_test.py logic)
    xpass_ops = {v["operation"] for v in xpass_list}

    # Operations that have fallback warnings AND appear in XPASS
    cpu_fallback_with_variants = {}
    ops_to_remove_from_xpass = set()

    for op_name in analyzer.fallback_ops:
        if op_name in xpass_ops:
            # Collect all XPASS variants for this operation
            cpu_fallback_with_variants[op_name] = [
                v for v in xpass_list if v["operation"] == op_name
            ]
            ops_to_remove_from_xpass.add(op_name)

    # Remove fallback ops from xpass_list
    xpass_list = [
        v for v in xpass_list if v["operation"] not in ops_to_remove_from_xpass
    ]

    # Update xpass_ops set after removal
    xpass_ops = {v["operation"] for v in xpass_list}

    # PRIORITY 2: Derive spyre-failed (mixed): ops with both XPASS and XFAIL variants
    # Remove those ops from the clean spyre_enabled / not_implemented groups
    mixed_ops = {v["operation"] for v in xpass_list} & {
        v["operation"] for v in xfail_list
    }
    pure_xpass = [v for v in xpass_list if v["operation"] not in mixed_ops]
    pure_xfail = [v for v in xfail_list if v["operation"] not in mixed_ops]

    # Build cpu_fallback list with full variant information (matching run_all_test.py format)
    cpu_fallback_list = []
    for op_name in sorted(analyzer.fallback_ops):
        if op_name in cpu_fallback_with_variants:
            # Include full variant information for ops that had XPASS variants
            cpu_fallback_list.append(
                {
                    "operation": op_name,
                    "variant_count": len(cpu_fallback_with_variants[op_name]),
                    "variants": cpu_fallback_with_variants[op_name],
                }
            )
        else:
            # Just the operation name (no XPASS variants found)
            cpu_fallback_list.append({"operation": op_name})

    yaml_file = _yaml_file_from_model_name(model_name)

    return {
        "run_id": run_id,
        "suite_name": suite_name,
        "model_name": model_name,
        "yaml_file": yaml_file,
        # Suite summary — counts are number of distinct operations (groups), not variants
        "summary": {
            "total_tests": stats["tests_total"],
            "spyre_enabled_count": len({v["operation"] for v in pure_xpass}),
            "not_implemented_count": len({v["operation"] for v in pure_xfail}),
            "cpu_fallback_count": len(analyzer.fallback_ops),
            "spyre_failed_count": len(mixed_ops),
        },
        # Operations breakdown
        "operations": {
            "spyre_enabled": _group_by_operation(pure_xpass),
            "not_implemented": _group_by_operation(pure_xfail),
            "cpu_fallback": cpu_fallback_list,
            "spyre_failed": _build_spyre_failed(xpass_list, xfail_list),
        },
        # Suite-level outcome (for ingest_model_ops.py aggregation)
        "suite_outcome": stats["outcome"],
        "suite_exit_code": stats["exit_code"],
        "suite_tests_total": stats["tests_total"],
        "suite_tests_passed": stats["tests_passed"],
        "suite_tests_failed": stats["tests_failed"],
        "suite_tests_skipped": stats["tests_skipped"],
        "suite_tests_error": stats["tests_error"],
        "suite_tests_xfail": stats["tests_xfail"],
        "suite_tests_xpass": stats["tests_xpass"],
        "suite_duration_s": stats["duration_s"],
        "triggered_at": _parse_ts(first_ts),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse model-ops GHA job logs → cleaned log + structured JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir",
        metavar="DIR",
        required=True,
        help="Directory containing raw GHA job log files (*.txt / *.log)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GHA run ID, e.g. 27674677047",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--log-out",
        metavar="FILE",
        default="model_ops_log.txt",
        help="Output cleaned plain-text log (default: model_ops_log.txt)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact (non-indented) JSON",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"[error] Not a directory: {log_dir}", file=sys.stderr)
        sys.exit(1)

    file_triples = _pick_log_files(log_dir)
    if not file_triples:
        print(f"[warn] No model-ops log files found in {log_dir}", file=sys.stderr)
        Path(args.out).write_text("[]")
        Path(args.log_out).write_text("")
        sys.exit(0)

    print(f"[info] Found {len(file_triples)} suite log file(s)", file=sys.stderr)

    all_records: list[dict] = []
    log_sections: list[str] = []

    for fpath, suite_name, model_name in file_triples:
        text = fpath.read_text(errors="replace")
        rec = parse_log_file(
            text, run_id=args.run_id, suite_name=suite_name, model_name=model_name
        )
        all_records.append(rec)

        # Summary for log file
        sep = "=" * 72
        xp = rec["summary"]["spyre_enabled_count"]
        xf = rec["summary"]["not_implemented_count"]
        fb = rec["summary"]["cpu_fallback_count"]
        sf = rec["summary"]["spyre_failed_count"]
        log_sections.append(
            f"{sep}\n"
            f"SUITE   : {suite_name}\n"
            f"MODEL   : {model_name}\n"
            f"FILE    : {fpath.name}\n"
            f"OUTCOME : {rec['suite_outcome']}\n"
            f"XPASS(spyre_enabled)={xp}  "
            f"XFAIL(not_implemented)={xf}  "
            f"fallback={fb}  "
            f"mixed(spyre_failed)={sf}\n"
            f"{sep}\n"
            f"{text.strip()}\n"
        )

        print(
            f"[info]  {fpath.name}  suite={suite_name!r}  "
            f"outcome={rec['suite_outcome']}  "
            f"xpass={xp}  xfail={xf}  fallback={fb}  mixed={sf}",
            file=sys.stderr,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(all_records)
    n_passed = sum(1 for r in all_records if r["suite_outcome"] == "passed")
    n_failed = sum(1 for r in all_records if r["suite_outcome"] == "failed")
    n_error = sum(1 for r in all_records if r["suite_outcome"] == "error")

    print("\n[info] ── Summary ──────────────────────────────────", file=sys.stderr)
    print(f"[info]  Total suites : {total}", file=sys.stderr)
    print(f"[info]  Passed       : {n_passed}", file=sys.stderr)
    print(f"[info]  Failed       : {n_failed}", file=sys.stderr)
    print(f"[info]  Error        : {n_error}", file=sys.stderr)

    # ── Write outputs ─────────────────────────────────────────────────────
    indent = None if args.compact else 2

    # (1) Full flat-array consumed by ingest_model_ops.py — all fields preserved
    Path(args.out).write_text(json.dumps(all_records, indent=indent))
    print(f"[info]  Ingest JSON     : {args.out}", file=sys.stderr)

    Path(args.log_out).write_text("\n".join(log_sections))
    print(f"[info]  Log             : {args.log_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
