#!/usr/bin/env python3
"""
Parses GitHub Actions test log output and produces a structured JSON report
suitable for ingestion into ClickHouse (hw_failure_diagnostics table).

Usage
-----
  # Single log file:
  python3 parse_hardware_failures.py \\
      --log-file run.log \\
      --run-id 27674677047 \\
      [--suite "Inductor Ops Reductions Scalar"]

  # Directory of *.log / *.txt files (one per suite):
  python3 parse_hardware_failures.py \\
      --log-dir ./logs/ \\
      --run-id 27674677047

Please note:
- Hardware identifiers (node name, PCI device, card serial, chip info) only
  appear in the env-var dump on the FINAL attempt (when DTLOG_LEVEL=Info is
  enabled).  The parser back-fills those values onto all earlier attempts of
  the same suite so every record is fully attributed.
- Multiple RAS errors in one attempt are all captured in `ras_events` (a JSON
  array). The `ras_*` top-level fields reflect the FIRST (earliest) event.
- The `failure_reason` is derived from the first RAS event's `name` field, so
  different hardware error types produce different reason codes.
"""

import argparse
import json
import regex as re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ----------------------------
# Regex patterns
# ----------------------------

# Attempt banners
RE_ATTEMPT_START = re.compile(
    r"=== Attempt (?P<attempt>\d+)/(?P<total>\d+):\s*(?P<suite>.+?)\s*==="
)
RE_ATTEMPT_FAILED = re.compile(r"=== Attempt \d+ FAILED \(exit=(?P<exit_code>\d+)\)")
RE_ATTEMPT_PASSED = re.compile(r"=== Attempt \d+ PASSED")

# -------------------- RAS errors ------------------------
# Matches EVERY ras_base.hpp ERRR line regardless of which fields are present.
# The JSON object starts at '{' and ends at the first '}' that closes the root
# object.  All known RAS blobs are single-line, so we match to end-of-line.
RE_RAS_LINE = re.compile(
    r"ERRR\s+"
    r"(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<year>\d{4})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d+)"
    r"\s+\[.*?ras_base\.hpp.*?\]\s+"
    r"(?P<blob>\{.+\})\s*$"
)

# Also catch RuntimeError: {...RAS...} lines (Python traceback form)
RE_RAS_RUNTIME_ERROR = re.compile(
    r'RuntimeError:\s*(?P<blob>\{[^}]*"name"\s*:\s*"RAS::[^}]+\})'
)

# Retry / stall signals
RE_HW_RETRY_BANNER = re.compile(r"Hardware RAS timeout detected", re.IGNORECASE)
RE_STALL_LINE = re.compile(r"\[stall-watcher\] No new output for (?P<secs>\d+)s")
RE_SIGNAL_EXIT = re.compile(r"SIGNAL EXIT", re.IGNORECASE)

# Process crash / signal patterns
# Matches: "Signal Received: 6 (Aborted)" or "Signal Received: 11 (Segmentation fault)"
RE_SIGNAL_RECEIVED = re.compile(
    r"Signal Received:\s*(?P<signum>\d+)\s*\((?P<signame>[^)]+)\)",
    re.IGNORECASE,
)
# Matches: "corrupted double-linked list", "double free or corruption", etc.
RE_HEAP_CORRUPTION = re.compile(
    r"(corrupted double-linked list|double free or corruption"
    r"|malloc(): corrupted top size"
    r"|free(): invalid pointer"
    r"|munmap_chunk\(\): invalid pointer)",
    re.IGNORECASE,
)
# Matches backtrace frame lines:  "E   /lib64/libc.so.6(...)[0x...]"
RE_BACKTRACE_FRAME = re.compile(
    # Matches GHA log lines like:
    #   "E           /lib64/libc.so.6(gsignal+0x16)[0x7f4b324be116]"
    #   "E           /home/senuser/.venv/bin/python3(_start+0x25)[0x561...]"
    # The "E" prefix + 2+ spaces is how pytest captures stderr in logs.
    # The path can be any absolute path - no extension restriction.
    r"^\s*E\s{2,}(?P<lib>/\S+?)(?:\([^)]*\))?\[(?P<addr>0x[0-9a-f]+)\]",
    re.MULTILINE,
)
# Matches backtrace start marker
RE_BACKTRACE_START = re.compile(r"\*{3,}\s*BACKTRACE\s*\*{3,}", re.IGNORECASE)
# Signal number -> human name (taken from the logs)
_SIGNAL_NAMES = {
    "1": "SIGHUP",
    "2": "SIGINT",
    "3": "SIGQUIT",
    "4": "SIGILL",
    "6": "SIGABRT",
    "7": "SIGBUS",
    "8": "SIGFPE",
    "9": "SIGKILL",
    "11": "SIGSEGV",
    "13": "SIGPIPE",
    "15": "SIGTERM",
}

# Hardware environment variables.
# Two independent sources feed these fields:
#   1. The "Gather runner info" composite action
#      (.github/actions/gather-runner-info) echoes
#      "GHA_RUNNER_POD_NODE_NAME <value>" / "GHA_RUNNER_POD_NAME <value>" /
#      "PCIDEVICE_IBM_COM_AIU_PF <value>" on every attempt, and its verbose
#      env dump additionally prints "PCIDEVICE_IBM_COM_AIU_PF=<value>" style
#      lines. This is the reliable source: it fires on every attempt.
#   2. The Spyre runtime's own DTLOG_LEVEL=Info dump, which only fires on the
#      FINAL attempt and uses a "KEY -> value" format.
# These patterns accept a plain space, "=", or "->" separator so either
# source matches. `(?!\w)` after the var name stops "GHA_RUNNER_POD_NAME"
# from matching as a prefix of "GHA_RUNNER_POD_NAMESPACE". Separators use
# `[ \t]*`, not `\s*` — `\s*` also matches newlines, which lets an empty
# value (var echoed with nothing after it) swallow the following line's key
# as its own value.
# The value character class is restricted (not `\S+`) to keep two things out:
#   - `$` and `"`: GHA prints a "script preview" line before a step runs,
#     containing the step's literal source text -- e.g.
#     `echo "GHA_RUNNER_POD_NODE_NAME $GHA_RUNNER_POD_NODE_NAME"` -- which
#     appears BEFORE the real output in the log and would otherwise be the
#     first (wrong) match `_first_env` finds, capturing garbage like
#     `$GHA_RUNNER_POD_NODE_NAME"` off the `$`-prefixed variable reference.
#   - ANSI escape bytes from that same preview line's syntax highlighting.
# `*` IS included: GHA's own secret redaction can replace a substring
# in-place with a literal "***" (e.g. a pod name containing a token that
# happens to match a registered secret), so real values can legitimately
# contain it -- excluding it would truncate the value at the mask.
RE_NODE_NAME = re.compile(
    r"GHA_RUNNER_POD_NODE_NAME(?!\w)[ \t]*(?:->|=)?[ \t]*(?P<v>[\w.*-]+)"
)
RE_POD_NAME = re.compile(
    r"GHA_RUNNER_POD_NAME(?!\w)[ \t]*(?:->|=)?[ \t]*(?P<v>[\w.*-]+)"
)
RE_PCI_DEVICE = re.compile(
    r"PCIDEVICE_IBM_COM_AIU_PF(?!\w)[ \t]*(?:->|=)?[ \t]*(?P<v>[0-9a-fA-F:.,]+)"
)
RE_AIU_RANK0 = re.compile(
    r"AIU_WORLD_RANK_0(?!\w)[ \t]*(?:->|=)?[ \t]*(?P<v>[0-9a-fA-F:.,]+)"
)
RE_PCI_DEV_ID = re.compile(
    r"pcidevid\.cpp.*?Device id \(for card idx \d+\):\s*(?P<v>[0-9a-f:.]+)"
)
RE_OPENED = re.compile(r"Opened:\s*SEN:VFIO:TYPE1:(?P<v>[0-9a-f:.]+)")

# Chip identifiers (also final-attempt only)
RE_RAW_ECID = re.compile(r"Raw ECID\s*=\s*(?P<v>0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+)")
RE_CHIP_COORDS = re.compile(
    r"CHIPY=(?P<chipy>0x[0-9a-fA-F]+)\s+CHIPX=(?P<chipx>0x[0-9a-fA-F]+)"
)
RE_WAFER_ID = re.compile(r"Mfg WaferID\s*=\s*(?P<v>\S+)")
RE_MFG_XY = re.compile(r"Mfg \(X,Y\)\s*=\s*\((?P<x>\d+),(?P<y>\d+)\)")
RE_CARD_SERIAL = re.compile(r"Card 11S S/N\s*=\s*(?P<v>\S+)")

# Log-line timestamps
RE_LOG_TS = re.compile(
    r"(?:ERRR|WARN|INFO|DBUG)\s+"
    r"(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<year>\d{4})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d+)"
)

# Pytest stats
RE_COLLECTED = re.compile(r"collected (?P<n>\d+) item")
RE_PY_PASSED = re.compile(r"(?P<n>\d+) passed")
RE_PY_FAILED = re.compile(r"(?P<n>\d+) failed")
RE_PY_ERROR = re.compile(r"(?P<n>\d+) error")

# Phase fingerprints
RE_PHASE_COLLECT = re.compile(r"ERROR collecting")
RE_PHASE_FIRMWARE = re.compile(r"initialize_firmware\.cpp")
RE_PHASE_RUNTIME = re.compile(r"start_runtime")

# ─────────────────────────────────────────────────────────────────────────────
# RAS name → failure_reason mapping
# Add new entries here as new error codes are discovered.
# ─────────────────────────────────────────────────────────────────────────────
_RAS_NAME_TO_REASON = {
    "RAS::CBRB::ResponseTimeout": "hardware_ras_timeout",
    "RAS::VFIO::DeviceOpenFail": "hardware_vfio_error",
    "RAS::PCI::BusFence": "hardware_pci_busfence",
}


def _ras_name_to_reason(name: str) -> str:
    """Map a RAS name string to a failure_reason label."""
    if name in _RAS_NAME_TO_REASON:
        return _RAS_NAME_TO_REASON[name]
    if name.startswith("RAS::"):
        return "hardware_ras_other"
    return "other"


# ----------------
# Helpers
# ----------------


def _parse_log_ts(line: str) -> str | None:
    m = RE_LOG_TS.search(line)
    if not m:
        return None
    try:
        dt = datetime.strptime(
            f"{m['year']}-{m['month']}-{m['day']} {m['time']}",
            "%Y-%m-%d %H:%M:%S.%f",
        )
        return dt.isoformat()
    except ValueError:
        return None


# Matches ANSI escape sequences (colour codes, cursor moves, resets, etc.)
# and stray ASCII control characters (NUL, ETX, BEL, BS, ...) that GHA
# injects into terminal-captured log lines.
_RE_ANSI = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")


def _clean(s: str) -> str:
    """Strip ANSI escape codes and non-printable control characters."""
    s = _RE_ANSI.sub("", s)
    s = _RE_CTRL.sub("", s)
    return s.strip()


def _first_env(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return _clean(m.group("v")) if m else ""


def _first_int(pattern: re.Pattern, text: str, group: str = "n") -> int:
    m = pattern.search(text)
    return int(m.group(group)) if m else 0


def _extract_crash_detail(chunk_lines: list[str], chunk: str) -> dict | None:
    """
    Detect a process crash (signal abort, segfault, heap corruption, etc.)
    and return a structured detail dict, or None if no crash found.

    Captured fields:
      signal_number   : e.g. "6"
      signal_name     : e.g. "SIGABRT" / "Aborted"
      error_message   : e.g. "corrupted double-linked list"
      backtrace_frames: list of the first 10 shared-library frames
      crash_pid       : pid from "Signal Received from pid=N" if present
    """
    sig_m = RE_SIGNAL_RECEIVED.search(chunk)
    heap_m = RE_HEAP_CORRUPTION.search(chunk)

    if not sig_m and not heap_m:
        return None

    detail: dict = {"type": "process_crash"}

    if sig_m:
        signum = sig_m.group("signum")
        signame_from_log = sig_m.group("signame").strip()
        detail["signal_number"] = signum
        detail["signal_name"] = _SIGNAL_NAMES.get(signum, signame_from_log)
        detail["signal_name_from_log"] = signame_from_log

    pid_m = re.search(r"Signal Received from pid=(?P<pid>\d+)", chunk, re.IGNORECASE)
    if pid_m:
        detail["crash_pid"] = pid_m.group("pid")

    if heap_m:
        detail["error_message"] = heap_m.group(0).strip()
    elif sig_m:
        detail["error_message"] = (
            f"Signal {detail['signal_number']} ({detail['signal_name']})"
        )

    # Collect backtrace frames (first 10 unique library paths)
    frames = []
    seen_libs: set[str] = set()
    for m in RE_BACKTRACE_FRAME.finditer(chunk):
        lib = m.group("lib")
        addr = m.group("addr")
        if lib not in seen_libs:
            seen_libs.add(lib)
            frames.append({"lib": lib, "addr": addr})
        if len(frames) >= 10:
            break
    detail["backtrace_frames"] = frames

    return detail


def _extract_all_ras_events(chunk_lines: list[str]) -> list[dict]:
    """
    Extract every RAS error event from the log chunk.
    Returns a list of dicts, each representing one RAS error, in order of
    appearance.  Deduplicates identical blobs (the same event is often logged
    twice: once by response_block_interface.cpp, once by ras_base.hpp).
    """
    events: list[dict] = []
    seen_blobs: set[str] = set()

    for line in chunk_lines:
        # Primary form: ERRR ... [ras_base.hpp] { ... }
        m = RE_RAS_LINE.match(line.strip())
        blob_str = None
        ts_str = None

        if m:
            blob_str = m.group("blob").strip()
            ts_str = _parse_log_ts(line)
        else:
            # Secondary form: RuntimeError: { ... }  (Python traceback)
            m2 = RE_RAS_RUNTIME_ERROR.search(line)
            if m2:
                blob_str = m2.group("blob").strip()
                ts_str = _parse_log_ts(line)

        if not blob_str or blob_str in seen_blobs:
            continue
        seen_blobs.add(blob_str)

        event: dict[str, Any] = {"timestamp": ts_str or "", "raw": blob_str}
        try:
            parsed = json.loads(blob_str)
            event.update(
                {
                    k: _clean(str(v)) if isinstance(v, str) else str(v)
                    for k, v in parsed.items()
                }
            )
        except json.JSONDecodeError:
            # Partial parse: pull out fields individually
            for field, pat in [
                ("code", re.compile(r'"code"\s*:\s*"([^"]+)"')),
                ("name", re.compile(r'"name"\s*:\s*"([^"]+)"')),
                ("description", re.compile(r'"description"\s*:\s*"([^"]+)"')),
                ("action", re.compile(r'"action"\s*:\s*"([^"]+)"')),
                ("category", re.compile(r'"category"\s*:\s*"([^"]+)"')),
                ("severity", re.compile(r'"severity"\s*:\s*"([^"]+)"')),
                ("message", re.compile(r'"message"\s*:\s*"([^"]+)"')),
            ]:
                fm = pat.search(blob_str)
                if fm:
                    event[field] = fm.group(1)

        events.append(event)

    return events


# --------------------------------
# Core parser
# --------------------------------


def parse_log(text: str, run_id: str, suite_hint: str = "") -> list[dict[str, Any]]:
    """
    Parse a (potentially multi-attempt) log blob.
    Returns one record per (suite, attempt).
    """
    lines = text.splitlines()
    records: list[dict[str, Any]] = []

    # Find attempt banners and slice the log into per-attempt chunks
    slices: list[tuple[int, int, int, int, str]] = []
    for i, line in enumerate(lines):
        m = RE_ATTEMPT_START.search(line)
        if m:
            slices.append(
                (
                    i,
                    len(lines),
                    int(m["attempt"]),
                    int(m["total"]),
                    m["suite"].strip() or suite_hint,
                )
            )

    # Fix slice ends
    for idx in range(len(slices) - 1):
        s = slices[idx]
        slices[idx] = (s[0], slices[idx + 1][0], s[2], s[3], s[4])

    # No banners → treat whole file as a single attempt
    if not slices:
        slices = [(0, len(lines), 1, 1, suite_hint)]

    for start, end, attempt_num, total_attempts, suite_name in slices:
        chunk_lines = lines[start:end]
        chunk = "\n".join(chunk_lines)

        # ── Template record ───────────────────────────────────────────────
        rec: dict[str, Any] = {
            # Identity
            "run_id": run_id,
            "suite_name": suite_name,
            "attempt": attempt_num,
            "total_attempts": total_attempts,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            # Outcome
            "outcome": "unknown",
            "exit_code": None,
            # Failure classification
            # Possible values:
            #   none | hardware_ras_timeout | hardware_vfio_error |
            #   hardware_pci_busfence | hardware_ras_other |
            #   process_crash (signal/abort/heap corruption) |
            #   stall | signal_exit | other
            "failure_reason": "none",
            # Full parsed RAS JSON of the primary (first) error, or {} if none.
            # For non-RAS failures (stall, other) this is {}.
            "failure_reason_detail": {},
            # Possible values:
            #   collection | firmware_init | runtime_init | execution | (empty)
            "failure_phase": "",
            "retry_trigger": "",
            # Primary RAS event (first one seen in this attempt)
            "ras_code": "",
            "ras_name": "",
            "ras_description": "",
            "ras_action": "",
            "ras_category": "",
            "ras_severity": "",
            "ras_message": "",
            # ALL RAS events as a JSON string (array) — for full auditability
            "ras_events_json": "[]",
            # Hardware identifiers
            "node_name": "",
            "pci_device": "",
            "aiu_world_rank0": "",
            "card_serial": "",
            "chip_ecid_raw": "",
            "chip_wafer_id": "",
            "chip_mfg_x": "",
            "chip_mfg_y": "",
            "chip_chipy": "",
            "chip_chipx": "",
            # Timestamps
            "first_error_ts": "",
            "attempt_start_ts": "",
            # Pytest stats
            "tests_collected": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_error": 0,
            # Stall info
            "stall_max_secs": 0,
        }

        # -------------------- Attempt start timestamp ----------------------------------------
        for line in chunk_lines[:20]:
            ts = _parse_log_ts(line)
            if ts:
                rec["attempt_start_ts"] = ts
                break

        # -------------------- Outcome --------------------
        for line in chunk_lines:
            mf = RE_ATTEMPT_FAILED.search(line)
            if mf:
                rec["outcome"] = "failed"
                rec["exit_code"] = int(mf["exit_code"])
                break
            if RE_ATTEMPT_PASSED.search(line):
                rec["outcome"] = "passed"
                break
        if rec["outcome"] == "unknown":
            # Prefer explicit pytest summary lines over generic keyword matching.
            # "N passed" with no "failed" or "error" in the pytest summary → passed.
            # GHA-level "Error: Process completed" line → failed.
            # Avoid false positives from INFO/WARN log lines containing "error".
            has_pytest_passed = bool(re.search(r"\d+ passed", chunk))
            has_pytest_failed = bool(re.search(r"\d+ failed", chunk))
            has_pytest_error = bool(re.search(r"\d+ error", chunk))
            has_gha_exit_error = bool(
                re.search(r"Error: Process completed with exit code [^0]", chunk)
            )
            if has_pytest_passed and not has_pytest_failed and not has_pytest_error:
                rec["outcome"] = "passed"
            elif has_pytest_failed or has_pytest_error or has_gha_exit_error:
                rec["outcome"] = "failed"

        # -------------------- Retry trigger ------------------------
        for line in chunk_lines:
            if "-->" in line and (
                "retry" in line.lower() or "detected" in line.lower()
            ):
                rec["retry_trigger"] = _clean(line)
                break

        # ------------------------ Extract ALL RAS events ------------------------
        ras_events = _extract_all_ras_events(chunk_lines)
        rec["ras_events_json"] = json.dumps(ras_events)

        # Populate top-level fields from the FIRST (earliest) RAS event
        if ras_events:
            first = ras_events[0]
            rec["ras_code"] = first.get("code", "")
            rec["ras_name"] = first.get("name", "")
            rec["ras_description"] = first.get("description", "")
            rec["ras_action"] = first.get("action", "")
            rec["ras_category"] = first.get("category", "")
            rec["ras_severity"] = first.get("severity", "")
            rec["ras_message"] = first.get("message", "")
            rec["first_error_ts"] = first.get("timestamp", "")

        # -------------------- Failure reason + detail ----------------------------
        crash_detail = _extract_crash_detail(chunk_lines, chunk)

        if ras_events:
            rec["failure_reason"] = _ras_name_to_reason(rec["ras_name"])
            # failure_reason_detail: the full parsed primary RAS event as a dict,
            # with timestamp and raw blob removed to keep it clean.
            detail = {
                k: v for k, v in ras_events[0].items() if k not in ("timestamp", "raw")
            }
            rec["failure_reason_detail"] = detail
        elif crash_detail and rec["outcome"] == "failed":
            rec["failure_reason"] = "process_crash"
            rec["failure_reason_detail"] = crash_detail
        elif RE_STALL_LINE.search(chunk) and rec["outcome"] == "failed":
            rec["failure_reason"] = "stall"
        elif RE_SIGNAL_EXIT.search(chunk) and rec["outcome"] == "failed":
            rec["failure_reason"] = "signal_exit"
        elif rec["outcome"] == "failed":
            rec["failure_reason"] = "other"
        # else: "none" (passed)

        # ---------------- Failure phase --------------------
        if RE_PHASE_COLLECT.search(chunk):
            rec["failure_phase"] = "collection"
        elif RE_PHASE_FIRMWARE.search(chunk) and ras_events:
            rec["failure_phase"] = "firmware_init"
        elif RE_PHASE_RUNTIME.search(chunk) and ras_events:
            rec["failure_phase"] = "runtime_init"
        elif rec["failure_reason"] != "none":
            rec["failure_phase"] = "execution"

        # ---------------- Hardware identifiers ------------------------
        # node_name/pod_name/pci_device/aiu_rank0 come from the "Gather
        # runner info" action, which runs ONCE per job as its own step,
        # BEFORE the "=== Attempt N/M ===" banners emitted by the test-runner
        # step. That output sits outside every per-attempt `chunk` (chunks
        # start at each banner line), so these fields are searched against
        # the full per-job log `text`, not `chunk` — they're job-wide
        # constants (same runner pod for every attempt) anyway, and `text`
        # is a superset of `chunk` so the old DTLOG-based match still works.
        #
        # Prefer the k8s node name; fall back to the runner pod name when the
        # node name env var isn't populated (e.g. some clusters only expose
        # GHA_RUNNER_POD_NAME via the downward API, not the node name).
        rec["node_name"] = _first_env(RE_NODE_NAME, text) or _first_env(
            RE_POD_NAME, text
        )
        rec["pci_device"] = (
            _first_env(RE_PCI_DEVICE, text)
            or _first_env(RE_PCI_DEV_ID, chunk)
            or _first_env(RE_OPENED, chunk)
        )
        rec["aiu_world_rank0"] = _first_env(RE_AIU_RANK0, text)
        rec["card_serial"] = _first_env(RE_CARD_SERIAL, chunk)
        rec["chip_ecid_raw"] = _first_env(RE_RAW_ECID, chunk)
        rec["chip_wafer_id"] = _first_env(RE_WAFER_ID, chunk)

        m_xy = RE_MFG_XY.search(chunk)
        if m_xy:
            rec["chip_mfg_x"] = m_xy["x"]
            rec["chip_mfg_y"] = m_xy["y"]

        m_coords = RE_CHIP_COORDS.search(chunk)
        if m_coords:
            rec["chip_chipy"] = m_coords["chipy"]
            rec["chip_chipx"] = m_coords["chipx"]

        # -------------------- Pytest stats --------------------------------
        rec["tests_collected"] = _first_int(RE_COLLECTED, chunk)
        for line in chunk_lines:
            if " passed" in line or " failed" in line or " error" in line:
                mp = RE_PY_PASSED.search(line)
                mf2 = RE_PY_FAILED.search(line)
                me = RE_PY_ERROR.search(line)
                if mp:
                    rec["tests_passed"] = int(mp.group("n"))
                if mf2:
                    rec["tests_failed"] = int(mf2.group("n"))
                if me:
                    rec["tests_error"] = int(me.group("n"))

        # ---------------------------- Stall info ----------------------------
        stall_secs = [
            int(m2["secs"])
            for line in chunk_lines
            if (m2 := RE_STALL_LINE.search(line))
        ]
        rec["stall_max_secs"] = max(stall_secs, default=0)

        records.append(rec)

    # -------- Back-fill hardware IDs from any attempt that has them --------
    # (DTLOG_LEVEL=Info only fires on the final attempt, so IDs only appear
    # there — propagate them to all earlier attempts of the same suite.)
    _hw_fields = (
        "node_name",
        "pci_device",
        "aiu_world_rank0",
        "card_serial",
        "chip_ecid_raw",
        "chip_wafer_id",
        "chip_mfg_x",
        "chip_mfg_y",
        "chip_chipy",
        "chip_chipx",
    )
    best: dict[str, str] = {f: "" for f in _hw_fields}
    for rec in records:
        for f in _hw_fields:
            if not best[f] and rec.get(f):
                best[f] = rec[f]
    for rec in records:
        for f in _hw_fields:
            if not rec.get(f) and best[f]:
                rec[f] = best[f]

    return records


# --------------------------------------------------------------------------------
# Filename ---> suite name extraction
# ------------------------------------------------------------------------------------

# Jobs that are CI infrastructure, not test suites — skip them entirely
_SKIP_JOB_NAMES = re.compile(
    r"^(detect changed files|run spyre unit tests|ingest|push.*(clickhouse|diagnostics))",
    re.IGNORECASE,
)


def _suite_from_filename(filename: str) -> str | None:
    """
    Extract a clean suite name from a GHA log filename.

    Returns None if the file should be skipped (meta/gate job or unrecognised).

    Examples
    --------
    "24_run-tests _ Inductor Ops Reductions Scalar.txt"  → "Inductor Ops Reductions Scalar"
    "run-tests _ Tensor Layout"                          → "Tensor Layout"
    "53_Detect changed files.txt"                        → None  (skipped)
    "0_Run Spyre unit tests.txt"                         → None  (skipped)
    """
    # Strip .txt extension (case-insensitive)
    stem = re.sub(r"\.txt$", "", filename, flags=re.IGNORECASE)
    # Strip leading numeric prefix  e.g. "24_"
    stem = re.sub(r"^\d+_", "", stem)
    stem = stem.strip()

    # Skip meta/gate jobs
    if _SKIP_JOB_NAMES.match(stem):
        return None

    # Strip "run-tests _ " prefix (the GHA job name format)
    m = re.match(r"^run-tests\s*_\s*(.+)$", stem, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Extension-less files without a "run-tests _ " prefix are almost always
    # the unnumbered GHA duplicate of a .txt file (same content) OR a stray
    # system file. Only keep them if they look like a GHA job name (contain spaces
    # and a capital letter, matching the "Suite Name" pattern).
    # This prevents oddities like bare filenames without spaces from slipping in.
    if re.search(r"[A-Z]", stem) and " " in stem:
        return stem
    return None


def _pick_files_from_dir(log_dir: Path) -> list[tuple[Path, str]]:
    """
    Scan log_dir and return (path, suite_name) pairs for every file that
    represents a test suite.

    Deduplication: GHA downloads often produce both a numbered .txt file
    ("24_run-tests _ Foo.txt") AND an unnumbered no-extension copy
    ("run-tests _ Foo").  We prefer the .txt variant and skip the duplicate.
    """
    # Collect all candidates
    candidates: list[tuple[Path, str]] = []
    for fpath in sorted(log_dir.iterdir()):
        if not fpath.is_file():
            continue
        # Skip macOS metadata files and other hidden files
        if fpath.name.startswith("."):
            continue
        # Accept .txt and .log files, plus extension-less files (GHA produces both)
        if fpath.suffix not in (".txt", ".log", ""):
            continue
        suite = _suite_from_filename(fpath.name)
        if suite is None:
            continue
        candidates.append((fpath, suite))

    # Deduplicate by suite name — keep the .txt (or .log) over extension-less
    seen: dict[str, Path] = {}
    for fpath, suite in candidates:
        if suite not in seen:
            seen[suite] = fpath
        else:
            # Prefer files with an extension over extension-less duplicates
            if fpath.suffix in (".txt", ".log") and seen[suite].suffix == "":
                seen[suite] = fpath

    return [(path, suite) for suite, path in sorted(seen.items())]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse GHA hardware failure logs → ClickHouse JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--log-file",
        metavar="FILE",
        help="Path to a single log file (use '-' for stdin)",
    )
    src.add_argument(
        "--log-dir",
        metavar="DIR",
        help=(
            "Directory of log files to scan. "
            "Accepts the exact folder structure produced by GHA log downloads: "
            "files named like '24_run-tests _ Inductor Ops Reductions Scalar.txt'. "
            "Meta/gate jobs (Detect changed files, Run Spyre unit tests) are "
            "skipped automatically. Duplicate numbered/.txt vs unnumbered files "
            "are deduplicated automatically."
        ),
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions run ID shown in the Actions URL, e.g. 27674677047",
    )
    p.add_argument(
        "--suite",
        default="",
        help=(
            "Suite name to use when parsing a single --log-file that has no "
            "'=== Attempt N/M: Suite Name ===' banner. "
            "Ignored when --log-dir is used (names are derived from filenames)."
        ),
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation). Smaller files, faster ingest.",
    )
    p.add_argument(
        "--out", metavar="FILE", help="Write JSON output to FILE instead of stdout."
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    all_records: list[dict[str, Any]] = []

    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            print(f"[error] Not a directory: {log_dir}", file=sys.stderr)
            sys.exit(1)

        file_suite_pairs = _pick_files_from_dir(log_dir)
        if not file_suite_pairs:
            print(
                f"[warn] No recognised test-suite log files found in {log_dir}",
                file=sys.stderr,
            )

        print(
            f"[info] Found {len(file_suite_pairs)} suite log file(s) to parse",
            file=sys.stderr,
        )

        for fpath, suite_name in file_suite_pairs:
            text = fpath.read_text(errors="replace")
            recs = parse_log(text, run_id=args.run_id, suite_hint=suite_name)
            all_records.extend(recs)
            outcomes = [r["outcome"] for r in recs]
            reasons = [
                r["failure_reason"] for r in recs if r["failure_reason"] != "none"
            ]
            print(
                f"[info]  {fpath.name}",
                file=sys.stderr,
            )
            print(
                f"         suite={suite_name!r}  attempts={len(recs)}"
                f"  outcomes={outcomes}  reasons={reasons or ['(none)']!r}",
                file=sys.stderr,
            )

    elif args.log_file == "-" or (not args.log_file and not sys.stdin.isatty()):
        text = sys.stdin.read()
        all_records = parse_log(text, run_id=args.run_id, suite_hint=args.suite)

    elif args.log_file:
        text = Path(args.log_file).read_text(errors="replace")
        all_records = parse_log(text, run_id=args.run_id, suite_hint=args.suite)

    else:
        print(
            "[error] Provide --log-file, --log-dir, or pipe via stdin.", file=sys.stderr
        )
        sys.exit(1)

    # Summary to stderr
    if all_records:
        total = len(all_records)
        failed = sum(1 for r in all_records if r["outcome"] == "failed")
        passed = sum(1 for r in all_records if r["outcome"] == "passed")
        by_reason: dict[str, int] = {}
        for r in all_records:
            k = r["failure_reason"]
            by_reason[k] = by_reason.get(k, 0) + 1
        print("\n[info] ── Summary ──────────────────────────────────", file=sys.stderr)
        print(f"[info]  Total attempt records : {total}", file=sys.stderr)
        print(f"[info]  Passed                : {passed}", file=sys.stderr)
        print(f"[info]  Failed                : {failed}", file=sys.stderr)
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"[info]    {reason:35s}: {count}", file=sys.stderr)

    output = json.dumps(all_records, indent=None if args.compact else 2)

    if args.out:
        Path(args.out).write_text(output)
        print(f"[info]  Output written to: {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
