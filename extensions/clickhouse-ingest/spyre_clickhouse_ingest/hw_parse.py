# Copyright 2026 The Torch-Spyre Authors.
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


"""Parse GHA test-log output into hw_failure_diagnostics records; shared per-repo."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import regex as re

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
# The JSON object starts at '{' and, since the blob group is greedy, ends at the
# last '}' on the line.  Applied with .search on an ANSI-stripped line (see
# RasEvents.extract_all), so it tolerates a "[unspecified] <color> " prefix
# before ERRR and any trailing content after the closing '}' (e.g. a color
# reset).  It must NOT anchor the blob to end-of-line: raw GHA logs colorize the
# whole line, so the closing '}' is not the last character.
RE_RAS_LINE = re.compile(
    r"ERRR\s+"
    r"(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<year>\d{4})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d+)"
    r"\s+\[.*?ras_base\.hpp.*?\]\s+"
    r"(?P<blob>\{.+\})"
)

# Also catch RuntimeError: {...RAS...} lines (Python traceback form)
RE_RAS_RUNTIME_ERROR = re.compile(
    r'RuntimeError:\s*(?P<blob>\{[^}]*"name"\s*:\s*"RAS::[^}]+\})'
)

# Retry / stall signals
RE_HW_RETRY_BANNER = re.compile(r"Hardware RAS timeout detected", re.IGNORECASE)
RE_STALL_LINE = re.compile(r"\[stall-watcher\] No new output for (?P<secs>\d+)s")
RE_SIGNAL_EXIT = re.compile(r"SIGNAL EXIT", re.IGNORECASE)

# Matches the "(pod-level retry)" job-name suffix _test_matrix.yaml's retry jobs stamp
# on a suite re-run on a fresh pod; tolerates the parens already being stripped.
RE_POD_LEVEL_RETRY_SUFFIX = re.compile(
    r"\(?\s*pod-level retry\s*\)?\s*$", re.IGNORECASE
)

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
#     first (wrong) match `first_env` finds, capturing garbage like
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
# gather-runner-info's own script preview -- `echo "PCIDEVICE_IBM_COM_AIU_PF
# ${PCIDEVICE_IBM_COM_AIU_PF:-}"` -- references the var name TWICE on one
# line. The first occurrence is followed by `$` (rejected, not in the value
# class), but the SECOND occurrence, inside `${...:-}`, is followed by `:`,
# which the old `[0-9a-fA-F:.,]+` class allowed -- capturing a bare ':' as
# the "value" before ever reaching the real output line. Real PCI addresses
# always start with a hex digit (e.g. "0000:..."), so requiring the first
# captured character to be one rejects that bare-colon false match.
RE_PCI_DEVICE = re.compile(
    r"PCIDEVICE_IBM_COM_AIU_PF(?!\w)[ \t]*(?:->|=)?[ \t]*"
    r"(?P<v>[0-9a-fA-F][0-9a-fA-F:.,]*)"
)
RE_AIU_RANK0 = re.compile(
    r"AIU_WORLD_RANK_0(?!\w)[ \t]*(?:->|=)?[ \t]*(?P<v>[0-9a-fA-F][0-9a-fA-F:.,]*)"
)
RE_PCI_DEV_ID = re.compile(
    r"pcidevid\.cpp.*?Device id \(for card idx \d+\):\s*(?P<v>[0-9a-f:.]+)"
)
RE_OPENED = re.compile(r"Opened:\s*SEN:VFIO:TYPE1:(?P<v>[0-9a-f:.]+)")

# Chip identifiers (also final-attempt only)
# vfio_hal_mnt.cpp prints the first ECID word with a doubled prefix
# ("Raw ECID = 0x0x0000000002038000 0x03e3..."), so `0x` must be repeatable —
# a single-`0x` pattern matches nothing at all.
RE_RAW_ECID = re.compile(
    r"Raw ECID\s*=\s*(?P<v>(?:0x)+[0-9a-fA-F]+\s+(?:0x)+[0-9a-fA-F]+)"
)
RE_CHIP_COORDS = re.compile(
    r"CHIPY=(?P<chipy>0x[0-9a-fA-F]+)\s+CHIPX=(?P<chipx>0x[0-9a-fA-F]+)"
)
RE_WAFER_ID = re.compile(r"Mfg WaferID\s*=\s*(?P<v>\S+)")
RE_MFG_XY = re.compile(r"Mfg \(X,Y\)\s*=\s*\((?P<x>\d+),(?P<y>\d+)\)")
RE_CARD_SERIAL = re.compile(r"Card 11S S/N\s*=\s*(?P<v>\S+)")

# Log-line timestamps.
#
# GHA prefixes virtually every log line with its own ISO-8601 stamp, so that is
# the reliable source. RE_LOG_TS matches the runtime's DTLOG format instead,
# which only appears once the device layer starts talking (~line 1600 of a
# typical job log) — it is kept as a fallback for logs captured without the
# GHA prefix.
RE_GHA_TS = re.compile(r"^﻿?(?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s")
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

# pytest's own terminal summary, e.g. "===== 4 failed, 96 passed in 120.5s =====".
# Counts are read from THIS line only: a hardware log's device chatter ("retry queue:
# 3 errors drained") and pytest's own per-file subtotals both contain "N failed"/
# "N passed" without being the run's verdict. The second alternative accepts a summary
# printed without the "=" rule, which a rule-only pattern would miss and zero the count.
RE_PYTEST_SUMMARY = re.compile(
    r"(?:={3,}[^=\n]*\b\d+ (?:passed|failed|error|skipped|xfailed|xpassed)\b"
    r"[^=\n]*={3,})"
    r"|(?:^\s*\d+ (?:passed|failed|error|skipped|xfailed|xpassed)\b[^\n]*"
    r"\bin \d+(?:\.\d+)?s)"
)

# GHA emits this itself when the step's process exits non-zero, read from the chunk:
# it is the legitimate failure signal for a crash that never reached a pytest summary.
RE_GHA_EXIT_ERROR = re.compile(r"Error: Process completed with exit code [^0]")

# Phase fingerprints
RE_PHASE_COLLECT = re.compile(r"ERROR collecting")
RE_PHASE_FIRMWARE = re.compile(r"initialize_firmware\.cpp")
RE_PHASE_RUNTIME = re.compile(r"start_runtime")

# RAS name → failure_reason mapping. Add new entries here as new codes are discovered.
_RAS_NAME_TO_REASON = {
    "RAS::CBRB::ResponseTimeout": "hardware_ras_timeout",
    "RAS::VFIO::DeviceOpenFail": "hardware_vfio_error",
    "RAS::PCI::BusFence": "hardware_pci_busfence",
}

# Matches ANSI escape sequences (colour codes, cursor moves, resets, etc.)
# and stray ASCII control characters (NUL, ETX, BEL, BS, ...) that GHA
# injects into terminal-captured log lines.
_RE_ANSI = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")

# Jobs that are CI infrastructure, not test suites — skip them entirely
_SKIP_JOB_NAMES = re.compile(
    r"^(detect changed files|run spyre unit tests|ingest|push.*(clickhouse|diagnostics)"
    r"|collect suites for|build torch-spyre wheel|generate test matrix"
    r"|test matrix result|report empty test suites)",
    re.IGNORECASE,
)


class RasClassifier:
    """Maps a RAS event name to its failure_reason label."""

    @staticmethod
    def name_to_reason(name: str) -> str:
        """A RAS name string's failure_reason label."""
        if name in _RAS_NAME_TO_REASON:
            return _RAS_NAME_TO_REASON[name]
        if name.startswith("RAS::"):
            return "hardware_ras_other"
        return "other"


class LogText:
    """Cleans terminal-captured log text and pulls env-var values out of it."""

    @staticmethod
    def clean(s: str) -> str:
        """Strip ANSI escape codes and non-printable control characters."""
        s = _RE_ANSI.sub("", s)
        s = _RE_CTRL.sub("", s)
        return s.strip()

    @staticmethod
    def first_env(pattern: re.Pattern, text: str) -> str:
        """The cleaned value of `pattern`'s first match in `text`, or ''."""
        m = pattern.search(text)
        return _clean(m.group("v")) if m else ""


class Timestamps:
    """Timestamp extraction from a log line."""

    @staticmethod
    def parse(line: str) -> str | None:
        """Timestamp for a log line: GHA's ISO-8601 prefix, else the DTLOG stamp."""
        m = RE_GHA_TS.match(line)
        if m:
            # GHA stamps are always UTC; drop the trailing Z so the value stays
            # naive-UTC like the DTLOG branch below and the ClickHouse column.
            try:
                return datetime.fromisoformat(m["iso"].removesuffix("Z")).isoformat()
            except ValueError:
                pass

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


class PytestSummary:
    """Reads pytest's own terminal-summary line out of a log chunk."""

    @staticmethod
    def summary_line(chunk_lines: list) -> str:
        """The LAST pytest summary line in the chunk (reruns leave several), or ''."""
        found = ""
        for line in chunk_lines:
            if RE_PYTEST_SUMMARY.search(line):
                found = line
        return found

    @staticmethod
    def first_int(pattern: re.Pattern, text: str, group: str = "n") -> int:
        """`pattern`'s first captured integer in `text`, or 0."""
        m = pattern.search(text)
        return int(m.group(group)) if m else 0


class CrashDetector:
    """Detects a process crash (signal abort, segfault, heap corruption) in a chunk."""

    @staticmethod
    def detect(chunk_lines: list[str], chunk: str) -> dict | None:
        """A structured crash-detail dict (signal, message, backtrace), or None."""
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

        pid_m = re.search(
            r"Signal Received from pid=(?P<pid>\d+)", chunk, re.IGNORECASE
        )
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


class RasEvents:
    """Extracts every RAS error event from a log chunk."""

    @staticmethod
    def extract_all(chunk_lines: list[str]) -> list[dict]:
        """Every RAS event in the chunk, in order; double-logged blobs are dropped."""
        events: list[dict] = []
        seen_blobs: set[str] = set()

        for raw_line in chunk_lines:
            # Raw GHA logs colorize every line; strip ANSI + control chars first so
            # both the timestamp prefix and the JSON blob are visible to the
            # regexes.  _clean is the same stripper used on extracted field values.
            line = _clean(raw_line)
            # Primary form: ERRR ... [ras_base.hpp] { ... }.  Use .search, not
            # .match: even after stripping color codes the line begins with a
            # "[unspecified] " source prefix before ERRR.
            m = RE_RAS_LINE.search(line)
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


class LogParser:
    """The core parser: one (potentially multi-attempt) log blob to hw records."""

    @staticmethod
    def parse(
        text: str,
        run_id: str,
        suite_hint: str = "",
        is_pod_level_retry: bool = False,
    ) -> list[dict[str, Any]]:
        """Parse a log blob into one record per (suite, attempt)."""
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
                "pod_level_retry": is_pod_level_retry,
                "ingested_at": datetime.now(UTC).isoformat(),
                # Outcome
                "outcome": "unknown",
                "exit_code": None,
                # Failure classification. failure_reason:
                #   none | hardware_ras_timeout | hardware_vfio_error |
                #   hardware_pci_busfence | hardware_ras_other |
                #   process_crash (signal/abort/heap corruption) |
                #   stall | signal_exit | other
                "failure_reason": "none",
                # Full parsed RAS JSON of the primary (first) error, or {} if none.
                "failure_reason_detail": {},
                # failure_phase: collection|firmware_init|runtime_init|execution|''
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

            # -------------------- Attempt start timestamp --------------------
            # Scan the whole chunk, not a leading window: a chunk opens with GHA
            # setup output and the first parseable stamp can be far in.
            for line in chunk_lines:
                ts = _parse_log_ts(line)
                if ts:
                    rec["attempt_start_ts"] = ts
                    break

            # Computed once: the outcome fallback below and the pytest counts further
            # down both read their numbers from this one line.
            summary = _pytest_summary_line(chunk_lines)

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
                # No attempt banner, so the verdict comes from pytest's summary LINE,
                # not the whole chunk. Searching the chunk let a device log's "3 errors
                # drained" mark a passing suite as failed. The GHA exit line stays
                # chunk-wide: it is the crash signal for an attempt with no summary.
                has_pytest_passed = bool(RE_PY_PASSED.search(summary))
                has_pytest_failed = bool(RE_PY_FAILED.search(summary))
                has_pytest_error = bool(RE_PY_ERROR.search(summary))
                has_gha_exit_error = bool(RE_GHA_EXIT_ERROR.search(chunk))
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
            # A RAS event on a PASSING attempt is a recovered fault: real hardware data
            # worth keeping in the ras_* columns, but not a failure reason -- without
            # this guard a passing run counts as a hardware failure in any dashboard
            # filtering on failure_reason != 'none'.
            ras_failure = bool(ras_events) and rec["outcome"] == "failed"

            # Only consumed by the `elif` below, which requires no RAS events. Computing
            # it eagerly ran the signal/heap scan plus a 10-frame backtrace walk over
            # whole chunk on the common hardware-failure path, then discarded it.
            crash_detail = (
                None if ras_events else _extract_crash_detail(chunk_lines, chunk)
            )

            if ras_failure:
                rec["failure_reason"] = _ras_name_to_reason(rec["ras_name"])
                # failure_reason_detail: the full parsed primary RAS event as a dict,
                # with timestamp and raw blob removed to keep it clean.
                detail = {
                    k: v
                    for k, v in ras_events[0].items()
                    if k not in ("timestamp", "raw")
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
            elif RE_PHASE_FIRMWARE.search(chunk) and ras_failure:
                rec["failure_phase"] = "firmware_init"
            elif RE_PHASE_RUNTIME.search(chunk) and ras_failure:
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

            # Same job-wide-constant reasoning as node_name/pci_device above: the
            # card and chip identity block is printed once by the device-setup step,
            # usually outside any attempt window, so it must be read from `text`.
            # Scoping these to `chunk` populated them in only ~1% of rows.
            rec["card_serial"] = _first_env(RE_CARD_SERIAL, text)
            rec["chip_ecid_raw"] = _first_env(RE_RAW_ECID, text)
            rec["chip_wafer_id"] = _first_env(RE_WAFER_ID, text)

            m_xy = RE_MFG_XY.search(text)
            if m_xy:
                rec["chip_mfg_x"] = m_xy["x"]
                rec["chip_mfg_y"] = m_xy["y"]

            m_coords = RE_CHIP_COORDS.search(text)
            if m_coords:
                rec["chip_chipy"] = m_coords["chipy"]
                rec["chip_chipx"] = m_coords["chipx"]

            # -------------------- Pytest stats --------------------------------
            # "collected N items" is unambiguous, so it may come from anywhere in chunk.
            rec["tests_collected"] = _first_int(RE_COLLECTED, chunk)
            # The rest come from the summary line only. Scanning every line and keeping
            # the last match let a per-file subtotal, or a captured echo of an earlier
            # summary, overwrite the real total. With no summary these stay 0.
            rec["tests_passed"] = _first_int(RE_PY_PASSED, summary)
            rec["tests_failed"] = _first_int(RE_PY_FAILED, summary)
            rec["tests_error"] = _first_int(RE_PY_ERROR, summary)

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


class SuiteFilenames:
    """Turns GHA log filenames into suite names, and picks the right files from dir."""

    @staticmethod
    def from_filename(filename: str) -> tuple[str, bool] | None:
        """(suite_name, is_pod_level_retry) from a GHA log filename, or None to skip."""
        # Strip .txt extension (case-insensitive)
        stem = re.sub(r"\.txt$", "", filename, flags=re.IGNORECASE)
        # Strip leading numeric prefix  e.g. "24_"
        stem = re.sub(r"^\d+_", "", stem)
        stem = stem.strip()

        # Detect + strip the pod-level-retry suffix, before the prefix cleanup below.
        is_pod_level_retry = bool(RE_POD_LEVEL_RETRY_SUFFIX.search(stem))
        if is_pod_level_retry:
            stem = RE_POD_LEVEL_RETRY_SUFFIX.sub("", stem).strip()

        # Strip a leading "run-tests" caller-job prefix before the skip check below,
        # however its separator survived filename sanitization: a literal "_" when the
        # original name used "/" with spaces, or whitespace when "/" was stripped bare.
        m = re.match(r"^run-tests[\s_]+(.+)$", stem, re.IGNORECASE)
        if m:
            stem = m.group(1).strip()

        # Skip meta/gate jobs
        if _SKIP_JOB_NAMES.match(stem):
            return None

        # Extension-less files without a "run-tests _ " prefix are almost always
        # the unnumbered GHA duplicate of a .txt file (same content) OR a stray
        # system file. Only keep them if they look like a GHA job name (contain spaces
        # and a capital letter, matching the "Suite Name" pattern).
        # This prevents oddities like bare filenames without spaces from slipping in.
        if re.search(r"[A-Z]", stem) and " " in stem:
            return stem, is_pod_level_retry
        return None

    @staticmethod
    def pick_from_dir(log_dir: Path) -> list[tuple[Path, str, bool]]:
        """Files in `log_dir` as (path, suite_name, is_pod_level_retry), deduped."""
        # Collect all candidates
        candidates: list[tuple[Path, str, bool]] = []
        for fpath in sorted(log_dir.iterdir()):
            if not fpath.is_file():
                continue
            # Skip macOS metadata files and other hidden files
            if fpath.name.startswith("."):
                continue
            # Accept .txt and .log files, plus extension-less files (GHA produces both)
            if fpath.suffix not in (".txt", ".log", ""):
                continue
            result = _suite_from_filename(fpath.name)
            if result is None:
                continue
            suite, is_pod_level_retry = result
            candidates.append((fpath, suite, is_pod_level_retry))

        # Dedup by (suite name, is_pod_level_retry): keep .txt/.log over extension-less.
        seen: dict[tuple[str, bool], Path] = {}
        for fpath, suite, is_pod_level_retry in candidates:
            key = (suite, is_pod_level_retry)
            if key not in seen:
                seen[key] = fpath
            else:
                # Prefer files with an extension over extension-less duplicates
                if fpath.suffix in (".txt", ".log") and seen[key].suffix == "":
                    seen[key] = fpath

        return [
            (path, suite, is_pod_level_retry)
            for (suite, is_pod_level_retry), path in sorted(seen.items())
        ]


# Function API: bare names, resolved at call time, so tests can still monkeypatch them.
_ras_name_to_reason = RasClassifier.name_to_reason
_clean = LogText.clean
_first_env = LogText.first_env
_parse_log_ts = Timestamps.parse
_pytest_summary_line = PytestSummary.summary_line
_first_int = PytestSummary.first_int
_extract_crash_detail = CrashDetector.detect
_extract_all_ras_events = RasEvents.extract_all
parse_log = LogParser.parse
_suite_from_filename = SuiteFilenames.from_filename
_pick_files_from_dir = SuiteFilenames.pick_from_dir
