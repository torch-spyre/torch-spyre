# Copyright 2025 The Torch-Spyre Authors.
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
FFDC (First Failure Data Capture) collector for torch-spyre.

Collects diagnostic context automatically on failure:
  - metadata: timestamp, versions, env
  - failure: category, exception, traceback
  - artifacts: paths to compiler outputs if present
  - runtime: kernel name, code_dir when available
  - hardware_state: placeholder until Spyre access is available

Usage:
    from torch_spyre.profiler._ffdc import collect, REQUIRED_FIELDS
    report = collect(exc, failure_category="compile")
"""

import itertools
import json
import os
import sys
import tempfile
import time
import traceback
import platform
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Failure category constants
CATEGORY_COMPILE = "compile"
CATEGORY_RUNTIME_LAUNCH = "runtime_launch"
CATEGORY_UNIMPLEMENTED = "unimplemented"
CATEGORY_UNKNOWN = "unknown"

# Fields required to consider a report "complete"
REQUIRED_FIELDS = [
    "metadata.timestamp",
    "metadata.torch_version",
    "metadata.python_version",
    "failure.category",
    "failure.exception_type",
    "failure.message",
    "failure.traceback",
    "environment.TORCH_COMPILE_DEBUG",
    "environment.TORCH_SPYRE_DEBUG",
    "environment.SPYRE_INDUCTOR_LOG",
    "artifacts.searched",
]

# Maximum number of FFDC report files to keep in the output directory.
# Oldest reports (by modification time) are deleted first.
_MAX_REPORTS = 50

# Maximum wall-clock seconds to spend searching for compiler artifacts.
# rglob scans can stall on slow or frozen filesystem mounts; this bounds
# the delay before the original exception is re-raised.
_ARTIFACT_SEARCH_TIMEOUT_S = 2.0


def _prune_old_reports(out_dir: Path, keep: int) -> None:
    """Delete the oldest ffdc_*.json files, retaining the newest ``keep`` files.

    Sorts by modification time so retention is age-based across all failure
    categories. Sorting by filename would group by category first (the filename
    is ffdc_{category}_{ts}_{pid}.json), causing recent reports of a
    later-sorting category to be evicted before older ones of an earlier-sorting
    category.
    """
    reports = sorted(
        out_dir.glob("ffdc_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in reports[keep:]:
        old.unlink(missing_ok=True)


def _default_output_dir() -> Path:
    """Return a user-writable directory for FFDC reports.

    Prefers the Torch Inductor cache dir (respects TORCHINDUCTOR_CACHE_DIR,
    falls back to ~/.cache/torch/inductor) so reports land alongside other
    Inductor artifacts. Falls back to tempfile.gettempdir() in environments
    where the Inductor cache is unavailable (e.g. import-only, no torch).
    """
    try:
        from torch._inductor.runtime.runtime_utils import cache_dir as _cache_dir

        return Path(_cache_dir()) / "torch-spyre" / "ffdc_reports"
    except Exception:
        return Path(tempfile.gettempdir()) / "torch-spyre-ffdc"


_ENV_KEYS = [
    "TORCH_COMPILE_DEBUG",
    "TORCH_SPYRE_DEBUG",
    "SPYRE_INDUCTOR_LOG",
    "SPYRE_INDUCTOR_LOG_LEVEL",
    "DUMP_SPYRE_CODE",
    "TORCH_LOGS",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "SENCORES",
    "USE_SPYRE_PROFILER",
]


def _safe_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unavailable"


def _safe_torch_spyre_version() -> str:
    try:
        from torch_spyre.version import __version__

        return __version__
    except Exception:
        return "unavailable"


def _collect_env() -> dict:
    return {k: os.environ.get(k, "") for k in _ENV_KEYS}


def _newest_compile_run(debug_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified run_* subdirectory, or None."""
    try:
        runs = [
            d for d in debug_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
        ]
        return max(runs, key=lambda d: d.stat().st_mtime) if runs else None
    except Exception:
        return None


def _collect_artifacts() -> dict:
    found: list[str] = []
    search_roots = [
        Path(os.getcwd()),
        Path(__file__).resolve().parent.parent.parent,  # repo root
        Path("/dev/shm"),
        Path("/tmp"),
    ]
    filename_patterns = [
        "fx_graph_readable.py",
        "fx_graph_transformed.py",
        "ir_pre_fusion.txt",
        "ir_post_fusion.txt",
        "output_code.py",
        "sdsc_*.json",
        "*.mlir",
        "*.ll",
        "graph_diagram.html",
        "*.log",
        "aot_model_*",
    ]
    for root in search_roots:
        debug_dir = root / "torch_compile_debug"
        if not debug_dir.exists():
            continue
        # Only search the newest run to avoid mixing artifacts from prior failures.
        run_dir = _newest_compile_run(debug_dir)
        if run_dir is None:
            continue
        for pattern in filename_patterns:
            try:
                found.extend(
                    str(m) for m in itertools.islice(run_dir.rglob(pattern), 5)
                )
            except Exception:
                pass

    # Also search the Spyre inductor cache for dxp_standalone bundle artifacts
    try:
        from torch._inductor.runtime.runtime_utils import cache_dir as _cache_dir

        spyre_cache = Path(_cache_dir()) / "inductor-spyre"
        if spyre_cache.exists():
            kernel_dirs = [d for d in spyre_cache.iterdir() if d.is_dir()]
            if kernel_dirs:
                newest_kernel = max(kernel_dirs, key=lambda d: d.stat().st_mtime)
                for pattern in ["sdsc_*.json", "*.mlir", "*.log"]:
                    try:
                        found.extend(
                            str(m)
                            for m in itertools.islice(newest_kernel.rglob(pattern), 5)
                        )
                    except Exception:
                        pass
    except Exception:
        pass

    unique = list(dict.fromkeys(found))
    return {
        "searched": True,
        "found_count": len(unique),
        "paths": unique[:20],
    }


def _collect_hardware_state() -> dict:
    """Best-effort hardware state. Real metrics require Spyre access."""
    state: dict = {"spyre_available": False}
    try:
        import torch

        if hasattr(torch, "spyre"):
            # Run the hardware probe without wait=True shutdown: exiting a
            # `with ThreadPoolExecutor` block calls shutdown(wait=True), which
            # would block until the thread finishes and defeat the timeout.
            _hw_pool = ThreadPoolExecutor(max_workers=1)
            _hw_future = _hw_pool.submit(torch.spyre.is_available)
            _hw_pool.shutdown(wait=False)
            try:
                state["spyre_available"] = _hw_future.result(timeout=1.0)
            except _FutureTimeoutError:
                state["note"] = "hardware probe timed out"
                return state
            if not state["spyre_available"]:
                state["note"] = "hardware state unavailable without Spyre access"
    except Exception:
        state["note"] = "hardware state check failed"
    return state


def collect(
    exc: Optional[BaseException] = None,
    failure_category: str = "unknown",
    kernel_name: Optional[str] = None,
    code_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Collect an FFDC report for the given failure context.

    Args:
        exc: The exception that triggered FFDC (or None for manual call).
        failure_category: One of compile, runtime_launch, unimplemented, unknown.
        kernel_name: Kernel name from SpyreSDSCKernelRunner if available.
        code_dir: Code directory from SpyreSDSCKernelRunner if available.
        output_dir: Directory to write report JSON. Defaults to the Inductor cache dir
            (``~/.cache/torch/inductor/torch-spyre/ffdc_reports``, respecting
            ``TORCHINDUCTOR_CACHE_DIR``), with a fallback to the system temp dir.

    Returns:
        dict with the full FFDC report.
    """
    t0 = time.monotonic()
    collector_errors: list = []
    missing_fields: list = []

    # --- metadata ---
    metadata: dict = {}
    try:
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "pid": os.getpid(),
            "python_version": sys.version,
            "torch_version": _safe_torch_version(),
            "torch_spyre_version": _safe_torch_spyre_version(),
            "platform": platform.platform(),
        }
    except Exception as e:
        collector_errors.append(f"metadata: {e}")

    # --- failure ---
    failure: dict = {"category": failure_category}
    try:
        if exc is not None:
            failure["exception_type"] = type(exc).__name__
            failure["message"] = str(exc)
            failure["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        else:
            failure["exception_type"] = None
            failure["message"] = "manual collection (no exception)"
            failure["traceback"] = None
    except Exception as e:
        collector_errors.append(f"failure: {e}")

    # --- environment ---
    environment: dict = {}
    try:
        environment = _collect_env()
    except Exception as e:
        collector_errors.append(f"environment: {e}")

    # --- artifacts ---
    artifacts: dict = {}
    try:
        # Use shutdown(wait=False) so a stalled rglob on a frozen mount does
        # not block past _ARTIFACT_SEARCH_TIMEOUT_S.  A `with` block would call
        # shutdown(wait=True) on exit, defeating the timeout.
        _art_pool = ThreadPoolExecutor(max_workers=1)
        _art_future = _art_pool.submit(_collect_artifacts)
        _art_pool.shutdown(wait=False)
        try:
            artifacts = _art_future.result(timeout=_ARTIFACT_SEARCH_TIMEOUT_S)
        except _FutureTimeoutError:
            artifacts = {"searched": False, "error": "artifact search timed out"}
            collector_errors.append("artifacts: timed out")
    except Exception as e:
        artifacts = {"searched": False, "error": str(e)}
        collector_errors.append(f"artifacts: {e}")

    # --- runtime context ---
    runtime: dict = {
        "kernel_name": kernel_name or None,
        "code_dir": code_dir or None,
    }

    # --- hardware state ---
    hardware_state: dict = {}
    try:
        hardware_state = _collect_hardware_state()
    except Exception as e:
        hardware_state = {"error": str(e)}
        collector_errors.append(f"hardware_state: {e}")

    elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

    # --- validate required fields ---
    # Derive flat from REQUIRED_FIELDS programmatically so adding a new entry
    # there never silently skews completeness_pct due to a missing .get() call.
    _nested = {
        "metadata": metadata,
        "failure": failure,
        "environment": environment,
        "artifacts": artifacts,
    }
    flat = {
        field: _nested.get(section, {}).get(key)
        for field in REQUIRED_FIELDS
        for section, key in [field.split(".", 1)]
    }
    missing_fields = [k for k, v in flat.items() if v is None]

    report: dict[str, Any] = {
        "metadata": metadata,
        "failure": failure,
        "environment": environment,
        "artifacts": artifacts,
        "runtime": runtime,
        "hardware_state": hardware_state,
        "collector": {
            "capture_latency_ms": elapsed_ms,
            "missing_fields": missing_fields,
            "collector_errors": collector_errors,
            "success": len(collector_errors) == 0,
            "completeness_pct": round(
                100
                * (len(REQUIRED_FIELDS) - len(missing_fields))
                / len(REQUIRED_FIELDS),
                1,
            ),
        },
    }

    # --- write report ---
    try:
        out_dir = Path(output_dir) if output_dir else _default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        safe_category = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in failure_category
        )[:32]
        report_path = out_dir / f"ffdc_{safe_category}_{ts}_{os.getpid()}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        report["_report_path"] = str(report_path)
        _prune_old_reports(out_dir, keep=_MAX_REPORTS)
    except Exception as e:
        report["_report_path"] = None
        report["collector"]["collector_errors"].append(f"write: {e}")
        report["collector"]["success"] = False

    return report


def get_diagnostic_report(
    output_dir: Optional[str] = None,
) -> Optional[dict]:
    """
    Return the most recent FFDC report as a dict, or None if none exist.

    Args:
        output_dir: Directory to search. Defaults to the Inductor cache dir
            (``~/.cache/torch/inductor/torch-spyre/ffdc_reports``, respecting
            ``TORCHINDUCTOR_CACHE_DIR``), with a fallback to the system temp dir.

    Returns:
        Parsed JSON dict of the most recent report, or None.
    """
    search_dir = Path(output_dir) if output_dir else _default_output_dir()
    if not search_dir.exists():
        return None

    # Sort by the timestamp embedded in the filename, not by the full filename.
    # Filenames are ffdc_{category}_{YYYYMMDDTHHMMSS}_{microseconds}_{pid}.json.
    # Sorting by the full name groups by category first, so a stale "unknown"
    # report would outrank a fresh "compile" report.  Sorting by st_mtime fails
    # on filesystems with 1-second resolution (same-second writes are misordered).
    # rsplit from the right handles category names that contain underscores
    # (e.g. runtime_launch): stem.rsplit('_', 3) yields
    # [category_prefix, YYYYMMDDTHHMMSS, microseconds, pid].
    def _ts_key(p: Path) -> str:
        parts = p.stem.rsplit("_", 3)
        return f"{parts[1]}_{parts[2]}" if len(parts) == 4 else ""

    reports = sorted(
        search_dir.glob("ffdc_*.json"),
        key=_ts_key,
        reverse=True,
    )
    if not reports:
        return None
    with open(reports[0]) as f:
        return json.load(f)
