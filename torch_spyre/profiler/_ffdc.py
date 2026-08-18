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

"""
FFDC (First Failure Data Capture) collector for torch-spyre.

Collects diagnostic context automatically on failure:
  - metadata: timestamp, versions, env
  - failure: category, exception, traceback, file, lineno
  - artifacts: paths to compiler outputs if present
  - runtime: kernel name, code_dir when available
  - hardware_state: placeholder until Spyre access is available

Usage:
    from torch_spyre.profiler._ffdc import collect, REQUIRED_FIELDS
    report = collect(exc, failure_category="compile_frontend")
"""

import functools
import itertools
import json
import logging
import os
import platform
import stat
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from weakref import WeakSet


T = TypeVar("T")

_log = logging.getLogger(__name__)
_FutureTimeoutError = TimeoutError


# Failure category constants. The category also encodes where the hook
# fired: frontend compiler, backend (bundle) compiler, or runtime.
CATEGORY_COMPILE_FRONTEND = "compile_frontend"
CATEGORY_COMPILE_BACKEND = "compile_backend"
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


def _call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    """Run ``fn`` in a daemon thread; raise on timeout or worker exception.

    Unlike ``ThreadPoolExecutor`` with ``shutdown(wait=False)``, daemon workers
    do not block interpreter shutdown if ``fn`` stalls past ``timeout_s``.
    """
    result_holder: list[tuple[str, Any]] = []

    def _worker() -> None:
        try:
            result_holder.append(("ok", fn()))
        except BaseException as exc:
            result_holder.append(("err", exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        raise _FutureTimeoutError()
    kind, value = result_holder[0]
    if kind == "err":
        raise value
    return value


def _prune_old_reports(out_dir: Path, keep: int) -> None:
    """Delete the oldest ffdc_*.json files, retaining the newest ``keep`` files.

    Sorts by modification time so retention is age-based across all failure
    categories. Sorting by filename would group by category first (the filename
    is ffdc_{category}_{ts}_{pid}.json), causing recent reports of a
    later-sorting category to be evicted before older ones of an earlier-sorting
    category.
    """
    try:
        paths = list(out_dir.glob("ffdc_*.json"))
    except OSError:
        return
    dated: list[tuple[float, Path]] = []
    for path in paths:
        try:
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode):
                continue
            dated.append((st.st_mtime, path))
        except OSError:
            continue
    dated.sort(key=lambda item: item[0], reverse=True)
    for _, old in dated[keep:]:
        try:
            old.unlink(missing_ok=True)
        except OSError:
            pass


def _restrict_owner_only(path: Path, mode: int) -> None:
    """Apply POSIX owner-only mode. No-op on Windows (DACLs are unchanged)."""
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _dump_json(handle: Any, report: dict) -> None:
    """Serialize ``report`` and flush it to durable storage."""
    json.dump(report, handle, indent=2, default=str)
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so the rename is durable."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _publish_json(report_path: Path, report: dict) -> None:
    """Atomically write ``report`` as UTF-8 JSON with POSIX mode ``0o600``.

    Owner-only permissions are POSIX-only. On Windows this still writes the
    file, but inherited DACLs are left unchanged. Data is flushed and fsynced
    before ``os.replace`` so a host crash cannot leave a truncated newest
    report.
    """
    tmp_path = report_path.with_name(report_path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = -1
    try:
        fd = os.open(tmp_path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            _dump_json(handle, report)
        os.replace(tmp_path, report_path)
        _restrict_owner_only(report_path, 0o600)
        _fsync_dir(report_path.parent)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def _default_output_dir() -> Path:
    """Return a user-writable directory for FFDC reports.

    Prefers the Torch Inductor cache dir (respects ``TORCHINDUCTOR_CACHE_DIR``,
    else ``<tempdir>/torchinductor_<user>`` via Inductor's ``cache_dir()``) so
    reports land alongside other Inductor artifacts. ``<tempdir>`` is
    ``tempfile.gettempdir()`` (typically ``/tmp`` on Linux, or ``$TMPDIR``).
    Falls back to ``<tempdir>/torch-spyre-ffdc`` when that cache root is
    unavailable.
    """
    try:
        from torch._inductor.runtime.runtime_utils import cache_dir as _cache_dir

        return Path(_cache_dir()) / "torch-spyre" / "ffdc_reports"
    except Exception:
        return Path(tempfile.gettempdir()) / "torch-spyre-ffdc"


def _is_safe_category_char(c: str) -> bool:
    """Return True if ``c`` is allowed in an FFDC report category filename."""
    return c.isascii() and (c.isalnum() or c in "_-")


def _report_sort_key(report_path: Path) -> Optional[str]:
    """Return the timestamp sort key for a valid FFDC filename, else None."""
    parts = report_path.stem.rsplit("_", 3)
    if len(parts) != 4:
        return None

    category_prefix, ts_seconds, ts_micros, pid = parts
    if not category_prefix.startswith("ffdc_"):
        return None
    category = category_prefix.removeprefix("ffdc_")
    if not category or not all(_is_safe_category_char(c) for c in category):
        return None
    if len(ts_seconds) != 15 or ts_seconds[8] != "T":
        return None
    if not (
        ts_seconds.isascii() and ts_seconds[:8].isdigit() and ts_seconds[9:15].isdigit()
    ):
        return None
    if not (ts_micros.isascii() and ts_micros.isdigit() and len(ts_micros) == 6):
        return None
    if not (pid.isascii() and pid.isdigit()):
        return None
    try:
        datetime.strptime(f"{ts_seconds}_{ts_micros}", "%Y%m%dT%H%M%S_%f")
    except ValueError:
        return None
    return f"{ts_seconds}_{ts_micros}"


_ENV_KEYS = [
    "TORCH_COMPILE_DEBUG",
    "TORCH_SPYRE_DEBUG",
    "SPYRE_INDUCTOR_LOG",
    "SPYRE_INDUCTOR_LOG_LEVEL",
    "DUMP_SPYRE_CODE",
    "TORCH_LOGS",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "SENCORES",
    "TORCH_SPYRE_FFDC",
]


def _is_ffdc_enabled() -> bool:
    """Return True when auto-capture is enabled via TORCH_SPYRE_FFDC=1."""
    return os.environ.get("TORCH_SPYRE_FFDC") == "1"


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
            try:
                state["spyre_available"] = _call_with_timeout(
                    torch.spyre.is_available, 1.0
                )
            except _FutureTimeoutError:
                state["note"] = "hardware probe timed out after 1.0s"
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
        failure_category: One of compile_frontend, compile_backend,
            runtime_launch, unimplemented, unknown.
        kernel_name: Kernel name from SpyreSDSCKernelRunner if available.
        code_dir: Code directory from SpyreSDSCKernelRunner if available.
        output_dir: Directory to write report JSON. Defaults to
            ``<Inductor cache root>/torch-spyre/ffdc_reports``, where the
            cache root is ``$TORCHINDUCTOR_CACHE_DIR`` or else
            ``<tempdir>/torchinductor_<user>`` (``<tempdir>`` is
            ``tempfile.gettempdir()``, typically ``/tmp`` on Linux,
            overridable via ``TMPDIR``). Falls back to
            ``<tempdir>/torch-spyre-ffdc`` if that root cannot be resolved.
            Owner-only modes (``0o700`` / ``0o600``) are POSIX-only; on
            Windows, inherited DACLs are left unchanged.

    Returns:
        dict with the full FFDC report.

    Auto-capture is gated on ``TORCH_SPYRE_FFDC=1``. When disabled, returns
    the same top-level schema with empty sections and no filesystem or
    thread work.
    """
    if not failure_category:
        failure_category = CATEGORY_UNKNOWN

    if not _is_ffdc_enabled():
        return {
            "metadata": {},
            "failure": {"category": failure_category},
            "environment": {},
            "artifacts": {"searched": False, "found_count": 0, "paths": []},
            "runtime": {
                "kernel_name": kernel_name or None,
                "code_dir": code_dir or None,
            },
            "hardware_state": {"spyre_available": False},
            "collector": {
                "capture_latency_ms": 0.0,
                "missing_fields": [],
                "collector_errors": [],
                "success": True,
                "completeness_pct": 0.0,
                "disabled": True,
            },
            "_report_path": None,
        }

    t0 = time.monotonic()
    collector_errors: list = []

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
            # Innermost frame = where the exception was raised.
            frames = traceback.extract_tb(exc.__traceback__)
            failure["file"] = frames[-1].filename if frames else None
            failure["lineno"] = frames[-1].lineno if frames else None
        else:
            failure["exception_type"] = None
            failure["message"] = "manual collection (no exception)"
            failure["traceback"] = None
            failure["file"] = None
            failure["lineno"] = None
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
        try:
            artifacts = _call_with_timeout(
                _collect_artifacts, _ARTIFACT_SEARCH_TIMEOUT_S
            )
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
        out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        _restrict_owner_only(out_dir, 0o700)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        safe_category = "".join(
            c if _is_safe_category_char(c) else "_" for c in failure_category
        )[:32]
        report_path = out_dir / f"ffdc_{safe_category}_{ts}_{os.getpid()}.json"
        _publish_json(report_path, report)
        report["_report_path"] = str(report_path.resolve())
        _prune_old_reports(out_dir, keep=_MAX_REPORTS)
    except Exception as e:
        report["_report_path"] = None
        report["collector"]["collector_errors"].append(f"write: {e}")
        report["collector"]["success"] = False

    return report


_FFDC_CAPTURED_ATTR = "_torch_spyre_ffdc_captured"
_captured_exceptions: WeakSet[BaseException] = WeakSet()


def _ffdc_already_captured(exc: Optional[BaseException]) -> bool:
    """True if ``exc`` or any ``__cause__`` / ``__context__`` was already hooked."""
    seen: set[int] = set()
    stack: list[BaseException] = []
    if exc is not None:
        stack.append(exc)
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if getattr(cur, _FFDC_CAPTURED_ATTR, False):
            return True
        try:
            if cur in _captured_exceptions:
                return True
        except Exception:
            pass
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None:
            stack.append(cur.__context__)
    return False


def _mark_ffdc_captured(exc: Optional[BaseException]) -> None:
    if exc is None:
        return
    try:
        setattr(exc, _FFDC_CAPTURED_ATTR, True)
    except Exception:
        pass
    try:
        _captured_exceptions.add(exc)
    except Exception:
        pass


def try_collect(
    exc: Optional[BaseException] = None,
    *,
    logger: Any = None,
    **kwargs: Any,
) -> None:
    """Best-effort ``collect`` for failure hooks; never raises.

    Call sites catch a primary failure, call this, then re-raise.
    Collection errors are swallowed here so they cannot replace that
    original exception.

    Nested hooks (backend ``sdsc``/``dbo-opt`` inside ``compile_fx``) must
    not rewrite an inner report as ``compile_frontend``. If this exception or
    its ``__cause__`` / ``__context__`` chain was already captured, skip.
    The inner hook is marked even when ``collect`` fails so a later outer
    hook cannot relabel the same failure.
    """
    if _ffdc_already_captured(exc):
        return
    _mark_ffdc_captured(exc)
    try:
        collect(exc, **kwargs)
    except Exception:
        if logger is not None:
            logger.debug("FFDC collection failed", exc_info=True)


def with_ffdc(
    failure_category: str,
    logger: Any,
    kernel_name_attr: str = "kernel_name",
    code_dir_attr: Optional[str] = "code_dir",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: wrap a runner method with FFDC capture, then re-raise.

    Reads ``self.{kernel_name_attr}`` and optionally ``self.{code_dir_attr}``.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:
                extra: dict[str, Any] = {
                    "failure_category": failure_category,
                    "logger": logger,
                }
                if hasattr(self, kernel_name_attr):
                    extra["kernel_name"] = getattr(self, kernel_name_attr)
                if code_dir_attr and hasattr(self, code_dir_attr):
                    extra["code_dir"] = getattr(self, code_dir_attr)
                try_collect(exc, **extra)
                raise

        return wrapper

    return decorator


def _read_regular_json(report_path: Path) -> Optional[Any]:
    """Load JSON from a regular file; skip FIFOs, devices, and symlinks.

    Parser failures (including ``RecursionError`` on deeply nested JSON)
    return ``None`` so a newer bomb cannot hide an older valid report.
    """
    try:
        if not stat.S_ISREG(os.lstat(report_path).st_mode):
            return None
    except OSError:
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = -1
    try:
        fd = os.open(report_path, flags)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return json.load(handle)
    except Exception:
        return None
    finally:
        if fd >= 0:
            os.close(fd)


def get_diagnostic_report(
    output_dir: Optional[str] = None,
) -> Optional[dict]:
    """
    Return the most recent valid FFDC report as a dict, or None if none remain.

    Args:
        output_dir: Directory to search. Defaults to
            ``<Inductor cache root>/torch-spyre/ffdc_reports``, where the
            cache root is ``$TORCHINDUCTOR_CACHE_DIR`` or else
            ``<tempdir>/torchinductor_<user>`` (``<tempdir>`` is
            ``tempfile.gettempdir()``, typically ``/tmp`` on Linux,
            overridable via ``TMPDIR``). Falls back to
            ``<tempdir>/torch-spyre-ffdc`` if that root cannot be resolved.

    Returns:
        Parsed JSON dict of the most recent valid report, or None. The
        returned dict includes ``_report_path`` with the absolute path of the
        loaded file. Corrupted, non-UTF-8, unreadable, invalidly named,
        non-regular (FIFO/symlink), or structurally invalid report files
        (for example missing a string ``failure.category``) are skipped.
        Returns None when no valid report remains.
    """
    search_dir = Path(output_dir) if output_dir else _default_output_dir()
    try:
        if not search_dir.is_dir():
            return None
        listed = list(search_dir.glob("ffdc_*.json"))
    except OSError:
        _log.debug("FFDC report directory unreadable: %s", search_dir, exc_info=True)
        return None

    # Sort by the timestamp embedded in the filename, not by the full filename.
    # Filenames are ffdc_{category}_{YYYYMMDDTHHMMSS}_{microseconds}_{pid}.json.
    # Sorting by the full name groups by category first, so a stale "unknown"
    # report would outrank a fresh "compile_frontend" report. Sorting by
    # st_mtime fails on filesystems with 1-second resolution (same-second
    # writes are misordered).
    # rsplit from the right handles category names that contain underscores
    # (e.g. runtime_launch).  Valid names split into:
    # [ffdc_{category}, YYYYMMDDTHHMMSS, microseconds, pid].
    candidates = []
    for report_path in listed:
        sort_key = _report_sort_key(report_path)
        if sort_key is not None:
            candidates.append((sort_key, report_path))
    reports = sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    )
    for _, report_path in reports:
        report = _read_regular_json(report_path)
        if not isinstance(report, dict):
            continue
        failure = report.get("failure")
        if not isinstance(failure, dict) or not isinstance(
            failure.get("category"), str
        ):
            continue
        try:
            report["_report_path"] = str(report_path.resolve())
        except OSError:
            report["_report_path"] = str(report_path)
        return report
    return None
