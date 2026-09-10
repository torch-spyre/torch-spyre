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


"""Structured timing records for the compiler frontend.

The INFO-level ``elapsed <ms> <pass>`` line in passes.py is readable but not
comparable: it carries no graph sizes, no run metadata, and only covers one of
the six pass pipelines. This records the same regions as JSON instead, so a
baseline taken today can be compared against one taken after a change.

Off unless ``config.timing``. Enable it with ``TORCH_SPYRE_TIMING=1`` and point
``TORCH_SPYRE_TIMING_OUT`` at a path to get a record::

    {"meta": {...},
     "events": [{"name": "pass:CustomPreSchedulingPasses:span_reduction",
                 "ordinal": 4, "parent_ordinal": 1,
                 "t_start_ns": ..., "t_end_ns": ...,
                 "inclusive_ns": ..., "self_ns": ...,
                 "meta": {"input_operations": 260, "output_operations": 260}},
                ...]}

``meta`` and ``error`` appear only when non-empty; ``open`` appears only on a
region still running when the record was written.

Event names have three shapes, so a reader can group without a lookup table:
``pipeline:<PipelineClass>``, ``pass:<PipelineClass>:<pass_name>``, and
``stage:<PipelineClass>:<what>`` for work a pipeline does around its passes.

Events nest, so ``inclusive_ns`` double-counts across levels and ``self_ns``
does not. Ordinals are assignment-ordered, which reproduces the wall-clock
timeline of one process even when compiles interleave across threads.

One record is written per process: the configured path gets the pid inserted
before its suffix (``rec.json`` -> ``rec.<pid>.json``), because a compile can
fan out into subprocesses and worker processes that inherit the environment,
and a single shared filename would leave one arbitrary winner.

``RECORDER_VERSION`` identifies the shape of the record above. Bump it when an
existing field changes meaning, is renamed, or is removed -- anything that
would make an older record misread by a reader written against the new shape.
Adding a new event name or a new ``meta`` key does not need a bump; readers
must tolerate unknown names and keys.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ContextManager, Optional

from . import config
from .logging_utils import get_inductor_logger


logger = get_inductor_logger("timing")

RECORDER_VERSION = 1


class _DiscardingDict(dict):
    """Swallows writes, so annotating a disabled region retains nothing.

    Covers the three mutators a call site would plausibly reach for. Anything
    else -- ``pop``, ``|=`` -- still behaves like a dict, and would write into a
    process-lifetime object, so keep annotations to these.
    """

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setdefault(self, key: Any, default: Any = None) -> Any:
        return default


@dataclass
class _Event:
    """One timed region."""

    name: str
    ordinal: int
    parent_ordinal: Optional[int]
    t_start_ns: int
    t_end_ns: int = 0
    inclusive_ns: int = 0
    self_ns: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_closed(self) -> bool:
        return self.t_end_ns != 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "ordinal": self.ordinal,
            "parent_ordinal": self.parent_ordinal,
            "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns,
            "inclusive_ns": self.inclusive_ns,
            "self_ns": self.self_ns,
        }
        if not self.is_closed:
            # Still running when the record was written: its durations are not
            # measurements, and a reader must not sum them.
            out["open"] = True
        if self.meta:
            out["meta"] = dict(self.meta)
        if self.error is not None:
            out["error"] = self.error
        return out


class TimingRecorder:
    """Collects timed regions for one process."""

    def __init__(self) -> None:
        self._events: list[_Event] = []
        self._stack_local = threading.local()
        self._lock = threading.Lock()
        self._next_ordinal = 0
        self.run_meta: dict[str, Any] = {}

    @property
    def events(self) -> tuple[_Event, ...]:
        with self._lock:
            return tuple(self._events)

    def stage(self, name: str, **meta: Any) -> ContextManager[_Event]:
        """Time a region, recording ``meta`` alongside it."""
        return _Region(self, name, meta)

    def set_run_meta(self, **kv: Any) -> None:
        self.run_meta.update(kv)

    def finalize(self) -> None:
        """Fill in ``self_ns``: inclusive time minus that of direct children.

        A region still open has no duration yet, so it is left at zero and
        excluded from its parent's subtraction -- otherwise an open child would
        drive a closed parent's self time negative.
        """
        # One snapshot for both passes: an event closing between them would be
        # counted in its own self time but missing from its parent's.
        events = self.events
        child_inclusive: dict[int, int] = {}
        for event in events:
            if event.parent_ordinal is None or not event.is_closed:
                continue
            child_inclusive[event.parent_ordinal] = (
                child_inclusive.get(event.parent_ordinal, 0) + event.inclusive_ns
            )
        for event in events:
            if not event.is_closed:
                event.self_ns = 0
                continue
            event.self_ns = event.inclusive_ns - child_inclusive.get(event.ordinal, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                **_run_metadata(),
                **self.run_meta,
                "recorder_version": RECORDER_VERSION,
                "clock": "time.perf_counter_ns",
                "pid": os.getpid(),
            },
            "events": [event.to_dict() for event in self.events],
        }

    def dump_json(self, path: str) -> None:
        """Write the record. Safe to call more than once; each call rewrites it."""
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            # default=str: a call site may record a sympy expression or a dtype,
            # and losing the whole record to one unserializable value is worse
            # than recording its repr.
            json.dump(self.to_dict(), handle, separators=(",", ":"), default=str)
        # Rename so a crash mid-write cannot leave a truncated record that a
        # roll-up would silently read as a complete one.
        os.replace(tmp, path)

    def _reset(self) -> None:
        """Drop all state. For tests; the recorder is process-wide."""
        with self._lock:
            self._events = []
            self._next_ordinal = 0
            self._stack_local = threading.local()
            self.run_meta = {}

    def _new_event(self, name: str, meta: dict[str, Any]) -> _Event:
        stack = self._stack()
        with self._lock:
            ordinal = self._next_ordinal
            self._next_ordinal = ordinal + 1
            event = _Event(
                name=name,
                ordinal=ordinal,
                parent_ordinal=stack[-1].ordinal if stack else None,
                t_start_ns=time.perf_counter_ns(),
                meta=meta,
            )
            self._events.append(event)
        stack.append(event)
        return event

    def _close_event(self, event: _Event, error: Optional[str]) -> None:
        event.t_end_ns = time.perf_counter_ns()
        event.inclusive_ns = event.t_end_ns - event.t_start_ns
        if error is not None:
            event.error = error
        stack = self._stack()
        # Unwind to and including this event: a region that raised past its own
        # __exit__ would otherwise leave the stack skewed for every later timer.
        while stack:
            if stack.pop() is event:
                break

    def _stack(self) -> list[_Event]:
        stack = getattr(self._stack_local, "stack", None)
        if stack is None:
            stack = []
            self._stack_local.stack = stack
        return stack


class _Region:
    __slots__ = ("_recorder", "_name", "_meta", "_event")

    def __init__(
        self, recorder: TimingRecorder, name: str, meta: dict[str, Any]
    ) -> None:
        self._recorder = recorder
        self._name = name
        self._meta = meta
        self._event: Optional[_Event] = None

    def __enter__(self) -> _Event:
        self._event = self._recorder._new_event(self._name, self._meta)
        return self._event

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._event is not None
        self._recorder._close_event(
            self._event, None if exc is None else f"{exc_type.__name__}: {exc}"
        )


class _NullRegion:
    """Stands in when timing is off, so call sites need no conditional.

    ``__enter__`` hands back a shared event whose ``meta`` discards writes, so
    a call site can annotate it (``event.meta[...] = ...``) without a branch and
    without retaining anything.
    """

    __slots__ = ()

    def __enter__(self) -> _Event:
        return _NULL_EVENT

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


_NULL_EVENT = _Event(
    name="", ordinal=-1, parent_ordinal=None, t_start_ns=0, meta=_DiscardingDict()
)
_NULL_REGION = _NullRegion()

RECORDER = TimingRecorder()


def _git_sha(path: str) -> str:
    """Short HEAD sha of the checkout containing ``path``, or "" if there is none."""
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha.isalnum() else ""


def _run_metadata() -> dict[str, Any]:
    """Substrate a record was produced on.

    A record that cannot say which torch-spyre it measured cannot be compared
    against a later one. ``torch_spyre.version`` appends the short sha only for
    a source checkout -- a wheel install reports a bare version -- so ``git_sha``
    is resolved from the loaded package's directory instead. It names whichever
    checkout contains ``torch_spyre_path``, which is torch-spyre for a source
    checkout but need not be for an install nested inside another repository;
    read the two together. Runs once per record written.
    """
    import torch

    import torch_spyre
    from torch_spyre.version import __version__ as spyre_version

    package_dir = os.path.dirname(os.path.abspath(torch_spyre.__file__))
    return {
        "torch_spyre_version": spyre_version,
        "torch_spyre_path": package_dir,
        "git_sha": _git_sha(package_dir),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }


def record_path(path: str, pid: Optional[int] = None) -> str:
    """Per-process destination derived from ``path``: ``x.json`` -> ``x.<pid>.json``."""
    stem, suffix = os.path.splitext(path)
    return f"{stem}.{pid if pid is not None else os.getpid()}{suffix}"


def is_enabled() -> bool:
    return config.timing


def stage(name: str, **meta: Any) -> ContextManager[_Event]:
    """Time a region when timing is on; otherwise do nothing."""
    if not config.timing:
        return _NULL_REGION
    return RECORDER.stage(name, **meta)


def set_run_meta(**kv: Any) -> None:
    if config.timing:
        RECORDER.set_run_meta(**kv)


def dump_and_finalize(path: Optional[str] = None) -> Optional[str]:
    """Finalize and write this process's record. Returns the path written.

    An explicit ``path`` is written verbatim, so a caller driving this directly
    can name an exact file. The configured destination is shared by every
    process that inherits the environment, so it gets the pid inserted.
    """
    if not config.timing:
        return None
    target = path or (record_path(config.timing_out) if config.timing_out else "")
    if not target:
        return None
    RECORDER.finalize()
    RECORDER.dump_json(target)
    return target


@atexit.register
def _dump_at_exit() -> None:
    # Compilation has no single completion point a record could hang off -- one
    # process may compile many graphs -- so the record is per process.
    try:
        dump_and_finalize()
    except Exception as exc:
        # A measurement must never be the reason a compile-and-run fails, but a
        # record that silently did not appear is worse than a noisy one.
        logger.warning("timing record not written: %s: %s", type(exc).__name__, exc)


__all__ = [
    "RECORDER",
    "RECORDER_VERSION",
    "TimingRecorder",
    "dump_and_finalize",
    "is_enabled",
    "record_path",
    "set_run_meta",
    "stage",
]
