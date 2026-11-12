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


"""Mirrors upstream's ``dynamo_timed`` phases into the timing record.

The pass pipelines and the per-kernel backend calls are timed, but they do not
add up to a compile: the enclosing ``stage:compile_fx:spyre_compile`` region was
carrying most of its time as self time, which is to say unattributed.  That
remainder is lowering, codegen, scheduler construction and AOTAutograd tracing --
none of which torch-spyre owns, so none of which a torch-spyre ``stage()`` call
can reach.

The cost is paid on every ``dynamo_timed`` region whether timing is on or not:
a wrapper call at each end of the region, plus the recording itself when timing
is on.  Installed unconditionally rather than only when timing is on at import,
so that enabling it at runtime works.

Upstream already times them.  Every ``dynamo_timed`` region, and a few regions
entered through ``chromium_event_timed`` directly, funnel through two methods on
``ChromiumEventLogger``, unconditionally and regardless of whether the caller
asked for a pt2 compile event.  Wrapping that pair mirrors all of them into the
record for the cost of two monkeypatches, instead of patching
``GraphLowering.run``, ``GraphLowering.codegen``, ``Scheduler.__init__`` and the
AOT entry point one at a time and then chasing whatever upstream adds next.

Nesting comes out right without being arranged.  Upstream's outermost phase,
``entire_frame_compile``, encloses the backend call, so
``stage:compile_fx:spyre_compile`` lands beneath it and Dynamo's own tracing is
that parent's self time -- the same subtraction the record already uses for the
frontend total.  A representative tree::

    stage:torch:entire_frame_compile        Dynamo, whole frame
      stage:torch:compile_attempt_0
        stage:torch:backend_compile         the call into Inductor
          stage:compile_fx:spyre_compile    ours; still the compile grouping key
            stage:torch:create_aot_dispatcher_function
              stage:torch:aot_trace_joint_graph
              stage:torch:GraphLowering.run           lowering
              stage:torch:GraphLowering.codegen
                stage:GraphLowering:update_scheduler  ours
                  stage:torch:Scheduler.__init__
                stage:torch:Scheduler.codegen
                  pipeline:CustomPreFusionPasses      ours
                  stage:SpyreAsyncCompile:backend_compile   per kernel
      stage:torch:build_guards

Things to know before changing this:

* **Phase names are upstream's, recorded verbatim.**  No mapping table to keep in
  step, and a name is greppable in the torch tree.  The cost is that a torch bump
  can rename one and silently empty a column in a committed baseline.  Only the
  four names a baseline reports are guarded: they are constants below, and a test
  asserts each one appears in a real compile.  Note ``phase_name=`` overrides the
  key, which is why Dynamo's whole-frame region is ``entire_frame_compile`` and
  not ``_compile.compile_inner``.

* **Upstream has a phase called ``backend_compile`` and so do we**, meaning
  something entirely different: upstream's is the call into Inductor, ours is one
  invocation of the Spyre kernel compiler.  The full names differ by owner
  segment, but a reader matching on the trailing segment alone would total the
  whole Inductor compile as backend time and report a frontend near zero.  Match
  ``timing_recorder.BACKEND_COMPILE_EVENT`` exactly.

* **A phase name can repeat, and the repeats nest.**  ``GraphLowering.run``
  recurses into while-loop bodies and subgraphs, so one lowering span can hold
  others of the same name, and totalling ``inclusive_ns`` by name counts the
  inner ones twice.  Take the outermost.  Upstream's ``compile_id`` is in each
  event's meta, but it does not separate these -- they share one; the nesting is
  what separates them.

* **Every compile in the process is recorded, not just Spyre ones.**  Dynamo
  traces before anything knows which device the graph is for, so a filter that
  could tell would also have to run too late to cover Dynamo.  A non-Spyre
  compile shows up as a phase tree with no ``stage:compile_fx:spyre_compile``
  under it, which is what a reader groups by anyway.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from . import config, timing_recorder
from .logging_utils import get_inductor_logger
from .timing_recorder import _Event


logger = get_inductor_logger("timing")

# The phases #4156 asks for by name, as they appear in the record.
DYNAMO_PHASE = "stage:torch:entire_frame_compile"
AOT_PHASE = "stage:torch:create_aot_dispatcher_function"
LOWERING_PHASE = "stage:torch:GraphLowering.run"
CODEGEN_PHASE = "stage:torch:GraphLowering.codegen"

#: Asserted against a real compile rather than trusted, so an upstream rename
#: fails a test instead of quietly emptying a baseline column.
REQUIRED_PHASES = (DYNAMO_PHASE, AOT_PHASE, LOWERING_PHASE, CODEGEN_PHASE)

# Upstream metadata worth keeping: which frame this phase belongs to, the key
# that phase_name displaced, and forward vs backward. The rest is Chromium
# bookkeeping that would only make records bigger.
_KEPT_METADATA = ("compile_id", "fn_name", "is_backward")

_install_lock = threading.Lock()
_installed = False
_originals: Optional[tuple[Any, Any]] = None
_tls = threading.local()


def phase_event_name(event_name: str) -> str:
    return f"stage:torch:{event_name}"


def _open_phases() -> list[tuple[str, _Event]]:
    """This thread's open bridged phases, innermost last.

    Per thread because upstream's own event stack is, and because async compile
    fans codegen out across threads.
    """
    stack = getattr(_tls, "phases", None)
    if stack is None:
        stack = []
        _tls.phases = stack
    return stack


def _begin(event_name: str, metadata: Any) -> None:
    meta = {}
    if isinstance(metadata, dict):
        meta = {k: metadata[k] for k in _KEPT_METADATA if k in metadata}
    event = timing_recorder.RECORDER._new_event(phase_event_name(event_name), meta)
    _open_phases().append((event_name, event))


def _end(event_name: str) -> None:
    # Search from the innermost, as upstream's own pop does: a name can nest
    # inside itself, and a name is all the end callback gives us.
    stack = _open_phases()
    for index in range(len(stack) - 1, -1, -1):
        if stack[index][0] == event_name:
            timing_recorder.RECORDER._close_event(stack[index][1], None)
            # Anything above it never got an end callback -- upstream truncates
            # its event stack wholesale on a Dynamo restart. Dropping them here
            # leaves them in the record marked open, which is what they were.
            del stack[index:]
            return
    # No match: the bridge was installed between this phase's start and its end,
    # or timing was switched on mid-compile. Nothing to close.


def install() -> None:
    """Start mirroring upstream phases. Idempotent.

    Installed unconditionally at import so that enabling timing at runtime works,
    with the gate in the callbacks -- the same place ``timing_recorder.stage``
    reads it.  A torch release that moves ``ChromiumEventLogger`` must not break
    ``import torch_spyre``, so a failure here is logged and dropped.
    """
    global _installed, _originals
    with _install_lock:
        if _installed:
            return
        try:
            from torch._dynamo.utils import ChromiumEventLogger
        except Exception as exc:
            logger.warning(
                "upstream compile phases will not be recorded: %s: %s",
                type(exc).__name__,
                exc,
            )
            return

        orig_start = ChromiumEventLogger.log_event_start
        orig_end = ChromiumEventLogger.log_event_end

        # *args rather than the real signature: this is a private upstream API
        # and a new keyword argument on it should not become a TypeError here.
        def log_event_start(self: Any, *args: Any, **kwargs: Any) -> None:
            orig_start(self, *args, **kwargs)
            if not config.timing:
                return
            try:
                # After the original, which measures cleanly: the phase's own
                # span then excludes upstream's start-side logging, and its
                # metadata has picked up the compile id.
                event_name = args[0] if args else kwargs["event_name"]
                metadata = args[2] if len(args) > 2 else kwargs.get("metadata")
                _begin(event_name, metadata)
            except Exception as exc:
                _warn_once(exc)

        def log_event_end(self: Any, *args: Any, **kwargs: Any) -> None:
            # Gated on open phases, not on config.timing: a phase begun with
            # timing on must still close if timing was turned off before its end.
            if _open_phases():
                try:
                    _end(args[0] if args else kwargs["event_name"])
                except Exception as exc:
                    _warn_once(exc)
            orig_end(self, *args, **kwargs)

        ChromiumEventLogger.log_event_start = log_event_start  # type: ignore[method-assign]
        ChromiumEventLogger.log_event_end = log_event_end  # type: ignore[method-assign]
        _originals = (orig_start, orig_end)
        _installed = True


def uninstall() -> None:
    """Restore upstream's methods. For tests; installation is process-wide."""
    global _installed, _originals, _tls
    with _install_lock:
        if not _installed or _originals is None:
            return
        from torch._dynamo.utils import ChromiumEventLogger

        ChromiumEventLogger.log_event_start = _originals[0]  # type: ignore[method-assign]
        ChromiumEventLogger.log_event_end = _originals[1]  # type: ignore[method-assign]
        _originals = None
        _installed = False
        # A fresh local drops every thread's stack, not just this one's.
        _tls = threading.local()


_warned: set[type[BaseException]] = set()


def _warn_once(exc: BaseException) -> None:
    """A measurement must not be why a compile fails, but a silent gap is worse.

    Once per exception type, so one transient error cannot mute a different kind
    of failure.
    """
    if type(exc) not in _warned:
        _warned.add(type(exc))
        logger.warning(
            "upstream compile phase not recorded: %s: %s", type(exc).__name__, exc
        )
    else:
        logger.debug("upstream compile phase not recorded", exc_info=exc)
