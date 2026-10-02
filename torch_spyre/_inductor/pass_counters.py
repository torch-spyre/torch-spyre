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


"""Deterministic counts of the frontend's analysis calls.

Timing says a pass is slow. A counter says *why*: how many times it asked the
same question. The two answer different things and fail differently -- a
stopwatch is noisy and unassertable, while a count is exact and can be a test.
The #4113 fix was confirmed by a call count, not by a stopwatch, and the
complexity audit (#4220) needs one falsifying counter per claim.

Counters are read two ways:

* From a timing record. ``passes.py`` snapshots before each pass and writes the
  non-zero deltas into that pass's event ``meta``, so a sweep gets counts and
  time side by side. Pipeline events carry the inclusive total, the same way
  ``inclusive_ns`` does.
* From a test. ``with counted_region() as counts: ...`` installs the wrappers
  regardless of the timing config and hands back what the region counted, so a
  regression guard can assert a bound without turning on recording.

Two mechanisms, split by who owns the code. For Spyre helpers the count lives
in the function body (every consumer imports them by name, so wrapping the
module attribute would miss them). For upstream's
``ComputedBuffer.get_read_writes`` -- which we cannot edit, and which 100-odd
sites call directly, bypassing our memo -- :func:`counting` installs a wrapper
and restores it on exit.

Counts are per-process and unsynchronized: the passes this is aimed at run on
the compiling thread, but a concurrently compiling worker would land in the
same totals.

New counter names need no ``RECORDER_VERSION`` bump; they are event ``meta``,
which readers already treat as open.
"""

from __future__ import annotations

import contextlib
import functools
import time
from typing import Any, Iterator

from torch._inductor.ir import ComputedBuffer


# Calls to pass_utils.op_read_writes, i.e. how many times a pass asked.
READ_WRITES_REQUESTS = "read_writes.requests"
# Of those, the ones the per-op memo could not serve.
READ_WRITES_MISSES = "read_writes.misses"
# ComputedBuffer.get_read_writes invocations: the sympy dependency extraction
# that actually costs something, counting the direct callers that skip the memo.
# A requests count far above this is a rescan that the memo is absorbing; a
# misses count far below this is a pass going around the memo.
READ_WRITES_EXTRACTIONS = "read_writes.extractions"
# Coordinate construction, the second analysis tier. Nothing memoizes these, so
# the count is the work.
DEVICE_COORDINATES = "device_coordinates"
HOST_COORDINATES = "host_coordinates"
# Nanoseconds spent inside ComputedBuffer.get_read_writes. The call count alone
# cannot size the waste: extraction cost varies with body complexity, so a pass
# doing 200 extractions per op is not necessarily dominated by them. Measured at
# ~143 us a call, which made that distinction decidable.
READ_WRITES_EXTRACT_NS = "read_writes.extract_ns"


class PassCounters:
    """One counting region's counts, keyed by counter name."""

    __slots__ = ("enabled", "counts")

    def __init__(self) -> None:
        self.enabled = False
        self.counts: dict[str, int] = {}

    def bump(self, name: str, n: int = 1) -> None:
        """Add to a counter; a no-op while counting is off.

        Hot call sites should still test ``COUNTERS.enabled`` themselves, so the
        disabled path is an attribute load and a branch rather than a call.
        """
        if self.enabled:
            self.counts[name] = self.counts.get(name, 0) + n

    def snapshot(self) -> dict[str, int]:
        """Totals so far, to be handed to :meth:`since`."""
        return dict(self.counts) if self.enabled else {}

    def since(self, before: dict[str, int]) -> dict[str, int]:
        """What accrued since ``before``, omitting counters that did not move.

        Omitting zeros is what keeps a pass that touches no IR (an FX-graph
        pass, say) from carrying five empty keys in every record.
        """
        if not self.enabled:
            return {}
        deltas: dict[str, int] = {}
        for name, total in self.counts.items():
            delta = total - before.get(name, 0)
            if delta:
                deltas[name] = delta
        return deltas

    def reset(self) -> None:
        self.counts = {}


COUNTERS = PassCounters()


@contextlib.contextmanager
def counting(install: bool = True) -> Iterator[PassCounters]:
    """Count analysis calls for the duration, then restore what was patched.

    ``install=False`` is the disabled path and does nothing at all -- no
    wrapper, so an uninstrumented compile pays nothing rather than paying a
    branch per call. Nesting is safe: an inner region sees the wrapper already
    installed and leaves it alone, since wrapping twice would double every
    count.

    The outermost region clears the totals on the way in and on the way out, so
    counts exist only inside a region and read as "this compile" rather than
    "this process". Nothing needs them afterwards: the pass loops take their
    deltas while the region is still open.
    """
    if not install or COUNTERS.enabled:
        yield COUNTERS
        return

    original = ComputedBuffer.get_read_writes

    @functools.wraps(original)
    def counted(self: ComputedBuffer, *args: Any, **kwargs: Any) -> Any:
        COUNTERS.bump(READ_WRITES_EXTRACTIONS)
        start = time.perf_counter_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            COUNTERS.bump(READ_WRITES_EXTRACT_NS, time.perf_counter_ns() - start)

    ComputedBuffer.get_read_writes = counted  # type: ignore[method-assign]
    COUNTERS.reset()
    COUNTERS.enabled = True
    try:
        yield COUNTERS
    finally:
        COUNTERS.enabled = False
        COUNTERS.reset()
        ComputedBuffer.get_read_writes = original  # type: ignore[method-assign]


@contextlib.contextmanager
def counted_region() -> Iterator[dict[str, int]]:
    """Yield a dict that, on exit, holds the counts accrued inside the region.

    Installs the wrappers if nothing has already, and reports a delta rather
    than absolute totals -- so a test reads its own region's numbers even when
    an enclosing compile is counting too, which absolute totals would silently
    pollute. This is the same snapshot-and-subtract the pass loops do.
    """
    deltas: dict[str, int] = {}
    with counting():
        before = COUNTERS.snapshot()
        try:
            yield deltas
        finally:
            deltas.update(COUNTERS.since(before))


__all__ = [
    "COUNTERS",
    "DEVICE_COORDINATES",
    "HOST_COORDINATES",
    "PassCounters",
    "READ_WRITES_EXTRACTIONS",
    "READ_WRITES_EXTRACT_NS",
    "READ_WRITES_MISSES",
    "READ_WRITES_REQUESTS",
    "counted_region",
    "counting",
]
