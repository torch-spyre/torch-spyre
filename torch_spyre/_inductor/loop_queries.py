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

"""``LoopProgram``: where an op runs, how often, and which loops it shares.

Three relations are kept apart:

* *placement*: the ``LoopNode`` bodies an op sits in (the tree);
* *execution*: how the op actually runs. An op placed in a body runs every trip of
  each node it sits in. That includes a non-member LX clone in a body and an
  allocation (a zero-element buffer, ``is_no_op()``): with pooled allocation an
  allocation is a static address reservation, so modelling it as happening every
  trip, where it is placed, is exact. The one exception today is an extern kernel
  in a body, which the loop scheduler moves before its outermost loop because
  Spyre's codegen cannot run one inside a counted loop;
* *membership*: whether a node holds a tiling entry for the op (what it tiles, how
  its accesses advance). This stays ``loop_facts``'s domain (``enclosing_loops``,
  ``counted_loop_path``) and is not what these queries answer.

Every query here is about execution. The program is the leaf ops in program order,
numbered from 0: an op's *slot*. Everything is computed once, in ``__init__``, from
one walk; nothing is cached across edits and nothing is stored on the IR ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy
from torch._inductor.ir import ExternKernel, Operation

from .loop_facts import _in_built_node, pending_levels
from .loop_node import LoopNode, OpsView
from .pass_utils import op_read_writes, trip_count_estimate


@dataclass(eq=False, frozen=True)
class Loop:
    """One loop of a program. A handle: compare by identity.

    ``node`` is ``None`` for a pending loop (lowered, not built into a node yet),
    which is identified by ``path``, its group path.
    """

    node: LoopNode | None
    path: tuple[int, ...]
    trip_count: Any


def _runs_once(op: Operation) -> bool:
    """An extern kernel runs once, before its outermost loop; an allocation
    (``is_no_op()``) does not."""
    if not isinstance(op, ExternKernel):
        return False
    is_no_op = getattr(op, "is_no_op", None)
    return not (is_no_op and is_no_op())


class _Leaf:
    """What the walk records per leaf; later queries add fields here."""

    __slots__ = ("op", "slot", "placement", "loops")

    def __init__(self, op, slot, placement, loops):
        self.op = op
        self.slot = slot
        self.placement = placement  # enclosing node loops, outermost first
        self.loops = loops  # execution loops, outermost first


class _BufferIndex:
    """Per-buffer accesses, built once from the program on the first buffer query."""

    __slots__ = ("uses", "readers", "intervals")

    def __init__(self) -> None:
        self.uses: dict[str, list[int]] = {}  # buffer -> sorted distinct times
        self.readers: dict[str, list[Operation]] = {}  # each reading op once
        self.intervals: dict[str, tuple[int, int] | None] = {}


class LoopProgram:
    """The loop structure of a graph's operations, answered per op."""

    def __init__(self, source) -> None:
        self._input_names: tuple[str, ...] = tuple(
            getattr(source, "graph_input_names", ())
        )
        self._index: _BufferIndex | None = None
        if hasattr(source, "operations"):
            source = source.operations
        if isinstance(source, OpsView):
            source = source.real
        self._leaves: list[_Leaf] = []
        self._by_id: dict[int, _Leaf] = {}
        self._node_loops: dict[int, Loop] = {}
        self._pending: dict[tuple[int, ...], Loop] = {}
        self._span: dict[int, list[int]] = {}  # id(loop) -> [first, last]
        self._walk(list(source), ())

    def _record(self, loop: Loop, slot: int) -> None:
        s = self._span.get(id(loop))
        if s is None:
            self._span[id(loop)] = [slot, slot]
        else:
            s[1] = slot

    def _walk(self, entries, stack: tuple[Loop, ...]) -> None:
        for op in entries:
            if isinstance(op, LoopNode):
                loop = Loop(op, (), op.trip_count)
                self._node_loops[id(op)] = loop
                self._walk(op.body, stack + (loop,))
                continue
            slot = len(self._leaves)
            for loop in stack:
                self._record(loop, slot)
            if _runs_once(op):
                loops: tuple[Loop, ...] = ()
            elif stack:
                loops = stack
            elif not _in_built_node(op) and (levels := pending_levels(op)):
                made = []
                for level in levels:
                    path = tuple(level.group_path)
                    pending = self._pending.get(path)
                    if pending is None:
                        pending = Loop(None, path, level.trip_count)
                        self._pending[path] = pending
                    self._record(pending, slot)
                    made.append(pending)
                loops = tuple(made)
            else:
                loops = ()
            leaf = _Leaf(op, slot, stack, loops)
            self._leaves.append(leaf)
            self._by_id[id(op)] = leaf

    def _leaf(self, op: Operation) -> _Leaf:
        leaf = self._by_id.get(id(op))
        if leaf is None:
            raise KeyError(op.get_name())
        return leaf

    def ops(self) -> list[Operation]:
        return [leaf.op for leaf in self._leaves]

    def slot(self, op: Operation) -> int:
        return self._leaf(op).slot

    def loops_of(self, op: Operation) -> tuple[Loop, ...]:
        return self._leaf(op).loops

    def outermost(self, op: Operation) -> Loop | None:
        loops = self._leaf(op).loops
        return loops[0] if loops else None

    def span(self, loop: Loop) -> tuple[int, int] | None:
        s = self._span.get(id(loop))
        return (s[0], s[1]) if s else None

    def executions(self, op: Operation) -> sympy.Expr | int:
        loops = self._leaf(op).loops
        if not loops:
            return 1
        return sympy.Mul(*(loop.trip_count for loop in loops))

    def executions_estimate(self, op: Operation) -> int:
        n = 1
        for loop in self._leaf(op).loops:
            n *= trip_count_estimate(loop.trip_count)
        return n

    def shared_loops(self, a: Operation, b: Operation) -> tuple[Loop, ...]:
        shared = []
        for x, y in zip(self.loops_of(a), self.loops_of(b)):
            if x is not y:
                break
            shared.append(x)
        return tuple(shared)

    def same_iteration(self, a: Operation, b: Operation) -> bool:
        return self.loops_of(a) == self.loops_of(b)

    def time(self, op: Operation) -> int:
        """The slot ``op`` takes effect at: its own, except for an extern kernel in
        a body, which runs before its outermost placement loop."""
        leaf = self._leaf(op)
        if leaf.placement and _runs_once(op):
            return self._span[id(leaf.placement[0])][0]
        return leaf.slot

    def loop_for(self, node: LoopNode) -> Loop:
        loop = self._node_loops.get(id(node))
        if loop is None:
            raise KeyError(node.get_name())
        return loop

    # Buffer queries. The index is built lazily on the first one; it is a function
    # of this program object only, so nothing outlives an edit of the graph.

    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    def _buffers(self) -> _BufferIndex:
        if self._index is not None:
            return self._index
        index = _BufferIndex()
        first: dict[str, tuple[int, Operation, bool]] = {}  # time, op, is a write
        times: dict[str, set[int]] = {}
        for leaf in self._leaves:
            op = leaf.op
            t = self.time(op)
            rw = op_read_writes(op)
            written = {d.name for d in rw.writes}
            read = {d.name for d in rw.reads}
            for name in written | read:
                times.setdefault(name, set()).add(t)
                # A write wins a tie: an op that updates a buffer in place makes it.
                cur = first.get(name)
                if cur is None or t < cur[0]:
                    first[name] = (t, op, name in written)
            for name in read:
                index.readers.setdefault(name, []).append(op)
        for name in self._input_names:
            times.setdefault(name, set())
        index.uses = {name: sorted(ts) for name, ts in times.items()}
        carried: dict[str, list[Loop]] = {}
        for loop in self._node_loops.values():
            assert loop.node is not None
            for carry in loop.node.carries:
                carried.setdefault(carry.storage_name, []).append(loop)
        for name, ts in index.uses.items():
            if not ts:
                index.intervals[name] = None
                continue
            lo, hi = ts[0], ts[-1]
            _, writer, wrote = first[name]
            # A graph input is born outside every loop, whoever writes it later.
            born_in = wrote and name not in self._input_names
            birth = self._leaf(writer).loops if born_in else ()
            widen = list(carried.get(name, ()))
            for op in index.readers.get(name, ()):
                for loop in self._leaf(op).loops:
                    if not any(loop is b for b in birth):
                        widen.append(loop)
                        break  # outermost loop that holds the read, not the birth
            for loop in widen:
                span = self.span(loop)
                if span is not None:
                    lo, hi = min(lo, span[0]), max(hi, span[1])
            index.intervals[name] = (lo, hi)
        self._index = index
        return index

    def uses(self, buffer: str) -> list[int]:
        """Sorted distinct ``time`` of the ops that read or write ``buffer``."""
        return list(self._buffers().uses.get(buffer, ()))

    def runtime_reads(self, buffer: str) -> int:
        """Sum of ``executions_estimate`` over the ops that read ``buffer``."""
        return sum(
            self.executions_estimate(op)
            for op in self._buffers().readers.get(buffer, ())
        )

    def live_interval(self, buffer: str) -> tuple[int, int] | None:
        """Inclusive slots ``buffer`` is live over; ``None`` when it is unused.

        The uses' range, widened for a read to the span of the outermost loop that
        holds the read but not the buffer's birth (a value read again every trip
        must survive the back edge), and for a carry to its node's span."""
        return self._buffers().intervals.get(buffer)
