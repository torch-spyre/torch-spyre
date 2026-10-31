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

"""``LoopNode``: a counted loop held as one operation in ``graph.operations``.

``splice_while_loops`` flattens each ``for_each_tile`` loop into ordinary ops
stamped with ``loop_info``.  Its final step, :func:`collapse`, wraps each loop's
ops into a ``LoopNode`` (nested for a nested loop), so a loop is a structural
unit from the first pass after the splice.  ``loop_info`` stays on the leaf
ops: the node adds structure and the per-loop facts (induction variable, trip
count, carries) but replaces nothing yet.

Passes that know nothing about nodes run against :class:`OpsView`, which
iterates the leaf ops in program order and forwards every mutation to the list
that owns the op.  At the end of pre-scheduling :func:`expand` flattens the
nodes away again, so the cost model, the scheduler and codegen see exactly
the flat list they saw before nodes existed.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterable, Iterator, Mapping, MutableSequence
from typing import Any

import sympy
from torch._inductor import dependencies, ir
from torch.utils._ordered_set import OrderedSet

from .loop_facts import counted_loop_path

# Set to False only by the no-behaviour-change test, to compare the pipeline
# with and without nodes.  With it off, ``collapse`` builds nothing, so no
# pass sees a view and ``expand`` has nothing to do.
BUILD_LOOP_NODES = True


class LoopNode(ir.Operation):
    """A counted loop: its body ops in execution order, plus per-loop facts.

    ``body`` holds ordinary operations and nested ``LoopNode``s.  Body ops keep
    their own names and stay in the graph's one buffer namespace; the node is
    not registered in ``graph.name_to_op``.

    ``origin_node``, ``origins``, ``stream_idx`` and ``mempool`` exist from
    construction because ``ir.Operation.get_origin_node`` / ``get_origins`` /
    ``get_stream_idx`` assert the attribute exists, and the scheduler asks.
    """

    def __init__(
        self,
        name: str,
        body: Iterable[ir.Operation],
        *,
        induction: sympy.Symbol | None,
        trip_count: Any,
        carries: Iterable[Any] = (),
        origin_node: Any = None,
        origins: OrderedSet[Any] | None = None,
    ) -> None:
        # ir.Operation is a dataclass; its __post_init__ is not run by a
        # subclass's own __init__, so set what it would set.
        self._config_patches: dict[str, Any] = {}
        self.operation_name = name
        self.body: list[ir.Operation] = list(body)
        self.induction = induction
        self.trip_count = trip_count
        self.carries: tuple[Any, ...] = tuple(carries)
        self.origin_node = origin_node
        self.origins: OrderedSet[Any] = (
            OrderedSet(origins) if origins is not None else OrderedSet()
        )
        self.stream_idx: int | None = None
        self.mempool: Any = None

    # ir.Operation's dataclass __eq__ compares (empty) field tuples, so any two
    # LoopNodes would compare equal and ``list.remove`` could take the wrong
    # one.  A node is identified by identity.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def get_name(self) -> str:
        assert self.operation_name is not None
        return self.operation_name

    def get_operation_name(self) -> str:
        assert self.operation_name is not None
        return self.operation_name

    @property
    def name(self) -> str:
        return self.get_name()

    def get_device(self):
        for op in all_ops(self.body):
            return op.get_device()
        return None

    def get_outputs(self) -> list[ir.Buffer]:
        outputs: list[ir.Buffer] = []
        for op in self.body:
            outputs.extend(op.get_outputs())
        return outputs

    def is_extern(self) -> bool:
        return False

    def get_origins(self) -> OrderedSet[Any]:
        return self.origins

    def mutation_targets(self) -> list[str]:
        """Names of the buffers the body's ``MutationLayoutSHOULDREMOVE`` ops write."""
        from .deadcode_elimination import _mutation_target

        targets: list[str] = []
        for op in all_ops(self.body):
            target = _mutation_target(op)
            if target is not None and target not in targets:
                targets.append(target)
        return targets

    def get_read_writes(self) -> dependencies.ReadWrites:
        """The plain union of the body's reads and writes, plus one ``StarDep``
        write per body mutation target, so a pass that judges liveness by
        writes sees that the loop writes into storage it does not own."""
        reads: OrderedSet[dependencies.Dep] = OrderedSet()
        writes: OrderedSet[dependencies.Dep] = OrderedSet()
        for op in self.body:
            rw = op.get_read_writes()
            reads |= OrderedSet(rw.reads)
            writes |= OrderedSet(rw.writes)
        for target in self.mutation_targets():
            writes.add(dependencies.StarDep(target))
        return dependencies.ReadWrites(reads, writes, OrderedSet())

    def has_side_effects(self) -> bool:
        """True when any leaf op of the body has a side effect (a mutation or
        an impure fallback), by ``deadcode_elimination``'s own predicate."""
        from .deadcode_elimination import _has_side_effects

        return any(_has_side_effects(op) for op in all_ops(self.body))

    def __repr__(self) -> str:
        return (
            f"LoopNode({self.operation_name}, trip_count={self.trip_count}, "
            f"induction={self.induction}, {len(self.body)} body entries)"
        )


def all_ops(operations: Iterable[ir.Operation]) -> Iterator[ir.Operation]:
    """Every leaf (non-node) op in program order, descending into ``LoopNode``s."""
    for op in operations:
        if isinstance(op, LoopNode):
            yield from all_ops(op.body)
        else:
            yield op


def has_loop_nodes(operations: Iterable[ir.Operation]) -> bool:
    return any(isinstance(op, LoopNode) for op in operations)


def owning_list(
    operations: list[ir.Operation], op: ir.Operation
) -> tuple[list[ir.Operation], int] | None:
    """``(list, index)`` of the list that directly holds ``op`` (by identity),
    searching nested nodes; ``None`` if ``op`` is nowhere in ``operations``."""
    for i, entry in enumerate(operations):
        if entry is op:
            return operations, i
        if isinstance(entry, LoopNode):
            found = owning_list(entry.body, op)
            if found is not None:
                return found
    return None


class OpsView(MutableSequence):
    """A flat read view over a nested op list; mutations go to the owning list.

    Iteration, ``len``, indexing and ``index`` see the leaf ops in program
    order, exactly the list a pass saw before nodes existed.  ``insert``,
    ``remove``, item assignment and ``del`` act on the list that directly holds
    the op at that flat position, so a new op placed beside an op in a body
    joins that body.  ``append`` and an insert at the end go to the top level.
    """

    def __init__(self, real: list[ir.Operation]) -> None:
        self.real = real
        self._cache: list[ir.Operation] | None = None

    def _flat(self) -> list[ir.Operation]:
        # Cached between edits: passes index the list in loops.  Every edit
        # goes through this view (no pass knows about nodes), and each
        # mutating method below drops the cache.
        if self._cache is None:
            self._cache = list(all_ops(self.real))
        return self._cache

    def _changed(self) -> None:
        self._cache = None

    def _locate(self, op: ir.Operation) -> tuple[list[ir.Operation], int]:
        found = owning_list(self.real, op)
        if found is None:
            raise ValueError(f"{op!r} is not in the operations list")
        return found

    def __iter__(self) -> Iterator[ir.Operation]:
        # By position, re-reading the length each step, as a list iterator
        # does: a pass that edits the list while iterating it sees exactly
        # what it saw on a flat list.
        i = 0
        while i < len(self):
            yield self._flat()[i]
            i += 1

    def __reversed__(self) -> Iterator[ir.Operation]:
        i = len(self) - 1
        while 0 <= i < len(self):
            yield self._flat()[i]
            i -= 1

    def __len__(self) -> int:
        return len(self._flat())

    def __contains__(self, op: object) -> bool:
        return op in self._flat()

    def __getitem__(self, i):
        return self._flat()[i]

    def index(self, op, *args) -> int:
        return self._flat().index(op, *args)

    def append(self, op: ir.Operation) -> None:
        # GraphLowering.register_operation appends newly lowered ops at the
        # end; the pass that lowered them then relocates them.
        self._changed()
        self.real.append(op)

    def insert(self, i: int, op: ir.Operation) -> None:
        flat = self._flat()
        n = len(flat)
        if i < 0:
            i = max(i + n, 0)
        self._changed()
        if i >= n:
            self.real.append(op)
            return
        lst, j = self._locate(flat[i])
        lst.insert(j, op)

    def remove(self, op: ir.Operation) -> None:
        # list.remove semantics (first entry that ``is`` or ``==`` op), then
        # remove that exact object from the list that owns it.
        flat = self._flat()
        found = flat[flat.index(op)]
        lst, j = self._locate(found)
        self._changed()
        del lst[j]

    def __setitem__(self, i, op) -> None:
        if isinstance(i, slice):
            self._reorder(i, op)
            return
        old = self._flat()[i]
        lst, j = self._locate(old)
        self._changed()
        lst[j] = op

    def __delitem__(self, i) -> None:
        if isinstance(i, slice):
            raise NotImplementedError("OpsView does not support slice deletion")
        lst, j = self._locate(self._flat()[i])
        self._changed()
        del lst[j]

    def _reorder(self, s: slice, new_flat: Iterable[ir.Operation]) -> None:
        """``view[:] = reordered``: the one wholesale rewrite after the splice
        (``dedup_and_promote_constants``).

        The view only receives a flat order, so it cannot tell whether a pass
        meant to move an op across a node boundary.  It therefore keeps every
        op in the list that holds it and only permutes ops WITHIN each list;
        nodes keep their place among the ops around them.  A pass that really
        wants to hoist or sink an op across a boundary needs a different API:
        this is the one place where the flat view hides a decision, and where
        the expanded order can differ from what the same rewrite would give on
        a flat list (a reordered op that sits in a body).
        """
        if s != slice(None):
            raise NotImplementedError("only view[:] = ... is supported")
        new_flat = list(new_flat)
        rank = {id(o): k for k, o in enumerate(new_flat)}
        owned = [id(o) for o in all_ops(self.real)]
        if len(new_flat) != len(owned) or set(rank) != set(owned):
            raise RuntimeError("a wholesale rewrite of the view added or dropped ops")

        def first_rank(entry: ir.Operation) -> int:
            # A node sorts by its earliest body op, so it keeps its position
            # relative to the ops around it.
            if isinstance(entry, LoopNode):
                return min(first_rank(x) for x in entry.body)
            return rank[id(entry)]

        def settle(lst: list[ir.Operation]) -> None:
            for entry in lst:
                if isinstance(entry, LoopNode):
                    settle(entry.body)
            lst.sort(key=first_rank)

        self._changed()
        settle(self.real)


_BUILDING_LOOP: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "torch_spyre_building_loop", default=False
)


@contextlib.contextmanager
def building_loop() -> Iterator[None]:
    """Mark code that builds or takes apart loops.

    The graph is legitimately inconsistent while a loop is half built, so the
    insertion API (and later the loop checker) relax their checks while this
    is set.  One flag per context (thread); nesting is allowed.
    """
    token = _BUILDING_LOOP.set(True)
    try:
        yield
    finally:
        _BUILDING_LOOP.reset(token)


def in_building_loop() -> bool:
    return _BUILDING_LOOP.get()


def _group_path(op: ir.Operation) -> tuple[int, ...]:
    # An ExternKernel keeps its loop_info (e.g. a hoisted loop-body constant)
    # but runs once, outside the loop, so it is not a member of any loop.
    return counted_loop_path(op)


def _carries_of(body: list[ir.Operation]) -> tuple[Any, ...]:
    """``_loop_carry_record``s on the ops directly in ``body``, deduplicated."""
    records: list[Any] = []
    for op in body:
        record = getattr(op, "_loop_carry_record", None)
        if record is not None and not any(r is record for r in records):
            records.append(record)
    return tuple(records)


def _wrap(
    ops: list[ir.Operation],
    depth: int,
    prefix: tuple[int, ...],
    inductions: Mapping[int, sympy.Symbol],
    counter: list[int],
) -> list[ir.Operation]:
    """Wrap runs of ``ops`` sharing a ``loop_group_id`` prefix of length
    ``depth + 1`` (and extending ``prefix``) into ``LoopNode``s, recursively.

    A node's body is the span from its first member to its last member.  An
    op that sits between two members but is not one (no ``loop_info``, an
    ``ExternKernel``, or an op of another loop) is kept in that span, where it
    is: moving it would change the program order, and :func:`expand` must give
    back exactly the list it was given.  Inside the node it is an ordinary
    body entry; it is wrapped further only if its own path extends the node's.
    """
    out: list[ir.Operation] = []
    i = 0
    while i < len(ops):
        path = _group_path(ops[i])
        if len(path) <= depth or path[:depth] != prefix:
            out.append(ops[i])
            i += 1
            continue
        key = path[: depth + 1]
        last = max(
            k for k in range(i, len(ops)) if _group_path(ops[k])[: depth + 1] == key
        )
        run = ops[i : last + 1]
        group_idx = key[-1]
        node = LoopNode(
            f"loopnode{counter[0]}",
            (),
            induction=inductions.get(group_idx),
            trip_count=ops[i].loop_info.loop_count[depth],  # type: ignore[attr-defined]
        )
        counter[0] += 1
        node.body = _wrap(run, depth + 1, key, inductions, counter)
        node.carries = _carries_of(node.body)
        out.append(node)
        i = last + 1
    return out


def collapse(
    graph: Any, *, inductions: Mapping[int, sympy.Symbol] | None = None
) -> None:
    """Wrap each counted loop's ops in ``graph.operations`` into ``LoopNode``s.

    ``inductions`` maps a level's group index (an entry of ``loop_group_id``)
    to its loop variable; a level missing from it gets ``induction=None``.
    The list is rewritten in place, so its identity is kept.
    """
    if not BUILD_LOOP_NODES:
        return
    operations = graph.operations
    assert not isinstance(operations, OpsView), "collapse needs the real list"
    if not any(_group_path(op) for op in operations):
        return
    with building_loop():
        operations[:] = _wrap(list(operations), 0, (), inductions or {}, [0])


def expand(graph: Any) -> None:
    """Flatten every ``LoopNode`` in ``graph.operations`` back to its leaf ops,
    in program order, in place.  The inverse of :func:`collapse`."""
    operations = graph.operations
    assert not isinstance(operations, OpsView), "expand needs the real list"
    if not has_loop_nodes(operations):
        return
    with building_loop():
        operations[:] = list(all_ops(operations))


def _inductions_of(
    operations: Iterable[ir.Operation],
    depth: int = 0,
    out: dict[int, sympy.Symbol] | None = None,
) -> dict[int, sympy.Symbol]:
    """Group index -> induction of every node in ``operations``, recovered from
    the first body member of each node, exactly as :func:`collapse` keyed it."""
    out = {} if out is None else out
    for op in operations:
        if not isinstance(op, LoopNode):
            continue
        if op.induction is not None:
            for leaf in all_ops(op.body):
                path = _group_path(leaf)
                if len(path) > depth:
                    out[path[depth]] = op.induction
                    break
        _inductions_of(op.body, depth + 1, out)
    return out


@contextlib.contextmanager
def expanded(graph: Any) -> Iterator[None]:
    """Run a block with the nodes expanded to today's flat list, then rebuild
    them from the ops' ``loop_group_id`` (keeping each loop's induction).

    For a pass whose edit the flat view cannot express: a wholesale reorder
    that moves an op across a node boundary (see ``OpsView._reorder``).
    """
    operations = graph.operations
    if not has_loop_nodes(operations):
        yield
        return
    inductions = _inductions_of(operations)
    expand(graph)
    try:
        yield
    finally:
        assert graph.operations is operations, "the pass rebound graph.operations"
        collapse(graph, inductions=inductions)
