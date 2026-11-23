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

Passes that read the graph flat run against :class:`OpsView`, which iterates
the leaf ops in program order and refuses edits: a pass that changes the graph
names the position with the helpers below (``insert_before``, ``move_after``,
``remove_op``, ...), each of which finds the list that owns the anchor.
After the pass loop the nodes stay: each reaches Inductor's scheduler as one
``LoopSchedulerNode`` (``loop_scheduler.py``), and the readers in between
descend into them.  :func:`expand` flattens them only on a pass error and in
the equivalence tests that compare against the flat path.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterable, Iterator, Mapping, MutableSequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy
from torch._inductor import dependencies, ir
from torch.utils._ordered_set import OrderedSet

from .loop_facts import counted_loop_path

if TYPE_CHECKING:
    import torch.fx


@dataclass(frozen=True)
class LoopCarryRecord:
    """Identity contract for one accumulator carried through a counted loop.

    ``for_each_tile`` lowering declares the carry with this record;
    ``loop_carries.apply_loop_carries`` then rewrites the update as an
    in-place write into the carry's initial storage.  Recording both ends
    lets scratchpad planning distinguish that closed, compiler-created
    mutation from an arbitrary user mutation, which must remain in HBM.

    ``loop_origin`` is the exact FX ``while_loop`` HOP node the carry belonged
    to, captured and validated at the splice site (see
    ``wsr/while_loop_bridge._validated_loop_origin``) before the WhileLoop IR
    node is deleted.  It is optional internal compiler metadata: ``None``
    whenever the origin could not be proven (nested loops, loops inside an
    ``invoke_subgraph``, or a chained carry whose origins gather an earlier
    loop's node), in which case the post-loop materialization plan declines
    and the carry keeps today's HBM behavior.  No other consumer of this
    record reads the field.

    ``storage_view`` is the view the update writes through when the carry's
    initial value is a view of the storage (a transposed init), else None.
    ``loop_carries.apply_loop_carries`` uses it as the mutation target, as the
    splice did when it made the write in place itself.
    """

    storage_name: str
    update_name: str
    loop_origin: "torch.fx.Node | None" = None
    storage_view: Any = field(default=None, compare=False, repr=False)


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
        group_id: int | None = None,
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
        # This loop's index among its siblings (its entry in a member's
        # ``loop_group_id``), and for each leaf op of the body, by name, what that
        # op tiles at this level: ``(tiled_dims, tiled_reduction_dims)``. A tiled dim
        # is not recoverable from the op's own index (the range is already divided
        # and an access need not move with the loop), so the lowering that divided
        # it records it here.
        # INVARIANT (stage 9.3a): ``tiling`` is written only by ``_fill_tiling`` /
        # ``record_loop_info`` (from the op's ``loop_info``), and each leaf op's
        # ``_loop_chain`` only by ``_set_chain``, which the node builder (``_wrap``),
        # the edit helpers below and ``expand`` call. A new edit site calls one of
        # those helpers; it never assigns either directly.
        self.group_id = group_id
        # False on a node built by hand (unit tests), whose facts nothing recorded.
        self.facts_recorded = False
        self.tiling: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
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
    """A flat READ-ONLY view over a nested op list.

    Iteration, ``len``, indexing and ``index`` see the leaf ops in program
    order, exactly the list a pass saw before nodes existed.  ``append`` (what
    ``GraphLowering.register_operation`` does with a newly lowered op) goes to
    the top level and ``hoist_to_front`` moves ops to the head of the graph.
    ``insert``, ``remove``, item assignment and ``del`` raise ``TypeError``: a
    pass that edits the graph says where an op goes with ``insert_before`` /
    ``insert_after`` / ``move_before`` / ``move_after`` / ``remove_op`` /
    ``replace_op``.
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

    # The view is read-only apart from ``append`` and ``hoist_to_front``. "Next to
    # this flat neighbour" does not say whether an op belongs inside a loop body or
    # beside the node, so a pass that edits the graph names where the op goes with
    # ``insert_before`` / ``insert_after`` / ``move_before`` / ``move_after`` /
    # ``remove_op`` / ``replace_op`` (below). The four list mutators exist only
    # because ``MutableSequence`` requires them; they fail loudly.
    def _refuse(self, what: str) -> TypeError:
        return TypeError(
            f"OpsView is a read-only view: {what} would place the op by its flat "
            "neighbour. Use insert_before / insert_after / move_before / move_after "
            "/ remove_op / replace_op from torch_spyre._inductor.loop_node."
        )

    def insert(self, i: int, op: ir.Operation) -> None:
        raise self._refuse("insert")

    def remove(self, op: ir.Operation) -> None:
        raise self._refuse("remove")

    def __setitem__(self, i, op) -> None:
        raise self._refuse("item assignment")

    def __delitem__(self, i) -> None:
        raise self._refuse("del")

    def hoist_to_front(self, ops: Iterable[ir.Operation]) -> None:
        """Move ``ops``, in the order given, to the head of the TOP-LEVEL list,
        taking each out of whichever list holds it (a loop body included).

        This is the one edit that crosses a node boundary on purpose, which
        ``view[:] = ...`` cannot express: it only permutes ops within the list
        that holds them. ``dedup_and_promote_constants`` uses it to take
        loop-invariant constants out of their loops. Drops the flat cache.
        """
        ops = list(ops)
        for op in ops:
            lst, i = self._locate(op)
            lst.pop(i)
        self.real[:0] = ops
        for op in ops:
            _set_chain(op, ())
        self._changed()


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


# ---------------------------------------------------------------------------
# Insertion points
# ---------------------------------------------------------------------------
#
# A pass that adds an op to a loop says WHERE, in terms of the node, and the op
# becomes a member (or not) because of where it is. The functions below act on
# the REAL nested list, not on OpsView: OpsView.insert places an op beside the
# op at a flat index, so "just before the loop" lands inside the body.
#
# The plan names six points; they are these four calls. Before and after "the
# loop" and before and after "a given node" are the same call on a
# :class:`LoopNode` anchor; "start of each iteration" and "the back edge" are
# ``insert_at_start`` and ``insert_on_back_edge``. An op that carries its own
# index expression is the caller's: it may use the variable of any enclosing
# loop, and ``loop_checker`` rule 2 verifies that between passes.
#
# Passes are handed a flat :class:`OpsView`, and these functions accept it (and
# drop its cached order): that is how a pass says WHERE an op belongs without
# going through ``view.insert``, which puts the op beside whatever op sits at a
# flat position. The placement rule is the anchor's:
#
#   * ``insert_before(view, node, op)`` / ``insert_after(view, node, op)``: the op
#     is outside the loop, in the list that holds the node, and runs once;
#   * ``insert_before(view, body_op, op)`` / ``insert_after(view, body_op, op)``:
#     the op is in that body and runs every trip;
#   * ``move_before`` / ``move_after`` take an op that is already in the graph (a
#     freshly lowered op was appended at the top level by
#     ``GraphLowering.register_operation``) out of its list and apply the same rule;
#   * ``remove_op`` removes an op from whichever list holds it.
#
# ``OpsView.append`` stays for the one caller that cannot choose, op registration.


def _real_list(operations: "list[ir.Operation] | OpsView") -> list[ir.Operation]:
    """The nested list behind ``operations``, with a view's cached order dropped
    (the caller is about to edit the lists it holds)."""
    if isinstance(operations, OpsView):
        operations._changed()
        return operations.real
    return operations


def _enclosing_chain(
    operations: list[ir.Operation], target: ir.Operation
) -> list[LoopNode] | None:
    """Nodes holding ``target``, outermost first; ``None`` if it is not there."""
    for entry in operations:
        if entry is target:
            return []
        if isinstance(entry, LoopNode):
            inner = _enclosing_chain(entry.body, target)
            if inner is not None:
                return [entry, *inner]
    return None


def _holder(
    operations: list[ir.Operation], anchor: ir.Operation
) -> tuple[list[ir.Operation], int]:
    found = owning_list(operations, anchor)
    if found is None:
        raise ValueError(f"{anchor!r} is not in the operations list")
    return found


def _set_chain(entry: ir.Operation, chain: tuple["LoopNode", ...]) -> None:
    """Point every leaf op under ``entry`` at its enclosing nodes, outermost first
    (``chain`` is what encloses ``entry`` itself)."""
    if isinstance(entry, LoopNode):
        inner = (*chain, entry)
        for child in entry.body:
            _set_chain(child, inner)
    else:
        entry._loop_chain = chain  # type: ignore[attr-defined]


def record_loop_info(op: ir.Operation) -> None:
    """``op.loop_info`` was (re)written after the op was placed: make the nodes
    around it hold the new tiled dims. A pass that stamps an op it has already
    put in a body calls this right after the stamp (the loop checker, rule 6,
    reports a node that disagrees with the stamp)."""
    _fill_tiling(op, overwrite=True)


def _fill_tiling(entry: ir.Operation, overwrite: bool = False) -> None:
    """Give each node enclosing a leaf op of ``entry`` the op's tiled dims at that
    level, read from the op's ``loop_info``, unless the node already has them
    (``overwrite`` replaces them)."""
    for leaf in all_ops([entry]):
        info = getattr(leaf, "loop_info", None)
        if info is None:
            continue
        name = leaf.get_name()
        for level, node in enumerate(getattr(leaf, "_loop_chain", ())):
            if overwrite or name not in node.tiling:
                node.tiling[name] = (
                    _level_dims(info.loop_tiled_dims, level),
                    _level_dims(info.loop_tiled_reduction_dims, level),
                )


def _level_dims(lists: Any, level: int) -> tuple[int, ...]:
    return tuple(lists[level]) if level < len(lists) else ()


def _adopt(operations: list[ir.Operation], entry: ir.Operation) -> None:
    """``entry`` was just put in ``operations``: record which nodes enclose it."""
    chain = _enclosing_chain(operations, entry)
    assert chain is not None
    _set_chain(entry, tuple(chain))
    _fill_tiling(entry)


def insert_before(
    operations: "list[ir.Operation] | OpsView",
    anchor: ir.Operation,
    op: ir.Operation,
) -> None:
    """Insert ``op`` immediately before ``anchor``, in the list that holds it.

    ``anchor`` may be an ordinary op or a :class:`LoopNode`. Before a node, the
    op runs once, outside the loop; before a body op, it runs every trip.

    ``operations`` may be the flat :class:`OpsView` a pass is handed. Unlike
    ``view.insert(view.index(x), op)``, naming a node puts the op outside it even
    when the node's first member is the next flat op.
    """
    real = _real_list(operations)
    lst, i = _holder(real, anchor)
    lst.insert(i, op)
    _adopt(real, op)


def insert_after(
    operations: "list[ir.Operation] | OpsView",
    anchor: ir.Operation,
    op: ir.Operation,
) -> None:
    """Insert ``op`` immediately after ``anchor``, in the list that holds it.

    After a :class:`LoopNode` the op runs once, after the last trip (the
    allocator's post-loop drain). After a body op it runs every trip.

    ``operations`` may be the flat :class:`OpsView` a pass is handed: the op then
    goes next to its anchor in the nested list. ``view.insert`` cannot say that, and
    puts an op inserted past the last flat position at the top level, outside the
    loop it was meant for.
    """
    real = _real_list(operations)
    lst, i = _holder(real, anchor)
    lst.insert(i + 1, op)
    _adopt(real, op)


def remove_op(operations: "list[ir.Operation] | OpsView", op: ir.Operation) -> None:
    """Remove ``op`` (by identity) from whichever list holds it.

    ``op`` may be a :class:`LoopNode`. Removing a body's last member leaves the
    node, empty: nothing here decides what an empty node means. Raises
    ``ValueError`` if ``op`` is not in ``operations``.
    """
    _detach(operations, op)
    for leaf in all_ops([op]):
        for node in getattr(leaf, "_loop_chain", ()):
            node.tiling.pop(leaf.get_name(), None)
        leaf._loop_chain = ()  # type: ignore[attr-defined]


def _detach(operations: "list[ir.Operation] | OpsView", op: ir.Operation) -> None:
    """Take ``op`` out of the list that holds it, keeping its recorded facts (a
    move puts it back)."""
    lst, i = _holder(_real_list(operations), op)
    del lst[i]


def replace_op(
    operations: "list[ir.Operation] | OpsView", old: ir.Operation, new: ir.Operation
) -> None:
    """Put ``new`` where ``old`` is, in whichever list holds ``old``.

    Raises ``ValueError`` if ``old`` is not in ``operations``.
    """
    real = _real_list(operations)
    lst, i = _holder(real, old)
    lst[i] = new
    _set_chain(new, tuple(getattr(old, "_loop_chain", ())))
    # ``new`` may reuse ``old``'s name (replace_computed_buffer_body does) with a
    # different stamp; its stamp is the authority.
    _fill_tiling(new, overwrite=True)


def top_level_entry(
    operations: "list[ir.Operation] | OpsView", op: ir.Operation
) -> ir.Operation:
    """The entry of the top-level list that holds ``op``: ``op`` itself when it is
    there, otherwise the outermost :class:`LoopNode` it sits in (at any depth).

    An op that must run once, outside every loop of an anchor (a clone hoisted
    before a loop, a drain after one), is placed beside this entry, not beside the
    anchor, which may be the loop's first or last member. Raises ``ValueError`` if
    ``op`` is not in ``operations``.
    """
    real = operations.real if isinstance(operations, OpsView) else operations
    for entry in real:
        if entry is op or (
            isinstance(entry, LoopNode) and owning_list(entry.body, op) is not None
        ):
            return entry
    raise ValueError(f"{op!r} is not in the operations list")


def _move(
    operations: "list[ir.Operation] | OpsView",
    anchor: ir.Operation,
    op: ir.Operation,
    after: bool,
) -> None:
    if op is anchor:
        raise ValueError(f"cannot move {op!r} relative to itself")
    real = _real_list(operations)
    # Find both before touching either list, so a missing one changes nothing.
    _holder(real, op)
    _holder(real, anchor)
    if isinstance(op, LoopNode) and owning_list(op.body, anchor) is not None:
        # Taking the node out would take the anchor with it.
        raise ValueError(f"cannot move {op!r} relative to {anchor!r}, which it holds")
    _detach(real, op)
    (insert_after if after else insert_before)(real, anchor, op)


def move_before(
    operations: "list[ir.Operation] | OpsView",
    anchor: ir.Operation,
    op: ir.Operation,
) -> None:
    """Move ``op``, already in the graph, to immediately before ``anchor``.

    The usual case is a freshly lowered op (``GraphLowering.register_operation``
    appended it at the top level) that a pass relocates next to its consumer. The
    placement rule is :func:`insert_before`'s: before a node the op is outside the
    loop, before a body op it is in that body. ``op`` may come from any list, and
    may be a :class:`LoopNode` (it moves with its body) as long as ``anchor`` is not
    inside it.
    """
    _move(operations, anchor, op, after=False)


def move_after(
    operations: "list[ir.Operation] | OpsView",
    anchor: ir.Operation,
    op: ir.Operation,
) -> None:
    """Move ``op``, already in the graph, to immediately after ``anchor``
    (:func:`insert_after`'s placement rule)."""
    _move(operations, anchor, op, after=True)


def insert_at_start(node: LoopNode, op: ir.Operation) -> None:
    """Insert ``op`` as the first entry of ``node``'s body: it runs every trip,
    before the rest of the body."""
    node.body.insert(0, op)
    _set_chain(op, _chain_of(node))
    _fill_tiling(op)


def insert_on_back_edge(node: LoopNode, op: ir.Operation) -> None:
    """Insert ``op`` at the end of ``node``'s body: the point every trip ends at.

    The end of the body is after every read of the old value, which is the
    property a carry copy needs: placed right after its update, the copy
    overwrote the value before its last reader in a prototype (relative error
    0.785). The position is the guarantee, so there is nothing to search for
    and nothing to check; a caller that wants an op NEAR an update, not at the
    end, wants :func:`insert_after` and owns that choice.
    """
    node.body.append(op)
    _set_chain(op, _chain_of(node))
    _fill_tiling(op)


def _chain_of(node: LoopNode) -> tuple[LoopNode, ...]:
    """The nodes enclosing the body of ``node``, outermost first, ``node`` last.
    Read from a leaf of ``node`` that is already placed; ``(node,)`` if it has none."""
    for leaf in all_ops(node.body):
        chain = getattr(leaf, "_loop_chain", None)
        if chain and node in chain:
            return chain[: chain.index(node) + 1]
    return (node,)


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
    only: frozenset[int] | None = None,
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
        if (
            len(path) <= depth
            or path[:depth] != prefix
            or (only is not None and depth == 0 and path[0] not in only)
        ):
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
            group_id=group_idx,
            induction=inductions.get(group_idx),
            trip_count=ops[i].loop_info.loop_count[depth],  # type: ignore[attr-defined]
        )
        counter[0] += 1
        node.body = _wrap(run, depth + 1, key, inductions, counter, only)
        node.carries = _carries_of(node.body)
        out.append(node)
        i = last + 1
    return out


_PENDING_ATTR = "pending_loop_inductions"


def request_loop_nodes(graph: Any, inductions: Mapping[int, sympy.Symbol]) -> None:
    """Ask for the loops whose group index is in ``inductions`` to become nodes.

    A lowering that builds a loop in the middle of the pass pipeline records it
    here instead of wrapping it: the pass that called it may still be iterating
    ``graph.operations`` (a flat view, or the real list), and a node wrapped under
    it would change what the rest of the pass sees. The pass loop calls
    :func:`build_pending_loop_nodes` once the pass has returned.
    """
    if inductions:
        pending = getattr(graph, _PENDING_ATTR, None) or {}
        pending.update(inductions)
        setattr(graph, _PENDING_ATTR, pending)


def build_pending_loop_nodes(graph: Any) -> None:
    """Wrap the loops :func:`request_loop_nodes` named, leaving every other op
    (existing nodes, flat loops no lowering has converted yet) where it is.

    Called with the real list, after the pass that built the loops has returned.
    Node names continue after the nodes already present.
    """
    pending = getattr(graph, _PENDING_ATTR, None)
    if not pending:
        return
    setattr(graph, _PENDING_ATTR, None)
    if not BUILD_LOOP_NODES:
        return
    operations = graph.operations
    assert not isinstance(operations, OpsView), "needs the real list"
    existing = sum(1 for _ in _nodes_in(operations))
    with building_loop():
        operations[:] = _wrap(
            list(operations), 0, (), pending, [existing], frozenset(pending)
        )
        _record_node_facts(operations)


def _nodes_in(operations: Iterable[ir.Operation]) -> Iterator["LoopNode"]:
    for op in operations:
        if isinstance(op, LoopNode):
            yield op
            yield from _nodes_in(op.body)


def _record_node_facts(operations: list[ir.Operation]) -> None:
    """Fill every node's per-member tiling and every leaf's node chain, after the
    nodes were built from the ops' ``loop_info``."""
    for entry in operations:
        _set_chain(entry, ())
        _fill_tiling(entry)
    for node in _nodes_in(operations):
        node.facts_recorded = True


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
        _record_node_facts(operations)


def expand(graph: Any) -> None:
    """Flatten every ``LoopNode`` in ``graph.operations`` back to its leaf ops,
    in program order, in place.  The inverse of :func:`collapse`."""
    operations = graph.operations
    assert not isinstance(operations, OpsView), "expand needs the real list"
    if not has_loop_nodes(operations):
        return
    with building_loop():
        operations[:] = list(all_ops(operations))
        for op in operations:
            op._loop_chain = ()  # type: ignore[attr-defined]
