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

"""A ``LoopNode`` reaching Inductor's scheduler as ONE scheduler node.

``LoopSchedulerNode`` wraps the scheduler nodes of a loop's body, and four
per-compile patches on ``torch._inductor.scheduler.Scheduler`` let it take
part in the scheduler's front half (dependency analysis, the topological
sorts, dead-node elimination, ancestors) like any other node.  The patches are
applied by :func:`loop_scheduler_patches_for_compile`, which ``patches.py``
enters for every Spyre compile and leaves in the same ``finally`` as its other
per-compile patches, so a CPU compile in the same process is untouched.

``CustomPreSchedulingPasses`` hands the scheduler the real nested list (it no
longer expands the nodes), so on a compiled graph every ``LoopNode`` reaches
``Scheduler.__init__``.  A ``LoopSchedulerNode`` is a
``CountedLoopSchedulerNode``, so the pre-fusion pass
``build_loop_scheduler_nodes`` passes it through unchanged (its
``_loop_group_id`` is None) and everything after it (fusion, kernel
preparation, HBM-pool planning, codegen) sees the counted loop it always saw.
Once the scheduler nodes exist, ``graph.operations`` is the flat leaf list
again (the structure lives in the scheduler nodes): kernel preparation's LX
read contracts read it and expect leaf ops.  A kernel op inside a node that
is not a member of that loop (no matching ``loop_info``) is refused, since it
would run every trip here and once on the flat path.

The patch reaches into private Inductor code.  It is supported only on the
torch releases in ``SUPPORTED_RELEASES`` and is refused with ``Unsupported``
on any other: it is required for a loop to compile, so it neither warns nor
falls back.  ``PATCHED_SURFACE`` lists every piece of Inductor it replaces or
relies on, so a torch upgrade knows exactly what to re-verify.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence

import torch
from torch._inductor import dependencies, ir
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    NodeUser,
    NopKernelSchedulerNode,
    Scheduler,
    refresh_group_node_dependencies,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .errors import Unsupported
from .loop_facts import counted_loop_path
from .loop_node import LoopNode, expand
from .scheduler import CountedLoopSchedulerNode

# One torch release at a time: the one ``pyproject.toml`` pins (``torch~=2.14.0``).
# A torch upgrade moves this with the pin; the upgrade-pytorch-version skill has a step
# that re-verifies ``PATCHED_SURFACE`` and updates this tuple.  Keep it a literal, not a
# value parsed from package metadata: a patch whose failure mode is "refuse to compile a
# loop" should not depend on metadata lookups that can fail at import.
SUPPORTED_RELEASES: tuple[tuple[int, int], ...] = ((2, 14),)

# Set to False only by the equivalence test, to compare the flow-through path
# with the path it replaced: ``CustomPreSchedulingPasses`` then expands every
# ``LoopNode`` after the pass loop, as before, and ``build_loop_scheduler_nodes``
# rebuilds the loops from the flat ops' ``loop_info``.
SCHEDULE_LOOP_NODES = True

# The Inductor surface this module depends on.  "patched": replaced for the
# compile by this module.  Every other entry is read, called or inherited and
# must keep the behaviour described for the patch to stay correct.
PATCHED_SURFACE: tuple[str, ...] = (
    # Patched (per compile, restored afterwards).
    "patched: Scheduler.create_scheduler_node -- a LoopNode becomes one "
    "LoopSchedulerNode; no-op and extern body ops are hoisted",
    "patched: Scheduler.update_zero_dim_cpu_tensor -- first call after "
    "Scheduler._init builds self.nodes; splices the hoisted ops in and puts "
    "graph.operations back to the leaf ops",
    "Scheduler._init -- reads graph.operations only to build self.nodes "
    "(the scheduler-time Spyre readers after it expect leaf ops)",
    "patched: Scheduler.compute_ancestors -- registers children in "
    "name_to_fused_node, copies order and ancestors down to them",
    "patched: Scheduler.dead_node_elimination -- a buffer a child of the same "
    "loop reads is live",
    # Read by, not patched.
    "Scheduler._init -- the order create_scheduler_node, "
    "update_zero_dim_cpu_tensor, compute_dependencies, "
    "topological_sort_schedule, dead_node_elimination, name_to_fused_node "
    "rebuild, compute_ancestors",
    "FusedSchedulerNode.__init__ -- NOT called (reads min_order/max_order, "
    "sets node=None)",
    "FusedSchedulerNode.update_mutated_names / add_fake_dep / can_inplace -- "
    "raise NotImplementedError; overridden",
    "FusedSchedulerNode.get_nodes / get_outputs / get_buffer_names -- inherited",
    "Scheduler.compute_dependencies -- asserts node.node; StarDep(alt_name) "
    "per mutation; additional_buffer_deps / additional_star_deps by node name; "
    "calls add_fake_dep, can_inplace, update_mutated_names; is_ordering_only "
    "branch on 2.14 (not run)",
    "Scheduler.topological_sort_schedule -- by unmet_dependencies",
    "Scheduler._populate_stream_assignments -- node.node.get_stream_idx()",
    "Scheduler._populate_mempool_assignments (2.14) -- node.node.get_mempool()",
    "refresh_group_node_dependencies -- union of the children",
    "init_group_node -- what the constructor reproduces by hand",
    "BaseSchedulerNode.add_fake_dep / set_read_writes / has_side_effects / "
    "update_mutated_names / prune_deps",
    "NodeUser -- (node, can_inplace, is_weak)",
    "SchedulerBuffer -- defining_op, users, set_users",
    "ir.Operation.get_stream_idx / get_origin_node / get_mempool / get_origins "
    "-- assert the attribute exists on the LoopNode",
    "Scheduler._init after compute_ancestors, with a LoopSchedulerNode in "
    "self.nodes: compute_input_distances (unmet_dependencies, "
    "name_to_buf[...].defining_op_name()), create_foreach_nodes, the second "
    "topological_sort_schedule, the _pre_fusion_custom_pass, fuse_nodes "
    "(Spyre's can_fuse_* are False), merge_loops, compute_last_usage "
    "(FusedSchedulerNode.set_last_usage) and _codegen -- run on the supported "
    "release by test_loop_scheduler_equivalence.py",
)


def _release() -> tuple[int, ...]:
    return tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])


def loop_scheduler_supported() -> bool:
    """Whether this torch release is one the scheduler patch supports."""
    return _release() in SUPPORTED_RELEASES


def refuse_release() -> Unsupported:
    supported = ", ".join(f"{major}.{minor}" for major, minor in SUPPORTED_RELEASES)
    return Unsupported(
        f"counted loops on torch {torch.__version__}: the LoopNode scheduler "
        f"patch supports only torch {supported}.x"
    )


class LoopSchedulerNode(CountedLoopSchedulerNode):
    """One scheduler node for one ``LoopNode``; its children are the body's
    scheduler nodes, a nested ``LoopNode`` being a nested ``LoopSchedulerNode``.

    It is created by ``Scheduler.create_scheduler_node``, BEFORE
    ``compute_dependencies``; stock Inductor builds group nodes only after
    ``compute_ancestors``.  So it does NOT call ``FusedSchedulerNode.__init__``:
    ``init_group_node`` reads the children's ``min_order`` / ``max_order``,
    which ``compute_ancestors`` assigns later (``AttributeError``), and sets
    ``node = None``, which ``compute_dependencies`` asserts against.  The
    fields ``init_group_node`` would set are set here, with placeholder order
    fields, and ``node`` is the ``LoopNode``.

    Its dependency data is the union of the children's
    (``refresh_group_node_dependencies``) with the reads narrowed to EXTERNAL
    ones: a buffer the loop itself writes is internal, and the ordering
    between children is the body's program order, which the scheduler never
    changes.  Every body buffer's ``defining_op`` is the outermost loop node,
    the node that is in ``Scheduler.nodes``.

    ``hoisted`` holds the no-op and extern scheduler nodes of the body
    (nested bodies included), in body order, until the scheduler splices them
    in front of the loop.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        snodes: Sequence[BaseSchedulerNode],
        loop_node: LoopNode,
        hoisted: Sequence[BaseSchedulerNode] = (),
    ) -> None:
        if not snodes:
            raise Unsupported(
                f"counted loop {loop_node.get_name()} whose body has no kernel"
            )
        BaseSchedulerNode.__init__(self, scheduler)
        self.snodes = list(snodes)
        self.loop_count = loop_node.trip_count
        self.node = loop_node
        self.loop_node = loop_node
        self.hoisted: list[BaseSchedulerNode] = list(hoisted)
        self.users: list[NodeUser] = []
        self.ancestors = OrderedSet()
        self.min_order = 0
        self.max_order = 0
        self.min_input_distance = 0
        self.max_input_distance = 0
        self.mutation_renames: dict[str, str] = {}
        self.last_usage = OrderedSet()
        self.written = False
        refresh_group_node_dependencies(self)
        # refresh_group_node_dependencies drops internal names from
        # unmet_dependencies only; compute_dependencies registers users from
        # read_writes.reads and set_read_writes rebuilds unmet_dependencies
        # from them, so the internal reads must leave read_writes too.
        self.set_read_writes(self._external(self.read_writes))
        self.outputs_by_name = {buf.get_name(): buf for buf in self.get_outputs()}
        self.group = max(self.snodes, key=lambda x: int(x.is_reduction())).group
        for buf in self.get_outputs():
            buf.defining_op = self

    def _external(self, rw: dependencies.ReadWrites) -> dependencies.ReadWrites:
        inner = self.get_buffer_names()
        return dependencies.ReadWrites(
            OrderedSet(d for d in rw.reads if d.name not in inner),
            rw.writes,
            rw.index_exprs,
            rw.range_vars,
            rw.var_ranges,
        )

    def get_name(self) -> str:  # type: ignore[override]
        # Not cached (FusedSchedulerNode's is): it is the LoopNode's name.
        return self.loop_node.get_name()

    def get_first_name(self) -> str:
        # The loop's own name, which is a key of name_to_fused_node from the
        # moment Scheduler._init builds it (a child's name is registered only
        # by the compute_ancestors patch).
        return self.get_name()

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        # A buffer read in one trip cannot be overwritten in place: the next
        # trip may read it again.
        return False

    def add_fake_dep(self, dep: dependencies.Dep) -> None:
        # compute_dependencies adds StarDep(alt_name) for every buffer a node
        # mutates ("run after the prior writer").  For a loop the prior writer
        # is often one of its own children; that ordering is the body's, and
        # as a dependency of the loop on itself it is a KeyError later.
        if dep.name in self.get_buffer_names():
            return
        BaseSchedulerNode.add_fake_dep(self, dep)

    def update_mutated_names(self, renames: dict[str, str]) -> None:
        # The loop's external reads take the current names, as any node's do,
        # so the loop is ordered after an earlier mutation of what it reads.
        BaseSchedulerNode.update_mutated_names(self, renames)
        # Each child sees the names it would see in the flat list: the
        # renames in force before the loop plus those of earlier children.
        _replay_renames(self.snodes, dict(renames))

    def has_side_effects(self) -> bool:  # type: ignore[override]
        return any(s.has_side_effects() for s in self.snodes)


def _replay_renames(snodes: Sequence[BaseSchedulerNode], renames: dict[str, str]):
    """``compute_dependencies``' renaming, applied to ``snodes`` in order."""

    def rename(name: str) -> str:
        while name in renames:
            name = renames[name]
        return name

    for child in snodes:
        if isinstance(child, LoopSchedulerNode):
            BaseSchedulerNode.update_mutated_names(child, renames)
            _replay_renames(child.snodes, renames)
            continue
        child.update_mutated_names(renames)
        for buf in child.get_outputs():
            for alt_name in buf.get_mutations():
                renames[rename(alt_name)] = buf.get_name()
                renames[alt_name] = buf.get_name()


def _descendants(node: LoopSchedulerNode) -> Iterator[BaseSchedulerNode]:
    """Every child, nested loop and leaf inside ``node``, depth first."""
    for child in node.snodes:
        yield child
        if isinstance(child, LoopSchedulerNode):
            yield from _descendants(child)


def _leaves(node: LoopSchedulerNode) -> Iterator[BaseSchedulerNode]:
    for child in _descendants(node):
        if not isinstance(child, LoopSchedulerNode):
            yield child


def _forward_additional_deps(loop: LoopSchedulerNode) -> None:
    """``compute_dependencies`` looks up ``V.graph.additional_buffer_deps``
    and ``additional_star_deps`` by the name of a node in ``Scheduler.nodes``;
    a child's entries are filed under the loop's name, minus the loop's own
    buffers (an internal ordering is the body's)."""
    inner = loop.get_buffer_names()
    for table in (V.graph.additional_buffer_deps, V.graph.additional_star_deps):
        names = [
            dep
            for leaf in _leaves(loop)
            if leaf.get_name() in table
            for dep in table[leaf.get_name()]
            if dep not in inner
        ]
        if names:
            table[loop.get_name()].update(names)


def _check_members(loop_node: LoopNode, kernels: Sequence[ir.Operation]) -> None:
    """Refuse a kernel op directly in ``loop_node``'s body that is not a
    member of that loop.

    A ``LoopSchedulerNode`` runs every kernel child ``trip_count`` times.  The
    path it replaced decided membership op by op, from ``loop_info``
    (``build_loop_scheduler_nodes``): an op without this loop's group path ran
    outside the loop, once.  ``collapse`` keeps an op that sits between two
    members inside the span, so such an op (an op a pass placed there without
    ``loop_info``) would silently run every trip on this path -- an
    accumulator's zero-fill re-run each trip, a wrong answer.  Every direct
    kernel child must therefore share one group path, and so be in the loop
    on both paths; a nested loop's members are checked at their own level.

    No pass produces such an op today: across the seven graphs of
    ``test_loop_scheduler_equivalence.py`` this check never fires.  It guards a
    future pass that places an op between two members without stamping its
    ``loop_info``.  Ops at the head or tail of a body are not refused: they
    are moved out at the hand-off (``loop_node.peel_non_members``).
    """
    paths = {op.get_name(): counted_loop_path(op) for op in kernels}
    # ``collapse`` wraps the ops sharing ``path[: depth + 1]`` and recurses on
    # longer paths, so the direct kernel children of a node built from
    # ``loop_info`` all have the SAME full path, and a node always starts at
    # a member.  (A node built by hand from ops without ``loop_info``, as the
    # unit tests do, has only empty paths and is not checked.)
    if len(set(paths.values())) <= 1:
        return
    outsiders = sorted(name for name, path in paths.items() if not path)
    detail = (
        f"{outsiders} are in no counted loop"
        if outsiders
        else f"their loop_group_ids differ: {sorted(set(paths.items()))}"
    )
    raise Unsupported(
        f"counted loop {loop_node.get_name()} holds kernel ops that are not all "
        f"members of it ({detail}); they would run every trip here and "
        "differently on the flat path"
    )


def _build_loop_scheduler_node(
    scheduler: Scheduler, loop_node: LoopNode
) -> LoopSchedulerNode:
    """Create the scheduler nodes of ``loop_node``'s body and wrap them.

    A no-op (an allocation that generates no code) or an extern kernel is not
    a kernel, and Spyre's codegen rejects it inside a counted loop.  The flat
    pipeline excludes them only by accident (``_loop_group_id`` finds no
    ``SchedulerNode`` in them).  Here it is explicit: they are created as
    ordinary nodes and placed immediately before their (outermost) loop.  One
    that reads a buffer the loop writes cannot go before it, and is refused.
    """
    children: list[BaseSchedulerNode] = []
    hoisted: list[BaseSchedulerNode] = []
    kernels: list[ir.Operation] = []
    for op in loop_node.body:
        if isinstance(op, LoopNode):
            inner = _build_loop_scheduler_node(scheduler, op)
            hoisted.extend(inner.hoisted)
            inner.hoisted = []
            children.append(inner)
            continue
        snode = scheduler.create_scheduler_node(op)
        if isinstance(snode, (NopKernelSchedulerNode, ExternKernelSchedulerNode)):
            hoisted.append(snode)
        else:
            children.append(snode)
            kernels.append(op)
    _check_members(loop_node, kernels)
    loop = LoopSchedulerNode(scheduler, children, loop_node, hoisted)
    inner_names = loop.get_buffer_names()
    for snode in hoisted:
        produced = {d.name for d in snode.read_writes.reads} & set(inner_names)
        if produced:
            raise Unsupported(
                f"{type(snode.node).__name__} {snode.get_name()} in counted loop "
                f"{loop.get_name()} reads {sorted(produced)}, which the loop "
                "writes, so it cannot be placed before the loop"
            )
    _forward_additional_deps(loop)
    return loop


_PATCHED = (
    "create_scheduler_node",
    "update_zero_dim_cpu_tensor",
    "compute_ancestors",
    "dead_node_elimination",
)


@contextlib.contextmanager
def _replace(replacements: dict) -> Iterator[None]:
    """Install ``replacements`` on ``Scheduler`` for the block; afterwards
    every method is exactly (``is``) what it was before."""
    saved = {name: Scheduler.__dict__[name] for name in replacements}
    for name, fn in replacements.items():
        setattr(Scheduler, name, fn)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(Scheduler, name, fn)
        for name, fn in saved.items():
            assert Scheduler.__dict__[name] is fn, f"Scheduler.{name} not restored"


@contextlib.contextmanager
def apply_loop_scheduler_patches() -> Iterator[None]:
    """Let a ``LoopNode`` reach the scheduler as one ``LoopSchedulerNode``.

    Raises ``Unsupported`` on entry on a torch release outside
    ``SUPPORTED_RELEASES``, installing nothing.  Nesting is allowed; each
    level restores what it found.  With no ``LoopNode`` in the graph every
    patched method takes the original path.
    """
    if not loop_scheduler_supported():
        raise refuse_release()
    orig = {name: Scheduler.__dict__[name] for name in _PATCHED}

    def create_scheduler_node(self, node):
        if isinstance(node, LoopNode):
            return _build_loop_scheduler_node(self, node)
        return orig["create_scheduler_node"](self, node)

    def update_zero_dim_cpu_tensor(self):
        # Scheduler._init calls this right after building self.nodes, before
        # anything indexes them: the first point the hoisted ops can join.
        if any(isinstance(n, LoopSchedulerNode) for n in self.nodes):
            spliced: list[BaseSchedulerNode] = []
            for node in self.nodes:
                if isinstance(node, LoopSchedulerNode):
                    spliced.extend(node.hoisted)
                    node.hoisted = []
                spliced.append(node)
            self.nodes = spliced
            # The loop structure now lives in the scheduler nodes, and
            # Scheduler._init has read graph.operations for the last time.
            # Everything that reads it from here on (kernel preparation's LX
            # read contracts: storage_has_single_writer,
            # _is_activation_source; Inductor's replace_operation_buffer)
            # expects leaf ops: a LoopNode has no get_mutation_names
            # (AttributeError), and its get_read_writes, the union of its
            # body's, would make a body op's buffer look written by another
            # op.  So put the leaf list back; each LoopNode keeps its body.
            expand(V.graph)
        return orig["update_zero_dim_cpu_tensor"](self)

    def compute_ancestors(self):
        loops = [n for n in self.nodes if isinstance(n, LoopSchedulerNode)]
        # After fusion stock Inductor maps every member's name to its group;
        # a loop node is a group from the start, so do the same.
        for loop in loops:
            for inner in _descendants(loop):
                self.name_to_fused_node[inner.get_name()] = loop
        result = orig["compute_ancestors"](self)
        for loop in loops:
            for inner in _descendants(loop):
                inner.ancestors = OrderedSet(loop.ancestors)
                inner.min_order = loop.min_order
                inner.max_order = loop.max_order
        return result

    def dead_node_elimination(self):
        # compute_dependencies registered users from the loop's EXTERNAL reads
        # only, so a buffer produced and consumed inside the loop has no user
        # and the stock pass removes it as dead (B.11 failure 7).  The loop
        # itself is judged first, by the stock pass, on its external users and
        # side effects; giving its internal buffers the loop as a user BEFORE
        # that (as the prototype did) would keep every loop with two chained
        # children alive forever.
        #
        # Afterwards, a loop that survives keeps EVERY body buffer: the loop
        # is atomic, so a child whose buffer was judged dead still runs and
        # still stores it, and codegen asserts a stored buffer is not removed.
        # A buffer a child reads gets the loop as its user.  Consequence: a
        # body buffer nobody reads is kept (dead compute, not wrong code);
        # Spyre's own deadcode_elimination has normally removed it already.
        loops = [n for n in self.nodes if isinstance(n, LoopSchedulerNode)]
        removed_before = OrderedSet(V.graph.removed_buffers)
        result = orig["dead_node_elimination"](self)
        for loop in loops:
            if loop.get_name() in V.graph.removed_operations:
                # A removed loop takes its body with it, as the flat list would.
                V.graph.removed_operations.update(
                    inner.get_name() for inner in _descendants(loop)
                )
                continue
            read = {d.name for leaf in _leaves(loop) for d in leaf.read_writes.reads}
            for buf in loop.get_outputs():
                name = buf.get_name()
                if name in V.graph.removed_buffers and name not in removed_before:
                    V.graph.removed_buffers.discard(name)
                if name in read and all(u.is_weak for u in buf.users):
                    buf.set_users([*buf.users, NodeUser(loop)])
        return result

    with _replace(
        {
            "create_scheduler_node": create_scheduler_node,
            "update_zero_dim_cpu_tensor": update_zero_dim_cpu_tensor,
            "compute_ancestors": compute_ancestors,
            "dead_node_elimination": dead_node_elimination,
        }
    ):
        yield


@contextlib.contextmanager
def _refuse_loop_nodes() -> Iterator[None]:
    """On an unsupported release: only a ``LoopNode`` is refused, loudly."""
    orig = Scheduler.__dict__["create_scheduler_node"]

    def create_scheduler_node(self, node):
        if isinstance(node, LoopNode):
            raise refuse_release()
        return orig(self, node)

    with _replace({"create_scheduler_node": create_scheduler_node}):
        yield


@contextlib.contextmanager
def loop_scheduler_patches_for_compile() -> Iterator[None]:
    """What ``patches.py`` enters for every Spyre compile.

    On a supported release this is :func:`apply_loop_scheduler_patches`.  On
    any other it must not fail a compile that has no loop, so it does not
    raise on entry: it installs only a ``create_scheduler_node`` that raises
    ``Unsupported`` when it is handed a ``LoopNode`` and is otherwise the
    original.
    """
    if loop_scheduler_supported():
        with apply_loop_scheduler_patches():
            yield
    else:
        with _refuse_loop_nodes():
            yield
