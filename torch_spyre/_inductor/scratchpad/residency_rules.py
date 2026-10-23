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


"""LX residency rules read from the IR: why an op output or a graph input may
not reside in LX, and which reads may take over a producer's storage in place.
The graph builder (``graph_builder``) sets each node's ``residency_reason``
and in-place parents from these."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Optional

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    ExternKernel,
    FallbackKernel,
    MutationLayoutSHOULDREMOVE,
    Operation,
    ReinterpretView,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.ir import FixedTiledLayout, SpyreEmptyFallback
from torch_spyre._inductor.loop_info import LoopCarryRecord
from torch_spyre._inductor.pass_utils import (
    indirect_info_from_op,
    op_read_writes,
    op_short_name,
)
from torch_spyre._inductor.scratchpad.graph_editor import (
    GraphEditor,
)
from torch_spyre._inductor.scratchpad.utils import (
    OP_OUTPUT_NOT_GOOD_FOR_LX_REUSE,
    _get_buffer_user_deps,
    _is_read_advancing_anywhere,
    _is_tiled_advancing,
    _would_produce_lx_back_gap,
    buffer_not_read_in_full,
    calculate_liveness,
    clone_at_graph_boundaries,
    get_op_pointwise_inputs,
    is_empty_tiled_layout,
)


def _extern_kernel_in_live_range(graph: GraphLowering, uses: list[int]) -> bool:
    """True if an opaque extern kernel runs at any point while the buffer is live.

    The LX scratchpad is a fixed per-core resource shared by *every* compiled
    Spyre program, and it is not threaded through the generated wrapper as a
    tensor -- a resident buffer is handed from one kernel launch to the next by
    its LX offset alone. An extern kernel is opaque: its body can launch other
    compiled programs (a nested ``torch.compile``, or any eager op, which
    torch-spyre compiles standalone via ``compile_once``), and those programs
    allocate the same LX offsets. A buffer left resident across such a call is
    therefore silently overwritten, and its consumer reads the other program's
    data.

    Being *accessed by* the extern kernel is the narrow case (already fatal,
    since the value must be a real HBM tensor to be passed to it); merely being
    live *across* one is equally fatal and is not visible from ``uses``
    membership alone.
    """
    if not uses:
        return False
    return any(
        isinstance(graph.operations[i], ExternKernel)
        and not isinstance(graph.operations[i], SpyreEmptyFallback)
        for i in range(min(uses), max(uses) + 1)
    )


def _multi_output_extern_kernel_in_live_range(
    graph: GraphLowering, uses: list[int]
) -> bool:
    """True if a genuinely multi-output FallbackKernel runs while the buffer is live.

    ``LxContextSwitchingPass`` (lx_context_switching.py's ``_select_bracket_targets``)
    does not bracket multi-output FallbackKernels -- WeakDep target resolution in
    ``_order_around_bracket`` is only confirmed correct for single-output kernels --
    so a buffer whose only risky crossing is one of those gets no dump/restore
    protection from the pass. Unlike the single-output case, this check runs
    unconditionally (not gated on config.enable_lx_context_switching): the flag
    only chooses between the old guard and the new pass for cases the new pass
    actually handles, and this is not one of them.
    """
    if not uses:
        return False
    for i in range(min(uses), max(uses) + 1):
        op = graph.operations[i]
        if not isinstance(op, FallbackKernel):
            continue
        outputs = getattr(op, "outputs", None)
        if outputs is not None and len(outputs) > 1:
            return True
    return False


def _is_loop_carry_storage(op: Any) -> bool:
    """True only for storage explicitly created as a counted-loop carry."""

    record = getattr(op, "_loop_carry_record", None)
    return isinstance(record, LoopCarryRecord) and record.storage_name == op.get_name()


def _get_op_name(op: Any) -> str:
    return op_short_name(op)


def _op_output_good_for_lx_reuse(op: Any) -> bool:
    if not isinstance(op, ComputedBuffer):
        return False
    if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
        return False
    # A CPU-resident ComputedBuffer has a plain FixedLayout
    # with no device_layout and can never be LX-pinned.
    if not isinstance(op.layout, FixedTiledLayout):
        return False
    return (
        _is_loop_carry_storage(op)
        or config.allow_all_ops_in_lx_planning
        or _get_op_name(op) not in OP_OUTPUT_NOT_GOOD_FOR_LX_REUSE
    )


def _read_count(uses: list[int]) -> int:
    """Reads residency would serve from LX. The first use is never one of
    them: it is either the producer's write (an intermediate) or the clone-in
    read a graph input cannot avoid.

    Deliberately not ``ResourceNode.read_count``, which counts the
    buffer's reads and so includes an input's clone-in; this is the savings,
    which discounts it in both cases (as ``spill_cost`` does)."""
    return max(0, len(uses) - 1)


def _is_index_or_indirectly_accessed(
    graph: GraphLowering,
    name: str,
    uses: list[int],
    op: Optional[Operation],
) -> bool:
    """True if ``name`` is an index tensor, or is itself accessed
    indirectly (a gather/scatter value tensor), on either side of its
    lifetime: as ``op``'s own indirect write (a Scatter target), or as a
    read/index operand of any consumer in ``uses``.

    Both the index tensor and the tensor it indexes into must stay off
    the scratchpad and resolve from HBM instead.
    """
    if isinstance(op, ComputedBuffer):
        writes = op_read_writes(op).writes
        if any(isinstance(dep, MemoryDep) and dep.is_indirect() for dep in writes):
            return True
    for u in uses:
        consumer = graph.operations[u]
        if not isinstance(consumer, ComputedBuffer):
            continue
        index_names, _, _ = indirect_info_from_op(consumer)
        if name in index_names:
            return True
        reads = op_read_writes(consumer).reads
        if any(
            dep.name == name and isinstance(dep, MemoryDep) and dep.is_indirect()
            for dep in reads
        ):
            return True
    return False


def _buffer_residency_reason(
    graph: GraphLowering,
    name: str,
    uses: list[int],
    op: Optional[Operation],
    *,
    mutated_buffers: set[str],
    graph_output_names: set[str],
    reinterpret_output_names: set[str],
    buf_user_deps: dict[str, list[tuple[Operation, MemoryDep]]],
    drain_plans: Collection[str] = (),
) -> Optional[str]:
    """The first check ``name`` fails, or ``None`` if it clears them all.

    Order matters. The unsized and op-kind guards come first because
    everything below assumes a placeable ``ComputedBuffer``; the back-gap
    probe comes last because it touches ``device_layout`` and is the most
    expensive. The graph-wide facts (``mutated_buffers``,
    ``graph_output_names`` ...) are computed once per solve by
    :meth:`_residency_reasons` and passed in, so this stays O(1) in the graph.

    Args:
        graph: The lowered graph.
        name: Buffer name (an op name -- graph inputs go through
            :meth:`_input_residency_reason`).
        uses: ``name``'s liveness (the op indices where it is accessed).
        op: ``name``'s producing op, or ``None`` if it has none.
        buf_user_deps: every buffer's ``(op, dep)`` users, from
            :func:`_get_buffer_user_deps`, for the read-side advancing check.
    """
    if op is None or not _op_output_good_for_lx_reuse(op):
        return "op not allowed"
    if not hasattr(getattr(op, "layout", None), "device_layout"):
        # No device layout => no computable footprint (e.g. a
        # MultiOutputLayout tuple op). There is nothing to place, and the
        # checks below would raise.
        return "unsized (no device layout)"
    if is_empty_tiled_layout(op.layout):
        # The joint solver does not consult the fixed-division judge until
        # after placement, so it must share this pre-allocation exclusion.
        return "empty tensor"
    # A counted-loop carry may stay resident: the solver's relation gates its
    # update's division against the carry's.
    if name in mutated_buffers and not _is_loop_carry_storage(op):
        return "mutation target"
    if _is_tiled_advancing(op) or _is_read_advancing_anywhere(name, buf_user_deps):
        # LX addresses cannot be expressed as affine.apply symbols today (see
        # compute_ops.py's generate_sdsc, which raises NotImplementedError via
        # _tensor_tiled_by_symbol for exactly this case), so a buffer whose
        # address advances per coarse-tile iteration must stay in HBM, where that is
        # supported -- whether the advance is on this buffer's own write
        # (_is_tiled_advancing) or on some other op's read of it
        # (_is_read_advancing_anywhere, e.g. a fixed-write full buffer
        # copied into a nested tile every outer iteration).
        return "tiled (advancing)"
    # PR3683's guard: reject residency outright rather than let LX context
    # switching (dump/restore around the risky call) handle it. Kept behind
    # the flag, not deleted, so the old (conservative) and new (context
    # switching) behaviors can still be compared -- see
    # LxContextSwitchingPass in lx_context_switching.py, which is the intended
    # long-term replacement for this check.
    if not config.enable_lx_context_switching and _extern_kernel_in_live_range(
        graph, uses
    ):
        return "extern kernel user or live across extern kernel"
    # Unconditional, regardless of the flag above: LxContextSwitchingPass does
    # not bracket multi-output FallbackKernels (see
    # _multi_output_extern_kernel_in_live_range's docstring), so a buffer live
    # across one gets no protection from either mechanism unless residency is
    # refused here.
    if config.enable_lx_context_switching and (
        _multi_output_extern_kernel_in_live_range(graph, uses)
    ):
        return "live across multi-output extern kernel"
    if _is_index_or_indirectly_accessed(graph, name, uses, op):
        # Index tensors and the value tensors they index into are read via
        # data-dependent (indirect) addressing, must stay in hbm.
        return "index tensor or indirectly accessed"
    if name in graph_output_names:
        # A graph output normally can't reside (the value must land back in
        # HBM), but with boundary cloning on it is pinned via an output clone
        # that still writes HBM once; that unavoidable write cancels from the
        # CP-SAT differential spill cost, so allow residency then.
        if not clone_at_graph_boundaries():
            return "graph output (no clone)"
        if name in reinterpret_output_names:
            return "graph output is a ReinterpretView"
        if name in mutated_buffers and name not in drain_plans:
            # The output clone is inserted after the producer, so it would
            # copy the value from before a later in-place update (e.g. a
            # loop carry returned from the graph).  A validated drain plan
            # re-anchors that clone after the whole counted loop instead, so
            # only the plan clears this refusal.
            return "graph output mutated after production"
    if buffer_not_read_in_full(graph, name):
        return "partial/offset read"
    if _read_count(uses) == 0:
        # Only the producer's write touches it, so residency saves nothing.
        return "no consumer reads it from LX"
    if _would_produce_lx_back_gap(graph, name, uses):
        # backGap fires when device_size[d] > it_dim_size; the backend
        # supports it for HBM but not for LX.
        return "lx back gap"
    return None


def _input_residency_reason(
    graph: GraphLowering,
    name: str,
    uses: list[int],
) -> Optional[str]:
    """The residency verdict for a *graph input*, which is pinned by cloning
    it into LX rather than by placing it directly.

    An input has no producing op, so the op-kind checks do not apply; what
    does apply is that the clone must be substitutable at every use, and that
    residency has to beat the clone-in transfer it costs. An input read only
    once is already in HBM and would need one transfer to clone, so pinning
    it saves nothing -- which ``_read_count`` (first use excluded) states
    directly.
    """
    if not clone_at_graph_boundaries():
        return "graph input (no clone)"
    if is_empty_tiled_layout(getattr(graph.try_get_buffer(name), "layout", None)):
        return "empty tensor"
    if _read_count(uses) == 0:
        return "no consumer reads it from LX"
    if _is_index_or_indirectly_accessed(graph, name, uses, None):
        return "index tensor or indirectly accessed"
    # See the matching comment in _buffer_residency_reason: kept behind
    # config.enable_lx_context_switching rather than deleted, so the
    # PR3683 guard and LxContextSwitchingPass's dump/restore can be
    # compared during rollout.
    if not config.enable_lx_context_switching and _extern_kernel_in_live_range(
        graph, uses
    ):
        return "extern kernel user or live across extern kernel"
    if not GraphEditor.all_uses_are_rewritable(graph, uses):
        return "use is not rewritable to the clone"
    if buffer_not_read_in_full(graph, name):
        return "partial/offset read"
    if _would_produce_lx_back_gap(graph, name, uses):
        return "lx back gap"
    return None


def _residency_reasons(
    graph: GraphLowering,
    names: "list[str] | set[str]",
    *,
    lifetimes: Optional[dict[str, list[int]]] = None,
    drain_plans: Collection[str] = (),
) -> dict[str, Optional[str]]:
    """:meth:`_buffer_residency_reason` over ``names``, as ``name -> reason``.

    Computes the graph-wide facts the per-buffer check needs once, here, and
    passes them down -- there is no shared context object. ``lifetimes`` is
    accepted so a caller that already has it skips recomputing.
    """
    if lifetimes is None:
        lifetimes = calculate_liveness(graph)
    op_by_name = {op.name: op for op in graph.operations}
    mutated_buffers = {
        op.layout.target.get_name()
        for op in graph.operations
        if isinstance(op.layout, MutationLayoutSHOULDREMOVE)
    }
    graph_output_names = set(graph.get_output_names())
    reinterpret_output_names = {
        go.get_name()
        for go in graph.graph_outputs
        if isinstance(go, ReinterpretView)
        or isinstance(getattr(go, "data", None), ReinterpretView)
    }
    buf_user_deps = _get_buffer_user_deps(graph)
    return {
        name: _buffer_residency_reason(
            graph,
            name,
            lifetimes.get(name, []),
            op_by_name.get(name),
            mutated_buffers=mutated_buffers,
            graph_output_names=graph_output_names,
            reinterpret_output_names=reinterpret_output_names,
            buf_user_deps=buf_user_deps,
            drain_plans=drain_plans,
        )
        for name in names
    }


def _op_inputs_good_for_lx_inplace(op: Any) -> list[str]:
    target = getattr(getattr(op, "origin_node", None), "target", None)
    if target is None:
        return []
    # A pointwise ATen target can consume a transposed/broadcast view after
    # lowering. Its tag does not prove element-for-element storage reuse.
    return get_op_pointwise_inputs(op)


def _inplace_edge_ok(
    *,
    child_pointwise_inputs: list[str],
    parent_name: str,
    child_device_layout: Any,
    parent_device_layout: Any,
    child_start: int,
    parent_end: int,
    child_size_per_core: Optional[int] = None,
    parent_size_per_core: Optional[int] = None,
    child_core_div_mismatch: bool = False,
    division_invariant: bool = False,
) -> bool:
    """Whether ``parent_name`` may be reused in place by the child buffer.

    The child (which writes at ``child_start``) reuses the parent's storage,
    so the parent must die exactly as the child is born. The conditions:

    - the parent is a pointwise-eligible read input of the child;
    - matching device layout (so the storage can alias);
    - single handoff tick (``parent_end == child_start``: the same op that reads
      the parent as its last use writes the child), the invariant the solvers'
      in-place relaxation relies on (``encode_no_overlap`` shortens the child by
      that one tick).
      Callers pass ticks widened by counted-loop lifetime overrides
      (``_handoff_child_start`` / ``_handoff_parent_end``), so a buffer kept
      live across a loop never qualifies;
    - matching per-core footprint and no core-division mismatch on the child.

    With ``division_invariant`` the last condition (per-core size + core-div) is
    skipped: those depend on a core division the joint solver has not chosen
    yet, so it enforces them itself (``eff_size`` equality + the
    ``cd_parent_matches`` gate). Only the first three (division-invariant)
    preconditions are checked.

    This is the sole definition of a legal in-place edge, shared by
    ``GraphBuilder._determine_in_place_division_invariant`` and
    ``GraphBuilder._build_graph``, so they cannot drift.
    """
    base_ok = (
        parent_name in child_pointwise_inputs
        and child_device_layout == parent_device_layout
        and parent_end == child_start
    )
    if division_invariant:
        return base_ok
    return (
        base_ok
        and child_size_per_core == parent_size_per_core
        and not child_core_div_mismatch
    )
