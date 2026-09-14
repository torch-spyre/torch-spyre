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

"""LX dump/restore around opaque (FallbackKernel) device calls.

A FallbackKernel's eager body is opaque: it can launch another compiled Spyre
program (a nested torch.compile, or any eager op -- torch-spyre compiles many
of those standalone via ops/eager.py's compile_once), and that program plans
its own LX scratchpad addresses from scratch, unaware that this graph already
has other buffers LX-resident at that exact point. Left alone, that silently
clobbers them (see the design doc / repro_lx_extern_clobber.py referenced in
PR3683's follow-up).

This module brackets every risky FallbackKernel with a per-buffer dump (LX ->
HBM) immediately before it and a per-buffer restore (HBM -> LX) immediately
after, so any buffer this graph still needs LX-resident survives the call
regardless of what the opaque kernel does to LX -- replacing PR3683's
"never pin the buffer to LX at all" guard with a real fix instead of a
blanket restriction. See config.enable_lx_context_switching and the matching
comments in scratchpad/allocator.py's residency-reason functions.
"""

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    FallbackKernel,
    MutationLayoutSHOULDREMOVE,
)
from torch._inductor.lowering import clone as clone_lowering

from torch_spyre._inductor import config
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.logging_utils import get_inductor_logger, warn_once
from torch_spyre._inductor.pass_utils import copy_op_metadata
from torch_spyre._inductor.scratchpad.allocator import ScratchpadOptimizationPass
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor
from torch_spyre._inductor.scratchpad.utils import calculate_liveness
from torch_spyre.ops.fallbacks import fallback_ops

logger = get_inductor_logger("lx_context_switching")


def mark_lx_safe(
    op: "torch._ops.OpOverload | torch._library.custom_ops.CustomOpDef",
) -> None:
    """Opt an op out of LX dump/restore bracketing.

    Use when an op's eager body is confirmed to never launch a nested
    torch.compile, so it can never clobber LX (see customops.py's usage
    for the safety contract and worked examples). Saves the small but
    nonzero dump/restore cost around every call site. Plain attribute
    assignment on the OpOverload/CustomOpDef -- torch.Tag is a closed C++
    enum with no room for a Spyre-specific value.
    """
    op._spyre_lx_safe = True


def _is_cpu_only_fallback(op_overload: object) -> bool:
    """True if `op_overload` is registered in ops/fallbacks.py's fallback_ops.

    Every op there shares the same `_fallback` kernel body (fallbacks.py),
    which unconditionally moves tensors to `device="cpu"` (the default for
    every call site in that file today) and runs there -- confirmed no
    device/LX interaction by inspection of that one shared code path.
    """
    return op_overload in fallback_ops


def _is_lx_resident(graph: GraphLowering, name: str) -> bool:
    buf = graph.get_buffer(name)
    if buf is None or not isinstance(buf, Buffer):
        return False
    layout = buf.maybe_get_layout()
    return isinstance(layout, FixedTiledLayout) and "lx" in layout.allocation


def _at_risk_buffers(
    graph: GraphLowering,
    fk_idx: int,
    liveness: dict[str, list[int]],
) -> list[Buffer]:
    """LX-resident buffers whose live range strictly straddles `fk_idx`.

    `liveness` is `calculate_liveness(graph)`'s {buf_name: sorted use-indices}
    (the same helper the allocator itself calls `lifetimes` for solver input).
    A buffer's live range is (uses[0], uses[-1] + 1) -- the op that writes it
    through the last op that reads it -- straddling `fk_idx` means already
    produced, still needed after: `uses[0] < fk_idx < uses[-1] + 1`.
    """
    at_risk = []
    for name, uses in liveness.items():
        if not uses or not _is_lx_resident(graph, name):
            continue
        if uses[0] < fk_idx < uses[-1] + 1:
            at_risk.append(graph.get_buffer(name))
    return at_risk


def _select_bracket_targets(
    graph: GraphLowering,
) -> list[tuple[FallbackKernel, list[Buffer]]]:
    """Target + buffer selection: which FallbackKernels to bracket, and with what.

    Layer 1 (cheap skip): a FallbackKernel whose op is a confirmed CPU-only
    fallback (_is_cpu_only_fallback) or explicitly opted out (mark_lx_safe)
    never launches a competing compiled program, so it is never a candidate,
    regardless of buffer lifetimes.

    Layer 2 (load-bearing): otherwise, bracket only if some LX-resident
    buffer's live range strictly straddles this op -- already produced, still
    needed after. This is the gate that actually matters: classifying op
    behavior by namespace/registry is unreliable in general (ops/eager.py
    registers many aten ops, e.g. mm/add/softmax/embedding, whose eager
    dispatch is itself a nested single-op torch.compile through the full
    Inductor pipeline including LX planning -- exactly this hazard), so
    buffer lifetime -- data this graph owns outright -- is what decides,
    not op identity.

    Genuinely multi-output FallbackKernels (more than one trailing
    MultiOutput -- upstream's FallbackKernel.create() wraps *every*
    Tensor-returning kernel in MultiOutputLayout, single-output included, so
    that alone is not the signal; len(op.outputs) > 1 is) are skipped with a
    warning: the WeakDep target resolution in _order_around_bracket is only
    confirmed correct for single-output kernels. Left unbracketed rather than
    risk an incorrect fix; allocator.py's
    _multi_output_extern_kernel_in_live_range refuses LX residency outright
    for any buffer live across one of these, so the gap here does not leave
    a buffer unprotected -- it just falls back to the conservative
    PR3683-style guard for this one case, same as when the flag is off.
    """
    liveness = calculate_liveness(graph)
    targets: list[tuple[FallbackKernel, list[Buffer]]] = []

    for fk_idx, op in enumerate(graph.operations):
        if not isinstance(op, FallbackKernel):
            continue
        outputs = getattr(op, "outputs", None)
        if outputs is None or len(outputs) > 1:
            # Dedupe by op kind (op_overload when available, else the
            # Operation subclass), not op.get_operation_name() -- that is a
            # per-instance buffer name (e.g. "buf42"), unique to this graph,
            # so it would never actually suppress repeats across compiles.
            op_kind = (
                repr(op.op_overload)
                if op.op_overload is not None
                else type(op).__name__
            )
            warn_once(
                logger,
                op_kind,
                "lx context switching: skipping multi-output FallbackKernel "
                "%s (not yet supported). If you need this supported, please "
                "file an issue and tag @chichun-charlie-liu.",
                op_kind,
            )
            continue

        op_overload = op.op_overload
        if isinstance(op_overload, torch._ops.OpOverload):
            if _is_cpu_only_fallback(op_overload):
                continue
            if getattr(op_overload, "_spyre_lx_safe", False):
                continue

        at_risk = _at_risk_buffers(graph, fk_idx, liveness)
        if at_risk:
            targets.append((op, at_risk))

    return targets


def _dump_buffers(
    graph: GraphLowering,
    editor: GraphEditor,
    target_op: FallbackKernel,
    target_fx_node: torch.fx.Node,
    at_risk_bufs: list[Buffer],
) -> dict[str, tuple[torch.fx.Node, ComputedBuffer]]:
    """Dump (LX -> HBM) each at-risk buffer, one at a time, right before
    target_fx_node. Returns {buf_name: (dump_fx, dump_buf)} for the restore
    loop below."""
    dumped: dict[str, tuple[torch.fx.Node, ComputedBuffer]] = {}

    for buf in at_risk_bufs:
        assert isinstance(buf, ComputedBuffer), (
            f"unexpected at-risk buffer type {type(buf)} ({buf})"
        )
        buf_fx = list(buf.origins)[0]  # only the clone's input arg -- not an
        # insertion point. target_fx_node (the FallbackKernel) is fixed for
        # the whole bracket, so every buffer's dump/restore pair anchors
        # there, however far buf_fx itself sits from the bracket.
        editor.fx_graph.inserting_before(target_fx_node)
        dump_fx = editor.fx_graph.create_node(
            "call_function", editor.clone_aten_op, (buf_fx,)
        )
        # no rewiring of buf's users -- dump has no FX consumers; it exists
        # only so the restore clone below can read it.

        layout = buf.layout
        assert isinstance(layout, FixedTiledLayout)
        dump_layout = FixedTiledLayout(
            layout.device,
            layout.dtype,
            list(layout.size),
            list(layout.stride),
            layout.device_layout,
            offset=layout.offset,
        )  # allocation dict starts empty: invisible to LX planning (already
        # run by this point) and picked up later by ordinary HBM/hbm_pool
        # buffer addressing, same as any other un-pinned intermediate.
        clone_tb = clone_lowering(buf)
        dump_buf = ComputedBuffer(
            name=None,
            layout=dump_layout,
            data=clone_tb.data.data,  # type: ignore[union-attr]
        )
        dump_buf.data.origins.add(dump_fx)
        dump_buf.origins.add(dump_fx)
        dump_buf.origin_node = dump_fx
        copy_op_metadata(buf, dump_buf)  # keep it in buf's coarse-tile/loop-group
        dump_buf.op_it_space_splits = getattr(buf, "op_it_space_splits", ({}, {}))
        dump_buf.name = graph.register_buffer(dump_buf)
        graph.register_operation(dump_buf)

        # Reposition into graph.operations right before target_op
        # (remove()+insert(), same pattern GraphEditor.push_allocation_with_clone
        # uses). target_op's own index drifts as we insert; recompute it.
        graph.operations.remove(dump_buf)
        graph.operations.insert(graph.operations.index(target_op), dump_buf)

        dumped[buf.get_name()] = (dump_fx, dump_buf)
        logger.info(
            "lx dump: %s (lx) -> %s (hbm), protecting it across %s",
            buf.get_name(),
            dump_buf.get_name(),
            target_fx_node,
        )

    return dumped


def _restore_buffers(
    graph: GraphLowering,
    editor: GraphEditor,
    target_op: FallbackKernel,
    target_fx_node: torch.fx.Node,
    at_risk_bufs: list[Buffer],
    dumped: dict[str, tuple[torch.fx.Node, ComputedBuffer]],
) -> list[ComputedBuffer]:
    """Restore (HBM -> LX) each at-risk buffer, one at a time, right after
    target_fx_node. Returns the restore ComputedBuffers, for ordering."""
    restored: list[ComputedBuffer] = []

    for buf in at_risk_bufs:
        dump_fx, dump_buf = dumped[buf.get_name()]

        editor.fx_graph.inserting_after(target_fx_node)
        restore_fx = editor.fx_graph.create_node(
            "call_function", editor.clone_aten_op, (dump_fx,)
        )
        # Reads dump_fx -- a REAL dependency, not synthetic; this is what
        # keeps reachability/DCE correct on its own for the dump leg.

        clone_tb = clone_lowering(dump_buf)
        restore_buf = ComputedBuffer(
            name=None,
            # MutationLayoutSHOULDREMOVE(buf).real_layout() returns buf's own
            # layout object verbatim (same device_layout/stride, not an
            # independently-chosen one), so this write lands at buf's exact
            # original LX address and there is no stride-mismatch risk from
            # the WrapperHandler-swap caveat (no consumer inner_fn is patched
            # here at all -- redirection is via the scheduler's
            # mutation_real_name map, not a rename).
            layout=MutationLayoutSHOULDREMOVE(buf),
            data=clone_tb.data.data,  # type: ignore[union-attr]
        )
        restore_buf.data.origins.add(restore_fx)
        restore_buf.origins.add(restore_fx)
        restore_buf.origin_node = restore_fx
        copy_op_metadata(buf, restore_buf)
        # Without this, restore_buf defaults to an untiled single-core view,
        # which disagrees with buf's own 32-core split as seen by buf's other
        # users. Since this write aliases buf's own layout object (via
        # MutationLayoutSHOULDREMOVE), that disagreement isn't just cosmetic:
        # the scheduler's LX-relayout consistency check (scheduler.py's
        # demote()) treats it as an invalid view and pops "lx" from buf's own
        # (shared) allocation dict -- silently demoting buf out of LX
        # entirely, turning the whole bracket into dead weight (dump/restore
        # clones around a buffer that no longer resides on LX at all).
        restore_buf.op_it_space_splits = getattr(buf, "op_it_space_splits", ({}, {}))
        restore_buf.name = graph.register_buffer(restore_buf)
        graph.register_operation(restore_buf)

        graph.operations.remove(restore_buf)
        graph.operations.insert(graph.operations.index(target_op) + 1, restore_buf)

        restored.append(restore_buf)
        logger.info(
            "lx restore: %s (hbm) -> %s (lx), after %s",
            dump_buf.get_name(),
            buf.get_name(),
            target_fx_node,
        )

    return restored


def _order_around_bracket(
    graph: GraphLowering,
    target_op: FallbackKernel,
    dump_bufs: list[ComputedBuffer],
    restore_bufs: list[ComputedBuffer],
) -> None:
    """Bookkeeping-only ordering: target_op must run after every dump, and
    every restore must run after target_op.

    Uses GraphLowering.additional_buffer_deps/additional_star_deps --
    upstream's own mechanism for a "fake dependency on an unused buffer... to
    prevent some specific reordering" (unused elsewhere in torch-spyre).
    Scheduler.__init__ converts these into real WeakDep edges (is_fake=True:
    ordering-only, no lifetime extension) once scheduler nodes exist, by
    iterating `additional_buffer_deps[node.get_name()]` for each scheduler
    node and adding a WeakDep on each entry.

    Two different names are in play here and they are NOT interchangeable:
    `register_buffer`/`register_operation` (graph.py) draw from independent
    "bufN"/"opN" counters, so a FallbackKernel's buffer name and operation
    name are, in general, different strings (unlike a plain ComputedBuffer,
    where they typically coincide). `BaseSchedulerNode.get_name()` delegates
    to the underlying op's get_operation_name() -- so a dict *key* here must
    be an operation name, to match the scheduler node currently being
    visited. But a dict *value* is a dependency target resolved via
    `Scheduler.name_to_buf` (a real buffer name) -- so a value naming a
    FallbackKernel must be its get_name(), not get_operation_name(), or
    `compute_ancestors` raises KeyError looking it up. Confirmed correct for
    single-output FallbackKernels; multi-output kernels are excluded
    upstream in _select_bracket_targets.
    """
    extern_op_name = target_op.get_operation_name()  # key: matches the
    # scheduler node currently being visited
    extern_buf_name = target_op.get_name()  # value: resolved via name_to_buf
    for dump_buf in dump_bufs:
        graph.additional_buffer_deps[extern_op_name].add(dump_buf.get_name())
    for restore_buf in restore_bufs:
        graph.additional_buffer_deps[restore_buf.get_operation_name()].add(
            extern_buf_name
        )


class LxContextSwitchingPass(ScratchpadOptimizationPass):
    """Bracket risky FallbackKernel calls with per-buffer LX dump/restore.

    Registered as a ScratchpadAllocator post_optimization_pass by
    select_allocator() (allocator.py), gated on
    config.enable_lx_context_switching -- see the matching comments on
    PR3683's _extern_kernel_in_live_range residency guard, which this pass is
    the intended long-term replacement for.
    """

    def apply_pass(self, graph: GraphLowering) -> None:
        if not config.enable_lx_context_switching:
            return

        targets = _select_bracket_targets(graph)
        if not targets:
            return

        editor = GraphEditor(graph)
        for target_op, at_risk_bufs in targets:
            target_fx_node = list(target_op.origins)[0]
            dumped = _dump_buffers(
                graph, editor, target_op, target_fx_node, at_risk_bufs
            )
            restore_bufs = _restore_buffers(
                graph, editor, target_op, target_fx_node, at_risk_bufs, dumped
            )
            dump_bufs = [dump_buf for _, dump_buf in dumped.values()]
            _order_around_bracket(graph, target_op, dump_bufs, restore_bufs)
