# Copyright 2025 The Torch-Spyre Authors.
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

import logging
from collections import defaultdict
from typing import cast

import torch

from .constants import ELIDED_COPY_BACK_ATTR
from .ir import FixedTiledLayout, SpyreEmptyFallback
from .optimize_restickify import EdgeCostMap
from .logging_utils import get_inductor_logger
from .pass_utils import redirect_computed_buffer_reads
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    Operation,
    ReinterpretView,
    StorageBox,
    TensorBox,
)
from torch_spyre._C import SpyreTensorLayout
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet


logger = get_inductor_logger("insert_restickify")


def _fixed_tiled(layout: FixedLayout, stl: SpyreTensorLayout) -> FixedTiledLayout:
    return FixedTiledLayout(
        layout.device,
        layout.dtype,
        layout.size,
        layout.stride,
        stl,
        offset=layout.offset,
    )


def _record_restickify(
    op: Operation,
    dep_name: str,
    target_layout: FixedTiledLayout,
    restickify_plan: dict,
) -> None:
    """Record that op's input arg_name must be restickified to target_layout.

    restickify_plan is the deferred execution queue: entries are recorded here during
    finalize_layouts and executed later by insert_restickify.
    """
    restickify_plan[op.get_name()].append(
        {"arg_name": dep_name, "target_layout": target_layout}
    )


def _create_restickify_node(
    restick_arg_info: dict, op: ComputedBuffer
) -> tuple[str, ComputedBuffer]:
    """
    Lower a restickify FX node for the given incompatible input arg.

    Inserts a spyre.restickify call into the FX graph, lowers it via
    graph_lowering.run_node(), and assigns the target layout.  Returns
    (old_buffer_name, new_computed_buffer).

    For synthetically-created buffers that have no FX node (e.g.
    coarse_tile_read_copy_* buffers created by coarse_tile.py), the FX env
    lookup is skipped and lower_restickify is called directly with a TensorBox
    wrapping the ComputedBuffer.
    """
    from .lowering import (
        lower_restickify,
    )  # deferred: lowering.py imports insert_restickify at module level

    arg_name = restick_arg_info["arg_name"]

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    # View ops (e.g. permute) lower to ReinterpretView with no buffer name and
    # are absent from env. Patch env from name_to_users so the search below can
    # resolve them.
    env = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            if not tb.data.origins:
                continue
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)

    # Search env by buffer name to find the FX node to pass to restickify.
    fx_arg_node = next(
        (
            fx_node
            for fx_node, tb in graph_lowering.env.items()
            if isinstance(fx_node, torch.fx.Node)
            and isinstance(tb, TensorBox)
            and tb.get_name() == arg_name
        ),
        None,
    )

    if fx_arg_node is None:
        # Synthetically-created buffers (e.g. coarse_tile_read_copy_*) have no
        # FX node. Build a TensorBox from the buffer and call lower_restickify
        # directly; realize() inside lower_restickify registers the output in
        # graph.buffers and graph.operations.
        arg_buf = graph_lowering.get_buffer(arg_name)
        assert isinstance(arg_buf, ComputedBuffer), (
            f"_create_restickify_node: buffer {arg_name!r} not found in env and is "
            f"{type(arg_buf).__name__}, not ComputedBuffer — cannot restickify"
        )
        arg_tb = TensorBox(StorageBox(arg_buf))
        # Insert a synthetic FX node for origins — downstream code (e.g.
        # _single_arg_op_layout in propagate_layouts.py) requires non-empty origins.
        first_compute_node = next(n for n in fx_graph.nodes if n.op != "placeholder")
        with fx_graph.inserting_before(first_compute_node):
            restick_fx_node = fx_graph.create_node(
                "call_function", torch.ops.spyre.restickify.default, ()
            )
        with (
            IRNode.current_origins(OrderedSet([restick_fx_node])),
            V.set_current_node(restick_fx_node),
        ):
            restick_tb = lower_restickify(arg_tb)
        restick_buff = restick_tb.data.data  # TensorBox -> StorageBox -> ComputedBuffer
        assert isinstance(restick_buff, ComputedBuffer), (
            f"Expected ComputedBuffer, got {type(restick_buff).__name__}"
        )
        restick_buff.origins = OrderedSet([restick_fx_node])
        graph_lowering.env[restick_fx_node] = restick_tb
        restick_buff.layout = restick_arg_info["target_layout"]
        return arg_name, restick_buff

    # Insert at a valid position in the FX graph; the operations list order is
    # authoritative pre-scheduler, not position in the FX graph.
    first_compute_node = next(n for n in fx_graph.nodes if n.op != "placeholder")
    with fx_graph.inserting_before(first_compute_node):
        restick_fx_node = fx_graph.create_node(
            "call_function", torch.ops.spyre.restickify.default, (fx_arg_node,)
        )
    # Lower the FX node; run_node registers the output in graph.buffers and graph.operations.
    restick_tb = graph_lowering.run_node(restick_fx_node)
    restick_buff = restick_tb.data.data  # TensorBox -> StorageBox -> ComputedBuffer
    assert isinstance(restick_buff, ComputedBuffer), (
        f"Expected ComputedBuffer, got {type(restick_buff).__name__}"
    )
    # origins is empty by default since spyre.restickify has no ATen decomposition;
    # set it to the synthetic FX node so code that expects non-empty origins doesn't crash.
    restick_buff.origins = OrderedSet([restick_fx_node])
    graph_lowering.env[restick_fx_node] = restick_tb

    restick_buff.layout = restick_arg_info["target_layout"]

    return arg_name, restick_buff


def insert_restickify_on_node_inputs(
    op: ComputedBuffer,
    resticks_needed: list[dict],
    operations: list[Operation],
) -> None:
    """Insert restickify nodes before op for each incompatible input, patch op's inner_fn
    to read the new buffer names, and reconstruct the consumer ComputedBuffer to
    invalidate its sizes cache.
    """
    name_map = {}
    try:
        op_index = operations.index(op)
    except ValueError:
        raise AssertionError(
            f"Consumer op {op.get_name()} not found in operations list"
        ) from None

    for restick_arg_info in resticks_needed:
        old_name, restick_buff = _create_restickify_node(restick_arg_info, op)
        name_map[old_name] = restick_buff.get_name()

        # lower_restickify calls pw.realize() which appends restick_buff to operations.
        # Move it to just before the consumer op to preserve topological order.
        operations.remove(restick_buff)
        operations.insert(op_index, restick_buff)
        op_index += 1  # consumer shifted right by 1

        # When coarse-tiling runs pre-stickification, the consumer op already
        # carries loop_info (loop_group_id + loop_count).  The restickify node
        # is inserted inside the same loop group, so it must inherit loop_info
        # to remain contiguous in build_loop_scheduler_nodes.
        if hasattr(op, "loop_info"):
            restick_buff.loop_info = op.loop_info

    # Patch inner_fn once with the full name_map covering all restickified args.
    redirect_computed_buffer_reads(
        op,
        name_map,
        operations,
        pass_name="insert_restickify",
        reason="redirect consumer to restickified input",
    )


def insert_restickify(graph: GraphLowering) -> None:
    """Insert restickify operations before all nodes in restickify_plan.

    Consumes graph.restickify_plan (built by finalize_layouts) and splices the
    necessary ComputedBuffer nodes into the operations list in-place.
    No scheduler state is touched.
    """
    operations = graph.operations
    restickify_plan = graph.restickify_plan
    if not restickify_plan:
        return

    for op in list(
        operations
    ):  # copy since insert_restickify_on_node_inputs mutates operations
        if isinstance(op, ComputedBuffer) and op.get_name() in restickify_plan:
            insert_restickify_on_node_inputs(
                op, restickify_plan[op.get_name()], operations
            )


def finalize_layouts(graph: GraphLowering) -> None:
    """Convert committed STLs (set by the optimizer) to FixedTiledLayouts and build
    graph.restickify_plan for insert_restickify.

    Two steps:
    - Commit: wrap each op's committed_stl in a FixedTiledLayout and assign it to
      op.layout; clean up optimizer-only attributes (layouts, restick_cost_fn,
      committed_stl).
    - Schedule restickifies: for each input edge where the committed input STL is
      incompatible with what the op requires, record a restickify in the plan.
    """
    operations = graph.operations
    for name in graph.graph_input_names:
        tensor_box = graph.graph_inputs[name]
        if (
            isinstance(tensor_box, TensorBox)
            and isinstance(tensor_box.data, StorageBox)
            and isinstance(tensor_box.data.data, InputBuffer)
            and hasattr(tensor_box, "layouts")
        ):
            input_buf = tensor_box.data.data
            assert hasattr(input_buf, "committed_stl"), (
                f"graph input {name} has no committed_stl — optimizer did not run"
            )
            stl = input_buf.committed_stl
            input_buf.layout = _fixed_tiled(input_buf.layout, stl)
            del tensor_box.layouts

    plan: dict = defaultdict(list)

    for op in operations:
        cost_fn = getattr(op, "restick_cost_fn", None)
        op_layouts = getattr(op, "layouts", None)
        committed = getattr(op, "committed_stl", None)
        for attr in ("layouts", "restick_cost_fn", "committed_stl"):
            if hasattr(op, attr):
                delattr(op, attr)

        # Commit the chosen STL and wrap in a FixedTiledLayout
        # Exclude mutation ops because their op.layout must not be set
        # until after the scheduler runs
        if op_layouts and not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            stl = committed if cost_fn else op_layouts[0]
            op.layout = _fixed_tiled(op.layout, cast(SpyreTensorLayout, stl))
            # Tiled-reduction scratch: propagate the reduction op's device
            # layout to accum_full so fill, combine, and copy all agree on
            # the device coordinate system.
            loop_info = getattr(op, "loop_info", None)
            if loop_info is not None and isinstance(op.layout, FixedTiledLayout):
                all_tiled_rdims_empty = all(
                    not dims
                    for dims in getattr(loop_info, "loop_tiled_reduction_dims", [])
                )
                if not all_tiled_rdims_empty:
                    # Propagate the reduction op's device layout to accum_full.
                    # Pre-stickify, _allocate_full_buffer assigned accum_full a
                    # generic layout; we now overwrite it with the same STL as
                    # the reduction op (they share the same output shape and
                    # stick orientation must agree for the combine to work).
                    accum_name = getattr(op, "_tiled_reduction_accum_name", None)
                    if accum_name is not None:
                        accum_buf = graph.get_buffer(accum_name)
                        accum_layout = accum_buf.layout
                        if isinstance(accum_layout, FixedTiledLayout):
                            # finalize_layouts already committed a generic STL
                            # (from propagate_spyre_tensor_layouts) to accum_full.
                            # Replace with the reduction op's actual STL so that
                            # fill, combine, and copy all agree on the device
                            # coordinate system.  Skip if already has the right
                            # STL (span-overflow path where _allocate_full_buffer
                            # already derived it from _resize_device_layout).
                            if accum_layout.device_layout != op.layout.device_layout:
                                accum_buf.layout = FixedTiledLayout(
                                    accum_layout.device,
                                    accum_layout.dtype,
                                    accum_layout.size,
                                    accum_layout.stride,
                                    op.layout.device_layout,
                                )
                        else:
                            # FixedLayout: wrap with the reduction op's STL.
                            accum_buf.layout = _fixed_tiled(
                                accum_layout, op.layout.device_layout
                            )

        # For each input edge, schedule a restickify if the input's committed STL
        # is incompatible with what this op requires on that edge.
        if not cost_fn:
            continue
        # Mutation ops targeting a SpyreEmptyFallback: the optimizer commits the
        # mutation op's output STL via AllSameNode (matching the new-value inputs).
        # The SpyreEmptyFallback was separately committed by AnyInNode (candidates[0]),
        # which may differ.  Overwrite the accumulator's FixedTiledLayout to match the
        # mutation op's committed STL so the backend sees consistent layouts.
        if isinstance(getattr(op, "layout", None), MutationLayoutSHOULDREMOVE):
            mut_target = op.layout.target
            while isinstance(mut_target, ReinterpretView):
                mut_target = mut_target.data
            mut_target_name = (
                mut_target.get_name() if hasattr(mut_target, "get_name") else ""
            )
            mut_target_buf = (
                graph.get_buffer(mut_target_name) if mut_target_name else None
            )
            if isinstance(mut_target_buf, SpyreEmptyFallback) and committed is not None:
                accum_layout = mut_target_buf.get_layout()
                if isinstance(accum_layout, (FixedTiledLayout, FixedLayout)):
                    new_layout = FixedTiledLayout(
                        accum_layout.device,
                        accum_layout.dtype,
                        accum_layout.size,
                        accum_layout.stride,
                        committed,
                    )
                    mut_target_buf.layout = new_layout
            elif isinstance(mut_target_buf, SpyreEmptyFallback) and committed is None:
                # committed_stl was cleaned up; fall back to the accumulator's layout.
                accum_layout = mut_target_buf.get_layout()
                if isinstance(accum_layout, FixedTiledLayout):
                    committed = accum_layout.device_layout
        for edge, target_stl in cost_fn.required_input_stls(committed):
            input_buf = graph.get_buffer(edge.dep.name)
            in_layout = input_buf.get_layout()
            if isinstance(in_layout, MutationLayoutSHOULDREMOVE):
                # Reading real_layout() through a mutation layout is only valid
                # once the target buffer's own layout is a committed
                # FixedTiledLayout. Three producers of this shape:
                #  - the copy-back elision optimization (propagate_layouts.py),
                #    which stamps ELIDED_COPY_BACK_ATTR on the producer; or
                #  - coarse_tile.py's nested output-dim + reduction-dim tiling
                #    (_insert_reduction_copy_op), which mutates directly into a
                #    SpyreEmptyFallback accumulator (accum_tile) — a legitimate
                #    in-group consumer (e.g. the next outer-tile iteration's
                #    copy-in) reads that copy op's own output the same way an
                #    ordinary producer's output would be read; or
                #  - coarse_tile.py's copy_out path for a MutationLayoutSHOULDREMOVE
                #    op whose target is a locally-created graph-output buffer
                #    (e.g. copy_forced(src, c) where c is returned directly) --
                #    _insert_copy_op's inserted coarse_tile_copy_* op reads the
                #    mutation op's own output the same way. The mutation target
                #    there is an ordinary ComputedBuffer, not a SpyreEmptyFallback,
                #    so this case is recognized by layout alone.
                mutation_target = in_layout.get_buffer()
                is_elided = getattr(input_buf, ELIDED_COPY_BACK_ATTR, False)
                is_committed_target = isinstance(
                    mutation_target.get_layout(), FixedTiledLayout
                )
                assert is_elided or is_committed_target, (
                    f"unexpected mutation layout on {edge.dep.name}"
                )
                in_layout = in_layout.real_layout()
            in_stl = in_layout.device_layout
            restick_stl = edge.layout(in_stl, target_stl)
            if restick_stl is None:
                continue
            if restick_stl is EdgeCostMap.INFEASIBLE:
                raise AssertionError(
                    f"finalize_layouts: restickify needed but infeasible for "
                    f"op={op.get_name()!r} input={edge.dep.name!r}: "
                    f"in_stl.stride_map={list(in_stl.stride_map)} "
                    f"target_stl.stride_map={list(target_stl.stride_map)}"
                )
            restick_target = _fixed_tiled(in_layout, restick_stl)
            logger.info(
                f"Injecting restickify on {op.get_name()} input {edge.dep.name}: "
                f"{list(in_stl.stride_map)} -> {list(target_stl.stride_map)}"
            )
            _record_restickify(op, edge.dep.name, restick_target, plan)

    V.graph.restickify_plan = plan
    if logger.isEnabledFor(logging.DEBUG):
        if plan:
            lines = ["restickify plan:"]
            for op_name, resticks in plan.items():
                consumer = graph.get_buffer(op_name)
                if isinstance(consumer, ComputedBuffer) and hasattr(
                    consumer.data, "reduction_type"
                ):
                    op_kind = f"reduction:{consumer.data.reduction_type}"
                elif isinstance(consumer, ComputedBuffer):
                    op_kind = "pointwise"
                else:
                    op_kind = type(consumer).__name__
                for r in resticks:
                    tgt = r["target_layout"]
                    arg_name = r["arg_name"]
                    arg_buf = graph.get_buffer(arg_name)
                    if (
                        isinstance(arg_buf, TensorBox)
                        and isinstance(arg_buf.data, StorageBox)
                        and isinstance(arg_buf.data.data, InputBuffer)
                    ):
                        buf_kind = "graph_input"
                    elif isinstance(arg_buf, ComputedBuffer):
                        buf_kind = "computed"
                    else:
                        buf_kind = type(arg_buf).__name__
                    lines.append(
                        f"  restickify {arg_name} ({buf_kind}) -> {op_name} ({op_kind})"
                        f"  stride_map={list(tgt.device_layout.stride_map)}"
                    )
            logger.debug("\n".join(lines))
        else:
            logger.debug("restickify plan: (none)")


def insert_post_mutation_restickify(graph: GraphLowering) -> None:
    """
    Insert pre/post ops around a slice mutation when the original layout cannot
    express the required stick offset.

    In that case, propagate_layouts picks an alternate layout and stores
    op._restickify_plan = (target_name, orig_stl, alt_stl). Because the
    restickify op cannot write its output in place, the mutation writes into a
    temporary buffer buf_tmp in alt_stl layout. This pass inserts:

      1. restickify op: arg0_1 (orig_stl) -> buf_tmp        (alt_stl)
      2. mutation op:   buf               -> buf_tmp[slice] (alt_stl)
      3. copy-back op:  buf_tmp (alt_stl) -> arg0_1         (alt_stl)
      4. set_spyre_tensor_layout(arg0_1, alt_stl)

    Both (1) and (3) are inserted as restickify IR nodes via
    _create_restickify_node. In (3), the input and output STLs are both alt_stl,
    so in the later codegen pass it reduces to an identity copy. Its layout is set
    to MutationLayoutSHOULDREMOVE(arg0_1), which makes the scheduler write the
    result back to arg0_1's original memory address.

    Both the mutation op and the copy-back use MutationLayoutSHOULDREMOVE to
    declare their write targets and reuse Inductor's existing mutation handling.

    The restickify op also records an input-layout override on
    buf_tmp._input_layout_overrides so work division and codegen both read
    arg0_1 using orig_stl for that op.

    arg0_1 is returned unchanged so DCI still reads from the same memory address.
    """
    operations = graph.operations
    tagged_ops = [op for op in operations if hasattr(op, "_restickify_plan")]
    if not tagged_ops:
        return

    for mutation_op in tagged_ops:
        target_name, orig_stl, alt_stl = mutation_op._restickify_plan
        del mutation_op._restickify_plan
        assert isinstance(mutation_op, ComputedBuffer)

        graph_input = graph.graph_inputs.get(target_name)
        assert graph_input is not None

        # Create fresh layouts here, since reusing base_layout would overwrite
        # arg0_1's address during hbm_pool_planning.
        target_input_buf = graph_input.data.data
        base_layout = target_input_buf.layout  # FixedTiledLayout(alt_stl)
        buf_tmp_layout = _fixed_tiled(base_layout, alt_stl)
        buf_copyback_layout = _fixed_tiled(base_layout, alt_stl)

        # Step 1: create restickify node: arg0_1 (orig_stl) -> buf_tmp (alt_stl)
        # This op must read arg0_1 as orig_stl, so record that override on buf_tmp.
        orig_stl_layout = _fixed_tiled(base_layout, orig_stl)
        _, buf_tmp = _create_restickify_node(
            {"arg_name": target_name, "target_layout": buf_tmp_layout},
            mutation_op,
        )
        buf_tmp_name = buf_tmp.get_name()
        buf_tmp._input_layout_overrides = {target_name: orig_stl_layout}

        # Step 2: retarget the mutation to buf_tmp while preserving the original slice offset.
        # A plain MutationLayoutSHOULDREMOVE(buf_tmp) would lose the offset; wrapping buf_tmp
        # in a ReinterpretView with the original slice layout keeps the offset and routes the
        # bytes into buf_tmp's allocation. This keeps the mutation on the standard mutation
        # path, where MutationLayoutSHOULDREMOVE buffers are not allocated by the wrapper.
        mutation_name = mutation_op.get_name()
        original_layout = mutation_op.layout
        assert isinstance(original_layout, MutationLayoutSHOULDREMOVE)
        slice_layout = original_layout.target.get_layout()
        # We only reach this pass because the write stick had a non-zero offset,
        # so the slice layout must carry it.
        assert slice_layout.offset != 0, (
            f"slice offset lost while retargeting mutation {mutation_name} "
            f"(target={type(original_layout.target).__name__}, "
            f"layout offset={slice_layout.offset!r}); the original slice offset "
            f"is not carried by the mutation target layout"
        )
        slice_view_of_buf_tmp = ReinterpretView(
            data=StorageBox(buf_tmp), layout=slice_layout
        )
        mutation_op.layout = MutationLayoutSHOULDREMOVE(slice_view_of_buf_tmp)

        # Step 3: create the copy-back node: buf_tmp (alt_stl) -> buf_copyback (alt_stl).
        # Since the input and output STLs are the same, this reduces to an identity copy
        # in the later codegen pass. MutationLayoutSHOULDREMOVE(arg0_1) makes this write
        # back to arg0_1's original storage and keeps the copy-back path live.
        _, buf_copyback = _create_restickify_node(
            {"arg_name": buf_tmp_name, "target_layout": buf_copyback_layout},
            mutation_op,
        )
        buf_copyback.layout = MutationLayoutSHOULDREMOVE(graph_input)

        # Anchor the set_spyre_tensor_layout emit on the mutation op, which
        # cannot be elided. The chain fuses into one kernel, so the emit fires
        # after the copy-back write-back.
        mutation_op._emit_set_layout = (target_name, alt_stl)

        # Insert buf_tmp before mutation, copy-back after mutation.
        # _create_restickify_node -> realize() appends buf_tmp/buf_copyback at
        # the end of operations, so they sit after mutation_op; removing them
        # first keeps the index arithmetic below correct.
        mutation_op_index = operations.index(mutation_op)
        operations.remove(buf_tmp)
        operations.insert(mutation_op_index, buf_tmp)
        # mutation_op is now at mutation_op_index + 1; insert copy-back after it.
        operations.remove(buf_copyback)
        operations.insert(mutation_op_index + 2, buf_copyback)

        logger.info(
            "insert_post_mutation_restickify: %s (orig->alt) before %s; copy-back %s->%s after %s",
            target_name,
            mutation_name,
            buf_tmp_name,
            target_name,
            mutation_name,
        )


def validate_no_restickify_on_mutation_targets(graph: GraphLowering) -> None:
    """Assert that no restickify was inserted on a mutation target buffer.

    A mutation op (MutationLayoutSHOULDREMOVE) writes directly into its target buffer.
    Restickifying that buffer would redirect the write to a temporary, silently breaking
    the in-place semantics.

    Must run after insert_restickify (so restickify_plan is populated) and before
    the scheduler (which resolves MutationLayoutSHOULDREMOVE to a concrete buffer
    address, after which mutation target identity is no longer recoverable).
    """
    assert hasattr(graph, "restickify_plan"), (
        "validate_no_restickify_on_mutation_targets must run after insert_restickify"
    )
    restickify_plan = graph.restickify_plan
    for op in graph.operations:
        if not isinstance(op, ComputedBuffer):
            continue
        layout = op.get_layout()
        if not isinstance(layout, MutationLayoutSHOULDREMOVE):
            continue
        target = layout.target
        while isinstance(target, ReinterpretView):
            target = target.data
        if not hasattr(target, "get_name"):
            continue
        target_name = target.get_name()
        for entry in restickify_plan.get(op.get_name(), []):
            if entry["arg_name"] == target_name:
                raise AssertionError(
                    f"restickify inserted on mutation target buffer {target_name!r} "
                    f"as input to its own mutation op {op.get_name()!r}"
                )
