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

import torch
from collections import defaultdict

from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import host_coordinates, device_coordinates
from .pass_utils import compute_restickify_target_layout
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    StorageBox,
    TensorBox,
)
from torch_spyre._C import SpyreTensorLayout
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet

from .errors import Unsupported

logger = get_inductor_logger("insert_restickify")


def _fixed_tiled(layout: FixedLayout, stl: SpyreTensorLayout) -> FixedTiledLayout:
    return FixedTiledLayout(
        layout.device, layout.dtype, layout.size, layout.stride, stl
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


class NameSwapHandler(WrapperHandler):
    """
    Wrapper to patch a node's inner_fn to use new buffer names after inserting
    nodes upstream that change the input buffers.
    """

    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


def _create_restickify_node(
    restick_arg_info: dict, op: ComputedBuffer
) -> tuple[str, ComputedBuffer]:
    """
    Lower a restickify FX node for the given incompatible input arg.

    Inserts a spyre.restickify call into the FX graph, lowers it via
    graph_lowering.run_node(), and assigns the target layout.  Returns
    (old_buffer_name, new_computed_buffer).
    """
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
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    )
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

    # Patch inner_fn once with the full name_map covering all restickified args.
    orig_inner = op.data.inner_fn

    def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig_inner(*args)

    object.__setattr__(op.data, "inner_fn", new_inner_fn)

    # Reconstruct ComputedBuffer as a fresh object so the instance-keyed cache
    # on get_default_sizes_body can be cleanly invalidated below.
    new_consumer_buffer = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=op.data,
        _split_size=op._split_size,
        _original_inner_fn=op._original_inner_fn,
        _original_ranges=op._original_ranges,
        _original_reduction_ranges=op._original_reduction_ranges,
    )
    new_consumer_buffer.operation_name = op.operation_name
    new_consumer_buffer.origins = op.origins
    # Replace op in the operations list with the reconstructed buffer.
    operations[op_index] = new_consumer_buffer
    V.graph.name_to_buffer[new_consumer_buffer.get_name()] = new_consumer_buffer

    # Invalidate the sizes/body cache so it is recomputed on next access with the patched inner_fn.
    ComputedBuffer.get_default_sizes_body.clear_cache(new_consumer_buffer)


def insert_restickify(operations: list[Operation]) -> None:
    """Insert restickify operations before all nodes in restickify_plan.

    Consumes V.graph.restickify_plan (built by finalize_layouts) and splices the
    necessary ComputedBuffer nodes into the operations list in-place.
    No scheduler state is touched.
    """
    restickify_plan = V.graph.restickify_plan
    if not restickify_plan:
        return

    for op in list(
        operations
    ):  # copy since insert_restickify_on_node_inputs mutates operations
        if isinstance(op, ComputedBuffer) and op.get_name() in restickify_plan:
            insert_restickify_on_node_inputs(
                op, restickify_plan[op.get_name()], operations
            )


def finalize_layouts(operations: list) -> None:
    """Convert committed STLs (set by the optimizer) to FixedTiledLayouts and build
    V.graph.restickify_plan for insert_restickify.

    Three steps:
    - Commit: wrap each op's committed_stl in a FixedTiledLayout and assign it to
      op.layout; clean up optimizer-only attributes (layouts, restick_cost_fn,
      committed_stl).
    - Schedule restickifies: for each input edge where the committed input STL is
      incompatible with what the op requires, record a restickify in the plan.
    - Mutation ops: check inputs of MutationLayoutSHOULDREMOVE ops and schedule
      restickifies where the input stick doesn't match the target buffer's stick.
    """
    for name in V.graph.graph_input_names:
        tensor_box = V.graph.graph_inputs[name]
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

    restickify_plan: dict = defaultdict(list)

    for op in operations:
        cost_fn = getattr(op, "restick_cost_fn", None)
        op_layouts = getattr(op, "layouts", None)
        committed = getattr(op, "committed_stl", None)
        for attr in ("layouts", "restick_cost_fn", "committed_stl"):
            if hasattr(op, attr):
                delattr(op, attr)

        # Commit the chosen STL and wrap in a FixedTiledLayout
        if op_layouts and not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            stl = committed if cost_fn else op_layouts[0]
            op.layout = _fixed_tiled(op.layout, stl)

        # For each input edge, schedule a restickify if the input's committed STL
        # is incompatible with what this op requires on that edge.
        if not cost_fn:
            continue
        for edge, target_stl in cost_fn.required_input_stls(committed):
            input_buf = V.graph.get_buffer(edge.dep.name)
            in_layout = input_buf.get_layout()
            in_stl = in_layout.device_layout
            restick_stl = edge.layout(in_stl, target_stl)
            if restick_stl is None:
                continue
            restick_target = _fixed_tiled(in_layout, restick_stl)
            logger.info(
                f"Injecting restickify on {op.get_name()} input {edge.dep.name}: "
                f"{list(in_stl.stride_map)} -> {list(target_stl.stride_map)}"
            )
            _record_restickify(op, edge.dep.name, restick_target, restickify_plan)

    # Handle mutation ops: check if their inputs need restickifying to match target buffer's stick.
    for op in operations:
        if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            continue
        target_layout = op.layout.target.get_layout()
        assert isinstance(target_layout, FixedTiledLayout), (
            f"mutation op {op.get_name()} target has no committed FixedTiledLayout"
        )
        target_stl = target_layout.device_layout
        read_writes = op.get_read_writes()
        output_dep = next(iter(read_writes.writes))
        for dep in read_writes.reads:
            if not isinstance(dep, MemoryDep):
                continue
            input_buf = V.graph.get_buffer(dep.name)
            in_layout = input_buf.get_layout()
            if not isinstance(in_layout, FixedTiledLayout):
                continue
            in_stl = in_layout.device_layout
            host_coords = host_coordinates(in_layout, dep)
            device_coords = device_coordinates(in_stl, dep)
            target_stick_expr = device_coordinates(target_stl, output_dep)[-1]
            in_stick_expr = device_coords[-1]

            if in_stick_expr == target_stick_expr:
                continue
            restick_stl = compute_restickify_target_layout(
                in_stl, in_layout, target_stick_expr, host_coords, device_coords
            )
            if restick_stl is None:
                raise Unsupported(
                    f"mutation op {op.get_name()} arg={dep.name}: cannot restickify "
                    f"{list(in_stl.stride_map)} -> {list(target_stl.stride_map)}"
                )
            restick_target = _fixed_tiled(in_layout, restick_stl)
            logger.info(
                f"Injecting restickify on {op.get_name()} input {dep.name}: "
                f"{list(in_stl.stride_map)} -> {list(target_stl.stride_map)}"
            )
            _record_restickify(op, dep.name, restick_target, restickify_plan)

    V.graph.restickify_plan = restickify_plan
