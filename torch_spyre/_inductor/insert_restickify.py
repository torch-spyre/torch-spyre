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

from .logging_utils import get_inductor_logger
from torch._inductor.ir import ComputedBuffer, Operation, TensorBox
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet

logger = get_inductor_logger("insert_restickify")


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
            "call_function", torch.ops.spyre.restickify, (fx_arg_node,)
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
    """Create a restickify node for each incompatible input arg of op."""
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

    # Invalidate the sizes/body cache so it is recomputed on next access with the patched inner_fn.
    ComputedBuffer.get_default_sizes_body.clear_cache(new_consumer_buffer)


def insert_restickify(operations: list[Operation]) -> None:
    """
    Insert restickify operations before all nodes in restickify_plan.

    Consumes V.graph.restickify_plan (built by propagate_spyre_tensor_layouts)
    and splices the necessary ComputedBuffer nodes into the operations list
    in-place.  No scheduler state is touched.
    """
    restickify_plan = getattr(V.graph, "restickify_plan", {})
    if not restickify_plan:
        return

    for op in list(
        operations
    ):  # copy since insert_restickify_on_node_inputs mutates operations
        if isinstance(op, ComputedBuffer) and op.get_name() in restickify_plan:
            insert_restickify_on_node_inputs(
                op, restickify_plan[op.get_name()], operations
            )
