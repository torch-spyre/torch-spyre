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

import math
from abc import ABC, abstractmethod
from typing import Callable

from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.lowering import clone as clone_lowering, lowerings
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V

from ..ir import FixedTiledLayout, TensorBox

from torch_spyre._inductor.scratchpad.utils import (
    OP_OUTPUT_GOOD_FOR_LX_REUSE,
    get_buffer_users,
    get_ncores_for_buffers,
)


class ScratchpadOptimizationPass(ABC):
    """
    Abstract class for optimization passes which are implemented to improve
    a graph's overall scratchpad memory utilization and/or memory latency.
    """

    @abstractmethod
    def apply_pass(self, graph: GraphLowering):
        """
        Accepts a candidate graph to be optimized and evaluated for scratchpad memory allocation.
        `graph` will be mutated according in an implementation defined way. The order and
        number of nodes in the graph may change as a result of an optimization pass.

        Args:
            graph (GraphLowering): The graph to be optimized for scratchpad memory allocation
        """
        pass


class _NameSwapHandler(WrapperHandler):
    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


class CloneInputNodesPass(ScratchpadOptimizationPass):
    def __init__(self, limit: int):
        self.limit = limit

    def _create_loop_hack_inner_fn(self, old_Loop, name_map):
        """Use ops_handler to swap the name of buffers"""

        def new_inner_fn(*args):
            # Pointwise has 1 pos arg index while Reduction has 2, i.e. (index, rindex)
            with V.set_ops_handler(_NameSwapHandler(V.ops, name_map)):
                return old_Loop.inner_fn(*args)

        # old_Loop could be a Pointwise or Reduction.
        kwargs = {k: getattr(old_Loop, k) for k in old_Loop.__dataclass_fields__.keys()}
        kwargs["inner_fn"] = new_inner_fn
        new_Loop = old_Loop.__class__(**kwargs)
        # Additional attr that are not included in dataclass_fields. NOTE it relies on a
        # special method to force reset attrs of a frozen dataclass, see ir.Loops.create()
        new_Loop._post_init_setattr("origins", old_Loop.origins)
        new_Loop._post_init_setattr("origin_node", old_Loop.origin_node)
        new_Loop._post_init_setattr("traceback", old_Loop.traceback)
        # .get_stack_traces() get info from "origins", no need to manually set anything
        # LoopBody will be created later when we call CompBuf.recompute()

        return new_Loop

    def _insert_op_after(
        self,
        graph,
        buf: TensorBox,
        lowering_func: Callable,
        buf_users: dict,
        operations: list[Operation],
    ) -> None:
        """
        Insert an operation using the provided lowering function (e.g. clone_lowering) in
        GraphLowering.operations list after the given op (buf, a TensorBox representing a
        ComputedBuffer). Will update GraphLowering FX graph and the operations list.
        For example, original ops list looks like:
            buf0 -> buf1 -> buf2
        insert a clone of buf0 and let buf1 read from it, will become
            buf0 ->(clone) buf3 -> buf1 ->buf2

        NOTE:
        - Simplified flow, everything is done before Scheduler. Only take care of FX and
          GraphLowering. list operations will be updated inplace, no need to return.
        - Even though it is not a necessary condition, we assume FX graph and Operations are
          fully consistent and we will try to maintain it that way.
        - To update existing users of the old buffer -> hack the inner_fn then refresh LoopIR
        """

        # Step 1: Add a new FX node for clone and update dependencies
        buf_name = buf.data.data.name  # buf is a TensorBox
        buf_fx = list(buf.origins)[0]  # .origin_node may not exist
        old_users = list(buf_fx.users.keys())
        # make sure the user-provided lowering_func is legit
        assert lowering_func in lowerings.values(), (
            f"The provided lowering function {lowering_func} is not properly registered."
        )
        LUTlower_func_to_op = {func: aten_op for aten_op, func in lowerings.items()}
        user_aten_op = LUTlower_func_to_op[lowering_func]
        # TODO this is a large dict, move it to upper scope so we only need to do it once
        # aten_op is the overloaded version, e.g. ops.aten.clone.*out* instead of .default
        fx_graph = graph.graph
        fx_graph.inserting_after(buf_fx)
        new_fx_node = fx_graph.create_node("call_function", user_aten_op, (buf_fx,))
        for user in old_users:
            user.args = tuple(new_fx_node if ar is buf_fx else ar for ar in user.args)
        graph.orig_gm.recompile()

        # Step 2: Create a new ComBuf of a Pointwise IR (need to support Reduction?)
        pw_ir_tb = lowering_func(buf)  # a TensorBox wrapping a PointwiseIR
        new_com_buf = ComputedBuffer(
            name=None,
            layout=FixedTiledLayout(
                buf.layout.device,
                buf.layout.dtype,
                buf.layout.size,
                buf.layout.stride,
                buf.layout.device_layout,
            ),  # create a new copy of FixedTiledLayout from buf's layout
            data=pw_ir_tb.data.data,
        )
        new_com_buf.origins.add(new_fx_node)
        new_com_buf.origin_node = new_fx_node
        # TODO why arg0 ComputedBuffer doesn't have this attr?
        new_com_buf.name = graph.register_buffer(new_com_buf)
        graph.register_operation(new_com_buf)
        new_buf_name = new_com_buf.name

        # Step 3: Update graph_lowering.name_to_users (a list of TensorBox), eg, existing
        # users of arg0, other than InpBuf and new_buf, should become users of new_buf.
        users_of_inp, users_of_new_buf = [], []
        for tb in graph.name_to_users[buf_name]:
            if tb.data.data.name in [buf_name, new_buf_name]:
                users_of_inp.append(tb)
            else:
                users_of_new_buf.append(tb)
        graph.name_to_users[buf_name] = users_of_inp
        graph.name_to_users[new_buf_name] = users_of_new_buf

        # Step 3b: Propagate per-core splits to the clone.
        # Users share the same per_core_view (pre-checked by core_div_mismatch guard).
        # op_it_space_splits keys are stride-based coefficients, which are layout-
        # invariant, so the first user's output_splits transfer without re-keying.
        # Reduction splits are dropped: the clone is Pointwise with no reduction axis.
        first_user = buf_users[buf_name][0]
        user_out_splits, _ = getattr(first_user, "op_it_space_splits", ({}, {}))
        new_com_buf.op_it_space_splits = (user_out_splits, {})

        # Step 4: Hack user nodes' inner_fn
        for old_com_buf in buf_users[buf_name]:
            # hack inner_fn with a nameSwapper ops handler and make a new LoopIR
            new_Loop = self._create_loop_hack_inner_fn(
                old_com_buf.data, name_map={buf_name: new_buf_name}
            )
            old_com_buf.data = new_Loop

        # NOTE: operations is a reference to graph_lowering.operations, which is already
        # updated when we call graph_lowering.register_operation() earlier. But the new Op
        # was appended at the end of the list, need to insert at the correct position.
        first_user = buf_users[buf_name][0]
        idx_to_first_user = operations.index(first_user)
        operations.remove(new_com_buf)
        operations.insert(idx_to_first_user, new_com_buf)

    def _try_insert_clone_op_for_inputs(
        self,
        graph,
        operations: list[Operation],
        lx_free_total: int,
        buf_users: dict[str, Operation],
        num_cores_buf: dict[str, int],
    ) -> None:
        """
        Check if any input tensors can fit onto scratchpad and needed more than once =>
        Add corresponding "clone operation" to copy it to scratchpad and reduce HBM read.
        """
        for inp_name in graph.graph_input_names:
            buf = graph.get_buffer(inp_name)  # this is a TensorBox
            dev_layout = buf.layout.device_layout
            ncores_inp = num_cores_buf.get(inp_name, -1)
            core_div_mismatch = ncores_inp == -1
            dev_size_per_core = (
                math.prod(dev_layout.device_size[:-1]) * 128 // ncores_inp
            )
            is_on_lx = buf.layout.allocation != {}
            used_only_once = len(buf_users[inp_name]) == 1
            if (
                used_only_once
                or core_div_mismatch
                or dev_size_per_core > lx_free_total
                or is_on_lx
            ):
                continue

            self._insert_op_after(graph, buf, clone_lowering, buf_users, operations)

            lx_free_total -= dev_size_per_core

    def apply_pass(
        self,
        graph: GraphLowering,
    ) -> None:
        """
        Check if any input tensors can fit onto scratchpad and needed more than once =>
        Add corresponding "clone operation" to copy it to scratchpad and reduce HBM read.
        """
        buf_users = get_buffer_users(graph)

        operations = graph.operations
        if "clone" in OP_OUTPUT_GOOD_FOR_LX_REUSE:
            self._try_insert_clone_op_for_inputs(
                graph,
                operations,
                self.limit,
                buf_users,
                get_ncores_for_buffers(graph),
                # TODO consider cache some of the Op/buf dep table for later use. Only
                # need to refresh input/clone related entries. Because next step,
                # allocator._generate_buffers(), aka buf_analysis, will call it again.
            )
