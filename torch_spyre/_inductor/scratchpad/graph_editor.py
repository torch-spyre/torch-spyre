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

from torch.fx.graph import Graph
from torch._inductor.graph import GraphLowering
from torch._inductor.ops_handler import WrapperHandler
from torch_spyre._inductor.pass_utils import copy_op_metadata
from torch._inductor.virtualized import V
from torch._inductor.ir import (
    ComputedBuffer,
    TensorBox,
    StorageBox,
    Buffer,
    Operation,
    Pointwise,
    Reduction,
)
from torch._inductor.lowering import clone as clone_lowering, lowerings

from torch_spyre._inductor.ir import FixedTiledLayout


class GraphEditor:
    def __init__(self, lowering: GraphLowering):
        self.lowering = lowering
        self.fx_graph: Graph = lowering.graph  # type: ignore

        for aten_op, func in lowerings.items():
            if func == clone_lowering:
                self.clone_aten_op = aten_op
                break
        else:
            raise KeyError("could not find the clone lowering op")

    def _graph_output_name(self, buffer: TensorBox | StorageBox | Buffer) -> str:
        # graph_outputs can hold TensorBox, StorageBox, or Buffer depending on how
        # Inductor constructed the graph.
        while not isinstance(buffer, Buffer):
            buffer = buffer.data
        return buffer.name

    def _replace_matching_buffer(
        self,
        buffer: TensorBox | StorageBox | Buffer,
        old_name: str,
        i: int,
        new: ComputedBuffer | TensorBox,
    ) -> bool:
        """If `buffer`'s name matches `old_name`, then replace it with `new` and return True;
        otherwise, do nothing and return False.

        If `buffer` is a `TensorBox` (containing a `StorageBox`) or `StorageBox`, wrap `new` up in
        the same way. If `new` is a `TensorBox` itself, it is assumed to be wrapped up in an
        appropriate way."""
        fs = []
        while not isinstance(buffer, Buffer):
            if isinstance(buffer, TensorBox):
                fs.append(TensorBox)
            else:
                assert isinstance(buffer, StorageBox), (
                    f"unexpected buffer type {type(buffer)} ({buffer})"
                )
                fs.append(StorageBox)
            buffer = buffer.data

        if buffer.name == old_name:
            if not isinstance(new, TensorBox):
                for f in fs[::-1]:
                    new = f(new)
            self.lowering.graph_outputs[i] = new
            return True
        else:
            return False

    def change_graph_output(
        self, old: ComputedBuffer | TensorBox, new: ComputedBuffer | TensorBox
    ) -> None:
        old_name = self._graph_output_name(old)
        for i, buffer in enumerate(self.lowering.graph_outputs):
            if self._replace_matching_buffer(buffer, old_name, i, new):
                return

        raise KeyError(f"could not find buffer {old_name} to replace as output")

    def push_allocation_with_clone(
        self,
        buffer: ComputedBuffer | TensorBox,
        address: int,
        buffer_users: list[Operation],
        *,
        input: bool,
    ) -> ComputedBuffer:
        """
        Insert an operation using clone_lowering in GraphLowering.operations list after the given op
        (buffer, a TensorBox or a ComputedBuffer). Will update GraphLowering FX graph and the
        operations list. If update_downstream is True, then update any operations following the
        given buffer to refer to the clone instead.

        The clone will be inserted after buf. For example, the original ops list may look like this
        (with buf0 taking the role of buf):
            buf0 -> buf1 -> buf2
        Insert a clone of buf0 and let buf1 read from it: this will become
            buf0 ->(clone) buf3 -> buf1 ->buf2

        Returns the new buffer so that either it or the original node can be allocated on the
        scratchpad.

        NOTE:
        - Even though it is not a necessary condition, we assume FX graph and Operations are fully
          consistent and we will try to maintain it that way.
        - To update existing users of the old buffer -> hack the inner_fn then refresh LoopIR.
        - TODO consider cache some of the Op/buf dep table for later use. Only
          need to refresh input/clone related entries. Because next step,
          allocator._generate_buffers(), aka buf_analysis, will call it again.
        """
        # Step 1: Add a new FX node for clone and update dependencies
        if isinstance(buffer, TensorBox):
            buf_name = buffer.data.data.name  # type: ignore
        else:
            assert isinstance(buffer, ComputedBuffer), (
                f"unexpected buffer type {type(buffer)} ({buffer})"
            )
            buf_name = buffer.name
        assert isinstance(buf_name, str)
        buf_fx = list(buffer.origins)[0]  # .origin_node may not exist
        old_users = list(buf_fx.users.keys())
        self.fx_graph.inserting_after(buf_fx)
        new_fx_node = self.fx_graph.create_node(
            "call_function", self.clone_aten_op, (buf_fx,)
        )
        for user in old_users:
            user.args = tuple(new_fx_node if ar is buf_fx else ar for ar in user.args)
        self.lowering.orig_gm.recompile()

        # Step 2: Create a new ComBuf of a Pointwise IR (need to support Reduction?)
        pw_ir_tb = clone_lowering(buffer)  # a TensorBox wrapping a PointwiseIR
        layout = buffer.layout
        assert isinstance(layout, FixedTiledLayout)
        new_com_buf = ComputedBuffer(
            name=None,
            layout=FixedTiledLayout(
                layout.device,
                layout.dtype,
                layout.size,  # pyright: ignore[reportArgumentType]
                layout.stride,  # pyright: ignore[reportArgumentType]
                layout.device_layout,
            ),  # create a new copy of FixedTiledLayout from buffer's layout
            data=pw_ir_tb.data.data,  # type: ignore
        )
        new_com_buf.origins.add(new_fx_node)
        new_com_buf.origin_node = new_fx_node
        # Copy loop-group metadata so the clone stays in the same coarse-tile
        # group as its neighbours and doesn't split a contiguous run.
        # For input clones the original buffer is an InputBuffer (no metadata),
        # so we inherit from the first consumer instead. For output clones the
        # original ComputedBuffer already carries the right metadata.
        copy_op_metadata(src=buffer_users[0] if input else buffer, dst=new_com_buf)
        # TODO why arg0 ComputedBuffer doesn't have this attr?
        new_com_buf.name = self.lowering.register_buffer(new_com_buf)
        self.lowering.register_operation(new_com_buf)
        new_buf_name = new_com_buf.name

        # Step 2b: Propagate per-core splits to the clone.
        # Users share the same per_core_view (pre-checked by core_div_mismatch guard).
        # op_it_space_splits keys are stride-based coefficients, which are layout-
        # invariant, so the first user's output_splits transfer without re-keying.
        # Reduction splits are dropped: the clone is Pointwise with no reduction axis.
        first_user = buffer_users[0]
        user_out_splits, _ = getattr(first_user, "op_it_space_splits", ({}, {}))
        new_com_buf.op_it_space_splits = (user_out_splits, {})

        if input:
            # Step 3: Update self.graph.name_to_users (a list of TensorBox), e.g., existing users of
            # arg0, other than InpBuf and new_buf, should become users of new_buf.
            users_of_inp = []
            users_of_new_buf = []
            for node in self.lowering.name_to_users[buf_name]:
                while not isinstance(node, Buffer):
                    assert hasattr(node, "data"), (
                        f"unexpected node type {type(node)} ({node})"
                    )
                    node = node.data
                if node.name in [buf_name, new_buf_name]:  # type: ignore
                    users_of_inp.append(node)
                else:
                    users_of_new_buf.append(node)
            self.lowering.name_to_users[buf_name] = users_of_inp
            self.lowering.name_to_users[new_buf_name] = users_of_new_buf

            # Step 4: Hack user nodes' inner_fn
            for old_com_buf in buffer_users:
                if GraphEditor.is_rewritable_consumer(old_com_buf):
                    self._swap_loops_input(old_com_buf, buf_name, new_buf_name)
                else:
                    raise NotImplementedError(
                        f"unexpected buffer user type {type(old_com_buf)} ({old_com_buf})"
                    )

        # NOTE: operations is a reference to graph.operations, which is already
        # updated when we call graph.register_operation() earlier. But the new Op
        # was appended at the end of the list, need to insert at the correct position.
        first_user = buffer_users[0]
        idx_to_first_user = self.lowering.operations.index(first_user)
        assert self.lowering.operations[-1].name == new_com_buf.name, (
            f"removing {new_com_buf.name} from "
            f"{[op.name for op in self.lowering.operations]}"
        )
        self.lowering.operations.pop()
        self.lowering.operations.insert(idx_to_first_user, new_com_buf)

        return new_com_buf

    @staticmethod
    def all_uses_are_rewritable(graph: GraphLowering, uses: list[int]) -> bool:
        return all(
            GraphEditor.is_rewritable_consumer(graph.operations[use]) for use in uses
        )

    @staticmethod
    def is_rewritable_consumer(op: Operation):
        """An op that wraps a Pointwise or Reduction.

        We encounter a FallbackKernel with some frequency, and that would be really useful to
        support as well. But the straightforward approach doesn't work, i.e.,

        def _swap_inputs_kernel_input(
            self, inputs_kernel: ir.InputsKernel, old_name: str, new_buffer: Buffer
        ):
            for i in range(len(inputs_kernel.inputs)):
                if inputs_kernel.input_name(i) == old_name:
                    inputs_kernel.inputs[i] = new_buffer
                    break

            inputs_kernel.get_free_symbol_uses.clear_cache(inputs_kernel)

        So instead we just allow ops that wrap a Pointwise or Reduction.
        """
        return hasattr(op, "data") and isinstance(op.data, Pointwise | Reduction)

    def _swap_loops_input(self, old_loop: Operation, old_name: str, new_name: str):
        """Hack inner_fn with a nameSwapper ops handler and make a new LoopIR."""
        assert isinstance(old_loop.data, Pointwise | Reduction)
        new_loop = self._create_loop_hack_inner_fn(
            old_loop.data, name_map={old_name: new_name}
        )
        old_loop.data = new_loop

    class _NameSwapHandler(WrapperHandler):
        def __init__(self, inner, name_map: dict[str, str]):
            super().__init__(inner)
            self._name_map = name_map

        def load(self, name, index):
            return super().load(self._name_map.get(name, name), index)

    def _create_loop_hack_inner_fn(
        self,
        old_loop: Pointwise | Reduction,
        name_map: dict[str, str],
    ) -> Pointwise | Reduction:
        """Use ops_handler to swap the name of buffers"""

        def new_inner_fn(*args):
            # Pointwise has 1 pos arg index while Reduction has 2, i.e. (index, rindex)
            with V.set_ops_handler(self._NameSwapHandler(V.ops, name_map)):
                return old_loop.inner_fn(*args)

        kwargs = {k: getattr(old_loop, k) for k in old_loop.__dataclass_fields__.keys()}
        kwargs["inner_fn"] = new_inner_fn
        new_loop = old_loop.__class__(**kwargs)
        # Additional attr that are not included in dataclass_fields. NOTE it relies on a
        # special method to force reset attrs of a frozen dataclas, see ir.Loops.create()
        new_loop._post_init_setattr("origins", old_loop.origins)
        new_loop._post_init_setattr("origin_node", old_loop.origin_node)
        new_loop._post_init_setattr("traceback", old_loop.traceback)
        # .get_stack_traces() get info from "origins", no need to manually set anything
        # LoopBody will be created later when we call CompBuf.recompute()

        return new_loop
