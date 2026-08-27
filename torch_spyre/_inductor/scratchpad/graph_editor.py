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
from torch_spyre._inductor.pass_utils import (
    commit_iteration_space_ownership,
    copy_op_metadata,
    iteration_space_from_op,
    invalidate_op_read_writes,
    op_read_writes,
)
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
                    f"unexpected buffer type {type(buffer)} while replacing '{old_name}' ({buffer})"
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
        buffer_users: list[Operation],
        *,
        input: bool,
        private: bool = False,
    ) -> ComputedBuffer:
        """Insert a clone; private clones rewire only ``buffer_users``."""
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
        if private:
            old_users = list(
                dict.fromkeys(
                    getattr(consumer, "origin_node", None)
                    or next(iter(consumer.origins))
                    for consumer in buffer_users
                )
            )
        self.fx_graph.inserting_after(buf_fx)
        new_fx_node = self.fx_graph.create_node(
            "call_function", self.clone_aten_op, (buf_fx,)
        )
        for user in old_users:
            user.replace_input_with(buf_fx, new_fx_node)
        self.lowering.orig_gm.recompile()

        layout = buffer.layout
        assert isinstance(layout, FixedTiledLayout)
        clone_layout = FixedTiledLayout(
            layout.device,
            layout.dtype,
            list(layout.size),
            list(layout.stride),
            layout.device_layout,
            offset=layout.offset,
        )
        # Input buffers have no loop metadata, so input clones inherit it from
        # their consumer. Output clones inherit it from their producer.
        metadata_source = buffer_users[0] if input else buffer
        assert isinstance(metadata_source, ComputedBuffer)
        clone_tb = clone_lowering(buffer)
        new_com_buf = ComputedBuffer(
            name=None,
            layout=clone_layout,
            data=clone_tb.data.data,  # type: ignore[union-attr]
        )
        new_com_buf.data.origins.add(new_fx_node)
        new_com_buf.origins.add(new_fx_node)
        new_com_buf.origin_node = new_fx_node
        copy_op_metadata(metadata_source, new_com_buf)
        new_com_buf.name = self.lowering.register_buffer(new_com_buf)
        self.lowering.register_operation(new_com_buf)
        new_buf_name = new_com_buf.get_name()

        # Clone loops mirror their source/consumer symbols before Scheduler, so
        # retain direct symbol ownership instead of round-tripping through index
        # coefficients. A clone has no reduction split.
        metadata_owner = getattr(metadata_source, "iteration_space_ownership", None)
        if input and metadata_owner is not None:
            read = next(
                (
                    dep
                    for dep in op_read_writes(metadata_source).reads
                    if dep.name == buf_name
                ),
                None,
            )
            clone_write = next(iter(op_read_writes(new_com_buf).writes))
            by_coeff = {
                read.index.coeff(sym): split
                for sym, split in metadata_owner.work_slices.items()
                if read is not None and read.index.coeff(sym) != 0
            }
            clone_splits = {
                sym: by_coeff.get(clone_write.index.coeff(sym), 1)
                for sym in iteration_space_from_op(new_com_buf)
            }
        else:
            clone_splits = {
                sym: metadata_owner.work_slices.get(sym, 1) if metadata_owner else 1
                for sym in iteration_space_from_op(new_com_buf)
            }
        commit_iteration_space_ownership(new_com_buf, clone_splits)

        if input:
            source_users = []
            clone_users = []
            private_user_names = {user.get_name() for user in buffer_users}
            for node in self.lowering.name_to_users[buf_name]:
                while not isinstance(node, Buffer):
                    assert hasattr(node, "data"), (
                        f"unexpected node type {type(node)} ({node})"
                    )
                    node = node.data
                keep_source = node.name in [buf_name, new_buf_name] or (
                    private and node.name not in private_user_names
                )
                if keep_source:
                    source_users.append(node)
                else:
                    clone_users.append(node)
            self.lowering.name_to_users[buf_name] = source_users
            self.lowering.name_to_users[new_buf_name] = clone_users

            for consumer in buffer_users:
                if GraphEditor.is_rewritable_consumer(consumer):
                    self._replace_loop_input(consumer, buf_name, new_buf_name)
                else:
                    raise NotImplementedError(
                        f"unexpected buffer user type {type(consumer)} ({consumer})"
                    )

        self.lowering.operations.remove(new_com_buf)
        self.lowering.operations.insert(
            self.lowering.operations.index(buffer_users[0]), new_com_buf
        )

        return new_com_buf

    def insert_clone_before_consumers(
        self,
        buffer: ComputedBuffer,
        consumers: list[ComputedBuffer],
    ) -> ComputedBuffer:
        return self.push_allocation_with_clone(
            buffer, consumers, input=True, private=True
        )

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

    def _replace_loop_input(
        self, old_loop: Operation, old_name: str, new_name: str
    ) -> None:
        """Replace one buffer load in a pointwise or reduction loop."""
        assert isinstance(old_loop.data, Pointwise | Reduction)
        new_loop = self._create_loop_hack_inner_fn(
            old_loop.data, name_map={old_name: new_name}
        )
        old_loop.data = new_loop
        # The dependency set changed; force the next query to retrace the loop.
        invalidate_op_read_writes(old_loop)

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
