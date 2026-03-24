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

from typing import Any, Sequence, Union

from torch._inductor.utils import IndentedBuffer
from torch._inductor.utils import (
    get_kernel_metadata,
    get_fused_kernel_name,
    sympy_product,
)
from torch._inductor.scheduler import (
    BaseScheduling,
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V
from torch._inductor.codecache import code_hash
from torch.utils._ordered_set import OrderedSet

from .spyre_kernel import SpyreKernel
from .pass_utils import iteration_space


class SuperDSCScheduling(BaseScheduling):
    kernel_type: type[Any] = SpyreKernel
    dsc_type: str = "sdsc"

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def flush(self):
        pass

    def ready_to_flush(self) -> bool:
        return False

    def can_buffer_be_removed_through_fusion(
        self, name: str, fused_node_names: OrderedSet[str]
    ) -> bool:
        """We need intermediate buffers to be allocated even if only used within a single Kernel"""
        return False

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        return False

    def codegen_node_schedule_with_kernel(
        self, node_schedule: Sequence[BaseSchedulerNode], kernel: SpyreKernel
    ):
        with kernel:
            for node in node_schedule:
                if isinstance(node, SchedulerNode):
                    var_ranges = iteration_space(node)
                    vars = list(var_ranges.keys())
                    index_vars = [
                        vars[: len(node._body.iter_vars)],
                        vars[len(node._body.iter_vars) :],
                    ]
                    node.codegen(index_vars)
                else:
                    raise RuntimeError(f"TODO: implement for {node}")

    def generate_node_schedule(self, nodes: Sequence[BaseSchedulerNode]):
        node_schedule: list[BaseSchedulerNode] = []
        done = OrderedSet[BaseSchedulerNode]()
        for node in nodes:
            if node in done:
                continue
            done.add(node)
            node_schedule.append(node)
        return node_schedule

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:
        assert self.scheduler
        nodes = [
            node
            for node in node.get_nodes()
            if node.get_name() not in self.scheduler.removed_ops
        ]
        if len(nodes) == 0:
            return

        node_schedule = self.generate_node_schedule(nodes)
        kernel = SpyreKernel()
        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
        kernel_name = self.define_kernel(src_code, node_schedule, kernel)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        with V.set_kernel_handler(kernel):
            for node in node_schedule:
                node.mark_run()

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel.kernel_name)

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

        self.free_buffers_in_scheduler()

    def define_kernel(self, src_code, node_schedule, kernel):
        """Codegen kernel definition to go in output wrapper code"""
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = get_fused_kernel_name(node_schedule, "original_aten")
            kernel_name = "_".join(
                [self.dsc_type, fused_name, wrapper.next_kernel_suffix()]
            )
            wrapper.src_to_kernel[src_code] = kernel_name
            buf = IndentedBuffer()
            buf.writeline(f"async_compile.{self.dsc_type}('{kernel_name}',")
            with buf.indent():
                buf.splice(f"{src_code}")
            buf.writeline(")")
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(kernel_name, buf.getvalue(), metadata_comment)

        return kernel_name
