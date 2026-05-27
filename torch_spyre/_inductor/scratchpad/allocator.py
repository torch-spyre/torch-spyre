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

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch._inductor.ir import ComputedBuffer, Operation, MutationLayoutSHOULDREMOVE
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.passes import (
    CloneInputNodesPass,
    ScratchpadOptimizationPass,
)
from torch_spyre._inductor.scratchpad.utils import (
    OP_OUTPUT_GOOD_FOR_LX_REUSE,
    OP_GOOD_FOR_LX_INPLACE,
    mem_usage_by_buf,
    calculate_liveness,
    get_ncores_for_buffers,
    GraphView,
)

from torch_spyre._inductor import config


class ScratchpadAllocator(ABC):
    """
    Abstract class for all implementations of ScratchpadAllocator
    """

    @abstractmethod
    def plan_allocation(self, graph: GraphLowering):
        """
        Accepts a graph to be considered for scratchpad memory according
        to its composition and the specific implementation used.

        Args:
            graph (GraphLowering): Graph to be considered for scratchpad planning
        """
        pass

    def _get_op_name(self, op: Any) -> str:
        target = getattr(getattr(op, "origin_node", None), "target", None)
        org_op_name = (
            getattr(target, "_opname", None)
            or getattr(target, "__name__", None)
            or getattr(target, "name", None)
            or str(target)
        )
        return org_op_name

    def _op_output_good_for_lx_reuse(self, op: Any) -> bool:
        return (
            isinstance(op, ComputedBuffer)
            and not isinstance(op.layout, MutationLayoutSHOULDREMOVE)
            and (
                config.allow_all_ops_in_lx_planning
                or (self._get_op_name(op) in OP_OUTPUT_GOOD_FOR_LX_REUSE)
            )
        )

    def _op_good_for_lx_inplace(self, op: Any) -> bool:
        return self._get_op_name(op) in OP_GOOD_FOR_LX_INPLACE

    def _filter_ops(self, graph: GraphLowering) -> list[Operation]:
        core_div_mismatch = get_ncores_for_buffers(graph)
        drop_list = set()

        # filter out by permitted operations
        for op in graph.operations:
            if not self._op_output_good_for_lx_reuse(op):
                drop_list.add(op.name)

        # filter out core division mismatches
        drop_list.update(
            [key for key, mismatch in core_div_mismatch.items() if mismatch == -1]
        )

        # filter out the graph inputs and outputs. The inputs shouldn't appear here anyways.
        # These can be relaxed once node cloning is implemented as a post-solve optimization
        # rather than just filling the scratchpad at t = 0
        drop_list.update(graph.get_output_names())
        drop_list.update(graph.graph_input_names)

        return [op for op in graph.operations if op.name not in drop_list]

    def _build_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
    ) -> list[LifetimeBoundBuffer]:
        lifetimes = calculate_liveness(graph)
        mem_usage = mem_usage_by_buf(GraphView(graph, self._filter_ops))
        in_place = {} if in_place is None else in_place
        buffers = []
        for output_name, info in mem_usage.items():
            buffers.append(
                LifetimeBoundBuffer(
                    output_name,
                    info["size_per_core"],
                    lifetimes[output_name]["liveness_start"],
                    lifetimes[output_name]["liveness_end"],
                    in_place_parents=in_place.get(output_name, []),
                )
            )

        return buffers

    def _determine_in_place(self, graph: GraphLowering) -> dict[str, list[str]]:
        allow_inplace: dict[str, list[str]] = {}
        graph_view = GraphView(graph, self._filter_ops)
        mem_usage = mem_usage_by_buf(graph_view)
        in_place_allowed = {
            op.name: self._op_good_for_lx_inplace(op) for op in graph_view.operations
        }
        lifetimes = calculate_liveness(graph)
        for buf_name, info in mem_usage.items():
            allow_inplace[buf_name] = []
            if not in_place_allowed[buf_name]:
                continue
            out_start = lifetimes[buf_name]["liveness_start"]
            out_ten_layout = graph.get_buffer(buf_name).layout.device_layout
            out_size = info["size_per_core"]
            for input_buf in info["op_inputs"]:
                in_end = lifetimes[input_buf]["liveness_end"]
                in_ten_layout = graph.get_buffer(input_buf).layout.device_layout
                in_size = mem_usage[input_buf]["size_per_core"]
                inp_i_size_match = out_size == in_size
                inp_i_lay_match = out_ten_layout == in_ten_layout
                inp_i_eol = in_end == out_start + 1
                no_core_div_mismatch = not info["core_div_mismatch"]
                if (
                    inp_i_size_match
                    and inp_i_lay_match
                    and inp_i_eol
                    and no_core_div_mismatch
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _generate_buffers(self, graph: GraphLowering) -> list[Operation]:
        in_place = self._determine_in_place(graph)
        buffers = self._build_bound_buffers(graph, in_place)
        return buffers

    def _push_allocation(
        self, graph: GraphLowering, buffers: list[LifetimeBoundBuffer]
    ):
        # push the allocation into the code generation
        for b in buffers:
            if b.address is not None:
                buf = graph.get_buffer(b.name)
                layout = buf.get_layout()
                layout.allocation["lx"] = b.address


class DefaultAllocator(ScratchpadAllocator):
    def __init__(
        self,
        layout_planning: MemoryPlanSolver | None = None,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        """Configure the allocator with an optional solver and graph passes.

        Args:
            layout_planning: Solver that assigns LX addresses to lifetime-bound
                buffers. Defaults to GreedyLayoutSolver sized to available LX memory.
            pre_optimization_passes: Graph passes applied before layout planning.
                Defaults to [CloneInputNodesPass].
            post_optimization_passes: Graph passes applied after layout planning.
                Defaults to no passes.
        """
        size = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))
        if layout_planning is None:
            layout_planning = GreedyLayoutSolver(size)
        if pre_optimization_passes is None:
            pre_optimization_passes = [CloneInputNodesPass(size)]
        if post_optimization_passes is None:
            post_optimization_passes = []

        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning = layout_planning

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, assign LX addresses to eligible buffers, then run post-passes.

        Args:
            graph: Lowered graph whose buffers will be assigned LX scratchpad
                addresses where viable.
        """
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        buffers = self._generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers)
        self._push_allocation(graph, allocation)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)


def scratchpad_planning(
    graph: GraphLowering,
    allocator: Optional[ScratchpadAllocator] = None,
) -> None:
    """Assign LX scratchpad addresses to eligible buffers in a lowered graph.

    Called after stickification and core-division are complete. Graph operations
    are expected to be in topological order as guaranteed by GraphLowering.

    Args:
        graph: Lowered graph to plan scratchpad memory for.
        allocator: Allocator strategy to use. Defaults to DefaultAllocator.
    """
    if allocator is None:
        allocator = DefaultAllocator()
    allocator.plan_allocation(graph)
