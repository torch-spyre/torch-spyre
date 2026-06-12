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

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch._inductor.ir import (
    TensorBox,
    ComputedBuffer,
    Operation,
    MutationLayoutSHOULDREMOVE,
    Reduction,
    ExternKernel,
)
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    concretize_expr,
    iteration_space_from_op,
    splits_by_index_coeff,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.passes import (
    ScratchpadOptimizationPass,
)
from torch_spyre._inductor.scratchpad.utils import (
    OP_OUTPUT_GOOD_FOR_LX_REUSE,
    OP_GOOD_FOR_LX_INPLACE,
    clone_at_graph_boundaries,
    mem_usage_by_buf,
    calculate_liveness,
    get_ncores_for_buffers,
    get_buffer_users,
    GraphView,
)
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor

from torch_spyre._inductor import config

logger = logging.getLogger(__name__)


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
        target = getattr(getattr(op, "origin_node", None), "target", None)
        if target is None:
            return False
        if self._get_op_name(op) in OP_GOOD_FOR_LX_INPLACE:
            # If the op is in the whitelist, return true
            return True
        if torch.Tag.pointwise in target.tags:
            # If the op is tagged as pointwise by pytorch upstream
            # return True. Works only for unary ops
            return True
        return False

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

        if not clone_at_graph_boundaries():
            # Without clone support, graph outputs cannot be LX-pinned: the caller
            # holds an HBM reference and there is no clone to redirect it to.
            # graph_input_names is a no-op here (inputs are not in graph.operations),
            # but kept for symmetry with _build_bound_buffers, which handles inputs
            # separately when clone is available.
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
        graph_output_names = set(graph.get_output_names())
        cloning_allowed = clone_at_graph_boundaries()
        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]
            if len(uses) <= 1:
                continue  # output is not read (only the write, or never touched)
            if any(isinstance(graph.operations[u], ExternKernel) for u in uses):
                continue
            if output_name in graph_output_names and not cloning_allowed:
                continue  # we can only allocate graph outputs if we're allowed to clone
            uses = lifetimes[output_name]
            buffers.append(
                LifetimeBoundBuffer(
                    output_name,
                    info["size_per_core"],
                    uses[0],
                    uses[-1] + 1,
                    in_place_parents=in_place.get(output_name, []),
                )
            )

        if cloning_allowed:
            ncores = get_ncores_for_buffers(graph)
            for input_name in graph.graph_input_names:
                uses = lifetimes[input_name]
                if len(uses) <= 1:
                    # Input read only once, or not at all. A non-input that's read only once still
                    # saves a roundtrip to HBM if it is allocated in LX, but the input is already
                    # present in HBM and would need to be cloned to LX explicitly, which costs one
                    # transfer anyway.
                    continue
                if not GraphEditor.all_uses_are_rewritable(graph, uses):
                    continue
                num_cores = ncores.get(input_name, -1)
                if num_cores < 0:
                    continue  # core division mismatch across consumers
                buf = graph.get_buffer(input_name)
                dev_layout = buf.layout.device_layout
                dev_size = math.prod(dev_layout.device_size[:-1]) * 128
                buffers.append(
                    LifetimeBoundBuffer(
                        input_name,
                        dev_size // num_cores,
                        uses[0],
                        uses[-1] + 1,
                        in_place_parents=[],
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
            out_start = lifetimes[buf_name][0]
            out_ten_layout = graph.get_buffer(buf_name).layout.device_layout
            out_size = info["size_per_core"]
            for input_buf in info["op_inputs"]:
                in_end = lifetimes[input_buf][-1]  # inclusive last use
                in_ten_layout = graph.get_buffer(input_buf).layout.device_layout
                in_size = mem_usage[input_buf]["size_per_core"]
                inp_i_size_match = out_size == in_size
                inp_i_lay_match = out_ten_layout == in_ten_layout
                inp_i_eol = in_end == out_start  # same op reads input and writes output
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
        """Push the allocation into the code generation. This includes cloning graph inputs and
        graph outputs:

        - A graph input B that is allocated into LX means that it is cloned; call the clone C. The
        downstream users of B are now made to use C. The LX allocation is effectuated by assigning
        it to C.

        - A graph output B that is allocated into LX means that it is cloned; call the clone C.
        Nothing changes for the downstream users. The LX allocation is effectuated by assigning it
        to B itself. The graph is made to have C as its output.

        - A buffer that is neither a graph input nor a graph output gets the LX allocation assigned
        to itself."""
        outputs = set(graph.get_output_names())
        inputs = set(graph.graph_input_names)

        buffer_users = get_buffer_users(graph)
        graph_editor = GraphEditor(graph)

        for b in buffers:
            if b.address is None:
                continue

            buf = graph.get_buffer(b.name)
            if b.name in inputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, b.address, buffer_users[b.name], input=True
                )
                self._set_one_allocation(new_buffer, b.address)

            elif b.name in outputs:
                new_buffer = graph_editor.push_allocation_with_clone(
                    buf, b.address, buffer_users[b.name], input=False
                )
                self._set_one_allocation(buf, b.address)
                graph_editor.change_graph_output(buf, new_buffer)

            else:
                self._set_one_allocation(buf, b.address)

    def _set_one_allocation(self, buf: TensorBox | ComputedBuffer, address: int):
        layout = buf.get_layout()
        layout.allocation["lx"] = address


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
                Defaults to no passes.
            post_optimization_passes: Graph passes applied after layout planning.
                Defaults to no passes.
        """
        size = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))
        if layout_planning is None:
            if config.layout_solver == "greedy":
                layout_planning = GreedyLayoutSolver(size)
            elif config.layout_solver == "bestfit":
                layout_planning = BestFitLayoutSolver(size)
            elif config.layout_solver == "firstfit":
                layout_planning = FirstFitLayoutSolver(size)
            else:
                raise ValueError(
                    f"Invalid layout_solver config option '{config.layout_solver}'."
                )
        if pre_optimization_passes is None:
            pre_optimization_passes = []
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


DEFAULT_VARIANT_CAP = 6


def _enum_split_options(op: Operation) -> list[tuple[dict, dict]]:
    """Generate split options based on the seed (current committed
    split) by flipping the split factor onto a different output dim.
    Returns ≤ DEFAULT_VARIANT_CAP options with the seed at index 0. If
    the seed is unsplit or reduction-axis-only, returns the seed alone.
    """
    seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    output_splits, reduction_splits = seed
    if not output_splits or not isinstance(op, ComputedBuffer):
        return [seed]

    # Reduction ops: don't flip for now.
    if isinstance(op.data, Reduction):
        return [seed]

    # Recover seed's per-symbol form to mutate the slicing.
    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    first_read = next(iter(rw.reads), None)
    read_index = first_read.index if first_read is not None else write_index
    iter_space = iteration_space_from_op(op)
    seed_per_sym = apply_splits_from_index_coeff(
        seed, write_index, read_index, iter_space
    )

    # Only single output-dim splits are flipped. Multi-dim splits (e.g.
    # k_fast (1, n, k)) aren't yet handled.
    sliced_output_syms = [
        s for s in seed_per_sym if seed_per_sym[s] > 1 and write_index.coeff(s) != 0
    ]
    if len(sliced_output_syms) != 1:
        return [seed]
    seed_sym = sliced_output_syms[0]
    seed_factor = int(seed_per_sym[seed_sym])

    options: list[tuple[dict, dict]] = [seed]
    seen: set[tuple] = {_canonical_key(seed)}
    for sym, extent in iter_space.items():
        extent_int = concretize_expr(extent)
        if (
            sym is seed_sym
            or write_index.coeff(sym) == 0
            or extent_int <= 1
            or extent_int % seed_factor != 0
        ):
            continue
        variant_per_sym = dict(seed_per_sym)
        variant_per_sym[seed_sym] = 1
        variant_per_sym[sym] = seed_factor
        variant = splits_by_index_coeff(variant_per_sym, write_index, read_index)
        key = _canonical_key(variant)
        if key in seen:
            continue
        options.append(variant)
        seen.add(key)
        if len(options) >= DEFAULT_VARIANT_CAP:
            break
    return options


def _canonical_key(splits: tuple[dict, dict]) -> tuple:
    """Hashable key for a (output_splits, reduction_splits) pair."""
    out, red = splits
    return (tuple(sorted(out.items())), tuple(sorted(red.items())))


class StrategyBCoOptimizingAllocator(DefaultAllocator):
    """`Strategy B` assumes work_distribution committed one best option (seed). Here we
    first add a few variants based on the seed, pick the combination that minimizes HBM
    bytes among all, then defer to DefaultAllocator's flow. As seed is in the search
    space, the worst case matches DefaultAllocator.
    """

    def plan_allocation(self, graph: GraphLowering):
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)

        # Enumerate options, run search, commit winners back to op_it_space_splits.
        ops = graph.operations
        options_per_op = [_enum_split_options(op) for op in ops]
        best_chosen = self._search(graph, ops, options_per_op)

        for op, opt_idx, options in zip(ops, best_chosen, options_per_op):
            chosen = options[opt_idx]
            if chosen != getattr(op, "op_it_space_splits", ({}, {})):
                op.op_it_space_splits = chosen

        # try insert clone again, as what was incompatible could be compatible now
        # TODO simplify the previous pre-opt (at the beginning of this func), we will
        # run check core-div-mismatch a few times due to clone-insertion, speed-up?
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)

        # Standard downstream flow on the now-fixed winning splits. Mirrors
        # DefaultAllocator.plan_allocation past the pre-passes.
        buffers = self._generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers)
        self._push_allocation(graph, allocation)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(
        self,
        graph: GraphLowering,
        ops: list[Operation],
        options_per_op: list[list[tuple[dict, dict]]],
    ) -> list[int]:
        """DFS over the option cross-product, scoring each leaf via
        _score_layout. Returns the option index per op for the leaf
        with minimum HBM bytes. No early-stop pruning — bounded by
        ≤ K^N leaves where N counts ops with >1 option (most return
        [seed]). Per-leaf cost is one full _generate_buffers +
        plan_layout pass; the `cache` param on _per_core_view_on_buf
        amortizes sympy work if it ever becomes hot.
        """
        chosen: list[int] = [0] * len(ops)
        best_total: float = math.inf
        best_chosen: list[int] = list(chosen)

        buf_total_bytes: dict[str, int] = {
            name: math.prod(buf.layout.device_layout.device_size[:-1]) * 128
            for name, buf in graph.name_to_buffer.items()
        }

        def recurse(op_idx: int) -> None:
            nonlocal best_total, best_chosen
            if op_idx == len(ops):
                hbm = self._score_layout(graph, buf_total_bytes)
                if hbm < best_total:
                    best_total = hbm
                    best_chosen = list(chosen)  # list() makes a copy
                return

            op = ops[op_idx]
            options = options_per_op[op_idx]

            # Mutate-and-undo: stash and restore op.op_it_space_splits.
            # If the op originally lacked the attribute, restore it as
            # ({}, {}) — equivalent to "unset" for all readers (which use
            # getattr(..., ({}, {})) or hasattr+empty-dict default).
            prev_split: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
            for opt_idx, option in enumerate(options):
                op.op_it_space_splits = option
                chosen[op_idx] = opt_idx
                recurse(op_idx + 1)
            op.op_it_space_splits = prev_split

        recurse(0)
        return best_chosen

    # ------------------------------------------------------------------
    # Leaf scoring
    # ------------------------------------------------------------------

    def _score_layout(
        self,
        graph: GraphLowering,
        buf_total_bytes: dict[str, int],
    ) -> int:
        """HBM bytes under the current split assignment: total device
        bytes of every buffer the solver couldn't pin. Non-committing
        (addresses land on throwaway buffers) and solver-agnostic.
        """
        buffers = self._generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers)
        pinned_names = {b.name for b in allocation if b.address is not None}

        return sum(
            total for name, total in buf_total_bytes.items() if name not in pinned_names
        )


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
