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
from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch._inductor.ir import (
    TensorBox,
    ComputedBuffer,
    Operation,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
    ExternKernel,
)
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    concretize_expr,
    iteration_space_from_op,
    splits_by_index_coeff,
    op_read_writes,
    _prepare_per_core_view,
    _per_core_view_from_prep,
)
from torch_spyre._inductor.work_division import enumerate_work_division_candidates
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
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
    buffer_not_read_in_full,
    GraphView,
    get_op_pointwise_inputs,
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
                # Clones are only pinned when the boundary-clone path is on; they
                # are never in the whitelist, so without this they'd be ineligible
                # and the inserted clones would not land in LX.
                or (config.lx_boundary_clones and self._get_op_name(op) == "clone")
            )
        )

    def _op_inputs_good_for_lx_inplace(self, op: Any) -> list[str]:
        target = getattr(getattr(op, "origin_node", None), "target", None)
        if target is None:
            return []
        reads = [dep.name for dep in op.get_read_writes().reads]
        if self._get_op_name(op) in OP_GOOD_FOR_LX_INPLACE:
            # If the op is in the whitelist, allow all inputs
            return reads
        if torch.Tag.pointwise in target.tags:
            # If the op is tagged as pointwise by pytorch upstream
            # allow all inputs. Does not work for all ops
            return reads
        return get_op_pointwise_inputs(op.data)

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

        # filter out intermediates read partially (sliced / multi-offset): the
        # single-base LX path mis-addresses such reads (see
        # buffer_not_read_in_full / compute_ops._start_addr_data), e.g. an
        # inner-dim slice x[:, :, 32:96] feeding a chained op. _build_bound_buffers
        # applies the same guard to graph input/output clones; this covers the
        # intermediate buffers. Overrides allow_all_ops_in_lx_planning by design.
        # Only check ops still eligible above: ops already dropped include
        # non-ComputedBuffer outputs (e.g. multi-output) whose layouts have no
        # size for buffer_not_read_in_full to inspect.
        drop_list.update(
            op.name
            for op in graph.operations
            if op.name not in drop_list and buffer_not_read_in_full(graph, op.name)
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
        self, graph: GraphLowering, in_place: Optional[dict[str, list[str]]]
    ) -> list[LifetimeBoundBuffer]:
        """Build the per-core-sized lifetime-bound buffers for the placement-only
        solvers. Each output read more than once (and not an extern consumer or
        an unclonable graph boundary) becomes a ``LifetimeBoundBuffer`` sized by
        ``size_per_core`` with its in-place parents; graph inputs are added when
        cloning is allowed. The joint core-division path uses
        ``_build_cd_bound_buffers`` instead."""
        lifetimes = calculate_liveness(graph)
        mem_usage = mem_usage_by_buf(GraphView(graph, self._filter_ops))
        in_place = {} if in_place is None else in_place
        buffers: list[LifetimeBoundBuffer] = []
        graph_output_names = set(graph.get_output_names())
        cloning_allowed = clone_at_graph_boundaries()
        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]
            if len(uses) <= 1:
                continue  # output is not read (only the write, or never touched)
            if any(isinstance(graph.operations[u], ExternKernel) for u in uses):
                continue
            if (
                output_name in graph_output_names
                and not cloning_allowed
                and buffer_not_read_in_full(graph, output_name)
            ):
                continue  # we can only allocate graph outputs if we're allowed to clone

            uses = lifetimes[output_name]
            parents = in_place.get(output_name, [])
            size = info["size_per_core"]
            buffers.append(
                LifetimeBoundBuffer(
                    output_name,
                    size,
                    uses,
                    first_use_is_read=False,
                    in_place_parents=parents,
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
                if buffer_not_read_in_full(graph, input_name):
                    # A consumer reads this input partially -- a sliced/
                    # multi-offset read (e.g. x[:, 0:512] + x[:, 512:1024], or
                    # x[:, :, 0:64]). The clone would be pinned to LX, which
                    # SDSC addresses by a single base, so partial reads
                    # mis-address and produce wrong results.
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
                        uses,
                        first_use_is_read=True,
                        in_place_parents=[],
                    )
                )

        return buffers

    def _determine_in_place(self, graph: GraphLowering) -> dict[str, list[str]]:
        allow_inplace: dict[str, list[str]] = {}
        graph_view = GraphView(graph, self._filter_ops)
        mem_usage = mem_usage_by_buf(graph_view)
        in_place_allowed = {
            op.name: self._op_inputs_good_for_lx_inplace(op)
            for op in graph_view.operations
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
                    input_buf in in_place_allowed[buf_name]
                    and inp_i_size_match
                    and inp_i_lay_match
                    and inp_i_eol
                    and no_core_div_mismatch
                ):
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _generate_buffers(
        self,
        graph: GraphLowering,
    ) -> list[Operation]:
        in_place = self._determine_in_place(graph)
        buffers = self._build_bound_buffers(graph, in_place)
        return buffers

    def _push_allocation(
        self, graph: GraphLowering, buffers: Sequence[LifetimeBoundBuffer]
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


def _lx_planning_size() -> int:
    """LX scratchpad bytes available to the layout solver."""
    return int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))


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
        # No config inspection here: the config -> (allocator, solver) mapping
        # lives in ``select_allocator``. A bare ``DefaultAllocator()`` defaults to
        # the greedy solver; any other solver is injected explicitly.
        if layout_planning is None:
            layout_planning = GreedyLayoutSolver(_lx_planning_size())
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
    rw = op_read_writes(op)
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


class CoOptimizingAllocator(ScratchpadAllocator):
    def __init__(
        self,
        pre_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
        post_optimization_passes: list[ScratchpadOptimizationPass] | None = None,
    ):
        """Joint core-division + LX-placement allocator. The solver is the
        OR-Tools ``CpSatLayoutSolver`` (``config.layout_solver == "cpsat"``)
        sized to available LX memory; ``pre_optimization_passes`` /
        ``post_optimization_passes`` (default none) run before / after layout
        planning.

        When the CP-SAT solver is unavailable (``ortools`` not installed) or a
        solve produces no feasible plan, planning falls back to the placement-only
        :class:`DefaultAllocator` (greedy) so a ``layout_solver="cpsat"`` request
        degrades to a correct plan instead of aborting the compile. The greedy
        path does not co-optimize core division, but every op keeps its
        upstream-chosen division, so the result is correct -- just less optimal.
        """
        size = _lx_planning_size()

        if pre_optimization_passes is None:
            pre_optimization_passes = []
        if post_optimization_passes is None:
            post_optimization_passes = []

        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes

        # Greedy fallback for when CP-SAT is unavailable or finds no plan.
        self._fallback = DefaultAllocator(layout_planning=GreedyLayoutSolver(size))

        self.layout_planning: Optional[MemoryPlanSolver]
        try:
            # Imported lazily so this module (and the greedy path) load even when
            # ortools is absent: CpSatLayoutSolver.__init__ raises ImportError
            # when ortools is missing, which we catch to fall back.
            from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
                CpSatLayoutSolver,
            )

            self.layout_planning = CpSatLayoutSolver(size)
        except ImportError as exc:
            logger.warning(
                "cpsat layout solver unavailable (%s); falling back to the "
                "default greedy allocator.",
                exc,
            )
            self.layout_planning = None

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, jointly solve core-division + LX placement, commit the
        chosen divisions, then run post-passes.

        Falls back to the greedy :class:`DefaultAllocator` when the CP-SAT solver
        is unavailable or fails to find a feasible plan. The fallback is taken
        before any allocation is pushed or any division committed, so the graph is
        never left half-planned (the pre-passes default to none and the buffer/
        division derivation is read-only)."""
        if self.layout_planning is None:
            self._fallback.plan_allocation(graph)
            return

        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        buffers = self._generate_cd_buffers(graph, self._division_map(graph))
        allocation = self.layout_planning.plan_layout(buffers)
        self._push_allocation(graph, allocation)
        self._commit_divisions(graph, allocation)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)

    def _division_map(self, graph: GraphLowering) -> dict[str, list[CoreDivision]]:
        """Per-op core-division candidates for the joint-division solve.

        Every op gets at least one ``CoreDivision`` so the slicing-match gate can
        constrain it. Pointwise / Reduction ops get the enumerated candidates;
        every other op falls back to a single fixed division read off its
        committed ``op_it_space_splits``. No op-kind pre-filter -- residency is
        gated per buffer (``residency_allowed``) and by the solver, so ineligible
        ops still participate as producers/consumers in the match.
        """
        max_cores = config.sencores
        return {
            op.name: self._enumerate_core_divisions(op, max_cores)
            for op in graph.operations
        }

    def _fixed_division(self, op: Operation) -> CoreDivision:
        """The op's upstream-committed division (``op.op_it_space_splits``) as a
        single pinned CoreDivision; a never-divided op yields a one-core empty
        split. Used as the fallback for ops with no enumerable candidates, so
        every buffer carries at least one division.
        """
        seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", None) or ({}, {})
        return CoreDivision(output_splits=dict(seed[0]), reduction_splits=dict(seed[1]))

    def _enumerate_core_divisions(
        self, op: Operation, max_cores: int
    ) -> list[CoreDivision]:
        """Core-division candidates for one eligible op (see ``_division_map``).

        Each ``enumerate_work_division_candidates`` split is encoded into the
        stride-keyed ``(output_splits, reduction_splits)`` form and deduped by
        slicing signature. Ops without a divisible iteration space, or whose
        space can't be enumerated, fall back to a single fixed division.
        """
        fixed = [self._fixed_division(op)]
        if not isinstance(op, ComputedBuffer) or not isinstance(
            op.data, (Pointwise, Reduction)
        ):
            return fixed
        rw = op_read_writes(op)
        write = next(iter(rw.writes), None)

        # this is essentially a dead branch but serves as a type narrowing below
        if write is None:
            return fixed
        write_index = write.index
        first_read = next(iter(rw.reads), None)
        read_index = first_read.index if first_read is not None else write_index

        try:
            candidates = enumerate_work_division_candidates(op, max_cores)
        except Unsupported as exc:
            # Symbolic stick dims etc. can't be enumerated; leave the op on its
            # upstream-chosen split (fixed division).
            logger.debug("skip joint division for %s: %s", op.name, exc)
            return fixed

        cds: list[CoreDivision] = []
        seen: set[tuple] = set()
        for cand in candidates:
            out_s, red_s = splits_by_index_coeff(cand, write_index, read_index)
            key = (
                tuple(sorted(out_s.items())),
                tuple(sorted(red_s.items())),
            )
            if key in seen:
                continue
            seen.add(key)
            cds.append(CoreDivision(output_splits=out_s, reduction_splits=red_s))
        return cds or fixed

    def _commit_divisions(
        self,
        graph: GraphLowering,
        allocation: Sequence[CoreDivisionBuffer],
    ) -> None:
        """Write the solver's chosen division back to ``op.op_it_space_splits``
        for *every* buffer the solver assigned one.

        The solver optimizes a core division for all buffers, not just resident
        ones: a resident producer and its consumers are pinned by
        ``_implicate_core_division`` to one shared slicing (so those commits are
        mutually consistent), while a spilled buffer is free of that gate -- its
        accesses round-trip through HBM, which re-slices on load -- so it takes
        its most parallel candidate. Committing the spilled buffers' divisions
        too lets the joint solve optimize work division across the whole graph,
        not only the LX-resident region.
        """
        op_by_name = {op.name: op for op in graph.operations}
        for buf in allocation:
            op = op_by_name.get(buf.name)
            if op is None or buf.chosen_division is None:
                continue
            cd = buf.core_divisions[buf.chosen_division]
            op.op_it_space_splits = (
                dict(cd.output_splits),
                dict(cd.reduction_splits),
            )

    def _generate_cd_buffers(
        self,
        graph: GraphLowering,
        divisions: dict[str, list[CoreDivision]],
    ) -> list[CoreDivisionBuffer]:
        in_place = self._determine_in_place(graph)
        buffers = self._build_cd_bound_buffers(graph, in_place, divisions)
        return buffers

    def _determine_in_place(self, graph: GraphLowering) -> dict[str, list[str]]:
        """Co-opt in-place candidates: keep only the *division-invariant*
        preconditions here and defer the division-dependent ones to the solver.

        The per-core size match and core-division compatibility depend on the
        division the ILP has not yet chosen, so they are enforced in the solver
        (``eff_size`` equality + the ``cd_parent_matches`` gate). What stays as a
        pre-filter is division-invariant: lifetime adjacency
        (``in_end == out_start``, the single-tick-handoff invariant the solver's
        no-overlap relaxation relies on but cannot re-derive) and identical device
        layouts (required for the storage to alias).
        """
        allow_inplace: dict[str, list[str]] = {}
        mem_usage = mem_usage_by_buf(graph)
        in_place_allowed = {
            op.name: self._op_inputs_good_for_lx_inplace(op) for op in graph.operations
        }
        lifetimes = calculate_liveness(graph)
        for buf_name, info in mem_usage.items():
            allow_inplace[buf_name] = []
            if not in_place_allowed[buf_name]:
                continue
            # Unplaceable producers (e.g. a ``MultiOutputLayout`` tuple op like
            # max-with-indices) carry no ``device_layout``: their storage cannot
            # alias an input, so skip rather than raise ``AttributeError``.
            out_layout = graph.get_buffer(buf_name).layout
            if not hasattr(out_layout, "device_layout"):
                continue
            out_start = lifetimes[buf_name][0]
            out_ten_layout = out_layout.device_layout
            for input_buf in info["op_inputs"]:
                in_layout = graph.get_buffer(input_buf).layout
                if not hasattr(in_layout, "device_layout"):
                    continue
                in_end = lifetimes[input_buf][-1]  # inclusive last use
                in_ten_layout = in_layout.device_layout
                inp_i_lay_match = out_ten_layout == in_ten_layout
                inp_i_eol = in_end == out_start  # same op reads input, writes output
                if inp_i_lay_match and inp_i_eol:
                    allow_inplace[buf_name].append(input_buf)
        return allow_inplace

    def _residency_by_buf(
        self,
        graph: GraphLowering,
        mem_usage: dict,
        op_by_name: dict[str, Operation],
        lifetimes: dict[str, list[int]],
    ) -> dict[str, bool]:
        """Whether each buffer in ``mem_usage`` may be pinned (resident) in LX.

        Every buffer is handed to the solver so it participates in the slicing
        match, but participation is not residency. A buffer may be *pinned* only
        if its producing op clears ``_op_output_good_for_lx_reuse``, has no
        ExternKernel consumer (extern ops read from HBM), is not the target of an
        in-place mutation, is off a graph boundary, is read in full (offset reads
        mis-address a single LX base), and is actually read. Otherwise it stays
        non-resident so it doesn't orphan its neighbours.
        """
        # Targets of a ``MutationLayoutSHOULDREMOVE`` op (e.g. a ``cat`` dest
        # filled by per-input ``copy_`` slices): the producing op reads nothing
        # -- its data arrives via offset writes -- so pinning it to one LX base
        # mis-addresses. The mutating ops are rejected by
        # ``_op_output_good_for_lx_reuse``, but their target is a normal layout
        # that would otherwise pass, so exclude it explicitly. Computed once so
        # the predicate stays linear in the graph.
        mutated_buffers = {
            op.layout.target.get_name()
            for op in graph.operations
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE)
        }
        graph_output_names = set(graph.get_output_names())
        out: dict[str, bool] = {}
        for name in mem_usage:
            op = op_by_name.get(name)
            uses = lifetimes[name]
            out[name] = (
                op is not None
                and self._op_output_good_for_lx_reuse(op)
                and not any(isinstance(graph.operations[u], ExternKernel) for u in uses)
                and name not in mutated_buffers
                and name not in graph_output_names
                and not buffer_not_read_in_full(graph, name)
                and not len(uses) <= 1
            )
        return out

    def _build_cd_bound_buffers(
        self,
        graph: GraphLowering,
        in_place: Optional[dict[str, list[str]]],
        divisions: dict[str, list[CoreDivision]],
    ) -> list[CoreDivisionBuffer]:
        """Build the ``CoreDivisionBuffer``s handed to the solver.

        Every buffer carries its candidate ``divisions`` and is sized by its
        *total* device footprint plus its producer edges (``parent_proj``); the
        solver picks a division and divides by its ``output_partition``. Because
        all buffers are on the same total scale, ``in_place_parents`` need no
        filtering."""
        lifetimes = calculate_liveness(graph)
        mem_usage = mem_usage_by_buf(graph)
        in_place = {} if in_place is None else in_place
        op_by_name = {op.name: op for op in graph.operations}

        # Caches the candidate-invariant view prep (``_prepare_per_core_view``)
        # keyed by (op, dep, buf), so a parent read by several consumers prepares
        # its write-view once and each op's sympy work is reused across divisions.
        prep_cache: dict = {}
        buffers: list[CoreDivisionBuffer] = []
        # Residency for every buffer up front: ``_cd_parent_matches`` consults the
        # same map so it never matches against a never-resident parent. Computed
        # before the loop because a parent can appear later than its consumer.
        residency_by_buf = self._residency_by_buf(
            graph, mem_usage, op_by_name, lifetimes
        )
        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]

            op = op_by_name.get(output_name)
            residency_allowed = residency_by_buf[output_name]

            buf_divisions = divisions[output_name]
            parents = in_place.get(output_name, [])
            size = info["size"]  # total footprint; solver divides per chosen cd
            parent_proj = info["op_inputs"]
            cd_parent_matches = self._cd_parent_matches(
                op,
                buf_divisions,
                parent_proj,
                divisions,
                op_by_name,
                prep_cache,
                residency_by_buf,
            )

            buffers.append(
                CoreDivisionBuffer(
                    output_name,
                    size,
                    uses,
                    first_use_is_read=True,
                    in_place_parents=parents,
                    core_divisions=buf_divisions,
                    parents=parent_proj,
                    cd_parent_matches=cd_parent_matches,
                    residency_allowed=residency_allowed,
                )
            )

        return buffers

    def _cd_parent_matches(
        self,
        consumer_op: Optional[Operation],
        consumer_divs: list[CoreDivision],
        parent_names: list[str],
        divisions: dict[str, list[CoreDivision]],
        op_by_name: dict[str, Operation],
        prep_cache: dict,
        residency_by_buf: dict[str, bool],
    ) -> dict[str, list[tuple[int, int]]]:
        """Physical slicing-match pairs for each divided producer this op reads.

        For producer ``P`` feeding this consumer, a ``(P_div_idx,
        consumer_div_idx)`` pair is compatible iff the two divisions induce the
        *same per-core slicing of ``P``* (``P``'s write-view equals the
        consumer's read-view, both via ``_per_core_view_on_buf`` in ``P``'s
        device-dim frame) AND use the *same total core count*. This is the
        per-core-view comparison ``get_ncores_for_buffers`` uses -- correct across
        reductions/reshapes, where a coeff-keyed signature would conflate axes.

        Excluded from matching (producer then falls back to HBM, always correct):
        a producer that can never be resident (``residency_by_buf`` False); a
        producer candidate whose write carries a partial reduction (output not
        final); and either side's candidate whose slicing of ``P`` is
        unrepresentable -- we never pin on a slicing we cannot verify.
        """
        if consumer_op is None:
            return {}
        matches: dict[str, list[tuple[int, int]]] = {}
        consumer_reads = op_read_writes(consumer_op).reads
        for parent in parent_names:
            # A never-resident producer always reads from HBM, so its division
            # can't constrain the consumer -- skip the match (and the write-index
            # lookup below, undefined for StarDep writers).
            if not residency_by_buf.get(parent, False):
                continue
            parent_divs = divisions[parent]
            parent_op = op_by_name[parent]
            write_dep = next(
                (
                    w
                    for w in op_read_writes(parent_op).writes
                    if w.name == parent and hasattr(w, "index")
                ),
                None,
            )
            read_dep = next(
                (r for r in consumer_reads if r.name == parent and hasattr(r, "index")),
                None,
            )
            if write_dep is None or read_dep is None:
                continue

            # Producer view per candidate on its own output ``parent``. ``None``
            # marks a candidate that cannot host a readable residency: a
            # partial-reduction write, or an unrepresentable slicing of ``parent``.
            prod_views: list[Optional[tuple]] = [
                view if (repr_ok and not partial) else None
                for view, partial, repr_ok in self._views_for_divs(
                    parent_op, write_dep, parent, parent_divs, prep_cache
                )
            ]
            # Consumer read-views: same unrepresentable guard. A clean empty view
            # (the split doesn't slice ``parent`` -> reads it whole) is
            # representable and legitimately matches a whole-buffer producer.
            cons_views: list[Optional[tuple]] = [
                view if repr_ok else None
                for view, _partial, repr_ok in self._views_for_divs(
                    consumer_op, read_dep, parent, consumer_divs, prep_cache
                )
            ]

            # A matched pair needs equal per-core slicing of ``parent`` AND equal
            # *total* core count. Equal views alone aren't enough: a producer on N
            # and consumer on M>N cores can share an identical (possibly empty)
            # slicing while the consumer's extra cores -- split on a broadcast axis
            # -- hold no copy and would read a stale/partial LX buffer. The joint
            # solver re-divides per buffer and can hit this, hence the gate; a
            # rejected pair just falls back to HBM.
            pairs = [
                (i, j)
                for i, pv in enumerate(prod_views)
                if pv is not None
                for j, cv in enumerate(cons_views)
                if cv is not None
                and pv == cv
                and parent_divs[i].cores_used == consumer_divs[j].cores_used
            ]
            matches[parent] = pairs
        return matches

    @staticmethod
    def _views_for_divs(op, dep, buf_name, divs, prep_cache: dict):
        """Per-core views of ``buf_name`` for each candidate division of ``op``.

        Prepares the candidate-invariant context once (``_prepare_per_core_view``
        -- the sympy-heavy op-level work) and evaluates every candidate from it
        via ``_per_core_view_from_prep``, so cost scales with the op rather than
        its candidate count.

        ``prep_cache`` is keyed by ``(op name, dep, buf_name)``: a producer's
        write-dep and a consumer's read-dep on the same buffer can be equal
        ``MemoryDep``s, so the op name keeps their preps distinct while a parent
        read by several consumers reuses its write-view prep.
        """
        key = (op.get_name(), dep, buf_name)
        out = []
        for cd in divs:
            coeff = (cd.output_splits, cd.reduction_splits)
            # Build the prep only when a candidate actually has a split:
            # ``_per_core_view_from_prep`` returns the whole-buffer view for a
            # no-split candidate before touching the prep, so a never-divided op
            # (e.g. a StarDep write with no ``.index``) is never prepared.
            if any(n > 1 for d in coeff for n in d.values()) and key not in prep_cache:
                prep_cache[key] = _prepare_per_core_view(op, dep, buf_name)
            out.append(_per_core_view_from_prep(prep_cache.get(key), coeff))
        return out


_PLACEMENT_SOLVERS: dict[str, type[MemoryPlanSolver]] = {
    "greedy": GreedyLayoutSolver,
    "bestfit": BestFitLayoutSolver,
    "firstfit": FirstFitLayoutSolver,
}


def select_allocator() -> ScratchpadAllocator:
    """Build the scratchpad allocator and inject its layout solver from config.

    This is the single place that maps config to an (allocator, solver) pair, so
    the allocators themselves take an explicit solver and never inspect config:

    * ``layout_solver == "cpsat"`` -> joint core-division + LX placement via
      :class:`CoOptimizingAllocator` (with a built-in greedy fallback). This
      wins over ``co_optimizing_lx_planning`` because CP-SAT runs its own
      core-division co-optimization.
    * ``co_optimizing_lx_planning`` -> gap-based co-optimization via
      :class:`StrategyBCoOptimizingAllocator`.
    * otherwise -> placement-only :class:`DefaultAllocator` with the configured
      gap-based solver (greedy/bestfit/firstfit).
    """
    if config.layout_solver == "cpsat":
        return CoOptimizingAllocator()

    try:
        solver_cls = _PLACEMENT_SOLVERS[config.layout_solver]
    except KeyError:
        raise ValueError(
            f"Invalid layout_solver config option '{config.layout_solver}'."
        )
    solver = solver_cls(_lx_planning_size())

    if config.co_optimizing_lx_planning:
        return StrategyBCoOptimizingAllocator(layout_planning=solver)
    return DefaultAllocator(layout_planning=solver)


def scratchpad_planning(
    graph: GraphLowering,
    allocator: Optional[ScratchpadAllocator] = None,
) -> None:
    """Assign LX scratchpad addresses to eligible buffers in a lowered graph.

    Called after stickification and core-division are complete. Graph operations
    are expected to be in topological order as guaranteed by GraphLowering.

    Args:
        graph: Lowered graph to plan scratchpad memory for.
        allocator: Allocator strategy to use. Defaults to the config-selected
            allocator (see :func:`select_allocator`).
    """
    if allocator is None:
        allocator = select_allocator()
    allocator.plan_allocation(graph)
