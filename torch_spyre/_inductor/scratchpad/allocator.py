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
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import sympy
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
from torch._inductor.dependencies import ReadWrites

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
    get_op_pointwise_inputs,
)
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import _is_matmul_op

logger = get_inductor_logger("scratchpad.allocator")


class ScratchpadAllocator(ABC):
    """
    Abstract class for all implementations of ScratchpadAllocator
    """

    def __init__(self) -> None:
        # Populated during plan_allocation: maps buffer/op name → reason string.
        # Stamped by _filter_ops, _build_bound_buffers, and plan_allocation
        # (for the solver decision). Reset at the start of each plan_allocation.
        self.reject_reasons: dict[str, str] = {}

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

    def _filter_ops(
        self,
        graph: GraphLowering,
        cache: Optional[dict] = None,
        rw_cache: Optional[dict[str, ReadWrites]] = None,
    ) -> list[Operation]:
        core_div_mismatch = get_ncores_for_buffers(graph, cache, rw_cache)
        drop_list = set()

        # filter out by permitted operations
        for op in graph.operations:
            if not self._op_output_good_for_lx_reuse(op):
                drop_list.add(op.name)
                self.reject_reasons[op.name] = "op not allowed"

        # filter out core division mismatches
        for key, mismatch in core_div_mismatch.items():
            if mismatch == -1:
                drop_list.add(key)
                self.reject_reasons[key] = "core div mismatch"

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
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
        cache: Optional[dict] = None,
    ) -> list[LifetimeBoundBuffer]:
        in_place = {} if in_place is None else in_place
        buffers = []
        graph_output_names = set(graph.get_output_names())
        cloning_allowed = clone_at_graph_boundaries()
        for output_name, info in mem_usage.items():
            uses = lifetimes[output_name]
            if len(uses) <= 1:
                self.reject_reasons[output_name] = "single use"
                continue  # output is not read (only the write, or never touched)
            if any(isinstance(graph.operations[u], ExternKernel) for u in uses):
                self.reject_reasons[output_name] = "extern kernel user"
                continue
            if output_name in graph_output_names and not cloning_allowed:
                self.reject_reasons[output_name] = "graph output (no clone)"
                continue  # we can only allocate graph outputs if we're allowed to clone
            buffers.append(
                LifetimeBoundBuffer(
                    output_name,
                    info["size_per_core"],
                    uses,
                    first_use_is_read=False,
                    in_place_parents=in_place.get(output_name, []),
                )
            )

        if cloning_allowed:
            ncores = get_ncores_for_buffers(graph, cache)
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
                        uses,
                        first_use_is_read=True,
                        in_place_parents=[],
                    )
                )

        return buffers

    def _determine_in_place(
        self,
        graph: GraphLowering,
        graph_view: "GraphView",
        mem_usage: dict,
        lifetimes: dict[str, list[int]],
    ) -> dict[str, list[str]]:
        allow_inplace: dict[str, list[str]] = {}
        in_place_allowed = {
            op.name: self._op_inputs_good_for_lx_inplace(op)
            for op in graph_view.operations
        }
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
        cache: Optional[dict] = None,
        timings: Optional[dict[str, float]] = None,
        lifetimes: Optional[dict[str, list[int]]] = None,
        rw_cache: Optional[dict[str, ReadWrites]] = None,
    ) -> list[Operation]:
        # Build graph_view + mem_usage once and share; both helpers below treat
        # them read-only. `lifetimes` is split-invariant, so the co-opt search
        # passes it in (computed here only for the single-shot path). `rw_cache`
        # (split-invariant {op name: ReadWrites}) is likewise threaded in so the
        # per-leaf core-div check doesn't re-trace get_read_writes() per op.
        #
        # TODO: graph_view + mem_usage still rebuilt per leaf; only their
        #   split-dependent part is the (cached) core-div check, so the rest
        #   could be hoisted out of the per-leaf path too.
        t0 = time.perf_counter()
        graph_view = GraphView(graph, lambda g: self._filter_ops(g, cache, rw_cache))
        t1 = time.perf_counter()
        mem_usage = mem_usage_by_buf(graph_view, cache, rw_cache)
        t2 = time.perf_counter()
        if timings is not None:
            timings["graph_view"] += t1 - t0
            timings["mem_usage"] += t2 - t1

        if lifetimes is None:
            lifetimes = calculate_liveness(graph)

        in_place = self._determine_in_place(graph, graph_view, mem_usage, lifetimes)
        buffers = self._build_bound_buffers(
            graph, in_place, mem_usage, lifetimes, cache
        )
        return buffers

    def _log_lx_pinning(self, graph: GraphLowering) -> None:
        """Log the final LX pinning decision for every op in the graph."""
        # Skip the per-op getattr walk unless DEBUG is on.
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for op in graph.operations:
            reason = self.reject_reasons.get(op.name, "lx")
            logger.debug(
                "lx_pinning: %s (%s) → %s",
                op.name,
                self._get_op_name(op),
                reason,
            )

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

        super().__init__()
        self.pre_optimization_passes = pre_optimization_passes
        self.post_optimization_passes = post_optimization_passes
        self.layout_planning = layout_planning

    def plan_allocation(self, graph: GraphLowering):
        """Run pre-passes, assign LX addresses to eligible buffers, then run post-passes.

        Args:
            graph: Lowered graph whose buffers will be assigned LX scratchpad
                addresses where viable.
        """
        self.reject_reasons = {}
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)
        buffers = self._generate_buffers(graph)
        allocation = self.layout_planning.plan_layout(buffers, log_lx_usage=True)
        for b in allocation:
            if b.address is None:
                self.reject_reasons[b.name] = "no room on scratchpad"
        self._push_allocation(graph, allocation)
        self._log_lx_pinning(graph)
        for p in self.post_optimization_passes:
            p.apply_pass(graph)


DEFAULT_VARIANT_CAP = 6


def _output_stride_to_device_size(op: Operation) -> dict[int, int]:
    """Map each output host stride to the device size of the device dim it lands on.

    A stickified host dim decomposes into an outer-stick dim (size = stick count)
    at stride ``stick_host_stride * elems_per_stick`` and a within-stick dim at
    stride ``stick_host_stride``; sticks are atomic, so a split on that host dim
    uses the outer-stick dim. Keying by stride lets a caller look up the true
    splittable size for an output dim by its coefficient in the write index.
    (Mirrors _per_core_view_on_buf's stride→device-dim placement.)
    """
    dev_layout = op.layout.device_layout
    device_size = dev_layout.device_size
    stride_map = dev_layout.stride_map
    elems_per_stick = dev_layout.device_dtype.elems_per_stick()
    stride_to_size: dict[int, int] = {}
    for i, s in enumerate(stride_map):
        if s <= 0:  # sentinel for collapsed / broadcast dims
            continue
        if s not in stride_to_size or device_size[i] != 1:
            stride_to_size[s] = device_size[i]
    if stride_map[-1] > 0:  # stickified dim -> bound by the outer-stick count
        stride_to_size[stride_map[-1]] = stride_to_size.get(
            stride_map[-1] * elems_per_stick, 1
        )
    return stride_to_size


def _split_fits_sticks(op: Operation, splits: tuple[dict, dict]) -> bool:
    """True if every output-dim factor in `splits` divides that dim's stick count.

    A split factor must divide the device size of the dim it lands on, which for
    the stickified dim is the stick count, not the element extent. Element-extent
    divisibility is not enough: N=128 with 64 elems/stick is only 2 sticks, yet
    128 % 4 == 0 would admit a 4-way split the SDSC bundler then rejects (SIGABRT).
    Checks output splits only; reduction (K) splits are bounded by the planner.

    A split whose stride has no entry in stride_to_size (e.g. it lands on a
    collapsed/broadcast device dim that _output_stride_to_device_size skips) is
    unplaceable and rejected: size defaults to 0, and `size <= 0` fails the check.
    (Plain `0 % factor == 0` would wrongly *admit* it.)
    """
    out_splits = splits[0]
    if not out_splits:
        return True
    stride_to_size = _output_stride_to_device_size(op)
    for stride, factor in out_splits.items():
        if factor <= 1:
            continue
        size = stride_to_size.get(int(stride), 0)
        if size <= 0 or size % factor != 0:
            return False
    return True


# TODO: helper for cross-matmul split transfer. Remove together with the
# block in _enum_split_options once work_dist assigns consistent splits.
def _matmul_axis_parse(
    op: Operation,
) -> dict[str, tuple[sympy.Symbol, int, int]]:
    """Parse a batched-matmul op into ``{role: (sym, extent, factor)}``.

    Role is one of "B", "M", "N", "K"; `sym` is the op's iter symbol for that
    axis, `extent` its splittable size, `factor` the current split from
    op.op_it_space_splits (1 if unsplit). Output is [B, M, N] (3D) or [M, N]
    (2D), so output symbols sorted by ascending stride spell N, M, B (B absent
    for 2D); the lone reduction symbol is K.

    For output dims, `extent` is the device size of the dim it maps to (the stick
    count for the stickified dim), via _output_stride_to_device_size — so a valid
    split must divide the stick count, not the element extent.
    """
    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads), write_index)
    iter_space = iteration_space_from_op(op)

    seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    per_sym = apply_splits_from_index_coeff(seed, write_index, read_index, iter_space)
    stride_to_size = _output_stride_to_device_size(op)

    # Derive axis symbols from the index free_symbols (not iter_space, which may
    # not enumerate every indexed symbol): output dims are in write_index, and K
    # is whatever the read index adds on top of the write.
    out_stride_sym = {int(write_index.coeff(s)): s for s in write_index.free_symbols}
    k_syms = read_index.free_symbols - write_index.free_symbols
    if not k_syms:
        raise ValueError(
            f"matmul {op.get_name()}: read index adds no reduction symbol over "
            f"the write index (read={read_index}, write={write_index})"
        )
    k_sym = next(iter(k_syms))

    roles: dict[str, tuple[sympy.Symbol, int, int]] = {}
    possible_roles = ["N", "M", "B"]
    for i, st in enumerate(sorted(out_stride_sym)):  # ascending, works for 2D and 3D
        sym = out_stride_sym[st]
        roles[possible_roles[i]] = (sym, stride_to_size[st], per_sym[sym])
    roles["K"] = (k_sym, concretize_expr(iter_space[k_sym]), per_sym[k_sym])

    return roles


# Batch factors to try, largest first. Only the largest one that fits is offered
# (see _factored_bm_splits): a bigger B split keeps more of the batch axis whole,
# and smaller-B variants don't aid reconciliation while multiplying the co-opt
# search space. m_fac = ncores // b_fac.
_FACTORED_B_FACTORS: tuple[int, ...] = (8, 4, 2)


def _bm_axes_from_roles(
    roles: dict[str, tuple[sympy.Symbol, int, int]],
) -> Optional[tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]]]:
    """B/M axes ((b_sym, b_extent), (m_sym, m_extent)) from _matmul_axis_parse
    roles, or None if either is absent."""
    b = roles.get("B")
    m = roles.get("M")
    if b is None or m is None:
        return None
    return (b[0], b[1]), (m[0], m[1])


def _reduction_bm_axes(
    op: Operation,
) -> Optional[tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]]]:
    """B/M axes for a reduction op, from its output dims.

    A reduction over N keeps B and M as output dims (e.g. write `512*d0 + d1`:
    B=d0, M=d1). Mirror _matmul_axis_parse's stride convention: sort output syms
    by ascending stride; the largest-stride dim is B (outermost), the next is M.
    Extents are stick-aware via _output_stride_to_device_size. None if < 2 output
    dims (nothing to factor).
    """
    write_index = next(iter(op.get_read_writes().writes)).index
    out_stride_sym = {int(write_index.coeff(s)): s for s in write_index.free_symbols}
    if len(out_stride_sym) < 2:
        return None
    stride_to_size = _output_stride_to_device_size(op)
    by_stride = sorted(out_stride_sym)  # ascending: [..., M, B]
    m_stride, b_stride = by_stride[-2], by_stride[-1]
    return (
        (out_stride_sym[b_stride], stride_to_size[b_stride]),
        (out_stride_sym[m_stride], stride_to_size[m_stride]),
    )


# TODO: companion to _matmul_axis_parse. Remove with the block in
# _enum_split_options once work_dist assigns consistent splits.
def _factored_bm_splits(
    op: Operation,
    bm_axes: Optional[tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]]],
) -> list[tuple[dict, dict]]:
    """Batch-major (B/b · M/m) full-core output split for `op`.

    `bm_axes` is ((b_sym, b_extent), (m_sym, m_extent)) for the op's batch and M
    output dims (from _bm_axes_from_roles for matmuls, _reduction_bm_axes for
    reductions). Returns at most ONE candidate: the largest-B full-core factoring
    (b_fac from _FACTORED_B_FACTORS, largest first; m_fac = ncores // b_fac) that
    divides both stick-count extents. Smaller-B factorings are not offered — they
    don't help reconciliation and only inflate the co-opt search space. Empty if
    no factoring fits (e.g. B too small). The caller's _split_fits_sticks is the
    final guard.
    """
    if bm_axes is None:
        return []
    (b_sym, b_extent), (m_sym, m_extent) = bm_axes
    ncores = config.sencores

    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads), write_index)

    for b_fac in _FACTORED_B_FACTORS:
        m_fac = ncores // b_fac
        if (
            b_fac * m_fac != ncores
            or b_fac > b_extent
            or b_extent % b_fac != 0
            or m_extent % m_fac != 0
        ):
            continue
        per_sym = {b_sym: b_fac, m_sym: m_fac}
        return [splits_by_index_coeff(per_sym, write_index, read_index)]
    return []


# TODO: companion to _matmul_axis_parse. Remove with the block in
# _enum_split_options once work_dist assigns consistent splits.
def _find_distinct_matmul_splits(
    ops: list[Operation],
) -> tuple[tuple[tuple[dict, dict], ...], tuple[dict[str, int], ...]]:
    """Collect the distinct matmul output-splits in `ops`.

    Returns ``(bases, roles)`` deduped by canonical key. `bases` are raw
    (output_splits, {}) tuples for the pointwise path — the matmuls' seed splits
    plus the factored batch-major (B/b · M/m) splits, so the softmax chain between
    two matmuls can adopt a shared B/M tiling. `roles` are the seed splits as
    {role: factor} maps (e.g. {"M": 4, "N": 8}) for the cross-matmul transfer.
    """
    seen: set[tuple] = set()
    bases: list[tuple[dict, dict]] = []
    roles: list[dict[str, int]] = []
    for op in ops:
        if not _is_matmul_op(op):
            continue
        op_roles = _matmul_axis_parse(op)
        out: dict = getattr(op, "op_it_space_splits", ({}, {}))[0]
        candidates: list[tuple[dict, dict]] = [(dict(out), {})] if out != {} else []
        candidates += _factored_bm_splits(op, _bm_axes_from_roles(op_roles))
        for base in candidates:
            key = _canonical_key(base)
            if key in seen:
                continue
            seen.add(key)
            bases.append(base)
        if out != {}:
            roles.append({r: f for r, (_s, _e, f) in op_roles.items()})
    return tuple(bases), tuple(roles)


# TODO: companion to _matmul_axis_parse. Remove with the block in
# _enum_split_options once work_dist assigns consistent splits.
def _check_and_add_matmul_option(
    op: Operation,
    seed: tuple[dict, dict],
    matmul_roles: tuple[dict[str, int], ...],
) -> list[tuple[dict, dict]]:
    """Options for matmul `op`: its seed, each other matmul's split transferred
    into this op's coordinates by axis role, plus factored batch-major (B/b · M/m)
    splits.

    work_dist can assign two matmuls inconsistent splits (e.g. QK {4096:4, 1:8}
    vs AV {128:32}); a shared axis (here M) then disagrees with the PW/softmax
    ops between them, forcing a core-div mismatch and blocking LX pinning.
    Offering each matmul the other's split lets the co-opt search pick a
    consistent assignment. A role absent on this op, or whose extent is not
    divisible by the source factor, does not transfer. The factored B/M splits
    cover the case where the two matmuls' N/K roles map to different physical
    dims and so can't be cross-transferred, but they still share the B and M
    output axes. Candidates that fail to reconcile a shared buffer's PerCoreView
    self-eliminate during scoring.
    """
    self_roles = _matmul_axis_parse(op)
    rw = op.get_read_writes()
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads), write_index)

    options: dict[tuple, tuple[dict, dict]] = {_canonical_key(seed): seed}
    for src in matmul_roles:
        per_sym = {}
        for role, (sym, extent, _factor) in self_roles.items():
            factor = src.get(role, 1)
            per_sym[sym] = factor if factor > 1 and extent % factor == 0 else 1
        if not any(f > 1 for f in per_sym.values()):
            continue
        candidate = splits_by_index_coeff(per_sym, write_index, read_index)
        options.setdefault(_canonical_key(candidate), candidate)
    for candidate in _factored_bm_splits(op, _bm_axes_from_roles(self_roles)):
        options.setdefault(_canonical_key(candidate), candidate)
    # Here the filter is mostly defense-in-depth: _matmul_axis_parse already
    # reports stick-count extents, so the `extent % factor == 0` gate above keeps
    # candidates stick-divisible. _split_fits_sticks still catches the residual
    # case it can't — a factor landing on a collapsed/broadcast dim (no stick
    # count). The seed is always kept (work_dist's own choice).
    return [
        opt for opt in options.values() if opt == seed or _split_fits_sticks(op, opt)
    ]


def _enum_split_options(
    op: Operation,
    extra_bases: tuple[tuple[dict, dict], ...] = (),
    matmul_roles: tuple[dict[str, int], ...] = (),
) -> list[tuple[dict, dict]]:
    """Split options for a pointwise op: the seed (index 0) plus variants
    that flip the split onto another output dim (≤ DEFAULT_VARIANT_CAP).

    `extra_bases` (the matmuls' output-splits) are offered on top so the op
    can adopt a matmul's tiling and pin its shared buffer to LX. Matmuls take
    their dedicated cross-matmul + factored-B/M path; non-matmul reductions are
    offered the factored B/M splits (so a softmax chain's max/sum can reconcile
    with the chain instead of forcing a core-div mismatch). Invalid bases
    self-eliminate during scoring.

    `matmul_roles` map BMNK to splits, {"M": 4, "N: 8, ...} then apply to other matmuls.
    """
    seed: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    is_output_splits_empty = seed[0] == {}
    is_computed_buf = isinstance(op, ComputedBuffer)
    is_reduction = is_computed_buf and isinstance(op.data, Reduction)
    is_matmul = is_reduction and _is_matmul_op(op)

    # TODO: let a matmul also consider the *other* matmuls' splits.
    # Remove once work_dist assigns consistent splits.
    if is_matmul and matmul_roles:
        return _check_and_add_matmul_option(op, seed, matmul_roles)

    # A non-matmul reduction (e.g. softmax max/sum) keeps B and M as output dims;
    # offer it the factored B/M splits so it can reconcile with a B/M-tiled chain
    # rather than being stuck on its seed and breaking the chain's per-core views.
    # We do NOT flip reductions onto other dims (their reduction axis is fixed).
    if is_reduction and not is_matmul and not is_output_splits_empty:
        red_options: dict[tuple, tuple[dict, dict]] = {_canonical_key(seed): seed}
        for base in _factored_bm_splits(op, _reduction_bm_axes(op)):
            red_options.setdefault(_canonical_key(base), base)
        return [
            opt
            for opt in red_options.values()
            if opt == seed or _split_fits_sticks(op, opt)
        ]

    # Only pointwise ops are flipped; reductions/matmuls keep work-division's
    # split. For compute-bound ops, prioritize PT utilization over LX pinning:
    # overriding a matmul's split to chase pinning regressed kernel time ~2.5x
    # (mlp-linear-kn.t, SENCORES=32; PT-util 66%→33%). Exclude future
    # compute-bound ops here too.
    # is_matmul implies is_reduction, so it's covered by the is_reduction term.
    if is_output_splits_empty or not is_computed_buf or is_reduction:
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

    sliced_output_syms = [
        s for s in seed_per_sym if seed_per_sym[s] > 1 and write_index.coeff(s) != 0
    ]

    # Dedup-and-collect in one dict: canonical key -> split tuple (the split
    # tuple itself is two dicts, so it can't be a key directly). Insertion
    # order is preserved, so the seed stays first.
    options: dict[tuple, tuple[dict, dict]] = {_canonical_key(seed): seed}

    # Only single output-dim splits are flipped. Multi-dim splits (e.g.
    # k_fast (1, n, k)) aren't yet handled.
    if len(sliced_output_syms) != 1:
        return [seed]
    seed_sym = sliced_output_syms[0]
    seed_factor = int(seed_per_sym[seed_sym])

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
        options.setdefault(_canonical_key(variant), variant)
        if len(options) >= DEFAULT_VARIANT_CAP:
            break

    # Let this pointwise op adopt a matmul's tiling to pin its shared buffer to
    # LX. High-value, so added regardless of DEFAULT_VARIANT_CAP (flips only).
    for base in extra_bases:
        options.setdefault(_canonical_key(base), base)
    # Load-bearing here (unlike the matmul path): the variant gate above tests
    # the element extent (extent_int % seed_factor), not the stick count, so it
    # can admit a factor that overflows the stickified dim's stick count — which
    # would SIGABRT the SDSC bundler. _split_fits_sticks drops those (and any
    # factor on a collapsed/broadcast dim). The seed is always kept: if it itself
    # is over-stick, that is work_dist's choice and not ours to discard here.
    return [
        opt for opt in options.values() if opt == seed or _split_fits_sticks(op, opt)
    ]


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
        self.reject_reasons = {}
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)

        # Enumerate options, run search, commit winners back to op_it_space_splits.
        ops = graph.operations

        # Distinct matmul output-splits (drop K) to seed the pointwise search.
        matmul_bases, matmul_roles = _find_distinct_matmul_splits(ops)

        options_per_op = [
            _enum_split_options(op, matmul_bases, matmul_roles) for op in ops
        ]
        t1 = time.perf_counter()
        best_chosen, timings, search_cache, search_lifetimes = self._search(
            graph, ops, options_per_op
        )
        t_search = time.perf_counter() - t1

        for op, opt_idx, options in zip(ops, best_chosen, options_per_op):
            chosen = options[opt_idx]
            if chosen != getattr(op, "op_it_space_splits", ({}, {})):
                op.op_it_space_splits = chosen

        n_paths = math.prod(len(o) for o in options_per_op)
        winner = {
            f"{ops[i].get_name()}({self._get_op_name(ops[i])})": options_per_op[i][
                best_chosen[i]
            ]
            for i in range(len(ops))
            if len(options_per_op[i]) > 1
        }
        logger.info(
            "co-opt search: %d paths in %.1fms (key components in "
            "_generate_buffers(): graph_view %.1fms + mem_usage %.1fms); "
            "winner=%s",
            n_paths,
            t_search * 1e3,
            timings["graph_view"] * 1e3,
            timings["mem_usage"] * 1e3,
            winner,
        )

        # try insert clone again, as what was incompatible could be compatible now
        # TODO simplify the previous pre-opt (at the beginning of this func), we will
        # run check core-div-mismatch a few times due to clone-insertion, speed-up?
        n_ops_before_clone = len(graph.operations)
        for p in self.pre_optimization_passes:
            p.apply_pass(graph)

        # Standard downstream flow on the now-fixed winning splits. Mirrors
        # DefaultAllocator.plan_allocation past the pre-passes. Reuse the search's
        # per-core-view cache + liveness only if the clone pass left the graph
        # unchanged: a clone insertion both appends an op (shifts the
        # position-indexed liveness) and rewrites input consumers' MemoryDep to read
        # the clone (changes the (op, splits, dep) cache key), so on any op-count
        # change both are stale and we rebuild from scratch (cache=lifetimes=None).
        clone_inserted = len(graph.operations) != n_ops_before_clone
        buffers = self._generate_buffers(
            graph,
            cache=None if clone_inserted else search_cache,
            lifetimes=None if clone_inserted else search_lifetimes,
        )
        allocation = self.layout_planning.plan_layout(buffers, log_lx_usage=True)
        for b in allocation:
            if b.address is None:
                self.reject_reasons[b.name] = "no room on scratchpad"
        self._push_allocation(graph, allocation)
        self._log_lx_pinning(graph)
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
    ) -> tuple[list[int], dict[str, float], dict, dict[str, list[int]]]:
        """DFS over the option cross-product, scoring each leaf via
        _score_layout. Returns (best option index per op, timing breakdown in
        seconds, _per_core_view_on_buf cache, liveness). The timing dict has
        keys `graph_view` and `mem_usage` — the two split-dependent
        shared-object builds inside _generate_buffers, which dominate per-leaf
        cost. Liveness is split-invariant and computed once here, not per leaf.
        No early-stop pruning — bounded by ≤ K^N leaves where N counts ops with
        >1 option (most return [seed]). Per-leaf cost is one full
        _generate_buffers + plan_layout pass; the `cache` param on
        _per_core_view_on_buf amortizes sympy work if it ever becomes hot. The
        cache and liveness are returned so the final commit pass can reuse them
        when the post-search clone pass leaves the graph unchanged (see
        plan_allocation).
        """
        chosen: list[int] = [0] * len(ops)
        best_total: float = math.inf
        best_chosen: list[int] = list(chosen)
        timings: dict[str, float] = {
            "graph_view": 0.0,
            "mem_usage": 0.0,
        }

        buf_total_bytes: dict[str, int] = {
            name: math.prod(buf.layout.device_layout.device_size[:-1]) * 128
            for name, buf in graph.name_to_buffer.items()
        }

        # get_read_writes() re-traces the store function over the iteration space
        # on every call and is NOT memoized upstream, yet its result is
        # split-invariant (the symbolic deps don't depend on op_it_space_splits).
        # The per-leaf _filter_ops/get_ncores path calls it for every op, so across
        # ~K^N leaves it dominates. Compute it once and thread it through as rw_cache
        # (same pattern as `cache`/`lifetimes`). Scoped to the search — the graph
        # mutates again in the commit pass, so it must not outlive the search.
        rw_cache: dict[str, ReadWrites] = {
            op.get_name(): op.get_read_writes() for op in graph.operations
        }

        # Liveness depends only on graph structure (not op.op_it_space_splits),
        # so compute it once for the whole search instead of per leaf.
        lifetimes = calculate_liveness(graph)

        # Memoize _per_core_view_on_buf across leaves. Keyed on
        # (op name, split values, dep) — see _per_core_view_on_buf for why
        # op name is required. A single dict is correct across the whole
        # search; scoped to this graph only since dep is not unique across
        # graphs.
        cache: dict = {}

        def recurse(op_idx: int) -> None:
            nonlocal best_total, best_chosen
            if op_idx == len(ops):
                hbm = self._score_layout(
                    graph, buf_total_bytes, cache, timings, lifetimes, rw_cache
                )
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
        return best_chosen, timings, cache, lifetimes

    # ------------------------------------------------------------------
    # Leaf scoring
    # ------------------------------------------------------------------

    def _score_layout(
        self,
        graph: GraphLowering,
        buf_total_bytes: dict[str, int],
        cache: Optional[dict] = None,
        timings: Optional[dict[str, float]] = None,
        lifetimes: Optional[dict[str, list[int]]] = None,
        rw_cache: Optional[dict[str, ReadWrites]] = None,
    ) -> int:
        """HBM bytes under the current split assignment: total device
        bytes of every buffer the solver couldn't pin. Non-committing
        (addresses land on throwaway buffers) and solver-agnostic.

        If `timings` is provided, _generate_buffers accumulates its
        `graph_view` / `mem_usage` sub-step seconds into it. `lifetimes`
        (split-invariant) is forwarded to avoid recomputing it per leaf;
        `rw_cache` ({op name: ReadWrites}, also split-invariant) avoids
        re-tracing get_read_writes() per leaf.
        """
        buffers = self._generate_buffers(graph, cache, timings, lifetimes, rw_cache)
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
