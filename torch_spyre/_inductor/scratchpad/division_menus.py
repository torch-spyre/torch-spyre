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


"""The candidate work divisions of an op, read from the IR: the split options
it may take, the legality and stick-fit checks, matmul and reduction axis
parsing, and the read-distance filter. The table strategy's candidate
generation (``graph_builder``) builds each node's menu from these."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Reduction,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.constants import (
    DEVICE_NAME,
    POOL_OPS,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _is_matmul_op,
    concretize_expr,
    indirect_access_subs_from_op,
    iteration_space_from_op,
    op_read_writes,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    TileSpec,
)
from torch_spyre._inductor.scratchpad.utils import (
    dep_has_constant_offset,
)
from torch_spyre._inductor.work_division import (
    _core_division,
    work_division_splits_are_legal,
)

logger = get_inductor_logger("scratchpad.division_menus")


class _DivisionMap(NamedTuple):
    """Every op's core-division candidates, and which of those lists are the
    whole legal space.

    A generated division has to be one the enumeration would have carried, so a
    solver may only generate for an op in ``enumerated``. An op pinned to its
    committed division (an offset-mutation component) or one whose candidates
    came from the pruning heuristic is deliberately narrower than its legal
    space, and is absent.
    """

    divisions: dict[str, list[CoreDivision]]
    enumerated: set[str]


def _work_slices(op, division: CoreDivision) -> dict:
    """Restore a complete symbol-keyed split map from a sparse candidate."""
    return {
        symbol: division.splits.get(symbol, 1) for symbol in iteration_space_from_op(op)
    }


def _is_cpu_host_buffer(op: Operation) -> bool:
    """True for a ComputedBuffer that is not on the Spyre device.

    CPU/host buffers participate in the joint division map (as producers or
    consumers in the slicing-match) but never reside in LX and are never
    re-sliced, so they keep their committed division.
    """
    if not isinstance(op, ComputedBuffer):
        return False
    layout = op.maybe_get_layout()
    return layout is None or layout.device.type != DEVICE_NAME


def _is_windowed_pool(op: Operation) -> bool:
    """True for a windowed pool (avgpoolfwd) reduction op.

    Its output spatial split cannot be re-chosen by the joint solver without
    risking a mis-addressed per-core input; see the pin in ``_division_map``.
    """
    return (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Reduction)
        and op.data.reduction_type in POOL_OPS
    )


def _is_indirect_access_op(op: Operation) -> bool:
    """True for a gather (``index``) or scatter (``index_put``) op.

    An indirect op accesses one operand through a runtime index
    (``IndirectAccess``): a gather reads ``src[idx]``, a scatter writes
    ``dest[idx]``. The work-division pass parallelizes these on the index-entry
    dim and never on the shared table/destination data dim (splitting the shared
    base is silently wrong). The joint solver does not preserve that split: a
    scatter's entry dim reaches ``_core_division`` as a reduction split (write
    coeff 0 through IndirectAccess) which the solver then avoids, and a gather's
    entry split is a plain tie the memory-only objective breaks toward a single
    core -- both drop the multicore entry-dim parallelism. Pin every indirect op
    to its fixed (work-division) division so co-optimization keeps the entry
    split, mirroring the keep_by_index pin. Correctness is unchanged either way
    (the shared data dim is never split); this restores the expected parallelism.
    """
    return isinstance(op, ComputedBuffer) and bool(indirect_access_subs_from_op(op))


def _reads_offset_slice(op: Operation) -> bool:
    """True for an op that reads an input at a constant (slice) offset.

    A sliced read -- ``exp(x[:, :, 32:96])`` reads its operand at index
    ``... + 32`` -- carries a non-zero constant term in the read index.
    Splitting the sliced dim across cores mis-addresses the per-core slice: the
    sliced dim is a non-stick device coordinate offset into a wider operand dim
    (``d2 + 32`` into a 128-wide dim in the restickified operand), so a per-core
    sub-range lands at a span the DSM read address cannot express -- silent ~44%
    error on ``exp(x[:, :, 32:96])`` over 128x192x256. The work-division pass
    picks a safe division for these ops (it never split the offset dim); the
    joint solver does, so pin the op to that fixed division, mirroring the
    keep_by_index pin. Correctness is unchanged (the fixed division is what the
    non-co-optimized path uses); only the offending split is removed. Blocking
    the offset dim alone is not enough -- it forces the solver onto a different
    unsafe split for a sliced reduction -- so the whole op is pinned. Indirect
    (data-dependent) offsets are handled by ``_is_indirect_access_op``.

    Shares ``dep_has_constant_offset`` with ``_writes_at_constant_offset``, the
    write-side detector behind ``ops_in_offset_mutation_component``: both ask the
    same question of a dep, so they must answer it the same way (in particular,
    per-core/coarse-tile shifts are symbolic and are not offsets).
    """
    if not isinstance(op, ComputedBuffer):
        return False
    return any(dep_has_constant_offset(dep) for dep in op_read_writes(op).reads)


def _fused_layout_group_ops(
    graph: GraphLowering, seed_reasons: dict[str, str]
) -> dict[str, str]:
    """Map each op in a fused layout group to the pin reason of its seed.

    ``seed_reasons`` maps a seed reduction type to the reason string reported
    when its group is pinned; every op in that seed's group inherits it.

    A group is one seed reduction plus the input producers it reads (one hop
    back) and the consumers of its output (one hop forward): the tightly coupled
    neighbours the work-division pass slices into a single mutually compatible
    per-core division. The joint solver, free to divide each op independently,
    can hand the group's members incompatible divisions and corrupt the shared
    per-core addressing/scheduling, so the caller pins the whole group to its
    fixed (work-division) division. Two op kinds need this identical treatment:

    * ``keepbyindex`` reproduces a fragile multi-stick search layout that its
      input restickifies and output clones carry too; an output clone splitting
      the search axis while the reduction keeps it whole drops the second search
      stick's kept values (silently wrong, ~2-3% of a 6x17x4x128 dim-3
      keep_by_index). Its own unsafe splits are separately blocked by
      ``keep_by_index_search_adjacent_blocked_vars``.
    * ``batchmatmulfp8`` fuses the fp8 quantize of its operands and the
      dequant/bias of its output into one SDSC bundle; leaving the matmul
      single-core-in-LX (``{}``) while its operands split aborts DeepTools L3
      scheduling (``distributeElemArrToTemporalLoops: Not enough elements to
      distribute``, a 4x128 @ 128x1024 fp8 scaled_mm). Only the fused neighbours
      need it -- the quantize chain feeding the operand producers is a separate
      bundle -- and a plain fp16 batchmatmul (no fused quantize) is not seeded.

    Both seeds are scanned in one pass, so adding a seed costs no extra graph
    walk.
    """
    seeds = [
        op
        for op in graph.operations
        if isinstance(op, ComputedBuffer)
        and isinstance(op.data, Reduction)
        and op.data.reduction_type in seed_reasons
    ]
    if not seeds:
        return {}
    # Reason per seed name, so producers and consumers inherit it below.
    reason_of_seed = {op.name: seed_reasons[op.data.reduction_type] for op in seeds}
    group: dict[str, str] = dict(reason_of_seed)
    # Producers of each seed's input buffers (restickifies / fp8 quantize).
    for seed in seeds:
        for dep in op_read_writes(seed).reads:
            if isinstance(dep, MemoryDep):
                group.setdefault(dep.name, reason_of_seed[seed.name])
    # Consumers of any seed output (output clones / dequant / bias-add).
    for op in graph.operations:
        if not isinstance(op, ComputedBuffer):
            continue
        for dep in op_read_writes(op).reads:
            if isinstance(dep, MemoryDep) and dep.name in reason_of_seed:
                group.setdefault(op.name, reason_of_seed[dep.name])
                break
    return group


def _fixed_core_division(op: Operation) -> CoreDivision:
    """The op's committed symbol-keyed division, or a one-core division."""
    ownership = getattr(op, "iteration_space_ownership", None)
    return _core_division(op, ownership.work_slices if ownership is not None else {})


def _legal_fixed_division(
    op: Operation, fixed: list[CoreDivision], reason: str
) -> list[CoreDivision]:
    """Return upstream division when it satisfies hard constraints."""
    division = fixed[0]
    if not isinstance(op, ComputedBuffer) or _split_option_is_legal(
        op, division.splits
    ):
        logger.debug("keep upstream division for %s: %s", op.name, reason)
        return fixed
    raise Unsupported(f"{op.name}: fixed split violates hard domain.")


def _split_option_is_legal(op: Operation, splits: dict[sympy.Symbol, int]) -> bool:
    """Return whether symbol-keyed splits satisfy hard domains."""
    return not isinstance(op, ComputedBuffer) or work_division_splits_are_legal(
        op, splits
    )


def _legal_split_options(
    op: Operation, options: Iterable[dict[sympy.Symbol, int]]
) -> list[dict[sympy.Symbol, int]]:
    """Return stick-valid candidates satisfying hard work-division domains."""
    return [
        option
        for option in options
        if _split_fits_sticks(op, option) and _split_option_is_legal(op, option)
    ]


DEFAULT_VARIANT_CAP = 6
# Try larger batch factors first. Keeping more of the batch axis whole offers
# the same reconciliation benefit with fewer co-optimization candidates.
_FACTORED_B_FACTORS: tuple[int, ...] = (8, 4, 2)


def _seed_splits(op: Operation) -> dict[sympy.Symbol, int]:
    ownership = getattr(op, "iteration_space_ownership", None)
    return {
        sym: int(ownership.work_slices.get(sym, 1)) if ownership is not None else 1
        for sym in iteration_space_from_op(op)
    }


def _output_stride_to_device_size(op: Operation) -> dict[int, int]:
    """Map each output host stride to the device size of the device dim it lands on.

    A stickified host dim decomposes into an outer-stick dim (size = stick count)
    at stride ``stick_host_stride * elems_per_stick`` and a within-stick dim at
    stride ``stick_host_stride``; sticks are atomic, so a split on that host dim
    uses the outer-stick dim. Keying by stride lets a caller look up the true
    splittable size for an output dim by its coefficient in the write index.
    (Mirrors _per_core_view_on_buf's stride→device-dim placement.)
    """
    layout = op.layout
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        # In-place mutations keep the mutation wrapper at pre-scheduler time;
        # the committed device layout lives on the mutation target.
        layout = layout.real_layout()
    dev_layout = layout.device_layout
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


def _split_fits_sticks(op: Operation, splits: dict[sympy.Symbol, int]) -> bool:
    """True if every output split divides its physical device dimension.

    A split factor must divide the device dimension it lands on. For the
    stickified host dimension that is the outer-stick count, not the element
    extent: an extent of 128 with 64 elements per stick has only two splittable
    sticks. Reduction-only symbols are absent from the write index and are not
    constrained here; work-division bounds those separately.

    A positive-coefficient output symbol with no device-stride entry is
    unplaceable (for example, a collapsed or broadcast dimension), so reject it
    rather than relying on modulo arithmetic with a missing size.
    """
    layout = op.layout
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    # CPU ComputedBuffers participate in the joint division map but never in LX
    # placement. They have no device geometry to validate against.
    if not isinstance(layout, FixedTiledLayout):
        return True
    write = next(iter(op_read_writes(op).writes), None)
    if write is None:
        return False
    sizes = _output_stride_to_device_size(op)
    for sym, factor in splits.items():
        stride = concretize_expr(write.index.coeff(sym))
        size = sizes.get(stride, 0)
        if factor > 1 and stride and (not size or size % factor):
            return False
    return True


def _output_axis_symbols(
    write: sympy.Expr, iter_syms: dict[sympy.Symbol, sympy.Expr]
) -> dict[int, sympy.Symbol]:
    """Map each output stride in ``write`` to the iteration symbol it scales.

    Only iteration symbols are axes. Under dynamic shapes the index also carries
    size symbols (``d0*s20 + d1``), whose coefficients are loop variables rather
    than strides, so ``write.free_symbols`` cannot be used directly.
    """
    return {
        concretize_expr(write.coeff(sym)): sym
        for sym in iter_syms
        if sym in write.free_symbols
    }


def _matmul_axis_parse(op: Operation) -> dict[str, tuple[sympy.Symbol, int, int]]:
    """Parse a matmul into ``{B|M|N|K: (symbol, extent, seed_factor)}``.

    Output symbols sorted by ascending write-index stride are N, M, B (with B
    absent for 2D matmuls); the symbol added by a read index is K. Output
    extents come from stick-aware device geometry, so generated split factors
    divide stick counts rather than element extents. The returned symbols and
    factors remain local and symbol-keyed.
    """
    rw = op_read_writes(op)
    write = next(iter(rw.writes)).index
    read = next((dep.index for dep in rw.reads), write)
    iter_syms = iteration_space_from_op(op)
    out_syms = _output_axis_symbols(write, iter_syms)
    k_syms = {sym for sym in iter_syms if sym in read.free_symbols}
    k_syms -= write.free_symbols
    if not k_syms:
        raise ValueError(f"matmul {op.get_name()} has no reduction axis")
    sizes = _output_stride_to_device_size(op)
    seed = _seed_splits(op)
    roles: dict[str, tuple[sympy.Symbol, int, int]] = {}
    for role, stride in zip(("N", "M", "B"), sorted(out_syms)):
        sym = out_syms[stride]
        roles[role] = (sym, sizes[stride], seed[sym])
    k_sym = min(k_syms, key=str)
    roles["K"] = (
        k_sym,
        concretize_expr(iter_syms[k_sym]),
        seed[k_sym],
    )
    return roles


def _bm_axes_from_roles(
    roles: dict[str, tuple[sympy.Symbol, int, int]],
) -> tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]] | None:
    """Return B/M ``(symbol, extent)`` pairs, or ``None`` when either is absent."""
    b, m = roles.get("B"), roles.get("M")
    return ((b[0], b[1]), (m[0], m[1])) if b is not None and m is not None else None


def _reduction_bm_axes(
    op: Operation,
) -> tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]] | None:
    """Return stick-aware B/M output axes for a non-matmul reduction.

    A reduction over N keeps B and M in its write. As in
    :func:`_matmul_axis_parse`, the largest output stride is B and the next is
    M. Reductions with fewer than two output axes cannot use this factorization.
    """
    write = next(iter(op_read_writes(op).writes)).index
    out_syms = _output_axis_symbols(write, iteration_space_from_op(op))
    if len(out_syms) < 2:
        return None
    m_stride, b_stride = sorted(out_syms)[-2:]
    sizes = _output_stride_to_device_size(op)
    return (out_syms[b_stride], sizes[b_stride]), (out_syms[m_stride], sizes[m_stride])


def _factored_bm_splits(
    bm_axes: tuple[tuple[sympy.Symbol, int], tuple[sympy.Symbol, int]] | None,
) -> list[dict[sympy.Symbol, int]]:
    """Offer at most one largest-B full-core B/M factorization.

    Smaller B factors do not add a useful reconciliation option but multiply the
    co-optimization search space. An empty result means B/M is absent or no
    factorization divides both stick-aware extents.
    """
    if bm_axes is None:
        return []
    (b_sym, b_size), (m_sym, m_size) = bm_axes
    for b_factor in _FACTORED_B_FACTORS:
        m_factor = config.sencores // b_factor
        if (
            b_factor * m_factor == config.sencores
            and b_size % b_factor == 0
            and m_size % m_factor == 0
        ):
            return [{b_sym: b_factor, m_sym: m_factor}]
    return []


def _candidate_key(splits: dict[sympy.Symbol, int]) -> tuple[tuple[str, int], ...]:
    """Return a stable, symbol-orderable deduplication key for one candidate."""
    return tuple(sorted(((str(sym), factor) for sym, factor in splits.items())))


def _output_profile(op: Operation, splits: dict[sympy.Symbol, int]) -> dict[int, int]:
    """Project output splits onto physical strides for cross-operation transfer.

    This is temporary candidate-generation metadata only; callers immediately
    reconstruct a symbol-keyed candidate for the target operation.
    """
    write = next(iter(op_read_writes(op).writes)).index
    return {
        concretize_expr(write.coeff(sym)): factor
        for sym, factor in splits.items()
        if factor > 1 and write.coeff(sym) != 0
    }


def _from_output_profile(
    op: Operation, profile: dict[int, int]
) -> dict[sympy.Symbol, int]:
    """Apply a transient physical output profile as a symbol-keyed candidate."""
    write = next(iter(op_read_writes(op).writes)).index
    return {
        sym: profile.get(concretize_expr(write.coeff(sym)), 1)
        if write.coeff(sym) != 0
        else 1
        for sym in iteration_space_from_op(op)
    }


def _find_distinct_matmul_splits(
    ops: list[Operation],
) -> tuple[tuple[dict[int, int], ...], tuple[dict[str, int], ...]]:
    """Collect distinct physical profiles and B/M/N/K factors from matmul seeds.

    Profiles let intervening pointwise operations offer a matching output
    division. Role factors let another matmul transfer a split despite using
    different local iteration symbols. Both are transient inputs to candidate
    generation; committed divisions remain symbol-keyed.
    """
    profiles: list[dict[int, int]] = []
    roles: list[dict[str, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for op in ops:
        if not _is_matmul_op(op):
            continue
        parsed = _matmul_axis_parse(op)
        candidates = [_output_profile(op, _seed_splits(op))]
        candidates += [
            _output_profile(op, candidate)
            for candidate in _factored_bm_splits(_bm_axes_from_roles(parsed))
        ]
        for profile in candidates:
            key = tuple(sorted(profile.items()))
            if profile and key not in seen:
                seen.add(key)
                profiles.append(profile)
        if candidates[0]:
            roles.append(
                {role: factor for role, (_sym, _size, factor) in parsed.items()}
            )
    return tuple(profiles), tuple(roles)


def _check_and_add_matmul_options(
    op: Operation,
    seed: dict[sympy.Symbol, int],
    matmul_roles: tuple[dict[str, int], ...],
) -> list[dict[sympy.Symbol, int]]:
    """Offer seed, cross-matmul, and factored B/M candidates for ``op``.

    Work distribution may choose incompatible splits for two matmuls joined by
    pointwise/reduction operations. Transferring each source's B/M/N/K factors
    gives the co-optimizer a chance to choose a compatible assignment. Missing
    roles or non-divisible extents default to one; view matching later rejects
    candidates that cannot share a physical buffer view.
    """
    parsed = _matmul_axis_parse(op)
    options = {_candidate_key(seed): seed}
    for source in matmul_roles:
        candidate = dict.fromkeys(iteration_space_from_op(op), 1)
        for role, (sym, extent, _factor) in parsed.items():
            factor = source.get(role, 1)
            candidate[sym] = factor if factor > 1 and extent % factor == 0 else 1
        options.setdefault(_candidate_key(candidate), candidate)
    for candidate in _factored_bm_splits(_bm_axes_from_roles(parsed)):
        full_candidate = {sym: 1 for sym in iteration_space_from_op(op)}
        full_candidate.update(candidate)
        options.setdefault(_candidate_key(full_candidate), full_candidate)
    return _legal_split_options(op, options.values())


def _enum_split_options(
    op: Operation,
    extra_profiles: tuple[dict[int, int], ...] = (),
    matmul_roles: tuple[dict[str, int], ...] = (),
) -> list[dict[sympy.Symbol, int]]:
    """Enumerate symbol-keyed candidates for the pruned co-optimization path.

    Matmuls use role transfer plus B/M factorizations. Other reductions offer
    B/M factorizations only because their reduction axis is fixed. Pointwise
    ops can move a single output split to another divisible output axis and can
    adopt a matmul's transient physical output profile. The seed is always kept;
    non-seed candidates must fit physical stick geometry.
    """
    seed = _seed_splits(op)
    is_computed = isinstance(op, ComputedBuffer)
    is_reduction = is_computed and isinstance(op.data, Reduction)
    is_matmul = is_reduction and _is_matmul_op(op)
    seed_profile = _output_profile(op, seed)
    if is_matmul and matmul_roles:
        return _check_and_add_matmul_options(op, seed, matmul_roles)
    if is_reduction:
        if not seed_profile:
            return [seed]
        options = {_candidate_key(seed): seed}
        for candidate in _factored_bm_splits(_reduction_bm_axes(op)):
            full_candidate = dict.fromkeys(iteration_space_from_op(op), 1)
            full_candidate.update(candidate)
            options.setdefault(_candidate_key(full_candidate), full_candidate)
        return _legal_split_options(op, options.values())
    if not is_computed or not seed_profile:
        return [seed]

    write = next(iter(op_read_writes(op).writes)).index
    sliced = [
        sym for sym, factor in seed.items() if factor > 1 and write.coeff(sym) != 0
    ]
    options = {_candidate_key(seed): seed}
    if len(sliced) == 1:
        source = sliced[0]
        factor = seed[source]
        for sym, extent in iteration_space_from_op(op).items():
            if (
                sym is not source
                and write.coeff(sym) != 0
                and (size := concretize_expr(extent)) > 1
                and size % factor == 0
            ):
                candidate = dict(seed)
                candidate[source], candidate[sym] = 1, factor
                options.setdefault(_candidate_key(candidate), candidate)
                if len(options) >= DEFAULT_VARIANT_CAP:
                    break
    for profile in extra_profiles:
        candidate = _from_output_profile(op, profile)
        options.setdefault(_candidate_key(candidate), candidate)
    return _legal_split_options(op, options.values())


def _op_read_span_is_evaluable(op: Operation) -> bool:
    """True when the read-distance filter can compute concrete post-tile spans.

    The span math needs a ``ComputedBuffer`` with a ``FixedTiledLayout`` whose
    ``device_layout`` is fully static (integer sizes, strides, stride map, stick
    size). For anything else -- a symbolic shape, a layout without
    ``device_layout`` -- the filter cannot prove a violation, so it leaves the
    option set untouched (fail open) rather than dropping candidates it cannot
    evaluate.
    """
    from torch_spyre._inductor.wsr.span_overflow_hint_analysis import (
        _layout_has_static_span_metadata,
    )

    if not isinstance(op, ComputedBuffer):
        return False
    try:
        layout = op.get_layout()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False
    if getattr(layout, "device_layout", None) is None:
        return False
    try:
        return _layout_has_static_span_metadata(layout)
    except (AttributeError, TypeError, ValueError):
        return False


def _spec_within_read_distance(
    op: ComputedBuffer, spec: TileSpec, max_cores: int
) -> bool:
    """True when tiling ``op`` by ``spec`` keeps every per-core read span within
    ``MAX_SPAN_BYTES`` (the MVLOC read-distance limit).

    Reuses ``_remaining_span_candidates_after_tile`` -- the post-tile span
    validator the span-overflow planner uses -- so a candidate is admitted only
    when it leaves no overflowing span, exactly as ``_search_min_cost_tile_plan``
    requires. The untiled ``spec`` (no axes) validates the op's own full-size
    read, so an op whose untiled read already overflows fails here.

    ``max_cores`` matches the planner's ``config.sencores`` gate, so the two
    agree on what "fits". Fails open: any evaluation error (e.g. a post-tile
    layout that cannot be reconstructed for this candidate) keeps the spec
    rather than dropping a possibly-valid tiling.
    """
    from torch_spyre._inductor.wsr.span_overflow_hint_analysis import (
        _remaining_span_candidates_after_tile,
    )

    split_by_host_dim = {
        axis.host_dim: axis.count for axis in spec.axes if not axis.is_reduction
    }
    k_split = next((axis.count for axis in spec.axes if axis.is_reduction), None)
    try:
        remaining = _remaining_span_candidates_after_tile(
            op, max_cores, split_by_host_dim, k_split=k_split
        )
    except (
        Unsupported,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
        IndexError,
    ):
        return True
    return not remaining


def _drop_read_distance_violations(
    op: Operation, options: list[TileSpec], max_cores: int
) -> list[TileSpec]:
    """Drop every discovered tiling whose per-core span exceeds the read-distance
    limit, raising ``Unsupported`` when none fit.

    The CP-SAT discovery path (``auto_coarse_tiling``) enumerates tilings with no
    span term in its cost model, so a chosen ``TileSpec`` is never otherwise
    checked against ``MAX_SPAN_BYTES``. This applies the span-overflow planner's
    own gate to the discovered option set: a coarse tiling only shrinks a
    per-core span, so an op under no pressure loses nothing, but the untiled
    option (and any tiling that fails to bring an overflowing read back under the
    limit) is removed for an op whose full-size read overflows -- the solve is
    then forced to pick a tiling that fits. When *no* candidate satisfies the
    limit it raises rather than returning an empty set, matching
    ``_search_min_cost_tile_plan``: an op that cannot be tiled to fit must abort,
    not fall through to an over-span plan.

    Ops whose span is not statically evaluable are returned unchanged (fail
    open): the filter only drops what it can prove violates the limit.
    """
    if not _op_read_span_is_evaluable(op):
        return options
    surviving = [
        spec for spec in options if _spec_within_read_distance(op, spec, max_cores)
    ]
    if not surviving:
        from torch_spyre._inductor.work_division import MAX_SPAN_BYTES

        raise Unsupported(
            f"Cannot tile {op.get_name()}: no discovered coarse tiling keeps the "
            "per-core read span within the read-distance limit of "
            f"{MAX_SPAN_BYTES / (1024**2):.3f} MB; the untiled read and every "
            "candidate tiling overflow it."
        )
    return surviving


def _intern_view_group(groups: dict[PerCoreView, int], view: PerCoreView) -> int:
    """Group index of ``view`` among ``groups``, keyed by physical ownership.

    A dict keyed on ``PerCoreView`` interns by structural equality, so two slot
    spellings of one partition (``core_id`` and ``core_id % 4`` over four cores)
    would become two groups and the solver would price two shuffles where one
    copy serves both consumers. Look up by ``same_partition`` instead and only
    then mint a new index; the first spelling seen stays the dict key.
    """
    for known, index in groups.items():
        if known.same_partition(view):
            return index
    index = len(groups)
    groups[view] = index
    return index
