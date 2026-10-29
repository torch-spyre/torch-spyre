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

"""Shared tiling and index-rewriting helpers for the loop passes.

These helpers were moved verbatim out of ``coarse_tile.py``, which re-exports
them.  ``for_each_tile_lowering.py`` and the span-overflow planner import them
from here so they do not depend on the coarse-tiling pass itself.
"""

from __future__ import annotations


from typing import NamedTuple

import sympy
from sympy import Expr

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.utils import sympy_index_symbol, sympy_subs
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    Layout,
    Loops,
    Operation,
    Pointwise,
    Reduction,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


from ..errors import Unsupported
from ..logging_utils import get_logger
from ..propagate_hints import DimHint
from ..pass_utils import (
    op_out_coords,
    host_coordinates,
    indirect_sizes_from_op,
    invalidate_op_read_writes,
    iteration_space_from_op,
    loop_var_ranges_from_dim_hints,
    op_read_writes,
    replace_computed_buffer_body,
)
from ..ir import FixedTiledLayout, _resize_device_layout
from .tile import compute_tile_index, compute_tile_stride, decompose_index_for_tiling


logger = get_logger("wsr.coarse_tile")


class _RetiledBufferInfo(NamedTuple):
    """Host shape/strides before and after a coarse-tile resize."""

    old_stride: tuple[Expr, ...]
    new_stride: tuple[Expr, ...]
    old_size: tuple[Expr, ...]
    new_size: tuple[Expr, ...]


class _LogicalIterationSymbol(NamedTuple):
    """One active loop symbol keyed by its stable raw dimension identity."""

    logical_dim: tuple[str, int]
    extent: Expr
    symbol: sympy.Symbol


class _IterationSymbolRemap(NamedTuple):
    """Order-preserving loop-symbol translation produced by a range rewrite."""

    before_symbols: tuple[sympy.Symbol, ...]
    pairs: tuple[tuple[sympy.Symbol, sympy.Symbol], ...]


class _DivideRangesResult(NamedTuple):
    retiled_info: _RetiledBufferInfo | None
    symbol_remap: _IterationSymbolRemap | None


def _cache_key(cached_method: object) -> str:
    """Return the cache attribute name used by a cache_on_self / cache_on_self_and_args method.

    cache_on_self uses key ``f"__{fn.__name__}_cache"``; cache_on_self_and_args uses
    ``f"__{class_name}_{fn.__name__}_cache"``.  Both patterns are captured as the
    ``key`` free variable in the method's ``.clear_cache`` closure — extract it once
    at module load so misspellings or upstream renames fail loudly on import.
    """
    clear_fn = getattr(cached_method, "clear_cache")  # AttributeError if absent
    for i, name in enumerate(clear_fn.__code__.co_freevars):
        if name == "key":
            return clear_fn.__closure__[i].cell_contents
    raise AttributeError(
        f"Cannot find 'key' in clear_cache closure of {cached_method!r}"
    )


# Resolve cache keys once at import time — any rename in upstream IR will raise
# AttributeError here rather than silently no-oping at runtime.
_LOOPS_FREE_SYMS_KEY = _cache_key(Loops.get_free_symbol_uses)
_LOOPS_INNER_FN_STR_KEY = _cache_key(Loops.inner_fn_str)
_LOOPS_INNER_FN_OPCOUNT_KEY = _cache_key(Loops.inner_fn_opcount)
_REDUCTION_FREE_SYMS_KEY = _cache_key(Reduction.get_free_symbol_uses)
_LAYOUT_FREE_SYMS_KEY = _cache_key(Layout.get_free_symbol_uses)
_COMPUTED_BUF_FREE_SYMS_KEY = _cache_key(ComputedBuffer.get_free_symbol_uses)
_COMPUTED_BUF_SIZES_KEY = _cache_key(ComputedBuffer.get_default_sizes_body)


def _clear_cache(obj: object, key: str) -> None:
    # cache_on_self/cache_on_self_and_args store results via object.__setattr__ to
    # bypass frozen-dataclass guards (Loops, Reduction, Layout); clearing must also
    # use object.__delattr__ — plain delattr() raises FrozenInstanceError.
    if hasattr(obj, key):
        object.__delattr__(obj, key)


def _raw_to_squeezed_pos(ir_node: ComputedBuffer) -> dict[int, int]:
    """Map ir_node's raw host-range positions to their squeezed d{i} number.

    Mirrors SpyreKernel._host_dim_to_index_symbol's own squeeze arithmetic
    (same source of truth, duplicated here because that method maps one
    dim at a time and _tiled_dims_for_dep needs the whole table to build a
    dep_dims membership test): output dims are numbered densely over
    ir_node.data.ranges, skipping unit-size (==1) entries; reduction dims
    continue the same counter, offset by n_output_dims, over
    ir_node.data.reduction_ranges. A raw dim squeezed out entirely (range
    == 1) has no entry -- it has no d{i} symbol for any dep.index to
    reference.
    """
    pos: dict[int, int] = {}
    it_idx = 0
    ranges = getattr(getattr(ir_node, "data", None), "ranges", None)
    if ranges is None:
        return pos
    for host_idx, r in enumerate(ranges):
        if int(r) != 1:
            pos[host_idx] = it_idx
            it_idx += 1
    # Raw reduction-dim keys are stored offset by the RAW (un-squeezed)
    # output-dim count -- see _planned_tile_extents_per_level's
    # n_output_dims = len(op.data.ranges). But the d{i} symbol they map to
    # continues from the SQUEEZED output-dim count (it_idx above) -- see
    # SpyreKernel._host_dim_to_index_symbol's n_output_dims = it_idx. These
    # two offsets differ whenever ranges contains a unit dim, so they must
    # not be conflated.
    raw_n_output_dims = len(ranges)
    squeezed_n_output_dims = it_idx
    reduction_ranges = getattr(ir_node.data, "reduction_ranges", None) or []
    red_it_idx = 0
    for host_idx, r in enumerate(reduction_ranges):
        if int(r) != 1:
            pos[raw_n_output_dims + host_idx] = squeezed_n_output_dims + red_it_idx
            red_it_idx += 1
    return pos


def _affine_point_read_index(
    index: Expr, loop_vars: set[sympy.Symbol]
) -> tuple[Expr, dict[sympy.Symbol, Expr]]:
    """Decompose a point-read index into base plus affine loop advances."""
    index = sympy.expand(sympy.sympify(index))
    active_loop_vars = loop_vars & index.free_symbols
    if not active_loop_vars:
        return index, {}

    coefficients: dict[sympy.Symbol, Expr] = {}
    for loop_var in active_loop_vars:
        coefficient = sympy.diff(index, loop_var)
        if coefficient.free_symbols:
            raise Unsupported(
                f"point-read index {index} is not affine in spliced loop "
                f"variables {sorted(map(str, loop_vars))} with a constant stride"
            )
        coefficients[loop_var] = coefficient

    base = sympy.simplify(
        index
        - sum(
            (coefficient * loop_var for loop_var, coefficient in coefficients.items()),
            sympy.S.Zero,
        )
    )
    if base.free_symbols & loop_vars:
        raise Unsupported(
            f"point-read index {index} is not a loop-invariant base plus "
            "constant per-level advances"
        )
    reconstructed = base + sum(
        (coefficient * loop_var for loop_var, coefficient in coefficients.items()),
        sympy.S.Zero,
    )
    if sympy.simplify(index - reconstructed) != 0:
        raise Unsupported(f"could not safely decompose point-read index {index}")
    return base, coefficients


def _stick_host_dim(op: ComputedBuffer, device_layout) -> int | None:
    """Authoritative stick host-dim index for ``op``'s output, recovered from
    coordinate identity (issue #3116).

    ``SpyreTensorLayout`` discards its ``dim_map`` at construction, so the
    host<->device dim identity is not carried on the layout object.  We recover
    only the stick host dim: the device layout's inner-stick coordinate has a
    single iteration symbol that also drives exactly one host coordinate, so
    ``matching_dim`` resolves it unambiguously — even when two host dims share a
    size (transposed flash-attn QK^T with ``Sq == Skv``), which defeats the
    size-based inference in ``_resize_device_layout``.

    This is the same identity mechanism ``_pick_stick_dim`` uses to choose a
    stick dim, so it is as reliable as the existing stick logic.  Returns
    ``None`` when identity cannot be resolved (single-symbol match not unique),
    so the caller falls back to size-based inference.

    The stick host dim is invariant under coarse tiling (tiling shrinks a range
    but does not change which axis is the stick), so this may be computed either
    before or after ``_divide_ranges`` mutates the ranges.
    """
    from ..pass_utils import (
        try_device_coordinates,
    )
    from ..views import matching_dim

    try:
        writes = op.get_read_writes().writes
        if not writes:
            return None
        out_dep = next(iter(writes))
        ind_sizes = indirect_sizes_from_op(op)
        dcoords = try_device_coordinates(device_layout, out_dep, ind_sizes)
        if not dcoords:  # None (unrepresentable stick) or empty → no identity
            return None
        hcoords = host_coordinates(op.get_layout(), out_dep, ind_sizes)
        return matching_dim(hcoords, dcoords[-1])
    except Exception:
        # Identity recovery is best-effort; any failure falls back to inference.
        return None


def _op_reads(op: ComputedBuffer) -> set[str]:
    """Return the set of buffer names op reads (via MemoryDep).

    Uses the memoized op_read_writes() helper: every call site of
    _op_reads is confined to planning-time (zero-mutation) contexts, so a
    memo scoped to the op instance cannot go stale here -- see
    _reads_buffer_cached's docstring for the general argument.
    """
    return {d.name for d in op_read_writes(op).reads if isinstance(d, MemoryDep)}


def _loop_var_to_ranges_pos(out_coords: list, sym: sympy.Symbol) -> int | None:
    """Return the position of loop variable sym in op.data.ranges, or None.

    Looks up sym in the op's output coordinates — the only reliable mapping
    from a loop variable symbol to its data.ranges position, since dep var
    numbering skips size-1 dims while data.ranges does not.

    Matches if sym is the coordinate's sole free symbol (the ordinary
    ``spyre_hint()`` case -- also covers a non-polynomial wrapper like
    ``floor(h)``, whose ``.coeff(sym)`` is 0 even though sym is clearly its
    only variable) OR sym has a nonzero coefficient in the coordinate (the
    WhileLoop-splice case below). A coefficient-only test would silently
    drop the ordinary case whenever compute_coordinates wraps a non-innermost
    dim's coordinate in ``floor()``, which it does whenever the dim isn't
    the fastest-varying one -- exactly span-overflow's tiled H dim (BHLD's
    dim 1) in a real (non-test-stubbed) op_out_coords call, so this must stay
    a two-way OR, not a coefficient-only test.

    A WhileLoop-splice loop_var (e.g. u0, see for_each_tile_lowering.py's
    _synthesize_dim_hints_for_group) can share a device coordinate with an
    already-tiled ordinary dim when the spliced body's per-iteration advance
    lands in the same host dim as that dim's own tiling -- e.g. coordinate
    ``d0 + 2*u0`` for a 2-row-per-iteration write into a dim tiled to size 2
    -- so requiring sym to be the ONLY free symbol never matches that case;
    the coefficient test is what catches it. Callers that consume this
    position (_tiled_dims_for_dep's _dim_is_read) already use the same
    coefficient test to decide whether a dependency reads a dim, so this
    keeps the two symbol/pos mappings consistent for that case.
    """
    for i, coord in enumerate(out_coords):
        free = coord.free_symbols
        if sym not in free:
            continue
        if len(free) == 1 or coord.coeff(sym) != 0:
            return i
    return None


def _splice_loop_vars(op: ComputedBuffer) -> set[sympy.Symbol]:
    """Every WhileLoop-splice loop_var symbol folded into op's own indices.

    Identified, as everywhere else in this file, by the DimHint carrying a
    non-None ``loop_var_range`` (see propagate_hints.py's DimHint
    docstring); ordinary ``spyre_hint()`` loop vars are real ``dep.ranges``
    keys and never appear here.
    """
    return set(loop_var_ranges_from_dim_hints(op))


def _hint_ranges_pos(
    op: ComputedBuffer, hint: DimHint, out_coords: list
) -> "tuple[int | None, bool]":
    """One DimHint's loop_var -> (position, is-a-reduction-dim).

    The single resolution every WhileLoop-splice-aware caller in this module
    uses, so the position a dim is planned at, divided at, and advanced at
    can never disagree. The position indexes ``op.data.ranges`` when the
    second element is False and ``op.data.reduction_ranges`` when it is True.

    ``DimHint.is_reduction`` is the caller's declared intent; the returned
    flag is what actually resolved. They agree for every ordinary
    ``spyre_hint()`` scope. For a WhileLoop-splice hint the resolution is
    authoritative: the synthesizer stamps one hint per op for the whole
    level and cannot know, per op, whether that level lands on an output dim
    or a reduction dim of THAT op (``split_m_fn``'s matmul: output dim M;
    ``split_k_fn``'s matmul: reduction dim K -- same synthesized hint, same
    loop_var, opposite answers).

    The read-side resolution (``for_each_tile_lowering.lookup_marker_dim``,
    a marker-map lookup -- NOT a heuristic; it replaced the deleted
    ``_loop_var_pos_from_reads`` numeric-coincidence guess) fires ONLY for a
    WhileLoop-splice hint, identified exactly as everywhere else in this
    file by ``loop_var_range is not None`` (see propagate_hints.py's
    DimHint docstring). An ordinary ``spyre_hint()`` scope keeps the
    pre-existing output-coordinates-only behaviour unchanged: its op is the
    full, untiled write, so its loop_var is always present in the output
    coordinates, and a read-derived guess there could only ever contradict
    the authoritative output mapping.
    """
    if hint.loop_var_range is None:
        # Ordinary spyre_hint() scope: unchanged pre-existing behaviour, with
        # the hint's own is_reduction selecting the lookup channel.
        if hint.is_reduction:
            if not isinstance(op.data, Reduction):
                return None, True
            return _loop_var_to_reduction_ranges_pos(op, hint.loop_var), True
        return _loop_var_to_ranges_pos(out_coords, hint.loop_var), False

    pos = _loop_var_to_ranges_pos(out_coords, hint.loop_var)
    if pos is not None:
        return pos, False
    if isinstance(op.data, Reduction):
        rpos = _loop_var_to_reduction_ranges_pos(op, hint.loop_var)
        if rpos is not None:
            return rpos, True

    from torch_spyre._inductor.wsr.for_each_tile_lowering import (
        _marker_dim,
        lookup_marker_dim,
    )

    # op itself may BE a surviving (STAR_DEP_KEPT) tile_dim_marker, not a
    # consumer reading one. lookup_marker_dim's _MARKER_MAPS lookup is keyed
    # by (consumer_name, dep) -- see _consume_tile_dim_markers, which never
    # records an entry keyed by the marker's own name -- so it cannot
    # resolve loop_var here even though the marker's read genuinely mentions
    # it (e.g. the outer WhileLoop's induction symbol baked into the
    # marker's own read index by lower_tile_dim_marker). tile_marker_dim is
    # stamped directly on the marker by lower_tile_dim_marker as the
    # marker's own output-coordinate position (dim indexes x.get_size(),
    # the same Pointwise ranges op_out_coords resolves against here) -- a
    # marker is always Pointwise, never Reduction, so this position is
    # always a non-reduction output dim, never a reduction dim.
    marker_dim = _marker_dim(op)
    if marker_dim is not None:
        return marker_dim, False

    resolved = lookup_marker_dim(op, hint.loop_var)
    if resolved is not None:
        return resolved

    # No read resolved through the marker map. That is the CORRECT,
    # expected outcome for an op that is genuinely loop-invariant at this
    # level -- e.g. a scalar bookkeeping op (iteration-counter update,
    # etc.) whose reads/writes never mention hint.loop_var at all, the
    # same "no tiled dim recorded" outcome the deleted
    # _loop_var_pos_from_reads produced for such ops (see its own
    # docstring and _synthesize_dim_hints_for_group's, which both treat
    # "no dim resolves" as legitimate, not exceptional). Only raise when
    # some read's index actually mentions loop_var -- that is the one case
    # a for_each_tile tile read is expected to be marker-tagged, so an
    # unresolved marker lookup there is a genuine gap (an unrecognized
    # shape, or a marker consumed without being recorded), not a
    # loop-invariant op silently passing through.
    #
    # A POINT read (``dep.var_names == ()``) is excluded from this check.
    # It has no iteration dim of its own to tag with a tile_dim_marker --
    # paged attention's in-body page-index read is exactly this shape,
    # ``dep.index == 32*u0`` with no ``d{i}`` coordinates at all -- so
    # loop_var can appear in its index without ever being resolvable
    # through the marker map. That is not a gap: it is the shape
    # _point_splice_advance_for_dep's caller (plan_coarse_tile_groups)
    # handles through squeezed_advance_per_read once this function returns
    # (None, False) for it, by stashing hint.loop_var into
    # unattributed_loop_vars. Counting it here as "mentions loop_var"
    # would raise on every legitimate point-splice-advance read instead of
    # letting that caller-side mechanism resolve it.
    rw = op.get_read_writes()
    mentions_loop_var = any(
        isinstance(dep, MemoryDep)
        and dep.var_names
        and isinstance(dep.index, sympy.Basic)
        and hint.loop_var in dep.index.free_symbols
        for dep in rw.reads
    )
    if mentions_loop_var:
        raise AssertionError(
            f"WhileLoop-splice hint's loop_var {hint.loop_var} appears in "
            f"op {op.get_name()!r}'s own read index, but has no "
            "tile_dim_marker entry for it, and no output/reduction-"
            "coordinate resolution either. Every for_each_tile tile is "
            "tagged by tile_dim_marker at construction time "
            "(for_each_tile.py's _tile()) -- an untagged read here means "
            "either a for_each_tile shape this pass does not yet "
            "recognize, or a marker that was consumed (erased) without "
            "being recorded. Silently guessing the tiled dimension risks "
            "the exact wrong-answer bug this mechanism replaces; raising "
            "here surfaces the gap instead."
        )
    return None, False


def reduction_loop_vars(op: ComputedBuffer) -> list[sympy.Symbol]:
    """Return the op's reduction loop variables, ordered as in
    ``op.data.reduction_ranges``.

    Uses dep-tracking symbols (d0, d1, ...) rather than SymT.R0_INDEX symbols
    (r0_0, r0_1, ...) which are a different namespace.  Finds reduction symbols
    by set-subtracting output index symbols from input index symbols, in
    dep.ranges order (which matches reduction_ranges order).

    This is the single source of truth for that derivation. Both directions go
    through it: ``_loop_var_to_reduction_ranges_pos`` (loop_var -> position) and
    coarse tiling's reduction-axis lowering (its inverse, position -> loop_var,
    in ``scratchpad.coarse_tiling.tile_spec_to_dim_hints``).

    A fused pointwise prologue can make the reduction read more than one
    operand (e.g. ``(x * bias[:, None]).sum(1)``), and a broadcast operand
    carries a strict subset of the reduction symbols (often none).  Pick the
    read dep that indexes the *most* reduction symbols, so a leading broadcast
    operand neither yields an empty result nor a short, mis-positioned list
    (consumers index this positionally by reduction-range position).
    """
    assert isinstance(op.data, Reduction)
    rw = op.get_read_writes()
    out_dep = next(iter(rw.writes))
    out_syms = out_dep.index.free_symbols
    # isinstance, not hasattr(d, "index"): StarDep.index is a property that
    # raises NotImplementedError rather than being absent, and hasattr only
    # swallows AttributeError -- a StarDep anywhere in rw.reads (e.g. a
    # bucketize read) would otherwise crash this list comprehension, which
    # (unlike the single next(...) it replaced) evaluates the predicate on
    # every dep rather than stopping at the first match.
    in_deps = [d for d in rw.reads if isinstance(d, MemoryDep)]
    if not in_deps:
        return []
    in_dep = max(
        in_deps,
        key=lambda d: len([s for s in d.ranges if s not in out_syms]),
    )
    return [s for s in in_dep.ranges if s not in out_syms]


def _loop_var_to_reduction_ranges_pos(
    op: ComputedBuffer, sym: sympy.Symbol
) -> int | None:
    """Return position of loop variable sym in op.data.reduction_ranges, or None.

    ``reduction_loop_vars`` is *squeezed*: Inductor mints no loop variable for a
    size-1 dim (``SqueezeView.squeezer``, whose ``!= 1`` test this mirrors), so
    the k-th reduction loop variable belongs to the k-th reduction dim whose
    extent is not 1. None also when the loop variables do not pair one-to-one
    with those dims, since no position is then trustworthy.
    """
    red_vars = reduction_loop_vars(op)
    if sym not in red_vars:
        return None
    not_one = [i for i, r in enumerate(op.data.reduction_ranges) if r != 1]
    if len(not_one) < len(red_vars):
        return None
    return not_one[red_vars.index(sym)]


def _loop_var_hinted_ranges(op: ComputedBuffer) -> dict[int, Expr]:
    """Return {pos in op.data.ranges: loop_var_range} for op's dim_hints.

    A WhileLoop-splice-synthesized DimHint (Task 5's
    _synthesize_dim_hints_for_group) marks its dim via loop_var_range being
    non-None -- see propagate_hints.py's DimHint docstring. For such a dim,
    op.data.ranges[pos] is ALREADY the per-iteration extent (the spliced
    body op's own shape); the hint's loop_var_range (the loop's trip count)
    is a multiplier over a larger extent that is never materialized as a
    real dim anywhere -- never a divisor of op.data.ranges[pos]. Callers
    that would otherwise divide a tiled dim's range by the level's loop
    count (_planned_tile_extents_per_level, _compute_per_tile_ranges_planned,
    _divide_ranges) must check this first and skip the divide for any pos
    present here, matching the resolution pattern _tiled_dims_for_dep
    already uses (pos_to_loop_var, built the same way, for its own
    coefficient-based dep.index test).
    """
    hints = getattr(op, "dim_hints", None) or ()
    if not hints:
        return {}
    var_ranges = loop_var_ranges_from_dim_hints(op)
    if not var_ranges:
        return {}
    out_coords = op_out_coords(op)
    result: dict[int, Expr] = {}
    for hint in hints:
        if hint.loop_var is None:
            continue
        rng = var_ranges.get(hint.loop_var)
        if rng is None:
            continue
        # _hint_ranges_pos decides which channel this hint's dim lands in,
        # not hint.is_reduction -- keep that single decision so the ranges
        # this exempts from division are exactly the ones planning tiled.
        pos, is_reduction = _hint_ranges_pos(op, hint, out_coords)
        if pos is not None and not is_reduction:
            result[pos] = rng
    return result


def _loop_var_hinted_reduction_ranges(op: ComputedBuffer) -> dict[int, Expr]:
    """Return {pos in op.data.reduction_ranges: loop_var_range} for dim_hints.

    Reduction-dim counterpart of _loop_var_hinted_ranges -- see its
    docstring for why a loop_var-hinted reduction dim must never be divided
    by the level's loop count either.
    """
    hints = getattr(op, "dim_hints", None) or ()
    if not hints or not isinstance(op.data, Reduction):
        return {}
    var_ranges = loop_var_ranges_from_dim_hints(op)
    if not var_ranges:
        return {}
    out_coords = op_out_coords(op)
    result: dict[int, Expr] = {}
    for hint in hints:
        if hint.loop_var is None:
            continue
        rng = var_ranges.get(hint.loop_var)
        if rng is None:
            continue
        # Same single-decision rule as _loop_var_hinted_ranges: the channel
        # comes from _hint_ranges_pos, not from hint.is_reduction.
        pos, is_reduction = _hint_ranges_pos(op, hint, out_coords)
        if pos is not None and is_reduction:
            result[pos] = rng
    return result


def _capture_logical_iteration_symbols(
    op: ComputedBuffer,
) -> tuple[_LogicalIterationSymbol, ...]:
    """Capture active symbols by raw output/reduction position.

    This identity is valid only for callers that preserve dimension order.
    Unit dimensions have no loop symbol and are deliberately omitted.
    """

    data = op.data
    if not isinstance(data, (Pointwise, Reduction)):
        raise Unsupported(
            f"coarse_tile: cannot capture iteration symbols for "
            f"{op.get_name()!r} with data type {type(data).__name__}"
        )

    logical_extents: list[tuple[tuple[str, int], Expr]] = [
        (("output", idx), extent) for idx, extent in enumerate(data.ranges)
    ]
    if isinstance(data, Reduction):
        logical_extents.extend(
            (("reduction", idx), extent)
            for idx, extent in enumerate(data.reduction_ranges)
        )

    active = [
        (logical_dim, extent)
        for logical_dim, extent in logical_extents
        if sympy.sympify(extent) != 1
    ]
    rw = op_read_writes(op)
    # range_vars retains reduction variables even when the inner function does
    # not use them in a memory index. iteration_space_from_op cannot recover
    # those symbols from dependencies (for example, a sum of a broadcast value).
    traced_symbols = tuple(
        symbol for symbol in rw.range_vars if isinstance(symbol, sympy.Symbol)
    )
    symbols = (
        traced_symbols
        if len(traced_symbols) == len(active)
        else tuple(iteration_space_from_op(op))
    )
    if len(active) != len(symbols):
        raise Unsupported(
            f"coarse_tile: cannot match logical dimensions to iteration symbols "
            f"for {op.get_name()!r}: logical_dimensions={active}, "
            f"iteration_symbols={symbols}"
        )

    return tuple(
        _LogicalIterationSymbol(logical_dim, extent, symbol)
        for (logical_dim, extent), symbol in zip(active, symbols)
    )


def _order_preserving_symbol_remap(
    op: ComputedBuffer,
    before: tuple[_LogicalIterationSymbol, ...],
    after: tuple[_LogicalIterationSymbol, ...],
) -> _IterationSymbolRemap:
    """Return the surviving old-to-new symbols for an order-preserving rewrite."""

    before_by_dim = {entry.logical_dim: entry for entry in before}
    after_by_dim = {entry.logical_dim: entry for entry in after}
    after_dims = tuple(entry.logical_dim for entry in after)
    surviving_dims = tuple(
        entry.logical_dim for entry in before if entry.logical_dim in after_by_dim
    )

    pairs = tuple(
        (before_by_dim[logical_dim].symbol, after_by_dim[logical_dim].symbol)
        for logical_dim in surviving_dims
    )
    monotone = (
        surviving_dims == after_dims
        and len({old for old, _ in pairs}) == len(pairs)
        and len({new for _, new in pairs}) == len(pairs)
    )
    if not monotone:
        raise Unsupported(
            f"coarse_tile: order-preserving dimension mapping failed for "
            f"{op.get_name()!r}: old_dimensions={before}, "
            f"new_dimensions={after}, attempted_mapping={pairs}"
        )

    return _IterationSymbolRemap(
        before_symbols=tuple(entry.symbol for entry in before), pairs=pairs
    )


def _fused_iteration_symbol_remap(
    op: ComputedBuffer,
    before_symbols: tuple[sympy.Symbol, ...],
    *,
    max_trailing_removals: int,
) -> _IterationSymbolRemap:
    """Preserve names when fused-loop symbols demonstrably retain their identity.

    ``_capture_logical_iteration_symbols`` cannot describe a many-logical-dims to
    one-loop-symbol fusion.  For that case, accept only the narrower invariant we
    can still prove: the post-rewrite symbols are an unchanged prefix of the old
    symbols.  Reduction variables are appended to the iteration space, so a
    reduction-range rewrite may additionally remove a known number of trailing
    symbols.  Output-range rewrites pass zero and therefore cannot silently
    reinterpret a renumbered surviving dimension.
    """

    after_symbols = tuple(iteration_space_from_op(op))
    removed = len(before_symbols) - len(after_symbols)
    if (
        removed < 0
        or removed > max_trailing_removals
        or before_symbols[: len(after_symbols)] != after_symbols
    ):
        raise Unsupported(
            f"coarse_tile: cannot safely preserve fused work-division symbols "
            f"for {op.get_name()!r}: before={tuple(map(str, before_symbols))}, "
            f"after={tuple(map(str, after_symbols))}"
        )

    return _IterationSymbolRemap(
        before_symbols=before_symbols,
        pairs=tuple((symbol, symbol) for symbol in after_symbols),
    )


def _divide_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> _DivideRangesResult:
    """Divide the specified iteration ranges of op by loop_count.

    For a ``Pointwise`` the full ranges are op.data.ranges.
    For a ``Reduction`` the non-reduction (outer) ranges are op.data.ranges;
    op.data.reduction_ranges are left untouched.

    ``tiled_dims`` is a list of positional indices into ``data.ranges``.
    All indices must be valid; an out-of-bounds index is a caller bug.

    A dim carrying a WhileLoop-splice loop_var hint is exempt from the
    divide -- see _loop_var_hinted_ranges's docstring: op.data.ranges[i] is
    already the per-iteration extent for such a dim, so it is left
    unchanged (and its layout.size entry, further below, likewise stays at
    its current value rather than shrinking).

    Also updates ``op.layout.size``, ``op.layout.stride``, and
    ``op.layout.device_layout`` so the layout describes the smaller per-tile
    buffer, not the full tensor.  Contiguous host strides are recomputed from
    the new size; the ``SpyreTensorLayout`` is rebuilt from the new host size
    and strides, preserving the within-stick dimension from the original layout.
    """
    data = op.data
    if not isinstance(data, (Pointwise, Reduction)):
        return _DivideRangesResult(None, None)

    # Keep this a true no-op when this level does not tile an output dim.
    # In particular, a reduction-only level can be processed after one of
    # its producers has already been retiled.  Invalidating the reduction's
    # cached read/write dependencies here would then rebuild them against the
    # producer's per-tile (possibly size-one) layout and lose the reduction
    # symbol before _divide_reduction_ranges has a chance to remap it.
    if not tiled_dims:
        return _DivideRangesResult(None, None)

    ranges = list(data.ranges)
    if not ranges:
        return _DivideRangesResult(None, None)

    hinted_ranges = _loop_var_hinted_ranges(op)

    before_symbols = None
    fused_before_symbols = None
    if tiled_dims and hasattr(op, "work_div_loop_info"):
        try:
            before_symbols = _capture_logical_iteration_symbols(op)
        except Unsupported:
            fused_before_symbols = tuple(iteration_space_from_op(op))

    for i in tiled_dims:
        assert 0 <= i < len(ranges), (
            f"coarse_tile: op {op.get_name()!r} tiled dim {i} out of bounds "
            f"(ranges has {len(ranges)} entries)"
        )
        if i in hinted_ranges:
            continue
        r = ranges[i]
        if isinstance(r, (int, sympy.Integer)) and isinstance(
            loop_count, (int, sympy.Integer)
        ):
            if int(r) % int(loop_count) != 0:
                raise Unsupported(
                    f"coarse_tile: op {op.get_name()!r} loop var d{i} range {r} "
                    f"is not divisible by loop_count {loop_count}.  All tiled "
                    f"dimensions must be evenly divisible by the loop trip count."
                )
            ranges[i] = sympy.Integer(int(r) // int(loop_count))
        else:
            ranges[i] = sympy.sympify(r) / sympy.sympify(loop_count)

    # Loops is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "ranges", ranges)

    # Invalidate Loops-level caches that read ranges.
    _clear_cache(data, _LOOPS_FREE_SYMS_KEY)
    _clear_cache(data, _LOOPS_INNER_FN_STR_KEY)
    _clear_cache(data, _LOOPS_INNER_FN_OPCOUNT_KEY)
    if isinstance(data, Reduction):
        _clear_cache(data, _REDUCTION_FREE_SYMS_KEY)

    # Invalidate ComputedBuffer-level caches derived from data.ranges.
    _clear_cache(op, _COMPUTED_BUF_SIZES_KEY)
    _clear_cache(op, _COMPUTED_BUF_FREE_SYMS_KEY)
    # ranges just changed unconditionally above, so any memoized
    # get_read_writes() result (pass_utils.op_read_writes) is stale
    # regardless of which capture path (if any) runs below -- invalidate
    # unconditionally rather than only inside the symbol-remap branch.
    # That disagreement is unreachable when coarse tiling runs
    # pre-stickification (nothing has populated the memo yet), but the
    # solver-driven path applies tilings *during* scratchpad planning, after
    # the first solve has memoized every op -- where it surfaced as
    # ``coarse_tile_local_dim_split_domains``'s extent assertion.
    invalidate_op_read_writes(op)

    symbol_remap = None
    if before_symbols is not None or fused_before_symbols is not None:
        if before_symbols is not None:
            symbol_remap = _order_preserving_symbol_remap(
                op, before_symbols, _capture_logical_iteration_symbols(op)
            )
        else:
            assert fused_before_symbols is not None
            symbol_remap = _fused_iteration_symbol_remap(
                op, fused_before_symbols, max_trailing_removals=0
            )

    # Sync layout.size, layout.stride, and layout.device_layout with the new ranges.
    layout = getattr(op, "layout", None)
    if not (isinstance(layout, FixedLayout) and len(layout.size) == len(ranges)):
        return _DivideRangesResult(None, symbol_remap)

    old_stride = tuple(layout.stride)
    old_size = tuple(layout.size)
    new_size = list(layout.size)
    for i in tiled_dims:
        new_size[i] = ranges[i]

    # Recompute strides for the smaller buffer preserving the order of dimensions
    layout.stride = compute_tile_stride(layout.size, old_stride, new_size)

    layout.size = new_size

    # Invalidate Layout- and ComputedBuffer-level caches that read size/stride.
    _clear_cache(layout, _LAYOUT_FREE_SYMS_KEY)
    _clear_cache(op, _COMPUTED_BUF_FREE_SYMS_KEY)
    retiled_info = (
        _RetiledBufferInfo(old_stride, tuple(layout.stride), old_size, tuple(new_size))
        if tiled_dims and old_stride != tuple(layout.stride)
        else None
    )

    # Rebuild SpyreTensorLayout for the new host size using device-native
    # reconstruction: transform the original device layout directly without
    # guessing a dim_order.
    if not isinstance(layout, FixedTiledLayout):
        return _DivideRangesResult(retiled_info, symbol_remap)
    # Capture old/new sizes as ints here, after the FixedTiledLayout guard,
    # so symbolic-size FixedLayout tests above are not affected.
    # layout.size is already the new (divided) size; reconstruct the old size
    # by multiplying tiled dims back up: old[i] = new[i] * loop_count.
    old_host_size = [int(s) for s in layout.size]
    for i in tiled_dims:
        if i in hinted_ranges:
            continue
        old_host_size[i] = int(new_size[i] * loop_count)
    new_size_ints = [int(s) for s in new_size]
    # Recover the authoritative stick host dim from coordinate identity so
    # _resize_device_layout does not have to infer it by size (ambiguous for
    # transposed same-size dims — issue #3116). Tiling-invariant, so safe here.
    stick_hd = _stick_host_dim(op, layout.device_layout)
    # The real strides, not the contiguous ones of the two sizes: this op may
    # write its output in a permuted order, and its stride_map follows that.
    layout.device_layout = _resize_device_layout(
        layout.device_layout,
        old_host_size,
        new_size_ints,
        stick_host_dim=stick_hd,
        old_host_stride=[int(s) for s in old_stride],
        new_host_stride=[int(s) for s in layout.stride],
    )
    return _DivideRangesResult(retiled_info, symbol_remap)


class _PointReadRebaseHandler(WrapperHandler):
    """Rebase only the exact point-read dependencies selected by planning."""

    def __init__(self, inner, rebases: dict[tuple[str, Expr], Expr]):
        super().__init__(inner)
        self._rebases = rebases

    def load(self, name, index):
        canonical_index = sympy.sympify(index)
        index = self._rebases.get((name, canonical_index), index)
        return super().load(name, index)


def _rebase_point_splice_reads(operations: list[Operation]) -> None:
    """Pin a direct point read's index to its iteration-0 base.

    Read-side counterpart of _rebase_splice_write_offset, for the one read
    shape that is neither staged into a copy buffer nor resolvable to a
    tiled dim: a POINT read (no iteration var) whose address moves with the
    spliced loop, e.g. paged attention's in-body page index, dep.index ==
    32*u0. _full_buffer_read_deps deliberately leaves such a read direct,
    and _point_splice_advance_for_dep records its per-trip step in
    squeezed_advance_per_read, from which SpyreKernel._general_tile_advance
    emits the device_tile_advance_expr. Leaving the u0 term in the index too
    would apply the step twice, and would leak a raw unbacked symbol into
    device_coordinates, which op_spec_validation's _check_symbol_consistency
    rejects since it is not an iteration_space key.

    Runs after the copy/reduction/copy-out passes so it sees each op's final
    body, and rebases a read only when squeezed_advance_per_read actually
    carries its advance: any other read keeps its index as planned, so a
    shape this does not cover still fails loudly downstream instead of
    silently losing its per-trip step.
    """
    for op in list(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        if not isinstance(op.data, (Pointwise, Reduction)):
            continue
        loop_vars = _splice_loop_vars(op)
        if not loop_vars:
            continue
        loop_info = getattr(op, "loop_info", None)
        advances = getattr(loop_info, "squeezed_advance_per_read", [])
        if not advances:
            continue
        rebases: dict[tuple[str, Expr], Expr] = {}
        for idx, dep in enumerate(op.get_read_writes().reads):
            if not (
                isinstance(dep, MemoryDep)
                and not dep.var_names
                and (dep.index.free_symbols & loop_vars)
                and idx < len(advances)
                and any(advances[idx])
            ):
                continue
            original_index = sympy.sympify(dep.index)
            base_index, _ = _affine_point_read_index(original_index, loop_vars)
            key = (dep.name, original_index)
            previous = rebases.setdefault(key, base_index)
            if previous != base_index:
                raise Unsupported(
                    f"point-read dependency {dep.name}[{original_index}] has "
                    "conflicting rebased indices"
                )
        if not rebases:
            continue

        def new_inner_fn(*args, _orig=op.data.inner_fn, _rebases=rebases):
            with V.set_ops_handler(_PointReadRebaseHandler(V.ops, _rebases)):
                return _orig(*args)

        object.__setattr__(op.data, "inner_fn", new_inner_fn)
        new_op = replace_computed_buffer_body(
            op,
            op.data,
            operations,
            pass_name="coarse_tile",
            reason="rebase point splice read to iteration 0",
        )
        V.graph.name_to_buffer[new_op.get_name()] = new_op
        logger.debug(
            "coarse_tile: rebased point splice reads %s for %s",
            sorted(f"{name}[{index}]" for name, index in rebases),
            new_op.get_name(),
        )


class _NameSwapHandler(WrapperHandler):
    """Redirect ops.load(name, index) calls for names present in name_map.

    See NameSwapHandler in pass_utils.py — same pattern (AGENTS.md
    "Compiler Pass Conventions": wrap inner_fn via a WrapperHandler, never
    reconstruct it from index expressions). Duplicated locally rather than
    imported to avoid a coarse_tile <-> insert_restickify import-order
    dependency; the two run at different, non-adjacent pipeline stages.

    Unlike insert_restickify's version, entries here also carry the
    full-buffer and tile-local strides for the swapped name: the copy buffer being swapped in is physically
    smaller than the full buffer it replaces, so tiled_op's original index
    (affine in its own loop vars, using full_buf's stride coefficients) no
    longer resolves to a valid offset into it.

    The incoming `index` at call time is tiled_op's own inner_fn tracing
    through this exact load, so it is affine in whatever loop variables that
    particular trace happens to use -- inner_fn may be retraced multiple
    times (e.g. by scheduler fusion checks) with a *different* dummy
    variable each time (d0, i0, q0, ... have all been observed for the same
    load site), so a single precomputed replacement expression captured at
    name-swap time would be wrong on every trace that doesn't
    happen to reuse those exact symbols. Instead, `index` is rescaled at
    call time: each additive term's coefficient is matched (by value) against
    full_strides to find its dimension, then replaced by that dimension's
    tile_strides coefficient -- this works for whatever free symbols this
    particular trace used, without needing to know them in advance.
    """

    def __init__(
        self,
        inner,
        name_map: dict[str, tuple[str, list[Expr], list[Expr]]],
    ):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        if name in self._name_map:
            new_name, full_strides, tile_strides = self._name_map[name]
            new_index = _rescale_index(
                index, full_strides, tile_strides, strip_constant=True
            )
            return super().load(new_name, new_index)
        return super().load(name, index)


class _LoopVarRebaseHandler(WrapperHandler):
    """Pin one advancing source load to its iteration-zero base address.

    A spliced ``for_each_tile`` body carries its induction variable directly
    in graph-input load indexes.  A staged copy removes that term and
    represents it through ``device_tile_advance_expr`` instead.  A direct
    read restored after planning must use the same single representation;
    otherwise the offset is applied twice and the raw unbacked symbol leaks
    into the OpSpec coordinates.
    """

    def __init__(self, inner, source_name: str, loop_var_zeros: dict[Expr, Expr]):
        super().__init__(inner)
        self._source_name = source_name
        self._loop_var_zeros = loop_var_zeros

    def load(self, name, index):
        if name == self._source_name:
            index = sympy_subs(index, self._loop_var_zeros)
        return super().load(name, index)


def _rescale_index(
    index: Expr,
    full_strides: list[Expr],
    tile_strides: list[Expr],
    strip_constant: bool = False,
    reject_ambiguous: bool = False,
) -> Expr:
    """Rescale an affine index's per-dimension coefficients.

    `index` is affine in some set of loop variables, with one additive term
    per dimension whose coefficient equals the matching entry in
    `full_strides` (plus, possibly, a constant offset term). Returns the
    same linear combination of the same variables with each dimension's
    coefficient replaced by the matching entry in `tile_strides`. Matching
    is by coefficient value rather than by variable identity because the
    variables `index` is expressed in are not known in advance -- see
    _NameSwapHandler.  When ``reject_ambiguous`` is true, a term that could
    name multiple equal-stride dimensions is accepted only if every match
    maps to the same target stride.

    Each additive term is matched against a candidate `full_stride` by
    dividing the term by it and checking the quotient is free of the
    stride's own symbols (see `_divides_evenly` below) -- NOT via
    `index.as_coefficients_dict()`, which only isolates a term's
    "coefficient" correctly when that coefficient is a plain number. When
    `full_strides` contains a genuinely symbolic stride (e.g. a level/tile
    symbol) and the matching term is `loop_var * symbolic_stride`, sympy
    normalizes that whole product into a single atom with numeric
    coefficient 1 -- `as_coefficients_dict()` would report the *entire
    product* as the "term" and never find a `full_strides` entry equal to
    1, silently failing to match a case this function is specifically
    meant to support.

    Two further subtleties in the matching, both because dimensions are
    identified by their stride *value* rather than their position:

    - An extent-1 dimension's stride can coincide with a larger dimension's
      stride (e.g. a size-[1, N] shape's dim-0 stride equals dim-1's full
      extent, same as a size-[M, N] shape's dim-0 stride). Matching
      smallest-remaining-first would let a degenerate extent-1 stride steal
      a match that belongs to a real, larger dimension. Matching
      largest-first instead defers ambiguity among small/degenerate strides
      as long as possible, since a larger stride can only coincide with
      another dimension of at least that size.
    - Two full_strides can be symbolically equal but differently-formed
      expressions (e.g. ``2*(s0 + 1)`` vs ``2*s0 + 2``) -- `_divides_evenly`
      falls back to a simplified quotient/difference check rather than
      relying on structural equality alone.
    """
    # Callers mix sympy coefficients with plain Python ints (the `layout.stride`
    # entries of a fully static buffer).
    # Everything below is sympy, so normalize once here rather than at each
    # append site -- an int reaching _sort_key raised AttributeError.
    full_strides = [sympy.sympify(stride) for stride in full_strides]
    tile_strides = [sympy.sympify(stride) for stride in tile_strides]

    def _divides_evenly(term: Expr, full_stride: Expr) -> tuple[bool, Expr]:
        """Return (matched, loop_var_part) if `full_stride` divides `term`.

        `full_stride` divides `term` cleanly when dividing it out of `term`
        leaves *exactly* the loop-variable part behind: no leftover free
        symbol from `full_stride` (it must fully cancel, not partially --
        e.g. dividing `c0*s0` by `s0` alone, not by some unrelated factor of
        it), and no leftover numeric scale factor (e.g. dividing `c0*128` by
        `4` leaves `32*c0`, i.e. still scaled by 32 -- not a clean divide,
        even though the quotient happens to be symbol-free). Checked both
        structurally and, if that's inconclusive, after simplifying the
        quotient (mirrors the structural-vs-simplified fallback this
        function has always used for coefficient matching).

        A zero `full_stride` (a broadcast/absent dimension) never matches
        any term: dividing by it produces sympy's `zoo` (complex infinity)
        rather than raising, and `zoo * loop_var` deceptively passes the
        coeff==1/no-leftover-symbol checks above (its free_symbols are just
        the loop_var's, same as a real match) -- silently producing a `nan`
        rescaled index several steps later instead of a loud failure. A term
        can only "divide evenly" into a stride that actually advances a
        dimension.
        """
        if full_stride == 0:
            return False, sympy.Integer(0)

        stride_syms = full_stride.free_symbols

        def _is_clean(quotient: Expr) -> bool:
            coeff, _ = quotient.as_coeff_Mul()
            return coeff == 1 and not (quotient.free_symbols & stride_syms)

        quotient = term / full_stride
        if _is_clean(quotient):
            return True, quotient
        simplified = sympy.simplify(quotient)
        if _is_clean(simplified):
            return True, simplified
        return False, sympy.Integer(0)

    def _sort_key(pair: tuple[Expr, Expr]) -> tuple[int, Expr]:
        # Sort largest-first without calling `<` directly on sympy Exprs --
        # that raises TypeError for expressions with free symbols, which
        # full_strides commonly contains (level/tile symbols). Concrete
        # integers compare among themselves by value; every symbolic stride
        # sorts ahead of every concrete one (a symbolic stride is a
        # multiple of some concrete extent, so it is at least as large),
        # and symbolic-vs-symbolic keeps original relative order (stable
        # sort) rather than guessing a magnitude.
        full_stride = pair[0]
        is_concrete = full_stride.is_number
        return (
            0 if is_concrete else 1,
            full_stride if is_concrete else sympy.Integer(0),
        )

    remaining = sorted(zip(full_strides, tile_strides), key=_sort_key, reverse=True)
    new_index: Expr = sympy.Integer(0)
    for term in sympy.Add.make_args(index):
        if term.is_number:
            if not strip_constant:
                new_index += term
            continue
        if not reject_ambiguous:
            for i, (full_stride, tile_stride) in enumerate(remaining):
                matched, loop_var_part = _divides_evenly(term, full_stride)
                if matched:
                    new_index += tile_stride * loop_var_part
                    del remaining[i]
                    break
            else:
                raise RuntimeError(
                    f"_rescale_index: no matching full_stride for term {term} "
                    f"in index {index}; full_strides={full_strides}"
                )
            continue

        matches = [
            (i, tile_stride, loop_var_part)
            for i, (full_stride, tile_stride) in enumerate(remaining)
            if (match := _divides_evenly(term, full_stride))[0]
            for loop_var_part in (match[1],)
        ]
        if not matches:
            raise RuntimeError(
                f"_rescale_index: no matching full_stride for term {term} "
                f"in index {index}; full_strides={full_strides}"
            )
        if reject_ambiguous and any(
            sympy.simplify(tile_stride - matches[0][1]) != 0
            for _i, tile_stride, _loop_var_part in matches[1:]
        ):
            raise RuntimeError(
                f"_rescale_index: ambiguous full_stride for term {term} "
                f"in index {index}; full_strides={full_strides}"
            )
        i, tile_stride, loop_var_part = matches[0]
        new_index += tile_stride * loop_var_part
        del remaining[i]
    return new_index


def _squeezed_retile_dims(
    info: _RetiledBufferInfo, consumer: ComputedBuffer
) -> list[int]:
    """Raw dims squeezed out of the old (tile-local) buffer but real in new.

    A dim with ``old_size[d] == 1`` has no ``d{i}`` symbol in an incoming
    load index at all -- Inductor's ``SqueezeView.squeezer`` drops unit-size
    dims unconditionally when the *reader's own* index for that dim was
    derived against this buffer's (by-then tile-local, size-1) layout, so
    ``compute_tile_index``/``_retile_load_index`` has no atom to rescale for
    that dim: rescaling can only touch coefficients already present in the
    incoming index (see ``_retile_load_index``'s docstring).

    A nonzero stride does *not* prove that a dimension became real: ordinary
    contiguous tensors retain nonzero strides on size-one dimensions.  Add a
    term only for an actual extent transition from one in the tile-local
    buffer to non-one in the full buffer.  In particular, decode attention's
    ``[B,H,Lq,D]`` output has ``Lq == 1`` in both buffers; treating its Lq
    stride as evidence of growth aliases the flattened projection's output-N
    loop onto Lq and turns a 4K read into a bogus 16M read.

    Re-minting uses a raw producer dimension as a positional consumer output
    dimension.  That mapping is valid only when the complete shapes agree.
    Rank-changing or shape-changing views need semantic view metadata to
    recover a missing coordinate; guessing positionally would silently read
    the wrong dimension, so reject such a true-growth case explicitly.
    """
    grown_dims = [
        d
        for d in range(len(info.old_size))
        if info.old_size[d] == 1
        and info.new_size[d] != 1
        and info.new_stride[d] != sympy.S.Zero
    ]
    if not grown_dims:
        return []

    consumer_ranges = tuple(consumer.data.ranges)
    if len(consumer_ranges) < len(info.new_size):
        raise Unsupported(
            "coarse_tile: cannot restore dimensions squeezed from a retiled "
            "producer through a rank-changing consumer view; "
            f"producer old_size={info.old_size}, new_size={info.new_size}, "
            f"consumer ranges={consumer_ranges}, grown_dims={grown_dims}"
        )

    # A unit consumer axis selects coordinate zero and needs no symbol.  For
    # every axis that does need a symbol, require the complete output shape to
    # match the producer shape before treating raw positions as identities.
    result = [d for d in grown_dims if int(consumer_ranges[d]) != 1]
    if result and any(
        sympy.simplify(actual - expected) != 0
        for actual, expected in zip(consumer_ranges, info.new_size)
    ):
        raise Unsupported(
            "coarse_tile: cannot restore dimensions squeezed from a retiled "
            "producer through a shape-changing consumer view; "
            f"producer old_size={info.old_size}, new_size={info.new_size}, "
            f"consumer ranges={consumer_ranges}, grown_dims={grown_dims}"
        )
    return result


def _is_dense_full_buffer_view(
    index: Expr, info: _RetiledBufferInfo, consumer: ComputedBuffer
) -> bool:
    """Whether ``index`` already densely addresses the complete new buffer.

    A rank-changing view can flatten a dimension that was unit-sized in the
    tile but is real in the full buffer.  For example, GQA flattens
    ``[Hkv=8, group=4]`` into ``[heads=32]``.  Its flattened coefficient is
    already the full buffer's group stride, so decomposing that coefficient
    against the tile-local Hkv stride corrupts it.  We can recognize the safe
    case without view metadata when both layouts are dense and the affine
    index is a complete, bijective permutation/reshape of the new buffer.

    A consumer tiled in a loop group of its own iterates only a tile of its
    ranges, yet a view with its own layout (a ``ReinterpretView``) keeps
    tracing its read at full scale; its index is matched against the ranges
    before division too.

    A missed recognition is safe but conservative: ``False`` keeps the
    caller on the general retile/``Unsupported`` path rather than accepting
    an unproven full-buffer mapping.
    """

    def _equal(lhs: Expr, rhs: Expr) -> bool:
        return sympy.simplify(lhs - rhs) == 0

    target_dims = [
        (stride, size)
        for size, stride in zip(info.new_size, info.new_stride, strict=True)
        if size != 1
    ]
    if any(stride == sympy.S.Zero for stride, _ in target_dims):
        return False
    target_dims.sort(key=lambda pair: pair[0])
    running = sympy.Integer(1)
    for stride, size in target_dims:
        if not _equal(stride, running):
            return False
        running *= size
    target_numel = running

    try:
        atoms, offset = decompose_index_for_tiling(
            index, dict.fromkeys(index.free_symbols, 1)
        )
    except Unsupported:
        return False
    if offset != 0 or len(atoms) != len(index.free_symbols):
        return False

    symbol_numbers: dict[sympy.Symbol, int] = {}
    for _coefficient, symbol in atoms:
        name = symbol.name
        split = len(name)
        while split > 0 and name[split - 1].isdigit():
            split -= 1
        if split == len(name):
            return False
        symbol_numbers[symbol] = int(name[split:])

    ranges = tuple(consumer.data.ranges)
    reduction_ranges = tuple(getattr(consumer.data, "reduction_ranges", None) or ())
    extent_sets = [(ranges, reduction_ranges)]
    if getattr(consumer, "loop_info", None) is not None:
        extent_sets.append(_undivided_ranges(consumer))

    # Symbols number the live (possibly divided) ranges; extents come from
    # either set.  Some retraces preserve raw dimension numbers (_i1/_i2/_i3
    # when raw dim 0 is unit), while extract_read_writes renumbers them
    # densely (d0/d1/d2).  Some use an independent r0/r1/... namespace for
    # reduction indices rather than continuing the d-numbering after outputs.
    n_out = len(ranges)
    all_ranges = [*ranges, *reduction_ranges]
    positions = [
        list(range(n_out)),
        [i for i, size in enumerate(ranges) if size != 1],
        list(range(len(all_ranges))),
        [i for i, size in enumerate(all_ranges) if size != 1],
    ]
    reduction_positions = [
        [n_out + i for i in range(len(reduction_ranges))],
        [n_out + i for i, size in enumerate(reduction_ranges) if size != 1],
    ]
    if all(symbol.name.lstrip("_").startswith("r") for symbol in symbol_numbers):
        positions += reduction_positions

    for extent_ranges, extent_reduction_ranges in extent_sets:
        extents = [*extent_ranges, *extent_reduction_ranges]
        for pos in positions:
            if any(number >= len(pos) for number in symbol_numbers.values()):
                continue
            indexed_dims = sorted(
                (
                    (coefficient, extents[pos[symbol_numbers[symbol]]])
                    for coefficient, symbol in atoms
                ),
                key=lambda pair: pair[0],
            )
            running = sympy.Integer(1)
            for coefficient, extent in indexed_dims:
                if not _equal(coefficient, running):
                    break
                running *= extent
            else:
                if _equal(running, target_numel):
                    return True
    return False


def _undivided_ranges(
    consumer: ComputedBuffer,
) -> tuple[tuple[Expr, ...], tuple[Expr, ...]]:
    """``consumer``'s ranges and reduction ranges before ``_divide_ranges``.

    Exact because ``_divide_ranges`` refuses a trip count that does not divide
    its range; a non-exact tiling would have to record the original extent.
    """
    info = consumer.loop_info  # type: ignore[attr-defined]
    ranges = list(consumer.data.ranges)
    reduction_ranges = list(getattr(consumer.data, "reduction_ranges", None) or ())
    hinted = _loop_var_hinted_ranges(consumer)
    hinted_reduction = _loop_var_hinted_reduction_ranges(consumer)
    for count, dims, reduction_dims in zip(
        info.loop_count, info.loop_tiled_dims, info.loop_tiled_reduction_dims
    ):
        for d in dims:
            if d not in hinted:
                ranges[d] = ranges[d] * count
        for d in reduction_dims:
            if d not in hinted_reduction:
                reduction_ranges[d] = reduction_ranges[d] * count
    return tuple(ranges), tuple(reduction_ranges)


def _index_var_prefix(free_symbols: "OrderedSet[Expr] | set[Expr]") -> str:
    """Infer the live loop-variable naming prefix from an index's own symbols.

    ``index_vars_squeeze``/``index_vars_no_squeeze`` number loop variables
    densely as ``f"{prefix}{i}"`` -- but the prefix varies across the
    different retracing passes that reuse this same redirect machinery
    (``d0``, ``q0``, ``_i0``, ``i0``, ... have all been observed for the
    exact same logical dimension at different call sites/passes). Minting a
    hardcoded ``d{i}`` symbol is only correct when the live trace happens to
    use that exact prefix; otherwise the minted symbol is foreign to this
    trace's variable space and later stages that concretize "unknown"
    symbols (e.g. ``concretize_index`` treating it as a size symbol) will
    silently replace it with a constant, dropping the dimension again after
    _retile_load_index believed it had restored it. Detect the real prefix
    from any sibling symbol already present in the index instead of
    assuming one.
    """
    for sym in sorted(free_symbols, key=str):
        name = sym.name
        i = len(name)
        while i > 0 and name[i - 1].isdigit():
            i -= 1
        if i < len(name):
            return name[:i]
    return "d"


def _consumer_own_dim_symbol(
    consumer: ComputedBuffer, dim: int, prefix: str = "d"
) -> Expr:
    """The consumer's own loop-variable symbol for raw output dim ``dim``.

    Mirrors ``SpyreKernel._host_dim_to_index_symbol``'s squeeze arithmetic:
    Inductor's ``index_vars_squeeze`` numbers loop variables densely (as
    ``f"{prefix}{i}"``) over ``consumer.data.ranges``'s non-unit dims only,
    so ``{prefix}{dim}`` is the correct symbol only when no unit dim
    precedes it. ``consumer`` is the outside consumer being redirected
    (e.g. ``div``), not the buffer it reads -- its own output ranges are
    never squeezed by the redirect (only the *read* buffer's tile-local
    layout was), so this always yields a real, live loop symbol already
    used elsewhere in the consumer's index (e.g. by an unretiled sibling
    read), PROVIDED ``prefix`` matches the naming convention live in that
    index -- see ``_index_var_prefix``, which callers should use to derive
    it rather than assuming the default.
    """
    it_idx = 0
    mapped = dim
    for host_idx, r in enumerate(consumer.data.ranges):
        if int(r) != 1:
            if host_idx == dim:
                mapped = it_idx
            it_idx += 1
    return sympy_index_symbol(f"{prefix}{mapped}")


def _index_already_at_new_scale(
    index: Expr, loop_syms: set, info: "_RetiledBufferInfo"
) -> bool:
    """Return True when index's atom coefficients already match new_stride.

    A consumer in the same tiling group as a retiled buffer can be resynced
    (by name) to a *replacement* ComputedBuffer object spliced in by Pass
    2/3 after ``_apply_plan`` already ran -- see coarse_tile_post_stickify's
    by-name resync comment above its ``_patch_retiled_load_indexes`` call.
    That replacement's inner_fn may have been retraced against the
    producer's already-mutated (new_stride) layout, in which case its load
    index is already correct and must not be decomposed against old_stride
    again -- doing so silently produces a wrong, merely plausible-looking
    index (two dims' coefficients can collide under old_stride's pairing
    even though the input was never stale -- see
    test_copy_running_max_4d_H4_Lq4).

    Detected by comparing the *set* of each atom's raw coefficient (before
    any tile-offset decomposition) against the set of old_stride vs.
    new_stride values for this buffer's real (non-irregular) dimensions.
    Each atom's coefficient is exactly one dimension's stride value in
    whatever scale the trace used -- a single loop variable that
    legitimately spans multiple dims (the "diagonal" case handled by
    compute_tile_offset's divmod chain, e.g. test_compute_tile_index_2d_diagonal)
    produces one atom whose coefficient is a *combined* value that matches
    neither set exactly, so this check only ever fires on the genuine
    already-new-scale case, never on a legitimate diagonal index. A
    coincidental match between the two sets can only happen when
    old_stride == new_stride for the dims involved (tiling never increases
    a dim's stride), which is a no-op either way.
    """
    try:
        atoms, _offset = decompose_index_for_tiling(index, dict.fromkeys(loop_syms, 1))
    except Unsupported:
        return False
    if not atoms:
        return False
    coeffs = {atom[0] for atom in atoms}
    dims = [
        d
        for d, (s, t) in enumerate(zip(info.old_size, info.old_stride))
        if s != 1 and t != 0
    ]
    old_set = {info.old_stride[d] for d in dims}
    # A dimension that a later nested tiling level squeezes to one has a zero
    # final stride.  It cannot contribute an atom to an already-retiled load,
    # so including that zero in ``new_set`` makes the equality test fail and
    # causes the fresh index to be rewritten a second time.  In GQA this
    # misidentifies the surviving Hkv coefficient as the now-squeezed group
    # coefficient and drops Hkv from every downstream read.
    new_set = {info.new_stride[d] for d in dims if info.new_stride[d] != sympy.S.Zero}
    return coeffs == new_set and coeffs != old_set


def _retile_load_index(
    buf_name: str,
    index: Expr,
    info: _RetiledBufferInfo,
    consumer: "ComputedBuffer | None" = None,
    preserve_target_stride_atoms: bool = False,
) -> Expr:
    """Rewrite a load index using compute_tile_index.  Raises Unsupported if
    the index cannot be decomposed (non-affine, or stride not in info.old_stride).

    Used by _RetileLoadIndexHandler and _NameAndIndexSwapHandler during real
    codegen.  Most incoming index coefficients are expressed in
    ``info.old_stride`` and are rewritten to ``info.new_stride``.  An outside
    view can, however, derive some coefficients from the full destination
    layout before its producer is redirected.  In the tile-to-full direction,
    ``preserve_target_stride_atoms`` keeps an atom whose coefficient exactly
    matches an unambiguous regular ``new_stride`` entry and exceeds the source
    tile's maximum physical offset (proving that it cannot be source-local).
    Remaining atoms still use ``compute_tile_index`` so view-combined
    tile-local coefficients retain the general decomposition supported by that
    helper.

    When ``consumer`` is given AND has no loop_info of its own (i.e. it is an
    "outside" consumer with no enclosing coarse-tile loop nest), any dim
    squeezed out of the old (tile-local) buffer but real in the new (full)
    buffer -- see _squeezed_retile_dims -- is added back as an explicit
    ``consumer``-own-symbol * new_stride term, since compute_tile_index
    cannot rescale a coefficient that isn't in the incoming index at all.
    Inside consumers (those with their own loop_info) are never patched this
    way: their incoming index is already derived from a real, enclosing loop
    nest via normal codegen and already carries a correct term for every
    real dimension -- adding another would double-count it. Only meaningful
    for the tile->full ("grow") direction (_NameAndIndexSwapHandler); the
    full->tile direction (_RetileLoadIndexHandler) never needs this --
    shrinking a buffer's own layout for a group-internal consumer doesn't
    lose dims from its index.

    The two handlers run in opposite directions:
    - _RetileLoadIndexHandler: full→tile (old_stride are the full strides,
      new_stride are the smaller tile strides).
    - _NameAndIndexSwapHandler: tile→full (old_stride are the tile strides,
      new_stride are the larger full strides).

    compute_tile_index is valid for both directions.  Its core,
    compute_tile_offset, does divmod(coeff, s) for each paired stride s in
    decreasing order.  Each atom's coefficient IS one of the old strides, so
    each divmod is exact (remainder zero).  Potential ambiguity from duplicate
    stride values is prevented by compute_tile_index's irregular-dimension
    filter: size==1 and stride==0 dimensions are excluded before pairing, and
    those are the only source of duplicate strides in a valid tensor layout.

    compute_tile_index uses var_ranges only as a key-set to distinguish loop
    variables from shape symbols.  We derive it from index.free_symbols so the
    call site never needs to know which prefix convention the calling inner_fn
    uses for its loop variables.
    """

    loop_syms = index.free_symbols
    dense_full_view = (
        consumer is not None
        and preserve_target_stride_atoms
        and _is_dense_full_buffer_view(index, info, consumer)
    )
    if dense_full_view:
        new_index = index
    elif not loop_syms:
        new_index = index
    elif not preserve_target_stride_atoms and _index_already_at_new_scale(
        index, loop_syms, info
    ):
        # This consumer's index was traced *after* the producer's layout was
        # already mutated to new_stride (e.g. built/retraced during Pass
        # 2/3, from a same-name replacement object resynced into group_ops
        # -- see coarse_tile_post_stickify's by-name resync comment), so it is
        # already correct at the target scale. Rewriting it again would
        # decompose already-new-scale coefficients against old_stride and
        # silently produce a wrong (but plausible-looking) index -- see
        # issue found via test_copy_running_max_4d_H4_Lq4.
        new_index = index
    else:
        index_to_retile = index
        preserved_index = sympy.S.Zero
        if preserve_target_stride_atoms:
            source_max_offset = sympy.simplify(
                sum(
                    (old_size - 1) * old_stride
                    for old_size, old_stride in zip(info.old_size, info.old_stride)
                    if old_size != 1 and old_stride != sympy.S.Zero
                )
            )
            regular_source_strides = {
                sympy.simplify(old_stride)
                for old_size, old_stride in zip(info.old_size, info.old_stride)
                if old_size != 1 and old_stride != sympy.S.Zero
            }
            regular_target_strides = {
                sympy.simplify(new_stride)
                for new_size, new_stride in zip(info.new_size, info.new_stride)
                if new_size != 1
                and new_stride != sympy.S.Zero
                and sympy.simplify(new_stride) not in regular_source_strides
            }
            for term in index.as_ordered_terms():
                term_syms = term.free_symbols & loop_syms
                if len(term_syms) != 1:
                    continue
                sym = next(iter(term_syms))
                coeff = sympy.simplify(term.coeff(sym))
                exceeds_source_span = sympy.simplify(
                    coeff - source_max_offset
                ).is_positive
                if coeff in regular_target_strides and exceeds_source_span is True:
                    preserved_index += term
            index_to_retile = sympy.expand(index - preserved_index)

        new_index = compute_tile_index(
            index_to_retile,
            dict.fromkeys(loop_syms, 1),
            info.old_size,
            info.old_stride,
            info.new_stride,
        )
        new_index += preserved_index

    # The squeezed-dim term injection is only meaningful for a consumer with
    # no loop_info of its own -- an "outside" consumer whose incoming index
    # was traced against buf_name's tile-local (squeezed) layout with no
    # enclosing coarse-tile loop nest to supply a term for a dim that layout
    # squeezed away. An "inside" consumer (has loop_info) already derives its
    # index from a real, enclosing loop nest that supplies a correct term for
    # every one of *its own* real dimensions; injecting anyway adds a bogus
    # term for any dim this consumer tiles as an output dim but that the new
    # layout does not vary over (e.g. a fully-reduced dim in an accum_full
    # buffer) -- see test_copy_accum_with_reduction_512x256_A4_B4. Such a
    # consumer still reaches the dense-full-view check above: one in another
    # loop group can read the whole buffer every iteration, its index traced
    # at full scale.
    if (
        consumer is not None
        and not hasattr(consumer, "loop_info")
        and not dense_full_view
    ):
        for d in _squeezed_retile_dims(info, consumer):
            # A consumer read by multiple _patch_consumers redirects (e.g.
            # buf24 reading both a _divide_ranges-mutated buffer and a
            # reduction accumulator) has its inner_fn wrapped more than
            # once. By the time a later wrap sees this index, an earlier
            # wrap may already have contributed a term for this exact dim
            # -- under whatever loop-symbol naming convention was live at
            # that earlier wrap's call site (which varies across retracing
            # passes: d0, i0, q0, ... are all seen for the same logical
            # dim). Detect that by coefficient, not by a hardcoded symbol
            # name: if some free symbol already carries coefficient
            # new_stride[d], this dim's term is already present and must
            # not be added again. Comparing by coefficient value (not
            # dimension identity) is safe only because strides are unique
            # across all non-irregular dims of a valid layout -- see this
            # function's docstring ("Potential ambiguity from duplicate
            # stride values"). If that uniqueness invariant is ever
            # weakened, this check must be revisited too.
            already_present = any(
                new_index.coeff(s) == info.new_stride[d] for s in new_index.free_symbols
            )
            if not already_present:
                prefix = _index_var_prefix(new_index.free_symbols)
                sym = _consumer_own_dim_symbol(consumer, d, prefix)
                new_index += sym * info.new_stride[d]

    logger.debug(
        "coarse_tile: retiled load index for %s: %s -> %s "
        "(old_size=%s old_stride=%s new_size=%s new_stride=%s)",
        buf_name,
        index,
        new_index,
        info.old_size,
        info.old_stride,
        info.new_size,
        info.new_stride,
    )
    return new_index


class _RetileLoadIndexHandler(WrapperHandler):
    """Ops handler that retiles loads from buffers whose host strides changed.

    Used during real codegen — calls _retile_load_index, which raises
    Unsupported if an index cannot be retiled.
    """

    def __init__(self, inner, infos_by_name: dict[str, _RetiledBufferInfo]):
        super().__init__(inner)
        self._infos_by_name = infos_by_name

    def load(self, name, index):
        if name in self._infos_by_name:
            index = _retile_load_index(name, index, self._infos_by_name[name])
        return super().load(name, index)


def _should_patch_retiled_load_indexes(
    op: Operation,
    group_id: tuple[int, ...],
    retiled_names: set[str],
) -> bool:
    """Return True when op is an exact-loop consumer of a retiled buffer."""
    if not isinstance(op, ComputedBuffer):
        return False
    if not isinstance(op.data, (Pointwise, Reduction)):
        return False
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None or loop_info.loop_group_id != group_id:
        return False
    # Fetch op's own read names once (memoized via _op_reads -- safe here
    # because _patch_retiled_load_indexes tests/mutates each op in
    # group_ops at most once) instead of re-tracing inner_fn once per name
    # in retiled_names.
    return not retiled_names.isdisjoint(_op_reads(op))


def _replace_group_op(
    group_ops: list[Operation], old_op: Operation, new_op: Operation
) -> None:
    """Keep the tiling group list in sync after replacing a ComputedBuffer body."""
    old_name = old_op.get_operation_name()
    for idx, group_op in enumerate(group_ops):
        if group_op is old_op or group_op.get_operation_name() == old_name:
            group_ops[idx] = new_op
            return


def _patch_retiled_load_indexes(
    group_id: tuple[int, ...],
    group_ops: list[Operation],
    retiled_infos: dict[str, _RetiledBufferInfo],
    operations: list[Operation],
) -> None:
    """Rewrite stale load indexes for consumers of buffers retiled by coarse tiling."""
    infos_by_name = {
        name: info
        for name, info in retiled_infos.items()
        if info.old_stride != info.new_stride
    }
    if not infos_by_name:
        return

    from ..pass_utils import replace_computed_buffer_body

    # Only ops that were already in the group when _apply_plan ran can hold a
    # stale (pre-divide) coefficient for a retiled buffer.  Ops inserted later
    # by Pass 1/2 (e.g. _insert_copy_op's copy_buf) read the retiled
    # buffer's already-updated layout directly, so rewriting them here would
    # double-apply the stride correction (see issue found while fixing
    # test_hint_restickify_stays_in_group).
    retiled_names = set(infos_by_name)
    for op in list(group_ops):
        if not _should_patch_retiled_load_indexes(op, group_id, retiled_names):
            continue

        orig_inner = op.data.inner_fn

        def new_inner_fn(
            *args,
            _infos=infos_by_name,
            _orig=orig_inner,
        ):
            with V.set_ops_handler(_RetileLoadIndexHandler(V.ops, _infos)):
                return _orig(*args)

        object.__setattr__(op.data, "inner_fn", new_inner_fn)
        invalidate_op_read_writes(op)
        new_op = replace_computed_buffer_body(
            op,
            op.data,
            operations,
            pass_name="coarse_tile",
            reason="rewrite retiled load indexes",
        )
        _replace_group_op(group_ops, op, new_op)
        V.graph.name_to_buffer[new_op.get_name()] = new_op
