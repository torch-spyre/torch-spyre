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

"""Span-overflow tile planning for oversized output-range ops."""

from __future__ import annotations

import itertools
import math

from dataclasses import dataclass

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_OP
from .errors import Unsupported
from .ir import FixedTiledLayout, _resize_device_layout
from .logging_utils import get_inductor_logger
from .pass_utils import (
    _fixed_read_layout,
    concretize_index,
    host_coordinates,
    indirect_info_from_op,
    op_out_coords,
)
from .views import compute_coordinates
from .work_division import MAX_SPAN_BYTES


logger = get_inductor_logger("span_overflow_hint_analysis")


@dataclass(frozen=True)
class ChunkingInfo:
    """Physical span facts that explain why one candidate needs tiling.

    ``per_core_span`` is the byte span we would see if work division gives no
    useful split for the selected physical coordinate.  ``selected_host_dim`` is
    the logical output dimension that can shrink that span through coarse
    tiling, and ``reason`` records whether the pressure came from output writes
    or from a direct input dependency.
    """

    total_bytes: int
    per_core_span: int
    core_split_estimate: int
    selected_device_dim_size: int
    selected_device_span_stride_elems: int
    selected_host_dim: int
    stick_elems: int
    reason: str | None = None


@dataclass(frozen=True)
class InputSpanInfo:
    """Input span facts that can be reduced by output-range tiling.

    This is used for Reduction/BMM ops where the output tensor can be small but
    a direct input read still spans too much memory.  ``controlling_symbol`` is
    the output-loop symbol, such as BMM ``M`` or ``N``; reduction-only symbols
    such as ``K`` are not represented here because this pass cannot tile
    reduction ranges yet.
    """

    chunking_info: ChunkingInfo
    dep_name: str
    controlling_symbol: sympy.Symbol


@dataclass(frozen=True)
class SpanOverflowTileLevel:
    """One coarse-tile loop level to apply to the op's iteration space.

    ``selected_host_dim`` is the output/data-range dimension to tile.  For an
    op with ranges ``[M, N]``, ``selected_host_dim=0`` tiles ``M`` and
    ``selected_host_dim=1`` tiles ``N``.  ``split_count`` is the number of
    equal-sized chunks to create for that dimension, so the dimension size must
    be divisible by it.  ``is_reduction`` is false for the current automatic
    span-overflow planner because it only tiles output ranges; reduction-range
    tiling would require partial-result accumulation.
    """

    selected_host_dim: int
    split_count: int
    is_reduction: bool = False


@dataclass(frozen=True)
class SpanOverflowTilePlan:
    """Complete coarse-tiling request returned by span-overflow analysis.

    ``levels`` is the actual tiling plan consumed by ``coarse_tile``.  It can
    contain one level, for example ``[(host_dim=0, split_count=32)]`` for the
    LM-head vocab dimension, or multiple levels when several output dimensions
    must be tiled together.  The levels are emitted outer-to-inner by host dim.
    Automatic split counts are bounded by ``_MAX_AUTO_TILE_SPLIT_COUNT``; a
    larger value like 769 is only reachable via a manual ``spyre_hint``, not
    through this automatic plan.

    ``chunking_infos`` records the physical span facts that caused the plan:
    total bytes, per-tile/per-core span estimate, selected physical span dim,
    corresponding host dim, stick size, and reason.  There may be more
    ``chunking_infos`` than ``levels`` because one tile level can fix multiple
    output/input span pressures.

    ``reason`` is a short human-readable summary used for diagnostics and logs,
    such as ``"output span overflow"`` or ``"input span overflow for arg0_1"``.
    """

    levels: tuple[SpanOverflowTileLevel, ...]
    chunking_infos: tuple[ChunkingInfo, ...]
    reason: str | None = None


@dataclass(frozen=True)
class SpanOverflowCandidate:
    """One overflowing output/input span and the output dim that can shrink it.

    Candidate collection is intentionally separate from choosing split counts:
    first find every host dim that can structurally reduce an overflow, then let
    the cost search decide the cheapest legal split combination.
    """

    chunking_info: ChunkingInfo
    source: str


# Keep the search bounded; most real cases should need one or two tiled dims.
_MAX_TILE_DIMS = 3
_MAX_TILE_COMBOS = 512
_MAX_SPLITS_PER_DIM = 16
# Current LoopSpec lowering fully unrolls each coarse-tile loop into separate
# SDSC specs, so keep automatic plans conservative.  Manual spyre_hint remains
# user-controlled and can request larger counts explicitly.
_MAX_AUTO_TILE_SPLIT_COUNT = 64


def _layout_has_static_span_metadata(layout: FixedTiledLayout) -> bool:
    """Return True when span planning can use concrete layout metadata.

    The planner needs integer host sizes, host strides, physical device sizes,
    stride maps, and stick size.  If any of these are symbolic, skip auto tiling
    and leave the op to existing compiler paths; symbolic-shape support belongs
    to a later phase.
    """
    try:
        for values in (
            layout.size,
            layout.stride,
            layout.device_layout.device_size,
            layout.device_layout.stride_map,
        ):
            for value in values:
                int(value)
        int(layout.device_layout.elems_per_stick())
    except (TypeError, ValueError):
        return False
    return True


def _post_tile_layout_for_splits(
    original_layout: FixedTiledLayout,
    split_by_host_dim: dict[int, int],
    op_name: str,
) -> FixedTiledLayout:
    """Build the actual per-tile Spyre layout for one or more host splits.

    The search never trusts arithmetic alone.  For a candidate like
    ``{dim0: 32, dim1: 2}``, this shrinks the logical host sizes first and then
    calls ``_resize_device_layout`` so validation uses the same physical layout
    reconstruction as real coarse tiling.  Only called from the automatic
    planner's own search/validation, so candidates here are already bounded by
    ``_MAX_AUTO_TILE_SPLIT_COUNT``.
    """
    new_size = list(original_layout.size)
    for selected_host_dim, split_count in split_by_host_dim.items():
        if split_count <= 0:
            raise Unsupported(
                f"Cannot auto-tile {op_name}: split_count must be positive, "
                f"got {split_count}."
            )
        if selected_host_dim >= len(new_size):
            raise Unsupported(
                f"Cannot auto-tile {op_name}: selected host dim "
                f"{selected_host_dim} is out of bounds for layout size {new_size}."
            )

        try:
            full_size = int(new_size[selected_host_dim])
        except (TypeError, ValueError) as exc:
            raise Unsupported(
                f"Cannot auto-tile {op_name}: selected host dim "
                f"{selected_host_dim} has non-integral layout size "
                f"{new_size[selected_host_dim]!r}."
            ) from exc

        if full_size % split_count != 0:
            raise Unsupported(
                f"Cannot auto-tile {op_name}: selected host dim size {full_size} "
                f"is not divisible by split_count {split_count}."
            )
        new_size[selected_host_dim] = full_size // split_count

    new_stride = list(FlexibleLayout.contiguous_strides(new_size))

    try:
        device_layout = _resize_device_layout(
            original_layout.device_layout,
            [int(s) for s in original_layout.size],
            [int(s) for s in new_size],
        )
    except RuntimeError as exc:
        raise Unsupported(
            f"Cannot auto-tile {op_name}: post-tile device layout reconstruction "
            f"failed: {exc}"
        ) from exc
    return FixedTiledLayout(
        original_layout.device,
        original_layout.dtype,
        new_size,
        new_stride,
        device_layout,
    )


def _within_stick_host_dim(layout: FixedTiledLayout) -> int | None:
    """Return the logical host dim represented by the final stick coordinate.

    Splitting this host dim is only legal when each tile still contains a whole
    number of Spyre sticks; otherwise a coarse-tile boundary would cut through a
    physical stick.

    Returns ``None`` when no host stride exactly matches the device layout's
    final stride-map entry, e.g. for a layout whose physical stick-carrying
    host dim doesn't correspond to a literal stride match.  Guessing a
    fallback dim here (as an earlier revision did, via ``len(host_stride) -
    1``) risks silently validating stick alignment against the wrong host
    dim if the guess is wrong.  Returning ``None`` instead lets the caller
    fail safe -- matching the same "skip/reject rather than guess" discipline
    used elsewhere in this pass (ambiguous BMM symbol maps return empty;
    ``_resize_device_layout`` raises rather than guess an ambiguous stick
    dim by elimination).
    """
    sm_last = int(list(layout.device_layout.stride_map)[-1])
    host_stride = [int(s) for s in layout.stride]
    return next((i for i, s in enumerate(host_stride) if s == sm_last), None)


def _post_tile_stick_alignment_error(
    original_layout: FixedTiledLayout,
    selected_host_dim: int,
    split_count: int,
) -> str | None:
    """Return a diagnostic if a split cuts through physical sticks.

    Non-stick dims are always fine here.  For the within-stick host dim, the
    post-tile size must remain divisible by ``elems_per_stick``.
    """
    if split_count <= 1:
        return None

    within_stick_dim = _within_stick_host_dim(original_layout)
    if within_stick_dim is None:
        # Cannot confidently identify the stick dim for this layout -- treat
        # the split as unsafe rather than risk silently validating against
        # the wrong dim (or not validating at all).
        return (
            f"cannot determine the stick-carrying host dim for {original_layout!r} "
            f"to validate split_count {split_count} against host dim "
            f"{selected_host_dim}"
        )
    if selected_host_dim != within_stick_dim:
        return None

    full_size = int(original_layout.size[selected_host_dim])
    tile_size = full_size // split_count
    stick_elems = original_layout.device_layout.elems_per_stick()
    if tile_size % stick_elems == 0:
        return None

    return (
        f"split_count {split_count} makes selected host dim {selected_host_dim} "
        f"tile size {tile_size}, which is not aligned to Spyre stick size "
        f"{stick_elems}; coarse-tile boundaries would cut through physical sticks"
    )


def _is_batch_matmul_reduction(op: ComputedBuffer) -> bool:
    """Return True for matmul/BMM reductions with output-dim tiling policy.

    ``F.linear`` lowers to a Reduction with ``reduction_type=batchmatmul`` even
    though the user did not call ``torch.bmm``.  For these ops, output dims
    such as M/N/vocab can be tiled; K remains a reduction-only dim.
    """
    return (
        isinstance(op.data, Reduction)
        and getattr(op.data, "reduction_type", None) == BATCH_MATMUL_OP
    )


def _output_symbol_to_dim(op: ComputedBuffer) -> dict[sympy.Symbol, int]:
    """Map each output iteration symbol to its logical output host dim.

    This answers: "which output dimension does this symbol represent?"  For an
    output write dependency like ``MemoryDep('buf1', 4096*d0 + d1,
    {d0: 49216, d1: 4096})``, the result is ``{d0: 0, d1: 1}``: symbol ``d0``
    controls host dim 0 and symbol ``d1`` controls host dim 1.  Span candidates
    use this map to turn a physical coordinate such as ``floor(d0 / 64)`` back
    into the coarse-tile dimension ``selected_host_dim=0``.
    """
    symbol_to_dim: dict[sympy.Symbol, int] = {}
    try:
        out_coords = op_out_coords(op)
    except (AttributeError, StopIteration, TypeError, RuntimeError):
        out_coords = []

    for dim, coord in enumerate(out_coords):
        for sym in getattr(coord, "free_symbols", ()):  # pragma: no branch - sympy API
            symbol_to_dim.setdefault(sym, dim)

    try:
        out_dep = next(iter(op.get_read_writes().writes))
    except (AttributeError, StopIteration, TypeError):
        return symbol_to_dim

    for dim, sym in enumerate(out_dep.ranges):
        symbol_to_dim.setdefault(sym, dim)
    return symbol_to_dim


def _bmm_output_symbol_to_dim(
    op: ComputedBuffer,
    input_deps: list[tuple[MemoryDep, FixedTiledLayout]],
) -> dict[sympy.Symbol, int]:
    """Map BMM output symbols while excluding the single K symbol.

    The generic output map says which symbols are output-controlled.  This BMM
    helper additionally checks input deps to make sure there is exactly one
    non-output symbol, the matmul reduction K.  Ambiguous cases return an empty
    map so the planner skips rather than guessing.
    """
    symbol_to_dim = _output_symbol_to_dim(op)
    if not symbol_to_dim:
        return {}

    reduction_symbols = {
        sym for dep, _ in input_deps for sym in dep.ranges if sym not in symbol_to_dim
    }
    if len(reduction_symbols) != 1:
        logger.debug(
            "span_overflow_bmm: op=%s skipped; expected one K symbol, got %s",
            op.get_name(),
            sorted(reduction_symbols, key=str),
        )
        return {}
    return symbol_to_dim


def _input_read_deps(op: ComputedBuffer) -> list[tuple[MemoryDep, FixedTiledLayout]]:
    """Return direct fixed-layout input MemoryDeps for span planning.

    Indirect/gather reads are skipped before reaching this path.  For each
    remaining MemoryDep, this resolves the producer buffer and captures its
    physical FixedTiledLayout so input span analysis uses the input tensor's own
    device layout, not the consumer output layout.
    """
    try:
        reads = op.get_read_writes().reads
    except (AttributeError, TypeError):
        return []

    deps: list[tuple[MemoryDep, FixedTiledLayout]] = []
    for dep in reads:
        try:
            if (
                not isinstance(dep, MemoryDep)
                or not isinstance(dep.index, sympy.Basic)
                or dep.is_indirect()
            ):
                continue
            buf = V.graph.get_buffer(dep.name)
            layout = _fixed_read_layout(buf)
        except (AttributeError, TypeError, RuntimeError):
            continue
        deps.append((dep, layout))
    return deps


def _range_size_for_symbol(dep: MemoryDep, sym: sympy.Symbol) -> int | None:
    """Return the concrete MemoryDep range size for ``sym`` if available."""
    try:
        return int(dep.ranges[sym])
    except (KeyError, TypeError, ValueError):
        return None


def _coordinate_span_elems(
    coord: sympy.Expr,
    dep: MemoryDep,
    split_by_symbol: dict[sympy.Symbol, int],
) -> int | None:
    """Return a conservative span bound for one physical device coordinate.

    ``split_by_symbol`` gives the hypothetical split count for each free symbol
    in ``coord``; a symbol absent from the map is treated as unsplit.  Plain
    affine/floor terms are evaluated at endpoints.  Modulo terms are bounded by
    their modulus because their maximum can occur just before wraparound rather
    than at either endpoint.

    A per-symbol term can contain more than one ``Mod`` atom on that same
    symbol with different moduli (e.g. two overlapping/interleaved stick
    strides both keyed on the same loop variable).  Each individual Mod atom
    is piecewise-linear in the symbol and only changes slope at its own
    wraparound point, so the extrema of their sum over the symbol's domain
    can only occur at one of those per-atom wraparound points or at the
    domain endpoints -- never strictly between them.  Evaluating the whole
    term at every such candidate point and taking the max/min therefore finds
    the true extrema, rather than assuming (as evaluating at a single
    critical point derived only from the smallest modulus would) that the
    smallest-modulus term's own wraparound point also maximizes every other
    Mod term summed alongside it.

    This evaluates each tile as if its symbol always starts at 0, not at that
    tile's offset into the full tensor.  That is safe here: current automatic
    coarse tiling fully unrolls each tile into its own separately generated
    SDSC spec (see span_overflow_hint_analysis.md), and _divide_ranges only
    shrinks the declared range for that spec -- it never rewrites the index
    expression to add a tile offset.  So a symbol's value within any one
    tile's generated spec is genuinely bounded by [0, per_tile_size) every
    time; the tile's real position in the full tensor is resolved afterward,
    by a separate physical address offset that this Mod computation never
    sees.  A Mod term therefore cannot see a later tile's actual wraparound
    window, because no later tile ever substitutes a shifted value here.
    """
    per_core_max = 0
    per_core_min = 0
    for sym in coord.free_symbols:
        range_size = _range_size_for_symbol(dep, sym)
        if range_size is None:
            return None
        split_count = split_by_symbol.get(sym, 1)
        if range_size % split_count != 0:
            return None
        range_size //= split_count
        term = coord.subs({other: 0 for other in coord.free_symbols - {sym}})
        if term.has(sympy.Mod):
            mod_atoms = term.atoms(sympy.Mod)
            if any(mod.args[0] != sym for mod in mod_atoms):
                # The critical-point trick below is only exact when each
                # Mod's argument is the bare symbol: its wraparound then
                # occurs exactly at sym = modulus - 1. A coefficient on the
                # argument (e.g. Mod(3*sym, 64)) wraps at other sym values
                # too -- the true max can occur anywhere sym*coefficient
                # crosses a multiple of the modulus, not just at sym =
                # modulus - 1. Evaluating only at that one point can
                # silently underestimate the span (confirmed: Mod(3*h, 64)
                # over h in [0, 100) evaluates to a max of 61 at the naive
                # critical point h=63, but the true max is 63, at h=21).
                # Fail safe rather than risk accepting a plan that still
                # overflows on real hardware.
                return None
            moduli = {int(mod.args[1]) for mod in mod_atoms}
            candidate_points = {min(range_size, modulus) - 1 for modulus in moduli}
            candidate_points.add(0)
            candidate_points.add(range_size - 1)
            values = [int(term.subs(sym, point)) for point in candidate_points]
            per_core_max += max(values)
            per_core_min += min(values)
            continue
        per_core_max += int(term.subs(sym, range_size - 1))
        per_core_min += int(term.subs(sym, 0))
    return per_core_max - per_core_min + 1


def _split_by_symbol_for_coord(
    coord: sympy.Expr,
    symbol_to_dim: dict[sympy.Symbol, int],
    split_by_host_dim: dict[int, int],
) -> dict[sympy.Symbol, int]:
    """Return split counts for output symbols present in coord."""
    return {
        sym: split_by_host_dim.get(symbol_to_dim[sym], 1)
        for sym in coord.free_symbols
        if sym in symbol_to_dim
    }


def _tile_aware_inner_stride_elems(
    device_coords: list[sympy.Expr],
    device_size: list[int],
    dep: MemoryDep,
    inner_start_dim: int,
    symbol_to_dim: dict[sympy.Symbol, int],
    split_by_host_dim: dict[int, int],
) -> int | None:
    """Return inner physical span size after applying output tile splits.

    For input reads, splitting one output dim can reduce the physical span
    nested under another output dim.  Example: a d1 coordinate may have an
    inner d2 coordinate; when the candidate combo splits both d1 and
    d2, the byte span for d1 must use the tiled d2 range instead of
    the full input tensor's physical d2 size.
    """
    stride_elems = 1
    for device_dim, coord in enumerate(
        device_coords[inner_start_dim:], inner_start_dim
    ):
        if not coord.free_symbols:
            # A constant coordinate (e.g. a broadcast dim) never shrinks under
            # any hypothetical split -- its physical extent still occupies the
            # full declared device_size, so use that directly instead of
            # re-evaluating a span of 1 for a term with nothing to vary.
            stride_elems *= device_size[device_dim]
            continue
        split_by_symbol = _split_by_symbol_for_coord(
            coord,
            symbol_to_dim,
            split_by_host_dim,
        )
        coord_span_elems = _coordinate_span_elems(coord, dep, split_by_symbol)
        if coord_span_elems is None:
            return None
        stride_elems *= min(coord_span_elems, device_size[device_dim])
    return stride_elems


def _device_coordinates_for_span(
    layout: FixedTiledLayout,
    dep: MemoryDep,
) -> list[sympy.Expr]:
    """Return physical coordinates for span planning.

    This pass analyzes only non-stick coordinates and all callers ignore
    ``coords[-1]``.  Use the same index concretization and coordinate
    decomposition as ``device_coordinates()``, but skip its final stick-expression
    validation so an unsupported stick form does not reject an otherwise valid
    non-stick span candidate.
    """
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(
        layout.device_layout.device_size,
        layout.device_layout.stride_map,
        dep.ranges,
        index,
        None,
    )


def _input_span_infos_controlled_by_output_dims(
    op: ComputedBuffer,
    max_cores: int,
    *,
    selected_host_dim: int | None = None,
    split_count: int = 1,
    split_by_host_dim: dict[int, int] | None = None,
) -> list[InputSpanInfo]:
    """Return overflowing input spans controlled by output dimensions.

    Input spans controlled by reduction-only symbols are intentionally skipped
    because output-range coarse tiling cannot split reduction ranges without
    partial-result accumulation.  ``split_by_host_dim`` models hypothetical
    output coarse tiles during combined post-tile validation.
    """
    if split_by_host_dim is None:
        split_by_host_dim = (
            {selected_host_dim: split_count} if selected_host_dim is not None else {}
        )

    input_deps = _input_read_deps(op)
    symbol_to_dim = (
        _bmm_output_symbol_to_dim(op, input_deps)
        if _is_batch_matmul_reduction(op)
        else _output_symbol_to_dim(op)
    )
    if not symbol_to_dim:
        return []

    infos: list[InputSpanInfo] = []
    for dep, layout in input_deps:
        if not _layout_has_static_span_metadata(layout):
            continue

        device_size = [int(s) for s in layout.device_layout.device_size]
        itemsize = layout.dtype.itemsize
        stick_elems = layout.device_layout.elems_per_stick()
        try:
            device_coords = _device_coordinates_for_span(layout, dep)
        except (TypeError, ValueError, RuntimeError, Unsupported):
            continue

        for device_dim, coord in enumerate(device_coords[:-1]):
            if not coord.free_symbols:
                continue

            output_syms = [sym for sym in coord.free_symbols if sym in symbol_to_dim]
            reduction_syms = [
                sym for sym in coord.free_symbols if sym not in symbol_to_dim
            ]
            if reduction_syms:
                logger.debug(
                    "span_overflow_input: op=%s dep=%s skipped coord=%s "
                    "controlled by reduction symbols %s",
                    op.get_name(),
                    dep.name,
                    coord,
                    reduction_syms,
                )
                continue

            # A coordinate can be jointly controlled by more than one output
            # symbol, for example two interleaved dims sharing one physical
            # stride.  Compute its span using every contributing dim's
            # hypothetical split, then register a candidate against each
            # contributing dim so the combo search can consider splitting
            # them together.
            split_by_symbol = _split_by_symbol_for_coord(
                coord,
                symbol_to_dim,
                split_by_host_dim,
            )
            coord_span_elems = _coordinate_span_elems(coord, dep, split_by_symbol)
            if coord_span_elems is None:
                continue

            inner_stride_elems = _tile_aware_inner_stride_elems(
                device_coords,
                device_size,
                dep,
                device_dim + 1,
                symbol_to_dim,
                split_by_host_dim,
            )
            if inner_stride_elems is None:
                continue

            # Span-overflow planning is intentionally conservative: coarse tiles
            # must make the span safe even if downstream work division provides
            # no additional split for this coordinate.  core_split_estimate=1
            # below is that assumption made explicit -- see the max_cores note
            # in plan_span_overflow_tile's docstring.
            per_core_span = coord_span_elems * inner_stride_elems * itemsize
            if per_core_span <= MAX_SPAN_BYTES:
                continue

            for controlling_symbol in output_syms:
                output_dim = symbol_to_dim[controlling_symbol]
                infos.append(
                    InputSpanInfo(
                        chunking_info=ChunkingInfo(
                            total_bytes=math.prod(device_size) * itemsize,
                            per_core_span=per_core_span,
                            core_split_estimate=1,
                            selected_device_dim_size=coord_span_elems,
                            selected_device_span_stride_elems=inner_stride_elems,
                            selected_host_dim=output_dim,
                            stick_elems=stick_elems,
                            reason=f"input span overflow for {dep.name}",
                        ),
                        dep_name=dep.name,
                        controlling_symbol=controlling_symbol,
                    )
                )
    return infos


def _output_write_dep(op: ComputedBuffer) -> MemoryDep | None:
    """Return the output MemoryDep used to convert host indexes to device coords.

    If the op has no concrete output write dependency, automatic span planning
    cannot safely map physical span pressure back to logical host dims.
    """
    try:
        dep = next(iter(op.get_read_writes().writes))
    except (AttributeError, StopIteration, TypeError):
        return None
    if not isinstance(dep, MemoryDep) or not isinstance(dep.index, sympy.Basic):
        return None
    return dep


def _output_dep_with_split_ranges(
    dep: MemoryDep,
    symbol_to_dim: dict[sympy.Symbol, int],
    split_by_host_dim: dict[int, int],
) -> MemoryDep | None:
    """Return ``dep`` with ranges shrunk to model a hypothetical output tile."""
    if not split_by_host_dim:
        return dep

    ranges: dict[sympy.Symbol, int] = {}
    for sym, size in dep.ranges.items():
        try:
            range_size = int(size)
        except (TypeError, ValueError):
            return None
        split_count = split_by_host_dim.get(symbol_to_dim.get(sym, -1), 1)
        if range_size % split_count != 0:
            return None
        ranges[sym] = range_size // split_count

    return MemoryDep(
        dep.name,
        dep.index,
        tuple(ranges.keys()),
        tuple(ranges.values()),
        getattr(dep, "mode", None),
    )


def _output_span_candidates_from_op(
    op: ComputedBuffer,
    *,
    layout: FixedTiledLayout | None = None,
    split_by_host_dim: dict[int, int] | None = None,
    op_name: str | None = None,
) -> list[SpanOverflowCandidate]:
    """Return output host dims whose physical write span needs tiling.

    This answers: "If work division gives this op no useful split, does any
    non-stick output device coordinate span more than ``MAX_SPAN_BYTES``, and
    which output host dim controls that coordinate?"  The result is a list of
    candidate host dims that coarse tiling can shrink structurally.

    The scan is physical-coordinate first, not logical-dim first: compute device
    coordinates from the output write ``MemoryDep``, skip the final within-stick
    coordinate, then map each remaining coordinate back to a logical output dim
    only when exactly one output symbol controls it.  For example,
    ``floor(d0 / 64)`` maps to host dim ``0`` while the stick coordinate
    ``d0 % 64`` is ignored.
    """
    layout = layout or op.layout
    split_by_host_dim = split_by_host_dim or {}

    name = op_name or op.get_name()
    out_dep = _output_write_dep(op)
    if out_dep is None:
        logger.debug(
            "span_overflow_output: op=%s skipped; output write MemoryDep unavailable",
            name,
        )
        return []

    symbol_to_dim = _output_symbol_to_dim(op)
    if not symbol_to_dim:
        logger.debug(
            "span_overflow_output: op=%s skipped; output symbol map unavailable",
            name,
        )
        return []

    out_dep = _output_dep_with_split_ranges(out_dep, symbol_to_dim, split_by_host_dim)
    if out_dep is None:
        logger.debug(
            "span_overflow_output: op=%s skipped; output ranges do not match split %s",
            name,
            split_by_host_dim,
        )
        return []

    try:
        device_coords = _device_coordinates_for_span(layout, out_dep)
    except (TypeError, ValueError, RuntimeError, Unsupported) as exc:
        logger.debug(
            "span_overflow_output: op=%s skipped; device-coordinate analysis failed: %s",
            name,
            exc,
        )
        return []

    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    candidates: list[SpanOverflowCandidate] = []
    for device_dim, coord in enumerate(device_coords[:-1]):
        if not coord.free_symbols:
            continue
        output_syms = [sym for sym in coord.free_symbols if sym in symbol_to_dim]
        # A coordinate can be jointly controlled by more than one output
        # symbol, for example two interleaved dims sharing one physical
        # stride -- ``floor(d1 / 64)`` maps cleanly to the output dim for
        # ``d1``; ``floor(d0/2) + floor(d1/64)`` is still tileable via both
        # ``d0`` and ``d1``.  Only a coordinate that also involves a
        # reduction-only symbol is unsafe to tile and is skipped here.
        if len(output_syms) != len(coord.free_symbols):
            logger.debug(
                "span_overflow_output: op=%s skipped coord=%s output_syms=%s",
                op.get_name(),
                coord,
                output_syms,
            )
            continue

        split_by_symbol = {
            sym: split_by_host_dim.get(symbol_to_dim[sym], 1) for sym in output_syms
        }
        coord_span_elems = _coordinate_span_elems(coord, out_dep, split_by_symbol)
        if coord_span_elems is None:
            continue
        per_core_span = (
            coord_span_elems * math.prod(device_size[device_dim + 1 :]) * itemsize
        )
        if per_core_span <= MAX_SPAN_BYTES:
            continue
        for controlling_symbol in output_syms:
            output_dim = symbol_to_dim[controlling_symbol]
            candidates.append(
                SpanOverflowCandidate(
                    chunking_info=ChunkingInfo(
                        total_bytes=math.prod(device_size) * itemsize,
                        per_core_span=per_core_span,
                        # core_split_estimate=1: see the max_cores note in
                        # plan_span_overflow_tile's docstring -- work division
                        # is not yet modeled, so this is always 1 today.
                        core_split_estimate=1,
                        selected_device_dim_size=coord_span_elems,
                        selected_device_span_stride_elems=math.prod(
                            device_size[device_dim + 1 :]
                        ),
                        selected_host_dim=output_dim,
                        stick_elems=layout.device_layout.elems_per_stick(),
                        reason="output span overflow",
                    ),
                    source="output",
                )
            )

    return _log_span_candidates(
        candidates,
        layout,
        op_name=op_name or op.get_name(),
    )


def _log_span_candidates(
    candidates: list[SpanOverflowCandidate],
    layout: FixedTiledLayout,
    *,
    op_name: str,
) -> list[SpanOverflowCandidate]:
    """Log candidate span facts and return the candidate list unchanged."""
    device_size = [int(s) for s in layout.device_layout.device_size]
    for candidate in candidates:
        info = candidate.chunking_info
        logger.info(
            "[span_overflow_hint_analysis] trigger=%s op=%s "
            "selected_host_dim=%d selected_device_dim_size=%d "
            "selected_device_span_stride_elems=%d per_tile_span=%.2f MB "
            "total=%.2f GB shape=%s dtype=%s device_size=%s "
            "span_limit=%.2f MB",
            candidate.source,
            op_name,
            info.selected_host_dim,
            info.selected_device_dim_size,
            info.selected_device_span_stride_elems,
            info.per_core_span / (1024**2),
            info.total_bytes / (1024**3),
            list(layout.size),
            layout.dtype,
            device_size,
            MAX_SPAN_BYTES / (1024**2),
        )
    return candidates


def _input_span_candidates(
    op: ComputedBuffer,
    max_cores: int,
    *,
    split_by_host_dim: dict[int, int] | None = None,
) -> list[SpanOverflowCandidate]:
    """Collect Reduction/BMM input spans controlled by output dimensions.

    This wraps the lower-level input scan and filters out host dims that have no
    legal nontrivial coarse split after output/input stick-alignment checks.
    """
    candidates: list[SpanOverflowCandidate] = []
    for info in _input_span_infos_controlled_by_output_dims(
        op,
        max_cores,
        split_by_host_dim=split_by_host_dim,
    ):
        host_dim = info.chunking_info.selected_host_dim
        if not _host_dim_has_legal_nontrivial_split(op, host_dim):
            logger.debug(
                "span_overflow_input: op=%s dep=%s host_dim=%d has no "
                "legal non-stick coarse split; skipping input candidate",
                op.get_name(),
                info.dep_name,
                host_dim,
            )
            continue
        candidates.append(
            SpanOverflowCandidate(info.chunking_info, source=f"input:{info.dep_name}")
        )
    return candidates


def _candidate_host_dims(
    candidates: list[SpanOverflowCandidate],
) -> list[int]:
    """Return unique candidate host dims ordered by strongest span pressure.

    Multiple output/input span candidates can point at the same logical output
    dim.  The search only needs to split that dim once, so this collapses
    candidates by ``selected_host_dim`` and orders dims by their largest
    observed byte span.  This ordering does not decide correctness; it only
    gives the combo search a stable pressure-first dim order.
    """
    max_span_by_dim: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for idx, candidate in enumerate(candidates):
        dim = candidate.chunking_info.selected_host_dim
        first_seen.setdefault(dim, idx)
        max_span_by_dim[dim] = max(
            max_span_by_dim.get(dim, 0), candidate.chunking_info.per_core_span
        )
    return sorted(
        max_span_by_dim,
        key=lambda dim: (-max_span_by_dim[dim], first_seen[dim]),
    )


def _host_dim_has_legal_nontrivial_split(op: ComputedBuffer, host_dim: int) -> bool:
    """Return True when ``host_dim`` has at least one legal split greater than 1."""
    try:
        candidates = _split_candidates_for_host_dim(op, host_dim)
    except Unsupported:
        return False
    return any(split > 1 for split in candidates)


def _candidate_required_split_count(candidate: SpanOverflowCandidate) -> int:
    """Return the minimum split needed if this candidate's dim tiled alone.

    This is ``ceil(per_core_span / MAX_SPAN_BYTES)``: if a dim creates a
    384 MB span and the limit is 256 MB, it needs at least split ``2`` when
    considered by itself.

    The combined search can still choose a different legal divisor or combine
    multiple smaller splits across dims; this value is only the starting lower
    bound for that dim's split candidates.
    """
    return max(1, math.ceil(candidate.chunking_info.per_core_span / MAX_SPAN_BYTES))


def _required_split_counts_by_host_dim(
    candidates: list[SpanOverflowCandidate],
) -> dict[int, int]:
    """Return the strongest required split count for each logical host dim.

    If output and input analysis both say host dim ``0`` can overflow, keep the
    larger split requirement.  Example: output needs split ``2`` but an input
    span controlled by the same dim needs split ``4``; the combo search should
    generate dim-0 divisors around ``4`` rather than around ``2``.
    """
    required_by_dim: dict[int, int] = {}
    for candidate in candidates:
        dim = candidate.chunking_info.selected_host_dim
        required_by_dim[dim] = max(
            required_by_dim.get(dim, 1),
            _candidate_required_split_count(candidate),
        )
    return required_by_dim


def _cap_split_candidates(
    legal_candidates: list[int],
    required_count: int,
) -> list[int]:
    """Bound legal divisors while keeping cheap and required-area choices.

    A large highly-composite dim can have many legal divisors.  Keep a compact
    set that still contains cheap small splits, divisors near ``required_count``,
    and a few large fallbacks, so the Cartesian combo search stays bounded.
    """
    if len(legal_candidates) <= _MAX_SPLITS_PER_DIM:
        return legal_candidates

    selected: list[int] = []

    def add(split: int) -> None:
        if split in legal_candidates and split not in selected:
            selected.append(split)

    add(1)
    for split in legal_candidates:
        if split > 1:
            add(split)
        if len(selected) >= min(5, _MAX_SPLITS_PER_DIM):
            break

    required_idx = next(
        (idx for idx, split in enumerate(legal_candidates) if split >= required_count),
        len(legal_candidates) - 1,
    )
    for idx in range(
        max(0, required_idx - 2),
        min(len(legal_candidates), required_idx + 4),
    ):
        add(legal_candidates[idx])

    for split in reversed(legal_candidates):
        add(split)
        if len(selected) >= _MAX_SPLITS_PER_DIM:
            break

    return sorted(selected)[:_MAX_SPLITS_PER_DIM]


def _host_dim_target_symbols(
    op: ComputedBuffer,
    host_dim: int,
) -> set[sympy.Symbol]:
    """Return output-loop symbols that represent one logical host dim.

    Input stick checks need this bridge: a split is chosen in consumer output
    dim space, but each input dependency may place the same symbol at a
    different input host dim or physical stick position.
    """
    if _is_batch_matmul_reduction(op):
        symbol_to_dim = _bmm_output_symbol_to_dim(op, _input_read_deps(op))
        if not symbol_to_dim:
            symbol_to_dim = _output_symbol_to_dim(op)
    else:
        symbol_to_dim = _output_symbol_to_dim(op)
    return {sym for sym, dim in symbol_to_dim.items() if dim == host_dim}


def _input_stick_alignment_error(
    op: ComputedBuffer,
    host_dim: int,
    split_count: int,
) -> str | None:
    """Return a diagnostic if splitting ``host_dim`` misaligns an input's sticks.

    ``host_dim`` is an index into the output op's iteration space.  The same
    iteration symbol also addresses each input dependency, but an input
    tensor's own physical layout (stride order, stick mapping) can differ
    from the output's, so stick alignment must be checked against each
    input's own layout independently of the output layout check.

    A symbol's position in ``dep.ranges`` reflects the op's shared, output-
    derived iteration order -- it is not guaranteed to match this specific
    input's own physical dimension order (e.g. for a transposed read), so we
    cannot just index by that position.  Instead, ``host_coordinates`` -- the
    same helper ``op_out_coords`` uses for the output -- derives this input's
    own per-dimension coordinate expressions directly from its layout and
    index, and we find which one(s) ``sym`` actually controls.

    A single input dimension's coordinate can be jointly controlled by
    ``sym`` together with another symbol (e.g. an interleaved/collapsed
    physical stride after a view or transpose).  Checking only dimensions
    where ``sym`` is the *sole* free symbol would silently skip stick
    alignment for that dimension entirely -- mirroring the sibling span
    candidate search (``_input_span_infos_controlled_by_output_dims``), we
    instead check every input dimension ``sym`` contributes to, whether or
    not other symbols also contribute to it, since splitting ``sym`` can cut
    a physical stick on that dimension either way.
    """
    if split_count <= 1:
        return None

    target_symbols = _host_dim_target_symbols(op, host_dim)
    if not target_symbols:
        return None

    for dep, layout in _input_read_deps(op):
        if not _layout_has_static_span_metadata(layout):
            continue
        input_coords = host_coordinates(layout, dep, None)
        for sym in target_symbols:
            matching_dims = [
                i for i, coord in enumerate(input_coords) if sym in coord.free_symbols
            ]
            for input_host_dim in matching_dims:
                error = _post_tile_stick_alignment_error(
                    layout, input_host_dim, split_count
                )
                if error is not None:
                    return (
                        f"input dependency {dep.name} host dim {input_host_dim}: "
                        f"{error}"
                    )
    return None


def _split_candidates_for_host_dim(
    op: ComputedBuffer,
    host_dim: int,
    required_count: int = 1,
) -> list[int]:
    """Return bounded legal split counts for one logical output host dim.

    The split count must divide the op's output range for ``host_dim`` exactly,
    because coarse tiling emits equal-sized loop tiles.  Then the candidates are
    filtered for output and input stick alignment: a split is legal only if the
    resulting tile boundary does not cut through physical sticks in the output
    layout or any direct input layout controlled by the same output symbol.
    """
    ranges = list(op.data.ranges)
    if host_dim >= len(ranges):
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim {host_dim} "
            f"is out of bounds for data ranges {ranges}."
        )
    try:
        full_size = int(ranges[host_dim])
    except (TypeError, ValueError) as exc:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim {host_dim} "
            f"has non-integral range {ranges[host_dim]!r}."
        ) from exc
    if full_size <= 1:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: selected host dim {host_dim} "
            f"has unsplittable range {full_size}."
        )
    candidates = sorted(
        {
            d
            for i in range(1, math.isqrt(full_size) + 1)
            if full_size % i == 0
            for d in (i, full_size // i)
        }
    )
    legal_candidates = [
        split
        for split in candidates
        if split == 1
        or (
            # A split that shrinks this dim's per-tile extent to exactly 1
            # element is rejected for Reduction ops: the codegen/DDC lowering
            # path can drop unit-size dims from the op's iteration space, which
            # can under-count the dimensions fixed-arity reduction templates
            # expect and crash native codegen (DtException: "Not enough
            # dimensions") rather than fail cleanly at the Python level.
            # Pointwise unit tiles remain legal; there is existing coverage for
            # full-size exact divisors on Pointwise ops.
            (not isinstance(op.data, Reduction) or full_size // split > 1)
            and split <= _MAX_AUTO_TILE_SPLIT_COUNT
            and _post_tile_stick_alignment_error(op.layout, host_dim, split) is None
            and _input_stick_alignment_error(op, host_dim, split) is None
        )
    ]
    capped_candidates = _cap_split_candidates(legal_candidates, required_count)
    logger.debug(
        "[span-overflow divisors] op=%s host_dim=%d required=%d max_auto_split=%d legal=%s capped=%s",
        op.get_name(),
        host_dim,
        required_count,
        _MAX_AUTO_TILE_SPLIT_COUNT,
        legal_candidates,
        capped_candidates,
    )
    if len(legal_candidates) > len(capped_candidates):
        logger.debug(
            "span_overflow_tile_search: op=%s host_dim=%d limiting %d split "
            "candidates to %d before combo search (required_count=%d)",
            op.get_name(),
            host_dim,
            len(legal_candidates),
            len(capped_candidates),
            required_count,
        )
    return capped_candidates


def _combo_cost(combo: tuple[int, ...]) -> tuple[int, int, int, tuple[int, ...]]:
    """Rank split combinations by compile/runtime cost.

    Prefer fewer total loop tiles first, then fewer tiled dimensions, then a
    smaller largest split.  The final tuple gives deterministic tie-breaking.
    """
    return (
        math.prod(combo),
        sum(split > 1 for split in combo),
        max(combo),
        combo,
    )


def _iter_split_combos(
    split_candidates: list[list[int]],
) -> list[tuple[int, ...]]:
    """Return bounded split combinations in increasing cost order.

    Each input list contains legal split counts for one candidate host dim.  The
    Cartesian product represents all ways to tile those dims together, including
    leaving a dim unsplit with count ``1``.  The search checks the cheapest
    combinations first and caps the total attempts for compile-time safety.
    """
    combos = sorted(itertools.product(*split_candidates), key=_combo_cost)
    if len(combos) > _MAX_TILE_COMBOS:
        logger.debug(
            "span_overflow_tile_search: truncating %d combos to %d",
            len(combos),
            _MAX_TILE_COMBOS,
        )
    return combos[:_MAX_TILE_COMBOS]


def _combined_tile_stick_alignment_error(
    op: ComputedBuffer,
    original_layout: FixedTiledLayout,
    split_by_host_dim: dict[int, int],
) -> str | None:
    """Return the first stick-alignment error for a multi-dim tile plan.

    Single-dim candidates are already pre-filtered, but this check keeps the
    final combined plan honest and reports whether the output layout or one of
    the input layouts would get a stick-cutting tile boundary.
    """
    for host_dim, split_count in split_by_host_dim.items():
        error = _post_tile_stick_alignment_error(original_layout, host_dim, split_count)
        if error is not None:
            return error
        error = _input_stick_alignment_error(op, host_dim, split_count)
        if error is not None:
            return error
    return None


def _remaining_span_candidates_after_tile(
    op: ComputedBuffer,
    max_cores: int,
    split_by_host_dim: dict[int, int],
) -> list[SpanOverflowCandidate]:
    """Return spans that still overflow after a hypothetical combined tile.

    This is the validation step.  It rebuilds the post-tile output layout using
    the actual Spyre layout resize logic, then reruns output and reduction input
    span analysis with those split counts.  A combo is accepted only when this
    returns no remaining candidates, meaning the tiled op is structurally within
    span even if work division gives no extra help.
    """
    tiled_layout = _post_tile_layout_for_splits(
        op.layout,
        split_by_host_dim,
        op.get_name(),
    )
    remaining = _output_span_candidates_from_op(
        op,
        layout=tiled_layout,
        split_by_host_dim=split_by_host_dim,
        op_name=f"{op.get_name()}:post_tile",
    )
    if isinstance(op.data, Reduction):
        remaining += _input_span_candidates(
            op,
            max_cores,
            split_by_host_dim=split_by_host_dim,
        )
    return remaining


def _search_min_cost_tile_plan(
    op: ComputedBuffer,
    max_cores: int,
    candidates: list[SpanOverflowCandidate],
) -> SpanOverflowTilePlan | None:
    """Find the cheapest combined coarse-tile plan that clears all candidates.

    The incoming candidates say which logical output dims can reduce an
    overflowing physical span.  This function converts those dims into legal
    split divisors, tries split combinations in cost order, validates each
    combination against reconstructed post-tile layouts, and returns the first
    plan that leaves no output/input span overflow.
    """
    host_dims = _candidate_host_dims(candidates)
    logger.debug(
        "[span-overflow search] op=%s candidates=%s host_dims=%s",
        op.get_name(),
        [
            {
                "source": candidate.source,
                "host_dim": candidate.chunking_info.selected_host_dim,
                "device_dim_size": candidate.chunking_info.selected_device_dim_size,
                "stride_elems": candidate.chunking_info.selected_device_span_stride_elems,
                "span_mb": candidate.chunking_info.per_core_span / (1024**2),
                "reason": candidate.chunking_info.reason,
            }
            for candidate in candidates
        ],
        host_dims,
    )
    if not host_dims:
        logger.debug(
            "[span-overflow search] op=%s no candidate host dims", op.get_name()
        )
        return None
    if len(host_dims) > _MAX_TILE_DIMS:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: span-overflow planning found "
            f"{len(host_dims)} candidate host dims {host_dims}, exceeding the "
            f"bounded search limit {_MAX_TILE_DIMS}."
        )

    required_by_dim = _required_split_counts_by_host_dim(candidates)
    split_candidates = [
        _split_candidates_for_host_dim(op, dim, required_by_dim.get(dim, 1))
        for dim in host_dims
    ]
    logger.debug(
        "[span-overflow search] op=%s required_by_dim=%s split_candidates=%s",
        op.get_name(),
        required_by_dim,
        dict(zip(host_dims, split_candidates)),
    )
    first_stick_error: str | None = None
    for combo in _iter_split_combos(split_candidates):
        split_by_host_dim = {
            dim: split for dim, split in zip(host_dims, combo) if split > 1
        }
        if not split_by_host_dim:
            continue

        stick_error = _combined_tile_stick_alignment_error(
            op, op.layout, split_by_host_dim
        )
        if stick_error is not None:
            first_stick_error = first_stick_error or stick_error
            logger.debug(
                "[span-overflow search] op=%s combo=%s rejected_stick=%s",
                op.get_name(),
                split_by_host_dim,
                stick_error,
            )
            continue

        try:
            remaining = _remaining_span_candidates_after_tile(
                op,
                max_cores,
                split_by_host_dim,
            )
        except Unsupported as exc:
            logger.debug(
                "span_overflow_tile_search: op=%s combo=%s rejected: %s",
                op.get_name(),
                split_by_host_dim,
                exc,
            )
            continue
        if remaining:
            logger.debug(
                "span_overflow_tile_search: op=%s combo=%s leaves %d spans",
                op.get_name(),
                split_by_host_dim,
                len(remaining),
            )
            continue

        # Emit loop levels outer-to-inner by host dimension; the search order can
        # differ because it is driven by span pressure.
        levels = tuple(
            SpanOverflowTileLevel(
                selected_host_dim=dim,
                split_count=split_by_host_dim[dim],
            )
            for dim in sorted(split_by_host_dim)
        )
        logger.info(
            "[span-overflow search] op=%s selected_split=%s levels=%s",
            op.get_name(),
            split_by_host_dim,
            [(level.selected_host_dim, level.split_count) for level in levels],
        )
        return SpanOverflowTilePlan(
            levels=levels,
            chunking_infos=tuple(candidate.chunking_info for candidate in candidates),
            reason="; ".join(
                sorted(
                    {
                        candidate.chunking_info.reason or candidate.source
                        for candidate in candidates
                    }
                )
            ),
        )

    if first_stick_error is not None:
        raise Unsupported(
            f"Cannot auto-tile {op.get_name()}: no legal combined split preserves "
            f"Spyre stick alignment. First rejected candidate: {first_stick_error}."
        )
    raise Unsupported(
        f"Cannot auto-tile {op.get_name()}: no combined split among host dims "
        f"{host_dims} makes all spans fit within "
        f"{MAX_SPAN_BYTES / (1024**2):.0f} MB after trying at most "
        f"{_MAX_TILE_COMBOS} combinations."
    )


def _has_indirect_reads(op: ComputedBuffer) -> bool:
    """Return True if the op uses indirect/gather-style input reads.

    Automatic span-overflow tiling currently handles direct MemoryDep address
    math only.  Indirect-access ops use their own SDSC/IDA lowering path and are
    skipped here.
    """
    try:
        _, _, indirect_sizes = indirect_info_from_op(op)
    except (AttributeError, RuntimeError, TypeError, Unsupported):
        return False
    return indirect_sizes is not None


def can_conform_pointwise_tile(
    op: ComputedBuffer,
    split_by_host_dim: dict[int, int],
    max_cores: int,
) -> bool:
    """Return True if op can safely reuse an already-planned group's tile split.

    Used to fuse a directly-connected Pointwise consumer into an upstream
    Pointwise chain's shared coarse-tile loop even when the consumer's own
    independent cost search (``plan_span_overflow_tile``) would have picked a
    different split: rather than re-run that search, check whether the
    upstream split is *also* a legal and sufficient plan for ``op``, so every
    op in the chain iterates in lockstep inside one loop.

    Legal means: every ``host_dim`` exists and its range divides evenly by
    the given split, and no output or input stick boundary is cut. Sufficient
    means: after applying the split, ``op`` itself has no remaining
    output/input span overflow — the upstream split must fully cover this
    op's own span pressure, not just partially help it.
    """
    if not (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Pointwise)
        and isinstance(op.layout, FixedTiledLayout)
        and not getattr(op, "dim_hints", [])
    ):
        return False
    if not _layout_has_static_span_metadata(op.layout):
        return False
    if _has_indirect_reads(op):
        return False

    ranges = list(op.data.ranges)
    for host_dim, split_count in split_by_host_dim.items():
        if host_dim >= len(ranges):
            return False
        try:
            full_size = int(ranges[host_dim])
        except (TypeError, ValueError):
            return False
        if full_size <= 1 or full_size % split_count != 0:
            return False

    if (
        _combined_tile_stick_alignment_error(op, op.layout, split_by_host_dim)
        is not None
    ):
        return False

    try:
        remaining = _remaining_span_candidates_after_tile(
            op, max_cores, split_by_host_dim
        )
    except Unsupported:
        return False
    return not remaining


def plan_span_overflow_tile(
    op: ComputedBuffer,
    max_cores: int,
) -> SpanOverflowTilePlan | None:
    """Return an automatic output-range coarse-tile plan for supported ops.

    Supported inputs are static ``FixedTiledLayout`` ``ComputedBuffer`` ops whose
    data is ``Pointwise`` or ``Reduction``.  The planner skips non-computed ops,
    flexible/dynamic layouts, scalar reductions with no output ranges, and
    indirect-access ops because those are handled outside this automatic
    span-overflow coarse-tiling path.

    The returned plan contains one or more output-range tile levels plus the
    physical span facts that caused them.  ``None`` means this op either is not
    eligible for this pass or has no output/input span that needs coarse tiling.

    ``max_cores`` is threaded through this function and everything it calls,
    but it has no effect on the emitted split today: every candidate's
    ``core_split_estimate`` is hardcoded to 1, so span math always assumes
    work division gives no help and ``max_cores``/``SENCORES`` is never
    consulted. Intentional for now -- the cost search doesn't yet model work
    division (see the "Work Division is not yet modeled" limitation).
    ``max_cores`` stays in the signature so that work can wire in without
    changing every caller.

    TODO: make a common planner for Work Division and Working Set Reduction
    together, so this pass can get a proper core_split_estimate instead of
    the hardcoded 1.
    """
    logger.debug(
        "[span-overflow planner] enter op=%s data=%s max_cores=%s",
        getattr(op, "get_name", lambda: "<unknown>")(),
        type(getattr(op, "data", None)).__name__,
        max_cores,
    )
    if not (
        isinstance(op, ComputedBuffer)
        and isinstance(op.layout, FixedTiledLayout)
        and isinstance(op.data, (Pointwise, Reduction))
    ):
        # TODO: decide whether MutationLayoutSHOULDREMOVE producers need
        # span-overflow planning, or whether they are safe to keep outside this
        # pass as copy-back/mutation intermediates.
        logger.debug("[span-overflow planner] skip unsupported op/type")
        return None

    # Span planning requires concrete layout/device metadata so it can rebuild
    # post-tile Spyre layouts and validate physical spans exactly.
    if not _layout_has_static_span_metadata(op.layout):
        logger.debug(
            "[span-overflow planner] skip op=%s reason=non_static_layout",
            op.get_name(),
        )
        return None

    logger.debug(
        "[span-overflow planner] op=%s ranges=%s reduction_ranges=%s layout_size=%s device_size=%s",
        op.get_name(),
        list(getattr(op.data, "ranges", [])),
        list(getattr(op.data, "reduction_ranges", [])),
        list(op.layout.size),
        list(op.layout.device_layout.device_size),
    )

    # Indirect-access ops are supported by separate lowering paths; automatic
    # span-overflow tiling only handles direct MemoryDep address math.
    if _has_indirect_reads(op):
        logger.debug(
            "[span-overflow planner] skip op=%s reason=indirect_reads",
            op.get_name(),
        )
        return None

    if isinstance(op.data, Pointwise):
        # Pointwise ops only need output-span analysis here.
        candidates = _output_span_candidates_from_op(op, op_name=op.get_name())
        logger.debug(
            "[span-overflow planner] op=%s pointwise_output_candidates=%d",
            op.get_name(),
            len(candidates),
        )
        return _search_min_cost_tile_plan(op, max_cores, candidates)

    if isinstance(op.data, Reduction):
        # Scalar/full reductions have no output range to coarse-tile.
        # Non-scalar reductions combine output-span candidates with input spans
        # controlled by output symbols; reduction-only input spans are
        # intentionally skipped until reduction-range tiling exists.
        if not list(op.data.ranges):
            logger.debug(
                "[span-overflow planner] skip op=%s reason=scalar_reduction",
                op.get_name(),
            )
            return None
        output_candidates = _output_span_candidates_from_op(op, op_name=op.get_name())
        input_candidates = _input_span_candidates(op, max_cores)
        candidates = output_candidates + input_candidates
        logger.debug(
            "[span-overflow planner] op=%s reduction_output_candidates=%d input_candidates=%d",
            op.get_name(),
            len(output_candidates),
            len(input_candidates),
        )
        return _search_min_cost_tile_plan(op, max_cores, candidates)

    return None
