# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""OpSpec-reading helpers.

The KTIR emitter consumes the *finished* ``OpSpec`` list and must agree on the
"decision / arithmetic" it implies: buffer identity, row-major strides,
iteration-space grouping, and reshape / broadcast alignment of pointwise
operands.  Those computations are **pure** (sympy / int over the ``op_specs``)
-- no backend emission primitives, no MLIR builder, no live Inductor kernel
state -- so they live here as plain functions rather than as base-class methods.

They are therefore **emitter-agnostic**: they read the ``OpSpec`` contract
without emitting any target IR, so one set can serve every OpSpec consumer.  The
KTIR emitter is simply the first such consumer (hence the KTIR references
below); future emitters are intended to share them, which is why the module is
named for the ``OpSpec`` it reads rather than for KTIR.

``__all__`` is the contract: those names are what a consumer may import.  Anything
underscore-prefixed is a step inside one of them -- reachable from a test, but not
something to build on.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from torch_spyre._inductor.op_spec import OpSpec, TensorArg

__all__ = [
    "PARALLEL",
    "REDUCTION",
    "align_reshape_plan",
    "buf_id",
    "core_divisions",
    "per_core_extent",
    "placeholder_axes",
    "reduction_indexing",
    "relayout_strides",
    "row_major_strides",
]


def row_major_strides(device_size: Sequence[int]) -> list[int]:
    """Row-major (C-contiguous) strides for a device-size list."""
    n = len(device_size)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * int(device_size[i + 1])
    return strides


def buf_id(arg: TensorArg) -> str:
    """Stable identity of the buffer an op arg refers to, for register threading.

    Keys on the op-spec ``name`` (the buffer name): unique per buffer and
    identical whether the buffer appears as an input or an output, so a
    fused-away intermediate threads its register value without aliasing.
    ``arg_index`` cannot serve as the identity -- distinct fused-away
    intermediates all carry the unassigned sentinel ``-1``.

    ``name`` must therefore be populated on every projected op arg (see
    ``create_tensor_arg``).  A ``None`` name means an unnamed arg reached
    projection, which would silently alias on ``-1``; raise loudly instead of
    falling back.
    """
    if arg.name is None:
        # ValueError (not NotImplementedError): under TORCH_SPYRE_KTIR=1
        # create_tensor_arg always populates the name, so a None here is a
        # broken internal invariant, not an unsupported-capability case.
        raise ValueError(
            "buf_id: TensorArg.name is None -- every projected op arg must "
            "carry a buffer name for register-threading identity (arg_index is "
            "-1 for fused intermediates and cannot disambiguate them)"
        )
    return arg.name


def _iteration_space_key(spec: OpSpec) -> tuple:
    """Hashable canonical form of ``spec.iteration_space`` for grouping.

    Two ops fuse iff this key matches: same symbols, same ranges, and same
    work divisions.  Symbols/ranges are compared by their string form so the
    key is order-independent and hashable.
    """
    return tuple(
        sorted(
            (str(sym), str(rng), int(div))
            for sym, (rng, div) in spec.iteration_space.items()
        )
    )


# Device-dim coordinate kinds, used to align pointwise operands whose device
# axes are ordered differently from the op's output tile (broadcast alignment).
_DIM_CONST = "const"  # no iteration-space symbol (an inserted / broadcast dim)
_DIM_BARE = "bare"  # coord == sym (a non-stick dim)
_DIM_WITHIN_STICK = "within_stick"  # coord == sym % stick  (within-stick lanes)
_DIM_OUTER_STICK = "outer_stick"  # coord == sym // stick (outer-stick chunks)


def _dim_info(coord: sympy.Expr) -> tuple[str, sympy.Symbol | None]:
    """Classify a device-dim coordinate as ``(kind, sym)``.

    Two device axes are the "same" logical dim iff they share a ``(kind, sym)``
    -- e.g. a weight's ``sym // stick`` outer-stick axis matches an output's
    ``sym // stick`` axis even when they sit at different positions.  A ``const``
    dim (extent 1, no symbol) carries no data and is dropped / broadcast.

    A coordinate carrying more than one iteration symbol (a single physical axis
    that folds two logical dims, e.g. ``a*8 + b``) is not a simple device axis.
    No legal reshape produces one today -- a plain reshape is a pure view whose
    strides stay stick-aligned (single symbol per axis), and a within-stick fold
    is rejected earlier at layout selection (the within-stick dim must be a full
    64-element stick).  Raise loudly rather than carrying a dead classification:
    if this fires, a new frontend construct reached alignment and both this
    helper and ``align_reshape_plan`` need real multi-symbol support.
    """
    syms = coord.free_symbols
    if not syms:
        return (_DIM_CONST, None)
    if len(syms) > 1:
        raise NotImplementedError(
            f"OpSpec alignment: device coordinate {coord!r} folds multiple "
            f"iteration symbols {syms} into one physical axis; no supported "
            "reshape produces this, so multi-symbol alignment is not implemented"
        )
    sym = next(iter(syms))
    if isinstance(coord, sympy.Symbol):
        return (_DIM_BARE, sym)
    if isinstance(coord, (sympy.Mod, ModularIndexing)):
        return (_DIM_WITHIN_STICK, sym)
    # Both spellings of the outer-stick index: torch's FloorDiv, and the plain
    # sympy ``floor(c/64)`` the projection actually produces -- the pointwise path
    # never noticed the difference because identical operand and output
    # coordinates take the fast path out of ``align_reshape_plan`` before any
    # classification happens.
    if isinstance(coord, (FloorDiv, sympy.floor)):
        return (_DIM_OUTER_STICK, sym)
    # A single-symbol coordinate that is none of the known device-axis forms
    # (e.g. ``2*d0 + 1``) is not a plain outer-stick chunk index.  Raise loudly
    # rather than silently classifying it as outer-stick -- same policy as the
    # multi-symbol case above; a later increment that legitimately produces such
    # a coordinate must extend this classifier explicitly.
    raise NotImplementedError(
        f"OpSpec alignment: single-symbol device coordinate {coord!r} is not a "
        "bare symbol, within-stick (Mod/ModularIndexing), or outer-stick "
        "(FloorDiv) form; classification is not implemented"
    )


def align_reshape_plan(
    in_coords: list[sympy.Expr],
    in_block: list[int],
    out_coords: list[sympy.Expr],
    out_block: list[int],
) -> tuple[list[int], list[int] | None] | None:
    """Plan to reshape + broadcast a pointwise operand tile into the op's output
    device-axis order.

    A pointwise op's operands may carry their device axes in a different order
    (and rank) from the output tile: a per-row reduction result broadcasts over
    the outer-stick dim, a channel weight broadcasts over the row dim, etc.
    Elementwise broadcast typically only auto-aligns *leading* unit dims, so a
    misaligned operand (outer-stick where the output has rows) must be reshaped
    to the output order first.

    Each output device axis is matched to an input axis by ``(kind, sym)`` (see
    ``_dim_info``); the within-stick axis maps last -> last.  Unmatched output
    axes get extent 1 (to be broadcast).  Returns ``(reshape_to, broadcast_to)``
    -- reshape the operand to ``reshape_to`` (skip if it already equals
    ``in_block``) then broadcast to ``broadcast_to`` (``None`` to skip).
    Returns ``None`` when the operand already matches the output (fast path:
    no reshape / broadcast emitted, keeping simple kernels byte-identical).

    Raises ``NotImplementedError`` for a cross-stick transpose (an input axis
    with extent > 1 that no output axis matches, or matched axes that would need
    a permute) -- that needs a restickify, not a reshape.
    """
    in_block = [int(b) for b in in_block]
    out_block = [int(b) for b in out_block]
    if list(in_coords) == list(out_coords) and in_block == out_block:
        return None

    in_info = [_dim_info(c) for c in in_coords]
    in_rank = len(in_coords)
    out_rank = len(out_coords)

    reshape_to = [1] * out_rank
    used: set[int] = set()
    # Within-stick lanes: the last input axis always maps to the last output axis.
    reshape_to[out_rank - 1] = in_block[in_rank - 1]
    used.add(in_rank - 1)
    matched_seq: list[int] = []  # input axes matched to output axes, in out order
    for o in range(out_rank - 1):
        okind, osym = _dim_info(out_coords[o])
        if okind == _DIM_CONST:
            continue
        for a in range(in_rank - 1):
            if a in used:
                continue
            if in_info[a] == (okind, osym):
                used.add(a)
                reshape_to[o] = in_block[a]
                matched_seq.append(a)
                break

    # A pure reshape preserves the row-major element order: the matched input
    # axes must appear in increasing order.  If not, the operand would need a
    # transpose (restickify) to align -- not supported on this path.
    if any(matched_seq[i] >= matched_seq[i + 1] for i in range(len(matched_seq) - 1)):
        raise NotImplementedError(
            "OpSpec alignment: pointwise operand needs a transpose (restickify) "
            "to align device axes; not supported yet"
        )
    prod_in = 1
    for b in in_block:
        prod_in *= b
    prod_reshape = 1
    for b in reshape_to:
        prod_reshape *= b
    if prod_in != prod_reshape:
        # An input axis with extent > 1 was dropped -> real data would be lost;
        # aligning it needs a cross-stick transpose (restickify).
        raise NotImplementedError(
            "OpSpec alignment: pointwise operand needs a cross-stick transpose "
            "(restickify) to align device axes; not supported yet"
        )
    broadcast_to = out_block if reshape_to != out_block else None
    return (reshape_to, broadcast_to)


def operand_indexing(
    in_coords: Sequence[sympy.Expr],
    in_extent: Sequence[int],
    out_coords: Sequence[sympy.Expr],
    out_extent: Sequence[int],
) -> tuple[int | None, ...]:
    """One pointwise operand's map row against the output's axes.

    The parallel counterpart of ``reduction_indexing``, and the same shape of
    answer: one entry per *operand* axis, being the axis of the OUTPUT that axis
    walks -- so the row is a projection of the output's iteration nest, which is
    what an all-parallel ``linalg.generic``'s ``indexing_maps`` states.  The
    iteration dims are the output's own axes, in order, because a pointwise op
    writes every element of its output exactly once.

    ``None`` is an axis the operand does not walk at all: its coordinate carries
    no iteration symbol and its extent is one element, so every iteration reads
    that one element and the map's result position is the CONSTANT 0.  That is the
    statistic read (``(d0,d1,d2) -> (d1, 0)``), the splat (``(d0,d1) -> (d0, 0)``)
    and the row broadcast (``(d0,d1,d2) -> (d0, 0, d2)``) -- three forms, one rule.

    Axes are matched by ``_dim_info`` classification, not by expression equality,
    so an outer-stick ``sym // 64`` matches an outer-stick ``sym // 64`` whatever
    it is spelled as, exactly as ``align_reshape_plan`` matches them.

    Refuses, rather than guessing, three shapes that would otherwise read the
    wrong elements silently:

    * an operand axis matching no output axis -- there is no dim for it to walk;
    * a matched axis whose extent differs from the output's -- that is a stretch
      or a shrink, and a map cannot say either;
    * matched axes out of increasing order -- a permutation, which needs a
      transpose (restickify) rather than a map, the same refusal
      ``reduction_indexing`` makes.

    ``in_extent``/``out_extent`` are the TILE extents, not the buffer's: a
    ``linalg`` operand's shape is the tile that was loaded, and a broadcast
    operand is loaded at one element on the axis it does not walk.
    """
    out_info = [_dim_info(coord) for coord in out_coords]
    row: list[int | None] = []
    matched: list[int] = []
    used: set[int] = set()
    for axis, coord in enumerate(in_coords):
        info = _dim_info(coord)
        size = int(in_extent[axis])
        if info == (_DIM_CONST, None):
            if size != 1:
                raise NotImplementedError(
                    f"OpSpec alignment: operand device axis {axis} is a constant "
                    f"coordinate over {size} elements, so it walks no iteration "
                    "dim and yet carries more than the one element a constant map "
                    "position reads"
                )
            row.append(None)
            continue
        candidates = [o for o in range(len(out_coords)) if o not in used]
        found = next((o for o in candidates if out_info[o] == info), None)
        if found is None:
            raise NotImplementedError(
                f"OpSpec alignment: operand device axis {axis} ({coord!r}) matches "
                "no output device axis, so there is no iteration dim for it to "
                "walk; aligning it needs a transpose (restickify)"
            )
        if int(out_extent[found]) != size:
            raise NotImplementedError(
                f"OpSpec alignment: operand device axis {axis} runs {size} "
                f"elements against the output axis {found}'s "
                f"{int(out_extent[found])}; a broadcast map reads one element or "
                "the whole axis, not a stretch of it"
            )
        used.add(found)
        matched.append(found)
        row.append(found)
    if any(matched[i] >= matched[i + 1] for i in range(len(matched) - 1)):
        raise NotImplementedError(
            "OpSpec alignment: the operand's device axes are permuted against the "
            "output's, which needs a transpose (restickify) rather than a map"
        )
    return tuple(row)


def _stick_term(coord: sympy.Expr) -> tuple[str, sympy.Symbol | None, int]:
    """``(kind, sym, stick)`` for one device coordinate of a relayout.

    ``_dim_info`` with the divisor kept: a relayout has to know HOW MANY lanes a
    stick holds to split a symbol into its stick and lane parts, which alignment
    never needed because it only matches axes of one kind against each other.
    ``stick`` is 0 for the two kinds that carry none.

    Only the bare symbol is accepted inside the split -- ``floor(s / 64)``, not
    ``floor(2 * s / 64)`` -- because a scaled symbol is a stride, and a stride
    inside a floor is not a stick split.
    """
    kind, sym = _dim_info(coord)
    if kind in (_DIM_CONST, _DIM_BARE):
        return kind, sym, 0
    if isinstance(coord, ModularIndexing):
        inner, div, stick = coord.args
        if div != 1:
            raise NotImplementedError(
                f"OpSpec relayout: coordinate {coord!r} divides before it takes the "
                "lane, which is not a stick split"
            )
    elif isinstance(coord, (sympy.Mod, FloorDiv)):
        inner, stick = coord.args
    else:  # ``sympy.floor(sym / stick)``, the projection's own spelling
        quotient = coord.args[0]
        stick = 1 / quotient.coeff(sym)
        inner = sympy.expand(quotient * stick)
    if inner != sym or not getattr(stick, "is_Integer", False) or int(stick) < 2:
        raise NotImplementedError(
            f"OpSpec relayout: coordinate {coord!r} is not a stick split of a bare "
            "symbol by a whole number of lanes"
        )
    return kind, sym, int(stick)


def relayout_strides(
    in_coords: Sequence[sympy.Expr],
    in_size: Sequence[int],
    out_coords: Sequence[sympy.Expr],
    out_size: Sequence[int],
    ranges: dict[sympy.Symbol, int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``(extent, in_strides, out_strides)``: a copy, in the output's axis order.

    A relayout -- a transpose, across sticks or not -- moves every element of
    one buffer to its place in another, and the two buffers order their device
    axes differently.  Both are viewed over ONE iteration space, the output's
    device axes in the output's order, so the output view is its own row-major
    layout and the copy between the views is the identity.  Everything the
    transpose is goes into the INPUT view's strides: for each iteration dim,
    how far the input's address moves when that dim steps by one.  That is the
    only form the backend lowers -- an input view at arbitrary strides into a
    row-major output -- so the output's strides are row-major by construction
    and asserted so.

    The iteration space is the output's axes REFINED: an output axis that walks a
    whole symbol the input splits into stick and lane (``s // 64``, ``s % 64``)
    becomes two dims, ``(range / 64, 64)``, because the input's address is not
    linear in ``s`` but is in its two parts.  Splitting a row-major axis in two
    keeps it row-major, so the output side is unaffected.  An axis of at most one
    stick needs no split: its stick index is 0 and its lane is the symbol.

    ``ranges`` is each symbol's iteration count.  It is compared against the
    extents to refuse padding, the one thing a stride cannot say: a stick split
    of a symbol whose range is not a whole number of sticks reads lanes the input
    holds no elements for.

    Refuses rather than guesses: a symbol the output does not walk exactly once
    (or as one stick/lane pair), an input axis whose symbol the output does not
    walk, an extent that disagrees with its symbol's range, and a view that reads
    one input element for two outputs -- each of which is not a copy.
    """
    in_size = [int(s) for s in in_size]
    out_size = [int(s) for s in out_size]
    if len(in_coords) != len(in_size) or len(out_coords) != len(out_size):
        raise NotImplementedError(
            "OpSpec relayout: a coordinate list and its device size differ in rank"
        )

    def rng(sym: sympy.Symbol) -> int:
        if sym not in ranges:
            raise NotImplementedError(
                f"OpSpec relayout: symbol {sym} has no iteration range"
            )
        return int(ranges[sym])

    # Each output axis, read once: what it walks and over how many elements.
    out_terms = [_stick_term(coord) for coord in out_coords]
    in_terms = [_stick_term(coord) for coord in in_coords]
    for side_terms, side_size, side in (
        (out_terms, out_size, "output"),
        (in_terms, in_size, "input"),
    ):
        for axis, (kind, _sym, _stick) in enumerate(side_terms):
            if kind == _DIM_CONST and side_size[axis] != 1:
                raise NotImplementedError(
                    f"OpSpec relayout: {side} device axis {axis} is a constant "
                    f"coordinate over {side_size[axis]} elements"
                )
    # Which input symbols are split, and at what stick: those are the output
    # axes that may need refining.
    split_by_input: dict[sympy.Symbol, int] = {}
    for kind, sym, stick in in_terms:
        if kind in (_DIM_WITHIN_STICK, _DIM_OUTER_STICK):
            if split_by_input.setdefault(sym, stick) != stick:
                raise NotImplementedError(
                    f"OpSpec relayout: the input splits {sym} at two stick sizes"
                )

    # The iteration dims, and each symbol as a linear sum over them.
    extent: list[int] = []
    out_strides: list[int] = []
    form: dict[sympy.Symbol, list[tuple[int, int]]] = {}
    out_row_major = row_major_strides(out_size)
    # A stick axis seen without its lane axis yet: sym -> (stick, sticks).
    pending_lane: dict[sympy.Symbol, tuple[int, int]] = {}
    for axis, (kind, sym, stick) in enumerate(out_terms):
        stride = out_row_major[axis]
        if kind == _DIM_CONST:
            extent.append(1)
            out_strides.append(stride)
            continue
        if sym in form and kind != _DIM_WITHIN_STICK:
            raise NotImplementedError(
                f"OpSpec relayout: the output walks {sym} on more than one axis"
            )
        if kind == _DIM_BARE:
            size = out_size[axis]
            if size != rng(sym):
                raise NotImplementedError(
                    f"OpSpec relayout: output axis {axis} runs {size} elements of "
                    f"{sym}, whose range is {rng(sym)}"
                )
            in_stick = split_by_input.get(sym, 0)
            if in_stick and size > in_stick:
                if size % in_stick:
                    raise NotImplementedError(
                        f"OpSpec relayout: {sym} runs {size} elements, not a whole "
                        f"number of {in_stick}-lane sticks, so splitting it reads "
                        "padding lanes"
                    )
                hi = len(extent)
                extent += [size // in_stick, in_stick]
                out_strides += [stride * in_stick, stride]
                form[sym] = [(hi, in_stick), (hi + 1, 1)]
            else:
                form[sym] = [(len(extent), 1)]
                extent.append(size)
                out_strides.append(stride)
            continue
        dim = len(extent)
        extent.append(out_size[axis])
        out_strides.append(stride)
        if kind == _DIM_OUTER_STICK:
            if sym in form:
                raise NotImplementedError(
                    f"OpSpec relayout: the output walks {sym}'s stick twice"
                )
            form[sym] = [(dim, stick)]
            pending_lane[sym] = (stick, out_size[axis])
        else:
            if sym not in pending_lane or pending_lane[sym][0] != stick:
                raise NotImplementedError(
                    f"OpSpec relayout: output lane axis {axis} of {sym} has no "
                    "stick axis before it at the same stick size"
                )
            sticks = pending_lane[sym][1]
            if out_size[axis] != stick or sticks * stick != rng(sym):
                raise NotImplementedError(
                    f"OpSpec relayout: output splits {sym} (range {rng(sym)}) into "
                    f"{sticks} x {out_size[axis]}, which is padded or partial"
                )
            form[sym].append((dim, 1))
            del pending_lane[sym]
    if pending_lane:
        raise NotImplementedError(
            f"OpSpec relayout: output stick axes {sorted(map(str, pending_lane))} "
            "have no lane axis"
        )
    assert tuple(out_strides) == tuple(row_major_strides(extent)), (
        "a refined output layout must stay row major"
    )

    # The input's strides over those dims: each input axis contributes its own
    # row-major stride times how its coordinate moves with each dim.
    in_row_major = row_major_strides(in_size)
    in_strides = [0] * len(extent)
    for axis, (kind, sym, stick) in enumerate(in_terms):
        if kind == _DIM_CONST:
            continue
        if sym not in form:
            raise NotImplementedError(
                f"OpSpec relayout: input axis {axis} walks {sym}, which the output "
                "does not; that is a reduction or a broadcast, not a copy"
            )
        terms = form[sym]
        stride = in_row_major[axis]
        if kind == _DIM_BARE:
            expected = rng(sym)
            for dim, coeff in terms:
                in_strides[dim] += stride * coeff
        else:
            # The input splits ``sym``.  Either the iteration space splits it the
            # same way (two dims, the lane one ``stick`` long), or ``sym`` is at
            # most one stick long and is its own lane.
            split = (
                len(terms) == 2
                and terms[0][1] == stick
                and extent[terms[1][0]] == stick
            )
            if not split and not (len(terms) == 1 and extent[terms[0][0]] <= stick):
                raise NotImplementedError(
                    f"OpSpec relayout: input axis {axis} splits {sym} at {stick} "
                    "lanes, which the iteration space does not"
                )
            sticks = -(-rng(sym) // stick)
            if kind == _DIM_OUTER_STICK:
                expected = sticks
                if split:
                    in_strides[terms[0][0]] += stride
            else:
                expected = stick
                if sticks * stick != rng(sym):
                    raise NotImplementedError(
                        f"OpSpec relayout: input lane axis {axis} of {sym} (range "
                        f"{rng(sym)}) is padded"
                    )
                in_strides[(terms[1] if split else terms[0])[0]] += stride
        if in_size[axis] != expected:
            raise NotImplementedError(
                f"OpSpec relayout: input axis {axis} holds {in_size[axis]} elements "
                f"where {sym} needs {expected}"
            )
    for dim, (size, stride) in enumerate(zip(extent, in_strides)):
        if size > 1 and stride == 0:
            raise NotImplementedError(
                f"OpSpec relayout: iteration dim {dim} moves no input element, so "
                "one input element would be copied to many outputs"
            )
    return tuple(extent), tuple(in_strides), tuple(out_strides)


def placeholder_axes(
    coords: Sequence[sympy.Expr], extent: Sequence[int]
) -> tuple[int, ...]:
    """Output axes standing in for something the op does not write.

    A projection keeps an axis the op does not produce in the output's
    ``device_size`` as a unit extent at a constant coordinate, so the output is
    the same rank as the input even though it carries less.  A consumer wants
    those axes gone: the accepted KTIR form stores a rank-2 tile into a rank-2
    view, and keeping them would demand a reshape between the compute op and the
    store.

    **Unary on purpose** -- it asks nothing about the inputs.  A constant
    coordinate alone does not identify a placeholder: an on-stick reduction's
    output carries *two* constant coordinates, one a placeholder and one the
    broadcast lane the store really walks, and only the extent separates them.
    Folding this into a relational helper is what made the on-stick shape reach a
    coordinate-matching refusal before anyone could ask the unary question.
    """
    return tuple(
        axis
        for axis, coord in enumerate(coords)
        if _dim_info(coord)[0] == _DIM_CONST and int(extent[axis]) == 1
    )


# The two ``linalg`` iterator names, as ``iterator_types`` spells them.  Produced
# here because ``reduction_indexing`` is what decides which dim is which; a
# consumer that builds a ``linalg.generic`` passes them through unchanged.
PARALLEL = "parallel"
REDUCTION = "reduction"


def reduction_indexing(
    in_coords: Sequence[sympy.Expr],
    in_extent: Sequence[int],
    out_coords: Sequence[sympy.Expr],
    out_extent: Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """``(iters, in_map, out_map)`` for one unary reduction over a squeezed output.

    The iteration nest a reduction wants, stated as the three things a
    ``linalg.generic`` would have to say and a ``linalg.reduce`` says implicitly:
    one iterator per iteration dim, and per operand one dim index per result
    position (a *projection*, so ``(0, 1, 2)`` is ``(d0..d3) -> (d0,d1,d2)``).
    Plain tuples rather than dialect types, so this module keeps knowing nothing
    about MLIR.

    An iteration dim is a distinct ``(kind, sym)`` classification of a device
    coordinate (see ``_dim_info``), numbered in the order the *input* axes
    introduce them.  Two things follow that a set-complement over kept axes
    cannot express:

    * an output axis whose coordinate is a bare constant of extent > 1 is not a
      placeholder but a **broadcast lane** -- a real iteration dim the store walks
      that the input does not have -- so it gets a fresh dim rather than a
      refusal.  That is the whole on-stick reduction, and the axis a reduction
      reduces on the way in can therefore be kept on the way out.
    * an axis can be both, which is why the answer is per-operand maps and not a
      flat list of reduced axes.

    ``out_coords``/``out_extent`` must already have their placeholder axes dropped
    (``placeholder_axes``): an extent-1 constant would otherwise be
    indistinguishable from a degenerate broadcast lane.

    Refuses three shapes, each of which would otherwise read the wrong elements
    silently: a kept axis that matches no input axis or matches one out of
    increasing order (that is a transpose, which needs a restickify); a kept axis
    whose extent changed; and a nest in which nothing reduces at all.  The last
    is not implied by the caller having labelled the spec a reduction -- that says
    what was *asked for*, this reads what the coordinates *are* -- and without it
    an all-parallel nest reaches ``dimensions = []``, which does not build.
    """
    dims: list[tuple[str, sympy.Symbol | None]] = []
    extents: list[int] = []
    in_map: list[int] = []
    for axis, coord in enumerate(in_coords):
        info = _dim_info(coord)
        size = int(in_extent[axis])
        if info in dims:
            # A repeated classification is one dim walked twice, so the two axes
            # must agree about how far it runs.
            dim = dims.index(info)
            if extents[dim] != size:
                raise NotImplementedError(
                    f"OpSpec reduction: input device axes {in_map.index(dim)} and "
                    f"{axis} are both {coord!r} but run {extents[dim]} and {size} "
                    "elements; one iteration dim cannot have two ranges"
                )
        else:
            dims.append(info)
            extents.append(size)
            dim = len(dims) - 1
        in_map.append(dim)

    # Only the dims the input introduced are matchable; anything past this is a
    # broadcast lane allocated below, and a second one must not match the first.
    from_input = len(dims)
    out_map: list[int] = []
    matched: list[int] = []  # the input dims kept, in output order
    for axis, coord in enumerate(out_coords):
        info = _dim_info(coord)
        size = int(out_extent[axis])
        if info in dims[:from_input]:
            dim = dims.index(info)
            if extents[dim] != size:
                raise NotImplementedError(
                    f"OpSpec reduction: kept output axis {axis} has extent {size} "
                    f"but its input axis has extent {extents[dim]}; a reduction "
                    "does not resize an axis it keeps"
                )
            matched.append(dim)
        elif info == (_DIM_CONST, None):
            # The broadcast lane: a real iteration dim with no input axis behind
            # it.  ``placeholder_axes`` has already taken the extent-1 constants,
            # so what is left here carries elements.
            dims.append(info)
            extents.append(size)
            dim = len(dims) - 1
        else:
            raise NotImplementedError(
                f"OpSpec reduction: output device axis {axis} ({coord!r}) matches "
                "no input axis; the surviving axes are permuted, which needs a "
                "transpose rather than a reduction"
            )
        out_map.append(dim)

    # A pure reduction preserves the row-major order of the axes it keeps.
    if any(matched[i] >= matched[i + 1] for i in range(len(matched) - 1)):
        raise NotImplementedError(
            "OpSpec reduction: the surviving axes are permuted, which needs a "
            "transpose rather than a reduction"
        )
    reduced = set(range(len(dims))) - set(out_map)
    if not reduced:
        raise NotImplementedError(
            "OpSpec reduction: the output carries every input device axis, so "
            "there is no axis to reduce"
        )
    iters = tuple(REDUCTION if dim in reduced else PARALLEL for dim in range(len(dims)))
    return iters, tuple(in_map), tuple(out_map)


# ---------------------------------------------------------------------------
# Work division: the core grid, as the iteration space states it
# ---------------------------------------------------------------------------


def core_divisions(
    iteration_space: dict[sympy.Symbol, tuple[sympy.Expr, int]],
) -> tuple[list[tuple[sympy.Symbol, int, int]], int]:
    """``(divisions, total_cores)`` for one iteration space.

    ``OpSpec.iteration_space`` maps each symbol to ``(range, work_division)``,
    where the division is the number of cores that symbol's range is split
    across -- decided upstream by ``work_division.py`` from ``config.sencores``.
    The core grid is one flat index in ``[0, total_cores)``, read as a mixed-radix
    number over the divided symbols, so each division is returned as
    ``(sym, div, inner)`` **innermost-first**: that symbol's portion of the grid
    is ``(core_id // inner) % div``.

    ``total_cores`` is the product of the divisions, so an undivided space gives
    ``([], 1)`` -- a single-core kernel, stated by the contract rather than
    assumed.
    """
    split = [
        (sym, int(div)) for sym, (_range, div) in iteration_space.items() if div > 1
    ]
    total_cores = 1
    for _sym, div in split:
        total_cores *= div
    divisions: list[tuple[sympy.Symbol, int, int]] = []
    inner = 1
    for sym, div in reversed(split):  # innermost first
        divisions.append((sym, div, inner))
        inner *= div
    return divisions, total_cores


def per_core_extent(
    arg: TensorArg, divisors: dict[sympy.Symbol, int]
) -> tuple[list[int], list[sympy.Symbol | None]]:
    """``(extent, symbol)`` per device axis: one core's share, and what divides it.

    Each device axis carries at most one iteration symbol -- a bare ``c_i`` or an
    outer-stick ``c_i // stick`` -- so at most one divisor applies to it, and the
    axis' per-core extent is its full extent divided by that divisor.  The
    trailing axis is the within-stick one and is never divided: a stick is the
    unit of transfer, so splitting it across cores would split a stick.

    The returned symbol list says which division each axis follows (``None`` for
    an axis no division touches), which is what turns a division into a step
    along that axis.
    """
    extent = [int(s) for s in arg.device_size]
    coords = list(arg.device_coordinates)
    if divisors and len(coords) != len(extent):
        raise NotImplementedError(
            f"OpSpec work division: {arg.name!r} carries {len(coords)} device "
            f"coordinate(s) for {len(extent)} device axes, so which axis a "
            "divided symbol walks cannot be told"
        )
    per_core: list[int] = []
    symbols: list[sympy.Symbol | None] = []
    last = len(extent) - 1
    for axis, size in enumerate(extent):
        symbol = None
        if divisors and axis != last:
            kind, symbol = _dim_info(coords[axis])
            if kind == _DIM_CONST:
                symbol = None
        divisor = divisors.get(symbol, 1) if symbol is not None else 1
        if divisor > 1 and size % divisor:
            raise NotImplementedError(
                f"OpSpec work division: device axis {axis} of {arg.name!r} has "
                f"extent {size}, which {divisor} cores do not divide evenly; a "
                "ragged per-core tile is not supported"
            )
        per_core.append(size // divisor)
        symbols.append(symbol if divisor > 1 else None)
    return per_core, symbols
