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

"""OpSpec -> KTIR emitter.

``generate_ktir`` is an OpSpec consumer: it consumes the finished
``list[OpSpec | LoopSpec]`` kernel contract (the same contract the SDSC bundle
emitter ``generate_bundle`` consumes) and emits **KTDP-dialect MLIR** directly.
The module is built with the ``mlir_ktdp`` Python builders, so the returned
``str(module)`` is canonical, verifier-checked MLIR that the golden snapshot
test consumes without drift.

It uses the OpSpec-reading helpers from ``opspec_utils`` to adapt the OpSpec
information to generate_ktir.

Base addresses are emitted either as func arguments or as baked
``arith.constant``s, selected by the ``bake_addresses`` option.  The baked
form is a temporary dataflow-scheduler#65 workaround, to be reverted when the
backend accepts symbolic addresses.

Structure
---------

``generate_ktir`` is three steps, in this order:

1. ``build_kernel_plan(specs)`` -- a **pure** recursive walk of the spec tree
   that runs every derivation, raises every ``NotImplementedError`` the emitter
   can raise, and returns a ``KernelPlan``: the grid, the buffers, and a tree of
   ``Step`` records for the body.  It imports nothing from ``mlir_ktdp``, so
   every rejection is reachable and testable where the dialect build is absent.
2. ``KtirBuilder.create(plan)`` -- the single ``mlir_ktdp`` import site; owns
   the context and the per-module state.
3. ``b.emit(plan.steps)`` -- a recursive walk of the step tree.  It reads no
   spec and derives nothing, so emission cannot refuse a request the plan
   accepted.

Adding an op is one ``RECIPES`` entry; giving an op it already has a second
spelling at some element format is one ``Arm`` inside that entry; adding an
emission *shape* is one method on ``KtirBuilder`` plus one ``Surface`` arm in
``compute``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.codegen.opspec_utils import (
    PARALLEL,
    REDUCTION,
    align_reshape_plan,
    buf_id,
    core_divisions,
    per_core_extent,
    placeholder_axes,
    reduction_indexing,
    row_major_strides,
)
from torch_spyre._inductor.constants import STAGGERED_EAS
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.pass_utils import coeff_through_floor

# The dialect handles: one module-level name each, None until _load_dialects()
# binds them.  Under TYPE_CHECKING they are the real imports, so `ir.Module` and
# `linalg.add` carry types; at runtime the block does not execute, so importing
# this module requires no dialect build.
if TYPE_CHECKING:
    from mlir_ktdp import ir
    from mlir_ktdp.dialects import arith, func, ktdp, linalg, scf, spyreop, tensor
else:
    ir = arith = func = ktdp = linalg = scf = spyreop = tensor = None


def _load_dialects() -> None:
    """Bind the dialect handles into this module, once.  The only import site."""
    global ir, arith, func, ktdp, linalg, scf, spyreop, tensor
    if ir is not None:
        return
    from mlir_ktdp import ir as _ir
    from mlir_ktdp.dialects import arith as _arith
    from mlir_ktdp.dialects import func as _func
    from mlir_ktdp.dialects import ktdp as _ktdp
    from mlir_ktdp.dialects import linalg as _linalg
    from mlir_ktdp.dialects import scf as _scf
    from mlir_ktdp.dialects import spyreop as _spyreop
    from mlir_ktdp.dialects import tensor as _tensor

    ir, arith, func, ktdp, linalg, scf, spyreop, tensor = (
        _ir,
        _arith,
        _func,
        _ktdp,
        _linalg,
        _scf,
        _spyreop,
        _tensor,
    )


def dialect_available() -> bool:
    """True when the bindings ``_load_dialects`` needs are importable."""
    try:
        _load_dialects()
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# What this emitter does not implement
# ---------------------------------------------------------------------------
#
# One helper, one exception type, and a *label* per capability.  The label is a
# stable token shared by the raise and its test, so grepping it finds both; the
# message says what is missing here, in this emitter, and what to pass instead
# when there is an alternative.
#
# A message never claims a consumer is the blocker, because a consumer's answer
# is not a property of this file: the same emitted text is accepted or rejected
# depending on which dbo-opt build and which device.mlir it meets (``verify.py``
# is where that is observed, against a real one).  A refusal here says what this
# emitter does not build.
#
# Derivations that reject a specific *value* of their own input -- an unsupported
# dtype, a tile advance that is not a lattice point of its view -- raise about
# that value at their own site and need no label: the input is wrong, not the
# emitter.


class Unimplemented(NotImplementedError):
    """A capability this emitter does not implement yet, named by its label."""


def _unimplemented(label: str, message: str) -> NoReturn:
    """Refuse a capability that is not implemented.  ``label`` joins raise+test."""
    raise Unimplemented(f"OpSpec->KTIR [{label}]: {message}")


# ---------------------------------------------------------------------------
# Records: what the derivations produce and the builders consume
# ---------------------------------------------------------------------------
#
# Every record is dialect-free -- ints, strings and sympy Exprs -- so the whole
# derivation layer is exercised (and unit-tested) without an ``mlir_ktdp`` build.
# The builders are the only code that turns a record into an ``ir`` object.


@dataclasses.dataclass(frozen=True)
class ElemTypes:
    """The two element types one buffer access involves.

    ``storage`` types the memref (the view), ``value`` types the tensor a load
    produces or a store consumes.  KTDP compares neither against the other --
    ``LoadOp``/``StoreOp`` verify shapes only -- so they are two fields rather
    than one, and today's derivation returns them equal.  Held as MLIR type
    *spellings* (``f16``, ``i32``), so the record stays dialect-free; the
    builder's ``named_type`` is what resolves a spelling against the imported
    ``ir``.  Spellings rather than builder names so that every element type is
    written the same way, whether or not its builder takes a width.

    ``NAMES`` is the supported-dtype table, and ``of`` the only way to get an
    ``ElemTypes`` from a device dtype -- so the names this record can hold, the
    dtypes that map to them, and the unsupported-dtype rejection are one place.
    The two fp16 device formats both map to ``f16``; extend ``NAMES`` (never fall
    through silently) as new dtypes are supported.
    """

    NAMES: ClassVar[dict[DataFormats, str]] = {
        DataFormats.IEEE_FP16: "f16",
        DataFormats.SEN169_FP16: "f16",  # for now treating dl169 as cosmetic
        DataFormats.IEEE_FP32: "f32",
        DataFormats.BFLOAT16: "bf16",
        DataFormats.IEEE_INT32: "i32",
    }

    storage: str
    value: str

    @classmethod
    def of(cls, dtype: DataFormats) -> ElemTypes:
        """The storage/value pair for a device dtype, or raise.

        One ``device_dtype`` means one type on both sides today; a load that
        reinterprets is why the record has two fields.
        """
        name = cls.NAMES.get(dtype)
        if name is None:
            raise NotImplementedError(
                f"OpSpec->KTIR: unsupported device dtype {dtype!r}"
            )
        return cls(storage=name, value=name)


@dataclasses.dataclass(frozen=True)
class Level:
    """One enclosing loop level, as the derivations see it.

    ``symbols`` is that level's entry of ``OpSpec.tiled_symbols`` (possibly
    empty: a level that does not tile this op) and ``trip`` is its trip count.
    No induction variable: levels are planned before any SSA value exists, and
    the builder supplies the variable of the loop it has open.
    """

    symbols: tuple[Any, ...]
    trip: int


@dataclasses.dataclass(frozen=True)
class Division:
    """One work-divided iteration symbol: a level the *cores* walk in parallel.

    A division is the outermost kind of level there is.  Where a ``Level``'s
    index comes from an ``scf.for``, a division's comes from this core's place in
    the grid -- ``(compute_tile_id // inner) % div`` -- so the two differ only in
    where the index value is read, and the plan carries them in one ordered list
    (divisions first, outermost) whose coefficients ``Access.index_coeffs``
    holds.

    ``symbol`` is the iteration symbol's name, kept for messages only: emission
    needs the two numbers.
    """

    symbol: str
    div: int  # how many cores share this symbol's range
    inner: int  # the grid stride of one step of this symbol


@dataclasses.dataclass(frozen=True)
class Layout:
    """A buffer's device extent and strides, in elements.

    ``extent`` entries are ``int``, or a sympy ``Expr`` for a dynamic extent.
    """

    extent: tuple[Any, ...]
    strides: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class Buffer:
    """One unique buffer referenced by the kernel; sole input to a memory view."""

    buf_id: str  # opspec_utils.buf_id(arg)
    arg_index: int  # position in the kernel call; -1 => not a kernel argument
    elems: ElemTypes
    layout: Layout
    base_elements: int | None  # ELEMENTS for the baked form; None => func arg
    space: str = "HBM"


@dataclasses.dataclass(frozen=True)
class Access:
    """One (OpSpec, TensorArg) access; sole input to an access tile.

    ``extent`` is the tile's own extent: ``device_size``, divided by whatever
    work division splits an axis across cores.  The *buffer* extent grows back
    out of it in ``_layout``.

    ``index_coeffs[i][l]`` is the step level ``l`` takes along view dim ``i``, so
    the index for dim ``i`` is ``sum_l index_coeffs[i][l] * iv_l``, where ``iv_l``
    is the index of the ``l``-th enclosing level -- this core's portion of a
    ``Division`` for the outermost ones, the induction variable of an enclosing
    ``scf.for`` for the rest.  A division and a loop differ only in where that
    index comes from, so one matrix covers both.  The record holds
    the coefficients only -- the variables exist during emission, not during
    planning -- and the builder zips them against the loops it has open.  This is
    the design's ``base_map`` as a matrix; the builder spells it the way
    hand-written loop kernels do, an identity ``base_map`` with one index
    expression per view dim, rather than a non-identity map over the induction
    variables.  The matrix is the same either way.

    ``elems`` is the access's own element type pair: a tile of an internal buffer
    has no ``Buffer`` to read one from, and a load that reinterprets would differ
    from its buffer's storage type anyway.

    ``buffer`` is what the access is a tile *of*, so a record carries its own way
    back to the view; ``None`` for an internal (threaded) buffer, which has no
    view because it never reaches memory.
    """

    extent: tuple[int, ...]
    index_coeffs: tuple[tuple[int, ...], ...]  # [view dim][level]
    elems: ElemTypes
    buffer: Buffer | None = None  # None for an internal (threaded) buffer


# ---------------------------------------------------------------------------
# Steps: the plan's instructions, which is all the builder is given
# ---------------------------------------------------------------------------
#
# A step is one thing to emit, resolved: no ``OpSpec``, no ``TensorArg``, no
# sympy, no SSA values.  ``KernelPlan.steps`` is a tree of them, and
# ``KtirBuilder.emit`` walks it, so everything the emitter needs to decide has
# been decided -- and every rejection has already been raised -- before emission
# begins.  The tree mirrors the spec tree's nesting because that nesting is what
# the loops are; what it does not carry is anything the emitter would have to
# interpret.


class Surface(enum.Enum):
    """The shape of the op that carries the payload.

    Chosen by the plan, so it is a step field rather than a decision emission
    makes: which shape an op comes out as follows from its operands' coordinates,
    and reading those is derivation.  Emission owning it would put a refusal
    behind a half-built module.

    ``BARE`` is a named linalg op (``linalg.add``), which states its own
    indexing; ``REDUCE`` is ``linalg.reduce`` with ``dimensions=``, which states
    only which axes go; ``GENERIC`` is ``linalg.generic``, the shape that has to
    state its maps and iterators because nothing else says them for it.
    """

    BARE = enum.auto()
    REDUCE = enum.auto()
    GENERIC = enum.auto()


@dataclasses.dataclass(frozen=True)
class Indexing:
    """What a ``linalg.generic`` must state, and nothing else.

    ``maps`` is the inputs in operand order and then the result last -- the order
    ``indexing_maps`` itself takes.  Each row is one iteration-dim index per
    result position, i.e. a *projection*: ``(0, 1, 2)`` against a rank-4 nest is
    ``(d0, d1, d2, d3) -> (d0, d1, d2)``.  Ints and strings rather than
    ``ir.AffineMap`` and iterator attributes, so the record stays dialect-free
    like every other one; ``KtirBuilder._affine_map`` is what turns a row into a
    map, the way ``access_tile`` already turns a rank into an identity.

    No ``extents`` field: ``linalg.generic`` infers its loop bounds from the
    operand shapes and the maps, so nothing would read one.

    A row is a bare dim index per position, which is every map in scope and not
    every map there is: a *linearised* map such as
    ``(d0, d1, d2, d3, d4) -> (d0, d2 * 64 + d3, d4)`` needs (coefficient, dim)
    terms, so nothing here generalises to one for free.
    """

    iters: tuple[str, ...]  # PARALLEL | REDUCTION, one per iteration dim
    maps: tuple[tuple[int, ...], ...]  # [operand][result position] -> dim


@dataclasses.dataclass(frozen=True)
class ComputeStep:
    """One compute op: what to read, what to apply, what to do with the result.

    ``ins`` is one ``(buf_id, Access)`` per operand, in the op's operand order.
    ``store`` is ``False`` for an internal result, which is bound in scope for a
    later step instead of being stored through ``out``.

    ``reduce_dims`` is **the iteration dims whose iterator is ``REDUCTION``**, and
    empty for every other surface.  On a ``REDUCE`` step those coincide with the
    input tile's own axes -- which is what ``dimensions=`` means -- because
    ``REDUCE`` is chosen exactly when the input map is the full identity; on a
    ``GENERIC`` step they do not, and the maps are what say so.

    ``indexing`` is carried on a ``GENERIC`` step and on no other, because it is
    read by one surface: a named op defines its own indexing and ``linalg.reduce``
    derives its maps from ``dimensions=``, so a per-operand map record on every
    step would be built three times and read once.

    ``attrs`` are the scalar arguments the payload builder takes beyond its
    operands -- softplus's ``beta``/``threshold`` -- as ``(name, value)`` pairs in
    the order the builder is called with them.  A tuple rather than a dict so the
    record stays hashable and frozen like every other field, and empty for every
    op that is a pure function of its operands, which is almost all of them.
    """

    op: str  # a KtirBuilder.RECIPES key
    surface: Surface
    ins: tuple[tuple[str, Access], ...]
    out: Access
    out_buf_id: str
    store: bool
    reduce_dims: tuple[int, ...] = ()
    indexing: Indexing | None = None
    attrs: tuple[tuple[str, float], ...] = ()
    # Which format's arm of the op this step wants.  Carried rather than
    # re-derived at emit time, where the args are no longer in reach: a step reads
    # no spec.  The format rather than the arm itself, because an arm holds a
    # deferred dialect reference and a step stays dialect-free.
    dtype: DataFormats | None = None


def dtype_of(spec: OpSpec) -> DataFormats:
    """The one device format \\p spec is asked for, or raise.

    An op name alone cannot tell an integer request from a float one -- both say
    ``add`` -- so the format is what picks the arm, and it has to be a single
    format: a mixed request has no arm to resolve to, and guessing one (taking the
    first, or any integer operand) would emit an intrinsic for the wrong type on
    the rest.  Mixed formats are refused here, in the plan, rather than being
    carried into emission where the choice is no longer visible.
    """
    formats = {arg.device_dtype for arg in spec.args}
    if len(formats) != 1:
        raise NotImplementedError(
            f"OpSpec->KTIR: {spec.op!r} mixes device formats "
            f"{sorted(f.name for f in formats)}; one op is one format"
        )
    return formats.pop()


@dataclasses.dataclass(frozen=True)
class LoopStep:
    """A counted loop, with the steps that go in its body.

    ``trip`` is an iteration count.
    """

    trip: int
    body: tuple[Step, ...]


Step = ComputeStep | LoopStep


# ---------------------------------------------------------------------------
# Derivations: one owner per OpSpec / TensorArg field
# ---------------------------------------------------------------------------


def _static(value) -> Any:
    """``value`` as a Python ``int`` when it is one, else ``value`` unchanged."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _mul(lhs, rhs) -> Any:
    """``lhs * rhs``, an ``int`` when both are."""
    return _static(lhs * rhs)


def _trip(loop: LoopSpec):
    """``loop``'s trip count: an ``int``, or the symbol it runs to.

    The only reader of ``LoopSpec.count``.  A symbolic count is returned as it
    is; whether one can be emitted is a question about the plan's
    ``symbolic_extent`` mode, so the plan decides it and this does the reading.
    """
    return _static(loop.count)


def _levels(spec: OpSpec, loops: Sequence[LoopSpec] = ()) -> list[Level]:
    """The enclosing loop levels for ``spec``, outermost-first.

    ``loops`` is the enclosing ``LoopSpec`` chain the walk is inside,
    outermost-first; ``()`` at function level.  ``OpSpec.tiled_symbols`` is
    innermost-first with one entry per enclosing level, so this is that list
    reversed and zipped against the loops -- the one place the two orderings
    meet, and therefore the place a mismatch between them is reported.

    With no enclosing loops the result is ``[]`` because ``tiled_symbols`` is
    empty and there is nothing to zip -- the general answer for a nest of depth
    zero, not a placeholder.
    """
    by_level = list(reversed(list(spec.tiled_symbols)))  # outermost-first
    if len(loops) != len(by_level):
        raise NotImplementedError(
            f"OpSpec->KTIR: op {spec.op!r} carries {len(by_level)} tiled_symbols "
            f"level(s) inside {len(loops)} enclosing loop(s); every enclosing "
            "level must have an entry, even an empty one"
        )
    levels: list[Level] = []
    for loop, symbols in zip(loops, by_level, strict=True):
        trip = _trip(loop)
        for symbol in symbols:
            declared = _static(spec.tiled_symbol_trip_counts.get(symbol, trip))
            if declared != trip:
                raise NotImplementedError(
                    f"OpSpec->KTIR: symbol {symbol} is declared with trip count "
                    f"{declared} but its loop level runs {trip} times"
                )
        levels.append(Level(symbols=tuple(symbols), trip=trip))
    return levels


def _advance_coeffs(arg: TensorArg, levels: Sequence[Level]) -> tuple[int, ...]:
    """Per-level linearized device-element step for ``arg``, one per level.

    ``device_tile_advance_expr`` is a single sum over the per-level symbols, so
    a level's own coefficient is the sum of its symbols' coefficients (a level
    with no symbols does not move this arg, hence ``0``).  ``None`` means the arg
    is not tiled at all: every level's step is ``0``.
    """
    expr = arg.device_tile_advance_expr
    if expr is None:
        return tuple(0 for _ in levels)
    coeffs: list[int] = []
    for level in levels:
        total = 0
        for symbol in level.symbols:
            coeff = _static(coeff_through_floor(expr, symbol))
            if not isinstance(coeff, int):
                raise NotImplementedError(
                    f"OpSpec->KTIR: tile-advance coefficient {coeff} for symbol "
                    f"{symbol} in {expr} is not an integer element count"
                )
            total += coeff
        coeffs.append(total)
    return tuple(coeffs)


def _advance(
    arg: TensorArg, levels: Sequence[Level], strides: Sequence[Any]
) -> list[tuple[int, ...]]:
    """``q[l][i]``: level ``l``'s step along view dim ``i``, in elements.

    The consumer of KTIR linearizes per-dim indices with the view's strides, and
    ``device_tile_advance_expr`` arrives already linearized, so this is that
    linearization's inverse against ``strides``: level ``l``'s coefficient
    ``c_l`` becomes the digit ``c_l / S_i`` on the one dim ``i`` it lands on.

    Coefficients are matched to dims smallest-first, dims innermost-first,
    excluding the trailing dim (a stick dim is never coarse-tiled, so nothing
    steps along it).  One dim per level: a level whose coefficient no remaining
    dim divides is left unassigned, which ``_solve_layout`` reports.  ``strides``
    entries that are not ``int`` are dims whose stride is not solved yet and are
    skipped, which is what makes the joint inner-to-outer solve possible.

    With no levels the result is ``[]``: there is nothing to decompose.
    """
    coeffs = _advance_coeffs(arg, levels)
    rank = len(strides)
    q = [[0] * rank for _ in levels]
    available = [
        i for i in range(rank - 2, -1, -1) if isinstance(strides[i], int)
    ]  # ascending stride
    ordered = sorted((coeff, level) for level, coeff in enumerate(coeffs) if coeff)
    for coeff, level_index in ordered:
        if coeff < 0:
            raise NotImplementedError(
                f"OpSpec->KTIR: negative tile advance {coeff} for level "
                f"{level_index} of {arg.name!r}; a view dim is walked backwards"
            )
        for position, dim in enumerate(available):
            if strides[dim] and coeff % strides[dim] == 0:
                q[level_index][dim] = coeff // strides[dim]
                del available[position]
                break
    return [tuple(row) for row in q]


def _static_extent(arg: TensorArg, extent: Any) -> int:
    """``extent`` as a whole number of elements, or a refusal.

    A symbolic device size would have to reach the kernel as an argument (and
    size a dynamic memref dim), which is not implemented.
    """
    if isinstance(extent, int):
        return extent
    raise NotImplementedError(
        f"OpSpec->KTIR: view extent {extent} of {arg.name!r} is symbolic; a "
        "symbolic device size is not supported yet"
    )


def _grown_extent(tile: Any, levels: Sequence[Level], steps: Sequence[int]) -> Any:
    """One dim's buffer extent: the tile extent plus what the levels walk over.

    ``E_i = A_i + sum_l q[l][i] * (T_l - 1)``.  The one implementation of that
    formula: ``_solve_layout`` uses it while solving strides and ``_layout`` uses
    it to build the record, so the two cannot disagree.
    """
    extent = tile
    for level, step in zip(levels, steps, strict=True):
        if step:
            extent = extent + step * (level.trip - 1)
    return _static(extent)


def _arrangement_layout(
    arrangement: Any, extent: tuple[Any, ...], strides: tuple[Any, ...]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """``(extent, strides)`` adjusted for the element order within a stick.

    ``element_arrangement`` is an element *order*, not a dtype conversion (one
    ``device_dtype`` covers the data type), so it is a layout fact: a rank and
    stride selector, of the shape the SDSC path already uses for a stick split.

    Label: ``staggered-element-arrangement``.
    """
    if arrangement in (None, ElementArrangement.STANDARD, ElementArrangement.QFP8CH):
        return extent, strides
    if arrangement in STAGGERED_EAS:
        _unimplemented(
            "staggered-element-arrangement",
            f"{arrangement!r} records a non-sequential element order inside the "
            "stick; the permutation has never been written down as numbers, so "
            "there is no rank/stride pair to emit for it",
        )
    raise NotImplementedError(
        f"OpSpec->KTIR: element arrangement {arrangement!r} has no layout rule"
    )


def _layout(
    arg: TensorArg,
    levels: Sequence[Level],
    q: Sequence[Sequence[int]],
) -> Layout:
    """``arg``'s buffer extent and strides, given the per-level steps ``q``.

    The buffer extent expands out of the tile extent by what the levels walk
    over (``_grown_extent``); strides are row-major of that extent.  The only
    place an extent becomes a memref dim, so the element arrangement is decided
    here -- and so is the demand that every extent be a whole number of elements,
    because a memref dim is either that or a dynamic size this emitter does not
    take yet.
    """
    tile = [_static(s) for s in arg.device_size]
    extent = tuple(
        _static_extent(arg, _grown_extent(tile[i], levels, [row[i] for row in q]))
        for i in range(len(tile))
    )
    extent, strides = _arrangement_layout(
        getattr(arg, "element_arrangement", None),
        extent,
        tuple(row_major_strides(extent)),
    )
    return Layout(extent=extent, strides=strides)


def _solve_layout(
    arg: TensorArg,
    levels: Sequence[Level],
) -> tuple[Layout, list[tuple[int, ...]]]:
    """``(Layout, q)`` for ``arg``: extents and per-level steps, solved together.

    They are mutually dependent -- a step is a multiple of a stride, a stride is
    a product of extents, an extent grows by a step -- so they cannot be given
    separate owners.  The solve runs innermost dim outward, which terminates
    because the trailing dim is a stick dim and is never coarse-tiled: its
    extent is ``device_size``' own, giving the first stride, and each further
    dim's extent is settled before the next stride needs it.

    With no levels this is one pass with nothing to decompose: the extent is
    ``device_size`` and the strides are row-major of it.
    """
    tile = [_static(s) for s in arg.device_size]
    rank = len(tile)
    strides: list[Any] = [None] * rank
    extent = list(tile)
    q: list[tuple[int, ...]] = [tuple([0] * rank) for _ in levels]
    for i in range(rank - 1, -1, -1):
        strides[i] = 1 if i == rank - 1 else _mul(extent[i + 1], strides[i + 1])
        q = _advance(arg, levels, strides)
        extent[i] = _grown_extent(tile[i], levels, [row[i] for row in q])
    coeffs = _advance_coeffs(arg, levels)
    seen: dict[int, int] = {}
    for level_index, coeff in enumerate(coeffs):
        if not coeff:
            continue
        if coeff in seen:
            raise NotImplementedError(
                f"OpSpec->KTIR: levels {seen[coeff]} and {level_index} of "
                f"{arg.name!r} both advance by {coeff} elements, so which view "
                "dim each walks cannot be told apart from the linearized advance"
            )
        seen[coeff] = level_index
        if not any(q[level_index]):
            raise NotImplementedError(
                f"OpSpec->KTIR: tile advance {coeff} elements (level "
                f"{level_index} of {arg.name!r}) is not a whole number of steps "
                f"along any dim of a view with strides {tuple(strides)}"
            )
    return _layout(arg, levels, q), q


def _divide(
    arg: TensorArg, symbols: Sequence[Any], divisors: dict
) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
    """``arg``'s per-core tile extent, and each division's step along each dim.

    The one place work division touches an access.  A division walks the axis
    its symbol addresses one per-core extent at a time, so its step along that
    axis *is* that extent, and the cores between them cover ``device_size``
    exactly (``A * D``).  The view is unaffected -- ``_solve_layout`` builds it
    from the whole ``device_size``, because every core addresses the same buffer
    -- so nothing else in the emitter has to know that cores exist.

    ``symbols`` is outermost-first, matching the plan's division order; the rows
    come back in that order so they prepend to the loop levels' rows.
    """
    if not symbols:
        return tuple(_static(s) for s in arg.device_size), []
    per_core, axis_symbol = per_core_extent(arg, divisors)
    rows = [
        tuple(
            per_core[axis] if axis_symbol[axis] == symbol else 0
            for axis in range(len(per_core))
        )
        for symbol in symbols
    ]
    return tuple(per_core), rows


def _squeezed(arg: TensorArg, axes: Sequence[int]) -> TensorArg:
    """``arg`` without ``axes``: same buffer, same bytes, fewer device axes.

    Dropping a unit axis renames nothing and moves nothing -- it contributes no
    elements and no stride -- so every later derivation sees an arg of the rank
    the emitted tile actually has, and ``buf_id`` still identifies one buffer.
    """
    drop = set(axes)
    keep = [axis for axis in range(len(arg.device_size)) if axis not in drop]
    return dataclasses.replace(
        arg,
        device_size=[arg.device_size[axis] for axis in keep],
        device_coordinates=[arg.device_coordinates[axis] for axis in keep],
    )


def _reduce_surface(
    iters: Sequence[str], in_map: Sequence[int], out_map: Sequence[int]
) -> Surface:
    """Which shape says the nest ``reduction_indexing`` derived.

    ``REDUCE`` iff the nest is what ``linalg.reduce`` *means*: an identity input
    map of the full rank, and an output map that is the identity with the reduced
    dims dropped.  ``mlir::linalg::ReduceOp`` derives its maps as
    ``getMultiDimIdentityMap(rank).dropResults(dimensions)``, so nothing else is
    expressible by ``dimensions=`` alone and everything else needs a generic.

    Testing the *input* map is the load-bearing half.  An on-stick reduction's
    output map is ``(1, 3)``, which *is* the identity of a rank-4 nest with
    ``(0, 2)`` dropped -- so an output-only test would accept it and emit a rank-2
    ``linalg.reduce`` over a rank-3 input, silently reducing the wrong elements.
    What disqualifies it is that its input map covers 3 of 4 dims.
    """
    rank = len(iters)
    reduced = {dim for dim, iterator in enumerate(iters) if iterator == REDUCTION}
    identity = tuple(range(rank))
    kept = tuple(dim for dim in identity if dim not in reduced)
    return (
        Surface.REDUCE
        if tuple(in_map) == identity and tuple(out_map) == kept
        else Surface.GENERIC
    )


def _parallel_surface(
    arm: Arm, operands: int, rank: int
) -> tuple[Surface, Indexing | None]:
    """Which shape carries a non-reducing payload, and what it has to state.

    Nothing is *derived* here and nothing needs to be: the pointwise arm's
    alignment refusal has already established that every operand's coordinates
    and extents equal the output's, which is precisely the identity condition, so
    the maps are known rather than read off the coordinates.  Deriving them
    instead would make the emitted form of ``add`` hostage to the dim-reuse rule
    ``reduction_indexing`` needs -- a coordinate list that repeated a
    classification would yield a non-identity map and silently turn a
    ``linalg.add`` into a ``linalg.generic``.

    So the choice is only about spelling, and it follows from the binding: a
    ``NAMED`` builder is an op the dialect already has, which says its own
    indexing and needs no record, while anything else has to state the identity
    maps and the all-parallel iterators itself -- which only a generic can do.
    That second arm is where a ``spyreop`` intrinsic lands: it is a scalar builder,
    so there is nothing to call it but a region.
    """
    if arm.kind is BindingKind.NAMED:
        return Surface.BARE, None
    identity = tuple(range(rank))
    return Surface.GENERIC, Indexing(
        iters=(PARALLEL,) * rank,
        maps=(identity,) * (operands + 1),
    )


def _access(
    arg: TensorArg,
    extent: Sequence[Any],
    rows: Sequence[Sequence[int]],
    layout: Layout,
    buffer: Buffer | None = None,
) -> Access:
    """The access record for one ``(OpSpec, TensorArg)``.

    ``extent`` is the per-core tile extent and ``rows`` is one step vector per
    enclosing level, outermost-first (divisions, then loops) -- which the builder
    multiplies by that level's index at emit time.

    With no levels every row is empty, so every index expression is the empty
    sum -- zero -- which is why an undivided, untiled access sits at the view's
    origin.
    """
    extent = tuple(_static(s) for s in extent)
    for value in extent:
        if not isinstance(value, int):
            raise NotImplementedError(
                f"OpSpec->KTIR: access tile extent {value} of {arg.name!r} is "
                "symbolic; a tile is sized in whole elements"
            )
    if len(extent) != len(layout.extent):
        raise AssertionError(
            f"access rank {len(extent)} != buffer rank {len(layout.extent)}"
        )
    index_coeffs = tuple(tuple(int(row[i]) for row in rows) for i in range(len(extent)))
    return Access(
        extent=extent,
        index_coeffs=index_coeffs,
        elems=ElemTypes.of(arg.device_dtype),
        buffer=buffer,
    )


# ---------------------------------------------------------------------------
# KernelPlan: everything the builder is given
# ---------------------------------------------------------------------------
#
# The plan is the whole instruction list: the grid, the buffers whose views and
# func parameters the kernel opens with, and the step tree that goes in its body.
# It is built by one walk of the spec tree, which is where the derivations run
# and therefore where every rejection is raised.  Emission consumes the plan and
# reads no spec, so it cannot discover a reason to refuse half-way through.


# The ``allocation`` keys memory planning uses for a buffer the kernel owns.
INTERNAL_SPACES: tuple[str, ...] = ("lx", "hbm_pool")


def is_internal(arg: TensorArg) -> bool:
    """Whether this buffer is one the kernel owns rather than one it is passed.

    Read from ``allocation``, which is where the contract already says it:
    ``"hbm"`` is a graph input or output, addressed directly, while ``"lx"`` and
    ``"hbm_pool"`` are intermediates that memory planning placed on the kernel's
    behalf.  ``create_tensor_arg`` uses that same distinction to decide what
    becomes a kernel argument at all (``spyre_kernel.py``: an ``lx`` /
    ``hbm_pool`` tensor is left out of ``spyre_kernel_args``, which is why those
    args carry ``arg_index == -1``) -- so this is that rule read back, not a
    second convention, and not the sentinel index, which says only "not passed".

    The two emitters answer differently because their granularity differs: one
    ``sdsc_execute`` per OpSpec forces SDSC to materialise the intermediate into
    the allocation it was given, while one KTIR func for the whole kernel lets it
    stay an SSA value -- no store, no view, no parameter, and no address for the
    scheduler to honour, which is what "the scheduler owns buffering" means here.
    """
    # Named positively: an allocation this emitter does not recognise at all is
    # not silently threaded, it reaches ``_buffer`` and is refused there.
    return any(space in (arg.allocation or {}) for space in INTERNAL_SPACES)


def _buffer(
    arg: TensorArg,
    layout: Layout,
    elems: ElemTypes,
    *,
    bake_addresses: bool = False,
) -> Buffer:
    """``arg``'s buffer record, rejecting what the emitter cannot address.

    The one place a ``TensorArg`` becomes a ``Buffer``, so every buffer-level
    rejection is here and the record the plan holds is the record the view is
    emitted from.  ``layout`` and ``elems`` are the other derivations' answers,
    passed in rather than re-derived.
    """
    # ``arg_index`` stays -1 for buffers the frontend does not pass to the
    # kernel, which today means an LX or HBM-pool allocation.  This emitter
    # constructs HBM memory views only.
    if arg.arg_index < 0:
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} is not a kernel argument "
            f"(allocation={arg.allocation!r}); only HBM buffers are supported"
        )
    return Buffer(
        buf_id=buf_id(arg),
        arg_index=arg.arg_index,
        elems=elems,
        layout=layout,
        # Resolved only for the baked form: the symbolic form takes its bases
        # from func arguments and never reads ``allocation["hbm"]``, whose units
        # differ between the two forms.
        base_elements=_base_address_elements(arg) if bake_addresses else None,
        space="HBM",
    )


@dataclasses.dataclass(frozen=True)
class PlanOptions:
    """Everything the caller chooses about one emission, in one value.

    One choice, and it is not a capability switch: what the kernel *does* comes
    from the OpSpec contract (its ``LoopSpec``s are its loops), so it is not the
    caller's to pick.  What is left is how to spell the one thing the contract
    does not decide.

    ``bake_addresses`` emits each base as an ``arith.constant`` in elements
    instead of a func argument, because ``ktdp.load`` requires a static memref
    offset, which a constant base gives only when the consumer is a ``linalg``
    op.  Canonical KTIR is symbolic; baking is the dataflow-scheduler#65
    workaround that dbo-opt requires.  The SDSC path makes the same choice from
    ``config.bundle_symbolic_args``.
    """

    bake_addresses: bool = False


def _divisions(specs: Sequence[Any]) -> tuple[list[Any], tuple[Division, ...]]:
    """``(symbols, divisions)``: the core grid the spec tree asks for.

    Read from ``OpSpec.iteration_space``, whose per-symbol work division is what
    ``work_division.py`` decided from ``config.sencores`` -- so the grid is a fact
    of the contract, the same one the SDSC path reads as its work slices, rather
    than a core count this emitter takes on the side.  Nothing else here reads
    ``config``.

    Every op in one kernel must ask for the same division: they share one grid,
    and one core runs one instance of the whole body.  The symbols come back
    outermost-first, so a division's coefficients prepend to the loop levels'.
    """
    spaces = [spec.iteration_space for spec in _op_specs(specs)]
    if not spaces:
        return [], ()
    divided, total = core_divisions(spaces[0])
    for space in spaces[1:]:
        if core_divisions(space)[0] != divided:
            raise NotImplementedError(
                "OpSpec->KTIR: the ops in this kernel ask for different work "
                "divisions, so they cannot share one grid; mixed work division "
                "within a kernel is not supported"
            )
    divided = list(reversed(divided))  # outermost-first
    symbols = [symbol for symbol, _div, _inner in divided]
    divisions = tuple(
        Division(symbol=str(symbol), div=div, inner=inner)
        for symbol, div, inner in divided
    )
    assert total == functools.reduce(lambda a, d: a * d.div, divisions, 1)
    return symbols, divisions


def _op_specs(specs: Sequence[Any]) -> Iterator[OpSpec]:
    """Every ``OpSpec`` in a spec tree, loop bodies included."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            yield from _op_specs(entry.body)
        elif isinstance(entry, OpSpec):
            yield entry


class KernelPlan:
    """One kernel, resolved: its grid, its buffers, and the steps for its body.

    Fills itself from a spec tree -- ``add_specs`` is the walk -- so the options,
    the buffers, the steps and the walk that produces them are one object rather
    than a dict threaded through free functions.  Filling it is what runs the
    derivations, so it is also where every ``NotImplementedError`` the emitter can
    raise comes from, and a plan that exists is a kernel that can be emitted.

    ``grid`` is resolved here rather than at emit time: the builder emits the grid
    it is given and does not know what a core is.

    """

    def __init__(self, options: PlanOptions | None = None) -> None:
        self.options = options or PlanOptions()
        self.grid: tuple[int, ...] = (1,)
        self.divisions: tuple[Division, ...] = ()
        self._symbols: list[Any] = []  # the divided symbols, outermost-first
        self._divisors: dict = {}
        self.buffers: dict[str, Buffer] = {}
        self.steps: tuple[Step, ...] = ()

    @property
    def parameters(self) -> list[Buffer]:
        """External buffers in ascending ``arg_index``.

        Ascending ``arg_index`` matches the positional order ``call_kernel``
        passes to ``.run(...)``, so the emitted func signature lines up with
        that binding.
        """
        return sorted(
            (e for e in self.buffers.values() if e.arg_index >= 0),
            key=lambda e: e.arg_index,
        )

    def add_specs(self, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]) -> None:
        """Plan ``specs`` into this plan's grid, buffers and steps."""
        self._symbols, self.divisions = _divisions(specs)
        self._divisors = {
            symbol: division.div
            for symbol, division in zip(self._symbols, self.divisions, strict=True)
        }
        cores = 1
        for division in self.divisions:
            cores *= division.div
        self.grid = (cores,)
        self.steps = self._steps(specs, ())
        self._check_internal_buffers(self.steps)

    def _check_internal_buffers(self, steps: Sequence[Step]) -> None:
        """A threaded buffer must be produced before it is read, and then read.

        A threaded value has no memory behind it, so the kernel has to contain
        both ends of it.  Either end missing means the intermediate reached this
        kernel without the op on the other side of it -- the fusion decision and
        the kernel boundary disagree -- and the buffer needs materialising
        instead.  Refused here rather than emitted: an unread producer would
        silently write nowhere, and an unproduced consumer has no value to read.
        """
        unread: dict[str, None] = {}  # threaded, produced, not yet read
        produced: set[str] = set()

        def walk(steps: Sequence[Step]) -> None:
            for step in steps:
                if isinstance(step, LoopStep):
                    walk(step.body)
                    continue
                for read_id, access in step.ins:
                    if access.buffer is not None:  # loaded from its own view
                        continue
                    if read_id not in produced:
                        raise NotImplementedError(
                            f"OpSpec->KTIR: buffer {read_id!r} is an intermediate "
                            "this kernel owns (its allocation is lx / hbm_pool, so "
                            "it is threaded as a value rather than loaded) but no "
                            "op in this kernel produces it; its producer is in "
                            "another kernel, which needs the buffer materialised"
                        )
                    unread.pop(read_id, None)
                if not step.store:
                    produced.add(step.out_buf_id)
                    unread[step.out_buf_id] = None

        walk(steps)
        for unread_id in unread:
            raise NotImplementedError(
                f"OpSpec->KTIR: buffer {unread_id!r} is an intermediate this kernel "
                "owns (its allocation is lx / hbm_pool, so it is threaded as a "
                "value rather than stored) but nothing in this kernel reads it; "
                "its consumer is in another kernel, which needs the buffer "
                "materialised"
            )

    def _steps(self, specs, loops: Sequence[LoopSpec]) -> tuple[Step, ...]:
        """Recursive: the steps for one spec list, inside the ``loops`` chain.

        ``loops`` is the enclosing ``LoopSpec`` chain, outermost-first, which is
        what ``_levels`` zips ``OpSpec.tiled_symbols`` against.  A nested list
        becomes a nested ``LoopStep.body``, so the step tree's nesting is the
        spec tree's nesting and the emitter never has to work out the depth.
        """
        steps: list[Step] = []
        for entry in specs:
            if isinstance(entry, UnimplementedOp):
                raise NotImplementedError(
                    f"OpSpec->KTIR: unimplemented op {entry.op!r}"
                )
            if isinstance(entry, LoopSpec):
                trip = _trip(entry)
                if not isinstance(trip, int):
                    # A symbolic count would have to reach the kernel as an
                    # argument, the same one a dynamic view dim needs.
                    raise NotImplementedError(
                        f"OpSpec->KTIR: loop trip count {entry.count} is symbolic; "
                        "a symbolic trip count is not supported yet"
                    )
                steps.append(
                    LoopStep(trip=trip, body=self._steps(entry.body, [*loops, entry]))
                )
                continue
            if not isinstance(entry, OpSpec):
                raise NotImplementedError(
                    f"OpSpec->KTIR: unexpected spec entry {type(entry).__name__}"
                )
            if entry.op not in KtirBuilder.RECIPES:
                raise NotImplementedError(
                    f"OpSpec->KTIR: op {entry.op!r} is not supported yet "
                    f"(registered: {sorted(KtirBuilder.RECIPES)})"
                )
            # One question about the op name, then one about its format: whether
            # the op exists at all is the table's business, and which of its
            # spellings this request reaches is the recipe's.
            arm = KtirBuilder.RECIPES[entry.op].arm(dtype_of(entry))
            if (arm.kind is BindingKind.COMBINER) != bool(entry.is_reduction):
                # Two independent statements of one bit -- what the recipe's
                # binding accumulates, and what the frontend labelled the request
                # -- and both directions are silent if unchecked.  An 'add' asked
                # for as a reduction would derive no reduced axis and come out as
                # a plain 'linalg.add' for a spec the frontend called a reduction;
                # a 'sum' asked for elementwise would reach a two-operand scalar
                # combiner with one operand and fail inside emission, which is the
                # one thing the plan/emission split exists to rule out.
                raise NotImplementedError(
                    f"OpSpec->KTIR: op {entry.op!r} is registered as "
                    f"{arm.kind.name} but this spec asks for "
                    f"{'a reduction' if entry.is_reduction else 'an elementwise op'}"
                )
            steps.append(self._compute_step(entry, loops))
        return tuple(steps)

    def _compute_step(self, spec: OpSpec, loops: Sequence[LoopSpec]) -> ComputeStep:
        """One op: roles/arity, aliasing, alignment, its buffers and its accesses.

        Every derivation for this op runs here, once: the layout and per-level
        steps solved for an arg are the ones its buffer records and the ones its
        access is built from, so a view and the tiles into it cannot disagree, and
        emission has nothing left to derive.
        """
        out, inputs = validated_roles(spec)
        out_extents = [int(s) for s in out.device_size]
        dtype = dtype_of(spec)
        recipe = KtirBuilder.RECIPES[spec.op]
        arm = recipe.arm(dtype)
        for arg in inputs:
            # In-place (input buffer aliases the output) is not supported yet.
            if buf_id(arg) == buf_id(out):
                raise NotImplementedError(
                    "OpSpec->KTIR: in-place ops (input aliases output) not supported"
                )
        reduce_dims: tuple[int, ...] = ()
        indexing: Indexing | None = None
        args = list(spec.args)
        if spec.is_reduction:
            # What iteration nest a reduction wants is a fact about its operands'
            # coordinates, so it is derived here (once) and carried on the step,
            # not re-derived from the op name at emit time.  Every reduction in
            # scope is unary, which is why the derivation takes one input.
            [source] = inputs
            placeholder = placeholder_axes(out.device_coordinates, out_extents)
            if placeholder:
                # The projection leaves an axis the op does not write in the output
                # as a unit extent; the reduced tile does not have it at all.
                # Squeezing the arg here, once and before ``_access_of``, is what
                # keeps every derivation after this point unaware that a reduction
                # is different: the output's view, tile, per-core division and
                # stored tensor are all the same (lower) rank.  It stays gated on
                # ``is_reduction`` because an *accepted* pointwise spec can carry a
                # unit constant axis on its inputs too, and squeezing only the
                # output would hand ``linalg.add`` operands of two ranks.
                squeezed = _squeezed(out, placeholder)
                args = [squeezed if arg is out else arg for arg in args]
                out = squeezed
            iters, in_map, out_map = reduction_indexing(
                source.device_coordinates,
                [int(s) for s in source.device_size],
                out.device_coordinates,
                [int(s) for s in out.device_size],
            )
            surface = _reduce_surface(iters, in_map, out_map)
            reduce_dims = tuple(
                dim for dim, iterator in enumerate(iters) if iterator == REDUCTION
            )
            if surface is Surface.GENERIC:
                # The one nest ``dimensions=`` cannot state, so the maps have to
                # travel with the step: the input covers three of four dims and
                # the lane axis is reduced on the way in and kept on the way out.
                indexing = Indexing(iters=iters, maps=(in_map, out_map))
        else:
            surface, indexing = _parallel_surface(arm, len(inputs), len(out_extents))
            for arg in inputs:
                # Reject broadcast / transpose operands: only operands whose
                # device axes already match the output tile exactly are supported.
                if (
                    align_reshape_plan(
                        list(arg.device_coordinates),
                        [int(s) for s in arg.device_size],
                        list(out.device_coordinates),
                        out_extents,
                    )
                    is not None
                ):
                    raise NotImplementedError(
                        "OpSpec->KTIR: broadcast / reshape operands not supported yet"
                    )
        levels = _levels(spec, loops)
        accesses = {buf_id(arg): self._access_of(arg, levels) for arg in args}
        # Every division must move this op's output: cores divide work by writing
        # different elements, so a division no output axis follows is cores
        # duplicating each other rather than sharing.  An *input* may legitimately
        # not follow one (every core reads the same operand), which is why this
        # asks the output only.
        out_coeffs = accesses[buf_id(out)].index_coeffs
        for level, division in enumerate(self.divisions):
            if not any(row[level] for row in out_coeffs):
                raise NotImplementedError(
                    f"OpSpec->KTIR: work division splits {division.symbol} across "
                    f"{division.div} cores, but no device axis of the output "
                    f"{out.name!r} follows it, so every core would compute the "
                    "same elements; dividing the within-stick axis or a reduced "
                    "axis (which needs a cross-core combine) reads like this"
                )
        # The scalar arguments the payload builder takes beyond its operands are
        # read here, once, from the spec's ``op_info`` -- the same place-and-time
        # discipline as ``reduce_dims`` and ``indexing`` -- so emission has nothing
        # left to derive and a malformed ``op_info`` is refused by the plan rather
        # than by a KeyError with a half-built module in hand.
        attrs: tuple[tuple[str, float], ...] = ()
        if recipe.attrs is not None:
            attrs = tuple(recipe.attrs(spec.op_info).items())
        return ComputeStep(
            op=spec.op,
            surface=surface,
            ins=tuple((buf_id(arg), accesses[buf_id(arg)]) for arg in inputs),
            out=accesses[buf_id(out)],
            out_buf_id=buf_id(out),
            reduce_dims=reduce_dims,
            indexing=indexing,
            attrs=attrs,
            dtype=dtype,
            # An internal buffer never reaches memory: it is threaded as a value,
            # so it gets no store, no func parameter, no view and no address.
            store=not is_internal(out),
        )

    def _access_of(self, arg: TensorArg, levels: Sequence[Level]) -> Access:
        """``arg``'s access at this depth, registering its buffer on the way.

        The buffer is registered first and handed to the access, so the record
        carries its own way back to the view the builder will bind for it.  The
        first record seen for a ``buf_id`` wins, which is the one every later
        access to that buffer points at.
        """
        layout, q = _solve_layout(arg, levels)
        elems = ElemTypes.of(arg.device_dtype)
        buffer = None
        if not is_internal(arg):
            buffer = self.buffers.setdefault(
                buf_id(arg),
                _buffer(arg, layout, elems, bake_addresses=self.options.bake_addresses),
            )
        # The divisions are the outermost levels, so their steps come first.
        extent, rows = _divide(arg, self._symbols, self._divisors)
        return _access(arg, extent, [*rows, *q], layout, buffer)


def _base_address_elements(arg: TensorArg) -> int:
    """``arg``'s buffer base address in ELEMENTS, for the baked form only.

    Read from ``allocation["hbm"]``, the same field the SDSC path resolves into
    the bundle start address (``superdsc.py:774`` -> ``startAddressCoreCorelet_``).
    Its units follow ``config.bundle_symbolic_args``: baked gives a byte address
    (arg 1 -> ``{'hbm': 17179869184}``), symbolic a bare sentinel ``arg_index``
    (arg 1 -> ``{'hbm': 1}``).  A memref offset indexes the *element* type, so
    the byte address is scaled down by the element size.
    """
    allocation = arg.allocation or {}
    # Key presence, not truthiness: a legitimate 'hbm' address of 0 exists.
    if "hbm" not in allocation:
        space = next(iter(allocation), None)
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} is not HBM-allocated "
            f"(allocation={allocation!r}); the emitter only emits HBM memory "
            f"views, so {space!r} allocations are out of scope"
        )
    byte_offset = allocation["hbm"]
    if byte_offset is None:
        raise NotImplementedError(
            f"OpSpec->KTIR: buffer {arg.name!r} has an unassigned 'hbm' "
            "address (None); memory planning must run before KTIR emission"
        )
    return int(byte_offset) // num_bytes(arg.device_dtype)


# ---------------------------------------------------------------------------
# build_kernel_plan: every rejection, with no mlir_ktdp
# ---------------------------------------------------------------------------


def build_kernel_plan(
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
    options: PlanOptions | None = None,
) -> KernelPlan:
    """The kernel's ``KernelPlan``, and every rejection on the way to it.  Pure.

    The whole-request checks are the grid (in ``KernelPlan.__init__``) and the
    empty-kernel check below; everything per-spec is ``KernelPlan.add_specs``.
    Imports nothing from ``mlir_ktdp`` (the dialect import is lazy, inside
    ``KtirBuilder.create``), so it is usable wherever ``import ktir`` works --
    which is everywhere.
    """
    plan = KernelPlan(options)
    plan.add_specs(specs)
    if not plan.buffers:
        raise NotImplementedError("OpSpec->KTIR: no OpSpec to emit")
    return plan


def validated_roles(spec: OpSpec) -> tuple[TensorArg, list[TensorArg]]:
    """``(output, inputs)`` for ``spec``, or raise.  Pure; shared with the plan walk.

    Handlers call this instead of re-deriving the roles, so the arity and
    single-output rejections have exactly one implementation.
    """
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(outputs) != 1:
        raise NotImplementedError(
            f"OpSpec->KTIR: expected exactly one output, got {len(outputs)}"
        )
    # Arity is a property of the op, not of the format it is asked for, so this
    # needs no arm and cannot be wrong about which spelling the request reaches.
    arity = KtirBuilder.RECIPES[spec.op].arity
    if len(inputs) != arity:
        raise NotImplementedError(
            f"OpSpec->KTIR: {spec.op!r} expects {arity} inputs, got {len(inputs)}"
        )
    return outputs[0], inputs


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
#
# A recipe declares one op: how many inputs it takes, what kind of thing its
# binding is, and the dialect builder itself.  The recipes live on
# ``KtirBuilder`` beside the surfaces that execute them.
#
# ``binding`` returns the builder rather than being it.  The call defers the
# dialect reference to emit time, keeping this module importable without a
# dialect build, and keeps the reference a literal that tooling can resolve.


class BindingKind(enum.Enum):
    """What ``Arm.binding()`` returns, which is what decides how it is used.

    Three kinds, and the surface follows from the kind plus (for a ``COMBINER``)
    the shape of the reduction:

      NAMED    a whole-op builder for a linalg op that already exists
               (``linalg.add``).  Elementwise; emitted bare.
      PAYLOAD  a scalar builder for a parallel body (a ``spyreop`` intrinsic).
               Elementwise, but no named op wraps it, so it needs a generic.
      COMBINER a two-operand scalar folded into the accumulator (``arith.addf``).
               The only kind that reduces.

    No separate ``reduces`` flag: reducing *is* ``kind is COMBINER``, because a
    named linalg op is elementwise and a parallel-body payload does not
    accumulate.  The kind belongs to the ``Arm`` and not to the ``Recipe``
    because it varies with the format: ``add`` is a named ``linalg`` op at floats
    and a ``spyreop`` payload at four-byte integers.
    """

    NAMED = enum.auto()
    PAYLOAD = enum.auto()
    COMBINER = enum.auto()


@dataclasses.dataclass(frozen=True)
class Arm:
    """One format's spelling of an op: its builder, how it is used, and its trigger.

    ``dtypes`` are the formats that reach this arm.  Empty claims every format no
    sibling arm claims, which is what all but a handful of arms want -- an op with
    one spelling is one arm with an empty ``dtypes``, and the float arm of an op
    that also has an integer one does not have to enumerate every float format to
    say "not that one".

    The trigger travels with the binding deliberately.  The alternative -- a
    second table keyed on op name, consulted first and fallen back out of -- makes
    the two spellings unequal: one table answers "is this op supported at all" and
    the other silently overrides it, so an op registered only in the second is
    reported unsupported while holding a perfectly good recipe.
    """

    kind: BindingKind
    binding: Callable[[], Any]
    dtypes: tuple[DataFormats, ...] = ()


def _arms(arms: Arm | tuple[Arm, ...]) -> tuple[Arm, ...]:
    """\\p arms as a tuple, whether it was written as one arm or several.

    Idempotent, so it is safe to call on an already-normalised field: it is what
    ``Recipe.__post_init__`` normalises *with* and what every reader goes through,
    which keeps the one-arm shorthand from being a second representation that some
    code path forgets to handle.
    """
    return (arms,) if isinstance(arms, Arm) else arms


@dataclasses.dataclass(frozen=True)
class Recipe:
    # Both ``arity`` and ``attrs`` are properties of the *op*, invariant across
    # formats, which is why they sit here and not on an arm: 'add' takes two
    # operands and reads no scalars from ``op_info`` whatever its element type is.
    # Stating them once is also what keeps two spellings of one op from drifting
    # apart on arity.
    arity: int
    # One ``Arm`` or a tuple of them; ``__post_init__`` promotes the bare one, so
    # the field is a tuple by the time anything reads it.  Written this way because
    # an op with a single spelling is the overwhelming majority and ``arms=Arm(...)``
    # is what that op means -- the ``(...,)`` around it would be noise on eleven of
    # the thirteen entries, and a stray missing comma turns a tuple into an ``Arm``
    # silently.
    arms: Arm | tuple[Arm, ...]
    # How to read the op's scalar arguments out of a spec's ``op_info``, for the
    # few ops whose builder takes more than operands (softplus).  ``None`` when
    # the op is a pure function of its operands, which is almost all of them.
    # A reader rather than the values themselves, because where they live in
    # ``op_info`` is the op's own business and the plan should not have to know.
    attrs: Callable[[dict[str, Any]], dict[str, float]] | None = None

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError(f"OpSpec->KTIR: arity must be >= 1, got {self.arity}")
        arms = _arms(self.arms)
        if not arms:
            raise ValueError("OpSpec->KTIR: a recipe needs at least one arm")
        if sum(1 for arm in arms if not arm.dtypes) > 1:
            raise ValueError(
                "OpSpec->KTIR: at most one arm may claim the unlisted formats"
            )
        claimed = [dtype for arm in arms for dtype in arm.dtypes]
        if len(claimed) != len(set(claimed)):
            raise ValueError(
                "OpSpec->KTIR: two arms claim the same format: "
                f"{sorted({d.name for d in claimed if claimed.count(d) > 1})}"
            )
        object.__setattr__(self, "arms", arms)

    def arm(self, dtype: DataFormats | None) -> Arm:
        """The arm \\p dtype reaches, or raise.

        A format nothing claims falls to the arm with an empty ``dtypes``; if no
        arm takes the unlisted formats either, the op does not exist at this one.
        """
        arms = _arms(self.arms)
        for candidate in arms:
            if dtype is not None and dtype in candidate.dtypes:
                return candidate
        for candidate in arms:
            if not candidate.dtypes:
                return candidate
        raise NotImplementedError(
            f"OpSpec->KTIR: no arm for {dtype.name if dtype else 'an unknown format'} "
            f"(registered: {sorted(d.name for a in arms for d in a.dtypes)})"
        )


# ---------------------------------------------------------------------------
# What the builder carries in scope
# ---------------------------------------------------------------------------


class ScopeStack:
    """Builder-owned lexical scope: open loops and live values.

    Pushed and popped by ``KtirBuilder.emit`` via ``with``.  A base frame is
    always present so values produced at function level have somewhere to live.
    It carries the core portions, if the kernel is work-divided: those are the
    outermost levels, in scope for the whole body and not tied to any loop.
    """

    def __init__(self) -> None:
        # ([index values], {buf_id: Value}), innermost last.  A frame's list is
        # the levels it opens: one iv for a loop, the core portions for the base
        # frame, none for a plain value scope.
        self._frames: list[tuple[list, dict[str, Any]]] = [([], {})]

    def bind_ivs(self, ivs: Sequence) -> None:
        """Give the current frame these level indices (the base frame's cores)."""
        self._frames[-1][0].extend(ivs)

    @contextlib.contextmanager
    def scope(self, iv: Any = None) -> Iterator[None]:
        self._frames.append(([] if iv is None else [iv], {}))
        try:
            yield
        finally:
            self._frames.pop()

    def produced(self, buf_id: str):
        """The ``Value`` a live step produced for ``buf_id``, else ``None``."""
        for _, produced in reversed(self._frames):
            if buf_id in produced:
                return produced[buf_id]
        return None

    def bind_produced(self, buf_id: str, value) -> None:
        self._frames[-1][1][buf_id] = value

    def ivs(self) -> list:
        """The index of every open level, outermost-first.

        What ``Access.index_coeffs`` is zipped against: one coefficient per
        enclosing level, in the same order the plan derived them -- the core
        portions of the work divisions first, then one induction variable per
        enclosing loop.
        """
        return [iv for ivs, _ in self._frames for iv in ivs]


# ---------------------------------------------------------------------------
# KtirBuilder
# ---------------------------------------------------------------------------


class KtirBuilder:
    """Owns the MLIR context, the dialect handles and per-module state.

    No method takes an ``OpSpec`` or a ``TensorArg``: the arguments are the
    plan's records (``Buffer``, ``Layout``, ``ElemTypes``, ``Access``, ``Step``),
    SSA values and primitives.  ``emit`` walks the plan's steps, so the builder
    is the only thing that touches the dialect and the plan is the only thing it
    reads -- there is no spec tree on this side of the boundary.

    Every ktdp shape method returns an SSA ``Value``, so ``val()`` does not
    appear at call sites.
    """

    def __init__(self, stack, plan: KernelPlan):
        self._stack = stack
        self.plan = plan
        self.env = ScopeStack()
        # Requires the live context entered by create().
        self.index_t = ir.IndexType.get()
        self.block_args: list = []
        self.views: dict[str, Any] = {}
        self.c0 = None
        self._text: str | None = None

    @classmethod
    def create(cls, plan: KernelPlan) -> KtirBuilder:
        """THE single lazy-import site, and the owner of the MLIR context.

        Module level stays ``mlir_ktdp``-free, so ``import ktir`` -- and
        therefore ``build_kernel_plan`` -- works where the dialect build is absent.

        The context is entered here rather than in ``module()`` because
        ``_func_param_types`` builds ``ir`` types and is called before the module
        is opened; ``module()`` closes it on the way out.
        """
        _load_dialects()

        stack = contextlib.ExitStack()
        try:
            ctx = stack.enter_context(ir.Context())
            stack.enter_context(ir.Location.unknown())
            ktdp.register_dialects(ctx)
            return cls(stack, plan)
        except BaseException:
            stack.close()
            raise

    # -- generic helpers ---------------------------------------------------

    @staticmethod
    def val(x):
        """The SSA ``Value`` of a builder result (builders return ``OpView`` or ``Value``)."""
        return x.result if hasattr(x, "result") else x

    @staticmethod
    def named_type(name: str):
        """The ``ir`` type for one ``ElemTypes`` entry (an MLIR type spelling).

        Parsed rather than dispatched to a builder so that one spelling works for
        every element type this emitter supports: the float builders take no
        argument while ``IntegerType`` takes a width, so naming builders would
        make the integer entries a second kind of name to be told apart by
        inspecting the string.
        """
        return ir.Type.parse(name)

    def icst_index(self, value: int):
        """A fresh ``arith.constant <value> : index``."""
        return self.val(arith.ConstantOp(self.index_t, int(value)))

    @staticmethod
    def _affine_map(rank: int, row: Sequence[int]):
        """One ``Indexing`` row as a projection of a ``rank``-dim iteration nest.

        ``(0, 1, 2)`` of rank 4 is ``(d0, d1, d2, d3) -> (d0, d1, d2)``: the row is
        one dim index per result position, so the map has ``rank`` dims, no
        symbols, and one expression per entry.  Returns the map itself and not an
        ``ir.AffineMapAttr`` -- ``indexing_maps`` takes maps, and an attribute
        raises there.
        """
        return ir.AffineMap.get(
            rank, 0, [ir.AffineExpr.get_dim(int(dim)) for dim in row]
        )

    # -- module scaffolding ------------------------------------------------

    @contextlib.contextmanager
    def open_kernel(self, kernel_name: str) -> Iterator[None]:
        """Open the kernel func with its views bound, and emit its body into it.

        ``module { func.func @kernel_name(...) { %c0, one memory view per buffer,
        <body>, return } }``.  The signature and the views are two faces of one
        decision -- where a base address comes from -- so they are made together
        here rather than in two functions a caller has to order correctly.  All of
        it comes off ``self.plan``, the plan this builder was created for.

        Baked bases need no func arguments and appear as ``arith.constant``s;
        symbolic bases are one ``index`` parameter each, in ``plan.parameters``
        order.  Deleting the baked arm reverts the dataflow-scheduler#65
        workaround.
        """
        baked = self.plan.options.bake_addresses
        buffers = self.plan.parameters
        # One base address per buffer, in the plan's order, or none at all.
        params = [] if baked else [self.index_t] * len(buffers)
        try:
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                # [] is the result list: a KTIR kernel returns nothing.
                fn = func.FuncOp(kernel_name, ir.FunctionType.get(params, []))
                i64 = ir.IntegerType.get_signless(64)
                # The plan resolved the grid; a core count is not the builder's
                # business.
                fn.attributes["grid"] = ir.ArrayAttr.get(
                    [ir.IntegerAttr.get(i64, int(g)) for g in self.plan.grid]
                )
                block = fn.add_entry_block()
                self.block_args = list(block.arguments)
                with ir.InsertionPoint(block):
                    self.c0 = self.icst_index(0)
                    self.env.bind_ivs(self.core_portions())
                    for position, buffer in enumerate(buffers):
                        if not baked:
                            base = self.block_args[position]
                        else:
                            # ``_buffer`` resolves an address for every buffer
                            # under this option, so a missing one is a plan bug.
                            assert buffer.base_elements is not None, (
                                f"baked plan without an address for {buffer.buf_id}"
                            )
                            base = self.icst_index(buffer.base_elements)
                        self.views[buffer.buf_id] = self.memory_view(base, buffer)
                    yield
                    func.ReturnOp([])  # no operands, matching the signature
            # Printed while the context is still alive.
            self._text = str(module)
        finally:
            self._stack.close()

    def core_portions(self) -> list:
        """This core's index along each division, outermost-first.

        One ``ktdp.get_compute_tile_id`` -- the flat grid index -- read as the
        mixed-radix number the plan's divisions describe: ``(id // inner) % div``.
        A term whose factor is trivial is not emitted, so a single divided symbol
        uses the id itself and the undivided case emits nothing at all.
        """
        if not self.plan.divisions:
            return []
        # A result *list*: the op is variadic in the bindings, single-result here.
        tile_id = self.val(ktdp.get_compute_tile_id([self.index_t]))
        portions = []
        for division in self.plan.divisions:
            index = tile_id
            if division.inner > 1:
                index = self.val(arith.DivUIOp(index, self.icst_index(division.inner)))
            if division.inner * division.div != self.plan.grid[0]:
                index = self.val(arith.RemUIOp(index, self.icst_index(division.div)))
            portions.append(index)
        return portions

    def finish(self) -> str:
        """The canonical MLIR text of the module built by ``open_kernel()``."""
        if self._text is None:
            raise AssertionError("KtirBuilder.finish() before open_kernel() completed")
        return self._text

    @contextlib.contextmanager
    def counted_loop(self, trip: int) -> Iterator[Any]:
        """``scf.for`` to ``trip`` step 1, yielding its induction variable.

        Everything emitted while the context is open goes in the loop body, and
        the terminator is closed on the way out: ``scf.for`` regions are not
        implicitly terminated by the builders, and the loop carries no iter_args
        because every value the body produces is stored to memory inside it.
        """
        lo, step = self.icst_index(0), self.icst_index(1)
        hi = self.icst_index(int(trip))
        for_op = scf.ForOp(lo, hi, step)
        with ir.InsertionPoint(for_op.body):
            yield for_op.induction_variable
            scf.YieldOp([])

    # -- the walk ----------------------------------------------------------

    def emit(self, steps: Sequence[Step]) -> None:
        """Emit a step list at the current insertion point.  Recursive.

        The whole emission: a loop opens a loop and recurses, anything else is a
        compute step.  There is no third case and nothing to decide -- the plan
        decided it -- so this walk raises nothing but the assertion that says the
        plan is malformed.
        """
        for step in steps:
            if isinstance(step, LoopStep):
                with self.counted_loop(step.trip) as iv, self.env.scope(iv=iv):
                    self.emit(step.body)
            elif isinstance(step, ComputeStep):
                self.compute(step)
            else:
                # Every step a plan can hold is handled above, so reaching here
                # is a plan bug, not an unsupported request.  AssertionError (not
                # TypeError) says exactly that.
                raise AssertionError(  # noqa: TRY004
                    f"unplanned step {type(step).__name__}"
                )

    def compute(self, step: ComputeStep) -> None:
        """Read the operands, emit the planned surface, dispose of the result.

        A ``match`` over literal calls rather than a dispatch table, because a
        table's arms are unreachable to the call-graph walk that asserts nothing
        on this path can refuse: ``self.SURFACES[step.surface](...)`` has no
        resolvable callee, so every surface would have to be declared a root
        instead of being *reached*.  The cost is that a new surface is two edits
        (a method and an arm), and a test parses this ``match`` to catch the
        second one being forgotten.
        """
        arm = self.RECIPES[step.op].arm(step.dtype)
        ins = [self.operand(buf_id, access) for buf_id, access in step.ins]
        match step.surface:
            case Surface.BARE:
                value = self._emit_bare(arm.binding(), ins, step)
            case Surface.REDUCE:
                value = self._emit_reduce(arm.binding(), ins, step)
            case Surface.GENERIC:
                value = self._emit_generic(arm.binding(), ins, step)
            case _:
                raise AssertionError(f"unplanned surface {step.surface} of {step.op!r}")
        self.result(step.out_buf_id, step.out if step.store else None, value)

    # -- ktdp shapes -------------------------------------------------------

    def memory_view(self, base, buffer: Buffer):
        """``ktdp.construct_memory_view`` for one buffer, at base address ``base``.

        Extent and strides come from ``buffer.layout``, the record ``_layout``
        derived, so the view says what the plan says, in whole element counts.
        """
        sizes = [int(e) for e in buffer.layout.extent]
        strides = [int(s) for s in buffer.layout.strides]
        memref_t = ir.MemRefType.get(sizes, self.named_type(buffer.elems.storage))
        # ``memory_space`` was the last attribute built as text, because no
        # builder was exposed for it; ktir-mlir-frontend#61 adds one, so it now
        # goes through the same verifier-checked API as everything else and a
        # rename breaks type checking rather than failing at runtime.
        #
        # The builder takes the tablegen-generated ``MemorySpaceKind``, not a
        # spelling, so the mapping names enum members.  ``global_`` carries the
        # trailing underscore mlir-tblgen adds to escape the Python keyword.
        # Keyed lookup rather than a fallback, so a space this mapping has not
        # been taught fails loudly instead of emitting an attribute the backend
        # cannot read.
        kind = {
            "HBM": ktdp.MemorySpaceKind.global_,
            "LX": ktdp.MemorySpaceKind.ct_local,
        }[buffer.space]
        memory_space = ktdp.MemorySpaceAttr.get(kind)
        return self.val(
            ktdp.construct_memory_view(
                result=memref_t,
                offset=base,
                # Every size is a literal, so both operand lists stay empty.
                sizes=[],
                strides=[],
                static_sizes=sizes,
                static_strides=strides,
                memory_space=memory_space,
                coordinate_set=self.coord_set(sizes),
            )
        )

    def access_tile(self, access: Access):
        """``ktdp.construct_access_tile`` for ``access``, into its buffer's view.

        The per-dim index is ``sum_l coeffs[i][l] * iv_l`` over the induction
        variables of the loops this builder has open -- the record holds the
        coefficients, the open loops supply the variables, and the two line up
        because the plan derived one coefficient per enclosing level.  A ``* 1``
        is not emitted, and an empty sum is the function-entry ``%c0`` rather than
        a fresh zero, so a dim no level walks indexes the view with the one zero.
        """
        sizes = list(access.extent)
        ivs = self.env.ivs()
        indices = []
        for coeffs in access.index_coeffs:
            terms = [
                iv if coeff == 1 else self.val(arith.MulIOp(iv, self.icst_index(coeff)))
                for coeff, iv in zip(coeffs, ivs, strict=True)
                if coeff
            ]
            indices.append(
                functools.reduce(
                    lambda lhs, rhs: self.val(arith.AddIOp(lhs, rhs)), terms
                )
                if terms
                else self.c0
            )
        identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(sizes)))
        # A threaded buffer has no view, and the plan gives it no access to tile:
        # reaching here without one is a plan bug, not an unsupported request.
        assert access.buffer is not None, "access tile of a buffer with no view"
        return self.val(
            ktdp.construct_access_tile(
                result=ktdp.AccessTileType.get(sizes, ir.IndexType.get()),
                base=self.views[access.buffer.buf_id],
                # How the view is indexed, and the order of the tile's own axes.
                # Both identity: the tile covers the view one-to-one.
                base_map=identity,
                access_tile_order=identity,
                indices=indices,
                # SSA operands for symbols in base_map; it uses none.
                symbol_operands=[],
                access_tile_set=self.coord_set(sizes),
            )
        )

    def operand(self, buf_id: str, access: Access):
        """An input operand's value: a live produced value, or an access + load.

        Reusing a produced value is what register-threaded fused intermediates
        will need; they are rejected today, so ``produced`` is always ``None``
        and this always loads.
        """
        produced = self.env.produced(buf_id)
        if produced is not None:
            return produced
        tensor_t = ir.RankedTensorType.get(
            list(access.extent), self.named_type(access.elems.value)
        )
        return self.val(
            ktdp.load(result=tensor_t, access_tile=self.access_tile(access))
        )

    def result(self, buf_id: str, access: Access | None, value) -> None:
        """Dispose of an op's result: thread it, or store it through ``access``.

        The mirror of ``operand`` on the way out.  ``access is None`` is an
        internal buffer: it has no view to store through, so the value is bound
        in scope for a later op to consume.
        """
        if access is None:
            self.env.bind_produced(buf_id, value)
        else:
            ktdp.store(data_tile=value, access_tile=self.access_tile(access))

    # -- compute -----------------------------------------------------------
    #
    # One entry per op, and it contributes only its dialect builder: what shape
    # that builder is wrapped in comes from ``step.surface``, which the plan chose.
    #
    # Bindings are dialect *functions*, not OpView classes: ``linalg.AddOp``
    # constructed directly leaves the named op's body region empty and fails
    # verification, while the OpDSL function generates that body.
    #
    # A repeated key here is ruff F601, so an op cannot be declared twice.
    RECIPES: ClassVar[dict[str, Recipe]] = {
        # ``add`` and ``mul`` are the two ops with more than one spelling: a named
        # linalg op at floats, and a ``spyreop`` intrinsic at four-byte integers
        # that splits its operands into halves and finds the carry with a pair of
        # scale factors.  The float arm lists no formats, so it takes every format
        # the integer arm does not claim.
        "add": Recipe(
            arity=2,
            arms=(
                Arm(kind=BindingKind.NAMED, binding=lambda: linalg.add),
                Arm(
                    kind=BindingKind.PAYLOAD,
                    binding=lambda: spyreop.addi32toi32,
                    dtypes=(DataFormats.IEEE_INT32,),
                ),
            ),
        ),
        "mul": Recipe(
            arity=2,
            arms=(
                Arm(kind=BindingKind.NAMED, binding=lambda: linalg.mul),
                Arm(
                    kind=BindingKind.PAYLOAD,
                    binding=lambda: spyreop.muli32toi32,
                    dtypes=(DataFormats.IEEE_INT32,),
                ),
            ),
        ),
        "sub": Recipe(
            arity=2, arms=Arm(kind=BindingKind.NAMED, binding=lambda: linalg.sub)
        ),
        "sum": Recipe(
            arity=1, arms=Arm(kind=BindingKind.COMBINER, binding=lambda: arith.addf)
        ),
        # The unary float ops whose payload is one ``spyreop`` scalar intrinsic.
        # There is no named linalg op behind any of them, so they are PAYLOADs and
        # land on ``Surface.GENERIC``: the recipe contributes the intrinsic and the
        # generic states the identity maps and the all-parallel iterators for it.
        #
        # The key is the pointwise-handler name the frontend already uses and the
        # binding is the ``spyreop`` op, which is why they differ for ``gelufwd``
        # -> ``spyreop.gelu``.  The intrinsic takes the tile's own f16 and owns its
        # f16->f32->approx->f16 internally, so the body is the one op with no
        # precision bracket around it (dataflow-scheduler#36).
        #
        # ``softplus`` is the one that takes more than its operand: ``attrs`` says
        # where in ``op_info`` its two scalars live.
        #
        # Not here: the remaining integer/address intrinsics (addi64toi64,
        # idx32toaddr) and other pointwise ops the device has no intrinsic for
        # (log, tanh, erf, relufwd).
        "exp": Recipe(
            arity=1, arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.exp)
        ),
        "sqrt": Recipe(
            arity=1, arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.sqrt)
        ),
        "sigmoid": Recipe(
            arity=1,
            arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.sigmoid),
        ),
        "reciprocal": Recipe(
            arity=1,
            arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.reciprocal),
        ),
        "gelufwd": Recipe(
            arity=1, arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.gelu)
        ),
        "layernormscale": Recipe(
            arity=1,
            arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.layernormscale),
        ),
        "softplus": Recipe(
            arity=1,
            arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.softplus),
            attrs=lambda info: {
                "beta": float(info["constants"]["softplusBeta"]),
                "threshold": float(info["constants"]["softplusThresh"]),
            },
        ),
        "silu": Recipe(
            arity=1, arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.silu)
        ),
        "rsqrt": Recipe(
            arity=1,
            arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.rsqrt),
        ),
        "realdiv": Recipe(
            arity=2,
            arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.realdiv),
        ),
    }

    # -- emission surfaces -------------------------------------------------
    #
    # One method per ``Surface``: the shape of the op that carries a recipe's
    # payload.  The surface owns the destination and the operand/result typing, so
    # the shape is written once however many kinds of payload reach for it, and it
    # chooses nothing -- the plan already did.
    #
    # ``linalg.reduce`` derives its indexing maps as the identity with the reduced
    # dimensions dropped, so it says only rank-reducing, identity-indexed
    # reductions.  A reduction over the stick axis is not one -- it reduces the
    # lane axis on the way in and keeps it on the way out -- which is why
    # ``Surface.GENERIC`` exists as a third shape rather than being folded in.
    #
    # Every surface writes its result into an uninitialised ``tensor.empty``: on
    # this path the destination is a pure destination -- the op writes every
    # element of it, or (a reduction) leaves materialising the identity to the
    # scheduler's reduction passes, for which a ``linalg.fill`` here would be a
    # second compute op to unpick.

    def _destination(self, step: ComputeStep):
        """``(extents, elt_t, dest)`` for ``step``'s result: what every surface needs.

        Emits the ``tensor.empty``, so a surface calls this once and first --
        before the op that writes into it, which is the order the module reads
        in.  Shared rather than repeated because the surfaces agree on it
        exactly, and one that disagreed would be describing a different
        destination, not a different shape.
        """
        extents = list(step.out.extent)
        elt_t = self.named_type(step.out.elems.value)
        return extents, elt_t, self.val(tensor.EmptyOp(extents, elt_t))

    def _emit_bare(self, build: Callable, ins: Sequence, step: ComputeStep):
        """``build`` called directly, shaped by the result tile.

        The surface for an op the dialect already names: no region to fill and no
        maps to state, because the named op's own definition says how its
        operands are indexed.  Every operand and the result share the result
        tile's extents, which is what makes the call legal without them.
        """
        extents, elt_t, dest = self._destination(step)
        return build(
            *ins,
            outs=[dest],
            result_tensors=[ir.RankedTensorType.get(extents, elt_t)],
        )

    def _emit_reduce(self, combine: Callable, ins: Sequence, step: ComputeStep):
        """``linalg.reduce`` over ``step.reduce_dims``, folding with ``combine``.

        The surface for a reduction that drops whole axes: ``linalg.reduce``
        indexes its operands for you and its region is fixed at two scalars, so
        all it needs is the dimensions and a combiner -- there is no room for the
        payload to be anything else, which is why it takes the two-argument
        function and builds the region itself.  Reducing in place also means no
        reshape: keeping the surviving axes in the output tile makes the result
        the shape the store's access tile already has.
        """
        extents, elt_t, dest = self._destination(step)

        def body(accumulated, element):
            return combine(accumulated, element)

        # The region builder reads the block argument types off the annotations,
        # and the element type is only known here, so they are set rather than
        # written.
        body.__annotations__ = {"accumulated": elt_t, "element": elt_t}
        return linalg.reduce(
            result=[ir.RankedTensorType.get(extents, elt_t)],
            inputs=list(ins),
            inits=[dest],
            dimensions=list(step.reduce_dims),
        )(body)

    def _emit_generic(self, payload: Callable, ins: Sequence, step: ComputeStep):
        """``linalg.generic`` stating the plan's maps and iterators.

        The surface for a nest nothing else can spell.  Two callers reach it and
        one body serves both: the block arguments are one per input and then the
        ``outs`` accumulator, which a reducing nest folds into and a parallel one
        drops -- so the arity works out either way (one input plus an accumulator,
        or two inputs with the accumulator dropped, both two arguments).

        Only ``dest`` is taken from ``_destination``: a generic's result type comes
        from its ``outs`` operand rather than being stated, and unlike
        ``_emit_reduce`` the region's block argument types are appended by the
        builder off the operand element types, so there is no annotation to set.

        ``step.attrs`` are the payload's non-operand scalars, passed as keyword
        arguments.  They were read from the spec's ``op_info`` when the step was
        planned, so nothing here knows what an op's attributes mean.
        """
        indexing = step.indexing
        # Every GENERIC step carries one (the plan's field invariant); reaching
        # here without it is a plan bug, not an unsupported request.
        assert indexing is not None, f"generic {step.op!r} with no indexing record"
        _extents, _elt_t, dest = self._destination(step)
        rank = len(indexing.iters)
        reducing = bool(step.reduce_dims)
        attrs = dict(step.attrs)

        def body(*args):
            return payload(*(args if reducing else args[:-1]), **attrs)

        return linalg.generic(
            inputs=list(ins),
            outputs=[dest],
            indexing_maps=[self._affine_map(rank, row) for row in indexing.maps],
            iterator_types=list(indexing.iters),
        )(body)

    # -- attributes --------------------------------------------------------

    @staticmethod
    def coord_set(sizes: Sequence[int]):
        """Per-dim bounding integer set ``(0 <= d_i <= size_i - 1)`` as an attribute.

        Built with ``ir.IntegerSet`` from ``AffineExpr`` constraints (no textual
        round-trip): for each dim ``i`` two inequalities ``d_i >= 0`` and
        ``-d_i + (size_i - 1) >= 0``, matching the ``affine_set`` MLIR prints.

        Every size is a constant here, so every bound is one too.
        """
        exprs = []
        eq_flags: list[bool] = []
        for i, size in enumerate(sizes):
            dim = ir.AffineExpr.get_dim(i)
            # d_i >= 0
            exprs.append(dim)
            eq_flags.append(False)
            # -d_i + (size_i - 1) >= 0
            bound = ir.AffineExpr.get_constant(int(size) - 1)
            neg_dim = ir.AffineExpr.get_mul(ir.AffineExpr.get_constant(-1), dim)
            exprs.append(ir.AffineExpr.get_add(neg_dim, bound))
            eq_flags.append(False)
        integer_set = ir.IntegerSet.get(len(sizes), 0, exprs, eq_flags)
        return ir.IntegerSetAttr.get(integer_set)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_ktir(
    kernel_name: str,
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
    **options,
) -> str:
    """Build a KTDP-dialect MLIR module for ``specs`` and return ``str(module)``.

    ``specs`` is the finished OpSpec kernel contract (the same value
    ``call_kernel`` passes positionally to ``.run(...)``).  Func parameters are
    the unique operand buffers in ascending ``arg_index`` order so the emitted
    signature matches that positional binding (or, in the baked form, no
    parameters at all and one ``arith.constant`` base address per buffer).

    Three steps: plan the kernel (which raises every rejection), open it, emit
    its steps.  The plan completes before ``KtirBuilder.create``, so an
    unsupported request fails fast -- and is testable -- whether or not
    ``mlir_ktdp`` is installed; and the emission consumes only the plan, so a
    request that got that far cannot be refused half-emitted.

    ``options`` are ``PlanOptions`` fields, spelled as keywords so a caller
    passes only what it chooses and the defaults live in one place.
    """
    known = {f.name for f in dataclasses.fields(PlanOptions)}
    if unknown := sorted(set(options) - known):
        raise TypeError(f"generate_ktir: unknown option(s) {unknown}")

    plan = build_kernel_plan(specs, PlanOptions(**options))
    b = KtirBuilder.create(plan)
    with b.open_kernel(kernel_name):
        b.emit(plan.steps)
    return b.finish()
