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
import logging
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
    operand_indexing,
    per_core_extent,
    placeholder_axes,
    reduction_indexing,
    row_major_strides,
)
from torch_spyre._inductor.constants import IDENTITY_OP, STAGGERED_EAS
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.pass_utils import coeff_through_floor

# The module's one logger, and the reason it has one is ``PlanFusion``: a table
# that declines silently is worse than an extra import, and the resources a
# fusion strands have to be reported somewhere.
# ``logging_utils`` is not ``mlir_ktdp``, so the constraint this module actually
# carries -- ``build_kernel_plan`` imports no dialect -- is untouched.
logger = get_inductor_logger("codegen.ktir")

# The dialect handles: one module-level name each, None until _load_dialects()
# binds them.  Under TYPE_CHECKING they are the real imports, so `ir.Module` and
# `linalg.add` carry types; at runtime the block does not execute, so importing
# this module requires no dialect build.
if TYPE_CHECKING:
    from mlir_ktdp import ir
    from mlir_ktdp.dialects import (
        arith,
        func,
        ktdp,
        linalg,
        math,
        scf,
        spyreop,
        tensor,
    )
else:
    ir = arith = func = ktdp = linalg = math = scf = spyreop = tensor = None


def _load_dialects() -> None:
    """Bind the dialect handles into this module, once.  The only import site."""
    global ir, arith, func, ktdp, linalg, math, scf, spyreop, tensor
    if ir is not None:
        return
    from mlir_ktdp import ir as _ir
    from mlir_ktdp.dialects import arith as _arith
    from mlir_ktdp.dialects import func as _func
    from mlir_ktdp.dialects import ktdp as _ktdp
    from mlir_ktdp.dialects import linalg as _linalg
    from mlir_ktdp.dialects import math as _math
    from mlir_ktdp.dialects import scf as _scf
    from mlir_ktdp.dialects import spyreop as _spyreop
    from mlir_ktdp.dialects import tensor as _tensor

    ir, arith, func, ktdp, linalg, math, scf, spyreop, tensor = (
        _ir,
        _arith,
        _func,
        _ktdp,
        _linalg,
        _math,
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
# depending on which backend build and which device spec it meets (``verify.py``
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

    ``FUSED`` is the second key: ``ElementArrangement.EXX2`` -- "reduction mode:
    two values per stick" -- is a buffer holding a mean and a mean of squares
    TOGETHER, which the dialect spells as one element of ``!spyreop.fp16_fused``
    rather than as two of ``f16``.  So the lookup is
    ``(device_dtype, element_arrangement) -> spelling``: the arrangement selects
    the table and the dtype the row.  MEASURED: ``ir.Type.parse`` resolves both
    fused spellings once ``ktdp.register_dialects`` has run, so ``named_type``
    needs nothing added for them.
    """

    NAMES: ClassVar[dict[DataFormats, str]] = {
        DataFormats.IEEE_FP16: "f16",
        DataFormats.SEN169_FP16: "f16",  # for now treating dl169 as cosmetic
        DataFormats.IEEE_FP32: "f32",
        DataFormats.BFLOAT16: "bf16",
        DataFormats.IEEE_INT32: "i32",
    }

    # Only the two formats the dialect has a fused spelling for.  An arrangement
    # this table has no row for is refused rather than silently unfused: a pair
    # read as a single float is the wrong half of a statistic, and it would
    # compile.
    FUSED: ClassVar[dict[DataFormats, str]] = {
        DataFormats.IEEE_FP16: "!spyreop.fp16_fused",
        DataFormats.SEN169_FP16: "!spyreop.fp16_fused",
        DataFormats.IEEE_FP32: "!spyreop.fp32_fused",
    }

    storage: str
    value: str

    @classmethod
    def of(cls, dtype: DataFormats, arrangement: Any = None) -> ElemTypes:
        """The storage/value pair for a device dtype and arrangement, or raise.

        One ``device_dtype`` means one type on both sides today; a load that
        reinterprets is why the record has two fields.

        ``arrangement`` is a *type* selection and not a stride adjustment (the
        pair is one element, so ``_arrangement_layout`` leaves the extent alone
        for it), which is why it is read here as well as there.
        """
        table = cls.FUSED if arrangement is ElementArrangement.EXX2 else cls.NAMES
        name = table.get(dtype)
        if name is None:
            raise NotImplementedError(
                f"OpSpec->KTIR: unsupported device dtype {dtype!r}"
                + (" at element arrangement EXX2" if table is cls.FUSED else "")
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
    """One ACCESS's view of a buffer; sole input to a memory view.

    Two kinds of field, and the difference is why there is a record per access
    rather than one per buffer: ``buf_id``, ``arg_index`` and ``base_elements``
    are IDENTITY and ADDRESS, which every access to the buffer shares (they key
    ``plan.parameters`` and ``KtirBuilder.bases``), while ``layout``, ``elems`` and
    ``space`` are how THIS access views it, and two stages legitimately differ (see
    ``KernelPlan._access_of`` for the measured case that needs this).

    ``KernelPlan.buffers`` holds one of these per ``buf_id`` -- the first seen --
    and what it is held for is the identity half: the signature has one parameter
    per buffer however many ways the stages view it.
    """

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
    planning -- and the builder zips them against the loops it has open.  The
    builder spells it the way hand-written loop kernels do, an identity ``base_map``
    with one index expression per view dim, rather than a non-identity map over the
    induction variables; the matrix is the same either way.

    ``elems`` is the access's own element type pair: a tile of an internal buffer
    has no ``Buffer`` to read one from, and a load that reinterprets would differ
    from its buffer's storage type anyway.  Its DERIVATION is the buffer's
    arrangement unless the recipe says the operand reads the buffer unfused
    (``Recipe.unfused``), which is why it is passed in rather than read off the arg.

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

    ``None`` in a row is the CONSTANT 0 result position: an axis the operand does
    not walk, read at its first element for every iteration.  It is what the three
    broadcast forms are made of -- ``(d0,d1,d2) -> (d1, 0)`` reads a statistic at
    the head of its stick, ``(d0,d1) -> (d0, 0)`` splats it back across one, and
    ``(d0,d1,d2) -> (d0, 0, d2)`` reads one row of a weight for every row of the
    output.  Zero and not an arbitrary constant, because a tile is placed by its
    access indices and an operand axis that walks nothing is read at the tile's own
    origin; a non-zero offset would be an addressing decision, and addressing is
    ``Access.index_coeffs``' business.

    A row is otherwise a bare dim index per position, which is every map in scope
    and not every map there is: a *linearised* map such as
    ``(d0, d1, d2, d3, d4) -> (d0, d2 * 64 + d3, d4)`` needs (coefficient, dim)
    terms, so nothing here generalises to one for free.
    """

    iters: tuple[str, ...]  # PARALLEL | REDUCTION, one per iteration dim
    maps: tuple[tuple[int | None, ...], ...]  # [operand][position] -> dim or const 0


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

    ``stage`` is this step's position in the kernel's stage order, counted over
    the whole step tree (loop bodies included) by ``KernelPlan._stages``.  One
    compute is one stage, which is the backend's own granularity: MEASURED, one
    compute becomes one ``local_schedule`` module.  It is on the step because the
    memory views a step tiles are per stage (see ``KtirBuilder.view``), so the
    emitter needs to know which stage is asking before it can answer with a view.
    """

    op: str  # a KtirBuilder.RECIPES key
    surface: Surface
    ins: tuple[tuple[str, Access], ...]
    out: Access
    out_buf_id: str
    store: bool
    stage: int = 0
    reduce_dims: tuple[int, ...] = ()
    indexing: Indexing | None = None
    attrs: tuple[tuple[str, float], ...] = ()
    # Which format's arm of the op this step wants.  Carried rather than
    # re-derived at emit time, where the args are no longer in reach: a step reads
    # no spec.  The format rather than the arm itself, because an arm holds a
    # deferred dialect reference and a step stays dialect-free.
    dtype: DataFormats | None = None
    # Whether any operand was broadcast against the output.  Carried for exactly
    # the reason ``dtype`` is: emission re-resolves the arm and has no spec in
    # reach to ask.  The two fields together are the ``Request`` the plan
    # dispatched on, so the arm emission resolves is the arm the surface on this
    # step was chosen from -- and neither field pulls a dialect handle onto the
    # step.  It is not derivable from ``surface`` alone: a GENERIC step is what a
    # broadcast operand forces, but also what any scalar PAYLOAD needs.
    broadcast: bool = False


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


def _arrangement(arg: TensorArg) -> Any:
    """``arg``'s element arrangement, read in one place.

    ``getattr`` because the field is a late addition to ``TensorArg`` and this
    module is handed args built by other layers; the default is the standard
    order, which is what an arg that does not carry the field means.
    """
    return getattr(arg, "element_arrangement", None)


def _arrangement_layout(
    arrangement: Any, extent: tuple[Any, ...], strides: tuple[Any, ...]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """``(extent, strides)`` adjusted for the element order within a stick.

    ``element_arrangement`` is an element *order*, not a dtype conversion (one
    ``device_dtype`` covers the data type), so it is a layout fact: a rank and
    stride selector, of the shape the SDSC path already uses for a stick split.

    Label: ``staggered-element-arrangement``.

    ``EXX2`` passes through with STANDARD, and that is the whole of its layout
    rule: a mean and a mean of squares held together are ONE element of
    ``!spyreop.fp16_fused``, so the buffer has the rank, extent and row-major
    strides its ``device_size`` states and the pair is a fact about the element
    TYPE (``ElemTypes.of``) rather than about the addressing.  MEASURED against a
    hand-written reference module for this op, whose fused output view is
    ``memref<256x64x!spyreop.fp16_fused>`` -- the same 256x64 an f16 output of
    that reduction would have, at the same strides.
    """
    if arrangement in (
        None,
        ElementArrangement.STANDARD,
        ElementArrangement.QFP8CH,
        ElementArrangement.EXX2,
    ):
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
        _arrangement(arg),
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


def _reads_stick_head(arg: TensorArg) -> bool:
    """Whether this INPUT reads a statistic sitting at the head of each stick.

    The signature of one: the innermost device axis carries a CONSTANT coordinate
    -- so no iteration dim walks it -- over a whole stick of elements.  That is
    what a reduction writes (MEASURED: our on-stick reductions produce
    ``[.., 64]`` at coordinate ``0``, because the hardware writes a whole stick at
    a time and the opaque reduction needs the rest of the stick to get the result
    to element 0), and a consumer of it wants the one element at the head.

    Asked of INPUTS only, and the producer is why: its output has exactly this
    shape and must keep writing all 64 lanes.

    Not "extent 1 already": a tile of one element is not a statistic read, it is
    an operand somebody already described that way, and there is nothing to narrow.

    A COARSE-TILED arg carries no coordinates at all (it addresses through
    ``device_tile_advance_expr``), so there is nothing here to read and it is not
    one of these: the question is about a coordinate, not about an extent.
    """
    if not arg.is_input or not len(arg.device_coordinates):
        return False
    coord = arg.device_coordinates[-1]
    return not getattr(coord, "free_symbols", None) and int(arg.device_size[-1]) > 1


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


def _reduction_nest(
    spec: OpSpec,
) -> tuple[TensorArg, tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """The iteration nest ``spec``'s reduction asks for, derived once.

    ``(out, iters, in_map, out_map)``, where ``out`` is the output arg with its
    placeholder axes squeezed away -- the same arg every later derivation in
    ``_compute_step`` uses, which is why the squeeze belongs here rather than
    beside the caller: a second derivation of this nest would be a second answer
    to drift from.

    Roles are read straight off ``args`` rather than through
    ``validated_roles``, which asks ``RECIPES`` for the arity: this runs on
    prospective fusion survivors too, and a fusion table must be able to ask
    about a spec before deciding to give it a name the table has a recipe for.
    Every reduction in scope is unary, so a non-unary one is refused here rather
    than unpacked.
    """
    inputs = [arg for arg in spec.args if arg.is_input]
    outputs = [arg for arg in spec.args if not arg.is_input]
    if len(inputs) != 1 or len(outputs) != 1:
        raise NotImplementedError(
            f"OpSpec->KTIR: reduction {spec.op!r} takes {len(inputs)} input(s) and "
            f"writes {len(outputs)} output(s); a reduction here is one of each"
        )
    [source], [out] = inputs, outputs
    placeholder = placeholder_axes(
        out.device_coordinates, [int(s) for s in out.device_size]
    )
    if placeholder:
        # The projection leaves an axis the op does not write in the output as a
        # unit extent; the reduced tile does not have it at all.
        out = _squeezed(out, placeholder)
    iters, in_map, out_map = reduction_indexing(
        source.device_coordinates,
        [int(s) for s in source.device_size],
        out.device_coordinates,
        [int(s) for s in out.device_size],
    )
    return out, tuple(iters), tuple(in_map), tuple(out_map)


def _reduction_surface(spec: OpSpec) -> Surface:
    """Which surface ``_compute_step`` will choose for this reduction.

    Extracted from ``_compute_step`` so the fusion table's viability predicates
    can ask the question before the step exists, and so there is exactly ONE
    derivation of it: if a second derivation drifts from this one, a fusion admits
    a form the device computes wrongly.

    For a reduction, ``Surface.GENERIC`` is "on-stick": the within-stick axis is
    among the reduced dims, so the nest is not what ``linalg.reduce`` means.
    """
    _out, iters, in_map, out_map = _reduction_nest(spec)
    return _reduce_surface(iters, in_map, out_map)


def _parallel_surface(
    arm: Arm, operands: int, rank: int
) -> tuple[Surface, Indexing | None]:
    """Which shape carries a non-reducing payload whose operands are ALIGNED.

    Nothing is *derived* here and nothing needs to be: ``align_reshape_plan`` has
    already answered that every operand's coordinates and extents equal the
    output's, which is precisely the identity condition, so the maps are known
    rather than read off the coordinates.  Deriving them instead would make the
    emitted form of ``add`` hostage to the dim-reuse rule ``reduction_indexing``
    needs -- a coordinate list that repeated a classification would yield a
    non-identity map and silently turn a ``linalg.add`` into a ``linalg.generic``.

    So the choice is only about spelling, and it follows from the binding: a
    ``NAMED`` builder is an op the dialect already has, which says its own
    indexing and needs no record, while anything else has to state the identity
    maps and the all-parallel iterators itself -- which only a generic can do.
    That second arm is where a ``spyreop`` intrinsic lands: it is a scalar builder,
    so there is nothing to call it but a region.

    ``_broadcast_surface`` is the other arm of the same question, for operands
    alignment says do NOT match the output.
    """
    if arm.kind is BindingKind.NAMED:
        return Surface.BARE, None
    identity = tuple(range(rank))
    return Surface.GENERIC, Indexing(
        iters=(PARALLEL,) * rank,
        maps=(identity,) * (operands + 1),
    )


def _broadcast_surface(
    arm: Arm,
    out: TensorArg,
    inputs: Sequence[TensorArg],
    accesses: dict[str, Access],
) -> tuple[Surface, Indexing | None]:
    """The same question for operands that do NOT all match the output.

    ``align_reshape_plan`` is the switch between this and ``_parallel_surface``:
    it answers ``None`` exactly when an operand's coordinates and extents are the
    output's, and an operand it has something to say about is one whose map row has
    to be read off the coordinates (``operand_indexing``).

    Every row is derived, including the aligned operands' (which come back as the
    identity), because ``indexing_maps`` is one attribute: a generic states a row
    per operand or none at all.  The result's row is the identity, because a
    pointwise op writes every element of its output once.

    The extents compared are the TILE extents from ``accesses``, not the buffers':
    a ``linalg`` operand's shape is what was loaded, and a broadcast operand is
    loaded at one element on the axis it does not walk.
    """
    out_extent = accesses[buf_id(out)].extent
    rank = len(out_extent)
    rows = tuple(
        operand_indexing(
            list(arg.device_coordinates),
            accesses[buf_id(arg)].extent,
            list(out.device_coordinates),
            out_extent,
        )
        for arg in inputs
    )
    if arm.kind is BindingKind.NAMED:
        # A named linalg op states its own indexing, which is the identity, so
        # there is nowhere to put a derived row.  Refused rather than forced into a
        # generic: what would go in that generic's body is a scalar spelling of the
        # op (``arith.subf`` for ``linalg.sub``) that no recipe declares, and
        # calling a whole-op builder with two scalars does not build.
        raise NotImplementedError(
            f"OpSpec->KTIR: operand of {out.name!r}'s op is broadcast against the "
            f"output ({rows}), but the op is a named linalg op, which states its "
            "own indexing; broadcast operands are supported on ops whose payload "
            "is a scalar builder"
        )
    identity = tuple(range(rank))
    return Surface.GENERIC, Indexing(iters=(PARALLEL,) * rank, maps=(*rows, identity))


def _access(
    arg: TensorArg,
    extent: Sequence[Any],
    rows: Sequence[Sequence[int]],
    layout: Layout,
    elems: ElemTypes,
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
        elems=elems,
        buffer=buffer,
    )


# ---------------------------------------------------------------------------
# KernelPlan: everything the builder is given
# ---------------------------------------------------------------------------
#
# The plan is the whole instruction list: the grid, the buffers whose func
# parameters the kernel opens with, and the step tree that goes in its body.
# It is built by one walk of the spec tree, which is where the derivations run
# and therefore where every rejection is raised.  Emission consumes the plan and
# reads no spec, so it cannot discover a reason to refuse half-way through.


# The ``allocation`` keys memory planning uses for a buffer the kernel owns.
INTERNAL_SPACES: tuple[str, ...] = ("lx", "hbm_pool")

# Ops whose whole content is a PLACEMENT: no arithmetic, no reindexing, not even
# a format change, only a statement about where the data should live.  In KTIR
# that is something a reading stage's own memory view and access tile already
# say, so such a spec becomes no stage at all -- its readers access its source
# instead (``KernelPlan._hint``).  MEASURED, the ``identity`` the frontend puts
# in front of ``F.layer_norm`` is exactly this: same ``device_size``, same
# ``device_coordinates``, same ``device_dtype``, and the only difference is
# ``allocation`` (``hbm,0`` -> ``lx,-1``).
#
# A TABLE and not a structural test, and that is load-bearing rather than
# stylistic.  "One input, one output, access-preserving, allocation differs" is
# also true of a perfectly ordinary ``sqrt`` writing an LX intermediate, so
# structure alone cannot tell a placement from a computation -- only the op name
# can.  Membership here is the claim "this op computes nothing"; every condition
# in ``_hint`` is then a REFUSAL, because a spec that claims to be a placement
# and is not describes real data movement.
ACCESS_ONLY_OPS: frozenset[str] = frozenset({IDENTITY_OP})


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
    the allocation it was given, while one KTIR func for the whole kernel CAN
    keep it as an SSA value -- no store, no view, no parameter, and no address
    for the scheduler to honour.

    Can, not must, and the difference is why this predicate is not the same
    question as "how is this buffer handled": threading only carries a value that
    never crosses a compute stage, and ``_check_threaded_buffers`` states what
    happens to the rest.
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
    workaround that the backend compiler requires.  The SDSC path makes the same choice from
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
        # ``link buf_id -> the SOURCE's Buffer``, filled by ``_hint`` and consulted
        # by ``_access_of``: a read of an access-only spec's result is built as an
        # access to its source instead.  Identity and address are what is taken
        # from it -- the reader keeps its own geometry and element type (T5).
        self.hints: dict[str, Buffer] = {}
        # The post-fusion vector this plan was filled from, held so ``_hint`` can
        # count a link's readers and writers across the WHOLE kernel rather than
        # across the one (possibly nested) list ``_stages`` is recursing over.
        self._vector: tuple[Any, ...] = ()
        self.steps: tuple[Step, ...] = ()
        # Handed out by ``_stages``, one per ``ComputeStep``, across the whole
        # tree: a step in a loop body is as much a stage as a top-level one, and
        # the count is what says how many stages the kernel has.
        self._next_stage = 0

    @property
    def parameters(self) -> list[Buffer]:
        """External buffers in ascending ``arg_index``.

        Ascending ``arg_index`` matches the positional order ``call_kernel``
        passes to ``.run(...)``, so the emitted func signature lines up with
        that binding.

        KNOWN GAP: a plan-time fusion can delete a buffer the caller still
        passes, so with the planners off this list can be one shorter than the
        call site's argument list.  Invisible on the path that runs, whose bases
        are baked constants and whose func takes no arguments.
        """
        return sorted(
            (e for e in self.buffers.values() if e.arg_index >= 0),
            key=lambda e: e.arg_index,
        )

    def add_specs(self, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]) -> None:
        """Plan ``specs`` into this plan's grid, buffers and steps."""
        # FIRST, and before ``_divisions``: fusing first is what makes the grid a
        # fact about the ops the kernel actually runs.  ``_divisions`` insists every
        # op ask for the same division, and the two specs of an absmax pair name
        # theirs in different symbol namespaces, so a divided pair is
        # self-contradictory right up until the fusion deletes one of them.  The
        # result is held, so ``_stages`` walks the same vector ``_divisions`` saw.
        specs = apply_plan_fusions(specs)
        self._vector = tuple(specs)
        self._symbols, self.divisions = _divisions(specs)
        self._divisors = {
            symbol: division.div
            for symbol, division in zip(self._symbols, self.divisions, strict=True)
        }
        cores = 1
        for division in self.divisions:
            cores *= division.div
        self.grid = (cores,)
        self._next_stage = 0
        self.hints.clear()
        self.steps = self._stages(specs, ())
        self._check_threaded_buffers(self.steps)

    def _check_threaded_buffers(self, steps: Sequence[Step]) -> None:
        """A threaded buffer must be produced before it is read, and then read.

        A threaded value has no memory behind it, so the kernel has to contain
        both ends of it.  Either end missing means the intermediate reached this
        kernel without the op on the other side of it -- the fusion decision and
        the kernel boundary disagree -- and the buffer needs materialising
        instead.  Refused here rather than emitted: an unread producer would
        silently write nowhere, and an unproduced consumer has no value to read.

        Both ends present is not sufficient: a threaded value must also not cross
        a COMPUTE STAGE, which is asked here too.  MEASURED: the backend compiler *aborts*
        on such a kernel rather than refusing it, so letting one through returns a
        crash rather than a diagnosis.

        The refusal names the two planning flags because they are the fix.  A
        buffer is threaded only because memory planning CLAIMED it (``lx`` /
        ``hbm_pool``), which is what keeps it out of ``spyre_kernel_args`` at
        ``arg_index == -1``; MEASURED with both planners off, the same intermediate
        arrives as an ordinary ``hbm`` buffer the wrapper allocates and passes, and
        a two-stage kernel then compiles and runs correctly with no emitter change.
        """
        unread: dict[str, None] = {}  # threaded, produced, not yet read
        produced: set[str] = set()
        # The stage each threaded buffer was produced in, so that a read from a
        # different one is recognised.  The stage counter runs across the whole
        # tree, so "another stage" is exactly "another step", loop bodies included.
        produced_in: dict[str, int] = {}

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
                    if produced_in[read_id] != step.stage:
                        raise NotImplementedError(
                            f"OpSpec->KTIR: buffer {read_id!r} is an intermediate "
                            "this kernel owns, so it is threaded as a value -- but "
                            f"it is written in stage {produced_in[read_id]} and read "
                            f"in stage {step.stage}, and a value cannot cross a "
                            "compute stage: the backend aborts on it. Memory "
                            "planning claimed this buffer, which is what makes it "
                            "threaded; set LX_PLANNING=0 and HBM_POOL_PLANNING=0 so "
                            "it stays an ordinary HBM buffer that the wrapper "
                            "allocates and passes, and this kernel emits a store "
                            "and a load instead"
                        )
                    unread.pop(read_id, None)
                if not step.store:
                    produced.add(step.out_buf_id)
                    produced_in[step.out_buf_id] = step.stage
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

    def _stages(self, specs, loops: Sequence[LoopSpec]) -> tuple[Step, ...]:
        """Recursive: the steps for one spec list, inside the ``loops`` chain.

        ``loops`` is the enclosing ``LoopSpec`` chain, outermost-first, which is
        what ``_levels`` zips ``OpSpec.tiled_symbols`` against.  A nested list
        becomes a nested ``LoopStep.body``, so the step tree's nesting is the
        spec tree's nesting and the emitter never has to work out the depth.

        Named for the stage rather than the step because numbering the stages is
        what this walk does that emission cannot: a spec is a stage, and only the
        walk sees the specs in one order across the nesting.  ``self._next_stage``
        rather than a parameter, so the recursion cannot restart the count in a
        loop body and hand two stages the same number.
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
                    LoopStep(trip=trip, body=self._stages(entry.body, [*loops, entry]))
                )
                continue
            if not isinstance(entry, OpSpec):
                raise NotImplementedError(
                    f"OpSpec->KTIR: unexpected spec entry {type(entry).__name__}"
                )
            if entry.op in ACCESS_ONLY_OPS:
                # No stage, no step, no stage number: this spec's one fact is
                # recorded for its readers and the spec itself is gone.  Before
                # the recipe lookup, because a placement deliberately has no
                # recipe -- giving it one would emit a copy the hardware does not
                # need.
                self._hint(entry, _levels(entry, loops))
                continue
            if entry.op not in KtirBuilder.RECIPES:
                raise NotImplementedError(
                    f"OpSpec->KTIR: op {entry.op!r} is not supported yet "
                    f"(registered: {sorted(KtirBuilder.RECIPES)})"
                )
            # The RECIPE and not an arm: reduction-ness is an op fact, which every
            # arm agrees on, and asking it without an arm is what lets the ONE arm
            # selection happen later in ``_compute_step``, once the operands have
            # been squeezed and it is known whether any of them is broadcast.
            recipe = KtirBuilder.RECIPES[entry.op]
            if recipe.reduces != bool(entry.is_reduction):
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
                    f"{_arms(recipe.arms)[0].kind.name} but this spec asks for "
                    f"{'a reduction' if entry.is_reduction else 'an elementwise op'}"
                )
            stage = self._next_stage
            self._next_stage += 1
            steps.append(self._compute_step(entry, loops, stage))
        return tuple(steps)

    def _hint(self, spec: OpSpec, levels: Sequence[Level]) -> None:
        """Record ``spec``'s placement for its readers, and emit no step for it.

        An access-only spec (``ACCESS_ONLY_OPS``) carries no computation: its
        input and output agree on ``device_size``, ``device_coordinates`` and
        ``device_dtype`` and differ only in ``allocation``.  There is nothing to
        lower -- the only thing an emitted stage could do with it is copy a buffer
        to itself at a different address -- so the right treatment is to let its
        readers access the SOURCE, which is what their own memory view, access
        tile and load already do.  MEASURED: a hand-written reference chain views
        its input directly in two stages and contains no staging op at all.

        This is NOT ``_collapse_producer``.  That rewrite deletes a producer into
        ONE surviving consumer and declines when the link has more than one reader,
        because a device primitive absorbs the producer's *work* and work can only
        be absorbed once.  Here there is no work, so removing the spec rewires EVERY
        reader, and several readers is the normal case rather than the exception.

        Every condition below is a refusal rather than a decline, because
        membership in ``ACCESS_ONLY_OPS`` is already the claim that this spec
        computes nothing.  A spec that makes that claim and does not meet it is
        describing real data movement -- a restickify, a relayout, a conversion --
        and it must be emitted or refused, never hinted away.

        What is DISCARDED is the hint itself: it said "stage this in LX", and the
        readers now read from wherever the source lives.  Nothing is reclaimed --
        this path issues no device allocate for ``lx`` or the pool in any case --
        and preserving the intent would mean viewing the link at ``memory_space =
        ct_local``, which is spellable except that an ``lx`` buffer has no address.
        """
        inputs = [arg for arg in spec.args if arg.is_input]
        outputs = [arg for arg in spec.args if not arg.is_input]
        if len(inputs) != 1 or len(outputs) != 1:
            # A spec with two inputs is computing something, whatever it is called.
            raise NotImplementedError(
                f"OpSpec->KTIR: op {spec.op!r} carries no computation, so it is "
                f"resolved as a placement hint, but it has {len(inputs)} input(s) "
                f"and {len(outputs)} output(s); a hint relates exactly one source "
                "to one link"
            )
        source, link = inputs[0], outputs[0]
        if not _access_preserving(source, link):
            # The one condition whose absence would be silent: a conversion or a
            # relayout has a source its readers do NOT already describe.
            raise NotImplementedError(
                f"OpSpec->KTIR: op {spec.op!r} does not write "
                f"{buf_id(link)!r} where it read {buf_id(source)!r} -- "
                f"size {list(source.device_size)} -> {list(link.device_size)}, "
                f"coordinates {list(source.device_coordinates)} -> "
                f"{list(link.device_coordinates)}, format "
                f"{source.device_dtype!r} -> {link.device_dtype!r} -- so it is "
                "real data movement rather than a placement, and this emitter "
                "does not emit it"
            )
        if not is_internal(link):
            # Only then are this kernel's reads all the reads there are: a buffer
            # in ``hbm`` may be read by a kernel this vector does not contain.
            raise NotImplementedError(
                f"OpSpec->KTIR: op {spec.op!r} places {buf_id(link)!r}, whose "
                f"allocation is {link.allocation!r} rather than one this kernel "
                "owns, so another kernel may read it and dropping the placement "
                "would leave that reader with a buffer nothing writes"
            )
        writers = _writers(buf_id(link), self._vector)
        if len(writers) != 1:
            raise NotImplementedError(
                f"OpSpec->KTIR: {buf_id(link)!r} is written by {len(writers)} ops "
                f"in this kernel ({sorted({w.op for w in writers})}), so dropping "
                f"the placement {spec.op!r} would leave the other writer's data "
                "being read through a buffer nothing wrote"
            )
        # The source's own record: identity and address, which is all a reader
        # takes from the map.  Its geometry and element type are replaced per
        # access in ``_access_of``, so a reader that views the buffer at another
        # element type (T5) is unaffected by which stage happened to be first.
        layout, _q = _solve_layout(source, levels)
        buffer = _buffer(
            source,
            layout,
            ElemTypes.of(source.device_dtype, _arrangement(source)),
            bake_addresses=self.options.bake_addresses,
        )
        self.buffers.setdefault(buffer.buf_id, buffer)
        self.hints[buf_id(link)] = buffer

    def _compute_step(
        self, spec: OpSpec, loops: Sequence[LoopSpec], stage: int
    ) -> ComputeStep:
        """One op: roles/arity, aliasing, alignment, its buffers and its accesses.

        Every derivation for this op runs here, once: the layout and per-level
        steps solved for an arg are the ones its buffer records and the ones its
        access is built from, so a view and the tiles into it cannot disagree, and
        emission has nothing left to derive.
        """
        out, inputs = validated_roles(spec)
        dtype = dtype_of(spec)
        recipe = KtirBuilder.RECIPES[spec.op]
        for arg in inputs:
            # In-place (input buffer aliases the output) is not supported yet.
            if buf_id(arg) == buf_id(out):
                raise NotImplementedError(
                    "OpSpec->KTIR: in-place ops (input aliases output) not supported"
                )
        reduce_dims: tuple[int, ...] = ()
        indexing: Indexing | None = None
        if spec.is_reduction:
            # What iteration nest a reduction wants is a fact about its operands'
            # coordinates, so it is derived here (once) and carried on the step,
            # not re-derived from the op name at emit time.
            squeezed, iters, in_map, out_map = _reduction_nest(spec)
            if squeezed is not out:
                # ``_reduction_nest`` squeezed the output's placeholder axes away.
                # Substituting the squeezed arg here, once and before
                # ``_access_of``, is what keeps every derivation after this point
                # unaware that a reduction is different: the output's view, tile,
                # per-core division and stored tensor are all the same (lower)
                # rank.  It stays gated on ``is_reduction`` because an *accepted*
                # pointwise spec can carry a unit constant axis on its inputs too,
                # and squeezing only the output would hand ``linalg.add`` operands
                # of two ranks.
                out = squeezed
            surface = _reduce_surface(iters, in_map, out_map)
            reduce_dims = tuple(
                dim for dim, iterator in enumerate(iters) if iterator == REDUCTION
            )
            if surface is Surface.GENERIC:
                # The one nest ``dimensions=`` cannot state, so the maps have to
                # travel with the step: the input covers three of four dims and
                # the lane axis is reduced on the way in and kept on the way out.
                indexing = Indexing(iters=iters, maps=(in_map, out_map))
        levels = _levels(spec, loops)
        broadcast = False
        if not spec.is_reduction:
            # A read of a statistic is squeezed the way its PRODUCER's output was,
            # so the reader's access has the rank of the buffer the producer
            # registered -- MEASURED, a reduction writes ``(256, 64)`` and its
            # consumer's spec describes the same buffer as ``(1, 256, 64)``.
            #
            # Pointwise only, and ``_reduction_nest`` is the reason: it reads the
            # INPUT's coordinates straight off the spec to derive ``in_map``, so
            # squeezing a reduction's input here would leave the map describing a
            # rank the loaded tensor no longer has.
            inputs = [
                _squeezed(
                    arg,
                    placeholder_axes(
                        arg.device_coordinates, [int(s) for s in arg.device_size]
                    ),
                )
                if _reads_stick_head(arg)
                else arg
                for arg in inputs
            ]
            # ``align_reshape_plan`` is the SWITCH, not a refusal (see
            # ``_broadcast_surface``).  Asked of every operand, because it is a
            # property of the whole op: one broadcast operand makes the op a
            # generic, and the aligned operands then need their (identity) rows
            # stated alongside it.  Here rather than below the accesses, because the
            # ARM is chosen on it and the surface is chosen from the arm; what needs
            # the accesses is ``_broadcast_surface``, whose map rows are about TILE
            # extents, and that call stays below them.
            broadcast = any(
                align_reshape_plan(
                    list(arg.device_coordinates),
                    [int(s) for s in arg.device_size],
                    list(out.device_coordinates),
                    [int(s) for s in out.device_size],
                )
                is not None
                for arg in inputs
            )
        # The op's spelling, chosen ONCE, from the whole request: the format, plus
        # whether an operand is broadcast (never, for a reduction, whose shape comes
        # from ``_reduce_surface`` -- the arm is asked for there only so that an op
        # that does not exist at this format is refused by the plan regardless).
        arm = recipe.arm(dtype, broadcast=broadcast)
        # Positions are the inputs in operand order and then the result, which is
        # what ``Recipe.unfused`` names: the element type an access reads a buffer
        # AT is the op's business, not the buffer's arrangement (see ``unfused``).
        accesses: dict[str, Access] = {}
        for position, arg in enumerate((*inputs, out)):
            access = self._access_of(
                arg,
                levels,
                head=_reads_stick_head(arg),
                unfused=position in recipe.unfused,
            )
            earlier = accesses.get(buf_id(arg))
            if earlier is not None and earlier.elems != access.elems:
                # Two accesses to one buffer at two element types WITHIN ONE STAGE.
                # The views are keyed ``(stage, buf_id)``, so the second would
                # silently take the first's view and load the wrong element type --
                # refused rather than keyed more finely, because neither target
                # needs it: the chain's stage 2 reads the pair base only as f16,
                # and the two types it does need are in two different stages.
                raise NotImplementedError(
                    f"OpSpec->KTIR: op {spec.op!r} reads buffer {buf_id(arg)!r} as "
                    f"both {earlier.elems.storage} and {access.elems.storage} in one "
                    "stage; a stage has one view per buffer, so two element types "
                    "in one stage are not supported"
                )
            accesses[buf_id(arg)] = access
        if not spec.is_reduction:
            # The two shapes a pointwise op can take, on the same bit that chose
            # the arm: the arm says what can be spelled, ``broadcast`` says what
            # has to be.
            if broadcast:
                surface, indexing = _broadcast_surface(arm, out, inputs, accesses)
            else:
                surface, indexing = _parallel_surface(
                    arm, len(inputs), len(accesses[buf_id(out)].extent)
                )
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
        # The scalar arguments the payload builder takes beyond its operands, read
        # from ``op_info`` here so that a malformed one is refused by the plan
        # rather than by a KeyError with a half-built module in hand.
        attrs: tuple[tuple[str, float], ...] = ()
        if recipe.attrs is not None:
            attrs = tuple(recipe.attrs(spec.op_info).items())
        return ComputeStep(
            op=spec.op,
            surface=surface,
            ins=tuple((buf_id(arg), accesses[buf_id(arg)]) for arg in inputs),
            out=accesses[buf_id(out)],
            out_buf_id=buf_id(out),
            stage=stage,
            reduce_dims=reduce_dims,
            indexing=indexing,
            attrs=attrs,
            dtype=dtype,
            broadcast=broadcast,
            # An internal buffer never reaches memory: it is threaded as a value,
            # so it gets no store, no func parameter, no view and no address.
            store=not is_internal(out),
        )

    def _access_of(
        self,
        arg: TensorArg,
        levels: Sequence[Level],
        *,
        head: bool = False,
        unfused: bool = False,
    ) -> Access:
        """``arg``'s access at this depth, registering its buffer on the way.

        The buffer record is built here and handed to the access, so the record
        carries its own way back to the view the builder will bind for it.  Each
        access gets its OWN record; ``self.buffers`` keeps the first one seen for a
        ``buf_id``, and what it is kept for is identity and the func signature --
        one parameter per buffer however many ways the stages view it.

        ``head`` narrows the TILE to one element on the innermost axis and leaves
        the VIEW alone -- the buffer is a whole stick per statistic either way, and
        which part of it this access reads is not a property of the buffer.  It is a
        hard constraint and not an optimisation: MEASURED against a hand-written
        negative reference module, a tile covering the whole innermost dimension is
        ``error: the tile covers more than the first element of its innermost
        dimension``, because the mean of squares sits sixteen bytes along the mean
        and a wider tile puts that offset on the next statistic.

        ``unfused`` is the recipe's word that THIS operand reads the buffer at its
        plain element type although the buffer holds fused statistics -- the mean
        out of the head of a pair's stick, as ``f16``.
        """
        layout, q = _solve_layout(arg, levels)
        elems = ElemTypes.of(arg.device_dtype, None if unfused else _arrangement(arg))
        buffer = None
        hinted = self.hints.get(buf_id(arg))
        if hinted is not None:
            # An access-only spec placed this buffer, so this read is a read of its
            # SOURCE (``_hint``).  Consulted before ``is_internal``, which would
            # otherwise thread the link as a value: a hinted buffer is not
            # threaded, it has a view, a real address and a real load.  Identity
            # and address come from the source's record; the geometry and element
            # type are THIS access's, so the map cannot pin a later stage to an
            # earlier one's view (the T5 property).
            buffer = dataclasses.replace(hinted, layout=layout, elems=elems)
        elif not is_internal(arg):
            # A ``Buffer`` PER ACCESS, built from this arg's own layout and element
            # types, and the registry keeps the first one.  The record does double
            # duty -- identity and address, which must be shared because
            # ``plan.parameters`` and ``KtirBuilder.bases`` are keyed by ``buf_id``;
            # geometry and element type, which are per access.  Sharing both through
            # one ``setdefault`` makes every stage's view of a buffer take the FIRST
            # stage's element type, and MEASURED, a layernorm chain needs two: one
            # base is viewed as ``memref<48x64x!spyreop.fp16_fused>`` where the pair
            # is written and as ``memref<48x64xf16>`` where the mean is read out of
            # the stick head.
            buffer = _buffer(
                arg, layout, elems, bake_addresses=self.options.bake_addresses
            )
            self.buffers.setdefault(buf_id(arg), buffer)
        # The divisions are the outermost levels, so their steps come first.
        extent, rows = _divide(arg, self._symbols, self._divisors)
        if head:
            extent = (*extent[:-1], 1)
        return _access(arg, extent, [*rows, *q], layout, elems, buffer)


def _base_address_elements(arg: TensorArg) -> int:
    """``arg``'s buffer base address in ELEMENTS, for the baked form only.

    Read from ``allocation["hbm"]``, the same field the SDSC path resolves into
    the bundle start address (``startAddressCoreCorelet_`` in ``superdsc``).
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


# ---------------------------------------------------------------------------
# PlanFusion: OpSpec sequences the device computes as one op, fused at PLAN time
# ---------------------------------------------------------------------------
#
# A SPECIALIZED FUSER FOR KTIR OPS.  Not general-purpose, and it must not grow
# into one: everything below is licensed by the fact that this module knows what
# it is about to emit and what the consequences of that emission are.  An
# upstream pass has no such licence, which is why this is not upstream.
#
# A fusion is not a legality decision.  The torch-spyre machinery already made
# that one, by handing this emitter a kernel holding both ops; the table only
# says WHICH sequences the device computes as one instruction, and declines on
# anything it does not recognise rather than guessing.  So every fusion here is
# OPPORTUNISTIC -- a strict subset of the scheduler's own fusion decisions,
# converting a chosen fusion into a better kernel and never creating one.
#
# A SPAN is the run of CONSECUTIVE OpSpecs a pattern matched -- ``specs[i:i+n]``
# for a pattern of n slots -- and it is the unit everything here works on: what a
# pattern matches and what a rewrite consumes.  For the one entry shipped, a span
# is two specs, an ``abs`` and the ``max`` that reads it.
#
# AN ENTRY IS A PATTERN AND A RESULT NAME.  The pattern is positional op names
# and reduction flags, a prefilter deciding which spans are considered at all;
# the result name is what the collapsed span is called.  The rewrite is SHARED
# (``_collapse_producer``) because nothing in it is specific to one entry: every
# condition it checks is a fact about deleting a producer and reading its source
# in its place.  There is exactly ONE rewrite, so no field names it; a field
# selecting the rewrite earns its place when there is a second thing to select.
#
# THERE IS NO KTIR COST MODEL.  Without a cost function over emitted kernels
# there is no basis on which to justify a MANDATORY fusion, so opportunistic is
# the most that can be defended.

_ABSMAX_OP = "absmax"


@dataclasses.dataclass(frozen=True)
class PlanFusion:
    """One sequence of OpSpecs the device computes as a single op.

    ``pattern``    ``(op name, is_reduction)`` per slot, in vector order.  A
                   cheap positional prefilter and nothing else: it decides which
                   spans the rewrite is even asked about, and it is deliberately
                   not where conditions live.
    ``result_op``  what the collapsed span is called.  A ``RECIPES`` key if the
                   kernel is to emit, but nothing here checks that: an
                   unemittable result is ``_stages``' refusal to make, and it
                   names the op.
    ``viable``     ``(fused) -> bool``: is the form about to be emitted one the
                   device computes CORRECTLY?  Separate from the rewrite because
                   it is a fact about the RESULT op on this hardware rather than
                   about collapsing anything, so it is the entry's only claim
                   about the device.  ``None`` means unconditional.
    ``why``        the device fact that makes the fusion a fusion.
    ``name``       for the logs.  A decline is silent by design -- almost every
                   span in every kernel is one -- so the name is what makes "why
                   did my absmax not fuse" answerable at ``debug``.

    Nothing validates the fields: an entry is source, and a malformed one fails
    where it is written the first time it is exercised rather than at import.
    """

    name: str
    pattern: tuple[tuple[str, bool], ...]
    result_op: str
    why: str
    viable: Callable[[OpSpec], bool] | None = None


def _decline(reason: str, *args: Any) -> None:
    """Log why a rewrite is declining, at ``debug``.

    Called immediately before the ``return None`` it explains, so the reason
    sits on the condition that produced it: a table that declines silently makes
    "why did my absmax not fuse" a question only this module can answer.  The
    pattern misses are the one class of decline with no line, because every span
    in every kernel that is not this pattern is one.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("plan fusion declines: " + reason, *args)
    return None


def _roles(spec: OpSpec) -> tuple[TensorArg, list[TensorArg]] | None:
    """``(output, inputs)`` for ``spec``, or None if it does not have exactly one.

    Read directly rather than through ``validated_roles``, which asks ``RECIPES``
    for the arity and would raise on ``abs`` -- an op with no recipe, and
    deliberately none, because the fused body is the only shape the device takes
    it in.  Returns None rather than raising: this is a matcher, and a spec whose
    roles it cannot read is a spec it declines.
    """
    inputs = [arg for arg in spec.args if arg.is_input]
    outputs = [arg for arg in spec.args if not arg.is_input]
    if len(outputs) != 1:
        return None
    return outputs[0], inputs


def _readers(link: str, specs: Sequence[Any]) -> tuple[tuple[OpSpec, TensorArg], ...]:
    """Every READ of buffer ``link`` in ``specs``, as ``(spec, arg)`` pairs.

    One entry per input arg naming it, so a spec reading the same buffer twice is
    listed twice.  That is the count the callers want: a producer may be deleted
    only when the read that replaces it is the only read there is, and an
    intermediate that survives needs one load per read.

    Loop bodies included, because a nested op reads the same buffer namespace.
    The scope is the list it is handed, which is also the limit: a fusion inside
    a ``LoopSpec`` body cannot see a reader outside that body, the same blind
    spot the recursion has.  Safe only in combination with ``kernel_local`` on
    the buffer -- a reader in another kernel is one no scope here would show.
    """
    return tuple(
        (spec, arg)
        for spec in _op_specs(specs)
        for arg in spec.args
        if arg.is_input and buf_id(arg) == link
    )


def _writers(link: str, specs: Sequence[Any]) -> tuple[OpSpec, ...]:
    """Every spec in ``specs`` that WRITES buffer ``link``.

    The mirror of ``_readers``, and it exists for one condition: an access-only
    spec may be dropped only when it is its link's sole writer, or the other
    writer's data would be read through a buffer nothing writes.  Loop bodies
    included, and scoped to the list it is handed for the same reason and with the
    same caveat ``_readers`` carries: safe only in combination with
    ``is_internal``, which is what makes this kernel's writes the only writes
    there can be.
    """
    return tuple(
        spec
        for spec in _op_specs(specs)
        for arg in spec.args
        if not arg.is_input and buf_id(arg) == link
    )


def _access_preserving(source: TensorArg, result: TensorArg) -> bool:
    """Whether an op writes its result exactly where it read its source.

    Same ``device_size``, same ``device_coordinates`` and same ``device_dtype``,
    which is what lets a consumer read the source in place of the result: the
    consumer's own description of its input then already describes the source, and
    nothing has to be translated between the two specs' iteration-space
    namespaces.  Anything that moves, resizes or reformats an element is not a
    drop-in.

    ``device_dtype`` is here because ``ACCESS_ONLY_OPS`` asks this question of a
    spec whose only content is a placement: a conversion has the same extent and
    the same coordinates as its source and is emphatically not a drop-in for it,
    so leaving the format out would hint away a real cast.

    Measured necessary, and it is the one condition whose absence is silent:
    without it a BROADCASTING ``abs`` fuses, and the ``absmax`` that comes out
    carries a [2, 256, 64] memory view over a 128-element buffer -- which
    the backend compiler ACCEPTS. An out-of-bounds read that compiles is worse than any
    refusal, so this is checked here and not left to a consumer.
    """
    return (
        list(source.device_size) == list(result.device_size)
        and list(source.device_coordinates) == list(result.device_coordinates)
        and source.device_dtype == result.device_dtype
    )


def _collapse_producer(
    span: Sequence[OpSpec], specs: Sequence[Any], result_op: str
) -> OpSpec | None:
    """A pair into one ``result_op`` reading the producer's own source.

    The one rewrite the table has, and it is shared: ``span`` is a producer and
    the consumer that survives it, and the only thing an entry contributes is
    the name the survivor comes out under.  None DECLINES, which is not an
    error: a span that is not this shape reaches ``_stages``, whose per-op refusal
    ("op 'abs' is not supported yet") is the truth about it.

    The producer is DELETED, not threaded.  The device primitive standing behind
    an entry does the pointwise pass's work inside the surviving op -- the min/max
    unit takes the absolute value as a mode bit -- so there is no intermediate left
    to put anywhere.  A fusion whose ops both survive would instead leave a value
    crossing a compute stage, which aborts the backend outright.

    Deleting the producer is what every condition below is about:

    * it must be unary, or there is no single source to read instead of it;
    * it must be ACCESS-PRESERVING, or the survivor's description of its input
      does not describe that source;
    * the link must be KERNEL-LOCAL and must be read exactly ONCE, by the
      survivor.  Two halves of one condition: locality (``TensorArg``, filled by
      the scheduler) rules out a reader this spec list cannot see, and the count
      rules out ``a = abs(x); amax(a, -1) + sum(a, -1)``, where the second
      consumer sits AFTER the pair -- so the pair is still adjacent, and a
      matcher trusting adjacency deletes a buffer the ``sum`` still reads.

    Only buffer IDENTITY moves across: name, arg index, allocation, format.  The
    extents and coordinates stay the survivor's own, because each spec writes
    its coordinates against its own iteration-space symbols and splicing one
    into the other would mix two namespaces -- a kernel that compiles and
    addresses the wrong elements, which is the worst failure available here.
    """
    # A pair, because collapsing a producer into its consumer is what this is;
    # an entry pairing a pattern of another length with it is a typo in the
    # table, and one that fails here the first time the pattern matches.
    producer, survivor = span
    roles = _roles(producer)
    if roles is None:
        _decline("%r does not write exactly one output", producer.op)
        return None
    producer_out, producer_ins = roles
    if len(producer_ins) != 1:
        _decline("producer %r is not unary", producer.op)
        return None
    [source] = producer_ins
    if not _access_preserving(source, producer_out):
        _decline(
            "producer %r is not access-preserving, so its source is not a "
            "drop-in for its result",
            producer.op,
        )
        return None
    if not producer_out.kernel_local:
        _decline(
            "link %s is not kernel-local, so deleting its producer would strand "
            "a reader outside this kernel",
            buf_id(producer_out),
        )
        return None
    link = buf_id(producer_out)
    reads = _readers(link, specs)
    if len(reads) != 1 or reads[0][0] is not survivor:
        _decline(
            "link %s is read %d time(s) in this kernel, not once by %r",
            link,
            len(reads),
            survivor.op,
        )
        return None
    [(_reader, read)] = reads
    args = [
        (
            dataclasses.replace(
                arg,
                name=source.name,
                arg_index=source.arg_index,
                allocation=dict(source.allocation),
                device_dtype=source.device_dtype,
            )
            if arg is read
            else arg
        )
        for arg in survivor.args
    ]
    return dataclasses.replace(survivor, op=result_op, args=args)


PLAN_FUSIONS: tuple[PlanFusion, ...] = (
    PlanFusion(
        name="absmax",
        pattern=(("abs", False), ("max", True)),
        result_op=_ABSMAX_OP,
        # False for fp32 on-stick absmax, which compiles and returns garbage.
        # MEASURED on device: for this one combination the backend emits its
        # ``SFP_SPLAT``/``SFP_REDUCE`` at ``mode=fp16`` on 4-byte lanes, so the
        # answer is NaN / ~1e38 with no diagnostic anywhere.  Declining costs a
        # working two-op kernel that this emitter cannot build either -- there is
        # no ``abs`` recipe -- so it buys a refusal in place of a wrong answer,
        # and nothing else.
        #
        # "On-stick" is exactly ``Surface.GENERIC`` for a reduction: the
        # within-stick axis is among the reduced dims, so the nest is not what
        # ``linalg.reduce`` means.  Asked of ``_reduction_surface``, which is the
        # derivation ``_compute_step`` itself will run, so the question cannot
        # drift from the emission.
        viable=lambda fused: not (
            dtype_of(fused) is DataFormats.IEEE_FP32
            and _reduction_surface(fused) is Surface.GENERIC
        ),
        why=(
            "the min/max unit takes the absolute value as a mode bit, so max(|x|) "
            "is one reduction and not a pointwise pass plus a reduction; and a "
            "standalone math.absf is refused by the backend anyway, so this is the "
            "only shape it takes an abs in"
        ),
    ),
)


def apply_plan_fusions(
    specs: Sequence[Any], table: Sequence[PlanFusion] = PLAN_FUSIONS
) -> tuple[Any, ...]:
    """``specs`` with every table match collapsed.

    Recurses into ``LoopSpec`` bodies because ``_divisions`` reads every op at
    every depth (``_op_specs``) and this runs before it.

    Matching is POSITIONAL and adjacent, which is what the vector is: a linear
    step order the emitted kernel executes in sequence.  A dataflow matcher
    would find pairs this misses, but it would then have to prove reordering
    them into adjacency is legal, which is a scheduling decision this layer has
    declined to make.  Adjacency is therefore not relied on for soundness: the
    condition it used to stand in for -- one reader of the link buffer -- is
    checked directly by the rewrite.
    """
    out: list[Any] = []
    i = 0
    while i < len(specs):
        entry = specs[i]
        if isinstance(entry, LoopSpec):
            # ``LoopSpec.body`` is declared a list, so the rebuilt body is one:
            # this is the contract's own type and not a copy taken for safety.
            body = apply_plan_fusions(entry.body, table)
            out.append(dataclasses.replace(entry, body=list(body)))
            i += 1
            continue
        for fusion in table:
            fused = _apply(fusion, specs, i)
            if fused is None:
                continue
            logger.debug(
                "plan fusion %r collapsed op %r into %r; buffer %s ceased to exist",
                fusion.name,
                entry.op,
                fused.op,
                ", ".join(buf_id(a) for a in entry.args if not a.is_input),
            )
            out.append(fused)
            i += len(fusion.pattern)
            break
        else:
            out.append(entry)
            i += 1
    return tuple(out)


def _apply(fusion: PlanFusion, specs: Sequence[Any], i: int) -> OpSpec | None:
    """The fused spec for ``specs[i : i + len(pattern)]``, or None: not a match.

    Three questions in order of cost, and the order is the point: the pattern is
    a positional comparison over every span of every kernel, the rewrite runs
    only on spans that pass it, and viability is asked of the spec the rewrite
    produced rather than of one it has to imagine.
    """
    # An empty pattern would match at zero length everywhere and advance the walk
    # by nothing.  A table is source, so this is a typo two hundred lines away
    # and not a vector to decline.
    assert fusion.pattern, f"fusion {fusion.name!r} has an empty pattern"
    span = specs[i : i + len(fusion.pattern)]
    if len(span) != len(fusion.pattern):
        return None
    for (op, is_reduction), spec in zip(fusion.pattern, span, strict=True):
        if not isinstance(spec, OpSpec):
            return None
        if spec.op != op or bool(spec.is_reduction) != is_reduction:
            return None
    fused = _collapse_producer(span, specs, fusion.result_op)
    if fused is None:
        return None
    if fusion.viable is not None:
        try:
            ok = fusion.viable(fused)
        except NotImplementedError as exc:
            # A derivation is entitled to refuse a shape it does not handle, and
            # a fusion nobody can decide about is one to leave alone: propagating
            # would turn a missing fusion into a crash blaming the derivation.
            _decline("viability of %r is undecidable: %s", fused.op, exc)
            return None
        if not ok:
            _decline(
                "%r is not viable on this operand (dtype/surface); the device "
                "computes this form incorrectly",
                fused.op,
            )
            return None
    return fused


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
    accumulate.  (``Recipe.reduces`` reads it off an arm for the whole op, which
    is sound because every arm of an op must agree on it.)

    The kind belongs to the ``Arm`` and not to the ``Recipe`` because it varies
    with what is asked for: ``add`` is a named ``linalg`` op at floats, a
    ``spyreop`` payload at four-byte integers, and an ``arith`` payload whenever
    an operand is broadcast, since only a generic's region can state a derived map
    row (``request_scalar_when_broadcast``).
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


def _written_here(fold: Callable[..., Any]) -> Callable[[], Callable[..., Any]]:
    """An ``Arm.binding`` for a body written here rather than named by a dialect.

    ``binding`` is a zero-argument callable because most of them are a dialect
    attribute -- ``lambda: arith.addf`` -- and the attribute cannot be reached at
    import time: the dialect handles are ``None`` until ``_load_dialects`` binds
    them.  A body written here as a Python function needs no such deferral, since
    it resolves ``arith`` and ``math`` inside its own body when it is called; this
    exists so the entry does not have to spell that as a lambda returning a lambda.
    """
    return lambda: fold


def _arms(arms: Arm | tuple[Arm, ...]) -> tuple[Arm, ...]:
    """\\p arms as a tuple, whether it was written as one arm or several.

    Idempotent, so it is safe to call on an already-normalised field: it is what
    ``Recipe.__post_init__`` normalises *with* and what every reader goes through,
    which keeps the one-arm shorthand from being a second representation that some
    code path forgets to handle.
    """
    return (arms,) if isinstance(arms, Arm) else arms


@dataclasses.dataclass(frozen=True)
class Request:
    """What a spec asks of an op, in the terms an arm can be chosen on.

    Every field is derivable from the SPEC ALONE -- no layout, no level, no core
    count -- and that invariant is what keeps selection orderable: the plan can
    answer all of them before it needs an arm, and a step can carry them so
    emission resolves the same arm with no spec in reach.  A discriminant that
    needed a layout could only be asked after the accesses were built, which is
    after the arm is needed to choose the surface those accesses feed.
    """

    dtype: DataFormats | None
    # Whether any operand is broadcast against the output -- that is,
    # ``align_reshape_plan`` has something to say about one of them.  A broadcast
    # operand's map row has to be STATED, and only a generic can state one, so
    # this bit decides whether a whole-op ``NAMED`` arm can serve the request at
    # all.
    broadcast: bool = False


# How a recipe picks among its arms, given what the spec asks for.  A dispatcher
# may only NARROW -- it returns one of the arms it was handed -- which
# ``Recipe.arm`` asserts, so a dispatcher cannot invent a spelling the table does
# not declare.
Dispatch = Callable[[tuple[Arm, ...], Request], Arm]


def request_by_dtype(arms: tuple[Arm, ...], request: Request) -> Arm:
    """The default: the arm claiming the format, else the one claiming the rest.

    A format nothing claims falls to an arm with an empty ``dtypes``; if no arm
    takes the unlisted formats either, the op does not exist at this one.

    With more than one such arm -- which ``Recipe.__post_init__`` allows only for
    arms of DIFFERENT kinds, one channel each -- the FIRST wins, so an entry lists
    the spelling it wants by default first.  ``add``'s ``linalg.add`` is ahead of
    its ``arith.addf`` for exactly that reason.
    """
    dtype = request.dtype
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


# The device formats an ``arith`` FLOAT scalar cannot take, out of the formats
# ``ElemTypes.NAMES`` supports (one, today: the ``i32`` row).  Kept in step with
# that table by ``TestArmDispatch``.
#
# Named here rather than spelled as ``Arm.dtypes`` on the float scalars, because
# ``dtypes`` is a POSITIVE claim and an explicit claim beats a dtype-less one
# (``request_by_dtype``): a scalar arm that listed the float formats would
# out-claim the dtype-less ``NAMED`` arm on the ALIGNED path and turn every
# ``linalg.add`` into a generic.  So the scalar arms stay dtype-less, and what a
# dtype-less scalar arm does not serve is stated once, here, where it is read.
_INTEGER_FORMATS: tuple[DataFormats, ...] = (DataFormats.IEEE_INT32,)


def request_scalar_when_broadcast(arms: tuple[Arm, ...], request: Request) -> Arm:
    """``request_by_dtype``, but a broadcast operand may not reach a NAMED arm.

    Filtered on ``kind`` because ``kind`` IS the distinction being made: a
    ``NAMED`` builder states its own (identity) indexing and has nowhere to put a
    derived map row, while a ``PAYLOAD`` scalar goes in a generic's region, which
    states every row.

    It DELEGATES rather than replacing, so the format still picks among what is
    left: an int32 broadcast ``add`` lands on ``spyreop.addi32toi32`` and not on
    ``arith.addf``.  And it filters only when the request really is broadcast, so
    an aligned operand keeps the named op it always had.

    A dtype-less scalar arm is an ``arith`` float builder (``_INTEGER_FORMATS``),
    so it is not eligible for an integer request: ``arith.subf`` of two ``i32``
    values does not verify, and letting one through would emit invalid IR for a
    request the plan could have refused.  ``sub`` has no integer intrinsic to fall
    to, so at int32 nothing survives the filter at all.

    ``and eligible`` is that case: the whole set goes through, the named arm comes
    back, and ``_broadcast_surface`` refuses naming the reason (an op whose only
    spelling states its own indexing) -- a better answer than "no arm for
    IEEE_INT32" for an op that plainly has one at that format.
    """
    eligible = tuple(
        arm
        for arm in arms
        if arm.kind is not BindingKind.NAMED
        and (arm.dtypes or request.dtype not in _INTEGER_FORMATS)
    )
    return request_by_dtype(
        eligible if request.broadcast and eligible else arms, request
    )


@dataclasses.dataclass(frozen=True)
class Recipe:
    # Both ``arity`` and ``attrs`` are properties of the *op*, invariant across
    # formats, which is why they sit here and not on an arm: 'add' takes two
    # operands and reads no scalars from ``op_info`` whatever its element type is.
    # Stating them once is also what keeps two spellings of one op from drifting
    # apart on arity.
    arity: int
    # One ``Arm`` or a tuple of them; ``__post_init__`` promotes the bare one, so
    # the field is a tuple by the time anything reads it.  Beware that a stray
    # missing comma turns an intended tuple into a single ``Arm`` silently.
    arms: Arm | tuple[Arm, ...]
    # Which operand positions this op reads (or writes) at the buffer's PLAIN
    # element type although the buffer's ``element_arrangement`` says it holds
    # fused statistics.  Positions are the inputs in operand order and then the
    # result last -- the order ``Indexing.maps`` takes -- so ``arity`` is the
    # result's position.
    #
    # It is on the RECIPE because it is a fact about the op and not about the
    # buffer.  MEASURED on the real layernorm vector, ``element_arrangement`` says
    # only "this buffer holds two values to a stick" -- it is propagated to every
    # arg naming a statistic buffer and does not say how an operand READS it, and
    # two ops read one such buffer two ways.  So the arrangement is the default and
    # the recipe has the last word.
    unfused: tuple[int, ...] = ()
    # How to read the op's scalar arguments out of a spec's ``op_info``, for the
    # few ops whose builder takes more than operands (softplus).  ``None`` when
    # the op is a pure function of its operands, which is almost all of them.
    # A reader rather than the values themselves, because where they live in
    # ``op_info`` is the op's own business and the plan should not have to know.
    attrs: Callable[[dict[str, Any]], dict[str, float]] | None = None
    # How this op picks among its arms; the default is the format alone, and an op
    # whose spelling also turns on whether an operand is broadcast says so here.
    #
    # A plain function and not a ``staticmethod``: the generated ``__init__``
    # binds it as an INSTANCE attribute, and instance attributes are not
    # descriptors, so ``self.dispatch(arms, request)`` passes no ``self``.
    dispatch: Dispatch = request_by_dtype

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError(f"OpSpec->KTIR: arity must be >= 1, got {self.arity}")
        arms = _arms(self.arms)
        if not arms:
            raise ValueError("OpSpec->KTIR: a recipe needs at least one arm")
        # Ambiguity is per KIND, not per recipe.  Two arms of the SAME kind with
        # the same claim are genuinely ambiguous -- nothing tells them apart, so
        # which one wins would be a fact about declaration order.  Two of
        # different kinds are the two channels a dispatcher discriminates on: fp16
        # ``add`` needs both a dtype-less ``linalg.add`` and a dtype-less
        # ``arith.addf``, one for aligned operands and one for broadcast.
        for kind in {arm.kind for arm in arms}:
            of_kind = [arm for arm in arms if arm.kind is kind]
            if sum(1 for arm in of_kind if not arm.dtypes) > 1:
                raise ValueError(
                    "OpSpec->KTIR: at most one arm may claim the unlisted formats"
                )
            claimed = [dtype for arm in of_kind for dtype in arm.dtypes]
            if len(claimed) != len(set(claimed)):
                raise ValueError(
                    "OpSpec->KTIR: two arms claim the same format: "
                    f"{sorted({d.name for d in claimed if claimed.count(d) > 1})}"
                )
        # Reduction-ness is asked of the RECIPE (``reduces``), before any arm is
        # chosen, so every arm has to answer it the same way or the property is
        # reading one arm and speaking for the others.
        if len({arm.kind is BindingKind.COMBINER for arm in arms}) > 1:
            raise ValueError(
                "OpSpec->KTIR: an op's arms must agree on whether it reduces, but "
                f"{sorted({arm.kind.name for arm in arms})} do not"
            )
        if any(not 0 <= position <= self.arity for position in self.unfused):
            raise ValueError(
                f"OpSpec->KTIR: unfused positions {self.unfused} name an operand "
                f"an arity-{self.arity} op does not have (the result is "
                f"{self.arity})"
            )
        object.__setattr__(self, "arms", arms)

    @property
    def reduces(self) -> bool:
        """Whether this op accumulates -- an op fact, not an arm fact.

        Sound because ``__post_init__`` makes every arm agree on it, so any arm
        answers for the recipe.  It exists so that the family check in ``_stages``
        gets its one bit without selecting an arm, which it cannot do yet.
        """
        return _arms(self.arms)[0].kind is BindingKind.COMBINER

    def arm(self, dtype: DataFormats | None, *, broadcast: bool = False) -> Arm:
        """The arm this request reaches, or raise.

        ``dtype`` stays positional because it is the one discriminant every op
        has; the rest of the request is keyword-only with a default, so a caller
        that has not derived a discriminant yet need not name it.
        """
        arms = _arms(self.arms)
        chosen = self.dispatch(arms, Request(dtype=dtype, broadcast=broadcast))
        # A dispatcher may narrow, never invent.  Identity rather than equality:
        # what is being asserted is that the arm came from THIS recipe, and two
        # structurally equal arms are two declarations rather than one.  An
        # assertion because a dispatcher is code in this module, so failing it is a
        # bug here and not an unsupported request.
        assert any(chosen is candidate for candidate in arms), (
            f"dispatcher {self.dispatch!r} returned an arm this recipe does not hold"
        )
        return chosen


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
        # ``(stage, buf_id) -> view``, filled by ``view()`` on demand: one view
        # per stage that tiles the buffer, never one per buffer.
        self.views: dict[tuple[int, str], Any] = {}
        # ``buf_id -> base address``, bound once by ``open_kernel``.
        self.bases: dict[str, Any] = {}
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
    def _affine_map(rank: int, row: Sequence[int | None]):
        """One ``Indexing`` row as a projection of a ``rank``-dim iteration nest.

        ``(0, 1, 2)`` of rank 4 is ``(d0, d1, d2, d3) -> (d0, d1, d2)``: the row is
        one dim index per result position, so the map has ``rank`` dims, no
        symbols, and one expression per entry.  Returns the map itself and not an
        ``ir.AffineMapAttr`` -- ``indexing_maps`` takes maps, and an attribute
        raises there.

        ``None`` is the constant 0 position, so ``(1, None)`` of rank 3 is
        ``(d0, d1, d2) -> (d1, 0)``: a broadcast operand read at the head of the
        axis it does not walk.  ``get_constant`` is the mechanism ``coord_set``
        already builds its bounds out of.
        """
        return ir.AffineMap.get(
            rank,
            0,
            [
                ir.AffineExpr.get_constant(0)
                if dim is None
                else ir.AffineExpr.get_dim(int(dim))
                for dim in row
            ],
        )

    # -- module scaffolding ------------------------------------------------

    @contextlib.contextmanager
    def open_kernel(self, kernel_name: str) -> Iterator[None]:
        """Open the kernel func with its bases bound, and emit its body into it.

        ``module { func.func @kernel_name(...) { %c0, <body>, return } }``.  The
        signature and the base addresses are two faces of one decision -- where a
        base address comes from -- so they are made together here rather than in
        two functions a caller has to order correctly.  All of it comes off
        ``self.plan``, the plan this builder was created for.

        Baked bases need no func arguments and appear as ``arith.constant``s;
        symbolic bases are one ``index`` parameter each, in ``plan.parameters``
        order.  Deleting the baked arm reverts the dataflow-scheduler#65
        workaround.

        The bases, and not the views: a view belongs to the stage that tiles it
        (``view()``), because two stages sharing one view abort the backend
        (``ComputeStep.stage``).  A base is per buffer and shared by every stage,
        which is why it stays here -- one func parameter or one constant per
        buffer either way, whatever the stages then do with it.
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
                        self.bases[buffer.buf_id] = base
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
        # The step's whole ``Request``, so this resolves the arm the plan chose the
        # surface from: an op with two spellings would otherwise get the default
        # one here and a body the surface below does not fit.
        arm = self.RECIPES[step.op].arm(step.dtype, broadcast=step.broadcast)
        ins = [self.operand(buf_id, access, step.stage) for buf_id, access in step.ins]
        match step.surface:
            case Surface.BARE:
                value = self._emit_bare(arm.binding(), ins, step)
            case Surface.REDUCE:
                value = self._emit_reduce(arm.binding(), ins, step)
            case Surface.GENERIC:
                value = self._emit_generic(arm.binding(), ins, step)
            case _:
                raise AssertionError(f"unplanned surface {step.surface} of {step.op!r}")
        self.result(
            step.out_buf_id, step.out if step.store else None, value, step.stage
        )

    # -- ktdp shapes -------------------------------------------------------

    def memory_view(self, base, buffer: Buffer):
        """``ktdp.construct_memory_view`` for one buffer, at base address ``base``.

        Extent and strides come from ``buffer.layout``, the record ``_layout``
        derived, so the view says what the plan says, in whole element counts.
        """
        sizes = [int(e) for e in buffer.layout.extent]
        strides = [int(s) for s in buffer.layout.strides]
        memref_t = ir.MemRefType.get(sizes, self.named_type(buffer.elems.storage))
        # The ``memory_space`` builder takes the tablegen-generated
        # ``MemorySpaceKind``, not a
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

    def view(self, stage: int, buffer: Buffer):
        """``stage``'s memory view of ``buffer``, emitted at first use in it.

        One view per (stage, buffer) and not one per buffer: sharing a view
        between two stages ABORTS the backend rather than refusing -- MEASURED on
        a hand-written reference chain of six computes, where deduping its seven
        duplicate views onto four aborts while the duplicates it ships with
        compile clean.  A stage's schedule is extracted
        into its own module, and the view has to go with it, so a second stage's
        use of the same view is a use the extraction cannot erase.

        At first use rather than up front, so a stage emits views for the buffers
        it tiles and no others.  A stage is one ``ComputeStep``, all of whose ops
        are emitted contiguously at one insertion point, so the view a stage
        emits dominates every tile that reads it -- including inside a loop body,
        where the whole stage lives.
        """
        key = (stage, buffer.buf_id)
        if key not in self.views:
            self.views[key] = self.memory_view(self.bases[buffer.buf_id], buffer)
        return self.views[key]

    def access_tile(self, access: Access, stage: int):
        """``ktdp.construct_access_tile`` for ``access``, into ``stage``'s view.

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
                base=self.view(stage, access.buffer),
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

    def operand(self, buf_id: str, access: Access, stage: int):
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
            ktdp.load(result=tensor_t, access_tile=self.access_tile(access, stage))
        )

    def result(self, buf_id: str, access: Access | None, value, stage: int) -> None:
        """Dispose of an op's result: thread it, or store it through ``access``.

        The mirror of ``operand`` on the way out.  ``access is None`` is an
        internal buffer: it has no view to store through, so the value is bound
        in scope for a later op to consume.
        """
        if access is None:
            self.env.bind_produced(buf_id, value)
        else:
            ktdp.store(data_tile=value, access_tile=self.access_tile(access, stage))

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
        # ``add``, ``mul`` and ``sub`` are the ops with more than one spelling, on
        # two discriminants at once:
        #
        #   * the FORMAT -- a named linalg op at floats, and a ``spyreop``
        #     intrinsic at four-byte integers that splits its operands into halves
        #     and finds the carry with a pair of scale factors.  A float arm lists
        #     no formats, so it takes every format the integer arm does not claim.
        #   * whether an operand is BROADCAST -- a named linalg op states its own
        #     identity indexing, so a broadcast operand's derived map row has
        #     nowhere to go (``_broadcast_surface``).  The ``arith`` scalar goes in
        #     a generic's region, which states every row, so
        #     ``request_scalar_when_broadcast`` reaches for it exactly then.
        #     MEASURED: this is what softmax's ``x - rowmax`` needs, and with it
        #     the KTIR path's softmax matches the SDSC path's exactly (verify.py).
        #
        # The named arm is FIRST in each entry because two dtype-less arms of
        # different kinds resolve in declaration order (``request_by_dtype``), and
        # the aligned operands that keep the named op are the common case: every
        # emitter golden is a named ``add``.
        "add": Recipe(
            arity=2,
            dispatch=request_scalar_when_broadcast,
            arms=(
                Arm(kind=BindingKind.NAMED, binding=lambda: linalg.add),
                Arm(kind=BindingKind.PAYLOAD, binding=lambda: arith.addf),
                Arm(
                    kind=BindingKind.PAYLOAD,
                    binding=lambda: spyreop.addi32toi32,
                    dtypes=(DataFormats.IEEE_INT32,),
                ),
            ),
        ),
        "mul": Recipe(
            arity=2,
            dispatch=request_scalar_when_broadcast,
            arms=(
                Arm(kind=BindingKind.NAMED, binding=lambda: linalg.mul),
                Arm(kind=BindingKind.PAYLOAD, binding=lambda: arith.mulf),
                Arm(
                    kind=BindingKind.PAYLOAD,
                    binding=lambda: spyreop.muli32toi32,
                    dtypes=(DataFormats.IEEE_INT32,),
                ),
            ),
        ),
        # No integer arm: there is no ``subi32toi32`` intrinsic, so an int32
        # broadcast ``sub`` still reaches ``_broadcast_surface``'s refusal -- the
        # scalar arm it would need does not exist at that format.  A missing op,
        # not a dispatch gap.
        "sub": Recipe(
            arity=2,
            dispatch=request_scalar_when_broadcast,
            arms=(
                Arm(kind=BindingKind.NAMED, binding=lambda: linalg.sub),
                Arm(kind=BindingKind.PAYLOAD, binding=lambda: arith.subf),
            ),
        ),
        "sum": Recipe(
            arity=1, arms=Arm(kind=BindingKind.COMBINER, binding=lambda: arith.addf)
        ),
        "max": Recipe(
            arity=1,
            arms=Arm(kind=BindingKind.COMBINER, binding=lambda: arith.maximumf),
        ),
        "min": Recipe(
            arity=1,
            arms=Arm(kind=BindingKind.COMBINER, binding=lambda: arith.minimumf),
        ),
        "prod": Recipe(
            arity=1, arms=Arm(kind=BindingKind.COMBINER, binding=lambda: arith.mulf)
        ),
        # Two callers reach this one recipe, which is why it is a recipe and not a
        # special case.  ``torch.any`` lowers to a genuine ``absmax`` reduction
        # (``lower_any_dim`` / ``lower_any_def`` in lowering.py), so the frontend
        # names this op itself; and the fusion table rewrites an ``abs`` feeding a
        # ``max`` into this same name, because the device computes both the same
        # way.
        # A combiner is just a callable, so this one emits three ops instead of
        # one -- the shape the device matches -- and needs no new surface.
        #
        # ``max(|acc|, |x|)``, and the body ORDER is a pattern key rather than a
        # computation: the device pattern matches exactly three ops -- ``math.absf``
        # at [0] and [1] and ``arith.maxnumf`` at [2] -- and replaces the whole
        # generic with one ``simdreduction_minmax`` whose ``x1``/``x2`` immediates
        # put the min/max unit in its absolute-value mode.  So nothing below is
        # lowered: the abs is a mode bit on the hardware compare, and these ops
        # exist to be recognised.  Python's left-to-right argument evaluation is
        # what emits them in the order the match requires, so a "simplification"
        # that reorders the expression -- or that writes ``maximumf``, the spelling
        # the bare ``max`` reduction uses, in place of ``maxnumf`` -- breaks the
        # match, and it breaks it SILENTLY: it fails as a non-match, not an error.
        _ABSMAX_OP: Recipe(
            arity=1,
            arms=Arm(
                kind=BindingKind.COMBINER,
                binding=_written_here(
                    lambda accumulated, element: arith.maxnumf(
                        math.absf(element), math.absf(accumulated)
                    )
                ),
            ),
        ),
        # The reduction that leaves a mean and a mean of squares in one element.
        #
        # Registered COMBINER because it reduces, and the equality check in
        # ``_stages`` is what ties those two statements together -- but the
        # binding IGNORES ``accumulated``, which no other combiner does.  That is
        # the same category as ``absmax`` above and carries the same warning: the
        # body is a MARKER FOR A DEVICE PATTERN, not a fold.  MEASURED, the
        # device's own unfusing pass replaces this op with the two
        # reductions it stands for -- the value accumulated and its square, each
        # scaled by one over the count on the way in -- so nothing below lowers
        # the generic as written.  If that pattern does not match, the generic
        # means "the last element wins", and it fails as a NON-MATCH rather than
        # as an error.
        #
        # The accumulator is still what types the result: ``%out`` is
        # ``!spyreop.fp16_fused`` because the output buffer's arrangement says so
        # (EXX2), and ``spyreop.exx2_fused`` returns exactly that from an f16.
        # ``spyreop.exx2``, the unfused two-result form, is a different op and is
        # not registered.
        "exx2": Recipe(
            arity=1,
            arms=Arm(
                kind=BindingKind.COMBINER,
                binding=_written_here(
                    lambda accumulated, element: spyreop.exx2_fused(element)
                ),
            ),
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
        # Not here: ``abs``.  A PAYLOAD arm bound to ``math.absf`` does emit, but
        # into a generic of its OWN, which is the wrong shape for the only thing
        # that wants it: the device reduces along the stick with an opaque SIMD
        # reduction matched by a PDL pattern in the device spec
        # (``KTIR_DEVICE_MLIR``) against the BODY of one reducing
        # ``linalg.generic``, and its ``absmax`` kind wants that body to be three
        # ops.  So ``abs_max`` is one fused combiner to build, not a pointwise op
        # to thread into a separate reduction.
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
        # The FUSED spelling, arity 1: the frontend hands this op one input
        # carrying the (mean, mean-of-squares) pair as one element of
        # ``!spyreop.fp16_fused``, which is what ``exx2_fused`` wrote.  The
        # two-operand ``spyreop.layernormscale``, which takes the mean and the mean
        # of squares apart, is what the backend's own unfusing pass
        # produces BELOW us -- MEASURED against a hand-written reference module for
        # this op -- so binding it here would be doing the backend's job with an
        # operand nobody supplies.
        "layernormscale": Recipe(
            arity=1,
            arms=Arm(
                kind=BindingKind.PAYLOAD,
                binding=lambda: spyreop.layernormscale_fused,
            ),
            # The RESULT (position 1, arity being 1) is a plain float, whatever the
            # output buffer's arrangement says.  MEASURED: the frontend flags that
            # buffer ``EXX2`` too -- it propagates the flag to every arg naming a
            # statistic buffer -- and the op is
            # ``... : !spyreop.fp16_fused -> f16``.
            unfused=(1,),
        ),
        # The normalisation itself: five operands, positional, no attributes.
        # MEASURED against a hand-written reference module -- the printed form
        # is ``spyreop.layernormnorm %x squares %sq scale %sc weight %w bias %b``,
        # and the builder takes them in that order.
        #
        # ``squares`` (1) and ``scale`` (2) are read UNFUSED, and MEASURED on the
        # real ``F.layer_norm`` vector both of their args carry ``EXX2``: the flag
        # follows the BUFFER and is propagated to every arg naming one.  This op
        # reads a plain ``f16`` out of the head of each stick in both cases --
        # MEASURED on a hand-written reference chain, where both views are
        # ``memref<48x64xf16>`` over bases the fused view also covers.
        "layernormnorm": Recipe(
            arity=5,
            arms=Arm(kind=BindingKind.PAYLOAD, binding=lambda: spyreop.layernormnorm),
            unfused=(1, 2),
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

        # ``linalg.reduce`` binds its region as (input element, init accumulator),
        # in that order -- MLIR's choice, not ours, and the printer names them
        # ``%in`` and ``%init`` to say so.  The parameters are therefore named in
        # THAT order and the fold is written acc-first, which is the order a
        # combiner means: ``acc = combine(acc, x)``.
        def body(element, accumulated):
            return combine(accumulated, element)

        # The region builder reads the block argument types off the annotations,
        # and the element type is only known here, so they are set rather than
        # written.
        body.__annotations__ = {"element": elt_t, "accumulated": elt_t}
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

        # A generic's region binds one argument per input and then the ``outs``
        # accumulator, so a reducing nest's accumulator arrives LAST.  A combiner
        # wants it first (see ``_emit_reduce``), hence the rotation; a parallel
        # nest drops it and keeps the inputs in the order the op names them,
        # which is the order ``sub`` and ``realdiv`` pin.
        def body(*args):
            if reducing:
                return payload(args[-1], *args[:-1], **attrs)
            return payload(*args[:-1], **attrs)

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
