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

"""Dialect-free tests for the OpSpec->KTIR emitter's *rejections*.

**Nothing in this file imports ``mlir_ktdp``, directly or transitively, and
nothing in it is skipped.**  That is the property ``ktir.build_kernel_plan``
exists to provide: every ``NotImplementedError`` the emitter can raise is raised
by a pure walk over the spec tree, before the lazy dialect import, so the whole rejection
surface is covered wherever ``import torch_spyre`` works.

``test_ktir_emitter.py`` holds the complement -- the golden MLIR snapshots, which
do need the dialect build and are skipped without it.  It imports the shared spec
builders from here (``make_op_spec`` and friends), so they stay dialect-free.
"""

import ast
import contextlib
import dataclasses
import importlib
import inspect
import sys
import unittest

import regex as re
import sympy

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.codegen import ktir
from torch_spyre._inductor.constants import STAGGERED_EAS
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

# ---------------------------------------------------------------------------
# Building a spec
#
# One builder, so a test states only what it is about and states it on one line:
# ``make_op_spec()`` is the whole pointwise contract the frontend produces, and
# every keyword is one deviation from it.  The ``TensorArg``s are built inside,
# because their two positional fields are not a test's to choose: ``arg_index``
# is the position among the HBM args, and an arg that memory planning placed (an
# ``lx`` / ``hbm_pool`` allocation) is not passed in at all, so it takes -1 and
# consumes no position -- the frontend's own rule, applied here rather than
# restated per fixture.
#
# The per-arg keywords (``names`` / ``sizes`` / ``allocations`` / ``advances``)
# are indexed over inputs then outputs, and may be short or hold ``None`` to
# leave an arg at its default.
# ---------------------------------------------------------------------------

FP16 = DataFormats.SEN169_FP16
ADD_SIZE = [16, 512, 64]


def make_op_spec(
    op: str = "add",
    *,
    inputs: int = 2,
    outputs: int = 1,
    names: list | None = None,
    size: list = ADD_SIZE,
    sizes: list | None = None,
    coords: list | None = None,
    coords_per_arg: list | None = None,
    dtype: DataFormats = FP16,
    arrangements: list | None = None,
    allocations: list | None = None,
    baked: bool = False,
    advances: list | None = None,
    kernel_locals: list | None = None,
    is_reduction: bool = False,
    divisions: dict | None = None,
    space: dict | None = None,
    tiled: list | None = None,
    trips: dict | None = None,
    first_arg_index: int = 0,
    op_info: dict | None = None,
) -> OpSpec:
    """A finished ``OpSpec``, defaulting to ``a + b`` at [16, 512, 64] fp16.

    That default is what the SuperDSC frontend produces for a pointwise add: two
    HBM inputs and one HBM output at identity ``(d0, ..., dn)`` coordinates, with
    the HBM address left unassigned (the symbolic form reads no address, and the
    baked form is the one that rejects that).

    The deviations, each a keyword:

    * ``op`` / ``inputs`` / ``outputs`` / ``names`` -- the op and its roles.
    * ``size`` for every arg, or ``sizes`` per arg; likewise ``coords`` for every
      arg or ``coords_per_arg`` for one list each, which a reduction needs because
      its output drops an axis.  ``coords=[]`` is a tiled arg, addressing through
      ``advances`` instead of coordinates.
    * ``arrangements`` per arg, for a buffer whose elements are not in the
      standard order -- ``ElementArrangement.EXX2`` is a statistic buffer holding
      two values in one element, and the default is ``STANDARD``.
    * ``allocations`` per arg, for an ``lx`` / ``hbm_pool`` intermediate or an
      unrecognised space; ``baked=True`` for the byte HBM address the baked form
      wants, which is the same field said the other way, so not both.
    * ``kernel_locals`` per arg: the bit the scheduler fills, saying nothing
      outside this kernel reads the buffer.  Defaults to False, as it does in
      the contract.
    * ``divisions`` maps a coordinate symbol's name to its work division;
      ``space`` replaces the iteration space outright (``{}`` for a tiled op).
    * ``tiled`` / ``trips`` are the loop-level symbols and trip counts, and
      ``first_arg_index`` continues the numbering for a second op in one kernel.
    * ``op_info`` is the op's auxiliary dict, which the recipes that take scalar
      arguments read (softplus's beta/threshold live in ``op_info["constants"]``);
      it defaults to empty, which every other op wants.
    """
    if allocations and baked:
        raise ValueError("make_op_spec: pass allocations= or baked=, not both")

    def at(per_arg: list | None, position: int):
        """``per_arg[position]``, treating short lists and ``None`` as default."""
        if per_arg is None or position >= len(per_arg):
            return None
        return per_arg[position]

    roles = [(True, i) for i in range(inputs)] + [(False, i) for i in range(outputs)]
    args, next_index = [], first_arg_index
    for position, (is_input, ordinal) in enumerate(roles):
        allocation = at(allocations, position)
        # An arg planning placed is not passed in: -1, and it takes no position.
        if allocation is not None and "hbm" not in allocation:
            arg_index = -1
        else:
            arg_index, next_index = next_index, next_index + 1
            if allocation is None:
                allocation = {"hbm": arg_index << 34 if baked else None}
        arg_size = at(sizes, position) or size
        arg_coords = at(coords_per_arg, position)
        if arg_coords is None:
            arg_coords = (
                sympy.symbols(f"d0:{len(arg_size)}") if coords is None else coords
            )
        args.append(
            TensorArg(
                is_input=is_input,
                arg_index=arg_index,
                device_dtype=dtype,
                device_size=list(arg_size),
                device_coordinates=list(arg_coords),
                allocation=allocation,
                name=at(names, position)
                or (f"arg{ordinal}" if is_input else f"buf{ordinal}"),
                device_tile_advance_expr=at(advances, position),
                element_arrangement=at(arrangements, position)
                or ElementArrangement.STANDARD,
            )
        )

    if space is None:
        out_size = args[-1].device_size
        space = {
            symbol: (extent, (divisions or {}).get(str(symbol), 1))
            for symbol, extent in zip(sympy.symbols(f"d0:{len(out_size)}"), out_size)
        }
    return OpSpec(
        op=op,
        is_reduction=is_reduction,
        iteration_space=space,
        args=args,
        op_info=op_info or {},
        tiled_symbols=tiled or [],
        tiled_symbol_trip_counts=trips or {},
    )


def make_chained_op_specs(
    ops: tuple = ("add", "mul"), *, owned: bool = False, **overrides
) -> list:
    """The ops of one kernel, each handing its result to the next.

    ``owned=False`` (the default) gives every intermediate an ordinary HBM buffer
    with an ``arg_index``, so the producing stage stores it and the consuming stage
    loads it -- what the frontend produces with the memory planners off.  One
    buffer keeps ONE index across the specs that share it, which is why the
    numbering advances over every arg rather than over the fresh inputs only.

    ``owned=True`` makes each intermediate an ``lx`` buffer instead: not passed
    in, no address, threaded as a value. That is what memory planning produces by
    default, and across a stage boundary it is refused -- so it is kept for the
    test that pins that refusal, and for nothing else.
    """
    lx = {"lx": 0}
    specs, next_arg = [], 0
    for level, op in enumerate(ops):
        # The first op reads two fresh inputs; every later one reads the previous
        # result and one fresh input.
        threaded = [] if level == 0 else [f"buf{level - 1}"]
        fresh = [f"arg{next_arg + i}" for i in range(2 - len(threaded))]
        last = level == len(ops) - 1
        specs.append(
            make_op_spec(
                op,
                names=[*threaded, *fresh, f"buf{level}"],
                allocations=[
                    *([lx if owned else None] if threaded else []),
                    *([None] * len(fresh)),
                    None if last or not owned else lx,
                ],
                first_arg_index=next_arg,
                **overrides,
            )
        )
        # An owned intermediate takes no slot (``arg_index == -1``); a passed one
        # takes the next, and the following spec starts AT it so the buffer they
        # share carries one index.
        next_arg += len(fresh) + (0 if owned and not last else 1)
        if not owned and not last:
            next_arg -= 1
    return specs


def make_nested_op_spec(*, levels: list, **overrides) -> tuple:
    """One op inside a loop nest, as ``(nest, op, loops)``.

    ``levels`` is ``[(symbol, trip count), ...]`` outermost-first, and it states
    the nest once: the ``scf.for`` trip counts, the op's ``tiled_symbols``
    (innermost-first, one entry per level) and its ``tiled_symbol_trip_counts``
    all come from it, so they cannot disagree.  A tiled op addresses through
    ``advances`` rather than coordinates, and its iteration space is empty.

    ``loops`` is the enclosing chain outermost-first -- what the plan walk would
    reach the op with, and what ``_levels`` takes.
    """
    spec = make_op_spec(
        coords=[],
        space={},
        tiled=[[symbol] for symbol, _ in reversed(levels)],
        trips=dict(levels),
        **overrides,
    )
    body: list = [spec]
    loops: list = []
    for _symbol, trip in reversed(levels):
        loops.insert(0, LoopSpec(count=trip, body=body))
        body = [loops[0]]
    return loops[0], spec, loops


def make_onstick_sum_specs(op: str = "sum", arrangements: list | None = None) -> list:
    """``sum(x[256, 128], dim=-1)`` on one core, as the frontend projects it.

    ``op`` names the reduction and ``arrangements`` is passed straight through, so
    the same vector serves any arity-1 reduction over the stick -- the shape is the
    fixture's contribution and the op is the caller's.

    The reduction runs along the *stick*, so it consumes both halves of the
    reduced symbol -- the outer-stick chunk index ``floor(c1 / 64)`` and the
    within-stick lane ``c1 % 64`` -- and the output nonetheless has 64 lanes at a
    constant coordinate.  Every number here is the frontend's own: device sizes
    [2, 256, 64] in and [1, 256, 64] out, the output's axis 0 a placeholder and
    its axis 2 the lane the D2H descriptor gathers across.

    Shared rather than local to one test class because both halves of the suite
    want it: the dialect-free plan assertions here, and the golden in
    ``test_ktir_emitter.py``.
    """
    rows, reduced = sympy.symbols("c0 c1")
    stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
    return [
        make_op_spec(
            op,
            is_reduction=True,
            inputs=1,
            arrangements=arrangements,
            sizes=[[2, 256, 64], [1, 256, 64]],
            coords_per_arg=[
                [stick, rows, lane],
                [sympy.Integer(0), rows, sympy.Integer(0)],
            ],
            space={rows: (256, 1), reduced: (128, 1)},
        )
    ]


def make_broadcast_op_spec(form: str = "row") -> OpSpec:
    """One pointwise op with a BROADCAST operand, in one of three forms."""
    d0, d1, d2 = sympy.symbols("d0 d1 d2")
    zero = sympy.Integer(0)
    if form == "row":
        return make_op_spec(
            "realdiv",
            sizes=[[16, 512, 64], [16, 1, 64], [16, 512, 64]],
            coords_per_arg=[[d0, d1, d2], [d0, zero, d2], [d0, d1, d2]],
        )
    if form == "stat":
        return make_op_spec(
            "realdiv",
            sizes=[[16, 512, 64], [512, 64], [16, 512, 64]],
            coords_per_arg=[[d0, d1, d2], [d1, zero], [d0, d1, d2]],
        )
    if form == "splat":
        return make_op_spec(
            "layernormscale",
            inputs=1,
            sizes=[[512, 64], [512, 64]],
            coords_per_arg=[[d0, zero], [d0, d1]],
            arrangements=[ElementArrangement.EXX2, ElementArrangement.STANDARD],
        )
    raise ValueError(f"make_broadcast_op_spec: unknown form {form!r}")


def make_statistic_reader_specs(reader: str = "realdiv") -> list:
    """A reduction, and a pointwise stage that READS the statistic it wrote."""
    rows, reduced = sympy.symbols("c0 c1")
    stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
    zero = sympy.Integer(0)
    statistic = [zero, rows, zero]
    produce = make_op_spec(
        "sum",
        is_reduction=True,
        inputs=1,
        names=["x0", "buf0"],
        sizes=[[2, 256, 64], [1, 256, 64]],
        coords_per_arg=[[stick, rows, lane], statistic],
        space={rows: (256, 1), reduced: (128, 1)},
    )
    e0, e1, e2 = sympy.symbols("e0 e1 e2")
    consume = make_op_spec(
        reader,
        inputs=2,
        names=["x1", "buf0", "out0"],
        sizes=[[2, 256, 64], [1, 256, 64], [2, 256, 64]],
        coords_per_arg=[[e0, e1, e2], [zero, e1, zero], [e0, e1, e2]],
        space={e0: (2, 1), e1: (256, 1), e2: (64, 1)},
        first_arg_index=2,
    )
    # ``buf0`` is ONE buffer at ONE index, which the per-spec numbering cannot
    # know; said here for the same reason ``TestAStageOwnsItsViews`` says it.
    consume.args[1].arg_index = 1  # buf0, as stage 0 numbered it
    consume.args[2].arg_index = 3  # out0, after x1
    return [produce, consume]


def make_two_element_type_specs() -> list:
    """Two stages over ONE buffer, each reading it at a different element type."""
    rows, reduced = sympy.symbols("c0 c1")
    stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
    zero = sympy.Integer(0)
    fused = ElementArrangement.EXX2
    produce = make_op_spec(
        "exx2",
        is_reduction=True,
        inputs=1,
        names=["x0", "pair"],
        sizes=[[2, 256, 64], [1, 256, 64]],
        coords_per_arg=[[stick, rows, lane], [zero, rows, zero]],
        arrangements=[None, fused],
        space={rows: (256, 1), reduced: (128, 1)},
    )
    e0, e1, e2 = sympy.symbols("e0 e1 e2")
    full, statistic = [e0, e1, e2], [zero, e1, zero]
    consume = make_op_spec(
        "layernormnorm",
        inputs=5,
        names=["x1", "pair", "s2", "s3", "s4", "out0"],
        sizes=[[2, 256, 64], [1, 256, 64], *([[2, 256, 64]] * 4)],
        coords_per_arg=[full, statistic, full, full, full, full],
        # The flag is on the BUFFER, so the reader's arg carries it too; that the
        # read is nonetheless f16 is the recipe's word and not the arg's.
        arrangements=[None, fused, None, None, None, None],
        space={e0: (2, 1), e1: (256, 1), e2: (64, 1)},
        first_arg_index=2,
    )
    # ``pair`` is ONE buffer at ONE index, which the per-spec numbering cannot
    # know; the args after it close the gap that leaves.
    consume.args[1].arg_index = 1  # pair, as stage 0 numbered it
    for position, index in enumerate((3, 4, 5, 6), start=2):
        consume.args[position].arg_index = index
    return [produce, consume]


def make_linked_op_specs(
    ops: tuple = ("abs", "max"),
    *,
    reductions: tuple = (False, True),
    edges: tuple = ((0, 1),),
    dangling: tuple = (),
    prefixes: tuple | None = None,
    link: dict | None = None,
    links: dict | None = None,
    link_local: bool = True,
    out_sizes: dict | None = None,
    out_coords: dict | None = None,
    in_sizes: dict | None = None,
    onstick: bool = False,
    chunks: int | None = None,
    rows: int = 256,
    lanes: int = 64,
    dtype: DataFormats = FP16,
    row_division: int = 1,
) -> list:
    """A kernel's OpSpec vector described TOPOLOGICALLY: stages and their links.

    Every link is ``kernel_local`` -- nothing outside the kernel reads it, which
    is what a link is -- unless ``link_local=False`` says the scheduler found a
    reader elsewhere.
    """
    if len(ops) != len(reductions):
        raise ValueError("make_linked_op_specs: one reduction flag per op")
    prefixes = prefixes or tuple(chr(ord("d") + index) for index in range(len(ops)))
    link = link or ({"lx": 0x1000} if onstick else {"hbm_pool": 0x2000})
    links, out_sizes, out_coords, in_sizes = (
        links or {},
        out_sizes or {},
        out_coords or {},
        in_sizes or {},
    )
    chunks = (2 if onstick else 64) if chunks is None else chunks
    full_size = [chunks, rows, lanes]
    reduced_size = [1, rows, lanes]

    def geometry(prefix: str) -> tuple[list, list, dict]:
        """One stage's coordinates for a full and a reduced buffer, and its space."""
        s0, s1, s2 = sympy.symbols(f"{prefix}0:3")
        if onstick:
            full = [sympy.floor(s1 / lanes), s0, sympy.Mod(s1, lanes)]
            reduced = [sympy.Integer(0), s0, sympy.Integer(0)]
            space = {s0: (rows, row_division), s1: (chunks * lanes, 1)}
        else:
            full = [s0, s1, s2]
            reduced = [sympy.Integer(0), s1, s2]
            space = {s0: (chunks, 1), s1: (rows, row_division), s2: (lanes, 1)}
        return full, reduced, space

    specs: list = []
    next_arg = 0
    for index, op in enumerate(ops):
        full, reduced, space = geometry(prefixes[index])
        incoming = [producer for producer, consumer in edges if consumer == index]
        outgoing = index in dangling or any(producer == index for producer, _ in edges)
        if incoming:
            names = [f"t{producer}" for producer in incoming]
            allocations = [dict(links.get(producer, link)) for producer in incoming]
            # A read is described at the extent its producer wrote, in the
            # READER's symbols: whose description is kept is the fuser's problem,
            # so the fixture must not make the two accidentally identical.
            sizes = [
                in_sizes.get(index)
                or (reduced_size if reductions[producer] else full_size)
                for producer in incoming
            ]
            coords = [
                reduced if reductions[producer] else full for producer in incoming
            ]
        else:
            names, allocations = [f"x{index}"], [None]
            sizes, coords = [in_sizes.get(index) or full_size], [full]
        reduction = reductions[index]
        # A reduction folds axis 0 away; a pointwise stage is the IDENTITY on its
        # first operand, which is what makes it access-preserving and is why the
        # result follows that operand rather than the stage's nominal extent.
        result_size = reduced_size if reduction else sizes[0]
        result_coords = reduced if reduction else coords[0]
        spec = make_op_spec(
            op,
            inputs=len(names),
            is_reduction=reduction,
            names=[*names, f"t{index}" if outgoing else f"out{index}"],
            sizes=[*sizes, out_sizes.get(index) or result_size],
            coords_per_arg=[*coords, out_coords.get(index) or result_coords],
            allocations=[
                *allocations,
                dict(links.get(index, link)) if outgoing else None,
            ],
            kernel_locals=[
                *([link_local] * len(names) if incoming else [False]),
                link_local and bool(outgoing),
            ],
            dtype=dtype,
            space=space,
            first_arg_index=next_arg,
        )
        specs.append(spec)
        next_arg += sum(1 for arg in spec.args if arg.arg_index >= 0)
    return specs


def make_absmax_pair(**overrides) -> list:
    """``amax(abs(x), ...)``: shape A, and the only fixture that names the entry."""
    return make_linked_op_specs(ops=("abs", "max"), **overrides)


def make_plan_fusion(**overrides) -> ktir.PlanFusion:
    """A table entry defined by the test, defaulting to a two-slot collapse."""
    entry = {
        "name": "probe",
        "pattern": (("abs", False), ("max", True)),
        "result_op": "fused",
        "why": "a probe entry, defined by the test that uses it",
    }
    entry.update(overrides)
    return ktir.PlanFusion(**entry)  # type: ignore[arg-type]


class TestValidateRejections(unittest.TestCase):
    """One test per rejection ``build_kernel_plan`` is responsible for.

    Each asserts the exception type and a distinguishing fragment of the
    message, so a rejection cannot silently turn into a different rejection.

    ``_rejects`` defaults to the symbolic address form, which reads no
    ``allocation["hbm"]``, so the rejection under test is the one the fixture is
    about rather than a missing address.
    """

    def _rejects(self, specs, fragment, **options):
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs, ktir.PlanOptions(**options))
        self.assertIn(fragment, str(ctx.exception))

    # -- whole-request capability ------------------------------------------

    def test_empty_spec_list_rejected(self):
        self._rejects([], "no OpSpec to emit")

    def test_mixed_work_division_rejected(self):
        """Two ops in one kernel, two grids: there is only one grid to emit."""
        specs = [make_op_spec(), make_op_spec(divisions={"d1": 2})]
        self._rejects(specs, "different work divisions")

    def test_ragged_work_division_rejected(self):
        """A division that does not divide the axis evenly has no per-core tile."""
        specs = [make_op_spec(divisions={"d1": 7})]  # 512 / 7 is not a whole tile
        self._rejects(specs, "do not divide evenly")

    # -- spec-tree shape ---------------------------------------------------

    def test_unimplemented_op_rejected(self):
        self._rejects([UnimplementedOp(op="atan2")], "unimplemented op 'atan2'")

    def test_unexpected_entry_rejected(self):
        self._rejects(["not a spec"], "unexpected spec entry str")

    def test_family_mismatch_rejected(self):
        """An ``add`` asked for as a reduction: the recipe is what has an
        emission, so the request is refused rather than emitted elementwise."""
        specs = [make_op_spec(is_reduction=True)]
        self._rejects(specs, "registered as NAMED")

    def test_unregistered_op_rejected(self):
        """An op with no recipe is rejected, and the message names what exists."""
        self.assertNotIn("atan2", ktir.KtirBuilder.RECIPES)
        self._rejects([make_op_spec("atan2")], "op 'atan2' is not supported yet")

    # -- per-op roles ------------------------------------------------------

    def test_multiple_outputs_rejected(self):
        specs = [make_op_spec(inputs=1, outputs=2)]
        self._rejects(specs, "expected exactly one output, got 2")

    def test_wrong_arity_rejected(self):
        self._rejects([make_op_spec(inputs=1)], "'add' expects 2 inputs, got 1")

    def test_in_place_rejected(self):
        """The output names an input, which is the aliasing this cannot emit."""
        specs = [make_op_spec(names=["arg0", "arg1", "arg0"])]
        self._rejects(specs, "in-place ops (input aliases output)")

    def test_stretched_operand_rejected(self):
        """A unit extent on an axis the operand's coordinate says it WALKS."""
        specs = [make_op_spec(sizes=[[1, 512, 64]])]
        self._rejects(specs, "not a stretch of it")

    def test_a_broadcast_operand_of_a_named_linalg_op_rejected(self):
        """A named ``linalg`` op states its own (identity) indexing, so a derived"""
        named_only = ktir.Recipe(
            arity=2,
            arms=ktir.Arm(kind=ktir.BindingKind.NAMED, binding=lambda: None),
        )
        specs = [
            make_op_spec(
                "named_only",
                sizes=[[16, 512, 1]],
                coords_per_arg=[
                    [*sympy.symbols("d0:2"), sympy.Integer(0)],
                    sympy.symbols("d0:3"),
                    sympy.symbols("d0:3"),
                ],
            )
        ]
        with mock.patch.dict(ktir.KtirBuilder.RECIPES, {"named_only": named_only}):
            self._rejects(specs, "named linalg op, which states its own indexing")

    # -- per-buffer --------------------------------------------------------

    def test_non_kernel_argument_buffer_rejected(self):
        """arg_index stays -1 for LX / HBM-pool buffers; only HBM is emitted.

        Set here rather than asked of the builder: an HBM buffer that is *also*
        not a kernel argument is the contradiction under test, and the builder
        ties -1 to a non-HBM allocation precisely so it cannot produce one.
        """
        specs = [make_op_spec()]
        specs[0].args[0].arg_index = -1
        self._rejects(specs, "is not a kernel argument")

    def test_symbolic_trip_count_rejected(self):
        nest = LoopSpec(count=sympy.Symbol("s0"), body=[make_op_spec()])
        self._rejects([nest], "trip count s0 is symbolic")

    def test_unsupported_dtype_rejected(self):
        self._rejects([make_op_spec(dtype=DataFormats.SENINT8)], "unsupported device")
        self.assertNotIn(DataFormats.SENINT8, ktir.ElemTypes.NAMES)

    def test_baked_non_hbm_allocation_rejected(self):
        """An allocation that is neither HBM nor one this emitter threads.

        Set here for the same reason as ``test_non_kernel_argument_buffer``: the
        builder would read an unrecognised allocation as an intermediate and give
        it -1, which is a different rejection than the one under test.
        """
        specs = [make_op_spec(baked=True)]
        specs[0].args[0].allocation = {"somewhere_new": 0x1000}
        self._rejects(specs, "is not HBM-allocated", bake_addresses=True)

    def test_threaded_input_without_a_producer_rejected(self):
        """An lx buffer this kernel reads but does not produce: threading it has
        no value to read, so it needs materialising."""
        specs = [make_op_spec(allocations=[{"lx": 0x1000}])]
        self._rejects(specs, "no op in this kernel produces it")

    def test_baked_unassigned_hbm_address_rejected(self):
        # [make_op_spec()] leaves every 'hbm' address None.
        self._rejects([make_op_spec()], "unassigned 'hbm' address", bake_addresses=True)


class TestRejectionsThroughGenerateKtir(unittest.TestCase):
    """``generate_ktir`` surfaces the rejections *without* reaching the dialect.

    These would pass vacuously if ``generate_ktir`` validated after importing
    ``mlir_ktdp``; they run here precisely because it validates first.
    """

    def test_family_mismatch_unsupported(self):
        specs = [make_op_spec(is_reduction=True)]
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs)

    def test_unregistered_op_unsupported(self):
        specs = [make_op_spec()]
        specs[0].op = "atan2"
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_atan2_0", specs)

    def test_ragged_work_division_unsupported(self):
        specs = [make_op_spec(divisions={"d1": 7})]
        with self.assertRaises(NotImplementedError):
            ktir.generate_ktir("ktir_fused_add_0", specs)

    def test_unknown_option_is_a_typeerror(self):
        """Options are PlanOptions fields; a typo is not silently ignored."""
        with self.assertRaises(TypeError) as ctx:
            ktir.generate_ktir("k", [make_op_spec()], bake_address=True)
        self.assertIn("bake_address", str(ctx.exception))


class TestPlanOptions(unittest.TestCase):
    """The caller's one choice, and it is about spelling, not capability.

    What the kernel does comes from the contract, so there is nothing here to
    turn a feature on with: no core count and no loop mode (a ``LoopSpec`` is a
    loop).
    """

    def test_defaults_are_the_canonical_form(self):
        options = ktir.PlanOptions()
        self.assertFalse(options.bake_addresses)  # symbolic addresses

    def test_options_are_only_about_spelling(self):
        self.assertEqual(
            sorted(f.name for f in dataclasses.fields(ktir.PlanOptions)),
            ["bake_addresses"],
        )


class TestWorkDivision(unittest.TestCase):
    """The grid, and the per-core tile, as ``iteration_space`` states them.

    ``work_division.py`` has already turned ``config.sencores`` into a per-symbol
    division by the time the emitter sees a spec, so the emitter reads the
    contract and never the config -- the same source the SDSC path reads as its
    work slices.
    """

    def test_an_undivided_space_is_one_core(self):
        plan = ktir.build_kernel_plan([make_op_spec()])
        self.assertEqual(plan.grid, (1,))
        self.assertEqual(plan.divisions, ())

    def test_the_grid_is_the_product_of_the_divisions(self):
        plan = ktir.build_kernel_plan([make_op_spec(divisions={"d1": 32})])
        self.assertEqual(plan.grid, (32,))
        self.assertEqual(plan.divisions, (ktir.Division(symbol="d1", div=32, inner=1),))

    def test_two_divided_symbols_are_mixed_radix(self):
        """Outermost-first, and ``inner`` is that symbol's stride in the grid."""
        plan = ktir.build_kernel_plan([make_op_spec(divisions={"d0": 2, "d1": 4})])
        self.assertEqual(plan.grid, (8,))
        self.assertEqual(
            plan.divisions,
            (
                ktir.Division(symbol="d0", div=2, inner=4),
                ktir.Division(symbol="d1", div=4, inner=1),
            ),
        )

    def test_the_tile_shrinks_and_the_view_does_not(self):
        """One core's tile is its share; every core addresses the whole buffer."""
        plan = ktir.build_kernel_plan([make_op_spec(divisions={"d1": 32})])
        for buffer in plan.parameters:
            with self.subTest(buf_id=buffer.buf_id):
                self.assertEqual(buffer.layout.extent, (16, 512, 64))
        step = plan.steps[0]
        self.assertEqual(step.out.extent, (16, 16, 64))  # 512 / 32 rows
        # The division walks dim 1 in per-core-extent steps, and nothing else.
        self.assertEqual(step.out.index_coeffs, ((0,), (16,), (0,)))

    def test_a_division_no_output_axis_follows_is_rejected(self):
        """A stick is the unit of transfer, so the lane axis is never divided --
        which leaves a division of the lane symbol with no axis to walk, and every
        core writing the same elements.  Refused rather than silently duplicated."""
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan([make_op_spec(divisions={"d2": 2})])
        self.assertIn("no device axis of the output", str(ctx.exception))


class TestKernelPlan(unittest.TestCase):
    """What ``build_kernel_plan`` returns: the func signature, before any emission."""

    def test_param_entries_are_ordered_by_arg_index(self):
        specs = [make_op_spec()]
        # Registration order (spec.args) is 0, 1, 2; shuffle it so the sort is
        # doing the work rather than agreeing with insertion order by luck.
        specs[0].args = [specs[0].args[2], specs[0].args[0], specs[0].args[1]]
        plan = ktir.build_kernel_plan(specs)
        self.assertEqual([e.arg_index for e in plan.parameters], [0, 1, 2])
        self.assertEqual([e.buf_id for e in plan.parameters], ["arg0", "arg1", "buf0"])
        # The plan holds the derived records, so the buffer's extent and its
        # row-major strides are readable here rather than only in the MLIR.
        self.assertEqual(plan.parameters[0].layout.extent, (16, 512, 64))
        self.assertEqual(plan.parameters[0].layout.strides, (32768, 64, 1))

    def test_symbolic_form_resolves_no_base_addresses(self):
        plan = ktir.build_kernel_plan([make_op_spec()])
        # Every 'hbm' address in the fixture is None and never read: the bases
        # are func arguments.
        self.assertEqual([e.base_elements for e in plan.parameters], [None] * 3)

    def test_baked_form_resolves_bases_in_elements(self):
        plan = ktir.build_kernel_plan(
            [make_op_spec(baked=True)],
            ktir.PlanOptions(bake_addresses=True),
        )
        # fp16: 2 bytes per element, so the byte slot halves.
        self.assertEqual(
            [e.base_elements for e in plan.parameters],
            [0, (1 << 34) // 2, (2 << 34) // 2],
        )

    def test_repeated_buffer_is_registered_once(self):
        specs = [make_op_spec()] + [make_op_spec()]
        plan = ktir.build_kernel_plan(specs)
        self.assertEqual(len(plan.buffers), 3)


class TestBaseAddressElements(unittest.TestCase):
    """``_base_address_elements`` in isolation, with no dialect and no config."""

    @staticmethod
    def _arg(allocation):
        """One input carrying ``allocation``, taken out of a whole spec."""
        return make_op_spec(allocations=[None, allocation]).args[1]

    def test_byte_address_scales_to_elements(self):
        # fp16: 2 bytes per element.  Zero is a real address, not "unset".
        self.assertEqual(
            ktir._base_address_elements(self._arg({"hbm": 1 << 34})), 1 << 33
        )
        self.assertEqual(ktir._base_address_elements(self._arg({"hbm": 0})), 0)

    def test_unassigned_or_non_hbm_rejected(self):
        for allocation in ({"hbm": None}, {"lx": 0x1000}, {"hbm_pool": 0x1000}, {}):
            with (
                self.subTest(alloc=allocation),
                self.assertRaises(NotImplementedError),
            ):
                ktir._base_address_elements(self._arg(allocation))


class TestInternalBufferSignal(unittest.TestCase):
    """``is_internal`` decides materialise-vs-thread, from ``allocation``.

    The same field ``create_tensor_arg`` uses to decide what becomes a kernel
    argument at all, so the two cannot disagree about which buffers the kernel
    owns.
    """

    def test_an_hbm_buffer_is_passed_in_not_owned(self):
        for arg in make_op_spec().args:
            self.assertFalse(ktir.is_internal(arg))

    def test_planning_placed_it_means_the_kernel_owns_it(self):
        for allocation in ({"lx": 0x1000}, {"hbm_pool": 0x2000}):
            with self.subTest(allocation=allocation):
                spec = make_op_spec(allocations=[None, None, allocation])
                self.assertTrue(ktir.is_internal(spec.args[-1]))

    def test_an_unrecognised_allocation_is_not_threaded(self):
        """Threading is chosen on a positive signal, so an allocation this
        emitter does not know reaches the buffer rejection instead."""
        spec = make_op_spec(allocations=[None, None, {"somewhere_new": 0}])
        self.assertFalse(ktir.is_internal(spec.args[-1]))

    def test_a_threaded_buffer_nothing_reads_is_rejected(self):
        """An intermediate whose consumer is in another kernel: not stored, and
        not read here either, so the op that produced it would write nowhere."""
        specs = [make_op_spec(allocations=[None, None, {"lx": 0x1000}])]
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("nothing in this kernel", str(ctx.exception))


class TestRecipes(unittest.TestCase):
    """One recipe per op, and every surface the plan can pick has an arm."""

    def test_every_recipe_is_complete(self):
        self.assertTrue(ktir.KtirBuilder.RECIPES)
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            with self.subTest(op=op):
                self.assertGreaterEqual(recipe.arity, 1)
                self.assertTrue(recipe.arms)
                # A reader, not the values: resolving one needs an ``op_info``.
                self.assertTrue(recipe.attrs is None or callable(recipe.attrs))
                for index, arm in enumerate(recipe.arms):
                    with self.subTest(arm=index):
                        self.assertIsInstance(arm.kind, ktir.BindingKind)
                        # A thunk, not the builder itself: resolving it here would
                        # need the dialect, which this module deliberately does
                        # not require.
                        self.assertTrue(callable(arm.binding))
                # A one-armed op has to be reachable at every format, so that arm
                # cannot list any: a lone arm claiming a format would make
                # ``Recipe.arm`` refuse every other one.
                if len(recipe.arms) == 1:
                    self.assertEqual(recipe.arms[0].dtypes, ())

        # Every kind is now registered by some arm, so the mirror assertion is
        # worth making: PAYLOAD stopped being a hook nothing reaches when the
        # ``spyreop`` intrinsics landed on it.
        self.assertEqual(
            {arm.kind for r in ktir.KtirBuilder.RECIPES.values() for arm in r.arms},
            set(ktir.BindingKind),
        )

        # Which surface a step gets is the plan's choice, not a recipe's, so
        # completeness on this side is about ``compute`` rather than about any one
        # op: every ``Surface`` must appear as a ``case`` pattern.  Read off the
        # AST because ``case _:`` alone turns a missing arm into a runtime
        # discovery, at which point a module is already half built.
        tree = ast.parse(inspect.getsource(ktir))
        builder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KtirBuilder"
        )
        compute = next(
            node
            for node in builder.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute"
        )
        cased = {
            node.pattern.value.attr
            for node in ast.walk(compute)
            if isinstance(node, ast.match_case)
            and isinstance(node.pattern, ast.MatchValue)
            and isinstance(node.pattern.value, ast.Attribute)
        }
        for surface in ktir.Surface:
            self.assertIn(surface.name, cased, f"compute has no case for {surface}")

    def test_recipe_rejects_a_nonsense_arity(self):
        """A duplicate op name is ruff F601; arity is checked at construction."""
        with self.assertRaises(ValueError):
            ktir.Recipe(arity=0, arms=self._arm())

    def test_a_lone_arm_is_promoted_to_a_tuple(self):
        """``arms=Arm(...)`` and ``arms=(Arm(...),)`` are the same recipe.

        Asserted because the shorthand would otherwise be a second representation
        of the field: anything reading ``recipe.arms`` directly must see a tuple
        however the entry was written, or it iterates an ``Arm``'s attributes.
        """
        arm = self._arm()
        self.assertEqual(ktir.Recipe(arity=1, arms=arm).arms, (arm,))
        self.assertEqual(ktir.Recipe(arity=1, arms=(arm,)).arms, (arm,))
        # And every registered entry has been normalised, whichever form it used.
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            with self.subTest(op=op):
                self.assertIsInstance(recipe.arms, tuple)

    @staticmethod
    def _arm(*dtypes):
        return ktir.Arm(
            kind=ktir.BindingKind.NAMED, binding=lambda: None, dtypes=tuple(dtypes)
        )

    def test_recipe_rejects_an_ambiguous_arm_set(self):
        """The two ways a format could resolve to more than one arm.

        Both are refused where the table is written rather than at the lookup,
        because a table that can be read two ways is wrong however it is read --
        and ``Recipe.arm`` returning the first match would make which arm wins a
        fact about declaration order.
        """
        with self.assertRaises(ValueError):
            ktir.Recipe(arity=1, arms=())
        with self.assertRaises(ValueError):
            # Two arms claiming every unlisted format.
            ktir.Recipe(arity=1, arms=(self._arm(), self._arm()))
        with self.assertRaises(ValueError):
            # Two arms claiming the same format.
            ktir.Recipe(
                arity=1,
                arms=(
                    self._arm(DataFormats.IEEE_INT32),
                    self._arm(DataFormats.IEEE_INT32),
                ),
            )

    def test_an_op_with_two_spellings_resolves_on_the_format(self):
        """``add`` is a named linalg op at floats and a spyreop payload at int32.

        The point of the arms: one entry per op, and the format picks the spelling.
        Asserted on the recipe rather than through a plan so it holds without a
        dialect build -- the bindings stay unresolved thunks.
        """
        recipe = ktir.KtirBuilder.RECIPES["add"]
        self.assertIs(recipe.arm(DataFormats.SEN169_FP16).kind, ktir.BindingKind.NAMED)
        self.assertIs(recipe.arm(DataFormats.IEEE_INT32).kind, ktir.BindingKind.PAYLOAD)
        # Arity is the op's, not the arm's, so both spellings agree on it by
        # construction rather than by two entries happening to match.
        self.assertEqual(recipe.arity, 2)

    def test_an_op_with_one_spelling_reaches_it_at_every_format(self):
        """``sub`` has no integer intrinsic, so its one arm takes every format."""
        recipe = ktir.KtirBuilder.RECIPES["sub"]
        for dtype in (DataFormats.SEN169_FP16, DataFormats.IEEE_INT32, None):
            with self.subTest(dtype=dtype):
                self.assertIs(recipe.arm(dtype).kind, ktir.BindingKind.NAMED)

    def test_the_format_reaches_the_step_and_picks_the_surface(self):
        """An int32 ``add`` plans as a generic, and the step carries the format.

        The whole path in one assertion: the spec's format picks the payload arm,
        the payload arm picks ``Surface.GENERIC`` (a scalar builder needs a region),
        and the format lands on the step so emission resolves the same arm without
        seeing the spec.
        """
        spec = make_op_spec("add", dtype=DataFormats.IEEE_INT32)
        [step] = ktir.build_kernel_plan([spec]).steps
        self.assertIs(step.dtype, DataFormats.IEEE_INT32)
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        # The same op at fp16 is the named linalg op, which states its own
        # indexing and so needs no record.
        [float_step] = ktir.build_kernel_plan([make_op_spec("add")]).steps
        self.assertIs(float_step.surface, ktir.Surface.BARE)
        self.assertIsNone(float_step.indexing)

    def test_a_spec_that_mixes_formats_is_refused_by_the_plan(self):
        """No arm resolves a mixed request, so the plan refuses to guess one.

        Taking any single operand's format would emit an intrinsic for the wrong
        type on the others, and the old ``any(... == INT32)`` rule did exactly that
        for one int32 operand among floats.
        """
        spec = make_op_spec("add")
        mixed = dataclasses.replace(
            spec,
            args=[
                dataclasses.replace(spec.args[0], device_dtype=DataFormats.IEEE_INT32),
                *spec.args[1:],
            ],
        )
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan([mixed])
        self.assertIn("mixes device formats", str(ctx.exception))

    def test_a_format_no_arm_takes_is_refused(self):
        """An op with only a claimed arm does not exist at any other format.

        The membership question the two-table arrangement got wrong: an op is
        supported at a format or it is not, and there is no second table to fall
        back out of.
        """
        recipe = ktir.Recipe(arity=1, arms=(self._arm(DataFormats.IEEE_INT32),))
        self.assertIs(recipe.arm(DataFormats.IEEE_INT32).kind, ktir.BindingKind.NAMED)
        with self.assertRaises(NotImplementedError) as ctx:
            recipe.arm(DataFormats.SEN169_FP16)
        self.assertIn("no arm for", str(ctx.exception))

    def test_a_reduction_asked_for_elementwise_is_rejected(self):
        """The other direction of the agreement check, and the dangerous one.

        ``sum``'s binding is a two-operand combiner; with nothing labelled as
        reduced it would be handed a single operand and fail *inside* emission,
        with a half-built module in hand.  Refused by the plan instead.
        """
        specs = [make_op_spec("sum", inputs=1, is_reduction=False)]
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("registered as COMBINER", str(ctx.exception))
        self.assertIn("elementwise", str(ctx.exception))

    def test_emit_asserts_on_an_unplanned_step(self):
        """The emitter's only remaining ``raise`` is this plan-bug guard.

        Called unbound with ``self=None``: the type check happens before any
        builder state is touched, which is why this needs no dialect build.
        ``UnimplementedOp`` cannot reach emission at all now -- a step tree holds
        only steps -- so the guard is about a malformed plan, not a rejected op.
        """
        with self.assertRaises(AssertionError):
            ktir.KtirBuilder.emit(None, [UnimplementedOp(op="atan2")])


class TestArmDispatch(unittest.TestCase):
    """Selecting an arm on MORE than the format: the second discriminant."""

    @staticmethod
    def _arm(kind=ktir.BindingKind.NAMED, *dtypes):
        return ktir.Arm(kind=kind, binding=lambda: None, dtypes=tuple(dtypes))

    def test_the_default_dispatcher_is_the_format_alone(self):
        """Every entry that did not ask for the new discriminant ignores it."""
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            if recipe.dispatch is not ktir.request_by_dtype:
                continue
            for dtype in (*ktir.ElemTypes.NAMES, None):
                with self.subTest(op=op, dtype=dtype):
                    self.assertIs(recipe.arm(dtype), recipe.arm(dtype, broadcast=True))

    def test_a_dispatcher_may_not_return_a_foreign_arm(self):
        """A dispatcher narrows; it does not invent."""
        foreign = self._arm()
        recipe = ktir.Recipe(
            arity=1, arms=self._arm(), dispatch=lambda arms, request: foreign
        )
        with self.assertRaises(AssertionError):
            recipe.arm(FP16)

    def test_two_default_arms_are_one_kind_ambiguous_and_two_kinds_a_channel_each(
        self,
    ):
        """The one-default-arm rule is per KIND, and why it has to be."""
        with self.assertRaises(ValueError):
            ktir.Recipe(arity=1, arms=(self._arm(), self._arm()))
        recipe = ktir.Recipe(
            arity=1,
            arms=(self._arm(), self._arm(ktir.BindingKind.PAYLOAD)),
            dispatch=ktir.request_scalar_when_broadcast,
        )
        self.assertIs(recipe.arm(FP16).kind, ktir.BindingKind.NAMED)
        self.assertIs(recipe.arm(FP16, broadcast=True).kind, ktir.BindingKind.PAYLOAD)

    def test_all_arms_of_an_op_must_agree_on_whether_it_reduces(self):
        """What makes ``Recipe.reduces`` sound, and it is asked before any arm."""
        with self.assertRaises(ValueError):
            ktir.Recipe(
                arity=1,
                arms=(self._arm(), self._arm(ktir.BindingKind.COMBINER, FP16)),
            )
        # And the two shapes that do agree are fine, whichever way they agree.
        self.assertFalse(
            ktir.Recipe(
                arity=1,
                arms=(self._arm(), self._arm(ktir.BindingKind.PAYLOAD, FP16)),
            ).reduces
        )
        self.assertTrue(
            ktir.Recipe(arity=1, arms=self._arm(ktir.BindingKind.COMBINER)).reduces
        )

    def test_a_reduction_mismatch_is_refused_before_any_arm_is_chosen(self):
        """The early family check asks the RECIPE, and asks nothing else."""

        def never(arms, request):
            raise AssertionError("the family check chose an arm")

        recipe = dataclasses.replace(ktir.KtirBuilder.RECIPES["add"], dispatch=never)
        with mock.patch.dict(ktir.KtirBuilder.RECIPES, {"add": recipe}):
            with self.assertRaises(NotImplementedError) as ctx:
                ktir.build_kernel_plan([make_op_spec(is_reduction=True)])
        self.assertIn("is registered as NAMED", str(ctx.exception))
        self.assertIn("a reduction", str(ctx.exception))

    def test_a_broadcast_operand_takes_the_scalar_arm_and_an_aligned_one_the_named(
        self,
    ):
        """The three entries' resolution table, and how it composes with format."""
        for op in ("add", "mul", "sub"):
            recipe = ktir.KtirBuilder.RECIPES[op]
            with self.subTest(op=op):
                self.assertIs(recipe.arm(FP16).kind, ktir.BindingKind.NAMED)
                self.assertIs(
                    recipe.arm(FP16, broadcast=True).kind, ktir.BindingKind.PAYLOAD
                )
        int32 = DataFormats.IEEE_INT32
        for op in ("add", "mul"):
            with self.subTest(op=op):
                arm = ktir.KtirBuilder.RECIPES[op].arm(int32, broadcast=True)
                self.assertIs(arm.kind, ktir.BindingKind.PAYLOAD)
                self.assertEqual(arm.dtypes, (int32,))
        self.assertIs(
            ktir.KtirBuilder.RECIPES["sub"].arm(int32, broadcast=True).kind,
            ktir.BindingKind.NAMED,
        )
        # The float scalars are dtype-less, so what they do NOT serve is listed
        # once, in ``_INTEGER_FORMATS``.  Kept in step with the supported-format
        # table here, because a new integer format added to ``NAMES`` alone would
        # silently resolve to ``arith.addf``.
        for dtype, spelling in ktir.ElemTypes.NAMES.items():
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    spelling.startswith("i"), dtype in ktir._INTEGER_FORMATS
                )

    def test_the_broadcast_flag_is_derived_from_coordinates_alone_and_reaches_step(
        self,
    ):
        """``broadcast`` is on the step for the reason ``dtype`` is."""
        for form in ("row", "stat", "splat"):
            with self.subTest(form=form):
                [step] = ktir.build_kernel_plan([make_broadcast_op_spec(form)]).steps
                self.assertTrue(step.broadcast)
        [aligned] = ktir.build_kernel_plan([make_op_spec()]).steps
        self.assertFalse(aligned.broadcast)

    def test_a_broadcast_sub_is_a_generic_and_an_aligned_one_is_still_the_named_op(
        self,
    ):
        """The gap this closes, end to end through the plan."""
        d0, d1, d2 = sympy.symbols("d0 d1 d2")
        broadcast = make_op_spec(
            "sub",
            sizes=[[16, 512, 64], [16, 1, 64], [16, 512, 64]],
            coords_per_arg=[
                [d0, d1, d2],
                [d0, sympy.Integer(0), d2],
                [d0, d1, d2],
            ],
        )
        [step] = ktir.build_kernel_plan([broadcast]).steps
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(step.indexing.maps, ((0, 1, 2), (0, None, 2), (0, 1, 2)))
        for op in ("add", "mul", "sub"):
            with self.subTest(op=op):
                [aligned] = ktir.build_kernel_plan([make_op_spec(op)]).steps
                self.assertIs(aligned.surface, ktir.Surface.BARE)
                self.assertIsNone(aligned.indexing)


class TestReduceSurface(unittest.TestCase):
    """Which of the two reduction shapes a loop nest can be emitted as.

    ``linalg.reduce`` is the compact spelling: hand it the dimensions to fold
    away and it works out the rest itself.  The price is that it can only say a
    reduction that reads its input with one loop per input dimension and leaves
    the surviving dimensions where they were.  Anything else has to be a
    ``linalg.generic``, which spells the correspondence out in full.  These tests
    go straight at that rule -- no spec is involved.
    """

    def test_a_plain_reduction_can_be_a_linalg_reduce(self):
        """Fold away the middle dimension of three, keep the other two in order."""
        self.assertIs(
            ktir._reduce_surface(
                ("parallel", "reduction", "parallel"), (0, 1, 2), (0, 2)
            ),
            ktir.Surface.REDUCE,
        )

    def test_a_reduction_over_the_stick_cannot_be_a_linalg_reduce(self):
        """The on-stick sum, and the reason it is worth a test of its own.

        Judged on its output alone, ``(1, 3)`` reads as "keep dimensions 1 and 3
        of four, fold away 0 and 2" -- which ``linalg.reduce`` says perfectly
        well.  What it cannot say is the input side: three input dimensions
        addressed by a loop nest of four, because the 64 lanes are read as one
        dimension and written as a different one.  ``linalg.reduce`` always reads
        its input with exactly one loop per input dimension.

        So if this rule is ever relaxed to look only at the output, this is the
        test that fails -- and without it the emitter would quietly build a
        two-dimensional ``linalg.reduce`` that sums the wrong elements.
        """
        iters = ("reduction", "parallel", "reduction", "parallel")
        self.assertEqual(
            tuple(d for d, it in enumerate(iters) if it == "reduction"), (0, 2)
        )
        self.assertIs(
            ktir._reduce_surface(iters, (0, 1, 2), (1, 3)), ktir.Surface.GENERIC
        )

    def test_a_reduction_that_also_reorders_cannot_be_a_linalg_reduce(self):
        """It folds dimensions away; it never moves the ones that survive.

        Here the two survivors come out swapped, which the compact spelling has
        no way to express.
        """
        self.assertIs(
            ktir._reduce_surface(
                ("parallel", "reduction", "parallel"), (0, 1, 2), (2, 0)
            ),
            ktir.Surface.GENERIC,
        )


class TestOnlyAReductionOutputIsSqueezed(unittest.TestCase):
    """A pointwise op keeps a size-1 output dimension; only a reduction drops one.

    Dropping a size-1 dimension is safe when a reduction left it behind, because
    nothing was ever written along it.  It is not safe in general, and this spec
    is the counterexample: an ``add`` whose operands and output all carry the
    same size-1 dimension.  It compiles today, and it works precisely *because*
    all three agree on it.  Drop it from the output alone and ``linalg.add``
    would be handed a two-dimensional result against three-dimensional operands,
    which fails when the module is verified -- inside emission, the one place
    nothing is allowed to fail.

    So the drop happens only for a reduction, and this test is the reason.
    """

    @staticmethod
    def _size_one_add():
        rows = sympy.Symbol("c1")
        return [
            make_op_spec(
                size=[1, 256, 64],
                coords=[sympy.Integer(0), rows, sympy.Mod(rows, 64)],
            )
        ]

    def test_a_size_one_dimension_is_kept_when_nothing_is_reduced(self):
        plan = ktir.build_kernel_plan(self._size_one_add())
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.BARE)
        self.assertEqual(step.out.extent, (1, 256, 64))
        for _buf_id, access in step.ins:
            self.assertEqual(access.extent, (1, 256, 64))


class TestAnOutputLaneIsNotATranspose(unittest.TestCase):
    """A reduction may write an axis its input reduced; it may not reorder axes.

    Both shapes reach the same matching walk, and before the broadcast lane had a
    home the on-stick one came out of it with the *wrong* diagnostic: its output
    lane matched no input axis, so it was reported as a permutation needing a
    restickify.  It is not a permutation -- nothing moved -- so the two cases have
    to be told apart, and a refusal that still fires for the real thing is what
    says the first case was widened rather than the check being weakened.
    """

    def test_a_reduced_axis_may_be_written_again(self):
        plan = ktir.build_kernel_plan(make_onstick_sum_specs())
        [step] = plan.steps
        self.assertEqual(step.out.extent, (256, 64))

    def test_reordered_surviving_axes_are_still_refused(self):
        """The same reduction with its two kept axes swapped on the way out."""
        lanes, rows = sympy.symbols("c0 c1")
        stick, lane = sympy.floor(lanes / 64), sympy.Mod(lanes, 64)
        specs = [
            make_op_spec(
                "sum",
                is_reduction=True,
                inputs=1,
                sizes=[[32, 256, 64], [64, 32]],
                coords_per_arg=[[stick, rows, lane], [lane, stick]],
                space={lanes: (2048, 1), rows: (256, 1)},
            )
        ]
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("transpose", str(ctx.exception))


class TestAPayloadWithNoNamedOpGetsAGeneric(unittest.TestCase):
    """An elementwise op the dialect has no named op for, and how it is spelled.

    ``sqrt`` is one: its binding is ``spyreop.sqrt``, a *scalar* builder, so there
    is nothing to call it but a region and the step has to state the identity maps
    itself.  Everything here is the plan's choice, made before any dialect is
    reached, which is why these run without a dialect build.
    """

    def test_every_spyreop_intrinsic_is_a_payload(self):
        """The kind is what puts them on the generic, so it is asserted per op.

        Registered as PAYLOAD and not NAMED: a ``spyreop`` op is not a ``linalg``
        named op, and calling one as if it were would hand a scalar builder tensor
        operands inside emission.
        """
        for op in (
            "exp",
            "sqrt",
            "sigmoid",
            "reciprocal",
            "gelufwd",
            "layernormscale",
            "softplus",
        ):
            with self.subTest(op=op):
                recipe = ktir.KtirBuilder.RECIPES[op]
                [arm] = recipe.arms
                self.assertIs(arm.kind, ktir.BindingKind.PAYLOAD)
                self.assertEqual(recipe.arity, 1)

    def test_the_identity_maps_are_stated_rather_than_implied(self):
        plan = ktir.build_kernel_plan([make_op_spec("sqrt", inputs=1)])
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(step.reduce_dims, ())
        # Rank 3, one map per input and then the result: the operand and the
        # destination are read one element at a time in the same order.
        self.assertEqual(step.indexing.iters, ("parallel",) * 3)
        self.assertEqual(step.indexing.maps, ((0, 1, 2), (0, 1, 2)))

    def test_a_scalar_argument_is_read_at_plan_time(self):
        """softplus's two scalars land on the step, so emission derives nothing.

        The values are on the record and the reader is not: what ``op_info`` looks
        like is a fact about the request, and the step is what emission sees.
        """
        spec = make_op_spec(
            "softplus",
            inputs=1,
            op_info={"constants": {"softplusBeta": 1.0, "softplusThresh": 20.0}},
        )
        [step] = ktir.build_kernel_plan([spec]).steps
        self.assertEqual(step.attrs, (("beta", 1.0), ("threshold", 20.0)))

    def test_an_op_with_no_scalar_arguments_carries_none(self):
        """``attrs`` is empty for every op that is a function of its operands.

        Asserted over every registered recipe rather than one, so an ``attrs``
        reader added to an op that does not want one shows up here.
        """
        for op, recipe in ktir.KtirBuilder.RECIPES.items():
            # A reduction wants coordinates that actually reduce, which its own
            # fixtures own; the claim here is about the pointwise ops.  Asked of
            # the recipe rather than of an arm, because whether an op reduces is an
            # op fact and needs no format to answer.
            if recipe.attrs is not None or recipe.reduces:
                continue
            with self.subTest(op=op):
                spec = make_op_spec(op, inputs=recipe.arity)
                [step] = ktir.build_kernel_plan([spec]).steps
                self.assertEqual(step.attrs, ())

    def test_a_missing_scalar_argument_is_the_plans_problem(self):
        """An ``op_info`` without the constants fails in the plan, not in emission.

        This is what reading the scalars at plan time buys: the failure arrives
        before ``KtirBuilder.create``, so there is no half-built module in hand.
        """
        with self.assertRaises(KeyError):
            ktir.build_kernel_plan([make_op_spec("softplus", inputs=1)])


class TestStepFieldsAgreeWithTheSurface(unittest.TestCase):
    """The price of two optional fields with one reader each, charged in one test.

    ``indexing`` is carried by the surface that reads it and by no other, and a
    nest with a reduced dim is never a bare named op.  Both are invariants of the
    plan rather than of any one fixture, so they are asserted over every accepted
    fixture in this file at once -- which is what stops the minimal record's
    optional fields drifting into a bug nobody's own test covers.
    """

    @staticmethod
    def _accepted_fixtures() -> dict:
        """Every spec list in this file that ``build_kernel_plan`` accepts."""
        n_stick, m = sympy.symbols("n_stick m")
        nest, _spec, _loops = make_nested_op_spec(
            levels=[(n_stick, 2), (m, 256)],
            size=[1, 1, 64],
            advances=[16384 * n_stick + 64 * m] * 3,
        )
        rows = sympy.Symbol("c1")
        lanes = sympy.Symbol("c0")
        stick, lane = sympy.floor(lanes / 64), sympy.Mod(lanes, 64)
        return {
            "pointwise": [make_op_spec()],
            "divided": [make_op_spec(divisions={"d1": 32})],
            "chained": make_chained_op_specs(("add", "mul")),
            "nested": [nest],
            "unit_axis_pointwise": [
                make_op_spec(
                    size=[1, 256, 64],
                    coords=[sympy.Integer(0), rows, sympy.Mod(rows, 64)],
                )
            ],
            "nonstick_reduction": [
                make_op_spec(
                    "sum",
                    is_reduction=True,
                    inputs=1,
                    sizes=[[32, 256, 64], [1, 32, 64]],
                    coords_per_arg=[
                        [stick, rows, lane],
                        [sympy.Integer(0), stick, lane],
                    ],
                    space={lanes: (2048, 32), rows: (256, 1)},
                )
            ],
            "onstick_reduction": make_onstick_sum_specs(),
            # A pointwise op whose payload is a ``spyreop`` intrinsic: the other
            # way onto ``Surface.GENERIC``, and the one that reaches it with no
            # reduced dim, which is the combination the two claims below split on.
            "intrinsic": [make_op_spec("sqrt", inputs=1)],
            "intrinsic_with_attrs": [
                make_op_spec(
                    "softplus",
                    inputs=1,
                    op_info={
                        "constants": {"softplusBeta": 1.0, "softplusThresh": 20.0}
                    },
                )
            ],
        }

    @staticmethod
    def _steps(steps):
        for step in steps:
            if isinstance(step, ktir.LoopStep):
                yield from TestStepFieldsAgreeWithTheSurface._steps(step.body)
            else:
                yield step

    def test_the_fixtures_cover_every_surface(self):
        """A vacuous invariant is the failure mode, so the coverage is asserted."""
        surfaces = {
            step.surface
            for specs in self._accepted_fixtures().values()
            for step in self._steps(ktir.build_kernel_plan(specs).steps)
        }
        self.assertEqual(surfaces, set(ktir.Surface))

    def test_a_generic_is_the_only_step_that_states_its_indexing(self):
        for name, specs in self._accepted_fixtures().items():
            for position, step in enumerate(
                self._steps(ktir.build_kernel_plan(specs).steps)
            ):
                with self.subTest(fixture=name, step=position):
                    self.assertIs(
                        step.indexing is not None, step.surface is ktir.Surface.GENERIC
                    )
                    if step.reduce_dims:
                        self.assertIsNot(step.surface, ktir.Surface.BARE)


def _tiled_reduction_specs() -> tuple:
    """The loop-nest shape of a hand-written 1-core KTIR ``sum`` kernel.

    Two ``scf.for`` levels over a [2, 256, 64] fp16 input reduced to a [2, 64]
    output: the outer level walks whole sticks (2 trips), the inner level walks
    rows within a stick (256 trips).  The input's tile is one row, the output's
    one stick, and each arg's ``device_tile_advance_expr`` is the linearized
    element offset for one step of each level:

        a: 16384*n_stick + 64*m     c: 64*n_stick

    Returns ``(spec, loops)``: the op, and the ``LoopSpec`` chain the plan walk
    would reach it with, read off one real nest so the trip counts the derivations
    see are the nest's own.
    """
    n_stick, m = sympy.symbols("n_stick m")
    _nest, spec, loops = make_nested_op_spec(
        levels=[(n_stick, 2), (m, 256)],  # outermost-first
        inputs=1,
        names=["a", "c"],
        sizes=[[1, 1, 64], [1, 64]],
        advances=[16384 * n_stick + 64 * m, 64 * n_stick],
        allocations=[{"hbm": 0}, {"hbm": 0}],
        dtype=DataFormats.IEEE_FP16,
    )
    return spec, loops


class TestLoopDerivations(unittest.TestCase):
    """``_levels`` / ``_solve_layout`` / ``_access`` against that ``sum`` nest.

    The numbers are pinned against a KTIR kernel a scheduler already consumes, so
    what the loop form should be is not this emitter's invention.
    """

    def test_levels_are_outermost_first_with_their_trip_counts(self):
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        self.assertEqual([lvl.trip for lvl in levels], [2, 256])
        # tiled_symbols is innermost-first; the levels come back outermost-first.
        self.assertEqual(
            [str(s) for lvl in levels for s in lvl.symbols], ["n_stick", "m"]
        )

    def test_levels_must_match_the_enclosing_nest(self):
        spec, loops = _tiled_reduction_specs()
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._levels(spec, loops[:1])
        self.assertIn("tiled_symbols", str(ctx.exception))

    def test_a_symbolic_trip_count_is_read_not_refused(self):
        """``_trip`` reads the count; whether one can be emitted is the plan's
        call, so this only checks the reading."""
        s0 = sympy.Symbol("s0")
        self.assertEqual(ktir._trip(LoopSpec(count=4, body=[])), 4)
        self.assertEqual(ktir._trip(LoopSpec(count=s0, body=[])), s0)

    def test_buffer_extent_grows_out_of_the_tile_extent(self):
        """``E_i = A_i + q[l][i] * (T_l - 1)``, matching that kernel's views."""
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        a, c = spec.args

        a_layout, a_q = ktir._solve_layout(a, levels)
        # 2 = 1 + 1*(2-1), 256 = 1 + 1*(256-1), and the stick dim is unchanged.
        self.assertEqual(a_layout.extent, (2, 256, 64))
        self.assertEqual(a_layout.strides, (16384, 64, 1))
        # One dim per level: the outer level walks dim 0, the inner walks dim 1.
        self.assertEqual(a_q, [(1, 0, 0), (0, 1, 0)])

        c_layout, c_q = ktir._solve_layout(c, levels)
        self.assertEqual(c_layout.extent, (2, 64))
        self.assertEqual(c_layout.strides, (64, 1))
        # The inner level does not move the output: it is the reduced dim.
        self.assertEqual(c_q, [(1, 0), (0, 0)])

    def test_access_indices_are_the_kernel_subscripts(self):
        """``%a_view[%n_stick, %m, %c0]`` and ``%c_view[%n_stick, %c0]``."""
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        a, c = spec.args

        a_layout, a_q = ktir._solve_layout(a, levels)
        # ``elems`` is the CALLER's answer: which element type an access reads a
        # buffer at is the op's business (``Recipe.unfused``), not the arg's.
        a_access = ktir._access(
            a, a.device_size, a_q, a_layout, ktir.ElemTypes.of(a.device_dtype)
        )
        # The tile extent is device_size, which is what tiling already baked in.
        self.assertEqual(a_access.extent, (1, 1, 64))
        # Per view dim, the step each level takes: dim 0 <- n_stick, dim 1 <- m,
        # dim 2 <- nothing, i.e. the constant zero the kernel spells as %c0.
        self.assertEqual(a_access.index_coeffs, ((1, 0), (0, 1), (0, 0)))

        c_layout, c_q = ktir._solve_layout(c, levels)
        c_access = ktir._access(
            c, c.device_size, c_q, c_layout, ktir.ElemTypes.of(c.device_dtype)
        )
        self.assertEqual(c_access.extent, (1, 64))
        self.assertEqual(c_access.index_coeffs, ((1, 0), (0, 0)))

    def test_untiled_access_sits_at_the_view_origin(self):
        """Depth zero is the general answer, not a special case."""
        arg = make_op_spec().args[0]
        layout, q = ktir._solve_layout(arg, [])
        self.assertEqual(layout.extent, (16, 512, 64))
        self.assertEqual(q, [])
        access = ktir._access(
            arg, arg.device_size, q, layout, ktir.ElemTypes.of(arg.device_dtype)
        )
        # One empty sum per dim: every index expression is zero.
        self.assertEqual(access.index_coeffs, ((), (), ()))

    def test_advance_no_dim_divides_is_reported(self):
        spec, loops = _tiled_reduction_specs()
        levels = ktir._levels(spec, loops)
        a = spec.args[0]
        # 100 elements is not a whole number of steps along any dim of a view
        # whose strides are 16384, 64 and 1 (the stick dim is never stepped).
        a.device_tile_advance_expr = 100 * sympy.Symbol("n_stick")
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._solve_layout(a, levels)
        self.assertIn("not a whole number of steps", str(ctx.exception))


class TestAThreadedValueMayNotCrossAStage(unittest.TestCase):
    """An owned intermediate read by a later stage is refused, and says what to set."""

    def test_an_owned_intermediate_read_by_a_later_stage_is_refused(self):
        with self.assertRaises(NotImplementedError) as caught:
            ktir.build_kernel_plan(make_chained_op_specs(("add", "mul"), owned=True))
        message = str(caught.exception)
        self.assertIn("written in stage 0 and read in stage 1", message)
        self.assertIn("cannot cross a compute stage", message)

    def test_the_refusal_names_both_planning_flags(self):
        """The actionable half: a reader must not have to guess the variable."""
        with self.assertRaises(NotImplementedError) as caught:
            ktir.build_kernel_plan(make_chained_op_specs(("add", "mul"), owned=True))
        message = str(caught.exception)
        self.assertIn("LX_PLANNING=0", message)
        self.assertIn("HBM_POOL_PLANNING=0", message)

    def test_the_same_chain_is_accepted_when_the_intermediate_is_passed(self):
        """The control, so the refusal is shown to be about the ALLOCATION only."""
        steps = ktir.build_kernel_plan(make_chained_op_specs(("add", "mul"))).steps
        self.assertEqual([step.stage for step in steps], [0, 1])


class TestStagesAreNumbered(unittest.TestCase):
    """``_stages`` hands each ``ComputeStep`` its own stage, over the whole tree."""

    @staticmethod
    def _two_stages_in_one_body() -> LoopSpec:
        """``(a + b) * c`` at one row per iteration of a two-level nest."""
        n_stick, m = sympy.symbols("n_stick m")
        tiled = {
            "coords": [],
            "space": {},
            "tiled": [[m], [n_stick]],  # innermost-first
            "trips": {n_stick: 2, m: 256},
            "size": [1, 1, 64],
            "advances": [16384 * n_stick + 64 * m] * 3,
        }
        return LoopSpec(
            count=2,
            body=[
                LoopSpec(
                    count=256,
                    body=[
                        make_op_spec(
                            "add",
                            names=["arg0", "arg1", "buf0"],
                            allocations=[None, None, None],
                            **tiled,
                        ),
                        # ``buf0`` is passed, not owned: it crosses a stage, and a
                        # value may not (``TestAThreadedValueMayNotCrossAStage``).
                        # It keeps stage 0's index, so the second spec starts at it.
                        make_op_spec(
                            "mul",
                            names=["buf0", "arg2", "buf1"],
                            allocations=[None, None, None],
                            first_arg_index=2,
                            **tiled,
                        ),
                    ],
                )
            ],
        )

    def test_each_op_in_a_chain_is_its_own_stage(self):
        steps = ktir.build_kernel_plan(make_chained_op_specs(("add", "mul"))).steps
        self.assertEqual([step.stage for step in steps], [0, 1])

    def test_a_loop_body_continues_the_kernels_count(self):
        """The counter is the plan's, so recursion cannot restart it at zero."""
        plan = ktir.build_kernel_plan([self._two_stages_in_one_body()])
        [outer] = plan.steps
        [inner] = outer.body
        self.assertEqual([step.stage for step in inner.body], [0, 1])


_TABLE_DEFAULT = object()


def fuse(specs, table=_TABLE_DEFAULT) -> tuple:
    """``apply_plan_fusions``, with the shipped table left as the default."""
    if table is _TABLE_DEFAULT:
        return ktir.apply_plan_fusions(specs)
    return ktir.apply_plan_fusions(specs, table)


class FusionCase(unittest.TestCase):
    """Base for the plan-fusion tests: one helper for the decline they share."""

    def assertDeclined(self, specs, table=_TABLE_DEFAULT, reason=None):
        """``specs`` came back unfused, in order, with nothing raised."""
        if reason is None:
            vector = fuse(specs, table)
        else:
            with self.assertLogs(ktir.logger, level="DEBUG") as captured:
                vector = fuse(specs, table)
            declines = [
                record.getMessage()
                for record in captured.records
                if "declines" in record.getMessage()
            ]
            self.assertTrue(any(reason in message for message in declines), declines)
        self.assertEqual([spec.op for spec in vector], [spec.op for spec in specs])
        return vector


class TestPlanFusionRewrite(FusionCase):
    """Recognition and rewrite: what replaces a span the table names."""

    def test_a_two_stage_span_the_table_names_becomes_one_stage(self):
        """DECISION: recognise the span positionally and replace the whole of it.

        With stages either side, because a match rewrites its own span and
        nothing around it.
        """
        before, after = make_op_spec("add"), make_op_spec("mul")
        vector = fuse([before, *make_absmax_pair(), after])
        self.assertEqual([spec.op for spec in vector], ["add", "absmax", "mul"])
        self.assertIs(vector[0], before)
        self.assertIs(vector[2], after)
        self.assertTrue(vector[1].is_reduction)
        self.assertIn("absmax", ktir.KtirBuilder.RECIPES)

    def test_a_span_the_pattern_does_not_name_is_left_alone(self):
        """DECISION: a slot is ``(op name, is_reduction)``, matched adjacently."""
        producer, consumer = make_absmax_pair()
        cases = {
            "another op": make_linked_op_specs(ops=("exp", "max")),
            "not a reduction": make_linked_op_specs(reductions=(False, False)),
            "a stage in between": [producer, make_op_spec("add"), consumer],
        }
        for label, specs in cases.items():
            with self.subTest(case=label):
                self.assertDeclined(specs)

    def test_a_kernel_local_link_in_plain_hbm_is_fused(self):
        """DECISION: locality, not a planner's placement, licenses the deletion.

        The link is an ordinary HBM buffer the wrapper allocates and passes --
        what the planners being off produces -- and it fuses because nothing
        outside the kernel reads it.  This case used to be refused.
        """
        pair = make_absmax_pair(link={"hbm": None})
        link = pair[0].args[-1]
        self.assertFalse(ktir.is_internal(link))
        self.assertTrue(link.kernel_local)
        self.assertEqual([spec.op for spec in fuse(pair)], ["absmax"])

    def test_the_fused_spec_keeps_the_survivors_access_and_the_sources_identity(self):
        """DECISION: splice buffer IDENTITY across, never access geometry."""
        pair = make_absmax_pair(in_sizes={1: [32, 256, 64]})
        producer_in, producer_out = pair[0].args
        survivor_read, survivor_out = pair[1].args

        [fused] = fuse(pair)
        read, out = fused.args

        # Identity: the producer's own source, so the link is gone entirely.
        self.assertEqual(read.name, producer_in.name)
        self.assertEqual(read.arg_index, producer_in.arg_index)
        self.assertEqual(read.allocation, producer_in.allocation)
        self.assertEqual(read.device_dtype, producer_in.device_dtype)
        self.assertNotEqual(read.name, producer_out.name)

        # Access: the survivor's own, in the survivor's namespace and not the
        # producer's.
        self.assertEqual(read.device_size, survivor_read.device_size)
        self.assertNotEqual(read.device_size, producer_in.device_size)
        self.assertEqual(read.device_coordinates, survivor_read.device_coordinates)
        self.assertEqual(out.device_coordinates, survivor_out.device_coordinates)
        survivor_prefix = str(next(iter(pair[1].iteration_space)))[0]
        symbols = {
            str(symbol)
            for coordinate in read.device_coordinates
            for symbol in coordinate.free_symbols
        }
        self.assertTrue(symbols)
        self.assertTrue(all(s.startswith(survivor_prefix) for s in symbols), symbols)
        # And the survivor's iteration space, which is what ``_divisions`` reads.
        self.assertEqual(fused.iteration_space, pair[1].iteration_space)


class TestPlanFusionDeclines(FusionCase):
    """Every condition the rewrite checks, and the vector it hands back."""

    def test_a_producer_that_is_not_unary_is_not_deleted(self):
        """DECISION: with two sources there is no single one to read instead."""
        producer, survivor = make_absmax_pair()
        source, link = producer.args
        second = dataclasses.replace(source, name="x_other", arg_index=2)
        producer = dataclasses.replace(producer, args=[source, second, link])
        self.assertDeclined([producer, survivor], reason="is not unary")

    @staticmethod
    def _converting_producer() -> list:
        """A pair whose producer writes its link at a different format."""
        producer, survivor = make_absmax_pair()
        source, link = producer.args
        link = dataclasses.replace(link, device_dtype=DataFormats.IEEE_FP32)
        return [dataclasses.replace(producer, args=[source, link]), survivor]

    def test_a_producer_that_does_not_preserve_access_is_not_deleted(self):
        """DECISION: the deleted producer must write where it read."""
        d0, d1, d2 = sympy.symbols("d0 d1 d2")
        cases = {
            # An extent check sees this one...
            "resizes": make_absmax_pair(out_sizes={0: [32, 256, 64]}),
            # ...and only a coordinate check sees this one.
            "moves elements": make_absmax_pair(out_coords={0: [d1, d0, d2]}),
            # The rewrite hands the survivor the source's format, so a producer
            # that converts is not a drop-in either.
            "reformats": self._converting_producer(),
        }
        for label, pair in cases.items():
            with self.subTest(case=label):
                self.assertDeclined(pair, reason="not access-preserving")

    def test_a_link_something_outside_the_kernel_reads_is_not_deleted(self):
        """DECISION: only a kernel-local buffer may be deleted."""
        self.assertDeclined(
            make_absmax_pair(link_local=False), reason="is not kernel-local"
        )

    def test_a_link_read_more_than_once_is_not_deleted(self):
        """DECISION: the link must be read exactly once, by the survivor."""
        vector = self.assertDeclined(
            make_linked_op_specs(
                ops=("abs", "max", "sum", "add"),
                reductions=(False, True, True, False),
                edges=((0, 1), (0, 2), (1, 3), (2, 3)),
            ),
            reason="read 2 time(s)",
        )
        self.assertEqual([spec.op for spec in vector], ["abs", "max", "sum", "add"])

    def test_the_viability_predicate_declines(self):
        """DECISION: decline a form the device computes INCORRECTLY.

        An undecidable question declines too, rather than raising.
        """
        fp16 = make_absmax_pair(onstick=True)
        fp32 = make_absmax_pair(onstick=True, dtype=DataFormats.IEEE_FP32, lanes=32)
        for pair in (fp16, fp32):
            self.assertIs(ktir._reduction_surface(pair[1]), ktir.Surface.GENERIC)
        # fp32 on-stick absmax compiles and returns garbage; fp16 is fine.
        self.assertDeclined(fp32, reason="is not viable on this operand")
        self.assertEqual([spec.op for spec in fuse(fp16)], ["absmax"])

        def undecidable(fused):
            raise NotImplementedError("no surface for this shape")

        self.assertDeclined(
            make_linked_op_specs(),
            (make_plan_fusion(viable=undecidable),),
            reason="no surface for this shape",
        )


class TestPlanFusionStructure(FusionCase):
    """Where the fuser runs from, which is not observable anywhere else."""

    def test_a_span_inside_a_loop_body_is_fused(self):
        """DECISION: recurse into loop bodies."""
        nest = LoopSpec(count=4, body=[LoopSpec(count=8, body=make_absmax_pair())])
        [result] = fuse([nest])
        self.assertEqual([spec.op for spec in result.body[0].body], ["absmax"])
        # The rebuilt bodies are lists, which is what ``LoopSpec.body`` declares.
        self.assertIsInstance(result.body, list)
        self.assertIsInstance(result.body[0].body, list)

    def test_a_divided_pair_plans_because_fusion_precedes_the_grid(self):
        """DECISION: fuse on the first line of ``add_specs``, before the grid."""
        pair = make_absmax_pair(row_division=32)
        with self.assertRaises(NotImplementedError) as ctx:
            ktir._divisions(pair)
        self.assertIn("different work divisions", str(ctx.exception))

        plan = ktir.build_kernel_plan(pair)
        self.assertEqual(plan.grid, (32,))
        self.assertEqual(plan.divisions, (ktir.Division(symbol="e1", div=32, inner=1),))
        self.assertEqual([step.op for step in plan.steps], ["absmax"])


class TestGenuineAbsmaxRecipe(unittest.TestCase):
    """``RECIPES['absmax']`` has a caller that is not the fusion table."""

    def test_a_standalone_absmax_reduction_plans_without_any_fusion(self):
        rows, reduced = sympy.symbols("c0 c1")
        stick, lane = sympy.floor(reduced / 64), sympy.Mod(reduced, 64)
        spec = make_op_spec(
            "absmax",
            is_reduction=True,
            inputs=1,
            sizes=[[2, 256, 64], [1, 256, 64]],
            coords_per_arg=[
                [stick, rows, lane],
                [sympy.Integer(0), rows, sympy.Integer(0)],
            ],
            space={rows: (256, 1), reduced: (128, 1)},
        )
        self.assertIn("absmax", ktir.KtirBuilder.RECIPES)
        plan = ktir.build_kernel_plan([spec])
        [step] = plan.steps
        self.assertEqual(step.op, "absmax")
        # A reduction's recipe must accumulate, and on-stick is the generic form.
        self.assertIs(
            ktir.KtirBuilder.RECIPES["absmax"].arm(FP16).kind,
            ktir.BindingKind.COMBINER,
        )
        self.assertIs(step.surface, ktir.Surface.GENERIC)


# ---------------------------------------------------------------------------
# What we generate
# ---------------------------------------------------------------------------


class TestRefusals(unittest.TestCase):
    """The labelled capabilities this emitter does not implement.

    A label is a token shared by the raise and this test, so grepping it finds
    both.  No message here claims a consumer is the blocker: this repository
    cannot run the backend compiler or the scheduler, so what they accept is not observable
    from these tests, and two labels that used to claim it were both wrong.
    """

    def test_staggered_arrangement_is_unimplemented(self):
        """FAILS ONCE THE STAGGERED LAYOUT IS IMPLEMENTED, deliberately.

        The permutation has never been written down as numbers, so unlike every
        other refusal there is no derivation behind this one.  The test fails the
        moment ``_arrangement_layout`` returns numbers instead of raising, which
        is the prompt to delete it along with the label.
        """
        arrangement = next(iter(STAGGERED_EAS))
        with self.assertRaises(ktir.Unimplemented) as ctx:
            ktir._arrangement_layout(arrangement, (16, 512, 64), (32768, 64, 1))
        self.assertIn("staggered-element-arrangement", str(ctx.exception))

    def test_standard_arrangement_is_plain_row_major(self):
        extent, strides = (16, 512, 64), (32768, 64, 1)
        for arrangement in (
            None,
            ElementArrangement.STANDARD,
            ElementArrangement.QFP8CH,
        ):
            with self.subTest(arrangement=arrangement):
                self.assertEqual(
                    ktir._arrangement_layout(arrangement, extent, strides),
                    (extent, strides),
                )

    def test_a_fused_arrangement_is_a_type_and_not_a_stride(self):
        """``EXX2`` selects an element TYPE, nothing else."""
        extent, strides = (256, 64), (64, 1)
        self.assertEqual(
            ktir._arrangement_layout(ElementArrangement.EXX2, extent, strides),
            (extent, strides),
        )

    def test_every_label_is_greppable_and_uniquely_owned(self):
        """Each label is raised from exactly one site, so grepping it is exact."""
        source = inspect.getsource(ktir)
        labels = re.findall(r'_unimplemented\(\s*\n?\s*"([^"]+)"', source)
        self.assertEqual(sorted(labels), sorted(set(labels)))
        self.assertEqual(sorted(labels), ["staggered-element-arrangement"])

    def test_no_refusal_message_blames_a_consumer(self):
        """A refusal says what is missing here, not what someone else rejects.

        Checked over the ``_unimplemented`` messages rather than the whole file:
        naming the backend is legitimate where it explains why an *option* exists
        (baking addresses), but not as the reason a capability is refused, which
        this repository cannot observe.
        """
        tree = ast.parse(inspect.getsource(ktir))
        messages = [
            " ".join(
                part.value
                for part in ast.walk(node.args[1])
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_unimplemented"
        ]
        self.assertEqual(len(messages), 1)
        for message in messages:
            with self.subTest(message=message[:40]):
                for blame in ("dbo-opt", "no consumer", "nothing lowers", "scheduler"):
                    self.assertNotIn(blame, message)


class TestBroadcastOperands(unittest.TestCase):
    """An operand that does not walk every axis of the output."""

    def test_each_form_derives_the_maps_the_chain_needs(self):
        """The three rows, against the hand-written chain's three maps."""
        for form, maps in (
            ("row", ((0, 1, 2), (0, None, 2), (0, 1, 2))),
            ("stat", ((0, 1, 2), (1, None), (0, 1, 2))),
            ("splat", ((0, None), (0, 1))),
        ):
            with self.subTest(form=form):
                plan = ktir.build_kernel_plan([make_broadcast_op_spec(form)])
                [step] = plan.steps
                self.assertIs(step.surface, ktir.Surface.GENERIC)
                self.assertEqual(step.indexing.maps, maps)
                # Pointwise: every iteration dim is the output's, all parallel.
                self.assertEqual(
                    step.indexing.iters, ("parallel",) * len(step.out.extent)
                )

    def test_an_aligned_operand_still_reaches_the_named_form(self):
        """The fast path, asserted where the derivation would also have applied:"""
        plan = ktir.build_kernel_plan([make_op_spec()])
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.BARE)
        self.assertIsNone(step.indexing)

    def test_a_broadcast_operand_beside_an_aligned_one_states_both_rows(self):
        """``indexing_maps`` is one attribute, so the aligned operand's identity"""
        plan = ktir.build_kernel_plan([make_broadcast_op_spec("row")])
        [step] = plan.steps
        self.assertEqual(step.indexing.maps[0], (0, 1, 2))


class TestOneBufferViewedAtTwoElementTypes(unittest.TestCase):
    """Two stages, one buffer, two element types -- and one address."""

    def test_each_access_carries_its_own_element_type_on_one_buf_id(self):
        plan = ktir.build_kernel_plan(make_two_element_type_specs())
        produce, consume = plan.steps
        [(link, statistic)] = [pair for pair in consume.ins if pair[0] == "pair"]
        self.assertEqual((produce.out_buf_id, link), ("pair", "pair"))
        self.assertEqual(produce.out.elems.storage, "!spyreop.fp16_fused")
        self.assertEqual(statistic.elems.storage, "f16")
        # One buffer, so one geometry: the views differ in element type only.
        self.assertEqual(produce.out.buffer.layout, statistic.buffer.layout)
        self.assertEqual(
            (produce.out.buffer.arg_index, statistic.buffer.arg_index), (1, 1)
        )

    def test_the_signature_still_lists_the_buffer_once(self):
        """The dedup by ``buf_id`` is what keeps the signature right while the"""
        plan = ktir.build_kernel_plan(make_two_element_type_specs())
        self.assertEqual([b.buf_id for b in plan.parameters].count("pair"), 1)
        self.assertEqual(
            [b.buf_id for b in plan.parameters],
            ["x0", "pair", "x1", "s2", "s3", "s4", "out0"],
        )

    def test_two_element_types_for_one_buffer_in_one_stage_are_refused(self):
        """Refused, not keyed more finely: the views are per ``(stage, buf_id)``,"""
        specs = make_two_element_type_specs()
        consume = specs[1]
        # Operand 3, which the recipe does not name and which therefore reads its
        # buffer at the buffer's fused arrangement, now names the very buffer
        # operand 1 reads unfused.
        consume.args[3] = dataclasses.replace(
            consume.args[1], element_arrangement=ElementArrangement.EXX2
        )
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.build_kernel_plan(specs)
        self.assertIn("two element types in one stage", str(ctx.exception))


class TestArityBeyondTwoAndPerOperandElementTypes(unittest.TestCase):
    """An op with five operands, and one of them read at another element type."""

    @staticmethod
    def _five_inputs(arrangements=None):
        return make_op_spec(
            "layernormnorm",
            inputs=5,
            size=[12, 64, 64],
            dtype=DataFormats.IEEE_FP32,
            arrangements=arrangements,
        )

    def test_five_operands_plan_with_one_map_each_and_one_for_the_result(self):
        plan = ktir.build_kernel_plan([self._five_inputs()])
        [step] = plan.steps
        self.assertEqual(len(step.ins), 5)
        self.assertEqual(len(plan.parameters), 6)
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(step.indexing.maps, ((0, 1, 2),) * 6)

    def test_the_recipe_and_not_the_arrangement_types_an_operand(self):
        """Three operands whose buffers all hold fused statistics; the recipe says"""
        recipe = ktir.KtirBuilder.RECIPES["layernormnorm"]
        self.assertEqual(recipe.unfused, (1, 2))  # squares and scale, as f16
        fused = ElementArrangement.EXX2
        plan = ktir.build_kernel_plan(
            [self._five_inputs([None, fused, fused, fused, None, None])]
        )
        [step] = plan.steps
        self.assertEqual(step.ins[1][1].elems.storage, "f32")
        self.assertEqual(step.ins[2][1].elems.storage, "f32")
        self.assertEqual(step.ins[3][1].elems.storage, "!spyreop.fp32_fused")

    def test_the_result_can_be_the_unfused_position(self):
        """``layernormscale_fused`` returns a plain float out of a buffer the"""
        self.assertEqual(ktir.KtirBuilder.RECIPES["layernormscale"].unfused, (1,))
        plan = ktir.build_kernel_plan(
            [
                make_op_spec(
                    "layernormscale",
                    inputs=1,
                    arrangements=[ElementArrangement.EXX2, ElementArrangement.EXX2],
                )
            ]
        )
        [step] = plan.steps
        self.assertEqual(step.ins[0][1].elems.storage, "!spyreop.fp16_fused")
        self.assertEqual(step.out.elems.storage, "f16")

    def test_an_unfused_position_the_op_does_not_have_is_a_typo(self):
        """A recipe is source, so a position past the result fails where it is"""
        with self.assertRaises(ValueError) as ctx:
            ktir.Recipe(
                arity=1,
                arms=ktir.Arm(
                    kind=ktir.BindingKind.PAYLOAD, binding=lambda: None, dtypes=()
                ),
                unfused=(2,),
            )
        self.assertIn("does not have", str(ctx.exception))


class TestReadingAStatisticAtTheHeadOfItsStick(unittest.TestCase):
    """A pointwise stage reading what a reduction wrote."""

    def test_the_reader_has_the_rank_the_producer_registered(self):
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        produce, consume = plan.steps
        [(_x1, _full), (link, statistic)] = consume.ins
        self.assertEqual(link, "buf0")
        self.assertEqual(len(statistic.extent), len(produce.out.extent))
        self.assertEqual(plan.buffers["buf0"].layout.extent, (256, 64))

    def test_the_tile_is_the_stick_head_and_the_view_is_the_whole_stick(self):
        """The negative test's constraint, stated as the two numbers it is about:"""
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        _produce, consume = plan.steps
        [_full, (_link, statistic)] = consume.ins
        self.assertEqual(statistic.extent, (256, 1))
        self.assertEqual(plan.buffers["buf0"].layout.extent[-1], 64)

    def test_the_producer_still_writes_the_whole_stick(self):
        """The asymmetry: the reduction's own output keeps all 64 lanes, which is"""
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        produce, _consume = plan.steps
        self.assertEqual(produce.out.extent, (256, 64))

    def test_the_squeezed_read_derives_the_statistic_map(self):
        """And the two capabilities meet: a rank-reduced operand at a one-element"""
        plan = ktir.build_kernel_plan(make_statistic_reader_specs())
        _produce, consume = plan.steps
        self.assertEqual(consume.indexing.maps, ((0, 1, 2), (1, None), (0, 1, 2)))


class TestAReducingBodyThatIgnoresItsAccumulator(unittest.TestCase):
    """A reduction registered ``COMBINER`` whose binding ignores ``accumulated``."""

    def test_it_reduces_and_is_registered_as_a_combiner(self):
        """Both statements of the one bit agree, so the equality check passes."""
        recipe = ktir.KtirBuilder.RECIPES["exx2"]
        self.assertEqual(recipe.arity, 1)
        self.assertIs(recipe.arm(FP16).kind, ktir.BindingKind.COMBINER)

    def test_the_plan_is_the_ordinary_on_stick_reduction(self):
        specs = make_onstick_sum_specs(
            "exx2", arrangements=[ElementArrangement.STANDARD, ElementArrangement.EXX2]
        )
        plan = ktir.build_kernel_plan(specs)
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(
            step.indexing.iters, ("reduction", "parallel", "reduction", "parallel")
        )
        self.assertEqual(step.indexing.maps, ((0, 1, 2), (1, 3)))
        # The accumulator's type is the output buffer's, and that is the pair:
        # ``exx2_fused`` returns it, so nothing else has to be told.
        self.assertEqual(step.out.elems.value, "!spyreop.fp16_fused")
        self.assertEqual(step.out.extent, (256, 64))


class TestFusedElementType(unittest.TestCase):
    """One buffer's ``element_arrangement`` decides its element TYPE."""

    def test_exx2_is_the_fused_spelling_of_its_dtype(self):
        for dtype, spelling in (
            (DataFormats.SEN169_FP16, "!spyreop.fp16_fused"),
            (DataFormats.IEEE_FP16, "!spyreop.fp16_fused"),
            (DataFormats.IEEE_FP32, "!spyreop.fp32_fused"),
        ):
            with self.subTest(dtype=dtype):
                fused = ktir.ElemTypes.of(dtype, ElementArrangement.EXX2)
                self.assertEqual((fused.storage, fused.value), (spelling, spelling))
                # The same dtype in the standard order is the plain float: the
                # arrangement is what selects the table.
                self.assertNotEqual(ktir.ElemTypes.of(dtype).storage, spelling)

    def test_a_dtype_with_no_fused_spelling_is_refused(self):
        """An integer pair has no spelling in the dialect, so it is not guessed:"""
        with self.assertRaises(NotImplementedError) as ctx:
            ktir.ElemTypes.of(DataFormats.IEEE_INT32, ElementArrangement.EXX2)
        self.assertIn("EXX2", str(ctx.exception))

    def test_the_buffer_and_the_access_both_take_the_fused_type(self):
        """The view and the tile agree, because one derivation answers both."""
        specs = [
            make_op_spec(
                "layernormscale",
                inputs=1,
                arrangements=[ElementArrangement.EXX2, ElementArrangement.STANDARD],
            )
        ]
        plan = ktir.build_kernel_plan(specs)
        [step] = plan.steps
        [(_buf_id, source)] = step.ins
        self.assertEqual(source.elems.storage, "!spyreop.fp16_fused")
        self.assertEqual(plan.buffers["arg0"].elems.storage, "!spyreop.fp16_fused")
        # The pair is one element, so the extent is the arg's own device_size.
        self.assertEqual(plan.buffers["arg0"].layout.extent, tuple(ADD_SIZE))
        # And the OUTPUT of this op is not fused: nothing propagates the flag.
        self.assertEqual(step.out.elems.storage, "f16")

    def test_layernormscale_binds_the_fused_form_at_arity_one(self):
        """The frontend hands this op the pair as ONE operand."""
        self.assertEqual(ktir.KtirBuilder.RECIPES["layernormscale"].arity, 1)


class TestWithoutTheDialectBuild(unittest.TestCase):
    """``ktir`` imports, and rejects, with ``mlir_ktdp`` made unimportable.

    The rest of this module relies on the dialect never being needed; here it is
    actively blocked, so the reliance is checked rather than assumed.
    """

    class _Blocker:
        """A ``sys.meta_path`` finder that refuses ``mlir_ktdp``."""

        def find_spec(self, name, path=None, target=None):
            if name == "mlir_ktdp" or name.startswith("mlir_ktdp."):
                raise ImportError(f"blocked: {name}")
            # None: every other name falls through to the real finders.

    @contextlib.contextmanager
    def _blocked(self):
        """A freshly imported ``ktir`` that cannot reach the dialect.

        A fresh module because ``_load_dialects`` caches its handles: an already
        loaded ``ktir`` in this process may have bound them before the block.
        """
        name = ktir.__name__
        blocker = self._Blocker()
        saved = {
            key: module
            for key, module in sys.modules.items()
            if key == "mlir_ktdp" or key.startswith("mlir_ktdp.")
        }
        saved[name] = sys.modules.pop(name)
        for key in saved:
            sys.modules.pop(key, None)
        sys.meta_path.insert(0, blocker)
        try:
            yield importlib.import_module(name)
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.pop(name, None)
            sys.modules.update(saved)

    def test_imports_and_rejects_without_the_dialect(self):
        with self._blocked() as fresh:
            self.assertFalse(fresh.dialect_available())
            # A rejection, not an ImportError: the plan walk runs first and needs
            # nothing from the dialect.
            with self.assertRaises(NotImplementedError) as ctx:
                fresh.generate_ktir("k", [UnimplementedOp(op="atan2")])
            self.assertIn("unimplemented op", str(ctx.exception))
            # And the derivations answer, dialect or no dialect.
            layout, _ = fresh._solve_layout(make_op_spec().args[0], [])
            self.assertEqual(layout.extent, (16, 512, 64))

    def test_emission_is_what_needs_the_dialect(self):
        # A *valid* request gets as far as the builder and no further.
        with self._blocked() as fresh, self.assertRaises(ImportError):
            fresh.generate_ktir("k", [make_op_spec()])


class TestScopeStack(unittest.TestCase):
    """The lexical scope the walk carries: induction variables and live values."""

    def test_produced_values_are_scoped(self):
        env = ktir.ScopeStack()
        self.assertIsNone(env.produced("buf0"))
        with env.scope():
            env.bind_produced("buf0", "v0")
            self.assertEqual(env.produced("buf0"), "v0")
        self.assertIsNone(env.produced("buf0"))

    def test_inner_scope_shadows_outer(self):
        env = ktir.ScopeStack()
        env.bind_produced("buf0", "outer")
        with env.scope():
            env.bind_produced("buf0", "inner")
            self.assertEqual(env.produced("buf0"), "inner")
        self.assertEqual(env.produced("buf0"), "outer")

    def test_value_only_scopes_are_not_loops(self):
        """A frame with no induction variable adds no level to zip against."""
        env = ktir.ScopeStack()
        with env.scope(iv="i"):
            with env.scope():  # a plain value scope
                self.assertEqual(env.ivs(), ["i"])
            with env.scope(iv="j"):
                self.assertEqual(env.ivs(), ["i", "j"])

    def test_ivs_are_innermost_last(self):
        env = ktir.ScopeStack()
        self.assertEqual(env.ivs(), [])
        with env.scope(iv="i"):
            with env.scope(iv="j"):
                self.assertEqual(env.ivs(), ["i", "j"])
            self.assertEqual(env.ivs(), ["i"])
        self.assertEqual(env.ivs(), [])


class TestEmissionCannotRefuse(unittest.TestCase):
    """Nothing reachable from ``KtirBuilder.emit`` can raise a rejection.

    This is the property the step tree buys: the plan runs every derivation and
    every guard, and emission consumes only the plan's records, so a request the
    plan accepted cannot be refused half-emitted.  Asserted over the call graph
    rather than trusted, because the failure mode is silent -- a derivation called
    from the emission path would keep working right up to the input that trips its
    guard, with a half-built module already in hand.
    """

    def test_only_plan_bug_assertions_are_reachable_from_emit(self):
        tree = ast.parse(inspect.getsource(ktir))
        builder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KtirBuilder"
        )
        methods = {
            node.name: node
            for node in builder.body
            if isinstance(node, ast.FunctionDef)
        }
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        seen: set[str] = set()
        pending = ["emit"]
        raised: list[tuple[str, str]] = []
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            node = methods.get(name) or functions.get(name)
            if node is None:  # a dialect builder, not ours
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise):
                    called = getattr(sub.exc, "func", sub.exc)
                    raised.append(
                        (
                            name,
                            getattr(called, "id", None) or getattr(called, "attr", ""),
                        )
                    )
                if isinstance(sub, ast.Call):
                    callee = getattr(sub.func, "id", None) or getattr(
                        sub.func, "attr", None
                    )
                    if callee in methods or callee in functions:
                        pending.append(callee)

        # Every explicit raise on the emission path is a malformed-plan assertion:
        # no NotImplementedError and no Unimplemented, which is what a refusal is
        # spelled as.  So none of the functions that *can* refuse -- the labelled
        # guard `_unimplemented`, and the derivations `_levels`, `_solve_layout`
        # and `_access` that call it or raise directly -- is reachable from here.
        #
        # Stated over the raise kinds rather than over those names: a name would
        # have to be kept in step with the source, and two of them once were not,
        # so they silently asserted nothing for as long as that lasted.
        self.assertTrue(raised, "expected the plan-bug assertions to be found")
        self.assertEqual({kind for _, kind in raised}, {"AssertionError"})


class TestNoModuleLevelDialectImport(unittest.TestCase):
    """The property this whole file depends on, asserted rather than assumed.

    ``ktir`` must not import ``mlir_ktdp`` at module level, or the plan walk --
    and every test above -- becomes unrunnable without the dialect build.
    """

    def test_ktir_has_no_top_level_mlir_ktdp_import(self):
        tree = ast.parse(inspect.getsource(ktir))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                self.assertFalse(
                    name.split(".")[0] == "mlir_ktdp",
                    f"ktir.py imports {name} at module level",
                )


if __name__ == "__main__":
    unittest.main()
