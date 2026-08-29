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
    allocations: list | None = None,
    baked: bool = False,
    advances: list | None = None,
    is_reduction: bool = False,
    divisions: dict | None = None,
    space: dict | None = None,
    tiled: list | None = None,
    trips: dict | None = None,
    first_arg_index: int = 0,
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
    * ``allocations`` per arg, for an ``lx`` / ``hbm_pool`` intermediate or an
      unrecognised space; ``baked=True`` for the byte HBM address the baked form
      wants, which is the same field said the other way, so not both.
    * ``divisions`` maps a coordinate symbol's name to its work division;
      ``space`` replaces the iteration space outright (``{}`` for a tiled op).
    * ``tiled`` / ``trips`` are the loop-level symbols and trip counts, and
      ``first_arg_index`` continues the numbering for a second op in one kernel.
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
        op_info={},
        tiled_symbols=tiled or [],
        tiled_symbol_trip_counts=trips or {},
    )


def make_chained_op_specs(ops: tuple = ("add", "mul"), **overrides) -> list:
    """The ops of one kernel, each threading its result into the next.

    Every op but the last writes an ``lx`` intermediate that the next op reads,
    which is the contract saying this kernel owns it: not passed in, no address,
    and nothing outside the kernel can reach it.  The fresh inputs and the final
    output are HBM args, numbered across the whole kernel rather than per op.
    """
    lx = {"lx": 0}
    specs, next_arg = [], 0
    for level, op in enumerate(ops):
        # The first op reads two fresh inputs; every later one reads the previous
        # result and one fresh input.  Only the last op's output is HBM.
        threaded = [] if level == 0 else [f"buf{level - 1}"]
        fresh = [f"arg{next_arg + i}" for i in range(2 - len(threaded))]
        specs.append(
            make_op_spec(
                op,
                names=[*threaded, *fresh, f"buf{level}"],
                allocations=[
                    *([lx] if threaded else []),
                    *([None] * len(fresh)),
                    None if level == len(ops) - 1 else lx,
                ],
                first_arg_index=next_arg,
                **overrides,
            )
        )
        next_arg += len(fresh)
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


def make_onstick_sum_specs() -> list:
    """``sum(x[256, 128], dim=-1)`` on one core, as the frontend projects it.

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
            "sum",
            is_reduction=True,
            inputs=1,
            sizes=[[2, 256, 64], [1, 256, 64]],
            coords_per_arg=[
                [stick, rows, lane],
                [sympy.Integer(0), rows, sympy.Integer(0)],
            ],
            space={rows: (256, 1), reduced: (128, 1)},
        )
    ]


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

    def test_broadcast_operand_rejected(self):
        # A unit outer-stick extent against the output's 16: a real broadcast.
        specs = [make_op_spec(sizes=[[1, 512, 64]])]
        self._rejects(specs, "broadcast / reshape operands")

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
                self.assertIsInstance(recipe.kind, ktir.BindingKind)
                # A thunk, not the builder itself: resolving it here would need
                # the dialect, which this module deliberately does not require.
                self.assertTrue(callable(recipe.binding))

        # Which surface a step gets is the plan's choice, not a recipe's, so
        # completeness on this side is about ``compute`` rather than about any one
        # op: every ``Surface`` must appear as a ``case`` pattern.  Read off the
        # AST because ``case _:`` alone turns a missing arm into a runtime
        # discovery, at which point a module is already half built.  Deliberately
        # *not* the mirror assertion that every ``BindingKind`` is used by some
        # recipe -- PAYLOAD is the registered-nowhere hook for ``spyreop``.
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
            ktir.Recipe(arity=0, kind=ktir.BindingKind.NAMED, binding=lambda: None)

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

    Nothing in ``RECIPES`` is a ``PAYLOAD`` yet -- registering the intrinsics that
    will be is one line each and no emitter change -- so the arm that serves them
    is exercised here with a recipe registered for the length of the test.  Its
    binding is never called: what is under test is the plan's choice, which is
    made before any dialect is reached.
    """

    @staticmethod
    @contextlib.contextmanager
    def _registered(op, recipe):
        """``recipe`` in ``RECIPES`` under ``op``, for the body of the ``with``."""
        ktir.KtirBuilder.RECIPES[op] = recipe
        try:
            yield
        finally:
            del ktir.KtirBuilder.RECIPES[op]

    def test_no_recipe_registers_a_payload_binding(self):
        """The state this test compensates for, asserted so it stays true.

        The moment an intrinsic is registered, this fails and the synthetic recipe
        below has a real counterpart to be replaced by.
        """
        kinds = {r.kind for r in ktir.KtirBuilder.RECIPES.values()}
        self.assertNotIn(ktir.BindingKind.PAYLOAD, kinds)

    def test_the_identity_maps_are_stated_rather_than_implied(self):
        recipe = ktir.Recipe(
            arity=1, kind=ktir.BindingKind.PAYLOAD, binding=lambda: None
        )
        with self._registered("probe", recipe):
            plan = ktir.build_kernel_plan([make_op_spec("probe", inputs=1)])
        [step] = plan.steps
        self.assertIs(step.surface, ktir.Surface.GENERIC)
        self.assertEqual(step.reduce_dims, ())
        # Rank 3, one map per input and then the result: the operand and the
        # destination are read one element at a time in the same order.
        self.assertEqual(step.indexing.iters, ("parallel",) * 3)
        self.assertEqual(step.indexing.maps, ((0, 1, 2), (0, 1, 2)))


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
        a_access = ktir._access(a, a.device_size, a_q, a_layout)
        # The tile extent is device_size, which is what tiling already baked in.
        self.assertEqual(a_access.extent, (1, 1, 64))
        # Per view dim, the step each level takes: dim 0 <- n_stick, dim 1 <- m,
        # dim 2 <- nothing, i.e. the constant zero the kernel spells as %c0.
        self.assertEqual(a_access.index_coeffs, ((1, 0), (0, 1), (0, 0)))

        c_layout, c_q = ktir._solve_layout(c, levels)
        c_access = ktir._access(c, c.device_size, c_q, c_layout)
        self.assertEqual(c_access.extent, (1, 64))
        self.assertEqual(c_access.index_coeffs, ((1, 0), (0, 0)))

    def test_untiled_access_sits_at_the_view_origin(self):
        """Depth zero is the general answer, not a special case."""
        arg = make_op_spec().args[0]
        layout, q = ktir._solve_layout(arg, [])
        self.assertEqual(layout.extent, (16, 512, 64))
        self.assertEqual(q, [])
        access = ktir._access(arg, arg.device_size, q, layout)
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


# ---------------------------------------------------------------------------
# What we generate
# ---------------------------------------------------------------------------


class TestRefusals(unittest.TestCase):
    """The labelled capabilities this emitter does not implement.

    A label is a token shared by the raise and this test, so grepping it finds
    both.  No message here claims a consumer is the blocker: this repository
    cannot run dbo-opt or the scheduler, so what they accept is not observable
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

    def test_every_label_is_greppable_and_uniquely_owned(self):
        """Each label is raised from exactly one site, so grepping it is exact."""
        source = inspect.getsource(ktir)
        labels = re.findall(r'_unimplemented\(\s*\n?\s*"([^"]+)"', source)
        self.assertEqual(sorted(labels), sorted(set(labels)))
        self.assertEqual(sorted(labels), ["staggered-element-arrangement"])

    def test_no_refusal_message_blames_a_consumer(self):
        """A refusal says what is missing here, not what someone else rejects.

        Checked over the ``_unimplemented`` messages rather than the whole file:
        naming dbo-opt is legitimate where it explains why an *option* exists
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
