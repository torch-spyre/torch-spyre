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

"""What the scheduler resolves from a symbolic loop count, and carries down.

Three numbers come out of one symbolic count and they are easy to confuse. For
``FloorDiv(s, 64)`` with ``s <= 512``:

    the dimension's ceiling   512   declared to the backend as ``max_value``
    the tile size              64   declared as ``granularity``, and the step
    the largest trip count      8   the SDSC's full pre-tiling extent

The first two are carried on ``count_symbol_bounds``. The third is
``max_trip_count``. Mixing them up is silent: every one is a plausible integer.


The max and the tile size are resolved while the ShapeEnv exists and then
carried on ``LoopSpec.count_symbol_bounds``, because codegen also runs in a
reload phase where the ShapeEnv is gone.

A value that has to survive into generated source needs four things, and
missing any one is a defect that shows up only on the reload and never in a
direct test: the field, the provenance-schema entry, the **cache-key** entry
and the serializer line. This area has lost one of the four on four separate
occasions, so there is a test here for each.

No device. The ShapeEnv is stubbed the way test_codegen.py stubs it.
"""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import sympy
import torch
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv

from utils_inductor import expects_raise

from torch_spyre._inductor.kernel_provenance import (
    _EXPECTED_LOOP_SPEC_SCHEMA,
    _validate_finalized_schema,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import LoopSpec
from torch_spyre._inductor.pass_utils import (
    concretize_expr,
    declared_dims,
    declared_dims_in,
    declared_max_or_none,
    decompose_tiled_count,
    max_trip_count,
    symbolic_count_bounds,
)
from torch_spyre._inductor.spyre_kernel import (
    _codegen_op_spec_list,
    _place_symbol,
    _sympy_literal,
)
from torch_spyre._inductor.op_spec import walk_loop_specs
from torch_spyre._inductor.views import _concretize_for_cmp

SYM = "s0"
TILE = 64
MAX = 512


def _sym():
    return sympy.Symbol(SYM, integer=True, positive=True)


def _graph_with_upper(upper, shape_env=True):
    """A stub V.graph whose ShapeEnv reports `upper` for anything.

    Same shape as test_codegen.py's `_mock_v`. A real ShapeEnv would need the
    range installed through a trace, which is more machinery than the thing
    under test reads.
    """
    env = (
        SimpleNamespace(bound_sympy=lambda _e: SimpleNamespace(lower=TILE, upper=upper))
        if shape_env
        else None
    )
    return SimpleNamespace(sizevars=SimpleNamespace(shape_env=env))


def _graph_with_bounds(bounds, hint=None):
    """A stub V.graph whose ShapeEnv answers per expression.

    ``bounds`` maps ``str(expr)`` to an upper bound; anything absent reports
    infinity, which is how an undeclared symbol looks.
    """

    def bound_sympy(expr):
        return SimpleNamespace(lower=TILE, upper=bounds.get(str(expr), sympy.oo))

    sizevars = SimpleNamespace(shape_env=SimpleNamespace(bound_sympy=bound_sympy))
    if hint is not None:
        sizevars.optimization_hint = lambda _e: hint
    return SimpleNamespace(sizevars=sizevars)


class TestTheThreeNumbersAreNotTheSameNumber(unittest.TestCase):
    """One graph, two functions, two correct and different answers."""

    def test_the_trip_count_and_the_dimension_ceiling_differ(self):
        count = FloorDiv(_sym(), TILE)
        graph = _graph_with_bounds(
            {SYM: sympy.Integer(MAX), str(count): sympy.Integer(8)}
        )

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            trips = max_trip_count(count)
            bounds = symbolic_count_bounds(count)

        self.assertEqual(trips, 8)
        self.assertEqual(bounds, {SYM: (MAX, TILE)})
        self.assertNotEqual(trips, bounds[SYM][0])


class TestMaxTripCount(unittest.TestCase):
    def test_a_concrete_count_is_itself(self):
        self.assertEqual(max_trip_count(sympy.Integer(4)), 4)
        self.assertEqual(max_trip_count(4), 4)

    def test_a_symbolic_count_takes_the_declared_maximum_not_the_hint(self):
        """The hint is one call's size. Baking it in specialises the SDSC."""
        count = FloorDiv(_sym(), TILE)
        graph = _graph_with_bounds({}, hint=5)

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            self.assertEqual(max_trip_count(count), MAX // TILE)

    def test_an_undeclared_count_is_refused_even_with_a_shape_env_bound(self):
        """A ``mark_dynamic(max=)`` dimension has a ShapeEnv ceiling but no
        reservation behind it: its layout guard carries the exact size, every
        size recompiles, and an extent sized from that ceiling would describe a
        buffer nobody allocated. Refused, with the declaration as the fix."""
        count = FloorDiv(_sym(), TILE)
        graph = _graph_with_bounds({str(count): sympy.Integer(8)}, hint=5)

        with V.set_graph_handler(graph):
            with self.assertRaises(Unsupported) as caught:
                max_trip_count(count)

        self.assertIn(SYM, str(caught.exception))
        self.assertIn("declaration covers", str(caught.exception))
        self.assertIn("dynamic=", str(caught.exception))

    def test_no_ceiling_refuses_and_names_the_symbol(self):
        """There is no extent to describe, so guessing one is the wrong answer."""
        count = FloorDiv(_sym(), TILE)

        with V.set_graph_handler(_graph_with_bounds({})):
            with self.assertRaises(Unsupported) as caught:
                max_trip_count(count)

        self.assertIn(SYM, str(caught.exception))


class TestConcretizeExprPrefersTheDeclaredMax(unittest.TestCase):
    """Geometry has to serve the range, not the warm-up call.

    Measured the other way round: a binary built against a 320-row warm-up was
    correct at 128 and 256 and wrong at 448 and 512. Over-declaring is
    harmless, under-declaring is silently wrong.
    """

    def test_a_declared_symbol_resolves_to_its_maximum(self):
        """Declared means moved with ``tensor.to("spyre", dynamic=...)`` and
        recorded for the compile by ``declared_dims``. The ShapeEnv is not
        consulted for it, which is why the stub here carries no bound."""
        graph = _graph_with_bounds({}, hint=320)

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            self.assertEqual(concretize_expr(_sym()), MAX)

    def test_an_undeclared_symbol_still_takes_the_hint(self):
        """The regression half. A dim that went dynamic by accident has no
        declaration, so nothing better than the hint exists and nothing changes."""
        graph = _graph_with_bounds({}, hint=320)

        with V.set_graph_handler(graph):
            self.assertEqual(concretize_expr(_sym()), 320)

    def test_a_shape_env_ceiling_alone_is_not_a_declaration(self):
        """A ``mark_dynamic(max=...)`` user, or a ``torch._check`` range on an
        undeclared dim, gives the ShapeEnv a ceiling and nothing else. Geometry
        for them stays at the hint, as before this change: resolving to the
        ceiling would need the buffer reserved for it, and only the declaration
        reserves."""
        graph = _graph_with_bounds({SYM: sympy.Integer(MAX)}, hint=320)

        with V.set_graph_handler(graph):
            self.assertEqual(concretize_expr(_sym()), 320)

    def test_a_compound_of_a_declared_symbol_is_evaluated_at_the_maximum(self):
        graph = _graph_with_bounds({}, hint=5)

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            self.assertEqual(concretize_expr(FloorDiv(_sym(), TILE)), MAX // TILE)
            self.assertEqual(concretize_expr(_sym() * 2), 2 * MAX)

    def test_concrete_values_are_untouched(self):
        """Which is why this change cannot reach a static kernel: every
        expression in one returns before the ShapeEnv is consulted at all."""
        self.assertEqual(concretize_expr(64), 64)
        self.assertEqual(concretize_expr(sympy.Integer(64)), 64)


class TestTheTwoConcretizersAgree(unittest.TestCase):
    """``views._concretize_for_cmp`` is a second copy of ``concretize_expr``.

    The copy exists only because a module-level import would cycle. When the two
    disagree about a symbol, a comparison inside ``compute_coordinates`` or
    tensor alignment picks its branch at one size while the geometry built around
    it uses another. These arms make a future divergence fail here instead of
    sitting in a comment.
    """

    def test_both_resolve_a_declared_symbol_to_its_maximum(self):
        graph = _graph_with_bounds({}, hint=320)

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            self.assertEqual(_concretize_for_cmp(_sym()), MAX)
            self.assertEqual(concretize_expr(_sym()), MAX)

    def test_both_still_take_the_hint_without_a_declaration(self):
        """The control, in both forms an undeclared dim takes: no ceiling at
        all, and a ShapeEnv ceiling from ``mark_dynamic(max=...)``. Neither copy
        changes for either."""
        for bounds in ({}, {SYM: sympy.Integer(MAX)}):
            graph = _graph_with_bounds(bounds, hint=320)

            with V.set_graph_handler(graph):
                self.assertEqual(_concretize_for_cmp(_sym()), 320)
                self.assertEqual(concretize_expr(_sym()), 320)

    def test_the_copy_keeps_its_infinity_sentinels(self):
        """Why the copy cannot become a thin forwarder. ``add_term`` is handed
        ``limit=sympy.oo`` for an index with a non-zero storage offset, and
        ``int(sympy.oo)`` raises."""
        self.assertEqual(_concretize_for_cmp(sympy.oo), math.inf)
        self.assertEqual(_concretize_for_cmp(-sympy.oo), -math.inf)
        self.assertEqual(_concretize_for_cmp(math.inf), math.inf)

    def test_concrete_values_reach_neither_path(self):
        """Which is why this cannot touch a static kernel: every expression in
        one returns before the ShapeEnv is consulted at all."""
        self.assertEqual(_concretize_for_cmp(64), 64)
        self.assertEqual(_concretize_for_cmp(sympy.Integer(64)), 64)


class TestTheProducer(unittest.TestCase):
    def test_a_tiled_count_carries_the_max_and_the_tile_size(self):
        with V.set_graph_handler(_graph_with_upper(sympy.Integer(MAX))):
            bounds = symbolic_count_bounds(FloorDiv(_sym(), TILE))

        self.assertEqual(bounds, {SYM: (MAX, TILE)})

    def test_a_bare_symbol_carries_a_tile_size_of_one(self):
        with V.set_graph_handler(_graph_with_upper(sympy.Integer(MAX))):
            bounds = symbolic_count_bounds(_sym())

        self.assertEqual(bounds, {SYM: (MAX, 1)})

    def test_the_max_is_the_symbol_ceiling_not_the_trip_count(self):
        """Easy to get backwards, because the function is named for the count.

        For ``FloorDiv(s, 64)`` with ``s <= 512`` the carried max is 512, the
        symbol's ceiling, not 8, the largest number of trips. The bundle
        declares it as ``max_value`` on the dimension parameter.
        """
        with V.set_graph_handler(_graph_with_upper(sympy.Integer(MAX))):
            bounds = symbolic_count_bounds(FloorDiv(_sym(), TILE))

        self.assertEqual(bounds[SYM][0], MAX)
        self.assertNotEqual(bounds[SYM][0], MAX // TILE)

    def test_a_concrete_count_carries_nothing(self):
        with V.set_graph_handler(_graph_with_upper(sympy.Integer(MAX))):
            self.assertEqual(symbolic_count_bounds(sympy.Integer(8)), {})
            self.assertEqual(symbolic_count_bounds(8), {})

    def test_an_unrecognised_shape_carries_nothing(self):
        with V.set_graph_handler(_graph_with_upper(sympy.Integer(MAX))):
            self.assertEqual(symbolic_count_bounds(_sym() * 3), {})

    def test_no_shape_env_carries_nothing(self):
        with V.set_graph_handler(_graph_with_upper(None, shape_env=False)):
            self.assertEqual(symbolic_count_bounds(FloorDiv(_sym(), TILE)), {})

    def test_no_ceiling_carries_nothing_and_says_so(self):
        """The one empty result worth a warning: the user asked and did not get.

        Without a ceiling there is nothing for the geometry to be built
        against, so the kernel specialises, which looks exactly like never
        having asked for a symbolic loop.
        """
        from torch_spyre._inductor import pass_utils

        with V.set_graph_handler(_graph_with_upper(sympy.oo)):
            with mock.patch.object(pass_utils.logger, "warning") as warned:
                bounds = symbolic_count_bounds(FloorDiv(_sym(), TILE))

        self.assertEqual(bounds, {})
        self.assertTrue(warned.called, "silent about a loop that will specialise")
        self.assertIn("no finite upper bound", warned.call_args[0][0])


class TestEveryLoopLevelIsBuiltInOnePlace(unittest.TestCase):
    """The outermost loop and each nested level both get their bounds.

    They are built at two sites, SpyreKernel.wrap_op_specs_in_loop for the
    outermost and the scheduler's _codegen_loop_body for each nested level. Both
    go through SpyreKernel.counted_loop, so the bounds are resolved in one place,
    and so is the refusal of a count over an undeclared dimension, which is why
    both arms declare the symbol.
    """

    def test_the_outermost_loop_carries_its_bounds(self):
        from torch_spyre._inductor.spyre_kernel import SpyreKernel

        kernel = SimpleNamespace(
            op_specs=["body-op"], counted_loop=SpyreKernel.counted_loop
        )
        with (
            declared_dims({SYM: (TILE, MAX, TILE)}),
            V.set_graph_handler(_graph_with_upper(sympy.Integer(MAX))),
        ):
            SpyreKernel.wrap_op_specs_in_loop(kernel, FloorDiv(_sym(), TILE))

        (loop,) = kernel.op_specs
        self.assertEqual(loop.body, ["body-op"])
        self.assertEqual(loop.count_symbol_bounds, {SYM: (MAX, TILE)})

    def test_a_nested_level_carries_its_bounds(self):
        from torch_spyre._inductor.scheduler import SuperDSCScheduling

        kernel = SimpleNamespace(op_specs=["outer-op"])

        def codegen_into_kernel(_nodes, k):
            k.op_specs.append("inner-op")

        scheduling = SimpleNamespace(
            _live_nodes=lambda node: [], _codegen_into_kernel=codegen_into_kernel
        )
        node = SimpleNamespace(loop_count=FloorDiv(_sym(), TILE))
        with (
            declared_dims({SYM: (TILE, MAX, TILE)}),
            V.set_graph_handler(_graph_with_upper(sympy.Integer(MAX))),
        ):
            SuperDSCScheduling._codegen_loop_body(scheduling, node, kernel)

        outer_op, loop = kernel.op_specs
        self.assertEqual(outer_op, "outer-op")
        self.assertEqual(loop.body, ["inner-op"])
        self.assertEqual(loop.count_symbol_bounds, {SYM: (MAX, TILE)})


class TestPlacingASymbolOnALaunchArgument(unittest.TestCase):
    """Which argument and dim the runtime reads the dimension from."""

    def test_the_first_match_wins_in_argument_order(self):
        actuals = ["arg0", "arg1"]
        sizes = {"arg0": [8, 16], "arg1": [_sym(), 4]}

        self.assertEqual(_place_symbol(SYM, actuals, sizes), (1, 0))

    def test_the_dim_index_is_the_position_within_that_argument(self):
        actuals = ["arg0"]
        sizes = {"arg0": [4, 8, _sym()]}

        self.assertEqual(_place_symbol(SYM, actuals, sizes), (0, 2))

    def test_argument_order_decides_not_dict_order(self):
        """``arg_index`` means a position in ``actuals``, so that is iterated.

        The sizes dict is keyed by name and its order is an implementation
        detail. Iterating it instead would make the placement depend on
        insertion order, which is the kind of thing that only shows up as a
        different binary on someone else's machine.
        """
        actuals = ["arg0", "arg1"]
        sizes = {"arg1": [_sym()], "arg0": [_sym()]}

        self.assertEqual(_place_symbol(SYM, actuals, sizes), (0, 0))

    def test_an_unplaceable_symbol_is_none_rather_than_a_guess(self):
        actuals = ["arg0"]
        sizes = {"arg0": [8, 16]}

        self.assertIsNone(_place_symbol(SYM, actuals, sizes))

    def test_a_missing_or_empty_size_is_skipped_not_an_error(self):
        actuals = ["arg0", "arg1", "arg2"]
        sizes = {"arg0": None, "arg1": [], "arg2": [_sym()]}

        self.assertEqual(_place_symbol(SYM, actuals, sizes), (2, 0))

    def test_matching_is_by_name_so_it_survives_the_reload(self):
        """A reloaded symbol is a different object with the same name."""
        actuals = ["arg0"]
        assumption_free = sympy.sympify(SYM)
        sizes = {"arg0": [assumption_free]}

        self.assertNotEqual(assumption_free, _sym())
        self.assertEqual(_place_symbol(SYM, actuals, sizes), (0, 0))


class TestWalkingTheSpecTree(unittest.TestCase):
    def test_nested_loops_come_outer_first(self):
        inner = LoopSpec(count=sympy.Integer(2), body=[])
        outer = LoopSpec(count=FloorDiv(_sym(), TILE), body=[inner])

        self.assertEqual(list(walk_loop_specs([outer])), [outer, inner])

    def test_a_tree_with_no_loops_is_empty(self):
        self.assertEqual(list(walk_loop_specs([])), [])


class TestMoreThanOneSymbolIsRefusedHere(unittest.TestCase):
    """The one case that raises rather than returning an empty map.

    Everything else that yields `{}` leaves a kernel that specialises, which is
    a worse binary and not a wrong one. Two symbols on one loop is a check
    failure on the backend side, so no kernel can come of it and the refusal
    belongs where the message can tell the caller what to do.
    """

    @expects_raise
    def test_two_symbols_refuse_and_name_both(self):
        other = sympy.Symbol("s1", integer=True, positive=True)
        count = FloorDiv(_sym() + other, TILE)

        with V.set_graph_handler(_graph_with_bounds({})):
            with self.assertRaises(Unsupported) as caught:
                symbolic_count_bounds(count)

        message = str(caught.exception)
        self.assertIn(SYM, message)
        self.assertIn("s1", message)

    @expects_raise
    def test_the_message_says_how_to_fix_it(self):
        """A refusal a caller cannot act on is only half a refusal."""
        other = sympy.Symbol("s1", integer=True, positive=True)

        with V.set_graph_handler(_graph_with_bounds({})):
            with self.assertRaises(Unsupported) as caught:
                symbolic_count_bounds(_sym() * other)

        self.assertIn("torch._check", str(caught.exception))

    def test_a_pair_tied_by_a_check_counts_as_one_symbol(self):
        """``torch._check(a == b)`` lands in the ShapeEnv's replacements, and the
        count is read after they are applied, so the pair the message above asks
        for is not then refused as two symbols. The tied spelling resolves to
        the surviving symbol and the bounds are keyed by it.
        """
        other = sympy.Symbol("s1", integer=True, positive=True)
        graph = _graph_with_upper(sympy.Integer(MAX))
        graph.sizevars.shape_env.replacements = {other: _sym()}

        with V.set_graph_handler(graph):
            bounds = symbolic_count_bounds(FloorDiv(other, TILE))
            # Two names that the replacement folds into one are not refused.
            folded = symbolic_count_bounds(FloorDiv(_sym() + other, TILE))

        self.assertEqual(bounds, {SYM: (MAX, TILE)})
        self.assertNotIn("s1", folded)

    def test_one_symbol_is_still_fine(self):
        """The control: the refusal must not catch the supported case."""
        graph = _graph_with_bounds({SYM: sympy.Integer(MAX)})

        with V.set_graph_handler(graph):
            self.assertEqual(
                symbolic_count_bounds(FloorDiv(_sym(), TILE)), {SYM: (MAX, TILE)}
            )


class TestItSurvivesTheRealSerializer(unittest.TestCase):
    """Through the same functions codegen uses, not a reimplementation."""

    @staticmethod
    def _round_trip(spec):
        buf = IndentedBuffer()
        _codegen_op_spec_list([spec], buf, _sympy_literal)
        source = buf.getvalue()
        namespace = {"LoopSpec": LoopSpec, "sympify": sympy.sympify}
        return eval(f"[{source}]", namespace), source  # noqa: S307

    def test_the_bounds_come_back_intact(self):
        spec = LoopSpec(
            count=FloorDiv(_sym(), TILE),
            body=[],
            count_symbol_bounds={SYM: (MAX, TILE)},
        )

        (reloaded,), source = self._round_trip(spec)

        self.assertIn("count_symbol_bounds", source)
        self.assertEqual(reloaded.count_symbol_bounds, {SYM: (MAX, TILE)})

    def test_the_sources_come_back_intact(self):
        spec = LoopSpec(
            count=FloorDiv(_sym(), TILE),
            body=[],
            count_symbol_bounds={SYM: (MAX, TILE)},
            count_symbol_sources={SYM: (0, 0)},
        )

        (reloaded,), source = self._round_trip(spec)

        self.assertIn("count_symbol_sources", source)
        self.assertEqual(reloaded.count_symbol_sources, {SYM: (0, 0)})

    def test_the_count_changes_class_and_spelling_but_not_meaning(self):
        """Which is why the bounds are plain ints keyed by name.

        The count is written as ``sympify('(s0//64)')`` and comes back as a
        different class whose ``str`` is also different. What survives is the
        decomposition, which is why that is the one interpretation point. The
        bounds are not an expression, so they do not take part in any of this.
        """
        spec = LoopSpec(
            count=FloorDiv(_sym(), TILE),
            body=[],
            count_symbol_bounds={SYM: (MAX, TILE)},
        )

        (reloaded,), _source = self._round_trip(spec)

        self.assertNotIsInstance(reloaded.count, FloorDiv)
        self.assertNotEqual(
            str(reloaded.count),
            str(FloorDiv(_sym(), TILE)),
            "the round trip is string-stable on this sympy build, so the cache "
            "key normalisation below is no longer needed. Check before removing",
        )

        symbol, tile = decompose_tiled_count(reloaded.count)
        self.assertEqual((str(symbol), tile), (SYM, TILE))
        self.assertEqual(reloaded.count_symbol_bounds, {SYM: (MAX, TILE)})

    def test_a_concrete_loop_emits_no_bounds_line(self):
        spec = LoopSpec(count=sympy.Integer(4), body=[])

        (reloaded,), source = self._round_trip(spec)

        self.assertNotIn("count_symbol_bounds", source)
        self.assertEqual(reloaded.count_symbol_bounds, {})


class TestTheProvenanceSchemaAgrees(unittest.TestCase):
    def test_the_schema_matches_the_dataclass(self):
        """The validator compares annotation strings character for character.

        It exists so a field added to the finalized execution schema cannot be
        left out of the bundle key, so this is the test that catches a
        mismatched annotation rather than a reviewer catching it.
        """
        _validate_finalized_schema()

    def test_the_new_field_is_in_the_schema(self):
        self.assertIn("count_symbol_bounds", _EXPECTED_LOOP_SPEC_SCHEMA)

    def test_a_concrete_loop_keeps_its_bundle_key(self):
        """Adding this field must not rename every existing kernel's events.

        The bundle key names every profiler event, so a symbolic feature that
        changed it for non-symbolic kernels would be churn with no information
        in it. The payload entry is therefore present only when there are
        bounds, and `test_pins_rich_canonical_bundle_key` in
        test_kernel_provenance.py is the golden value that holds us to it.
        """
        from torch_spyre._inductor.kernel_provenance import _canonical_spec

        payload = _canonical_spec(LoopSpec(count=sympy.Integer(4), body=[]))

        self.assertNotIn("count_symbol_bounds", payload)

    def test_sources_are_also_conditional_in_the_payload(self):
        from torch_spyre._inductor.kernel_provenance import _canonical_spec

        payload = _canonical_spec(LoopSpec(count=sympy.Integer(4), body=[]))

        self.assertNotIn("count_symbol_sources", payload)

    def test_a_symbolic_loop_does_change_its_bundle_key(self):
        """The other half: when there are bounds, they are in the identity."""
        from torch_spyre._inductor.kernel_provenance import _canonical_spec

        without = _canonical_spec(LoopSpec(count=FloorDiv(_sym(), TILE), body=[]))
        with_bounds = _canonical_spec(
            LoopSpec(
                count=FloorDiv(_sym(), TILE),
                body=[],
                count_symbol_bounds={SYM: (MAX, TILE)},
            )
        )

        self.assertNotEqual(without, with_bounds)


class TestTheCacheKeySeparatesThem(unittest.TestCase):
    """The requirement this area has lost twice before."""

    @staticmethod
    def _key(bounds):
        from torch_spyre.execution.kernel_cache import compute_specs_hash

        return compute_specs_hash(
            [
                LoopSpec(
                    count=FloorDiv(_sym(), TILE),
                    body=[],
                    count_symbol_bounds=bounds,
                )
            ]
        )

    def test_two_declared_maxima_do_not_share_a_cache_entry(self):
        """Same count string, different range, so a different binary.

        The max reaches the bundle's input_arg and never the SDSC JSON, so
        without an explicit entry these two hash identically and the second
        caller is served a binary built for the first one's range.
        """
        self.assertNotEqual(
            self._key({SYM: (512, TILE)}), self._key({SYM: (1024, TILE)})
        )

    def test_two_tile_sizes_do_not_share_a_cache_entry(self):
        self.assertNotEqual(self._key({SYM: (MAX, 64)}), self._key({SYM: (MAX, 128)}))

    def test_two_sources_do_not_share_a_cache_entry(self):
        """Reading the same dimension off a different argument is a different
        bundle, because the parameter list it declares differs."""
        from torch_spyre.execution.kernel_cache import compute_specs_hash

        def key(sources):
            return compute_specs_hash(
                [
                    LoopSpec(
                        count=FloorDiv(_sym(), TILE),
                        body=[],
                        count_symbol_bounds={SYM: (MAX, TILE)},
                        count_symbol_sources=sources,
                    )
                ]
            )

        self.assertNotEqual(key({SYM: (0, 0)}), key({SYM: (1, 0)}))
        self.assertNotEqual(key({SYM: (0, 0)}), key({SYM: (0, 1)}))

    def test_identical_bounds_do_share_one(self):
        self.assertEqual(self._key({SYM: (MAX, TILE)}), self._key({SYM: (MAX, TILE)}))

    def test_the_key_survives_the_reload_spelling(self):
        """A reloaded kernel must find its own cache entry.

        The count is hashed as a string and the reload changes that string, so
        without normalisation a reloaded kernel computes a different key from
        the one that produced it and recompiles -- in the one feature whose
        whole purpose is not recompiling.
        """
        from torch_spyre.execution.kernel_cache import compute_specs_hash

        before = FloorDiv(_sym(), TILE)
        after = sympy.sympify(str(before))
        self.assertNotEqual(str(before), str(after), "fixture is not exercising it")

        keys = [
            compute_specs_hash(
                [LoopSpec(count=count, body=[], count_symbol_bounds={SYM: (MAX, TILE)})]
            )
            for count in (before, after)
        ]

        self.assertEqual(keys[0], keys[1])


class TestTheRealKernelCacheRoundTrip(unittest.TestCase):
    """Write an entry, reload the spec, and find the entry again.

    The tests above check the serializer and the key one at a time. This one
    goes through the kernel cache itself, in a scratch cache root: the first
    compile commits an entry under its key, the reloaded spec must compute the
    same key and hit it, and the loop_dimension kind stored with the entry must
    come back with its bounds and source intact.
    """

    def test_a_reloaded_kernel_hits_the_entry_its_first_compile_wrote(self):
        import os
        import tempfile

        from torch_spyre._inductor.codegen.compute_ops import SymbolKind
        from torch_spyre.execution import kernel_cache

        spec = LoopSpec(
            count=FloorDiv(_sym(), TILE),
            body=[],
            count_symbol_bounds={SYM: (MAX, TILE)},
            count_symbol_sources={SYM: (0, 0)},
        )
        kinds = [
            SymbolKind.loop_dimension(
                granularity=TILE,
                max_value=MAX,
                pytorch_sym=SYM,
                arg_index=0,
                dim_index=0,
            ),
            SymbolKind.kernel(0),
        ]

        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": root}),
        ):
            key = kernel_cache.compute_specs_hash([spec])
            tmp_dir = kernel_cache.allocate_compile_dir(key, kernel_name="k")
            for artifact in (*kernel_cache._REQUIRED_ARTIFACTS, "sdsc_0.json"):
                path = os.path.join(tmp_dir, artifact)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write("{}")
            kernel_cache.save_symbol_kinds(tmp_dir, kinds)
            kernel_cache.commit_compile_dir(tmp_dir, key)

            (reloaded,), _source = TestItSurvivesTheRealSerializer._round_trip(spec)
            reloaded_key = kernel_cache.compute_specs_hash([reloaded])
            cached_dir = kernel_cache.get_cached_kernel_dir(reloaded_key)

            self.assertEqual(reloaded_key, key)
            self.assertIsNotNone(cached_dir, "the reloaded kernel missed its entry")
            self.assertEqual(kernel_cache.load_symbol_kinds(cached_dir), kinds)
        self.assertEqual(reloaded.count_symbol_bounds, {SYM: (MAX, TILE)})
        self.assertEqual(reloaded.count_symbol_sources, {SYM: (0, 0)})


class TestTheFloorDivSpellingCanary(unittest.TestCase):
    """The reload path depends on how a FloorDiv prints. Fail loudly if it moves.

    The serializer writes a count as ``sympify('<str(count)>')`` and the cache
    key hashes the count after one more sympify. Both rely on
    ``str(FloorDiv(s, G))`` being ``(s//G)``, which sympify reads back as
    ``floor(s/G)``. If sympy or torch prints FloorDiv another way, the reloaded
    count may not decompose and the key may not normalise, so check
    ``decompose_tiled_count`` and ``_normalized_expr_str`` before updating this.
    """

    def test_floordiv_prints_as_floor_division(self):
        self.assertEqual(str(FloorDiv(_sym(), TILE)), f"({SYM}//{TILE})")

    def test_the_printed_count_reads_back_as_the_same_decomposition(self):
        from torch_spyre.execution.kernel_cache import _normalized_expr_str

        count = FloorDiv(_sym(), TILE)
        reread = sympy.sympify(str(count))

        symbol, tile = decompose_tiled_count(reread)
        self.assertEqual((str(symbol), tile), (SYM, TILE))
        self.assertEqual(_normalized_expr_str(reread), _normalized_expr_str(count))


class TestReadingTheDeclarationsOffTheGraphInputs(unittest.TestCase):
    """``_declared_dims_of`` pairs each placeholder's real input (the
    reservation) with its fake input (the ShapeEnv symbol). Driven here with
    stand-in placeholders and an injected reservation reader, no device."""

    @staticmethod
    def _symint(name, hint):
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        return torch.SymInt(
            SymNode(
                sympy.Symbol(name, integer=True, positive=True),
                ShapeEnv(),
                int,
                hint=hint,
            )
        )

    @classmethod
    def _placeholder(cls, name, sizes):
        real = torch.empty(tuple(s if isinstance(s, int) else 1 for s in sizes))
        fake = SimpleNamespace(dim=lambda: len(sizes), size=lambda d: sizes[d])
        return real, SimpleNamespace(
            op="placeholder",
            name=name,
            meta={"grapharg": SimpleNamespace(example=real), "example_value": fake},
        )

    @staticmethod
    def _graph(*nodes):
        return SimpleNamespace(graph=SimpleNamespace(nodes=list(nodes)))

    def test_a_declared_input_is_recorded_by_its_symbol(self):
        from torch_spyre._inductor import _declared_dims_of

        real, node = self._placeholder("x", [self._symint(SYM, 256), 64])
        spec = {0: {"min": TILE, "max": MAX, "granularity": TILE}}

        declared = _declared_dims_of(
            self._graph(node), [real], reserved_dims_of=lambda t: spec
        )

        self.assertEqual(declared, {SYM: (TILE, MAX, TILE)})

    def test_two_inputs_with_one_contract_agree(self):
        from torch_spyre._inductor import _declared_dims_of

        real_a, a = self._placeholder("a", [self._symint(SYM, 256), 64])
        real_b, b = self._placeholder("b", [self._symint(SYM, 256), 64])
        spec = {0: {"min": TILE, "max": MAX, "granularity": TILE}}

        declared = _declared_dims_of(
            self._graph(a, b), [real_a, real_b], reserved_dims_of=lambda t: spec
        )

        self.assertEqual(declared, {SYM: (TILE, MAX, TILE)})

    def test_two_contracts_for_one_symbol_are_refused(self):
        from torch_spyre._inductor import _declared_dims_of

        real_a, a = self._placeholder("a", [self._symint(SYM, 256), 64])
        real_b, b = self._placeholder("b", [self._symint(SYM, 256), 64])
        specs = {
            id(real_a): {0: {"min": TILE, "max": MAX, "granularity": TILE}},
            id(real_b): {0: {"min": TILE, "max": 2 * MAX, "granularity": TILE}},
        }

        with self.assertRaises(Unsupported) as caught:
            _declared_dims_of(
                self._graph(a, b),
                [real_a, real_b],
                reserved_dims_of=lambda t: specs[id(t)],
            )

        self.assertIn("One symbol carries one contract", str(caught.exception))

    def test_a_declared_and_an_undeclared_input_sharing_the_symbol_are_refused(
        self,
    ):
        from torch_spyre._inductor import _declared_dims_of

        real_a, a = self._placeholder("a", [self._symint(SYM, 256), 64])
        real_b, b = self._placeholder("b", [self._symint(SYM, 256), 64])
        specs = {id(real_a): {0: {"min": TILE, "max": MAX, "granularity": TILE}}}

        with self.assertRaises(Unsupported) as caught:
            _declared_dims_of(
                self._graph(a, b),
                [real_a, real_b],
                reserved_dims_of=lambda t: specs.get(id(t)),
            )

        self.assertIn(SYM, str(caught.exception))
        self.assertIn("undeclared on another", str(caught.exception))

    def test_an_undeclared_input_with_its_own_symbol_is_left_alone(self):
        from torch_spyre._inductor import _declared_dims_of

        real_a, a = self._placeholder("a", [self._symint(SYM, 256), 64])
        real_b, b = self._placeholder("b", [self._symint("s1", 300), 64])
        specs = {id(real_a): {0: {"min": TILE, "max": MAX, "granularity": TILE}}}

        declared = _declared_dims_of(
            self._graph(a, b),
            [real_a, real_b],
            reserved_dims_of=lambda t: specs.get(id(t)),
        )

        self.assertEqual(declared, {SYM: (TILE, MAX, TILE)})


class TestTheDeclaredDimsOfACompile(unittest.TestCase):
    """``declared_dims`` is what makes a symbol declared for the helpers above.

    The Spyre ``compile_fx`` wrapper fills it from the graph inputs that carry a
    ``tensor.to("spyre", dynamic=...)`` reservation, before lowering. These arms
    pin what the helpers do with it, with no ShapeEnv bound anywhere, so a
    declared dimension needs no ``torch._check`` range to get its geometry.
    """

    def test_outside_a_compile_nothing_is_declared(self):
        self.assertEqual(declared_dims_in(_sym()), {})
        self.assertIsNone(declared_max_or_none(_sym()))

    def test_the_block_declares_and_restores(self):
        with declared_dims({SYM: (TILE, MAX, TILE)}):
            self.assertEqual(declared_dims_in(_sym()), {SYM: (TILE, MAX, TILE)})
            with declared_dims({}):
                self.assertEqual(declared_dims_in(_sym()), {})
            self.assertEqual(declared_dims_in(_sym()), {SYM: (TILE, MAX, TILE)})
        self.assertEqual(declared_dims_in(_sym()), {})

    def test_a_mixed_expression_takes_the_maximum_for_the_declared_part(self):
        """One declared symbol and one that is not: the declared one is built at
        its maximum and the other keeps the hint, so the declared range is still
        covered. Falling back to the hint for both would rebuild the one-size
        failure for the declared symbol."""
        other = sympy.Symbol("s1", integer=True, positive=True)
        graph = _graph_with_bounds({})
        # This call's sizes: 128 rows for the declared dim, 5 for the other.
        graph.sizevars.optimization_hint = lambda e: int(
            sympy.sympify(e).subs({_sym(): 128, other: 5})
        )

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            self.assertEqual(declared_dims_in(_sym() + other), {SYM: (TILE, MAX, TILE)})
            self.assertIsNone(declared_max_or_none(_sym() + other))
            self.assertEqual(concretize_expr(_sym() + other), MAX + 5)

    def test_a_size_that_shrinks_with_the_dimension_is_refused(self):
        """The maximum is the safe end only for a size that grows with the
        dimension. ``MAX - s`` is largest at this call's hint, so geometry built
        at the maximum would not cover the call being compiled."""
        graph = _graph_with_bounds({}, hint=128)

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            with self.assertRaises(Unsupported) as caught:
                concretize_expr(sympy.Integer(MAX) - _sym())

        self.assertIn("monotonically", str(caught.exception))

    def test_a_declared_count_needs_no_shape_env_bound(self):
        """The trip count of a loop over a declared dim resolves from the
        declaration, so a traced function that declares and does not also write
        ``torch._check(rows <= max)`` still gets its geometry."""
        graph = _graph_with_bounds({})

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            self.assertEqual(max_trip_count(FloorDiv(_sym(), TILE)), MAX // TILE)
            self.assertEqual(
                symbolic_count_bounds(FloorDiv(_sym(), TILE)), {SYM: (MAX, TILE)}
            )


if __name__ == "__main__":
    unittest.main()
