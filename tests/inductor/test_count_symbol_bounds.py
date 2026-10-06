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

The first two are carried on ``count_symbol_bounds``, which is what this file
covers. The third is resolved separately, where the SDSC needs it. Mixing them
up is silent: every one of the three is a plausible integer.


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

import unittest
from types import SimpleNamespace
from unittest import mock

import sympy
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor.kernel_provenance import (
    _EXPECTED_LOOP_SPEC_SCHEMA,
    _validate_finalized_schema,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import LoopSpec
from torch_spyre._inductor.pass_utils import (
    decompose_tiled_count,
    symbolic_count_bounds,
)
from torch_spyre._inductor.spyre_kernel import (
    _codegen_op_spec_list,
    _place_symbol,
    _sympy_literal,
)
from torch_spyre._inductor.op_spec import walk_loop_specs

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

    def test_two_symbols_refuse_and_name_both(self):
        other = sympy.Symbol("s1", integer=True, positive=True)
        count = FloorDiv(_sym() + other, TILE)

        with V.set_graph_handler(_graph_with_bounds({})):
            with self.assertRaises(Unsupported) as caught:
                symbolic_count_bounds(count)

        message = str(caught.exception)
        self.assertIn(SYM, message)
        self.assertIn("s1", message)

    def test_the_message_says_how_to_fix_it(self):
        """A refusal a caller cannot act on is only half a refusal."""
        other = sympy.Symbol("s1", integer=True, positive=True)

        with V.set_graph_handler(_graph_with_bounds({})):
            with self.assertRaises(Unsupported) as caught:
                symbolic_count_bounds(_sym() * other)

        self.assertIn("torch._check", str(caught.exception))

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


if __name__ == "__main__":
    unittest.main()
