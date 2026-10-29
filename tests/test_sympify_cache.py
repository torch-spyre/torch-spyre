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

"""CPU-only tests for the memoized sympify of generated wrapper modules.

Run with:
    python -m pytest tests/test_sympify_cache.py -v
"""

import inspect
import unittest

import sympy
from sympy.core.parameters import evaluate

from torch_spyre._inductor import wrapper
from torch_spyre._inductor.op_spec import _cached_sympify, wrapper_sympify

# Expression strings as emitted by sympy_str() into generated wrapper modules.
_EXPRS = [
    "0",
    "1",
    "4096",
    "c0",
    "c4",
    "z0",
    "z0 + 7",
    "c1 + 40",
    "floor(c1/32)",
    "floor(c1/64) + 192",
    "Mod(c1, 32)",
    "Mod(core_id, 32)",
    "Mod(floor(core_id/8), 4)",
    "Mod(floor(Mod(core_id, 4)), 4)",
    "Mod(floor(Mod(floor(core_id/8), 4)), 4)",
    "_tile_adv_while_loop_body_graph_0_0_op7_lvl0",
    "floor(1024*_tile_adv_while_loop_body_graph_0_0_op7_lvl0)",
]

# A fragment of a generated OpSpec, evaluated the way a wrapper module is.
_GENERATED = (
    "({sympify('c0'): (sympify('2'), 1), sympify('c1'): (sympify('4096'), 32)}, "
    "{sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 32)')}, "
    "[sympify('floor(c1/32)'), sympify('c0'), sympify('Mod(c1, 32)')])"
)


def _signature(expr):
    """Structure plus the assumptions of every free symbol."""
    symbols = sorted(
        (s.name, tuple(sorted(s.assumptions0.items()))) for s in expr.free_symbols
    )
    return sympy.srepr(expr), symbols


class TestWrapperSympify(unittest.TestCase):
    def setUp(self):
        _cached_sympify.cache_clear()

    def tearDown(self):
        _cached_sympify.cache_clear()

    def test_cold_and_cached_equal_sympy(self):
        cold = [wrapper_sympify(s) for s in _EXPRS]
        cached = [wrapper_sympify(s) for s in _EXPRS]
        for text, a, b in zip(_EXPRS, cold, cached):
            ref = sympy.sympify(text)
            for got in (a, b):
                self.assertEqual(got, ref, text)
                self.assertEqual(_signature(got), _signature(ref), text)

    def test_cache_hit_returns_equal_object(self):
        first = wrapper_sympify("Mod(floor(core_id/8), 4)")
        hits = _cached_sympify.cache_info().hits
        second = wrapper_sympify("Mod(floor(core_id/8), 4)")
        self.assertEqual(_cached_sympify.cache_info().hits, hits + 1)
        self.assertEqual(second, first)
        self.assertIs(second, first)
        self.assertEqual(second, sympy.sympify("Mod(floor(core_id/8), 4)"))

    def test_non_str_passes_through(self):
        c0 = sympy.Symbol("c0")
        args = [4, 2.5, c0, sympy.Integer(7), sympy.floor(c0 / 32), [1, "c0"]]
        for a in args:
            self.assertEqual(wrapper_sympify(a), sympy.sympify(a))
        # Extra sympify arguments also bypass the cache.
        got = wrapper_sympify("0.5*c0", rational=True)
        self.assertEqual(got, sympy.sympify("0.5*c0", rational=True))
        self.assertEqual(_cached_sympify.cache_info().currsize, 0)

    def test_cache_is_bounded(self):
        maxsize = _cached_sympify.cache_info().maxsize
        self.assertIsNotNone(maxsize)
        self.assertGreaterEqual(maxsize, 1024)

    def test_evaluate_flag_is_honoured(self):
        text = "2*3 + c0"
        evaluated = wrapper_sympify(text)
        with evaluate(False):
            unevaluated = wrapper_sympify(text)
            ref = sympy.sympify(text)
        self.assertEqual(sympy.srepr(unevaluated), sympy.srepr(ref))
        self.assertEqual(evaluated, sympy.sympify(text))
        self.assertNotEqual(sympy.srepr(evaluated), sympy.srepr(unevaluated))

    def test_generated_fragment_equals_sympy(self):
        results = []
        for _ in range(2):  # cold, then cached
            ns = {}
            exec(
                "from torch_spyre._inductor.op_spec import wrapper_sympify as sympify",
                ns,
            )
            results.append(eval(_GENERATED, ns))
        ref = eval(_GENERATED, {"sympify": sympy.sympify})
        for got in results:
            self.assertEqual(got, ref)

    def test_wrapper_header_imports_wrapper_sympify(self):
        self.assertIn(
            "SPYRE_WRAPPER_IMPORTS",
            inspect.getsource(wrapper.SpyrePythonWrapperCodegen.write_header),
        )
        header = wrapper.SPYRE_WRAPPER_IMPORTS
        self.assertIn(
            "from torch_spyre._inductor.op_spec import wrapper_sympify as sympify",
            header,
        )


if __name__ == "__main__":
    unittest.main()
