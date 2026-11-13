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

"""Tests for the compiler side of the reserved_dims contract.

The functions here only read a few attributes of the graph, so they run against a
stand-in and need no device. The behaviour across two compiled kernels is in
test_cross_kernel_reserved_dims.py.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import sympy
import torch
from torch._inductor.ir import FixedLayout, InputBuffer, StorageBox, TensorBox
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges

from torch_spyre._inductor import reserved_dims_contract as contract
from torch_spyre._inductor.errors import Unsupported

COLS = 64

s0 = sympy.Symbol("s0", integer=True, positive=True)
s1 = sympy.Symbol("s1", integer=True, positive=True)


def make_graph(ranges=None, contracts=None):
    """A stand-in for GraphLowering with the attributes the contract code reads.

    ``ranges`` is ``{symbol: (low, high)}`` for the ShapeEnv; a high of None means
    no upper bound, like a freshly created dynamic dim.
    """
    shape_env = ShapeEnv()
    for symbol, (low, high) in (ranges or {}).items():
        shape_env.var_to_range[symbol] = ValueRanges(
            low, int_oo if high is None else high
        )
    graph = SimpleNamespace(
        sizevars=SimpleNamespace(shape_env=shape_env),
        graph_input_names=[],
        graph_inputs={},
    )
    graph._spyre_dim_contracts = contracts or {}
    return graph


def add_input(graph, name, size):
    """Register a graph input of the given (possibly symbolic) size."""
    layout = FixedLayout(torch.device("cpu"), torch.float16, size, [COLS, 1])
    buffer = InputBuffer(name=name, layout=layout)
    graph.graph_input_names.append(name)
    graph.graph_inputs[name] = TensorBox(StorageBox(buffer))
    return torch.empty(0)


def run_pass(graph, tensor, reserved_dims):
    """Run the read pass with ``reserved_dims`` standing in for the tensor's own."""
    with V.set_real_inputs([tensor]):
        with mock.patch.object(
            contract, "get_reserved_dims", lambda _tensor: reserved_dims
        ):
            contract.apply_reserved_dims_contract(graph)


def declared(dim_min, dim_max, granularity):
    return {0: {"min": dim_min, "max": dim_max, "granularity": granularity}}


class TestOutputGranularity(unittest.TestCase):
    """The step between an output's legal sizes: the gcd of the size over every
    legal input size, or 1 whenever that cannot be worked out."""

    CONTRACTS = {s0: (128, 512, 64), s1: (96, 384, 96)}

    def test_size_expressions_of_common_ops(self):
        cases = {
            # One 128-element row per 64-row tile: multiples of 128.
            "tiled sum": (128 * FloorDiv(s0, 64), 128),
            # cat([x, x]) doubles every size, and the granularity with it.
            "concat with itself": (2 * s0, 128),
            # Multiples of 64 plus 16: only 16 is common.
            "pad by 16 rows": (s0 + 16, 16),
            # 127, 191, 255, ...: nothing in common.
            "drop the first row": (s0 - 1, 1),
            # Multiples of 64 plus multiples of 96: 32 is common.
            "concat of two inputs": (s0 + s1, 32),
        }
        for op, (size, expected) in cases.items():
            with self.subTest(op=op):
                self.assertEqual(
                    contract.output_granularity(size, self.CONTRACTS), expected
                )

    def test_first_legal_size_is_the_next_multiple_of_the_granularity(self):
        # min 100 at granularity 64 starts at 128, so s0 + 16 stays at 16. Starting
        # at 100 itself would give the sizes 116, 180, ... and a granularity of 4.
        size = contract.output_granularity(s0 + 16, {s0: (100, 512, 64)})
        self.assertEqual(size, 16)

    def test_falls_back_to_one_when_it_cannot_tell(self):
        cases = {
            "symbol without a contract": (2 * s0, {}),
            # Granularity 1 over [1, 100000] has more legal sizes than the limit.
            "too many sizes to enumerate": (2 * s0, {s0: (1, 100000, 1)}),
        }
        for reason, (size, contracts) in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(contract.output_granularity(size, contracts), 1)


class TestApplyReservedDimsContract(unittest.TestCase):
    def test_declared_range_and_granularity_reach_the_compile(self):
        graph = make_graph(ranges={s0: (2, None)})
        x = add_input(graph, "arg0", [s0, COLS])

        run_pass(graph, x, declared(128, 512, 64))

        bounds = graph.sizevars.shape_env.bound_sympy(s0)
        self.assertEqual((bounds.lower, bounds.upper), (128, 512))
        # The ShapeEnv has no place for granularity, so it is kept beside it.
        self.assertEqual(contract.get_dim_contracts(graph), {s0: (128, 512, 64)})

    def test_a_contract_only_tightens_an_existing_range(self):
        graph = make_graph(ranges={s0: (200, 1000)})
        x = add_input(graph, "arg0", [s0, COLS])

        run_pass(graph, x, declared(128, 512, 64))

        bounds = graph.sizevars.shape_env.bound_sympy(s0)
        self.assertEqual((bounds.lower, bounds.upper), (200, 512))

    def test_input_without_a_contract_is_left_alone(self):
        # Every ordinary compile takes this path.
        graph = make_graph(ranges={s0: (2, None)})
        x = add_input(graph, "arg0", [s0, COLS])

        run_pass(graph, x, None)

        self.assertEqual(contract.get_dim_contracts(graph), {})
        bounds = graph.sizevars.shape_env.bound_sympy(s0)
        self.assertEqual((bounds.lower, bounds.upper), (2, int_oo))

    def test_static_dim_has_no_symbol_to_constrain(self):
        graph = make_graph()
        x = add_input(graph, "arg0", [128, COLS])

        run_pass(graph, x, declared(128, 512, 64))

        self.assertEqual(contract.get_dim_contracts(graph), {})


class TestOutputReservedDims(unittest.TestCase):
    def graph(self, upper=512):
        return make_graph(
            ranges={s0: (128, upper), s1: (96, 384)},
            contracts={s0: (128, 512, 64), s1: (96, 384, 96)},
        )

    def test_static_output_is_not_stamped(self):
        self.assertIsNone(contract.output_reserved_dims(self.graph(), (128, COLS)))

    def test_contract_follows_the_output_size(self):
        cases = {
            "same size": (s0, {0: (128, 512, 64)}),
            # A sum over 64-row tiles leaves s0 // 64 rows: 2 to 8, in steps of 1.
            "divided size": (FloorDiv(s0, 64), {0: (2, 8, 1)}),
        }
        for name, (size, expected) in cases.items():
            with self.subTest(size=name):
                stamp = contract.output_reserved_dims(self.graph(), (size, COLS))
                self.assertEqual(stamp, expected)

    def test_output_with_no_upper_bound_is_unknown_not_static(self):
        graph = self.graph(upper=None)
        self.assertIsNone(contract.output_reserved_dims(graph, (s0, COLS)))

    def test_more_than_one_symbolic_dim_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "more than one symbolic dim"):
            contract.output_reserved_dims(self.graph(), (s0, s1, COLS))


if __name__ == "__main__":
    unittest.main()
