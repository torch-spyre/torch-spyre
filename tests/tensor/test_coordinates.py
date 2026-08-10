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

import sympy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import FixedLayout
from torch_spyre._C import DataFormats, SpyreTensorLayout
from torch_spyre._inductor.codegen.superdsc import _get_device_dim_order
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import TensorArg
from torch_spyre._inductor.pass_utils import (
    device_coordinates,
    try_device_coordinates,
)
from torch_spyre._inductor.propagate_layouts import (
    MatmulPreferredOrder,
    PropArg,
    _check_supported_input_sticks,
    _default_matmul_input_layout,
    _matches_preferred_matmul_device_order,
    _matmul_preferred_input_orders,
    _preferred_matmul_output_dim_order,
    find_stick_compatible_input_layout,
)
from torch_spyre._inductor.spyre_kernel import _preserve_shared_weight_unit_bmm_dim
from torch_spyre._inductor.views import compute_coordinates, tiling_expr_to_device_expr
from torch.utils._sympy.functions import ModularIndexing

p0, p1, p2, p3, p4, p5 = sympy.symbols("p0 p1 p2 p3 p4 p5", integer=True)


class TestCoordinates(TestCase):
    def setUp(self):
        torch.manual_seed(0xAFFE)

    def test_compute_coordinates(self):
        # B, S, E -> B, E/H, S, H
        cx = compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 2, p1: 32, p2: 256, p3: 128},
            1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
        )
        self.assertEqual(cx, [p0, p2, 128 * p1 + p3])

        # B, S, E -> B*S, E
        cx = compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 512, p1: 4096},
            4096 * p0 + p1,
        )
        self.assertEqual(cx, [p0 // 256, p0 % 256, p1])

        # B, S, E -> B*S, E (explicit Mod in index)
        cx = compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 512, p1: 4096},
            4096 * (p0 % 256) + p1,
        )
        self.assertEqual(cx, [0, p0 % 256, p1])

        # B, S, E -> B*S, E (via ModularIndexing)
        cx = compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 512, p1: 4096},
            4096 * ModularIndexing(p0, 1, 256) + p1,
        )
        self.assertEqual(cx, [0, p0 % 256, p1])

        # dim of size 1 with stride>0
        cx = compute_coordinates(
            [3, 1, 128],
            [128, 128, 1],
            {p0: 3, p1: 128},
            128 * p0 + p1,
        )
        self.assertEqual(cx, [p0, 0, p1])

        # dim of size 1 with stride<0
        cx = compute_coordinates(
            [3, 1, 128],
            [128, -1, 1],
            {p0: 3, p1: 128},
            128 * p0 + p1,
        )
        self.assertEqual(cx, [p0, 0, p1])

        # dims of size 1
        cx = compute_coordinates(
            [4, 1, 1, 3, 1, 128],
            [384, 384, -1, 128, -1, 1],
            {p0: 4, p1: 1, p2: 1, p3: 3, p4: 1, p5: 128},
            384 * p0 + 128 * p3 + p5,
        )
        self.assertEqual(cx, [p0, 0, 0, p3, 0, p5])

        # dim with stride==0
        cx = compute_coordinates(
            [3, 42, 128],
            [128, 0, 1],
            {p0: 3, p1: 42, p2: 128},
            128 * p0 + p1,
        )
        self.assertEqual(cx, [p0, 0, p1])

        # split(x, dim=0, sections=3)[1]: offset = 5760 * 3 = 17280
        cx = compute_coordinates(
            [9, 15, 384],
            [5760, 384, 1],
            {p0: 3, p1: 15, p2: 384},
            5760 * p0 + 384 * p1 + p2 + 17280,
        )
        self.assertEqual(cx, [p0 + 3, p1, p2])

        # split(x, dim=1, sections=3)[1]: offset = 384 * 5 = 1920
        cx = compute_coordinates(
            [9, 15, 384],
            [5760, 384, 1],
            {p0: 9, p1: 5, p2: 384},
            5760 * p0 + 384 * p1 + p2 + 1920,
        )
        self.assertEqual(cx, [p0, p1 + 5, p2])

        # split(x, dim=2, sections=3)[1]: offset = 1 * 128 = 128
        cx = compute_coordinates(
            [9, 15, 384],
            [5760, 384, 1],
            {p0: 9, p1: 15, p2: 128},
            5760 * p0 + 384 * p1 + p2 + 128,
        )
        self.assertEqual(cx, [p0, p1, p2 + 128])

        # offset spanning dimensions
        cx = compute_coordinates(
            [10, 20, 30],
            [600, 30, 1],
            {p0: 10, p1: 20, p2: 30},
            600 * p0 + 30 * p1 + p2 + 1855,
        )
        # offset 1855 = 3*600 + 1*30 + 25*1
        self.assertEqual(cx, [p0 + 3, p1 + 1, p2 + 25])

    def test_compute_device_coordinates(self):
        # B, S, E -> B, E/H, S, H
        cx = compute_coordinates(
            [256, 64, 2, 64],
            [4096, 64, 1048576, 1],
            {p0: 2, p1: 32, p2: 256, p3: 128},
            1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
        )
        self.assertEqual(cx, [p2, 2 * p1 + p3 // 64, p0, p3 % 64])

        # B, S, E -> B*S, E
        cx = compute_coordinates(
            [256, 64, 2, 64],
            [4096, 64, 1048576, 1],
            {p0: 512, p1: 4096},
            4096 * p0 + p1,
        )
        self.assertEqual(cx, [p0 % 256, p1 // 64, p0 // 256, p1 % 64])

        # split(x, dim=0, sections=3)[1]: offset = 5760 * 3 = 17280
        cx = compute_coordinates(
            [15, 6, 9, 64],
            [384, 64, 5760, 1],
            {p0: 3, p1: 15, p2: 384},
            5760 * p0 + 384 * p1 + p2 + 17280,
        )
        self.assertEqual(cx, [p1, p2 // 64, p0 + 3, p2 % 64])

        # split(x, dim=1, sections=3)[1]: offset = 384 * 5 = 1920
        cx = compute_coordinates(
            [15, 6, 9, 64],
            [384, 64, 5760, 1],
            {p0: 9, p1: 5, p2: 384},
            5760 * p0 + 384 * p1 + p2 + 1920,
        )
        self.assertEqual(cx, [p1 + 5, p2 // 64, p0, p2 % 64])

        # split(x, dim=2, sections=3)[1]: offset = 1 * 128 = 128
        cx = compute_coordinates(
            [15, 6, 9, 64],
            [384, 64, 5760, 1],
            {p0: 9, p1: 15, p2: 128},
            5760 * p0 + 384 * p1 + p2 + 128,
        )
        self.assertEqual(cx, [p1, p2 // 64 + 2, p0, p2 % 64])

        # non-contiguous strides with offset
        cx = compute_coordinates(
            [256, 64, 2, 64],
            [4096, 64, 1048576, 1],
            {p0: 2, p1: 32, p2: 256, p3: 128},
            1048576 * p0 + 128 * p1 + 4096 * p2 + p3 + 200,
        )
        # offset 200 = 0*1048576 + 0*4096 + 3*64 + 8*1
        self.assertEqual(cx, [p2, 2 * p1 + p3 // 64 + 3, p0, p3 % 64 + 8])

        # splitting the stick dimension
        cx = compute_coordinates(
            [15, 6, 9, 64],
            [384, 64, 5760, 1],
            {p0: 9, p1: 15, p2: 128},
            5760 * p0 + 384 * p1 + p2 + 128,
        )
        self.assertEqual(cx, [p1, p2 // 64 + 2, p0, p2 % 64])


class TestUnrepresentableStickCandidates(TestCase):
    """Cover the skip-unrepresentable-candidate behavior added for the
    ``floor(var/N)`` cross-stick crash (transpose feeding a matmul).

    A candidate device layout can have a stick expression the backend cannot
    represent (e.g. ``floor(d2/128)``). ``device_coordinates`` raises
    ``Unsupported`` on such sticks; the enumeration sites use
    ``try_device_coordinates`` to skip them instead of aborting the compile
    when another candidate is valid.
    """

    def _dtype(self):
        # Device data format for fp16 (SEN169_FP16); read off a scratch STL so
        # the test does not hard-code the enum value.
        return SpyreTensorLayout([1, 1], torch.float16).device_dtype

    def _traced_scenario(self):
        """The exact (dep, unrepresentable STL, representable STL) triple from
        the Granite SDPA linear-projection failure.

        dep index ``4096*d0 + d2`` over ranges {d0:512, d1:4096, d2:4096}:
          * bad  STL -> stick expr ``floor(d2/128)`` (cross-stick, unrepresentable)
          * good STL -> stick expr ``d2`` (bare var, representable)
        """
        dev = self._dtype()
        d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True, nonnegative=True)
        dep = MemoryDep("buf", 4096 * d0 + d2, (d0, d1, d2), (512, 4096, 4096))
        bad = SpyreTensorLayout([512, 128, 1, 1, 64], [4096, 1, 8192, -1, 128], dev)
        good = SpyreTensorLayout([512, 1, 1, 64], [4096, -1, -1, 1], dev)
        return dep, bad, good

    def test_device_coordinates_raises_try_returns_none(self):
        dep, bad, good = self._traced_scenario()
        # The strict variant raises on the unrepresentable stick ...
        with self.assertRaises(Unsupported):
            device_coordinates(bad, dep, None)
        # ... while the non-raising variant returns None for it.
        self.assertIsNone(try_device_coordinates(bad, dep, None))
        # A representable candidate still returns coordinates from both.
        self.assertIsNotNone(try_device_coordinates(good, dep, None))
        d2 = sympy.Symbol("d2", integer=True, nonnegative=True)
        self.assertEqual(device_coordinates(good, dep, None)[-1].free_symbols, {d2})

    def test_reversed_dim_rejected(self):
        # prims.rev / Tensor.flip(0) on a (4, 64) tensor reads
        # x[64*(3 - p0) + p1], i.e. p0 carries a negative coefficient.  No
        # device coordinate can walk a dim backwards, and the term used to be
        # dropped silently, leaving coord=3 for every p0 (issue #3558).
        with self.assertRaisesRegex(Unsupported, "runs backwards"):
            compute_coordinates(
                [4, 64],
                [64, 1],
                {p0: 4, p1: 64},
                192 - 64 * p0 + p1,
            )

        # Same for a reversal of the innermost (stick) dim.
        with self.assertRaisesRegex(Unsupported, "runs backwards"):
            compute_coordinates(
                [4, 64],
                [64, 1],
                {p0: 4, p1: 64},
                64 * p0 - p1 + 63,
            )

        # The guard keys off the direction of travel, not the sign of ``step``:
        # an ascending term whose ``step`` is dragged negative by a constant
        # folded into it must still be accepted.
        cx = compute_coordinates(
            [4, 64],
            [64, 1],
            {p0: 4, p1: 64},
            64 * p0 + p1 - 5,
        )
        self.assertEqual(len(cx), 2)

        # The ordinary ascending access is untouched.
        cx = compute_coordinates(
            [4, 64],
            [64, 1],
            {p0: 4, p1: 64},
            64 * p0 + p1,
        )
        self.assertEqual(cx, [p0, p1])

    def test_check_supported_input_sticks_tolerates_mixed_list(self):
        # arg with one unrepresentable candidate and one valid one: the guard
        # must not raise (it previously aborted the whole compile).
        dep, bad, good = self._traced_scenario()
        arg = PropArg(dep, None, [bad, good])
        _check_supported_input_sticks([arg], "batchmatmul")  # must not raise

    def test_check_supported_input_sticks_all_unrepresentable(self):
        # When every candidate is unrepresentable the guard still does not
        # raise here (the hard failure comes later, from layout selection).
        dep, bad, _ = self._traced_scenario()
        arg = PropArg(dep, None, [bad])
        _check_supported_input_sticks([arg], "batchmatmul")  # must not raise


class TestMatmulPreferredLayout(TestCase):
    def test_matches_preferred_order_with_permuted_batch_dims(self):
        b0, b1, m, k = sympy.symbols("b0 b1 m k", integer=True, nonnegative=True)
        dep = MemoryDep(
            "x",
            4096 * b0 + 1024 * b1 + 128 * m + k,
            (b0, b1, m, k),
            (2, 4, 8, 128),
        )
        size = [2, 4, 8, 128]
        stride = [4096, 1024, 128, 1]

        preferred_a = SpyreTensorLayout(size, stride, torch.float16, [2, 0, 1, 3])
        preferred_b = SpyreTensorLayout(size, stride, torch.float16, [2, 1, 0, 3])
        not_preferred = SpyreTensorLayout(size, stride, torch.float16, [0, 1, 2, 3])
        order = MatmulPreferredOrder(frozenset({b0, b1}), k, m)

        self.assertTrue(_matches_preferred_matmul_device_order(preferred_a, dep, order))
        self.assertTrue(_matches_preferred_matmul_device_order(preferred_b, dep, order))
        self.assertFalse(
            _matches_preferred_matmul_device_order(not_preferred, dep, order)
        )

    def test_find_stick_compatible_input_layout_prefers_matmul_order(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 1024 * b + 128 * m + k, (b, m, k), (2, 8, 128))
        size = [2, 8, 128]
        stride = [1024, 128, 1]
        first_valid = SpyreTensorLayout(size, stride, torch.float16, [0, 1, 2])
        preferred = SpyreTensorLayout(size, stride, torch.float16, [1, 0, 2])
        arg = PropArg(dep, None, [first_valid, preferred])

        result = find_stick_compatible_input_layout(
            arg,
            k,
            "batchmatmul",
            "x",
            MatmulPreferredOrder(frozenset({b}), k, m),
        )

        self.assertEqual(result, preferred)

    def test_matches_preferred_order_for_2d_lhs(self):
        m, k = sympy.symbols("m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 128 * m + k, (m, k), (8, 128))
        stl = SpyreTensorLayout([8, 128], [128, 1], torch.float16, [0, 1])

        self.assertEqual(device_coordinates(stl, dep, None), [k // 64, m, k % 64])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset(), k, m)
            )
        )

    def test_matches_preferred_order_for_2d_rhs(self):
        k, n = sympy.symbols("k n", integer=True, nonnegative=True)
        dep = MemoryDep("y", 128 * k + n, (k, n), (8, 128))
        stl = SpyreTensorLayout([8, 128], [128, 1], torch.float16, [0, 1])

        self.assertEqual(device_coordinates(stl, dep, None), [n // 64, k, n % 64])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset(), n, k)
            )
        )

    def test_preferred_input_orders_with_y_broadcast_batch(self):
        b, m, k, n = sympy.symbols("b m k n", integer=True, nonnegative=True)
        x_dep = MemoryDep("x", 1024 * b + 128 * m + k, (b, m, k), (2, 8, 128))
        y_dep = MemoryDep("y", 64 * k + n, (k, n), (128, 64))

        x_order, y_order = _matmul_preferred_input_orders(
            x_dep, y_dep, [b, m, n], 2, k, n
        )

        self.assertEqual(x_order, MatmulPreferredOrder(frozenset({b}), k, m))
        self.assertEqual(y_order, MatmulPreferredOrder(frozenset(), n, k))

    def test_preferred_input_orders_with_x_broadcast_batch(self):
        b, m, k, n = sympy.symbols("b m k n", integer=True, nonnegative=True)
        x_dep = MemoryDep("x", 128 * m + k, (m, k), (8, 128))
        y_dep = MemoryDep("y", 8192 * b + 64 * k + n, (b, k, n), (2, 128, 64))

        x_order, y_order = _matmul_preferred_input_orders(
            x_dep, y_dep, [b, m, n], 2, k, n
        )

        self.assertEqual(x_order, MatmulPreferredOrder(frozenset(), k, m))
        self.assertEqual(y_order, MatmulPreferredOrder(frozenset({b}), n, k))

    def test_preferred_input_orders_keep_y_when_row_dim_is_folded(self):
        b, m, k, n = sympy.symbols("b m k n", integer=True, nonnegative=True)
        x_dep = MemoryDep("x", 128 * b + k, (b, m, k), (2, 1, 128))
        y_dep = MemoryDep("y", 64 * k + n, (k, n), (128, 64))

        x_order, y_order = _matmul_preferred_input_orders(
            x_dep, y_dep, [b, 0, n], 2, k, n
        )

        self.assertIsNone(x_order)
        self.assertEqual(y_order, MatmulPreferredOrder(frozenset(), n, k))

    def test_preferred_input_orders_project_folded_host_axes(self):
        b, m, k, n = sympy.symbols("b m k n", integer=True, nonnegative=True)
        y_dep = MemoryDep("y", 128 * k + n, (k, n), (128, 128))
        cases = [
            (
                MemoryDep("x", 128 * b + k, (b, m, k), (4, 1, 128)),
                [b, 0, k],
                [b, n],
                None,
            ),
            (
                MemoryDep("x", 128 * m + k, (b, m, k), (1, 8, 128)),
                [0, m, k],
                [m, n],
                MatmulPreferredOrder(frozenset(), k, m),
            ),
        ]

        for x_dep, x_host_coords, out_coords, expected_x_order in cases:
            x_order, y_order = _matmul_preferred_input_orders(
                x_dep,
                y_dep,
                out_coords,
                1,
                k,
                n,
                x_host_coords,
                [k, n],
            )

            self.assertEqual(x_order, expected_x_order)
            self.assertEqual(y_order, MatmulPreferredOrder(frozenset(), n, k))

    def test_default_matmul_input_layout_keeps_batch_before_stick_for_folded_row(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        layout = FixedLayout(
            torch.device("spyre"),
            torch.float16,
            [4, 1, 4096],
            [4096, 4096, 1],
        )
        dep = MemoryDep("x", 4096 * b + k, (b, m, k), (4, 1, 4096))
        arg = PropArg(dep, layout, [])

        stl = _default_matmul_input_layout(arg, k)

        self.assertIsNotNone(stl)
        device_coords = device_coordinates(stl, dep, None)
        sdsc_arg = TensorArg(
            True,
            0,
            DataFormats.SEN169_FP16,
            list(stl.device_size),
            device_coords,
            {"hbm": 0},
        )

        self.assertEqual(device_coords, [0, k // 64, b, k % 64])
        self.assertEqual(_get_device_dim_order(sdsc_arg, {})[0], [b, k])

    def test_preserve_shared_weight_unit_bmm_keeps_outer_stick_before_row(self):
        m, k, n = sympy.symbols("m k n", integer=True, nonnegative=True)
        x_arg = TensorArg(
            True,
            0,
            DataFormats.SEN169_FP16,
            [1, 64, 512, 64],
            [sympy.S.Zero, k // 64, m, k % 64],
            {"hbm": 0},
        )
        y_arg = TensorArg(
            True,
            1,
            DataFormats.SEN169_FP16,
            [64, 4096, 64],
            [n // 64, k, n % 64],
            {"hbm": 1},
        )
        out_arg = TensorArg(
            False,
            2,
            DataFormats.SEN169_FP16,
            [1, 64, 512, 64],
            [sympy.S.Zero, n // 64, m, n % 64],
            {"hbm": 2},
        )

        _preserve_shared_weight_unit_bmm_dim(
            "batchmatmul",
            {m: (512, 1), n: (4096, 1), k: (4096, 1)},
            [x_arg, y_arg, out_arg],
            {"shared_weight_unit_bmm": {"batch_dim": 0}},
        )

        unit = x_arg.device_coordinates[0]
        self.assertEqual(str(unit), "_spyre_bmm_unit")
        self.assertEqual(x_arg.device_coordinates, [unit, k // 64, m, k % 64])
        self.assertEqual(out_arg.device_coordinates, [unit, n // 64, m, n % 64])
        self.assertEqual(_get_device_dim_order(x_arg, {})[0], [m, k, unit])
        self.assertEqual(_get_device_dim_order(out_arg, {})[0], [m, n, unit])

    def test_preferred_order_allows_folded_batch_dim(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 128 * m + k, (b, m, k), (1, 8, 128))
        stl = SpyreTensorLayout([1, 8, 128], [1024, 128, 1], torch.float16, [1, 0, 2])

        self.assertEqual(device_coordinates(stl, dep, None), [0, k // 64, m, k % 64])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset(), k, m)
            )
        )

    def test_preferred_order_rejects_folded_row_dim(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 128 * b + k, (b, m, k), (2, 1, 128))
        stl = SpyreTensorLayout([2, 1, 128], [128, 128, 1], torch.float16, [1, 0, 2])

        self.assertFalse(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset({b}), k, m)
            )
        )

    def test_preferred_order_allows_folded_lhs_stick_outer_dim(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 512 * b + 64 * m + k, (b, m, k), (2, 8, 64))
        stl = SpyreTensorLayout([2, 8, 64], [512, 64, 1], torch.float16, [1, 0, 2])

        self.assertEqual(device_coordinates(stl, dep, None), [b, 0, m, k])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset({b}), k, m)
            )
        )

    def test_preferred_order_rejects_lhs_row_before_outer_stick(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        cases = [
            (
                (1, 8, 128),
                128 * m + k,
                [m, k // 64, 0, k % 64],
                frozenset(),
                [0, 1, 2],
            ),
            (
                (2, 8, 64),
                512 * b + 64 * m + k,
                [m, 0, b, k],
                frozenset({b}),
                [0, 1, 2],
            ),
        ]

        for size, index, expected_coords, batch_vars, dim_order in cases:
            stride = [size[1] * size[2], size[2], 1]
            dep = MemoryDep("x", index, (b, m, k), size)
            stl = SpyreTensorLayout(list(size), stride, torch.float16, dim_order)

            self.assertEqual(device_coordinates(stl, dep, None), expected_coords)
            self.assertFalse(
                _matches_preferred_matmul_device_order(
                    stl, dep, MatmulPreferredOrder(batch_vars, k, m)
                )
            )

    def test_preferred_order_allows_lhs_with_single_folded_dim(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        cases = [
            ((1, 8, 128), 128 * m + k, [0, k // 64, m, k % 64], frozenset()),
            ((2, 8, 64), 512 * b + 64 * m + k, [b, 0, m, k], frozenset({b})),
        ]

        for size, index, expected_coords, batch_vars in cases:
            stride = [size[1] * size[2], size[2], 1]
            dep = MemoryDep("x", index, (b, m, k), size)
            stl = SpyreTensorLayout(list(size), stride, torch.float16, [1, 0, 2])

            self.assertEqual(device_coordinates(stl, dep, None), expected_coords)
            self.assertTrue(
                _matches_preferred_matmul_device_order(
                    stl, dep, MatmulPreferredOrder(batch_vars, k, m)
                )
            )

    def test_preferred_order_rejects_folded_lhs_stick_outer_after_row(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 64 * m + k, (b, m, k), (1, 8, 64))
        stl = SpyreTensorLayout([1, 8, 64], [512, 64, 1], torch.float16, [0, 1, 2])

        self.assertEqual(device_coordinates(stl, dep, None), [m, 0, 0, k])
        self.assertFalse(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset(), k, m)
            )
        )

    def test_find_stick_compatible_input_layout_prefers_folded_lhs_with_batch_one(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 64 * m + k, (b, m, k), (1, 8, 64))
        first_valid = SpyreTensorLayout(
            [1, 8, 64], [512, 64, 1], torch.float16, [0, 1, 2]
        )
        preferred = SpyreTensorLayout(
            [1, 8, 64], [512, 64, 1], torch.float16, [1, 0, 2]
        )
        arg = PropArg(dep, None, [first_valid, preferred])

        result = find_stick_compatible_input_layout(
            arg,
            k,
            "batchmatmul",
            "x",
            MatmulPreferredOrder(frozenset(), k, m),
        )

        self.assertEqual(device_coordinates(preferred, dep, None), [0, 0, m, k])
        self.assertEqual(result, preferred)

    def test_preferred_order_allows_folded_rhs_stick_outer_dim(self):
        b, k, n = sympy.symbols("b k n", integer=True, nonnegative=True)
        dep = MemoryDep("y", 8192 * b + 64 * k + n, (b, k, n), (2, 128, 64))
        stl = SpyreTensorLayout([2, 128, 64], [8192, 64, 1], torch.float16, [1, 0, 2])

        self.assertEqual(device_coordinates(stl, dep, None), [b, 0, k, n])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset({b}), n, k)
            )
        )

    def test_preferred_order_rejects_folded_rhs_stick_outer_after_reduction(self):
        b, k, n = sympy.symbols("b k n", integer=True, nonnegative=True)
        dep = MemoryDep("y", 64 * k + n, (b, k, n), (1, 128, 64))
        stl = SpyreTensorLayout([1, 128, 64], [8192, 64, 1], torch.float16, [0, 1, 2])

        self.assertEqual(device_coordinates(stl, dep, None), [k, 0, 0, n])
        self.assertFalse(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset(), n, k)
            )
        )

    def test_preferred_order_allows_folded_rhs_stick_outer_before_reduction(self):
        b, k, n = sympy.symbols("b k n", integer=True, nonnegative=True)
        dep = MemoryDep("y", 64 * k + n, (b, k, n), (1, 128, 64))
        stl = SpyreTensorLayout([1, 128, 64], [8192, 64, 1], torch.float16, [1, 0, 2])

        self.assertEqual(device_coordinates(stl, dep, None), [0, 0, k, n])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset(), n, k)
            )
        )

    def test_preferred_order_selects_rhs_with_batch_one_and_outer_stick_gt_one(self):
        b, k, n = sympy.symbols("b k n", integer=True, nonnegative=True)
        order = MatmulPreferredOrder(frozenset(), n, k)
        dep = MemoryDep("y", 128 * k + n, (b, k, n), (1, 128, 128))
        first_valid = SpyreTensorLayout(
            [1, 128, 128], [16384, 128, 1], torch.float16, [0, 1, 2]
        )
        preferred = SpyreTensorLayout(
            [1, 128, 128], [16384, 128, 1], torch.float16, [1, 0, 2]
        )
        preferred_coords = device_coordinates(preferred, dep, None)
        arg = PropArg(dep, None, [first_valid, preferred])

        result = find_stick_compatible_input_layout(
            arg,
            n,
            "batchmatmul",
            "y",
            order,
        )
        sdsc_arg = TensorArg(
            True,
            1,
            DataFormats.SEN169_FP16,
            list(preferred.device_size),
            preferred_coords,
            {"hbm": 1},
        )

        self.assertEqual(preferred_coords, [0, n // 64, k, n % 64])
        self.assertTrue(_matches_preferred_matmul_device_order(preferred, dep, order))
        self.assertEqual(_get_device_dim_order(sdsc_arg, {})[0], [k, n])
        self.assertEqual(result, preferred)

    def test_preferred_matmul_output_dim_order(self):
        self.assertEqual(_preferred_matmul_output_dim_order(2, 1), [0, 1])
        self.assertEqual(_preferred_matmul_output_dim_order(3, 2), [1, 0, 2])
        self.assertEqual(
            _preferred_matmul_output_dim_order(3, 2, [1, 8, 128]), [1, 0, 2]
        )
        self.assertEqual(
            _preferred_matmul_output_dim_order(3, 2, [2, 1, 128]), [0, 1, 2]
        )
        self.assertEqual(
            _preferred_matmul_output_dim_order(3, 2, [2, 8, 128]), [1, 0, 2]
        )
        self.assertEqual(_preferred_matmul_output_dim_order(4, 3), [2, 0, 1, 3])
        self.assertEqual(_preferred_matmul_output_dim_order(4, 2), [3, 0, 1, 2])
        self.assertEqual(
            _preferred_matmul_output_dim_order(4, 3, [1, 4, 8, 128]),
            [2, 0, 1, 3],
        )

    def test_preferred_output_dim_order_produces_2d_stick_row_device_order(self):
        m, n = sympy.symbols("m n", integer=True, nonnegative=True)
        dep = MemoryDep("out", 128 * m + n, (m, n), (8, 128))
        stl = SpyreTensorLayout(
            [8, 128],
            [128, 1],
            torch.float16,
            _preferred_matmul_output_dim_order(2, 1),
        )

        self.assertEqual(device_coordinates(stl, dep, None), [n // 64, m, n % 64])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset(), n, m)
            )
        )

    def test_preferred_output_dim_order_produces_batch_stick_row_device_order(self):
        b, m, n = sympy.symbols("b m n", integer=True, nonnegative=True)
        dep = MemoryDep("out", 512 * b + 64 * m + n, (b, m, n), (2, 8, 64))
        stl = SpyreTensorLayout(
            [2, 8, 64],
            [512, 64, 1],
            torch.float16,
            _preferred_matmul_output_dim_order(3, 2, [2, 8, 64]),
        )

        self.assertEqual(device_coordinates(stl, dep, None), [b, 0, m, n])
        self.assertTrue(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset({b}), n, m)
            )
        )

    def test_preferred_matmul_output_layout_with_size_one_dims(self):
        b, m, n = sympy.symbols("b m n", integer=True, nonnegative=True)
        cases = [
            ((2, 1, 128), 128 * b + n, [0, n // 64, b, n % 64]),
            ((1, 8, 64), 64 * m + n, [0, 0, m, n]),
            ((1, 8, 128), 128 * m + n, [0, n // 64, m, n % 64]),
            ((2, 8, 64), 512 * b + 64 * m + n, [b, 0, m, n]),
            ((2, 8, 128), 1024 * b + 128 * m + n, [b, n // 64, m, n % 64]),
            ((2, 8, 1), 8 * b + m, [b, 0, m, 0]),
            ((1, 1, 128), n, [0, n // 64, 0, n % 64]),
            ((2, 1, 1), b, [0, 0, b, 0]),
        ]

        for size, index, expected_coords in cases:
            stride = [size[1] * size[2], size[2], 1]
            stl = SpyreTensorLayout(
                list(size),
                stride,
                torch.float16,
                _preferred_matmul_output_dim_order(3, 2, list(size)),
            )
            dep = MemoryDep("out", index, (b, m, n), size)

            self.assertEqual(device_coordinates(stl, dep, None), expected_coords)

    def test_preferred_x_layout_with_k_size_one_has_no_symbolic_stick(self):
        b, m, k = sympy.symbols("b m k", integer=True, nonnegative=True)
        dep = MemoryDep("x", 8 * b + m, (b, m, k), (2, 8, 1))
        stl = SpyreTensorLayout([2, 8, 1], [8, 1, 1], torch.float16, [1, 0, 2])

        self.assertEqual(device_coordinates(stl, dep, None), [b, 0, m, 0])
        self.assertFalse(
            _matches_preferred_matmul_device_order(
                stl, dep, MatmulPreferredOrder(frozenset({b}), k, m)
            )
        )


class TestTilingExprToDeviceExpr(TestCase):
    def test_tiling_expr_row_major(self):
        # [1024, 4096] tensor tiled 2x4 times (generic stick format)
        index = 4096 * 512 * p0 + 1024 * p1
        result = tiling_expr_to_device_expr([64, 1024, 64], [64, 4096, 1], index)
        self.assertEqual(result, 32768 * p0 + 1048576 * p1)

    def test_tiling_expr_column_major(self):
        # [4096, 1024] tensor tiled 4x2 times (generic stick format) transposed before use
        index = 512 * p0 + 1024 * 1024 * p1
        result = tiling_expr_to_device_expr([16, 4096, 64], [64, 1024, 1], index)
        self.assertEqual(result, 2097152 * p0 + 65536 * p1)

    def test_tiling_expr_row_major_transposed_restickified(self):
        # [1024, 4096] tensor tiled 2x4 times (generic stick format) transposed
        # and restickified before use
        index = 512 * p0 + 1024 * 1024 * p1
        result = tiling_expr_to_device_expr([64, 1024, 64], [65536, 1, 1024], index)
        self.assertEqual(result, 32768 * p0 + 1048576 * p1)

    def test_tiling_expr_bare_symbol_degenerate_substitution(self):
        # index == p0 with coefficient 1 and no other additive term: sympy
        # auto-simplifies Mul(1, p0) to the bare Symbol p0, so
        # index.xreplace({p0: 1}) returns a raw Python int 1 (not
        # sympy.Integer(1)) rather than the usual sympy numeric type -- the
        # degenerate case that used to make the function's second .xreplace
        # call crash with AttributeError: 'int' object has no attribute
        # 'xreplace'. This mirrors the real _general_tile_advance call shape
        # when a tiled dim's extent is 1 and no other term survives
        # substitution (see tests/inductor/test_coarse_tile_e2e.py's
        # test_hint_nested_loop_with_scratchpad).
        index = p0
        result = tiling_expr_to_device_expr([64, 1024, 64], [64, 4096, 1], index)
        self.assertEqual(result, p0)


if __name__ == "__main__":
    run_tests()
