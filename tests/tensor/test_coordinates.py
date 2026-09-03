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

from types import SimpleNamespace

import sympy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import FixedLayout
from torch._inductor.virtualized import V
from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    SpyreTensorLayout,
    get_device_dtype,
)
from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_OP,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import (
    device_coordinates,
    try_device_coordinates,
)
from torch_spyre._inductor.propagate_layouts import (
    PropArg,
    _check_supported_input_sticks,
    _find_alt_target_stl,
    find_stick_compatible_input_layout,
)
from torch_spyre._inductor.views import (
    _decompose_constant_offset,
    align_tensors,
    compute_coordinates,
    normalize_coordinates,
    tiling_expr_to_device_expr,
)
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

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

    def test_compute_coordinates_mixed_radix_flattened_dim(self):
        """A flattened H*D loop maps back to separate H and D coordinates."""
        lq, hd = sympy.symbols("lq hd", integer=True, nonnegative=True)
        with V.set_graph_handler(SimpleNamespace()):
            repeat_info: dict = {}
            cx = compute_coordinates(
                [1, 32, 64, 128],
                [262144, 8192, 128, 1],
                {lq: 64, hd: 4096},
                128 * lq
                + 8192 * ModularIndexing(hd, 128, 32)
                + ModularIndexing(hd, 1, 128),
                repeat_info_out=repeat_info,
            )
            self.assertEqual(
                cx,
                [0, sympy.Mod(FloorDiv(hd, 128), 32), lq, sympy.Mod(hd, 128)],
            )

            terms = normalize_coordinates(
                {lq: 64, hd: 4096},
                [1, 32, 64, 128],
                cx,
                lambda: sympy.Symbol("z0"),
            )
            high_digit = next(
                term for term in terms if term.var == hd and term.dim_size == 32
            )
            self.assertEqual(high_digit.den, 128)
            self.assertEqual(high_digit.mod, 4096)

            iteration_space, tensors, remap = align_tensors(
                {lq: (64, 1), hd: (4096, 1)},
                [{"size": [1, 32, 64, 128], "coordinates": cx}],
                repeat_info=repeat_info,
            )
            self.assertEqual(iteration_space[hd][0], 128)
            self.assertEqual(remap[hd][0], (hd, 1))
            high_var = remap[hd][1][0]
            self.assertEqual(iteration_space[high_var][0], 32)
            self.assertTrue(all(size > 0 for size in tensors[0]["size"]))

    def test_compute_coordinates_rejects_overlapping_moduli(self):
        """Multiple Mods remain unsupported unless they form one digit chain."""
        with self.assertRaisesRegex(Unsupported, "multiple Mod"):
            compute_coordinates(
                [4, 6],
                [6, 1],
                {p0: 24},
                6 * (p0 % 4) + p0 % 6,
            )

    def test_compute_coordinates_rejects_fractional_mixed_radix_coefficient(self):
        """An unsupported digit scale is rejected before normalization."""
        with self.assertRaisesRegex(Unsupported, "multiple Mod"):
            compute_coordinates(
                [4, 6],
                [6, 1],
                {p0: 24},
                sympy.Rational(3, 2) * (p0 % 4) + 6 * sympy.Mod(FloorDiv(p0, 4), 6),
            )

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

    def test_offset_across_padded_row_stays_stick_offset_free(self):
        # Regression: a non-stick offset on a padded row (row width not a
        # multiple of elem_in_stick) must not leak a residual onto the stick
        # coordinate. See _decompose_constant_offset.
        cases = [
            (
                "single_row",  # base=(4,100) fp16 [1:, :], offset=100 == 1 row
                [2, 4, 64],
                [64, 100, 1],
                {p0: 3, p1: 100},
                100 + 100 * p0 + p1,
                [p1 // 64, p0 + 1, p1 % 64],
            ),
            (
                "multi_row",  # base=(5,100)[2:, :], offset=200 == 2 rows
                [2, 5, 64],
                [64, 100, 1],
                {p0: 3, p1: 100},
                200 + 100 * p0 + p1,
                [p1 // 64, p0 + 2, p1 % 64],
            ),
            (
                "wider_padding",  # base=(4,130) pads to 192 (3 sticks), [1:, :]
                [3, 4, 64],
                [64, 130, 1],
                {p0: 3, p1: 130},
                130 + 130 * p0 + p1,
                [p1 // 64, p0 + 1, p1 % 64],
            ),
            (
                "multi_dim",  # base=(3,3,100)[1:, :, :], offset=300 == 1 block
                [2, 3, 3, 64],
                [64, 300, 100, 1],
                {p0: 2, p1: 3, p2: 100},
                300 + 300 * p0 + 100 * p1 + p2,
                [p2 // 64, p0 + 1, p1, p2 % 64],
            ),
            (
                "middle_dim",  # base=(3,5,100)[:, 2:, :], offset=200 == 2 rows
                [2, 3, 5, 64],
                [64, 500, 100, 1],
                {p0: 3, p1: 3, p2: 100},
                200 + 500 * p0 + 100 * p1 + p2,
                [p2 // 64, p0, p1 + 2, p2 % 64],
            ),
        ]
        for label, size, stride, var_ranges, index, expected in cases:
            with self.subTest(label):
                cx = compute_coordinates(size, stride, var_ranges, index)
                self.assertEqual(cx, expected)

    def test_decompose_constant_offset_unpeelable_falls_back(self):
        # remaining != 0 after peeling every dim -> return False, untouched.
        coordinates = [sympy.S.Zero, sympy.S.Zero]
        handled = _decompose_constant_offset(
            sympy.Integer(1), [10, 10], [200, 2], coordinates
        )
        self.assertFalse(handled)
        self.assertEqual(coordinates, [sympy.S.Zero, sympy.S.Zero])

    def test_decompose_constant_offset_rejects_symbolic_offset(self):
        # A genuinely symbolic offset can't be compared against a concrete
        # stride, so this raises rather than silently mis-peeling -- which is
        # why compute_coordinates guards this call with `not offset.free_symbols`.
        s0 = sympy.Symbol("s0", integer=True, nonnegative=True)
        with self.assertRaises(TypeError):
            _decompose_constant_offset(
                s0, [10, 10], [200, 2], [sympy.S.Zero, sympy.S.Zero]
            )


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


class TestFactorizedMatmulCandidates(TestCase):
    def _scenario(self, layouts):
        lq, generated, contraction = sympy.symbols(
            "lq generated contraction", integer=True, nonnegative=True
        )
        dep = MemoryDep(
            "x",
            4096 * lq + contraction,
            (lq, generated, contraction),
            (8, 4096, 4096),
        )
        host = FixedLayout(
            torch.device("cpu"),
            torch.float16,
            [1, 8, 4096],
            [32768, 4096, 1],
        )
        return PropArg(dep, host, layouts), contraction

    def _same_graph_attention_scenario(
        self,
        host_stride=(262144, 4096, 128, 1),
        contraction_range=4096,
    ):
        """SDPA's BLHD producer viewed as BL(H*D) by a fused o_proj."""
        lq, generated, contraction = sympy.symbols(
            "lq generated contraction", integer=True, nonnegative=True
        )
        dep = MemoryDep(
            "x",
            4096 * lq + contraction,
            (lq, generated, contraction),
            (64, 4096, contraction_range),
        )
        host_size = [1, 64, 32, 128]
        host = FixedLayout(
            torch.device("cpu"),
            torch.float16,
            host_size,
            list(host_stride),
        )
        if tuple(host_stride) == (262144, 4096, 128, 1):
            source = SpyreTensorLayout(
                [64, 2, 32, 64],
                [4096, 64, 128, 1],
                get_device_dtype(torch.float16),
            )
        else:
            source = SpyreTensorLayout(
                host_size,
                list(host_stride),
                torch.float16,
                [0, 1, 2, 3],
            )
        return PropArg(dep, host, [source]), contraction, source

    def test_canonicalization_is_independent_of_candidate_order(self):
        """Canonical layout is returned regardless of candidate list order."""
        dtype = get_device_dtype(torch.float16)
        factorized = SpyreTensorLayout([8, 2, 32, 64], [4096, 64, 128, 1], dtype)
        canonical = SpyreTensorLayout(
            [1, 8, 4096], [32768, 4096, 1], torch.float16, [0, 1, 2]
        )

        for layouts in ([factorized, canonical], [canonical, factorized]):
            with self.subTest(first=layouts[0]):
                arg, contraction = self._scenario(layouts)
                result = find_stick_compatible_input_layout(
                    arg, contraction, BATCH_MATMUL_OP, "x"
                )
                self.assertEqual(result, canonical)

    def test_canonicalizes_same_graph_attention_flatten(self):
        """Contiguous H,D producer dims become one o_proj contraction dim."""
        arg, contraction, source = self._same_graph_attention_scenario()
        expected = SpyreTensorLayout(
            [1, 64, 4096],
            [262144, 4096, 1],
            torch.float16,
            [0, 1, 2],
        )

        result = find_stick_compatible_input_layout(
            arg, contraction, BATCH_MATMUL_OP, "x"
        )

        self.assertEqual(result, expected)
        with V.set_graph_handler(SimpleNamespace()):
            self.assertEqual(
                device_coordinates(result, arg.dep, None),
                [
                    arg.dep.var_names[0],
                    sympy.floor(contraction / 64),
                    0,
                    sympy.Mod(contraction, 64),
                ],
            )

    def test_rejects_noncontiguous_or_partial_factorized_chain(self):
        """Only a full, gap-free mixed-radix view is safe to collapse."""
        scenarios = (
            self._same_graph_attention_scenario(host_stride=(262144, 4096, 256, 1)),
            self._same_graph_attention_scenario(contraction_range=2048),
        )
        for arg, contraction, source in scenarios:
            with self.subTest(
                stride=arg.layout.stride,
                contraction_range=arg.dep.ranges[contraction],
            ):
                with self.assertRaisesRegex(Unsupported, "full-range contiguous"):
                    find_stick_compatible_input_layout(
                        arg, contraction, BATCH_MATMUL_OP, "x"
                    )

    def test_nonstandard_matmul_layout_is_not_canonicalized(self):
        """Non-STANDARD formats are returned unchanged by Pass 3."""
        dtype = get_device_dtype(torch.float16)
        qfp8wt = SpyreTensorLayout(
            [8, 2, 32, 64],
            [4096, 64, 128, 1],
            dtype,
            ElementArrangement.QFP8WT,
        )
        arg, contraction = self._scenario([qfp8wt])
        result = find_stick_compatible_input_layout(
            arg, contraction, BATCH_MATMUL_FP8_OP, "x"
        )
        self.assertEqual(result, qfp8wt)


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


class TestFindAltTargetStlBoolStickSize(TestCase):
    """_find_alt_target_stl must size a bool mutation target's stick from its
    real physical format (target_stl.device_dtype), not target_layout.dtype's
    hardcoded SEN169_FP16 assumption -- a bool held in IEEE_FP32 has a 32-elem
    stick, not 64. Both cases below write host_size [64, 128] at column
    offset 32 (a ``mask[:, 32:64].copy_(upd)``-style mutation): offset 32 is a
    whole stick for IEEE_FP32 (32) but not for SEN169_FP16 (64), so the same
    logical write must be treated differently depending on physical format.

    This is a pure layout-resolution test: it calls _find_alt_target_stl
    directly with hand-built layout objects, so it never reaches torch.compile
    or the hardware compiler. That matters because an actual compiled
    mutation into an IEEE_FP32-backed bool currently fails end-to-end on two
    unrelated, lower-level gaps (ReStickifyOpHBM rejects IEEE_FP32 outright --
    see test_restickify_fp32_unsupported_xfail in test_inductor_ops.py -- and
    separately the DL op scheduler finds no candidate for a fused copy/slice
    into IEEE_FP32). Neither gap is specific to this stick-size computation,
    so this test isolates the one thing this fix actually changes.
    """

    def _write_dep(self):
        # mask[:, 32:64].copy_(upd) over a [64, 128] host tensor: offset 32
        # into the row-major index 128*d0 + d1.
        d0, d1 = sympy.symbols("d0 d1", integer=True, nonnegative=True)
        return MemoryDep("mask_buf", 128 * d0 + d1 + 32, (d0, d1), (64, 32))

    def test_fp32_backed_bool_offset_is_stick_aligned(self):
        # Pre-fix, get_elem_in_stick(target_layout.dtype) would use bool's
        # hardcoded SEN169_FP16 stick (64) here regardless of target_stl,
        # wrongly conclude offset 32 is not stick-aligned, and search for an
        # alt layout. The fix resolves the real IEEE_FP32 stick (32), under
        # which offset 32 is already aligned, so no alt is needed.
        target_layout = FixedLayout(
            torch.device("cpu"), torch.bool, [64, 128], [128, 1]
        )
        target_stl = SpyreTensorLayout([64, 128], torch.float32)
        self.assertEqual(target_stl.device_dtype, DataFormats.IEEE_FP32)
        self.assertIsNone(
            _find_alt_target_stl(target_layout, target_stl, self._write_dep())
        )

    def test_fp16_backed_bool_offset_needs_alt(self):
        # Contrast case: for a bool actually backed by SEN169_FP16, stick=64
        # is correct, and offset 32 genuinely is not stick-aligned -- an alt
        # stick dim is required, same as the pre-fix code would have found.
        target_layout = FixedLayout(
            torch.device("cpu"), torch.bool, [64, 128], [128, 1]
        )
        target_stl = SpyreTensorLayout([64, 128], torch.float16)
        self.assertEqual(target_stl.device_dtype, DataFormats.SEN169_FP16)
        self.assertIsNotNone(
            _find_alt_target_stl(target_layout, target_stl, self._write_dep())
        )


class TestNormalizeCoordinatesFusion(TestCase):
    """``normalize_coordinates``' contiguous-device-dim fusion.

    The fusion loop is a single-pass adjacent-pair scan, so an inert placeholder
    term -- a size-1 device dim with a constant-zero coordinate -- used to break a
    fusion run even though the emitted layout discards it anyway. Leaving the run
    broken splits one logical axis across two device dims, and a matmul reading
    such a layout ends up contracting two axes, which the backend cannot schedule
    (deeptools ``getMinParamBmm``'s ``out_reuse_dim`` DT_CHECK).
    """

    def _normalize(self, var_ranges, size, coordinates):
        counter = [0]

        def synthetic_var():
            counter[0] += 1
            return sympy.Symbol(f"z{counter[0] - 1}")

        return normalize_coordinates(dict(var_ranges), size, coordinates, synthetic_var)

    def _addr(self, terms):
        """Flat device address encoded by a dense term list (last term = stick)."""
        stride = sympy.Integer(1)
        addr = sympy.S.Zero
        for term in reversed(terms):
            if term.var is None:
                coord = term.offset
            else:
                coord = (
                    term.num * sympy.floor(sympy.Mod(term.var, term.mod) / term.den)
                    + term.offset
                )
            addr += stride * coord
            stride *= term.dim_size
        return addr

    def test_placeholder_does_not_block_fusion(self):
        """``[B=1, H=16, seq=1, D=128]`` SDPA output read as one flat 2048 axis.

        ``get_generic_stick_layout``'s rank-4 map puts the squeezed ``seq`` dim
        between ``H`` and the non-stick half of ``D``, and the squeezed ``B`` dim
        just before the stick. ``H`` and ``D``'s outer half must still fuse into a
        single 32-wide dim, so the matmul consuming this buffer contracts exactly
        one axis.
        """
        k = sympy.Symbol("c1")
        terms = self._normalize(
            {k: 2048},
            [16, 1, 2, 1, 64],
            [
                sympy.floor(k / 128),
                sympy.S.Zero,
                sympy.floor(sympy.Mod(k, 128) / 64),
                sympy.S.Zero,
                sympy.Mod(k, 64),
            ],
        )
        self.assertEqual([int(t.dim_size) for t in terms], [32, 64])
        # ... and the fused dim addresses exactly what the two dims did.
        addr = self._addr(terms)
        for val in range(2048):
            self.assertEqual(int(addr.subs({k: val})), val)

    def test_fusion_declined_when_outer_term_has_offset(self):
        """An offset on the outer term counts in units of that term's ``den``.

        Fusing shrinks ``den``, which would silently rescale the offset, so the
        fusion must not happen across a placeholder in that case.
        """
        k = sympy.Symbol("c1")
        terms = self._normalize(
            {k: 1024},
            [16, 1, 2, 1, 64],
            [
                4 + sympy.floor(k / 128),
                sympy.S.Zero,
                sympy.floor(sympy.Mod(k, 128) / 64),
                sympy.S.Zero,
                sympy.Mod(k, 64),
            ],
        )
        self.assertEqual([int(t.dim_size) for t in terms], [16, 2, 64])
        addr = self._addr(terms)
        for val in (0, 1, 63, 64, 127, 128, 1023):
            self.assertEqual(int(addr.subs({k: val})), 512 + val)

    def test_fusion_declined_when_pair_is_not_dense(self):
        """A gap between the two dims (3*64 < 256) makes the fusion inexact."""
        k = sympy.Symbol("c1")
        terms = self._normalize(
            {k: 1536},
            [8, 1, 3, 64],
            [
                sympy.floor(k / 256),
                sympy.S.Zero,
                sympy.floor(sympy.Mod(k, 256) / 64),
                sympy.Mod(k, 64),
            ],
        )
        self.assertEqual([int(t.dim_size) for t in terms], [8, 3, 64])

    def test_adjacent_fusion_unchanged(self):
        """No placeholder: the historical predicate is untouched."""
        k = sympy.Symbol("c1")
        terms = self._normalize(
            {k: 2048},
            [16, 2, 64],
            [
                sympy.floor(k / 128),
                sympy.floor(sympy.Mod(k, 128) / 64),
                sympy.Mod(k, 64),
            ],
        )
        self.assertEqual([int(t.dim_size) for t in terms], [32, 64])


if __name__ == "__main__":
    run_tests()
