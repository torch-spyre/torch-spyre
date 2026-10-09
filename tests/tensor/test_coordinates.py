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
from unittest import mock

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
    compute_restickify_needed,
    device_coordinates,
    try_device_coordinates,
)
from torch_spyre._inductor.nonstick_dim_order import _matmul_reorder_stl
from torch_spyre._inductor.propagate_layouts import (
    PropArg,
    _check_supported_input_sticks,
    _compute_dim_order,
    _find_alt_target_stl,
    _flat_dense_projection_x_layout,
    find_stick_compatible_input_layout,
)
from torch_spyre._inductor.views import (
    _decompose_constant_offset,
    align_tensors,
    compute_coordinates,
    normalize_coordinates,
    tiling_expr_to_device_expr,
    UnalignedStickSplit,
)
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

p0, p1, p2, p3, p4, p5 = sympy.symbols("p0 p1 p2 p3 p4 p5", integer=True)

# Patched to True so compute_restickify_needed applies its matmul-edge rule
# (a factorized input must restickify) without building a matmul op.
_IS_MATMUL_OP = "torch_spyre._inductor.pass_utils._is_matmul_op"


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

    def test_align_tensors_rejects_internal_boundary_inside_stick(self):
        """A peer's factorization must not split an int32 physical stick."""
        entry, head, width = sympy.symbols(
            "entry head width", integer=True, nonnegative=True
        )
        indirect = sympy.Symbol("indirect", integer=True, nonnegative=True)
        tensors = [
            {
                "size": [2, 32],
                "coordinates": [
                    sympy.floor(entry / 32),
                    sympy.Mod(entry, 32),
                ],
            },
            {
                "size": [128, 8, 4, 1, 64],
                "coordinates": [
                    head + 8 * sympy.Mod(entry, 16),
                    sympy.floor(entry / 16),
                    sympy.floor(width / 64),
                    sympy.S.Zero,
                    sympy.Mod(width, 64),
                ],
            },
            {
                "size": [128, 8, 4, 1, 64],
                "coordinates": [
                    indirect,
                    head,
                    sympy.floor(width / 64),
                    sympy.S.Zero,
                    sympy.Mod(width, 64),
                ],
            },
        ]

        with self.assertRaisesRegex(
            UnalignedStickSplit,
            r"boundary 16 for entry cuts tensor 0.*stick of 32",
        ):
            align_tensors(
                {entry: (64, 2), head: (8, 1), width: (256, 1)},
                tensors,
                {indirect: 128},
            )

    def test_align_tensors_rejects_nonzero_truncated_stick_split(self):
        """Reject a sub-stick split even when integer division is nonzero."""
        entry, head, width = sympy.symbols(
            "entry head width", integer=True, nonnegative=True
        )
        tensors = [
            {
                "size": [3, 32],
                "coordinates": [
                    sympy.floor(entry / 32),
                    sympy.Mod(entry, 32),
                ],
            },
            {
                "size": [384, 2, 4, 1, 64],
                "coordinates": [
                    head + 8 * sympy.Mod(entry, 48),
                    sympy.floor(entry / 48),
                    sympy.floor(width / 64),
                    sympy.S.Zero,
                    sympy.Mod(width, 64),
                ],
            },
        ]

        with self.assertRaisesRegex(
            UnalignedStickSplit,
            r"boundary 48 for entry cuts tensor 0.*stick of 32",
        ):
            align_tensors(
                {entry: (96, 3), head: (8, 1), width: (256, 1)},
                tensors,
            )

    def test_align_tensors_accepts_noncanonical_stick_variable_split(self):
        """A shared variable need not describe canonical outer-stick tiling."""
        entry = sympy.Symbol("entry", integer=True, nonnegative=True)
        _, aligned, _ = align_tensors(
            {entry: (4, 2)},
            [
                {
                    "size": [2, 64],
                    "coordinates": [sympy.floor(entry / 2), entry],
                }
            ],
        )

        self.assertTrue(all(size > 0 for tensor in aligned for size in tensor["size"]))

    def test_align_tensors_accepts_stick_aligned_source(self):
        """A dense scatter source contributes no sub-stick index boundary."""
        entry, head, width = sympy.symbols(
            "entry head width", integer=True, nonnegative=True
        )
        tensors = [
            {
                "size": [2, 32],
                "coordinates": [
                    sympy.floor(entry / 32),
                    sympy.Mod(entry, 32),
                ],
            },
            {
                "size": [64, 8, 4, 1, 64],
                "coordinates": [
                    entry,
                    head,
                    sympy.floor(width / 64),
                    sympy.S.Zero,
                    sympy.Mod(width, 64),
                ],
            },
        ]

        _, aligned, _ = align_tensors(
            {entry: (64, 2), head: (8, 1), width: (256, 1)}, tensors
        )

        self.assertTrue(all(size > 0 for tensor in aligned for size in tensor["size"]))

    def test_align_tensors_accepts_partial_stick_endpoint(self):
        """A short final stick is legal because its endpoint is not a split."""
        entry = sympy.Symbol("entry", integer=True, nonnegative=True)
        _, aligned, _ = align_tensors(
            {entry: (7, 1)},
            [
                {
                    "size": [1, 32],
                    "coordinates": [
                        sympy.floor(entry / 32),
                        sympy.Mod(entry, 32),
                    ],
                },
                {
                    "size": [7, 64],
                    "coordinates": [entry, sympy.S.Zero],
                },
            ],
        )

        self.assertTrue(all(size > 0 for tensor in aligned for size in tensor["size"]))

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


class TestAlignTensorsStridedTerm(TestCase):
    """A coordinate that steps over a device dim (``2*d1``) must keep the dim's
    footprint exact (issue #4050).

    ``h.view(B, L, H, D)[..., :64]`` on a ``[B, L, H*D]`` input reads the
    stick-tile dim as ``2*d1``: one head is two sticks and the slice keeps one.
    """

    def _tensors(self, stick_coord):
        d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True, nonnegative=True)
        it_space = {d0: (sympy.Integer(8), 1), d1: (sympy.Integer(4), 1), d2: (64, 1)}
        zero = sympy.S.Zero
        inp = {"size": [8, 8, 1, 64], "coordinates": [d0, stick_coord(d1), zero, d2]}
        out = {"size": [4, 1, 1, 8, 64], "coordinates": [d1, zero, zero, d0, d2]}
        return (d0, d1, d2), it_space, inp, out

    def test_stride_realized_as_gap_keeps_footprint(self):
        (d0, d1, d2), it_space, inp, out = self._tensors(lambda d1: 2 * d1)
        _, tensors, _ = align_tensors(it_space, [inp, out])
        aligned_in = tensors[0]
        # 8 rows x 8 sticks x 64: a stride-2 walk over 4 heads must not inflate
        # the stick-tile dim to 8 iterations *and* add a gap of 2 (it did, and
        # doubled the row stride so output row l read input row 2*l).
        self.assertEqual(sympy.prod(aligned_in["size"]), 8 * 8 * 64)
        sizes = [int(s) for s in aligned_in["size"]]
        gap_idx = sizes.index(2)
        self.assertEqual(sizes[gap_idx - 1], 4, "stick-tile dim counted in iterations")
        self.assertEqual(aligned_in["coordinates"][gap_idx - 1], d1)
        self.assertEqual(aligned_in["coordinates"][gap_idx], 0)

    def test_stride_with_residual_offset_selects_gap_position(self):
        """``2*d1 + 1`` (e.g. ``[..., 64:]`` or ``[:, 1::2]``) keeps the +1.

        The residual is below the stride, so it cannot live on the variable's
        own coordinate; it hangs off a synthetic size-1 variable in the gap dim,
        the same way an elided dim with an offset is restored.
        """
        (d0, d1, d2), it_space, inp, out = self._tensors(lambda d1: 2 * d1 + 1)
        new_it_space, tensors, _ = align_tensors(it_space, [inp, out])
        aligned_in = tensors[0]
        self.assertEqual(sympy.prod(aligned_in["size"]), 8 * 8 * 64)
        sizes = [int(s) for s in aligned_in["size"]]
        gap_coord = aligned_in["coordinates"][sizes.index(2)]
        synthetic = gap_coord.free_symbols
        self.assertEqual(len(synthetic), 1)
        z = next(iter(synthetic))
        self.assertEqual(gap_coord, z + 1)
        self.assertEqual(new_it_space[z][0], 1)
        # The other tensor gains a size-1 dim for the synthetic variable.
        self.assertIn(z, {s for c in tensors[1]["coordinates"] for s in c.free_symbols})

    def test_stride_not_dividing_dim_is_rejected(self):
        d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True, nonnegative=True)
        it_space = {d0: (8, 1), d1: (3, 1), d2: (64, 1)}
        zero = sympy.S.Zero
        inp = {"size": [8, 7, 1, 64], "coordinates": [d0, 2 * d1, zero, d2]}
        out = {"size": [3, 1, 1, 8, 64], "coordinates": [d1, zero, zero, d0, d2]}
        with self.assertRaisesRegex(Unsupported, "does not divide"):
            align_tensors(it_space, [inp, out])


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
            [1, 8, 4096], [32768, 4096, 1], torch.float16, [1, 0, 2]
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
            [1, 0, 2],
        )

        result = find_stick_compatible_input_layout(
            arg, contraction, BATCH_MATMUL_OP, "x"
        )

        self.assertEqual(result, expected)
        with V.set_graph_handler(SimpleNamespace()):
            self.assertEqual(
                device_coordinates(result, arg.dep, None),
                [
                    0,
                    sympy.floor(contraction / 64),
                    arg.dep.var_names[0],
                    sympy.Mod(contraction, 64),
                ],
            )

    def _attention_output_scenario(self, rows, heads=8, head_dim=128):
        """The o_proj input in Granite's rank-local attention: SDPA's output,
        kept by the attention kernel as [H, D/64, L, 64] and read by o_proj as
        the [L, H*D] matrix.  ``rows`` = 1 is a decode step."""
        m, generated, contraction = sympy.symbols(
            "m generated contraction", integer=True, nonnegative=True
        )
        K = heads * head_dim
        if rows == 1:
            # M = 1 folds the row variable away.
            dep = MemoryDep("x", contraction, (generated, contraction), (4096, K))
        else:
            dep = MemoryDep(
                "x",
                K * m + contraction,
                (m, generated, contraction),
                (rows, 4096, K),
            )
        host = FixedLayout(
            torch.device("cpu"),
            torch.float16,
            [1, rows, heads, head_dim],
            [rows * K, K, head_dim, 1],
        )
        source = SpyreTensorLayout(
            [heads, head_dim // 64, rows, 64],
            [head_dim, 64, K, 1],
            get_device_dtype(torch.float16),
        )
        return PropArg(dep, host, [source]), m, contraction, K

    def test_o_proj_input_is_k_sticks_outer(self):
        """o_proj reads its input K-sticks outer, the order of a 2-D [M, K]
        read, not rows outer: the batch of one does not take the tiles."""
        for rows in (512, 64):
            with self.subTest(rows=rows):
                arg, m, contraction, K = self._attention_output_scenario(rows)
                with V.set_graph_handler(SimpleNamespace()):
                    source = device_coordinates(arg.layouts[0], arg.dep, None)
                # The source is factorized (the contraction var sits on two
                # outer axes), so the edge restickifies to Pass 3's layout.
                self.assertEqual(
                    sum(contraction in c.free_symbols for c in source[:-1]), 2
                )

                result = find_stick_compatible_input_layout(
                    arg, contraction, BATCH_MATMUL_OP, "x"
                )

                self.assertEqual(
                    result,
                    SpyreTensorLayout(
                        [1, rows, K], [rows * K, K, 1], torch.float16, [1, 0, 2]
                    ),
                )
                self.assertEqual(list(result.device_size), [1, K // 64, rows, 64])
                with V.set_graph_handler(SimpleNamespace()):
                    self.assertEqual(
                        device_coordinates(result, arg.dep, None),
                        [
                            0,
                            sympy.floor(contraction / 64),
                            m,
                            sympy.Mod(contraction, 64),
                        ],
                    )
                    # On a matmul edge the factorized source restickifies to
                    # exactly this layout.
                    with mock.patch(_IS_MATMUL_OP, return_value=True):
                        self.assertEqual(
                            compute_restickify_needed(
                                arg.layouts[0], arg.layout, arg.dep, result, arg.dep
                            ),
                            (True, result),
                        )

                # The flat-M builder lays the same read out the same way when
                # o_proj's output is 2-D.
                generated = arg.dep.var_names[1]
                y = PropArg(
                    MemoryDep(
                        "y",
                        K * generated + contraction,
                        arg.dep.var_names,
                        arg.dep.size,
                    ),
                    FixedLayout(torch.device("cpu"), torch.float16, [4096, K], [K, 1]),
                    [SpyreTensorLayout([4096, K], [K, 1], torch.float16, [1, 0])],
                )
                flat = _flat_dense_projection_x_layout(
                    arg,
                    y,
                    FixedLayout(
                        torch.device("cpu"), torch.float16, [rows, 4096], [4096, 1]
                    ),
                    MemoryDep(
                        "out", 4096 * m + generated, arg.dep.var_names, arg.dep.size
                    ),
                    contraction,
                    rows,
                    4096,
                )
                data_dims = [i for i, s in enumerate(result.device_size) if s != 1]
                self.assertEqual(
                    [result.device_size[i] for i in data_dims], list(flat.device_size)
                )
                self.assertEqual(
                    [result.stride_map[i] for i in data_dims], list(flat.stride_map)
                )

    def test_o_proj_input_at_decode_is_unchanged(self):
        """With one row every non-stick dim has size one: nothing moves."""
        arg, _, contraction, K = self._attention_output_scenario(1)

        result = find_stick_compatible_input_layout(
            arg, contraction, BATCH_MATMUL_OP, "x"
        )

        self.assertEqual(
            result, SpyreTensorLayout([1, 1, K], [K, K, 1], torch.float16, [0, 1, 2])
        )

    def test_real_leading_batch_keeps_host_dim_order(self):
        """A leading batch that holds data keeps the host dim order."""
        b, m, generated, contraction = sympy.symbols(
            "b m generated contraction", integer=True, nonnegative=True
        )
        B, L, H, D = 2, 64, 8, 128
        K = H * D
        dep = MemoryDep(
            "x",
            L * K * b + K * m + contraction,
            (b, m, generated, contraction),
            (B, L, 4096, K),
        )
        host = FixedLayout(
            torch.device("cpu"), torch.float16, [B, L, H, D], [L * K, K, D, 1]
        )
        source = SpyreTensorLayout(
            [B, H, D // 64, L, 64],
            [L * K, D, 64, K, 1],
            get_device_dtype(torch.float16),
        )

        result = find_stick_compatible_input_layout(
            PropArg(dep, host, [source]), contraction, BATCH_MATMUL_OP, "x"
        )

        self.assertEqual(
            result,
            SpyreTensorLayout([B, L, K], [L * K, K, 1], torch.float16, [0, 1, 2]),
        )

    def test_batch_of_one_before_a_real_batch_is_invisible(self):
        """[1, B, M, K] is stored exactly like [B, M, K]: the batch of one
        moves out of the tile slot and the real batch keeps it."""
        b, m, generated, contraction = sympy.symbols(
            "b m generated contraction", integer=True, nonnegative=True
        )
        B, L, H, D = 2, 64, 8, 128
        K = H * D
        dep = MemoryDep(
            "x",
            L * K * b + K * m + contraction,
            (b, m, generated, contraction),
            (B, L, 4096, K),
        )
        host = FixedLayout(
            torch.device("cpu"),
            torch.float16,
            [1, B, L, H, D],
            [B * L * K, L * K, K, D, 1],
        )
        source = SpyreTensorLayout(
            [B, H, D // 64, L, 64],
            [L * K, D, 64, K, 1],
            get_device_dtype(torch.float16),
        )

        result = find_stick_compatible_input_layout(
            PropArg(dep, host, [source]), contraction, BATCH_MATMUL_OP, "x"
        )

        self.assertEqual(
            result,
            SpyreTensorLayout(
                [1, B, L, K], [B * L * K, L * K, K, 1], torch.float16, [1, 2, 0, 3]
            ),
        )
        real_batch = SpyreTensorLayout(
            [B, L, K], [L * K, K, 1], torch.float16, [0, 1, 2]
        )

        def physical(stl):
            return [
                (size, step)
                for size, step in zip(stl.device_size, stl.stride_map)
                if size != 1
            ]

        self.assertEqual(physical(result), physical(real_batch))
        self.assertEqual(physical(result), [(L, K), (K // 64, 64), (B, L * K), (64, 1)])

    def test_real_batch_below_the_rows_is_where_nonstick_dim_order_differs(self):
        """Two rules choose which dim sits in the slot between a matmul input's
        two stick dims, on different layouts. Pass 3 (_compute_dim_order, the
        output rule) builds the target a factorized input is restickified to;
        nonstick_dim_order reorders the producer's committed_stl after
        optimize_restickify and before restickify insertion, so it never sees
        the Pass 3 target. It moves the largest data dim into the slot.
        With a real batch B they
        agree while B >= M and differ when the rows M outnumber it: Pass 3
        keeps B in the slot, the reorder moves M there. That was already so
        for [B, M, K]; a batch of one in front, [1, B, M, K], now behaves the
        same way instead of putting the batch of one in the slot."""
        b, m, generated, contraction = sympy.symbols(
            "b m generated contraction", integer=True, nonnegative=True
        )
        H, D, L = 8, 128, 64
        K = H * D
        for B in (2, 64, 128):
            for lead in ((), (1,)):
                with self.subTest(B=B, M=L, leading=lead):
                    n = len(lead)
                    dep = MemoryDep(
                        "x",
                        L * K * b + K * m + contraction,
                        (b, m, generated, contraction),
                        (B, L, 4096, K),
                    )
                    host = FixedLayout(
                        torch.device("cpu"),
                        torch.float16,
                        [*lead, B, L, H, D],
                        [B * L * K] * n + [L * K, K, D, 1],
                    )
                    source = SpyreTensorLayout(
                        [B, H, D // 64, L, 64],
                        [L * K, D, 64, K, 1],
                        get_device_dtype(torch.float16),
                    )
                    target = find_stick_compatible_input_layout(
                        PropArg(dep, host, [source]), contraction, BATCH_MATMUL_OP, "x"
                    )

                    # The producer of the same [*lead, B, M, K] tensor, laid out
                    # by the output rule, then reordered as nonstick_dim_order
                    # reorders a matmul input's producer.
                    size = [*lead, B, L, K]
                    coords = [0] * n + [b, m, contraction]
                    producer = SpyreTensorLayout(
                        size,
                        [B * L * K] * n + [L * K, K, 1],
                        torch.float16,
                        _compute_dim_order(n + 2, size, coords),
                    )
                    write = MemoryDep(
                        "x",
                        L * K * b + K * m + contraction,
                        (b, m, contraction),
                        (B, L, K),
                    )
                    with V.set_graph_handler(SimpleNamespace()):
                        reordered = (
                            _matmul_reorder_stl(producer, write, "x") or producer
                        )

                    # The slot is the dim between the stick's tile index and
                    # its 64 elements.
                    self.assertEqual(target.device_size[-2], B)
                    self.assertEqual(producer.device_size[-2], B)
                    self.assertEqual(reordered.device_size[-2], max(B, L))
                    self.assertEqual(target.stride_map[-2], L * K)
                    self.assertEqual(producer.stride_map[-2], L * K)
                    self.assertEqual(reordered.stride_map[-2], K if B < L else L * K)

    def test_unread_leading_dim_does_not_take_the_tiles(self):
        """A leading dim the read never moves along holds no data for it, so
        it leaves the tile slot to the rows, whatever its size."""
        m, generated, contraction = sympy.symbols(
            "m generated contraction", integer=True, nonnegative=True
        )
        L, K = 64, 1024
        for batch in (1, 2):
            for offset in range(batch):
                with self.subTest(batch=batch, offset=offset):
                    # Reads one fixed batch: the index has no batch variable.
                    dep = MemoryDep(
                        "x",
                        offset * L * K + K * m + contraction,
                        (m, generated, contraction),
                        (L, 4096, K),
                    )
                    host = FixedLayout(
                        torch.device("cpu"),
                        torch.float16,
                        [batch, L, K],
                        [L * K, K, 1],
                    )
                    candidate = SpyreTensorLayout(
                        [batch, L, K], [L * K, K, 1], torch.float16, [0, 1, 2]
                    )

                    result = find_stick_compatible_input_layout(
                        PropArg(dep, host, [candidate]),
                        contraction,
                        BATCH_MATMUL_OP,
                        "x",
                    )

                    self.assertEqual(
                        result,
                        SpyreTensorLayout(
                            [batch, L, K], [L * K, K, 1], torch.float16, [1, 0, 2]
                        ),
                    )
                    self.assertEqual(list(result.device_size), [batch, K // 64, L, 64])

    def test_other_projection_inputs_keep_their_layout(self):
        """q/k/v, gate/up and down read an unfactorized [1, M, K] activation:
        whatever layout its producer chose, the matmul edge needs no
        restickify, and each 2-D weight keeps its own layout."""
        m, generated, contraction = sympy.symbols(
            "m generated contraction", integer=True, nonnegative=True
        )
        for K, N in ((4096, 1536), (4096, 3200), (3200, 4096)):
            for rows in (512, 64):
                dep = MemoryDep(
                    "x",
                    K * m + contraction,
                    (m, generated, contraction),
                    (rows, N, K),
                )
                host = FixedLayout(
                    torch.device("cpu"),
                    torch.float16,
                    [1, rows, K],
                    [rows * K, K, 1],
                )
                for dim_order in ([0, 1, 2], [1, 0, 2]):
                    with self.subTest(K=K, N=N, rows=rows, dim_order=dim_order):
                        candidate = SpyreTensorLayout(
                            [1, rows, K], [rows * K, K, 1], torch.float16, dim_order
                        )
                        with V.set_graph_handler(SimpleNamespace()):
                            coords = device_coordinates(candidate, dep, None)
                        self.assertEqual(
                            sum(contraction in c.free_symbols for c in coords[:-1]), 1
                        )
                        required = find_stick_compatible_input_layout(
                            PropArg(dep, host, [candidate]),
                            contraction,
                            BATCH_MATMUL_OP,
                            "x",
                        )
                        with (
                            V.set_graph_handler(SimpleNamespace()),
                            mock.patch(_IS_MATMUL_OP, return_value=True),
                        ):
                            self.assertEqual(
                                compute_restickify_needed(
                                    candidate, host, dep, required, dep
                                ),
                                (False, None),
                            )
                with self.subTest(K=K, N=N, rows=rows, operand="weight"):
                    y_dep = MemoryDep(
                        "y",
                        K * generated + contraction,
                        (m, generated, contraction),
                        (rows, N, K),
                    )
                    weight = SpyreTensorLayout([N, K], [K, 1], torch.float16, [1, 0])
                    required = find_stick_compatible_input_layout(
                        PropArg(
                            y_dep,
                            FixedLayout(
                                torch.device("cpu"), torch.float16, [N, K], [K, 1]
                            ),
                            [weight],
                        ),
                        generated,
                        BATCH_MATMUL_OP,
                        "y",
                    )
                    self.assertEqual(required, weight)

                    batched_weight = SpyreTensorLayout(
                        [1, N, K], [N * K, K, 1], torch.float16, [0, 2, 1]
                    )
                    required = find_stick_compatible_input_layout(
                        PropArg(
                            y_dep,
                            FixedLayout(
                                torch.device("cpu"),
                                torch.float16,
                                [1, N, K],
                                [N * K, K, 1],
                            ),
                            [batched_weight],
                        ),
                        generated,
                        BATCH_MATMUL_OP,
                        "y",
                    )
                    self.assertEqual(
                        required,
                        SpyreTensorLayout(
                            [1, N, K], [N * K, K, 1], torch.float16, [2, 0, 1]
                        ),
                    )
                    self.assertEqual(list(required.device_size), [1, N // 64, K, 64])

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

    def test_flat_dense_projection_layout(self):
        """A contiguous BLHD view read as [B*L, H*D] gets canonical flat M."""
        m, generated, contraction = sympy.symbols(
            "m generated contraction", integer=True, nonnegative=True
        )
        M, N, K = 2048, 768, 768
        ranges = (M, N, K)
        x_dep = MemoryDep("x", K * m + contraction, (m, generated, contraction), ranges)
        y_dep = MemoryDep(
            "y", K * generated + contraction, (m, generated, contraction), ranges
        )
        out_dep = MemoryDep(
            "out", N * m + generated, (m, generated, contraction), ranges
        )
        x_host = FixedLayout(
            torch.device("cpu"),
            torch.float16,
            [4, 512, 12, 64],
            [393216, 768, 64, 1],
        )
        y_host = FixedLayout(torch.device("cpu"), torch.float16, [K, N], [N, 1])
        output = FixedLayout(torch.device("cpu"), torch.float16, [M, N], [N, 1])
        source = SpyreTensorLayout(
            [512, 12, 1, 4, 64],
            [768, 64, 64, 393216, 1],
            get_device_dtype(torch.float16),
        )
        weight = SpyreTensorLayout(
            [12, 768, 64],
            [49152, 1, 768],
            get_device_dtype(torch.float16),
        )
        x = PropArg(x_dep, x_host, [source])
        y = PropArg(y_dep, y_host, [weight])

        result = _flat_dense_projection_x_layout(
            x, y, output, out_dep, contraction, M, N
        )
        expected = SpyreTensorLayout([M, K], [K, 1], torch.float16, [0, 1])

        self.assertEqual(result, expected)
        with V.set_graph_handler(SimpleNamespace()):
            self.assertEqual(
                device_coordinates(result, x_dep, None),
                [sympy.floor(contraction / 64), m, sympy.Mod(contraction, 64)],
            )

            compatible, compatible_target = compute_restickify_needed(
                source, x_host, x_dep, result, x_dep
            )
            needed, target = compute_restickify_needed(
                source,
                x_host,
                x_dep,
                result,
                x_dep,
                require_exact=True,
            )
        self.assertFalse(compatible)
        self.assertIsNone(compatible_target)
        self.assertTrue(needed)
        self.assertEqual(target, expected)

        strided_x = PropArg(
            x_dep,
            FixedLayout(
                torch.device("cpu"),
                torch.float16,
                [4, 512, 12, 64],
                [786432, 1536, 64, 1],
            ),
            [source],
        )
        self.assertIsNone(
            _flat_dense_projection_x_layout(
                strided_x, y, output, out_dep, contraction, M, N
            )
        )

        fp32_source = SpyreTensorLayout(
            [512, 24, 1, 4, 32],
            [768, 32, 32, 393216, 1],
            get_device_dtype(torch.float32),
        )
        self.assertIsNone(
            _flat_dense_projection_x_layout(
                PropArg(
                    x_dep,
                    FixedLayout(
                        torch.device("cpu"),
                        torch.float32,
                        [4, 512, 12, 64],
                        [393216, 768, 64, 1],
                    ),
                    [fp32_source],
                ),
                y,
                output,
                out_dep,
                contraction,
                M,
                N,
            )
        )

    def test_flat_dense_projection_rejects_true_bmm(self):
        """A rank-3 output and per-batch weight retain genuine BMM geometry."""
        batch, m, generated, contraction = sympy.symbols(
            "batch m generated contraction", integer=True, nonnegative=True
        )
        B, M, N, K = 4, 512, 768, 768
        ranges = (B, M, N, K)
        x_dep = MemoryDep(
            "x",
            M * K * batch + K * m + contraction,
            (batch, m, generated, contraction),
            ranges,
        )
        y_dep = MemoryDep(
            "y",
            K * N * batch + N * contraction + generated,
            (batch, m, generated, contraction),
            ranges,
        )
        out_dep = MemoryDep(
            "out",
            M * N * batch + N * m + generated,
            (batch, m, generated, contraction),
            ranges,
        )
        x_host = FixedLayout(
            torch.device("cpu"), torch.float16, [B, M, K], [M * K, K, 1]
        )
        y_host = FixedLayout(
            torch.device("cpu"), torch.float16, [B, K, N], [K * N, N, 1]
        )
        output = FixedLayout(
            torch.device("cpu"), torch.float16, [B, M, N], [M * N, N, 1]
        )
        source = SpyreTensorLayout(
            [12, M, B, 64],
            [64, K, M * K, 1],
            get_device_dtype(torch.float16),
        )
        weight = SpyreTensorLayout(
            [12, K, B, 64],
            [64, N, K * N, 1],
            get_device_dtype(torch.float16),
        )

        result = _flat_dense_projection_x_layout(
            PropArg(x_dep, x_host, [source]),
            PropArg(y_dep, y_host, [weight]),
            output,
            out_dep,
            contraction,
            M,
            N,
        )

        self.assertIsNone(result)


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
