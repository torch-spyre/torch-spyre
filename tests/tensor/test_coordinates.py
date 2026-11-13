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

import contextlib
import itertools
import math
from types import SimpleNamespace
from unittest import mock

import pytest
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
    _check_stick_expr_supported,
    compute_restickify_needed,
    device_coordinates,
    is_stick_expr_offset_free,
    restickify_dense_stl_to_sparse,
    sparse_stl_from_host,
    sparse_target_for_unservable_read,
    try_device_coordinates,
)
from torch_spyre._inductor.nonstick_dim_order import _flat_dense_projection_x_layout
from torch_spyre._inductor.propagate_layouts import (
    PropArg,
    _check_supported_input_sticks,
    _find_alt_target_stl,
    generate_stl_with_required_stick,
)
from torch_spyre._inductor import views as views_module
from torch_spyre._inductor.views import (
    _compute_coordinates_memo,
    _compute_coordinates_uncached,
    _coordinates_memo_key,
    _decompose_constant_offset,
    align_tensors,
    compute_coordinates,
    normalize_coordinates,
    tiling_expr_to_device_expr,
    UnalignedStickSplit,
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

    def test_stepped_nonstick_coordinate_bounds(self):
        # A valid ::step slice reaches at most rows - 1, including when
        # rows is not divisible by step. Test both common stick widths.
        for width in (32, 64):
            for rows, step in ((97, 3), (96, 3), (98, 5), (99, 3)):
                with self.subTest(width=width, rows=rows, step=step):
                    count = (rows + step - 1) // step
                    coords = compute_coordinates(
                        [rows, 256 // width, 2, width],
                        [256, width, rows * 256, 1],
                        {p0: 2, p1: count, p2: 256},
                        rows * 256 * p0 + step * 256 * p1 + p2,
                    )
                    self.assertEqual(coords, [step * p1, p2 // width, p0, p2 % width])

    def test_stepped_nonstick_coordinate_preserves_wrap(self):
        # A flattened batch/row axis genuinely crosses the original row
        # boundary. rows=99 exercises landing exactly on the boundary.
        for rows in (97, 99):
            with self.subTest(rows=rows):
                count = (2 * rows + 2) // 3
                coords = compute_coordinates(
                    [rows, 4, 2, 64],
                    [256, 64, rows * 256, 1],
                    {p0: count, p1: 256},
                    3 * 256 * p0 + p1,
                )
                self.assertEqual(
                    coords,
                    [(3 * p0) % rows, p1 // 64, (3 * p0) // rows, p1 % 64],
                )
                # Compare physical addresses at every selected row against
                # the original batch/row coordinates, including the wrap.
                for t in range(count):
                    for c in (0, 63, 64, 255):
                        actual = sum(
                            coord.subs({p0: t, p1: c}) * stride
                            for coord, stride in zip(coords, (512, 128, 64, 1))
                        )
                        batch, row = divmod(3 * t, rows)
                        expected = row * 512 + (c // 64) * 128 + batch * 64 + c % 64
                        self.assertEqual(actual, expected)

    def test_stepped_coordinate_last_value_at_boundary(self):
        # max == boundary must retain both the carry and modulo; max ==
        # boundary - 1 may drop them. This catches a strict-comparison error.
        for count in (2, 3):
            coords = compute_coordinates([2, 6], [6, 1], {p0: count}, 3 * p0)
            expected = [0, 3 * p0] if count == 2 else [(3 * p0) // 6, (3 * p0) % 6]
            self.assertEqual(coords, expected)

    def test_flat_read_across_unaligned_rows_wraps_stick_by_row_stride(self):
        # Dense fp16 [8, 65]: device [2, 8, 64], stride_map [64, 65, 1]. Element
        # c0 = 65 is row 1, column 0, so it sits at position 0 of its stick; the
        # stick coordinate must wrap by the row stride 65 first (issue #5872).
        c0 = sympy.Symbol("c0", integer=True, nonnegative=True)
        coords = compute_coordinates([2, 8, 64], [64, 65, 1], {c0: 520}, c0)
        self.assertEqual(coords[-1], sympy.Mod(sympy.Mod(c0, 65), 64))
        for flat in range(520):
            row, col = divmod(flat, 65)
            actual = [int(c.subs({c0: flat})) for c in coords]
            self.assertEqual(actual, [col // 64, row, col % 64], f"c0={flat}")

    def test_wrap_by_every_larger_stride_folds_when_strides_divide(self):
        # Where the larger strides divide each other, the chain folds back to the
        # single wrap, so the coordinates of aligned layouts do not change.
        c0 = sympy.Symbol("c0", integer=True, nonnegative=True)
        coords = compute_coordinates([2, 8, 64], [64, 128, 1], {c0: 1024}, c0)
        self.assertEqual(coords[-1], sympy.Mod(c0, 64))

    def test_wrap_chain_drops_strides_that_a_later_one_divides(self):
        # Device [3, 4, 1, 2, 64], strides [256, 64, 64, 768, 1] (a 4-D aligned
        # tensor): d0 crosses the strides 768 and 256, and 256 divides 768.  The
        # chain Mod(Mod(64*d0, 768), 256) must stay a single wrap.  sympy folds
        # it to 64*Mod(Mod(d0, 12)**2, 4), which is not the same function.
        d0, d2 = sympy.symbols("d0 d2", integer=True, nonnegative=True)
        coords = compute_coordinates(
            [3, 4, 1, 2, 64],
            [256, 64, 64, 768, 1],
            {d0: 24, d2: 64},
            64 * d0 + d2,
        )
        self.assertEqual(coords[1], sympy.Mod(d0, 4))
        self.assertFalse(any(c.has(sympy.Pow) for c in coords))
        for value in range(24):
            actual = [int(c.subs({d0: value, d2: 0})) for c in coords]
            self.assertEqual(actual[:2], [(value % 12) // 4, value % 4], f"d0={value}")
            self.assertEqual(actual[3], value // 12, f"d0={value}")

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


class TestComputeCoordinatesMemo(TestCase):
    """compute_coordinates is memoized on its arguments; a hit must be
    indistinguishable from computing the answer again."""

    def setUp(self):
        _compute_coordinates_memo.cache_clear()

    def _mixed_radix_call(self, repeat_info):
        lq, hd = sympy.symbols("lq hd", integer=True, nonnegative=True)
        return compute_coordinates(
            [1, 32, 64, 128],
            [262144, 8192, 128, 1],
            {lq: 64, hd: 4096},
            128 * lq
            + 8192 * ModularIndexing(hd, 128, 32)
            + ModularIndexing(hd, 1, 128),
            repeat_info_out=repeat_info,
        )

    def test_a_hit_returns_what_the_uncached_call_returns(self):
        lq, hd = sympy.symbols("lq hd", integer=True, nonnegative=True)
        args = (
            [1, 32, 64, 128],
            [262144, 8192, 128, 1],
            {lq: 64, hd: 4096},
            128 * lq
            + 8192 * ModularIndexing(hd, 128, 32)
            + ModularIndexing(hd, 1, 128),
        )
        with V.set_graph_handler(SimpleNamespace()):
            expected_info: dict = {}
            expected = _compute_coordinates_uncached(*args, None, expected_info)
            first_info: dict = {}
            first = self._mixed_radix_call(first_info)
            second_info: dict = {}
            second = self._mixed_radix_call(second_info)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(first_info, expected_info)
        self.assertEqual(second_info, expected_info)
        info = _compute_coordinates_memo.cache_info()
        self.assertEqual((info.misses, info.hits), (1, 1))

    def test_each_caller_gets_its_own_list_and_repeat_info(self):
        with V.set_graph_handler(SimpleNamespace()):
            first_info: dict = {}
            first = self._mixed_radix_call(first_info)
            expected, expected_info = list(first), repr(first_info)
            # Callers edit what they are handed; the memo must not see it.
            first[0] = sympy.Symbol("edited")
            for info in first_info.values():
                info["kind"] = "edited"
                for digit in info.get("digits", ()):
                    digit["coeff"] = -1
            second_info: dict = {}
            second = self._mixed_radix_call(second_info)
        self.assertEqual(second, expected)
        self.assertEqual(repr(second_info), expected_info)

    def test_an_error_is_raised_every_time_and_not_memoized(self):
        for _ in range(2):
            with self.assertRaisesRegex(Unsupported, "runs backwards"):
                compute_coordinates(
                    [4, 64], [64, 1], {p0: 4, p1: 64}, 192 - 64 * p0 + p1
                )
        self.assertEqual(_compute_coordinates_memo.cache_info().currsize, 0)

    def test_graph_dependent_calls_are_not_memoized(self):
        # A symbolic range or a dynamic size symbol is compared through
        # V.graph's size hints, so the arguments alone do not decide the result.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        self.assertIsNone(
            _coordinates_memo_key(
                [4, 64], [64, 1], {p0: s0, p1: 64}, 64 * p0 + p1, None
            )
        )
        self.assertIsNone(
            _coordinates_memo_key([4, 64], [64, 1], {p1: 64}, 64 * s0 + p1, None)
        )
        self.assertIsNotNone(
            _coordinates_memo_key([4, 64], [64, 1], {p0: 4, p1: 64}, 64 * p0 + p1, None)
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
        new_it_space, tensors, _ = align_tensors(it_space, [inp, out])
        aligned_in = tensors[0]
        # 8 rows x 8 sticks x 64: a stride-2 walk over 4 heads must not inflate
        # the stick-tile dim to 8 iterations *and* add a gap of 2 (it did, and
        # doubled the row stride so output row l read input row 2*l).
        self.assertEqual(sympy.prod(aligned_in["size"]), 8 * 8 * 64)
        sizes = [int(s) for s in aligned_in["size"]]
        gap_idx = sizes.index(2)
        self.assertEqual(sizes[gap_idx - 1], 4, "stick-tile dim counted in iterations")
        self.assertEqual(aligned_in["coordinates"][gap_idx - 1], d1)
        # Even without a residual offset the gap dim hangs off a synthetic size-1
        # variable: a variable-free gap dim is misaddressed by the backend.
        gap_coord = aligned_in["coordinates"][gap_idx]
        self.assertEqual(len(gap_coord.free_symbols), 1)
        z = next(iter(gap_coord.free_symbols))
        self.assertEqual(gap_coord, z)
        self.assertEqual(new_it_space[z][0], 1)

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

    def test_outer_stride_partial_group_preserves_addresses(self):
        # The final group need not be physically padded: only the selected
        # coordinates are accessed. Compare the complete address expressions,
        # including offsets, after alignment introduces its synthetic gap.
        r, c = sympy.symbols("r c", integer=True, nonnegative=True)
        for rows, step, start in [(97, 3, 0), (98, 5, 0), (97, 3, 1), (7, 3, 3)]:
            for width in (32, 64):
                for leading_unit in (False, True):
                    with self.subTest(
                        rows=rows,
                        step=step,
                        start=start,
                        width=width,
                        leading_unit=leading_unit,
                    ):
                        count = len(range(start, rows, step))
                        inp = {
                            "size": [rows, width],
                            "coordinates": [step * r + start, c],
                        }
                        out = {"size": [count, width], "coordinates": [r, c]}
                        if leading_unit:
                            inp["size"].insert(0, 1)
                            inp["coordinates"].insert(0, sympy.S.Zero)
                        space, tensors, _ = align_tensors(
                            {r: (count, 1), c: (width, 1)}, [inp, out]
                        )
                        synthetic = {v: 0 for v in space if v not in (r, c)}
                        for v in synthetic:
                            self.assertEqual(space[v][0], 1)
                        for tensor, expected in zip(
                            tensors, [(step * r + start) * width + c, r * width + c]
                        ):
                            address = sympy.S.Zero
                            for extent, coord in zip(
                                tensor["size"], tensor["coordinates"]
                            ):
                                address = address * extent + coord
                            # Alignment may retain floor(c / width); it is
                            # zero throughout the column iteration range.
                            for column in range(width):
                                substitutions = {**synthetic, c: column}
                                self.assertEqual(
                                    sympy.expand(address.subs(substitutions)),
                                    sympy.expand(expected.subs(substitutions)),
                                )
                        self.assertLess(
                            (start + step * (count - 1)) * width + width - 1,
                            rows * width,
                        )

    def test_outer_stride_partial_group_rejects_out_of_bounds(self):
        r, c = sympy.symbols("r c", integer=True, nonnegative=True)
        for count, start in [(34, 0), (33, 1), (33, -1)]:
            with self.subTest(count=count, start=start):
                with self.assertRaisesRegex(Unsupported, "does not divide"):
                    align_tensors(
                        {r: (count, 1), c: (64, 1)},
                        [
                            {"size": [97, 64], "coordinates": [3 * r + start, c]},
                            {"size": [count, 64], "coordinates": [r, c]},
                        ],
                    )

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
        bad = SpyreTensorLayout(
            [512, 128, 1, 1, 64],
            [4096, 1, 8192, 0, 128],
            dev,
            {(0,): 512, (1,): 128, (3,): 1, (2, 4): 64},
        )
        good = SpyreTensorLayout(
            [512, 1, 1, 64],
            [4096, 0, 0, 1],
            dev,
            {(0,): 512, (1, 3): 1, (2,): 1},
        )
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
                {(0,): 64, (2,): 32, (1, 3): 128},
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
        factorized = SpyreTensorLayout(
            [8, 2, 32, 64],
            [4096, 64, 128, 1],
            dtype,
            {(0,): 8, (2,): 32, (1, 3): 128},
        )
        canonical = SpyreTensorLayout(
            [1, 8, 4096], [32768, 4096, 1], torch.float16, [0, 1, 2]
        )

        for layouts in ([factorized, canonical], [canonical, factorized]):
            with self.subTest(first=layouts[0]):
                arg, contraction = self._scenario(layouts)
                result = generate_stl_with_required_stick(
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

        result = generate_stl_with_required_stick(
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
        """Only a gap-free mixed-radix view, windowed on whole inner dims, collapses."""
        scenarios = (
            # Heads stride 256 over a 128-wide D leaves a gap between heads.
            self._same_graph_attention_scenario(host_stride=(262144, 4096, 256, 1)),
            # 2000 elements end in the middle of a head (not a multiple of D).
            self._same_graph_attention_scenario(contraction_range=2000),
        )
        for arg, contraction, source in scenarios:
            with self.subTest(
                stride=arg.layout.stride,
                contraction_range=arg.dep.ranges[contraction],
            ):
                with self.assertRaisesRegex(Unsupported, "contiguous host dimension"):
                    generate_stl_with_required_stick(
                        arg, contraction, BATCH_MATMUL_OP, "x"
                    )

    def test_windowed_factorized_chain_collapses_to_the_full_chain(self):
        """A contraction var covering whole heads of a gap-free view collapses.

        A ``for_each_tile`` over head groups reads one window of the chain
        (here 16 of 32 heads, 2048 of 4096 elements); the window's position
        enters through the loop index, not the host coordinates.  The canonical
        layout must span the whole chain, not just the window.
        """
        arg, contraction, _ = self._same_graph_attention_scenario(
            contraction_range=2048
        )
        result = generate_stl_with_required_stick(
            arg, contraction, BATCH_MATMUL_OP, "x"
        )
        self.assertEqual(list(result.device_size), [64, 64, 1, 64])
        self.assertEqual(list(result.stride_map), [4096, 64, 0, 1])

    def test_outer_loop_var_in_the_head_coordinate_collapses_the_whole_chain(self):
        """An outer variable added to the head coordinate does not block collapse.

        The dep index ``4096*outer + contraction`` makes the head coordinate
        ``32*outer + floor(contraction/128)``: a loop variable selects a group of
        32 heads and the contraction variable walks one group.  Only the part of
        a host coordinate that varies with the contraction variable must depend
        on it alone, so the outer term is allowed (issue #5441).  The canonical
        layout spans the whole 64-head chain.
        """
        outer, contraction = sympy.symbols(
            "outer contraction", integer=True, nonnegative=True
        )
        heads = 64
        host = FixedLayout(
            torch.device("cpu"),
            torch.float16,
            [1, heads, 1, 128],
            [heads * 128, 128, 128, 1],
        )
        source = SpyreTensorLayout(
            [heads, 1, 2, 1, 64],
            [128, 0, 64, 0, 1],
            get_device_dtype(torch.float16),
            {(0,): heads, (1,): 1, (2, 4): 128, (3,): 1},
        )
        dep = MemoryDep(
            "x", 4096 * outer + contraction, (outer, contraction), (heads // 32, 4096)
        )
        result = generate_stl_with_required_stick(
            PropArg(dep, host, [source]), contraction, BATCH_MATMUL_OP, "x"
        )
        self.assertEqual(list(result.device_size), [1, 128, 1, 64])
        self.assertEqual(list(result.stride_map), [0, 64, 0, 1])

    def test_nonstandard_matmul_layout_is_not_canonicalized(self):
        """Non-STANDARD formats are returned unchanged by Pass 3."""
        dtype = get_device_dtype(torch.float16)
        _ds = [8, 2, 32, 64]
        qfp8wt = SpyreTensorLayout(
            _ds,
            [4096, 64, 128, 1],
            dtype,
            {(i,): _ds[i] for i in range(len(_ds))},
            ElementArrangement.QFP8WT,
        )
        arg, contraction = self._scenario([qfp8wt])
        result = generate_stl_with_required_stick(
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
        # substitution.
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
    unrelated, lower-level gaps (an op that needs a restickify has no feasible
    layout for IEEE_FP32: it fails with "no mechanism to resolve stick
    incompatibility" -- see test_restickify_fp32_unsupported in
    test_inductor_ops.py -- and separately the DL op scheduler finds no
    candidate for a fused copy/slice
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


class TestFlatDenseProjectionLayout(TestCase):
    """Unit tests for _flat_dense_projection_x_layout in nonstick_dim_order."""

    def test_flat_dense_projection_layout(self):
        """A contiguous BLHD view read as [B*L, H*D] gets a canonical flat-M STL."""
        B, seq, heads, head_dim = 4, 512, 12, 64
        M, K, N = B * seq, heads * head_dim, 768

        m, generated, contraction = sympy.symbols(
            "m generated contraction", integer=True, nonnegative=True
        )
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
            [B, seq, heads, head_dim],
            [seq * K, K, head_dim, 1],
        )
        y_host = FixedLayout(torch.device("cpu"), torch.float16, [K, N], [N, 1])
        output = FixedLayout(torch.device("cpu"), torch.float16, [M, N], [N, 1])
        source = SpyreTensorLayout(
            [seq, heads, 1, B, 64],
            [K, head_dim, head_dim, seq * K, 1],
            get_device_dtype(torch.float16),
            {(0,): seq, (1, 4): K, (2,): 1, (3,): B},
        )
        weight = SpyreTensorLayout(
            [heads, K, 64],
            [head_dim * N, 1, K],
            get_device_dtype(torch.float16),
            {(0,): heads, (1,): K, (2,): 64},
        )
        x = PropArg(x_dep, x_host, [source])
        y = PropArg(y_dep, y_host, [weight])

        result = _flat_dense_projection_x_layout(
            x, y, output, out_dep, contraction, M, N
        )
        self.assertIsNotNone(result)
        # SpyreTensorLayout stickifies [M, K] to [K//64, M, 64].
        self.assertEqual(list(result.device_size), [K // 64, M, 64])
        self.assertEqual(list(result.stride_map), [64, K, 1])

        # A strided (non-contiguous) x must be rejected.
        strided_x = PropArg(
            x_dep,
            FixedLayout(
                torch.device("cpu"),
                torch.float16,
                [B, seq, heads, head_dim],
                [seq * K * 2, K * 2, head_dim, 1],
            ),
            [source],
        )
        self.assertIsNone(
            _flat_dense_projection_x_layout(
                strided_x, y, output, out_dep, contraction, M, N
            )
        )

        # FP32 must be rejected.
        fp32_source = SpyreTensorLayout(
            [seq, heads, 1, B, 32],
            [K, head_dim, head_dim, seq * K, 1],
            get_device_dtype(torch.float32),
            {(0,): seq, (1,): heads, (2,): 1, (3,): B, (4,): 32},
        )
        fp32_x = PropArg(
            x_dep,
            FixedLayout(
                torch.device("cpu"),
                torch.float32,
                [B, seq, heads, head_dim],
                [seq * K, K, head_dim, 1],
            ),
            [fp32_source],
        )
        self.assertIsNone(
            _flat_dense_projection_x_layout(
                fp32_x, y, output, out_dep, contraction, M, N
            )
        )

    def test_flat_dense_projection_rejects_true_bmm(self):
        """A rank-3 output and per-batch weight are not collapsed (genuine BMM)."""
        batch, m_sym, generated, contraction = sympy.symbols(
            "batch m generated contraction", integer=True, nonnegative=True
        )
        B, M, N, K = 4, 512, 768, 768
        ranges = (B, M, N, K)
        x_dep = MemoryDep(
            "x",
            M * K * batch + K * m_sym + contraction,
            (batch, m_sym, generated, contraction),
            ranges,
        )
        y_dep = MemoryDep(
            "y",
            K * N * batch + N * contraction + generated,
            (batch, m_sym, generated, contraction),
            ranges,
        )
        out_dep = MemoryDep(
            "out",
            M * N * batch + N * m_sym + generated,
            (batch, m_sym, generated, contraction),
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
            {(0, 3): K, (1,): M, (2,): B},
        )
        weight = SpyreTensorLayout(
            [12, K, B, 64],
            [64, N, K * N, 1],
            get_device_dtype(torch.float16),
            {(0, 3): K, (1,): K, (2,): B},
        )
        x = PropArg(x_dep, x_host, [source])
        y = PropArg(y_dep, y_host, [weight])

        result = _flat_dense_projection_x_layout(
            x, y, output, out_dep, contraction, M, N
        )
        self.assertIsNone(result)


def _contiguous_host(shape, dtype=torch.float16):
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    return FixedLayout(torch.device("spyre:0"), dtype, list(shape), strides)


def _non_unit_dims(stl):
    """(size, stride) of every device dim of size other than 1, outermost first."""
    return [
        (size, stride)
        for size, stride in zip(stl.device_size, stl.stride_map)
        if size != 1
    ]


def _flat_read(numel):
    c0 = sympy.Symbol("c0", integer=True, nonnegative=True)
    return c0, MemoryDep("arg0_1", c0, (c0,), (numel,))


class TestHostSparseLayout(TestCase):
    """``sparse_stl_from_host`` keeps the host dims and puts the former stick dim on
    a new size-1 dim.  The former stick dim is padded at its end, so it has to be
    the outermost device dim: only then does the padding sit at the end of the
    buffer, and not between the slices of another dim."""

    def test_2d_layout(self):
        sp = sparse_stl_from_host(_contiguous_host([8, 65]))
        self.assertEqual(list(sp.device_size), [65, 1, 8, 64])
        self.assertEqual(list(sp.stride_map), [1, 0, 65, 0])

    def test_3d_former_stick_dim_is_outermost(self):
        sp = sparse_stl_from_host(_contiguous_host([2, 3, 65]))
        self.assertEqual(list(sp.device_size), [65, 3, 1, 2, 64])
        self.assertEqual(list(sp.stride_map), [1, 65, 0, 195, 0])

    def test_former_stick_dim_outermost_for_every_rank(self):
        for shape in ([8, 65], [2, 3, 65], [4, 2, 100], [2, 3, 4, 65]):
            with self.subTest(shape=shape):
                sp = sparse_stl_from_host(_contiguous_host(shape))
                self.assertEqual(sp.device_size[0], shape[-1])
                self.assertEqual(sp.stride_map[0], 1)

    def test_every_host_dim_and_the_stick_stay_in_the_layout(self):
        sp = sparse_stl_from_host(_contiguous_host([2, 3, 4, 65]))
        self.assertEqual(sorted(sp.device_size), sorted([65, 3, 4, 1, 2, 64]))
        # One device dim per host dim plus the new size-1 dim and the stick; the
        # host strides are kept, the new dims have none.
        self.assertEqual(sorted(sp.stride_map), sorted([1, 260, 65, 0, 780, 0]))

    def test_size_one_dims_do_not_change_the_rule(self):
        for shape in ([1, 65], [2, 1, 65], [1, 3, 65], [1, 1, 65]):
            with self.subTest(shape=shape):
                sp = sparse_stl_from_host(_contiguous_host(shape))
                self.assertEqual(sp.device_size[0], 65)
                self.assertEqual(sp.stride_map[0], 1)

    def test_dtype_override(self):
        host = _contiguous_host([2, 3, 65])
        default = sparse_stl_from_host(host)
        as_bool = sparse_stl_from_host(host, torch.bool)
        self.assertEqual(
            default.device_dtype, SpyreTensorLayout([1, 1], torch.float16).device_dtype
        )
        self.assertEqual(
            as_bool.device_dtype, SpyreTensorLayout([1, 1], torch.bool).device_dtype
        )
        self.assertEqual(list(as_bool.device_size), list(default.device_size))


class TestSparseLayoutServesRead(TestCase):
    """A flat read that splits a stick on the dense layout is served by the input's
    sparse layout.  Whatever form the coordinates take, they have to land on the
    element the read names."""

    def _assert_flat_read_exact(self, sp, numel):
        c0, dep = _flat_read(numel)
        coords = try_device_coordinates(sp, dep, None)
        self.assertIsNotNone(coords)
        strides = list(sp.stride_map)
        for flat in range(numel):
            values = [int(c.subs({c0: flat})) for c in coords]
            # The stick and any size-1 dim sit at coordinate 0.
            for value, stride in zip(values, strides):
                if stride == 0:
                    self.assertEqual(value, 0, f"c0={flat}")
            # Contiguous host: the host offset of the element is the flat index.
            offset = sum(v * st for v, st in zip(values, strides) if st > 0)
            self.assertEqual(offset, flat, f"c0={flat}")

    def test_flat_read_is_served_and_exact(self):
        for shape in (
            [8, 65],
            [3, 65],
            [4, 130],
            [2, 100],
            [2, 3, 65],
            [4, 2, 100],
            # 4D, including size-1 dims in the middle and in the front.
            [2, 3, 4, 65],
            [2, 2, 2, 100],
            [3, 1, 2, 65],
            [1, 2, 3, 65],
        ):
            with self.subTest(shape=shape):
                host = _contiguous_host(shape)
                numel = math.prod(shape)
                dense = SpyreTensorLayout(shape, torch.float16)
                _, dep = _flat_read(numel)
                sparse = sparse_target_for_unservable_read(dense, host, dep)
                self.assertIsNotNone(sparse)
                # Size-1 dims carry no data, so only the non-unit dims must match.
                self.assertEqual(
                    _non_unit_dims(sparse), _non_unit_dims(sparse_stl_from_host(host))
                )
                self._assert_flat_read_exact(sparse, numel)

    def test_flat_read_is_exact_for_fp32(self):
        # fp32: 32 elements per stick, so 33 is the unaligned row length.
        shape = [4, 33]
        host = _contiguous_host(shape, torch.float32)
        dense = SpyreTensorLayout(shape, torch.float32)
        _, dep = _flat_read(math.prod(shape))
        sparse = sparse_target_for_unservable_read(dense, host, dep)
        self.assertIsNotNone(sparse)
        self._assert_flat_read_exact(sparse, math.prod(shape))

    def test_aligned_read_is_not_redirected(self):
        for shape in ([8, 128], [2, 3, 64], [2, 2, 2, 128], [2, 2, 3, 64]):
            with self.subTest(shape=shape):
                host = _contiguous_host(shape)
                dense = SpyreTensorLayout(shape, torch.float16)
                _, dep = _flat_read(math.prod(shape))
                self.assertIsNone(sparse_target_for_unservable_read(dense, host, dep))

    def test_sparse_input_is_not_redirected(self):
        host = _contiguous_host([8, 65])
        _, dep = _flat_read(520)
        self.assertIsNone(
            sparse_target_for_unservable_read(sparse_stl_from_host(host), host, dep)
        )

    def test_restickify_target_for_unserved_read(self):
        # compute_restickify_needed asks for the sparse layout instead of
        # returning "no restickify needed" or "infeasible".
        for shape in ([8, 65], [2, 3, 65], [2, 3, 4, 65], [3, 1, 2, 65]):
            with self.subTest(shape=shape):
                host = _contiguous_host(shape)
                numel = math.prod(shape)
                dense = SpyreTensorLayout(shape, torch.float16)
                _, dep = _flat_read(numel)
                needed, target = compute_restickify_needed(
                    dense, host, dep, SpyreTensorLayout([numel], torch.float16), dep
                )
                self.assertTrue(needed)
                self.assertIsNotNone(target)
                self.assertEqual(
                    _non_unit_dims(target), _non_unit_dims(sparse_stl_from_host(host))
                )


class TestRestickifyDenseStlToSparse(TestCase):
    """The target built from the dense layout keeps its outer dims and matches the
    host-built layout for a contiguous host."""

    def test_matches_host_layout_for_contiguous_hosts(self):
        for shape in ([8, 65], [2, 3, 65], [4, 2, 100], [2, 3, 4, 65]):
            with self.subTest(shape=shape):
                host = _contiguous_host(shape)
                dense = SpyreTensorLayout(shape, torch.float16)
                target = restickify_dense_stl_to_sparse(dense, host)
                self.assertIsNotNone(target)
                self.assertEqual(
                    _non_unit_dims(target), _non_unit_dims(sparse_stl_from_host(host))
                )
                self.assertEqual(
                    list(target.stride_map), list(sparse_stl_from_host(host).stride_map)
                )

    def test_keeps_the_dense_outer_dim_order(self):
        # The dense layout puts host dim 0 outside host dim 1 (dim_order [1, 0, 2]).
        # The target must keep that order, which the host-shape layout does not.
        shape = [2, 3, 65]
        host = _contiguous_host(shape)
        dense = SpyreTensorLayout(shape, host.stride, torch.float16, [1, 0, 2])
        target = restickify_dense_stl_to_sparse(dense, host)
        self.assertIsNotNone(target)
        self.assertEqual(list(target.device_size), [65, 2, 1, 3, 64])
        self.assertEqual(list(target.stride_map), [1, 195, 0, 65, 0])
        self.assertNotEqual(
            list(target.device_size), list(sparse_stl_from_host(host).device_size)
        )
        # The target serves the flat read exactly.
        c0, dep = _flat_read(math.prod(shape))
        coords = try_device_coordinates(target, dep, None)
        self.assertIsNotNone(coords)
        for flat in range(math.prod(shape)):
            values = [int(c.subs({c0: flat})) for c in coords]
            offset = sum(v * st for v, st in zip(values, target.stride_map) if st > 0)
            self.assertEqual(offset, flat, f"c0={flat}")


class TestStickExprOffsetFree(TestCase):
    def test_stick_split_by_a_row_boundary_is_not_a_stick(self):
        c0 = sympy.Symbol("c0", integer=True, nonnegative=True)
        self.assertFalse(
            is_stick_expr_offset_free(sympy.Mod(sympy.Mod(c0, 65), 64), 64)
        )

    def test_plain_stick_expressions_stay_supported(self):
        c0 = sympy.Symbol("c0", integer=True, nonnegative=True)
        self.assertTrue(is_stick_expr_offset_free(sympy.Mod(c0, 64), 64))
        self.assertTrue(is_stick_expr_offset_free(c0, 64))
        self.assertTrue(is_stick_expr_offset_free(sympy.S.Zero, 64))


def _reshape_read(rows, cols, a_n, b_n, stick=64):
    """Dense ``x[rows, cols]`` read as ``reshape(a_n, b_n)`` (``a_n == 1``: flat).

    ``stick`` is the number of elements per stick (64 for fp16, 32 for fp32).
    Returns the device size, the coordinates, the loop ranges, and every point
    where the coordinates differ from the true (tile, row, stick position).
    """
    tiles = math.ceil(cols / stick)
    if tiles > 1:
        size, stride = [tiles, rows, stick], [stick, cols, 1]
    else:
        size, stride = [rows, stick], [cols, 1]
    c0, c1 = sympy.symbols("c0 c1", integer=True, nonnegative=True)
    if a_n == 1:
        var_ranges, index, names = {c0: b_n}, c0, (c0,)
    else:
        var_ranges, index, names = {c0: a_n, c1: b_n}, b_n * c0 + c1, (c0, c1)
    coords = compute_coordinates(size, stride, var_ranges, index)
    wrong = []
    for point in itertools.product(*(range(var_ranges[v]) for v in names)):
        subs = {v: sympy.Integer(p) for v, p in zip(names, point)}
        row, col = divmod(int(index.xreplace(subs)), cols)
        want = (col // stick, row, col % stick) if tiles > 1 else (row, col)
        got = tuple(int(sympy.floor(c.xreplace(subs))) for c in coords)
        if got != want:
            wrong.append((point, got, want))
    return size, coords, var_ranges, wrong


# The reads of ``x[R, C].reshape(-1, K)`` swept below.  Rows and columns include
# unaligned (65, 100), partly aligned (96, 192) and aligned (128) cases; K covers
# a divisor of a stick (32), a whole stick (64), and rows that do not divide it.
_SWEEP_GRID = [
    (rows, cols, k, stick)
    for rows, cols, stick in itertools.product(
        (2, 3, 8), (65, 96, 100, 128, 192), (64, 32)
    )
    for k in (32, 64, 96, 128)
    if (rows * cols) % k == 0 and k < rows * cols
]

# Reads whose coordinates are wrong although ``_check_stick_expr_supported``
# accepts the stick coordinate, so layout selection trusts them (they fail later,
# in codegen).  Several loop variables carry across a row stride that does not
# divide them, which ``_has_misaligned_step`` does not see.
# TODO(#5872): one method that is exact for every read empties these sets.
_WRONG_AND_ACCEPTED = {
    "default": {
        (2, 96, 64, 32),
        (2, 96, 64, 64),
        (2, 192, 128, 32),
        (2, 192, 128, 64),
        (3, 128, 96, 32),
        (8, 96, 64, 32),
        (8, 96, 64, 64),
        (8, 100, 32, 32),
        (8, 192, 128, 32),
        (8, 192, 128, 64),
    },
    "per_variable": {
        (2, 96, 64, 32),
        (2, 96, 64, 64),
        (2, 192, 96, 64),
        (2, 192, 128, 32),
        (2, 192, 128, 64),
        (3, 128, 96, 32),
        (3, 128, 96, 64),
        (3, 192, 96, 64),
        (8, 96, 64, 32),
        (8, 96, 64, 64),
        (8, 96, 128, 32),
        (8, 100, 32, 32),
        (8, 192, 96, 64),
        (8, 192, 128, 32),
        (8, 192, 128, 64),
    },
    "whole_index": set(),
}


class TestReshapeReadCoordinates(TestCase):
    """The coordinates of reshaped reads of an unaligned dense tensor (issue #5872).

    Every point is checked against the true (tile, row, position) of the element.
    The tests below record the scope: exact and accepted, exact but not yet
    expressible by the frontend, and not exact (missing carry).

    The sweep tests at the end compare every read of ``x[R, C].reshape(-1, K)``
    (issue #5872), at both stick sizes, with the true position of each element,
    for each way of building the coordinates:

    - ``default``: ``compute_coordinates`` as it is, which builds the coordinates
      per loop variable and switches to the whole index when
      ``_has_misaligned_step`` fires;
    - ``per_variable``: never switches;
    - ``whole_index``: always decomposes the whole index.

    A read is "accepted" when ``_check_stick_expr_supported`` takes its stick
    coordinate, which is all layout selection checks.  The dangerous reads are
    the wrong ones that are accepted; they are listed in ``_WRONG_AND_ACCEPTED``,
    so a read that becomes exact makes the test fail until it is removed.
    """

    def test_flat_and_reshaped_reads_are_exact(self):
        for rows, cols, a_n, b_n in (
            (8, 65, 1, 520),
            (3, 65, 1, 195),
            (4, 130, 1, 520),
            (2, 100, 1, 200),
            (16, 30, 1, 480),
            (8, 96, 1, 768),
            (8, 65, 4, 130),
            (8, 65, 2, 260),
            (3, 128, 2, 192),
        ):
            with self.subTest(read=(rows, cols, a_n, b_n)):
                self.assertEqual(_reshape_read(rows, cols, a_n, b_n)[3], [])

    def test_misaligned_step_read_is_exact(self):
        # Issue #5872 case 2: [3, 128].reshape(-1, 96), step 96 is not a multiple
        # of the stride 64 it lands on.
        self.assertEqual(_reshape_read(3, 128, 4, 96)[3], [])

    # The reads below are exact, but ``Term.from_coordinate`` cannot express the
    # coordinate, so the frontend rejects them.
    # TODO: when it can, these become positive tests (and join the list above).
    def _assert_frontend_rejects(self, read, error, match):
        size, coords, var_ranges, wrong = _reshape_read(*read)
        self.assertEqual(wrong, [])
        counter = [0]

        def synthetic_var():
            counter[0] += 1
            return sympy.Symbol(f"z{counter[0]}", integer=True, nonnegative=True)

        with self.assertRaisesRegex(error, match):
            normalize_coordinates(dict(var_ranges), size, coords, synthetic_var)

    def test_stepped_modular_coordinate_is_rejected_by_the_frontend(self):
        # [3, 128].reshape(4, 96): the coordinate is Mod(3*c0, 4).
        self._assert_frontend_rejects(
            (3, 128, 4, 96),
            Unsupported,
            r"Unsupported modular coordinate expression Mod\(3\*c0, 4\)",
        )

    def test_non_unit_fraction_is_rejected_by_the_frontend(self):
        # [8, 96].reshape(6, 128): the coordinate has 4*c0/3 (issue #1353).
        self._assert_frontend_rejects(
            (8, 96, 6, 128), AssertionError, "Unsupported coordinate expression"
        )

    def test_non_unit_fraction_with_modulus_is_rejected_by_the_frontend(self):
        # [8, 65].reshape(5, 104): the coordinate has 13*Mod(3*c0, 5)/64.
        self._assert_frontend_rejects(
            (8, 65, 5, 104), AssertionError, "Unsupported coordinate expression"
        )

    # LIMIT: several loop variables add into one device dim and the sum can exceed
    # it, but the carry into the next dim is not computed, so the coordinates are
    # wrong.  The frontend rejects all of them today (no wrong value reaches the
    # device); this will matter once it can express Mod(3*c0, 8).
    # TODO(#5872): a coordinate method that is exact for every read replaces these.
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="#5872: [3,128].reshape(8,48): stick coordinate c1 + 16*Mod(3*c0, 4) "
        "reaches 95 > 64 and the carry is lost (96 of 384 elements wrong)",
    )
    def test_carry_between_variables_3x128_to_8x48(self):
        self.assertEqual(_reshape_read(3, 128, 8, 48)[3], [])

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="#5872: [8,65].reshape(65,8): stick coordinate "
        "c1 + Mod(Mod(8*c0, 65), 64) lacks the carry (35 of 520 elements wrong)",
    )
    def test_carry_between_variables_8x65_to_65x8(self):
        self.assertEqual(_reshape_read(8, 65, 65, 8)[3], [])

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="#5872: [2,100].reshape(4,50): sympy squares the inner Mod, "
        "Mod(25*Mod(c0, 2)**2, 32) (72 of 200 elements wrong; the device result is "
        "right only because the operator reads the sparse layout)",
    )
    def test_carry_between_variables_2x100_to_4x50(self):
        self.assertEqual(_reshape_read(2, 100, 4, 50)[3], [])

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="#5872: [8,96].reshape(12,64): the row coordinate floor(2*c0/3) and the "
        "carry are wrong (384 of 768 elements wrong)",
    )
    def test_carry_between_variables_8x96_to_12x64(self):
        self.assertEqual(_reshape_read(8, 96, 12, 64)[3], [])

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        forced = {"default": None, "per_variable": False, "whole_index": True}
        cls.reads = {}
        for mode, force in forced.items():
            patch = (
                mock.patch.object(
                    views_module, "_has_misaligned_step", return_value=force
                )
                if force is not None
                else contextlib.nullcontext()
            )
            # compute_coordinates is memoized on its arguments, which do not
            # include the patched helper: drop the memo so each mode recomputes.
            _compute_coordinates_memo.cache_clear()
            with patch:
                cls.reads[mode] = {read: cls._classify(*read) for read in _SWEEP_GRID}
            _compute_coordinates_memo.cache_clear()

    @staticmethod
    def _classify(rows, cols, k, stick):
        _, coords, _, wrong = _reshape_read(rows, cols, rows * cols // k, k, stick)
        try:
            _check_stick_expr_supported(coords[-1], stick)
            accepted = True
        except Unsupported:
            accepted = False
        return not wrong, accepted

    def _wrong_and_accepted(self, mode):
        return {r for r, (exact, ok) in self.reads[mode].items() if ok and not exact}

    def test_default_wrong_reads_accepted_by_the_stick_check_are_known(self):
        self.assertEqual(
            self._wrong_and_accepted("default"), _WRONG_AND_ACCEPTED["default"]
        )

    def test_per_variable_wrong_reads_accepted_by_the_stick_check_are_known(self):
        self.assertEqual(
            self._wrong_and_accepted("per_variable"),
            _WRONG_AND_ACCEPTED["per_variable"],
        )

    def test_whole_index_is_exact_for_every_read(self):
        reads = self.reads["whole_index"]
        inexact = [r for r, (exact, _) in reads.items() if not exact]
        self.assertEqual(inexact, [])

    def test_default_is_never_worse_than_per_variable(self):
        # The switch to the whole index only ever fixes a read.
        worse = [
            r
            for r, (exact, _) in self.reads["default"].items()
            if self.reads["per_variable"][r][0] and not exact
        ]
        self.assertEqual(worse, [])

    def test_whole_index_gives_up_reads_the_stick_check_accepts(self):
        # Exact, but not in a form the stick check recognizes: why it cannot
        # simply replace the per-variable construction.
        default, whole = self.reads["default"], self.reads["whole_index"]
        lost = [
            r for r, (exact, ok) in default.items() if exact and ok and not whole[r][1]
        ]
        self.assertGreater(len(lost), 0)


if __name__ == "__main__":
    run_tests()
