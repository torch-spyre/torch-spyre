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
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import (
    device_coordinates,
    is_stick_expr_offset_free,
    try_device_coordinates,
)
from torch_spyre._inductor.propagate_layouts import (
    PropArg,
    _check_supported_input_sticks,
)
from torch_spyre._inductor.views import (
    align_tensors,
    compute_coordinates,
    normalize_coordinates,
    tiling_expr_to_device_expr,
)
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


class TestScaledModStickExpr(TestCase):
    """Regression coverage for a strided (step>1) slice landing inside a
    stick, e.g. ``t[:, ::2].to(torch.float32).to(torch.float16)`` on a
    [4, 128] fp16 tensor.

    sympy auto-canonicalizes ``Mod(2*d1, 64)`` to ``2*Mod(d1, 32)`` --
    algebraically identical, just regrouped. Three independent spots assumed
    the un-factored ``Mod(var, elems_per_stick)`` shape and broke on the
    canonicalized one:

    1. ``is_stick_expr_offset_free`` rejected the scaled-Mod form outright
       (``Unexpected stick expression 2*(Mod(d1, 32))``).
    2. ``align_tensors``'s split-tracking identified "the stick term" by
       comparing variable identity (``var != stick_dim[i]``) rather than
       position, so an unrelated outer ("which stick") term sharing the same
       variable was mistaken for the stick term and had its ``mod`` dropped
       from ``splits`` (``ValueError: 64 is not in list``).
    3. ``align_tensors``'s "ensure stick dim var occurs twice" fallback
       rebuilt the stick coordinate from scratch as plain ``var //
       elems_per_stick`` / ``var % elems_per_stick``, silently discarding the
       ``2x`` scale factor whenever the outer segment had been renamed to a
       synthetic var (e.g. ``z0``) rather than reusing the original name.
    """

    def _dtype(self):
        return SpyreTensorLayout([1, 1], torch.float16).device_dtype

    def test_device_coordinates_accepts_scaled_mod(self):
        dev = self._dtype()
        d0, d1 = sympy.symbols("d0 d1", integer=True, nonnegative=True)
        # arg0_1: [4, 128] fp16, tiled into 2 sticks of 64; dep reads every
        # other column (the `::2` slice), ranges {d0: 4, d1: 64}.
        dep = MemoryDep("buf", 128 * d0 + 2 * d1, (d0, d1), (4, 64))
        stl = SpyreTensorLayout([2, 4, 64], [64, 128, 1], dev)
        coords = device_coordinates(stl, dep, None)  # must not raise
        self.assertEqual(coords[-1], 2 * sympy.Mod(d1, 32))

    def test_is_stick_expr_offset_free_scaled_mod_forms(self):
        d1 = sympy.Symbol("d1", integer=True, nonnegative=True)
        # sympy's canonicalized coeff*Mod(var, N) form, coeff*N == stick size.
        self.assertTrue(is_stick_expr_offset_free(sympy.Mod(2 * d1, 64), 64))
        # Un-scaled forms still work as before.
        self.assertTrue(is_stick_expr_offset_free(sympy.Mod(d1, 64), 64))
        self.assertTrue(is_stick_expr_offset_free(d1, 64))
        # A coefficient that does not evenly divide the stick size is not a
        # representable stick expression.
        self.assertFalse(is_stick_expr_offset_free(3 * sympy.Mod(d1, 64), 128))

    def test_align_tensors_scaled_mod_stick_dim(self):
        d0, d1 = sympy.symbols("d0 d1", integer=True, nonnegative=True)
        iteration_space = {d0: (4, 1), d1: (64, 1)}
        tensors = [
            {
                "size": [2, 4, 64],
                "coordinates": [sympy.floor(d1 / 32), d0, 2 * sympy.Mod(d1, 32)],
            },
        ]
        _, new_tensors = align_tensors(iteration_space, tensors)  # must not raise
        # The final (stick-dim) coordinate must retain the *2 scale factor;
        # it must not collapse to the unscaled `Mod(d1, 64)`.
        self.assertEqual(new_tensors[0]["coordinates"][-1], 2 * sympy.Mod(d1, 32))

    def test_align_tensors_matmul_unaffected(self):
        # Matmul's "outer chunk" term legitimately shares its variable with
        # the stick term (e.g. floor(c2/64) alongside Mod(c2, 64)) without
        # any coefficient. This must decompose exactly as before the fix.
        c0, c1, c2 = sympy.symbols("c0 c1 c2", integer=True, nonnegative=True)
        iteration_space = {c0: (64, 4), c1: (256, 4), c2: (128, 2)}
        tensors = [
            {
                "size": [2, 64, 64],
                "coordinates": [sympy.floor(c2 / 64), c0, sympy.Mod(c2, 64)],
            },
        ]
        new_splits, new_tensors = align_tensors(iteration_space, tensors)
        self.assertEqual(new_splits, iteration_space)
        self.assertEqual(
            new_tensors[0],
            {
                "size": [2, 64, 64],
                "coordinates": [sympy.floor(c2 / 64), c0, sympy.Mod(c2, 64)],
            },
        )


if __name__ == "__main__":
    run_tests()
