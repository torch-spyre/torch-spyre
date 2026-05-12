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

"""IR-level unit tests for insert_padding_ir.

Tests hook into CustomPreSchedulingPasses after insert_padding_ir runs to inspect
the operations list directly, without requiring end-to-end compilation to succeed.
"""

from typing import Any, Callable, Optional, TypeVarTuple, Unpack, override

import unittest
from unittest.mock import patch

import torch
from torch._inductor import config as t_inductor_config
from torch._inductor.ir import (
    ComputedBuffer,
    Operation,
    Reduction,
)

from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.ir import SpyreConstantFallback, SpyreEmptyFallback
from torch_spyre._inductor.passes import CustomPreSchedulingPasses


Ts = TypeVarTuple("Ts")


# ---------------------------------------------------------------------------
# Hooks into CustomPreSchedulingPasses
# ---------------------------------------------------------------------------


class CustomPreSchedulingPassesWithCapture(CustomPreSchedulingPasses):
    """Subclass of CustomPreSchedulingPasses that captures the operations list
    after all built-in passes (including insert_padding_ir) have run."""

    test_instance: Optional["TestInsertPaddingIR"] = None

    @classmethod
    def initialize(cls, test_instance: "TestInsertPaddingIR") -> None:
        cls.test_instance = test_instance

    @override
    def __call__(self, operations: list[Operation]) -> None:
        assert self.test_instance is not None
        super().__call__(operations)
        self.test_instance.captured_operations = list(operations)


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class TestInsertPaddingIR(unittest.TestCase):
    """IR-level structural tests for insert_padding_ir.

    Each test compiles a small matmul function, captures the operations list
    after CustomPreSchedulingPasses finishes (which includes insert_padding_ir),
    and asserts structural properties of the resulting operation sequence.
    """

    captured_operations: list[Operation] = []

    def setUp(self) -> None:
        torch.manual_seed(0xAFFE)
        self.patchers: list[Any] = []

        self.patchers.append(t_inductor_config.patch("force_disable_caches", True))
        self.patchers.append(ts_inductor_config.patch("sencores", 1))

        CustomPreSchedulingPassesWithCapture.initialize(self)
        self.patchers.append(
            patch.object(
                passes,
                "CustomPreSchedulingPasses",
                CustomPreSchedulingPassesWithCapture,
            )
        )

        for p in self.patchers:
            p.__enter__()

        torch.compiler.reset()

    def tearDown(self) -> None:
        for p in self.patchers:
            p.__exit__(None, None, None)
        torch.compiler.reset()

    def compile_and_capture(
        self,
        fn: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ) -> list[Operation]:
        """Compile ``fn`` with the given Spyre-device args and return the
        captured operations list after CustomPreSchedulingPasses."""
        self.captured_operations = []
        compiled = torch.compile(fn, fullgraph=True)
        compiled(*args)
        return self.captured_operations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matmul_ops(operations: list[Operation]) -> list[ComputedBuffer]:
        """Return all ComputedBuffer operations with BATCH_MATMUL_OP reduction type."""
        result = []
        for op in operations:
            if not isinstance(op, ComputedBuffer):
                continue
            data = op.data
            if isinstance(data, Reduction) and data.reduction_type == BATCH_MATMUL_OP:
                result.append(op)
        return result

    @staticmethod
    def _ops_before(
        operations: list[Operation], target: ComputedBuffer
    ) -> list[Operation]:
        """Return all operations that appear before ``target`` in the list."""
        idx = operations.index(target)
        return operations[:idx]

    @staticmethod
    def _empty_fallback_ops(operations: list[Operation]) -> list[SpyreEmptyFallback]:
        return [op for op in operations if isinstance(op, SpyreEmptyFallback)]

    @staticmethod
    def _overwrite_ops(ops: list[Operation]) -> list[ComputedBuffer]:
        """Return ComputedBuffers whose origin_node calls spyre.overwrite."""
        result = []
        for op in ops:
            if not isinstance(op, ComputedBuffer):
                continue
            origin = getattr(op, "origin_node", None)
            if origin is not None and hasattr(origin, "target"):
                if origin.target is torch.ops.spyre.overwrite.default:
                    result.append(op)
        return result

    @staticmethod
    def _constant_nodes(ops: list[Operation]) -> list[SpyreConstantFallback]:
        """Return SpyreConstantFallback ops (fill-value constants for padding)."""
        return [op for op in ops if isinstance(op, SpyreConstantFallback)]

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_mm_unaligned_k_pads(self) -> None:
        """2D mm with K=67 (unaligned) — padding ops are inserted before matmul."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        # 67 is not a multiple of stick_size (64), so padding should occur.
        assert 67 % stick_size != 0

        x = torch.randn(55, 67, dtype=dtype, device="spyre")
        w = torch.randn(67, 128, dtype=dtype, device="spyre")

        def fn(x, w):
            return x @ w

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one matmul op")
        mm = matmuls[0]

        # reduction_ranges should be updated to K_padded (next stick boundary).
        k_padded = ((67 + stick_size - 1) // stick_size) * stick_size
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(
            k_actual,
            k_padded,
            f"reduction_ranges not updated: {k_actual} != {k_padded}",
        )

        # There should be overwrite ops before the matmul.
        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        # Expect at least 2 overwrites: fill pad + copy original, for both x and y.
        self.assertGreaterEqual(
            len(overwrites), 2, "Expected at least 2 overwrite ops before matmul"
        )

    def test_mm_aligned_k_no_padding(self) -> None:
        """2D mm with K=128 (aligned) — no padding ops inserted."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 128 % stick_size == 0

        x = torch.randn(55, 128, dtype=dtype, device="spyre")
        w = torch.randn(128, 64, dtype=dtype, device="spyre")

        def fn(x, w):
            return x @ w

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one matmul op")
        mm = matmuls[0]

        # reduction_ranges should remain K=128.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(k_actual, 128, f"K should stay 128, got {k_actual}")

        # No overwrite ops should appear before the matmul.
        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        self.assertEqual(len(overwrites), 0, "Expected no overwrite ops for aligned K")

    def test_bmm_3d_unaligned_k_pads(self) -> None:
        """3D bmm (B,M,K)×(B,K,N) with K=67 — padding inserted before bmm."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x = torch.randn(2, 55, 67, dtype=dtype, device="spyre")
        w = torch.randn(2, 67, 128, dtype=dtype, device="spyre")

        def fn(x, w):
            return torch.bmm(x, w)

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one batched matmul op")
        mm = matmuls[0]

        k_padded = ((67 + stick_size - 1) // stick_size) * stick_size
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(k_actual, k_padded)

        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        self.assertGreaterEqual(len(overwrites), 2)

    def test_bmm_3d_2d_unaligned_k_pads(self) -> None:
        """3D×2D bmm: (B,M,K)×(K,N) with K=67 — padding on both x and y."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x = torch.randn(2, 55, 67, dtype=dtype, device="spyre")
        w = torch.randn(67, 128, dtype=dtype, device="spyre")

        def fn(x, w):
            return x @ w

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        k_padded = ((67 + stick_size - 1) // stick_size) * stick_size
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), k_padded)

        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        # 2 overwrites per argument = 4 total.
        self.assertGreaterEqual(len(overwrites), 4)

    def test_matmul_4d_unaligned_k_pads(self) -> None:
        """4D matmul (B,H,M,K)×(B,H,K,N) with K=67 — padding inserted."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x = torch.randn(2, 3, 55, 67, dtype=dtype, device="spyre")
        w = torch.randn(2, 3, 67, 128, dtype=dtype, device="spyre")

        def fn(x, w):
            return x @ w

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        k_padded = ((67 + stick_size - 1) // stick_size) * stick_size
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), k_padded)

        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        self.assertGreaterEqual(len(overwrites), 2)

    def test_einsum_mk_kn_mn_pads(self) -> None:
        """einsum('mk,kn->mn') with K=67 — x is 2D but inner_fn sees 3D via mm_to_bmm."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x = torch.randn(55, 67, dtype=dtype, device="spyre")
        w = torch.randn(67, 128, dtype=dtype, device="spyre")

        def fn(x, w):
            return torch.einsum("mk,kn->mn", x, w)

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        k_padded = ((67 + stick_size - 1) // stick_size) * stick_size
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), k_padded)

    def test_fill_cache_shared_across_same_dtype(self) -> None:
        """Two matmuls with the same shapes share the spyre.constant node via fill_cache.

        The fill_cache key is (fill_value, device, dtype).  All padding — x and y
        for both matmuls — uses fill_value=0.0 and the same device and dtype, so
        exactly one spyre.constant node is lowered and reused for all four pad
        sequences.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x = torch.randn(2, 55, 67, dtype=dtype, device="spyre")
        w1 = torch.randn(2, 67, 128, dtype=dtype, device="spyre")
        w2 = torch.randn(2, 67, 128, dtype=dtype, device="spyre")

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        ops = self.compile_and_capture(fn, (x, w1, w2))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 2, "Expected 2 matmul ops")

        # With fill_cache, the single (0.0, device, float16) key is shared across
        # all padding calls, so exactly 1 spyre.constant node is lowered total.
        constant_ops = self._constant_nodes(ops)
        self.assertEqual(
            len(constant_ops),
            1,
            f"Expected 1 spyre.constant (cache shared across all padding), got {len(constant_ops)}",
        )

    def test_origin_node_set_on_rebuilt_matmul(self) -> None:
        """Rebuilt matmul ComputedBuffer retains origin_node from the original.

        This is required by LX planning (scratchpad.py:298) which accesses
        op.origin_node.target._opname directly.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x = torch.randn(55, 67, dtype=dtype, device="spyre")
        w = torch.randn(67, 128, dtype=dtype, device="spyre")

        def fn(x, w):
            return x @ w

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        self.assertIsNotNone(
            mm.origin_node,
            "origin_node should not be None after _rebuild_matmul",
        )

    def test_padded_buffer_sizes_x_and_y(self) -> None:
        """Padded x and y buffers have the correct sizes with K_padded.

        spyre.empty lowers to SpyreEmptyFallback.  We identify the padded
        SpyreEmptyFallback ops by their tensor size.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        B, M, K, N = 2, 55, 67, 128
        k_padded = ((K + stick_size - 1) // stick_size) * stick_size

        x = torch.randn(B, M, K, dtype=dtype, device="spyre")
        w = torch.randn(B, K, N, dtype=dtype, device="spyre")

        def fn(x, w):
            return torch.bmm(x, w)

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        ops_before = self._ops_before(ops, mm)

        # Padded buffers are the SpyreEmptyFallback ops before the matmul whose
        # size matches either [B,M,k_padded] (x) or [B,k_padded,N] (y).
        empty_sizes = [
            [int(s) for s in op.get_size()]
            for op in ops_before
            if isinstance(op, SpyreEmptyFallback)
        ]

        expected_x = [B, M, k_padded]
        expected_y = [B, k_padded, N]
        self.assertIn(
            expected_x,
            empty_sizes,
            f"x padded size {expected_x} not found in {empty_sizes}",
        )
        self.assertIn(
            expected_y,
            empty_sizes,
            f"y padded size {expected_y} not found in {empty_sizes}",
        )

    def test_padded_buffer_preserves_stick_dimension(self) -> None:
        """Padded buffers have the same within-stick dimension as their originals.

        ``lower_pad_sequence`` derives the padded buffer's ``SpyreTensorLayout``
        from the original buffer's ``stride_map[-1]`` so that
        ``device_coordinates[-1]`` (the stick coordinate expression) is identical
        for both.  Concretely, ``stride_map[-1]`` must be the same for the padded
        ``SpyreEmptyFallback`` as it was for the original input buffer.

        This test covers three cases:
        - 2D mm: original [M, K], padded [M, K_padded] — K is the stick dim.
        - 3D bmm: original [B, M, K], padded [B, M, K_padded] — K is the stick dim.
        - einsum mk,kn→mn: x_buf is 2D [M, K] but the matmul inner_fn accesses it
          as 3D [1, M, K] via mm_to_bmm_pass; padded_size is [1, M, K_padded].
          The within-stick dim must be recovered from the 3D view's strides, not
          from x_buf's 2D ``stride_map``.

        In all cases the within-stick dimension is K (the last host dim), so
        ``stride_map[-1] == 1`` for every padded buffer.  The test would catch a
        regression that confused the stick dim (e.g. producing ``stride_map[-1] ==
        K_padded`` from a default layout constructed with the wrong dim_order).
        """
        from torch_spyre._inductor.ir import FixedTiledLayout

        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        cases: list[
            tuple[str, Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]
        ] = [
            (
                "mm [55,67]x[67,128]",
                lambda x, w: x @ w,
                (
                    torch.randn(55, 67, dtype=dtype, device="spyre"),
                    torch.randn(67, 128, dtype=dtype, device="spyre"),
                ),
            ),
            (
                "bmm [2,55,67]x[2,67,128]",
                lambda x, w: torch.bmm(x, w),
                (
                    torch.randn(2, 55, 67, dtype=dtype, device="spyre"),
                    torch.randn(2, 67, 128, dtype=dtype, device="spyre"),
                ),
            ),
            (
                "einsum mk,kn->mn [55,67]x[67,128]",
                lambda x, w: torch.einsum("mk,kn->mn", x, w),
                (
                    torch.randn(55, 67, dtype=dtype, device="spyre"),
                    torch.randn(67, 128, dtype=dtype, device="spyre"),
                ),
            ),
        ]

        for name, fn, args in cases:
            with self.subTest(case=name):
                ops = self.compile_and_capture(fn, args)
                matmuls = self._matmul_ops(ops)
                self.assertEqual(len(matmuls), 1, f"{name}: expected 1 matmul")
                mm = matmuls[0]
                ops_before = self._ops_before(ops, mm)

                padded_empties = [
                    op for op in ops_before if isinstance(op, SpyreEmptyFallback)
                ]
                self.assertGreaterEqual(
                    len(padded_empties),
                    2,
                    f"{name}: expected at least 2 padded buffers",
                )

                for empty in padded_empties:
                    layout = empty.get_layout()
                    self.assertIsInstance(
                        layout,
                        FixedTiledLayout,
                        f"{name}: padded buffer has wrong layout type {type(layout)}",
                    )
                    sm_last = int(list(layout.device_layout.stride_map)[-1])
                    self.assertEqual(
                        sm_last,
                        1,
                        f"{name}: padded buffer stride_map[-1]={sm_last}, "
                        f"expected 1 (K is within-stick dim); "
                        f"size={[int(s) for s in empty.get_size()]}",
                    )


if __name__ == "__main__":
    unittest.main()
