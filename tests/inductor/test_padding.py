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
        """2D mm with K=67 (unaligned) — x and y are both padded before the matmul."""
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

        # reduction_ranges is updated to K_padded so the hardware iterates
        # r_K = 0..K_padded-1; the pad region of x and y is zero-filled.
        k_padded = ((67 + stick_size - 1) // stick_size) * stick_size
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(
            k_actual,
            k_padded,
            f"reduction_ranges should be K_padded={k_padded}, got {k_actual}",
        )

        # 4 overwrite ops before the matmul: fill + copy for x, fill + copy for y.
        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        self.assertGreaterEqual(
            len(overwrites), 4, "Expected at least 4 overwrite ops before matmul"
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
        """3D bmm (B,M,K)×(B,K,N) with K=67 — both x and y are padded before bmm."""
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
        # reduction_ranges is updated to K_padded.
        self.assertEqual(int(reduction.reduction_ranges[0]), k_padded)

        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        self.assertGreaterEqual(len(overwrites), 4)

    def test_bmm_3d_2d_unaligned_k_pads(self) -> None:
        """3D×2D bmm: (B,M,K)×(K,N) with K=67 — both x and y are padded."""
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
        # reduction_ranges is updated to K_padded.
        self.assertEqual(int(reduction.reduction_ranges[0]), k_padded)

        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
        # 4 overwrites: fill + copy for x, fill + copy for y.
        self.assertGreaterEqual(len(overwrites), 4)

    def test_matmul_4d_unaligned_k_pads(self) -> None:
        """4D matmul (B,H,M,K)×(B,H,K,N) with K=67 — both x and y are padded."""
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
        self.assertGreaterEqual(len(overwrites), 4)

    def test_einsum_mk_kn_mn_pads(self) -> None:
        """einsum('mk,kn->mn') with K=67 — both x and y are padded to K_padded."""
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
        # reduction_ranges is updated to K_padded.
        self.assertEqual(int(reduction.reduction_ranges[0]), k_padded)

    def test_padding_constants_deduped(self) -> None:
        """Two matmuls with the same shapes yield exactly one spyre.constant after dedup.

        Both matmuls pad x and y with fill_value=0.0 at the same dtype, so four
        spyre.constant FX nodes are created (one per pad sequence) and lowered to four
        SpyreConstantFallback IR ops.  dedup_and_promote_constants then merges them into
        one canonical constant and moves it to the head of operations.
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

        # dedup_and_promote_constants merges all (0.0, fp16, spyre) constants into one.
        constant_ops = self._constant_nodes(ops)
        self.assertEqual(
            len(constant_ops),
            1,
            f"Expected 1 spyre.constant after IR dedup, got {len(constant_ops)}",
        )

        # The surviving constant must be at the head of operations.
        self.assertIs(
            ops[0],
            constant_ops[0],
            "Expected the surviving spyre.constant to be the first operation",
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
        """Both x and y are padded; host K-dims are extended to k_padded.

        spyre.empty lowers to SpyreEmptyFallback.  Two SpyreEmptyFallback ops
        appear before the matmul: one for x_padded with host size [B, M, K_padded]
        and one for y_padded with host size [B, K_padded, N].
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

        padded_empties = [op for op in ops_before if isinstance(op, SpyreEmptyFallback)]
        # Both x and y are padded — exactly two SpyreEmptyFallback ops.
        self.assertEqual(
            len(padded_empties),
            2,
            f"Expected 2 padded buffers (x and y), found {len(padded_empties)}: "
            f"{[[int(s) for s in op.get_size()] for op in padded_empties]}",
        )

        host_sizes = sorted([int(s) for s in op.get_size()] for op in padded_empties)
        # x_padded: [B, M, K_padded]; y_padded: [B, K_padded, N].
        self.assertIn(
            [B, M, k_padded],
            host_sizes,
            f"x_padded size [B,M,K_padded]=[{B},{M},{k_padded}] not found in {host_sizes}",
        )
        self.assertIn(
            [B, k_padded, N],
            host_sizes,
            f"y_padded size [B,K_padded,N]=[{B},{k_padded},{N}] not found in {host_sizes}",
        )

        # Each padded buffer's host K-dim must equal K_padded.
        for empty in padded_empties:
            host_size = [int(s) for s in empty.get_size()]
            self.assertIn(
                k_padded,
                host_size,
                f"k_padded={k_padded} not found in host_size={host_size}",
            )

    def test_padded_buffer_preserves_stick_dimension(self) -> None:
        """Both padded buffers (x and y) preserve the original within-stick stride.

        ``lower_pad_sequence`` constructs each padded buffer's ``SpyreTensorLayout``
        from the padded host size/stride so that ``device_coordinates[-1]`` (the
        stick coordinate expression) is identical for both the original and padded
        buffers.  Concretely, ``stride_map[-1]`` must be 1 for every padded
        ``SpyreEmptyFallback``.

        x is sticked on K (the reduction dim), y is sticked on N (the output dim).
        Both have contiguous within-stick strides, so ``stride_map[-1] == 1``.
        The test catches a regression that confused the stick dim (e.g. producing
        ``stride_map[-1] == K_padded`` from a default layout with the wrong dim_order).
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
                self.assertEqual(
                    len(padded_empties),
                    2,
                    f"{name}: expected exactly 2 padded buffers (x and y)",
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
                        f"expected 1 (within-stick dim is contiguous); "
                        f"size={[int(s) for s in empty.get_size()]}",
                    )


if __name__ == "__main__":
    unittest.main()
