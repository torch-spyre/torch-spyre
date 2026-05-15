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
        """2D mm with K=67 (unaligned) — y is padded before the matmul; x is not."""
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
        # r_K = 0..K_padded-1; x's tail is handled by pt hardware masking.
        k_padded = ((67 + stick_size - 1) // stick_size) * stick_size
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(
            k_actual,
            k_padded,
            f"reduction_ranges should be K_padded={k_padded}, got {k_actual}",
        )

        # 2 overwrite ops before the matmul: fill + copy for y only.
        ops_before = self._ops_before(ops, mm)
        overwrites = self._overwrite_ops(ops_before)
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
        """3D bmm (B,M,K)×(B,K,N) with K=67 — y is padded before bmm; x is not."""
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
        self.assertGreaterEqual(len(overwrites), 2)

    def test_bmm_3d_2d_unaligned_k_pads(self) -> None:
        """3D×2D bmm: (B,M,K)×(K,N) with K=67 — y is padded; x is not."""
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
        # 2 overwrites: fill + copy for y only.
        self.assertGreaterEqual(len(overwrites), 2)

    def test_matmul_4d_unaligned_k_pads(self) -> None:
        """4D matmul (B,H,M,K)×(B,H,K,N) with K=67 — y is padded; x is not."""
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
        """einsum('mk,kn->mn') with K=67 — y is padded to K_padded; x is not."""
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

    def test_fill_cache_shared_across_same_dtype(self) -> None:
        """Two matmuls with the same shapes share the spyre.constant node via fill_cache.

        The fill_cache key is (fill_value, device, dtype).  Both x and y are padded per
        matmul, but all pad operations use fill_value=0.0 and the same dtype, so exactly
        one spyre.constant node is lowered and reused across all pad sequences.
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

    def test_padded_buffer_sizes_y_only(self) -> None:
        """Only y is padded; its host K-dim is extended to k_padded.

        spyre.empty lowers to SpyreEmptyFallback.  Exactly one SpyreEmptyFallback
        op appears before the matmul: y_padded with host size [B, K_padded, N].
        x is not padded — neither allocated nor copied; it is consumed via its
        original TensorBox and the pt hardware is intended to mask its K tail.
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
        # Only y is padded — exactly one SpyreEmptyFallback op.
        self.assertEqual(
            len(padded_empties),
            1,
            f"Expected 1 padded buffer (y only), found {len(padded_empties)}: "
            f"{[[int(s) for s in op.get_size()] for op in padded_empties]}",
        )

        # y_padded: [B, K_padded, N].
        host_size = [int(s) for s in padded_empties[0].get_size()]
        self.assertEqual(
            host_size,
            [B, k_padded, N],
            f"y_padded size should be [{B},{k_padded},{N}], got {host_size}",
        )

    def test_x_layout_override_stride_map(self) -> None:
        """The matmul's op_info carries an x_layout_override with K_padded as the M-row stride.

        insert_padding_ir stores a (x_name, FixedTiledLayout) pair on the matmul's
        op_info under "x_layout_override".  The FixedTiledLayout has K_padded as the
        M-row host stride and an overlay SpyreTensorLayout (stride_map[M_dim]=K_padded).
        SpyreKernel.load() uses this override so the index expression and stride_map
        are consistent, preventing the spurious r_K//K overflow term.
        """
        from torch_spyre._inductor.ir import FixedTiledLayout

        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        K = 67
        assert K % stick_size != 0
        k_padded = ((K + stick_size - 1) // stick_size) * stick_size

        x = torch.randn(55, K, dtype=dtype, device="spyre")
        w = torch.randn(K, 128, dtype=dtype, device="spyre")

        def fn(x, w):
            return x @ w

        ops = self.compile_and_capture(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        reduction = mm.data
        assert isinstance(reduction, Reduction)
        op_info = getattr(reduction, "op_info", None)
        self.assertIsNotNone(op_info, "matmul op_info should not be None after padding")
        override = op_info.get("x_layout_override")
        self.assertIsNotNone(override, "op_info should contain 'x_layout_override'")
        override_name, override_layout = override
        self.assertIsInstance(override_name, str)
        self.assertIsInstance(override_layout, FixedTiledLayout)

        # stride_map should contain K_padded (the M-row stride).
        stride_map = list(override_layout.device_layout.stride_map)
        self.assertIn(
            k_padded,
            stride_map,
            f"x_layout_override.device_layout.stride_map should contain K_padded={k_padded}; "
            f"got {stride_map}",
        )

        # Host strides should use K_padded as the M-row stride.
        host_stride = [int(s) for s in override_layout.stride]
        self.assertIn(
            k_padded,
            host_stride,
            f"x_layout_override.stride should contain K_padded={k_padded}; "
            f"got {host_stride}",
        )

    def test_padded_buffer_preserves_stick_dimension(self) -> None:
        """y's padded buffer preserves the original within-stick stride.

        ``lower_pad_sequence`` constructs the padded buffer's ``SpyreTensorLayout``
        from the padded host size/stride so that ``device_coordinates[-1]`` (the
        stick coordinate expression) is identical for both the original and padded
        buffers.  Concretely, ``stride_map[-1]`` must be 1 for the padded
        ``SpyreEmptyFallback``.

        y is sticked on N (the output dim) with a contiguous within-stick stride,
        so ``stride_map[-1] == 1``.  The test catches a regression that confused
        the stick dim (e.g. producing ``stride_map[-1] == K_padded`` from a default
        layout with the wrong dim_order).
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
                    1,
                    f"{name}: expected exactly 1 padded buffer (y only)",
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
