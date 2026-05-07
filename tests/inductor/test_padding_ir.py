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
    MultiOutput,
    Operation,
    Reduction,
)

from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
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
    def _multioutput_ops(operations: list[Operation]) -> list[MultiOutput]:
        return [op for op in operations if isinstance(op, MultiOutput)]

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
    def _full_nodes(ops: list[Operation]) -> list[MultiOutput]:
        """Return MultiOutput ops whose origin_node comes from a spyre.full node."""
        result = []
        for op in ops:
            if not isinstance(op, MultiOutput):
                continue
            origin = getattr(op, "origin_node", None)
            if origin is not None and hasattr(origin, "target"):
                if origin.target is torch.ops.spyre.full.default:
                    result.append(op)
        return result

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
        """Two matmuls with the same shapes share spyre.full nodes via fill_cache.

        With K=67 and N=128 for both matmuls, each pad sequence (x and y) has
        the same one_stick_size cache key.  The second matmul's padding reuses
        the same spyre.full nodes as the first, so the total spyre.full count
        equals the unique (one_stick_size, device, dtype) combinations — not
        2 × (per-matmul count).

        For 3D bmm with K=67→128 and N=128:
          x padding: one_stick_size = [1, 1, stick_size]
          y padding: one_stick_size = [1, 1, N]  (= [1, 1, 128])
        These are two distinct cache keys, so 2 spyre.full nodes total (both
        reused by the second matmul).
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

        # Count spyre.full MultiOutput ops.  With fill_cache, each unique
        # one_stick_size key is lowered once.  For this 3D bmm the two keys are
        # [1, 1, stick_size] and [1, 1, N=128], so expect exactly 2 full nodes —
        # not 4 (2 per matmul without caching).
        full_ops = self._full_nodes(ops)
        self.assertEqual(
            len(full_ops),
            2,
            f"Expected 2 spyre.full (one per unique shape, cache shared), got {len(full_ops)}",
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

        spyre.empty lowers to FallbackKernel + MultiOutput.  The MultiOutput's
        origin_node is overwritten by subsequent overwrite ops (because the same
        empty_tb is returned from run_node for those calls), so we identify the
        padded MultiOutputs by their tensor size rather than origin_node target.
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

        # Padded buffers are the MultiOutput ops before the matmul whose size
        # matches either [B,M,k_padded] (x) or [B,k_padded,N] (y).
        mo_sizes = [
            [int(s) for s in op.get_size()]
            for op in ops_before
            if isinstance(op, MultiOutput)
        ]

        expected_x = [B, M, k_padded]
        expected_y = [B, k_padded, N]
        self.assertIn(
            expected_x, mo_sizes, f"x padded size {expected_x} not found in {mo_sizes}"
        )
        self.assertIn(
            expected_y, mo_sizes, f"y padded size {expected_y} not found in {mo_sizes}"
        )


if __name__ == "__main__":
    unittest.main()
