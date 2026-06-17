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

"""IR-level unit tests for insert_bmm_padding.

Tests hook into CustomPreSchedulingPasses after insert_bmm_padding runs to inspect
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
from torch._inductor.graph import GraphLowering

from torch_spyre._C import get_elem_in_stick
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.ir import FixedTiledLayout, SpyreConstantFallback
from torch_spyre._inductor.passes import CustomPreSchedulingPasses


Ts = TypeVarTuple("Ts")


# ---------------------------------------------------------------------------
# Hooks into CustomPreSchedulingPasses
# ---------------------------------------------------------------------------


class CustomPreSchedulingPassesWithCapture(CustomPreSchedulingPasses):
    """Subclass of CustomPreSchedulingPasses that captures the operations list
    after all built-in passes (including insert_bmm_padding) have run."""

    test_instance: Optional["TestInsertPaddingIR"] = None

    @classmethod
    def initialize(cls, test_instance: "TestInsertPaddingIR") -> None:
        cls.test_instance = test_instance

    @override
    def __call__(self, graph: GraphLowering) -> None:
        assert self.test_instance is not None
        super().__call__(graph)
        self.test_instance.captured_operations = list(graph.operations)


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class TestInsertPaddingIR(unittest.TestCase):
    """IR-level structural tests for insert_bmm_padding.

    Each test compiles a small matmul function, captures the operations list
    after CustomPreSchedulingPasses finishes (which includes insert_bmm_padding),
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

    def compile_and_run(
        self,
        fn: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ) -> tuple[list[Operation], torch.Tensor]:
        """Compile ``fn`` once, capture IR operations, and return (ops, result)."""
        self.captured_operations = []
        compiled = torch.compile(fn, fullgraph=True)
        result = compiled(*args)
        return self.captured_operations, result

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
    def _padded_buf_ops(operations: list[Operation]) -> list[ComputedBuffer]:
        """Return ComputedBuffer operations whose origin_node.target is aten.constant_pad_nd."""
        padded_buf_ops = []
        aten_op = torch.ops.aten.constant_pad_nd.default
        for op in operations:
            if isinstance(op, ComputedBuffer):
                if getattr(getattr(op, "origin_node", None), "target", None) == aten_op:
                    padded_buf_ops.append(op)
        return padded_buf_ops

    @staticmethod
    def _constant_ops(ops: list[Operation]) -> list[SpyreConstantFallback]:
        """Return SpyreConstantFallback ops (fill-value constants for padding)."""
        return [op for op in ops if isinstance(op, SpyreConstantFallback)]

    def _assert_constant_pad_nd_ops(self, ops: list[Operation]) -> None:
        """Assert the constant_pad_nd 4-op pattern."""

        # Expected 4 operations (all with FixedTiledLayout):
        # 1. ComputedBuffer: output buffer allocation
        # 2. SpyreConstantFallback: padding fill constant
        # 3. ComputedBuffer: fill padding region (reads constant, writes output)
        # 4. ComputedBuffer: copy input data (reads input, writes output)

        self.assertEqual(len(ops), 4, f"Expected 4 ops, got {len(ops)}")

        for op in ops:
            self.assertTrue(
                isinstance(op.get_layout(), FixedTiledLayout),
                f"{type(op).__name__} should have FixedTiledLayout",
            )

        computed_buffer_ops = [op for op in ops if isinstance(op, ComputedBuffer)]
        self.assertEqual(len(computed_buffer_ops), 3, "Expected 3 computed buffers")

        padded_buf_ops = next(iter(self._padded_buf_ops(computed_buffer_ops)), None)
        self.assertIsNotNone(padded_buf_ops, "Expected 1 output buffer")

        constant_op = next(iter(self._constant_ops(ops)), None)
        self.assertIsNotNone(constant_op, "Expected 1 constant buffer")

        mutation_ops = [op for op in computed_buffer_ops if op is not padded_buf_ops]
        self.assertEqual(len(mutation_ops), 2, "Expected 2 mutation ops")

        # Verify dependencies (mutation ops write to output buffer):
        # - Fill padding region: reads constant, writes output
        # - Copy input data: reads input, writes output

        for op in ops:
            self.assertTrue(
                padded_buf_ops.origin_node in op.origins,
                f"{op.name} origins should contain {padded_buf_ops.origin_node}",
            )
        for op in mutation_ops:
            self.assertTrue(
                op.get_layout() == padded_buf_ops.get_layout(),
                f"Mutation op {op.name} should have same layout as output buffer {padded_buf_ops.name}",
            )
        self.assertTrue(
            any(constant_op.name in op.get_read_names() for op in mutation_ops),
            "One mutation op should read from constant buffer",
        )
        self.assertTrue(
            any("arg" in name for op in mutation_ops for name in op.get_read_names()),
            "One mutation op should read from input buffer",
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_mm_unaligned_k_pads(self) -> None:
        """2D mm with K=67 (unaligned) — only y is padded before the matmul.

        x is untouched; reduction_ranges stays at K=67.
        y gets 2 overwrites (fill + copy).
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        # 67 is not a multiple of stick_size (64), so padding should occur.
        assert 67 % stick_size != 0

        x_cpu = torch.randn(55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one matmul op")
        mm = matmuls[0]

        # reduction_ranges stays at K (not K_padded): the K→K_padded extension
        # happens at SDSC codegen time, not in the IR.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        k_actual = int(reduction.reduction_ranges[0])
        self.assertEqual(
            k_actual,
            67,
            f"reduction_ranges should stay at K=67, got {k_actual}",
        )

        # 2 overwrite ops before the matmul: fill + copy for y only (x untouched).
        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_mm_aligned_k_no_padding(self) -> None:
        """2D mm with K=128 (aligned) — no padding ops inserted."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 128 % stick_size == 0

        x_cpu = torch.randn(55, 128, dtype=dtype)
        w_cpu = torch.randn(128, 64, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
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
        self.assertEqual(len(ops_before), 0, "Expected no padding ops for aligned K")

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_bmm_3d_unaligned_k_pads(self) -> None:
        """3D bmm (B,M,K)×(B,K,N) with K=67 — only y is padded before bmm."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 55, 67, dtype=dtype)
        w_cpu = torch.randn(2, 67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return torch.bmm(x, w)

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one batched matmul op")
        mm = matmuls[0]

        # reduction_ranges stays at K.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_bmm_3d_2d_unaligned_k_pads(self) -> None:
        """3D×2D bmm: (B,M,K)×(K,N) with K=67 — only y is padded."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        # reduction_ranges stays at K.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_matmul_4d_unaligned_k_pads(self) -> None:
        """4D matmul (B,H,M,K)×(B,H,K,N) with K=67 — only y is padded."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 3, 55, 67, dtype=dtype)
        w_cpu = torch.randn(2, 3, 67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        # reduction_ranges stays at K.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_einsum_mk_kn_mn_pads(self) -> None:
        """einsum('mk,kn->mn') with K=67 — y is padded; reduction_ranges stays at K."""
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return torch.einsum("mk,kn->mn", x, w)

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        # reduction_ranges stays at K=67.
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), 67)

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_padding_constants_deduped(self) -> None:
        """Two matmuls with the same shapes yield exactly one spyre.constant after dedup.

        Both matmuls pad only y with fill_value=0.0 at the same dtype, so two
        spyre.constant FX nodes are created (one per pad sequence) and lowered to two
        SpyreConstantFallback IR ops.  dedup_and_promote_constants then merges them into
        one canonical constant and moves it to the head of operations.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(2, 55, 67, dtype=dtype)
        w1_cpu = torch.randn(2, 67, 128, dtype=dtype)
        w2_cpu = torch.randn(2, 67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w1 = w1_cpu.to(device="spyre")
        w2 = w2_cpu.to(device="spyre")

        def fn(x, w1, w2):
            return torch.bmm(x, w1) + torch.bmm(x, w2)

        ops, result = self.compile_and_run(fn, (x, w1, w2))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 2, "Expected 2 matmul ops")

        # dedup_and_promote_constants merges all (0.0, fp16, spyre) constants into one.
        constant_ops = self._constant_ops(ops)
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

        torch.testing.assert_close(
            fn(x_cpu, w1_cpu, w2_cpu), result.cpu(), atol=0.1, rtol=0.1
        )

    def test_origin_node_set_on_rebuilt_matmul(self) -> None:
        """Rebuilt matmul ComputedBuffer retains origin_node from the original.

        This is required by LX planning (scratchpad.py:298) which accesses
        op.origin_node.target._opname directly.
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        x_cpu = torch.randn(55, 67, dtype=dtype)
        w_cpu = torch.randn(67, 128, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        self.assertIsNotNone(
            mm.origin_node,
            "origin_node should not be None after _rebuild_matmul",
        )

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_padded_buffer_sizes_y_only(self) -> None:
        """Only y is padded; x is untouched.  y_padded has host size [B, K_padded, N].

        For the 3D bmm shape [B=2, K=67, N=128]: SpyreTensorLayout for
        [B, K_padded, N] sticked on N lays out device dims as
        [K_padded, N_sticks, B, stick_size], so device_size[-4] == K_padded.

        Exactly one padded buffer op appears before the matmul.
        """

        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        B, M, K, N = 2, 55, 67, 128
        k_padded = ((K + stick_size - 1) // stick_size) * stick_size

        x_cpu = torch.randn(B, M, K, dtype=dtype)
        w_cpu = torch.randn(B, K, N, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return torch.bmm(x, w)

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]

        ops_before = self._ops_before(ops, mm)

        padded_buf_ops = self._padded_buf_ops(ops_before)
        # Only y is padded — exactly one padded buffer.
        self.assertEqual(
            len(padded_buf_ops),
            1,
            f"Expected 1 padded buffer (y only), found {len(padded_buf_ops)}: "
            f"{[[int(s) for s in op.get_size()] for op in padded_buf_ops]}",
        )

        y_padded_buf_ops = padded_buf_ops[0]

        # y_padded's host size is [B, K_padded, N]: lower_pad_sequence builds it at the
        # padded shape and no host-downgrade is applied.  The IR iteration space (via
        # reduction_ranges) stays at K; only the buffer allocation is K_padded.
        host_size = [int(s) for s in y_padded_buf_ops.get_size()]
        self.assertEqual(
            host_size,
            [B, k_padded, N],
            f"y_padded host size should be [B,k_padded,N]=[{B},{k_padded},{N}], "
            f"got {host_size}",
        )

        # y's device_layout.device_size must reflect K_padded in the K-row device dim.
        # For [B, K_padded, N] sticked on N: device_size = [K_padded, N_sticks, B, stick_size].
        # K_padded sits at device_size[-4] (index -stride_idx-2 with stride_idx=2 for K).
        layout = y_padded_buf_ops.get_layout()
        self.assertIsInstance(layout, FixedTiledLayout)
        assert isinstance(layout, FixedTiledLayout)
        dev_size = list(layout.device_layout.device_size)
        self.assertEqual(
            dev_size[-4],
            k_padded,
            f"y device_size[-4] should be K_padded={k_padded}, "
            f"got {dev_size[-4]}; full device_size={dev_size}",
        )

        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_padded_buffer_preserves_stick_dimension(self) -> None:
        """y's padded buffer preserves the original within-stick stride.

        ``lower_pad_sequence`` constructs the padded buffer's ``SpyreTensorLayout``
        from the padded host size/stride so that ``device_coordinates[-1]`` (the
        stick coordinate expression) is identical for both the original and padded
        buffers.  Concretely, ``stride_map[-1]`` must be 1 for the padded buffer.

        y is sticked on N (the output dim), which is contiguous, so
        ``stride_map[-1] == 1``.  The test catches a regression that confused the
        stick dim (e.g. producing ``stride_map[-1] == K_padded`` from a default
        layout with the wrong dim_order).
        """

        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        assert 67 % stick_size != 0

        def make_args(*shapes):
            cpu = tuple(torch.randn(*s, dtype=dtype) for s in shapes)
            dev = tuple(t.to(device="spyre") for t in cpu)
            return cpu, dev

        cpu0, dev0 = make_args((55, 67), (67, 128))
        cpu1, dev1 = make_args((2, 55, 67), (2, 67, 128))
        cpu2, dev2 = make_args((55, 67), (67, 128))

        cases: list[tuple[str, Callable[..., torch.Tensor], tuple, tuple]] = [
            ("mm [55,67]x[67,128]", lambda x, w: x @ w, dev0, cpu0),
            ("bmm [2,55,67]x[2,67,128]", lambda x, w: torch.bmm(x, w), dev1, cpu1),
            (
                "einsum mk,kn->mn [55,67]x[67,128]",
                lambda x, w: torch.einsum("mk,kn->mn", x, w),
                dev2,
                cpu2,
            ),
        ]

        for name, fn, args, cpu_args in cases:
            with self.subTest(case=name):
                ops, result = self.compile_and_run(fn, args)
                matmuls = self._matmul_ops(ops)
                self.assertEqual(len(matmuls), 1, f"{name}: expected 1 matmul")
                mm = matmuls[0]
                ops_before = self._ops_before(ops, mm)

                padded_buf_ops = self._padded_buf_ops(ops_before)
                self.assertEqual(
                    len(padded_buf_ops),
                    1,
                    f"{name}: expected exactly 1 padded buffer (y only)",
                )

                for op in padded_buf_ops:
                    layout = op.get_layout()
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
                        f"size={[int(s) for s in op.get_size()]}",
                    )

                torch.testing.assert_close(
                    fn(*cpu_args), result.cpu(), atol=0.1, rtol=0.1
                )

    def test_mm_square_unaligned_k_pads_y_only(self) -> None:
        """Square mm (M==K==N) with unaligned K — only y is padded, not x.

        Verifies that x/y identification works correctly when both inputs
        have the same shape (M==K==N).
        """
        dtype = torch.float16
        stick_size = get_elem_in_stick(dtype)
        K = 67
        assert K % stick_size != 0

        x_cpu = torch.randn(K, K, dtype=dtype)
        w_cpu = torch.randn(K, K, dtype=dtype)
        x = x_cpu.to(device="spyre")
        w = w_cpu.to(device="spyre")

        def fn(x, w):
            return x @ w

        ops, result = self.compile_and_run(fn, (x, w))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1)
        mm = matmuls[0]
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), K)
        ops_before = self._ops_before(ops, mm)
        self._assert_constant_pad_nd_ops(ops_before)
        torch.testing.assert_close(fn(x_cpu, w_cpu), result.cpu(), atol=0.1, rtol=0.1)

    def test_4d_matmul_xt_restickify_pads_y(self) -> None:
        """4D matmul where x is transposed (forcing restickify) and K is unaligned.

        x is stored as [B, H, K, M] and transposed to [B, H, M, K] for the
        matmul.  Restickify reorders x's device dims; insert_bmm_padding must
        preserve the original inner_fn's x loader unchanged so work_division
        sees valid device coordinates.  Regression test for that fix.
        """
        dtype = torch.float16
        B, H, M, K, N = 12, 32, 256, 4, 128

        x_cpu = torch.randn(B, H, K, M, dtype=dtype) * 0.01
        y_cpu = torch.randn(B, H, K, N, dtype=dtype) * 0.01
        x = x_cpu.to(device="spyre")
        y = y_cpu.to(device="spyre")

        def fn(x, y):
            return torch.matmul(x.transpose(-1, -2), y)

        ops, result = self.compile_and_run(fn, (x, y))
        matmuls = self._matmul_ops(ops)
        self.assertEqual(len(matmuls), 1, "Expected exactly one matmul op")
        mm = matmuls[0]
        reduction = mm.data
        assert isinstance(reduction, Reduction)
        self.assertEqual(int(reduction.reduction_ranges[0]), K)
        # ops_before contains a restickify op for x in addition to the 4-op
        # padding sequence for y, so _assert_constant_pad_nd_ops (which expects
        # exactly 4 ops) cannot be used here.  Correctness is verified by assert_close.
        torch.testing.assert_close(fn(x_cpu, y_cpu), result.cpu(), atol=0.1, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
