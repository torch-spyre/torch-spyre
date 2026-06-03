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

"""End-to-end compilation tests for the coarse-tiling loop IR.

These tests drive the full Spyre compilation pipeline (CustomPreSchedulingPasses
→ scheduler → SpyreKernel codegen) and inspect the generated Python wrapper
source to verify that LoopSpec entries appear when coarse tiling is active.

No Spyre hardware is required: torch.compile() exercises the full codegen path
and run_and_get_code() captures the generated source without executing on device.
launch_kernel is mocked to prevent actual device execution.

Test classes
------------
- TestCoarseTileGroupsFnResidual: tests using coarse_tiling_groups_fn that
  have not yet been ported to spyre_hints (tiled_dims and baseline tests).
- TestCoarseTileSpyreHints: the primary test class; all new coarse-tiling
  tests should be added here using the spyre_hint API.
"""

import sys
import os

import sympy
import torch
import unittest
from unittest.mock import patch as mock_patch

from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.ir import ComputedBuffer, Operation

from torch_spyre._inductor import config
import torch_spyre._inductor.propagate_named_dims as _pnd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils_inductor import compare_with_cpu  # noqa: E402

# Path to mock for disabling actual device kernel execution.
_LAUNCH_KERNEL = "torch_spyre.execution.kernel_runner.launch_kernel"


# ---------------------------------------------------------------------------
# Module-level groups-function helpers
# (must be module-level so they are picklable by the Inductor cache machinery)
# ---------------------------------------------------------------------------


def _groups_all_k4(operations: list[Operation]):
    """One group: all ComputedBuffers, loop count K=4."""
    ops = [op for op in operations if isinstance(op, ComputedBuffer)]
    return [(ops, sympy.Integer(4))] if ops else []


def _groups_nested_k2_m4(operations: list[Operation]):
    """One group: all ops share nested K=2 outer (dim 0) / M=4 inner (dim 1) loops."""
    ops = [op for op in operations if isinstance(op, ComputedBuffer)]
    if not ops:
        return []
    return [(ops, [(0, sympy.Integer(2), [0]), (0, sympy.Integer(4), [1])])]


def _groups_per_op_tiled_dim(operations: list[Operation]):
    """Two groups each tiling a different iteration-space dimension.

    Group 0: first ComputedBuffer, K=4, tiled_dims=[0] (tile dim 0, the default).
    Group 1: second ComputedBuffer, K=4, tiled_dims=[0, 1] (tile both dims 0 and 1,
             exercises the per-group tiled_dims path).
    """
    ops = [op for op in operations if isinstance(op, ComputedBuffer)]
    groups = []
    if ops[:1]:
        groups.append((ops[:1], sympy.Integer(4)))  # default: tile dim 0
    if ops[1:]:
        groups.append(
            (ops[1:], sympy.Integer(4), [0, 1])
        )  # override: tile dims 0 and 1
    return groups


# ===========================================================================
# Residual coarse_tiling_groups_fn tests
#
# These tests exercise capabilities not yet expressible via spyre_hints:
#   - test_per_group_tiled_dims: per-group tiled_dims=[0, 1] override
#     (multi-dim flat tiling in one group, not a nested loop nest)
#
# test_no_tiling_baseline and the unrolled-loop tests are kept here until
# the groups_fn config is removed; their hint-based equivalents already
# exist in TestCoarseTileSpyreHints.
#
# See PR 2 for the tiled_dims extension to spyre_hints; PR 3 removes this
# class and coarse_tiling_groups_fn entirely.
# ===========================================================================


class TestCoarseTileGroupsFnResidual(InductorTestCase):
    """Residual tests using coarse_tiling_groups_fn.

    Kept until spyre_hints covers all scenarios.  See class docstring above.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    # ------------------------------------------------------------------
    # Baseline: no tiling flag → LoopSpec must NOT appear
    # ------------------------------------------------------------------

    def test_no_tiling_baseline(self):
        x = torch.randn(256, 128, dtype=torch.float16).to("spyre")

        def fn(x):
            return torch.abs(x)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x)
        self.assertTrue(len(source_codes) > 0)
        # LoopSpec appears as an import even without tiling; check for a call.
        self.assertNotIn("LoopSpec(", source_codes[0])

    # ------------------------------------------------------------------
    # generate_bundle receives LoopSpec in the spec tree
    # ------------------------------------------------------------------

    # test_generate_bundle_receives_loop_spec is disabled: the torch.compile
    # cache (AOT autograd / fxgraph) is poisoned by earlier tests in the same
    # session that call generate_bundle directly, causing generate_bundle to be
    # bypassed on a cache hit.  The essential coverage — that generate_bundle
    # handles a LoopSpec and emits affine.apply — is provided by
    # TestCompileOpSpecSymbolMapping.test_generate_bundle_emits_affine_apply_for_tiled_loop
    # in test_sdsc_tiled_address.py.
    #
    # @config.patch({"coarse_tiling": True, "coarse_tiling_groups_fn": _groups_all_k4})
    # def test_generate_bundle_receives_loop_spec(self): ...

    # ------------------------------------------------------------------
    # Per-group tiled_dims: two ops tiling different iteration dimensions
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_per_op_tiled_dim,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_per_group_tiled_dims(self):
        """Two ops in separate groups tile different iteration-space dimensions.

        op_a = abs(x): 2-D iteration space [B, D].
          Group 0 uses the default tiled_dims (None → tile dim 0).
          After tiling K=4: iteration space [B/4, D].

        op_b = neg(x.T): operates on a transposed view so its natural
          iteration space is also [B, D] but logically "D-major".
          Group 1 uses tiled_dims=[0, 1] (tile both dims 0 and 1).
          After tiling K=4: iteration space [B/4, D/4].

        Both groups should produce separate LoopSpec(count=sympify('4'))
        entries in the generated source, confirming that each group's
        tiled_dims was applied independently.
        """
        B, D = 256, 128
        x = torch.randn(B, D, dtype=torch.float16).to("spyre")
        y = torch.randn(B, D, dtype=torch.float16).to("spyre")

        def fn(x, y):
            return torch.abs(x), torch.neg(y)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x, y)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]

        # Both groups produce a LoopSpec with count=4.
        loop_spec_count = src.count("LoopSpec(")
        self.assertGreaterEqual(
            loop_spec_count,
            2,
            f"Expected ≥2 LoopSpec entries (one per group), "
            f"got {loop_spec_count}\n\nSource:\n{src}",
        )
        self.assertIn(
            "sympify('4')",
            src,
            "Expected loop count 4 in generated source",
        )

    # ------------------------------------------------------------------
    # Unrolled loop execution tests (unroll_loops=True)
    # Source inspection: unrolling passes LoopSpec through async_compile
    # with concrete per-iteration HBM addresses in each unrolled OpSpec.
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_nested_k2_m4,
            "unroll_loops": True,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_unrolled_source_calls_sdsc(self):
        """Nested K=2 × M=4 loop with unroll_loops=True compiles cleanly.

        The generated wrapper passes a LoopSpec to async_compile.sdsc().
        SpyreAsyncCompile.sdsc() calls unroll_loop_specs internally, replacing
        the LoopSpec with K_outer × K_inner = 8 flat copies per op before
        invoking generate_bundle.  The source must still contain LoopSpec (it's
        part of the sdsc() call-site), and subprocess.run must be called (the
        dxp_standalone backend invocation after successful unrolling+bundling).
        """
        a = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
        b = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
        c = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")

        def fn(a, b, c):
            y = a + b
            z = y * c
            return z

        cfn = torch.compile(fn)
        subprocess_calls = []

        def _record_subprocess(*args, **kwargs):
            subprocess_calls.append(args)

        with (
            mock_patch(_LAUNCH_KERNEL),
            mock_patch("subprocess.run", side_effect=_record_subprocess),
        ):
            _, source_codes = run_and_get_code(cfn, a, b, c)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        # The generated source passes a LoopSpec to async_compile.sdsc().
        self.assertIn("LoopSpec(", src)
        # subprocess.run was called — unroll_loop_specs + generate_bundle
        # completed without error before invoking dxp_standalone.
        self.assertTrue(
            len(subprocess_calls) > 0,
            "Expected subprocess.run to be called (dxp_standalone invocation)",
        )

    # ------------------------------------------------------------------
    # Softmax-shaped tiling: reduction + pointwise, unrolled.
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_all_k4,
            "unroll_loops": True,
            "sencores": 1,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_unrolled_softmax_shaped_execution(self):
        """Unrolled K=4 tiling of a softmax-shaped pointwise+reduce chain.

        Tiles the batch dimension (dim 0) of a softmax-like computation:
          max_val = x.amax(dim=-1, keepdim=True)   # reduction (non-tiled dim)
          x_shifted = x - max_val                   # pointwise broadcast
          exp_x = x_shifted.exp()                   # pointwise
          sum_exp = exp_x.sum(dim=-1, keepdim=True) # reduction (non-tiled dim)
          out = exp_x / sum_exp                     # pointwise broadcast

        The reductions collapse dim 1 (the D dimension); the loop tiles dim 0
        (the B dimension), so no tiled dim overlaps with the reduction dim.
        insert_tiling_propagation propagates the reduction outputs to full-sized
        buffers so outside consumers (later pointwise stages) see the complete
        batch.  sencores=1 avoids core-division/scratchpad issues.
        """
        B, D = 256, 64
        x = torch.randn(B, D, dtype=torch.float16)

        def softmax_fn(x):
            max_val = x.amax(dim=-1, keepdim=True)
            x_shifted = x - max_val
            exp_x = x_shifted.exp()
            sum_exp = exp_x.sum(dim=-1, keepdim=True)
            return exp_x / sum_exp

        compare_with_cpu(
            softmax_fn,
            x,
            run_compile=True,
            run_eager=False,
            atol=0.1,
            rtol=0.1,
        )


# ===========================================================================
# spyre_hint-driven coarse tiling
# These tests verify that coarse tiling is driven automatically by
# spyre_hint(num_tiles_per_dim=...) annotations, without requiring a
# coarse_tiling_groups_fn.  Named tensor dimensions must be declared and
# annotated on device tensors for the hint resolver to map dimension names
# to loop variables.
# ===========================================================================


_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


class TestCoarseTileSpyreHints(InductorTestCase):
    """Coarse tiling driven by spyre_hint(num_tiles_per_dim=...) annotations."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    # ------------------------------------------------------------------
    # Baseline: no hints -> no tiling
    # ------------------------------------------------------------------

    def test_hint_no_tiling_baseline(self):
        """Without spyre_hint annotations, coarse tiling must not fire."""
        x = torch.randn(256, 128, dtype=torch.float16).to("spyre")

        def fn(x):
            return torch.abs(x)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x)
        self.assertTrue(len(source_codes) > 0)
        # LoopSpec appears as an import even without tiling; check for a call.
        self.assertNotIn("LoopSpec(", source_codes[0])

    # ------------------------------------------------------------------
    # Single pointwise op
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_single_group_pointwise(self):
        """spyre_hint(num_tiles_per_dim={"A": 4}) tiles a pointwise abs into 4 iterations."""
        from torch_spyre._inductor import spyre_hint

        # 256 rows × 128 cols.  Tiling the outermost dim by 4 → 64 rows/iter.
        A, B = 256, 128
        x = torch.randn(A, B, dtype=torch.float16)

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"A": 4}):
                return torch.abs(x)

        x_dev = x.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(x_dev, ["A", "B"])

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec call in generated source")
        self.assertIn(
            "sympify('4')",
            src,
            "Expected loop count 4 in generated source",
        )

    # ------------------------------------------------------------------
    # Softmax-shaped chain (pointwise-reduce-pointwise)
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_softmax_shaped(self):
        """Tile the pointwise-reduce-pointwise stages of a softmax-like kernel.

        softmax(x, dim=-1) lowers to roughly:
          max_val = x.amax(dim=-1, keepdim=True)   # reduction
          x_shifted = x - max_val                   # pointwise broadcast sub
          exp_x = x_shifted.exp()                   # pointwise
          sum_exp = exp_x.sum(dim=-1, keepdim=True) # reduction
          out = exp_x / sum_exp                     # pointwise broadcast div

        All stages share the batch (row) dimension B.  Tiling over that
        dimension by K=4 means each loop iteration processes B/K rows.
        """
        from torch_spyre._inductor import spyre_hint

        B, D = 256, 128  # batch = 256 rows, each of length 128
        x = torch.randn(B, D, dtype=torch.float16)

        def softmax_fn(x):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                max_val = x.amax(dim=-1, keepdim=True)
                x_shifted = x - max_val
                exp_x = x_shifted.exp()
                sum_exp = exp_x.sum(dim=-1, keepdim=True)
                return exp_x / sum_exp

        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        cfn = torch.compile(softmax_fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "LoopSpec(",
            src,
            "Expected LoopSpec call in generated source for softmax-shaped fn",
        )
        self.assertIn(
            "sympify('4')",
            src,
            "Expected loop count 4 in generated softmax source",
        )

    # ------------------------------------------------------------------
    # Nested hints: outer K=2, inner M=4 on a single op
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_nested_loop_two_dims(self):
        """Nested spyre_hint scopes produce a two-level tiling loop.

        Input shape [1024, 4096]: outer hint tiles dim A by 2 (512 rows/iter),
        inner hint tiles dim B by 4 (1024 cols/iter).  Both ops (add and mul)
        share the nested LoopSpec.  Generated source must contain two LoopSpec
        entries with counts 2 and 4.
        """
        from torch_spyre._inductor import spyre_hint

        A, B = 1024, 4096
        a = torch.randn(A, B, dtype=torch.float16)
        b = torch.randn(A, B, dtype=torch.float16)
        c = torch.randn(A, B, dtype=torch.float16)

        def fn(a, b, c):
            with spyre_hint(num_tiles_per_dim={"A": 2}):
                with spyre_hint(num_tiles_per_dim={"B": 4}):
                    y = a + b
                    z = y * c
                    return z

        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        c_dev = c.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(a_dev, ["A", "B"])
        _name_tensor_dims(b_dev, ["A", "B"])
        _name_tensor_dims(c_dev, ["A", "B"])

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev, c_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('2')", src, "Expected outer loop count 2")
        self.assertIn("sympify('4')", src, "Expected inner loop count 4")
        # The nested LoopSpec must appear inside another LoopSpec.
        self.assertGreaterEqual(
            src.count("LoopSpec("),
            2,
            f"Expected ≥2 LoopSpec entries for nested loops\n\nSource:\n{src}",
        )

    # ------------------------------------------------------------------
    # Scratchpad (LX) allocation for intermediate tiled buffer — hint syntax
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_nested_loop_with_scratchpad(self):
        """Design-doc small example: y=a+b; z=y*c with nested K=2×M=4 hints.

        This is the canonical spyre_hint(num_tiles_per_dim=...) version of the
        small example from docs/source/compiler/coarse_tiling_loops.md.

        Shape [1024, 4096], outer hint tiles A-dim by 2 (512 rows/iter),
        inner hint tiles B-dim by 4 (1024 cols/iter).  With lx_planning
        enabled, the intermediate result y=a+b is allocated to LX scratchpad
        (it is only consumed within the loop body); the final output z stays
        in HBM.

        Assertions:
        - LoopSpec entries are emitted (tiling is active).
        - At least one TensorArg carries allocation={'lx': ...}.
        - The output buffer allocation uses 'hbm'.
        - The per-tile sizes 512 and 1024 appear in the generated source.
        """
        from torch_spyre._inductor import spyre_hint

        A, B = 1024, 4096
        a = torch.randn(A, B, dtype=torch.float16)
        b = torch.randn(A, B, dtype=torch.float16)
        c = torch.randn(A, B, dtype=torch.float16)

        def fn(a, b, c):
            with spyre_hint(num_tiles_per_dim={"A": 2}):
                with spyre_hint(num_tiles_per_dim={"B": 4}):
                    y = a + b
                    z = y * c
                    return z

        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        c_dev = c.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(a_dev, ["A", "B"])
        _name_tensor_dims(b_dev, ["A", "B"])
        _name_tensor_dims(c_dev, ["A", "B"])

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev, c_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('2')", src, "Expected outer loop count 2")
        self.assertIn("sympify('4')", src, "Expected inner loop count 4")
        self.assertGreaterEqual(
            src.count("LoopSpec("),
            2,
            f"Expected ≥2 LoopSpec entries for nested loops\n\nSource:\n{src}",
        )
        self.assertIn(
            "allocation={'lx'",
            src,
            "Expected intermediate TensorArg with lx allocation",
        )
        self.assertIn(
            "allocation={'hbm'",
            src,
            "Expected output TensorArg with hbm allocation",
        )
        # Per-tile shape: K=2 over 1024 rows → 512 rows/tile;
        # M=4 over 4096 cols → 1024 cols/tile.
        self.assertIn("512", src, "Expected per-tile row count 512")
        self.assertIn("1024", src, "Expected per-tile col count 1024")

    # ------------------------------------------------------------------
    # Unrolled nested loops via hints: source calls sdsc
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": True,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_unrolled_source_calls_sdsc(self):
        """Nested K=2 × M=4 hint tiling with unroll_loops=True compiles cleanly.

        The generated wrapper passes a LoopSpec to async_compile.sdsc().
        SpyreAsyncCompile.sdsc() calls unroll_loop_specs internally before
        invoking generate_bundle.  The source must still contain LoopSpec (it
        is part of the sdsc() call-site), and subprocess.run must be called
        (the dxp_standalone invocation after successful unrolling+bundling).
        """
        from torch_spyre._inductor import spyre_hint

        A, B = 1024, 4096
        a = torch.randn(A, B, dtype=torch.float16)
        b = torch.randn(A, B, dtype=torch.float16)
        c = torch.randn(A, B, dtype=torch.float16)

        def fn(a, b, c):
            with spyre_hint(num_tiles_per_dim={"A": 2}):
                with spyre_hint(num_tiles_per_dim={"B": 4}):
                    y = a + b
                    z = y * c
                    return z

        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        c_dev = c.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(a_dev, ["A", "B"])
        _name_tensor_dims(b_dev, ["A", "B"])
        _name_tensor_dims(c_dev, ["A", "B"])

        cfn = torch.compile(fn)
        subprocess_calls = []

        def _record_subprocess(*args, **kwargs):
            subprocess_calls.append(args)

        with (
            mock_patch(_LAUNCH_KERNEL),
            mock_patch("subprocess.run", side_effect=_record_subprocess),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev, c_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src)
        self.assertTrue(
            len(subprocess_calls) > 0,
            "Expected subprocess.run to be called (dxp_standalone invocation)",
        )

    # ------------------------------------------------------------------
    # Unrolled softmax-shaped execution via hints
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "unroll_loops": True,
            "sencores": 1,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_unrolled_softmax_shaped_execution(self):
        """Unrolled K=4 hint-tiled softmax-shaped pointwise+reduce chain.

        Tiles the batch dimension (dim 0) of a softmax-like computation using
        spyre_hint(num_tiles_per_dim={"B": 4}).  sencores=1 avoids
        core-division issues.  The reductions collapse dim 1 (D); the loop
        tiles dim 0 (B), so no tiled dim overlaps with the reduction dim.
        """
        from torch_spyre._inductor import spyre_hint

        B, D = 256, 64
        x = torch.randn(B, D, dtype=torch.float16)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x, ["B", "D"])

        def softmax_fn(x):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                max_val = x.amax(dim=-1, keepdim=True)
                x_shifted = x - max_val
                exp_x = x_shifted.exp()
                sum_exp = exp_x.sum(dim=-1, keepdim=True)
                return exp_x / sum_exp

        compare_with_cpu(
            softmax_fn,
            x,
            run_compile=True,
            run_eager=False,
            atol=0.1,
            rtol=0.1,
        )

    # ------------------------------------------------------------------
    # Two ops with different slice counts -> two separate groups
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_two_groups(self):
        """Two separate tiling groups produce two LoopSpec entries in the source."""
        from torch_spyre._inductor import spyre_hint

        A, B = 256, 128
        x = torch.randn(A, B, dtype=torch.float16)
        y = torch.randn(A, B, dtype=torch.float16)

        def fn(x, y):
            # Two independent pointwise ops: each becomes its own group.
            with spyre_hint(num_tiles_per_dim={"A": 4}):
                out_x = torch.abs(x)
            with spyre_hint(num_tiles_per_dim={"A": 8}):
                out_y = torch.neg(y)
            return out_x, out_y

        x_dev = x.to("spyre")
        y_dev = y.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(x_dev, ["A", "B"])
        _name_tensor_dims(y_dev, ["A", "B"])

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x_dev, y_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        loop_spec_count = src.count("LoopSpec(")
        self.assertGreaterEqual(
            loop_spec_count,
            2,
            f"Expected ≥2 LoopSpec entries, got {loop_spec_count}\n\nSource:\n{src}",
        )

    # ------------------------------------------------------------------
    # Op inside hint scope with no matching named dim
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_group_includes_op_with_no_matching_dim(self):
        """An op inside a hint scope whose loop vars don't match the hinted dim stays in the group.

        torch.full lowers to a scalar-fill pointwise with no named loop variables.
        It has the hint but no loop var maps to "M", so it gets a scope-marker
        DimHint.  Its hint_id set still matches the surrounding ops so grouping
        is not broken.  The generated source must contain a single LoopSpec
        covering all ops.
        """
        from torch_spyre._inductor import spyre_hint

        M, K = 256, 64
        x = torch.randn(M, K, dtype=torch.float16)

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                # torch.full produces a scalar-fill with no M/K loop dim mapping.
                bias = torch.full(x.shape, 0.5, dtype=x.dtype, device=x.device)
                return x + bias

        x_dev = x.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _name_tensor_dims(x_dev, ["M", "K"])

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('4')", src, "Expected loop count 4")
        self.assertEqual(
            src.count("LoopSpec("),
            1,
            "Op with no matching dim must not break the group into two LoopSpec entries",
        )

    # ------------------------------------------------------------------
    # Hint propagation through mm_to_bmm_pass
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_survives_mm_to_bmm_rewrite(self):
        """spyre_hint is not dropped when mm_to_bmm_pass rewrites mm -> bmm.

        A 3D matmul inside a spyre_hint scope is decomposed to mm then rewritten
        back to bmm by mm_to_bmm_pass.  copy_fx_custom_meta must propagate the
        hint onto the new bmm node so assign_dim_hints can tile it.
        """
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 2, 128, 64, 32
        x = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        y = torch.randn(K, N, dtype=torch.float16) * 0.01

        def fn(x, y):
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                return torch.matmul(x, y)

        x_dev = x.to("spyre")
        y_dev = y.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x_dev, ["B", "M", "K"])
        _name_tensor_dims(y_dev, ["K", "N"])

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x_dev, y_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "LoopSpec(",
            src,
            "Expected LoopSpec: hint must survive mm->bmm rewrite",
        )
        self.assertIn("sympify('4')", src, "Expected loop count 4 after bmm rewrite")

    # ------------------------------------------------------------------
    # Hint propagation into inserted restickify nodes
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_restickify_stays_in_group(self):
        """A restickify node inserted inside a hint scope lands in the same group.

        output * correction triggers a restickify because output is col-major
        from a preceding transpose while correction is row-major.  The inserted
        restickify buffer must carry the hint metadata from its consumer so that
        assign_dim_hints includes it in the hinted group.  If it were ungrouped
        the LoopSpec count would cover fewer ops and the generated source would
        reflect a split group.
        """
        from torch_spyre._inductor import spyre_hint

        M, N = 256, 64
        x = torch.randn(M, N, dtype=torch.float16)
        scale = torch.randn(M, dtype=torch.float16)

        def fn(x, scale):
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                # transpose + contiguous forces a restickify on x before the mul
                x_t = x.transpose(0, 1).contiguous().transpose(0, 1)
                return x_t * scale.unsqueeze(-1)

        x_dev = x.to("spyre")
        scale_dev = scale.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x_dev, ["M", "N"])
        _name_tensor_dims(scale_dev, ["M"])

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x_dev, scale_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "LoopSpec(",
            src,
            "Expected LoopSpec: restickify must not break the hint group",
        )
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    # ------------------------------------------------------------------
    # Softmax with row-tiling: large [NROW, NCOL] tensor
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_softmax_row_tiling(self):
        """spyre_hint(num_tiles_per_dim={"NROW": 4}) tiles softmax over the row dimension."""
        from torch_spyre._inductor import spyre_hint

        NROW, NCOL = 16384, 4096
        x = torch.rand(NROW, NCOL, dtype=torch.float16)

        _declare_tensor_dim("NROW", NROW)
        _declare_tensor_dim("NCOL", NCOL)

        def fn(x, dim=-1):
            _name_tensor_dims(x, ["NROW", "NCOL"])
            with spyre_hint(num_tiles_per_dim={"NROW": 4}):
                return torch.softmax(x, dim)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=0.1, rtol=0.1)

    # ------------------------------------------------------------------
    # Matmul with row-tiling: tile the M dimension of x @ y
    # ------------------------------------------------------------------

    @config.patch({"coarse_tiling": True})
    def test_hint_matmul_row_tiling(self):
        """spyre_hint(num_tiles_per_dim={"M": 4}) tiles matmul over the row (M) dimension."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 256, 128, 64
        x = torch.randn(M, K, dtype=torch.float16) * 0.01
        y = torch.randn(K, N, dtype=torch.float16) * 0.01

        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(x, y):
            _name_tensor_dims(x, ["M", "K"])
            _name_tensor_dims(y, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                return x @ y

        compare_with_cpu(
            fn, x, y, run_compile=True, run_eager=False, atol=0.01, rtol=0.01
        )

    @config.patch({"coarse_tiling": True})
    def test_hint_flash_attention(self):
        """Flash attention tiled over H (4 slices) and Lk (2 slices) via nested spyre_hints."""
        import math
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk, D = 1, 8, 256, 256, 64
        block_size = 128

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)

        scale = 1.0 / math.sqrt(math.sqrt(D))
        lk_slices = Lk // block_size

        def flash(queries, keys, values):
            output = torch.zeros_like(queries)
            M = torch.full(
                (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
            )
            with spyre_hint(
                num_tiles_per_dim={"B": 1}
            ):  # 3 nested scopes exercises multi-hint logic
                with spyre_hint(num_tiles_per_dim={"H": 4}):
                    with spyre_hint(num_tiles_per_dim={"Lk": lk_slices}):
                        keys_T = keys.transpose(-1, -2).contiguous()
                        denominator = torch.zeros(
                            (B, H, Lq), device=queries.device, dtype=torch.float16
                        )
                        scores = torch.matmul(queries * scale, keys_T * scale)
                        scores = scores.transpose(-1, -2).contiguous()
                        block_max = torch.amax(scores, dim=-2)
                        max_running = torch.maximum(M, block_max)
                        exp_scores = torch.exp(scores - max_running.unsqueeze(-2))
                        correction = torch.exp(M - max_running)
                        denominator = denominator * correction + exp_scores.sum(dim=-2)
                        output = output * correction.unsqueeze(-1) + torch.matmul(
                            exp_scores.transpose(-1, -2), values
                        )
                        M = max_running
            return output / denominator.unsqueeze(-1)

        # CPU reference first, then device setup — matching the driver pattern exactly
        ref = flash(queries_t, keys_t, values_t)

        queries_dev = queries_t.to("spyre")
        keys_dev = keys_t.to("spyre")
        values_dev = values_t.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])

        result = torch.compile(flash)(queries_dev, keys_dev, values_dev).cpu()
        torch.testing.assert_close(
            result,
            ref,
            equal_nan=True,
            atol=0.01,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )

    @config.patch({"coarse_tiling": True})
    def test_hint_h_tiling_elementwise(self):
        """spyre_hint(num_tiles_per_dim={"H": 2}) tiles elementwise multiply over the H dimension.

        Regression test for a bug in _byte_stride_for_arg (unroll.py) where
        align_tensors rewrites device_coordinates but leaves stride_map stale,
        causing per-tile HBM base addresses to advance by the wrong amount when
        the tiled dimension is not the outermost host dimension (e.g. H in BHLD).
        """
        from torch_spyre._inductor import spyre_hint

        torch.manual_seed(42)
        B, H, Lq, Lk, D = 1, 8, 256, 256, 64  # Lk == Lq intentionally; same seq-len

        Q = torch.randn(B, H, Lq, D, dtype=torch.float16)
        V = torch.randn(B, H, Lk, D, dtype=torch.float16)

        def fn(q, v):
            with spyre_hint(num_tiles_per_dim={"H": 2}):
                return q * v

        ref = fn(Q, V)

        Q_dev = Q.to("spyre")
        V_dev = V.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(Q_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(V_dev, ["B", "H", "Lk", "D"])

        result = torch.compile(fn)(Q_dev, V_dev).cpu()
        torch.testing.assert_close(result, ref, atol=0.02, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
