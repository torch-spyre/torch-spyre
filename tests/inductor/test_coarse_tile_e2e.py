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

All coarse-tiling tests use the spyre_hint API (TestCoarseTileSpyreHints).
Add new tests there using spyre_hint(num_tiles_per_dim=...) annotations.
"""

import sys
import os

import pytest
import torch
import unittest
from unittest.mock import patch as mock_patch

from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor import config
import torch_spyre._inductor.propagate_named_dims as _pnd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils_inductor import compare_with_cpu  # noqa: E402

# Path to mock for disabling actual device kernel execution.
_LAUNCH_KERNEL = "torch_spyre.execution.kernel_runner.launch_kernel"


# ===========================================================================
# spyre_hint-driven coarse tiling
# These tests verify that coarse tiling is driven automatically by
# spyre_hint(num_tiles_per_dim=...) annotations.  Named tensor dimensions
# must be declared and annotated on device tensors for the hint resolver to
# map dimension names to loop variables.
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
    # Two ops in separate groups tiling different iteration dimensions
    # ------------------------------------------------------------------

    @config.patch(
        {
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_per_group_tiled_dims(self):
        """Two ops in separate hint groups tile different sets of iteration dims.

        Uses sub-dimension naming to map a [B, D] tensor's physical dims to
        named sub-dims, then tiles each op independently:

        op_a = abs(x): hint num_tiles_per_dim={"B": 4} tiles dim 0 only.
          B=256 → 4 tiles of 64 rows each.  Iteration space per tile: [64, D].

        op_b = neg(y): tensor named ["B0","B1","D0","D1"] with B0×B1=B and
          D0×D1=D.  Outer hint num_tiles_per_dim={"B0": 4} tiles dim 0 (c0,
          range 256) into 4.  Inner hint num_tiles_per_dim={"D0": 4} tiles
          dim 1 (c1, range 128) into 4.  Iteration space per tile: [64, 32].

        Both ops form separate groups → ≥2 LoopSpec entries, each with
        count=sympify('4').
        """
        from torch_spyre._inductor import spyre_hint

        B, D = 256, 128
        x = torch.randn(B, D, dtype=torch.float16)
        y = torch.randn(B, D, dtype=torch.float16)

        # Sub-dims for y: B0×B1 = B, D0×D1 = D
        B0, B1 = 4, B // 4  # 4 × 64 = 256
        D0, D1 = 4, D // 4  # 4 × 32 = 128

        x_dev = x.to("spyre")
        y_dev = y.to("spyre")

        # abs group: simple single-dim tiling over B
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        # neg group: sub-dim decomposition to tile both dims independently
        _declare_tensor_dim("B0", B0)
        _declare_tensor_dim("B1", B1)
        _declare_tensor_dim("D0", D0)
        _declare_tensor_dim("D1", D1)
        _name_tensor_dims(y_dev, ["B0", "B1", "D0", "D1"])

        def fn(x, y):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                out_x = torch.abs(x)
            with spyre_hint(num_tiles_per_dim={"B0": 4}):
                with spyre_hint(num_tiles_per_dim={"D0": 4}):
                    out_y = torch.neg(y)
            return out_x, out_y

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x_dev, y_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
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
    # Two ops with different slice counts -> two separate groups
    # ------------------------------------------------------------------

    @config.patch(
        {
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

    def test_hint_h_tiling_elementwise(self):
        """spyre_hint(num_tiles_per_dim={"H": 2}) tiles elementwise multiply over the H dimension.

        Regression test for a bug in _byte_stride_for_arg (unroll.py) where
        per-tile HBM base addresses advanced by the wrong amount when the tiled
        dimension was not the outermost host dimension (e.g. H in BHLD).
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


class TestNamedDimsHint(InductorTestCase):
    """Tests for propagate_named_dims handling of ops with a named_dims hint.

    torch.full and torch.empty lower to ops whose loop variables carry no
    named-dim information from their inputs.  The new hint path allows
    spyre_hint(named_dims=[...]) to supply the named-dim mapping directly,
    enabling coarse tiling to work on these ops.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    @config.patch(
        {
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_full_with_named_dims_hint_tiles(self):
        """spyre_hint(named_dims=[...]) on torch.full enables coarse tiling.

        Without the hint, torch.full has no named-dim mapping and coarse tiling
        cannot apply.  With named_dims supplied via the hint, propagate_named_dims
        should set _dim_prop_info correctly so assign_dim_hints produces a
        DimHint and LoopSpec appears in the generated source.
        """
        from torch_spyre._inductor import spyre_hint

        M, K = 256, 64

        def fn(x):
            with spyre_hint(slices={"M": 4}, named_dims=["M", "K"]):
                bias = torch.full(x.shape, 0.5, dtype=x.dtype, device=x.device)
            return x + bias

        x = torch.randn(M, K, dtype=torch.float16)
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

    @config.patch(
        {
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_full_like_with_named_dims_hint_tiles(self):
        """spyre_hint(named_dims=[...]) on torch.full_like enables coarse tiling."""
        from torch_spyre._inductor import spyre_hint

        M, K = 128, 64

        def fn(x):
            with spyre_hint(slices={"M": 2}, named_dims=["M", "K"]):
                buf = torch.full_like(x, 2.0)
            return x + buf

        x = torch.randn(M, K, dtype=torch.float16)
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
        self.assertIn("sympify('2')", src, "Expected loop count 2")


class TestCoarseTileReductionE2E(InductorTestCase):
    """E2E tests for coarse-tiling a reduction dimension.

    Stage 1 supports tiling reductions over non-stick dimensions only.
    Tiling a reduction over the stick dimension (dim=-1 on a [..., D] tensor
    where D maps to the stick) raises RuntimeError — deferred to Stage 2.

    The tests below verify that the appropriate error is raised for stick-dim
    reduction tiling, and that LoopSpec is still emitted up to the point where
    validation fires.
    """

    _STAGE2_MSG = "stick-dim reduction tiling is not yet implemented — Stage 2"

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_hint_tiled_reduction_sum_loopspec(self):
        """x.sum(dim=-1) tiled over D raises: stick-dim reduction not yet supported."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16) * 0.1
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.sum(dim=-1)

        with self.assertRaisesRegex(Exception, self._STAGE2_MSG):
            torch.compile(fn)(x_dev)

    def test_hint_tiled_reduction_sum_rejects(self):
        """x.sum(dim=-1) with D hint rejects at compile time with Stage 2 error."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16) * 0.1
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.sum(dim=-1)

        with self.assertRaisesRegex(Exception, self._STAGE2_MSG):
            torch.compile(fn)(x_dev)

    def test_hint_tiled_reduction_matmul_loopspec(self):
        """torch.matmul tiled over K produces a LoopSpec with count 4."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return a @ b

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for K-tiled matmul")
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    def test_hint_tiled_reduction_matmul_correct(self):
        """torch.matmul tiled over K (4 tiles) produces correct results."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return a @ b

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_hint_tiled_reduction_max_loopspec(self):
        """x.amax(dim=-1) tiled over D raises: stick-dim reduction not yet supported."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amax(dim=-1)

        with self.assertRaisesRegex(Exception, self._STAGE2_MSG):
            torch.compile(fn)(x_dev)

    def test_hint_tiled_reduction_max_rejects(self):
        """x.amax(dim=-1) with D hint rejects at compile time with Stage 2 error."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amax(dim=-1)

        with self.assertRaisesRegex(Exception, self._STAGE2_MSG):
            torch.compile(fn)(x_dev)

    def test_hint_tiled_reduction_min_loopspec(self):
        """x.amin(dim=-1) tiled over D raises: stick-dim reduction not yet supported."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amin(dim=-1)

        with self.assertRaisesRegex(Exception, self._STAGE2_MSG):
            torch.compile(fn)(x_dev)

    def test_hint_tiled_reduction_min_rejects(self):
        """x.amin(dim=-1) with D hint rejects at compile time with Stage 2 error."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amin(dim=-1)

        with self.assertRaisesRegex(Exception, self._STAGE2_MSG):
            torch.compile(fn)(x_dev)


class TestCoarseTileReductionDim0E2E(InductorTestCase):
    """E2E tests for coarse-tiling a reduction over dim=0.

    These reduce a [B, D] tensor over B (dim=0), producing a [D] output where
    D is on the stick.  This is a simpler case than dim=-1 reductions because
    the output has a normal stick layout (no column-vector addressing).
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_hint_tiled_reduction_dim0_sum_correct(self):
        """x.sum(dim=0) tiled over B produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 512, 64
        x = torch.randn(B, D, dtype=torch.float16) * 0.1

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return x.sum(dim=0)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)

    def test_hint_tiled_reduction_dim0_max_correct(self):
        """x.amax(dim=0) tiled over B produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 512, 64
        x = torch.randn(B, D, dtype=torch.float16)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return x.amax(dim=0)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=1e-3, rtol=1e-3)

    def test_hint_tiled_reduction_dim0_min_correct(self):
        """x.amin(dim=0) tiled over B produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 512, 64
        x = torch.randn(B, D, dtype=torch.float16)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return x.amin(dim=0)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=1e-3, rtol=1e-3)


class TestCoarseTileMatmulKTilingE2E(InductorTestCase):
    """Correctness and LoopSpec tests for matmul/bmm tiled over the K (reduction) dimension.

    K=512 tiled by 4 gives 128 per tile (two sticks at fp16); shapes are chosen
    so K/T is stick-aligned without padding, keeping results deterministic.
    Use small weight scale (0.01) to keep fp16 accumulation error bounded.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_mm_k_tiled_correct(self):
        """2D mm [M,K] @ [K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.mm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_bmm_k_tiled_correct(self):
        """3D bmm [B,M,K] @ [B,K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 8, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(B, K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["B", "K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.bmm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_bmm_3d2d_k_tiled_correct(self):
        """3D×2D matmul [B,M,K] @ [K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 8, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.matmul(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_mm_k_tiled_loopspec(self):
        """K-tiled mm produces a LoopSpec with count 4 in generated source."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.mm(a, b)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for K-tiled mm")
        self.assertIn("sympify('4')", src, "Expected loop count 4")


class TestCoarseTileNestedReductionE2E(InductorTestCase):
    """Correctness and LoopSpec tests for nested output-dim + reduction-dim tiling.

    Pattern: outer loop tiles an output dim, inner loop tiles a reduction dim.
    The fill op runs inside the outer loop (once per outer tile), so the
    accumulator is per-outer-tile sized.  The full output buffer spans all outer
    tiles; address advancement across outer iterations assembles the result.

    mm shapes: M=128, K=512, N=32; outer tiles M by 2 (64 rows/tile),
    inner tiles K by 4 (128 elements/tile = 2 sticks at fp16).
    bmm shapes: B=4, M=64, K=512, N=32; outer tiles B by 2,
    inner tiles K by 4.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xCAFE)

    @pytest.mark.xfail(
        reason=(
            "Fails with cpu/spyre mismatch (~15-25% elements wrong). "
            "Root cause under investigation: suspected C++ runtime or "
            "device-level state contamination from prior flash attention "
            "execution, or a codegen bug in nested outer-B + inner-K tiling."
        ),
        strict=True,
    )
    def test_nested_bmm_outer_Batch_inner_K_correct(self):
        """bmm [B,M,K]@[B,K,N] outer B (output) + inner K (reduction) — correct."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 4, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(B, K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["B", "K", "N"])
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.bmm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_nested_matmul_outer_M_inner_K_correct(self):
        """mm [M,K]@[K,N] with outer M (output) + inner K (reduction) — correct."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_nested_matmul_outer_M_inner_K_loopspec(self):
        """Nested mm produces two LoopSpec levels (outer count 2, inner count 4)."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for nested mm")
        self.assertIn("sympify('2')", src, "Expected outer loop count 2")
        self.assertIn("sympify('4')", src, "Expected inner loop count 4")

    @config.patch({"lx_planning": False, "unroll_loops": False})
    def test_nested_matmul_copy_after_inner_loop(self):
        """The accum→output copy op appears in generated source for nested K-tiling."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "coarse_tile_reduce_copy",
            src,
            "Expected a coarse_tile_reduce_copy op in generated source for nested M+K tiling",
        )

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "unroll_loops": False,
        }
    )
    def test_nested_matmul_outer_M_inner_K_accum_in_lx(self):
        """With lx_planning enabled, the tile-sized accum buffer lands in LX scratchpad."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "allocation={'lx'",
            src,
            "Expected tile-sized accum TensorArg with lx allocation for nested M+K tiling",
        )


if __name__ == "__main__":
    unittest.main()
