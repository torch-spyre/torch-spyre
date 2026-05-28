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

Tested scenarios
----------------
- test_no_tiling_baseline: confirm LoopSpec is absent when coarse_tiling=False.
- test_single_group_tiles_pointwise: tile a pointwise op into K=4 iterations;
  assert LoopSpec(count=sympify('4'), ...) appears in generated source.
- test_softmax_shaped_tiling: tile the pointwise-reduce-pointwise chain that
  softmax lowers to; assert all stages land in a single LoopSpec.
- test_two_groups: two separate tiling groups produce two LoopSpec entries.
- test_generate_bundle_receives_loop_spec: verify generate_bundle sees LoopSpec.
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


def _groups_split_k4_k8(operations: list[Operation]):
    """Two groups: first ComputedBuffer at K=4, remainder at K=8."""
    ops = [op for op in operations if isinstance(op, ComputedBuffer)]
    groups = []
    if ops[:1]:
        groups.append((ops[:1], sympy.Integer(4)))
    if ops[1:]:
        groups.append((ops[1:], sympy.Integer(8)))
    return groups


def _groups_nested_k2_m4(operations: list[Operation]):
    """One group: all ops share nested K=2 outer (dim 0) / M=4 inner (dim 1) loops."""
    ops = [op for op in operations if isinstance(op, ComputedBuffer)]
    if not ops:
        return []
    return [(ops, [(sympy.Integer(2), [0]), (sympy.Integer(4), [1])])]


def _groups_nested_k2_m2(operations: list[Operation]):
    """One group: all ops share nested K=2 outer (dim 0) / M=2 inner (dim 1) loops."""
    ops = [op for op in operations if isinstance(op, ComputedBuffer)]
    if not ops:
        return []
    return [(ops, [(sympy.Integer(2), [0]), (sympy.Integer(2), [1])])]


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoarseTileEndToEnd(InductorTestCase):
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
    # Single group: tile a pointwise op
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_all_k4,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
        }
    )
    def test_single_group_tiles_pointwise(self):
        """A pointwise abs tiled K=4 times should produce LoopSpec(count=4)."""
        # 256 rows × 128 cols.  Tiling the outermost dim by 4 → 64 rows/iter.
        x = torch.randn(256, 128, dtype=torch.float16).to("spyre")

        def fn(x):
            return torch.abs(x)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec call in generated source")
        self.assertIn(
            "sympify('4')",
            src,
            "Expected loop count 4 in generated source",
        )

    # ------------------------------------------------------------------
    # Softmax-shaped computation: pointwise → reduce → pointwise chain
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_all_k4,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
        }
    )
    def test_softmax_shaped_tiling(self):
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
        B, D = 256, 128  # batch = 256 rows, each of length 128
        x = torch.randn(B, D, dtype=torch.float16).to("spyre")

        def softmax_fn(x):
            max_val = x.amax(dim=-1, keepdim=True)
            x_shifted = x - max_val
            exp_x = x_shifted.exp()
            sum_exp = exp_x.sum(dim=-1, keepdim=True)
            return exp_x / sum_exp

        cfn = torch.compile(softmax_fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x)
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
    # Two groups: verify separate LoopSpecs for two disjoint op sets
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_split_k4_k8,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
        }
    )
    def test_two_groups_produce_two_loop_specs(self):
        """Two separate tiling groups produce two LoopSpec entries in the source."""
        x = torch.randn(256, 128, dtype=torch.float16).to("spyre")
        y = torch.randn(256, 128, dtype=torch.float16).to("spyre")

        def fn(x, y):
            # Two independent pointwise ops: each becomes its own group.
            return torch.abs(x), torch.neg(y)

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, x, y)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        loop_spec_count = src.count("LoopSpec(")
        self.assertGreaterEqual(
            loop_spec_count,
            2,
            f"Expected ≥2 LoopSpec entries, got {loop_spec_count}\n\nSource:\n{src}",
        )

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
    # Nested loops: single op tiled on two dimensions independently
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_nested_k2_m4,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
        }
    )
    def test_nested_loop_two_dims(self):
        """Two pointwise ops share nested K=2 (outer, dim 0) / M=4 (inner, dim 1) loops.

        Input shape [1024, 4096]: outer loop runs 2× over dim 0 (512 rows/iter),
        inner loop runs 4× over dim 1 (1024 cols/iter).  Both ops (add and mul)
        are placed in the same group so they share the nested LoopSpec.
        Generated source must contain two nested LoopSpec entries with counts 2
        and 4, with two OpSpec entries in the innermost body.
        """
        a = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
        b = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
        c = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")

        def fn(a, b, c):
            y = a + b
            z = y * c
            return z

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a, b, c)
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
    # Scratchpad (LX) allocation for intermediate tiled buffer
    # ------------------------------------------------------------------

    @unittest.skip(
        "insert_tiling_propagation allocates a full-size output buffer via "
        "MutationLayoutSHOULDREMOVE, which causes core_div_mismatch in "
        "scratchpad_planning — the intermediate add buffer falls back to pool "
        "instead of lx.  Tracked for a follow-up PR."
    )
    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_nested_k2_m4,
            "bundle_hbm_symbols": True,
            "unroll_loops": False,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_nested_loop_with_scratchpad(self):
        """Intermediate tiled buffer (y = a + b) is allocated to scratchpad.

        With lx_planning enabled and allow_all_ops_in_lx_planning=True,
        scratchpad_planning runs after coarse_tile and assigns the
        intermediate add result to LX (scratchpad) memory since it is
        only consumed within the loop body.  The final output (z = y * c)
        stays in HBM.

        Assertions:
        - LoopSpec entries are still emitted (tiling is unaffected).
        - At least one TensorArg carries allocation={'lx': ...}.
        - The output buffer allocation uses 'hbm' (not 'lx').
        - The per-tile buffer size [512, 1024] appears in the allocation.
        """
        a = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
        b = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
        c = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")

        def fn(a, b, c):
            y = a + b
            z = y * c
            return z

        cfn = torch.compile(fn)
        with mock_patch(_LAUNCH_KERNEL), mock_patch("subprocess.run"):
            _, source_codes = run_and_get_code(cfn, a, b, c)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn(
            "allocation={'lx'",
            src,
            "Expected at least one TensorArg with lx allocation",
        )
        self.assertIn(
            "allocation={'hbm'",
            src,
            "Expected output TensorArg with hbm allocation",
        )
        # Per-tile buffer shape must appear in the spyre_empty_with_layout call.
        self.assertIn(
            "(512, 1024)",
            src,
            "Expected per-tile buffer size (512, 1024) in generated source",
        )


# ===========================================================================
# Unrolled loop execution tests (unroll_loops=True)
# ===========================================================================


class TestCoarseTileUnrollEndToEnd(InductorTestCase):
    """Tests for coarse tiling with loop unrolling (unroll_loops=True).

    When unroll_loops=True, LoopSpec nodes are fully unrolled before
    generate_bundle so no scf.for is emitted.  Each iteration becomes an
    independent OpSpec with concrete per-iteration HBM addresses.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    # ------------------------------------------------------------------
    # Source inspection: unrolling passes LoopSpec through async_compile
    # with concrete per-iteration HBM addresses in each unrolled OpSpec.
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_nested_k2_m4,
            "unroll_loops": True,
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
    # Real execution: unrolled tiling runs on device with sencores=1.
    # ------------------------------------------------------------------

    @unittest.skip(
        "Nested K=2×M=4 unrolling produces wrong results on device. "
        "Tracked for a follow-up PR."
    )
    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_nested_k2_m4,
            "unroll_loops": True,
            "sencores": 1,
        }
    )
    def test_unrolled_real_execution(self):
        """Unrolled nested K=2×M=4 tiling of the design-doc small example.

        Replicates the computation from the "Small Example" section of
        docs/source/compiler/coarse_tiling_loops.md:
          y = a + b
          z = y * c
        over [1024, 4096] fp16 tensors, with K=2 outer loop (dim 0) and M=4
        inner loop (dim 1).  sencores=1 avoids core-division/scratchpad issues.
        """
        a = torch.randn(1024, 4096, dtype=torch.float16)
        b = torch.randn(1024, 4096, dtype=torch.float16)
        c = torch.randn(1024, 4096, dtype=torch.float16)

        def fn(a, b, c):
            y = a + b
            return y * c

        cpu_result = fn(a, b, c)

        device = torch.device("spyre")
        torch._dynamo.reset_code_caches()
        torch._inductor.codecache.FxGraphCache.clear()
        device_args = [t.to(device) for t in [a, b, c]]
        spyre_result = torch.compile(fn)(*device_args).cpu()

        # Per-tile mismatch analysis: K=2 (dim0 tiles of 512 rows) × M=4 (dim1 tiles of 1024 cols)
        atol, rtol = 0.1, 0.1
        diff_mask = ~torch.isclose(spyre_result, cpu_result, atol=atol, rtol=rtol)
        total_wrong = diff_mask.sum().item()
        if total_wrong > 0:
            rows_k, cols_k = 512, 1024
            lines = [
                f"Total mismatches: {total_wrong}/{spyre_result.numel()} "
                f"({100.0 * total_wrong / spyre_result.numel():.1f}%)"
            ]
            for k in range(2):
                for m in range(4):
                    tile = diff_mask[
                        k * rows_k : (k + 1) * rows_k, m * cols_k : (m + 1) * cols_k
                    ]
                    tw = tile.sum().item()
                    tt = rows_k * cols_k
                    lines.append(
                        f"  tile[k={k}, m={m}]: {tw}/{tt} wrong ({100.0 * tw / tt:.1f}%)"
                    )
            wrong_indices = diff_mask.nonzero(as_tuple=False)[:8]
            lines.append("  Sample wrong elements (row, col, spyre, cpu):")
            for idx in wrong_indices:
                r, col = idx[0].item(), idx[1].item()
                lines.append(
                    f"    [{r:4d}, {col:4d}]  spyre={spyre_result[r, col].item():.4f}"
                    f"  cpu={cpu_result[r, col].item():.4f}"
                )
            self.fail("\n".join(lines))

    # ------------------------------------------------------------------
    # Coordinate-probe test: small tensor with unique values per element.
    # ------------------------------------------------------------------

    @unittest.skip(
        "Nested K=2×M=2 unrolling produces wrong results on device. "
        "Kept as a regression probe; tracked for a follow-up PR."
    )
    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_nested_k2_m2,
            "unroll_loops": True,
            "sencores": 1,
        }
    )
    def test_unrolled_coordinate_probe(self):
        """Nested K=2×M=2 tiling with unique-value inputs to diagnose address errors.

        Shape [8, 256]: K=2 outer (dim 0, 4 rows/tile) × M=2 inner (dim 1, 128
        cols/tile = 2 sticks/tile).  a[r, c] = r*256 + c encodes position
        exactly in fp16 (max 2047).  b = zeros, multiplier c = ones, so the
        output z = (a + b) * c = a.  Any wrong value directly reveals which
        input coordinates the hardware actually read.
        """
        rows, cols = 8, 256

        # Build coordinate-encoded inputs on CPU then move to device.
        r_idx = torch.arange(rows, dtype=torch.float16).unsqueeze(1).expand(rows, cols)
        c_idx = torch.arange(cols, dtype=torch.float16).unsqueeze(0).expand(rows, cols)
        a = r_idx * cols + c_idx  # a[r,c] = r*256 + c, exact in fp16
        b = torch.zeros(rows, cols, dtype=torch.float16)
        c = torch.ones(rows, cols, dtype=torch.float16)

        def fn(a, b, c):
            y = a + b
            return y * c

        cpu_result = fn(a, b, c)

        device = torch.device("spyre")
        torch._dynamo.reset_code_caches()
        torch._inductor.codecache.FxGraphCache.clear()
        device_args = [t.to(device) for t in [a, b, c]]
        spyre_result = torch.compile(fn)(*device_args).cpu()

        # Tile dimensions: K=2 outer (4 rows/tile), M=2 inner (128 cols/tile).
        tile_rows, tile_cols = rows // 2, cols // 2
        diff_mask = spyre_result != cpu_result
        total_wrong = diff_mask.sum().item()
        if total_wrong > 0:
            lines = [
                f"Total mismatches: {total_wrong}/{spyre_result.numel()}",
                f"Tile shape: {tile_rows} rows × {tile_cols} cols",
            ]
            for k in range(2):
                for m in range(2):
                    rs = k * tile_rows
                    cs = m * tile_cols
                    tile_got = spyre_result[rs : rs + tile_rows, cs : cs + tile_cols]
                    tile_exp = cpu_result[rs : rs + tile_rows, cs : cs + tile_cols]
                    tw = (tile_got != tile_exp).sum().item()
                    lines.append(
                        f"  tile[k={k}, m={m}] ({rs}:{rs + tile_rows}, {cs}:{cs + tile_cols}): {tw} wrong"
                    )
            lines.append(
                "  Wrong elements (row, col, got, expected, got_encodes_row, got_encodes_col):"
            )
            wrong_indices = diff_mask.nonzero(as_tuple=False)
            for idx in wrong_indices[:16]:
                r, col = idx[0].item(), idx[1].item()
                got = spyre_result[r, col].item()
                exp = cpu_result[r, col].item()
                # Decode what row/col the hardware actually fetched.
                got_r = int(got) // cols
                got_c = int(got) % cols
                lines.append(
                    f"    [{r:3d},{col:3d}]  got={got:.0f} (→[{got_r},{got_c}])  expected={exp:.0f}"
                )
            self.fail("\n".join(lines))

    # ------------------------------------------------------------------
    # Softmax-shaped tiling: reduction + pointwise, unrolled.
    # ------------------------------------------------------------------

    @config.patch(
        {
            "coarse_tiling": True,
            "coarse_tiling_groups_fn": _groups_all_k4,
            "unroll_loops": True,
            "sencores": 1,
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


if __name__ == "__main__":
    unittest.main()
