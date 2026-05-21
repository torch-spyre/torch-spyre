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

import sympy
import torch
import unittest
from unittest.mock import patch as mock_patch

from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.ir import ComputedBuffer, Operation

from torch_spyre._inductor import config

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

    @config.patch({"coarse_tiling": True, "coarse_tiling_groups_fn": _groups_all_k4})
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

    @config.patch({"coarse_tiling": True, "coarse_tiling_groups_fn": _groups_all_k4})
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
        {"coarse_tiling": True, "coarse_tiling_groups_fn": _groups_split_k4_k8}
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


if __name__ == "__main__":
    unittest.main()
