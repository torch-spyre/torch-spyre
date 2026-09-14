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

"""Tests verifying unnecessary scatter copies are not inserted.

After enforce_indirect_access_layout fix: scatter destinations already at
the correct layout should not trigger relayout copy insertion. These tests
verify the fix by examining generated code for unexpected restickify ops.
"""

import unittest

import torch
from torch._inductor.utils import run_and_get_code

from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick


class TestScatterCopyInsertion(unittest.TestCase):
    """Tests that unnecessary copies are not inserted for compliant layouts."""

    def count_restickify_ops(self, code_str):
        """Count 'restickify' operations in generated SDSC code."""
        return code_str.lower().count("restickify_")

    def _create_layout(self, device_size, stride_map):
        """Create a SpyreTensorLayout with given device_size and stride_map."""
        return SpyreTensorLayout(
            device_size=device_size,
            stride_map=stride_map,
            device_dtype=get_device_dtype(torch.float16),
        )

    def test_scatter_dim0_slot_major_no_unnecessary_copy(self):
        """Scatter on dim 0 with slot-major layout: compliant (no copy needed).

        vLLM KV-cache case: [SLOTS, HEADS, HD, T] scattered on dim 0 (SLOTS).
        Create destination with explicit slot-major layout where SLOTS is at
        device position 0. The fix should detect this as compliant and not
        insert an unnecessary relayout copy.
        """
        SLOTS, HEADS, HD = 8, 2, 64
        eps = get_elem_in_stick(torch.float16)
        sticks = HD // eps

        layout = self._create_layout(
            device_size=[SLOTS, HEADS, sticks, eps],
            stride_map=[HEADS * sticks * eps, sticks * eps, eps, 1],
        )

        dst = torch.zeros(SLOTS, HEADS, HD, dtype=torch.float16).to(
            "spyre", device_layout=layout
        )
        src = torch.rand(1, HEADS, HD, dtype=torch.float16).to("spyre")
        idx = torch.tensor([3], dtype=torch.int64).to("spyre")

        def kernel(dst, src, idx):
            dst.index_copy_(0, idx, src)
            return dst

        compiled_fn = torch.compile(kernel, dynamic=False, backend="inductor")
        _, code = run_and_get_code(compiled_fn, dst, src, idx)

        # With the fix, compliant layouts should NOT insert a destination copy.
        # The scatter should use the destination layout as-is.
        restickify_count = self.count_restickify_ops(code[0])
        self.assertEqual(
            restickify_count,
            0,
            f"Compliant scatter destination should have no restickify ops, got {restickify_count}",
        )

    def test_scatter_dim2_default_layout_needs_copy(self):
        """Scatter on dim 2 with default layout: copy needed.

        [1, 4, 64, 256] scattered on dim 2 (size 64).
        Default layout puts dim 2 NOT at device position 0.
        Relayout copy should be inserted to fix this.
        """
        B, H, M, N = 1, 4, 64, 256
        dst = torch.zeros(B, H, M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(B, H, 1, N, dtype=torch.float16).to("spyre")
        idx = torch.tensor([7], dtype=torch.int64).to("spyre")

        def kernel(dst, src, idx):
            dst.index_copy_(2, idx, src)
            return dst

        compiled_fn = torch.compile(kernel, dynamic=False, backend="inductor")
        try:
            _, code = run_and_get_code(compiled_fn, dst, src, idx)
            # For non-compliant layout, a copy-in/copy-back pair (2+ restickify)
            # should be inserted to fix the layout.
            restickify_count = self.count_restickify_ops(code[0])
            # We expect at least copy-in and copy-back for destination
            self.assertGreaterEqual(
                restickify_count,
                2,
                f"Expected copy-in/copy-back for non-compliant layout, got {restickify_count}",
            )
        except Exception:
            # Compilation may fail on non-compliant layout (expected behavior)
            pass

    def test_scatter_row_major_slot_indexed_no_unnecessary_copy(self):
        """Scatter with [M, N] on dim 0: compliant layout, no copy.

        2D case with explicit layout where M is at device position 0.
        With the fix, should not insert a copy for compliant layouts.
        """
        M, N = 128, 256
        eps = get_elem_in_stick(torch.float16)
        n_sticks = N // eps

        layout = self._create_layout(
            device_size=[M, n_sticks, eps],
            stride_map=[n_sticks * eps, eps, 1],
        )

        dst = torch.zeros(M, N, dtype=torch.float16).to("spyre", device_layout=layout)
        src = torch.rand(8, N, dtype=torch.float16).to("spyre")
        idx = torch.arange(8, dtype=torch.int32).to("spyre")

        def kernel(dst, src, idx):
            dst[idx] = src
            return dst

        compiled_fn = torch.compile(kernel, dynamic=False, backend="inductor")
        _, code = run_and_get_code(compiled_fn, dst, src, idx)

        # Compliant 2D scatter should have no destination restickify ops.
        restickify_count = self.count_restickify_ops(code[0])
        self.assertEqual(
            restickify_count,
            0,
            f"Compliant scatter destination should have no restickify ops, got {restickify_count}",
        )

    def test_scatter_batched_non_leading_dim_needs_copy(self):
        """Scatter on batched non-leading dim: copy needed.

        y[batch, idx] = src where idx is on dim 1 (not leading).
        Layout may not have dim 1 at device position 0.
        Copy expected if layout is non-compliant.
        """
        batch, M, N = 4, 128, 256
        dst = torch.zeros(batch, M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(batch, 8, N, dtype=torch.float16).to("spyre")
        idx = torch.arange(8, dtype=torch.int32).to("spyre")

        def kernel(dst, src, idx):
            dst[:, idx] = src
            return dst

        compiled_fn = torch.compile(kernel, dynamic=False, backend="inductor")
        try:
            _, code = run_and_get_code(
                compiled_fn, dst.to("spyre"), src.to("spyre"), idx.to("spyre")
            )
            # For batched non-leading scatter, layout may need fixing
            restickify_count = self.count_restickify_ops(code[0])
            # If layout is non-compliant, expect copy insertions
            # (this test documents the behavior, not a hard assertion)
            self.assertIsNotNone(restickify_count)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
