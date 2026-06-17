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

# Owner(s): ["module: spyre"]

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch_spyre.model_utils import (
    _dma_to_spyre_dim_order_swapped,
    load_model_to_spyre,
    patch_module_to_for_spyre,
)


@instantiate_parametrized_tests
class TestLoadModelToSpyre(TestCase):
    """Tests for torch_spyre.model_utils (issue #1339)."""

    def setUp(self):
        torch.manual_seed(0xAFFE)

    # ── core layout (issue #1339) ──────────────────────────────────

    def test_linear_weight_has_dim_order_swapped(self):
        """A 2D Linear weight gets stickified on dim 0 (out_features).
        For a (1024, 4096) weight, that's 1024/64 = 16 sticks on dim 0.
        """
        from torch_spyre._C import get_spyre_tensor_layout

        w = torch.randn(1024, 4096, dtype=torch.float16)
        dev = _dma_to_spyre_dim_order_swapped(w)
        layout = get_spyre_tensor_layout(dev)
        self.assertEqual(layout.device_size[0], 16)

    def test_dim_order_rejects_non_2d(self):
        """The dim_order helper only accepts 2D weights."""
        with self.assertRaises(AssertionError):
            _dma_to_spyre_dim_order_swapped(torch.randn(4, dtype=torch.float16))

    # ── routing ────────────────────────────────────────────────────

    def test_load_model_routes_linear_weight_through_dim_order(self):
        """Linear weight ends up on Spyre with dim_order layout;
        other params (bias) use default layout."""
        from torch_spyre._C import get_spyre_tensor_layout

        model = nn.Linear(64, 128, dtype=torch.float16)
        load_model_to_spyre(model)

        self.assertEqual(model.weight.device.type, "spyre")
        self.assertEqual(model.bias.device.type, "spyre")
        # Linear weight: device_size[0] = out_features/64 = 128/64 = 2
        weight_layout = get_spyre_tensor_layout(model.weight)
        self.assertEqual(weight_layout.device_size[0], 2)

    def test_load_model_handles_layernorm(self):
        """Non-Linear params/buffers reach Spyre via the default path."""
        model = nn.LayerNorm(128, dtype=torch.float16)
        load_model_to_spyre(model)
        self.assertEqual(model.weight.device.type, "spyre")
        self.assertEqual(model.bias.device.type, "spyre")

    # ── dtype contract (PR #2258) ──────────────────────────────────

    @parametrize(
        "src_dtype,target_dtype",
        [
            (torch.float32, torch.float16),
            (torch.float16, torch.bfloat16),
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float16),
        ],
    )
    def test_explicit_dtype_is_honored(self, src_dtype, target_dtype):
        """model.to('spyre', dtype=X) puts every param on device as X.
        copy_tensor (PR #2258) converts during the DMA."""
        model = nn.Linear(64, 128, dtype=src_dtype)
        load_model_to_spyre(model, dtype=target_dtype)
        for p in model.parameters():
            self.assertEqual(p.device.type, "spyre")
            self.assertEqual(p.dtype, target_dtype)

    def test_dtype_none_preserves_source(self):
        """Without dtype, each tensor keeps its source dtype on device."""
        model = nn.Linear(64, 128, dtype=torch.float32)
        load_model_to_spyre(model)
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(model.bias.dtype, torch.float32)

    def test_unsupported_dtype_raises(self):
        """Dtypes that map to DataFormats.INVALID are rejected."""
        model = nn.Linear(4, 4)
        with self.assertRaises(ValueError):
            load_model_to_spyre(model, dtype=torch.complex64)

    # ── idempotency ────────────────────────────────────────────────

    def test_load_model_idempotent(self):
        """Second call on an already-loaded model is a no-op
        (params already on Spyre are skipped)."""
        model = nn.Linear(64, 128, dtype=torch.float16)
        load_model_to_spyre(model)
        ptr = model.weight.data_ptr()
        load_model_to_spyre(model)
        self.assertEqual(model.weight.data_ptr(), ptr)

    # ── nn.Module.to() patch ───────────────────────────────────────

    def test_patch_module_to_is_idempotent(self):
        original = nn.Module.to
        try:
            patch_module_to_for_spyre()
            first = nn.Module.to
            patch_module_to_for_spyre()  # second call should no-op
            self.assertIs(nn.Module.to, first)
            self.assertTrue(getattr(nn.Module.to, "_spyre_patched", False))
        finally:
            nn.Module.to = original

    def test_model_to_spyre_applies_optimal_layout(self):
        """End-to-end: model.to('spyre') routes through the patched
        nn.Module.to and lands the Linear weight with dim_order layout."""
        from torch_spyre._C import get_spyre_tensor_layout

        original = nn.Module.to
        try:
            patch_module_to_for_spyre()
            model = nn.Linear(64, 128, dtype=torch.float16)
            model.to("spyre")
            self.assertEqual(model.weight.device.type, "spyre")
            layout = get_spyre_tensor_layout(model.weight)
            self.assertEqual(layout.device_size[0], 2)  # 128/64
        finally:
            nn.Module.to = original


if __name__ == "__main__":
    run_tests()
