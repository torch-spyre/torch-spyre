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

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch_spyre.model_utils import (
    _dma_to_spyre_default,
    _dma_to_spyre_dim_order_swapped,
    _dma_to_spyre_indirect_access,
    dma_moe_expert_weight_to_spyre,
    dma_moe_per_expert_scale_to_spyre,
    load_model_to_spyre,
    patch_module_to_for_spyre,
)

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import safetensors  # noqa: F401

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# fp16 lands on device as DLFLOAT16 (SEN169_FP16, 9 mantissa bits), so a
# round-trip is never bit-exact: 2**-10 half-ULP, and fp16 subnormals come
# back halved (abs error <= 3.1e-5).
DLFLOAT16_RTOL = 2e-3
DLFLOAT16_ATOL = 1e-4


def _spyre_available() -> bool:
    """Return True iff a physical Spyre device is accessible."""
    try:
        from torch_spyre.constants import DEVICE_NAME

        torch.zeros(1, dtype=torch.float16, device=DEVICE_NAME)
        return True
    except Exception:
        return False


#: Decorator: skip the test when no Spyre hardware is present.
requires_spyre = unittest.skipUnless(_spyre_available(), "requires Spyre hardware")
requires_transformers = unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers not installed"
)
requires_safetensors = unittest.skipUnless(
    HAS_SAFETENSORS, "safetensors not installed"
)


@instantiate_parametrized_tests
class TestLoadModelToSpyre(TestCase):
    """Tests for torch_spyre.model_utils (issue #1339)."""

    def setUp(self):
        torch.manual_seed(0xAFFE)

    # ── core layout (issue #1339) ──────────────────────────────────

    @requires_spyre
    def test_linear_weight_has_dim_order_swapped(self):
        """A 2D Linear weight gets stickified on dim 0 (out_features).
        For a (1024, 4096) weight, that's 1024/64 = 16 sticks on dim 0.
        """
        from torch_spyre._C import get_spyre_tensor_layout

        w = torch.randn(1024, 4096, dtype=torch.float16)
        dev = _dma_to_spyre_dim_order_swapped(w)
        layout = get_spyre_tensor_layout(dev)
        self.assertEqual(layout.device_size[0], 16)

    @requires_spyre
    def test_dim_order_rejects_non_2d(self):
        """The dim_order helper only accepts 2D weights."""
        with self.assertRaises(AssertionError):
            _dma_to_spyre_dim_order_swapped(torch.randn(4, dtype=torch.float16))

    @requires_spyre
    def test_dma_helpers_accept_an_explicit_device(self):
        """All DMA helpers target another device and restore current."""
        from torch_spyre._C import get_spyre_tensor_layout

        if torch.spyre.device_count() < 2:
            self.skipTest("requires at least two Spyre devices")

        previous = torch.spyre.current_device()
        device = torch.device("spyre", (previous + 1) % torch.spyre.device_count())
        # The runtime currently initializes one stream-pool device per process,
        # so exercise allocation/layout placement without starting a second
        # device's DMA stream. Existing single-device tests cover the copies.
        with mock.patch("torch_spyre.model_utils.copy_tensor") as copy_tensor:
            default = _dma_to_spyre_default(
                torch.randn(128, 256, dtype=torch.float16), device=device
            )
            linear = _dma_to_spyre_dim_order_swapped(
                torch.randn(128, 256, dtype=torch.float16), device=device
            )
            embedding = _dma_to_spyre_indirect_access(
                torch.randn(1000, 256, dtype=torch.float16), device=device
            )
            expert = dma_moe_expert_weight_to_spyre(
                torch.randn(3, 64, 128, dtype=torch.float16), device=device
            )
            scale = dma_moe_per_expert_scale_to_spyre(
                torch.arange(3, dtype=torch.float16), device=device
            )

        self.assertEqual(copy_tensor.call_count, 5)

        self.assertEqual(default.device, device)
        self.assertEqual(linear.device, device)
        self.assertEqual(
            list(get_spyre_tensor_layout(linear).device_size), [2, 256, 64]
        )
        self.assertEqual(embedding.device, device)
        self.assertEqual(
            list(get_spyre_tensor_layout(embedding).device_size), [1000, 4, 64]
        )
        self.assertEqual(expert.device, device)
        self.assertEqual(
            list(get_spyre_tensor_layout(expert).device_size), [3, 64, 2, 64]
        )
        self.assertEqual(scale.device, device)
        self.assertEqual(list(get_spyre_tensor_layout(scale).device_size), [3, 1, 64])
        self.assertEqual(torch.spyre.current_device(), previous)

    @requires_spyre
    def test_low_level_layout_allocation_uses_explicit_device(self):
        """The layout allocator targets its device without changing the caller's."""
        from torch_spyre._C import SpyreTensorLayout, spyre_empty_with_layout

        if torch.spyre.device_count() < 2:
            self.skipTest("requires at least two Spyre devices")

        previous = torch.spyre.current_device()
        device_index = (previous + 1) % torch.spyre.device_count()
        device = torch.device("spyre", device_index)
        layout = SpyreTensorLayout([128, 256], torch.float16)

        tensor = spyre_empty_with_layout(
            (128, 256), (256, 1), torch.float16, layout, device=device
        )

        self.assertEqual(tensor.device, device)
        self.assertEqual(torch.spyre.current_device(), previous)

    @requires_spyre
    def test_dma_helpers_accept_an_integer_device(self):
        """Integer destinations are normalized before entering the C++ binding."""
        device_index = torch.spyre.current_device()
        linear = _dma_to_spyre_dim_order_swapped(
            torch.randn(128, 256, dtype=torch.float16), device=device_index
        )
        self.assertEqual(linear.device, torch.device("spyre", device_index))

    # ── embedding gather-optimal (indirect-access) layout ──────────

    @requires_spyre
    def test_embedding_has_indirect_access_layout(self):
        """A 2D Embedding table gets device dims [rows, D // eps, eps] with the
        vocab dim outermost. For a (1000, 256) fp16 table, eps=64, so the
        device layout is [1000, 4, 64]."""
        from torch_spyre._C import get_spyre_tensor_layout

        w = torch.randn(1000, 256, dtype=torch.float16)
        dev = _dma_to_spyre_indirect_access(w)
        layout = get_spyre_tensor_layout(dev)
        self.assertEqual(list(layout.device_size), [1000, 4, 64])
        # device_size only checks the block shape; round-trip through the
        # layout so a wrong stride_map entry (correct shape, scrambled data)
        # fails loudly instead of passing silently.
        torch.testing.assert_close(
            dev.cpu(), w, rtol=DLFLOAT16_RTOL, atol=DLFLOAT16_ATOL
        )

    @requires_spyre
    def test_embedding_stick_size_is_dtype_aware(self):
        """elems_per_stick is 32 at fp32, so a (1000, 256) fp32 table splits
        into 256/32 = 8 sticks, not 4."""
        from torch_spyre._C import get_spyre_tensor_layout

        w = torch.randn(1000, 256, dtype=torch.float32)
        dev = _dma_to_spyre_indirect_access(w)
        layout = get_spyre_tensor_layout(dev)
        self.assertEqual(list(layout.device_size), [1000, 8, 32])
        # Round-trip so a wrong stride_map entry fails loudly, not just a
        # device_size check (the fp32 stick split is 8x32, not 4x64). fp32 maps
        # to IEEE_FP32, so this round-trip is lossless -- defaults suffice.
        torch.testing.assert_close(dev.cpu(), w)

    @requires_spyre
    def test_embedding_non_tiling_hidden_dim_warns_and_falls_back(self):
        """A hidden dim that isn't a multiple of the stick size warns and
        returns None so the caller uses the default layout."""
        w = torch.randn(1000, 100, dtype=torch.float16)  # 100 % 64 != 0
        with self.assertWarns(UserWarning):
            dev = _dma_to_spyre_indirect_access(w)
        self.assertIsNone(dev)

    @requires_spyre
    def test_indirect_access_rejects_non_2d(self):
        """The indirect-access helper only accepts 2D tables."""
        with self.assertRaises(AssertionError):
            _dma_to_spyre_indirect_access(torch.randn(4, dtype=torch.float16))

    @requires_spyre
    def test_moe_expert_weight_layout(self):
        from torch_spyre._C import get_spyre_tensor_layout

        weight = torch.randn(3, 64, 128, dtype=torch.float16)
        device_weight = dma_moe_expert_weight_to_spyre(weight)

        layout = get_spyre_tensor_layout(device_weight)
        self.assertEqual(list(layout.device_size), [3, 64, 2, 64])
        torch.testing.assert_close(
            device_weight.cpu(), weight, rtol=DLFLOAT16_RTOL, atol=DLFLOAT16_ATOL
        )

    @requires_spyre
    def test_moe_per_expert_scale_layout(self):
        from torch_spyre._C import get_spyre_tensor_layout

        scale = torch.arange(3, dtype=torch.float16)
        device_scale = dma_moe_per_expert_scale_to_spyre(scale)

        self.assertEqual(device_scale.shape, (3, 64))
        layout = get_spyre_tensor_layout(device_scale)
        self.assertEqual(list(layout.device_size), [3, 1, 64])
        torch.testing.assert_close(device_scale.cpu(), scale[:, None].expand(-1, 64))

    @requires_spyre
    def test_load_model_routes_embedding_through_indirect_access(self):
        """An nn.Embedding table lands on Spyre with the indirect-access layout
        (vocab dim outermost), not the default layout."""
        from torch_spyre._C import get_spyre_tensor_layout

        model = nn.Embedding(1000, 256, dtype=torch.float16)
        load_model_to_spyre(model)

        self.assertEqual(model.weight.device.type, "spyre")
        layout = get_spyre_tensor_layout(model.weight)
        self.assertEqual(list(layout.device_size), [1000, 4, 64])

    @requires_spyre
    def test_load_model_embedding_non_tiling_falls_back(self):
        """An embedding whose hidden dim doesn't tile still loads (default
        layout) and warns."""
        model = nn.Embedding(1000, 100, dtype=torch.float16)
        with self.assertWarns(UserWarning):
            load_model_to_spyre(model)
        self.assertEqual(model.weight.device.type, "spyre")

    # ── routing ────────────────────────────────────────────────────

    @requires_spyre
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

    @requires_spyre
    def test_load_model_handles_layernorm(self):
        """Non-Linear params/buffers reach Spyre via the default path."""
        model = nn.LayerNorm(128, dtype=torch.float16)
        load_model_to_spyre(model)
        self.assertEqual(model.weight.device.type, "spyre")
        self.assertEqual(model.bias.device.type, "spyre")

    # ── _apply is honored (matches nn.Module.to semantics) ─────────

    @requires_spyre
    def test_submodule_apply_override_keeps_params_on_cpu(self):
        """A child that overrides _apply governs its own subtree: its params
        stay on CPU while a sibling Linear still gets the dim_order layout."""
        from torch_spyre._C import get_spyre_tensor_layout

        class KeepOnCpu(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 4, dtype=torch.float16))

            def _apply(self, fn, recurse=True):
                return self

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(64, 128, dtype=torch.float16)
                self.keep = KeepOnCpu()

        model = Root()
        load_model_to_spyre(model)

        self.assertEqual(model.keep.weight.device.type, "cpu")
        self.assertEqual(model.lin.weight.device.type, "spyre")
        self.assertEqual(get_spyre_tensor_layout(model.lin.weight).device_size[0], 2)

    @requires_spyre
    def test_instance_apply_override_keeps_params_on_cpu(self):
        """An instance-level _apply monkeypatch (the runner's Attention pattern)
        is honored just like a class-level override."""
        model = nn.Linear(8, 8, dtype=torch.float16)
        model._apply = lambda fn, recurse=True, _m=model: _m
        load_model_to_spyre(model)
        self.assertEqual(model.weight.device.type, "cpu")

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
    @requires_spyre
    def test_explicit_dtype_is_honored(self, src_dtype, target_dtype):
        """model.to('spyre', dtype=X) puts every param on device as X.
        copy_tensor (PR #2258) converts during the DMA."""
        model = nn.Linear(64, 128, dtype=src_dtype)
        load_model_to_spyre(model, dtype=target_dtype)
        for p in model.parameters():
            self.assertEqual(p.device.type, "spyre")
            self.assertEqual(p.dtype, target_dtype)

    @requires_spyre
    def test_dtype_none_preserves_source(self):
        """Without dtype, each tensor keeps its source dtype on device."""
        model = nn.Linear(64, 128, dtype=torch.float32)
        load_model_to_spyre(model)
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(model.bias.dtype, torch.float32)

    @requires_spyre
    def test_unsupported_dtype_raises(self):
        """Dtypes that map to DataFormats.INVALID are rejected."""
        model = nn.Linear(4, 4)
        with self.assertRaises(ValueError):
            load_model_to_spyre(model, dtype=torch.complex64)

    # ── idempotency ────────────────────────────────────────────────

    @requires_spyre
    def test_load_model_idempotent(self):
        """Second call on an already-loaded model is a no-op
        (params already on Spyre are skipped)."""
        model = nn.Linear(64, 128, dtype=torch.float16)
        load_model_to_spyre(model)
        ptr = model.weight.data_ptr()
        load_model_to_spyre(model)
        self.assertEqual(model.weight.data_ptr(), ptr)

    # ── nn.Module.to() patch ───────────────────────────────────────

    @requires_spyre
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

    @requires_spyre
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


# ── HF from_pretrained open-target monkey-patch ────────────────────
# Companion to safetensors patch (#3962): stock HF still opens
# checkpoints with device="cpu". These tests cover the resolver and the
# wrapper that force safe_open(device="spyre") when device_map is a
# uniform Spyre map. No hardware required except the last test.


class TestHfSafeOpenMonkeypatch(TestCase):
    """CPU tests for _resolve_checkpoint_open_target + the HF wrapper."""

    def setUp(self):
        # The wrapper tests rebind ``_original`` to a stub. Remember the real
        # transformers loader so tearDown can put it back; otherwise the stub
        # leaks into every later test in the process.
        self._hf_original = None
        if not HAS_TRANSFORMERS:
            return
        import transformers.modeling_utils as mu

        existing = mu.PreTrainedModel._load_pretrained_model
        self._hf_original = (
            existing._original
            if getattr(existing, "_spyre_hf_open_patched", False)
            else existing
        )

    def tearDown(self):
        if self._hf_original is None:
            return
        import transformers.modeling_utils as mu

        from torch_spyre._monkey_patch import (
            _patch_transformers_safe_open_for_spyre,
        )

        mu.PreTrainedModel._load_pretrained_model = staticmethod(self._hf_original)
        _patch_transformers_safe_open_for_spyre()

    def _run_wrapper_with_device_map(self, device_map):
        """Install the wrapper over a stock-HF stub that always opens on CPU.

        Returns the ``(device, backend)`` the stub's ``safe_open`` call was
        actually rewritten to.
        """
        import safetensors as st
        import transformers.modeling_utils as mu

        from torch_spyre._monkey_patch import (
            _patch_transformers_safe_open_for_spyre,
        )

        opened = {}

        def fake_safe_open(
            filename, framework="pt", device=None, backend="mmap", **kw
        ):
            opened["device"] = device
            opened["backend"] = backend
            return mock.MagicMock()

        existing = mu.PreTrainedModel._load_pretrained_model
        if getattr(existing, "_spyre_hf_open_patched", False):
            mu.PreTrainedModel._load_pretrained_model = staticmethod(
                existing._original
            )

        with mock.patch.object(st, "safe_open", fake_safe_open):
            _patch_transformers_safe_open_for_spyre()
            wrapped = mu.PreTrainedModel._load_pretrained_model
            self.assertTrue(getattr(wrapped, "_spyre_hf_open_patched", False))

            def stock_hf_opens_on_cpu(*args, **kwargs):
                mu.safe_open(
                    "ckpt.safetensors",
                    framework="pt",
                    device="cpu",
                    backend="mmap",
                )
                return None

            wrapped._original = stock_hf_opens_on_cpu
            wrapped(None, None, [], SimpleNamespace(device_map=device_map))

        return opened["device"], opened["backend"]

    def _resolve(self, device_map):
        from torch_spyre._monkey_patch import (
            _ensure_safetensors_custom_device_stubs,
            _resolve_checkpoint_open_target,
        )

        _ensure_safetensors_custom_device_stubs()
        return _resolve_checkpoint_open_target(device_map)

    def test_resolve_none_is_cpu(self):
        self.assertEqual(self._resolve(None), ("mmap", "cpu"))

    def test_resolve_cpu_map_is_cpu(self):
        self.assertEqual(self._resolve({"": "cpu"}), ("mmap", "cpu"))

    def test_resolve_mps(self):
        self.assertEqual(
            self._resolve({"": torch.device("mps")}), ("pread", "mps")
        )

    def test_resolve_mixed_map_falls_back(self):
        from torch_spyre.constants import DEVICE_NAME

        self.assertEqual(
            self._resolve({"": DEVICE_NAME, "lm_head": "cpu"}),
            ("mmap", "cpu"),
        )

    def test_resolve_spyre_when_stub_installed(self):
        from torch_spyre.constants import DEVICE_NAME

        self.assertEqual(
            self._resolve({"": DEVICE_NAME}), ("mmap", DEVICE_NAME)
        )

    def test_resolve_unknown_device_without_hook_is_cpu(self):
        self.assertEqual(self._resolve({"": "npu"}), ("mmap", "cpu"))

    @requires_transformers
    @requires_safetensors
    def test_wrapper_forces_safe_open_device_to_spyre(self):
        """Stock HF opens on CPU; the wrapper must rewrite that to Spyre."""
        from torch_spyre.constants import DEVICE_NAME

        device, backend = self._run_wrapper_with_device_map({"": DEVICE_NAME})
        self.assertEqual(device, DEVICE_NAME)
        self.assertEqual(backend, "mmap")

    @requires_transformers
    @requires_safetensors
    def test_wrapper_cpu_device_map_stays_on_cpu(self):
        """device_map='cpu' (hf-adapters today) must not be rewritten to Spyre."""
        device, backend = self._run_wrapper_with_device_map({"": "cpu"})
        self.assertEqual(device, "cpu")
        self.assertEqual(backend, "mmap")

    @requires_transformers
    @requires_safetensors
    def test_wrapper_restores_real_loader_between_tests(self):
        """tearDown must put the genuine transformers loader back."""
        import transformers.modeling_utils as mu

        self._run_wrapper_with_device_map({"": "cpu"})
        installed = mu.PreTrainedModel._load_pretrained_model
        self.assertIsNot(installed._original, self._hf_original)
        self.tearDown()
        restored = mu.PreTrainedModel._load_pretrained_model
        self.assertIs(restored._original, self._hf_original)

    @requires_spyre
    @requires_safetensors
    def test_forced_spyre_open_lands_linear_layout(self):
        """Resolver +  safe_open: a Linear .weight gets dim_order=[1,0]."""
        import os
        import tempfile

        import safetensors.torch as st_torch
        from torch_spyre._C import get_spyre_tensor_layout
        from torch_spyre.constants import DEVICE_NAME
        from torch_spyre._monkey_patch import (
            _ensure_safetensors_custom_device_stubs,
            _resolve_checkpoint_open_target,
        )

        _ensure_safetensors_custom_device_stubs()
        backend, device = _resolve_checkpoint_open_target({"": DEVICE_NAME})
        self.assertEqual((backend, device), ("mmap", DEVICE_NAME))

        weight = torch.randn(128, 64, dtype=torch.float16)
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        tmp.close()
        try:
            st_torch.save_file(
                {"model.layers.0.self_attn.q_proj.weight": weight}, tmp.name
            )
            loaded = st_torch.load_file(tmp.name, device=device)
            tensor = loaded["model.layers.0.self_attn.q_proj.weight"]
            self.assertEqual(tensor.device.type, DEVICE_NAME)
            layout = get_spyre_tensor_layout(tensor)
            self.assertEqual(layout.device_size[0], 2)  # 128/64
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    run_tests()
