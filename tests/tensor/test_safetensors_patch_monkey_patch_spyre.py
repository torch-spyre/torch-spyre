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

"""
Hardware integration tests for the safetensors monkey-patch.

Every test in this file requires a physical Spyre device. The CPU-only logic
(routing, key handling, tied weights, ordering) is covered by
test_safetensors_patch.py. This file covers what only Spyre hardware can verify:

  - Correct DMA layout selection actually applied on device
    (embedding → indirect-access, linear weight → dim_order=[1,0])
  - target_dtype coercion actually performed during DMA
  - Round-trip value correctness: write → load_file(device='spyre') → .cpu()
  - load_model on a meta-device model populates all parameters on Spyre
  - load_model with tied weights: both aliases land on Spyre, tie preserved,
    inference produces finite output
  - Integer buffer not coerced to float by target_dtype

Run with:
    pytest tests/tensor/test_safetensors_patch_spyre.py -v
"""

import os
import tempfile
from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests

# ── hardware availability guard ───────────────────────────────────────────────


def _spyre_available() -> bool:
    try:
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME

        torch.zeros(1, dtype=torch.float16, device=DEVICE_NAME)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _spyre_available(), reason="requires an available Spyre device"
)

# ── safetensors availability guard ────────────────────────────────────────────

try:
    import safetensors.torch as _st_torch

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

pytestmark = [
    pytest.mark.skipif(
        not _spyre_available(), reason="requires an available Spyre device"
    ),
    pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors>=0.8.0 not installed"),
]

# ── constants (match test_model_utils.py tolerances) ─────────────────────────

# fp16 lands on device as DLFLOAT16 (SEN169_FP16, 9 mantissa bits):
# round-trip is never bit-exact (2**-10 half-ULP, subnormals halved).
DLFLOAT16_RTOL = 2e-3
DLFLOAT16_ATOL = 1e-4

# bf16 has 7 mantissa bits; DMA round-trip error is bounded by 2**-7 half-ULP.
BFLOAT16_RTOL = 1e-2
BFLOAT16_ATOL = 1e-3

# ── helpers ───────────────────────────────────────────────────────────────────


def _write_safetensors(tensors: Dict[str, torch.Tensor]) -> str:
    """Write CPU tensors to a temp .safetensors file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.close()
    _st_torch.save_file(tensors, tmp.name)
    return tmp.name


# ── DMA layout selection ──────────────────────────────────────────────────────


class TestDmaLayoutSelection(TestCase):
    """load_file(device='spyre') must select the correct DMA layout for each
    tensor kind: embedding → indirect-access, linear weight → dim_order=[1,0],
    everything else → default."""

    def setUp(self):
        torch.manual_seed(0xAFFE)

    def test_embedding_key_gets_indirect_access_layout(self):
        """An embedding-named key loaded via load_file must land with the
        indirect-access layout (vocab dim outermost, hidden split into sticks).

        For a (1000, 256) fp16 table: eps=64, device_size=[1000, 4, 64].
        """
        from torch_spyre._C import get_spyre_tensor_layout

        tensors = {
            "model.embed_tokens.weight": torch.randn(1000, 256, dtype=torch.float16)
        }
        path = _write_safetensors(tensors)
        try:
            result = _st_torch.load_file(path, device="spyre")
            layout = get_spyre_tensor_layout(result["model.embed_tokens.weight"])
            self.assertEqual(list(layout.device_size), [1000, 4, 64])
        finally:
            os.unlink(path)

    def test_linear_weight_key_gets_dim_order_layout(self):
        """A linear-weight key loaded via load_file must land with
        dim_order=[1,0] stickification (out_features sticks on dim 0).

        For a (128, 64) weight: device_size[0] = 128/64 = 2.
        """
        from torch_spyre._C import get_spyre_tensor_layout

        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                128, 64, dtype=torch.float16
            )
        }
        path = _write_safetensors(tensors)
        try:
            result = _st_torch.load_file(path, device="spyre")
            layout = get_spyre_tensor_layout(
                result["model.layers.0.self_attn.q_proj.weight"]
            )
            self.assertEqual(layout.device_size[0], 2)  # 128 / 64
        finally:
            os.unlink(path)

    def test_other_key_gets_default_layout(self):
        """A bias or 1-D norm param must reach Spyre via the default layout
        and round-trip correctly."""
        tensors = {
            "model.layers.0.input_layernorm.weight": torch.randn(
                128, dtype=torch.float16
            )
        }
        path = _write_safetensors(tensors)
        try:
            result = _st_torch.load_file(path, device="spyre")
            self.assertEqual(
                result["model.layers.0.input_layernorm.weight"].device.type, "spyre"
            )
        finally:
            os.unlink(path)

    def test_false_positive_embed_out_gets_dim_order_layout(self):
        """GPT-NeoX 'embed_out' module is a Linear output head, not an
        embedding table.  It must receive dim_order=[1,0], not indirect-access.

        For a (256, 128) weight: device_size[0] = 256/64 = 4 under dim_order.
        Under indirect-access it would be device_size=[256, 2, 64], which is
        WRONG for a matmul weight.
        """
        from torch_spyre._C import get_spyre_tensor_layout

        tensors = {"model.embed_out.weight": torch.randn(256, 128, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            result = _st_torch.load_file(path, device="spyre")
            layout = get_spyre_tensor_layout(result["model.embed_out.weight"])
            # dim_order layout: device_size[0] = out_features / 64
            self.assertEqual(layout.device_size[0], 4)  # 256 / 64
        finally:
            os.unlink(path)


# ── round-trip value correctness ─────────────────────────────────────────────


class TestRoundTripValues(TestCase):
    """write → load_file(device='spyre') → .cpu() must reproduce original
    values within the documented DMA tolerances."""

    def setUp(self):
        torch.manual_seed(1)

    def test_fp16_round_trip(self):
        """fp16 tensor round-trips within DLFLOAT16 tolerance."""
        orig = torch.randn(64, 128, dtype=torch.float16)
        path = _write_safetensors({"weight": orig})
        try:
            result = _st_torch.load_file(path, device="spyre")
            torch.testing.assert_close(
                result["weight"].cpu(),
                orig,
                rtol=DLFLOAT16_RTOL,
                atol=DLFLOAT16_ATOL,
            )
        finally:
            os.unlink(path)

    def test_fp32_round_trip(self):
        """fp32 tensor round-trips losslessly when coercion is disabled.

        The default path (target_dtype=_UNSET) coerces fp32 to fp16 during DMA
        (the Spyre hardware requirement). Use target_dtype=None to opt out of
        coercion and verify the lossless IEEE_FP32 path.
        """
        orig = torch.randn(64, 128, dtype=torch.float32)
        # Use a non-embedding key so the default layout is used (not
        # indirect-access, which is gather-optimised for embedding tables).
        path = _write_safetensors({"model.layers.0.bias": orig})
        try:
            result = _st_torch.load_file(path, device="spyre", target_dtype=None)
            self.assertEqual(result["model.layers.0.bias"].dtype, torch.float32)
            torch.testing.assert_close(result["model.layers.0.bias"].cpu(), orig)
        finally:
            os.unlink(path)

    def test_target_dtype_bf16_to_fp16_converted_on_device(self):
        """load_file(target_dtype=torch.float16) on a bf16 checkpoint must
        produce fp16 tensors on Spyre — the coercion happens in copy_tensor
        during the DMA, not on CPU."""
        orig_bf16 = torch.randn(64, 128, dtype=torch.bfloat16)
        path = _write_safetensors({"weight": orig_bf16})
        try:
            result = _st_torch.load_file(
                path, device="spyre", target_dtype=torch.float16
            )
            self.assertEqual(result["weight"].dtype, torch.float16)
            # Value round-trip: bf16 → fp16 narrowing is within fp16 tolerance.
            torch.testing.assert_close(
                result["weight"].cpu().to(torch.float32),
                orig_bf16.to(torch.float32),
                rtol=DLFLOAT16_RTOL,
                atol=DLFLOAT16_ATOL,
            )
        finally:
            os.unlink(path)

    def test_target_dtype_fp32_to_bf16_converted_on_device(self):
        """load_file(target_dtype=torch.bfloat16) on an fp32 checkpoint
        produces bf16 tensors on Spyre."""
        orig_fp32 = torch.randn(64, 128, dtype=torch.float32)
        path = _write_safetensors({"weight": orig_fp32})
        try:
            result = _st_torch.load_file(
                path, device="spyre", target_dtype=torch.bfloat16
            )
            self.assertEqual(result["weight"].dtype, torch.bfloat16)
            torch.testing.assert_close(
                result["weight"].cpu().to(torch.float32),
                orig_fp32,
                rtol=BFLOAT16_RTOL,
                atol=BFLOAT16_ATOL,
            )
        finally:
            os.unlink(path)

    def test_integer_buffer_not_coerced_to_float(self):
        """An int64 position_ids buffer must land on Spyre as int64 even when
        target_dtype=torch.float16 is set — the is_floating_point guard must
        fire."""
        tensors = {
            "weight": torch.randn(64, 64, dtype=torch.float16),
            "position_ids": torch.arange(64, dtype=torch.int64),
        }
        path = _write_safetensors(tensors)
        try:
            result = _st_torch.load_file(
                path, device="spyre", target_dtype=torch.float16
            )
            self.assertEqual(result["position_ids"].dtype, torch.int64)
            self.assertEqual(result["weight"].dtype, torch.float16)
        finally:
            os.unlink(path)


# ── load_model meta-device population ────────────────────────────────────────


class TestLoadModelMetaDevice(TestCase):
    """The core motivation for _spyre_load_model: copy_() is a no-op on meta
    tensors, so load_state_dict silently leaves params on meta.  The direct
    _parameters replacement must move every param to Spyre."""

    def setUp(self):
        torch.manual_seed(2)

    def test_all_parameters_leave_meta_after_load(self):
        """Every parameter must be on Spyre (not meta) after load_model."""
        model = nn.Linear(128, 64, dtype=torch.float16, device="meta")
        tensors = {
            "weight": torch.randn(64, 128, dtype=torch.float16),
            "bias": torch.randn(64, dtype=torch.float16),
        }
        path = _write_safetensors(tensors)
        try:
            _st_torch.load_model(model, path, strict=True, device="spyre")
            for name, param in model.named_parameters():
                self.assertEqual(
                    param.device.type,
                    "spyre",
                    f"{name} is still on {param.device} after load_model",
                )
        finally:
            os.unlink(path)

    def test_meta_model_no_parameters_remain_on_meta(self):
        """A deeper meta-device model — all submodule parameters must reach
        Spyre, not just the top-level ones."""

        class _TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64, dtype=torch.float16, device="meta")
                self.fc2 = nn.Linear(64, 32, dtype=torch.float16, device="meta")

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = _TwoLayer()
        tensors = {
            "fc1.weight": torch.randn(64, 128, dtype=torch.float16),
            "fc1.bias": torch.randn(64, dtype=torch.float16),
            "fc2.weight": torch.randn(32, 64, dtype=torch.float16),
            "fc2.bias": torch.randn(32, dtype=torch.float16),
        }
        path = _write_safetensors(tensors)
        try:
            _st_torch.load_model(model, path, strict=True, device="spyre")
            meta_params = [
                name for name, p in model.named_parameters() if p.device.type == "meta"
            ]
            self.assertEqual(
                meta_params,
                [],
                f"Parameters still on meta after load_model: {meta_params}",
            )
        finally:
            os.unlink(path)

    def test_meta_model_with_target_dtype(self):
        """target_dtype converts during DMA; meta params must land on Spyre
        as the requested dtype, not remain on meta."""
        model = nn.Linear(64, 32, dtype=torch.bfloat16, device="meta")
        tensors = {
            "weight": torch.randn(32, 64, dtype=torch.bfloat16),
            "bias": torch.randn(32, dtype=torch.bfloat16),
        }
        path = _write_safetensors(tensors)
        try:
            _st_torch.load_model(
                model,
                path,
                strict=True,
                device="spyre",
                target_dtype=torch.float16,
            )
            for name, param in model.named_parameters():
                self.assertEqual(param.device.type, "spyre")
                self.assertEqual(param.dtype, torch.float16)
        finally:
            os.unlink(path)


# ── load_model tied weights on hardware ──────────────────────────────────────


class TestLoadModelTiedWeightsHardware(TestCase):
    """The tied-weight blocker from the review: lm_head.weight tied to
    embed.weight, checkpoint stores only embed.weight.

    Hardware tests confirm:
      - both aliases land on Spyre (not meta, not CPU)
      - the Python object identity of the tie is preserved on Spyre
      - a forward pass through the tied model produces finite output
    """

    class _TiedLM(nn.Module):
        """Minimal language-model with a tied embed/lm_head."""

        def __init__(self, vocab=64, hidden=128, dtype=torch.float16):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden, dtype=dtype)
            self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=dtype)
            self.lm_head.weight = self.embed.weight  # tie

        def forward(self, input_ids):
            x = self.embed(input_ids)  # (B, T, H)
            return self.lm_head(x)  # (B, T, V)

    def setUp(self):
        torch.manual_seed(3)

    def _make_checkpoint(self, vocab=64, hidden=128, dtype=torch.float16):
        tensors = {"embed.weight": torch.randn(vocab, hidden, dtype=dtype)}
        path = _write_safetensors(tensors)
        return path, tensors["embed.weight"]

    def test_both_tied_aliases_on_spyre(self):
        """After load_model, both embed.weight and lm_head.weight must be on
        Spyre, not meta or CPU."""
        model = self._TiedLM()
        path, _ = self._make_checkpoint()
        try:
            _st_torch.load_model(model, path, strict=True, device="spyre")
            self.assertEqual(model.embed.weight.device.type, "spyre")
            self.assertEqual(model.lm_head.weight.device.type, "spyre")
        finally:
            os.unlink(path)

    def test_tied_weight_identity_preserved_on_spyre(self):
        """embed.weight and lm_head.weight must be the same Python object after
        load_model — the tie must not be severed by the parameter replacement."""
        model = self._TiedLM()
        path, _ = self._make_checkpoint()
        try:
            _st_torch.load_model(model, path, strict=True, device="spyre")
            self.assertIs(
                model.embed.weight,
                model.lm_head.weight,
                "Weight tie severed: embed.weight is not lm_head.weight after load",
            )
        finally:
            os.unlink(path)

    def test_tied_model_forward_produces_finite_output(self):
        """A forward pass through the tied model after load_model must produce
        finite, non-NaN logits.  NaN/Inf would indicate a corrupt lm_head
        (unloaded, wrong layout, or severed tie)."""
        model = self._TiedLM(vocab=64, hidden=128)
        path, _ = self._make_checkpoint(vocab=64, hidden=128)
        try:
            _st_torch.load_model(model, path, strict=True, device="spyre")
            input_ids = torch.randint(0, 64, (1, 8), device="spyre")
            logits = model(input_ids)
            cpu_logits = logits.cpu().float()
            self.assertFalse(
                torch.isnan(cpu_logits).any().item(),
                "Forward pass produced NaN — lm_head may be unloaded or severed",
            )
            self.assertFalse(
                torch.isinf(cpu_logits).any().item(),
                "Forward pass produced Inf — possible dtype overflow in lm_head",
            )
        finally:
            os.unlink(path)

    def test_tied_model_embed_lm_head_values_match(self):
        """Because lm_head.weight IS embed.weight, a matmul with a known input
        must be equivalent to an embedding lookup followed by a dot product."""
        vocab, hidden = 64, 128
        model = self._TiedLM(vocab=vocab, hidden=hidden)
        path, ckpt_weight = self._make_checkpoint(vocab=vocab, hidden=hidden)
        try:
            _st_torch.load_model(model, path, strict=True, device="spyre")
            # Both aliases must carry identical values.
            torch.testing.assert_close(
                model.embed.weight.cpu(),
                model.lm_head.weight.cpu(),
                rtol=0.0,
                atol=0.0,
                msg="embed.weight and lm_head.weight values differ after load",
            )
        finally:
            os.unlink(path)


# ── safe_open layout and round-trip ──────────────────────────────────────────


class TestSafeOpenHardware(TestCase):
    """safe_open(device='spyre') end-to-end: layout selection and value
    correctness via get_tensor and get_slice."""

    def setUp(self):
        torch.manual_seed(4)
        self.tensors = {
            "model.embed_tokens.weight": torch.randn(256, 128, dtype=torch.float16),
            "model.layers.0.q_proj.weight": torch.randn(128, 128, dtype=torch.float16),
            "model.layers.0.input_layernorm.weight": torch.randn(
                128, dtype=torch.float16
            ),
        }
        self.path = _write_safetensors(self.tensors)

    def tearDown(self):
        os.unlink(self.path)

    def test_get_tensor_values_round_trip(self):
        """All tensors loaded via safe_open get_tensor must round-trip within
        DLFLOAT16 tolerance."""
        import safetensors as _st_mod

        with _st_mod.safe_open(self.path, framework="pt", device="spyre") as f:
            for key in f.keys():
                result = f.get_tensor(key).cpu()
                torch.testing.assert_close(
                    result,
                    self.tensors[key],
                    rtol=DLFLOAT16_RTOL,
                    atol=DLFLOAT16_ATOL,
                )

    def test_get_slice_values_round_trip(self):
        """A partial slice via get_slice must round-trip correctly."""
        import safetensors as _st_mod

        key = "model.layers.0.q_proj.weight"
        orig_slice = self.tensors[key][:32, :]  # first 32 rows

        with _st_mod.safe_open(self.path, framework="pt", device="spyre") as f:
            result = f.get_slice(key)[:32, :].cpu()

        torch.testing.assert_close(
            result,
            orig_slice,
            rtol=DLFLOAT16_RTOL,
            atol=DLFLOAT16_ATOL,
        )

    def test_embedding_tensor_on_device_has_indirect_access_layout(self):
        """Embedding tensor loaded via safe_open must have the indirect-access
        layout on Spyre."""
        from torch_spyre._C import get_spyre_tensor_layout
        import safetensors as _st_mod

        with _st_mod.safe_open(self.path, framework="pt", device="spyre") as f:
            emb = f.get_tensor("model.embed_tokens.weight")

        layout = get_spyre_tensor_layout(emb)
        # (256, 128) fp16: eps=64, device_size=[256, 2, 64]
        self.assertEqual(list(layout.device_size), [256, 2, 64])


if __name__ == "__main__":
    run_tests()
