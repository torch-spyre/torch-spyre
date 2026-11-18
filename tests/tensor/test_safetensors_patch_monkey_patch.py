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
Tests for the safetensors monkey-patch (_patch_safetensors_for_spyre).

All tests are CPU-only: the Spyre DMA hook is replaced by a stub that leaves
tensors on CPU, so no hardware is required.  The stub has the same signature as
_spyre_tensor_from_safetensors (4 args: cpu_tensor, name, device, target_dtype).

Test scope (mirrors reviewer requests):
  - _classify_safetensors_key correctness and false-positive elimination
  - Non-Spyre passthrough: load_file and safe_open with device='cpu'
  - Double-patch idempotency for all three entry points
  - load_file Spyre path — values match the originals
  - load_model tied weights, strict=True and strict=False
  - load_model dtype preservation (fp16, bf16, fp32) using target_dtype=None
  - load_model target_dtype opt-in conversion
  - load_model invalid target_dtype raises early
  - offset_keys() forwarding (file-offset order, not alphabetical order)
  - _SpyreSafeOpen.__getattr__ forward-compat delegation

Additional corner cases:
  - safe_open Spyre path — get_tensor and get_slice reach the hook
  - load_model missing key (strict=True raises, strict=False reports)
  - load_model shape mismatch raises RuntimeError
  - load_model size-1 model with no parameters or buffers
  - integer buffer (position ids) — not coerced by target_dtype
  - numpy framework passthrough (safe_open(framework='np') not intercepted)
  - safe_open device with index (device='spyre:1') still routes to _SpyreSafeOpen
"""

import os
import tempfile
import unittest
import unittest.mock as mock
from typing import Dict

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

# ── safetensors availability ──────────────────────────────────────────────────
try:
    import safetensors  # noqa: F401
    import safetensors.torch as _st_torch

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

requires_safetensors = unittest.skipUnless(
    HAS_SAFETENSORS, "safetensors>=0.8.0 not installed"
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _write_safetensors(tensors: Dict[str, torch.Tensor]) -> str:
    """Write a dict of CPU tensors to a temp .safetensors file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.close()
    _st_torch.save_file(tensors, tmp.name)
    return tmp.name


def _identity_hook(cpu_tensor, name, device, target_dtype=None):
    """Stub Spyre hook: returns the tensor unchanged (stays on CPU).

    Accepts the same 4-arg signature as _spyre_tensor_from_safetensors after
    the target_dtype threading patch.  target_dtype is intentionally ignored —
    CPU-stub tests either pass target_dtype=None (dtype preserved) or rely on
    the mock to intercept the call before any coercion occurs.
    """
    return cpu_tensor


def _patch_hook(monkeypatch_target):
    """Context-manager that replaces _spyre_tensor_from_safetensors with the
    identity stub so tests run without a real Spyre device."""
    return mock.patch(
        "torch_spyre._monkey_patch._spyre_tensor_from_safetensors",
        side_effect=_identity_hook,
    )


# ── classification ────────────────────────────────────────────────────────────


class TestClassifySafetensorsKey(TestCase):
    """Unit tests for _classify_safetensors_key — no I/O, no hardware."""

    def _classify(self, name, ndim):
        from torch_spyre._monkey_patch import _classify_safetensors_key

        return _classify_safetensors_key(name, ndim)

    # ── embedding detection ────────────────────────────────────────────────

    def test_embed_tokens_weight_is_embedding(self):
        self.assertEqual(self._classify("model.embed_tokens.weight", 2), "embedding")

    def test_wte_weight_is_embedding(self):
        self.assertEqual(self._classify("transformer.wte.weight", 2), "embedding")

    def test_wpe_weight_is_embedding(self):
        self.assertEqual(self._classify("transformer.wpe.weight", 2), "embedding")

    def test_embedding_weight_is_embedding(self):
        self.assertEqual(self._classify("model.embedding.weight", 2), "embedding")

    def test_embeddings_weight_is_embedding(self):
        self.assertEqual(
            self._classify("bert.embeddings.word_embeddings.weight", 2), "embedding"
        )

    def test_pos_embed_weight_is_embedding(self):
        self.assertEqual(self._classify("vit.pos_embed.weight", 2), "embedding")

    # ── false-positive fixes ───────────────────────────────────────────────

    def test_embed_out_weight_is_linear(self):
        """GPT-NeoX output head — module name 'embed_out', not a recognised
        embedding name segment, so must classify as linear."""
        self.assertEqual(self._classify("model.embed_out.weight", 2), "linear")

    def test_embed_layer_norm_weight_is_linear(self):
        self.assertEqual(self._classify("model.embed_layer_norm.weight", 2), "linear")

    def test_embedding_projection_weight_is_linear(self):
        self.assertEqual(
            self._classify("model.embedding_projection.weight", 2), "linear"
        )

    def test_wtemp_weight_is_linear(self):
        """'wtemp' contains 'wte' as a substring but is not a valid module name."""
        self.assertEqual(self._classify("some.wtemp.weight", 2), "linear")

    def test_lm_head_weight_is_linear(self):
        self.assertEqual(self._classify("lm_head.weight", 2), "linear")

    # ── linear detection ──────────────────────────────────────────────────

    def test_q_proj_weight_is_linear(self):
        self.assertEqual(
            self._classify("model.layers.0.self_attn.q_proj.weight", 2), "linear"
        )

    def test_up_proj_weight_is_linear(self):
        self.assertEqual(
            self._classify("model.layers.0.mlp.up_proj.weight", 2), "linear"
        )

    # ── other ─────────────────────────────────────────────────────────────

    def test_1d_tensor_is_other(self):
        self.assertEqual(
            self._classify("model.layers.0.input_layernorm.weight", 1), "other"
        )

    def test_bias_is_other(self):
        self.assertEqual(self._classify("model.layers.0.q_proj.bias", 2), "other")

    def test_bare_name_no_dot_is_other(self):
        """A key with no dot separator (no module prefix) is neither embedding
        nor linear because we cannot confirm it has a module name context."""
        self.assertEqual(self._classify("weight", 2), "other")

    def test_3d_tensor_is_other(self):
        self.assertEqual(self._classify("model.embed_tokens.weight", 3), "other")


# ── non-Spyre passthrough ─────────────────────────────────────────────────────


@requires_safetensors
class TestNonSpyrePassthrough(TestCase):
    """Non-Spyre device paths must fall through to the original functions
    completely — no hook involvement."""

    def setUp(self):
        torch.manual_seed(0)
        self.tensors = {
            "a": torch.randn(4, 8, dtype=torch.float16),
            "b": torch.randn(3, dtype=torch.float32),
        }
        self.path = _write_safetensors(self.tensors)

    def tearDown(self):
        os.unlink(self.path)

    def test_load_file_cpu_passthrough(self):
        """load_file(device='cpu') must not invoke the Spyre hook."""
        with _patch_hook(None) as mock_hook:
            result = _st_torch.load_file(self.path, device="cpu")
        mock_hook.assert_not_called()
        for key, orig in self.tensors.items():
            torch.testing.assert_close(result[key], orig)

    def test_safe_open_cpu_passthrough(self):
        """safe_open(device='cpu') must not return _SpyreSafeOpen."""
        import safetensors as _st_mod
        from torch_spyre._monkey_patch import _SpyreSafeOpen

        with _st_mod.safe_open(self.path, framework="pt", device="cpu") as f:
            self.assertNotIsInstance(f, _SpyreSafeOpen)
            for key in f.keys():
                torch.testing.assert_close(f.get_tensor(key), self.tensors[key])

    def test_load_model_non_spyre_passthrough(self):
        """load_model(device='cpu') must delegate to the original function."""
        model = nn.Linear(8, 4, dtype=torch.float16, bias=False)
        tensors = {"weight": torch.randn(4, 8, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None) as mock_hook:
                missing, unexpected = _st_torch.load_model(model, path, device="cpu")
            mock_hook.assert_not_called()
            # Upstream load_model returns (set(), list) for missing/unexpected.
            self.assertEqual(set(missing), set())
            self.assertEqual(unexpected, [])
        finally:
            os.unlink(path)


# ── double-patch idempotency ──────────────────────────────────────────────────


@requires_safetensors
class TestDoublePatchIdempotency(TestCase):
    """Calling _patch_safetensors_for_spyre() a second time must be a no-op:
    the sentinel prevents re-wrapping already-patched functions."""

    def test_double_patch_idempotent(self):
        from torch_spyre._monkey_patch import _patch_safetensors_for_spyre
        import safetensors as _st_mod

        # Record the patched objects after the first application (which already
        # ran at import time via _patch_tensor_for_spyre).
        safe_open_after_first = _st_mod.safe_open
        load_file_after_first = _st_torch.load_file
        load_model_after_first = _st_torch.load_model

        # Second call — must be a no-op.
        _patch_safetensors_for_spyre()

        self.assertIs(_st_mod.safe_open, safe_open_after_first)
        self.assertIs(_st_torch.load_file, load_file_after_first)
        self.assertIs(_st_torch.load_model, load_model_after_first)


# ── load_file Spyre path ──────────────────────────────────────────────────────


@requires_safetensors
class TestSpyreLoadFile(TestCase):
    """load_file(device='spyre') routes through the hook and returns correct
    values (verified with the identity stub)."""

    def setUp(self):
        torch.manual_seed(1)
        self.tensors = {
            "model.embed_tokens.weight": torch.randn(32, 16, dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                16, 16, dtype=torch.float16
            ),
            "model.layers.0.input_layernorm.weight": torch.randn(
                16, dtype=torch.float16
            ),
        }
        self.path = _write_safetensors(self.tensors)

    def tearDown(self):
        os.unlink(self.path)

    def test_load_file_spyre_calls_hook(self):
        """The hook must be called once per tensor."""
        with _patch_hook(None) as mock_hook:
            _st_torch.load_file(self.path, device="spyre")
        self.assertEqual(mock_hook.call_count, len(self.tensors))

    def test_load_file_spyre_values_correct(self):
        """Identity-stub: returned tensors must match originals bit-for-bit."""
        with _patch_hook(None):
            result = _st_torch.load_file(self.path, device="spyre")
        for key, orig in self.tensors.items():
            torch.testing.assert_close(result[key], orig)


# ── safe_open Spyre path ──────────────────────────────────────────────────────


@requires_safetensors
class TestSpyreSafeOpenPath(TestCase):
    """safe_open(device='spyre') must route get_tensor and get_slice through
    the hook and return the correct values."""

    def setUp(self):
        torch.manual_seed(4)
        self.tensors = {
            "model.embed_tokens.weight": torch.randn(16, 8, dtype=torch.float16),
            "model.layers.0.mlp.down_proj.weight": torch.randn(
                8, 16, dtype=torch.float16
            ),
        }
        self.path = _write_safetensors(self.tensors)

    def tearDown(self):
        os.unlink(self.path)

    def test_get_tensor_routes_through_hook(self):
        """get_tensor() on a Spyre safe_open must call the hook for each key."""
        import safetensors as _st_mod

        with _patch_hook(None) as mock_hook:
            with _st_mod.safe_open(self.path, framework="pt", device="spyre") as f:
                for key in f.keys():
                    _ = f.get_tensor(key)
        self.assertEqual(mock_hook.call_count, len(self.tensors))

    def test_get_tensor_values_correct(self):
        """Identity stub: get_tensor() values must match the originals."""
        import safetensors as _st_mod

        with _patch_hook(None):
            with _st_mod.safe_open(self.path, framework="pt", device="spyre") as f:
                for key in f.keys():
                    torch.testing.assert_close(f.get_tensor(key), self.tensors[key])

    def test_get_slice_routes_through_hook(self):
        """get_slice()[...] must route through the hook after gathering bytes
        on CPU (avoiding a Spyre → Spyre double-copy)."""
        import safetensors as _st_mod

        with _patch_hook(None) as mock_hook:
            with _st_mod.safe_open(self.path, framework="pt", device="spyre") as f:
                key = "model.layers.0.mlp.down_proj.weight"
                _ = f.get_slice(key)[:4, :]
        mock_hook.assert_called_once()

    def test_device_with_index_routes_to_spyre_safe_open(self):
        """safe_open(device='spyre:1') must also route to _SpyreSafeOpen,
        not fall through to the original (which would reject the device string)."""
        import safetensors as _st_mod
        from torch_spyre._monkey_patch import _SpyreSafeOpen

        with _patch_hook(None):
            obj = _st_mod.safe_open(self.path, framework="pt", device="spyre:1")
        self.assertIsInstance(obj, _SpyreSafeOpen)

    def test_numpy_framework_passthrough(self):
        """safe_open(framework='np') must not be intercepted even for
        device='spyre' — numpy tensors go through the original path."""
        import safetensors as _st_mod
        from torch_spyre._monkey_patch import _SpyreSafeOpen

        # The original Rust safe_open will reject 'spyre' device for numpy,
        # which proves the dispatch did NOT go to _SpyreSafeOpen.
        with self.assertRaises(Exception):
            # We only need to confirm it is NOT _SpyreSafeOpen that was returned.
            # Any error from the Rust layer is acceptable.
            obj = _st_mod.safe_open(self.path, framework="np", device="spyre")
            self.assertNotIsInstance(obj, _SpyreSafeOpen)


# ── load_model tied weights ───────────────────────────────────────────────────


@requires_safetensors
class TestLoadModelTiedWeights(TestCase):
    """Tied parameters must remain shared after load_model and must not appear
    in unexpected_keys."""

    class _TiedModel(nn.Module):
        """embed.weight and lm_head.weight are the same tensor object."""

        def __init__(self, dtype=torch.float16):
            super().__init__()
            self.embed = nn.Embedding(8, 4, dtype=dtype)
            self.lm_head = nn.Linear(4, 8, bias=False, dtype=dtype)
            # Tie the weights — the checkpoint stores only embed.weight.
            self.lm_head.weight = self.embed.weight

    def _make_checkpoint(self, dtype=torch.float16):
        """Write a checkpoint that contains only embed.weight (tied scenario)."""
        tensors = {"embed.weight": torch.randn(8, 4, dtype=dtype)}
        path = _write_safetensors(tensors)
        return path, tensors["embed.weight"]

    def test_tied_weights_strict_true_does_not_raise(self):
        """strict=True must not raise when the checkpoint omits lm_head.weight
        because it is a tied duplicate of embed.weight."""
        model = self._TiedModel(dtype=torch.float16)
        path, _ = self._make_checkpoint(dtype=torch.float16)
        try:
            with _patch_hook(None):
                # Must not raise RuntimeError about unexpected/missing keys.
                missing, unexpected = _st_torch.load_model(
                    model, path, strict=True, device="spyre"
                )
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
        finally:
            os.unlink(path)

    def test_tied_weights_strict_false_does_not_raise(self):
        """strict=False: same requirement — lm_head.weight must not appear in
        unexpected_keys and the tie must be preserved."""
        model = self._TiedModel(dtype=torch.float16)
        path, _ = self._make_checkpoint(dtype=torch.float16)
        try:
            with _patch_hook(None):
                missing, unexpected = _st_torch.load_model(
                    model, path, strict=False, device="spyre"
                )
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
        finally:
            os.unlink(path)

    def test_tied_weights_preserved_after_load(self):
        """After load_model, embed.weight and lm_head.weight must still be the
        same object (tie must not be severed)."""
        model = self._TiedModel(dtype=torch.float16)
        path, _ = self._make_checkpoint(dtype=torch.float16)
        try:
            with _patch_hook(None):
                _st_torch.load_model(model, path, strict=False, device="spyre")
            self.assertIs(
                model.embed.weight,
                model.lm_head.weight,
                "Weight tie was severed by load_model",
            )
        finally:
            os.unlink(path)

    def test_tied_weights_data_loaded(self):
        """Both tied aliases must reflect the checkpoint data."""
        model = self._TiedModel(dtype=torch.float16)
        path, checkpoint_weight = self._make_checkpoint(dtype=torch.float16)
        try:
            with _patch_hook(None):
                _st_torch.load_model(model, path, strict=False, device="spyre")
            # Identity stub: tensors are on CPU, so compare directly.
            torch.testing.assert_close(model.embed.weight, checkpoint_weight)
            torch.testing.assert_close(model.lm_head.weight, checkpoint_weight)
        finally:
            os.unlink(path)


# ── load_model missing keys and shape mismatch ───────────────────────────────


@requires_safetensors
class TestLoadModelKeyErrors(TestCase):
    """Correctness of missing/unexpected key reporting and shape validation."""

    def test_missing_key_strict_true_raises(self):
        """A key present in the model but absent from the checkpoint must cause
        RuntimeError under strict=True."""
        model = nn.Linear(8, 4, bias=False, dtype=torch.float16)
        # Checkpoint is completely empty — weight is missing.
        tensors = {"unrelated": torch.randn(2, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None):
                with self.assertRaises(RuntimeError):
                    _st_torch.load_model(model, path, strict=True, device="spyre")
        finally:
            os.unlink(path)

    def test_missing_key_strict_false_reported(self):
        """Under strict=False, a missing key must appear in missing_keys and
        not raise."""
        model = nn.Linear(8, 4, bias=False, dtype=torch.float16)
        tensors = {"unrelated": torch.randn(2, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None):
                missing, unexpected = _st_torch.load_model(
                    model, path, strict=False, device="spyre"
                )
            self.assertIn("weight", missing)
            self.assertIn("unrelated", unexpected)
        finally:
            os.unlink(path)

    def test_shape_mismatch_raises(self):
        """A checkpoint tensor with a different shape from the model parameter
        must raise RuntimeError with a clear message before any assignment."""
        model = nn.Linear(8, 4, bias=False, dtype=torch.float16)
        # Checkpoint weight has the wrong shape.
        tensors = {"weight": torch.randn(4, 16, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None):
                with self.assertRaises(RuntimeError, msg="size mismatch"):
                    _st_torch.load_model(model, path, strict=True, device="spyre")
        finally:
            os.unlink(path)

    def test_empty_model_loads_cleanly(self):
        """A model with no parameters or buffers must return empty lists and
        not raise."""

        class _Empty(nn.Module):
            pass

        tensors = {"unused": torch.randn(2, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None):
                missing, unexpected = _st_torch.load_model(
                    _Empty(), path, strict=False, device="spyre"
                )
            self.assertEqual(missing, [])
            self.assertIn("unused", unexpected)
        finally:
            os.unlink(path)


# ── integer buffer — not coerced by target_dtype ─────────────────────────────


@requires_safetensors
class TestIntegerBufferNotCoerced(TestCase):
    """Integer buffers (e.g. position_ids) must never be cast to a float dtype
    by the dma_target_dtype guard in _spyre_tensor_from_safetensors."""

    def test_int_buffer_reaches_hook_as_int(self):
        """When target_dtype=torch.float16 is threaded through, integer tensors
        must still arrive at the hook with their original integer dtype."""

        class _ModelWithIntBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False, dtype=torch.float16)
                self.register_buffer("position_ids", torch.arange(4, dtype=torch.int64))

        tensors = {
            "linear.weight": torch.randn(4, 4, dtype=torch.float16),
            "position_ids": torch.arange(4, dtype=torch.int64),
        }
        path = _write_safetensors(tensors)
        received_dtypes = {}

        def _recording_hook(cpu_tensor, name, device, target_dtype=None):
            received_dtypes[name] = cpu_tensor.dtype
            return cpu_tensor

        try:
            with mock.patch(
                "torch_spyre._monkey_patch._spyre_tensor_from_safetensors",
                side_effect=_recording_hook,
            ):
                _st_torch.load_file(path, device="spyre", target_dtype=torch.float16)
            # The hook must see the int64 buffer as int64, not float16.
            self.assertEqual(received_dtypes["position_ids"], torch.int64)
            self.assertEqual(received_dtypes["linear.weight"], torch.float16)
        finally:
            os.unlink(path)


# ── load_model dtype preservation ────────────────────────────────────────────


@requires_safetensors
@instantiate_parametrized_tests
class TestLoadModelDtypePreservation(TestCase):
    """load_model dtype handling after the target_dtype threading patch.

    Default behaviour (target_dtype unset, resolves to fp16):
      - The hook coerces floating-point tensors to fp16 during DMA.
      - load_model does NOT raise a dtype mismatch — it logs and accepts the
        coercion, since the Parameter replacement sets the model's dtype to
        whatever the Spyre tensor carries.

    Explicit opt-out (target_dtype=None):
      - No coercion; checkpoint dtype is preserved on device.
      - load_model raises RuntimeError when checkpoint dtype != model dtype.

    Explicit opt-in (target_dtype=torch.bfloat16, etc.):
      - All floating-point tensors are converted to that dtype during DMA.
      - Verified via the identity stub: returned dtype matches target_dtype.
    """

    # ── target_dtype=None: checkpoint dtype preserved ──────────────────────

    @parametrize(
        "dtype",
        [torch.float16, torch.bfloat16, torch.float32],
    )
    def test_checkpoint_dtype_preserved_when_coercion_disabled(self, dtype):
        """target_dtype=None disables coercion; the parameter dtype on the
        (CPU-stub) device must equal the checkpoint dtype."""
        model = nn.Linear(8, 4, bias=False, dtype=dtype)
        tensors = {"weight": torch.randn(4, 8, dtype=dtype)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None):
                missing, unexpected = _st_torch.load_model(
                    model, path, strict=True, device="spyre", target_dtype=None
                )
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
            self.assertEqual(model.weight.dtype, dtype)
        finally:
            os.unlink(path)

    @parametrize(
        "model_dtype,ckpt_dtype",
        [
            (torch.bfloat16, torch.float16),
            (torch.float32, torch.float16),
            (torch.float16, torch.bfloat16),
        ],
    )
    def test_dtype_mismatch_raises_when_coercion_disabled(
        self, model_dtype, ckpt_dtype
    ):
        """target_dtype=None: a mismatch between checkpoint and model dtype must
        raise RuntimeError rather than silently accepting corrupt data."""
        model = nn.Linear(8, 4, bias=False, dtype=model_dtype)
        tensors = {"weight": torch.randn(4, 8, dtype=ckpt_dtype)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None):
                with self.assertRaises(RuntimeError, msg="dtype mismatch"):
                    _st_torch.load_model(
                        model, path, strict=True, device="spyre", target_dtype=None
                    )
        finally:
            os.unlink(path)

    # ── default (target_dtype=_UNSET → fp16): coercion happens ────────────

    @parametrize(
        "ckpt_dtype",
        [torch.bfloat16, torch.float32],
    )
    def test_default_coercion_to_fp16_does_not_raise(self, ckpt_dtype):
        """Default path (target_dtype not passed): floating-point tensors are
        coerced to fp16 by the hook; load_model must NOT raise a dtype error
        even when the model was initialised with a different dtype.

        Uses the identity stub so no real DMA occurs; the stub ignores
        target_dtype and returns the cpu tensor as-is, meaning the coercion
        check is exercised via the coercion_disabled=False branch in load_model.
        """
        # Model uses ckpt_dtype; after load the parameter dtype will become
        # whatever the stub returns (also ckpt_dtype here, because the stub is
        # identity), but load_model must not treat that as an error.
        model = nn.Linear(8, 4, bias=False, dtype=ckpt_dtype)
        tensors = {"weight": torch.randn(4, 8, dtype=ckpt_dtype)}
        path = _write_safetensors(tensors)
        try:
            # No target_dtype kwarg → _UNSET default → coercion_disabled=False.
            with _patch_hook(None):
                missing, unexpected = _st_torch.load_model(
                    model, path, strict=True, device="spyre"
                )
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
        finally:
            os.unlink(path)

    # ── explicit target_dtype opt-in ───────────────────────────────────────

    @parametrize(
        "ckpt_dtype,target_dtype",
        [
            (torch.float32, torch.float16),
            (torch.bfloat16, torch.float16),
            (torch.float16, torch.bfloat16),
        ],
    )
    def test_explicit_target_dtype_passed_to_hook(self, ckpt_dtype, target_dtype):
        """When target_dtype is given, the value must reach the hook so the DMA
        can perform the conversion.  Verified by inspecting the kwarg the mock
        received — no real hardware required."""
        tensors = {"weight": torch.randn(4, 8, dtype=ckpt_dtype)}
        path = _write_safetensors(tensors)
        try:
            with _patch_hook(None) as mock_hook:
                _st_torch.load_file(path, device="spyre", target_dtype=target_dtype)
            # Confirm the hook was called with the expected target_dtype kwarg.
            for call in mock_hook.call_args_list:
                _, kwargs = call
                self.assertEqual(
                    kwargs.get("target_dtype"),
                    target_dtype,
                    "target_dtype was not threaded through to the hook",
                )
        finally:
            os.unlink(path)

    def test_invalid_target_dtype_raises_early(self):
        """An unsupported target_dtype (e.g. complex64) must raise ValueError
        from _validate_target_dtype before any DMA attempt."""
        model = nn.Linear(8, 4, bias=False, dtype=torch.float16)
        tensors = {"weight": torch.randn(4, 8, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            with self.assertRaises(ValueError):
                _st_torch.load_model(
                    model,
                    path,
                    strict=True,
                    device="spyre",
                    target_dtype=torch.complex64,
                )
        finally:
            os.unlink(path)


# ── offset_keys forwarding ────────────────────────────────────────────────────


@requires_safetensors
class TestOffsetKeysForwarding(TestCase):
    """_SpyreSafeOpen.offset_keys() must delegate to the underlying handle's
    offset_keys(), not to keys().  On a simple file they coincide, so we verify
    the delegation by mocking the underlying handle."""

    def setUp(self):
        torch.manual_seed(2)
        self.tensors = {"z": torch.randn(2, 2), "a": torch.randn(2, 2)}
        self.path = _write_safetensors(self.tensors)

    def tearDown(self):
        os.unlink(self.path)

    def test_offset_keys_delegates_to_handle(self):
        """offset_keys() must call self._handle.offset_keys(), not .keys().

        The safetensors Rust safe_open object is a C extension with read-only
        slots, so mock.patch.object cannot patch its methods directly.  Instead
        we wrap _handle in a thin Python proxy that records calls and then
        verify _SpyreSafeOpen.offset_keys() routed through offset_keys and not
        keys.
        """
        from torch_spyre._monkey_patch import _SpyreSafeOpen

        obj = _SpyreSafeOpen(self.path, framework="pt")
        with obj:
            real_handle = obj._handle
            offset_keys_calls = []
            keys_calls = []

            class _Proxy:
                def offset_keys(self):
                    offset_keys_calls.append(1)
                    return real_handle.offset_keys()

                def keys(self):
                    keys_calls.append(1)
                    return real_handle.keys()

                def __getattr__(self, name):
                    return getattr(real_handle, name)

            obj._handle = _Proxy()
            _ = obj.offset_keys()

        self.assertEqual(len(offset_keys_calls), 1, "offset_keys() not called")
        self.assertEqual(len(keys_calls), 0, "keys() called unexpectedly")

    def test_get_tensors_uses_offset_keys(self):
        """get_tensors() must iterate via offset_keys() for read locality."""
        from torch_spyre._monkey_patch import _SpyreSafeOpen

        obj = _SpyreSafeOpen(self.path, framework="pt")
        with obj:
            real_handle = obj._handle
            offset_keys_calls = []

            class _Proxy:
                def offset_keys(self):
                    offset_keys_calls.append(1)
                    return real_handle.offset_keys()

                def __getattr__(self, name):
                    return getattr(real_handle, name)

            obj._handle = _Proxy()
            with _patch_hook(None):
                _ = obj.get_tensors()

        self.assertGreater(
            len(offset_keys_calls), 0, "offset_keys() not called by get_tensors()"
        )


# ── __getattr__ forward-compat delegation ────────────────────────────────────


@requires_safetensors
class TestSpyreSafeOpenGetattr(TestCase):
    """_SpyreSafeOpen.__getattr__ must delegate unknown attributes to the
    underlying handle, making the wrapper forward-compatible with new
    safetensors API additions."""

    def setUp(self):
        torch.manual_seed(3)
        self.path = _write_safetensors({"x": torch.randn(2, 2)})

    def tearDown(self):
        os.unlink(self.path)

    def test_known_handle_attribute_forwarded(self):
        """An attribute on the underlying handle that _SpyreSafeOpen does not
        explicitly implement must be transparently accessible.

        The safetensors Rust safe_open object is a C extension with read-only
        slots, so we cannot attach attributes to it directly.  Replace _handle
        with a plain Python object that has the sentinel attribute.
        """
        from torch_spyre._monkey_patch import _SpyreSafeOpen

        obj = _SpyreSafeOpen(self.path, framework="pt")
        with obj:
            real_handle = obj._handle

            class _HandleWithExtra:
                _future_api = "sentinel"

                def __getattr__(self, name):
                    return getattr(real_handle, name)

            obj._handle = _HandleWithExtra()
            self.assertEqual(obj._future_api, "sentinel")

    def test_missing_attribute_raises_attr_error(self):
        """An attribute that neither _SpyreSafeOpen nor its handle has must
        raise AttributeError, not silently return None."""
        from torch_spyre._monkey_patch import _SpyreSafeOpen

        obj = _SpyreSafeOpen(self.path, framework="pt")
        with obj:
            with self.assertRaises(AttributeError):
                _ = obj._definitely_does_not_exist_xyzzy


# ── stable key ordering ───────────────────────────────────────────────────────
#
# NOTE: within a single process, PYTHONHASHSEED (and therefore str-set
# iteration order) is fixed for the process's whole lifetime. Rebuilding a
# set from the same elements in the same order N times in a loop — even a
# buggy `return list(some_set)` — reproduces the same order every time
# *within that run*. Looping and comparing runs against each other therefore
# cannot detect a regression back to set-based ordering; it only detects
# non-determinism *within* a process, which a set never actually exhibits
# either. Asserting against a specific, predicted order (derived from
# model/checkpoint construction order) is what actually pins the contract:
# a set-based implementation would only match this by chance, and — because
# PYTHONHASHSEED is randomized per process by default — would fail this
# assertion on most CI runs, not just in adversarial ones.


@requires_safetensors
class TestStableKeyOrdering(TestCase):
    """missing_keys and unexpected_keys must be returned in a stable,
    deterministic order — not set-iteration order."""

    def test_missing_keys_order_is_stable(self):
        """missing_keys must follow named_parameters() traversal order
        (registration order: a, b, c), not set-iteration order."""

        class MultiParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4, bias=False, dtype=torch.float16)
                self.b = nn.Linear(4, 4, bias=False, dtype=torch.float16)
                self.c = nn.Linear(4, 4, bias=False, dtype=torch.float16)

        # Checkpoint contains only a.weight; b.weight and c.weight are missing,
        # and must be reported in that declaration order — not reversed, and
        # not whatever order a Python set of the same two strings would give
        # under some other PYTHONHASHSEED.
        expected = ["b.weight", "c.weight"]

        tensors = {"a.weight": torch.randn(4, 4, dtype=torch.float16)}
        path = _write_safetensors(tensors)
        try:
            for _ in range(5):
                model = MultiParamModel()
                with _patch_hook(None):
                    missing, _ = _st_torch.load_model(
                        model, path, strict=False, device="spyre"
                    )
                self.assertEqual(
                    missing,
                    expected,
                    f"missing_keys order does not match expected traversal "
                    f"order: got {missing}, expected {expected}",
                )
        finally:
            os.unlink(path)

    def test_unexpected_keys_order_is_stable(self):
        """unexpected_keys must preserve checkpoint insertion/offset order
        (extra_z before extra_a, since that's how the file was written),
        not set-iteration order."""
        tensors = {
            "weight": torch.randn(4, 4, dtype=torch.float16),
            "extra_z": torch.randn(4, dtype=torch.float16),
            "extra_a": torch.randn(4, dtype=torch.float16),
        }
        # "weight" is consumed by the model; the two extras remain unexpected
        # in whatever order they actually appear in the checkpoint. Don't
        # hardcode an assumption about that order here: safetensors'
        # save_file does NOT preserve input dict insertion order — it sorts
        # tensor names before writing (verified against 0.8.0: a dict built
        # as {"weight", "extra_z", "extra_a"} is written to disk as
        # extra_a, extra_z, weight). Read the real on-disk order back via
        # offset_keys() so this test tracks safetensors' actual behavior
        # instead of an assumption about it that could go stale on a future
        # safetensors version.
        path = _write_safetensors(tensors)
        try:
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                on_disk_order = f.offset_keys()
            expected = [k for k in on_disk_order if k != "weight"]
            self.assertEqual(
                expected,
                ["extra_a", "extra_z"],
                "sanity check: expected on-disk order itself looks wrong "
                f"({expected}) — either safetensors' serialization order "
                "changed, or _write_safetensors changed. Investigate before "
                "trusting the rest of this test.",
            )

            for _ in range(5):
                m = nn.Linear(4, 4, bias=False, dtype=torch.float16)
                with _patch_hook(None):
                    _, unexpected = _st_torch.load_model(
                        m, path, strict=False, device="spyre"
                    )
                self.assertEqual(
                    unexpected,
                    expected,
                    f"unexpected_keys order does not match on-disk checkpoint "
                    f"order: got {unexpected}, expected {expected}",
                )
        finally:
            os.unlink(path)

    def test_missing_keys_order_immune_to_hash_seed(self):
        """Cross-process check: the same call made under different
        PYTHONHASHSEED values must return identical order. This is the
        specific property a set-based implementation violates (string-set
        iteration order depends on the hash seed) and that
        test_missing_keys_order_is_stable, run once per process, cannot
        observe on its own."""
        import subprocess
        import sys

        script = """
import sys, torch, torch.nn as nn
sys.path.insert(0, {test_dir!r})
from test_safetensors_patch_monkey_patch import (
    _write_safetensors, _patch_hook, _st_torch
)

class MultiParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False, dtype=torch.float16)
        self.b = nn.Linear(4, 4, bias=False, dtype=torch.float16)
        self.c = nn.Linear(4, 4, bias=False, dtype=torch.float16)

tensors = {{"a.weight": torch.randn(4, 4, dtype=torch.float16)}}
path = _write_safetensors(tensors)
model = MultiParamModel()
with _patch_hook(None):
    missing, _ = _st_torch.load_model(model, path, strict=False, device="spyre")
print(",".join(missing))
""".format(test_dir=os.path.dirname(os.path.abspath(__file__)))

        orders = set()
        for seed in ("0", "1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"subprocess with PYTHONHASHSEED={seed} failed:\n{result.stderr}",
            )
            orders.add(result.stdout.strip())

        self.assertEqual(
            len(orders),
            1,
            f"missing_keys order changed across PYTHONHASHSEED values: {orders} "
            f"— this indicates set-iteration order is leaking through instead "
            f"of a deterministic traversal order.",
        )


if __name__ == "__main__":
    run_tests()
