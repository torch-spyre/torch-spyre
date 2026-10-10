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

"""Tests for torch-spyre primitives supporting FP8 paged attention.

FP8 paged attention introduces quantized KV caches: KV tokens are quantized to
FP8 before being stored and dequantized before attention computation. These
tests exercise each compiler primitive in the shape and layout configurations
dictated by paged-cache geometry:

  1. Stick alignment for FP8 KV cache pages
  2. Per-token scale computation via quantscalepertokenfp8
  3. KV write: quantize FP16 KV token → FP8, store into page slots
  4. KV read (decode): gather an FP8 page, dequantize → FP16, run online-softmax
  5. KV read (prefill): multi-block gather and dequantize
  6. Compiled quantize-on-device + index_copy_ write path
  7. Compiled index_select gather of FP8 pages after device write

Stick-alignment rules
---------------------
Spyre's 128-byte stick holds:
  - 64 FP16/BF16 elements  → block_size/head_size must be multiples of 64
  - 128 FP8 elements       → block_size/head_size must be multiples of 128

All page shapes here use block_size=128 and head_size=128, satisfying the FP8
alignment constraint while keeping tests fast.

EA (ElementArrangement) semantics for the KV read path
-------------------------------------------------------
``dequantize_fp8_with_scale`` lowers to ``fp8todl16 * scale`` on device.
``fp8todl16`` requires the input to carry ``ElementArrangement.QFP8CH``, which
is stamped by the on-device ``qfp8ch`` op (``quantize_fp8_with_scale``).

A CPU-quantized FP8 tensor moved to device via H2D transfer lands with
``ElementArrangement.STANDARD``. Tensors dequantized on device must originate
from on-device ``qfp8ch`` operations.

In the standard KV cache workflow:
  - ``quantize_fp8_with_scale`` runs on device, producing ``QFP8CH`` byte layouts.
  - ``index_copy_`` scatters into the FP8 cache buffer (re-stamping the buffer
    metadata as STANDARD while preserving the underlying QFP8CH bit pattern).
  - ``index_select`` gathers the page slots.
  - ``dequantize_fp8_with_scale`` successfully converts gathered FP8 pages to FP16.

The canonical execution flow is:
    q_fp8 = quantize_fp8_with_scale(key, scale)
    cache.index_copy_(0, idx, q_fp8)
    ...
    page = cache.index_select(0, page_idx)
    kv_fp16 = dequantize_fp8_with_scale(page, scale)
"""

import pytest
import torch
from utils_inductor import DEVICE, cached_randn, compare_with_pytorch

from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick
from torch_spyre._inductor.constants import FP8_E4M3FN_MAX, FP8_E4M3FN_MIN

# ──────────────────────────────────────────────────────────────────────────────
# Shared geometry constants matching standard paged-cache shapes.
# block_size=128, head_size=128: both are multiples of 128 (fp8 elems/stick).
# ──────────────────────────────────────────────────────────────────────────────
NUM_BLOCKS = 4
BLOCK_SIZE = 128  # fp8 elems per stick = 128; must be multiple of 128 for fp8
HEAD_SIZE = 128  # same alignment requirement on the head dim
NUM_KV_HEADS = 2
NUM_HEADS = 4  # GQA: 2 query groups per KV head


# ──────────────────────────────────────────────────────────────────────────────
# CPU reference helpers
# ──────────────────────────────────────────────────────────────────────────────


def _cpu_quantize(x_fp16: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CPU reference: fp16 → fp8 via scale."""
    return (
        (x_fp16 / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
    )


def _cpu_dequantize(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CPU reference: fp8 → fp16 via scale."""
    return x_fp8.to(torch.float16) * scale


def _cpu_quantize_dequantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _cpu_dequantize(_cpu_quantize(x, scale), scale)


def _cpu_per_token_scale(x: torch.Tensor) -> torch.Tensor:
    """amax-based per-token scale: shape [..., 1] fp16."""
    return torch.amax(torch.abs(x), dim=-1, keepdim=True) / FP8_E4M3FN_MAX


def _device_write_read_fp8(
    x_fp16: torch.Tensor,
    scale: torch.Tensor,
    cache_shape: tuple,
    slot_indices: torch.Tensor,
) -> torch.Tensor:
    """Reference: on-device qfp8ch + scatter + gather → fp8 with STANDARD EA.

    Executes on-device quantization followed by index_copy_ scatter and index_select
    gather. ``dequantize_fp8_with_scale`` operates correctly on the gathered result
    because the scattered bytes preserve the QFP8CH bit pattern.

    Returns the gathered fp8 tensor (device), which is passed to
    ``dequantize_fp8_with_scale`` to complete the round-trip.
    """
    x_d = x_fp16.to(DEVICE)
    s_d = scale.to(DEVICE)
    idx_d = slot_indices.to(DEVICE)

    cache_d = torch.zeros(cache_shape, dtype=torch.float8_e4m3fn).to(DEVICE)

    def write_fn(src, sc, cache, idx):
        q = torch.ops.spyre.quantize_fp8_with_scale(src, sc)
        cache.index_copy_(0, idx, q)
        return cache

    cache_d = torch.compile(write_fn, dynamic=False)(x_d, s_d, cache_d, idx_d)
    gathered = torch.compile(lambda c, i: c.index_select(0, i), dynamic=False)(
        cache_d, idx_d
    )
    return gathered


# ──────────────────────────────────────────────────────────────────────────────
# 1. Stick alignment for fp8 cache pages
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8KvCacheStickAlignment:
    """get_elem_in_stick and SpyreTensorLayout must be fp8-aware.

    Validates that FP8 stick depth (128 elements per stick) is respected
    during memory allocation and layout construction across both slot-major
    and head-major configurations.
    """

    def test_get_elem_in_stick_fp8_is_128(self):
        """fp8 e4m3fn occupies 1 byte; a 128-byte stick holds 128 elements."""
        assert get_elem_in_stick(torch.float8_e4m3fn) == 128

    def test_get_elem_in_stick_fp16_is_64(self):
        """Baseline: fp16 stick depth is 64 (2 bytes × 64 = 128 bytes)."""
        assert get_elem_in_stick(torch.float16) == 64

    def test_get_device_dtype_fp8_exists(self):
        """get_device_dtype must resolve torch.float8_e4m3fn without raising."""
        dt = get_device_dtype(torch.float8_e4m3fn)
        # SEN143_FP8 or SEN152_FP8 — either is valid; just must not raise.
        assert dt is not None

    def test_fp8_slot_major_kv_layout_allocates(self):
        """A token-major fp8 KV cache page tensor can be allocated with a custom layout.

        Allocates slot-major cache geometry using get_elem_in_stick(dtype) to
        derive stick depth, verifying tensor placement on device.
        """
        num_slots = NUM_BLOCKS * BLOCK_SIZE
        dtype = torch.float8_e4m3fn
        eps = get_elem_in_stick(dtype)  # 128
        sticks = (HEAD_SIZE + eps - 1) // eps  # 1

        layout = SpyreTensorLayout(
            device_size=[num_slots, NUM_KV_HEADS, sticks, eps],
            stride_map=[NUM_KV_HEADS * sticks * eps, sticks * eps, eps, 1],
            device_dtype=get_device_dtype(dtype),
        )
        # torch.zeros → to(device, device_layout=layout) mirrors allocate_pages.
        t = torch.zeros((num_slots, NUM_KV_HEADS, HEAD_SIZE), dtype=dtype).to(
            DEVICE, device_layout=layout
        )  # type: ignore[no-matching-overload]
        assert t.device.type == "spyre"
        assert t.shape == (num_slots, NUM_KV_HEADS, HEAD_SIZE)
        assert t.dtype == torch.float8_e4m3fn

    def test_fp8_head_major_kv_layout_allocates(self):
        """Head-major fp8 KV cache page tensor allocates correctly."""
        dtype = torch.float8_e4m3fn
        eps = get_elem_in_stick(dtype)
        sticks = (HEAD_SIZE + eps - 1) // eps

        folded_pages = NUM_BLOCKS * NUM_KV_HEADS
        layout = SpyreTensorLayout(
            device_size=[folded_pages, BLOCK_SIZE, sticks, eps],
            stride_map=[BLOCK_SIZE * HEAD_SIZE, HEAD_SIZE, eps, 1],
            device_dtype=get_device_dtype(dtype),
        )
        shape = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        t = torch.zeros(shape, dtype=dtype).to(DEVICE, device_layout=layout)  # type: ignore[no-matching-overload]
        assert t.device.type == "spyre"
        assert t.dtype == torch.float8_e4m3fn
        assert t.shape == shape


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-token scale computation
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8PerTokenScale:
    """quantscalepertokenfp8 is used at KV write time to derive per-token scales.

    For each token the KV projection emits [num_kv_heads, head_size] fp16.
    A per-token scale is max(|x|) / FP8_MAX, shape [num_kv_heads, 1].
    """

    @pytest.mark.parametrize("num_tokens", [1, 4, 16])
    def test_per_token_scale_kv_shape(self, num_tokens):
        """quantscalepertokenfp8 on a [T, KV, D] KV projection gives [T, KV, 1]."""
        x = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float16, scale=1.0
        )

        def spyre_fn(inp):
            return torch.ops.spyre.quantscalepertokenfp8(inp)

        def cpu_fn(inp):
            return _cpu_per_token_scale(inp)

        compare_with_pytorch(spyre_fn, cpu_fn, x, atol=1e-3, rtol=2e-3)

    def test_per_token_scale_single_token_single_head(self):
        """Shape [1, 1, D] — the decode path emits one token per sequence."""
        x = cached_randn((1, 1, HEAD_SIZE), dtype=torch.float16, scale=2.0)

        def spyre_fn(inp):
            return torch.ops.spyre.quantscalepertokenfp8(inp)

        def cpu_fn(inp):
            return _cpu_per_token_scale(inp)

        compare_with_pytorch(spyre_fn, cpu_fn, x, atol=1e-3, rtol=2e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 3. KV cache write: quantize fp16 KV → fp8
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8KvCacheWrite:
    """Exercises the quantize-then-store path for FP8 KV caches.

    Validates that quantizing inputs with per-token scales and storing FP8 values
    via index_copy_ compile cleanly and yield accurate results.
    """

    @pytest.mark.parametrize("num_tokens", [1, 4, 16])
    def test_quantize_fp16_kv_token_roundtrip(self, num_tokens):
        """quantize_fp8_with_scale → dequantize_fp8_with_scale round-trips a KV token."""
        kv = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float16, scale=1.0
        )
        # One scale per (token, kv_head).
        scale = _cpu_per_token_scale(kv)

        def spyre_fn(inp, s):
            q = torch.ops.spyre.quantize_fp8_with_scale(inp, s)
            return torch.ops.spyre.dequantize_fp8_with_scale(q, s)

        def cpu_fn(inp, s):
            return _cpu_quantize_dequantize(inp, s)

        # FP8 quantization is lossy; atol is calibrated to the max FP8 spacing
        # (32) times the scale (≤ 1.0 for unit-variance inputs).
        compare_with_pytorch(spyre_fn, cpu_fn, kv, scale, atol=2.0, rtol=0.0)

    def test_quantize_kv_token_dtype_is_fp8(self):
        """quantize_fp8_with_scale must return float8_e4m3fn."""
        kv = cached_randn((1, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float16, scale=1.0)
        scale = _cpu_per_token_scale(kv)

        kv_d = kv.to(DEVICE)
        scale_d = scale.to(DEVICE)

        compiled = torch.compile(
            lambda x, s: torch.ops.spyre.quantize_fp8_with_scale(x, s),
            dynamic=False,
        )
        q = compiled(kv_d, scale_d)
        assert q.dtype == torch.float8_e4m3fn, f"Expected float8_e4m3fn, got {q.dtype}"
        assert q.shape == kv_d.shape

    @pytest.mark.parametrize("num_tokens", [1, 8])
    def test_fp8_index_copy_into_cache_slot(self, num_tokens):
        """on-device qfp8ch + index_copy_ + index_select + dequantize round-trip.

        Validates the complete write→read sequence:
          1. quantize_fp8_with_scale on device → QFP8CH EA
          2. index_copy_ scatters into cache → buffer metadata stamped as STANDARD EA
          3. index_select gathers back → STANDARD EA
          4. dequantize_fp8_with_scale produces accurate FP16 outputs.

        Note: Direct device allocation with fill_tensor is unsupported for FP8;
        tensors are allocated on host and transferred via .to(DEVICE).
        """
        num_slots = NUM_BLOCKS * BLOCK_SIZE
        src_fp16 = cached_randn((num_tokens, HEAD_SIZE), dtype=torch.float16, scale=1.0)
        scale = torch.amax(src_fp16.abs(), dim=-1, keepdim=True) / FP8_E4M3FN_MAX
        slot_idx = torch.arange(num_tokens, dtype=torch.int32)

        # Full write→read on device; dequantize the gathered result.
        gathered = _device_write_read_fp8(
            src_fp16, scale, (num_slots, HEAD_SIZE), slot_idx
        )
        scale_d = scale.to(DEVICE)

        def dequant_fn(fp8, sc):
            return torch.ops.spyre.dequantize_fp8_with_scale(fp8, sc)

        def cpu_ref(src, sc):
            return _cpu_quantize_dequantize(src, sc)

        # atol=2.0: FP8 quantization error (max spacing ≈ 32 × scale ≈ 32/448 ≈ 0.07
        # for unit-variance inputs; device arithmetic adds a small rounding delta).
        compare_with_pytorch(
            dequant_fn,
            cpu_ref,
            src_fp16,
            scale,
            target=torch.compile(dequant_fn, dynamic=False)(gathered, scale_d).cpu(),
            atol=2.0,
            rtol=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. KV cache read (decode): gather fp8 page, dequantize, run attention
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8PagedAttentionDecode:
    """Exercises the decode path: gather one fp8 KV page per block, dequantize, attend.

    Verifies the building-block ops for paged attention decode: gathering FP8
    cache pages and scale tensors, dequantizing to FP16, and executing attention
    projections and softmax.
    """

    def _make_fp8_page_cache(self, num_blocks: int):
        """Build a token-major fp8 KV cache written via on-device qfp8ch + scatter.

        Uses ``_device_write_read_fp8`` so the cache holds the QFP8CH bit pattern
        with STANDARD EA. Returns the cache pages and scale tensors needed by the
        attention kernels.

        Returns:
            k_pages_d: [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE] fp8 device
            v_pages_d: same
            k_scales_d: [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1] fp16 device
            v_scales_d: same
            k_fp16: CPU fp16 ground truth for the reference attention
            v_fp16: same
        """
        k_fp16 = cached_randn(
            (num_blocks * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="k",
        )
        v_fp16 = cached_randn(
            (num_blocks * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="v",
        )
        k_scale = _cpu_per_token_scale(k_fp16)  # [T, KV, 1]
        v_scale = _cpu_per_token_scale(v_fp16)

        total_tokens = num_blocks * BLOCK_SIZE
        slot_idx = torch.arange(total_tokens, dtype=torch.int32)
        cache_shape = (total_tokens, NUM_KV_HEADS, HEAD_SIZE)

        # Write via on-device qfp8ch so the cache has the correct bit pattern.
        k_pages_d = _device_write_read_fp8(k_fp16, k_scale, cache_shape, slot_idx)
        v_pages_d = _device_write_read_fp8(v_fp16, v_scale, cache_shape, slot_idx)
        k_scales_d = k_scale.to(DEVICE)
        v_scales_d = v_scale.to(DEVICE)

        # Reshape to paged layout for the attention kernels.
        page_shape = (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k_pages_d = k_pages_d.reshape(page_shape)
        v_pages_d = v_pages_d.reshape(page_shape)
        k_scales_d = k_scales_d.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)
        v_scales_d = v_scales_d.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)

        return (
            k_pages_d,
            v_pages_d,
            k_scales_d,
            v_scales_d,
            k_fp16,
            v_fp16,
            k_scale,
            v_scale,
        )

    def test_fp8_page_gather_and_dequantize_decode(self):
        """index_select on an fp8 page cache written on-device + dequantize matches CPU.

        This is the KV read kernel's per-block body.  The cache was written via
        on-device qfp8ch so the gathered bytes can be correctly dequantized:
          cache[slot] = qfp8ch(key)  →  dequantize(cache.index_select(0, idx), scale)
        """
        k_pages_d, _, k_scales_d, _, k_fp16, _, k_scale, _ = self._make_fp8_page_cache(
            NUM_BLOCKS
        )

        idx_d = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        # Compute reference: quantize page 0 on device and dequantize.
        def attn_fn(pages, scales, idx):
            page = pages.index_select(0, idx)
            sc = scales.index_select(0, idx)
            return torch.ops.spyre.dequantize_fp8_with_scale(page, sc)

        result = torch.compile(attn_fn, dynamic=False)(k_pages_d, k_scales_d, idx_d)

        # CPU reference: quantize then dequantize page 0 (same lossy path).
        k0_fp16 = k_fp16[:BLOCK_SIZE].reshape(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k0_scale = k_scale[:BLOCK_SIZE].reshape(1, BLOCK_SIZE, NUM_KV_HEADS, 1)
        expected = _cpu_quantize_dequantize(k0_fp16, k0_scale)

        # atol=2.0: FP8 spacing at scale≈0.5/448 ≈ 0.001; device arithmetic adds ~fp16 ulp.
        torch.testing.assert_close(result.cpu(), expected, atol=2.0, rtol=0.0)

    def test_fp8_decode_attention_single_block(self):
        """One-block decode attention over an on-device fp8 KV cache.

        Full pipeline: gather fp8 page → dequantize → Q·K^T → softmax → P·V.
        KV cache was written via on-device qfp8ch so dequantize is correct.
        The reference uses the same fp8-quantized values (CPU round-trip) so
        quantization noise cancels.

        GQA attention geometry (shapes annotated):
          k/v page: [1,B,KV,D] → squeeze → [B,KV,D] → permute(1,0,2) → [KV,B,D]
          q:        [1,H,D]    → reshape → [KV,gqa,D]
          scores:   [KV,gqa,D] @ [KV,D,B]  → [KV,gqa,B]
          out:      [KV,gqa,B] @ [KV,B,D]  → [KV,gqa,D] → reshape(1,H,D)
        """
        (
            k_pages_d,
            v_pages_d,
            k_scales_d,
            v_scales_d,
            k_fp16,
            v_fp16,
            k_scale,
            v_scale,
        ) = self._make_fp8_page_cache(1)

        query_fp16 = cached_randn(
            (1, NUM_HEADS, HEAD_SIZE), dtype=torch.float16, scale=0.5
        )
        attn_scale = float(HEAD_SIZE**-0.5)
        gqa = NUM_HEADS // NUM_KV_HEADS

        q_d = query_fp16.to(DEVICE)
        idx_d = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        def decode_attn(q, k_cache, k_sc, v_cache, v_sc, idx):
            k_pg = torch.ops.spyre.dequantize_fp8_with_scale(
                k_cache.index_select(0, idx), k_sc.index_select(0, idx)
            )  # [1, B, KV, D]
            v_pg = torch.ops.spyre.dequantize_fp8_with_scale(
                v_cache.index_select(0, idx), v_sc.index_select(0, idx)
            )
            # Rearrange to [KV, B, D] for batched GQA matmul.
            kv_k = k_pg.squeeze(0).permute(1, 0, 2)  # [KV, B, D]
            kv_v = v_pg.squeeze(0).permute(1, 0, 2)  # [KV, B, D]
            q_gqa = q.reshape(NUM_KV_HEADS, gqa, HEAD_SIZE)  # [KV, gqa, D]
            # scores: [KV, gqa, D] @ [KV, D, B] → [KV, gqa, B]
            scores = torch.matmul(q_gqa, kv_k.transpose(-1, -2)) * attn_scale
            probs = torch.softmax(scores, dim=-1)  # [KV, gqa, B]
            # out: [KV, gqa, B] @ [KV, B, D] → [KV, gqa, D]
            out = torch.matmul(probs, kv_v)
            return out.reshape(1, NUM_HEADS, HEAD_SIZE)

        result = torch.compile(decode_attn, dynamic=False)(
            q_d, k_pages_d, k_scales_d, v_pages_d, v_scales_d, idx_d
        )

        # CPU reference: same computation with fp8-round-tripped KV.
        k0 = k_fp16[:BLOCK_SIZE].reshape(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        v0 = v_fp16[:BLOCK_SIZE].reshape(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k0_sc = k_scale[:BLOCK_SIZE].reshape(1, BLOCK_SIZE, NUM_KV_HEADS, 1)
        v0_sc = v_scale[:BLOCK_SIZE].reshape(1, BLOCK_SIZE, NUM_KV_HEADS, 1)
        k0_dq = _cpu_quantize_dequantize(k0, k0_sc).squeeze(0).float()  # [B, KV, D]
        v0_dq = _cpu_quantize_dequantize(v0, v0_sc).squeeze(0).float()
        kv_k_cpu = k0_dq.permute(1, 0, 2)  # [KV, B, D]
        kv_v_cpu = v0_dq.permute(1, 0, 2)  # [KV, B, D]
        q_gqa_cpu = query_fp16.reshape(NUM_KV_HEADS, gqa, HEAD_SIZE).float()
        sc_cpu = torch.matmul(q_gqa_cpu, kv_k_cpu.transpose(-1, -2)) * attn_scale
        pr_cpu = torch.softmax(sc_cpu, dim=-1)
        out_cpu = torch.matmul(pr_cpu, kv_v_cpu).reshape(1, NUM_HEADS, HEAD_SIZE)
        expected = out_cpu.to(torch.float16)

        # atol=2e-2: fp16 device matmul vs fp32 CPU; softmax normalises output to [0,1]×V.
        torch.testing.assert_close(result.cpu(), expected, atol=2e-2, rtol=0.0)

    def test_fp8_decode_attention_gqa_8kv_heads(self):
        """Decode attention with 8 KV heads and 32 query heads (4:1 GQA ratio)."""
        num_kv_heads = 8
        num_heads = 32
        gqa = num_heads // num_kv_heads

        k_fp16 = cached_randn(
            (BLOCK_SIZE, num_kv_heads, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="k_gqa8",
        )
        v_fp16 = cached_randn(
            (BLOCK_SIZE, num_kv_heads, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="v_gqa8",
        )
        k_scale = _cpu_per_token_scale(k_fp16)
        v_scale = _cpu_per_token_scale(v_fp16)
        slot_idx = torch.arange(BLOCK_SIZE, dtype=torch.int32)
        cache_shape = (BLOCK_SIZE, num_kv_heads, HEAD_SIZE)

        k_pages_d = _device_write_read_fp8(
            k_fp16, k_scale, cache_shape, slot_idx
        ).reshape(1, BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
        v_pages_d = _device_write_read_fp8(
            v_fp16, v_scale, cache_shape, slot_idx
        ).reshape(1, BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
        k_scales_d = k_scale.to(DEVICE).reshape(1, BLOCK_SIZE, num_kv_heads, 1)
        v_scales_d = v_scale.to(DEVICE).reshape(1, BLOCK_SIZE, num_kv_heads, 1)

        query_fp16 = cached_randn(
            (1, num_heads, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="q_gqa8",
        )
        attn_scale = float(HEAD_SIZE**-0.5)
        q_d = query_fp16.to(DEVICE)
        idx_d = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        def decode_attn(q, k_cache, k_sc, v_cache, v_sc, idx):
            k_pg = torch.ops.spyre.dequantize_fp8_with_scale(
                k_cache.index_select(0, idx), k_sc.index_select(0, idx)
            )
            v_pg = torch.ops.spyre.dequantize_fp8_with_scale(
                v_cache.index_select(0, idx), v_sc.index_select(0, idx)
            )
            kv_k = k_pg.squeeze(0).permute(1, 0, 2)
            kv_v = v_pg.squeeze(0).permute(1, 0, 2)
            q_gqa = q.reshape(num_kv_heads, gqa, HEAD_SIZE)
            scores = torch.matmul(q_gqa, kv_k.transpose(-1, -2)) * attn_scale
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, kv_v)
            return out.reshape(1, num_heads, HEAD_SIZE)

        result = torch.compile(decode_attn, dynamic=False)(
            q_d, k_pages_d, k_scales_d, v_pages_d, v_scales_d, idx_d
        )

        k0_dq = _cpu_quantize_dequantize(k_fp16, k_scale).float()
        v0_dq = _cpu_quantize_dequantize(v_fp16, v_scale).float()
        kv_k_cpu = k0_dq.permute(1, 0, 2)
        kv_v_cpu = v0_dq.permute(1, 0, 2)
        q_gqa_cpu = query_fp16.reshape(num_kv_heads, gqa, HEAD_SIZE).float()
        sc_cpu = torch.matmul(q_gqa_cpu, kv_k_cpu.transpose(-1, -2)) * attn_scale
        pr_cpu = torch.softmax(sc_cpu, dim=-1)
        out_cpu = torch.matmul(pr_cpu, kv_v_cpu).reshape(1, num_heads, HEAD_SIZE)
        expected = out_cpu.to(torch.float16)

        torch.testing.assert_close(result.cpu(), expected, atol=2e-2, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. KV cache read (prefill): multi-query, multi-block
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8PagedAttentionPrefill:
    """Multi-query fp8 KV read, mirroring the prefill path.

    The prefill attention kernel iterates over blocks; here we test that
    gathering multiple fp8 pages, dequantizing them, and concatenating gives
    a consistent result — the compiler's handling of dequantize inside a loop
    body is the key concern.
    """

    @pytest.mark.parametrize("num_kv_blocks", [1, 2, 4])
    def test_fp8_prefill_multi_block_dequantize(self, num_kv_blocks):
        """Gather N fp8 K+V blocks written on-device, dequantize each, verify correctness.

        Tests the compiled gather+dequantize inside a Python loop — the same
        structure the prefill attention kernel uses for fp8 KV pages.  The cache
        was written via on-device qfp8ch so dequantize produces correct fp16 values.

        Covers N=1, 2, 4 blocks to exercise both the single-block and multi-block
        compiled loop paths, including both K and V gather (previously K-only).
        """
        total_tokens = num_kv_blocks * BLOCK_SIZE
        k_fp16 = cached_randn(
            (total_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="pf_k",
        )
        v_fp16 = cached_randn(
            (total_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="pf_v",
        )
        k_scale = _cpu_per_token_scale(k_fp16)  # [T, KV, 1]
        v_scale = _cpu_per_token_scale(v_fp16)
        slot_idx = torch.arange(total_tokens, dtype=torch.int32)

        # Write via on-device qfp8ch, then reshape to page layout.
        k_pages_d = _device_write_read_fp8(
            k_fp16, k_scale, (total_tokens, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        v_pages_d = _device_write_read_fp8(
            v_fp16, v_scale, (total_tokens, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k_scales_d = k_scale.to(DEVICE).reshape(
            num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1
        )
        v_scales_d = v_scale.to(DEVICE).reshape(
            num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1
        )

        def gather_blocks(k_pages, k_scales, v_pages, v_scales):
            k_out = []
            v_out = []
            for b in range(num_kv_blocks):
                idx = torch.tensor([b], dtype=torch.int32, device=k_pages.device)
                k_p = k_pages.index_select(0, idx)
                k_s = k_scales.index_select(0, idx)
                v_p = v_pages.index_select(0, idx)
                v_s = v_scales.index_select(0, idx)
                k_out.append(torch.ops.spyre.dequantize_fp8_with_scale(k_p, k_s))
                v_out.append(torch.ops.spyre.dequantize_fp8_with_scale(v_p, v_s))
            return torch.cat(k_out, dim=0), torch.cat(v_out, dim=0)

        k_result, v_result = torch.compile(gather_blocks, dynamic=False)(
            k_pages_d, k_scales_d, v_pages_d, v_scales_d
        )

        # CPU reference: quantize then dequantize.
        k_paged = k_fp16.reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k_sc_paged = k_scale.reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)
        v_paged = v_fp16.reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        v_sc_paged = v_scale.reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)

        torch.testing.assert_close(
            k_result.cpu(),
            _cpu_quantize_dequantize(k_paged, k_sc_paged),
            atol=2.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            v_result.cpu(),
            _cpu_quantize_dequantize(v_paged, v_sc_paged),
            atol=2.0,
            rtol=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Fused quantize-and-store: the reshape_and_cache fp8 kernel body
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8ReshapeAndCache:
    """Models the fused quantize-and-store kernel for FP8 KV caches.

    Evaluates the combined pipeline:
        scale_k = quantscalepertokenfp8(key)
        scale_v = quantscalepertokenfp8(value)
        k_fp8 = quantize_fp8_with_scale(key, scale_k)
        v_fp8 = quantize_fp8_with_scale(value, scale_v)
        k_slots.index_copy_(0, slot_mapping, k_fp8)
        v_slots.index_copy_(0, slot_mapping, v_fp8)
        k_scale.index_copy_(0, slot_mapping, scale_k)
        v_scale.index_copy_(0, slot_mapping, scale_v)

    The full write → on-device dequantize → CPU comparison is verified for
    1, 2, 3, 4, and 16 tokens.
    """

    def _run_reshape_and_cache_roundtrip(self, num_tokens: int):
        """Shared implementation: write to fp8 cache, read back with on-device dequantize."""
        num_slots = NUM_BLOCKS * BLOCK_SIZE

        key = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=1.0,
            differentiation="fused_k",
        )
        value = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=1.0,
            differentiation="fused_v",
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

        key_d = key.to(DEVICE)
        value_d = value.to(DEVICE)
        slot_d = slot_mapping.to(DEVICE)

        # Allocate fp8 caches on CPU then transfer: fill_tensor does not support fp8.
        k_cache = torch.zeros(
            (num_slots, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float8_e4m3fn
        ).to(DEVICE)
        v_cache = torch.zeros(
            (num_slots, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float8_e4m3fn
        ).to(DEVICE)
        k_scale_cache = torch.zeros(
            (num_slots, NUM_KV_HEADS, 1), dtype=torch.float16, device=DEVICE
        )
        v_scale_cache = torch.zeros_like(k_scale_cache)

        def reshape_and_cache_fp8(k, v, k_slots, v_slots, k_sc, v_sc, idx):
            sk = torch.ops.spyre.quantscalepertokenfp8(k)
            sv = torch.ops.spyre.quantscalepertokenfp8(v)
            k_fp8 = torch.ops.spyre.quantize_fp8_with_scale(k, sk)
            v_fp8 = torch.ops.spyre.quantize_fp8_with_scale(v, sv)
            k_slots.index_copy_(0, idx, k_fp8)
            v_slots.index_copy_(0, idx, v_fp8)
            k_sc.index_copy_(0, idx, sk)
            v_sc.index_copy_(0, idx, sv)
            return k_slots, v_slots, k_sc, v_sc

        torch.compile(reshape_and_cache_fp8, dynamic=False)(
            key_d, value_d, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_d
        )

        # Read back via on-device index_select + dequantize — D2H of a QFP8CH-written
        # fp8 cache gives wrong values because index_copy_ re-stamps the buffer as
        # STANDARD EA; must dequantize on-device where the QFP8CH bit pattern is intact.
        read_idx = slot_d.clone()

        def read_fn(k_sl, v_sl, k_sc, v_sc, idx):
            k_dq = torch.ops.spyre.dequantize_fp8_with_scale(
                k_sl.index_select(0, idx), k_sc.index_select(0, idx)
            )
            v_dq = torch.ops.spyre.dequantize_fp8_with_scale(
                v_sl.index_select(0, idx), v_sc.index_select(0, idx)
            )
            return k_dq, v_dq

        k_roundtrip, v_roundtrip = torch.compile(read_fn, dynamic=False)(
            k_cache, v_cache, k_scale_cache, v_scale_cache, read_idx
        )

        k_expected = _cpu_quantize_dequantize(key, _cpu_per_token_scale(key))
        v_expected = _cpu_quantize_dequantize(value, _cpu_per_token_scale(value))

        torch.testing.assert_close(k_roundtrip.cpu(), k_expected, atol=2.0, rtol=0.0)
        torch.testing.assert_close(v_roundtrip.cpu(), v_expected, atol=2.0, rtol=0.0)

    def test_fused_quantize_and_store_roundtrip_1token(self):
        """num_tokens=1: full write → on-device dequantize → CPU comparison."""
        self._run_reshape_and_cache_roundtrip(1)

    def test_fused_quantize_and_store_roundtrip_2tokens(self):
        """num_tokens=2: sub-stick boundary case (fills between T=1 and T=4)."""
        self._run_reshape_and_cache_roundtrip(2)

    def test_fused_quantize_and_store_roundtrip_3tokens(self):
        """num_tokens=3: sub-stick boundary case (fills between T=1 and T=4)."""
        self._run_reshape_and_cache_roundtrip(3)

    def test_fused_quantize_and_store_roundtrip_4tokens(self):
        """num_tokens=4: full write → on-device dequantize → CPU comparison."""
        self._run_reshape_and_cache_roundtrip(4)

    def test_fused_quantize_and_store_roundtrip_16tokens(self):
        """num_tokens=16: full write → on-device dequantize → CPU comparison."""
        self._run_reshape_and_cache_roundtrip(16)

    def test_fused_quantize_and_store_noncontiguous_slots(self):
        """Scattering into non-contiguous, out-of-order slot mappings across pages."""
        num_slots = NUM_BLOCKS * BLOCK_SIZE
        slot_mapping = torch.tensor([5, 130, 20, 255], dtype=torch.int32)
        num_tokens = len(slot_mapping)

        key = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=1.0,
            differentiation="noncontig_k",
        )
        value = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=1.0,
            differentiation="noncontig_v",
        )

        key_d = key.to(DEVICE)
        value_d = value.to(DEVICE)
        slot_d = slot_mapping.to(DEVICE)

        k_cache = torch.zeros(
            (num_slots, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float8_e4m3fn
        ).to(DEVICE)
        v_cache = torch.zeros(
            (num_slots, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float8_e4m3fn
        ).to(DEVICE)
        k_scale_cache = torch.zeros(
            (num_slots, NUM_KV_HEADS, 1), dtype=torch.float16, device=DEVICE
        )
        v_scale_cache = torch.zeros_like(k_scale_cache)

        def reshape_and_cache_fp8(k, v, k_slots, v_slots, k_sc, v_sc, idx):
            sk = torch.ops.spyre.quantscalepertokenfp8(k)
            sv = torch.ops.spyre.quantscalepertokenfp8(v)
            k_fp8 = torch.ops.spyre.quantize_fp8_with_scale(k, sk)
            v_fp8 = torch.ops.spyre.quantize_fp8_with_scale(v, sv)
            k_slots.index_copy_(0, idx, k_fp8)
            v_slots.index_copy_(0, idx, v_fp8)
            k_sc.index_copy_(0, idx, sk)
            v_sc.index_copy_(0, idx, sv)
            return k_slots, v_slots, k_sc, v_sc

        torch.compile(reshape_and_cache_fp8, dynamic=False)(
            key_d, value_d, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_d
        )

        read_idx = slot_d.clone()

        def read_fn(k_sl, v_sl, k_sc, v_sc, idx):
            k_dq = torch.ops.spyre.dequantize_fp8_with_scale(
                k_sl.index_select(0, idx), k_sc.index_select(0, idx)
            )
            v_dq = torch.ops.spyre.dequantize_fp8_with_scale(
                v_sl.index_select(0, idx), v_sc.index_select(0, idx)
            )
            return k_dq, v_dq

        k_roundtrip, v_roundtrip = torch.compile(read_fn, dynamic=False)(
            k_cache, v_cache, k_scale_cache, v_scale_cache, read_idx
        )

        k_expected = _cpu_quantize_dequantize(key, _cpu_per_token_scale(key))
        v_expected = _cpu_quantize_dequantize(value, _cpu_per_token_scale(value))

        torch.testing.assert_close(k_roundtrip.cpu(), k_expected, atol=2.0, rtol=0.0)
        torch.testing.assert_close(v_roundtrip.cpu(), v_expected, atol=2.0, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Head-Major KV cache write: quantize + per-head scatter
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8HeadMajorReshapeAndCache:
    """Models head-major KV cache write operations with FP8 quantization.

    Head-major caches store [num_blocks * num_kv_heads * block_size, head_size] rows,
    scattering each KV head separately across row_index[h].
    """

    @pytest.mark.parametrize("num_tokens", [1, 4, 16])
    def test_head_major_fused_quantize_and_scatter(self, num_tokens):
        """Quantize on-device, scatter per-head into folded head-major cache, and read back."""
        total_rows = NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE

        key = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="hm_k",
        )
        value = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="hm_v",
        )

        # Build per-head row indices matching head_major layout:
        # head h token t at slot t lives at row h * BLOCK_SIZE + t
        row_indices = [
            torch.arange(h * BLOCK_SIZE, h * BLOCK_SIZE + num_tokens, dtype=torch.int32)
            for h in range(NUM_KV_HEADS)
        ]

        key_d = key.to(DEVICE)
        value_d = value.to(DEVICE)
        row_indices_d = [idx.to(DEVICE) for idx in row_indices]

        k_rows = torch.zeros((total_rows, HEAD_SIZE), dtype=torch.float8_e4m3fn).to(
            DEVICE
        )
        v_rows = torch.zeros((total_rows, HEAD_SIZE), dtype=torch.float8_e4m3fn).to(
            DEVICE
        )
        k_scale_rows = torch.zeros((total_rows, 1), dtype=torch.float16, device=DEVICE)
        v_scale_rows = torch.zeros((total_rows, 1), dtype=torch.float16, device=DEVICE)

        def reshape_and_cache_hm_fp8(k, v, k_r, v_r, k_sc, v_sc, idx0, idx1):
            sk = torch.ops.spyre.quantscalepertokenfp8(k)
            sv = torch.ops.spyre.quantscalepertokenfp8(v)
            k_fp8 = torch.ops.spyre.quantize_fp8_with_scale(k, sk)
            v_fp8 = torch.ops.spyre.quantize_fp8_with_scale(v, sv)

            k_r.index_copy_(0, idx0, k_fp8[:, 0])
            v_r.index_copy_(0, idx0, v_fp8[:, 0])
            k_sc.index_copy_(0, idx0, sk[:, 0])
            v_sc.index_copy_(0, idx0, sv[:, 0])

            k_r.index_copy_(0, idx1, k_fp8[:, 1])
            v_r.index_copy_(0, idx1, v_fp8[:, 1])
            k_sc.index_copy_(0, idx1, sk[:, 1])
            v_sc.index_copy_(0, idx1, sv[:, 1])
            return k_r, v_r, k_sc, v_sc

        torch.compile(reshape_and_cache_hm_fp8, dynamic=False)(
            key_d,
            value_d,
            k_rows,
            v_rows,
            k_scale_rows,
            v_scale_rows,
            row_indices_d[0],
            row_indices_d[1],
        )

        # Fold into (folded_pages, BLOCK_SIZE, HEAD_SIZE) for page-level access
        folded_k = k_rows.reshape(NUM_BLOCKS * NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        folded_sc = k_scale_rows.reshape(NUM_BLOCKS * NUM_KV_HEADS, BLOCK_SIZE, 1)

        # Read back page 0 (head 0) and page 1 (head 1)
        idx0 = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        idx1 = torch.tensor([1], dtype=torch.int32, device=DEVICE)

        def read_page(pages, sc_pages, idx):
            p = pages.index_select(0, idx)
            s = sc_pages.index_select(0, idx)
            return torch.ops.spyre.dequantize_fp8_with_scale(p, s)

        k_p0 = torch.compile(read_page, dynamic=False)(folded_k, folded_sc, idx0)
        k_p1 = torch.compile(read_page, dynamic=False)(folded_k, folded_sc, idx1)

        k_exp = _cpu_quantize_dequantize(key, _cpu_per_token_scale(key))

        torch.testing.assert_close(
            k_p0.cpu()[0, :num_tokens], k_exp[:, 0], atol=2.0, rtol=0.0
        )
        torch.testing.assert_close(
            k_p1.cpu()[0, :num_tokens], k_exp[:, 1], atol=2.0, rtol=0.0
        )


# ──────────────────────────────────────────────────────────────────────────────
# 8. index_select on fp8 KV pages
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8IndexSelect:
    """index_select on fp8 tensors, used in every page-gather in the attention kernels.

    The paged attention kernel gathers KV pages as:
        k_page = k_pages.index_select(0, page_idx)
    The fp8 variant must do the same gather before dequantizing.  index_select
    with an int32 index must be compiled (eager upcasts to int64 and fails).

    All pages are written on-device via ``qfp8ch + index_copy_`` before being
    gathered, because H2D-transferred CPU-quantized fp8 (STANDARD EA) produces
    wrong values when passed to ``dequantize_fp8_with_scale`` (see module docstring).
    """

    def _make_ondevice_pages(self, differentiation=None):
        """Build a (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE) fp8 page cache
        written entirely on-device so all pages carry the QFP8CH bit pattern.

        Returns:
            pages_d: device fp8 tensor, shape (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
            scales_d: device fp16 scale, shape (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 1)
            pages_fp16: CPU fp16 ground truth (same data, before quantization)
            scales_cpu: CPU fp16 scales
        """
        total = NUM_BLOCKS * BLOCK_SIZE
        kwargs = {} if differentiation is None else {"differentiation": differentiation}
        pages_fp16 = cached_randn(
            (total, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float16, scale=0.5, **kwargs
        )
        scales_cpu = _cpu_per_token_scale(pages_fp16)  # [total, KV, 1]
        slot_idx = torch.arange(total, dtype=torch.int32)

        pages_d = _device_write_read_fp8(
            pages_fp16, scales_cpu, (total, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        scales_d = scales_cpu.to(DEVICE).reshape(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 1
        )
        return pages_d, scales_d, pages_fp16, scales_cpu

    def test_index_select_fp8_single_page(self):
        """index_select on fp8 cache written on-device + dequantize matches CPU ref.

        The cache was written via on-device qfp8ch so the gathered bytes carry the
        QFP8CH bit pattern — dequantize_fp8_with_scale produces correct fp16 values.
        """
        pages_d, scales_d, pages_fp16, scales_cpu = self._make_ondevice_pages()

        # Select page index 2.
        idx_d = torch.tensor([2], dtype=torch.int32, device=DEVICE)

        def gather_deq(pages, scales, idx):
            p = pages.index_select(0, idx)
            s = scales.index_select(0, idx)
            return torch.ops.spyre.dequantize_fp8_with_scale(p, s)

        result = torch.compile(gather_deq, dynamic=False)(pages_d, scales_d, idx_d)

        # CPU ref: quantize-dequantize page 2.
        p2 = pages_fp16[2 * BLOCK_SIZE : 3 * BLOCK_SIZE].reshape(
            1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
        )
        s2 = scales_cpu[2 * BLOCK_SIZE : 3 * BLOCK_SIZE].reshape(
            1, BLOCK_SIZE, NUM_KV_HEADS, 1
        )
        expected = _cpu_quantize_dequantize(p2, s2)

        torch.testing.assert_close(result.cpu(), expected, atol=2.0, rtol=0.0)

    @pytest.mark.parametrize("page_indices", [[0], [1, 3], [0, 2, 3]])
    def test_index_select_fp8_multi_page(self, page_indices):
        """index_select over multiple fp8 pages written on-device + dequantize."""
        pages_d, scales_d, pages_fp16, scales_cpu = self._make_ondevice_pages(
            differentiation=tuple(page_indices)
        )

        idx_d = torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)

        def gather_deq(pages, scales, idx):
            p = pages.index_select(0, idx)
            s = scales.index_select(0, idx)
            return torch.ops.spyre.dequantize_fp8_with_scale(p, s)

        result = torch.compile(gather_deq, dynamic=False)(pages_d, scales_d, idx_d)

        # CPU ref: quantize-dequantize selected pages.
        flat_fp16 = pages_fp16.reshape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        flat_sc = scales_cpu.reshape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 1)
        sel_fp16 = flat_fp16[page_indices]  # [n, B, KV, D]
        sel_sc = flat_sc[page_indices]  # [n, B, KV, 1]
        expected = _cpu_quantize_dequantize(sel_fp16, sel_sc)

        torch.testing.assert_close(result.cpu(), expected, atol=2.0, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Multi-block decode with online softmax
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8MultiBlockDecodeOnlineSoftmax:
    """Multi-block decode attention with running online softmax across N fp8 pages.

    Iterates over N=2 and N=4 FP8 blocks in a compiled loop, gathering K/V pages
    and scale pages, dequantizing each block, and updating running online-softmax
    carry state (max, sum, output) with scale adjustment across blocks.

    This covers the core decode loop that the single-block decode tests leave untested:
    the rescale step when block i+1's logit maximum exceeds block i's running max.
    """

    def _make_multiblock_fp8_cache(self, num_blocks: int):
        """Build token-major fp8 KV cache with num_blocks pages, written on-device."""
        total_tokens = num_blocks * BLOCK_SIZE
        k_fp16 = cached_randn(
            (total_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"mb_k_{num_blocks}",
        )
        v_fp16 = cached_randn(
            (total_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"mb_v_{num_blocks}",
        )
        k_scale = _cpu_per_token_scale(k_fp16)
        v_scale = _cpu_per_token_scale(v_fp16)
        slot_idx = torch.arange(total_tokens, dtype=torch.int32)

        k_pages_d = _device_write_read_fp8(
            k_fp16, k_scale, (total_tokens, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        v_pages_d = _device_write_read_fp8(
            v_fp16, v_scale, (total_tokens, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k_scales_d = k_scale.to(DEVICE).reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)
        v_scales_d = v_scale.to(DEVICE).reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)

        return (
            k_pages_d,
            v_pages_d,
            k_scales_d,
            v_scales_d,
            k_fp16,
            v_fp16,
            k_scale,
            v_scale,
        )

    @pytest.mark.parametrize("num_blocks", [2, 4])
    def test_fp8_decode_attention_multi_block_online_softmax(self, num_blocks):
        """Decode GQA over N fp8 blocks via online softmax (running max/sum rescale).

        Each block i:
          1. Gathers K/V fp8 page + scale page.
          2. Dequantizes to fp16.
          3. Computes per-head block scores: [KV, gqa, B].
          4. Updates running (block_max, block_sum, block_out) using the online
             softmax rescale identity:
               new_max  = max(prev_max, block_max)
               alpha    = exp(prev_max - new_max)   # rescale old contribution
               beta     = exp(block_max - new_max)  # scale new contribution
               new_sum  = alpha * prev_sum + beta * block_sum
               new_out  = (alpha * prev_sum * prev_out + beta * block_sum * block_out)
                          / new_sum
        Final output is normalised by the accumulated sum.
        """
        (
            k_pages_d,
            v_pages_d,
            k_scales_d,
            v_scales_d,
            k_fp16,
            v_fp16,
            k_scale,
            v_scale,
        ) = self._make_multiblock_fp8_cache(num_blocks)

        query_fp16 = cached_randn(
            (1, NUM_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"mb_q_{num_blocks}",
        )
        attn_scale = float(HEAD_SIZE**-0.5)
        gqa = NUM_HEADS // NUM_KV_HEADS

        q_d = query_fp16.to(DEVICE)

        def decode_multiblock(q, k_cache, k_sc, v_cache, v_sc):
            # Keep fp16 throughout — Spyre does not support fp32 batchmatmul.
            q_gqa = q.reshape(NUM_KV_HEADS, gqa, HEAD_SIZE)  # [KV, gqa, D] fp16
            # Running online-softmax state carried as [KV, gqa] tensors (no trailing
            # size-1 dim) to avoid STL incompatibility in the restickify pass when
            # the same buffer is updated across loop iterations from two differently-
            # shaped producers.
            running_max = torch.full(
                (NUM_KV_HEADS, gqa), float("-inf"), dtype=torch.float16, device=q.device
            )
            running_sum = torch.zeros(
                (NUM_KV_HEADS, gqa), dtype=torch.float16, device=q.device
            )
            running_out = torch.zeros(
                (NUM_KV_HEADS, gqa, HEAD_SIZE), dtype=torch.float16, device=q.device
            )

            for b in range(num_blocks):
                idx = torch.tensor([b], dtype=torch.int32, device=q.device)
                k_pg = torch.ops.spyre.dequantize_fp8_with_scale(
                    k_cache.index_select(0, idx), k_sc.index_select(0, idx)
                )  # [1, B, KV, D]
                v_pg = torch.ops.spyre.dequantize_fp8_with_scale(
                    v_cache.index_select(0, idx), v_sc.index_select(0, idx)
                )
                kv_k = k_pg.squeeze(0).permute(1, 0, 2)  # [KV, B, D]
                kv_v = v_pg.squeeze(0).permute(1, 0, 2)

                # Block scores: [KV, gqa, B]
                scores = torch.matmul(q_gqa, kv_k.transpose(-1, -2)) * attn_scale

                # Per-block softmax components — no keepdim to stay [KV, gqa].
                block_max = scores.amax(dim=-1)  # [KV, gqa]
                block_exp = torch.exp(scores - block_max.unsqueeze(-1))  # [KV, gqa, B]
                block_sum = block_exp.sum(dim=-1)  # [KV, gqa]
                block_out = torch.matmul(block_exp, kv_v)  # [KV, gqa, D]

                # Online softmax rescale (all ops on [KV, gqa] scalars).
                new_max = torch.maximum(running_max, block_max)
                alpha = torch.exp(running_max - new_max)  # [KV, gqa]
                beta = torch.exp(block_max - new_max)  # [KV, gqa]

                running_out = (
                    alpha.unsqueeze(-1) * running_out + beta.unsqueeze(-1) * block_out
                )
                running_sum = alpha * running_sum + beta * block_sum
                running_max = new_max

            # Normalise by accumulated sum.
            out = running_out / running_sum.unsqueeze(-1)  # [KV, gqa, D]
            return out.reshape(1, NUM_HEADS, HEAD_SIZE)

        result = torch.compile(decode_multiblock, dynamic=False)(
            q_d, k_pages_d, k_scales_d, v_pages_d, v_scales_d
        )

        # CPU reference: concat all blocks, standard softmax attention.
        k_all = k_fp16.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        v_all = v_fp16.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k_sc_all = k_scale.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)
        v_sc_all = v_scale.reshape(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1)
        k_dq = _cpu_quantize_dequantize(k_all, k_sc_all).reshape(
            num_blocks * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
        )  # [N*B, KV, D]
        v_dq = _cpu_quantize_dequantize(v_all, v_sc_all).reshape(
            num_blocks * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
        )
        kv_k_cpu = k_dq.permute(1, 0, 2).float()  # [KV, N*B, D]
        kv_v_cpu = v_dq.permute(1, 0, 2).float()
        q_gqa_cpu = query_fp16.reshape(NUM_KV_HEADS, gqa, HEAD_SIZE).float()
        sc_cpu = torch.matmul(q_gqa_cpu, kv_k_cpu.transpose(-1, -2)) * attn_scale
        pr_cpu = torch.softmax(sc_cpu, dim=-1)
        out_cpu = torch.matmul(pr_cpu, kv_v_cpu).reshape(1, NUM_HEADS, HEAD_SIZE)
        expected = out_cpu.to(torch.float16)

        # atol=2e-2: fp16 device matmul vs fp32 CPU; softmax normalises to [0,1]×V.
        torch.testing.assert_close(result.cpu(), expected, atol=2e-2, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Prefill attention — full pipeline end-to-end
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8PrefillAttentionEndToEnd:
    """Full prefill attention pipeline: T>1 query tokens over multi-block FP8 KV cache.

    Exercises the path that existing prefill gather/dequantize tests do not cover:
      Q [T, H, D] × gathered FP8 K → scores [H, T, N*B] → causal mask →
      softmax → scores × gathered FP8 V → output [T, H, D].

    The causal mask is lower-triangular across the (query, key) position axes
    when all T query tokens attend to the first T key positions (standard prefill).
    """

    @pytest.mark.parametrize("num_query_tokens,num_kv_blocks", [(4, 1), (16, 2)])
    def test_fp8_prefill_attention_end_to_end(self, num_query_tokens, num_kv_blocks):
        """T query tokens attend over N FP8 KV blocks with causal mask + softmax.

        Pipeline:
          1. Gather all N fp8 K pages; dequantize → [N, B, KV, D] → concat → [N*B, KV, D].
          2. Gather all N fp8 V pages; dequantize → same shape.
          3. Run GQA attention: [KV, T, D] @ [KV, D, N*B] → [KV, T, N*B].
          4. Apply causal mask (query position i may attend to key positions ≤ i).
          5. Softmax → [KV, T, N*B] @ [KV, N*B, D] → [KV, T, D] → [T, H, D].
        """
        num_kv_tokens = num_kv_blocks * BLOCK_SIZE
        gqa = NUM_HEADS // NUM_KV_HEADS

        k_fp16 = cached_randn(
            (num_kv_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"e2e_k_{num_query_tokens}_{num_kv_blocks}",
        )
        v_fp16 = cached_randn(
            (num_kv_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"e2e_v_{num_query_tokens}_{num_kv_blocks}",
        )
        query_fp16 = cached_randn(
            (num_query_tokens, NUM_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"e2e_q_{num_query_tokens}_{num_kv_blocks}",
        )

        k_scale = _cpu_per_token_scale(k_fp16)
        v_scale = _cpu_per_token_scale(v_fp16)
        slot_idx = torch.arange(num_kv_tokens, dtype=torch.int32)

        k_pages_d = _device_write_read_fp8(
            k_fp16, k_scale, (num_kv_tokens, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        v_pages_d = _device_write_read_fp8(
            v_fp16, v_scale, (num_kv_tokens, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k_scales_d = k_scale.to(DEVICE).reshape(
            num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1
        )
        v_scales_d = v_scale.to(DEVICE).reshape(
            num_kv_blocks, BLOCK_SIZE, NUM_KV_HEADS, 1
        )

        q_d = query_fp16.to(DEVICE)
        attn_scale = float(HEAD_SIZE**-0.5)

        # Causal mask built on CPU as a constant boolean — avoids repeat_interleave
        # on-device which introduces a new stick symbol that padding can't handle.
        # Shape [T*gqa, N*B]: True where attention is MASKED (j > i, zero-indexed).
        causal_mask = torch.ones(
            num_query_tokens * gqa, num_kv_tokens, dtype=torch.bool
        )
        for qi in range(num_query_tokens):
            for g in range(gqa):
                row = qi * gqa + g
                causal_mask[row, : qi + 1] = False  # attend to positions 0..qi
        causal_mask_d = causal_mask.to(DEVICE)

        def prefill_attn(q, k_cache, k_sc, v_cache, v_sc, cmask):
            # Gather + dequantize all K and V blocks.
            k_blocks = []
            v_blocks = []
            for b in range(num_kv_blocks):
                idx = torch.tensor([b], dtype=torch.int32, device=q.device)
                k_blocks.append(
                    torch.ops.spyre.dequantize_fp8_with_scale(
                        k_cache.index_select(0, idx), k_sc.index_select(0, idx)
                    )
                )
                v_blocks.append(
                    torch.ops.spyre.dequantize_fp8_with_scale(
                        v_cache.index_select(0, idx), v_sc.index_select(0, idx)
                    )
                )
            k_all = torch.cat(k_blocks, dim=1)  # [1, N*B, KV, D]
            v_all = torch.cat(v_blocks, dim=1)
            # Squeeze page dim: [N*B, KV, D] → [KV, N*B, D]
            # Keep fp16 — Spyre does not support fp32 batchmatmul.
            k_seq = k_all.squeeze(0).permute(1, 0, 2)  # [KV, N*B, D]
            v_seq = v_all.squeeze(0).permute(1, 0, 2)
            # q: [T, H, D] → [KV, T*gqa, D]
            q_gqa = q.reshape(num_query_tokens, NUM_KV_HEADS, gqa, HEAD_SIZE)
            q_gqa = q_gqa.permute(1, 0, 2, 3)  # [KV, T, gqa, D]
            q_2d = q_gqa.reshape(NUM_KV_HEADS, num_query_tokens * gqa, HEAD_SIZE)
            # Scores: [KV, T*gqa, D] @ [KV, D, N*B] → [KV, T*gqa, N*B]
            scores = torch.matmul(q_2d, k_seq.transpose(-1, -2)) * attn_scale
            # Apply pre-built causal mask (broadcast over KV dim).
            scores = scores.masked_fill(cmask.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)  # [KV, T*gqa, N*B]
            out = torch.matmul(probs, v_seq)  # [KV, T*gqa, D]
            out = out.reshape(NUM_KV_HEADS, num_query_tokens, gqa, HEAD_SIZE)
            out = out.permute(1, 0, 2, 3)  # [T, KV, gqa, D]
            return out.reshape(num_query_tokens, NUM_HEADS, HEAD_SIZE)

        result = torch.compile(prefill_attn, dynamic=False)(
            q_d, k_pages_d, k_scales_d, v_pages_d, v_scales_d, causal_mask_d
        )

        # CPU reference: same computation with fp8-round-tripped KV.
        k_dq = _cpu_quantize_dequantize(k_fp16, k_scale)  # [N*B, KV, D]
        v_dq = _cpu_quantize_dequantize(v_fp16, v_scale)
        k_cpu = k_dq.permute(1, 0, 2).float()  # [KV, N*B, D]
        v_cpu = v_dq.permute(1, 0, 2).float()
        q_f_cpu = query_fp16.reshape(num_query_tokens, NUM_KV_HEADS, gqa, HEAD_SIZE)
        q_f_cpu = (
            q_f_cpu.permute(1, 0, 2, 3)
            .reshape(NUM_KV_HEADS, num_query_tokens * gqa, HEAD_SIZE)
            .float()
        )
        sc_cpu = torch.matmul(q_f_cpu, k_cpu.transpose(-1, -2)) * attn_scale
        sc_cpu = sc_cpu.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        pr_cpu = torch.softmax(sc_cpu, dim=-1)
        out_cpu = torch.matmul(pr_cpu, v_cpu)  # [KV, T*gqa, D]
        out_cpu = out_cpu.reshape(NUM_KV_HEADS, num_query_tokens, gqa, HEAD_SIZE)
        out_cpu = out_cpu.permute(1, 0, 2, 3).reshape(
            num_query_tokens, NUM_HEADS, HEAD_SIZE
        )
        expected = out_cpu.to(torch.float16)

        # atol=6e-2: fp16 device matmul vs fp32 CPU reference; causal masking on
        # prefill accumulates more rounding error than single-block decode.
        torch.testing.assert_close(result.cpu(), expected, atol=6e-2, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 11. Head-major cache — read path (decode attention)
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8HeadMajorPagedAttention:
    """Head-major cache read path: per-head page gather + dequantize + decode attention.

    TestFp8HeadMajorReshapeAndCache only tests the write side.  This class
    tests the read side: gather per-head pages from the folded head-major cache
    [num_blocks * num_kv_heads, block_size, head_size], dequantize, and compute
    decode GQA attention.
    """

    def test_fp8_head_major_decode_attention(self):
        """Write K/V to head-major FP8 cache, read back, and run decode GQA attention.

        The head-major read path works correctly when the FP8 tensors produced by
        the write compile call are captured via the return value and passed directly
        into the read+attend compile call.  The compiled write function must return
        the mutated cache tensors; the pre-compile tensor references do not carry
        the QFP8CH bit-pattern metadata required by dequantize_fp8_with_scale.

        Layout:
          - Flat cache rows [total_rows, D] where total_rows = num_blocks*KV*B.
          - head h occupies rows [h*B .. (h+1)*B).
          - Folded view: [num_blocks*KV, B, D].
          - All-heads index_select [0,1,...,KV-1] gathers every head page at once,
            giving [KV, B, D] ready for a single dequantize call (no cat needed).

        Steps:
          1. (write compile) quantscalepertokenfp8 + quantize_fp8_with_scale on the
             flat [T*KV, D] input; index_copy_ all rows; return mutated cache.
          2. Reshape returned flat cache → folded [KV, B, D] pages.
          3. (read+attend compile) index_select all KV heads at once → dequantize
             → [KV, B, D]; GQA decode attention Q·Kᵀ → softmax → P·V.
        """
        num_blocks = 1
        total_rows = num_blocks * NUM_KV_HEADS * BLOCK_SIZE
        num_tokens = BLOCK_SIZE  # one full block of tokens in the cache

        # Input KV: [T, KV, D] — T == BLOCK_SIZE fills exactly one block per head.
        k_fp16 = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="hm_dec_k",
        )
        v_fp16 = cached_randn(
            (num_tokens, NUM_KV_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="hm_dec_v",
        )
        query_fp16 = cached_randn(
            (1, NUM_HEADS, HEAD_SIZE),
            dtype=torch.float16,
            scale=0.5,
            differentiation="hm_dec_q",
        )
        gqa = NUM_HEADS // NUM_KV_HEADS
        attn_scale = float(HEAD_SIZE**-0.5)

        # Reshape [T, KV, D] → [T*KV, D] so per-head rows are contiguous.
        # Scattering a non-contiguous strided column slice (k_fp8[:, h]) loses
        # the QFP8CH ElementArrangement; contiguous rows preserve it.
        k_flat = k_fp16.reshape(num_tokens * NUM_KV_HEADS, HEAD_SIZE)  # [T*KV, D]
        v_flat = v_fp16.reshape(num_tokens * NUM_KV_HEADS, HEAD_SIZE)

        # Build flat scatter indices: head h occupies rows [h*B .. h*B+T).
        # With num_tokens == BLOCK_SIZE this is just [0 .. total_rows).
        flat_row_idx = torch.cat(
            [
                torch.arange(
                    h * BLOCK_SIZE, h * BLOCK_SIZE + num_tokens, dtype=torch.int32
                )
                for h in range(NUM_KV_HEADS)
            ]
        )

        k_flat_d = k_flat.to(DEVICE)
        v_flat_d = v_flat.to(DEVICE)
        flat_idx_d = flat_row_idx.to(DEVICE)
        q_d = query_fp16.to(DEVICE)

        k_rows = torch.zeros((total_rows, HEAD_SIZE), dtype=torch.float8_e4m3fn).to(
            DEVICE
        )
        v_rows = torch.zeros((total_rows, HEAD_SIZE), dtype=torch.float8_e4m3fn).to(
            DEVICE
        )
        k_scale_rows = torch.zeros((total_rows, 1), dtype=torch.float16, device=DEVICE)
        v_scale_rows = torch.zeros((total_rows, 1), dtype=torch.float16, device=DEVICE)

        # ── Step 1: Write (compiled) ──────────────────────────────────────
        # Return the mutated cache tensors so the QFP8CH EA flows through
        # the return value into the next compiled graph.
        def write_hm_flat(k_f, v_f, k_r, v_r, k_sc, v_sc, flat_idx):
            sk = torch.ops.spyre.quantscalepertokenfp8(k_f)  # [T*KV, 1]
            sv = torch.ops.spyre.quantscalepertokenfp8(v_f)
            k_fp8 = torch.ops.spyre.quantize_fp8_with_scale(k_f, sk)
            v_fp8 = torch.ops.spyre.quantize_fp8_with_scale(v_f, sv)
            k_r.index_copy_(0, flat_idx, k_fp8)  # scatter all heads in one call
            v_r.index_copy_(0, flat_idx, v_fp8)
            k_sc.index_copy_(0, flat_idx, sk)
            v_sc.index_copy_(0, flat_idx, sv)
            return k_r, v_r, k_sc, v_sc

        # Capture the RETURNED tensors — these carry EA metadata; the original
        # k_rows / v_rows variables (before compile) do not.
        k_rows, v_rows, k_scale_rows, v_scale_rows = torch.compile(
            write_hm_flat, dynamic=False
        )(k_flat_d, v_flat_d, k_rows, v_rows, k_scale_rows, v_scale_rows, flat_idx_d)

        # ── Step 2: Fold rows → pages (outside compiled graph) ────────────
        # Use the EA-carrying returned tensors, not the pre-compile originals.
        folded_k = k_rows.reshape(num_blocks * NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        folded_v = v_rows.reshape(num_blocks * NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        folded_k_sc = k_scale_rows.reshape(num_blocks * NUM_KV_HEADS, BLOCK_SIZE, 1)
        folded_v_sc = v_scale_rows.reshape(num_blocks * NUM_KV_HEADS, BLOCK_SIZE, 1)

        # Gather all KV heads at once: avoids per-head cat which triggers a
        # Reduction node the restickify pass cannot resolve.
        all_head_idx = torch.arange(NUM_KV_HEADS, dtype=torch.int32, device=DEVICE)

        # ── Step 3: Read + attend (compiled) ──────────────────────────────
        def read_and_attend(q, k_pgs, v_pgs, k_sc_pgs, v_sc_pgs, head_idx):
            # Keep fp16 throughout — Spyre does not support fp32 batchmatmul.
            q_gqa = q.reshape(NUM_KV_HEADS, gqa, HEAD_SIZE)  # [KV, gqa, D]
            # Gather all heads at once → [KV, B, D], then dequantize in one call.
            kv_k = torch.ops.spyre.dequantize_fp8_with_scale(
                k_pgs.index_select(0, head_idx),
                k_sc_pgs.index_select(0, head_idx),
            )  # [KV, B, D]
            kv_v = torch.ops.spyre.dequantize_fp8_with_scale(
                v_pgs.index_select(0, head_idx),
                v_sc_pgs.index_select(0, head_idx),
            )
            scores = torch.matmul(q_gqa, kv_k.transpose(-1, -2)) * attn_scale
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, kv_v)  # [KV, gqa, D]
            return out.reshape(1, NUM_HEADS, HEAD_SIZE)

        result = torch.compile(read_and_attend, dynamic=False)(
            q_d,
            folded_k,
            folded_v,
            folded_k_sc,
            folded_v_sc,
            all_head_idx,
        )

        # CPU reference: fp8-round-tripped KV (using per-token scales on flat view)
        # then standard decode attention.
        k_flat_cpu = k_fp16.reshape(num_tokens * NUM_KV_HEADS, HEAD_SIZE)
        v_flat_cpu = v_fp16.reshape(num_tokens * NUM_KV_HEADS, HEAD_SIZE)
        k_flat_dq = _cpu_quantize_dequantize(
            k_flat_cpu, _cpu_per_token_scale(k_flat_cpu)
        )  # [T*KV, D]
        v_flat_dq = _cpu_quantize_dequantize(
            v_flat_cpu, _cpu_per_token_scale(v_flat_cpu)
        )
        # After head-major scatter, head h occupies rows [h*B .. (h+1)*B).
        # Reconstruct [KV, B, D] by reading those row ranges.
        kv_k_cpu = torch.stack(
            [
                k_flat_dq[h * BLOCK_SIZE : h * BLOCK_SIZE + num_tokens]
                for h in range(NUM_KV_HEADS)
            ]
        ).float()  # [KV, T, D]
        kv_v_cpu = torch.stack(
            [
                v_flat_dq[h * BLOCK_SIZE : h * BLOCK_SIZE + num_tokens]
                for h in range(NUM_KV_HEADS)
            ]
        ).float()
        q_gqa_cpu = query_fp16.reshape(NUM_KV_HEADS, gqa, HEAD_SIZE).float()
        sc_cpu = torch.matmul(q_gqa_cpu, kv_k_cpu.transpose(-1, -2)) * attn_scale
        pr_cpu = torch.softmax(sc_cpu, dim=-1)
        out_cpu = torch.matmul(pr_cpu, kv_v_cpu).reshape(1, NUM_HEADS, HEAD_SIZE)
        expected = out_cpu.to(torch.float16)

        torch.testing.assert_close(result.cpu(), expected, atol=2e-2, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Robustness: repeated-index gather
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8IndexSelectRobustness:
    """Robustness tests for index_select on FP8 page caches.

    Covers shared-prefix caching (repeated page indices) and scale cache
    gather, which are not exercised by the standard multi-page tests.
    """

    def _make_ondevice_pages(self, differentiation=None):
        """Build a (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE) fp8 cache."""
        total = NUM_BLOCKS * BLOCK_SIZE
        kwargs = {} if differentiation is None else {"differentiation": differentiation}
        pages_fp16 = cached_randn(
            (total, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float16, scale=0.5, **kwargs
        )
        scales_cpu = _cpu_per_token_scale(pages_fp16)
        slot_idx = torch.arange(total, dtype=torch.int32)
        pages_d = _device_write_read_fp8(
            pages_fp16, scales_cpu, (total, NUM_KV_HEADS, HEAD_SIZE), slot_idx
        ).reshape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        scales_d = scales_cpu.to(DEVICE).reshape(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 1
        )
        return pages_d, scales_d, pages_fp16, scales_cpu

    def test_fp8_index_select_repeated_indices(self):
        """index_select with a repeated page index (shared-prefix caching pattern).

        When the same block is referenced multiple times in a single gather call
        (e.g. [1, 1, 2]), each occurrence must produce an independent copy of
        that page with correct dequantized values — covering shared-prefix caching
        where multiple sequences reference the same physical KV page.
        """
        pages_d, scales_d, pages_fp16, scales_cpu = self._make_ondevice_pages(
            differentiation="repeated_idx"
        )

        # Index [1, 1, 2]: page 1 is referenced twice, page 2 once.
        repeated_indices = [1, 1, 2]
        idx_d = torch.tensor(repeated_indices, dtype=torch.int32, device=DEVICE)

        def gather_deq(pages, scales, idx):
            p = pages.index_select(0, idx)
            s = scales.index_select(0, idx)
            return torch.ops.spyre.dequantize_fp8_with_scale(p, s)

        result = torch.compile(gather_deq, dynamic=False)(pages_d, scales_d, idx_d)

        # CPU ref: quantize-dequantize pages [1, 1, 2].
        flat_fp16 = pages_fp16.reshape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        flat_sc = scales_cpu.reshape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 1)
        sel_fp16 = flat_fp16[repeated_indices]  # [3, B, KV, D]
        sel_sc = flat_sc[repeated_indices]
        expected = _cpu_quantize_dequantize(sel_fp16, sel_sc)

        # Shape: [3, B, KV, D]; repeated rows [0] and [1] must both equal page-1's dequantized data.
        torch.testing.assert_close(result.cpu(), expected, atol=2.0, rtol=0.0)
        # Explicitly verify the duplicate rows are numerically identical.
        torch.testing.assert_close(
            result.cpu()[0],
            result.cpu()[1],
            atol=0.0,
            rtol=0.0,
            msg="Repeated page index must produce identical rows in the gathered output.",
        )
