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

"""Tests for the Spyre cos / sin decompositions.

``aten.cos`` and ``aten.sin`` used to be CPU-offloaded fallback ops. They are
now decomposed on device by ``spyre_cos`` / ``spyre_sin`` in
``torch_spyre/_inductor/decompositions.py``, built entirely from arithmetic
primitives Spyre lowers natively (``floor``, ``mul``, ``add``, ``sub``).
This file tests those two functions -- directly on CPU for the numerics, and
through ``torch.cos`` / ``torch.sin`` under ``torch.compile`` for the
registration and on-device path.

All model YAML files (tests/resource/models/*.yaml, non-_spyre variants)
show fp32 as the input dtype at the cos/sin call site:

  Model                          shape (prefill)    shape (decode)
  granite-4.1-8b                 [1, 29, 128]       [1, 1, 128]
  granite-3.3-8b-instruct        [1, 41, 128]       —
  Meta-Llama-3.1-8B-Instruct     [1, 12, 128]       [1, 1, 128]
  Qwen2.5-7B-Instruct            [1, 39, 128]       —
  Ministral-3-14B-Instruct-2512  [1, 14, 128]       —
  Mistral-Small-3.2-24B (text)   [1, 855, 128]      [1, 1, 128]
  Mistral-Small-3.2-24B (vision) [1064, 64]         —
  gpt-oss-20b (non-contiguous)   [1, 11, 32]        —
  gemma-4-26B-A4B-it             [1, 34, 256]       [1, 1, 256]

Algorithm under test
--------------------
Step 1 — Cody-Waite two-term range reduction (ops: floor, mul, add, sub):

    PI is split into two parts so that PI_HI + PI_LO == π exactly in fp64:
        PI_HI = 3.1415927410125732   (nearest fp32 to π)
        PI_LO = -8.742278012618954e-8 (fp64 residual: π - PI_HI)

    k    = floor(x / π + 0.5)              # round-to-nearest via floor
    x_r  = (x − k × PI_HI) − k × PI_LO    # two-step subtraction, |x_r| ≤ π/2
    sign = 1 − 2 × (k − 2 × floor(k / 2)) # (−1)^k

    torch.round is NOT used — it is not implemented in the Spyre codegen.
    floor(x + 0.5) is the standard round-to-nearest equivalent using only floor.

    Using a single-term π (PI_fp32 = 3.1415927) accumulates error at
    ~8.7e-8 per unit of k; at k=318 (x≈1000) the residual x_r is off by
    ~2.8e-5, dominating the polynomial error. The two-term split reduces
    the x_r error to machine-noise levels (< 2e-12 for |x| ≤ 128000).

Step 2 — degree-9 Horner polynomial on |x_r| ≤ π/2:

    cos(x_r) ≈ 1 + x²(−1/2 + x²(1/24 + x²(−1/720 + x²/40320)))
    sin(x_r) ≈ x_r(1 + x²(−1/6 + x²(1/120 + x²(−1/5040 + x²/362880))))
    cos(x)   = sign × cos(x_r)
    sin(x)   = sign × sin(x_r)

    The polynomial itself has a ~2.5e-5 rounding floor in fp32 for cos
    (from partial cancellation of large terms near |x_r| = π/2).  This is
    irreducible without switching to fp64 evaluation.

Accuracy summary (measured on RoPE-realistic inputs):
    worst-case cos error: ~5.3e-5   (head_dim=128/256, seq_len=1064)
    worst-case sin error: ~5.3e-5   (head_dim=128/256, seq_len=1064)
    safe tolerance:        1e-4

Note: fp32 itself can only represent values near 128000 with spacing ~0.008,
which is larger than π; a linspace sweep at |x|_max=128000 is meaningless in
fp32. The tests below are bounded to the largest real RoPE input (seq_len=1064,
inv_freq[0]=1.0 → |x|_max=1063).
"""

import math
import warnings

import pytest
import torch

from torch_spyre._inductor.decompositions import (
    get_spyre_decomp_table,
    spyre_cos,
    spyre_sin,
)
from torch_spyre.ops.fallbacks import FallbackWarning, fallback_ops

# ---------------------------------------------------------------------------
# Section 0: registration -- cheap guard, no device needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["cos", "sin"])
def test_registered_as_decomposition_not_fallback(op_name):
    """cos / sin must reach Inductor as a decomposition, never as a fallback.

    Re-adding either op to ``register_fallback_default`` would silently put
    the CPU round-trip back on every RoPE forward pass, so assert both halves.
    """
    op = getattr(torch.ops.aten, op_name).default
    assert op in get_spyre_decomp_table(), (
        f"{op} is missing from the Spyre decomposition table"
    )
    assert op not in fallback_ops, (
        f"{op} is registered as a CPU fallback; it has a Spyre decomposition"
    )


# ---------------------------------------------------------------------------
# Section 1: CPU-only numerical accuracy of the decomposition bodies
# ---------------------------------------------------------------------------


class TestTaylorAccuracyCpu:
    """Validate decomposition accuracy on CPU across a broad input range.

    ``spyre_cos`` / ``spyre_sin`` are plain functions over torch ops, so they
    evaluate on CPU tensors unchanged. These tests need no Spyre device.

    The meaningful upper bound for fp32 tests is |x|_max ~ 1063 (the actual
    maximum RoPE embedding value for seq_len=1064, inv_freq[0]=1.0).  Values
    larger than ~10000 cannot be meaningfully tested in fp32 because the fp32
    representable spacing (0.008 at 128000) exceeds π, making the ground-truth
    cos/sin input itself ill-defined.
    """

    @pytest.mark.parametrize("x_max", [math.pi, 10.0, 100.0, 1000.0])
    def test_cos_fp32_accuracy(self, x_max):
        """Max fp32 error < 1e-4 across the swept range."""
        x = torch.linspace(-x_max, x_max, steps=10001, dtype=torch.float32)
        max_err = (spyre_cos(x) - torch.cos(x)).abs().max().item()
        assert max_err < 1e-4, f"|x|_max={x_max}: max cos error {max_err:.3e} >= 1e-4"

    @pytest.mark.parametrize("x_max", [math.pi, 10.0, 100.0, 1000.0])
    def test_sin_fp32_accuracy(self, x_max):
        """Max fp32 error < 1e-4 across the swept range."""
        x = torch.linspace(-x_max, x_max, steps=10001, dtype=torch.float32)
        max_err = (spyre_sin(x) - torch.sin(x)).abs().max().item()
        assert max_err < 1e-4, f"|x|_max={x_max}: max sin error {max_err:.3e} >= 1e-4"

    def test_boundary_values(self):
        """Values at k*π: cos(k*π) = (−1)^k, sin(k*π) = 0."""
        boundaries = torch.tensor(
            [k * math.pi for k in range(-4, 5)], dtype=torch.float32
        )
        assert (spyre_cos(boundaries) - torch.cos(boundaries)).abs().max() < 1e-4
        assert (spyre_sin(boundaries) - torch.sin(boundaries)).abs().max() < 1e-4

    def test_rope_realistic_inputs(self):
        """Inputs drawn from the actual RoPE frequency formula.

        inv_freq[k] = 1 / (10000 ^ (2k / head_dim)).
        emb = cat([inv_freq @ pos_ids, inv_freq @ pos_ids]).
        Covers head_dim ∈ {64, 128, 256} and seq_len ∈ {1, 29, 128, 855, 1064}.
        Worst case: head_dim=128/256, seq_len=1064 → max error ~ 5.3e-5.
        """
        for head_dim in (64, 128, 256):
            d = head_dim // 2
            inv_freq = 1.0 / (10000.0 ** (torch.arange(d, dtype=torch.float32) / d))
            for seq_len in (1, 29, 128, 855, 1064):
                pos_ids = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
                emb = torch.cat([pos_ids * inv_freq, pos_ids * inv_freq], dim=-1)

                cos_err = (spyre_cos(emb) - torch.cos(emb)).abs().max().item()
                sin_err = (spyre_sin(emb) - torch.sin(emb)).abs().max().item()
                assert cos_err < 1e-4, (
                    f"head_dim={head_dim}, seq_len={seq_len}: cos error {cos_err:.3e}"
                )
                assert sin_err < 1e-4, (
                    f"head_dim={head_dim}, seq_len={seq_len}: sin error {sin_err:.3e}"
                )


# ---------------------------------------------------------------------------
# Section 2: Spyre compilation and correctness, via torch.cos / torch.sin
# ---------------------------------------------------------------------------

# Shapes taken directly from the non-_spyre model YAML files.
# All use fp32 at the cos/sin call site.
_MODEL_SHAPES = [
    pytest.param((1, 29, 128), id="granite_4_1_8b_prefill"),
    pytest.param((1, 1, 128), id="decode_128"),  # granite / llama / mistral decode
    pytest.param((1, 12, 128), id="llama_3_1_8b_prefill"),
    pytest.param((1, 39, 128), id="qwen2_5_7b_prefill"),
    pytest.param((1, 14, 128), id="ministral_3_14b_prefill"),
    pytest.param((1, 855, 128), id="mistral_small_text_prefill"),
    pytest.param((1064, 64), id="mistral_small_vision_pixtral"),
    pytest.param((1, 11, 32), id="gpt_oss_20b_prefill"),  # non-contiguous in model
    pytest.param((1, 34, 256), id="gemma4_26b_prefill"),
    pytest.param((1, 1, 256), id="gemma4_26b_decode"),
]


def _compile_on_spyre(op, x):
    """Compile ``op`` for Spyre and return the CPU-side result.

    Turns FallbackWarning into an error: a CPU offload here would mean the
    decomposition did not take effect, which is the whole point of the change.
    """

    @torch.compile(dynamic=False)
    def fn(t):
        return op(t)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        return fn(x.to("spyre")).cpu()


@pytest.mark.parametrize("shape", _MODEL_SHAPES)
def test_cos_compiles_and_accurate(shape):
    """torch.cos compiles on Spyre without a fallback and matches CPU."""
    x = torch.randn(*shape, dtype=torch.float32)
    out = _compile_on_spyre(torch.cos, x)
    torch.testing.assert_close(out, torch.cos(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", _MODEL_SHAPES)
def test_sin_compiles_and_accurate(shape):
    """torch.sin compiles on Spyre without a fallback and matches CPU."""
    x = torch.randn(*shape, dtype=torch.float32)
    out = _compile_on_spyre(torch.sin, x)
    torch.testing.assert_close(out, torch.sin(x), atol=1e-4, rtol=1e-4)


def test_cos_sin_non_contiguous():
    """Non-contiguous input: gpt-oss-20b actual stride pattern [352, 1, 11].

    The logical shape is [1, 11, 32] with stride[1]=1 and stride[2]=11
    (transposed last two dims relative to row-major).  Pointwise ops
    delegate stride handling to the compiler; this confirms no crash and
    correct output.
    """
    shape = (1, 11, 32)
    storage = torch.randn(352, dtype=torch.float32)  # stride[0]=352
    x = storage.as_strided(shape, stride=(352, 1, 11))

    @torch.compile(dynamic=False)
    def fn(t):
        return torch.cos(t), torch.sin(t)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        cos_out, sin_out = fn(x.to("spyre"))
    torch.testing.assert_close(cos_out.cpu(), torch.cos(x), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sin_out.cpu(), torch.sin(x), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Section 3: End-to-end RoPE kernel test
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard rotate_half used in apply_rotary_pos_emb."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def test_rope_end_to_end():
    """Full RoPE pipeline on Spyre using the cos/sin decompositions.

    Mirrors the HuggingFace Transformers pattern:
        emb       = cat([inv_freq @ pos_ids, inv_freq @ pos_ids])  # fp32
        cos_emb   = cos(emb)                                       # fp32, STANDARD EA
        sin_emb   = sin(emb)
        q_embed   = q * cos_emb + rotate_half(q) * sin_emb
        k_embed   = k * cos_emb + rotate_half(k) * sin_emb

    Uses Llama-3.1-8B prefill shape: emb [1, S=12, D=128], q/k [B=1, H=8, S=12, D=128].
    All tensors kept in fp32 for a clean EA-STANDARD path.  In production models
    the same fp32 cos/sin would be cast to fp16 before the rotary multiply, but
    that cast produces a FP32_TO_DL16 (staggered) EA on Spyre which cannot be
    directly multiplied against STANDARD fp16 q/k unless the staggered operand
    broadcasts on its stick dimension.  What this test pins down is that the
    decomposed ops compile and execute correctly on Spyre inside a real RoPE
    graph; the fp16 integration path in real models is a separate concern (the
    model compiles today because the cast and rotary apply typically fall inside
    a single graph, or cos/sin are pre-computed as STANDARD fp16 via a different
    lowering path).

    EA note: fp32 cos/sin output has STANDARD EA.  All q/k tensors are fp32
    STANDARD.  The cos_emb/sin_emb tensors are unsqueezed from [1, S, D] to
    [1, 1, S, D] so they broadcast over the H dimension in the rotary multiply;
    this is a size-1 broadcast on dim 1 which is EA-compatible.
    """
    B, H, S, D = 1, 8, 12, 128

    @torch.compile(dynamic=False)
    def rope_spyre(q, k, emb):
        """cos/sin + rotary multiply, all in fp32, all STANDARD EA."""
        cos_emb = torch.cos(emb).unsqueeze(1)  # [1, 1, S, D]
        sin_emb = torch.sin(emb).unsqueeze(1)
        return (
            q * cos_emb + _rotate_half(q) * sin_emb,
            k * cos_emb + _rotate_half(k) * sin_emb,
        )

    def rope_ref(q, k, emb):
        cos_emb = torch.cos(emb).unsqueeze(1)
        sin_emb = torch.sin(emb).unsqueeze(1)
        return (
            q * cos_emb + _rotate_half(q) * sin_emb,
            k * cos_emb + _rotate_half(k) * sin_emb,
        )

    torch.manual_seed(0)
    d = D // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(d, dtype=torch.float32) / d))
    pos_ids = torch.arange(S, dtype=torch.float32)
    freqs = torch.outer(pos_ids, inv_freq)  # [S, d]
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)  # [1, S, D]

    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)

    q_ref, k_ref = rope_ref(q, k, emb)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        q_out, k_out = rope_spyre(q.to("spyre"), k.to("spyre"), emb.to("spyre"))

    torch.testing.assert_close(q_out.cpu(), q_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k_out.cpu(), k_ref, atol=1e-4, rtol=1e-4)
