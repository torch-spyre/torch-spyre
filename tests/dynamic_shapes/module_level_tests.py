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

"""Consolidated standalone test suite for Spyre modules with dynamic batch dimension.

Consolidated dynamic batch module tests, one parametrized test per distinct module.
Tests shared by several models carry the model='common' label and list the models in `note`.

Each test case follows the belowdynamic batch dimension rules:
- Range rules: 8 <= lo < hi <= 32000, multiples of granularity (= tile), lo >= 2 * tile, hi <= 32 * tile.
- Seven runtime sizes tested per case (min, max, and 5 intermediate multiples).
- Operands with symbolic dim 0 are at least 2-D (1D token ids -> [T, 1] columns).
- Standalone: provides all necessary fallbacks for execution on both host and Spyre hardware.
- Full XFAIL mapping: every known issue and failure reason is mapped and marked with strict=True.
"""

from dataclasses import dataclass
import functools
import hashlib
from pathlib import Path
import sys
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from inductor.utils_inductor import compare_with_cpu  # type: ignore[attr-defined]
from torch._dynamo.utils import counters
from torch._inductor.exc import InductorError

# ---------------------------------------------------------------------------
# Setup sys.path
# ---------------------------------------------------------------------------

_INDUCTOR_UTILS_DIR = Path(__file__).parent.resolve()
if str(_INDUCTOR_UTILS_DIR) not in sys.path and _INDUCTOR_UTILS_DIR.exists():
    sys.path.insert(0, str(_INDUCTOR_UTILS_DIR))


def _make_generator(*args) -> torch.Generator:
    """Return a seeded Generator whose seed is a stable SHA-256 hash of args."""
    key = repr(args).encode()
    seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@functools.lru_cache(maxsize=None)
def cached_randn(
    shape, differentiation=None, abs=False, dtype=torch.float16, scale=1.0
):
    """Return a deterministically-seeded random tensor cached by arguments."""
    gen = _make_generator(shape, differentiation, abs, dtype, scale)
    out = torch.randn(shape, dtype=dtype, generator=gen) * scale
    return out if not abs else torch.abs(out)


@functools.lru_cache(maxsize=None)
def cached_randint(shape, differentiation=None, low=0, high=512, dtype=torch.int32):
    """Return a cached integer tensor with given shape and dtype."""
    gen = _make_generator(shape, differentiation, low, high, dtype)
    return torch.randint(low, high, shape, dtype=dtype, generator=gen)


# ---------------------------------------------------------------------------
# Global Constants, Tolerances & XFAIL Reason Definitions
# ---------------------------------------------------------------------------

TIGHT = 1e-2
ATTN_TOL = 2e-2
LOOSE = 1e-1
FP8_TOL = 2e-1

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP16_MAX = float(torch.finfo(torch.float16).max)
SCALE_MIN = 1.1920928955078125e-07

# XFAIL definitions: (reason, raises) with the exact exception each tracked issue raises.
# Every compile failure surfaces from torch.compile as torch._inductor.exc.InductorError
# (the inner AssertionError / KeyError / RuntimeError / Unsupported is wrapped by it).
_LAYER_NORM_IN_LOOP = (
    "F.layer_norm in for_each_tile body: Incompatible ElementArrangement in multi-arg op, op: while_loop, #6612",
    InductorError,
)
_ROW_AXIS_RESHAPE = (
    "view changing the tile row axis: AssertionError affine stride is not divisible by the loop step, #6613",
    InductorError,
)
_ROW_SHRINK_STRIDE = (
    "reshape of the tile row axis: AssertionError affine stride 128 is not divisible by the loop step 512, #6613",
    InductorError,
)
_FP8_INDUCTOR_BUF = (
    "FP8 linear in for_each_tile: KeyError No FX node for while_loop_body_graph_0_0_buf9 in split_multi_ops, #6698",
    InductorError,
)
_INDEX_COPY_IN_LOOP = (
    "index_copy_ in for_each_tile: Unsupported indirect symbol tmp0 not found in indirect_sizes, #6483 / #6412",
    InductorError,
)
_KV_CACHE_WRITE_FAIL = (
    "index_copy_ on a cloned 4-D KV block in for_each_tile with a matmul-computed source compiles but writes the wrong slot row on Spyre (output mismatches CPU), #6412",
    AssertionError,
)
_VISION_MASK_FAIL = (
    "AssertionError in _consume_tile_dim_markers: multiple newly-inlined MemoryDep reads, #6479",
    InductorError,
)
_VISION_POOL_PROJ_FAIL = (
    "reshape(9, -1, C).sum(0) in for_each_tile: AssertionError Unsupported coordinate expression 64*c2/9 (non-unit fraction in views.py Term.from_coordinate), #1353",
    InductorError,
)
_MOE_CODEGEN_FAIL = (
    "fp32 -> int32 cast in for_each_tile: Unsupported unexpected argument PointwiseOp(floor) to mul in codegen, #6410",
    InductorError,
)
_SIGLIP_ENCODER_FAIL = (
    "SigLIP encoder: Unsupported batchmatmul cannot restickify any input layout of x to carry x_var=d3, #6556",
    InductorError,
)
_YOLO_CONV_FAIL = (
    "Unsupported batchmatmul cannot restickify any input layout of x to carry x_var=d3, #6556",
    InductorError,
)
_YOLO_BN_FAIL = (
    "LoopStructureError after insert_restickify_padding: the loop structure is inconsistent, #6487",
    InductorError,
)
_YOLO_MAXPOOL_FAIL = (
    "AssertionError _delinearize_index: recovered coordinate is out of bounds in for_each_tile, #6492",
    InductorError,
)
_YOLO_UPSAMPLE_FAIL = (
    "Unsupported multiple indirect symbols in arg1_1: [tmp0, tmp1], #4471",
    InductorError,
)
_YOLO_SPPF_FAIL = (
    "RuntimeError Cannot resolve target for 'index_expr' in split_multi_ops, #6644",
    InductorError,
)
_YOLO_C3K_FAIL = (
    "AssertionError assert len(syms) == 1 in _pad_restickify_input for transpose fused into matmul, #4671",
    InductorError,
)
DEPTHWISE_PAD = "Depthwise conv2d currently only supports zero padding"
DEPTHWISE_VIEW = "Input and output channels and groups should all be equal"


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Empty Dynamo cache per test."""
    torch._dynamo.reset()
    yield


# ---------------------------------------------------------------------------
# Canonical / Shared Module Bodies
# ---------------------------------------------------------------------------


def embedding_3table_body(
    input_ids, positions, w_word, w_seg, w_pos, ln_weight, ln_bias, eps=1e-5
):
    """Embedding with word, segment, position tables + LayerNorm (BERT/RoBERTa/GraniteEmbedding)."""
    input_ids = input_ids[:, 0]
    positions = positions[:, 0]
    token_type_ids = torch.zeros_like(input_ids)
    inputs_embeds = F.embedding(input_ids, w_word)
    tte = F.embedding(token_type_ids, w_seg)
    pe = F.embedding(positions, w_pos)
    x = torch.add(inputs_embeds, tte)
    x = torch.add(x, pe)
    return F.layer_norm(x, (w_word.shape[1],), ln_weight, ln_bias, eps=eps)


def embedding_single_lookup_body(input_ids, weight, multiplier=1.0):
    """Single table embedding with optional multiplier."""
    ids = input_ids[:, 0]
    out = F.embedding(ids, weight)
    if multiplier != 1.0:
        out = out * multiplier
    return out


def rms_norm_plain_body(hidden, weight=None, eps=1e-5):
    """Plain RMSNorm without residual (supports weight=None for weightless RMSNorm)."""
    orig_dtype = hidden.dtype
    x32 = hidden.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    variance = torch.add(variance, eps)
    inv = torch.rsqrt(variance)
    if weight is not None:
        normed = torch.mul(x32, inv).to(weight.dtype)
        out = torch.mul(normed, weight)
    else:
        out = torch.mul(x32, inv).to(orig_dtype)
    return out.to(orig_dtype)


def rms_norm_fused_body(hidden, residual, weight, eps=1e-5):
    """Fused Add + RMSNorm returning (normalized, new_residual)."""
    orig_dtype = hidden.dtype
    x32 = hidden.to(torch.float32) + residual.to(torch.float32)
    new_residual = x32.to(torch.float16)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    variance = torch.add(variance, eps)
    inv = torch.rsqrt(variance)
    normed = torch.mul(x32, inv).to(weight.dtype)
    out = torch.mul(normed, weight)
    return out.to(orig_dtype), new_residual


def layer_norm_standard_body(x, ln_weight, ln_bias, eps=1e-12):
    """Standard LayerNorm."""
    return F.layer_norm(x, (ln_weight.shape[0],), ln_weight, ln_bias, eps=eps)


def linear_matmul_body(x, weight_t):
    """Matrix multiplication x @ weight_t."""
    return torch.matmul(x, weight_t)


def linear_with_bias_body(x, weight, bias):
    """Linear with bias: x @ weight + bias."""
    return torch.add(torch.matmul(x, weight), bias)


def bert_qkv_body(hidden, w_qkv_t, b_qkv):
    """BERT QKV projection with split + contiguous view."""
    qkv = torch.matmul(hidden, w_qkv_t)
    qkv = torch.add(qkv, b_qkv)
    q, k, v = qkv.split([768, 768, 768], dim=-1)
    return tuple(t.contiguous().view(-1, 12, 64) for t in (q, k, v))


def bert_self_output_body(attn_out, residual, w_o_t, b_o, ln_weight, ln_bias):
    """BERT self-output: linear + bias + residual + layer norm."""
    x = torch.matmul(attn_out, w_o_t)
    x = torch.add(x, b_o)
    x = torch.add(x, residual)
    return F.layer_norm(x, (768,), ln_weight, ln_bias, eps=1e-12)


def bert_intermediate_body(x, w_1_t, b_1):
    """BERT intermediate: linear + bias + GELU."""
    x = torch.matmul(x, w_1_t)
    x = torch.add(x, b_1)
    return F.gelu(x, approximate="none")


def bert_output_body(x, attn_output, w_2_t, b_2, ln_weight, ln_bias):
    """BERT output: linear + bias + residual + layer norm."""
    x = torch.matmul(x, w_2_t)
    x = torch.add(x, b_2)
    x = torch.add(x, attn_output)
    return F.layer_norm(x, (768,), ln_weight, ln_bias, eps=1e-12)


def bert_token_classifier_body(hidden, cls_wt, cls_b):
    """BERT token classifier: linear + bias + softmax."""
    logits = torch.matmul(hidden, cls_wt)
    logits = torch.add(logits, cls_b)
    return F.softmax(logits, dim=-1)


def bert_mlm_head_body(x, w_d, b_d, ln_weight, ln_bias, w_dec, b_dec):
    """BERT MLM head: dense + GELU + LN + decoder."""
    x = torch.matmul(x, w_d.transpose(-1, -2))
    x = torch.add(x, b_d)
    x = F.gelu(x, approximate="none")
    x = F.layer_norm(x, (768,), ln_weight, ln_bias, eps=1e-12)
    x = torch.matmul(x, w_dec.transpose(-1, -2))
    return torch.add(x, b_dec)


def roberta_self_output_body(a, res, w, b, lw, lb):
    """RoBERTa self-output: linear + bias + residual + LN."""
    x = torch.matmul(a, w)
    x = torch.add(x, b)
    x = torch.add(x, res)
    return F.layer_norm(x, (1024,), lw, lb, eps=1e-5)


def granite_emb_qkv_heads_body(x, w, b):
    """Granite Embedding QKV projection with reshape + transpose."""
    qkv = torch.matmul(x, w)
    qkv = torch.add(qkv, b)
    return qkv.reshape(-1, 512, 12, 64).transpose(1, 2)


def ministral_qkv_split_body(qkv):
    """Ministral QKV split + contiguous."""
    q, k, v = qkv.split([4096, 1024, 1024], dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def mistral_vision_mask_body(seg):
    """Mistral vision attention mask (section 11.3)."""
    m = torch.matmul(seg, seg.transpose(0, 1))
    mask = (1.0 - m) * (torch.finfo(torch.float16).min / 2)
    return mask.reshape(1, 1, *mask.shape)


def _mistral_rope_rotate_matmul(x, cos, sin, m):
    """Mistral vision RoPE rotation with pair swap matrix m."""
    return x * cos + torch.matmul(x, m) * sin


def mistral_vision_block_body(
    x, seg, cos, sin, attn_norm_w, w_qkv_t, m, w_o_t, ffn_norm_w, w_gate_up_t, w_down_t
):
    """Mistral vision block (section 11.4): x [P, 1024], seg [P, 64], cos / sin [P, 64]."""
    x = x.unsqueeze(0)  # [1, P, 1024]
    mask = mistral_vision_mask_body(seg)  # [1, 1, P, P]
    length = x.shape[1]
    h = rms_norm_plain_body(x, attn_norm_w)
    q, k, v = torch.matmul(h, w_qkv_t).chunk(3, dim=-1)
    q, k, v = (t.reshape(1, length, 16, 64).transpose(1, 2) for t in (q, k, v))
    q = _mistral_rope_rotate_matmul(q, cos, sin, m)
    k = _mistral_rope_rotate_matmul(k, cos, sin, m)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=64.0**-0.5)
    attn = attn.transpose(1, 2).reshape(1, length, 1024)
    h = x + torch.matmul(attn, w_o_t)
    gate_up = torch.matmul(rms_norm_plain_body(h, ffn_norm_w), w_gate_up_t)
    act = F.silu(gate_up[..., :4096]) * gate_up[..., 4096:]
    return h + torch.matmul(act, w_down_t)


def qkv_proj_split_body(x, weight_t, nq=32, nkv=8, hd=128):
    """QKV projection followed by split."""
    qkv = torch.matmul(x, weight_t)
    q_size, kv_size = nq * hd, nkv * hd
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return (
        q.contiguous().view(-1, nq, hd),
        k.contiguous().view(-1, nkv, hd),
        v.contiguous().view(-1, nkv, hd),
    )


def granite_4_1_embed_body(input_ids, weight):
    """Granite 4.1 embedding with 12x multiplier."""
    return embedding_single_lookup_body(input_ids, weight, multiplier=12.0)


def granite_4_1_residual_body(branch, residual):
    """Granite 4.1 residual combine with 0.22 multiplier."""
    return residual_combine_body(branch, residual, multiplier=0.22)


def gemma4_dense_mlp_body(x, w_gu, w_d):
    """Gemma 4 dense MLP: gate-up proj + SiLU-and-mul + down proj."""
    return torch.matmul(silu_and_mul_body(torch.matmul(x, w_gu)), w_d)


def _gemma4_rotary(x, rot):
    """Rotary helper for Gemma 4: x [S, h, hd], rot [S, 2, 2, inner]."""
    x_pairs = x.view(-1, x.shape[1], 1, 2, x.shape[2] // 2)
    return (rot.unsqueeze(1) * x_pairs).sum(dim=-2)


def _gemma4_self_attn(nq, nkv, hd, hidden, positions, k_cache, v_cache, params):
    """Gemma 4 self attention."""
    w_qkv_t, q_w, k_w, rope, slot, mask, w_o_t = params
    q_size, kv_size, inner = nq * hd, nkv * hd, hd // 2
    qkv = torch.matmul(hidden, w_qkv_t)
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = rms_norm_plain_body(q.unflatten(-1, (nq, hd)), q_w, eps=1e-6)
    k = rms_norm_plain_body(k.unflatten(-1, (nkv, hd)), k_w, eps=1e-6)
    rot = rope.index_select(0, positions.flatten()).view(-1, 2, 2, inner)
    q = _gemma4_rotary(q, rot).reshape(-1, nq, hd)
    k = _gemma4_rotary(k, rot).reshape(-1, nkv, hd)
    v = (
        rms_norm_plain_body(v.unflatten(-1, (nkv, hd)), None, eps=1e-6)
        if q_w is not None
        else v
    )
    k_blk, v_blk = k_cache.clone(), v_cache.clone()
    k_blk.index_copy_(2, slot, k.unsqueeze(2))
    v_blk.index_copy_(2, slot, v.unsqueeze(2))
    qg = q.view(-1, nkv, nq // nkv, hd)
    keys_t = k_blk.transpose(-1, -2).contiguous()
    scores = torch.matmul(qg, keys_t) * 1.0 + mask
    block_max = torch.amax(scores, dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - block_max).contiguous()
    denominator = exp_scores.sum(dim=-1, keepdim=True)
    attn = torch.div(torch.matmul(exp_scores, v_blk), denominator)
    return torch.matmul(attn.reshape(-1, q_size), w_o_t)


def gemma4_attention_body(hidden, positions, ks, vs, kf, vf, *params):
    """Gemma 4 attention: sliding (16, 8, 256) and full (16, 2, 512)."""
    sliding = _gemma4_self_attn(16, 8, 256, hidden, positions, ks, vs, params[:7])
    full = _gemma4_self_attn(16, 2, 512, hidden, positions, kf, vf, params[7:])
    return sliding, full


def _gemma4_gathered_token(
    hidden, router_logits, gate, up, down, chunk_scale, chunk_offsets
):
    """Gathered MoE token routing."""
    probs = torch.softmax(router_logits.to(torch.float32), dim=-1).to(
        router_logits.dtype
    )
    weights, indices = torch.topk(probs.expand(2, -1).contiguous(), 8, dim=-1)
    weights, indices = weights[:1], indices[:1]
    weights = weights / weights.sum(-1, keepdim=True)
    rows = hidden.shape[0] * 8
    expert_fp32 = (
        indices.reshape(-1)[:, None].expand(rows, 64).contiguous().to(torch.float32)
    )
    entries = (expert_fp32[:, None, :] * chunk_scale + chunk_offsets).reshape(
        rows * 4, 64
    )
    slice_index = entries.amax(-1).to(torch.int32)
    inputs = (
        hidden.reshape(-1, 1, 4, 704)
        .expand(-1, 8, 4, 704)
        .contiguous()
        .reshape(rows * 4, 1, 704)
    )
    gate_out = (
        torch.bmm(inputs, gate[slice_index].reshape(rows * 4, 704, 704))
        .reshape(rows, 4, 1, 704)
        .sum(dim=1)
    )
    up_out = (
        torch.bmm(inputs, up[slice_index].reshape(rows * 4, 704, 704))
        .reshape(rows, 4, 1, 704)
        .sum(dim=1)
    )
    act = F.gelu(gate_out, approximate="tanh") * up_out
    act_rep = act[:, None].expand(rows, 4, 1, 704).reshape(rows * 4, 1, 704)
    parts = torch.bmm(act_rep, down[slice_index].reshape(rows * 4, 704, 704)).reshape(
        -1, 8, 2816
    )
    return (parts * weights[..., None]).sum(dim=1)


def gemma4_moe_body(hidden, router_logits, gate, up, down, chunk_scale, chunk_offsets):
    """Gemma 4 MoE gathered experts body."""
    rows = [
        _gemma4_gathered_token(
            hidden[t : t + 1],
            router_logits[t : t + 1],
            gate,
            up,
            down,
            chunk_scale,
            chunk_offsets,
        )
        for t in range(hidden.shape[0])
    ]
    return torch.cat(rows)


def gemma4_decoder_layer_body(hidden, positions, k_cache, v_cache, *p):
    """Gemma 4 17-step decoder layer body."""
    in_w = p[0]
    attn = p[1:8]
    post_attn_w, pre_ff_w, w_gate_up_t, w_down_t, post_ff1_w, pre_ff2_w = p[8:14]
    r_scale, w_proj_t, gate, up, down, post_ff2_w, post_ff_w, layer_scalar = p[14:22]
    chunk_scale, chunk_offsets = p[22:]
    residual = hidden
    h = rms_norm_plain_body(residual, in_w, eps=1e-6)
    h = _gemma4_self_attn(16, 8, 256, h, positions, k_cache, v_cache, attn)
    h = rms_norm_plain_body(h, post_attn_w, eps=1e-6)
    h = h + residual
    residual = h
    h1 = rms_norm_plain_body(h, pre_ff_w, eps=1e-6)
    h1 = gemma4_dense_mlp_body(h1, w_gate_up_t, w_down_t)
    h1 = rms_norm_plain_body(h1, post_ff1_w, eps=1e-6)
    h2 = rms_norm_plain_body(residual, pre_ff2_w, eps=1e-6)
    r_x = rms_norm_plain_body(residual, None, eps=1e-6) * (2816.0**-0.5) * r_scale
    router_logits = torch.matmul(r_x, w_proj_t)
    h2 = gemma4_moe_body(h2, router_logits, gate, up, down, chunk_scale, chunk_offsets)
    h2 = rms_norm_plain_body(h2, post_ff2_w, eps=1e-6)
    h = h1 + h2
    h = rms_norm_plain_body(h, post_ff_w, eps=1e-6)
    h = h + residual
    return h * layer_scalar


def gemma4_vision_pool_proj_body(x):
    """Gemma 4 vision pool projection: reshape + sum."""
    return x.reshape(9, -1, 1152).sum(0)


def granite_vision_siglip_encoder_body(
    hidden_states,
    ln1_w,
    ln1_b,
    w_qkv_t,
    b_qkv,
    attn_mask,
    w_out_t,
    b_out,
    ln2_w,
    ln2_b,
    w_fc1_t,
    b_fc1,
    w_fc2_t,
    b_fc2,
):
    """SigLIP encoder layer with LayerNorm, QKV attention, MLP and residuals."""
    b_tile = hidden_states.shape[0]
    residual = hidden_states
    ln1 = F.layer_norm(
        hidden_states, (1152,), ln1_w.flatten(), ln1_b.flatten(), eps=1e-6
    )
    qkv = torch.matmul(ln1, w_qkv_t) + b_qkv
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(b_tile, 729, 16, 72).transpose(1, 2)
    k = k.view(b_tile, 729, 16, 72).transpose(1, 2)
    v = v.view(b_tile, 729, 16, 72).transpose(1, 2)
    q = F.pad(q, (0, 56, 0, 39))
    k = F.pad(k, (0, 56, 0, 39))
    v = F.pad(v, (0, 56, 0, 39))
    q_scaled = q * (72.0**-0.5)
    scores = torch.matmul(q_scaled, k.transpose(-1, -2)) + attn_mask
    block_max = torch.amax(scores, dim=-1)
    exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
    denominator = exp_scores.sum(dim=-1)
    attn_out = torch.matmul(exp_scores, v) / denominator.unsqueeze(-1)
    attn_crop = attn_out[:, :, :729, :72]
    attn_flat = attn_crop.transpose(1, 2).reshape(b_tile, 729, 1152)
    out_proj = torch.matmul(attn_flat, w_out_t) + b_out
    hidden_states = residual + out_proj
    residual2 = hidden_states
    ln2 = F.layer_norm(
        hidden_states, (1152,), ln2_w.flatten(), ln2_b.flatten(), eps=1e-6
    )
    fc1 = torch.matmul(ln2, w_fc1_t) + b_fc1
    gelu = F.gelu(fc1, approximate="tanh")
    fc2 = torch.matmul(gelu, w_fc2_t) + b_fc2
    return residual2 + fc2


def granite_vision_llava_projector_body(x, w1, b1, w2, b2):
    """LLaVA multimodal projector: gelu linear + linear."""
    return linear_with_bias_body(F.gelu(linear_with_bias_body(x, w1, b1)), w2, b2)


def granite_vision_merge_patch_body(p, nl):
    """Granite Vision merge patch embeddings."""
    return torch.cat([p, nl.expand(p.shape[0], -1, -1)], dim=1)


def granite_vision_splice_embeddings_body(inp, mm, m):
    """Granite Vision splice embeddings: torch.where."""
    return torch.where(m, mm, inp)


def yolo_conv_body(x, w, bn_w, bn_b, rm, rv):
    """YOLO Conv: conv2d + batch_norm + silu."""
    return F.silu(
        F.batch_norm(
            F.conv2d(x, w, None, stride=2, padding=1),
            rm,
            rv,
            bn_w,
            bn_b,
            training=False,
            eps=1e-3,
        )
    )


def yolo_conv2d_body(x, w):
    """YOLO Conv2d: conv2d only."""
    return F.conv2d(x, w, None, stride=2, padding=1)


def yolo_batchnorm_body(x, bn_w, bn_b, rm, rv):
    """YOLO BatchNorm2d."""
    return F.batch_norm(x, rm, rv, bn_w, bn_b, training=False, eps=1e-3)


def yolo_silu_body(x):
    """YOLO SiLU activation."""
    return F.silu(x)


def yolo_c3k2_body(x, *params):
    """YOLO C3k2 block."""
    return yolo_c3k2(YOLOParams(params), x, 64, 0.25, False)


def yolo_bottleneck_body(x, *params):
    """YOLO Bottleneck."""
    return yolo_bottleneck(YOLOParams(params), x, 16)


def yolo_c3k_body(x, *params):
    """YOLO C3k block."""
    return yolo_c3k(YOLOParams(params), x, 64)


def yolo_sequential_body(x, *params):
    """YOLO Sequential (two Bottleneck ops)."""
    P = YOLOParams(params)
    return yolo_bottleneck(P, yolo_bottleneck(P, x, 32, e=1.0), 32, e=1.0)


def yolo_sppf_body(x, *params):
    """YOLO SPPF."""
    return yolo_sppf(YOLOParams(params), x)


def yolo_identity_body(x):
    """YOLO Identity."""
    return x.clone()


def yolo_maxpool_body(x):
    """YOLO MaxPool2d (3x cascaded)."""
    return F.max_pool2d(F.max_pool2d(F.max_pool2d(x, 5, 1, 2), 5, 1, 2), 5, 1, 2)


def yolo_upsample_body(x):
    """YOLO Upsample."""
    return F.interpolate(x, scale_factor=2.0, mode="nearest")


def yolo_concat_body(a, b):
    """YOLO Concat along dim=1."""
    return torch.cat((a, b), dim=1)


def _rotate(x, rot):
    """Rotary embedding rotation helper: x [T, h * hd], rot [T, 2, 2, hd]."""
    x_pairs = x.view(x.shape[0], -1, 2, rot.shape[-1])  # [T, h, 2, hd]
    rot = rot.unsqueeze(1)  # [T, 1, 2, 2, hd]
    x_pairs = x_pairs.unsqueeze(-3)  # [T, h, 1, 2, hd]
    prod = torch.mul(rot, x_pairs)  # [T, h, 2, 2, hd]
    out = prod.sum(dim=-2)  # [T, h, 2, hd]
    out = out.flatten(start_dim=-2)  # [T, h, hd * 2]
    return out.view(x.shape)


def rotary_emb_body(positions, query, key, cache):
    """RoPE rotation on query and key tensors using 2x2 rotation matrix."""
    idx = positions[:, 0].flatten()  # [T]
    rot = cache.index_select(0, idx)  # [T, 2 * 2 * hd]
    rot = rot.view(-1, 2, 2, rot.shape[-1] // 4)  # [T, 2, 2, hd]
    return _rotate(query, rot), _rotate(key, rot)


def kv_cache_write_body(carry, key, value, base):
    """Paged KV cache write update."""
    base_idx = base[:, 0]
    k_rows, v_rows = (t.clone() for t in carry)
    key = key * 1.0
    value = value * 1.0
    nkv = key.shape[1]
    block = 128
    for h in range(nkv):
        k_rows.index_copy_(0, base_idx + h * block, key[:, h])
    for h in range(nkv):
        v_rows.index_copy_(0, base_idx + h * block, value[:, h])
    return k_rows, v_rows


def attn_prefill_body(
    query, query_row_index, mask_tile, k_pages, v_pages, page_ids, scale=0.0078125
):
    """Paged attention prefill kernel body."""
    nq, hd = query.shape[1], query.shape[2]
    q_scaled = query * scale
    keys_t = k_pages.transpose(-1, -2).contiguous()
    scores = torch.matmul(q_scaled.view(-1, nq, hd), keys_t) + mask_tile
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_pages)
    return out.view(-1, nq * hd)


def attn_decode_single_body(
    q_in, query_row_index, page_ids, mask, k_pages, v_pages, scale=0.0884
):
    """Single sequence decode attention body."""
    nq, hd = q_in.shape[1], q_in.shape[2]
    q_scaled = q_in * scale
    keys_t = k_pages.transpose(-1, -2).contiguous()
    scores = torch.matmul(q_scaled, keys_t) + mask
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_pages)
    return out.view(-1, nq * hd)


def attn_batched_decode_body(
    q_in, rep_row_ids, page_ids, mask_rows, k_pages, v_pages, scale=0.0884
):
    """Batched decode attention body."""
    nq, hd = q_in.shape[1], q_in.shape[2]
    q_scaled = q_in * scale
    keys_t = k_pages.transpose(-1, -2).contiguous()
    scores = torch.matmul(q_scaled, keys_t) + mask_rows
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_pages)
    return out.view(-1, nq * hd)


def sdpa_rect_body(q, k, v, mask, scale=0.125, lx=512):
    """Rectangular scaled dot-product attention."""
    nh, d = q.shape[1], q.shape[2]
    q, k, v = (t.view(-1, lx, nh, d).transpose(1, 2) for t in (q, k, v))
    attn = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, is_causal=False
    )
    return attn.transpose(1, 2).reshape(-1, nh, d)


def sdpa_decomposed_body(q, k, v, mask, scale=0.125):
    """Decomposed SDPA op sequence."""
    q_scaled = q * scale
    keys_t = k.transpose(-1, -2).contiguous()
    scores = torch.matmul(q_scaled, keys_t) + mask
    block_max = torch.amax(scores, dim=-1)
    scores = torch.sub(scores, block_max.unsqueeze(-1))
    exp_scores = torch.exp(scores).contiguous()
    denominator = exp_scores.sum(dim=-1)
    output_tile = torch.matmul(exp_scores, v)
    return torch.div(output_tile, denominator.unsqueeze(-1))


def silu_and_mul_body(gate_up):
    """SwiGLU silu and mul activation: silu(gate) * up."""
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def residual_combine_body(branch, residual, multiplier=1.0):
    """Residual combination: residual + branch * multiplier."""
    if multiplier != 1.0:
        return residual + branch * multiplier
    return residual + branch


def cls_pool_body(idx, hidden):
    """Extract CLS token embedding: torch.index_select(hidden, 0, idx[:, 0])."""
    idx = idx[:, 0]
    return torch.index_select(hidden, 0, idx)


def l2_normalize_body(pooled, eps=1e-12):
    """L2 normalization along last dimension."""
    norm = torch.sqrt(torch.sum(pooled.pow(2), dim=-1, keepdim=True) + eps)
    return pooled / norm


def mean_pool_body(hidden, row_mask, lens):
    """Mean pooling across sequence tokens."""
    masked = hidden * row_mask
    summed = torch.sum(masked, dim=0, keepdim=True)
    return summed / torch.clamp(lens, min=1.0)


def fp8_linear_body(x, weight, weight_scale):
    """FP8 quantized linear matrix multiplication."""
    x_max = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    x_scale = torch.clamp(x_max / FP8_MAX, min=SCALE_MIN)
    x_fp8 = torch.clamp(x / x_scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    w_scale = weight_scale.unsqueeze(0)
    out_f32 = torch.matmul(x_fp8.to(torch.float32), weight.to(torch.float32))
    return (out_f32 * (x_scale * w_scale)).to(torch.float16)


# ---------------------------------------------------------------------------
# Case Specification Dataclass & Runtime Helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    name: str
    model: str  # "common" when the test is shared by several models (see note)
    tile: int
    lo: int
    hi: int
    body: Callable
    make_inputs: Callable
    n_dyn: int = 1
    tol: float = LOOSE
    n_carry: int = 0
    out_dim: int = 0
    chunked_reference: bool = False
    ref_native_dtype: bool = False
    cpu_body: Optional[Callable] = None
    xfail_reason: Optional[
        str | tuple[str, type[BaseException] | tuple[type[BaseException], ...]]
    ] = None
    xfail_raises: type[BaseException] | tuple[type[BaseException], ...] = Exception
    note: str = ""  # models / modules this single test stands in for


def runtime_sizes(lo: int, hi: int, tile: int) -> list[int]:
    """Return seven runtime sizes: min, max, and five intermediate multiples."""
    steps = (hi - lo) // tile
    assert steps >= 6, f"[{lo}, {hi}] step {tile} has no room for five intermediates"
    mids = [lo + tile * round(steps * k / 6) for k in range(1, 6)]
    assert len(set(mids)) == 5 and lo < mids[0] and mids[-1] < hi, mids
    return [mids[2], lo, hi, mids[0], mids[4], mids[1], mids[3]]


def traced_module(case: Case):
    """Build the compiled for_each_tile traced module function."""
    n_dyn = case.n_dyn
    lo, hi, tile, module_body = case.lo, case.hi, case.tile, case.body
    n_carry = case.n_carry

    def traced(*args):
        dyn, static = args[:n_dyn], args[n_dyn:]
        carry0, invariants = static[:n_carry], static[n_carry:]
        rows = dyn[0].shape[0]
        for other in dyn[1:]:
            torch._check(other.shape[0] == rows)
        torch._check(rows >= lo)
        torch._check(rows <= hi)
        torch._check(rows % tile == 0)
        dims = (0,) * n_dyn + (None,) * len(invariants)

        if n_carry > 0:

            def reduce_body(carry, tiles):
                return module_body(carry, *tiles), None

            carry, _out = for_each_tile(
                reduce_body,
                (*dyn, *invariants),
                dims=dims,
                tile_size=tile,
                init=tuple(carry0),
            )
            return carry
        else:

            def body(_carry, tiles):
                return None, module_body(*tiles)

            _carry, out = for_each_tile(
                body,
                (*dyn, *invariants),
                dims=dims,
                tile_size=tile,
                out_dim=case.out_dim,
            )
            return out

    return traced


def cpu_reference(case: Case):
    """Build the CPU reference evaluator."""
    n_dyn = case.n_dyn
    n_carry = case.n_carry
    ref_fn = case.cpu_body if case.cpu_body is not None else case.body

    def reference(*args):
        dyn, static = args[:n_dyn], args[n_dyn:]
        carry0, invariants = static[:n_carry], static[n_carry:]
        rows = dyn[0].shape[0]

        if n_carry > 0:
            carry = carry0
            for i in range(0, rows, case.tile):
                tiles = tuple(t[i : i + case.tile] for t in dyn) + tuple(invariants)
                res = ref_fn(carry, *tiles)
                carry = res if isinstance(res, (tuple, list)) else (res,)
            return _to_cpu_f32(carry)

        if not case.chunked_reference:
            return _to_cpu_f32(ref_fn(*args))

        parts = [
            _to_cpu_f32(ref_fn(*(t[i : i + case.tile] for t in dyn), *invariants))
            for i in range(0, rows, case.tile)
        ]
        if isinstance(parts[0], tuple):
            return tuple(
                torch.cat([p[k] for p in parts], dim=case.out_dim)
                for k in range(len(parts[0]))
            )
        return torch.cat(parts, dim=case.out_dim)

    return reference


def _to_cpu_f32(out):
    """Transfer device output to fp32 on CPU."""
    if isinstance(out, (tuple, list)):
        return tuple(_to_cpu_f32(o) for o in out)
    if isinstance(out, torch.Tensor):
        return out.cpu().float()
    return out


# ---------------------------------------------------------------------------
# Input Generators across Models
# ---------------------------------------------------------------------------
def _w(shape, tag, scale=0.02, dtype=torch.float16):
    return cached_randn(shape, differentiation=tag, scale=scale, dtype=dtype)


def _x(shape, tag, scale=1.0, dtype=torch.float16):
    return cached_randn(shape, differentiation=tag, scale=scale, dtype=dtype)


def _make_emb_3table_inputs(vocab, h, max_pos=512, seg_rows=64, tag="emb"):
    def make(rows):
        ids = cached_randint((rows, 1), f"{tag}_ids", 0, vocab, torch.int64)
        pos = (torch.arange(rows, dtype=torch.int64) % max_pos).unsqueeze(-1)
        ln_w = 1.0 + cached_randn((h,), differentiation=f"{tag}_ln_w", scale=0.1)
        ln_b = cached_randn((h,), differentiation=f"{tag}_ln_b", scale=0.1)
        static = (
            _w((vocab, h), f"{tag}_word"),
            _w((seg_rows, h), f"{tag}_seg"),
            _w((max_pos, h), f"{tag}_pos"),
            ln_w.to(torch.float16),
            ln_b.to(torch.float16),
        )
        return (ids, pos), static

    return make


def _make_single_emb_inputs(vocab, h, tag="emb_single", multiplier=1.0):
    def make(rows):
        ids = cached_randint((rows, 1), f"{tag}_ids", 0, vocab, torch.int64)
        weight = _w((vocab, h), f"{tag}_table")
        return (ids,), (weight,)

    return make


def _make_norm_plain_inputs(h, tag="norm_p"):
    def make(rows):
        weight = (1.0 + cached_randn((1, h), differentiation=f"{tag}_w", scale=0.1)).to(
            torch.float16
        )
        return (_x((rows, h), f"{tag}_x"),), (weight,)

    return make


def _make_norm_fused_inputs(h, tag="norm_f"):
    def make(rows):
        weight = (1.0 + cached_randn((1, h), differentiation=f"{tag}_w", scale=0.1)).to(
            torch.float16
        )
        return (_x((rows, h), f"{tag}_x"), _x((rows, h), f"{tag}_res")), (weight,)

    return make


def _make_qkv_proj_inputs(h, qkv, nq=32, nkv=8, hd=128, tag="qkv"):
    def make(rows):
        return (_x((rows, h), f"{tag}_x"),), (_w((h, qkv), f"{tag}_w"),)

    return make


def _make_rotary_inputs(nq, nkv, hd, max_pos=131072, tag="rot"):
    # cache shape: [max_pos, 2*2*(hd//2)] = [max_pos, 2*hd] when hd=128 -> [max_pos, 256]
    # query/key are flattened: [T, nq*hd] and [T, nkv*hd]  (2-D, not 3-D)
    def make(rows):
        pos = (torch.arange(rows, dtype=torch.int64) % max_pos).unsqueeze(-1)  # [T, 1]
        q = _x((rows, nq * hd), f"{tag}_q")
        k = _x((rows, nkv * hd), f"{tag}_k")
        cache = _w((max_pos, 2 * 2 * (hd // 2)), f"{tag}_cache", scale=0.5)
        return (pos, q, k), (cache,)

    return make


def _make_kv_write_inputs(nkv, hd, num_pages=8, block=128, tag="kv_write"):
    def make(rows):
        blocks = max(rows // block, num_pages)
        rows_per_block = nkv * block
        table_size = (blocks * rows_per_block, hd)
        t = torch.arange(rows, dtype=torch.int64)
        base = ((t // block) * rows_per_block + t % block).unsqueeze(-1)
        key = _x((rows, nkv, hd), f"{tag}_k")
        val = _x((rows, nkv, hd), f"{tag}_v")
        k_pages = _x(table_size, f"{tag}_kp", scale=0.1)
        v_pages = _x(table_size, f"{tag}_vp", scale=0.1)
        return (key, val, base), (k_pages, v_pages)

    return make


def _make_attn_prefill_inputs(nq, hd, ctx=512, tag="attn_pref"):
    def make(rows):
        q = _x((rows, nq, hd), f"{tag}_q")
        q_idx = (torch.arange(rows, dtype=torch.int64) % ctx).unsqueeze(-1)
        mask = torch.zeros(rows, 1, 1, ctx, dtype=torch.float16)
        k_pages = _w((ctx, hd), f"{tag}_kp")
        v_pages = _w((ctx, hd), f"{tag}_vp")
        page_ids = torch.arange(ctx // 128, dtype=torch.int64)
        return (q, q_idx, mask), (k_pages, v_pages, page_ids)

    return make


def _make_attn_decode_single_inputs(nq, hd, ctx=512, tag="attn_dec1"):
    def make(rows):
        q = _x((rows, nq, hd), f"{tag}_q")
        q_idx = torch.zeros((rows, 1), dtype=torch.int64)
        page_ids = torch.zeros((rows, 1), dtype=torch.int64)
        mask = torch.zeros(rows, 1, 1, ctx, dtype=torch.float16)
        k_pages = _w((ctx, hd), f"{tag}_kp")
        v_pages = _w((ctx, hd), f"{tag}_vp")
        return (q, q_idx, page_ids, mask), (k_pages, v_pages)

    return make


def _make_attn_batched_decode_inputs(nq, hd, chunks=4, block=128, tag="attn_dec_b"):
    def make(rows):
        q = _x((rows, nq, hd), f"{tag}_q")
        rep_ids = torch.zeros((rows, 1), dtype=torch.int64)
        page_ids = torch.zeros((rows, chunks), dtype=torch.int64)
        mask = torch.zeros(rows, 1, 1, chunks * block, dtype=torch.float16)
        k_pages = _w((chunks * block, hd), f"{tag}_kp")
        v_pages = _w((chunks * block, hd), f"{tag}_vp")
        return (q, rep_ids, page_ids, mask), (k_pages, v_pages)

    return make


def _make_linear_inputs(in_dim, out_dim, tag="linear"):
    def make(rows):
        return (_x((rows, in_dim), f"{tag}_x"),), (_w((in_dim, out_dim), f"{tag}_w"),)

    return make


def _make_silu_inputs(dim, tag="silu"):
    def make(rows):
        return (_x((rows, 2 * dim), f"{tag}_gu"),), ()

    return make


def _make_residual_inputs(dim, tag="res"):
    def make(rows):
        return (
            _x((rows, dim), f"{tag}_branch"),
            _x((rows, dim), f"{tag}_residual"),
        ), ()

    return make


def _make_fp8_linear_inputs(k, n, tag="fp8_lin"):
    def make(rows):
        x = _x((rows, k), f"{tag}_x")
        w_fp8 = _w((k, n), f"{tag}_w").to(FP8_DTYPE)
        scale = torch.ones(n, dtype=torch.float32)
        return (x,), (w_fp8, scale)

    return make


def _make_gemma4_attn_inputs(rows):
    ks = _x((rows, 8, 128, 256), "g4_att_ks")
    vs = _x((rows, 8, 128, 256), "g4_att_vs")
    kf = _x((rows, 2, 128, 512), "g4_att_kf")
    vf = _x((rows, 2, 128, 512), "g4_att_vf")
    pos = (torch.arange(rows, dtype=torch.int64) % 2048).unsqueeze(-1)
    h = _x((rows, 2816), "g4_att_h")
    w_qkv_s = _w((2816, (16 + 2 * 8) * 256), "g4_att_qkvs")
    qw_s = (1.0 + cached_randn((256,), "g4_att_qws", scale=0.1) * 0.25).to(
        torch.float16
    )
    kw_s = (1.0 + cached_randn((256,), "g4_att_kws", scale=0.1) * 0.25).to(
        torch.float16
    )
    rope_s = _w((2048, 4 * 128), "g4_att_rope_s")
    slot = torch.tensor([5], dtype=torch.int64)
    mask = torch.zeros(1, 1, 1, 128, dtype=torch.float16)
    wo_s = _w((16 * 256, 2816), "g4_att_wos")
    w_qkv_f = _w((2816, (16 + 2 * 2) * 512), "g4_att_qkvf")
    qw_f = (1.0 + cached_randn((512,), "g4_att_qwf", scale=0.1) * 0.25).to(
        torch.float16
    )
    kw_f = (1.0 + cached_randn((512,), "g4_att_kwf", scale=0.1) * 0.25).to(
        torch.float16
    )
    rope_f = _w((2048, 4 * 256), "g4_att_rope_f")
    wo_f = _w((16 * 512, 2816), "g4_att_wof")
    static = (
        w_qkv_s,
        qw_s,
        kw_s,
        rope_s,
        slot,
        mask,
        wo_s,
        w_qkv_f,
        qw_f,
        kw_f,
        rope_f,
        slot,
        mask,
        wo_f,
    )
    return (h, pos, ks, vs, kf, vf), static


def _make_gemma4_moe_inputs(rows):
    h = _x((rows, 2816), "g4_moe_h")
    r_logits = _x((rows, 128), "g4_moe_r")
    gate = _w((512, 704, 704), "g4_moe_gate")
    up = _w((512, 704, 704), "g4_moe_up")
    down = _w((512, 704, 704), "g4_moe_down")
    c_scale = torch.full((1, 1, 1), 4.0, dtype=torch.float32)
    c_offsets = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    return (h, r_logits), (gate, up, down, c_scale, c_offsets)


def _make_siglip_encoder_inputs(rows):
    hidden_states = _x((rows, 729, 1152), "siglip_hs")
    ln1_w = (1.0 + cached_randn((1, 1152), "siglip_ln1_w", scale=0.1)).to(torch.float16)
    ln1_b = cached_randn((1, 1152), "siglip_ln1_b", scale=0.01).to(torch.float16)
    w_qkv_t = _w((1152, 3 * 1152), "siglip_wqkv")
    b_qkv = cached_randn((1, 3 * 1152), "siglip_bqkv", scale=0.01).to(torch.float16)
    attn_mask = torch.zeros((1, 1, 1, 768), dtype=torch.float16)
    attn_mask[..., 729:] = torch.finfo(torch.float16).min / 2
    w_out_t = _w((1152, 1152), "siglip_wout")
    b_out = cached_randn((1, 1152), "siglip_bout", scale=0.01).to(torch.float16)
    ln2_w = (1.0 + cached_randn((1, 1152), "siglip_ln2_w", scale=0.1)).to(torch.float16)
    ln2_b = cached_randn((1, 1152), "siglip_ln2_b", scale=0.01).to(torch.float16)
    w_fc1_t = _w((1152, 4304), "siglip_wfc1")
    b_fc1 = cached_randn((1, 4304), "siglip_bfc1", scale=0.01).to(torch.float16)
    w_fc2_t = _w((4304, 1152), "siglip_wfc2")
    b_fc2 = cached_randn((1, 1152), "siglip_bfc2", scale=0.01).to(torch.float16)
    static = (
        ln1_w,
        ln1_b,
        w_qkv_t,
        b_qkv,
        attn_mask,
        w_out_t,
        b_out,
        ln2_w,
        ln2_b,
        w_fc1_t,
        b_fc1,
        w_fc2_t,
        b_fc2,
    )
    return (hidden_states,), static


def _make_gemma4_decoder_layer_inputs(rows):
    h = _x((rows, 2816), "g4_dec_h", scale=1.0) + 2.0
    pos = (torch.arange(rows, dtype=torch.int64) % 2048).unsqueeze(-1)
    ks = _x((rows, 8, 128, 256), "g4_dec_ks")
    vs = _x((rows, 8, 128, 256), "g4_dec_vs")
    in_w = (1.0 + cached_randn((2816,), "g4_dec_in_w", scale=0.1)).to(torch.float16)
    w_qkv_s = _w((2816, (16 + 2 * 8) * 256), "g4_dec_qkvs")
    qw_s = (1.0 + cached_randn((256,), "g4_dec_qws", scale=0.1) * 0.25).to(
        torch.float16
    )
    kw_s = (1.0 + cached_randn((256,), "g4_dec_kws", scale=0.1) * 0.25).to(
        torch.float16
    )
    rope_s = _w((2048, 4 * 128), "g4_dec_rope_s")
    slot = torch.tensor([5], dtype=torch.int64)
    mask = torch.zeros(1, 1, 1, 128, dtype=torch.float16)
    wo_s = _w((16 * 256, 2816), "g4_dec_wos")
    post_attn_w = (1.0 + cached_randn((2816,), "g4_dec_paw", scale=0.1)).to(
        torch.float16
    )
    pre_ff_w = (1.0 + cached_randn((2816,), "g4_dec_pffw", scale=0.1)).to(torch.float16)
    w_gu = _w((2816, 4224), "g4_dec_wgu")
    w_d = _w((2112, 2816), "g4_dec_wd")
    post_ff1_w = (1.0 + cached_randn((2816,), "g4_dec_pff1w", scale=0.1)).to(
        torch.float16
    )
    pre_ff2_w = (1.0 + cached_randn((2816,), "g4_dec_pff2w", scale=0.1)).to(
        torch.float16
    )
    r_scale = (1.0 + cached_randn((2816,), "g4_dec_rscale", scale=0.1)).to(
        torch.float16
    )
    w_proj_t = _w((2816, 128), "g4_dec_wprojt")
    gate = _w((512, 704, 704), "g4_dec_gate")
    up = _w((512, 704, 704), "g4_dec_up")
    down = _w((512, 704, 704), "g4_dec_down")
    post_ff2_w = (1.0 + cached_randn((2816,), "g4_dec_pff2w_out", scale=0.1)).to(
        torch.float16
    )
    post_ff_w = (1.0 + cached_randn((2816,), "g4_dec_pffw_out", scale=0.1)).to(
        torch.float16
    )
    layer_scalar = torch.tensor([0.8], dtype=torch.float16)
    c_scale = torch.full((1, 1, 1), 4.0, dtype=torch.float32)
    c_offsets = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    static = (
        in_w,
        w_qkv_s,
        qw_s,
        kw_s,
        rope_s,
        slot,
        mask,
        wo_s,
        post_attn_w,
        pre_ff_w,
        w_gu,
        w_d,
        post_ff1_w,
        pre_ff2_w,
        r_scale,
        w_proj_t,
        gate,
        up,
        down,
        post_ff2_w,
        post_ff_w,
        layer_scalar,
        c_scale,
        c_offsets,
    )
    return (h, pos, ks, vs), static


def _segments(rows):
    """One-hot [P, 64] segment ids."""
    ids = (torch.arange(rows, dtype=torch.int64) // 32) % 64
    return F.one_hot(ids, 64).to(torch.float16)


def _vision_rope_tables(vhd=64, max_pos=12100):
    """cos / sin tables [max_pos, vhd] fp16."""
    inv_freq = 10000.0 ** (-torch.arange(vhd, dtype=torch.float32) / vhd)
    angle = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv_freq
    return angle.cos().to(torch.float16), angle.sin().to(torch.float16)


def _pair_permutation(vhd=64):
    """Pair swap permutation matrix m [vhd, vhd]."""
    m = torch.zeros(vhd, vhd)
    even = torch.arange(0, vhd, 2)
    m[even, even + 1] = 1.0
    m[even + 1, even] = 1.0
    return m.to(torch.float16)


def _make_mistral_vision_block_inputs(rows):
    cos_table, sin_table = _vision_rope_tables()
    positions = (torch.arange(rows, dtype=torch.int64) * 37) % 12100
    dyn = (
        _x((rows, 1024), "mvb_x"),
        _segments(rows),
        cos_table.index_select(0, positions),
        sin_table.index_select(0, positions),
    )
    static = (
        (1.0 + cached_randn((1, 1024), "mvb_attn_norm", scale=0.1)).to(torch.float16),
        _w((1024, 3072), "mvb_wqkv"),
        _pair_permutation(),
        _w((1024, 1024), "mvb_wo"),
        (1.0 + cached_randn((1, 1024), "mvb_ffn_norm", scale=0.1)).to(torch.float16),
        _w((1024, 8192), "mvb_wgu"),
        _w((4096, 1024), "mvb_wd"),
    )
    return dyn, static


# ---------------------------------------------------------------------------
# YOLO26 Param Helper Class & Bodies
# ---------------------------------------------------------------------------
class YOLOParams:
    def __init__(self, tensors=None, tag_prefix="yolo"):
        self._src = None if tensors is None else iter(tensors)
        self.made: list[torch.Tensor] = []
        self._tag = tag_prefix

    def _take(self, shape, scale=1.0, shift=0.0, absolute=False):
        if self._src is not None:
            return next(self._src)
        tag = (self._tag, len(self.made))
        t = cached_randn(
            tuple(shape),
            differentiation=tag,
            abs=absolute,
            dtype=torch.float32,
            scale=scale,
        )
        t = t + shift if shift else t
        self.made.append(t)
        return t

    def conv_w(self, c_out, c_in_per_group, k):
        fan_in = c_in_per_group * k * k
        return self._take((c_out, c_in_per_group, k, k), scale=fan_in**-0.5)

    def conv_b(self, c_out):
        return self._take((c_out,), scale=0.1)

    def bn(self, c):
        mean = self._take((c,), scale=0.1)
        var = self._take((c,), scale=0.1, shift=1.0, absolute=True)
        weight = self._take((c,), scale=0.1, shift=1.0)
        bias = self._take((c,), scale=0.1)
        return weight, bias, mean, var


def yolo_conv(P, x, c2, k=1, s=1, g=1, act=True):
    c1 = x.shape[1]
    w = P.conv_w(c2, c1 // g, k)
    x = F.conv2d(x, w, None, stride=s, padding=k // 2, groups=g)
    bn_w, bn_b, rm, rv = P.bn(c2)
    x = F.batch_norm(x, rm, rv, bn_w, bn_b, training=False, eps=1e-3)
    return F.silu(x) if act else x


def yolo_bottleneck(P, x, c2, e=0.5, shortcut=True):
    c_ = int(c2 * e)
    y = yolo_conv(P, yolo_conv(P, x, c_, 3), c2, 3)
    return x + y if shortcut and x.shape[1] == c2 else y


def yolo_c3k(P, x, c2, n=2, e=0.5):
    c_ = int(c2 * e)
    x1 = yolo_conv(P, x, c_, 1)
    x2 = yolo_conv(P, x, c_, 1)
    for _ in range(n):
        x2 = yolo_bottleneck(P, x2, c_, e=1.0, shortcut=True)
    return yolo_conv(P, torch.cat((x2, x1), 1), c2, 1)


def yolo_c3k2(P, x, c2, e=0.25, use_c3k=False):
    c_ = int(c2 * e * 2)
    x1 = yolo_conv(P, x, c_, 1)
    x2 = yolo_conv(P, x, c_, 1)
    for _ in range(2):
        x2 = yolo_c3k(P, x2, c_) if use_c3k else yolo_bottleneck(P, x2, c_, e=1.0)
    return yolo_conv(P, torch.cat((x2, x1), 1), c2, 1)


def yolo_sppf(P, x, c2=256, k=5):
    c1 = x.shape[1]
    c_ = c1 // 2
    x = yolo_conv(P, x, c_, 1)

    def m(t):
        return F.max_pool2d(t, k, stride=1, padding=k // 2)

    y1 = m(x)
    y2 = m(y1)
    y3 = m(y2)
    return yolo_conv(P, torch.cat((x, y1, y2, y3), 1), c2, 1)


@functools.lru_cache(maxsize=None)
def _yolo_weights(fn_name: str) -> tuple[torch.Tensor, ...]:
    """Extract flattened static weight tensors for a YOLO module via a recording pass."""
    P = YOLOParams(tag_prefix=f"yolo_rec_{fn_name}")
    if fn_name == "c3k2":
        yolo_c3k2(P, torch.zeros(1, 32, 64, 64, dtype=torch.float32), 64, 0.25, False)
    elif fn_name == "bottleneck":
        yolo_bottleneck(P, torch.zeros(1, 16, 64, 64, dtype=torch.float32), 16)
    elif fn_name == "c3k":
        yolo_c3k(P, torch.zeros(1, 64, 16, 16, dtype=torch.float32), 64)
    elif fn_name == "sequential":
        yolo_bottleneck(
            P,
            yolo_bottleneck(
                P, torch.zeros(1, 32, 16, 16, dtype=torch.float32), 32, e=1.0
            ),
            32,
            e=1.0,
        )
    elif fn_name == "sppf":
        yolo_sppf(P, torch.zeros(1, 256, 8, 8, dtype=torch.float32))
    return tuple(P.made)


# ---------------------------------------------------------------------------
# Consolidated Parametrized Test Cases Table with Full XFAIL Mapping
# ---------------------------------------------------------------------------

CASES: tuple[Case, ...] = (
    # =======================================================================
    # 1. BERT & RoBERTa & Granite Embedding (3-Table Embedding, SDPA, Norms, Pooling)
    # =======================================================================
    Case(
        "embedding_3table",
        "common",
        64,
        128,
        1024,
        embedding_3table_body,
        _make_emb_3table_inputs(30528, 768, tag="bert_emb"),
        n_dyn=2,
        tol=TIGHT,
        xfail_reason=_LAYER_NORM_IN_LOOP,
        note="Models: BERT, RoBERTa",
    ),
    Case(
        "bert_qkv_projection",
        "bert",
        64,
        192,
        1536,
        bert_qkv_body,
        lambda r: (
            (_x((r, 768), "b_qkv_h"),),
            (_w((768, 2304), "b_w_qkv"), _w((2304,), "b_b_qkv")),
        ),
    ),
    Case(
        "bert_sdpa_rectangular",
        "bert",
        2048,
        4096,
        28672,
        sdpa_rect_body,
        lambda r: (
            (_x((r, 12, 64), "b_q"), _x((r, 12, 64), "b_k"), _x((r, 12, 64), "b_v")),
            (torch.zeros(4, 1, 1, 512, dtype=torch.float16),),
        ),
        n_dyn=3,
        chunked_reference=True,
    ),
    Case(
        "bert_sdpa_decomposed",
        "bert",
        4,
        8,
        32,
        sdpa_decomposed_body,
        lambda r: (
            (
                _x((r, 12, 512, 64), "b_qd"),
                _x((r, 12, 512, 64), "b_kd"),
                _x((r, 12, 512, 64), "b_vd"),
            ),
            (torch.zeros(4, 1, 1, 512, dtype=torch.float16),),
        ),
        n_dyn=3,
        chunked_reference=True,
    ),
    Case(
        "bert_self_output",
        "bert",
        128,
        256,
        3072,
        bert_self_output_body,
        lambda r: (
            (_x((r, 768), "b_so_a"), _x((r, 768), "b_so_res")),
            (
                _w((768, 768), "b_so_w"),
                _w((768,), "b_so_b"),
                1.0 + _w((768,), "b_so_lw", scale=0.1),
                _w((768,), "b_so_lb", scale=0.1),
            ),
        ),
        n_dyn=2,
        xfail_reason=_LAYER_NORM_IN_LOOP,
    ),
    Case(
        "bert_intermediate",
        "bert",
        128,
        384,
        2560,
        bert_intermediate_body,
        lambda r: (
            (_x((r, 768), "b_int"),),
            (_w((768, 3072), "b_w1"), _w((3072,), "b_b1")),
        ),
    ),
    Case(
        "bert_output",
        "bert",
        256,
        512,
        4096,
        bert_output_body,
        lambda r: (
            (_x((r, 3072), "b_out_x"), _x((r, 768), "b_out_a")),
            (
                _w((3072, 768), "b_out_w"),
                _w((768,), "b_out_b"),
                1.0 + _w((768,), "b_out_lw", scale=0.1),
                _w((768,), "b_out_lb", scale=0.1),
            ),
        ),
        n_dyn=2,
        xfail_reason=_LAYER_NORM_IN_LOOP,
    ),
    Case(
        "bert_layer_norm",
        "bert",
        64,
        192,
        1984,
        layer_norm_standard_body,
        lambda r: (
            (_x((r, 768), "b_ln_x"),),
            (1.0 + _w((768,), "b_ln_w", scale=0.1), _w((768,), "b_ln_b", scale=0.1)),
        ),
        tol=TIGHT,
        xfail_reason=_LAYER_NORM_IN_LOOP,
    ),
    Case(
        "bert_token_classifier",
        "bert",
        512,
        1024,
        16384,
        bert_token_classifier_body,
        lambda r: (
            (_x((r, 768), "b_tok_h"),),
            (_w((768, 9), "b_tok_w"), _w((9,), "b_tok_b")),
        ),
    ),
    Case(
        "bert_mlm_head",
        "bert",
        64,
        128,
        1536,
        bert_mlm_head_body,
        lambda r: (
            (_x((r, 768), "b_mlm_x"),),
            (
                _w((768, 768), "b_mlm_wd"),
                _w((768,), "b_mlm_bd"),
                1.0 + _w((768,), "b_mlm_lw", scale=0.1),
                _w((768,), "b_mlm_lb", scale=0.1),
                _w((30522, 768), "b_mlm_wdec"),
                _w((30522,), "b_mlm_bdec"),
            ),
        ),
        xfail_reason=_LAYER_NORM_IN_LOOP,
    ),
    Case(
        "self_output_linear_residual_layer_norm",
        "common",
        128,
        256,
        3072,
        roberta_self_output_body,
        lambda r: (
            (_x((r, 1024), "rob_so_a"), _x((r, 1024), "rob_so_res")),
            (
                _w((1024, 1024), "rob_so_w"),
                _w((1024,), "rob_so_b"),
                1.0 + _w((1024,), "rob_so_lw", scale=0.1),
                _w((1024,), "rob_so_lb", scale=0.1),
            ),
        ),
        n_dyn=2,
        xfail_reason=_LAYER_NORM_IN_LOOP,
        note="Models: RoBERTa, Granite Embedding",
    ),
    Case(
        "cls_pool",
        "common",
        4,
        8,
        64,
        cls_pool_body,
        lambda r: (
            ((torch.arange(r, dtype=torch.int32) * 512).unsqueeze(-1),),
            (_x((r * 512, 1024), "rob_h"),),
        ),
        n_dyn=1,
        tol=TIGHT,
        note="Models: RoBERTa, Granite Embedding",
    ),
    Case(
        "l2_normalize",
        "common",
        4,
        8,
        32,
        l2_normalize_body,
        lambda r: ((_x((r, 1024), "rob_l2"),), ()),
        tol=TIGHT,
        note="Models: RoBERTa, Granite Embedding",
    ),
    Case(
        "roberta_mean_pool",
        "roberta",
        512,
        1024,
        16384,
        mean_pool_body,
        lambda r: (
            (_x((r, 1024), "rob_mp_h"), torch.ones((r, 1), dtype=torch.float16)),
            (torch.tensor([[512.0]], dtype=torch.float16),),
        ),
        n_dyn=2,
        tol=TIGHT,
        xfail_reason=_ROW_AXIS_RESHAPE,
    ),
    Case(
        "granite_emb_embedding",
        "granite_embedding",
        64,
        128,
        1024,
        embedding_3table_body,
        _make_emb_3table_inputs(131072, 768, max_pos=576, tag="ge_emb"),
        n_dyn=2,
        tol=LOOSE,
        ref_native_dtype=True,
        xfail_reason=_LAYER_NORM_IN_LOOP,
    ),
    Case(
        "granite_emb_bert_self_attention_qkv",
        "granite_embedding",
        512,
        1024,
        16384,
        granite_emb_qkv_heads_body,
        lambda r: (
            (_x((r, 768), "ge_qkv_x"),),
            (_w((768, 2304), "ge_qkv_w"), _w((2304,), "ge_qkv_b")),
        ),
        xfail_reason=_ROW_SHRINK_STRIDE,
    ),
    # =======================================================================
    # 2. Mistral Small 3.2 24B & Ministral 3 14B
    # =======================================================================
    Case(
        "mistral_embedding",
        "mistral",
        64,
        128,
        1024,
        embedding_single_lookup_body,
        _make_single_emb_inputs(16384, 5120, "mist_emb"),
        tol=TIGHT,
        ref_native_dtype=True,
    ),
    Case(
        "rms_norm_plain",
        "common",
        32,
        96,
        704,
        rms_norm_plain_body,
        _make_norm_plain_inputs(5120, "mist_lnp"),
        tol=TIGHT,
        note="Models: Mistral, Granite 4.2, Gemma 4",
    ),
    Case(
        "mistral_input_layernorm_fused",
        "mistral",
        32,
        64,
        800,
        rms_norm_fused_body,
        _make_norm_fused_inputs(5120, "mist_lnf"),
        n_dyn=2,
        tol=TIGHT,
    ),
    Case(
        "qkv_proj_split",
        "common",
        64,
        128,
        896,
        qkv_proj_split_body,
        _make_qkv_proj_inputs(5120, 6144, 32, 8, 128, "mist_qkv"),
        note="Models: Mistral, Granite 4.2",
    ),
    Case(
        "rotary_emb",
        "common",
        64,
        192,
        960,
        rotary_emb_body,
        _make_rotary_inputs(32, 8, 128, 131072, "mist_rot"),
        n_dyn=3,
        tol=TIGHT,
        note="Models: Mistral, Granite 4.2",
    ),
    Case(
        "kv_cache_write_native_dtype_ref",
        "common",
        128,
        256,
        1280,
        kv_cache_write_body,
        _make_kv_write_inputs(8, 128, 8, 128, "mist_kvw"),
        n_dyn=3,
        tol=TIGHT,
        n_carry=2,
        chunked_reference=True,
        ref_native_dtype=True,
        xfail_reason=_INDEX_COPY_IN_LOOP,
        note="Models: Mistral, Ministral",
    ),
    Case(
        "attn_prefill_kernel",
        "common",
        64,
        128,
        640,
        attn_prefill_body,
        _make_attn_prefill_inputs(32, 128, 512, "mist_apref"),
        n_dyn=3,
        tol=ATTN_TOL,
        chunked_reference=True,
        note="Models: Mistral, Ministral, Granite Vision",
    ),
    Case(
        "attn_decode_single",
        "common",
        4,
        8,
        40,
        attn_decode_single_body,
        _make_attn_decode_single_inputs(32, 128, 512, "mist_adec1"),
        n_dyn=4,
        tol=ATTN_TOL,
        chunked_reference=True,
        note="Models: Mistral, Ministral, Granite 4.2, Granite 4.1",
    ),
    Case(
        "attn_batched_decode",
        "common",
        4,
        12,
        60,
        attn_batched_decode_body,
        _make_attn_batched_decode_inputs(32, 128, 4, 128, "mist_adecb"),
        n_dyn=4,
        tol=ATTN_TOL,
        chunked_reference=True,
        note="Models: Mistral, Ministral, Granite 4.2, Granite 4.1",
    ),
    Case(
        "linear_matmul",
        "common",
        64,
        192,
        1152,
        linear_matmul_body,
        _make_linear_inputs(4096, 5120, "mist_o"),
        note="Models: Mistral (o_proj, gate_up_proj, down_proj), Ministral (gate_up_proj)",
    ),
    Case(
        "silu_and_mul",
        "common",
        64,
        256,
        1344,
        silu_and_mul_body,
        _make_silu_inputs(32768, "mist_silu"),
        tol=TIGHT,
        note="Models: Mistral, Ministral, Granite 4.2, Granite 4.1, Granite 3.3 FP8",
    ),
    Case(
        "mistral_lm_head",
        "mistral",
        8,
        16,
        64,
        linear_matmul_body,
        _make_linear_inputs(5120, 16384, "mist_lm"),
        ref_native_dtype=True,
    ),
    Case(
        "mistral_vision_attention_mask",
        "mistral",
        64,
        192,
        1024,
        mistral_vision_mask_body,
        lambda r: ((_segments(r),), ()),
        tol=TIGHT,
        chunked_reference=True,
        xfail_reason=_VISION_MASK_FAIL,
    ),
    Case(
        "mistral_vision_block",
        "mistral",
        64,
        128,
        512,
        mistral_vision_block_body,
        _make_mistral_vision_block_inputs,
        n_dyn=4,
        out_dim=1,
        chunked_reference=True,
        xfail_reason=_VISION_MASK_FAIL,
    ),
    Case(
        "ministral_qkv_split",
        "ministral",
        128,
        256,
        2048,
        ministral_qkv_split_body,
        lambda r: ((_x((r, 6144), "min_qkvs"),), ()),
        tol=TIGHT,
    ),
    # =======================================================================
    # 3. Granite 4.2 8B, Granite 4.1 8B & Granite 3.3 8B
    # =======================================================================
    Case(
        "embedding_single_lookup",
        "common",
        512,
        1024,
        6144,
        embedding_single_lookup_body,
        _make_single_emb_inputs(100352, 4096, "g42_emb"),
        tol=TIGHT,
        note="Models: Granite 4.2, Gemma 4. Gemma 4's real 262144x2816 fp16 table (1.4 GB) is not tested: its per-core span (704 MB) exceeds the 256 MB hardware limit (MAX_SPAN_BYTES) and it fails without for_each_tile too",
    ),
    Case(
        "granite_4_2_attn_residual",
        "granite_4_2",
        256,
        1024,
        5120,
        residual_combine_body,
        _make_residual_inputs(4096, "g42_res"),
        n_dyn=2,
        tol=TIGHT,
    ),
    Case(
        "embedding_single_lookup_x12",
        "common",
        64,
        128,
        1024,
        granite_4_1_embed_body,
        _make_single_emb_inputs(100352, 4096, "g41_emb", multiplier=12.0),
        tol=TIGHT,
        note="Models: Granite 4.1, Granite 3.3 (12x embedding multiplier)",
    ),
    Case(
        "kv_cache_write",
        "common",
        64,
        128,
        768,
        kv_cache_write_body,
        _make_kv_write_inputs(8, 128, 7, 128, "g41_kvw"),
        n_dyn=3,
        tol=TIGHT,
        n_carry=2,
        chunked_reference=True,
        xfail_reason=_INDEX_COPY_IN_LOOP,
        note="Models: Granite 4.1, Granite 4.2, Granite 3.3 FP8",
    ),
    Case(
        "granite_4_1_attn_residual",
        "granite_4_1",
        64,
        128,
        1152,
        granite_4_1_residual_body,
        _make_residual_inputs(4096, "g41_res"),
        n_dyn=2,
        tol=TIGHT,
    ),
    # =======================================================================
    # 4. FP8 Modules (Granite 4.1 FP8 & Granite 3.3 FP8)
    # =======================================================================
    Case(
        "fp8_linear",
        "common",
        64,
        128,
        896,
        fp8_linear_body,
        _make_fp8_linear_inputs(4096, 6144, "g41_fp8_qkv"),
        tol=FP8_TOL,
        xfail_reason=_FP8_INDUCTOR_BUF,
        note="Models: Granite 4.1 FP8 (qkv/o/gate_up/down_proj), Granite 3.3 FP8 (qkv_proj)",
    ),
    # =======================================================================
    # 5. Gemma 4 Modules
    # =======================================================================
    Case(
        "gemma4_dense_mlp",
        "gemma4",
        64,
        128,
        1024,
        gemma4_dense_mlp_body,
        lambda r: (
            (_x((r, 2816), "gem_mlp"),),
            (_w((2816, 4224), "gem_wgu"), _w((2112, 2816), "gem_wd")),
        ),
    ),
    Case(
        "gemma4_attention",
        "gemma4",
        8,
        16,
        96,
        gemma4_attention_body,
        _make_gemma4_attn_inputs,
        n_dyn=6,
        tol=ATTN_TOL,
        xfail_reason=_KV_CACHE_WRITE_FAIL,
    ),
    Case(
        "gemma4_moe",
        "gemma4",
        4,
        8,
        32,
        gemma4_moe_body,
        _make_gemma4_moe_inputs,
        n_dyn=2,
        chunked_reference=True,
        ref_native_dtype=True,
        xfail_reason=_MOE_CODEGEN_FAIL,
    ),
    Case(
        "gemma4_decoder_layer",
        "gemma4",
        4,
        12,
        36,
        gemma4_decoder_layer_body,
        _make_gemma4_decoder_layer_inputs,
        n_dyn=4,
        chunked_reference=True,
        ref_native_dtype=True,
        xfail_reason=_MOE_CODEGEN_FAIL,
    ),
    Case(
        "gemma4_vision_pool_proj",
        "gemma4",
        64,
        128,
        1024,
        gemma4_vision_pool_proj_body,
        lambda r: ((_x((r, 9, 1152), "gem_vpp"),), ()),
        xfail_reason=_VISION_POOL_PROJ_FAIL,
    ),
    # =======================================================================
    # 6. Granite Vision 3.3 2B & Multimodal Modules
    # =======================================================================
    Case(
        "granite_vision_siglip_encoder_layer",
        "granite_vision",
        4,
        12,
        72,
        granite_vision_siglip_encoder_body,
        _make_siglip_encoder_inputs,
        xfail_reason=_SIGLIP_ENCODER_FAIL,
    ),
    Case(
        "granite_vision_llava_projector",
        "granite_vision",
        4,
        20,
        100,
        granite_vision_llava_projector_body,
        lambda r: (
            (_x((r, 729, 4608), "gv_proj_x"),),
            (
                _w((4608, 2048), "gv_pw1"),
                _w((2048,), "gv_pb1"),
                _w((2048, 2048), "gv_pw2"),
                _w((2048,), "gv_pb2"),
            ),
        ),
    ),
    Case(
        "granite_vision_merge_patch",
        "granite_vision",
        4,
        8,
        96,
        granite_vision_merge_patch_body,
        lambda r: ((_x((r, 729, 2048), "gv_mp_p"),), (_x((1, 1, 2048), "gv_mp_nl"),)),
    ),
    Case(
        "granite_vision_splice_embeddings",
        "granite_vision",
        8,
        16,
        128,
        granite_vision_splice_embeddings_body,
        lambda r: (
            (
                _x((r, 2048), "gv_sp_in"),
                _x((r, 2048), "gv_sp_mm"),
                torch.randint(0, 2, (r, 1), dtype=torch.bool),
            ),
            (),
        ),
        n_dyn=3,
        tol=TIGHT,
    ),
    # =======================================================================
    # 7. YOLO26 Dynamic Batch Modules
    # =======================================================================
    Case(
        "yolo26_Conv",
        "yolo26",
        4,
        12,
        48,
        yolo_conv_body,
        lambda r: (
            (_x((r, 3, 256, 256), "yolo_conv_x", dtype=torch.float32),),
            (
                torch.randn(16, 3, 3, 3),
                torch.ones(16),
                torch.zeros(16),
                torch.zeros(16),
                torch.ones(16),
            ),
        ),
        xfail_reason=_YOLO_CONV_FAIL,
    ),
    Case(
        "yolo26_Conv2d",
        "yolo26",
        2,
        32,
        64,
        yolo_conv2d_body,
        lambda r: (
            (_x((r, 3, 256, 256), "yolo_c2d_x", dtype=torch.float32),),
            (_w((16, 3, 3, 3), "yolo_c2d_w", dtype=torch.float32),),
        ),
        xfail_reason=_YOLO_CONV_FAIL,
    ),
    Case(
        "yolo26_BatchNorm2d",
        "yolo26",
        2,
        10,
        60,
        yolo_batchnorm_body,
        lambda r: (
            (_x((r, 16, 128, 128), "yolo_bn_x", dtype=torch.float32),),
            (torch.ones(16), torch.zeros(16), torch.zeros(16), torch.ones(16)),
        ),
        tol=TIGHT,
        xfail_reason=_YOLO_BN_FAIL,
    ),
    Case(
        "yolo26_SiLU",
        "yolo26",
        2,
        14,
        56,
        yolo_silu_body,
        lambda r: ((_x((r, 16, 128, 128), "yolo_silu_x", dtype=torch.float32),), ()),
        tol=TIGHT,
    ),
    Case(
        "yolo26_C3k2",
        "yolo26",
        2,
        20,
        60,
        yolo_c3k2_body,
        lambda r: (
            (_x((r, 32, 64, 64), "yolo_c3k2_x", dtype=torch.float32),),
            _yolo_weights("c3k2"),
        ),
        xfail_reason=_YOLO_CONV_FAIL,
    ),
    Case(
        "yolo26_Bottleneck",
        "yolo26",
        8,
        120,
        240,
        yolo_bottleneck_body,
        lambda r: (
            (_x((r, 16, 64, 64), "yolo_bot_x", dtype=torch.float32),),
            _yolo_weights("bottleneck"),
        ),
        xfail_reason=_YOLO_CONV_FAIL,
    ),
    Case(
        "yolo26_C3k",
        "yolo26",
        32,
        512,
        1024,
        yolo_c3k_body,
        lambda r: (
            (_x((r, 64, 16, 16), "yolo_c3k_x", dtype=torch.float32),),
            _yolo_weights("c3k"),
        ),
        xfail_reason=_YOLO_C3K_FAIL,
    ),
    Case(
        "yolo26_Sequential",
        "yolo26",
        64,
        1024,
        2048,
        yolo_sequential_body,
        lambda r: (
            (_x((r, 32, 16, 16), "yolo_seq_x", dtype=torch.float32),),
            _yolo_weights("sequential"),
        ),
        xfail_reason=_YOLO_BN_FAIL,
    ),
    Case(
        "yolo26_SPPF",
        "yolo26",
        32,
        320,
        640,
        yolo_sppf_body,
        lambda r: (
            (_x((r, 256, 8, 8), "yolo_sppf_x", dtype=torch.float32),),
            _yolo_weights("sppf"),
        ),
        xfail_reason=_YOLO_SPPF_FAIL,
    ),
    Case(
        "yolo26_Identity",
        "yolo26",
        64,
        896,
        1792,
        yolo_identity_body,
        lambda r: ((_x((r, 128, 8, 8), "yolo_id_x", dtype=torch.float32),), ()),
        tol=TIGHT,
    ),
    Case(
        "yolo26_MaxPool2d",
        "yolo26",
        4,
        64,
        128,
        yolo_maxpool_body,
        lambda r: ((_x((r, 128, 8, 8), "yolo_mp_x", dtype=torch.float32),), ()),
        tol=TIGHT,
        xfail_reason=_YOLO_MAXPOOL_FAIL,
    ),
    Case(
        "yolo26_Upsample",
        "yolo26",
        8,
        128,
        256,
        yolo_upsample_body,
        lambda r: ((_x((r, 256, 8, 8), "yolo_up_x", dtype=torch.float32),), ()),
        tol=TIGHT,
        xfail_reason=_YOLO_UPSAMPLE_FAIL,
    ),
    Case(
        "yolo26_Concat",
        "yolo26",
        8,
        72,
        144,
        yolo_concat_body,
        lambda r: (
            (
                _x((r, 256, 16, 16), "yolo_cat_a", dtype=torch.float32),
                _x((r, 128, 16, 16), "yolo_cat_b", dtype=torch.float32),
            ),
            (),
        ),
        n_dyn=2,
        tol=TIGHT,
    ),
)

# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------


def _case_param(case: Case):
    """Format pytest parameter with xfail mark if tracked by an open issue."""
    marks = []
    if case.xfail_reason:
        if isinstance(case.xfail_reason, tuple):
            reason, raises = case.xfail_reason
        else:
            reason, raises = case.xfail_reason, case.xfail_raises
        marks.append(pytest.mark.xfail(strict=True, raises=raises, reason=reason))
    return pytest.param(case, id=case.name, marks=marks)


@pytest.mark.parametrize("case", [_case_param(c) for c in CASES])
def test_dynamic_batch_modules(case: Case):
    """Run dynamic batch module test across min, max and 5 intermediate runtime sizes."""
    if isinstance(counters, dict) and "stats" in counters:
        counters["stats"].clear()

    dyn_cpu, static_cpu = case.make_inputs(case.hi)
    assert all(t.dim() >= 2 for t in dyn_cpu), (
        f"{case.name}: declared dimension cannot be on a 1-D tensor"
    )

    spec = {"min": case.lo, "max": case.hi, "granularity": case.tile}
    dyn = [t.to(DEVICE_NAME, dynamic={0: spec}) for t in dyn_cpu]
    static = [t.to(DEVICE_NAME) for t in static_cpu]

    fn = torch.compile(traced_module(case), backend="inductor", fullgraph=True)

    reference = cpu_reference(case)

    def _f32_cast(t):
        if case.ref_native_dtype:
            return t
        return t.float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t

    for rows in runtime_sizes(case.lo, case.hi, case.tile):
        for on_device, on_host in zip(dyn, dyn_cpu):
            if on_device.size(0) != rows:
                on_device.resize_((rows, *on_host.shape[1:]))

        out = _to_cpu_f32(fn(*dyn, *static))
        cpu_args = [_f32_cast(t[:rows]) for t in dyn_cpu] + [
            _f32_cast(t) for t in static_cpu
        ]

        compare_with_cpu(
            reference,
            *cpu_args,
            atol=case.tol,
            rtol=case.tol,
            target=out,
            run_eager=False,
            cpu_compile=False,
        )

    if (
        isinstance(counters, dict)
        and "stats" in counters
        and "unique_graphs" in counters["stats"]
    ):
        assert counters["stats"]["unique_graphs"] == 1, (
            f"{case.name}: a runtime size caused an unexpected graph recompile"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
