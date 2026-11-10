# Copyright 2026 The Torch-Spyre Authors.
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

"""Gemma 4 26B-A4B-it MoE functional verification tests for Torch-Spyre.

Shapes and structure follow the Gemma 4 26B-A4B-it config and Transformers v5.15.0,
traced for a 12-token prefill and 1-token decode steps (96 and 8 routed rows):

* 3.1 Router (``Gemma4TextRouter``).
* 3.2 MoE experts as Gemma 4 runs them (``grouped_mm_experts_forward``): sort,
  ``perm // K`` gather, ``histc``/``cumsum`` offsets, ``_grouped_mm`` and an FP32
  combine. This is the main gate.
* 3.3 MoE numerics and routing edge cases, on the same path.
* 3.4 Expert FFN and layout.
* 3.5 Attention and the decoder layer. Sliding-window layers have 8 KV heads of
  256 and RoPE theta 1e4. Full layers have 2 KV heads of 512, take V from k_proj
  (``attention_k_eq_v``) and use proportional RoPE over a quarter of the head with
  theta 1e6. Both apply Q/K/V RMS norms and ``scale=1.0``. The decoder layer runs
  the dense MLP and the MoE in parallel. Prefill is causal, and decode attends over
  a 12-token KV cache; the 1024-token sliding window is longer than both.
* 3.6 A Spyre implementation option, kept small: expert segments padded to 32-row
  tiles and an ``index_add`` combine. It covers ops the grouped_mm path does not
  reach.

Weights are synthetic. Every scenario also runs with ``execution_mode="cpu"``. That
case uses no Spyre device and is never xfailed, so a bug in a reference or an
assertion cannot hide behind a backend xfail.
"""

from __future__ import annotations

import functools
import math
import os
import sys

import pytest
import regex as re
import torch
from torch._inductor.exc import InductorError

sys.path.insert(0, os.path.dirname(__file__))
from utils_inductor import compare_with_cpu


# ---------------------------------------------------------------------------
# Gemma / FVT constants
# ---------------------------------------------------------------------------

HIDDEN = 2816
EXPERTS = 128
K8 = 8
MOE_INTERMEDIATE = 704
GATE_UP = MOE_INTERMEDIATE * 2
TILE = 32
PREFILL_T = 12
DECODE_T = 1
ALL_EXPERTS_T = EXPERTS // K8
NUM_Q_HEADS = 16
# Sliding-window attention layers (25 of 30).
NUM_KV_HEADS = 8
HEAD_DIM = 256
SLIDING_ROPE_THETA = 10_000.0
# Full attention layers (5 of 30). K and V share k_proj (attention_k_eq_v).
NUM_GLOBAL_KV_HEADS = 2
GLOBAL_HEAD_DIM = 512
GLOBAL_ROPE_THETA = 1_000_000.0
GLOBAL_PARTIAL_ROTARY = 0.25
DENSE_INTERMEDIATE = 2112
RMS_NORM_EPS = 1e-6


EXACT_ATOL = 0.0
EXACT_RTOL = 0.0
FP32_ATOL = 0.001
FP32_RTOL = 0.001
BF16_ATOL = 0.005
BF16_RTOL = 0.005
FP32_REF_ATOL = 0.002
# BF16 against an FP32 reference on CPU: at most 0.013 for attention (outputs up to
# 3.7) and the dense MLP (up to 2.7), and 0.064 for the decoder layer (up to 7.5,
# where one BF16 step is 0.03). Each tolerance allows about twice that.
# Two BF16 results computed in a different order can each round one step apart, so
# comparisons between them also allow four BF16 steps of relative error.
ATTN_ATOL = 0.03
ATTN_RTOL = 2**-6
MLP_ATOL = 0.03
MLP_RTOL = 2**-6
DECODER_ATOL = 0.125
DECODER_RTOL = 2**-6
SEED = 20260908


def _compare_mode(execution_mode, fn, *args, atol=BF16_ATOL, rtol=BF16_RTOL, **kwargs):
    if execution_mode == "cpu":
        return

    if execution_mode not in ("eager", "compiled"):
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    compiled = execution_mode == "compiled"
    try:
        compare_with_cpu(
            fn,
            *args,
            atol=atol,
            rtol=rtol,
            run_compile=compiled,
            run_eager=not compiled,
            cpu_compile=compiled,
            **kwargs,
        )
    except Exception as e:
        # raises= pins only the exception type. A fix that moves the failure to the
        # next unsupported op of the same type must fail here, not stay xfailed.
        if _EXPECTED_MATCH is not None and not re.search(_EXPECTED_MATCH, str(e)):
            pytest.fail(
                f"Expected a failure matching {_EXPECTED_MATCH!r}, "
                f"got {type(e).__name__}: {e}",
                pytrace=False,
            )
        raise


# Failure-message pattern of the running case's xfail marker, if any.
_EXPECTED_MATCH = None


@pytest.fixture(autouse=True)
def _xfail_match(request):
    global _EXPECTED_MATCH
    mark = request.node.get_closest_marker("xfail")
    _EXPECTED_MATCH = mark.kwargs.get("match") if mark else None
    yield
    _EXPECTED_MATCH = None


def _xfail(raises, reason, match):
    """Strict xfail on ``raises`` whose message must also match ``match``."""
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason, match=match)


def _modes(eager=None, compiled=None):
    return [
        "cpu",
        pytest.param("eager", marks=eager or ()),
        pytest.param("compiled", marks=compiled or ()),
    ]


def _cases(eager=None, compiled=None, decode_eager=None, decode_compiled=None):
    cases = []
    for T, name in ((PREFILL_T, "prefill12"), (DECODE_T, "decode1")):
        decode = T == DECODE_T
        e = decode_eager if decode and decode_eager is not None else eager
        c = decode_compiled if decode and decode_compiled is not None else compiled
        cases += [
            pytest.param("cpu", T, id=f"cpu-{name}"),
            pytest.param("eager", T, marks=e or (), id=f"eager-{name}"),
            pytest.param("compiled", T, marks=c or (), id=f"compiled-{name}"),
        ]
    return cases


# Failures shared by every scenario that runs the full grouped MoE path.
GROUPED_MOE_EAGER = _xfail(
    InductorError,
    "#3193: routing flatten raises no mechanism to resolve stick incompatibility",
    "no mechanism to resolve stick incompatibility",
)
GROUPED_MOE_COMPILED = _xfail(
    InductorError,
    "#4648: restickify padding finds no input host dim for the new stick symbol",
    "restickify padding: no input host dim carries new-stick symbol",
)
STABLE_SORT = _xfail(
    NotImplementedError,
    "#4500: aten::sort.values_stable has no Spyre kernel",
    r"aten::sort\.values_stable",
)
FLOOR_DIVIDE = _xfail(
    NotImplementedError,
    "#4643: aten::floor_divide has no Spyre kernel",
    "aten::floor_divide",
)
TILE_OFFSET_CUMSUM = _xfail(
    InductorError,
    "#4754: no offset-free alternative stick dim for mutation target",
    "no offset-free alternative stick dim for mutation target",
)
TOPK_FLOAT_IDS = _xfail(
    AssertionError,
    "#5847: topk returns float32 indices",
    r"'dtype' do not match: torch\.float32 != torch\.int64",
)
TOPK_WRONG_VALUES = _xfail(
    AssertionError,
    "#4639: compiled K=8 topk returns incorrect values",
    "Tensor-likes are not close",
)
INDEX_2D = _xfail(
    NotImplementedError,
    "#1219: aten::index.Tensor supports only a 1-D index",
    r"aten::index\.Tensor supports only",
)
INDEX_ADD_DUPLICATES = _xfail(
    AssertionError,
    "#4411: index_add keeps the last duplicate instead of summing",
    "Tensor-likes are not close",
)
# At T=1 the routing flatten succeeds and the grouped path stops at the stable sort.
SORT_AT_DECODE = dict(decode_eager=STABLE_SORT, decode_compiled=STABLE_SORT)
MIXED_EA = _xfail(
    InductorError,
    "#4714: BF16 x FP32 broadcast multiply raises mixed EA with no staggered operand",
    "Multi-arg pointwise with mixed EA",
)
FLOAT_IDS_AS_INDEX = _xfail(
    RuntimeError,
    "#5847: float32 topk ids cannot index per_expert_scale",
    "tensors used as indices must be long, int, byte or bool",
)
FLOAT_IDS_AS_INDEX_EAGER = _xfail(
    IndexError,
    "#5847: float32 topk ids cannot index per_expert_scale",
    "tensors used as indices must be long, int, byte or bool",
)
SORT_MULTI_OUTPUT = _xfail(
    InductorError,
    "#5831: split_multi_ops fails on multi-output sort",
    "'tuple' object has no attribute 'to'",
)
# The grouped_mm MoE stops at its sort: no stable-sort kernel eager, and
# split_multi_ops rejects the multi-output sort compiled.
GROUPED_MM_SORT = dict(eager=STABLE_SORT, compiled=SORT_MULTI_OUTPUT)
UPCAST_RMS_NORM = _xfail(
    AssertionError,
    "#5893: FP32 RMSNorm of an upcast BF16 input returns wrong values, so eager "
    "attention outputs NaN",
    "Greatest absolute difference: nan",
)


def _seed() -> None:
    torch.manual_seed(SEED)


def _assert_close(actual, expected, *, atol=BF16_ATOL, rtol=BF16_RTOL):
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_close(a, e, atol=atol, rtol=rtol)
        return
    assert torch.allclose(
        actual.float().cpu(), expected.float().cpu(), atol=atol, rtol=rtol
    ), f"max_abs={(actual.float().cpu() - expected.float().cpu()).abs().max().item()}"


def _assert_no_nan_inf(x: torch.Tensor) -> None:
    assert torch.isfinite(x.float()).all().item()


def _router_logits(T: int) -> torch.Tensor:
    _seed()
    return torch.randn(T, EXPERTS, dtype=torch.bfloat16)


def _routing_weights(logits: torch.Tensor):
    # Gemma 4: FP32 softmax over all 128 experts -> top-8 -> renormalize.
    router_probabilities = torch.softmax(
        logits,
        dim=-1,
        dtype=torch.float32,
    )
    values, indices = torch.topk(
        router_probabilities,
        8,
        dim=-1,
    )
    values = values / values.sum(dim=-1, keepdim=True)
    return values, indices


def _flatten_routing(
    hidden: torch.Tensor, weights: torch.Tensor, expert_ids: torch.Tensor
):
    T, K = expert_ids.shape
    token_of_row = torch.arange(T, device=hidden.device).repeat_interleave(K)
    row_weight = weights.reshape(-1)
    routed_hidden = hidden.repeat_interleave(K, dim=0)
    expert_of_row = expert_ids.reshape(-1)
    return routed_hidden, row_weight, token_of_row, expert_of_row


def _group(expert_of_row: torch.Tensor):
    sort_perm = torch.argsort(expert_of_row, stable=True)
    expert_sorted = expert_of_row[sort_perm]
    counts = torch.bincount(expert_sorted, minlength=EXPERTS)
    group_off = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=counts.device), counts.cumsum(0)]
    )
    return sort_perm, expert_sorted, counts, group_off


def _tile_metadata(counts: torch.Tensor):
    tiles = (counts + TILE - 1) // TILE
    padded = tiles * TILE
    pad_off = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=counts.device), padded.cumsum(0)]
    )
    tile_expert = torch.arange(
        EXPERTS, dtype=torch.long, device=counts.device
    ).repeat_interleave(tiles)
    return tiles, padded, pad_off, tile_expert


def _dst_pos(counts, group_off, pad_off):
    starts = pad_off[:-1].repeat_interleave(counts)
    intra = torch.arange(group_off[-1], device=group_off.device) - group_off[
        :-1
    ].repeat_interleave(counts)
    return starts + intra


def _pad_columns(routed_sorted, weight_sorted, token_sorted, dst, n_pad, sink):
    """Place each sorted row at ``dst``, so every expert starts on a tile."""
    out_x = torch.zeros(
        n_pad,
        routed_sorted.shape[1],
        dtype=routed_sorted.dtype,
        device=routed_sorted.device,
    )
    out_w = torch.zeros(n_pad, dtype=weight_sorted.dtype, device=weight_sorted.device)
    out_t = torch.full(
        (n_pad,), sink, dtype=token_sorted.dtype, device=token_sorted.device
    )
    out_x[dst] = routed_sorted
    out_w[dst] = weight_sorted
    out_t[dst] = token_sorted
    return out_x, out_w, out_t


@functools.lru_cache(maxsize=None)
def _expert_bank_3d():
    """All 128 experts as _grouped_mm takes them: [128, 2816, 1408], [128, 704, 2816]."""
    gate_up = torch.empty(EXPERTS, HIDDEN, GATE_UP, dtype=torch.bfloat16)
    down = torch.empty(EXPERTS, MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    for e in range(EXPERTS):
        g = torch.Generator(device="cpu")
        g.manual_seed(SEED + e)
        gate_up[e] = torch.randn(HIDDEN, GATE_UP, generator=g) * 0.01
        down[e] = torch.randn(MOE_INTERMEDIATE, HIDDEN, generator=g) * 0.01
    return gate_up, down


def _expert_slab(e):
    """Return expert ``e``'s (gate_up, down) slabs, as views into the 3-D bank."""
    gate_up, down = _expert_bank_3d()
    return gate_up[e], down[e]


def _expert_weights(expert_ids):
    """Return the expert slabs needed by a scenario."""
    return {e: _expert_slab(e) for e in sorted({int(x) for x in expert_ids})}


def _expert_ffn(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor):
    gu = x @ gate_up
    gate, up = gu.split(MOE_INTERMEDIATE, dim=-1)
    return (torch.nn.functional.gelu(gate, approximate="tanh") * up) @ down


def _moe_reference(hidden, expert_ids, weights, expert_bank):
    T, K = expert_ids.shape
    out = torch.zeros(T, HIDDEN, dtype=torch.bfloat16, device=hidden.device)
    flat_x = hidden.repeat_interleave(K, dim=0)
    flat_e = expert_ids.reshape(-1)
    flat_w = weights.reshape(-1)
    flat_token = torch.arange(T, device=hidden.device).repeat_interleave(K)
    for e in torch.unique(flat_e).tolist():
        e = int(e)
        mask = flat_e == e
        rows = flat_x[mask]
        ge, de = expert_bank[e]
        y = _expert_ffn(rows, ge.to(hidden.device), de.to(hidden.device))
        y = y * flat_w[mask].to(y.dtype).unsqueeze(1)
        out.index_add_(0, flat_token[mask], y)
    return out


def _grouped_moe(hidden, expert_ids, weights, expert_bank):
    routed, row_w, token, expert = _flatten_routing(hidden, weights, expert_ids)
    perm, expert_sorted, counts, group_off = _group(expert)
    routed = routed[perm]
    row_w = row_w[perm]
    token = token[perm]
    _, _, pad_off, tile_expert = _tile_metadata(counts)
    n_pad = int(pad_off[-1].item())
    dst = _dst_pos(counts, group_off, pad_off)
    routed_pad, weight_pad, token_pad = _pad_columns(
        routed, row_w, token, dst, n_pad, hidden.shape[0]
    )

    out = torch.zeros(
        hidden.shape[0] + 1,
        HIDDEN,
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    for tile in range(int(tile_expert.numel())):
        e = int(tile_expert[tile].item())
        start = tile * TILE
        stop = start + TILE
        ge, de = expert_bank[e]
        if ge.device != routed_pad.device:
            ge, de = ge.to(routed_pad.device), de.to(routed_pad.device)
        y = _expert_ffn(routed_pad[start:stop], ge, de)
        y = y * weight_pad[start:stop].to(y.dtype).unsqueeze(1)
        out.index_add_(0, token_pad[start:stop], y)
    return out[:-1], (perm, counts, group_off, pad_off, tile_expert, dst)


def _deterministic_all_experts(T=ALL_EXPERTS_T, K=K8):
    return ((torch.arange(T).unsqueeze(1) * K + torch.arange(K)) % EXPERTS).long()


def _per_expert_scale() -> torch.Tensor:
    """Non-uniform BF16 [128] scale, the dtype Gemma4TextRouter stores it in."""
    return (0.5 + torch.arange(EXPERTS, dtype=torch.float32) / EXPERTS).to(
        torch.bfloat16
    )


def _router_params():
    """Synthetic Gemma4TextRouter parameters: scale [2816] and proj [128, 2816]."""
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED + 5000)
    scale = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    proj = torch.randn(EXPERTS, HIDDEN, generator=g).to(torch.bfloat16)
    return scale, proj, _per_expert_scale()


def _gemma4_router(hidden, scale, proj, per_expert_scale, ids=None):
    """Gemma4TextRouter.forward: weights [T, 8] FP32 and expert ids [T, 8].

    ``ids`` replaces the top-8 choice; the weights are those experts' renormalized
    probabilities.
    """
    x = _rms_norm(hidden)
    x = x * scale * HIDDEN**-0.5
    logits = x @ proj.t()
    if ids is None:
        weights, ids = _routing_weights(logits)
    else:
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32).gather(-1, ids)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * per_expert_scale[ids], ids


def _hf_dispatch(hidden, weights, ids):
    """Sort routed rows by expert and gather their activations and weights."""
    expert_ids_g, perm = torch.sort(ids.reshape(-1))
    return expert_ids_g, perm, hidden[perm // K8], weights.reshape(-1)[perm]


def _hf_offsets(expert_ids_g):
    """Cumulative row offsets of each expert segment, as _grouped_mm takes them."""
    tokens_per_expert = torch.histc(
        expert_ids_g.float(), bins=EXPERTS, min=0, max=EXPERTS - 1
    )
    return torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)


def _hf_experts(x_g, offsets, gate_up, down):
    gu = torch._grouped_mm(x_g, gate_up, offs=offsets)
    gate, up = gu.chunk(2, dim=-1)
    act = torch.nn.functional.gelu(gate, approximate="tanh") * up
    return torch._grouped_mm(act, down, offs=offsets)


def _hf_combine(y_g, w_g, perm, T):
    """Weight, restore token order and sum each token's K rows in FP32."""
    weighted = y_g * w_g.unsqueeze(-1)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=perm.device)
    return weighted[inv_perm].view(T, K8, HIDDEN).sum(dim=1)


def _hf_moe(hidden, weights, ids, gate_up, down):
    """Transformers grouped_mm_experts_forward, without the EP sentinel masks."""
    expert_ids_g, perm, x_g, w_g = _hf_dispatch(hidden, weights, ids)
    y_g = _hf_experts(x_g, _hf_offsets(expert_ids_g), gate_up, down)
    return _hf_combine(y_g, w_g, perm, hidden.shape[0]).to(hidden.dtype)


def _moe_fp32_reference(hidden, ids, weights, bank):
    h = hidden.float()
    out = torch.zeros_like(h)
    flat_x = h.repeat_interleave(K8, 0)
    flat_ids = ids.reshape(-1)
    flat_w = weights.reshape(-1)
    flat_token = torch.arange(hidden.shape[0]).repeat_interleave(K8)
    for e in torch.unique(flat_ids).tolist():
        e = int(e)
        mask = flat_ids == e
        rows = flat_x[mask]
        gu = bank[e][0].float()
        down = bank[e][1].float()
        gu = gu.to(h.device)
        down = down.to(h.device)
        y = _expert_ffn(rows, gu, down).float()
        y = y * flat_w[mask].unsqueeze(1)
        out.index_add_(0, flat_token[mask], y)
    return out


# ---------------------------------------------------------------------------
# Gemma 4 attention, dense MLP and decoder layer (Transformers v5.15.0)
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Gemma4RMSNorm in FP32, then an optional weight.

    Uses ``rsqrt(mean(x^2) + eps)``, which equals Gemma 4's ``pow(..., -0.5)``.
    """
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    if weight is not None:
        out = out * weight.float()
    return out.type_as(x)


def _attention_dims(kind):
    """(KV heads, head dim, RoPE theta, rotary fraction) of a sliding or full layer."""
    if kind == "sliding":
        return NUM_KV_HEADS, HEAD_DIM, SLIDING_ROPE_THETA, 1.0
    return (
        NUM_GLOBAL_KV_HEADS,
        GLOBAL_HEAD_DIM,
        GLOBAL_ROPE_THETA,
        GLOBAL_PARTIAL_ROTARY,
    )


def _rope_cos_sin(positions, kind):
    """Gemma4TextRotaryEmbedding tables [T, head_dim], cast to BF16 as the layer gets them.

    Full layers use proportional RoPE: only the first ``fraction * head_dim / 2``
    frequencies are non-zero, so the other dimensions pass through unrotated.
    """
    _, d, theta, fraction = _attention_dims(kind)
    angles = int(fraction * d // 2)
    inv_freq = 1.0 / theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / d)
    inv_freq = torch.cat((inv_freq, torch.zeros(d // 2 - angles)))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x, cos, sin):
    """apply_rotary_pos_emb for x [T, heads, D] and tables [T, D]."""
    return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]


def _sdpa(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q[None], k[None], v[None], is_causal=causal, scale=1.0
    )[0]


def _manual_attention(q, k, v, causal):
    """Reference for _sdpa: explicit scores, causal mask and softmax."""
    scores = q @ k.transpose(-1, -2)
    if causal:
        T, S = q.shape[-2], k.shape[-2]
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _gemma4_attention(
    x,
    cos,
    sin,
    q_w,
    k_w,
    v_w,
    o_w,
    q_norm,
    k_norm,
    past_k=None,
    past_v=None,
    attend=_sdpa,
):
    """Gemma4TextAttention.forward for x [T, 2816], with an optional KV cache.

    ``v_w is None`` is a full layer with ``attention_k_eq_v``: V comes from k_proj.
    """
    T, d = x.shape[0], q_norm.shape[0]
    kv_heads = k_w.shape[1] // d
    q = _rms_norm((x @ q_w).view(T, NUM_Q_HEADS, d), q_norm)
    k = (x @ k_w).view(T, kv_heads, d)
    v = k if v_w is None else (x @ v_w).view(T, kv_heads, d)
    q = _apply_rope(q, cos, sin).transpose(0, 1)
    k = _apply_rope(_rms_norm(k, k_norm), cos, sin).transpose(0, 1)
    v = _rms_norm(v).transpose(0, 1)
    if past_k is not None:
        k = torch.cat((past_k, k), dim=1)
        v = torch.cat((past_v, v), dim=1)
    groups = NUM_Q_HEADS // kv_heads
    k = k.repeat_interleave(groups, dim=0)
    v = v.repeat_interleave(groups, dim=0)
    out = attend(q, k, v, causal=past_k is None and T > 1)
    return out.transpose(0, 1).reshape(T, NUM_Q_HEADS * d) @ o_w


def _bf16_weights(seed, *shapes):
    """Random BF16 weights [in, out] scaled by in**-0.5, so activations stay near unit."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return [
        (torch.randn(*shape, generator=g) * shape[0] ** -0.5).to(torch.bfloat16)
        for shape in shapes
    ]


def _norm_weights(seed, n, count, scale=1.0):
    """``count`` BF16 RMSNorm weights near ``scale``."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return [
        ((1 + 0.1 * torch.randn(n, generator=g)) * scale).to(torch.bfloat16)
        for _ in range(count)
    ]


def _attention_weights(kind):
    """Synthetic weights in _gemma4_attention order: q, k, v (None for full), o, norms."""
    kv_heads, d, _, _ = _attention_dims(kind)
    seed = SEED + (6000 if kind == "sliding" else 7000)
    q, k, v, o = _bf16_weights(
        seed,
        (HIDDEN, NUM_Q_HEADS * d),
        (HIDDEN, kv_heads * d),
        (HIDDEN, kv_heads * d),
        (NUM_Q_HEADS * d, HIDDEN),
    )
    # With scale=1.0, trained q/k norms keep the scores small. d**-0.25 does the
    # same for random weights, so the softmax is not close to one-hot.
    q_norm, k_norm = _norm_weights(seed + 1, d, 2, scale=d**-0.25)
    return q, k, v if kind == "sliding" else None, o, q_norm, k_norm


def _attention_inputs(kind, T):
    """Normed input, RoPE tables and KV cache for a causal prefill (T=12) or a decode
    step at position 12 over a 12-token cache (T=1)."""
    _seed()
    x = _rms_norm(torch.randn(T, HIDDEN, dtype=torch.bfloat16))
    past = 0 if T == PREFILL_T else PREFILL_T
    cos, sin = _rope_cos_sin(torch.arange(past, past + T), kind)
    if not past:
        return x, cos, sin, None, None
    kv_heads, d, _, _ = _attention_dims(kind)
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED + 8000)
    # Cached keys are already normed and rotated; cached values are normed.
    past_k = (torch.randn(kv_heads, past, d, generator=g) * d**-0.25).to(torch.bfloat16)
    past_v = torch.randn(kv_heads, past, d, generator=g).to(torch.bfloat16)
    return x, cos, sin, past_k, past_v


def _dense_mlp_weights():
    """Gemma4TextMLP gate, up and down projections as [in, out]."""
    return _bf16_weights(
        SEED + 9000,
        (HIDDEN, DENSE_INTERMEDIATE),
        (HIDDEN, DENSE_INTERMEDIATE),
        (DENSE_INTERMEDIATE, HIDDEN),
    )


def _dense_mlp(x, gate_w, up_w, down_w):
    """Gemma4TextMLP: down(gelu_tanh(gate(x)) * up(x))."""
    act = torch.nn.functional.gelu(x @ gate_w, approximate="tanh")
    return (act * (x @ up_w)) @ down_w


DECODER_NORMS = (
    "input",
    "post_attention",
    "pre_feedforward",
    "post_feedforward_1",
    "pre_feedforward_2",
    "post_feedforward_2",
    "post_feedforward",
)


def _decoder_weights(kind):
    """Every tensor a decoder layer reads, in _decoder_layer's order."""
    return (
        *_attention_weights(kind),
        *_dense_mlp_weights(),
        *_router_params(),
        *_expert_bank_3d(),
        *_norm_weights(SEED + 10000, HIDDEN, len(DECODER_NORMS)),
    )


def _moe_fp32(hidden, weights, ids, gate_up, down):
    """FP32 reference for _hf_moe, one expert at a time."""
    bank = {e: (gate_up[e], down[e]) for e in torch.unique(ids).tolist()}
    return _moe_fp32_reference(hidden, ids, weights, bank)


def _decoder_layer(
    x,
    cos,
    sin,
    past_k,
    past_v,
    *weights,
    attend=_sdpa,
    moe=_hf_moe,
    route_ids=None,
    return_ids=False,
):
    """Gemma4TextDecoderLayer.forward. layer_scalar is 1 and there is no per-layer input.

    ``route_ids`` replaces the router's top-8 choice, so a reference can follow the
    same experts as the run it checks; the routing weights are still computed.
    """
    q, k, v, o, q_norm, k_norm = weights[:6]
    mlp_gate, mlp_up, mlp_down = weights[6:9]
    router_scale, router_proj, per_expert_scale = weights[9:12]
    experts_gate_up, experts_down = weights[12:14]
    norm = dict(zip(DECODER_NORMS, weights[14:]))

    residual = x
    h = _rms_norm(x, norm["input"])
    h = _gemma4_attention(
        h, cos, sin, q, k, v, o, q_norm, k_norm, past_k, past_v, attend=attend
    )
    h = residual + _rms_norm(h, norm["post_attention"])

    residual = h
    mlp = _dense_mlp(_rms_norm(h, norm["pre_feedforward"]), mlp_gate, mlp_up, mlp_down)
    mlp = _rms_norm(mlp, norm["post_feedforward_1"])
    # The router reads the residual before any feed-forward norm.
    route_w, route_ids = _gemma4_router(
        residual, router_scale, router_proj, per_expert_scale, ids=route_ids
    )
    experts = moe(
        _rms_norm(residual, norm["pre_feedforward_2"]),
        route_w,
        route_ids,
        experts_gate_up,
        experts_down,
    )
    experts = _rms_norm(experts, norm["post_feedforward_2"])
    out = residual + _rms_norm(mlp + experts, norm["post_feedforward"])
    return (out, route_ids) if return_ids else out


def _f32(tensors):
    return [t.float() if isinstance(t, torch.Tensor) else t for t in tensors]


# ---------------------------------------------------------------------------
# 3.1 Router (Gemma4TextRouter)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=TOPK_FLOAT_IDS, compiled=TOPK_WRONG_VALUES, decode_compiled=TOPK_FLOAT_IDS
    ),
)
def test_G4_RT_001_batch_routing(execution_mode, T):
    """Gemma 4-style FP32 router softmax followed by Top-K(K=8)."""
    _seed()

    x = torch.randn(
        T,
        EXPERTS,
        dtype=torch.bfloat16,
    )

    def fn(x):
        router_probabilities = torch.softmax(
            x,
            dim=-1,
            dtype=torch.float32,
        )
        values, ids = torch.topk(
            router_probabilities,
            8,
            dim=-1,
        )
        return values, ids

    _compare_mode(
        execution_mode,
        fn,
        x,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    ref_values, ref_ids = fn(x)
    assert ref_values.shape == (T, K8)
    assert ref_ids.shape == (T, K8)
    assert ref_values.dtype == torch.float32
    assert ref_ids.dtype == torch.long
    assert int(ref_ids.min()) >= 0 and int(ref_ids.max()) < 128


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=TOPK_FLOAT_IDS, compiled=TOPK_WRONG_VALUES, decode_compiled=TOPK_FLOAT_IDS
    ),
)
def test_G4_RT_002_routing_weight_normalization(execution_mode, T):
    """
    Verify Gemma 4 routing weights:

        softmax over all 128 experts
        -> top-8 expert selection
        -> renormalize selected weights
    """
    _seed()
    logits = _router_logits(T)

    def fn(x):
        return _routing_weights(x)

    _compare_mode(
        execution_mode,
        fn,
        logits,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    router_probabilities = torch.softmax(
        logits,
        dim=-1,
        dtype=torch.float32,
    )

    # Select the top-8 experts.
    ref_weights, ref_ids = torch.topk(
        router_probabilities,
        8,
        dim=-1,
    )

    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

    ref_ids = ref_ids.long()

    assert ref_weights.shape == (T, K8)
    assert ref_ids.shape == (T, K8)

    torch.testing.assert_close(
        ref_weights.sum(dim=-1),
        torch.ones(T, dtype=torch.float32),
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    got_weights, got_ids = fn(logits)

    torch.testing.assert_close(
        got_ids,
        ref_ids,
        atol=0,
        rtol=0,
    )

    torch.testing.assert_close(
        got_weights,
        ref_weights,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )


@pytest.mark.parametrize(("execution_mode", "T"), _cases(eager=INDEX_2D))
def test_G4_RT_003_per_expert_scale(execution_mode, T):
    """Scale renormalized weights per expert, as Gemma4TextRouter does."""
    weights, expert_ids = _routing_weights(_router_logits(T))
    per_expert_scale = _per_expert_scale()

    def fn(w, ids, scale):
        return w * scale[ids]

    _compare_mode(
        execution_mode,
        fn,
        weights,
        expert_ids,
        per_expert_scale,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    got = fn(weights, expert_ids, per_expert_scale)
    assert got.dtype == torch.float32
    for t, k in ((0, 0), (T - 1, K8 - 1)):
        e = int(expert_ids[t, k])
        assert got[t, k].item() == pytest.approx(
            weights[t, k].item() * per_expert_scale[e].item()
        )


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=FLOAT_IDS_AS_INDEX_EAGER, compiled=FLOAT_IDS_AS_INDEX),
)
def test_G4_RT_004_router_from_hidden_states(execution_mode, T):
    """Gemma4TextRouter: RMS norm, scale, proj, softmax, top-8, renorm, rescale."""
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    params = _router_params()

    def fn(x, scale, proj, per_expert_scale):
        return _gemma4_router(x, scale, proj, per_expert_scale)

    _compare_mode(execution_mode, fn, hidden, *params, atol=BF16_ATOL, rtol=BF16_RTOL)

    weights, ids = fn(hidden, *params)
    assert weights.shape == ids.shape == (T, K8)
    assert weights.dtype == torch.float32
    # Renormalized weights sum to 1 before the per-expert scale.
    torch.testing.assert_close(
        (weights / params[2][ids].float()).sum(dim=-1),
        torch.ones(T),
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )
    assert all(row.unique().numel() == K8 for row in ids)


# ---------------------------------------------------------------------------
# 3.2 MoE experts as Gemma 4 runs them (grouped_mm_experts_forward)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=STABLE_SORT,
        compiled=_xfail(
            InductorError,
            "#5831: split_multi_ops fails on multi-output sort",
            "'tuple' object has no attribute 'to'",
        ),
    ),
)
def test_G4_GM_001_sort_dispatch(execution_mode, T):
    """torch.sort by expert, then gather rows with perm // K."""
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights = torch.rand(T, K8)
    # Distinct ids, so the unstable sort has one valid order. GM_004 and GM_005
    # cover duplicate ids, where only the combined output is defined.
    ids = (torch.arange(T * K8) * 37 % EXPERTS).reshape(T, K8)

    _compare_mode(
        execution_mode,
        _hf_dispatch,
        hidden,
        weights,
        ids,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )

    expert_ids_g, perm, x_g, w_g = _hf_dispatch(hidden, weights, ids)
    assert torch.all(expert_ids_g[:-1] < expert_ids_g[1:])
    assert torch.equal(expert_ids_g, ids.reshape(-1)[perm])
    assert torch.equal(x_g, hidden.repeat_interleave(K8, dim=0)[perm])
    assert torch.equal(w_g, weights.reshape(-1)[perm])


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=_xfail(
            NotImplementedError, "#4071: aten::histc has no Spyre kernel", "aten::histc"
        ),
        compiled=_xfail(
            NotImplementedError, "#4071: aten::histc has no Spyre kernel", "aten::histc"
        ),
    ),
)
def test_G4_GM_002_histc_cumsum_offsets(execution_mode, T):
    """Per-expert row counts with histc, then int32 cumulative offsets."""
    _, ids = _routing_weights(_router_logits(T))
    expert_ids_g = torch.sort(ids.reshape(-1)).values

    _compare_mode(
        execution_mode,
        _hf_offsets,
        expert_ids_g,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )

    offsets = _hf_offsets(expert_ids_g)
    assert offsets.dtype == torch.int32
    assert offsets.shape == (EXPERTS,)
    counts = torch.bincount(expert_ids_g, minlength=EXPERTS)
    assert torch.equal(offsets, counts.cumsum(0).to(torch.int32))
    assert offsets[-1].item() == T * K8


@pytest.mark.parametrize(("execution_mode", "T"), _cases())
def test_G4_GM_003_grouped_mm_expert_ffn(execution_mode, T):
    """Both expert projections as _grouped_mm over all 128 experts."""
    _seed()
    x = torch.randn(T * K8, HIDDEN, dtype=torch.bfloat16)
    _, ids = _routing_weights(_router_logits(T))
    expert_ids_g = torch.sort(ids.reshape(-1)).values
    offsets = _hf_offsets(expert_ids_g)
    gate_up, down = _expert_bank_3d()

    _compare_mode(
        execution_mode,
        _hf_experts,
        x,
        offsets,
        gate_up,
        down,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = _hf_experts(x, offsets, gate_up, down)
    assert got.shape == (T * K8, HIDDEN)
    # Each row must go through its own expert's weights.
    for i in (0, T * K8 - 1):
        e = int(expert_ids_g[i])
        _assert_close(got[i : i + 1], _expert_ffn(x[i : i + 1], *_expert_slab(e)))


@pytest.mark.parametrize(
    ("execution_mode", "T"), _cases(eager=MIXED_EA, compiled=MIXED_EA)
)
def test_G4_GM_004_inverse_permute_fp32_combine(execution_mode, T):
    """Weight rows, restore token order and sum each token's 8 rows in FP32."""
    _seed()
    y_g = torch.randn(T * K8, HIDDEN, dtype=torch.bfloat16)
    weights, ids = _routing_weights(_router_logits(T))
    _, perm = torch.sort(ids.reshape(-1))
    w_g = weights.reshape(-1)[perm]

    def fn(y, w, p):
        return _hf_combine(y, w, p, T)

    _compare_mode(
        execution_mode,
        fn,
        y_g,
        w_g,
        perm,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    got = fn(y_g, w_g, perm)
    assert got.dtype == torch.float32
    ref = torch.zeros(T, HIDDEN)
    for i, row in enumerate(perm.tolist()):
        ref[row // K8] += y_g[i].float() * w_g[i]
    torch.testing.assert_close(got, ref, atol=FP32_ATOL, rtol=FP32_RTOL)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=FLOAT_IDS_AS_INDEX_EAGER, compiled=FLOAT_IDS_AS_INDEX),
)
def test_G4_GM_005_router_and_grouped_mm_moe(execution_mode, T):
    """Gemma4TextRouter followed by grouped_mm_experts_forward."""
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    params = _router_params()
    gate_up, down = _expert_bank_3d()

    def fn(x, scale, proj, per_expert_scale, gu, dn):
        weights, ids = _gemma4_router(x, scale, proj, per_expert_scale)
        return _hf_moe(x, weights, ids, gu, dn)

    _compare_mode(
        execution_mode,
        fn,
        hidden,
        *params,
        gate_up,
        down,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = fn(hidden, *params, gate_up, down)
    assert got.shape == (T, HIDDEN)
    assert got.dtype == torch.bfloat16
    weights, ids = _gemma4_router(hidden, *params)
    bank = _expert_weights(torch.unique(ids).tolist())
    ref = _moe_fp32_reference(hidden, ids, weights, bank)
    assert (got.float() - ref).abs().max().item() <= FP32_REF_ATOL


# ---------------------------------------------------------------------------
# 3.3 MoE numerics and routing edge cases (grouped_mm path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("execution_mode", "T"), _cases(**GROUPED_MM_SORT))
def test_G4_DT_002_fp32_reference_vs_bf16_execution(execution_mode, T):
    """grouped_mm MoE in BF16 stays within a justified bound of an FP32 reference."""
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights, ids = _routing_weights(_router_logits(T))
    gate_up, down = _expert_bank_3d()
    fp32_ref = _moe_fp32(hidden, weights, ids, gate_up, down)
    assert torch.isfinite(fp32_ref).all()

    # Spyre is checked against the independent FP32 reference, not the CPU
    # grouped_mm path, so it cannot pass by matching a bug in _hf_moe.
    _compare_mode(
        execution_mode,
        _hf_moe,
        hidden,
        weights,
        ids,
        gate_up,
        down,
        atol=FP32_REF_ATOL,
        rtol=0.0,
        cpu_eager_result=fp32_ref.to(torch.bfloat16),
    )

    got = _hf_moe(hidden, weights, ids, gate_up, down)
    assert got.dtype == torch.bfloat16
    assert (got.float() - fp32_ref).abs().max().item() <= FP32_REF_ATOL


def _run_case_with_ids(expert_ids, execution_mode):
    """Run the grouped_mm MoE with fixed routing; return CPU counts and offsets."""
    T, K = expert_ids.shape
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights = torch.full((T, K), 1.0 / K, dtype=torch.float32)
    gate_up, down = _expert_bank_3d()

    _compare_mode(
        execution_mode,
        _hf_moe,
        hidden,
        weights,
        expert_ids,
        gate_up,
        down,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = _hf_moe(hidden, weights, expert_ids, gate_up, down)
    ref = _moe_fp32(hidden, weights, expert_ids, gate_up, down)
    assert (got.float() - ref).abs().max().item() <= FP32_REF_ATOL
    flat = expert_ids.reshape(-1)
    return torch.bincount(flat, minlength=EXPERTS), _hf_offsets(torch.sort(flat).values)


@pytest.mark.parametrize(("execution_mode", "T"), _cases(**GROUPED_MM_SORT))
def test_G4_EC_001_repeated_expert_selection_within_token(execution_mode, T):
    # Each token picks one expert K times; tokens pick different experts.
    ids = (torch.arange(T) % EXPERTS).unsqueeze(1).expand(T, K8).contiguous()
    counts, offsets = _run_case_with_ids(ids, execution_mode)
    assert torch.all(counts[:T] == K8)
    assert counts[T:].sum().item() == 0
    assert offsets[T - 1].item() == offsets[-1].item() == T * K8


@pytest.mark.parametrize(("execution_mode", "T"), _cases(**GROUPED_MM_SORT))
def test_G4_EC_002_empty_expert_segments(execution_mode, T):
    """Experts 8-127 get no rows, so their grouped_mm segments are empty."""
    ids = torch.arange(K8, dtype=torch.long).expand(T, K8).contiguous()
    counts, offsets = _run_case_with_ids(ids, execution_mode)
    assert torch.all(counts[:K8] == T)
    assert counts[K8:].sum().item() == 0
    assert torch.all(offsets[K8 - 1 :] == T * K8)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=_xfail(
            AssertionError,
            "#4639: eager topk selects ids outside the tied pool",
            "Tensor-likes are not close",
        ),
        compiled=SORT_MULTI_OUTPUT,
        decode_eager=_xfail(
            InductorError,
            "#4404: restickify restore expects a padding-prepended gap dim",
            "restickify restore: expected padding-prepended",
        ),
    ),
)
def test_G4_EC_003_topk_tied_candidates(execution_mode, T):
    logits = torch.zeros(T, EXPERTS, dtype=torch.bfloat16)
    logits[:, :8] = 1.0

    def fn(x):
        router_probabilities = torch.softmax(
            x,
            dim=-1,
            dtype=torch.float32,
        )
        values, ids = torch.topk(
            router_probabilities,
            8,
            dim=-1,
        )
        return values, ids

    def tie_order_free(x):
        # topk does not define the order of tied values, so count the ids that
        # fall in the tied pool instead of comparing positions.
        values, ids = fn(x)
        return values, (ids < K8).float().sum(dim=-1)

    _compare_mode(
        execution_mode,
        tie_order_free,
        logits,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    values, ids = fn(logits)
    assert ids.shape == (T, K8)

    expected_pool = torch.arange(K8, dtype=torch.long)
    for row in ids:
        assert torch.equal(row.sort().values, expected_pool)
    # softmax of eight logits of 1 and 120 logits of 0.
    expected = math.e / (K8 * math.e + (EXPERTS - K8))
    torch.testing.assert_close(
        values,
        torch.full((T, K8), expected, dtype=torch.float32),
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    _run_case_with_ids(ids, execution_mode)


@pytest.mark.parametrize("execution_mode", _modes(**GROUPED_MM_SORT))
def test_G4_EC_004_all_128_experts_active(execution_mode):
    """16 tokens route to all 128 experts once each: every segment has one row."""
    ids = _deterministic_all_experts()
    assert torch.unique(ids).numel() == EXPERTS
    counts, offsets = _run_case_with_ids(ids, execution_mode)
    assert torch.all(counts == 1)
    assert torch.equal(offsets, torch.arange(1, EXPERTS + 1, dtype=torch.int32))


@pytest.mark.parametrize(("execution_mode", "T"), _cases(**GROUPED_MM_SORT))
def test_G4_EC_005_single_expert_routing_all_tokens(execution_mode, T):
    """Every routed row goes to expert 0; the other 127 segments are empty."""
    ids = torch.zeros(T, K8, dtype=torch.long)
    counts, offsets = _run_case_with_ids(ids, execution_mode)
    assert counts[0].item() == T * K8
    assert counts[1:].sum().item() == 0
    assert torch.all(offsets == T * K8)


# ---------------------------------------------------------------------------
# 3.4 Expert FFN and layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("execution_mode", "T"), _cases())
def test_G4_FFN_001_routed_rows_one_expert(execution_mode, T):
    _seed()
    x = torch.randn(T * K8, HIDDEN, dtype=torch.bfloat16)
    bank = _expert_weights([0])

    def fn(a, gu, down):
        return _expert_ffn(a, gu, down)

    _compare_mode(
        execution_mode,
        fn,
        x,
        *bank[0],
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    ref = fn(x, *bank[0])
    assert ref.shape == (T * K8, HIDDEN)
    assert ref.dtype == torch.bfloat16

    x2 = x.clone()
    x2[0, 0] += 1
    y2 = fn(x2, *bank[0])
    assert torch.equal(ref[1:].cpu(), y2[1:].cpu())


@pytest.mark.parametrize(("execution_mode", "T"), _cases())
def test_G4_LY_001_restickify_k8(execution_mode, T):
    """Production K=8 router-weight flattening and restickify path."""
    K = K8
    logits = _router_logits(T)
    weights, _ = _routing_weights(logits)

    def fn(w):
        return w.reshape(T * K)

    _compare_mode(
        execution_mode,
        fn,
        weights,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    flat = fn(weights)
    assert flat.shape == (T * K,)
    for t in range(T):
        assert torch.equal(flat[t * K : (t + 1) * K], weights[t])


# ---------------------------------------------------------------------------
# 3.5 Attention, dense MLP and decoder layer
# ---------------------------------------------------------------------------


def _attention_case(kind, execution_mode, T):
    x, cos, sin, past_k, past_v = _attention_inputs(kind, T)
    weights = _attention_weights(kind)

    def fn(h, c, s, pk, pv, *w):
        return _gemma4_attention(h, c, s, *w, pk, pv)

    _compare_mode(
        execution_mode,
        fn,
        x,
        cos,
        sin,
        past_k,
        past_v,
        *weights,
        atol=ATTN_ATOL,
        rtol=ATTN_RTOL,
    )

    got = fn(x, cos, sin, past_k, past_v, *weights)
    assert got.shape == (T, HIDDEN)
    assert got.dtype == torch.bfloat16
    ref = _gemma4_attention(
        *_f32((x, cos, sin, *weights, past_k, past_v)), attend=_manual_attention
    )
    assert (got.float() - ref).abs().max().item() <= ATTN_ATOL
    if past_k is None:
        # Causal prefill: the first token attends only to itself.
        first = fn(x[:1], cos[:1], sin[:1], None, None, *weights)
        _assert_close(got[:1], first, atol=ATTN_ATOL, rtol=ATTN_RTOL)
    return cos, sin, weights


@pytest.mark.parametrize(("execution_mode", "T"), _cases(eager=UPCAST_RMS_NORM))
def test_G4_AT_001_sliding_window_attention(execution_mode, T):
    """Sliding layer: 16 Q / 8 KV heads of 256, RoPE theta 1e4 over the full head."""
    _, _, weights = _attention_case("sliding", execution_mode, T)
    assert weights[2] is not None


@pytest.mark.parametrize(("execution_mode", "T"), _cases(eager=UPCAST_RMS_NORM))
def test_G4_AT_002_full_attention_k_eq_v(execution_mode, T):
    """Full layer: 2 KV heads of 512, V from k_proj, proportional RoPE theta 1e6."""
    cos, sin, weights = _attention_case("full", execution_mode, T)
    assert weights[2] is None
    # Proportional RoPE rotates 64 of the 256 frequency pairs; the rest pass through.
    rotated = int(GLOBAL_PARTIAL_ROTARY * GLOBAL_HEAD_DIM // 2)
    half = GLOBAL_HEAD_DIM // 2
    still = torch.cat(
        (torch.arange(rotated, half), torch.arange(half + rotated, 2 * half))
    )
    assert torch.all(cos[:, still] == 1)
    assert torch.all(sin[:, still] == 0)


@pytest.mark.parametrize(("execution_mode", "T"), _cases())
def test_G4_MLP_001_dense_mlp(execution_mode, T):
    """Gemma4TextMLP: 2816 -> 2112 -> 2816 with GELU tanh, beside the MoE."""
    _seed()
    x = _rms_norm(torch.randn(T, HIDDEN, dtype=torch.bfloat16))
    weights = _dense_mlp_weights()

    _compare_mode(execution_mode, _dense_mlp, x, *weights, atol=MLP_ATOL, rtol=MLP_RTOL)

    got = _dense_mlp(x, *weights)
    assert got.shape == (T, HIDDEN)
    assert got.dtype == torch.bfloat16
    ref = _dense_mlp(*_f32((x, *weights)))
    assert (got.float() - ref).abs().max().item() <= MLP_ATOL


def _decoder_case(kind, execution_mode, T):
    _, cos, sin, past_k, past_v = _attention_inputs(kind, T)
    _seed()
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights = _decoder_weights(kind)

    def fn(h, c, s, pk, pv, *w):
        return _decoder_layer(h, c, s, pk, pv, *w)

    _compare_mode(
        execution_mode,
        fn,
        x,
        cos,
        sin,
        past_k,
        past_v,
        *weights,
        atol=DECODER_ATOL,
        rtol=DECODER_RTOL,
    )

    got, ids = _decoder_layer(x, cos, sin, past_k, past_v, *weights, return_ids=True)
    assert got.shape == (T, HIDDEN)
    assert got.dtype == torch.bfloat16
    _assert_no_nan_inf(got)
    # BF16 and FP32 routers can pick different experts for near-tied tokens, so
    # the FP32 reference follows the experts the BF16 run chose.
    ref = _decoder_layer(
        *_f32((x, cos, sin, past_k, past_v, *weights)),
        attend=_manual_attention,
        moe=_moe_fp32,
        route_ids=ids,
    )
    assert (got.float() - ref).abs().max().item() <= DECODER_ATOL


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=FLOAT_IDS_AS_INDEX_EAGER, compiled=FLOAT_IDS_AS_INDEX),
)
def test_G4_E2E_002_decoder_layer_sliding(execution_mode, T):
    """Gemma4TextDecoderLayer with sliding-window attention, dense MLP and MoE."""
    _decoder_case("sliding", execution_mode, T)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=FLOAT_IDS_AS_INDEX_EAGER, compiled=FLOAT_IDS_AS_INDEX),
)
def test_G4_E2E_003_decoder_layer_full(execution_mode, T):
    """Gemma4TextDecoderLayer with full attention (K = V), dense MLP and MoE."""
    _decoder_case("full", execution_mode, T)


# ---------------------------------------------------------------------------
# 3.6 Tile-32 strategy (Spyre implementation option)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "execution_mode", _modes(eager=FLOOR_DIVIDE, compiled=TILE_OFFSET_CUMSUM)
)
def test_G4_AB_003_pad_expert_segments_to_tile32(execution_mode):
    counts = torch.tensor([0, 1, 31, 32, 33, 63, 64] + [0] * 121)

    def fn(c):
        tiles, padded, pad_off, tile_expert = _tile_metadata(c)
        return tiles, padded, pad_off, tile_expert

    _compare_mode(
        execution_mode,
        fn,
        counts,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )

    tiles, padded, pad_off, tile_expert = fn(counts)
    assert torch.all(padded % TILE == 0)
    assert torch.all(padded >= counts)
    assert pad_off.shape == (129,)
    assert pad_off[-1].item() % TILE == 0
    assert tile_expert.numel() == tiles.sum().item()
    # Each expert owns a contiguous run of exactly ``tiles[e]`` tiles.
    for e in range(EXPERTS):
        positions = torch.where(tile_expert == e)[0]
        assert positions.numel() == int(tiles[e])
        if positions.numel():
            assert torch.equal(positions, torch.arange(positions[0], positions[-1] + 1))


@pytest.mark.parametrize(
    "execution_mode",
    _modes(
        eager=_xfail(
            NotImplementedError,
            "#673: repeat_interleave.Tensor has no Spyre kernel",
            r"aten::repeat_interleave\.Tensor",
        ),
        compiled=_xfail(
            NotImplementedError,
            "#673: repeat_interleave.Tensor has no Spyre kernel",
            r"aten::repeat_interleave\.Tensor",
        ),
    ),
)
def test_G4_AB_008_exact_dst_pos_construction(execution_mode):
    counts = torch.zeros(EXPERTS, dtype=torch.long)
    counts[0] = 1
    counts[1] = 31
    counts[2] = 32
    counts[3] = 33
    counts[4] = 64
    counts[5] = 17
    group_off = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    _, padded, pad_off, _ = _tile_metadata(counts)

    def fn(c, go, po):
        return _dst_pos(c, go, po)

    _compare_mode(
        execution_mode,
        fn,
        counts,
        group_off,
        pad_off,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )

    dst = fn(counts, group_off, pad_off)
    n_pad = int(pad_off[-1])
    assert dst.shape == (int(counts.sum()),)
    assert torch.unique(dst).numel() == dst.numel()
    assert int(dst.min()) >= 0 and int(dst.max()) < n_pad
    for e in range(EXPERTS):
        c = int(counts[e])
        if c:
            s = int(group_off[e])
            p = int(pad_off[e])
            assert int(dst[s]) == p
            assert torch.equal(dst[s : s + c], torch.arange(p, p + c))
    assert n_pad % TILE == 0
    assert torch.all(padded % TILE == 0)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=_xfail(
            NotImplementedError,
            "#692: aten::_index_put_impl_ has no Spyre kernel",
            "aten::_index_put_impl_",
        ),
        compiled=_xfail(
            InductorError,
            "#4327: restickify op_spec has 3 args",
            "restickify op_spec has 3 args",
        ),
    ),
)
def test_G4_AB_006_pad_activations_weights_token_ids(execution_mode, T):
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights, expert_ids = _routing_weights(_router_logits(T))
    routed, row_w, token, expert = _flatten_routing(hidden, weights, expert_ids)
    perm, _, counts, group_off = _group(expert)
    _, _, pad_off, _ = _tile_metadata(counts)
    n_pad = int(pad_off[-1])
    dst = _dst_pos(counts, group_off, pad_off)

    def fn(x, w, t, d, n):
        return _pad_columns(x, w, t, d, n, hidden.shape[0])

    _compare_mode(
        execution_mode,
        fn,
        routed[perm],
        row_w[perm],
        token[perm],
        dst,
        n_pad,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    xpad, wpad, tpad = fn(routed[perm], row_w[perm], token[perm], dst, n_pad)
    assert torch.equal(xpad[dst], routed[perm])
    assert torch.equal(wpad[dst], row_w[perm])
    assert torch.equal(tpad[dst], token[perm])

    # Expert segments start on pad_off, with padding rows between them.
    active = counts > 0
    assert torch.all(pad_off[:-1][active] % TILE == 0)
    assert torch.equal(dst[group_off[:-1][active]], pad_off[:-1][active])

    padding = torch.ones(n_pad, dtype=torch.bool)
    padding[dst] = False
    assert padding.any()
    assert torch.count_nonzero(xpad[padding]) == 0
    assert torch.count_nonzero(wpad[padding]) == 0
    assert torch.all(tpad[padding] == T)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=GROUPED_MOE_EAGER, compiled=GROUPED_MOE_COMPILED, **SORT_AT_DECODE),
)
def test_G4_AB_007_per_tile_static_loop_scatter_add(execution_mode, T):
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights, expert_ids = _routing_weights(_router_logits(T))
    used = torch.unique(expert_ids).tolist()
    bank = _expert_weights(used)

    def fn(x, ids, w):
        return _grouped_moe(x, ids, w, bank)

    _compare_mode(
        execution_mode,
        fn,
        hidden,
        expert_ids,
        weights,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got, meta = fn(hidden, expert_ids, weights)
    ref = _moe_reference(hidden, expert_ids, weights, bank)
    assert got.shape == (T, HIDDEN)
    assert got.dtype == torch.bfloat16
    _assert_close(got, ref)
    _assert_no_nan_inf(got)

    perm, counts, group_off, pad_off, tile_expert, dst = meta
    assert perm.numel() == T * K8
    assert group_off.shape == (129,)
    assert pad_off[-1].item() % TILE == 0
    assert tile_expert.numel() == pad_off[-1].item() // TILE
    assert dst.numel() == T * K8


@pytest.mark.parametrize(
    "execution_mode", _modes(eager=INDEX_ADD_DUPLICATES, compiled=INDEX_ADD_DUPLICATES)
)
def test_G4_BE_002_fixed_tile_sink_row_scatter_reduce(execution_mode):
    """Scatter-reduce one fixed 32-row tile into destination rows."""
    _seed()
    T = 8
    seg_out = torch.randn(TILE, HIDDEN, dtype=torch.bfloat16)
    dst = torch.tensor(
        [0, 0, 1, 2, 2, 2, T, T] + [3] * 24,
        dtype=torch.long,
    )

    def fn(seg, indices):
        out = torch.zeros(T + 1, HIDDEN, dtype=seg.dtype, device=seg.device)
        return out.index_add(0, indices, seg)

    _compare_mode(
        execution_mode,
        fn,
        seg_out,
        dst,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    out = fn(seg_out, dst)
    # Accumulate in FP32 and cast once, as index_add does not round per row.
    ref = torch.zeros(T + 1, HIDDEN, dtype=torch.float32)
    for i, d in enumerate(dst.tolist()):
        ref[d] += seg_out[i].float()
    _assert_close(out, ref.to(out.dtype))
    ref_sink = seg_out[6].float() + seg_out[7].float()
    _assert_close(
        out[-1].unsqueeze(0),
        ref_sink.unsqueeze(0).to(out.dtype),
    )
