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

Shapes come from a traced Gemma 4 26B-A4B run (Transformers v5.15.0): a 12-token
prefill and 1-token decode steps, so 96 and 8 routed rows. Two MoE paths are
covered:

* 3.11 follows Transformers' ``grouped_mm_experts_forward``, the path the model
  runs: sort, ``perm // K`` gather, ``histc``/``cumsum`` offsets, ``_grouped_mm``
  and an FP32 ``view(T, K, H).sum(1)`` combine.
* 3.3 to 3.10 cover a Spyre implementation strategy (Approach B): expert segments
  padded to 32-row tiles, with an ``index_add`` combine. The model does not run it.

Attention is simplified: it covers only the 256-dim sliding-window layers, rotates
the first 64 features, scales by ``1/sqrt(head_dim)`` and has no Q/K norm. Expert
weights are synthetic.

Every scenario also runs with ``execution_mode="cpu"``. That case uses no Spyre
device and is never xfailed, so a bug in a reference or an assertion cannot hide
behind a backend xfail.
"""

from __future__ import annotations

import functools
import math
import os
import sys

import pytest
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
NUM_KV_HEADS = 8
HEAD_DIM = 256
ROTARY_DIM = 64
RMS_NORM_EPS = 1e-6


EXACT_ATOL = 0.0
EXACT_RTOL = 0.0
FP32_ATOL = 0.001
FP32_RTOL = 0.001
BF16_ATOL = 0.005
BF16_RTOL = 0.005
FP32_REF_ATOL = 0.002
SEED = 20260908


def _compare_mode(execution_mode, fn, *args, atol=BF16_ATOL, rtol=BF16_RTOL, **kwargs):
    if execution_mode == "cpu":
        return

    if execution_mode == "eager":
        compare_with_cpu(
            fn,
            *args,
            atol=atol,
            rtol=rtol,
            run_compile=False,
            run_eager=True,
            cpu_compile=False,
            **kwargs,
        )
        return

    if execution_mode == "compiled":
        compare_with_cpu(
            fn,
            *args,
            atol=atol,
            rtol=rtol,
            run_compile=True,
            run_eager=False,
            cpu_compile=True,
            **kwargs,
        )
        return

    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _xfail(raises, reason):
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


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
)
GROUPED_MOE_COMPILED = _xfail(
    InductorError,
    "#4648: restickify padding finds no input host dim for the new stick symbol",
)
STABLE_SORT = _xfail(
    NotImplementedError, "#4500: aten::sort.values_stable has no Spyre kernel"
)
FLOOR_DIVIDE = _xfail(
    NotImplementedError, "#4643: aten::floor_divide has no Spyre kernel"
)
TILE_OFFSET_CUMSUM = _xfail(
    InductorError, "#4754: no offset-free alternative stick dim for mutation target"
)
TOPK_FLOAT_IDS = _xfail(AssertionError, "#5847: topk returns float32 indices")
TOPK_WRONG_VALUES = _xfail(
    AssertionError, "#4639: compiled K=8 topk returns incorrect values"
)
UNIQUE = _xfail(NotImplementedError, "#4645: aten::_unique2 has no Spyre kernel")
INDEX_2D = _xfail(
    NotImplementedError, "#1219: aten::index.Tensor supports only a 1-D index"
)
INDEX_ADD_DUPLICATES = _xfail(
    AssertionError, "#4411: index_add keeps the last duplicate instead of summing"
)
# At T=1 the routing flatten succeeds and the grouped path stops at the stable sort.
SORT_AT_DECODE = dict(decode_eager=STABLE_SORT, decode_compiled=STABLE_SORT)
MIXED_EA = _xfail(
    InductorError,
    "#4714: BF16 x FP32 broadcast multiply raises mixed EA with no staggered operand",
)
FLOAT_IDS_AS_INDEX = _xfail(
    RuntimeError, "#5847: float32 topk ids cannot index per_expert_scale"
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
def _expert_slab(e):
    """Return the deterministic (gate_up, down) BF16 slabs of expert ``e``."""
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED + e)
    gate_up = torch.randn(HIDDEN, GATE_UP, generator=g, dtype=torch.float32) * 0.01
    down = (
        torch.randn(MOE_INTERMEDIATE, HIDDEN, generator=g, dtype=torch.float32) * 0.01
    )
    return gate_up.to(torch.bfloat16), down.to(torch.bfloat16)


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


def _assert_group_metadata(expert_ids):
    flat = expert_ids.reshape(-1)
    perm, sorted_e, counts, group_off = _group(flat)
    assert perm.numel() == flat.numel()
    assert torch.equal(torch.sort(perm).values, torch.arange(flat.numel()))
    assert torch.all(sorted_e[:-1] <= sorted_e[1:])
    assert counts.shape == (EXPERTS,)
    assert group_off.shape == (EXPERTS + 1,)
    assert group_off[0].item() == 0
    assert group_off[-1].item() == flat.numel()
    return perm, sorted_e, counts, group_off


def _multi_expert_ffn(x, expert_ids, bank):
    flat_ids = expert_ids.reshape(-1)
    out = torch.empty_like(x)
    for e in torch.unique(flat_ids).tolist():
        e = int(e)
        mask = flat_ids == e
        ge, de = bank[e]
        if ge.device != x.device:
            ge, de = ge.to(x.device), de.to(x.device)
        out[mask] = _expert_ffn(x[mask], ge, de)
    return out


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + RMS_NORM_EPS)).to(x.dtype)


def _rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply Gemma rotary position embedding to the first ROTARY_DIM features."""
    T = x.shape[-2]
    positions = torch.arange(T, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=x.device)
            / ROTARY_DIM
        )
    )
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, :, :]
    sin = angles.sin()[None, :, :]
    rotary = x[..., :ROTARY_DIM]
    even = rotary[..., 0::2]
    odd = rotary[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    return torch.cat((rotated, x[..., ROTARY_DIM:]), dim=-1)


def _gemma_attention(
    x: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
) -> torch.Tensor:
    """Small Gemma-style attention path: 16 Q heads, 8 KV heads, 256-D heads."""
    q = (x @ q_proj).reshape(x.shape[0], NUM_Q_HEADS, HEAD_DIM).transpose(0, 1)
    k = (x @ k_proj).reshape(x.shape[0], NUM_KV_HEADS, HEAD_DIM).transpose(0, 1)
    v = (x @ v_proj).reshape(x.shape[0], NUM_KV_HEADS, HEAD_DIM).transpose(0, 1)

    k = k.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=0)
    v = v.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=0)

    q = _rope(q, theta=1_000_000.0)
    k = _rope(k, theta=1_000_000.0)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    probs = torch.softmax(scores, dim=-1)
    context = torch.matmul(probs, v.float())
    context = context.transpose(0, 1).reshape(x.shape[0], NUM_Q_HEADS * HEAD_DIM)
    return (context @ o_proj.float()).to(x.dtype)


def _transformer_block(x, expert_ids, weights, bank, q_proj, k_proj, v_proj, o_proj):
    normed = _rms_norm(x)
    attn = _gemma_attention(normed, q_proj, k_proj, v_proj, o_proj)
    after_attn = x + attn
    moe_input = _rms_norm(after_attn)
    moe, _ = _grouped_moe(moe_input, expert_ids, weights, bank)
    return after_attn + moe


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


def _gemma4_router(hidden, scale, proj, per_expert_scale):
    """Gemma4TextRouter.forward: weights [T, 8] FP32 and expert ids [T, 8]."""
    x = _rms_norm(hidden)
    x = x * scale * HIDDEN**-0.5
    logits = x @ proj.t()
    weights, ids = _routing_weights(logits)
    return weights * per_expert_scale[ids], ids


@functools.lru_cache(maxsize=None)
def _expert_bank_3d():
    """All 128 experts as _grouped_mm takes them: [128, 2816, 1408], [128, 704, 2816]."""
    gate_up = torch.stack([_expert_slab(e)[0] for e in range(EXPERTS)])
    down = torch.stack([_expert_slab(e)[1] for e in range(EXPERTS)])
    return gate_up, down


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
# 3.1 Router / Top-K Selection / Routing Weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=TOPK_FLOAT_IDS, compiled=TOPK_WRONG_VALUES, decode_compiled=TOPK_FLOAT_IDS
    ),
)
def test_G4_RT_001_batch_routing(execution_mode, T):
    """Gemma 4-style FP32 router softmax followed by Top-K(K=8)."""
    torch.manual_seed(20260908)

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
    ("execution_mode", "T"), _cases(eager=MIXED_EA, compiled=FLOAT_IDS_AS_INDEX)
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
# 3.3 Routed-row formation / grouping / dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution_mode", "T"), _cases(eager=STABLE_SORT, compiled=STABLE_SORT)
)
def test_G4_AB_001_routed_row_permutation_integrity(execution_mode, T):
    """One expert-sort permutation must preserve all routed-row columns."""
    _seed()
    K = K8
    expert = (torch.arange(T * K) * 37 % EXPERTS).long()
    token = torch.arange(T).repeat_interleave(K)
    weight = torch.arange(T * K, dtype=torch.float32) / 1000
    activation = torch.arange(T * K * HIDDEN, dtype=torch.int64).reshape(T * K, HIDDEN)

    def fn(e, t, w, a):
        perm = torch.argsort(e, stable=True)
        return perm, e[perm], t[perm], w[perm], a[perm]

    _compare_mode(
        execution_mode,
        fn,
        expert,
        token,
        weight,
        activation,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )

    perm, sorted_e, sorted_t, sorted_w, sorted_a = fn(expert, token, weight, activation)
    assert torch.equal(torch.sort(perm).values, torch.arange(T * K))
    assert torch.all(sorted_e[:-1] <= sorted_e[1:])

    assert torch.equal(sorted_t, token[perm])
    assert torch.equal(sorted_w, weight[perm])
    assert torch.equal(sorted_a, activation[perm])


@pytest.mark.parametrize(
    "execution_mode", _modes(eager=STABLE_SORT, compiled=STABLE_SORT)
)
def test_G4_AB_002_bincount_segment_boundaries(execution_mode):
    expert = _deterministic_all_experts().reshape(-1)

    def fn(e):
        _, sorted_e, counts, group_off = _group(e)
        return sorted_e, counts, group_off

    _compare_mode(
        execution_mode,
        fn,
        expert,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )

    sorted_e, counts, group_off = fn(expert)
    _, expected_sorted_e, expected_counts, expected_group_off = _assert_group_metadata(
        expert
    )
    assert torch.equal(sorted_e, expected_sorted_e)
    assert torch.equal(counts, expected_counts)
    assert torch.equal(group_off, expected_group_off)
    assert counts.sum().item() == ALL_EXPERTS_T * K8
    for e in range(EXPERTS):
        s, t = group_off[e].item(), group_off[e + 1].item()
        assert t - s == counts[e].item()
        if counts[e] == 0:
            assert s == t
        else:
            assert torch.all(sorted_e[s:t] == e)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=_xfail(
            InductorError, "#3193: no mechanism to resolve stick incompatibility"
        ),
        compiled=_xfail(
            InductorError, "#3193: no mechanism to resolve stick incompatibility"
        ),
        **SORT_AT_DECODE,
    ),
)
def test_G4_AB_005_reorder_activations_weights_token_ids(execution_mode, T):
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights, expert_ids = _routing_weights(_router_logits(T))

    def fn(x, w, ids):
        routed, row_w, token, expert = _flatten_routing(x, w, ids)
        perm = torch.argsort(expert, stable=True)
        return (
            routed[perm],
            row_w[perm],
            token[perm],
            expert[perm],
        )

    _compare_mode(
        execution_mode,
        fn,
        hidden,
        weights,
        expert_ids,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got_x, got_w, got_t, sorted_e = fn(hidden, weights, expert_ids)
    assert got_x.shape == (T * K8, HIDDEN)
    _, expected_w, expected_t, expected_expert = _flatten_routing(
        hidden, weights, expert_ids
    )
    expected_perm = torch.argsort(expected_expert, stable=True)
    assert torch.equal(got_t, expected_t[expected_perm])
    assert torch.equal(got_w, expected_w[expected_perm])

    _, _, _, expert_flat = _flatten_routing(hidden, weights, expert_ids)
    assert torch.all(sorted_e[:-1] <= sorted_e[1:])
    assert torch.equal(
        torch.sort(sorted_e).values,
        torch.sort(expert_flat).values,
    )


@pytest.mark.parametrize(
    "execution_mode",
    _modes(
        eager=_xfail(
            NotImplementedError, "#673: repeat_interleave.Tensor has no Spyre kernel"
        ),
        compiled=_xfail(
            NotImplementedError, "#673: repeat_interleave.Tensor has no Spyre kernel"
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


# ---------------------------------------------------------------------------
# 3.4 Tile preparation
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


@pytest.mark.parametrize(
    "execution_mode", _modes(eager=FLOOR_DIVIDE, compiled=TILE_OFFSET_CUMSUM)
)
def test_G4_AB_004_tile_to_expert_assignment(execution_mode):
    counts = torch.zeros(EXPERTS, dtype=torch.long)
    counts[[0, 2, 7, 127]] = torch.tensor([1, 33, 64, 32])

    def fn(c):
        tiles, _, _, tile_expert = _tile_metadata(c)
        return tiles, tile_expert

    _compare_mode(
        execution_mode,
        fn,
        counts,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )

    tiles, tile_expert = fn(counts)
    assert tile_expert.numel() == tiles.sum().item()
    for e in range(EXPERTS):
        expected = int(tiles[e])
        actual = int((tile_expert == e).sum())
        assert actual == expected
        if expected:
            positions = torch.where(tile_expert == e)[0]
            assert torch.equal(
                positions,
                torch.arange(positions[0], positions[-1] + 1),
            )


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=_xfail(
            NotImplementedError, "#692: aten::_index_put_impl_ has no Spyre kernel"
        ),
        compiled=_xfail(InductorError, "#4327: restickify op_spec has 3 args"),
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


# ---------------------------------------------------------------------------
# 3.5 Expert FFN
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


@pytest.mark.parametrize(("execution_mode", "T"), _cases(eager=UNIQUE, compiled=UNIQUE))
def test_G4_FFN_002_routed_rows_across_8_experts(execution_mode, T):
    _seed()
    expert_ids = (torch.arange(T).unsqueeze(1) + torch.arange(K8)) % K8
    expert_ids = expert_ids.long()
    x = torch.randn(T * K8, HIDDEN, dtype=torch.bfloat16)
    bank = _expert_weights(range(K8))

    def fn(a, ids):
        return _multi_expert_ffn(a, ids, bank)

    _compare_mode(
        execution_mode,
        fn,
        x,
        expert_ids,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    ref = fn(x, expert_ids)
    assert ref.shape == (T * K8, HIDDEN)


# ---------------------------------------------------------------------------
# 3.6 Weighted combine / Approach B static tile loop
# ---------------------------------------------------------------------------


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
    _assert_close(got, ref)
    _assert_no_nan_inf(got)

    perm, counts, group_off, pad_off, tile_expert, dst = meta
    assert perm.numel() == T * K8
    assert group_off.shape == (129,)
    assert pad_off[-1].item() % TILE == 0
    assert tile_expert.numel() == pad_off[-1].item() // TILE
    assert dst.numel() == T * K8


# ---------------------------------------------------------------------------
# 3.7 Backend / layout
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("execution_mode", _modes())
def test_G4_BE_001_per_tile_expert_slab_select(execution_mode):
    _seed()
    tile_expert = torch.tensor([0, 7, 31, 127], dtype=torch.long)
    bank = _expert_weights(tile_expert.tolist())
    x = torch.randn(32, HIDDEN, dtype=torch.bfloat16)

    def fn(a, expert_id):
        outputs = []
        for e in expert_id.tolist():
            gu, down = bank[int(e)]
            if gu.device != a.device:
                gu, down = gu.to(a.device), down.to(a.device)
            outputs.append(_expert_ffn(a, gu, down))
        return torch.stack(outputs)

    _compare_mode(
        execution_mode,
        fn,
        x,
        tile_expert,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = fn(x, tile_expert)
    assert got.shape == (4, 32, HIDDEN)


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


# ---------------------------------------------------------------------------
# 3.8 End-to-end Gemma MoE / transformer integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=GROUPED_MOE_EAGER, compiled=GROUPED_MOE_COMPILED, **SORT_AT_DECODE),
)
def test_G4_E2E_001_moe_layer(execution_mode, T):
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    logits = _router_logits(T)
    weights, expert_ids = _routing_weights(logits)
    bank = _expert_weights(torch.unique(expert_ids).tolist())

    def fn(x, ids, w):
        return _grouped_moe(x, ids, w, bank)[0]

    _compare_mode(
        execution_mode,
        fn,
        hidden,
        expert_ids,
        weights,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = fn(hidden, expert_ids, weights)
    ref = _moe_reference(hidden, expert_ids, weights, bank)
    assert got.shape == (T, HIDDEN)
    _assert_close(got, ref)
    _assert_no_nan_inf(got)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=_xfail(RuntimeError, "#3179: rope calls pow, an unimplemented operation"),
        compiled=_xfail(InductorError, "#4443: unexpected stick expression"),
    ),
)
def test_G4_E2E_002_transformer_block_attention_moe(execution_mode, T):
    _seed()
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights, ids = _routing_weights(_router_logits(T))

    bank = _expert_weights(torch.unique(ids).tolist())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 4000)
    q_proj = (
        torch.randn(HIDDEN, NUM_Q_HEADS * HEAD_DIM, generator=gen, dtype=torch.float32)
        * 0.01
    ).to(torch.bfloat16)
    k_proj = (
        torch.randn(HIDDEN, NUM_KV_HEADS * HEAD_DIM, generator=gen, dtype=torch.float32)
        * 0.01
    ).to(torch.bfloat16)
    v_proj = (
        torch.randn(HIDDEN, NUM_KV_HEADS * HEAD_DIM, generator=gen, dtype=torch.float32)
        * 0.01
    ).to(torch.bfloat16)
    o_proj = (
        torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN, generator=gen, dtype=torch.float32)
        * 0.01
    ).to(torch.bfloat16)

    def fn(a, b, c, q, k, v, o):
        return _transformer_block(a, b, c, bank, q, k, v, o)

    _compare_mode(
        execution_mode,
        fn,
        x,
        ids,
        weights,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = fn(x, ids, weights, q_proj, k_proj, v_proj, o_proj)
    assert got.shape == (T, HIDDEN)
    assert got.dtype == torch.bfloat16
    _assert_no_nan_inf(got)

    normed = _rms_norm(x)
    attn = _gemma_attention(normed, q_proj, k_proj, v_proj, o_proj)
    after_attn = x + attn
    moe_input = _rms_norm(after_attn)
    ref_moe = _moe_reference(moe_input, ids, weights, bank)
    ref = after_attn + ref_moe
    _assert_close(got, ref)


# ---------------------------------------------------------------------------
# 3.9 Numerical / dtype correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=GROUPED_MOE_EAGER, compiled=GROUPED_MOE_COMPILED, **SORT_AT_DECODE),
)
def test_G4_DT_001_bf16_end_to_end_numerical(execution_mode, T):
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights, ids = _routing_weights(_router_logits(T))
    bank = _expert_weights(torch.unique(ids).tolist())

    def fn(x, i, w):
        return _grouped_moe(x, i, w, bank)[0]

    _compare_mode(
        execution_mode,
        fn,
        hidden,
        ids,
        weights,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = fn(hidden, ids, weights)
    ref = _moe_reference(hidden, ids, weights, bank)
    assert got.dtype == torch.bfloat16
    assert got.shape == (T, HIDDEN)
    _assert_no_nan_inf(got)
    _assert_close(got, ref)


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=GROUPED_MOE_EAGER, compiled=GROUPED_MOE_COMPILED, **SORT_AT_DECODE),
)
def test_G4_DT_002_fp32_reference_vs_bf16_execution(execution_mode, T):
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights, ids = _routing_weights(_router_logits(T))
    bank = _expert_weights(torch.unique(ids).tolist())
    fp32_ref = _moe_fp32_reference(hidden, ids, weights, bank)
    assert torch.isfinite(fp32_ref).all()

    def fn(x, i, w):
        return _grouped_moe(x, i, w, bank)[0]

    _compare_mode(
        execution_mode,
        fn,
        hidden,
        ids,
        weights,
        atol=FP32_REF_ATOL,
        rtol=0.0,
        cpu_eager_result=fp32_ref.to(torch.bfloat16),
    )

    got = fn(hidden, ids, weights)
    assert (got.float() - fp32_ref).abs().max().item() <= FP32_REF_ATOL


# ---------------------------------------------------------------------------
# 3.10 Routing / grouping edge cases
# ---------------------------------------------------------------------------


def _run_case_with_ids(expert_ids, execution_mode):
    T, K = expert_ids.shape
    _seed()
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    weights = torch.full((T, K), 1.0 / K, dtype=torch.float32)
    bank = _expert_weights(torch.unique(expert_ids).tolist())

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
    _assert_close(got, ref)
    return meta


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=GROUPED_MOE_EAGER, compiled=GROUPED_MOE_COMPILED, **SORT_AT_DECODE),
)
def test_G4_EC_001_repeated_expert_selection_within_token(execution_mode, T):
    # Each token picks one expert K times; tokens pick different experts.
    ids = (torch.arange(T) % EXPERTS).unsqueeze(1).expand(T, K8).contiguous()
    meta = _run_case_with_ids(ids, execution_mode)
    counts = meta[1]
    assert torch.all(counts[:T] == K8)
    assert counts[T:].sum().item() == 0


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=GROUPED_MOE_EAGER, compiled=GROUPED_MOE_COMPILED, **SORT_AT_DECODE),
)
def test_G4_EC_002_empty_expert_segments(execution_mode, T):
    ids = torch.arange(K8, dtype=torch.long).expand(T, K8)
    meta = _run_case_with_ids(ids, execution_mode)
    counts = meta[1]
    assert counts[:K8].sum().item() == T * K8
    assert counts[K8:].sum().item() == 0
    _, _, _, pad_off, _, _ = meta
    assert pad_off[-1].item() % TILE == 0


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=_xfail(
            AssertionError, "#4639: eager topk selects ids outside the tied pool"
        ),
        compiled=GROUPED_MOE_COMPILED,
        decode_eager=_xfail(
            InductorError,
            "#4404: restickify restore expects a padding-prepended gap dim",
        ),
        decode_compiled=STABLE_SORT,
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
        atol=FP32_ATOL * expected,
        rtol=FP32_RTOL,
    )

    _run_case_with_ids(ids, execution_mode)


@pytest.mark.parametrize("execution_mode", _modes(eager=INDEX_2D, compiled=INDEX_2D))
def test_G4_EC_004_all_128_experts_active(execution_mode):
    ids = _deterministic_all_experts()
    assert torch.unique(ids).numel() == EXPERTS
    counts = torch.bincount(ids.reshape(-1), minlength=EXPERTS)
    assert counts.shape == (EXPERTS,)
    assert torch.all(counts == ALL_EXPERTS_T * K8 // EXPERTS)

    fingerprints = torch.arange(EXPERTS, dtype=torch.float32)

    def fn(x):
        return fingerprints[x]

    selected = fn(ids)
    assert selected.shape == ids.shape
    assert torch.equal(
        torch.bincount(ids.reshape(-1), minlength=EXPERTS),
        counts,
    )

    _compare_mode(
        execution_mode,
        fn,
        ids,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(eager=GROUPED_MOE_EAGER, compiled=GROUPED_MOE_COMPILED, **SORT_AT_DECODE),
)
def test_G4_EC_005_single_expert_routing_all_tokens(execution_mode, T):
    ids = torch.zeros(T, K8, dtype=torch.long)
    meta = _run_case_with_ids(ids, execution_mode)
    counts = meta[1]
    assert counts[0].item() == T * K8
    assert counts[1:].sum().item() == 0
    assert meta[4].numel() == math.ceil(T * K8 / TILE)


# ---------------------------------------------------------------------------
# 3.11 Transformers grouped_mm path (what Gemma 4 runs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution_mode", "T"),
    _cases(
        eager=STABLE_SORT,
        compiled=_xfail(
            InductorError, "#5831: split_multi_ops fails on multi-output sort"
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
        eager=_xfail(NotImplementedError, "#4071: aten::histc has no Spyre kernel"),
        compiled=_xfail(NotImplementedError, "#4071: aten::histc has no Spyre kernel"),
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
    ("execution_mode", "T"), _cases(eager=MIXED_EA, compiled=FLOAT_IDS_AS_INDEX)
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
