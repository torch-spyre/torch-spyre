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

"""Gemma 4 26B-A4B-it MoE functional verification tests for Torch-Spyre."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
T64 = 64
T8 = 8
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 256
ROTARY_DIM = 64
RMS_NORM_EPS = 1e-6


EXACT_ATOL = 0.0
EXACT_RTOL = 0.0
FP32_ATOL = float(os.getenv("GEMMA_FVT_FP32_ATOL", "0.001"))
FP32_RTOL = float(os.getenv("GEMMA_FVT_FP32_RTOL", "0.001"))
BF16_ATOL = float(os.getenv("GEMMA_FVT_BF16_ATOL", "0.005"))
BF16_RTOL = float(os.getenv("GEMMA_FVT_BF16_RTOL", "0.005"))
SEED = 20260908


def _compare_mode(execution_mode, fn, *args, atol=BF16_ATOL, rtol=BF16_RTOL):
    if execution_mode == "eager":
        compare_with_cpu(
            fn,
            *args,
            atol=atol,
            rtol=rtol,
            run_compile=False,
            run_eager=True,
            cpu_compile=False,
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
        )
        return

    raise ValueError(f"Unsupported execution mode: {execution_mode}")


def _seed() -> None:
    torch.manual_seed(SEED)


def _assert_exact(actual, expected, name="tensor"):
    """Exact comparison for integer/index/structural metadata."""
    assert torch.equal(actual, expected), (
        f"{name} mismatch: actual={actual}, expected={expected}"
    )


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


def _route(logits: torch.Tensor):
    values, indices = torch.topk(logits, K8, dim=-1)
    return values.float(), indices.long()


def _routing_weights(logits: torch.Tensor):
    values, indices = _route(logits)
    weights = torch.softmax(values, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), indices


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


def _pad_columns(routed_sorted, weight_sorted, token_sorted, n_pad, sink):
    n = routed_sorted.shape[0]
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
    out_x[:n] = routed_sorted
    out_w[:n] = weight_sorted
    out_t[:n] = token_sorted
    return out_x, out_w, out_t


def _expert_weights(expert_ids, *, dtype=torch.bfloat16, device="cpu"):
    """Create only the expert slabs needed by a scenario."""
    ids = sorted({int(x) for x in expert_ids})
    banks = {}
    for e in ids:
        g = torch.Generator(device="cpu")
        g.manual_seed(SEED + e)
        gate_up = torch.randn(HIDDEN, GATE_UP, generator=g, dtype=torch.float32) * 0.01
        down = (
            torch.randn(MOE_INTERMEDIATE, HIDDEN, generator=g, dtype=torch.float32)
            * 0.01
        )
        banks[e] = (gate_up.to(dtype).to(device), down.to(dtype).to(device))
    return banks


def _expert_ffn(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor):
    gu = x @ gate_up
    gate, up = gu.split(MOE_INTERMEDIATE, dim=-1)
    return (torch.nn.functional.gelu(gate) * up) @ down


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
        routed, row_w, token, n_pad, hidden.shape[0]
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


def _deterministic_all_experts(T=64, K=8):
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
# 3.1 Router / Top-K Selection
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Tracked by #4639: K=8 routing produces incorrect results on Spyre"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_RT_001_batch_routing(execution_mode):
    """64 tokens independently select exactly 8 experts."""
    logits = _router_logits(T64)

    def fn(x):
        values, ids = torch.topk(x, K8, dim=-1)
        return values.float(), ids

    _compare_mode(execution_mode, fn, logits, atol=BF16_ATOL, rtol=BF16_RTOL)

    ref_values, ref_ids = fn(logits)
    assert ref_values.shape == (T64, K8)
    assert ref_ids.shape == (T64, K8)
    assert ref_ids.dtype == torch.long
    assert int(ref_ids.min()) >= 0 and int(ref_ids.max()) < EXPERTS


# ---------------------------------------------------------------------------
# 3.3 Routed-row formation / grouping / dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_AB_001_routed_row_permutation_integrity(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #4500: stable sort (aten::sort.values_stable) unsupported"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4650: compiled advanced indexing fails with unsupported multi-arg pointwise layout"
        )
    """One expert-sort permutation must preserve all routed-row columns."""
    _seed()
    T, K = T64, K8
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


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_AB_002_bincount_segment_boundaries(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #4500: stable sort (aten::sort.values_stable) unsupported"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4650: compiled advanced indexing fails with unsupported multi-arg pointwise layout"
        )
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
    assert counts.sum().item() == T64 * K8
    for e in range(EXPERTS):
        s, t = group_off[e].item(), group_off[e + 1].item()
        assert t - s == counts[e].item()
        if counts[e] == 0:
            assert s == t
        else:
            assert torch.all(sorted_e[s:t] == e)


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_AB_005_reorder_activations_weights_token_ids(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    _seed()
    hidden = torch.randn(T64, HIDDEN, dtype=torch.bfloat16)
    weights, expert_ids = _routing_weights(_router_logits(T64))

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
    assert got_x.shape == (T64 * K8, HIDDEN)
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


@pytest.mark.skip(reason="Tracked by #673: aten::repeat_interleave unsupported")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
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


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_AB_003_pad_expert_segments_to_tile32(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #4643: aten::floor_divide unsupported for tile metadata construction"
        )
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


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_AB_004_tile_to_expert_assignment(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #4643: aten::floor_divide unsupported for tile metadata construction"
        )
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


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_AB_006_pad_activations_weights_token_ids(execution_mode):
    _seed()
    hidden = torch.randn(T64, HIDDEN, dtype=torch.bfloat16)
    weights, expert_ids = _routing_weights(_router_logits(T64))
    routed, row_w, token, expert = _flatten_routing(hidden, weights, expert_ids)
    perm = torch.argsort(expert, stable=True)
    counts = torch.bincount(expert[perm], minlength=EXPERTS)
    _, _, pad_off, _ = _tile_metadata(counts)
    n_pad = int(pad_off[-1])

    def fn(x, w, t, n):
        return _pad_columns(x, w, t, n, hidden.shape[0])

    _compare_mode(
        execution_mode,
        fn,
        routed[perm],
        row_w[perm],
        token[perm],
        n_pad,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    xpad, wpad, tpad = fn(routed[perm], row_w[perm], token[perm], n_pad)
    n_real = T64 * K8
    assert torch.equal(xpad[:n_real], routed[perm])
    assert torch.equal(wpad[:n_real], row_w[perm])
    assert torch.equal(tpad[:n_real], token[perm])
    assert torch.count_nonzero(xpad[n_real:]) == 0
    assert torch.count_nonzero(wpad[n_real:]) == 0
    assert torch.all(tpad[n_real:] == T64)


# ---------------------------------------------------------------------------
# 3.5 Expert FFN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_FFN_002_512_pairs_one_expert(execution_mode):
    _seed()
    x = torch.randn(T64 * K8, HIDDEN, dtype=torch.bfloat16)
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
    assert ref.shape == (T64 * K8, HIDDEN)
    assert ref.dtype == torch.bfloat16

    x2 = x.clone()
    x2[0, 0] += 1
    y2 = fn(x2, *bank[0])
    assert torch.equal(ref[1:].cpu(), y2[1:].cpu())


@pytest.mark.skip(
    reason="Tracked by #4645: aten::_unique2 unsupported during multi-expert routing"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_FFN_003_512_pairs_across_8_experts(execution_mode):
    _seed()
    T, K = T64, K8
    expert_ids = (torch.arange(T).unsqueeze(1) + torch.arange(K)) % K8
    expert_ids = expert_ids.long()
    x = torch.randn(T64 * K8, HIDDEN, dtype=torch.bfloat16)
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
    assert ref.shape == (T64 * K8, HIDDEN)


# ---------------------------------------------------------------------------
# 3.6 Weighted combine / Approach-B static tile loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_AB_007_per_tile_static_loop_scatter_add(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4648: restickify padding fails for Gemma 4 MoE routed tensors"
        )
    _seed()
    hidden = torch.randn(T64, HIDDEN, dtype=torch.bfloat16)
    weights, expert_ids = _routing_weights(_router_logits(T64))
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
    assert got.shape == (T64, HIDDEN)
    _assert_close(got, ref)
    _assert_no_nan_inf(got)

    perm, counts, group_off, pad_off, tile_expert, dst = meta
    assert perm.numel() == T64 * K8
    assert group_off.shape == (129,)
    assert pad_off[-1].item() % TILE == 0
    assert tile_expert.numel() == pad_off[-1].item() // TILE
    assert dst.numel() == T64 * K8


# ---------------------------------------------------------------------------
# 3.7 Backend / layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_LY_002_restickify_k8(execution_mode):
    """Production K=8 router-weight flattening and restickify path."""
    T, K = T64, K8
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
    assert flat.shape == (512,)
    for t in range(T):
        assert torch.equal(flat[t * K : (t + 1) * K], weights[t])


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
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


@pytest.mark.skip(reason="Tracked by #3507: Enable Tensor.index_add_ / aten::index_add")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_BE_002_fixed_tile_sink_row_scatter_reduce(execution_mode):
    _seed()
    T = 8
    seg_out = torch.randn(TILE, HIDDEN, dtype=torch.bfloat16)
    dst = torch.tensor(
        [0, 0, 1, 2, 2, 2, T, T] + [3] * 24,
        dtype=torch.long,
    )

    def fn(seg, indices):
        out = torch.zeros(T + 1, HIDDEN, dtype=seg.dtype)
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
    ref = torch.zeros_like(out)
    for i, d in enumerate(dst.tolist()):
        ref[d] += seg_out[i]
    _assert_close(out, ref)
    ref_sink = seg_out[6].float() + seg_out[7].float()
    _assert_close(
        out[-1].unsqueeze(0),
        ref_sink.unsqueeze(0).to(out.dtype),
    )


# ---------------------------------------------------------------------------
# 3.8 End-to-end Gemma MoE / transformer integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_E2E_001_moe_t8(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4648: restickify padding fails for Gemma 4 MoE routed tensors"
        )
    _seed()
    T = T8
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
    assert got.shape == (8, HIDDEN)
    _assert_close(got, ref)
    _assert_no_nan_inf(got)


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_E2E_002_transformer_block_attention_moe(execution_mode):
    if execution_mode == "eager":
        pytest.skip("Tracked by #3179: aten::pow unimplemented")
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4443: unexpected stick expression during compiled MoE execution"
        )
    """Gemma-style Transformer block: attention -> residual -> MoE -> residual."""
    _seed()
    T = T8
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


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_DT_001_bf16_end_to_end_numerical(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4648: restickify padding fails for Gemma 4 MoE routed tensors"
        )
    _seed()
    hidden = torch.randn(T64, HIDDEN, dtype=torch.bfloat16)
    weights, ids = _routing_weights(_router_logits(T64))
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
    assert got.shape == (T64, HIDDEN)
    _assert_no_nan_inf(got)
    _assert_close(got, ref)


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_DT_002_fp32_reference_vs_bf16_execution(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4648: restickify padding fails for Gemma 4 MoE routed tensors"
        )
    _seed()
    hidden = torch.randn(T64, HIDDEN, dtype=torch.bfloat16)
    weights, ids = _routing_weights(_router_logits(T64))
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
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    got = fn(hidden, ids, weights)
    diff = got.float() - fp32_ref
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    max_rel = (diff.abs() / fp32_ref.abs().clamp_min(1e-8)).max().item()

    assert math.isfinite(max_abs)
    assert math.isfinite(mean_abs)
    assert math.isfinite(max_rel)


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


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_EC_001_repeated_expert_selection_within_token(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4648: restickify padding fails for Gemma 4 MoE routed tensors"
        )
    ids = torch.zeros(T8, K8, dtype=torch.long)
    meta = _run_case_with_ids(ids, execution_mode)
    counts = meta[1]
    assert counts[0].item() == T8 * K8
    assert counts[1:].sum().item() == 0


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_EC_002_empty_expert_segments(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4648: restickify padding fails for Gemma 4 MoE routed tensors"
        )
    ids = torch.arange(K8, dtype=torch.long).expand(T8, K8)
    meta = _run_case_with_ids(ids, execution_mode)
    counts = meta[1]
    assert counts[:K8].sum().item() == T8 * K8
    assert counts[K8:].sum().item() == 0
    _, _, _, pad_off, _, _ = meta
    assert pad_off[-1].item() % TILE == 0


@pytest.mark.skip(
    reason="Tracked by #4649: Top-K tie handling returns incorrect expert IDs"
)
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_EC_003_topk_tied_candidates(execution_mode):
    T = 8
    logits = torch.zeros(T, EXPERTS, dtype=torch.bfloat16)
    logits[:, :8] = 1.0

    def fn(x):
        values, ids = torch.topk(x, K8, dim=-1)
        return values.float(), ids

    _compare_mode(
        execution_mode,
        fn,
        logits,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )

    values, ids = fn(logits)
    assert ids.shape == (T, K8)

    expected_pool = torch.arange(K8, dtype=torch.long)
    for row in ids:
        assert torch.equal(row.sort().values, expected_pool)
    assert torch.all(values == 1.0)

    _run_case_with_ids(ids, execution_mode)


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_EC_004_all_128_experts_active(execution_mode):
    ids = _deterministic_all_experts()
    assert torch.unique(ids).numel() == EXPERTS
    counts = torch.bincount(ids.reshape(-1), minlength=EXPERTS)
    assert counts.shape == (EXPERTS,)
    assert torch.all(counts == T64 * K8 // EXPERTS)

    fingerprints = torch.arange(EXPERTS, dtype=torch.float32)

    def fn(x):
        return fingerprints[x]

    selected = fn(ids)
    assert selected.shape == ids.shape
    assert torch.equal(
        torch.bincount(ids.reshape(-1), minlength=EXPERTS),
        counts,
    )

    if execution_mode == "eager":
        pytest.skip(
            "#1219: CPU tensor cannot be indexed with Spyre tensor in eager execution"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "#4650: compiled advanced indexing fails with "
            "unsupported multi-arg pointwise layout"
        )

    _compare_mode(
        execution_mode,
        fn,
        ids,
        atol=EXACT_ATOL,
        rtol=EXACT_RTOL,
    )


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
def test_G4_EC_005_single_expert_routing_all_tokens(execution_mode):
    if execution_mode == "eager":
        pytest.skip(
            "Tracked by #3193: pointwise layout propagation cannot resolve stick incompatibility in Gemma 4 MoE"
        )
    if execution_mode == "compiled":
        pytest.skip(
            "Tracked by #4648: restickify padding fails for Gemma 4 MoE routed tensors"
        )
    ids = torch.zeros(T64, K8, dtype=torch.long)
    meta = _run_case_with_ids(ids, execution_mode)
    counts = meta[1]
    assert counts[0].item() == T64 * K8
    assert counts[1:].sum().item() == 0
    assert meta[4].numel() == T64 * K8 // TILE
