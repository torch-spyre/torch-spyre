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

"""spyre::linear_attention: the algorithm, the op's contract, and the op on device.

Four levels, in the order a failure is worth reading:

  1. ALGORITHM -- on CPU, no device. ``spyre_linear_attention`` (the real
     decomposition, which only needs plain torch ops, ``for_each_tile`` and the
     CPU-built mask) against a token-by-token fp32 recurrence that shares no
     structure with it, and the recurrence against the O(L^2) masked form so the
     yardstick itself is checked. fp32 on both sides, so tolerances are tight.
     It also covers the feature map against ``F.elu`` and that the chunk loop
     traces to ONE ``scan`` whatever the sequence length.

  2. CONTRACT -- on CPU/meta, no device. Defaults, the tuple return, and that
     every unsupported call raises ``Unsupported`` from both the op's fake
     (trace time) and the decomposition, with a message naming the problem.

  3. THE OP -- compiled on Spyre against the CPU reference, on prepared Q/K/V.

  4. THE WHOLE PATH -- compiled on Spyre: the feature map, the normalizer, the op
     and the state hand-off between a prefill call and single-token decode
     calls, against an independent float64 CPU reference that uses ``F.elu``.
     Level 3 cannot see a feature-map failure because it never applies one.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_linear_attention.py -v
"""

import unittest

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F
from torch._inductor.codecache import FxGraphCache
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx

from torch_spyre._inductor.decompositions import spyre_linear_attention
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.linear_attention import (
    MAX_CHUNK_SIZE,
    check_linear_attention,
    chunkwise_linear_attention,
    default_scale,
    reference_linear_attention,
)
from torch_spyre._inductor.sliding_window_plan import STICK
from utils_inductor import DEVICE, cached_randn, compare_with_cpu

# fp32 CPU tolerance. Across the shape grid below (L up to 300, Dk/Dv up to 128,
# chunk 64-256) the worst error of the chunked form against the recurrence is
# 4.8e-5 on the output and 6.1e-5 on the state, both about 7e-7 of the largest
# value. 5e-4 is ~10x that, which is accumulation-order slack and nothing more.
# At 129 chunks (L = 8256) an fp32 emulation of the same algorithm stays within
# 1.5e-4 of the recurrence on output and state, so the same tolerance holds.
FP32_ATOL = 5e-4
FP32_RTOL = 1e-4

# fp16 device tolerance. A CPU half-precision run of this same chunked algorithm
# against the fp32 recurrence (L up to 512, D up to 128, outputs reaching ~12)
# was within 0.0078 on both the output and the state. 0.05 leaves ~6x for the
# device's accumulation order. This is calibrated on that CPU emulation, not on
# hardware; tighten it from device results rather than loosening it.
FP16_ATOL = 0.05
FP16_RTOL = 0.05

# fp16 tolerance for a feature map alone. The output is fp16, so a value of ~21
# rounds by up to ~8e-3 and one near 1 by up to ~5e-4. The device's ``exp`` is an
# approximation whose error is not characterized here, so 1e-2 is ~20 half-ulps at
# 1. Calibrated by reasoning, not on hardware; tighten it from device results.
FEATURE_MAP_ATOL = 1e-2
FEATURE_MAP_RTOL = 1e-2

# The chunk-count limit the loop had while it was unrolled into the graph. The
# limit is gone; the tests below run past it to show that.
FORMER_UNROLL_LIMIT = 128

# Normalizer epsilon, as in docs/source/user_guide/examples/linear_attention.py.
EPS = 1e-3
HEAD_DIM = 64


def feature_map(x):
    """``elu(x) + 1`` (alpha = 1) from operations Spyre already supports.

    Identical to the feature map in the linear attention example. ``F.elu`` does
    not compile for Spyre because Inductor decomposes it through ``expm1``.
    """
    return torch.exp(torch.minimum(x, torch.zeros_like(x))) + torch.relu(x)


# Inputs that put every branch of ``elu(x) + 1`` under test: negative, near zero
# (both signs, and exactly zero) and positive. The wide set adds values whose
# unclamped ``exp`` would overflow fp16 (``exp(12)`` is 1.6e5; the fp16 maximum is
# 65504) and one that underflows it.
FEATURE_MAP_PROBES = (-8.0, -2.0, -0.5, -1e-2, -1e-3, 0.0, 1e-3, 1e-2, 0.5, 1.0, 2.0)
FEATURE_MAP_WIDE_PROBES = (
    -20.0,
    -8.0,
    -4.0,
    -1.0,
    -0.25,
    -1e-2,
    -1e-3,
    0.0,
    1e-3,
    1e-2,
    0.25,
    1.0,
    4.0,
    12.0,
    20.0,
)


def _inputs(batch, heads, seqlen, key_dim=64, value_dim=64, dtype=torch.float16):
    """Q and V are unit variance; K is scaled by ``Dk ** -0.5`` so the state,
    a sum of ``L`` products, stays well inside fp16 range."""
    query = cached_randn(
        (batch, heads, seqlen, key_dim), differentiation=1, dtype=dtype
    )
    key = cached_randn(
        (batch, heads, seqlen, key_dim),
        differentiation=2,
        dtype=dtype,
        scale=key_dim**-0.5,
    )
    value = cached_randn(
        (batch, heads, seqlen, value_dim), differentiation=3, dtype=dtype
    )
    return query, key, value


def _state(batch, heads, key_dim=64, value_dim=64, dtype=torch.float16):
    return cached_randn(
        (batch, heads, key_dim, value_dim), differentiation=4, dtype=dtype, scale=0.5
    )


def _quadratic_form(query, key, value, scale, is_causal, state=None):
    """The O(L^2) definition in float64: ``((Q K^T) * M) V``."""
    q = query.double() * scale
    k, v = key.double(), value.double()
    scores = torch.einsum("bhlk,bhmk->bhlm", q, k)
    if is_causal:
        length = query.size(2)
        scores = scores * torch.tril(torch.ones(length, length, dtype=torch.double))
    out = torch.einsum("bhlm,bhmv->bhlv", scores, v)
    if state is not None:
        out = out + torch.einsum("bhlk,bhkv->bhlv", q, state.double())
    return out


def _assert_fp32_close(actual, expected, what):
    torch.testing.assert_close(
        actual, expected, atol=FP32_ATOL, rtol=FP32_RTOL, msg=lambda m: f"{what}\n{m}"
    )


def _assert_fp16_close(actual, expected, what):
    """Compare on the host: the device has no kernel for the check itself."""
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.float(),
        atol=FP16_ATOL,
        rtol=FP16_RTOL,
        msg=lambda m: f"{what}\n{m}",
    )


# --------------------------------------------------------------------------- #
# Level 1: the algorithm, on CPU.                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("with_state", [False, True], ids=["fresh", "seeded"])
def test_reference_matches_quadratic_form(is_causal, with_state):
    """The yardstick is itself right: recurrence == O(L^2) masked form."""
    query, key, value = _inputs(2, 3, 96, 64, 128, dtype=torch.float32)
    state = _state(2, 3, 64, 128, dtype=torch.float32) if with_state else None
    scale = default_scale(64)

    output, final_state = reference_linear_attention(
        query, key, value, state, scale, is_causal
    )

    expected = _quadratic_form(query, key, value, scale, is_causal, state)
    torch.testing.assert_close(output.double(), expected, atol=1e-4, rtol=1e-4)
    # The final state is the seed plus the sum of every k^T v.
    summed = torch.einsum("bhlk,bhlv->bhkv", key.double(), value.double())
    if state is not None:
        summed = summed + state.double()
    torch.testing.assert_close(final_state.double(), summed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "is_causal,chunk_size,seqlen",
    [
        (True, 64, 64),  # a single chunk: no loop, the body runs once
        (True, 64, 256),  # several chunks: a for_each_tile loop
        (True, 128, 256),
        (True, 64, 100),  # ragged: padded to 128 and sliced back
        (True, 256, 300),  # ragged against a large chunk
        (True, 64, 1),  # one token, padded to a whole chunk
        (True, 64, 64 * (FORMER_UNROLL_LIMIT + 1)),  # past the old unrolled limit
        (False, 64, 256),
        (False, 128, 100),
    ],
    ids=lambda v: str(v),
)
def test_decomposition_matches_reference(is_causal, chunk_size, seqlen):
    # Dv != Dk so a transposed or swapped state cannot hide.
    query, key, value = _inputs(2, 2, seqlen, 64, 128, dtype=torch.float32)

    output, state = spyre_linear_attention(
        query, key, value, None, chunk_size, is_causal
    )
    ref_output, ref_state = reference_linear_attention(
        query, key, value, None, None, is_causal
    )

    assert output.shape == (2, 2, seqlen, 128)
    assert state.shape == (2, 2, 64, 128)
    _assert_fp32_close(output, ref_output, "output")
    _assert_fp32_close(state, ref_state, "state")


@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
def test_initial_state_is_honoured(is_causal):
    query, key, value = _inputs(1, 2, 192, 64, 64, dtype=torch.float32)
    seed = _state(1, 2, dtype=torch.float32)

    output, state = spyre_linear_attention(query, key, value, seed, 64, is_causal)
    ref_output, ref_state = reference_linear_attention(
        query, key, value, seed, None, is_causal
    )

    _assert_fp32_close(output, ref_output, "output")
    _assert_fp32_close(state, ref_state, "state")
    # And a seed must actually change the answer, or the test proves nothing.
    unseeded, _ = spyre_linear_attention(query, key, value, None, 64, is_causal)
    assert (output - unseeded).abs().max() > 1e-2


def test_scale_defaults_to_inverse_sqrt_key_dim_and_is_linear():
    query, key, value = _inputs(1, 2, 128, 64, 64, dtype=torch.float32)

    unit_out, _ = spyre_linear_attention(query, key, value, None, 64, True, 1.0)
    default_out, _ = spyre_linear_attention(query, key, value)
    scaled_out, _ = spyre_linear_attention(query, key, value, None, 64, True, 2.0)

    # The output is linear in the query scale, so every other scale follows from
    # the unit-scale result. The default must be Dk ** -0.5.
    _assert_fp32_close(default_out, unit_out * default_scale(64), "default scale")
    _assert_fp32_close(scaled_out, unit_out * 2.0, "scale=2.0")


def test_state_threading_matches_one_shot():
    """Chunked prefill: feeding one call's state into the next continues the
    sequence exactly."""
    query, key, value = _inputs(1, 2, 256, 64, 64, dtype=torch.float32)

    whole_out, whole_state = spyre_linear_attention(query, key, value, None, 64, True)
    first_out, first_state = spyre_linear_attention(
        query[:, :, :128], key[:, :, :128], value[:, :, :128], None, 64, True
    )
    second_out, second_state = spyre_linear_attention(
        query[:, :, 128:], key[:, :, 128:], value[:, :, 128:], first_state, 64, True
    )

    _assert_fp32_close(torch.cat([first_out, second_out], dim=2), whole_out, "output")
    _assert_fp32_close(second_state, whole_state, "state")


def test_decode_steps_match_full_sequence():
    """Serving: prefill a ragged prompt, then decode one token per call. Each
    single-token call is padded to a whole chunk, and neither the padding nor
    the repeated state hand-off may leak into the result."""
    prompt, steps = 65, 3
    query, key, value = _inputs(1, 2, prompt + steps, 64, 64, dtype=torch.float32)

    out, state = spyre_linear_attention(
        query[:, :, :prompt], key[:, :, :prompt], value[:, :, :prompt], None, 64, True
    )
    rows = [out]
    for t in range(prompt, prompt + steps):
        row, state = spyre_linear_attention(
            query[:, :, t : t + 1],
            key[:, :, t : t + 1],
            value[:, :, t : t + 1],
            state,
            64,
            True,
        )
        rows.append(row)

    ref_out, ref_state = reference_linear_attention(query, key, value)
    _assert_fp32_close(torch.cat(rows, dim=2), ref_out, "decoded output")
    _assert_fp32_close(state, ref_state, "decoded state")


def test_chunk_size_does_not_change_the_result():
    """Chunking is a schedule, not semantics: the answer is independent of it."""
    query, key, value = _inputs(1, 2, 100, 64, 64, dtype=torch.float32)

    small, small_state = spyre_linear_attention(query, key, value, None, 64, True)
    large, large_state = spyre_linear_attention(query, key, value, None, 256, True)

    _assert_fp32_close(small, large, "output")
    _assert_fp32_close(small_state, large_state, "state")


@pytest.mark.parametrize(
    "height,width,diagonal",
    [(64, 64, 0), (64, 64, -1), (64, 64, 3), (4, 7, 0), (7, 4, 1)],
    ids=lambda v: str(v),
)
def test_tril_mask_matches_torch_tril(height, width, diagonal):
    mask = torch.ops.spyre.tril_mask(
        height, width, diagonal, torch.float16, torch.device("cpu")
    )
    expected = torch.tril(torch.ones(height, width), diagonal).to(torch.float16)
    assert torch.equal(mask, expected)


def _trace_chunk_loop(num_chunks, chunk_size=64):
    """Trace the chunk recurrence the way an AOT decomposition reaches it."""
    query, key, value = _inputs(
        1, 2, num_chunks * chunk_size, 64, 64, dtype=torch.float32
    )
    mask = torch.tril(torch.ones(chunk_size, chunk_size))
    args = (query, key, value, mask)

    def recurrence(q, k, v, m):
        return chunkwise_linear_attention(
            q, k, v, chunk_size=chunk_size, intra_mask=m, scale=default_scale(64)
        )

    return make_fx(recurrence, tracing_mode="fake")(*args), args


def _graph_modules(gm):
    return [m for m in gm.modules() if isinstance(m, torch.fx.GraphModule)]


def _scan_nodes(gm):
    return [
        node
        for m in _graph_modules(gm)
        for node in m.graph.nodes
        if node.target is torch.ops.higher_order.scan
    ]


def test_chunk_loop_traces_to_one_scan_whatever_the_length():
    """The chunk iteration is a loop in the traced graph, not one copy of the
    body per chunk: the graph has one ``scan`` and its size does not depend on
    the number of chunks, including past the old unrolled limit of 128."""
    counts = {}
    for num_chunks in (4, FORMER_UNROLL_LIMIT + 1):
        gm, _ = _trace_chunk_loop(num_chunks)
        assert len(_scan_nodes(gm)) == 1, f"{num_chunks} chunks"
        counts[num_chunks] = [len(list(m.graph.nodes)) for m in _graph_modules(gm)]
    assert counts[4] == counts[FORMER_UNROLL_LIMIT + 1]


def test_single_chunk_traces_without_a_loop():
    gm, _ = _trace_chunk_loop(1)
    assert _scan_nodes(gm) == []


@pytest.mark.parametrize("num_chunks", [1, 4], ids=lambda n: f"{n}_chunks")
def test_traced_chunk_loop_matches_reference(num_chunks):
    """The traced graph, run eagerly, still computes the recurrence."""
    gm, args = _trace_chunk_loop(num_chunks)
    query, key, value, _ = args

    output, state = gm(*args)

    ref_output, ref_state = reference_linear_attention(query, key, value)
    _assert_fp32_close(output, ref_output, "output")
    _assert_fp32_close(state, ref_state, "state")


def _probe_input(probes, shape, differentiation, dtype):
    """Random values with ``probes`` written over the first elements."""
    # cached_randn returns one shared tensor per argument set; clone before writing.
    flat = cached_randn(
        (int(torch.Size(shape).numel()),),
        differentiation=differentiation,
        dtype=dtype,
        scale=3.0,
    ).clone()
    flat[: len(probes)] = torch.tensor(probes, dtype=dtype)
    return flat.view(shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_feature_map_matches_elu_plus_one(dtype):
    """The Spyre-safe formula is ``elu(x) + 1``: negative, near-zero, positive."""
    x = _probe_input(FEATURE_MAP_WIDE_PROBES, (64, 64), 32, dtype)

    torch.testing.assert_close(feature_map(x), F.elu(x) + 1.0, atol=1e-6, rtol=1e-6)


def test_feature_map_does_not_overflow_in_fp16():
    """``exp`` is only ever applied to values <= 0, so a large positive input
    cannot overflow in the branch that is not selected."""
    x = torch.tensor([12.0, 20.0, 88.0], dtype=torch.float16)

    # The inputs must overflow an unclamped exp, or this test proves nothing.
    assert torch.exp(x).isinf().all()
    assert torch.equal(feature_map(x), x + 1.0)


def _raw_inputs(batch, heads, seqlen, dtype=torch.float16):
    """Q/K before the feature map, with every region of ``elu(x) + 1`` present in
    every row (the probes sit in the first columns), and unit-variance V."""
    shape = (batch, heads, seqlen, HEAD_DIM)
    probes = torch.tensor(FEATURE_MAP_PROBES, dtype=dtype)
    # cached_randn returns one shared tensor per argument set; clone before writing.
    q_raw = cached_randn(shape, differentiation=21, dtype=dtype).clone()
    k_raw = cached_randn(shape, differentiation=22, dtype=dtype).clone()
    q_raw[..., : probes.numel()] = probes
    k_raw[..., : probes.numel()] = probes
    value = cached_randn(shape, differentiation=23, dtype=dtype)
    return q_raw, k_raw, value


def _normalized_linear_attention(q_raw, k_raw, v, initial_state):
    """Compiled-region body, as in the linear attention example: feature map,
    normalizer from a block of ones appended to V, the op, then the division.

    ``k`` is scaled by ``Dk ** -0.5`` so the fp16 state stays in range.
    """
    q = feature_map(q_raw)
    k = feature_map(k_raw) * k_raw.size(-1) ** -0.5
    ones = torch.ones(*v.shape[:-1], STICK, dtype=v.dtype, device=v.device)
    v_aug = torch.cat([v, ones], dim=-1)
    out_aug, state = torch.ops.spyre.linear_attention(
        q, k, v_aug, initial_state, 64, True, 1.0
    )
    value_dim = v.size(-1)
    numerator = out_aug[..., :value_dim]
    denominator = out_aug[..., value_dim : value_dim + 1]  # every column is equal
    return numerator / (denominator + EPS), state


def _reference_queries(q_raw):
    """The feature map written as ``F.elu`` in float64, so it shares no code with
    the device formulation."""
    return F.elu(q_raw.double()) + 1.0


def _reference_keys(k_raw):
    """As ``_reference_queries``, with the ``Dk ** -0.5`` key scale applied."""
    return (F.elu(k_raw.double()) + 1.0) * k_raw.size(-1) ** -0.5


def _reference_normalized(q_raw, k_raw, v):
    """Normalized causal linear attention in float64, as the quadratic form."""
    q, k = _reference_queries(q_raw), _reference_keys(k_raw)
    length = q.size(2)
    scores = torch.einsum("bhlk,bhmk->bhlm", q, k)
    scores = scores * torch.tril(torch.ones(length, length, dtype=torch.double))
    return (scores @ v.double()) / (scores.sum(-1, keepdim=True) + EPS)


def _reference_state(k_raw, v):
    """The ``[B, H, Dk, Dv + STICK]`` state after the whole sequence: ``sum
    phi(k)^T v`` and, in the appended block, ``sum phi(k)`` in every column."""
    k = _reference_keys(k_raw)
    kv = torch.einsum("bhlk,bhlv->bhkv", k, v.double())
    normalizer = k.sum(2).unsqueeze(-1).expand(-1, -1, -1, STICK)
    return torch.cat([kv, normalizer], dim=-1)


def test_normalized_path_on_cpu_matches_reference():
    """The compiled-region body's logic, on CPU where the op runs the reference
    recurrence: feature map, appended ones for the normalizer, prefill, then
    single-token decode through the carried state. This checks the pieces the
    device test composes and the independent float64 reference itself."""
    prompt, steps = 100, 3
    q_raw, k_raw, v = _raw_inputs(1, 2, prompt + steps, dtype=torch.float32)

    out, state = _normalized_linear_attention(
        q_raw[:, :, :prompt], k_raw[:, :, :prompt], v[:, :, :prompt], None
    )
    _assert_fp32_close(
        state,
        _reference_state(k_raw[:, :, :prompt], v[:, :, :prompt]).float(),
        "state after prefill",
    )
    rows = [out]
    for t in range(prompt, prompt + steps):
        row, state = _normalized_linear_attention(
            q_raw[:, :, t : t + 1], k_raw[:, :, t : t + 1], v[:, :, t : t + 1], state
        )
        rows.append(row)

    _assert_fp32_close(
        torch.cat(rows, dim=2),
        _reference_normalized(q_raw, k_raw, v).float(),
        "normalized output",
    )
    _assert_fp32_close(state, _reference_state(k_raw, v).float(), "final state")


# --------------------------------------------------------------------------- #
# Level 2: the op's contract, on CPU and meta.                                #
# --------------------------------------------------------------------------- #


def test_cpu_op_defaults_and_return_shape():
    query, key, value = _inputs(1, 2, 128, 64, 128, dtype=torch.float32)

    result = torch.ops.spyre.linear_attention(query, key, value)

    assert len(result) == 2
    output, state = result
    assert output.shape == (1, 2, 128, 128)
    assert state.shape == (1, 2, 64, 128)
    ref_output, ref_state = reference_linear_attention(query, key, value)
    _assert_fp32_close(output, ref_output, "output")
    _assert_fp32_close(state, ref_state, "state")


def test_fake_op_shapes():
    """Trace-time shapes, with a ragged length and Dv != Dk."""
    query = torch.empty(2, 4, 100, 64, dtype=torch.float16, device="meta")
    key = torch.empty(2, 4, 100, 64, dtype=torch.float16, device="meta")
    value = torch.empty(2, 4, 100, 128, dtype=torch.float16, device="meta")

    output, state = torch.ops.spyre.linear_attention(query, key, value)

    assert output.shape == (2, 4, 100, 128)
    assert state.shape == (2, 4, 64, 128)
    assert output.dtype == state.dtype == torch.float16


def _meta(*shape, dtype=torch.float16):
    return torch.empty(shape, dtype=dtype, device="meta")


GUARD_CASES = [
    ("rank", r"must be \[B, H, L, D\]"),
    ("empty_sequence", "every dimension must be positive"),
    ("key_shape", "must equal query shape"),
    ("value_shape", "must match query"),
    ("dtype_mismatch", "must share a dtype"),
    ("integer_dtype", "not a floating-point type"),
    ("state_shape", "initial_state shape"),
    ("state_dtype", "initial_state dtype"),
    ("chunk_unaligned", f"must be a positive multiple of {STICK}"),
    ("chunk_zero", f"must be a positive multiple of {STICK}"),
    ("chunk_too_large", "exceeds the maximum"),
    ("nonfinite_scale", "must be finite"),
]


def _guard_inputs(case):
    query = _meta(1, 4, 128, 64)
    key = _meta(1, 4, 128, 64)
    value = _meta(1, 4, 128, 128)
    state, chunk_size, scale = None, 64, None
    if case == "rank":
        query = _meta(4, 128, 64)
    elif case == "empty_sequence":
        query, key, value = _meta(1, 4, 0, 64), _meta(1, 4, 0, 64), _meta(1, 4, 0, 128)
    elif case == "key_shape":
        key = _meta(1, 2, 128, 64)
    elif case == "value_shape":
        value = _meta(1, 4, 64, 128)
    elif case == "dtype_mismatch":
        value = _meta(1, 4, 128, 128, dtype=torch.float32)
    elif case == "integer_dtype":
        query = _meta(1, 4, 128, 64, dtype=torch.int32)
        key = _meta(1, 4, 128, 64, dtype=torch.int32)
        value = _meta(1, 4, 128, 128, dtype=torch.int32)
    elif case == "state_shape":
        state = _meta(1, 4, 64, 64)  # Dv is 128
    elif case == "state_dtype":
        state = _meta(1, 4, 64, 128, dtype=torch.float32)
    elif case == "chunk_unaligned":
        chunk_size = STICK + 1
    elif case == "chunk_zero":
        chunk_size = 0
    elif case == "chunk_too_large":
        chunk_size = MAX_CHUNK_SIZE + STICK
    elif case == "nonfinite_scale":
        scale = float("nan")
    return query, key, value, state, chunk_size, scale


@pytest.mark.parametrize("case,match", GUARD_CASES, ids=[c for c, _ in GUARD_CASES])
def test_unsupported_calls_raise(case, match):
    """Both the op's fake (trace time) and the decomposition refuse the call."""
    query, key, value, state, chunk_size, scale = _guard_inputs(case)

    with pytest.raises(Unsupported, match=match):
        torch.ops.spyre.linear_attention(
            query, key, value, state, chunk_size, True, scale
        )
    with pytest.raises(Unsupported, match=match):
        spyre_linear_attention(query, key, value, state, chunk_size, True, scale)


def test_mismatched_devices_are_refused():
    query = _meta(1, 4, 128, 64)
    key = torch.empty(1, 4, 128, 64, dtype=torch.float16)  # CPU
    value = _meta(1, 4, 128, 64)

    reason = check_linear_attention(query, key, value, None, 64, None)

    assert reason is not None and "one device" in reason


@pytest.mark.parametrize(
    "seqlen,chunk_size,key_dim,value_dim,has_state",
    [
        (1, 64, 64, 64, False),
        (100, 64, 64, 128, True),
        # The chunk loop is a for_each_tile loop, so no number of chunks is refused.
        (64 * 1024, 64, 64, 64, False),
        (MAX_CHUNK_SIZE * 1024, MAX_CHUNK_SIZE, 128, 128, False),
        (256, MAX_CHUNK_SIZE, 64, 64, False),  # largest chunk, mostly padding
    ],
    ids=lambda v: str(v),
)
def test_supported_calls_are_accepted(
    seqlen, chunk_size, key_dim, value_dim, has_state
):
    query = _meta(1, 2, seqlen, key_dim)
    key = _meta(1, 2, seqlen, key_dim)
    value = _meta(1, 2, seqlen, value_dim)
    state = _meta(1, 2, key_dim, value_dim) if has_state else None

    assert check_linear_attention(query, key, value, state, chunk_size, 0.5) is None


# --------------------------------------------------------------------------- #
# Level 3: the op, compiled on Spyre.                                         #
# --------------------------------------------------------------------------- #


def _op(query, key, value, initial_state, chunk_size, is_causal, scale):
    # Return a plain tuple so the CPU and device results are the same type for
    # compare_with_cpu, whatever container the compiled graph hands back.
    output, state = torch.ops.spyre.linear_attention(
        query, key, value, initial_state, chunk_size, is_causal, scale
    )
    return output, state


def _compare(
    query,
    key,
    value,
    *,
    initial_state=None,
    chunk_size=64,
    is_causal=True,
    scale=None,
    **kwargs,
):
    compare_with_cpu(
        _op,
        query,
        key,
        value,
        initial_state,
        chunk_size,
        is_causal,
        scale,
        atol=FP16_ATOL,
        rtol=FP16_RTOL,
        run_eager=False,
        **kwargs,
    )


class TestLinearAttentionOp(unittest.TestCase):
    """The compiled op against the CPU fp32 reference, output and final state."""

    def setUp(self):
        torch._dynamo.reset()

    def test_prefill_causal(self):
        # Four chunks of 64.
        _compare(*_inputs(1, 4, 256))

    def test_prefill_larger_chunk(self):
        _compare(*_inputs(1, 4, 256), chunk_size=128)

    def test_prefill_value_dim_differs_from_key_dim(self):
        # The state is [Dk, Dv]; a transposed state shows up only when they differ.
        _compare(*_inputs(1, 4, 256, key_dim=64, value_dim=128))

    def test_prefill_key_dim_128(self):
        _compare(*_inputs(1, 4, 128, key_dim=128, value_dim=64))

    def test_prefill_ragged_length(self):
        # 100 rows pad to 128; the padded rows are sliced from the output and
        # add nothing to the state.
        _compare(*_inputs(1, 4, 100))

    def test_prefill_batch_and_heads(self):
        _compare(*_inputs(2, 4, 128))

    def test_noncausal(self):
        _compare(*_inputs(1, 4, 256), is_causal=False)

    def test_initial_state(self):
        _compare(*_inputs(1, 4, 128), initial_state=_state(1, 4))

    def test_decode_step(self):
        # One token against a carried state: padded to a whole chunk on device.
        _compare(*_inputs(1, 4, 1), initial_state=_state(1, 4))

    def test_explicit_scale(self):
        _compare(*_inputs(1, 4, 128), scale=0.5)

    def test_split_prefill_threads_state(self):
        """Two calls in one compiled graph, the second seeded by the first's
        final state, reproduce the one-shot reference over the whole sequence."""
        query, key, value = _inputs(1, 4, 256)

        def two_calls(q, k, v):
            first, state = torch.ops.spyre.linear_attention(
                q[:, :, :128], k[:, :, :128], v[:, :, :128], None, 64, True, None
            )
            second, state = torch.ops.spyre.linear_attention(
                q[:, :, 128:], k[:, :, 128:], v[:, :, 128:], state, 64, True, None
            )
            return torch.cat([first, second], dim=2), state

        compare_with_cpu(
            two_calls,
            query,
            key,
            value,
            atol=FP16_ATOL,
            rtol=FP16_RTOL,
            run_eager=False,
            cpu_eager_result=reference_linear_attention(query, key, value),
        )


# --------------------------------------------------------------------------- #
# Level 4: the whole path, compiled on Spyre.                                 #
# --------------------------------------------------------------------------- #


def _to_device(x):
    # Slicing a CPU tensor along L leaves a strided view; hand the device a dense copy.
    return x.contiguous().to(DEVICE)


class TestLinearAttentionCompiledPath(unittest.TestCase):
    """What Level 3 cannot see: the feature map on device, the normalizer, the
    op and the prefill/decode state hand-off in one compiled region, against an
    independent float64 CPU reference. ``dynamic=False`` keeps every call a
    static-shape compile, so the chunk loop has a literal trip count."""

    def setUp(self):
        torch._dynamo.reset()
        # run_and_get_code reads the generated source, which a cache hit skips.
        FxGraphCache.clear()

    def test_feature_map_on_device(self):
        """The reported failure: ``elu`` on device compiles through ``expm1``,
        which Spyre has no kernel for. The replacement must compile and agree with
        ``elu(x) + 1`` for negative, near-zero and positive inputs, including ones
        whose unclamped ``exp`` would overflow fp16."""
        x = _probe_input(FEATURE_MAP_WIDE_PROBES, (64, 64), 31, torch.float16)
        expected = F.elu(x.float()) + 1.0  # fp32 on the host, a different formula

        actual = torch.compile(feature_map, dynamic=False)(_to_device(x)).cpu()

        torch.testing.assert_close(
            actual.float(),
            expected,
            atol=FEATURE_MAP_ATOL,
            rtol=FEATURE_MAP_RTOL,
        )

    def test_prefill_then_decode_matches_cpu_reference(self):
        """Feature map, normalizer and op in one compiled region. A multi-chunk
        ragged prefill, then single-token decode calls seeded by the carried
        state, compared with the quadratic-form reference over the whole sequence."""
        # 200 rows: three full chunks of 64 and a ragged fourth of 8.
        prompt, steps = 200, 3
        q_raw, k_raw, v = _raw_inputs(1, 2, prompt + steps)
        expected = _reference_normalized(q_raw, k_raw, v)
        compiled = torch.compile(_normalized_linear_attention, dynamic=False)

        out, state = compiled(
            _to_device(q_raw[:, :, :prompt]),
            _to_device(k_raw[:, :, :prompt]),
            _to_device(v[:, :, :prompt]),
            None,
        )

        _assert_fp16_close(out, expected[:, :, :prompt], "prefill output")
        _assert_fp16_close(
            state,
            _reference_state(k_raw[:, :, :prompt], v[:, :, :prompt]),
            "state after prefill",
        )
        for t in range(prompt, prompt + steps):
            row, state = compiled(
                _to_device(q_raw[:, :, t : t + 1]),
                _to_device(k_raw[:, :, t : t + 1]),
                _to_device(v[:, :, t : t + 1]),
                state,
            )
            _assert_fp16_close(row, expected[:, :, t : t + 1], f"decode row {t}")
        _assert_fp16_close(state, _reference_state(k_raw, v), "state after decode")

    def _compile_op(self, num_chunks):
        """Compile the bare op over ``num_chunks`` chunks of 64; return the result
        and the generated source."""
        query, key, value = _inputs(1, 1, 64 * num_chunks)
        actual, sources = run_and_get_code(
            torch.compile(_op, dynamic=False),
            _to_device(query),
            _to_device(key),
            _to_device(value),
            None,
            64,
            True,
            None,
        )
        return (query, key, value), actual, "\n".join(sources)

    def _assert_one_counted_loop(self, source, num_chunks):
        # One LoopSpec however many chunks there are, and its trip count is the
        # chunk count. An unrolled body would leave no loop and a copy per chunk.
        self.assertEqual(source.count("LoopSpec("), 1, source)
        self.assertIn(f"count=sympify('{num_chunks}')", source)

    def test_chunk_loop_stays_a_loop_in_the_compiled_representation(self):
        inputs, (output, state), source = self._compile_op(4)

        self._assert_one_counted_loop(source, 4)
        ref_output, ref_state = reference_linear_attention(*inputs)
        _assert_fp16_close(output, ref_output, "output")
        _assert_fp16_close(state, ref_state, "state")

    def test_chunk_loop_beyond_the_former_unroll_limit(self):
        """129 chunks was refused while the loop was unrolled. It compiles, and
        is still one loop. Only the structure and finiteness are asserted: the
        fp16 state accumulated over 8256 rows has no tolerance calibrated on
        hardware yet."""
        num_chunks = FORMER_UNROLL_LIMIT + 1
        _, (output, state), source = self._compile_op(num_chunks)

        self._assert_one_counted_loop(source, num_chunks)
        self.assertEqual(tuple(output.shape), (1, 1, 64 * num_chunks, 64))
        self.assertEqual(tuple(state.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(output.cpu().float()).all())
        self.assertTrue(torch.isfinite(state.cpu().float()).all())


if __name__ == "__main__":
    unittest.main()
