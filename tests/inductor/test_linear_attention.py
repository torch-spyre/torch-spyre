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

Three levels, in the order a failure is worth reading:

  1. ALGORITHM -- on CPU, no device. ``spyre_linear_attention`` (the real
     decomposition, which only needs plain torch ops plus the CPU-built mask)
     against a token-by-token fp32 recurrence that shares no structure with it,
     and the recurrence against the O(L^2) masked form so the yardstick itself
     is checked. fp32 on both sides, so tolerances are tight.

  2. CONTRACT -- on CPU/meta, no device. Defaults, the tuple return, and that
     every unsupported call raises ``Unsupported`` from both the op's fake
     (trace time) and the decomposition, with a message naming the problem.

  3. THE OP -- compiled on Spyre against the CPU reference.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_linear_attention.py -v
"""

import unittest

import pytest
import torch
import torch._dynamo

from torch_spyre._inductor.decompositions import spyre_linear_attention
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.linear_attention import (
    MAX_CHUNK_SIZE,
    MAX_CHUNKS,
    check_linear_attention,
    default_scale,
    reference_linear_attention,
)
from torch_spyre._inductor.sliding_window_plan import STICK
from utils_inductor import cached_randn, compare_with_cpu

# fp32 CPU tolerance. Across the shape grid below (L up to 300, Dk/Dv up to 128,
# chunk 64-256) the worst error of the chunked form against the recurrence is
# 4.8e-5 on the output and 6.1e-5 on the state, both about 7e-7 of the largest
# value. 5e-4 is ~10x that, which is accumulation-order slack and nothing more.
FP32_ATOL = 5e-4
FP32_RTOL = 1e-4

# fp16 device tolerance. A CPU half-precision run of this same chunked algorithm
# against the fp32 recurrence (L up to 512, D up to 128, outputs reaching ~12)
# was within 0.0078 on both the output and the state. 0.05 leaves ~6x for the
# device's accumulation order. This is calibrated on that CPU emulation, not on
# hardware; tighten it from device results rather than loosening it.
FP16_ATOL = 0.05
FP16_RTOL = 0.05


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
        (True, 64, 64),  # a single chunk
        (True, 64, 256),  # several chunks
        (True, 128, 256),
        (True, 64, 100),  # ragged: padded to 128 and sliced back
        (True, 256, 300),  # ragged against a large chunk
        (True, 64, 1),  # one token, padded to a whole chunk
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
    ("too_many_chunks", "unrolled-loop limit"),
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
    elif case == "too_many_chunks":
        length = STICK * (MAX_CHUNKS + 1)
        query = _meta(1, 1, length, 64)
        key = _meta(1, 1, length, 64)
        value = _meta(1, 1, length, 128)
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
        (64 * MAX_CHUNKS, 64, 64, 64, False),  # exactly the unrolled limit
        (MAX_CHUNK_SIZE * MAX_CHUNKS, MAX_CHUNK_SIZE, 128, 128, False),
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


if __name__ == "__main__":
    unittest.main()
