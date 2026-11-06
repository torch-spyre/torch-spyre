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

"""Chunkwise linear attention, expressed with matmuls Spyre already supports.

Softmax attention materializes an ``L x L`` score matrix, so its cost grows
with the square of the sequence length and its decode state (the KV cache)
grows with ``L``. Linear attention drops the softmax. With the feature map
already applied by the caller, causal linear attention is the recurrence::

    S_t = S_{t-1} + k_t^T v_t          # state, [Dk, Dv]
    o_t = scale * q_t S_t              # output row, [Dv]

The state ``S`` has a fixed size, so decode costs ``O(Dk * Dv)`` per token no
matter how long the context is, and prefill costs ``O(L * Dk * Dv)``.

A token-by-token scan is hardware-hostile (one tiny matmul per token). The
chunkwise form processes ``C`` tokens per step and keeps the matmuls big.
For the chunk covering rows ``[s, e)``, with ``S`` the state *before* it::

    inter = Q_c S                          # contribution of everything earlier
    intra = ((Q_c K_c^T) * M) V_c          # inside the chunk, M lower-triangular
    o_c   = scale * (inter + intra)
    S     = S + K_c^T V_c

Everything here is plain ``matmul``, ``transpose().contiguous()``, an
elementwise multiply with a small ``[C, C]`` mask, ``add`` and ``cat``: the same
primitives ``flash_attention.py`` already runs on Spyre. This module imports no
torch_spyre code, so the algorithm can be tested against the reference below on
CPU without a device.

Phase 1 scope (see ``check_linear_attention`` for what is rejected):

* Plain, ungated linear attention. Feature maps (for example ``elu(x) + 1``)
  and any normalization are the caller's job.
* Equal query, key and value head counts (no grouped-query form) and equal
  query, key and value lengths.
* The chunk loop is unrolled into the graph. ``MAX_CHUNKS`` bounds the graph
  size; a loop-based lowering and a fused kernel are the follow-ups.
"""

import math
from typing import Optional

import torch

from .sliding_window_plan import STICK

# The intra-chunk score tile is [chunk_size, chunk_size] per (batch, head), so
# the chunk size bounds the largest temporary this algorithm creates.
MAX_CHUNK_SIZE = 512
DEFAULT_CHUNK_SIZE = STICK
# The chunk loop is unrolled, so the number of chunks is the number of copies of
# the loop body in the traced graph. Refusing past this keeps compile time
# bounded and tells the caller to raise ``chunk_size`` instead.
MAX_CHUNKS = 128


def default_scale(key_dim: int) -> float:
    """Scale applied to the query when the caller passes none: ``Dk ** -0.5``."""
    return key_dim**-0.5


def padded_sequence_length(seqlen: int, chunk_size: int) -> int:
    """Smallest multiple of ``chunk_size`` that holds ``seqlen`` rows."""
    return -(-seqlen // chunk_size) * chunk_size


def check_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    chunk_size: int,
    scale: Optional[float],
) -> Optional[str]:
    """Why this call is outside Phase 1, or ``None`` when it is supported.

    Reads only ``shape``, ``dtype`` and ``device``, so it works on fake and
    meta tensors and can run at trace time.
    """
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.dim() != 4:
            return (
                f"{name} must be [B, H, L, D], got rank {tensor.dim()} "
                f"with shape {tuple(tensor.shape)}"
            )
    batch, heads, seqlen, key_dim = query.shape
    value_dim = value.size(-1)
    if min(batch, heads, seqlen, key_dim, value_dim) <= 0:
        return (
            "every dimension must be positive, got query "
            f"{tuple(query.shape)}, value {tuple(value.shape)}"
        )
    if tuple(key.shape) != tuple(query.shape):
        return (
            f"key shape {tuple(key.shape)} must equal query shape "
            f"{tuple(query.shape)}; grouped-query and cross-length linear "
            "attention are not supported yet"
        )
    if tuple(value.shape[:3]) != tuple(query.shape[:3]):
        return (
            f"value shape {tuple(value.shape)} must match query "
            f"{tuple(query.shape)} in batch, heads and length"
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return (
            f"query, key and value must share a dtype, got {query.dtype}, "
            f"{key.dtype} and {value.dtype}"
        )
    if not query.dtype.is_floating_point:
        return f"dtype {query.dtype} is not a floating-point type"
    if key.device != query.device or value.device != query.device:
        return (
            "query, key and value must be on one device, got "
            f"{query.device}, {key.device} and {value.device}"
        )
    if initial_state is not None:
        expected = (batch, heads, key_dim, value_dim)
        if tuple(initial_state.shape) != expected:
            return (
                f"initial_state shape {tuple(initial_state.shape)} must be "
                f"{expected} = [B, H, Dk, Dv]"
            )
        if initial_state.dtype != query.dtype:
            return (
                f"initial_state dtype {initial_state.dtype} must match query "
                f"dtype {query.dtype}"
            )
        if initial_state.device != query.device:
            return (
                f"initial_state device {initial_state.device} must match "
                f"query device {query.device}"
            )
    if chunk_size < STICK or chunk_size % STICK != 0:
        return f"chunk_size={chunk_size} must be a positive multiple of {STICK}"
    if chunk_size > MAX_CHUNK_SIZE:
        return f"chunk_size={chunk_size} exceeds the maximum of {MAX_CHUNK_SIZE}"
    num_chunks = padded_sequence_length(seqlen, chunk_size) // chunk_size
    if num_chunks > MAX_CHUNKS:
        return (
            f"{num_chunks} chunks of {chunk_size} rows exceed the unrolled-loop "
            f"limit of {MAX_CHUNKS}; raise chunk_size (maximum {MAX_CHUNK_SIZE})"
        )
    if scale is not None and not math.isfinite(scale):
        return f"scale={scale} must be finite"
    return None


def pad_sequence_end(x: torch.Tensor, pad_rows: int) -> torch.Tensor:
    """Append ``pad_rows`` zero rows along the sequence axis of ``[B, H, L, D]``.

    Padding goes at the *end* on purpose. Causality means appended rows cannot
    influence earlier outputs, and a zero key and value add exactly zero to the
    state, so the final state is unchanged and the extra output rows are simply
    sliced off.
    """
    if pad_rows == 0:
        return x
    zeros = torch.zeros(
        (*x.shape[:2], pad_rows, x.shape[-1]), device=x.device, dtype=x.dtype
    )
    return torch.cat([x, zeros], dim=2)


def chunkwise_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    chunk_size: int,
    intra_mask: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear attention over ``[B, H, L, D]`` inputs; returns ``(output, state)``.

    ``L`` must already be a multiple of ``chunk_size`` (see
    ``pad_sequence_end``). ``output`` is ``[B, H, L, Dv]`` and ``state`` is the
    final ``[B, H, Dk, Dv]`` state, which includes ``initial_state``.

    Causal calls need ``intra_mask``: a ``[chunk_size, chunk_size]`` tensor in
    the query dtype holding 1 on and below the diagonal and 0 above it. It is
    an argument so the Spyre path can build it with a device-side custom op
    while CPU tests pass ``torch.tril``. Masking multiplies, so scores must be
    finite.

    Non-causal calls need no loop at all: every row reads the same state, the
    sum of ``K^T V`` over the whole sequence.
    """
    length = query.size(2)
    if length % chunk_size != 0:
        raise ValueError(
            f"sequence length {length} must be a multiple of chunk_size {chunk_size}"
        )

    # Scale Q once, as SDPA does, rather than once per chunk.
    q = query * scale

    if not is_causal:
        keys_t = key.transpose(-1, -2).contiguous()
        update = torch.matmul(keys_t, value)
        state = update if initial_state is None else initial_state + update
        return torch.matmul(q, state), state

    if intra_mask is None:
        raise ValueError("a causal call requires intra_mask")

    state = initial_state
    outputs = []
    for start in range(0, length, chunk_size):
        end = start + chunk_size
        q_c = q[:, :, start:end, :]
        k_c = key[:, :, start:end, :]
        v_c = value[:, :, start:end, :]

        # K^T for this chunk feeds both the intra-chunk scores and the state
        # update, so materialize it once.
        keys_t = k_c.transpose(-1, -2).contiguous()
        scores = torch.matmul(q_c, keys_t) * intra_mask
        out_c = torch.matmul(scores, v_c)
        if state is not None:
            # The first chunk of a call with no initial state has nothing
            # earlier to read, so skip the Q @ S matmul against zeros.
            out_c = out_c + torch.matmul(q_c, state)
        outputs.append(out_c)

        update = torch.matmul(keys_t, v_c)
        state = update if state is None else state + update

    assert state is not None  # length > 0, so the loop ran at least once
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-2)
    return output, state


def reference_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token fp32 recurrence: the definition the chunked form must match.

    Deliberately shares no structure with ``chunkwise_linear_attention``. It
    accumulates in fp32 whatever the input dtype and casts the results back,
    so it is the accuracy yardstick for the half-precision device path. Cost is
    one small step per token, which is fine for tests and useless for serving.
    """
    batch, heads, seqlen, key_dim = query.shape
    value_dim = value.size(-1)
    q_scale = default_scale(key_dim) if scale is None else scale

    q = query.float() * q_scale
    k = key.float()
    v = value.float()
    if initial_state is None:
        state = torch.zeros(
            (batch, heads, key_dim, value_dim), dtype=torch.float32, device=q.device
        )
    else:
        state = initial_state.float()

    if not is_causal:
        state = state + torch.einsum("bhlk,bhlv->bhkv", k, v)
        output = torch.einsum("bhlk,bhkv->bhlv", q, state)
        return output.to(query.dtype), state.to(query.dtype)

    rows = []
    for t in range(seqlen):
        # Include token t before reading: the diagonal is part of the sum.
        state = state + k[:, :, t, :, None] * v[:, :, t, None, :]
        rows.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t, :], state))
    output = torch.stack(rows, dim=2)
    return output.to(query.dtype), state.to(query.dtype)
