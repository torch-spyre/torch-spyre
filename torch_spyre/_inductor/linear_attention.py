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

Everything inside a chunk is plain ``matmul``, ``transpose().contiguous()``, an
elementwise multiply with a small ``[C, C]`` mask and ``add``: the same
primitives ``flash_attention.py`` already runs on Spyre. The loop over chunks
is a ``for_each_tile`` loop. ``S`` is its tensor carry, Q, K and V are sliced
along the sequence axis, the mask is an invariant operand, and each step's
output chunk is laid back along the same axis. The traced graph therefore holds
one copy of the chunk body however long the sequence is.

The algorithm needs no Spyre device, so it can be tested against the reference
below on CPU. It does import ``for_each_tile`` from the Spyre frontend, so the
``torch_spyre`` package must be importable.

Phase 1 scope (see ``check_linear_attention`` for what is rejected):

* Plain, ungated linear attention. Feature maps (for example ``elu(x) + 1``)
  and any normalization are the caller's job.
* Equal query, key and value head counts (no grouped-query form) and equal
  query, key and value lengths.
* There is no limit on the number of chunks here. The backend compiler unrolls
  counted loops by default (``SPYRE_BACKEND_LOOP_UNROLL``), so its build time
  still grows with the chunk count even though the traced graph does not.
"""

import math
from typing import Optional

import torch

from .sliding_window_plan import STICK
from .wsr.for_each_tile import for_each_tile

# The intra-chunk score tile is [chunk_size, chunk_size] per (batch, head), so
# the chunk size bounds the largest temporary this algorithm creates.
MAX_CHUNK_SIZE = 512
DEFAULT_CHUNK_SIZE = STICK


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
    # This loop runs once per argument at trace time and emits no tensor ops, so
    # it is validation, not part of the compiled graph.
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
    finite. The loop body receives it as an invariant operand: a
    ``for_each_tile`` body may not close over a tensor.

    Non-causal calls need no loop at all: every row reads the same state, the
    sum of ``K^T V`` over the whole sequence.
    """
    length = query.size(2)
    if length % chunk_size != 0:
        raise ValueError(
            f"sequence length {length} must be a multiple of chunk_size {chunk_size}"
        )

    # Scale Q once, as SDPA does, rather than once per chunk. It is also what
    # lets the loop body take Q as an operand instead of closing over a scale.
    q = query * scale

    if not is_causal:
        keys_t = key.transpose(-1, -2).contiguous()
        update = torch.matmul(keys_t, value)
        state = update if initial_state is None else initial_state + update
        return torch.matmul(q, state), state

    if intra_mask is None:
        raise ValueError("a causal call requires intra_mask")

    batch, heads, _, key_dim = query.shape
    value_dim = value.size(-1)
    if initial_state is None:
        initial = torch.zeros(
            (batch, heads, key_dim, value_dim), dtype=query.dtype, device=query.device
        )
    else:
        # The carry must keep its metadata (shape, dtype and strides) from one
        # step to the next, and ``state + update`` returns a dense tensor.
        initial = initial_state.contiguous()

    def chunk_step(state, tiles):
        q_c, k_c, v_c, mask = tiles

        # K^T for this chunk feeds both the intra-chunk scores and the state
        # update, so materialize it once.
        keys_t = k_c.transpose(-1, -2).contiguous()
        scores = torch.matmul(q_c, keys_t) * mask
        # Read ``state`` for the output before producing ``next_state``: the
        # loop lowering can then update the carry in place without keeping a
        # snapshot of the old value for a later reader.
        out_c = torch.matmul(scores, v_c) + torch.matmul(q_c, state)
        next_state = state + torch.matmul(keys_t, v_c)
        return next_state, out_c

    operands = (q, key, value, intra_mask)
    num_chunks = length // chunk_size
    if num_chunks == 1:
        # One chunk is not worth a loop (this is every single-token decode),
        # and the SDPA decomposition does the same for a one-tile map.
        state, output = chunk_step(initial, operands)
        return output, state

    state, output = for_each_tile(
        chunk_step,
        operands,
        dims=(2, 2, 2, None),  # Q, K, V along L; the mask is read whole.
        tile_size=chunk_size,
        init=initial,
        out_dim=2,
    )
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

    The ``for t in range(seqlen)`` loop below is intentionally a Python loop. This
    function only ever runs eagerly on CPU, as the registered CPU kernel of
    ``spyre::linear_attention`` and as the test oracle, and is never traced
    into a Spyre graph. Expressing it with ``for_each_tile`` would make the
    yardstick share the machinery it is there to check.
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
