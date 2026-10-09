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


"""Workloads the frontend timing sweep compiles.

Each builder returns a callable plus its arguments, already on the Spyre device. Each
body is copied from a test that compiles today, named in its docstring; when that test
changes, update the copy.

Model-shaped families answer "how long does a real shape take". Mechanism probes
(``elementwise_chain``, ``fanout``, ``dup_constants``) move one axis a specific pass
scales on, so a superlinear pass can be attributed.
"""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch
import torch.nn.functional as F

from torch_spyre._inductor.propagate_hints import spyre_hint
from torch_spyre.constants import DEVICE_NAME


@dataclass
class Workload:
    """A compilable callable plus the arguments to compile it against."""

    fn: Callable[..., Any]
    args: tuple[Any, ...]
    #: Builder parameters, defaults included. ``build`` fills these and ``name`` in, so
    #: a builder sets ``params`` only for a value it resolves itself.
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""


def _randn(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    # Built on CPU then moved: device init is not what the sweep measures, and a seeded
    # CPU tensor keeps a point's inputs identical across samples.
    return torch.randn(*shape, dtype=dtype).to(DEVICE_NAME)


def build_flash(
    *,
    B: int = 1,
    H: int = 8,
    Lq: int = 256,
    Lk: int = 256,
    D: int = 64,
    block_size: int = 128,
) -> Workload:
    """Block-tiled flash attention with an online softmax.

    From ``test_building_blocks.py::test_flash_attention``, with separate query and key
    lengths. The loop unrolls at trace time, so the graph carries ``Lk / block_size``
    copies of its body.
    """
    torch.manual_seed(0)
    Q = _randn(B, H, Lq, D)
    K = _randn(B, H, Lk, D)
    V = _randn(B, H, Lk, D)

    def flash(Q, K, V):
        output = torch.zeros_like(Q)
        M = torch.full((B, H, Lq), float("-inf"), device=Q.device, dtype=torch.float16)
        denominator = torch.zeros((B, H, Lq), device=Q.device, dtype=torch.float16)
        scale = 1.0 / math.sqrt(D)

        for start in range(0, Lk, block_size):
            end = start + block_size
            K_block = K[:, :, start:end, :]
            V_block = V[:, :, start:end, :]
            K_block_T = K_block.transpose(-1, -2).contiguous()

            scores = torch.matmul(Q, K_block_T) * scale
            # Transposed to keep the reduction off the stick dimension.
            scores = scores.transpose(-1, -2).contiguous()
            block_max = torch.amax(scores, dim=-2)
            max_running = torch.maximum(M, block_max)

            exp_scores = torch.exp(scores - max_running.unsqueeze(-2))
            correction = torch.exp(M - max_running)

            denominator = denominator * correction + exp_scores.sum(dim=-2)
            output = output * correction.unsqueeze(-1) + torch.bmm(
                exp_scores.transpose(-1, -2).flatten(0, 1), V_block.flatten(0, 1)
            ).unflatten(0, (B, H))

            M = max_running

        return output / denominator.unsqueeze(-1)

    return Workload(fn=flash, args=(Q, K, V))


def build_mlp(
    *,
    seq_len: int = 256,
    emb_dim: int = 1024,
    layers: int = 1,
    intermediate: int = 0,
) -> Workload:
    """Stacked SwiGLU MLP, from ``test_building_blocks.py::test_mlp``.

    ``intermediate`` defaults to ``4 * emb_dim``, a ratio real models do not use, so
    realistic points set it.
    """
    inter = intermediate or 4 * emb_dim
    torch.manual_seed(0)
    x = _randn(seq_len, emb_dim)
    weights = [
        (_randn(emb_dim, inter), _randn(emb_dim, inter), _randn(inter, emb_dim))
        for _ in range(layers)
    ]

    def mlp(x, weights):
        for gate, up, down in weights:
            gate_out = x @ gate
            up_out = x @ up
            x = (up_out * F.silu(gate_out)) @ down
        return x

    return Workload(fn=mlp, args=(x, weights), params={"intermediate": inter})


def _head_dim(E: int, heads: int) -> int:
    if E % heads:
        raise ValueError(f"E {E} not divisible by heads {heads}")
    head_dim = E // heads
    if head_dim % 64:
        # 64 fp16 elements is one stick; a fractional head lands as an
        # "Unsupported coordinate expression 5*c0/2" assertion deep in lowering.
        raise ValueError(
            f"head_dim {head_dim} (E {E} / heads {heads}) is not a multiple of 64"
        )
    return head_dim


def _rms_norm(t: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return t * torch.rsqrt((t * t).mean(-1, keepdim=True) + eps) * weight


def _split_heads(
    t: torch.Tensor, B: int, S: int, heads: int, head_dim: int, name: str = "H"
) -> torch.Tensor:
    # Above S=512 (_SDPA_MAX_SEQUENCE_TILE_SIZE) the SDPA decomposition tiles S, which
    # fails with "reshape split a named dim, re-annotate" unless this split is hinted.
    # The hint goes here, not on the inputs, because q, k and v are computed in-graph.
    with spyre_hint(named_dims=["B", "S", name, "D"]):
        split = t.reshape(B, S, heads, head_dim)
    with spyre_hint(named_dims=["B", name, "S", "D"]):
        return split.transpose(1, 2)


def _attention(q, k, v, B: int, S: int, E: int, **sdpa_kwargs: Any) -> torch.Tensor:
    with spyre_hint(named_dims=["B", "H", "S", "D"]):
        attn = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    with spyre_hint(named_dims=["B", "S", "H", "D"]):
        back = attn.transpose(1, 2)
    with spyre_hint(named_dims=["B", "S", "E"]):
        merged = back.reshape(B, S, E)
    return merged


def _swiglu(h: torch.Tensor, gate, up, down) -> torch.Tensor:
    return (h @ up * F.silu(h @ gate)) @ down


def build_transformer_block(
    *,
    B: int = 1,
    S: int = 512,
    E: int = 4096,
    heads: int = 32,
    intermediate: int = 14336,
) -> Workload:
    """Pre-norm decoder block at Llama-3.1-8B's dimensions.

    Hidden 4096, 32 heads of 128 and intermediate 14336, as captured in
    ``tests/resource/models/Meta-Llama-3.1-8B-Instruct.yaml``. Shapes stay 4-D through
    attention: a 3-D query makes the SDPA decomposition index a missing dimension, and
    projecting a 2-D activation hits #3193.
    """
    head_dim = _head_dim(E, heads)
    torch.manual_seed(0)
    x = _randn(B, S, E)
    norm1, norm2 = _randn(E), _randn(E)
    wq, wk, wv, wo = (_randn(E, E) for _ in range(4))
    gate, up = _randn(E, intermediate), _randn(E, intermediate)
    down = _randn(intermediate, E)

    def block(x, norm1, norm2, wq, wk, wv, wo, gate, up, down):
        h = _rms_norm(x, norm1, 1e-6)
        q = _split_heads(h @ wq, B, S, heads, head_dim)
        k = _split_heads(h @ wk, B, S, heads, head_dim)
        v = _split_heads(h @ wv, B, S, heads, head_dim)
        x = x + _attention(q, k, v, B, S, E) @ wo
        h = _rms_norm(x, norm2, 1e-6)
        return x + _swiglu(h, gate, up, down)

    return Workload(fn=block, args=(x, norm1, norm2, wq, wk, wv, wo, gate, up, down))


def build_control_flow(
    *, M: int = 8, K: int = 12, N: int = 6, tile: int = 2
) -> Workload:
    """A ``for_each_tile`` map over M, lowered through the scan HOP.

    From ``for_each_tile_fixtures.py::split_m_fn``. The loop body is the graph, so no
    parameter grows it: this family is for coverage. It is the only control flow the
    backend handles; ``torch.cond`` has no lowering and carry-mode split-K is an
    expected failure (#4460).
    """
    from torch_spyre._inductor.wsr import for_each_tile

    # fp16, cast before the transfer: the backend has no fp32 batchmatmul, and
    # test_for_each_tile_e2e.py notes that casting after the transfer produces garbage.
    torch.manual_seed(0)
    X = _randn(M, K)
    Y = _randn(K, N)

    def split_m(X, Y):
        def body(_, ops):
            x_tile, y_whole = ops
            return None, x_tile @ y_whole

        _, out = for_each_tile(body, (X, Y), dims=(0, None), tile_size=tile, out_dim=0)
        return out

    return Workload(fn=split_m, args=(X, Y))


# ---------------------------------------------------------------------------
# Granite 3.3 8B, from its published config and the shapes captured in
# ``tests/resource/models/granite-3.3-8b-instruct.yaml``. Unlike Llama-3.1-8B, its
# intermediate is 12800 and its attention is grouped-query.
GRANITE_E = 4096
GRANITE_HEADS = 32
GRANITE_KV_HEADS = 8
GRANITE_INTERMEDIATE = 12800
GRANITE_VOCAB = 49159
# Scalar multipliers Granite applies; each one is an operation in the graph.
GRANITE_ATTENTION_MULTIPLIER = 0.0078125
GRANITE_EMBEDDING_MULTIPLIER = 12.0
GRANITE_LOGITS_SCALING = 16.0
GRANITE_RESIDUAL_MULTIPLIER = 0.22
GRANITE_RMS_EPS = 1e-5


def build_granite_layer(
    *,
    B: int = 1,
    S: int = 512,
    layers: int = 1,
    E: int = GRANITE_E,
    heads: int = GRANITE_HEADS,
    kv_heads: int = GRANITE_KV_HEADS,
    intermediate: int = GRANITE_INTERMEDIATE,
) -> Workload:
    """Granite 3.3 8B decoder layers, stacked ``layers`` deep.

    32 query heads share 8 key/value heads, so SDPA runs with ``enable_gqa=True``, as in
    ``test_building_blocks.py::_run_granite_gqa_with_finite_broadcast_mask``. Query and
    key/value heads get different dim names because they differ in size.
    """
    head_dim = _head_dim(E, heads)
    if heads % kv_heads:
        raise ValueError(f"heads {heads} not divisible by kv_heads {kv_heads}")

    torch.manual_seed(0)
    x = _randn(B, S, E)
    kv_width = kv_heads * head_dim
    # norm1, norm2, wq, wk, wv, wo, gate, up, down
    shapes = [(E,), (E,), (E, E), (E, kv_width), (E, kv_width), (E, E)]
    shapes += [(E, intermediate), (E, intermediate), (intermediate, E)]
    weights = [tuple(_randn(*shape) for shape in shapes) for _ in range(layers)]

    def stack(x, weights):
        for norm1, norm2, wq, wk, wv, wo, gate, up, down in weights:
            h = _rms_norm(x, norm1, GRANITE_RMS_EPS)
            q = _split_heads(h @ wq, B, S, heads, head_dim, "H")
            k = _split_heads(h @ wk, B, S, kv_heads, head_dim, "Hkv")
            v = _split_heads(h @ wv, B, S, kv_heads, head_dim, "Hkv")
            merged = _attention(
                q, k, v, B, S, E, scale=GRANITE_ATTENTION_MULTIPLIER, enable_gqa=True
            )
            x = x + GRANITE_RESIDUAL_MULTIPLIER * (merged @ wo)
            h = _rms_norm(x, norm2, GRANITE_RMS_EPS)
            mlp_out = _swiglu(h, gate, up, down)
            x = x + GRANITE_RESIDUAL_MULTIPLIER * mlp_out
        return x

    return Workload(fn=stack, args=(x, weights))


def build_granite_lm_head(
    *,
    B: int = 1,
    S: int = 512,
    E: int = GRANITE_E,
    vocab: int = GRANITE_VOCAB,
    chunks: int = 4,
) -> Workload:
    """Granite's final norm and language-model head, split over the vocabulary.

    Unsplit, work division rejects the [4096, 49159] weight: its per-core span is
    384.5 MB against a 256 MB limit, and fewer cores only raises it. The chunks are
    returned rather than concatenated, so cat's lowering is not part of the cost.
    """
    torch.manual_seed(0)
    if chunks < 1:
        raise ValueError(f"chunks {chunks} must be at least 1")
    x = _randn(B, S, E)
    norm = _randn(E)
    # The remainder rides on the last chunk, so the widths still sum to vocab.
    width = vocab // chunks
    widths = [width] * chunks
    widths[-1] += vocab - width * chunks
    heads = [_randn(E, w) for w in widths]

    def lm_head(x, norm, heads):
        h = _rms_norm(x, norm, GRANITE_RMS_EPS)
        return tuple((h @ head) / GRANITE_LOGITS_SCALING for head in heads)

    return Workload(fn=lm_head, args=(x, norm, heads))


def build_granite_embedding(
    *, S: int = 512, E: int = GRANITE_E, vocab: int = GRANITE_VOCAB
) -> Workload:
    """Granite's token embedding, as ``index_select``: ``embedding`` has no lowering.

    From ``test_indirect_access_gather.py::test_index_select``, with its int32 index.
    The only indirect access in the sweep. Its numerics are an expected failure, which
    a frontend-only compile never reaches.
    """
    torch.manual_seed(0)
    table = _randn(vocab, E)
    ids = torch.randint(0, vocab, (S,), dtype=torch.int32).to(DEVICE_NAME)

    def embed(table, ids):
        return torch.index_select(table, 0, ids) * GRANITE_EMBEDDING_MULTIPLIER

    return Workload(fn=embed, args=(table, ids))


# ---------------------------------------------------------------------------
# Mechanism probes: small graphs whose only purpose is to move one axis.


def build_elementwise_chain(
    *, ops: int = 64, rows: int = 256, cols: int = 1024
) -> Workload:
    """``ops`` pointwise operations and no matmul, for per-operation pass cost.

    Each operation stays its own buffer because ``enable_spyre_context`` forces
    ``Loops.has_large_inner_fn`` (``patches.py``); without that the chain would fuse
    into one operation.
    """
    torch.manual_seed(0)
    x = _randn(rows, cols)

    def chain(x):
        for i in range(ops):
            # Cycled so the graph is a mix of unary and scalar-binary operations rather
            # than the same node repeated, which planning could treat as one shape.
            step = i % 4
            if step == 0:
                x = torch.relu(x)
            elif step == 1:
                x = x * 1.0009765625
            elif step == 2:
                x = x + 0.5
            else:
                x = F.silu(x)
        return x

    return Workload(fn=chain, args=(x,))


def build_fanout(*, consumers: int = 8, rows: int = 256, cols: int = 1024) -> Workload:
    """One buffer read by ``consumers`` operations, the lookup axis behind #4113."""
    torch.manual_seed(0)
    x = _randn(rows, cols)

    def fanout(x):
        producer = torch.relu(x)
        total = producer * 1.0
        for i in range(2, consumers + 1):
            total = total + producer * float(i)
        return total

    return Workload(fn=fanout, args=(x,))


def build_dup_constants(
    *, dups: int = 4, B: int = 2, M: int = 8, N: int = 32
) -> Workload:
    """``dups`` unaligned bmms over one activation: one dedup group of size ``dups``.

    Each unaligned K emits an identical padding constant. Fixture from
    ``test_dedup_constants.py``: fp16, K one element past a stick boundary.
    """
    from torch_spyre._C import get_elem_in_stick

    torch.manual_seed(0)
    K = get_elem_in_stick(torch.float16) + 1
    x = _randn(B, M, K)
    weights = [_randn(B, K, N) for _ in range(dups)]

    def dup(x, weights):
        out = torch.bmm(x, weights[0])
        for w in weights[1:]:
            out = out + torch.bmm(x, w)
        return out

    return Workload(fn=dup, args=(x, weights))


#: Workload name -> builder. Add an entry here and a point in sweep_plan.json.
BUILDERS: dict[str, Callable[..., Workload]] = {
    "control_flow": build_control_flow,
    "dup_constants": build_dup_constants,
    "elementwise_chain": build_elementwise_chain,
    "fanout": build_fanout,
    "flash": build_flash,
    "granite_embedding": build_granite_embedding,
    "granite_layer": build_granite_layer,
    "granite_lm_head": build_granite_lm_head,
    "mlp": build_mlp,
    "transformer_block": build_transformer_block,
}


def build(name: str, **params: Any) -> Workload:
    """Build workload ``name`` with ``params``, rejecting unknown names loudly."""
    try:
        builder = BUILDERS[name]
    except KeyError:
        known = ", ".join(sorted(BUILDERS))
        raise SystemExit(
            f"unknown workload {name!r}; known workloads: {known}"
        ) from None
    workload = builder(**params)
    bound = inspect.signature(builder).bind(**params)
    bound.apply_defaults()
    resolved = {**bound.arguments, **workload.params}
    return replace(workload, name=name, params=resolved)
