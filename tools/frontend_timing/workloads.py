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

Each builder returns a callable plus its arguments, already on the Spyre device. The
sweep compiles them and measures the compile, so nothing here is about numerics -- but
a workload that does not compile measures nothing, so each is derived from a test that
passes today, named in its docstring.

These bodies are COPIES rather than imports from ``tests/``. A baseline is only
comparable against a later one if the workload did not move in between, and test
helpers move for test reasons. When a source test changes, reconcile it by hand.

Sizes are parameters rather than constants because #4117 is about how compile time
scales with graph size, and graph size is what these parameters drive: flash unrolls
its block loop at trace time, so ``Lk / block_size`` inner bodies reach the compiler,
and the MLP's ``layers`` multiplies its body directly.

Two kinds of family live here. The model-shaped ones -- ``granite_layer``,
``granite_lm_head``, ``granite_embedding``, ``transformer_block``, ``mlp``, ``flash`` --
answer "how long does a real shape take". The mechanism probes --
``elementwise_chain``, ``fanout``, ``dup_constants`` -- move a single axis a specific
pass scales on, so a superlinear pass can be attributed rather than merely observed.
A probe is not a workload anyone runs; it is an instrument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from torch_spyre._inductor.propagate_hints import spyre_hint
from torch_spyre.constants import DEVICE_NAME


@dataclass
class Workload:
    """A compilable callable plus the arguments to compile it against."""

    name: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    params: dict[str, Any] = field(default_factory=dict)


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

    From ``tests/inductor/test_building_blocks.py::test_flash_attention``, generalized
    to separate query and key lengths. The ``for`` loop is unrolled during tracing, so
    the graph carries ``Lk / block_size`` copies of the inner body -- this is the knob
    that grows the graph without changing its shape.
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

    return Workload(
        name="flash",
        fn=flash,
        args=(Q, K, V),
        params={"B": B, "H": H, "Lq": Lq, "Lk": Lk, "D": D, "block_size": block_size},
    )


def build_mlp(
    *,
    seq_len: int = 256,
    emb_dim: int = 1024,
    layers: int = 1,
    intermediate: int = 0,
) -> Workload:
    """Stacked SwiGLU MLP.

    From ``tests/inductor/test_building_blocks.py::test_mlp``, with a layer loop so the
    graph grows linearly in ``layers`` while every other dimension holds still.
    ``intermediate`` defaults to ``4 * emb_dim``; real models do not use that ratio
    (Llama-3.1-8B is 4096 -> 14336), so a realistic point sets it.
    """
    inter = intermediate or 4 * emb_dim
    torch.manual_seed(0)
    x = _randn(seq_len, emb_dim)
    weights = []
    for _ in range(layers):
        weights.append(
            (_randn(emb_dim, inter), _randn(emb_dim, inter), _randn(inter, emb_dim))
        )

    def mlp(x, weights):
        for gate, up, down in weights:
            gate_out = x @ gate
            up_out = x @ up
            x = (up_out * F.silu(gate_out)) @ down
        return x

    return Workload(
        name="mlp",
        fn=mlp,
        args=(x, weights),
        params={
            "seq_len": seq_len,
            "emb_dim": emb_dim,
            "layers": layers,
            "intermediate": inter,
        },
    )


def build_transformer_block(
    *,
    B: int = 1,
    S: int = 512,
    E: int = 4096,
    heads: int = 32,
    intermediate: int = 14336,
) -> Workload:
    """Pre-norm decoder block: RMS norm, self-attention, RMS norm, SwiGLU MLP.

    Defaults are Llama-3.1-8B's own dimensions, as captured in
    ``tests/resource/models/Meta-Llama-3.1-8B-Instruct.yaml``: hidden 4096, 32 heads of
    128, intermediate 14336. ``S`` is the axis worth moving -- 1 is decode, hundreds to
    thousands is prefill.

    Two things are load-bearing in how this is written. Shapes stay 4-D through
    attention: a 3-D ``(heads, seq, head_dim)`` query makes Spyre's SDPA decomposition
    index a dimension that is not there, and projecting from a 2-D activation with no
    batch dim leaves the restickify pass unable to reconcile a per-head layout -- the
    shape of #3193. And the per-head reshape carries ``spyre_hint`` named dims: without
    them the compile fails above ``S=512`` with "layout dim 2 has 2 loop vars but only 1
    name(s) ['max_seqlen_q'] -- reshape split a named dim, re-annotate", because the
    decomposition tiles the sequence dimension at that size. Annotated, S=1 through 2048
    all compile.
    """
    if E % heads:
        raise ValueError(f"E {E} not divisible by heads {heads}")
    head_dim = E // heads
    if head_dim % 64:
        # 64 fp16 elements is one stick; a fractional head lands as an
        # "Unsupported coordinate expression 5*c0/2" assertion deep in lowering.
        raise ValueError(
            f"head_dim {head_dim} (E {E} / heads {heads}) is not a multiple of 64"
        )

    torch.manual_seed(0)
    x = _randn(B, S, E)
    norm1, norm2 = _randn(E), _randn(E)
    wq, wk, wv, wo = (_randn(E, E) for _ in range(4))
    gate, up = _randn(E, intermediate), _randn(E, intermediate)
    down = _randn(intermediate, E)

    def rms_norm(t, weight):
        return t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-6) * weight

    def per_head(t):
        with spyre_hint(named_dims=["B", "S", "H", "D"]):
            split = t.reshape(B, S, heads, head_dim)
        with spyre_hint(named_dims=["B", "H", "S", "D"]):
            return split.transpose(1, 2)

    def block(x, norm1, norm2, wq, wk, wv, wo, gate, up, down):
        h = rms_norm(x, norm1)
        q, k, v = per_head(h @ wq), per_head(h @ wk), per_head(h @ wv)
        with spyre_hint(named_dims=["B", "H", "S", "D"]):
            attn = F.scaled_dot_product_attention(q, k, v)
        with spyre_hint(named_dims=["B", "S", "H", "D"]):
            back = attn.transpose(1, 2)
        with spyre_hint(named_dims=["B", "S", "E"]):
            merged = back.reshape(B, S, E)
        x = x + merged @ wo
        h = rms_norm(x, norm2)
        return x + (h @ up * F.silu(h @ gate)) @ down

    return Workload(
        name="transformer_block",
        fn=block,
        args=(x, norm1, norm2, wq, wk, wv, wo, gate, up, down),
        params={"B": B, "S": S, "E": E, "heads": heads, "intermediate": intermediate},
    )


def build_control_flow(
    *, M: int = 8, K: int = 12, N: int = 6, tile: int = 2
) -> Workload:
    """A ``for_each_tile`` map over M, which lowers through the scan HOP.

    From ``tests/inductor/for_each_tile_fixtures.py::split_m_fn``, the map-mode case
    that ``test_for_each_tile_e2e.py`` runs end to end. It is the only control flow the
    backend handles today: ``torch.cond`` has no lowering, and the carry-mode split-K
    case is an expected failure (#4460).
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

    return Workload(
        name="control_flow",
        fn=split_m,
        args=(X, Y),
        params={"M": M, "K": K, "N": N, "tile": tile},
    )


# ---------------------------------------------------------------------------
# Granite 3.3 8B.

# From Granite 3.3 8B's published config, cross-checked against the shapes captured in
# ``tests/resource/models/granite-3.3-8b-instruct.yaml``. Kept together because a point
# that mixes these with Llama's dimensions measures neither model: Granite's
# intermediate is 12800 where Llama-3.1-8B's is 14336, and Granite is grouped-query
# (32 query heads over 8 key/value heads) where Llama-3.1-8B here is not.
GRANITE_E = 4096
GRANITE_HEADS = 32
GRANITE_KV_HEADS = 8
GRANITE_INTERMEDIATE = 12800
GRANITE_VOCAB = 49159
#: Granite's full depth. Nothing here compiles 40 layers in one graph -- see the README
#: on extrapolation -- but the depth axis is fitted towards this number.
GRANITE_LAYERS = 40
# Granite scales four things a plain Llama-shaped block does not. They are only scalar
# multiplies, but they are operations, so a faithful graph carries them.
GRANITE_ATTENTION_MULTIPLIER = 0.0078125
GRANITE_EMBEDDING_MULTIPLIER = 12.0
GRANITE_LOGITS_SCALING = 16.0
GRANITE_RESIDUAL_MULTIPLIER = 0.22
GRANITE_RMS_EPS = 1e-5


def _rms_norm(t: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return t * torch.rsqrt((t * t).mean(-1, keepdim=True) + GRANITE_RMS_EPS) * weight


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

    Grouped-query attention is what separates this from ``transformer_block``: 32 query
    heads share 8 key/value heads, so the projections are asymmetric and SDPA runs with
    ``enable_gqa=True``. That path is exercised by
    ``test_building_blocks.py::_run_granite_gqa_with_finite_broadcast_mask``
    (``B, H, N_KV, D = 1, 32, 8, 128``) and was only fixed in #4793, so it is thinly
    covered and worth sweeping.

    Named dims come from ``spyre_hint`` on the reshapes rather than from the eager
    ``name_tensor_dims`` API, even though the >512 GQA test uses the latter: here q, k
    and v are *computed* from projections inside the graph, so naming the input tensors
    would never reach the reshape that splits the head dimension. Above
    ``S=512`` -- ``_SDPA_MAX_SEQUENCE_TILE_SIZE`` in ``decompositions.py`` -- the
    decomposition must tile the query dimension, which needs those names to exist.

    The query and key/value head dimensions carry *different* names because they are
    different sizes; one name at two sizes is a conflict, not a shortcut.
    """
    if E % heads:
        raise ValueError(f"E {E} not divisible by heads {heads}")
    head_dim = E // heads
    if head_dim % 64:
        # 64 fp16 elements is one stick; a fractional head lands as an
        # "Unsupported coordinate expression 5*c0/2" assertion deep in lowering.
        raise ValueError(
            f"head_dim {head_dim} (E {E} / heads {heads}) is not a multiple of 64"
        )
    if heads % kv_heads:
        raise ValueError(f"heads {heads} not divisible by kv_heads {kv_heads}")

    torch.manual_seed(0)
    x = _randn(B, S, E)
    kv_width = kv_heads * head_dim
    weights = []
    for _ in range(layers):
        weights.append(
            (
                _randn(E),
                _randn(E),
                _randn(E, E),
                _randn(E, kv_width),
                _randn(E, kv_width),
                _randn(E, E),
                _randn(E, intermediate),
                _randn(E, intermediate),
                _randn(intermediate, E),
            )
        )

    def per_head(t, n_heads, head_name):
        with spyre_hint(named_dims=["B", "S", head_name, "D"]):
            split = t.reshape(B, S, n_heads, head_dim)
        with spyre_hint(named_dims=["B", head_name, "S", "D"]):
            return split.transpose(1, 2)

    def stack(x, weights):
        for norm1, norm2, wq, wk, wv, wo, gate, up, down in weights:
            h = _rms_norm(x, norm1)
            q = per_head(h @ wq, heads, "H")
            k = per_head(h @ wk, kv_heads, "Hkv")
            v = per_head(h @ wv, kv_heads, "Hkv")
            with spyre_hint(named_dims=["B", "H", "S", "D"]):
                attn = F.scaled_dot_product_attention(
                    q, k, v, scale=GRANITE_ATTENTION_MULTIPLIER, enable_gqa=True
                )
            with spyre_hint(named_dims=["B", "S", "H", "D"]):
                back = attn.transpose(1, 2)
            with spyre_hint(named_dims=["B", "S", "E"]):
                merged = back.reshape(B, S, E)
            x = x + GRANITE_RESIDUAL_MULTIPLIER * (merged @ wo)
            h = _rms_norm(x, norm2)
            mlp_out = (h @ up * F.silu(h @ gate)) @ down
            x = x + GRANITE_RESIDUAL_MULTIPLIER * mlp_out
        return x

    return Workload(
        name="granite_layer",
        fn=stack,
        args=(x, weights),
        params={
            "B": B,
            "S": S,
            "layers": layers,
            "E": E,
            "heads": heads,
            "kv_heads": kv_heads,
            "intermediate": intermediate,
        },
    )


def build_granite_lm_head(
    *,
    B: int = 1,
    S: int = 512,
    E: int = GRANITE_E,
    vocab: int = GRANITE_VOCAB,
    chunks: int = 4,
) -> Workload:
    """Granite's final norm and language-model head, split over the vocabulary.

    A 4096 -> 49159 projection is 201M parameters in one matmul, and no amount of
    decoder-layer sweeping ever reaches it.

    It does not fit unsplit. Measured: work division rejects the whole weight with
    "per-core tensor span 384.500 MB (shape=[4096, 49159]) exceeds hardware limit of
    256.00 MB". ``SENCORES`` cannot rescue that -- 32 cores is already the maximum and
    fewer cores means more per core -- so the projection is split into ``chunks``
    matmuls, which is what a real implementation does for a vocabulary this size. That
    makes ``chunks`` a working-set axis worth sweeping in its own right.

    The chunks are returned rather than concatenated: the concatenation is not the work
    being measured, and keeping it out avoids making this family's cost depend on
    whether cat lowers well.
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
        h = _rms_norm(x, norm)
        return tuple((h @ head) / GRANITE_LOGITS_SCALING for head in heads)

    return Workload(
        name="granite_lm_head",
        fn=lm_head,
        args=(x, norm, heads),
        params={"B": B, "S": S, "E": E, "vocab": vocab, "chunks": chunks},
    )


def build_granite_embedding(
    *, S: int = 512, E: int = GRANITE_E, vocab: int = GRANITE_VOCAB
) -> Workload:
    """Granite's token embedding, as the gather it lowers to.

    There is no ``embedding`` lowering in the backend, so this is written as
    ``index_select`` over the table -- which is the same memory access and is a
    supported frontend path, from
    ``tests/inductor/test_indirect_access_gather.py::test_index_select``. The index is
    int32, as every scenario in that file uses.

    Worth its own family because it is the only indirect access in the sweep: it is the
    one point that reaches ``enforce_indirect_access_layout`` at all. The backend's
    indirect-access numerics are an expected failure today, which does not matter
    here -- a frontend-only compile never runs the kernel.
    """
    torch.manual_seed(0)
    table = _randn(vocab, E)
    ids = torch.randint(0, vocab, (S,), dtype=torch.int32).to(DEVICE_NAME)

    def embed(table, ids):
        return torch.index_select(table, 0, ids) * GRANITE_EMBEDDING_MULTIPLIER

    return Workload(
        name="granite_embedding",
        fn=embed,
        args=(table, ids),
        params={"S": S, "E": E, "vocab": vocab},
    )


# ---------------------------------------------------------------------------
# Mechanism probes: small graphs whose only purpose is to move one axis.


def build_elementwise_chain(
    *, ops: int = 64, rows: int = 256, cols: int = 1024
) -> Workload:
    """A chain of ``ops`` pointwise operations.

    The cheapest possible graph-size axis, and the only one that grows the operation
    count without paying for a single matmul -- so a pass whose cost is per-operation
    shows up here uncontaminated by BMM planning.

    This works because ``enable_spyre_context`` forces ``Loops.has_large_inner_fn`` to
    return True (``patches.py``), realizing every operation as its own buffer instead of
    fusing the chain into one inner function. On a backend without that, the whole chain
    would collapse to a single operation and this family would measure nothing.
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

    return Workload(
        name="elementwise_chain",
        fn=chain,
        args=(x,),
        params={"ops": ops, "rows": rows, "cols": cols},
    )


def build_fanout(*, consumers: int = 8, rows: int = 256, cols: int = 1024) -> Workload:
    """One produced buffer read by ``consumers`` operations.

    Grows the consumer count while holding the producer and the operation shapes still,
    which is the axis a consumer-index or users-lookup cost scales on. That is the
    mechanism class behind #4113 -- a reverse index rebuilt per consumer -- and the repo
    has no users abstraction, so nothing else in the sweep moves this axis on its own.
    """
    torch.manual_seed(0)
    x = _randn(rows, cols)

    def fanout(x):
        producer = torch.relu(x)
        total = producer * 1.0
        for i in range(2, consumers + 1):
            total = total + producer * float(i)
        return total

    return Workload(
        name="fanout",
        fn=fanout,
        args=(x,),
        params={"consumers": consumers, "rows": rows, "cols": cols},
    )


def build_dup_constants(
    *, dups: int = 4, B: int = 2, M: int = 8, N: int = 32
) -> Workload:
    """``dups`` unaligned bmms over one shared activation.

    Each unaligned K emits a padding constant, and they are identical, so dedup sees one
    duplicate group of size ``dups``. That is the natural axis of
    ``dedup_and_promote_constants`` -- the pass whose complexity row is already measured
    as ``operations x duplicates`` -- and this is the only family that moves it.

    The fixture is ``tests/inductor/test_dedup_constants.py``'s: fp16, K one element
    past a stick boundary, several weights sharing one activation.
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

    return Workload(
        name="dup_constants",
        fn=dup,
        args=(x, weights),
        params={"dups": dups, "B": B, "M": M, "N": N},
    )


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
    return builder(**params)
