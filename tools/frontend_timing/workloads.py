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


#: Workload name -> builder. Add an entry here and a point in sweep_plan.json.
BUILDERS: dict[str, Callable[..., Workload]] = {
    "flash": build_flash,
    "mlp": build_mlp,
    "transformer_block": build_transformer_block,
    "control_flow": build_control_flow,
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
