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


def build_mlp(*, seq_len: int = 256, emb_dim: int = 1024, layers: int = 1) -> Workload:
    """Stacked SwiGLU MLP.

    From ``tests/inductor/test_building_blocks.py::test_mlp``, with a layer loop so the
    graph grows linearly in ``layers`` while every other dimension holds still.
    """
    torch.manual_seed(0)
    x = _randn(seq_len, emb_dim)
    weights = []
    for _ in range(layers):
        weights.append(
            (
                _randn(emb_dim, 4 * emb_dim),
                _randn(emb_dim, 4 * emb_dim),
                _randn(4 * emb_dim, emb_dim),
            )
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
        params={"seq_len": seq_len, "emb_dim": emb_dim, "layers": layers},
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
