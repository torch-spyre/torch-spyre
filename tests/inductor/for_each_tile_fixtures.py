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

"""Minimal for_each_tile fixtures for while_loop-splice tests.

A pure map, a pure carry, and an online-softmax multi-leaf carry -- give
test_for_each_tile_lowering.py and test_for_each_tile_e2e.py real
`while_loop` FX nodes to drive via `for_each_tile`
(torch_spyre._inductor.wsr.for_each_tile, landed in #4136).

Not collected by pytest directly (no `test_` prefix, no CI config entry) --
see tests/inductor/utils_inductor.py for the same pattern.
"""

import contextlib
from collections.abc import Callable

import torch
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor.wsr import for_each_tile


M, K, N = 8, 12, 6


def matmul_inputs() -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    torch.manual_seed(0)
    X = torch.randn(M, K)
    Y = torch.randn(K, N)
    return (X, Y), X @ Y


def split_m_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Case A: tile M as a map. Y is invariant; the result tile lays along dim 0."""

    def body(_, ops):
        x_tile, y_whole = ops
        return None, x_tile @ y_whole

    _, out = for_each_tile(body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
    return out


def split_k_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Case C: co-indexed split-K matmul; carry accumulates the partial product."""

    def body(acc, ops):
        x_tile, y_tile = ops
        return acc + x_tile @ y_tile, None

    final, _ = for_each_tile(
        body,
        (X, Y),
        dims=(-1, 0),
        tile_size=3,
        init=torch.zeros(M, N, device=X.device, dtype=X.dtype),
    )
    return final


LQ, LK, D = 128, 256, 128
SOFTMAX_TILE_SIZE = 128


def attention_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    Q = torch.randn(LQ, D, dtype=torch.float16)
    K = torch.randn(LK, D, dtype=torch.float16)
    V = torch.randn(LK, D, dtype=torch.float16)
    return Q, K, V


def online_softmax_fn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """Case D: single for_each_tile loop, 3-leaf carry (m, denom, acc).

    Q is closed over whole (not tiled -- the outer Q-loop is deferred to a
    follow-on nested fixture). K/V are Kind.SLICE, co-indexed and tiled along
    Lk. carry = (m, denom, acc): running max, running sum-of-exp, weighted
    accumulator -- the online-softmax recurrence flash attention's inner
    loop needs.
    """

    def body(carry, tiles):
        m, denom, acc = carry
        k_tile, v_tile = tiles
        scores = Q @ k_tile.transpose(-1, -2)
        m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)
        denom_new = denom * correction + p.sum(dim=-1, keepdim=True)
        acc_new = acc * correction + p @ v_tile
        return (m_new, denom_new, acc_new), None

    m0 = torch.full((Q.shape[0], 1), float("-inf"), device=Q.device, dtype=Q.dtype)
    denom0 = torch.zeros((Q.shape[0], 1), device=Q.device, dtype=Q.dtype)
    acc0 = torch.zeros_like(Q)

    (m, denom, acc), _ = for_each_tile(
        body,
        (K, V),
        dims=(0, 0),
        tile_size=SOFTMAX_TILE_SIZE,
        init=(m0, denom0, acc0),
    )
    return acc / denom


def online_softmax_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    tile_size: int = SOFTMAX_TILE_SIZE,
) -> torch.Tensor:
    """Eager fp32 online-softmax, looped in Python -- isolates fixture-math bugs.

    Computed once here and compared against a straightforward softmax(Q @
    K.T) @ V, so a mismatch between "compiled online_softmax_fn" and this
    function isolates a lowering bug, while a mismatch between this function
    and the straightforward softmax would indicate a fixture-math bug caught
    before it ever reaches the compiler.
    """
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    lk = Kf.shape[0]
    m = torch.full((Qf.shape[0], 1), float("-inf"), dtype=torch.float32)
    denom = torch.zeros((Qf.shape[0], 1), dtype=torch.float32)
    acc = torch.zeros_like(Qf)
    for start in range(0, lk, tile_size):
        k_tile = Kf[start : start + tile_size]
        v_tile = Vf[start : start + tile_size]
        scores = Qf @ k_tile.transpose(-1, -2)
        m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)
        denom = denom * correction + p.sum(dim=-1, keepdim=True)
        acc = acc * correction + p @ v_tile
        m = m_new
    naive = torch.softmax(Qf @ Kf.transpose(-1, -2), dim=-1) @ Vf
    torch.testing.assert_close(acc / denom, naive, atol=1e-2, rtol=1e-2)
    return acc / denom


@contextlib.contextmanager
def _post_grad_graphs():
    """Capture each post-grad graph right after decompose_scan_to_while_loop runs.

    post_grad_custom_post_pass fires BEFORE that decomposition, so a custom
    pass cannot see the while_loop node; wrapping the decomposition itself
    can. Inductor has no built-in hook for "give me the post-grad graph",
    so this monkey-patches the (now permanent, #4136-landed) decomposition
    function for the duration of one compile.
    """
    import torch._inductor.fx_passes.post_grad as pg

    seen: list[torch.fx.GraphModule] = []
    original = pg.decompose_scan_to_while_loop

    def wrapper(gm):
        out = original(gm)
        seen.append(gm)
        return out

    pg.decompose_scan_to_while_loop = wrapper
    try:
        yield seen
    finally:
        pg.decompose_scan_to_while_loop = original


def capture_post_grad_while_loop(
    fn: Callable[..., torch.Tensor], args: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.fx.GraphModule]:
    """Compile fn(*args); return (output, the post-grad graph module).

    Asserts the returned graph module contains a real `while_loop` node
    once fully lowered by decompose_scan_to_while_loop -- these fixtures
    exist specifically to drive that node into torch-spyre's
    CustomPreSchedulingPasses.
    """
    torch._dynamo.reset()
    with _post_grad_graphs() as graphs:
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        out, _code = run_and_get_code(compiled, *args)
    assert graphs, "no post-grad graph captured (FX graph cache hit?)"
    gm = graphs[-1]
    found = any(
        "while_loop" in str(node.target)
        for node in gm.graph.nodes
        if node.op == "call_function"
    )
    assert found, "expected a while_loop node in the post-grad graph"
    return out, gm
