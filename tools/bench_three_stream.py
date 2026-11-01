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

"""2-stream vs 3-stream wall-time benchmark.

Runs eight scenarios back-to-back, each comparing two-stream (default) against
three-stream (TORCH_SPYRE_THREE_STREAM=1):

  1. single_matmul    — one x @ weight per step (merged HostCompute+H2D, s_prep)
  2. pointwise_chain  — abs → relu → neg per step (standalone H2D, exercises s_dma)
  3. matmul_chain     — two matmuls + relu per step (longer step sequence)
  4. m_sweep          — M in [1,2,4,8,16,32,128]; prints a crossover table
  5. decode_attn      — bmm: (1, heads, 1, head_dim) @ (1, heads, head_dim, seqlen)
  6. prefill          — M=512, same K/N; should show ~1.0x (no regression)
  7. deep_mlp         — 4× linear+gelu chain; more corrections per step
  8. tall_thin_kn     — M=1, K=512, N=8192; largest relative correction fraction

Usage:
    python tools/bench_three_stream.py
    python tools/bench_three_stream.py --steps 200 --m 1 --k 8192 --n 8192
    python tools/bench_three_stream.py --steps 100 --m 4 --k 4096 --n 4096

Output per topology per scenario:

    ┌─ single_matmul ──────────────────────────────────────────────────────┐
      two_stream    total_ms=X.X  per_step_ms=X.XXX  min_ms=X.X  cv=X.X%
      three_stream  total_ms=X.X  per_step_ms=X.XXX  min_ms=X.X  cv=X.X%
    ▲ 3-stream FASTER  +X.X ms/step (X.XXXx)
"""

import argparse
import os
import statistics
import time
from collections.abc import Callable

import torch
import torch._dynamo
import torch_spyre  # noqa: F401 — registers the spyre backend


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=1, help="Batch/M dim (default: 1)")
    p.add_argument("--k", type=int, default=4096, help="K dim (default: 4096)")
    p.add_argument("--n", type=int, default=4096, help="N dim (default: 4096)")
    p.add_argument(
        "--steps", type=int, default=100, help="Decode steps per trial (default: 100)"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("BENCH_WARMUP", "5")),
        help="Warmup iterations before measurement (default: 5)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=int(os.environ.get("BENCH_REPS", "5")),
        help="Number of full trials to take the median over (default: 5)",
    )
    # decode_attn knobs
    p.add_argument(
        "--heads", type=int, default=16, help="Attention heads (default: 16)"
    )
    p.add_argument(
        "--head-dim", type=int, default=64, help="Head dimension (default: 64)"
    )
    p.add_argument(
        "--seqlen", type=int, default=512, help="KV sequence length (default: 512)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_trial(compiled_fn: Callable, x: torch.Tensor, steps: int) -> float:
    """Run `steps` back-to-back steps without draining between them.

    No .cpu() inside the loop — that would block the CPU and kill the overlap.
    Synchronize once at the end so wall time covers the full device pipeline.
    """
    t0 = time.perf_counter()
    for _ in range(steps):
        compiled_fn(x)
    torch.spyre.synchronize()
    return (time.perf_counter() - t0) * 1e3  # ms


def _compile_and_benchmark(
    topology_name: str,
    env_val: str | None,
    fn_factory: Callable[[], Callable],
    x: torch.Tensor,
    steps: int,
    warmup: int,
    reps: int,
) -> float:
    """Compile fn_factory() under the given topology and measure wall time.

    Args:
        topology_name:  "two_stream" or "three_stream".
        env_val:        Value for TORCH_SPYRE_THREE_STREAM, or None to unset.
        fn_factory:     Callable that returns the Python function to compile.
                        Called after dynamo reset so each topology gets a fresh
                        compile and its own kernel binary.
        x:              Input tensor already on device.
        steps:          Steps per trial.
        warmup:         Warmup trials (triggers compilation).
        reps:           Measurement trials; median is reported.

    Returns:
        Median total wall time in ms over all reps.
    """
    if env_val is None:
        os.environ.pop("TORCH_SPYRE_THREE_STREAM", None)
    else:
        os.environ["TORCH_SPYRE_THREE_STREAM"] = env_val

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn_factory(), backend="inductor")

    for _ in range(warmup):
        _run_trial(compiled_fn, x, steps)

    totals = [_run_trial(compiled_fn, x, steps) for _ in range(reps)]

    median = statistics.median(totals)
    minimum = min(totals)
    std = statistics.pstdev(totals) if len(totals) > 1 else 0.0
    cv = std / statistics.fmean(totals) * 100.0 if totals else 0.0

    print(
        f"  {topology_name:<14s}  "
        f"total_ms={median:7.1f}  per_step_ms={median / steps:.3f}  "
        f"min_ms={minimum:7.1f}  cv={cv:4.1f}%"
    )
    return median


def _run_scenario(
    scenario_name: str,
    fn_factory: Callable[[], Callable],
    x: torch.Tensor,
    args: argparse.Namespace,
    title: str = "",
) -> float:
    """Run two_stream then three_stream for one scenario and print the speedup.

    Returns the speedup ratio (two_ms / three_ms).
    """
    width = 70
    display = f"{scenario_name}  —  {title}" if title else scenario_name
    fill = max(0, width - len(display) - 4)
    print(f"┌─ {display} {'─' * fill}┐")
    two_ms = _compile_and_benchmark(
        "two_stream",
        None,
        fn_factory,
        x,
        args.steps,
        args.warmup,
        args.reps,
    )
    three_ms = _compile_and_benchmark(
        "three_stream",
        "1",
        fn_factory,
        x,
        args.steps,
        args.warmup,
        args.reps,
    )
    speedup = two_ms / three_ms if three_ms > 0 else float("inf")
    delta_ms = (two_ms - three_ms) / args.steps  # ms saved per step (positive = faster)
    if speedup > 1.02:
        icon, verdict = (
            "▲",
            f"3-stream FASTER  +{delta_ms:.3f} ms/step  ({speedup:.3f}x)",
        )
    elif speedup < 0.98:
        icon, verdict = (
            "▼",
            f"3-stream SLOWER  {delta_ms:.3f} ms/step  ({speedup:.3f}x)",
        )
    else:
        icon, verdict = "~", f"no meaningful difference  ({speedup:.3f}x)"
    print(f"{icon} {verdict}")
    return speedup


# ---------------------------------------------------------------------------
# Scenario functions
# ---------------------------------------------------------------------------


def bench_single_matmul(args: argparse.Namespace) -> None:
    """Scenario 1: one decode-projection matmul per step.

    x @ weight uses a merged HostCompute+H2D step (s_prep in both topologies).
    This is the baseline workload where correction overhead is most visible at
    small M.
    """

    def fn_factory() -> Callable:
        weight = torch.randn(args.k, args.n, dtype=torch.float16).to("spyre")

        def fn(x: torch.Tensor) -> torch.Tensor:
            return x @ weight

        return fn

    x = torch.randn(args.m, args.k, dtype=torch.float16).to("spyre")
    _run_scenario(
        "single_matmul",
        fn_factory,
        x,
        args,
        title=f"Single matmul  x@W  (M={args.m}, K={args.k}, N={args.n})",
    )


def bench_pointwise_chain(args: argparse.Namespace) -> None:
    """Scenario 2: abs → relu → neg pointwise chain per step.

    Pointwise ops with scalar-only corrections produce standalone H2D steps,
    which are routed to s_dma in the three-stream topology.  This is the
    workload that directly exercises the new DMA stream path.
    """

    def fn_factory() -> Callable:
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.neg(torch.relu(torch.abs(x)))

        return fn

    x = torch.randn(args.m, args.k, dtype=torch.float16).to("spyre")
    _run_scenario(
        "pointwise_chain",
        fn_factory,
        x,
        args,
        title="Pointwise chain  abs→relu→neg  (exercises s_dma)",
    )


def bench_matmul_chain(args: argparse.Namespace) -> None:
    """Scenario 3: two matmuls + relu per step.

    Produces a longer step sequence (HostCompute, H2D, Compute, HostCompute,
    H2D, Compute) that better stresses cross-step overlap and correction
    pipelining between consecutive launches.
    """

    def fn_factory() -> Callable:
        w1 = torch.randn(args.k, args.n, dtype=torch.float16).to("spyre")
        w2 = torch.randn(args.n, args.n, dtype=torch.float16).to("spyre")

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x @ w1) @ w2

        return fn

    x = torch.randn(args.m, args.k, dtype=torch.float16).to("spyre")
    _run_scenario(
        "matmul_chain",
        fn_factory,
        x,
        args,
        title="Two matmuls + relu  (longer step sequence)",
    )


def bench_m_sweep(args: argparse.Namespace) -> None:
    """Scenario 4: sweep M in [1, 2, 4, 8, 16, 32, 128] over fixed K/N.

    The correction blob is fixed per op regardless of M, so the correction
    fraction shrinks as M grows and compute dominates.  This sweep finds the
    exact crossover point where 3-stream stops helping on the current hardware.
    """
    sweep_values = [1, 2, 4, 8, 16, 32, 128]
    width = 70
    header = f"M sweep  x@W  K={args.k}  N={args.n}  — crossover table"
    fill = max(0, width - len(header) - 4)
    print(f"┌─ m_sweep  —  {header} {'─' * fill}┐")
    print(f"  {'M':>5}  {'speedup':>9}  verdict")
    print(f"  {'─' * 5}  {'─' * 9}  {'─' * 26}")

    for m in sweep_values:

        def fn_factory(m: int = m) -> Callable:
            weight = torch.randn(args.k, args.n, dtype=torch.float16).to("spyre")

            def fn(x: torch.Tensor) -> torch.Tensor:
                return x @ weight

            return fn

        x = torch.randn(m, args.k, dtype=torch.float16).to("spyre")
        label = f"m_sweep_m{m}"
        speedup = _run_scenario(
            label,
            fn_factory,
            x,
            args,
            title=f"M={m}  K={args.k}  N={args.n}",
        )
        if speedup > 1.02:
            icon, note = "▲", "faster"
        elif speedup < 0.98:
            icon, note = "▼", "slower"
        else:
            icon, note = "~", "no diff"
        print(f"  {m:>5}  {speedup:>8.3f}x  {icon} {note}")
        print()


def bench_decode_attn(args: argparse.Namespace) -> None:
    """Scenario 5: decode-step attention via batched matmul.

    Simulates Q @ K^T in a KV-cache decode: Q is (1, heads, 1, head_dim),
    K is (1, heads, head_dim, seqlen).  Q stays tiny (single query token) while
    the K dimension grows with sequence length.  This is the most realistic
    decode workload where 3-stream matters in production.
    """
    heads = args.heads
    head_dim = args.head_dim
    seqlen = args.seqlen

    def fn_factory() -> Callable:
        # K is fixed (the KV cache) — allocate once on device.
        k_cache = torch.randn(1, heads, head_dim, seqlen, dtype=torch.float16).to(
            "spyre"
        )

        def fn(q: torch.Tensor) -> torch.Tensor:
            # q: (1, heads, 1, head_dim) — the single query token
            return torch.matmul(q, k_cache)

        return fn

    q = torch.randn(1, heads, 1, head_dim, dtype=torch.float16).to("spyre")
    _run_scenario(
        "decode_attn",
        fn_factory,
        q,
        args,
        title=f"KV-cache decode  Q@K  (heads={heads}, head_dim={head_dim}, seqlen={seqlen})",
    )


def bench_prefill(args: argparse.Namespace) -> None:
    """Scenario 6: prefill with M=512, same K/N as decode.

    At large M compute dominates corrections entirely — 3-stream should give
    ~1.0x.  Running this alongside bench_single_matmul confirms there is no
    regression on the prefill path.
    """
    prefill_m = 512

    def fn_factory() -> Callable:
        weight = torch.randn(args.k, args.n, dtype=torch.float16).to("spyre")

        def fn(x: torch.Tensor) -> torch.Tensor:
            return x @ weight

        return fn

    x = torch.randn(prefill_m, args.k, dtype=torch.float16).to("spyre")
    _run_scenario(
        "prefill_m512",
        fn_factory,
        x,
        args,
        title=f"Prefill  x@W  (M=512, K={args.k}, N={args.n})  — expect ~1.0x",
    )


def bench_deep_mlp(args: argparse.Namespace) -> None:
    """Scenario 7: four linear+gelu layers in one compiled step.

    More ops per step means more correction uploads per step, giving the DMA
    stream more independent work to pipeline.  This is the closest approximation
    to a real decode MLP block (e.g., the FFN in a transformer layer).
    """

    def fn_factory() -> Callable:
        w1 = torch.randn(args.k, args.n, dtype=torch.float16).to("spyre")
        w2 = torch.randn(args.n, args.n, dtype=torch.float16).to("spyre")
        w3 = torch.randn(args.n, args.n, dtype=torch.float16).to("spyre")
        w4 = torch.randn(args.n, args.k, dtype=torch.float16).to("spyre")

        def fn(x: torch.Tensor) -> torch.Tensor:
            x = torch.nn.functional.gelu(x @ w1)
            x = torch.nn.functional.gelu(x @ w2)
            x = torch.nn.functional.gelu(x @ w3)
            return x @ w4

        return fn

    x = torch.randn(args.m, args.k, dtype=torch.float16).to("spyre")
    _run_scenario(
        "deep_mlp",
        fn_factory,
        x,
        args,
        title="Deep MLP  4x(linear+gelu)  (more corrections per step)",
    )


def bench_tall_thin_kn(args: argparse.Namespace) -> None:
    """Scenario 8: M=1, K=512, N=8192 — tall-and-thin projection.

    Small K means very little compute per step while the correction blob stays
    the same size.  This gives the largest possible correction-to-compute ratio
    and is the best case for 3-stream regardless of N.
    """
    thin_k = 512
    wide_n = 8192

    def fn_factory() -> Callable:
        weight = torch.randn(thin_k, wide_n, dtype=torch.float16).to("spyre")

        def fn(x: torch.Tensor) -> torch.Tensor:
            return x @ weight

        return fn

    x = torch.randn(1, thin_k, dtype=torch.float16).to("spyre")
    _run_scenario(
        "tall_thin_kn",
        fn_factory,
        x,
        args,
        title="Tall-thin projection  x@W  (M=1, K=512, N=8192)  — max correction ratio",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    print(
        f"Workload dims: M={args.m}  K={args.k}  N={args.n}  "
        f"steps={args.steps}  warmup={args.warmup}  reps={args.reps}"
    )
    print(
        f"Attention dims: heads={args.heads}  head_dim={args.head_dim}  "
        f"seqlen={args.seqlen}"
    )
    print(
        "Note: HostCompute+H2D correction overhead is fixed per step regardless "
        "of M.\n      Small M = corrections dominate = best case for 3-stream."
    )
    print("=" * 80)

    bench_single_matmul(args)
    print("-" * 80)
    bench_pointwise_chain(args)
    print("-" * 80)
    bench_matmul_chain(args)
    print("-" * 80)
    bench_m_sweep(args)
    print("-" * 80)
    bench_decode_attn(args)
    print("-" * 80)
    bench_prefill(args)
    print("-" * 80)
    bench_deep_mlp(args)
    print("-" * 80)
    bench_tall_thin_kn(args)
    print("=" * 80)


if __name__ == "__main__":
    main()
