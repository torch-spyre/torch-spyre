#!/usr/bin/env python3
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

"""Is the 3-stream topology faster than 2-stream on a transformer stack?

Times load, compile and run of a synthetic transformer-block stack under
2-stream (SPYRE_HAZARD_TRACKER=1) and 3-stream (+ SPYRE_DMA_STREAM=1). Each
topology runs in its own subprocess because the hazard tracker is latched at
runtime start. Topologies alternate over several rounds, order flipped each
round.

Softmax is written as exp/sum so Inductor does not rewrite attention into the
multi-output SDPA kernel (that rewrite currently breaks compile with
KeyError 'bufN' once the model has two or more attention layers). There is no
max-subtraction, so values are only meaningful for timing.

Usage:
    python3 tools/transformer_stream_timing.py [--layers 6] [--seq 128]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULT_PREFIX = "STREAM_TIMING_RESULT "

TOPOLOGIES = {
    "2-stream": {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": None},
    "3-stream": {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": "1"},
}

D_MODEL = 256
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS  # 32
D_HEAD_PAD = 64  # padded to a full fp16 stick
FFN = 2048

# A round's 3-vs-2 delta smaller than this counts as no difference.
MIN_EFFECT_PCT = 2.0


def build_model(layers: int):
    import torch
    import torch.nn.functional as F
    from torch import nn

    embed_pad = N_HEADS * D_HEAD_PAD

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(D_MODEL, embed_pad)
            self.k_proj = nn.Linear(D_MODEL, embed_pad)
            self.v_proj = nn.Linear(D_MODEL, embed_pad)
            self.out_proj = nn.Linear(embed_pad, D_MODEL)
            self.linear1 = nn.Linear(D_MODEL, FFN)
            self.linear2 = nn.Linear(FFN, D_MODEL)
            self.norm1 = nn.LayerNorm(D_MODEL)
            self.norm2 = nn.LayerNorm(D_MODEL)
            self.scaling = float(D_HEAD) ** -0.5

        def forward(self, x):
            T, B, _ = x.shape
            H, Dh = N_HEADS, D_HEAD_PAD
            q = (self.q_proj(x) * self.scaling).view(T, B * H, Dh).transpose(0, 1)
            k = self.k_proj(x).view(T, B * H, Dh).transpose(0, 1)
            v = self.v_proj(x).view(T, B * H, Dh).transpose(0, 1)
            e = torch.bmm(q, k.transpose(1, 2)).exp()
            attn = e / e.sum(-1, keepdim=True)
            out = torch.bmm(attn, v).transpose(0, 1).contiguous().view(T, B, H * Dh)
            x = self.norm1(x + self.out_proj(out))
            return self.norm2(x + self.linear2(F.relu(self.linear1(x))))

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(layers))

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    return Stack().eval()


def measure(args) -> dict:
    import torch
    import torch_spyre  # noqa: F401

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    out = {}

    def sync():
        torch.spyre.synchronize()

    # load: build the model and put weights and inputs on the device
    t0 = time.perf_counter()
    model = build_model(args.layers).to(args.device, dtype=dtype)
    shape = (args.seq, args.batch_size, D_MODEL)
    xs = [torch.randn(shape).to(args.device, dtype=dtype) for _ in range(args.iters)]
    sync()
    out["load_s"] = time.perf_counter() - t0

    compiled = torch.compile(model, dynamic=False)

    with torch.no_grad():
        # compile: first call = compile + one run
        t0 = time.perf_counter()
        compiled(xs[0]).cpu()
        first = time.perf_counter() - t0
        warm = []
        for _ in range(args.warmup):
            t0 = time.perf_counter()
            compiled(xs[0]).cpu()
            warm.append(time.perf_counter() - t0)
        # Estimate, first call minus one warm call.
        out["compile_s"] = max(first - statistics.mean(warm), 0.0)

        # run: steady-state time per forward, independent inputs, one sync
        samples = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for x in xs:
                compiled(x)
            sync()
            samples.append((time.perf_counter() - t0) / args.iters)
        out["run_ms"] = statistics.median(samples) * 1e3
    return out


def run_topology(name: str, args) -> dict | None:
    env = os.environ.copy()
    for key, value in TOPOLOGIES[name].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--seq", str(args.seq),
        "--layers", str(args.layers),
        "--batch-size", str(args.batch_size),
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--device", args.device,
        "--dtype", args.dtype,
        "--worker",
    ]  # fmt: skip
    print(f"[{name}] running ...", flush=True)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
        else:
            print(line)
    if result is None:
        print(f"[{name}] FAILED (exit code {proc.returncode}); see stderr above")
    return result


def pct(two: float, three: float) -> float:
    """(3-stream / 2-stream - 1) * 100. Negative means 3-stream is faster."""
    return (three / two - 1) * 100


def print_report(rounds: list[tuple[dict, dict]], args) -> None:
    print()
    print(
        f"layers={args.layers} seq={args.seq} batch={args.batch_size} "
        f"forwards/repeat={args.iters} repeats={args.repeats} dtype={args.dtype}"
    )
    rows = [
        ("load (s)", "load_s"),
        ("compile estimate (s)", "compile_s"),
        ("run (ms/forward)", "run_ms"),
    ]
    for i, (two, three) in enumerate(rounds, 1):
        print()
        print(f"round {i}")
        print(f"  {'phase':<24}{'2-stream':>12}{'3-stream':>12}{'3 vs 2':>10}")
        for label, key in rows:
            a, b = two[key], three[key]
            print(f"  {label:<24}{a:>12.3f}{b:>12.3f}{pct(a, b):>+9.1f}%")
    print()
    print("compile = first call minus a warm call; run excludes load and compile.")
    print("3 vs 2 is negative where 3-stream is faster.")

    deltas = [pct(two["run_ms"], three["run_ms"]) for two, three in rounds]
    faster = sum(d < -MIN_EFFECT_PCT for d in deltas)
    slower = sum(d > MIN_EFFECT_PCT for d in deltas)
    n = len(deltas)
    median = statistics.median(deltas)
    print()
    if faster == n:
        print(
            f"VERDICT: 3-stream is FASTER in all {n} rounds (median {-median:.1f}% faster)."
        )
    elif slower == n:
        print(
            f"VERDICT: 3-stream is SLOWER in all {n} rounds (median {median:.1f}% slower)."
        )
    else:
        print(
            f"VERDICT: INCONCLUSIVE - 3-stream faster in {faster}, slower in "
            f"{slower}, within +/-{MIN_EFFECT_PCT:.0f}% in {n - faster - slower} "
            f"of {n} rounds."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seq", type=int, default=128, help="tokens per forward")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=20, help="forwards per repeat")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--rounds", type=int, default=3, help="2-stream/3-stream rounds")
    p.add_argument("--device", default="spyre")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        print(RESULT_PREFIX + json.dumps(measure(args)), flush=True)
        return

    rounds = []
    for i in range(args.rounds):
        order = list(TOPOLOGIES)
        if i % 2:
            order.reverse()  # cancel any first-process-runs-first effect
        print(f"--- round {i + 1}/{args.rounds} ---", flush=True)
        res = {n: run_topology(n, args) for n in order}
        if any(r is None for r in res.values()):
            sys.exit(1)
        rounds.append((res["2-stream"], res["3-stream"]))
    print_report(rounds, args)


if __name__ == "__main__":
    main()
