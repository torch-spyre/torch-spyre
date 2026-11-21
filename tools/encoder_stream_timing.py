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

"""Encoder workload: how much does 3-stream help over 2-stream?

An encoder-style transformer stack (independent forwards, no data dependency
between them) is timed under 2-stream (SPYRE_HAZARD_TRACKER=1) and 3-stream
(+ SPYRE_DMA_STREAM=1). Each topology runs in its own subprocess because the
hazard tracker is latched at runtime start. Topologies alternate over rounds,
order flipped each round.

Reported per round: load, compile estimate, run (total time per forward) and
enqueue (time the host spends launching one forward before the final sync).
While the host runs on the caller thread, enqueue is close to run and neither
topology can overlap much; when host work moves off the caller thread (pending
host-side PRs), enqueue should drop below run and any topology difference
should show. Run this script unchanged on that branch.

--control compares 2-stream against 2-stream. Its "difference" is the noise
floor that a real 3-vs-2 result has to beat.

Each block is compiled as its own graph. Compiling two or more attention layers
in one graph fails with KeyError 'bufN' (multi-output SDPA kernel plus dead-code
elimination), and writing attention as plain bmm/exp/sum fails in the matmul
layout and restickify passes. One attention call per graph avoids both, at the
cost of a graph boundary between layers.

Usage:
    python3 tools/encoder_stream_timing.py [--layers 6] [--seq 128] [--rounds 6]
    python3 tools/encoder_stream_timing.py --control
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULT_PREFIX = "ENCODER_STREAM_RESULT "

TWO_STREAM = {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": None}
THREE_STREAM = {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": "1"}

D_MODEL = 256
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS  # 32
D_HEAD_PAD = 64  # padded to a full fp16 stick, as in detr_model.py
FFN = 2048

# A round's difference smaller than this counts as no difference.
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
            attn = torch.bmm(q, k.transpose(1, 2)).softmax(-1)
            out = torch.bmm(attn, v).transpose(0, 1).contiguous().view(T, B, H * Dh)
            x = self.norm1(x + self.out_proj(out))
            return self.norm2(x + self.linear2(F.relu(self.linear1(x))))

    return nn.ModuleList(Block() for _ in range(layers)).eval()


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

    blocks = [torch.compile(b, dynamic=False) for b in model]

    def forward(x):
        for b in blocks:
            x = b(x)
        return x

    with torch.no_grad():
        # compile: first call = compile + one run
        t0 = time.perf_counter()
        forward(xs[0]).cpu()
        first = time.perf_counter() - t0
        warm = []
        for _ in range(args.warmup):
            t0 = time.perf_counter()
            forward(xs[0]).cpu()
            warm.append(time.perf_counter() - t0)
        # Estimate, first call minus one warm call.
        out["compile_s"] = max(first - statistics.mean(warm), 0.0)

        # run: independent forwards, one sync at the end. enqueue is the time
        # until the last forward has been launched; run adds the wait for the
        # device to finish.
        run, enqueue = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for x in xs:
                forward(x)
            t_enq = time.perf_counter()
            sync()
            t_end = time.perf_counter()
            enqueue.append((t_enq - t0) / args.iters)
            run.append((t_end - t0) / args.iters)
        out["run_ms"] = statistics.median(run) * 1e3
        out["enqueue_ms"] = statistics.median(enqueue) * 1e3
    return out


def run_topology(label: str, env_overrides: dict, args) -> dict | None:
    env = os.environ.copy()
    for key, value in env_overrides.items():
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
    print(f"[{label}] running ...", flush=True)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
        else:
            print(line)
    if result is None:
        print(f"[{label}] FAILED (exit code {proc.returncode}); see stderr above")
    return result


def pct(a: float, b: float) -> float:
    """(b / a - 1) * 100. Negative means b is faster than a."""
    return (b / a - 1) * 100


def print_report(rounds: list[tuple[dict, dict]], args, b_label: str) -> None:
    print()
    print(
        f"layers={args.layers} seq={args.seq} batch={args.batch_size} "
        f"forwards/repeat={args.iters} repeats={args.repeats} dtype={args.dtype}"
    )
    rows = [
        ("load (s)", "load_s"),
        ("compile estimate (s)", "compile_s"),
        ("run (ms/forward)", "run_ms"),
        ("enqueue (ms/forward)", "enqueue_ms"),
    ]
    for i, (a, b) in enumerate(rounds, 1):
        print()
        print(f"round {i}")
        print(f"  {'phase':<24}{'2-stream':>12}{b_label:>16}{'diff':>10}")
        for label, key in rows:
            print(
                f"  {label:<24}{a[key]:>12.3f}{b[key]:>16.3f}"
                f"{pct(a[key], b[key]):>+9.1f}%"
            )
        print(
            f"  enqueue / run: {a['enqueue_ms'] / a['run_ms']:.0%} (2-stream), "
            f"{b['enqueue_ms'] / b['run_ms']:.0%} ({b_label})"
        )
    print()
    print("compile = first call minus a warm call; run excludes load and compile.")
    print("diff is negative where the right column is faster.")
    print("enqueue near 100% of run means the host launch path is the limit.")

    deltas = [pct(a["run_ms"], b["run_ms"]) for a, b in rounds]
    faster = sum(d < -MIN_EFFECT_PCT for d in deltas)
    slower = sum(d > MIN_EFFECT_PCT for d in deltas)
    n = len(deltas)
    median = statistics.median(deltas)
    print()
    if faster == n:
        print(
            f"VERDICT: {b_label} is FASTER in all {n} rounds "
            f"(median {-median:.1f}% faster)."
        )
    elif slower == n:
        print(
            f"VERDICT: {b_label} is SLOWER in all {n} rounds "
            f"(median {median:.1f}% slower)."
        )
    else:
        print(
            f"VERDICT: INCONCLUSIVE - {b_label} faster in {faster}, slower in "
            f"{slower}, within +/-{MIN_EFFECT_PCT:.0f}% in {n - faster - slower} "
            f"of {n} rounds."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seq", type=int, default=128, help="tokens per forward")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=50, help="forwards per repeat")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--rounds", type=int, default=6, help="alternating rounds")
    p.add_argument("--device", default="spyre")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument(
        "--control",
        action="store_true",
        help="compare 2-stream against 2-stream to measure the noise floor",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        print(RESULT_PREFIX + json.dumps(measure(args)), flush=True)
        return

    b_label = "2-stream (ctl)" if args.control else "3-stream"
    b_env = TWO_STREAM if args.control else THREE_STREAM
    sides = {"a": ("2-stream", TWO_STREAM), "b": (b_label, b_env)}
    rounds = []
    for i in range(args.rounds):
        order = ["a", "b"]
        if i % 2:
            order.reverse()  # cancel any first-process-runs-first effect
        print(f"--- round {i + 1}/{args.rounds} ---", flush=True)
        res = {s: run_topology(*sides[s], args) for s in order}
        if any(r is None for r in res.values()):
            sys.exit(1)
        rounds.append((res["a"], res["b"]))
    print_report(rounds, args, b_label)


if __name__ == "__main__":
    main()
