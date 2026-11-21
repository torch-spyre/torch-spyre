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

"""Does the 3-stream topology help encoder-style work more than decoder-style?

The same synthetic transformer-block stack is timed under three call patterns,
each in 2-stream and 3-stream (SPYRE_DMA_STREAM) mode:

  independent   encoder-like: N forwards on separate inputs, no dependency
                between them, one sync at the end.
  chained       decoder-like: each output is the next input (device-side data
                dependency), one sync at the end.
  chained+host  autoregressive: each output goes through the host
                (.cpu() -> .to(device)) before the next step, as token
                sampling does.

The model is identical across patterns, so only the data dependency changes.
If decoders gain less from the DMA stream, the 3-vs-2 delta should shrink from
`independent` to `chained` to `chained+host`.

SPYRE_HAZARD_TRACKER is latched at runtime start, so each topology runs in its
own subprocess (same approach as detr_timing.py).

Usage:
    python3 tools/encoder_vs_decoder_timing.py [--seq 128] [--layers 6]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULT_PREFIX = "ENCDEC_TIMING_RESULT "

TOPOLOGIES = {
    "2-stream": {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": None},
    "3-stream": {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": "1"},
}
PATTERNS = ["independent", "chained", "chained+host"]

D_MODEL = 256
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS  # 32
D_HEAD_PAD = 64  # padded to a full fp16 stick, as in detr_model.py
FFN = 2048


def build_model(layers: int):
    import torch
    import torch.nn.functional as F
    from torch import nn

    embed_pad = N_HEADS * D_HEAD_PAD

    class Block(nn.Module):
        # Mirrors the DETR encoder layer that compiled: separate q/k/v
        # projections (no in_proj slicing), padded head dim, (T, B, C) layout.
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

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(layers))

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    return Stack().eval()


def install_missing_buf_debug() -> None:
    """On KeyError in Scheduler.compute_ancestors, say who reads the missing
    buffer and whether anything produces or removed it. Installed before the
    compile, so spyre's own per-compile wrapper sits on top of this one."""
    from torch._inductor.scheduler import Scheduler
    from torch._inductor.virtualized import V

    orig = Scheduler.compute_ancestors

    def compute_ancestors(self):
        try:
            return orig(self)
        except KeyError as e:
            missing = e.args[0]
            w = lambda *a: print("[debug-missing-buf]", *a, file=sys.stderr, flush=True)  # noqa: E731
            w("missing buffer:", missing)
            w("  in graph.removed_buffers:", missing in V.graph.removed_buffers)
            w("  in graph.name_to_buffer :", missing in V.graph.name_to_buffer)
            w("  in scheduler.name_to_buf:", missing in self.name_to_buf)
            for n in self.nodes:
                reads = {d.name for d in n.read_writes.reads}
                if missing in reads:
                    w(f"  READ BY {n.get_name()} ({type(n.node).__name__}):")
                    w("    all reads:", sorted(reads))
                    try:
                        ir_reads = {d.name for d in n.node.get_read_writes().reads}
                        w("    reads per the IR op itself:", sorted(ir_reads))
                        w(
                            "    extra reads added at scheduling:",
                            sorted(reads - ir_reads),
                        )
                    except Exception as exc:
                        w("    (could not read the IR op's own reads:", exc, ")")
                    w("    writes   :", sorted(d.name for d in n.read_writes.writes))
                    w("    origins  :", [str(o) for o in list(n.node.origins)[:5]])
                if missing in {d.name for d in n.read_writes.writes}:
                    w(f"  WRITTEN BY {n.get_name()}")
            for table_name in ("additional_buffer_deps", "additional_star_deps"):
                for key, vals in getattr(V.graph, table_name, {}).items():
                    if missing in vals:
                        w(f"  V.graph.{table_name}[{key!r}] contains {missing}")
            w("  buffers the scheduler knows:", sorted(self.name_to_buf)[:40])
            raise

    Scheduler.compute_ancestors = compute_ancestors


def install_removed_buffer_trace(name: str) -> None:
    """Print the call stack at the moment `name` is added to the graph's
    removed_buffers, which names the pass that removed it."""
    import traceback

    from torch._inductor.virtualized import V
    from torch.utils._ordered_set import OrderedSet

    orig_add = OrderedSet.add

    def add(self, elem):
        if elem == name:
            try:
                is_removed = self is V.graph.removed_buffers
            except Exception:
                is_removed = False
            if is_removed:
                print(f"[trace-removed] {name} added to removed_buffers by:",
                      file=sys.stderr, flush=True)  # fmt: skip
                traceback.print_stack(limit=8, file=sys.stderr)
                # Who writes and who reads `name` in graph.operations right now?
                # (The op being removed is still in the list at this point.)
                w = lambda *a: print("[trace-removed]", *a, file=sys.stderr, flush=True)  # noqa: E731
                for op in list(V.graph.operations):
                    try:
                        rw = op.get_read_writes()
                        reads = {d.name for d in rw.reads}
                        writes = {d.name for d in rw.writes}
                    except Exception as exc:
                        w(f"  could not read deps of {op.get_operation_name()}: {exc}")
                        continue
                    if name in writes:
                        origins = [str(o) for o in list(getattr(op, "origins", ()))[:4]]
                        w(f"  WRITTEN BY {op.get_operation_name()} "
                          f"({type(op).__name__}) origins={origins}")  # fmt: skip
                    if name in reads:
                        w(
                            f"  READ BY    {op.get_operation_name()} ({type(op).__name__})"
                        )
                w("  (no READ BY line above = nothing in graph.operations read it "
                  "when it was removed)")  # fmt: skip
        return orig_add(self, elem)

    OrderedSet.add = add


def measure(args) -> dict:
    import torch
    import torch_spyre  # noqa: F401

    if args.debug_missing_buf:
        install_missing_buf_debug()
    if args.trace_removed:
        install_removed_buffer_trace(args.trace_removed)

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    out = {}

    def sync():
        torch.spyre.synchronize()

    # ---- load: build the model and put weights and inputs on the device ----
    t0 = time.perf_counter()
    model = build_model(args.layers).to(args.device, dtype=dtype)
    shape = (args.seq, args.batch_size, D_MODEL)
    xs = [torch.randn(shape).to(args.device, dtype=dtype) for _ in range(args.iters)]
    sync()
    out["load_s"] = time.perf_counter() - t0

    compiled = torch.compile(model, dynamic=False)
    n = args.iters

    with torch.no_grad():
        # ---- compile: first call = compile + one run ----
        t0 = time.perf_counter()
        compiled(xs[0]).cpu()
        out["first_s"] = time.perf_counter() - t0
        warm = []
        for _ in range(args.warmup):
            t0 = time.perf_counter()
            compiled(xs[0]).cpu()
            warm.append(time.perf_counter() - t0)
        # Estimate, first call minus one warm call.
        out["compile_s"] = (
            max(out["first_s"] - statistics.mean(warm), 0.0) if warm else None
        )

        # ---- run: steady-state per-step time for each call pattern ----

        def run_independent():
            outs = [compiled(x) for x in xs]
            sync()
            return outs

        def run_chained():
            y = xs[0]
            for _ in range(n):
                y = compiled(y)
            sync()
            return y

        def run_chained_host():
            y = xs[0]
            for _ in range(n):
                y = compiled(y).cpu().to(args.device, dtype=dtype)
            sync()
            return y

        runners = {
            "independent": run_independent,
            "chained": run_chained,
            "chained+host": run_chained_host,
        }
        for name, fn in runners.items():
            samples = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) / n)
            out[name] = {
                "per_step_ms": statistics.median(samples) * 1e3,
                "min_ms": min(samples) * 1e3,
                "max_ms": max(samples) * 1e3,
            }
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


def print_phase_times(rounds: list[tuple[dict, dict]]) -> None:
    """Per round: load, compile and each run pattern, with the 3-stream vs
    2-stream difference in percent for every row. No averaging across rounds."""
    rows = [
        ("load (s)", lambda r: r["load_s"]),
        ("compile estimate (s)", lambda r: r["compile_s"]),
        *((f"run {p} (ms/step)", lambda r, p=p: r[p]["per_step_ms"]) for p in PATTERNS),
    ]

    def fmt(v: float | None) -> str:
        return f"{v:>12.3f}" if v is not None else f"{'n/a':>12}"

    def fmt_pct(v: float | None) -> str:
        return f"{v:>+11.1f}%" if v is not None else f"{'n/a':>12}"

    for i, (two, three) in enumerate(rounds, 1):
        print()
        print(f"round {i}")
        print(f"  {'phase':<28}{'2-stream':>12}{'3-stream':>12}{'3 vs 2':>12}")
        for label, fn in rows:
            a, b = fn(two), fn(three)
            pct = None if not a or b is None else (b / a - 1) * 100
            print(f"  {label:<28}{fmt(a)}{fmt(b)}{fmt_pct(pct)}")
    print()
    print("compile = first call minus a warm call; run excludes load and compile.")
    print("3 vs 2 is negative where 3-stream is faster.")


# A round's 3-vs-2 delta smaller than this is treated as no difference.
MIN_EFFECT_PCT = 2.0


def round_delta_pct(two: dict, three: dict) -> float:
    """Mean over patterns of (3-stream / 2-stream - 1) in percent."""
    return statistics.mean(
        (three[p]["per_step_ms"] / two[p]["per_step_ms"] - 1) * 100 for p in PATTERNS
    )


def print_verdict(rounds: list[tuple[dict, dict]], args) -> None:
    deltas = [round_delta_pct(two, three) for two, three in rounds]
    print()
    print(
        f"seq={args.seq} layers={args.layers} batch={args.batch_size} "
        f"steps/pattern={args.iters} repeats={args.repeats} dtype={args.dtype}"
    )
    print()
    print(f"{'round':<8}{'2-stream ms':>14}{'3-stream ms':>14}{'3 vs 2':>10}")
    for i, ((two, three), d) in enumerate(zip(rounds, deltas), 1):
        a = statistics.mean(two[p]["per_step_ms"] for p in PATTERNS)
        b = statistics.mean(three[p]["per_step_ms"] for p in PATTERNS)
        print(f"{i:<8}{a:>14.3f}{b:>14.3f}{d:>+9.1f}%")
    print("(ms = mean over the three call patterns)")
    print_phase_times(rounds)

    median = statistics.median(deltas)
    faster = sum(d < -MIN_EFFECT_PCT for d in deltas)
    slower = sum(d > MIN_EFFECT_PCT for d in deltas)
    print()
    if faster == len(deltas):
        print(f"VERDICT: 3-stream is FASTER in all {len(deltas)} rounds "
              f"(median {-median:.1f}% faster).")  # fmt: skip
    elif slower == len(deltas):
        print(f"VERDICT: 3-stream is SLOWER in all {len(deltas)} rounds "
              f"(median {median:.1f}% slower).")  # fmt: skip
    else:
        print(f"VERDICT: INCONCLUSIVE - 3-stream faster in {faster}, slower in "
              f"{slower}, within +/-{MIN_EFFECT_PCT:.0f}% in "
              f"{len(deltas) - faster - slower} of {len(deltas)} rounds.")  # fmt: skip
        print("Round-to-round noise is as large as any topology effect.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--seq", type=int, default=128, help="tokens per step (avoid 100: #1353)"
    )
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=20, help="steps per pattern")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", default="spyre")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="alternating 2-stream/3-stream rounds (order flips each round)",
    )
    p.add_argument(
        "--debug-missing-buf",
        action="store_true",
        help="with --worker: on KeyError 'bufN' in compute_ancestors, print who "
        "reads the missing buffer",
    )
    p.add_argument(
        "--trace-removed",
        metavar="BUF",
        help="with --worker: print the call stack when BUF (e.g. buf9) is added "
        "to removed_buffers",
    )
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
    print_verdict(rounds, args)


if __name__ == "__main__":
    main()
