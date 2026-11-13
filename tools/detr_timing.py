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

"""Time DETR load / compile / run on Spyre: 2-stream vs 3-stream topology.

Topologies (see torch_spyre/csrc/spyre_stream.cpp, SpyreStream::launch):
  2-stream: SPYRE_HAZARD_TRACKER=1
  3-stream: SPYRE_HAZARD_TRACKER=1 SPYRE_DMA_STREAM=1

SPYRE_HAZARD_TRACKER is latched when the runtime starts, so each topology is
measured in its own subprocess. Per topology it reports:
  * load time    - model_gen() + input image load + transfer to the device
  * compile time - first call of the torch.compile'd model minus the mean
                   steady-state run time (the first call includes one run)
  * run time     - mean/min/max over --iters calls after compilation

Usage:
    python3 tools/detr_timing.py /path/to/detr [--batch-size 1] [--iters 10]
"""

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULT_PREFIX = "DETR_TIMING_RESULT "

# name -> env overrides. None removes the variable from the child environment.
TOPOLOGIES = {
    "2-stream": {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": None},
    "3-stream": {"SPYRE_HAZARD_TRACKER": "1", "SPYRE_DMA_STREAM": "1"},
}


def import_detr(detr_dir: Path):
    """Import ``detr_model.py`` from *detr_dir*."""
    model_file = detr_dir / "detr_model.py"
    if not model_file.is_file():
        sys.exit(f"error: {model_file} not found")
    spec = importlib.util.spec_from_file_location("detr_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(args) -> dict:
    """Run one load/compile/run measurement in the current process."""
    import torch
    import torch_spyre  # noqa: F401

    dtype = getattr(torch, args.dtype)
    detr = import_detr(args.detr_dir)

    def run_once(fn, x):
        # Copy outputs back to the host to force completion.
        return tuple(o.cpu() for o in fn(x))

    # ---- load ----
    t0 = time.perf_counter()
    model = detr.model_gen(args.batch_size)
    x = (
        torch.load(detr.ART_DIR / "input_image.pt", weights_only=False)
        .float()
        .contiguous()
        .repeat(args.batch_size, 1, 1, 1)
    )
    model = model.to(args.device, dtype=dtype)
    x = x.to(args.device, dtype=dtype)
    load_s = time.perf_counter() - t0

    # ---- compile (first call) ----
    compiled = torch.compile(model, dynamic=False)
    with torch.no_grad():
        t0 = time.perf_counter()
        run_once(compiled, x)
        first_s = time.perf_counter() - t0

        # ---- run ----
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            run_once(compiled, x)
            times.append(time.perf_counter() - t0)

    run_mean = statistics.mean(times) if times else 0.0
    return {
        "load_s": load_s,
        "first_s": first_s,
        "compile_s": max(first_s - run_mean, 0.0),
        "run_mean_s": run_mean,
        "run_min_s": min(times) if times else 0.0,
        "run_max_s": max(times) if times else 0.0,
        "iters": len(times),
    }


def run_topology(name: str, args) -> dict | None:
    """Measure one topology in a fresh subprocess; None on failure."""
    env = os.environ.copy()
    for key, value in TOPOLOGIES[name].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.detr_dir),
        "--batch-size",
        str(args.batch_size),
        "--iters",
        str(args.iters),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--worker",
    ]
    print(f"[{name}] running ...", flush=True)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    print(f"[{name}] FAILED (exit code {proc.returncode}); see stderr above")
    return None


def print_comparison(results: dict, args) -> None:
    rows = [
        ("load time (s)", "load_s", 1.0, "{:.3f}"),
        ("first call (s)", "first_s", 1.0, "{:.3f}"),
        ("compile time (s)", "compile_s", 1.0, "{:.3f}"),
        ("run mean (ms)", "run_mean_s", 1e3, "{:.3f}"),
        ("run min (ms)", "run_min_s", 1e3, "{:.3f}"),
        ("run max (ms)", "run_max_s", 1e3, "{:.3f}"),
    ]
    names = list(results)
    print()
    print(f"DETR dir       : {args.detr_dir}")
    print(f"batch size     : {args.batch_size}")
    print(f"device / dtype : {args.device} / {args.dtype}")
    print(f"timed iters    : {args.iters}")
    print()
    header = f"{'':<18}" + "".join(f"{n:>14}" for n in names)
    if len(names) == 2:
        header += f"{'3 vs 2':>12}"
    print(header)
    for label, key, scale, fmt in rows:
        vals = [results[n][key] * scale if results[n] else None for n in names]
        line = f"{label:<18}" + "".join(
            f"{(fmt.format(v) if v is not None else 'FAILED'):>14}" for v in vals
        )
        if len(vals) == 2 and all(v is not None for v in vals) and vals[0] > 0:
            line += f"{(vals[1] / vals[0] - 1) * 100:>+11.1f}%"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("detr_dir", type=Path, help="folder containing detr_model.py")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10, help="timed runs")
    parser.add_argument("--device", default="spyre")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="dtype for model and input on the device",
    )
    parser.add_argument(
        "--topology",
        default="both",
        choices=["2-stream", "3-stream", "both"],
        help="topology to measure (default: both, compared)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.detr_dir = args.detr_dir.expanduser().resolve()

    if args.worker:
        print(RESULT_PREFIX + json.dumps(measure(args)), flush=True)
        return

    names = list(TOPOLOGIES) if args.topology == "both" else [args.topology]
    results = {n: run_topology(n, args) for n in names}
    print_comparison(results, args)
    if any(r is None for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
