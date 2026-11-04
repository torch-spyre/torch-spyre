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

"""Time DETR model load, compile and run on Spyre.

Reports:
  * load time    - ``model_gen()`` + input image load + transfer to the device
  * compile time - first call of the ``torch.compile``d model (includes one
                   execution; the steady-state run time is subtracted to
                   estimate pure compile time)
  * run time     - mean/min/max over ``--iters`` calls after compilation

Usage:
    python3 tools/detr_timing.py /path/to/detr [--batch-size 1] [--iters 10]
"""

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_spyre  # noqa: F401


def import_detr(detr_dir: Path):
    """Import ``detr_model.py`` from *detr_dir*."""
    model_file = detr_dir / "detr_model.py"
    if not model_file.is_file():
        sys.exit(f"error: {model_file} not found")
    spec = importlib.util.spec_from_file_location("detr_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_once(fn, x):
    """Run *fn* and force completion by copying outputs back to the host."""
    out = fn(x)
    return tuple(o.cpu() for o in out)


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
    args = parser.parse_args()

    detr_dir = args.detr_dir.expanduser().resolve()
    dtype = getattr(torch, args.dtype)
    detr = import_detr(detr_dir)

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
    compile_s = max(first_s - run_mean, 0.0)

    print(f"DETR dir        : {detr_dir}")
    print(f"batch size      : {args.batch_size}")
    print(f"device / dtype  : {args.device} / {args.dtype}")
    print(f"load time       : {load_s:.3f} s")
    print(f"first call      : {first_s:.3f} s (compile + 1 run)")
    print(f"compile time    : {compile_s:.3f} s (first call - mean run)")
    if times:
        print(
            f"run time        : mean {run_mean * 1e3:.3f} ms, "
            f"min {min(times) * 1e3:.3f} ms, max {max(times) * 1e3:.3f} ms "
            f"({len(times)} iters)"
        )


if __name__ == "__main__":
    main()
