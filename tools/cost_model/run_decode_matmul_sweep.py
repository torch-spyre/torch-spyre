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

"""Replay the decode weight-delivery calibration on one Spyre card.

From a configured development environment at the repository root::

    python tools/cost_model/run_decode_matmul_sweep.py --output /tmp/decode-full
    python tools/cost_model/run_decode_matmul_sweep.py --sample --output /tmp/decode-67
    python tools/cost_model/run_decode_matmul_sweep.py --sample --list

The default replays all 361 recorded splits; --sample selects the same 67 cases
as the 2026-10-16 stack check. --shape, --rows, and --split restrict either plan.
Each case runs sequentially in its own process with a fresh cache, three warmups,
and 15 synchronized profiled calls. Numerical agreement, one kernel per call,
actual compiled splits, and HBM weight residency are checked before accepting it.

Defaults are card 3, SENCORES=32, DXP_LX_FRAC_AVAIL=0.2, and both LX planners on.
The output retains the plan, settings, raw timings, extracted features, logs, and
traces. Times are raw: the historical data's measured 2.8% card-2 correction is
not assumed to hold on another stack. Use one card for a comparable fresh sweep.
No model download is needed; inputs and weights use seeded random tensors.
"""

import argparse
import json
import os
from pathlib import Path
import random
import runpy
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/inductor/decode_matmul_sweep.py"


def json_number(value):
    if getattr(value, "is_Integer", False):
        return int(value)
    if getattr(value, "is_number", False):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def make_plan(args):
    sweep = runpy.run_path(str(DATA))["SWEEP"]
    plan = []
    for (shape, rows), points in sorted(sweep.items()):
        if args.shape and shape not in args.shape:
            continue
        if rows not in args.rows:
            continue
        selected = {(4, 8, 1), (8, 4, 1), (1, 16, 1), (2, 16, 1), (8, 2, 2)}
        selected.add(tuple(min(points, key=lambda point: point[3])[:3]))
        for m, n, k, original_us in points:
            if args.sample and (m, n, k) not in selected:
                continue
            if args.split and (m, n, k) != tuple(args.split):
                continue
            if m * n * k > args.sencores:
                continue
            plan.append(
                dict(shape=shape, rows=rows, m=m, n=n, k=k, original_us=original_us)
            )
    random.Random(5046).shuffle(plan)
    return plan


def measure(case, output):
    # Import torch before the backend, and only in the child that owns the card.
    import torch
    import torch_spyre  # noqa: F401
    from torch.profiler import ProfilerActivity, profile
    from torch_spyre._C import DataFormats, SpyreTensorLayout
    from torch_spyre._inductor import config, cost_model, dump_cost_model, spyre_hint
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS
    from torch_spyre._inductor.work_division import _decode_weight_delivery_us
    from torch_spyre._inductor.wsr import propagate_named_dims as pnd

    shapes = runpy.run_path(str(DATA))["SHAPES"]
    rows = case["rows"]
    K, N = shapes[case["shape"]]
    hint = {axis: case[axis.lower()] for axis in ("M", "N", "K")}
    # These are the captured Granite projection layouts used by the original sweep.
    layout = SpyreTensorLayout([N // 64, K, 64], [64, N, 1], DataFormats.SEN169_FP16)
    generator = torch.Generator().manual_seed(0)
    x = (torch.randn(rows, K, generator=generator) * 0.02).half().to("spyre")
    weight = (
        (torch.randn(K, N, generator=generator) * 0.02)
        .half()
        .to("spyre", device_layout=layout)
    )
    for name, size in (("M", rows), ("K", K), ("N", N)):
        pnd.declare_tensor_dim(name, size)
    pnd.name_tensor_dims(x, ["M", "K"])
    pnd.name_tensor_dims(weight, ["K", "N"])
    reference = x.cpu().float() @ weight.cpu().float()

    def matmul(x, weight):
        with spyre_hint(
            work_div={axis: split for axis, split in hint.items() if split > 1}
        ):
            return x @ weight

    compiled = torch.compile(matmul, dynamic=False)
    with torch.inference_mode():
        actual = compiled(x, weight)
        for _ in range(3):
            compiled(x, weight)
        torch.spyre.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as profiler:
            for _ in range(15):
                compiled(x, weight)
                torch.spyre.synchronize()
    trace = output / "trace.json"
    profiler.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    samples = sorted(event["dur"] for event in events if event.get("cat") == "kernel")
    assert len(samples) == 15, f"Expected one kernel per call; got {len(samples) / 15}"
    actual = actual.cpu().float()
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    features = dump_cost_model.LAST_FEATS
    matmuls = [op for op in features if op.is_matmul]
    assert len(matmuls) == 1, "Expected one matmul in the extracted features"
    (mm,) = matmuls
    splits = [int(mm.matmul_m_split), int(mm.matmul_n_split), int(mm.reduction_cores)]
    assert splits == [case[axis] for axis in ("m", "n", "k")], splits
    weights = [a for a in mm.args if a.role == "input" and a.elems == K * N]
    assert len(weights) == 1 and not weights[0].is_lx, "Weight must stream from HBM"
    return {
        **case,
        "measured_us": statistics.median(samples),
        "kernel_samples_us": samples,
        "actual_splits": splits,
        "max_abs_err": (actual - reference).abs().max().item(),
        "delivery_prediction_us": _decode_weight_delivery_us(
            K * N * 2, case["m"] * case["n"] * case["k"], case["m"], case["k"]
        ),
        "current_prediction_us": float(cost_model.predict_ops(features, _COST_PARAMS))
        / 1000,
        "features": [cost_model.op_to_dict(op) for op in features],
        "torch": torch.__version__,
        "config": {
            "card": os.environ["SPYRE_DEVICES"],
            "sencores": config.sencores,
            "dxp_lx_frac_avail": config.dxp_lx_frac_avail,
            "lx_planning": config.lx_planning,
            "co_optimizing_lx_planning": config.co_optimizing_lx_planning,
            "layout_solver": config.layout_solver,
            "compile_threads": os.environ["TORCHINDUCTOR_COMPILE_THREADS"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory for this run")
    parser.add_argument(
        "--sample", action="store_true", help="replay the 67-case subset"
    )
    parser.add_argument(
        "--list", action="store_true", help="print plan without hardware"
    )
    parser.add_argument(
        "--shape", nargs="+", choices=["qkv", "o_proj", "gate_up", "down"]
    )
    parser.add_argument(
        "--rows", nargs="+", type=int, choices=[8, 16, 32], default=[8, 16, 32]
    )
    parser.add_argument("--split", nargs=3, type=int, metavar=("M", "N", "K"))
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--sencores", type=int, choices=[16, 32], default=32)
    parser.add_argument("--lx-frac", type=float, default=0.2)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.lx_frac <= 1:
        parser.error("--lx-frac must be between 0 and 1")
    if args.worker_case:
        result = measure(json.loads(args.worker_case), args.output)
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2, default=json_number)
        )
        return
    plan = make_plan(args)
    if not plan:
        parser.error("no recorded cases match this selection")
    if args.list:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --list is used")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    env.update(
        SPYRE_DEVICES=str(args.card),
        SENCORES=str(args.sencores),
        DXP_LX_FRAC_AVAIL=str(args.lx_frac),
        LX_PLANNING="1",
        CO_OPTIMIZING_LX_PLANNING="1",
        OMP_NUM_THREADS="4",
        SPYRE_DUMP_COST="1",
        TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")])),
    )
    recorded_env = {
        key: value
        for key, value in env.items()
        if key
        in {
            "SPYRE_DEVICES",
            "SENCORES",
            "DXP_LX_FRAC_AVAIL",
            "LX_PLANNING",
            "CO_OPTIMIZING_LX_PLANNING",
            "OMP_NUM_THREADS",
            "SPYRE_DUMP_COST",
            "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
            "TORCHINDUCTOR_COMPILE_THREADS",
            "DEEPTOOLS_INSTALL_DIR",
            "SENDNN_INSTALL_DIR",
            "LIBAIUPTI_INSTALL_DIR",
            "SPYRE_COMMS_INSTALL_DIR",
        }
    }
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        ).strip()
    )
    (output / "manifest.json").write_text(
        json.dumps(
            dict(revision=revision, dirty=dirty, env=recorded_env, cases=len(plan)),
            indent=2,
        )
    )
    for index, case in enumerate(plan, 1):
        tag = "{shape}-M{rows}-m{m}n{n}k{k}".format(**case)
        directory = output / tag
        directory.mkdir()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(directory / "cache")
        with (directory / "run.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(directory),
                    "--worker-case",
                    json.dumps(case),
                ],
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=True,
            )
        result = json.loads((directory / "result.json").read_text())
        with (output / "results.jsonl").open("a") as records:
            records.write(json.dumps(result) + "\n")
        print(f"{index}/{len(plan)} {tag}: {result['measured_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
