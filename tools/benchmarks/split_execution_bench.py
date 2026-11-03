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

"""Wall-clock benchmark: split execution (tracker on) vs single stream (off).

Compares per-launch latency of back-to-back compiled mm launches with
SPYRE_HAZARD_TRACKER=1 (HostCompute on S_prep, Compute on S_dev) against
SPYRE_HAZARD_TRACKER=0 (every step on S_dev).

The difference between the two modes is the combined effect of stream
splitting and flex hazard tracking; the two cannot be separated with this
flag alone. It is not a measurement of individual Signal/Wait pairs.

The runtime latches the tracker flag at startup, so every measurement runs in
its own spawned child process. Modes are interleaved round by round so slow
drift (thermal, other device users) affects both modes equally. All children
share one compile cache, so both modes run the same compiled program.

Requires Spyre hardware. Example:

    python tools/benchmarks/split_execution_bench.py --rounds 5 --json out.json
"""

import argparse
import json
import multiprocessing
import pathlib
import statistics
import sys
import tempfile
import time
import traceback
from typing import Any

# name -> (M, K, N). mm_large is device-bound (kernel time >> host compute).
WORKLOADS: dict[str, tuple[int, int, int]] = {
    "mm_large": (1024, 8192, 1024),
    "mm_small": (256, 1024, 256),
}

MODES = {"tracker_on": "1", "tracker_off": "0"}


def _child_bench(
    shape: tuple[int, int, int], num_launches: int, reps: int, warmup: int
) -> dict[str, Any]:
    """Child: return per-launch latency in ms for each timed repetition."""
    import torch
    import torch._dynamo

    import torch_spyre  # noqa: F401  # registers the backend
    from torch_spyre import _C

    m, k, n = shape
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = torch.device("spyre")

    # Distinct inputs per launch so consecutive launches do not alias.
    pairs = [
        (
            torch.randn(m, k, dtype=torch.float16).to(device),
            torch.randn(k, n, dtype=torch.float16).to(device),
        )
        for _ in range(num_launches)
    ]
    compiled = torch.compile(torch.mm, backend="inductor")

    held_bytes: list[int] = []

    def run_once() -> float:
        # No device-to-host reads in the loop: they would serialize launches.
        torch.accelerator.synchronize(device)
        free_before, _ = torch.accelerator.get_memory_info("spyre")
        t0 = time.perf_counter()
        outs = [compiled(a, b) for a, b in pairs]
        torch.accelerator.synchronize(device)
        elapsed = time.perf_counter() - t0
        # Outputs are still alive here: this is the memory the batch holds
        # once drained, not the transient peak while launches are in flight.
        free_after, _ = torch.accelerator.get_memory_info("spyre")
        held_bytes.append(free_before - free_after)
        del outs
        return elapsed * 1000.0 / num_launches

    compiled(*pairs[0])  # trigger compilation outside the timed region
    for _ in range(warmup):
        run_once()
    held_bytes.clear()
    samples = [run_once() for _ in range(reps)]
    return {
        "tracker_enabled": _C.get_hazard_tracker_enabled(),
        "per_launch_ms": samples,
        "held_mib": statistics.median(held_bytes) / 2**20,
    }


def _child_wrapper(env: dict[str, str], target: Any, args: tuple, conn: Any) -> None:
    import os

    try:
        # The runtime reads env vars once at startup: set them first.
        os.environ.update(env)
        conn.send((True, target(*args)))
    except Exception as e:  # noqa: BLE001
        conn.send((False, f"{e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def _run_child(
    env: dict[str, str], target: Any, args: tuple, timeout: float
) -> dict[str, Any]:
    """Run target(*args) in a spawned child with env applied before startup."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_wrapper, args=(env, target, args, child_conn))
    p.start()
    child_conn.close()

    ok: bool | None = None
    payload: Any = None
    if parent_conn.poll(timeout):
        try:
            ok, payload = parent_conn.recv()
        except EOFError:
            pass
    p.join(timeout=5.0)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"child timed out after {timeout}s")
    parent_conn.close()

    if ok is not True:
        raise RuntimeError(f"child failed (exitcode={p.exitcode}):\n{payload}")
    return payload


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "n": float(len(samples)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--workload",
        action="append",
        choices=sorted(WORKLOADS),
        help="workload(s) to run (default: all)",
    )
    parser.add_argument(
        "--launches",
        type=int,
        nargs="+",
        default=[12],
        help="launches enqueued before each sync; several values sweep the "
        "queue depth (throughput and held memory vs depth)",
    )
    parser.add_argument("--reps", type=int, default=5, help="timed reps per child")
    parser.add_argument("--warmup", type=int, default=2, help="untimed reps per child")
    parser.add_argument(
        "--rounds", type=int, default=3, help="interleaved on/off child pairs"
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds/child")
    parser.add_argument("--json", type=pathlib.Path, help="write results to this file")
    ns = parser.parse_args()

    if any(n < 1 for n in ns.launches):
        parser.error("--launches values must be >= 1")
    for name in ("reps", "rounds"):
        if getattr(ns, name) < 1:
            parser.error(f"--{name} must be >= 1")
    if ns.warmup < 0:
        parser.error("--warmup must be >= 0")

    workloads = ns.workload or sorted(WORKLOADS)
    results: dict[str, Any] = {
        "reps": ns.reps,
        "warmup": ns.warmup,
        "rounds": ns.rounds,
        "runs": [],
    }

    with tempfile.TemporaryDirectory(prefix="spyre_bench_cache_") as cache:
        for wl in workloads:
            shape = WORKLOADS[wl]
            for depth in ns.launches:
                samples: dict[str, list[float]] = {mode: [] for mode in MODES}
                held: dict[str, list[float]] = {mode: [] for mode in MODES}
                for rnd in range(ns.rounds):
                    for mode, flag in MODES.items():
                        env = {
                            "TORCHINDUCTOR_CACHE_DIR": cache,
                            "SPYRE_NUM_CPUS": "1",
                            "SPYRE_HAZARD_TRACKER": flag,
                        }
                        out = _run_child(
                            env,
                            _child_bench,
                            (shape, depth, ns.reps, ns.warmup),
                            ns.timeout,
                        )
                        if out["tracker_enabled"] != (flag == "1"):
                            raise RuntimeError(
                                f"{wl}/{mode}: tracker flag not applied "
                                f"(runtime reports {out['tracker_enabled']})"
                            )
                        samples[mode].extend(out["per_launch_ms"])
                        held[mode].append(out["held_mib"])
                    print(
                        f"[{wl} x{depth}] round {rnd + 1}/{ns.rounds} done",
                        file=sys.stderr,
                    )

                summary = {mode: _summarize(s) for mode, s in samples.items()}
                for mode in MODES:
                    summary[mode]["held_mib"] = statistics.median(held[mode])
                ratio = (
                    summary["tracker_on"]["median_ms"]
                    / summary["tracker_off"]["median_ms"]
                )
                results["runs"].append(
                    {
                        "workload": wl,
                        "shape_mkn": list(shape),
                        "launches": depth,
                        "summary": summary,
                        "on_over_off_median": ratio,
                        "samples": samples,
                    }
                )

    print(f"\nreps={ns.reps} rounds={ns.rounds} (ms per launch; held = MiB)")
    print(
        f"{'workload':<10} {'depth':>5} {'mode':<12} {'median':>9} {'min':>9} "
        f"{'max':>9} {'stdev':>9} {'held_MiB':>9}"
    )
    for r in results["runs"]:
        for mode, s in r["summary"].items():
            print(
                f"{r['workload']:<10} {r['launches']:>5} {mode:<12} "
                f"{s['median_ms']:>9.3f} {s['min_ms']:>9.3f} {s['max_ms']:>9.3f} "
                f"{s['stdev_ms']:>9.3f} {s['held_mib']:>9.1f}"
            )
        print(
            f"{r['workload']:<10} {r['launches']:>5} on/off median ratio: "
            f"{r['on_over_off_median']:.3f}"
        )

    if ns.json:
        ns.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
