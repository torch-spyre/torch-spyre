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

"""Benchmark split-execution (SPYRE_HAZARD_TRACKER=1) vs single-stream (0)."""

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

WORKLOADS: dict[str, tuple[int, int, int]] = {
    "mm_large": (1024, 8192, 1024),
    "mm_small": (256, 1024, 256),
    "mm_tiled": (1024, 1024, 1024),
}

TILE_SIZES: dict[str, int] = {"mm_tiled": 256}

MODES = {"tracker_on": "1", "tracker_off": "0"}


def build_compiled(tile_size: int | None) -> Any:
    """Return compiled mm or tiled_mm with for_each_tile."""
    import torch

    if tile_size is None:
        return torch.compile(torch.mm, backend="inductor")

    from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

    def tiled_mm(x: Any, y: Any) -> Any:
        def body(_: Any, ops: Any) -> Any:
            x_tile, y_whole = ops
            return None, x_tile @ y_whole

        _, out = for_each_tile(
            body, (x, y), dims=(0, None), tile_size=tile_size, out_dim=0
        )
        return out

    return torch.compile(tiled_mm, backend="inductor", fullgraph=True)


def _child_bench(
    shape: tuple[int, int, int],
    tile_size: int | None,
    num_launches: int,
    reps: int,
    warmup: int,
) -> dict[str, Any]:
    """Measure per-launch latency in ms and held memory in a child process."""
    import torch
    import torch._dynamo

    import torch_spyre  # noqa: F401
    from torch_spyre import _C

    m, k, n = shape
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = torch.device("spyre")

    pairs = [
        (
            torch.randn(m, k, dtype=torch.float16).to(device),
            torch.randn(k, n, dtype=torch.float16).to(device),
        )
        for _ in range(num_launches)
    ]
    compiled = build_compiled(tile_size)

    held_bytes: list[int] = []

    def run_once() -> float:
        torch.accelerator.synchronize(device)
        free_before, _ = torch.accelerator.get_memory_info("spyre")
        t0 = time.perf_counter()
        outs = [compiled(a, b) for a, b in pairs]
        torch.accelerator.synchronize(device)
        elapsed = time.perf_counter() - t0
        free_after, _ = torch.accelerator.get_memory_info("spyre")
        held_bytes.append(free_before - free_after)
        del outs
        return elapsed * 1000.0 / num_launches

    compiled(*pairs[0])
    for _ in range(warmup):
        run_once()
    held_bytes.clear()
    samples = [run_once() for _ in range(reps)]

    torch.accelerator.synchronize(device)
    return {
        "tracker_enabled": _C.get_hazard_tracker_enabled(),
        "per_launch_ms": samples,
        "held_mib": statistics.median(held_bytes) / 2**20,
    }


def _child_wrapper(env: dict[str, str], target: Any, args: tuple, conn: Any) -> None:
    import os

    try:
        os.environ.update(env)
        result = target(*args)
        conn.send((True, result))
    except Exception as e:  # noqa: BLE001
        conn.send((False, f"{e}\n{traceback.format_exc()}"))
    finally:
        conn.close()
        os._exit(0)


def _run_child(
    env: dict[str, str], target: Any, args: tuple, timeout: float
) -> dict[str, Any]:
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

    p.join(timeout=10.0)
    if p.is_alive():
        p.terminate()
        p.join()
        if ok is None:
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
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="launches enqueued before each sync",
    )
    parser.add_argument("--reps", type=int, default=5, help="timed reps per child")
    parser.add_argument("--warmup", type=int, default=2, help="untimed reps per child")
    parser.add_argument(
        "--rounds", type=int, default=3, help="interleaved on/off child pairs"
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds/child")
    parser.add_argument("--json", type=pathlib.Path, help="write results to JSON")
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
            tile_size = TILE_SIZES.get(wl)
            for depth in ns.launches:
                samples: dict[str, list[float]] = {mode: [] for mode in MODES}
                held: dict[str, list[float]] = {mode: [] for mode in MODES}
                for rnd in range(ns.rounds):
                    for mode, flag in MODES.items():
                        print(
                            f"[{wl} x{depth}] round {rnd + 1}/{ns.rounds} {mode}...",
                            file=sys.stderr,
                            flush=True,
                        )
                        env = {
                            "TORCHINDUCTOR_CACHE_DIR": cache,
                            "SPYRE_NUM_CPUS": "1",
                            "SPYRE_HAZARD_TRACKER": flag,
                        }
                        out = _run_child(
                            env,
                            _child_bench,
                            (shape, tile_size, depth, ns.reps, ns.warmup),
                            ns.timeout,
                        )
                        if out["tracker_enabled"] != (flag == "1"):
                            raise RuntimeError(
                                f"{wl}/{mode}: tracker flag not applied "
                                f"(runtime reports {out['tracker_enabled']})"
                            )
                        samples[mode].extend(out["per_launch_ms"])
                        held[mode].append(out["held_mib"])
                        time.sleep(0.5)

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
                        "tile_size": tile_size,
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
