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

"""Profiler-based device utilization: split execution vs single stream.

Profiles back-to-back compiled mm launches with SPYRE_HAZARD_TRACKER=1 and
=0 and reports, from the exported chrome trace:

  - kernel busy time (union of ``cat == "kernel"`` intervals),
  - kernel window (first kernel start to last kernel end),
  - utilization = busy / window, and the idle gaps between kernels,
  - transfer time (``cat == "gpu_memcpy"``), which includes correction H2Ds,
  - number of ``launch::HostCompute`` CPU spans (tracker on only).

Utilization is measured over the kernel window, so it shows how tightly the
kernels are packed, not how busy the device is over the whole run.

Each measurement is one profiling session in its own spawned child (the
tracker flag is latched at startup, and re-profiling in one process can be
disturbed by timestamp calibration). Modes are interleaved round by round and
share one compile cache.

Requires Spyre hardware and a build with the profiler (USE_SPYRE_PROFILER=1).
Example:

    python tools/benchmarks/split_execution_profile.py --rounds 5 --json p.json
"""

import argparse
import json
import math
import pathlib
import statistics
import sys
import tempfile
from typing import Any

from split_execution_bench import (
    MODES,
    TILE_SIZES,
    WORKLOADS,
    _run_child,
    build_compiled,
)

# Profiler/calibration settings recommended for the split-execution trace tests.
PROFILER_ENV = {
    "USE_SPYRE_PROFILER": "1",
    "FLEX_TIMESTAMP_CALIBRATION": "always",
    "FLEX_SKIP_TIMESTAMP_CALIBRATION": "0",
}


def _finite_complete_events(trace_events: list[dict]) -> list[dict]:
    """Complete (ph == "X") events with a finite ts."""
    out = []
    for ev in trace_events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        ts = ev.get("ts")
        if (
            isinstance(ts, (int, float))
            and not isinstance(ts, bool)
            and math.isfinite(ts)
        ):
            out.append(ev)
    return out


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of [start, end) intervals, sorted and non-overlapping."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def trace_metrics(trace_events: list[dict]) -> dict[str, Any]:
    """Utilization metrics (all times in microseconds) from chrome trace events."""
    events = _finite_complete_events(trace_events)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    memcpys = [e for e in events if e.get("cat") == "gpu_memcpy"]
    host_spans = [
        e
        for e in events
        if e.get("name") == "launch::HostCompute" and e.get("cat") != "kernel"
    ]
    transfer_us = float(sum(e.get("dur", 0) for e in memcpys))

    if not kernels:
        return {
            "num_kernels": 0,
            "kernel_busy_us": 0.0,
            "window_us": 0.0,
            "utilization": 0.0,
            "idle_us": 0.0,
            "max_gap_us": 0.0,
            "transfer_us": transfer_us,
            "host_compute_spans": len(host_spans),
        }

    merged = _merge_intervals([(e["ts"], e["ts"] + e.get("dur", 0)) for e in kernels])
    busy = sum(end - start for start, end in merged)
    window = merged[-1][1] - merged[0][0]
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    return {
        "num_kernels": len(kernels),
        "kernel_busy_us": float(busy),
        "window_us": float(window),
        "utilization": busy / window if window > 0 else 1.0,
        "idle_us": float(window - busy),
        "max_gap_us": float(max(gaps)) if gaps else 0.0,
        "transfer_us": transfer_us,
        "host_compute_spans": len(host_spans),
    }


def _child_profile(
    shape: tuple[int, int, int], tile_size: int | None, num_launches: int
) -> dict[str, Any]:
    """Child: profile num_launches back-to-back mm launches once."""
    import torch
    import torch._dynamo
    from torch.profiler import ProfilerActivity, profile
    from torch.testing._internal.common_utils import TemporaryFileName

    import torch_spyre  # noqa: F401  # registers the backend
    from torch_spyre import _C

    m, k, n = shape
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = torch.device("spyre")

    # Upload before profiling so only correction H2Ds show up in the trace.
    pairs = [
        (
            torch.randn(m, k, dtype=torch.float16).to(device),
            torch.randn(k, n, dtype=torch.float16).to(device),
        )
        for _ in range(num_launches)
    ]
    compiled = build_compiled(tile_size)
    compiled(*pairs[0])  # compile outside the profiled region
    torch.accelerator.synchronize(device)

    with TemporaryFileName(mode="w+") as trace_path:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        ) as prof:
            # No device-to-host reads here: they would serialize the launches.
            outs = [compiled(a, b) for a, b in pairs]
            torch.accelerator.synchronize(device)
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            trace_data = json.load(f)
    del outs

    result = trace_metrics(trace_data.get("traceEvents", []))
    result["tracker_enabled"] = _C.get_hazard_tracker_enabled()
    return result


_METRIC_KEYS = (
    "kernel_busy_us",
    "window_us",
    "utilization",
    "idle_us",
    "max_gap_us",
    "transfer_us",
    "host_compute_spans",
)


def _aggregate(rounds: list[dict[str, Any]]) -> dict[str, float]:
    """Median of each metric across rounds."""
    return {key: statistics.median(r[key] for r in rounds) for key in _METRIC_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--workload",
        action="append",
        choices=sorted(WORKLOADS),
        help="workload(s) to run (default: all)",
    )
    parser.add_argument("--launches", type=int, default=12, help="launches profiled")
    parser.add_argument(
        "--rounds", type=int, default=3, help="interleaved on/off child pairs"
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds/child")
    parser.add_argument("--json", type=pathlib.Path, help="write results to this file")
    ns = parser.parse_args()

    if ns.launches < 2:
        parser.error("--launches must be >= 2 (gaps need two kernels)")
    if ns.rounds < 1:
        parser.error("--rounds must be >= 1")

    workloads = ns.workload or sorted(WORKLOADS)
    results: dict[str, Any] = {
        "launches": ns.launches,
        "rounds": ns.rounds,
        "workloads": {},
    }

    with tempfile.TemporaryDirectory(prefix="spyre_profile_cache_") as cache:
        for wl in workloads:
            shape = WORKLOADS[wl]
            tile_size = TILE_SIZES.get(wl)
            per_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
            for rnd in range(ns.rounds):
                for mode, flag in MODES.items():
                    env = {
                        "TORCHINDUCTOR_CACHE_DIR": cache,
                        "SPYRE_NUM_CPUS": "1",
                        "SPYRE_HAZARD_TRACKER": flag,
                        **PROFILER_ENV,
                    }
                    out = _run_child(
                        env,
                        _child_profile,
                        (shape, tile_size, ns.launches),
                        ns.timeout,
                    )
                    if out["tracker_enabled"] != (flag == "1"):
                        raise RuntimeError(
                            f"{wl}/{mode}: tracker flag not applied "
                            f"(runtime reports {out['tracker_enabled']})"
                        )
                    if out["num_kernels"] == 0:
                        raise RuntimeError(
                            f"{wl}/{mode}: no kernel events in the trace "
                            "(check the build has the profiler and "
                            "USE_SPYRE_PROFILER=1)"
                        )
                    per_mode[mode].append(out)
                print(f"[{wl}] round {rnd + 1}/{ns.rounds} done", file=sys.stderr)

            results["workloads"][wl] = {
                "shape_mkn": list(shape),
                "tile_size": tile_size,
                "summary": {mode: _aggregate(r) for mode, r in per_mode.items()},
                "rounds": per_mode,
            }

    print(f"\nlaunches={ns.launches} rounds={ns.rounds} (median across rounds)")
    print(
        f"{'workload':<10} {'mode':<12} {'util':>7} {'busy_ms':>9} {'window_ms':>10} "
        f"{'idle_ms':>8} {'maxgap_ms':>10} {'xfer_ms':>8} {'hc_spans':>8}"
    )
    for wl, r in results["workloads"].items():
        for mode, s in r["summary"].items():
            print(
                f"{wl:<10} {mode:<12} {s['utilization']:>7.1%} "
                f"{s['kernel_busy_us'] / 1000:>9.3f} {s['window_us'] / 1000:>10.3f} "
                f"{s['idle_us'] / 1000:>8.3f} {s['max_gap_us'] / 1000:>10.3f} "
                f"{s['transfer_us'] / 1000:>8.3f} {s['host_compute_spans']:>8.0f}"
            )

    if ns.json:
        ns.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
