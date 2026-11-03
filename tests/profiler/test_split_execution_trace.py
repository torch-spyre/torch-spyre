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

"""Split-execution trace tests (issue #2889)."""

import json
import math

import pytest
import torch
import torch._dynamo
from torch.profiler import ProfilerActivity, profile
from torch.testing._internal.common_utils import TemporaryFileName

import torch_spyre  # noqa: F401  # registers the backend

# Device-bound shape: kernel time >> host compute time.
_MM_M, _MM_K, _MM_N = 2048, 8192, 2048
_MM_DTYPE = torch.float16
_NUM_LAUNCHES = 12
# Minimum share of launches (after the first) that must pass each overlap check.
_MIN_RAN_AHEAD_FRACTION = 0.8
_MIN_PARTLY_INSIDE_FRACTION = 0.5


def _complete_events(trace_events: list[dict]) -> list[dict]:
    """Complete (ph=='X') events with finite ts."""
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


def _kernel_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("cat") == "kernel"]


def _memcpy_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("cat") == "gpu_memcpy"]


def _host_compute_spans(events: list[dict]) -> list[dict]:
    """CPU-side launch::HostCompute spans."""
    return [
        e
        for e in events
        if e.get("name") == "launch::HostCompute" and e.get("cat") != "kernel"
    ]


def _skew_us(events: list[dict]) -> float:
    """Median host-to-device clock skew in µs."""
    cpu_by_corr: dict[int, dict] = {}
    kernel_list: list[dict] = []

    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        corr = ev.get("args", {}).get("correlation")
        if corr is None:
            continue
        if (
            ev.get("cat") == "privateuse1_runtime"
            and ev.get("name") == "aiuLaunchControlBlocks"
        ):
            cpu_by_corr[int(corr)] = ev
        elif ev.get("cat") == "kernel":
            kernel_list.append(ev)

    deltas = []
    for k_ev in kernel_list:
        corr = int(k_ev.get("args", {}).get("correlation", -1))
        if corr in cpu_by_corr:
            cpu_ev = cpu_by_corr[corr]
            deltas.append(k_ev["ts"] - cpu_ev["ts"])

    if not deltas:
        return 0.0
    deltas.sort()
    mid = len(deltas) // 2
    return deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2.0


def _starts_inside(inner: dict, outer: dict) -> bool:
    return outer["ts"] <= inner["ts"] < outer["ts"] + outer.get("dur", 0)


def _ends_before(a: dict, b: dict) -> bool:
    return (a["ts"] + a.get("dur", 0)) < (b["ts"] + b.get("dur", 0))


@pytest.fixture(scope="module")
def compiled_mm():
    """Compile and warm up mm once, in the pytest process (the profiler cannot
    span processes)."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = torch.device("spyre")

    a = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE).to(device)
    b = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE).to(device)

    fn = torch.compile(torch.mm, backend="inductor")
    fn(a, b)
    torch.accelerator.synchronize(device)
    return fn


def _profile_back_to_back(compiled_fn, num_launches: int) -> dict:
    """Run mm num_launches times under the profiler.

    Returns "events" (calibrated timestamps, unadjusted), "outputs", "refs" and
    "skew_us" (diagnostic only). No extra skew correction: it would absorb real
    queuing delay and could fake overlap.
    """
    device = torch.device("spyre")
    torch.manual_seed(3)

    pairs_cpu = [
        (
            torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE),
            torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE),
        )
        for _ in range(num_launches)
    ]
    # Upload before profiling so only correction H2Ds appear in the trace.
    pairs_spyre = [(a.to(device), b.to(device)) for a, b in pairs_cpu]
    torch.accelerator.synchronize(device)

    refs = [torch.mm(a.float(), b.float()) for a, b in pairs_cpu]

    with TemporaryFileName(mode="w+") as trace_path:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        ) as prof:
            # No device-to-host reads here: they would serialize the launches.
            outs = [compiled_fn(a_s, b_s) for a_s, b_s in pairs_spyre]
            torch.accelerator.synchronize(device)

        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            trace_data = json.load(f)

    outputs_cpu = [out.cpu().float() for out in outs]

    raw_events = _complete_events(trace_data.get("traceEvents", []))
    # Skew is printed only, never applied to events.
    skew = _skew_us(raw_events)

    return {
        "events": raw_events,
        "outputs": outputs_cpu,
        "refs": refs,
        "skew_us": skew,
    }


def _print_overlap_counts(
    kernels: list[dict], hc_spans: list[dict]
) -> tuple[int, int, int]:
    """Print per-launch overlap; return (ran_ahead, partly_inside, launches).

    Both lists must be sorted by ts.
    """
    num_spans = min(len(kernels), len(hc_spans))
    ran_ahead = partly_inside = 0
    for m in range(1, num_spans):
        hc_start = hc_spans[m]["ts"]
        hc_end = hc_start + hc_spans[m].get("dur", 0)

        # ran_ahead: HostCompute[m] ends before kernel[m-1] ends.
        k_prev_end = kernels[m - 1]["ts"] + kernels[m - 1].get("dur", 0)
        ahead = hc_end < k_prev_end
        if ahead:
            ran_ahead += 1

        # partly_inside: HostCompute[m] intersects any earlier kernel.
        overlapping_kernel_idx = None
        for j in range(m):
            k_start = kernels[j]["ts"]
            k_end = k_start + kernels[j].get("dur", 0)
            if hc_start < k_end and k_start < hc_end:
                overlapping_kernel_idx = j
                break

        if overlapping_kernel_idx is not None:
            partly_inside += 1
            j = overlapping_kernel_idx
            k_start = kernels[j]["ts"]
            k_end = k_start + kernels[j].get("dur", 0)
            overlap_info = (
                f"overlaps kernel[{j}] (k_dur={k_end - k_start:.0f}µs, "
                f"k_start→hc_start={hc_start - k_start:+.0f}µs, "
                f"hc_end→k_end={k_end - hc_end:+.0f}µs)"
            )
        else:
            k0_start = kernels[0]["ts"]
            overlap_info = (
                f"no overlap (ended {k0_start - hc_end:.0f}µs before kernel[0] start)"
                if hc_end < k0_start
                else "no overlap"
            )

        print(
            f"  launch {m}: hc_dur={hc_end - hc_start:.0f}µs  "
            f"ran_ahead={'YES' if ahead else 'no'}  "
            f"{overlap_info}"
        )

    launches = max(num_spans - 1, 0)
    print(
        f"  → partly_inside={partly_inside}/{launches}  ran_ahead={ran_ahead}/{launches}"
    )
    return ran_ahead, partly_inside, launches


@pytest.mark.requires_spyre_profiler
class TestSplitExecutionTrace:
    """Trace checks for split execution (issue #2889).

    Needs SPYRE_HAZARD_TRACKER=1 (latched at startup) and USE_SPYRE_PROFILER=1.
    Set FLEX_TIMESTAMP_CALIBRATION=always and FLEX_SKIP_TIMESTAMP_CALIBRATION=0
    to avoid per-session DMA stalls.
    """

    @pytest.fixture(autouse=True)
    def _require_tracker_on(self):
        import os

        if os.environ.get("SPYRE_HAZARD_TRACKER") != "1":
            pytest.skip(
                "SPYRE_HAZARD_TRACKER=1 is required for split-execution trace tests"
            )

    def test_host_compute_span_present(self, compiled_mm):
        """The launch::HostCompute CPU span is in the trace."""
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES)
        hc_spans = _host_compute_spans(result["events"])
        assert hc_spans, (
            "No launch::HostCompute spans in the trace: AIUPTI events are "
            "missing (check USE_SPYRE_PROFILER=1) or the span was removed."
        )

    def test_host_compute_runs_ahead_of_kernel(self, compiled_mm):
        """HostCompute for later launches overlaps earlier kernels.

        ran_ahead: HostCompute[m] ends before kernel[m-1] ends.
        partly_inside: HostCompute[m] intersects an earlier kernel
        (hc_start < k_end and k_start < hc_end).

        The counts are the same with the tracker off, so this checks
        asynchronous dispatch, not the split itself.
        """
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES)
        kernels = sorted(_kernel_events(result["events"]), key=lambda e: e["ts"])
        hc_spans = sorted(_host_compute_spans(result["events"]), key=lambda e: e["ts"])
        num_spans = min(len(kernels), len(hc_spans))
        assert num_spans >= 2, (
            f"Need at least 2 kernels and 2 HostCompute spans, got "
            f"{len(kernels)} kernels and {len(hc_spans)} spans."
        )

        print(
            f"\nskew_us={result['skew_us']:.2f} (diagnostic only)  spans={num_spans - 1}"
        )
        ran_ahead, partly_inside, launches = _print_overlap_counts(kernels, hc_spans)
        # Worst of 10 device runs: ran_ahead 11/11, partly_inside 7/11.
        min_ran_ahead = math.ceil(_MIN_RAN_AHEAD_FRACTION * launches)
        min_partly_inside = math.ceil(_MIN_PARTLY_INSIDE_FRACTION * launches)
        assert ran_ahead >= min_ran_ahead, (
            f"HostCompute finished before the preceding kernel ended for only "
            f"{ran_ahead}/{launches} launches (need {min_ran_ahead}); host prep "
            f"is not running ahead of device compute."
        )
        assert partly_inside >= min_partly_inside, (
            f"HostCompute overlapped a preceding in-flight kernel for only "
            f"{partly_inside}/{launches} launches (need {min_partly_inside}); "
            f"host prep is not running concurrently with device compute."
        )

    def test_raw_edge_kernel_after_h2d(self, compiled_mm):
        """RAW edge: kernel k starts after its own correction H2D ends."""
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES)

        kernels = sorted(_kernel_events(result["events"]), key=lambda e: e["ts"])
        memcpys = sorted(_memcpy_events(result["events"]), key=lambda e: e["ts"])

        assert kernels, "No kernel events — check AIUPTI."
        assert memcpys, "No memcpy (H2D) events — check AIUPTI."

        # One correction H2D per launch, so h2ds[k] pairs with kernels[k].
        h2ds = sorted(
            [m for m in memcpys if "HtoD" in m.get("name", "")],
            key=lambda e: e["ts"],
        )
        assert h2ds, "No HtoD memcpy events — check AIUPTI."

        num_pairs = min(len(kernels), len(h2ds))
        violations = []
        for k in range(num_pairs):
            h2d_end = h2ds[k]["ts"] + h2ds[k].get("dur", 0)
            if kernels[k]["ts"] < h2d_end:
                violations.append(
                    f"kernel[{k}] ts={kernels[k]['ts']:.3f} started before "
                    f"H2D[{k}] ended at {h2d_end:.3f} "
                    f"(delta={kernels[k]['ts'] - h2d_end:.3f} µs)"
                )

        assert not violations, (
            "RAW edge violation(s) — kernel(s) started before their correction "
            "H2D completed:\n"
            + "\n".join(violations)
            + f"\nskew_us={result['skew_us']:.2f}"
        )

    def test_war_edge_h2d_after_preceding_kernel(self, compiled_mm):
        """WAR edge: correction H2D k+1 starts after kernel k ends."""
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES)

        kernels = sorted(_kernel_events(result["events"]), key=lambda e: e["ts"])
        h2ds = sorted(
            [
                m
                for m in _memcpy_events(result["events"])
                if "HtoD" in m.get("name", "")
            ],
            key=lambda e: e["ts"],
        )

        assert len(kernels) >= 2, "Need ≥ 2 kernel events to check WAR edge."
        assert len(h2ds) >= 2, "Need ≥ 2 H2D events to check WAR edge."

        violations = []
        num_pairs = min(len(kernels), len(h2ds)) - 1
        for k in range(num_pairs):
            kernel_k_end = kernels[k]["ts"] + kernels[k].get("dur", 0)
            h2d_kp1_start = h2ds[k + 1]["ts"]
            if h2d_kp1_start < kernel_k_end:
                violations.append(
                    f"H2D[{k + 1}] ts={h2d_kp1_start:.3f} started before "
                    f"kernel[{k}] ended at {kernel_k_end:.3f} "
                    f"(delta={h2d_kp1_start - kernel_k_end:.3f} µs)"
                )

        assert not violations, (
            "WAR edge violation(s) — H2D started before preceding kernel "
            "completed (missing WAR hazard edge):\n"
            + "\n".join(violations)
            + f"\nskew_us={result['skew_us']:.2f}"
        )

    def test_profiled_outputs_match_cpu(self, compiled_mm):
        """Outputs collected inside the profiled region match the CPU reference."""
        result = _profile_back_to_back(compiled_mm, _NUM_LAUNCHES)

        for k, (out, ref) in enumerate(zip(result["outputs"], result["refs"])):
            rel_frob = (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()
            assert rel_frob < 0.01, (
                f"Launch {k}: relative Frobenius error {rel_frob:.4e} exceeds 1% "
                f"inside profiled region — profiling may have corrupted execution."
            )

    def test_reprofiled_window_after_calibration(self, compiled_mm):
        """A second profiling session still yields valid kernel events.

        Needs FLEX_TIMESTAMP_CALIBRATION=always and
        FLEX_SKIP_TIMESTAMP_CALIBRATION=0 (calibrate at startup, not in-session).
        """
        device = torch.device("spyre")
        torch.manual_seed(5)

        a = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE).to(device)
        b = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE).to(device)

        with TemporaryFileName(mode="w+") as _:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            ) as prof1:
                compiled_mm(a, b)
                torch.accelerator.synchronize(device)
            prof1.export_chrome_trace(_)

        a2 = torch.randn(_MM_M, _MM_K, dtype=_MM_DTYPE).to(device)
        b2 = torch.randn(_MM_K, _MM_N, dtype=_MM_DTYPE).to(device)

        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            ) as prof2:
                compiled_mm(a2, b2)
                torch.accelerator.synchronize(device)
            prof2.export_chrome_trace(trace_path)
            with open(trace_path) as f:
                trace_data = json.load(f)

        events = _complete_events(trace_data.get("traceEvents", []))
        kernels = _kernel_events(events)

        assert kernels, (
            "No kernel events in re-profiled window (second session).  "
            "The calibration stall may have disrupted DMA/AIUPTI in the second "
            "session.  Set FLEX_TIMESTAMP_CALIBRATION=always and "
            "FLEX_SKIP_TIMESTAMP_CALIBRATION=0 to move calibration to runtime start."
        )

        invalid = [
            e
            for e in kernels
            if not (
                isinstance(e.get("ts"), (int, float))
                and math.isfinite(e["ts"])
                and e["ts"] > 0
                and e.get("dur", 0) > 0
            )
        ]
        assert not invalid, (
            f"{len(invalid)} kernel event(s) in re-profiled window have zero or "
            f"non-finite ts/dur — device clock may be broken after calibration stall: "
            + ", ".join(str(e.get("ts")) for e in invalid[:5])
        )
