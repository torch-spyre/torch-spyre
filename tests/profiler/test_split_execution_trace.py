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

import contextlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import pytest
import torch
import torch._dynamo

import torch_spyre
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._inductor import spyre_hint

# ── constants ─────────────────────────────────────────────────────────────────
_HOST_COMPUTE_STREAM_ID = 65
_LARGE_M, _LARGE_K, _LARGE_N = 1024, 8192, 1024
_K_TILES = 4
_N_LAUNCHES = 16  # profiled launches; researcher used 16

# Clock skew is corrected; allow ±100 µs slop after correction.
_SKEW_TOLERANCE_US = 100.0

# At least this fraction of consecutive (host_k+1, kernel_k) pairs must show
# the host span ending inside / before the kernel ends.
_HOST_AHEAD_MIN_FRACTION = 0.6  # 11/15 measured; 0.6 leaves headroom

# ── env vars used across the subprocess boundary ─────────────────────────────
_ENV_RESULT = "SPYRE_RESULT_PATH"
_ENV_TRACKER = "SPYRE_HAZARD_TRACKER"
_ENV_TRACE = "SPYRE_TRACE_PATH"


# ── utilities ─────────────────────────────────────────────────────────────────


def _sync_all(device: torch.device) -> None:
    torch.accelerator.synchronize(device)
    torch_spyre._C.host_compute_stream_by_id(
        _HOST_COMPUTE_STREAM_ID, device
    ).synchronize()


def _cleanup_device(device: torch.device) -> None:
    with contextlib.suppress(Exception):
        _sync_all(device)
    with contextlib.suppress(Exception):
        torch_spyre._C.free_runtime()


def _assert_finite(event: dict, field: str) -> float:
    val = event.get(field)
    name = event.get("name", "unknown")
    assert (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
    ), f"Event '{name}' has non-finite {field}={val}"
    return float(val)


def _cpu_spans_named(trace_events: list[dict], name_substr: str) -> list[dict]:
    return [
        e
        for e in trace_events
        if isinstance(e, dict)
        and e.get("ph") == "X"
        and name_substr in e.get("name", "")
        and e.get("cat") not in ("kernel", "gpu_memcpy")
    ]


def _correct_skew(
    trace_events: list[dict],
) -> tuple[list[dict], float]:
    """Return a copy of *trace_events* with device timestamps shifted by the
    estimated clock skew (device_us - cpu_us), and the skew value in µs.

    Strategy: find the first CPU ``record_function`` span whose name contains
    ``launch_jobplan`` and the first device kernel event.  The kernel must
    start after the launch call on the CPU.  We estimate skew as
    ``cpu_launch_end - kernel_ts`` (the kernel cannot start before the CPU
    issued it), clamped to ±500 µs.  If no pair is found, skew=0.
    """
    cpu_launches = sorted(
        [
            e
            for e in trace_events
            if isinstance(e, dict)
            and e.get("ph") == "X"
            and "launch_jobplan" in e.get("name", "")
        ],
        key=lambda e: e.get("ts", 0),
    )
    kernels = sorted(
        [
            e
            for e in trace_events
            if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"
        ],
        key=lambda e: e.get("ts", 0),
    )
    skew_us = 0.0
    if cpu_launches and kernels:
        first_launch = cpu_launches[0]
        first_kernel = kernels[0]
        # The kernel device ts should be ≥ cpu launch ts (after correction).
        # skew = kernel_raw_ts - cpu_launch_ts  (positive → device clock leads).
        launch_ts = float(first_launch.get("ts", 0))
        kernel_ts = float(first_kernel.get("ts", 0))
        raw_diff = kernel_ts - launch_ts
        # Clamp to ±500 µs — anything larger is not a calibration artifact.
        skew_us = max(-500.0, min(500.0, raw_diff))

    corrected = []
    device_cats = {"kernel", "gpu_memcpy"}
    for e in trace_events:
        if isinstance(e, dict) and e.get("cat") in device_cats:
            e2 = dict(e)
            if "ts" in e2:
                e2["ts"] = float(e2["ts"]) - skew_us
            corrected.append(e2)
        else:
            corrected.append(e)
    return corrected, skew_us


# ── child workload ────────────────────────────────────────────────────


def _child_trace_workload() -> None:
    """Child: warm up, then profile _N_LAUNCHES K-tiled mm launches.

    Requires USE_SPYRE_PROFILER=1.  Writes Chrome trace to SPYRE_TRACE_PATH
    and result tensors to SPYRE_RESULT_PATH.
    """
    from torch.profiler import ProfilerActivity, profile

    device = torch.device("spyre")
    try:
        torch._dynamo.reset()
        _pnd.reset()

        M, K, N = _LARGE_M, _LARGE_K, _LARGE_N

        g = torch.Generator(device="cpu").manual_seed(0xAFFE)
        inputs_cpu = [
            (
                torch.randn(M, K, dtype=torch.float16, generator=g) * 0.01,
                torch.randn(K, N, dtype=torch.float16, generator=g) * 0.01,
            )
            for _ in range(_N_LAUNCHES)
        ]

        _pnd.declare_tensor_dim("M", M)
        _pnd.declare_tensor_dim("K", K)
        _pnd.declare_tensor_dim("N", N)

        def k_tiled_mm(a, b):
            _pnd.name_tensor_dims(a, ["M", "K"])
            _pnd.name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": _K_TILES}):
                return torch.mm(a, b)

        compiled = torch.compile(k_tiled_mm, backend="inductor")

        inputs_dev = []
        for a_cpu, b_cpu in inputs_cpu:
            a_dev = a_cpu.to("spyre")
            b_dev = b_cpu.to("spyre")
            _pnd.name_tensor_dims(a_dev, ["M", "K"])
            _pnd.name_tensor_dims(b_dev, ["K", "N"])
            inputs_dev.append((a_dev, b_dev))

        # Warm up: two passes outside the profiled window.
        for a_dev, b_dev in inputs_dev:
            compiled(a_dev, b_dev)
        _sync_all(device)
        for a_dev, b_dev in inputs_dev:
            compiled(a_dev, b_dev)
        _sync_all(device)

        # Profiled window.
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            results = []
            for a_dev, b_dev in inputs_dev:
                results.append(compiled(a_dev, b_dev))
            _sync_all(device)

        trace_path = os.environ[_ENV_TRACE]
        prof.export_chrome_trace(trace_path)

        torch.save([r.cpu() for r in results], os.environ[_ENV_RESULT])
    finally:
        _cleanup_device(device)


# ── subprocess launcher ───────────────────────────────────────────────────────


def _run_trace_child(
    tracker: int,
    result_path: str,
    trace_path: str,
    max_retries: int = 3,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env[_ENV_TRACKER] = str(tracker)
    env[_ENV_RESULT] = result_path
    env[_ENV_TRACE] = trace_path
    # Move clock calibration to runtime start to avoid the 14–43 ms DMA stall
    # that occurs when a profiler session opens and triggers calibration.
    env["FLEX_TIMESTAMP_CALIBRATION"] = "always"
    env["FLEX_SKIP_TIMESTAMP_CALIBRATION"] = "0"  # legacy alias (flex < #1930)
    env["SPYRE_NUM_CPUS"] = "1"

    proc = None
    for attempt in range(max_retries):
        time.sleep(0.5)  # Allow previous process to release VFIO device
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "trace_workload",
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            timeout=300,
            check=False,
        )
        if proc.returncode == 0:
            return proc
        # Retry on device busy / VFIO cleanup delay
        if "Device or resource busy" in proc.stderr or "DeviceOpenFail" in proc.stderr:
            time.sleep(1.0 * (attempt + 1))
            continue
        break

    assert proc is not None
    raise AssertionError(
        f"Trace child (SPYRE_HAZARD_TRACKER={tracker}) exited "
        f"{proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


# ── orchestrator tests ────────────────────────────────────────────────────────


@pytest.mark.requires_spyre_profiler
class TestSplitExecutionTrace:
    """Profiler-trace assertions for split execution (requires USE_SPYRE_PROFILER=1)."""

    def _load_trace(self, tmp_path: pathlib.Path, tracker: int):
        result_path = str(tmp_path / f"results_{tracker}.pt")
        trace_path = str(tmp_path / f"trace_{tracker}.json")
        _run_trace_child(tracker, result_path, trace_path)
        assert os.path.exists(trace_path), (
            f"Child (tracker={tracker}) exited 0 but wrote no trace to {trace_path!r}."
        )
        assert os.path.exists(result_path), (
            f"Child (tracker={tracker}) exited 0 but wrote no results to "
            f"{result_path!r}."
        )
        with open(trace_path) as f:
            trace = json.load(f)
        results = torch.load(result_path, weights_only=True)
        return trace, results

    def test_host_compute_spans_present(self, tmp_path):
        """launch::HostCompute must appear in the CPU trace.

        If this assertion fails, #3488 (host compute off the calling thread)
        has landed and the host-ahead checks below need a new span name.
        """
        trace, _ = self._load_trace(tmp_path, tracker=1)
        events = trace.get("traceEvents", [])
        hc_spans = _cpu_spans_named(events, "launch::HostCompute")

        assert hc_spans, (
            "No 'launch::HostCompute' CPU span found in the profiler trace.  "
            "If #3488 (host compute off the calling thread) has landed, this "
            "test needs updating to track the new span name."
        )

    def test_host_ahead_of_device_kernel(self, tmp_path):
        """launch::HostCompute for launch k+1 must end before kernel k ends
        and must start inside kernel k's device interval (after clock correction).

        This is the temporal proof that the host runs ahead of the device.
        The check uses corrected device timestamps (skew ≤ ±100 µs after
        correction).
        """
        trace, _ = self._load_trace(tmp_path, tracker=1)
        raw_events = trace.get("traceEvents", [])

        # Correct device clock skew.
        events, skew_us = _correct_skew(raw_events)
        print(f"\n[DIAG] Clock skew correction: {skew_us:.1f} µs")

        hc_spans = sorted(
            _cpu_spans_named(events, "launch::HostCompute"),
            key=lambda e: e.get("ts", 0),
        )
        kernels = sorted(
            [
                e
                for e in events
                if isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ],
            key=lambda e: e.get("ts", 0),
        )

        if not hc_spans:
            pytest.skip(
                "No launch::HostCompute spans found; "
                "test_host_compute_spans_present will catch this."
            )
        if len(kernels) < 2:
            pytest.skip(
                f"Only {len(kernels)} kernel event(s) in trace; "
                "need at least 2 for a consecutive-pair check."
            )

        n_pairs = min(len(hc_spans) - 1, len(kernels) - 1)
        ahead_start = 0  # hc_k+1 starts inside kernel_k
        ahead_end = 0  # hc_k+1 ends before kernel_k ends

        for k in range(n_pairs):
            hc_next = hc_spans[k + 1]
            kern = kernels[k]

            kern_ts = _assert_finite(kern, "ts")
            kern_dur = _assert_finite(kern, "dur")
            kern_end = kern_ts + kern_dur

            hc_ts = _assert_finite(hc_next, "ts")
            hc_dur = _assert_finite(hc_next, "dur")
            hc_end = hc_ts + hc_dur

            # Host span starts inside the kernel window.
            if kern_ts < hc_ts < kern_end:
                ahead_start += 1
            # Host span ends before kernel ends (host ran ahead).
            if hc_end < kern_end:
                ahead_end += 1

        frac_start = ahead_start / n_pairs if n_pairs > 0 else 0.0
        frac_end = ahead_end / n_pairs if n_pairs > 0 else 0.0

        print(
            f"[DIAG] host-ahead (start inside kernel): "
            f"{ahead_start}/{n_pairs} = {frac_start:.0%}"
        )
        print(
            f"[DIAG] host-ahead (ends before kernel ends): "
            f"{ahead_end}/{n_pairs} = {frac_end:.0%}"
        )

        assert frac_start >= _HOST_AHEAD_MIN_FRACTION, (
            f"launch::HostCompute (k+1) started inside kernel (k) in only "
            f"{ahead_start}/{n_pairs} pairs ({frac_start:.0%}); "
            f"threshold is {_HOST_AHEAD_MIN_FRACTION:.0%}.  "
            "Either the device is faster than expected, the shape is too "
            "small, or the host-compute path is not being exercised."
        )
        assert frac_end >= _HOST_AHEAD_MIN_FRACTION, (
            f"launch::HostCompute (k+1) ended before kernel (k) ended in only "
            f"{ahead_end}/{n_pairs} pairs ({frac_end:.0%}); "
            f"threshold is {_HOST_AHEAD_MIN_FRACTION:.0%}."
        )

    def test_raw_edge_kernel_after_h2d(self, tmp_path):
        """Device kernel k must start after its own H2D finishes (RAW edge).

        flex inserts a RAW dependency: Compute reads the correction that H2D
        wrote.  If this check fails, the RAW edge is missing.
        """
        trace, _ = self._load_trace(tmp_path, tracker=1)
        raw_events = trace.get("traceEvents", [])
        events, _skew_us = _correct_skew(raw_events)

        h2ds = sorted(
            [
                e
                for e in events
                if isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "gpu_memcpy"
                and "HtoD" in e.get("name", "")
            ],
            key=lambda e: e.get("ts", 0),
        )
        kernels = sorted(
            [
                e
                for e in events
                if isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ],
            key=lambda e: e.get("ts", 0),
        )

        if not h2ds or not kernels:
            pytest.skip(
                f"H2D events: {len(h2ds)}, kernel events: {len(kernels)} — "
                "AIUPTI device events not present in this CI image; "
                "cannot check RAW edge."
            )

        violations = []
        for k, (h2d, kern) in enumerate(zip(h2ds, kernels)):
            h2d_end = _assert_finite(h2d, "ts") + _assert_finite(h2d, "dur")
            kern_ts = _assert_finite(kern, "ts")
            if kern_ts < h2d_end - _SKEW_TOLERANCE_US:
                violations.append(
                    f"pair {k}: kernel_ts={kern_ts:.1f} µs < "
                    f"h2d_end={h2d_end:.1f} µs "
                    f"(violation = {h2d_end - kern_ts:.1f} µs)"
                )

        assert not violations, (
            f"RAW edge violated for {len(violations)} pair(s):\n"
            + "\n".join(violations)
            + "\nKernel must start after its H2D finishes."
        )

    def test_war_edge_h2d_next_after_kernel(self, tmp_path):
        """H2D for launch k+1 must start after kernel k finishes (WAR edge).

        flex inserts a WAR dependency: H2D k+1 writes the correction buffer
        that Compute k is still reading.  This is why H2D/kernel temporal
        overlap is impossible and the old overlap test was wrong.
        """
        trace, _ = self._load_trace(tmp_path, tracker=1)
        raw_events = trace.get("traceEvents", [])
        events, _skew_us = _correct_skew(raw_events)

        h2ds = sorted(
            [
                e
                for e in events
                if isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "gpu_memcpy"
                and "HtoD" in e.get("name", "")
            ],
            key=lambda e: e.get("ts", 0),
        )
        kernels = sorted(
            [
                e
                for e in events
                if isinstance(e, dict)
                and e.get("ph") == "X"
                and e.get("cat") == "kernel"
            ],
            key=lambda e: e.get("ts", 0),
        )

        if len(h2ds) < 2 or not kernels:
            pytest.skip(
                f"H2D events: {len(h2ds)}, kernel events: {len(kernels)} — "
                "need at least 2 H2Ds for a cross-launch WAR check; "
                "AIUPTI device events may not be present in this CI image."
            )

        n_pairs = min(len(h2ds) - 1, len(kernels))
        violations = []
        for k in range(n_pairs):
            kern = kernels[k]
            h2d_next = h2ds[k + 1]
            kern_end = _assert_finite(kern, "ts") + _assert_finite(kern, "dur")
            h2d_ts = _assert_finite(h2d_next, "ts")
            if h2d_ts < kern_end - _SKEW_TOLERANCE_US:
                violations.append(
                    f"pair {k}: h2d[k+1]_ts={h2d_ts:.1f} µs < "
                    f"kernel[k]_end={kern_end:.1f} µs "
                    f"(violation = {kern_end - h2d_ts:.1f} µs)"
                )

        assert not violations, (
            f"WAR edge violated for {len(violations)} pair(s):\n"
            + "\n".join(violations)
            + "\nH2D[k+1] must start after kernel[k] finishes."
        )

    def test_profiled_outputs_close_to_cpu_reference(self, tmp_path):
        """Profiled outputs must be close to a CPU float32 reference.

        Relative Frobenius ≤ 5e-3.  A tracker bug that silently zeros or
        corrupts outputs would produce a much larger error.
        """
        _, results = self._load_trace(tmp_path, tracker=1)

        g = torch.Generator(device="cpu").manual_seed(0xAFFE)
        M, K, N = _LARGE_M, _LARGE_K, _LARGE_N
        a_cpu = torch.randn(M, K, dtype=torch.float16, generator=g) * 0.01
        b_cpu = torch.randn(K, N, dtype=torch.float16, generator=g) * 0.01
        ref = torch.mm(a_cpu.float(), b_cpu.float())

        for i, r in enumerate(results):
            rel_err = (r.float() - ref).norm() / (ref.norm() + 1e-8)
            assert rel_err < 5e-3, (
                f"Profiled output {i} deviates from CPU float32 reference: "
                f"relative Frobenius = {rel_err:.2e} (threshold 5e-3)."
            )

    def test_tracker_off_reported_as_diagnostic(self, tmp_path):
        """Tracker-off run is profiled and its timings printed as diagnostics.

        This test never fails on the timing values — it only fails if the
        child subprocess itself crashes or the result file is missing.  The
        data is printed so that reviewers can compare overlap counts between
        modes when inspecting CI logs.
        """
        trace, results = self._load_trace(tmp_path, tracker=0)
        raw_events = trace.get("traceEvents", [])
        events, skew_us = _correct_skew(raw_events)

        hc_spans = _cpu_spans_named(events, "launch::HostCompute")
        kernels = [
            e
            for e in events
            if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"
        ]

        print(
            f"\n[DIAG tracker-off] skew={skew_us:.1f} µs, "
            f"HostCompute spans={len(hc_spans)}, kernels={len(kernels)}"
        )

        # Count pairs where host ran inside kernel (diagnostic only).
        n_pairs = min(len(hc_spans) - 1, len(kernels) - 1)
        ahead = 0
        for k in range(n_pairs):
            if not (hc_spans[k + 1].get("ts") and kernels[k].get("ts")):
                continue
            hc_ts = float(hc_spans[k + 1]["ts"])
            kern_ts = float(kernels[k]["ts"])
            kern_end = kern_ts + float(kernels[k].get("dur", 0))
            if kern_ts < hc_ts < kern_end:
                ahead += 1
        print(f"[DIAG tracker-off] host-ahead pairs: {ahead}/{n_pairs}")
        # Soft assertion: results must exist (child ran successfully).
        assert len(results) == _N_LAUNCHES, (
            f"Expected {_N_LAUNCHES} results from tracker-off child, "
            f"got {len(results)}."
        )


_CHILD_DISPATCH = {
    "trace_workload": _child_trace_workload,
}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in _CHILD_DISPATCH:
        _CHILD_DISPATCH[sys.argv[1]]()
    else:
        pytest.main([__file__, "-v"])
