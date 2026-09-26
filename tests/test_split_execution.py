import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import pytest
import torch
import torch._dynamo
from torch.profiler import ProfilerActivity, profile

import torch_spyre
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._inductor import spyre_hint
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

_HOST_COMPUTE_STREAM_START = 65
_PIPELINE_DEPTH = 4

_ENV_RESULT = "SPYRE_RESULT_PATH"
_ENV_TRACKER = "SPYRE_HAZARD_TRACKER"
_ENV_TRACE = "SPYRE_TRACE_PATH"


def _run_child_subprocess(
    tracker: int,
    output: str,
    child_name: str,
    *,
    trace_path: str | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env[_ENV_TRACKER] = str(tracker)
    env[_ENV_RESULT] = output
    if trace_path is not None:
        env[_ENV_TRACE] = trace_path
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "--no-header",
            "--tb=short",
            "-k",
            child_name,
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Child subprocess (SPYRE_HAZARD_TRACKER={tracker}, "
            f"child={child_name}) exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc


def _collect_parity_results(tmp_path, child_name: str) -> tuple:
    off_path = str(tmp_path / "off.pt")
    on_path = str(tmp_path / "on.pt")
    _run_child_subprocess(0, off_path, child_name)
    _run_child_subprocess(1, on_path, child_name)
    for label, path in (("tracker-off", off_path), ("tracker-on", on_path)):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Child for '{child_name}' ({label}) exited 0 but never wrote "
                f"{path!r} — the child test was probably skipped entirely."
            )
    results_off = torch.load(off_path, weights_only=True)
    results_on = torch.load(on_path, weights_only=True)
    return results_off, results_on


def _skip_if_child_process() -> None:
    if os.environ.get(_ENV_RESULT):
        pytest.skip("orchestrator test — not run inside child subprocesses")


def _sync_all(device: torch.device) -> None:
    torch.accelerator.synchronize(device)
    torch_spyre._C.host_compute_stream_by_id(
        _HOST_COMPUTE_STREAM_START, device
    ).synchronize()


def _skip_if_not_child_process() -> None:
    if not os.environ.get(_ENV_RESULT):
        pytest.skip("child-only test — run via _run_child_subprocess(), not directly")


def _assert_plan_has_prep_steps(runner: SpyreSDSCKernelRunner) -> None:
    plan = runner.jobplan
    has_prep = any(
        plan.get_step_stream_role(i) == "Prep" for i in range(plan.num_steps())
    )
    assert has_prep, (
        f"JobPlan for '{runner.kernel_name}' has no Prep-role steps — "
        "deeptools did not emit a ComputeOnHost triple for this shape, so the "
        "split execution path is never exercised and parity passes vacuously."
    )


def _patch_runner_assert_prep() -> None:
    # Intentionally not restored — each child subprocess is a fresh process.
    seen = set()
    orig_run = SpyreSDSCKernelRunner.run

    def _checked_run(self, *args, **kwargs):
        if id(self) not in seen:
            seen.add(id(self))
            _assert_plan_has_prep_steps(self)
        return orig_run(self, *args, **kwargs)

    setattr(SpyreSDSCKernelRunner, "run", _checked_run)


def test_child_k_tiled_mm_workload() -> None:
    """Child: single K-tiled mm with 4 tiles. Forces Prep+Device steps in the JobPlan so the split path is exercised."""
    _skip_if_not_child_process()
    torch._dynamo.reset()
    _pnd.reset()

    M, K, N = 64, 512, 32
    device = torch.device("spyre")
    torch.manual_seed(0xAFFE)
    a_cpu = torch.randn(M, K, dtype=torch.float16) * 0.01
    b_cpu = torch.randn(K, N, dtype=torch.float16) * 0.01

    _pnd.declare_tensor_dim("M", M)
    _pnd.declare_tensor_dim("K", K)
    _pnd.declare_tensor_dim("N", N)

    def k_tiled_mm_fn(a, b):
        _pnd.name_tensor_dims(a, ["M", "K"])
        _pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(num_tiles_per_dim={"K": 4}):
            return torch.mm(a, b)

    a_dev = a_cpu.to("spyre")
    b_dev = b_cpu.to("spyre")
    _pnd.name_tensor_dims(a_dev, ["M", "K"])
    _pnd.name_tensor_dims(b_dev, ["K", "N"])
    compiled = torch.compile(k_tiled_mm_fn, backend="inductor")
    _patch_runner_assert_prep()
    result = compiled(a_dev, b_dev)
    _sync_all(device)

    torch.save([result.cpu()], os.environ[_ENV_RESULT])


def test_child_pipelined_k_tiled_mm_workload() -> None:
    """Child: PIPELINE_DEPTH K-tiled mm launches in sequence, pipelining prep and device streams across launches."""
    _skip_if_not_child_process()
    torch._dynamo.reset()
    _pnd.reset()

    M, K, N = 64, 512, 32
    device = torch.device("spyre")
    torch.manual_seed(0xAFFE)
    inputs = [
        (
            torch.randn(M, K, dtype=torch.float16) * 0.01,
            torch.randn(K, N, dtype=torch.float16) * 0.01,
        )
        for _ in range(_PIPELINE_DEPTH)
    ]

    _pnd.declare_tensor_dim("M", M)
    _pnd.declare_tensor_dim("K", K)
    _pnd.declare_tensor_dim("N", N)

    def k_tiled_mm_fn(a, b):
        _pnd.name_tensor_dims(a, ["M", "K"])
        _pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(num_tiles_per_dim={"K": 4}):
            return torch.mm(a, b)

    compiled = torch.compile(k_tiled_mm_fn, backend="inductor")
    _patch_runner_assert_prep()
    results = []
    for a_cpu, b_cpu in inputs:
        a_dev = a_cpu.to("spyre")
        b_dev = b_cpu.to("spyre")
        _pnd.name_tensor_dims(a_dev, ["M", "K"])
        _pnd.name_tensor_dims(b_dev, ["K", "N"])
        results.append(compiled(a_dev, b_dev))

    _sync_all(device)

    torch.save([r.cpu() for r in results], os.environ[_ENV_RESULT])


class TestWithinLaunchOverlapCorrectness(unittest.TestCase):
    def test_tracker_on_off_parity_within_launch(self):
        """Tracker-on and tracker-off must be byte-identical for a single K-tiled mm. A missing H2D→Compute edge within the launch corrupts K-tile accumulation."""
        _skip_if_child_process()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            results_off, results_on = _collect_parity_results(
                tmp_path, "test_child_k_tiled_mm_workload"
            )
            torch.testing.assert_close(
                results_on[0],
                results_off[0],
                atol=0,
                rtol=0,
                msg=(
                    "K-tiled mm differs between tracker-on and tracker-off — "
                    "a missing cross-stream H2D→Compute edge corrupts the "
                    "partial accumulation within the same launch."
                ),
            )


class TestAcrossLaunchOverlapCorrectness(unittest.TestCase):
    def test_tracker_on_off_parity_pipelined_launches(self):
        """Tracker-on and tracker-off must be byte-identical across all pipelined launches. A missing cross-launch ordering edge lets launch N+1 prep race launch N compute."""
        _skip_if_child_process()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            results_off, results_on = _collect_parity_results(
                tmp_path, "test_child_pipelined_k_tiled_mm_workload"
            )
            for i, (r_off, r_on) in enumerate(zip(results_off, results_on)):
                torch.testing.assert_close(
                    r_on,
                    r_off,
                    atol=0,
                    rtol=0,
                    msg=(
                        f"Launch {i}: tracker-on differs from tracker-off — "
                        "cross-stream ordering is wrong for pipelined launches."
                    ),
                )


class TestFallbackTrackerOff(unittest.TestCase):
    def test_tracker_off_with_prep_steps_matches_tracker_on(self):
        """Tracker-off falls back to single-stream even when the JobPlan has Prep steps. Result must match tracker-on."""
        _skip_if_child_process()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            results_off, results_on = _collect_parity_results(
                tmp_path, "test_child_k_tiled_mm_workload"
            )
            torch.testing.assert_close(
                results_off[0],
                results_on[0],
                atol=0,
                rtol=0,
                msg=(
                    "K-tiled mm with tracker-off does not match tracker-on — "
                    "single-stream fallback produces a different result."
                ),
            )


def _intervals_overlap(ts_a, dur_a, ts_b, dur_b):
    return ts_a < ts_b + dur_b and ts_b < ts_a + dur_a


def _assert_finite_timing(event):
    ts = event.get("ts")
    dur = event.get("dur")
    name = event.get("name", "unknown")
    assert (
        isinstance(ts, (int, float)) and not isinstance(ts, bool) and math.isfinite(ts)
    ), f"Event '{name}' has non-finite ts={ts}"
    assert (
        isinstance(dur, (int, float))
        and not isinstance(dur, bool)
        and math.isfinite(dur)
        and dur > 0
    ), f"Event '{name}' has non-positive or non-finite dur={dur}"


_PROFILER_PIPELINE_DEPTH = 16


def test_child_overlap_workload() -> None:
    """Child: warms up then profiles PROFILER_PIPELINE_DEPTH K-tiled mm launches. Requires USE_SPYRE_PROFILER=1. Exports Chrome trace to SPYRE_TRACE_PATH."""
    _skip_if_not_child_process()

    if os.environ.get("USE_SPYRE_PROFILER") != "1":
        pytest.skip("requires USE_SPYRE_PROFILER=1")
    try:
        import torch_spyre  # noqa: F401
    except RuntimeError as exc:
        pytest.skip(f"Spyre device unavailable: {exc}")

    torch._dynamo.reset()
    _pnd.reset()

    M, K, N = 512, 4096, 512
    device = torch.device("spyre")
    torch.manual_seed(0xAFFE)
    inputs = [
        (
            torch.randn(M, K, dtype=torch.float16) * 0.01,
            torch.randn(K, N, dtype=torch.float16) * 0.01,
        )
        for _ in range(_PROFILER_PIPELINE_DEPTH)
    ]

    _pnd.declare_tensor_dim("M", M)
    _pnd.declare_tensor_dim("K", K)
    _pnd.declare_tensor_dim("N", N)

    def k_tiled_mm_fn(a, b):
        _pnd.name_tensor_dims(a, ["M", "K"])
        _pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(num_tiles_per_dim={"K": 4}):
            return torch.mm(a, b)

    compiled = torch.compile(k_tiled_mm_fn, backend="inductor")

    device_inputs = []
    for a_cpu, b_cpu in inputs:
        a_dev = a_cpu.to("spyre")
        b_dev = b_cpu.to("spyre")
        _pnd.name_tensor_dims(a_dev, ["M", "K"])
        _pnd.name_tensor_dims(b_dev, ["K", "N"])
        device_inputs.append((a_dev, b_dev))

    for a_dev, b_dev in device_inputs:
        compiled(a_dev, b_dev)
    _sync_all(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        for a_dev, b_dev in device_inputs:
            compiled(a_dev, b_dev)
        _sync_all(device)

    trace_path = os.environ[_ENV_TRACE]
    prof.export_chrome_trace(trace_path)

    results = []
    for a_dev, b_dev in device_inputs:
        results.append(compiled(a_dev, b_dev))
    _sync_all(device)
    torch.save([r.cpu() for r in results], os.environ[_ENV_RESULT])


# Temporarily commented out to surface the full failure output for diagnosis.
# Restore once issue #2520 (per-stream pinned buffers) is resolved.
# @pytest.mark.xfail(
#     reason=(
#         "Across-launch H2D/Compute overlap is blocked by a WAR hazard on the "
#         "shared per-JobPlan correction buffer (pinned_buffers). The hazard "
#         "tracker correctly serialises H2D[N+1] behind Compute[N] because both "
#         "touch the same HBM region. Fix tracked in issue #2520 "
#         "(per-stream pinned buffers). Remove xfail once that lands."
#     ),
#     strict=False,
# )
def test_pipelined_launches_use_distinct_streams_and_overlap():
    """Asserts H2D and kernel events land on distinct stream tids and their intervals overlap across launches. Blocked by WAR hazard on shared pinned_buffers until issue #2520."""
    _skip_if_child_process()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        result_path = str(tmp_path / "results.pt")
        trace_path = str(tmp_path / "trace.json")

        _run_child_subprocess(
            1, result_path, "test_child_overlap_workload", trace_path=trace_path
        )

        if not os.path.exists(trace_path):
            pytest.skip(
                "Child skipped (Spyre device unavailable or USE_SPYRE_PROFILER!=1)"
            )

        with open(trace_path) as f:
            trace = json.load(f)

    events = trace.get("traceEvents", [])

    h2d_events = [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("ph") == "X"
        and e.get("cat") == "gpu_memcpy"
        and "HtoD" in e.get("name", "")
    ]
    kernel_events = [
        e
        for e in events
        if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") == "kernel"
    ]

    assert h2d_events, "Expected at least one H2D memcpy event in PrivateUse1 trace"
    assert kernel_events, "Expected at least one kernel event in PrivateUse1 trace"

    for e in h2d_events + kernel_events:
        _assert_finite_timing(e)

    print(f"\n[DIAG] H2D events ({len(h2d_events)}):")
    for e in h2d_events:
        print(
            f"  tid={e.get('tid')} ts={e['ts']} dur={e['dur']} end={e['ts'] + e['dur']} name={e.get('name')}"
        )
    print(f"[DIAG] kernel events ({len(kernel_events)}):")
    for e in kernel_events:
        print(
            f"  tid={e.get('tid')} ts={e['ts']} dur={e['dur']} end={e['ts'] + e['dur']} name={e.get('name')}"
        )

    h2d_tids = {e["tid"] for e in h2d_events if "tid" in e}
    kernel_tids = {e["tid"] for e in kernel_events if "tid" in e}

    if not h2d_tids or not kernel_tids:
        pytest.skip(
            "tid not present in PrivateUse1 trace for H2D or kernel events — "
            "stream-distinctness check deferred until profiler exposes stream IDs."
        )

    assert h2d_tids.isdisjoint(kernel_tids), (
        f"H2D and kernel events share stream IDs {h2d_tids & kernel_tids} — "
        "expected H2D on S_prep and Compute on S_dev to use distinct streams."
    )

    found_overlap = any(
        _intervals_overlap(h["ts"], h["dur"], k["ts"], k["dur"])
        for h in h2d_events
        for k in kernel_events
    )
    assert found_overlap, (
        "No H2D/kernel interval overlap found in the trace — "
        "pipelining is not occurring across launches even with "
        f"SPYRE_HAZARD_TRACKER=1.  "
        f"H2D events: {len(h2d_events)}, kernel events: {len(kernel_events)}. "
        "Check that the hazard tracker is inserting cross-stream H2D→Compute "
        "edges and that S_prep was registered with track_hazards=true."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
