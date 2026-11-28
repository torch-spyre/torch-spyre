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

"""Lightweight profiling check run against every Flex PR.

Flex owns the runtime, memory and telemetry code that the Spyre profiler
depends on (timestamp calibration, AIUPTI activity collection). A Flex change
can therefore break profiling without touching any torch-spyre code.

Placeholder tests
-----------------
These tests are temporary placeholders. Their purpose is to establish and
verify the mechanism that runs torch-spyre tests on every Flex PR (the Flex PR
integration automation): that a Flex PR run reaches torch-spyre, selects this
config, executes the tests on Spyre hardware, and reports a result that can
fail the run. They will be replaced with dedicated Flex-profiling test cases
once the mechanism is confirmed to work.

Duplication with ``test_spyre_profiler.py``
-------------------------------------------
``_find_shifted_kernel_timestamps`` and the matmul workload are intentionally
copied from ``TestSpyreProfiler`` rather than shared, and two of the tests
mirror ``test_kernel_timestamp_after_cpu_dispatch`` and
``test_no_zero_timestamp_or_duration``. Because this file is a placeholder that
will be replaced, a shared helper would add coupling that is removed again
later. The copy is also kept self-contained on purpose: the existing tests are
gated by ``@pytest.mark.requires_spyre_profiler``, which skips silently unless
``USE_SPYRE_PROFILER=1`` is set at run time, and a silent skip on a Flex PR
would read as a pass. Here a missing device, or a profiler that produces no
device events, is a hard failure, so a broken workflow cannot go unnoticed.

Selection
---------
Selected only by the ``integration`` label in
``test_flex_pr_profiling_config.yaml``, the tier that the Spyre CI
``trigger-pr-validation`` run executes for a Flex PR. It is deliberately not
labelled ``regression`` or ``trunk``, so it does not run on runners that may
lack the Spyre device or the profiler extension.
"""

import json
import math

import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from torch.testing._internal.common_utils import TemporaryFileName

_M, _K, _N = 1024, 2048, 1024
_PROFILED_MATMULS = 4


def _has_invalid_timing(event: dict) -> bool:
    return event.get("ts") in (0, None) or event.get("dur") in (0, None)


def _find_shifted_kernel_timestamps(events):
    """Return correlated (cpu_launch, kernel) pairs and those starting too early.

    ``aiuLaunchControlBlocks`` events (``cat == "privateuse1_runtime"``) and
    ``cat == "kernel"`` events share ``args["correlation"]``. A pair is
    "shifted" when the kernel's hardware timestamp precedes the CPU launch,
    meaning the device and host clocks are not correctly correlated.
    """
    cpu_by_corr: dict[int, dict] = {}
    kernels: list[dict] = []

    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        ts = ev.get("ts")
        if not (
            isinstance(ts, (int, float))
            and not isinstance(ts, bool)
            and math.isfinite(ts)
        ):
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
            kernels.append(ev)

    pairs = [
        (cpu_by_corr[int(k["args"]["correlation"])], k)
        for k in kernels
        if int(k["args"]["correlation"]) in cpu_by_corr
    ]
    shifted = [(cpu, k) for cpu, k in pairs if k["ts"] < cpu["ts"]]
    return pairs, shifted


@pytest.fixture(scope="module")
def trace_events():
    """Profile a short matmul workload once and return its Chrome trace events."""
    if not (hasattr(torch, "spyre") and torch.spyre.is_available()):
        pytest.fail(
            "Spyre device is not available. The Flex PR profiling check must "
            "run on Spyre hardware; failing instead of skipping so the run "
            "cannot pass without exercising the profiler."
        )

    device = torch.device("spyre")

    # Inputs stay on CPU so the profiled region contains explicit HtoD copies.
    a_cpu = torch.randn(_M, _K, dtype=torch.float16)
    b_cpu = torch.randn(_K, _N, dtype=torch.float16)

    compiled_matmul = torch.compile(torch.matmul, backend="inductor")

    # Warm up outside the profiled region so compilation is not in the trace,
    # then drain the asynchronous device work.
    compiled_matmul(a_cpu.to(device), b_cpu.to(device))
    torch.spyre.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        a = a_cpu.to(device)
        b = b_cpu.to(device)
        for _ in range(_PROFILED_MATMULS):
            output = compiled_matmul(a, b)
        # Force a memset and a DtoH copy into the profiled region.
        scratch = torch.zeros((_M, _N), dtype=torch.float16, device=device)
        output.cpu()
        del scratch
        # Flush device work so async events land in the trace before export.
        torch.spyre.synchronize()

    with TemporaryFileName(mode="w+") as trace_path:
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            data = json.load(f)

    assert "traceEvents" in data, "Chrome trace is missing 'traceEvents'"
    events = data["traceEvents"]
    assert isinstance(events, list), "'traceEvents' must be a list"

    kernel_events = [e for e in events if e.get("cat") == "kernel"]
    assert kernel_events, (
        "The profiler produced no device kernel events. The profiler "
        "extension may be missing from this build, or Flex/libaiupti is "
        "not delivering AIUPTI activity records."
    )
    return events


def test_profiler_emits_device_events(trace_events):
    """The trace carries the device event categories Flex telemetry produces."""
    memcpy = [e for e in trace_events if e.get("cat") == "gpu_memcpy"]
    assert any("HtoD" in e.get("name", "") for e in memcpy), (
        "Expected at least one HtoD memcpy event"
    )
    assert any("DtoH" in e.get("name", "") for e in memcpy), (
        "Expected at least one DtoH memcpy event"
    )
    assert any(
        e.get("cat") == "gpu_memset" and e.get("name") == "Memset (Device)"
        for e in trace_events
    ), "Expected at least one device memset event"


def test_no_zero_timestamp_or_duration(trace_events):
    """No device event may have ts == 0 or dur == 0."""
    device_events = [
        e
        for e in trace_events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    invalid = [e for e in device_events if _has_invalid_timing(e)]
    assert not invalid, (
        f"{len(invalid)} of {len(device_events)} device event(s) have invalid "
        "ts/dur: " + ", ".join(e.get("name", "<unnamed>") for e in invalid[:10])
    )


def test_kernel_timestamp_after_cpu_dispatch(trace_events):
    """Every kernel starts after the CPU dispatch event that launched it."""
    pairs, shifted = _find_shifted_kernel_timestamps(trace_events)

    assert pairs, (
        "No correlated aiuLaunchControlBlocks <-> kernel pairs found in the trace."
    )

    details = [
        f"  kernel={k.get('name', '<unnamed>')} (ts={k['ts']:.3f}) started "
        f"before cpu_launch={cpu.get('name', '<unnamed>')} (ts={cpu['ts']:.3f}), "
        f"delta={k['ts'] - cpu['ts']:+.3f} us "
        f"[correlation={k.get('args', {}).get('correlation')}]"
        for cpu, k in shifted[:10]
    ]
    assert not shifted, (
        f"{len(shifted)} of {len(pairs)} correlated kernel(s) have a timestamp "
        "earlier than their CPU dispatch (shifted hardware clock):\n"
        + "\n".join(details)
    )
