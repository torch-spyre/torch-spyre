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

"""Exercise ordinary torch.profiler export through the native Spyre session."""

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def native_traces(tmp_path_factory):
    if os.environ.get("USE_SPYRE_PROFILER", "1") == "0":
        pytest.skip("Spyre profiler support is disabled")
    if not os.environ.get("RUNTIME_INSTALL_DIR"):
        pytest.skip("Requires the native build environment (RUNTIME_INSTALL_DIR)")
    output = tmp_path_factory.mktemp("native_spyre_trace")
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", MAX_JOBS="4")
    # An isolated process avoids backend auto-init and scopes the synthetic
    # Kineto child profiler to these CPU-only tests.
    subprocess.run(
        [sys.executable, __file__, str(output)], env=env, check=True, timeout=300
    )
    return output


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _flows(events):
    flows = {}
    for event in events:
        if event.get("ph") in ("s", "f") and event.get("cat") == "ac2g":
            flows.setdefault(event["id"], []).append(event)
    return flows


@pytest.mark.parametrize(
    "filename",
    [
        "trace.json",
        "trace.json.gz",
        "repeat.json",
        "scheduled-2.json",
        "scheduled-4.json",
    ],
)
def test_standard_export_contains_both_flow_stages(native_traces, filename):
    trace = _read(native_traces / filename)
    events = trace["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X"]
    cpu = next(event for event in spans if event["name"] == "test_cpu_origin")
    markers = [event for event in spans if event["name"] == "aiuCpuLaunch"]
    submits = [event for event in spans if event["name"] == "aiuSubmitToHardware"]
    launches = [event for event in spans if event["name"] == "aiuLaunchControlBlocks"]
    kernels = [event for event in spans if event["name"] == "synthetic_kernel"]
    assert len(markers) == len(launches) == len(kernels) == 2
    assert len(submits) == 6
    assert all(event["dur"] == 5 for event in submits)
    assert all(event["dur"] == 2 for event in kernels)
    assert all(event["args"]["stream"] == 3 for event in kernels)
    assert all(event["dur"] == 0 for event in markers)
    assert all(
        (event["pid"], event["tid"], event["ts"]) == (cpu["pid"], cpu["tid"], cpu["ts"])
        for event in markers
    )
    assert {event["args"]["correlation"] for event in markers} == {7, 8}
    flows = _flows(events)
    assert len(flows) == 4
    for flow_id, endpoints in flows.items():
        assert len(endpoints) == 2
        start = next(event for event in endpoints if event["ph"] == "s")
        finish = next(event for event in endpoints if event["ph"] == "f")
        assert start["ts"] <= finish["ts"]
        source_spans, target_spans = (
            (markers, submits) if flow_id > 2**32 else (launches, kernels)
        )
        for endpoint, candidates in ((start, source_spans), (finish, target_spans)):
            assert any(
                (endpoint["pid"], endpoint["tid"], endpoint["ts"])
                == (candidate["pid"], candidate["tid"], candidate["ts"])
                for candidate in candidates
            )
    # Wide, missing, cross-process and calibration origins produce no CPU flow.
    wide = next(event for event in submits if event["args"]["correlation"] == 10)
    assert wide["args"]["external_correlation"] > 2**40
    assert "External id" not in wide["args"]
    assert "spyre_submission_links" not in trace
    names = {
        (e["pid"], e["tid"]): e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    normal = next(e for e in submits if e["args"]["correlation"] == 7)
    assert names[normal["pid"], normal["tid"]] == f"test-scheduler ({normal['tid']})"
    unnamed = next(e for e in submits if e["args"]["correlation"] == 8)
    assert names[unnamed["pid"], unnamed["tid"]] == f"Thread {unnamed['tid']}"
    orphan = next(e for e in submits if e["args"]["correlation"] == 9)
    assert names[orphan["pid"], orphan["tid"]] == f"Thread {orphan['tid']}"
    calibration = next(e for e in submits if e["args"]["correlation"] == 11)
    assert (
        names[calibration["pid"], calibration["tid"]] == "Spyre calibration (device 0)"
    )


def test_device_flows_require_both_endpoints(native_traces):
    events = _read(native_traces / "flow-pairs.json")["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X"]
    copies = [event for event in spans if event["name"] == "Memcpy (HtoD)"]
    assert {event["args"]["correlation"] for event in copies} == {13, 14, 16, 17}
    assert all(event["args"]["stream"] == 5 for event in copies)
    assert any(event["name"] == "synthetic_kernel" for event in spans)
    assert any(event["args"].get("correlation") == 18 for event in spans)
    # Only 13 has both a non-calibration launch and a device record. Orphan
    # memcpy/kernel records, calibration and submit-only IDs emit no endpoints.
    flows = _flows(events)
    assert flows.keys() == {13}
    assert sorted(event["ph"] for event in flows[13]) == ["f", "s"]
    finish = next(event for event in flows[13] if event["ph"] == "f")
    copy = next(event for event in copies if event["args"]["correlation"] == 13)
    assert (finish["pid"], finish["tid"], finish["ts"]) == (
        copy["pid"],
        copy["tid"],
        copy["ts"],
    )


def test_runtime_data_metadata(native_traces):
    events = _read(native_traces / "runtime-metadata.json")["traceEvents"]
    spans = {
        event["args"]["correlation"]: event
        for event in events
        if event.get("ph") == "X" and "correlation" in event.get("args", {})
    }
    barrier, send = spans[40], spans[41]
    assert barrier["name"] == "aiuIssueBarrier"
    assert barrier["args"]["fence_size"] == 3
    assert send["name"] == "aiuLaunchP2PRdmaSend"
    assert send["args"]["num_dest_ranks"] == 5


@pytest.mark.parametrize("prefix", ["reused", "scheduled-reused"])
def test_capture_state_does_not_leak(native_traces, prefix):
    snapshots = [_read(native_traces / f"{prefix}-{window}.json") for window in (0, 1)]
    flows = [_flows(snapshot["traceEvents"]) for snapshot in snapshots]
    assert {key for key in flows[0] if key < 2**32} == {20, 21}
    assert {key for key in flows[1] if key < 2**32} == {30, 31}
    for capture in flows:
        assert len(capture) == 3  # Two device flows and one CPU-to-submit flow.
        assert all(
            sorted(event["ph"] for event in endpoints) == ["f", "s"]
            for endpoints in capture.values()
        )
    # Submission IDs stay unique even when these captures share an export.
    assert not flows[0].keys() & flows[1].keys()
    events = snapshots[1]["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X"]
    stale = [event for event in spans if event.get("args", {}).get("correlation") == 20]
    assert {event["name"] for event in stale} == {"synthetic_kernel", "Memcpy (HtoD)"}
    assert all("External id" not in event["args"] for event in stale)
    assert any(event.get("args", {}).get("correlation") == 21 for event in spans)
    names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    submit = next(event for event in spans if event["name"] == "aiuSubmitToHardware")
    assert names[submit["pid"], submit["tid"]] == f"Thread {submit['tid']}"
    lookups = _read(native_traces / f"{prefix}-lookups.json")
    # Even an unsuccessful lookup of the old CPU origin indicates stale state.
    assert lookups[0]["origin"] not in lookups[1]["lookups"]
    assert lookups[1]["origin"] in lookups[1]["lookups"]


def _capture_native_traces(output):
    import torch
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    csrc = root / "torch_spyre/csrc"
    runtime = Path(os.environ["RUNTIME_INSTALL_DIR"])
    aiupti = Path(os.environ.get("LIBAIUPTI_INSTALL_DIR") or runtime)
    external = Path(os.environ.get("SHARED_DEPS_DIR", csrc / "external"))
    fixture = load(
        name="spyre_native_trace_fixture",
        sources=[
            str(Path(__file__).with_name("native_trace_fixture.cpp")),
            *(
                str(csrc / "profiler" / name)
                for name in (
                    "AiuptiActivityApi.cpp",
                    "AiuptiActivityProfiler.cpp",
                    "AiuptiActivityHandlers.cpp",
                )
            ),
            str(csrc / "kernel_provenance_registry.cpp"),
        ],
        extra_include_paths=[
            str(csrc),
            str(csrc / "profiler"),
            str(runtime / "include"),
            str(aiupti / "include"),
            str(external / "nlohmann/include"),
            str(Path(torch.__file__).parent / "include/kineto"),
        ],
        extra_cflags=[
            "-std=c++20",
            "-O0",
            "-DHAS_AIUPTI",
            "-DUSE_KINETO",
            "-DFMT_HEADER_ONLY",
        ],
        extra_ldflags=[
            f"-L{runtime / 'lib'}",
            "-lflex",
            f"-L{aiupti / 'lib'}",
            "-laiupti",
        ],
    )
    fixture.register_profiler()
    for filename in ("trace.json", "trace.json.gz", "repeat.json"):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            fixture.emit_batches()
        prof.export_chrome_trace(str(output / filename))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=2),
        on_trace_ready=lambda p: p.export_chrome_trace(
            str(output / f"scheduled-{p.step_num}.json")
        ),
    ) as prof:
        for _ in range(4):
            if prof.current_action == torch.profiler.ProfilerAction.RECORD_AND_SAVE:
                fixture.emit_batches()
            prof.step()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        fixture.emit_flow_pairs()
    prof.export_chrome_trace(str(output / "flow-pairs.json"))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        fixture.emit_runtime_metadata()
    prof.export_chrome_trace(str(output / "runtime-metadata.json"))

    lookups = []
    for window in range(2):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            origin = fixture.emit_capture_window(window)
        prof.export_chrome_trace(str(output / f"reused-{window}.json"))
        lookups.append({"origin": origin, "lookups": fixture.cpu_lookups()})
    (output / "reused-lookups.json").write_text(json.dumps(lookups))

    lookups = []

    def save_window(prof):
        prof.export_chrome_trace(str(output / f"scheduled-reused-{len(lookups)}.json"))
        lookups.append({"origin": origin, "lookups": fixture.cpu_lookups()})

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=2),
        on_trace_ready=save_window,
    ) as prof:
        for _ in range(4):
            if prof.current_action == torch.profiler.ProfilerAction.RECORD_AND_SAVE:
                origin = fixture.emit_capture_window(len(lookups))
            prof.step()
    (output / "scheduled-reused-lookups.json").write_text(json.dumps(lookups))


if __name__ == "__main__":
    _capture_native_traces(Path(sys.argv[1]))
