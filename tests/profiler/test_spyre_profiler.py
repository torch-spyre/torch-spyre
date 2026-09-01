# Copyright 2025 The Torch-Spyre Authors.
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

import inspect
import json
import math
import unittest

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, _memory_profiler, profile
from torch.testing._internal.common_utils import (
    TemporaryFileName,
    TestCase,
    skipIfTorchDynamo,
)

Test_spyre = None
if hasattr(torch, "spyre"):
    Test_spyre = torch.spyre.is_available()
else:
    Test_spyre = False


class _ProfilerMLP(torch.nn.Module):
    """Small stick-aligned model for compiled device-event provenance."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestSpyreProfiler(TestCase):
    @unittest.skipUnless(Test_spyre, "requires spyre device")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_basic_profile(self):
        # ---------------------------------------------------------------------
        # TEMPORARY WORKAROUND (2026-08) — do NOT read prof.events() here yet.
        #
        # Background — libaiupti PR #114 (ABI/stride mismatch):
        #   #114 appended 5x uint64 `cycles_ts1..5` (+40 bytes) to
        #   AIUpti_ActivityCompute (and _ActivityMemcpy) *after* the `name[128]`
        #   field. `name`'s own offset didn't move — but the record WALKER
        #   advances by sizeof(AIUpti_ActivityCompute) (libaiupti
        #   aiupti_api.cpp::aiuptiActivityGetNextRecord). If libaiupti and
        #   kineto-spyre are not rebuilt in lockstep they disagree on that size,
        #   so after the first record every subsequent record is read at the
        #   wrong offset and the kernel `name` lands on garbage bytes ->
        #   UnicodeDecodeError when prof.events() -> _parse_kineto_results ->
        #   evt.name() decodes it. Real fix = rebuild libaiupti + kineto-spyre
        #   together. Tracked in #114.
        #
        # Why the body is stubbed (no `as prof`, assertTrue(True)):
        #   Reading prof.events() is what triggers the buffer walk and the crash.
        #   We still run capture + teardown (the `with profile(...)` block) but
        #   never decode the corrupt kernel name. Restore the real check (below)
        #   once the two libs are rebuilt in lockstep.
        #
        # WHY STUBBING THIS ALSO "FIXED" test_event_list /
        # test_profiler_timestamp_consistency (the surprising part):
        #   All three tests run in ONE shared process. The libaiupti record
        #   walker uses a *process-global* `static std::unordered_map
        #   current_buffer_map` (aiupti_api.cpp) for per-buffer read offsets,
        #   erase()'d only when a walk finishes cleanly. When the OLD
        #   test_basic_profile called prof.events() FIRST, its walk aborted
        #   mid-buffer (garbage `kind` -> AIUPTI_ERROR, or the Python decode threw
        #   mid-iteration) BEFORE that cleanup, leaving stale global profiler
        #   state (dirty offset map / undrained ready-buffer deque) that the next
        #   test inherited -> the stall/corruption seen in the later tests. So
        #   test_basic_profile was the TRIGGER, not just a victim: not walking
        #   the buffer here leaves the shared state clean and the later tests
        #   pass. This is cross-test coupling through mutable C++ globals, NOT a
        #   real fix — the #114 mismatch is still present. If the later tests
        #   start stalling/failing again, suspect that shared state first.
        #   (Hypothesis from code reading; not verified on hardware.)
        # ---------------------------------------------------------------------
        device = "spyre"
        x = torch.randn(4, device=device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            with_stack=False,
        ) as prof:
            x *= 2
            # TODO(#114): check with_stack=True once libaiupti + kineto-spyre are rebuilt in lockstep.
        names = [e.name for e in prof.events()]
        self.assertTrue("aten::mul_" in names)

    @unittest.skipUnless(Test_spyre, "require spyre device")
    def test_event_list(self):
        device = torch.device("spyre")
        x, y = (torch.rand((4, 4), dtype=torch.float16).to(device) for _ in range(2))

        with profile(with_stack=True) as prof:
            z = torch.add(x, y)
            z = F.gelu(z)
            z = torch.sum(z)

        event_list = torch.autograd.profiler_util.EventList(prof.events())

        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            with open(fname) as f:
                json.load(f)

        event_list.table()

    @unittest.skipIf(not Test_spyre, "spyre device required")
    def test_profiler_timestamp_consistency(self):
        """Verify that FunctionEvent timestamps can reconstruct Chrome trace ts values."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            x = torch.randn(32, 32, device="spyre")
            torch.add(x, x)

        trace_start_ns = prof.profiler.kineto_results.trace_start_ns()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            base_time_ns = j.get("baseTimeNanoseconds", 0)

            fe_mm = next((e for e in prof.events() if e.name == "aten::add"), None)
            json_mm = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e["name"] == "aten::add" and e["ph"] == "X"
                ),
                None,
            )

            absolute_ns = int(fe_mm.time_range.start * 1000) + trace_start_ns
            recovered_ts = (absolute_ns - base_time_ns) / 1000
            self.assertEqual(
                recovered_ts,
                json_mm["ts"],
                msg="Recovered Chrome trace ts doesn't match Json for aten::add",
            )


def test_package_importable():
    """
    Verify that the torch_spyre.profiler package can be imported
    without requiring Spyre hardware.
    """
    import torch_spyre.profiler  # noqa: F401


def test_chrome_trace_is_valid_json(tmp_path):
    """
    Verify that export_chrome_trace() produces valid JSON with at least one event.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    trace_file = tmp_path / "spyre_trace.json"

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        x = torch.randn(10, 10)
        _ = torch.matmul(x, x)

    prof.export_chrome_trace(str(trace_file))

    # Ensure the file exists and contains valid JSON
    assert trace_file.exists(), "Chrome trace file was not created"

    with open(trace_file, "r") as f:
        data = json.load(f)

    # Chrome traces typically contain a "traceEvents" list
    assert isinstance(data, dict), "Trace JSON must be a dictionary"
    assert "traceEvents" in data, "Trace JSON must contain 'traceEvents'"
    assert len(data["traceEvents"]) > 0, "Trace JSON must contain at least one event"


def _run_trace_analyzer_overlap_verification(trace_data):
    """Run Trace Analyzer overlap verification and return its result and report."""
    pytest.importorskip("aiu_trace_analyzer", minversion="1.3.0")
    from aiu_trace_analyzer.core.acelyzer import Acelyzer

    analyzer = Acelyzer(
        ["-i", "api://jsonbuffer", "-V", "--disable_file"],
        in_data=trace_data,
    )
    analyzer.run()
    report = analyzer.get_output_data()

    overlap_result = next(
        (
            result
            for result in report.get("test_results", [])
            if result.get("test") == "Compute Overlap Check"
        ),
        None,
    )

    assert overlap_result is not None, (
        "AIU Trace Analyzer did not return a Compute Overlap Check result"
    )

    return overlap_result, report


@pytest.mark.requires_spyre_profiler
def test_synchronize_callable():
    """
    Ensure that torch.spyre.synchronize() is callable without error.
    This test requires Spyre hardware and USE_SPYRE_PROFILER=1.
    """
    import torch

    # Verify the attribute exists
    assert hasattr(torch, "spyre"), "torch.spyre namespace is missing"
    assert hasattr(torch.spyre, "synchronize"), "torch.spyre.synchronize() is missing"

    x = torch.randn((64, 64), dtype=torch.float16, device="spyre")
    y = torch.randn((64, 64), dtype=torch.float16, device="spyre")

    z = torch.matmul(x, y)

    torch.spyre.synchronize()

    # .cpu() performs an implicit synchronization, so this test does not
    # independently verify synchronize(). It serves as an end-to-end correctness
    # and API smoke test.
    result = z.cpu()

    assert result.numel() == 64 * 64
    assert torch.isfinite(result).all()

    torch.testing.assert_close(result, atol=1e-1, rtol=1e-1)


@pytest.mark.requires_spyre_profiler
def test_compiled_kernel_event_keys_match_captured_debug_handles(monkeypatch):
    """Real events carry compiler keys and direct handles from the same process."""
    from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
    from torch_spyre._inductor.profiler_event import (
        AIUPTI_ACTIVITY_NAME_MAX_BYTES,
        extract_kernel_provenance_key,
    )
    from torch_spyre.execution.async_compile import SpyreAsyncCompile

    captures = []
    original_sdsc = SpyreAsyncCompile.sdsc

    def capture_sdsc(self, kernel_name, specs, pool_size=0):
        runner = original_sdsc(self, kernel_name, specs, pool_size=pool_size)
        handles = []

        def collect(spec_list):
            for spec in spec_list:
                if isinstance(spec, OpSpec) and spec.debug_handle is not None:
                    handles.append(spec.debug_handle)
                elif isinstance(spec, LoopSpec):
                    collect(spec.body)

        collect(specs)
        if runner.kernel_provenance is not None:
            captures.append(
                (runner.kernel_provenance, runner.profiler_event_name, tuple(handles))
            )
        return runner

    monkeypatch.setattr(SpyreAsyncCompile, "sdsc", capture_sdsc)
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch._dynamo.reset()

    model = _ProfilerMLP().half().to("spyre").eval()
    x = torch.randn(2, 128, dtype=torch.float16, device="spyre")
    compiled = torch.compile(model, fullgraph=True)

    with torch.no_grad():
        compiled(x)
        torch.spyre.synchronize()

        assert captures, "compilation produced no provenance-aware Spyre runners"

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            result = compiled(x)
            torch.spyre.synchronize()

    assert result.shape == (2, 128)

    with TemporaryFileName(mode="w+") as fname:
        prof.export_chrome_trace(fname)
        with open(fname) as f:
            trace = json.load(f)

    events = trace["traceEvents"]
    for descriptor, event_name, handles in captures:
        assert event_name is not None
        expected_ids = tuple(dict.fromkeys(str(handle.id) for handle in handles))
        assert descriptor.debug_handle_ids == expected_ids

        matching_events = [
            event
            for event in events
            if extract_kernel_provenance_key(event.get("name", "")) == descriptor.key
        ]
        assert matching_events, (
            f"no device event contained kernel provenance key {descriptor.key}"
        )
        assert all(
            event.get("cat") == "kernel" and event.get("ph") == "X"
            for event in matching_events
        )
        assert all(
            event["name"].startswith(f"{event_name}#")
            and event["name"].rsplit("#", 1)[1].isdecimal()
            for event in matching_events
        )
        assert all(
            len(event["name"].encode("ascii")) <= AIUPTI_ACTIVITY_NAME_MAX_BYTES
            for event in matching_events
        )
        for event in matching_events:
            args = event.get("args", {})
            assert args.get("provenance_key") == descriptor.key
            debug_handles = args.get("debug_handles")
            assert isinstance(debug_handles, list), (
                "args.debug_handles must be a JSON array, not a quoted string"
            )
            assert all(isinstance(handle_id, str) for handle_id in debug_handles)
            assert debug_handles == list(descriptor.debug_handle_ids)

    def lineage(handle):
        yield handle
        for constituent in handle.fused_from:
            yield from lineage(constituent)

    source_line = inspect.getsourcelines(_ProfilerMLP.forward)[1] + 1
    source_handles = [
        candidate
        for _, _, handles in captures
        for handle in handles
        for candidate in lineage(handle)
        if candidate.source is not None
    ]
    captured_lineage = [
        (
            handle.source.file,
            handle.source.start_line,
            handle.aten_op,
        )
        for handle in source_handles
    ]
    assert any(
        handle.source.file.endswith("test_spyre_profiler.py")
        and handle.source.start_line == source_line
        and handle.aten_op == "aten.linear.default"
        for handle in source_handles
    ), (
        "the captured provenance did not contain the model's linear source line; "
        f"captured lineage: {captured_lineage}"
    )
    assert any(
        len(handle.fused_from) >= 2 for _, _, handles in captures for handle in handles
    ), "the compiled kernel did not retain its fused provenance constituents"


@pytest.mark.requires_spyre_profiler
def test_kineto_memcpy_and_memset_events_captured():
    """
    Confirm that H2D memcpy, D2H memcpy, and memset events are captured
    in the kineto-spyre Chrome trace when profiling with PrivateUse1.

    Triggered operations:
      - H2D: cpu_tensor.to("spyre")
      - memset: torch.zeros(..., device="spyre")
      - D2H: device_tensor.cpu()

    Note: P2P (device-to-device) transfers are out of scope for this test.
    """
    cpu_src = torch.randn(64, 64, dtype=torch.float16)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        device_tensor = cpu_src.to("spyre")
        _ = torch.zeros(64, 64, dtype=torch.float16, device="spyre")
        _ = device_tensor.cpu()
        torch.spyre.synchronize()

    with TemporaryFileName(mode="w+") as fname:
        prof.export_chrome_trace(fname)
        with open(fname) as f:
            trace = json.load(f)

    assert "traceEvents" in trace, (
        "Chrome trace is missing 'traceEvents' key — export may have failed"
    )
    events = trace["traceEvents"]

    # "gpu_memcpy" / "gpu_memset" are emitted by libkineto's ActivityType::type_string()
    # (upstream kineto, not kineto-spyre-specific) and have been stable across all kineto
    # versions used by torch-spyre. kineto-spyre maps Spyre memory activities to these
    # standard ActivityType values; the Chrome trace writer produces these category strings.
    h2d_events = [
        e
        for e in events
        if e.get("cat") == "gpu_memcpy" and "HtoD" in e.get("name", "")
    ]
    assert h2d_events, (
        "Expected at least one H2D memcpy event in the kineto-spyre trace"
    )

    d2h_events = [
        e
        for e in events
        if e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "")
    ]
    assert d2h_events, (
        "Expected at least one D2H memcpy event in the kineto-spyre trace"
    )

    memset_events = [e for e in events if e.get("cat") == "gpu_memset"]
    assert memset_events, "Expected at least one memset event in the kineto-spyre trace"


@pytest.mark.requires_spyre_profiler
def test_trace_analyzer_device_overlap(tmp_path):
    """Verify Spyre device overlaps using AIU Trace Analyzer."""
    trace_file = tmp_path / "trace_analyzer_overlap.json"

    x = torch.randn((64, 64), dtype=torch.float16, device="spyre")
    y = torch.randn((64, 64), dtype=torch.float16, device="spyre")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        result = torch.matmul(x, y)
        result = F.gelu(result)
        result = torch.sum(result)
        torch.spyre.synchronize()

    prof.export_chrome_trace(str(trace_file))

    assert trace_file.exists(), "Chrome trace file was not created"

    trace_data = trace_file.read_bytes()

    trace_json = json.loads(trace_data)
    trace_events = trace_json.get("traceEvents", [])
    device_events, _ = _find_device_overlaps(trace_events)

    assert len(device_events) >= 2, (
        "Expected at least two Spyre device events for Trace Analyzer overlap validation"
    )

    overlap_result, report = _run_trace_analyzer_overlap_verification(trace_data)

    if overlap_result.get("result") != "pass":
        overlap_errors = [
            error
            for error in report.get("errors", [])
            if error.get("finding") == "overlaps"
        ]

        pytest.fail(
            "AIU Trace Analyzer detected invalid Spyre device overlap(s):\n"
            f"{overlap_errors}"
        )


class TestMemoryProfilerTimeline(TestCase):
    @unittest.skipIf(not Test_spyre, "spyre device required")
    def test_memory_timeline_no_id_spyre(self) -> None:
        # On CPU the default behavior is to simply forward to malloc. That
        # means that when we free `x` the allocator doesn't actually know how
        # many bytes are in the allocation, and thus there's no point to
        # calling `c10::reportMemoryUsageToProfiler`. So in order to test that
        # memory profiler processes this case correctly we need to use device
        # where we do always keep a record.
        x = torch.ones((1024,), device="spyre")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            # We never see `x` used so we don't know the storage is for a
            # Tensor, but we do still see the free event.
            del x

            # For empty we see the allocation and free, but not any use.
            # So this also cannot be identified as a Tensor.
            y = torch.empty((64,))
            del y

            z = torch.empty((256,))
            z.view_as(z)  # Show `z` to the profiler
            del z

        memory_profile = prof._memory_profile()

        expected = [
            # x
            (_memory_profiler.Action.PREEXISTING, 4096),
            (_memory_profiler.Action.DESTROY, 4096),
            #
            # y
            (_memory_profiler.Action.CREATE, 256),
            (_memory_profiler.Action.DESTROY, 256),
            #
            # z
            (_memory_profiler.Action.CREATE, 1024),
            (_memory_profiler.Action.DESTROY, 1024),
        ]

        actual = [(action, size) for _, action, _, size in memory_profile.timeline]

        self.assertGreaterEqual(len(actual), len(expected))

        for (act_action, act_size), (exp_action, exp_size) in zip(actual, expected):
            self.assertEqual(act_action, exp_action)
            self.assertGreaterEqual(
                act_size, exp_size, f"Expected at least {exp_size}, got {act_size}"
            )
            # Allow generous allocator padding/alignment overhead. 4x is chosen as a
            # middle ground: 2x risks false failures from allocator rounding, while
            # 8x would allow large over-reporting bugs to pass unnoticed.
            self.assertLessEqual(
                act_size,
                exp_size * 4,
                f"Expected at most {exp_size * 4}, got {act_size}",
            )

    def test_memory_timeline_no_id_cpu(self) -> None:
        x = torch.ones((1024,), device="cpu")

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            # We never see `x` used so we don't know the storage is for a
            # Tensor, but we do still see the free event.
            del x

            # For empty we see the allocation and free, but not any use.
            # So this also cannot be identified as a Tensor.
            y = torch.empty((64,))
            del y

            z = torch.empty((256,))
            z.view_as(z)  # Show `z` to the profiler
            del z

        memory_profile = prof._memory_profile()

        expected = [
            #
            # y
            (_memory_profiler.Action.CREATE, 256),
            (_memory_profiler.Action.DESTROY, 256),
            #
            # z
            (_memory_profiler.Action.CREATE, 1024),
            (_memory_profiler.Action.DESTROY, 1024),
        ]

        actual = [(action, size) for _, action, _, size in memory_profile.timeline]

        for event in expected:
            self.assertTrue(event in actual, f"event: {event} was not found in actual.")


def _find_device_overlaps(events):
    """Return valid positive-duration Spyre device events and overlapping event pairs."""
    device_events = []

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X" or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue

        timestamp = event.get("ts")
        duration = event.get("dur")
        name = event.get("name", "unknown")

        assert (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(timestamp)
        ), (
            f"Spyre device event {name} must have a finite numeric timestamp "
            f"(ts={timestamp}, dur={duration})"
        )

        assert (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(duration)
            and duration > 0
        ), (
            f"Spyre device event {name} must have a finite positive duration "
            f"(ts={timestamp}, dur={duration})"
        )

        device_events.append(event)

    sorted_events = sorted(device_events, key=lambda event: event["ts"])
    overlaps = []

    # Check overlaps on the global Spyre device timeline for this validation.
    # Stream-aware overlap checking will be handled in a follow-up using Trace
    # Analyzer once stream IDs are available consistently for kernel and memory events.
    # Kernel-memory overlaps are currently treated as invalid and are flagged.
    # This assumption may need to be revisited as Spyre event ordering semantics
    # and profiler marker placement are clarified.
    # AIU kernel events are currently expected to represent independent execution
    # intervals rather than nested parent-child function calls. Therefore, fully
    # nested kernel spans are intentionally treated as overlaps. If nested
    # parent-child kernel execution becomes valid in the future, this validation
    # should be revisited.
    for first_index, first_event in enumerate(sorted_events):
        first_end_time = first_event["ts"] + first_event["dur"]

        for second_event in sorted_events[first_index + 1 :]:
            if second_event["ts"] >= first_end_time:
                break

            if (
                first_event.get("cat") != "kernel"
                and second_event.get("cat") != "kernel"
            ):
                continue

            overlaps.append((first_event, second_event))

    return sorted_events, overlaps


def test_find_device_overlaps():
    """Verify overlap detection with simple synthetic Spyre device events."""
    clean_events = [
        {"ph": "X", "cat": "kernel", "name": "kernel_1", "ts": 0, "dur": 10},
        {"ph": "X", "cat": "kernel", "name": "kernel_2", "ts": 10, "dur": 10},
    ]

    overlap_events = [
        {"ph": "X", "cat": "kernel", "name": "kernel_1", "ts": 0, "dur": 10},
        {"ph": "X", "cat": "kernel", "name": "kernel_2", "ts": 5, "dur": 10},
    ]

    kernel_memory_overlap_events = [
        {"ph": "X", "cat": "kernel", "name": "kernel_1", "ts": 0, "dur": 10},
        {
            "ph": "X",
            "cat": "gpu_memcpy",
            "name": "Memcpy (HtoD)",
            "ts": 5,
            "dur": 10,
        },
        {
            "ph": "X",
            "cat": "gpu_memset",
            "name": "Memset (Device)",
            "ts": 20,
            "dur": 5,
        },
    ]

    memory_overlap_events = [
        {
            "ph": "X",
            "cat": "gpu_memcpy",
            "name": "Memcpy (HtoD)",
            "ts": 0,
            "dur": 10,
        },
        {
            "ph": "X",
            "cat": "gpu_memset",
            "name": "Memset (Device)",
            "ts": 5,
            "dur": 10,
        },
    ]

    clean_device_events, clean_overlaps = _find_device_overlaps(clean_events)
    overlap_device_events, overlaps = _find_device_overlaps(overlap_events)
    kernel_memory_device_events, kernel_memory_overlaps = _find_device_overlaps(
        kernel_memory_overlap_events
    )
    memory_device_events, memory_overlaps = _find_device_overlaps(memory_overlap_events)

    assert len(clean_device_events) == 2
    assert clean_overlaps == []

    assert len(overlap_device_events) == 2
    assert len(overlaps) == 1

    assert len(kernel_memory_device_events) == 3
    assert len(kernel_memory_overlaps) == 1

    assert len(memory_device_events) == 2
    assert memory_overlaps == []

    kernel_event, memory_event = kernel_memory_overlaps[0]
    assert kernel_event["cat"] == "kernel"
    assert memory_event["cat"] == "gpu_memcpy"

    first_event, second_event = overlaps[0]
    assert first_event["name"] == "kernel_1"
    assert second_event["name"] == "kernel_2"

    first_start_time = first_event["ts"]
    second_start_time = second_event["ts"]
    first_end_time = first_start_time + first_event["dur"]
    second_end_time = second_start_time + second_event["dur"]

    overlap_start = max(first_start_time, second_start_time)
    overlap_end = min(first_end_time, second_end_time)
    overlap_time = overlap_end - overlap_start

    # Overlap duration is expressed in trace time units.
    assert overlap_time == 5


def test_find_device_overlaps_invalid_events():
    """Verify invalid Spyre device interval data is rejected."""
    zero_duration_event = [
        {"ph": "X", "cat": "kernel", "name": "kernel_zero", "ts": 0, "dur": 0}
    ]

    missing_timestamp_event = [
        {"ph": "X", "cat": "kernel", "name": "kernel_missing_ts", "dur": 10}
    ]

    missing_duration_event = [
        {"ph": "X", "cat": "kernel", "name": "kernel_missing_dur", "ts": 0}
    ]

    with pytest.raises(AssertionError):
        _find_device_overlaps(zero_duration_event)

    with pytest.raises(AssertionError):
        _find_device_overlaps(missing_timestamp_event)

    with pytest.raises(AssertionError):
        _find_device_overlaps(missing_duration_event)


@pytest.mark.requires_spyre_profiler
def test_kernel_time_overlap(tmp_path):
    """Reject non-finite timestamps or non-positive durations, then verify no overlaps."""
    trace_file = tmp_path / "kernel_overlap_trace.json"

    x = torch.randn((64, 64), dtype=torch.float16, device="spyre")
    y = torch.randn((64, 64), dtype=torch.float16, device="spyre")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        result = torch.matmul(x, y)
        result = F.gelu(result)
        result = torch.sum(result)
        torch.spyre.synchronize()

    prof.export_chrome_trace(str(trace_file))

    assert trace_file.exists(), "Chrome trace file was not created"

    with trace_file.open("r", encoding="utf-8") as trace:
        trace_data = json.load(trace)

    assert isinstance(trace_data, dict), "Trace JSON must be a dictionary"
    assert "traceEvents" in trace_data, "Chrome trace is missing the 'traceEvents' key"

    trace_events = trace_data["traceEvents"]
    assert isinstance(trace_events, list), "'traceEvents' must contain a list"

    device_events, overlaps = _find_device_overlaps(trace_events)

    assert len(device_events) >= 2, (
        "Expected at least two Spyre device events for overlap validation"
    )

    if overlaps:
        overlap_details = []

        for first_event, second_event in overlaps[:10]:
            first_end_time = first_event["ts"] + first_event["dur"]
            second_end_time = second_event["ts"] + second_event["dur"]

            overlap_start = max(first_event["ts"], second_event["ts"])
            overlap_end = min(first_end_time, second_end_time)
            overlap_time = overlap_end - overlap_start

            overlap_details.append(
                f"{first_event.get('name', 'unknown')} "
                f"(ts={first_event['ts']}, dur={first_event['dur']}, end={first_end_time}) overlaps "
                f"{second_event.get('name', 'unknown')} "
                f"(ts={second_event['ts']}, dur={second_event['dur']}, end={second_end_time}) by "
                f"{overlap_time:.3f} trace time units"
            )

        pytest.fail(
            f"{len(overlaps)} Spyre device overlap(s) detected:\n"
            + "\n".join(overlap_details)
        )
