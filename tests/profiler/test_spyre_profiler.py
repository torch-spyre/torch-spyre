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

import json
import pytest
import unittest
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, _memory_profiler
from torch.testing._internal.common_utils import (
    skipIfTorchDynamo,
    TemporaryFileName,
    TestCase,
)


Test_spyre = None
if hasattr(torch, "spyre"):
    Test_spyre = torch.spyre.is_available()
else:
    Test_spyre = False


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
    from torch.profiler import profile, ProfilerActivity

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

    result = z.cpu()

    assert result.numel() == 64 * 64
    assert torch.isfinite(result).all()


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
