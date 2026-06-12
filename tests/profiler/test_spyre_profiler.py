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
from torch.profiler import profile, ProfilerActivity
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
        device = "spyre"
        x = torch.randn(4, device=device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            with_stack=True,
        ) as prof:
            x *= 2

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

    x = torch.randn(64, 64, device="spyre")
    y = torch.randn(64, 64, device="spyre")

    z = torch.matmul(x, y)

    torch.spyre.synchronize()

    result = z.cpu()

    assert result.numel() == 64 * 64
    assert torch.isfinite(result).all()
