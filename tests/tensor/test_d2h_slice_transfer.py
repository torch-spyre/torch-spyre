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

"""D2H views must transfer their covered sticks, including offset and padding."""

import json

import pytest
import torch

import torch_spyre  # noqa: F401
from torch_spyre._C import DataFormats, SpyreTensorLayout


@pytest.fixture(scope="module", autouse=True)
def initialize_spyre_runtime():
    # The explicit-layout allocator is called below the dispatch path that
    # normally creates the device runtime on the first tensor allocation.
    torch.empty(1, device="spyre")


@pytest.mark.parametrize(
    "dtype,device_format",
    [
        (torch.float16, DataFormats.SEN169_FP16),
        (torch.bfloat16, DataFormats.SEN169_FP16),
        (torch.int64, DataFormats.IEEE_INT32),
        (torch.float32, DataFormats.IEEE_FP32),
    ],
)
@pytest.mark.parametrize("width", [128, 130])
@pytest.mark.parametrize("row", [0, 3, 7])
def test_row_view_transfers_only_covered_sticks(
    dtype, device_format, width, row, tmp_path
):
    rows = 8
    host = ((torch.arange(rows * width) % 64) - 32).to(dtype).reshape(1, rows, width)
    eps = device_format.elems_per_stick()
    sticks = (width + eps - 1) // eps
    # The row-major physical layout produced by the compiled LM head.
    layout = SpyreTensorLayout(
        [rows, sticks, 1, eps], [width, eps, -1, 1], device_format
    )
    device = host.to("spyre", device_layout=layout)
    view = device[:, row, :]
    torch.spyre.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        actual = view.cpu()
    torch.testing.assert_close(actual, host[:, row, :], rtol=0, atol=0)
    path = tmp_path / "trace.json"
    prof.export_chrome_trace(str(path))
    copies = [
        e
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("cat") == "gpu_memcpy" and "DtoH" in e["name"]
    ]
    assert copies, "The profiler must record the actual device transfer"
    assert sum(e["args"]["bytes"] for e in copies) == sticks * 128


@pytest.mark.parametrize(
    "dtype,device_format",
    [
        (torch.float16, DataFormats.SEN169_FP16),
        (torch.float32, DataFormats.IEEE_FP32),
    ],
)
def test_view_starting_inside_a_stick(dtype, device_format):
    rows, width = 8, 130
    host = ((torch.arange(rows * width) % 64) - 32).to(dtype).reshape(rows, width)
    eps = device_format.elems_per_stick()
    layout = SpyreTensorLayout(
        [rows, (width + eps - 1) // eps, eps], [width, eps, 1], device_format
    )
    device = host.to("spyre", device_layout=layout)
    # The source offset must retain its position within the first DMA stick.
    actual = device[3, 5:63].cpu()
    torch.testing.assert_close(actual, host[3, 5:63], rtol=0, atol=0)


def test_strided_batch_slice_and_destination_dtype():
    host = ((torch.arange(2 * 8 * 128) % 64) - 32).to(torch.float16).reshape(2, 8, 128)
    layout = SpyreTensorLayout(
        [2, 8, 2, 1, 64], [1024, 128, 64, -1, 1], DataFormats.SEN169_FP16
    )
    device = host.to("spyre", device_layout=layout)
    # There is a gap between the selected rows. The bounded DMA must preserve
    # the original source strides when rebasing, and use the device item size.
    actual = device[:, 3, :].to(device="cpu", dtype=torch.float32)
    torch.testing.assert_close(actual, host[:, 3, :].float(), rtol=0, atol=0)
