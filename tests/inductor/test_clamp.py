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

import pytest
import torch


@pytest.mark.parametrize(
    "operation", ["clamp_min", "clamp_max", "lower_only", "upper_only", "both"]
)
def test_clamp_preserves_unspecified_bound(operation):
    """SEN169 intermediates can exceed IEEE fp16's finite range."""

    def fn(x):
        large = x * 4.0
        if operation == "clamp_min":
            bounded = large.clamp_min(1.0)
        elif operation == "clamp_max":
            bounded = large.clamp_max(-1.0)
        elif operation == "lower_only":
            bounded = large.clamp(min=1.0)
        elif operation == "upper_only":
            bounded = large.clamp(max=-1.0)
        else:
            bounded = large.clamp(min=-1.0, max=1.0)
        return bounded * 0.25

    values = torch.tensor([-32768, -8192, -1, 0, 1, 8192, 32768], dtype=torch.float16)
    values = values[:, None].expand(-1, 64).contiguous()
    # Compute the reference in fp32 so the enlarged intermediate stays finite.
    expected = fn(values.float()).half()
    with torch.inference_mode():
        actual = torch.compile(fn, fullgraph=True)(values.to("spyre")).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0.002)
