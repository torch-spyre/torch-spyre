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

"""Tests for launching simple compiled ops through JobPlan execution."""

import os

import pytest
import torch


class TestLaunchJobPlan:
    """Test suite for JobPlan-backed compiled op execution."""

    def test_dump_spyre_code_abs_matches_cpu(self):
        """Run a simple compiled op with `DUMP_SPYRE_CODE=1` and compare to CPU."""
        x = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.abs(x)

        previous = os.environ.get("DUMP_SPYRE_CODE")
        os.environ["DUMP_SPYRE_CODE"] = "1"
        try:
            # Compile and run on Spyre
            compiled_fn = torch.compile(torch.abs, backend="inductor")
            spyre_result = compiled_fn(x.to("spyre")).cpu()

            # Compare results
            torch.testing.assert_close(
                spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
            )
        finally:
            if previous is None:
                del os.environ["DUMP_SPYRE_CODE"]
            else:
                os.environ["DUMP_SPYRE_CODE"] = previous

    def test_dump_spyre_code_mul_matches_cpu(self):
        """Run a simple compiled binary op with `DUMP_SPYRE_CODE=1` and compare to CPU."""
        x = torch.randn(64, dtype=torch.float16)
        y = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.mul(x, y)

        previous = os.environ.get("DUMP_SPYRE_CODE")
        os.environ["DUMP_SPYRE_CODE"] = "1"
        try:
            # Compile and run on Spyre
            compiled_fn = torch.compile(torch.mul, backend="inductor")
            spyre_result = compiled_fn(x.to("spyre"), y.to("spyre")).cpu()

            # Compare results
            torch.testing.assert_close(
                spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
            )
        finally:
            if previous is None:
                del os.environ["DUMP_SPYRE_CODE"]
            else:
                os.environ["DUMP_SPYRE_CODE"] = previous


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
