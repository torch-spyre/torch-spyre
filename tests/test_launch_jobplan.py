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
from contextlib import contextmanager

import pytest
import torch


@contextmanager
def set_env_vars(**env_vars):
    """Context manager to temporarily set environment variables."""
    previous = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            os.environ[key] = value
        yield
    finally:
        for key, prev_value in previous.items():
            if prev_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev_value


class TestLaunchJobPlan:
    """Test suite for JobPlan-backed compiled op execution."""

    @pytest.mark.parametrize(
        "env_vars",
        [
            {"DUMP_SPYRE_CODE": "1"},
            {"DUMP_SPYRE_CODE": "1", "BUNDLE_HBM_SYMBOLS": "1"},
        ],
        ids=["SpyreCode, no symbols", "SpyreCode, fake symbols"],
    )
    def test_abs_matches_cpu(self, env_vars):
        """Run compiled abs op with various env settings and compare to CPU."""
        x = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.abs(x)

        with set_env_vars(**env_vars):
            compiled_fn = torch.compile(torch.abs, backend="inductor")
            spyre_result = compiled_fn(x.to("spyre")).cpu()

            torch.testing.assert_close(
                spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
            )

    @pytest.mark.parametrize(
        "env_vars",
        [
            {"DUMP_SPYRE_CODE": "1"},
            {"DUMP_SPYRE_CODE": "1", "BUNDLE_HBM_SYMBOLS": "1"},
        ],
        ids=["SpyreCode, no symbols", "SpyreCode, fake symbols"],
    )
    def test_mul_matches_cpu(self, env_vars):
        """Run compiled mul op with various env settings and compare to CPU."""
        x = torch.randn(64, dtype=torch.float16)
        y = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.mul(x, y)

        with set_env_vars(**env_vars):
            compiled_fn = torch.compile(torch.mul, backend="inductor")
            spyre_result = compiled_fn(x.to("spyre"), y.to("spyre")).cpu()

            torch.testing.assert_close(
                spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
