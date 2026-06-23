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
from typing import Tuple

import pytest
from torch.testing._internal.common_utils import TestCase
import torch
import torch._dynamo

from torch_spyre._inductor import config as _spyre_config


def _run_compiled_op(op_name: str, symbolic_args: bool) -> None:
    """
    Compile an op with DUMP_SPYRE_CODE=1 and run it on Spyre, comparing to CPU.

    Uses a fresh dynamo compile cache each call to ensure the kernel runner is
    re-instantiated with the current DUMP_SPYRE_CODE env var value.  Runs
    in-process (no subprocess) so the Spyre VFIO device opened by the test
    session is reused rather than triggering a second exclusive open from a
    child process.
    """
    torch._dynamo.reset()

    op_fn = getattr(torch, op_name)

    torch.manual_seed(42)
    inputs: Tuple[torch.Tensor, ...]
    if op_name == "abs":
        inputs = (torch.randn(64, dtype=torch.float16),)
    elif op_name == "mul":
        inputs = (
            torch.randn(64, dtype=torch.float16),
            torch.randn(64, dtype=torch.float16),
        )
    else:
        raise ValueError(f"Unknown op: {op_name}")

    cpu_result = op_fn(*inputs)

    old_val = os.environ.get("DUMP_SPYRE_CODE")
    try:
        os.environ["DUMP_SPYRE_CODE"] = "1"
        with _spyre_config.patch(bundle_symbolic_args=symbolic_args):  # type: ignore[attr-defined]
            compiled_fn = torch.compile(op_fn, backend="inductor")
            spyre_inputs = tuple(inp.to("spyre") for inp in inputs)
            spyre_result = compiled_fn(*spyre_inputs).cpu()
    finally:
        if old_val is None:
            os.environ.pop("DUMP_SPYRE_CODE", None)
        else:
            os.environ["DUMP_SPYRE_CODE"] = old_val

    torch.testing.assert_close(
        spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
    )


class TestLaunchJobPlan(TestCase):
    """Test suite for JobPlan-backed compiled op execution."""

    def test_abs_matches_cpu_no_symbols(self):
        """Run compiled abs op without symbolic args and compare to CPU."""
        _run_compiled_op("abs", symbolic_args=False)

    def test_abs_matches_cpu_with_symbols(self):
        """Run compiled abs op with symbolic args and compare to CPU."""
        _run_compiled_op("abs", symbolic_args=True)

    def test_mul_matches_cpu_no_symbols(self):
        """Run compiled mul op without symbolic args and compare to CPU."""
        _run_compiled_op("mul", symbolic_args=False)

    def test_mul_matches_cpu_with_symbols(self):
        """Run compiled mul op with symbolic args and compare to CPU."""
        _run_compiled_op("mul", symbolic_args=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
