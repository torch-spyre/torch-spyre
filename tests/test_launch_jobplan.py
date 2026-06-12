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
import subprocess
import sys

import pytest


def run_compiled_op_in_subprocess(op_name: str, env_vars: dict) -> bool:
    """
    Run a compiled op test in an isolated subprocess with specific env vars.

    This ensures each test runs with a fresh torch.compile cache and config,
    avoiding the need for torch._dynamo.reset_code_caches().
    """
    code = f"""
import torch

# Get the op function
op_fn = getattr(torch, "{op_name}")

# Generate inputs based on op
if "{op_name}" == "abs":
    inputs = (torch.randn(64, dtype=torch.float16),)
elif "{op_name}" == "mul":
    inputs = (
        torch.randn(64, dtype=torch.float16),
        torch.randn(64, dtype=torch.float16),
    )
else:
    raise ValueError(f"Unknown op: {{op_name}}")

# Run on CPU
cpu_result = op_fn(*inputs)

# Compile and run on Spyre
compiled_fn = torch.compile(op_fn, backend="inductor")
spyre_inputs = tuple(inp.to("spyre") for inp in inputs)
spyre_result = compiled_fn(*spyre_inputs).cpu()

torch.testing.assert_close(
    spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
)
print("PASS")
"""
    env = os.environ.copy()
    env.update(env_vars)

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    return "PASS" in result.stdout


class TestLaunchJobPlan:
    """Test suite for JobPlan-backed compiled op execution."""

    def test_abs_matches_cpu_no_symbols(self):
        """Run compiled abs op without symbolic args and compare to CPU."""
        assert run_compiled_op_in_subprocess("abs", {"DUMP_SPYRE_CODE": "1"})

    def test_abs_matches_cpu_with_symbols(self):
        """Run compiled abs op with symbolic args and compare to CPU."""
        assert run_compiled_op_in_subprocess(
            "abs", {"DUMP_SPYRE_CODE": "1", "BUNDLE_SYMBOLIC_ARGS": "1"}
        )

    def test_mul_matches_cpu_no_symbols(self):
        """Run compiled mul op without symbolic args and compare to CPU."""
        assert run_compiled_op_in_subprocess("mul", {"DUMP_SPYRE_CODE": "1"})

    def test_mul_matches_cpu_with_symbols(self):
        """Run compiled mul op with symbolic args and compare to CPU."""
        assert run_compiled_op_in_subprocess(
            "mul", {"DUMP_SPYRE_CODE": "1", "BUNDLE_SYMBOLIC_ARGS": "1"}
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
