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
import tempfile
from typing import Tuple

import pytest
from torch.testing._internal.common_utils import TestCase
import torch
import torch._dynamo
import torch_spyre

from torch_spyre._inductor import config as _spyre_config
from test_prepare_kernel import TestPrepareKernel as tpk


def _run_compiled_op(op_name: str, symbolic_args: bool) -> None:
    """
    Compile an op with SpyreCode and run it on Spyre, comparing to CPU.

    Uses a fresh dynamo compile cache each call to ensure the kernel runner is
    re-instantiated. Runs in-process (no subprocess) so the Spyre VFIO device
    opened by the test session is reused rather than triggering a second
    exclusive open from a child process.
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

    old_sym = os.environ.get("BUNDLE_SYMBOLIC_ARGS")
    try:
        # Keep the C++ prepare_kernel env var in sync with the Python config
        # patch: prepare_kernel reads BUNDLE_SYMBOLIC_ARGS directly from the
        # process environment, so patching only the Python config is insufficient.
        os.environ["BUNDLE_SYMBOLIC_ARGS"] = "1" if symbolic_args else "0"
        with _spyre_config.patch(bundle_symbolic_args=symbolic_args):  # type: ignore[attr-defined]
            compiled_fn = torch.compile(op_fn, backend="inductor")
            spyre_inputs = tuple(inp.to("spyre") for inp in inputs)
            spyre_result = compiled_fn(*spyre_inputs).cpu()
    finally:
        if old_sym is None:
            os.environ.pop("BUNDLE_SYMBOLIC_ARGS", None)
        else:
            os.environ["BUNDLE_SYMBOLIC_ARGS"] = old_sym

    torch.testing.assert_close(
        spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
    )


class TestLaunchJobPlan(TestCase):
    """Test suite for JobPlan-backed compiled op execution.

    Each op is exercised twice: once with symbolic_args=True (the default since
    BUNDLE_SYMBOLIC_ARGS=1 was made the process default) and once with
    symbolic_args=False (the non-default override path, retained as a regression
    guard for users who explicitly disable symbolic args).
    """

    def test_abs_matches_cpu_no_symbols(self):
        """abs with symbolic_args=False (non-default override path)."""
        _run_compiled_op("abs", symbolic_args=False)

    def test_abs_matches_cpu_with_symbols(self):
        """abs with symbolic_args=True (default path)."""
        _run_compiled_op("abs", symbolic_args=True)

    def test_mul_matches_cpu_no_symbols(self):
        """mul with symbolic_args=False (non-default override path)."""
        _run_compiled_op("mul", symbolic_args=False)

    def test_mul_matches_cpu_with_symbols(self):
        """mul with symbolic_args=True (default path)."""
        _run_compiled_op("mul", symbolic_args=True)

    def test_invalid_hcm_metadata_surfaces_on_synchronize(self):
        """Host callback failures should surface as RuntimeError on stream synchronize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_exec_plan = [
                {
                    "command": "ComputeOnHost",
                    "properties": {
                        "ohandle": "output_buffer",
                        "size": "1024",
                        "ishape": ["0"],
                        "ihandle": "",
                        "hcm": {
                            "vdci": {},
                            "senConstants": [],
                        },
                    },
                },
                {
                    "command": "DataTransfer",
                    "properties": {
                        "dirn": "false",
                        "host_handle": "output_buffer",
                        "dev_ptr": "120259084288",
                        "size": "1024",
                    },
                },
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            test_pk = tpk()
            spyrecode_dir = test_pk.create_mock_spyrecode(
                tmpdir, job_exec_plan=job_exec_plan
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)
            stream = torch.Stream("spyre")

            with stream:
                with pytest.raises(RuntimeError, match="Expect one DCI"):
                    torch_spyre._C.launch_jobplan(job_plan, [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
