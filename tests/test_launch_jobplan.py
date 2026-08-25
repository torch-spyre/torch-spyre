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

import json
import os
import tempfile
from typing import Tuple

import pytest
from torch.testing._internal.common_utils import TestCase
import torch
import torch._dynamo
import torch_spyre

from test_prepare_kernel import TestPrepareKernel as tpk


def _run_compiled_op(op_name: str) -> None:
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

    compiled_fn = torch.compile(op_fn, backend="inductor")
    spyre_inputs = tuple(inp.to("spyre") for inp in inputs)
    spyre_result = compiled_fn(*spyre_inputs).cpu()

    torch.testing.assert_close(
        spyre_result, cpu_result, atol=0.1, rtol=0.1, equal_nan=True
    )


class TestLaunchJobPlan(TestCase):
    """Test suite for JobPlan-backed compiled op execution."""

    def test_abs_matches_cpu(self):
        _run_compiled_op("abs")

    def test_mul_matches_cpu(self):
        _run_compiled_op("mul")

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


def _build_d2h_jobplan(tmpdir: str, dev_ptr: int, size_bytes: int):
    """Build a JobPlan with a single D2H DataTransfer step from dev_ptr.

    prepare_kernel resolves a D2H whose dev_ptr is in a tensor segment into
    JobPlanStepD2H whose device address is looked up from the launch args at
    launch time.  This lets us drive that deferred path with mock SpyreCode
    instead of relying on a particular backend-compiler output.
    """
    spyrecode_dir = os.path.join(tmpdir, "spyreCodeDir")
    os.makedirs(spyrecode_dir, exist_ok=True)

    spyrecode_json = {
        "JobPreparationPlan": [
            {"command": "Allocate", "properties": {"size": "1024"}},
            {
                "command": "InitTransfer",
                "properties": {
                    "init_bin_file": "init_binary.bin",
                    "dev_ptr": "120259084288",
                    "size": "1024",
                },
            },
        ],
        "JobExecPlan": [
            {
                "command": "DataTransfer",
                "properties": {
                    "dirn": "true",  # D2H
                    "host_handle": "d2h_output",
                    "dev_ptr": str(dev_ptr),
                    "size": str(size_bytes),
                },
            },
        ],
    }
    with open(os.path.join(spyrecode_dir, "spyrecode.json"), "w") as f:
        json.dump(spyrecode_json, f)
    with open(os.path.join(spyrecode_dir, "init_binary.bin"), "wb") as f:
        f.write(b"\x00" * 1024)

    return torch_spyre._C.prepare_kernel(spyrecode_dir)  # type: ignore[attr-defined]


class TestD2HFromTensorSegment(TestCase):
    """Drive the D2H path via prepare_kernel +
    launch_jobplan with mock SpyreCode.
    """

    @pytest.fixture(autouse=True)
    def _prepare_with_symbolic_args(self):
        torch.zeros(1, device="spyre")
        old_val = os.environ.get("BUNDLE_SYMBOLIC_ARGS")
        os.environ["BUNDLE_SYMBOLIC_ARGS"] = "0"
        try:
            yield
        finally:
            if old_val is None:
                os.environ.pop("BUNDLE_SYMBOLIC_ARGS", None)
            else:
                os.environ["BUNDLE_SYMBOLIC_ARGS"] = old_val

    def test_tensor_segment_d2h_out_of_range(self):
        """D2H from a tensor segment at offset 0 resolves and launches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_plan = _build_d2h_jobplan(tmpdir, 34359738368, 128)
            assert job_plan.get_step_type(0) == "D2H"

            inp = torch.zeros(128, dtype=torch.float16, device="spyre")
            out = torch.zeros(128, dtype=torch.float16, device="spyre")
            with pytest.raises(
                RuntimeError, match="D2H tensor-segment lookup out of range"
            ):
                torch_spyre._C.launch_jobplan(job_plan, [inp, out])

    def test_tensor_segment_d2h(self):
        """D2H from a tensor segment at a non-zero offset
        exercises the offset arithmetic in JobPlanStepD2H::construct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_plan = _build_d2h_jobplan(tmpdir, 0, 128)
            assert job_plan.get_step_type(0) == "D2H"

            inp = torch.zeros(128, dtype=torch.float16, device="spyre")
            out = torch.zeros(128, dtype=torch.float16, device="spyre")
            torch_spyre._C.launch_jobplan(job_plan, [inp, out])

    def test_tensor_segment_d2h_out_of_bounds(self):
        """D2H from a tensor segment at a non-zero offset
        exercises the offset arithmetic in JobPlanStepD2H::construct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_plan = _build_d2h_jobplan(tmpdir, 256, 128)
            assert job_plan.get_step_type(0) == "D2H"

            inp = torch.zeros(128, dtype=torch.float16, device="spyre")
            out = torch.zeros(128, dtype=torch.float16, device="spyre")
            with pytest.raises(RuntimeError, match="D2H transfer out of bounds"):
                torch_spyre._C.launch_jobplan(job_plan, [inp, out])


class TestSymbolicArg(TestCase):
    """
    Unit tests for the SymbolicArg typed payload.

    Notes:
        Tests will need to be reworked once kdimension is implemented.
    """

    def test_kdimension_entry_reaches_construct_and_raises(self):
        """A kDimension entry traverses Python → pybind → LaunchContext →
        construct() and raises 'kDimension is not yet implemented'.

        Uses a real-symbols ComputeOnHost step (non-{0} ishape) so the
        payload is consumed rather than short-circuited by the fake-symbols
        nullptr path.  The raise proves the entry survived the full carrier
        path and was read positionally from slot 0.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            t = torch.zeros(64, dtype=torch.float16, device="spyre")
            payload = [
                torch_spyre._C.SymbolicArg(
                    kind=torch_spyre._C.SymbolicArgKind.kDimension,
                    tensor_id=0,
                    dim_index=0,
                )
            ]

            stream = torch.Stream("spyre")
            with stream:
                with pytest.raises(
                    RuntimeError, match="kDimension is not yet implemented"
                ):
                    torch_spyre._C.launch_jobplan(job_plan, [t], symbolic_args=payload)

    def test_symbolic_arg_attributes_and_repr_roundtrip(self):
        """SymbolicArg fields and repr survive pybind construction.

        No hardware required — pure Python boundary check.
        """
        addr_arg = torch_spyre._C.SymbolicArg(
            kind=torch_spyre._C.SymbolicArgKind.kAddress,
            tensor_id=1,
        )
        self.assertEqual(addr_arg.kind, torch_spyre._C.SymbolicArgKind.kAddress)
        self.assertEqual(addr_arg.tensor_id, 1)
        self.assertEqual(addr_arg.dim_index, -1)
        self.assertEqual(addr_arg.value, -1)

        dim_arg = torch_spyre._C.SymbolicArg(
            kind=torch_spyre._C.SymbolicArgKind.kDimension,
            tensor_id=0,
            dim_index=2,
            value=48,
        )
        self.assertEqual(dim_arg.kind, torch_spyre._C.SymbolicArgKind.kDimension)
        self.assertEqual(dim_arg.tensor_id, 0)
        self.assertEqual(dim_arg.dim_index, 2)
        self.assertEqual(dim_arg.value, 48)

        r = repr(addr_arg)
        self.assertIn("tensor_id=1", r)
        self.assertIn("dim_index=-1", r)

    def test_tensor_id_out_of_range_raises(self):
        """A payload entry whose tensor_id exceeds the tensor list length
        raises a loud bounds-check error rather than silently reading OOB.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            t = torch.zeros(64, dtype=torch.float16, device="spyre")
            # tensor list has 1 entry (index 0); tensor_id=5 is out of range
            payload = [
                torch_spyre._C.SymbolicArg(
                    kind=torch_spyre._C.SymbolicArgKind.kAddress,
                    tensor_id=5,
                )
            ]

            stream = torch.Stream("spyre")
            with stream:
                with pytest.raises(RuntimeError, match="tensor_id=5 out of range"):
                    torch_spyre._C.launch_jobplan(job_plan, [t], symbolic_args=payload)

    def test_kaddress_ordering_forward_and_reversed_differ(self):
        """Forward and reversed payloads over two tensors produce different
        resolved vectors, each matching the expected per-slot address.

        Uses _resolve_symbolic_args, which calls
        JobPlanStepHostCompute::resolveSymbolicArgs — the same function used
        by the typed-payload resolution path at launch time — so the result is
        identical to what would be passed to deeptools.
        """
        t0 = torch.zeros(64, dtype=torch.float16, device="spyre")
        t1 = torch.zeros(64, dtype=torch.float16, device="spyre")

        kAddr = torch_spyre._C.SymbolicArgKind.kAddress

        # Ground-truth address for each tensor: resolve each alone at slot 0
        # so the result is independent of ordering.
        addr_t0 = torch_spyre._C._resolve_symbolic_args(
            [t0], [torch_spyre._C.SymbolicArg(kind=kAddr, tensor_id=0)]
        )[0]
        addr_t1 = torch_spyre._C._resolve_symbolic_args(
            [t1], [torch_spyre._C.SymbolicArg(kind=kAddr, tensor_id=0)]
        )[0]

        # Tensors must be at distinct addresses — if they coincidentally share
        # one the reversed payload would produce the same vector and the
        # ordering assertion would be meaningless.
        self.assertNotEqual(
            addr_t0,
            addr_t1,
            "t0 and t1 share a device address; ordering test would be meaningless",
        )

        payload_fwd = [
            torch_spyre._C.SymbolicArg(kind=kAddr, tensor_id=0),
            torch_spyre._C.SymbolicArg(kind=kAddr, tensor_id=1),
        ]
        payload_rev = [
            torch_spyre._C.SymbolicArg(kind=kAddr, tensor_id=1),
            torch_spyre._C.SymbolicArg(kind=kAddr, tensor_id=0),
        ]

        resolved_fwd = torch_spyre._C._resolve_symbolic_args([t0, t1], payload_fwd)
        resolved_rev = torch_spyre._C._resolve_symbolic_args([t0, t1], payload_rev)

        # The two vectors must differ.
        self.assertNotEqual(
            resolved_fwd,
            resolved_rev,
            "Forward and reversed payloads produced identical resolved vectors",
        )

        # Forward: slot 0 → t0, slot 1 → t1.
        self.assertEqual(
            resolved_fwd[0], addr_t0, "slot 0 of forward payload must be addr_t0"
        )
        self.assertEqual(
            resolved_fwd[1], addr_t1, "slot 1 of forward payload must be addr_t1"
        )

        # Reversed: slot 0 → t1, slot 1 → t0.
        self.assertEqual(
            resolved_rev[0], addr_t1, "slot 0 of reversed payload must be addr_t1"
        )
        self.assertEqual(
            resolved_rev[1], addr_t0, "slot 1 of reversed payload must be addr_t0"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
