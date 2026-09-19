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

import os
import sys
import tempfile

import pytest
import torch
import torch._dynamo
from torch_spyre import _C as _spyre_C
from torch.testing._internal.common_utils import TestCase

from test_prepare_kernel import TestPrepareKernel as tpk

# Make the inductor test utilities importable from tests/inductor/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inductor"))
from utils_inductor import _compile_and_run  # noqa: E402

import torch_spyre._inductor.wsr.propagate_named_dims as _pnd  # noqa: E402
from torch_spyre._inductor import spyre_hint  # noqa: E402

# Stream ID of S_prep (kHostComputeStreamStartPerDevice = 65 in spyre_stream.cpp).
_HOST_COMPUTE_STREAM_START = 65

# Number of successive launches used for pipelining tests (C2).
_PIPELINE_DEPTH = 4


def _sync_all(device: torch.device) -> None:
    """Drain both S_dev and S_prep (ID 65). Ensures both streams are completely idle before checking results"""
    torch.accelerator.synchronize(device)
    prep = _spyre_C.host_compute_stream_by_id(_HOST_COMPUTE_STREAM_START, device)
    prep.synchronize()


class TestSingleIterationCorrectness(TestCase):
    """Single tiled abs launch (A÷4) with tracker on: result matches CPU."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        if not _spyre_C.get_hazard_tracker_enabled():
            self.skipTest("Correctness test requires SPYRE_HAZARD_TRACKER=1")
        torch._dynamo.reset()
        _pnd.reset()

    def _setup_tiled_input(self, x_cpu):
        """Move x_cpu to device and attach named dims for A-tiling."""
        x_dev = x_cpu.to("spyre")
        _pnd.declare_tensor_dim("A", x_cpu.shape[0])
        _pnd.declare_tensor_dim("B", x_cpu.shape[1])
        _pnd.name_tensor_dims(x_dev, ["A", "B"])
        return x_dev

    def _tiled_abs(self, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return torch.abs(x)

    def test_single_tiled_launch_matches_cpu(self):
        """Single tiled abs launch result matches CPU — tracker-inserted edge is correct."""
        torch.manual_seed(0xC0A75E)
        x_cpu = torch.randn((256, 256), dtype=torch.float16)
        cpu_result = torch.abs(x_cpu)

        x_dev = self._setup_tiled_input(x_cpu)
        spyre_result = _compile_and_run(self._tiled_abs, [x_dev], "spyre")

        torch.testing.assert_close(
            spyre_result.cpu(),
            cpu_result,
            atol=0.1,
            rtol=0.1,
            msg="Tiled abs differs from CPU — cross-stream H2D→Compute edge may be missing.",
        )
        self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)


class TestWithinLaunchOverlapCorrectness(TestCase):
    """K-tiled mm (K÷4, 4 prep→compute handoffs per launch): result matches CPU."""

    # M, K, N chosen so K/T is stick-aligned; small scale keeps fp16 error bounded.
    _M, _K, _N = 64, 512, 32

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        if not _spyre_C.get_hazard_tracker_enabled():
            self.skipTest("Within-launch overlap test requires SPYRE_HAZARD_TRACKER=1")
        torch._dynamo.reset()
        _pnd.reset()

    def _declare_matmul_dims(self):
        _pnd.declare_tensor_dim("M", self._M)
        _pnd.declare_tensor_dim("K", self._K)
        _pnd.declare_tensor_dim("N", self._N)

    def _setup_matmul_inputs(self, a_cpu, b_cpu):
        a_dev = a_cpu.to("spyre")
        b_dev = b_cpu.to("spyre")
        _pnd.name_tensor_dims(a_dev, ["M", "K"])
        _pnd.name_tensor_dims(b_dev, ["K", "N"])
        return a_dev, b_dev

    @staticmethod
    def _k_tiled_mm(a, b):
        """mm tiled over K÷4 — emits 4 repeated HostCompute→H2D→Compute handoffs."""
        _pnd.name_tensor_dims(a, ["M", "K"])
        _pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(num_tiles_per_dim={"K": 4}):
            return torch.mm(a, b)

    def test_k_tiled_matmul_matches_cpu(self):
        """K-tiled mm result matches CPU — all 4 prep→compute handoffs ordered correctly."""
        torch.manual_seed(0xAFFE)
        a_cpu = torch.randn(self._M, self._K, dtype=torch.float16) * 0.01
        b_cpu = torch.randn(self._K, self._N, dtype=torch.float16) * 0.01
        cpu_result = torch.mm(a_cpu, b_cpu)

        self._declare_matmul_dims()
        a_dev, b_dev = self._setup_matmul_inputs(a_cpu, b_cpu)
        spyre_result = _compile_and_run(self._k_tiled_mm, [a_dev, b_dev], "spyre")

        torch.testing.assert_close(
            spyre_result.cpu(),
            cpu_result,
            atol=0.05,
            rtol=0.05,
            msg=(
                "K-tiled mm differs from CPU — a missing cross-stream edge on one "
                "of the 4 K-tiles would corrupt the partial accumulation."
            ),
        )
        self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)


class TestAcrossLaunchPipelining(TestCase):
    """Split routing smoke test: HostCompute reaches S_prep (DCI error proves it)."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        if not _spyre_C.get_hazard_tracker_enabled():
            self.skipTest("C2 requires SPYRE_HAZARD_TRACKER=1")

    def _make_split_plan(self, tmpdir):
        spyrecode_dir = tpk().create_mock_spyrecode(
            tmpdir,
            exec_command="ComputeOnHost",
            exec_properties={
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["0"],  # fake-symbols: skips DataConvertInfoGenerate
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            },
        )
        return _spyre_C.prepare_kernel(spyrecode_dir)

    def test_pipelined_launches_reach_host_compute_step(self):
        """HostCompute reaches S_prep (split active): mock HCM raises 'Expect one DCI'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_plan = self._make_split_plan(tmpdir)

            self.assertEqual(job_plan.num_steps(), 3)
            self.assertEqual(
                [job_plan.get_step_stream_role(i) for i in range(3)],
                ["Prep", "Prep", "Dev"],
            )

            stream = torch.Stream(self.device)
            with stream:
                with self.assertRaisesRegex(RuntimeError, "Expect one DCI"):
                    _spyre_C.launch_jobplan(job_plan, [])


class TestAcrossLaunchPipeliningCorrectness(TestCase):
    """N pipelined K-tiled mm launches (different addresses): each matches CPU."""

    _M, _K, _N = 64, 512, 32

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        if not _spyre_C.get_hazard_tracker_enabled():
            self.skipTest("Across-launch correctness requires SPYRE_HAZARD_TRACKER=1")
        torch._dynamo.reset()
        _pnd.reset()

    def _declare_dims(self):
        _pnd.declare_tensor_dim("M", self._M)
        _pnd.declare_tensor_dim("K", self._K)
        _pnd.declare_tensor_dim("N", self._N)

    @staticmethod
    def _k_tiled_mm(a, b):
        _pnd.name_tensor_dims(a, ["M", "K"])
        _pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(num_tiles_per_dim={"K": 4}):
            return torch.mm(a, b)

    def test_pipelined_launches_each_match_cpu(self):
        """_PIPELINE_DEPTH pipelined launches (no sync between) each match their CPU reference."""
        torch.manual_seed(0xAFFE)
        inputs = [
            (
                torch.randn(self._M, self._K, dtype=torch.float16) * 0.01,
                torch.randn(self._K, self._N, dtype=torch.float16) * 0.01,
            )
            for _ in range(_PIPELINE_DEPTH)
        ]
        cpu_results = [torch.mm(a, b) for a, b in inputs]

        self._declare_dims()

        # Compile once, then fire all launches back-to-back (pipelined).
        compiled = torch.compile(self._k_tiled_mm, backend="inductor")
        spyre_results = []
        for a_cpu, b_cpu in inputs:
            a_dev = a_cpu.to("spyre")
            b_dev = b_cpu.to("spyre")
            _pnd.name_tensor_dims(a_dev, ["M", "K"])
            _pnd.name_tensor_dims(b_dev, ["K", "N"])
            spyre_results.append(compiled(a_dev, b_dev))

        # Single sync after all launches — this is what creates the overlap.
        _sync_all(self.device)

        for i, (spyre_out, cpu_out) in enumerate(zip(spyre_results, cpu_results)):
            torch.testing.assert_close(
                spyre_out.cpu(),
                cpu_out,
                atol=0.05,
                rtol=0.05,
                msg=(
                    f"Launch {i} result differs from CPU — "
                    "cross-stream ordering may be wrong for pipelined launches."
                ),
            )
        self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)


class TestFallbackNoPrepSteps(TestCase):
    """Pure ComputeOnDevice plan: no Prep roles, router never splits, result matches CPU."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        torch._dynamo.reset()

    def test_no_prep_roles_in_pure_compute_plan(self):
        """C5a structural — every step has role Dev, no Prep steps exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(tmpdir)
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            for i in range(job_plan.num_steps()):
                self.assertEqual(
                    job_plan.get_step_stream_role(i),
                    "Dev",
                    msg=f"step {i} has unexpected Prep role in a pure-compute plan (C5a)",
                )

    def test_pure_compute_launch_matches_cpu(self):
        """C5a runtime — compiled abs runs on device and matches CPU reference."""
        x = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.abs(x)

        compiled = torch.compile(torch.abs, backend="inductor")
        spyre_result = compiled(x.to("spyre")).cpu()

        torch.testing.assert_close(spyre_result, cpu_result, atol=0.1, rtol=0.1)
        self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)

    def test_pure_compute_pipelined_matches_cpu(self):
        """C5a pipelining — repeated compiled abs calls all match CPU reference."""
        x = torch.randn(64, dtype=torch.float16)
        cpu_result = torch.abs(x)

        compiled = torch.compile(torch.abs, backend="inductor")
        spyre_x = x.to("spyre")

        for _ in range(_PIPELINE_DEPTH):
            spyre_result = compiled(spyre_x).cpu()
            torch.testing.assert_close(spyre_result, cpu_result, atol=0.1, rtol=0.1)

        self.assertEqual(_spyre_C.get_device_state(), _spyre_C.SpyreDeviceState.Ok)


class TestFallbackTrackerOff(TestCase):
    """Tracker off: plan shape unchanged, all steps route to S_dev, HostCompute still fires."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")
        if _spyre_C.get_hazard_tracker_enabled():
            self.skipTest("C5b requires SPYRE_HAZARD_TRACKER=0 (tracker-off path)")

    def test_plan_shape_unchanged_when_tracker_off(self):
        """C5b structural — prepare emits the same bare triple regardless of tracker flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            self.assertEqual(job_plan.num_steps(), 3)
            self.assertEqual(
                [job_plan.get_step_type(i) for i in range(3)],
                ["HostCompute", "H2D", "Compute"],
            )
            # Roles are still baked in at prepare time; the router ignores them
            # at launch when the tracker is off.
            self.assertEqual(
                [job_plan.get_step_stream_role(i) for i in range(3)],
                ["Prep", "Prep", "Dev"],
            )

    def test_correction_triple_launch_reaches_host_compute_step(self):
        """HostCompute fires on S_dev (tracker off): mock HCM raises 'Expect one DCI'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir,
                exec_command="ComputeOnHost",
                exec_properties={
                    "ohandle": "output_buffer",
                    "size": "1024",
                    "ishape": ["0"],
                    "ihandle": "",
                    "hcm": {"vdci": {}, "senConstants": []},
                },
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            stream = torch.Stream(self.device)
            with stream:
                with self.assertRaisesRegex(RuntimeError, "Expect one DCI"):
                    _spyre_C.launch_jobplan(job_plan, [])

    def test_pipelined_launches_tracker_off_reach_host_compute_step(self):
        """C5b pipelining — repeated launches all route HostCompute to S_dev."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = tpk().create_mock_spyrecode(
                tmpdir,
                exec_command="ComputeOnHost",
                exec_properties={
                    "ohandle": "output_buffer",
                    "size": "1024",
                    "ishape": ["0"],
                    "ihandle": "",
                    "hcm": {"vdci": {}, "senConstants": []},
                },
            )
            job_plan = _spyre_C.prepare_kernel(spyrecode_dir)

            stream = torch.Stream(self.device)
            with stream:
                with self.assertRaisesRegex(RuntimeError, "Expect one DCI"):
                    for _ in range(_PIPELINE_DEPTH):
                        _spyre_C.launch_jobplan(job_plan, [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
