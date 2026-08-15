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

from torch_spyre._inductor import config as _spyre_config
from torch_spyre.execution import kernel_runner
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


class TestCorrectionBufferWarBarrier(TestCase):
    """Regression guard for the reused correction-buffer WAR pipeline barrier.

    The program-correction path emits a HostCompute step that writes a pinned
    "correction" buffer, followed by an H2D that reads it. The JobPlan (and thus
    the pinned buffer) is built once and reused across every launch of a compiled
    op. Because flex dispatches the HostCompute callback INLINE on the caller
    thread, the HostCompute step must carry ``pipeline_barrier=True`` so a later
    launch's callback waits for the prior launch's H2D to finish reading the
    buffer before overwriting it (cross-launch write-after-read; see flex #1479
    and the ``JobPlanStepHostCompute`` constructor in job_plan.h).

    This complements the mock-based ``test_pipeline_barrier_correction_sequence``
    in test_prepare_kernel.py, which never compiles a real op and never exercises
    reuse. Here we:

      1. Compile a REAL correction-path op (BUNDLE_SYMBOLIC_ARGS=1, the process
         default), so the JobPlan comes from the real compile -> prepare_kernel
         path rather than a hand-built mock.
      2. Launch it MORE THAN ONCE, so the same JobPlan and pinned correction
         buffer are genuinely reused across iterations (the cross-launch reuse
         the barrier protects).
      3. Assert, on the JobPlan that is actually launched each iteration, that
         the HostCompute step carries ``pipeline_barrier=True``.

    Polarity as a regression guard: this PASSES on current (correct) code and
    fails deterministically if a future change drops the barrier line
    (job_plan.h:449) or stops emitting the HostCompute step. It does not depend
    on reproducing the underlying data race numerically -- that race is timing
    dependent and cannot be triggered reliably on real hardware, so a numeric
    assertion would be a false guard (green even when the barrier is dropped).
    """

    NUM_ITERATIONS = 3

    def test_reused_correction_buffer_keeps_barrier_across_launches(self):
        torch._dynamo.reset()

        captured = []
        orig_launch = kernel_runner.launch_jobplan

        def _spy(job_plan, args):
            captured.append(job_plan)
            return orig_launch(job_plan, args)

        old_sym = os.environ.get("BUNDLE_SYMBOLIC_ARGS")
        os.environ["BUNDLE_SYMBOLIC_ARGS"] = "1"
        kernel_runner.launch_jobplan = _spy
        try:
            with _spyre_config.patch(bundle_symbolic_args=True):  # type: ignore[attr-defined]
                torch.manual_seed(42)
                compiled_fn = torch.compile(torch.abs, backend="inductor")
                # Launch the SAME compiled fn multiple times so the JobPlan (and
                # its pinned correction buffer) is reused across iterations.
                for _ in range(self.NUM_ITERATIONS):
                    x = torch.randn(64, dtype=torch.float16).to("spyre")
                    compiled_fn(x).cpu()
        finally:
            kernel_runner.launch_jobplan = orig_launch
            if old_sym is None:
                os.environ.pop("BUNDLE_SYMBOLIC_ARGS", None)
            else:
                os.environ["BUNDLE_SYMBOLIC_ARGS"] = old_sym

        # The compiled op must have launched on every iteration.
        assert len(captured) >= self.NUM_ITERATIONS, (
            f"expected >= {self.NUM_ITERATIONS} launches, got {len(captured)}"
        )

        # The correction path must reuse a single JobPlan across launches; that
        # reuse is the whole reason the WAR barrier is needed. If each launch
        # rebuilt the JobPlan (and buffer), there would be no cross-iteration
        # hazard and this guard would be testing the wrong thing.
        assert len({id(jp) for jp in captured}) == 1, (
            "expected one reused JobPlan across launches, got "
            f"{len({id(jp) for jp in captured})} distinct JobPlans; the "
            "correction buffer is no longer reused, so the cross-iteration WAR "
            "this test guards is not being exercised"
        )

        hostcompute_seen = 0
        for job_plan in captured:
            for idx in range(job_plan.num_steps()):
                if job_plan.get_step_type(idx) != "HostCompute":
                    continue
                hostcompute_seen += 1
                assert job_plan.get_step_pipeline_barrier(idx) is True, (
                    f"HostCompute step {idx} must carry pipeline_barrier=True to "
                    "close the cross-launch WAR on the reused pinned correction "
                    "buffer (flex #1479). With the barrier dropped, flex runs the "
                    "next launch's inline host callback while the prior launch's "
                    "H2D is still reading the shared buffer, corrupting the "
                    "correction data"
                )

        # Fail loudly rather than pass vacuously if the correction path stopped
        # emitting a HostCompute step (e.g. a compiler change): then the barrier
        # this test guards would silently no longer exist.
        assert hostcompute_seen >= self.NUM_ITERATIONS, (
            "no HostCompute step found on every launched JobPlan; the "
            "program-correction path did not run, so the WAR barrier guard was "
            "not exercised"
        )


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

    return torch_spyre._C.prepare_kernel(spyrecode_dir)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
