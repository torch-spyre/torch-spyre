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

"""Tests for PrepareKernel Python bindings and JobPlan verification."""

import copy
import json
import os
import tempfile

import pytest
import torch
import torch_spyre


@pytest.fixture(scope="module", autouse=True)
def initialize_runtime():
    """Initialize Spyre runtime before running tests."""
    # Initialize torch with spyre device to start runtime
    torch.zeros(1, device="spyre")
    yield
    # Runtime cleanup happens automatically


class TestPrepareKernel:
    """Test suite for PrepareKernel and JobPlan bindings."""

    def create_mock_spyrecode(
        self,
        tmpdir,
        exec_command="ComputeOnDevice",
        exec_properties=None,
        job_exec_plan=None,
    ):
        """Create a mock SpyreCode directory structure for testing.

        Args:
            tmpdir: Temporary directory path
            exec_command: Command type for JobExecPlan (default: "ComputeOnDevice")
            exec_properties: Properties dict for the exec command (default: auto-generated)

        Returns:
            Path to the SpyreCode directory
        """
        spyrecode_dir = os.path.join(tmpdir, "spyreCodeDir")
        os.makedirs(spyrecode_dir, exist_ok=True)

        # Auto-generate properties if not provided
        if job_exec_plan is None:
            if exec_properties is None:
                if exec_command == "ComputeOnDevice":
                    exec_properties = {"job_bin_ptr": "120259084288"}
                elif exec_command == "ComputeOnHost":
                    exec_properties = {
                        "ohandle": "output_buffer",
                        "size": "1024",
                        "ishape": ["64", "16"],
                        "ihandle": "",
                        "hcm": {"vdci": {}, "senConstants": []},
                    }

            # Build JobExecPlan
            job_exec_plan = [{"command": exec_command, "properties": exec_properties}]

            # If ComputeOnHost, add required H2D and Compute steps
            if exec_command == "ComputeOnHost":
                # Add H2D transfer (transfers output_buffer to device)
                job_exec_plan.append(
                    {
                        "command": "DataTransfer",
                        "properties": {
                            "dirn": "false",
                            "host_handle": "output_buffer",
                            "dev_ptr": "120259084288",
                            "size": "1024",
                        },
                    }
                )
                # Add Compute step
                job_exec_plan.append(
                    {
                        "command": "ComputeOnDevice",
                        "properties": {"job_bin_ptr": "120259084288"},
                    }
                )
        else:
            job_exec_plan = copy.deepcopy(job_exec_plan)

        # Create a minimal spyrecode.json
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
            "JobExecPlan": job_exec_plan,
        }

        # Write spyrecode.json
        with open(os.path.join(spyrecode_dir, "spyrecode.json"), "w") as f:
            json.dump(spyrecode_json, f, indent=2)

        # Create a dummy binary file
        with open(os.path.join(spyrecode_dir, "init_binary.bin"), "wb") as f:
            f.write(b"\x00" * 1024)

        return spyrecode_dir

    def test_prepare_kernel_basic(self):
        """Test basic PrepareKernel functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)

            # Call prepare_kernel
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            # Verify JobPlan was created
            assert job_plan is not None
            assert isinstance(job_plan, torch_spyre._C.JobPlan)

    def test_job_plan_num_steps(self):
        """Test JobPlan.num_steps() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            # Should have 1 step (ComputeOnDevice)
            assert job_plan.num_steps() == 1

    def test_job_plan_allocation_size(self):
        """Test JobPlan.job_allocation_size() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            # Should match the allocated size (1024 bytes)
            assert job_plan.job_allocation_size() == 1024

    def test_job_plan_step_type(self):
        """Test JobPlan.get_step_type() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            # First step should be ComputeSpecialize
            assert job_plan.get_step_type(0) == "Compute"

    def test_prepare_kernel_invalid_directory(self):
        """Test PrepareKernel with invalid directory."""
        with pytest.raises(RuntimeError, match="SpyreCode directory does not exist"):
            torch_spyre._C.prepare_kernel("/nonexistent/directory")

    def test_prepare_kernel_missing_json(self):
        """Test PrepareKernel with missing spyrecode.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory but no spyrecode.json
            with pytest.raises(RuntimeError, match="spyrecode.json not found"):
                torch_spyre._C.prepare_kernel(tmpdir)

    def test_job_plan_step_index_out_of_range(self):
        """Test JobPlan methods with out-of-range index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            # Should raise error for out-of-range index
            with pytest.raises(RuntimeError, match="Step index out of range"):
                job_plan.get_step_type(999)

    def test_compute_on_host_valid(self):
        """A valid ComputeOnHost triple builds and is instrumented for overlap.

        A well-formed ComputeOnHost entry (ohandle, size, ishape, ihandle, hcm)
        translates to the single triple [HostCompute, H2D, Compute], which the
        two-stream overlap pass (edit 6) rewrites into the 7-step instrumented
        plan and validate() accepts:

            [HostCompute, WaitBack, H2D, SignalForward, WaitForward, Compute,
             SignalBack]

        HostCompute/H2D + their events run on S_prep; Compute + its events run on
        S_dev. The WaitBack sits AFTER HostCompute and BEFORE H2D (the Placement
        Invariant), and the two back-event steps key off the same correction
        region_id resolved from the bound H2D device address.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )

            # Should succeed without exceptions (prepare_kernel runs validate()).
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan is not None
            assert isinstance(job_plan, torch_spyre._C.JobPlan)

            # Instrumented two-stream plan: 7 steps in the byte-identical order.
            assert job_plan.num_steps() == 7
            expected_types = [
                "HostCompute",
                "WaitBack",
                "H2D",
                "SignalForward",
                "WaitForward",
                "Compute",
                "SignalBack",
            ]
            assert [job_plan.get_step_type(i) for i in range(7)] == expected_types

            # Stream roles: HostCompute/WaitBack/H2D/SignalForward on S_prep;
            # WaitForward/Compute/SignalBack on S_dev.
            expected_roles = ["Prep", "Prep", "Prep", "Prep", "Dev", "Dev", "Dev"]
            assert [
                job_plan.get_step_stream_role(i) for i in range(7)
            ] == expected_roles

            # Placement Invariant: WaitBack (1) is strictly between HostCompute
            # (0) and H2D (2).
            assert job_plan.get_step_type(0) == "HostCompute"
            assert job_plan.get_step_type(1) == "WaitBack"
            assert job_plan.get_step_type(2) == "H2D"

            # Both back-event steps key off the SAME correction region_id.
            assert job_plan.get_step_region_id(1) == job_plan.get_step_region_id(6)

    def test_hazard_tracker_emits_bare_split_triple(self):
        """SPYRE_HAZARD_TRACKER: the correction triple is split WITHOUT events.

        With the hazard tracker enabled, PrepareKernel skips the static edge-3/4
        event-step block (and its scalar-region_id TORCH_CHECK). The plain
        [HostCompute, H2D, Compute] triple survives, still carrying its by-type
        roles [Prep, Prep, Dev] -- so the launch router splits it across
        S_prep/S_dev while flex inserts the cross-stream RAW/WAR events
        dynamically at enqueue. No event steps are emitted (3 steps, not 7).
        """
        prev = torch_spyre._C.get_hazard_tracker_enabled()
        try:
            torch_spyre._C.set_hazard_tracker_enabled(True)
            with tempfile.TemporaryDirectory() as tmpdir:
                spyrecode_dir = self.create_mock_spyrecode(
                    tmpdir, exec_command="ComputeOnHost"
                )
                # Must NOT raise: the getNumProgramRegions()<=1 TORCH_CHECK lives
                # inside the skipped block, so it does not run in hazard mode.
                job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

                # Bare split triple: 3 steps, NO event steps inserted.
                assert job_plan.num_steps() == 3
                assert [job_plan.get_step_type(i) for i in range(3)] == [
                    "HostCompute",
                    "H2D",
                    "Compute",
                ]
                # Roles are assigned by step type in the ctors, so the bare
                # triple already carries the split roles -- the split is real.
                assert [job_plan.get_step_stream_role(i) for i in range(3)] == [
                    "Prep",
                    "Prep",
                    "Dev",
                ]
                # pipeline_barrier stays True on EVERY step of the bare split:
                # overlap comes only from the S_prep/S_dev split + flex's
                # dynamic cross-stream events, never from relaxing a barrier.
                assert [job_plan.get_step_pipeline_barrier(i) for i in range(3)] == [
                    True,
                    True,
                    True,
                ]
        finally:
            torch_spyre._C.set_hazard_tracker_enabled(prev)

    def test_hazard_tracker_disabled_by_default(self):
        """Flag OFF: the default static-edge path is byte-identical (7 steps).

        Guards against an accidental default flip: with the hazard tracker
        explicitly disabled, the SAME ComputeOnHost triple is rewritten into the
        7-step instrumented plan with the default roles -- exactly the shipped
        overlap path (cf. test_compute_on_host_valid).
        """
        prev = torch_spyre._C.get_hazard_tracker_enabled()
        try:
            torch_spyre._C.set_hazard_tracker_enabled(False)
            with tempfile.TemporaryDirectory() as tmpdir:
                spyrecode_dir = self.create_mock_spyrecode(
                    tmpdir, exec_command="ComputeOnHost"
                )
                job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

                assert job_plan.num_steps() == 7
                assert [job_plan.get_step_stream_role(i) for i in range(7)] == [
                    "Prep",
                    "Prep",
                    "Prep",
                    "Prep",
                    "Dev",
                    "Dev",
                    "Dev",
                ]
        finally:
            torch_spyre._C.set_hazard_tracker_enabled(prev)

    def test_compute_on_host_missing_ohandle(self):
        """Test that missing ohandle field raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "size": "1024",
                "ishape": ["64", "16"],
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="ComputeOnHost command missing 'ohandle' property"
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_missing_size(self):
        """Test that missing size field raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "ishape": ["64", "16"],
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="ComputeOnHost command missing 'size' property"
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_missing_ishape(self):
        """Test that missing ishape field raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="ComputeOnHost command missing 'ishape' property"
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_missing_ihandle(self):
        """Test that missing ihandle field raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["64", "16"],
                "hcm": {"vdci": {}, "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="ComputeOnHost command missing 'ihandle' property"
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_missing_hcm(self):
        """Test that missing hcm field raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["64", "16"],
                "ihandle": "",
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="ComputeOnHost command missing 'hcm' property"
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_malformed_hcm_string(self):
        """Test that malformed hcm (string instead of object) raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["64", "16"],
                "ihandle": "",
                "hcm": "invalid_hcm_string",
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            # Should raise RuntimeError (exact message depends on JSON/import failure)
            with pytest.raises(RuntimeError):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_malformed_ishape_non_array(self):
        """Test that malformed ishape (non-array) raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": "64",
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="ComputeOnHost 'ishape' must be an array"
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_malformed_ishape_elements(self):
        """Test that malformed ishape elements (non-string) raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": [64, 16],
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="ComputeOnHost 'ishape' elements must be strings"
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_compute_on_host_invalid_ihandle(self):
        """Test that invalid ihandle (non-existent buffer) raises RuntimeError.

        Verifies that when ihandle references a buffer name that was never
        created, a RuntimeError is raised with the buffer name in the error
        message.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["64", "16"],
                "ihandle": "nonexistent_buffer",  # References a buffer that doesn't exist
                "hcm": {"vdci": {}, "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError,
                match="ihandle 'nonexistent_buffer' not found in pinned buffer map",
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_invalid_hcm_metadata_raises_runtime_error(self):
        """Invalid HCM metadata should raise a clean RuntimeError during prepare_kernel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["64", "16"],
                "ihandle": "",
                "hcm": {"vdci": "invalid", "senConstants": []},
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError,
                match="Failed to parse SpyreCode command: .*vdci field",
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_stoull_allocate_negative_size(self):
        """Test that negative size in Allocate command is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_exec_plan = [
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                }
            ]

            spyrecode_dir = os.path.join(tmpdir, "spyreCodeDir")
            os.makedirs(spyrecode_dir, exist_ok=True)

            spyrecode_json = {
                "JobPreparationPlan": [
                    {"command": "Allocate", "properties": {"size": "-1024"}},
                    {
                        "command": "InitTransfer",
                        "properties": {
                            "init_bin_file": "init_binary.bin",
                            "dev_ptr": "120259084288",
                            "size": "1024",
                        },
                    },
                ],
                "JobExecPlan": job_exec_plan,
            }

            with open(os.path.join(spyrecode_dir, "spyrecode.json"), "w") as f:
                json.dump(spyrecode_json, f, indent=2)

            with open(os.path.join(spyrecode_dir, "init_binary.bin"), "wb") as f:
                f.write(b"\x00" * 1024)

            with pytest.raises(
                RuntimeError,
                match="negative value not allowed for unsigned integer",
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    def test_stoull_allocate_negative_size_with_leading_whitespace(self):
        """Test that negative size with leading whitespace is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_exec_plan = [
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                }
            ]

            spyrecode_dir = os.path.join(tmpdir, "spyreCodeDir")
            os.makedirs(spyrecode_dir, exist_ok=True)

            spyrecode_json = {
                "JobPreparationPlan": [
                    {"command": "Allocate", "properties": {"size": "  -512"}},
                    {
                        "command": "InitTransfer",
                        "properties": {
                            "init_bin_file": "init_binary.bin",
                            "dev_ptr": "120259084288",
                            "size": "1024",
                        },
                    },
                ],
                "JobExecPlan": job_exec_plan,
            }

            with open(os.path.join(spyrecode_dir, "spyrecode.json"), "w") as f:
                json.dump(spyrecode_json, f, indent=2)

            with open(os.path.join(spyrecode_dir, "init_binary.bin"), "wb") as f:
                f.write(b"\x00" * 1024)

            with pytest.raises(
                RuntimeError,
                match="negative value not allowed for unsigned integer",
            ):
                torch_spyre._C.prepare_kernel(spyrecode_dir)

    @staticmethod
    def _prepare_with_symbolic_args(spyrecode_dir, symbolic_args):
        """Run prepare_kernel with BUNDLE_SYMBOLIC_ARGS set, then restore it."""
        old_val = os.environ.get("BUNDLE_SYMBOLIC_ARGS")
        try:
            os.environ["BUNDLE_SYMBOLIC_ARGS"] = "1" if symbolic_args else "0"
            return torch_spyre._C.prepare_kernel(spyrecode_dir)
        finally:
            if old_val is None:
                os.environ.pop("BUNDLE_SYMBOLIC_ARGS", None)
            else:
                os.environ["BUNDLE_SYMBOLIC_ARGS"] = old_val

    def test_d2h_tensor_segment(self):
        """D2H from a tensor segment builds a (deferred) D2H step when
        addresses are bound (BUNDLE_SYMBOLIC_ARGS != "1")."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "dirn": "true",
                "host_handle": "d2h_output",
                "dev_ptr": "0",
                "size": "1024",
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="DataTransfer", exec_properties=properties
            )

            job_plan = self._prepare_with_symbolic_args(
                spyrecode_dir, symbolic_args=False
            )

            assert job_plan.num_steps() == 1
            assert job_plan.get_step_type(0) == "D2H"

    def test_d2h_tensor_segment_symbolic(self):
        """D2H from a tensor segment is rejected under symbolic args: the
        transfer must go through the program segment in that mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            properties = {
                "dirn": "true",
                "host_handle": "d2h_output",
                "dev_ptr": "0",
                "size": "1024",
            }
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="DataTransfer", exec_properties=properties
            )

            with pytest.raises(
                RuntimeError, match="D2H dev_ptr must be in program segment"
            ):
                self._prepare_with_symbolic_args(spyrecode_dir, symbolic_args=True)

    def test_pipeline_barrier_dma_steps_default_true(self):
        """H2D and D2H steps must carry pipeline_barrier=True by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # H2D lives at index 2 of the instrumented correction plan
            # ([HostCompute, WaitBack, H2D, ...]).
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.get_step_type(2) == "H2D"
            assert job_plan.get_step_pipeline_barrier(2) is True, (
                "H2D step must carry pipeline_barrier=True by default"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # D2H: standalone DataTransfer with dirn="true"
            job_exec_plan = [
                {
                    "command": "DataTransfer",
                    "properties": {
                        "dirn": "true",
                        "host_handle": "output_buffer",
                        "dev_ptr": "120259084288",
                        "size": "1024",
                    },
                }
            ]
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, job_exec_plan=job_exec_plan
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.get_step_type(0) == "D2H"
            assert job_plan.get_step_pipeline_barrier(0) is True, (
                "D2H step must carry pipeline_barrier=True by default"
            )

    def test_pipeline_barrier_correction_sequence(self):
        """Every step of the instrumented correction plan keeps barrier=True.

        The two-stream PoC preserves STRICT per-stream FIFO for ALL ops,
        including HostCompute: overlap comes from the S_prep/S_dev stream split
        plus the cross-stream forward/back events, NOT from relaxing any op's
        pipeline_barrier. So no step in the instrumented plan
        ([HostCompute, WaitBack, H2D, SignalForward, WaitForward, Compute,
        SignalBack]) may opt out of the barrier -- in particular HostCompute must
        NOT carry the old barrier=False.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.num_steps() == 7

            # Strict FIFO everywhere: barrier=True on every step, no exceptions.
            for i in range(job_plan.num_steps()):
                assert job_plan.get_step_pipeline_barrier(i) is True, (
                    f"step {i} ({job_plan.get_step_type(i)}) must carry "
                    "pipeline_barrier=True; the PoC keeps strict per-stream FIFO "
                    "for all ops and gets overlap from the stream split + events, "
                    "not from a barrier opt-out"
                )

            # Explicitly guard the regressed case: HostCompute is no longer
            # barrier=False.
            assert job_plan.get_step_type(0) == "HostCompute"
            assert job_plan.get_step_pipeline_barrier(0) is True, (
                "HostCompute must keep pipeline_barrier=True (edit 1 removed the "
                "old overlap-via-barrier-opt-out); overlap is via S_prep/S_dev"
            )

    def test_pipeline_barrier_pure_compute_true(self):
        """A standalone ComputeOnDevice step must carry pipeline_barrier=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.num_steps() == 1
            assert job_plan.get_step_type(0) == "Compute"
            assert job_plan.get_step_pipeline_barrier(0) is True, (
                "Compute step must carry pipeline_barrier=True: consumer of "
                "DMA'd inputs (RAW hazard). Inert under STRICT_ORDERING; "
                "load-bearing under OP_ORDERING."
            )

    def test_get_step_pipeline_barrier_out_of_range(self):
        """get_step_pipeline_barrier must raise for an out-of-range index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            with pytest.raises(RuntimeError, match="Step index out of range"):
                job_plan.get_step_pipeline_barrier(999)

    def test_edge4_rolling_slot_cross_plan_hit(self):
        """#2: the rolling edge-4 slot carries a back-event ACROSS two launches.

        Two independently-prepared ComputeOnHost plans (distinct kernels: they
        differ in ishape) whose correction H2D binds into the SAME single
        program region must resolve to the SAME scalar region_id. That key
        collapse is what lets launch i's SignalBack hand its back-event to launch
        i+1's WaitBack -- the cross-kernel WAR the shipped e2e never asserts. The
        region_id is derived from the Allocate size + the bound H2D dev_ptr (via
        the singleton allocator's single program region); ishape is pure
        HostCompute metadata and does NOT feed region_id, so the two plans share
        it. We take both region_ids from REAL prepared plans, then drive the REAL
        setEdge4Slot/getEdge4Slot map (through the test hooks that
        SignalBack/WaitBack construct use) to prove: publish under plan A's
        SignalBack region_id, look up under plan B's WaitBack region_id, get the
        SAME Event (pointer identity) with edge4_slot_hits going 0 -> 1.
        """
        with (
            tempfile.TemporaryDirectory() as tdir_a,
            tempfile.TemporaryDirectory() as tdir_b,
        ):
            dir_a = self.create_mock_spyrecode(
                tdir_a,
                exec_command="ComputeOnHost",
                exec_properties={
                    "ohandle": "output_buffer",
                    "size": "1024",
                    "ishape": ["64", "16"],
                    "ihandle": "",
                    "hcm": {"vdci": {}, "senConstants": []},
                },
            )
            dir_b = self.create_mock_spyrecode(
                tdir_b,
                exec_command="ComputeOnHost",
                exec_properties={
                    "ohandle": "output_buffer",
                    "size": "1024",
                    "ishape": ["16", "64"],  # distinct kernel, same region_id
                    "ihandle": "",
                    "hcm": {"vdci": {}, "senConstants": []},
                },
            )
            plan_a = torch_spyre._C.prepare_kernel(dir_a)
            plan_b = torch_spyre._C.prepare_kernel(dir_b)

            assert plan_a.num_steps() == 7 and plan_b.num_steps() == 7

            # Key collapse: plan A's SignalBack (idx 6) and plan B's WaitBack
            # (idx 1) resolve to the SAME correction region_id even though they
            # come from two independent prepare_kernel calls for two kernels.
            rid_a_signal = plan_a.get_step_region_id(6)
            rid_b_wait = plan_b.get_step_region_id(1)
            assert rid_a_signal == rid_b_wait, (
                "two ComputeOnHost plans whose correction H2D binds into the "
                "single program region must share region_id; got "
                f"{rid_a_signal} (A.SignalBack) vs {rid_b_wait} (B.WaitBack)"
            )

            # Isolate this hit/skip sequence: the mock device address yields a
            # DETERMINISTIC region_id, so a slot published by a prior test would
            # otherwise be observed here.
            torch_spyre._C._test_clear_edge4_slots()
            torch_spyre._C.reset_edge4_slot_stats()
            assert torch_spyre._C.edge4_slot_hits() == 0
            assert torch_spyre._C.edge4_slot_skips() == 0

            # Launch i (plan A) SignalBack publishes a back-event under its
            # region_id.
            published = torch_spyre._C._test_edge4_signal_store(rid_a_signal)
            assert published != 0, "SignalBack must publish a non-null back-event"

            # Launch i+1 (plan B) WaitBack looks up under ITS region_id. Because
            # the key collapsed, it finds the SAME Event -> HIT.
            found = torch_spyre._C._test_edge4_wait_probe(rid_b_wait)
            assert found == published, (
                "WaitBack in launch i+1 must retrieve the SAME back-event "
                "published by SignalBack in launch i (rolling-slot pointer "
                "identity across the collapsed region_id key)"
            )
            assert torch_spyre._C.edge4_slot_hits() == 1, (
                "the cross-launch WaitBack must HIT the published slot (0 -> 1)"
            )
            assert torch_spyre._C.edge4_slot_skips() == 0, (
                "no skip when the slot is published"
            )

    def test_edge4_rolling_slot_disjoint_region_skips(self):
        """#2 negative: a WaitBack keyed on a region_id NOT in the map MISSES.

        getEdge4Slot returns nullptr (probe -> 0) and edge4_slot_skips
        increments. The map is non-empty (a back-event is published under the
        plan's real region_id), but a DISJOINT key still misses -- proving the
        slot is keyed by region_id, not "whatever was last published". A first
        launch (empty slot) takes this same no-op skip path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            plan = torch_spyre._C.prepare_kernel(spyrecode_dir)
            rid = plan.get_step_region_id(6)

            torch_spyre._C._test_clear_edge4_slots()
            torch_spyre._C.reset_edge4_slot_stats()

            published = torch_spyre._C._test_edge4_signal_store(rid)
            assert published != 0

            disjoint_rid = rid + 0x1000000
            assert disjoint_rid != rid
            missed = torch_spyre._C._test_edge4_wait_probe(disjoint_rid)
            assert missed == 0, (
                "a disjoint region_id must MISS (nullptr), not alias the published slot"
            )
            assert torch_spyre._C.edge4_slot_skips() == 1, (
                "a WaitBack on a region_id with no published back-event must SKIP"
            )
            assert torch_spyre._C.edge4_slot_hits() == 0, (
                "no hit on a disjoint region_id"
            )

    def test_project_real_plan_identity_valid_permuted_rejected(self):
        """#7a: project a REAL prepared plan through validate()'s exact path.

        validate() classifies each JobPlanStep (classifyStep) and reads its
        baked-in role() to build the (kind, role) sequence it checks. The
        check_job_plan_step_ordering binding takes NAME lists and bypasses that
        projection, so a wiring bug in the step -> (kind, role) mapping would slip
        past it. _test_project_and_check_ordering runs the REAL projection over
        REAL step objects in a caller-given index order:
          - identity order [0..6] must reproduce validate()'s acceptance ('');
          - a permuted order that puts the real WaitBack before the real
            HostCompute must be REJECTED by the same projection (the Placement
            Invariant enforced over real steps, not name lists).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            plan = torch_spyre._C.prepare_kernel(spyrecode_dir)
            assert plan.num_steps() == 7

            err_identity = torch_spyre._C._test_project_and_check_ordering(
                plan, [0, 1, 2, 3, 4, 5, 6]
            )
            assert err_identity == "", (
                "real-step projection in canonical order must be accepted, "
                f"matching validate(); got: {err_identity!r}"
            )

            err_permuted = torch_spyre._C._test_project_and_check_ordering(
                plan, [1, 0, 2, 3, 4, 5, 6]
            )
            assert err_permuted != "", (
                "projecting the real steps with WaitBack before HostCompute must "
                "be rejected by the Placement Invariant"
            )
            assert "HostCompute" in err_permuted


# The canonical instrumented two-stream projection (edit 6), as parallel
# (StepKind, StreamRole) name lists. This is what checkJobPlanStepOrdering must
# accept; the negative tests below mutate it to violate the Placement Invariant.
_VALID_KINDS = [
    "HostCompute",
    "WaitBack",
    "H2D",
    "SignalForward",
    "WaitForward",
    "Compute",
    "SignalBack",
]
_VALID_ROLES = ["Prep", "Prep", "Prep", "Prep", "Dev", "Dev", "Dev"]


class TestStepOrderingValidator:
    """Direct tests of the P2-14 two-stream step-ordering validator.

    Exercised through the check_job_plan_step_ordering binding, which calls the
    pure checker over projected (StepKind, StreamRole) sequences. This lets the
    Placement-Invariant NEGATIVE case be tested without constructing real steps
    (a real HostCompute needs a deeptools::Hcm plus pinned host buffers). The
    validator returns '' when valid, else a human-readable error string.
    """

    def test_valid_instrumented_ordering_accepted(self):
        """The canonical 7-step instrumented plan is accepted (returns '')."""
        err = torch_spyre._C.check_job_plan_step_ordering(_VALID_KINDS, _VALID_ROLES)
        assert err == "", f"expected valid ordering, got error: {err!r}"

    def test_placement_invariant_waitback_before_hostcompute_rejected(self):
        """NEGATIVE: WaitBack placed BEFORE HostCompute on S_prep is rejected.

        This is the core Placement Invariant: the edge-4 WaitBack must sit
        AFTER the HostCompute and BEFORE the H2D. Moving it to the front of the
        prep stream must be flagged (the prep stream must begin with
        HostCompute).
        """
        kinds = [
            "WaitBack",
            "HostCompute",
            "H2D",
            "SignalForward",
            "WaitForward",
            "Compute",
            "SignalBack",
        ]
        roles = list(_VALID_ROLES)  # WaitBack still on Prep, just mis-ordered
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "WaitBack-before-HostCompute must be rejected"
        assert "HostCompute" in err

    def test_placement_invariant_waitback_after_h2d_rejected(self):
        """NEGATIVE: WaitBack placed AFTER H2D on S_prep is rejected."""
        kinds = [
            "HostCompute",
            "H2D",
            "WaitBack",
            "SignalForward",
            "WaitForward",
            "Compute",
            "SignalBack",
        ]
        roles = list(_VALID_ROLES)
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "WaitBack-after-H2D must be rejected"

    def test_compute_on_prep_stream_rejected(self):
        """NEGATIVE: a device Compute mis-assigned to S_prep is rejected."""
        kinds = list(_VALID_KINDS)
        roles = ["Prep", "Prep", "Prep", "Prep", "Dev", "Prep", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "Compute on the prep stream must be rejected"

    def test_unmatched_forward_events_rejected(self):
        """NEGATIVE: a SignalForward with no matching WaitForward is rejected."""
        kinds = [
            "HostCompute",
            "WaitBack",
            "H2D",
            "SignalForward",
            "Compute",
            "SignalBack",
        ]
        roles = ["Prep", "Prep", "Prep", "Prep", "Dev", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "unmatched forward events must be rejected"

    def test_legacy_single_stream_plan_still_valid(self):
        """A legacy plan (no HostCompute, no events) stays unconditionally valid.

        Guards backward-compat: pure ComputeOnDevice, standalone D2H, and tensor
        .to() moves have neither a HostCompute nor an event step, so the checker
        must not impose the two-stream shape on them.
        """
        assert torch_spyre._C.check_job_plan_step_ordering(["Compute"], ["Dev"]) == ""
        assert torch_spyre._C.check_job_plan_step_ordering(["D2H"], ["Dev"]) == ""
        assert (
            torch_spyre._C.check_job_plan_step_ordering(
                ["H2D", "Compute"], ["Dev", "Dev"]
            )
            == ""
        )

    def test_hazard_tracker_bare_split_ordering_valid(self):
        """LOCK: a HostCompute-led, split, NO-events plan is valid.

        Under SPYRE_HAZARD_TRACKER the correction triple is split across
        S_prep/S_dev but carries NO event steps (flex inserts the cross-stream
        events dynamically). The ordering validator must accept it: has_event is
        false, so the forward-guard -- which requires a matched Ef pair for
        two-stream plans -- is skipped. This locks that property so a future
        validator change that would break hazard mode fails here loudly.
        Contrast test_legacy_single_stream_plan_still_valid (no HostCompute):
        this plan HAS a HostCompute but still no events.
        """
        assert (
            torch_spyre._C.check_job_plan_step_ordering(
                ["HostCompute", "H2D", "Compute"], ["Prep", "Prep", "Dev"]
            )
            == ""
        )

    def test_first_launch_waitback_noop_ordering_valid(self):
        """The instrumented plan is valid even though its WaitBack is unmatched.

        Back events are cross-launch: a single plan carries a WaitBack whose
        paired SignalBack is in a DIFFERENT launch (and vice-versa). The checker
        must NOT count-match them, so the standalone instrumented plan -- exactly
        what the first launch replays, where the WaitBack no-ops on an empty
        rolling slot -- is accepted.
        """
        assert (
            torch_spyre._C.check_job_plan_step_ordering(_VALID_KINDS, _VALID_ROLES)
            == ""
        )

    def test_back_events_without_forward_pair_rejected(self):
        """#6 NEGATIVE: a two-stream plan with events but NO forward Ef pair.

        Back events alone (WaitBack/SignalBack) already make this a two-stream
        plan (has_event), but with zero SignalForward/WaitForward the
        cross-stream H2D -> Compute RAW (edge-3) is left unserialized -- the
        device Compute would race the H2D that feeds it. The count-match check
        only catches ASYMMETRIC forward events (n_sig != n_wait); this
        zero-forward case (n_sig == n_wait == 0) must be caught by the dedicated
        forward-guard, whose message is uniquely "found none". This plan passes
        both per-stream walks and the count-match, so it reaches -- and must be
        rejected by -- that guard.
        """
        kinds = ["HostCompute", "WaitBack", "H2D", "Compute", "SignalBack"]
        roles = ["Prep", "Prep", "Prep", "Dev", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", (
            "a two-stream plan with back events but no forward Ef pair must be "
            "rejected (the H2D -> Compute RAW would be unserialized)"
        )
        assert "found none" in err, (
            f"expected the zero-forward forward-guard message, got: {err!r}"
        )

    def test_hostcompute_on_dev_stream_rejected(self):
        """#7b NEGATIVE: HostCompute mis-assigned to the device stream (S_dev).

        Placement Invariant: HostCompute belongs on S_prep. Flipping step 0's
        role to Dev leaves S_prep beginning with WaitBack, which the S_prep walk
        rejects (prep must begin with HostCompute).
        """
        roles = ["Dev", "Prep", "Prep", "Prep", "Dev", "Dev", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(_VALID_KINDS, roles)
        assert err != "", "HostCompute on the device stream must be rejected"

    def test_h2d_on_dev_stream_rejected(self):
        """#7b NEGATIVE: H2D mis-assigned to the device stream (S_dev).

        Placement Invariant: the correction H2D belongs on S_prep. Flipping H2D's
        role to Dev leaves S_prep with SignalForward where H2D must be (rejected
        by the S_prep walk) and puts a forbidden H2D on the device stream.
        """
        roles = ["Prep", "Prep", "Dev", "Prep", "Dev", "Dev", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(_VALID_KINDS, roles)
        assert err != "", "H2D on the device stream must be rejected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
