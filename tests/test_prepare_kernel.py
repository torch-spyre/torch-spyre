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
        """Test that a valid ComputeOnHost command builds successfully.

        Verifies that a well-formed ComputeOnHost entry with all required
        fields (ohandle, size, ishape, ihandle, hcm) successfully builds
        a JobPlan.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )

            # Should succeed without exceptions
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            # Verify JobPlan was created
            assert job_plan is not None
            assert isinstance(job_plan, torch_spyre._C.JobPlan)

            # Verify it has 3 steps (HostCompute, H2D, Compute)
            assert job_plan.num_steps() == 3

            # Verify the step types
            assert job_plan.get_step_type(0) == "HostCompute"
            assert job_plan.get_step_type(1) == "H2D"
            assert job_plan.get_step_type(2) == "Compute"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
