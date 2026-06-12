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

import os
import tempfile
import json
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

    def create_mock_spyrecode(self, tmpdir):
        """Create a mock SpyreCode directory structure for testing.

        Args:
            tmpdir: Temporary directory path

        Returns:
            Path to the SpyreCode directory
        """
        spyrecode_dir = os.path.join(tmpdir, "spyreCodeDir")
        os.makedirs(spyrecode_dir, exist_ok=True)

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
            "JobExecPlan": [
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                }
            ],
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
