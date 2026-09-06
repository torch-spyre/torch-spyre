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

import base64
import copy
import hashlib
import json
import os
import tempfile

import pytest
import torch
import torch_spyre


@pytest.fixture(scope="module")
def initialize_runtime():
    """Initialize Spyre runtime before running tests."""
    # Initialize torch with spyre device to start runtime
    torch.zeros(1, device="spyre")
    yield
    # Runtime cleanup happens automatically


@pytest.fixture
def registry_key(request):
    """Derive a reproducible canonical key from the current test node."""
    digest = hashlib.sha256(request.node.nodeid.encode("utf-8")).digest()[:10]
    return base64.b32encode(digest).decode("ascii").lower()


def _registry_event_name(key):
    return f"spyre_kernel_v1_registry_test_{key}"


def test_kernel_provenance_registry_insert_and_duplicate(registry_key):
    event_name = _registry_event_name(registry_key)
    before = torch_spyre._C.kernel_provenance_registry_stats()

    assert torch_spyre._C.register_kernel_provenance(event_name, ["11", "12"])
    after_insert = torch_spyre._C.kernel_provenance_registry_stats()
    assert after_insert["entries"] == before["entries"] + 1

    assert torch_spyre._C.register_kernel_provenance(event_name, ["11", "12"])
    after_duplicate = torch_spyre._C.kernel_provenance_registry_stats()
    assert after_duplicate["entries"] == after_insert["entries"]
    assert after_duplicate["conflicts"] == after_insert["conflicts"]
    before_lookup = torch_spyre._C.kernel_provenance_registry_stats()
    assert torch_spyre._C.lookup_kernel_provenance(registry_key) == ["11", "12"]
    after_lookup = torch_spyre._C.kernel_provenance_registry_stats()
    assert after_lookup["hits"] == before_lookup["hits"] + 1


def test_kernel_provenance_registry_rejects_conflict_without_overwrite(registry_key):
    event_name = _registry_event_name(registry_key)
    assert torch_spyre._C.register_kernel_provenance(event_name, ["21"])
    before = torch_spyre._C.kernel_provenance_registry_stats()

    assert not torch_spyre._C.register_kernel_provenance(event_name, ["22"])

    after = torch_spyre._C.kernel_provenance_registry_stats()
    assert after["entries"] == before["entries"]
    assert after["conflicts"] == before["conflicts"] + 1
    assert torch_spyre._C.lookup_kernel_provenance(registry_key) == ["21"]


def test_kernel_provenance_registry_miss_and_unparseable_name(registry_key):
    before = torch_spyre._C.kernel_provenance_registry_stats()
    assert torch_spyre._C.lookup_kernel_provenance(registry_key) is None
    after_miss = torch_spyre._C.kernel_provenance_registry_stats()
    assert after_miss["misses"] == before["misses"] + 1

    assert not torch_spyre._C.register_kernel_provenance("not_an_event", ["31"])
    after_invalid = torch_spyre._C.kernel_provenance_registry_stats()
    assert after_invalid["entries"] == after_miss["entries"]
    assert after_invalid["conflicts"] == after_miss["conflicts"]


@pytest.mark.usefixtures("initialize_runtime")
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

    def test_profiler_name_overrides_spyrecode_name_and_adds_step_suffix(self):
        """Compiler provenance name identifies every device-compute step."""
        profiler_name = "spyre_kernel_v1_fused_mm_" + "a" * 16
        job_exec_plan = [
            {
                "command": "ComputeOnDevice",
                "properties": {
                    "job_bin_ptr": "120259084288",
                    "name": "legacy_name",
                },
            },
            {
                "command": "ComputeOnDevice",
                "properties": {"job_bin_ptr": "120259084288"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, job_exec_plan=job_exec_plan
            )

            job_plan = torch_spyre._C.prepare_kernel(
                spyrecode_dir,
                profiler_name=profiler_name,
            )

            assert job_plan.get_step_name(0) == f"{profiler_name}#0"
            assert job_plan.get_step_name(1) == f"{profiler_name}#1"

    def test_profiler_name_accepts_exact_aiupti_limit(self):
        """The finalized name may fill the AIUPTI buffer through byte 127."""
        suffix = "#0"
        profiler_name = "p" * (
            torch_spyre._C.AIUPTI_ACTIVITY_NAME_MAX_BYTES - len(suffix)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)

            job_plan = torch_spyre._C.prepare_kernel(
                spyrecode_dir,
                profiler_name=profiler_name,
            )

            assert job_plan.get_step_name(0) == profiler_name + suffix

    def test_profiler_name_rejects_final_name_over_aiupti_limit(self):
        """The C++ boundary checks the name after adding the step suffix."""
        suffix = "#0"
        profiler_name = "p" * (
            torch_spyre._C.AIUPTI_ACTIVITY_NAME_MAX_BYTES - len(suffix) + 1
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)

            with pytest.raises(
                RuntimeError,
                match="profiler-visible compute name exceeds AIUPTI limit",
            ):
                torch_spyre._C.prepare_kernel(
                    spyrecode_dir,
                    profiler_name=profiler_name,
                )

    def test_spyrecode_compute_name_is_preserved_without_profiler_name(self):
        """Existing named SpyreCode plans retain their current behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir,
                exec_properties={
                    "job_bin_ptr": "120259084288",
                    "name": "legacy_name",
                },
            )

            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.get_step_name(0) == "legacy_name"

    def test_overlong_spyrecode_compute_name_remains_backend_controlled(self):
        """A backend label over the provenance limit must not fail preparation."""
        backend_name = "b" * (torch_spyre._C.AIUPTI_ACTIVITY_NAME_MAX_BYTES + 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir,
                exec_properties={
                    "job_bin_ptr": "120259084288",
                    "name": backend_name,
                },
            )

            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.get_step_name(0) == backend_name

    def test_overlong_directory_compute_name_fallback_does_not_fail(self):
        """Graceful provenance fallback must not make preparation fatal."""
        prefix = "k" * torch_spyre._C.AIUPTI_ACTIVITY_NAME_MAX_BYTES
        with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)

            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            expected = os.path.join(
                os.path.basename(tmpdir),
                "spyreCodeDir",
                "bundle.mlir#0",
            )
            assert len(expected) > torch_spyre._C.AIUPTI_ACTIVITY_NAME_MAX_BYTES
            assert job_plan.get_step_name(0) == expected

    def test_directory_compute_name_fallback_is_preserved(self):
        """Older unnamed plans retain the directory-derived fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(tmpdir)

            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            expected = os.path.join(
                os.path.basename(tmpdir),
                "spyreCodeDir",
                "bundle.mlir#0",
            )
            assert job_plan.get_step_name(0) == expected

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

    def test_prepare_emits_bare_split_triple(self):
        """prepare emits the bare split triple, independent of the flag.

        PrepareKernel emits the plain [HostCompute, H2D, Compute] triple with
        NO cross-stream event steps, carrying its by-type roles [Prep, Prep,
        Dev] -- so the launch router splits it across S_prep/S_dev while flex
        inserts the cross-stream RAW/WAR edges dynamically at enqueue. There is
        no plan rewrite and no event-step emission (3 steps, never 7).

        SPYRE_HAZARD_TRACKER only affects launch-time routing (whether the
        S_prep/S_dev split engages), never the prepared plan's shape, so this
        holds regardless of the flag.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            # Must NOT raise: prepare emits the plain triple unconditionally
            # (the static-edge block and its region-count TORCH_CHECK are
            # gone), so nothing here depends on the program-region count.
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            # Bare split triple: 3 steps, NO event steps inserted.
            assert job_plan.num_steps() == 3
            assert [job_plan.get_step_type(i) for i in range(3)] == [
                "HostCompute",
                "H2D",
                "Compute",
            ]
            # Roles are assigned by step type in the ctors, so the bare triple
            # already carries the split roles -- the split is real.
            assert [job_plan.get_step_stream_role(i) for i in range(3)] == [
                "Prep",
                "Prep",
                "Dev",
            ]
            # pipeline_barrier stays True on EVERY step of the bare split:
            # overlap comes only from the S_prep/S_dev split + flex's dynamic
            # cross-stream events, never from relaxing a barrier.
            assert [job_plan.get_step_pipeline_barrier(i) for i in range(3)] == [
                True,
                True,
                True,
            ]

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
            # H2D lives at index 1 of the correction triple
            # ([HostCompute, H2D, Compute]).
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.get_step_type(1) == "H2D"
            assert job_plan.get_step_pipeline_barrier(1) is True, (
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
        """Every step of the correction triple keeps barrier=True.

        The two-stream PoC preserves STRICT per-stream FIFO for ALL ops,
        including HostCompute: overlap comes from the S_prep/S_dev stream split
        plus flex's dynamic cross-stream RAW/WAR edges, NOT from relaxing any
        op's pipeline_barrier. So no step in the correction triple
        ([HostCompute, H2D, Compute]) may opt out of the barrier -- in
        particular HostCompute must NOT carry the old barrier=False.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.num_steps() == 3

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

    def test_project_real_plan_identity_valid_permuted_rejected(self):
        """#7a: project a REAL prepared plan through validate()'s exact path.

        validate() classifies each JobPlanStep (classifyStep) and reads its
        baked-in role() to build the (kind, role) sequence it checks. The
        check_job_plan_step_ordering binding takes NAME lists and bypasses that
        projection, so a wiring bug in the step -> (kind, role) mapping would slip
        past it. _test_project_and_check_ordering runs the REAL projection over
        REAL step objects in a caller-given index order:
          - identity order [0, 1, 2] must reproduce validate()'s acceptance ('');
          - a permuted order that puts the real H2D before the real HostCompute
            must be REJECTED by the same projection (the prep stream must begin
            with HostCompute), enforced over real steps, not name lists.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spyrecode_dir = self.create_mock_spyrecode(
                tmpdir, exec_command="ComputeOnHost"
            )
            plan = torch_spyre._C.prepare_kernel(spyrecode_dir)
            assert plan.num_steps() == 3

            err_identity = torch_spyre._C._test_project_and_check_ordering(
                plan, [0, 1, 2]
            )
            assert err_identity == "", (
                "real-step projection in canonical order must be accepted, "
                f"matching validate(); got: {err_identity!r}"
            )

            err_permuted = torch_spyre._C._test_project_and_check_ordering(
                plan, [1, 0, 2]
            )
            assert err_permuted != "", (
                "projecting the real steps with H2D before HostCompute must "
                "be rejected (prep stream must begin with HostCompute)"
            )
            assert "HostCompute" in err_permuted


# The canonical correction triple, as parallel (StepKind, StreamRole) name
# lists: [HostCompute(Prep), H2D(Prep), Compute(Dev)]. This is what
# checkJobPlanStepOrdering must accept; the negative tests below mutate it to
# violate the per-stream role ordering.
_VALID_KINDS = ["HostCompute", "H2D", "Compute"]
_VALID_ROLES = ["Prep", "Prep", "Dev"]


class TestStepOrderingValidator:
    """Direct tests of the P2-14 two-stream step-ordering validator.

    Exercised through the check_job_plan_step_ordering binding, which calls the
    pure checker over projected (StepKind, StreamRole) sequences. This lets the
    role-ordering NEGATIVE cases be tested without constructing real steps (a
    real HostCompute needs a deeptools::Hcm plus pinned host buffers). The
    validator returns '' when valid, else a human-readable error string.
    """

    def test_valid_bare_triple_ordering_accepted(self):
        """The canonical correction triple is accepted (returns '')."""
        err = torch_spyre._C.check_job_plan_step_ordering(_VALID_KINDS, _VALID_ROLES)
        assert err == "", f"expected valid ordering, got error: {err!r}"

    def test_h2d_before_hostcompute_rejected(self):
        """NEGATIVE: H2D before HostCompute on S_prep is rejected.

        The prep stream must be exactly HostCompute -> H2D, so it must begin
        with HostCompute. Putting the H2D first is flagged.
        """
        kinds = ["H2D", "HostCompute", "Compute"]
        roles = ["Prep", "Prep", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "H2D-before-HostCompute must be rejected"
        assert "HostCompute" in err

    def test_missing_h2d_on_prep_rejected(self):
        """NEGATIVE: a HostCompute-led plan with no H2D on S_prep is rejected.

        The prep stream must BEGIN with HostCompute -> H2D; dropping the H2D
        leaves prep as just [HostCompute], which the S_prep walk rejects.
        """
        kinds = ["HostCompute", "Compute"]
        roles = ["Prep", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "HostCompute with no following H2D must be rejected"
        assert "H2D" in err

    def test_compute_on_prep_stream_rejected(self):
        """NEGATIVE: a device Compute mis-assigned to S_prep is rejected.

        Routing the Compute to Prep leaves prep as [HostCompute, H2D, Compute];
        Compute is a device op and is not permitted on the prep stream, which
        carries only HostCompute / H2D.
        """
        kinds = ["HostCompute", "H2D", "Compute"]
        roles = ["Prep", "Prep", "Prep"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "Compute on the prep stream must be rejected"

    def test_legacy_single_stream_plan_still_valid(self):
        """A legacy plan (no HostCompute) stays unconditionally valid.

        Guards backward-compat: pure ComputeOnDevice, standalone D2H, and tensor
        .to() moves have no HostCompute, so the checker must not impose the
        two-stream shape on them.
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
        """LOCK: the HostCompute-led, split, NO-events triple is valid.

        Under SPYRE_HAZARD_TRACKER the correction triple is split across
        S_prep/S_dev and carries NO event steps (flex inserts the cross-stream
        edges dynamically at enqueue). The ordering validator must accept the
        bare [HostCompute(Prep), H2D(Prep), Compute(Dev)] triple. This locks that
        property so a future validator change that would break hazard mode fails
        here loudly. Contrast test_legacy_single_stream_plan_still_valid (no
        HostCompute): this plan HAS a HostCompute but still no events.
        """
        assert (
            torch_spyre._C.check_job_plan_step_ordering(
                ["HostCompute", "H2D", "Compute"], ["Prep", "Prep", "Dev"]
            )
            == ""
        )

    def test_trailing_d2h_on_dev_accepted(self):
        """The contract is ordering-only, not an exact triple: S_dev carries
        Compute AND D2H (see StreamRole in job_plan.h), so a longer plan
        HostCompute -> H2D -> Compute -> D2H is valid.

        S_prep = [HostCompute, H2D], S_dev = [Compute, D2H]. This locks the
        relaxation so a future re-tightening to the bare triple fails here.
        """
        kinds = ["HostCompute", "H2D", "Compute", "D2H"]
        roles = ["Prep", "Prep", "Dev", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err == "", f"HostCompute -> H2D -> Compute -> D2H must be valid: {err!r}"

    def test_multiple_compute_on_dev_accepted(self):
        """More than one Compute on S_dev is valid (ordering-only contract).

        S_prep = [HostCompute, H2D], S_dev = [Compute, Compute]. The dev walk
        requires only that the stream BEGIN with Compute and carry nothing
        outside {Compute, D2H}.
        """
        kinds = ["HostCompute", "H2D", "Compute", "Compute"]
        roles = ["Prep", "Prep", "Dev", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err == "", f"a second Compute on S_dev must be valid: {err!r}"

    def test_d2h_on_prep_stream_rejected(self):
        """NEGATIVE: a D2H on S_prep is rejected.

        D2H is a device op and belongs on S_dev (see StreamRole in job_plan.h);
        role assignment never routes it to Prep, so a D2H on prep signals a
        role-assignment bug. Prep carries only HostCompute / H2D.
        """
        kinds = ["HostCompute", "H2D", "D2H"]
        roles = ["Prep", "Prep", "Prep"]
        err = torch_spyre._C.check_job_plan_step_ordering(kinds, roles)
        assert err != "", "D2H on the prep stream must be rejected"
        assert "prep stream" in err

    def test_hostcompute_on_dev_stream_rejected(self):
        """NEGATIVE: HostCompute mis-assigned to the device stream (S_dev).

        HostCompute belongs on S_prep. Flipping step 0's role to Dev leaves
        S_prep beginning with H2D, which the S_prep walk rejects (prep must
        begin with HostCompute).
        """
        roles = ["Dev", "Prep", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(_VALID_KINDS, roles)
        assert err != "", "HostCompute on the device stream must be rejected"
        assert "HostCompute" in err

    def test_h2d_on_dev_stream_rejected(self):
        """NEGATIVE: H2D mis-assigned to the device stream (S_dev).

        The correction H2D belongs on S_prep. Flipping H2D's role to Dev leaves
        S_prep as just [HostCompute] (rejected: expected H2D after HostCompute)
        and puts a forbidden H2D on the device stream.
        """
        roles = ["Prep", "Dev", "Dev"]
        err = torch_spyre._C.check_job_plan_step_ordering(_VALID_KINDS, roles)
        assert err != "", "H2D on the device stream must be rejected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
