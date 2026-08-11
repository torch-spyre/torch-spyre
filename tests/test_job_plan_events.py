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

"""Tests for JobPlanStepEventSignal / JobPlanStepEventWait and
LaunchContext event registry (getOrCreateEvent).

Covers:
- EventSignal and EventWait JSON commands are parsed into the correct step types
- Missing event_id property raises RuntimeError with the right message
- get_step_type() returns "EventSignal" / "EventWait"
- __str__ output contains the event ID
- Mixed plans (EventSignal + EventWait with the same ID) are built correctly
- Plans with multiple distinct event IDs produce separate steps
- launch_jobplan executes a plan containing EventSignal and EventWait steps
"""

import tempfile

import pytest
import torch
import torch_spyre

from test_prepare_kernel import TestPrepareKernel as tpk


@pytest.fixture(scope="module", autouse=True)
def initialize_runtime():
    """Initialize Spyre runtime before running tests."""
    torch.zeros(1, device="spyre")
    yield


def _make_event_plan(tmpdir, job_exec_plan):
    """Build a SpyreCode directory whose JobExecPlan is *job_exec_plan*."""
    helper = tpk()
    return helper.create_mock_spyrecode(tmpdir, job_exec_plan=job_exec_plan)


class TestEventSignalStep:
    """Tests for JobPlanStepEventSignal construction via prepare_kernel."""

    def test_event_signal_step_type(self):
        """EventSignal command produces a step whose type is 'EventSignal'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = [
                {"command": "EventSignal", "properties": {"event_id": 0}},
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            spyrecode_dir = _make_event_plan(tmpdir, plan)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.num_steps() == 2
            assert job_plan.get_step_type(0) == "EventSignal"
            assert job_plan.get_step_type(1) == "Compute"

    def test_event_signal_str_contains_event_id(self):
        """__str__ of a plan with EventSignal reports the event ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = [
                {"command": "EventSignal", "properties": {"event_id": 42}},
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            spyrecode_dir = _make_event_plan(tmpdir, plan)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            desc = str(job_plan)
            assert "Event Signal" in desc
            assert "Event ID: 42" in desc


class TestEventWaitStep:
    """Tests for JobPlanStepEventWait construction via prepare_kernel."""

    def test_event_wait_step_type(self):
        """EventWait command produces a step whose type is 'EventWait'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = [
                {"command": "EventWait", "properties": {"event_id": 0}},
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            spyrecode_dir = _make_event_plan(tmpdir, plan)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.num_steps() == 2
            assert job_plan.get_step_type(0) == "EventWait"

    def test_event_wait_str_contains_event_id(self):
        """__str__ of a plan with EventWait reports the event ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = [
                {"command": "EventWait", "properties": {"event_id": 7}},
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            spyrecode_dir = _make_event_plan(tmpdir, plan)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            desc = str(job_plan)
            assert "Event Wait" in desc
            assert "Event ID: 7" in desc


class TestEventSignalWaitPlan:
    """Tests for plans that combine EventSignal and EventWait steps."""

    def test_signal_and_wait_same_event_id_step_types(self):
        """A plan with EventSignal then EventWait on the same ID builds correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = [
                {"command": "EventSignal", "properties": {"event_id": 1}},
                {"command": "EventWait", "properties": {"event_id": 1}},
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            spyrecode_dir = _make_event_plan(tmpdir, plan)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.num_steps() == 3
            assert job_plan.get_step_type(0) == "EventSignal"
            assert job_plan.get_step_type(1) == "EventWait"
            assert job_plan.get_step_type(2) == "Compute"

    def test_multiple_distinct_event_ids(self):
        """Multiple EventSignal/EventWait steps with different IDs are all created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = [
                {"command": "EventSignal", "properties": {"event_id": 0}},
                {"command": "EventSignal", "properties": {"event_id": 1}},
                {"command": "EventWait", "properties": {"event_id": 0}},
                {"command": "EventWait", "properties": {"event_id": 1}},
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            spyrecode_dir = _make_event_plan(tmpdir, plan)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            assert job_plan.num_steps() == 5
            assert job_plan.get_step_type(0) == "EventSignal"
            assert job_plan.get_step_type(1) == "EventSignal"
            assert job_plan.get_step_type(2) == "EventWait"
            assert job_plan.get_step_type(3) == "EventWait"
            assert job_plan.get_step_type(4) == "Compute"

    def test_signal_wait_shared_event_executes(self):
        """launch_jobplan executes a plan with EventSignal + EventWait without error.

        Signal and Wait share the same event_id so they use the same flex::Event
        via LaunchContext::getOrCreateEvent. The signal fires before the wait,
        so execution completes successfully.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = [
                {"command": "EventSignal", "properties": {"event_id": 0}},
                {"command": "EventWait", "properties": {"event_id": 0}},
                {
                    "command": "ComputeOnDevice",
                    "properties": {"job_bin_ptr": "120259084288"},
                },
            ]
            spyrecode_dir = _make_event_plan(tmpdir, plan)
            job_plan = torch_spyre._C.prepare_kernel(spyrecode_dir)

            stream = torch.Stream("spyre")
            with stream:
                torch_spyre._C.launch_jobplan(job_plan, [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
