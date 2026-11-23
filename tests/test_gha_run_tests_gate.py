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

"""Pins the `run-tests` skip conditions in torch_spyre_tests.yaml.

Fork PR runs read every repo/org variable as empty, so a clause that exempts fork PRs from a
variable-controlled skip lets them run the full suite while the variables say to skip it.
"""

from pathlib import Path

import yaml

_WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "torch_spyre_tests.yaml"
)


def _run_tests_if() -> str:
    with open(_WORKFLOW) as f:
        return yaml.safe_load(f)["jobs"]["run-tests"]["if"]


def test_run_tests_skips_are_variable_controlled():
    cond = _run_tests_if()
    for var in ("GHA_FULL_MERGE_QUEUE_TESTS", "GHA_TESTS_ON_AUTO_MERGE"):
        assert f"vars.{var}" in cond


def test_run_tests_has_no_fork_bypass():
    assert "head.repo.full_name" not in _run_tests_if()
