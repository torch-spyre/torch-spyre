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

"""JUnit XML helpers and CI run-coordinate resolution shared by every ingest."""

import uuid


class JUnitXml:
    """Reads the parts of a JUnit document the ingests need."""

    @staticmethod
    def extract_properties(tc_el) -> list[tuple[str, str]]:
        """The testcase's (name, value) property pairs, blank names dropped."""
        props: list[tuple[str, str]] = []
        props_el = tc_el.find("properties")
        if props_el is None:
            return props
        for p in props_el.findall("property"):
            name = p.get("name", "").strip()
            value = p.get("value", "").strip()
            if name:
                props.append((name, value))
        return props

    @staticmethod
    def promote_xpass(raw_cases, suite_attrs) -> None:
        """Relabel bare cases as xpass for the suite's non-strict xpass failures."""
        failures = int(suite_attrs.get("failures", 0))
        true_fail_raw = sum(1 for c in raw_cases if c["status"] in ("failed", "error"))
        strict_xpass_raw = sum(1 for c in raw_cases if c["status"] == "xpass")
        non_strict = max(0, failures - true_fail_raw - strict_xpass_raw)

        promoted = 0
        for c in raw_cases:
            if promoted >= non_strict:
                break
            if c["_is_bare"]:
                c["status"] = "xpass"
                promoted += 1


class RunCoordinates:
    """The CI coordinates a leg's run_id is threaded from or derived from."""

    @staticmethod
    def threaded_run_id(args) -> str:
        """--run-id when it is a real UUID, else '' so the caller derives one."""
        raw = (getattr(args, "run_id", "") or "").strip()
        try:
            return str(uuid.UUID(raw))
        except (ValueError, AttributeError, TypeError):
            return ""

    @staticmethod
    def gha_run_id(args) -> str:
        """--gha-run-id when numeric, else '' -- non-numeric is not a GHA run."""
        raw = (getattr(args, "gha_run_id", "") or "").strip()
        try:
            int(raw)
            return raw
        except (ValueError, TypeError):
            return ""

    @classmethod
    def runner_run_id(cls, args, run_id: str) -> str:
        """This leg's own id: --gha-run-id when GHA-dispatched, else the run uuid."""
        return cls.gha_run_id(args) or run_id

    @classmethod
    def source_and_external(cls, args, run_id: str) -> tuple:
        """(source, external_run_id) for this leg, from whichever CI dispatched it."""
        gha = cls.gha_run_id(args)
        if gha:
            return "gha", gha
        jenkins_key = (getattr(args, "jenkins_run_key", "") or "").strip()
        if jenkins_key:
            return "jenkins", jenkins_key
        # No CI coordinate: rows stay joinable within this ingest, not to an artifact.
        return "local", run_id


# Function API, kept so installed consumers import one definition, not a copy.
extract_properties = JUnitXml.extract_properties
promote_xpass = JUnitXml.promote_xpass
_threaded_run_id = RunCoordinates.threaded_run_id
_runner_run_id = RunCoordinates.runner_run_id
source_and_external_run_id = RunCoordinates.source_and_external
