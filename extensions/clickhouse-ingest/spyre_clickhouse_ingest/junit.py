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


def extract_properties(tc_el):
    props = []
    props_el = tc_el.find("properties")
    if props_el is None:
        return props
    for p in props_el.findall("property"):
        name = p.get("name", "").strip()
        value = p.get("value", "").strip()
        if name:
            props.append((name, value))
    return props


def promote_xpass(raw_cases, suite_attrs):
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


def _threaded_run_id(args) -> str:
    """--run-id when it is a real UUID, else "" so the caller mints one.

    The flag has always carried a Jenkins BUILD_NUMBER historically, which is not a UUID and
    must not land in test_runs.run_id (a UUID column). Only a well-formed uuid is honoured.
    """
    raw = (getattr(args, "run_id", "") or "").strip()
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# ── Main ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def _runner_run_id(args, run_id: str) -> str:
    """This leg's own run id: --gha-run-id when GHA-dispatched, else the same uuid as run_id."""
    raw = (getattr(args, "gha_run_id", "") or "").strip()
    if raw:
        try:
            int(raw)
            return raw
        except (ValueError, TypeError):
            pass
    return run_id


def source_and_external_run_id(args, run_id: str):
    """(source, external_run_id) for this leg, from whichever CI dispatched it.

    A numeric --gha-run-id means GHA dispatched it. Otherwise the leg is
    Jenkins-dispatched and its own externalizable id ('folder/job#123') is the run
    coordinate -- the SAME value the orchestrator hashes on its side of the join, so
    neither side has to thread a minted uuid.
    `source` is required precisely because a GHA run id and a Jenkins build number
    share a number space.

    Only reached when no THREADED uuid was supplied -- see run_id_for(), which prefers
    --run-id and leaves this as the coordinate-hashing fallback.
    """
    gha = (getattr(args, "gha_run_id", "") or "").strip()
    if gha:
        try:
            int(gha)
            return "gha", gha
        except (ValueError, TypeError):
            pass
    jenkins_key = (getattr(args, "jenkins_run_key", "") or "").strip()
    if jenkins_key:
        return "jenkins", jenkins_key
    # No CI coordinate at all: fall back to the run uuid so the rows are still
    # self-consistent and joinable WITHIN this ingest, just not to an artifact.
    return "local", run_id
