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

"""failure_* props on a verdict: kept only where the leg did not pass, and a reason that
arrives after the write-once verdict still lands, in artifact_result_reasons."""

import json
from pathlib import Path

import regex as re
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR
from spyre_clickhouse_ingest.results import _first_failure, _leg_failure
from spyre_clickhouse_ingest.schema import (
    ARTIFACT_RESULT_REASONS,
    ARTIFACT_RESULTS,
    FAILURE_REASON_VALUES,
    SchemaError,
)
from spyre_clickhouse_ingest.writer import ArtifactWriter

import pytest

from test_artifact_writer import FakeClient, _rows

BUNDLE_SCHEMA = (
    Path(__file__).parents[1] / "spyre_clickhouse_ingest" / "bundle.schema.json"
)
AID = "2b397099-6200-52fb-98c4-b603961a0582"
RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"
WHY = {"failure_reason": "infra_timeout", "failure_subreason": "card_lock"}


def _verdict(client, state, props):
    return ArtifactWriter.insert_result(
        client,
        "spyre_v2",
        artifact_id=AID,
        run_id=RUN,
        test_type="regression",
        state=state,
        arch="s390x",
        props=props,
    )


@pytest.mark.parametrize("state", ["passed", "running"])
def test_a_passing_verdict_drops_any_failure_props(state):
    client = FakeClient(counts=[0])
    _verdict(client, state, {**WHY, "failure_detail": "stale", "gating": "true"})
    (row,) = _rows(client, ARTIFACT_RESULTS)
    assert row["props"] == {"gating": "true"}


def test_a_failed_verdict_keeps_its_reason_and_caps_the_detail():
    client = FakeClient(counts=[0])
    _verdict(client, "error", {**WHY, "failure_detail": "x" * 900})
    (row,) = _rows(client, ARTIFACT_RESULTS)
    assert row["props"]["failure_reason"] == "infra_timeout"
    assert len(row["props"]["failure_detail"]) == ArtifactWriter.FAILURE_DETAIL_MAX


def test_an_unknown_reason_is_recorded_as_unknown_keeping_what_was_sent():
    props = ArtifactWriter.failure_props(
        "error", {"failure_reason": "infra_result_lost"}
    )
    assert props["failure_reason"] == "unknown"
    assert props["failure_subreason"] == "infra_result_lost"


def test_a_repeat_verdict_lands_its_reason_beside_the_first_one():
    client = FakeClient(counts=[1])
    _verdict(client, "error", {**WHY, "source": "jenkins", "run_url": "https://ci/1"})
    assert _rows(client, ARTIFACT_RESULTS) == []
    (reason,) = _rows(client, ARTIFACT_RESULT_REASONS)
    assert reason["failure_reason"] == "infra_timeout"
    assert reason["confidence"] == 3
    assert reason["source"] == "jenkins"
    assert reason["failure_log_url"] == "https://ci/1"


def test_a_repeat_without_a_reason_writes_nothing():
    client = FakeClient(counts=[1])
    _verdict(client, "failed", {"source": "jenkins"})
    assert client.inserts == []


def test_a_missing_reasons_table_costs_only_the_reason():
    class NoTable(FakeClient):
        def insert(self, table, rows, column_names=None, database=None):
            raise RuntimeError("UNKNOWN_TABLE")

    assert _verdict(NoTable(counts=[1]), "error", WHY) is True


def test_the_reasons_table_refuses_a_code_outside_the_taxonomy():
    row = dict.fromkeys(ARTIFACT_RESULT_REASONS.columns, "")
    row.update(result_kind="functional", test_type="unit", source="backfill")
    row["failure_reason"] = "flaky"
    with pytest.raises(SchemaError):
        ARTIFACT_RESULT_REASONS.row(row)
    assert "unknown" in FAILURE_REASON_VALUES


def test_cli_defaults_name_the_first_failing_case():
    acc = {"failed": 2, "total": 9}
    _first_failure(
        acc,
        [
            {"name": "test_ok", "status": "passed"},
            {
                "name": "test_add",
                "status": "failed",
                "fail_message": "AssertionError\n...",
            },
            {"name": "test_mul", "status": "error", "fail_message": "boom"},
        ],
    )
    assert _leg_failure("failed", acc, {}) == {
        "failure_reason": "test_failure",
        "failure_detail": "2 of 9 failed: test_add: AssertionError",
        "failure_confidence": "1",
    }
    caseless = {"failed": 0, "total": 0}
    assert _leg_failure("error", caseless, {})["failure_subreason"] == "no_cases"
    assert _leg_failure("passed", acc, {}) == {}


def test_any_caller_failure_prop_replaces_every_default():
    caseless = {"failed": 0, "total": 0}
    assert _leg_failure("error", caseless, {"failure_reason": "infra_hardware"}) == {}


@pytest.mark.parametrize("given,kept", [("12.5", "12"), (30, "30"), ("soon", None)])
def test_wait_is_whole_seconds_or_dropped(given, kept):
    props = ArtifactWriter.failure_props("error", {**WHY, "failure_wait_s": given})
    assert props.get("failure_wait_s") == kept


def test_a_cli_default_lands_late_at_its_own_confidence():
    client = FakeClient(counts=[1])
    _verdict(
        client, "error", {"failure_reason": "ingest_error", "failure_confidence": "1"}
    )
    (reason,) = _rows(client, ARTIFACT_RESULT_REASONS)
    assert reason["confidence"] == 1


def test_the_reason_vocabulary_matches_the_ddl_check_and_the_bundle_schema():
    ddl = (SCHEMA_DIR / "20-artifacts.sql").read_text()
    check = re.search(r"chk_failure_reason CHECK failure_reason IN\s*\(([^)]*)\)", ddl)
    assert set(re.findall(r"'([a-z_]+)'", check.group(1))) == FAILURE_REASON_VALUES
    perf = json.loads(BUNDLE_SCHEMA.read_text())["properties"]["perf"]["properties"]
    assert set(perf["failure_reason"]["enum"]) == FAILURE_REASON_VALUES


def test_first_failure_names_a_capability_case_and_skips_one_that_names_nothing():
    acc: dict = {}
    _first_failure(acc, [{"status": "failed", "properties": []}])
    assert "first_failure" not in acc
    op = [("capability.test_type", "model_ops"), ("capability.name", "aten.mm")]
    _first_failure(acc, [{"status": "error", "properties": op, "fail_message": "boom"}])
    assert acc["first_failure"] == "aten.mm: boom"
