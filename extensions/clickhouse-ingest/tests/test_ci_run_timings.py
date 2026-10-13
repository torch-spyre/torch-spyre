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

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import regex as re
from spyre_clickhouse_ingest import ci_run_timings
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.schema import (
    TIMING_BUILD_STATE_VALUES,
    TIMING_EXECUTOR_VALUES,
    TIMING_TEST_STATE_VALUES,
    CiRunTimings,
    SchemaError,
)

DDL = Path(__file__).resolve().parents[1] / "schema" / "48-ci-run-timings.sql"
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def _ms(iso: str) -> int:
    """Epoch ms of a UTC time, the form the Jenkins side sends."""
    return int(
        datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000
    )


def _at(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def _batch(**over):
    batch = {
        "run": {
            "run_key": "Spyre/orchestrator#4242",
            "trigger_kind": "upstream",
            "trigger_source": "spyre-test",
            "preset": "trigger-pr-validation",
            "build_mode": "pr",
            "trigger_pr": "github.ibm.com/ai-chip-toolchain/deeptools#4500",
            "pr_components": ["github.com/torch-spyre/torch-spyre#3372"],
            "sha": "0123456789abcdef",
            "base_ref": "main",
            "build_url": "https://jenkins/job/orchestrator/4242/",
            "verdict": "green",
            "result": "SUCCESS",
            "superseded": "false",
            "pickup_path": "webhook",
            "comment_at": "2026-10-08T10:00:00Z",
            "picked_up_at": _ms("2026-10-08T10:00:05"),
            "scheduled_at": _ms("2026-10-08T10:00:06"),
            "started_at": _ms("2026-10-08T10:00:10"),
            "pr_queued_at": _ms("2026-10-08T10:00:30"),
            "pr_running_at": _ms("2026-10-08T10:02:00"),
            "ended_at": _ms("2026-10-08T11:30:00"),
        },
        "builds": [
            {
                "component": "deeptools",
                "artifact_name": "deeptools",
                "arch": "amd64",
                "id12": "aaaaaaaaaaaa",
                "kind": "rpm",
                "state": "built",
                "result": "SUCCESS",
                "url": "https://jenkins/job/component-build/1/",
                "agent": "build-x86-1",
                "queued_at": _ms("2026-10-08T10:02:10"),
                "started_at": _ms("2026-10-08T10:02:40"),
                "ended_at": _ms("2026-10-08T10:20:40"),
            },
            {
                "component": "flex",
                "artifact_name": "flex",
                "arch": "amd64",
                "kind": "rpm",
                "state": "dropped",
            },
        ],
        "tests": [
            {
                "component": "torch-spyre",
                "artifact_name": "torch-spyre-dev",
                "arch": "x86_64",
                "modes": "integration",
                "state": "passed",
                "result": "SUCCESS",
                "gating": True,
                "url": "https://jenkins/job/component-build/3/",
                "queued_at": _ms("2026-10-08T10:41:05"),
                "started_at": _ms("2026-10-08T10:41:20"),
                "ended_at": _ms("2026-10-08T11:25:00"),
                "exec": {
                    "executor": "gha-ephemeral",
                    "provision_started_at": _ms("2026-10-08T10:41:30"),
                    "provision_ended_at": _ms("2026-10-08T10:42:30"),
                    "dispatched_at": "2026-10-08T10:42:40.500Z",
                    "started_at": "2026-10-08T10:46:40Z",
                    "ended_at": "2026-10-08T11:20:00Z",
                    "runs": 1,
                    "jobs": 4,
                    "result": "success",
                    "urls": [
                        "https://github.com/torch-spyre/torch-spyre/actions/runs/1"
                    ],
                    "run_keys": ["gha:torch-spyre/torch-spyre/1#1"],
                },
            },
            {
                "component": "torch-spyre",
                "artifact_name": "torch-spyre-dev",
                "arch": "s390x",
                "modes": "smoke,integration",
                "result": "FAILURE",
                "gating": "unstable",
                "runner_died": "true",
                "failure_reason": "runner_lost",
                "failed_stage": "Test",
                "queued_at": _ms("2026-10-08T10:41:05"),
                "started_at": _ms("2026-10-08T10:41:06"),
                "ended_at": _ms("2026-10-08T11:29:00"),
                "exec": {
                    "executor": "jenkins-local",
                    "provision_started_at": _ms("2026-10-08T10:41:10"),
                    "provision_ended_at": _ms("2026-10-08T11:11:10"),
                    "started_at": _ms("2026-10-08T11:11:15"),
                    "ended_at": _ms("2026-10-08T11:28:00"),
                    "runs": 1,
                    "jobs": 1,
                    "result": "failure",
                    "cards": "0,1",
                },
            },
        ],
    }
    batch.update(over)
    return batch


def _rows(batch=None):
    return ci_run_timings.build_rows(batch or _batch(), now=NOW)


def _ddl_body():
    return (
        DDL.read_text()
        .split("CREATE TABLE IF NOT EXISTS ci_run_timings", 1)[1]
        .split("ENGINE", 1)[0]
    )


def _ddl_columns():
    """(name, rest-of-line) for every column the DDL declares, in order."""
    return [
        (line.split()[0], line)
        for line in _ddl_body().splitlines()
        if line.startswith("    ")
        and not line.startswith("     ")
        and line.split()
        and not line.lstrip().startswith(("--", "(", ")", "CONSTRAINT"))
    ]


# ── the model and the DDL ──


def test_model_columns_are_the_ddls_insertable_columns_in_order():
    skip = {"audit_uuid", "audit_timestamp"}
    insertable = [
        c
        for c, line in _ddl_columns()
        if " MATERIALIZED " not in line and c not in skip
    ]
    assert list(CiRunTimings.columns) == insertable


def test_every_span_is_materialized_and_keeps_a_null_end_null():
    spans = {c: line for c, line in _ddl_columns() if c.endswith("_ms")}
    assert len(spans) == 11
    for col, line in spans.items():
        assert "MATERIALIZED CAST(if(dateDiff('millisecond'," in line, col
        # greatest(0, NULL) is 0 in ClickHouse, which would average a missing time in as 0.
        assert "greatest" not in line, col


def test_partition_key_is_the_non_nullable_run_start():
    ddl = DDL.read_text()
    assert "PARTITION BY toYYYYMM(run_started_at)" in ddl
    (line,) = [line for c, line in _ddl_columns() if c == "run_started_at"]
    assert "Nullable" not in line
    assert "run_started_at" in CiRunTimings.required


def test_ddl_check_sets_match_the_model():
    ddl = DDL.read_text()

    def in_list(prefix):
        m = re.search(re.escape(prefix) + r" IN \(([^)]*)\)", ddl)
        return {v.strip().strip("'") for v in m.group(1).split(",")}

    assert in_list("entry") == {"build", "test"}
    assert in_list("entry = 'build' AND state") == TIMING_BUILD_STATE_VALUES
    assert in_list("entry = 'test' AND state") == TIMING_TEST_STATE_VALUES
    assert in_list("CHECK executor") == TIMING_EXECUTOR_VALUES


def test_order_by_keeps_every_leg_and_attempt():
    assert (
        "ORDER BY (run_key, entry, component, artifact_name, arch, leg, attempt)"
        in DDL.read_text()
    )


# ── one row per entry ──


def test_one_row_per_build_and_per_test_leg():
    rows = _rows()
    assert [(r["entry"], r["component"], r["arch"]) for r in rows] == [
        ("build", "deeptools", "x86_64"),
        ("build", "flex", "x86_64"),
        ("test", "torch-spyre", "x86_64"),
        ("test", "torch-spyre", "s390x"),
    ]
    for r in rows:
        CiRunTimings.row(r)


def test_run_fields_repeat_on_every_row():
    for r in _rows():
        assert r["run_key"] == "Spyre/orchestrator#4242"
        assert (r["repo"], r["pr_number"]) == ("deeptools", 4500)
        assert (r["trigger_kind"], r["build_mode"]) == ("upstream", "pr")
        assert (r["sha"], r["base_ref"]) == ("0123456789abcdef", "main")
        assert r["superseded"] is False
        assert r["run_started_at"] == _at("2026-10-08T10:00:10")
        assert r["pr_queued_at"] == _at("2026-10-08T10:00:30")
        assert r["comment_at"] == _at("2026-10-08T10:00:00")
        assert r["verdict"] == "green" and r["run_result"] == "success"
        assert r["updated_at"] == NOW


def test_a_build_entry_has_no_executor():
    r = _rows()[0]
    assert (r["state"], r["result"], r["kind"]) == ("built", "success", "rpm")
    assert r["agent"] == "build-x86-1"
    assert r["queued_at"] == _at("2026-10-08T10:02:10")
    assert r["test_modes"] == [] and r["attempt"] == 1
    assert r["executor"] == "" and r["exec_started_at"] is None
    assert r["exec_urls"] == [] and r["cards"] == []


def test_a_gha_leg_fills_the_exec_columns():
    r = _rows()[2]
    assert (r["executor"], r["state"], r["gating"]) == (
        "gha-ephemeral",
        "passed",
        "true",
    )
    assert r["provision_started_at"] == _at("2026-10-08T10:41:30")
    assert r["exec_dispatched_at"] == datetime(
        2026, 10, 8, 10, 42, 40, 500000, tzinfo=timezone.utc
    )
    assert r["exec_started_at"] == _at("2026-10-08T10:46:40")
    assert r["exec_ended_at"] == _at("2026-10-08T11:20:00")
    assert (r["exec_runs"], r["exec_jobs"], r["exec_result"]) == (1, 4, "success")
    assert r["exec_run_keys"] == ["gha:torch-spyre/torch-spyre/1#1"]


def test_a_jenkins_local_leg_fills_the_same_columns():
    r = _rows()[3]
    assert r["executor"] == "jenkins-local"
    # The card-lock wait is the provision phase, not test time.
    assert r["provision_ended_at"] == _at("2026-10-08T11:11:10")
    assert r["exec_dispatched_at"] is None
    assert r["exec_started_at"] == _at("2026-10-08T11:11:15")
    assert r["cards"] == ["0", "1"]
    assert r["test_modes"] == ["integration", "smoke"]
    assert r["state"] == "failed" and r["gating"] == "unstable"
    assert r["runner_died"] is True
    assert (r["failure_reason"], r["failed_stage"]) == ("runner_lost", "Test")


def test_pr_components_cover_the_trigger_and_test_with_companions():
    rows = {(r["entry"], r["component"]): r["is_pr_component"] for r in _rows()}
    assert rows[("build", "deeptools")] is True
    assert rows[("test", "torch-spyre")] is True
    assert rows[("build", "flex")] is False


def test_pr_components_may_be_bare_component_names():
    batch = _batch()
    batch["run"]["pr_components"] = "flex"
    rows = {r["component"]: r["is_pr_component"] for r in _rows(batch)}
    assert rows["flex"] is True and rows["torch-spyre"] is False


def test_a_non_pr_run_is_written_with_an_empty_trigger_pr():
    batch = _batch()
    batch["run"].update(
        trigger_pr="",
        pr_components=[],
        comment_at="",
        trigger_source="main-push",
        base_ref="",
    )
    rows = _rows(batch)
    assert len(rows) == 4
    for r in rows:
        assert (r["trigger_pr"], r["repo"], r["pr_number"]) == ("", "", 0)
        assert r["base_ref"] == ""
        assert r["is_pr_component"] is False
        assert r["comment_at"] is None
        CiRunTimings.row(r)


def test_a_main_push_names_its_repo_and_merged_pr_without_a_trigger_pr():
    batch = _batch()
    batch["run"].update(
        trigger_pr="", pr_components=[], trigger_source="main-push", base_ref=""
    )
    batch["run"].update(repo="torch-spyre", pr_number="5294")
    for r in _rows(batch):
        assert (r["trigger_pr"], r["repo"], r["pr_number"]) == ("", "torch-spyre", 5294)
        assert r["is_pr_component"] is False
        CiRunTimings.row(r)


def test_a_trigger_pr_outranks_a_given_repo_and_pr_number():
    batch = _batch()
    batch["run"].update(repo="other", pr_number="1")
    assert {(r["repo"], r["pr_number"]) for r in _rows(batch)} == {("deeptools", 4500)}


def test_a_retried_build_keeps_both_attempts():
    batch = _batch()
    first = dict(batch["builds"][0], state="failed", result="FAILURE")
    batch["builds"][0]["attempt"] = 2
    batch["builds"].insert(0, first)
    rows = [r for r in _rows(batch) if r["component"] == "deeptools"]
    assert [(r["attempt"], r["state"]) for r in rows] == [(1, "failed"), (2, "built")]


def test_entries_sharing_a_key_are_refused():
    batch = _batch()
    batch["builds"].append(dict(batch["builds"][0]))
    with pytest.raises(ValueError, match="repeat a key"):
        _rows(batch)


def test_legs_differing_only_in_modes_are_distinct():
    batch = _batch()
    batch["tests"].append(dict(batch["tests"][0], modes="smoke"))
    assert len(_rows(batch)) == 5


@pytest.mark.parametrize(
    "entry, e, state",
    [
        ("test", {"state": "passed"}, "passed"),
        ("test", {"state": "ERROR"}, "error"),
        ("test", {"state": "running", "result": "SUCCESS"}, "passed"),
        ("test", {"result": "UNSTABLE"}, "failed"),
        ("test", {"result": "ABORTED"}, "error"),
        ("test", {}, ""),
        ("build", {"state": "Reused"}, "reused"),
        ("build", {"result": "FAILURE"}, ""),
    ],
)
def test_entry_state(entry, e, state):
    assert ci_run_timings.entry_state(entry, e) == state


def test_a_never_started_workflow_reads_never_started():
    batch = _batch()
    batch["tests"][0]["exec"]["result"] = "__never_started__"
    assert _rows(batch)[2]["exec_result"] == "never_started"


# ── the DDL CHECKs, enforced before the server sees the row ──


@pytest.mark.parametrize(
    "entry, state",
    [("build", "passed"), ("test", "built"), ("test", "dropped"), ("build", "error")],
)
def test_a_state_of_the_other_entry_is_refused(entry, state):
    r = _rows()[0 if entry == "build" else 2]
    with pytest.raises(SchemaError, match="violates the DDL CHECK"):
        CiRunTimings.row(dict(r, state=state))


def test_an_unknown_executor_is_refused():
    with pytest.raises(SchemaError, match="executor"):
        CiRunTimings.row(dict(_rows()[2], executor="gha"))


# ── times ──


@pytest.mark.parametrize("value", [None, "", 0, "0"])
def test_unknown_times(value):
    assert ci_run_timings.ts(value) is None


def test_times_parse_from_millis_and_iso():
    expect = datetime(2026, 10, 8, 10, 0, 0, 250000, tzinfo=timezone.utc)
    assert ci_run_timings.ts(_ms("2026-10-08T10:00:00") + 250) == expect
    assert ci_run_timings.ts(str(_ms("2026-10-08T10:00:00") + 250)) == expect
    assert ci_run_timings.ts("2026-10-08T10:00:00.250Z") == expect
    assert ci_run_timings.ts("2026-10-08T12:00:00.250+02:00") == expect


@pytest.mark.parametrize(
    "trigger_pr, expect",
    [
        ("github.ibm.com/ai-chip-toolchain/flex#1457", ("flex", 1457)),
        ("github.com/torch-spyre/torch-spyre#3372", ("torch-spyre", 3372)),
        ("", ("", 0)),
        ("torch-spyre", ("", 0)),
    ],
)
def test_split_trigger_pr(trigger_pr, expect):
    assert ci_run_timings.split_trigger_pr(trigger_pr) == expect


# ── the batch contract ──


def test_unknown_keys_are_refused():
    batch = _batch()
    batch["run"]["comment_time"] = "2026-10-08T10:00:00Z"
    batch["tests"][0]["exec"]["first_job_start"] = "x"
    batch["nodes"] = []
    with pytest.raises(ValueError) as e:
        ci_run_timings.build_rows(batch)
    assert "run.comment_time" in str(e.value)
    assert "tests[0].exec.first_job_start" in str(e.value)
    assert "section 'nodes'" in str(e.value)


@pytest.mark.parametrize("key", ["run_key", "started_at"])
def test_a_batch_without_a_run_key_or_start_is_refused(key):
    batch = _batch()
    batch["run"][key] = ""
    with pytest.raises(ValueError, match=f"run.{key} \\(missing\\)"):
        ci_run_timings.build_rows(batch)


class FakeClient:
    def __init__(self):
        self.inserts = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows, column_names, database))


def test_write_batch_inserts_in_model_order_into_the_named_database():
    c = FakeClient()
    assert ci_run_timings.write_batch(c, "spyre_v2", _batch()) == 4
    table, rows, cols, db = c.inserts[0]
    assert (table, db) == ("ci_run_timings", "spyre_v2")
    assert cols == list(CiRunTimings.columns)
    assert not {"leg", "queue_ms", "exec_ms"} & set(cols)
    assert len(rows) == 4 and all(len(r) == len(cols) for r in rows)


def test_cli_dry_run_prints_one_row_per_entry(tmp_path, capsys):
    f = tmp_path / "batch.json"
    f.write_text(json.dumps(_batch()))
    cli(["ci-run-timings", "write", str(f), "--dry-run"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["entry"], r["component"]) for r in lines] == [
        ("build", "deeptools"),
        ("build", "flex"),
        ("test", "torch-spyre"),
        ("test", "torch-spyre"),
    ]
