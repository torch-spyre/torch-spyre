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

"""The test-result write path across re-run attempts: a newer attempt must replace, not be refused."""

from spyre_clickhouse_ingest import (
    ArtifactWriter,
    cases_already_ingested,
    drop_older_case_attempts,
    insert_test_results,
)
from spyre_clickhouse_ingest.schema import TEST_CASE_RUNS

RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"


class FakeClient:
    def __init__(self, run_count=0):
        self.run_count = run_count
        self.inserts, self.queries, self.commands = [], [], []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows, column_names))

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters or {}))
        rows = [(self.run_count,)] if "count()" in sql else []

        class R:
            result_rows = rows

        return R()

    def command(self, sql, parameters=None):
        self.commands.append((sql, parameters or {}))


def _case(name="test_x", status="passed"):
    return {"classname": "T", "name": name, "status": status, "properties": []}


def _run_props(client):
    (_, rows, cols) = next(i for i in client.inserts if i[0] == TEST_CASE_RUNS.name)
    return [dict(zip(cols, r))["props"] for r in rows]


def test_attempt_is_stamped_on_each_outcome_row():
    c = FakeClient()
    insert_test_results(c, "db", "torch-spyre", RUN, [_case()], "a.xml", attempt=2)
    assert _run_props(c)[0]["run_attempt"] == "2"


def test_no_attempt_leaves_props_unchanged():
    c = FakeClient()
    insert_test_results(c, "db", "torch-spyre", RUN, [_case()], "a.xml")
    assert "run_attempt" not in _run_props(c)[0]


def test_dedup_with_an_attempt_counts_only_that_attempt_or_later():
    c = FakeClient(run_count=0)
    assert not cases_already_ingested(c, "db", RUN, "torch-spyre", "a.xml", attempt=2)
    sql, params = c.queries[-1]
    assert "toUInt32OrZero(props['run_attempt']) >= {attempt:UInt32}" in sql
    assert params["attempt"] == 2 and params["sf"] == "a.xml"


def test_older_attempts_are_deleted_for_that_file_only():
    c = FakeClient(run_count=5)
    drop_older_case_attempts(c, "db", RUN, "torch-spyre", "a.xml", 2)
    (sql, params) = c.commands[0]
    assert sql.startswith("DELETE FROM db.test_case_runs")
    assert "props['source_file'] = {sf:String}" in sql
    assert "< {attempt:UInt32}" in sql
    assert params == {
        "component": "torch-spyre",
        "run_id": RUN,
        "sf": "a.xml",
        "attempt": 2,
    }


def test_no_attempt_never_deletes():
    # Jenkins and manual re-ingests pass no attempt: first-write-wins, nothing removed.
    c = FakeClient()
    drop_older_case_attempts(c, "db", RUN, "torch-spyre", "a.xml", 0)
    assert c.commands == []


def test_a_drop_recounts_the_run_counters_the_insert_only_mv_cannot_subtract():
    c = FakeClient(run_count=5)
    drop_older_case_attempts(c, "db", RUN, "torch-spyre", "a.xml", 2)
    sqls = [sql for sql, _ in c.commands]
    assert sqls[1].startswith("DELETE FROM db.run_case_counters")
    assert sqls[2].startswith("INSERT INTO db.run_case_counters SELECT")
    assert "FROM db.test_case_runs" in sqls[2]
    # Recounted per run, not per file: the counters hold one row per run.
    assert "source_file" not in sqls[2]


def test_nothing_older_means_no_delete_and_no_recount():
    c = FakeClient(run_count=0)
    drop_older_case_attempts(c, "db", RUN, "torch-spyre", "a.xml", 2)
    assert c.commands == []


AID = "5f0e7d3c-2b1a-5c4d-8e9f-0a1b2c3d4e5f"
BASE = "6a1f8e4d-3c2b-5d5e-9fa0-1b2c3d4e5f60"


def _verdict(c, attempt):
    return ArtifactWriter.insert_gha_result(
        c,
        "db",
        artifact_id=AID,
        component="torch-spyre",
        arch="x86_64",
        run_id=RUN,
        test_type="regression",
        state="passed",
        base_artifact_id=AID,
        attempt=attempt,
    )


def test_a_rerun_verdict_replaces_the_earlier_attempts():
    c = FakeClient(run_count=0)
    assert _verdict(c, 2)
    (sql, params) = c.commands[0]
    assert sql.startswith("DELETE FROM db.artifact_results")
    assert "< {attempt:UInt32}" in sql and "state != 'running'" in sql
    assert params["attempt"] == 2
    (_, rows, cols) = next(i for i in c.inserts if i[0] == "artifact_results")
    assert dict(zip(cols, rows[0]))["props"]["run_attempt"] == "2"


def test_a_verdict_from_this_attempt_is_not_rewritten():
    c = FakeClient(run_count=1)
    assert _verdict(c, 2)
    assert c.commands == [] and not [i for i in c.inserts if i[0] == "artifact_results"]


def test_no_attempt_keeps_first_write_wins_for_verdicts():
    c = FakeClient(run_count=1)
    assert _verdict(c, 0)
    assert c.commands == []


def _declared(name, status="passed", backend="spyre", **extra):
    props = [
        ("tag", "platform__ppc64le"),
        ("capability.test_type", "model_ops"),
        ("capability.subject", "m-1"),
        ("capability.name", "torch.mul"),
        ("capability.sig.input_shapes", '["[1,2]"]'),
        ("capability.sig.input_dtypes", '["torch.bfloat16"]'),
        ("capability.backend", backend),
        ("capability.tag", "torch.mul.1"),
        *extra.items(),
    ]
    return {"classname": "T", "name": name, "status": status, "properties": props}


def _rows(client, table):
    (_, rows, cols) = next(i for i in client.inserts if i[0] == table.name)
    return [dict(zip(cols, r)) for r in rows]


def test_a_declared_capability_is_written_alongside_the_outcome():
    from spyre_clickhouse_ingest.identity import CapabilityId
    from spyre_clickhouse_ingest.schema import CAPABILITIES, CAPABILITY_RUNS

    c = FakeClient()
    cases = [
        _declared(
            "a", backend="cpu", **{"capability.prop.fallback_ops": "aten.mul.Tensor"}
        ),
        _declared("b", status="xfail"),
        _declared("c", status="error"),
        _declared("d", status="skipped"),
        _case("plain"),
    ]
    assert (
        insert_test_results(c, "db", "torch-spyre", RUN, cases, "a.xml", attempt=2) == 5
    )
    runs = {r["props"]["test_name"]: r for r in _rows(c, CAPABILITY_RUNS)}
    assert sorted(runs) == ["a", "b", "c"]
    assert (runs["a"]["status"], runs["a"]["backend"]) == ("passed", "cpu")
    assert runs["b"]["status"] == "not_implemented"
    assert runs["c"]["status"] == "failed"
    assert runs["a"]["arch"] == "ppc64le" and runs["a"]["test_type"] == "model_ops"
    assert runs["a"]["props"] == {
        "test_name": "a",
        "fallback_ops": "aten.mul.Tensor",
        "run_attempt": "2",
        "shard": "a.xml",
    }
    (ident,) = _rows(c, CAPABILITIES)
    assert ident["tags"] == ["torch.mul.1"]
    # Sig keys hash sorted, so the order a test records them in cannot mint a new id.
    assert ident["capability_id"] == CapabilityId.derive(
        "torch-spyre",
        "model_ops",
        "m-1",
        "torch.mul",
        {"input_dtypes": '["torch.bfloat16"]', "input_shapes": '["[1,2]"]'},
        ("input_dtypes", "input_shapes"),
    )
    # capability.* is routed, so it neither lands in props nor counts as ignored.
    assert not [k for p in _run_props(c) for k in p if k.startswith("capability.")]


def test_capability_properties_are_not_reported_as_ignored(capsys):
    insert_test_results(
        FakeClient(), "db", "torch-spyre", RUN, [_declared("a")], "a.xml"
    )
    assert "ignored" not in capsys.readouterr().err


def test_verdicts_already_landed_for_the_file_are_not_rewritten():
    from spyre_clickhouse_ingest.schema import CAPABILITY_RUNS

    c = FakeClient(run_count=1)
    insert_test_results(c, "db", "torch-spyre", RUN, [_declared("a")], "a.xml")
    assert not [i for i in c.inserts if i[0] == CAPABILITY_RUNS.name]
    sql, params = c.queries[-1]
    assert "props['shard'] = {shard:String}" in sql and params["shard"] == "a.xml"


def test_older_attempts_drop_the_files_capability_verdicts_too():
    c = FakeClient(run_count=5)
    drop_older_case_attempts(c, "db", RUN, "torch-spyre", "a.xml", 2)
    sql, params = c.commands[-1]
    assert sql.startswith("DELETE FROM db.capability_runs")
    assert "props['shard'] = {sf:String}" in sql and "< {attempt:UInt32}" in sql
    assert params["sf"] == "a.xml"


def test_every_batch_of_one_type_lands_when_the_file_is_new():
    from spyre_clickhouse_ingest.schema import CAPABILITY_RUNS

    class Landing(FakeClient):
        def insert(self, table, rows, column_names=None, database=None):
            super().insert(table, rows, column_names, database)
            if table == CAPABILITY_RUNS.name:
                self.run_count = 1  # the first batch is now visible to a dedup query

    other_sig = _declared("b")
    other_sig["properties"] = [
        p for p in other_sig["properties"] if p[0] != "capability.sig.input_dtypes"
    ]
    c = Landing()
    insert_test_results(
        c, "db", "torch-spyre", RUN, [_declared("a"), other_sig], "a.xml"
    )
    written = [r for t, rows, _ in c.inserts if t == CAPABILITY_RUNS.name for r in rows]
    assert len(written) == 2


def _without(case, key):
    case["properties"] = [p for p in case["properties"] if p[0] != key]
    return case


def test_an_incomplete_or_conflicting_declaration_is_skipped_and_counted(capsys):
    from spyre_clickhouse_ingest.schema import CAPABILITY_RUNS

    conflicting = _declared("e")
    conflicting["properties"].append(("capability.name", "torch.add"))
    blank = _without(_declared("c"), "capability.name")
    blank["properties"].append(("capability.name", " "))
    cases = [
        _without(_declared("a"), "capability.subject"),
        _without(_declared("b"), "capability.test_type"),
        blank,
        conflicting,
        _declared("ok"),
    ]
    c = FakeClient()
    # The outcomes still land; only the verdicts are withheld.
    assert insert_test_results(c, "db", "torch-spyre", RUN, cases, "a.xml") == 5
    assert [r["props"]["test_name"] for r in _rows(c, CAPABILITY_RUNS)] == ["ok"]
    err = capsys.readouterr().err
    assert "1 capability declaration(s) with no capability.subject" in err
    assert "1 capability declaration(s) with no capability.test_type" in err
    assert "1 capability declaration(s) with no capability.name" in err
    assert "1 capability declaration(s) with conflicting capability.name" in err


def test_unknown_capability_keys_are_reported_and_dropped(capsys):
    from spyre_clickhouse_ingest.schema import CAPABILITY_RUNS

    case = _declared("a", **{"capability.fallback_ops": "x", "capability.sig.": "y"})
    c = FakeClient()
    insert_test_results(c, "db", "torch-spyre", RUN, [case], "a.xml")
    (run,) = _rows(c, CAPABILITY_RUNS)
    assert "fallback_ops" not in run["props"]
    err = capsys.readouterr().err
    assert "unknown key capability.fallback_ops" in err
    assert "unknown key capability.sig." in err


def test_a_case_without_capability_properties_declares_nothing():
    from spyre_clickhouse_ingest import capability_declaration

    assert capability_declaration(_case("plain")) == (None, "")
    assert capability_declaration({"properties": None}) == (None, "")


def test_a_reingested_retry_replaces_rather_than_adds_verdicts():
    from spyre_clickhouse_ingest.schema import CAPABILITY_RUNS

    # Attempt 2 of a file whose attempt-1 verdicts landed: they are deleted, then rewritten.
    c = FakeClient(run_count=3)
    drop_older_case_attempts(c, "db", RUN, "torch-spyre", "a.xml", 2)
    assert any(s.startswith("DELETE FROM db.capability_runs") for s, _ in c.commands)
    c.run_count = 0
    insert_test_results(
        c, "db", "torch-spyre", RUN, [_declared("a")], "a.xml", attempt=2
    )
    (run,) = _rows(c, CAPABILITY_RUNS)
    assert run["props"]["run_attempt"] == "2"
