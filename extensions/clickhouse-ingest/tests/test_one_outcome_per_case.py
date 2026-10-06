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

"""A run holds one outcome per case, so run_case_counters (which sums rows) counts it once."""

from spyre_clickhouse_ingest import CaseId, insert_test_results
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR
from spyre_clickhouse_ingest.schema import TEST_CASE_RUNS

RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"
OTHER = "2b7191f9-e172-658f-bc74-2bf00c29be1d"
TCID = CaseId.derive("torch-spyre", "T", "test_x", [])


class HeldClient:
    """A client whose run already holds `held`: [(status, ran_in, run_attempt, source_file,
    [fail_message, prior_status, prior_message])]."""

    def __init__(self, held=()):
        self.held = [(*h, "", "", "")[:7] for h in held]
        self.inserts, self.commands = [], []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows, column_names))

    def query(self, sql, parameters=None):
        rows = []
        if "audit_uuid" in sql:
            rows = [
                (
                    TCID,
                    f"00000000-0000-7000-8000-00000000000{i}",
                    st,
                    0.0,
                    msg,
                    *rest,
                    ps,
                    pm,
                )
                for i, (st, *rest, msg, ps, pm) in enumerate(self.held)
            ]

        class R:
            result_rows = rows

        return R()

    def command(self, sql, parameters=None):
        self.commands.append((sql, parameters or {}))


def _case(status, name="test_x"):
    return {"classname": "T", "name": name, "status": status, "properties": []}


def _write(c, *statuses, source_file="b.xml", attempt=0):
    cases = [_case(s) for s in statuses]
    return insert_test_results(
        c, "db", "torch-spyre", RUN, cases, source_file, attempt=attempt
    )


def _written(c):
    return [
        dict(zip(cols, r))["status"]
        for t, rows, cols in c.inserts
        if t == TEST_CASE_RUNS.name
        for r in rows
    ]


def _deleted(c):
    return [p["uuids"] for sql, p in c.commands if "audit_uuid IN" in sql]


def test_a_newer_attempt_of_the_file_replaces_the_older_and_recounts():
    c = HeldClient([("failed", RUN, "1", "b.xml")])
    _write(c, "passed", attempt=2)
    assert _written(c) == ["passed"]
    assert _deleted(c) == [["00000000-0000-7000-8000-000000000000"]]
    assert any(
        sql.startswith("INSERT INTO db.run_case_counters") for sql, _ in c.commands
    )


def _written_props(c):
    return [
        dict(zip(cols, r))["props"]
        for t, rows, cols in c.inserts
        if t == TEST_CASE_RUNS.name
        for r in rows
    ]


def test_a_newer_attempt_keeps_the_failure_it_replaces_as_prior_status():
    c = HeldClient([("failed", RUN, "1", "b.xml", "AssertionError: mismatch")])
    _write(c, "passed", attempt=2)
    props = _written_props(c)[0]
    assert props["result.prior_status"] == "failed"
    assert props["result.prior_message"] == "AssertionError: mismatch"


def test_a_prior_mark_on_the_replaced_attempt_carries_forward():
    c = HeldClient([("passed", RUN, "1", "b.xml", "", "error", "card fault")])
    _write(c, "passed", attempt=2)
    assert _written_props(c)[0]["result.prior_status"] == "error"


def test_a_replaced_pass_or_another_files_failure_leaves_no_prior_status():
    c = HeldClient([("passed", RUN, "1", "b.xml")])
    _write(c, "passed", attempt=2)
    assert "result.prior_status" not in _written_props(c)[0]
    c = HeldClient([("failed", OTHER, "1", "b.xml", "reused")])
    _write(c, "passed", attempt=2)
    assert "result.prior_status" not in _written_props(c)[0]


def test_a_rerun_re_ingesting_an_unchanged_failure_records_no_prior_status():
    # Attempt 2 re-reads every report of the run, including ones it did not re-run.
    c = HeldClient([("failed", RUN, "1", "b.xml")])
    _write(c, "failed", attempt=2)
    assert _written(c) == ["failed"] and _deleted(c)
    assert "result.prior_status" not in _written_props(c)[0]


def test_an_older_attempt_arriving_late_is_not_written():
    c = HeldClient([("passed", RUN, "2", "b.xml")])
    assert _write(c, "failed", attempt=1) == 0
    assert c.commands == []


def test_an_exact_copy_is_not_written_again():
    c = HeldClient([("passed", RUN, "", "b.xml")])
    assert _write(c, "passed") == 0
    assert c.commands == []


def test_differing_outcomes_in_one_attempt_of_a_file_both_stay():
    # test_T_spyre / test_t_spyre share an id until ids keep the name's case.
    c = HeldClient([("xfail", RUN, "", "b.xml")])
    _write(c, "passed")
    assert _written(c) == ["passed"] and _deleted(c) == []


def test_executed_outcomes_from_different_files_both_stay():
    # hf-adapters' base and _adapter configs run one entry name with different parameters.
    c = HeldClient([("passed", RUN, "", "a.xml")])
    _write(c, "failed")
    assert _written(c) == ["failed"] and _deleted(c) == []


def test_an_executed_outcome_replaces_a_skip_from_another_file():
    c = HeldClient([("skipped", RUN, "", "a.xml")])
    _write(c, "passed")
    assert _written(c) == ["passed"] and _deleted(c)
    c = HeldClient([("passed", RUN, "", "a.xml")])
    assert _write(c, "skipped") == 0


def test_an_executed_outcome_replaces_a_reused_copy_even_in_the_same_file():
    c = HeldClient([("failed", OTHER, "", "b.xml")])
    _write(c, "passed")
    assert _written(c) == ["passed"] and _deleted(c)


def test_one_batch_keeps_differing_repeats_and_drops_exact_ones():
    c = HeldClient()
    _write(c, "xfail", "passed", "passed")
    assert _written(c) == ["xfail", "passed"] and c.commands == []


def test_migration_009_is_rerunnable():
    sql = (SCHEMA_DIR / "migrations" / "009_one_outcome_per_case.sql").read_text()
    assert sql.startswith("-- RERUNNABLE")
    # The writer compares durations to the ms, so the migration must too.
    assert "round(duration_s, 3)" in sql
