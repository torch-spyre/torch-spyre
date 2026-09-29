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
    c = FakeClient()
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
