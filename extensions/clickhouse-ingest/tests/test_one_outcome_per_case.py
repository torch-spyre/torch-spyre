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

import re

from spyre_clickhouse_ingest import CaseId, TestResultWriter, insert_test_results
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR
from spyre_clickhouse_ingest.schema import TEST_CASE_RUNS

RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"
OTHER = "2b7191f9-e172-658f-bc74-2bf00c29be1d"
TCID = CaseId.derive("torch-spyre", "T", "test_x", [])


class HeldClient:
    """A client whose run already holds `held`: [(status, ran_in, run_attempt), ...]."""

    def __init__(self, held=()):
        self.held = list(held)
        self.inserts, self.commands = [], []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows, column_names))

    def query(self, sql, parameters=None):
        rows = []
        if "audit_uuid" in sql:
            rows = [
                (TCID, f"00000000-0000-7000-8000-00000000000{i}", st, ran_in, att)
                for i, (st, ran_in, att) in enumerate(self.held)
            ]

        class R:
            result_rows = rows

        return R()

    def command(self, sql, parameters=None):
        self.commands.append((sql, parameters or {}))


def _case(status, name="test_x"):
    return {"classname": "T", "name": name, "status": status, "properties": []}


def _written(c):
    return [
        dict(zip(cols, r))["status"]
        for t, rows, cols in c.inserts
        if t == TEST_CASE_RUNS.name
        for r in rows
    ]


def _deleted(c):
    return [p["uuids"] for sql, p in c.commands if "audit_uuid IN" in sql]


def test_a_worse_outcome_replaces_the_held_one_and_recounts():
    c = HeldClient([("passed", RUN, "")])
    insert_test_results(c, "db", "torch-spyre", RUN, [_case("failed")], "b.xml")
    assert _written(c) == ["failed"]
    assert _deleted(c) == [["00000000-0000-7000-8000-000000000000"]]
    assert any(
        sql.startswith("INSERT INTO db.run_case_counters") for sql, _ in c.commands
    )


def test_a_better_or_equal_outcome_is_not_written_again():
    for held in ("failed", "passed"):
        c = HeldClient([(held, RUN, "")])
        assert (
            insert_test_results(c, "db", "torch-spyre", RUN, [_case("passed")], "b.xml")
            == 0
        )
        assert c.commands == []


def test_an_executed_outcome_replaces_a_reused_copy_even_when_better():
    c = HeldClient([("failed", OTHER, "")])
    insert_test_results(c, "db", "torch-spyre", RUN, [_case("passed")], "b.xml")
    assert _written(c) == ["passed"] and _deleted(c)


def test_a_later_attempt_replaces_a_worse_earlier_one():
    c = HeldClient([("failed", RUN, "1")])
    insert_test_results(
        c, "db", "torch-spyre", RUN, [_case("passed")], "b.xml", attempt=2
    )
    assert _written(c) == ["passed"] and _deleted(c)


def test_a_case_twice_in_one_batch_is_written_once_worst_first():
    c = HeldClient()
    insert_test_results(
        c, "db", "torch-spyre", RUN, [_case("passed"), _case("error"), _case("skipped")]
    )
    assert _written(c) == ["error"] and c.commands == []


def test_migration_007_ranks_statuses_as_the_writer_does():
    sql = (SCHEMA_DIR / "migrations" / "007_one_outcome_per_case.sql").read_text()
    names, ranks = re.search(r"\[('skipped'[^\]]*)\],\s*\[([^\]]*)\]", sql).groups()
    order = dict(zip(re.findall(r"'(\w+)'", names), map(int, ranks.split(","))))
    assert order == TestResultWriter.SEVERITY
    assert sql.startswith("-- RERUNNABLE")
