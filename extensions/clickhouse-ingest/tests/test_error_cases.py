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

"""Collection errors and xdist internal errors each reach v2 as their own error case."""

from pathlib import Path

from spyre_clickhouse_ingest import error_case_names, insert_test_results, results

DATA = Path(__file__).parent / "data" / "junit"
RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"


class Recorder:
    def __init__(self):
        self.inserts = {}

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.setdefault(table, []).extend(
            dict(zip(column_names, r)) for r in rows
        )

    def query(self, sql, parameters=None):
        class R:
            result_rows = []

        return R()

    def command(self, sql, parameters=None):
        pass


def _ingest(xml: str):
    run, cases = results.parse_test_xml(DATA / xml)
    client = Recorder()
    written = insert_test_results(client, "db", "torch-spyre", RUN, cases, xml)
    return run, cases, written, client.inserts


def _errors(cases):
    return sorted((c["classname"], c["name"]) for c in cases if c["status"] == "error")


def test_each_shard_collection_error_is_its_own_case():
    run, cases, written, rows = _ingest("collection-errors.xml")
    assert _errors(cases) == [
        ("junit-test-quality-shard-0", "collection[collection-errors]"),
        ("junit-test-quality-shard-1", "collection[collection-errors]"),
        ("junit-test-upstream-distributed", "collection[collection-errors]"),
    ]
    assert (run["total_tests"], run["errors"], written) == (5, 3, 5)
    errored = [r for r in rows["test_case_runs"] if r["status"] == "error"]
    assert len({r["test_case_id"] for r in errored}) == 3
    assert all(r["fail_message"].startswith("collection failure") for r in errored)


def test_each_xdist_internal_error_is_its_own_case_not_a_rerun():
    run, cases, written, rows = _ingest("xdist-internal-errors.xml")
    assert _errors(cases) == [
        ("pytest", "internal[xdist-internal-errors]"),
        ("pytest", "internal[xdist-internal-errors]#2"),
        ("pytest", "internal[xdist-internal-errors]#3"),
    ]
    assert not any(k == "result.reruns" for c in cases for k, _ in c["properties"])
    assert (run["total_tests"], run["errors"], written) == (4, 3, 4)
    errored = [r for r in rows["test_case_runs"] if r["status"] == "error"]
    assert len({r["test_case_id"] for r in errored}) == 3
    assert "DeviceOpenFail" in errored[0]["fail_message"]


def test_the_case_ids_are_stable_across_ingests():
    first = _ingest("xdist-internal-errors.xml")[3]["test_case_runs"]
    again = _ingest("xdist-internal-errors.xml")[3]["test_case_runs"]
    assert [r["test_case_id"] for r in first] == [r["test_case_id"] for r in again]


def test_only_nameless_and_internal_errors_are_renamed():
    assert error_case_names(
        [
            ("T", "test_x", "error"),
            ("T", "", "failed"),
            ("pytest", "internal", "passed"),
            ("T", "test_x", "error"),
            ("", " ", "error"),
        ],
        "junit",
    ) == ["test_x", "", "internal", "test_x", "collection[junit]"]


def test_a_collection_error_leg_fails_as_a_test_failure_naming_the_case():
    _, cases = results.parse_test_xml(DATA / "collection-errors.xml")
    acc = {"failed": 3, "total": 5}
    results._first_failure(acc, cases)
    reason = results._leg_failure("failed", acc, {})
    assert reason["failure_reason"] == "test_failure"
    assert reason["failure_detail"] == (
        "3 of 5 failed: collection[collection-errors]: collection failure"
    )
