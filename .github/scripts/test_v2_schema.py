"""Pins the insert layer's contract. The model only changes HOW a (column_names, row) pair is
built, so pair equality against the pre-refactor output is a complete correctness proof."""

import pytest
from v2_schema import (
    BENCHMARKS,
    BENCHMARK_RUNS,
    TEST_CASES,
    TEST_CASE_RUNS,
    TABLES,
    STATUS_VALUES,
    SchemaError,
    insert,
    insert_identities,
)


class FakeClient:
    def __init__(self, known=()):
        self.known = list(known)
        self.inserts = []

    def insert(self, table, rows, column_names=None):
        self.inserts.append((table, rows, column_names))

    def query(self, sql, parameters=None):
        asked = set(parameters["ids"])

        class R:
            result_rows = [(k,) for k in self.known if str(k) in asked]

        return R()


# ── column order is the pre-refactor order, exactly ─────────────────────────────────────


def test_column_order_matches_the_pre_refactor_lists():
    # These are the literal column_names lists the three scripts passed before the refactor.
    assert list(TEST_CASES.columns) == [
        "test_case_id",
        "component",
        "classname",
        "name",
        "tags",
    ]
    assert list(TEST_CASE_RUNS.columns) == [
        "run_id",
        "test_case_id",
        "component",
        "status",
        "duration_s",
        "fail_message",
    ]
    assert list(BENCHMARKS.columns) == ["benchmark_id", "name", "tags", "props"]
    assert list(BENCHMARK_RUNS.columns) == [
        "run_id",
        "benchmark_id",
        "backend",
        "measurements",
        "iterations",
        "props",
    ]


def test_row_is_ordered_by_columns_not_by_dict_insertion():
    # A dict built in a different order must still produce the DDL-ordered row; this is the
    # whole point of the model.
    scrambled = {
        "name": "test_y",
        "tags": ["a"],
        "component": "c",
        "classname": "k",
        "test_case_id": "u",
    }
    assert TEST_CASES.row(scrambled) == ["u", "c", "k", "test_y", ["a"]]


# ── the mistakes it now makes impossible ────────────────────────────────────────────────


def test_unknown_column_is_refused():
    with pytest.raises(SchemaError, match="no such column"):
        TEST_CASES.row(
            {
                "test_case_id": "u",
                "component": "c",
                "classname": "k",
                "name": "n",
                "tags": [],
                "run_id": "oops",
            }
        )


def test_missing_column_is_refused_not_silently_shifted():
    with pytest.raises(SchemaError, match="missing column"):
        TEST_CASES.row({"test_case_id": "u", "component": "c", "name": "n", "tags": []})


def test_v1_column_set_cannot_be_written_to_the_v2_table():
    # The live hazard: spyre.test_cases has 14 columns, spyre_v2.test_cases has 6, and the two
    # are told apart only by which connection is used. Naming a v1 column now fails loudly here
    # instead of reaching the server.
    with pytest.raises(SchemaError, match="no such column"):
        TEST_CASES.row(
            {
                "test_case_id": "u",
                "component": "c",
                "classname": "k",
                "name": "n",
                "tags": [],
                "op_name": "matmul",
                "dtype": "fp16",
            }
        )


def test_empty_required_column_is_refused():
    with pytest.raises(SchemaError, match="must be non-empty"):
        TEST_CASES.row(
            {
                "test_case_id": "u",
                "component": "",
                "classname": "k",
                "name": "n",
                "tags": [],
            }
        )


@pytest.mark.parametrize("status", sorted(STATUS_VALUES))
def test_every_ddl_allowed_status_is_accepted(status):
    row = TEST_CASE_RUNS.row(
        {
            "run_id": "r",
            "test_case_id": "t",
            "component": "c",
            "status": status,
            "duration_s": 1.0,
            "fail_message": "",
        }
    )
    assert row[3] == status


def test_status_outside_the_ddl_check_is_refused_before_the_server_sees_it():
    with pytest.raises(SchemaError, match="violates the DDL CHECK"):
        TEST_CASE_RUNS.row(
            {
                "run_id": "r",
                "test_case_id": "t",
                "component": "c",
                "status": "PASSED",
                "duration_s": 1.0,
                "fail_message": "",
            }
        )


# ── insert() ────────────────────────────────────────────────────────────────────────────


def test_insert_passes_column_names_and_ordered_rows():
    c = FakeClient()
    n = insert(
        c,
        TEST_CASE_RUNS,
        [
            {
                "run_id": "r",
                "test_case_id": "t",
                "component": "c",
                "status": "passed",
                "duration_s": 0.5,
                "fail_message": "",
            }
        ],
    )
    assert n == 1
    table, rows, cols = c.inserts[0]
    assert table == "test_case_runs"
    assert cols == list(TEST_CASE_RUNS.columns)
    assert rows == [["r", "t", "c", "passed", 0.5, ""]]


def test_insert_of_nothing_does_not_call_the_client():
    c = FakeClient()
    assert insert(c, TEST_CASES, []) == 0
    assert c.inserts == []


# ── identity dedup: the defect that lived in two repos and not the third ─────────────────


def test_identity_dedup_skips_rows_the_table_already_holds():
    c = FakeClient(known=["known-id"])
    n = insert_identities(
        c,
        TEST_CASES,
        {
            "known-id": {
                "test_case_id": "known-id",
                "component": "c",
                "classname": "k",
                "name": "a",
                "tags": [],
            },
            "new-id": {
                "test_case_id": "new-id",
                "component": "c",
                "classname": "k",
                "name": "b",
                "tags": [],
            },
        },
    )
    assert n == 1
    assert c.inserts[0][1] == [["new-id", "c", "k", "b", []]]


def test_identity_dedup_writes_nothing_when_all_are_known():
    c = FakeClient(known=["a", "b"])
    assert (
        insert_identities(
            c,
            TEST_CASES,
            {
                "a": {
                    "test_case_id": "a",
                    "component": "c",
                    "classname": "k",
                    "name": "1",
                    "tags": [],
                },
                "b": {
                    "test_case_id": "b",
                    "component": "c",
                    "classname": "k",
                    "name": "2",
                    "tags": [],
                },
            },
        )
        == 0
    )
    assert c.inserts == []


def test_identity_dedup_on_a_fact_table_is_a_programming_error():
    with pytest.raises(SchemaError, match="no identity column"):
        insert_identities(FakeClient(), TEST_CASE_RUNS, {"x": {}})


def test_fact_tables_declare_no_identity_and_dimensions_do():
    assert TEST_CASES.identity == "test_case_id"
    assert BENCHMARKS.identity == "benchmark_id"
    assert TEST_CASE_RUNS.identity is None
    assert BENCHMARK_RUNS.identity is None


def test_registry_covers_exactly_the_four_v2_tables():
    assert set(TABLES) == {
        "test_cases",
        "test_case_runs",
        "benchmarks",
        "benchmark_runs",
    }
