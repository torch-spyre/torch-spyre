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

"""Pins the v2 write gate, including the column diff.

The gate exists because a v2 table can be present but stale: prod once had a `benchmarks`
without `component`, which leads the identity hash, and an existence-only check passed it -- the
gap then surfaced as an opaque arity error inside the insert. These tests pin that the column
diff runs BY DEFAULT, since a caller who must opt in is a caller who will forget.
"""

import pytest
from spyre_clickhouse_ingest import schema
from spyre_clickhouse_ingest.client import tables_present

FUNCTIONAL = (schema.TEST_CASES, schema.TEST_CASE_RUNS)


class FakeClient:
    """Answers EXISTS from `present` and system.columns from `columns`."""

    def __init__(self, present, columns=None):
        self.present = set(present)
        self.columns = columns or {}
        self.column_queries = 0

    def command(self, sql):
        return any(f"{t}" in sql for t in self.present)

    def query(self, sql, parameters=None):
        self.column_queries += 1
        table = (parameters or {})["t"]
        cols = self.columns.get(table, [])

        class R:
            result_rows = [(c,) for c in cols]

        return R()


def _full(tables):
    return {t.name: list(t.columns) for t in tables}


def test_absent_table_fails_the_gate():
    client = FakeClient(present=["test_cases"], columns=_full(FUNCTIONAL))
    assert tables_present(client, "db", tables=FUNCTIONAL) is False


def test_complete_tables_pass():
    names = [t.name for t in FUNCTIONAL]
    client = FakeClient(present=names, columns=_full(FUNCTIONAL))
    assert tables_present(client, "db", tables=FUNCTIONAL) is True


def test_column_diff_runs_by_default():
    """The prod incident: table exists, one hashed column missing."""
    names = [t.name for t in FUNCTIONAL]
    cols = _full(FUNCTIONAL)
    cols["test_cases"].remove("component")
    client = FakeClient(present=names, columns=cols)
    # No check_columns= passed: the default must still catch it.
    assert tables_present(client, "db", tables=FUNCTIONAL) is False
    assert client.column_queries > 0


def test_column_diff_can_be_declined():
    """Opting out skips the round trip, and then a stale table passes."""
    names = [t.name for t in FUNCTIONAL]
    cols = _full(FUNCTIONAL)
    cols["test_cases"].remove("component")
    client = FakeClient(present=names, columns=cols)
    assert tables_present(client, "db", tables=FUNCTIONAL, check_columns=False) is True
    assert client.column_queries == 0


def test_extra_live_columns_are_not_a_failure():
    """The model is a floor, not an exact match: a newer server may carry more."""
    names = [t.name for t in FUNCTIONAL]
    cols = _full(FUNCTIONAL)
    cols["test_case_runs"].append("some_future_column")
    client = FakeClient(present=names, columns=cols)
    assert tables_present(client, "db", tables=FUNCTIONAL) is True


@pytest.mark.parametrize("missing", ["benchmarks", "benchmark_runs"])
def test_benchmark_pair_is_gated_the_same_way(missing):
    pair = (schema.BENCHMARKS, schema.BENCHMARK_RUNS)
    cols = _full(pair)
    cols[missing].remove("component")
    client = FakeClient(present=[t.name for t in pair], columns=cols)
    assert tables_present(client, "db", tables=pair) is False


def test_default_tables_are_the_functional_pair():
    """A caller that passes no `tables` gets the JUnit pair, not the benchmark one."""
    client = FakeClient(present=[t.name for t in FUNCTIONAL], columns=_full(FUNCTIONAL))
    assert tables_present(client, "db") is True
