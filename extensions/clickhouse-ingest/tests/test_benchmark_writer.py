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

"""The benchmark write path: what it must refuse, and what it must not duplicate."""

from spyre_clickhouse_ingest import insert_benchmarks_v2, v2_benchmarks_already_ingested
from spyre_clickhouse_ingest.schema import BENCHMARK_RUNS, BENCHMARKS

_KEYS = ("run_mode", "tensor_parallel")


class FakeClient:
    """Records inserts; `known` is what the dimension already holds."""

    def __init__(self, known=(), run_count=0):
        self.known = list(known)
        self.run_count = run_count
        self.inserts = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows, column_names, database))

    def query(self, sql, parameters=None):
        if "count()" in sql:
            rows = [(self.run_count,)]
        else:
            asked = set(parameters["ids"])
            rows = [(k,) for k in self.known if str(k) in asked]

        class R:
            result_rows = rows

        return R()


def _bench(name="serve_g33", **kw):
    b = {
        "name": name,
        "tags": ["mode__serve"],
        "props": {"model": "granite"},
        "backend": "spyre",
        "measurements": {"avg_latency": [6.1, 6.2]},
        "iterations": 2,
        "disc": {"run_mode": "serve", "tensor_parallel": "1"},
        "disc_keys": _KEYS,
    }
    b.update(kw)
    return b


RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"


def _rows(client, table):
    return [i[1] for i in client.inserts if i[0] == table.name]


def test_writes_one_fact_row_and_one_identity_row():
    c = FakeClient()
    assert insert_benchmarks_v2(c, "db", "spyre-inference", RUN, [_bench()]) == 1
    assert len(_rows(c, BENCHMARKS)[0]) == 1
    assert len(_rows(c, BENCHMARK_RUNS)[0]) == 1


def test_every_metric_of_one_benchmark_is_one_row():
    # One row per metric would multiply every trend point by the metric count.
    c = FakeClient()
    n = insert_benchmarks_v2(
        c,
        "db",
        "spyre-inference",
        RUN,
        [
            _bench(measurements={"avg_latency": [6.1]}),
            _bench(measurements={"p99_latency": [7.0]}),
        ],
    )
    assert n == 1
    (row,) = _rows(c, BENCHMARK_RUNS)[0]
    measurements = row[BENCHMARK_RUNS.columns.index("measurements")]
    assert measurements == {"avg_latency": [6.1], "p99_latency": [7.0]}


def test_run_props_merge_across_entries_for_one_fact_row():
    # A sparser earlier entry must not drop a field a later one set for the same key.
    c = FakeClient()
    insert_benchmarks_v2(
        c,
        "db",
        "spyre-inference",
        RUN,
        [
            _bench(measurements={"avg_latency": [6.1]}, run_props={}),
            _bench(measurements={"p99_latency": [7.0]}, run_props={"host": "node1"}),
        ],
    )
    (row,) = _rows(c, BENCHMARK_RUNS)[0]
    assert row[BENCHMARK_RUNS.columns.index("props")] == {"host": "node1"}


def test_two_backends_are_two_rows_not_one():
    # backend is not in the identity, so both sides share a benchmark_id but are still
    # distinct measurements.
    c = FakeClient()
    n = insert_benchmarks_v2(
        c,
        "db",
        "spyre-inference",
        RUN,
        [_bench(backend="spyre"), _bench(backend="cpu")],
    )
    assert n == 2
    ident = _rows(c, BENCHMARKS)[0]
    assert len(ident) == 1, "one benchmark, two backends -- not two benchmarks"


def test_a_benchmark_with_no_measurements_is_dropped_not_inserted():
    # The DDL's CHECK length(measurements) > 0 rejects the row, failing the whole insert.
    c = FakeClient()
    assert insert_benchmarks_v2(c, "db", "c", RUN, [_bench(measurements={})]) == 0
    assert _rows(c, BENCHMARK_RUNS) == [] or _rows(c, BENCHMARK_RUNS)[0] == []


def test_a_dropped_benchmark_does_not_leave_an_orphan_identity_row():
    # An identity row with no fact row makes the dimension lie about what was measured.
    c = FakeClient()
    insert_benchmarks_v2(
        c, "db", "c", RUN, [_bench(name="kept"), _bench(name="empty", measurements={})]
    )
    ident = _rows(c, BENCHMARKS)[0]
    names = [r[BENCHMARKS.columns.index("name")] for r in ident]
    assert names == ["kept"]


def test_an_unidentifiable_benchmark_is_skipped_not_collided():
    # A blank name hashes to a real uuid every such benchmark would share.
    c = FakeClient()
    assert insert_benchmarks_v2(c, "db", "c", RUN, [_bench(name="")]) == 0


def test_a_known_identity_is_not_reinserted():
    # benchmarks is a plain MergeTree and the collision is across runs, so in-run dedup is
    # not enough.
    first = FakeClient()
    insert_benchmarks_v2(first, "db", "spyre-inference", RUN, [_bench()])
    bid = _rows(first, BENCHMARKS)[0][0][0]
    again = FakeClient(known=[bid])
    insert_benchmarks_v2(again, "db", "spyre-inference", RUN, [_bench()])
    assert _rows(again, BENCHMARKS) in ([], [[]]), "identity re-inserted"
    assert len(_rows(again, BENCHMARK_RUNS)[0]) == 1, "the fact row must still land"


def test_already_ingested_detects_a_prior_run():
    # Without this a re-ingest doubles every number behind a mean, which looks plausible.
    assert v2_benchmarks_already_ingested(FakeClient(run_count=3), "db", RUN, "c")
    assert not v2_benchmarks_already_ingested(FakeClient(run_count=0), "db", RUN, "c")


def test_rows_are_ordered_by_the_schema_model():
    c = FakeClient()
    insert_benchmarks_v2(c, "db", "spyre-inference", RUN, [_bench()])
    for table in (BENCHMARKS, BENCHMARK_RUNS):
        cols = next(i[2] for i in c.inserts if i[0] == table.name)
        assert cols == list(table.columns)
