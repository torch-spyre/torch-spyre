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

from spyre_clickhouse_ingest import insert_benchmarks, benchmarks_already_ingested
from spyre_clickhouse_ingest.schema import BENCHMARK_RUNS, BENCHMARKS

_KEYS = ("run_mode", "tensor_parallel")


class FakeClient:
    """Records inserts; `known` is what the dimension already holds."""

    def __init__(self, known=(), run_count=0):
        self.known = list(known)
        self.run_count = run_count
        self.inserts = []
        self.queries = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows, column_names, database))

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters or {}))
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
    assert insert_benchmarks(c, "db", "spyre-inference", RUN, [_bench()]) == 1
    assert len(_rows(c, BENCHMARKS)[0]) == 1
    assert len(_rows(c, BENCHMARK_RUNS)[0]) == 1


def test_every_metric_of_one_benchmark_is_one_row():
    # One row per metric would multiply every trend point by the metric count.
    c = FakeClient()
    n = insert_benchmarks(
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


def test_repeated_metric_key_keeps_every_sample():
    # Two entries sharing (benchmark, backend) AND a metric key are two samples of it.
    # Overwriting kept only the last and froze variance at zero.
    c = FakeClient()
    n = insert_benchmarks(
        c,
        "db",
        "spyre-inference",
        RUN,
        [
            _bench(measurements={"kernel_mean_ms": [6.1], "p99_latency": [7.0]}),
            _bench(measurements={"kernel_mean_ms": [6.4]}),
        ],
    )
    assert n == 1
    (row,) = _rows(c, BENCHMARK_RUNS)[0]
    measurements = row[BENCHMARK_RUNS.columns.index("measurements")]
    assert measurements == {"kernel_mean_ms": [6.1, 6.4], "p99_latency": [7.0]}


def test_run_props_merge_across_entries_for_one_fact_row():
    # A sparser earlier entry must not drop a field a later one set for the same key.
    c = FakeClient()
    insert_benchmarks(
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
    n = insert_benchmarks(
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
    assert insert_benchmarks(c, "db", "c", RUN, [_bench(measurements={})]) == 0
    assert _rows(c, BENCHMARK_RUNS) == [] or _rows(c, BENCHMARK_RUNS)[0] == []


def test_a_dropped_benchmark_does_not_leave_an_orphan_identity_row():
    # An identity row with no fact row makes the dimension lie about what was measured.
    c = FakeClient()
    insert_benchmarks(
        c, "db", "c", RUN, [_bench(name="kept"), _bench(name="empty", measurements={})]
    )
    ident = _rows(c, BENCHMARKS)[0]
    names = [r[BENCHMARKS.columns.index("name")] for r in ident]
    assert names == ["kept"]


def test_an_unidentifiable_benchmark_is_skipped_not_collided():
    # A blank name hashes to a real uuid every such benchmark would share.
    c = FakeClient()
    assert insert_benchmarks(c, "db", "c", RUN, [_bench(name="")]) == 0


def test_a_known_identity_is_not_reinserted():
    # benchmarks is a plain MergeTree and the collision is across runs, so in-run dedup is
    # not enough.
    first = FakeClient()
    insert_benchmarks(first, "db", "spyre-inference", RUN, [_bench()])
    bid = _rows(first, BENCHMARKS)[0][0][0]
    again = FakeClient(known=[bid])
    insert_benchmarks(again, "db", "spyre-inference", RUN, [_bench()])
    assert _rows(again, BENCHMARKS) in ([], [[]]), "identity re-inserted"
    assert len(_rows(again, BENCHMARK_RUNS)[0]) == 1, "the fact row must still land"


def test_already_ingested_detects_a_prior_run():
    # Without this a re-ingest doubles every number behind a mean, which looks plausible.
    assert benchmarks_already_ingested(FakeClient(run_count=3), "db", RUN, "c")
    assert not benchmarks_already_ingested(FakeClient(run_count=0), "db", RUN, "c")


def test_already_ingested_scopes_by_report_kind():
    # One invocation ingests a kernel report and a benchmark report under ONE run_id, so a
    # (component, run_id)-only key makes the second file look already-ingested and drops it.
    c = FakeClient(run_count=3)
    assert benchmarks_already_ingested(c, "db", RUN, "c", "benchmark")
    sql, params = c.queries[-1]
    assert "props['report_kind']" in sql
    assert params["kind"] == "benchmark"
    # Omitted, the scope stays as it was, so rows predating the key still match.
    c2 = FakeClient(run_count=3)
    assert benchmarks_already_ingested(c2, "db", RUN, "c")
    assert "report_kind" not in c2.queries[-1][0]


def test_report_kind_is_stamped_and_not_overridable_by_the_producer():
    # The dedup above reads this prop, so a producer prop of the same name must not win.
    c = FakeClient()
    insert_benchmarks(
        c,
        "db",
        "spyre-inference",
        RUN,
        [_bench(run_props={"report_kind": "spoofed", "host": "node1"})],
        report_kind="kernel",
    )
    (row,) = _rows(c, BENCHMARK_RUNS)[0]
    props = row[BENCHMARK_RUNS.columns.index("props")]
    assert props["report_kind"] == "kernel"
    assert props["host"] == "node1"


def test_tags_are_unioned_across_entries_for_one_identity():
    # The id hashes tags NORMALIZED and SORTED, so the same members in different case or
    # order are one bid. Replacing the list wholesale kept only whichever entry ran last.
    c = FakeClient()
    insert_benchmarks(
        c,
        "db",
        "spyre-inference",
        RUN,
        [
            _bench(tags=["mode__serve", "tier__perf"]),
            _bench(tags=["TIER__PERF", "mode__serve"]),
        ],
    )
    (row,) = _rows(c, BENCHMARKS)[0]
    tags = row[BENCHMARKS.columns.index("tags")]
    # ONE canonical spelling per member. Unioning the raw strings instead put both
    # 'TIER__PERF' and 'tier__perf' on this single identity, so has(tags,'tier__perf')
    # answered differently depending on which spelling a caller guessed.
    assert tags == ["mode__serve", "tier__perf"], tags


def test_name_is_first_write_wins_not_last():
    # bid already agrees name in substance across entries sharing it; this only picks
    # which literal spelling survives, deterministically rather than by arrival order.
    c = FakeClient()
    insert_benchmarks(
        c,
        "db",
        "spyre-inference",
        RUN,
        [_bench(name="Serve_G33"), _bench(name="serve_g33")],
    )
    (row,) = _rows(c, BENCHMARKS)[0]
    assert row[BENCHMARKS.columns.index("name")] == "Serve_G33"


def test_iterations_sum_across_merged_entries():
    # Two entries sharing a fact key each contribute their own distinct iteration count;
    # max() under-counts once more than one entry contributes samples.
    c = FakeClient()
    insert_benchmarks(
        c,
        "db",
        "spyre-inference",
        RUN,
        [
            _bench(measurements={"avg_latency": [6.1]}, iterations=2),
            _bench(measurements={"avg_latency": [6.4]}, iterations=3),
        ],
    )
    (row,) = _rows(c, BENCHMARK_RUNS)[0]
    assert row[BENCHMARK_RUNS.columns.index("iterations")] == 5


def test_already_ingested_scopes_by_source_file():
    # A sharded run can pass several same-kind XMLs under one run_id; report_kind alone
    # would let the first shard block the rest.
    c = FakeClient(run_count=3)
    assert benchmarks_already_ingested(c, "db", RUN, "c", "kernel", "shard-1.xml")
    sql, params = c.queries[-1]
    assert "props['source_file']" in sql
    assert params["sf"] == "shard-1.xml"


def test_source_file_is_stamped_and_not_overridable_by_the_producer():
    c = FakeClient()
    insert_benchmarks(
        c,
        "db",
        "spyre-inference",
        RUN,
        [_bench(run_props={"source_file": "spoofed", "host": "node1"})],
        report_kind="kernel",
        source_file="shard-1.xml",
    )
    (row,) = _rows(c, BENCHMARK_RUNS)[0]
    props = row[BENCHMARK_RUNS.columns.index("props")]
    assert props["source_file"] == "shard-1.xml"
    assert props["host"] == "node1"


def test_rows_are_ordered_by_the_schema_model():
    c = FakeClient()
    insert_benchmarks(c, "db", "spyre-inference", RUN, [_bench()])
    for table in (BENCHMARKS, BENCHMARK_RUNS):
        cols = next(i[2] for i in c.inserts if i[0] == table.name)
        assert cols == list(table.columns)
