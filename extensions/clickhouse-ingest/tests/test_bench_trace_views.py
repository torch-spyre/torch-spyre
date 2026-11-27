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

"""Schema 87 and 90 on chdb: the step and span views over a 4-rank x 64-step x (40 layers x 6 spans) trace written through
trace_bundle and the benchmark writer, a golden of their output, and the guards (invalidation, ragged arrays, re-upload,
the exclusion list)."""

import json
import os
from pathlib import Path

import pytest
import trace_fixture as tf
from spyre_clickhouse_ingest import bundle, trace_bundle
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier
from spyre_clickhouse_ingest.writer import BenchmarkWriter, DryRunClient

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests",
    "20-artifacts",
    "30-benchmarks",
    "49-result-uploads",
    "50-artifact-views",
    "60-benchmark-views",
    "81-bench-environments",
    "82-run-placements",
    "83-result-invalidations",
    "87-bench-trace-views",
    "90-bench-span-values",
)
GOLDEN = Path(__file__).parent / "data" / "trace_views_golden.json"
RUN = "aaaaaaaa-0000-5000-8000-000000000001"
KEY = "run_id = '" + RUN + "'"


def only(run):
    """chdb sessions share one engine in a process, so every test reads its own run."""
    return f"run_id = '{run}'"


class Db:
    def __init__(self, files=FILES):
        self.s = session.Session()
        for name in files:
            for stmt in SchemaApplier.statements(
                (SCHEMA_DIR / f"{name}.sql").read_text()
            ):
                self.s.query(stmt)

    def rows(self, sql):
        raw = self.s.query(sql, "JSONEachRow").data()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]

    def insert(self, table, *rows):
        body = "\n".join(json.dumps(r, default=str) for r in rows)
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    def load(self, trace, run=RUN, mutate=None):
        """Write the trace as the bundle path does: one writer call per results file."""
        c = DryRunClient()
        for name, body in trace_bundle.emit(trace).items():
            if mutate:
                mutate(name, body)
            BenchmarkWriter.insert(c, "d", trace["component"], run, bundle.generic_entries(body["benchmarks"]),
                                   report_kind="benchmarks", source_file=name)  # fmt: skip
        for table in ("benchmarks", "benchmark_runs"):
            if c.rows.get(table):
                self.insert(table, *c.rows[table])
        return c


@pytest.fixture(scope="module")
def full():
    d = Db()
    d.load(tf.trace())
    yield d
    d.s.close()


def test_the_trace_is_four_ranks_by_sixty_four_steps_by_240_spans(full):
    (n,) = full.rows(
        f"SELECT count() AS n, uniqExact(rank) AS r FROM v_bench_spans WHERE {KEY}"
    )
    assert (n["n"], n["r"]) == (4 * 64 * 40 * 6, 4)
    (n,) = full.rows(f"SELECT count() AS n FROM v_bench_step_records WHERE {KEY}")
    assert n["n"] == 4 * 64


def test_records_carry_roles_and_the_unnested_metrics(full):
    r = full.rows(
        f"SELECT * FROM v_bench_step_records WHERE {KEY} AND rank = 2 AND pos = 5"
    )[0]
    assert (
        r["step"],
        r["kind"],
        r["cold"],
        r["total_signal"],
        r["total_rule"],
        r["total_rule_declared"],
    ) == ("itl_0004", "itl", 0, "sig_total", "max", 1)
    assert r["total"] == tf.step_values(2, 4)["sig_total"] and r["wait_rule"] == "mean"
    assert set(r["metrics"]) == {
        "sig_total",
        "sig_wait",
        "sig_compute",
        "sig_busy",
        "sig_aux",
    }
    assert not any(k.startswith(("summary.", "headline.")) for k in r["metrics"])


def test_cross_rank_value_is_max_for_latency_and_mean_for_compute(full):
    for pos in (3, 31):
        a = full.rows(
            f"SELECT * FROM v_bench_step_rank_agg WHERE {KEY} AND pos = {pos}"
        )[0]
        step = pos - 1
        vals = [tf.step_values(r, step) for r in range(4)]
        assert a["total"] == pytest.approx(max(v["sig_total"] for v in vals))
        assert a["compute"] == pytest.approx(sum(v["sig_compute"] for v in vals) / 4)
        assert a["slowest_rank"] == (2 if pos != 31 else 2) and a["n_ranks"] == 4
    a = full.rows(f"SELECT * FROM v_bench_step_rank_agg WHERE {KEY} AND pos = 31")[0]
    assert (a["awaited_rank"], a["cold"]) == (1, 0)


def test_headline_is_the_median_over_warm_steps_and_null_without_any(full):
    h = {
        r["kind"]: r
        for r in full.rows(f"SELECT * FROM v_bench_step_headline WHERE {KEY}")
    }
    warm = sorted(
        max(tf.step_values(r, s)["sig_total"] for r in range(4)) for s in range(2, 64)
    )
    assert h["itl"]["n_warm_steps"] == 62 and h["itl"]["headline"] == pytest.approx(
        (warm[30] + warm[31]) / 2
    )
    assert h["ttft"]["n_warm_steps"] == 0 and h["ttft"]["headline"] is None
    assert h["itl"]["slowest_rank"] == 2 and h["itl"]["n_ranks"] == 4
    assert h["itl"]["rank_skew"] == pytest.approx(
        max(h["itl"]["rank_medians"].values()) - min(h["itl"]["rank_medians"].values())
    )
    assert 7.5 < h["itl"]["rank_skew"] < 8.5


def test_the_headline_in_the_rows_agrees_with_the_view(full):
    (m,) = full.rows(f"SELECT measurements['headline.itl.sig_total'][1] AS h FROM benchmark_runs WHERE {KEY} AND props['run.rank'] = '0'"
                     " AND mapContains(measurements, 'step_kind')")  # fmt: skip
    (h,) = full.rows(
        f"SELECT headline FROM v_bench_step_headline WHERE {KEY} AND kind = 'itl'"
    )
    assert m["h"] == pytest.approx(h["headline"])


def test_lag_and_wait(full):
    lag = {
        (r["rank"], r["pos"]): r
        for r in full.rows(f"SELECT * FROM v_bench_step_lag WHERE {KEY}")
    }
    assert lag[(2, 5)]["lag"] == pytest.approx(8.0) and lag[(0, 5)][
        "lag"
    ] == pytest.approx(0.0)
    assert [r for (rk, p), r in lag.items() if p == 5 and r["is_awaited_rank"]][0][
        "rank"
    ] == 2
    assert lag[(1, 31)]["is_awaited_rank"] and not lag[(2, 31)]["is_awaited_rank"]
    assert (
        lag[(1, 31)]["wait_spread"]
        == pytest.approx(100.0 + 8.0 - 2.0 + 2.0 - 2.0, abs=1e-6)
        or lag[(1, 31)]["wait_spread"] > 90
    )


def test_outliers_are_the_stall_step_on_every_rank_with_the_excess_split(full):
    out = full.rows(f"SELECT * FROM v_bench_step_outliers WHERE {KEY} ORDER BY rank")
    assert [(o["rank"], o["step"]) for o in out] == [(r, "itl_0030") for r in range(4)]
    assert all(
        o["z"] > 3.5 and o["excess"] == pytest.approx(100.0, abs=1.0) for o in out
    )
    by = {o["rank"]: o for o in out}
    assert by[1]["excess_compute"] == pytest.approx(100.0, abs=1.0) and by[1][
        "excess_wait"
    ] == pytest.approx(0.0, abs=1e-6)
    assert by[0]["excess_wait"] == pytest.approx(100.0, abs=1e-6) and by[0][
        "excess_host"
    ] == pytest.approx(0.0, abs=1.0)


def test_rank_coverage(full):
    (c,) = full.rows(f"SELECT * FROM v_bench_rank_coverage WHERE {KEY}")
    assert (c["ranks_present"], c["world_size"], c["complete"]) == (4, 4, 1)


def test_span_queries_answer_the_perf_lab_questions(full):
    per_layer = full.rows(
        f"SELECT rank, layer, sum(dur_ns) AS ns FROM v_bench_spans WHERE {KEY} AND lane = 'collective' GROUP BY rank, layer ORDER BY rank, layer"
    )
    assert len(per_layer) == 4 * 40
    cold = full.rows(
        f"SELECT rank, max(dur_ns) AS ns FROM v_bench_spans WHERE {KEY} AND name = 'wait_peer' AND step = 0 AND layer = 0 GROUP BY rank ORDER BY rank"
    )
    assert [c["ns"] for c in cold] == list(tf.COLD_WAIT_NS)
    tl = full.rows(
        f"SELECT seq, parent_seq, depth, name FROM v_bench_spans WHERE {KEY} AND rank = 2 AND step = 3 AND layer = 0 ORDER BY start_ns, seq"
    )
    assert (
        [t["name"] for t in tl] == [n for _, n in tf.LANES]
        and tl[2]["parent_seq"] == tl[0]["seq"]
        and tl[2]["depth"] == 1
    )


def test_the_fast_path_equals_the_view_exactly(full):
    cols = "rank, prompt, repeat, step, seq, parent_seq, depth, layer, lane, name, start_ns, dur_ns, case_id"
    (a,) = full.rows(
        f"SELECT count() AS n FROM (SELECT {cols} FROM v_bench_spans WHERE {KEY} EXCEPT SELECT {cols} FROM v_bench_spans_fast WHERE {KEY})"
    )
    (b,) = full.rows(
        f"SELECT count() AS n FROM (SELECT {cols} FROM v_bench_spans_fast WHERE {KEY} EXCEPT SELECT {cols} FROM v_bench_spans WHERE {KEY})"
    )
    (c,) = full.rows(f"SELECT count() AS n FROM v_bench_spans_fast WHERE {KEY}")
    assert (a["n"], b["n"], c["n"]) == (0, 0, 4 * 64 * 240)


def test_the_view_output_matches_the_golden(full):
    got = {
        "headline": full.rows(f"SELECT kind, n_warm_steps, headline, rule, rule_declared, rank_skew, slowest_rank, n_ranks FROM v_bench_step_headline WHERE {KEY} ORDER BY kind"),
        "outliers": full.rows(f"SELECT rank, step, value, median, z, excess, excess_wait, excess_compute, excess_host FROM v_bench_step_outliers WHERE {KEY} ORDER BY rank"),
        "lag_pos_5": full.rows(f"SELECT rank, total, lag, wait, is_awaited_rank, wait_spread_share FROM v_bench_step_lag WHERE {KEY} AND pos = 5 ORDER BY rank"),
        "cross_rank_pos_1_2_31": full.rows(f"SELECT pos, step, cold, total, wait, compute, busy, slowest_rank, fastest_rank, awaited_rank FROM v_bench_step_rank_agg WHERE {KEY} AND pos IN (1, 2, 31) ORDER BY pos"),
        "spans_by_lane": full.rows(f"SELECT lane, rank, count() AS n, sum(dur_ns) AS ns, max(dur_ns) AS mx FROM v_bench_spans WHERE {KEY} GROUP BY lane, rank ORDER BY lane, rank"),
        "collective_layer_0_3": full.rows(f"SELECT rank, layer, sum(dur_ns) AS ns FROM v_bench_spans WHERE {KEY} AND lane = 'collective' AND layer IN (0, 3) GROUP BY rank, layer ORDER BY rank, layer"),
    }  # fmt: skip
    got = json.loads(json.dumps(got))
    if os.environ.get("TRACE_GOLDEN_WRITE"):
        Path(os.environ["TRACE_GOLDEN_WRITE"]).write_text(
            json.dumps(got, indent=1, sort_keys=True) + "\n"
        )

    def near(a, b, path=""):
        if isinstance(a, float) or isinstance(b, float):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), path
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), path
            [near(a[k], b[k], f"{path}/{k}") for k in a]
        elif isinstance(a, list):
            assert len(a) == len(b), path
            [near(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))]
        else:
            assert a == b, path

    near(got, json.loads(GOLDEN.read_text()))


def test_an_invalidated_repeat_and_a_partial_rank_set_are_visible():
    d = Db()
    R, R2 = (
        "aaaaaaaa-0000-5000-8000-0000000000a1",
        "aaaaaaaa-0000-5000-8000-0000000000a2",
    )
    K, K2 = only(R), only(R2)
    d.load(tf.trace(steps=8, layers=2), R)
    d.insert("result_invalidations", {"invalidation_id": "dddddddd-0000-5000-8000-000000000001", "ts": "2026-11-16 00:00:00.000",
                                      "state": "invalidated", "actor": "amy@example.invalid", "reason_code": "bad_card",
                                      "scope_kind": "run_benchmark", "run_id": R,
                                      "benchmark_id": d.rows(f"SELECT DISTINCT benchmark_id FROM benchmark_runs WHERE {K} AND mapContains(measurements, 'step_kind')")[0]["benchmark_id"],
                                      "repeat": 0})  # fmt: skip
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_step_records WHERE {K}")[0]["n"] == 0
    )
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_spans WHERE {K}")[0]["n"]
        == 4 * 8 * 12
    )
    t = tf.trace(steps=8, layers=2)
    t["ranks"] = t["ranks"][:2]
    t["case"]["props"]["world_size"] = 4
    d.load(t, R2)
    (c,) = d.rows(
        f"SELECT ranks_present, world_size, complete FROM v_bench_rank_coverage WHERE {K2}"
    )
    assert (c["ranks_present"], c["world_size"], c["complete"]) == (2, 4, 0)


def test_a_ragged_span_row_is_skipped_and_never_fails_the_insert():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000b1"
    K = only(R)

    def ragged(name, body):
        for row in body["benchmarks"]:
            if row["props"].get("kernel_name") == "matmul" and name.startswith("rank1"):
                row["measurements"]["dur_ns"] = row["measurements"]["dur_ns"][:-1]

    d.load(tf.trace(steps=4, layers=2), R, mutate=ragged)
    assert d.rows(f"SELECT count() AS n FROM benchmark_runs WHERE {K}")[0]["n"] == 4 * 7
    n_view = d.rows(f"SELECT count() AS n FROM v_bench_spans WHERE {K}")[0]["n"]
    n_fast = d.rows(f"SELECT count() AS n FROM bench_span_values WHERE {K}")[0]["n"]
    assert n_view == n_fast == 4 * 4 * 12 - 4 * 2


def test_a_reupload_replaces_never_mixes():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000c1"
    K = only(R)
    d.load(tf.trace(steps=4, layers=3), R)
    d.s.query("SELECT sleep(0.01)")
    d.load(tf.trace(steps=4, layers=2), R)
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_spans WHERE {K}")[0]["n"]
        == 4 * 4 * 12
    )
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_spans_fast WHERE {K}")[0]["n"]
        == 4 * 4 * 12
    )
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_step_records WHERE {K}")[0]["n"]
        == 4 * 4
    )


def test_trace_rows_stay_out_of_the_trend_views_via_the_shared_exclusion_list(full):
    types = {
        r["record_type"]
        for r in full.rows("SELECT record_type FROM benchmark_excluded_record_types")
    }
    assert {"step_trace", "span"} <= types
    for view in ("v_benchmark_trend", "v_benchmark_backend_compare"):
        ids = f"benchmark_id IN (SELECT benchmark_id FROM benchmark_runs WHERE {KEY})"
        assert full.rows(f"SELECT count() AS n FROM {view} WHERE {ids}")[0]["n"] == 0, (
            view
        )


def test_mv_less_deployment_still_reads_through_the_view():
    d = Db(FILES[:-1])
    R = "aaaaaaaa-0000-5000-8000-0000000000d1"
    d.load(tf.trace(steps=4, layers=2), R)
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_spans WHERE {only(R)}")[0]["n"]
        == 4 * 4 * 12
    )


def test_a_non_finite_value_never_fails_the_insert_and_is_skipped_everywhere():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000e1"
    d.load(tf.trace(steps=4, layers=1), R)
    bid = d.rows(
        f"SELECT DISTINCT benchmark_id FROM benchmark_runs WHERE {only(R)} AND mapContains(measurements, 'dur_ns') LIMIT 1"
    )[0]["benchmark_id"]
    other = "aaaaaaaa-0000-5000-8000-0000000000e2"
    for bad in ("nan", "inf", "-inf"):
        d.s.query(
            "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements, iterations, props) VALUES "
            f"('{other}', '{bid}', 'synth-trace', 'synth-backend', map('step', [0.], 'seq', [0.], 'parent_seq', [-1.], 'depth', [0.],"
            f" 'layer', [0.], 'start_ns', [1.], 'dur_ns', [{bad}]), 1, map('run.rank', '{bad == 'nan' and 7 or 8}'))"
        )
    d.s.query(
        "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements, iterations, props) VALUES "
        f"('{other}', '{bid}', 'synth-trace', 'synth-backend', map('step_index', [0., nan], 'step_kind', [0., 1.], 'cold', [0., 0.], 'sig_total', [1., 2.]),"
        " 2, map('run.rank', '9', 'role.total', 'sig_total'))"
    )
    assert (
        d.rows(f"SELECT count() AS n FROM benchmark_runs WHERE {only(other)}")[0]["n"]
        == 4
    )
    assert (
        d.rows(f"SELECT count() AS n FROM bench_span_values WHERE {only(other)}")[0][
            "n"
        ]
        == 0
    )
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_spans WHERE {only(other)}")[0]["n"]
        == 0
    )
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_step_records WHERE {only(other)}")[0][
            "n"
        ]
        == 0
    )


def test_two_backends_in_one_run_stay_apart():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000e3"
    for backend in ("synth-backend", "synth-cpu"):
        t = tf.trace(steps=4, layers=1)
        t["backend"] = backend
        d.load(t, R)
    K = only(R)
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_step_records WHERE {K}")[0]["n"]
        == 2 * 4 * 4
    )
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_spans WHERE {K}")[0]["n"]
        == 2 * 4 * 4 * 6
    )
    assert (
        d.rows(f"SELECT count() AS n FROM v_bench_spans_fast WHERE {K}")[0]["n"]
        == 2 * 4 * 4 * 6
    )
    assert {
        r["backend"]
        for r in d.rows(f"SELECT DISTINCT backend FROM v_bench_step_headline WHERE {K}")
    } == {"synth-backend", "synth-cpu"}


def test_ranks_are_matched_by_step_index_not_by_position():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000e4"

    def drop_step_4_on_rank_1(name, body):
        if name.startswith("rank1"):
            step_row = body["benchmarks"][0]
            step_row["measurements"] = {
                k: v[:4] + v[5:] if len(v) == 8 else v
                for k, v in step_row["measurements"].items()
            }

    d.load(tf.trace(steps=8, layers=1), R, mutate=drop_step_4_on_rank_1)
    rows = {
        r["step_index"]: r
        for r in d.rows(
            f"SELECT step_index, n_ranks, step FROM v_bench_step_rank_agg WHERE {only(R)}"
        )
    }
    assert (
        rows[4]["n_ranks"] == 3 and rows[5]["n_ranks"] == 4 and rows[7]["n_ranks"] == 4
    )
    assert rows[5]["step"] == "itl_0005"


def test_the_mv_never_throws_whatever_order_the_optimizer_runs_it_in():
    d = Db()
    d.s.query("SET query_plan_filter_push_down = 0")
    d.s.query("SET optimize_move_to_prewhere = 0")
    R = "aaaaaaaa-0000-5000-8000-0000000000e5"
    d.load(tf.trace(steps=2, layers=1), R)
    bid = d.rows(
        f"SELECT DISTINCT benchmark_id FROM benchmark_runs WHERE {only(R)} AND mapContains(measurements, 'dur_ns') LIMIT 1"
    )[0]["benchmark_id"]
    d.s.query(
        "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements, iterations, props) VALUES "
        f"('{R}', '{bid}', 'synth-trace', 'synth-backend', map('step', [0., 1.], 'seq', [0.], 'parent_seq', [-1.], 'depth', [0.],"
        " 'layer', [0.], 'start_ns', [1.], 'dur_ns', [1.]), 1, map('run.rank', '6'))"
    )
    assert (
        d.rows(
            f"SELECT count() AS n FROM bench_span_values WHERE {only(R)} AND rank = 6"
        )[0]["n"]
        == 0
    )


def test_duplicate_step_indexes_and_incomplete_rank_sets_are_not_complete_measurements():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000e6"

    def dup(name, body):
        if name.startswith("rank1"):
            row = body["benchmarks"][0]
            row["measurements"]["step_index"][2] = row["measurements"]["step_index"][1]

    d.load(tf.trace(steps=6, layers=1), R, mutate=dup)
    assert (
        d.rows(
            f"SELECT uniqExact(rank) AS n FROM v_bench_step_records WHERE {only(R)}"
        )[0]["n"]
        == 3
    )
    (c,) = d.rows(
        f"SELECT ranks_present, complete FROM v_bench_rank_coverage WHERE {only(R)}"
    )
    assert (c["ranks_present"], c["complete"]) == (3, 0)
    R2 = "aaaaaaaa-0000-5000-8000-0000000000e7"
    t = tf.trace(steps=4, layers=1)
    t["ranks"][3]["rank"] = 5
    d.load(t, R2)
    (c,) = d.rows(
        f"SELECT ranks_present, world_size, complete FROM v_bench_rank_coverage WHERE {only(R2)}"
    )
    assert (c["ranks_present"], c["world_size"], c["complete"]) == (4, 4, 0)


def test_out_of_range_span_values_are_skipped_by_the_view_and_the_mv():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000f1"
    d.load(tf.trace(steps=2, layers=1), R)
    bid = d.rows(
        f"SELECT DISTINCT benchmark_id FROM benchmark_runs WHERE {only(R)} AND mapContains(measurements, 'dur_ns') LIMIT 1"
    )[0]["benchmark_id"]
    base = {
        "step": "0.",
        "seq": "0.",
        "parent_seq": "-1.",
        "depth": "0.",
        "layer": "0.",
        "start_ns": "1.",
        "dur_ns": "1.",
    }
    for rank, (key, bad) in enumerate(
        [
            ("dur_ns", "1e30"),
            ("dur_ns", "-5."),
            ("step", "-1."),
            ("seq", "-2."),
            ("depth", "300."),
            ("layer", "99999."),
            ("start_ns", "1e30"),
        ],
        20,
    ):
        m = ", ".join(f"'{k}', [{bad if k == key else v}]" for k, v in base.items())
        d.s.query(
            "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements, iterations, props) VALUES "
            f"('{R}', '{bid}', 'synth-trace', 'synth-backend', map({m}), 1, map('run.rank', '{rank}'))"
        )
    assert (
        d.rows(
            f"SELECT count() AS n FROM v_bench_spans WHERE {only(R)} AND rank >= 20"
        )[0]["n"]
        == 0
    )
    assert (
        d.rows(
            f"SELECT count() AS n FROM bench_span_values WHERE {only(R)} AND rank >= 20"
        )[0]["n"]
        == 0
    )


def test_coverage_flags_a_rank_that_reported_fewer_steps():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000f2"

    def short(name, body):
        if name.startswith("rank1"):
            row = body["benchmarks"][0]
            row["measurements"] = {
                k: v[:4] if len(v) == 8 else v for k, v in row["measurements"].items()
            }

    d.load(tf.trace(steps=8, layers=1), R, mutate=short)
    (c,) = d.rows(
        f"SELECT complete, min_ranks_per_step, steps_complete FROM v_bench_rank_coverage WHERE {only(R)}"
    )
    assert (c["complete"], c["min_ranks_per_step"], c["steps_complete"]) == (1, 3, 0)


def test_fast_view_picks_the_same_upload_as_the_slow_view_when_ts_and_audit_order_disagree():
    d = Db()
    R = "aaaaaaaa-0000-5000-8000-0000000000f3"
    d.load(tf.trace(steps=4, layers=1), R)
    row = d.rows(
        f"SELECT benchmark_id, measurements, props FROM benchmark_runs WHERE {only(R)} AND mapContains(measurements, 'dur_ns') AND props['run.rank'] = '0' LIMIT 1"
    )[0]
    keep = {k: v[:2] for k, v in row["measurements"].items()}
    d.insert("benchmark_runs", {"ts": "2020-01-01 00:00:00", "run_id": R, "benchmark_id": row["benchmark_id"], "component": "synth-trace",
                                "backend": "synth-backend", "measurements": keep, "iterations": 2, "props": row["props"],
                                "audit_timestamp": "2099-01-01 00:00:00.000"})  # fmt: skip
    slow = d.rows(f"SELECT count() AS n FROM v_bench_spans WHERE {only(R)}")[0]["n"]
    fast = d.rows(f"SELECT count() AS n FROM v_bench_spans_fast WHERE {only(R)}")[0][
        "n"
    ]
    assert slow == fast
