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

"""perf_api on chdb over the synthetic fixture: every layer's numbers, the invalidation handling, and that each query is bounded."""

import json
import os
import uuid
from datetime import timedelta

import pytest
import regex
from spyre_clickhouse_ingest import perf_api as api
from spyre_clickhouse_ingest import perf_fixture as fx

SERVER = os.environ.get(
    "PERF_API_TEST_CLICKHOUSE"
)  # host:port of a throwaway ClickHouse server; adds a clickhouse_connect run

PLACEHOLDER = regex.compile(r"\{(\w+):([^{}]+)\}")
BOUNDS = ("LIMIT", "{t_from:", "run_id = {", "run_id IN {", "benchmark_id = {")
NOW = fx.night(60) + timedelta(hours=4)


def literal(v):
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(literal(x) for x in v) + "]"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def bind(sql, params):
    return PLACEHOLDER.sub(
        lambda m: f"CAST({literal(params[m.group(1)])} AS {m.group(2)})", sql
    )


class Base:
    queries = None

    def prepare(self):
        self.queries = []
        for stmt in fx.schema_statements():
            if self.skip(stmt):
                continue
            self.exec(stmt)
        fx.load(fx.build(), self.insert)
        self.exec(fx.VERDICT_FILL_SQL)
        return self

    def insert(self, table, rows):
        self.load(table, "\n".join(json.dumps(r) for r in rows))


class ChdbDb(Base):
    def __init__(self):
        from chdb import session

        self.s = session.Session()

    def skip(self, stmt):
        return (
            "REFRESH EVERY" in stmt
        )  # chdb has no refreshable MV; the table is filled by hand

    def exec(self, sql):
        self.s.query(sql)

    def load(self, table, body):
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    def __call__(self, sql, params=None):
        self.queries.append(sql)
        raw = self.s.query(bind(sql, params or {}), "JSONEachRow").data()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]

    def close(self):
        self.s.close()


class ServerDb(Base):
    """The same checks through clickhouse_connect against a real server, in a database dropped afterwards."""

    def __init__(self, hostport):
        import clickhouse_connect

        host, port = hostport.split(":")
        admin = clickhouse_connect.get_client(host=host, port=int(port))
        self.name = f"perf_api_{uuid.uuid4().hex[:8]}"
        admin.command(f"CREATE DATABASE {self.name}")
        admin.close()
        self.c = clickhouse_connect.get_client(
            host=host, port=int(port), database=self.name
        )
        self.run = api.clickhouse_runner(self.c)

    def skip(self, stmt):
        return False

    def exec(self, sql):
        self.c.command(sql)

    def load(self, table, body):
        self.c.raw_insert(table, insert_block=body.encode(), fmt="JSONEachRow")

    def __call__(self, sql, params=None):
        self.queries.append(sql)
        return self.run(sql, params)

    def close(self):
        self.c.command(f"DROP DATABASE IF EXISTS {self.name}")
        self.c.close()


@pytest.fixture(scope="module", params=["chdb", "server"])
def db(request):
    if request.param == "server" and not SERVER:
        pytest.skip("PERF_API_TEST_CLICKHOUSE not set")
    if request.param == "chdb":
        pytest.importorskip("chdb.session")
    d = (ChdbDb() if request.param == "chdb" else ServerDb(SERVER)).prepare()
    yield d
    d.close()


def rid(comp, arch, i):
    return fx.uid("run", comp, arch, i)


def bid(db, name):
    return db(
        f"SELECT toString(benchmark_id) AS b FROM benchmarks WHERE name = '{name}' LIMIT 1"
    )[0]["b"]


TP4 = fx.uid("run", "tp4")
MATMUL = "synth_matmul_bf16_1024"


def test_every_query_is_bounded(db):
    api.overview(db, NOW)
    api.suites(db, NOW, arch="x86_64")
    api.series(db, bid(db, MATMUL), NOW, metric="total_duration_ms", arch="x86_64")
    api.run_detail(db, rid(fx.TS, "x86_64", 46), NOW)
    api.environment(db, rid(fx.TS, "x86_64", 46), NOW)
    api.steps(db, TP4, NOW)
    api.spans(db, TP4, NOW)
    api.references(db, NOW)
    for sql in db.queries:
        assert any(b in sql for b in BOUNDS), sql[:200]
    assert not any(regex.search(r"\b(spyre\.|run_properties)", q) for q in db.queries)


def test_overview_tiles_per_arch_and_component(db):
    tiles = {
        (t["arch"], t["component"]): t for t in api.overview(db, NOW)["data"]["tiles"]
    }
    assert set(tiles) == {
        ("x86_64", fx.TS),
        ("x86_64", fx.SI),
        ("ppc64le", fx.TS),
        ("s390x", fx.TS),
    }
    t = tiles[("x86_64", fx.TS)]
    assert t["benchmarks_in_window"] == 28 and t["coverage"]["ratio"] == 1.0
    assert not t["freshness"]["stale"] and t["freshness"]["age_hours"] < 36
    assert sum(t["verdicts"][k] for k in ("fail", "warn", "pass", "no_baseline")) == 28


def test_overview_as_of_shows_the_regression_and_excludes_an_invalidated_run(db):
    t = {
        (t["arch"], t["component"]): t
        for t in api.overview(db, as_of=fx.night(47) + timedelta(hours=4))["data"][
            "tiles"
        ]
    }
    assert t[("x86_64", fx.TS)]["verdicts"]["fail"] == 1
    assert t[("x86_64", fx.SI)]["verdicts"]["fail"] == 1
    assert t[("ppc64le", fx.TS)]["verdicts"]["fail"] == 0
    old = {
        (t["arch"], t["component"]): t
        for t in api.overview(db, as_of=fx.night(22))["data"]["tiles"]
    }[("x86_64", fx.TS)]
    assert old["invalidated_runs_in_window"] == 1
    assert (
        old["runs_in_window"] == 6
    )  # 7 nights in the window minus the invalidated one


def test_overview_with_no_data_says_so(db):
    r = api.overview(db, arch="s390x", component=fx.SI)
    assert r["data"]["tiles"] == [] and r["meta"]["no_data"] is True


def test_suites_rank_failures_first_and_page(db):
    r = api.suites(
        db,
        as_of=fx.night(47) + timedelta(hours=4),
        arch="x86_64",
        component=fx.TS,
        limit=10,
    )
    items = r["data"]["items"]
    assert items[0]["name"] == MATMUL and items[0]["verdict"] == "fail"
    assert (
        "total_duration_ms" in items[0]["regressed_metrics"]
        and items[0]["worst_delta_pct"] > 10
    )
    assert r["meta"]["total"] == 28 and len(items) == 10 and r["meta"]["next_cursor"]
    assert items[0]["group"] == "synth-core" and len(items[0]["history"]) == 10
    p2 = api.suites(
        db, arch="x86_64", component=fx.TS, limit=10, cursor=r["meta"]["next_cursor"]
    )
    assert len(p2["data"]["items"]) == 10 and not {
        i["benchmark_id"] for i in p2["data"]["items"]
    } & {i["benchmark_id"] for i in items}
    only_fail = api.suites(
        db, as_of=fx.night(47) + timedelta(hours=4), arch="x86_64", status="fail"
    )["data"]["items"]
    assert {i["name"] for i in only_fail} == {MATMUL, "synth_serve_tp4_c8"}


def test_series_markers_band_and_invalidated_point(db):
    r = api.series(
        db, bid(db, MATMUL), NOW, metric="total_duration_ms", arch="x86_64", days=100
    )
    d = r["data"]
    assert d["count"] == 60 and d["series"]["source_job"] == "synth-nightly-perf-x86_64"
    by_night = {p["run_id"]: p for p in d["points"]}
    p45, p46, p30 = (by_night[rid(fx.TS, "x86_64", i)] for i in (45, 46, 30))
    assert (
        p45["verdict"]["status"] == "suspected"
        and p46["verdict"]["status"] == "regressed"
    )
    assert (
        p46["verdict"]["band"]["low"]
        < p46["verdict"]["baseline_value"]
        < p46["verdict"]["band"]["high"]
    )
    assert p46["value"] > p46["verdict"]["band"]["high"]
    assert [c["at_run_id"] for c in d["subject_changes"]] == [
        rid(fx.TS, "x86_64", 30)
    ] and p30["subject_changed"]
    assert [c["at_run_id"] for c in d["env_changes"]] == [rid(fx.TS, "x86_64", 38)]
    p20 = by_night[rid(fx.TS, "x86_64", 20)]
    assert (
        p20["invalidation_state"] == "invalidated"
        and p20["verdict"].get("excluded") is True
    )
    # schema 85: the invalidated run is out of every baseline, so nothing lists it
    assert not any(
        p["verdict"].get("baseline_includes_invalidated") for p in d["points"]
    )
    assert rid(fx.TS, "x86_64", 20) not in {
        b
        for v in [by_night[rid(fx.TS, "x86_64", 30)]]
        for b in v["verdict"].get("baseline_includes_invalidated", [])
    }
    kept = api.series(
        db,
        bid(db, MATMUL),
        NOW,
        metric="total_duration_ms",
        arch="x86_64",
        days=100,
        include_invalid=False,
    )
    assert kept["data"]["count"] == 59
    assert r["meta"]["series_options"][0]["metric"] in (
        "total_duration_ms",
        "kernel_mean_ms",
    )


def test_series_errors_and_the_ab_runs_stay_out_of_the_nightly_series(db):
    with pytest.raises(api.ApiError) as e:
        api.series(db, bid(db, fx.TP4_TRACE), NOW)
    assert e.value.status == 422 and e.value.code == "not_a_measurement_series"
    with pytest.raises(api.ApiError) as e:
        api.series(db, "00000000-0000-0000-0000-000000000001", NOW)
    assert e.value.status == 404
    ab = api.series(db, bid(db, fx.AB_BENCH), NOW, days=100)
    assert (
        ab["data"]["series"]["source_job"] == "synth-ab-perf-x86_64"
        and ab["data"]["count"] == 20
    )


def test_series_without_stored_verdict_is_pending(db):
    b = bid(db, "synth_gelu_m")
    r = api.series(
        db,
        b,
        NOW,
        metric="total_duration_ms",
        arch="s390x",
        days=100,
        t_to="2026-09-20T00:00:00Z",
        t_from="2026-09-10T00:00:00Z",
    )
    assert r["data"]["count"] == 4
    assert all(p["verdict"]["status"] in ("no_baseline",) for p in r["data"]["points"])


def test_run_detail_repeats_cv_verdicts_and_layer_availability(db):
    r = api.run_detail(db, rid(fx.TS, "x86_64", 46), NOW, limit=5)["data"]
    assert (
        r["run"]["arch"] == "x86_64"
        and r["run"]["subject"]["id12"]
        and r["run"]["invalidation_state"] == "valid"
    )
    assert (
        r["run"]["validity"] == "valid"
        and r["run"]["validity_reasons_source"] == "derived"
        and r["run"]["validity_reasons"] == ["shared_env"]
    )
    b = r["benchmarks"][0]
    m = (
        b["metrics"]["total_duration_ms"]
        if "total_duration_ms" in b["metrics"]
        else next(iter(b["metrics"].values()))
    )
    assert (
        m["n_repeats"] == 3
        and len(m["repeats"]) == 3
        and m["cv_pct"] > 0
        and b["verdicts"]
    )
    assert r["availability"] == {
        **r["availability"],
        "environment": True,
        "placements": True,
        "steps": False,
        "spans": False,
        "verdicts": True,
    }
    full = api.run_detail(db, rid(fx.TS, "x86_64", 46), NOW, limit=50)
    assert full["meta"]["total"] == 28 and full["meta"]["next_cursor"] is None
    mm = next(b for b in full["data"]["benchmarks"] if b["name"] == MATMUL)
    assert mm["verdicts"]["total_duration_ms"]["status"] == "regressed"


def test_run_detail_states_invalid_runs_and_partial_invalidation(db):
    bad = api.run_detail(db, rid(fx.TS, "x86_64", 20), NOW)["data"]["run"]
    assert (
        bad["invalidation_state"] == "invalidated"
        and bad["invalidations"][0]["reason_code"] == "bad_card"
    )
    assert "invalidated" in bad["validity_reasons"]
    part = api.run_detail(db, rid(fx.TS, "ppc64le", 10), NOW)["data"]["run"]
    assert part["invalidation_state"] == "partial" and {
        h["repeat"] for h in part["invalidations"]
    } == {1}
    sus = api.run_detail(db, rid(fx.TS, "x86_64", 52), NOW)["data"]["run"]
    assert sus["invalidation_state"] == "suspect"


def test_run_detail_unknown_run_is_404(db):
    with pytest.raises(api.ApiError) as e:
        api.run_detail(db, "00000000-0000-0000-0000-000000000002", NOW)
    assert e.value.status == 404


def test_environment_fingerprint_node_qualification_and_placement(db):
    e = api.environment(db, rid(fx.TS, "x86_64", 46), NOW)["data"]
    assert e["node_class"] == "synth-xeon-2s" and e["pinning_label"] == "none"
    assert (
        e["fingerprint"]["values"]["device.firmware"] == "fw-1.1"
        and e["labels"]["node_class"] == "synth-xeon-2s"
    )
    assert e["qualification"]["in_force_at_run"] is True
    assert [p["repeat"] for p in e["placements"]] == [1, 2, 3] and e["placements"][0][
        "devices"
    ][0]["serial"].startswith("SYN-")
    assert e["placements"][0]["pinning"]["policy"] == "none" and e["clean"]["run"][
        "run.clean.exclusive"
    ] in ("observed", "none")
    assert e["request"]["name"] == "synth-env-synth-xeon-2s"
    assert (
        len(next(iter(e["nodes"].values()))["env_history"]) == 2
    )  # fw-1.0 then fw-1.1
    z = api.environment(db, rid(fx.TS, "s390x", 5), NOW)["data"]
    assert (
        z["qualification"]["in_force_at_run"] is False
        and z["qualification"]["entries"][0]["outcome"] == "fail"
    )


def test_environment_without_capture_flags_what_is_missing(db):
    with pytest.raises(api.ApiError):
        api.environment(db, "00000000-0000-0000-0000-000000000003", NOW)


def test_steps_headline_lag_outliers_and_rank_coverage(db):
    d = api.steps(db, TP4, NOW)["data"]
    assert (
        d["ranks"] == [0, 1, 2, 3]
        and len(d["steps"]) == 64
        and d["steps"][0] == "ttft"
        and d["steps"][1] == "itl_0001"
    )
    assert d["roles"]["total"] == "step_ms" and d["signal"] == "step_ms"
    itl = d["headline"]["itl"]
    assert (
        itl["slowest_rank"] == 2
        and itl["rank_skew"] > 3
        and itl["rule"] == "max"
        and itl["rule_declared"]
    )
    assert (
        d["headline"]["ttft"]["value"] is None
        and d["headline"]["ttft"]["n_warm_steps"] == 0
    )  # the only prefill is cold
    assert d["rank_coverage"] == {"present": 4, "world_size": 4, "complete": True}
    assert d["cold"][0] and not any(d["cold"][1:])
    stalls = {(o["rank"], o["step"]) for o in d["outliers"]}
    assert (
        (1, "itl_0012") in stalls
        and (1, "itl_0047") in stalls
        and (0, "itl_0030") in stalls
    )
    assert all(row[0] == 0 or row[0] is not None for row in d["lag"])
    assert d["awaited_rank"][5] == 2  # the slow card waits least
    assert d["consistency"] and abs(d["consistency"][0]["rel_diff_pct"]) < 0.01


def test_steps_no_data_and_unknown_signal(db):
    r = api.steps(db, rid(fx.TS, "x86_64", 46), NOW)
    assert r["meta"]["no_data"] is True
    with pytest.raises(api.ApiError) as e:
        api.steps(db, TP4, NOW, signal="nope")
    assert e.value.code == "unknown_signal"
    with pytest.raises(api.ApiError) as e:
        api.steps(db, TP4, NOW, max_rows=100)
    assert e.value.status == 413


def test_spans_summary_slow_rank_timeline_and_filters(db):
    s = api.spans(db, TP4, NOW)["data"]
    assert s["available"] == {
        "spans": 4 * 6 * 8 * 40,
        "ranks": 4,
        "steps": 8,
        "lanes": ["collective", "device", "host"],
    }
    rs = s["rank_summary"]
    assert (
        rs["slowest_device_rank"] == 2
        and rs["shortest_collective_rank"] == 2
        and rs["longest_collective_rank"] != 2
    )
    assert (
        len(s["collective_by_layer"]) == 4 * 40
        and s["kernels"][0]["total_ns"] >= s["kernels"][-1]["total_ns"]
    )
    assert {k["lane"] for k in s["kernels"]} == {
        "collective",
        "device",
        "host",
    } and len(s["kernels"]) == 6
    t = api.spans(db, TP4, NOW, view="timeline", rank=2, step=3)
    spans = t["data"]["spans"]
    assert len(spans) == 40 * 6 and [x["start_ns"] for x in spans] == sorted(
        x["start_ns"] for x in spans
    )
    c = api.spans(db, TP4, NOW, view="kernels", lane="collective", rank=1)["data"]
    assert {k["lane"] for k in c["kernels"]} == {"collective"} and len(
        c["kernels"]
    ) == 2
    with pytest.raises(api.ApiError) as e:
        api.spans(db, TP4, NOW, view="timeline")
    assert e.value.status == 400
    assert api.spans(db, rid(fx.TS, "x86_64", 46), NOW)["meta"]["no_data"] is True


def test_spans_of_an_invalidated_run_are_gone(db):
    db.exec(
        "INSERT INTO result_invalidations (ts, invalidation_id, scope_kind, run_id, state, reason_code, actor) "
        f"VALUES ('2026-11-15 00:00:00', '{fx.uid('inv', 'tp4')}', 'run', '{TP4}', 'invalidated', 'bad_trace', 'amy@example.invalid')"
    )
    with pytest.raises(api.ApiError) as e:
        api.spans(db, TP4, NOW)
    assert e.value.status == 410
    assert (
        api.run_detail(db, TP4, NOW)["data"]["run"]["invalidation_state"]
        == "invalidated"
    )
    db.exec(
        "INSERT INTO result_invalidations (ts, invalidation_id, scope_kind, run_id, state, reason_code, actor) "
        f"VALUES ('2026-11-15 00:00:01', '{fx.uid('inv', 'tp4')}', 'run', '{TP4}', 'restored', 'bad_trace', 'amy@example.invalid')"
    )
    assert api.spans(db, TP4, NOW)["data"]["available"]["spans"] > 0


def test_compare_runs_fairness_flags(db):
    same = api.compare(
        db,
        NOW,
        a_run=rid(fx.TS, "x86_64", 44),
        b_run=rid(fx.TS, "x86_64", 46),
        benchmark_id=bid(db, MATMUL),
    )["data"]
    row = next(r for r in same["rows"] if r["metric"] == "total_duration_ms")
    assert (
        9 < row["delta_pct"] < 15
        and row["within_noise"] is False
        and same["comparable"]["subject"]
        and same["fair"]
    )
    across = api.compare(
        db, NOW, a_run=rid(fx.TS, "x86_64", 29), b_run=rid(fx.TS, "x86_64", 30)
    )["data"]
    assert (
        across["comparable"]["subject"] is False
        and across["a"]["subject"]["build"] == "build-a"
    )
    byenv = api.compare(
        db, NOW, a_run=rid(fx.TS, "x86_64", 37), b_run=rid(fx.TS, "x86_64", 39)
    )["data"]
    assert byenv["comparable"]["env_id"] is False and byenv["fair"] is False
    bysub = api.compare(db, NOW, a_subject=db(f"SELECT toString(artifact_id) AS a FROM artifacts WHERE props['build'] = 'build-a' AND arch = 'x86_64' AND component = '{fx.TS}' LIMIT 1")[0]["a"],
                        b_run=rid(fx.TS, "x86_64", 30), arch="x86_64")["data"]  # fmt: skip
    assert bysub["a"]["resolved"]["by"] == "subject"


def test_derive_decode_per_token_and_refusals(db):
    ab_runs = [
        r["r"]
        for r in db(
            "SELECT DISTINCT toString(run_id) AS r FROM benchmark_runs WHERE props['run.suite.name'] = 'synth-pinning-ab' ORDER BY r LIMIT 50"
        )
    ]
    d = api.compare(
        db,
        NOW,
        kind="derive",
        run_ids=ab_runs[:3],
        n_benchmark=bid(db, fx.AB_BENCH),
        one_benchmark=bid(db, fx.AB_BENCH_1),
        n_tokens=256,
    )["data"]
    assert len(d["results"]) == 3 and not d["refused"]
    for x in d["results"]:
        assert 7 < x["value"] < 11 and x["per_repeat"] and x["env_id"]
    mixed = api.compare(
        db,
        NOW,
        kind="derive",
        run_ids=[ab_runs[0], rid(fx.TS, "x86_64", 46)],
        n_benchmark=bid(db, fx.AB_BENCH),
        one_benchmark=bid(db, fx.AB_BENCH_1),
        n_tokens=256,
    )["data"]
    assert [x["reason"] for x in mixed["refused"]] == ["missing_case"] and len(
        mixed["results"]
    ) == 1
    with pytest.raises(api.ApiError) as e:
        api.compare(
            db,
            NOW,
            kind="derive",
            run_ids=[rid(fx.TS, "x86_64", 46)],
            n_benchmark=bid(db, fx.AB_BENCH),
            one_benchmark=bid(db, fx.AB_BENCH_1),
            n_tokens=256,
        )
    assert e.value.status == 422 and e.value.code == "derive_refused"


def test_pinned_vs_unpinned_effect_table(db):
    d = api.compare(
        db,
        NOW,
        kind="effect",
        benchmark_id=bid(db, fx.AB_BENCH),
        metric="latency_ms",
        arch="x86_64",
    )["data"]
    eff = {e["node_class"]: e for e in d["effects"]}
    assert set(eff) == {"synth-xeon-2s", "synth-xeon-1s"}
    two, one = eff["synth-xeon-2s"], eff["synth-xeon-1s"]
    assert (
        -22 < two["effect_pct"] < -14
        and two["within_noise"] is False
        and two["better_pct"] > 14
        and two["n"] == 5
        and two["n_baseline"] == 5
    )
    assert abs(one["effect_pct"]) < 2 and one["within_noise"] is True
    assert d["groups"]["synth-xeon-2s"]["none"]["runs"] == 5


def test_references_list_values_and_compare(db):
    lst = api.references(db, NOW)["data"]["items"]
    assert (
        len(lst) == 1
        and lst[0]["state"] == "current"
        and lst[0]["comparability_verdict"] == "partial"
    )
    det = api.references(db, NOW, key=lst[0]["key"])["data"]
    assert {v["name"] for v in det["values"]} == {fx.AB_BENCH, fx.AB_BENCH_1} and det[
        "reference"
    ]["submitted_by"]
    assert api.references(db, NOW, state="retired")["meta"]["no_data"] is True
    ab_run = db(
        "SELECT toString(run_id) AS r FROM benchmark_runs WHERE props['run.suite.name'] = 'synth-pinning-ab' LIMIT 1"
    )[0]["r"]
    c = api.compare(
        db, NOW, kind="reference", run_id=ab_run, key=lst[0]["key"], metric="latency_ms"
    )["data"]
    assert len(c["rows"]) == 2 and all(r["ours"] for r in c["rows"])
    line = api.series(
        db, bid(db, fx.AB_BENCH), NOW, days=100, reference_key=lst[0]["key"]
    )["data"]["reference"]
    assert line["value"] == 2300.0 and line["state"] == "current"
    with pytest.raises(api.ApiError) as e:
        api.references(db, NOW, key="nope")
    assert e.value.status == 404


def test_reference_rows_never_reach_the_measured_layers(db):
    ref_run = db(
        "SELECT toString(run_id) AS r FROM benchmark_runs WHERE props['run.ref_only'] = '1' LIMIT 1"
    )[0]["r"]
    assert api.run_detail(db, ref_run, NOW)["data"]["benchmarks"] == []
    assert not db(
        f"SELECT 1 FROM benchmark_metric_verdicts WHERE run_id = '{ref_run}' LIMIT 1"
    )


def test_route_dispatch_and_errors(db):
    code, body = api.route(db, "/api/v2/perf/overview", {"arch": "x86_64"}, NOW)
    assert code == 200 and body["data"]["tiles"]
    code, body = api.route(
        db,
        f"/api/v2/perf/benchmarks/{bid(db, MATMUL)}/series",
        {"from": "2026-10-01T00:00:00Z", "metric": "total_duration_ms"},
        NOW,
    )
    assert code == 200 and body["data"]["count"] > 0
    assert api.route(db, "/api/v2/perf/runs/not-a-uuid", {}, NOW)[0] == 400
    assert api.route(db, "/api/v2/perf/nope", {}, NOW)[0] == 404
    assert (
        api.route(db, "/api/v2/perf/suites", {"days": "400"}, NOW)[1]["error"]["code"]
        == "bad_parameter"
    )
    assert api.route(db, "/api/v2/perf/overview", {"as_of": "yesterday"}, NOW)[0] == 400
    code, body = api.route(db, f"/api/v2/perf/runs/{TP4}/steps", {}, NOW)
    assert code == 200 and body["meta"]["cache"]["etag"]


def test_fixture_is_deterministic_and_has_the_declared_shape():
    a, b = fx.build(), fx.build()
    assert a == b
    s = fx.stats(a)
    assert s["tables"]["result_invalidations"] == 3 and s["runs"] == 4 * 60 + 20 + 1 + 1
    assert (
        len(
            {
                r["props"]["arch"]
                for r in a["benchmark_runs"]
                if r["props"].get("run.tier") == "nightly"
            }
        )
        == 3
    )
    assert (
        len(
            {
                r["component"]
                for r in a["benchmark_runs"]
                if r["props"].get("run.tier") == "nightly"
            }
        )
        == 2
    )
    assert (
        len(
            [
                b
                for b in a["benchmarks"]
                if b["props"]["record_type"] in ("op", "serving")
            ]
        )
        >= 40
    )


def test_compare_sides_resolved_by_environment(db):
    envs = [
        r["e"]
        for r in db(
            f"SELECT DISTINCT props['env.id'] AS e FROM benchmark_runs WHERE benchmark_id = '{bid(db, MATMUL)}' AND props['arch'] = 'x86_64' ORDER BY e LIMIT 10"
        )
    ]
    assert len(envs) == 2
    d = api.compare(
        db,
        NOW,
        a_env=envs[0],
        b_env=envs[1],
        benchmark_id=bid(db, MATMUL),
        **{"from": "2026-09-01T00:00:00Z"},
    )["data"]
    assert (
        d["a"]["resolved"]["by"] == "environment" and d["comparable"]["env_id"] is False
    )


def test_an_invalidated_repeat_leaves_the_aggregates_but_not_the_listing(db):
    ppc = rid(fx.TS, "ppc64le", 10)  # the node window covers repeat 1 only
    run = api.run_detail(db, ppc, NOW, limit=200)["data"]
    b = next(x for x in run["benchmarks"] if x["name"] == "synth_add_s")
    m = next(iter(b["metrics"].values()))
    assert (
        b["repeats"] == [1, 2, 3]
        and b["invalidated_repeats"] == [1]
        and m["n_repeats"] == 2
    )
    pt = next(
        p
        for p in api.series(
            db,
            bid(db, "synth_add_s"),
            NOW,
            metric="total_duration_ms",
            arch="ppc64le",
            days=100,
        )["data"]["points"]
        if p["run_id"] == ppc
    )
    assert pt["n_repeats"] == 2 and pt["invalidation_state"] == "partial"
    cmp = api.compare(
        db,
        NOW,
        a_run=ppc,
        b_run=rid(fx.TS, "ppc64le", 11),
        benchmark_id=bid(db, "synth_add_s"),
    )["data"]
    assert (
        cmp["rows"][0]["a"]["n_repeats"] == 2 and cmp["rows"][0]["b"]["n_repeats"] == 3
    )


def test_a_forged_cursor_is_a_400_and_boolean_flags_parse(db):
    assert api.route(db, "/api/v2/perf/suites", {"cursor": "W10"}, NOW)[0] == 400
    assert api.route(db, "/api/v2/perf/suites", {"cursor": "!!"}, NOW)[0] == 400
    assert api.route(db, "/api/v2/perf/suites", {"limit": "x"}, NOW)[0] == 400
    assert (
        api.flag("False") is False
        and api.flag("no") is False
        and api.flag(None) is True
        and api.flag("1") is True
    )


def test_steps_consistency_has_one_entry_per_rank(db):
    c = api.steps(db, TP4, NOW)["data"]["consistency"]
    assert sorted(x["rank"] for x in c) == [0, 1, 2, 3]


def test_steps_headline_and_outliers_from_the_views_equal_the_python_derivation(db):
    via_views = api.steps(db, TP4, NOW)["data"]
    via_python = api.steps(db, TP4, NOW, _views=False)["data"]
    assert via_views["headline"] == via_python["headline"]
    assert via_views["outliers"] == via_python["outliers"] and via_views["outliers"]


def test_spans_from_the_fast_table_equal_the_view(db, monkeypatch):
    assert api._span_source(db, TP4) == "v_bench_spans_fast"
    fast = [
        api.spans(db, TP4, NOW, view=v, rank=2, step=3)["data"]
        for v in ("summary", "kernels", "timeline")
    ]
    monkeypatch.setattr(api, "_span_source", lambda run, rid: "v_bench_spans")
    slow = [
        api.spans(db, TP4, NOW, view=v, rank=2, step=3)["data"]
        for v in ("summary", "kernels", "timeline")
    ]
    assert fast == slow and fast[2]["spans"]


def test_span_source_falls_back_only_when_the_optional_table_is_missing():
    def missing(sql, params=None):
        raise RuntimeError("Unknown table expression identifier 'bench_span_values'")

    def transient(sql, params=None):
        raise RuntimeError("Connection reset by peer")

    assert api._span_source(missing, TP4) == "v_bench_spans"
    with pytest.raises(RuntimeError):
        api._span_source(transient, TP4)
