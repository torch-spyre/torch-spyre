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

"""perf_api anchor line and anchored status (series, overview) on chdb, over the synthetic fixture plus one accepted anchor."""

from datetime import timedelta

import pytest
from spyre_clickhouse_ingest import perf_api as api
from spyre_clickhouse_ingest import perf_fixture as fx
from test_perf_api import MATMUL, NOW, ChdbDb, bid, rid

pytest.importorskip("chdb.session")

KEY = "perf.synth.anchor"
ART = fx.uid("anchor", "artifact")
RUN = fx.uid("anchor", "run")
ACCEPT_NIGHT = 40


@pytest.fixture(scope="module")
def db():
    d = ChdbDb().prepare()
    src = d(
        "SELECT benchmark_id, component, backend, measurements FROM benchmark_runs "
        f"WHERE run_id = toUUID('{rid(fx.TS, 'x86_64', ACCEPT_NIGHT)}') AND benchmark_id = toUUID('{bid(d, MATMUL)}')"
    )
    samples = {}
    for r in src:
        for k, v in r["measurements"].items():
            samples.setdefault(
                (str(r["benchmark_id"]), r["component"], r["backend"], k), []
            ).extend(v)
    rows = [
        {
            "ts": fx.fmt(fx.night(ACCEPT_NIGHT)), "run_id": RUN, "benchmark_id": b, "component": c, "backend": be,
            "measurements": {m: v}, "iterations": len(v),
            "props": {"run.ref_only": "1", "run.record_type": "reference", "ref.key": KEY},
        }
        for (b, c, be, m), v in samples.items()
    ]  # fmt: skip
    ts = fx.fmt(fx.night(ACCEPT_NIGHT) + timedelta(hours=3))
    props = {
        "ref.key": KEY, "ref.version": "1", "ref.class": "perf", "ref.label": "Synthetic", "ref.anchor": "1",
        "ref.approved_by": "amy@example.invalid", "ref.approved_at": ts,
        "ref.anchor.from_run_id": rid(fx.TS, "x86_64", ACCEPT_NIGHT), "ref.anchor.as_of": fx.fmt(fx.night(ACCEPT_NIGHT)),
    }  # fmt: skip
    d.insert("benchmark_runs", rows)
    d.insert("artifacts", [{"ts": ts, "artifact_id": ART, "component": "reference", "arch": "x86_64", "kind": "generic",
                            "artifact_name": KEY, "origin": "built", "props": props}])  # fmt: skip
    d.insert("result_uploads", [{"ts": ts + ".000", "upload_id": fx.uid("anchor", "upload"), "state": "published", "actor": "amy@example.invalid",
                                 "artifact_id": ART, "run_id": RUN, "format": "benchmarks", "result_kind": "performance", "test_type": "perf",
                                 "arch": "x86_64", "filename": "r.json", "size_bytes": 1, "content_sha256": "0", "bundle_url": ""}])  # fmt: skip
    d.insert("artifact_tags", [{"ts": ts, "tag": "ref-current", "tag_family": "reference", "artifact_id": ART, "refs": [],
                                "published_refs": [], "props": {"ref.key": KEY}}])  # fmt: skip
    yield d
    d.close()


def points(db, **kw):
    r = api.series(
        db, bid(db, MATMUL), NOW, metric="total_duration_ms", arch="x86_64", **kw
    )
    return r, {p["run_id"]: p for p in r["data"]["points"]}


def test_series_carries_the_anchor_line_and_each_points_anchored_status(db):
    r, pts = points(db)
    a = r["data"]["anchor"]
    assert (
        a["state"] == "anchored"
        and a["version"] == "1"
        and a["ref_key"] == KEY
        and a["accepted_by"] == "amy@example.invalid"
    )
    assert (
        a["from_run_id"] == rid(fx.TS, "x86_64", ACCEPT_NIGHT)
        and a["band"]["low"] < a["value"] < a["band"]["high"]
    )
    assert pts[rid(fx.TS, "x86_64", 30)]["anchored"]["status"] == "no_anchor"
    assert pts[rid(fx.TS, "x86_64", 44)]["anchored"]["status"] == "unchanged"
    first = pts[rid(fx.TS, "x86_64", 45)]["anchored"]
    assert (first["status"], first["reason"], first["n_exceeding_runs"]) == (
        "suspected",
        "single_run",
        1,
    )
    last = pts[rid(fx.TS, "x86_64", 60)]["anchored"]
    assert (
        last["status"],
        last["reason"],
        last["n_exceeding_runs"],
        last["days_exceeding"],
    ) == ("regressed", "anchor_regressed", 15, 15)
    sus = pts[rid(fx.TS, "x86_64", 52)][
        "anchored"
    ]  # a suspect run: judged and capped, neither extends nor breaks the streak
    assert (sus["status"], sus["reason"]) == ("suspected", "suspect_data")
    assert last["since_ts"].startswith(fx.fmt(fx.night(45))[:10])
    assert a["latest"]["status"] == "regressed" and a["persistent"] is True
    assert r["data"]["summary"]["last_anchored_status"] == "regressed"


def test_a_series_without_an_anchor_says_none(db):
    r = api.series(
        db, bid(db, "synth_add_s"), NOW, metric="total_duration_ms", arch="x86_64"
    )
    assert r["data"]["anchor"] == {"state": "none"}
    assert r["data"]["points"] and all(
        p["anchored"] is None for p in r["data"]["points"]
    )


def test_overview_counts_anchored_series_and_lists_the_persistent_regressions(db):
    d = api.overview(db, NOW)["data"]
    tile = {(t["arch"], t["component"]): t for t in d["tiles"]}[("x86_64", fx.TS)]
    assert tile["anchored"]["regressed"] == 1 and tile["anchored"]["no_anchor"] == 27
    other = {(t["arch"], t["component"]): t for t in d["tiles"]}[("ppc64le", fx.TS)]
    assert other["anchored"] == {
        "regressed": 0,
        "suspected": 0,
        "improved": 0,
        "unchanged": 0,
        "no_anchor": 28,
    }
    (p,) = [
        x for x in d["persistent_regressions"] if x["metric"] == "total_duration_ms"
    ]
    assert (
        p["name"] == MATMUL
        and p["n_exceeding_runs"] == 15
        and p["anchor_version"] == "1"
        and 11 < p["worse_pct"] < 13
    )


def test_overview_as_of_the_early_regression_shows_the_streak_so_far(db):
    d = api.overview(db, as_of=fx.night(47) + timedelta(hours=4))["data"]
    (p,) = [
        x for x in d["persistent_regressions"] if x["metric"] == "total_duration_ms"
    ]
    assert p["n_exceeding_runs"] == 3 and p["days_exceeding"] == 2


def test_a_database_without_schema_88_degrades_to_a_note():
    def run(sql, params=None):
        if "v_benchmark_anchor" in sql:
            raise RuntimeError(
                "Code: 60. DB::Exception: Unknown table expression identifier 'v_benchmark_anchor' (UNKNOWN_TABLE)"
            )
        return []

    assert api.anchored_read(run, "SELECT 1 FROM v_benchmark_anchor", {}) is None
    with pytest.raises(RuntimeError):
        api.anchored_read(
            lambda s, p=None: (_ for _ in ()).throw(RuntimeError("boom")),
            "SELECT 1",
            {},
        )
    tiles, persistent, note = api._overview_anchored(run, {}, "")
    assert (tiles, persistent) == ({}, []) and "schema 88" in note[0]


def test_anchor_queries_are_bounded(db):
    db.queries.clear()
    points(db)
    api.overview(db, NOW)
    anchored = [q for q in db.queries if "v_benchmark_anchor" in q]
    assert anchored and all(
        any(b in q for b in ("LIMIT", "{t_from:")) for q in anchored
    )


def test_a_newest_run_without_an_anchored_row_is_unknown_not_the_run_before():
    def run(sql, params=None):
        if "FROM v_benchmark_anchored_verdicts" in sql:
            return [{"run_id": "r1", "status": "regressed", "reason": "anchor_regressed", "anchor_value": 10.0, "delta_pct": 12.0,
                     "worse_pct": 12.0, "floor_pct": 3.0, "since_ts": None, "n_exceeding_runs": 3, "days_exceeding": 2, "version": "1"}]  # fmt: skip
        return [{"anchor_artifact_id": "a", "ref_key": "k", "version": "1", "anchor_from": None, "accepted_by": "", "accepted_at": "",
                 "from_run_id": "", "n_samples": 5, "anchor_value": 10.0, "anchor_mad": 0.1, "source_job": ""}]  # fmt: skip

    pts = [
        {"run_id": "r1", "invalidation_state": "valid"},
        {"run_id": "r2", "invalidation_state": "valid"},
    ]
    sel = {"arch": "x86_64", "backend": "b", "source_job": "j", "metric": "m"}
    line, _ = api._attach_anchor(
        run, pts, {"t_from": "x", "t_to": "y"}, "c", "bid", sel, None, None
    )
    assert pts[0]["anchored"]["status"] == "regressed" and pts[1]["anchored"] is None
    assert line["latest"] is None and line["persistent"] is False
    pts = [
        {"run_id": "r1", "invalidation_state": "valid"},
        {"run_id": "r2", "invalidation_state": "invalidated"},
    ]
    line, _ = api._attach_anchor(
        run, pts, {"t_from": "x", "t_to": "y"}, "c", "bid", sel, None, None
    )
    assert (
        line["latest"]["status"] == "regressed" and line["persistent"] is True
    )  # an invalidated newest point is skipped
