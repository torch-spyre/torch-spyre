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

"""bench_invalidation on real chdb: preview, the write rules, ordering, and the bundle kind."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from bench_chdb import REPEATS, ChdbClient, Db, place
from bench_verdict_fixture import BENCHMARKS, Fixture, run_uuid
from spyre_clickhouse_ingest import bench_invalidation as bi
from spyre_clickhouse_ingest.offline import BundleError

pytest.importorskip("chdb.session")

ID1 = "018f0000-0000-7000-8000-000000000001"
ID2 = "018f0000-0000-7000-8000-000000000002"
KEY = ("op-alpha", "kernel_mean_ms")
SLOW = {"op-alpha": 0.5}
BAD = {"nodes": ["n3"], "serials": ["SNBAD"]}


@pytest.fixture
def world():
    d = Db()
    f = Fixture(d)
    f.history(22)
    for n in range(22):
        place(d, n, lambda n, rep: f"2026-10-{n + 1:02d} 02:{rep:02d}:00")
    yield f, ChdbClient(d)
    d.s.close()


def poison(f, count=4):
    """`count` consecutive slow runs on card SNBAD, then one healthy night after them."""
    runs = []
    for i in range(count):
        run = f.run(slow=SLOW)
        place(
            f.db,
            run,
            lambda r, rep: f"2026-11-{i + 1:02d} 02:{rep:02d}:00",
            pid="dddddddd-0000-5000-8000-000000000bad",
            **BAD,
        )
        runs.append(run)
    return runs, f.run()


def ev(state="invalidated", inv=ID1, kind="device", scope=None, **kw):
    scope = scope if scope is not None else {"device_serial": "SNBAD"}
    return {"invalidation_id": inv, "scope_kind": kind, "scope": scope, "state": state, "reason_code": "faulty_hardware",
            "reason": "card SNBAD ran hot", **kw}  # fmt: skip


def status(f, run):
    return {k: v[:2] for k, v in f.verdicts(run).items()}


def stored(c):
    return c.db.rows(
        "SELECT state, matched_runs, actor, props, ts FROM result_invalidations ORDER BY ts"
    )


# --- preview -----------------------------------------------------------------------------------


def test_preview_lists_runs_series_and_benchmarks(world):
    f, c = world
    runs, _ = poison(f)
    p = bi.preview(c, ev(), with_verdicts=False)
    assert p["matched_runs"] == 4 and p["matched_repeats"] == 4 * REPEATS
    assert {r["run_id"] for r in p["runs"]} == set(runs)
    assert len(p["series"]) == len(BENCHMARKS) and {s["runs"] for s in p["series"]} == {
        4
    }
    assert (
        len(p["benchmarks"]) == len(BENCHMARKS)
        and p["needs_confirm"] is False
        and not p["noop"]
    )


def actual(f, run, name, metric):
    return status(f, run).get((name, metric), ("absent",))[0]


def test_the_previewed_verdict_changes_are_the_ones_that_happen(world):
    f, c = world
    runs, nxt = poison(f, 10)
    all_runs = (*runs, nxt)
    before = {r: status(f, r) for r in all_runs}
    assert before[runs[-1]][KEY][0] == "regressed"
    _, median_before, _ = f.baseline(nxt)
    vc = bi.preview(c, ev())["verdict_changes"]
    assert vc["available"] and vc["total"] > 0, vc
    bi.apply(c, [ev()], actor="amy@example.com")
    changed = {
        (run, name, metric): actual(f, run, name, metric)
        for run in all_runs
        for (name, metric), (was, _) in before[run].items()
        if actual(f, run, name, metric) != was
    }
    assert {
        (r["run_id"], r["name"], r["metric"]): r["after"] for r in vc["rows"]
    } == changed
    assert (
        not f.verdicts(runs[0])
        and f.baseline(nxt)[1] < median_before
        and 9.5 < f.baseline(nxt)[1] < 10.5
    )


def test_the_previewed_restore_is_the_inverse(world):
    f, c = world
    runs, nxt = poison(f, 10)
    bi.apply(c, [ev()], actor="amy@example.com")
    p = bi.preview(c, ev(state="restored"))
    vc = p["verdict_changes"]
    assert p["state_before"] == "invalidated" and vc["available"] and vc["total"] > 0
    assert all(
        r["before"] == "absent" or r["run_id"] == nxt or r["before"] != r["after"]
        for r in vc["rows"]
    )
    bi.apply(c, [ev(state="restored")], actor="amy@example.com")
    assert f.verdicts(runs[0])
    assert {(r["run_id"], r["name"], r["metric"]): r["after"] for r in vc["rows"]} == {
        (r["run_id"], r["name"], r["metric"]): actual(
            f, r["run_id"], r["name"], r["metric"]
        )
        for r in vc["rows"]
    }


def test_a_noop_preview_says_so(world):
    f, c = world
    poison(f)
    bi.apply(c, [ev()], actor="amy@example.com")
    assert bi.preview(c, ev(), with_verdicts=False)["noop"] is True


# --- writing -----------------------------------------------------------------------------------


def test_apply_writes_one_row_with_the_actor_channel_and_recomputed_count(world):
    f, c = world
    poison(f)
    (r,) = bi.apply(
        c, [ev(matched_runs=999)], actor="amy@example.com", props={"upload_id": "u1"}
    )
    assert r["action"] == "write" and r["matched_runs"] == 4
    (row,) = stored(c)
    assert (
        row["state"] == "invalidated"
        and row["matched_runs"] == 4
        and row["actor"] == "amy@example.com"
    )
    assert row["props"] == {"channel": "api", "upload_id": "u1"}


def test_event_times_are_inserted_zone_aware(world):
    f, c = world
    poison(f)
    seen = []

    class Spy(ChdbClient):
        def insert(self, table, rows, column_names=None):
            seen.extend(rows)
            super().insert(table, rows, column_names)

    bi.apply(
        Spy(c.db),
        [
            ev(
                window_from="2026-10-01T00:00:00Z",
                window_to="2026-12-01T00:00:00Z",
                kind="node",
                scope={"node": "n3"},
            )
        ],
        actor="a",
    )
    ts, wf, wt = seen[0][0], seen[0][9], seen[0][10]
    assert all(
        d.tzinfo is not None and d.utcoffset() == timedelta(0) for d in (ts, wf, wt)
    )


def test_the_same_state_again_writes_nothing(world):
    f, c = world
    poison(f)
    bi.apply(c, [ev()], actor="a")
    (r,) = bi.apply(c, [ev()], actor="a")
    assert r["action"] == "noop" and len(stored(c)) == 1


def test_events_of_one_id_in_one_millisecond_still_resolve_in_order(world):
    f, c = world
    poison(f)
    same = "2026-12-01T00:00:00+00:00"
    bi.apply(
        c,
        [ev(ts=same), ev(state="restored", ts=same), ev(state="suspect", ts=same)],
        actor="a",
    )
    ts = [r["ts"] for r in stored(c)]
    assert ts == sorted(set(ts)) and len(ts) == 3
    assert c.db.rows(
        f"SELECT state FROM v_invalidation_state WHERE invalidation_id = '{ID1}'"
    ) == [{"state": "suspect"}]


def test_an_event_dated_before_the_latest_is_moved_after_it(world):
    f, c = world
    poison(f)
    bi.apply(c, [ev(ts="2030-01-01T00:00:00+00:00")], actor="a")
    bi.apply(c, [ev(state="restored", ts="2026-01-01T00:00:00+00:00")], actor="a")
    assert c.db.rows(
        f"SELECT state FROM v_invalidation_state WHERE invalidation_id = '{ID1}'"
    ) == [{"state": "restored"}]


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"reason": "  "}, "reason is required"),
        ({"reason_code": "because"}, "reason_code"),
        ({"state": "gone"}, "state"),
        ({"scope_kind": "query"}, "reserved"),
        ({"scope": {}}, "needs scope.device_serial"),
        ({"scope": {"device_serial": "x", "node": "n"}}, "does not take"),
        ({"window_from": "yesterday"}, "ISO-8601"),
        (
            {
                "window_from": "2026-02-01T00:00:00Z",
                "window_to": "2026-01-01T00:00:00Z",
            },
            "after window_to",
        ),
        ({"evidence_url": "file:///x"}, "http"),
        ({"invalidation_id": "nope"}, "UUID"),
    ],
)
def test_a_bad_event_is_refused_before_anything_is_written(world, bad, message):
    f, c = world
    with pytest.raises(bi.InvalidationError, match=message):
        bi.apply(c, [ev(), ev(**bad)], actor="a")
    assert stored(c) == []


def test_run_scope_takes_no_window_and_restore_needs_something_to_restore(world):
    f, c = world
    with pytest.raises(bi.InvalidationError, match="window applies"):
        bi.normalize(
            ev(
                kind="run",
                scope={"run_id": run_uuid(1)},
                window_from="2026-10-01T00:00:00Z",
            )
        )
    with pytest.raises(
        bi.InvalidationError, match="new invalidation cannot become restored"
    ):
        bi.apply(c, [ev(state="restored")], actor="a")


def test_a_later_event_must_repeat_the_scope(world):
    f, c = world
    poison(f)
    bi.apply(c, [ev()], actor="a")
    with pytest.raises(bi.InvalidationError, match="different scope"):
        bi.apply(c, [ev(state="restored", scope={"device_serial": "SN1"})], actor="a")


def test_a_big_scope_needs_a_confirmation_that_matches_the_preview(world):
    f, c = world
    poison(f, count=4)
    with pytest.raises(
        bi.InvalidationError, match="exceed the confirmation threshold 3"
    ):
        bi.apply(c, [ev()], actor="a", confirm_above=3)
    assert stored(c) == []
    token = bi.preview(c, ev(), limit=3, with_verdicts=False)["confirm_token"]
    with pytest.raises(bi.InvalidationError, match="token mismatch"):
        bi.apply(
            c, [ev(confirmed=True, confirm_token="0" * 12)], actor="a", confirm_above=3
        )
    bi.apply(c, [ev(confirmed=True, confirm_token=token)], actor="a", confirm_above=3)
    assert len(stored(c)) == 1


def test_a_run_added_after_the_preview_changes_the_token(world):
    f, c = world
    poison(f, count=2)
    token = bi.preview(c, ev(), limit=1, with_verdicts=False)["confirm_token"]
    run = f.run(slow=SLOW)
    place(
        f.db,
        run,
        lambda r, rep: "2026-11-20 02:00:00",
        pid="dddddddd-0000-5000-8000-000000000bad",
        **BAD,
    )
    with pytest.raises(bi.InvalidationError, match="token mismatch"):
        bi.apply(
            c, [ev(confirmed=True, confirm_token=token)], actor="a", confirm_above=1
        )


def test_restoring_a_big_scope_is_confirmed_too(world):
    f, c = world
    poison(f, count=4)
    bi.apply(c, [ev()], actor="a")
    with pytest.raises(bi.InvalidationError, match="confirmation threshold"):
        bi.apply(c, [ev(state="restored")], actor="a", confirm_above=3)


def test_dry_run_writes_nothing(world):
    f, c = world
    poison(f)
    (r,) = bi.apply(c, [ev()], actor="a", dry_run=True)
    assert r["action"] == "write" and stored(c) == []


def test_the_threshold_comes_from_the_environment(monkeypatch):
    assert bi.threshold() == 50
    monkeypatch.setenv(bi.THRESHOLD_ENV, "7")
    assert bi.threshold() == 7
    monkeypatch.setenv(bi.THRESHOLD_ENV, "x")
    with pytest.raises(bi.InvalidationError):
        bi.threshold()


# --- the bundle kind ---------------------------------------------------------------------------


def args(**kw):
    return SimpleNamespace(
        dry_run=False,
        uploader="amy@example.com",
        bundle_prop=[("upload_id", "u9")],
        **kw,
    )


def test_a_bundle_goes_through_ingest_and_is_idempotent(world, tmp_path):
    f, c = world
    poison(f)
    root = bi.write_bundle(tmp_path / "b", [ev()])
    assert bi.is_bundle(root)
    report = bi.ingest(root, args(), client=c)
    assert report["status"] == "ingested" and report["events"][0]["matched_runs"] == 4
    (row,) = stored(c)
    assert (
        row["props"]["channel"] == "bundle"
        and row["props"]["upload_id"] == "u9"
        and len(row["props"]["bundle_sha256"]) == 64
    )
    assert (
        bi.ingest(root, args(), client=c)["status"] == "duplicate"
        and len(stored(c)) == 1
    )


def test_a_bundle_dry_run_reports_and_writes_nothing(world, tmp_path):
    f, c = world
    poison(f)
    root = bi.write_bundle(tmp_path / "b", [ev()])
    a = args()
    a.dry_run = True
    assert bi.ingest(root, a, client=c)["status"] == "would-ingest" and stored(c) == []


def test_a_bundle_cannot_name_its_actor(world, tmp_path):
    f, c = world
    poison(f)
    root = bi.write_bundle(tmp_path / "b", [{**ev(), "actor": "mallory@example.com"}])
    bi.ingest(root, args(), client=c)
    assert stored(c)[0]["actor"] == "amy@example.com"


def test_a_tampered_or_stray_bundle_is_rejected(world, tmp_path):
    f, c = world
    root = bi.write_bundle(tmp_path / "b", [ev()])
    (root / "results" / "invalidation.json").write_text(
        json.dumps({"events": [ev(reason="other")]})
    )
    with pytest.raises(BundleError, match="sha256"):
        bi.ingest(root, args(), client=c)
    root = bi.write_bundle(tmp_path / "c", [ev()])
    (root / "results" / "extra.json").write_text("{}")
    with pytest.raises(BundleError, match="holds only"):
        bi.ingest(root, args(), client=c)


def test_a_bad_event_in_a_bundle_is_a_rejection_not_a_retry(world, tmp_path):
    f, c = world
    root = bi.write_bundle(tmp_path / "b", [ev(reason="")])
    with pytest.raises(BundleError) as err:
        bi.ingest(root, args(), client=c)
    assert err.value.code == 2


def test_results_from_bundle_dispatches_the_kind(world, tmp_path, monkeypatch):
    from spyre_clickhouse_ingest import bundle
    from spyre_clickhouse_ingest.client import ClickHouse

    f, c = world
    poison(f)
    root = bi.write_bundle(tmp_path / "b", [ev()])
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(ClickHouse, "connect", classmethod(lambda cls, **kw: c))
    report = bundle._ingest(
        root,
        args(trusted_job_prefix=[], expect_key="", bundle_url="", channel_trust="lab"),
    )
    assert report["status"] == "ingested" and len(stored(c)) == 1


def test_a_replayed_old_bundle_cannot_undo_a_restore(world, tmp_path):
    f, c = world
    poison(f)
    old = bi.write_bundle(tmp_path / "a", [ev()])
    bi.ingest(old, args(), client=c)
    bi.apply(c, [ev(state="restored")], actor="a")
    assert bi.ingest(old, args(), client=c)["status"] == "duplicate"
    assert c.db.rows(
        f"SELECT state FROM v_invalidation_state WHERE invalidation_id = '{ID1}'"
    ) == [{"state": "restored"}]


def test_two_events_of_a_new_id_cannot_change_scope(world):
    f, c = world
    poison(f)
    with pytest.raises(bi.InvalidationError, match="different scope"):
        bi.apply(
            c, [ev(state="suspect"), ev(scope={"device_serial": "SN1"})], actor="a"
        )
    assert stored(c) == []


def test_the_actor_is_never_the_bundles_uploader(world, tmp_path):
    f, c = world
    poison(f)
    root = bi.write_bundle(tmp_path / "b", [ev()], uploader="mallory")
    a = args()
    a.uploader = ""
    bi.ingest(root, a, client=c)
    assert stored(c)[0]["actor"] != "mallory"


def test_a_future_event_time_does_not_pin_the_clock(world):
    f, c = world
    poison(f)
    bi.apply(c, [ev(ts="2099-01-01T00:00:00+00:00")], actor="a")
    assert datetime.fromisoformat(stored(c)[0]["ts"]).year < 2099


def test_a_window_with_microseconds_keeps_its_scope(world):
    f, c = world
    poison(f)
    n = {
        "kind": "node",
        "scope": {"node": "n3"},
        "window_from": "2026-10-01T00:00:00.123456Z",
    }
    bi.apply(c, [ev(**n)], actor="a")
    bi.apply(c, [ev(state="restored", **n)], actor="a")


def test_a_big_scope_confirms_with_the_server_side_token(world):
    f, c = world
    poison(f)
    t1 = bi.preview(c, ev(), limit=1, with_verdicts=False)["confirm_token"]
    assert (
        t1 and t1 == bi.preview(c, ev(), limit=1, with_verdicts=False)["confirm_token"]
    )
