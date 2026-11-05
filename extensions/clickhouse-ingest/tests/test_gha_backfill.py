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

from datetime import datetime, timedelta, timezone

import pytest
from spyre_clickhouse_ingest import gha_backfill, gha_runs
from test_gha_runs import NOW, REPO, Clock, FakeAPI, Store, _run

HOUR = timedelta(hours=1)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(gha_runs.time, "time", lambda: c.now)
    monkeypatch.setattr(gha_runs.time, "sleep", c.sleep)
    return c


TO = NOW - timedelta(hours=7)


def _ts(minutes):
    return f"{TO + timedelta(minutes=minutes):%Y-%m-%dT%H:%M:%SZ}"


def _runs(n, status="completed", step=10):
    """n runs, one every `step` minutes from TO, id 1000 + i."""
    return [
        _run(
            id=1000 + i,
            created_at=_ts(step * i),
            run_started_at=_ts(step * i),
            updated_at=_ts(step * i),
            status=status,
            conclusion="success" if status == "completed" else None,
        )
        for i in range(n)
    ]


def _keys(i):
    return f"gha:{REPO}/{1000 + i}#1"


def _api(runs, clock, **kw):
    return FakeAPI(runs, limit=10000, remaining=10000, clock=clock, **kw)


def _fill(api, store, workers=1, top=NOW, to=TO):
    return gha_backfill.backfill(api, store, "db", REPO, top, to, HOUR, workers)


def test_slices_run_newest_first_on_hour_edges():
    out = list(gha_backfill.slices(NOW, NOW - timedelta(hours=2), HOUR))
    assert [s.lo for s in out] == [NOW - HOUR, NOW - 2 * HOUR]
    assert [s.hi for s in out] == [NOW, NOW - HOUR]


def test_a_slice_is_done_only_when_every_run_is_stored_and_finished(clock):
    api, store = _api(_runs(6, status="in_progress"), clock), Store()
    (open_,) = _fill(api, store, to=TO, top=TO + HOUR)
    assert open_.fetched == 6 and not open_.done
    assert len(store.keys()) == 6


def test_a_finished_slice_is_done_and_a_repeat_fetches_nothing(clock):
    runs, store = _runs(6), Store()
    first = _fill(_api(runs, clock), store, to=TO, top=TO + HOUR)
    assert [s.done for s in first] == [True] and first[0].fetched == 6
    api = _api(runs, clock)
    again = _fill(api, store, to=TO, top=TO + HOUR)
    assert again[0].done and again[0].fetched == 0
    assert not any("/jobs" in p for p in api.paths)


def test_an_interrupted_backfill_resumes_to_the_same_rows(clock):
    runs = _runs(40)
    whole = Store()
    _fill(_api(runs, clock), whole)
    store = Store()
    cut = _api(runs, clock, deadline=clock.now + 20 * 60)
    part = _fill(cut, store)
    assert any(s.reason == "deadline reached" for s in part)
    assert 0 < len(store.keys()) < len(whole.keys())
    _fill(_api(runs, clock), store)
    assert sorted(store.keys()) == sorted(whole.keys())


def test_the_forward_poll_leaves_the_backlog_a_hole_and_backfill_closes_it(clock):
    runs = _runs(40)
    store = Store()
    newest = datetime.fromisoformat(runs[-1]["created_at"].replace("Z", "+00:00"))
    # The forward cursor jumped to the present: only the newest run is stored.
    gha_runs.poll(FakeAPI(runs[-1:], clock=clock), store, "db", REPO, newest, NOW)
    mark = gha_runs.watermark(store, "db", REPO)
    assert mark == newest and len(store.keys()) == 1
    start, _ = gha_runs.window_start(
        NOW, timedelta(hours=1), mark, timedelta(hours=1), timedelta(days=90)
    )
    assert start > TO
    _fill(_api(runs, clock), store)
    assert sorted(store.keys()) == sorted(_keys(i) for i in range(40))
    assert gha_runs.watermark(store, "db", REPO) == mark


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_slices_store_the_same_rows_as_sequential(clock, workers):
    runs = _runs(40)
    seq, par = Store(), Store()
    _fill(_api(runs, clock), seq, workers=1)
    out = _fill(_api(runs, clock), par, workers=workers)
    assert all(s.done for s in out)
    key = lambda r: (r["run_key"], r["state"], str(r["updated_at"]))  # noqa: E731
    assert sorted(map(key, par.rows)) == sorted(map(key, seq.rows))


def test_a_rate_stop_leaves_a_prefix_of_complete_slices(clock):
    runs = _runs(40)
    store = Store()
    api = FakeAPI(runs, remaining=90, clock=clock)
    out = _fill(api, store, workers=1)
    assert any(s.reason.startswith("budget reached") for s in out)
    edge = gha_backfill.complete_down_to(out, NOW)
    assert NOW > edge > TO
    stored_ids = {k for k in store.keys()}
    want = {
        _keys(i)
        for i, r in enumerate(runs)
        if datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) >= edge
    }
    assert want <= stored_ids
    assert [s.done for s in out if s.hi > edge][-1] is True


def test_complete_down_to_stops_at_the_first_open_slice():
    n = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
    s = [
        gha_backfill.Slice(n - HOUR, n, done=True),
        gha_backfill.Slice(n - 2 * HOUR, n - HOUR, done=False),
        gha_backfill.Slice(n - 3 * HOUR, n - 2 * HOUR, done=True),
    ]
    assert gha_backfill.complete_down_to(s, n) == n - HOUR


def test_a_secondary_limit_403_waits_retry_after():
    class Err:
        code = 403
        headers = {"Retry-After": "30", "X-RateLimit-Remaining": "4000"}

    assert gha_runs.GitHub._backoff(Err, 0) == 30.0


def test_requests_are_spaced_by_min_interval(clock):
    api = _api(_runs(1), clock, min_interval=2.0)
    for _ in range(3):
        api._pace()
    assert clock.now - NOW.timestamp() >= 4.0
