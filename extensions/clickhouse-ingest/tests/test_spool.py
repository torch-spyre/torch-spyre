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

"""Two-tier spool: drain-on-write, atomic claims, trust gate, caps, and the plain-table probe."""

import multiprocessing
import os
import signal
import threading

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import spool as sp
from spyre_clickhouse_ingest.resilience import Policy, resilient_insert

ROWS = [[i, f"r{i}"] for i in range(6)]
COLS = ["a", "b"]
FAST = dict(sleep=lambda s: None, log=lambda m: None)


def err(code, name="X"):
    return DatabaseError(f"Code: {code}", code=code, name=name)


class Cluster:
    """Replicated-shaped: a block is stored once per token. Plain: always stored."""

    def __init__(self, replicated=True, down=False, script=()):
        self.replicated, self.down, self.script = replicated, down, list(script)
        self.rows, self.tokens, self.calls = [], set(), []

    def insert(self, table, data, column_names=None, database=None, settings=None):
        tok = (settings or {}).get("insert_deduplication_token")
        self.calls.append((table, len(data), tok))
        if self.down:
            raise err(285, "TOO_FEW_LIVE_REPLICAS")
        if self.script:
            e = self.script.pop(0)
            if e:
                raise e
        if self.replicated and tok and tok in self.tokens:
            return
        self.rows.extend(map(tuple, data))
        if tok:
            self.tokens.add(tok)

    def query(self, sql, parameters=None):
        from types import SimpleNamespace

        return SimpleNamespace(result_rows=[[len(self.rows)]])


class FakeShared:
    """An in-memory stand-in for the Artifactory tier with an atomic claim."""

    def __init__(self, reachable=True):
        self.files, self.lock, self.reachable = {}, threading.Lock(), reachable
        self.ages = {}
        self.held = {}

    def _up(self):
        if not self.reachable:
            raise OSError("artifactory unreachable")

    def put(self, name, data):
        self._up()
        with self.lock:
            self.files[name] = data
            self.ages.setdefault(name, 0.0)

    def list(self):
        self._up()
        with self.lock:
            return [(n, self.ages.get(n, 0.0)) for n in sorted(self.files)]

    def claim(self, name, owner):
        self._up()
        with self.lock:
            if name not in self.files or name in self.held:
                return None
            self.held[name] = owner
            return name

    def get(self, name):
        self._up()
        return self.files[name]

    def delete(self, name):
        self._up()
        with self.lock:
            self.files.pop(name, None)
            self.held.pop(name, None)

    def release(self, name):
        self._up()
        with self.lock:
            self.held.pop(name, None)


def not_stored(c):
    """A probe for a plain table: an item is stored when its rows are already in the cluster."""

    def probe(item):
        have = sum(1 for r in item.rows if tuple(r) in set(c.rows))
        return have

    return probe


def spool_of(tmp_path, shared=None, trusted=True, **kw):
    return sp.TieredSpool(
        local=tmp_path / "local", shared=shared, trusted=trusted, **kw
    )


# ── drain on write ────────────────────────────────────────────────────────────


def test_drain_replays_pending_items_before_the_callers_own_write(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[100, "old"]], "tok-old")
    c = Cluster()
    resilient_insert(
        c,
        "t",
        ROWS,
        COLS,
        scope="x",
        policy=Policy(batch_rows=100),
        replicated=True,
        spool=s,
        **FAST,
    )
    assert c.calls[0][2] == "tok-old"  # the spooled item went first
    assert len(c.rows) == 1 + len(ROWS)
    assert s.pending_count() == 0


def test_drain_is_bounded_by_the_item_budget(tmp_path):
    s = spool_of(tmp_path, drain_max_items=2)
    for i in range(5):
        s.put("t", COLS, [[i, "x"]], f"tok-{i}")
    c = Cluster()
    resilient_insert(
        c,
        "t",
        ROWS,
        COLS,
        scope="x",
        policy=Policy(batch_rows=100),
        replicated=True,
        spool=s,
        **FAST,
    )
    assert sum(1 for call in c.calls if str(call[2]).startswith("tok-")) == 2
    assert s.pending_count() == 3


def test_drain_is_bounded_by_the_time_budget(tmp_path):
    now = [0.0]
    s = spool_of(tmp_path, drain_budget_s=1.0, clock=lambda: now[0])
    for i in range(5):
        s.put("t", COLS, [[i, "x"]], f"tok-{i}")
    c = Cluster()
    orig = c.insert

    def slow(*a, **k):
        now[0] += 0.6
        return orig(*a, **k)

    c.insert = slow
    s.drain(c, replicated=True, **FAST)
    assert 1 <= sum(1 for call in c.calls if str(call[2]).startswith("tok-")) <= 3
    assert s.pending_count() >= 2


def test_a_dead_cluster_does_not_make_the_callers_write_wait_on_the_drain(tmp_path):
    s = spool_of(tmp_path, drain_max_items=50)
    for i in range(20):
        s.put("t", COLS, [[i, "x"]], f"tok-{i}")
    c = Cluster(down=True)
    waits = []
    resilient_insert(
        c, "t", ROWS, COLS, scope="x",
        policy=Policy(batch_rows=100, deadline_s=0.5, backoff_s=1), replicated=True, spool=s,
        sleep=waits.append, log=lambda m: None,
    )  # fmt: skip
    # The drain stops at its first failure instead of retrying 20 items.
    assert len(c.calls) <= 3
    assert s.pending_count() >= 20


def test_drain_runs_at_process_start(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "a"]], "tok-1")
    c = Cluster()
    assert sp.drain_at_start(c, s, replicated=True, **FAST) == 1
    assert c.rows == [(1, "a")]


# ── exactly-once under races and crashes ──────────────────────────────────────


def test_two_writers_racing_replay_each_item_exactly_once(tmp_path):
    s1, s2 = spool_of(tmp_path), spool_of(tmp_path)
    for i in range(30):
        s1.put("t", COLS, [[i, "x"]], f"tok-{i}")
    c = Cluster(replicated=False)  # plain: a double replay WOULD double the rows
    barrier = threading.Barrier(2)

    def go(s):
        barrier.wait()
        s.drain(c, replicated=False, probe=not_stored(c), force=True, **FAST)

    ts = [threading.Thread(target=go, args=(s,)) for s in (s1, s2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(c.rows) == 30 and len(set(c.rows)) == 30


def test_two_writers_racing_on_the_shared_tier_replay_each_item_once(tmp_path):
    shared = FakeShared()
    s1 = spool_of(tmp_path / "a", shared=shared)
    s2 = spool_of(tmp_path / "b", shared=shared)
    for i in range(20):
        s1.put_shared("t", COLS, [[i, "x"]], f"tok-{i}")
    c = Cluster(replicated=False)
    barrier = threading.Barrier(2)

    def go(s):
        barrier.wait()
        s.drain(c, replicated=False, probe=not_stored(c), force=True, **FAST)

    ts = [threading.Thread(target=go, args=(s,)) for s in (s1, s2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(c.rows) == 20 and len(set(c.rows)) == 20
    assert (
        shared.files == {}
    )  # every item was replayed and deleted by exactly one writer


def test_a_stale_claim_expires_and_is_replayed(tmp_path):
    now = [1000.0]
    s = spool_of(tmp_path, lease_s=60, clock=lambda: now[0])
    s.put("t", COLS, [[1, "a"]], "tok-1")
    item = next(iter(s.local.pending()))
    claimed = s.local.claim(item, owner="dead-pid", now=now[0])
    assert claimed is not None and s.pending_count() == 0
    now[0] += 30
    s.reap_claims()
    assert s.pending_count() == 0  # lease still valid
    now[0] += 120
    s.reap_claims()
    assert s.pending_count() == 1


def _replayer(path, started, hang):
    s = sp.TieredSpool(local=path, shared=None, trusted=True)

    class Hang:
        def insert(self, *a, **k):
            started.set()
            hang.wait(30)

    s.drain(
        Hang(), replicated=True, sleep=lambda x: None, log=lambda m: None, force=True
    )


def test_kill_9_during_replay_loses_nothing_and_duplicates_nothing(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "a"]], "tok-1")
    started, hang = multiprocessing.Event(), multiprocessing.Event()
    p = multiprocessing.Process(
        target=_replayer, args=(str(tmp_path / "local"), started, hang)
    )
    p.start()
    assert started.wait(10)
    os.kill(p.pid, signal.SIGKILL)
    p.join()
    assert s.pending_count() == 0  # the dead process still holds the claim
    s.lease_s = 0
    s.reap_claims()
    assert s.pending_count() == 1
    c = Cluster(replicated=True)
    c.tokens.add("tok-1")  # the killed replay had landed before dying
    c.rows.append((1, "a"))
    s.drain(c, replicated=True, force=True, **FAST)
    assert c.rows == [(1, "a")] and s.pending_count() == 0


# ── plain tables: probe, never a blind resend ─────────────────────────────────


def test_plain_table_ambiguous_replay_is_probed_not_resent(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "a"], [2, "b"]], "tok-1", ambiguous=True)
    c = Cluster(replicated=False)
    c.rows.extend(
        [(1, "a"), (2, "b")]
    )  # the batch had landed before the spool was written
    s.drain(c, replicated=False, probe=lambda item: len(c.rows), force=True, **FAST)
    assert len(c.rows) == 2 and s.pending_count() == 0


def test_plain_table_with_a_clean_probe_is_resent(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "a"]], "tok-1", ambiguous=True)
    c = Cluster(replicated=False)
    s.drain(c, replicated=False, probe=lambda item: len(c.rows), force=True, **FAST)
    assert c.rows == [(1, "a")]


def test_a_plain_batch_refused_before_any_write_is_resent_without_a_probe(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "a"]], "tok-1")
    c = Cluster(replicated=False)
    out = s.drain(c, replicated=False, probe=None, force=True, **FAST)
    assert c.rows == [(1, "a")] and s.pending_count() == 0 and out.replayed == 1


def test_a_spooled_batch_is_not_mistaken_for_stored_by_earlier_batches(tmp_path):
    # Batch 1 lands, batch 2 is refused (285) and spooled; the table already holds 2 rows of
    # this scope, which must not make batch 2 look stored.
    c = Cluster(replicated=False, script=[None, err(285)] + [err(285)] * 9)
    s = spool_of(tmp_path)
    rows = [[i, "x"] for i in range(4)]
    resilient_insert(
        c, "t", rows, COLS, scope="r",
        policy=Policy(batch_rows=2, deadline_s=0, backoff_s=1),
        replicated=False, spool=s, **FAST,
    )  # fmt: skip
    c.script.clear()
    s.drain(c, replicated=False, probe=lambda item: len(c.rows), force=True, **FAST)
    assert sorted(c.rows) == sorted(map(tuple, rows)) and s.pending_count() == 0


def test_a_partly_stored_ambiguous_batch_is_held_for_a_person(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "a"], [2, "b"]], "tok-1", ambiguous=True)
    c = Cluster(replicated=False)
    c.rows.append((1, "a"))
    out = s.drain(c, replicated=False, probe=not_stored(c), force=True, **FAST)
    assert out.held == 1 and s.pending_count() == 1 and c.rows == [(1, "a")]


def test_two_spools_in_one_process_have_distinct_owners(tmp_path):
    a = spool_of(tmp_path / "a", writer="w")
    b = spool_of(tmp_path / "b", writer="w")
    assert a.owner != b.owner


def test_an_unreadable_shared_item_is_set_aside_not_retried_forever(tmp_path):
    shared = FakeShared()
    shared.put("w/bad.jsonl", b"{not json")
    shared.put("w/partial.jsonl", b'{"table": "t"}')
    s = spool_of(tmp_path, shared=shared)
    s.drain(Cluster(), replicated=True, force=True, **FAST)
    assert sorted(shared.files) == ["w/bad.jsonl.corrupt", "w/partial.jsonl.corrupt"]
    assert s.pending_count() == 0
    s.drain(Cluster(), replicated=True, force=True, **FAST)
    assert len(shared.files) == 2


def test_plain_table_without_a_probe_is_reported_and_kept(tmp_path):
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "a"]], "tok-1", ambiguous=True)
    c = Cluster(replicated=False)
    out = s.drain(c, replicated=False, probe=None, **FAST)
    assert c.rows == [] and s.pending_count() == 1 and out.held == 1


def test_an_item_older_than_the_dedup_window_gets_the_probe_even_when_replicated(
    tmp_path,
):
    now = [10_000.0]
    s = spool_of(tmp_path, dedup_window_s=3600, clock=lambda: now[0])
    s.put("t", COLS, [[1, "a"]], "tok-1", ambiguous=True, created=now[0] - 7200)
    c = Cluster(replicated=True)
    c.rows.append(
        (1, "a")
    )  # it landed, but its token has aged out of the server's window
    s.drain(c, replicated=True, probe=lambda item: len(c.rows), **FAST)
    assert c.rows == [(1, "a")] and s.pending_count() == 0


# ── shared tier: fallback, trust, caps ────────────────────────────────────────


def test_unreachable_shared_tier_falls_back_to_local(tmp_path):
    shared = FakeShared(reachable=False)
    s = spool_of(tmp_path, shared=shared)
    where = s.put_any("t", COLS, ROWS, "tok-1")
    assert where == "local" and s.local_count() == 1


def test_reachable_shared_tier_is_used_when_trusted(tmp_path):
    shared = FakeShared()
    s = spool_of(tmp_path, shared=shared)
    assert s.put_any("t", COLS, ROWS, "tok-1") == "shared"
    assert len(shared.files) == 1


def test_untrusted_run_refuses_the_shared_tier(tmp_path):
    shared = FakeShared()
    s = spool_of(tmp_path, shared=shared, trusted=False)
    assert s.put_any("t", COLS, ROWS, "tok-1") == "local"
    assert shared.files == {}
    assert s.drain(Cluster(), replicated=True, **FAST).from_shared == 0


def test_trust_comes_from_the_environment_and_defaults_closed(monkeypatch):
    for k in (
        "CH_WRITE_SHARED_SPOOL_TRUSTED",
        "GITHUB_EVENT_NAME",
        "GITHUB_REPOSITORY",
    ):
        monkeypatch.delenv(k, raising=False)
    assert sp.shared_allowed() is False
    monkeypatch.setenv("CH_WRITE_SHARED_SPOOL_TRUSTED", "1")
    assert sp.shared_allowed() is True


@pytest.mark.parametrize("event", ["pull_request", "pull_request_target", "fork"])
def test_a_pull_request_run_is_never_trusted_even_with_the_flag(monkeypatch, event):
    monkeypatch.setenv("CH_WRITE_SHARED_SPOOL_TRUSTED", "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    assert sp.shared_allowed() is False


@pytest.fixture
def actions_env(monkeypatch):
    for k in ("GITHUB_EVENT_PATH", "GITHUB_REF_PROTECTED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CH_WRITE_SHARED_SPOOL_TRUSTED", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF_PROTECTED", "true")
    return monkeypatch


@pytest.mark.parametrize(
    "event",
    [
        "issue_comment",
        "workflow_run",
        "pull_request_review",
        "pull_request_review_comment",
        "merge_group",
        "repository_dispatch",
        "",
    ],
)
def test_events_outside_the_allowlist_are_refused(actions_env, event):
    actions_env.setenv("GITHUB_EVENT_NAME", event)
    assert sp.shared_allowed() is False


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_allowlisted_events_on_a_protected_ref_are_trusted(actions_env, event):
    actions_env.setenv("GITHUB_EVENT_NAME", event)
    assert sp.shared_allowed() is True


@pytest.mark.parametrize("protected", ["false", ""])
def test_an_unprotected_ref_is_refused(actions_env, protected):
    actions_env.setenv("GITHUB_EVENT_NAME", "push")
    actions_env.setenv("GITHUB_REF_PROTECTED", protected)
    assert sp.shared_allowed() is False


def test_a_fork_pr_is_never_trusted(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_WRITE_SHARED_SPOOL_TRUSTED", "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    ev = tmp_path / "ev.json"
    ev.write_text('{"pull_request": {"head": {"repo": {"fork": true}}}}')
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    assert sp.shared_allowed() is False


def test_size_cap_refuses_new_items_and_reports(tmp_path):
    s = spool_of(tmp_path, max_bytes=300)
    assert s.put_any("t", COLS, ROWS, "tok-1") in ("local", "shared")
    with pytest.raises(sp.SpoolFull):
        for i in range(50):
            s.put_any("t", COLS, ROWS * 5, f"tok-{i}")


def test_age_cap_drops_nothing_silently_it_quarantines_and_counts(tmp_path):
    now = [0.0]
    s = spool_of(tmp_path, max_age_s=100, clock=lambda: now[0])
    s.put("t", COLS, [[1, "a"]], "tok-1", created=now[0])
    now[0] = 500
    stats = s.stats()
    assert stats.pending == 1 and stats.oldest_age_s == 500 and stats.over_age == 1
    s.expire()
    assert s.pending_count() == 0 and s.quarantined_count() == 1


def test_stats_report_pending_count_oldest_age_and_bytes(tmp_path):
    now = [100.0]
    s = spool_of(tmp_path, clock=lambda: now[0])
    s.put("t", COLS, ROWS, "tok-1", created=40.0)
    s.put("t", COLS, ROWS, "tok-2", created=90.0)
    st = s.stats()
    assert st.pending == 2 and st.oldest_age_s == 60 and st.bytes > 0


def test_clickhouse_unreachable_keeps_the_item_and_the_shared_claim_is_released(
    tmp_path,
):
    shared = FakeShared()
    s = spool_of(tmp_path, shared=shared)
    s.put_shared("t", COLS, ROWS, "tok-1")
    s.drain(Cluster(down=True), replicated=True, force=True, **FAST)
    assert any(n.endswith("tok-1.jsonl") for n in shared.files)
    assert shared.held == {}  # the lock was given back, so the next writer can retry


def test_rows_never_appear_in_a_log_line(tmp_path):
    logs = []
    s = spool_of(tmp_path)
    s.put("t", COLS, [[1, "SECRET-ROW-VALUE"]], "tok-1")
    s.drain(
        Cluster(down=True),
        replicated=True,
        sleep=lambda x: None,
        log=logs.append,
        force=True,
    )
    assert not any("SECRET-ROW-VALUE" in m for m in logs)
