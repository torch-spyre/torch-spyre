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

"""Quorum and replication failures: classification, deadline, tokens, degrade policy, spool."""

import json
import multiprocessing
import os
import signal
import time

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

from spyre_clickhouse_ingest import resilience
from spyre_clickhouse_ingest.resilience import (
    Policy,
    Spool,
    batch_token,
    classify,
    resilient_insert as _resilient_insert,
)


def resilient_insert(*a, **k):
    k.setdefault("feeds_views", False)  # these clusters model clean refusals
    return _resilient_insert(*a, **k)


def err(code, name="X"):
    return DatabaseError(
        f"Code: {code}. DB::Exception: x ({name})", code=code, name=name
    )


# ── classification ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "error,expected",
    [
        # Nothing was stored: the quorum check fails before the write.
        (err(285, "TOO_FEW_LIVE_REPLICAS"), "retry"),
        # The previous quorum write has not finished; this one was refused untouched.
        (err(286, "UNSATISFIED_QUORUM_FOR_PREVIOUS_WRITE"), "retry"),
        # The block may or may not be stored: only a token (Replicated) or a probe decides.
        (err(319, "UNKNOWN_STATUS_OF_INSERT"), "ambiguous"),
        # Quorum timeout: the block may sit on fewer replicas than asked.
        (err(159, "TIMEOUT_EXCEEDED"), "ambiguous"),
        (err(242, "TABLE_IS_READ_ONLY"), "retry"),
        (err(252, "TOO_MANY_PARTS"), "retry"),
        # May hit while a materialized view is being fed, after the source block landed.
        (err(241, "MEMORY_LIMIT_EXCEEDED"), "ambiguous"),
        (err(999, "KEEPER_EXCEPTION"), "ambiguous"),
        (OperationalError("Read timed out"), "ambiguous"),
        (ConnectionResetError("reset"), "ambiguous"),
        # Deterministic: never retry.
        (err(60, "UNKNOWN_TABLE"), "fatal"),
        (err(47, "UNKNOWN_IDENTIFIER"), "fatal"),
        (err(497, "ACCESS_DENIED"), "fatal"),
        (err(516, "AUTHENTICATION_FAILED"), "fatal"),
        (err(469, "VIOLATED_CONSTRAINT"), "fatal"),
        (ValueError("bad row"), "fatal"),
    ],
)
def test_classification_table(error, expected):
    assert classify(error, feeds_views=False) == expected


@pytest.mark.parametrize("code", [242, 252, 285, 286])
def test_pre_write_codes_prove_nothing_when_a_view_may_have_the_block(code):
    assert classify(err(code)) == "ambiguous"
    assert classify(err(code), feeds_views=False) == "retry"


def test_a_status_only_5xx_without_a_code_is_ambiguous():
    e = DatabaseError("HTTP driver received HTTP status 503, server response: reset")
    assert classify(e) == "ambiguous"


def test_the_code_wins_over_the_text():
    e = DatabaseError(
        "HTTP status 500 ... Code: 60", code=285, name="TOO_FEW_LIVE_REPLICAS"
    )
    assert classify(e, feeds_views=False) == "retry"


# ── tokens ──────────────────────────────────────────────────────────────────────


def test_token_is_deterministic_and_sensitive_to_content_and_scope():
    a = [[1, "x"], [2, "y"]]
    assert batch_token("t", "scope", a) == batch_token(
        "t", "scope", [[1, "x"], [2, "y"]]
    )
    assert batch_token("t", "scope", a) != batch_token(
        "t", "scope", [[1, "x"], [2, "z"]]
    )
    assert batch_token("t", "scope", a) != batch_token("t", "other", a)
    assert batch_token("t", "scope", a) != batch_token("u", "scope", a)
    assert batch_token("t", "scope", a) != batch_token("t", "scope", list(reversed(a)))


def test_token_is_short_enough_for_a_setting():
    assert len(batch_token("t", "s", [[1]])) <= 80


# ── a fake transport that fails the way a degraded cluster does ───────────────


class Cluster:
    """Stores a batch once per distinct token on a replicated table; always on a plain one."""

    def __init__(self, script=(), replicated=True, store_on_fail=False):
        self.script, self.replicated, self.store_on_fail = (
            list(script),
            replicated,
            store_on_fail,
        )
        self.rows, self.seen, self.calls = [], set(), []

    def insert(self, table, data, column_names=None, database=None, settings=None):
        settings = settings or {}
        self.calls.append(dict(settings))
        step = self.script.pop(0) if self.script else None
        token = settings.get("insert_deduplication_token")
        dup = self.replicated and token and token in self.seen
        if step is not None:
            if self.store_on_fail and not dup:
                self.rows.extend(data)
                if token:
                    self.seen.add(token)
            raise step
        if dup:
            return
        self.rows.extend(data)
        if token:
            self.seen.add(token)


ROWS = [[i, f"r{i}"] for i in range(7)]
FAST = dict(sleep=lambda s: None, log=lambda m: None)


def run(cluster, policy=None, **kw):
    return resilient_insert(
        cluster,
        "t",
        ROWS,
        ["a", "b"],
        scope="run-1",
        policy=policy or Policy(batch_rows=3, deadline_s=60, backoff_s=1),
        replicated=cluster.replicated,
        **{**FAST, **kw},
    )


def test_quorum_failure_then_recovery_stores_each_row_once():
    c = Cluster(script=[err(285), err(285)])
    assert run(c).stored == 7
    assert c.rows == ROWS


def test_ambiguous_failure_that_was_stored_is_not_doubled_on_a_replicated_table():
    c = Cluster(script=[err(319)], store_on_fail=True)
    assert run(c).stored == 7
    assert c.rows == ROWS


def test_ambiguous_failure_on_a_plain_table_without_a_probe_is_spooled_not_resent(
    tmp_path,
):
    c = Cluster(script=[err(319)], replicated=False, store_on_fail=True)
    out = run(c, spool=Spool(tmp_path))
    assert out.stored < 7 and out.spooled > 0 and out.ambiguous
    held = list(Spool(tmp_path).pending())
    assert held[0].ambiguous and not any(i.ambiguous for i in held[1:])
    assert c.rows.count(ROWS[0]) == 1


def test_too_many_parts_retries_only_when_the_table_feeds_no_view():
    assert classify(err(252), feeds_views=False) == "retry"
    assert classify(err(252)) == "ambiguous"


def test_ambiguity_survives_later_refusals_and_dates_from_the_first_attempt(tmp_path):
    # Attempt 1 times out but is stored; attempts 2 and 3 are refused until the deadline.
    now = [1000.0]
    c = Cluster(script=[err(319), err(285), err(285), err(285)], store_on_fail=True)
    sp = Spool(tmp_path, clock=lambda: now[0])

    def sleep(s):
        now[0] += s

    resilient_insert(
        c, "t", ROWS[:3], ["a", "b"], scope="s",
        policy=Policy(batch_rows=3, deadline_s=5, backoff_s=2),
        replicated=True, spool=sp, sleep=sleep, clock=lambda: now[0], log=lambda m: None,
    )  # fmt: skip
    (item,) = list(sp.pending())
    assert item.ambiguous
    assert item.created == pytest.approx(1000.0)


def test_ambiguous_plain_batch_without_a_spool_is_loud():
    c = Cluster(script=[err(319)], replicated=False, store_on_fail=True)
    with pytest.raises(resilience.WriteAmbiguous) as ei:
        run(c)
    assert ei.value.outcome.ambiguous


def test_a_full_spool_raises_and_does_not_hide_an_ambiguous_plain_batch(tmp_path):
    c = Cluster(script=[err(319)], replicated=False, store_on_fail=True)
    with pytest.raises(resilience.WriteAmbiguous) as ei:
        run(c, spool=Spool(tmp_path, max_bytes=1))
    assert ei.value.outcome.ambiguous
    c = Cluster(script=[err(285)] * 99, replicated=False)
    with pytest.raises(resilience.SpoolFull):
        run(
            c,
            Policy(batch_rows=3, deadline_s=0, backoff_s=1),
            spool=Spool(tmp_path / "b", max_bytes=1),
        )


def test_a_dead_cluster_costs_one_deadline_not_one_per_batch(tmp_path):
    now, slept = [0.0], []
    c = Cluster(script=[err(285)] * 99)

    def sleep(s):
        slept.append(s)
        now[0] += s

    out = resilient_insert(
        c,
        "t",
        [[i, "x"] for i in range(10)],
        ["a", "b"],
        scope="s",
        policy=Policy(batch_rows=2, deadline_s=10, backoff_s=2, backoff_max_s=4),
        replicated=False,
        spool=Spool(tmp_path),
        sleep=sleep,
        clock=lambda: now[0],
        log=lambda m: None,
    )
    assert out.spooled == 10 and out.stored == 0
    assert sum(slept) <= 10
    assert len(c.calls) == len(slept) + 1


def test_identical_batches_in_one_load_get_distinct_tokens():
    b = [[1, "a"]]
    assert batch_token("t", "s", b, 1) != batch_token("t", "s", b, 2)


def test_fatal_error_is_raised_at_once():
    c = Cluster(script=[err(60, "UNKNOWN_TABLE")] * 9)
    with pytest.raises(DatabaseError):
        run(c)
    assert len(c.calls) == 1


# ── backoff and deadline ───────────────────────────────────────────────────────


def test_backoff_doubles_and_is_capped():
    waits = []
    c = Cluster(script=[err(285)] * 5)
    run(
        c,
        Policy(batch_rows=100, deadline_s=1e6, backoff_s=2, backoff_max_s=7),
        sleep=waits.append,
    )
    assert waits[:4] == [2, 4, 7, 7]


def test_the_deadline_stops_retrying_and_spools(tmp_path):
    now = [0.0]
    c = Cluster(script=[err(285)] * 99)

    def sleep(s):
        now[0] += s

    out = run(
        c,
        Policy(batch_rows=3, deadline_s=10, backoff_s=4),
        spool=Spool(tmp_path),
        sleep=sleep,
        clock=lambda: now[0],
    )
    assert out.stored == 0 and out.spooled == 7
    assert now[0] <= 10 + 4


def test_without_a_spool_an_exhausted_deadline_raises():
    now = [0.0]
    c = Cluster(script=[err(285)] * 99)
    with pytest.raises(resilience.WriteUnavailable):
        run(
            c,
            Policy(batch_rows=3, deadline_s=5, backoff_s=2),
            sleep=lambda s: now.__setitem__(0, now[0] + s),
            clock=lambda: now[0],
        )


# ── settings: quorum, async, tokens ───────────────────────────────────────────


def test_replicated_inserts_carry_token_quorum_and_async_dedup():
    c = Cluster()
    run(c, Policy(batch_rows=3, quorum="2", deadline_s=60, backoff_s=1))
    s = c.calls[0]
    assert s["insert_deduplication_token"] and s["insert_quorum"] == "2"
    assert s["async_insert_deduplicate"] == 1 and s["wait_for_async_insert"] == 1


def test_plain_tables_get_no_token_and_no_quorum():
    c = Cluster(replicated=False)
    run(c, Policy(batch_rows=3, quorum="2"))
    assert "insert_deduplication_token" not in c.calls[0]
    assert "insert_quorum" not in c.calls[0]


def test_the_token_is_stable_across_retries_of_the_same_batch():
    c = Cluster(script=[err(285)])
    run(c)
    assert (
        c.calls[0]["insert_deduplication_token"]
        == c.calls[1]["insert_deduplication_token"]
    )


# ── degrade policy ────────────────────────────────────────────────────────────


def test_degradable_table_drops_quorum_after_the_quorum_window():
    now = [0.0]
    c = Cluster(script=[err(285)] * 3)
    run(
        c,
        Policy(
            batch_rows=100,
            quorum="2",
            degrade_to="1",
            degrade_after_s=3,
            deadline_s=60,
            backoff_s=2,
        ),
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        clock=lambda: now[0],
    )
    assert c.calls[0]["insert_quorum"] == "2"
    assert c.calls[-1].get("insert_quorum") in (None, "1", 1, 0)
    assert len(c.rows) == 7


def test_a_strict_table_never_drops_quorum():
    now = [0.0]
    c = Cluster(script=[err(285)] * 6)
    run(
        c,
        Policy(
            batch_rows=100, quorum="2", degrade_to=None, deadline_s=1e6, backoff_s=2
        ),
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        clock=lambda: now[0],
    )
    assert all(call["insert_quorum"] == "2" for call in c.calls)


# ── spool ─────────────────────────────────────────────────────────────────────


def test_spool_roundtrip_keeps_the_token_and_the_rows(tmp_path):
    sp = Spool(tmp_path)
    tok = batch_token("t", "s", ROWS)
    sp.put("t", ["a", "b"], ROWS, tok, database="spyre_v2")
    [item] = list(sp.pending())
    assert (
        item.table == "t"
        and item.token == tok
        and item.rows == ROWS
        and item.database == "spyre_v2"
    )


def _writer(path, started):
    sp = Spool(path)
    for i in range(500):
        sp.put("t", ["a"], [[i]], batch_token("t", "s", [[i]]))
        if i == 20:
            started.set()


def test_kill_9_mid_write_leaves_only_whole_spool_files(tmp_path):
    started = multiprocessing.Event()
    p = multiprocessing.Process(target=_writer, args=(str(tmp_path), started))
    p.start()
    assert started.wait(10)
    os.kill(p.pid, signal.SIGKILL)
    p.join()
    sp = Spool(tmp_path)
    items = list(sp.pending())
    assert len(items) >= 20
    for it in items:
        assert it.rows and it.token  # every file parses; none is truncated
    assert len({i.token for i in items}) == len(items)


def test_a_corrupt_spool_file_is_quarantined_not_fatal(tmp_path):
    (tmp_path / "0000000000001-bad.json").write_text("{not json")
    sp = Spool(tmp_path)
    sp.put("t", ["a"], [[1]], "tok")
    assert len(list(sp.pending())) == 1
    assert (tmp_path / "0000000000001-bad.json.corrupt").exists()


def test_spool_file_is_plain_json(tmp_path):
    sp = Spool(tmp_path)
    sp.put("t", ["a"], [[1]], "tok")
    [f] = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    assert json.loads((tmp_path / f).read_text())["table"] == "t"


def test_time_is_injectable_not_real():
    assert resilience.time is time


def test_a_refused_connection_is_a_retry_because_nothing_was_sent():
    assert classify(ConnectionRefusedError("refused")) == "retry"
    try:
        raise OperationalError("Error executing HTTP request")
    except OperationalError as wrapped:
        try:
            raise ConnectionRefusedError("refused")
        except ConnectionRefusedError as cause:
            wrapped.__cause__ = cause
        assert classify(wrapped) == "retry"
    # The text alone proves nothing, and a timeout may have sent the request.
    assert (
        classify(OperationalError("Failed to establish a new connection"))
        == "ambiguous"
    )
    assert classify(OperationalError("Read timed out")) == "ambiguous"


def test_error_text_never_reaches_a_log_or_a_message():
    secret = "row-data-secret"
    logs = []
    c = Cluster(script=[DatabaseError(f"Code: 285 {secret}", code=285)] * 5)
    with pytest.raises(resilience.WriteUnavailable) as ei:
        run(
            c,
            Policy(batch_rows=3, deadline_s=1, backoff_s=2),
            log=logs.append,
            sleep=lambda s: None,
        )
    assert secret not in str(ei.value) and not any(secret in m for m in logs)


def test_replicated_inserts_ask_dependent_views_to_deduplicate_too():
    c = Cluster()
    run(c, feeds_views=True)
    assert all(
        s.get("deduplicate_blocks_in_dependent_materialized_views") == 1
        for s in c.calls
    )
    c = Cluster()
    resilient_insert(
        c, "t", ROWS, ["a", "b"], scope="s", replicated=True, feeds_views=False, **FAST
    )
    assert "deduplicate_blocks_in_dependent_materialized_views" not in c.calls[0]


# ── the plain spool ───────────────────────────────────────────────────────────


def test_spool_files_and_directory_are_owner_only(tmp_path):
    import stat

    sp = Spool(tmp_path / "s")
    p = sp.put("t", ["a"], [[1]], "tok", ambiguous=True)
    assert stat.S_IMODE(os.stat(sp.dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    sp.put("t", ["a"], [[1]], "tok")
    sp.put("t", ["a"], [[1]], "tok2")
    assert {stat.S_IMODE(os.stat(f).st_mode) for f in sp.files()} == {0o600}


def test_ambiguity_is_a_field_in_the_json(tmp_path):
    sp = Spool(tmp_path)
    p = sp.put("t", ["a"], [[1]], "tok", ambiguous=True)
    assert json.loads(p.read_text())["ambiguous"] is True
    assert p.name.endswith("-tok.json")


def test_parking_a_token_again_never_clears_ambiguity(tmp_path):
    sp = Spool(tmp_path)
    sp.put("t", ["a"], [[1]], "tok")
    sp.put("t", ["a"], [[1]], "tok", ambiguous=True)
    assert [i.ambiguous for i in sp.pending()] == [True]
    sp.put("t", ["a"], [[1]], "tok")  # e.g. a rerun of the same load
    assert [i.ambiguous for i in sp.pending()] == [True] and len(sp.files()) == 1


def test_a_full_spool_raises_instead_of_dropping(tmp_path):
    sp = Spool(tmp_path, max_bytes=300)
    sp.put("t", ["a"], [[1]], "tok-1")
    with pytest.raises(resilience.SpoolFull):
        for i in range(2, 50):
            sp.put("t", ["a"], [[i]], f"tok-{i}")
    assert len(sp.files()) >= 1  # nothing already parked was removed


@pytest.mark.parametrize("bad", ["*", "a*", "../x", "a/b", "[a]", "", "x" * 201, ".."])
def test_drop_refuses_anything_but_a_plain_token(tmp_path, bad):
    sp = Spool(tmp_path)
    sp.put("t", ["a"], [[1]], "tok")
    with pytest.raises(ValueError):
        sp.drop(bad)
    assert len(sp.files()) == 1


def test_the_spool_command_lists_and_drops(tmp_path, monkeypatch, capsys):
    from spyre_clickhouse_ingest import spool_cli

    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path))
    sp = Spool(tmp_path)
    sp.put("t", ["a"], [[1]], "tok-1", ambiguous=True)
    sp.put("t", ["a"], [[2]], "tok-2")
    assert spool_cli.main(["list"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {r["token"]: r["held"] for r in rows} == {"tok-1": True, "tok-2": False}
    assert "rows" in rows[0] and "[[" not in json.dumps(rows)  # row data is not printed
    assert spool_cli.main(["drop", "*"]) == 2 and len(sp.files()) == 2
    assert spool_cli.main(["drop", "tok-1"]) == 0 and len(sp.files()) == 1
    assert spool_cli.main(["drop", "nope"]) == 1
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "missing"))
    assert spool_cli.main(["list"]) == 2
