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
from pathlib import Path

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


class NoJitter:
    """A jitter source that always waits the full ceiling, so waits are exact."""

    def uniform(self, lo, hi):
        return hi


FAST = dict(sleep=lambda s: None, log=lambda m: None, rng=NoJitter())


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


def test_backoff_is_fully_jittered_and_seedable():
    import random

    def waits_for(seed):
        waits = []
        c = Cluster(script=[err(285)] * 6)
        run(
            c,
            Policy(batch_rows=100, deadline_s=1e6, backoff_s=2, backoff_max_s=7),
            sleep=waits.append,
            rng=random.Random(seed),
        )
        return waits

    a = waits_for(1)
    assert a == waits_for(1) and a != waits_for(2)
    ceilings = [2, 4, 7, 7, 7, 7]
    assert all(0 < w <= c for w, c in zip(a, ceilings))
    assert len(set(a)) > 1 and a[:4] != [2, 4, 7, 7]


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


# ── classification of transient server codes and transport errors ──────────────


@pytest.mark.parametrize(
    "code,name,expected_views,expected_clean",
    [
        # 289 is raised by sequential-consistency reads, never by the insert sink itself.
        (289, "REPLICA_IS_NOT_IN_QUORUM", "ambiguous", "retry"),
        # Refused at query admission, before anything runs, whatever feeds from the table.
        (202, "TOO_MANY_SIMULTANEOUS_QUERIES", "retry", "retry"),
        (203, "NO_FREE_CONNECTION", "ambiguous", "ambiguous"),
        (225, "NO_ZOOKEEPER", "ambiguous", "ambiguous"),
        (415, "ALL_REPLICAS_LOST", "ambiguous", "ambiguous"),
        (439, "CANNOT_SCHEDULE_TASK", "ambiguous", "ambiguous"),
        (244, "UNEXPECTED_ZOOKEEPER_ERROR", "ambiguous", "ambiguous"),
        (745, "SERVER_OVERLOADED", "ambiguous", "ambiguous"),
        (1017, "ASYNC_INSERT_FLUSH_TIMEOUT", "ambiguous", "ambiguous"),
        # An unknown server code may be transient: ambiguous, so it is resent only under a token.
        (7777, "NOT_A_KNOWN_CODE", "ambiguous", "ambiguous"),
    ],
)
def test_transient_server_codes(code, name, expected_views, expected_clean):
    assert classify(err(code, name)) == expected_views
    assert classify(err(code, name), feeds_views=False) == expected_clean


def test_164_is_a_retry_when_the_replica_is_not_active_but_fatal_for_a_readonly_user():
    inactive = DatabaseError(
        "Code: 164. DB::Exception: Replica is not active right now", code=164
    )
    assert classify(inactive, feeds_views=False) == "retry"
    assert classify(inactive) == "ambiguous"
    ro = DatabaseError(
        "Code: 164. DB::Exception: u: Cannot execute query in readonly mode", code=164
    )
    assert classify(ro) == "fatal" and classify(ro, feeds_views=False) == "fatal"


@pytest.mark.parametrize("code", [6, 38, 41, 44, 46, 70, 117, 192, 193])
def test_more_deterministic_codes_are_fatal(code):
    assert classify(err(code)) == "fatal"


def test_transport_errors_by_type():
    from urllib3.exceptions import (
        ConnectTimeoutError,
        MaxRetryError,
        ProtocolError,
        ReadTimeoutError,
    )

    # No connection was made: nothing was sent.
    assert classify(ConnectTimeoutError(None, "connect timed out")) == "retry"
    assert classify(MaxRetryError(None, "/", ConnectTimeoutError(None, "t"))) == "retry"
    wrapped = OperationalError("Error executing HTTP request")
    wrapped.__cause__ = ConnectTimeoutError(None, "connect timed out")
    assert classify(wrapped) == "retry"
    # The request may have been sent.
    assert classify(ReadTimeoutError(None, "/", "read")) == "ambiguous"
    assert classify(ProtocolError("Connection aborted")) == "ambiguous"
    assert (
        classify(MaxRetryError(None, "/", ReadTimeoutError(None, "/", "r")))
        == "ambiguous"
    )


# ── a balancer's error is not a ClickHouse error ──────────────────────────────

GENERIC = "The ClickHouse server returned an error"


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_gateway_status_with_the_text_hidden_is_ambiguous(status):
    # show_clickhouse_errors off: generic text, no status in it. The balancer sent no
    # X-ClickHouse-Exception-Code header, so the driver set no code.
    assert classify(DatabaseError(GENERIC)) == "ambiguous"
    assert classify(OperationalError(GENERIC)) == "ambiguous"


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_gateway_status_with_the_text_shown_is_ambiguous(status):
    e = DatabaseError(
        f"HTTP driver received HTTP status {status}, server response: <html>"
    )
    assert classify(e) == "ambiguous"
    assert classify(OperationalError(f"HTTP driver received HTTP status {status}")) == (
        "ambiguous"
    )


def test_a_clickhouse_error_with_a_code_is_not_taken_for_a_gateway_error():
    assert classify(DatabaseError(GENERIC, code=60, name=None)) == "fatal"
    assert classify(DatabaseError(GENERIC, code=285)) == "ambiguous"


def test_client_side_errors_stay_fatal():
    from clickhouse_connect.driver.exceptions import DataError, ProgrammingError

    assert classify(ProgrammingError("bad parameter")) == "fatal"
    assert classify(DataError("cannot serialize")) == "fatal"
    assert classify(DatabaseError("HTTP driver received HTTP status 404")) == "fatal"


# ── the code is read from ClickHouse's own text only ──────────────────────────


def test_code_text_is_parsed_only_from_a_clickhouse_exception_line():
    from spyre_clickhouse_ingest.resilience import _code

    assert _code(ValueError("Code: 60. DB::Exception: Table x doesn't exist")) == 60
    assert _code(ValueError("bad row, see Code: 285 in the file")) is None
    assert _code(ValueError("xCode: 285. DB::Exception: y")) is None
    assert classify(ValueError("row says Code: 285. nothing else")) == "fatal"


# ── a fatal error after an ambiguous attempt still says the batch may be stored ──


def test_fatal_after_ambiguous_carries_maybe_stored_and_the_outcome():
    c = Cluster(script=[err(319), err(60, "UNKNOWN_TABLE")], store_on_fail=True)
    with pytest.raises(DatabaseError) as ei:
        run(c)
    assert ei.value.maybe_stored is True
    out = ei.value.outcome
    assert out.ambiguous and out.unwritten == [1, 2, 3]


def test_fatal_with_no_earlier_ambiguity_is_not_maybe_stored():
    c = Cluster(script=[err(60, "UNKNOWN_TABLE")])
    with pytest.raises(DatabaseError) as ei:
        run(c)
    assert ei.value.maybe_stored is False and not ei.value.outcome.ambiguous


# ── unknown codes: resent only where a token makes it safe ────────────────────


def test_an_unknown_code_is_retried_on_a_replicated_table_under_the_same_token():
    c = Cluster(script=[err(7777, "NOT_A_KNOWN_CODE")], store_on_fail=True)
    assert run(c).stored == 7 and c.rows == ROWS


def test_an_unknown_code_on_a_plain_table_is_parked_not_resent(tmp_path):
    c = Cluster(script=[err(7777)], replicated=False, store_on_fail=True)
    out = run(c, spool=Spool(tmp_path))
    assert out.ambiguous and c.rows.count(ROWS[0]) == 1
    assert list(Spool(tmp_path).pending())[0].ambiguous


def test_an_unknown_code_on_a_window_zero_table_is_parked_not_resent(tmp_path):
    c = Cluster(script=[err(7777)], replicated=True, store_on_fail=True)
    out = run(c, spool=Spool(tmp_path), dedup_window=0)
    assert out.ambiguous and c.rows.count(ROWS[0]) == 1


# ── tables with no dedup window ───────────────────────────────────────────────


def test_a_window_zero_table_is_handled_like_a_plain_one(tmp_path):
    c = Cluster(script=[err(319)], replicated=False, store_on_fail=True)
    out = resilient_insert(
        c, "t", ROWS, ["a", "b"], scope="s", replicated=True, dedup_window=0,
        policy=Policy(batch_rows=3, quorum="2"), spool=Spool(tmp_path), **FAST,
    )  # fmt: skip
    assert out.ambiguous and out.spooled
    assert c.rows.count(ROWS[0]) == 1  # not resent
    first = c.calls[0]
    assert "insert_deduplication_token" not in first
    assert "async_insert_deduplicate" not in first
    assert first["insert_quorum"] == "2"  # quorum is about durability, not dedup


def test_a_window_zero_table_still_retries_a_refusal_before_any_write():
    c = Cluster(script=[err(285), err(285)], replicated=False)
    out = resilient_insert(
        c, "t", ROWS, ["a", "b"], scope="s", replicated=True, dedup_window=0,
        feeds_views=False, **FAST,
    )  # fmt: skip
    assert out.stored == 7 and c.rows == ROWS


def test_a_nonzero_or_unknown_window_sends_the_token():
    for window in (None, 1000):
        c = Cluster()
        resilient_insert(
            c, "t", ROWS, ["a", "b"], scope="s", replicated=True,
            dedup_window=window, **FAST,
        )  # fmt: skip
        assert c.calls[0]["insert_deduplication_token"]


class FakeQuery:
    def __init__(self, rows):
        self.result_rows = rows


class SystemClient:
    def __init__(self, tables, server_default=1000, broken=False):
        self.tables, self.server_default, self.broken = tables, server_default, broken
        self.queries = []
        self.database = "spyre_v2"

    def query(self, sql, parameters=None):
        self.queries.append(sql)
        if self.broken:
            raise OperationalError("down")
        if "system.tables" in sql:
            hit = self.tables.get((parameters["db"], parameters["t"]))
            return FakeQuery([hit] if hit else [])
        return FakeQuery([[str(self.server_default)]])


def test_dedup_window_reads_the_table_setting_once_and_caches():
    cl = SystemClient(
        {
            ("spyre_v2", "a"): ("ReplicatedMergeTree", "ReplicatedMergeTree(...) ORDER BY x SETTINGS replicated_deduplication_window = 0, index_granularity = 8192"),
            ("spyre_v2", "b"): ("ReplicatedMergeTree", "ReplicatedMergeTree(...) ORDER BY x"),
            ("spyre_v2", "c"): ("MergeTree", "MergeTree ORDER BY x"),
        },
        server_default=250,
    )  # fmt: skip
    cache = {}
    from spyre_clickhouse_ingest.resilience import dedup_window

    assert dedup_window(cl, "a", None, cache) == 0
    assert dedup_window(cl, "b", "spyre_v2", cache) == 250  # the server default
    assert (
        dedup_window(cl, "c", "spyre_v2", cache) == 0
    )  # a plain table deduplicates nothing
    n = len(cl.queries)
    assert dedup_window(cl, "a", None, cache) == 0 and len(cl.queries) == n


def test_dedup_window_is_unknown_when_unreadable_and_is_not_cached():
    from spyre_clickhouse_ingest.resilience import dedup_window

    cache = {}
    assert dedup_window(SystemClient({}), "gone", "spyre_v2", cache) is None
    assert dedup_window(SystemClient({}, broken=True), "t", "d", cache) is None
    assert cache == {}


# ── tokens come from content ──────────────────────────────────────────────────


def test_the_token_does_not_depend_on_batch_position_or_call():
    a, b = [[1, "x"]], [[2, "y"]]
    c1, c2 = Cluster(), Cluster()
    resilient_insert(c1, "t", a + b, ["a", "b"], scope="s", replicated=True,
                     policy=Policy(batch_rows=1), **FAST)  # fmt: skip
    resilient_insert(c2, "t", b + a, ["a", "b"], scope="s", replicated=True,
                     policy=Policy(batch_rows=1), **FAST)  # fmt: skip
    tok = lambda c: [x["insert_deduplication_token"] for x in c.calls]  # noqa: E731
    assert sorted(tok(c1)) == sorted(tok(c2))
    assert tok(c1)[0] == batch_token("t", "s", a)


def test_two_calls_with_identical_rows_share_a_token_unless_the_scope_differs():
    rows = [[1, "x"]]
    assert batch_token("t", "load-1", rows) == batch_token("t", "load-1", rows)
    assert batch_token("t", "load-1", rows) != batch_token("t", "load-2", rows)


def test_a_different_batch_size_gives_different_tokens():
    # The README states this: rerun a load with the batch size it started with.
    c1, c2 = Cluster(), Cluster()
    run(c1, Policy(batch_rows=3))
    run(c2, Policy(batch_rows=4))
    t1 = {x["insert_deduplication_token"] for x in c1.calls}
    t2 = {x["insert_deduplication_token"] for x in c2.calls}
    assert not (t1 & t2)


def test_identical_batches_in_one_call_are_told_apart():
    c = Cluster()
    resilient_insert(c, "t", [[1, "a"]] * 3, ["a", "b"], scope="s", replicated=True,
                     policy=Policy(batch_rows=1), **FAST)  # fmt: skip
    toks = [x["insert_deduplication_token"] for x in c.calls]
    assert len(set(toks)) == 3 and len(c.rows) == 3


def test_token_encoding_is_canonical():
    import datetime
    import decimal
    import uuid

    base = [[{"b": 1, "a": 2}, {3, 1, 2}, {"x", "y", "z"}]]
    same = [[{"a": 2, "b": 1}, {2, 3, 1}, {"z", "y", "x"}]]
    assert batch_token("t", "s", base) == batch_token("t", "s", same)
    assert batch_token("t", "s", [[1]]) != batch_token("t", "s", [[1.0]])
    assert batch_token("t", "s", [[1]]) != batch_token("t", "s", [["1"]])
    assert batch_token("t", "s", [[True]]) != batch_token("t", "s", [[1]])
    assert batch_token("t", "s", [[None]]) != batch_token("t", "s", [["None"]])
    rich = [[datetime.datetime(2026, 1, 2, 3, 4, 5), datetime.date(2026, 1, 2),
             decimal.Decimal("1.50"), uuid.UUID(int=7), b"\x00\xff"]]  # fmt: skip
    assert batch_token("t", "s", rich) == batch_token("t", "s", [list(rich[0])])
    with pytest.raises(TypeError):
        batch_token("t", "s", [[object()]])


def test_the_token_does_not_depend_on_the_hash_seed():
    import subprocess
    import sys

    code = (
        "from spyre_clickhouse_ingest.resilience import batch_token;"
        "print(batch_token('t','s',[[{'alpha','beta','gamma'},{'k1':{'x','y'}}]]))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout  # fmt: skip
        for seed in ("1", "2", "123")
    }
    assert len(outs) == 1


# ── retry bounds ──────────────────────────────────────────────────────────────


def test_the_deadline_is_one_budget_for_the_whole_call(tmp_path):
    now, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s

    # Batch 1 needs 3 retries (6 s of its 10 s budget), batch 2 then fails for good.
    c = Cluster(script=[err(285)] * 3 + [None] + [err(285)] * 99)
    out = run(
        c, Policy(batch_rows=3, deadline_s=10, backoff_s=2, backoff_max_s=2),
        spool=Spool(tmp_path), sleep=sleep, clock=lambda: now[0],
    )  # fmt: skip
    assert out.stored == 3 and out.spooled == 4
    assert sum(slept) <= 10  # not 10 per batch


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-1", "abc", "1e400"])
@pytest.mark.parametrize(
    "var", ["DEADLINE_S", "BACKOFF_S", "BACKOFF_MAX_S", "BATCH_ROWS"]
)
def test_env_values_that_are_not_sane_fall_back_with_one_warning(
    monkeypatch, capsys, var, bad
):
    resilience._warned.clear()
    monkeypatch.setenv("CH_WRITE_" + var, bad)
    default = Policy()
    p1 = Policy.from_env()
    Policy.from_env()
    key = var.lower()
    assert getattr(p1, key) == getattr(default, key)
    assert capsys.readouterr().err.count("CH_WRITE_" + var) == 1


def test_degrade_after_s_zero_is_allowed_but_negative_is_not(monkeypatch):
    resilience._warned.clear()
    monkeypatch.setenv("CH_WRITE_DEGRADE_AFTER_S", "0")
    assert Policy.from_env().degrade_after_s == 0
    monkeypatch.setenv("CH_WRITE_DEGRADE_AFTER_S", "-5")
    assert Policy.from_env().degrade_after_s == Policy().degrade_after_s


def test_sane_env_values_are_used(monkeypatch):
    for k, v in dict(BATCH_ROWS="100", DEADLINE_S="12.5", BACKOFF_S="0.5",
                     BACKOFF_MAX_S="9", QUORUM="auto", DEGRADE_TO="1").items():  # fmt: skip
        monkeypatch.setenv("CH_WRITE_" + k, v)
    p = Policy.from_env()
    assert (p.batch_rows, p.deadline_s, p.backoff_s, p.backoff_max_s) == (
        100,
        12.5,
        0.5,
        9,
    )
    assert p.quorum == "auto" and p.degrade_to == "1"


def test_a_backoff_max_below_backoff_does_not_survive_from_env(monkeypatch):
    resilience._warned.clear()
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "40")
    monkeypatch.setenv("CH_WRITE_BACKOFF_MAX_S", "5")
    p = Policy.from_env()
    assert p.backoff_max_s >= p.backoff_s


@pytest.mark.parametrize(
    "kw",
    [
        dict(deadline_s=float("nan")),
        dict(deadline_s=float("inf")),
        dict(deadline_s=-1),
        dict(backoff_s=0),
        dict(backoff_s=float("nan")),
        dict(backoff_s=5, backoff_max_s=1),
        dict(batch_rows=0),
        dict(batch_rows=float("inf")),
    ],
)
def test_a_policy_built_in_code_refuses_pathological_values(kw):
    with pytest.raises(ValueError):
        Policy(**kw)


def test_a_long_run_of_retries_does_not_overflow():
    c = Cluster(script=[err(285)] * 2000)
    out = run(c, Policy(batch_rows=100, deadline_s=1e12, backoff_s=1, backoff_max_s=2))
    assert out.stored == 7


def test_degrade_needs_a_quorum_to_degrade_from():
    # degrade_to without quorum must not invent a quorum after the degrade window.
    now = [0.0]
    c = Cluster(script=[err(285)] * 4)
    out = run(
        c,
        Policy(batch_rows=100, quorum=None, degrade_to="1", degrade_after_s=1,
               deadline_s=1e6, backoff_s=2),
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        clock=lambda: now[0],
    )  # fmt: skip
    assert out.stored == 7 and not out.degraded
    assert all("insert_quorum" not in call for call in c.calls)


# ── a park that fails still reports the outcome ───────────────────────────────


def _failing_put(exc):
    def put(self, *a, **k):
        raise exc

    return put


@pytest.mark.parametrize(
    "exc",
    [
        OSError(28, "No space left on device"),
        FileNotFoundError("gone"),
        PermissionError("no"),
    ],
)
def test_a_park_failure_on_a_replicated_table_reports_the_outcome(
    tmp_path, monkeypatch, exc
):
    monkeypatch.setattr(Spool, "put", _failing_put(exc))
    c = Cluster(script=[None, err(319)] + [err(285)] * 99, store_on_fail=True)
    with pytest.raises(resilience.WriteAmbiguous) as ei:
        run(c, Policy(batch_rows=3, deadline_s=0, backoff_s=1), spool=Spool(tmp_path))
    out = ei.value.outcome
    assert out.stored == 3 and out.spooled == 0
    assert out.unwritten == [2, 3]
    assert "may be stored" in str(ei.value)


def test_a_full_spool_on_a_replicated_table_keeps_the_type_and_the_outcome(tmp_path):
    c = Cluster(script=[None] + [err(285)] * 99)
    with pytest.raises(resilience.SpoolFull) as ei:
        run(
            c, Policy(batch_rows=3, deadline_s=0, backoff_s=1),
            spool=Spool(tmp_path, max_bytes=1),
        )  # fmt: skip
    assert ei.value.outcome.stored == 3 and ei.value.outcome.unwritten == [2, 3]


def test_a_park_failure_after_a_spill_reports_the_outcome(tmp_path, monkeypatch):
    sp = Spool(tmp_path)
    real = Spool.put
    calls = []

    def put(self, *a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real(self, *a, **k)

    monkeypatch.setattr(Spool, "put", put)
    c = Cluster(script=[err(285)] * 99)
    with pytest.raises(resilience.WriteAmbiguous) as ei:
        run(c, Policy(batch_rows=3, deadline_s=0, backoff_s=1), spool=sp)
    assert ei.value.outcome.spooled == 3 and ei.value.outcome.unwritten == [2, 3]
    assert "may be stored" not in str(ei.value)


def test_without_a_spool_the_deadline_error_carries_the_outcome():
    now = [0.0]
    c = Cluster(script=[err(285)] * 99)
    with pytest.raises(resilience.WriteUnavailable) as ei:
        run(
            c, Policy(batch_rows=3, deadline_s=5, backoff_s=2),
            sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0],
        )  # fmt: skip
    assert ei.value.outcome.unwritten == [1, 2, 3]


# ── spool robustness ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("body", ["[]", "null", '"text"', "3", '{"table": 1}', "{"])
def test_a_json_file_that_is_not_a_batch_is_quarantined_not_fatal(tmp_path, body):
    (tmp_path / "0000000000001-bad.json").write_text(body)
    sp = Spool(tmp_path)
    sp.put("t", ["a"], [[1]], "tok")
    assert [i.token for i in sp.pending()] == ["tok"]
    assert (tmp_path / "0000000000001-bad.json.corrupt").exists()


def test_a_transient_read_error_does_not_quarantine_the_file(tmp_path, monkeypatch):
    sp = Spool(tmp_path)
    p = sp.put("t", ["a"], [[1]], "tok")
    real = Path.read_text

    def flaky(self, *a, **k):
        if self == p:
            raise OSError(5, "Input/output error")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert list(sp.pending()) == []
    assert p.exists() and not p.with_name(p.name + ".corrupt").exists()
    assert sp.problems()["unreadable"] == [p.name]
    monkeypatch.undo()
    assert [i.token for i in sp.pending()] == ["tok"] and not sp.problems()[
        "unreadable"
    ]


def test_corrupt_and_temp_files_count_toward_the_cap_and_are_reported(tmp_path):
    (tmp_path / "0000000000001-bad.json").write_text("{" + "x" * 400)
    sp = Spool(tmp_path, max_bytes=500)
    list(sp.pending())
    (tmp_path / ".junk.1234.tmp").write_text("y" * 200)
    assert sp.problems()["corrupt"] == ["0000000000001-bad.json.corrupt"]
    assert sp.problems()["tmp"] == [".junk.1234.tmp"]
    with pytest.raises(resilience.SpoolFull):
        sp.put("t", ["a"], [[1]], "tok")


def test_a_stale_temp_file_is_cleaned_and_a_young_one_is_kept(tmp_path):
    old, young = tmp_path / ".a.1.tmp", tmp_path / ".b.2.tmp"
    old.write_text("x")
    young.write_text("x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    Spool(tmp_path)
    assert not old.exists() and young.exists()


def test_put_holds_the_spool_lock(tmp_path):
    import threading

    sp = Spool(tmp_path)
    done = threading.Event()
    with sp._locked():
        t = threading.Thread(
            target=lambda: (sp.put("t", ["a"], [[1]], "tok"), done.set())
        )
        t.start()
        assert not done.wait(0.3)  # blocked while another holder has the lock
    assert done.wait(5)
    t.join()


def _put_same_token(path, ambiguous):
    Spool(path).put("t", ["a"], [[1]], "tok", ambiguous=ambiguous)


def test_concurrent_parks_of_one_token_leave_one_held_file(tmp_path):
    ctx = multiprocessing.get_context("fork")
    procs = [
        ctx.Process(target=_put_same_token, args=(str(tmp_path), i == 5))
        for i in range(12)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sp = Spool(tmp_path)
    assert len(sp.files()) == 1
    assert [i.ambiguous for i in sp.pending()] == [True]


def test_a_symlinked_spool_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(OSError):
        Spool(tmp_path / "link")


def test_another_users_spool_directory_is_refused_and_not_chmodded(
    tmp_path, monkeypatch
):
    d = tmp_path / "theirs"
    d.mkdir(mode=0o755)
    os.chmod(d, 0o755)
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(d).st_uid + 1)
    with pytest.raises(PermissionError):
        Spool(d)
    import stat

    assert stat.S_IMODE(os.stat(d).st_mode) == 0o755


def test_your_own_existing_directory_is_accepted_and_locked_down(tmp_path):
    import stat

    d = tmp_path / "mine"
    d.mkdir(mode=0o755)
    os.chmod(d, 0o755)
    Spool(d)
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700


# ── the spool command ─────────────────────────────────────────────────────────


def test_list_exits_75_with_fail_if_pending_only_when_something_is_parked(
    tmp_path, monkeypatch, capsys
):
    from spyre_clickhouse_ingest import spool_cli

    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path))
    sp = Spool(tmp_path)
    assert spool_cli.main(["list", "--fail-if-pending"]) == 0
    sp.put("t", ["a"], [[1]], "tok-1", ambiguous=True)
    assert spool_cli.main(["list"]) == 0  # the default exit is unchanged
    assert spool_cli.main(["list", "--fail-if-pending"]) == 75
    sp.drop("tok-1")
    (tmp_path / "0000000000001-bad.json").write_text("[]")
    capsys.readouterr()
    assert spool_cli.main(["list", "--fail-if-pending"]) == 75
    assert "corrupt file(s)" in capsys.readouterr().err
    assert spool_cli.main(["list", "--bogus"]) == 2
