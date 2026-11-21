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

"""A dead cluster is paid for once, parked work is never read as complete, hooks stay in the CLI."""

import importlib.util
import json
import logging
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError
from test_dry_run import RUN, WHEEL, XML, LiveClient

from spyre_clickhouse_ingest import resilience, results, writes
from spyre_clickhouse_ingest.batching import BatchParked
from guard_helpers import Pol as Policy
from guard_helpers import guarded as resilient_insert
from spyre_clickhouse_ingest.breaker import Breaker, Limits
from spyre_clickhouse_ingest.resilience import Spool, WriteUnavailable
from spyre_clickhouse_ingest.writer import TestResultWriter

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT.parents[1] / ".github" / "scripts"


def err(code):
    return DatabaseError(f"Code: {code}. x", code=code, name="X")


class Clock:
    """A fake `time`: sleeping advances it."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def module(self):
        import time

        return types.SimpleNamespace(
            monotonic=self.monotonic, sleep=self.sleep, time=time.time
        )


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for name in (
        "CH_WRITE_RESILIENCE",
        "CH_WRITE_SPOOL",
        "CH_WRITE_ASSUME_REPLICATED",
        "CH_V2_DATABASES",
        "CH_WRITE_DEADLINE_S",
        "CH_WRITE_BACKOFF_S",
        "RUNNER_TEMP",
        "WORKSPACE",
        "CLICKHOUSE_V2_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(writes, "_parked", [])
    monkeypatch.setattr(writes, "_spools", {})


# ── 1. one deadline per process ────────────────────────────────────────────────


def test_a_shared_breaker_makes_a_dead_cluster_cost_one_deadline(tmp_path):
    clock, breaker, spool = Clock(), Breaker(), Spool(tmp_path / "sp")
    policy = Policy(deadline_s=100, backoff_s=10, backoff_max_s=10, breaker_s=1000)
    sent = []

    class Dead:
        def insert(self, *a, **k):
            sent.append(1)
            raise err(285)

    for i in range(6):
        out = resilient_insert(
            Dead(), "t", [[i]], ["a"], scope=f"s{i}", replicated=True,
            feeds_views=False, policy=policy, spool=spool, sleep=clock.sleep,
            clock=clock.monotonic, breaker=breaker, log=lambda m: None,
        )  # fmt: skip
        assert out.spooled == 1
    assert clock.t <= 110  # one deadline, not six
    assert len(spool.items()) == 6  # every batch is still held durably
    first_call_sends = len(sent)
    resilient_insert(
        Dead(), "t", [[9]], ["a"], scope="s9", replicated=True, feeds_views=False,
        policy=policy, spool=spool, sleep=clock.sleep, clock=clock.monotonic,
        breaker=breaker, log=lambda m: None,
    )  # fmt: skip
    assert len(sent) == first_call_sends  # an open breaker sends nothing


def test_an_open_breaker_without_a_spool_raises_at_once():
    clock, breaker = Clock(), Breaker()
    breaker.trip(0.0, 1000)
    with pytest.raises(WriteUnavailable):
        resilient_insert(
            object(), "t", [[1]], ["a"], scope="s", replicated=True,
            sleep=clock.sleep, clock=clock.monotonic, breaker=breaker,
        )  # fmt: skip
    assert clock.t == 0


def test_the_breaker_closes_after_its_cooldown(tmp_path):
    clock, breaker = Clock(), Breaker()
    breaker.trip(0.0, 50)
    sent = []

    class Up:
        def insert(self, *a, **k):
            sent.append(1)

    clock.t = 60
    out = resilient_insert(
        Up(), "t", [[1]], ["a"], scope="s", replicated=True, spool=Spool(tmp_path / "s"),
        sleep=clock.sleep, clock=clock.monotonic, breaker=breaker, log=lambda m: None,
    )  # fmt: skip
    assert out.stored == 1 and sent
    assert not breaker.tripped  # the probe's success closed it


def test_a_failed_probe_costs_the_probe_deadline_and_reopens(tmp_path):
    clock, breaker, spool = Clock(), Breaker(), Spool(tmp_path / "sp")
    policy = Policy(deadline_s=300, backoff_s=2, backoff_max_s=2, breaker_probe_s=10)
    breaker.trip(0.0, 50)
    clock.t = 60
    sent = []

    class Dead:
        def insert(self, *a, **k):
            sent.append(1)
            raise err(285)

    def call(i):
        return resilient_insert(
            Dead(), "t", [[i]], ["a"], scope=f"s{i}", replicated=True, feeds_views=False,
            policy=policy, spool=spool, sleep=clock.sleep, clock=clock.monotonic,
            breaker=breaker, log=lambda m: None,
        )  # fmt: skip

    assert call(1).spooled == 1
    assert clock.t - 60 <= 12  # the probe, not the full deadline
    n = len(sent)
    assert call(2).spooled == 1 and len(sent) == n  # reopened: nothing is sent


def test_only_one_caller_probes_at_a_time():
    breaker = Breaker()
    breaker.trip(0.0, 10)
    assert breaker.acquire(5.0) == Breaker.OPEN
    assert breaker.acquire(11.0) == Breaker.PROBE
    assert breaker.acquire(11.0) == Breaker.OPEN
    breaker.release()  # a probe that proved nothing
    assert breaker.acquire(12.0) == Breaker.PROBE
    breaker.close()
    assert breaker.acquire(13.0) == Breaker.CLOSED


def test_a_dead_host_does_not_park_the_healthy_one(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(resilience, "time", clock.module())
    monkeypatch.setattr(writes, "time", clock.module())
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "40")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "10")
    monkeypatch.setenv("CH_WRITE_BREAKER_S", "1000")
    monkeypatch.setenv("CH_V2_DATABASES", "db")
    monkeypatch.setenv("CH_WRITE_ASSUME_REPLICATED", "*")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))

    class Host:
        database = "db"

        def __init__(self, name, up):
            self.ch_host, self.up, self.rows = name, up, []

        def query(self, sql, parameters=None):
            if not self.up:
                raise err(210)
            return types.SimpleNamespace(result_rows=[("ReplicatedMergeTree", 0)])

        def insert(self, table, rows, column_names=None, database=None, settings=None):
            if not self.up:
                raise err(210)
            self.rows.extend(rows)

    dead, live = Host("v2.invalid", False), Host("v1.invalid", True)
    out = writes.try_insert(dead, "t", [[1]], ["a"], "db")
    assert out.spooled == 1
    ok = writes.try_insert(live, "t", [[2]], ["a"], "db")
    assert ok.stored == 1 and live.rows == [[2]]
    assert writes.breaker_for(dead, "t", "db").tripped
    assert not writes.breaker_for(live, "t", "db").tripped


def test_an_unread_plain_database_table_is_refused_while_its_host_is_down(monkeypatch):
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "0.001")
    c = Hidden()
    writes.breaker_for(c, "t", "db").trip(time.monotonic(), 1e9)
    with pytest.raises(WriteUnavailable):  # may be Replicated: never sent tokenless
        writes.try_insert(c, "t", [[1]], ["a"], "db")
    assert c.inserts == []


def test_n_tables_on_a_dead_cluster_cost_about_one_deadline(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(resilience, "time", clock.module())
    monkeypatch.setattr(writes, "time", clock.module())
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "100")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "10")
    monkeypatch.setenv("CH_WRITE_BREAKER_S", "1000")
    monkeypatch.setenv("CH_V2_DATABASES", "db")
    monkeypatch.setenv("CH_WRITE_ASSUME_REPLICATED", "*")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    asked = []

    class Dead:
        database = "db"

        def query(self, sql, parameters=None):
            asked.append(sql)
            raise err(210)

        def insert(self, *a, **k):
            asked.append("insert")
            raise err(210)

    c = Dead()
    for i in range(6):
        out = writes.try_insert(c, f"t{i}", [[i]], ["a"], "db")
        assert out is not None and out.spooled == 1
    assert clock.t <= 250  # a few waits of the first table, nothing for the other five
    assert len(Spool(tmp_path / "sp").items()) == 6
    after = len(asked)
    writes.try_insert(c, "t_late", [[1]], ["a"], "db")
    assert len(asked) == after  # nothing is sent while the breaker is open


# ── 7. a table the server does not list fails closed ───────────────────────────


class Hidden:
    database = "db"

    def __init__(self):
        self.inserts = []

    def query(self, sql, parameters=None):
        return types.SimpleNamespace(result_rows=[])

    def insert(self, *a, **k):
        self.inserts.append(a)


def test_a_hidden_table_in_a_replicated_database_is_not_written_plain(monkeypatch):
    monkeypatch.setenv("CH_V2_DATABASES", "db")
    c = Hidden()
    with pytest.raises(WriteUnavailable):
        writes.try_insert(c, "t", [[1]], ["a"], "db")
    assert c.inserts == []


def test_a_hidden_table_can_be_assumed_replicated(monkeypatch, tmp_path):
    monkeypatch.setenv("CH_V2_DATABASES", "db")
    monkeypatch.setenv("CH_WRITE_ASSUME_REPLICATED", "db.t")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))
    c = Hidden()
    out = writes.try_insert(c, "t", [[1]], ["a"], "db")
    assert out is not None and out.stored == 1


def test_a_hidden_table_in_a_plain_database_keeps_the_plain_path():
    assert writes.try_insert(Hidden(), "t", [[1]], ["a"], "db") is None


# ── 6. the exit hook belongs to the CLI ────────────────────────────────────────

PARK = """
import sys
from clickhouse_connect.driver.exceptions import DatabaseError
from spyre_clickhouse_ingest import writes
from spyre_clickhouse_ingest.batching import insert_chunks
{install}
class C:
    database = "d"
    def query(self, sql, parameters=None):
        class R: result_rows = [("ReplicatedMergeTree", 0)]
        return R()
    def insert(self, *a, **k):
        raise DatabaseError("Code: 285. x", code=285, name="X")
try:
    insert_chunks(C(), "t", [[1]], ["a"])
except Exception:
    pass
{tail}
"""


def _status(tmp_path, install="", tail=""):
    p = subprocess.run(
        [sys.executable, "-c", PARK.format(install=install, tail=tail)],
        capture_output=True, text=True, cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT),
             "CH_WRITE_SPOOL": str(tmp_path / "sp"), "CH_WRITE_DEADLINE_S": "0.001"},
    )  # fmt: skip
    return p.returncode


def test_importing_the_module_leaves_the_exit_status_alone(tmp_path):
    assert _status(tmp_path) == 0
    assert _status(tmp_path, tail="raise SystemExit(3)") == 3


def test_a_cli_that_installs_the_hook_exits_75_after_a_park(tmp_path):
    assert _status(tmp_path, install="writes.install_exit_hook()") == 75


def test_the_exit_hook_flushes_logging_before_it_replaces_the_status(monkeypatch):
    calls = []
    monkeypatch.setattr(logging, "shutdown", lambda: calls.append("logging"))
    monkeypatch.setattr(writes.os, "_exit", lambda code: calls.append(code))
    monkeypatch.setattr(writes, "_parked", [("t", 1)])
    monkeypatch.setattr(writes, "_failed", False)
    monkeypatch.setattr(writes, "_exit_code", None)
    writes._finish()
    assert calls == ["logging", writes.EXIT_PARKED]


def _installs_in_entry(path):
    """Is install_exit_hook called inside `entry`/`main` guard of the file, not at import?"""
    import ast

    tree = ast.parse(path.read_text())

    def calls_hook(node):
        return any(
            isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", ""))
            == "install_exit_hook"
            for n in ast.walk(node)
        )

    for node in tree.body:  # only a module-level call would run on import
        if isinstance(node, ast.Expr) and calls_hook(node):
            return False
    for node in tree.body:
        guard = (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and getattr(node.test.left, "id", "") == "__name__"
        )
        entry = isinstance(node, ast.FunctionDef) and node.name == "entry"
        if (guard or entry) and calls_hook(node):
            return True
    return False


def test_every_cli_entry_point_installs_the_hook_inside_its_entry():
    pkg = ROOT / "spyre_clickhouse_ingest"
    named = [
        pkg / f for f in ("__main__.py", "results.py", "dispatch.py", "artifacts.py")
    ]
    named += [
        SCRIPTS / f
        for f in (
            "ingest_hw_diagnostics.py", "ingest_xml.py",
        )
    ]  # fmt: skip
    for path in named:
        assert _installs_in_entry(path), path.name


def test_a_library_consumer_reports_parked_rows_explicitly(tmp_path, capsys):
    writes._parked.append(("t", 3))
    writes.report_parked()
    assert "[PARKED]" in capsys.readouterr().err and writes.parked() == []


# ── 2. no verdict from a run that lost a file ──────────────────────────────────


def _two_files(tmp_path):
    d = tmp_path / "xml"
    d.mkdir()
    (d / "a.xml").write_text(XML)
    (d / "b.xml").write_text(XML)
    return d


def _ingest(monkeypatch, tmp_path, *extra):
    live = LiveClient()
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.setattr(results, "get_client", lambda: live)
    results.main(
        ["--xml-dir", str(_two_files(tmp_path)), "--run-id", RUN, "--run-attempt", "2",
         "--trigger-type", "regression", "--arch", "x86_64", "--artifact", WHEEL,
         "--repository", "torch-spyre/torch-spyre", "--sha", "ab12", *extra]
    )  # fmt: skip
    return live


def _fail_first(monkeypatch, exc):
    real, seen = results.replace_case_attempt, []

    def replace(*a, **k):
        seen.append(1)
        if len(seen) == 1:
            raise exc
        return real(*a, **k)

    monkeypatch.setattr(results, "replace_case_attempt", replace)


def _verdicts(live):
    return [i for i in live.inserts if i[1] == "artifact_results"]


def test_control_all_files_landing_records_the_verdict(monkeypatch, tmp_path):
    assert _verdicts(_ingest(monkeypatch, tmp_path, "--schema", "v2"))


@pytest.mark.parametrize(
    "exc",
    [BatchParked("t: parked", 0, 3), RuntimeError("v2 insert failed")],
    ids=["parked", "failed"],
)
def test_a_run_with_a_parked_or_failed_file_gets_no_verdict(
    monkeypatch, tmp_path, capsys, exc
):
    _fail_first(monkeypatch, exc)
    live = _ingest(monkeypatch, tmp_path, "--schema", "v2")
    assert _verdicts(live) == []
    assert "verdict withheld" in capsys.readouterr().err


def test_withheld_verdicts_fail_a_strict_run(monkeypatch, tmp_path):
    _fail_first(monkeypatch, BatchParked("t: parked", 0, 3))
    with pytest.raises(SystemExit) as e:
        _ingest(monkeypatch, tmp_path, "--schema", "v2", "--strict")
    assert e.value.code == 1


# ── 3. a parked v1 unit does not end the run ───────────────────────────────────


# ── 4. counters: a concurrent leg is not a stale counter ───────────────────────


class Counters:
    def __init__(self, sums, facts=5):
        self.sums, self.facts, self.commands = list(sums), facts, []

    def query(self, sql, parameters=None):
        if "sum(total_tests)" in sql:
            s = self.sums.pop(0) if len(self.sums) > 1 else self.sums[0]
            return types.SimpleNamespace(result_rows=[(s,)])
        if "count()" in sql and "system.columns" not in sql:
            return types.SimpleNamespace(result_rows=[(self.facts,)])
        return types.SimpleNamespace(result_rows=[(1,)])

    def command(self, sql, parameters=None):
        self.commands.append(sql.split()[0] + " " + sql.split()[2])


def test_counters_that_catch_up_within_the_settle_are_not_stale():
    c = Counters([3, 5])  # the other leg's counter rows landed in between
    assert not TestResultWriter._counters_stale(c, "db", RUN, "comp")


def test_counters_that_stay_short_are_stale():
    assert TestResultWriter._counters_stale(Counters([3]), "db", RUN, "comp")


def test_a_recount_that_double_counted_a_concurrent_leg_is_redone():
    c = Counters([8])  # still 8 against 5 facts after the first recount
    TestResultWriter._recount(c, "db", RUN, "comp")
    assert [x for x in c.commands if x.startswith("INSERT")] == [
        "INSERT db.run_case_counters"
    ] * 2


def test_a_recount_that_agrees_is_done_once():
    c = Counters([5])
    TestResultWriter._recount(c, "db", RUN, "comp")
    assert len([x for x in c.commands if x.startswith("INSERT")]) == 1


# ── breaker: only an INSERT closes it, and a host's total cost is capped ───────


def test_a_lookup_that_reads_fine_does_not_close_the_breaker(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(resilience, "time", clock.module())
    monkeypatch.setattr(writes, "time", clock.module())
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "300")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "2")
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))

    class ReadsOnly:  # system.tables answers, every Replicated insert is refused
        database = "db"
        ch_host = "h"

        def query(self, sql, parameters=None):
            return types.SimpleNamespace(result_rows=[("ReplicatedMergeTree", 0)])

        def insert(self, *a, **k):
            raise err(285)

    c = ReadsOnly()
    writes.breaker_for(c, "t", "db").trip(0.0, 50)
    clock.t = 60  # cooldown over: the next caller probes
    out = writes.try_insert(c, "fresh_table", [[1]], ["a"], "db")
    assert out.spooled == 1
    assert clock.t - 60 <= 12  # the probe deadline, not the 300 s one
    assert writes.breaker_for(c, "t", "db").tripped


def test_a_failing_table_cannot_cost_a_deadline_each_time_another_succeeds():
    clock, breaker = Clock(), Breaker()
    policy = Policy(
        deadline_s=300, backoff_s=30, backoff_max_s=30, breaker_s=60, budget_s=900,
        budget_window_s=10**9,
    )  # fmt: skip
    spool_dir = None  # no spool: a failing write raises, the loop moves on

    class Table:
        def __init__(self, ok):
            self.ok = ok

        def insert(self, *a, **k):
            if not self.ok:
                raise err(285)

    a, b = Table(False), Table(True)

    def put(client, i):
        try:
            resilient_insert(
                client, "t", [[i]], ["x"], scope=str(i), replicated=True,
                feeds_views=False, policy=policy, spool=spool_dir,
                sleep=clock.sleep, clock=clock.monotonic, breaker=breaker,
                log=lambda m: None,
            )  # fmt: skip
        except WriteUnavailable:
            pass

    waited = 0.0
    for i in range(20):
        clock.t += 100  # past the cooldown, so each round probes again
        before = clock.t
        put(b, i)  # the success closes the breaker...
        put(a, i)  # ...and a pays its deadline again, until the budget is spent
        waited += clock.t - before
    assert waited <= 900 + 300 + 100


def test_a_write_trip_does_not_end_a_probe_in_flight():
    breaker = Breaker()
    breaker.trip(0.0, 10)
    assert breaker.acquire(11.0) == Breaker.PROBE
    breaker.trip(12.0, 10)  # a writer that started before the breaker opened
    assert breaker.acquire(30.0) == Breaker.OPEN  # still being probed
    breaker.trip(31.0, 10, from_probe=True)
    assert breaker.acquire(50.0) == Breaker.PROBE


def test_clients_without_a_host_get_distinct_stable_keys():
    class C:
        pass

    a, b = C(), C()
    ka, kb = writes._host_of(a, "t", None), writes._host_of(b, "t", None)
    assert ka != kb and writes._host_of(a, "t", None) == ka


# ── breaker budget: real elapsed time, refilled after a quiet window ───────────


def test_a_healthy_but_slow_host_is_never_charged_or_parked(tmp_path):
    clock, breaker = Clock(), Breaker()
    policy = Policy(batch_rows=1, breaker_probe_s=10, budget_s=900)

    class Slow:
        def insert(self, *a, **k):
            clock.t += 20  # every call lands, after longer than the probe deadline

    out = resilient_insert(
        Slow(), "t", [[i] for i in range(100)], ["a"], scope="s", replicated=True,
        policy=policy, sleep=clock.sleep, clock=clock.monotonic, breaker=breaker,
        log=lambda m: None,
    )  # fmt: skip
    assert out.stored == 100 and out.spooled == 0 and breaker.spent == 0


def test_the_cost_of_a_host_drains_so_a_recovering_one_is_forgiven():
    breaker = Breaker()
    policy = Policy(budget_s=900, budget_window_s=1800)
    t = 1000.0
    for _ in range(20):  # 100 s of trouble every 25 minutes, for eight hours
        breaker.charge(100, t, policy)
        assert breaker.acquire(t) == Breaker.CLOSED
        t += 1500
    assert breaker.spent < 900


def test_an_exhausted_budget_parks_the_host_for_a_window_then_starts_over():
    breaker = Breaker()
    policy = Policy(budget_s=100, budget_window_s=1800)
    breaker.charge(150, 1000.0, policy)
    assert breaker.acquire(1100.0) == Breaker.OPEN
    assert breaker.acquire(2900.0) == Breaker.CLOSED
    assert breaker.spent == 0.0


def test_one_bad_table_neither_trips_nor_drains_the_host(tmp_path):
    clock, breaker = Clock(), Breaker()
    policy = Policy(deadline_s=300, backoff_s=30, backoff_max_s=30, budget_s=900)

    class Bad:
        def insert(self, *a, **k):
            raise err(252)  # too many parts: this table's own refusal

    class Good:
        def insert(self, *a, **k):
            pass

    for i in range(4):
        out = resilient_insert(
            Bad(), "bad", [[i]], ["a"], scope=f"b{i}", replicated=True,
            feeds_views=False, policy=policy, spool=Spool(tmp_path / "sp"),
            sleep=clock.sleep, clock=clock.monotonic, breaker=breaker, log=lambda m: None,
        )  # fmt: skip
        assert out.spooled == 1
    ok = resilient_insert(
        Good(), "good", [[1]], ["a"], scope="g", replicated=True, policy=policy,
        sleep=clock.sleep, clock=clock.monotonic, breaker=breaker, log=lambda m: None,
    )  # fmt: skip
    assert ok.stored == 1 and breaker.spent == 0 and not breaker.tripped


def test_a_batch_after_a_breaker_park_cannot_land_ahead_of_it(tmp_path):
    class Flaky(Breaker):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def acquire(self, now):
            self.calls += 1
            return Breaker.OPEN if self.calls == 1 else Breaker.CLOSED

    sent = []

    class Up:
        def insert(self, *a, **k):
            sent.append(1)

    out = resilient_insert(
        Up(), "t", [[1], [2], [3]], ["a"], scope="s", replicated=True,
        policy=Policy(batch_rows=1), spool=Spool(tmp_path / "sp"), breaker=Flaky(),
        log=lambda m: None,
    )  # fmt: skip
    assert out.spooled == 3 and not sent  # the later batches followed the first


@pytest.mark.parametrize("bad", ["nan", "-5", "0", "inf", "abc"])
def test_unusable_budget_settings_fall_back_to_the_default(monkeypatch, bad, capsys):
    for name in ("BUDGET_S", "BUDGET_WINDOW_S", "BREAKER_PROBE_S", "BREAKER_S",
                 "BACKOFF_S"):  # fmt: skip
        monkeypatch.setenv("CH_WRITE_" + name, bad)
    for _ in range(50):  # from_env runs on every insert
        p, limits = Policy.from_env(), Limits.from_env()
    assert limits == Limits() and p.backoff_s == 2
    assert capsys.readouterr().err.count("ignored (") == 5  # once per setting


def test_retries_stop_at_the_remaining_budget(tmp_path):
    clock, breaker = Clock(), Breaker()
    policy = Policy(deadline_s=300, backoff_s=10, backoff_max_s=10, budget_s=50)

    class Dead:
        def insert(self, *a, **k):
            raise err(285)

    out = resilient_insert(
        Dead(), "t", [[1]], ["a"], scope="s", replicated=True, feeds_views=False,
        policy=policy, spool=Spool(tmp_path / "sp"), sleep=clock.sleep,
        clock=clock.monotonic, breaker=breaker, log=lambda m: None,
    )  # fmt: skip
    assert out.spooled == 1 and clock.t <= 50  # not the 300 s deadline


def test_a_lookup_is_charged_its_real_elapsed_time(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(writes, "time", clock.module())
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "300")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "2")

    class Slow:
        database = "db"
        ch_host = "h"

        def query(self, sql, parameters=None):
            clock.t += 120  # a connect or read timeout
            raise err(210)

    c = Slow()
    writes.target(c, "t", "db")
    assert writes.breaker_for(c, "t", "db").spent >= 240


# ── v1 keeps its legacy path: no engine lookup, plain inserts ──────────────────


def test_v1_inserts_never_look_up_an_engine_or_carry_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(LiveClient, "ENGINE", "ReplicatedMergeTree")
    queries, live = [], None

    class Watched(LiveClient):
        def query(self, sql, parameters=None):
            queries.append(sql)
            return super().query(sql, parameters)

    live = Watched()
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setattr(results, "get_client", lambda: live)
    results.main(["--xml-dir", str(_two_files(tmp_path)), "--run-id", RUN, "--schema",
                  "v1", "--trigger-type", "regression", "--arch", "x86_64"])  # fmt: skip
    assert not [q for q in queries if "system.tables" in q]
    v1 = {t for _, t, _ in live.inserts}
    assert {"test_runs", "test_cases"} <= v1
    assert all(settings == {} for _, settings in live.settings)


def test_the_hw_diagnostics_v1_insert_is_a_plain_insert():
    from spyre_clickhouse_ingest.hw_diagnostics import RecordLoader
    from spyre_clickhouse_ingest.hw_schema import HwFailureDiagnostics

    seen = []

    class C:
        def insert(self, **kw):
            seen.append(kw)

    cols = HwFailureDiagnostics.COLUMN_NAMES
    RecordLoader.insert(C(), [[None] * len(cols)], "hw", resilient=False)
    assert seen and seen[0]["table"] == "hw"


def test_a_lookup_that_succeeds_after_retries_is_charged_the_time_it_lost(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(writes, "time", clock.module())
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "600")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "2")

    class Slow:
        database = "db"
        ch_host = "h"
        left = 2

        def query(self, sql, parameters=None):
            if self.left:
                self.left -= 1
                clock.t += 120
                raise err(210)
            return types.SimpleNamespace(result_rows=[("ReplicatedMergeTree", 0)])

    c = Slow()
    assert writes.target(c, "t", "db").replicated
    assert writes.breaker_for(c, "t", "db").spent >= 240


def test_three_tables_on_a_dead_replicated_host_all_park_whatever_their_order(
    monkeypatch, tmp_path
):
    clock = Clock()
    monkeypatch.setattr(resilience, "time", clock.module())
    monkeypatch.setattr(writes, "time", clock.module())
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "40")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "10")
    monkeypatch.setenv("CH_V2_DATABASES", "db")  # no CH_WRITE_ASSUME_REPLICATED
    monkeypatch.setenv("CH_WRITE_SPOOL", str(tmp_path / "sp"))

    class Dead:
        database = "db"
        ch_host = "v2"

        def query(self, sql, parameters=None):
            raise err(210)

        def insert(self, *a, **k):
            raise err(210)

    c = Dead()
    outs = [writes.try_insert(c, t, [[1]], ["a"], "db") for t in ("a", "b", "c")]
    assert [o.spooled for o in outs] == [1, 1, 1]
    assert len(Spool(tmp_path / "sp").items()) == 3


# ── the v1 scripts insert plainly ──────────────────────────────────────────────


class Plain:
    """Any query answers empty; records inserts, and fails the test on a lookup."""

    database = "spyre"

    def __init__(self):
        self.inserts, self.lookups = [], []

    def command(self, sql, *a, **k):
        return 1

    def query(self, sql, *a, **k):
        if "system.tables" in sql:
            self.lookups.append(sql)
        return types.SimpleNamespace(result_rows=[[0]])

    def insert(self, *a, **k):
        self.inserts.append((a, k))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_model_ops_inserts_plainly(monkeypatch, tmp_path):
    mod = _script("ingest_model_ops")
    c = Plain()
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setattr(mod, "get_client", lambda role="writer": c)
    monkeypatch.setattr(mod, "ensure_tables", lambda x: None)
    monkeypatch.setattr(mod, "build_suite_row", lambda *a: ["s"])
    monkeypatch.setattr(mod, "_build_variant_rows", lambda *a: [["v"]])
    data = tmp_path / "ops.json"
    data.write_text(json.dumps([{"suite_name": "suite", "model_name": "m"}]))
    monkeypatch.setattr(sys, "argv", ["x", "--json-file", str(data)])
    mod.main()
    assert [a[0] for a, _ in c.inserts] == ["model_ops_suites", "model_ops_variants"]
    assert not c.lookups and all("settings" not in k for _, k in c.inserts)


def test_pytorch_dispatch_inserts_plainly(monkeypatch, tmp_path):
    mod = _script("ingest_pytorch_dispatch")
    c = Plain()
    monkeypatch.setattr(mod, "get_client", lambda role="writer": c)
    payload = tmp_path / "p.json"
    payload.write_text(json.dumps({"delivery_id": "d1"}))
    monkeypatch.setattr(sys, "argv", ["x", "--payload-file", str(payload)])
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    mod.main()
    assert c.inserts and not c.lookups
    assert c.inserts[0][1].get("table") == "pytorch_ci_dispatches"


def test_hw_diagnostics_v1_leg_inserts_plainly(monkeypatch, tmp_path):
    mod = _script("ingest_hw_diagnostics")
    c = Plain()
    monkeypatch.setattr(mod, "get_client", lambda role="writer": c)
    monkeypatch.setattr(mod, "ensure_extra_columns", lambda *a, **k: None)
    monkeypatch.setattr(mod, "build_row", lambda rec, ctx: [None] * len(mod_cols()))
    data = tmp_path / "hw.json"
    data.write_text(json.dumps([{"suite_name": "s", "run_id": "7"}]))
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--json-file", str(data), "--schema", "v1", "--trigger-type", "regression",
         "--arch", "x86_64", "--gha-run-id", "7", "--table", "hw_failure_diagnostics"],
    )  # fmt: skip
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    mod.main()
    assert c.inserts and not c.lookups
    assert c.inserts[0][1]["table"] == "hw_failure_diagnostics"


def mod_cols():
    from spyre_clickhouse_ingest.hw_schema import HwFailureDiagnostics

    return HwFailureDiagnostics.COLUMN_NAMES


# ── accounting per attempt, 242 as the host's, 252 bounded ─────────────────────


def _dead_tables(tmp_path, code, n, calls=1):
    clock, breaker = Clock(), Breaker()
    policy = Policy(deadline_s=300, backoff_s=30, backoff_max_s=30, breaker_s=60,
                    budget_s=10_000, budget_window_s=10**9)  # fmt: skip
    spool = Spool(tmp_path / "sp")

    class Dead:
        def insert(self, *a, **k):
            raise err(code)

    for c in range(calls):
        for i in range(n):
            resilient_insert(
                Dead(), f"t{i}", [[c]], ["a"], scope=f"{c}.{i}", replicated=True,
                feeds_views=False, policy=policy, spool=spool, sleep=clock.sleep,
                clock=clock.monotonic, breaker=breaker, log=lambda m: None,
            )  # fmt: skip
    return clock, breaker, spool


def test_a_keeper_outage_on_ten_tables_costs_about_one_deadline(tmp_path):
    clock, breaker, spool = _dead_tables(tmp_path, 242, 10, calls=4)
    assert clock.t <= 330  # 242 is the host's: the first table trips it, the rest park
    assert len(spool.items()) == 40


def test_too_many_parts_on_ten_tables_costs_two_deadlines_at_most(tmp_path):
    clock, breaker, spool = _dead_tables(tmp_path, 252, 10, calls=4)
    assert clock.t <= 660  # one table on its own, then a second makes it the host's
    assert len(spool.items()) == 40


def test_timeouts_before_a_final_table_refusal_still_cost_the_host(tmp_path):
    clock, breaker = Clock(), Breaker()
    policy = Policy(deadline_s=300, backoff_s=1, backoff_max_s=1, breaker_s=60,
                    budget_s=10_000)  # fmt: skip
    codes = iter([210] * 2 + [252] * 1000)

    class Mixed:
        def insert(self, *a, **k):
            clock.t += 100  # a timeout, or the refusal that comes after it
            raise err(next(codes))

    out = resilient_insert(
        Mixed(), "t", [[1]], ["a"], scope="s", replicated=True, feeds_views=False,
        policy=policy, spool=Spool(tmp_path / "sp"), sleep=clock.sleep,
        clock=clock.monotonic, breaker=breaker, log=lambda m: None,
    )  # fmt: skip
    assert out.spooled == 1
    assert breaker.tripped and breaker.spent >= 200


def test_guard_an_early_table_refusal_does_not_zero_the_cost_of_a_later_outage(
    tmp_path,
):
    clock, breaker = Clock(), Breaker()
    policy = Policy(deadline_s=600, backoff_s=1, backoff_max_s=1, budget_s=10_000)
    codes = iter([252, 210, 210, None])

    class Mixed:
        def insert(self, *a, **k):
            clock.t += 50
            code = next(codes)
            if code:
                raise err(code)

    out = resilient_insert(
        Mixed(), "t", [[1]], ["a"], scope="s", replicated=True, feeds_views=False,
        policy=policy, sleep=clock.sleep, clock=clock.monotonic, breaker=breaker,
        log=lambda m: None,
    )  # fmt: skip
    assert out.stored == 1 and breaker.spent >= 100  # the two timeouts were the host's


def test_replay_against_a_dead_host_costs_one_deadline_not_one_per_item(
    monkeypatch, tmp_path
):
    clock = Clock()
    monkeypatch.setattr(resilience, "time", clock.module())
    monkeypatch.setattr(writes, "time", clock.module())
    monkeypatch.setenv("CH_WRITE_DEADLINE_S", "40")
    monkeypatch.setenv("CH_WRITE_BACKOFF_S", "10")
    monkeypatch.setenv("CH_V2_DATABASES", "db")
    monkeypatch.setenv("CH_WRITE_ASSUME_REPLICATED", "*")
    spool = Spool(tmp_path / "sp")
    for i in range(4):
        spool.put(f"t{i}", ["a"], [[i]], f"tok-{i}", "db", scope="s")

    class Dead:
        database = "db"
        ch_host = "v2"

        def query(self, sql, parameters=None):
            raise err(210)

        def insert(self, *a, **k):
            raise err(210)

    c = Dead()
    for item in list(spool.pending()):
        assert writes.replay(c, item).startswith("not stored")
    assert clock.t <= 120  # the first item pays; the others find the breaker open


def test_a_scoped_refusal_is_forgotten_once_the_table_takes_an_insert(tmp_path):
    clock, breaker = Clock(), Breaker()
    policy = Policy(deadline_s=300, backoff_s=30, backoff_max_s=30, budget_s=10_000)
    state = {"a": True}

    class Table:
        def __init__(self, name):
            self.name = name

        def insert(self, *a, **k):
            clock.t += 5
            if self.name == "a" and state["a"]:
                raise err(252)

    def put(name, db="d"):
        return resilient_insert(
            Table(name), name, [[1]], ["x"], scope=f"{name}{clock.t}", replicated=True,
            feeds_views=False, policy=policy, database=db, spool=Spool(tmp_path / "sp"),
            sleep=clock.sleep, clock=clock.monotonic, breaker=breaker, log=lambda m: None,
        )  # fmt: skip

    put("a")  # a is overloaded for a while: its own trouble, not the host's
    assert not breaker.tripped
    state["a"] = False
    clock.t += 100
    assert put("a").stored == 1  # recovered: forgotten
    clock.t += 1400  # still inside the default window
    state["a"] = True  # now a healthy table b has one bad moment of its own
    state_b = Table("b")
    state_b.insert = lambda *a, **k: (_ for _ in ()).throw(err(252))
    before = clock.t
    out = resilient_insert(
        state_b, "b", [[1]], ["x"], scope="b", replicated=True, feeds_views=False,
        policy=policy, database="d", spool=Spool(tmp_path / "sp"), sleep=clock.sleep,
        clock=clock.monotonic, breaker=breaker, log=lambda m: None,
    )  # fmt: skip
    assert out.spooled == 1 and not breaker.tripped and breaker.spent == 0
    assert (
        clock.t - before >= 270
    )  # it still waited its own deadline (less one jittered wait)


def test_same_table_name_in_two_databases_is_two_tables(tmp_path):
    breaker = Breaker()
    assert not breaker.note_scoped("v1.t", 0.0, 1800)
    assert breaker.note_scoped("v2.t", 10.0, 1800)  # a second table: now the host's


def test_a_keeper_outage_seen_through_one_table_first_realistic_rounds(tmp_path):
    """Table a (read-only, 242) is written first every round, b and c are healthy.

    Recorded outcome: a is host-class, so it trips the breaker and is the probe each round
    (~12 s); b and c, written after it, are parked every round until a is fixed. The first
    round costs one deadline; the budget never runs out because later rounds are probes.
    """
    clock, breaker = Clock(), Breaker()
    policy = (
        Policy()
    )  # the defaults: 300 s deadline, 60 s cooldown, 10 s probe, 900/1800
    spool = Spool(tmp_path / "sp")

    class A:
        def insert(self, *a, **k):
            clock.t += 5
            raise err(242)

    class Ok:
        def insert(self, *a, **k):
            clock.t += 1

    costs = {"a": [], "b": [], "c": []}
    parked = {"b": 0, "c": 0}
    for rnd in range(20):
        clock.t += 100  # past the cooldown: a probes again
        for name, client in (("a", A()), ("b", Ok()), ("c", Ok())):
            before = clock.t
            out = resilient_insert(
                client, name, [[rnd]], ["x"], scope=f"{rnd}{name}", replicated=True,
                feeds_views=False, policy=policy, spool=spool, sleep=clock.sleep,
                clock=clock.monotonic, breaker=breaker, log=lambda m: None,
            )  # fmt: skip
            costs[name].append(clock.t - before)
            if name != "a":
                parked[name] += out.spooled
    assert costs["a"][0] <= 305 and max(costs["a"][1:]) <= 15
    assert max(costs["b"] + costs["c"]) == 0
    assert parked == {"b": 20, "c": 20}  # parked, not lost: replay restores them
    assert breaker.spent < 900
