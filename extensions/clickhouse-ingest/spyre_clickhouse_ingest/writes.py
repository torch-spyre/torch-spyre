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

"""The one switch that routes ingest inserts through `breaker.guarded_insert`
(`resilience.resilient_insert` behind a per-host breaker).

`CH_WRITE_RESILIENCE=auto` (default) applies it to tables whose engine is Replicated*; a plain
table, or one whose engine cannot be read, keeps the caller's own path unchanged. `off` disables
it. The server profile of the writer login owns `insert_quorum`; this module only lowers it per
request for the DEGRADABLE tables, never for a strict one.
"""

from __future__ import annotations

import atexit
import itertools
import logging
import os
import sys
import time
from dataclasses import dataclass

from .breaker import Breaker, Limits, guarded_insert
from .resilience import (
    Outcome,
    Policy,
    Spool,
    WriteUnavailable,
    classify,
    dedup_window,
    new_scope,
    safe,
)

EXIT_PARKED = 75
LOOKUP_FAILURE_TTL_S = 60.0
# Rebuilt from CI history by the backfills; every other table is strict (quorum never lowered).
DEGRADABLE = frozenset({"pipeline_runs", "pipeline_run_links", "ci_run_timings"})

_ENGINE_SQL = (
    "SELECT engine, length(dependencies_table) FROM system.tables "
    "WHERE database = {db:String} AND name = {t:String}"
)
_ENGINE_SQL_CURRENT = _ENGINE_SQL.replace("{db:String}", "currentDatabase()")

_parked: list[tuple[str, int]] = []
_spools: dict[str, Spool | None] = {}
# One per host: a dead host costs one deadline, however many inserts and tables follow, and a
# healthy host is not parked for it. Per table would pay a deadline for every table.
_breakers: dict[str, Breaker] = {}
_client_keys = itertools.count(1)


def _host_of(client, table: str, database: str | None) -> str:
    backend = (
        client.route(table=table, database=database)
        if hasattr(client, "route")
        else client
    )
    host = getattr(backend, "ch_host", None)
    if host:
        return str(host)
    key = getattr(
        backend, "_spyre_host_key", None
    )  # no host: one breaker per client object
    if key is None:
        key = f"@{next(_client_keys)}"
        try:
            backend._spyre_host_key = key
        except Exception:  # noqa: BLE001 -- a client that refuses attributes shares one key
            key = "@"
    return key


def breaker_for(client, table: str, database: str | None) -> Breaker:
    return _breakers.setdefault(_host_of(client, table, database), Breaker())


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def enabled() -> bool:
    return _env("CH_WRITE_RESILIENCE").lower() not in ("off", "0", "false", "no")


_process_scope = ""


def ambient_scope() -> str:
    """What a batch token is scoped to: this attempt of this job (and leg), else this process.

    A token exists so a RESEND of the same batch inside one attempt is a no-op. A re-run
    (another attempt, another leg, another process) is a new batch and gets new tokens.
    """
    global _process_scope
    if _env("CH_WRITE_SCOPE"):
        return _env("CH_WRITE_SCOPE")
    ci = [
        _env(n)
        for n in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "CH_WRITE_LEG")
        if _env(n)
    ]
    if ci:
        return ".".join(ci)
    if _env("BUILD_TAG"):
        return _env("BUILD_TAG")
    _process_scope = _process_scope or new_scope()
    return _process_scope


@dataclass(frozen=True)
class Target:
    replicated: bool
    feeds_views: bool


def _assumed(db: str, table: str) -> bool:
    """CH_WRITE_ASSUME_REPLICATED: `*` (every table of a CH_V2_DATABASES database), or a comma
    list of `table` / `db.table`."""
    from .client import ClickHouseEnv

    listed = {
        x.strip() for x in _env("CH_WRITE_ASSUME_REPLICATED").split(",") if x.strip()
    }
    if "*" in listed and db in ClickHouseEnv.v2_databases():
        return True
    return bool({table, f"{db}.{table}"} & listed)


UNAVAILABLE = object()  # the lookup kept failing for a reason a retry could cure
HIDDEN = object()  # the lookup worked and system.tables does not list the table
BREAKER_OPEN = object()  # not asked: the cluster already cost this process a deadline


def _unreadable(db: str, table: str, kind: str) -> Target | None:
    """The answer when the engine cannot be read: "down" (the server could not be asked),
    "hidden" (it does not list the table) or "denied" (the lookup failed in a way a retry cannot
    cure: a refusal, a proxy 4xx, an unexpected answer).

    In a Replicated database a table is Replicated, so when the server is down it is parked for
    replay (the same answer the open breaker gives); a table it does not list, or a lookup the
    login may not run, raises: nothing could replay the first, and the second must not become a
    tokenless write. Outside such a database a denied lookup keeps the plain path.
    CH_WRITE_ASSUME_REPLICATED overrides all three.
    """
    from .client import ClickHouseEnv

    if _assumed(db, table):
        return Target(True, True)
    if db in ClickHouseEnv.v2_databases():
        if kind == "down":
            return Target(True, True)
        if kind == "hidden":
            raise WriteUnavailable(
                f"{db}.{table}: the server does not list this table; not writing untokened"
            )
        if kind == "denied":
            raise WriteUnavailable(
                f"{db}.{table}: the engine lookup failed (see the warning above for the cause); "
                "not writing untokened into a Replicated database. Set "
                "CH_WRITE_ASSUME_REPLICATED to send it with a token"
            )
    return None


def _lookup(
    client, db: str, table: str, deadline_s: float, backoff_s: float, breaker: Breaker
):
    """Engine facts, retrying a retriable failure inside the deadline; None if unreadable.

    UNAVAILABLE when the server could not be asked, or does not list the table.
    """
    sql, params = (
        (_ENGINE_SQL, {"db": db, "t": table})
        if db
        else (_ENGINE_SQL_CURRENT, {"t": table})
    )
    started = time.monotonic()
    mode = breaker.acquire(started)
    if mode == Breaker.OPEN:
        return BREAKER_OPEN
    if mode == Breaker.PROBE:
        deadline_s = 0.0  # one try
    try:
        return _read_engine(
            client, sql, params, table, started, deadline_s, backoff_s, breaker, mode
        )
    finally:
        if (
            mode == Breaker.PROBE
        ):  # a read proves nothing about writes: the probe is not won
            breaker.release()


def _read_engine(
    client, sql, params, table, started, deadline_s, backoff_s, breaker, mode
):
    attempt = 0
    limits = Limits.from_env()
    breaker.charge(0.0, started, limits)  # publishes the budget
    deadline_s = min(deadline_s, breaker.remaining(started))
    while True:
        attempt_started = time.monotonic()
        try:
            found = client.query(sql, parameters=params).result_rows
            if not found:
                print(
                    f"[warn] {table} is not listed in system.tables; treating it as "
                    "unreadable",
                    file=sys.stderr,
                )
                return HIDDEN
            engine, deps = found[0][:2]
            if attempt:  # retries before it answered were time the host cost
                breaker.charge(attempt_started - started, time.monotonic(), limits)
            if isinstance(engine, str) and isinstance(deps, int):
                return Target(engine.startswith("Replicated"), bool(deps))
            return None
        except Exception as err:  # noqa: BLE001
            attempt += 1
            pause = min(backoff_s * 2 ** (attempt - 1), 30.0)
            fatal = classify(err) == "fatal"
            elapsed = time.monotonic() - started  # real time, timeouts included
            if fatal or elapsed + pause > deadline_s:
                if not fatal:
                    breaker.charge(elapsed, time.monotonic(), limits)
                    breaker.trip(
                        time.monotonic(),
                        limits.breaker_s,
                        from_probe=mode == Breaker.PROBE,
                    )
                print(
                    f"[warn] engine lookup for {table} failed: {safe(err)}",
                    file=sys.stderr,
                )
                return None if fatal else UNAVAILABLE
            time.sleep(pause)


def target(client, table: str, database: str | None) -> Target | None:
    """Engine facts for `database.table`, cached on the client for its lifetime (a table
    converted mid-process is not noticed; clients here are per process).

    A lookup that still fails after retries is treated as Replicated, with views, when
    CH_WRITE_ASSUME_REPLICATED names the table or the table is in a Replicated database
    (CH_V2_DATABASES), so it is parked; a table such a database does not list raises
    WriteUnavailable; otherwise None, and the caller takes the plain path. A failure is
    remembered for LOOKUP_FAILURE_TTL_S, and one that exhausts its deadline opens its host's
    breaker, so N tables cost one deadline (while it is open an unread table in a Replicated
    database is assumed Replicated and parked, any other raises WriteUnavailable).
    """
    if not database and "." in table:
        database, _, table = table.partition(".")
    db = database or getattr(client, "database", None) or ""
    key = (db, table)
    cache = getattr(client, "_spyre_targets", None)
    if isinstance(cache, dict) and key in cache:
        hit = cache[key]
        if not isinstance(hit, tuple):
            return hit
        if (
            time.monotonic() < hit[0]
        ):  # a recent failed lookup: do not wait the deadline again
            return _unreadable(db, table, hit[1])
    policy = Policy.from_env()
    found = _lookup(
        client,
        db,
        table,
        policy.deadline_s,
        policy.backoff_s,
        breaker_for(client, table, db),
    )
    if found is BREAKER_OPEN:
        # Engine unknown and the host is down: a Replicated database's tables are Replicated, so
        # park for replay; any other unread table may be Replicated too, so refuse, never send it
        # tokenless.
        from .client import ClickHouseEnv

        if _assumed(db, table) or db in ClickHouseEnv.v2_databases():
            return Target(True, True)
        raise WriteUnavailable(
            f"{db}.{table}: engine unknown and its host is unavailable; not writing untokened"
        )
    if found is None or found is UNAVAILABLE or found is HIDDEN:
        kind = {None: "denied"}.get(found) or (
            "down" if found is UNAVAILABLE else "hidden"
        )
        try:
            if not isinstance(cache, dict):
                cache = client._spyre_targets = {}
            cache[key] = (time.monotonic() + LOOKUP_FAILURE_TTL_S, kind)
        except Exception:  # noqa: BLE001 -- a client that refuses attributes is just not cached
            pass
        return _unreadable(db, table, kind)
    try:
        if not isinstance(cache, dict):
            cache = client._spyre_targets = {}
        cache[key] = found
    except Exception:  # noqa: BLE001 -- a client that refuses attributes is just not cached
        pass
    return found


def spool_dir() -> str:
    """CH_WRITE_SPOOL, else the CI scratch dir that is private to the job."""
    if _env("CH_WRITE_SPOOL"):
        return _env("CH_WRITE_SPOOL")
    for var, leaf in (("RUNNER_TEMP", "ch-spool"), ("WORKSPACE", ".ch-spool")):
        if _env(var):
            return os.path.join(_env(var), leaf)
    return ""


def oldest_age(spool: Spool) -> float | None:
    """Seconds since the oldest parked file was written, None when the spool is empty."""
    mtimes = []
    for p in spool.files():
        try:
            mtimes.append(p.stat().st_mtime)
        except OSError:
            pass
    return spool.clock() - min(mtimes) if mtimes else None


def spool_from_env() -> Spool | None:
    where = spool_dir()
    if not where:
        return None
    if where not in _spools:
        try:
            _spools[where] = Spool(where)
            age = oldest_age(_spools[where])
            if age is not None and age > 86400:
                print(
                    f"[warn] spool {where} holds {len(_spools[where].files())} batch(es), "
                    f"oldest {age / 3600:.0f} h: handle them (spool list) before it fills",
                    file=sys.stderr,
                )
        except OSError as err:
            print(
                f"[warn] spool {where} unusable ({err}); parking disabled",
                file=sys.stderr,
            )
            _spools[where] = None
    return _spools[where]


def policy_for(table: str, batch_rows: int) -> Policy:
    policy = Policy.from_env()
    policy.batch_rows = batch_rows
    if table not in DEGRADABLE:
        policy.degrade_to = None
    return policy


def window_of(backend, table: str, database: str | None) -> int | None:
    """The table's dedup window, read once per client (an unreadable one for LOOKUP_FAILURE_TTL_S);
    None (unknown) assumes a nonzero one."""
    cache = getattr(backend, "_spyre_windows", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            backend._spyre_windows = cache
        except Exception:  # noqa: BLE001 -- a client that refuses attributes is just not cached
            pass
    key = (database or "", table)
    hit = cache.get(key)
    if hit and (hit[1] is None or time.monotonic() < hit[1]):
        return hit[0]
    window = dedup_window(backend, table, database)
    cache[key] = (
        window,
        None if window is not None else time.monotonic() + LOOKUP_FAILURE_TTL_S,
    )
    return window


def try_insert(
    client, table, rows, columns, database=None, batch_rows: int = 5000
) -> Outcome | None:
    """Insert through resilience when the target is Replicated; None means "not handled".

    Raises WriteUnavailable (deadline, no spool) or SpoolFull; both mean rows did not land.
    """
    if not enabled():
        return None
    if not database and "." in table:
        database, _, table = table.partition(".")
    found = target(client, table, database)
    if not (found and found.replicated):
        return None
    backend = (
        client.route(table=table, database=database)
        if hasattr(client, "route")
        else client
    )
    breaker = breaker_for(client, table, database)
    out = guarded_insert(
        client,
        table,
        rows,
        columns,
        scope=ambient_scope(),
        policy=policy_for(table, batch_rows),
        replicated=True,
        # a dead host is not asked again: the batch goes to the spool whatever the window is
        dedup_window=None
        if breaker.blocked(time.monotonic())
        else window_of(backend, table, database),
        database=database,
        spool=spool_from_env(),
        feeds_views=found.feeds_views,
        host=getattr(backend, "ch_host", None),
        breaker=breaker,
        sleep=time.sleep,  # one clock for the breaker, the lookup and the retry loop
        clock=time.monotonic,
    )
    if out.spooled:
        _note_parked(table, out.spooled)
    return out


def insert(client, table, rows, column_names, database=None) -> None:
    """Drop-in for `client.insert(table, rows, column_names=...)` at the single-shot v1 sites."""
    out = try_insert(client, table, rows, column_names, database)
    if out is None:
        kw = {"database": database} if database else {}
        client.insert(table, rows, column_names=list(column_names), **kw)
    elif out.spooled:
        from .batching import BatchParked

        raise BatchParked(
            f"{table}: {out.spooled} row(s) parked, {out.stored} stored",
            out.stored,
            out.spooled,
        )


_failed = False
_exit_code = None
_hooked = False


def install_exit_hook() -> None:
    """Make a run that parked rows exit 75 unless it fails for a worse reason.

    For CLI entry points only: it wraps `sys.exit` and replaces the process status at exit, which
    a library consumer must not inherit from a mere import.
    """
    global _hooked
    if _hooked:
        return
    _hooked = True
    atexit.register(_finish)
    plain_exit, plain_hook = sys.exit, sys.excepthook

    def exit_(code=None):
        global _exit_code
        _exit_code = code
        plain_exit(code)

    def hook(kind, value, tb):
        global _failed
        _failed = not isinstance(value, ParkedRows)
        plain_hook(kind, value, tb)

    sys.exit, sys.excepthook = exit_, hook


class ParkedRows(Exception):
    """Marker base for exceptions that mean "rows were parked, not lost"."""


def _finish() -> None:
    had = bool(_parked)
    report_parked()
    clean = not _failed and _exit_code in (None, 0)
    if had and clean:
        try:
            logging.shutdown()  # os._exit skips the handlers registered before this one
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(EXIT_PARKED)


def _note_parked(table: str, rows: int) -> None:
    _parked.append((table, rows))


def parked() -> list[tuple[str, int]]:
    return list(_parked)


def report_parked(stream=None) -> None:
    """One loud summary of what was parked, also on the GitHub step summary when there is one."""
    if not _parked:
        return
    stream = stream or sys.stderr
    totals: dict[str, int] = {}
    for table, n in _parked:
        totals[table] = totals.get(table, 0) + n
    where = spool_dir() or "(no spool)"
    lines = [f"{t}: {n} row(s) parked" for t, n in sorted(totals.items())]
    print(
        f"[PARKED] rows not stored, held in {where}: " + "; ".join(lines),
        "\n  list: python -m spyre_clickhouse_ingest spool list",
        file=stream,
    )
    summary = _env("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a") as fh:
                fh.write(
                    "### ClickHouse rows parked\n\n"
                    + "\n".join(f"- {line}" for line in lines)
                    + f"\n\nSpool: `{where}`\n"
                )
        except OSError:
            pass
    _parked.clear()


def exit_code(rc) -> int:
    """`rc` unless it is success and rows were parked, which is EXIT_PARKED.

    A non-zero result is recorded so the exit hook leaves that status alone.
    """
    global _exit_code
    code = 0 if rc in (None, 0) else (rc if isinstance(rc, int) else 1)
    code = EXIT_PARKED if code == 0 and _parked else code
    if code:
        _exit_code = code
    return code


def clear_exit() -> None:
    """Forget a recorded exit status that the caller caught (a SystemExit swallowed in-process)."""
    global _exit_code
    _exit_code = None


def replay(client, item) -> str:
    """Send one parked batch again under its original token. Returns "sent" or why it was not.

    Only a Replicated table with a dedup window is replayed: the token makes a batch that already
    landed a no-op. A batch that may be stored is refused when it was sent without a token, is
    older than CH_WRITE_REPLAY_MAX_AGE_S, or was parked for another host; a plain table, a table
    with window 0 and an unreadable engine are left for a person.
    """
    from .resilience import _send, _settings, check_token

    check_token(item.token)
    backend = (
        client.route(table=item.table, database=item.database)
        if hasattr(client, "route")
        else client
    )
    host = getattr(backend, "ch_host", None)
    parked_for = item.meta.get("host")
    if parked_for and host and parked_for != host:
        return (
            f"parked for host {parked_for}, connected to {host}: connect to that host"
        )
    breaker = breaker_for(client, item.table, item.database)
    if breaker.blocked(time.monotonic()):
        return "not stored: the host is unavailable (breaker open)"
    found = target(client, item.table, item.database)
    if not (found and found.replicated):
        return "target is not a Replicated table (or its engine is unreadable): check by hand"
    if window_of(backend, item.table, item.database) == 0:
        return "the table deduplicates nothing (window 0), so a resend could double: check by hand"
    if item.ambiguous:
        if not item.meta.get("token_sent", True):
            return "ambiguous and sent without a token: check by hand"
        max_age = float(_env("CH_WRITE_REPLAY_MAX_AGE_S") or 3600)
        age = time.time() - (item.created or 0)
        if age > max_age:
            return (
                f"ambiguous and {age / 3600:.1f} h old, past the {max_age / 3600:.1f} h dedup "
                "window (CH_WRITE_REPLAY_MAX_AGE_S): check by hand"
            )
    policy, limits = policy_for(item.table, max(1, len(item.rows))), Limits.from_env()
    start = time.monotonic()
    attempt = 0
    while True:
        mode = breaker.acquire(time.monotonic())
        if mode == Breaker.OPEN:
            return "not stored: the host is unavailable (breaker open)"
        try:
            _send(
                backend, item.table, item.rows, item.columns, item.database,
                _settings(policy, item.token, True, False, found.feeds_views),
            )  # fmt: skip
        except Exception as err:  # noqa: BLE001
            kind = classify(err, found.feeds_views)
            attempt += 1
            pause = min(policy.backoff_s * 2 ** (attempt - 1), policy.backoff_max_s)
            spent = time.monotonic() - start
            # The same token makes a resend safe whatever the error said.
            if kind == "fatal" or spent + pause > policy.deadline_s:
                if kind != "fatal":
                    breaker.trip(
                        time.monotonic(),
                        limits.breaker_s,
                        from_probe=mode == Breaker.PROBE,
                    )
                elif mode == Breaker.PROBE:
                    breaker.release()
                return f"not stored: {safe(err)}"
            time.sleep(pause)
        else:
            if breaker.tripped:
                breaker.close()
            return "sent"
