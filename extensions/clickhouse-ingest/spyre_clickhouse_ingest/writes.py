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

"""Sends inserts into Replicated tables through `resilience.resilient_insert`.

`CH_WRITE_RESILIENCE=auto` (default; `off` disables) applies it when the target's engine is
Replicated*. A plain table, or one whose engine cannot be read, returns None and the caller keeps
its own path. Rows that cannot land are parked in the spool and the call raises BatchParked. Once
a call to a table has parked, later calls to that table get one attempt each, so an outage costs
one CH_WRITE_DEADLINE_S per table, not one per call. `finish` turns a clean run that parked rows into
EXIT_PARKED, unless the run also wrote the authoritative v1 rows (see `finish`).
"""

from __future__ import annotations

import datetime
import decimal
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass

from .batching import MAX_ROWS, BatchFailed, BatchParked
from .resilience import (
    Policy,
    Spool,
    SpoolItem,
    WriteFailed,
    _settings,
    check_token,
    dedup_window,
    resilient_insert,
    safe,
)

EXIT_PARKED = 75
# replicated_deduplication_window's server default (10000 on 26.3, the tests' image), assumed when
# the setting is readable neither in system.merge_tree_settings nor through getMergeTreeSetting().
DEFAULT_WINDOW = 10000
# Fact tables whose writers also dedup per case, replace older attempts and feed counters and
# verdicts: a replay would skip all of that, so a parked batch of these is re-ingested instead.
NO_REPLAY = frozenset({"test_case_runs", "capability_runs", "artifact_results"})
_ENGINE_SQL = (
    "SELECT engine, length(dependencies_table) FROM system.tables "
    "WHERE database = {db:String} AND name = {t:String}"
)
_ENGINE_SQL_CURRENT = _ENGINE_SQL.replace("{db:String}", "currentDatabase()")

_parked: dict[str, int] = {}  # table -> rows; bounded by the tables a process writes
_spools: dict[str, Spool | None] = {}
_calls: dict[tuple[str, str], int] = {}
_process = uuid.uuid4().hex
# "spent": (database, table) -> when a call to it last used up its deadline; cleared by the next
# call to that table that stores its rows, so one failing table never slows a healthy one.
_state: dict = {"spent": {}, "v1": False}
_lock = threading.Lock()


def reset() -> None:
    """Forget this process's parked rows, spools, call counts and spent deadline (tests)."""
    with _lock:
        _parked.clear()
        _spools.clear()
        _calls.clear()
        _state.update(spent={}, v1=False)


def _spent_recently(key, deadline_s: float) -> bool:
    """A call to this table spent its deadline less than one deadline ago."""
    with _lock:
        spent = _state["spent"].get(key)
        if spent is not None and time.monotonic() - spent >= deadline_s:
            del _state["spent"][key]
            spent = None
        return spent is not None


def _mark_spent(key, spent: bool) -> None:
    with _lock:
        if spent:
            _state["spent"][key] = time.monotonic()
        else:
            _state["spent"].pop(key, None)


def note_v1_written() -> None:
    """The run also writes the authoritative v1 rows: a park alone does not fail it."""
    _state["v1"] = True


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def enabled() -> bool:
    return _env("CH_WRITE_RESILIENCE").lower() not in ("off", "0", "false", "no")


@dataclass(frozen=True)
class Target:
    feeds_views: bool


def _backend(client, table: str, database: str | None):
    route = getattr(client, "route", None)
    return route(table=table, database=database) if callable(route) else client


def _flag(name: str) -> bool:
    return _env(name).lower() not in ("", "0", "false", "no", "off")


def _cache(backend, name: str) -> dict:
    cache = getattr(backend, name, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(backend, name, cache)
        except Exception:  # noqa: BLE001 -- a client that takes no attributes is not cached
            pass
    return cache


def target(backend, table: str, database: str | None) -> Target | None:
    """A Replicated target and whether a materialized view reads it; None for anything else,
    including a lookup that fails or a table the login cannot see. Read once per client."""
    key = (database or "", table)
    cache = _cache(backend, "_ch_write_targets")
    if key in cache:
        return cache[key]
    sql, params = (
        (_ENGINE_SQL, {"db": database, "t": table})
        if database
        else (_ENGINE_SQL_CURRENT, {"t": table})
    )
    found = None
    try:
        rows = backend.query(sql, parameters=params).result_rows
        if rows and str(rows[0][0]).startswith("Replicated"):
            found = Target(feeds_views=bool(rows[0][1]))
    except Exception as err:  # noqa: BLE001 -- unreadable keeps the caller's own path
        print(f"[warn] engine lookup for {table} failed: {safe(err)}", file=sys.stderr)
        return None
    cache[key] = found
    return found


def window(backend, table: str, database: str | None) -> int:
    """The table's dedup window, or DEFAULT_WINDOW (logged once) when it cannot be read."""
    cache = _cache(backend, "_ch_write_windows")
    found = dedup_window(backend, table, database, cache)
    if found is None:
        try:  # readable by logins that may not read system.merge_tree_settings
            found = cache[(database or "", table)] = int(
                backend.command(
                    "SELECT getMergeTreeSetting('replicated_deduplication_window')"
                )
            )
        except Exception:  # noqa: BLE001 -- older servers lack the function
            pass
    if found is None:
        print(
            f"[warn] dedup window of {table} unreadable; assuming the server default "
            f"{DEFAULT_WINDOW}",
            file=sys.stderr,
        )
        found = cache[(database or "", table)] = DEFAULT_WINDOW
    return found


def _aware(row):
    """Naive datetimes as aware local time: the same instant the driver sends, and the spool
    then records the offset, so a replay on another host's time zone stores the same value."""
    return [
        c.astimezone() if isinstance(c, datetime.datetime) and c.tzinfo is None else c
        for c in row
    ]


def spool_dir() -> str:
    """CH_WRITE_SPOOL, else the job-private CI scratch directory."""
    if _env("CH_WRITE_SPOOL"):
        return _env("CH_WRITE_SPOOL")
    return os.path.join(_env("RUNNER_TEMP"), "ch-spool") if _env("RUNNER_TEMP") else ""


def spool_from_env() -> Spool | None:
    where = spool_dir()
    if not where:
        return None
    if where not in _spools:
        try:
            _spools[where] = Spool(where)
        except OSError as err:
            print(f"[warn] spool {where} unusable ({err}); no parking", file=sys.stderr)
            _spools[where] = None
    return _spools[where]


def scope_for(database: str | None, table: str) -> str:
    """Unique per call in this process, so identical rows of two calls are never merged."""
    key = (database or "", table)
    with _lock:
        _calls[key] = n = _calls.get(key, 0) + 1
    return f"{_process}:{key[0]}.{table}#{n}"


def insert(
    client, table, rows, columns, database=None, batch_rows: int = MAX_ROWS
) -> int | None:
    """Rows stored through resilient_insert, or None when the target is not Replicated.

    Raises BatchParked when rows were parked and BatchFailed when they could not land.
    """
    if not enabled() or not rows:
        return None
    if not database and "." in table:
        database, _, table = table.partition(".")
    backend = _backend(client, table, database)
    database = database or getattr(backend, "database", None) or None
    found = target(backend, table, database)
    if found is None:
        return None
    policy = Policy.from_env()
    policy.batch_rows, policy.degrade_to = batch_rows, None
    key = (database or "", table)
    if _spent_recently(key, policy.deadline_s):
        policy.deadline_s = 0.0
    try:
        out = resilient_insert(
            backend,
            table,
            [_aware(r) for r in rows],
            columns,
            scope=scope_for(database, table),
            policy=policy,
            replicated=True,
            dedup_window=window(backend, table, database),
            database=database,
            spool=spool_from_env(),
            feeds_views=found.feeds_views,
        )
    except WriteFailed as err:
        _mark_spent(key, True)
        stored = err.outcome.stored if err.outcome else 0
        raise BatchFailed(f"{table}: {err}", stored) from err
    _mark_spent(key, bool(out.spooled))
    if out.spooled:
        name = f"{database}.{table}" if database else table
        with _lock:
            _parked[name] = _parked.get(name, 0) + out.spooled
        raise BatchParked(
            f"{table}: {out.spooled} row(s) parked, {out.stored} stored",
            out.stored,
            out.spooled,
        )
    return out.stored


def parked() -> dict[str, int]:
    return dict(_parked)


def finish(rc):
    """Report and forget parked rows; `rc`, or EXIT_PARKED for a clean run that parked them.

    A run that also wrote the v1 rows (authoritative) stays clean with a warning, unless
    CH_WRITE_STRICT is set.
    """
    with _lock:
        totals, v1 = dict(_parked), _state["v1"]
        _parked.clear()
        _state["v1"] = False  # one run per finish
    if not totals:
        return rc
    where = spool_dir() or "(no spool)"
    lines = [f"{t}: {n} row(s) parked" for t, n in sorted(totals.items())]
    print(
        f"[PARKED] rows not stored, held in {where}: {'; '.join(lines)}\n"
        "  see: python -m spyre_clickhouse_ingest spool list",
        file=sys.stderr,
    )
    if _env("GITHUB_STEP_SUMMARY"):
        try:
            with open(_env("GITHUB_STEP_SUMMARY"), "a") as fh:
                fh.write("### ClickHouse rows parked\n\n")
                fh.write("".join(f"- {line}\n" for line in lines))
                fh.write(f"\nSpool: `{where}`\n")
        except OSError:
            pass
    if rc not in (0, None):
        return rc
    if v1 and not _flag("CH_WRITE_STRICT"):
        print(
            "::warning title=ClickHouse::v2 rows parked, not stored; v1 rows were written"
        )
        return rc
    return EXIT_PARKED


def run_cli(main, *args):
    """`main(*args)`'s exit status, with parked rows folded in by `finish`."""
    try:
        rc = main(*args)
    except SystemExit as stop:
        rc = stop.code
    return finish(rc)


def replay(client, item: SpoolItem) -> str:
    """Send one parked batch again under its original token: "sent", or why it was not.

    Only a Replicated table with a nonzero dedup window qualifies, so a batch that did land is a
    no-op. A batch that may be stored is also refused when it is older than
    CH_WRITE_REPLAY_MAX_AGE_S (default 3600, the server's replicated_deduplication_window_seconds),
    since ClickHouse may have evicted its token. Tables in NO_REPLAY are never replayed. Replay
    before any re-run of the same ingest: a re-run sends the rows again under new tokens.
    """
    check_token(item.token)
    if item.table in NO_REPLAY:
        return (
            f"{item.table} is not replayed: re-run the ingest for this file (a replay would "
            "skip the per-case dedup, the older-attempt delete, the counters and the verdict)"
        )
    backend = _backend(client, item.table, item.database)
    found = target(backend, item.table, item.database)
    if found is None:
        return "not a Replicated table, or its engine is unreadable: check by hand"
    if window(backend, item.table, item.database) == 0:
        return "the table deduplicates nothing (window 0): check by hand"
    if item.ambiguous:
        max_age = float(_env("CH_WRITE_REPLAY_MAX_AGE_S") or 3600)
        age = time.time() - (item.created or 0)
        if age > max_age:
            return f"may be stored and is {age / 3600:.1f} h old: check by hand"
    settings = _settings(Policy.from_env(), item.token, True, False, found.feeds_views)
    try:
        rows, why = _typed(backend, item)
        if why:
            return f"{why}: check by hand"
        kw = {"database": item.database} if item.database else {}
        backend.insert(
            item.table, rows, column_names=list(item.columns), settings=settings, **kw
        )
    except Exception as err:  # noqa: BLE001 -- reported per batch; the file stays
        return f"not stored: {safe(err)}"
    return "sent"


# The spool keeps cells as JSON text; these types are turned back into what the first send carried.
_PARSE = (
    ("DateTime", datetime.datetime.fromisoformat),
    ("Date", datetime.date.fromisoformat),
    ("UUID", uuid.UUID),
    ("Decimal", decimal.Decimal),
)
_NESTED = ("Array(", "Map(", "Tuple(", "Nested(", "Variant(", "Dynamic", "JSON")


def _unwrap(kind: str) -> str:
    changed = True
    while changed:
        changed = False
        for wrapper in ("Nullable(", "LowCardinality("):
            if kind.startswith(wrapper) and kind.endswith(")"):
                kind, changed = kind[len(wrapper) : -1], True
    return kind


def _bytes_text(v) -> bool:
    """A cell, or a value inside it, that the spool wrote from bytes as `b'...'` text."""
    if isinstance(v, str):
        return v[:2] in ("b'", 'b"')
    if isinstance(v, dict):
        return any(_bytes_text(x) for kv in v.items() for x in kv)
    return isinstance(v, list) and any(_bytes_text(x) for x in v)


def _typed(backend, item: SpoolItem) -> tuple[list, str]:
    """(rows with DateTime, Date, UUID and Decimal cells restored, why they cannot be)."""
    where = (
        "database = {db:String}" if item.database else "database = currentDatabase()"
    )
    types = dict(
        backend.query(
            f"SELECT name, type FROM system.columns WHERE {where} AND table = {{t:String}}",
            parameters={"db": item.database or "", "t": item.table},
        ).result_rows
    )
    parsers = []
    for i, col in enumerate(item.columns):
        kind = _unwrap(types.get(col, ""))
        if kind.startswith(_NESTED) and any(k in kind for k, _ in _PARSE):
            return [], f"column {col} nests {kind}, which the spool flattened to text"
        if "String" in kind and any(_bytes_text(r[i]) for r in item.rows):
            return [], f"column {col} holds bytes that the spool wrote as text"
        parsers.append(next((f for k, f in _PARSE if kind.startswith(k)), None))
    rows = [
        [f(v) if f and isinstance(v, str) else v for f, v in zip(parsers, row)]
        for row in item.rows
    ]
    return rows, ""
