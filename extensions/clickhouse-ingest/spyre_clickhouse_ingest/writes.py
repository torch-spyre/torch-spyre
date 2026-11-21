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
its own path. Rows that cannot land are parked in the spool; the call raises BatchParked and the
CLI exits EXIT_PARKED after writing a rows-free summary.
"""

from __future__ import annotations

import datetime
import decimal
import os
import sys
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
_ENGINE_SQL = (
    "SELECT engine, length(dependencies_table) FROM system.tables "
    "WHERE database = {db:String} AND name = {t:String}"
)
_ENGINE_SQL_CURRENT = _ENGINE_SQL.replace("{db:String}", "currentDatabase()")

_parked: list[tuple[str, int]] = []
_spools: dict[str, Spool | None] = {}
_calls: dict[tuple[str, str], int] = {}
_process = uuid.uuid4().hex


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
    _calls[key] = _calls.get(key, 0) + 1
    return f"{_process}:{key[0]}.{table}#{_calls[key]}"


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
    found = target(backend, table, database)
    if found is None:
        return None
    policy = Policy.from_env()
    policy.batch_rows, policy.degrade_to = batch_rows, None
    try:
        out = resilient_insert(
            backend,
            table,
            rows,
            columns,
            scope=scope_for(database, table),
            policy=policy,
            replicated=True,
            dedup_window=dedup_window(
                backend, table, database, _cache(backend, "_ch_write_windows")
            ),
            database=database,
            spool=spool_from_env(),
            feeds_views=found.feeds_views,
        )
    except WriteFailed as err:
        stored = err.outcome.stored if err.outcome else 0
        raise BatchFailed(f"{table}: {err}", stored) from err
    if out.spooled:
        _parked.append((f"{database}.{table}" if database else table, out.spooled))
        raise BatchParked(
            f"{table}: {out.spooled} row(s) parked, {out.stored} stored",
            out.stored,
            out.spooled,
        )
    return out.stored


def parked() -> list[tuple[str, int]]:
    return list(_parked)


def finish(rc):
    """`rc`, or EXIT_PARKED when it is success and rows were parked; reports and forgets them."""
    if not _parked:
        return rc
    totals: dict[str, int] = {}
    for table, n in _parked:
        totals[table] = totals.get(table, 0) + n
    _parked.clear()
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
    return EXIT_PARKED if rc in (0, None) else rc


def replay(client, item: SpoolItem) -> str:
    """Send one parked batch again under its original token: "sent", or why it was not.

    Only a Replicated table with a nonzero dedup window qualifies, so a batch that did land is a
    no-op. A batch that may be stored is also refused when the window is unreadable or the
    batch is older than CH_WRITE_REPLAY_MAX_AGE_S (default 3600), since ClickHouse may have
    evicted its token.
    """
    check_token(item.token)
    backend = _backend(client, item.table, item.database)
    found = target(backend, item.table, item.database)
    if found is None:
        return "not a Replicated table, or its engine is unreadable: check by hand"
    window = dedup_window(
        backend, item.table, item.database, _cache(backend, "_ch_write_windows")
    )
    if window == 0:
        return "the table deduplicates nothing (window 0): check by hand"
    if item.ambiguous:
        if window is None:
            return "may be stored and the dedup window is unreadable: check by hand"
        max_age = float(_env("CH_WRITE_REPLAY_MAX_AGE_S") or 3600)
        age = time.time() - (item.created or 0)
        if age > max_age:
            return f"may be stored and is {age / 3600:.1f} h old: check by hand"
    settings = _settings(Policy.from_env(), item.token, True, False, found.feeds_views)
    try:
        rows = _typed(backend, item)
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


def _typed(backend, item: SpoolItem) -> list:
    """The parked rows with their top-level DateTime, Date, UUID and Decimal cells restored."""
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
    for col in item.columns:
        kind = types.get(col, "")
        for wrapper in ("Nullable(", "LowCardinality("):
            kind = kind[len(wrapper) :] if kind.startswith(wrapper) else kind
        parsers.append(next((f for k, f in _PARSE if kind.startswith(k)), None))
    return [
        [f(v) if f and isinstance(v, str) else v for f, v in zip(parsers, row)]
        for row in item.rows
    ]
