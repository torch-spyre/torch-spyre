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

"""Writes that survive a degraded replicated cluster, without losing or doubling a batch.

With insert_quorum set, an INSERT fails while fewer replicas are up than the quorum asks for,
during a pod restart, or when Keeper blips.

* `classify` sorts an error into retry (nothing was stored), ambiguous (the block may be
  stored) or fatal (a resend cannot help).
* On a Replicated table every batch carries a deterministic `insert_deduplication_token`, so
  resending an ambiguous batch inside the dedup window cannot double it, and neither can a second
  sender.
* A plain table has no such guard: a batch that may be stored is never sent again here. It is
  parked for a person, or the call raises WriteAmbiguous when there is no spool to hold it.
* Retries back off to a total deadline. A table whose data is regenerable may then drop to a
  lower quorum; a strict one never does.
* Past the deadline a batch is parked as a file, with the token it was sent under. Nothing in
  this module sends parked batches again: see the README for how a person handles them.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# Refused before the write on the table itself (quorum, read-only, parts limit). A materialized
# view's target can raise the same codes after the source block landed, so with a dependent view
# they prove nothing; ClickHouse's order between the source commit and the view pushes is not
# something the docs or chdb let us rely on.
PRE_WRITE_CODES = frozenset({242, 252, 285, 286})
# The block may be stored on some replicas or in some views: coordination, timeouts, network,
# and a memory limit hit while a materialized view was being fed.
AMBIGUOUS_CODES = frozenset({159, 209, 210, 241, 319, 999, 1000})
# A resend cannot change the answer: schema, permission, syntax and constraint errors.
FATAL_CODES = frozenset({16, 27, 36, 43, 47, 53, 60, 62, 81, 439, 469, 497, 516})


class WriteUnavailable(RuntimeError):
    """The cluster did not accept the write before the deadline and no spool was given."""


class SpoolFull(RuntimeError):
    """The spool is at its size cap; the write fails loudly instead of dropping a batch."""


class WriteAmbiguous(RuntimeError):
    """A plain-table batch may or may not be stored and there is no spool to hold it."""

    def __init__(self, message: str, outcome: "Outcome"):
        super().__init__(message)
        self.outcome = outcome


def _code(err: BaseException) -> int | None:
    code = getattr(err, "code", None)
    if isinstance(code, int):
        return code
    text = str(err)
    head, found, tail = text.partition("Code: ")
    digits = "".join(ch for ch in tail[:5] if ch.isdigit())
    return int(digits) if found and digits else None


def _never_connected(err: BaseException) -> bool:
    """The connection was refused or the host did not resolve: no request left the client.

    Walks the exception chain by type; the message text is not trusted (it can be hidden by the
    driver and can carry row data).
    """
    try:
        from urllib3.exceptions import MaxRetryError, NewConnectionError
    except ImportError:  # pragma: no cover
        MaxRetryError = NewConnectionError = ()  # type: ignore[assignment,misc]
    seen: set[int] = set()
    e: BaseException | None = err
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, (ConnectionRefusedError, socket.gaierror)):
            return True
        if NewConnectionError and isinstance(e, NewConnectionError):
            return True
        if MaxRetryError and isinstance(e, MaxRetryError):
            if isinstance(getattr(e, "reason", None), NewConnectionError):
                return True
        e = e.__cause__ or e.__context__
    return False


def safe(err: BaseException) -> str:
    """What may be logged or put in an exception message: the type and server code, never the
    driver's text, which can carry row data."""
    code = _code(err)
    return type(err).__name__ + (f"(code={code})" if code is not None else "")


def classify(err: BaseException, feeds_views: bool = True) -> str:
    """'retry', 'ambiguous' or 'fatal'. The server's code beats the message text.

    The pre-write codes (read-only, too many parts, not enough replicas, unsatisfied quorum) are a
    retry only when the table feeds no materialized view; with one, a view's target can raise them
    after the source block landed. Callers that know the table has no dependent view pass
    feeds_views=False.
    """
    code = _code(err)
    if code in FATAL_CODES:
        return "fatal"
    if code in PRE_WRITE_CODES:
        return "ambiguous" if feeds_views else "retry"
    if code in AMBIGUOUS_CODES:
        return "ambiguous"
    if _never_connected(err):
        return "retry"
    if isinstance(err, OSError):
        return "ambiguous"
    try:
        from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
    except ImportError:
        return "fatal"
    if isinstance(err, OperationalError):
        return "ambiguous"
    if isinstance(err, DatabaseError) and "HTTP status 5" in str(err):
        return "ambiguous"
    return "fatal"


def batch_token(
    table: str, scope: str, rows: Sequence[Sequence], index: int = 0
) -> str:
    """A stable token for one batch: the same rows in the same order, at the same position,
    under the same scope. The position keeps two identical batches in one load apart."""
    h = hashlib.sha256()
    h.update(f"{table}\x00{scope}\x00{index}\x00".encode())
    h.update(json.dumps(rows, default=str, separators=(",", ":")).encode())
    return f"{table}-{h.hexdigest()[:40]}"


@dataclass
class Policy:
    """Per-writer settings. `quorum` None sends no insert_quorum; `degrade_to` None is strict."""

    batch_rows: int = 5000
    deadline_s: float = 300.0
    backoff_s: float = 2.0
    backoff_max_s: float = 30.0
    quorum: str | None = None
    degrade_to: str | None = None
    degrade_after_s: float = 60.0

    @classmethod
    def from_env(cls, prefix: str = "CH_WRITE_") -> "Policy":
        def f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(prefix + name, default))
            except ValueError:
                return default

        quorum = os.environ.get(prefix + "QUORUM") or None
        degrade = os.environ.get(prefix + "DEGRADE_TO") or None
        return cls(
            batch_rows=int(f("BATCH_ROWS", 5000)),
            deadline_s=f("DEADLINE_S", 300),
            backoff_s=f("BACKOFF_S", 2),
            backoff_max_s=f("BACKOFF_MAX_S", 30),
            quorum=quorum,
            degrade_to=degrade,
            degrade_after_s=f("DEGRADE_AFTER_S", 60),
        )


@dataclass
class SpoolItem:
    path: Path
    table: str
    columns: list
    rows: list
    token: str
    database: str | None
    created: float | None = None
    # True when a failed attempt may have stored the batch. A batch refused before any write
    # (quorum, read-only) is not ambiguous.
    ambiguous: bool = False
    scope: str | None = None


def _write_private(path: Path, doc: dict) -> None:
    """Write `doc` to `path` through an owner-only temp file, so a kill leaves the old file or
    the new one."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Spool:
    """A directory of whole-file JSON batches that could not be sent. It holds row data, so the
    directory is 0700 and the files 0600. Nothing here sends a batch again.

    A batch a failed attempt may have stored carries `"ambiguous": true` in its JSON. Parking the
    same token again keeps one file and can only set that flag, never clear it. A full spool
    raises SpoolFull instead of dropping anything.
    """

    def __init__(
        self, directory, *, max_bytes: int = 256 * 1024 * 1024, clock=time.time
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.dir, 0o700)
        self.max_bytes, self.clock = max_bytes, clock

    def files(self, token: str | None = None) -> list[Path]:
        """Item files, oldest first; only those of one batch when `token` is given."""
        return sorted(
            p
            for p in self.dir.iterdir()
            if p.name[:1].isdigit()
            and p.name.endswith(".json")
            and (token is None or p.name.endswith(f"-{token}.json"))
        )

    def put(
        self,
        table,
        columns,
        rows,
        token,
        database=None,
        *,
        created: float | None = None,
        ambiguous: bool = False,
        scope: str | None = None,
    ) -> Path:
        created = self.clock() if created is None else float(created)
        doc = {
            "table": table,
            "columns": list(columns),
            "rows": [list(r) for r in rows],
            "token": token,
            "database": database,
            "created": created,
            "ambiguous": bool(ambiguous),
            "scope": scope,
        }
        existing = self.files(token)
        if existing:
            if ambiguous:
                old = json.loads(existing[0].read_text())
                if not old.get("ambiguous"):
                    old["ambiguous"] = True
                    _write_private(existing[0], old)
            return existing[0]
        used = sum(p.stat().st_size for p in self.files())
        if used + len(json.dumps(doc, default=str)) > self.max_bytes:
            raise SpoolFull(
                f"spool holds {used} bytes; cap {self.max_bytes}. Writing more would drop data"
            )
        # Strictly increasing, so files sort in the order they were written.
        ms = max(int(created * 1000), getattr(self, "_last_ms", 0) + 1)
        self._last_ms = ms
        final = self.dir / f"{ms:013d}-{token}.json"
        _write_private(final, doc)
        return final

    def pending(self) -> Iterator[SpoolItem]:
        for p in self.files():
            try:
                d = json.loads(p.read_text())
                yield SpoolItem(
                    p,
                    d["table"],
                    d["columns"],
                    d["rows"],
                    d["token"],
                    d.get("database"),
                    None if d.get("created") is None else float(d["created"]),
                    bool(d.get("ambiguous")),
                    d.get("scope"),
                )
            except (ValueError, KeyError, OSError):
                try:
                    p.rename(p.with_name(p.name + ".corrupt"))
                except OSError:
                    pass

    def items(self) -> list[dict]:
        """One summary per parked batch, without its rows."""
        return [
            {
                "token": i.token,
                "table": i.table,
                "rows": len(i.rows),
                "held": i.ambiguous,
                "created": i.created,
                "file": i.path.name,
            }  # fmt: skip
            for i in self.pending()
        ]

    def drop(self, token: str) -> int:
        """Delete a parked batch, after a person has dealt with its rows."""
        check_token(token)
        files = self.files(token)
        for p in files:
            p.unlink(missing_ok=True)
        return len(files)


_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def check_token(token: str) -> None:
    """A token is plain characters only: it is matched against file names."""
    if (
        not token
        or len(token) > 200
        or set(token) - _TOKEN_CHARS
        or token in (".", "..")
    ):
        raise ValueError(f"not a spool token: {token!r}")


@dataclass
class Outcome:
    stored: int = 0
    spooled: int = 0
    ambiguous: list = field(default_factory=list)
    degraded: bool = False


def _settings(
    policy: Policy,
    token: str | None,
    replicated: bool,
    degraded: bool,
    feeds_views: bool = True,
) -> dict:
    s: dict[str, Any] = {}
    if replicated:
        if token:
            s["insert_deduplication_token"] = token
        if feeds_views:
            # Without it a retry whose source block was deduplicated may skip a view target that
            # missed the block the first time (older servers default to 0). The target needs its
            # own dedup window (Replicated, or non_replicated_deduplication_window).
            s["deduplicate_blocks_in_dependent_materialized_views"] = 1
        s["async_insert_deduplicate"] = 1
        s["wait_for_async_insert"] = 1
        quorum = policy.degrade_to if degraded else policy.quorum
        if quorum:
            s["insert_quorum"] = quorum
    return s


def _send(client, table, batch, columns, database, settings):
    kw = {"database": database} if database else {}
    if settings:
        kw["settings"] = settings
    client.insert(table, batch, column_names=list(columns), **kw)


def resilient_insert(
    client,
    table: str,
    rows: Sequence[Sequence],
    columns: Sequence[str],
    *,
    scope: str,
    policy: Policy | None = None,
    replicated: bool = False,
    database: str | None = None,
    spool: Spool | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    feeds_views: bool = True,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> Outcome:
    """Insert `rows` in batches, surviving quorum and Keeper failures.

    Whatever the call cannot land goes to `spool`, which never sends it again.

    `scope` names what the rows belong to (a run id, a file); it feeds the batch token so two
    different loads of identical rows are not mistaken for one. Returns an Outcome; raises
    WriteUnavailable past the deadline when no spool is given, WriteAmbiguous when a plain-table
    batch may be stored and no spool can hold it, and the original error when it is fatal.
    The deadline covers the call's first unlandable batch: once one batch is spooled the rest go
    straight to the spool, so a dead cluster costs one deadline, not one per batch.
    """
    policy = policy or Policy()
    wait = sleep or time.sleep
    now = clock or time.monotonic
    out = Outcome()
    size = max(1, policy.batch_rows)
    batches = [list(rows[i : i + size]) for i in range(0, len(rows), size)]

    wall = spool.clock if spool is not None else time.time

    def park(n, batch, tok, ambiguous, why, first_attempt_ago=0.0):
        meta = {
            "ambiguous": ambiguous,
            "scope": scope,
            "created": wall() - first_attempt_ago,
        }
        spool.put(table, columns, batch, tok, database, **meta)
        out.spooled += len(batch)
        log(f"  [spooled] {table} batch {n}/{len(batches)} ({len(batch)} rows): {why}")

    spilled = False
    for n, batch in enumerate(batches, 1):
        tok = batch_token(table, scope, batch, n)
        token = tok if replicated else None
        if spilled:
            park(n, batch, tok, False, "earlier batch could not land")
            continue
        start = now()
        attempt = 0
        maybe_stored = False  # sticky: set by the first ambiguous error of this batch
        while True:
            elapsed = now() - start
            degraded = bool(
                policy.degrade_to
                and policy.quorum
                and elapsed >= policy.degrade_after_s
            )
            try:
                _send(
                    client,
                    table,
                    batch,
                    columns,
                    database,
                    _settings(policy, token, replicated, degraded, feeds_views),
                )
                out.stored += len(batch)
                out.degraded = out.degraded or degraded
                break
            except Exception as err:
                kind = classify(err, feeds_views)
                if kind == "fatal":
                    raise
                maybe_stored = maybe_stored or kind == "ambiguous"
                if kind == "ambiguous" and not replicated:
                    # A plain table cannot tell a stored block from a lost one.
                    out.ambiguous.append((n, safe(err)))
                    if spool is None:
                        raise WriteAmbiguous(
                            f"{table}: batch {n}/{len(batches)} may be stored and cannot be "
                            f"resent ({safe(err)})",
                            out,
                        ) from err
                    try:
                        park(n, batch, tok, True, f"ambiguous, held: {safe(err)}")
                    except Exception as full:
                        raise WriteAmbiguous(
                            f"{table}: batch {n}/{len(batches)} may be stored and could not "
                            f"be spooled ({safe(full)})",
                            out,
                        ) from err
                    spilled = True
                    break
                attempt += 1
                wait_s = min(
                    policy.backoff_s * 2 ** (attempt - 1), policy.backoff_max_s
                )
                if now() - start + wait_s > policy.deadline_s:
                    if spool is None:
                        raise WriteUnavailable(
                            f"{table}: batch {n}/{len(batches)} not accepted within "
                            f"{policy.deadline_s:.0f}s ({safe(err)})"
                        ) from err
                    park(n, batch, tok, maybe_stored, safe(err), now() - start)
                    spilled = True
                    break
                log(
                    f"  [retry] {table} batch {n}/{len(batches)} attempt {attempt}: {safe(err)}; in {wait_s:.0f}s"
                )
                wait(wait_s)
    return out


def new_scope() -> str:
    """A fresh scope for a load that has no natural id."""
    return uuid.uuid4().hex
