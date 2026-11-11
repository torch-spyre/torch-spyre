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

"""Writes that survive a degraded replicated cluster.

With insert_quorum set, an INSERT fails while fewer replicas are up than the quorum asks for,
during a pod restart, or when Keeper blips. This module keeps those writes exactly-once:

* `classify` sorts an error into retry (nothing was stored), ambiguous (the block may be
  stored) or fatal (a resend cannot help).
* On a Replicated table every batch carries a deterministic `insert_deduplication_token`, so
  resending an ambiguous batch cannot double it. A plain table has no such guard: an ambiguous
  batch there is spooled as held and resent only after a probe that identifies that batch's own
  rows (see `spool.TieredSpool.drain`); without a spool the call raises WriteAmbiguous.
* A batch refused before any write (quorum, read-only, too many parts) is never ambiguous and
  is simply sent again, on either kind of table.
* Retries back off to a total deadline. A table whose data is regenerable may then drop to a
  lower quorum; a strict one never does.
* Past the deadline the batch goes to a durable spool, replayed later with the same token.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tempfile
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
    # (quorum, read-only) is not ambiguous and can always be sent again.
    ambiguous: bool = False
    scope: str | None = None
    # Name of a shared-tier copy that may exist beside this one; the local copy is then held.
    twin: str | None = None


CLAIM = ".claimed-"
# A file whose name ends in .amb is a batch that may be stored. The state lives in the name so
# that recording it is one atomic rename of the claimed file, which fails if the claim is gone.
AMB = ".amb"


def is_item_name(name: str) -> bool:
    return name[:1].isdigit() and (
        name.endswith(".json") or name.endswith(".json" + AMB)
    )


def with_marker(name: str, on: bool) -> str:
    """`name` with the may-be-stored marker set or cleared, keeping any claim suffix."""
    base, sep, rest = name.partition(CLAIM)
    if on and not base.endswith(AMB):
        base += AMB
    elif not on and base.endswith(AMB):
        base = base[: -len(AMB)]
    return base + sep + rest


class Spool:
    """A directory of whole-file JSON batches. A file is written to a temp name and renamed, so
    a kill leaves either the finished file or none. The directory and files are owner-only: they
    hold row data."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.dir, 0o700)
        except OSError:
            pass

    def files(self) -> list[Path]:
        """Unclaimed item files, oldest first."""
        return sorted(p for p in self.dir.iterdir() if is_item_name(p.name))

    def copies(self, token: str) -> list[Path]:
        """Every file for a token, claimed or not."""
        return sorted(
            p
            for p in self.dir.iterdir()
            if p.name[:1].isdigit()
            and f"-{token}.json" in p.name
            and not p.name.endswith((".corrupt", ".quarantine", ".tmp"))
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
        twin: str | None = None,
    ) -> Path:
        created = time.time() if created is None else float(created)
        doc = {
            "table": table,
            "columns": list(columns),
            "rows": [list(r) for r in rows],
            "token": token,
            "database": database,
            "created": created,
            "ambiguous": bool(ambiguous),
            "scope": scope,
            "twin": twin,
        }
        # The token names the file: spooling the same batch twice leaves one file. Spooling it
        # again as possibly stored upgrades that file, never the reverse.
        existing = self.copies(token)
        if existing:
            if ambiguous:
                existing = [self._mark(p, True) for p in existing]
            return existing[0]
        # Strictly increasing, so files sort in the order they were written.
        ms = max(int(created * 1000), getattr(self, "_last_ms", 0) + 1)
        self._last_ms = ms
        final = self.dir / with_marker(f"{ms:013d}-{token}.json", bool(ambiguous))
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(doc, fh, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return final

    @staticmethod
    def _mark(path: Path, on: bool) -> Path:
        """Atomically set or clear the marker on one file; the original path if it is gone."""
        target = path.with_name(with_marker(path.name, on))
        if target == path:
            return path
        try:
            os.rename(path, target)
        except OSError:
            return path
        return target

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
                    bool(d.get("ambiguous")) or p.name.endswith(AMB),
                    d.get("scope"),
                    d.get("twin"),
                )
            except (ValueError, KeyError, OSError):
                try:
                    p.rename(p.with_name(p.name + ".corrupt"))
                except OSError:
                    pass

    def done(self, item: SpoolItem) -> None:
        try:
            item.path.unlink()
        except OSError:
            pass


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
    spool: Any = None,
    probe: Callable[[Any], int] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    feeds_views: bool = True,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> Outcome:
    """Insert `rows` in batches, surviving quorum and Keeper failures.

    With a TieredSpool the call first drains what earlier writes spooled (bounded by the spool's
    own item and time budget), then sends its own rows; whatever it cannot land goes to the spool.

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
    if spool is not None and hasattr(spool, "drain"):
        spool.drain(
            client,
            replicated=replicated,
            probe=probe,
            policy=policy,
            sleep=wait,
            log=log,
        )
    size = max(1, policy.batch_rows)
    batches = [list(rows[i : i + size]) for i in range(0, len(rows), size)]

    wall = getattr(spool, "now", time.time)

    def park(n, batch, tok, ambiguous, why, first_attempt_ago=0.0):
        meta = {
            "ambiguous": ambiguous,
            "scope": scope,
            "created": wall() - first_attempt_ago,
        }
        if hasattr(spool, "put_any"):
            spool.put_any(table, columns, batch, tok, database, **meta)
        else:
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


def replay(
    client,
    spool: Spool,
    *,
    replicated: bool = True,
    policy: Policy | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> int:
    """Send every spooled batch with its original token; delete each that lands. Returns count."""
    policy = policy or Policy()
    sent = 0
    for item in list(spool.pending()):
        if item.ambiguous and not replicated:
            log(f"  [replay] {item.path.name}: ambiguous on a plain table; held")
            continue
        try:
            _send(
                client,
                item.table,
                item.rows,
                item.columns,
                item.database,
                _settings(policy, item.token, replicated, False),
            )
        except Exception as err:
            if classify(err) == "fatal":
                log(f"  [replay] {item.path.name}: fatal {safe(err)}; kept")
            else:
                log(f"  [replay] {item.path.name}: {safe(err)}; kept for the next run")
            continue
        spool.done(item)
        sent += 1
    return sent


def new_scope() -> str:
    """A fresh scope for a load that has no natural id."""
    return uuid.uuid4().hex
