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

import datetime
import decimal
import enum
import hashlib
import ipaddress
import json
import math
import os
import random
import socket
import stat
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import regex as re

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]

# Refused before the write on the table itself (quorum, read-only, parts limit, 289 from a
# sequential-consistency read). A materialized view's target can raise the same codes after the
# source block landed, so with a dependent view they prove nothing; ClickHouse's order between the
# source commit and the view pushes is not something the docs or chdb let us rely on.
PRE_WRITE_CODES = frozenset({242, 252, 285, 286, 289})
# Refused at query admission (ProcessList), before anything runs, whatever feeds from the table.
ADMISSION_CODES = frozenset({202})
# The block may be stored on some replicas or in some views: coordination, timeouts, network,
# thread-pool exhaustion mid-pipeline, async-flush timeout, and a memory limit hit while a
# materialized view was being fed.
AMBIGUOUS_CODES = frozenset(
    {159, 203, 209, 210, 225, 241, 244, 319, 415, 439, 745, 999, 1000, 1017}
)
# A resend cannot change the answer: schema, permission, syntax, parse and constraint errors.
FATAL_CODES = frozenset(
    {6, 15, 16, 26, 27, 36, 38, 41, 43, 44, 46, 47, 53, 60, 62, 70, 81, 117}
    | {192, 193, 194, 195, 469, 497, 516}
)
READONLY = (
    164  # a user's readonly mode is permanent; "Replica is not active" is transient
)
_CH_TEXT_CODE = re.compile(r"(?<![A-Za-z0-9])Code: (\d{1,5})\. DB::Exception:")
_HTTP_STATUS = re.compile(r"HTTP status (\d{3})\b")
_TRANSIENT_STATUS = frozenset({408, 429}) | frozenset(range(500, 600))


class WriteFailed(RuntimeError):
    """Base of the errors that say rows did not land; `outcome` says what did."""

    outcome: "Outcome | None" = None


class WriteUnavailable(WriteFailed):
    """The cluster did not accept the write before the deadline and no spool was given."""


class SpoolFull(WriteFailed):
    """The spool is at its size cap; the write fails loudly instead of dropping a batch."""


class WriteAmbiguous(WriteFailed):
    """A batch may or may not be stored and there is no spool to hold it (or it could not)."""

    def __init__(self, message: str, outcome: "Outcome"):
        super().__init__(message)
        self.outcome = outcome


def _code(err: BaseException) -> int | None:
    """The server's error code: the driver's structured attribute, else ClickHouse's own
    `Code: N. DB::Exception:` text. A bare 'Code: N' elsewhere in a message is not trusted."""
    code = getattr(err, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    found = _CH_TEXT_CODE.search(str(err))
    return int(found.group(1)) if found else None


def _chain(err: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    e: BaseException | None = err
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def _never_connected(err: BaseException) -> bool:
    """The connection was refused, timed out connecting, or the host did not resolve: no request
    left the client.

    Walks the exception chain by type; the message text is not trusted (it can be hidden by the
    driver and can carry row data).
    """
    try:
        from urllib3.exceptions import (
            ConnectTimeoutError,
            MaxRetryError,
            NewConnectionError,
        )
    except ImportError:  # pragma: no cover
        return any(
            isinstance(e, (ConnectionRefusedError, socket.gaierror))
            for e in _chain(err)
        )
    never = (
        ConnectionRefusedError,
        socket.gaierror,
        NewConnectionError,
        ConnectTimeoutError,
    )
    for e in _chain(err):
        if isinstance(e, never):
            return True
        if isinstance(e, MaxRetryError) and isinstance(
            getattr(e, "reason", None), never
        ):
            return True
    return False


def _transport_error(err: BaseException) -> bool:
    """A urllib3 error or OS-level error somewhere in the chain: the request may have been sent."""
    try:
        from urllib3.exceptions import HTTPError
    except ImportError:  # pragma: no cover
        return False
    return any(isinstance(e, (HTTPError, OSError)) for e in _chain(err))


def _http_gateway_error(err: BaseException) -> bool:
    """A non-2xx reply that is not a ClickHouse error: a balancer or proxy answered (408, 429,
    5xx), or the server died before it could name an error.

    clickhouse_connect raises exactly `DatabaseError` (or `OperationalError` after its own retries
    of 429/503/504) for an HTTP error, with `code` set from the X-ClickHouse-Exception-Code header.
    No code means no ClickHouse exception: the HTTP status is not exposed as an attribute, so this
    holds with show_clickhouse_errors off. When the text is shown its status decides (a 401, 403
    or 404 is not transient). Client-side subclasses (ProgrammingError, DataError) are not matched.
    """
    try:
        from clickhouse_connect.driver.exceptions import DatabaseError
    except ImportError:  # pragma: no cover
        return False
    for e in _chain(err):
        if getattr(e, "code", None) is not None:
            return False
        if not isinstance(e, DatabaseError):
            continue
        shown = _HTTP_STATUS.search(str(e))
        if shown:
            return int(shown.group(1)) in _TRANSIENT_STATUS
        if type(e) is DatabaseError:
            return True  # status hidden: a balancer or proxy answered
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
    feeds_views=False. A server code this module has never seen is 'ambiguous': it may be a
    transient fault, and `resilient_insert` resends an ambiguous batch only where a token makes
    that safe, else parks it.
    """
    code = _code(err)
    if code == READONLY:
        if "readonly mode" in str(err):
            return "fatal"
        return "ambiguous" if feeds_views else "retry"
    if code in FATAL_CODES:
        return "fatal"
    if code in ADMISSION_CODES:
        return "retry"
    if code in PRE_WRITE_CODES:
        return "ambiguous" if feeds_views else "retry"
    if code in AMBIGUOUS_CODES:
        return "ambiguous"
    if code is not None:
        return "ambiguous"
    if _never_connected(err):
        return "retry"
    if _transport_error(err):
        return "ambiguous"
    try:
        from clickhouse_connect.driver.exceptions import OperationalError
    except ImportError:
        return "fatal"
    if isinstance(err, OperationalError) or _http_gateway_error(err):
        return "ambiguous"
    return "fatal"


def _canon(v: Any) -> Any:
    """A JSON-able form of one cell that does not depend on process state (hash seed, repr)."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return {"f": repr(v)}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return {"set": sorted(_dumps(_canon(x)) for x in v)}
    if isinstance(v, dict):
        return {"map": sorted([_dumps(_canon(k)), _canon(x)] for k, x in v.items())}
    if isinstance(v, datetime.datetime):
        return {"dt": v.isoformat()}
    if isinstance(v, (datetime.date, datetime.time)):
        return {"d": v.isoformat()}
    if isinstance(v, decimal.Decimal):
        return {"dec": str(v)}
    if isinstance(v, uuid.UUID):
        return {"uuid": str(v)}
    if isinstance(v, (bytes, bytearray)):
        return {"b": bytes(v).hex()}
    if isinstance(v, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return {"ip": str(v)}
    if isinstance(v, enum.Enum):
        return _canon(v.value)
    item = getattr(v, "item", None)  # numpy scalars
    if callable(item) and type(v).__module__ == "numpy":
        return _canon(item())
    raise TypeError(
        f"cannot derive a stable batch token from a {type(v).__name__} cell"
    )


def _dumps(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def batch_token(
    table: str, scope: str, rows: Sequence[Sequence], index: int = 1
) -> str:
    """A stable token for one batch, derived only from the table, the caller's `scope` and the
    content of the rows in order. The same rows under the same scope always give the same token,
    in this process or another, whatever the batch's position in a call.

    Two batches that are meant to be separate but carry identical rows must use different scopes;
    inside one `resilient_insert` call the 2nd, 3rd... identical batch is told apart by `index`
    (its occurrence, 1 for the first). A cell of a type with no stable encoding raises TypeError.
    """
    h = hashlib.sha256()
    h.update(f"v2\x00{table}\x00{scope}\x00".encode())
    if index > 1:
        h.update(f"occurrence={index}\x00".encode())
    h.update(_dumps([_canon(list(r)) for r in rows]).encode())
    return f"{table}-{h.hexdigest()[:40]}"


def _tokens(table: str, scope: str, batches: Sequence[Sequence]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for b in batches:
        base = batch_token(table, scope, b)
        seen[base] = seen.get(base, 0) + 1
        out.append(
            base if seen[base] == 1 else batch_token(table, scope, b, seen[base])
        )
    return out


_warned: set[tuple[str, str]] = set()


def _warn_once(name: str, raw: str, why: str) -> None:
    if (name, raw) not in _warned:
        _warned.add((name, raw))
        print(
            f"[warn] {name}={raw!r} ignored ({why}); using the default", file=sys.stderr
        )


@dataclass
class Policy:
    """Per-writer settings. `quorum` None sends no insert_quorum; `degrade_to` None is strict.

    `deadline_s` is one budget for the whole call, counted from its first failed attempt.
    Retries wait a random time up to `backoff_s` doubling to `backoff_max_s` (full jitter).
    """

    batch_rows: int = 5000
    deadline_s: float = 300.0
    backoff_s: float = 2.0
    backoff_max_s: float = 30.0
    quorum: str | None = None
    degrade_to: str | None = None
    degrade_after_s: float = 60.0

    def __post_init__(self):
        for name in ("deadline_s", "backoff_s", "backoff_max_s", "degrade_after_s"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"Policy.{name} must be finite and >= 0, got {v!r}")
        if self.backoff_s <= 0:
            raise ValueError("Policy.backoff_s must be > 0")
        if self.backoff_max_s < self.backoff_s:
            raise ValueError("Policy.backoff_max_s must be >= backoff_s")
        if not isinstance(self.batch_rows, int) or self.batch_rows < 1:
            raise ValueError(
                f"Policy.batch_rows must be an int >= 1, got {self.batch_rows!r}"
            )

    @classmethod
    def from_env(cls, prefix: str = "CH_WRITE_") -> "Policy":
        """Read the CH_WRITE_* variables. A value that is not a finite number in range falls back
        to its default with one warning; it never reaches the retry loop."""
        defaults = cls()

        def num(name: str, default: float, lo: float, hi: float, lo_open=False):
            raw = os.environ.get(prefix + name)
            if raw is None or not raw.strip():
                return default
            try:
                v = float(raw)
            except ValueError:
                _warn_once(prefix + name, raw, "not a number")
                return default
            if not math.isfinite(v) or v > hi or v < lo or (lo_open and v == lo):
                _warn_once(
                    prefix + name, raw, f"outside {'(' if lo_open else '['}{lo}, {hi}]"
                )
                return default
            return v

        rows = num("BATCH_ROWS", defaults.batch_rows, 1, 1_000_000)
        backoff = num("BACKOFF_S", defaults.backoff_s, 0, 3600, lo_open=True)
        backoff_max = num(
            "BACKOFF_MAX_S", defaults.backoff_max_s, 0, 86400, lo_open=True
        )
        if backoff_max < backoff:
            _warn_once(prefix + "BACKOFF_MAX_S", str(backoff_max), "below BACKOFF_S")
            backoff_max = max(backoff, defaults.backoff_max_s)
        return cls(
            batch_rows=int(rows),
            deadline_s=num("DEADLINE_S", defaults.deadline_s, 0, 86400, lo_open=True),
            backoff_s=backoff,
            backoff_max_s=backoff_max,
            quorum=os.environ.get(prefix + "QUORUM") or None,
            degrade_to=os.environ.get(prefix + "DEGRADE_TO") or None,
            degrade_after_s=num("DEGRADE_AFTER_S", defaults.degrade_after_s, 0, 86400),
        )


_WINDOW = re.compile(r"replicated_deduplication_window\s*=\s*(\d+)")


def dedup_window(
    client, table: str, database: str | None = None, cache=None
) -> int | None:
    """The table's `replicated_deduplication_window` (0: ClickHouse deduplicates nothing), read
    once per (database, table) into `cache`. 0 for a non-Replicated engine. None when the table
    or the setting cannot be read: callers then assume a nonzero window.

    A table's own SETTINGS (engine_full) win over the server's MergeTree default. The setting
    counts blocks; replicated_deduplication_window_seconds also bounds it and is not read here.
    """
    key = (database or "", table)
    if cache is not None and key in cache:
        return cache[key]
    window: int | None = None
    try:
        res = client.query(
            "SELECT engine, engine_full FROM system.tables "
            "WHERE database = {db:String} AND name = {t:String}",
            parameters={"db": database or _current_database(client), "t": table},
        )
        rows = res.result_rows
        if rows:
            engine, full = rows[0][0], rows[0][1] or ""
            if not str(engine).startswith("Replicated"):
                window = 0
            else:
                found = _WINDOW.search(full)
                if found:
                    window = int(found.group(1))
                else:
                    d = client.query(
                        "SELECT value FROM system.merge_tree_settings "
                        "WHERE name = 'replicated_deduplication_window'"
                    ).result_rows
                    window = int(d[0][0]) if d else None
    except Exception:  # noqa: BLE001 - unreadable means unknown, not zero
        window = None
    if cache is not None and window is not None:
        cache[key] = window
    return window


def _current_database(client) -> str:
    return getattr(client, "database", None) or "default"


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


_STALE_TMP_S = 3600.0


class Spool:
    """A directory of whole-file JSON batches that could not be sent. It holds row data, so the
    directory is 0700 and the files 0600. Nothing here sends a batch again.

    A batch a failed attempt may have stored carries `"ambiguous": true` in its JSON. Parking the
    same token again keeps one file and can only set that flag, never clear it. A full spool
    raises SpoolFull instead of dropping anything; the cap counts every file in the directory,
    including quarantined `.corrupt` files and temp files. `put` and `drop` take a lock file, so
    processes sharing the directory do not race on one token (flock: POSIX, local filesystems).

    The directory must be a real directory owned by the caller: a symlink or another user's
    directory is refused (PermissionError / NotADirectoryError, both OSError).
    """

    def __init__(
        self, directory, *, max_bytes: int = 256 * 1024 * 1024, clock=time.time
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = os.lstat(self.dir)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise NotADirectoryError(f"spool {self.dir} is not a plain directory")
        owner = getattr(os, "geteuid", lambda: st.st_uid)()
        if st.st_uid != owner:
            raise PermissionError(f"spool {self.dir} is owned by another user")
        os.chmod(self.dir, 0o700)
        self.max_bytes, self.clock = max_bytes, clock
        self.unreadable: list[str] = []
        self._clean_stale_tmp()

    @contextmanager
    def _locked(self):
        if fcntl is None:  # pragma: no cover
            yield
            return
        fd = os.open(self.dir / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing releases the lock

    def _clean_stale_tmp(self) -> None:
        """Remove temp files a killed writer left behind. A young one may be a live writer's."""
        for p in self.tmp_files():
            try:
                if time.time() - p.stat().st_mtime > _STALE_TMP_S:
                    p.unlink()
            except OSError:
                pass

    def tmp_files(self) -> list[Path]:
        return sorted(
            p
            for p in self.dir.iterdir()
            if p.name.startswith(".") and p.name.endswith(".tmp")
        )

    def corrupt_files(self) -> list[Path]:
        """Files quarantined because they would not parse as a batch."""
        return sorted(p for p in self.dir.iterdir() if p.name.endswith(".corrupt"))

    def problems(self) -> dict[str, list[str]]:
        """What `pending` could not return: quarantined files, files that could not be read
        (checked by the last `pending`), and leftover temp files. All count toward the cap."""
        return {
            "corrupt": [p.name for p in self.corrupt_files()],
            "unreadable": list(self.unreadable),
            "tmp": [p.name for p in self.tmp_files()],
        }

    def files(self, token: str | None = None) -> list[Path]:
        """Item files, oldest first; only those of one batch when `token` is given."""
        return sorted(
            p
            for p in self.dir.iterdir()
            if p.name[:1].isdigit()
            and p.name.endswith(".json")
            and (token is None or p.name.endswith(f"-{token}.json"))
        )

    def _used(self) -> int:
        total = 0
        for p in self.dir.iterdir():
            if p.name != ".lock":
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

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
        with self._locked():
            for existing in self.files(token):
                try:
                    old = json.loads(existing.read_text())
                    if not isinstance(old, dict):
                        raise ValueError("not an object")
                except (ValueError, TypeError):
                    existing.rename(existing.with_name(existing.name + ".corrupt"))
                    continue
                if ambiguous and not old.get("ambiguous"):
                    old["ambiguous"] = True
                    _write_private(existing, old)
                return existing
            used = self._used()
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
        """Parked batches, oldest first. A file that does not parse is quarantined as `.corrupt`;
        one that cannot be read right now (an I/O error) is left in place and listed in
        `unreadable`."""
        self.unreadable = []
        for p in self.files():
            try:
                text = p.read_text()
            except OSError:
                self.unreadable.append(p.name)
                continue
            try:
                d = json.loads(text)
                item = SpoolItem(
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
            except (ValueError, KeyError, TypeError, AttributeError):
                try:
                    p.rename(p.with_name(p.name + ".corrupt"))
                except OSError:
                    pass
                continue
            yield item

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
        with self._locked():
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
    """What a call did. `ambiguous` lists (batch number, error) of batches that may be stored;
    `unwritten` lists batch numbers that were neither stored nor parked, set when the call
    raises."""

    stored: int = 0
    spooled: int = 0
    ambiguous: list = field(default_factory=list)
    degraded: bool = False
    unwritten: list = field(default_factory=list)


def _settings(
    policy: Policy,
    token: str | None,
    replicated: bool,
    degraded: bool,
    feeds_views: bool = True,
    dedup: bool = True,
) -> dict:
    s: dict[str, Any] = {}
    if replicated:
        if dedup:
            if token:
                s["insert_deduplication_token"] = token
            if feeds_views:
                # Without it a retry whose source block was deduplicated may skip a view target
                # that missed the block the first time (older servers default to 0). The target
                # needs its own dedup window (Replicated, or non_replicated_deduplication_window).
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
    dedup_window: int | None = None,
    database: str | None = None,
    spool: Spool | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    rng: Any = None,
    feeds_views: bool = True,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> Outcome:
    """Insert `rows` in batches, surviving quorum and Keeper failures.

    Whatever the call cannot land goes to `spool`, which never sends it again.

    `scope` names what the rows belong to (a run id, a file); with the table and the rows it makes
    the batch token, so two different loads of identical rows need different scopes or the second
    is deduplicated away. `dedup_window` is the table's replicated_deduplication_window (see
    `dedup_window()`): 0 means the table deduplicates nothing, so it is handled like a plain
    table (no token; an ambiguous batch is parked, never resent); None assumes a nonzero window.

    Returns an Outcome. Raises WriteUnavailable past the deadline when no spool is given,
    WriteAmbiguous when a batch may be stored and no spool can hold it, SpoolFull when the spool is
    at its cap, and the original error when it is fatal; all carry `outcome` (the original error
    gets `outcome` and `maybe_stored` attributes). The deadline is one budget per call, counted
    from its first failed attempt: once one batch is spooled the rest go straight to the spool, so
    a dead cluster costs one deadline, not one per batch.
    """
    policy = policy or Policy()
    wait = sleep or time.sleep
    now = clock or time.monotonic
    uniform = (rng or random).uniform
    out = Outcome()
    size = max(1, policy.batch_rows)
    batches = [list(rows[i : i + size]) for i in range(0, len(rows), size)]
    tokens = _tokens(table, scope, batches)
    dedup = replicated and dedup_window != 0

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

    def park_or_raise(n, batch, tok, ambiguous, why, first_attempt_ago=0.0):
        try:
            park(n, batch, tok, ambiguous, why, first_attempt_ago)
        except Exception as failed:
            out.unwritten = list(range(n, len(batches) + 1))
            if isinstance(failed, SpoolFull):
                failed.outcome = out
                raise
            raise WriteAmbiguous(
                f"{table}: batch {n}/{len(batches)} "
                f"{'may be stored and ' if ambiguous else ''}could not be spooled "
                f"({safe(failed)})",
                out,
            ) from failed

    spilled = False
    trouble_start: float | None = None
    for n, batch in enumerate(batches, 1):
        tok = tokens[n - 1]
        token = tok if dedup else None
        if spilled:
            park_or_raise(n, batch, tok, False, "earlier batch could not land")
            continue
        start = now()
        attempt = 0
        maybe_stored = False  # sticky: set by the first ambiguous error of this batch
        while True:
            tried_at = now()
            degraded = bool(
                policy.degrade_to
                and policy.quorum
                and tried_at - start >= policy.degrade_after_s
            )
            try:
                _send(
                    client,
                    table,
                    batch,
                    columns,
                    database,
                    _settings(policy, token, replicated, degraded, feeds_views, dedup),
                )
                out.stored += len(batch)
                out.degraded = out.degraded or degraded
                break
            except Exception as err:
                kind = classify(err, feeds_views)
                if kind == "fatal":
                    if maybe_stored:
                        out.ambiguous.append(
                            (n, "fatal error after an ambiguous attempt")
                        )
                    out.unwritten = list(range(n, len(batches) + 1))
                    try:
                        err.maybe_stored = maybe_stored  # type: ignore[attr-defined]
                        err.outcome = out  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001 - an exception type without a __dict__
                        pass
                    raise
                if trouble_start is None:
                    trouble_start = tried_at
                maybe_stored = maybe_stored or kind == "ambiguous"
                if kind == "ambiguous" and not dedup:
                    # Without a working token a stored block cannot be told from a lost one.
                    out.ambiguous.append((n, safe(err)))
                    if spool is None:
                        out.unwritten = list(range(n, len(batches) + 1))
                        raise WriteAmbiguous(
                            f"{table}: batch {n}/{len(batches)} may be stored and cannot be "
                            f"resent ({safe(err)})",
                            out,
                        ) from err
                    try:
                        park(n, batch, tok, True, f"ambiguous, held: {safe(err)}")
                    except Exception as full:
                        out.unwritten = list(range(n, len(batches) + 1))
                        raise WriteAmbiguous(
                            f"{table}: batch {n}/{len(batches)} may be stored and could not "
                            f"be spooled ({safe(full)})",
                            out,
                        ) from err
                    spilled = True
                    break
                attempt += 1
                ceiling = min(
                    policy.backoff_s * 2 ** min(attempt - 1, 32), policy.backoff_max_s
                )
                wait_s = max(uniform(0.0, ceiling), min(ceiling, 0.001))
                if now() - trouble_start + wait_s > policy.deadline_s:
                    if spool is None:
                        out.unwritten = list(range(n, len(batches) + 1))
                        if maybe_stored:
                            out.ambiguous.append((n, safe(err)))
                        exc = WriteUnavailable(
                            f"{table}: batch {n}/{len(batches)} not accepted within "
                            f"{policy.deadline_s:.0f}s ({safe(err)})"
                        )
                        exc.outcome = out
                        raise exc from err
                    park_or_raise(n, batch, tok, maybe_stored, safe(err), now() - start)
                    spilled = True
                    break
                log(
                    f"  [retry] {table} batch {n}/{len(batches)} attempt {attempt}: {safe(err)}; in {wait_s:.1f}s"
                )
                wait(wait_s)
    return out


def new_scope() -> str:
    """A fresh scope for a load that has no natural id."""
    return uuid.uuid4().hex
