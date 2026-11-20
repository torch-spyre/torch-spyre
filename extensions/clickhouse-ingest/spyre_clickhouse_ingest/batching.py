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

"""Chunked, retried inserts: one HTTP request per batch instead of one per file.

A 35k-case JUnit file sent as a single request outlives the connection's write timeout on a
slow link. Batches bound each request; a failed batch is retried with backoff, but only for
transient errors (timeouts, connection loss, 5xx).

The tables are mostly plain MergeTree, which ignores insert_deduplication_token, so a blind
resend can double a batch. `insert_chunks` therefore resends only when an `already_landed`
probe shows nothing of the failed batch was kept. With no probe a batch is sent once, unless
the caller marks the table `idempotent` (a ReplacingMergeTree whose key a resend repeats).
A probe that cannot answer, or a kept batch, raises BatchAmbiguous and nothing is resent.
Residual risk: a request still committing after the settle delay, or another writer adding
rows to the probed scope (that fails safe, as a spurious BatchAmbiguous).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

MAX_ROWS = int(os.environ.get("CH_INGEST_BATCH_ROWS", "5000"))
ATTEMPTS = int(os.environ.get("CH_INGEST_BATCH_ATTEMPTS", "5"))
BACKOFF_S = float(os.environ.get("CH_INGEST_BATCH_BACKOFF_S", "2"))
SETTLE_S = float(os.environ.get("CH_INGEST_BATCH_SETTLE_S", "2"))
BACKOFF_MAX_S = 60.0

# Server errors that stay wrong on a resend even when they arrive as HTTP 5xx.
DETERMINISTIC_CODES = frozenset(
    {16, 27, 36, 43, 47, 53, 60, 62, 81, 439, 469, 497, 516}
)


# Replication and quorum failures that pass once the cluster recovers.
REPLICATION_CODES = frozenset({242, 285, 286, 319, 999})


class BatchFailed(RuntimeError):
    """A batch failed through every retry; `landed` rows of this call are already stored."""

    def __init__(self, message: str, landed: int):
        super().__init__(message)
        self.landed = landed


class BatchAmbiguous(BatchFailed):
    """A batch failed but may have been stored; resending could duplicate it."""


def _after(text: str, marker: str) -> int | None:
    head, found, tail = text.partition(marker)
    digits = "".join(ch for ch in tail[:4].lstrip() if ch.isdigit())
    return int(digits) if found and digits else None


def retriable(err: BaseException) -> bool:
    """Timeouts, connection loss and HTTP 5xx; never auth, CHECK or schema errors."""
    if isinstance(err, OSError):
        return True
    try:  # the driver is absent on images that only parse and never connect
        from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
    except ImportError:
        return False
    if isinstance(err, OperationalError):
        return True
    if isinstance(err, DatabaseError):
        # Newer drivers carry the server code on the exception and drop the status from the text.
        code = getattr(err, "code", None)
        if not isinstance(code, int):
            code = _after(str(err), "Code: ")
        if code in DETERMINISTIC_CODES:
            return False
        if code in REPLICATION_CODES:
            return True
        status = _after(str(err), "HTTP status ")
        return bool(status and status >= 500)
    return False


def chunks(rows: Sequence[T], size: int = MAX_ROWS):
    size = max(1, size)
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def insert_chunks(
    client,
    table: str,
    rows: Sequence[Sequence],
    column_names: Sequence[str],
    database: str | None = None,
    *,
    size: int = MAX_ROWS,
    attempts: int = ATTEMPTS,
    backoff_s: float = BACKOFF_S,
    settle_s: float = SETTLE_S,
    already_landed: Callable[[], int] | None = None,
    idempotent: bool = False,
    sleep: Callable[[float], None] | None = None,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> int:
    """Insert `rows` in batches of `size`; return the number stored.

    already_landed() is the table's current row count for this call's scope (e.g. the file).
    Without it, and without `idempotent`, a failed batch is not resent: the table may hold it.
    """
    total = len(rows)
    if not total:
        return 0
    if not already_landed and not idempotent:
        attempts = 1
    wait_for = sleep or time.sleep
    kw = {"database": database} if database else {}
    batches = list(chunks(rows, size))
    base = 0
    if already_landed:
        try:
            base = already_landed()
        except Exception as perr:
            raise BatchFailed(
                f"{table}: row-count probe failed before sending ({perr!r}); nothing sent",
                0,
            ) from perr
    done = 0
    for n, batch in enumerate(batches, 1):
        for attempt in range(1, attempts + 1):
            try:
                client.insert(table, batch, column_names=list(column_names), **kw)
                break
            except Exception as err:
                if already_landed:
                    ambiguous = _kept_after(
                        already_landed, base + done, settle_s, wait_for
                    )
                    if ambiguous:
                        raise BatchAmbiguous(
                            f"{table}: batch {n}/{len(batches)} failed ({err!r}); {ambiguous}",
                            done,
                        ) from err
                if attempt == attempts or not retriable(err):
                    raise BatchFailed(
                        f"{table}: batch {n}/{len(batches)} ({len(batch)} rows) failed "
                        f"after {attempt} attempt(s): {err!r}",
                        done,
                    ) from err
                wait = min(backoff_s * 2 ** (attempt - 1), BACKOFF_MAX_S)
                log(
                    f"  [retry] {table} batch {n}/{len(batches)} attempt {attempt} "
                    f"failed ({err!r}); retrying in {wait:.0f}s"
                )
                wait_for(wait)
        done += len(batch)
        if len(batches) > 1:
            log(f"  {table}: {done}/{total} rows")
    return done


def _kept_after(
    probe: Callable[[], int],
    expected: int,
    settle_s: float,
    wait_for: Callable[[float], None],
) -> str:
    """'' when the failed batch left no rows; else why a resend is unsafe.

    Probes, waits `settle_s` for a request the client gave up on to finish committing, and
    probes again. A probe that fails, or any row count other than `expected`, is unsafe.
    """
    try:
        first = probe()
        if first == expected and settle_s > 0:
            wait_for(settle_s)
            first = probe()
    except Exception as perr:
        return f"the row-count probe failed ({perr!r}), so whether it was stored is unknown"
    if first == expected:
        return ""
    return f"{first - expected:+d} rows differ from the {expected} stored before it"
