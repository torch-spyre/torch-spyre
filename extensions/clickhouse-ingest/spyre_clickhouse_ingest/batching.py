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
slow link. Batches bound each request; a failed batch is retried with backoff. The target
tables are plain MergeTree, which ignores insert_deduplication_token, so a retry after an
ambiguous failure could double a batch: `insert_chunks` takes an `already_landed` probe and
raises BatchAmbiguous instead of resending when the server may have kept the failed batch.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Sequence

MAX_ROWS = int(os.environ.get("CH_INGEST_BATCH_ROWS", "5000"))
ATTEMPTS = int(os.environ.get("CH_INGEST_BATCH_ATTEMPTS", "5"))
BACKOFF_S = float(os.environ.get("CH_INGEST_BATCH_BACKOFF_S", "2"))
BACKOFF_MAX_S = 60.0


class BatchFailed(RuntimeError):
    """A batch failed through every retry; `landed` rows of this call are already stored."""

    def __init__(self, message: str, landed: int):
        super().__init__(message)
        self.landed = landed


class BatchAmbiguous(BatchFailed):
    """A batch failed but may have been stored; resending could duplicate it."""


def chunks(rows: Sequence[Any], size: int = MAX_ROWS):
    size = max(1, size)
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def insert_chunks(
    client,
    table: str,
    rows: Sequence[Any],
    column_names: Sequence[str],
    database: str | None = None,
    *,
    size: int = MAX_ROWS,
    attempts: int = ATTEMPTS,
    backoff_s: float = BACKOFF_S,
    already_landed: Callable[[], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> int:
    """Insert `rows` in batches of `size`; return the number stored.

    already_landed() is the table's current row count for this call's scope (e.g. the file).
    With it, a batch that errors is only resent when the count shows nothing of it was kept.
    """
    total = len(rows)
    if not total:
        return 0
    base = already_landed() if already_landed else 0
    done = 0
    kw = {"database": database} if database else {}
    batches = list(chunks(rows, size))
    for n, batch in enumerate(batches, 1):
        for attempt in range(1, attempts + 1):
            try:
                client.insert(table, batch, column_names=list(column_names), **kw)
                break
            except (
                Exception
            ) as err:  # transport and server errors alike; classified below
                if already_landed:
                    kept = already_landed() - base - done
                    if kept:
                        raise BatchAmbiguous(
                            f"{table}: batch {n}/{len(batches)} failed ({err!r}) after "
                            f"{kept} of its {len(batch)} rows were stored",
                            done + kept,
                        ) from err
                if attempt == attempts:
                    raise BatchFailed(
                        f"{table}: batch {n}/{len(batches)} ({len(batch)} rows) failed "
                        f"after {attempts} attempts: {err!r}",
                        done,
                    ) from err
                wait = min(backoff_s * 2 ** (attempt - 1), BACKOFF_MAX_S)
                log(
                    f"  [retry] {table} batch {n}/{len(batches)} attempt {attempt} "
                    f"failed ({err!r}); retrying in {wait:.0f}s"
                )
                sleep(wait)
        done += len(batch)
        if len(batches) > 1:
            log(f"  {table}: {done}/{total} rows")
    return done
