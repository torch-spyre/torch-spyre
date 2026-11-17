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

"""Fill past GitHub Actions runs newest slice first, beside the forward `poll`.

    python -m spyre_clickhouse_ingest.gha_runs backfill --repo torch-spyre/torch-spyre \\
        --to 2026-10-20T00:00Z --slice-hours 1 --workers 4 --deadline-minutes 50

The poll's watermark is the newest stored run, so filling newest first moves it to the present
and the poll then keeps the present current. What lies between the old watermark and the
present is no longer visible in pipeline_runs, so no cursor is derived from it. Each slice of
creation time is instead checked on its own, against GitHub: it is complete when every listed
run attempt is stored, finished and current. A complete slice costs the listing only, an
incomplete one is filled, and any slice can be repeated or interrupted: rows replace by
(run, attempt). Running the same command again, with the same --to, is the whole resume story.

Slices are walked newest first by --workers threads sharing one rate budget, so a stop leaves
the newest slices done. The report names the lower edge of the contiguous complete prefix
(`complete down to`); pass it as --from to skip the part already verified.
"""

import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .gha_runs import (
    EXIT_BUDGET,
    EXIT_MORE,
    GitHub,
    Stop,
    attempt_rows,
    connect,
    stored,
)
from .schema import PipelineRuns

INSERT_ROWS = 500


@dataclass
class Slice:
    lo: datetime
    hi: datetime
    runs: int = 0
    fetched: int = 0
    rows: int = 0
    seconds: float = 0.0
    done: bool = False
    skipped: list[str] = field(default_factory=list)
    reason: str = ""

    def label(self) -> str:
        return f"{self.lo:%Y-%m-%dT%H:%MZ}..{self.hi:%Y-%m-%dT%H:%MZ}"


def slices(top: datetime, to: datetime, size: timedelta) -> Iterator[Slice]:
    """[lo, hi) windows of creation time from top (exclusive) down to to, edges on multiples of size."""
    step = int(size.total_seconds())
    hi = top.replace(microsecond=0)
    while hi > to:
        lo = max(
            datetime.fromtimestamp(((hi.timestamp() - 1) // step) * step, timezone.utc),
            to,
        )
        yield Slice(lo, hi)
        hi = lo


def fill(
    gh: GitHub,
    client,
    db: str | None,
    repo: str,
    sl: Slice,
    write: threading.Lock,
    dry_run: bool,
) -> None:
    """Store the missing or unfinished attempts of the runs created in sl; set sl.done."""
    began = time.time()
    known = (
        stored(client, db or "", repo, sl.lo, sl.hi + timedelta(days=1))
        if client is not None
        else {}
    )
    batch: list[dict[str, Any]] = []
    clean = True
    try:
        listed = list(gh.runs(repo, sl.lo, sl.hi - timedelta(seconds=1)))
        sl.runs = len(listed)
        for latest in listed:
            for a in range(1, int(latest.get("run_attempt") or 1) + 1):
                got = attempt_rows(gh, repo, latest, a, known)
                if got is None:
                    clean = False
                    sl.skipped.append(f"{latest['id']}#{a}")
                elif got:
                    sl.fetched += 1
                    batch += got
                    # The workflow row leads; a run still going is stored but not done.
                    clean &= got[0]["state"] == "finished"
    except Stop as stop:
        sl.reason, clean = str(stop), False
    if client is not None and not dry_run:
        with write:
            for i in range(0, len(batch), INSERT_ROWS):
                PipelineRuns.insert(client, batch[i : i + INSERT_ROWS], db=db)
    sl.rows, sl.done = len(batch), clean
    sl.seconds = time.time() - began


def backfill(
    gh: GitHub,
    client,
    db: str | None,
    repo: str,
    top: datetime,
    to: datetime,
    size: timedelta,
    workers: int = 4,
    dry_run: bool = False,
) -> list[Slice]:
    """Fill every slice from top down to to; the slices attempted, newest first."""
    todo = list(slices(top, to, size))
    lock, write, halt = threading.Lock(), threading.Lock(), threading.Event()
    nxt, ran = [0], {}

    def worker() -> None:
        while not halt.is_set():
            with lock:
                if nxt[0] >= len(todo):
                    return
                i = nxt[0]
                nxt[0] += 1
            fill(gh, client, db, repo, todo[i], write, dry_run)
            with lock:
                ran[i] = todo[i]
            if todo[i].reason:
                halt.set()

    threads = [threading.Thread(target=worker) for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [ran[i] for i in sorted(ran)]


def complete_down_to(done: list[Slice], top: datetime) -> datetime:
    """Lower edge of the newest-first run of complete slices, or top when none is."""
    edge = top
    for s in sorted(done, key=lambda s: s.hi, reverse=True):
        if not s.done or s.hi < edge - timedelta(seconds=1):
            break
        edge = s.lo
    return edge


def run(args) -> None:
    import os

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("[error] GITHUB_TOKEN is unset")
    now = datetime.now(timezone.utc)
    top = args.top or now
    to = args.to or top - timedelta(days=args.max_catchup_days)
    if to >= top:
        sys.exit("[error] --to must be older than --from")
    client, db = (None, None) if args.dry_run else connect(args.database)
    gh = GitHub(
        token,
        reserve=args.reserve,
        deadline=time.time() + args.deadline_minutes * 60
        if args.deadline_minutes
        else None,
        wait_for_reset=args.wait_for_reset,
        min_interval=1 / args.max_rps if args.max_rps else 0.0,
    )
    began = time.time()
    done = backfill(
        gh,
        client,
        db,
        args.repo,
        top,
        to,
        timedelta(hours=args.slice_hours),
        args.workers,
        args.dry_run,
    )
    for s in done:
        print(
            f"[slice] {s.label()} runs={s.runs} fetched={s.fetched} rows={s.rows} "
            f"{'done' if s.done else 'open'} {s.seconds:.1f}s"
            + (f" ({s.reason})" if s.reason else ""),
            file=sys.stderr,
        )
    edge = complete_down_to(done, top)
    stopped = next((s.reason for s in done if s.reason), "")
    print(
        f"[info] {args.repo}: {len(done)} slice(s) in {time.time() - began:.0f}s, "
        f"{sum(s.fetched for s in done)} attempt(s) fetched, {gh.requests} request(s), "
        f"rate limit {gh.first or 'unseen'} -> {gh.rate()}",
        file=sys.stderr,
    )
    print(f"[info] complete down to {edge:%Y-%m-%dT%H:%M:%SZ}", file=sys.stderr)
    open_ = [s.label() for s in done if not s.done and not s.reason]
    if open_:
        print(
            f"[warn] {len(open_)} slice(s) not complete (runs still going, or GitHub "
            f"would not serve them): {', '.join(open_[:8])}",
            file=sys.stderr,
        )
    if stopped:
        print(f"[notice] {stopped}; run again with the same --to", file=sys.stderr)
        sys.exit(EXIT_BUDGET if gh.exhausted() and not gh.wait_for_reset else EXIT_MORE)
