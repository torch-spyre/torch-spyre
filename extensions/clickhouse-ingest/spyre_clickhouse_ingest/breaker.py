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

"""A per-host circuit breaker around `resilience.resilient_insert`.

`resilient_insert` gives one call one deadline. A CI step makes many calls (one per table and
file), so a dead host would cost a deadline each. `guarded_insert` shares what one call learned
with the next:

* Closed -> a call that ends on a host failure opens the breaker for a cooldown -> one call probes
  with a short deadline while the rest go straight to the spool -> an INSERT that succeeds closes
  it (a read proves nothing about writes), a failure reopens it.
* A leaky budget bounds the connectivity trouble one process may spend on a host: reaching it
  parks the host's writes for a window, and the cost drains at budget/window per second.
* A table's own refusal (252, too many parts) is not the host's trouble unless a second table
  refuses the same way within the window.
* A constrained writer profile can refuse an optional per-request setting: it is dropped once
  and the batch is sent again; the dedup token and insert_quorum are never dropped.

Everything is done around the merged module, which is unchanged apart from what the spool stores.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .resilience import (
    Outcome,
    Policy,
    Spool,
    SpoolFull,
    WriteAmbiguous,
    WriteUnavailable,
    _code,
    _tokens,
    _warn_once,
    classify,
    resilient_insert,
    safe,
)

# One table's own refusal (too many parts): it neither trips the breaker nor is charged to the
# host's budget on its own. 242 (read-only table) is not here: a Keeper outage produces it on
# every Replicated table.
TABLE_SCOPED_CODES = frozenset({252})
SETTING_REFUSED = frozenset({164, 452})
VIEW_DEDUP = "deduplicate_blocks_in_dependent_materialized_views"
# The dedup token, insert_quorum and wait_for_async_insert are deliberately absent: dropping them
# could resend a batch undeduplicated, land a strict table's batch unquorumed, or acknowledge an
# asynchronous insert before it is durable.
OPTIONAL_SETTINGS = (VIEW_DEDUP, "async_insert_deduplicate")
REQUIRED_SETTINGS = (
    "insert_deduplication_token",
    "insert_quorum",
    "wait_for_async_insert",
)


@dataclass
class Limits:
    """How long a host may cost the process. All values are finite and > 0.

    After a write gives up on a host, later writes park at once for `breaker_s`; then one write
    probes with a deadline of `breaker_probe_s`. `budget_s` seconds of connectivity trouble, over
    every table, park the host for `budget_window_s`; the cost drains at budget_s/budget_window_s.
    """

    breaker_s: float = 60.0
    breaker_probe_s: float = 10.0
    budget_s: float = 900.0
    budget_window_s: float = 1800.0

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Limits.{f.name} must be finite and > 0, got {v!r}")

    @classmethod
    def from_env(cls, prefix: str = "CH_WRITE_") -> "Limits":
        """Read CH_WRITE_BREAKER_S, _BREAKER_PROBE_S, _BUDGET_S, _BUDGET_WINDOW_S. A value that is
        not a finite positive number falls back to its default with one warning."""
        out: dict[str, float] = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(prefix + f.name.upper())
            if raw is None or not raw.strip():
                continue
            try:
                v = float(raw)
            except ValueError:
                _warn_once(prefix + f.name.upper(), raw, "not a number")
                continue
            if not math.isfinite(v) or v <= 0:
                _warn_once(prefix + f.name.upper(), raw, "not finite and > 0")
                continue
            out[f.name] = v
        return cls(**out)


class Breaker:
    """State for one host, shared by every call to it in the process."""

    CLOSED, OPEN, PROBE = "closed", "open", "probe"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.until = 0.0
        self.tripped = False
        self.probing = False
        self.spent = 0.0
        self.stamp = 0.0
        self.exhausted_until = 0.0
        self.scoped: dict[str, float] = {}
        self.budget = float("inf")
        self.window = 1.0

    def _leak(self, now: float) -> None:
        if self.stamp and self.window > 0:
            drained = (now - self.stamp) * self.budget / self.window
            self.spent = max(0.0, self.spent - max(0.0, drained))
        self.stamp = now

    def charge(self, seconds: float, now: float, limits: Limits) -> None:
        """Account time the host cost the process. A success does not refill the budget, or tables
        failing on their own would take a full deadline each whenever another table succeeds."""
        with self._lock:
            self.budget, self.window = limits.budget_s, limits.budget_window_s
            self._leak(now)
            if seconds > 0:
                self.spent += seconds
            if self.spent >= self.budget:
                self.exhausted_until = now + self.window
                self.spent = 0.0

    def remaining(self, now: float) -> float:
        with self._lock:
            self._leak(now)
            return max(0.0, self.budget - self.spent)

    def acquire(self, now: float) -> str:
        with self._lock:
            if now < self.exhausted_until:
                return self.OPEN
            if not self.tripped:
                return self.CLOSED
            if now < self.until or self.probing:
                return self.OPEN
            self.probing = True
            return self.PROBE

    def blocked(self, now: float) -> bool:
        """True when `acquire` would answer OPEN (without taking the probe)."""
        with self._lock:
            if now < self.exhausted_until:
                return True
            return self.tripped and (now < self.until or self.probing)

    def trip(self, now: float, cooldown_s: float, from_probe: bool = False) -> None:
        with self._lock:
            self.tripped = True
            if from_probe:  # another writer's trip leaves a probe in flight alone
                self.probing = False
            self.until = now + cooldown_s

    def note_scoped(self, table: str, now: float, window_s: float) -> bool:
        """Record a table-scoped refusal; True once a second distinct table refused within the
        window, which is then the host's trouble."""
        with self._lock:
            self.scoped = {
                t: at for t, at in self.scoped.items() if now - at < window_s
            }
            self.scoped[table] = now
            return len(self.scoped) >= 2

    def forget_scoped(self, table: str) -> None:
        with self._lock:
            self.scoped.pop(table, None)

    def close(self) -> None:
        with self._lock:
            self.tripped = self.probing = False
            self.until = 0.0

    def release(self) -> None:
        """A probe that proved nothing lets the next caller probe."""
        with self._lock:
            self.probing = False


def _refuses_setting(err: BaseException) -> bool:
    """The server (452/164) or the driver (it reads the profile's readonly flags) refused a
    per-request setting."""
    return _code(err) in SETTING_REFUSED or (
        type(err).__name__ == "ProgrammingError" and "readonly" in str(err)
    )


class SettingRefused(RuntimeError):
    """The writer profile refuses a per-request setting that cannot be dropped (the dedup token,
    insert_quorum, wait_for_async_insert). `resilient_insert` reads it as fatal: a resend cannot change the answer."""


class Watched:
    """The client as `resilient_insert` sees it: every INSERT is timed and judged for the breaker,
    and an optional setting the profile refuses is dropped once. Other attributes pass through."""

    def __init__(self, client, breaker, limits, feeds_views, now, log):
        self._client, self._breaker, self._limits = client, breaker, limits
        self._feeds_views, self._now, self._log = feeds_views, now, log
        self.lost = 0.0  # seconds this host cost the call
        self.ok = False
        self.failed_host = False  # a host failure since the last success or answer
        self._retry_from: float | None = None
        self._refused: set[str] = set()
        self._held: dict[
            str, BaseException
        ] = {}  # token -> the error of an ambiguous attempt

    def __getattr__(self, name):
        return getattr(self._client, name)

    def insert(self, table, data, *args, **kw):
        started = self._now()
        if (
            self._retry_from is not None
        ):  # the backoff since a host failure was lost time too
            self.lost += started - self._retry_from
            self._retry_from = None
        settings = dict(kw.get("settings") or {})
        token = settings.get("insert_deduplication_token")
        while True:
            live = {k: v for k, v in settings.items() if k not in self._refused}
            if live:
                kw["settings"] = live
            else:
                kw.pop("settings", None)
            if VIEW_DEDUP in self._refused and token in self._held:
                # Without view dedup a resend of a batch that may be stored can double a view.
                raise self._held[token]
            attempt = self._now()
            try:
                result = self._client.insert(table, data, *args, **kw)
            except Exception as err:  # noqa: BLE001
                named = (
                    [
                        k
                        for k in OPTIONAL_SETTINGS + REQUIRED_SETTINGS
                        if k in str(err) and (k in live or k in REQUIRED_SETTINGS)
                    ]
                    if _refuses_setting(err)
                    else []
                )
                if any(k in REQUIRED_SETTINGS for k in named) or (
                    not named and _code(err) == 452  # a profile constraint never lifts
                ):
                    self.failed_host = False  # the host answered
                    raise SettingRefused(
                        f"{table}: the login's profile refuses a per-request setting"
                        + (f" ({', '.join(named)})" if named else "")
                    ) from err
                if named and not (VIEW_DEDUP in named and token in self._held):
                    self._refused.update(named)
                    self._log(
                        f"  [settings] WARNING {table}: profile refuses "
                        f"{', '.join(named)}; dropped"
                    )
                    continue
                self._judge(err, table, kw.get("database"), attempt, token)
                raise
            self.ok, self.failed_host = True, False
            self._breaker.forget_scoped(table_key(table, kw.get("database")))
            if self._breaker.tripped:
                self._breaker.close()  # only a successful INSERT proves the host takes writes
            return result

    def _judge(self, err, table, database, attempt, token) -> None:
        kind = classify(err, self._feeds_views)
        if kind == "fatal":
            self.failed_host = False  # the host answered
            return
        if kind == "ambiguous" and token:
            self._held.setdefault(token, err)
        host = True
        if _code(err) in TABLE_SCOPED_CODES:
            host = self._breaker.note_scoped(
                table_key(table, database), self._now(), self._limits.budget_window_s
            )
        if host:
            self.failed_host = True
            self.lost += self._now() - attempt
            self._retry_from = self._now()


def table_key(table: str, database: str | None) -> str:
    return f"{database or ''}.{table}"


class _HostSpool:
    """The spool as one call writes to it: each parked batch records the host and whether its
    send carried a dedup token (a replay needs both to judge it)."""

    def __init__(self, spool: Spool, host: str | None, token_sent: bool):
        self._spool, self._meta = spool, {"host": host, "token_sent": token_sent}
        self.clock = spool.clock

    def put(self, *args, **kw):
        return self._spool.put(*args, meta=self._meta, **kw)


def _park_all(spool, table, rows, columns, scope, database, size, meta, log) -> Outcome:
    out = Outcome()
    batches = [list(rows[i : i + size]) for i in range(0, len(rows), size)]
    for n, (batch, tok) in enumerate(zip(batches, _tokens(table, scope, batches)), 1):
        try:
            spool.put(table, columns, batch, tok, database, scope=scope, meta=meta)
        except Exception as failed:
            out.unwritten = list(range(n, len(batches) + 1))
            if isinstance(failed, SpoolFull):
                failed.outcome = out
                raise
            raise WriteAmbiguous(
                f"{table}: batch {n}/{len(batches)} could not be spooled ({safe(failed)})",
                out,
            ) from failed
        out.spooled += len(batch)
        log(
            f"  [spooled] {table} batch {n}/{len(batches)} ({len(batch)} rows): "
            "an earlier write gave up on the cluster"
        )
    return out


def guarded_insert(
    client,
    table: str,
    rows: Sequence[Sequence],
    columns: Sequence[str],
    *,
    scope: str,
    breaker: Breaker | None = None,
    host: str | None = None,
    policy: Policy | None = None,
    limits: Limits | None = None,
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
    """`resilient_insert` behind the host's breaker and budget.

    While the breaker is open every batch goes straight to the spool (WriteUnavailable without
    one). A call that ends on a host failure opens it; a success closes it. Same returns and
    raises as `resilient_insert`.
    """
    policy = policy or Policy()
    limits = limits or Limits.from_env()
    breaker = breaker or Breaker()
    now = clock or time.monotonic
    meta = {"host": host, "token_sent": bool(replicated and dedup_window != 0)}
    mode = breaker.acquire(now())
    if mode == Breaker.OPEN:
        if spool is None:
            exc = WriteUnavailable(
                f"{table}: an earlier write gave up on the cluster; not retrying"
            )
            size = max(1, policy.batch_rows)
            exc.outcome = Outcome(
                unwritten=list(range(1, max(1, -(-len(rows) // size)) + 1))
            )
            raise exc
        return _park_all(
            spool, table, rows, columns, scope, database, max(1, policy.batch_rows),
            meta, log,
        )  # fmt: skip
    watched = Watched(client, breaker, limits, feeds_views, now, log)
    try:
        deadline = policy.deadline_s
        if mode == Breaker.PROBE:
            deadline = min(deadline, limits.breaker_probe_s)
        breaker.charge(0.0, now(), limits)  # publishes the budget
        deadline = min(deadline, breaker.remaining(now()))
        call_policy = dataclasses.replace(policy, deadline_s=deadline)
        return resilient_insert(
            watched,
            table,
            rows,
            columns,
            scope=scope,
            policy=call_policy,
            replicated=replicated,
            dedup_window=dedup_window,
            database=database,
            spool=_HostSpool(spool, host, meta["token_sent"]) if spool else None,  # type: ignore[arg-type]
            sleep=sleep,
            clock=clock,
            rng=rng,
            feeds_views=feeds_views,
            log=log,
        )
    finally:
        settle(watched, breaker, limits, mode, now)


def settle(watched: Watched, breaker: Breaker, limits: Limits, mode: str, now) -> None:
    """Charge what the call cost the host, trip the breaker if it ended on a host failure, and
    hand back a probe slot that was taken."""
    breaker.charge(watched.lost, now(), limits)
    if watched.failed_host:
        breaker.trip(now(), limits.breaker_s, from_probe=mode == Breaker.PROBE)
    if mode == Breaker.PROBE:
        breaker.release()
