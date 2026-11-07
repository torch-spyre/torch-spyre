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

"""A two-tier spool that drains itself.

Batches a write could not land go to a local directory, or to a shared store that outlives the
runner. Every write, and process start, first replays what is pending (bounded, so it never holds
up the caller's own write) and only then sends its own rows.

* A claim is an atomic rename. Two writers racing on one item cannot both win. A claim carries a
  lease; a writer that died leaves a claim that expires and is replayed.
* Replicated tables: the original `insert_deduplication_token` makes a replay exactly-once while
  the server still holds the token. An item older than the dedup window, or one a failed attempt
  may have stored, is probed first, as a plain table always is.
* The shared tier holds row data. It is used only by trusted runs (see `shared_allowed`) and any
  failure there falls back to the local directory.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from . import resilience as rz

CLAIM = ".claimed-"
DEDUP_WINDOW_S = 3600.0


class SpoolFull(RuntimeError):
    """The spool is at its size cap; the write must fail loudly instead of dropping data."""


# ── trust ─────────────────────────────────────────────────────────────────────


def shared_allowed(env=None) -> bool:
    """True only for a run that may put row data in the shared store.

    Default closed. An explicit CH_WRITE_SHARED_SPOOL_TRUSTED=1 is required and is still refused
    for pull-request runs and for forks, because those run code the team did not review with
    secrets in reach.
    """
    env = os.environ if env is None else env
    if (env.get("CH_WRITE_SHARED_SPOOL_TRUSTED") or "").strip() not in (
        "1",
        "true",
        "yes",
    ):
        return False
    event = (env.get("GITHUB_EVENT_NAME") or "").strip()
    if event in ("pull_request", "pull_request_target", "fork"):
        return False
    path = env.get("GITHUB_EVENT_PATH")
    if path:
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return False
        pr = doc.get("pull_request") or {}
        if (pr.get("head") or {}).get("repo", {}).get("fork"):
            return False
        if (doc.get("repository") or {}).get("fork"):
            return False
    return True


# ── tiers ─────────────────────────────────────────────────────────────────────


class Shared(Protocol):
    def put(self, name: str, data: bytes) -> None: ...
    def list(self) -> list[tuple[str, float]]: ...
    def claim(self, name: str, owner: str) -> str | None: ...
    def get(self, name: str) -> bytes: ...
    def delete(self, name: str) -> None: ...
    def release(self, name: str) -> None: ...


class LocalTier(rz.Spool):
    """The local directory, with atomic claims."""

    def claim(
        self, item: rz.SpoolItem, owner: str, now: float | None = None
    ) -> Path | None:
        target = item.path.with_name(
            f"{item.path.name}{CLAIM}{owner}@{int(now if now is not None else time.time())}"
        )
        try:
            os.rename(item.path, target)
        except OSError:
            return None
        return target

    def release(self, claimed: Path) -> None:
        base = claimed.name.split(CLAIM)[0]
        try:
            os.rename(claimed, claimed.with_name(base))
        except OSError:
            pass

    def claimed(self):
        return sorted(self.dir.glob(f"*.json{CLAIM}*"))


@dataclass
class Stats:
    pending: int = 0
    bytes: int = 0
    oldest_age_s: float = 0.0
    over_age: int = 0
    claimed: int = 0
    quarantined: int = 0


@dataclass
class DrainResult:
    replayed: int = 0
    held: int = 0
    failed: int = 0
    from_shared: int = 0


def _num(v):
    return None if v is None else float(v)


def _encode(doc: dict) -> bytes:
    return json.dumps(doc, default=str, separators=(",", ":")).encode()


class TieredSpool:
    """Local first, then the shared store; drains both."""

    def __init__(
        self,
        local,
        shared: Shared | None = None,
        *,
        trusted: bool | None = None,
        writer: str = "w",
        lease_s: float = 300.0,
        drain_max_items: int = 20,
        drain_budget_s: float = 10.0,
        dedup_window_s: float = DEDUP_WINDOW_S,
        max_bytes: int = 256 * 1024 * 1024,
        max_age_s: float = 7 * 86400.0,
        clock: Callable[[], float] | None = None,
    ):
        self.local = LocalTier(local)
        self.shared = shared
        self.trusted = shared_allowed() if trusted is None else trusted
        self.writer, self.lease_s = writer, lease_s
        self.drain_max_items, self.drain_budget_s = drain_max_items, drain_budget_s
        self.dedup_window_s = dedup_window_s
        self.max_bytes, self.max_age_s = max_bytes, max_age_s
        self.now = clock or time.time
        self.owner = f"{os.getpid()}-{writer}"

    # -- writing ---------------------------------------------------------------

    def put(self, table, columns, rows, token, database=None, **meta) -> Path:
        meta.setdefault("created", self.now())
        self._check_cap(rows)
        return self.local.put(table, columns, rows, token, database, **meta)

    def put_shared(self, table, columns, rows, token, database=None, **meta) -> None:
        if self.shared is None or not self.trusted:
            raise PermissionError("the shared spool is not available to this run")
        doc = {
            "table": table, "columns": list(columns), "rows": [list(r) for r in rows],
            "token": token, "database": database, "created": meta.get("created", self.now()),
            "scope_count": meta.get("scope_count"), "ambiguous": bool(meta.get("ambiguous")),
            "scope": meta.get("scope"),
        }  # fmt: skip
        self.shared.put(f"{self.writer}/{token}.jsonl", _encode(doc))

    def put_any(self, table, columns, rows, token, database=None, **meta) -> str:
        """Shared when this run may use it and it answers, else local. Raises SpoolFull."""
        self._check_cap(rows)
        if self.shared is not None and self.trusted:
            try:
                self.put_shared(table, columns, rows, token, database, **meta)
                return "shared"
            except (OSError, PermissionError):
                pass
        meta.setdefault("created", self.now())
        self.local.put(table, columns, rows, token, database, **meta)
        return "local"

    def _check_cap(self, rows) -> None:
        extra = len(_encode({"r": [list(r) for r in rows]}))
        if self.stats().bytes + extra > self.max_bytes:
            raise SpoolFull(
                f"spool at {self.stats().bytes} bytes; cap {self.max_bytes}. "
                "Writing more would drop data; failing the write instead."
            )

    # -- observing -------------------------------------------------------------

    def pending_count(self) -> int:
        return self.local_count() + len(self._shared_names())

    def local_count(self) -> int:
        return len(list(self.local.dir.glob("*.json")))

    def quarantined_count(self) -> int:
        return len(list(self.local.dir.glob("*.quarantine"))) + len(
            list(self.local.dir.glob("*.corrupt"))
        )

    def _shared_names(self):
        if self.shared is None or not self.trusted:
            return []
        try:
            return [
                (n, a)
                for n, a in self.shared.list()
                if CLAIM not in n and not n.endswith(".lock")
            ]
        except OSError:
            return []

    def stats(self) -> Stats:
        st = Stats()
        now = self.now()
        ages = []
        for p in self.local.dir.glob("*.json"):
            try:
                created = json.loads(p.read_text()).get("created")
                created = p.stat().st_mtime if created is None else created
                st.bytes += p.stat().st_size
            except (OSError, ValueError):
                continue
            ages.append(now - float(created))
        for _, age in self._shared_names():
            ages.append(float(age))
        st.pending = len(ages)
        st.oldest_age_s = max(ages) if ages else 0.0
        st.over_age = sum(1 for a in ages if a > self.max_age_s)
        st.claimed = len(self.local.claimed())
        st.quarantined = self.quarantined_count()
        return st

    def expire(self) -> int:
        """Move items past the age cap aside. Never deletes: a person decides."""
        moved, now = 0, self.now()
        for item in list(self.local.pending()):
            if item.created is not None and now - item.created > self.max_age_s:
                try:
                    os.rename(
                        item.path, item.path.with_name(item.path.name + ".quarantine")
                    )
                    moved += 1
                except OSError:
                    pass
        return moved

    def reap_claims(self) -> int:
        """Give a dead writer's claim back once its lease has run out."""
        n, now = 0, self.now()
        for c in self.local.claimed():
            try:
                stamp = int(c.name.rsplit("@", 1)[1])
            except (IndexError, ValueError):
                stamp = 0
            if now - stamp >= self.lease_s:
                self.local.release(c)
                n += 1
        return n

    # -- draining --------------------------------------------------------------

    def drain(
        self,
        client,
        *,
        replicated: bool,
        probe: Callable[[rz.SpoolItem], int] | None = None,
        policy: rz.Policy | None = None,
        sleep=None,
        log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
        force: bool = False,
    ) -> DrainResult:
        """Replay pending items, local then shared, within the item and time budget.

        `force` lifts the budget (the backstop job). The first failure stops the drain: a cluster
        that cannot take one batch will not take the next, and the caller's own write must not
        wait behind a dead one.
        """
        res = DrainResult()
        start = self.now()
        self.reap_claims()
        if self.shared is not None and self.trusted and hasattr(self.shared, "reap"):
            try:
                self.shared.reap(self.lease_s)
            except OSError:
                pass
        budget = float("inf") if force else self.drain_budget_s
        cap = 10**9 if force else self.drain_max_items

        def room() -> bool:
            return (
                res.replayed + res.held + res.failed < cap
                and self.now() - start < budget
            )

        for item in list(self.local.pending()):
            if not room():
                return res
            claimed = self.local.claim(item, self.owner, self.now())
            if claimed is None:
                continue
            outcome = self._replay_one(
                client, item, replicated, probe, policy, sleep, log
            )
            if outcome == "ok":
                claimed.unlink(missing_ok=True)
                res.replayed += 1
            elif outcome == "held":
                self.local.release(claimed)
                res.held += 1
            else:
                self.local.release(claimed)
                res.failed += 1
                return res

        if self.shared is None or not self.trusted:
            return res
        try:
            names = [
                n
                for n, _ in self.shared.list()
                if CLAIM not in n and not n.endswith(".lock")
            ]
        except OSError:
            return res
        for name in names:
            if not room():
                return res
            try:
                mine = self.shared.claim(name, self.owner)
                if mine is None:
                    continue
                doc = json.loads(self.shared.get(mine))
            except (OSError, ValueError):
                continue
            item = rz.SpoolItem(
                Path(name), doc["table"], doc["columns"], doc["rows"], doc["token"],
                doc.get("database"), _num(doc.get("created")), doc.get("scope_count"),
                bool(doc.get("ambiguous")), doc.get("scope"),
            )  # fmt: skip
            outcome = self._replay_one(
                client, item, replicated, probe, policy, sleep, log
            )
            try:
                if outcome == "ok":
                    self.shared.delete(mine)
                    res.replayed += 1
                    res.from_shared += 1
                elif outcome == "held":
                    self.shared.release(mine)
                    res.held += 1
                else:
                    self.shared.release(mine)
                    res.failed += 1
                    return res
            except OSError:
                return res
        return res

    def _replay_one(self, client, item, replicated, probe, policy, sleep, log) -> str:
        """'ok', 'held' (needs a person or a probe) or 'failed' (try again later)."""
        policy = policy or rz.Policy()
        aged_out = (
            item.created is not None and self.now() - item.created > self.dedup_window_s
        )
        needs_probe = (not replicated) or aged_out or item.ambiguous
        if needs_probe and (item.ambiguous or not replicated or aged_out):
            if (
                item.scope_count is not None
                or item.ambiguous
                or aged_out
                or not replicated
            ):
                if probe is None:
                    if not replicated or aged_out:
                        log(
                            f"  [spool] {item.token}: held (no row-count probe for a table without dedup)"
                        )
                        return "held"
                else:
                    try:
                        have = probe(item)
                    except Exception as err:  # noqa: BLE001
                        log(
                            f"  [spool] {item.token}: probe failed ({type(err).__name__}); held"
                        )
                        return "held"
                    base = item.scope_count if item.scope_count is not None else 0
                    if have - base >= len(item.rows):
                        return "ok"  # already stored: nothing to send
                    if have - base != 0:
                        log(
                            f"  [spool] {item.token}: {have - base:+d} rows differ; held"
                        )
                        return "held"
        try:
            rz._send(
                client, item.table, item.rows, item.columns, item.database,
                rz._settings(policy, item.token, replicated, False),
            )  # fmt: skip
        except Exception as err:  # noqa: BLE001
            kind = rz.classify(err)
            log(f"  [spool] {item.token}: {kind} ({type(err).__name__}); kept")
            return "held" if kind == "fatal" else "failed"
        return "ok"


def drain_at_start(client, spool: TieredSpool, *, replicated: bool, **kw) -> int:
    """Call once when a process starts, before its first write."""
    return spool.drain(client, replicated=replicated, **kw).replayed


def compress(data: bytes) -> bytes:
    """zstd when available, else the bytes unchanged; the caller names the file accordingly."""
    try:
        import zstandard as zstd  # noqa: PLC0415
    except ImportError:
        return data
    buf = io.BytesIO()
    with zstd.ZstdCompressor(level=3).stream_writer(buf, closefd=False) as w:
        w.write(data)
    return buf.getvalue()


def metrics(spool: TieredSpool) -> dict:
    """The numbers a monitor writes: pending count, oldest age, bytes, items past the age cap."""
    st = spool.stats()
    return {
        "ch_spool_pending": st.pending,
        "ch_spool_oldest_age_s": round(st.oldest_age_s, 1),
        "ch_spool_bytes": st.bytes,
        "ch_spool_over_age": st.over_age,
        "ch_spool_claimed": st.claimed,
        "ch_spool_quarantined": st.quarantined,
    }


def from_env(writer: str, env=None, **kw) -> TieredSpool | None:
    """A TieredSpool from CH_WRITE_SPOOL (local dir) and, for a trusted run, the shared tier.

    None when no spool is configured, so a caller without one keeps its old behaviour.
    """
    env = os.environ if env is None else env
    local = (env.get("CH_WRITE_SPOOL") or "").strip()
    if not local:
        return None
    trusted = shared_allowed(env)
    shared = None
    if trusted:
        from .artifactory_spool import ArtifactorySpool  # noqa: PLC0415

        shared = ArtifactorySpool.from_env(env)
    return TieredSpool(local, shared, trusted=trusted, writer=writer, **kw)
