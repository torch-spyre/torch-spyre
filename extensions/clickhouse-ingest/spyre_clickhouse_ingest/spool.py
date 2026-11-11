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
* A batch refused before any write (`ambiguous` False) is sent again as is, on any table.
* A batch a failed attempt may have stored (`ambiguous` True) is safe to resend only on a
  Replicated table inside its dedup window, where the original `insert_deduplication_token`
  makes the replay exactly-once. On a plain table, or past the window, it is held until
  `probe(item)` reports how many of THIS batch's own rows are stored: all of them drops the
  item, none sends it, anything between holds it for a person. The probe must identify the
  batch (its key columns or a per-batch marker), never count a whole scope: other writers' and
  earlier batches' rows in the scope would make a lost batch look stored.
* Nothing in the repo calls `resilient_insert` or `TieredSpool` yet, and no probe exists. The
  schema has no per-batch marker (the sort keys are not unique per batch), so `probe=None` is
  the only live path and every ambiguous plain-table batch is held. A real probe needs a batch
  marker, such as a batch-id column or `insert_deduplication_token` on a Replicated table;
  counting rows by sort key would call a lost batch stored.
* A crash between a successful send and the file's removal resends a plain-table batch once
  more; only a Replicated table can absorb that.
* The shared tier holds row data. It is used only by trusted runs (see `shared_allowed`) and any
  failure there falls back to the local directory.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from . import resilience as rz

CLAIM = ".claimed-"
CORRUPT = ".corrupt"
# clickhouse-connect's send_receive_timeout default; a lease must outlast one send.
SEND_TIMEOUT_S = 300.0
LEASE_MARGIN_S = 60.0
# Conservative: ClickHouse keeps tokens for the last N blocks per table (replicated_deduplication_window)
# and for 7 days; a busy table can evict a token well before the time bound.
DEDUP_WINDOW_S = 3600.0
TRUSTED_EVENTS = frozenset({"push", "schedule", "workflow_dispatch"})


class Unreachable(OSError):
    """The shared tier refused the connection before sending anything: nothing was stored."""


class SpoolFull(RuntimeError):
    """The spool is at its size cap; the write must fail loudly instead of dropping data."""


# ── trust ─────────────────────────────────────────────────────────────────────


def shared_allowed(env=None) -> bool:
    """True only for a run that may put row data in the shared store.

    Default closed. An explicit CH_WRITE_SHARED_SPOOL_TRUSTED=1 is required. Inside GitHub Actions
    the event must also be on an allowlist (push, schedule, workflow_dispatch) and the ref
    protected: every other event (pull_request*, issue_comment, workflow_run, merge_group, ...)
    can run code the team did not review with secrets in reach. Outside Actions the flag alone
    decides.
    """
    env = os.environ if env is None else env
    if (env.get("CH_WRITE_SHARED_SPOOL_TRUSTED") or "").strip() not in (
        "1",
        "true",
        "yes",
    ):
        return False
    event = (env.get("GITHUB_EVENT_NAME") or "").strip()
    if env.get("GITHUB_ACTIONS") or event:
        if event not in TRUSTED_EVENTS:
            return False
        if (env.get("GITHUB_REF_PROTECTED") or "").strip().lower() != "true":
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
        lease_s: float | None = None,
        send_timeout_s: float | None = SEND_TIMEOUT_S,
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
        if lease_s is None:
            lease_s = (send_timeout_s or 240.0) + 2 * LEASE_MARGIN_S
        elif send_timeout_s is not None and lease_s < send_timeout_s + LEASE_MARGIN_S:
            raise ValueError(
                f"lease_s={lease_s} must exceed send_timeout_s={send_timeout_s} by "
                f"{LEASE_MARGIN_S}s, or a slow send outlives its claim"
            )
        shared_lease = getattr(shared, "lease_s", None)
        if shared_lease is not None:
            if (
                send_timeout_s is not None
                and shared_lease < send_timeout_s + LEASE_MARGIN_S
            ):
                raise ValueError(
                    f"the shared tier's lease_s={shared_lease} must exceed send_timeout_s="
                    f"{send_timeout_s} by {LEASE_MARGIN_S}s"
                )
            lease_s = min(lease_s, shared_lease)
        self.writer, self.lease_s = writer, lease_s
        self.drain_max_items, self.drain_budget_s = drain_max_items, drain_budget_s
        self.dedup_window_s = dedup_window_s
        self.max_bytes, self.max_age_s = max_bytes, max_age_s
        self.now = clock or time.time
        self.owner = f"{os.getpid()}-{writer}-{uuid.uuid4().hex[:8]}"

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
            "ambiguous": bool(meta.get("ambiguous")), "scope": meta.get("scope"),
        }  # fmt: skip
        self.shared.put(f"{self.writer}/{token}.jsonl", _encode(doc))

    def put_any(self, table, columns, rows, token, database=None, **meta) -> str:
        """Shared when this run may use it and it answers, else local. Raises SpoolFull."""
        self._check_cap(rows)
        if self.shared is not None and self.trusted:
            try:
                self.put_shared(table, columns, rows, token, database, **meta)
                return "shared"
            except (Unreachable, PermissionError):
                pass
            except OSError:
                # The PUT may have landed before it failed; the local copy must not be resent
                # blindly beside it.
                meta["ambiguous"] = True
                meta["twin"] = f"{self.writer}/{token}.jsonl"
        meta.setdefault("created", self.now())
        self.local.put(table, columns, rows, token, database, **meta)
        return "local"

    def _check_cap(self, rows) -> None:
        extra = len(_encode({"r": [list(r) for r in rows]}))
        used = 0
        for p in self.local.dir.glob("*.json"):
            try:
                used += p.stat().st_size
            except OSError:
                pass
        if used + extra > self.max_bytes:
            raise SpoolFull(
                f"spool at {used} bytes; cap {self.max_bytes}. "
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
                if CLAIM not in n and not n.endswith((".lock", CORRUPT))
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
        self._check_client_timeout(client)
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
                res.replayed + res.failed < cap  # held items must not starve the rest
                and self.now() - start < budget
            )

        for listed in list(self.local.pending()):
            if not room():
                return res
            claimed = self.local.claim(listed, self.owner, self.now())
            if claimed is None:
                continue
            item = self._read_claimed(claimed)
            if item is None:
                self.local.release(claimed)
                continue
            outcome = self._replay_one(
                client, item, replicated, probe, policy, sleep, log,
                self._guarded(lambda amb, c=claimed: self._mark_local(c, amb)),
            )  # fmt: skip
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
                if CLAIM not in n and not n.endswith((".lock", CORRUPT))
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
                raw = self.shared.get(mine)
            except OSError:
                continue
            try:
                doc = json.loads(raw)
                item = self._item_from_doc(Path(name), doc)
            except (ValueError, KeyError, TypeError):
                # Unreadable for good: set it aside so it is not claimed and failed every run.
                try:
                    self.shared.put(f"{name}{CORRUPT}", raw)
                    self.shared.delete(mine)
                except OSError:
                    pass
                log(f"  [spool] {name}: unreadable; set aside")
                res.failed += 1
                continue
            outcome = self._replay_one(
                client, item, replicated, probe, policy, sleep, log,
                self._guarded(lambda amb, n=name, it=item: self._mark_shared(n, it, amb)),
            )  # fmt: skip
            try:
                if outcome == "ok":
                    res.replayed += 1
                    res.from_shared += 1
                    self.shared.delete(mine)
                elif outcome == "held":
                    self.shared.release(mine)
                    res.held += 1
                else:
                    self.shared.release(mine)
                    res.failed += 1
                    return res
            except OSError:
                # The item was marked "may be stored" before it was sent, so a leftover copy
                # is held (or deduplicated by its token), never resent blindly.
                log(f"  [spool] {name}: shared tier failed after the send")
                return res
        return res

    # -- replay state ----------------------------------------------------------

    def _guarded(self, mark):
        """Clearing "may be stored" is only safe while this drainer still plainly holds the claim.

        A drainer that outlived its lease may find the item already replayed and deleted by
        another one; writing the clean record back would resurrect it. A send is bounded below the
        lease, so a late clear is skipped and the item stays held.
        """
        claimed_at = self.now()

        def persist(ambiguous: bool) -> bool:
            if (
                not ambiguous
                and self.now() - claimed_at >= self.lease_s - LEASE_MARGIN_S
            ):
                return False
            return mark(ambiguous)

        return persist

    @staticmethod
    def _item_from_doc(path: Path, doc: dict) -> rz.SpoolItem:
        return rz.SpoolItem(
            path, doc["table"], doc["columns"], doc["rows"], doc["token"],
            doc.get("database"), _num(doc.get("created")),
            bool(doc.get("ambiguous")), doc.get("scope"), doc.get("twin"),
        )  # fmt: skip

    def _read_claimed(self, claimed: Path) -> rz.SpoolItem | None:
        """The item as it is on disk now: a snapshot taken before the claim may be stale."""
        try:
            return self._item_from_doc(claimed, json.loads(claimed.read_text()))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _mark_local(path: Path, ambiguous: bool, clear_twin: bool = False) -> bool:
        """Durably rewrite `ambiguous` in the file at `path`; the original survives a failure.

        The temp name matches neither the *.json nor the claim glob, so a crash or a full disk
        cannot leave a half-written file that a reaper would release over the good one.
        """
        tmp = path.with_name(f".rw-{uuid.uuid4().hex[:8]}.tmp")
        try:
            doc = json.loads(path.read_text())
            doc["ambiguous"] = ambiguous
            if clear_twin:
                doc["twin"] = None
            with open(tmp, "w") as fh:
                json.dump(doc, fh, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return True
        except (OSError, ValueError):
            try:
                tmp.unlink()
            except OSError:
                pass
            return False

    def _mark_shared(self, name: str, item: rz.SpoolItem, ambiguous: bool) -> bool:
        doc = {
            "table": item.table, "columns": list(item.columns), "rows": item.rows,
            "token": item.token, "database": item.database, "created": item.created,
            "ambiguous": ambiguous, "scope": item.scope,
        }  # fmt: skip
        try:
            self.shared.put(name, _encode(doc))
            return True
        except OSError:
            return False

    def _replay_one(
        self, client, item, replicated, probe, policy, sleep, log, persist
    ) -> str:
        """'ok', 'held' (needs a person or a probe) or 'failed' (try again later).

        Write-ahead: before the send the item is persisted as "may be stored" (`persist(True)`),
        so a crash, a lease that expires mid-send or a lost response can only leave a held item,
        never a clean one that gets resent blindly. If that record cannot be written, nothing is
        sent. The record is cleared again only on an error that proves nothing was written.
        """
        policy = policy or rz.Policy()
        if item.twin:
            log(f"  [spool] {item.token}: held (a shared copy may exist beside it)")
            return "held"
        aged_out = (
            item.created is not None and self.now() - item.created > self.dedup_window_s
        )
        if item.ambiguous and (not replicated or aged_out):
            if probe is None:
                log(f"  [spool] {item.token}: held (may be stored; no probe to tell)")
                return "held"
            try:
                have = probe(item)
            except Exception as err:  # noqa: BLE001
                log(
                    f"  [spool] {item.token}: probe failed ({type(err).__name__}); held"
                )
                return "held"
            if have >= len(item.rows):
                return "ok"  # already stored: nothing to send
            if have != 0:
                log(
                    f"  [spool] {item.token}: {have}/{len(item.rows)} rows stored; held"
                )
                return "held"
        was_clean = not item.ambiguous
        if was_clean:
            if not persist(True):
                log(f"  [spool] {item.token}: could not record the attempt; not sent")
                return "failed"
            item.ambiguous = True
        try:
            rz._send(
                client, item.table, item.rows, item.columns, item.database,
                rz._settings(policy, item.token, replicated, False),
            )  # fmt: skip
        except Exception as err:  # noqa: BLE001
            kind = rz.classify(err)
            if kind == "retry" and was_clean:
                persist(
                    False
                )  # best effort: on failure the item stays held, which is safe
            log(f"  [spool] {item.token}: {kind} ({type(err).__name__}); kept")
            return "held" if kind == "fatal" else "failed"
        return "ok"

    # -- held items: what a person can do ---------------------------------------

    def items(self) -> list[dict]:
        """Every pending item, local and shared, for `spool list`. Rows are not included."""
        out = []
        for it in self.local.pending():
            out.append(self._describe("local", it.path.name, it))
        if self.shared is not None and self.trusted:
            try:
                for name, _ in self._shared_names():
                    out.append(self._describe("shared", name, self._shared_item(name)))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return out

    def _describe(self, tier, name, it) -> dict:
        return {
            "tier": tier, "name": name, "token": it.token, "table": it.table,
            "rows": len(it.rows), "held": it.ambiguous, "created": it.created,
        }  # fmt: skip

    def _shared_item(self, name) -> rz.SpoolItem:
        return self._item_from_doc(Path(name), json.loads(self.shared.get(name)))

    def release_held(self, token: str) -> int:
        """Mark items with this token as safe to resend. A person calls this only after checking
        the table: the batch's rows are NOT stored (or may be stored twice harmlessly)."""
        n = 0
        for p in self.local.dir.glob(f"*-{token}.json"):
            n += self._mark_local(p, False, clear_twin=True)
        for name, _ in self._shared_names():
            if name.endswith(f"/{token}.jsonl"):
                n += self._mark_shared(name, self._shared_item(name), False)
        return n

    def drop(self, token: str) -> int:
        """Delete items with this token, after a person has checked the rows are stored or not
        wanted."""
        n = 0
        for p in self.local.dir.glob(f"*-{token}.json"):
            p.unlink(missing_ok=True)
            n += 1
        for name, _ in self._shared_names():
            if name.endswith(f"/{token}.jsonl"):
                self.shared.delete(name)
                n += 1
        return n

    def _check_client_timeout(self, client) -> None:
        """A send must finish inside the lease, or a second drainer takes the item mid-send."""
        t = getattr(client, "timeout", None)
        if isinstance(t, (int, float)) and t + LEASE_MARGIN_S > self.lease_s:
            raise ValueError(
                f"client timeout {t}s needs a lease_s above {t + LEASE_MARGIN_S}s "
                f"(have {self.lease_s}s)"
            )


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

        lease = kw.get("lease_s") or (
            (kw.get("send_timeout_s") or SEND_TIMEOUT_S) + 2 * LEASE_MARGIN_S
        )
        shared = ArtifactorySpool.from_env(env, lease_s=lease)
        kw["lease_s"] = lease
    return TieredSpool(local, shared, trusted=trusted, writer=writer, **kw)
