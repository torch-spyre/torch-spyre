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

"""Seeded fault injection over write, spool and replay.

Properties, for plain and Replicated tables, across process crashes at every send/rewrite/
delete step, a lease that expires mid-send, shared-tier failures, a full disk and a second
drainer: no row is stored twice, and no acknowledged row is lost (stored, or still in the spool).
FUZZ_SEEDS=1000 runs a longer sweep.
"""

import errno
import json
import os
import random
from collections import Counter
from pathlib import Path

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

from spyre_clickhouse_ingest import resilience as rz
from spyre_clickhouse_ingest import spool as sp

WINDOW = 3600.0
LEASE = 420.0
NSEEDS = int(os.environ.get("FUZZ_SEEDS", "12"))
VARIANTS = {
    "base": {},
    "race": {"race": 1},
    "crash": {"crash": 1},
    "shared": {"shared": 1},
    "fs": {"fs": 1, "crash": 1},
    "all": {"race": 1, "crash": 1, "shared": 1, "fs": 1},
}


class Crash(BaseException):
    """The process dies here; nothing after it runs."""


def derr(code):
    return DatabaseError(f"Code: {code}", code=code)


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


class Server:
    def __init__(self, rng, replicated, clock, feat, make_other):
        self.rng, self.replicated, self.clock, self.feat = rng, replicated, clock, feat
        self.rows, self.tok, self.faults = [], {}, True
        self.make_other, self.in_hook = make_other, False

    def _store(self, data, tok, k=None):
        rows = data if k is None else data[:k]
        for i, row in enumerate(rows):
            key = (tok, i)
            if self.replicated and tok:
                if key in self.tok and self.clock() - self.tok[key] < WINDOW:
                    continue
                self.tok[key] = self.clock()
            self.rows.append(row[0])

    def insert(self, table, data, column_names=None, database=None, settings=None):
        tok = (settings or {}).get("insert_deduplication_token")
        rng = self.rng
        if (
            self.feat.get("race")
            and self.faults
            and not self.in_hook
            and rng.random() < 0.15
        ):
            self.in_hook = True  # a send slower than the lease: another drainer gets in
            try:
                self.clock.t += LEASE + 1
                self.make_other().drain(
                    self, replicated=self.replicated, force=True, log=lambda m: None
                )
            finally:
                self.in_hook = False
        if not self.faults:
            self._store(data, tok)
            return
        r = rng.random()
        if r < 0.20:
            raise derr(rng.choice([285, 242, 286]))  # refused before any write
        if r < 0.45:
            self._store(data, tok)  # stored, then the response is an error
            raise rng.choice(
                [derr(319), derr(159), OSError("reset"), OperationalError("t")]
            )
        if r < 0.55:
            raise rng.choice([derr(319), OSError("x")])  # request lost
        if r < 0.62 and len(data) > 1:
            self._store(data, tok, k=rng.randint(1, len(data) - 1))
            raise OSError("partial")
        if self.feat.get("crash") and r < 0.72:
            if rng.random() < 0.6:
                self._store(data, tok)
            raise Crash()
        self._store(data, tok)


class FakeShared:
    def __init__(self, rng, clock, p):
        self.rng, self.clock, self.p = rng, clock, p
        self.files, self.locks, self.on = {}, {}, True

    def _fault(self):
        if self.on and self.rng.random() < self.p:
            if self.rng.random() < 0.3:
                raise sp.Unreachable("refused")
            return self.rng.random() < 0.5  # True: the operation took effect anyway
        return None

    def _do(self, landed, effect):
        if landed is not None:
            if landed:
                effect()
            raise OSError("shared failed")
        effect()

    def put(self, name, data):
        self._do(self._fault(), lambda: self.files.__setitem__(name, data))

    def list(self):
        if self._fault() is not None:
            raise OSError("list failed")
        return sorted((n, 0.0) for n in self.files)

    def claim(self, name, owner):
        if self._fault() is not None:
            raise OSError("claim failed")
        lk = self.locks.get(name)
        if lk and self.clock() - lk[1] < LEASE:
            return None
        self.locks[name] = (owner, self.clock())
        return name

    def get(self, name):
        if self._fault() is not None:
            raise OSError("get failed")
        return self.files[name]

    def _drop(self, name):
        self.files.pop(name, None)
        self.locks.pop(name, None)

    def delete(self, name):
        self._do(self._fault(), lambda: self._drop(name))

    def release(self, name):
        self._do(self._fault(), lambda: self.locks.pop(name, None))

    def reap(self, lease):
        for n, (_, ts) in list(self.locks.items()):
            if self.clock() - ts >= lease:
                del self.locks[n]


def spooled_ids(local_dir, shared):
    ids = set()
    for p in Path(local_dir).iterdir():
        if p.name.endswith(".tmp"):
            continue
        try:
            ids |= {r[0] for r in json.loads(p.read_text())["rows"]}
        except Exception:  # noqa: BLE001
            ids.add(("unreadable", p.name))
    for n, data in (shared.files if shared else {}).items():
        try:
            ids |= {r[0] for r in json.loads(data)["rows"]}
        except Exception:  # noqa: BLE001
            ids.add(("unreadable", n))
    return ids


def run_seed(seed, replicated, feat, tmp_path, monkeypatch):
    rng = random.Random(seed)
    clock = Clock()
    d = tmp_path / f"s{seed}-{replicated}"
    d.mkdir()
    shared = FakeShared(rng, clock, 0.25) if feat.get("shared") else None
    fs_on = [False]

    def flaky(real):
        def wrapper(*a, **k):
            if fs_on[0] and rng.random() < 0.08:
                if rng.random() < 0.5:
                    raise OSError(errno.ENOSPC, "No space left on device")
                if rng.random() < 0.5:
                    raise Crash()
                real(*a, **k)  # the step happened, then the process died
                raise Crash()
            return real(*a, **k)

        return wrapper

    monkeypatch.setattr(os, "replace", flaky(os.replace))
    monkeypatch.setattr(os, "fsync", flaky(os.fsync))
    monkeypatch.setattr(
        sp.LocalTier, "release", _crashing_release(rng, fs_on, sp.LocalTier.release)
    )

    def mk(owner):
        return sp.TieredSpool(
            d, shared, trusted=bool(shared), writer=owner, dedup_window_s=WINDOW,
            lease_s=LEASE, send_timeout_s=300.0, clock=clock,
            drain_max_items=rng.choice([2, 20]),
        )  # fmt: skip

    srv = Server(rng, replicated, clock, feat, lambda: mk(f"o{rng.randint(0, 99)}"))
    nid, acked = 0, []
    for proc in range(rng.randint(1, 4)):
        spool = mk(f"p{proc}")
        n = rng.randint(2, 10)
        ids = list(range(nid, nid + n))
        nid += n
        pol = rz.Policy(
            batch_rows=rng.randint(1, 4), deadline_s=rng.choice([0, 3, 9]), backoff_s=1
        )

        def sleep(s):
            clock.t += s

        try:
            rz.resilient_insert(
                srv, "t", [[i] for i in ids], ["id"], scope=f"s{seed}-{proc}", policy=pol,
                replicated=replicated, spool=spool, sleep=sleep, clock=clock, log=lambda m: None,
            )  # fmt: skip
            acked += ids
        except (Crash, rz.WriteAmbiguous, rz.WriteUnavailable, sp.SpoolFull):
            pass
        fs_on[0] = bool(feat.get("fs"))
        for _ in range(rng.randint(0, 3)):
            clock.t += rng.choice([1, 60, 400, 5000, -100])
            try:
                mk("d").drain(
                    srv,
                    replicated=replicated,
                    policy=pol,
                    sleep=sleep,
                    log=lambda m: None,
                )
            except (Crash, OSError):
                pass
        fs_on[0] = False
        clock.t += rng.choice([1, 400, 5000])
    srv.faults = False
    if shared:
        shared.on = False
    clock.t += 10000
    for _ in range(3):
        mk("final").drain(srv, replicated=replicated, force=True, log=lambda m: None)
        clock.t += 400
    counts = Counter(srv.rows)
    dup = sorted(i for i, k in counts.items() if k > 1)
    held = spooled_ids(d, shared)
    lost = sorted(i for i in acked if i not in counts and i not in held)
    return dup, lost


def _crashing_release(rng, fs_on, real):
    def release(self, claimed):
        if fs_on[0] and rng.random() < 0.2:
            raise Crash()
        return real(self, claimed)

    return release


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("seed", [11, 23, *range(NSEEDS)])
def test_no_row_is_stored_twice_or_lost(
    tmp_path, monkeypatch, seed, variant, replicated
):
    dup, lost = run_seed(seed, replicated, VARIANTS[variant], tmp_path, monkeypatch)
    assert not dup, f"stored twice: {dup}"
    assert not lost, f"lost: {lost}"
