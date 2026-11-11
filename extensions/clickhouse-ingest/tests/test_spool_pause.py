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

"""A drainer that crashes or stalls past its lease at every I/O step of a replay.

For each step of drainer A the schedule either kills A there or pauses it for a lease (+1 s, then
twice that) while drainer B drains everything, then lets A continue; recovery drains follow. The
shared tier is the real ArtifactorySpool over a fake HTTP service. Properties, plain and
Replicated tables: no row is stored twice, none is lost (stored, or still in the spool).
PAUSE_STRIDE=1 tries every step and PAUSE_SEEDS=N adds N seeds; the default samples every few.
"""

import io
import itertools
import json
import os
import pathlib
import random
import time
import urllib.error
from collections import Counter, defaultdict

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

from spyre_clickhouse_ingest import spool as sp
from spyre_clickhouse_ingest.artifactory_spool import ArtifactorySpool

WINDOW = 3600.0
STRIDE = int(os.environ.get("PAUSE_STRIDE", "4"))
SEEDS = [1, 2, 4, *range(100, 100 + int(os.environ.get("PAUSE_SEEDS", "0")))]
LEASE = sp.SEND_TIMEOUT_S + 2 * sp.LEASE_MARGIN_S
W = [1_000_000.0]


class Crash(BaseException):
    """The process dies here."""


class Art:
    """A fake Artifactory HTTP service: PUT/GET/DELETE/list, optional landed-then-failed faults."""

    def __init__(self, rng):
        self.f, self.mt, self.rng, self.fault, self.hook = {}, {}, rng, 0.0, None

    def __call__(self, req, timeout=None, context=None):
        m, url = req.get_method(), req.full_url
        if self.hook:
            self.hook(m, url)
        path = url.split("http://a/", 1)[1]
        landed = None
        if self.fault and self.rng.random() < self.fault:
            landed = self.rng.random() < 0.5

        def done(status, body=b""):
            class R:
                headers = None

                def read(self):
                    return body

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            r = R()
            r.status = status
            return r

        def nf():
            return urllib.error.HTTPError(url, 404, "nf", {}, io.BytesIO())

        if path.startswith("api/storage/"):
            pre = path.split("?")[0][len("api/storage/") :] + "/"
            files = [
                {
                    "uri": "/" + k[len(pre) :],
                    "lastModified": time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(self.mt[k])
                    ),
                }
                for k in sorted(self.f)
                if k.startswith(pre)
            ]
            if not files:
                raise nf()
            return done(200, json.dumps({"files": files}).encode())
        if m == "PUT":
            if landed is False:
                raise TimeoutError("t")
            self.f[path], self.mt[path] = req.data, W[0]
            if landed:
                raise TimeoutError("t")
            return done(201)
        if m == "GET":
            if path not in self.f:
                raise nf()
            return done(200, self.f[path])
        existed = path in self.f  # DELETE
        if landed is False:
            raise TimeoutError("t")
        self.f.pop(path, None)
        self.mt.pop(path, None)
        if landed:
            raise TimeoutError("t")
        if not existed:
            raise nf()
        return done(204)


class Server:
    def __init__(self, rng, replicated):
        self.rng, self.rep, self.fault, self.hook = rng, replicated, 0.0, None
        self.n, self.seen, self.toks = Counter(), {}, defaultdict(set)

    def store(self, data, tok):
        if self.rep and tok:
            if tok in self.seen and W[0] - self.seen[tok] < WINDOW:
                return
            self.seen[tok] = W[0]
        for r in data:
            self.n[r[0]] += 1

    def insert(self, table, data, column_names=None, database=None, settings=None):
        tok = (settings or {}).get("insert_deduplication_token")
        for r in data:
            self.toks[r[0]].add(tok)
        if self.hook:
            self.hook("insert-pre")
        o = "ok"
        if self.fault and self.rng.random() < self.fault:
            o = self.rng.choice(
                ["refuse", "conn", "stored_err", "lost", "stored_err", "oserr"]
            )
        if o == "refuse":
            raise DatabaseError("Code: 285", code=285)
        if o == "conn":
            raise ConnectionRefusedError("refused")
        if o == "lost":
            raise OperationalError("read timeout")
        if o == "oserr":
            raise OSError("reset")
        self.store(data, tok)
        if self.hook:
            self.hook("insert-post")
        if o == "stored_err":
            raise DatabaseError("Code: 319", code=319)


class Ev:
    def __init__(self):
        self.on = False
        self.k = self.kind = self.stall = None
        self.cnt, self.fired = 0, False


EV = Ev()


def ev(name, phase):
    if not EV.on:
        return
    EV.cnt += 1
    if EV.k == EV.cnt and not EV.fired:
        EV.fired = True
        if EV.kind == "crash":
            raise Crash(f"{name}:{phase}")
        EV.on = False
        try:
            EV.stall()
        finally:
            EV.on = True


def wrap(monkeypatch, obj, attr, name):
    orig = getattr(obj, attr)

    def f(*a, **kw):
        ev(name, "pre")
        r = orig(*a, **kw)
        ev(name, "post")
        return r

    monkeypatch.setattr(obj, attr, f)


def build(seed, rep, layout, ld):
    rng = random.Random(seed)
    art, srv = Art(rng), Server(rng, rep)

    def mk(writer):
        shared = (
            None
            if layout == "local"
            else ArtifactorySpool(
                "t",
                base="http://a",
                repo="r",
                opener=art,
                clock=lambda: W[0],
                sleep=lambda s: None,
            )
        )
        return sp.TieredSpool(
            ld, shared, trusted=layout != "local", writer=writer, clock=lambda: W[0],
            dedup_window_s=WINDOW, drain_max_items=50, feeds_views=False,
        )  # fmt: skip

    seeder = mk("seed")
    rid, items = itertools.count(1), []
    for i in range(rng.choice([2, 3])):
        rows = [[next(rid), "x"] for _ in range(rng.choice([1, 3]))]
        tok = f"t-{seed}-{i}"
        amb = rng.random() < 0.5
        age = rng.choice([10, 100, 4000]) if amb else rng.choice([10, 100])
        meta = dict(ambiguous=amb, created=W[0] - age, scope="s")
        tier = (
            rng.choice(["local", "shared"])
            if layout == "mixed"
            else ("shared" if layout == "shared" else "local")
        )
        if amb and rng.random() < 0.5:  # it did land before the spool was written
            srv.store(rows, tok if rep else None)
            srv.seen[tok] = W[0] - age if rep else 0
        put = seeder.put_shared if tier == "shared" else seeder.put
        put("t", ["id", "v"], rows, tok, None, **meta)
        items.append(rows)
    if layout == "twin":  # a put_any whose shared PUT landed and then failed
        rows = [[next(rid), "x"] for _ in range(2)]
        tok = f"tw-{seed}"
        orig = art.__call__

        def put_landed(req, timeout=None, context=None):
            if req.get_method() == "PUT":
                key = f"r/clickhouse-spool/seed/{tok}.jsonl"
                art.f[key], art.mt[key] = req.data, W[0]
                raise TimeoutError("t")
            return orig(req, timeout, context)

        seeder.shared._open = put_landed
        seeder.put_any(
            "t",
            ["id", "v"],
            rows,
            tok,
            None,
            ambiguous=False,
            created=W[0] - 5,
            scope="s",
        )
        seeder.shared._open = art
        items.append(rows)
    return rng, art, srv, mk, items


def present_ids(ld, art):
    ids = set()
    for p in pathlib.Path(ld).iterdir():
        if p.name.startswith(".rw-") or p.name.endswith(".tmp"):
            continue
        try:
            ids |= {r[0] for r in json.loads(p.read_text())["rows"]}
        except Exception:  # noqa: BLE001
            ids.add(("unreadable", p.name))
    for k, v in art.f.items():
        if not k.endswith(".lock"):
            try:
                ids |= {r[0] for r in json.loads(v)["rows"]}
            except Exception:  # noqa: BLE001
                ids.add(("unreadable", k))
    return ids


def run_case(tmp_path, monkeypatch, seed, rep, layout, kind, k, delta, case):
    W[0] = 1_000_000.0
    ld = tmp_path / f"c{case}"
    ld.mkdir()
    rng, art, srv, mk, items = build(seed, rep, layout, ld)
    allids = [r[0] for it in items for r in it]
    a = mk("A")

    def stall():
        W[0] += delta
        srv.fault = art.fault = 0.0
        mk("B").drain(
            srv, replicated=rep, sleep=lambda s: None, log=lambda m: None, force=True
        )

    EV.__init__()
    EV.k, EV.kind, EV.stall = k, kind, stall
    with monkeypatch.context() as m:
        for obj, attr, name in (
            (os, "rename", "rename"), (os, "replace", "replace"), (os, "fsync", "fsync"),
            (pathlib.Path, "unlink", "unlink"), (pathlib.Path, "read_text", "read"),
        ):  # fmt: skip
            wrap(m, obj, attr, name)
        srv.hook = lambda n: ev("srv-" + n, "x")
        art.hook = lambda meth, u: ev("http-" + meth, "x")
        EV.on = True
        srv.fault = 0.3
        try:
            a.drain(
                srv,
                replicated=rep,
                sleep=lambda s: None,
                log=lambda m: None,
                force=True,
            )
        except Crash:
            pass
        EV.on = False
        srv.hook = art.hook = None
    viol = []

    def check(stage):
        pres = present_ids(ld, art)
        for i in allids:
            if srv.n[i] > 1:
                viol.append(f"stored twice: row {i} ({stage})")
            if srv.n[i] == 0 and i not in pres:
                viol.append(f"lost: row {i} ({stage})")
            if rep and len(srv.toks[i] - {None}) > 1:
                viol.append(f"two tokens: row {i}")

    check("after A")
    srv.fault = art.fault = 0.0
    for rnd in range(4):
        W[0] += 400
        mk(f"R{rnd}").drain(
            srv, replicated=rep, sleep=lambda s: None, log=lambda m: None, force=True
        )
    check("after recovery")
    return viol, EV.cnt


@pytest.fixture(autouse=True)
def _no_tls_context(monkeypatch):
    monkeypatch.setattr(
        "ssl.create_default_context", lambda *a, **k: None
    )  # slow, unused


CASES = [("crash", 0), ("stall", 0), ("stall", LEASE + 1), ("stall", 2 * LEASE)]


@pytest.mark.parametrize("rep", [False, True])
@pytest.mark.parametrize("layout", ["local", "shared", "mixed", "twin"])
@pytest.mark.parametrize("kind,delta", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_a_crash_or_pause_at_any_step_never_doubles_or_loses_a_row(
    tmp_path, monkeypatch, seed, kind, delta, layout, rep
):
    _, steps = run_case(
        tmp_path, monkeypatch, seed, rep, layout, kind, 10**9, delta, "dry"
    )
    bad = []
    for n, k in enumerate(range(1, steps + 1, STRIDE)):
        viol, _ = run_case(
            tmp_path, monkeypatch, seed, rep, layout, kind, k, delta, f"{n}"
        )
        if viol:
            bad.append((k, viol[:2]))
    assert not bad, bad
