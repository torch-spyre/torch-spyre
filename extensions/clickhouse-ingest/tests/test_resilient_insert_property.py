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

"""Seeded fault injection over resilient_insert and the plain Spool.

Properties, for plain and Replicated tables, with and without a view on the table: no row is
stored twice, every row of a call that returned is stored or parked, a Replicated batch is always
sent under one token, and a plain-table batch is not sent again once a send may have stored it.
"""

import random
from collections import Counter, defaultdict

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

from spyre_clickhouse_ingest import resilience as rz

WINDOW = 3600.0


class Server:
    def __init__(self, rng, replicated, clock):
        self.rng, self.replicated, self.clock = rng, replicated, clock
        self.n, self.seen = Counter(), {}
        self.toks, self.sends_after_doubt = defaultdict(set), 0
        self.doubt = set()

    def insert(self, table, data, column_names=None, database=None, settings=None):
        tok = (settings or {}).get("insert_deduplication_token")
        ids = tuple(r[0] for r in data)
        self.toks[ids].add(tok)
        if not self.replicated and ids in self.doubt:
            self.sends_after_doubt += 1
        r = self.rng.random()
        if r < 0.25:
            raise DatabaseError("Code: 285", code=285)  # refused before any write
        if r < 0.40:
            self.doubt.add(ids)
            raise OperationalError("read timeout")  # request lost: stored or not
        stored = ids[: self.rng.randint(1, len(ids))] if r < 0.55 else ids
        for i, row_id in enumerate(stored):
            key = (tok, i)
            if self.replicated and tok:
                if key in self.seen and self.clock() - self.seen[key] < WINDOW:
                    continue
                self.seen[key] = self.clock()
            self.n[row_id] += 1
        if r < 0.70:
            self.doubt.add(ids)
            raise DatabaseError("Code: 319", code=319)  # stored, then an error


def parked_ids(spool):
    return {r[0] for item in spool.pending() for r in item.rows}


def run(seed, replicated, feeds_views, tmp_path):
    rng = random.Random(seed)
    t = [1_000_000.0]
    clock = lambda: t[0]  # noqa: E731

    def sleep(s):
        t[0] += s

    srv = Server(rng, replicated, clock)
    cap = rng.choice([256 * 1024 * 1024, 700])
    spool = rz.Spool(
        tmp_path / f"s{seed}-{replicated}-{feeds_views}", max_bytes=cap, clock=clock
    )
    ids = list(range(rng.randint(3, 14)))
    pol = rz.Policy(
        batch_rows=rng.randint(1, 4), deadline_s=rng.choice([0, 3, 9]), backoff_s=1
    )
    returned = []
    for call in range(1 + (replicated and rng.random() < 0.5)):
        if call:
            t[0] += 60  # a rerun of the same load, inside the dedup window
        try:
            rz.resilient_insert(
                srv, "t", [[i] for i in ids], ["id"], scope=f"seed{seed}", policy=pol,
                replicated=replicated, spool=spool, sleep=sleep, clock=clock,
                feeds_views=feeds_views, log=lambda m: None,
            )  # fmt: skip
            returned = ids
        except (rz.WriteAmbiguous, rz.WriteUnavailable, rz.SpoolFull):
            returned = []  # loud: the caller knows
    return srv, spool, returned


@pytest.mark.parametrize("feeds_views", [False, True])
@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("seed", [11, 23, *range(60)])
def test_nothing_is_doubled_and_nothing_is_lost_silently(
    tmp_path, seed, replicated, feeds_views
):
    srv, spool, returned = run(seed, replicated, feeds_views, tmp_path)
    assert all(k == 1 for k in srv.n.values()), f"stored twice: {srv.n}"
    parked = parked_ids(spool)
    lost = [i for i in returned if srv.n[i] == 0 and i not in parked]
    assert not lost, f"lost silently: {lost}"
    if replicated:
        assert all(len(toks) == 1 for toks in srv.toks.values())
    else:
        assert srv.sends_after_doubt == 0
