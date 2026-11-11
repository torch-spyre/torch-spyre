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

"""Seeded fault injection over write + spool + replay: no row is stored twice or lost."""

import random

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest import spool as sp
from spyre_clickhouse_ingest.resilience import (
    Policy,
    WriteAmbiguous,
    resilient_insert,
)

WINDOW = 3600.0
COLS = ["id"]


def err(code):
    return DatabaseError(f"Code: {code}", code=code)


class Server:
    """A table with injected faults. Replicated: a token dedups only inside the window."""

    def __init__(self, rng, replicated, clock):
        self.rng, self.replicated, self.clock = rng, replicated, clock
        self.rows, self.tokens, self.faults = [], {}, True

    def insert(self, table, data, column_names=None, database=None, settings=None):
        tok = (settings or {}).get("insert_deduplication_token")
        r = self.rng.random() if self.faults else 1.0
        if r < 0.15:
            raise err(285)  # refused before any write
        stored_before_fault = r < 0.45
        lost = r < 0.60
        if not (lost and not stored_before_fault):
            dup = (
                self.replicated
                and tok in self.tokens
                and self.clock() - self.tokens[tok] < WINDOW
            )
            if not dup:
                self.rows.extend(row[0] for row in data)
            if tok:
                self.tokens[tok] = self.clock()
        if r < 0.60:
            raise err(319)  # response lost; stored or not


def run_seed(seed, replicated, tmp_path):
    rng = random.Random(seed)
    t = [1_000_000.0]
    clock = lambda: t[0]  # noqa: E731

    def sleep(s):
        t[0] += s

    srv = Server(rng, replicated, clock)
    spool = sp.TieredSpool(
        tmp_path / f"s{seed}{replicated}", None, trusted=False,
        dedup_window_s=WINDOW, clock=clock, drain_max_items=rng.choice([2, 20]),
    )  # fmt: skip
    ids = list(range(rng.randint(3, 14)))
    rows = [[i] for i in ids]
    pol = Policy(
        batch_rows=rng.randint(1, 4), deadline_s=rng.choice([0, 3, 9]), backoff_s=1
    )
    try:
        resilient_insert(
            srv, "t", rows, COLS, scope=f"seed{seed}", policy=pol, replicated=replicated,
            spool=spool, sleep=sleep, clock=clock, log=lambda m: None,
        )  # fmt: skip
    except WriteAmbiguous:
        pass
    for _ in range(rng.randint(0, 4)):
        t[0] += rng.choice([1, 60, 5000])
        spool.drain(
            srv, replicated=replicated, policy=pol, sleep=sleep, log=lambda m: None
        )
    srv.faults = False
    t[0] += rng.choice([1, 5000])
    for _ in range(3):
        spool.drain(
            srv,
            replicated=replicated,
            policy=pol,
            sleep=sleep,
            force=True,
            log=lambda m: None,
        )
    spooled = {r[0] for item in spool.local.pending() for r in item.rows}
    return ids, srv.rows, spooled


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("seed", list(range(150)) + [11, 23])
def test_no_row_is_stored_twice_or_lost(tmp_path, seed, replicated):
    ids, stored, spooled = run_seed(seed, replicated, tmp_path)
    assert len(stored) == len(set(stored)), f"duplicated rows: {sorted(stored)}"
    # A row is stored, or still held in the spool for a person; a held row was never sent
    # twice, but a held row that WAS stored is also fine (the probe-less path cannot tell).
    lost = set(ids) - set(stored) - spooled
    assert not lost, f"lost rows: {sorted(lost)}"
