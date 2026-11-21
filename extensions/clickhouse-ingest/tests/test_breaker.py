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

"""The breaker layer around the merged `resilient_insert`, and what the spool now keeps."""

import datetime as dt
import decimal
import uuid

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError

from spyre_clickhouse_ingest.breaker import Breaker, Limits, guarded_insert
from spyre_clickhouse_ingest.resilience import (
    Policy,
    Spool,
    SpoolFull,
    WriteUnavailable,
)


def err(code):
    return DatabaseError(f"Code: {code}. x", code=code, name="X")


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Down:
    def insert(self, *a, **k):
        raise err(210)


class Up:
    def __init__(self):
        self.calls = []

    def insert(self, table, data, column_names=None, database=None, settings=None):
        self.calls.append(dict(settings or {}))


def run(client, spool=None, breaker=None, clock=None, **kw):
    clock = clock or Clock()
    return guarded_insert(
        client, "t", [[1]], ["a"], scope="s", replicated=True,
        policy=Policy(deadline_s=30, backoff_s=10, backoff_max_s=10),
        spool=spool, breaker=breaker or Breaker(), sleep=clock.sleep,
        clock=clock.monotonic, log=lambda m: None, **kw,
    )  # fmt: skip


@pytest.mark.parametrize(
    "field", ["breaker_s", "breaker_probe_s", "budget_s", "budget_window_s"]
)
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "5"])
def test_limits_must_be_finite_and_positive(field, bad):
    with pytest.raises(ValueError):
        Limits(**{field: bad})


def test_an_open_breaker_parks_every_batch_with_the_host_it_was_for(tmp_path):
    breaker, spool, clock = Breaker(), Spool(tmp_path / "sp"), Clock()
    run(Down(), spool, breaker, clock, host="ha")
    assert breaker.tripped
    rows = [[i] for i in range(7)]
    up = Up()
    out = guarded_insert(
        up, "t", rows, ["a"], scope="s2", replicated=True, host="ha",
        policy=Policy(batch_rows=3), spool=spool, breaker=breaker,
        clock=clock.monotonic, sleep=clock.sleep, log=lambda m: None,
    )  # fmt: skip
    assert up.calls == [] and out.spooled == 7 and out.stored == 0
    items = spool.items()
    assert len(items) == 4 and {i["host"] for i in items} == {"ha"}
    # only the first call's batch, which was sent, may be stored
    assert sum(i["held"] for i in items) == 1


def test_an_open_breaker_without_a_spool_raises_with_every_batch_unwritten():
    breaker, clock = Breaker(), Clock()
    with pytest.raises(WriteUnavailable):
        run(Down(), None, breaker, clock)
    with pytest.raises(WriteUnavailable) as e:
        run(Up(), None, breaker, clock)
    assert e.value.outcome.unwritten == [1]


def test_a_full_spool_stops_the_parking_of_an_open_breaker(tmp_path):
    breaker, clock = Breaker(), Clock()
    run(Down(), Spool(tmp_path / "sp"), breaker, clock)
    tiny = Spool(tmp_path / "tiny", max_bytes=10)
    with pytest.raises(SpoolFull) as e:
        run(Up(), tiny, breaker, clock)
    assert e.value.outcome.unwritten == [1]


def test_the_parked_file_records_whether_it_carried_a_token(tmp_path):
    spool = Spool(tmp_path / "sp")
    run(Down(), spool, Breaker(), dedup_window=0, host="h")
    (item,) = spool.pending()
    assert item.ambiguous and item.meta == {"host": "h", "token_sent": False}


def test_a_success_closes_the_breaker_and_a_failed_probe_reopens_it(tmp_path):
    breaker, spool, clock = Breaker(), Spool(tmp_path / "sp"), Clock()
    run(Down(), spool, breaker, clock)
    assert breaker.acquire(clock.t) == Breaker.OPEN
    clock.t += 61
    run(Down(), spool, breaker, clock)  # the probe fails
    assert breaker.tripped and breaker.acquire(clock.t) == Breaker.OPEN
    clock.t += 61
    assert run(Up(), spool, breaker, clock).stored == 1
    assert not breaker.tripped


def test_spool_rows_keep_their_types(tmp_path):
    row = [
        uuid.UUID("00000000-0000-4000-8000-000000000042"),
        dt.datetime(2026, 11, 13, 1, 2, 3, 456000),
        dt.date(2026, 11, 13),
        decimal.Decimal("12.3456"),
        b"\x00\xff",
        {"a": 1, "b": {"c": [1, 2]}},
        None,
        ["x", "y"],
    ]
    spool = Spool(tmp_path / "sp")
    spool.put("t", list("abcdefgh"), [row], "t-1")
    (item,) = spool.pending()
    assert item.rows == [row]


def test_a_spool_token_is_plain_characters_and_matched_whole(tmp_path):
    spool = Spool(tmp_path / "sp")
    with pytest.raises(ValueError):
        spool.put("t", ["a"], [[1]], "../evil")
    spool.put("t", ["a"], [[1]], "t-abc123")
    assert spool.drop("abc123") == 0 and spool.drop("t-abc123") == 1
