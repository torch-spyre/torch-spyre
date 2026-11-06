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

"""Batched inserts: bounded requests, retry with backoff, and no blind resend of a kept batch."""

import pytest

from spyre_clickhouse_ingest.batching import (
    BatchAmbiguous,
    BatchFailed,
    chunks,
    insert_chunks,
)

ROWS = [[i] for i in range(12)]


class Client:
    def __init__(self, fail_at=(), keep_failed=False):
        self.calls, self.stored, self.fail_at, self.keep = (
            [],
            0,
            set(fail_at),
            keep_failed,
        )

    def insert(self, table, rows, column_names=None, database=None):
        self.calls.append(len(rows))
        if len(self.calls) in self.fail_at:
            if self.keep:
                self.stored += len(rows)
            raise TimeoutError("The write operation timed out")
        self.stored += len(rows)


def run(client, **kw):
    return insert_chunks(
        client,
        "t",
        ROWS,
        ["a"],
        size=5,
        backoff_s=0,
        sleep=lambda s: None,
        log=lambda m: None,
        **kw,
    )


def test_chunks_bound_every_request():
    assert [len(c) for c in chunks(ROWS, 5)] == [5, 5, 2]


def test_all_rows_stored_in_batches():
    c = Client()
    assert run(c) == 12
    assert c.calls == [5, 5, 2]


def test_transient_failure_is_retried_and_rows_not_duplicated():
    c = Client(fail_at=[2])
    assert run(c) == 12
    assert c.stored == 12
    assert c.calls == [5, 5, 5, 2]


def test_gives_up_after_attempts_and_reports_what_landed():
    c = Client(fail_at=range(2, 100))
    with pytest.raises(BatchFailed) as e:
        run(c, attempts=3)
    assert e.value.landed == 5 and "batch 2/3" in str(e.value)
    assert not isinstance(e.value, BatchAmbiguous)


def test_failed_batch_the_server_kept_is_not_resent():
    c = Client(fail_at=[2], keep_failed=True)
    with pytest.raises(BatchAmbiguous) as e:
        run(c, already_landed=lambda: c.stored)
    assert e.value.landed == 10
    assert c.calls == [5, 5]


def test_failed_batch_not_kept_is_resent_when_probe_says_clean():
    c = Client(fail_at=[2])
    assert run(c, already_landed=lambda: c.stored) == 12
    assert c.stored == 12


def test_empty_is_a_noop():
    c = Client()
    assert insert_chunks(c, "t", [], ["a"]) == 0 and c.calls == []
