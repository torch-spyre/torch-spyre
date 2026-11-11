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
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

from spyre_clickhouse_ingest import batching
from spyre_clickhouse_ingest.batching import (
    BatchAmbiguous,
    BatchFailed,
    chunks,
    insert_chunks,
    retriable,
)

ROWS = [[i] for i in range(12)]


class Client:
    def __init__(self, fail_at=(), keep_failed=False, error=None):
        self.calls, self.stored, self.fail_at, self.keep = (
            [],
            0,
            set(fail_at),
            keep_failed,
        )
        self.error = error or OperationalError("Read timed out")

    def insert(self, table, rows, column_names=None, database=None):
        self.calls.append(len(rows))
        if len(self.calls) in self.fail_at:
            if self.keep:
                self.stored += len(rows)
            raise self.error
        self.stored += len(rows)


def run(client, **kw):
    kw.setdefault("already_landed", lambda: client.stored)
    return insert_chunks(
        client,
        "t",
        ROWS,
        ["a"],
        size=5,
        backoff_s=0,
        settle_s=0,
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
        run(c)
    assert e.value.landed == 5
    assert c.calls == [5, 5]


def test_failed_batch_not_kept_is_resent_when_probe_says_clean():
    c = Client(fail_at=[2])
    assert run(c) == 12
    assert c.stored == 12


def test_empty_is_a_noop():
    c = Client()
    assert insert_chunks(c, "t", [], ["a"]) == 0 and c.calls == []


def test_without_a_probe_a_failed_batch_is_sent_once():
    c = Client(fail_at=[2])
    with pytest.raises(BatchFailed):
        run(c, already_landed=None)
    assert c.calls == [5, 5]


def test_without_a_probe_an_idempotent_table_is_retried():
    c = Client(fail_at=[2])
    assert run(c, already_landed=None, idempotent=True) == 12
    assert c.calls == [5, 5, 5, 2]


def test_probe_failing_before_the_first_batch_sends_nothing():
    c = Client()

    def down():
        raise OperationalError("connection refused")

    with pytest.raises(BatchFailed) as e:
        run(c, already_landed=down)
    assert c.calls == [] and e.value.landed == 0
    assert not isinstance(e.value, BatchAmbiguous)


def test_probe_failing_after_an_error_is_ambiguous_and_keeps_the_landed_count():
    c = Client(fail_at=[2])
    n = {"calls": 0}

    def probe():
        n["calls"] += 1
        if n["calls"] > 1:
            raise OperationalError("connection refused")
        return c.stored

    with pytest.raises(BatchAmbiguous) as e:
        run(c, already_landed=probe)
    assert e.value.landed == 5 and "probe failed" in str(e.value)
    assert c.calls == [5, 5]


@pytest.mark.parametrize("delta", [-3, 1, 7])
def test_any_count_other_than_expected_is_ambiguous(delta):
    c = Client(fail_at=[2])
    n = {"calls": 0}

    def probe():
        n["calls"] += 1
        return c.stored + (delta if n["calls"] > 1 else 0)

    with pytest.raises(BatchAmbiguous):
        run(c, already_landed=probe)
    assert c.calls == [5, 5]


def test_a_batch_that_commits_during_the_settle_delay_is_caught():
    c = Client(fail_at=[2])
    waits = []
    state = {"late": 0}

    def probe():
        return c.stored + state["late"]

    def settle(sec):
        waits.append(sec)
        state["late"] = 5

    with pytest.raises(BatchAmbiguous):
        insert_chunks(
            c,
            "t",
            ROWS,
            ["a"],
            size=5,
            backoff_s=0,
            settle_s=2,
            already_landed=probe,
            sleep=settle,
            log=lambda m: None,
        )
    assert waits == [2] and c.calls == [5, 5]


def test_sleep_is_resolved_at_call_time(monkeypatch):
    slept = []
    monkeypatch.setattr(batching.time, "sleep", slept.append)
    c = Client(fail_at=[2])
    insert_chunks(
        c, "t", ROWS, ["a"], size=5, backoff_s=3, settle_s=0,
        already_landed=lambda: c.stored, log=lambda m: None,
    )  # fmt: skip
    assert slept == [3]


@pytest.mark.parametrize(
    "err,expected",
    [
        (OperationalError("Read timed out"), True),
        (TimeoutError("write timed out"), True),
        (ConnectionResetError("reset"), True),
        (
            DatabaseError("HTTP driver received HTTP status 503, server response: x"),
            True,
        ),
        (
            DatabaseError(
                "HTTP status 500, server response: Code: 999. DB::Exception: boom"
            ),
            True,
        ),
        (
            DatabaseError(
                "HTTP status 403, server response: Code: 497. DB::Exception: x"
            ),
            False,
        ),
        (
            DatabaseError(
                "HTTP status 400, server response: Code: 469. DB::Exception: x"
            ),
            False,
        ),
        (
            DatabaseError(
                "HTTP status 500, server response: Code: 47. DB::Exception: x"
            ),
            False,
        ),
        (ValueError("bad row"), False),
    ],
)
def test_only_transient_errors_are_retriable(err, expected):
    assert retriable(err) is expected


def test_a_deterministic_error_fails_at_once_without_backoff():
    c = Client(
        fail_at=range(1, 99),
        error=DatabaseError(
            "HTTP status 403, server response: Code: 497. DB::Exception"
        ),
    )
    slept = []
    with pytest.raises(BatchFailed) as e:
        insert_chunks(
            c, "t", ROWS, ["a"], size=5, backoff_s=1, settle_s=0,
            already_landed=lambda: c.stored, sleep=slept.append, log=lambda m: None,
        )  # fmt: skip
    assert c.calls == [5] and slept == [] and "1 attempt" in str(e.value)


class Probe(Client):
    """A client that answers the scope queries the way ClickHouse would and records them."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.queries = []

    def query(self, sql, parameters=None):
        from types import SimpleNamespace

        self.queries.append((sql, dict(parameters or {})))
        return SimpleNamespace(result_rows=[[self.stored]])


def _case(i):
    from datetime import datetime

    return {
        "case_id": f"c{i}", "classname": "k", "name": f"n{i}", "op_name": "",
        "dtype": "", "status": "passed", "duration_s": 0.1, "fail_message": "",
        "triggered_at": datetime(2026, 1, 1), "properties": [("p", "v")],
    }  # fmt: skip


def test_insert_cases_probes_its_own_run_and_workflow():
    from spyre_clickhouse_ingest.results import insert_cases

    c = Probe(fail_at=[1], keep_failed=True)
    with pytest.raises(BatchAmbiguous):
        insert_cases(c, "run-1", [_case(i) for i in range(3)], workflow="wf-a")
    sql, params = c.queries[-1]
    assert "FROM test_cases" in sql and params == {"r": "run-1", "w": "wf-a"}


def test_insert_properties_probes_its_own_run():
    from spyre_clickhouse_ingest.results import insert_properties

    c = Probe(fail_at=[1], keep_failed=True)
    with pytest.raises(BatchAmbiguous):
        insert_properties(c, "run-9", [_case(i) for i in range(3)])
    sql, params = c.queries[-1]
    assert "FROM run_properties" in sql and params == {"r": "run-9"}


def test_table_insert_passes_the_scope_and_idempotence_through():
    from spyre_clickhouse_ingest import schema

    rows = [{"a": 1}]
    seen = {}

    def spy(client, table, ordered, cols, **kw):
        seen.update(kw)
        return len(ordered)

    probe = lambda: 3  # noqa: E731
    from unittest import mock

    as_list = classmethod(lambda cls, r: list(r.values()))
    with (
        mock.patch.object(schema, "insert_chunks", spy),
        mock.patch.object(schema.TestCaseRuns, "row", as_list),
        mock.patch.object(schema.PipelineRuns, "row", as_list),
    ):
        schema.TestCaseRuns.insert(object(), rows, landed=probe)
        assert seen["already_landed"] is probe and seen["idempotent"] is False
        schema.PipelineRuns.insert(object(), rows)
        assert seen["already_landed"] is None and seen["idempotent"] is True


def test_every_replacing_merge_tree_model_is_marked_idempotent():
    import regex as re
    from pathlib import Path

    from spyre_clickhouse_ingest import schema

    replacing = set()
    for p in sorted(Path("schema").glob("*.sql")):
        # Each CREATE TABLE is followed by its own ENGINE line; comments hold stray ';'.
        for m in re.finditer(
            r"CREATE TABLE IF NOT EXISTS (\w+)\b(?:(?!CREATE TABLE).)*?\nENGINE\s*=\s*(\w+)",
            p.read_text(),
            re.S,
        ):
            if m.group(2) == "ReplacingMergeTree":
                replacing.add(m.group(1))
    models = {
        c.name: c
        for c in vars(schema).values()
        if isinstance(c, type) and issubclass(c, schema.Table) and c.name
    }
    for name, cls in models.items():
        assert cls.idempotent == (name in replacing), name
