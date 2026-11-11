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

"""The Artifactory spool tier against a fake service that implements MOVE like the real one."""

import io
import json
import threading
import urllib.error
import urllib.parse

import pytest

from spyre_clickhouse_ingest import spool as sp
from spyre_clickhouse_ingest.artifactory_spool import ArtifactorySpool


class Resp(io.BytesIO):
    def __init__(self, body=b"", status=200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeArtifactory:
    """PUT, GET, DELETE, list and an atomic MOVE: a second MOVE of a moved file answers 400."""

    def __init__(self):
        self.files, self.lock, self.down, self.requests = (
            {},
            threading.Lock(),
            False,
            [],
        )

    def __call__(self, req, timeout=None, context=None):
        if self.down:
            raise urllib.error.URLError("down")
        url, method = req.full_url, req.get_method()
        self.requests.append((method, url))
        assert req.get_header("Authorization", "").startswith("Bearer ")
        with self.lock:
            if "/api/storage/" in url:
                files = [
                    {
                        "uri": "/" + n.split("/", 2)[2],
                        "lastModified": "2026-11-07T00:00:00.000Z",
                    }
                    for n in self.files
                ]
                return Resp(json.dumps({"files": files}).encode())
            if "/api/move/" in url:
                src = urllib.parse.unquote(url.split("/api/move/")[1].split("?")[0])
                dst = urllib.parse.unquote(url.split("to=/")[1])
                if src not in self.files:
                    raise urllib.error.HTTPError(url, 400, "gone", {}, io.BytesIO(b""))
                self.files[dst] = self.files[
                    src
                ]  # a racing second MOVE can also succeed
                return Resp()
            key = urllib.parse.unquote(url.split("/artifactory/")[1])
            if method == "PUT":
                self.files[key] = req.data
                return Resp(status=201)
            if method == "GET":
                if key not in self.files:
                    raise urllib.error.HTTPError(url, 404, "nf", {}, io.BytesIO(b""))
                return Resp(self.files[key])
            if method == "DELETE":
                self.files.pop(key, None)
                return Resp(status=204)
        raise AssertionError(url)


def make(fake=None, **kw):
    fake = fake or FakeArtifactory()
    kw.setdefault("settle_s", 0.05)
    kw.setdefault("base", "https://artifactory.example.test/artifactory")
    kw.setdefault("repo", "spool-repo")
    return fake, ArtifactorySpool("tkn", opener=fake, **kw)


def test_put_list_get_delete_roundtrip():
    fake, a = make()
    a.put("w/tok1.jsonl", b"hello")
    assert [n for n, _ in a.list()] == ["w/tok1.jsonl"]
    assert a.get("w/tok1.jsonl") == b"hello"
    a.delete("w/tok1.jsonl")
    assert a.list() == []


def test_exactly_one_of_many_parallel_claims_wins_even_though_move_is_not_a_lock():
    fake, a = make()
    a.put("w/tok1.jsonl", b"x")
    wins, barrier = [], threading.Barrier(6)

    def go(i):
        barrier.wait()
        wins.append(a.claim("w/tok1.jsonl", f"o{i}"))

    ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len([w for w in wins if w]) == 1


def test_a_held_lock_blocks_a_second_claimer_until_the_lease_runs_out():
    now = [1000.0]
    fake, a = make(clock=lambda: now[0], lease_s=60)
    a.put("w/tok1.jsonl", b"x")
    assert a.claim("w/tok1.jsonl", "o1") == "w/tok1.jsonl"
    assert a.claim("w/tok1.jsonl", "o2") is None
    now[0] += 61
    assert a.claim("w/tok1.jsonl", "o2") == "w/tok1.jsonl"


def test_release_lets_another_writer_claim_it():
    fake, a = make()
    a.put("w/tok1.jsonl", b"x")
    a.claim("w/tok1.jsonl", "o1")
    a.release("w/tok1.jsonl")
    assert a.claim("w/tok1.jsonl", "o2") == "w/tok1.jsonl"


def test_reap_removes_only_expired_locks():
    now = [1000.0]
    fake, a = make(clock=lambda: now[0], lease_s=60)
    a.put("w/a.jsonl", b"x")
    a.put("w/b.jsonl", b"x")
    a.claim("w/a.jsonl", "dead")
    now[0] += 30
    a.claim("w/b.jsonl", "alive")
    now[0] += 40
    assert a.reap(lease_s=60) == 1
    assert a.claim("w/a.jsonl", "o9") == "w/a.jsonl"
    assert a.claim("w/b.jsonl", "o9") is None


def test_delete_removes_the_item_and_its_lock():
    fake, a = make()
    a.put("w/tok1.jsonl", b"x")
    a.claim("w/tok1.jsonl", "o1")
    a.delete("w/tok1.jsonl")
    assert fake.files == {}


def test_an_unreachable_service_raises_oserror_without_the_token_or_url():
    fake, a = make()
    fake.down = True
    with pytest.raises(OSError) as e:
        a.put("w/x.jsonl", b"x")
    assert "tkn" not in str(e.value) and "artifactory.example.test" not in str(e.value)


def test_a_missing_token_means_no_shared_tier():
    assert ArtifactorySpool.from_env({}) is None
    full = {
        "CH_WRITE_ARTIFACTORY_TOKEN": "t",
        "CH_WRITE_ARTIFACTORY_URL": "https://artifactory.example.test/artifactory",
        "CH_WRITE_ARTIFACTORY_REPO": "spool-repo",
    }
    assert ArtifactorySpool.from_env(full) is not None
    for missing in full:
        partial = {k: v for k, v in full.items() if k != missing}
        assert ArtifactorySpool.from_env(partial) is None, missing
    assert ArtifactorySpool.from_env({"ARTIFACTORY_TOKEN": "t"}) is None


def test_end_to_end_untrusted_never_touches_artifactory(tmp_path):
    fake, a = make()
    s = sp.TieredSpool(tmp_path, a, trusted=False)
    assert s.put_any("t", ["a"], [[1]], "tok-1") == "local"
    assert fake.requests == []


def test_end_to_end_shared_write_then_another_writer_replays_it(tmp_path):
    fake, a = make()
    w1 = sp.TieredSpool(tmp_path / "a", a, trusted=True, writer="w1")
    w2 = sp.TieredSpool(tmp_path / "b", a, trusted=True, writer="w2")
    assert w1.put_any("t", ["a"], [[1]], "tok-1") == "shared"
    sent = []

    class C:
        def insert(self, table, rows, column_names=None, database=None, settings=None):
            sent.append((table, rows, settings["insert_deduplication_token"]))

    res = w2.drain(
        C(), replicated=True, sleep=lambda s: None, log=lambda m: None, force=True
    )
    assert res.from_shared == 1 and sent == [("t", [[1]], "tok-1")] and fake.files == {}


def test_metrics_names():
    import tempfile

    s = sp.TieredSpool(tempfile.mkdtemp(), None, trusted=False)
    assert set(sp.metrics(s)) >= {
        "ch_spool_pending",
        "ch_spool_oldest_age_s",
        "ch_spool_bytes",
    }


def test_list_ages_are_utc_whatever_the_local_timezone(monkeypatch):
    import calendar
    import time

    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        fake, a = make()
        a.put("w/x.jsonl", b"1")
        now = calendar.timegm((2026, 11, 7, 1, 0, 0))  # 1h after lastModified
        a.clock = lambda: now
        assert a.list()[0][1] == 3600
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
