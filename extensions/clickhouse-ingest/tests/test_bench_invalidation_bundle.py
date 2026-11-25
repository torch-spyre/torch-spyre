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

"""A bench-invalidation bundle sealed by the offline writer is written by the invalidation writer, on the
admin kinds' channel rules."""

import json
from types import SimpleNamespace

import pytest
from bench_chdb import ChdbClient, Db, place
from bench_verdict_fixture import Fixture
from spyre_clickhouse_ingest import bundle, offline
from spyre_clickhouse_ingest import client as client_mod

pytest.importorskip("chdb.session")

INV = "018f0000-0000-7000-8000-000000000001"
EVENT = {
    "invalidation_id": INV,
    "scope_kind": "node",
    "scope": {"node": "n1"},
    "state": "invalidated",
    "reason_code": "faulty_hardware",
    "reason": "card fault",
}


@pytest.fixture
def client(monkeypatch):
    d = Db()
    f = Fixture(d)
    f.history(22)
    for n in range(22):
        place(d, n, lambda n, rep: f"2026-10-{n + 1:02d} 02:{rep:02d}:00")
    c = ChdbClient(d)
    monkeypatch.setattr(client_mod.ClickHouse, "connect", staticmethod(lambda **k: c))
    monkeypatch.setattr(
        client_mod.ClickHouseEnv, "target_database", staticmethod(lambda: "default")
    )
    yield c
    d.s.close()


def sealed(tmp_path, events, name="a.tgz"):
    records = tmp_path / "events.json"
    records.write_text(json.dumps({"events": events}))
    args = offline.parser().parse_args(
        ["--offline", "--kind", "bench-invalidation", "--records-file", str(records),
         "--out", str(tmp_path / name)]
    )  # fmt: skip
    return offline.write(args)


def ingest(path, trust="lab", **over):
    args = SimpleNamespace(
        from_bundle=str(path), dry_run=False, uploader="amy@example.com", bundle_prop=[],
        channel_trust=trust, expect_key="", trusted_job_prefix=[], append_to_run=False,
    )  # fmt: skip
    args.__dict__.update(over)
    with offline.opened(path) as root:
        return bundle._ingest(root, args)


def stored(c):
    rows = c.query("SELECT state, actor FROM result_invalidations").result_rows
    return [tuple(r) for r in rows]


def test_a_sealed_bundle_is_written_by_the_invalidation_writer(client, tmp_path):
    out = sealed(tmp_path, [EVENT])
    with offline.opened(out) as root:
        assert [p.name for p in (root / "results").iterdir()] == ["invalidation.json"]
    assert ingest(out, dry_run=True)["status"] == "would-ingest"
    assert stored(client) == []
    report = ingest(out)
    assert (report["status"], report["trust"]) == ("ingested", "lab")
    assert report["receipt"].startswith("RCPT-")
    assert stored(client) == [("invalidated", "amy@example.com")]
    assert ingest(out)["status"] == "duplicate"


@pytest.mark.parametrize(
    "over,message",
    [
        ({"trust": "external"}, "trust ci or lab, not external"),
        ({"trust": ""}, "trust ci or lab, not none"),
        ({"expect_key": "k"}, "not an inbox bundle"),
    ],
)
def test_the_admin_channel_rules_hold(client, tmp_path, over, message):
    with pytest.raises(offline.BundleError, match=message):
        ingest(sealed(tmp_path, [EVENT]), **over)
    assert stored(client) == []


def test_an_event_the_writer_would_refuse_is_refused_at_seal(tmp_path):
    with pytest.raises(offline.BundleError, match="reason_code"):
        sealed(tmp_path, [{**EVENT, "reason_code": "because"}])


def test_a_bundle_with_an_unknown_trust_label_is_rejected_not_crashed(client, tmp_path):
    out = sealed(tmp_path, [EVENT], "t")
    meta = json.loads((out / "bundle.json").read_text())
    (out / "bundle.json").write_text(json.dumps({**meta, "trust": "zzz"}))
    with pytest.raises(offline.BundleError):
        ingest(out)
    assert stored(client) == []


def test_a_window_is_an_event_field(tmp_path):
    scoped = {**EVENT, "scope": {"node": "n1", "window_from": "2026-11-01T00:00:00Z"}}
    with pytest.raises(offline.BundleError, match="window_from"):
        sealed(tmp_path, [scoped])
    windowed = {**EVENT, "window_from": "2026-11-01T00:00:00Z"}
    assert sealed(tmp_path, [windowed], "w")
