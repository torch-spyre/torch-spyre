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

"""v_bench_references on chdb: a key's newest tag row decides current / retired / superseded, and only published uploads are listed."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests",
    "20-artifacts",
    "30-benchmarks",
    "49-result-uploads",
    "50-artifact-views",
    "86-bench-references",
)
KEY = "perf.synth.decode.ref.latency"


def aid(n):
    return f"aaaaaaaa-0000-5000-8000-00000000000{n}"


def run(n):
    return f"bbbbbbbb-0000-5000-8000-00000000000{n}"


class Db:
    def __init__(self):
        self.s = session.Session()
        for name in FILES:
            for stmt in SchemaApplier.statements(
                (SCHEMA_DIR / f"{name}.sql").read_text()
            ):
                self.s.query(stmt)

    def insert(self, table, *rows):
        self.s.query(
            f"INSERT INTO {table} FORMAT JSONEachRow\n"
            + "\n".join(json.dumps(r) for r in rows)
        )

    def version(self, n, state="published", key=KEY, component="reference"):
        self.insert("artifacts", {"ts": f"2026-11-0{n} 00:00:00", "artifact_id": aid(n), "component": component, "arch": "x86_64",
                                  "kind": "generic", "artifact_name": key, "origin": "built",
                                  "props": {"ref.key": key, "ref.version": str(n), "ref.class": "perf", "ref.label": "Synthetic",
                                            "ref.comparability.hardware": "different"}})  # fmt: skip
        self.insert("result_uploads", {"ts": f"2026-11-0{n} 00:00:00.000", "upload_id": f"cccccccc-0000-7000-8000-00000000000{n}", "state": state,
                                       "actor": "amy@example.invalid", "artifact_id": aid(n), "run_id": run(n), "format": "benchmarks",
                                       "result_kind": "performance", "test_type": "perf", "arch": "x86_64", "filename": "r.json",
                                       "size_bytes": 1, "content_sha256": "0", "bundle_url": ""})  # fmt: skip

    def tag(self, n, tag, ts, key=KEY):
        self.insert("artifact_tags", {"ts": ts, "tag": tag, "tag_family": "reference", "artifact_id": aid(n), "refs": [],
                                      "published_refs": [], "props": {"ref.key": key}})  # fmt: skip

    def states(self):
        raw = self.s.query(
            "SELECT version, state, run_id, comparability FROM v_bench_references ORDER BY version",
            "JSONEachRow",
        ).data()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]


@pytest.fixture
def db():
    d = Db()
    yield d
    d.s.close()


def test_the_newest_tag_row_decides_the_state_of_every_version(db):
    db.version(1)
    db.version(2)
    db.tag(1, "ref-current", "2026-11-01 00:00:00")
    assert [(r["version"], r["state"]) for r in db.states()] == [
        ("1", "current"),
        ("2", "superseded"),
    ]
    db.tag(2, "ref-current", "2026-11-02 00:00:00")
    assert [(r["version"], r["state"]) for r in db.states()] == [
        ("1", "superseded"),
        ("2", "current"),
    ]
    db.tag(2, "ref-retired", "2026-11-03 00:00:00")
    assert [(r["version"], r["state"]) for r in db.states()] == [
        ("1", "superseded"),
        ("2", "retired"),
    ]
    db.tag(
        1, "ref-current", "2026-11-04 00:00:00"
    )  # rollback: the older version is current again
    assert [(r["version"], r["state"]) for r in db.states()] == [
        ("1", "current"),
        ("2", "superseded"),
    ]


def test_only_published_versions_are_listed_and_keys_are_independent(db):
    db.version(1, state="pending")
    db.version(2)
    db.version(3, key="perf.synth.other")
    db.tag(2, "ref-current", "2026-11-02 00:00:00")
    db.tag(3, "ref-current", "2026-11-03 00:00:00", key="perf.synth.other")
    rows = db.states()
    assert [(r["version"], r["state"]) for r in rows] == [
        ("2", "current"),
        ("3", "current"),
    ]
    assert rows[0]["run_id"] == run(2) and rows[0]["comparability"] == {
        "ref.comparability.hardware": "different"
    }


def test_a_reference_is_not_a_tag_resolution_row(db):
    db.version(1)
    db.tag(1, "ref-current", "2026-11-01 00:00:00")
    assert db.s.query("SELECT count() FROM v_tag_resolution", "CSV").data().strip() in (
        b"0",
        "0",
    )


def test_a_published_version_with_no_tag_row_is_untagged(db):
    db.version(1)
    assert [(r["version"], r["state"]) for r in db.states()] == [("1", "untagged")]
    db.tag(1, "ref-current", "2026-11-01 00:00:00")
    assert db.states()[0]["state"] == "current"
