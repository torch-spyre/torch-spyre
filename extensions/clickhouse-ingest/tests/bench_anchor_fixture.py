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

"""Helpers for the anchored-verdict tests: the schema stack on chdb, and a reference-from-run anchor over the synthetic history."""

import json
import uuid
from datetime import timedelta

import pytest
from bench_verdict_fixture import (
    ARCH,
    BACKEND,
    BENCHMARKS,
    COMPONENT,
    NS,
    START,
    bench_id,
    run_uuid,
)
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests",
    "20-artifacts",
    "30-benchmarks",
    "49-result-uploads",
    "50-artifact-views",
    "60-benchmark-views",
    "62-benchmark-verdicts",
    "81-bench-environments",
    "82-run-placements",
    "83-result-invalidations",
    "84-bench-views",
    "85-benchmark-verdict-views",
    "86-bench-references",
    "88-bench-anchored-verdicts",
)
SHIFT = {n: 0.12 for n, *_ in BENCHMARKS}
KEY = ("op-alpha", "kernel_mean_ms")
REF_KEY = "perf.synth.anchor"
NIGHTS = 105  # nights 0-44 normal, 45-104 shifted by +12 %


class Db:
    def __init__(self, nulls=False):
        self.s = session.Session()
        if nulls:
            self.s.query("SET join_use_nulls = 1")
        for name in FILES:
            if name in getattr(self, "skip", ()):
                continue
            path = SCHEMA_DIR / f"{name}.sql"
            for o in SchemaApplier.objects(path, path.read_text()):
                if o.kind != "refresh":
                    self.s.query(o.sql)

    def rows(self, sql):
        raw = self.s.query(sql, "JSONEachRow").data()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]

    def insert(self, table, *rows):
        body = "\n".join(json.dumps(r) for r in rows)
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")


def stamp(n):
    return (START + timedelta(days=n)).strftime("%Y-%m-%d %H:%M:%S")


def ref_id(n):
    return str(uuid.uuid5(NS, f"ref-artifact:{n}"))


def ref_run(n):
    return str(uuid.uuid5(NS, f"ref-run:{n}"))


def accept(
    f,
    night,
    n,
    *,
    key=REF_KEY,
    source_job="",
    src_run=None,
    anchor="1",
    tag="ref-current",
    tag_ts=None,
    current=True,
    state="published",
    only=None,
    carry=None,
):
    """Reference-from-run: the samples of history night `night` become anchor version `n`.

    The value rows are ref_only copies of that run's rows; `only` limits the benchmarks copied. `carry` maps a
    benchmark name to an earlier night whose values are carried into this version with that night as their own time."""
    d = f.db
    src = d.rows(
        "SELECT benchmark_id, component, backend, measurements FROM benchmark_runs "
        f"WHERE run_id = toUUID('{src_run or run_uuid(night)}')"
    )
    byb = {}
    for r in src:
        for k, v in r["measurements"].items():
            byb.setdefault((r["benchmark_id"], k), []).extend(v)
    rows = []
    for name, cnight in (carry or {}).items():
        for r in d.rows(
            "SELECT benchmark_id, measurements FROM benchmark_runs "
            f"WHERE run_id = toUUID('{run_uuid(cnight)}') AND benchmark_id = toUUID('{bench_id(name)}')"
        ):
            for k, v in r["measurements"].items():
                rows.append(
                    {
                        "ts": stamp(cnight),
                        "run_id": ref_run(n),
                        "benchmark_id": r["benchmark_id"],
                        "component": COMPONENT,
                        "backend": BACKEND,
                        "measurements": {k: v},
                        "iterations": len(v),
                        "props": {
                            "run.ref_only": "1",
                            "run.record_type": "reference",
                            "ref.key": key,
                            "anchor.as_of": stamp(cnight),
                        },
                    }
                )
    for (bid, metric), samples in byb.items():
        if only and bid not in {bench_id(x) for x in only}:
            continue
        if carry and bid in {bench_id(x) for x in carry}:
            continue
        rows.append(
            {
                "ts": stamp(night),
                "run_id": ref_run(n),
                "benchmark_id": bid,
                "component": COMPONENT,
                "backend": BACKEND,
                "measurements": {metric: samples},
                "iterations": len(samples),
                "props": {
                    "run.ref_only": "1",
                    "run.record_type": "reference",
                    "ref.key": key,
                    **({"anchor.source_job": source_job} if source_job else {}),
                },
            }
        )
    d.insert("benchmark_runs", *rows)
    acc_ts = (START + timedelta(days=night, hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    props = {
        "ref.key": key,
        "ref.version": str(n),
        "ref.class": "perf",
        "ref.label": "Synthetic",
        "ref.origin": "generated",
        "ref.approved_by": "amy@example.invalid",
        "ref.approved_at": acc_ts,
        "ref.anchor.from_run_id": run_uuid(night),
        "ref.anchor.as_of": stamp(night),
    }
    if anchor:
        props["ref.anchor"] = anchor
    d.insert("artifacts", {"ts": acc_ts, "artifact_id": ref_id(n), "component": "reference", "arch": ARCH, "kind": "generic",
                           "artifact_name": key, "origin": "built", "props": props})  # fmt: skip
    d.insert("result_uploads", {"ts": acc_ts + ".000", "upload_id": str(uuid.uuid5(NS, f"upload:{n}")), "state": state,
                                "actor": "amy@example.invalid", "artifact_id": ref_id(n), "run_id": ref_run(n), "format": "benchmarks",
                                "result_kind": "performance", "test_type": "perf", "arch": ARCH, "filename": "r.json",
                                "size_bytes": 1, "content_sha256": "0", "bundle_url": ""})  # fmt: skip
    if current:
        d.insert("artifact_tags", {"ts": tag_ts or acc_ts, "tag": tag, "tag_family": "reference", "artifact_id": ref_id(n),
                                   "refs": [], "published_refs": [], "props": {"ref.key": key}})  # fmt: skip


def history(f, shift_from=45, nights=NIGHTS, slow=None):
    f.register_benchmarks()
    return [
        f.run(slow=(slow or SHIFT) if n >= shift_from else None) for n in range(nights)
    ]
