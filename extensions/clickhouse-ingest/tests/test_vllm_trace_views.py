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

"""vllm-trace rows on real chdb: they land, link to their artifact, and are not perf verdicts."""

import json
import regex as re
import uuid
from pathlib import Path

import pytest
from spyre_clickhouse_ingest import vllm_trace
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier
from spyre_clickhouse_ingest.writer import insert_benchmarks

session = pytest.importorskip("chdb.session")
FLEET = Path(__file__).parent / "data" / "vllm_trace" / "fleet"
AID = "aaaaaaaa-aaaa-5aaa-8aaa-aaaaaaaaaaaa"
FILES = ("10-functional-tests.sql", "20-artifacts.sql", "30-benchmarks.sql", "46-capabilities.sql",
         "49-result-uploads.sql", "50-artifact-views.sql", "60-benchmark-views.sql",
         "62-benchmark-verdicts.sql")  # fmt: skip


def fleet_ctx():
    return vllm_trace.load(FLEET / "out" / "w1", FLEET / "rep", "w1")


class Chdb:
    """The slice of clickhouse_connect's client the benchmark writer uses, over a chdb session."""

    def __init__(self):
        self.s = session.Session()

    def run(self, sql, fmt="TSV"):
        return str(self.s.query(sql, fmt)).rstrip("\n")

    command = run

    @staticmethod
    def _bind(sql, parameters):
        def lit(m):
            v = (parameters or {})[m.group(1)]
            if isinstance(v, list):
                return "[" + ",".join(f"'{x}'" for x in v) + "]"
            return f"'{v}'"

        return re.sub(r"\{(\w+):[^}]+\}", lit, sql)

    def query(self, sql, parameters=None):
        out = self.run(self._bind(sql, parameters), "JSONCompact")
        rows = json.loads(out)["data"] if out else []

        class R:
            result_rows = [tuple(r) for r in rows]

        return R()

    def insert(self, table, rows, column_names=None, database=None):
        body = "\n".join(
            json.dumps(dict(zip(column_names, r)), default=str) for r in rows
        )
        self.run(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    def rows(self, sql):
        return [json.loads(x) for x in self.run(sql, "JSONEachRow").splitlines()]


@pytest.fixture
def db():
    c = Chdb()
    for name in FILES:
        text = (SCHEMA_DIR / name).read_text()
        for stmt in SchemaApplier.statements(text):
            if (
                "REFRESH" in stmt.upper().split("AS", 1)[0]
                or name == "20-artifacts.sql"
                and "CREATE VIEW" in stmt
            ):
                continue
            c.run(stmt)
    yield c
    c.s.close()


def _land(db, ctx, run_key):
    rid = str(uuid.uuid5(uuid.NAMESPACE_URL, run_key))
    insert_benchmarks(db, "", vllm_trace.COMPONENT, rid, vllm_trace.entries(ctx),
                      report_kind=vllm_trace.REPORT_KIND)  # fmt: skip
    db.run(
        "INSERT INTO artifact_results (artifact_id, run_id, result_kind, test_type, state, arch, "
        f"duration_s, props) VALUES ('{AID}', '{rid}', 'performance', 'perf', 'passed', 'x86_64', 1, {{}})"
    )
    return rid


def test_rows_land_and_link_to_the_artifact(db):
    rid = _land(db, fleet_ctx(), "w1")
    got = db.rows(
        "SELECT record_type, count() n, any(toString(artifact_id)) a FROM v_benchmark_results_enriched "
        f"WHERE run_id = '{rid}' GROUP BY record_type ORDER BY record_type"
    )
    assert {r["record_type"]: r["n"] for r in got} == {
        "compile_pass": 3, "trace_kernel": 4, "trace_phase": 6, "trace_step": 2,
    }  # fmt: skip
    assert {r["a"] for r in got} == {AID}
    props = db.rows(f"SELECT props FROM benchmark_runs WHERE run_id = '{rid}' LIMIT 1")[
        0
    ]["props"]
    assert (
        props["report_kind"] == "vllm-trace"
        and props["artifact_id12"] == "aaaaaaaaaaaa"
    )


def test_trace_rows_are_not_perf_verdicts(db):
    # 12 runs of the same trace workload would carry a baseline for every metric.
    for i in range(12):
        _land(db, fleet_ctx(), f"w1-{i}")
    n = "SELECT count() n FROM v_benchmark_metric_verdicts"
    assert db.rows(n)[0]["n"] == 0
    assert db.rows("SELECT count() n FROM v_benchmark_gate")[0]["n"] == 0
    # The guard is what excludes them: without it the same rows reach the verdict view.
    view = (SCHEMA_DIR / "62-benchmark-verdicts.sql").read_text()
    guard = " AND record_type NOT IN ('trace_phase', 'trace_step', 'trace_kernel', 'compile_pass')"
    assert guard in view
    db.run("DROP VIEW v_benchmark_gate")
    db.run("DROP VIEW v_benchmark_metric_verdicts")
    for stmt in SchemaApplier.statements(view.replace(guard, "")):
        if stmt.lstrip().startswith(
            "CREATE VIEW IF NOT EXISTS v_benchmark_metric_verdicts"
        ):
            db.run(stmt)
    assert db.rows(n)[0]["n"] > 0


def test_a_vllm_series_is_still_gated(db):
    ctx = fleet_ctx()
    entry = {
        "name": "serve_x", "tags": [], "backend": "spyre",
        "props": {"record_type": "model", "run_mode": "serve"},
        "measurements": {"mean_ttft_ms": [100.0, 101.0, 99.0]}, "iterations": 3,
        "run_props": {"arch": "x86_64"}, "disc": {"record_type": "model", "run_mode": "serve"},
        "disc_keys": ("record_type", "run_mode"),
    }  # fmt: skip
    for i in range(12):
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"v{i}"))
        insert_benchmarks(db, "", "spyre-inference", rid, [entry], report_kind="vllm")
        db.run(
            "INSERT INTO artifact_results (artifact_id, run_id, result_kind, test_type, state, arch, "
            f"duration_s, props) VALUES ('{AID}', '{rid}', 'performance', 'perf', 'passed', 'x86_64', 1, {{}})"
        )
    _land(db, ctx, "w1")
    kinds = db.rows("SELECT DISTINCT metric FROM v_benchmark_metric_verdicts")
    assert [r["metric"] for r in kinds] == ["mean_ttft_ms"]
