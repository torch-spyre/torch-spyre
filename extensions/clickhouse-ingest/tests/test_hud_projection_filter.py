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

"""The HUD projection takes measured model-level spyre-inference rows only (never bench-engine rows), and the latest
migration of the MV leaves it identical to its schema file."""

import pytest
import regex as re
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

MIGRATION = SCHEMA_DIR / "migrations" / "029_hud_mv_skips_bench_rows.sql"


def _mv(sql: str) -> str:
    """The MV statement with comments stripped (a ';' inside a comment ends no statement)."""
    (stmt,) = [
        x
        for x in SchemaApplier.statements(sql)
        if re.match(
            r"CREATE MATERIALIZED VIEW (?:IF NOT EXISTS )?oss_ci_benchmark_v3_mv\b", x
        )
    ]
    return re.sub(r"\s+", " ", stmt.replace("IF NOT EXISTS ", ""))


def test_the_migration_recreates_the_mv_as_the_schema_file_defines_it():
    schema = (SCHEMA_DIR / "70-vllm-hud-projection.sql").read_text()
    assert _mv(MIGRATION.read_text()) == _mv(schema)


@pytest.mark.parametrize(
    "backend, record_type, source_file, projected, report_kind",
    [
        ("spyre", "model", "vllm-benchmark/1/serve.json", True, "vllm"),
        ("spyre", "model", "vllm-benchmark/1/serve.json", True, ""),
        ("spyre", "model", "vllm-benchmark/1/serve.json", True, "-"),
        ("spyre", "model", "serve.json", False, "bench"),
        ("spyre", "model", "ci-job/serve.json", False, "bench"),
        ("whatif", "model", "vllm-benchmark/1/serve.json", False, "vllm"),
        ("roofline", "model", "vllm-benchmark/1/serve.json", False, "vllm"),
        ("spyre", "kernel", "vllm-benchmark/1/serve.json", False, "vllm"),
        ("spyre", "step", "vllm-benchmark/1/serve.json", False, "vllm"),
        ("spyre", "model", "optbench-c1/granite/A.json", False, "vllm"),
    ],
)
def test_only_measured_model_rows_reach_the_hud(
    backend, record_type, source_file, projected, report_kind
):
    session = pytest.importorskip("chdb.session")

    s = session.Session()
    # 70's by-tag MV reads the artifact views, so their schema files load first.
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "30-benchmarks.sql",
        "49-result-uploads.sql",
        "50-artifact-views.sql",
        "52-cross-views.sql",
        "60-benchmark-views.sql",
        "70-vllm-hud-projection.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            s.query(stmt)
    bid, rid = (
        "e3403447-8dfa-5692-a90f-97155fbad195",
        "21e0c6d2-9463-58c6-b93a-f6c6d5f758ed",
    )
    s.query(
        "INSERT INTO benchmarks (benchmark_id, component, name, props) VALUES "
        f"('{bid}', 'spyre-inference', 'serve_x', map('record_type', '{record_type}', 'model', 'm'))"
    )
    kind_pair = f", 'report_kind', '{report_kind}'" if report_kind != "-" else ""
    s.query(
        "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements, props) "
        f"VALUES ('{rid}', '{bid}', 'spyre-inference', '{backend}', map('ttft_ms', [1.0]), "
        f"map('source_file', '{source_file}', 'arch', 'x86_64'{kind_pair}))"
    )
    n = s.query("SELECT count() FROM oss_ci_benchmark_v3", "CSV").data()
    n = n.decode() if isinstance(n, bytes) else n
    assert int(n.strip()) == (1 if projected else 0)


def test_the_mv_migrations_can_be_rerun():
    # Rerunning 023 or 025 reinstalls the MV without the later filter; 029 is the way back.
    for n in (
        "023_hud_mv_measured_model_rows.sql",
        "025_hud_mv_skips_reference_rows.sql",
        "029_hud_mv_skips_bench_rows.sql",
    ):
        assert SchemaApplier.RERUNNABLE.search(
            (SCHEMA_DIR / "migrations" / n).read_text()
        )


# Row shapes each BENCH_MODE leaves in benchmark_runs for spyre-inference (the only component the HUD takes).
# off: the legacy vLLM writer only. side-by-side: that row plus the engine's own `bench` bundle row (same
# benchmark_id, its own run). replace: the engine's results written through the legacy writer (XML / native mirror),
# so the rows are legacy-shaped; a native `bench` bundle in replace would be a `bench` row with no legacy twin.
MODE_ROWS = {
    "off": [("legacy", "vllm-benchmark/1/serve.json", "vllm")],
    "side-by-side": [
        ("legacy", "vllm-benchmark/1/serve.json", "vllm"),
        ("bench", "serve.json", "bench"),
    ],
    "replace": [("mirror", "vllm-benchmark/1/serve.json", "vllm")],
    "replace-native-bench": [("native", "serve.json", "bench")],
}
HUD_ROWS = {
    "off": {"legacy"},
    "side-by-side": {"legacy"},
    "replace": {"mirror"},
    "replace-native-bench": set(),
}


@pytest.mark.parametrize("mode", sorted(MODE_ROWS))
def test_the_hud_gets_one_row_per_run_in_every_bench_mode(mode):
    session = pytest.importorskip("chdb.session")
    import uuid

    s = session.Session()
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "30-benchmarks.sql",
        "49-result-uploads.sql",
        "50-artifact-views.sql",
        "52-cross-views.sql",
        "60-benchmark-views.sql",
        "70-vllm-hud-projection.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            s.query(stmt)
    bid = str(uuid.uuid4())
    s.query(
        "INSERT INTO benchmarks (benchmark_id, component, name, props) VALUES "
        f"('{bid}', 'spyre-inference', 'serve_x', map('record_type', 'model', 'model', 'm'))"
    )
    runs = {}
    for label, source_file, kind in MODE_ROWS[mode]:
        runs[label] = str(uuid.uuid4())
        s.query(
            "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements, props) "
            f"VALUES ('{runs[label]}', '{bid}', 'spyre-inference', 'spyre', map('ttft_ms', [1.0]), "
            f"map('source_file', '{source_file}', 'arch', 'x86_64', 'report_kind', '{kind}'))"
        )
    out = s.query("SELECT DISTINCT run_id FROM oss_ci_benchmark_v3", "CSV").data()
    out = out.decode() if isinstance(out, bytes) else out
    seen = {x.strip('"') for x in out.split()}
    assert {k for k, v in runs.items() if v in seen} == HUD_ROWS[mode]
