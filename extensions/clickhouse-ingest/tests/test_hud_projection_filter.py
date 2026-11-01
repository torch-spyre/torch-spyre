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

"""The HUD projection takes measured model-level spyre-inference rows only, and migration 023
leaves the MV identical to its schema file."""

import pytest
import regex as re
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

MIGRATION = SCHEMA_DIR / "migrations" / "023_hud_mv_measured_model_rows.sql"


def _mv(sql: str) -> str:
    m = re.search(
        r"CREATE MATERIALIZED VIEW (?:IF NOT EXISTS )?oss_ci_benchmark_v3_mv\b.*?;",
        sql,
        re.S,
    )
    assert m
    return re.sub(r"\s+", " ", m.group(0).replace("IF NOT EXISTS ", ""))


def test_the_migration_recreates_the_mv_as_the_schema_file_defines_it():
    schema = (SCHEMA_DIR / "70-vllm-hud-projection.sql").read_text()
    assert _mv(MIGRATION.read_text()) == _mv(schema)


@pytest.mark.parametrize(
    "backend, record_type, source_file, projected",
    [
        ("spyre", "model", "vllm-benchmark/1/serve.json", True),
        ("whatif", "model", "vllm-benchmark/1/serve.json", False),
        ("roofline", "model", "vllm-benchmark/1/serve.json", False),
        ("spyre", "kernel", "vllm-benchmark/1/serve.json", False),
        ("spyre", "step", "vllm-benchmark/1/serve.json", False),
        ("spyre", "model", "optbench-c1/granite/A.json", False),
    ],
)
def test_only_measured_model_rows_reach_the_hud(
    backend, record_type, source_file, projected
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
    s.query(
        "INSERT INTO benchmark_runs (run_id, benchmark_id, component, backend, measurements, props) "
        f"VALUES ('{rid}', '{bid}', 'spyre-inference', '{backend}', map('ttft_ms', [1.0]), "
        f"map('source_file', '{source_file}', 'arch', 'x86_64'))"
    )
    n = s.query("SELECT count() FROM oss_ci_benchmark_v3", "CSV").data()
    n = n.decode() if isinstance(n, bytes) else n
    assert int(n.strip()) == (1 if projected else 0)
