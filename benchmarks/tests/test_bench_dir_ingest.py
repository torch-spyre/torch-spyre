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

"""The engine's own results of a perf report, ingested as a connected CI bench-dir run on chdb."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytest.importorskip("chdb.session")
pytest.importorskip("spyre_bench")
pytest.importorskip("lxml")
sys.path.insert(0, str(ROOT / "extensions" / "clickhouse-ingest" / "tests"))
from test_bench_bundle import AID, db  # noqa: E402,F401


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmarks" / "tools" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_engine_results_are_stored_under_the_catalog_ids(tmp_path, db):  # noqa: F811
    from spyre_clickhouse_ingest.__main__ import main as cli

    ids = load("check_identity_ids")
    cases = ids.declared()
    done = tmp_path / "done"
    done.mkdir()
    (done / "report.xml").write_text(ids.report_xml(cases))
    out = tmp_path / "res"
    env = {
        **os.environ,
        "BENCH_PURPOSE": "nightly",
        "BENCH_ENVIRONMENT": "one-device",
        "BENCH_SELECTION": "perf-core",
        "BENCH_SUBJECT_ID": AID,
        "BENCH_SUITE_FROM": str(done),
        "BENCH_PASS_ENV": "BENCH_SUITE_FROM",
        "BENCH_REUSE": "never",
        "BENCH_QUIET": "1",
    }
    catalog = ["--catalog", str(ROOT / "benchmarks")]
    py = [sys.executable, "-m", "spyre_bench"]
    plan = tmp_path / "plan.json"
    subprocess.run(
        [*py, "plan", *catalog, "--out", str(plan)], env=env, check=True, cwd=ROOT
    )
    subprocess.run(
        [*py, "run", *catalog, "--plan", str(plan), "--out", str(out)],
        env=env,
        check=True,
        cwd=ROOT,
    )
    code = cli(
        ["results", "--bench-dir", str(out), "--arch", "x86_64", "--trigger-type", "perf",
         "--workflow", "bench-integration-tests", "--run-url", "https://ci/run/1",
         "--channel-trust", "ci"]
    )  # fmt: skip
    assert code == 0
    stored = {r[0] for r in db.rows("SELECT toString(benchmark_id) FROM benchmarks")}
    assert stored == {stored_id for _, _, stored_id in cases}
    series = db.rows(
        "SELECT DISTINCT extract(props['source_file'], '^([^/]+)/') FROM benchmark_runs"
    )
    assert series == [("bench-integration-tests",)]
    keys = db.rows(
        "SELECT arraySort(mapKeys(measurements)) FROM benchmark_runs WHERE benchmark_id IN "
        "(SELECT benchmark_id FROM benchmarks WHERE name = 'matmul') LIMIT 1"
    )
    assert keys == [(["total_duration_ms"],)]
