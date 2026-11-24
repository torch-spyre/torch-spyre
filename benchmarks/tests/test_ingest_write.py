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

"""A whole perf report, split into cases and ingested through the real write path, stores the catalog's ids."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmarks" / "tools"

pytest.importorskip("spyre_bench")
pytest.importorskip("yaml")
pytest.importorskip("lxml")
pytest.importorskip("clickhouse_connect")


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class RecordingClient:
    """A v2 database that accepts every write and keeps the inserted rows."""

    def __init__(self, columns):
        self.columns = columns
        self.inserts = []

    def command(self, sql, parameters=None):
        return 1 if sql.split()[0] in ("SELECT", "EXISTS") else None

    def query(self, sql, parameters=None):
        rows = self.columns if "system.columns" in sql else []

        class Result:
            result_rows = rows

        return Result()

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows))


def test_the_split_report_is_stored_under_the_catalog_ids(tmp_path, monkeypatch):
    ids = load("check_identity_ids")
    runner = load("run_kernel_case")
    results, _ = ids.load_ingest()
    from spyre_clickhouse_ingest import schema

    cases = ids.declared()
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "report.xml").write_text(ids.report_xml(cases))
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    for name, inp, _ in cases:
        disc = inp["disc"]
        args = (
            [
                "--model",
                inp["name"],
                "--config",
                disc["config_name"],
                "--mode",
                disc["run_mode"],
            ]
            if disc["record_type"] == "model"
            else ["--op", inp["name"], "--shape", disc["input_shapes"]]
        )
        raw = tmp_path / "raw" / name
        assert (
            runner.main([*args, "--suite-dir", str(suite), "--raw-dir", str(raw)]) == 0
        )
        shutil.copyfile(raw / "report.xml", xml_dir / f"{name}.xml")

    def tables(cls=schema.Table):
        for sub in cls.__subclasses__():
            yield sub
            yield from tables(sub)

    client = RecordingClient([(c,) for t in tables() for c in t.columns])
    monkeypatch.setenv("CLICKHOUSE_HOST", "ch.invalid")
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    monkeypatch.delenv("INGEST_SCHEMA", raising=False)
    monkeypatch.setattr(results, "get_client", lambda: client)
    run_id = "6c1c4a54-3d0a-5e2f-9a43-0f7a1e9d6b11"
    results.main(
        [
            "--xml-dir", str(xml_dir), "--run-id", run_id, "--arch", "x86_64",
            "--trigger-type", "perf", "--schema", "v2", "--workflow", "bench-on-demand",
            "--component", "torch-spyre",
        ]
    )  # fmt: skip

    stored = [r[0] for t, rows in client.inserts if t == "benchmarks" for r in rows]
    assert sorted(stored) == sorted(stored_id for _, _, stored_id in cases)
    assert len(stored) == len(set(stored)) == len(cases)
    runs = [r for t, rows in client.inserts if t == "benchmark_runs" for r in rows]
    assert len(runs) == len(cases)
