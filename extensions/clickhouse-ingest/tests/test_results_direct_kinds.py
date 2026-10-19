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

"""Connected `results` for vllm, sendnn and benchmarks: the rows an uploaded bundle would give."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from spyre_clickhouse_ingest import bundle, client, offline, schema
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.writer import DryRunClient

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
DATA = Path(__file__).parent / "data"
EVAL = {"benchmarks": [
    {"name": "lm_eval/gsm8k", "props": {"model": "org/m"}, "backend": "spyre",
     "measurements": {"exact_match": 0.5}, "iterations": 10},
    {"name": "mteb/STS12", "props": {"model": "org/e"}, "backend": "cpu", "measurements": {"spearman": 0.7}},
]}  # fmt: skip
RUN = ["--artifact", f"id:{AID}", "--arch", "s390x", "--jenkins-run-key", "evals#7",
       "--run-url", "https://ci/evals/7/"]  # fmt: skip


@pytest.fixture
def recorded(monkeypatch):
    """Each bundle `results` hands the relay's ingest: its checked bundle.json and result files."""
    seen = []

    def ingest(root, args):
        meta = offline.check(root)
        seen.append((meta, sorted(p.name for p in (root / "results").iterdir())))
        return {"status": "ingested", "run_key": meta.get("run_key", "")}

    monkeypatch.setattr(bundle, "_ingest", ingest)
    return seen


def _evals(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "evals.json").write_text(json.dumps(EVAL))
    return d


def test_benchmarks_go_through_the_relays_ingest_with_their_run_flags(
    tmp_path, recorded
):
    argv = ["results", "--benchmarks-dir", str(_evals(tmp_path)), "--trigger-type", "quality",
            "--component", "model-quality", *RUN]  # fmt: skip
    assert cli(argv) == 0
    ((meta, files),) = recorded
    assert (meta["kind"], meta["test_type"], meta["component"]) == (
        "benchmarks",
        "quality",
        "model-quality",
    )
    assert (meta["run_key"], meta["run_url"], meta["artifact_id"]) == (
        "evals#7",
        "https://ci/evals/7/",
        AID,
    )
    assert files == ["evals.json"]


def test_sendnn_is_recorded_from_its_report(tmp_path, recorded):
    argv = ["results", "--kind", "sendnn", "--sendnn-report", str(DATA / "sendnn" / "report_sendnn.txt"),
            "--triggered-at", "2026-10-06T11:56:00Z", *RUN]  # fmt: skip
    assert cli(argv) == 0
    ((meta, files),) = recorded
    assert (meta["kind"], meta["started_at"], files) == (
        "sendnn",
        "2026-10-06T11:56:00Z",
        ["report_sendnn.txt"],
    )


def test_vllm_is_recorded_with_its_run_context(tmp_path, recorded):
    src = tmp_path / "vllm-out"
    shutil.copytree(DATA / "vllm_bundle" / "results", src)
    assert (
        cli(
            [
                "results",
                "--vllm-results-dir",
                str(src),
                "--perf",
                "head_sha=881a59d2",
                *RUN,
            ]
        )
        == 0
    )
    ((meta, files),) = recorded
    assert (meta["kind"], meta["perf"]["head_sha"]) == ("vllm", "881a59d2")
    assert files == sorted(p.name for p in src.iterdir())


def test_a_kind_missing_its_input_is_refused_not_read_as_junit(
    tmp_path, recorded, capsys
):
    code = cli(["results", "--kind", "sendnn", *RUN])
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (
        code == offline.REJECTED and "--sendnn-report" in out["reason"] and not recorded
    )


def test_direct_benchmarks_write_the_rows_and_verdict_a_bundle_would(
    tmp_path, monkeypatch, capsys
):
    c = DryRunClient()
    from spyre_clickhouse_ingest import client as ch

    resolution = SimpleNamespace(
        artifact_id=AID, arch="s390x", component="model-quality"
    )
    monkeypatch.setattr(ch.ClickHouse, "connect", lambda **kw: c)
    monkeypatch.setattr(client, "tables_present", lambda *a, **k: True)
    monkeypatch.setattr(
        bundle, "resolve_artifact", lambda *a: (resolution, "artifact_id")
    )
    monkeypatch.setattr(bundle, "existing_run", lambda *a: (set(), set()))
    monkeypatch.setattr(bundle, "verdict_recorded", lambda *a: True)
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    argv = ["results", "--benchmarks-dir", str(_evals(tmp_path)), "--trigger-type", "quality",
            "--component", "model-quality", *RUN]  # fmt: skip
    assert cli(argv) == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (report["status"], report["benchmark_runs"]) == ("ingested", 2)
    assert {r["backend"] for r in c.rows[schema.BENCHMARK_RUNS.name]} == {
        "spyre",
        "cpu",
    }
    (verdict,) = c.rows[schema.ARTIFACT_RESULTS.name]
    assert (verdict["test_type"], verdict["props"]["source"]) == (
        "quality",
        bundle.SOURCE,
    )


def test_junit_still_reads_xml_directly(tmp_path, recorded, capsys):
    with pytest.raises(SystemExit):
        cli(["results", "--xml-dir", str(tmp_path), "--trigger-type", "fvt"])
    assert not recorded and "No XML files found" in capsys.readouterr().out
