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

"""report_sendnn.txt as an offline bundle kind: parsed onto torch-spyre's own benchmark ids."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spyre_clickhouse_ingest import offline, sendnn
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.identity import BenchmarkId
from spyre_clickhouse_ingest.results import _bench_entries

DATA = Path(__file__).parent / "data" / "sendnn"
AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
# The ids prod spyre_v2 holds for these benchmarks, as both CI's report.xml (backend spyre) and
# the hand-loaded sendnn runs (backend sendnn) wrote them.
PROD_IDS = {
    "matmul": "3c086bfe-db98-51f8-a661-908f073dc177",
    "granite": "f0ebde0b-2e2d-5b05-80c8-0ec6c860daa8",
}


def _ids(path):
    return {
        e["name"]: BenchmarkId.derive(
            sendnn.COMPONENT, e["name"], e["tags"], e["disc"], e["disc_keys"]
        )
        for e in _bench_entries(sendnn.parse(path))
    }


def test_the_merged_report_lands_on_prods_benchmark_ids():
    assert _ids(DATA / "report_sendnn.txt") == PROD_IDS
    matmul, granite = _bench_entries(sendnn.parse(DATA / "report_sendnn.txt"))
    assert {matmul["backend"], granite["backend"]} == {"sendnn"}
    assert matmul["measurements"] == {
        "total_duration_ms": [20.0], "spyre_ms": [10.0], "kernel_mean_ms": [1.0],
        "memory_transfer_mean_ms": [9.0], "mem_size_mb": [30.0], "pt_util_percent": [0.5],
    }  # fmt: skip
    assert granite["props"] == {
        "record_type": "model", "config_name": "bs1_pl512_prefill_only_full_model",
        "batch_size": "1", "prompt_length": "512",
    }  # fmt: skip


def test_the_single_stack_report_reads_its_only_column():
    (rec,) = sendnn.parse(DATA / "report_sendnn_single_stack.txt")
    assert (rec["operation_name"], rec["input_shapes"]) == ("softmax", "[[4, 1, 4096]]")
    assert (rec["total_duration_ms"], rec["cpu_ms"], rec["pt_util_percent"]) == (
        2.0,
        1.5,
        0.0,
    )


def test_a_torch_spyre_only_report_has_no_sendnn_rows(tmp_path):
    txt = tmp_path / "report_tsp.txt"
    txt.write_text(
        "Op: softmax  Shape: [[4, 1, 4096]]\nMetric torch-spyre\n---\nwall_clock_ms.mean_ms 0.4\n"
    )
    assert sendnn.parse(txt) == []


def _bundle(tmp_path, *extra, report=DATA / "report_sendnn.txt"):
    out = tmp_path / "b"
    argv = ["results", "--offline", "--out", str(out), "--kind", "sendnn",
            "--sendnn-report", str(report), "--artifact", f"id:{AID}", "--arch", "s390x",
            "--triggered-at", "2026-10-06T11:56:00Z", *extra]  # fmt: skip
    assert cli(argv) == 0
    return out


def test_offline_writes_a_sendnn_bundle_with_test_type_implied(tmp_path):
    out = _bundle(tmp_path)
    meta = json.loads((out / "bundle.json").read_text())
    assert (meta["kind"], meta["started_at"]) == ("sendnn", "2026-10-06T11:56:00Z")
    assert [f["path"] for f in meta["files"]] == ["results/report_sendnn.txt"]
    assert offline.check(out)["test_type"] == "perf"


@pytest.mark.parametrize(
    "text, meta, why",
    [
        ("hello\n", {}, "not a report_sendnn.txt"),
        ("Op: x  Shape: [[1]]\nMetric torch-spyre\n", {}, "not a report_sendnn.txt"),
        (None, {"test_type": "regression"}, "test_type perf"),
    ],
)
def test_a_bad_sendnn_bundle_is_rejected(tmp_path, text, meta, why):
    out = _bundle(tmp_path)
    if text is not None:
        (out / "results" / "report_sendnn.txt").write_text(text)
    old = json.loads((out / "bundle.json").read_text())
    (out / "bundle.json").write_text(
        json.dumps({**{k: v for k, v in old.items() if k != "files"}, **meta})
    )
    with pytest.raises(offline.BundleError) as err:
        offline.check(out)
    assert why in str(err.value)


class CapturingClient:
    def __init__(self):
        self.inserted = {}

    def set_client_setting(self, k, v):
        pass

    def query(self, sql, parameters=None):
        if "count()" in sql:
            table = (
                "artifact_results" if "artifact_results" in sql else "benchmark_runs"
            )
            return SimpleNamespace(result_rows=[[len(self.inserted.get(table, []))]])
        return SimpleNamespace(result_rows=[])

    def insert(self, table, rows, column_names=None, database=None, **_):
        self.inserted.setdefault(table, []).extend(
            dict(zip(column_names, r)) for r in rows
        )


def test_a_sendnn_bundle_writes_benchmark_runs_and_a_perf_verdict(
    tmp_path, monkeypatch, capsys
):
    from spyre_clickhouse_ingest import client, resolver

    out = _bundle(tmp_path)
    ch = CapturingClient()
    monkeypatch.setattr(client.ClickHouse, "connect", lambda **kw: ch)
    monkeypatch.setattr(client, "tables_present", lambda *a, **kw: True)
    monkeypatch.setattr(
        resolver,
        "resolve",
        lambda *a, **kw: SimpleNamespace(
            artifact_id=AID, arch="s390x", component="torch-spyre"
        ),
    )
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    capsys.readouterr()
    code = cli(["results", "--from-bundle", str(out), "--strict", "--uploader", "u@ibm.com",
                "--bundle-url", "https://dash/b", "--bundle-prop", "channel=dashboard"])  # fmt: skip
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (code, report["status"], report["benchmark_runs"]) == (0, "ingested", 2)
    runs = {r["benchmark_id"]: r for r in ch.inserted["benchmark_runs"]}
    assert set(runs) == set(PROD_IDS.values())
    run = runs[PROD_IDS["matmul"]]
    assert (run["component"], run["backend"], run["run_id"]) == (
        "torch-spyre",
        "sendnn",
        report["run_id"],
    )
    assert run["props"] == {
        "report_kind": "perf-suite-sendnn-txt", "source_file": "report_sendnn.txt",
        "platform": "s390x", "run_utc": "2026-10-06 11:56:00",
    }  # fmt: skip
    (verdict,) = ch.inserted["artifact_results"]
    assert (verdict["artifact_id"], verdict["result_kind"], verdict["test_type"]) == (
        AID,
        "performance",
        "perf",
    )
    assert (verdict["props"]["source"], verdict["props"]["channel"]) == (
        "bundle",
        "dashboard",
    )


def test_a_retry_after_a_lost_verdict_completes_the_upload(
    tmp_path, monkeypatch, capsys
):
    from spyre_clickhouse_ingest import client, resolver

    out = _bundle(tmp_path)
    ch = CapturingClient()
    real_insert = ch.insert

    def lose_first_verdict(table, rows, **kw):
        if table == "artifact_results" and not getattr(ch, "lost", False):
            ch.lost = True
            raise ConnectionError("verdict insert dropped")
        real_insert(table, rows, **kw)

    ch.insert = lose_first_verdict
    monkeypatch.setattr(client.ClickHouse, "connect", lambda **kw: ch)
    monkeypatch.setattr(client, "tables_present", lambda *a, **kw: True)
    monkeypatch.setattr(
        resolver,
        "resolve",
        lambda *a, **kw: SimpleNamespace(
            artifact_id=AID, arch="s390x", component="torch-spyre"
        ),
    )
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    args = ["results", "--from-bundle", str(out), "--strict", "--uploader", "u@ibm.com"]
    capsys.readouterr()
    assert cli(args) != 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "failed"
    assert cli(args) == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (report["status"], report["benchmark_runs"]) == ("ingested", 0)
    assert (
        len(ch.inserted["benchmark_runs"]) == 2
        and len(ch.inserted["artifact_results"]) == 1
    )
