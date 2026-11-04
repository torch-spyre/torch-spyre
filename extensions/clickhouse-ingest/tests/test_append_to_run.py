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

"""`results --from-bundle --append-to-run`: a second round of files for a manual run."""

import json
from types import SimpleNamespace

import pytest

from spyre_clickhouse_ingest import bundle, client, offline, resolver, schema
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.writer import (
    BenchmarkWriter,
    DryRunClient,
    TestResultWriter,
)

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
KEY = "manual:me:r1"
UPLOAD = "01a13151-2096-79bf-bd9c-311b8c8bc729"
FIRST = ("93c0abb3-ed25-5934-b811-31b6c149ba47", "s390x", "fvt", "functional", "junit")
JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="s" tests="1"
 timestamp="2026-10-08T10:00:00+00:00"><testcase classname="a.b" name="test_x" time="2"/>
 </testsuite></testsuites>"""
EVAL = {"benchmarks": [{"name": "e/a", "props": {"record_type": "quality"},
                        "backend": "spyre", "measurements": {"acc": 0.5}}]}  # fmt: skip


class Fake:
    """spyre_v2 as the append path reads it: the run's first verdict, files, shas, counts."""

    def __init__(
        self, target=FIRST, shas=("old",), taken=(), listed="", counts=(0, 1, 2.0)
    ):
        self.target, self.shas, self.taken, self.listed = target, shas, taken, listed
        self.counts, self.inserted = counts, []

    def set_client_setting(self, k, v):
        pass

    def query(self, sql, parameters=None):
        if "bundle_kind" in sql:
            rows = [list(self.target)] if self.target else []
        elif "bundle_files" in sql:
            rows = [[self.listed]]
        elif "bundle_sha256" in sql:
            rows = [[s] for s in self.shas]
        elif "upload_id') !=" in sql or "upload_id'] !=" in sql:
            rows = [[f] for f in self.taken]
        elif "source_file" in sql:
            rows = [[f] for f in self.taken]
        elif "countIf(status" in sql:
            rows = [list(self.counts)]
        elif "v_artifact_results_published" in sql:
            rows = [["passed", 3.0]]
        else:
            rows = [[1]]
        return SimpleNamespace(result_rows=rows)

    def insert(self, table, rows, column_names=None, database=None, **_):
        self.inserted.append((table, [dict(zip(column_names, r)) for r in rows]))


@pytest.fixture
def online(monkeypatch):
    state = SimpleNamespace(client=Fake(), calls=[])
    monkeypatch.setattr(client.ClickHouse, "connect", lambda **kw: state.client)
    monkeypatch.setattr(client, "tables_present", lambda *a, **k: True)
    monkeypatch.setattr(
        resolver, "resolve",
        lambda *a, **k: SimpleNamespace(artifact_id=AID, arch="s390x", component="torch-spyre"),
    )  # fmt: skip
    monkeypatch.setattr(
        bundle, "write_junit",
        lambda db, root, meta, aid, rid, props, strict: state.calls.append(
            bundle.results_argv(meta, root, aid, rid, props, strict)
        ),
    )  # fmt: skip
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    return state


def _root(tmp_path, name="b", meta=None, files=None):
    root = tmp_path / name
    (root / "results").mkdir(parents=True)
    for fname, text in (files or {"two.xml": JUNIT}).items():
        (root / "results" / fname).write_text(text)
    (root / "bundle.json").write_text(json.dumps({
        "artifact_id": AID, "test_type": "fvt", "arch": "s390x", "run_key": KEY, **(meta or {})
    }))  # fmt: skip
    return root


def _run(path, capsys, *extra, append=True, upload=UPLOAD):
    capsys.readouterr()
    argv = [
        "results",
        "--from-bundle",
        str(path),
        "--strict",
        "--uploader",
        "me",
        *extra,
    ]
    if append:
        argv.append("--append-to-run")
    if upload:
        argv += ["--bundle-prop", f"upload_id={upload}"]
    code = cli(argv)
    return code, json.loads(capsys.readouterr().out.splitlines()[-1])


def test_the_flag_is_in_results_help(capsys):
    with pytest.raises(SystemExit):
        cli(["results", "--help"])
    assert "--append-to-run" in capsys.readouterr().out


def test_an_append_is_accepted_for_a_run_that_has_verdicts(tmp_path, online, capsys):
    code, report = _run(_root(tmp_path), capsys)
    assert (code, report["status"], report["append"]) == (0, "ingested", True)
    argv = online.calls[0]
    results = {argv[i + 1] for i, a in enumerate(argv) if a == "--result-prop"}
    cases = {argv[i + 1] for i, a in enumerate(argv) if a == "--case-prop"}
    assert {f"upload_id={UPLOAD}", "round=1"} <= results == results | cases
    assert cases == {f"upload_id={UPLOAD}", "round=1"}


def test_the_junit_verdict_is_recounted_over_the_whole_run(tmp_path, online, capsys):
    online.client.counts = (1, 3, 2.0)
    _run(_root(tmp_path), capsys)
    (verdict,) = [
        r for t, rows in online.client.inserted if t == "artifact_results" for r in rows
    ]
    assert (verdict["state"], verdict["duration_s"]) == ("failed", 5.0)
    assert verdict["props"]["upload_id"] == UPLOAD and verdict["props"]["round"] == "1"
    assert verdict["props"]["failure_detail"].startswith("1 of 3 failed")


def test_without_the_flag_a_run_with_verdicts_is_still_refused(
    tmp_path, online, capsys
):
    code, report = _run(_root(tmp_path), capsys, append=False)
    assert (
        code == offline.REJECTED and "already recorded by another" in report["reason"]
    )


def test_the_same_sha_is_a_duplicate(tmp_path, online, capsys):
    root = _root(tmp_path)
    online.client.shas = (offline.digest(root),)
    code, report = _run(root, capsys)
    assert (code, report["status"]) == (0, "duplicate") and online.calls == []


def test_a_round_needs_its_upload_id(tmp_path, online, capsys):
    code, report = _run(_root(tmp_path), capsys, upload="")
    assert code == offline.REJECTED and "upload_id" in report["reason"]


def test_a_jenkins_run_is_refused(tmp_path, online, capsys):
    root = _root(tmp_path, meta={"run_key": "Spyre/orchestrator#12"})
    code, report = _run(root, capsys)
    assert code == offline.REJECTED and "not a manual run" in report["reason"]


def test_a_run_with_no_results_is_refused(tmp_path, online, capsys):
    online.client.target = None
    code, report = _run(_root(tmp_path), capsys)
    assert code == offline.REJECTED and "no recorded results" in report["reason"]


@pytest.mark.parametrize(
    "meta, target, why",
    [
        ({"arch": "x86_64"}, FIRST, "arch s390x"),
        ({"test_type": "svt"}, FIRST, "test_type fvt"),
        ({}, ("0" * 8 + AID[8:],) + FIRST[1:], "artifact"),
        ({}, FIRST[:4] + ("vllm",), "is a vllm run"),
        ({}, FIRST[:3] + ("performance", ""), "is performance"),
    ],
)
def test_a_mismatch_with_the_run_is_refused(
    tmp_path, online, capsys, meta, target, why
):
    online.client.target = target
    code, report = _run(_root(tmp_path, meta=meta), capsys)
    assert code == offline.REJECTED and why in report["reason"]


def test_a_file_name_the_run_holds_is_refused_never_overwritten(
    tmp_path, online, capsys
):
    online.client.taken = ("two.xml",)
    code, report = _run(_root(tmp_path), capsys)
    assert code == offline.REJECTED and "['two.xml']" in report["reason"]
    assert online.calls == []


def test_a_name_held_only_by_a_verdicts_file_list_is_refused(tmp_path, online, capsys):
    online.client.listed = "one.xml\ntwo.xml"
    code, report = _run(_root(tmp_path), capsys)
    assert code == offline.REJECTED and "two.xml" in report["reason"]


def test_a_tables_style_file_name_is_refused_on_a_second_round(
    tmp_path, online, capsys
):
    online.client.taken = ("test_case_runs.json",)
    root = _root(tmp_path, files={"test_case_runs.json": "{}"})
    code, report = _run(root, capsys)
    assert code == offline.REJECTED and "test_case_runs.json" in report["reason"]


def test_a_retry_of_the_same_round_is_not_refused_for_its_own_files(
    tmp_path, online, capsys
):
    # Rows tagged with this upload_id are excluded from the names the run "holds".
    assert _run(_root(tmp_path), capsys)[1]["status"] == "ingested"


def test_dry_run_checks_but_writes_nothing(tmp_path, online, capsys):
    code, report = _run(_root(tmp_path), capsys, "--dry-run")
    assert (code, report["status"]) == (0, "would-ingest") and online.calls == []


def test_a_generic_benchmarks_round_tags_rows_and_supersedes_the_verdict(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(client, "tables_present", lambda *a, **k: True)
    root = tmp_path / "g"
    (root / "results").mkdir(parents=True)
    (root / "results" / "evals.json").write_text(json.dumps(EVAL))
    meta = {"kind": "benchmarks", "test_type": "quality", "component": "spyre-inference",
            "arch": "s390x", "run_key": KEY, "artifact_id": AID}  # fmt: skip
    c = DryRunClient(Fake())
    props = {"upload_id": UPLOAD, "round": "1", "source": "bundle"}
    bundle.write_generic(c, "spyre_v2", root, meta, AID, "r", props)
    (run,) = c.rows[schema.BENCHMARK_RUNS.name]
    assert run["props"]["upload_id"] == UPLOAD and run["props"]["round"] == "1"
    (verdict,) = c.rows[schema.ARTIFACT_RESULTS.name]
    assert verdict["props"]["upload_id"] == UPLOAD and verdict["duration_s"] == 3.0


def test_a_first_upload_tags_upload_id_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "tables_present", lambda *a, **k: True)
    root = tmp_path / "g"
    (root / "results").mkdir(parents=True)
    (root / "results" / "evals.json").write_text(json.dumps(EVAL))
    meta = {"kind": "benchmarks", "test_type": "quality", "component": "spyre-inference",
            "arch": "s390x", "run_key": KEY, "artifact_id": AID}  # fmt: skip
    c = DryRunClient()
    bundle.write_generic(c, "spyre_v2", root, meta, AID, "r", {"upload_id": UPLOAD})
    (run,) = c.rows[schema.BENCHMARK_RUNS.name]
    assert run["props"]["upload_id"] == UPLOAD and "round" not in run["props"]


def test_sendnn_and_vllm_rows_take_the_row_props(monkeypatch, tmp_path):
    from spyre_clickhouse_ingest import sendnn, vllm

    seen = {}
    monkeypatch.setattr(
        BenchmarkWriter, "insert",
        lambda *a, **k: seen.setdefault("sendnn", (a, k)) and 1,
    )  # fmt: skip
    monkeypatch.setattr(BenchmarkWriter, "already_ingested", lambda *a, **k: False)
    monkeypatch.setattr(sendnn, "parse", lambda p: [])
    monkeypatch.setattr("spyre_clickhouse_ingest.results._bench_entries", lambda e: [])
    sendnn.write(None, "db", "r", tmp_path / "x.txt", {}, {"upload_id": UPLOAD})
    assert seen["sendnn"][0][-1] == {"upload_id": UPLOAD}
    monkeypatch.setattr(vllm, "tables_present", lambda *a, **k: True)
    monkeypatch.setattr(vllm, "benchmarks_already_ingested", lambda *a, **k: False)
    seen.clear()
    monkeypatch.setattr(vllm, "insert_benchmarks", lambda *a, **k: seen.update(k) or 1)
    vllm.write_benchmarks(Fake(), "db", [], "r", row_props={"upload_id": UPLOAD})
    assert seen["row_props"] == {"upload_id": UPLOAD}


def test_the_prop_changes_no_row_id():
    case = {"classname": "T", "name": "t", "status": "passed", "properties": []}
    bench = [{"name": "b", "tags": [], "backend": "x", "props": {"k": "v"}, "disc": {"k": "v"},
              "disc_keys": ("k",), "measurements": {"m": [1.0]}}]  # fmt: skip

    def ids(row_props):
        c = DryRunClient()
        TestResultWriter.insert(
            c, "db", "comp", "r", [dict(case)], "a.xml", row_props=row_props
        )
        BenchmarkWriter.insert(
            c, "db", "comp", "r", [dict(bench[0])], "k", "a.json", row_props=row_props
        )
        runs = c.rows[schema.TEST_CASE_RUNS.name][0]
        brun = c.rows[schema.BENCHMARK_RUNS.name][0]
        return runs["test_case_id"], brun["benchmark_id"], runs["props"], brun["props"]

    plain, tagged = ids(None), ids({"upload_id": UPLOAD, "round": "1"})
    assert plain[:2] == tagged[:2]
    assert tagged[2]["upload_id"] == UPLOAD and tagged[3]["round"] == "1"
    assert "upload_id" not in plain[2] and "upload_id" not in plain[3]
