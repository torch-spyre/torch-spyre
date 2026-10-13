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

"""`results --offline | --upload | --from-bundle`: results recorded offline, ingested later."""

import io
import json
import shutil
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from spyre_clickhouse_ingest import bundle, offline, results, schema
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.identity import RunId
from spyre_clickhouse_ingest.junit import RunCoordinates

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
OTHER = "0" * 8 + AID[8:]
IMAGE = "icr.io/ai_sw_accel/2.0/prod/spyre-inference-devel@sha256:" + "b6" * 32
JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="s" tests="2"
 timestamp="2026-10-08T10:00:00+00:00"><testcase classname="a.b" name="test_x" time="1"/>
 <testcase classname="a.b" name="test_y" time="1"><failure message="boom"/></testcase>
 </testsuite></testsuites>"""
VLLM_DATA = Path(__file__).parent / "data" / "vllm_bundle"


def _xml(tmp_path, text=JUNIT):
    d = tmp_path / "xml"
    d.mkdir(exist_ok=True)
    (d / "junit.xml").write_text(text)
    return d


def _offline(tmp_path, *extra, out="b", artifact=f"id:{AID}", capsys=None):
    """`results --offline --out` with the usual flags; returns the bundle path."""
    path = tmp_path / out
    argv = ["results", "--offline", "--out", str(path), "--xml-dir", str(_xml(tmp_path)),
            "--artifact", artifact, "--arch", "s390x", "--trigger-type", "fvt", *extra]  # fmt: skip
    assert cli(argv) == 0
    if capsys:
        capsys.readouterr()
    return path


def _meta(path):
    return json.loads((path / "bundle.json").read_text())


def _edit(path, **fields):
    meta = {**_meta(path), **fields}
    (path / "bundle.json").write_text(
        json.dumps({k: v for k, v in meta.items() if v is not None})
    )


def _minimal(tmp_path, meta=None, name="m"):
    root = tmp_path / name
    (root / "results").mkdir(parents=True)
    (root / "results" / "junit.xml").write_text(JUNIT)
    (root / "bundle.json").write_text(
        json.dumps(meta or {"artifact_id": AID, "test_type": "fvt"})
    )
    return root


def _rejected(path, code=offline.REJECTED):
    with pytest.raises(offline.BundleError) as err:
        offline.check(path)
    assert err.value.code == code
    return str(err.value)


# --- the format, offline ------------------------------------------------------------------


def test_schema_tiers_are_the_ddl_check_set():
    assert (
        set(offline.schema()["properties"]["test_type"]["enum"])
        == schema.TEST_TYPE_VALUES
    )


def test_offline_writes_a_bundle_from_the_usual_flags(tmp_path, capsys):
    path = _offline(tmp_path, "--component", "spyre-inference", "--jenkins-run-key",
                    "Spyre-Test/testing/Jenkinsfile.x#7", "--run-url", "https://j/7/")  # fmt: skip
    report = json.loads(capsys.readouterr().out)
    meta = _meta(path)
    assert (meta["artifact_id"], meta["test_type"], meta["arch"]) == (
        AID,
        "fvt",
        "s390x",
    )
    assert meta["run_key"] == "Spyre-Test/testing/Jenkinsfile.x#7" == report["run_key"]
    assert [f["path"] for f in meta["files"]] == ["results/junit.xml"]
    assert meta["started_at"] == "2026-10-08T10:00:00Z"
    assert report["run_id"] == RunId.derive("jenkins", meta["run_key"], "s390x", "fvt")
    assert report["upload_path"].startswith(
        f"zsp/next/s390x/v2-results/inbox/{AID}/Spyre-Test_testing_"
    )


def test_offline_tgz_and_validate_only(tmp_path, capsys):
    tgz = _offline(tmp_path, out="b.tgz", artifact=f"image:{IMAGE}", capsys=capsys)
    assert cli(["results", "--from-bundle", str(tgz), "--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "valid"
    assert report["upload_path"].split("/")[5] == "sha256-" + "b6" * 32
    assert report["run_key"].startswith("manual:<uploader>:")


def test_the_minimal_bundle_is_an_artifact_a_tier_and_a_result(tmp_path):
    assert offline.check(_minimal(tmp_path))["test_type"] == "fvt"
    image_only = _minimal(tmp_path, {"image": IMAGE, "test_type": "fvt"}, name="i")
    assert "only with arch" in _rejected(image_only)
    _edit(image_only, arch="s390x")
    assert offline.check(image_only)["image"] == IMAGE
    vllm_only = tmp_path / "v"
    shutil.copytree(VLLM_DATA, vllm_only)
    (vllm_only / "bundle.json").write_text(
        json.dumps({"kind": "vllm", "artifact_id": AID})
    )
    assert offline.check(vllm_only)["test_type"] == "perf"


@pytest.mark.parametrize(
    "meta, why",
    [
        ({"artifact_id": AID}, "test_type is required"),
        ({"test_type": "fvt"}, "needs artifact_id or image"),
        (
            {"artifact_id": "", "image": "", "test_type": "fvt"},
            "needs artifact_id or image",
        ),
        (
            {"artifact_id": AID, "test_type": "fvt", "schema_version": 2},
            "newer than this ingest",
        ),
        ({"artifact_id": AID, "test_type": "nightly"}, "is not one of"),
        (
            {"artifact_id": AID, "test_type": "fvt", "colour": "red"},
            "unknown key 'colour'",
        ),
        ({"artifact_id": AID, "test_type": "fvt", "tag_family": "x"}, "verdicts only"),
        ({"artifact": f"id:{OTHER}", "artifact_id": AID, "test_type": "fvt"}, "differ"),
        (
            {
                "artifact_id": AID,
                "test_type": "fvt",
                "run_key": "J#1",
                "jenkins_run_key": "J#2",
            },
            "differ",
        ),
        (
            {"artifact_id": AID, "test_type": "fvt", "started_at": "2026-10-08 10:00"},
            "with a zone",
        ),
        (
            {"artifact_id": AID, "test_type": "fvt", "run_key": "a b#1"},
            "does not match",
        ),
        ({"image": IMAGE[:-1], "arch": "s390x", "test_type": "fvt"}, "does not match"),
        (
            {"artifact_id": AID, "test_type": "fvt", "component": "Torch"},
            "does not match",
        ),
    ],
)
def test_bad_bundle_json_is_rejected(tmp_path, meta, why):
    assert why in _rejected(_minimal(tmp_path, meta))


def test_files_are_checked_only_when_listed(tmp_path):
    path = _offline(tmp_path)
    (path / "results" / "extra.xml").write_text(JUNIT)
    assert "not in files[]" in _rejected(path)
    (path / "results" / "extra.xml").unlink()
    (path / "results" / "junit.xml").write_text(JUNIT.replace("boom", "bang"))
    assert "sha256 mismatch" in _rejected(path)
    (path / "results" / "junit.xml").unlink()
    assert "missing" in _rejected(path, offline.INCOMPLETE)
    _edit(path, files=None)
    (path / "results" / "other.xml").write_text(JUNIT)
    assert offline.check(path)


def test_files_outside_results_and_attachments_are_rejected(tmp_path):
    root = _minimal(tmp_path)
    (root / "notes.txt").write_text("x")
    assert "belong in results/" in _rejected(root)


def test_a_bundle_with_no_cases_records_nothing(tmp_path):
    root = _minimal(tmp_path)
    (root / "results" / "junit.xml").write_text(
        "<testsuites><testsuite name='s'/></testsuites>"
    )
    assert "no <testcase>" in _rejected(root)


def test_a_tgz_escaping_its_directory_is_rejected(tmp_path, capsys):
    tgz = tmp_path / "evil.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        info = tarfile.TarInfo("../bundle.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    assert (
        cli(["results", "--from-bundle", str(tgz), "--validate-only"])
        == offline.REJECTED
    )


def test_the_digest_covers_content_not_the_bundle_json_alone(tmp_path):
    root = _minimal(tmp_path)
    before = offline.digest(root)
    (root / "results" / "junit.xml").write_text(JUNIT.replace("boom", "bang"))
    assert offline.digest(root) != before


def test_the_offline_path_needs_only_the_standard_library(tmp_path):
    """No clickhouse-connect, regex or yaml: an air-gapped host has only a --no-deps wheel."""
    xml = _xml(tmp_path)
    code = (
        "import sys\n"
        "for m in ('clickhouse_connect', 'regex', 'yaml'): sys.modules[m] = None\n"
        "from spyre_clickhouse_ingest.__main__ import main\n"
        f"assert main(['results', '--offline', '--out', {str(tmp_path / 'b.tgz')!r}, '--xml-dir', {str(xml)!r},"
        f" '--artifact', 'id:{AID}', '--arch', 's390x', '--trigger-type', 'fvt']) == 0\n"
        f"assert main(['results', '--from-bundle', {str(tmp_path / 'b.tgz')!r}, '--validate-only']) == 0\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_offline_flags_are_results_flags():
    """Each flag the offline parser reads means the same in `results`."""
    online = {
        a.option_strings[0]: a.dest
        for a in results.build_parser()._actions
        if a.option_strings
    }
    for a in offline.parser()._actions:
        if a.option_strings and a.dest != "help":
            assert online.get(a.option_strings[0]) == a.dest, a.option_strings


def test_a_jenkins_key_hashes_as_the_connected_run_and_a_manual_one_as_a_bundle():
    key = "Spyre-Test/testing/Jenkinsfile.spyreinference#131"
    meta = {"run_key": key, "arch": "s390x", "test_type": "svt"}
    connected = SimpleNamespace(run_id="", gha_run_id="", jenkins_run_key=key)
    assert bundle.run_id(meta) == RunId.for_args(connected, "", "s390x", "svt")
    assert RunCoordinates.source_and_external(connected, "") == ("jenkins", key)
    manual = {**meta, "run_key": "manual:me:abc"}
    assert bundle.run_id(manual) == RunId.derive(
        "bundle", "manual:me:abc", "s390x", "svt"
    )


# --- results --from-bundle, against a fake spyre_v2 ----------------------------------------


class FakeClient:
    """Answers ingest's existence reads, and the verdict check."""

    def __init__(self, verdicts=(), files=(), landed=True):
        self.verdicts, self.files, self.landed = verdicts, files, landed
        self.settings = {}

    def set_client_setting(self, k, v):
        self.settings[k] = v

    def query(self, sql, parameters=None):
        if "count()" in sql:
            rows = [[int(self.landed)]]
        elif "test_case_runs" in sql:
            rows = [[f] for f in self.files]
        else:
            rows = [[v] for v in self.verdicts]
        return SimpleNamespace(result_rows=rows)


def _resolution(aid, arch="s390x", component="torch-spyre"):
    return SimpleNamespace(artifact_id=aid, arch=arch, component=component)


@pytest.fixture
def online(monkeypatch):
    """--from-bundle against a fake spyre_v2: `calls` holds each `results` argv."""
    state = SimpleNamespace(client=FakeClient(), calls=[], resolved=AID, by_spec={})
    from spyre_clickhouse_ingest import client, resolver

    def resolve(spec, arch, **kw):
        aid = state.by_spec.get(spec, state.resolved)
        return _resolution(aid) if aid else None

    monkeypatch.setattr(client.ClickHouse, "connect", lambda **kw: state.client)
    monkeypatch.setattr(resolver, "resolve", resolve)
    monkeypatch.setattr(
        bundle, "write_junit",
        lambda db, root, meta, aid, rid, props, strict: state.calls.append(
            bundle.results_argv(meta, root, aid, rid, props, strict)
        ),
    )  # fmt: skip
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    return state


def _ingest(path, capsys, *extra):
    capsys.readouterr()
    code = cli(["results", "--from-bundle", str(path), "--strict", *extra])
    return code, json.loads(capsys.readouterr().out.splitlines()[-1])


def _flags(argv):
    return dict(zip(argv[::2], argv[1::2]))


def test_a_minimal_bundle_takes_arch_component_and_run_key_from_artifact_and_content(
    tmp_path, online, capsys
):
    root = _minimal(tmp_path)
    code, report = _ingest(
        root, capsys, "--uploader", "jdoe", "--bundle-url", "https://art/b/"
    )
    assert (code, report["status"]) == (0, "ingested")
    sha = offline.digest(root)
    assert report["run_key"] == f"manual:jdoe:{sha}" and report["bundle_sha256"] == sha
    argv = online.calls[0]
    flags = _flags(argv)
    assert (flags["--arch"], flags["--component"]) == ("s390x", "torch-spyre")
    assert flags["--artifact"] == f"id:{AID}" and flags["--lookup"] == "only"
    assert flags["--run-id"] == RunId.derive(
        "bundle", f"manual:jdoe:{sha}", "s390x", "fvt"
    )
    assert flags["--run-url"] == "https://art/b/" and "--jenkins-run-key" not in argv
    props = {argv[i + 1] for i, a in enumerate(argv) if a == "--result-prop"}
    assert {"source=bundle", "uploader=jdoe", f"bundle_sha256={sha}"} <= props
    assert argv[-1] == "--strict" and "--dry-run" not in argv


def test_a_re_upload_of_the_same_content_is_a_duplicate(tmp_path, online, capsys):
    root = _minimal(tmp_path)
    online.client.verdicts = [offline.digest(root)]
    code, report = _ingest(root, capsys, "--uploader", "jdoe")
    assert (code, report["status"]) == (0, "duplicate") and online.calls == []


@pytest.mark.parametrize(
    "verdicts, files, why",
    [
        ([""], [], "already recorded by another run"),
        (["f" * 64], [], "already recorded by another run"),
        ([], ["other.xml"], "already has cases from ['other.xml']"),
    ],
)
def test_a_run_key_taken_by_other_results_is_rejected(
    tmp_path, online, capsys, verdicts, files, why
):
    root = _minimal(
        tmp_path, {"artifact_id": AID, "test_type": "fvt", "run_key": "manual:me:r1"}
    )
    online.client.verdicts, online.client.files = verdicts, files
    code, report = _ingest(root, capsys)
    assert (code, report["status"]) == (offline.REJECTED, "rejected") and why in report[
        "reason"
    ]


def test_a_partly_ingested_bundle_is_completed(tmp_path, online, capsys):
    online.client.files = ["junit.xml"]
    assert _ingest(_minimal(tmp_path), capsys)[1]["status"] == "ingested"


def test_an_unrecorded_artifact_is_rejected(tmp_path, online, capsys):
    online.resolved = ""
    code, report = _ingest(_minimal(tmp_path), capsys)
    assert code == offline.REJECTED and "artifact not recorded" in report["reason"]


def test_the_image_is_the_fallback_for_an_unrecorded_id(tmp_path, online, capsys):
    root = _minimal(
        tmp_path,
        {"artifact_id": AID, "image": IMAGE, "arch": "s390x", "test_type": "fvt"},
    )
    online.by_spec = {f"id:{AID}": "", f"image:{IMAGE}": OTHER}
    code, report = _ingest(root, capsys)
    assert (code, report["artifact_id"], report["artifact_from"]) == (0, OTHER, "image")
    assert _flags(online.calls[0])["--artifact"] == f"id:{OTHER}"


def test_an_id_and_an_image_naming_different_artifacts_are_rejected(
    tmp_path, online, capsys
):
    root = _minimal(
        tmp_path,
        {"artifact_id": AID, "image": IMAGE, "arch": "s390x", "test_type": "fvt"},
    )
    online.by_spec = {f"image:{IMAGE}": OTHER}
    code, report = _ingest(root, capsys)
    assert code == offline.REJECTED and "different artifacts" in report["reason"]


def test_the_inbox_folder_must_be_the_id_or_the_image_digest(tmp_path, online, capsys):
    root = _minimal(
        tmp_path,
        {"artifact_id": AID, "image": IMAGE, "arch": "s390x", "test_type": "fvt"},
    )
    assert _ingest(root, capsys, "--expect-key", "sha256-" + "b6" * 32)[0] == 0
    code, report = _ingest(root, capsys, "--expect-key", OTHER)
    assert code == offline.REJECTED and "is not its artifact_id" in report["reason"]


def test_an_untrusted_jenkins_key_is_rejected(tmp_path, online, capsys):
    root = _minimal(
        tmp_path,
        {"artifact_id": AID, "test_type": "fvt", "run_key": "Spyre/orchestrator#12"},
    )
    code, report = _ingest(root, capsys, "--trusted-job-prefix", "Spyre-Test/testing/")
    assert code == offline.REJECTED and "not from a trusted job" in report["reason"]
    assert _ingest(root, capsys, "--trusted-job-prefix", "Spyre/")[0] == 0
    assert "--jenkins-run-key" in online.calls[0]


def test_a_verdict_that_did_not_land_is_a_retry(tmp_path, online, capsys):
    online.client.landed = False
    code, report = _ingest(_minimal(tmp_path), capsys)
    assert (code, report["status"]) == (offline.FAILED, "failed")


def test_dry_run_reads_on_a_readonly_connection_and_writes_nothing(
    tmp_path, online, capsys
):
    code, report = _ingest(_minimal(tmp_path), capsys, "--dry-run")
    assert (code, report["status"]) == (0, "would-ingest")
    assert online.calls == [] and online.client.settings == {"readonly": "2"}


def test_result_props_reach_the_verdict_and_replace_its_source(monkeypatch):
    written = []
    monkeypatch.setattr(
        results, "ensure",
        lambda *a, **kw: SimpleNamespace(dry_run=False, identity=SimpleNamespace(artifact_id=AID)),
    )  # fmt: skip
    monkeypatch.setattr(
        results, "insert_artifact_result", lambda *a, **kw: written.append(kw) or True
    )
    args = SimpleNamespace(
        artifact=f"id:{AID}", artifact_id="", lookup="only", registry="off", tag_family="",
        tags=[], tag_date=None, origin="promoted", sources=[], identity_deps=[], context_deps=[],
        props=[], tag_props=[], run_url="https://art/b/", dry_run=False, arch="s390x",
        repository="", branch="", sha="", jenkins_run_key="J#1", gha_run_id="", component="",
        result_props=[("source", "bundle"), ("uploader", "jdoe")], run_attempt=0,
    )  # fmt: skip
    legs = {
        ("d3ea9749-67a5-5bd1-8471-a290d0c67fc9", "fvt"): {
            "failed": 0,
            "total": 2,
            "duration_s": 1.0,
        }
    }
    assert results._write_named_artifact_verdicts(None, "spyre_v2", args, legs)
    assert written[0]["props"] == {
        "run_url": "https://art/b/",
        "source": "bundle",
        "uploader": "jdoe",
    }


# --- vLLM bench bundles ---------------------------------------------------------------------

# What prod spyre_v2.benchmarks holds for the live spyre-inference leg's two benchmarks.
PROD_IDS = {
    "latency_granite8B_tp1_in64_out64": "9d066729-8f4e-5e55-9871-2f226a441285",
    "throughput_granite8B_tp1_in64_out64": "f00e5306-f7fb-53ff-802e-470084835bd4",
}
MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"


def _vllm(tmp_path, *extra):
    src = tmp_path / "vllm-out"
    shutil.copytree(VLLM_DATA / "results", src)
    shutil.copy(VLLM_DATA / "attachments" / "latency_granite8B_tp1_in64_out64.cmd", src)
    out = tmp_path / "perf"
    argv = ["results", "--offline", "--out", str(out), "--vllm-results-dir", str(src),
            "--artifact", f"id:{AID}", "--arch", "x86_64", "--perf", "head_sha=881a59d2",
            "--perf", "head_branch=main", *extra]  # fmt: skip
    assert cli(argv) == 0
    return out


class CapturingClient:
    """Captures inserts; a count is of the rows inserted so far, every other probe is empty."""

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


def test_offline_vllm_bundles_the_bench_json_and_its_commands(tmp_path):
    meta = _meta(_vllm(tmp_path))
    assert (meta["kind"], meta["perf"]["head_sha"]) == ("vllm", "881a59d2")
    assert "test_type" not in meta and "component" not in meta
    assert {f["path"] for f in meta["files"]} == {
        "results/latency_granite8B_tp1_in64_out64.json",
        "results/latency_granite8B_tp1_in64_out64.pytorch.json",
        "results/throughput_granite8B_tp1_in64_out64.json",
        "results/throughput_granite8B_tp1_in64_out64.pytorch.json",
        "attachments/latency_granite8B_tp1_in64_out64.cmd",
    }


@pytest.mark.parametrize(
    "setup, why",
    [
        (
            lambda out: (out / "results" / "x.xml").write_text(JUNIT),
            "results/*.json only",
        ),
        (
            lambda out: (out / "results" / "decode_tp1.json").write_text("{}"),
            "must start with one of",
        ),
        (
            lambda out: (out / "results" / "latency_x.json").write_text('{"x": 1}'),
            "no vLLM",
        ),
        (lambda out: _edit(out, perf={"tensor_parallel": "4"}), "differ from perf"),
        (lambda out: _edit(out, test_type="regression"), "test_type perf"),
    ],
)
def test_a_bad_vllm_bundle_is_rejected(tmp_path, setup, why):
    out = _vllm(tmp_path)
    setup(out)
    _edit(out, files=None)
    assert why in _rejected(out)


def test_a_vllm_bundle_writes_the_live_legs_rows(tmp_path, monkeypatch, capsys):
    """Same benchmark ids prod holds, same measurement and prop shapes; only source differs."""
    from spyre_clickhouse_ingest import client, resolver, vllm

    out = _vllm(tmp_path)
    ch = CapturingClient()
    monkeypatch.setattr(client.ClickHouse, "connect", lambda **kw: ch)
    monkeypatch.setattr(client, "tables_present", lambda *a, **kw: True)
    monkeypatch.setattr(vllm, "tables_present", lambda *a, **kw: True)
    monkeypatch.setattr(
        resolver,
        "resolve",
        lambda *a, **kw: _resolution(AID, "x86_64", "spyre-inference"),
    )
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    code, report = _ingest(out, capsys, "--bundle-url", "https://art/p/")
    assert (code, report["status"], report["benchmark_runs"]) == (0, "ingested", 2)

    ids = {r["name"]: r["benchmark_id"] for r in ch.inserted["benchmarks"]}
    assert ids == PROD_IDS
    runs = {r["benchmark_id"]: r for r in ch.inserted["benchmark_runs"]}
    lat = runs[PROD_IDS["latency_granite8B_tp1_in64_out64"]]
    thr = runs[PROD_IDS["throughput_granite8B_tp1_in64_out64"]]
    assert sorted(lat["measurements"]) == [
        "avg_latency",
        "latency",
        "p10_latency",
        "p25_latency",
        "p50_latency",
        "p75_latency",
        "p90_latency",
        "p99_latency",
    ]
    assert lat["measurements"]["latency"] == [0.7093, 0.7155] and lat["iterations"] == 2
    assert sorted(thr["measurements"]) == [
        "elapsed_time",
        "requests_per_second",
        "tokens_per_second",
    ]
    assert (
        thr["iterations"] == 1
        and lat["backend"] == "spyre"
        and lat["run_id"] == report["run_id"]
    )
    assert {k: v for k, v in lat["props"].items() if not k.startswith("unit.")} == {
        "report_kind": "vllm", "repo": "spyre-inference", "head_branch": "main", "workflow_id": "0",
        "run_attempt": "1", "job_id": "0", "head_sha": "881a59d2", "arch": "x86_64",
        "hardware_type": "IBM_Spyre",
    }  # fmt: skip
    (bench,) = [b for b in ch.inserted["benchmarks"] if b["name"].startswith("latency")]
    assert bench["props"] == {"record_type": "model", "run_mode": "latency", "tensor_parallel": "1",
                              "input_len": "64", "output_len": "64", "model": MODEL}  # fmt: skip

    (verdict,) = ch.inserted["artifact_results"]
    assert (verdict["artifact_id"], verdict["run_id"]) == (AID, report["run_id"])
    assert (verdict["result_kind"], verdict["test_type"], verdict["state"]) == (
        "performance",
        "perf",
        "passed",
    )
    assert verdict["duration_s"] == pytest.approx(3.162)
    assert (
        verdict["props"]["source"] == "bundle"
        and verdict["props"]["run_url"] == "https://art/p/"
    )


def test_a_native_file_alone_takes_its_model_from_perf(tmp_path):
    out = _vllm(tmp_path, "--perf", f"model={MODEL}")
    for f in (out / "results").glob("*.pytorch.json"):
        f.unlink()
    _edit(out, files=None)
    rows = bundle.vllm_rows(out, offline.check(out) | {"arch": "x86_64"})
    assert {json.loads(r["extra"])["model"] for r in rows} == {MODEL}


# --- results --upload, against a local Artifactory stand-in ----------------------------------


class FakeArtifactory(BaseHTTPRequestHandler):
    """PUT stores; GET serves stored files and api/storage folder listings; `fail` 503s N PUTs."""

    store: dict = {}
    seen: list = []
    fail = 0

    def log_message(self, *a):
        pass

    def do_PUT(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).seen.append(
            ("PUT", self.path, self.headers.get("Authorization", ""))
        )
        if type(self).fail:
            type(self).fail -= 1
            self.send_response(503)
            self.end_headers()
            return
        type(self).store[self.path] = body
        self.send_response(201)
        self.end_headers()

    def do_GET(self):
        prefix = "/artifactory/api/storage/repo/"
        if self.path.startswith(prefix):
            folder = "/artifactory/repo/" + self.path[len(prefix) :] + "/"
            kids = {
                p[len(folder) :].split("/")[0]
                for p in type(self).store
                if p.startswith(folder)
            }
            body = json.dumps(
                {"children": [{"uri": "/" + k} for k in sorted(kids)]}
            ).encode()
            code = 200 if kids else 404
        else:
            body, code = (
                type(self).store.get(self.path, b""),
                200 if self.path in type(self).store else 404,
            )
        self.send_response(code)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def art(monkeypatch):
    FakeArtifactory.store, FakeArtifactory.seen, FakeArtifactory.fail = {}, [], 0
    server = HTTPServer(("127.0.0.1", 0), FakeArtifactory)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(offline.time, "sleep", lambda s: None)
    monkeypatch.delenv("ARTIFACTORY_USER", raising=False)
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "tok")
    yield SimpleNamespace(
        base=f"http://127.0.0.1:{server.server_port}/artifactory", cls=FakeArtifactory
    )
    server.shutdown()


def _upload(art, *argv, capsys):
    capsys.readouterr()
    code = cli(["results", *argv, "--base-url", art.base, "--repo", "repo"])
    return code, json.loads(capsys.readouterr().out.splitlines()[-1])


def test_offline_upload_puts_one_tgz_in_the_inbox(tmp_path, art, capsys):
    argv = ["--offline", "--upload", "--xml-dir", str(_xml(tmp_path)), "--artifact", f"id:{AID}",
            "--arch", "s390x", "--trigger-type", "fvt"]  # fmt: skip
    code, report = _upload(art, *argv, capsys=capsys)
    assert (code, report["status"]) == (0, "uploaded")
    ((method, path, auth),) = art.cls.seen
    assert path.startswith(
        f"/artifactory/repo/zsp/next/s390x/v2-results/inbox/{AID}/manual_"
    )
    assert path.endswith("-fvt.tgz") and auth == "Bearer tok"
    tgz = tmp_path / "got.tgz"
    tgz.write_bytes(art.cls.store[path])
    with offline.opened(tgz) as root:
        assert offline.check(root)["artifact_id"] == AID


def test_a_folder_upload_puts_bundle_json_last_and_retries_a_503(tmp_path, art, capsys):
    path = _offline(tmp_path, capsys=capsys)
    art.cls.fail = 1
    code, _ = _upload(art, "--upload", str(path), "--as-folder", capsys=capsys)
    puts = [p for m, p, _ in art.cls.seen]
    assert code == 0 and puts[0] == puts[1] and puts[-1].endswith("/bundle.json")
    assert puts[1].endswith("/results/junit.xml")


def test_upload_without_a_token_fails_clearly(tmp_path, art, capsys, monkeypatch):
    monkeypatch.delenv("ARTIFACTORY_TOKEN")
    monkeypatch.setenv("HOME", str(tmp_path))
    code, report = _upload(
        art, "--upload", str(_offline(tmp_path, capsys=capsys)), capsys=capsys
    )
    assert code == offline.FAILED and "no Artifactory token" in report["reason"]


def test_the_token_can_come_from_netrc(tmp_path, art, capsys, monkeypatch):
    monkeypatch.delenv("ARTIFACTORY_TOKEN")
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".netrc").write_text("machine 127.0.0.1 login me password pw\n")
    (tmp_path / ".netrc").chmod(0o600)
    assert (
        _upload(art, "--upload", str(_offline(tmp_path, capsys=capsys)), capsys=capsys)[
            0
        ]
        == 0
    )
    assert art.cls.seen[-1][2] == "Basic bWU6cHc="


@pytest.mark.parametrize(
    "where, status", [("processed", "ingested"), ("rejected", "rejected")]
)
def test_wait_reports_the_relays_outcome(tmp_path, art, capsys, where, status):
    path = _offline(tmp_path, capsys=capsys)
    with offline.opened(path) as root:
        meta, dig = offline.check(root), offline.digest(root)
    done = f"/artifactory/repo/zsp/next/s390x/v2-results/{where}/{AID}/{offline.bundle_name(meta, dig)}.tgz"
    art.cls.store[done] = b"x"
    art.cls.store[done + ".REJECTED.json"] = json.dumps(
        {"reason": "sha256 mismatch"}
    ).encode()
    code, report = _upload(art, "--upload", str(path), "--wait", "1", capsys=capsys)
    assert (code, report["status"]) == (0, status)
    assert report.get("reason", "") == (
        "sha256 mismatch" if where == "rejected" else ""
    )


@pytest.mark.parametrize(
    "branch, recorded",
    [("main", "main"), ("881a59d", ""), ("881a59d284dd6c225d09c416545766a1718a2f09", ""),
     ("release-1.2", "release-1.2"), ("cafe", "cafe")],
)  # fmt: skip
def test_a_sha_is_never_recorded_as_the_vllm_branch(tmp_path, branch, recorded):
    from spyre_clickhouse_ingest import vllm

    rows = vllm.extract_rows(
        str(VLLM_DATA / "results"), branch, "881a59d2", "", "0", "wf", 0
    )
    assert {r["head_branch"] for r in rows} == {recorded}
