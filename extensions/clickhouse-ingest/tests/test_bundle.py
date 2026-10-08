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

"""`bundle init|seal|validate|ingest`: offline results, recorded later under their own run."""

import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from spyre_clickhouse_ingest import bundle, results, schema
from spyre_clickhouse_ingest.identity import RunId
from spyre_clickhouse_ingest.junit import RunCoordinates

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="s" tests="2"
 timestamp="2026-10-08T10:00:00+00:00"><testcase classname="a.b" name="test_x" time="1"/>
 <testcase classname="a.b" name="test_y" time="1"><failure message="boom"/></testcase>
 </testsuite></testsuites>"""


def _init(tmp_path, *extra, name="b"):
    out = tmp_path / name
    argv = ["init", "--artifact", f"id:{AID}", "--component", "spyre-inference"]
    argv += ["--arch", "s390x", "--test-type", "fvt", "--out", str(out), *extra]
    assert bundle.main(argv) == 0
    return out


def _sealed(tmp_path, *extra, xml=JUNIT, name="b"):
    out = _init(tmp_path, *extra, name=name)
    (out / "results" / "junit.xml").write_text(xml)
    assert bundle.main(["seal", str(out)]) == 0
    return out


def _meta(path):
    return json.loads((path / "bundle.json").read_text())


def _edit(path, **fields):
    meta = {**_meta(path), **fields}
    (path / "bundle.json").write_text(
        json.dumps({k: v for k, v in meta.items() if v is not None})
    )


def _rejected(path, code=bundle.REJECTED):
    with pytest.raises(bundle.BundleError) as err:
        bundle.check(path)
    assert err.value.code == code
    return str(err.value)


def test_schema_tiers_are_the_ddl_check_set():
    assert (
        set(bundle.schema()["properties"]["test_type"]["enum"])
        == schema.TEST_TYPE_VALUES
    )


def test_init_writes_a_manual_skeleton_that_seal_completes(tmp_path):
    out = _init(tmp_path, "--runner", "image=icr.io/x@sha256:" + "a" * 64)
    meta = _meta(out)
    assert meta["artifact_id"] == AID and "artifact" not in meta
    assert meta["run_key"].startswith("manual:") and meta["files"] == []
    assert meta["runner"]["image"].startswith("icr.io/x")
    _rejected(out)
    (out / "results" / "junit.xml").write_text(JUNIT)
    assert bundle.main(["seal", str(out)]) == 0
    meta = bundle.check(out)
    assert [f["path"] for f in meta["files"]] == ["results/junit.xml"]
    assert meta["started_at"] == "2026-10-08T10:00:00Z" and meta["ended_at"]


def test_validate_reads_a_tgz(tmp_path, capsys):
    out = _sealed(tmp_path)
    tgz = tmp_path / "b.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(out, arcname="b")
    assert bundle.main(["validate", str(tgz)]) == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["status"] == "valid" and report["run_id"] == bundle.run_id(_meta(out))


def test_a_tgz_escaping_its_directory_is_rejected(tmp_path):
    tgz = tmp_path / "evil.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        info = tarfile.TarInfo("../bundle.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    assert bundle.main(["validate", str(tgz)]) == bundle.REJECTED


@pytest.mark.parametrize(
    "change, why",
    [
        ({"colour": "red"}, "unknown key 'colour'"),
        ({"test_type": "nightly"}, "is not one of"),
        ({"run_key": "manual:me:not-a-uuid"}, "does not match"),
        ({"started_at": "2026-10-08 10:00"}, "with a zone"),
        ({"tag_family": "nightly-supply-chain"}, "manual bundle records verdicts only"),
        ({"artifact": "id:" + "0" * 8 + AID[8:]}, "different ids"),
        ({"artifact_id": None}, None),
    ],
)
def test_bad_bundle_json_is_rejected(tmp_path, change, why):
    out = _sealed(tmp_path)
    _edit(out, **change)
    if (
        why is None
    ):  # artifact_id dropped and no artifact spec: neither names the artifact
        assert "needs artifact_id or artifact" in _rejected(out)
    else:
        assert why in _rejected(out)


def test_files_must_match_the_directory(tmp_path):
    out = _sealed(tmp_path)
    (out / "results" / "extra.xml").write_text(JUNIT)
    assert "not in files[]" in _rejected(out)
    (out / "results" / "extra.xml").unlink()
    (out / "results" / "junit.xml").write_text(JUNIT.replace("boom", "bang"))
    assert "sha256 mismatch" in _rejected(out)
    (out / "results" / "junit.xml").unlink()
    assert "missing" in _rejected(out, bundle.INCOMPLETE)


def test_a_bundle_with_no_cases_records_nothing_and_is_rejected(tmp_path):
    out = _init(tmp_path)
    (out / "results" / "empty.xml").write_text(
        "<testsuites><testsuite name='s'/></testsuites>"
    )
    assert bundle.main(["seal", str(out)]) == bundle.REJECTED
    assert "no <testcase>" in _rejected(out)


def test_stf_isolated_bundle_fits_with_schema_version_and_files(tmp_path):
    """spyre-test-framework's ISOLATED upload, plus the two fields v1 requires."""
    out = tmp_path / "fvt"
    (out / "results").mkdir(parents=True)
    (out / "results" / "z1-junit.xml").write_text(JUNIT)
    stf = {
        "artifact": f"id:{AID}",
        "artifact_id": AID,
        "tag_family": "",
        "jenkins_run_key": "Spyre-Test/testing/Jenkinsfile.spyreinference#131",
        "run_url": "https://jenkins/job/Spyre-Test/job/testing/job/Jenkinsfile.spyreinference/131/",
        "arch": "s390x",
        "component": "spyre-inference",
        "test_type": "fvt",
        "started_at": "2026-10-08T09:00:00Z",
    }
    (out / "bundle.json").write_text(json.dumps(stf))
    assert "missing 'schema_version'" in _rejected(out)
    files = [
        {
            "path": "results/z1-junit.xml",
            "sha256": bundle.sha256(out / "results/z1-junit.xml"),
        }
    ]
    _edit(out, schema_version=1, files=files)
    meta = bundle.check(out)
    assert (
        bundle.bundle_name(meta)
        == "Spyre-Test_testing_Jenkinsfile.spyreinference_131-fvt"
    )


def test_a_jenkins_key_hashes_as_the_connected_run_and_a_manual_one_as_a_bundle(
    tmp_path,
):
    key = "Spyre-Test/testing/Jenkinsfile.spyreinference#131"
    meta = {"run_key": key, "arch": "s390x", "test_type": "svt"}
    connected = SimpleNamespace(run_id="", gha_run_id="", jenkins_run_key=key)
    assert bundle.run_id(meta) == RunId.for_args(connected, "", "s390x", "svt")
    manual = {**meta, "run_key": "manual:me:7d8121d7-e19f-4844-897b-f9b1fe876278"}
    assert bundle.run_id(manual) == RunId.derive(
        "bundle", manual["run_key"], "s390x", "svt"
    )
    assert RunCoordinates.source_and_external(connected, "") == ("jenkins", key)


class FakeClient:
    """Answers the two existence reads ingest makes, and counts the verdict check."""

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


@pytest.fixture
def online(monkeypatch):
    """ingest against a fake spyre_v2: `calls` holds each `results` argv."""
    state = SimpleNamespace(client=FakeClient(), calls=[], resolved=AID)
    from spyre_clickhouse_ingest import client, resolver

    monkeypatch.setattr(client.ClickHouse, "connect", lambda **kw: state.client)
    monkeypatch.setattr(
        resolver, "resolve",
        lambda spec, arch, **kw: SimpleNamespace(artifact_id=state.resolved) if state.resolved else None,
    )  # fmt: skip
    monkeypatch.setattr(results, "main", lambda argv: state.calls.append(argv))
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    return state


def _ingest(path, capsys, *extra):
    code = bundle.main(["ingest", str(path), "--strict", *extra])
    return code, json.loads(capsys.readouterr().out.splitlines()[-1])


def test_ingest_records_through_results_by_id_lookup_only(tmp_path, online, capsys):
    out = _sealed(tmp_path)
    code, report = _ingest(
        out, capsys, "--uploader", "jdoe", "--bundle-url", "https://art/b/"
    )
    assert (code, report["status"]) == (0, "ingested")
    argv = online.calls[0]
    pairs = dict(zip(argv[::2], argv[1::2]))
    assert pairs["--artifact"] == f"id:{AID}" and pairs["--lookup"] == "only"
    assert pairs["--run-id"] == bundle.run_id(_meta(out)) == report["run_id"]
    assert pairs["--run-url"] == "https://art/b/" and "--jenkins-run-key" not in argv
    props = [argv[i + 1] for i, a in enumerate(argv) if a == "--result-prop"]
    assert {
        "source=bundle",
        "uploader=jdoe",
        f"bundle_sha256={bundle.digest(out)}",
    } <= set(props)
    assert argv[-1] == "--strict" and "--dry-run" not in argv


def test_a_re_ingest_of_the_same_bundle_is_a_duplicate(tmp_path, online, capsys):
    out = _sealed(tmp_path)
    online.client.verdicts = [bundle.digest(out)]
    code, report = _ingest(out, capsys)
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
    out = _sealed(tmp_path)
    online.client.verdicts, online.client.files = verdicts, files
    code, report = _ingest(out, capsys)
    assert (code, report["status"]) == (bundle.REJECTED, "rejected") and why in report[
        "reason"
    ]
    assert online.calls == []


def test_a_partly_ingested_bundle_is_completed(tmp_path, online, capsys):
    out = _sealed(tmp_path)
    online.client.files = ["junit.xml"]
    assert _ingest(out, capsys)[1]["status"] == "ingested"


def test_an_unrecorded_artifact_is_rejected(tmp_path, online, capsys):
    online.resolved = ""
    code, report = _ingest(_sealed(tmp_path), capsys)
    assert code == bundle.REJECTED and "is not recorded" in report["reason"]


def test_an_untrusted_jenkins_key_is_rejected(tmp_path, online, capsys):
    out = _sealed(tmp_path, "--run-key", "Spyre/orchestrator#12")
    code, report = _ingest(out, capsys, "--trusted-job-prefix", "Spyre-Test/testing/")
    assert code == bundle.REJECTED and "not from a trusted job" in report["reason"]
    code, report = _ingest(out, capsys, "--trusted-job-prefix", "Spyre/")
    assert code == 0 and "--jenkins-run-key" in online.calls[0]


def test_a_verdict_that_did_not_land_is_a_retry(tmp_path, online, capsys):
    online.client.landed = False
    code, report = _ingest(_sealed(tmp_path), capsys)
    assert (code, report["status"]) == (bundle.FAILED, "failed")


def test_dry_run_reads_on_a_readonly_connection_and_writes_nothing(
    tmp_path, online, capsys
):
    code, report = _ingest(_sealed(tmp_path), capsys, "--dry-run")
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
