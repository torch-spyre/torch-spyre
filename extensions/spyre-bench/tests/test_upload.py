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

"""spyre-bench upload: dry run by default, sealing through the ingest's offline module, the inbox transfer, exit codes.
python -m pytest -q -p no:cacheprovider tests/test_upload.py   (the real-ingest tests need an ingest with bench bundles)
"""

import json
import tarfile
import types
import urllib.error
from pathlib import Path

import pytest

from spyre_bench import cli, ingest_adapter, upload

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
ENV = "eeeeeeee-0000-5000-8000-000000000001"
RUN = "4b2b2c4e-1a7e-5c8e-9d2f-0a1b2c3d4e5f"


def result(name="decode"):
    reps = [{"run.repeat": i, "status": "ok", "samples": {"latency_ms": [10.0 + i]}, "placement": {"nodes": ["n-01"]}}
            for i in (1, 2, 3)]  # fmt: skip
    return {"apiVersion": "bench/v1alpha1", "kind": "Result", "metadata": {"name": f"{RUN}.{name}"},
            "status": {"case": {"component": "comp", "name": name, "tags": [], "disc": {}}, "subject": {"artifact_id": AID},
                       "environment": {"env_id": ENV, "fingerprint_version": 1, "fingerprint": {}, "labels": {}},
                       "repeats": reps, "run_props": {"run.trust": "internal", "run.purpose": "nightly"}}}  # fmt: skip


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "res" / "cases" / "decode-1"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(result()))
    (tmp_path / "res" / "run.json").write_text(json.dumps({"arch": "x86_64"}))
    return tmp_path / "res"


@pytest.fixture
def real_ingest():
    try:
        return ingest_adapter.offline()
    except ingest_adapter.IngestUnavailable as e:
        pytest.skip(str(e))


def sent(monkeypatch, reply=None):
    """Capture the transfer instead of making it."""
    calls = []

    def fake(path, args):
        calls.append((Path(path), args))
        return json.dumps(
            reply or {"status": "uploaded", "url": "https://art.example.com/r/x.tgz", "bundle_sha256": "s"}
        )

    monkeypatch.setattr(ingest_adapter.offline(), "upload", fake)
    return calls


def test_the_verb_is_listed():
    assert "upload" in cli.VERBS


def test_a_run_directory_is_sealed_checked_and_not_sent_without_apply(run_dir, real_ingest, monkeypatch, capsys):
    calls = sent(monkeypatch)
    assert upload.main([str(run_dir), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert (doc["status"], doc["kind"], doc["cases"], doc["repeats"]) == ("dry-run", "bench", 1, 3)
    assert doc["run_key"] == f"manual:spyre-bench:{RUN}" and doc["receipt"].startswith("RCPT-")
    assert doc["trust_label"] == "external" and calls == []
    assert Path(doc["bundle"]) == run_dir / "upload" / "bundle.tgz" and Path(doc["bundle"]).is_file()


def test_the_sealed_bundle_drops_the_engines_trust_claim_and_the_redacted_names(
    run_dir, real_ingest, monkeypatch, capsys
):
    sent(monkeypatch)
    assert upload.main([str(run_dir), "--json", "--redact-host", "--redact-user", "--seal-trust", "lab"]) == 0
    doc = json.loads(capsys.readouterr().out)
    with tarfile.open(doc["bundle"]) as t:
        meta = json.loads(t.extractfile("bundle.json").read())
        body = t.extractfile(next(m for m in t.getmembers() if m.name.startswith("results/"))).read().decode()
    assert "runner" not in meta and meta["trust"] == "lab" and '"run.trust"' not in body


def test_apply_puts_the_bundle_in_the_inbox_and_prints_the_receipt(run_dir, real_ingest, monkeypatch, capsys):
    calls = sent(monkeypatch)
    monkeypatch.setenv("ARTIFACTORY_BASE_URL", "https://art.example.com/artifactory")
    monkeypatch.setenv("ARTIFACTORY_V2_RESULTS_REPO", "results")
    monkeypatch.setenv("ARTIFACTORY_V2_RESULTS_ROOT", "{arch}/v2-results")
    assert upload.main([str(run_dir), "--apply", "--wait", "5"]) == 0
    out = capsys.readouterr().out
    assert "uploaded" in out and "RCPT-" in out
    ((path, args),) = calls
    assert path.name == "bundle.tgz" and (args.base_url, args.repo, args.root, args.wait) == (
        "https://art.example.com/artifactory",
        "results",
        "{arch}/v2-results",
        5,
    )


def test_a_rejected_outcome_exits_2_and_a_network_failure_exits_5(run_dir, real_ingest, monkeypatch, capsys):
    sent(monkeypatch, {"status": "rejected", "reason": "x"})
    assert upload.main([str(run_dir), "--apply"]) == 2
    monkeypatch.setattr(
        ingest_adapter.offline(), "upload", lambda *a: (_ for _ in ()).throw(urllib.error.URLError("down"))
    )
    assert upload.main([str(run_dir), "--apply"]) == 5
    assert "inbox:" in capsys.readouterr().out


def test_an_existing_bundle_is_sent_as_it_is(run_dir, real_ingest, monkeypatch, capsys):
    assert upload.main([str(run_dir), "--json"]) == 0
    bundle = json.loads(capsys.readouterr().out)["bundle"]
    calls = sent(monkeypatch)
    assert (
        upload.main(
            [bundle, "--apply", "--json", "--base-url", "https://art.example.com", "--repo", "r", "--root", "{arch}/x"]
        )
        == 0
    )
    assert calls[0][0] == Path(bundle)


def test_a_directory_without_results_and_a_missing_path_are_invalid(tmp_path, real_ingest, capsys):
    (tmp_path / "empty").mkdir()
    assert upload.main([str(tmp_path / "empty")]) == 2
    assert upload.main([str(tmp_path / "nope")]) == 2


def test_via_api_is_refused_with_the_reason(run_dir, capsys):
    assert upload.main([str(run_dir), "--via", "api"]) == 2
    assert "browser login only" in capsys.readouterr().err


def test_an_ingest_that_predates_bench_bundles_asks_for_an_upgrade(run_dir, monkeypatch, capsys):
    stale = types.SimpleNamespace()
    monkeypatch.setattr(ingest_adapter.importlib, "import_module", lambda name: stale)
    assert upload.main([str(run_dir)]) == 2
    assert "upgrade the ingest first" in capsys.readouterr().err
