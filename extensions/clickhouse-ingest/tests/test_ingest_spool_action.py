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

"""The ingest-spool composite action: run its real shell steps and check the trust gate."""

import subprocess
from pathlib import Path

import pytest
import yaml

ACTION = Path(__file__).resolve().parents[3] / ".github/actions/ingest-spool/action.yml"
DOC = yaml.safe_load(ACTION.read_text())
STEPS = {s["id"]: s for s in DOC["runs"]["steps"] if "id" in s}


def run_step(step_id, env, tmp_path):
    out, genv, summary = tmp_path / "out", tmp_path / "env", tmp_path / "sum"
    for f in (out, genv, summary):
        f.write_text("")
    full = {
        "PATH": "/usr/bin:/bin",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(out),
        "GITHUB_ENV": str(genv),
        "GITHUB_STEP_SUMMARY": str(summary),
        **env,
    }
    r = subprocess.run(
        ["bash", "-c", STEPS[step_id]["run"]], env=full, capture_output=True, text=True
    )
    kv = dict(
        line.split("=", 1) for line in out.read_text().splitlines() if "=" in line
    )
    return r, kv, genv.read_text()


BASE = {
    "TRUSTED_IN": "true",
    "TOKEN_IN": "tok-secret",
    "DIR_IN": "",
    "EVENT": "push",
    "HEAD_FORK": "",
    "REPO_FORK": "",
}


def test_a_trusted_push_run_enables_the_shared_tier(tmp_path):
    r, kv, genv = run_step("setup", BASE, tmp_path)
    assert kv["trusted"] == "true"
    assert (
        "CH_WRITE_SHARED_SPOOL_TRUSTED=1" in genv
        and "CH_WRITE_ARTIFACTORY_TOKEN=tok-secret" in genv
    )
    assert "::add-mask::tok-secret" in r.stdout


@pytest.mark.parametrize(
    "override",
    [
        {"EVENT": "pull_request"},
        {"EVENT": "pull_request_target"},
        {"HEAD_FORK": "true"},
        {"REPO_FORK": "true"},
        {"TRUSTED_IN": "false"},
        {"TRUSTED_IN": ""},
        {"TOKEN_IN": ""},
    ],
)
def test_an_untrusted_run_never_exports_the_token_or_the_flag(tmp_path, override):
    r, kv, genv = run_step("setup", {**BASE, **override}, tmp_path)
    assert kv["trusted"] == "false"
    assert "tok-secret" not in genv and "SHARED_SPOOL_TRUSTED" not in genv
    assert "tok-secret" not in r.stdout + r.stderr


def test_the_local_spool_is_always_set_up_even_when_untrusted(tmp_path):
    _, kv, genv = run_step("setup", {**BASE, "EVENT": "pull_request"}, tmp_path)
    assert (
        f"CH_WRITE_SPOOL={kv['spool-dir']}" in genv and Path(kv["spool-dir"]).is_dir()
    )


def test_finish_counts_pending_batches_and_warns(tmp_path):
    d = tmp_path / "sp"
    d.mkdir()
    (d / "a.json").write_text("{}")
    (d / "b.jsonl").write_text("{}")
    (d / "c.json.corrupt").write_text("{}")
    r, kv, _ = run_step("finish", {"DIR": str(d)}, tmp_path)
    assert kv["pending"] == "2" and "::warning::" in r.stdout


def test_finish_with_an_empty_spool_is_silent(tmp_path):
    d = tmp_path / "sp"
    d.mkdir()
    r, kv, _ = run_step("finish", {"DIR": str(d)}, tmp_path)
    assert kv["pending"] == "0" and "::warning::" not in r.stdout


def test_the_upload_runs_only_after_the_finish_step_found_something():
    up = [s for s in DOC["runs"]["steps"] if "upload-artifact" in s.get("uses", "")][0]
    assert "pending != '0'" in up["if"] and "finish" in up["if"]
    assert (
        up["uses"].split("@")[1].split(" ")[0]
        == "ea165f8d65b6e75b540449e92b4886f43607fa02"
    )


def test_the_token_input_is_never_echoed_to_the_log_unmasked():
    run = STEPS["setup"]["run"]
    assert 'echo "${TOKEN_IN}"' not in run and "echo $TOKEN_IN" not in run
