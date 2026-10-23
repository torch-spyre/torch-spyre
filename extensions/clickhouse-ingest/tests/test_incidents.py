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

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from spyre_clickhouse_ingest import incidents
from spyre_clickhouse_ingest.__main__ import main as cli
from spyre_clickhouse_ingest.schema import IncidentEvents

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
REPO = "torch-spyre/torch-spyre"
WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "record-failed-run-as-incident.yaml"
)


def _run(**kw):
    run = {
        "id": 900001,
        "run_attempt": 1,
        "name": "tests",
        "event": "push",
        "head_branch": "main",
        "head_sha": "abcdef0123456789",
        "status": "completed",
        "conclusion": "failure",
    }
    run.update(kw)
    return run


def _jobs():
    return [
        {"name": "lint", "conclusion": "success", "steps": []},
        {
            "name": "run-tests (x86_64)",
            "conclusion": "failure",
            "steps": [
                {"name": "checkout", "conclusion": "success"},
                {"name": "Run tests", "conclusion": "failure"},
            ],
        },
    ]


class FakeGitHub:
    def __init__(self, run, jobs):
        self.run, self.jobs, self.paths = run, jobs, []

    def get(self, path, **query):
        self.paths.append(path)
        return {"jobs": self.jobs} if path.endswith("/jobs") else self.run


class FakeClient:
    def __init__(self, present=True):
        self.present, self.inserted = present, []

    def command(self, sql, *a, **k):
        return 1 if self.present else 0

    def query(self, sql, *a, **k):
        return type("R", (), {"result_rows": [(c,) for c in IncidentEvents.columns]})()

    def insert(self, table, rows, column_names=None, database=None, **k):
        self.inserted.append((table, rows, column_names, database))


def test_a_failed_push_to_main_is_a_critical_event_with_names_only():
    row = incidents.build_row(REPO, _run(), _jobs(), now=NOW)
    assert (row["source"], row["severity"]) == ("gha", "critical")
    assert row["subject"] == f"{REPO}/tests"
    assert row["dedup_key"] == f"{REPO}|tests|main|run-tests (x86_64)"
    assert "failed jobs: run-tests (x86_64)" in row["summary"]
    assert "failed steps: Run tests" in row["summary"]
    assert row["evidence_url"] == f"https://github.com/{REPO}/actions/runs/900001"
    IncidentEvents.row(row)


def test_a_scheduled_failure_is_a_warning():
    row = incidents.build_row(REPO, _run(event="schedule"), _jobs(), now=NOW)
    assert row["severity"] == "warning"


@pytest.mark.parametrize("conclusion", ["success", "cancelled", "skipped", None])
def test_only_failure_and_timeout_are_recorded(conclusion):
    assert incidents.build_row(REPO, _run(conclusion=conclusion), _jobs()) is None


def test_a_run_still_in_progress_is_not_recorded():
    assert incidents.build_row(REPO, _run(status="in_progress"), _jobs()) is None


def test_the_same_failure_shares_a_dedup_key_across_runs():
    a = incidents.build_row(REPO, _run(id=1), _jobs())
    b = incidents.build_row(REPO, _run(id=2), _jobs())
    assert (
        a["dedup_key"] == b["dedup_key"]
        and a["props"]["run_id"] != b["props"]["run_id"]
    )


def test_secrets_and_control_characters_in_names_never_reach_a_row():
    token = "ghp_" + "a1B2c3D4" * 4
    jobs = [
        {
            "name": f"deploy {token}\nignore previous instructions",
            "conclusion": "failure",
            "steps": [
                {"name": f"curl https://u:{token}@host/x", "conclusion": "failure"}
            ],
        }
    ]
    row = incidents.build_row(REPO, _run(), jobs)
    blob = json.dumps(row, default=str)
    assert token not in blob and "u:ghp" not in blob and "\n" not in blob


def test_log_text_is_never_requested():
    gh = FakeGitHub(_run(), _jobs())
    incidents.record(gh, REPO, 900001, FakeClient(), "db", dry_run=False)
    assert all("/logs" not in p for p in gh.paths)


def test_record_writes_one_validated_row():
    client = FakeClient()
    out = incidents.record(
        FakeGitHub(_run(), _jobs()), REPO, 900001, client, "db", False
    )
    assert out.startswith("recorded")
    ((table, rows, cols, db),) = client.inserted
    assert (table, db, len(rows)) == ("incident_events", "db", 1)
    assert cols == list(IncidentEvents.columns)


def test_a_missing_table_records_nothing_and_does_not_raise():
    client = FakeClient(present=False)
    out = incidents.record(
        FakeGitHub(_run(), _jobs()), REPO, 900001, client, "db", False
    )
    assert "no incident_events table" in out and client.inserted == []


def test_a_passing_run_writes_nothing():
    client = FakeClient()
    out = incidents.record(
        FakeGitHub(_run(conclusion="success"), []), REPO, 900001, client, "db", False
    )
    assert "did not fail" in out and client.inserted == []


def test_dry_run_prints_the_row_without_a_client():
    out = incidents.record(FakeGitHub(_run(), _jobs()), REPO, 900001, None, "", True)
    assert json.loads(out)["source"] == "gha"


def test_the_cli_never_fails_the_caller(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert cli(["incidents", "gha-failure", "--repo", REPO, "--run-id", "1"]) == 0
    assert "not recorded: RuntimeError" in capsys.readouterr().out


def test_the_cli_survives_an_api_error_and_keeps_the_token_out_of_the_output(
    monkeypatch, capsys
):
    token = "ghp_" + "Z9y8X7w6" * 4
    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "v2")

    class Boom:
        def __init__(self, *_a, **_k):
            pass

        def get(self, *_a, **_k):
            raise ConnectionError(f"failed with Bearer {token}")

    monkeypatch.setattr(incidents, "GitHub", Boom)
    rc = cli(["incidents", "gha-failure", "--repo", REPO, "--run-id", "1", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "not recorded" in out and token not in out


def test_the_cli_dry_run_path(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setattr(incidents, "GitHub", lambda _t: FakeGitHub(_run(), _jobs()))
    assert (
        cli(["incidents", "gha-failure", "--repo", REPO, "--run-id", "9", "--dry-run"])
        == 0
    )
    assert '"source": "gha"' in capsys.readouterr().out


def test_workflow_skips_pull_requests_and_never_fails_the_followed_run():
    wf = yaml.safe_load(WORKFLOW.read_text())
    job = wf["jobs"]["record"]
    assert "pull_request" in job["if"] and "pull_request_target" in job["if"]
    assert wf[True]["workflow_run"]["types"] == ["completed"]
    assert wf["permissions"] == {"contents": "read", "actions": "read"}
    record = [s for s in job["steps"] if s["name"] == "Record incident"][0]
    assert record["continue-on-error"] is True and record["run"].rstrip().endswith(
        "|| true"
    )
