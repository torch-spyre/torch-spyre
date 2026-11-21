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

"""Tests for .github/scripts/report_test_failures_issue.py.

The script is not a package module, so it is loaded by path. All ``gh``
interaction goes through the module's run_gh, which tests stub per call.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"


@pytest.fixture(scope="module")
def report() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "report_test_failures_issue", _SCRIPTS / "report_test_failures_issue.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GhCall:
    def __init__(self, args, stdout=""):
        self.args = args
        self.stdout = stdout


def _stub_gh(report, monkeypatch, responses):
    """Responses: list of (predicate substring, stdout) consulted in order."""

    calls: list = []

    def fake_run_gh(args, check=True):
        calls.append(GhCall(args))
        joined = " ".join(args)
        for needle, stdout in responses:
            if needle in joined:
                return types.SimpleNamespace(stdout=stdout, stderr="")
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(report, "run_gh", fake_run_gh)
    return calls


def _write_failure_xml(tmp_path: Path, suite: str, classname: str, name: str):
    (tmp_path / f"{suite}.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites><testsuite name='pytest'>"
        f"<testcase classname='{classname}' name='{name}'>"
        "<failure message='unsupported op'>traceback</failure></testcase>"
        "</testsuite></testsuites>\n",
        encoding="utf-8",
    )


def _write_codeowners(tmp_path: Path) -> Path:
    path = tmp_path / "CODEOWNERS"
    path.write_text("* @owner\n", encoding="utf-8")
    return path


def test_bucketing_by_report_name_prefix(report):
    assert report.bucket_for("inductor-test_config")[0] == "inductor"
    assert report.bucket_for("profiler-test_profiler")[0] == "profiler"
    assert report.bucket_for("nn-test_embedding")[0] == "nn"
    assert report.bucket_for("optim-test_adam")[0] == "optim"
    assert report.bucket_for("test_torch")[0] == "eager-ops"
    assert report.bucket_for("test_linalg")[0] == "eager-ops"


def test_resolve_owners_last_match_wins(report, tmp_path):
    codeowners = tmp_path / "CODEOWNERS"
    codeowners.write_text(
        "# comment line\n"
        "* @fallback\n"
        "/torch_spyre/ops/ @alice @bob\n"
        "torch_spyre/_inductor/ @carol\n",
        encoding="utf-8",
    )
    assert report.resolve_owners(str(codeowners), "torch_spyre/ops/__init__.py") == [
        "@alice",
        "@bob",
    ]
    # Directory pattern owns everything beneath it.
    assert report.resolve_owners(str(codeowners), "torch_spyre/_inductor/foo.py") == [
        "@carol"
    ]
    assert report.resolve_owners(str(codeowners), "docs/source/x.md") == ["@fallback"]


def test_missing_codeowners_warns_and_returns_no_owners(report):
    assert report.resolve_owners("/no/such/CODEOWNERS", "torch_spyre/ops/x.py") == []


def test_render_table_carries_dedup_markers_and_respects_budget(report):
    tests = {f"c::test_{i}": ("suite", "failure", "msg") for i in range(5)}
    table = report.render_table("eager-ops", tests, budget=10000)
    for i in range(5):
        digest = report.digest("eager-ops", f"c::test_{i}")
        assert f"<!-- crcr-test:{digest} -->" in table
    # A tiny budget truncates rows and says so.
    cramped = report.render_table("eager-ops", tests, budget=200)
    assert "omitted" in cramped


def test_known_digests_reads_body_and_comments(report, monkeypatch):
    digest = report.digest("nn", "c::test_a")
    body = f"table <!-- crcr-test:{digest} -->"
    comments = [{"body": "<!-- crcr-test:000000000000 -->"}]
    _stub_gh(
        report,
        monkeypatch,
        [("issue view", json.dumps({"body": body, "comments": comments}))],
    )
    issue = {"number": 5}
    assert report.known_digests("o/r", issue) == {digest, "000000000000"}


def test_issue_number_from_url(report):
    assert report.issue_number_from_url("https://x/pull/12") == 12
    assert report.issue_number_from_url("https://x/pull/12/") == 12
    assert report.issue_number_from_url("not-a-url") is None


def test_main_creates_issues_and_writes_output_map(report, tmp_path, monkeypatch):
    _write_failure_xml(tmp_path, "test_torch", "c.SPYRE", "test_a")
    _write_failure_xml(tmp_path, "nn-test_embedding", "c.SPYRE", "test_b")
    # No open issues yet; creation returns the new issue URLs.
    calls = _stub_gh(
        report,
        monkeypatch,
        [
            ("issue list", "[]"),
            (
                "issue create",
                "https://github.com/o/r/issues/31",
            ),
        ],
    )
    out_file = tmp_path / "issue_map.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_test_failures_issue.py",
            "--xml-dir",
            str(tmp_path),
            "--repo",
            "o/r",
            "--run-url",
            "https://example/runs/1",
            "--pytorch-sha",
            "abc123def456",
            "--torch-spyre-sha",
            "fed654",
            "--report-artifact",
            "upstream-beta-reports-abc123def456",
            "--codeowners",
            str(_write_codeowners(tmp_path)),
            "--output-json",
            str(out_file),
        ],
    )
    report.main()

    result = json.loads(out_file.read_text())["buckets"]
    assert set(result) == {"eager-ops", "nn"}
    assert result["eager-ops"]["action"] == "created"
    assert result["eager-ops"]["issue_number"] == 31
    assert result["eager-ops"]["new_tests"] == ["c.SPYRE::test_a"]
    assert result["nn"]["issue_number"] == 31  # same stub response
    created = [c for c in calls if "issue create" in " ".join(c.args)]
    assert len(created) == 2  # one per bucket
    # The issue body cites owners and carries the bucket marker.
    body = created[0].args[created[0].args.index("--body") + 1]
    assert "@owner" in body
    assert "<!-- crcr-bucket:eager-ops -->" in body


def test_main_comments_only_new_failures(report, tmp_path, monkeypatch):
    _write_failure_xml(tmp_path, "test_torch", "c.SPYRE", "test_a")
    _write_failure_xml(tmp_path, "test_torch2", "c.SPYRE", "test_b")
    known = report.digest("eager-ops", "c.SPYRE::test_a")
    issue = {
        "number": 8,
        "url": "https://github.com/o/r/issues/8",
        "body": f"<!-- crcr-bucket:eager-ops --> ... <!-- crcr-test:{known} -->",
    }
    calls = _stub_gh(
        report,
        monkeypatch,
        [
            ("issue list", json.dumps([issue])),
            (
                "issue view",
                json.dumps({"body": issue["body"], "comments": []}),
            ),
        ],
    )
    out_file = tmp_path / "issue_map.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_test_failures_issue.py",
            "--xml-dir",
            str(tmp_path),
            "--repo",
            "o/r",
            "--run-url",
            "https://example/runs/1",
            "--pytorch-sha",
            "abc123def456",
            "--torch-spyre-sha",
            "fed654",
            "--report-artifact",
            "upstream-beta-reports-abc123def456",
            "--codeowners",
            str(_write_codeowners(tmp_path)),
            "--output-json",
            str(out_file),
        ],
    )
    report.main()

    result = json.loads(out_file.read_text())["buckets"]["eager-ops"]
    assert result["action"] == "commented"
    assert result["issue_number"] == 8
    assert result["new_tests"] == ["c.SPYRE::test_b"]
    commented = [c for c in calls if "issue comment" in " ".join(c.args)]
    assert len(commented) == 1
    body = commented[0].args[commented[0].args.index("--body") + 1]
    # The comment lists only the new failure, not the known one.
    assert "test_b" in body
    assert "test_a" not in body


def test_main_dry_run_touches_no_gh_and_maps_buckets(report, tmp_path, monkeypatch):
    _write_failure_xml(tmp_path, "test_torch", "c.SPYRE", "test_a")
    calls = _stub_gh(report, monkeypatch, [])
    out_file = tmp_path / "issue_map.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_test_failures_issue.py",
            "--xml-dir",
            str(tmp_path),
            "--repo",
            "o/r",
            "--run-url",
            "https://example/runs/1",
            "--pytorch-sha",
            "abc123def456",
            "--torch-spyre-sha",
            "fed654",
            "--report-artifact",
            "upstream-beta-reports-abc123def456",
            "--codeowners",
            str(_write_codeowners(tmp_path)),
            "--dry-run",
            "--output-json",
            str(out_file),
        ],
    )
    report.main()

    result = json.loads(out_file.read_text())["buckets"]["eager-ops"]
    assert result["action"] == "dry-run"
    assert result["issue_number"] is None
    assert result["new_tests"] == ["c.SPYRE::test_a"]
    assert calls == []


def test_main_without_failures_writes_empty_map(report, tmp_path, monkeypatch):
    (tmp_path / "test_torch.xml").write_text(
        "<testsuites><testsuite name='pytest'><testcase classname='c' "
        "name='ok'/></testsuite></testsuites>",
        encoding="utf-8",
    )
    calls = _stub_gh(report, monkeypatch, [])
    out_file = tmp_path / "issue_map.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_test_failures_issue.py",
            "--xml-dir",
            str(tmp_path),
            "--repo",
            "o/r",
            "--run-url",
            "https://example/runs/1",
            "--pytorch-sha",
            "abc123def456",
            "--torch-spyre-sha",
            "fed654",
            "--report-artifact",
            "upstream-beta-reports-abc123def456",
            "--output-json",
            str(out_file),
        ],
    )
    report.main()
    assert json.loads(out_file.read_text()) == {"buckets": {}}
    assert calls == []
