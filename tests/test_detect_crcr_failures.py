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

"""Tests for .github/scripts/detect_crcr_failures.py.

The script is not a package module, so it is loaded by path. ``gh`` calls
are stubbed by replacing the module's ``subprocess`` binding; the totals
cross-check uses the real sibling parse_junit_xml_counts.count_testcases.
"""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"


def _load_by_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def detect() -> types.ModuleType:
    return _load_by_path("detect_crcr_failures", _SCRIPTS / "detect_crcr_failures.py")


def _write_xml(tmp_path: Path, name: str, testcases: str) -> Path:
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites><testsuite name='pytest'>"
        f"{testcases}</testsuite></testsuites>\n",
        encoding="utf-8",
    )
    return path


def _stub_gh(prs_json: str, detect: types.ModuleType, monkeypatch) -> None:
    """Replaces the detect module's subprocess binding with a fake gh."""

    class Result:
        def __init__(self, stdout: str, returncode: int = 0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    fake = types.SimpleNamespace(
        run=lambda args, **kw: Result(prs_json)  # noqa: ARG005
    )
    monkeypatch.setattr(detect, "subprocess", fake)


def test_failures_and_errors_are_extracted_with_detail(detect, tmp_path, monkeypatch):
    _write_xml(
        tmp_path,
        "test_torch.xml",
        "<testcase classname='c.SPYRE' name='test_a_spyre' time='1'>"
        "<failure message='first line&#10;second line'>traceback text</failure>"
        "</testcase>"
        "<testcase classname='c.SPYRE' name='test_b' time='2'>"
        "<error message='boom'>error text</error></testcase>"
        "<testcase classname='c' name='test_ok' time='3'/>"
        "<testcase classname='c' name='test_skip' time='0'><skipped/>"
        "</testcase>",
    )
    _stub_gh("[]", detect, monkeypatch)

    failures = detect.collect_failures(tmp_path)
    assert sorted(failures) == [
        "c.SPYRE::test_a_spyre",
        "c.SPYRE::test_b",
    ]
    a = failures["c.SPYRE::test_a_spyre"]
    assert a["kind"] == "failure"
    assert a["message"] == "first line"
    assert a["text"] == "traceback text"
    assert a["suite"] == "test_torch"
    assert a["digest"] == detect.skip_digest("c.SPYRE::test_a_spyre")
    assert failures["c.SPYRE::test_b"]["kind"] == "error"


def test_childless_failure_element_is_still_a_failure(detect, tmp_path):
    # ElementTree elements are falsy when childless; the parser must compare
    # against None, never use them as booleans.
    _write_xml(
        tmp_path,
        "test_torch.xml",
        "<testcase classname='c' name='test_a'><failure message='x'/></testcase>",
    )
    assert list(detect.collect_failures(tmp_path)) == ["c::test_a"]


def test_totals_reuse_the_reporting_path_counter(detect, tmp_path):
    _write_xml(
        tmp_path,
        "test_torch.xml",
        "<testcase classname='c' name='test_a'><failure message='x'/>"
        "</testcase>"
        "<testcase classname='c' name='test_b'/>"
        "<testcase classname='c' name='test_c'><skipped/></testcase>",
    )
    count_testcases = detect._load_count_testcases()
    totals = detect.collect_totals(tmp_path, count_testcases)
    assert totals == {
        "total": 3,
        "passed": 1,
        "failed": 1,
        "errors": 0,
        "skipped": 1,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_duplicate_testcases_collapse_to_one_unique_failure(
    detect, tmp_path, monkeypatch
):
    # Overlapping bundles list the same testcase twice; first (sorted path
    # order) wins and the unique list is what downstream triages.
    for name in ("a_test_torch.xml", "b_test_torch.xml"):
        _write_xml(
            tmp_path,
            name,
            "<testcase classname='c' name='test_a'><failure message='x'>"
            "text</failure></testcase>",
        )
    _stub_gh("[]", detect, monkeypatch)

    failures = detect.collect_failures(tmp_path)
    assert list(failures) == ["c::test_a"]


def test_unparsable_xml_is_skipped_with_a_warning(detect, tmp_path, capsys):
    (tmp_path / "broken.xml").write_text("<not-xml", encoding="utf-8")
    assert detect.collect_failures(tmp_path) == {}
    assert "broken.xml" in capsys.readouterr().err


def test_open_skip_coverage_unions_markers_and_picks_newest_pr(detect, monkeypatch):
    digest_a = detect.skip_digest("c::test_a")
    prs = json.dumps(
        [
            {
                "number": 7,
                "url": "https://example/pull/7",
                "headRefName": "crcr/auto-skip-old",
                "body": f"covered <!-- crcr-skip:{digest_a} -->",
            },
            {
                "number": 9,
                "url": "https://example/pull/9",
                "headRefName": "crcr/auto-skip-new",
                "body": "no markers",
            },
        ]
    )
    _stub_gh(prs, detect, monkeypatch)

    open_skip_pr, covered = detect.open_skip_coverage("o/r", "crcr-auto-skip")
    assert covered == {digest_a}
    assert open_skip_pr == {
        "number": 9,
        "url": "https://example/pull/9",
        "head_branch": "crcr/auto-skip-new",
    }


def test_gh_failure_is_fail_closed(detect, monkeypatch):
    class Result:
        stdout = ""
        stderr = "gh is down"
        returncode = 1

    monkeypatch.setattr(
        detect, "subprocess", types.SimpleNamespace(run=lambda a, **kw: Result())
    )
    with pytest.raises(SystemExit):
        detect.open_skip_coverage("o/r", "crcr-auto-skip")


def test_main_writes_json_outputs_and_github_output(detect, tmp_path, monkeypatch):
    _write_xml(
        tmp_path,
        "test_torch.xml",
        "<testcase classname='c' name='test_a'><failure message='x'>"
        "text</failure></testcase>",
    )
    digest_a = detect.skip_digest("c::test_a")
    prs = json.dumps(
        [
            {
                "number": 7,
                "url": "https://example/pull/7",
                "headRefName": "crcr/auto-skip-abc",
                "body": f"<!-- crcr-skip:{digest_a} -->",
            }
        ]
    )
    _stub_gh(prs, detect, monkeypatch)

    out_file = tmp_path / "failures.json"
    gh_output = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_output))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detect_crcr_failures.py",
            "--xml-dir",
            str(tmp_path),
            "--repo",
            "o/r",
            "--run-url",
            "https://example/runs/1",
            "--pytorch-sha",
            "abc123",
            "--output-file",
            str(out_file),
        ],
    )
    detect.main()

    payload = json.loads(out_file.read_text())
    assert payload["context"]["pytorch_sha"] == "abc123"
    assert payload["context"]["run_url"] == "https://example/runs/1"
    assert payload["totals"]["total"] == 1
    assert payload["open_skip_pr"]["number"] == 7
    (test,) = payload["tests"]
    assert test["test_id"] == "c::test_a"
    # The failure is covered by the open PR: nothing new to recover.
    assert test["already_in_skip_pr"] is True
    assert payload["outputs"]["new_failure_count"] == 0
    assert payload["outputs"]["has_failures"] is True
    assert payload["outputs"]["skip_needed"] is False

    written = gh_output.read_text()
    assert "has_failures=true" in written
    assert "skip_needed=false" in written
    assert "new_failure_count=0" in written


def test_mass_failure_thresholds_suppress_recovery(detect, tmp_path, monkeypatch):
    cases = "".join(
        f"<testcase classname='c' name='test_{i}'><failure message='x'/></testcase>"
        for i in range(6)
    )
    _write_xml(tmp_path, "test_torch.xml", cases)
    _stub_gh("[]", detect, monkeypatch)

    out_file = tmp_path / "failures.json"
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detect_crcr_failures.py",
            "--xml-dir",
            str(tmp_path),
            "--repo",
            "o/r",
            "--max-failures",
            "5",
            "--max-failure-ratio",
            "0.5",
            "--output-file",
            str(out_file),
        ],
    )
    detect.main()
    payload = json.loads(out_file.read_text())
    # 6 > 5 absolute, and 6/6 > 0.5 relative: a mass failure notifies but
    # never auto-skips.
    assert payload["outputs"]["mass_failure"] is True
    assert payload["outputs"]["skip_needed"] is False
    assert payload["outputs"]["new_failure_count"] == 6


def test_subprocess_binding_is_the_real_module_by_default(detect):
    # The stubbing strategy above replaces the module-level name; assert the
    # production binding is the real subprocess so the stub cannot silently
    # leak into other tests' module state.
    assert detect.subprocess is subprocess
