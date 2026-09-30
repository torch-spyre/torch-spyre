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

"""tests/oot_framework/utils/mark_retried.py: the whole-file retry label the ingest stores."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

HELPER = Path(__file__).resolve().parent / "oot_framework" / "utils" / "mark_retried.py"
_spec = importlib.util.spec_from_file_location("mark_retried", HELPER)
assert _spec is not None and _spec.loader is not None
mark_retried = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mark_retried)


def _write(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text(
        "<testsuites><testsuite name='pytest'>"
        "<testcase classname='c' name='test_a'>"
        "<properties><property name='tag' value='testtype__unit'/></properties></testcase>"
        "<testcase classname='c' name='test_b'><failure message='x'/></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return path


def _props(path):
    return {
        tc.get("name"): [(p.get("name"), p.get("value")) for p in tc.iter("property")]
        for tc in ElementTree.parse(path).getroot().iter("testcase")
    }


def test_every_testcase_is_marked_and_existing_properties_kept(tmp_path):
    path = _write(tmp_path)
    assert mark_retried.mark(str(path), "stall") == 2
    props = _props(path)
    assert props["test_a"] == [("tag", "testtype__unit"), ("result.retried", "stall")]
    assert props["test_b"] == [("result.retried", "stall")]
    # The outcome is untouched.
    assert (
        ElementTree.parse(path).getroot().find(".//testcase[@name='test_b']/failure")
        is not None
    )


def test_retries_nest_innermost_first_and_repeat_once(tmp_path):
    path = _write(tmp_path)
    for kind in ("signal", "stall", "pod", "stall"):
        mark_retried.mark(str(path), kind)
    assert dict(_props(path)["test_b"])["result.retried"] == "signal,stall,pod"


def test_an_unknown_kind_is_refused(tmp_path):
    path = _write(tmp_path)
    run = subprocess.run(
        [sys.executable, str(HELPER), "flaky", str(path)],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0 and "usage" in run.stderr
    assert "result.retried" not in path.read_text()
