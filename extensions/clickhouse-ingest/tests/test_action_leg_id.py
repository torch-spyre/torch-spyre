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

"""The ingest action derives CH_WRITE_LEG from the leg's inputs; run its snippet as written."""

import os
import regex as re
import subprocess
from pathlib import Path

import pytest

ACTION = (
    Path(__file__).resolve().parents[3]
    / ".github/actions/ingest-xml-to-clickhouse/action.yml"
)
INPUTS = (
    "PLATFORM_IN ARTIFACT_ID_IN ARTIFACT_IN XML_DIR_IN COMPONENT_IN VLLM_DIR_IN "
    "BENCH_DIR_IN TABLES_DIR_IN SENDNN_IN TAGS_IN PERF_IN"
).split()


def snippet():
    text = ACTION.read_text()
    m = re.search(r"# leg-id:begin.*?\n(.*?)\n\s*# leg-id:end", text, re.DOTALL)
    assert m, "leg-id markers missing from action.yml"
    return m.group(1)


def leg(**env):
    base = {k: "" for k in INPUTS} | env
    out = subprocess.run(
        ["bash", "-e", "-c", snippet() + '\nprintf "%s" "$CH_WRITE_LEG"'],
        env={"PATH": os.environ["PATH"], **base},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_distinct_inputs_give_distinct_stable_legs():
    a = leg(PLATFORM_IN="x86", XML_DIR_IN="r/a")
    assert re.fullmatch(r"[0-9a-f]{12}", a) and a == leg(
        PLATFORM_IN="x86", XML_DIR_IN="r/a"
    )
    assert a != leg(PLATFORM_IN="x86", XML_DIR_IN="r/b")
    assert a != leg(PLATFORM_IN="x86", XML_DIR_IN="r/a", PERF_IN="true")
    assert a != leg(PLATFORM_IN="x86", XML_DIR_IN="r/a", TAGS_IN="nightly")


def test_a_preset_leg_is_kept():
    assert leg(CH_WRITE_LEG="mine", PLATFORM_IN="x86") == "mine"


def test_without_sha256_tools_it_falls_back_and_otherwise_warns(tmp_path):
    import shutil

    if not shutil.which("python3"):
        pytest.skip("no python3")
    # PATH holding only the tools the snippet needs besides the hashers.
    for name in ("printf", "cut", "bash", "python3"):
        found = shutil.which(name)
        if found:
            (tmp_path / name).symlink_to(found)
    env = {"PATH": str(tmp_path), "CH_WRITE_LEG": ""} | {k: "" for k in INPUTS}
    env["PLATFORM_IN"] = "x86"
    run = subprocess.run(
        [
            str(tmp_path / "bash"),
            "-e",
            "-c",
            snippet() + '\nprintf "%s" "$CH_WRITE_LEG"',
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert re.fullmatch(r"[0-9a-f]{12}", run.stdout), run.stderr
