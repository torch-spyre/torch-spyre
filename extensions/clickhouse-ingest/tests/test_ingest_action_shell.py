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

"""The ingest-xml-to-clickhouse action never lets an input reach its shell as code: no run: block
interpolates ${{ ... }}, and a hostile input value arrives at the ingest as one literal argument."""

import json
import subprocess
from pathlib import Path

import pytest
import regex as re

yaml = pytest.importorskip("yaml")

ACTION = (
    Path(__file__).resolve().parents[3]
    / ".github/actions/ingest-xml-to-clickhouse/action.yml"
)
HOSTILE = "x$(touch PWNED)`touch PWNED`\"; touch PWNED; '\nnext line $HOME"
INPUT = re.compile(r"\$\{\{\s*inputs\.([\w-]+)\s*(?:\|\|\s*'([^']*)')?\s*\}\}")


def steps():
    return yaml.safe_load(ACTION.read_text())["runs"]["steps"]


def test_no_run_block_interpolates_an_expression():
    for s in steps():
        assert "${{" not in s.get("run", ""), s.get("name")


def run_ingest_step(tmp_path, hostile):
    step = next(
        s for s in steps() if s.get("name") == "Ingest XML files into ClickHouse"
    )
    env = {
        k: INPUT.sub(lambda m: hostile.get(m.group(1), m.group(2) or ""), v)
        for k, v in step["env"].items()
    }
    env["VENV"] = str(tmp_path / "venv")
    (tmp_path / "venv/bin").mkdir(parents=True)
    (tmp_path / "venv/bin/activate").write_text("")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    fake = bin_ / "python"
    fake.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$ARGV_OUT"\n')
    fake.chmod(0o755)
    env.update(PATH=f"{bin_}:/usr/bin:/bin", ARGV_OUT=str(tmp_path / "argv"),
               GITHUB_SERVER_URL="https://example.test", GITHUB_REPOSITORY="o/r", GITHUB_RUN_ID="1")  # fmt: skip
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["bash", "-e", "-c", step["run"]], cwd=work, env=env, check=True)
    assert not (work / "PWNED").exists() and not (tmp_path / "PWNED").exists()
    return (tmp_path / "argv").read_bytes().split(b"\0")[:-1]


@pytest.mark.parametrize("field, flag", [("branch", "--branch"), ("workflow", "--workflow"),
                                         ("sha", "--sha"), ("trigger-type", "--trigger-type")])  # fmt: skip
def test_a_hostile_input_is_one_literal_argument(tmp_path, field, flag):
    argv = run_ingest_step(tmp_path, {"xml-dir": "xml", field: HOSTILE})
    assert argv[argv.index(flag.encode()) + 1] == HOSTILE.encode()


def test_a_hostile_tag_and_dir_stay_literal(tmp_path):
    argv = run_ingest_step(tmp_path, {"xml-dir": HOSTILE.replace("\n", " "), "tag": "a$(touch PWNED)b"})  # fmt: skip
    assert b"--xml-dir" in argv and json.dumps(argv[-1].decode())
