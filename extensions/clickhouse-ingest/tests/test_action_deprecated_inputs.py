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

"""The ingest actions accept the old clickhouse-* inputs as deprecated fallbacks for clickhouse-v2-*."""

import subprocess
from pathlib import Path

import pytest
import regex as re

yaml = pytest.importorskip("yaml")

ACTIONS = Path(__file__).resolve().parents[3] / ".github/actions"
XML = ACTIONS / "ingest-xml-to-clickhouse/action.yml"
HW = ACTIONS / "ingest-hw-diagnostics-to-clickhouse/action.yaml"
INPUT = re.compile(
    r"\$\{\{\s*inputs\.([\w-]+)\s*(?:\|\|\s*inputs\.([\w-]+))?\s*(?:\|\|\s*'([^']*)')?\s*\}\}"
)
OLD = {"clickhouse-host": "old.host", "clickhouse-port": "1111", "clickhouse-user": "old",
       "clickhouse-pass": "oldpw", "clickhouse-db": "spyre", "schema": "v1"}  # fmt: skip
NEW = {"clickhouse-v2-host": "new.host", "clickhouse-v2-port": "2222", "clickhouse-v2-user": "new",
       "clickhouse-v2-pass": "newpw"}  # fmt: skip
STEP = {XML: "Ingest XML files into ClickHouse", HW: "Ingest into ClickHouse"}
GATE = {XML: "Gate on ClickHouse config and XML dir", HW: "Gate on ClickHouse config"}


def step(action, name):
    return next(
        s
        for s in yaml.safe_load(action.read_text())["runs"]["steps"]
        if s.get("name") == name
    )


def env_of(s, inputs):
    return {k: INPUT.sub(lambda m: inputs.get(m.group(1)) or inputs.get(m.group(2) or "") or m.group(3) or "", v)
            for k, v in s.get("env", {}).items()}  # fmt: skip


def gate(action, tmp_path, inputs):
    out = tmp_path / "out"
    out.write_text("")
    env = env_of(step(action, GATE[action]), {"xml-dir": "x", **inputs})
    env.update(GITHUB_OUTPUT=str(out), GITHUB_EVENT_NAME="push", PATH="/usr/bin:/bin")
    subprocess.run(
        ["bash", "-e", "-c", step(action, GATE[action])["run"]],
        env=env,
        check=True,
        cwd=tmp_path,
    )
    return "configured=true" in out.read_text()


def ingest(action, tmp_path, inputs, fake_exit=0):
    """Run the ingest step with a fake interpreter; return (exit code, env the library would see, stdout)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name in ("python", "python3"):
        fake = bin_ / name
        fake.write_text(
            '#!/bin/bash\nenv | grep -E "^(CLICKHOUSE_|INGEST_SCHEMA)" | sort > "$ENV_OUT"\nprintf "%s\\n" "$@" > "$ENV_OUT.argv"\nexit ${FAKE_EXIT:-0}\n'
        )
        fake.chmod(0o755)
    (tmp_path / "venv/bin").mkdir(parents=True)
    (tmp_path / "venv/bin/activate").write_text("")
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/activate").write_text("")
    s = step(action, STEP[action])
    env = env_of(s, {"xml-dir": "x", "gha-run-id": "1", **inputs})
    env.update(VENV=str(tmp_path / "venv"), PATH=f"{bin_}:/usr/bin:/bin", ENV_OUT=str(tmp_path / "seen"),
               ROOT=str(tmp_path), JSON_FILE="j.json", GITHUB_SERVER_URL="https://x", GITHUB_REPOSITORY="o/r",
               GITHUB_RUN_ID="1", RUN_ID="1", REPO="o/r", INGEST_SCHEMA="both", FAKE_EXIT=str(fake_exit))  # fmt: skip
    r = subprocess.run(
        ["bash", "-e", "-c", re.sub(r"\$\{\{.*?\}\}", "", s["run"])],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    seen = (tmp_path / "seen").read_text() if (tmp_path / "seen").exists() else ""
    return (
        r.returncode,
        dict(line.split("=", 1) for line in seen.splitlines()),
        r.stdout,
    )


@pytest.mark.parametrize("action", [XML, HW])
@pytest.mark.parametrize("name, inputs, host, user, port", [
    ("old inputs only", OLD, "old.host", "old", "1111"),
    ("v2 inputs only", NEW, "new.host", "new", "2222"),
    ("mixed: pairs resolve together", {**OLD, "clickhouse-v2-host": "new.host", "clickhouse-v2-pass": "newpw"},
     "new.host", "old", None),
])  # fmt: skip
def test_the_effective_connection_per_caller_shape(
    tmp_path, action, name, inputs, host, user, port
):
    assert gate(action, tmp_path, inputs) is True
    rc, seen, _ = ingest(action, tmp_path, inputs)
    assert rc == 0, name
    assert (seen["CLICKHOUSE_HOST"], seen["CLICKHOUSE_V2_HOST"]) == (host, host)
    assert (seen["CLICKHOUSE_USER"], seen["CLICKHOUSE_V2_USER"]) == (user, user)
    assert seen.get("CLICKHOUSE_PORT") == port
    assert seen["CLICKHOUSE_PASS"] == ("newpw" if user == "new" else "oldpw")
    # schema and the legacy database are ignored: always v2
    assert (
        seen.get("INGEST_SCHEMA", "v2") == "v2"
        and seen["CLICKHOUSE_DB"] == seen["CLICKHOUSE_DB_V2"] == "spyre_v2"
    )


@pytest.mark.parametrize("action", [XML, HW])
def test_an_empty_host_skips_cleanly(tmp_path, action):
    assert (
        gate(action, tmp_path, {"clickhouse-v2-user": "u", "clickhouse-v2-pass": "p"})
        is False
    )


@pytest.mark.parametrize("action", [XML, HW])
@pytest.mark.parametrize("host_input", ["clickhouse-host", "clickhouse-v2-host"])
def test_a_host_without_a_login_fails(tmp_path, action, host_input):
    inputs = {host_input: "h"}
    assert gate(action, tmp_path, inputs) is True
    rc, seen, _ = ingest(action, tmp_path, inputs)
    assert rc != 0 and not seen


@pytest.mark.parametrize("action", [XML, HW])
def test_a_non_v2_schema_warns_and_is_ignored(tmp_path, action):
    rc, seen, out = ingest(action, tmp_path, OLD)
    assert (
        rc == 0
        and "schema input is deprecated" in out
        and seen.get("INGEST_SCHEMA", "v2") == "v2"
    )


@pytest.mark.parametrize("action", [XML, HW])
def test_a_v2_user_with_only_the_old_password_is_not_a_mixed_pair(tmp_path, action):
    inputs = {
        "clickhouse-v2-host": "h",
        "clickhouse-v2-user": "new",
        "clickhouse-pass": "oldpw",
    }
    rc, seen, _ = ingest(action, tmp_path, inputs)
    assert rc != 0 and not seen


def test_xml_passes_strict_unless_strict_is_false(tmp_path):
    for strict, want in (("false", False), ("true", True), ("", True)):
        d = tmp_path / (strict or "default")
        d.mkdir()
        s = step(XML, STEP[XML])
        bin_ = d / "bin"
        bin_.mkdir()
        fake = bin_ / "python"
        fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ENV_OUT"\n')
        fake.chmod(0o755)
        (d / "venv/bin").mkdir(parents=True)
        (d / "venv/bin/activate").write_text("")
        env = env_of(s, {"xml-dir": "x", "strict": strict, **NEW})
        env.update(VENV=str(d / "venv"), PATH=f"{bin_}:/usr/bin:/bin", ENV_OUT=str(d / "argv"),
                   GITHUB_SERVER_URL="https://x", GITHUB_REPOSITORY="o/r", GITHUB_RUN_ID="1")  # fmt: skip
        subprocess.run(["bash", "-e", "-c", s["run"]], env=env, cwd=d, check=True)
        assert ("--strict" in (d / "argv").read_text().split()) is want


def test_v2_artifact_overrides_the_callers_job_env(tmp_path):
    action = ACTIONS / "v2-artifact/action.yml"
    s = next(
        x
        for x in yaml.safe_load(action.read_text())["runs"]["steps"]
        if x.get("env", {}).get("CH_V2_HOST_IN")
    )
    (tmp_path / "venv/bin").mkdir(parents=True)
    py = tmp_path / "venv/bin/python"
    py.write_text(
        '#!/bin/bash\nenv | grep -E "^CLICKHOUSE_" | sort > "$ENV_OUT"\necho "{}"\n'
    )
    py.chmod(0o755)
    env = env_of(
        s,
        {
            "clickhouse-v2-host": "new.host",
            "clickhouse-v2-user": "new",
            "clickhouse-v2-pass": "newpw",
        },
    )
    env.update(VENV=str(tmp_path / "venv"), PATH="/usr/bin:/bin", ENV_OUT=str(tmp_path / "seen"),
               GITHUB_OUTPUT=str(tmp_path / "o"), GITHUB_STEP_SUMMARY=str(tmp_path / "sum"),
               CLICKHOUSE_V2_HOST="evil.host", CLICKHOUSE_V2_USER="evil", CLICKHOUSE_V2_PASSWORD="evilpw",
               CLICKHOUSE_V2_PORT="9", CLICKHOUSE_PORT="9")  # fmt: skip
    r = subprocess.run(
        ["bash", "-c", s["run"]], env=env, cwd=tmp_path, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
    seen = dict(
        line.split("=", 1) for line in (tmp_path / "seen").read_text().splitlines()
    )
    assert seen["CLICKHOUSE_V2_HOST"] == seen["CLICKHOUSE_HOST"] == "new.host"
    assert (
        seen["CLICKHOUSE_V2_USER"] == "new"
        and seen["CLICKHOUSE_V2_PASSWORD"] == "newpw"
    )
    assert "CLICKHOUSE_PORT" not in seen and "CLICKHOUSE_V2_PORT" not in seen


def test_the_actions_write_v2_even_when_the_job_sets_ingest_schema_both(tmp_path):
    (tmp_path / "x").mkdir()
    rc, seen, _ = ingest(XML, tmp_path / "x", NEW)
    assert (
        rc == 0 and seen["INGEST_SCHEMA"] == "v2"
    )  # the harness injects INGEST_SCHEMA=both
    argv = (tmp_path / "x" / "seen.argv").read_text().split()
    assert argv[argv.index("--schema") + 1] == "v2"


@pytest.mark.parametrize("action", [XML, HW])
@pytest.mark.parametrize("strict, code", [("false", 0), ("true", 75), ("", 75)])
def test_parked_rows_exit_75_unless_strict_is_false(tmp_path, action, strict, code):
    rc, _, out = ingest(action, tmp_path, {**NEW, "strict": strict}, fake_exit=75)
    assert rc == code
    assert ("::warning title=ClickHouse::rows were parked" in out) is (code == 0)


@pytest.mark.parametrize("action", [XML, HW])
def test_strict_false_does_not_hide_other_failures(tmp_path, action):
    rc, _, _ = ingest(action, tmp_path, {**NEW, "strict": "false"}, fake_exit=1)
    assert rc == 1


def test_the_hw_script_passes_strict_unless_strict_is_false(tmp_path):
    for strict, want in (("false", False), ("true", True), ("", True)):
        d = tmp_path / (strict or "default")
        d.mkdir()
        ingest(HW, d, {**NEW, "strict": strict})
        assert ("--strict" in (d / "seen.argv").read_text().split()) is want
