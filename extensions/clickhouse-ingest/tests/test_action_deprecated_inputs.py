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
    ("a v2 host takes the v2 login and port, never the deprecated ones", {**OLD, **NEW}, "new.host", "new", "2222"),
    ("a v2 host without a v2 port does not inherit the deprecated port", {**OLD, **NEW, "clickhouse-v2-port": ""},
     "new.host", "new", None),
    ("a deprecated host with a v2 login keeps its own port", {**OLD, "clickhouse-v2-user": "new",
     "clickhouse-v2-pass": "newpw"}, "old.host", "new", "1111"),
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
@pytest.mark.parametrize("v2", [{}, {"clickhouse-v2-pass": "newpw"}])
def test_a_v2_host_never_receives_the_deprecated_login(tmp_path, action, v2):
    inputs = {**OLD, "clickhouse-v2-host": "dev.host", **v2}
    rc, seen, out = ingest(action, tmp_path, inputs)
    assert rc != 0 and not seen and "login is missing" in out


@pytest.mark.parametrize("action", [XML, HW])
def test_a_v2_host_with_its_own_login_ignores_the_deprecated_one(tmp_path, action):
    rc, seen, _ = ingest(action, tmp_path, {**OLD, **NEW})
    assert rc == 0 and seen["CLICKHOUSE_USER"] == "new" and seen["CLICKHOUSE_PASS"] == "newpw"  # fmt: skip


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


def run_v2_artifact(tmp_path, inputs, fail=False):
    """The ensure step as GitHub runs it (`bash --noprofile --norc -eo pipefail`), with a fake interpreter."""
    action = ACTIONS / "v2-artifact/action.yml"
    s = next(
        x
        for x in yaml.safe_load(action.read_text())["runs"]["steps"]
        if x.get("env", {}).get("CH_V2_HOST_IN")
    )
    (tmp_path / "venv/bin").mkdir(parents=True)
    py = tmp_path / "venv/bin/python"
    py.write_text(
        '#!/bin/bash\nenv | grep -E "^CLICKHOUSE_" | sort > "$ENV_OUT"\nprintf "%s\\n" "$@" >> "$ENV_OUT.argv"\necho "{}"\nexit ${FAKE_RC:-0}\n'
    )
    py.chmod(0o755)
    env = env_of(s, inputs)
    env.update(VENV=str(tmp_path / "venv"), PATH="/usr/bin:/bin", ENV_OUT=str(tmp_path / "seen"), FAKE_RC="1" if fail else "0",
               GITHUB_OUTPUT=str(tmp_path / "o"), GITHUB_STEP_SUMMARY=str(tmp_path / "sum"),
               CLICKHOUSE_V2_HOST="evil.host", CLICKHOUSE_V2_USER="evil", CLICKHOUSE_V2_PASSWORD="evilpw",
               CLICKHOUSE_V2_PORT="9", CLICKHOUSE_PORT="9")  # fmt: skip
    r = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", s["run"]],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    seen_file = tmp_path / "seen"
    seen = dict(ln.split("=", 1) for ln in seen_file.read_text().splitlines()) if seen_file.exists() else {}  # fmt: skip
    return r, seen


def test_v2_artifact_overrides_the_callers_job_env(tmp_path):
    # Every multi-line input is empty here: repeat() must not return 1 and kill the -e step.
    r, seen = run_v2_artifact(
        tmp_path,
        {"artifact": "id:x", "arch": "x86_64", **NEW, "clickhouse-v2-port": ""},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert seen["CLICKHOUSE_V2_HOST"] == seen["CLICKHOUSE_HOST"] == "new.host"
    assert (
        seen["CLICKHOUSE_V2_USER"] == "new"
        and seen["CLICKHOUSE_V2_PASSWORD"] == "newpw"
    )
    assert "CLICKHOUSE_PORT" not in seen and "CLICKHOUSE_V2_PORT" not in seen


def test_v2_artifact_passes_each_non_empty_line_once(tmp_path):
    r, _ = run_v2_artifact(
        tmp_path,
        {
            "artifact": "id:x",
            "arch": "x86_64",
            **NEW,
            "tag": "a\n\n  b  \n",
            "source": "",
        },
    )
    assert r.returncode == 0, r.stdout + r.stderr
    argv = (tmp_path / "seen.argv").read_text().split("\n")
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--tag"] == ["a", "b"]
    assert "--source" not in argv


@pytest.mark.parametrize("v2", [{}, {"clickhouse-v2-pass": "newpw"}])
def test_v2_artifact_never_sends_the_deprecated_login_to_a_v2_host(tmp_path, v2):
    r, seen = run_v2_artifact(
        tmp_path,
        {
            "artifact": "id:x",
            "arch": "x86_64",
            **OLD,
            "clickhouse-v2-host": "dev.host",
            **v2,
        },
    )
    assert r.returncode != 0 and not seen and "login is missing" in r.stdout


@pytest.mark.parametrize("strict, code", [("false", 0), ("true", 1), ("", 1)])
def test_v2_artifact_fails_unless_strict_is_false(tmp_path, strict, code):
    r, _ = run_v2_artifact(
        tmp_path,
        {"artifact": "id:x", "arch": "x86_64", **NEW, "strict": strict},
        fail=True,
    )
    assert r.returncode == code, r.stdout + r.stderr
    assert "was not resolved or recorded" in r.stdout


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


def test_the_hw_action_run_blocks_interpolate_no_expression():
    for st in yaml.safe_load(HW.read_text())["runs"]["steps"]:
        assert "${{" not in st.get("run", ""), st.get("name")


@pytest.mark.parametrize(
    "table, ok",
    [
        ("my_table", True),
        ("spyre.t", False),
        ("t; DROP TABLE x", False),
        ("$(id)", False),
    ],
)
def test_the_hw_table_input_must_be_an_identifier(tmp_path, table, ok):
    rc, _, out = ingest(HW, tmp_path, {**NEW, "table": table})
    argv = (tmp_path / "seen.argv").read_text().split() if ok else []
    assert (rc == 0) is ok
    assert ("--table" in argv) is ok and (ok or "plain SQL identifier" in out)  # fmt: skip
