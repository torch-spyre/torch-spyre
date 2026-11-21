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

"""Wherever the ClickHouse settings travel in CI, the v2-host and schema-login settings travel too."""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
ACTIONS = REPO / ".github/actions"
WORKFLOWS = REPO / ".github/workflows"
V2_SECRETS = (
    "CLICKHOUSE_V2_HOST", "CLICKHOUSE_V2_PORT", "CLICKHOUSE_V2_USER", "CLICKHOUSE_V2_PASSWORD"
)  # fmt: skip
V2_INPUTS = (
    "clickhouse-v2-host",
    "clickhouse-v2-port",
    "clickhouse-v2-user",
    "clickhouse-v2-pass",
)
SCHEMA_INPUTS = ("clickhouse-schema-user", "clickhouse-schema-pass")
VARS = ("CH_REQUIRE_V2_HOST", "CH_V2_DATABASES")
# Ingest actions whose script runs ensure-table DDL.
DDL_ACTIONS = {
    "ingest-hw-diagnostics-to-clickhouse",
    "ingest-model-ops-to-clickhouse",
    "ingest-pytorch-dispatch-to-clickhouse",
}
DDL_SCRIPTS = (
    "ingest_hw_diagnostics.py",
    "ingest_model_ops.py",
    "ingest_pytorch_dispatch.py",
)


def load(path):
    return yaml.safe_load(path.read_text())


def actions():
    for path in sorted(ACTIONS.glob("*/action.y*ml")):
        yield path.parent.name, load(path)


def workflows():
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        doc = load(path)
        yield path.name, doc, doc.get("on", doc.get(True)) or {}


def ingest_actions():
    """Actions that take ClickHouse settings: name -> declared inputs."""
    return {
        name: set(doc.get("inputs") or {})
        for name, doc in actions()
        if "clickhouse-host" in (doc.get("inputs") or {})
    }


def test_every_ingest_action_takes_the_new_settings_and_maps_them():
    found = ingest_actions()
    assert len(found) >= 5, found
    for name, inputs in found.items():
        want = (set(V2_INPUTS) if "clickhouse-db-v2" in inputs else set()) | (
            set(SCHEMA_INPUTS) if name in DDL_ACTIONS else set()
        )
        assert want <= inputs, f"{name} lacks {sorted(want - inputs)}"
        text = next(ACTIONS.glob(f"{name}/action.y*ml")).read_text()
        for i in want:
            assert f"${{{{ inputs.{i} }}}}" in text, f"{name} never reads {i}"


def test_every_use_of_an_ingest_action_passes_the_settings_it_takes():
    takes = {
        n: {i for i in inputs if i in V2_INPUTS + SCHEMA_INPUTS}
        for n, inputs in ingest_actions().items()
    }
    checked = 0
    for wf, doc, _ in workflows():
        for job_name, job in (doc.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                uses = step.get("uses", "")
                name = uses.rsplit("/", 1)[-1]
                if not uses.startswith("./.github/actions/") or name not in takes:
                    continue
                checked += 1
                given = set(step.get("with") or {})
                assert takes[name] <= given, (
                    f"{wf}:{job_name} uses {name} without {sorted(takes[name] - given)}"
                )
                if "clickhouse-db-v2" in given:
                    env = set(doc.get("env") or {}) | set(job.get("env") or {})
                    assert set(VARS) <= env, f"{wf}:{job_name} does not export {VARS}"
    assert checked >= 5


def test_reusable_workflows_declare_and_callers_forward_the_v2_secrets():
    forwarded = declared = 0
    for wf, doc, on in workflows():
        call = on.get("workflow_call") if isinstance(on, dict) else None
        names = set((call or {}).get("secrets") or {})
        if "CLICKHOUSE_HOST" in names:
            declared += 1
            want = set(V2_SECRETS) | (
                {f"{n}_DEV" for n in V2_SECRETS}
                if "CLICKHOUSE_HOST_DEV" in names
                else set()
            )
            assert want <= names, f"{wf} does not declare {sorted(want - names)}"
        for job_name, job in (doc.get("jobs") or {}).items():
            secrets = job.get("secrets")
            if isinstance(secrets, dict) and "CLICKHOUSE_HOST" in secrets:
                forwarded += 1
                missing = set(V2_SECRETS) - set(secrets)
                assert not missing, (
                    f"{wf}:{job_name} does not forward {sorted(missing)}"
                )
    assert declared >= 2 and forwarded >= 6


def env_blocks(doc):
    for job_name, job in (doc.get("jobs") or {}).items():
        yield job_name, job, job.get("env") or {}
        for step in job.get("steps") or []:
            yield job_name, job, step.get("env") or {}


def test_every_env_that_names_the_v2_database_names_the_v2_host():
    checked = 0
    for wf, doc, _ in workflows():
        for job_name, job, env in env_blocks(doc):
            if "CLICKHOUSE_DB_V2" not in env:
                continue
            checked += 1
            missing = set(V2_SECRETS) - set(env)
            assert not missing, f"{wf}:{job_name} env lacks {sorted(missing)}"
    assert checked >= 2


def test_jobs_that_run_a_ddl_script_directly_carry_the_schema_login():
    checked = 0
    for wf, doc, _ in workflows():
        for job_name, job in (doc.get("jobs") or {}).items():
            runs = " ".join(str(s.get("run", "")) for s in job.get("steps") or [])
            if not any(s in runs for s in DDL_SCRIPTS):
                continue
            checked += 1
            env = job.get("env") or {}
            assert {"CLICKHOUSE_SCHEMA_USER", "CLICKHOUSE_SCHEMA_PASSWORD"} <= set(
                env
            ), f"{wf}:{job_name} runs a DDL script without the schema login"
    assert checked >= 1


@pytest.mark.parametrize("wf", ["integration-tests.yaml", "push-to-clickhouse.yaml"])
def test_an_unset_prod_v2_secret_never_falls_through_to_the_dev_one(wf):
    text = (WORKFLOWS / wf).read_text()
    for name in V2_SECRETS:
        lines = [
            ln for ln in text.splitlines() if ln.strip().startswith(f"{name}: ${{{{")
        ]
        assert lines and all("toJSON(secrets." in ln for ln in lines), lines
