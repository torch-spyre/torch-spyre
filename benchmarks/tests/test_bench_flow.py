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

"""The bench-* make targets (older-engine tolerance, stub flow) and the shape of the nightly workflow."""

import json
import os
import regex as re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "bench-on-demand.yml"

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None, reason="make not installed"
)


def make(*args, env=None):
    base = {k: v for k, v in os.environ.items() if not k.startswith("BENCH_")}
    base["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{base['PATH']}"
    return subprocess.run(
        ["make", "-s", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        env={**base, **(env or {})},
        check=False,
    )


@pytest.fixture
def old_engine(tmp_path):
    """An engine that only knows `validate`, like the one installed before plan/run/verdict existed."""
    script = tmp_path / "old_engine.py"
    script.write_text(
        "import sys\n"
        "if '--help' in sys.argv[1:]:\n"
        "    print('usage: old VERB\\n\\n  validate     validate')\n"
        "else:\n"
        "    sys.exit(3)\n"
    )
    return f"{sys.executable} {script}"


@pytest.mark.parametrize(
    "target", ["bench-plan", "bench-run", "bench-verdict", "bench-report"]
)
def test_an_older_engine_is_noticed_and_tolerated(target, old_engine, tmp_path):
    base = [target, f"BENCH={old_engine}", f"BENCH_RESULTS_DIR={tmp_path / 'r'}"]
    out = make(*base, "BENCH_OPTIONAL=1")
    assert out.returncode == 0
    assert "skipped" in out.stderr
    strict = make(*base)
    assert strict.returncode == 2
    assert "Install spyre-bench: uv pip install" in strict.stderr
    assert make(*base, "BENCH_OPTIONAL=1", "BENCH_REQUIRED=1").returncode == 2


@pytest.mark.parametrize("value", ["1", "0", "yes"])
def test_bench_required_turns_the_notice_into_a_failure(value, old_engine, tmp_path):
    out = make(
        "bench-plan",
        f"BENCH={old_engine}",
        f"BENCH_REQUIRED={value}",
        f"BENCH_RESULTS_DIR={tmp_path / 'r'}",
    )
    assert out.returncode == 2
    assert "skipped" in out.stderr


def test_a_missing_engine_is_noticed_and_tolerated(tmp_path):
    args = [
        "bench-validate",
        "BENCH=no-such-engine-binary",
        f"BENCH_RESULTS_DIR={tmp_path}",
    ]
    out = make(*args, "BENCH_OPTIONAL=1")
    assert out.returncode == 0
    assert "not installed" in out.stderr
    assert make(*args).returncode == 2


def test_the_nightly_smoke_runs_the_whole_flow_without_cards(tmp_path):
    results = tmp_path / "res"
    out = make("bench-nightly-smoke", f"BENCH_RESULTS_DIR={results}")
    assert out.returncode == 0, out.stderr
    assert (
        "plan: run 25" in out.stdout
        and "run: 25 ok" in out.stdout
        and "verdict rows" in out.stdout
    )
    plan = json.loads((results / "plan" / "plan.json").read_text())
    assert (
        plan["stub"] and plan["purpose"] == "nightly" and plan["summary"]["run"] == 25
    )
    run = json.loads((results / "run.json").read_text())
    assert run["counts"]["ok"] == 25 and run["run_id"]
    assert not list(results.rglob("*.xml"))


def test_the_smoke_selection_plans_one_case(tmp_path):
    results = tmp_path / "res"
    out = make(
        "bench-nightly-smoke", f"BENCH_RESULTS_DIR={results}", "BENCH_SELECTION=smoke"
    )
    assert out.returncode == 0, out.stderr
    plan = json.loads((results / "plan" / "plan.json").read_text())
    assert [c["name"] for c in plan["cases"]] == ["matmul_prefill-n4096"]


def test_bench_run_plans_first_unless_given_a_plan():
    given = make("-n", "bench-run", "BENCH_PLAN=p.json")
    assert "--plan 'p.json'" in given.stdout and "spyre-bench plan" not in given.stdout
    assert "SENPERFORMANCE" in given.stdout
    chained = make("-n", "bench-run").stdout
    assert chained.index("spyre-bench plan") < chained.index("spyre-bench run")
    assert "--plan 'bench-results/plan/plan.json'" in chained
    assert "SENPERFORMANCE" not in make("-n", "bench-plan").stdout


def test_bench_variables_are_exported_to_the_engine():
    out = subprocess.run(
        ["make", "-s", "-C", str(ROOT), "-f", "Makefile", "-f", "-", "show"],
        input="show:\n\t@env | grep '^BENCH_' | cut -d= -f1 | sort\n",
        capture_output=True,
        text=True,
        check=False,
    )
    names = set(out.stdout.split())
    assert {
        "BENCH_PURPOSE", "BENCH_ENVIRONMENT", "BENCH_SELECTION", "BENCH_CASES", "BENCH_TIER", "BENCH_SUITES",
        "BENCH_SUBJECT", "BENCH_SUBJECT_ID", "BENCH_REUSE", "BENCH_PLAN", "BENCH_SET", "BENCH_INGEST",
        "BENCH_INVOKER", "BENCH_CLEAN", "BENCH_MAX_WAIT", "BENCH_MAX_HOLD", "BENCH_LEASE_FILE", "BENCH_QUIET",
    } <= names  # fmt: skip


def workflow():
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def test_the_workflow_is_manual_only():
    on = workflow()["on"]
    assert "schedule" not in on
    assert set(on["workflow_dispatch"]["inputs"]) >= {
        "selection",
        "clickhouse_env",
        "smoke",
        "ingest",
    }
    assert on["workflow_dispatch"]["inputs"]["clickhouse_env"]["default"] == "dev"
    assert "pull_request" not in on


def test_the_workflow_is_advisory_and_serialised():
    wf = workflow()
    assert (
        wf["concurrency"]["group"]
        and wf["concurrency"]["cancel-in-progress"] == "${{ inputs.pr_number != '' }}"
    )
    assert wf["jobs"]["bench"]["continue-on-error"] == "true"
    assert wf["permissions"] == {"contents": "read"}
    for step in wf["jobs"]["bench"]["steps"]:
        if step.get("id") in {"plan", "run", "collect", "ingest", "verdict"}:
            assert step["continue-on-error"] == "true", step["id"]


def _scalars(node, key=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _scalars(v, str(k))
    elif isinstance(node, list):
        for v in node:
            yield from _scalars(v, key)
    elif isinstance(node, str):
        yield key, " ".join(node.split())


_SECRET = re.compile(r"secrets\.(CLICKHOUSE_\w+)")
_EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.S)
PROD_GUARD = "== 'prod' &&"


def unguarded_prod_secret_expressions(text):
    """Every `${{ }}` expression that reads a non-_DEV CLICKHOUSE secret without the explicit prod guard."""
    bad = []
    for key, value in _scalars(yaml.safe_load(text)):
        if key == "PROD_SET":
            continue
        for expr in _EXPR.findall(value):
            expr = " ".join(expr.split())
            prod_refs = [n for n in _SECRET.findall(expr) if not n.endswith("_DEV")]
            if prod_refs and PROD_GUARD not in expr:
                bad.append(expr)
    return bad


def test_prod_credentials_are_reachable_only_through_an_explicit_prod():
    text = WORKFLOW.read_text()
    assert unguarded_prod_secret_expressions(text) == []
    assert "inputs.clickhouse_env || 'dev'" in text
    assert "vars.BENCH_CLICKHOUSE_ENV" not in text


def _mutated(tmp_path, extra_env=None, extra_run=None):
    wf = yaml.safe_load(WORKFLOW.read_text())
    step = wf["jobs"]["bench"]["steps"][0]
    if extra_env:
        step.setdefault("env", {}).update(extra_env)
    if extra_run:
        step["run"] = extra_run
    path = tmp_path / "wf.yml"
    path.write_text(yaml.safe_dump(wf))
    return path.read_text()


@pytest.mark.parametrize(
    "extra_env,extra_run",
    [
        (
            {"LEAK": "${{ secrets.CLICKHOUSE_HOST_DEV || secrets.CLICKHOUSE_HOST }}"},
            None,
        ),
        (
            None,
            "echo ${{ inputs.clickhouse_env == 'prod' && secrets.CLICKHOUSE_HOST || '' }}\n"
            "echo ${{ secrets.CLICKHOUSE_HOST }}\n",
        ),
    ],
    ids=["dev-prefix-exemption", "unguarded-in-shell-block"],
)
def test_the_prod_guard_check_catches_leaks(tmp_path, extra_env, extra_run):
    assert unguarded_prod_secret_expressions(_mutated(tmp_path, extra_env, extra_run))


def test_the_prod_guard_check_accepts_guarded_and_dev_only_expressions(tmp_path):
    ok = {
        "A": "${{ inputs.clickhouse_env == 'prod' && secrets.CLICKHOUSE_HOST || secrets.CLICKHOUSE_HOST_DEV }}",
        "B": "${{ secrets.CLICKHOUSE_HOST_DEV }}",
    }
    assert unguarded_prod_secret_expressions(_mutated(tmp_path, ok)) == []


def _collect_guard_snippet():
    wf = yaml.safe_load(WORKFLOW.read_text())
    run = next(
        s["run"] for s in wf["jobs"]["bench"]["steps"] if s.get("id") == "collect"
    )
    head = run.split('if [ "${bench_env}" = "prod" ]')[0]
    return head


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize(
    "env_value,do_ingest,rc",
    [
        ("prod", "true", 0),
        ("PROD", "true", 0),
        ("dev", "true", 0),
        ("staging", "true", 1),
        ("prod ", "true", 1),
        ("staging", "false", 0),
        ("prod ", "false", 0),
    ],
)
def test_the_clickhouse_env_guard_only_rejects_when_ingesting(env_value, do_ingest, rc):
    r = subprocess.run(
        ["bash", "-c", _collect_guard_snippet()],
        env={
            "PATH": os.environ["PATH"],
            "BENCH_CLICKHOUSE_ENV": env_value,
            "DO_INGEST": do_ingest,
        },
        capture_output=True,
        text=True,
    )
    assert r.returncode == rc, r.stdout + r.stderr


def test_results_leave_only_through_the_ingest_action():
    text = WORKFLOW.read_text()
    assert "upload-artifact" not in text
    assert "./.github/actions/ingest-xml-to-clickhouse" in text
    assert "gh run" not in text and "echo ${{ secrets" not in text


def test_the_runner_labels_are_the_ones_the_perf_job_uses():
    perf = (ROOT / ".github" / "workflows" / "_test_matrix.yaml").read_text()
    for label in ("x86_64", "spyre_pf_x1", "image_spyre_backend"):
        assert label in perf and label in WORKFLOW.read_text()


def test_the_smoke_mode_ingests_nothing_and_needs_no_cards():
    steps = {s.get("id") or s["name"]: s for s in workflow()["jobs"]["bench"]["steps"]}
    for key in ("plan", "run", "collect", "ingest", "verdict"):
        assert (
            "!inputs.smoke" in steps[key]["if"]
            or "collect.outputs.ready" in steps[key]["if"]
        ), key
    assert "ubuntu-latest" in workflow()["jobs"]["bench"]["runs-on"]


def stub_engine():
    return f"{sys.executable} {ROOT}/benchmarks/tools/bench_stub.py"


def run_json(tmp_path, stored):
    results = tmp_path / "res"
    results.mkdir()
    (results / "run.json").write_text(
        json.dumps({"run_id": "r", "stored_run_ids": stored})
    )
    return results


def test_the_verdict_fails_loudly_without_a_stored_run_id(tmp_path):
    results = run_json(tmp_path, [])
    out = make(
        "bench-verdict", f"BENCH_RESULTS_DIR={results}", f"BENCH={stub_engine()}"
    )
    assert out.returncode == 2
    assert "no stored run id known" in out.stderr


def test_the_verdict_uses_the_stored_id(tmp_path):
    results = run_json(tmp_path, ["s"])
    out = make(
        "bench-verdict", f"BENCH_RESULTS_DIR={results}", f"BENCH={stub_engine()}"
    )
    assert out.returncode == 0 and "verdict rows: 1" in out.stdout


def test_an_explicit_run_id_needs_no_run_json(tmp_path):
    (tmp_path / "res").mkdir()
    out = make(
        "bench-verdict",
        f"BENCH_RESULTS_DIR={tmp_path / 'res'}",
        f"BENCH={stub_engine()}",
        "BENCH_RUN_ID=abc",
    )
    assert out.returncode == 0


def test_a_verdict_with_no_rows_fails(tmp_path):
    empty = tmp_path / "empty_engine.py"
    empty.write_text(
        "import sys\n"
        "if '--help' in sys.argv:\n    print('usage\\n\\n  verdict   x')\n"
        'else:\n    print(\'{"summary": {"total": 0}}\')\n'
    )
    (tmp_path / "res").mkdir()
    out = make(
        "bench-verdict",
        f"BENCH_RESULTS_DIR={tmp_path / 'res'}",
        f"BENCH={sys.executable} {empty}",
        "BENCH_RUN_ID=abc",
    )
    assert out.returncode == 2 and "no verdict rows" in out.stderr


def test_prod_never_falls_back_to_dev():
    text = WORKFLOW.read_text()
    assert "refusing to fall back to dev" in text


MATRIX = ROOT / ".github" / "workflows" / "_test_matrix.yaml"
INTEGRATION = ROOT / ".github" / "workflows" / "integration-tests.yaml"


def perf_steps():
    steps = yaml.safe_load(MATRIX.read_text())["jobs"]["perf"]["steps"]
    return {s["id"]: s for s in steps if "id" in s}


def test_bench_mode_is_off_by_default_and_forwarded():
    for path in (MATRIX, INTEGRATION):
        spec = yaml.safe_load(path.read_text())[True]
        call = spec.get("workflow_call") or spec["workflow_dispatch"]
        # PyYAML (1.1) reads a bare `off` as False; the Actions parser reads the string.
        assert call["inputs"]["bench_mode"]["default"] in ("off", False)
    run_tests = yaml.safe_load(INTEGRATION.read_text())["jobs"]["run-tests"]
    assert run_tests["with"]["bench_mode"] == "${{ inputs.bench_mode }}"


def test_the_legacy_perf_ingest_is_skipped_only_in_replace_mode():
    steps = perf_steps()
    assert "inputs.bench_mode != 'replace'" in steps["ingest"]["if"]
    assert steps["ingest"]["with"]["workflow"] == "${{ github.workflow }}"
    assert "schema" not in steps["ingest"]["with"]  # the ingest action writes v2 only


@pytest.mark.parametrize("step", ["bench-mirror", "bench-ingest-dir"])
def test_the_mirror_is_advisory_outside_replace(step):
    s = perf_steps()[step]
    assert s["continue-on-error"] in (True, "${{ inputs.bench_mode != 'replace' }}")
    assert "side-by-side" in perf_steps()["bench-mirror"]["if"]


def test_side_by_side_files_a_connected_ci_run_in_its_own_series():
    with_ = perf_steps()["bench-ingest-dir"]["with"]
    assert with_["workflow"] == "${{ format('bench-{0}', github.workflow) }}"
    assert with_["channel-trust"] == "ci" and "schema" not in with_
    assert "run-id" not in with_ and "purpose" not in with_
    assert (
        "inputs.bench_mode == 'side-by-side'" in perf_steps()["bench-ingest-dir"]["if"]
    )
    mirror = perf_steps()["bench-mirror"]
    assert mirror["env"]["BENCH_INGEST"] == "0"
    assert mirror["env"]["BENCH_SUITE_FROM"] == "${{ env.RESULTS_DIR }}"


def test_replace_keeps_the_legacy_series_run_id_and_writes_v2_only():
    step = perf_steps()["bench-ingest"]
    assert "inputs.bench_mode == 'replace'" in step["if"]
    assert "continue-on-error" not in step
    with_ = step["with"]
    assert with_["workflow"] == "${{ github.workflow }}"
    assert with_["run-id"] == "${{ inputs.run_id }}" and "schema" not in with_


def test_the_banner_reads_the_ingest_step_of_the_active_mode():
    env = [
        s
        for s in yaml.safe_load(MATRIX.read_text())["jobs"]["perf"]["steps"]
        if s.get("name") == "Perf ingest banners"
    ][0]["env"]
    assert env["BENCH_INGEST_OUTCOME"] == (
        "${{ inputs.bench_mode == 'replace' && steps.bench-ingest.outcome || steps.bench-ingest-dir.outcome }}"
    )
    mirror = perf_steps()["bench-mirror"]
    assert (
        "BENCH_SELECTION" not in mirror["env"]
    )  # a selection would be unioned with BENCH_CASES
    assert 'BENCH_CASES="$(paste' in mirror["run"]


def test_a_bare_validate_and_plan_work_with_the_engine_installed(tmp_path):
    out = make("bench-validate", "bench-plan", f"BENCH_RESULTS_DIR={tmp_path / 'r'}")
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "r" / "plan" / "plan.json").is_file()


def test_the_default_environment_is_one_the_catalog_defines():
    text = (ROOT / "benchmarks" / "project.yaml").read_text()
    assert "kind: Environment\nmetadata:\n  name: one-device" in text
    assert "BENCH_ENVIRONMENT ?= one-device" in (ROOT / "mk" / "bench.mk").read_text()


def test_the_pre_commit_hook_stays_soft_without_the_engine():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        h for r in config["repos"] for h in r["hooks"] if h["id"] == "bench-validate"
    )
    assert "BENCH_OPTIONAL=1" in hook["entry"]
