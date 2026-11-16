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
    out = make(target, f"BENCH={old_engine}", f"BENCH_RESULTS_DIR={tmp_path / 'r'}")
    assert out.returncode == 0
    assert "skipped" in out.stderr


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
    out = make(
        "bench-validate", "BENCH=no-such-engine-binary", f"BENCH_RESULTS_DIR={tmp_path}"
    )
    assert out.returncode == 0
    assert "not installed" in out.stderr


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
        "target",
        "prebaked_image",
        "run_id",
        "purpose",
        "engine_ref",
        "smoke",
        "ingest",
    }
    assert on["workflow_dispatch"]["inputs"]["target"]["default"] == "dev"
    assert "pull_request" not in on


def test_the_workflow_is_advisory_and_serialised():
    wf = workflow()
    assert (
        wf["concurrency"]["group"]
        and wf["concurrency"]["cancel-in-progress"] == "false"
    )
    assert wf["jobs"]["bench"]["continue-on-error"] == "true"
    assert wf["permissions"] == {"contents": "read"}
    for step in wf["jobs"]["bench"]["steps"]:
        if step.get("id") in {"plan", "run", "collect", "ingest", "verdict"}:
            assert step["continue-on-error"] == "true", step["id"]


def expressions(node, key=""):
    """(key, value) of every string in the parsed workflow."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from expressions(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from expressions(v, key)
    elif isinstance(node, str):
        yield key, " ".join(node.split())


def test_prod_credentials_are_reachable_only_through_an_explicit_prod():
    text = WORKFLOW.read_text()
    uses = [
        (k, v)
        for k, v in expressions(workflow())
        if "secrets.CLICKHOUSE_" in v and k != "PROD_SET"
    ]
    assert uses
    for key, value in uses:
        assert "== 'prod' &&" in value and "_DEV" in value, (key, value)
    assert "inputs.target || 'dev'" in text
    assert "vars.BENCH_CLICKHOUSE_ENV" not in text


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


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """A clone whose origin/main holds one commit, plus a second commit that is on no branch."""
    root = tmp_path_factory.mktemp("git")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.com", "HOME": str(root), "PATH": os.environ["PATH"]}  # fmt: skip

    def git(*args, cwd):
        return subprocess.run(
            ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
        ).stdout.strip()

    origin = root / "origin.git"
    git("init", "-q", "--bare", "-b", "main", str(origin), cwd=root)
    work = root / "work"
    git("clone", "-q", str(origin), str(work), cwd=root)
    git("commit", "-q", "--allow-empty", "-m", "on main", cwd=work)
    git("push", "-q", "origin", "HEAD:main", cwd=work)
    on_main = git("rev-parse", "HEAD", cwd=work)
    git("commit", "-q", "--allow-empty", "-m", "unmerged", cwd=work)
    off_main = git("rev-parse", "HEAD", cwd=work)
    git("fetch", "-q", "origin", cwd=work)
    return {"cwd": str(work), "on_main": on_main, "off_main": off_main}


def check_inputs(cwd=None, **given):
    """Run the plan job's validation step in bash with these inputs; returns (code, output)."""
    step = next(
        s for s in workflow()["jobs"]["plan"]["steps"] if s.get("id") == "check"
    )
    env = {
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_OUTPUT": os.devnull,
        "PATH": os.environ["PATH"],
    }
    env.update({k: "" for k in step["env"]})
    env["PROD_SET"] = "true"
    env.update(given)
    out = subprocess.run(
        ["bash", "-c", step["run"]],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )
    return out.returncode, out.stdout + out.stderr


@pytest.mark.parametrize(
    "given",
    [
        {},
        {"IMAGE_IN": "image_spyre_backend"},
        {"RUN_ID_IN": "0b2d3c4e-1a2b-4c3d-8e9f-0a1b2c3d4e5f"},
        {"ENGINE_IN": "feat/x.y"},
        {"TARGET_IN": "prod", "ENGINE_IN": "main"},
        {"TARGET_IN": "prod", "PURPOSE_IN": "on-demand"},
        {"TARGET_IN": "dev", "PURPOSE_IN": "nightly", "ACTOR_IN": "someone"},
    ],
)
def test_valid_inputs_pass_the_plan_check(given, repo):
    assert check_inputs(cwd=repo["cwd"], **given)[0] == 0


@pytest.mark.parametrize(
    "given, message",
    [
        ({"IMAGE_IN": "ubuntu-latest"}, "prebaked_image"),
        ({"IMAGE_IN": "image_x; echo"}, "prebaked_image"),
        ({"RUN_ID_IN": "0B2D3C4E-1A2B-4C3D-8E9F-0A1B2C3D4E5F"}, "run_id"),
        ({"RUN_ID_IN": "not-a-uuid"}, "run_id"),
        ({"ENGINE_IN": "x y"}, "engine_ref"),
        ({"ENGINE_IN": "$(id)"}, "engine_ref"),
        ({"ENGINE_IN": "a..b"}, "engine_ref"),
        ({"ENGINE_IN": "-x"}, "engine_ref"),
        ({"ENGINE_IN": "refs/pull/1/head"}, "engine_ref"),
        ({"ENGINE_IN": "feat/"}, "engine_ref"),
        ({"REF_IN": "a b"}, "ref is"),
        ({"REF_IN": "refs/pull/1/merge"}, "ref is"),
        ({"REF_IN": "--upload-pack=x"}, "ref is"),
        ({"SELECTION_IN": "a;b"}, "selection"),
        ({"TARGET_IN": "prod", "GITHUB_REF": "refs/heads/feat"}, "main only"),
        ({"TARGET_IN": "prod", "ENGINE_IN": "feat/x"}, "prod takes"),
        ({"TARGET_IN": "prod", "REF_IN": "feat/x"}, "prod takes"),
    ],
)
def test_bad_inputs_stop_the_workflow_before_any_run(given, message, repo):
    code, out = check_inputs(cwd=repo["cwd"], **given)
    assert code == 1 and message in out


def test_prod_takes_a_full_sha_only_when_it_is_on_main(repo):
    ok = check_inputs(
        cwd=repo["cwd"],
        TARGET_IN="prod",
        ENGINE_IN=repo["on_main"],
        REF_IN=repo["on_main"],
    )
    assert ok[0] == 0
    for field in ("ENGINE_IN", "REF_IN"):
        code, out = check_inputs(
            cwd=repo["cwd"], TARGET_IN="prod", **{field: repo["off_main"]}
        )
        assert code == 1 and "not on main" in out


def test_a_sha_that_is_not_fetched_cannot_reach_prod(repo):
    code, out = check_inputs(cwd=repo["cwd"], TARGET_IN="prod", ENGINE_IN="a" * 40)
    assert code == 1 and "not on main" in out


@pytest.mark.parametrize("purpose", ["nightly", "weekly"])
def test_baseline_purposes_on_prod_need_a_listed_dispatcher(purpose, repo):
    base = {"TARGET_IN": "prod", "PURPOSE_IN": purpose, "ACTOR_IN": "orchestrator"}
    code, out = check_inputs(cwd=repo["cwd"], **base)
    assert code == 1 and "BENCH_BASELINE_DISPATCHERS" in out
    assert check_inputs(cwd=repo["cwd"], DISPATCHERS_IN="someone-else", **base)[0] == 1
    assert (
        check_inputs(
            cwd=repo["cwd"], DISPATCHERS_IN="orchestrator-2,orchestrator", **base
        )[0]
        == 0
    )
    assert (
        check_inputs(cwd=repo["cwd"], DISPATCHERS_IN="orchestrator-2", **base)[0] == 1
    )


def test_the_prod_secret_gate_names_every_prod_secret():
    expr = workflow()["jobs"]["bench"]["env"]["PROD_SET"]
    for name in ("HOST", "PORT", "USER", "PASS", "DB", "DB_V2"):
        assert f"secrets.CLICKHOUSE_{name} != ''" in expr


def test_every_other_job_waits_for_the_checks():
    jobs = workflow()["jobs"]
    assert (
        "plan" in jobs["artifact_identity"]["needs"]
        and "plan" in jobs["bench"]["needs"]
    )
    assert "needs.plan.outputs.image" in jobs["bench"]["runs-on"]


def test_prebaked_image_runs_make_in_the_image_tree_and_run_id_is_the_engine_id():
    text = WORKFLOW.read_text()
    assert "/home/senuser/torch-spyre" in text and 'make -C "${TREE}"' in text
    assert "BENCH_RUN_ID: ${{ inputs.run_id }}" in text
    assert "purpose: ${{ inputs.purpose }}" in text


def test_prod_with_a_missing_secret_stops_in_the_plan_job(repo):
    code, out = check_inputs(cwd=repo["cwd"], TARGET_IN="prod", PROD_SET="false")
    assert code == 1 and "every prod ClickHouse secret" in out
    assert check_inputs(cwd=repo["cwd"], TARGET_IN="dev", PROD_SET="false")[0] == 0


def collect_step():
    return next(
        s for s in workflow()["jobs"]["bench"]["steps"] if s.get("id") == "collect"
    )


@pytest.mark.parametrize(
    "prod_set, refused", [("false", True), ("", True), ("true", False)]
)
def test_collect_refuses_prod_without_every_secret(prod_set, refused, tmp_path):
    env = {"PATH": os.environ["PATH"], "BENCH_CLICKHOUSE_ENV": "prod", "PROD_SET": prod_set, "TREE": str(tmp_path),
           "BENCH_RESULTS_DIR": str(tmp_path / "res"), "RUN_ID_IN": "", "CLICKHOUSE_HOST": "", "DO_INGEST": "true",
           "GITHUB_OUTPUT": os.devnull}  # fmt: skip
    out = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", collect_step()["run"]],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert ("refusing to fall back to dev" in out.stdout + out.stderr) is refused


def test_the_purpose_and_the_dispatcher_reach_the_scripts():
    wf = workflow()
    ingest = next(s for s in wf["jobs"]["bench"]["steps"] if s.get("id") == "ingest")
    assert ingest["with"]["purpose"] == "${{ inputs.purpose }}"
    plan_env = next(s for s in wf["jobs"]["plan"]["steps"] if s.get("id") == "check")[
        "env"
    ]
    assert plan_env["ACTOR_IN"] == "${{ github.triggering_actor }}"
    assert plan_env["DISPATCHERS_IN"] == "${{ vars.BENCH_BASELINE_DISPATCHERS }}"
    assert wf["jobs"]["bench"]["if"].count("needs.plan.result == 'success'") == 1


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
    assert steps["ingest"]["with"]["schema"] == "both"


@pytest.mark.parametrize("step", ["bench-mirror", "bench-ingest"])
def test_the_mirror_is_advisory_except_when_it_replaces_the_legacy_ingest(step):
    s = perf_steps()[step]
    assert s["continue-on-error"] == "${{ inputs.bench_mode != 'replace' }}"
    assert "side-by-side" in perf_steps()["bench-mirror"]["if"]


def test_the_mirror_ingests_under_its_own_series_next_to_the_legacy_one():
    with_ = perf_steps()["bench-ingest"]["with"]
    assert (
        with_["workflow"]
        == "${{ inputs.bench_mode == 'replace' && github.workflow || format('bench-{0}', github.workflow) }}"
    )
    assert with_["run-id"] == "${{ steps.bench-mirror.outputs.ingest_run_id }}"
    assert (
        "replace" in with_["schema"]
        and "v2" in with_["schema"]
        and "purpose" not in with_
    )
    mirror = perf_steps()["bench-mirror"]
    assert mirror["env"]["BENCH_INGEST"] == "0"
    assert mirror["env"]["BENCH_SUITE_FROM"] == "${{ env.RESULTS_DIR }}"
