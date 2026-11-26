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

"""The retry-matrix collector: .github/scripts/collect_retry_suites.py.

Loaded by path, like the other .github/scripts tests. No GitHub API and no network: every
case here drives the pure part -- turning a failed leg's `matrix.suite` descriptor into the
entry one retry job can run, and splitting the failures between the early and final waves.

The mapping these pin is a contract with _test_matrix.yaml: the retry job reads its runner
label, config paths, report name and flags out of the matrix entry, so a descriptor
normalized wrong sends a suite to the wrong pool or names the wrong report, and the retry
either cannot be scheduled or overwrites another suite's JUnit XML.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".github" / "scripts" / "collect_retry_suites.py"

# What the `test` job passes to run-test-suite for a PR run, and what nightly passes.
PR_FLAGS = "-v"
NIGHTLY_FLAGS = "--cpu-compile -vvv -s"
CONFIG_ROOT = "tests/configs/torch_spyre_tests"


@pytest.fixture(scope="module")
def collect():
    spec = importlib.util.spec_from_file_location("collect_retry_suites", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def single(
    collect, suite, config_root=CONFIG_ROOT, flags=PR_FLAGS, test_type="regression"
):
    return collect.normalize_single(suite, config_root, flags, test_type)


# ---------------------------------------------------------------------------
# Single-card descriptors -- mirror the `test` job's Derive report paths step.
# ---------------------------------------------------------------------------


def test_single_config_becomes_root_qualified_path(collect):
    entry = single(collect, {"name": "Eager ops", "config": "test_spyre_config.yaml"})
    assert entry["config_paths"] == f"{CONFIG_ROOT}/test_spyre_config.yaml"
    assert entry["report_name"] == "test_spyre_config.xml"
    assert entry["runner"] == "spyre_pf_x1"
    assert entry["parallel"] == "true"
    assert entry["test_type"] == "regression"


def test_nested_config_reports_under_its_basename(collect):
    # `basename "${CONFIG%.yaml}"`: the report is named after the file, not the subdir, so
    # it keeps matching the first attempt's artifact that the retry overwrites.
    entry = single(collect, {"name": "Inductor", "config": "inductor/test_ops.yaml"})
    assert entry["config_paths"] == f"{CONFIG_ROOT}/inductor/test_ops.yaml"
    assert entry["report_name"] == "test_ops.xml"


def test_bucket_runs_every_member_under_the_bucket_report(collect):
    # A bucket packs several short configs into one leg: all of them run in one invocation
    # and the report is named after the bucket, not after any one member.
    entry = single(
        collect,
        {
            "name": "Short configs 1",
            "config": "a_config.yaml",
            "configs": "a_config.yaml b_config.yaml c_config.yaml",
            "report": "bucket-1",
        },
    )
    assert entry["config_paths"] == " ".join(
        f"{CONFIG_ROOT}/{m}"
        for m in ["a_config.yaml", "b_config.yaml", "c_config.yaml"]
    )
    assert entry["report_name"] == "bucket-1.xml"


def test_suite_overrides_win_over_the_caller_defaults(collect):
    entry = single(
        collect,
        {
            "name": "Big suite",
            "config": "big_config.yaml",
            "runner": "spyre_pf_x4",
            "extra_test_flags": "-v -k not_slow",
        },
    )
    assert entry["runner"] == "spyre_pf_x4"
    assert entry["extra_test_flags"] == "-v -k not_slow"


def test_caller_flags_reach_the_retry_verbatim(collect):
    # Nightly's flags are multi-word; they must arrive as one value, not be re-split.
    entry = single(collect, {"name": "N", "config": "c.yaml"}, flags=NIGHTLY_FLAGS)
    assert entry["extra_test_flags"] == NIGHTLY_FLAGS


def test_config_root_follows_the_caller(collect):
    # model_ops_tests and the upstream suites scan their own subdir.
    entry = single(
        collect,
        {"name": "M", "config": "gpt.yaml"},
        config_root="tests/configs/model_ops_tests",
    )
    assert entry["config_paths"] == "tests/configs/model_ops_tests/gpt.yaml"


# ---------------------------------------------------------------------------
# Multi-card descriptors -- mirror the `test_multi_spyre` job.
# ---------------------------------------------------------------------------


def test_multi_card_suite_gets_the_two_card_pool_and_no_xdist(collect):
    entry = collect.normalize_multi(
        {
            "name": "Test Multi-Spyre Allreduce",
            "config": "distributed_tests/test_allreduce.yaml",
        },
        PR_FLAGS,
    )
    assert entry["runner"] == "spyre_pf_x2"
    # The distributed suites drive torchrun themselves.
    assert entry["parallel"] == "false"
    # tests/configs/, not the caller's config_root: that job spells its paths that way.
    assert (
        entry["config_paths"] == "tests/configs/distributed_tests/test_allreduce.yaml"
    )
    assert entry["report_name"] == "test_allreduce.xml"
    # That job sets no TORCH_SPYRE_TEST_TYPE, and the OOT framework defaults it to "".
    assert entry["test_type"] == ""


def test_multi_card_takes_the_multi_spyre_flags(collect):
    entry = collect.normalize_multi(
        {"name": "M", "config": "distributed_tests/x.yaml"}, ""
    )
    assert entry["extra_test_flags"] == ""


# ---------------------------------------------------------------------------
# Wave assignment -- every failed suite retried once, by exactly one wave.
# ---------------------------------------------------------------------------


def build(collect, singles=(), multis=(), exclude=()):
    return collect.build_matrix(
        list(singles),
        list(multis),
        CONFIG_ROOT,
        PR_FLAGS,
        "",
        "regression",
        exclude=exclude,
    )


def test_both_card_classes_share_one_matrix(collect):
    entries = build(
        collect,
        singles=[{"name": "Eager", "config": "eager.yaml"}],
        multis=[{"name": "Allreduce", "config": "distributed_tests/allreduce.yaml"}],
    )
    assert [e["runner"] for e in entries] == ["spyre_pf_x1", "spyre_pf_x2"]


def test_final_wave_skips_what_the_early_wave_claimed(collect):
    early = build(collect, singles=[{"name": "Eager", "config": "eager.yaml"}])
    # The final wave re-reads every descriptor, the early one's included.
    final = build(
        collect,
        singles=[
            {"name": "Eager", "config": "eager.yaml"},
            {"name": "Late", "config": "late.yaml"},
        ],
        exclude=[collect.entry_key(e) for e in early],
    )
    assert [e["name"] for e in final] == ["Late"]


def test_an_early_wave_that_claimed_nothing_excludes_nothing(collect):
    # A skipped or empty early wave must never cost a failed suite its retry.
    final = build(
        collect, singles=[{"name": "Eager", "config": "eager.yaml"}], exclude=[]
    )
    assert [e["name"] for e in final] == ["Eager"]


def test_a_repeated_descriptor_is_retried_once(collect):
    entries = build(
        collect,
        singles=[{"name": "Eager", "config": "eager.yaml"}] * 2,
    )
    assert len(entries) == 1


def test_same_named_suites_on_different_configs_both_retry(collect):
    # The exclusion key is the whole entry, so sharing a display name is not sharing a slot.
    entries = build(
        collect,
        singles=[
            {"name": "Ops", "config": "a.yaml"},
            {"name": "Ops", "config": "b.yaml"},
        ],
    )
    assert len(entries) == 2


def test_a_descriptor_with_no_runner_is_still_schedulable(collect):
    # An entry with an empty runner label would leave `runs-on` unsatisfiable and the leg
    # would sit unscheduled until the job timeout.
    entries = build(
        collect, singles=[{"name": "Eager", "config": "eager.yaml", "runner": ""}]
    )
    assert entries[0]["runner"] == "spyre_pf_x1"


# ---------------------------------------------------------------------------
# The CLI as _test_matrix.yaml spells it.
# ---------------------------------------------------------------------------


REQUIRED_ARGV = [
    "--repo=o/r",
    "--run-id=99",
    "--call-id=abc123",
    "--run-attempt=1",
    "--config-root=tests/configs/torch_spyre_tests",
]


@pytest.mark.parametrize("flags", [PR_FLAGS, NIGHTLY_FLAGS, ""])
def test_flag_values_that_start_with_a_dash_parse(collect, flags):
    # The whole point of the --opt=value form in the workflow: argparse reads a bare "-v"
    # or "--cpu-compile ..." as another option and exits 2, which took out both collector
    # jobs before the workflow was spelled this way.
    args = collect.build_parser().parse_args(
        REQUIRED_ARGV
        + [f"--extra-test-flags={flags}", f"--extra-test-flags-multi={flags}"]
    )
    assert args.extra_test_flags == flags
    assert args.extra_test_flags_multi == flags


def test_the_early_wave_waits_and_the_final_wave_does_not(collect):
    early = collect.build_parser().parse_args(REQUIRED_ARGV + ["--wait-for-failure"])
    final = collect.build_parser().parse_args(REQUIRED_ARGV + ["--exclude-matrix={}"])
    assert early.wait_for_failure and not final.wait_for_failure
    assert final.exclude_matrix == "{}"


# ---------------------------------------------------------------------------
# Artifact-name scoping -- one call's, one attempt's descriptors only.
# ---------------------------------------------------------------------------


def test_select_scopes_to_this_call_and_attempt(collect):
    artifacts = {
        "failed-suite-abc123-1-0": 1,
        "failed-suite-abc123-1-7": 2,
        "failed-suite-multi-abc123-1-3": 3,
        # Another call of this workflow in the same run (nightly calls it twice).
        "failed-suite-def456-1-0": 4,
        # An earlier attempt, whose artifacts outlive it.
        "failed-suite-abc123-2-0": 5,
        "test_spyre_config.xml": 6,
    }
    single, multi = collect.select(artifacts, "abc123-1-")
    assert sorted(single) == ["failed-suite-abc123-1-0", "failed-suite-abc123-1-7"]
    assert sorted(multi) == ["failed-suite-multi-abc123-1-3"]


# ---------------------------------------------------------------------------
# The couplings to _test_matrix.yaml that a one-sided edit would silently break.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def collector_steps():
    """The `run:` body of every step in _test_matrix.yaml that invokes the collector."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "_test_matrix.yaml").read_text()
    )
    # The invocation, not a mention: another job's comment names the script too.
    invocation = "python3 .github/scripts/collect_retry_suites.py"
    steps = [
        step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if invocation in step.get("run", "")
    ]
    # One per wave: the early (polling) collector and the final one.
    assert len(steps) == 2, steps
    return steps


@pytest.mark.parametrize("option", ["--extra-test-flags", "--extra-test-flags-multi"])
def test_the_workflow_passes_the_flag_options_as_opt_equals_value(
    collector_steps, option
):
    # argparse reads a bare "-v" as another option and exits 2, which took out both
    # collector jobs before the workflow was spelled this way.
    for step in collector_steps:
        assert f"{option}=" in step, f"{option} must use --opt=value form:\n{step}"
        assert f"{option} " not in step, f"{option} given a separate argument:\n{step}"


def test_the_fanin_job_name_matches_the_job_the_poller_waits_on(collect):
    # The poller stops when this job completes; a rename on either side leaves it polling
    # to its deadline, which holds up the final wave behind it.
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "_test_matrix.yaml").read_text()
    )
    declared = workflow["jobs"]["legs_complete"]["name"]
    # The job interpolates call_id where the script's template has its placeholder.
    assert declared.split("${{")[0] == collect.FANIN_JOB_NAME.split("{call_id}")[0]
    assert declared.endswith(")") and collect.FANIN_JOB_NAME.endswith(")")


def test_the_early_collector_tolerates_its_own_failure(collect):
    """The early wave is only a head start; its failures must not reach the caller's gate."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "_test_matrix.yaml").read_text()
    )
    assert workflow["jobs"]["collect_failed_suites_early"]["continue-on-error"] is True
    # The final collector must NOT: it is the only thing that reads every descriptor.
    assert "continue-on-error" not in workflow["jobs"]["collect_failed_suites"]
