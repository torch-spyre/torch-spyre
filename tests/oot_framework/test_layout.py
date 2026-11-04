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

"""Tests for oot_framework.utils.test_layout and filter_configs.py's grouping."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from oot_framework.utils import test_layout as L

FILTER = Path(__file__).resolve().parent / "utils" / "filter_configs.py"


def rec(key, t_files=None, runner="spyre_pf_x1", flags="", glob=None):
    return {
        "key": key,
        "dir": str(Path(key).parent),
        "runner": runner,
        "flags": flags,
        "global": glob,
        "files": t_files
        or [f"${{TORCH_DEVICE_ROOT}}/tests/{Path(key).stem.replace('_config', '')}.py"],
    }


def T(minutes, runs=10):
    """Timings as load_timings returns them, every config with `runs` recorded runs."""
    return {k: {"p50_min": float(v), "runs": runs} for k, v in minutes.items()}


POLICY = {
    **L.DEFAULT_POLICY,
    "target_job_minutes": 8.0,
    "max_members": 16,
}


def test_no_policy_means_every_config_alone():
    configs = [rec("inductor/test_a_config.yaml"), rec("inductor/test_b_config.yaml")]
    assert L.plan(configs, None, {}) == {
        "inductor/test_a_config.yaml": "",
        "inductor/test_b_config.yaml": "",
    }


def test_short_configs_share_a_bucket_and_long_ones_run_alone():
    configs = [
        rec("inductor/test_a_config.yaml"),
        rec("inductor/test_b_config.yaml"),
        rec("inductor/test_long_config.yaml"),
    ]
    timings = {
        "inductor/test_a_config.yaml": 1.0,
        "inductor/test_b_config.yaml": 2.0,
        "inductor/test_long_config.yaml": 9.0,
    }
    out = L.plan(configs, POLICY, T(timings))
    assert (
        out["inductor/test_a_config.yaml"] == out["inductor/test_b_config.yaml"] != ""
    )
    assert out["inductor/test_long_config.yaml"] == ""


def test_bucket_stays_under_the_time_cap():
    configs = [rec(f"inductor/test_{i}_config.yaml") for i in range(10)]
    timings = {c["key"]: 3.0 for c in configs}
    out = L.plan(configs, POLICY, T(timings))
    buckets = {}
    for k, b in out.items():
        if b:
            buckets.setdefault(b, []).append(k)
    for members in buckets.values():
        total = 3.0 * len(members)
        assert total + L.step_overhead(total) <= 8.0


def test_bucket_respects_max_members():
    configs = [rec(f"inductor/test_{i}_config.yaml") for i in range(40)]
    timings = {c["key"]: 0.01 for c in configs}
    out = L.plan(configs, {**POLICY, "max_members": 5}, T(timings))
    counts = {}
    for b in out.values():
        counts[b] = counts.get(b, 0) + 1
    assert max(counts.values()) <= 5


@pytest.mark.parametrize(
    "field,a,b",
    [
        ("runner", "spyre_pf_x1", "spyre_pf_x2"),
        ("flags", "", "-v"),
        ("global", None, {"supported_dtypes": [{"name": "float16"}]}),
    ],
)
def test_never_mixes_runner_flags_or_global(field, a, b):
    x, y = rec("inductor/test_a_config.yaml"), rec("inductor/test_b_config.yaml")
    x[field], y[field] = a, b
    out = L.plan([x, y], POLICY, T({x["key"]: 0.5, y["key"]: 0.5}))
    assert out[x["key"]] == "" and out[y["key"]] == ""


def test_never_puts_two_configs_on_one_test_file_together():
    shared = ["${TORCH_DEVICE_ROOT}/tests/inductor/test_shared.py"]
    configs = [
        rec("inductor/test_a_config.yaml", shared),
        rec("inductor/test_b_config.yaml", shared),
    ]
    out = L.plan(configs, POLICY, T({c["key"]: 0.5 for c in configs}))
    assert (
        out["inductor/test_a_config.yaml"] == ""
        or out["inductor/test_a_config.yaml"] != out["inductor/test_b_config.yaml"]
    )


def test_groups_differing_only_in_global_get_distinct_bucket_names():
    g1, g2 = (
        {"supported_dtypes": [{"name": "float16"}]},
        {"supported_dtypes": [{"name": "int32"}]},
    )
    configs = [
        rec("d/test_a_config.yaml", glob=g1),
        rec("d/test_b_config.yaml", glob=g1),
        rec("d/test_c_config.yaml", glob=g2),
        rec("d/test_e_config.yaml", glob=g2),
    ]
    out = L.plan(configs, POLICY, T({c["key"]: 0.5 for c in configs}))
    assert out["d/test_a_config.yaml"] == out["d/test_b_config.yaml"]
    assert out["d/test_c_config.yaml"] == out["d/test_e_config.yaml"]
    assert out["d/test_a_config.yaml"] != out["d/test_c_config.yaml"]


def test_config_sharing_a_test_file_with_another_config_runs_alone():
    configs = [rec(f"inductor/test_{c}_config.yaml") for c in "abc"]
    configs[0]["shared_file"] = True
    out = L.plan(configs, POLICY, T({c["key"]: 0.5 for c in configs}))
    assert out["inductor/test_a_config.yaml"] == ""
    assert (
        out["inductor/test_b_config.yaml"] == out["inductor/test_c_config.yaml"] != ""
    )


def test_exclude_keeps_a_directory_out_of_buckets():
    configs = [
        rec("distributed_tests/test_a_config.yaml"),
        rec("distributed_tests/test_b_config.yaml"),
    ]
    out = L.plan(
        configs,
        {**POLICY, "exclude": ["distributed_tests/"]},
        T({c["key"]: 0.5 for c in configs}),
    )
    assert set(out.values()) == {""}


def test_config_with_no_timing_runs_alone():
    configs = [
        rec("inductor/test_new_config.yaml"),
        rec("inductor/test_a_config.yaml"),
        rec("inductor/test_b_config.yaml"),
    ]
    out = L.plan(
        configs,
        POLICY,
        T({"inductor/test_a_config.yaml": 1.0, "inductor/test_b_config.yaml": 1.0}),
    )
    assert out["inductor/test_new_config.yaml"] == ""
    assert (
        out["inductor/test_a_config.yaml"] == out["inductor/test_b_config.yaml"] != ""
    )


@pytest.mark.parametrize("runs,grouped", [(0, False), (4, False), (5, True), (6, True)])
def test_config_runs_alone_until_it_has_min_runs(runs, grouped):
    configs = [rec("inductor/test_new_config.yaml"), rec("inductor/test_a_config.yaml")]
    timings = {
        **T({"inductor/test_a_config.yaml": 1.0}),
        **T({"inductor/test_new_config.yaml": 1.0}, runs=runs),
    }
    out = L.plan(configs, {**POLICY, "min_runs": 5}, timings)
    assert (out["inductor/test_new_config.yaml"] != "") is grouped


def test_plan_is_deterministic_whatever_the_input_order():
    configs = [rec(f"inductor/test_{i}_config.yaml") for i in range(30)]
    timings = {c["key"]: (i % 7) * 0.4 for i, c in enumerate(configs)}
    assert L.plan(configs, POLICY, T(timings)) == L.plan(
        list(reversed(configs)), POLICY, T(timings)
    )


@pytest.mark.parametrize(
    "bad",
    [
        '{"target_job_minutes": 0}',
        '{"max_members": 1}',
        '{"min_runs": 0}',
        '{"target_job_minutes": "x"}',
        '{"threshold_minutes": 4.0}',
        '{"unknown_knob": 3}',
        '{"exclude": "distributed_tests/"}',
        "not json",
    ],
)
def test_malformed_policy_is_an_error(tmp_path, bad):
    p = tmp_path / "policy.json"
    p.write_text(bad)
    with pytest.raises(L.LayoutError):
        L.load_policy(str(p))


def test_missing_policy_turns_grouping_off(tmp_path):
    assert L.load_policy(str(tmp_path / "absent.json")) is None


@pytest.mark.parametrize(
    "entry",
    [{"p50_min": -1, "runs": 9}, {"p50_min": 1.0}, {"p50_min": 1.0, "runs": "9"}, 3],
)
def test_malformed_timings_is_an_error(tmp_path, entry):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"configs": {"a.yaml": entry}}))
    with pytest.raises(L.LayoutError):
        L.load_timings(str(p))


def _write_tree(root, configs):
    for rel, body in configs.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)


def _cfg(path, glob=""):
    g = f"  global:\n    {glob}\n" if glob else ""
    return (
        f"test_suite_config:\n  labels: [regression]\n{g}  files:\n    - path: {path}\n"
    )


def _run(tmp_path, cfgdir, policy, timings, extra=()):
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps(policy))
    tim = tmp_path / "timings.json"
    tim.write_text(json.dumps({"configs": timings}))
    r = subprocess.run(
        [
            sys.executable,
            str(FILTER),
            "--config-dir",
            str(cfgdir),
            "--test-type",
            "regression",
            "--layout-policy",
            str(pol),
            "--timings",
            str(tim),
            "--format",
            "matrix-json",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    return (
        r.returncode,
        (json.loads(r.stdout)["suite"] if r.returncode == 0 else []),
        r.stderr,
    )


def test_filter_configs_folds_a_bucket_into_one_matrix_entry(tmp_path):
    cfgdir = tmp_path / "configs"
    _write_tree(
        cfgdir,
        {
            "inductor/test_a_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_a.py"
            ),
            "inductor/test_b_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_b.py"
            ),
            "inductor/test_big_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_big.py"
            ),
        },
    )
    rc, suites, err = _run(
        tmp_path,
        cfgdir,
        {"target_job_minutes": 8},
        {
            "inductor/test_a_config.yaml": {"p50_min": 1, "runs": 9},
            "inductor/test_b_config.yaml": {"p50_min": 1, "runs": 9},
            "inductor/test_big_config.yaml": {"p50_min": 12, "runs": 9},
        },
    )
    assert rc == 0, err
    bucket = [s for s in suites if s.get("configs")]
    assert len(bucket) == 1
    assert bucket[0]["configs"].split() == [
        "inductor/test_a_config.yaml",
        "inductor/test_b_config.yaml",
    ]
    assert bucket[0]["report"].startswith("bucket_")
    assert bucket[0]["name"] == "Inductor / A, B (2)"
    assert [s["config"] for s in suites if not s.get("configs")] == [
        "inductor/test_big_config.yaml"
    ]


def test_filter_configs_never_buckets_a_test_file_split_across_configs(tmp_path):
    cfgdir = tmp_path / "configs"
    shared = "${TORCH_DEVICE_ROOT}/tests/inductor/test_ops.py"
    _write_tree(
        cfgdir,
        {
            "inductor/test_ops_a_config.yaml": _cfg(shared),
            "inductor/test_ops_b_config.yaml": _cfg(shared),
            "inductor/test_x_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_x.py"
            ),
            "inductor/test_y_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_y.py"
            ),
        },
    )
    rc, suites, err = _run(
        tmp_path,
        cfgdir,
        {"target_job_minutes": 8},
        {
            k: {"p50_min": 0.5, "runs": 9}
            for k in (
                "inductor/test_ops_a_config.yaml",
                "inductor/test_ops_b_config.yaml",
                "inductor/test_x_config.yaml",
                "inductor/test_y_config.yaml",
            )
        },
    )
    assert rc == 0, err
    assert [s["configs"].split() for s in suites if s.get("configs")] == [
        ["inductor/test_x_config.yaml", "inductor/test_y_config.yaml"]
    ]


def test_filter_configs_without_a_policy_is_unchanged(tmp_path):
    cfgdir = tmp_path / "configs"
    _write_tree(
        cfgdir,
        {
            "inductor/test_a_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_a.py"
            ),
            "inductor/test_b_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_b.py"
            ),
        },
    )
    r = subprocess.run(
        [
            sys.executable,
            str(FILTER),
            "--config-dir",
            str(cfgdir),
            "--test-type",
            "regression",
            "--layout-policy",
            str(tmp_path / "absent.json"),
            "--format",
            "matrix-json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    suites = json.loads(r.stdout)["suite"]
    assert [s["config"] for s in suites] == [
        "inductor/test_a_config.yaml",
        "inductor/test_b_config.yaml",
    ]
    assert not any(s.get("configs") for s in suites)


def test_filter_configs_applies_on_the_trunk_scan_root(tmp_path):
    cfgdir = tmp_path / "configs"
    _write_tree(
        cfgdir,
        {
            "torch_spyre_tests/inductor/test_a_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_a.py"
            ),
            "torch_spyre_tests/inductor/test_b_config.yaml": _cfg(
                "${TORCH_DEVICE_ROOT}/tests/inductor/test_b.py"
            ),
        },
    )
    rc, suites, err = _run(
        tmp_path,
        cfgdir,
        {"target_job_minutes": 8},
        {
            "inductor/test_a_config.yaml": {"p50_min": 1, "runs": 9},
            "inductor/test_b_config.yaml": {"p50_min": 1, "runs": 9},
        },
    )
    assert rc == 0, err
    assert len([s for s in suites if s.get("configs")]) == 1
