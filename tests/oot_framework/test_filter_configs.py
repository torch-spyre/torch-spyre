# Copyright 2025 The Torch-Spyre Authors.
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
"""Tests for ci_group folding in tests/oot_framework/utils/filter_configs.py."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "utils" / "filter_configs.py"


def _write(root: Path, rel: str, labels: str, group: str = "", extra: str = "") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    g = f"  ci_group: {group}\n" if group else ""
    p.write_text(
        f"test_suite_config:\n  labels: [{labels}]\n{g}{extra}"
        "  files:\n    - path: x.py\n"
    )


def _run(root: Path, *args: str):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config-dir",
            str(root),
            "--format",
            "matrix-json",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def _suites(root: Path, *args: str) -> list:
    r = _run(root, *args)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)["suite"]


def test_ungrouped_entry_is_unchanged(tmp_path):
    _write(tmp_path, "a/test_x_config.yaml", "unit, regression")
    (s,) = _suites(tmp_path, "--test-type", "regression")
    assert s["config"] == "a/test_x_config.yaml"
    assert s["configs"] == "a/test_x_config.yaml"
    assert s["report"] == "test_x_config"


def test_group_folds_into_one_entry(tmp_path):
    for n in ("a", "b", "c"):
        _write(tmp_path, f"d/test_{n}_config.yaml", "regression", "logging")
    _write(tmp_path, "d/test_z_config.yaml", "regression")
    suites = _suites(tmp_path, "--test-type", "regression")
    assert len(suites) == 2
    g = next(s for s in suites if s["configs"].count(" ") == 2)
    assert g["configs"] == (
        "d/test_a_config.yaml d/test_b_config.yaml d/test_c_config.yaml"
    )
    assert g["report"] == "group_d_logging"
    assert "3 suites" in g["name"]


def test_group_is_scoped_to_directory(tmp_path):
    _write(tmp_path, "d1/test_a_config.yaml", "regression", "g")
    _write(tmp_path, "d2/test_b_config.yaml", "regression", "g")
    suites = _suites(tmp_path, "--test-type", "regression")
    assert sorted(s["report"] for s in suites) == ["group_d1_g", "group_d2_g"]


def test_tier_filter_applies_before_grouping(tmp_path):
    _write(tmp_path, "d/test_a_config.yaml", "smoke, regression", "g")
    _write(tmp_path, "d/test_b_config.yaml", "regression", "g")
    (s,) = _suites(tmp_path, "--test-type", "smoke")
    assert s["configs"] == "d/test_a_config.yaml"


def test_runner_mismatch_is_an_error(tmp_path):
    _write(tmp_path, "d/test_a_config.yaml", "regression", "g")
    _write(tmp_path, "d/test_b_config.yaml", "regression", "g")
    rmap = tmp_path / "runners.yaml"
    rmap.write_text("d/test_b_config.yaml: spyre_pf_x2\n")
    r = _run(tmp_path, "--test-type", "regression", "--runner-map", str(rmap))
    assert r.returncode != 0
    assert "share runner and flags" in r.stderr


GLOBAL_F16 = "  global:\n    supported_dtypes:\n      - name: float16\n"
GLOBAL_F32 = "  global:\n    supported_dtypes:\n      - name: float32\n"


def test_differing_global_blocks_cannot_share_a_group(tmp_path):
    _write(tmp_path, "d/test_a_config.yaml", "regression", "g", extra=GLOBAL_F16)
    _write(tmp_path, "d/test_b_config.yaml", "regression", "g", extra=GLOBAL_F32)
    r = _run(tmp_path, "--test-type", "regression")
    assert r.returncode != 0
    assert "different 'global:' blocks" in r.stderr


def test_global_block_versus_none_cannot_share_a_group(tmp_path):
    _write(tmp_path, "d/test_a_config.yaml", "regression", "g", extra=GLOBAL_F16)
    _write(tmp_path, "d/test_b_config.yaml", "regression", "g")
    assert _run(tmp_path, "--test-type", "regression").returncode != 0


def test_identical_global_blocks_may_share_a_group(tmp_path):
    _write(tmp_path, "d/test_a_config.yaml", "regression", "g", extra=GLOBAL_F16)
    _write(tmp_path, "d/test_b_config.yaml", "regression", "g", extra=GLOBAL_F16)
    (s,) = _suites(tmp_path, "--test-type", "regression")
    assert s["configs"].count(" ") == 1


def test_paths_format_ignores_groups(tmp_path):
    _write(tmp_path, "d/test_a_config.yaml", "regression", "g")
    _write(tmp_path, "d/test_b_config.yaml", "regression", "g")
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config-dir",
            str(tmp_path),
            "--test-type",
            "regression",
            "--format",
            "paths",
        ],
        capture_output=True,
        text=True,
    )
    assert len(r.stdout.split()) == 2


def test_multi_card_matrix_matches_distributed_ci_groups():
    """test_multi_spyre's hand-written matrix must mirror distributed_tests ci_groups.

    That job lists its suites in _test_matrix.yaml rather than generating them, so a
    new distributed config, or a changed ci_group, would otherwise silently run
    ungrouped or not at all.
    """
    import yaml

    repo = Path(__file__).resolve().parents[2]
    wf = yaml.safe_load((repo / ".github/workflows/_test_matrix.yaml").read_text())
    listed = wf["jobs"]["test_multi_spyre"]["strategy"]["matrix"]["suite"]
    in_workflow = sorted(sorted(s["configs"].split()) for s in listed)

    groups: dict = {}
    for f in sorted((repo / "tests/configs/distributed_tests").glob("*.yaml")):
        tsc = yaml.safe_load(f.read_text())["test_suite_config"]
        key = tsc.get("ci_group") or f.stem
        groups.setdefault(key, []).append(f"distributed_tests/{f.name}")
    assert in_workflow == sorted(sorted(v) for v in groups.values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
