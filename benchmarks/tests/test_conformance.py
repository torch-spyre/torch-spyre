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

"""The identity blocks of benchmarks/ against the real ingest derivation, and the cost-hint lint."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "benchmarks" / "tools" / "check_identity_ids.py"

pytest.importorskip("spyre_bench")
pytest.importorskip("yaml")
pytest.importorskip("lxml")
pytest.importorskip("clickhouse_connect")


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("check_identity_ids", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_identity_ids"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def catalog(tmp_path):
    dst = tmp_path / "benchmarks"
    shutil.copytree(
        ROOT / "benchmarks",
        dst,
        ignore=shutil.ignore_patterns("tests", "tools", "__pycache__"),
    )
    return dst


# The first-shape ids of the ops before the identity blocks existed (report, kernels, total): a series that exists
# today must keep its id.
PRE_IDENTITY_IDS = {
    ("matmul", "report", ""): "ca0a80b2-3610-59d3-9af2-1ec73250ac0d",
    (
        "matmul",
        "kernel",
        "spyre_kernel_v1_fused_add#2@2",
    ): "e756b63d-6012-5617-b487-9d3c52cbee47",
    (
        "matmul",
        "kernel",
        "spyre_kernel_v1_fused_add#2@1",
    ): "a1ffc1ca-20c0-5d91-a36d-8089f1fbcb9f",
    ("matmul", "total", "Total"): "be8a6293-d023-525c-a8c5-6adc506940f6",
    ("pointwise_add", "report", ""): "a3f641e5-1e92-55b6-b530-dc2245fedb9f",
    (
        "pointwise_add",
        "kernel",
        "spyre_kernel_v1_fused_add#2@2",
    ): "78c588c1-cf56-54bd-9aec-9303029400ec",
    (
        "pointwise_add",
        "kernel",
        "spyre_kernel_v1_fused_add#2@1",
    ): "587dd475-2a20-54c6-ad6f-bc1254b4473b",
    ("pointwise_add", "total", "Total"): "9f8e8621-e431-5061-8ec8-9825e543d853",
    ("softmax", "report", ""): "c754d519-198d-5492-92c5-783284305245",
    (
        "softmax",
        "kernel",
        "spyre_kernel_v1_fused_add#2@2",
    ): "7256b70e-6c9a-5ba8-9003-f544fc58557d",
    (
        "softmax",
        "kernel",
        "spyre_kernel_v1_fused_add#2@1",
    ): "ad0d8be5-ba49-52ac-a5f6-a12fabd90a9b",
    ("softmax", "total", "Total"): "6d14e18d-2e0d-5660-a8f2-2b563daec032",
}


def golden_rows(path):
    rows = [ln.split("\t") for ln in Path(path).read_text().splitlines()[1:]]
    return {(r[1], r[2], r[4], r[3]): r[5] for r in rows}


def test_repo_catalog_conforms_to_the_ingest(tool):
    code, msgs = tool.check()
    assert code == 0, msgs


def test_existing_series_keep_their_ids(tool):
    golden = golden_rows(tool.GOLDEN)
    first_shape = "1_512_4096__4096_4096"
    for (name, row, kernel), want in PRE_IDENTITY_IDS.items():
        assert golden[(name, row, kernel, first_shape)] == want


def test_every_case_is_an_ingest_identity(tool):
    cases = tool.declared()
    assert cases
    assert {inp["profile"] for _, inp, _ in cases} == {"perf-kernel"}
    assert {inp["component"] for _, inp, _ in cases} == {"torch-spyre"}


def test_a_definition_that_changes_an_id_fails(tool, catalog, tmp_path):
    golden = tmp_path / "golden.tsv"
    assert tool.check(catalog, golden, update=True)[0] == 0
    assert tool.check(catalog, golden)[0] == 0
    ops = catalog / "ops" / "softmax.yaml"
    ops.write_text(ops.read_text().replace("value: op_benchmark", "value: other_mode"))
    code, msgs = tool.check(catalog, golden)
    assert code == 1
    assert any("softmax" in m for m in msgs)


def test_a_discriminator_the_parser_does_not_write_disagrees_with_the_ingest(
    tool, catalog, tmp_path
):
    ops = catalog / "ops" / "matmul.yaml"
    ops.write_text(
        ops.read_text().replace(
            "{key: config_name, value: ''}", "{key: config_name, value: cfg}"
        )
    )
    code, msgs = tool.check(catalog, tmp_path / "golden.tsv")
    assert code == 1
    assert any("the ingest writes" in m or "golden" in m for m in msgs)


@pytest.mark.parametrize(
    "edit, expect",
    [
        (
            (
                "  est_hours: 0.1\n",
                "  est_hours: 0.1\n  schedule: {preferred: nightly}\n",
            ),
            "cadence",
        ),
        (("  est_hours: 0.1\n", ""), "no est_hours"),
        (("    mode: ingest\n", "    mode: engine\n"), "needs identity"),
    ],
)
def test_cost_and_identity_lint(tool, catalog, edit, expect):
    ops = catalog / "ops" / "pointwise_add.yaml"
    ops.write_text(ops.read_text().replace(*edit))
    problems = tool.lint(catalog)
    assert any(expect in p for p in problems), problems


def test_a_schedule_annotation_is_refused(tool, catalog):
    project = catalog / "project.yaml"
    project.write_text(
        project.read_text().replace(
            "  name: ops-core\n",
            "  name: ops-core\n  annotations:\n    schedule.preferred: nightly\n",
        )
    )
    assert any("cadence" in p for p in tool.lint(catalog))


def test_the_checked_in_catalog_has_no_schedule_and_one_cost_hint(tool):
    assert tool.lint() == []
