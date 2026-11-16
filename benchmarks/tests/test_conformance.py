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


# The report ids of every series the catalog declares, as the ingest stored them before the catalog existed
# (op rows keyed by shape, model rows by config and mode): a series that exists today must keep its id.
SERIES_IDS = """
attention||[[1, 32, 512, 128], [1, 32, 4096, 128], [1, 32, 4096, 128]]|op_benchmark 1deb8f5d-6628-5b2f-ac94-362ffd6569a8
attention||[[4, 32, 1, 128], [4, 32, 4096, 128], [4, 32, 4096, 128]]|op_benchmark ecbc75e9-9744-5976-a36b-4e1a038fe950
attention_gqa||[[1, 32, 512, 128], [1, 8, 4096, 128], [1, 8, 4096, 128]]|op_benchmark 877de30a-7d96-544e-a417-dd6b6e356361
attention_gqa||[[4, 32, 1, 128], [4, 8, 4096, 128], [4, 8, 4096, 128]]|op_benchmark bf575f36-61cc-5327-b852-c53279714a52
layernorm||[[1, 512, 4096]]|op_benchmark accc7627-50b1-54e0-8dce-447e379a936e
matmul||[[1, 512, 4096], [4096, 1024]]|op_benchmark 6c2336a8-2c6b-584c-bfe9-804db3207341
matmul||[[1, 512, 4096], [4096, 12800]]|op_benchmark 9cdac7d4-93b8-572f-8caa-5a76c9e40822
matmul||[[1, 512, 4096], [4096, 4096]]|op_benchmark c0994a96-44c0-5069-ab2d-924354712344
matmul||[[4, 1, 4096], [4096, 1024]]|op_benchmark bbb198c2-f835-538d-8e52-f8ede09b41df
matmul||[[4, 1, 4096], [4096, 12800]]|op_benchmark 3c086bfe-db98-51f8-a661-908f073dc177
matmul||[[4, 1, 4096], [4096, 4096]]|op_benchmark 522a7bbb-f406-5692-a260-c6ec42dfb698
mlp||[[1, 512, 4096]]|op_benchmark 582802b0-fb9e-542e-998b-dad64b1b81a4
mlp||[[4, 1, 4096]]|op_benchmark 2b762d5f-e8eb-5056-a6fd-da525f286b3b
mlp-linear-kn-t||[[1, 512, 4096]]|op_benchmark ff8cff94-39b8-5e65-84eb-2a930cfe6467
mlp-linear-kn-t||[[4, 1, 4096]]|op_benchmark 33a91ea9-df47-554a-a0e6-f52fffd14ba2
pointwise_add||[[1, 512, 4096], [1, 512, 4096]]|op_benchmark 9b763db9-f0f2-55e5-8701-6e7ec800bd38
pointwise_add||[[4, 1, 4096], [4, 1, 4096]]|op_benchmark d3f28b8a-33d8-50b8-be4c-0684594a3537
softmax||[[1, 512, 4096]]|op_benchmark d7f2cdc7-5745-5dca-8944-3a038bc2589b
softmax||[[4, 1, 4096]]|op_benchmark a6ce2409-dfd4-57d7-9178-666ccfced973
granite|full_model||decode_only 57080893-ff44-5133-92a7-21f6585bdeb5
granite|full_model||generate 230277c7-0a03-536a-92bb-a31241448dc9
granite|full_model||prefill_only c214d6b3-cf67-5f71-81ff-e7db0db942df
granite|one_block||decode_only 85cf3877-f161-568b-afb4-614228b16109
granite|one_block||generate 8b365796-bf8a-507a-a97f-59dedff2926d
granite|one_block||prefill_only 2103423c-ed79-5f3b-8041-dfa54ee6e313
"""


def series_ids():
    out = {}
    for line in SERIES_IDS.strip().splitlines():
        key, _, want = line.rpartition(" ")
        name, config, shapes, mode = key.split("|")
        out[(name, config, shapes, mode)] = want
    return out


def golden_rows(path):
    rows = [ln.split("\t") for ln in Path(path).read_text().splitlines()[1:]]
    return {(r[1], r[2], r[3], r[4], r[5], r[6]): r[7] for r in rows}


def test_repo_catalog_conforms_to_the_ingest(tool):
    code, msgs = tool.check()
    assert code == 0, msgs


def test_existing_series_keep_their_ids(tool):
    golden = golden_rows(tool.GOLDEN)
    pinned = series_ids()
    assert len(pinned) == 25
    for (name, config, shapes, mode), want in pinned.items():
        assert golden[(name, "report", config, shapes, mode, "")] == want


def test_the_catalog_declares_exactly_the_pinned_series(tool):
    declared = {
        (
            inp["name"],
            inp["disc"]["config_name"],
            inp["disc"]["input_shapes"],
            inp["disc"]["run_mode"],
        )
        for _, inp, _ in tool.declared()
    }
    assert declared == set(series_ids())


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


def test_a_model_definition_that_changes_an_id_fails(tool, catalog, tmp_path):
    golden = tmp_path / "golden.tsv"
    assert tool.check(catalog, golden, update=True)[0] == 0
    models = catalog / "models" / "granite.yaml"
    models.write_text(models.read_text().replace("value: model}", "value: op}"))
    code, msgs = tool.check(catalog, golden)
    assert code == 1
    assert any("granite" in m for m in msgs)


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
