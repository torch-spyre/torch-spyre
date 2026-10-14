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

"""Generic `benchmarks` bundles (accuracy evals, harnesses with no runner format) and the
identity of vLLM concurrency sweeps."""

import json

import pytest

from spyre_clickhouse_ingest import bundle, client, offline, schema, vllm
from spyre_clickhouse_ingest.identity import BenchmarkId
from spyre_clickhouse_ingest.writer import DryRunClient

AID = "93c0abb3-ed25-5934-b811-31b6c149ba47"
RID = "4b2b2c4e-1a7e-5c8e-9d2f-0a1b2c3d4e5f"
EVAL = {
    "benchmarks": [
        {"name": "lm_eval/gsm8k", "props": {"record_type": "quality", "model": "org/m", "max_new_tokens": 256},
         "backend": "spyre", "measurements": {"exact_match": 0.5, "lm_eval_acc": [0.5]}, "iterations": 200},
        {"name": "mteb/STS12", "props": {"record_type": "quality", "model": "org/e"},
         "backend": "spyre", "measurements": {"spearman": 0.7}},
        {"name": "mteb/STS12", "props": {"record_type": "quality", "model": "org/e"},
         "backend": "cpu", "measurements": {"spearman": 0.71}},
    ]
}  # fmt: skip


def _bundle(tmp_path, data=EVAL, **meta):
    root = tmp_path / "b"
    (root / "results").mkdir(parents=True)
    (root / "results" / "evals.json").write_text(json.dumps(data))
    fields = {"artifact_id": AID, "kind": "benchmarks", "test_type": "quality",
              "component": "spyre-inference", "arch": "s390x", **meta}  # fmt: skip
    (root / "bundle.json").write_text(
        json.dumps({k: v for k, v in fields.items() if v is not None})
    )
    return root


def test_a_quality_benchmarks_bundle_validates(tmp_path):
    meta = offline.check(_bundle(tmp_path))
    assert (meta["kind"], meta["test_type"]) == ("benchmarks", "quality")


@pytest.mark.parametrize(
    "meta,data,problem",
    [
        ({"test_type": "unit"}, EVAL, "test_type perf or quality"),
        ({"component": None}, EVAL, "names its component"),
        ({}, {"benchmarks": []}, "non-empty"),
        (
            {},
            {"benchmarks": [{"name": "x", "measurements": {"a": float("nan")}}]},
            "finite",
        ),
        ({}, {"benchmarks": [{"name": "x", "measurements": {}}]}, "needs measurements"),
        ({}, {"benchmarks": [{"measurements": {"a": 1}}]}, "needs a name"),
        (
            {},
            {"benchmarks": [{"name": "x", "measurements": {"a": 1}, "extra": 1}]},
            "unknown key",
        ),
        (
            {},
            {
                "benchmarks": [
                    {"name": "x", "measurements": {"a": 1}, "props": {"p": [1]}}
                ]
            },
            "props maps",
        ),
    ],
)
def test_a_bad_benchmarks_bundle_is_refused(tmp_path, meta, data, problem):
    with pytest.raises(offline.BundleError, match=problem):
        offline.check(_bundle(tmp_path, data, **meta))


def test_every_prop_keys_the_benchmark_and_backend_is_the_comparison_axis():
    entries = bundle.generic_entries(EVAL["benchmarks"])
    ids = [BenchmarkId.derive("spyre-inference", e["name"], e["tags"], e["disc"], e["disc_keys"])
           for e in entries]  # fmt: skip
    assert ids[1] == ids[2] and ids[0] != ids[1]
    assert entries[0]["measurements"] == {"exact_match": [0.5], "lm_eval_acc": [0.5]}
    assert entries[0]["disc_keys"] == ("max_new_tokens", "model", "record_type")


def test_a_benchmarks_bundle_writes_rows_and_a_quality_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "tables_present", lambda *a, **k: True)
    c = DryRunClient()
    root = _bundle(tmp_path)
    meta = offline.check(root)
    n = bundle.write_generic(c, "spyre_v2", root, meta, AID, RID, {"source": "upload"})
    runs = c.rows[schema.BENCHMARK_RUNS.name]
    assert n and len(runs) == 3
    assert {r["backend"] for r in runs} == {"spyre", "cpu"}
    assert {r["props"]["report_kind"] for r in runs} == {"benchmarks"}
    assert len(c.rows[schema.BENCHMARKS.name]) == 2
    (verdict,) = c.rows[schema.ARTIFACT_RESULTS.name]
    assert (verdict["test_type"], verdict["result_kind"]) == ("quality", "performance")


def test_quality_is_a_tier_the_ddl_admits():
    assert "quality" in schema.TEST_TYPE_VALUES
    assert "benchmarks" in offline.KINDS


SWEEP = "serving_roberta-large-ner-english_c64_n10000_dataset_len512_seqs4"


def test_a_sweep_name_keys_its_concurrency_and_limits():
    assert vllm.parse_input_shapes(SWEEP) == {
        "concurrency": "64", "num_prompts": "10000", "max_model_len": "512", "max_num_seqs": "4",
    }  # fmt: skip
    assert offline.shapes(SWEEP) == vllm.parse_input_shapes(SWEEP)


def _entry(name):
    row = {"metric": "mean_e2el_ms", "actual": 1.0,
           "extra": json.dumps({"test_name": name, "model": "org/m", "device": "spyre"})}  # fmt: skip
    (entry,) = vllm.bench_entries([row])
    return entry


def test_only_sweep_names_take_the_sweep_identity_keys():
    assert _entry(SWEEP)["disc_keys"] == vllm.SWEEP_ID_KEYS
    assert _entry(SWEEP)["props"]["run_mode"] == "serving"
    # An existing benchmark keeps its identity: the sweep keys would re-key it.
    tp = _entry("serve_granite33-8b_tp1_in64_out64")
    assert tp["disc_keys"] == vllm.BENCH_ID_KEYS


def test_sweeps_at_two_concurrencies_are_two_benchmarks():
    a, b = _entry(SWEEP), _entry(SWEEP.replace("_c64_", "_c16_"))
    ids = {
        BenchmarkId.derive("spyre-inference", e["name"], [], e["disc"], e["disc_keys"])
        for e in (a, b)
    }
    assert len(ids) == 2


ZAI = (
    "icr.io/zai_pytorch/v0.1.0_4q26/test_spyre-0.5.0_cp-3.12_ubi-10.2@sha256:"
    + "ab" * 32
)


def test_create_artifact_records_an_unbuilt_image_as_upstream():
    c = DryRunClient()
    meta = {
        "image": ZAI,
        "arch": "s390x",
        "create_artifact": {"component": "zai-pytorch"},
    }
    r = bundle.create_artifact(c, "spyre_v2", meta, dry_run=True)
    (art,) = c.rows["artifacts"]
    assert (art["origin"], art["component"], art["arch"]) == (
        "upstream",
        "zai-pytorch",
        "s390x",
    )
    assert art["artifact_name"] == "test_spyre-0.5.0_cp-3.12_ubi-10.2"
    assert r.artifact_id == art["artifact_id"]
    (ref,) = c.rows["artifact_refs"]
    assert ref["content_digest"] == "sha256:" + "ab" * 32


def test_create_artifact_needs_an_image_and_its_arch(tmp_path):
    root = _bundle(tmp_path, artifact_id=None, arch=None, image=ZAI,
                   create_artifact={"component": "zai-pytorch"})  # fmt: skip
    with pytest.raises(offline.BundleError, match="create_artifact records an image"):
        offline.check(root)
    root = _bundle(
        tmp_path / "ok",
        artifact_id=None,
        image=ZAI,
        create_artifact={"component": "zai-pytorch"},
    )
    assert offline.check(root)["create_artifact"] == {"component": "zai-pytorch"}
