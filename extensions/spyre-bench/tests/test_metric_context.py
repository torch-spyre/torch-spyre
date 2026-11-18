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

"""Metric context: vocabulary, renderer, comparability, pack declarations and the validate lint."""

import json
from pathlib import Path

from spyre_bench import engine, validate
from spyre_bench import metric_context as nc

V = nc.Vocab.load()
E = nc.EMPTY

FULL = {
    "compile_cache": "warm",
    "shape_warmup": "complete",
    "window": "steady",
    "concurrency": 8,
    "input_len": 2048,
    "output_len": 256,
    "length_mode": "fixed",
    "kv_ladder": "pow2/128",
    "prefix_cache": "off",
    "dtype": "fp16",
    "tp": 1,
    "model": "Model C",
    "backend": "spyre",
    "arch": "x86_64",
    "device_type": "dev-a",
    "device_count": 4,
    "build": "000000000a03",
    "clean": "enforced",
    "token_basis": "tokenizer",
    "mem_placement": "r1",
    "ttft_def": "arrival_to_first_token",
    "itl_def": "token_gap",
}


def rec(metric="llm.output_throughput", **over):
    ctx = {**FULL, **over.pop("ctx", {})}
    ctx = {k: v for k, v in ctx.items() if v is not None}
    return {"metric": metric, "value": "<value>", "unit": "token/s", "statistic": "value", "n": 3, "ctx": ctx, **over}


def test_vocabulary_size_and_lint():
    assert len(V.facets) >= 60
    assert V.lint() == []
    assert {f["group"] for f in V.facets.values()} == {
        "workload",
        "state",
        "build",
        "environment",
        "method",
        "judgement",
    }


def test_classify_engine_and_legacy_names():
    c = lambda m: (V.classify(m) or {}).get("id")  # noqa: E731
    assert c("llm.output_throughput") == "throughput"
    assert c("output_throughput") == "throughput"
    assert c("mean_ttft_ms") == "latency_ttft"
    assert c("llm.itl.median") == "latency_itl"
    assert c("median_tpot_ms") == "latency_itl"
    assert c("p99_e2el_ms") == "latency_e2e"
    assert c("kernel.time.mean") == "kernel_time"
    assert c("memory_transfer_mean_ms") == "kernel_time"
    assert c("mem_size_mb") == "memory"
    assert c("exact_match") == "quality_score"
    assert c("pt_util_percent") == "utilisation"
    assert c("ratio") == "ratio"
    assert c("made_up_metric") is None


def test_complete_record_derives_kv_span_and_cache_state():
    r = rec()
    assert nc.missing(r, V) == []
    vals, derived = nc.resolve(r, V)
    assert vals["kv_span"] == "2048-2304" and "kv_span" in derived
    assert vals["cache_state"] == "warm" and "cache_state" in derived


def test_cache_state_summary_rules():
    cs = lambda cc, sw: nc.resolve(rec(ctx={"compile_cache": cc, "shape_warmup": sw}), V)[0]["cache_state"]  # noqa: E731
    assert cs("warm", "complete") == "warm"
    assert cs("cold", "complete") == "cold"
    assert cs("warm", "none") == "cold"
    assert cs("warm", "partial") == "partial"


def test_missing_required_is_reported_and_number_still_rendered():
    r = rec(ctx={"compile_cache": None, "output_len": None})
    m = nc.missing(r, V)
    assert "cache_state" in m and "output_len" in m and "kv_span" in m
    out = nc.render(r, V)
    assert out.startswith("<value> token/s")  # the number is never hidden
    assert "[CONTEXT INCOMPLETE:" in out


def test_empty_slots_are_rendered_for_every_missing_required_facet():
    r = rec(ctx={"compile_cache": None, "output_len": None, "dtype": None})
    line = nc.context_line(r, V)
    assert f"cache {E}" in line and f"in 2k/out {E}" in line and f"KV {E}" in line and f"dtype {E}" in line
    flags = {f: s for _, t, s in nc.chips(r, V) for f in [t]}
    assert any(s == "empty" for s in flags.values())
    assert nc.context_line(rec(), V).count(E) == 0


def test_fully_empty_context_shows_all_slots():
    bare = {"metric": "llm.output_throughput", "value": "<value>", "unit": "token/s", "ctx": {}}
    line = nc.context_line(bare, V)
    for lab in ("cache", "load", "in", "KV", "dtype", "TP", "arch", "build", "clean"):
        assert f"{lab} {E}" in line or f"{lab} " in line
    assert f"load {E}" in line and f"KV {E}" in line and f"build {E}" in line


def test_unknown_value_counts_as_missing_but_none_value_does_not():
    assert "clean" in nc.missing(rec(ctx={"clean": "unknown"}), V)
    assert nc.missing(rec(ctx={"clean": "none"}), V) == []


def test_alternative_requirement():
    r = rec(ctx={"concurrency": None, "request_rate": 4})
    assert nc.missing(r, V) == []
    assert "rate 4/s" in nc.context_line(r, V)
    assert "concurrency|request_rate" in nc.missing(rec(ctx={"concurrency": None}), V)
    assert f"load {E}" in nc.context_line(rec(ctx={"concurrency": None}), V)


def test_conditional_facet_applies_only_to_spyre_backend():
    assert "kv_ladder" in nc.missing(rec(ctx={"kv_ladder": None}), V)
    assert nc.missing(rec(ctx={"kv_ladder": None, "backend": "cuda"}), V) == []


def test_context_line_order_and_content():
    line = nc.context_line(rec(), V)
    assert line.startswith("warm · window steady · c=8 · in 2k/out 256 · KV 2k→2304")
    assert "ladder pow2/128 · lengths fixed · prefix-cache off · fp16 · TP1" in line
    assert "build a03 · clean" in line


def test_kv_span_is_the_requested_range_not_the_bucket():
    vals, _ = nc.resolve(rec(ctx={"kv_bucket": "4096"}), V)
    assert vals["kv_span"] == "2048-2304" and vals["kv_bucket"] == "4096"


def test_expanded_view_flags_missing_and_derived():
    rows = nc.expanded_view(rec(ctx={"dtype": None}), V)
    flat = {f: (v, flag) for g in rows.values() for f, v, flag in g}
    assert flat["dtype"] == (E, "required, not recorded")
    assert flat["kv_span"][1] == "derived" and flat["arch"][1] == ""


def test_compare_comparable_across_builds():
    out = nc.compare(rec(ctx={"build": "000000000a02"}), rec(ctx={"build": "000000000a03"}), V)
    assert out["status"] == "comparable"


def test_compare_differs_lists_all_required_differences():
    a = rec(ctx={"compile_cache": "cold", "input_len": 2048})
    b = rec(ctx={"compile_cache": "warm", "input_len": 8192})
    out = nc.compare(a, b, V)
    names = [d[0] for d in out["differs"]]
    assert out["status"] == "differs"
    assert "cache_state" in names and "input_len" in names and "kv_span" in names
    assert out["line"].startswith("differs: cold/warm cold vs warm")


def test_compare_incomplete_blocks_verdict_but_not_display():
    a, b = rec(ctx={"output_len": None}), rec()
    out = nc.compare(a, b, V)
    assert out["status"] == "incomplete" and "output_len" in out["unknown"]
    assert "<value>" in nc.render(a, V)


def test_compare_cross_arch_with_vary_axis():
    a, b = rec(ctx={"arch": "x86_64"}), rec(ctx={"arch": "s390x"})
    assert nc.compare(a, b, V)["status"] == "differs"
    assert nc.compare(a, b, V, vary=("build", "arch"))["status"] == "comparable"


def test_compare_different_metric():
    assert nc.compare(rec(), rec(metric="llm.request_throughput"), V)["status"] == "different_metric"


def test_mem_placement_required_known_for_itl_and_kernel_but_compared_as_note():
    itl = "llm.itl.median"
    assert "mem_placement" in nc.missing(rec(itl, ctx={"mem_placement": None}), V)
    assert "mem_placement" not in nc.missing(rec("llm.output_throughput", ctx={"mem_placement": None}), V)
    out = nc.compare(rec(itl, ctx={"mem_placement": "r1"}), rec(itl, ctx={"mem_placement": "r3"}), V)
    assert out["status"] == "comparable" and out["also"][0][0] == "mem_placement"
    assert "mem_placement r1 vs r3" in out["line"]


def test_kernel_time_requires_placement_and_timer():
    k = {
        "metric": "kernel.time.mean",
        "value": "<value>",
        "unit": "s",
        "statistic": "mean",
        "n": 15,
        "ctx": {
            "compile_cache": "warm",
            "shape_warmup": "complete",
            "op_shape": "1x2816x262144",
            "dtype": "fp16",
            "batch_size": 1,
            "backend": "spyre",
            "arch": "x86_64",
            "device_type": "dev-a",
            "device_count": 1,
            "build": "000000000a03",
            "clean": "enforced",
            "timer": "wall",
            "mem_placement": "r1",
        },
    }
    assert nc.missing(k, V) == []
    k["ctx"].pop("mem_placement")
    assert nc.missing(k, V) == ["mem_placement"]


def test_length_drift_reports_actual_vs_requested():
    r = rec(ctx={"actual_input_len": 1700.0, "actual_output_len": 255.0})
    assert [d[0] for d in nc.length_drift(r)] == ["actual_input_len"]
    assert nc.length_drift(r, tol=0.2) == []


def test_page_header_splits_common_from_varying():
    a, b = rec(ctx={"concurrency": 1}), rec(ctx={"concurrency": 8})
    common, varying = nc.page_header([a, b], V)
    assert "fp16" in common and "c=1" not in common
    assert varying[0] == ["c=1"] and varying[1] == ["c=8"]


def test_scan_rows_counts():
    ok, total, bad = nc.scan_rows([rec(), rec(ctx={"dtype": None}), {"metric": "nope", "value": "<value>"}], V)
    assert (ok, total) == (1, 3) and bad[0][0] == 1 and bad[1][1] == ["metric class"]


def test_completeness_fraction():
    have, total = nc.completeness(rec(ctx={"dtype": None}), V)
    assert total - have == 1


EX = Path(__file__).resolve().parent.parent / "examples"


def _cat(name):
    return engine.Catalog([str(EX / name)], [], [str(EX / "central")])


def test_vocabulary_carries_design_parameter_and_expert_concurrency():
    assert "expert_concurrency" in V.facets and "design_parameter" in {c["id"] for c in V.classes}
    assert V.classify("param.chunk_tokens")["id"] == "design_parameter"
    assert not [f for f in V.facets if "group" in f]


def test_shipped_pack_metrics_have_valid_classes_and_no_registry_errors():
    cat = _cat("llm-demo")
    assert cat.errors == []
    reg = cat.reg
    for name, spec in reg.metrics.items():
        if spec.get("class"):
            assert V.class_by_id(spec["class"]), name
    assert reg.metrics["llm.itl.median"]["class"] == "latency_itl"


def test_pack_errors_flag_unknown_class_and_facet():
    errs = nc.pack_errors(
        [{"name": "x.m", "class": "nope"}, {"name": "x.n", "context": {"required": ["model", "bogus"]}}], V
    )
    assert any("class 'nope'" in e for e in errs) and any("bogus" in e for e in errs)
    assert nc.pack_errors([{"name": "x.m", "class": "throughput", "context": {"required": ["model|dtype"]}}], V) == []


def test_required_facets_override_replaces_class_default():
    spec = {"name": "llm.output_throughput", "class": "throughput", "context": {"required": ["model"]}}
    assert nc.required_facets(spec, V) == ["model"]
    assert "token_basis" in nc.required_facets({"name": "llm.output_throughput", "class": "throughput"}, V)


def test_lint_warns_on_undeclared_author_facets_and_accepts_declared():
    cat = _cat("kernel-demo")
    assert not cat.errors, cat.errors
    warns = nc.lint_catalog(cat)
    assert warns and all("needs facet timer" in w or "matches no metric class" in w for w in warns)
    for b in cat.of("Benchmark"):
        b.spec["context"] = {"timer": "wall"}
    assert nc.lint_catalog(cat) == []


def test_lint_flags_unknown_context_key_and_validate_stays_exit_zero(capsys):
    cat = _cat("kernel-demo")
    next(iter(cat.of("Benchmark"))).spec["context"] = {"not_a_facet": 1}
    assert any("not_a_facet" in w for w in nc.lint_catalog(cat))
    rc = validate.main([str(EX / "kernel-demo"), "--central", str(EX / "central"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and any("needs facet timer" in w for w in out[0]["warnings"])
