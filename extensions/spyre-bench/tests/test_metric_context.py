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
    assert (len(V.facets), len(V.classes)) == (92, 16)
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
    for chip in (f"cache {E}", f"load {E}", f"in {E}/out {E}", f"KV {E}", f"dtype {E}", f"TP {E}", f"arch {E}",
                 f"build {E}", f"clean {E}"):  # fmt: skip
        assert chip in line.split(" \u00b7 "), chip
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


def test_lint_default_is_one_summary_line_and_detail_lists_facets():
    cat = _cat("kernel-demo")
    assert not cat.errors, cat.errors
    quiet = nc.lint_catalog(cat)
    assert len(quiet) == 1 and "undeclared context facets; run validate --lint-context" in quiet[0]
    detail = nc.lint_catalog(cat, detail=True)
    assert detail and all("needs facet timer" in w or "matches no metric class" in w for w in detail)
    for b in cat.of("Benchmark"):
        b.spec["context"] = {"timer": "wall"}
    assert nc.lint_catalog(cat) == [] and nc.lint_catalog(cat, detail=True) == []


def test_lint_flags_unknown_context_key_and_validate_stays_exit_zero(capsys):
    cat = _cat("kernel-demo")
    next(iter(cat.of("Benchmark"))).spec["context"] = {"not_a_facet": 1, "x-mine": 2}
    quiet = nc.lint_catalog(cat)
    assert any("context.not_a_facet" in w for w in quiet) and not any("x-mine" in w for w in quiet)
    argv = [str(EX / "kernel-demo"), "--central", str(EX / "central"), "--json"]
    rc = validate.main(argv)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and len(out[0]["warnings"]) == 1
    rc = validate.main(argv + ["--lint-context"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and any("needs facet timer" in w for w in out[0]["warnings"])


def test_shipped_non_llm_metrics_do_not_inherit_llm_requirements():
    reg = _cat("llm-demo").reg
    for name in ("spyre.compiled_graphs",):
        assert nc.metric_class(reg.metrics[name], V)["id"] == "diagnostic"
        assert nc.required_facets(reg.metrics[name], V) == []
    assert (
        nc.metric_class({"name": "comms.latency", "class": "collective"}, V)["required"]
        == V.class_by_id("collective")["required"]
    )
    assert "token_basis" not in V.class_by_id("collective")["required"]


# ---- pack declarations drive the renderer
GOODPUT = {"name": "llm.goodput_at_slo", "class": "throughput"}


def _bound(*specs):
    return V.bind({s["name"]: s for s in specs})


def test_declared_class_drives_missing_chips_and_compare():
    bare = {"metric": "llm.goodput_at_slo", "value": 1, "statistic": "value"}
    assert nc.missing(bare, V) == ["metric class"]
    B = _bound(GOODPUT)
    assert "token_basis" in nc.missing(bare, B) and "metric class" not in nc.missing(bare, B)
    assert any(txt.startswith("tokens") for _, txt, _ in nc.chips(bare, B))
    a, b = rec("llm.goodput_at_slo"), rec("llm.goodput_at_slo", ctx={"build": "000000000a02"})
    assert nc.compare(a, b, V)["status"] == "different_metric"
    assert nc.compare(a, b, B)["status"] == "comparable"
    assert nc.completeness(bare, B)[1] == len(V.class_by_id("throughput")["required"])


def test_declared_context_required_replaces_class_default_everywhere():
    spec = {**GOODPUT, "context": {"required": ["model", "dtype"], "optional": ["tp"]}}
    B = _bound(spec)
    r = rec("llm.goodput_at_slo", ctx={"dtype": None})
    assert nc.missing(r, B) == ["dtype"]
    assert [t for _, t, s in nc.chips(r, B) if s == "empty"] == [f"dtype {E}"]
    flat = {f for g in nc.expanded_view(r, B).values() for f, _, _ in g}
    assert flat == {"model", "dtype", "tp"}
    a, b = rec("llm.goodput_at_slo", ctx={"input_len": 1}), rec("llm.goodput_at_slo", ctx={"input_len": 2})
    assert nc.compare(a, b, B)["status"] == "comparable"  # input_len is no longer required
    assert nc.compare(a, b, V.bind({}))["status"] == "different_metric"


def test_custom_x_class_and_facets_carry_no_requirements():
    B = _bound({"name": "x.m", "class": "x-mine"}, {"name": "x.n", "context": {"required": ["x-foo", "model"]}})
    r = {"metric": "x.m", "value": 1, "ctx": {}}
    assert nc.missing(r, B) == []
    assert nc.missing({"metric": "x.n", "value": 1, "ctx": {}}, B) == ["model"]
    assert nc.pack_errors([{"name": "x.m", "class": "x-mine", "context": {"required": ["x-foo"]}}], V) == []
    assert nc.pack_errors([{"name": "x.m", "class": "mine"}], V)


# ---- placeholders and malformed values
def test_placeholder_values_are_missing():
    for junk in ("\u2014", "-", "n/a", "N/A", "  ", "Unknown", "null", "?", "none", "None", "\u2013"):
        assert "model" in nc.missing(rec(ctx={"model": junk}), V), junk
    assert nc.missing(rec(ctx={"clean": "none"}), V) == []  # none is a legal clean value
    assert nc.missing(rec(ctx={"model": "a\u2014b"}), V) == []


def test_em_dash_inside_a_legal_value_is_not_an_empty_chip():
    cs = nc.chips(rec(ctx={"model": "a\u2014b"}), V)
    assert ("model a\u2014b", "known") in [(t, s) for _, t, s in cs]
    assert all(s == "known" for _, _, s in cs)


def test_half_known_in_out_chip_is_empty_by_state_not_by_text():
    cs = {tuple(ids): s for ids, _, s in nc.chips(rec(ctx={"output_len": None}), V)}
    assert cs[("input_len", "output_len")] == "empty"
    assert cs[("kv_span",)] == "empty"


JUNK = ["abc", "128", "", None, "\u2014", -1, 1e30, float("nan"), [], {}, "1-", "-", "a-b", "9" * 400, True]


def test_junk_values_never_raise():
    for fid in V.facets:
        for junk in JUNK:
            for metric in ("llm.output_throughput", "kernel.time.mean", "llm.itl.median", "nope"):
                r = rec(metric, ctx={fid: junk})
                nc.missing(r, V)
                nc.render(r, V)
                nc.expanded_view(r, V)
                nc.length_drift(r)
                nc.compare(r, rec(metric), V)
                nc.compare(rec(metric), r, V, vary="build")
    for k, v in (("kv_span", "abc"), ("kv_span", "128"), ("input_len", "abc")):
        assert nc.render(rec(ctx={k: v}), V).startswith("<value>")


# ---- compare details
def test_vary_string_is_one_facet_not_characters():
    a, b = rec(ctx={"build": "000000000a02"}), rec(ctx={"build": "000000000a03"})
    assert nc.compare(a, b, V, vary="build")["status"] == "comparable"
    assert nc.compare(rec(ctx={"arch": "x86_64"}), rec(ctx={"arch": "s390x"}), V, vary="arch")["status"] == "comparable"


def test_numeric_facets_compare_as_numbers():
    assert nc.compare(rec(ctx={"tp": "1"}), rec(ctx={"tp": 1.0}), V)["status"] == "comparable"
    assert nc.compare(rec(ctx={"concurrency": 8}), rec(ctx={"concurrency": "8.0"}), V)["status"] == "comparable"
    assert nc.compare(rec(ctx={"tp": 1}), rec(ctx={"tp": 2}), V)["status"] == "differs"


def test_compare_both_sides_incomplete_lists_union():
    out = nc.compare(rec(ctx={"dtype": None}), rec(ctx={"output_len": None}), V)
    assert out["status"] == "incomplete"
    assert {"dtype", "output_len", "kv_span"} <= set(out["unknown"])


def test_unknown_backend_lists_conditional_facet_and_backend_missing():
    m = nc.missing(rec(ctx={"backend": None, "kv_ladder": None}), V)
    assert "backend" in m and "kv_ladder" in m


# ---- review round 3
def test_none_is_a_value_only_where_the_facet_allows_it():
    for fid in ("device_type", "quant_scheme", "baseline", "clean", "cpu_pinning"):
        assert nc.resolve(rec(ctx={fid: "none"}), V)[0].get(fid) == "none", fid
    for junk in ("none", "None", "NA", "N/A"):
        assert "model" not in nc.resolve(rec(ctx={"model": junk}), V)[0], junk


def test_clean_unknown_is_missing_and_has_no_chip_mapping():
    assert "clean" in nc.missing(rec(ctx={"clean": "unknown"}), V)
    assert "unknown" not in nc.CLEAN_CHIP


def test_nonfinite_and_empty_containers_are_unrecorded():
    class FloatLike:  # stands in for a numpy scalar
        def __init__(self, v):
            self.v = v

        def __float__(self):
            return self.v

    for junk in (
        float("nan"),
        float("inf"),
        float("-inf"),
        [],
        {},
        (),
        FloatLike(float("nan")),
        FloatLike(float("inf")),
    ):
        assert "tp" in nc.missing(rec(ctx={"tp": junk}), V), repr(junk)
        assert "TPnan" not in nc.context_line(rec(ctx={"tp": junk}), V)
    assert "tp" not in nc.missing(rec(ctx={"tp": FloatLike(2.0)}), V)


RECORD_SHAPES = [None, 5, "x", [], {}, {"metric": 7}, {"metric": ["a"]}, {"metric": {"a": 1}, "ctx": 3},
                 {"metric": "llm.output_throughput", "ctx": [1], "statistic": ["p"], "n": {}, "value": None},
                 {"metric": "llm.output_throughput", "ctx": {1: 2, None: 3}, "statistic": {"a"}, "unit": 4},
                 {"metric": "llm.output_throughput", "statistic": 7, "n": float("nan")}]  # fmt: skip


def test_every_public_function_is_total_over_record_shapes():
    for r in RECORD_SHAPES:
        nc.resolve(r, V)
        nc.missing(r, V)
        nc.completeness(r, V)
        nc.is_complete(r, V)
        nc.chips(r, V, expanded=True)
        nc.context_line(r, V)
        nc.stat_label(r)
        nc.render(r, V)
        nc.expanded_view(r, V)
        nc.length_drift(r)
        nc.page_header([r, rec()], V)
        nc.scan_rows([r], V)
        for s in RECORD_SHAPES:
            nc.compare(r, s, V)


def test_compare_is_symmetric_when_a_conditional_facet_toggles():
    a = rec(ctx={"backend": "spyre", "kv_ladder": "L1"})
    b = rec(ctx={"backend": "cpu", "kv_ladder": None})
    ab, ba = nc.compare(a, b, V, vary=("backend",)), nc.compare(b, a, V, vary=("backend",))
    assert ab["status"] == ba["status"] == "differs"
    assert [(f, y, x) for f, x, y in ab["differs"]] == ba["differs"]


def test_enum_and_bool_rendering():
    assert nc._chip_value("cpu_pinning", "None", set()) == "off"
    assert nc._chip_value("cpu_pinning", "explicit", set()) == "on"
    assert "prefix-cache off" in nc.context_line(rec(ctx={"prefix_cache": False}), V)
    assert "prefix-cache False" not in nc.context_line(rec(ctx={"prefix_cache": False}), V)


def test_every_required_facet_of_every_class_gets_an_empty_chip_when_absent():
    for cls in V.classes:
        r = {"metric": "m", "value": 1, "ctx": {}}
        B = V.bind({"m": {"name": "m", "class": cls["id"]}})
        shown = {f for ids, _, state in nc.chips(r, B) if state == "empty" for f in ids}
        for req in cls["required"]:
            alts = req.split("|")
            if all(a in ("statistic", "sample_n") for a in alts):
                continue
            assert any(a in shown for a in alts), (cls["id"], req)


def test_lint_warns_on_illegal_enum_value_in_benchmark_context():
    cat = _cat("kernel-demo")
    next(iter(cat.of("Benchmark"))).spec["context"] = {"timer": "stopwatch", "sampling": "anything"}
    w = nc.lint_catalog(cat)
    assert any("context.timer='stopwatch'" in x for x in w) and not any("sampling" in x for x in w)


def test_third_party_metrics_without_a_class_are_unclassified_and_quiet():
    cat = engine.Catalog([str(EX / "postgres-demo")], [str(EX / "packs" / "postgres-example")], [str(EX / "central")])
    assert not cat.errors, cat.errors
    B = V.bind(cat.reg.metrics, cat.reg.builtin_metrics)
    assert "pg.latency.mean" in cat.reg.metrics and "pg.latency.mean" not in cat.reg.builtin_metrics
    assert B.cls_for("pg.latency.mean")["id"] == "unclassified"
    assert nc.missing({"metric": "pg.latency.mean", "value": 1}, B) == []
    assert not [w for w in nc.lint_catalog(cat, detail=True) if "pg." in w]


# ---- review round 4
def test_declared_requirements_on_a_third_party_metric_are_linted():
    spec = {"name": "pg.x", "context": {"required": ["timer"]}}
    B = V.bind({"pg.x": spec}, builtin=set())
    assert B.cls_for("pg.x")["required"] == ["timer"] and B.cls_for("pg.x")["id"] != "unclassified"
    assert V.bind({"pg.x": {"name": "pg.x"}}, builtin=set()).cls_for("pg.x")["id"] == "unclassified"


def test_yaml_boolean_off_is_prefix_cache_off_everywhere():
    assert nc.resolve(rec(ctx={"prefix_cache": False}), V)[0]["prefix_cache"] == "off"
    out = nc.compare(rec(ctx={"prefix_cache": False}), rec(ctx={"prefix_cache": "off"}), V)
    assert out["status"] == "comparable"
    assert nc.compare(rec(ctx={"prefix_cache": False}), rec(ctx={"prefix_cache": "on_warm"}), V)["status"] == "differs"
    f = V.facets["prefix_cache"]
    assert nc._context_value_warning(f, "prefix_cache", False) is None
    assert "ambiguous" in nc._context_value_warning(f, "prefix_cache", True)
    assert "YAML null" in nc._context_value_warning(f, "prefix_cache", None)
    assert "stopwatch" in nc._context_value_warning(V.facets["timer"], "timer", "stopwatch")


def test_containers_without_recorded_leaves_are_unrecorded_and_others_render_as_counts():
    class Arr:
        def __init__(self, *v):
            self.v = v

        def __len__(self):
            return len(self.v)

        def __getitem__(self, i):
            return self.v[i]

    for junk in ([float("nan")], [None], [[None], [float("nan")]], {"a": None}, Arr(float("nan"), float("nan"))):
        assert "tp" in nc.missing(rec(ctx={"tp": junk}), V), repr(junk)
    line = nc.context_line(rec(ctx={"tp": [1, 2], "concurrency": {"a": 1}}), V)
    assert "TP 2 values" in line and "c=1 values" in line and "[1, 2]" not in line


def test_builtin_metrics_are_recognised_through_relative_and_symlinked_paths(tmp_path):
    import os

    pack = Path(engine.__file__).resolve().parent / "packs" / "kernel"
    link = tmp_path / "kernel-link"
    link.symlink_to(pack)
    for path in (link, Path(os.path.relpath(pack)), pack / ".." / "kernel"):
        reg = engine.Registry()
        reg.load(path)
        assert "kernel.time.mean" in reg.builtin_metrics, path


# ---- review round 5: hostile and pathological values
def _fast(fn, limit=1.0):
    import time

    t0 = time.monotonic()
    out = fn()
    assert time.monotonic() - t0 < limit
    return out


def _everything(r):
    nc.missing(r, V)
    nc.render(r, V)
    nc.expanded_view(r, V)
    nc.compare(r, rec(), V)
    nc.page_header([r, rec()], V)


def test_self_containing_containers_finish_fast_and_read_as_unrecorded():
    import yaml

    cyc = [1, 2, 3, 4, 5, 6, 7]
    cyc.append(cyc)
    anchored = yaml.safe_load("a: &a [1, 2, *a]")["a"]
    d = {"k": 1}
    d["self"] = d
    wide_deep = [[[[[[[[list(range(30))] * 30] * 30]] * 30] * 30]] * 30] * 30
    for junk in (cyc, anchored, d, wide_deep):
        r = rec(ctx={"tp": junk, "concurrency": junk})
        _fast(lambda: _everything(r))
        assert "tp" in nc.missing(r, V)


def test_depth_cap_and_node_budget_are_pinned():
    def nest(levels, leaf=5):
        v = leaf
        for _ in range(levels):
            v = [v]
        return v

    known = lambda x: "tp" not in nc.missing(rec(ctx={"tp": x}), V)  # noqa: E731
    assert nc.MAX_DEPTH == 8 and nc.MAX_NODES == 10_000
    assert known(nest(9)) and not known(nest(10))  # 9 brackets = containers at depths 0..8
    assert known([5] * 10_000) and not known([5] * 10_001)
    assert not known(nest(30, leaf=None))


class _Hostile:
    def __init__(self, exc, where):
        self.exc, self.where = exc, where

    def _go(self, name):
        if name == self.where:
            raise self.exc("boom")

    def __float__(self):
        self._go("float")
        return 1.0

    def __len__(self):
        self._go("len")
        return 1

    def __getitem__(self, i):
        self._go("getitem")
        return 1

    def __iter__(self):
        self._go("iter")
        return iter([1])

    def __bool__(self):
        self._go("bool")
        return True

    def __str__(self):
        self._go("str")
        return "h"


def test_hostile_objects_never_escape_the_renderer():
    for exc in (RuntimeError, RecursionError, MemoryError, KeyError):
        for where in ("float", "len", "getitem", "iter", "bool", "str"):
            h = _Hostile(exc, where)
            for fid in ("tp", "model", "concurrency", "kv_span", "input_len", "clean"):
                _everything(rec(ctx={fid: h}))
            _everything({"metric": "llm.output_throughput", "value": h, "unit": "x", "n": h, "statistic": h, "ctx": {}})


def test_base_exceptions_still_propagate():
    import pytest

    for exc in (KeyboardInterrupt, SystemExit):
        with pytest.raises(exc):
            nc.render(rec(ctx={"tp": _Hostile(exc, "getitem")}), V)


def test_hostile_length_of_another_type_is_not_a_container_and_core_dir_resolves(tmp_path, monkeypatch):
    import os

    class BadLen:
        def __len__(self):
            raise RuntimeError("x")

    assert nc._clen(BadLen()) is None
    link = tmp_path / "core-link"
    link.symlink_to(engine.CORE_DIR)
    for path in (link, Path(os.path.relpath(engine.CORE_DIR))):
        monkeypatch.setattr(engine, "CORE_DIR", path)
        assert "core.duration" in engine.Registry().builtin_metrics, path
