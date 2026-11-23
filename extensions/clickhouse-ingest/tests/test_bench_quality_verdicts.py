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

"""Quality rows in the verdict reader and the `Spyre Quality` check: separate summary, never gating, never
regressed on untrusted data. A fake source stands in for ClickHouse, so no engine is needed."""

import importlib.util
import json
from pathlib import Path

import pytest

from spyre_clickhouse_ingest import bench_verdicts as bv

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "bench_check.py"
spec = importlib.util.spec_from_file_location("bench_check", SCRIPT)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

RUN = "11111111-1111-4111-8111-111111111111"
SHA = "0123456789abcdef0123456789abcdef01234567"


def verdict_row(metric, status="regressed", bench="b1", name="eval-case", reason=""):
    return {
        "run_id": RUN, "benchmark_id": bench, "component": "c", "backend": "spyre", "arch": "x86_64",
        "source_job": "j", "name": name, "input_shapes": "", "ts": "2026-11-16 00:00:00", "metric": metric,
        "tier": "advisory", "higher_is_better": 1, "n_baseline": 3, "n_current": 1, "baseline_value": 0.6,
        "current_value": 0.5, "delta_pct": -16.0, "worse_pct": 16.0, "floor_pct": 5.0, "status": status,
        "reason": reason,
    }  # fmt: skip


class Fake:
    def __init__(
        self,
        verdicts,
        validity=("valid",),
        test_type="quality",
        n_repeats=1,
        trust="internal",
    ):
        self.verdicts, self.validity, self.test_type = verdicts, validity, test_type
        self.n_repeats, self.trust = n_repeats, trust

    def rows(self, sql):
        if "FROM v_benchmark_metric_verdicts" in sql:
            return self.verdicts
        if "FROM v_bench_results " in sql:
            return [
                {"run_id": v["run_id"], "benchmark_id": v["benchmark_id"], "backend": v["backend"],
                 "metric": v["metric"], "n_repeats": self.n_repeats, "cv_pct": 0.0, "is_suspect": 0}
                for v in self.verdicts
            ]  # fmt: skip
        if (
            "FROM v_invalidated_runs" in sql
            or "FROM v_benchmark_anchored_verdicts" in sql
        ):
            return []
        if "FROM benchmark_runs" in sql:
            return [{"run_id": RUN, "validity": list(self.validity), "trust": [self.trust],
                     "purpose": ["nightly"], "reasons": []}]  # fmt: skip
        if "FROM v_benchmark_results_enriched" in sql:
            types = (
                [self.test_type]
                if isinstance(self.test_type, str)
                else list(self.test_type)
            )
            return [{"run_id": RUN, "test_type": types}]
        raise AssertionError(sql)


def read(verdicts, **kw):
    return bv.read_verdicts(Fake(verdicts, **kw), run_id=RUN)


def only(report, metric):
    return next(r for r in report["results"] if r["metric"] == metric)


def test_quality_regression_needs_judged_trusted_data_and_ignores_the_repeat_cap():
    rep = read([verdict_row("llm.eval.exact_match")])
    r = rep["results"][0]
    assert (r["kind"], r["verdict"], r["reason"]) == (
        "quality",
        "regressed",
        "",
    )  # one repeat is fine for a score


@pytest.mark.parametrize("validity", ["invalid", "suspect", "external"])
def test_invalid_or_suspect_data_is_never_regressed(validity):
    r = read([verdict_row("llm.eval.exact_match")], validity=(validity,))["results"][0]
    assert (r["verdict"], r["reason"]) == ("suspected", f"validity_{validity}")


def test_worst_validity_of_a_mixed_run_wins():
    r = read([verdict_row("llm.eval.exact_match")], validity=("valid", "invalid"))[
        "results"
    ][0]
    assert r["verdict"] == "suspected"


def test_external_trust_is_never_regressed():
    r = read([verdict_row("llm.eval.exact_match")], trust="external")["results"][0]
    assert (r["verdict"], r["reason"]) == ("suspected", "trust_external")


def test_a_mixed_run_where_one_row_has_no_validity_is_unjudged():
    r = read([verdict_row("llm.eval.exact_match")], validity=("valid", ""))["results"][
        0
    ]
    assert (r["verdict"], r["reason"]) == ("suspected", "validity_unknown")


def test_a_run_mixing_test_types_is_classed_by_metric_name():
    rep = read(
        [
            verdict_row("llm.eval.exact_match", bench="q"),
            verdict_row("kernel_mean_ms", bench="p"),
        ],
        test_type=("perf", "quality"),
    )
    assert {r["metric"]: r["kind"] for r in rep["results"]} == {
        "llm.eval.exact_match": "quality",
        "kernel_mean_ms": "perf",
    }


def test_a_status_outside_the_known_verdicts_does_not_break_the_summary():
    rep = read([verdict_row("llm.eval.exact_match", status="mystery")])
    assert rep["summary"]["by_kind"]["quality"]["counts"]["mystery"] == 1


def test_a_bare_name_companion_is_diagnostic():
    names = [
        "exact_match",
        "exact_match_stderr",
        "exact_match.n_samples",
        "accuracy.invalid_frac",
        "rouge1.error_frac",
    ]
    rep = read([verdict_row(m) for m in names], test_type="perf")
    assert [r["diagnostic"] for r in rep["results"]] == [False, True, True, True, True]
    assert {r["kind"] for r in rep["results"]} == {"quality"}


def test_unjudged_data_is_never_regressed():
    r = read([verdict_row("llm.eval.exact_match")], validity=("",))["results"][0]
    assert (r["verdict"], r["reason"]) == ("suspected", "validity_unknown")


def test_perf_keeps_its_repeat_cap_and_does_not_need_a_validity():
    r = read(
        [verdict_row("kernel_mean_ms")], validity=("",), test_type="perf", n_repeats=1
    )["results"][0]
    assert (r["kind"], r["verdict"], r["reason"]) == (
        "perf",
        "suspected",
        "few_repeats",
    )


def test_kind_comes_from_the_test_type_or_the_metric_name():
    assert (
        read([verdict_row("anything")], test_type="quality")["results"][0]["kind"]
        == "quality"
    )
    assert (
        read([verdict_row("llm.parity.ref_coverage")], test_type="perf")["results"][0][
            "kind"
        ]
        == "quality"
    )
    assert (
        read([verdict_row("accuracy")], test_type="perf")["results"][0]["kind"]
        == "quality"
    )


def test_diagnostic_metrics_are_marked_and_not_counted_or_actionable():
    rows = [
        verdict_row("llm.eval.exact_match"),
        verdict_row("llm.eval.exact_match_stderr", status="regressed"),
        verdict_row("llm.eval.n_samples", status="regressed"),
        verdict_row("llm.eval.invalid_frac", status="regressed"),
    ]
    rep = read(rows)
    assert [r["diagnostic"] for r in rep["results"]] == [False, True, True, True]
    assert rep["summary"]["counts"]["regressed"] == 1
    assert rep["summary"]["actionable"] == 1
    assert len(bv.actionable(rep)) == 1
    assert "stderr" not in bv.to_markdown(rep)


def test_summary_splits_quality_from_perf():
    rep = read(
        [
            verdict_row("llm.eval.exact_match", bench="q"),
            verdict_row("kernel_mean_ms", bench="p", status="pass"),
        ],
        test_type="perf",
    )
    by = rep["summary"]["by_kind"]
    assert (
        by["quality"]["counts"]["regressed"] == 1 and by["quality"]["actionable"] == 1
    )
    assert by["perf"]["counts"]["pass"] == 1 and by["perf"]["actionable"] == 0


def test_a_view_that_calls_it_unchanged_is_a_pass():
    assert (
        read([verdict_row("llm.eval.exact_match", status="unchanged")])["results"][0][
            "verdict"
        ]
        == "pass"
    )


class Args:
    dashboard_url_template = "https://dash.example/runs/{run_id}/{benchmark_id}"
    annotation_path = "benchmarks/x.yaml"
    max_annotations = 10
    top = 10
    run_broken = ""
    detail = False


def payload(rep, **over):
    args = Args()
    for k, v in over.items():
        setattr(args, k, v)
    return bc.build_payload(rep, "quality", "x86_64", SHA, args)


def test_quality_check_shows_state_and_counts_only():
    rep = read(
        [
            verdict_row("llm.eval.exact_match"),
            verdict_row("llm.eval.exact_match_stderr"),
        ]
    )
    out = payload(rep)
    text = json.dumps(out)
    assert out["name"] == "Spyre Quality (x86_64)" and out["conclusion"] == "success"
    assert "1 case(s) recorded" in out["output"]["summary"]
    for word in (
        "regressed",
        "suspected",
        "improved",
        "0.5",
        "0.6",
        "baseline",
        "details_url",
    ):
        assert word not in text, word
    assert out["output"]["annotations"] == []


def test_counts_only_check_counts_this_kind_invalidations_only():
    rep = read([verdict_row("llm.eval.exact_match")])
    rep["results"].append({**rep["results"][0], "kind": "perf", "invalidated": True})
    rep["summary"]["invalidated"] = 1
    assert "invalidated" not in payload(rep)["output"]["summary"]
    rep["results"][0]["invalidated"] = True
    rep["results"].append({**rep["results"][0], "invalidated": False})
    assert "1 invalidated result(s) are excluded" in payload(rep)["output"]["summary"]


def test_quality_check_is_neutral_without_results_and_failure_only_when_broken():
    empty = read([])
    assert payload(empty)["conclusion"] == "neutral"
    assert payload(empty, run_broken="engine exited 2")["conclusion"] == "failure"


def test_quality_detail_lists_flagged_rows_like_the_perf_check():
    rep = read([verdict_row("llm.eval.exact_match")], validity=("suspect",))
    out = payload(rep, detail=True)
    assert out["conclusion"] == "neutral"
    assert "suspected (validity_suspect)" in out["output"]["summary"]
    assert "regressed" not in out["output"]["summary"]
    assert out["details_url"].startswith("https://dash.example/runs/")


def test_diagnostic_rows_never_reach_the_check():
    rep = read([verdict_row("llm.eval.exact_match_stderr")])
    assert bc.rows_of(rep, "quality", "x86_64") == []


def test_suspect_invalidation_demotes_a_quality_regression():
    class WithSuspect(Fake):
        def rows(self, sql):
            if "FROM v_invalidated_runs" in sql:
                return [
                    {
                        "run_id": RUN,
                        "benchmark_id": "b1",
                        "repeat": 1,
                        "state": "suspect",
                        "reason_code": "bad_card",
                    }
                ]
            return super().rows(sql)

    r = bv.read_verdicts(
        WithSuspect([verdict_row("llm.eval.exact_match")]), run_id=RUN
    )["results"][0]
    assert (r["verdict"], r["reason"]) == ("suspected", "suspect_data")


def test_a_hand_built_quality_regression_is_downgraded_by_the_poster():
    rep = read([verdict_row("llm.eval.exact_match")])
    rep["results"][0]["validity"] = "invalid"
    rep["results"][0]["verdict"] = "regressed"
    out = payload(rep, detail=True)
    assert (
        "regressed" not in out["output"]["summary"] and out["conclusion"] == "neutral"
    )


def test_perf_check_does_not_carry_quality_invalidations_or_run_wide_notes():
    rep = read(
        [
            verdict_row("llm.eval.exact_match", bench="q"),
            verdict_row("kernel_mean_ms", bench="p"),
        ],
        test_type="perf",
    )
    rep["results"][0]["invalidated"] = True
    rep["summary"]["invalidated"] = 1
    rep["notes"] = ["run 1 is invalidated; the verdict view drops its samples"]
    rep["excluded_runs"] = ["1"]
    args = Args()
    perf = bc.build_payload(rep, "perf", "x86_64", SHA, args)["output"]["summary"]
    assert "invalidated" not in perf and "Note:" not in perf
