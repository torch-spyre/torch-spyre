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

"""Comparison-target rows (references) and a run's internal trace records stay out of the
readers that mean "a benchmark's measurements" or "a build", on real chdb.

`dump(extras=False)` over existing-style rows is pinned to data/reader_exclusion_golden.json,
recorded from the schema before the exclusion; `dump(extras=True)` adds reference rows,
reference artifacts and trace records, and must read back the same, also under join_use_nulls.

Regenerate the golden ONLY from a schema that is meant to read the same rows differently (it is
the reader contract, not a snapshot to refresh): load `build(<old schema dir>)`, then write
`json.dumps(dump(c))`, one row per line. It pins the verdict view's Mann-Whitney scores and
floors (rounded to 4 places by the view), so a chdb upgrade, a verdict-policy change or PR #5991
can move it; review such a diff before accepting it."""

import json
import uuid
from pathlib import Path

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

GOLDEN = Path(__file__).parent / "data" / "reader_exclusion_golden.json"
FILES = (
    "10-functional-tests.sql",
    "20-artifacts.sql",
    "25-artifact-dispatch.sql",
    "26-dispatch-targets.sql",
    "30-benchmarks.sql",
    "49-result-uploads.sql",
    "50-artifact-views.sql",
    "52-cross-views.sql",
    "55-dispatch-views.sql",
    "60-benchmark-views.sql",
    "62-benchmark-verdicts.sql",
    "63-perf-regression-gate.sql",
    "70-vllm-hud-projection.sql",
)
TRACE_TYPES = (
    "step_trace",
    "span",
    "trace_phase",
    "reference",
    "trace_step",
    "trace_kernel",
    "compile_pass",
)
EVERY = [
    "artifact.tagged",
    "artifact.recorded",
    "artifact.promoted",
    "results.recorded",
]
FEED_PARAMS = {
    "event_types": EVERY,
    "component": "",
    "arch": "",
    "kind": "",
    "artifact_name": "",
    "tag_family": "",
    "tag_pattern": "",
    "tag_props": {},
    "exclude_families": [],
    "exclude_tag_patterns": [],
    "result_test_types": [],
    "result_states": [],
    "require": [],
    "unless_exists": [],
}


def u(name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


A_SI, A_TS, A_REF, A_ORPHAN = u("a-si"), u("a-ts"), u("a-ref"), u("a-orphan")
B_SERVE, B_MATMUL = u("b-serve"), u("b-matmul")
B_STEP, B_SPAN, B_PHASE = u("b-step"), u("b-span"), u("b-phase")
SERIES = [u(f"series-{i}") for i in range(12)]
REF_RUNS = [u(f"ref-{i}") for i in range(4)]
B_REFCASE = u("b-refcase")
B_MORE = [u("b-tstep"), u("b-tkernel"), u("b-cpass")]


def literal(v):
    if isinstance(v, dict):
        return (
            "map("
            + ", ".join(f"{literal(k)}, {literal(x)}" for k, x in v.items())
            + ")"
        )
    if isinstance(v, list):
        return "[" + ", ".join(literal(x) for x in v) + "]"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


class Chdb:
    def __init__(self):
        self.s = session.Session()

    def command(self, sql):
        return str(self.s.query(sql, "TSV")).rstrip("\n")

    def rows(self, sql):
        return [
            json.loads(x) for x in str(self.s.query(sql, "JSONEachRow")).splitlines()
        ]

    def insert(self, table, rows):
        body = "\n".join(json.dumps(r) for r in rows)
        self.command(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")


def load(schema_dir, c):
    """Fills a fresh database in the applier's order: tables and insert-fed MVs, then views."""
    objs = [
        o
        for name in FILES
        for o in SchemaApplier.objects(
            schema_dir / name, (schema_dir / name).read_text()
        )
    ]
    for kinds in (("table", "mv"), ("view",)):
        for o in objs:
            if o.kind in kinds:
                c.command(o.sql)


def at(ts):
    return {"ts": ts, "audit_timestamp": ts + ".000"}


def artifact(aid, component, name, ts, kind="image", origin="built"):
    return {
        **at(ts), "artifact_id": aid, "component": component, "arch": "x86_64",
        "kind": kind, "artifact_name": name, "origin": origin,
        "props": {"id12": aid.replace("-", "")[:12]},
    }  # fmt: skip


def tag(name, family, aid, ts):
    return {**at(ts), "tag": name, "tag_family": family, "artifact_id": aid}


def verdict(aid, run, ts, kind="performance", test_type="perf", state="passed"):
    return {
        **at(ts), "artifact_id": aid, "run_id": run, "result_kind": kind,
        "test_type": test_type, "state": state, "arch": "x86_64", "duration_s": 1.0,
    }  # fmt: skip


def bench(bid, component, name, **props):
    return {"benchmark_id": bid, "component": component, "name": name, "props": props}


def run_row(run, bid, component, backend, measurements, ts, **props):
    return {
        "ts": ts, "run_id": run, "benchmark_id": bid, "component": component,
        "backend": backend, "measurements": measurements,
        "props": {"arch": "x86_64", "source_file": f"job-perf-x86_64/{run[:8]}/x.json", **props},
    }  # fmt: skip


def populate(c, extras):
    c.insert(
        "artifacts",
        [
            artifact(A_SI, "spyre-inference", "si-dev", "2026-11-01 10:00:00"),
            artifact(A_TS, "torch-spyre", "ts-dev", "2026-11-01 10:30:00"),
        ]
        + (
            [
                artifact(
                    A_REF,
                    "reference",
                    "ref-v1",
                    "2026-11-09 10:00:00",
                    "generic",
                    "promoted",
                )
            ]
            if extras
            else []
        ),
    )
    c.insert(
        "artifact_tags",
        [
            tag("nightly", "nightly", A_SI, "2026-11-01 11:00:00"),
            tag("nightly-2026-11-01", "nightly", A_SI, "2026-11-01 11:00:00"),
            tag("nightly-2026-11-01", "nightly", A_TS, "2026-11-01 11:00:00"),
            tag("main", "main", A_TS, "2026-11-02 11:00:00"),
            # No artifacts row behind it: the views keep such a tag and verdict.
            tag("orphan-tag", "nightly", A_ORPHAN, "2026-11-02 12:00:00"),
        ]
        + (
            [
                tag("ref-current", "reference", A_REF, "2026-11-09 11:00:00"),
                tag("ref-v1", "reference", A_REF, "2026-11-09 11:00:00"),
                # A reference artifact under a build family: excluded by its component.
                tag("odd", "nightly", A_REF, "2026-11-09 11:00:00"),
                # A reference-family tag on a build: excluded by its family.
                tag("ref-stray", "reference", A_TS, "2026-11-09 11:00:00"),
                # A reference tag with no artifacts row behind it.
                tag(
                    "ref-orphan", "reference", u("a-ref-orphan"), "2026-11-09 11:30:00"
                ),
            ]
            if extras
            else []
        ),
    )
    benchmarks = [
        bench(B_SERVE, "spyre-inference", "serve_x", record_type="model", model="m",
              run_mode="serve", tensor_parallel="1", input_len="128", output_len="128"),
        bench(B_MATMUL, "torch-spyre", "matmul", record_type="op", config_name="c",
              input_shapes="[1]", run_mode="eager"),
    ]  # fmt: skip
    if extras:
        benchmarks += [
            bench(B_STEP, "torch-spyre", "step_x", record_type="step_trace", world_size="4"),
            bench(B_SPAN, "spyre-inference", "span_x", record_type="span", lane="0"),
            bench(B_PHASE, "spyre-inference", "phase_x", record_type="trace_phase", phase="p"),
            # A case typed as a reference but on an unmarked run: the series views skip it.
            bench(B_MORE[0], "spyre-inference", "tstep_x", record_type="trace_step"),
            bench(B_MORE[1], "spyre-inference", "tkernel_x", record_type="trace_kernel"),
            bench(B_MORE[2], "spyre-inference", "cpass_x", record_type="compile_pass"),
            bench(B_REFCASE, "spyre-inference", "refcase_x", record_type="reference"),
        ]  # fmt: skip
    c.insert("benchmarks", benchmarks)

    runs, verdicts = [], []
    for i, rid in enumerate(SERIES):
        ts = f"2026-11-{i + 1:02d} 12:00:00"
        slow = i >= 10
        runs.append(run_row(
            rid, B_SERVE, "spyre-inference", "spyre",
            {"mean_ttft_ms": [150.0 if slow else v for v in (100.0, 101.0, 99.0)],
             "output_throughput": [600.0 if slow else v for v in (1000.0, 1002.0, 998.0)]},
            ts, repo="spyre-inference", head_branch="main", head_sha=f"{i:040d}",
            workflow_id=str(1000 + i), job_id=str(i), source_file=f"vllm-benchmark/{i}/serve.json",
        ))  # fmt: skip
        verdicts.append(verdict(A_SI, rid, ts))
        if extras:
            runs.append(run_row(
                rid, B_PHASE, "spyre-inference", "spyre",
                {"mean_ttft_ms": [900.0 + i], "wall_s": [1.0]}, ts,
                source_file=f"vllm-benchmark/{i}/serve.json",
            ))  # fmt: skip
    o0, o1 = u("op-0"), u("op-1")
    runs += [
        run_row(o0, B_MATMUL, "torch-spyre", "sendnn",
                {"kernel_mean_ms": [1.0], "total_duration_ms": [3.0]}, "2026-11-10 09:00:00"),
        run_row(o1, B_MATMUL, "torch-spyre", "spyre",
                {"kernel_mean_ms": [1.1], "spyre_ms": [2.0], "cpu_ms": [4.0],
                 "total_duration_ms": [3.3]}, "2026-11-11 09:00:00"),
    ]  # fmt: skip
    verdicts += [
        verdict(A_TS, o0, "2026-11-10 09:00:00"),
        verdict(A_TS, o1, "2026-11-11 09:00:00"),
    ]
    orphan, unlinked = u("orphan-run"), u("unlinked-run")
    runs += [
        run_row(orphan, B_MATMUL, "torch-spyre", "spyre", {"kernel_mean_ms": [2.0]}, "2026-11-12 09:00:00"),
        run_row(unlinked, B_MATMUL, "torch-spyre", "spyre", {"kernel_mean_ms": [2.0]}, "2026-11-12 10:00:00"),
    ]  # fmt: skip
    verdicts.append(verdict(A_ORPHAN, orphan, "2026-11-12 09:00:00"))
    verdicts.append(verdict(A_SI, u("failed-0"), "2026-11-05 13:00:00", state="failed"))
    fun = u("functional-0")
    verdicts.append(verdict(A_SI, fun, "2026-11-01 13:00:00", "functional", "unit"))
    c.insert("test_cases", [{"test_case_id": u("tc"), "component": "spyre-inference",
                             "classname": "T", "name": "t"}])  # fmt: skip
    c.insert("test_case_runs", [{"run_id": fun, "test_case_id": u("tc"),
                                 "component": "spyre-inference", "status": "passed"}])  # fmt: skip

    if extras:
        ref = dict(
            repo="spyre-inference", head_branch="main", head_sha="ref", job_id="1"
        )
        marks = [
            {
                "run.ref_only": "1",
                "run.record_type": "reference",
                "run.trust": "external",
            },
            {"run.ref_only": "1"},
            {"run.record_type": "reference"},
            {"run.record_type": "reference"},
        ]
        runs += [
            run_row(REF_RUNS[0], B_SERVE, "spyre-inference", "spyre",
                    {"mean_ttft_ms": [1.0], "output_throughput": [9999.0]},
                    "2026-11-09 12:00:00", **ref, **marks[0]),
            run_row(REF_RUNS[1], B_SERVE, "spyre-inference", "spyre",
                    {"mean_ttft_ms": [1.0], "output_throughput": [9999.0]},
                    "2026-11-09 12:30:00", **ref, **marks[1]),
            # A model row on the HUD's own filters, marked only by its record type.
            run_row(REF_RUNS[3], B_SERVE, "spyre-inference", "spyre",
                    {"mean_ttft_ms": [1.0], "output_throughput": [9999.0]},
                    "2026-11-09 12:45:00", **ref, **marks[3]),
            *[run_row(SERIES[4], b, "spyre-inference", "spyre", {"mean_ttft_ms": [7.0], "wall_s": [1.0]}, "2026-11-05 12:00:00") for b in B_MORE],
            run_row(SERIES[5], B_REFCASE, "spyre-inference", "spyre",
                    {"mean_ttft_ms": [5.0], "wall_s": [1.0]}, "2026-11-06 12:00:00"),
            # Between the two real runs: it would become the newer sendnn baseline of o1.
            run_row(REF_RUNS[2], B_MATMUL, "torch-spyre", "sendnn",
                    {"kernel_mean_ms": [0.1], "total_duration_ms": [0.2]},
                    "2026-11-10 20:00:00", **marks[2]),
        ]  # fmt: skip
        verdicts += [verdict(A_REF, r, "2026-11-09 13:00:00") for r in REF_RUNS]
        verdicts += [
            verdict(A_REF, u("ref-failed"), "2026-11-09 14:00:00", state="failed"),
            verdict(
                A_REF, u("ref-functional"), "2026-11-09 15:00:00", "functional", "unit"
            ),
        ]
        runs += [
            run_row(SERIES[3], B_STEP, "torch-spyre", "spyre",
                    {"start_ns": [1.0, 2.0, 3.0], "seq": [0.0, 1.0, 2.0], "spyre_ms": [2.0],
                     "cpu_ms": [4.0]}, "2026-11-04 12:00:00"),
            # A sendnn row of a trace type: the target-side rows of the same type would pair with it.
            run_row(u("phase-sendnn"), B_PHASE, "spyre-inference", "sendnn",
                    {"mean_ttft_ms": [800.0]}, "2026-10-31 12:00:00"),
            run_row(SERIES[3], B_SPAN, "spyre-inference", "spyre",
                    {"start_ns": [1.0, 2.0, 3.0], "seq": [0.0, 1.0, 2.0]}, "2026-11-04 12:00:00"),
        ]  # fmt: skip
    c.insert(
        "artifact_subscriptions",
        [{"subscription_id": "wild", "enabled": True, "not_before": "2000-01-01 00:00:00.000",
          "event_types": EVERY, "target_type": "webhook",
          "target": "https://hooks.example.com/x"}],
    )  # fmt: skip
    c.insert("benchmark_runs", runs)
    c.insert("artifact_results", verdicts)


Q = {
    "enriched": "SELECT toString(run_id) r, toString(benchmark_id) b, backend, record_type, arch, toString(run_ts) ts, toString(artifact_id) a, toString(measurements) m FROM v_benchmark_results_enriched ORDER BY r, b, backend",
    "wide": "SELECT toString(run_id) r, toString(benchmark_id) b, backend, operation_name, record_type, total_duration_ms, cpu_ms, spyre_ms, kernel_mean_ms FROM v_benchmark_wide ORDER BY r, b, backend",
    "trend": "SELECT toString(day) d, toString(benchmark_id) b, name, backend, arch, metric, avg_value, min_value, max_value, runs FROM v_benchmark_trend ORDER BY d, b, backend, metric",
    "compare": "SELECT toString(run_id) r, toString(benchmark_id) b, backend, baseline_backend, pairing, metric, value, baseline_value, ratio, gap_hours, toString(baseline_run_id) br FROM v_benchmark_backend_compare ORDER BY r, b, pairing, metric",
    "verdicts": "SELECT toString(run_id) r, toString(benchmark_id) b, backend, metric, tier, status, reason, n_baseline, n_current, baseline_value, current_value, delta_pct, worse_pct, if(n_baseline < 10, NULL, score) score, if(n_baseline < 10, NULL, floor_pct) floor_pct FROM v_benchmark_metric_verdicts ORDER BY r, b, metric",
    "gate": "SELECT toString(run_id) r, toString(benchmark_id) b, gate_status, arraySort(regressed_metrics) rm, arraySort(warn_metrics) wm, primary_metric, worst_delta_pct FROM v_benchmark_gate ORDER BY r, b",
    "perf_gate": "SELECT toString(artifact_id) a, toString(run_id) r, gate_status, arraySort(arrayMap(x -> toString(x), failing_benchmarks)) f, arraySort(arrayMap(x -> toString(x), warn_benchmarks)) w, n_no_baseline, n_benchmarks FROM v_perf_regression_gate ORDER BY r",
    "run_artifacts": "SELECT toString(run_id) r, component, arch, toString(run_ts) ts, toString(artifact_id) a, artifact_name, artifact_id12, tag, tag_family FROM v_benchmark_run_artifacts ORDER BY r, tag",
    "tag_list": "SELECT tag, tag_family, toString(first_ts) f, toString(last_ts) l, promotion_count, artifact_count, component_count, arch_list, is_rolling FROM v_tag_list ORDER BY tag",
    "tag_resolution": "SELECT tag, tag_family, component, arch, artifact_name, toString(artifact_id) a, toString(resolved_ts) t, promotion_count, is_rolling FROM v_tag_resolution ORDER BY tag, component",
    "tag_results": "SELECT tag, tag_family, component, toString(artifact_id) a, toString(run_id) r, result_kind, test_type, state, total_tests, passed, run_arch FROM v_tag_results ORDER BY tag, r",
    "tag_artifacts": "SELECT tag, tag_family, component, arch, toString(artifact_id) a, artifact_name, kind, origin, id12 FROM v_tag_artifacts ORDER BY tag, a",
    "tier_trend": "SELECT toString(day) d, tag_family, result_kind, test_type, run_arch, component, runs, artifacts, total_tests, passed FROM v_tier_trend ORDER BY d, tag_family, test_type, component",
    "results_enriched": "SELECT toString(artifact_id) a, toString(run_id) r, component, result_kind, test_type, state, total_tests, passed, run_arch, toString(ts) ts FROM v_artifact_results_enriched ORDER BY r, result_kind",
    "coverage": "SELECT toString(day) d, arch, runs, runs_linked, runs_missing_result, runs_missing_cases, runs_no_case_output, coverage FROM v_run_coverage ORDER BY d, arch",
    "subscription_artifacts": "SELECT toString(artifact_id) a, subscription_id, component FROM v_subscription_artifacts ORDER BY a, subscription_id",
    "subscribers": "SELECT event_type, event_key, toString(artifact_id) a, subscription_id, reason FROM v_artifact_subscribers ORDER BY event_type, event_key, a",
    "feed": "SELECT event_type, event_key, toString(artifact_id) a, match_reason FROM v_feed_events("
    + ", ".join(f"{k}={literal(v)}" for k, v in FEED_PARAMS.items())
    + ") ORDER BY event_type, event_key, a",
    "events": "SELECT event_type, event_key, toString(artifact_id) a FROM v_artifact_events ORDER BY event_type, event_key, a",
    "dispatch_verdicts": "SELECT toString(artifact_id) a, toString(run_id) r, test_type, state FROM v_dispatch_verdicts ORDER BY r",
    "result_reasons": "SELECT toString(artifact_id) a, toString(run_id) r, state, failure_reason FROM v_artifact_result_reasons ORDER BY r",
    "hud": "SELECT toString(run_id) r, tupleElement(metric, 'name') m, tupleElement(metric, 'benchmark_values') v FROM oss_ci_benchmark_v3 ORDER BY r, m",
}


FACTS = {
    "ref_only_rows": "SELECT count() n FROM benchmark_runs WHERE props['run.ref_only'] = '1'",
    "ref_type_rows": "SELECT count() n FROM benchmark_runs WHERE props['run.record_type'] = 'reference'",
    "ref_artifacts": "SELECT count() n FROM artifacts WHERE component = 'reference'",
    "ref_tags": "SELECT count() n FROM artifact_tags WHERE tag_family = 'reference'",
    "ref_verdicts": f"SELECT count() n FROM artifact_results WHERE artifact_id = '{A_REF}'",
    "runs_stored": "SELECT count() n FROM benchmark_runs",
    "runs_measured": "SELECT count() n FROM v_benchmark_runs_measured",
    "ref_runs_measured": "SELECT count() n FROM v_benchmark_runs_measured WHERE run_id IN ("
    + ", ".join(f"'{r}'" for r in REF_RUNS)
    + ")",
    "hud_ref_rows": "SELECT count() n FROM oss_ci_benchmark_v3 WHERE run_id IN ("
    + ", ".join(f"'{r}'" for r in REF_RUNS)
    + ")",
    "enriched_trace": "SELECT record_type, count() n FROM v_benchmark_results_enriched WHERE record_type IN ('step_trace', 'span', 'trace_phase', 'reference', 'trace_step', 'trace_kernel', 'compile_pass') GROUP BY record_type ORDER BY record_type",
    "orphans_default": "SELECT (SELECT count() FROM v_tag_list WHERE tag = 'orphan-tag') tags, (SELECT count() FROM v_artifact_results_enriched WHERE run_id = '%s') results, (SELECT count() FROM v_benchmark_results_enriched WHERE run_id = '%s') bench"
    % (u("orphan-run"), u("unlinked-run")),
    "orphans_nulls": "SELECT (SELECT count() FROM v_tag_list WHERE tag = 'orphan-tag') tags, (SELECT count() FROM v_artifact_results_enriched WHERE run_id = '%s') results, (SELECT count() FROM v_benchmark_results_enriched WHERE run_id = '%s') bench SETTINGS join_use_nulls = 1"
    % (u("orphan-run"), u("unlinked-run")),
    "excluded_list": "SELECT record_type FROM benchmark_excluded_record_types ORDER BY record_type",
}


def build(schema_dir=SCHEMA_DIR, extras=False, nulls=False):
    c = Chdb()
    if nulls:
        c.command("SET join_use_nulls = 1")
    load(schema_dir, c)
    populate(c, extras)
    return c


def dump(c):
    return {name: c.rows(sql) for name, sql in Q.items()}


def snapshot(extras, nulls=False):
    """Everything the tests read, taken in one session: chdb keeps one live session per process."""
    c = build(extras=extras, nulls=nulls)
    try:
        return {
            "dump": dump(c),
            "facts": {k: c.rows(q) for k, q in FACTS.items()},
        }
    finally:
        c.s.close()


@pytest.fixture(scope="module")
def base():
    return snapshot(extras=False)


@pytest.fixture(scope="module")
def extra():
    return snapshot(extras=True)


@pytest.fixture(scope="module")
def base_nulls():
    return snapshot(extras=False, nulls=True)


@pytest.fixture(scope="module")
def extra_nulls():
    return snapshot(extras=True, nulls=True)


def test_existing_style_rows_read_as_they_did_before_the_exclusion(base):
    assert json.loads(json.dumps(base["dump"])) == json.loads(GOLDEN.read_text())


def test_the_fixture_exercises_every_view(base):
    # A view that returns nothing could not show an exclusion.
    assert [name for name, rows in base["dump"].items() if not rows] == []


def test_references_and_artifacts_added_to_the_fixture_are_all_stored(extra):
    n = {k: v[0]["n"] for k, v in extra["facts"].items() if k.startswith("ref_")}  # noqa: E501
    assert n == {"ref_only_rows": 2, "ref_type_rows": 3, "ref_artifacts": 1, "ref_tags": 4,
                 "ref_verdicts": 6, "ref_runs_measured": 0}  # fmt: skip


def test_measured_view_drops_either_reference_marker(extra):
    f = {
        k: extra["facts"][k][0]["n"]
        for k in ("ref_runs_measured", "runs_stored", "runs_measured")
    }
    assert f["ref_runs_measured"] == 0
    assert f["runs_stored"] - f["runs_measured"] == 4


def test_the_excluded_list_is_the_one_the_readers_share(extra):
    got = [r["record_type"] for r in extra["facts"]["excluded_list"]]
    assert got == sorted(["reference", "step_trace", "span", "trace_phase", "trace_step",
                          "trace_kernel", "compile_pass"])  # fmt: skip


def test_trace_records_stay_in_the_enriched_view(extra):
    # Readers of a run's trace read these rows here; only the series views skip them.
    got = {r["record_type"]: r["n"] for r in extra["facts"]["enriched_trace"]}
    assert got == {
        "step_trace": 1,
        "span": 1,
        "trace_phase": 13,
        "reference": 1,
        "trace_step": 1,
        "trace_kernel": 1,
        "compile_pass": 1,
    }


READERS_OF_TRACE_RECORDS = ("enriched", "wide")


@pytest.mark.parametrize("reader", sorted(Q))
def test_reader_is_unchanged_by_references_and_trace_records(reader, base, extra):
    want, got = base["dump"][reader], extra["dump"][reader]
    if reader in READERS_OF_TRACE_RECORDS:
        got = [r for r in got if r["record_type"] not in TRACE_TYPES]
    assert got == want


def test_the_same_holds_under_join_use_nulls(base_nulls, extra_nulls):
    for reader in Q:
        got = extra_nulls["dump"][reader]
        if reader in READERS_OF_TRACE_RECORDS:
            got = [r for r in got if r["record_type"] not in TRACE_TYPES]
        assert got == base_nulls["dump"][reader], reader


def test_the_hud_projection_takes_no_reference_row(base, extra):
    assert extra["facts"]["hud_ref_rows"][0]["n"] == 0
    assert extra["dump"]["hud"] == base["dump"]["hud"]


@pytest.mark.parametrize("setting", ["orphans_default", "orphans_nulls"])
def test_rows_with_no_artifact_behind_them_are_kept(setting, base, extra):
    # The exclusion tests a joined column; an unjoined row must survive it, with or without nulls.
    for snap in (base, extra):
        assert snap["facts"][setting] == [{"tags": 1, "results": 1, "bench": 1}]
