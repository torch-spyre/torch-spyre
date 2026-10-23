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


"""Tests for tools/frontend_timing/summarize.py, loaded by path since it is a script.

Outside tests/inductor/, whose session fixtures touch the Spyre device.
"""

import csv
import importlib.util
import json
import os
import sys

import pytest

_SUMMARIZE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "frontend_timing",
    "summarize.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fts_summarize", _SUMMARIZE)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its own module out of sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summarize_mod = _load_module()


def _record(
    tmp_path,
    name,
    *,
    workload="mlp",
    params=None,
    compile_ms=100.0,
    backend_ms=0.0,
    ops=42,
    pipeline=True,
    passes=None,
    extra_events=None,
    meta=None,
    compiled=True,
):
    """Write one synthetic record and return its path."""
    ms = 1_000_000
    events = [
        {
            "name": summarize_mod.COMPILE_EVENT,
            "ordinal": 0,
            "parent_ordinal": None,
            "inclusive_ns": int(compile_ms * ms),
            "self_ns": int(compile_ms * ms),
        },
    ]
    if pipeline:
        events.append(
            {
                "name": summarize_mod.GRAPH_PIPELINE,
                "ordinal": 1,
                "parent_ordinal": 0,
                "inclusive_ns": int(compile_ms * ms * 0.5),
                "self_ns": 0,
                "meta": {"input_operations": ops, "output_operations": ops},
            }
        )
    if backend_ms:
        events.append(
            {
                "name": "stage:SpyreAsyncCompile:backend_compile",
                "ordinal": 2,
                "parent_ordinal": 0,
                "inclusive_ns": int(backend_ms * ms),
                "self_ns": int(backend_ms * ms),
                "meta": {"tool": "dbo-opt"},
            }
        )
    for index, (pass_name, pass_ms) in enumerate((passes or {}).items()):
        events.append(
            {
                "name": summarize_mod.GRAPH_PASS_PREFIX + pass_name,
                "ordinal": 10 + index,
                "parent_ordinal": 1,
                "inclusive_ns": int(pass_ms * ms),
                "self_ns": int(pass_ms * ms),
                "meta": {"input_operations": ops, "output_operations": ops},
            }
        )
    events += extra_events or []

    payload = {
        "meta": {
            "workload": workload,
            "sample": 1,
            "cold": True,
            "compiled": compiled,
            "params": dict(params or {}),
            **(meta or {}),
        },
        "events": events,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _summary(tmp_path):
    """The single point summarized from ``tmp_path``."""
    records, _ = summarize_mod.load_records(str(tmp_path))
    (summary,) = summarize_mod.summarize(records)
    return summary


class TestLoading:
    def test_reads_records_and_ignores_subdirectories(self, tmp_path):
        _record(tmp_path, "a.json")
        warmup = tmp_path / "warmup"
        warmup.mkdir()
        _record(warmup, "discarded.json", compile_ms=9999.0)

        records, problems = summarize_mod.load_records(str(tmp_path))
        assert len(records) == 1
        assert problems == []

    def test_record_without_a_compile_is_excluded_and_reported(self, tmp_path):
        path = tmp_path / "died.json"
        path.write_text(json.dumps({"meta": {"workload": "mlp"}, "events": []}))
        _record(tmp_path, "good.json")

        records, problems = summarize_mod.load_records(str(tmp_path))
        # A process that died before compiling would otherwise read as a fast one.
        assert len(records) == 1
        assert any("no stage:compile_fx:spyre_compile" in p for p in problems)

    def test_record_written_mid_region_is_excluded(self, tmp_path):
        _record(
            tmp_path,
            "open.json",
            extra_events=[
                {
                    "name": "stage:SpyreAsyncCompile:generate_bundle",
                    "ordinal": 3,
                    "parent_ordinal": 0,
                    "inclusive_ns": 0,
                    "self_ns": 0,
                    "open": True,
                }
            ],
        )
        records, problems = summarize_mod.load_records(str(tmp_path))
        assert records == []
        assert any("mid-region" in p for p in problems)

    def test_a_swallowed_error_does_not_exclude_a_successful_compile(self, tmp_path):
        # The recorder stamps an error on any region an exception passed through, even
        # one the caller swallows -- provenance collection does exactly that. Excluding
        # on that dropped successful compiles.
        _record(
            tmp_path,
            "swallowed.json",
            extra_events=[
                {
                    "name": "stage:SpyreAsyncCompile:kernel_provenance",
                    "ordinal": 4,
                    "parent_ordinal": 1,
                    "inclusive_ns": 1_000_000,
                    "self_ns": 1_000_000,
                    "error": "KeyError: 'origins'",
                }
            ],
        )
        records, problems = summarize_mod.load_records(str(tmp_path))
        assert len(records) == 1
        assert problems == []

    def test_a_compile_that_did_not_complete_is_excluded(self, tmp_path):
        _record(
            tmp_path,
            "failed.json",
            compiled=False,
            extra_events=[
                {
                    "name": summarize_mod.GRAPH_PASS_PREFIX + "span_reduction",
                    "ordinal": 4,
                    "parent_ordinal": 1,
                    "inclusive_ns": 1_000_000,
                    "self_ns": 1_000_000,
                    "error": "InductorError: Unsupported",
                }
            ],
        )
        _record(tmp_path, "good.json")
        records, problems = summarize_mod.load_records(str(tmp_path))
        # A record is written at process exit even when the compile raised, and those
        # times measure a failure rather than a compile.
        assert len(records) == 1
        assert any("did not complete" in p for p in problems)
        # The erroring region is still named, as a pointer to what went wrong.
        assert any("span_reduction" in p for p in problems)

    def test_unreadable_record_is_reported_not_fatal(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not json")
        _record(tmp_path, "good.json")
        records, problems = summarize_mod.load_records(str(tmp_path))
        assert len(records) == 1
        assert any("unreadable" in p for p in problems)

    def test_missing_directory_is_a_clean_error(self, tmp_path):
        with pytest.raises(SystemExit):
            summarize_mod.load_records(str(tmp_path / "nope"))


class TestSummary:
    def test_minimum_across_samples(self, tmp_path):
        for index, ms in enumerate((100.0, 400.0, 200.0)):
            _record(tmp_path, f"s{index}.json", params={"layers": 2}, compile_ms=ms)
        summary = _summary(tmp_path)

        # The minimum: co-tenant interference only ever makes a compile slower, so the
        # fastest sample is the least contaminated one. Median would be 200, mean 233.
        assert summary.total_ms == pytest.approx(100.0)
        assert summary.samples == 3
        assert summary.point == "mlp-layers2"

    def test_frontend_is_total_minus_backend(self, tmp_path):
        _record(tmp_path, "s.json", compile_ms=500.0, backend_ms=200.0)
        summary = _summary(tmp_path)

        assert summary.total_ms == pytest.approx(500.0)
        assert summary.backend_ms == pytest.approx(200.0)
        assert summary.frontend_ms == pytest.approx(300.0)

    def test_frontend_only_records_have_no_backend_share(self, tmp_path):
        _record(tmp_path, "s.json", compile_ms=500.0, meta={"frontend_only": True})
        summary = _summary(tmp_path)
        # Absent, not zero: a skipped backend and a 0 ms backend are different facts.
        assert summary.backend_ms is None
        assert "backend_ms" not in summary.measurements
        assert summary.frontend_ms == pytest.approx(500.0)

    def test_points_are_grouped_by_workload_and_parameters(self, tmp_path):
        _record(tmp_path, "a.json", params={"layers": 1})
        _record(tmp_path, "b.json", params={"layers": 1})
        _record(tmp_path, "c.json", params={"layers": 4})
        _record(tmp_path, "d.json", workload="flash", params={"Lk": 512})

        records, _ = summarize_mod.load_records(str(tmp_path))
        summaries = summarize_mod.summarize(records)
        points = [s.point for s in summaries]
        assert points == ["flash-Lk512", "mlp-layers1", "mlp-layers4"]
        assert [s.samples for s in summaries] == [1, 2, 1]

    def test_run_metadata_is_not_mistaken_for_a_parameter(self, tmp_path):
        _record(
            tmp_path,
            "s.json",
            params={"layers": 2},
            meta={"git_sha": "abc1234", "pid": 999, "torch_version": "2.13.0"},
        )
        summary = _summary(tmp_path)
        # Otherwise every sample lands in its own point and nothing has a median.
        assert summary.params == {"layers": 2}
        assert summary.point == "mlp-layers2"

    def test_graph_size_comes_from_the_pipeline_event(self, tmp_path):
        _record(tmp_path, "s.json", ops=260)
        summary = _summary(tmp_path)
        assert summary.graph_operations == 260

    def test_passes_and_buckets_are_collected_by_prefix(self, tmp_path):
        _record(
            tmp_path,
            "s.json",
            passes={"span_reduction": 30.0, "deadcode_elimination": 5.0},
            backend_ms=10.0,
        )
        summary = _summary(tmp_path)

        span = "pass.CustomPreSchedulingPasses.span_reduction_ms"
        passes_ms = summary.regions_ms("pass.")
        assert passes_ms[span] == 30.0
        # Ranked, so the expensive region reads first.
        assert list(passes_ms)[0].endswith("span_reduction_ms")
        stages = summary.regions_ms("stage.")
        assert "stage.SpyreAsyncCompile.backend_compile_ms" in stages


class TestRendering:
    def test_markdown_carries_every_point(self, tmp_path):
        _record(tmp_path, "a.json", params={"layers": 1})
        _record(tmp_path, "b.json", workload="flash", params={"Lk": 512})
        records, _ = summarize_mod.load_records(str(tmp_path))
        table = summarize_mod.render_markdown(summarize_mod.summarize(records))
        assert "mlp-layers1" in table and "flash-Lk512" in table

    def test_csv_rows_carry_graph_size_for_plotting(self, tmp_path):
        _record(
            tmp_path,
            "a.json",
            params={"layers": 2},
            ops=99,
            passes={"span_reduction": 30.0},
        )
        records, _ = summarize_mod.load_records(str(tmp_path))
        out = tmp_path / "out.csv"
        summarize_mod.write_csv(summarize_mod.summarize(records), str(out))

        with open(out, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert {r["region"] for r in rows} >= {"total_ms", "frontend_ms"}
        # Graph size on every row is what lets a plot skip a join.
        assert all(r["graph_operations"] == "99" for r in rows)
        assert all(json.loads(r["params"]) == {"layers": 2} for r in rows)


class TestMeasurements:
    """The per-sample arrays the warehouse column wants."""

    def test_metrics_are_arrays_of_samples_not_medians(self, tmp_path):
        _record(tmp_path, "a.json", compile_ms=100.0)
        _record(tmp_path, "b.json", compile_ms=300.0)
        summary = _summary(tmp_path)

        # Both samples survive: benchmark_runs.measurements is an array per metric so
        # variance stays recomputable, and a median here would throw that away.
        assert sorted(summary.measurements["total_ms"]) == [100.0, 300.0]
        assert summary.total_ms == 100.0

    def test_metric_names_follow_the_grammar(self, tmp_path):
        _record(tmp_path, "a.json", passes={"span_reduction": 30.0}, backend_ms=10.0)
        summary = _summary(tmp_path)

        keys = summary.measurements
        assert "pass.CustomPreSchedulingPasses.span_reduction_ms" in keys
        assert "stage.SpyreAsyncCompile.backend_compile_ms" in keys
        # No colons anywhere: these become warehouse Map keys.
        assert not [k for k in summary.measurements if ":" in k]

    def test_the_compile_root_is_not_also_a_stage_metric(self, tmp_path):
        _record(tmp_path, "a.json", compile_ms=100.0)
        summary = _summary(tmp_path)

        # It is already total_ms; emitting it twice would let a careless sum over
        # stage.* double the entire compile.
        assert summarize_mod.metric_key(summarize_mod.COMPILE_EVENT) not in (
            summary.measurements
        )

    def test_frontend_is_subtracted_inside_each_sample(self, tmp_path):
        _record(tmp_path, "a.json", compile_ms=100.0, backend_ms=40.0)
        _record(tmp_path, "b.json", compile_ms=300.0, backend_ms=100.0)
        summary = _summary(tmp_path)

        # 60 and 200, then the median -- not median(total) - median(backend), which
        # would pair numbers from two different processes.
        assert sorted(summary.measurements["frontend_ms"]) == [60.0, 200.0]
        assert summary.frontend_ms == 60.0

    def test_a_metric_only_one_sample_carries_stays_short(self, tmp_path):
        _record(tmp_path, "a.json", passes={"span_reduction": 30.0})
        _record(tmp_path, "b.json")
        summary = _summary(tmp_path)

        # Padding the absent sample with a zero would report a 50% improvement that
        # never happened.
        assert summary.measurements[
            "pass.CustomPreSchedulingPasses.span_reduction_ms"
        ] == [30.0]
        assert len(summary.measurements["total_ms"]) == 2


class TestCounters:
    """Analysis-call counters ride along in pass meta, with no list to keep in sync."""

    def _counting_pass(self, ms=10.0, **counters):
        return {
            "name": summarize_mod.GRAPH_PASS_PREFIX + "dedup_and_promote_constants",
            "ordinal": 20,
            "parent_ordinal": 1,
            "inclusive_ns": int(ms * 1_000_000),
            "self_ns": int(ms * 1_000_000),
            "meta": {"input_operations": 42, "output_operations": 42, **counters},
        }

    def test_unknown_numeric_pass_meta_becomes_a_counter(self, tmp_path):
        _record(
            tmp_path,
            "a.json",
            extra_events=[self._counting_pass(**{"read_writes.extractions": 900})],
        )
        summary = _summary(tmp_path)

        # Generic on purpose: a counter added to the compiler must appear here without
        # this file or summarize.py being edited.
        assert summary.measurements["counter.read_writes.extractions"] == [900.0]

    def test_graph_sizes_are_not_mistaken_for_counters(self, tmp_path):
        _record(tmp_path, "a.json", extra_events=[self._counting_pass()])
        summary = _summary(tmp_path)

        assert not [k for k in summary.measurements if "counter.input_operations" in k]
        assert not [k for k in summary.measurements if "counter.output_operations" in k]

    def test_counters_are_summed_across_passes(self, tmp_path):
        first = self._counting_pass(**{"read_writes.misses": 5})
        second = dict(self._counting_pass(**{"read_writes.misses": 7}), ordinal=21)
        second["name"] = summarize_mod.GRAPH_PASS_PREFIX + "span_reduction"
        _record(tmp_path, "a.json", extra_events=[first, second])
        summary = _summary(tmp_path)

        assert summary.measurements["counter.read_writes.misses"] == [12.0]


class TestArmsAndRunMetrics:
    def test_two_arms_of_one_point_stay_separate(self, tmp_path):
        _record(tmp_path, "a.json", params={"S": 512})
        _record(
            tmp_path,
            "b.json",
            params={"S": 512},
            meta={"env_arm": "SPYRE_LX_PLANNER_RELAYOUT=0"},
        )
        records, _ = summarize_mod.load_records(str(tmp_path))
        summaries = summarize_mod.summarize(records)

        # Averaging an arm into its own control is the one thing an arm must never do.
        assert len(summaries) == 2
        assert {s.arm for s in summaries} == {"", "SPYRE_LX_PLANNER_RELAYOUT=0"}
        assert any("+SPYRE_LX_PLANNER_RELAYOUT=0" in s.point for s in summaries)

    def test_the_arm_is_not_treated_as_a_workload_parameter(self, tmp_path):
        _record(tmp_path, "a.json", params={"S": 512}, meta={"env_arm": "SENCORES=1"})
        summary = _summary(tmp_path)
        assert summary.params == {"S": 512}

    def test_run_level_metrics_are_collected(self, tmp_path):
        _record(
            tmp_path,
            "a.json",
            meta={
                "peak_rss_kb": 2_500_000,
                "compile_wall_ms": 3610.0,
                "kernels_skipped": 17,
            },
        )
        summary = _summary(tmp_path)

        assert summary.measurements["peak_rss_kb"] == [2_500_000.0]
        assert summary.measurements["compile_wall_ms"] == [3610.0]
        assert summary.measurements["kernels_skipped"] == [17.0]
        assert summary.params == {}


class TestRowsFile:
    def test_rows_file_shape(self, tmp_path):
        _record(
            tmp_path,
            "a.json",
            params={"layers": 2},
            passes={"span_reduction": 30.0},
            meta={"git_sha": "deadbee", "torch_version": "2.13.0"},
        )
        records, _ = summarize_mod.load_records(str(tmp_path))
        out = tmp_path / "rows.json"
        summarize_mod.write_json(
            summarize_mod.summarize(records), records, str(out), tier="nightly"
        )
        payload = json.loads(out.read_text())

        assert payload["schema"] == summarize_mod.ROWS_SCHEMA
        assert payload["meta"]["tier"] == "nightly"
        assert payload["meta"]["git_sha"] == "deadbee"
        assert payload["meta"]["generated_at"]
        (point,) = payload["points"]
        assert point["workload"] == "mlp"
        assert point["params"] == {"layers": 2}
        assert point["arm"] == ""
        assert point["samples"] == 1
        assert point["measurements"]["total_ms"] == [100.0]

    def test_provenance_comes_from_the_record_not_the_environment(self, tmp_path):
        _record(tmp_path, "a.json", meta={"git_sha": "abc1234"})
        records, _ = summarize_mod.load_records(str(tmp_path))
        # The sweep may be summarized on a different machine from the one that ran it.
        assert summarize_mod.provenance(records)["git_sha"] == "abc1234"


class TestAbsentIsNotZero:
    def test_a_record_with_no_pre_scheduling_pipeline_reports_no_graph_size(
        self, tmp_path
    ):
        # Only the compile root, so nothing ever reported a graph. A 0 here would read
        # as a graph that shrank to nothing, which on a delta is a total regression.
        _record(tmp_path, "bare.json", pipeline=False)
        records, problems = summarize_mod.load_records(str(tmp_path))
        assert problems == []
        (summary,) = summarize_mod.summarize(records)
        assert "graph_operations" not in summary.measurements
        assert "graph_nodes" not in summary.measurements
        assert summary.graph_operations is None
        # ...and the table prints a dash rather than inventing a zero.
        assert "| - |" in summarize_mod.render_markdown([summary])

    def test_graph_nodes_comes_from_a_pipeline_that_reported_one(self, tmp_path):
        _record(
            tmp_path,
            "a.json",
            extra_events=[
                {
                    # The pipeline that actually opens a region on a real compile.
                    # CustomPreGradPasses returns first unless a node carries
                    # meta["val"] on a Spyre device, so reading it produced
                    # graph_nodes in 0 of 78 records of a real sweep.
                    "name": summarize_mod.FX_PIPELINE,
                    "ordinal": 30,
                    "parent_ordinal": 0,
                    "inclusive_ns": 5_000_000,
                    "self_ns": 5_000_000,
                    "meta": {"input_nodes": 77, "output_nodes": 77, "passes": 3},
                }
            ],
        )
        summary = _summary(tmp_path)
        assert summary.measurements["graph_nodes"] == [77.0]
        # 'passes' is a graph descriptor, not an analysis count.
        assert "counter.passes" not in summary.measurements


class TestStageCounters:
    """Counts on stage regions, which the README's add-a-metric recipe produces."""

    @staticmethod
    def _stage(name, ordinal, **meta):
        return {
            "name": name,
            "ordinal": ordinal,
            "parent_ordinal": 0,
            "inclusive_ns": 1_000_000,
            "self_ns": 1_000_000,
            "meta": meta,
        }

    def test_a_count_on_a_stage_region_is_collected(self, tmp_path):
        # The README tells a contributor to wrap a region in stage(...) and put counts
        # in its meta, and says the summarizer needs no change. It did.
        _record(
            tmp_path,
            "a.json",
            extra_events=[
                self._stage("stage:Scratchpad:cpsat_solve", 30, variables=1200)
            ],
        )
        summary = _summary(tmp_path)
        key = "counter.Scratchpad.cpsat_solve.variables"
        assert summary.measurements[key] == [1200.0]

    def test_stage_counts_are_keyed_by_region_not_merged(self, tmp_path):
        # Two regions can carry the same key for different things -- the scratchpad
        # records `buffers` on more than one -- and summing them reports neither.
        _record(
            tmp_path,
            "a.json",
            extra_events=[
                self._stage("stage:Scratchpad:prepare_buffers", 30, buffers=40),
                self._stage("stage:Scratchpad:cpsat_solve", 31, buffers=7),
            ],
        )
        summary = _summary(tmp_path)
        prepare = "counter.Scratchpad.prepare_buffers.buffers"
        solve = "counter.Scratchpad.cpsat_solve.buffers"
        assert summary.measurements[prepare] == [40.0]
        assert summary.measurements[solve] == [7.0]
        assert "counter.buffers" not in summary.measurements

    def test_pass_counts_still_sum_under_their_bare_key(self, tmp_path):
        first = {
            "name": summarize_mod.GRAPH_PASS_PREFIX + "a",
            "ordinal": 40,
            "parent_ordinal": 1,
            "inclusive_ns": 1_000_000,
            "self_ns": 1_000_000,
            "meta": {"read_writes.misses": 5},
        }
        second = dict(first, ordinal=41, name=summarize_mod.GRAPH_PASS_PREFIX + "b")
        _record(tmp_path, "a.json", extra_events=[first, second])
        summary = _summary(tmp_path)
        # Per-pass instances of one measurement; a compile total is what is wanted.
        assert summary.measurements["counter.read_writes.misses"] == [10.0]


class TestDurationsAreNotCounters:
    def test_a_clock_named_counter_goes_under_duration(self, tmp_path):
        # read_writes.extract_ns and cpsat_solve.solver_wall_s are clocks a pass
        # reports through the same meta dict as its counts. On a real sweep they
        # spread 14% and 64% within one point while every count beside them was
        # identical, so emitting them as counter.* made "the counters are
        # reproducible" false of the prefix as a whole.
        event = {
            "name": summarize_mod.GRAPH_PASS_PREFIX + "a",
            "ordinal": 30,
            "parent_ordinal": 1,
            "inclusive_ns": 1_000_000,
            "self_ns": 1_000_000,
            "meta": {"read_writes.misses": 5, "read_writes.extract_ns": 900_000},
        }
        _record(tmp_path, "a.json", extra_events=[event])
        summary = _summary(tmp_path)
        assert summary.measurements["counter.read_writes.misses"] == [5.0]
        assert summary.measurements["duration.read_writes.extract_ns"] == [900_000.0]
        assert "counter.read_writes.extract_ns" not in summary.measurements

    def test_the_rows_schema_says_which_shape_it_is(self, tmp_path):
        # Renaming a metric changes the rows file, so a consumer has to be able
        # to tell the two apart.
        assert summarize_mod.ROWS_SCHEMA.endswith("/2")


class TestSolveTimeBound:
    @staticmethod
    def _solve(status):
        return {
            "name": "stage:Scratchpad:cpsat_solve",
            "ordinal": 40,
            "parent_ordinal": 1,
            "inclusive_ns": 60_000_000_000,
            "self_ns": 60_000_000_000,
            "meta": {"status": status, "variables": 100, "constraints": 200},
        }

    def test_a_feasible_solve_is_flagged(self, tmp_path):
        # FEASIBLE means the solve stopped at its time limit and returned
        # whichever plan it had: a different plan does different downstream work,
        # and the point cannot report a harder solve as more time.
        _record(tmp_path, "a.json", extra_events=[self._solve("FEASIBLE")])
        assert _summary(tmp_path).measurements["solve_time_bound"] == [1.0]

    def test_an_optimal_solve_is_not(self, tmp_path):
        _record(tmp_path, "a.json", extra_events=[self._solve("OPTIMAL")])
        assert _summary(tmp_path).measurements["solve_time_bound"] == [0.0]


class TestReentrantRegions:
    def test_a_region_inside_itself_is_counted_once(self, tmp_path):
        # GraphLowering.run recurses through subgraphs and the _recursive_*
        # passes are named for doing so, so summing every event of one name
        # counts the inner time at each level: 71% over on a Granite layer at
        # S=1024, 184% on _recursive_joint_graph_passes.
        outer = {
            "name": "stage:torch:GraphLowering.run",
            "ordinal": 50,
            "parent_ordinal": 0,
            "inclusive_ns": 100_000_000,
            "self_ns": 40_000_000,
        }
        inner = dict(outer, ordinal=51, parent_ordinal=50, inclusive_ns=60_000_000)
        _record(tmp_path, "a.json", extra_events=[outer, inner])
        records, _ = summarize_mod.load_records(str(tmp_path))
        totals = records[0].by_prefix("stage:torch:GraphLowering.run")
        assert totals["stage:torch:GraphLowering.run"] == 100_000_000

    def test_nesting_through_an_unrelated_region_still_counts_once(self, tmp_path):
        # The recursion need not be immediate, so the whole parent chain is walked.
        outer = {
            "name": "stage:torch:GraphLowering.run",
            "ordinal": 60,
            "parent_ordinal": 0,
            "inclusive_ns": 90_000_000,
            "self_ns": 10_000_000,
        }
        between = {
            "name": "stage:torch:fx_codegen_and_compile",
            "ordinal": 61,
            "parent_ordinal": 60,
            "inclusive_ns": 80_000_000,
            "self_ns": 10_000_000,
        }
        inner = dict(outer, ordinal=62, parent_ordinal=61, inclusive_ns=70_000_000)
        _record(tmp_path, "a.json", extra_events=[outer, between, inner])
        records, _ = summarize_mod.load_records(str(tmp_path))
        totals = records[0].by_prefix("stage:torch:GraphLowering.run")
        assert totals["stage:torch:GraphLowering.run"] == 90_000_000

    def test_sibling_regions_of_one_name_both_count(self, tmp_path):
        # Two entries that do not nest are two separate spans of real time.
        first = {
            "name": "stage:torch:GraphLowering.run",
            "ordinal": 70,
            "parent_ordinal": 0,
            "inclusive_ns": 30_000_000,
            "self_ns": 30_000_000,
        }
        second = dict(first, ordinal=71, inclusive_ns=20_000_000)
        _record(tmp_path, "a.json", extra_events=[first, second])
        records, _ = summarize_mod.load_records(str(tmp_path))
        totals = records[0].by_prefix("stage:torch:GraphLowering.run")
        assert totals["stage:torch:GraphLowering.run"] == 50_000_000


class TestIdentityConflicts:
    def test_records_from_one_build_have_no_conflict(self, tmp_path):
        for name in ("a.json", "b.json"):
            _record(tmp_path, name, meta={"git_sha": "abc1234"})
        records, _ = summarize_mod.load_records(str(tmp_path))
        assert summarize_mod.identity_conflicts(records) == []

    def test_two_builds_in_one_directory_are_detected(self, tmp_path):
        _record(tmp_path, "old.json", meta={"git_sha": "aaaaaaa"})
        _record(tmp_path, "new.json", meta={"git_sha": "bbbbbbb"})
        records, _ = summarize_mod.load_records(str(tmp_path))
        conflicts = summarize_mod.identity_conflicts(records)
        assert any("git_sha" in c for c in conflicts)

    def test_a_differing_config_is_detected(self, tmp_path):
        _record(tmp_path, "a.json", meta={"spyre_config": {"sencores": 32}})
        _record(tmp_path, "b.json", meta={"spyre_config": {"sencores": 1}})
        records, _ = summarize_mod.load_records(str(tmp_path))
        # The arm facility sets config deliberately and labels it; an unlabelled
        # difference means a stray export changed the compile.
        conflicts = summarize_mod.identity_conflicts(records)
        assert any("spyre_config" in c for c in conflicts)

    def test_per_record_config_entries_are_not_a_conflict(self, tmp_path):
        # timing_out is each sample's own destination, set per child by the
        # driver, so comparing the config verbatim rejected every sweep with
        # more than one sample -- which is every real one.
        for n in (1, 2):
            dest = str(tmp_path / f"mlp-run{n}.json")
            cfg = dict(sencores=32, timing_out=dest)
            _record(tmp_path, f"r{n}.json", meta={"spyre_config": cfg})
        records, _ = summarize_mod.load_records(str(tmp_path))
        assert summarize_mod.identity_conflicts(records) == []

    def test_config_ignore_prefix_is_honoured(self, tmp_path):
        # The config names its own volatile entries; a reader that ignores that
        # list reports a conflict the config already said not to compare.
        for n, cpus in ((1, 8), (2, 16)):
            cfg = {"_cache_config_ignore_prefix": ["num_cpus"], "num_cpus": cpus}
            _record(tmp_path, f"r{n}.json", meta={"spyre_config": cfg})
        records, _ = summarize_mod.load_records(str(tmp_path))
        assert summarize_mod.identity_conflicts(records) == []

    def test_main_summarizes_a_sweep_whose_records_differ_only_per_sample(
        self, tmp_path, capsys
    ):
        for n in (1, 2, 3):
            dest = str(tmp_path / f"mlp-run{n}.json")
            cfg = dict(sencores=32, timing_out=dest)
            _record(tmp_path, f"mlp-run{n}.json", meta={"spyre_config": cfg})
        assert summarize_mod.main([str(tmp_path)]) == 0
        assert "refusing to summarize" not in capsys.readouterr().err

    def test_an_arm_changing_the_config_is_not_a_conflict(self, tmp_path):
        # An arm exists to change the config: the three shipped arms set
        # lx_planner_relayout, lx_solver_relayout_groups_per_edge and sencores,
        # all read at import. Comparing every record against every other
        # therefore rejected every weekly sweep, which is the only tier holding
        # an arm and the only tier scaling.py can fit.
        for n in (1, 2):
            cfg = dict(sencores=32)
            _record(tmp_path, f"c{n}.json", meta={"spyre_config": cfg})
        for n in (1, 2):
            cfg = dict(sencores=1)
            _record(
                tmp_path,
                f"a{n}.json",
                meta={"spyre_config": cfg, "env_arm": "SENCORES=1"},
            )
        records, _ = summarize_mod.load_records(str(tmp_path))
        assert summarize_mod.identity_conflicts(records) == []

    def test_a_stray_config_difference_inside_one_arm_is_still_caught(self, tmp_path):
        for n, cores in ((1, 1), (2, 2)):
            cfg = dict(sencores=cores)
            _record(
                tmp_path,
                f"a{n}.json",
                meta={"spyre_config": cfg, "env_arm": "SENCORES=1"},
            )
        records, _ = summarize_mod.load_records(str(tmp_path))
        conflicts = summarize_mod.identity_conflicts(records)
        assert any("within arm SENCORES=1" in c for c in conflicts)

    def test_allow_mixed_still_says_what_it_tolerated(self, tmp_path, capsys):
        _record(tmp_path, "old.json", meta={"git_sha": "aaaaaaa"})
        _record(tmp_path, "new.json", meta={"git_sha": "bbbbbbb"})
        assert summarize_mod.main([str(tmp_path), "--allow-mixed"]) == 0
        err = capsys.readouterr().err
        assert "warning: git_sha differs" in err

    def test_main_refuses_mixed_records_unless_told_otherwise(self, tmp_path, capsys):
        _record(tmp_path, "old.json", meta={"git_sha": "aaaaaaa"})
        _record(tmp_path, "new.json", meta={"git_sha": "bbbbbbb"})
        assert summarize_mod.main([str(tmp_path)]) == 1
        assert "refusing to summarize" in capsys.readouterr().err
        assert summarize_mod.main([str(tmp_path), "--allow-mixed"]) == 0
