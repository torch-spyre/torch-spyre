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


"""Tests for the frontend timing summarizer (tools/frontend_timing/summarize.py).

Loaded by path rather than imported: the tool is a script, not a package, and it depends
on nothing but the standard library.

Deliberately outside tests/inductor/, whose session fixtures touch the Spyre device: the
summarizer is pure JSON handling, and a tool test that needs no hardware should not
require any.
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
    passes=None,
    extra_events=None,
    meta=None,
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
        {
            "name": summarize_mod.GRAPH_PIPELINE,
            "ordinal": 1,
            "parent_ordinal": 0,
            "inclusive_ns": int(compile_ms * ms * 0.5),
            "self_ns": 0,
            "meta": {"input_operations": ops, "output_operations": ops},
        },
    ]
    if backend_ms:
        events.append(
            {
                "name": "stage:SpyreAsyncCompile:backend_compile",
                "ordinal": 2,
                "parent_ordinal": 0,
                "inclusive_ns": int(backend_ms * ms),
                "self_ns": int(backend_ms * ms),
                "meta": {"tool": "dxp_standalone"},
            }
        )
    for index, (pass_name, pass_ms) in enumerate((passes or {}).items()):
        events.append(
            {
                "name": f"pass:CustomPreSchedulingPasses:{pass_name}",
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
            **(params or {}),
            **(meta or {}),
        },
        "events": events,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


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

    def test_failed_compile_is_excluded(self, tmp_path):
        _record(
            tmp_path,
            "failed.json",
            extra_events=[
                {
                    "name": "pass:CustomPreSchedulingPasses:span_reduction",
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
        assert any("compile failed" in p for p in problems)

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
    def test_medians_across_samples(self, tmp_path):
        for index, ms in enumerate((100.0, 400.0, 200.0)):
            _record(tmp_path, f"s{index}.json", params={"layers": 2}, compile_ms=ms)
        records, _ = summarize_mod.load_records(str(tmp_path))
        (summary,) = summarize_mod.summarize(records)

        # Median, not mean: the mean here would be 233.3.
        assert summary.total_ms == pytest.approx(200.0)
        assert summary.samples == 3
        assert summary.point == "mlp-layers2"

    def test_frontend_is_total_minus_backend(self, tmp_path):
        _record(tmp_path, "s.json", compile_ms=500.0, backend_ms=200.0)
        records, _ = summarize_mod.load_records(str(tmp_path))
        (summary,) = summarize_mod.summarize(records)

        assert summary.total_ms == pytest.approx(500.0)
        assert summary.backend_ms == pytest.approx(200.0)
        assert summary.frontend_ms == pytest.approx(300.0)

    def test_frontend_only_records_have_no_backend_share(self, tmp_path):
        _record(tmp_path, "s.json", compile_ms=500.0, meta={"frontend_only": True})
        records, _ = summarize_mod.load_records(str(tmp_path))
        (summary,) = summarize_mod.summarize(records)
        assert summary.backend_ms == 0.0
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
        records, _ = summarize_mod.load_records(str(tmp_path))
        (summary,) = summarize_mod.summarize(records)
        # Otherwise every sample lands in its own point and nothing has a median.
        assert summary.params == {"layers": 2}
        assert summary.point == "mlp-layers2"

    def test_graph_size_comes_from_the_pipeline_event(self, tmp_path):
        _record(tmp_path, "s.json", ops=260)
        records, _ = summarize_mod.load_records(str(tmp_path))
        (summary,) = summarize_mod.summarize(records)
        assert summary.graph_operations == 260

    def test_passes_and_buckets_are_collected_by_prefix(self, tmp_path):
        _record(
            tmp_path,
            "s.json",
            passes={"span_reduction": 30.0, "deadcode_elimination": 5.0},
            backend_ms=10.0,
        )
        records, _ = summarize_mod.load_records(str(tmp_path))
        (summary,) = summarize_mod.summarize(records)

        span = "pass:CustomPreSchedulingPasses:span_reduction"
        assert summary.passes_ms[span] == 30.0
        # Ranked, so the expensive region reads first.
        assert list(summary.passes_ms)[0].endswith("span_reduction")
        assert "stage:SpyreAsyncCompile:backend_compile" in summary.buckets_ms


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
        assert {r["region"] for r in rows} >= {"total", "frontend"}
        # Graph size on every row is what lets a plot skip a join.
        assert all(r["graph_operations"] == "99" for r in rows)
        assert all(json.loads(r["params"]) == {"layers": 2} for r in rows)
