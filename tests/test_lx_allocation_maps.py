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

"""Contract tests for offline LX allocation summaries and interactive maps."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools/lx_planning/allocation_maps"
spec = importlib.util.spec_from_file_location("lx_allocation_parse", TOOLS / "parse.py")
assert spec is not None and spec.loader is not None
parse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse)


def _record(graph_name: str = "../../outside") -> dict:
    operations = [{"op_index": 0, "op_type": "pointwise", "origins": []}]
    # An LX buffer lives at slot 0 only; the rejected relayout candidate has
    # no LX address and must not be reported as an HBM spill.
    buffers = [
        {
            "name": "resident",
            "address_bytes": 64,
            "reserved_size_bytes": 32,
            "allocated_size_bytes": 32,
            "start_time": 0,
            "end_time": 1,
            "uses": [0],
        },
        {
            "name": "__spyre_lx_relayout__:copy:rejected",
            "address_bytes": None,
            "reserved_size_bytes": 32,
            "allocated_size_bytes": 32,
            "start_time": 1,
            "end_time": 2,
            "uses": [0],
            "is_synthetic_relayout": True,
        },
    ]
    return {
        "schema": "spyre-lx-allocation-graph-trace-v2",
        "solver": "CpSatLayoutSolver",
        "aot_graph_name": graph_name,
        "allocation_time_unit": "solver_slot",
        "operation_count": 1,
        "buffer_count": 2,
        "resident_count": 1,
        "capacity_bytes": 128,
        "global_watermark_bytes": 96,
        "max_operation_watermark_bytes": 96,
        "timeline_matches_global_watermark": True,
        "all_resident_spans_within_capacity": True,
        "operations": operations,
        "buffers": buffers,
        "timeline": [
            {
                "op_index": 0,
                "allocation_tick": 0,
                "watermark_bytes": 96,
                "active_resident_buffers": ["resident"],
                "buffers_at_watermark": ["resident"],
            }
        ],
    }


def test_nonresident_summary_does_not_claim_spills():
    summary = parse._summary(_record(), 0)
    assert summary["resident_count"] == 1
    assert summary["nonresident_count"] == 1
    assert "spilled_count" not in summary


def test_solver_slot_timeline_checks_selected_span():
    record = _record()
    assert not parse._validate_record(record)
    record["buffers"][0]["start_time"] = 1
    assert "operation 0 timeline watermark mismatch" in parse._validate_record(record)
    # A second op's solver slot is 2, not the operation-index tick 1.
    record = _record()
    record["operations"].append({"op_index": 1, "op_type": "pointwise", "origins": []})
    record["operation_count"] = 2
    record["buffers"][0]["start_time"] = 2
    record["buffers"][0]["end_time"] = 3
    record["timeline"][0].update(
        watermark_bytes=0,
        active_resident_buffers=[],
        buffers_at_watermark=[],
    )
    record["timeline"].append(
        {
            "op_index": 1,
            "allocation_tick": 2,
            "watermark_bytes": 96,
            "active_resident_buffers": ["resident"],
            "buffers_at_watermark": ["resident"],
        }
    )
    assert not parse._validate_record(record)
    record["timeline"][1]["allocation_tick"] = 1
    assert "operation 1 allocation tick mismatch" in parse._validate_record(record)


def test_capture_uses_solver_selected_cut_lifetime(tmp_path):
    # A tiled output is copied out after its write. Its chosen box can end
    # before the last untiled graph use: this exercises the real CP-SAT extract.
    script = """
import importlib.util
import os
from types import SimpleNamespace

import torch
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType, CoreDivision, CoreDivisionBuffer, TileAxis, TileSpec,
)

spec = importlib.util.spec_from_file_location("capture_hook", os.environ["CAPTURE_HOOK"])
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
buffer = CoreDivisionBuffer(
    name="cut_tile", size=8192, uses=[0, 2],
    core_divisions=[CoreDivision(tiling=TileSpec((TileAxis(0, 2),)))],
    op_position=0, boundary=BufferType.Output,
)
solver = CpSatLayoutSolver([buffer], 8192)
allocation = solver.plan_layout_and_core_divisions()
op = SimpleNamespace(
    name="cut_tile", origins=(), data=None,
    get_name=lambda: "cut_tile",
    get_operation_name=lambda: "cut_tile",
    get_read_writes=lambda: SimpleNamespace(reads=()),
)
graph = SimpleNamespace(
    operations=[op], graph_input_names=[], env={},
    try_get_buffer=lambda _: None,
)
allocator = SimpleNamespace(
    _capture_slot_spans=solver._capture_slot_spans,
    size=8192, layout_planning=CpSatLayoutSolver,
)
hook._capture_pre_commit(allocator, graph, allocation, [])
print(buffer.end_time)
"""
    env = os.environ.copy()
    env.update(
        SPYRE_GRAPH_CAPTURE_DIR=str(tmp_path),
        CAPTURE_HOOK=str(TOOLS / "sitecustomize.py"),
    )
    run = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    capture = next(tmp_path.glob("allocations.*.jsonl"))
    record = json.loads(capture.read_text(encoding="utf-8"))
    assert run.stdout.strip() == "3"  # original exclusive end, in op indices
    assert record["allocation_time_unit"] == "solver_slot"
    assert record["buffers"][0]["end_time"] == 2  # selected copy slot end
    assert record["buffers"][0]["address_bytes"] is not None
    assert not parse._validate_record(record)


def test_plot_keeps_untrusted_graph_name_inside_output_dir(tmp_path):
    record = _record()
    capture = tmp_path / "capture.jsonl"
    capture.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "plots"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "plot.py"),
            str(capture),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (manifest,) = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for key in ("summary", "interactive_html", "html_ledger", "operation_ledger"):
        generated = output / manifest[key]
        assert generated.parent == output
        assert generated.is_file()
        assert ".." not in manifest[key]
    assert not list(output.glob("*.png"))
    assert "png" not in manifest
    assert not (tmp_path.parent / "outside-solve-0-interactive.html").exists()
