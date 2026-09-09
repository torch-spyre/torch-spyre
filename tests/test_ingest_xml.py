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

"""Parser tests for .github/scripts/ingest_xml.py.

The script is not importable as a module (it lives outside the package and pulls
in clickhouse_connect at import time), so it is loaded by path with the driver
stubbed out. Only the pure parse path is exercised — no ClickHouse required.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

INGEST_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "ingest_xml.py"
)


@pytest.fixture(scope="module")
def ingest():
    sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))
    spec = importlib.util.spec_from_file_location("ingest_xml", INGEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_xml(tmp_path, testcases: str) -> Path:
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites><testsuite name="pytest" tests="0">\n'
        f"{testcases}\n"
        "</testsuite></testsuites>\n"
    )
    path = tmp_path / "report.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def _case(name: str, time: str) -> str:
    return (
        f'<testcase classname="perf.benchmark" name="{name}" time="{time}"></testcase>'
    )


# Shapes are part of the grouping key, so every case for one row repeats them.
SHAPES = "1_512_4096__4096_4096"


def test_op_report_metrics_pivot_into_one_row(ingest, tmp_path):
    """An op report's six metrics collapse to a single row.

    compiler_ms is the op-report spelling of compile_ms, and mem_size arrives in
    MB rather than ms.
    """
    cases = "\n".join(
        _case(f"perf_matmul_{metric}_{SHAPES}", value)
        for metric, value in [
            ("wall_clock_ms", "12.5"),
            ("cpu_ms", "3.0"),
            ("spyre_ms", "9.5"),
            ("kernel_ms", "8.0"),
            ("memory_transfer_ms", "1.5"),
            ("compiler_ms", "440.0"),
            ("mem_size_MB", "64.0"),
        ]
    )
    _, benchmarks = ingest.parse_benchmark_xml(_write_xml(tmp_path, cases))

    assert len(benchmarks) == 1
    row = benchmarks[0]
    assert row["operation_name"] == "matmul"
    assert row["total_duration_ms"] == 12.5
    assert row["kernel_mean_ms"] == 8.0
    assert row["compile_ms"] == 440.0
    assert row["mem_size_mb"] == 64.0
    assert row["runtime_ms"] is None


def test_granite_compile_spelling_lands_in_the_same_column(ingest, tmp_path):
    """Granite reports say compile_ms where op reports say compiler_ms."""
    cases = "\n".join(
        [
            _case("perf_granite_wall_clock_ms_bs1_pl512", "20.0"),
            _case("perf_granite_compile_ms_bs1_pl512", "500.0"),
            _case("perf_granite_runtime_ms_bs1_pl512", "7.5"),
        ]
    )
    _, benchmarks = ingest.parse_benchmark_xml(_write_xml(tmp_path, cases))

    assert len(benchmarks) == 1
    row = benchmarks[0]
    assert row["compile_ms"] == 500.0
    assert row["runtime_ms"] == 7.5
    assert row["mem_size_mb"] is None


def test_metrics_absent_from_the_xml_stay_null(ingest, tmp_path):
    """Reports predating the op-cost metrics must not gain bogus zeros."""
    cases = _case(f"perf_matmul_wall_clock_ms_{SHAPES}", "12.5")
    _, benchmarks = ingest.parse_benchmark_xml(_write_xml(tmp_path, cases))

    row = benchmarks[0]
    assert row["compile_ms"] is None
    assert row["runtime_ms"] is None
    assert row["mem_size_mb"] is None


@pytest.mark.parametrize(
    "name",
    [
        # Kernel XMLs are ingested by a separate parser.
        "kernel_matmul_wall_clock_ms",
        # compiler? must not swallow a longer op name.
        "perf_matmul_compilers_ms",
        "perf_matmul_bogus_ms",
    ],
)
def test_unrecognised_names_are_skipped(ingest, name, tmp_path):
    _, benchmarks = ingest.parse_benchmark_xml(_write_xml(tmp_path, _case(name, "1.0")))
    assert benchmarks == []


def test_every_stored_metric_has_a_column(ingest):
    """The insert column list must cover what the parser emits."""
    metrics = ingest._PERF_NAME_RE.groupindex
    assert "metric" in metrics
    for column in ("compile_ms", "runtime_ms", "mem_size_mb"):
        assert column in ingest._PERF_BENCHMARK_COLUMNS
