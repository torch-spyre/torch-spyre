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

"""The ops case runner and the report collector, with the ingest's own parser as the judge."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmarks" / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = load("run_kernel_case")
collector = load("collect_ingest_xml")

REPORT = (
    '<?xml version="1.0"?><testsuites><testsuite name="spyre-perf-suite" timestamp="2026-01-01T00:00:00Z">'
    '<testcase name="perf_matmul_wall_clock_ms_1_512_4096__4096_4096" time="1.0"/>'
    '<testcase name="perf_matmul_kernel_ms_1_512_4096__4096_4096" time="0.5"/>'
    '<testcase name="perf_matmul_wall_clock_ms_1_128_1024__1024_1024" time="2.0"/>'
    '<testcase name="perf_matmul_extra_wall_clock_ms_1_512_4096__4096_4096" time="3.0"/>'
    '<testcase name="perf_softmax_wall_clock_ms_1_128_1024" time="4.0"/>'
    "</testsuite></testsuites>"
)


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """A fake perf entry point that writes REPORT and counts how often it ran."""
    script = tmp_path / "fake_suite.py"
    count = tmp_path / "runs.txt"
    script.write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(count)!r}).open('a').write('x')\n"
        f"pathlib.Path(sys.argv[1], 'report.xml').write_text({REPORT!r})\n"
    )
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    monkeypatch.setenv("BENCH_SUITE_CMD", f"{sys.executable} {script} {suite_dir}")
    return suite_dir, count


def run_case(suite_dir, raw, op, shape):
    return runner.main(
        [
            "--op",
            op,
            "--shape",
            shape,
            "--suite-dir",
            str(suite_dir),
            "--raw-dir",
            str(raw),
        ]
    )


def names(raw):
    return sorted(
        tc.get("name")
        for tc in runner.ET.parse(raw / "report.xml").getroot().iter("testcase")
    )


def test_a_case_keeps_only_its_op_and_shape(suite, tmp_path):
    suite_dir, _ = suite
    raw = tmp_path / "raw"
    assert run_case(suite_dir, raw, "matmul", "1_512_4096__4096_4096") == 0
    assert names(raw) == [
        "perf_matmul_kernel_ms_1_512_4096__4096_4096",
        "perf_matmul_wall_clock_ms_1_512_4096__4096_4096",
    ]


def test_the_suite_runs_once_for_all_cases(suite, tmp_path):
    suite_dir, count = suite
    assert run_case(suite_dir, tmp_path / "a", "matmul", "1_512_4096__4096_4096") == 0
    assert run_case(suite_dir, tmp_path / "b", "softmax", "1_128_1024") == 0
    assert count.read_text() == "x"


def test_a_case_without_rows_fails(suite, tmp_path):
    suite_dir, _ = suite
    assert run_case(suite_dir, tmp_path / "raw", "matmul", "9_9") == 3
    assert not (tmp_path / "raw" / "report.xml").exists()


def test_the_cases_split_the_report_without_overlap(suite, tmp_path):
    suite_dir, _ = suite
    seen = []
    for op, shape in [
        ("matmul", "1_512_4096__4096_4096"),
        ("matmul", "1_128_1024__1024_1024"),
        ("softmax", "1_128_1024"),
    ]:
        raw = tmp_path / f"{op}-{shape}"
        assert run_case(suite_dir, raw, op, shape) == 0
        seen += names(raw)
    assert len(seen) == len(set(seen))
    assert not any("extra" in n for n in seen)


def test_the_ingest_parser_reads_a_collected_case_as_its_stored_id(suite, tmp_path):
    pytest.importorskip("lxml")
    pytest.importorskip("clickhouse_connect")
    pytest.importorskip("spyre_bench")
    ids = load("check_identity_ids")
    results, benchmark_id = ids.load_ingest()
    suite_dir, _ = suite
    case = next(
        c
        for c in ids.declared()
        if c[1]["disc"]["input_shapes"] == "1_512_4096__4096_4096"
        and c[1]["name"] == "matmul"
    )
    raw = tmp_path / "raw"
    assert (
        run_case(suite_dir, raw, case[1]["name"], case[1]["disc"]["input_shapes"]) == 0
    )
    _, records = results.parse_benchmark_xml(
        raw / "report.xml", "bench-on-demand", "", "x86_64"
    )
    written = [
        benchmark_id.derive(
            results.COMPONENT_DEFAULT, e["name"], e["tags"], e["disc"], e["disc_keys"]
        )
        for e in results._bench_entries(records)
    ]
    assert written == [case[2]]


def test_the_collector_gathers_only_cases_that_ran_ok(tmp_path):
    results = tmp_path / "res"
    for name, status in [("a_s1", "ok"), ("b_s1", "failed"), ("c_s1", "ok")]:
        (results / "cases" / name / "raw").mkdir(parents=True)
        (results / "cases" / name / "raw" / "report.xml").write_text("<testsuites/>")
    (results / "run.json").write_text(
        json.dumps(
            {
                "run_id": "11111111-1111-1111-1111-111111111111",
                "cases": [
                    {"name": n, "status": s, "dir": f"cases/{n}"}
                    for n, s in [("a_s1", "ok"), ("b_s1", "failed"), ("c_s1", "ok")]
                ],
            }
        )
    )
    run_id, stored, count = collector.collect(results, tmp_path / "xml")
    assert (run_id, stored, count) == (
        "11111111-1111-1111-1111-111111111111",
        "",
        2,
    )
    assert sorted(p.name for p in (tmp_path / "xml").iterdir()) == [
        "a_s1.xml",
        "c_s1.xml",
    ]
