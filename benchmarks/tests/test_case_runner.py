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

MM = "[[1, 512, 4096], [4096, 1024]]"
MM2 = "[[4, 1, 4096], [4096, 1024]]"
SM = "[[1, 512, 4096]]"


def model_tc(op, metric, time, config, mode, model="granite"):
    tags = "".join(
        f'<property name="tag" value="{t}"/>'
        for t in (f"op__{op}", f"model__{model}", f"config__{config}", f"mode__{mode}")
    )
    return (
        f'<testcase classname="perf.benchmark" name="perf_{op}_{metric}_ms" time="{time}">'
        f"<properties>{tags}</properties></testcase>"
    )


REPORT = (
    '<?xml version="1.0"?><testsuites><testsuite name="spyre-perf-suite" timestamp="2026-01-01T00:00:00Z">'
    f'<testcase classname="perf.benchmark" name="perf_matmul_wall_clock_ms_{MM}" time="1.0"/>'
    f'<testcase classname="perf.benchmark" name="perf_matmul_kernel_ms_{MM}" time="0.5"/>'
    f'<testcase classname="perf.benchmark" name="perf_matmul_wall_clock_ms_{MM2}" time="2.0"/>'
    f'<testcase classname="perf.benchmark" name="perf_matmul_extra_wall_clock_ms_{MM}" time="3.0"/>'
    f'<testcase classname="perf.benchmark" name="perf_softmax_wall_clock_ms_{SM}" time="4.0"/>'
    + model_tc(
        "granite_bs4_pl128_decode_only_full_model",
        "wall_clock",
        5,
        "full_model",
        "decode_only",
    )
    + model_tc(
        "granite_bs4_pl128_decode_only_full_model",
        "spyre",
        4,
        "full_model",
        "decode_only",
    )
    + model_tc(
        "granite_bs4_pl128_decode_only_one_block",
        "wall_clock",
        6,
        "one_block",
        "decode_only",
    )
    + model_tc(
        "other_bs1_pl512_prefill_only_full_model",
        "wall_clock",
        7,
        "full_model",
        "prefill_only",
        "other",
    )
    + "</testsuite></testsuites>"
)


def kernel_tc(op, config, mode, shape, kernel):
    tags = [
        f"op__{op}",
        f"kernel__{kernel}",
        f"config__{config}",
        f"mode__{mode}",
        f"input_shape__{shape}",
    ]
    props = "".join(f'<property name="tag" value="{t}"/>' for t in tags)
    return (
        f'<testcase classname="kernel_benchmark" name="perf_{op}_{kernel}" time="0.1">'
        f"<properties>{props}</properties></testcase>"
    )


KERNELS = (
    '<?xml version="1.0"?><testsuites><testsuite name="spyre-perf-suite" timestamp="2026-01-01T00:00:00Z">'
    + kernel_tc("matmul", "null", "null", MM, "k_a")
    + kernel_tc("matmul", "null", "null", MM2, "k_b")
    + kernel_tc(
        "granite_bs4_pl128_decode_only_full_model",
        "full_model",
        "decode_only",
        "null",
        "k_c",
    )
    + kernel_tc(
        "granite_bs4_pl128_decode_only_one_block",
        "one_block",
        "decode_only",
        "null",
        "k_d",
    )
    + "</testsuite></testsuites>"
)


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """A fake perf entry point that writes REPORT and a kernel breakdown and counts how often it ran."""
    script = tmp_path / "fake_suite.py"
    count = tmp_path / "runs.txt"
    script.write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(count)!r}).open('a').write('x')\n"
        f"pathlib.Path(sys.argv[1], 'report.xml').write_text({REPORT!r})\n"
        f"pathlib.Path(sys.argv[1], 'kernels.xml').write_text({KERNELS!r})\n"
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


def run_model(suite_dir, raw, config, mode, model="granite"):
    return runner.main(
        [
            "--model",
            model,
            "--config",
            config,
            "--mode",
            mode,
            "--suite-dir",
            str(suite_dir),
            "--raw-dir",
            str(raw),
        ]
    )


def names(raw, file="report.xml"):
    return sorted(
        tc.get("name") for tc in runner.ET.parse(raw / file).getroot().iter("testcase")
    )


def test_a_case_keeps_only_its_op_and_shape(suite, tmp_path):
    suite_dir, _ = suite
    raw = tmp_path / "raw"
    assert run_case(suite_dir, raw, "matmul", MM) == 0
    assert names(raw) == [
        f"perf_matmul_kernel_ms_{MM}",
        f"perf_matmul_wall_clock_ms_{MM}",
    ]


def test_the_suite_runs_once_for_all_cases(suite, tmp_path):
    suite_dir, count = suite
    assert run_case(suite_dir, tmp_path / "a", "matmul", MM) == 0
    assert run_case(suite_dir, tmp_path / "b", "softmax", SM) == 0
    assert run_model(suite_dir, tmp_path / "c", "full_model", "decode_only") == 0
    assert count.read_text() == "x"


def test_a_case_without_rows_fails(suite, tmp_path):
    suite_dir, _ = suite
    assert run_case(suite_dir, tmp_path / "raw", "matmul", "9_9") == 3
    assert run_model(suite_dir, tmp_path / "m", "one_block", "generate") == 3
    assert not (tmp_path / "raw" / "report.xml").exists()


def test_a_case_keeps_its_kernel_rows(suite, tmp_path):
    suite_dir, _ = suite
    raw = tmp_path / "raw"
    assert run_case(suite_dir, raw, "matmul", MM) == 0
    assert names(raw, "kernels.xml") == ["perf_matmul_k_a"]


def test_a_model_case_keeps_its_config_and_mode_only(suite, tmp_path):
    suite_dir, _ = suite
    raw = tmp_path / "raw"
    assert run_model(suite_dir, raw, "full_model", "decode_only") == 0
    assert names(raw) == [
        "perf_granite_bs4_pl128_decode_only_full_model_spyre_ms",
        "perf_granite_bs4_pl128_decode_only_full_model_wall_clock_ms",
    ]
    assert names(raw, "kernels.xml") == [
        "perf_granite_bs4_pl128_decode_only_full_model_k_c"
    ]
    assert (
        run_model(suite_dir, tmp_path / "o", "full_model", "prefill_only", "other") == 0
    )


def test_the_cases_split_the_report_without_overlap(suite, tmp_path):
    suite_dir, _ = suite
    seen = []
    for op, shape in [("matmul", MM), ("matmul", MM2), ("softmax", SM)]:
        raw = tmp_path / f"{op}-{shape}"
        assert run_case(suite_dir, raw, op, shape) == 0
        seen += names(raw)
    for config in ("full_model", "one_block"):
        raw = tmp_path / config
        assert run_model(suite_dir, raw, config, "decode_only") == 0
        seen += names(raw)
    assert len(seen) == len(set(seen))
    assert not any("extra" in n for n in seen)


def test_a_finished_run_is_copied_instead_of_run_again(suite, tmp_path, monkeypatch):
    suite_dir, count = suite
    done = tmp_path / "done"
    done.mkdir()
    (done / "report.xml").write_text(REPORT)
    (done / "kernels.xml").write_text(KERNELS)
    monkeypatch.setenv("BENCH_SUITE_FROM", str(done))
    assert run_case(suite_dir, tmp_path / "raw", "softmax", SM) == 0
    assert not count.exists()


def test_the_ingest_parser_reads_a_collected_case_as_its_stored_id(suite, tmp_path):
    pytest.importorskip("lxml")
    pytest.importorskip("clickhouse_connect")
    pytest.importorskip("spyre_bench")
    ids = load("check_identity_ids")
    results, benchmark_id = ids.load_ingest()
    suite_dir, _ = suite
    declared = {c[0]: c for c in ids.declared()}
    for case_name, args in [
        ("matmul_prefill-n1024", ("matmul", MM)),
        ("granite_full-model_decode", ("full_model", "decode_only")),
    ]:
        raw = tmp_path / case_name
        code = (
            run_case(suite_dir, raw, *args)
            if len(args) == 2 and args[0] == "matmul"
            else run_model(suite_dir, raw, *args)
        )
        assert code == 0
        _, records = results.parse_benchmark_xml(
            raw / "report.xml", "bench-on-demand", "", "x86_64"
        )
        written = [
            benchmark_id.derive(
                results.COMPONENT_DEFAULT,
                e["name"],
                e["tags"],
                e["disc"],
                e["disc_keys"],
            )
            for e in results._bench_entries(records)
        ]
        assert written == [declared[case_name][2]]
        _, kernels = results.parse_kernel_xml(raw / "kernels.xml", "bench-on-demand")
        assert len(kernels) == 1


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


def test_coverage_lists_report_rows_the_catalog_does_not_declare(
    suite, tmp_path, capsys
):
    pytest.importorskip("lxml")
    pytest.importorskip("clickhouse_connect")
    pytest.importorskip("spyre_bench")
    coverage = load("check_report_coverage")
    suite_dir, _ = suite
    runner.suite_report(suite_dir)
    assert coverage.main([str(suite_dir)]) == 0
    out = capsys.readouterr().out
    assert "not declared:" in out and "absent:" in out
    assert "not declared: op | matmul_extra | " + MM in out
    assert "not declared: model | other | full_model | prefill_only" in out
    assert coverage.main([str(suite_dir), "--strict"]) == 1


def test_coverage_is_clean_for_a_report_of_exactly_the_catalog(tmp_path, capsys):
    pytest.importorskip("lxml")
    pytest.importorskip("clickhouse_connect")
    pytest.importorskip("spyre_bench")
    ids = load("check_identity_ids")
    coverage = load("check_report_coverage")
    (tmp_path / "report.xml").write_text(ids.report_xml(ids.declared()))
    assert coverage.main([str(tmp_path), "--strict"]) == 0
    assert (
        "not declared 0; declared but absent from the report 0"
        in capsys.readouterr().out
    )


def test_a_kernel_tag_spelled_differently_from_the_report_name_still_matches(tmp_path):
    case = runner.Case(op="matmul", shape=MM)
    assert case.owns_tags({"op": "matmul", "input_shape": "1x512x4096x4096x1024"})
    assert not case.owns_tags({"op": "matmul", "input_shape": "4x1x4096x4096x1024"})
    assert runner.Case(model="granite", config="c", mode="m").owns_tags(
        {"op": "granite_x", "config": "c", "mode": "m"}
    )


def test_strict_coverage_fails_on_a_kernel_testcase_no_case_claims(suite, tmp_path):
    pytest.importorskip("lxml")
    pytest.importorskip("clickhouse_connect")
    pytest.importorskip("spyre_bench")
    coverage = load("check_report_coverage")
    ids = load("check_identity_ids")
    (tmp_path / "report.xml").write_text(ids.report_xml(ids.declared()))
    (tmp_path / "kernels.xml").write_text(KERNELS.replace("matmul", "nosuchop"))
    assert coverage.main([str(tmp_path), "--strict"]) == 1


def test_a_report_without_model_rows_lists_only_the_declared_cases_it_holds(tmp_path):
    pytest.importorskip("lxml")
    pytest.importorskip("clickhouse_connect")
    pytest.importorskip("spyre_bench")
    ids = load("check_identity_ids")
    coverage = load("check_report_coverage")
    ops = [c for c in ids.declared() if c[1]["disc"]["record_type"] == "op"]
    (tmp_path / "report.xml").write_text(ids.report_xml(ops))
    listed = tmp_path / "cases.txt"
    assert coverage.main([str(tmp_path), "--strict", "--cases-file", str(listed)]) == 0
    names = listed.read_text().split()
    assert sorted(names) == sorted(c[0] for c in ops) and len(names) == 19


def test_strict_coverage_fails_on_an_xml_no_case_can_carry(tmp_path):
    pytest.importorskip("lxml")
    pytest.importorskip("clickhouse_connect")
    pytest.importorskip("spyre_bench")
    ids = load("check_identity_ids")
    coverage = load("check_report_coverage")
    (tmp_path / "report.xml").write_text(ids.report_xml(ids.declared()))
    (tmp_path / "extra.xml").write_text("<testsuites><testsuite/></testsuites>")
    assert coverage.main([str(tmp_path), "--strict"]) == 1


def test_the_zero_rule_covers_the_whole_report_and_cpu_stands_in_for_wall_clock():
    def tc(metric, value, op="a"):
        return f'<testcase classname="perf.benchmark" name="perf_{op}_{metric}_ms_[[1]]" time="{value}"/>'

    tree = runner.ET.ElementTree(
        runner.ET.fromstring(
            "<testsuites><testsuite>"
            + tc("runtime", 0.0, "a")
            + tc("runtime", 2.0, "b")
            + tc("cpu", 3.0, "b")
            + "</testsuite></testsuites>"
        )
    )
    assert runner.uncaptured(tree) == set()
    only = runner.ET.ElementTree(
        runner.ET.fromstring(
            "<testsuites><testsuite>" + tc("runtime", 0.0) + "</testsuite></testsuites>"
        )
    )
    assert runner.uncaptured(only) == {"runtime_ms"}
    assert runner.samples_of(tree, ())["total_duration_ms"] == [3.0]


def test_two_shapes_with_the_same_digits_are_a_lint_failure(tool_ids=None):
    ids = load("check_identity_ids")
    a = (
        "x_a",
        {"name": "x", "disc": {"record_type": "op", "input_shapes": "[[1, 2], [3]]"}},
        "",
    )
    b = (
        "x_b",
        {"name": "x", "disc": {"record_type": "op", "input_shapes": "[[1], [2, 3]]"}},
        "",
    )
    assert ids.shape_collisions([a, b])
    assert not ids.shape_collisions([a])
