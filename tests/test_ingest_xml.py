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

"""Parser and perf-dispatch tests for .github/scripts/ingest_xml.py.

The script is not a package module, so it is loaded by path. clickhouse_connect
is stubbed before import. Parse tests need no ClickHouse; dispatch tests use a
FakeClient.
"""

import importlib.util
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import pytest

INGEST_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "ingest_xml.py"
)

# The ingest imports the shared library from extensions/; it is in this repo, so put it on
# sys.path rather than requiring an install for a parse-only test.
_CHLIB = Path(__file__).resolve().parents[1] / "extensions" / "clickhouse-ingest"
if str(_CHLIB) not in sys.path:
    sys.path.insert(0, str(_CHLIB))


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


# --- classifier + quality + perf 0-row dispatch ---------------------


HF_CLASSNAME = "spyre_perf_suite.benchmark"

FULL_PROVENANCE = {
    "torch-spyre": {"commit": "abc1234", "branch": "main", "version": None},
    "flex": {"commit": "def5678", "branch": "main"},
    "deeptools": {"commit": "aaa111", "branch": "master"},
    "spyre-comms": {"commit": "bbb222", "branch": "main"},
}


class _Result:
    def __init__(self, rows):
        self.result_rows = rows


class FakeClient:
    """Answers system.columns / source_file dedup from a declared schema."""

    def __init__(self, tables=None, already_ingested=0):
        self.tables = tables if tables is not None else {}
        self.already_ingested = already_ingested
        self.inserts = []
        self.commands = []

    def query(self, sql, parameters=None):
        name = (parameters or {}).get("t", "")
        if "system.columns" in sql:
            return _Result([[c] for c in self.tables.get(name, [])])
        if "FROM benchmark_runs WHERE source_file" in sql:
            return _Result([[self.already_ingested]])
        if "FROM test_runs" in sql:
            return _Result([[0]])
        raise AssertionError(f"unexpected query: {sql}")

    def command(self, sql):
        self.commands.append(sql)

    def insert(self, table, rows, column_names=None):
        self.inserts.append((table, rows, column_names))


def _root(testcases, suite_name="pytest", suites_name=""):
    suites_attr = f" name='{suites_name}'" if suites_name else ""
    return ElementTree.fromstring(
        f"<testsuites{suites_attr}>"
        f"<testsuite name='{suite_name}'>{testcases}</testsuite>"
        "</testsuites>"
    )


def _write_suite(
    tmp_path,
    testcases,
    *,
    filename="report.xml",
    suite_name="spyre-perf-suite",
    version_info=None,
):
    props = ""
    if version_info is not None:
        props = (
            "<properties><property name='version_info' "
            f"value='{escape(json.dumps(version_info))}'/></properties>"
        )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuites name="{suite_name}">'
        f'<testsuite name="{suite_name}" tests="0">'
        f"{props}{testcases}</testsuite></testsuites>\n"
    )
    path = tmp_path / filename
    path.write_text(xml, encoding="utf-8")
    return path


def _hf_case(name: str, time: str) -> str:
    return (
        f'<testcase classname="{HF_CLASSNAME}" name="{name}" time="{time}"></testcase>'
    )


FULL_RUN_SCHEMA = {
    "benchmark_runs": [
        "run_id",
        "source_file",
        "version_info",
        "created_at",
        "workflow",
        "platform",
        "run_type",
        "quality",
        "regression_eligible",
    ],
    "perf_benchmarks": [
        "benchmark_id",
        "run_id",
        "record_type",
        "operation_name",
        "compile_ms",
        "runtime_ms",
        "mem_size_mb",
    ],
}


def _run_main(ingest, monkeypatch, xml_path, client, extra_argv=None):
    monkeypatch.setenv("CLICKHOUSE_HOST", "stub")
    monkeypatch.setattr(ingest, "get_client", lambda: client)
    argv = ["ingest_xml.py", "--xml-file", str(xml_path)]
    if extra_argv:
        argv.extend(extra_argv)
    monkeypatch.setattr(sys, "argv", argv)
    ingest.main()
    return client


def test_ordinary_junit_is_not_a_benchmark_xml(ingest):
    root = _root(
        "<testcase classname='tests.test_foo.TestBar' name='test_x' time='0.1'/>"
    )
    assert ingest.is_benchmark_xml(root) is False
    assert ingest.is_kernel_benchmark_xml(root) is False


def test_ordinary_junit_main_does_not_insert_benchmarks(ingest, monkeypatch, tmp_path):
    xml = tmp_path / "junit.xml"
    xml.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites><testsuite name='pytest'>"
        "<testcase classname='tests.test_foo.TestBar' name='test_x' "
        "time='0.1'/></testsuite></testsuites>\n",
        encoding="utf-8",
    )
    client = FakeClient(dict(FULL_RUN_SCHEMA))
    _run_main(ingest, monkeypatch, xml, client)
    tables = [t for t, _, _ in client.inserts]
    assert "benchmark_runs" not in tables
    assert "perf_benchmarks" not in tables
    assert "test_runs" in tables


def test_empty_pytest_suite_is_not_a_benchmark_xml(ingest):
    root = _root("", suite_name="pytest")
    assert ingest.is_benchmark_xml(root) is False


def test_empty_report_xml_is_a_benchmark_envelope(ingest, tmp_path):
    """Filename report.xml, not suite name, must be enough for an empty file."""
    path = _write_suite(tmp_path, "", filename="report.xml", suite_name="pytest")
    root = ElementTree.parse(path).getroot()
    assert ingest.is_benchmark_xml(root, path) is True
    assert ingest.is_benchmark_xml(root, tmp_path / "junit.xml") is False
    assert ingest.is_kernel_benchmark_xml(root) is False


def test_empty_spyre_perf_suite_is_a_benchmark_envelope(ingest):
    root = _root("", suite_name="spyre-perf-suite", suites_name="spyre-perf-suite")
    assert ingest.is_benchmark_xml(root) is True


def test_hf_classname_is_benchmark_xml(ingest):
    root = _root(
        _hf_case("perf_matmul_wall_clock_ms_1_512", "12.5"),
        suite_name="spyre-perf-suite",
    )
    assert ingest.is_benchmark_xml(root) is True
    assert ingest.is_kernel_benchmark_xml(root) is False


def test_mixed_junit_is_not_stolen_as_benchmark_xml(ingest):
    """all() must stay: one stray benchmark classname must not take the file."""
    root = _root(
        "<testcase classname='tests.test_foo.TestBar' name='test_x' time='0.1'/>"
        f'<testcase classname="{HF_CLASSNAME}" '
        "name='perf_matmul_wall_clock_ms_1' time='1.0'/>"
    )
    assert ingest.is_benchmark_xml(root) is False


def test_full_provenance_is_valid_and_regression_eligible(ingest):
    quality, eligible = ingest.classify_run_quality(json.dumps(FULL_PROVENANCE))
    assert quality == "valid"
    assert eligible == 1


def test_incomplete_version_info_is_visible_not_regression_eligible(ingest):
    missing = dict(FULL_PROVENANCE)
    del missing["spyre-comms"]
    quality, eligible = ingest.classify_run_quality(json.dumps(missing))
    assert quality == "incomplete"
    assert eligible == 0
    assert ingest.classify_run_quality(None) == ("incomplete", 0)
    empty_commit = dict(FULL_PROVENANCE)
    empty_commit["flex"] = {"commit": "N/A"}
    assert ingest.classify_run_quality(json.dumps(empty_commit)) == (
        "incomplete",
        0,
    )
    whitespace = dict(FULL_PROVENANCE)
    whitespace["flex"] = {"commit": "   "}
    assert ingest.classify_run_quality(json.dumps(whitespace)) == (
        "incomplete",
        0,
    )


def test_non_string_commit_is_incomplete(ingest):
    for bad in (True, 123, {"sha": "abc"}, ["abc"]):
        payload = dict(FULL_PROVENANCE)
        payload["flex"] = {"commit": bad}
        assert ingest.classify_run_quality(json.dumps(payload)) == (
            "incomplete",
            0,
        )


def test_quality_columns_are_stored_when_present(ingest):
    client = FakeClient(dict(FULL_RUN_SCHEMA))
    ingest.insert_benchmark_run(
        client,
        1,
        {
            "source_file": "report.xml",
            "created_at": datetime.now(UTC),
            "version_info": json.dumps(FULL_PROVENANCE),
        },
    )
    _, rows, columns = client.inserts[0]
    assert "quality" in columns
    assert "regression_eligible" in columns
    assert rows[0][columns.index("quality")] == "valid"
    assert rows[0][columns.index("regression_eligible")] == 1


def test_quality_columns_are_omitted_when_absent(ingest):
    client = FakeClient(
        {
            "benchmark_runs": [
                "run_id",
                "source_file",
                "version_info",
                "created_at",
                "workflow",
                "platform",
                "run_type",
            ]
        }
    )
    ingest.insert_benchmark_run(
        client,
        1,
        {
            "source_file": "report.xml",
            "created_at": datetime.now(UTC),
            "version_info": json.dumps(FULL_PROVENANCE),
        },
    )
    _, rows, columns = client.inserts[0]
    assert "quality" not in columns
    assert "regression_eligible" not in columns
    assert len(rows[0]) == len(columns)


def test_success_full_provenance_inserts_benchmark_rows(ingest, monkeypatch, tmp_path):
    xml = _write_suite(
        tmp_path,
        _hf_case(f"perf_matmul_wall_clock_ms_{SHAPES}", "12.5"),
        version_info=FULL_PROVENANCE,
    )
    client = FakeClient(dict(FULL_RUN_SCHEMA))
    _run_main(ingest, monkeypatch, xml, client, extra_argv=["--trigger-type", "perf"])
    tables = [t for t, _, _ in client.inserts]
    assert "benchmark_runs" in tables
    assert "perf_benchmarks" in tables
    _, rows, columns = next(
        (t, r, c) for t, r, c in client.inserts if t == "benchmark_runs"
    )
    assert rows[0][columns.index("quality")] == "valid"
    assert rows[0][columns.index("regression_eligible")] == 1


def test_incomplete_version_info_still_inserts(ingest, monkeypatch, tmp_path):
    incomplete = dict(FULL_PROVENANCE)
    del incomplete["deeptools"]
    xml = _write_suite(
        tmp_path,
        _hf_case(f"perf_matmul_wall_clock_ms_{SHAPES}", "12.5"),
        version_info=incomplete,
    )
    client = FakeClient(dict(FULL_RUN_SCHEMA))
    _run_main(ingest, monkeypatch, xml, client, extra_argv=["--trigger-type", "perf"])
    _, rows, columns = next(
        (t, r, c) for t, r, c in client.inserts if t == "benchmark_runs"
    )
    assert rows[0][columns.index("quality")] == "incomplete"
    assert rows[0][columns.index("regression_eligible")] == 0


def test_missing_report_with_trigger_type_perf_exits_nonzero(
    ingest, monkeypatch, tmp_path
):
    empty_dir = tmp_path / "xml"
    empty_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_xml.py",
            "--xml-dir",
            str(empty_dir),
            "--trigger-type",
            "perf",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        ingest.main()
    assert caught.value.code not in (0, None)


def test_zero_records_with_trigger_type_perf_exits_nonzero(
    ingest, monkeypatch, tmp_path
):
    xml = _write_suite(tmp_path, "", filename="report.xml", suite_name="pytest")
    client = FakeClient(dict(FULL_RUN_SCHEMA))
    with pytest.raises(SystemExit) as caught:
        _run_main(
            ingest,
            monkeypatch,
            xml,
            client,
            extra_argv=["--trigger-type", "perf"],
        )
    assert caught.value.code not in (0, None)
    assert client.inserts == []


def test_zero_records_without_perf_still_exits_zero(ingest, monkeypatch, tmp_path):
    xml = _write_suite(tmp_path, "", filename="report.xml")
    client = FakeClient(dict(FULL_RUN_SCHEMA))
    _run_main(ingest, monkeypatch, xml, client)
    assert client.inserts == []


def test_perf_reingest_of_existing_source_file_exits_zero(
    ingest, monkeypatch, tmp_path
):
    xml = _write_suite(
        tmp_path,
        _hf_case(f"perf_matmul_wall_clock_ms_{SHAPES}", "12.5"),
        version_info=FULL_PROVENANCE,
    )
    client = FakeClient(dict(FULL_RUN_SCHEMA), already_ingested=1)
    _run_main(ingest, monkeypatch, xml, client, extra_argv=["--trigger-type", "perf"])
    assert client.inserts == []


class _Args:
    """Minimal argparse.Namespace stand-in for the component resolver."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_component_defaults_to_this_repos_product(ingest):
    assert ingest.v2_component(_Args(component="")) == "torch-spyre"


def test_component_honours_an_explicit_override(ingest):
    # The borrowed-script case: hf-adapters' perf cell runs spyre-perf-suite through THIS
    # script, so its rows must name hf-adapters, not the script's owner.
    assert ingest.v2_component(_Args(component="hf-adapters")) == "hf-adapters"


def test_component_treats_blank_as_absent(ingest):
    assert ingest.v2_component(_Args(component="   ")) == "torch-spyre"


def test_component_survives_a_caller_that_passes_no_flag(ingest):
    # An older caller's Namespace has no `component` attribute at all; falling back rather
    # than raising keeps the ingest working while the callers are updated.
    assert ingest.v2_component(_Args()) == "torch-spyre"


def test_component_changes_test_case_identity(ingest):
    # Why a wrong stamp is not merely a mislabel: component is a test_case_id hash input, so
    # the same test reconciles to a different identity under a different component. This is
    # the defect --component exists to prevent.
    # From the library, which the ingest now uses rather than a local copy.
    from spyre_clickhouse_ingest import v2_test_case_id

    a = v2_test_case_id("torch-spyre", "T", "test_x", [])
    b = v2_test_case_id("hf-adapters", "T", "test_x", [])
    assert a and b and a != b


def test_ingest_uses_the_shared_library_not_a_local_copy(ingest):
    # The point of extensions/clickhouse-ingest is that ONE definition runs. A local copy that
    # merely agrees today passes every value-based test while drifting silently, so assert
    # object identity: editing the library must change what the ingest executes.
    import spyre_clickhouse_ingest as lib

    for name in (
        "v2_canonical_arch",
        "v2_component",
        "v2_run_id_for",
        "v2_already_ingested",
        "insert_v2",
        "extract_properties",
        "promote_xpass",
        "v2_source_and_external_run_id",
        "get_client",
        "v2_database",
        "v2_tables_present",
    ):
        assert getattr(ingest, name) is getattr(lib, name), name
    assert ingest.v2_schema is lib.schema
