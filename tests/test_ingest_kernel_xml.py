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

"""Kernel-XML tests for .github/scripts/ingest_xml.py.

The script is not importable as a module (it lives outside the package and pulls
in clickhouse_connect at import time), so it is loaded by path with the driver
stubbed out. No ClickHouse required: the client is a fake that records what the
script asked for.
"""

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import pytest

INGEST_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "ingest_xml.py"
)

KERNEL_CLASSNAME = "spyre_perf_suite.kernel_benchmark"
OP_CLASSNAME = "spyre_perf_suite.benchmark"


@pytest.fixture(scope="module")
def ingest():
    # Stub the driver so the script imports without a database, and take it back
    # out afterwards — a bare sys.modules insert would leak into every later test
    # in the process that genuinely imports clickhouse_connect.
    had = "clickhouse_connect" in sys.modules
    sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))
    try:
        spec = importlib.util.spec_from_file_location("ingest_xml", INGEST_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if not had:
            sys.modules.pop("clickhouse_connect", None)


class _Result:
    def __init__(self, rows):
        self.result_rows = rows


class FakeClient:
    """Answers system.columns / system.tables from a declared schema."""

    def __init__(self, tables=None, already_ingested=0):
        # {table: [column, ...]}
        self.tables = tables if tables is not None else {}
        self.already_ingested = already_ingested
        self.inserts = []
        self.commands = []

    def query(self, sql, parameters=None):
        name = (parameters or {}).get("t", "")
        if "system.columns" in sql:
            return _Result([[c] for c in self.tables.get(name, [])])
        if "system.tables" in sql:
            return _Result([[1 if name in self.tables else 0]])
        if "FROM benchmark_runs WHERE source_file" in sql:
            return _Result([[self.already_ingested]])
        raise AssertionError(f"unexpected query: {sql}")

    def command(self, sql):
        self.commands.append(sql)

    def insert(self, table, rows, column_names=None):
        self.inserts.append((table, rows, column_names))


def _root(ingest, testcases):
    return ElementTree.fromstring(
        f"<testsuites><testsuite name='pytest'>{testcases}</testsuite></testsuites>"
    )


# --- the DDL this PR removed must not come back -----------------------------


def test_script_defines_no_kernel_ddl(ingest):
    """perf_kernels / run_type come from the dashboard migration, not from here."""
    assert not hasattr(ingest, "PERF_KERNELS_DDL")


# --- detection ordering -----------------------------------------------------


def test_both_predicates_match_a_kernel_xml(ingest):
    """Why order matters in main(): is_benchmark_xml also claims these files.

    The ordering itself is pinned by test_kernel_xml_never_reaches_the_op_branch.
    """
    root = _root(
        ingest,
        f"<testcase classname='{KERNEL_CLASSNAME}' name='kernel_matmul_Total_1x512' time='1.5'/>",
    )
    assert ingest.is_kernel_benchmark_xml(root) is True
    # is_benchmark_xml also matches — which is exactly why order matters.
    assert ingest.is_benchmark_xml(root) is True


def test_op_report_is_not_mistaken_for_a_kernel_xml(ingest):
    root = _root(
        ingest,
        f"<testcase classname='{OP_CLASSNAME}' name='perf_matmul_wall_clock_ms_1x512' time='2.0'/>",
    )
    assert ingest.is_kernel_benchmark_xml(root) is False
    assert ingest.is_benchmark_xml(root) is True


# --- degradation when the migration has not been applied --------------------


def test_run_type_is_dropped_when_the_column_is_absent(ingest):
    client = FakeClient({"benchmark_runs": ["run_id", "source_file", "created_at"]})
    ingest.insert_benchmark_run(
        client, 1, {"source_file": "f.xml", "created_at": datetime.now(UTC)}
    )
    _, rows, columns = client.inserts[0]
    assert "run_type" not in columns
    assert len(rows[0]) == len(columns)


def test_run_type_is_stored_when_the_column_exists(ingest):
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
            "source_file": "f.xml",
            "created_at": datetime.now(UTC),
            "run_type": "kernel",
        },
    )
    _, rows, columns = client.inserts[0]
    assert "run_type" in columns
    assert rows[0][columns.index("run_type")] == "kernel"


def test_table_exists_reports_a_missing_perf_kernels(ingest):
    client = FakeClient({"benchmark_runs": []})
    assert ingest._table_exists(client, "perf_kernels") is False


def test_table_exists_reports_a_present_perf_kernels(ingest):
    client = FakeClient({"perf_kernels": ["run_id"]})
    assert ingest._table_exists(client, "perf_kernels") is True


def test_kernel_insert_is_a_noop_when_there_are_no_kernels(ingest):
    client = FakeClient({"perf_kernels": ["run_id"]})
    ingest.insert_perf_kernels(client, 1, [])
    assert client.inserts == []


# --- the dispatch itself, driven through main() -----------------------------


KERNEL_XML = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    "<testsuites><testsuite name='pytest' tests='1'>"
    f"<testcase classname='{KERNEL_CLASSNAME}' name='kernel_matmul_MatmulOp_1x512x4096' time='1.500'>"
    # tags travel as <property name="tag" value="key__value"/>
    "<properties>"
    "<property name='tag' value='op__matmul'/>"
    "<property name='tag' value='kernel__MatmulOp'/>"
    "<property name='tag' value='input_shape__1x512x4096'/>"
    "<property name='tag' value='metric__kernel'/>"
    "<property name='tag' value='torch_spyre_ms__1.500'/>"
    "<property name='tag' value='sendnn_ms__2.000'/>"
    "<property name='tag' value='ratio__0.75'/>"
    "<property name='tag' value='pt_util__40.0'/>"
    "<property name='tag' value='num_runs__5'/>"
    "</properties></testcase>"
    "</testsuite></testsuites>\n"
)


def _run_main(ingest, monkeypatch, tmp_path, client):
    xml = tmp_path / "spyre_kernel_report.xml"
    xml.write_text(KERNEL_XML, encoding="utf-8")
    monkeypatch.setenv("CLICKHOUSE_HOST", "stub")
    monkeypatch.setattr(ingest, "get_client", lambda: client)
    monkeypatch.setattr(
        sys, "argv", ["ingest_xml.py", "--xml-file", str(xml), "--workflow", "test"]
    )
    ingest.main()
    return client


FULL_SCHEMA = {
    "benchmark_runs": [
        "run_id",
        "source_file",
        "version_info",
        "created_at",
        "workflow",
        "platform",
        "run_type",
    ],
    "perf_kernels": ["run_id", "kernel_name"],
}


def test_kernel_xml_is_skipped_before_anything_is_written(
    ingest, monkeypatch, tmp_path
):
    """No perf_kernels table: nothing may be written, so a retry can pick it up.

    Writing the benchmark_runs row first would record source_file, and the dedup
    check keys on it — every later run would skip the file permanently.
    """
    client = FakeClient({"benchmark_runs": FULL_SCHEMA["benchmark_runs"]})
    _run_main(ingest, monkeypatch, tmp_path, client)
    assert client.inserts == []


def test_kernel_xml_is_skipped_when_only_run_type_is_missing(
    ingest, monkeypatch, tmp_path
):
    """Partial migration: perf_kernels landed, run_type did not.

    Ingesting here would write a run row that is neither marked as a kernel run
    nor backed by perf_benchmarks rows — the artifact this branch removes.
    """
    client = FakeClient(
        {
            "benchmark_runs": [
                c for c in FULL_SCHEMA["benchmark_runs"] if c != "run_type"
            ],
            "perf_kernels": FULL_SCHEMA["perf_kernels"],
        }
    )
    _run_main(ingest, monkeypatch, tmp_path, client)
    assert client.inserts == []


def test_kernel_xml_never_reaches_the_op_branch(ingest, monkeypatch, tmp_path):
    """With the migration applied the file ingests as kernels, not as a benchmark."""
    client = FakeClient(dict(FULL_SCHEMA))
    _run_main(ingest, monkeypatch, tmp_path, client)
    tables = [t for t, _, _ in client.inserts]
    assert "perf_kernels" in tables
    assert "perf_benchmarks" not in tables


def test_main_issues_no_kernel_ddl(ingest, monkeypatch, tmp_path):
    client = FakeClient(dict(FULL_SCHEMA))
    _run_main(ingest, monkeypatch, tmp_path, client)
    joined = " ".join(client.commands).lower()
    assert "perf_kernels" not in joined
    assert "run_type" not in joined
