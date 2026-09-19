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

"""Pins the hardware-diagnostics parse/ingest behaviour that review found wrong.

Lives in the root suite, not under extensions/, because nothing runs the library's own tests in
CI yet -- and these are the cases where a regression silently mis-attributes a hardware fault.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / ".github" / "scripts"

# The scripts import the shared library from extensions/; it is in this repo, so put it on
# sys.path rather than requiring an install for a parse-only test.
_CHLIB = _ROOT / "extensions" / "clickhouse-ingest"
if str(_CHLIB) not in sys.path:
    sys.path.insert(0, str(_CHLIB))

# Stubbed at MODULE scope, before the first library import below: the package __init__ imports
# client.py, which imports clickhouse_connect, and the test venv has no ClickHouse driver -- these
# tests parse logs and build rows, they never open a connection. Stubbing inside a fixture is too
# late for a test that imports a library submodule directly.
sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))

RAS_LINE = (
    'ERRR 15.09.2026 10:25:09.123456 [ras_base.hpp: 74] {"Device":"/dev/vfio/1",'
    '"action":"information","category":"configuration","code":"0xf40a",'
    '"message":"timeout","name":"RAS::CBRB::ResponseTimeout","severity":"ERROR"}'
)


def _load(name: str):
    """Load a .github/scripts entry point by path -- a script, not an importable package."""
    sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parse_script():
    return _load("parse_hw_failures")


@pytest.fixture(scope="module")
def ingest_script():
    return _load("ingest_hw_diagnostics")


def _one(chunk: str) -> dict:
    from spyre_clickhouse_ingest.hw_parse import parse_log

    records = parse_log(chunk, run_id="1", suite_hint="S")
    assert len(records) == 1, records
    return records[0]


# ── failure attribution ───────────────────────────────────────────────────────────────────


def test_recovered_ras_on_a_passing_attempt_is_not_a_failure():
    rec = _one(f"=== Attempt 1/3: S ===\n{RAS_LINE}\n=== Attempt 1 PASSED\n")
    assert rec["outcome"] == "passed"
    assert rec["failure_reason"] == "none"
    assert rec["failure_phase"] == ""


def test_recovered_ras_on_a_passing_attempt_still_records_the_event():
    # The point of gating only the classification: a recovered fault is real hardware data, and
    # a "fix" that moved the whole ras_* block under the guard would pass the test above.
    rec = _one(f"=== Attempt 1/3: S ===\n{RAS_LINE}\n=== Attempt 1 PASSED\n")
    assert rec["ras_name"] == "RAS::CBRB::ResponseTimeout"
    assert rec["ras_code"] == "0xf40a"
    assert len(json.loads(rec["ras_events_json"])) == 1


def test_ras_on_a_failing_attempt_still_classifies():
    rec = _one(f"=== Attempt 1/2: S ===\n{RAS_LINE}\n=== Attempt 1 FAILED (exit=1)\n")
    assert rec["outcome"] == "failed"
    assert rec["failure_reason"] == "hardware_ras_timeout"


def test_firmware_line_on_a_passing_attempt_gets_no_failure_phase():
    rec = _one(
        f"=== Attempt 1/1: S ===\n{RAS_LINE}\n"
        "initialize_firmware.cpp: setting up\n=== Attempt 1 PASSED\n"
    )
    assert rec["failure_reason"] == "none"
    # A phase without a reason is an internally inconsistent row.
    assert rec["failure_phase"] == ""


# ── outcome fallback (no attempt banner) ──────────────────────────────────────────────────


def test_device_chatter_mentioning_errors_does_not_fail_a_passing_suite():
    rec = _one(
        "INFO 15.09.2026 10:00:00.1 [dt] retry queue: 3 errors drained\n"
        "==================== 12 passed in 8.11s ====================\n"
    )
    assert rec["outcome"] == "passed"


def test_gha_process_exit_still_marks_failed():
    # The fallback's legitimate arm: a crash that never printed a pytest summary.
    rec = _one("some output\nError: Process completed with exit code 1.\n")
    assert rec["outcome"] == "failed"


# ── pytest counts ─────────────────────────────────────────────────────────────────────────


def test_counts_come_from_the_summary_not_a_per_file_subtotal():
    rec = _one(
        "=== Attempt 1/1: S ===\ntests/test_foo.py ....  4 passed\n"
        "============ 4 failed, 96 passed in 120.5s ============\n"
    )
    assert (rec["tests_passed"], rec["tests_failed"]) == (96, 4)


def test_a_captured_echo_after_the_summary_does_not_win():
    rec = _one(
        "=== Attempt 1/1: S ===\n"
        "============ 4 failed, 96 passed in 120.5s ============\n"
        "captured stdout: 1 passed in 0.1s\n"
    )
    assert (rec["tests_passed"], rec["tests_failed"]) == (96, 4)


def test_a_summary_without_the_equals_rule_is_still_read():
    # Guards the regression the narrower "=====" -only anchor would have introduced: real counts
    # silently becoming 0.
    rec = _one("=== Attempt 1/1: S ===\n7 passed, 1 failed in 3.2s\n")
    assert (rec["tests_passed"], rec["tests_failed"]) == (7, 1)


def test_no_summary_yields_zero_counts():
    rec = _one("=== Attempt 1/1: S ===\nnothing useful here\n")
    assert (rec["tests_passed"], rec["tests_failed"], rec["tests_error"]) == (0, 0, 0)


# ── ingest ────────────────────────────────────────────────────────────────────────────────


def test_meta_only_records_are_filtered_out():
    from spyre_clickhouse_ingest.hw_diagnostics import filter_suite_records

    assert (
        filter_suite_records([{"suite_name": ".DS_Store"}, {"suite_name": "  "}]) == []
    )


def test_all_records_filtered_exits_zero_without_connecting(
    ingest_script, tmp_path, monkeypatch, capsys
):
    # The content of the fix is that it never opens a session: the emptiness re-check has to run
    # before the connect, not just before records[0].
    json_file = tmp_path / "hw.json"
    json_file.write_text(json.dumps([{"suite_name": ".DS_Store"}]))

    def boom(*a, **k):
        raise AssertionError("connected to ClickHouse despite having no records")

    monkeypatch.setattr(ingest_script, "get_client", boom)
    monkeypatch.setattr(
        sys, "argv", ["ingest_hw_diagnostics.py", "--json-file", str(json_file)]
    )
    with pytest.raises(SystemExit) as exc:
        ingest_script.main()
    assert exc.value.code == 0
    assert "nothing to ingest" in capsys.readouterr().out


def test_insert_rows_rejects_an_arity_mismatch():
    from spyre_clickhouse_ingest.hw_diagnostics import insert_rows

    with pytest.raises(ValueError, match="disagree"):
        insert_rows(object(), [[1, 2, 3]])


def test_build_row_matches_the_column_count():
    from spyre_clickhouse_ingest.hw_diagnostics import RunContext, build_row
    from spyre_clickhouse_ingest.hw_schema import HW_COLUMN_NAMES

    row = build_row({"suite_name": "S"}, RunContext(run_id="r"))
    assert len(row) == len(HW_COLUMN_NAMES)


def test_dedup_and_migration_honour_the_table_argument():
    from spyre_clickhouse_ingest.hw_schema import already_ingested, ensure_extra_columns

    class FakeClient:
        def __init__(self):
            self.queries = []
            self.commands = []

        def query(self, sql, parameters=None):
            self.queries.append(sql)
            return types.SimpleNamespace(result_rows=[[0]])

        def command(self, sql):
            self.commands.append(sql)

    client = FakeClient()
    already_ingested(client, "r", "w", table="scratch_hw")
    ensure_extra_columns(client, table="scratch_hw")
    assert "FROM scratch_hw" in client.queries[0]
    assert all("ALTER TABLE scratch_hw" in c for c in client.commands)


# ── one definition, not a copy ────────────────────────────────────────────────────────────


def test_scripts_use_the_shared_library_not_a_local_copy(parse_script, ingest_script):
    # Object identity, not equality: a local copy that merely agrees today passes every
    # value-based test while drifting silently. Editing the library must change what runs.
    from spyre_clickhouse_ingest import client, hw_diagnostics, hw_parse, hw_schema

    for name in ("parse_log", "_pick_files_from_dir"):
        assert getattr(parse_script, name) is getattr(hw_parse, name), name
    for name in ("build_row", "filter_suite_records", "insert_rows", "load_records"):
        assert getattr(ingest_script, name) is getattr(hw_diagnostics, name), name
    for name in ("already_ingested", "ensure_extra_columns"):
        assert getattr(ingest_script, name) is getattr(hw_schema, name), name
    # Pins the deleted duplicate: this script had its own copy of get_client.
    assert ingest_script.get_client is client.get_client
