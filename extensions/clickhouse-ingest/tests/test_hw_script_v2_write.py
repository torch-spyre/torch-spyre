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

"""ingest_hw_diagnostics.py writes v2 only: a failed write exits 1 only under --strict."""

import runpy
import sys
from pathlib import Path

import pytest

from spyre_clickhouse_ingest import client as ch
from spyre_clickhouse_ingest import hw_diagnostics as hw
from spyre_clickhouse_ingest import writes

SCRIPT = (
    Path(__file__).resolve().parents[3] / ".github/scripts/ingest_hw_diagnostics.py"
)
RUN = "00000000-0000-4000-8000-000000000001"


class Live:
    def __init__(self, table_exists=True):
        self.table_exists = table_exists

    def command(self, sql, *a, **k):
        return int(self.table_exists) if sql.startswith("EXISTS") else 1

    def query(self, sql, *a, **k):
        class R:
            result_rows = [(0,)]

        return R()


def run_script(monkeypatch, tmp_path, *flags, client=None, insert=None, table=None):
    monkeypatch.setattr(ch, "get_client", lambda **k: client or Live())
    monkeypatch.setattr(hw, "load_records", lambda p: [{"suite_name": "s"}])
    monkeypatch.setattr(hw, "filter_suite_records", lambda r: r)
    monkeypatch.setattr(hw, "build_row", lambda rec, ctx: ["row"])
    monkeypatch.setattr(hw, "insert_rows", insert or (lambda *a, **k: None))
    monkeypatch.setenv("CLICKHOUSE_DB_V2", "spyre_v2")
    (tmp_path / "x.json").write_text("[]")
    writes.reset()
    argv = [str(SCRIPT), "--json-file", str(tmp_path / "x.json"), "--run-id", RUN]
    argv += ["--table", table] if table else []
    monkeypatch.setattr(sys, "argv", argv + list(flags))
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    return e.value.code


def boom(*a, **k):
    raise RuntimeError("denied")


@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_a_failed_insert_exits_1_only_under_strict(
    monkeypatch, tmp_path, capsys, flags, code
):
    assert run_script(monkeypatch, tmp_path, *flags, insert=boom) == code
    cap = capsys.readouterr()
    assert "insert into spyre_v2.hw_failure_diagnostics failed" in cap.err
    assert ("::warning title=ClickHouse::insert into" in cap.out) is (code == 0)


@pytest.mark.parametrize("flags, code", [(["--strict"], 1), ([], 0)])
def test_a_missing_v2_table_is_a_failed_write(
    monkeypatch, tmp_path, capsys, flags, code
):
    rc = run_script(monkeypatch, tmp_path, *flags, client=Live(table_exists=False))
    assert rc == code and "does not exist" in capsys.readouterr().err


def test_a_successful_insert_exits_0(monkeypatch, tmp_path):
    assert run_script(monkeypatch, tmp_path, "--strict") in (0, None)


def test_a_table_that_is_not_an_identifier_is_refused(monkeypatch, tmp_path, capsys):
    rc = run_script(monkeypatch, tmp_path, table="t; DROP TABLE x")
    assert rc == 1 and "plain SQL identifier" in capsys.readouterr().err
