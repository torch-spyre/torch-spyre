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

"""copy_reused_cases: a delta run reports its whole tier, marked with who ran what.

A delta run executes only the set difference of a tier (measured on torch-spyre: a
regression run reusing integration runs 75 of 136 configs, 19 if it also reuses unit).
The rest are copied in from the covering run so `GROUP BY run_id` is correct without a
union view. `props['ran_in']` names the run that ACTUALLY executed each case.

These tests pin the three properties that were verified against the live dev instance
and that a refactor would quietly break:
  * ran_in is PRESERVED on a copy-of-a-copy, which is what makes reuse recursive with no
    chain to walk;
  * a re-copy from the same source adds nothing (the XML path's own dedup keys on
    props['source_file'], which copies inherit, so it cannot see a re-copy -- without the
    guard here a second call doubled 4 rows to 8);
  * a DIFFERENT source run can still contribute.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / ".github" / "scripts"


@pytest.fixture(scope="module")
def ing():
    """Load ingest_xml.py by path -- it is a script, not an importable package."""
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "ingest_xml", _SCRIPTS / "ingest_xml.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:  # lxml / clickhouse_connect / regex absent
        pytest.skip(f"ingest_xml deps unavailable: {exc}")
    return mod


class FakeRows:
    def __init__(self, rows):
        self.result_rows = rows


class FakeCH:
    """Records commands and answers the guard query from a per-(run, src) row count."""

    def __init__(self, present=None):
        # {(run_id, src_run): row_count} -- what the table already holds.
        self.present = dict(present or {})
        self.commands = []

    def query(self, sql, parameters=None):
        p = parameters or {}
        if "props['ran_in'] = {src:String}" in sql:
            n = self.present.get((p.get("run_id"), p.get("src")), 0)
            return FakeRows([[n]])
        raise AssertionError(f"unexpected query: {sql}")

    def command(self, sql, parameters=None):
        self.commands.append((sql, parameters or {}))
        # A real INSERT makes those rows present, so a second call is a no-op.
        p = parameters or {}
        self.present[(p.get("run_id"), p.get("src"))] = 1


# ── the copy itself ─────────────────────────────────────────────────────────────


def test_no_covered_tiers_is_a_no_op(ing):
    c = FakeCH()
    assert ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", []) == 0
    assert c.commands == []


def test_blank_source_run_is_skipped(ing):
    """A tier with no resolvable covering run must not produce an INSERT with an empty id."""
    c = FakeCH()
    assert (
        ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("integration", "")])
        == 0
    )
    assert c.commands == []


def test_copies_one_tier(ing):
    c = FakeCH()
    n = ing.copy_reused_cases(
        c, "db", "run-1", "torch-spyre", [("integration", "src-1")]
    )
    assert n == 1
    sql, params = c.commands[0]
    assert "INSERT INTO db.test_case_runs" in sql
    assert params["run_id"] == "run-1"
    assert params["src"] == "src-1"
    assert params["tier"] == "integration"


def test_copy_is_scoped_to_the_tiers_own_cases(ing):
    """The covering run may have executed a WIDER set; importing all of it would credit
    this tier with cases that do not belong to it."""
    c = FakeCH()
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("integration", "src-1")])
    sql, _ = c.commands[0]
    assert "has(c.tags, concat('testtype__', {tier:String}))" in sql


def test_ran_in_is_preserved_not_overwritten(ing):
    """The recursion property: a copy of a copy still names the ORIGINAL executor, so
    there is no chain to walk and no cycle to guard against."""
    c = FakeCH()
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("integration", "src-1")])
    sql, _ = c.commands[0]
    # ran_in comes from the source row when present, and only falls back to the source
    # run_id for a legacy row written before ran_in existed.
    assert "mapContains(cr.props,'ran_in')" in sql
    assert "cr.props['ran_in']" in sql
    assert "{run_id:UUID}" in sql.split("mapUpdate")[0], (
        "run_id is the TARGET, not ran_in"
    )


# ── idempotency ────────────────────────────────────────────────────────────────


def test_recopy_from_the_same_source_adds_nothing(ing):
    """Without this guard a second call doubled 4 rows to 8 on the live instance."""
    c = FakeCH()
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("integration", "src-1")])
    assert len(c.commands) == 1
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("integration", "src-1")])
    assert len(c.commands) == 1, "the second copy must be refused"


def test_a_second_tier_from_the_same_source_adds_nothing(ing):
    """Tier tags OVERLAP -- a case commonly carries both integration and regression -- so
    the source's rows are already present and a second tier copy is correctly a no-op.
    An earlier tier-keyed guard got this wrong in the other direction, refusing nothing
    and doubling instead."""
    c = FakeCH()
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("integration", "src-1")])
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("regression", "src-1")])
    assert len(c.commands) == 1


def test_a_different_source_run_still_contributes(ing):
    """The guard must be keyed on the SOURCE, not on the target run, or a run covered by
    two different prior runs would only ever import the first."""
    c = FakeCH()
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("integration", "src-1")])
    ing.copy_reused_cases(c, "db", "run-1", "torch-spyre", [("unit", "src-2")])
    assert len(c.commands) == 2
    assert {p["src"] for _, p in c.commands} == {"src-1", "src-2"}


def test_several_tiers_in_one_call(ing):
    c = FakeCH()
    n = ing.copy_reused_cases(
        c,
        "db",
        "run-1",
        "torch-spyre",
        [("integration", "src-1"), ("unit", "src-2"), ("smoke", "src-3")],
    )
    assert n == 3
    assert len(c.commands) == 3


# ── the executed rows carry ran_in too ─────────────────────────────────────────


def test_executed_rows_are_stamped_with_this_run(ing):
    """`ran_in = run_id` is what makes "how much did we actually execute" answerable:
    countIf(props['ran_in'] = run_id). Without it every reuse copy inflates that count."""
    src = (_SCRIPTS / "ingest_xml.py").read_text()
    insert_v2 = src[src.index("def insert_v2(") :]
    insert_v2 = insert_v2[: insert_v2.index("\ndef ")]
    assert '"ran_in": run_id' in insert_v2
    assert "source_file" in insert_v2, (
        "the shard discriminator must survive alongside it"
    )
