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

"""The analyst-layer tables: models match the DDL, vocabularies match the CHECKs, views avoid the known pitfalls."""

import regex as re
from pathlib import Path

import pytest

from spyre_clickhouse_ingest import schema
from spyre_clickhouse_ingest.schema import TABLES, SchemaError

SCHEMA = Path(__file__).resolve().parents[1] / "schema"
DDL = (SCHEMA / "34-analyst-layers.sql").read_text()
VIEWS = (SCHEMA / "68-analyst-layers-views.sql").read_text()
NAMES = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", DDL)
SERVER = {"audit_uuid", "audit_timestamp"}
# ReplacingMergeTree(updated_at) collapses to the newest row per ORDER BY key.
REPLACING = {n for n in NAMES if n.startswith("assistant_") and n != "assistant_audit"}
VIEWED = {
    "v_analyst_role_memory_current": "analyst_role_memory",
    "v_analyst_project_files_current": "analyst_project_files",
    "v_analyst_profile_changes_current": "analyst_profile_changes",
    "v_goal_scores_current": "goal_scores",
    "v_goal_proposals_current": "goal_proposals",
}


def _block(table: str) -> str:
    return re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\(.*?\n\)\s*\nENGINE[^;]*;", DDL, re.S
    ).group(0)


def _ddl_columns(table: str) -> list[str]:
    body = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\s*\nENGINE", DDL, re.S
    ).group(1)
    cols, in_constraint = [], False
    for line in body.splitlines():
        line = line.split("--")[0].strip()
        if not line:
            continue
        if line.startswith("CONSTRAINT") or in_constraint:
            in_constraint = not line.rstrip(",").endswith(")") and "CHECK" in line
            continue
        cols.append(line.split()[0])
    return cols


def test_every_ddl_table_has_a_model_and_the_reverse():
    assert len(NAMES) == 24
    assert set(NAMES) <= set(TABLES)
    assert {n for n in TABLES if n in NAMES} == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_model_columns_are_the_ddl_columns(name):
    assert [c for c in _ddl_columns(name) if c not in SERVER] == list(
        TABLES[name].columns
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_table_carries_the_audit_columns(name):
    block = _block(name)
    assert "audit_uuid      UUID DEFAULT generateUUIDv7()" in block
    assert "audit_timestamp DateTime64(3) DEFAULT now64(3)" in block


def _good(model):
    row = {c: "" for c in model.columns}
    row.update(props={})
    for c in model.required:
        row[c] = "x"
    for c, allowed in model.enums:
        row[c] = sorted(allowed - {""})[0]
    return row


@pytest.mark.parametrize("name", NAMES)
def test_a_good_row_is_ordered_by_columns(name):
    model = TABLES[name]
    assert len(model.row(_good(model))) == len(model.columns)


@pytest.mark.parametrize("name", [n for n in NAMES if TABLES[n].enums])
def test_a_value_outside_the_ddl_check_is_refused(name):
    model = TABLES[name]
    col, _ = model.enums[0]
    with pytest.raises(SchemaError, match=col):
        model.row({**_good(model), col: "nonsense"})


@pytest.mark.parametrize("name", NAMES)
def test_an_empty_required_column_is_refused(name):
    model = TABLES[name]
    with pytest.raises(SchemaError, match="non-empty"):
        model.row({**_good(model), model.required[0]: ""})


def test_ddl_vocabularies_equal_the_models():
    seen = 0
    for name in NAMES:
        for col, allowed in TABLES[name].enums:
            listed = re.search(rf"CHECK {col} IN\s*\(([^)]*)\)", _block(name)).group(1)
            assert set(re.findall(r"'([^']*)'", listed)) == set(allowed), (name, col)
            seen += 1
    assert seen >= 20


def test_memory_status_accepts_quarantined_and_retracted_in_both_memory_tables():
    for status in ("quarantined", "retracted"):
        assert status in schema.MEMORY_STATUS_VALUES
        assert status in schema.ANALYST_ROLE_MEMORY_STATUS_VALUES
    ddl43 = (SCHEMA / "43-analyst.sql").read_text()
    assert "'quarantined', 'retracted')" in ddl43
    migration = (
        SCHEMA / "migrations" / "020_analyst_memory_quarantine.sql"
    ).read_text()
    assert "DROP CONSTRAINT IF EXISTS chk_memory_status" in migration
    assert "'quarantined', 'retracted'" in migration


def test_assistant_state_tables_are_replacing_and_the_rest_are_plain_merge_trees():
    for name in NAMES:
        engine = re.search(r"ENGINE = (.*)", _block(name)).group(1)
        want = "ReplacingMergeTree(updated_at)" if name in REPLACING else "MergeTree()"
        assert engine == want, name


def test_only_log_like_tables_expire():
    expiring = {n for n in NAMES if "\nTTL " in _block(n)}
    assert expiring >= {
        "analyst_orch_steps",
        "analyst_agent_calls",
        "access_decisions",
        "recovery_actions",
    }
    for state in (
        "analyst_role_memory",
        "analyst_outcomes",
        "goal_proposals",
        "assistant_profiles",
        "analyst_work_items",
    ):
        assert state not in expiring or state == "analyst_work_items"


@pytest.mark.parametrize("view,src", VIEWED.items())
def test_views_are_the_newest_row_per_key_over_their_table(view, src):
    sql = VIEWS[VIEWS.index(f"CREATE VIEW IF NOT EXISTS {view} AS") :].split(";")[0]
    assert f"FROM {src}\n" in sql
    assert "GROUP BY" in sql
    assert " FINAL" not in sql


def test_view_aggregates_are_not_aliased_to_their_source_columns():
    # An alias equal to a source column is substituted into the other aggregates (ILLEGAL_AGGREGATION).
    for m in re.finditer(r"argMax\((\w+), \w+\) AS (\w+)", VIEWS):
        assert m.group(2) == "cur_" + m.group(1)


def test_a_dead_memory_stays_visible_to_the_reader_that_filters_it():
    view = VIEWS[VIEWS.index("v_analyst_role_memory_current") :].split(";")[0]
    assert "WHERE" not in view


def test_views_keep_props_because_readers_use_the_family_and_role_keys():
    for view in VIEWED:
        sql = VIEWS[VIEWS.index(f"CREATE VIEW IF NOT EXISTS {view} AS") :].split(";")[0]
        assert "argMax(props, " in sql


def test_a_private_role_memory_is_active_until_it_expires():
    assert "active" in schema.ANALYST_ROLE_MEMORY_STATUS_VALUES
    assert "active" not in schema.MEMORY_STATUS_VALUES
