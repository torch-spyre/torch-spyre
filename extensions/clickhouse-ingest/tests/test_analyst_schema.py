"""analyst_* tables: the models match the DDL, bad rows are refused, and the views read the newest effective row."""

from pathlib import Path

import pytest
import regex as re

from spyre_clickhouse_ingest.schema import (
    ANALYST_ASSIGNMENTS,
    ANALYST_MEMORY,
    ANALYST_SESSIONS,
    ANALYST_TURNS,
    CASE_FILES,
    CASE_FILE_STATUS_VALUES,
    ASSIGNMENT_MODE_VALUES,
    MEMORY_KIND_VALUES,
    MEMORY_STATUS_VALUES,
    SESSION_STATE_VALUES,
    TURN_OUTCOME_VALUES,
    SchemaError,
    TABLES,
)

SCHEMA = Path(__file__).resolve().parents[1] / "schema"
DDL = (SCHEMA / "43-analyst.sql").read_text()
VIEWS = (SCHEMA / "58-analyst-views.sql").read_text()
MODELS = [
    ANALYST_ASSIGNMENTS,
    ANALYST_SESSIONS,
    ANALYST_TURNS,
    ANALYST_MEMORY,
    CASE_FILES,
]


def _ddl_columns(table: str) -> list[str]:
    body = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\s*\nENGINE", DDL, re.S
    ).group(1)
    cols = []
    for line in body.splitlines():
        line = line.split("--")[0].strip()
        if not line or line.startswith("CONSTRAINT"):
            continue
        cols.append(line.split()[0])
    return cols


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_model_columns_are_the_ddl_columns(model):
    server = {"audit_uuid", "audit_timestamp"}
    assert [c for c in _ddl_columns(model.name) if c not in server] == list(
        model.columns
    )
    assert model.name in TABLES


def _row(model, **over):
    row = {c: "" for c in model.columns}
    row.update(props={}, **over)
    return row


GOOD = {
    ANALYST_ASSIGNMENTS: dict(
        role="triage",
        model_alias="triage-small",
        mode="shadow",
        weight=100,
        effective_from="2026-10-23 10:00:00.000",
        set_by="ashok",
    ),
    ANALYST_SESSIONS: dict(
        session_id="s1",
        updated_at="2026-10-23 10:00:00.000",
        incident_id="inc-1",
        role="investigate",
        state="open",
    ),
    ANALYST_TURNS: dict(
        session_id="s1",
        step=1,
        ts="2026-10-23 10:00:01.000",
        role="triage",
        model_alias="triage-small",
        confidence=0.5,
        tokens_in=100,
        tokens_out=20,
        cost_usd=0.0,
        outcome="ok",
    ),
    ANALYST_MEMORY: dict(
        memory_id="m1",
        updated_at="2026-10-23 10:00:00.000",
        subject="artifactory-401",
        proposed_by="curate",
        kind="lesson",
        status="proposed",
        evidence_ids=[],
        confidence=0.7,
        expires_at=None,
    ),
    CASE_FILES: dict(
        incident_id="inc-1",
        version=1,
        updated_at="2026-10-23 10:00:00.000",
        status="open",
        through_step=0,
        keys={},
        doc="{}",
    ),
}


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_a_good_row_is_ordered_by_columns(model):
    out = model.row(_row(model, **GOOD[model]))
    assert len(out) == len(model.columns)


@pytest.mark.parametrize(
    "model,over",
    [
        (ANALYST_ASSIGNMENTS, {"mode": "auto"}),
        (ANALYST_ASSIGNMENTS, {"role": ""}),
        (ANALYST_ASSIGNMENTS, {"set_by": ""}),
        (ANALYST_SESSIONS, {"state": "paused"}),
        (ANALYST_SESSIONS, {"incident_id": ""}),
        (ANALYST_TURNS, {"outcome": "maybe"}),
        (ANALYST_TURNS, {"model_alias": ""}),
        (ANALYST_MEMORY, {"kind": "rumour"}),
        (ANALYST_MEMORY, {"status": "trusted"}),
        (ANALYST_MEMORY, {"proposed_by": ""}),
        (CASE_FILES, {"status": "merged"}),
        (CASE_FILES, {"version": 0}),
        (CASE_FILES, {"doc": ""}),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_rows_outside_the_ddl_checks_are_refused(model, over):
    with pytest.raises(SchemaError):
        model.row(_row(model, **{**GOOD[model], **over}))


def _block(table: str) -> str:
    block = DDL[DDL.index(f"EXISTS {table}") :]
    return block[: block.index("ENGINE")]


def test_ddl_declares_the_same_vocabularies_as_the_models():
    for table, col, allowed in [
        ("analyst_assignments", "mode", ASSIGNMENT_MODE_VALUES),
        ("analyst_sessions", "state", SESSION_STATE_VALUES),
        ("analyst_turns", "outcome", TURN_OUTCOME_VALUES),
        ("analyst_memory", "kind", MEMORY_KIND_VALUES),
        ("analyst_memory", "status", MEMORY_STATUS_VALUES),
        ("case_files", "status", CASE_FILE_STATUS_VALUES),
    ]:
        listed = re.search(rf"CHECK {col} IN\s*\(([^)]*)\)", _block(table)).group(1)
        assert set(re.findall(r"'([^']*)'", listed)) == set(allowed), (table, col)


def test_weight_and_confidence_are_bounded_in_the_ddl():
    assert "CHECK weight <= 100" in _block("analyst_assignments")
    assert "confidence >= 0 AND confidence <= 1" in _block("analyst_turns")
    assert "confidence >= 0 AND confidence <= 1" in _block("analyst_memory")


def test_turns_expire_but_the_state_tables_do_not():
    assert "TTL toDateTime(ts) + INTERVAL 180 DAY" in DDL
    assert DDL.count("TTL") == 1


def test_a_staged_assignment_is_not_selected_before_it_takes_effect():
    view = VIEWS[VIEWS.index("v_analyst_assignments_current") :]
    assert "WHERE effective_from <= now64(3)" in view


def test_view_aggregates_are_not_aliased_to_their_source_columns():
    # An alias equal to a source column is substituted into the other aggregates (ILLEGAL_AGGREGATION).
    for src in ("model_alias", "mode", "weight", "set_by", "second_approver", "reason"):
        assert f"AS {src}," not in VIEWS.split("v_analyst_memory_current")[0]
    assert "argMax(status, updated_at)       AS cur_status" in VIEWS


def test_expired_memory_is_left_out_whatever_its_stored_status():
    assert "cur_status != 'expired'" in VIEWS
    assert "cur_expires_at IS NULL OR cur_expires_at > now64(3)" in VIEWS
