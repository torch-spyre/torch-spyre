"""incident_events / incident_log: the models match the DDL, bad rows are refused, the view reads the newest state."""

import regex as re
from pathlib import Path

import pytest

from spyre_clickhouse_ingest.schema import (
    INCIDENT_CAUSE_VALUES,
    INCIDENT_EVENTS,
    INCIDENT_LOG,
    SchemaError,
    TABLES,
)

SCHEMA = Path(__file__).resolve().parents[1] / "schema"
DDL = (SCHEMA / "42-incident-ledger.sql").read_text()
VIEW = (SCHEMA / "57-incident-views.sql").read_text()


def _ddl_columns(table: str) -> list[str]:
    body = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\s*\nENGINE", DDL, re.S
    ).group(1)
    cols = []
    for line in body.splitlines():
        line = line.split("--")[0].strip()
        if not line or line.startswith(("CONSTRAINT", "('")):
            continue
        if not re.match(r"[a-z_]+\s", line):
            continue
        cols.append(line.split()[0])
    return cols


@pytest.mark.parametrize("model", [INCIDENT_EVENTS, INCIDENT_LOG], ids=lambda m: m.name)
def test_model_columns_are_the_ddl_columns(model):
    server = {"audit_uuid", "audit_timestamp"}
    assert [c for c in _ddl_columns(model.name) if c not in server] == list(
        model.columns
    )
    assert model.name in TABLES


def _event(**over):
    row = {c: "" for c in INCIDENT_EVENTS.columns}
    row.update(
        ts="2026-10-23 10:00:00.000",
        source="jenkins",
        subject="Spyre/orchestrator",
        severity="critical",
        cause_class="credentials",
        signature="artifactory-401",
        props={},
    )
    row.update(over)
    return row


def _log(**over):
    row = {c: "" for c in INCIDENT_LOG.columns}
    row.update(
        incident_id="inc-1",
        updated_at="2026-10-23 10:00:00.000",
        updated_by="router",
        actor_kind="system",
        state="open",
        severity="critical",
        cause_class="unknown",
        props={},
    )
    row.update(over)
    return row


def test_good_rows_are_ordered_by_columns():
    ev = INCIDENT_EVENTS.row(_event())
    assert ev[INCIDENT_EVENTS.columns.index("subject")] == "Spyre/orchestrator"
    lg = INCIDENT_LOG.row(_log())
    assert lg[INCIDENT_LOG.columns.index("state")] == "open"


@pytest.mark.parametrize(
    "model,row",
    [
        (INCIDENT_EVENTS, _event(severity="fatal")),
        (INCIDENT_EVENTS, _event(cause_class="gremlins")),
        (INCIDENT_EVENTS, _event(subject="")),
        (INCIDENT_LOG, _log(state="closed")),
        (INCIDENT_LOG, _log(actor_kind="robot")),
        (INCIDENT_LOG, _log(incident_id="")),
        (INCIDENT_LOG, _log(updated_by="")),
    ],
    ids=["severity", "cause", "subject", "state", "actor", "id", "author"],
)
def test_rows_outside_the_ddl_checks_are_refused(model, row):
    with pytest.raises(SchemaError):
        model.row(row)


def test_ddl_declares_the_same_vocabularies_as_the_model():
    for table, col, allowed in [
        ("incident_events", "severity", {"info", "warning", "critical"}),
        (
            "incident_log",
            "state",
            {"open", "acknowledged", "mitigated", "resolved", "dismissed"},
        ),
        ("incident_log", "actor_kind", {"system", "agent", "ask_spyre", "human"}),
        ("incident_events", "cause_class", INCIDENT_CAUSE_VALUES),
        ("incident_log", "cause_class", INCIDENT_CAUSE_VALUES),
    ]:
        # Scoped to this table's block: the same column name is constrained in both tables.
        block = DDL[DDL.index(f"EXISTS {table}") :]
        block = block[: block.index("ENGINE")]
        listed = re.search(rf"CHECK {col} IN\s*\(([^)]*)\)", block).group(1)
        assert set(re.findall(r"'([^']*)'", listed)) == set(allowed)


def test_the_view_takes_the_newest_row_and_marks_active_states():
    assert "argMax(state, updated_at)" in VIEW
    assert "state IN ('open', 'acknowledged')" in VIEW


def test_unset_milestones_are_null_not_the_epoch():
    # minIf over no rows is the epoch; reporting it would show an ack in 1970.
    assert VIEW.count("toDateTime64(0, 3, 'UTC'), NULL") >= 4
