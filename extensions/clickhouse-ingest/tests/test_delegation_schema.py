"""delegation_* tables: the models match the DDL, bad rows are refused, and the views read a grant's true state."""

from pathlib import Path

import pytest
import regex as re

from spyre_clickhouse_ingest.schema import (
    DELEGATION_GRANT_EVENTS,
    DELEGATION_GRANTS,
    GRANT_EVENT_VALUES,
    GRANT_RESULT_VALUES,
    SchemaError,
    TABLES,
)

SCHEMA = Path(__file__).resolve().parents[1] / "schema"
DDL = (SCHEMA / "44-delegation.sql").read_text()
VIEWS = (SCHEMA / "59-delegation-views.sql").read_text()
MODELS = [DELEGATION_GRANTS, DELEGATION_GRANT_EVENTS]
HASH = "a" * 64


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


def _grant(**over):
    row = {c: "" for c in DELEGATION_GRANTS.columns}
    row.update(
        grant_id="g1",
        grantor="ashok@example.com",
        grantee="triage-agent",
        tools=["acknowledge_incident"],
        scope_json="{}",
        expires_at="2026-10-23 14:00:00.000",
        max_uses=5,
        condition="",
        created_at="2026-10-23 10:00:00.000",
        created_by="ashok@example.com",
        consent_text_hash=HASH,
        props={},
    )
    row.update(over)
    return row


def _event(**over):
    row = {c: "" for c in DELEGATION_GRANT_EVENTS.columns}
    row.update(
        grant_id="g1",
        ts="2026-10-23 10:05:00.000",
        event="used",
        actor="triage-agent",
        tool="acknowledge_incident",
        args_hash="b" * 16,
        result="ok",
        on_behalf_of="ashok@example.com",
        props={},
    )
    row.update(over)
    return row


def test_good_rows_are_ordered_by_columns():
    assert len(DELEGATION_GRANTS.row(_grant())) == len(DELEGATION_GRANTS.columns)
    assert len(DELEGATION_GRANT_EVENTS.row(_event())) == len(
        DELEGATION_GRANT_EVENTS.columns
    )


@pytest.mark.parametrize(
    "model,row",
    [
        (DELEGATION_GRANTS, _grant(grantee="ashok@example.com")),
        (DELEGATION_GRANTS, _grant(tools=[])),
        (DELEGATION_GRANTS, _grant(consent_text_hash="short")),
        (DELEGATION_GRANTS, _grant(consent_text_hash="")),
        (DELEGATION_GRANTS, _grant(created_by="")),
        (DELEGATION_GRANT_EVENTS, _event(event="approved")),
        (DELEGATION_GRANT_EVENTS, _event(result="maybe")),
        (DELEGATION_GRANT_EVENTS, _event(actor="")),
    ],
    ids=[
        "self-grant",
        "no-tools",
        "short-hash",
        "no-hash",
        "no-author",
        "event",
        "result",
        "actor",
    ],
)
def test_rows_outside_the_ddl_checks_are_refused(model, row):
    with pytest.raises(SchemaError):
        model.row(row)


def _block(table: str) -> str:
    block = DDL[DDL.index(f"EXISTS {table}") :]
    return block[: block.index("ENGINE")]


def test_ddl_declares_the_same_vocabularies_as_the_models():
    for col, allowed in [
        ("event", GRANT_EVENT_VALUES),
        ("result", GRANT_RESULT_VALUES),
    ]:
        listed = re.search(
            rf"CHECK {col} IN\s*\(([^)]*)\)", _block("delegation_grant_events")
        ).group(1)
        assert set(re.findall(r"'([^']*)'", listed)) == set(allowed), col


def test_a_grant_cannot_outlive_its_creation_or_be_to_its_own_grantor_in_the_ddl():
    block = _block("delegation_grants")
    assert "CHECK expires_at > created_at" in block
    assert "CHECK grantor != grantee" in block
    assert "length(consent_text_hash) = 64" in block


def test_the_audit_table_expires_but_the_grants_do_not():
    assert DDL.count("TTL") == 1
    assert "TTL toDateTime(ts) + INTERVAL 365 DAY" in DDL


def test_state_is_ordered_revoked_then_expired_then_exhausted_then_active():
    block = VIEWS[VIEWS.index("multiIf(") : VIEWS.index(") AS state")]
    assert re.findall(r"'(\w+)',?\s*$", block, re.M) == [
        "revoked",
        "expired",
        "exhausted",
        "active",
    ]


def test_uses_count_only_successful_uses_and_unset_milestones_are_null():
    assert "countIf(event = 'used' AND result = 'ok')" in VIEWS
    assert VIEWS.count("toDateTime64(0, 3, 'UTC'), NULL") == 2


def test_view_aggregates_are_not_aliased_to_their_source_columns():
    for src in ("ts", "event", "result", "actor", "tool"):
        assert f"AS {src}," not in VIEWS.split("v_delegation_usage")[0]


def test_the_current_view_exposes_the_props_recorded_at_consent():
    # The gateway reads the grantor's rights and groups at consent from here, in the same query as the state.
    assert "g.props AS props" in VIEWS.split("v_delegation_usage")[0]
