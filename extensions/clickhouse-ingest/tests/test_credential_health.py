"""credential_inventory / credential_probes: the models match the DDL, and bad rows are refused."""

import re
from pathlib import Path

import pytest

from spyre_clickhouse_ingest.schema import (
    CREDENTIAL_INVENTORY,
    CREDENTIAL_OWNERS,
    CREDENTIAL_PROBES,
    SchemaError,
    TABLES,
)

SCHEMA = Path(__file__).resolve().parents[1] / "schema"
DDL = (SCHEMA / "41-credential-health.sql").read_text()
VIEW = (SCHEMA / "56-credential-views.sql").read_text()


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


@pytest.mark.parametrize(
    "model",
    [CREDENTIAL_INVENTORY, CREDENTIAL_PROBES, CREDENTIAL_OWNERS],
    ids=lambda m: m.name,
)
def test_model_columns_are_the_ddl_columns(model):
    # `ts`, `audit_uuid` and `audit_timestamp` are server defaults, never inserted.
    server = {"ts", "audit_uuid", "audit_timestamp"}
    assert [c for c in _ddl_columns(model.name) if c not in server] == list(
        model.columns
    )
    assert model.name in TABLES


def _probe(**over):
    row = {
        "system": "jenkins",
        "scope": "system",
        "credential_id": "SpyreBld-Artifactory-PAT",
        "probe_site": "jenkins",
        "probe": "ping",
        "status": "ok",
        "error_class": "",
        "http_status": 200,
        "latency_ms": 12,
        "expires_at": None,
        "fingerprint": "a0563ad8",
        "detail": "",
        "job_run": "",
        "props": {},
    }
    row.update(over)
    return row


def test_probe_row_is_ordered_by_columns():
    row = CREDENTIAL_PROBES.row(_probe())
    assert row[CREDENTIAL_PROBES.columns.index("credential_id")] == (
        "SpyreBld-Artifactory-PAT"
    )


@pytest.mark.parametrize(
    "over",
    [{"status": "expired"}, {"probe_site": "laptop"}, {"credential_id": ""}],
    ids=["status", "site", "id"],
)
def test_probe_outside_the_ddl_checks_is_refused(over):
    with pytest.raises(SchemaError):
        CREDENTIAL_PROBES.row(_probe(**over))


def test_inventory_state_outside_the_ddl_check_is_refused():
    row = {c: "" for c in CREDENTIAL_INVENTORY.columns}
    row.update(
        system="jenkins",
        scope="system",
        credential_id="x",
        observed_at="2026-10-22",
        state="deleted",
        consumers=[],
        props={},
    )
    with pytest.raises(SchemaError):
        CREDENTIAL_INVENTORY.row(row)


def test_ddl_declares_the_same_vocabularies_as_the_model():
    for table, col, allowed in [
        ("credential_inventory", "state", {"bound", "unbound", "gone"}),
        ("credential_probes", "status", {"ok", "failing", "unknown"}),
        ("credential_probes", "probe_site", {"jenkins", "gha", "dashboard", "cluster"}),
    ]:
        # Scoped to this table's block: `status` and `state` CHECKs sit in different tables.
        block = DDL[DDL.index(f"EXISTS {table}") :]
        listed = re.search(rf"CHECK {col} IN \(([^)]*)\)", block).group(1)
        assert set(re.findall(r"'([^']*)'", listed)) == allowed


def test_failing_wins_over_expiry_in_the_health_class():
    # The view orders its classes failing > expired > expiring > ok; a 401 must read red.
    block = VIEW[VIEW.index("multiIf(") : VIEW.index(") AS health")]
    # Result labels end their line; the same words inside a condition do not.
    order = re.findall(r"'(\w+)',?\s*$", block, re.M)
    assert order == ["unprobed", "failing", "expired", "expiring", "ok", "unknown"]


def test_owner_edits_never_live_in_the_table_discovery_rewrites():
    # Discovery replaces credential_inventory rows every run, so a human-entered owner there
    # would be erased within the hour.
    assert "owner" not in CREDENTIAL_INVENTORY.columns
    assert (
        "owner" in CREDENTIAL_OWNERS.columns
        and "linked_email" in CREDENTIAL_OWNERS.columns
    )


def test_owner_row_needs_an_author_and_a_time():
    row = {c: "" for c in CREDENTIAL_OWNERS.columns}
    row.update(
        system="jenkins",
        scope="system",
        credential_id="x",
        rotation_days=0,
        last_rotated_at=None,
        props={},
        updated_at="2026-10-23 00:00:00.000",
    )
    with pytest.raises(SchemaError, match="updated_by"):
        CREDENTIAL_OWNERS.row(row)
    CREDENTIAL_OWNERS.row({**row, "updated_by": "someone@example.com"})


def test_view_takes_the_newest_owner_row_per_credential():
    assert "argMax(owner, updated_at)" in VIEW and "FROM credential_owners" in VIEW
