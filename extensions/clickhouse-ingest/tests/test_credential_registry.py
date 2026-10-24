"""credential_registry.yaml: rows match the table models, unknowns stay unknown, no secret shapes."""

import json
import regex as re
from datetime import datetime, timezone

import pytest

from spyre_clickhouse_ingest import credential_registry as reg
from spyre_clickhouse_ingest.schema import CREDENTIAL_INVENTORY, CREDENTIAL_OWNERS

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
REQUIRED_IDS = {
    "CURVEBENDER_SPYRE_CI_API_KEY",
    "CURVEBENDER_SPYRE_CI_MANAGE_API_KEY",
    "prometheus-alerts-reader-token",
    "analyst-jenkins-read-token",
    "analyst-github-app-installation-token",
    "slack-spyre-ci-bot-token-test-workspace",
}


def test_required_credentials_are_registered():
    assert REQUIRED_IDS <= {e["id"] for e in reg.load()}


def test_rows_validate_against_the_table_models():
    inv, own = reg.rows(NOW)
    assert len(inv) == len(own) == len(reg.load())
    for r in inv:
        CREDENTIAL_INVENTORY.row(r)
    for r in own:
        CREDENTIAL_OWNERS.row(r)


def test_unknown_rotation_and_rotation_date_are_never_filled():
    _, own = reg.rows(NOW)
    assert all(r["rotation_days"] == 0 and r["last_rotated_at"] is None for r in own)


def test_unused_management_key_has_no_consumer_and_no_probe():
    inv, _ = reg.rows(NOW)
    row = next(r for r in inv if r["credential_id"].endswith("MANAGE_API_KEY"))
    assert row["consumers"] == [] and row["state"] == "unbound"
    assert row["props"]["probe_status"] == "none"


def test_ai_key_is_in_use_with_a_consumer():
    inv, _ = reg.rows(NOW)
    row = next(r for r in inv if r["credential_id"] == "CURVEBENDER_SPYRE_CI_API_KEY")
    assert row["state"] == "bound" and row["consumers"]


def test_unassigned_owner_stays_empty():
    _, own = reg.rows(NOW)
    by_id = {r["credential_id"]: r for r in own}
    assert by_id["prometheus-alerts-reader-token"]["owner"] == ""
    assert by_id["analyst-jenkins-read-token"]["owner_email"] == ""


def test_keys_are_unique():
    entries = reg.load()
    keys = [(e["system"], e["scope"], e["id"]) for e in entries]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "field", ["system", "scope", "id", "kind", "status", "probe", "stored_in"]
)
def test_every_entry_has_the_field(field):
    assert all(field in e for e in reg.load())


def test_no_secret_shaped_text_in_the_file():
    text = (
        reg.resources.files("spyre_clickhouse_ingest") / "credential_registry.yaml"
    ).read_text()
    assert not re.search(
        r"sk-[A-Za-z0-9]{10,}|xox[bap]-|ghp_|eyJ[A-Za-z0-9_-]{10,}|[A-Za-z0-9+/]{40,}",
        text,
    )


def test_bad_status_is_refused(monkeypatch):
    monkeypatch.setattr(
        reg.yaml,
        "safe_load",
        lambda _t: {
            "entries": [
                {
                    "id": "x",
                    "system": "s",
                    "scope": "c",
                    "status": "weird",
                    "probe": {"status": "none"},
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="status"):
        reg.load()


def test_cli_writes_jsonl(tmp_path):
    a, b = tmp_path / "i.jsonl", tmp_path / "o.jsonl"
    assert reg.main(["--inventory-out", str(a), "--owners-out", str(b)]) == 0
    assert all(json.loads(x)["credential_id"] for x in a.read_text().splitlines())
    assert len(b.read_text().splitlines()) == len(reg.load())
