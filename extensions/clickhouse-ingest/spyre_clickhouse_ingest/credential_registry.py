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

"""credential_registry.yaml as credential_inventory and credential_owners rows.

Covers credentials Jenkins discovery cannot see or that are not bound yet. Names only; a value
never enters the file. An empty owner, `rotation_days` 0 and a NULL `last_rotated_at` mean unknown.
Loading a Jenkins-visible id after discovery replaces that observation (latest observed_at wins),
so load only ids discovery has not seen.
"""

import argparse
import json
from datetime import datetime, timezone
from importlib import resources

import yaml

from .schema import CREDENTIAL_INVENTORY, CREDENTIAL_OWNERS

UPDATED_BY = "credential_registry"
STATUS_STATE = {"in_use": "bound", "unused": "unbound", "planned": "unbound"}
PROBE_STATUSES = frozenset({"existing", "designed", "none"})


def load() -> list[dict]:
    text = resources.files(__package__).joinpath("credential_registry.yaml").read_text()
    entries = yaml.safe_load(text)["entries"]
    seen = set()
    for e in entries:
        key = (e["system"], e["scope"], e["id"])
        if key in seen:
            raise ValueError(f"duplicate credential {key}")
        seen.add(key)
        if e["status"] not in STATUS_STATE:
            raise ValueError(
                f"{e['id']}: status {e['status']!r} not in {sorted(STATUS_STATE)}"
            )
        if e["probe"]["status"] not in PROBE_STATUSES:
            raise ValueError(f"{e['id']}: probe status {e['probe']['status']!r}")
    return entries


def _ts(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def inventory_rows(entries: list[dict], now: datetime) -> list[dict]:
    return [
        {
            "system": e["system"],
            "scope": e["scope"],
            "credential_id": e["id"],
            "kind": e["kind"],
            "description": e["description"][:512],
            "consumers": list(e["consumers"]),
            "state": STATUS_STATE[e["status"]],
            "source_created_at": None,
            "source_updated_at": None,
            "last_bound_at": None,
            "observed_at": _ts(now),
            "job_run": "",
            "props": {
                "registered_by": UPDATED_BY,
                "status": e["status"],
                "probe_type": e["probe"]["type"],
                "probe_status": e["probe"]["status"],
                "probe_call": e["probe"]["call"],
                "stored_in": "; ".join(e["stored_in"]),
            },
        }
        for e in entries
    ]


def owner_rows(entries: list[dict], now: datetime) -> list[dict]:
    return [
        {
            "system": e["system"],
            "scope": e["scope"],
            "credential_id": e["id"],
            "owner": e["owner"],
            "owner_email": e["owner_email"],
            "backup_owner": "",
            "linked_identity": "",
            "linked_email": "",
            "issuer": e["issuer"],
            "rotation_days": 0,
            "last_rotated_at": None,
            "runbook_url": "",
            "notes": e["notes"],
            "updated_at": _ts(now),
            "updated_by": UPDATED_BY,
            "props": {},
        }
        for e in entries
    ]


def rows(now: datetime | None = None) -> tuple[list[dict], list[dict]]:
    """(inventory rows, owner rows), each checked against its table model."""
    now = now or datetime.now(timezone.utc)
    entries = load()
    inv, own = inventory_rows(entries, now), owner_rows(entries, now)
    for r in inv:
        CREDENTIAL_INVENTORY.row(r)
    for r in own:
        CREDENTIAL_OWNERS.row(r)
    return inv, own


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory-out", default="credential_inventory.registry.jsonl")
    ap.add_argument("--owners-out", default="credential_owners.registry.jsonl")
    args = ap.parse_args(argv)
    inv, own = rows()
    for path, data in ((args.inventory_out, inv), (args.owners_out, own)):
        with open(path, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in data))
    print(f"credential_registry: {len(inv)} inventory rows, {len(own)} owner rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
