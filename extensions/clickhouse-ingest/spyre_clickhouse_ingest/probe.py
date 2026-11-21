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

"""`probe`: a write/read round trip with a deterministic dedup token on a throwaway table.

Writes N rows through the resilient path, sends the same batch again under the same token, and
reads the table back: a Replicated table must hold N rows, not 2N. The writer login only inserts
and selects; the table is created once by the schema login (`probe --ensure`) and its rows expire
by TTL. `probe(client, table)` is importable and returns the result dict.

Exit status: 0 healthy, 1 not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

from . import writes
from .resilience import (
    Policy,
    batch_token,
    resilient_insert,
    safe,
    _send,
    _settings,
)

TABLE = "zz_write_probe"
COLUMNS = ["probe_id", "seq"]
# The bare engine resolves its keeper path only inside a Replicated database; pass --engine
# with explicit arguments anywhere else.
DDL = (
    "CREATE TABLE IF NOT EXISTS {table} (probe_id String, seq UInt32, "
    "written_at DateTime DEFAULT now()) ENGINE = {engine} "
    "ORDER BY (probe_id, seq) TTL written_at + INTERVAL 1 DAY"
)


def split(table: str) -> tuple[str | None, str]:
    db, _, name = table.rpartition(".")
    return (db or None), name


def _count(client, table: str, probe_id: str) -> int:
    return int(
        client.query(
            f"SELECT count() FROM {table} WHERE probe_id = {{p:String}}",
            parameters={"p": probe_id},
        ).result_rows[0][0]
    )


def probe(
    client,
    table: str,
    *,
    rows: int = 3,
    read_timeout_s: float = 30.0,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    db, name = split(table)
    result: dict = {"ok": False, "table": table, "rows": rows}
    try:
        found = writes.target(client, name, db)
        if found is None:
            result["error"] = "table not found or engine unreadable"
            return result
        result["replicated"] = found.replicated
        probe_id = uuid.uuid4().hex
        batch = [[probe_id, i] for i in range(1, rows + 1)]
        policy = Policy.from_env()
        policy.batch_rows = rows
        policy.degrade_to = None
        t0 = clock()
        out = resilient_insert(
            client, name, batch, COLUMNS, scope=probe_id, policy=policy,
            replicated=found.replicated, database=db, feeds_views=found.feeds_views,
        )  # fmt: skip
        result["write_ms"] = int((clock() - t0) * 1000)
        result["stored"] = out.stored
        result["dedup"] = "skipped"
        if found.replicated:
            token = batch_token(name, probe_id, batch, 1)
            _send(
                client, name, batch, COLUMNS, db,
                _settings(policy, token, True, False, found.feeds_views),
            )  # fmt: skip
            result["dedup"] = "held"
        t1 = clock()
        seen = -1
        while True:
            seen = _count(client, table, probe_id)
            if seen >= rows or clock() - t1 >= read_timeout_s:
                break
            sleep(1.0)
        result["read_ms"] = int((clock() - t1) * 1000)
        result["count"] = seen
        if found.replicated and seen > rows:
            result["dedup"] = "doubled"
        result["ok"] = out.stored == rows and seen == rows
    except Exception as err:  # noqa: BLE001 -- the probe reports, it never raises
        result["error"] = safe(err)
    return result


def main(argv=None) -> int:
    from .client import ClickHouse, ClickHouseEnv

    p = argparse.ArgumentParser(prog="probe", description=__doc__.split("\n\n")[0])
    p.add_argument("--table", help=f"db.table (default: <CLICKHOUSE_DB_V2>.{TABLE})")
    p.add_argument(
        "--ensure", action="store_true", help="create the table (schema login)"
    )
    p.add_argument(
        "--engine", default="ReplicatedMergeTree", help="engine for --ensure"
    )
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--read-timeout", type=float, default=30.0)
    args = p.parse_args(argv)
    db = ClickHouseEnv.target_database() or ClickHouseEnv.database()
    table = args.table or f"{db}.{TABLE}"
    if args.ensure:
        ClickHouse.connect(database=db, role="schema").command(
            DDL.format(table=table, engine=args.engine)
        )
    result = probe(
        ClickHouse.connect(database=db),
        table,
        rows=args.rows,
        read_timeout_s=args.read_timeout,
    )
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    from .writes import install_exit_hook

    install_exit_hook()
    sys.exit(main())
