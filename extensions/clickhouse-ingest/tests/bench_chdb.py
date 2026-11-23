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

"""chdb helpers for the invalidation tests: the schema subset, placements, and a clickhouse_connect-shaped client."""

import json
import regex as re
import uuid
from datetime import datetime, timezone

import pytest
from bench_verdict_fixture import run_uuid
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

FILES = (
    "10-functional-tests",
    "20-artifacts",
    "30-benchmarks",
    "49-result-uploads",
    "50-artifact-views",
    "60-benchmark-views",
    "62-benchmark-verdicts",
    "81-bench-environments",
    "82-run-placements",
    "83-result-invalidations",
    "85-benchmark-verdict-views",
    "89-bench-invalidation-preview",
)
NIL = "00000000-0000-0000-0000-000000000000"
REPEATS = 3
# run index -> (node, serial); runs 4 and 5 ran on a different card of node n1; run 6 spanned two nodes.
LAYOUT = {n: ("n1", "SN1") if n % 2 == 0 else ("n2", "SN2") for n in range(22)}
LAYOUT[4] = ("n1", "SN9")
LAYOUT[5] = ("n1", "SN9")
PID = {n: str(uuid.uuid5(uuid.UUID(NIL), f"pl:{n}")) for n in range(40)}


class Db:
    def __init__(self):
        self.s = session.Session()
        for name in FILES:
            path = SCHEMA_DIR / f"{name}.sql"
            for o in SchemaApplier.objects(path, path.read_text()):
                if o.kind != "refresh":
                    self.s.query(o.sql)

    def rows(self, sql):
        raw = self.s.query(sql, "JSONEachRow").data()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        return [json.loads(line) for line in raw.splitlines()]

    def insert(self, table, *rows):
        body = "\n".join(json.dumps(r) for r in rows)
        self.s.query(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")


def device(node, serial):
    return {"node": node, "serial": serial, "pci": "", "firmware": "", "link": ""}


def place(db, n, started, nodes=None, serials=None, pid=None):
    if not nodes:
        node, serial = LAYOUT[n]
        nodes, serials = (
            ([node], [serial]) if n != 6 else (["n1", "n2"], ["SN1", "SN2"])
        )
    for rep in range(REPEATS):
        db.insert(
            "run_placements",
            {"run_id": run_uuid(n) if isinstance(n, int) else n, "repeat": rep, "env_id": run_uuid(900),
             "placement_id": pid or PID[n if isinstance(n, int) else 0], "started_at": started(n, rep), "nodes": nodes,
             "devices": [device(nd, sn) for nd, sn in zip(nodes, serials)]},
        )  # fmt: skip


class Result:
    def __init__(self, names, rows):
        self.column_names, self.result_rows = names, rows


def literal(value, kind):
    if kind.startswith("Array(UUID)"):
        return "[" + ", ".join(f"toUUID('{v}')" for v in value) + "]"
    if kind in ("String", "UUID"):
        return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"
    return str(int(value))


def typed(value, kind):
    if value is None:
        return None
    if kind.startswith("DateTime"):
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return value


class ChdbClient:
    """The slice of clickhouse_connect the invalidation module uses, over a chdb Db."""

    def __init__(self, db):
        self.db = db

    def query(self, sql, parameters=None):
        params = parameters or {}
        sql = re.sub(
            r"\{(\w+):([^}]+)\}", lambda m: literal(params[m.group(1)], m.group(2)), sql
        )
        raw = self.db.s.query(sql, "JSONCompact").data()
        doc = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        kinds = [c["type"] for c in doc["meta"]]
        rows = [[typed(v, k) for v, k in zip(row, kinds)] for row in doc["data"]]
        return Result([c["name"] for c in doc["meta"]], rows)

    def insert(self, table, rows, column_names=None):
        def cell(v):
            if isinstance(v, datetime):
                return v.strftime("%Y-%m-%d %H:%M:%S.") + f"{v.microsecond // 1000:03d}"
            return str(v) if isinstance(v, uuid.UUID) else v

        self.db.insert(
            table, *[{c: cell(v) for c, v in zip(column_names, row)} for row in rows]
        )

    def set_client_setting(self, name, value):
        pass
