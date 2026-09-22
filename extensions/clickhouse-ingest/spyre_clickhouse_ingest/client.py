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

"""ClickHouse connection and v2 database/table presence.

The v2 database is a NAME, not a second connection: one instance holds both generations,
so a single client serves both provided every v2 statement is qualified.
"""

import os

import clickhouse_connect

from . import schema


def _env(name: str, default: str = "") -> str:
    """An env var, treating BLANK as absent.

    GitHub Actions exports an unset secret as the empty string, so os.environ.get(name, default)
    returns "" and never the default -- which made every fallback below unreachable and turned a
    missing CLICKHOUSE_PORT into `int("")` with an opaque ValueError.
    """
    return (os.environ.get(name) or "").strip() or default


def get_client(*, verify: bool = True):
    """The one ClickHouse connection factory for every ingest in this repo.

    `verify` exists because one ingest talks to an endpoint whose certificate does not validate;
    it is a parameter rather than a second copy of this function.

    CLICKHOUSE_SECURE=0 drops to plain HTTP, which is the only way to reach a local container
    (no TLS) -- without it this factory cannot be exercised outside CI, so a test either skips
    the real insert or hand-rolls a second connection that no longer matches production.
    """
    host = _env("CLICKHOUSE_HOST")
    if not host:
        raise SystemExit(
            "CLICKHOUSE_HOST is unset or empty -- check the workflow's secrets mapping"
        )
    secure = _env("CLICKHOUSE_SECURE", "1") not in ("0", "false", "no")
    password = _env("CLICKHOUSE_PASS")
    if not password and secure:
        raise SystemExit(
            "CLICKHOUSE_PASS is unset or empty -- check the workflow's secrets mapping"
        )
    port_raw = _env("CLICKHOUSE_PORT", "443")
    try:
        port = int(port_raw)
    except ValueError:
        raise SystemExit(f"CLICKHOUSE_PORT is not a number: {port_raw!r}") from None
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        user=_env("CLICKHOUSE_USER", "default"),
        password=password,
        database=_env("CLICKHOUSE_DB", "spyre"),
        secure=secure,
        verify=verify,
    )


def client_summary() -> str:
    """A host:port/database string for logging, read through the same resolver as the connection,
    so a banner cannot claim a port the client did not use."""
    return (
        f"{_env('CLICKHOUSE_HOST')}:{_env('CLICKHOUSE_PORT', '443')}"
        f"/{_env('CLICKHOUSE_DB', 'spyre')}"
    )


def target_database() -> str:
    """The v2 database name, or "" when v2 is not configured.

    A NAME rather than a second connection: the same instance holds both generations, so
    one client serves both provided every v2 statement is QUALIFIED. Qualifying is not
    optional -- `benchmark_runs` exists in both with incompatible shapes (v1 has run_id
    UInt64 + source_file, v2 has run_id UUID and no source_file), so an unqualified name
    resolves against whichever database the connection holds and silently hits the wrong
    table.
    """
    return os.environ.get("CLICKHOUSE_DB_V2", "").strip()


def tables_present(client, db: str, tables=None, check_columns: bool = True) -> bool:
    """v2 write path is skipped unless every table it needs exists, so this can be
    deployed before the migration without erroring on every run.

    `tables` defaults to the functional pair the JUnit ingests need; a benchmark writer
    must pass its own, since the two write paths land in separate migrations.

    Names come from the schema model, not string literals: this check is mirrored in the
    product repos, and a hardcoded name could drift from the table it checks while still
    looking correct.

    check_columns diffs each table's live columns against the schema model, and defaults ON:
    prod once had a `benchmarks` missing `component` (which leads the identity hash) and an
    existence-only gate passed it, so the gap surfaced as an opaque column-mismatch deep
    inside the insert. A caller that has to opt in is a caller that will forget, and the cost
    is one extra query per table per ingest. Pass False only to skip the round trip knowingly.
    """
    for t in tables or (schema.TEST_CASES, schema.TEST_CASE_RUNS):
        if not bool(client.command(f"EXISTS TABLE {t.qualified(db)}")):
            return False
        if check_columns:
            rows = client.query(
                "SELECT name FROM system.columns "
                "WHERE database = {db:String} AND table = {t:String}",
                parameters={"db": db, "t": t.name},
            ).result_rows
            if set(t.columns) - {r[0] for r in rows}:
                return False
    return True
