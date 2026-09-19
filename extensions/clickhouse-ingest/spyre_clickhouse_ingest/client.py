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

The v2 database is a NAME, not a second connection: one instance holds both generations, so a
single client serves both provided every v2 statement is qualified.
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
    """
    host = _env("CLICKHOUSE_HOST")
    if not host:
        raise SystemExit(
            "CLICKHOUSE_HOST is unset or empty -- check the workflow's secrets mapping"
        )
    password = _env("CLICKHOUSE_PASS")
    if not password:
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
        secure=True,
        verify=verify,
    )


def client_summary() -> str:
    """A host:port/database string for logging, read through the same resolver as the connection,
    so a banner cannot claim a port the client did not use."""
    return (
        f"{_env('CLICKHOUSE_HOST')}:{_env('CLICKHOUSE_PORT', '443')}"
        f"/{_env('CLICKHOUSE_DB', 'spyre')}"
    )


def v2_database() -> str:
    """The v2 database name, or "" when v2 is not configured.

    A NAME rather than a second connection: the same instance holds both generations, so one
    client serves both provided every v2 statement is QUALIFIED. Qualifying is not optional --
    `benchmark_runs` exists in both with incompatible shapes (v1 has run_id UInt64 +
    source_file, v2 has run_id UUID and no source_file), so an unqualified name resolves
    against whichever database the connection holds and silently hits the wrong table.
    """
    return os.environ.get("CLICKHOUSE_DB_V2", "").strip()


def v2_tables_present(client, db: str) -> bool:
    """v2 write path is skipped unless BOTH tables exist, so this script can be
    deployed before the migration without erroring on every run.

    Names come from the schema model, not string literals: this file is copied across the
    product repos and the copies are compared for MEANING, so a hardcoded name here could
    drift from the table it is meant to check while still looking correct.
    """
    return all(
        bool(client.command(f"EXISTS TABLE {t.qualified(db)}"))
        for t in (schema.TEST_CASES, schema.TEST_CASE_RUNS)
    )
