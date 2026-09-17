"""ClickHouse connection and v2 database/table presence.

The v2 database is a NAME, not a second connection: one instance holds both generations, so a
single client serves both provided every v2 statement is qualified.
"""

import os

import clickhouse_connect

from . import schema


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
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
