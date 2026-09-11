"""The v2 ClickHouse schema, as data — one module, byte-identical in all three product repos.

WHY THIS EXISTS. Every v2 insert used to be a positional list paired with a separate
column_names list, and the same four tables were assembled independently in three repos.
Nothing tied a row's field order to the column list except the author reading both. The
audited state was correct (20/20 inserts column-named, 17/17 arity right), so this is not a
bug fix -- it makes a whole class of mistake unrepresentable, and it removes the divergence
that let one real defect live in two repos and not the third: hf-adapters and spyre-inference
never dedup identity rows across runs, so a case seen in N runs became N rows in test_cases.

WHY A TABLE MODEL AND NOT AN INGESTER CLASS. The three ingest scripts run two ways --
directly from a checkout by GitHub Actions, and from inside a baked test image via
`uv run --no-project --with lxml --with clickhouse-connect --with regex`. `--no-project` is
deliberate (uv otherwise tries to sync the torch-spyre project and exits 2, dropping the
ingest), so there is no sys.path beyond the script's own directory and those three wheels.
Nothing here may be imported from another repo or installed as a package: this file is COPIED,
and a drift check keeps the copies honest. Per-repo variation is one constant, COMPONENT,
which is why a class hierarchy would have been the wrong shape.

WHY NOT THE DRIVER'S OWN SCHEMA SUPPORT. clickhouse-connect has none to use. Its `ColumnDef`
is what DESCRIBE TABLE returns -- it reads a live table's schema, it cannot declare one or check
a row against it -- and `Client.insert` takes "a sequence of sequences" plus an ordered column-name
list, so positional rows are the native API shape rather than a style choice here. Passing
`column_names='*'` only moves the order to whatever the server currently reports, which couples
every row to the live DDL instead of removing the hazard. A declarative layer does exist in
clickhouse-sqlalchemy, and it would subsume most of this file -- but adding a dependency is what
the `uv run --no-project` runtime above rules out. Revisit if that constraint ever lifts.

WHAT IT DELIBERATELY DOES NOT DO. No runtime type coercion (duration_s typed Float32 accepts
a str), no ClickHouse type mapping, and no identity computation -- run_id and test_case_id
arrive already computed by the uuid5 helpers, which are untouched by design: changing them
re-keys the warehouse and silently breaks v2_already_ingested dedup, producing duplicate rows
rather than an error.
"""

from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

# The DDL's CONSTRAINT chk_status, re-expressed. It cannot be read from the server at ingest
# time, so it is duplicated here -- keep in step with functional_tests_v2.sql.
STATUS_VALUES = frozenset({"passed", "failed", "error", "skipped", "xfail", "xpass"})


class SchemaError(ValueError):
    """A row that the DDL would reject, or that names a column the table does not have."""


@dataclass(frozen=True)
class Table:
    """One v2 table: its columns in DDL order, and how a row is built.

    `columns` is the single place the order lives. Rows are built from a dict keyed by
    column name, so a field can never be assigned to the wrong column by position.
    """

    name: str
    columns: tuple[str, ...]
    # Columns that must be non-empty, mirroring the DDL's CHECK constraints.
    required: tuple[str, ...] = ()
    # id column for cross-run identity dedup; None for fact tables, which append freely.
    identity: str | None = None

    def row(self, values: dict[str, Any]) -> list[Any]:
        """Order one row by `columns`. Raises on an unknown or missing column.

        The raise is the point: an inserted or renamed column shows up here, at the call
        site, instead of shifting every later value into the wrong column.
        """
        unknown = set(values) - set(self.columns)
        if unknown:
            raise SchemaError(
                f"{self.name}: no such column(s) {sorted(unknown)}; "
                f"table has {list(self.columns)}"
            )
        missing = set(self.columns) - set(values)
        if missing:
            raise SchemaError(f"{self.name}: missing column(s) {sorted(missing)}")
        for col in self.required:
            if values[col] in ("", None):
                raise SchemaError(f"{self.name}: column '{col}' must be non-empty")
        if "status" in self.columns and values["status"] not in STATUS_VALUES:
            raise SchemaError(
                f"{self.name}: status {values['status']!r} violates the DDL CHECK "
                f"(allowed: {sorted(STATUS_VALUES)})"
            )
        return [values[c] for c in self.columns]

    def qualified(self, db: str | None) -> str:
        """`db.table` when a database is given, bare table otherwise.

        Every v2 statement is qualified because one client now serves both generations:
        `benchmark_runs` exists in v1 AND v2 with incompatible shapes, so an unqualified
        name would resolve against whichever database the connection happens to hold.
        """
        return f"{db}.{self.name}" if db else self.name


# ── the four v2 tables, columns in DDL order ────────────────────────────────────────────
# `ts` is omitted from every one: it is DEFAULT now() and letting the server set it keeps the
# ingest clock out of the data.

TEST_CASES = Table(
    name="test_cases",
    columns=("test_case_id", "component", "classname", "name", "tags"),
    required=("component", "name"),
    identity="test_case_id",
)

TEST_CASE_RUNS = Table(
    name="test_case_runs",
    # props carries source_file, the per-XML discriminator the dedup checks: a sharded run is
    # many files under ONE run_id, so a run-level check would let the first shard block the rest.
    columns=(
        "run_id",
        "test_case_id",
        "component",
        "status",
        "duration_s",
        "fail_message",
        "props",
    ),
    required=("component",),
)

BENCHMARKS = Table(
    name="benchmarks",
    columns=("benchmark_id", "component", "name", "tags", "props"),
    required=("component", "name"),
    identity="benchmark_id",
)

BENCHMARK_RUNS = Table(
    name="benchmark_runs",
    # component leads the identity hash and the sort key, so two repos writing the same
    # benchmark name stay distinct rows rather than colliding on one benchmark_id.
    columns=(
        "run_id",
        "benchmark_id",
        "component",
        "backend",
        "measurements",
        "iterations",
        "props",
    ),
    required=("component",),
)

TABLES = {t.name: t for t in (TEST_CASES, TEST_CASE_RUNS, BENCHMARKS, BENCHMARK_RUNS)}


def insert(
    client, table: Table, rows: Sequence[dict[str, Any]], db: str | None = None
) -> int:
    """Insert dicts into `table`, ordering every row through the one column list."""
    if not rows:
        return 0
    ordered = [table.row(r) for r in rows]
    client.insert(
        table.name, ordered, column_names=list(table.columns), database=db or None
    )
    return len(ordered)


def insert_identities(
    client, table: Table, rows: dict[Any, dict[str, Any]], db: str | None = None
) -> int:
    """Insert only the identity rows the dimension does not already hold.

    Both dimensions are plain MergeTree, so re-inserting a known identity APPENDS a duplicate
    rather than collapsing it -- one case seen in 36 runs became 36 rows. Deduping within a run
    is not enough because the collision is across runs. Centralising it here is what stops the
    check from being present in one repo and missing in the other two.
    """
    if not rows:
        return 0
    if not table.identity:
        raise SchemaError(f"{table.name} has no identity column")
    ids = [str(k) for k in rows]
    known = {
        str(r[0])
        for r in client.query(
            f"SELECT {table.identity} FROM {table.qualified(db)} "
            f"WHERE {table.identity} IN {{ids:Array(UUID)}}",
            parameters={"ids": ids},
        ).result_rows
    }
    fresh = [v for k, v in rows.items() if str(k) not in known]
    return insert(client, table, fresh, db=db)
