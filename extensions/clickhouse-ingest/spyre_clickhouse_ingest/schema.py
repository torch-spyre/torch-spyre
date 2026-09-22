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

"""The v2 ClickHouse schema, as data — one module, imported by every consumer.

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
ingest), so there is no sys.path beyond the script's own directory and what `--with` installs.
This module was originally COPIED per repo for that reason; it is now installed as this package
via `--with`, so the copies and their drift check are gone. Per-repo variation is one constant,
COMPONENT, which is why a class hierarchy would have been the wrong shape -- and why
`component_of` takes the default as a PARAMETER rather than reading it from here.

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
re-keys the warehouse and silently breaks cases_already_ingested dedup, producing duplicate rows
rather than an error.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# The DDL's CHECK constraints, re-expressed. They cannot be read from the server at ingest
# time, so they are duplicated here -- keep in step with schema/10-functional-tests.sql
# (status) and schema/20-artifacts.sql (the rest).
STATUS_VALUES = frozenset({"passed", "failed", "error", "skipped", "xfail", "xpass"})
KIND_VALUES = frozenset({"image", "rpm", "wheel", "generic"})
ORIGIN_VALUES = frozenset({"built", "copied", "promoted", "upstream"})
METHOD_VALUES = frozenset({"container-pull", "dnf", "pip", "download"})
REF_KIND_VALUES = frozenset({"pullspec", "glob", "url"})
RESULT_KIND_VALUES = frozenset({"functional", "performance", "image"})
TEST_TYPE_VALUES = frozenset(
    {"smoke", "unit", "integration", "regression", "trunk", "perf", "capability"}
)
# capability_runs.test_type: which capability analysis produced the row. A sibling vocabulary to
# TEST_TYPE_VALUES, not a subset of it -- artifact_results calls the whole family 'capability'
# (one tier alongside regression/perf), and these name the analyses within it.
CAPABILITY_TYPE_VALUES = frozenset({"model_ops", "model_support"})
STATE_VALUES = frozenset({"passed", "failed", "error", "running"})
# capability_runs.status. NOT the test_case_runs vocabulary: a capability that is
# not_implemented is an unsupported capability, not a skipped test, and a CPU fallback is a
# `passed` here with backend='cpu' rather than a status of its own.
CAPABILITY_STATUS_VALUES = frozenset({"passed", "failed", "not_implemented"})

# NOT constrained, deliberately: the DDL documents tag_family as a declared, extensible set
# ('nightly | weekly | main | pr') with no CHECK, so validating it here would reject a channel
# the schema permits. Same for arch, which carries two spellings by table family.

# A dep entry in artifacts.identity_deps / context_deps is "<component>@<id12>" -- e.g.
# 'flex@d026bd2d255e' -- or 'base=<sha256>' for a base image named by content, or a bare
# component name when nothing pinned it. NOT a uuid: id12 is a hash INPUT to artifact_id, so
# artifact_id cannot be recovered from the string. A reader resolves it via props['id12'].
# This is stated here because it is the contract a reader must not guess: a dashboard route
# that looked these up with `artifact_id IN (...)` matched zero rows and rendered nothing,
# with no error, until it was found by querying prod.
DEP_ENTRY_SEP = "@"
DEP_BASE_PREFIX = "base="


def dep_id12(entry: str) -> str:
    """The id12 a dep entry names, or '' when it names none (bare name, or 'base=<sha>')."""
    s = str(entry or "")
    if not s or s.startswith(DEP_BASE_PREFIX) or DEP_ENTRY_SEP not in s:
        return ""
    return s.rsplit(DEP_ENTRY_SEP, 1)[1]


def dep_component(entry: str) -> str:
    """The component a dep entry names, without its pin."""
    s = str(entry or "")
    if s.startswith(DEP_BASE_PREFIX):
        return ""
    return s.rsplit(DEP_ENTRY_SEP, 1)[0] if DEP_ENTRY_SEP in s else s


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
    # column -> the DDL CHECK's allowed set. Declared per table rather than inferred from a
    # column name, so two tables can constrain the same name differently: `state` here is the
    # artifact_results vocabulary, which is NOT test_case_runs' `status` set.
    enums: tuple[tuple[str, frozenset[str]], ...] = ()
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
        for col, allowed in self.enums:
            if values[col] not in allowed:
                raise SchemaError(
                    f"{self.name}: {col} {values[col]!r} violates the DDL CHECK "
                    f"(allowed: {sorted(allowed)})"
                )
        return [values[c] for c in self.columns]

    def qualified(self, db: str | None) -> str:
        """`db.table` when a database is given, bare table otherwise.

        Every v2 statement is qualified because one client now serves both generations:
        `benchmark_runs` exists in v1 AND v2 with incompatible shapes, so an unqualified
        name would resolve against whichever database the connection happens to hold.
        """
        return f"{db}.{self.name}" if db else self.name


# ── the v2 functional/benchmark tables, columns in DDL order ────────────────────────────
# Source of truth: schema/10-functional-tests.sql, alongside this file.
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
    enums=(("status", STATUS_VALUES),),
)

# Source of truth: schema/30-benchmarks.sql, alongside this file. `measurements` there is
# Map(String, Array(Float64)) -- a metric's samples, not one number; this model does no type
# coercion, so a scalar is refused by the server rather than here.
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

# ── the four v2 ARTIFACT tables, columns in DDL order ───────────────────────────────────
# Source of truth: schema/20-artifacts.sql, alongside this file. Modelled here for the same
# reason as the tables above -- the writer and the readers had no shared statement of a row's
# shape, and the artifact tables are where that actually cost us.
#
# `ts` omitted throughout, as above: DEFAULT now() on the server.

# Source of truth: schema/46-capabilities.sql. The identity/observation split mirrors
# test_cases/test_case_runs for the same reason: 246,292 v1 rows carried only 46,607 distinct
# identities, so a flat table repeated the subject and the input signature on every row.
CAPABILITIES = Table(
    name="capabilities",
    columns=(
        "capability_id",
        "component",
        "test_type",
        "subject",
        "name",
        "tags",
        "props",
    ),
    required=("component", "test_type", "name"),
    identity="capability_id",
)

CAPABILITY_RUNS = Table(
    name="capability_runs",
    columns=(
        "run_id",
        "capability_id",
        "component",
        "test_type",
        "arch",
        "status",
        "backend",
        "fail_reason",
        "props",
    ),
    required=("component", "test_type"),
    enums=(("status", CAPABILITY_STATUS_VALUES),),
)

ARTIFACTS = Table(
    name="artifacts",
    # sources is Array(Tuple(repo, git_ref, git_sha)) -- a 3-element sequence per source, in
    # that order. identity_deps/context_deps are Array(String) of dep entries (see dep_id12).
    columns=(
        "artifact_id",
        "component",
        "arch",
        "kind",
        "artifact_name",
        "origin",
        "identity_deps",
        "context_deps",
        "sources",
        "props",
    ),
    # arch is required by the DDL's own comment: one id12 exists per arch plus a 'multi'
    # pointer, and dropping it collided 1,043 rows.
    required=("component", "arch"),
    # Not `identity=`: artifacts is plain MergeTree precisely so a duplicate artifact_id stays
    # visible as the producer bug it is. Dedup here would hide it.
    enums=(("kind", KIND_VALUES), ("origin", ORIGIN_VALUES)),
)

ARTIFACT_REFS = Table(
    name="artifact_refs",
    columns=(
        "artifact_id",
        "method",
        "ref_kind",
        "index_uri",
        "ref",
        "content_digest",
        "props",
    ),
    required=("ref",),
    enums=(("method", METHOD_VALUES), ("ref_kind", REF_KIND_VALUES)),
)

ARTIFACT_TAGS = Table(
    name="artifact_tags",
    # refs mirrors artifact_refs' shape for the tag's own published addresses; published_refs
    # is the flat list. tag_family is NOT enum-checked -- the DDL declares it without a CHECK.
    columns=(
        "tag",
        "tag_family",
        "artifact_id",
        "refs",
        "published_refs",
        "props",
    ),
    required=("tag",),
)

ARTIFACT_RESULTS = Table(
    name="artifact_results",
    # No stored counters: the v2 DDL removed total_tests/passed/failed because a stored copy is
    # a second source of truth that drifts once a delta run copies a covering run's cases in.
    # Readers derive them from test_case_runs.
    columns=(
        "artifact_id",
        "run_id",
        "result_kind",
        "test_type",
        "state",
        "arch",
        "duration_s",
        "props",
    ),
    enums=(
        ("result_kind", RESULT_KIND_VALUES),
        ("test_type", TEST_TYPE_VALUES),
        ("state", STATE_VALUES),
    ),
)

TABLES = {
    t.name: t
    for t in (
        TEST_CASES,
        TEST_CASE_RUNS,
        BENCHMARKS,
        BENCHMARK_RUNS,
        ARTIFACTS,
        ARTIFACT_REFS,
        ARTIFACT_TAGS,
        ARTIFACT_RESULTS,
        CAPABILITIES,
        CAPABILITY_RUNS,
    )
}


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
