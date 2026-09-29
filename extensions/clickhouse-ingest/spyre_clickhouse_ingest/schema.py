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

"""The v2 ClickHouse schema as classes: one class per table, sharing `Table`."""

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

# The DDL's CHECK constraints, re-expressed: unreadable from the server at ingest
# time. Keep in step with schema/10-functional-tests.sql and schema/20-artifacts.sql.
STATUS_VALUES = frozenset({"passed", "failed", "error", "skipped", "xfail", "xpass"})
KIND_VALUES = frozenset({"image", "rpm", "wheel", "generic"})
ORIGIN_VALUES = frozenset({"built", "copied", "promoted", "upstream"})
METHOD_VALUES = frozenset({"container-pull", "dnf", "pip", "download"})
REF_KIND_VALUES = frozenset({"pullspec", "glob", "url"})
RESULT_KIND_VALUES = frozenset({"functional", "performance", "image"})
TEST_TYPE_VALUES = frozenset(
    {"smoke", "unit", "integration", "regression", "trunk", "perf", "capability"}
)
# Which capability analysis produced a capability_runs row -- a sibling vocabulary to
# TEST_TYPE_VALUES, not a subset of it.
CAPABILITY_TYPE_VALUES = frozenset({"model_ops", "model_support"})
STATE_VALUES = frozenset({"passed", "failed", "error", "running"})
# capability_runs.status: not_implemented is unsupported, not a skipped test.
CAPABILITY_STATUS_VALUES = frozenset({"passed", "failed", "not_implemented"})

# NOT constrained, deliberately: the DDL declares tag_family and arch without a CHECK.

# A dep entry is "<component>@<id12>", 'base=<sha256>', or a bare name -- NOT a uuid,
# since id12 is a hash INPUT to artifact_id. A reader resolves it via props['id12'].
DEP_ENTRY_SEP = "@"
DEP_BASE_PREFIX = "base="


class SchemaError(ValueError):
    """A row the DDL would reject, or one naming a column the table does not have."""


class DepEntry:
    """One entry of artifacts.identity_deps / context_deps."""

    SEP = DEP_ENTRY_SEP
    BASE_PREFIX = DEP_BASE_PREFIX

    @classmethod
    def id12(cls, entry: str) -> str:
        """The id12 the entry names, or '' when it names none."""
        s = str(entry or "")
        if not s or s.startswith(cls.BASE_PREFIX) or cls.SEP not in s:
            return ""
        return s.rsplit(cls.SEP, 1)[1]

    @classmethod
    def component(cls, entry: str) -> str:
        """The component the entry names, without its pin."""
        s = str(entry or "")
        if s.startswith(cls.BASE_PREFIX):
            return ""
        return s.rsplit(cls.SEP, 1)[0] if cls.SEP in s else s


class Table:
    """Base for one v2 table: its columns in DDL order, its CHECKs, its write path."""

    # The single place column order lives; `ts` is omitted throughout (DEFAULT now()).
    name: str = ""
    columns: tuple[str, ...] = ()
    # Columns that must be non-empty, mirroring the DDL's CHECK constraints.
    required: tuple[str, ...] = ()
    # column -> allowed set, declared per table so two tables can differ on one name.
    enums: tuple[tuple[str, frozenset[str]], ...] = ()
    # id column for cross-run identity dedup; None for fact tables, which append freely.
    identity: str | None = None
    # The TypedDict shape a caller should build for this table -- checked by mypy/the editor,
    # never at runtime; `row()` below is still what actually validates a row.
    Row: type = dict

    @classmethod
    def row(cls, values: Mapping[str, Any]) -> list[Any]:
        """Order one row by `columns`, raising on an unknown, missing or bad value."""
        unknown = set(values) - set(cls.columns)
        if unknown:
            raise SchemaError(
                f"{cls.name}: no such column(s) {sorted(unknown)}; "
                f"table has {list(cls.columns)}"
            )
        missing = set(cls.columns) - set(values)
        if missing:
            raise SchemaError(f"{cls.name}: missing column(s) {sorted(missing)}")
        for col in cls.required:
            if values[col] in ("", None):
                raise SchemaError(f"{cls.name}: column '{col}' must be non-empty")
        for col, allowed in cls.enums:
            if values[col] not in allowed:
                raise SchemaError(
                    f"{cls.name}: {col} {values[col]!r} violates the DDL CHECK "
                    f"(allowed: {sorted(allowed)})"
                )
        return [values[c] for c in cls.columns]

    @classmethod
    def qualified(cls, db: str | None) -> str:
        """`db.table` when a database is given, bare table otherwise."""
        return f"{db}.{cls.name}" if db else cls.name

    @classmethod
    def insert(
        cls, client, rows: Sequence[Mapping[str, Any]], db: str | None = None
    ) -> int:
        """Insert dicts, ordering every row through the one column list."""
        if not rows:
            return 0
        ordered = [cls.row(r) for r in rows]
        client.insert(
            cls.name, ordered, column_names=list(cls.columns), database=db or None
        )
        return len(ordered)

    @classmethod
    def insert_identities(
        cls, client, rows: Mapping[Any, Mapping[str, Any]], db: str | None = None
    ) -> int:
        """Insert only the identity rows the dimension does not already hold."""
        if not rows:
            return 0
        if not cls.identity:
            raise SchemaError(f"{cls.name} has no identity column")
        ids = [str(k) for k in rows]
        known = {
            str(r[0])
            for r in client.query(
                f"SELECT {cls.identity} FROM {cls.qualified(db)} "
                f"WHERE {cls.identity} IN {{ids:Array(UUID)}}",
                parameters={"ids": ids},
            ).result_rows
        }
        fresh = [v for k, v in rows.items() if str(k) not in known]
        return cls.insert(client, fresh, db=db)

    @classmethod
    def present(cls, client, db: str, check_columns: bool = True) -> bool:
        """True when the table exists and holds at least the columns modelled here."""
        if not bool(client.command(f"EXISTS TABLE {cls.qualified(db)}")):
            return False
        if not check_columns:
            return True
        rows = client.query(
            "SELECT name FROM system.columns "
            "WHERE database = {db:String} AND table = {t:String}",
            parameters={"db": db, "t": cls.name},
        ).result_rows
        return not set(cls.columns) - {r[0] for r in rows}

    @classmethod
    def count_rows(cls, client, db: str, where: str, params: dict) -> int:
        """count() over this table under `where`, using named-parameter placeholders."""
        rows = client.query(
            f"SELECT count() FROM {cls.qualified(db)} WHERE {where}",
            parameters=params,
        ).result_rows
        return int(rows[0][0]) if rows else 0


# ── functional/benchmark tables (10-functional-tests.sql, 30-benchmarks.sql) ──


class TestCaseRow(TypedDict):
    test_case_id: str
    component: str
    classname: str
    name: str
    tags: list[str]


class TestCases(Table):
    """Test identity: one row per (component, classname, name, tags)."""

    name = "test_cases"
    columns = ("test_case_id", "component", "classname", "name", "tags")
    required = ("component", "name")
    identity = "test_case_id"
    Row = TestCaseRow


class TestCaseRunRow(TypedDict):
    run_id: str
    test_case_id: str
    component: str
    status: str
    duration_s: float
    fail_message: str
    props: dict[str, str]


class TestCaseRuns(Table):
    """One test's outcome in one run; props carries the source_file discriminator."""

    name = "test_case_runs"
    columns = (
        "run_id",
        "test_case_id",
        "component",
        "status",
        "duration_s",
        "fail_message",
        "props",
    )
    required = ("component",)
    enums = (("status", STATUS_VALUES),)
    Row = TestCaseRunRow


class BenchmarkRow(TypedDict):
    benchmark_id: str
    component: str
    name: str
    tags: list[str]
    props: dict[str, str]


class Benchmarks(Table):
    """Benchmark identity, component-scoped so two repos cannot collide on one name."""

    name = "benchmarks"
    columns = ("benchmark_id", "component", "name", "tags", "props")
    required = ("component", "name")
    identity = "benchmark_id"
    Row = BenchmarkRow


class BenchmarkRunRow(TypedDict):
    run_id: str
    benchmark_id: str
    component: str
    backend: str
    measurements: dict[str, list[float]]
    iterations: int
    props: dict[str, str]


class BenchmarkRuns(Table):
    """One benchmark's measurements in one run; `measurements` is a metric's SAMPLES."""

    name = "benchmark_runs"
    columns = (
        "run_id",
        "benchmark_id",
        "component",
        "backend",
        "measurements",
        "iterations",
        "props",
    )
    required = ("component",)
    Row = BenchmarkRunRow


# ── capability tables (schema/46-capabilities.sql) ──


class CapabilityRow(TypedDict):
    capability_id: str
    component: str
    test_type: str
    subject: str
    name: str
    tags: list[str]
    props: dict[str, str]


class Capabilities(Table):
    """Capability identity: one row per (subject, capability), mirroring test_cases."""

    name = "capabilities"
    columns = (
        "capability_id",
        "component",
        "test_type",
        "subject",
        "name",
        "tags",
        "props",
    )
    required = ("component", "test_type", "name")
    identity = "capability_id"
    Row = CapabilityRow


class CapabilityRunRow(TypedDict):
    run_id: str
    capability_id: str
    component: str
    test_type: str
    arch: str
    status: str
    backend: str
    fail_reason: str
    props: dict[str, str]


class CapabilityRuns(Table):
    """One capability's verdict; `backend` is a column, never part of the id."""

    name = "capability_runs"
    columns = (
        "run_id",
        "capability_id",
        "component",
        "test_type",
        "arch",
        "status",
        "backend",
        "fail_reason",
        "props",
    )
    required = ("component", "test_type")
    enums = (("status", CAPABILITY_STATUS_VALUES),)
    Row = CapabilityRunRow


# ── artifact tables (schema/20-artifacts.sql) ──


class ArtifactRow(TypedDict):
    artifact_id: str
    component: str
    arch: str
    kind: str
    artifact_name: str
    origin: str
    identity_deps: list[str]
    context_deps: list[str]
    sources: list[tuple[str, str, str]]
    props: dict[str, str]


class Artifacts(Table):
    """One built artifact; no `identity` -- a dup artifact_id is a producer bug."""

    name = "artifacts"
    # sources is Array(Tuple(repo, git_ref, git_sha)); dep arrays hold DepEntry strings.
    columns = (
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
    )
    # arch is required: one id12 exists per arch plus 'multi'; dropping it collides.
    required = ("component", "arch")
    enums = (("kind", KIND_VALUES), ("origin", ORIGIN_VALUES))
    Row = ArtifactRow


class ArtifactRefRow(TypedDict):
    artifact_id: str
    method: str
    ref_kind: str
    index_uri: str
    ref: str
    content_digest: str
    props: dict[str, str]


class ArtifactRefs(Table):
    """How a consumer obtains an artifact, and the address that takes."""

    name = "artifact_refs"
    columns = (
        "artifact_id",
        "method",
        "ref_kind",
        "index_uri",
        "ref",
        "content_digest",
        "props",
    )
    required = ("ref",)
    enums = (("method", METHOD_VALUES), ("ref_kind", REF_KIND_VALUES))
    Row = ArtifactRefRow


class ArtifactTagRow(TypedDict):
    tag: str
    tag_family: str
    artifact_id: str
    refs: list[tuple[str, str, str, str]]
    published_refs: list[str]
    props: dict[str, str]


class ArtifactTags(Table):
    """What a channel tag pointed to as of ts; tag_family is NOT enum-checked."""

    name = "artifact_tags"
    columns = ("tag", "tag_family", "artifact_id", "refs", "published_refs", "props")
    required = ("tag",)
    Row = ArtifactTagRow


class ArtifactResultRow(TypedDict):
    artifact_id: str
    run_id: str
    result_kind: str
    test_type: str
    state: str
    arch: str
    duration_s: float
    props: dict[str, str]


class ArtifactResults(Table):
    """One leg's verdict on one artifact; counters are derived, never stored."""

    name = "artifact_results"
    columns = (
        "artifact_id",
        "run_id",
        "result_kind",
        "test_type",
        "state",
        "arch",
        "duration_s",
        "props",
    )
    enums = (
        ("result_kind", RESULT_KIND_VALUES),
        ("test_type", TEST_TYPE_VALUES),
        ("state", STATE_VALUES),
    )
    Row = ArtifactResultRow


# Constant API, kept so installed consumers name one table model rather than copying it.
TEST_CASES = TestCases
TEST_CASE_RUNS = TestCaseRuns
BENCHMARKS = Benchmarks
BENCHMARK_RUNS = BenchmarkRuns
CAPABILITIES = Capabilities
CAPABILITY_RUNS = CapabilityRuns
ARTIFACTS = Artifacts
ARTIFACT_REFS = ArtifactRefs
ARTIFACT_TAGS = ArtifactTags
ARTIFACT_RESULTS = ArtifactResults

TABLES = {
    t.name: t
    for t in (
        TestCases,
        TestCaseRuns,
        Benchmarks,
        BenchmarkRuns,
        Artifacts,
        ArtifactRefs,
        ArtifactTags,
        ArtifactResults,
        Capabilities,
        CapabilityRuns,
    )
}

dep_id12 = DepEntry.id12
dep_component = DepEntry.component


def insert(
    client,
    table: type[Table],
    rows: Sequence[Mapping[str, Any]],
    db: str | None = None,
) -> int:
    """Insert dicts into `table` -- the free-function form of `Table.insert`."""
    return table.insert(client, rows, db=db)


def insert_identities(
    client,
    table: type[Table],
    rows: Mapping[Any, Mapping[str, Any]],
    db: str | None = None,
) -> int:
    """Insert unknown identity rows -- free function form of `insert_identities`."""
    return table.insert_identities(client, rows, db=db)
