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
RESULT_KIND_VALUES = frozenset({"functional", "performance", "capability"})
# Which capability analysis produced a capability_runs row; also the test_type of a
# result_kind='capability' verdict.
CAPABILITY_TYPE_VALUES = frozenset(
    {"model_ops", "model_modules", "model_support", "model_serving"}
)
# spyre-test-framework's stages, each its own leg.
SUITE_STAGE_VALUES = frozenset(
    {"fvt", "fvt-static", "fvt-dynamic", "svt", "svt-static", "svt-dynamic"}
)
# quality: accuracy evals (lm-eval tasks, MTEB, RAG), benchmark rows apart from perf's gates.
TEST_TYPE_VALUES = (
    frozenset(
        {"smoke", "unit", "integration", "regression", "trunk", "perf", "quality"}
    )
    | SUITE_STAGE_VALUES
    | CAPABILITY_TYPE_VALUES
)
STATE_VALUES = frozenset({"passed", "failed", "error", "running"})
# artifact_result_reasons.failure_reason, and props['failure_reason'] on a non-passing verdict:
# spyre-frameworks' diagnose_failure.py categories plus the outcomes a log cannot show.
FAILURE_REASON_VALUES = frozenset(
    {
        "test_failure",
        "build_error",
        "infra_disk",
        "infra_network",
        "infra_auth",
        "infra_capacity",
        "infra_hardware",
        "infra_timeout",
        "infra_env",
        "superseded",
        "aborted",
        "dependency_failed",
        "pipeline_error",
        "ingest_error",
        "unknown",
    }
)
# capability_runs.status: not_implemented is unsupported, not a skipped test; undetermined is a
# test that broke before giving a verdict.
CAPABILITY_STATUS_VALUES = frozenset(
    {"passed", "failed", "not_implemented", "undetermined"}
)

RUN_SOURCE_VALUES = frozenset({"jenkins", "gha"})
PIPELINE_TYPE_VALUES = frozenset(
    {"orchestrator", "component-build", "product-test", "gha-workflow", "gha-job"}
)
RUN_STATE_VALUES = frozenset({"running", "finished"})
# ci_run_timings: one row per build or test entry. A build was built, reused from a concurrent
# build, dropped (already published) or failed; '' is a state not reported.
TIMING_ENTRY_VALUES = frozenset({"build", "test"})
TIMING_BUILD_STATE_VALUES = frozenset({"built", "reused", "dropped", "failed", ""})
TIMING_TEST_STATE_VALUES = frozenset({"passed", "failed", "error", ""})
TIMING_STATE_VALUES = {
    "build": TIMING_BUILD_STATE_VALUES,
    "test": TIMING_TEST_STATE_VALUES,
}
TIMING_EXECUTOR_VALUES = frozenset(
    {"gha-ephemeral", "gha-standing", "jenkins-local", "jenkins-job", ""}
)
TARGET_TYPE_VALUES = frozenset({"jenkins", "gha_dispatch", "webhook"})
EVENT_TYPE_VALUES = frozenset(
    {"artifact.tagged", "artifact.recorded", "artifact.promoted", "results.recorded"}
)
DISPATCH_STATE_VALUES = frozenset({"queued", "triggered", "failed", "skipped"})
REQUESTED_BY_VALUES = frozenset({"user", "pipeline"})
DISPATCHED_BY_VALUES = REQUESTED_BY_VALUES | {"auto"}

# NOT constrained, deliberately: the DDL declares tag_family and arch without a CHECK.

# A dep entry is "<component>@<id12>", 'base=<sha256>', or a bare name -- NOT a uuid,
# since id12 is a hash INPUT to artifact_id. A reader resolves it via props['id12'].
DEP_ENTRY_SEP = "@"
DEP_BASE_PREFIX = "base="
IDENTITY_LOOKUP_CHUNK = 2000


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
        # Chunked: the ids travel as one HTTP form field, and ClickHouse rejects a field over
        # http_max_field_value_size (128 KiB) -- a 12k-case merged junit is ~470 KiB.
        known = {
            str(r[0])
            for i in range(0, len(ids), IDENTITY_LOOKUP_CHUNK)
            for r in client.query(
                f"SELECT {cls.identity} FROM {cls.qualified(db)} "
                f"WHERE {cls.identity} IN {{ids:Array(UUID)}}",
                parameters={"ids": ids[i : i + IDENTITY_LOOKUP_CHUNK]},
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
    """Test identity: one row per (component, classname, name, identity tags)."""

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
    tags: list[str]
    measurements: dict[str, float]


class TestCaseRuns(Table):
    """One test's outcome in one run: run-context tags, recorded measurements and results."""

    name = "test_case_runs"
    columns = (
        "run_id",
        "test_case_id",
        "component",
        "status",
        "duration_s",
        "fail_message",
        "props",
        "tags",
        "measurements",
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


class ArtifactResultReasonRow(TypedDict):
    artifact_id: str
    run_id: str
    result_kind: str
    test_type: str
    failure_reason: str
    failure_subreason: str
    failure_detail: str
    failure_log_url: str
    failure_wait_s: int
    confidence: int
    source: str
    props: dict[str, str]


class ArtifactResultReasons(Table):
    """Why a verdict did not pass, written after the verdict itself; one row per source."""

    name = "artifact_result_reasons"
    columns = (
        "artifact_id",
        "run_id",
        "result_kind",
        "test_type",
        "failure_reason",
        "failure_subreason",
        "failure_detail",
        "failure_log_url",
        "failure_wait_s",
        "confidence",
        "source",
        "props",
    )
    required = ("source",)
    enums = (
        ("result_kind", RESULT_KIND_VALUES),
        ("test_type", TEST_TYPE_VALUES),
        ("failure_reason", FAILURE_REASON_VALUES),
    )
    Row = ArtifactResultReasonRow


class PipelineRuns(Table):
    """One CI execution (Jenkins build or GHA run attempt / job), upserted start -> end.

    Jenkins rows arrive as JSONEachRow from vars/pushToClickhouse.groovy; this model is the
    shape the gha_runs poller writes through.
    """

    name = "pipeline_runs"
    columns = (
        "run_key",
        "updated_at",
        "source",
        "pipeline_type",
        "state",
        "job_name",
        "build_number",
        "attempt",
        "build_url",
        "agent",
        "parent_run_key",
        "started_at",
        "ended_at",
        "queue_ms",
        "duration_ms",
        "build_ms",
        "test_ms",
        "trigger_kind",
        "trigger_source",
        "preset",
        "build_mode",
        "repo",
        "pr_number",
        "sha",
        "test_with",
        "component",
        "arches",
        "result",
        "verdict",
        "superseded",
        "reached_normal_completion",
        "lane_results",
        "nodes_built",
        "nodes_reused",
        "nodes_dropped",
        "tests_total",
        "tests_failed",
        "ch_write_failures",
        "failure_reason",
        "failure_is_infra",
        "failure_evidence",
        "failed_stage",
        "fail_log_tail",
        "props",
    )
    required = ("run_key", "job_name")
    enums = (
        ("source", RUN_SOURCE_VALUES),
        ("pipeline_type", PIPELINE_TYPE_VALUES),
        ("state", RUN_STATE_VALUES),
    )


class CiRunTimings(Table):
    """One component build or test leg of an orchestrator run, trigger to teardown.

    Executor-neutral: a GHA workflow and a Jenkins `make test` fill the same provision/exec
    columns. `leg` and every `*_ms` are MATERIALIZED in the DDL, so they are not modelled here:
    the server derives them and ignores a value a writer supplies.
    """

    name = "ci_run_timings"
    columns = (
        "run_key",
        "updated_at",
        "entry",
        "component",
        "artifact_name",
        "arch",
        "id12",
        "kind",
        "test_modes",
        "attempt",
        "trigger_kind",
        "trigger_source",
        "preset",
        "build_mode",
        "trigger_pr",
        "repo",
        "pr_number",
        "sha",
        "base_ref",
        "is_pr_component",
        "build_url",
        "verdict",
        "run_result",
        "superseded",
        "pickup_path",
        "comment_at",
        "picked_up_at",
        "run_scheduled_at",
        "run_started_at",
        "pr_queued_at",
        "pr_running_at",
        "run_ended_at",
        "state",
        "result",
        "gating",
        "url",
        "agent",
        "queued_at",
        "started_at",
        "ended_at",
        "executor",
        "provision_started_at",
        "provision_ended_at",
        "exec_dispatched_at",
        "exec_started_at",
        "exec_ended_at",
        "exec_runs",
        "exec_jobs",
        "exec_result",
        "exec_urls",
        "exec_run_keys",
        "cards",
        "runner_died",
        "failure_reason",
        "failed_stage",
        "props",
    )
    # run_started_at is the partition key and not Nullable.
    required = ("run_key", "entry", "component", "arch", "run_started_at")
    enums = (
        ("entry", TIMING_ENTRY_VALUES),
        ("executor", TIMING_EXECUTOR_VALUES),
    )

    @classmethod
    def row(cls, values: Mapping[str, Any]) -> list[Any]:
        """As Table.row, plus chk_timing_state: the allowed states depend on the entry."""
        ordered = super().row(values)
        allowed = TIMING_STATE_VALUES[values["entry"]]
        if values["state"] not in allowed:
            raise SchemaError(
                f"{cls.name}: {values['entry']} state {values['state']!r} violates the "
                f"DDL CHECK (allowed: {sorted(allowed)})"
            )
        return ordered


class ArtifactSubscriptions(Table):
    """Who an artifact triggers; one row per version, the newest updated_at wins."""

    name = "artifact_subscriptions"
    columns = (
        "subscription_id",
        "updated_at",
        "enabled",
        "event_types",
        "tag_family",
        "tag_pattern",
        "tag_props",
        "exclude_families",
        "exclude_tag_patterns",
        "result_test_types",
        "result_states",
        "component",
        "arch",
        "kind",
        "artifact_name",
        "not_before",
        "require",
        "unless_exists",
        "max_per_hour",
        "max_per_day",
        "coalesce_minutes",
        "target_type",
        "target",
        "params",
        "payload_fields",
        "credential_id",
        "owner",
        "updated_by",
        "notes",
        "props",
    )
    required = ("subscription_id", "target")
    enums = (("target_type", TARGET_TYPE_VALUES),)


class DispatchRequests(Table):
    """A person's or pipeline's "run this subscription against this artifact"."""

    name = "dispatch_requests"
    columns = (
        "request_id",
        "requested_at",
        "subscription_id",
        "artifact_id",
        "tag",
        "requested_by",
        "requester",
        "params",
        "reason",
        "props",
    )
    required = ("request_id", "subscription_id", "artifact_id")
    enums = (("requested_by", REQUESTED_BY_VALUES),)


class ArtifactDispatches(Table):
    """One dispatch, upserted on every state change; written only by the dispatcher."""

    name = "artifact_dispatches"
    columns = (
        "dispatch_id",
        "updated_at",
        "subscription_id",
        "artifact_id",
        "event_type",
        "event_key",
        "tag",
        "event_ts",
        "requested_by",
        "requester",
        "request_id",
        "requested_at",
        "state",
        "target_type",
        "target",
        "params",
        "payload",
        "target_url",
        "build_url",
        "error",
        "attempts",
        "next_attempt_at",
        "dispatcher_url",
        "props",
    )
    required = ("dispatch_id", "subscription_id", "artifact_id")
    enums = (
        ("requested_by", DISPATCHED_BY_VALUES),
        ("state", DISPATCH_STATE_VALUES),
        ("target_type", TARGET_TYPE_VALUES | {""}),
    )


class DispatchTargets(Table):
    """The dispatcher's allow-list, one snapshot per publish; an empty list is one '' / '' row."""

    name = "dispatch_targets"
    columns = (
        "target_type",
        "target_prefix",
        "credential_id",
        "published_at",
        "job_run",
    )
    required = ("published_at",)


CREDENTIAL_STATE_VALUES = frozenset({"bound", "unbound", "gone"})
PROBE_STATUS_VALUES = frozenset({"ok", "failing", "unknown"})
PROBE_SITE_VALUES = frozenset({"jenkins", "gha", "dashboard", "cluster"})


class CredentialInventory(Table):
    """What discovery found per credential; the latest observed_at wins."""

    name = "credential_inventory"
    columns = (
        "system",
        "scope",
        "credential_id",
        "kind",
        "description",
        "consumers",
        "state",
        "source_created_at",
        "source_updated_at",
        "last_bound_at",
        "observed_at",
        "job_run",
        "props",
    )
    required = ("system", "scope", "credential_id", "observed_at")
    enums = (("state", CREDENTIAL_STATE_VALUES),)


class CredentialProbes(Table):
    """One probe of one credential, written by the site that holds the secret."""

    name = "credential_probes"
    columns = (
        "system",
        "scope",
        "credential_id",
        "probe_site",
        "probe",
        "status",
        "error_class",
        "http_status",
        "latency_ms",
        "expires_at",
        "fingerprint",
        "detail",
        "job_run",
        "props",
    )
    required = ("system", "scope", "credential_id", "probe")
    enums = (("status", PROBE_STATUS_VALUES), ("probe_site", PROBE_SITE_VALUES))


class CredentialOwners(Table):
    """What a person supplies about a credential; append-only, the newest row per credential wins."""

    name = "credential_owners"
    columns = (
        "system",
        "scope",
        "credential_id",
        "owner",
        "owner_email",
        "backup_owner",
        "linked_identity",
        "linked_email",
        "issuer",
        "rotation_days",
        "last_rotated_at",
        "runbook_url",
        "notes",
        "updated_at",
        "updated_by",
        "props",
    )
    required = ("system", "scope", "credential_id", "updated_at", "updated_by")


INCIDENT_SEVERITY_VALUES = frozenset({"info", "warning", "critical"})
INCIDENT_STATE_VALUES = frozenset(
    {"open", "acknowledged", "mitigated", "resolved", "dismissed"}
)
INCIDENT_ACTOR_VALUES = frozenset({"system", "agent", "ask_spyre", "human"})
INCIDENT_CAUSE_VALUES = frozenset(
    {
        "credentials",
        "resources",
        "network",
        "hardware",
        "upstream",
        "pipeline",
        "data",
        "unknown",
    }
)


class IncidentEvents(Table):
    """One normalized signal from any alert source; the router attaches it to an incident."""

    name = "incident_events"
    columns = (
        "ts",
        "source",
        "subject",
        "severity",
        "signature",
        "cause_class",
        "summary",
        "evidence_url",
        "dedup_key",
        "incident_id",
        "job_run",
        "props",
    )
    required = ("source", "subject", "severity")
    enums = (
        ("severity", INCIDENT_SEVERITY_VALUES),
        ("cause_class", INCIDENT_CAUSE_VALUES),
    )


class IncidentLog(Table):
    """Append-only incident timeline; the newest row per incident is its current state."""

    name = "incident_log"
    columns = (
        "incident_id",
        "updated_at",
        "updated_by",
        "actor_kind",
        "state",
        "severity",
        "title",
        "cause_class",
        "signature",
        "owner",
        "owner_email",
        "action",
        "note",
        "thread_url",
        "evidence_url",
        "props",
    )
    required = (
        "incident_id",
        "updated_at",
        "updated_by",
        "actor_kind",
        "state",
        "severity",
    )
    enums = (
        ("actor_kind", INCIDENT_ACTOR_VALUES),
        ("state", INCIDENT_STATE_VALUES),
        ("severity", INCIDENT_SEVERITY_VALUES),
        ("cause_class", INCIDENT_CAUSE_VALUES),
    )


ASSIGNMENT_MODE_VALUES = frozenset({"shadow", "propose", "live"})
SESSION_STATE_VALUES = frozenset({"open", "closed", "abandoned"})
TURN_OUTCOME_VALUES = frozenset({"ok", "rejected_schema", "error", "timeout", "budget"})
MEMORY_KIND_VALUES = frozenset({"case_summary", "lesson", "signature_candidate"})
MEMORY_STATUS_VALUES = frozenset(
    {"proposed", "confirmed", "rejected", "expired", "quarantined", "retracted"}
)


class AnalystAssignments(Table):
    """Which model alias plays a role for which incidents; the newest effective row wins."""

    name = "analyst_assignments"
    columns = (
        "role",
        "scope_cause_class",
        "scope_severity",
        "scope_source",
        "scope_repo",
        "model_alias",
        "mode",
        "weight",
        "effective_from",
        "set_by",
        "second_approver",
        "reason",
        "props",
    )
    required = ("role", "model_alias", "mode", "effective_from", "set_by")
    enums = (("mode", ASSIGNMENT_MODE_VALUES),)


class AnalystSessions(Table):
    """One investigation of one incident by one role; the newest row per session is its state."""

    name = "analyst_sessions"
    columns = (
        "session_id",
        "updated_at",
        "incident_id",
        "role",
        "state",
        "parent_session_id",
        "props",
    )
    required = ("session_id", "updated_at", "incident_id", "role")
    enums = (("state", SESSION_STATE_VALUES),)


class AnalystTurns(Table):
    """One model call within a session."""

    name = "analyst_turns"
    columns = (
        "session_id",
        "step",
        "ts",
        "role",
        "model_alias",
        "served_model",
        "prompt_version",
        "input_ref",
        "output",
        "verdict",
        "confidence",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "outcome",
        "props",
    )
    required = ("session_id", "step", "ts", "role", "model_alias")
    enums = (("outcome", TURN_OUTCOME_VALUES),)


class AnalystMemory(Table):
    """What the analyst has learned; append-only versions, the newest row per memory_id wins."""

    name = "analyst_memory"
    columns = (
        "memory_id",
        "updated_at",
        "kind",
        "subject",
        "body",
        "evidence_ids",
        "confidence",
        "status",
        "proposed_by",
        "confirmed_by",
        "expires_at",
        "props",
    )
    required = ("memory_id", "updated_at", "subject", "proposed_by")
    enums = (("kind", MEMORY_KIND_VALUES), ("status", MEMORY_STATUS_VALUES))


CASE_FILE_STATUS_VALUES = frozenset({"open", "closed"})


class CaseFiles(Table):
    """The analyst's working summary of an incident; append-only versions, the newest wins."""

    name = "case_files"
    columns = (
        "incident_id",
        "version",
        "updated_at",
        "status",
        "through_step",
        "keys",
        "doc",
        "props",
    )
    required = ("incident_id", "version", "updated_at", "doc")
    enums = (("status", CASE_FILE_STATUS_VALUES),)

    @classmethod
    def row(cls, values):
        out = super().row(values)
        if int(values["version"]) < 1:
            raise SchemaError(f"{cls.name}: version starts at 1")
        return out


ANALYST_ROLE_MEMORY_SCOPE_VALUES = frozenset({"private", "team", "shared"})
ANALYST_ROLE_MEMORY_KIND_VALUES = frozenset(
    {"case_summary", "lesson", "signature_candidate", "prompt_note", "failure_case"}
)
ANALYST_ROLE_MEMORY_STATUS_VALUES = frozenset(
    {
        "proposed",
        "confirmed",
        "rejected",
        "expired",
        "quarantined",
        "retracted",
        "active",
    }
)
ANALYST_PROJECT_FILES_KIND_VALUES = frozenset({"case_file", "note", "artefact"})
ANALYST_OUTCOMES_OUTCOME_VALUES = frozenset({"accepted", "rejected", "overturned"})
ANALYST_OUTCOMES_JUDGED_KIND_VALUES = frozenset({"human", "validator"})
ANALYST_WORK_ITEMS_STATUS_VALUES = frozenset(
    {"runnable", "waiting", "waiting_human", "parked", "closed"}
)
ANALYST_AGENT_CALLS_KIND_VALUES = frozenset({"ask", "cross_ref"})
ANALYST_AGENT_CALLS_OUTCOME_VALUES = frozenset({"ok", "refused", "timeout", "failed"})
ANALYST_AGENT_CALLS_STANCE_VALUES = frozenset(
    {"", "agree", "disagree", "unsure", "answer"}
)
ACCESS_DECISIONS_ROW_KIND_VALUES = frozenset({"decision", "complaint"})
ACCESS_DECISIONS_OUTCOME_VALUES = frozenset(
    {"allow", "redact", "deny", "route", "summarise", "declassify", "complaint"}
)
RECOVERY_ACTIONS_ROW_KIND_VALUES = frozenset(
    {"action", "refusal", "demotion", "reinstatement"}
)
RECOVERY_ACTIONS_MODE_VALUES = frozenset({"dry_run", "live", "n/a"})
ANALYST_OPTIMIZER_RECORDS_KIND_VALUES = frozenset(
    {"run", "audit", "suggestion", "pr_plan", "feedback"}
)
GOAL_SCORES_ROW_KIND_VALUES = frozenset({"kr", "role"})
GOAL_REVIEWS_STATUS_VALUES = frozenset(
    {"proposed", "awaiting_reviewer", "budget", "rejected_schema"}
)
GOAL_PROPOSALS_STATUS_VALUES = frozenset(
    {"proposed", "approved", "rejected", "applied"}
)
ASSISTANT_PROFILES_STATE_VALUES = frozenset({"active", "paused"})
ASSISTANT_INBOX_STATUS_VALUES = frozenset(
    {"new", "needs_reply", "ignored", "deferred", "delegated", "escalated"}
)
ASSISTANT_DRAFTS_STATUS_VALUES = frozenset(
    {"pending", "sent", "rejected", "blocked", "dry_run"}
)


class AnalystRoleMemory(Table):
    """Per-role memory: private to a role, a team proposal or a shared proposal."""

    name = "analyst_role_memory"
    columns = (
        "memory_id",
        "updated_at",
        "scope",
        "namespace",
        "kind",
        "subject",
        "body",
        "evidence_ids",
        "confidence",
        "status",
        "proposed_by",
        "confirmed_by",
        "author_role",
        "author_family",
        "expires_at",
        "props",
    )
    required = (
        "memory_id",
        "updated_at",
        "scope",
        "namespace",
        "subject",
        "proposed_by",
    )
    enums = (
        ("scope", ANALYST_ROLE_MEMORY_SCOPE_VALUES),
        ("kind", ANALYST_ROLE_MEMORY_KIND_VALUES),
        ("status", ANALYST_ROLE_MEMORY_STATUS_VALUES),
    )


class AnalystProjectFiles(Table):
    """A role's bounded project space."""

    name = "analyst_project_files"
    columns = (
        "role",
        "project",
        "name",
        "kind",
        "text",
        "updated_at",
        "deleted",
        "props",
    )
    required = ("role", "project", "name", "updated_at")
    enums = (("kind", ANALYST_PROJECT_FILES_KIND_VALUES),)


class AnalystOutcomes(Table):
    """How a role's answer was judged, by a person or an independent validator, never by the judged role."""

    name = "analyst_outcomes"
    columns = (
        "outcome_id",
        "recorded_at",
        "role",
        "incident_id",
        "turn_ref",
        "outcome",
        "judged_by",
        "judged_kind",
        "judged_family",
        "lesson_id",
        "note",
        "props",
    )
    required = ("outcome_id", "recorded_at", "role", "judged_by")
    enums = (
        ("outcome", ANALYST_OUTCOMES_OUTCOME_VALUES),
        ("judged_kind", ANALYST_OUTCOMES_JUDGED_KIND_VALUES),
    )


class AnalystProfileChanges(Table):
    """A proposed change to a role's profile, backed by recorded outcomes."""

    name = "analyst_profile_changes"
    columns = (
        "change_id",
        "created_at",
        "proposed_by",
        "target_role",
        "target",
        "text",
        "evidence_outcomes",
        "status",
        "decided_by",
        "props",
    )
    required = ("change_id", "created_at", "proposed_by", "target_role")


class AnalystWorkItems(Table):
    """The orchestrator's durable work items; the newest row per item_id is the item."""

    name = "analyst_work_items"
    columns = (
        "item_id",
        "updated_at",
        "started_at",
        "status",
        "reason",
        "not_before",
        "step_count",
        "escalations",
        "waits",
        "actions_proposed",
        "invalid_streak",
        "error_streak",
        "stale_cycles",
        "stall_escalated",
        "last_progress_cycle",
        "digest",
        "observations",
        "routed",
        "last_rejection",
        "alias_override",
        "inc_digest",
        "base_digest",
        "base_cycle",
        "mode_note",
        "props",
    )
    required = ("item_id", "updated_at", "started_at")
    enums = (("status", ANALYST_WORK_ITEMS_STATUS_VALUES),)


class AnalystOrchSteps(Table):
    """One row per orchestrator step, carrying the item as it stands afterwards (the checkpoint a restart recovers from)."""

    name = "analyst_orch_steps"
    columns = (
        "item_id",
        "step_no",
        "ts",
        "cycle_no",
        "kind",
        "outcome",
        "detail",
        "proposal_hash",
        "model_alias",
        "tokens",
        "item_after",
        "props",
    )
    required = ("item_id", "step_no", "ts")


class AnalystOrchEvents(Table):
    """Orchestrator events: cycles, stalls, dead-man alerts, role lifecycle."""

    name = "analyst_orch_events"
    columns = (
        "ts",
        "kind",
        "cycle_no",
        "reason",
        "detail",
        "actor",
        "props",
    )
    required = ("ts", "kind")


class AnalystRoleLifecycle(Table):
    """Created-role lifecycle: proposed, reviewed, drafted, promoted, retired."""

    name = "analyst_role_lifecycle"
    columns = (
        "name",
        "updated_at",
        "state",
        "actor",
        "action",
        "detail",
        "spec",
        "expires_at",
        "props",
    )
    required = ("name", "updated_at", "state", "action")


class AnalystAgentCalls(Table):
    """Every agent-to-agent call, refused or not."""

    name = "analyst_agent_calls"
    columns = (
        "ts",
        "incident_id",
        "kind",
        "caller",
        "callee",
        "request_type",
        "request_hash",
        "outcome",
        "reason",
        "depth",
        "tokens",
        "latency_ms",
        "stance",
        "confidence",
        "answer_hash",
        "cost_class",
        "props",
    )
    required = ("ts", "kind", "caller", "callee")
    enums = (
        ("kind", ANALYST_AGENT_CALLS_KIND_VALUES),
        ("outcome", ANALYST_AGENT_CALLS_OUTCOME_VALUES),
        ("stance", ANALYST_AGENT_CALLS_STANCE_VALUES),
    )


class AccessDecisions(Table):
    """The information-access gate's decision log."""

    name = "access_decisions"
    columns = (
        "ts",
        "run_id",
        "seq",
        "row_kind",
        "sender",
        "on_behalf_of",
        "dest_kind",
        "recipients",
        "label",
        "outcome",
        "code",
        "digest",
        "rule",
        "approved_by",
        "props",
    )
    required = ("ts", "run_id", "seq", "sender")
    enums = (
        ("row_kind", ACCESS_DECISIONS_ROW_KIND_VALUES),
        ("outcome", ACCESS_DECISIONS_OUTCOME_VALUES),
    )


class RecoveryActions(Table):
    """The recovery agent's audit: an action row per state change (newest per action_id wins), refusals, and signature demotions."""

    name = "recovery_actions"
    columns = (
        "action_id",
        "updated_at",
        "row_kind",
        "mode",
        "actuator",
        "host",
        "repo",
        "pr",
        "head_sha",
        "run_id",
        "job",
        "signature",
        "cause_class",
        "idem_key",
        "attempt",
        "status",
        "outcome",
        "outcome_ts",
        "reason",
        "evidence_url",
        "incident_id",
        "detail",
        "props",
    )
    required = ("action_id", "updated_at", "row_kind")
    enums = (
        ("row_kind", RECOVERY_ACTIONS_ROW_KIND_VALUES),
        ("mode", RECOVERY_ACTIONS_MODE_VALUES),
    )


class AnalystOptimizerRecords(Table):
    """What the optimizer role writes: runs, audit steps, suggestions, PR plans and the outcome of past suggestions."""

    name = "analyst_optimizer_records"
    columns = (
        "ts",
        "kind",
        "fingerprint",
        "doc",
        "props",
    )
    required = ("kind", "doc")
    enums = (("kind", ANALYST_OPTIMIZER_RECORDS_KIND_VALUES),)


class GoalScores(Table):
    """A scored review cycle: one row per key result and one per role."""

    name = "goal_scores"
    columns = (
        "cycle",
        "scored_at",
        "row_kind",
        "subject",
        "owner",
        "value",
        "n",
        "raw",
        "score",
        "counter_value",
        "breached",
        "coverage",
        "flags",
        "frozen",
        "krs",
        "props",
    )
    required = ("cycle", "scored_at", "row_kind", "subject")
    enums = (("row_kind", GOAL_SCORES_ROW_KIND_VALUES),)


class GoalReviews(Table):
    """A reviewer model's review of one role for one cycle."""

    name = "goal_reviews"
    columns = (
        "review_id",
        "cycle",
        "role",
        "reviewer_role",
        "reviewer_model",
        "reviewer_family",
        "created_at",
        "rating",
        "score",
        "strengths",
        "gaps",
        "status",
        "flags",
        "tokens",
        "props",
    )
    required = ("review_id", "cycle", "role", "created_at")
    enums = (("status", GOAL_REVIEWS_STATUS_VALUES),)


class GoalProposals(Table):
    """Proposals out of a review."""

    name = "goal_proposals"
    columns = (
        "proposal_id",
        "created_at",
        "cycle",
        "proposed_by",
        "proposer_kind",
        "target_role",
        "type",
        "target",
        "text",
        "review_id",
        "status",
        "decided_by",
        "requires_human",
        "detail",
        "props",
    )
    required = ("proposal_id", "created_at", "proposed_by", "target_role")
    enums = (("status", GOAL_PROPOSALS_STATUS_VALUES),)


class GoalComments(Table):
    """A person's comment on a review."""

    name = "goal_comments"
    columns = (
        "comment_id",
        "created_at",
        "review_id",
        "author",
        "verdict",
        "text",
        "props",
    )
    required = ("comment_id", "created_at", "review_id", "author")


class GoalAudit(Table):
    """Who did what in the goals layer."""

    name = "goal_audit"
    columns = (
        "ts",
        "actor",
        "action",
        "ref",
        "detail",
        "props",
    )
    required = ("ts", "actor", "action")


class AssistantProfiles(Table):
    """A person's opted-in assistant."""

    name = "assistant_profiles"
    columns = (
        "person",
        "display",
        "state",
        "card",
        "live_adapters",
        "backup_person",
        "allowed_targets",
        "created_at",
        "updated_at",
        "props",
    )
    required = ("person", "updated_at")
    enums = (("state", ASSISTANT_PROFILES_STATE_VALUES),)


class AssistantPrivateMemory(Table):
    """What an assistant knows about its person."""

    name = "assistant_private_memory"
    columns = (
        "person",
        "preferences",
        "working_hours",
        "focus",
        "owned_areas",
        "updated_at",
        "props",
    )
    required = ("person", "updated_at")


class AssistantInbox(Table):
    """Inbound items for a person's assistant; the item and its triage status in one row."""

    name = "assistant_inbox"
    columns = (
        "item_id",
        "recipient",
        "source",
        "sender",
        "subject",
        "body",
        "links",
        "urgency",
        "incident_ids",
        "respond_by",
        "thread",
        "received_at",
        "status",
        "updated_at",
        "props",
    )
    required = ("item_id", "recipient", "received_at", "updated_at")
    enums = (("status", ASSISTANT_INBOX_STATUS_VALUES),)


class AssistantDrafts(Table):
    """Drafts awaiting a person's approval, and what became of them."""

    name = "assistant_drafts"
    columns = (
        "draft_id",
        "person",
        "item_id",
        "kind",
        "adapter",
        "target",
        "thread",
        "text",
        "status",
        "urgent",
        "payload",
        "created_at",
        "updated_at",
        "props",
    )
    required = ("draft_id", "person", "updated_at")
    enums = (("status", ASSISTANT_DRAFTS_STATUS_VALUES),)


class AssistantCommitments(Table):
    """Commitments and maintenance windows an assistant reminds its person about."""

    name = "assistant_commitments"
    columns = (
        "commitment_id",
        "person",
        "text",
        "due_at",
        "kind",
        "lead_minutes",
        "reminded",
        "done",
        "updated_at",
        "props",
    )
    required = ("commitment_id", "person", "due_at", "updated_at")


class AssistantQueue(Table):
    """Messages held for a person's next digest."""

    name = "assistant_queue"
    columns = (
        "queue_id",
        "person",
        "text",
        "queued_at",
        "drained",
        "updated_at",
        "props",
    )
    required = ("queue_id", "person", "queued_at", "updated_at")


class AssistantAudit(Table):
    """Assistant lifecycle and send audit."""

    name = "assistant_audit"
    columns = (
        "ts",
        "actor",
        "person",
        "event",
        "ref",
        "detail",
        "props",
    )
    required = ("ts", "actor", "person", "event")


GRANT_EVENT_VALUES = frozenset({"created", "used", "denied", "revoked", "expired"})
GRANT_RESULT_VALUES = frozenset({"ok", "denied", "error"})


class DelegationGrants(Table):
    """One consented grant; written once, ended by an event or by expires_at, never updated."""

    name = "delegation_grants"
    columns = (
        "grant_id",
        "grantor",
        "grantee",
        "tools",
        "scope_json",
        "expires_at",
        "max_uses",
        "condition",
        "created_at",
        "created_by",
        "consent_text_hash",
        "props",
    )
    required = (
        "grant_id",
        "grantor",
        "grantee",
        "expires_at",
        "created_at",
        "created_by",
        "consent_text_hash",
    )

    @classmethod
    def row(cls, values):
        out = super().row(values)
        if values["grantor"] == values["grantee"]:
            raise SchemaError(f"{cls.name}: a grant cannot be to its own grantor")
        if len(str(values["consent_text_hash"])) != 64:
            raise SchemaError(
                f"{cls.name}: consent_text_hash must be a sha256 hex digest"
            )
        if not values["tools"]:
            raise SchemaError(f"{cls.name}: a grant needs at least one tool")
        return out


class DelegationGrantEvents(Table):
    """Everything done under a grant: creation, each use and refusal, revocation. The audit trail."""

    name = "delegation_grant_events"
    columns = (
        "grant_id",
        "ts",
        "event",
        "actor",
        "tool",
        "args_hash",
        "result",
        "on_behalf_of",
        "props",
    )
    required = ("grant_id", "event", "actor", "result")
    enums = (("event", GRANT_EVENT_VALUES), ("result", GRANT_RESULT_VALUES))


STATUS_STATE_VALUES = frozenset({"ok", "degraded", "failing", "unknown"})
STATUS_UPLOAD_PATH_VALUES = frozenset({"primary", "buffered"})
ANNOTATION_KIND_VALUES = frozenset({"human", "agent", "system"})
FETCH_OUTCOME_VALUES = frozenset(
    {"ok", "unreachable", "http_error", "bad_shape", "not_modified"}
)
FETCH_INDICATOR_VALUES = frozenset(
    {"", "none", "minor", "major", "critical", "maintenance", "unknown"}
)
EXTERNAL_KIND_VALUES = frozenset({"incident", "maintenance", "degradation", "notice"})
EXTERNAL_IMPACT_VALUES = frozenset(
    {"none", "minor", "major", "critical", "maintenance", "unknown"}
)
EXTERNAL_STATUS_VALUES = frozenset(
    {
        "scheduled",
        "in_progress",
        "investigating",
        "identified",
        "monitoring",
        "resolved",
        "completed",
        "cancelled",
    }
)
REMINDER_THRESHOLD_VALUES = frozenset({"7d", "1d", "1h", "cancelled"})
FEEDBACK_TARGET_VALUES = frozenset(
    {"signature_match", "incident", "status_post", "component", "answer", "failure"}
)
FEEDBACK_SIGNAL_VALUES = frozenset(
    {"up", "down", "affected", "not_affected", "miss", "chip"}
)
FEEDBACK_SOURCE_VALUES = frozenset({"explicit", "implicit", "reaction"})
MAX_STATUS_DETAIL = 500
MAX_ANNOTATION_TEXT = 2000
MAX_EXTERNAL_TITLE = 200
MAX_FEEDBACK_DETAIL = 500
BUCKET_SECONDS = 300


def status_sample_id(vantage: str, component: str, probe: str, ts) -> str:
    """The idempotency key of a probe: the same probe in the same 5-minute bucket repeats it."""
    epoch = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(ts)
    return f"{vantage}|{component}|{probe}|{epoch - epoch % BUCKET_SECONDS}"


class StatusSamples(Table):
    """One probe of one component from one vantage; a retry or a late upload repeats sample_id."""

    name = "status_samples"
    columns = (
        "sample_id",
        "ts",
        "vantage",
        "component",
        "probe",
        "state",
        "http_status",
        "latency_ms",
        "error_class",
        "detail",
        "ingested_at",
        "upload_path",
        "props",
    )
    required = ("sample_id", "ts", "vantage", "component", "probe")
    enums = (
        ("state", STATUS_STATE_VALUES),
        ("upload_path", STATUS_UPLOAD_PATH_VALUES),
    )

    @classmethod
    def row(cls, values):
        out = super().row(values)
        if len(str(values["detail"])) > MAX_STATUS_DETAIL:
            raise SchemaError(f"{cls.name}: detail is over {MAX_STATUS_DETAIL} chars")
        return out


class StatusAnnotations(Table):
    """A note layered over the computed state; an edit supersedes, and never changes the state."""

    name = "status_annotations"
    columns = (
        "annotation_id",
        "ts",
        "component",
        "kind",
        "author",
        "text",
        "expires_at",
        "supersedes",
        "props",
    )
    required = ("annotation_id", "ts", "author")
    enums = (("kind", ANNOTATION_KIND_VALUES),)

    @classmethod
    def row(cls, values):
        out = super().row(values)
        if len(str(values["text"])) > MAX_ANNOTATION_TEXT:
            raise SchemaError(f"{cls.name}: text is over {MAX_ANNOTATION_TEXT} chars")
        if values["supersedes"] == values["annotation_id"]:
            raise SchemaError(f"{cls.name}: a note cannot supersede itself")
        return out


class StatusLeases(Table):
    """A named lease; the newest heartbeat per lease_name wins and a passed expires_at frees it."""

    name = "status_leases"
    columns = ("lease_name", "holder", "heartbeat_at", "expires_at", "props")
    required = ("lease_name", "holder", "heartbeat_at", "expires_at")

    @classmethod
    def row(cls, values):
        out = super().row(values)
        if values["expires_at"] <= values["heartbeat_at"]:
            raise SchemaError(f"{cls.name}: expires_at must be after heartbeat_at")
        return out


class ExternalFeedFetches(Table):
    """One fetch of one status feed from one vantage; a failed fetch is a row too."""

    name = "external_feed_fetches"
    columns = (
        "fetched_at",
        "vantage",
        "source",
        "feed_url",
        "outcome",
        "failed_phase",
        "http_status",
        "latency_ms",
        "bytes",
        "body_hash",
        "page_indicator",
        "job_run",
        "props",
    )
    required = ("vantage", "source", "feed_url")
    enums = (
        ("outcome", FETCH_OUTCOME_VALUES),
        ("page_indicator", FETCH_INDICATOR_VALUES),
    )


class ExternalEvents(Table):
    """One observation of one external event by one vantage; event_id is stable across both."""

    name = "external_events"
    columns = (
        "observed_at",
        "event_id",
        "vantage",
        "source",
        "component",
        "upstream_component",
        "kind",
        "impact",
        "status",
        "starts_at",
        "ends_at",
        "title",
        "url",
        "content_hash",
        "upstream_updated_at",
        "props",
    )
    required = ("event_id", "vantage", "source", "title")
    enums = (
        ("kind", EXTERNAL_KIND_VALUES),
        ("impact", EXTERNAL_IMPACT_VALUES),
        ("status", EXTERNAL_STATUS_VALUES),
    )

    @classmethod
    def row(cls, values):
        out = super().row(values)
        if len(str(values["title"])) > MAX_EXTERNAL_TITLE:
            raise SchemaError(f"{cls.name}: title is over {MAX_EXTERNAL_TITLE} chars")
        start, end = values["starts_at"], values["ends_at"]
        if start is not None and end is not None and end < start:
            raise SchemaError(f"{cls.name}: ends_at is before starts_at")
        return out


class ExternalReminders(Table):
    """A maintenance reminder that went out; written after the send, keyed so it is sent once."""

    name = "external_reminders"
    columns = (
        "sent_at",
        "reminder_key",
        "event_id",
        "threshold",
        "window_starts_at",
        "recipients",
        "channel",
        "message_ref",
        "sent_by_vantage",
        "props",
    )
    required = ("reminder_key", "event_id", "window_starts_at")
    enums = (("threshold", REMINDER_THRESHOLD_VALUES),)


class FeedbackSignals(Table):
    """One person's (or one behaviour's) signal on one target; evidence, never a label by itself."""

    name = "feedback_signals"
    columns = (
        "created_at",
        "target_type",
        "target_id",
        "signal",
        "detail",
        "source",
        "actor",
        "weight_hint",
        "signature_id",
        "signature_version",
        "props",
    )
    required = ("target_id", "actor")
    enums = (
        ("target_type", FEEDBACK_TARGET_VALUES),
        ("signal", FEEDBACK_SIGNAL_VALUES),
        ("source", FEEDBACK_SOURCE_VALUES),
    )

    @classmethod
    def row(cls, values):
        out = super().row(values)
        if len(str(values["detail"])) > MAX_FEEDBACK_DETAIL:
            raise SchemaError(f"{cls.name}: detail is over {MAX_FEEDBACK_DETAIL} chars")
        if float(values["weight_hint"]) < 0:
            raise SchemaError(f"{cls.name}: weight_hint is negative")
        if values["target_type"] == "signature_match" and not values["signature_id"]:
            raise SchemaError(
                f"{cls.name}: a signature_match signal names its signature"
            )
        return out


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
ARTIFACT_RESULT_REASONS = ArtifactResultReasons
PIPELINE_RUNS = PipelineRuns
CI_RUN_TIMINGS = CiRunTimings
ARTIFACT_SUBSCRIPTIONS = ArtifactSubscriptions
DISPATCH_REQUESTS = DispatchRequests
ARTIFACT_DISPATCHES = ArtifactDispatches
DISPATCH_TARGETS = DispatchTargets
CREDENTIAL_INVENTORY = CredentialInventory
CREDENTIAL_PROBES = CredentialProbes
CREDENTIAL_OWNERS = CredentialOwners
INCIDENT_EVENTS = IncidentEvents
INCIDENT_LOG = IncidentLog
ANALYST_ASSIGNMENTS = AnalystAssignments
ANALYST_SESSIONS = AnalystSessions
ANALYST_TURNS = AnalystTurns
ANALYST_MEMORY = AnalystMemory
CASE_FILES = CaseFiles
ANALYST_ROLE_MEMORY = AnalystRoleMemory
ANALYST_PROJECT_FILES = AnalystProjectFiles
ANALYST_OUTCOMES = AnalystOutcomes
ANALYST_PROFILE_CHANGES = AnalystProfileChanges
ANALYST_WORK_ITEMS = AnalystWorkItems
ANALYST_ORCH_STEPS = AnalystOrchSteps
ANALYST_ORCH_EVENTS = AnalystOrchEvents
ANALYST_ROLE_LIFECYCLE = AnalystRoleLifecycle
ANALYST_AGENT_CALLS = AnalystAgentCalls
ACCESS_DECISIONS = AccessDecisions
RECOVERY_ACTIONS = RecoveryActions
ANALYST_OPTIMIZER_RECORDS = AnalystOptimizerRecords
GOAL_SCORES = GoalScores
GOAL_REVIEWS = GoalReviews
GOAL_PROPOSALS = GoalProposals
GOAL_COMMENTS = GoalComments
GOAL_AUDIT = GoalAudit
ASSISTANT_PROFILES = AssistantProfiles
ASSISTANT_PRIVATE_MEMORY = AssistantPrivateMemory
ASSISTANT_INBOX = AssistantInbox
ASSISTANT_DRAFTS = AssistantDrafts
ASSISTANT_COMMITMENTS = AssistantCommitments
ASSISTANT_QUEUE = AssistantQueue
ASSISTANT_AUDIT = AssistantAudit
DELEGATION_GRANTS = DelegationGrants
DELEGATION_GRANT_EVENTS = DelegationGrantEvents
STATUS_SAMPLES = StatusSamples
STATUS_ANNOTATIONS = StatusAnnotations
STATUS_LEASES = StatusLeases
EXTERNAL_FEED_FETCHES = ExternalFeedFetches
EXTERNAL_EVENTS = ExternalEvents
EXTERNAL_REMINDERS = ExternalReminders
FEEDBACK_SIGNALS = FeedbackSignals

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
        ArtifactResultReasons,
        Capabilities,
        CapabilityRuns,
        PipelineRuns,
        CiRunTimings,
        ArtifactSubscriptions,
        DispatchRequests,
        ArtifactDispatches,
        DispatchTargets,
        CredentialInventory,
        CredentialProbes,
        CredentialOwners,
        IncidentEvents,
        IncidentLog,
        AnalystAssignments,
        AnalystSessions,
        AnalystTurns,
        AnalystMemory,
        CaseFiles,
        AnalystRoleMemory,
        AnalystProjectFiles,
        AnalystOutcomes,
        AnalystProfileChanges,
        AnalystWorkItems,
        AnalystOrchSteps,
        AnalystOrchEvents,
        AnalystRoleLifecycle,
        AnalystAgentCalls,
        AccessDecisions,
        RecoveryActions,
        AnalystOptimizerRecords,
        GoalScores,
        GoalReviews,
        GoalProposals,
        GoalComments,
        GoalAudit,
        AssistantProfiles,
        AssistantPrivateMemory,
        AssistantInbox,
        AssistantDrafts,
        AssistantCommitments,
        AssistantQueue,
        AssistantAudit,
        DelegationGrants,
        DelegationGrantEvents,
        StatusSamples,
        StatusAnnotations,
        StatusLeases,
        ExternalFeedFetches,
        ExternalEvents,
        ExternalReminders,
        FeedbackSignals,
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
