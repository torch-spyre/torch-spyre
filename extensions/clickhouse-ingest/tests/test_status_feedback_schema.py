"""status_*, external_* and feedback_* tables: models match the DDL, bad rows are refused, views avoid the known pitfalls."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import regex as re

from spyre_clickhouse_ingest import schema
from spyre_clickhouse_ingest.schema import (
    EXTERNAL_EVENTS,
    EXTERNAL_FEED_FETCHES,
    EXTERNAL_REMINDERS,
    FEEDBACK_SIGNALS,
    STATUS_ANNOTATIONS,
    STATUS_SAMPLES,
    SchemaError,
    TABLES,
    status_sample_id,
)

SCHEMA = Path(__file__).resolve().parents[1] / "schema"
FILES = {
    "status_samples": "31-status.sql",
    "status_annotations": "31-status.sql",
    "external_feed_fetches": "32-external-feeds.sql",
    "external_events": "32-external-feeds.sql",
    "external_reminders": "32-external-feeds.sql",
    "feedback_signals": "33-feedback.sql",
}
VIEW_FILES = (
    "65-status-views.sql",
    "66-external-feed-views.sql",
    "67-feedback-views.sql",
)
VIEWS = {f: (SCHEMA / f).read_text() for f in VIEW_FILES}
MODELS = [
    STATUS_SAMPLES,
    STATUS_ANNOTATIONS,
    EXTERNAL_FEED_FETCHES,
    EXTERNAL_EVENTS,
    EXTERNAL_REMINDERS,
    FEEDBACK_SIGNALS,
]
T0 = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _block(table: str) -> str:
    text = (SCHEMA / FILES[table]).read_text()
    block = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\s*\nENGINE", text, re.S
    ).group(1)
    return block


def _ddl_columns(table: str) -> list[str]:
    cols, in_constraint = [], False
    for line in _block(table).splitlines():
        line = line.split("--")[0].strip()
        if not line:
            continue
        if line.startswith("CONSTRAINT"):
            in_constraint = not line.rstrip(",").endswith(")")
            continue
        if in_constraint:
            in_constraint = not line.rstrip(",").endswith(")")
            continue
        cols.append(line.split()[0])
    return cols


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_model_columns_are_the_ddl_columns(model):
    # `bucket` is MATERIALIZED from ts, so it is not inserted.
    server = {"audit_uuid", "audit_timestamp", "bucket"}
    assert [c for c in _ddl_columns(model.name) if c not in server] == list(
        model.columns
    )
    assert model.name in TABLES


def _fill(model, base, over):
    row = {c: "" for c in model.columns}
    row.update(props={}, **base)
    row.update(over)
    return row


def _sample(**over):
    return _fill(
        STATUS_SAMPLES,
        dict(
            sample_id="vm|jenkins|http_ping|1",
            ts=T0,
            vantage="build-vm",
            component="jenkins",
            probe="http_ping",
            state="ok",
            http_status=200,
            latency_ms=12,
            ingested_at=T0,
            upload_path="primary",
        ),
        over,
    )


def _note(**over):
    return _fill(
        STATUS_ANNOTATIONS,
        dict(
            annotation_id="a1",
            ts=T0,
            component="jenkins",
            kind="human",
            author="ashok@example.com",
            text="restarting the controller",
            expires_at=None,
            supersedes="",
        ),
        over,
    )


def _fetch(**over):
    return _fill(
        EXTERNAL_FEED_FETCHES,
        dict(
            fetched_at=T0,
            vantage="laptop",
            source="githubstatus",
            feed_url="https://www.githubstatus.com/api/v2/summary.json",
            outcome="ok",
            page_indicator="none",
            http_status=200,
            latency_ms=100,
            bytes=1000,
        ),
        over,
    )


def _event(**over):
    return _fill(
        EXTERNAL_EVENTS,
        dict(
            observed_at=T0,
            event_id="githubstatus:abc",
            vantage="laptop",
            source="githubstatus",
            component="github-com",
            kind="maintenance",
            impact="maintenance",
            status="scheduled",
            starts_at=T0,
            ends_at=T0,
            title="Planned maintenance",
            upstream_updated_at=None,
        ),
        over,
    )


def _reminder(**over):
    return _fill(
        EXTERNAL_REMINDERS,
        dict(
            sent_at=T0,
            reminder_key="githubstatus:abc|1d|1",
            event_id="githubstatus:abc",
            threshold="1d",
            window_starts_at=T0,
            recipients=["ci-team@example.com"],
        ),
        over,
    )


def _signal(**over):
    return _fill(
        FEEDBACK_SIGNALS,
        dict(
            created_at=T0,
            target_type="signature_match",
            target_id="Spyre/orchestrator#12",
            signal="up",
            source="explicit",
            actor="ashok@example.com",
            weight_hint=1.0,
            signature_id="artifactory-401",
            signature_version="1",
        ),
        over,
    )


GOOD = [
    (STATUS_SAMPLES, _sample()),
    (STATUS_ANNOTATIONS, _note()),
    (EXTERNAL_FEED_FETCHES, _fetch()),
    (EXTERNAL_EVENTS, _event()),
    (EXTERNAL_REMINDERS, _reminder()),
    (FEEDBACK_SIGNALS, _signal()),
]


@pytest.mark.parametrize("model,row", GOOD, ids=lambda x: getattr(x, "name", ""))
def test_good_rows_are_ordered_by_columns(model, row):
    out = model.row(row)
    assert len(out) == len(model.columns)
    assert out == [row[c] for c in model.columns]


BAD = [
    (STATUS_SAMPLES, _sample(state="up"), "state"),
    (STATUS_SAMPLES, _sample(upload_path="later"), "upload_path"),
    (STATUS_SAMPLES, _sample(detail="x" * 501), "500"),
    (STATUS_SAMPLES, _sample(sample_id=""), "sample_id"),
    (STATUS_ANNOTATIONS, _note(kind="bot"), "kind"),
    (STATUS_ANNOTATIONS, _note(text="x" * 2001), "2000"),
    (STATUS_ANNOTATIONS, _note(supersedes="a1"), "supersede"),
    (EXTERNAL_FEED_FETCHES, _fetch(outcome="timeout"), "outcome"),
    (EXTERNAL_FEED_FETCHES, _fetch(page_indicator="severe"), "page_indicator"),
    (EXTERNAL_EVENTS, _event(kind="outage"), "kind"),
    (EXTERNAL_EVENTS, _event(status="over"), "status"),
    (EXTERNAL_EVENTS, _event(impact="huge"), "impact"),
    (EXTERNAL_EVENTS, _event(title="t" * 201), "200"),
    (EXTERNAL_EVENTS, _event(title=""), "title"),
    (
        EXTERNAL_EVENTS,
        _event(starts_at=T0, ends_at=datetime(2026, 10, 24, 11, tzinfo=timezone.utc)),
        "ends_at",
    ),
    (EXTERNAL_REMINDERS, _reminder(threshold="2h"), "threshold"),
    (EXTERNAL_REMINDERS, _reminder(reminder_key=""), "reminder_key"),
    (FEEDBACK_SIGNALS, _signal(target_type="repo"), "target_type"),
    (FEEDBACK_SIGNALS, _signal(signal="love"), "signal"),
    (FEEDBACK_SIGNALS, _signal(source="bot"), "source"),
    (FEEDBACK_SIGNALS, _signal(detail="d" * 501), "500"),
    (FEEDBACK_SIGNALS, _signal(weight_hint=-1.0), "weight"),
    (FEEDBACK_SIGNALS, _signal(signature_id=""), "signature"),
    (FEEDBACK_SIGNALS, _signal(actor=""), "actor"),
]


@pytest.mark.parametrize("model,row,why", BAD, ids=lambda x: str(x)[:24])
def test_rows_outside_the_ddl_checks_are_refused(model, row, why):
    with pytest.raises(SchemaError, match=why):
        model.row(row)


def test_a_signal_on_something_other_than_a_signature_needs_no_signature_id():
    FEEDBACK_SIGNALS.row(
        _signal(target_type="component", signature_id="", signal="affected")
    )


def test_an_open_ended_maintenance_window_is_allowed():
    EXTERNAL_EVENTS.row(_event(starts_at=T0, ends_at=None))
    EXTERNAL_EVENTS.row(_event(starts_at=None, ends_at=None))


def test_ddl_vocabularies_equal_the_models():
    checks = [
        ("status_samples", "state", schema.STATUS_STATE_VALUES),
        ("status_samples", "upload_path", schema.STATUS_UPLOAD_PATH_VALUES),
        ("status_annotations", "kind", schema.ANNOTATION_KIND_VALUES),
        ("external_feed_fetches", "outcome", schema.FETCH_OUTCOME_VALUES),
        ("external_feed_fetches", "page_indicator", schema.FETCH_INDICATOR_VALUES),
        ("external_events", "kind", schema.EXTERNAL_KIND_VALUES),
        ("external_events", "impact", schema.EXTERNAL_IMPACT_VALUES),
        ("external_events", "status", schema.EXTERNAL_STATUS_VALUES),
        ("external_reminders", "threshold", schema.REMINDER_THRESHOLD_VALUES),
        ("feedback_signals", "target_type", schema.FEEDBACK_TARGET_VALUES),
        ("feedback_signals", "signal", schema.FEEDBACK_SIGNAL_VALUES),
        ("feedback_signals", "source", schema.FEEDBACK_SOURCE_VALUES),
    ]
    for table, col, allowed in checks:
        listed = re.search(rf"CHECK {col} IN\s*\(([^)]*)\)", _block(table)).group(1)
        assert set(re.findall(r"'([^']*)'", listed)) == set(allowed), (table, col)


def test_length_caps_in_the_ddl_equal_the_model_caps():
    pairs = [
        ("status_samples", "detail", schema.MAX_STATUS_DETAIL),
        ("status_annotations", "text", schema.MAX_ANNOTATION_TEXT),
        ("external_events", "title", schema.MAX_EXTERNAL_TITLE),
        ("feedback_signals", "detail", schema.MAX_FEEDBACK_DETAIL),
    ]
    for table, col, cap in pairs:
        assert f"length({col}) <= {cap}" in _block(table), (table, col)


def test_sample_id_repeats_inside_a_bucket_and_changes_across_buckets():
    a = status_sample_id(
        "vm",
        "jenkins",
        "http_ping",
        datetime(2026, 10, 24, 12, 1, 5, tzinfo=timezone.utc),
    )
    b = status_sample_id(
        "vm",
        "jenkins",
        "http_ping",
        datetime(2026, 10, 24, 12, 4, 59, tzinfo=timezone.utc),
    )
    c = status_sample_id(
        "vm",
        "jenkins",
        "http_ping",
        datetime(2026, 10, 24, 12, 5, 0, tzinfo=timezone.utc),
    )
    assert a == b != c
    assert a.startswith("vm|jenkins|http_ping|")
    assert status_sample_id("laptop", "jenkins", "http_ping", T0) != status_sample_id(
        "vm", "jenkins", "http_ping", T0
    )


def test_samples_dedupe_by_id_keep_the_original_time_and_expire():
    text = (SCHEMA / "31-status.sql").read_text()
    block = text[
        text.index("EXISTS status_samples") : text.index("EXISTS status_annotations")
    ]
    assert "ReplacingMergeTree(ingested_at)" in block
    assert "ORDER BY (component, vantage, bucket, sample_id)" in block
    assert "bucket       DateTime('UTC') MATERIALIZED toStartOfFiveMinutes(ts)" in block
    assert "TTL toDateTime(ts) + INTERVAL 400 DAY" in block


def test_the_append_only_tables_are_plain_mergetree_and_only_the_log_like_ones_expire():
    external = (SCHEMA / "32-external-feeds.sql").read_text()
    assert external.count("TTL") == 3
    assert (SCHEMA / "33-feedback.sql").read_text().count("TTL") == 0
    assert "ReplacingMergeTree" not in external


@pytest.mark.parametrize("name", VIEW_FILES)
def test_view_aggregates_are_not_aliased_to_their_source_columns(name):
    # An alias equal to a source column is substituted into the other aggregates (ILLEGAL_AGGREGATION).
    text = VIEWS[name]
    for src in (
        "ts",
        "state",
        "status",
        "impact",
        "kind",
        "starts_at",
        "ends_at",
        "title",
        "signal",
        "actor",
        "created_at",
    ):
        assert not re.search(
            rf"\b(?:argMax|max|min|any|anyLast)\w*\([^)]*\)\s+AS {src}\b", text
        ), (name, src)


def test_unset_times_are_null_not_the_epoch():
    health = VIEWS["66-external-feed-views.sql"]
    assert "toDateTime64(0, 3, 'UTC'), NULL" in health
    assert "toDateTime64(0, 3, 'UTC'), NULL" in VIEWS["65-status-views.sql"]


def test_ratios_are_guarded_against_division_by_zero():
    assert "if(known_buckets = 0, NULL" in VIEWS["65-status-views.sql"]
    assert "if(labelled_matches = 0, NULL" in VIEWS["67-feedback-views.sql"]


def test_quorum_order_unknown_then_operational_then_the_failure_cases():
    block = VIEWS["65-status-views.sql"]
    block = block[block.index("multiIf(") : block.index(") AS state")]
    lines = [ln.strip() for ln in block.splitlines()[1:] if ln.strip()]
    # each condition line, in order: nothing reported is unknown before anything else, a clean run is
    # operational before any failure is looked at, and the quorum arms follow
    assert lines[0].startswith("a.reporting = 0,") and lines[0].endswith("'unknown',")
    assert lines[1].startswith("a.failing = 0 AND a.slow = 0,") and lines[1].endswith(
        "'operational',"
    )
    assert lines[2].startswith("a.failing = 0,") and lines[2].endswith("'degraded',")
    assert lines[-1] == "'partial_outage'"
    for state in (
        "unknown",
        "operational",
        "degraded",
        "major_outage",
        "partial_outage",
    ):
        assert f"'{state}'" in block


def test_the_daily_down_rule_mirrors_the_current_quorum_rule():
    text = VIEWS["65-status-views.sql"]
    daily = text[text.index("EXISTS v_status_daily") :]
    # 2 vantages: down only when both fail. 3 or more: down at half or more. 1 vantage: only on the 3rd bucket in a row.
    assert (
        "(n_known = 2 AND n_fail = 2) OR (n_known >= 3 AND n_fail * 2 >= n_known)"
        in daily
    )
    assert "OR (one_fail AND one_fail_run = 3)" in daily
    assert "RANGE BETWEEN 600 PRECEDING AND CURRENT ROW" in daily
    current = text[
        text.index("EXISTS v_status_current") : text.index("EXISTS v_status_daily")
    ]
    assert "a.failing * 2 < a.reporting" in current
    assert "a.reporting = 2 AND a.failing = 1" in current


def test_a_major_outage_needs_two_buckets_and_a_single_vantage_three():
    current = VIEWS["65-status-views.sql"]
    current = current[
        current.index("EXISTS v_status_current") : current.index(
            "EXISTS v_status_daily"
        )
    ]
    assert "if(r.all_fail_pair, 'major_outage', 'partial_outage')" in current
    assert "if(r.one_fail_triple, 'partial_outage', 'degraded')" in current
    assert "(all0 AND all1) OR (all1 AND all2)" in current
    assert "(one0 AND one1 AND one2) OR (one1 AND one2 AND one3)" in current


def test_only_live_vantages_count_towards_the_quorum():
    text = VIEWS["65-status-views.sql"]
    assert text.count("AND v_alive") >= 5
    assert "AND vantage_alive" in text


def test_the_signature_key_keeps_two_signatures_on_one_failure_apart():
    text = VIEWS["67-feedback-views.sql"]
    assert "GROUP BY target_type, target_id, signature_id, actor, axis" in text
    assert "GROUP BY target_type, target_id, signature_id;" in text


def test_a_changed_mind_is_one_current_signal_because_up_and_down_share_an_axis():
    text = VIEWS["67-feedback-views.sql"]
    assert "multiIf(signal IN ('up', 'down'), 'vote'," in text
    assert "signal IN ('affected', 'not_affected'), 'impact'," in text
    assert "argMax(signal, created_at)            AS cur_signal" in text
    # the scores read the newest signal on an axis, never the raw signal column
    scores = text[text.index("EXISTS v_feedback_scores") :]
    assert not re.search(r"(?<!cur_)\bsignal\b\s*(IN|=)", scores)


def test_only_explicit_votes_and_reactions_are_verdicts_and_disputes_are_left_out():
    text = VIEWS["67-feedback-views.sql"]
    assert text.count("cur_source IN ('explicit', 'reaction')") >= 3
    assert "countIf(human_up > 0 AND human_down = 0)" in text
    assert "countIf(human_down > 0 AND human_up = 0)" in text
    assert "countIf(human_up > 0 AND human_down > 0)" in text
    assert "countIf(cur_source = 'implicit'" in text


def test_an_event_past_its_end_is_not_open_whatever_its_status():
    text = VIEWS["66-external-feed-views.sql"]
    assert "(cur_ends IS NULL OR cur_ends > now64(3)) AS is_open" in text


def test_only_the_view_files_define_views_and_only_the_table_files_define_tables():
    for name in VIEW_FILES:
        assert "CREATE TABLE" not in VIEWS[name]
    for f in set(FILES.values()):
        assert "CREATE VIEW" not in (SCHEMA / f).read_text()


def test_file_numbers_are_unique_and_tables_sort_before_their_views():
    numbers = [p.name.split("-")[0] for p in SCHEMA.glob("[0-9]*.sql")]
    assert len(numbers) == len(set(numbers))
    assert "31" < "65" and "32" < "66" and "33" < "67"
