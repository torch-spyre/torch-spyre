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

"""Runs schema/55's views on chdb: v_feed_events must match exactly what v_artifact_subscribers
matches for the same filters stored as a subscription, and seen_at / matched_at must order a
late verdict after the events that landed before it."""

import json

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier

session = pytest.importorskip("chdb.session")

A1 = "11111111-1111-5111-8111-111111111111"  # torch-spyre image, x86_64
A2 = "22222222-2222-5222-8222-222222222222"  # hf-adapters wheel, s390x
R_SMOKE = "aaaaaaaa-0000-5000-8000-000000000001"
R_INTEG = "aaaaaaaa-0000-5000-8000-000000000002"
R_A2 = "aaaaaaaa-0000-5000-8000-000000000003"
R_RETRACTED = "aaaaaaaa-0000-5000-8000-000000000004"
EVERY = [
    "artifact.tagged",
    "artifact.recorded",
    "artifact.promoted",
    "results.recorded",
]

# Filters as artifact_subscriptions columns; anything left out is "any".
CASES = {
    "any-tag": {},
    "family": {"tag_family": "nightly"},
    "pattern-anchored": {"tag_pattern": "nightly-.*"},
    "pattern-partial": {"tag_pattern": "ightly"},
    "props": {"tag_props": {"writer": "ci"}},
    "exclude-family": {"exclude_families": ["main"]},
    "exclude-pattern": {"exclude_tag_patterns": ["release-.*"]},
    "results-filtered": {
        "event_types": ["results.recorded"],
        "result_test_types": ["smoke"],
        "result_states": ["failed"],
    },
    "require-state": {"require": ["smoke=passed"]},
    "require-type": {"require": ["integration"]},
    "unless": {"unless_exists": ["integration=failed"]},
    "artifact-filters": {
        "event_types": EVERY,
        "component": "hf-adapters",
        "arch": "s390x",
        "kind": "wheel",
        "artifact_name": "hf-adapters",
    },
    "everything": {"event_types": EVERY},
}

PARAMS = {
    "event_types": ("Array(String)", ["artifact.tagged"]),
    "component": ("String", ""),
    "arch": ("String", ""),
    "kind": ("String", ""),
    "artifact_name": ("String", ""),
    "tag_family": ("String", ""),
    "tag_pattern": ("String", ""),
    "tag_props": ("Map(String, String)", {}),
    "exclude_families": ("Array(String)", []),
    "exclude_tag_patterns": ("Array(String)", []),
    "result_test_types": ("Array(String)", []),
    "result_states": ("Array(String)", []),
    "require": ("Array(String)", []),
    "unless_exists": ("Array(String)", []),
}
EVENT = "event_type, event_key, toString(artifact_id) AS artifact_id, match_reason, toString(matched_at) AS matched_at"


def literal(v):
    if isinstance(v, dict):
        return (
            "map("
            + ", ".join(f"{literal(k)}, {literal(x)}" for k, x in v.items())
            + ")"
        )
    if isinstance(v, list):
        return "[" + ", ".join(literal(x) for x in v) + "]"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


class Chdb:
    def __init__(self):
        self.s = session.Session()

    def command(self, sql):
        return str(self.s.query(sql, "TSV")).rstrip("\n")

    def rows(self, sql):
        return [
            json.loads(x) for x in str(self.s.query(sql, "JSONEachRow")).splitlines()
        ]

    def insert(self, table, rows):
        body = "\n".join(json.dumps(r) for r in rows)
        self.command(f"INSERT INTO {table} FORMAT JSONEachRow\n{body}")

    def subscribe(
        self, sid, enabled=True, not_before="2000-01-01 00:00:00.000", **filters
    ):
        self.insert(
            "artifact_subscriptions",
            [
                {
                    "subscription_id": sid,
                    "enabled": enabled,
                    "not_before": not_before,
                    "event_types": filters.pop("event_types", ["artifact.tagged"]),
                    "target_type": "webhook",
                    "target": "https://hooks.example.com/feed-test",
                    **filters,
                }
            ],
        )

    def subscribers(self, sid):
        return self.rows(
            f"SELECT {EVENT}, reason FROM v_artifact_subscribers "
            f"WHERE subscription_id = {literal(sid)} ORDER BY event_type, event_key, artifact_id"
        )

    def feed(self, filters):
        args = ", ".join(
            f"{k}={literal(filters.get(k, default))}"
            for k, (_, default) in PARAMS.items()
        )
        return self.rows(
            f"SELECT {EVENT} FROM v_feed_events({args}) "
            "ORDER BY event_type, event_key, artifact_id"
        )


@pytest.fixture(scope="module")
def db():
    c = Chdb()
    for name in (
        "10-functional-tests.sql",
        "20-artifacts.sql",
        "25-artifact-dispatch.sql",
        "26-dispatch-targets.sql",
        "49-result-uploads.sql",
        "50-artifact-views.sql",
        "55-dispatch-views.sql",
    ):
        for stmt in SchemaApplier.statements((SCHEMA_DIR / name).read_text()):
            c.command(stmt)
    c.insert(
        "artifacts",
        [
            {
                "ts": "2026-10-10 10:00:00",
                "audit_timestamp": "2026-10-10 10:00:00.000",
                "artifact_id": A1,
                "component": "torch-spyre",
                "arch": "x86_64",
                "kind": "image",
                "artifact_name": "torch-spyre-dev",
                "origin": "built",
            },
            {
                "ts": "2026-10-10 11:00:00",
                "audit_timestamp": "2026-10-10 11:00:00.000",
                "artifact_id": A2,
                "component": "hf-adapters",
                "arch": "s390x",
                "kind": "wheel",
                "artifact_name": "hf-adapters",
                "origin": "promoted",
            },
        ],
    )
    tag = lambda t, fam, aid, at, **props: {  # noqa: E731
        "ts": at,
        "audit_timestamp": at + ".000",
        "tag": t,
        "tag_family": fam,
        "artifact_id": aid,
        "props": props,
    }
    c.insert(
        "artifact_tags",
        [
            tag(
                "nightly-2026-10-10", "nightly", A1, "2026-10-10 12:00:00", writer="ci"
            ),
            # A re-insert of the same tag is not a new event: event_ts and seen_at stay the first.
            tag(
                "nightly-2026-10-10",
                "nightly",
                A1,
                "2026-10-10 18:00:00",
                writer="other",
            ),
            tag("main", "main", A1, "2026-10-10 12:05:00"),
            tag("release-1.0", "release", A2, "2026-10-10 13:00:00"),
        ],
    )
    result = lambda aid, run, tt, state, ts, landed: {  # noqa: E731
        "ts": ts,
        "audit_timestamp": landed,
        "artifact_id": aid,
        "run_id": run,
        "result_kind": "functional",
        "test_type": tt,
        "state": state,
        "arch": "x86_64",
        "duration_s": 1.0,
    }
    c.insert(
        "artifact_results",
        [
            # An upload backdated by two days lands after the tag events: the late verdict.
            result(
                A1,
                R_SMOKE,
                "smoke",
                "running",
                "2026-10-08 09:00:00",
                "2026-10-11 08:00:00.000",
            ),
            result(
                A1,
                R_SMOKE,
                "smoke",
                "passed",
                "2026-10-08 09:30:00",
                "2026-10-11 09:00:00.000",
            ),
            result(
                A1,
                R_INTEG,
                "integration",
                "failed",
                "2026-10-10 14:00:00",
                "2026-10-10 14:00:00.000",
            ),
            result(
                A2,
                R_A2,
                "smoke",
                "failed",
                "2026-10-10 15:00:00",
                "2026-10-10 15:00:00.000",
            ),
            result(
                A2,
                R_RETRACTED,
                "regression",
                "passed",
                "2026-10-10 16:00:00",
                "2026-10-10 16:00:00.000",
            ),
        ],
    )
    c.insert(
        "result_uploads",
        [
            {
                "ts": "2026-10-10 17:00:00.000",
                "upload_id": "bbbbbbbb-0000-7000-8000-000000000001",
                "state": "retracted",
                "actor": "someone@example.com",
                "artifact_id": A2,
                "run_id": R_RETRACTED,
                "format": "junit",
                "result_kind": "functional",
                "test_type": "regression",
                "arch": "s390x",
                "filename": "junit.xml",
                "size_bytes": 1,
                "content_sha256": "",
                "bundle_url": "",
            },
        ],
    )
    for sid, filters in CASES.items():
        c.subscribe(sid, **json.loads(json.dumps(filters)))
    yield c
    c.s.close()


@pytest.mark.parametrize("sid", sorted(CASES))
def test_feed_matches_exactly_what_a_stored_subscription_matches(db, sid):
    stored = [
        {k: v for k, v in r.items() if k != "reason"} for r in db.subscribers(sid)
    ]
    assert db.feed(CASES[sid]) == stored


@pytest.mark.parametrize(
    "sid, expected",
    [
        ("any-tag", {"nightly-2026-10-10", "main", "release-1.0"}),
        ("family", {"nightly-2026-10-10"}),
        ("pattern-anchored", {"nightly-2026-10-10"}),
        ("pattern-partial", set()),
        ("props", {"nightly-2026-10-10"}),
        ("results-filtered", {f"{R_A2}/smoke"}),
        ("artifact-filters", {"", "release-1.0", f"{R_A2}/smoke"}),
    ],
)
def test_the_cases_are_not_vacuous(db, sid, expected):
    assert {r["event_key"] for r in db.subscribers(sid)} == expected


def test_match_reason_names_what_stops_an_event_and_reason_adds_the_gates(db):
    by_key = lambda sid: {r["event_key"]: r for r in db.subscribers(sid)}  # noqa: E731
    assert by_key("exclude-family")["main"]["match_reason"] == "excluded"
    assert by_key("exclude-pattern")["release-1.0"]["match_reason"] == "excluded"
    assert (
        by_key("require-type")["release-1.0"]["match_reason"] == "waiting: integration"
    )
    assert by_key("unless")["main"]["match_reason"] == "unless: integration=failed"
    assert by_key("unless")["release-1.0"]["match_reason"] == ""
    db.subscribe("gated-off", enabled=False, exclude_families=["main"])
    db.subscribe(
        "gated-later", not_before="2030-01-01 00:00:00.000", exclude_families=["main"]
    )
    for sid, gate in (("gated-off", "disabled"), ("gated-later", "before not_before")):
        rows = by_key(sid)
        assert {r["reason"] for r in rows.values()} == {gate}
        assert rows["main"]["match_reason"] == "excluded"


def test_a_late_verdict_is_seen_when_it_landed_not_at_its_run_time(db):
    (row,) = db.rows(
        "SELECT toString(event_ts) AS ts, toString(seen_at) AS seen FROM v_artifact_events "
        f"WHERE event_type = 'results.recorded' AND event_key = '{R_SMOKE}/smoke'"
    )
    assert row == {"ts": "2026-10-08 09:30:00", "seen": "2026-10-11 09:00:00.000"}


def test_a_required_verdict_moves_matched_at_to_when_it_landed(db):
    rows = {r["event_key"]: r for r in db.subscribers("require-state")}
    # Tagged 10-10 12:00; smoke passed only landed 10-11 09:00, so a poller sees it then.
    assert rows["nightly-2026-10-10"]["matched_at"] == "2026-10-11 09:00:00.000"
    assert rows["nightly-2026-10-10"]["match_reason"] == ""
    assert rows["release-1.0"]["match_reason"] == "waiting: smoke=passed"
    plain = {r["event_key"]: r for r in db.subscribers("any-tag")}
    assert plain["nightly-2026-10-10"]["matched_at"] == "2026-10-10 12:00:00.000"


def test_a_retracted_upload_is_no_event(db):
    assert not db.rows(
        f"SELECT 1 FROM v_artifact_events WHERE event_key = '{R_RETRACTED}/regression'"
    )


def test_every_version_of_a_subscription_is_kept(db):
    db.subscribe("versioned", tag_family="nightly")
    db.subscribe("versioned", tag_family="weekly", updated_by="editor@example.com")
    db.command("OPTIMIZE TABLE artifact_subscriptions FINAL")
    rows = db.rows(
        "SELECT tag_family, updated_by FROM artifact_subscription_versions "
        "WHERE subscription_id = 'versioned' ORDER BY updated_at, tag_family"
    )
    assert rows == [
        {"tag_family": "nightly", "updated_by": ""},
        {"tag_family": "weekly", "updated_by": "editor@example.com"},
    ]
    (live,) = db.rows(
        "SELECT tag_family FROM artifact_subscriptions FINAL WHERE subscription_id = 'versioned'"
    )
    assert live == {"tag_family": "weekly"}


def test_dispatch_targets_reads_the_latest_publish_only(db):
    def publish(at, run, *lines):
        rows = [
            {
                "target_type": t,
                "target_prefix": p,
                "credential_id": c,
                "published_at": at,
                "job_run": run,
            }
            for t, p, c in lines
        ]
        db.insert("dispatch_targets", rows)

    def current():
        return db.rows(
            "SELECT target_type, target_prefix, credential_id, job_run FROM v_dispatch_targets "
            "ORDER BY target_type"
        )

    publish(
        "2026-10-10 10:00:00.000",
        "run-1",
        ("webhook", "https://hooks.example.com/", "example-webhook-token"),
    )
    publish(
        "2026-10-11 10:00:00.000",
        "run-2",
        ("jenkins", "Example/folder/", ""),
        ("gha_dispatch", "example/", "example-app"),
    )
    assert current() == [
        {
            "target_type": "gha_dispatch",
            "target_prefix": "example/",
            "credential_id": "example-app",
            "job_run": "run-2",
        },
        {
            "target_type": "jenkins",
            "target_prefix": "Example/folder/",
            "credential_id": "",
            "job_run": "run-2",
        },
    ]
    # An emptied list is one sentinel row: the view shows no targets, not the previous list.
    publish("2026-10-12 10:00:00.000", "run-3", ("", "", ""))
    assert current() == []
