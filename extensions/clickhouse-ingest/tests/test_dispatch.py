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

"""Pins the dispatch contract the spyre-frameworks dispatcher and v2Dispatch rely on: derived
ids, strict rendering, and models that match their DDL column for column."""

import pytest
import regex as re
from spyre_clickhouse_ingest import dispatch
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier
from spyre_clickhouse_ingest.dispatch import DispatchId, Template
from spyre_clickhouse_ingest.schema import (
    ARTIFACT_DISPATCHES,
    ARTIFACT_SUBSCRIPTIONS,
    DISPATCH_REQUESTS,
    SchemaError,
)

AID = "7001e238-d08f-54a3-91e3-0f3acd757ab4"
DIGEST = "sha256:62a7fc698601d02d02b3b6de285cb75fc280fcf1a937deef82f4d70403fa15c4"


def ddl_columns(table: str) -> list:
    """The column names of `table`'s CREATE, in order, from the schema directory."""
    for path, text in SchemaApplier.selected_files(SCHEMA_DIR):
        for o in SchemaApplier.objects(path, text):
            if o.name == table:
                body = o.sql.split("(", 1)[1]
                return [
                    m.group(1)
                    for m in re.finditer(r"^\s{4}(\w+)\s", body, flags=re.M)
                    if m.group(1) not in ("CONSTRAINT", "INDEX")
                ]
    raise AssertionError(f"{table} not in the schema")


@pytest.mark.parametrize(
    "model", [ARTIFACT_SUBSCRIPTIONS, DISPATCH_REQUESTS, ARTIFACT_DISPATCHES]
)
def test_model_columns_are_the_ddl_minus_audit(model):
    assert list(model.columns) == [
        c for c in ddl_columns(model.name) if c not in ("audit_uuid", "audit_timestamp")
    ]


TAGGED = "artifact.tagged"


def test_auto_id_is_stable_and_keyed_on_subscription_artifact_and_event():
    a = DispatchId.auto("stf-torchspyre", AID, TAGGED, "ci-cd-tech-preview-v1")
    assert a == DispatchId.auto("stf-torchspyre", AID, TAGGED, "ci-cd-tech-preview-v1")
    assert a != DispatchId.auto("stf-torchspyre", AID, TAGGED, "ci-cd-tech-preview-v2")
    assert a != DispatchId.auto(
        "stf-spyrebackend", AID, TAGGED, "ci-cd-tech-preview-v1"
    )
    assert a != DispatchId.auto(
        "stf-torchspyre", AID, "results.recorded", "ci-cd-tech-preview-v1"
    )


def test_request_id_never_collides_with_an_auto_id():
    assert DispatchId.request(AID) != DispatchId.auto("", AID, "", "")


def test_render_fills_the_artifact_fields():
    ctx = dispatch.context(
        {
            "artifact_id": AID,
            "tag": "ci-cd-tech-preview-v1",
            "component": "hf-adapters",
        },
        {
            "digest": DIGEST,
            "pull_spec": f"icr.io/x/hf-adapters-devel@{DIGEST}",
            "id12": "62a7fc698601",
        },
    )
    out = Template.render(
        {
            "IMAGE_DIGEST": "{digest_bare}",
            "ARTIFACT_ID": "{artifact_id}",
            "TEST_CADENCE": "weekly",
            "IMAGE": "{pull_spec}",
        },
        ctx,
    )
    assert out == {
        "IMAGE_DIGEST": DIGEST.removeprefix("sha256:"),
        "ARTIFACT_ID": AID,
        "TEST_CADENCE": "weekly",
        "IMAGE": f"icr.io/x/hf-adapters-devel@{DIGEST}",
    }


@pytest.mark.parametrize("template", ["{no_such_field}", "{digest}"])
def test_an_unknown_or_empty_placeholder_refuses_the_render(template):
    """An rpm has no digest: sending IMAGE_DIGEST='' would run the job against nothing."""
    with pytest.raises(KeyError):
        Template.render({"X": template}, dispatch.context({"artifact_id": AID}, {}))


def test_dispatch_row_fills_defaults_and_passes_the_model():
    row = dispatch.dispatch_row(
        dispatch_id=DispatchId.auto("s", AID, TAGGED, "t"),
        subscription_id="s",
        artifact_id=AID,
        requested_by="auto",
        state="queued",
        target_type="jenkins",
        target="Spyre-Test/testing/Jenkinsfile.torchspyre",
    )
    assert len(ARTIFACT_DISPATCHES.row(row)) == len(ARTIFACT_DISPATCHES.columns)


def test_dispatch_state_outside_the_ddl_check_is_refused():
    row = dispatch.dispatch_row(
        dispatch_id=AID,
        subscription_id="s",
        artifact_id=AID,
        requested_by="auto",
        state="running",
        target_type="jenkins",
        target="x",
    )
    with pytest.raises(SchemaError):
        ARTIFACT_DISPATCHES.row(row)


class FakeClient:
    def __init__(self, fits: int):
        self.fits, self.inserts = fits, []

    def query(self, sql, parameters=None):
        fits = self.fits

        class R:
            result_rows = [(fits,)]

        return R()

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows))


def test_request_refuses_a_subscription_the_artifact_does_not_fit():
    with pytest.raises(ValueError, match="does not match"):
        dispatch.request(FakeClient(0), "db", subscription_id="s", artifact_id=AID)


def test_request_writes_one_row_and_returns_its_dispatch_id():
    client = FakeClient(1)
    out = dispatch.request(
        client,
        "db",
        subscription_id="s",
        artifact_id=AID,
        params={"TEST_CADENCE": "nightly"},
        requester="https://jenkins/job/x/1/",
    )
    assert [t for t, _ in client.inserts] == ["dispatch_requests"]
    assert out["dispatch_id"] == DispatchId.request(out["request_id"])
    assert out["written"] is True


def test_a_dry_run_request_writes_nothing():
    client = FakeClient(1)
    assert (
        dispatch.request(
            client, "db", subscription_id="s", artifact_id=AID, dry_run=True
        )["written"]
        is False
    )
    assert client.inserts == []


def test_store_reads_return_utc_aware_datetimes():
    """A naive value re-inserted would be read as the agent's local time (IST: 5:30 off)."""
    from datetime import datetime, timezone

    class Client:
        def query(self, sql, parameters=None):
            class R:
                column_names = ("tag_ts", "tag")
                result_rows = [(datetime(2026, 9, 29, 14, 26, 32), "t")]

            return R()

    [row] = dispatch.DispatchStore(Client(), "db").events(datetime.now(timezone.utc))
    assert row["tag_ts"] == datetime(2026, 9, 29, 14, 26, 32, tzinfo=timezone.utc)


NOW = __import__("datetime").datetime(
    2026, 10, 9, 12, 0, tzinfo=__import__("datetime").timezone.utc
)


def event(minutes_ago, **over):
    from datetime import timedelta

    r = {
        "subscription_id": "s",
        "component": "hf-adapters",
        "arch": "s390x",
        "artifact_id": f"a{minutes_ago}",
        "event_type": TAGGED,
        "event_key": f"t{minutes_ago}",
        "event_ts": NOW - timedelta(minutes=minutes_ago),
        "coalesce_minutes": 0,
        "max_per_hour": 0,
        "max_per_day": 0,
    }
    return {**r, **over}


def decisions(rows, counts=None):
    return [
        (r["artifact_id"], d) for r, d, _ in dispatch.decide(rows, counts or {}, NOW)
    ]


def test_without_limits_everything_fires():
    assert decisions([event(5), event(3)]) == [("a5", "fire"), ("a3", "fire")]


def test_coalescing_fires_only_the_newest_once_its_window_closes():
    rows = [event(m, coalesce_minutes=30) for m in (90, 75, 50, 10)]
    # 90 is superseded by 75, 75 by 50 (25 min later); 50's window closed with nothing newer
    # inside it; 10's window is still open.
    assert decisions(rows) == [
        ("a90", "coalesced"),
        ("a75", "coalesced"),
        ("a50", "fire"),
        ("a10", "coalescing"),
    ]


def test_coalescing_is_per_component_and_arch():
    rows = [
        event(60, coalesce_minutes=30),
        event(55, coalesce_minutes=30, arch="x86_64"),
    ]
    assert decisions(rows) == [("a60", "fire"), ("a55", "fire")]


def test_rate_limits_defer_the_excess_counting_what_was_already_sent():
    rows = [event(m, max_per_hour=2) for m in (30, 20, 10)]
    assert decisions(rows, {"s": (1, 1)}) == [
        ("a30", "fire"),
        ("a20", "throttled"),
        ("a10", "throttled"),
    ]
    assert decisions([event(5, max_per_day=3)], {"s": (0, 3)}) == [("a5", "throttled")]


def test_payload_is_the_chosen_subset_and_refuses_unknown_fields():
    ctx = {
        "artifact_id": AID,
        "tag": "t",
        "digest": DIGEST,
        "verdicts": [{"test_type": "trunk"}],
    }
    assert dispatch.payload([], ctx) == {"artifact_id": AID, "tag": "t"}
    assert dispatch.payload(["digest", "verdicts"], ctx) == {
        "digest": DIGEST,
        "verdicts": [{"test_type": "trunk"}],
    }
    with pytest.raises(ValueError, match="secret"):
        dispatch.payload(["secret"], ctx)


class PreviewStore:
    def __init__(self, rows, counts=None):
        self.rows, self._counts = rows, counts or {}

    def events(self, since, **kw):
        return self.rows

    def counts(self, now):
        return self._counts


def test_preview_explains_every_event_and_writes_nothing():
    rows = [
        {**event(90), "reason": "", "dispatch_state": "triggered"},
        {**event(80), "reason": "excluded", "dispatch_state": ""},
        {**event(70), "reason": "waiting: trunk=passed", "dispatch_state": ""},
        {**event(60, max_per_day=1), "reason": "", "dispatch_state": ""},
        {**event(50, max_per_day=1), "reason": "", "dispatch_state": ""},
    ]
    out = [
        (r["outcome"], r["why"])
        for r in dispatch.preview(PreviewStore(rows), NOW, subscription_id="s")
    ]
    assert out == [
        ("dispatched (triggered)", ""),
        ("not dispatched", "excluded"),
        ("not dispatched", "waiting: trunk=passed"),
        ("would fire", ""),
        ("throttled", "1/h, 1/day sent"),
    ]
