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

"""The GHA leg's artifact row and verdict: what must land, and what must be refused."""

import uuid

from spyre_clickhouse_ingest import (
    gha_artifact_id,
    insert_gha_artifact_result,
    installed_digest,
)
from spyre_clickhouse_ingest.schema import (
    ARTIFACT_RESULTS,
    ARTIFACTS,
    ORIGIN_VALUES,
    STATE_VALUES,
    TEST_TYPE_VALUES,
)

# The id _package-image stamps into spyre-backend-dev/amd64, verified against prod.
BASE = "2b397099-6200-52fb-98c4-b603961a0582"
INSTALLED = "torch-spyre@07379f50 ibm-flex-devel"
RUN = "1a6080e8-d061-547f-ab63-1af99b18ad0c"


class FakeClient:
    """Records inserts; `counts` is what a count() query reports, per call, in order."""

    def __init__(self, counts=()):
        self.counts = list(counts)
        self.inserts = []
        self.queries = []

    def insert(self, table, rows, column_names=None, database=None):
        self.inserts.append((table, rows, column_names, database))

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters or {}))
        n = self.counts.pop(0) if self.counts else 0

        class R:
            result_rows = [(n,)]

        return R()


def _rows(client, table):
    """The rows inserted into `table`, as dicts keyed by column name."""
    out = []
    for name, rows, cols, _db in client.inserts:
        if name == table.name:
            out.extend(dict(zip(cols, r)) for r in rows)
    return out


def _call(client, **kw):
    args = {
        "artifact_id": gha_artifact_id("torch-spyre", BASE, INSTALLED, "amd64"),
        "component": "torch-spyre",
        "arch": "amd64",
        "run_id": RUN,
        "test_type": "regression",
        "state": "passed",
        "base_artifact_id": BASE,
        "installed": INSTALLED,
        "repo": "torch-spyre/torch-spyre",
        "git_ref": "main",
        "git_sha": "07379f50deadbeef",
        "run_url": "https://github.com/torch-spyre/torch-spyre/actions/runs/42",
    }
    args.update(kw)
    return insert_gha_artifact_result(client, "db", **args)


def test_writes_the_artifact_and_its_verdict_together():
    # Either both land or neither does: a verdict with no artifact row is unjoinable.
    c = FakeClient()
    assert _call(c) is True
    assert len(_rows(c, ARTIFACTS)) == 1
    assert len(_rows(c, ARTIFACT_RESULTS)) == 1


def test_the_hash_inputs_stay_readable_beside_the_opaque_id():
    # uuid5 is one-way, so a row not carrying what was hashed can never be verified.
    c = FakeClient()
    _call(c)
    row = _rows(c, ARTIFACTS)[0]
    assert row["artifact_name"] == BASE
    assert row["props"]["id12"] == installed_digest(INSTALLED)
    assert row["props"]["installed"] == INSTALLED
    # Recomputing from the row alone must reproduce the id it is stored under.
    assert (
        gha_artifact_id(
            "torch-spyre", row["artifact_name"], row["props"]["installed"], "amd64"
        )
        == row["artifact_id"]
    )


def test_origin_is_a_value_the_ddl_admits():
    # The docstring once said origin='gha'; chk_origin admits no such value.
    c = FakeClient()
    _call(c)
    assert _rows(c, ARTIFACTS)[0]["origin"] in ORIGIN_VALUES


def test_the_base_image_is_named_as_an_identity_dep():
    # 'base=' is the dep-entry shape for a base named by content: a GHA id has no id12.
    c = FakeClient()
    _call(c)
    assert _rows(c, ARTIFACTS)[0]["identity_deps"] == [f"base={BASE}"]


def test_sources_carries_the_commit_the_tier_delta_joins_on():
    # resolve_covered_tiers.py ARRAY JOINs this on git_sha -- the delta's only route in.
    c = FakeClient()
    _call(c)
    assert _rows(c, ARTIFACTS)[0]["sources"] == [
        ("torch-spyre/torch-spyre", "main", "07379f50deadbeef")
    ]


def test_arch_is_folded_so_one_leg_hashes_as_one():
    # Jenkins says amd64 and GHA says x86_64 for the same machine.
    c1, c2 = FakeClient(), FakeClient()
    _call(c1, arch="amd64")
    _call(c2, arch="x86_64")
    assert (
        _rows(c1, ARTIFACTS)[0]["arch"] == _rows(c2, ARTIFACTS)[0]["arch"] == "x86_64"
    )


def test_refuses_a_partial_identity_rather_than_writing_one():
    # A blank field hashes to a real uuid every incomplete artifact would share.
    for blank in ("component", "arch", "run_id", "artifact_id"):
        c = FakeClient()
        assert _call(c, **{blank: ""}) is False, blank
        assert c.inserts == [], blank


def test_a_known_artifact_is_not_recorded_twice():
    # Plain MergeTree, no dedup key, so the guard is the writer's job.
    c = FakeClient(counts=[1, 0])  # artifact known, verdict not
    assert _call(c) is True
    assert _rows(c, ARTIFACTS) == []
    assert len(_rows(c, ARTIFACT_RESULTS)) == 1


def test_a_re_ingest_does_not_duplicate_the_verdict():
    c = FakeClient(counts=[1, 1])  # both already there
    assert _call(c) is True
    assert c.inserts == []


def test_one_run_may_report_several_tiers():
    # A multi-tier leg writes one row per tier: two facts, not a duplicate.
    c = FakeClient()
    _call(c, test_type="integration")
    _call(c, test_type="regression")
    tiers = [r["test_type"] for r in _rows(c, ARTIFACT_RESULTS)]
    assert tiers == ["integration", "regression"]


def test_the_verdict_row_is_valid_against_the_ddl_check_sets():
    c = FakeClient()
    _call(c)
    row = _rows(c, ARTIFACT_RESULTS)[0]
    assert row["state"] in STATE_VALUES
    assert row["test_type"] in TEST_TYPE_VALUES
    assert row["result_kind"] == "functional"
    assert uuid.UUID(row["artifact_id"]).version == 5
    assert row["run_id"] == RUN


def test_installed_digest_is_order_independent_and_empty_for_nothing():
    # Order-sensitive would mint a fresh identity per re-run; '' is the base image.
    assert installed_digest("b a") == installed_digest("a,b") != ""
    assert installed_digest("a a b") == installed_digest("a b")
    assert installed_digest("") == ""
    assert installed_digest("   ") == ""


def test_an_unchanged_image_gets_a_verdict_but_no_artifact_row():
    # The prebaked path: artifact_id == base, so the artifact is the orchestrator's and
    # already recorded. Our own row would be the duplicate the plain MergeTree surfaces.
    c = FakeClient()
    assert _call(c, artifact_id=BASE, installed="") is True
    assert _rows(c, ARTIFACTS) == []
    assert len(_rows(c, ARTIFACT_RESULTS)) == 1
    assert _rows(c, ARTIFACT_RESULTS)[0]["artifact_id"] == BASE
