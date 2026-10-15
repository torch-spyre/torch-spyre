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
from spyre_clickhouse_ingest.writer import ArtifactWriter

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
        if "count()" not in sql:
            # A record lookup (resolver.Lookup) finds nothing.
            class Empty:
                result_rows: list = []

            return Empty()
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


def test_result_recorded_query_excludes_seeded_running_rows():
    # Jenkins seeds a 'running' row before dispatch (Part C); it must not count as "already
    # recorded" or the leg's own terminal insert_gha_result call would be silently skipped.
    c = FakeClient(counts=[1])
    assert (
        ArtifactWriter.result_recorded(c, "db", BASE, RUN, "functional", "regression")
        is True
    )
    sql, params = c.queries[0]
    assert "state != 'running'" in sql
    assert params == {
        "artifact_id": BASE,
        "run_id": RUN,
        "result_kind": "functional",
        "test_type": "regression",
    }


def test_a_seeded_running_row_does_not_block_the_terminal_insert():
    # End-to-end through insert_gha_artifact_result: artifact already known (count=1), and
    # the ONLY existing result row is the seed -- FakeClient can't filter by state itself, so
    # this only proves the call path still inserts when result_recorded's own guard (tested
    # above) says "not yet recorded".
    c = FakeClient(counts=[1, 0])  # artifact known; no non-running verdict yet
    assert _call(c, state="passed") is True
    assert len(_rows(c, ARTIFACT_RESULTS)) == 1
    assert _rows(c, ARTIFACT_RESULTS)[0]["state"] == "passed"


def test_an_unchanged_image_gets_a_verdict_but_no_artifact_row():
    # The prebaked path: artifact_id == base, so the artifact is the orchestrator's and
    # already recorded. Our own row would be the duplicate the plain MergeTree surfaces.
    c = FakeClient()
    assert _call(c, artifact_id=BASE, installed="") is True
    assert _rows(c, ARTIFACTS) == []
    assert len(_rows(c, ARTIFACT_RESULTS)) == 1
    assert _rows(c, ARTIFACT_RESULTS)[0]["artifact_id"] == BASE


# ── the batch a pipeline writer hands over (artifacts write) ─────────────────────────────

NODE = {
    "component": "spyre-backend",
    "artifact_name": "spyre-backend-dev",
    "id12": "5e67196b1a4c",
    "arch": "amd64",
    "kind": "image",
    "ref": "registry.example/spyre-backend-dev:amd64-dev-5e67196b1a4c",
}


def _batch():
    from spyre_clickhouse_ingest.artifacts import write_batch

    c = FakeClient()
    done = write_batch(
        c,
        "db",
        {
            "artifacts": [
                {
                    "artifact": NODE,
                    "sources": [{"repo": "r", "git_ref": "main", "git_sha": "abc"}],
                    "props": {"run_url": "u"},
                }
            ],
            "tags": [
                {
                    "artifact": NODE,
                    "tag": "latest",
                    "tag_family": "main",
                    "ref": "registry.example/spyre-backend-dev:amd64",
                    "props": {"source": "jenkins"},
                }
            ],
            "results": [
                {
                    "artifact": NODE,
                    "run_id": RUN,
                    "test_type": "perf",
                    "state": "passed",
                }
            ],
        },
    )
    return c, done


def test_a_batch_writes_under_the_jenkins_artifact_id():
    from spyre_clickhouse_ingest import ArtifactId

    c, done = _batch()
    assert done == {"artifacts": 1, "tags": 1, "results": 1}
    aid = ArtifactId.derive(
        "spyre-backend", "spyre-backend-dev", "5e67196b1a4c", "x86_64"
    )
    for t in (ARTIFACTS, ARTIFACT_RESULTS):
        assert [r["artifact_id"] for r in _rows(c, t)] == [aid]
    art = _rows(c, ARTIFACTS)[0]
    assert (art["arch"], art["sources"], art["props"]["run_url"]) == (
        "x86_64",
        [("r", "main", "abc")],
        "u",
    )


def test_a_tag_names_its_own_moving_ref_not_the_artifacts():
    from spyre_clickhouse_ingest.schema import ArtifactTags

    c, _ = _batch()
    (tag,) = _rows(c, ArtifactTags)
    assert tag["refs"] == [
        (
            "container-pull",
            "pullspec",
            "registry.example",
            "registry.example/spyre-backend-dev:amd64",
        )
    ]
    assert tag["props"] == {"id12": "5e67196b1a4c", "source": "jenkins"}


def test_a_batch_result_takes_the_artifacts_arch_and_derives_its_kind():
    c, _ = _batch()
    (res,) = _rows(c, ARTIFACT_RESULTS)
    assert (res["arch"], res["result_kind"]) == ("x86_64", "performance")


def test_an_unknown_kind_is_recorded_as_a_download():
    assert ArtifactWriter.ref_shape("file") == ArtifactWriter.REF_SHAPE["generic"]


def test_a_misspelled_batch_key_is_refused_before_anything_is_written():
    import pytest
    from spyre_clickhouse_ingest.artifacts import write_batch

    c = FakeClient()
    batch = {
        "tags": [
            {"artifact": {**NODE, "artifactName": "x"}, "tag": "t", "tagFamily": "f"}
        ],
    }
    with pytest.raises(
        ValueError, match=r"tags\[0\]\.tagFamily.*tags\[0\]\.artifact\.artifactName"
    ):
        write_batch(c, "db", batch)
    assert c.inserts == []


def test_a_tag_entry_without_a_ref_names_the_artifacts_own():
    from spyre_clickhouse_ingest.artifacts import write_batch
    from spyre_clickhouse_ingest.schema import ArtifactTags

    for entry in ({}, {"ref": ""}):
        c = FakeClient()
        write_batch(
            c,
            "db",
            {"tags": [{"artifact": NODE, "tag": "t", "tag_family": "f", **entry}]},
        )
        (tag,) = _rows(c, ArtifactTags)
        assert [r[3] for r in tag["refs"]] == [NODE["ref"]], entry
