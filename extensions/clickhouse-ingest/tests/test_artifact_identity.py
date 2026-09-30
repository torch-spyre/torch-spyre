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

"""ArtifactIdentity: every way of naming an artifact reduces to one artifact_id."""

import pytest

from spyre_clickhouse_ingest import (
    ArtifactIdentity,
    gha_artifact_id,
    insert_artifact,
)
from spyre_clickhouse_ingest.schema import ARTIFACT_REFS, ARTIFACT_TAGS, ARTIFACTS

from test_artifact_writer import BASE, INSTALLED, FakeClient, _rows

DIGEST = "sha256:62a7fc6986014c8f7c9d1ad5e7b4b0a7f1d6b1b2c3d4e5f60718293a4b5c6d7e"
IMAGE = f"registry.example.com/team/hf-adapters-devel@{DIGEST}"


def test_matches_an_id_the_jenkins_writer_put_in_prod():
    # spyre_v2.artifacts: torch-spyre / torch-spyre-dev / 5e67196b1a4c / x86_64.
    ident = ArtifactIdentity("torch-spyre", "torch-spyre-dev", "5e67196b1a4c", "amd64")
    assert ident.artifact_id == "16f513f4-6423-575b-b62a-809ae77fa56a"


def test_image_is_named_by_its_per_arch_digest():
    ident = ArtifactIdentity.from_image(IMAGE, "s390x")
    assert (ident.component, ident.artifact_name, ident.id12, ident.kind) == (
        "hf-adapters",
        "hf-adapters-devel",
        "62a7fc698601",
        "image",
    )
    # A tag in the ref does not change the identity; only the digest does.
    tagged = IMAGE.replace("@", ":v1@")
    assert ArtifactIdentity.from_image(tagged, "s390x").artifact_id == ident.artifact_id


def test_image_without_a_digest_is_refused():
    with pytest.raises(ValueError, match="sha256"):
        ArtifactIdentity.from_image(
            "registry.example.com/team/torch-spyre:latest", "s390x"
        )


def test_a_jenkins_image_matches_its_recorded_id_through_overrides():
    # The orchestrator records component/config name/content-identity id12, none of which
    # a digest implies: its tag carries the id12 (amd64-dev-5e67196b1a4c).
    ref = "registry.example.com/next/torch-spyre:amd64-dev-5e67196b1a4c@" + DIGEST
    spec = f"image:{ref};component=torch-spyre;name=torch-spyre-dev;id12=5e67196b1a4c"
    ident = ArtifactIdentity.parse(spec, "amd64", "hf-adapters")
    assert ident.artifact_id == "16f513f4-6423-575b-b62a-809ae77fa56a"
    assert ident.content_digest == DIGEST


def test_an_image_component_is_explicit_or_the_name_less_its_dev_suffix():
    dev = "registry.example.com/team/torch-spyre-dev@" + DIGEST
    assert (
        ArtifactIdentity.parse(f"image:{dev}", "s390x", "x").component == "torch-spyre"
    )
    assert (
        ArtifactIdentity.parse(f"image:{dev};component=spyre", "s390x", "x").component
        == "spyre"
    )


@pytest.mark.parametrize(
    "spec",
    [
        "image:registry.example.com/team/torch-spyre@sha256:abc",
        "image:registry.example.com/team/torch-spyre@sha256:" + "z" * 64,
        f"image:{IMAGE};id12=nothex",
        f"image:{IMAGE};flavour=x",
        f"image:{IMAGE};component",
    ],
)
def test_a_malformed_image_spec_is_refused(spec):
    with pytest.raises(ValueError):
        ArtifactIdentity.parse(spec, "s390x", "torch-spyre")


@pytest.mark.parametrize(
    "spec",
    [
        "generic:https://h.example.com/p/bundle/",
        "generic:https://h.example.com/p/bundle/#",
        "generic:https://h.example.com/p/bundle/#" + "ab" * 6,
        "generic:https://h.example.com/p/bundle/#" + "zz" * 32,
        "generic:#" + "ab" * 32,
    ],
)
def test_a_generic_spec_needs_a_url_and_a_full_hex_sha256(spec):
    # Accepting these once wrote a junk id12 (e.g. 'https://exam') that write-once kept.
    with pytest.raises(ValueError):
        ArtifactIdentity.parse(spec, "s390x", "spyre")


def test_gha_form_is_the_existing_gha_id():
    ident = ArtifactIdentity.from_gha("torch-spyre", BASE, INSTALLED, "amd64")
    assert ident.artifact_id == gha_artifact_id("torch-spyre", BASE, INSTALLED, "amd64")


def test_parse_accepts_every_spelling():
    # torch-spyre's suite in hf-adapters' image: the artifact is still hf-adapters'.
    img = ArtifactIdentity.parse(f"image:{IMAGE}", "s390x", "torch-spyre")
    assert (img.component, img.id12) == ("hf-adapters", "62a7fc698601")
    gen = ArtifactIdentity.parse(
        "generic:https://h/p/bundle/#" + "ab" * 32, "s390x", "spyre"
    )
    assert (gen.kind, gen.artifact_name, gen.id12) == ("generic", "bundle", "ab" * 6)
    rec = (
        f"{gha_artifact_id('torch-spyre', BASE, INSTALLED, 'amd64')}|{BASE}|{INSTALLED}"
    )
    assert (
        ArtifactIdentity.parse(rec, "amd64", "torch-spyre").artifact_id
        == rec.split("|")[0]
    )
    # A bare id carries no hash inputs, so there is nothing to register.
    assert ArtifactIdentity.parse(BASE, "amd64", "torch-spyre") is None


def test_insert_writes_artifact_ref_and_tag_once():
    ident = ArtifactIdentity.from_image(IMAGE, "s390x")
    c = FakeClient(counts=[0, 0, 0])
    aid = insert_artifact(
        c,
        "db",
        ident,
        origin="promoted",
        tags=[("release-2026-09-22", "release", {}), ("weekly-W39-18", "release", {})],
    )
    assert aid == ident.artifact_id
    (art,) = _rows(c, ARTIFACTS)
    assert (art["origin"], art["props"]["id12"], art["props"]["ref"]) == (
        "promoted",
        "62a7fc698601",
        IMAGE,
    )
    (ref,) = _rows(c, ARTIFACT_REFS)
    assert (ref["method"], ref["index_uri"], ref["content_digest"]) == (
        "container-pull",
        "registry.example.com",
        DIGEST,
    )
    # One artifact under two names: a dated tag and the producer's own.
    tags = _rows(c, ARTIFACT_TAGS)
    assert sorted((t["tag"], t["tag_family"]) for t in tags) == [
        ("release-2026-09-22", "release"),
        ("weekly-W39-18", "release"),
    ]
    assert {t["artifact_id"] for t in tags} == {aid}

    # Already recorded and already tagged: only the idempotent ref row is re-sent.
    again = FakeClient(counts=[1, 1, 1])
    insert_artifact(again, "db", ident, tags=[("release-2026-09-22", "release", {})])
    assert not _rows(again, ARTIFACTS) and not _rows(again, ARTIFACT_TAGS)


def test_result_kind_follows_the_test_type():
    from spyre_clickhouse_ingest import insert_artifact_result
    from spyre_clickhouse_ingest.schema import ARTIFACT_RESULTS

    aid = ArtifactIdentity.from_image(IMAGE, "s390x").artifact_id
    kinds = {}
    for test_type in ("fvt", "perf", "model_ops"):
        c = FakeClient(counts=[0])
        insert_artifact_result(
            c,
            "db",
            artifact_id=aid,
            run_id=aid,
            test_type=test_type,
            state="passed",
            arch="s390x",
        )
        kinds[test_type] = _rows(c, ARTIFACT_RESULTS)[0]["result_kind"]
    assert kinds == {
        "fvt": "functional",
        "perf": "performance",
        "model_ops": "capability",
    }
