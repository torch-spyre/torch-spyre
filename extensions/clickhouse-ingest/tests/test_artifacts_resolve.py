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

"""`artifacts resolve` / `ensure`: any spec, one artifact id, whichever writer asks."""

import json
import urllib.error
from datetime import date, datetime, timezone

import pytest

from spyre_clickhouse_ingest import ArtifactIdentity, artifacts
from spyre_clickhouse_ingest.identity import ArtifactId
from spyre_clickhouse_ingest.registry import (
    TAG_FAMILIES_ENV,
    Registry,
    dated,
    dated_tag,
    family_of,
    family_tag,
    load_tag_families,
    tag_families,
)
from spyre_clickhouse_ingest.resolver import (
    Lookup,
    NeedsRegistry,
    ensure,
    ensure_artifact,
    resolve,
)
from spyre_clickhouse_ingest.writer import ArtifactWriter
from spyre_clickhouse_ingest.schema import ARTIFACT_REFS, ARTIFACT_TAGS, ARTIFACTS

REPO = "ai_sw_accel/2.0/prod/torch-spyre-devel"
IMAGE = f"icr.io/{REPO}"
LEAF = "sha256:" + "b6" * 32
OTHER = "sha256:" + "6b" * 32
LIST = "sha256:" + "19" * 32
DAY = date(2026, 10, 4)


def _index(*entries):
    return {
        "manifests": [
            {"digest": d, "platform": {"architecture": a}} for a, d in entries
        ]
    }


class FakeRegistry(Registry):
    """Serves manifests, blobs and tags from dicts, never the network."""

    def __init__(self, served, tags=(), repo=REPO):
        super().__init__()
        self.served = served
        self._tags = {repo: list(tags)}

    def _get(self, repo, path, accept=""):
        ref = path.split("/", 1)[1]
        if ref not in self.served:
            raise urllib.error.HTTPError(path, 404, "not found", {}, None)
        return self.served[ref]


class FakeLookup(Lookup):
    """Answers every lookup from `rows` (artifact_id -> row) and `by` (method -> artifact_id)."""

    def __init__(self, rows=(), **by):
        super().__init__(client=object(), db="db")
        self.rows = {r[0]: r for r in rows}
        self.by = by
        self.asked = []

    def _hit(self, method, *args):
        self.asked.append((method, args))
        aid = self.by.get(method)
        return self.rows.get(aid) if aid else None

    def by_id(self, aid):
        return self.rows.get(aid)

    def by_refs(self, kind, arch, refs):
        return self._hit("refs", kind, arch, tuple(refs))

    def by_digest(self, arch, digest):
        return self._hit("digest", arch, digest)

    def by_rpm_file(self, arch, filename, rpm_name):
        return self._hit("rpm_file", arch, filename, rpm_name)

    def by_tag(self, arch, pullspec):
        return self._hit("tag", arch, pullspec)

    def by_id12(self, kind, arch, name, id12):
        return self._hit("id12", kind, arch, name, id12)


def _row(component, name, id12, arch, kind, ref=""):
    return (
        ArtifactId.derive(component, name, id12, arch),
        component,
        name,
        id12,
        arch,
        kind,
        ref,
    )


LISTED = {LIST: (LIST, _index(("ppc64le", OTHER), ("s390x", LEAF))), LEAF: (LEAF, {})}

# -- family tags ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag_family, tag, built, expected",
    [
        (
            "nightly-supply-chain",
            "nightly-20261004",
            None,
            "nightly-supply-chain-2026-10-04",
        ),
        (
            "snap-supply-chain",
            "snap-20261004T002701_277",
            None,
            "snap-supply-chain-2026-10-04T002701",
        ),
        ("snap-supply-chain", "snap-20261005", None, "snap-supply-chain-2026-10-05"),
        ("weekly-supply-chain", "weekly-W40", DAY, "weekly-supply-chain-2026-w40"),
        (
            "weekly-supply-chain",
            "weekly-W01",
            date(2026, 12, 30),
            "weekly-supply-chain-2027-w01",
        ),
        ("ci-cd-tech-preview", "ci-cd-tech-preview-v3", None, "ci-cd-tech-preview-v3"),
    ],
)
def test_each_registry_tag_has_one_v2_name_in_its_family(
    tag_family, tag, built, expected
):
    assert family_tag(tag_family, tag, built) == expected


# -- images --------------------------------------------------------------------------------


def test_a_manifest_list_its_leaf_and_a_tag_resolve_to_one_artifact():
    served = {**LISTED, "nightly-latest": (LIST, LISTED[LIST][1])}
    reg = FakeRegistry(served)
    ids = {
        resolve(f"image:{IMAGE}{ref}", "s390x", registry=reg)["artifact_id"]
        for ref in (f"@{LIST}", f"@{LEAF}", ":nightly-latest")
    }
    assert ids == {ArtifactIdentity.from_image(f"{IMAGE}@{LEAF}", "s390x").artifact_id}


def test_an_unregistered_image_derives_what_register_and_ingest_record():
    out = resolve(
        f"image:{IMAGE}:snap-latest@{LIST}", "s390x", registry=FakeRegistry(LISTED)
    )
    assert (out["source"], out["lookup"], out["artifact"]) == (
        "derived",
        "none",
        f"image:{IMAGE}@{LEAF}",
    )
    for spec in (
        out["artifact"],
        f"image:icr.io/ai_sw_accel_dev/torch-spyre/torch-spyre-devel:snap-latest@{LEAF}",
    ):
        assert (
            ArtifactIdentity.parse(spec, "s390x", "x").artifact_id == out["artifact_id"]
        )


def test_an_existing_record_found_by_its_leaf_wins_over_the_derived_id():
    recorded = _row("torch-spyre", "torch-spyre-dev", "9e28cf2e5c48", "x86_64", "image")
    reg = FakeRegistry({LIST: (LIST, _index(("amd64", LEAF))), LEAF: (LEAF, {})})
    out = resolve(f"image:{IMAGE}@{LIST}", "amd64", registry=reg,
                  lookup=FakeLookup([recorded], digest=recorded[0]))  # fmt: skip
    assert (out["artifact_id"], out["source"]) == (recorded[0], "existing")


def _labelled(
    component="torch-spyre", repo="ai_sw_accel/2.0/next/builds/amd64/torch-spyre"
):
    aid = ArtifactId.derive(component, "torch-spyre-dev", "9e28cf2e5c48", "x86_64")
    labels = {"spyre.artifact.id": aid, "spyre.artifact.id12": "9e28cf2e5c48",
              "spyre.artifact.name": "torch-spyre-dev", "spyre.artifact.arch": "x86_64"}  # fmt: skip
    served = {
        LEAF: (LEAF, {"config": {"digest": "sha256:cfg"}}),
        "sha256:cfg": ("", {"config": {"Labels": labels}}),
    }
    return aid, FakeRegistry(served, repo=repo)


def test_an_orchestrator_image_is_named_by_its_own_label_without_a_database():
    aid, reg = _labelled()
    out = resolve(
        f"image:icr.io/ai_sw_accel/2.0/next/builds/amd64/torch-spyre@{LEAF}",
        "x86_64",
        registry=reg,
    )
    assert (out["artifact_id"], out["source"]) == (aid, "label")


def test_a_label_inherited_from_a_base_image_is_ignored():
    aid, reg = _labelled(repo="ai_sw_accel/2.0/prod/hf-adapters-devel")
    out = resolve(
        f"image:icr.io/ai_sw_accel/2.0/prod/hf-adapters-devel@{LEAF}",
        "x86_64",
        registry=reg,
    )
    assert out["artifact_id"] != aid and out["source"] == "derived"


def test_the_snap_builds_own_tag_beats_the_days_aggregate():
    served = {
        **LISTED,
        "snap-20261004": (LIST, _index(("s390x", LEAF))),
        "snap-20261004T002701_277": (LIST, _index(("s390x", LEAF))),
    }
    tags = ["snap-20261003T000000_1", "snap-20261004", "snap-20261004T002701_277"]
    out = resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=FakeRegistry(served, tags),
                  tag_family="snap-supply-chain", tag_date=DAY)  # fmt: skip
    assert (out["tag"], out["tag_family"], out["registry_tag"]) == (
        "snap-supply-chain-2026-10-04T002701",
        "snap-supply-chain",
        "snap-20261004T002701_277",
    )


def test_a_family_with_no_registry_tag_is_dated_by_the_tag_date():
    reg = FakeRegistry(
        {**LISTED, "ci-cd-tech-preview-v3": (LEAF, {})}, ["ci-cd-tech-preview-v3"]
    )
    out = resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=reg,
                  tag_family="snap-supply-chain", tag_date=date(2026, 10, 6))  # fmt: skip
    assert (out["tag"], out["tag_family"]) == (
        "snap-supply-chain-2026-10-06",
        "snap-supply-chain",
    )
    out = resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=reg,
                  tag_family="weekly-supply-chain", tag_date=date(2026, 10, 6))  # fmt: skip
    assert out["tag"] == "weekly-supply-chain-2026-w41"
    # Without a date, a family with no registry tag names none; another family's tag is not taken.
    out = resolve(
        f"image:{IMAGE}@{LEAF}",
        "s390x",
        registry=reg,
        tag_family="nightly-supply-chain",
    )
    assert (out["tag"], out["tags"]) == ("", [])


def test_a_full_tech_preview_tag_replaces_the_resolved_one():
    reg = FakeRegistry(
        {**LISTED, "ci-cd-tech-preview-v2": (LEAF, {})}, ["ci-cd-tech-preview-v2"]
    )
    spec = f"image:{IMAGE}@{LEAF}"
    out = resolve(spec, "s390x", registry=reg, tag_family="ci-cd-tech-preview")
    assert out["tag"] == "ci-cd-tech-preview-v2"
    out = resolve(spec, "s390x", registry=reg, tag_family="ci-cd-tech-preview",
                  tags=["ci-cd-tech-preview-v3"])  # fmt: skip
    assert out["tags"] == [["ci-cd-tech-preview-v3", "ci-cd-tech-preview"]]


def test_a_full_tag_takes_its_family_from_its_prefix():
    out = resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=FakeRegistry(LISTED),
                  tags=["nightly-supply-chain-2026-10-04"])  # fmt: skip
    assert (out["tag"], out["tag_family"]) == (
        "nightly-supply-chain-2026-10-04",
        "nightly-supply-chain",
    )
    reg = FakeRegistry(LISTED)
    out = resolve(
        f"image:{IMAGE}@{LEAF}", "s390x", registry=reg, tags=["release-2026-10-04"]
    )
    assert out["tags"] == [["release-2026-10-04", "release"]]
    # A tag no family prefix names needs its family; it never defaults to release.
    with pytest.raises(ValueError, match="names no tag family"):
        resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=reg, tags=["rc1"])
    for family in ("pr", "release"):
        out = resolve(
            f"image:{IMAGE}@{LEAF}",
            "s390x",
            registry=reg,
            tags=[("torch-spyre#5206", family)],
        )
        assert out["tags"] == [["torch-spyre#5206", family]]
    out = resolve(
        f"image:{IMAGE}@{LEAF}",
        "s390x",
        registry=reg,
        tag_family="main",
        tags=["torch-spyre@f7a6afb683d0"],
    )
    assert (out["tag"], out["tag_family"]) == ("torch-spyre@f7a6afb683d0", "main")
    with pytest.raises(ValueError, match="unknown tag_family"):
        resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=reg, tags=[("rc1", "nope")])


def test_a_tag_in_another_family_is_added_beside_the_resolved_one():
    reg = FakeRegistry(
        {**LISTED, "nightly-20261004": (LIST, LISTED[LIST][1])}, ["nightly-20261004"]
    )
    out = resolve(f"image:{IMAGE}@{LIST}", "s390x", registry=reg, tag_family="nightly-supply-chain",
                  tags=["ci-cd-tech-preview-v3", ("rc1", "release")])  # fmt: skip
    assert (out["tag"], out["tag_family"]) == (
        "nightly-supply-chain-2026-10-04",
        "nightly-supply-chain",
    )
    assert out["tags"] == [
        ["nightly-supply-chain-2026-10-04", "nightly-supply-chain"],
        ["ci-cd-tech-preview-v3", "ci-cd-tech-preview"],
        ["rc1", "release"],
    ]


def test_a_weekly_tag_takes_the_iso_year_of_the_image_build():
    served = {
        **LISTED,
        "weekly-W40": (LIST, _index(("s390x", LEAF))),
        LEAF: (LEAF, {"config": {"digest": "sha256:cfg"}}),
        "sha256:cfg": ("", {"created": "2026-10-01T08:00:00Z"}),
    }
    out = resolve(f"image:{IMAGE}@{LIST}", "s390x",
                  registry=FakeRegistry(served, ["weekly-W39", "weekly-W40"]),
                  tag_family="weekly-supply-chain")  # fmt: skip
    assert out["tag"] == "weekly-supply-chain-2026-w40"


def test_an_unresolvable_image_resolves_to_nothing():
    assert resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=FakeRegistry({})) is None
    assert (
        resolve(f"image:{IMAGE}@{LIST}", "x86_64", registry=FakeRegistry(LISTED))
        is None
    )


# -- rpm / wheel / generic / bare id -------------------------------------------------------

RPM_GLOB = "ibm-aiu-toolbox-e2e-*.bc23d29628db.*.x86_64"
RPM_FILE = "ibm-aiu-toolbox-e2e-1.0.0-0.next.1+3.bc23d29628db.el10.x86_64.rpm"


def test_an_rpm_file_finds_the_record_its_glob_names():
    recorded = _row(
        "aiu-toolbox", "ibm-aiu-toolbox-e2e", "bc23d29628db", "amd64", "rpm", RPM_GLOB
    )
    out = resolve(
        f"rpm:{RPM_FILE}", "x86_64", lookup=FakeLookup([recorded], rpm_file=recorded[0])
    )
    assert (out["artifact_id"], out["source"], out["component"]) == (
        recorded[0],
        "existing",
        "aiu-toolbox",
    )


def test_an_unrecorded_rpm_derives_the_producers_recipe():
    out = resolve(f"rpm:{RPM_GLOB};component=aiu-toolbox", "amd64")
    assert out["artifact_id"] == ArtifactId.derive(
        "aiu-toolbox", "ibm-aiu-toolbox-e2e", "bc23d29628db", "amd64"
    )
    assert (out["kind"], out["source"], out["refs"]) == (
        "rpm",
        "derived",
        [["dnf", "glob", RPM_GLOB]],
    )


def test_a_wheel_pin_derives_the_producers_recipe():
    pin = "apache-tvm-ffi==0.1.14.post1+146f67a53e78"
    out = resolve(f"wheel:{pin}", "ppc64le")
    assert out["artifact_id"] == ArtifactId.derive(
        "apache-tvm-ffi", "apache-tvm-ffi", "146f67a53e78", "ppc64le"
    )
    assert out["refs"] == [["pip", "url", pin]]


def test_a_wheel_file_name_finds_the_record_its_pin_names():
    recorded = _row(
        "apache-tvm-ffi", "apache-tvm-ffi", "146f67a53e78", "ppc64le", "wheel"
    )
    lookup = FakeLookup([recorded], refs=recorded[0])
    out = resolve("wheel:apache_tvm_ffi-0.1.14.post1+146f67a53e78-cp312-cp312-linux_ppc64le.whl",
                  "ppc64le", lookup=lookup)  # fmt: skip
    assert out["artifact_id"] == recorded[0]
    assert "apache-tvm-ffi==0.1.14.post1+146f67a53e78" in lookup.asked[0][1][2]


def test_a_recorded_wheel_wins_even_with_another_component():
    recorded = _row(
        "hf-adapters", "hf_adapters_spyre", "b091a38e5da2", "amd64", "wheel"
    )
    out = resolve("wheel:hf_adapters_spyre==0.1+b091a38e5da2", "x86_64",
                  lookup=FakeLookup([recorded], refs=recorded[0]))  # fmt: skip
    assert (out["artifact_id"], out["component"]) == (recorded[0], "hf-adapters")


def test_a_generic_file_is_found_by_url_and_sha_else_derived_from_its_sha():
    url = "https://na.artifactory.swg-devops.com/artifactory/r/next/noarch/llvm/llvm-src-080ddeea9a07.tgz"
    recorded = _row(
        "llvm", "llvm-src-080ddeea9a07.tgz", "080ddeea9a07", "x86_64", "generic", url
    )
    lookup = FakeLookup([recorded], refs=recorded[0])
    found = resolve(
        f"generic:{url}#{'ab' * 32}", "x86_64", lookup=lookup, registry="off"
    )
    assert found["artifact_id"] == recorded[0]
    # A URL alone can be overwritten: it is not looked up.
    assert resolve(f"generic:{url}", "x86_64", lookup=lookup, registry="off") is None
    out = resolve(f"generic:{url}#{'ab' * 32};component=llvm", "x86_64")
    assert (out["source"], out["artifact_id"]) == (
        "derived",
        ArtifactId.derive("llvm", "llvm-src-080ddeea9a07.tgz", "ab" * 6, "x86_64"),
    )


def test_a_bare_artifact_id_must_exist():
    recorded = _row("llvm", "x.tgz", "080ddeea9a07", "x86_64", "generic")
    assert (
        resolve(recorded[0], "x86_64", lookup=FakeLookup([recorded]))["source"]
        == "existing"
    )
    assert (
        resolve(
            ArtifactId.derive("a", "b", "c" * 12, "s390x"), "s390x", lookup=FakeLookup()
        )
        is None
    )


def test_a_recorded_row_whose_inputs_do_not_hash_to_its_id_is_refused():
    bad = (
        "00000000-0000-5000-8000-000000000000",
        "x",
        "y",
        "c" * 12,
        "s390x",
        "image",
        "",
    )
    with pytest.raises(ValueError):
        resolve(bad[0], "s390x", lookup=FakeLookup([bad]))


GHA_BASE = "d9e898d6-14b2-59c4-8bdf-7339ffaf553d"
GHA_INSTALLED = "hf-adapters@fa67696108bd,torch-spyre@f7a6afb683d0"
# A recorded prod row: what derive-gha-artifact-id uploaded for that leg.
GHA_AID = "5f8462e0-4d76-5ccd-b5b4-41d66561dc85"
GHA_RECORD = f"gha:{GHA_AID}|{GHA_BASE}|{GHA_INSTALLED}"


def test_a_gha_record_derives_its_own_id_or_finds_its_row():
    out = resolve(GHA_RECORD, "amd64", lookup=FakeLookup(), component="hf-adapters")
    assert (out["source"], out["artifact_id"], out["refs"]) == (
        "derived",
        GHA_AID,
        [],
    )
    row = (GHA_AID, "hf-adapters", GHA_BASE, "4b6dc1216d46", "x86_64", "image", "")
    assert (
        resolve(GHA_RECORD, "x86_64", lookup=FakeLookup([row]))["source"] == "existing"
    )


def test_a_gha_record_whose_fields_do_not_hash_to_its_id_is_refused():
    with pytest.raises(ValueError):
        resolve(GHA_RECORD, "x86_64", lookup=FakeLookup(), component="torch-spyre")
    assert resolve(f"gha:{GHA_AID}", "x86_64", lookup=FakeLookup()) is None


def test_ensure_chains_a_gha_delta_on_its_base():
    client = FakeClient()
    for _ in range(2):
        identity = ensure_artifact(client, "db", GHA_RECORD, "x86_64",
                                   component="hf-adapters")  # fmt: skip
    assert identity.artifact_id == GHA_AID
    (art,) = client.tables[ARTIFACTS.name]
    assert (art["kind"], art["artifact_name"], art["identity_deps"]) == (
        "image",
        GHA_BASE,
        [f"base={GHA_BASE}"],
    )
    assert (art["props"]["id12"], art["props"]["source"]) == ("4b6dc1216d46", "gha")
    assert client.tables[ARTIFACT_REFS.name] == []


def test_an_unknown_spec_is_refused():
    with pytest.raises(ValueError):
        resolve("tarball:x", "s390x")


# -- ensure_artifact -----------------------------------------------------------------------


class FakeClient:
    """A tiny in-memory spyre_v2: counts answer from what was inserted."""

    def __init__(self):
        self.tables = {"artifacts": [], "artifact_refs": [], "artifact_tags": []}

    def insert(self, table, rows, column_names=None, database=None):
        self.tables.setdefault(table, []).extend(
            dict(zip(column_names, r)) for r in rows
        )

    def query(self, sql, parameters=None):
        p = parameters or {}
        n = 0
        if "FROM db.artifacts WHERE artifact_id" in sql and "count()" in sql:
            n = sum(
                r["artifact_id"] == p.get("artifact_id")
                for r in self.tables["artifacts"]
            )
        elif "artifact_refs" in sql and "count()" in sql:
            n = sum(
                r["artifact_id"] == p["artifact_id"] and r["ref"] == p["ref"]
                for r in self.tables["artifact_refs"]
            )
        elif "artifact_tags" in sql and "count()" in sql:
            n = sum(
                r["artifact_id"] == p["artifact_id"] and r["tag"] == p["tag"]
                for r in self.tables["artifact_tags"]
            )
        rows = [(n,)] if "count()" in sql else []

        class R:
            result_rows = rows

        return R()


def test_ensure_records_an_artifact_its_ref_and_tags_once():
    client = FakeClient()
    reg = FakeRegistry(
        {**LISTED, "nightly-20261004": (LIST, LISTED[LIST][1])}, ["nightly-20261004"]
    )
    for _ in range(2):
        identity = ensure_artifact(client, "db", f"image:{IMAGE}@{LIST}", "s390x", origin="promoted",
                                   tags=[("rc1", "release")], tag_family="nightly-supply-chain", registry=reg)  # fmt: skip
    assert (
        identity.artifact_id
        == ArtifactIdentity.from_image(f"{IMAGE}@{LEAF}", "s390x").artifact_id
    )
    assert len(client.tables[ARTIFACTS.name]) == 1
    assert [r["ref"] for r in client.tables[ARTIFACT_REFS.name]] == [f"{IMAGE}@{LEAF}"]
    assert sorted(r["tag"] for r in client.tables[ARTIFACT_TAGS.name]) == [
        "nightly-supply-chain-2026-10-04",
        "rc1",
    ]


def test_ensure_refuses_a_spec_that_names_nothing():
    with pytest.raises(ValueError):
        ensure_artifact(
            FakeClient(),
            "db",
            f"image:{IMAGE}@{OTHER}",
            "s390x",
            registry=FakeRegistry({}),
        )


def test_the_cli_prints_the_resolution_as_json(monkeypatch, capsys):
    monkeypatch.setattr(Registry, "from_env", lambda: FakeRegistry(LISTED))
    artifacts.main(
        ["resolve", "--image", f"{IMAGE}@{LIST}", "--arch", "s390x", "--no-lookup"]
    )
    out = json.loads(capsys.readouterr().out)
    assert (out["artifact"], out["lookup"], out["source"]) == (
        f"image:{IMAGE}@{LEAF}",
        "none",
        "derived",
    )
    with pytest.raises(SystemExit):
        artifacts.main(
            ["resolve", "--image", f"{IMAGE}@{OTHER}", "--arch", "s390x", "--no-lookup"]
        )


def test_the_cli_dates_a_family_tag_today_unless_given_a_tag_date(monkeypatch, capsys):
    monkeypatch.setattr(Registry, "from_env", lambda: FakeRegistry(LISTED))
    base = ["resolve", "--image", f"{IMAGE}@{LIST}", "--arch", "s390x", "--no-lookup"]
    artifacts.main([*base, "--tag-family", "nightly-supply-chain"])
    today = datetime.now(timezone.utc).date().isoformat()
    assert json.loads(capsys.readouterr().out)["tag"] == f"nightly-supply-chain-{today}"
    artifacts.main(
        [*base, "--tag-family", "nightly-supply-chain", "--tag-date", "2026-09-26"]
    )
    assert (
        json.loads(capsys.readouterr().out)["tag"] == "nightly-supply-chain-2026-09-26"
    )
    artifacts.main(base)
    assert json.loads(capsys.readouterr().out)["tag"] == ""
    artifacts.main([*base, "--tag", "ci-cd-tech-preview-v3"])
    out = json.loads(capsys.readouterr().out)
    assert (out["tag"], out["tag_family"]) == (
        "ci-cd-tech-preview-v3",
        "ci-cd-tech-preview",
    )
    assert "channel" not in out


# -- tag_families.yaml ---------------------------------------------------------------------


def test_the_packaged_families_are_the_recorded_ones():
    supply_chain = {"snap-supply-chain", "nightly-supply-chain", "weekly-supply-chain"}
    assert set(tag_families()) == supply_chain | {
        "ci-cd-tech-preview", "release", "pr", "main", "nightly", "weekly", "snap"
    }  # fmt: skip
    assert {f for f in tag_families() if dated(f)} == supply_chain
    # A family with no registry tag and no fallback is named only by an explicit --tag.
    assert dated_tag("pr", DAY) == "" and family_tag("main", "main-1") == ""


def test_an_override_file_replaces_the_families(tmp_path, monkeypatch):
    path = tmp_path / "families.yaml"
    path.write_text(
        "rc:\n"
        "  registry_tag: '^rc(?P<build>\\d+)$'\n"
        "  tag: '{family}-{build}'\n"
        "  fallback: '{family}-{date:%Y%m%d}'\n"
        "  dated: true\n"
        "  prefix: true\n"
    )
    monkeypatch.setenv(TAG_FAMILIES_ENV, str(path))
    assert set(tag_families()) == {"rc"}
    assert (family_of("rc-9"), family_of("nightly-supply-chain-2026-10-04")) == (
        "rc",
        "",
    )
    reg = FakeRegistry({**LISTED, "rc7": (LIST, LISTED[LIST][1])}, ["rc7"])
    out = resolve(f"image:{IMAGE}@{LIST}", "s390x", registry=reg, tag_family="rc")
    assert (out["tag"], out["tag_family"], out["registry_tag"]) == ("rc-7", "rc", "rc7")
    assert dated_tag("rc", DAY) == "rc-20261004"


@pytest.mark.parametrize(
    "text, fault",
    [
        ("x:\n  registry_tag: '^(unclosed'\n  tag: '{family}'\n", "bad registry_tag"),
        ("x:\n  fallback: '{family}-{branch}'\n", "unknown field"),
        ("x:\n  colour: red\n", "unknown key"),
        ("x: {}\nx: {}\n", "duplicate key"),
        ("x:\n  registry_tag: '^x$'\n", "needs a tag"),
    ],
)
def test_a_malformed_families_file_fails_loudly(tmp_path, text, fault):
    path = tmp_path / "families.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=fault):
        load_tag_families(str(path))


# -- modes ---------------------------------------------------------------------------------


class Unreadable:
    """A database that must not be read."""

    def query(self, *args, **kwargs):
        raise AssertionError("read the database")


class Recording:
    """A database holding nothing; keeps every query's parameters."""

    def __init__(self):
        self.params = []

    def query(self, sql, parameters=None):
        self.params.append(parameters or {})

        class R:
            result_rows = []

        return R()


class NoRegistry(Registry):
    def _get(self, *args, **kwargs):
        raise AssertionError("called the registry")


WHEEL = "wheel:apache-tvm-ffi==0.1.14.post1+146f67a53e78"


def test_an_id_is_verified_against_its_record_and_needs_a_database():
    recorded = _row("llvm", "x.tgz", "080ddeea9a07", "x86_64", "generic")
    for spec in (recorded[0], f"id:{recorded[0]}"):
        out = resolve(spec, "x86_64", lookup=FakeLookup([recorded]))
        assert (out["artifact_id"], out["source"], out["lookup"]) == (
            recorded[0],
            "existing",
            "db",
        )
    for lookup in ("off", "only"):
        with pytest.raises(ValueError):
            resolve(f"id:{recorded[0]}", "x86_64", lookup=lookup)


def test_lookup_off_reads_nothing_and_only_resolves_nothing_unrecorded():
    out = resolve(WHEEL, "ppc64le", client=Unreadable(), db="db", lookup="off")
    assert (out["source"], out["lookup"]) == ("derived", "none")
    assert resolve(WHEEL, "ppc64le", client=Recording(), db="db", lookup="only") is None
    assert resolve(WHEEL, "ppc64le", client=Recording(), db="db")["source"] == "derived"


def test_registry_off_never_calls_it_and_fails_when_the_spec_needs_it(monkeypatch):
    monkeypatch.setattr(Registry, "from_env", lambda: NoRegistry())
    with pytest.raises(NeedsRegistry):
        resolve(f"image:{IMAGE}:nightly-latest", "s390x", registry="off")
    url = "https://na.artifactory.swg-devops.com/artifactory/r/x.tgz"
    out = resolve(f"generic:{url}#{'ab' * 32};component=llvm", "x86_64", registry="off")
    assert out["source"] == "derived"


def test_a_complete_identity_makes_no_registry_call_and_is_never_rebound():
    aid = ArtifactId.derive("torch-spyre", "torch-spyre-dev", "9e28cf2e5c48", "x86_64")
    named_by = ";component=torch-spyre;name=torch-spyre-dev;id12=9e28cf2e5c48"
    for spec in (
        f"image:{IMAGE}:nightly-latest{named_by}",
        f"generic:https://h/x.tgz{named_by}",
    ):
        out = resolve(spec, "amd64", registry=NoRegistry(), lookup=FakeLookup())
        assert (out["artifact_id"], out["source"], out["arch"]) == (
            aid,
            "given",
            "x86_64",
        )
    identity = ArtifactIdentity(component="flex", artifact_name="ibm-flex", id12="c" * 12,
                                arch="x86_64", kind="rpm", ref=f"ibm-flex-*.{'c' * 12}.*.x86_64")  # fmt: skip
    out = resolve(identity, registry=NoRegistry(), lookup=FakeLookup())
    assert (out["artifact_id"], out["source"]) == (identity.artifact_id, "given")
    # A row filed under that id with other inputs is corrupt, not a match.
    bad = (
        aid,
        "torch-spyre",
        "torch-spyre-devel",
        "9e28cf2e5c48",
        "x86_64",
        "image",
        "",
    )
    with pytest.raises(ValueError):
        resolve(
            f"image:{IMAGE}:nightly-latest{named_by}",
            "x86_64",
            lookup=FakeLookup([bad]),
        )


def test_a_multi_arch_image_keeps_its_list_digest():
    out = resolve(
        f"image:{IMAGE}@{LIST}", "multi", registry=NoRegistry(), lookup=FakeLookup()
    )
    assert (out["artifact"], out["arch"], out["manifest_list"]) == (
        f"image:{IMAGE}@{LIST}",
        "multi",
        LIST,
    )
    assert (
        out["artifact_id"]
        == ArtifactIdentity.from_image(f"{IMAGE}@{LIST}", "multi").artifact_id
    )


def test_a_moving_ref_is_never_looked_up():
    db = Recording()
    reg = FakeRegistry({**LISTED, "nightly-latest": (LIST, LISTED[LIST][1])})
    resolve(f"image:{IMAGE}:nightly-latest", "s390x", client=db, db="db", registry=reg)
    assert db.params and not any("nightly-latest" in str(p) for p in db.params)
    db = Recording()
    assert resolve("wheel:apache-tvm-ffi", "ppc64le", client=db, db="db") is None
    assert db.params == []
    resolve("wheel:apache-tvm-ffi==0.1.14", "ppc64le", client=db, db="db")
    assert db.params[0]["refs"] == ["apache-tvm-ffi==0.1.14", "apache_tvm_ffi==0.1.14"]


def test_a_dry_run_reports_what_it_would_write_and_writes_nothing():
    client = FakeClient()
    out = ensure(
        client, "db", WHEEL, "ppc64le", tags=[("rc1", "release")], dry_run=True
    )
    assert (out["written"], out["dry_run"], out["tag"]) == (True, True, "rc1")
    assert client.tables == {"artifacts": [], "artifact_refs": [], "artifact_tags": []}
    out = ensure(client, "db", WHEEL, "ppc64le", tags=[("rc1", "release")])
    assert (out["written"], len(client.tables[ARTIFACTS.name])) == (True, 1)
    assert (
        ensure(client, "db", WHEEL, "ppc64le", tags=[("rc1", "release")], dry_run=True)[
            "written"
        ]
        is False
    )


# -- one write path ------------------------------------------------------------------------


def test_a_batch_artifact_writes_what_insert_artifact_wrote():
    entry = {
        "artifact": {"component": "flex", "artifact_name": "ibm-flex", "id12": "c" * 12,
                     "arch": "x86_64", "kind": "rpm", "ref": f"ibm-flex-*.{'c' * 12}.*.x86_64"},
        "origin": "built",
        "sources": [{"repo": "ai-chip-toolchain/flex", "git_ref": "main", "git_sha": "ab12"}],
        "identity_deps": ["base=x"],
        "context_deps": ["ctx"],
        "props": {"run_url": "https://ci/1"},
    }  # fmt: skip
    old, new = FakeClient(), FakeClient()
    ArtifactWriter.insert_artifact(
        old, "db", artifacts.batch_identity(entry["artifact"]), origin="built",
        sources=[("ai-chip-toolchain/flex", "main", "ab12")], identity_deps=["base=x"],
        context_deps=["ctx"], props={"run_url": "https://ci/1"},
    )  # fmt: skip
    assert artifacts.write_batch(new, "db", {"artifacts": [entry]})["artifacts"] == 1
    assert new.tables == old.tables


def test_a_batch_spec_entry_is_resolved_then_recorded():
    batch = {
        "artifacts": [
            {"spec": WHEEL, "arch": "ppc64le", "tag_family": "release", "tags": ["rc1"]}
        ]
    }
    client = FakeClient()
    assert artifacts.write_batch(client, "db", batch)["artifacts"] == 1
    assert [r["tag"] for r in client.tables[ARTIFACT_TAGS.name]] == ["rc1"]


def test_a_gha_record_writes_what_insert_gha_result_wrote():
    repo = ("torch-spyre/hf-adapters", "main", "fa67696108bd")
    old, new = FakeClient(), FakeClient()
    ArtifactWriter.insert_gha_result(
        old, "db", artifact_id=GHA_AID, component="hf-adapters", arch="x86_64", run_id=GHA_BASE,
        test_type="regression", state="passed", base_artifact_id=GHA_BASE, installed=GHA_INSTALLED,
        repo=repo[0], git_ref=repo[1], git_sha=repo[2], run_url="https://gha/1",
    )  # fmt: skip
    ensure(new, "db", GHA_RECORD, "x86_64", component="hf-adapters", sources=[repo],
           run_url="https://gha/1")  # fmt: skip
    assert new.tables[ARTIFACTS.name] == old.tables[ARTIFACTS.name]
    assert new.tables[ARTIFACT_REFS.name] == []


def test_a_rewrite_adds_no_ref_twice_and_a_later_tag_keeps_its_ref():
    entry = {"artifact": {"component": "flex", "artifact_name": "ibm-flex", "id12": "c" * 12, "arch": "amd64",
                          "kind": "rpm", "ref": f"ibm-flex-*.{'c' * 12}.*.x86_64"}}  # fmt: skip
    client = FakeClient()
    for _ in range(2):
        artifacts.write_batch(client, "db", {"artifacts": [entry]})
    (art,) = client.tables[ARTIFACTS.name]
    assert art["arch"] == "x86_64"
    assert len(client.tables[ARTIFACT_REFS.name]) == 1
    ArtifactWriter.insert_artifact(
        client, "db", artifacts.batch_identity(entry["artifact"])
    )
    assert len(client.tables[ARTIFACT_REFS.name]) == 1
    ensure(
        client,
        "db",
        artifacts.batch_identity(entry["artifact"]),
        tags=[("rc1", "release")],
    )
    (tag,) = client.tables[ARTIFACT_TAGS.name]
    assert [r[3] for r in tag["refs"]] == [entry["artifact"]["ref"]]
    assert len(client.tables[ARTIFACT_REFS.name]) == 1


def test_an_offline_dry_run_needs_no_database_and_shows_its_rows(monkeypatch, capsys):
    out = ensure(
        None,
        "",
        WHEEL,
        "ppc64le",
        lookup="off",
        tags=[("rc1", "release")],
        dry_run=True,
    )
    assert (out["written"], out["lookup"]) == (True, "none")
    assert [r["artifact_id"] for r in out["rows"][ARTIFACTS.name]] == [
        out["artifact_id"]
    ]
    assert [r["tag"] for r in out["rows"][ARTIFACT_TAGS.name]] == ["rc1"]
    with pytest.raises(ValueError):
        ensure(None, "", WHEEL, "ppc64le", lookup="off")
    for var in ("CLICKHOUSE_HOST", "CLICKHOUSE_DB_V2"):
        monkeypatch.delenv(var, raising=False)
    artifacts.main(
        [
            "ensure",
            "--artifact",
            WHEEL,
            "--arch",
            "ppc64le",
            "--lookup",
            "off",
            "--dry-run",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert (printed["dry_run"], printed["artifact_id"]) == (True, out["artifact_id"])
    assert (
        printed["rows"][ARTIFACT_REFS.name][0]["ref"]
        == "apache-tvm-ffi==0.1.14.post1+146f67a53e78"
    )
    assert "rows" not in resolve(WHEEL, "ppc64le", lookup="off").as_dict()


def test_a_gha_record_with_no_component_is_refused_by_name():
    with pytest.raises(ValueError, match="needs ;component="):
        resolve(GHA_RECORD, "x86_64", lookup="off")
    assert (
        resolve(f"{GHA_RECORD};component=hf-adapters", "x86_64", lookup="off")[
            "artifact_id"
        ]
        == GHA_AID
    )


def test_a_given_manifest_list_is_recorded_as_given_with_no_leaf():
    spec = f"image:{IMAGE}@{LIST};component=torch-spyre;name=torch-spyre-devel;id12={'9e' * 6}"
    client = FakeClient()
    out = ensure(client, "db", spec, "multi", origin="copied", registry=NoRegistry())
    (art,) = client.tables[ARTIFACTS.name]
    (ref,) = client.tables[ARTIFACT_REFS.name]
    assert (out["source"], art["origin"], art["arch"]) == ("given", "copied", "multi")
    assert (ref["ref"], ref["content_digest"]) == (f"{IMAGE}@{LIST}", LIST)
    assert out["artifact_id"] == ArtifactId.derive(
        "torch-spyre", "torch-spyre-devel", "9e" * 6, "multi"
    )


class Unauthorized(Registry):
    """icr.io with no credentials: every request is refused."""

    def _get(self, repo, path, accept=""):
        raise urllib.error.HTTPError(path, 401, "Unauthorized", {}, None)


@pytest.mark.parametrize("registry", ["off", Unauthorized()])
def test_with_no_registry_a_digest_finds_its_record_else_derives_as_given(registry):
    recorded = _row("torch-spyre", "torch-spyre-devel", LEAF[7:19], "s390x", "image")
    out = resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=registry,
                  lookup=FakeLookup([recorded], digest=recorded[0]))  # fmt: skip
    assert (out["artifact_id"], out["source"]) == (recorded[0], "existing")
    out = resolve(
        f"image:{IMAGE}@{LIST}", "s390x", registry=registry, lookup=FakeLookup()
    )
    # What the ingest recorded before it asked the registry: the digest it was given.
    assert (out["artifact_id"], out["source"], out["leaf"]) == (
        ArtifactIdentity.parse(f"image:{IMAGE}@{LIST}", "s390x", "x").artifact_id,
        "derived",
        "",
    )
    assert out["registry"].startswith("off" if registry == "off" else "unreachable")


def test_with_no_registry_a_content_addressed_tag_finds_its_record():
    tagged = f"icr.io/{REPO}:s390x-dev-28c3f5709879"
    recorded = _row("torch-spyre", "torch-spyre-dev", "28c3f5709879", "s390x", "image")
    lookup = FakeLookup([recorded], tag=recorded[0])
    out = resolve(f"image:{tagged}", "s390x", registry=Unauthorized(), lookup=lookup)
    assert (out["artifact_id"], out["source"]) == (recorded[0], "existing")
    assert ("tag", ("s390x", tagged)) in lookup.asked
    with pytest.raises(NeedsRegistry):
        resolve(f"image:{tagged}", "s390x", registry="off", lookup=FakeLookup())


class TagRows(Recording):
    """A database whose artifact_refs hold one pullspec under `rows`."""

    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def query(self, sql, parameters=None):
        out = super().query(sql, parameters)
        out.result_rows = self.rows
        return out


def test_a_tag_lookup_takes_only_a_content_addressed_tag_held_once():
    one = _row("torch-spyre", "torch-spyre-dev", "28c3f5709879", "s390x", "image")
    two = _row("torch-spyre", "torch-spyre-dev", "28c3f5709879", "s390x", "image", "x")
    spec = f"icr.io/{REPO}:s390x-dev-28c3f5709879"
    assert Lookup(TagRows([one]), "db").by_tag("s390x", spec) == one
    assert Lookup(TagRows([one, two]), "db").by_tag("s390x", spec) is None
    db = TagRows([one])
    assert Lookup(db, "db").by_tag("s390x", f"icr.io/{REPO}:nightly-latest") is None
    assert db.params == []


def test_ensure_records_a_digest_pinned_image_with_no_registry_credentials(monkeypatch):
    monkeypatch.setattr(Registry, "from_env", lambda: Unauthorized())
    client = FakeClient()
    out = ensure(client, "db", f"image:{IMAGE}@{LEAF}", "s390x", origin="promoted")
    assert (out["source"], out["written"]) == ("derived", True)
    assert (
        out["artifact_id"]
        == ArtifactIdentity.from_image(f"{IMAGE}@{LEAF}", "s390x").artifact_id
    )
