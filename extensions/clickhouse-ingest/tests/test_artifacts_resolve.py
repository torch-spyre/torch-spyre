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
from datetime import date

import pytest

from spyre_clickhouse_ingest import ArtifactIdentity, artifacts
from spyre_clickhouse_ingest.identity import ArtifactId
from spyre_clickhouse_ingest.registry import Registry, channel_tag
from spyre_clickhouse_ingest.resolver import Lookup, ensure_artifact, resolve
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

# -- channel tags --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channel, tag, built, expected",
    [
        ("nightly", "nightly-20261004", None, "nightly-supply-chain-2026-10-04"),
        (
            "snap",
            "snap-20261004T002701_277",
            None,
            "snap-supply-chain-2026-10-04T002701",
        ),
        ("snap", "snap-20261005", None, "snap-supply-chain-2026-10-05"),
        ("weekly", "weekly-W40", DAY, "weekly-supply-chain-2026-w40"),
        ("weekly", "weekly-W01", date(2026, 12, 30), "weekly-supply-chain-2027-w01"),
        ("ci-cd-tech-preview", "ci-cd-tech-preview-v3", None, "ci-cd-tech-preview-v3"),
    ],
)
def test_each_registry_channel_tag_has_one_v2_name(channel, tag, built, expected):
    assert channel_tag(channel, tag, built) == expected


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
                  channel="snap", day=DAY)  # fmt: skip
    assert (out["tag"], out["tag_family"], out["registry_tag"]) == (
        "snap-supply-chain-2026-10-04T002701",
        "snap-supply-chain",
        "snap-20261004T002701_277",
    )


def test_a_dated_run_stays_in_its_channel_when_the_registry_has_no_tag():
    reg = FakeRegistry(
        {**LISTED, "ci-cd-tech-preview-v3": (LEAF, {})}, ["ci-cd-tech-preview-v3"]
    )
    out = resolve(
        f"image:{IMAGE}@{LEAF}",
        "s390x",
        registry=reg,
        channel="snap",
        day=date(2026, 10, 6),
    )
    assert (out["tag"], out["tag_family"]) == (
        "snap-supply-chain-2026-10-06",
        "snap-supply-chain",
    )
    out = resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=reg, channel="nightly")
    assert (out["tag"], out["tag_family"]) == (
        "ci-cd-tech-preview-v3",
        "ci-cd-tech-preview",
    )


def test_a_weekly_tag_takes_the_iso_year_of_the_image_build():
    served = {
        **LISTED,
        "weekly-W40": (LIST, _index(("s390x", LEAF))),
        LEAF: (LEAF, {"config": {"digest": "sha256:cfg"}}),
        "sha256:cfg": ("", {"created": "2026-10-01T08:00:00Z"}),
    }
    out = resolve(f"image:{IMAGE}@{LIST}", "s390x",
                  registry=FakeRegistry(served, ["weekly-W39", "weekly-W40"]), channel="weekly")  # fmt: skip
    assert out["tag"] == "weekly-supply-chain-2026-w40"


def test_an_unresolvable_image_resolves_to_nothing():
    assert resolve(f"image:{IMAGE}@{LEAF}", "s390x", registry=FakeRegistry({})) == {}
    assert (
        resolve(f"image:{IMAGE}@{LIST}", "x86_64", registry=FakeRegistry(LISTED)) == {}
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


def test_a_generic_file_is_found_by_url_else_derived_from_its_sha():
    url = "https://na.artifactory.swg-devops.com/artifactory/r/next/noarch/llvm/llvm-src-080ddeea9a07.tgz"
    recorded = _row(
        "llvm", "llvm-src-080ddeea9a07.tgz", "080ddeea9a07", "x86_64", "generic", url
    )
    assert (
        resolve(
            f"generic:{url}", "x86_64", lookup=FakeLookup([recorded], refs=recorded[0])
        )["artifact_id"]
        == recorded[0]
    )
    assert resolve(f"generic:{url}", "x86_64", lookup=FakeLookup()) == {}
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
        == {}
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


def test_an_unknown_spec_is_refused():
    with pytest.raises(ValueError):
        resolve("tarball:x", "s390x")


# -- ensure_artifact -----------------------------------------------------------------------


class FakeClient:
    """A tiny in-memory spyre_v2: counts answer from what was inserted."""

    def __init__(self):
        self.tables = {"artifacts": [], "artifact_refs": [], "artifact_tags": []}

    def insert(self, table, rows, column_names=None, database=None):
        self.tables[table] += [dict(zip(column_names, r)) for r in rows]

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
                                   tags=[("rc1", "release")], channel="nightly", registry=reg)  # fmt: skip
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
    monkeypatch.setattr(artifacts, "Registry", lambda **kw: FakeRegistry(LISTED))
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
