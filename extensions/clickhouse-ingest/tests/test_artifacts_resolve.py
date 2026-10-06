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

"""`artifacts resolve`: one tested image, one artifact id and tag, whichever writer asks."""

import json
import urllib.error
from datetime import date

import pytest

from spyre_clickhouse_ingest import ArtifactIdentity
from spyre_clickhouse_ingest import artifacts
from spyre_clickhouse_ingest.registry import Registry, channel_tag, resolve

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
    """Serves manifests and tags from dicts, never the network."""

    def __init__(self, served, tags=()):
        super().__init__()
        self.served = served
        self._tags = {REPO: list(tags)}

    def _get(self, repo, path, accept=""):
        ref = path.split("/", 1)[1]
        if ref not in self.served:
            raise urllib.error.HTTPError(path, 404, "not found", {}, None)
        return self.served[ref]


LISTED = {LIST: (LIST, _index(("ppc64le", OTHER), ("s390x", LEAF))), LEAF: (LEAF, {})}


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


def test_a_manifest_list_and_its_leaf_resolve_to_one_artifact():
    reg = FakeRegistry(LISTED)
    by_list, by_leaf = (
        resolve(reg, f"{IMAGE}@{LIST}", "s390x"),
        resolve(reg, f"{IMAGE}@{LEAF}", "s390x"),
    )
    assert by_list["artifact_id"] == by_leaf["artifact_id"]
    assert by_list["artifact"] == f"image:{IMAGE}@{LEAF}"
    assert (by_list["manifest_list"], by_leaf["manifest_list"]) == (LIST, "")


def test_the_id_is_what_register_and_ingest_record_for_the_same_image():
    out = resolve(FakeRegistry(LISTED), f"image:{IMAGE}:snap-latest@{LIST}", "s390x")
    # artifacts register / ingest_xml --artifact parse the spec; the image build registers
    # its dev-registry tagged ref, which names the same per-arch digest.
    for spec in (
        out["artifact"],
        f"image:icr.io/ai_sw_accel_dev/torch-spyre/torch-spyre-devel:snap-latest@{LEAF}",
    ):
        assert (
            ArtifactIdentity.parse(spec, "s390x", "hf-adapters").artifact_id
            == out["artifact_id"]
        )


def test_the_snap_builds_own_tag_beats_the_days_aggregate():
    served = {
        **LISTED,
        "snap-20261004": (LIST, _index(("s390x", LEAF))),
        "snap-20261004T002701_277": (LIST, _index(("s390x", LEAF))),
    }
    tags = ["snap-20261003T000000_1", "snap-20261004", "snap-20261004T002701_277"]
    out = resolve(FakeRegistry(served, tags), f"{IMAGE}@{LEAF}", "s390x", "snap", DAY)
    assert (out["tag"], out["tag_family"]) == (
        "snap-supply-chain-2026-10-04T002701",
        "snap-supply-chain",
    )
    assert out["registry_tag"] == "snap-20261004T002701_277"


def test_a_dated_run_stays_in_its_channel_when_the_registry_has_no_tag():
    served = {**LISTED, "ci-cd-tech-preview-v3": (LEAF, {})}
    reg = FakeRegistry(served, ["ci-cd-tech-preview-v3"])
    out = resolve(reg, f"{IMAGE}@{LEAF}", "s390x", "snap", date(2026, 10, 6))
    assert (out["tag"], out["tag_family"]) == (
        "snap-supply-chain-2026-10-06",
        "snap-supply-chain",
    )
    # Without a day the registry's own channel tag names it.
    out = resolve(reg, f"{IMAGE}@{LEAF}", "s390x", "nightly")
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
    out = resolve(
        FakeRegistry(served, ["weekly-W39", "weekly-W40"]),
        f"{IMAGE}@{LIST}",
        "s390x",
        "weekly",
    )
    assert out["tag"] == "weekly-supply-chain-2026-w40"


def test_an_unresolvable_image_resolves_to_nothing():
    assert resolve(FakeRegistry({}), f"{IMAGE}@{LEAF}", "s390x") == {}
    assert resolve(FakeRegistry(LISTED), f"quay.io/{REPO}@{LEAF}", "s390x") == {}
    assert resolve(FakeRegistry(LISTED), f"{IMAGE}@{LIST}", "x86_64") == {}


def test_the_cli_prints_the_resolution_as_json(monkeypatch, capsys):
    monkeypatch.setattr(artifacts, "Registry", lambda **kw: FakeRegistry(LISTED))
    artifacts.main(["resolve", "--image", f"{IMAGE}@{LIST}", "--arch", "s390x"])
    out = json.loads(capsys.readouterr().out)
    assert out["artifact"] == f"image:{IMAGE}@{LEAF}"
    with pytest.raises(SystemExit):
        artifacts.main(["resolve", "--image", f"{IMAGE}@{OTHER}", "--arch", "s390x"])
