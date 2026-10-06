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

"""Register artifacts that no test ingest names: one built image, or a whole release.

    python -m spyre_clickhouse_ingest.artifacts register --artifact image:<ref>@<digest> ...
    python -m spyre_clickhouse_ingest.artifacts release <manifest.json>
    python -m spyre_clickhouse_ingest.artifacts write <batch.json>
    python -m spyre_clickhouse_ingest.artifacts resolve --image <repo>@<digest> --arch s390x

`resolve` reads only the registry (ICR_USERNAME / ICR_PASSWORD) and prints, as JSON, the
artifact a tested image is: artifact_id, the per-arch leaf `artifact` spec to register or
ingest it by, and its supply-chain tag and tag_family (registry.resolve).

A batch is what a pipeline writer (the Jenkins orchestrator) hands over in one call; every
entry names its artifact by the four hash inputs plus kind and ref:
    {"artifacts": [{"artifact": {...}, "origin": "built", "sources": [{"repo", "git_ref",
                    "git_sha"}], "identity_deps": [], "context_deps": [], "props": {}}],
     "tags": [{"artifact": {...}, "tag": "...", "tag_family": "...", "ref": "", "props": {}}],
     "results": [{"artifact": {...}, "run_id": "<uuid>", "test_type": "unit", "state": "passed",
                  "arch": "", "result_kind": "", "duration_s": 0.0, "props": {}}]}
with "artifact" = {"component", "artifact_name", "id12", "arch", "kind", "ref"}.

A release manifest is JSON:
    {"name": "<release name>", "date": "YYYY-MM-DD", "family": "<tag family of name>",
     "sources": [{"repo": "...", "ref": "v0.5.0-rc.1", "sha": "<40 hex>"}],
     "images": [{"ref": "<registry>/<repo>:<tag>@sha256:<per-arch digest>", "arch": "s390x"},
                {"ref": "<registry>/<repo>:<tag>@sha256:<manifest-list digest>", "arch": "multi",
                 "manifests": {"s390x": "sha256:<per-arch digest>", ...}}],
     "generic": [{"url": "https://...", "sha256": "<hex>", "arch": "x86_64",
                  "component": "spyre-runtimes"}]}
Every artifact is tagged with the release's name (in `family`, default `release`), plus
`release-<date>` and the rolling `release` pointer (both family `release`). List each
manifest list beside its per-arch images: a consumer that pulled by tag holds only its digest.
"""

import argparse
import json
import os
import sys
from datetime import date
from itertools import zip_longest
from pathlib import Path

from .identity import ArtifactIdentity, DerivedId
from .registry import CHANNELS, Registry, resolve
from .writer import ArtifactWriter

RELEASE_FAMILY = "release"


def release_identities(manifest: dict) -> list:
    """Every artifact a release manifest names, as ArtifactIdentity values."""
    out = [
        ArtifactIdentity.from_image(i["ref"], i["arch"], i.get("component", ""))
        for i in manifest.get("images", [])
    ]
    out += [
        ArtifactIdentity.from_generic(
            g["url"], g["sha256"], g["component"], g["arch"], g.get("name", "")
        )
        for g in manifest.get("generic", [])
    ]
    return out


def register_release(client, db: str, manifest: dict, run_url: str = "") -> list:
    """Record and tag every artifact of `manifest`; returns their artifact_ids."""
    name, date = manifest["name"], manifest["date"]
    sources = [
        (s["repo"], s.get("ref", ""), s["sha"]) for s in manifest.get("sources", [])
    ]
    tag_props = {"run_url": run_url, "source": "release"}
    tags = [
        (name, manifest.get("family") or RELEASE_FAMILY, tag_props),
        (f"release-{date}", RELEASE_FAMILY, tag_props),
        ("release", RELEASE_FAMILY, tag_props),
    ]
    # A manifest list's content is its per-arch images, recorded as `<arch>=<digest>`.
    deps = [
        [f"{a}={d}" for a, d in sorted(i.get("manifests", {}).items())]
        for i in manifest.get("images", [])
    ]
    return [
        ArtifactWriter.insert_artifact(
            client,
            db,
            identity,
            origin="promoted",
            sources=sources,
            identity_deps=identity_deps,
            props={"release": name, "run_url": run_url, "source": "release"},
            tags=tags,
        )
        for identity, identity_deps in zip_longest(
            release_identities(manifest), deps, fillvalue=[]
        )
    ]


def _source(value: str) -> tuple:
    """`repo@ref@sha` or `repo@sha` -> (repo, git_ref, git_sha)."""
    parts = value.split("@")
    if len(parts) == 2:
        return parts[0], "", parts[1]
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError(f"--source wants repo@[ref@]sha, got {value!r}")


# The keys each batch entry may carry. An unknown key is refused: a misspelled one would
# otherwise leave its column defaulted, and the row would land looking valid.
BATCH_KEYS = {
    "artifact": {"component", "artifact_name", "id12", "arch", "kind", "ref"},
    "artifacts": {
        "artifact",
        "origin",
        "sources",
        "identity_deps",
        "context_deps",
        "props",
    },
    "tags": {"artifact", "tag", "tag_family", "ref", "props"},
    "results": {
        "artifact",
        "run_id",
        "test_type",
        "state",
        "arch",
        "result_kind",
        "duration_s",
        "props",
    },
}


def check_batch(batch: dict) -> None:
    """Raise ValueError naming every unknown section or key in `batch`."""
    bad = [f"section {k!r}" for k in batch if k not in ("artifacts", "tags", "results")]
    for section in ("artifacts", "tags", "results"):
        for i, e in enumerate(batch.get(section, [])):
            where = f"{section}[{i}]"
            bad += [f"{where}.{k}" for k in e if k not in BATCH_KEYS[section]]
            bad += [
                f"{where}.artifact.{k}"
                for k in e.get("artifact", {})
                if k not in BATCH_KEYS["artifact"]
            ]
    if bad:
        raise ValueError("unknown batch key(s): " + ", ".join(bad))


def batch_identity(a: dict) -> ArtifactIdentity:
    """The `artifact` object of a batch entry."""
    return ArtifactIdentity(
        component=a.get("component", ""),
        artifact_name=a.get("artifact_name", ""),
        id12=a.get("id12", ""),
        arch=DerivedId.arch(a.get("arch", "")),
        kind=a.get("kind") or "image",
        ref=a.get("ref", ""),
    )


def write_batch(client, db: str, batch: dict) -> dict:
    """Write every entry of `batch`; returns how many of each landed or already existed."""
    check_batch(batch)
    done = {"artifacts": 0, "tags": 0, "results": 0}
    for e in batch.get("artifacts", []):
        sources = [
            (s.get("repo", ""), s.get("git_ref", ""), s.get("git_sha", ""))
            for s in e.get("sources", [])
        ]
        done["artifacts"] += bool(
            ArtifactWriter.insert_artifact(
                client,
                db,
                batch_identity(e["artifact"]),
                origin=e.get("origin") or "built",
                sources=sources,
                identity_deps=e.get("identity_deps", []),
                context_deps=e.get("context_deps", []),
                props=e.get("props"),
            )
        )
    for e in batch.get("tags", []):
        done["tags"] += ArtifactWriter.insert_tag(
            client,
            db,
            batch_identity(e["artifact"]),
            e.get("tag", ""),
            e.get("tag_family", ""),
            ref=e.get("ref") or None,
            props=e.get("props"),
        )
    for e in batch.get("results", []):
        identity = batch_identity(e["artifact"])
        done["results"] += ArtifactWriter.insert_result(
            client,
            db,
            artifact_id=identity.artifact_id,
            run_id=e.get("run_id", ""),
            test_type=e.get("test_type", ""),
            state=e.get("state", ""),
            arch=e.get("arch") or identity.arch,
            result_kind=e.get("result_kind", ""),
            duration_s=e.get("duration_s") or 0.0,
            props=e.get("props"),
        )
    return done


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="", help="default: $CLICKHOUSE_DB_V2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser(
        "register", help="one artifact, e.g. an image a build just pushed"
    )
    reg.add_argument(
        "--artifact",
        required=True,
        help="image:<ref>@<digest> | generic:<url>#<sha256>",
    )
    reg.add_argument("--arch", required=True)
    reg.add_argument(
        "--component",
        default="",
        help="the artifact's component; required for generic, else guessed from the image name",
    )
    reg.add_argument("--origin", default="built")
    reg.add_argument("--source", action="append", type=_source, default=[])
    reg.add_argument("--identity-dep", action="append", default=[])
    reg.add_argument("--run-url", default="")
    reg.add_argument("--tag", action="append", default=[], help="repeatable")
    reg.add_argument("--tag-family", default=RELEASE_FAMILY)

    rel = sub.add_parser("release", help="every artifact of a release manifest")
    rel.add_argument("manifest", type=Path)
    rel.add_argument("--run-url", default="")
    rel.add_argument("--dry-run", action="store_true", help="print the identities only")

    wr = sub.add_parser("write", help="a batch of artifacts, tags and results (JSON)")
    wr.add_argument("batch", type=Path)

    res = sub.add_parser("resolve", help="a tested image's artifact id and tag (JSON)")
    res.add_argument(
        "--image", required=True, help="[image:]<host>/<repo>[:tag]@<digest>"
    )
    res.add_argument("--arch", required=True)
    res.add_argument("--channel", default="", choices=["", *CHANNELS])
    res.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="the run's day: orders the search, and names the tag when no registry tag does",
    )
    res.add_argument("--name", default="", help="ci-cd-tech-preview: the release name")

    args = parser.parse_args(argv)
    if args.cmd == "resolve":
        registry = Registry(
            username=os.environ.get("ICR_USERNAME", ""),
            password=os.environ.get("ICR_PASSWORD", ""),
        )
        out = resolve(
            registry, args.image, args.arch, args.channel, args.date, args.name
        )
        print(json.dumps(out, sort_keys=True))
        if not out:
            sys.exit(1)
        return
    if args.cmd == "release" and args.dry_run:
        for i in release_identities(json.loads(args.manifest.read_text())):
            print(
                f"  {i.artifact_id}  {i.kind:7} {i.arch:7} {i.artifact_name} {i.id12}"
            )
        return

    from .client import ClickHouse, ClickHouseEnv

    db = args.database or ClickHouseEnv.target_database()
    if not db:
        sys.exit("[error] no database: pass --database or set CLICKHOUSE_DB_V2")
    client = ClickHouse.connect(database=db)
    if args.cmd == "write":
        done = write_batch(client, db, json.loads(args.batch.read_text()))
        print(f"[info] {db}: " + ", ".join(f"{v} {k}" for k, v in done.items()))
        return
    if args.cmd == "release":
        ids = register_release(
            client, db, json.loads(args.manifest.read_text()), args.run_url
        )
        print(f"[info] {len([i for i in ids if i])} artifact(s) registered in {db}")
        return
    spec = args.artifact
    if args.component and spec.startswith("image:") and ";component=" not in spec:
        spec += f";component={args.component}"
    identity = ArtifactIdentity.parse(spec, args.arch, args.component)
    if identity is None:
        sys.exit(f"[error] --artifact {args.artifact!r} names no identity")
    tag_props = {"run_url": args.run_url}
    aid = ArtifactWriter.insert_artifact(
        client,
        db,
        identity,
        origin=args.origin,
        sources=args.source,
        identity_deps=args.identity_dep,
        props={"run_url": args.run_url, "source": "jenkins"},
        tags=[(t, args.tag_family, tag_props) for t in args.tag],
    )
    if not aid:
        sys.exit(1)
    print(aid)


if __name__ == "__main__":
    main()
