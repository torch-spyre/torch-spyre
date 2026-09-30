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

A release manifest is JSON:
    {"name": "<release name>", "date": "YYYY-MM-DD",
     "sources": [{"repo": "...", "ref": "v0.5.0-rc.1", "sha": "<40 hex>"}],
     "images": [{"ref": "<registry>/<repo>:<tag>@sha256:<per-arch digest>", "arch": "s390x"}],
     "generic": [{"url": "https://...", "sha256": "<hex>", "arch": "x86_64",
                  "component": "spyre-runtimes"}]}
Every artifact is tagged with the release's name and `release-<date>` (family `release`),
plus the rolling `release` pointer.
"""

import argparse
import json
import sys
from pathlib import Path

from .identity import ArtifactIdentity
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
        (name, RELEASE_FAMILY, tag_props),
        (f"release-{date}", RELEASE_FAMILY, tag_props),
        ("release", RELEASE_FAMILY, tag_props),
    ]
    return [
        ArtifactWriter.insert_artifact(
            client,
            db,
            identity,
            origin="promoted",
            sources=sources,
            props={"release": name, "run_url": run_url, "source": "release"},
            tags=tags,
        )
        for identity in release_identities(manifest)
    ]


def _source(value: str) -> tuple:
    """`repo@ref@sha` or `repo@sha` -> (repo, git_ref, git_sha)."""
    parts = value.split("@")
    if len(parts) == 2:
        return parts[0], "", parts[1]
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError(f"--source wants repo@[ref@]sha, got {value!r}")


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

    args = parser.parse_args(argv)
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
