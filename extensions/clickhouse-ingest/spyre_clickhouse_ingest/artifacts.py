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
    python -m spyre_clickhouse_ingest.artifacts resolve --artifact <spec> --arch s390x \
        [--tag-family nightly-supply-chain] [--tag-date YYYY-MM-DD] [--tag <tag> ...]
    python -m spyre_clickhouse_ingest.artifacts ensure --artifact <spec> --arch s390x ...

`resolve` prints, as JSON, the one artifact a spec (resolver.py's grammar) names: the existing
spyre_v2 record when there is one (read-only, from CLICKHOUSE_* + CLICKHOUSE_DB_V2; without
them `lookup` is "none" and the id is derived), for an image its per-arch leaf, and its
`tag` / `tag_family` (and every pair in `tags`). `ensure` does the same and records it,
tagged (resolver.ensure_artifact). Registry
access: ICR_USERNAME / ICR_PASSWORD; Artifactory: ARTIFACTORY_USER / ARTIFACTORY_TOKEN.

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
from datetime import date, datetime, timezone
from itertools import zip_longest
from pathlib import Path

from .identity import ArtifactIdentity, DerivedId
from .registry import DATED, FAMILIES, Registry
from .resolver import Lookup, ensure_artifact, resolve
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

    res = sub.add_parser("resolve", help="the artifact a spec names (JSON)")
    ens = sub.add_parser("ensure", help="resolve a spec and record it, tagged (JSON)")
    for p in (res, ens):
        what = p.add_mutually_exclusive_group(required=True)
        what.add_argument(
            "--artifact", help="image:/rpm:/wheel:/generic: spec or an artifact_id"
        )
        what.add_argument("--image", help="shorthand for --artifact image:<ref>")
        p.add_argument("--arch", required=True)
        p.add_argument(
            "--tag-family",
            default="",
            choices=["", *FAMILIES, RELEASE_FAMILY],
            help="the family to tag it in: its tag there comes from the registry, else from "
            "--tag-date; also the family of each --tag whose prefix names none",
        )
        p.add_argument(
            "--tag",
            action="append",
            default=[],
            help="repeatable full tag; one in --tag-family (given, or its prefix's) replaces "
            "the resolved tag, any other is added",
        )
        p.add_argument(
            "--tag-date",
            type=date.fromisoformat,
            default=None,
            help="the run's day: orders the registry search and dates the tag when the registry "
            "has none; weekly tags take its ISO week (default with a dated --tag-family: "
            "today, UTC)",
        )
    res.add_argument(
        "--no-lookup", action="store_true", help="derive only; read no database"
    )
    ens.add_argument("--origin", default="built")
    ens.add_argument("--run-url", default="")
    ens.add_argument(
        "--identity-dep", action="append", default=[], help="e.g. base=<sha256>"
    )

    args = parser.parse_args(argv)
    if args.cmd in ("resolve", "ensure"):
        spec = args.artifact or "image:" + args.image.removeprefix("image:")
        registry = Registry(
            username=os.environ.get("ICR_USERNAME", ""),
            password=os.environ.get("ICR_PASSWORD", ""),
        )
        from .client import ClickHouse, ClickHouseEnv

        db = args.database or ClickHouseEnv.target_database()
        if args.tag_family in DATED and args.tag_date is None:
            args.tag_date = datetime.now(timezone.utc).date()
        if args.cmd == "resolve":
            lookup = Lookup()
            if not args.no_lookup and db and ClickHouseEnv.host():
                try:
                    client = ClickHouse.connect(database=db)
                    client.set_client_setting("readonly", "2")
                    client.query("SELECT 1")
                    lookup = Lookup(client, db)
                except Exception as err:  # noqa: BLE001 -- derive-only is the documented fallback
                    print(
                        f"[warn] no spyre_v2 lookup ({err}); deriving only",
                        file=sys.stderr,
                    )
                    lookup = Lookup()
            out = resolve(
                spec, args.arch, lookup=lookup, registry=registry,
                tag_family=args.tag_family, tag_date=args.tag_date, tags=args.tag,
            )  # fmt: skip
            out.pop("identity", None)
            print(json.dumps(out, sort_keys=True))
            if not out:
                sys.exit(1)
            return
        if not db:
            sys.exit("[error] no database: pass --database or set CLICKHOUSE_DB_V2")
        identity = ensure_artifact(
            ClickHouse.connect(database=db), db, spec, args.arch,
            origin=args.origin, tags=args.tag, tag_family=args.tag_family,
            tag_date=args.tag_date, run_url=args.run_url,
            identity_deps=args.identity_dep, registry=registry,
        )  # fmt: skip
        print(
            json.dumps(
                {
                    "artifact_id": identity.artifact_id,
                    "artifact": f"{identity.kind}:{identity.ref}",
                }
            )
        )
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
