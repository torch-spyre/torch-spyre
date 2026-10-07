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

"""Any artifact spec -> its one spyre_v2 artifact: the existing record if there is one, else
the identity registration would derive. Every writer that names an artifact goes through here
(`resolve`, `ensure_artifact`), so one artifact gets one id whichever writer saw it first.

Spec grammar (each optionally followed by `;component=`, `;name=`, `;id12=`):
    image:<host>/<repo>[:tag][@sha256:<digest>]   a tag, manifest list or per-arch leaf
    rpm:<file name | URL | NEVRA | dnf glob>
    wheel:<name==version | file name | URL>
    generic:<url>[#<sha256>]
    gha:<artifact_id>|<base_artifact_id>|<installed>  derive-gha-artifact-id's record
    <artifact_id>                                  a bare uuid, which must exist

Order: normalise (image -> per-arch leaf and its labels; Artifactory file -> its sha256 and
`spyre.identity` property), then look the artifact up in spyre_v2 (a bare id; the image's
`spyre.artifact.id` label; its refs: the exact ref, the leaf digest, an rpm glob matching the
file; then name + id12), and only when none matches derive it with ArtifactIdentity.from_*.
Orchestrator-built artifacts hash the orchestrator's inputs into id12, so for them only the
lookup (or the label) reproduces the recorded id.

Tags: `tag_family` (snap-supply-chain, nightly-supply-chain, weekly-supply-chain,
ci-cd-tech-preview, release) names the artifact by that family's tag in the registry, else by
`tag_date` (weekly: its ISO week); full `tags` replace it when in that family, else add to it.
"""

import base64
import dataclasses
import fnmatch
import json
import os
import urllib.error
import urllib.request
import uuid

from . import schema
from .identity import ID_SEP, ArtifactIdentity, DerivedId
from .registry import RELEASE, Registry, dated_tag, family_of, split_image, tag_of
from .writer import ArtifactWriter

KINDS = ("image", "rpm", "wheel", "generic")
# Every spelling an arch is stored under; the identity hash folds them, the columns do not.
ARCH_ALIASES = {"x86_64": ("x86_64", "amd64", "x86", "x86-64")}
# The orchestrator's image labels: its own record of the artifact it built.
LABEL_ID, LABEL_ID12, LABEL_NAME = (
    "spyre.artifact.id",
    "spyre.artifact.id12",
    "spyre.artifact.name",
)
FILE_IDENTITY_PROP = "spyre.identity"


def _aliases(arch: str) -> list:
    a = DerivedId.arch(arch)
    return list(ARCH_ALIASES.get(a, (a,)))


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class Artifactory:
    """Read-only Artifactory storage API: a file's sha256 and properties, by its URL."""

    def __init__(self, user: str = "", token: str = ""):
        self.user, self.token = user, token

    @classmethod
    def from_env(cls) -> "Artifactory":
        return cls(
            os.environ.get("ARTIFACTORY_USER", ""),
            os.environ.get("ARTIFACTORY_TOKEN", ""),
        )

    def info(self, url: str) -> dict:
        """{'sha256', 'properties'} of the file at `url`; {} when it is not an Artifactory URL
        or cannot be read."""
        head, sep, path = url.partition("/artifactory/")
        if not (sep and path) or not self.token:
            return {}
        storage = f"{head}/artifactory/api/storage/{path.replace('+', '%2B')}"
        auth = (
            "Basic " + base64.b64encode(f"{self.user}:{self.token}".encode()).decode()
            if self.user
            else f"Bearer {self.token}"
        )
        out: dict = {}
        for query, key in (("", "checksums"), ("?properties", "properties")):
            req = urllib.request.Request(
                storage + query, headers={"Authorization": auth}
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    out[key] = json.load(r).get(key) or {}
            except (urllib.error.URLError, ValueError, OSError):
                out[key] = {}
        return {
            "sha256": (out["checksums"] or {}).get("sha256", ""),
            "properties": {
                k: (v or [""])[0] for k, v in (out["properties"] or {}).items()
            },
        }


class Lookup:
    """Existing spyre_v2 artifacts, read on `client`; a no-op without one."""

    COLS = (
        "toString(artifact_id), component, artifact_name, props['id12'], arch, kind, "
        "props['ref']"
    )

    def __init__(self, client=None, db: str = ""):
        self.client, self.db = client, db

    def _one(self, where: str, params: dict):
        if self.client is None:
            return None
        rows = self.client.query(
            f"SELECT {self.COLS} FROM {self.db}.artifacts WHERE {where} "
            "ORDER BY ts DESC LIMIT 1",
            parameters=params,
        ).result_rows
        return rows[0] if rows else None

    def _by_ref(self, ref_where: str, kind: str, arch: str, params: dict):
        return self._one(
            f"artifact_id IN (SELECT artifact_id FROM {self.db}.artifact_refs "
            f"WHERE {ref_where}) AND kind = {{kind:String}} AND arch IN {{arch:Array(String)}}",
            {**params, "kind": kind, "arch": _aliases(arch)},
        )

    def by_id(self, aid: str):
        return self._one("artifact_id = {a:UUID}", {"a": aid})

    def by_refs(self, kind: str, arch: str, refs: list):
        refs = [r for r in refs if r]
        return refs and self._by_ref(
            "ref IN {refs:Array(String)}", kind, arch, {"refs": refs}
        )

    def by_digest(self, arch: str, digest: str):
        return digest and self._by_ref(
            "content_digest = {d:String} OR endsWith(ref, {at:String})",
            "image",
            arch,
            {"d": digest, "at": "@" + digest},
        )

    def by_rpm_file(self, arch: str, filename: str, rpm_name: str):
        """The record whose dnf glob (`<name>-*.<id12>.*.<arch>`) matches the file. The glob's
        name must be the file's whole package name: comms and comms-devel share one id12."""
        if self.client is None or not (filename and rpm_name):
            return None
        globs = self.client.query(
            f"SELECT DISTINCT ref FROM {self.db}.artifact_refs "
            "WHERE startsWith(ref, {p:String}) AND position(ref, '*') > 0",
            parameters={"p": rpm_name + "-*"},
        ).result_rows
        hits = [
            g for (g,) in globs if fnmatch.fnmatchcase(filename.removesuffix(".rpm"), g)
        ]
        return hits and self._by_ref(
            "ref IN {refs:Array(String)}", "rpm", arch, {"refs": hits}
        )

    def by_id12(self, kind: str, arch: str, name: str, id12: str):
        return (
            name
            and id12
            and self._one(
                "kind = {kind:String} AND artifact_name = {n:String} "
                "AND props['id12'] = {i:String} AND arch IN {arch:Array(String)}",
                {"kind": kind, "n": name, "i": id12, "arch": _aliases(arch)},
            )
        )


def _identity_of(row) -> ArtifactIdentity:
    """A recorded artifact as an identity; refuses a row whose inputs do not hash to its id."""
    aid, component, name, id12, arch, kind, ref = row
    identity = ArtifactIdentity(
        component=component,
        artifact_name=name,
        id12=id12,
        arch=DerivedId.arch(arch),
        kind=kind,
        ref=ref,
    )
    if identity.artifact_id != aid:
        raise ValueError(
            f"artifact {aid}: its recorded inputs hash to {identity.artifact_id}"
        )
    return identity


def _options(spec: str) -> tuple:
    body, *opts = spec.split(";")
    over = dict(o.split("=", 1) for o in opts if "=" in o)
    if set(over) - {"component", "name", "id12"} or len(over) != len(opts):
        raise ValueError(f"unknown or malformed option(s) in {spec!r}")
    return body, over


def resolve(
    spec: str,
    arch: str,
    *,
    lookup: Lookup | None = None,
    registry: Registry | None = None,
    artifactory: Artifactory | None = None,
    tag_family: str = "",
    tag_date=None,
    tags=(),
    component: str = "",
) -> dict:
    """The artifact `spec` names on `arch`, as a dict (see the module docstring); {} when it
    names nothing. `identity` is the ArtifactIdentity; every other value is JSON-ready.

    `tag`/`tag_family` name it in `tag_family` (see `named`); `tags` lists every
    [tag, tag_family] pair: that one plus the rest of `tags`."""
    lookup = lookup or Lookup()
    spec = (spec or "").strip()
    kind, sep, rest = spec.partition(":")
    out = {"lookup": "db" if lookup.client is not None else "none"}
    if not sep and _is_uuid(spec):
        row = lookup.by_id(spec)
        if not row:
            return {}
        result = _result(out, _identity_of(row), "existing", spec)
        return _named(result, dated_tag(tag_family, tag_date), tag_family, tags)
    if kind == "gha":
        return _gha(out, rest, arch, lookup, component)
    if kind not in KINDS:
        raise ValueError(
            f"spec must be image:, rpm:, wheel:, generic:, gha: or an artifact_id: {spec!r}"
        )
    body, over = _options(rest)
    if kind != "image":
        # An image names its own component; a file takes the caller's unless it says one.
        over.setdefault("component", component)
    if kind == "image":
        return _image(
            out,
            body,
            over,
            arch,
            lookup,
            registry or Registry.from_env(),
            tag_family,
            tag_date,
            tags,
        )
    file_url = body.split("#", 1)[0] if body.startswith("http") else ""
    info = (artifactory or Artifactory.from_env()).info(file_url) if file_url else {}
    prop_id12 = (info.get("properties", {}).get(FILE_IDENTITY_PROP) or "")[:12]
    filename = body.split("#", 1)[0].rsplit("/", 1)[-1]
    args = (
        over.get("component", ""),
        over.get("name", ""),
        over.get("id12", "") or prop_id12,
    )
    try:
        if kind == "rpm":
            derived = ArtifactIdentity.from_rpm(file_url or body, arch, *args)
        elif kind == "wheel":
            derived = ArtifactIdentity.from_wheel(body, arch, *args)
        else:
            url, _, sha = body.partition("#")
            derived = ArtifactIdentity.from_generic(
                url, sha or info.get("sha256", ""), args[0], arch, args[1]
            )
    except ValueError:
        derived = None
    if derived is not None and not derived.artifact_id:
        # Not derivable (no component, or a generic URL with no readable sha256): only a
        # lookup names it.
        derived = None
    if kind == "rpm":
        rpm_name = derived.artifact_name if derived else ""
        row = lookup.by_refs("rpm", arch, [body, file_url]) or lookup.by_rpm_file(
            arch, filename, rpm_name
        )
    elif kind == "wheel":
        pins = [derived.ref] if derived else []
        # A wheel's file name escapes `-` in its distribution name as `_`; pins keep either.
        pins += [p.replace("_", "-") for p in pins] + [
            p.replace("-", "_") for p in pins
        ]
        row = lookup.by_refs("wheel", arch, [*dict.fromkeys(pins), body, file_url])
    else:
        row = lookup.by_refs("generic", arch, [body.partition("#")[0]])
    if not row and derived:
        row = lookup.by_id12(kind, arch, derived.artifact_name, derived.id12)
    if row:
        result = _result(out, _identity_of(row), "existing", spec)
    elif derived:
        result = _result(out, derived, "derived", spec)
    else:
        return {}
    return _named(result, dated_tag(tag_family, tag_date), tag_family, tags)


def named(resolved_tag: str, tag_family: str, tags=()) -> list:
    """Every (tag, tag_family) an artifact is tagged by.

    Each of `tags` (a tag, or a (tag, family) pair) takes the family its prefix names, else
    the given family, else `tag_family`, else release. One in `tag_family` replaces
    `resolved_tag`, the tag the registry or the date gave that family; the rest are added.
    """
    given = []
    for t in tags:
        tag, family = (t, "") if isinstance(t, str) else t
        if tag:
            given.append((tag, family_of(tag) or family or tag_family or RELEASE))
    if tag_family and resolved_tag and not any(f == tag_family for _, f in given):
        given.insert(0, (resolved_tag, tag_family))
    return list(dict.fromkeys(given))


def _named(result: dict, resolved_tag: str, tag_family: str, tags) -> dict:
    pairs = named(resolved_tag, tag_family, tags)
    primary = next(
        (p for p in pairs if p[1] == tag_family), pairs[0] if pairs else ("", "")
    )
    result.update(tag=primary[0], tag_family=primary[1], tags=[list(p) for p in pairs])
    return result


def _gha(out, rest, arch, lookup, component) -> dict:
    """A GHA leg's delta on a prebaked image; refused when the record's fields do not hash
    to its id (the component it was derived under is not the caller's)."""
    body, over = _options(rest)
    aid, base, installed = ([p.strip() for p in body.split(ID_SEP)] + ["", ""])[:3]
    row = _is_uuid(aid) and lookup.by_id(aid)
    if row:
        return _result(out, _identity_of(row), "existing", "gha:" + body)
    derived = ArtifactIdentity.from_gha(
        over.get("component") or component, base, installed, arch
    )
    if not (base and derived.artifact_id):
        return {}
    if derived.artifact_id != DerivedId.norm(aid):
        raise ValueError(f"gha record {aid}: its inputs hash to {derived.artifact_id}")
    return _result(out, derived, "derived", "gha:" + body)


def _image(out, body, over, arch, lookup, registry, tag_family, tag_date, tags) -> dict:
    host, path, tag, digest = split_image(body)
    if host != registry.host:
        row = lookup.by_refs("image", arch, [body])
        if row:
            result = _result(out, _identity_of(row), "existing", body)
        elif digest:
            derived = ArtifactIdentity.from_image(body, arch, **_image_over(over))
            result = _result(out, derived, "derived", body)
        else:
            return {}
        return _named(result, dated_tag(tag_family, tag_date), tag_family, tags)
    if not digest and tag:
        digest = registry.manifest(path, tag)[0]
    leaf, listed = registry.leaf(path, digest, arch) if digest else ("", "")
    if not leaf:
        row = lookup.by_refs("image", arch, [body])
        if not row:
            return {}
        result = _result(out, _identity_of(row), "existing", body)
        return _named(result, dated_tag(tag_family, tag_date), tag_family, tags)
    pinned = f"{host}/{path}@{leaf}"
    labels = registry.labels(path, leaf)
    labelled = _labelled(labels, path, arch, over)
    row = (
        (labelled and lookup.by_id(labelled.artifact_id))
        or lookup.by_refs("image", arch, [body, pinned])
        or lookup.by_digest(arch, leaf)
    )
    if row:
        identity, source = _identity_of(row), "existing"
    elif labelled:
        identity, source = (
            dataclasses.replace(labelled, ref=pinned, content_digest=leaf),
            "label",
        )
    else:
        identity, source = (
            ArtifactIdentity.from_image(pinned, arch, **_image_over(over)),
            "derived",
        )
    result = _result(out, identity, source, body)
    resolved_tag, registry_tag = tag_of(registry, path, leaf, tag_family, tag_date)
    result.update(
        artifact=f"image:{pinned}",
        leaf=leaf,
        manifest_list=listed,
        registry_tag=registry_tag,
    )
    return _named(result, resolved_tag, tag_family, tags)


def _image_over(over: dict) -> dict:
    return {
        "component": over.get("component", ""),
        "name": over.get("name", ""),
        "id12": over.get("id12", ""),
    }


def _labelled(labels: dict, path: str, arch: str, over: dict):
    """The orchestrator's own identity from its labels, when they hash to its recorded id.

    An image built FROM a labelled base inherits the base's labels; hashed with this image's
    component they do not reproduce the base's id, so they are ignored.
    """
    aid, id12, name = (
        labels.get(LABEL_ID),
        labels.get(LABEL_ID12),
        labels.get(LABEL_NAME),
    )
    if not (aid and id12 and name):
        return None
    component = over.get("component") or path.rsplit("/", 1)[-1]
    identity = ArtifactIdentity(
        component=component,
        artifact_name=name,
        id12=id12,
        arch=DerivedId.arch(labels.get("spyre.artifact.arch") or arch),
        kind="image",
    )
    return identity if identity.artifact_id == aid else None


def _result(out: dict, identity: ArtifactIdentity, source: str, spec: str) -> dict:
    kind = identity.kind
    method, ref_kind = ArtifactWriter.ref_shape(kind)
    refs = [[method, ref_kind, identity.ref]] if identity.ref else []
    return {
        **out,
        "artifact_id": identity.artifact_id,
        "artifact": f"{kind}:{identity.ref}" if identity.ref else spec,
        "component": identity.component,
        "artifact_name": identity.artifact_name,
        "arch": identity.arch,
        "kind": kind,
        "refs": refs,
        "tag": "",
        "tag_family": "",
        "tags": [],
        "source": source,
        "identity": identity,
    }


def ensure_artifact(
    client,
    db: str,
    spec: str,
    arch: str,
    *,
    origin: str = "built",
    tags=(),
    tag_family: str | None = None,
    tag_date=None,
    run_url: str = "",
    component: str = "",
    sources=(),
    identity_deps=(),
    registry: Registry | None = None,
    artifactory: Artifactory | None = None,
) -> ArtifactIdentity:
    """Resolve `spec` and make sure spyre_v2 holds it: the artifact and its ref when new, and
    each of its tags once -- `tag_family`'s tag (from the registry, else dated by `tag_date`)
    and `tags`, combined as `named` describes (an explicit ci-cd-tech-preview-v3 replaces the
    resolved tech-preview tag). Returns the identity; raises ValueError when the spec names
    nothing."""
    r = resolve(
        spec,
        arch,
        lookup=Lookup(client, db),
        registry=registry,
        artifactory=artifactory,
        tag_family=tag_family or "",
        tag_date=tag_date,
        tags=tags,
        component=component,
    )
    if not r:
        raise ValueError(f"{spec!r} names no artifact on {arch}")
    identity = r["identity"]
    # The canonical ref (an image's pinned leaf) is added to an existing record once.
    ref = (
        r["artifact"].partition(":")[2]
        if r["artifact"].startswith(identity.kind + ":")
        else identity.ref
    )
    if ArtifactWriter.ref_recorded(client, db, identity.artifact_id, ref):
        ref = ""
    identity = dataclasses.replace(
        identity, ref=ref, content_digest=r.get("leaf", identity.content_digest)
    )
    props = {"run_url": run_url}
    # A GHA delta is recorded as insert_gha_result records it: chained on its base image.
    base = dict(identity.inputs).get("base_artifact_id", "")
    ArtifactWriter.insert_artifact(
        client,
        db,
        identity,
        origin=origin,
        sources=sources,
        identity_deps=[
            *identity_deps,
            *([f"{schema.DEP_BASE_PREFIX}{base}"] if base else []),
        ],
        props={
            "run_url": run_url,
            "resolved_from": r["source"],
            **({"source": "gha"} if base else {}),
        },
        tags=[(t, f, props) for t, f in r["tags"]],
    )
    return dataclasses.replace(identity, ref=ref or r["identity"].ref)


# Function API name, beside ensure_artifact.
resolve_artifact = resolve
