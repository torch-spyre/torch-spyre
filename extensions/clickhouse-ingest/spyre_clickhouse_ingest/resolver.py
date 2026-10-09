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

"""Any artifact spec -> its one spyre_v2 artifact. Every writer that names an artifact goes
through `resolve` (read-only) or `ensure` (records it), so one artifact gets one id whichever
writer saw it first.

Specs, simple -> advanced (README: "Naming an artifact"):
    <artifact_id> | id:<artifact_id>              this exact artifact; it must be recorded
    image:<host>/<repo>[:tag][@sha256:<digest>]   a list digest resolves to the per-arch leaf;
                                                  arch `multi` keeps the list
    rpm:<file | URL | NEVRA | dnf glob>
    wheel:<name==version | file | URL>
    generic:<url>[#<sha256>]
    gha:<artifact_id>|<base_artifact_id>|<installed>   derive-gha-artifact-id's record
    any of the above + ;component=<c>;name=<n>;id12=<12 hex>   an authoritative identity

An under-specified spec is looked up first and the existing record wins; only immutable refs
(a digest, name==version, an rpm NEVRA or glob, a generic sha) are looked up, never a moving
tag or a bare name. A recorded image digest is answered with no registry call; with no registry
answer (off, or no credentials) an unrecorded image is found by a tag ending in its id12, else
derived from its digest. An authoritative identity is derived from its inputs with no registry
call and never rebound; a gha: record or an explicit id is verified, never guessed.

Tags: `tag_family` (tag_families.yaml) names the artifact by that family's registry tag, else
by `tag_date`; each full `tag` takes the family its prefix names, else `tag_family` (a tag
with neither is refused), and replaces the resolved tag of its family.
"""

import base64
import dataclasses
import fnmatch
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

from .identity import ID_SEP, ArtifactIdentity, DerivedId, _hex_token
from .registry import (
    ICR_HOST,
    MISC,
    Registry,
    canonical,
    dated_tag,
    family_of,
    split_image,
    tag_families,
    tag_of,
)
from .writer import ArtifactWriter

KINDS = ("image", "rpm", "wheel", "generic")
LOOKUP_MODES = ("auto", "off", "only")
REGISTRY_MODES = ("auto", "off")
# Every spelling an arch is stored under; the identity hash folds them, the columns do not.
ARCH_ALIASES = {"x86_64": ("x86_64", "amd64", "x86", "x86-64")}
# The orchestrator's image labels: its own record of the artifact it built.
LABEL_ID, LABEL_ID12, LABEL_NAME = (
    "spyre.artifact.id",
    "spyre.artifact.id12",
    "spyre.artifact.name",
)
FILE_IDENTITY_PROP = "spyre.identity"
IDENTITY_OPTIONS = ("component", "name", "id12")


class NeedsRegistry(ValueError):
    """The spec cannot be resolved without the registry, which is off or unreachable."""


def _aliases(arch: str) -> list:
    a = DerivedId.arch(arch)
    return list(ARCH_ALIASES.get(a, (a,)))


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


@dataclasses.dataclass
class Resolution:
    """The one artifact a spec names, and (after `ensure`) whether anything was written.

    The same fields are the JSON every surface prints (`as_dict`). It also reads like the
    dict `resolve` used to return: `r["artifact_id"]`, `r.get("leaf")`.
    """

    artifact_id: str
    artifact: str
    kind: str
    component: str
    artifact_name: str
    id12: str
    arch: str
    refs: list
    source: str  # given | existing | label | derived
    lookup: str  # db | none
    tag: str = ""
    tag_family: str = ""
    tags: list = dataclasses.field(default_factory=list)
    written: bool = False
    dry_run: bool = False
    leaf: str = ""
    manifest_list: str = ""
    registry_tag: str = ""
    # canonical tag -> the raw spelling it was given or found as (recorded as tag prop registry_tag)
    spellings: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)
    # used | off | unreachable: <why>; '' when the spec needed no registry.
    registry: str = ""
    # A dry run's would-be rows, table -> [row]; only printed for a dry run.
    rows: dict = dataclasses.field(default_factory=dict)
    identity: ArtifactIdentity | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    def as_dict(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("identity", "spellings")
            and (f.name != "rows" or self.dry_run)
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


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

    COLS = "toString(artifact_id), component, artifact_name, props['id12'], arch, kind, props['ref']"

    def __init__(self, client=None, db: str = ""):
        self.client, self.db = client, db

    def _one(self, where: str, params: dict):
        if self.client is None:
            return None
        rows = self.client.query(
            f"SELECT {self.COLS} FROM {self.db}.artifacts WHERE {where} ORDER BY ts DESC LIMIT 1",
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

    def digest_arches(self, digest: str) -> set:
        """Every arch an image digest is recorded under (any arch), folded."""
        if self.client is None or not digest:
            return set()
        rows = self.client.query(
            f"SELECT DISTINCT arch FROM {self.db}.artifacts WHERE kind = 'image' AND artifact_id IN "
            f"(SELECT artifact_id FROM {self.db}.artifact_refs WHERE content_digest = {{d:String}} "
            f"OR endsWith(ref, {{at:String}}))",
            parameters={"d": digest, "at": "@" + digest},
        ).result_rows
        return {DerivedId.arch(a) for (a,) in rows}

    def by_tag(self, arch: str, pullspec: str):
        """The one image recorded under a content-addressed `pullspec` (its tag ends in the
        record's id12); a tag with no id12 moves, and one held by several records moved."""
        token = _hex_token(pullspec.rsplit(":", 1)[-1])
        if self.client is None or not token:
            return None
        rows = self.client.query(
            f"SELECT {self.COLS} FROM {self.db}.artifacts WHERE artifact_id IN "
            f"(SELECT artifact_id FROM {self.db}.artifact_refs WHERE ref = {{r:String}}) "
            "AND kind = 'image' AND props['id12'] = {i:String} AND arch IN {arch:Array(String)} "
            "ORDER BY ts DESC LIMIT 1 BY artifact_id LIMIT 2",
            parameters={"r": pullspec, "i": token, "arch": _aliases(arch)},
        ).result_rows
        return rows[0] if len(rows) == 1 else None

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
        if not (name and id12):
            return None
        return self._one(
            "kind = {kind:String} AND artifact_name = {n:String} "
            "AND props['id12'] = {i:String} AND arch IN {arch:Array(String)}",
            {"kind": kind, "n": name, "i": id12, "arch": _aliases(arch)},
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
    if set(over) - set(IDENTITY_OPTIONS) or len(over) != len(opts):
        raise ValueError(f"unknown or malformed option(s) in {spec!r}")
    return body.strip(), over


def misc_warning(tag: str) -> str:
    return (
        f"  [warn] tag {tag!r} names no tag family by its prefix and was given none: filed "
        f"under {MISC!r}; pass its real family (--tag-family pr|main|nightly|weekly|snap|release)"
    )


def is_misc_fallback(tag, tag_family: str) -> bool:
    """Would `named` file this tag under misc for want of any family?"""
    tag, family = (tag, "") if isinstance(tag, str) else tag
    return bool(tag) and not (family_of(tag) or family or tag_family)


def named(resolved_tag: str, tag_family: str, tags=()) -> list:
    """Every (tag, tag_family) an artifact is tagged by.

    Each of `tags` (a tag, or a (tag, family) pair) takes the family its prefix names, else
    the given family, else `tag_family`, else misc (with a warning); an unknown family raises
    ValueError. One in `tag_family` replaces
    `resolved_tag`, the tag the registry or the date gave that family; the rest are added.
    A tag in another spelling of its family's prefix is returned under the family's own.
    """
    given = []
    known = tag_families()
    for t in tags:
        tag, family = (t, "") if isinstance(t, str) else t
        if not tag:
            continue
        tag = canonical(tag)
        family = family_of(tag) or family or tag_family
        if not family:
            print(misc_warning(tag), file=sys.stderr)
            family = MISC
        if family not in known:
            raise ValueError(f"tag {tag!r}: unknown tag_family {family!r}")
        given.append((tag, family))
    if tag_family and resolved_tag and not any(f == tag_family for _, f in given):
        given.insert(0, (canonical(resolved_tag), tag_family))
    return list(dict.fromkeys(given))


class _Resolver:
    """One resolution's settings: the lookup, the registry and Artifactory clients, the tag
    options. Clients are built only when a spec needs them."""

    def __init__(
        self, arch, lookup, registry, artifactory, tag_family, tags, tag_date, component
    ):
        self.arch, self.component = arch, component
        self.tag_family, self.tags, self.tag_date = (
            tag_family or "",
            list(tags or ()),
            tag_date,
        )
        self.lookup_mode = lookup if isinstance(lookup, str) else "auto"
        self.lookup = lookup if isinstance(lookup, Lookup) else Lookup()
        if self.lookup_mode not in LOOKUP_MODES:
            raise ValueError(f"lookup must be one of {LOOKUP_MODES}: {lookup!r}")
        if self.lookup_mode == "off":
            self.lookup = Lookup()
        self.registry_mode = registry if isinstance(registry, str) else "auto"
        if self.registry_mode not in REGISTRY_MODES:
            raise ValueError(f"registry must be one of {REGISTRY_MODES}: {registry!r}")
        self._registry = registry if isinstance(registry, Registry) else None
        self._artifactory = artifactory

    def registry(self, why: str) -> Registry:
        if self.registry_mode == "off":
            raise NeedsRegistry(f"registry='off', but {why}")
        if self._registry is None:
            self._registry = Registry.from_env()
        return self._registry

    def artifactory(self) -> "Artifactory | None":
        if self.registry_mode == "off":
            return None
        if self._artifactory is None:
            self._artifactory = Artifactory.from_env()
        return self._artifactory

    def result(self, identity, source, spec, resolved_tag="", **extra) -> Resolution:
        method, ref_kind = ArtifactWriter.ref_shape(identity.kind)
        pairs = named(resolved_tag, self.tag_family, self.tags)
        given = [resolved_tag] + [t if isinstance(t, str) else t[0] for t in self.tags]
        spellings = {canonical(t): t for t in given if t and canonical(t) != t}
        primary = next(
            (p for p in pairs if p[1] == self.tag_family),
            pairs[0] if pairs else ("", ""),
        )
        return Resolution(
            artifact_id=identity.artifact_id,
            artifact=f"{identity.kind}:{identity.ref}" if identity.ref else spec,
            kind=identity.kind,
            component=identity.component,
            artifact_name=identity.artifact_name,
            id12=identity.id12,
            arch=identity.arch,
            refs=[[method, ref_kind, identity.ref]] if identity.ref else [],
            source=source,
            lookup="db" if self.lookup.client is not None else "none",
            tag=primary[0],
            tag_family=primary[1],
            tags=[list(p) for p in pairs],
            spellings=spellings,
            identity=identity,
            **extra,
        )

    def existing(self, row, spec, **extra) -> Resolution:
        return self.result(
            _identity_of(row),
            "existing",
            spec,
            dated_tag(self.tag_family, self.tag_date),
            **extra,
        )

    # -- forms ----------------------------------------------------------------------------

    def by_id(self, aid: str, spec: str):
        if self.lookup.client is None:
            raise ValueError(
                f"{spec!r} names an artifact_id, which only a lookup can verify"
            )
        row = self.lookup.by_id(aid)
        return self.existing(row, spec) if row else None

    def given(self, kind: str, body: str, over: dict, spec: str):
        """An authoritative identity: derived from its inputs; a lookup only spots a duplicate."""
        identity = ArtifactIdentity(
            component=over["component"],
            artifact_name=over["name"],
            id12=over["id12"],
            arch=DerivedId.arch(self.arch),
            kind=kind,
            ref=body,
            content_digest=split_image(body)[3] if kind == "image" else "",
        )
        return self.given_identity(identity, spec)

    def given_identity(self, identity: ArtifactIdentity, spec: str):
        if not identity.artifact_id:
            raise ValueError(f"{spec!r}: its identity inputs derive no id")
        row = self.lookup.by_id(identity.artifact_id)
        if row:
            _identity_of(row)  # refuses a recorded row whose inputs are not these
        elif self.lookup_mode == "only":
            return None
        return self.result(
            identity, "given", spec, dated_tag(self.tag_family, self.tag_date)
        )

    def gha(self, rest: str, spec: str):
        """A GHA leg's delta on a prebaked image; refused when the record's fields do not hash
        to its id (the component it was derived under is not the caller's)."""
        body, over = _options(rest)
        aid, base, installed = ([p.strip() for p in body.split(ID_SEP)] + ["", ""])[:3]
        row = _is_uuid(aid) and self.lookup.by_id(aid)
        if row:
            return self.existing(row, "gha:" + body)
        if self.lookup_mode == "only":
            return None
        component = over.get("component") or self.component
        if base and not component:
            raise ValueError(
                f"gha: spec needs ;component=<c> (or a component): {spec!r}"
            )
        derived = ArtifactIdentity.from_gha(component, base, installed, self.arch)
        if not (base and derived.artifact_id):
            return None
        if derived.artifact_id != DerivedId.norm(aid):
            raise ValueError(
                f"gha record {aid}: its inputs hash to {derived.artifact_id}"
            )
        return self.result(
            derived, "derived", "gha:" + body, dated_tag(self.tag_family, self.tag_date)
        )

    def image(self, body: str, over: dict, spec: str):
        host, path, tag, digest = split_image(body)
        multi = DerivedId.arch(self.arch) == "multi"
        if multi or host != (self._registry.host if self._registry else ICR_HOST):
            # A manifest list (or another registry's image) is its own digest: no leaf to find.
            if (
                multi
                and tag
                and not digest
                and host == (self._registry.host if self._registry else ICR_HOST)
            ):
                # A list by tag (a promoted stream tag): the registry's digest for it now.
                digest = self.registry(
                    f"{spec!r} names a manifest list by tag"
                ).manifest(path, tag)[0]
                if not digest:
                    return None
                body = f"{host}/{path}@{digest}"
            if not digest:
                raise ValueError(
                    f"{spec!r}: a manifest list or foreign image needs its @digest"
                )
            row = self.lookup.by_digest(self.arch, digest)
            if row:
                return self.existing(row, body)
            if self.lookup_mode == "only":
                return None
            identity = ArtifactIdentity.from_image(body, self.arch, **_image_over(over))
            resolved, registry_tag = (
                self.tagged(path, digest)
                if multi
                else (dated_tag(self.tag_family, self.tag_date), "")
            )
            return self.result(identity, "derived", body, resolved, manifest_list=digest if multi else "",
                               registry_tag=registry_tag)  # fmt: skip
        # A recorded digest needs no registry for its id (the record keeps the id it was filed
        # under); the registry, when it answers, still names its tag in tag_family.
        row = digest and self.lookup.by_digest(self.arch, digest)
        if not row and digest:
            # Recorded under another arch: a per-arch leaf named with the wrong --arch.
            other = self.lookup.digest_arches(digest) - {
                "multi",
                DerivedId.arch(self.arch),
            }
            if other:
                raise ValueError(
                    f"{spec!r} is recorded as {sorted(other)}, not {self.arch!r}"
                )
        if row:
            resolved, registry_tag = self.tagged(path, digest)
            return self.result(
                _identity_of(row), "existing", body, resolved, registry_tag=registry_tag
            )
        try:
            registry = self.registry(f"{spec!r} needs its per-arch leaf")
            if not digest and tag:
                # A tag moves: only the registry's answer for it right now is an identity.
                digest = registry.manifest(path, tag)[0]
            leaf, listed = (
                registry.leaf(path, digest, self.arch) if digest else ("", "")
            )
            labels = registry.labels(path, leaf) if leaf else {}
            # A single-arch manifest is its own leaf: its platform must be the one asked for.
            platform = (
                registry.config(path, leaf).get("architecture", "")
                if leaf and not listed
                else ""
            )
        except (NeedsRegistry, OSError) as err:
            return self.offline(host, path, tag, digest, over, body, spec, err)
        if not leaf:
            return None
        if platform and DerivedId.arch(platform) != DerivedId.arch(self.arch):
            raise ValueError(f"{spec!r} is a {platform} image, not {self.arch!r}")
        pinned = f"{host}/{path}@{leaf}"
        labelled = _labelled(labels, path, self.arch, over)
        row = (
            labelled and self.lookup.by_id(labelled.artifact_id)
        ) or self.lookup.by_digest(self.arch, leaf)
        if row:
            identity, source = _identity_of(row), "existing"
        elif self.lookup_mode == "only":
            return None
        elif labelled:
            identity, source = (
                dataclasses.replace(labelled, ref=pinned, content_digest=leaf),
                "label",
            )
        else:
            identity, source = (
                ArtifactIdentity.from_image(pinned, self.arch, **_image_over(over)),
                "derived",
            )
        resolved, registry_tag = self.tagged(path, leaf)
        r = self.result(
            identity,
            source,
            body,
            resolved,
            leaf=leaf,
            manifest_list=listed,
            registry_tag=registry_tag,
            registry="used",
        )
        r.artifact = f"image:{pinned}"
        return r

    def offline(self, host, path, tag, digest, over, body, spec, why):
        """No registry answer (off, or unreachable, e.g. no credentials) for an unrecorded
        digest: the record of a content-addressed tag, else the digest as given."""
        state = "off" if self.registry_mode == "off" else f"unreachable: {why}"
        row = tag and self.lookup.by_tag(self.arch, f"{host}/{path}:{tag}")
        if row:
            return self.existing(row, body, registry=state)
        if self.lookup_mode == "only":
            return None
        if not digest:
            raise NeedsRegistry(
                f"{spec!r}: its tag names no one record (one with no id12 moves), and "
                f"the registry is {state}"
            )
        identity = ArtifactIdentity.from_image(body, self.arch, **_image_over(over))
        return self.result(
            identity, "derived", body, dated_tag(self.tag_family, self.tag_date),
            registry=state,
        )  # fmt: skip

    def tagged(self, path: str, digest: str) -> tuple:
        """(tag, registry_tag) of `digest` in tag_family: the registry's, else the dated one."""
        if self.tag_family and self.registry_mode != "off":
            try:
                return tag_of(
                    self.registry("tags"), path, digest, self.tag_family, self.tag_date
                )
            except OSError:
                pass
        return dated_tag(self.tag_family, self.tag_date), ""

    def file(self, kind: str, body: str, over: dict, spec: str):
        over.setdefault("component", self.component)
        file_url = body.split("#", 1)[0] if body.startswith("http") else ""
        artifactory = self.artifactory() if file_url else None
        info = artifactory.info(file_url) if artifactory else {}
        prop_id12 = (info.get("properties", {}).get(FILE_IDENTITY_PROP) or "")[:12]
        filename = body.split("#", 1)[0].rsplit("/", 1)[-1]
        args = (
            over.get("component", ""),
            over.get("name", ""),
            over.get("id12", "") or prop_id12,
        )
        url, _, sha = body.partition("#")
        sha = sha or info.get("sha256", "")
        try:
            if kind == "rpm":
                derived = ArtifactIdentity.from_rpm(file_url or body, self.arch, *args)
            elif kind == "wheel":
                derived = ArtifactIdentity.from_wheel(body, self.arch, *args)
            else:
                derived = ArtifactIdentity.from_generic(
                    url, sha, args[0], self.arch, args[1]
                )
        except ValueError:
            derived = None
        if derived is not None and not derived.artifact_id:
            derived = None  # no component, or a generic URL with no readable sha256
        # Looked up only by immutable refs: never a bare wheel name or a generic URL alone.
        if kind == "rpm":
            rpm_name = derived.artifact_name if derived else ""
            row = self.lookup.by_refs(
                "rpm", self.arch, [body, file_url]
            ) or self.lookup.by_rpm_file(self.arch, filename, rpm_name)
        elif kind == "wheel":
            pins = [derived.ref] if derived else [body] if "==" in body else []
            # A wheel's file name escapes `-` in its distribution name as `_`; pins keep either.
            pins += [p.replace("_", "-") for p in pins] + [
                p.replace("-", "_") for p in pins
            ]
            row = self.lookup.by_refs("wheel", self.arch, list(dict.fromkeys(pins)))
        else:
            row = self.lookup.by_refs("generic", self.arch, [url]) if sha else None
        if not row and derived is not None:
            row = self.lookup.by_id12(
                kind, self.arch, derived.artifact_name, derived.id12
            )
        if row:
            return self.existing(row, spec)
        if derived is None or self.lookup_mode == "only":
            return None
        return self.result(
            derived, "derived", spec, dated_tag(self.tag_family, self.tag_date)
        )


def _image_over(over: dict) -> dict:
    return {k: over.get(k, "") for k in IDENTITY_OPTIONS}


def _labelled(labels: dict, path: str, arch: str, over: dict):
    """The orchestrator's own identity from its labels, when they hash to its recorded id.

    An image built FROM a labelled base inherits the base's labels; hashed with this image's
    component they do not reproduce the base's id, so they are ignored.
    """
    aid, id12, name = (labels.get(k) for k in (LABEL_ID, LABEL_ID12, LABEL_NAME))
    if not (aid and id12 and name):
        return None
    identity = ArtifactIdentity(
        component=over.get("component") or path.rsplit("/", 1)[-1],
        artifact_name=name,
        id12=id12,
        arch=DerivedId.arch(labels.get("spyre.artifact.arch") or arch),
        kind="image",
    )
    return identity if identity.artifact_id == aid else None


def resolve(
    spec: "str | ArtifactIdentity",
    arch: str = "",
    *,
    client=None,
    db: str = "",
    lookup="auto",
    registry="auto",
    artifactory: Artifactory | None = None,
    tag_family: str = "",
    tags=(),
    tag_date=None,
    component: str = "",
) -> "Resolution | None":
    """The artifact `spec` names on `arch` (read-only); None when it names nothing. An
    ArtifactIdentity as `spec` is an authoritative identity given field by field.

    lookup: 'auto' (existing record wins), 'off' (derive only, no database) or 'only' (must be
    recorded), or a Lookup; it reads `client`/`db`. registry: 'auto' (called only when the
    spec needs it; unreachable, it is treated as off) or 'off' (never; NeedsRegistry when the
    database cannot stand in for it), or a Registry. Raises
    ValueError for a malformed spec or a record that does not hash to its id.
    """
    if isinstance(lookup, str) and lookup != "off" and client is not None:
        lookup_obj = Lookup(client, db)
    else:
        lookup_obj = lookup if isinstance(lookup, Lookup) else None
    r = _Resolver(
        arch, lookup, registry, artifactory, tag_family, tags, tag_date, component
    )
    if lookup_obj is not None and r.lookup_mode != "off":
        r.lookup = lookup_obj
    if r.lookup_mode == "only" and r.lookup.client is None:
        raise ValueError("lookup='only' needs a database to look in")
    if isinstance(spec, ArtifactIdentity):
        return r.given_identity(spec, f"{spec.kind}:{spec.ref}")
    spec = (spec or "").strip()
    kind, sep, rest = spec.partition(":")
    if not sep and _is_uuid(spec):
        return r.by_id(spec, spec)
    if kind == "id":
        return r.by_id(rest.strip(), spec)
    if kind == "gha":
        return r.gha(rest, spec)
    if kind not in KINDS:
        raise ValueError(
            f"spec must be id:, image:, rpm:, wheel:, generic:, gha: or an artifact_id: {spec!r}"
        )
    body, over = _options(rest)
    if all(over.get(k) for k in IDENTITY_OPTIONS):
        return r.given(kind, body, over, spec)
    if kind == "image":
        return r.image(body, over, spec)
    return r.file(kind, body, over, spec)


def ensure(
    client,
    db: str,
    spec: "str | ArtifactIdentity",
    arch: str = "",
    *,
    origin: str = "built",
    lookup="auto",
    registry="auto",
    artifactory: Artifactory | None = None,
    tag_family: str | None = None,
    tags=(),
    tag_date=None,
    component: str = "",
    sources=(),
    identity_deps=(),
    context_deps=(),
    props=None,
    tag_props=None,
    run_url: str = "",
    dry_run: bool = False,
) -> Resolution:
    """Resolve `spec` (as `resolve`, with lookups on `client`) and make sure spyre_v2 holds it:
    the artifact with its recorded fields when new, its canonical ref, and each tag once.
    `written` says whether anything was (with `dry_run`: would be) written. Raises ValueError
    when the spec names nothing."""
    r = resolve(
        spec, arch, client=client, db=db, lookup=lookup, registry=registry, artifactory=artifactory,
        tag_family=tag_family or "", tags=tags, tag_date=tag_date, component=component,
    )  # fmt: skip
    if r is None:
        raise ValueError(f"{spec!r} names no artifact on {arch or 'any arch'}")
    if client is None and not dry_run:
        raise ValueError("ensure writes through a client; without one, pass dry_run")
    sink = _Capture(client) if dry_run else client
    identity = r.identity
    ref = (
        r.artifact.partition(":")[2]
        if r.artifact.startswith(identity.kind + ":")
        else identity.ref
    )
    new_ref = bool(ref) and not ArtifactWriter.ref_recorded(
        sink, db, identity.artifact_id, ref
    )
    new_artifact = r.source != "existing" and not ArtifactWriter.artifact_recorded(
        sink, db, identity.artifact_id
    )
    tag_props = {**({"run_url": run_url} if run_url else {}), **(tag_props or {})}
    new_tags = [
        (t, f)
        for t, f in r.tags
        if not ArtifactWriter.tag_recorded(sink, db, t, identity.artifact_id)
    ]
    r.written, r.dry_run = bool(new_artifact or new_ref or new_tags), dry_run
    if r.written:
        _write(sink, db, r, identity, ref, new_tags, origin, sources, identity_deps,
               context_deps, props, tag_props, run_url)  # fmt: skip
    if dry_run:
        r.rows = sink.rows
    return r


def _write(client, db, r, identity, ref, new_tags, origin, sources, identity_deps, context_deps,
           props, tag_props, run_url) -> None:  # fmt: skip
    """ensure's writes, existence-checked row by row."""

    def props_of(tag: str) -> dict:
        raw = r.spellings.get(tag)
        return {**tag_props, **({"registry_tag": raw} if raw else {})}

    base = dict(identity.inputs).get("base_artifact_id", "")
    if base:
        # A GHA delta is recorded as insert_gha_result records it, keyed on the record's id.
        ArtifactWriter.insert_gha_artifact(
            client, db, identity.artifact_id, identity.component, base,
            dict(identity.inputs).get("installed", ""), identity.arch, sources=sources, run_url=run_url,
        )  # fmt: skip
        for t, f in new_tags:
            ArtifactWriter.insert_tag(client, db, identity, t, f, props=props_of(t))
        return
    identity = dataclasses.replace(
        identity, ref=ref, content_digest=r.leaf or identity.content_digest
    )
    ArtifactWriter.insert_artifact(
        client,
        db,
        identity,
        origin=origin,
        sources=sources,
        identity_deps=identity_deps,
        context_deps=context_deps,
        props={**({"run_url": run_url} if run_url else {}), **(props or {})},
        tags=[(t, f, props_of(t)) for t, f in new_tags],
    )


class _Capture:
    """A dry run's client: reads go to the database (none: nothing recorded), inserts are kept."""

    def __init__(self, client):
        self.client, self.rows = client, {}

    def query(self, sql, parameters=None):
        if self.client is not None:
            return self.client.query(sql, parameters=parameters)
        return _Rows([(0,)] if "count()" in sql else [])

    def insert(self, table, rows, column_names=None, database=None, **kwargs):
        self.rows.setdefault(table, []).extend(dict(zip(column_names, r)) for r in rows)


class _Rows:
    def __init__(self, rows):
        self.result_rows = rows


def ensure_artifact(
    client, db: str, spec: str, arch: str, **options
) -> ArtifactIdentity:
    """`ensure`, returning only the identity (the first API, kept for its callers)."""
    return ensure(client, db, spec, arch, **options).identity


resolve_artifact = resolve
