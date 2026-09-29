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

"""Derived, never-minted uuid5 identities: one class per kind, sharing `DerivedId`."""

import hashlib
import json
import uuid
from dataclasses import dataclass

import regex as re

from .junit import RunCoordinates

ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com")

ID_SEP = "|"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX12 = re.compile(r"[0-9a-f]{12}")

# The component stamped on rows when the caller names none. A DEFAULT, not a constant: a
# test cell may run another component's suite, and component is a hash input.
COMPONENT_DEFAULT = "torch-spyre"

# Full-metadata identity record, one per component layer (each Containerfile overwrites its
# own). Preferred read path.
SPYRE_ARTIFACT_JSON_FILE = "/home/senuser/spyre_artifact.json"

# Pre-JSON-rollout format: a bare id, beside installed_rpms.txt. Kept as a fallback so images
# built before the rollout (or an unstamped standalone build) still derive something.
BASE_ARTIFACT_ID_FILE = "/home/senuser/spyre_artifact_id.txt"


class DerivedId:
    """Base for every derived identity: normalisation plus the shared uuid5 hash."""

    NAMESPACE = ID_NAMESPACE
    SEP = ID_SEP

    @staticmethod
    def norm(value) -> str:
        """Canonical scalar form: stripped and lowercased."""
        return ("" if value is None else str(value)).strip().lower()

    @staticmethod
    def arch(value) -> str:
        """amd64/x86/x86-64 all fold to x86_64, so one leg hashes as one."""
        a = DerivedId.norm(value)
        return "x86_64" if a in ("amd64", "x86", "x86-64", "x86_64") else a

    @classmethod
    def hash(cls, *parts: str) -> str:
        """uuid5 of the parts joined by SEP, as a string."""
        return str(uuid.uuid5(cls.NAMESPACE, cls.SEP.join(parts)))

    @classmethod
    def complete(cls, *values) -> bool:
        """True when every required field is non-blank; a blank one refuses the id."""
        return all(cls.norm(v) for v in values)

    @classmethod
    def tag_part(cls, tags) -> str:
        """Tags as a deduped, sorted, comma-joined string -- a SET, not a sequence."""
        return ",".join(sorted({t for t in (cls.norm(x) for x in (tags or [])) if t}))

    @classmethod
    def disc_part(cls, disc, disc_keys) -> str:
        """The per-producer discriminators, in `disc_keys` order; absent keys emit."""
        disc = disc or {}
        return ",".join(f"{k}={cls.norm(disc.get(k))}" for k in disc_keys or ())


class RunId(DerivedId):
    """Identity of one leg: (source, external_run_id, arch, test_type)."""

    @classmethod
    def derive(
        cls, source: str, external_run_id: str, arch: str, test_type: str
    ) -> str:
        """The leg's uuid, or '' when any field is missing."""
        if not cls.complete(source, external_run_id, arch, test_type):
            return ""
        return cls.hash(
            cls.norm(source),
            cls.norm(external_run_id),
            cls.arch(arch),
            cls.norm(test_type),
        )

    @classmethod
    def for_args(cls, args, run_id: str, arch: str, tier: str) -> str:
        """The threaded --run-id uuid when there is one, else the coordinate hash."""
        threaded = RunCoordinates.threaded_run_id(args)
        if threaded:
            return threaded
        source, external = RunCoordinates.source_and_external(args, run_id)
        return cls.derive(source, external, arch, tier)


class CaseId(DerivedId):
    """Content identity of a test, so the same test reconciles across runs."""

    @classmethod
    def derive(cls, component: str, classname: str, name: str, tags) -> str:
        """The test's uuid, or '' with no component/name; classname may be blank."""
        if not cls.complete(component, name):
            return ""
        return cls.hash(
            cls.norm(component),
            cls.norm(classname),
            cls.norm(name),
            cls.tag_part(tags),
        )

    @staticmethod
    def tags_for(case: dict) -> list:
        """The case's tags as an ARRAY of `namespace__value` strings."""
        tags = set()
        for pname, pvalue in case.get("properties", []) or []:
            if pname == "tag":
                if pvalue:
                    tags.add(pvalue)
            elif "__" in pname:
                # Some emitters put the namespace__value in the property NAME instead.
                tags.add(pname)
        return sorted(tags)


class ArtifactId(DerivedId):
    """Content identity of an artifact: (component, artifact_name, id12, arch)."""

    FILE = BASE_ARTIFACT_ID_FILE
    JSON_FILE = SPYRE_ARTIFACT_JSON_FILE

    @classmethod
    def derive(cls, component: str, artifact_name: str, id12: str, arch: str) -> str:
        """The artifact's uuid, or '' without a component and an arch."""
        if not (cls.norm(component) and cls.arch(arch)):
            return ""
        return cls.hash(
            cls.norm(component),
            cls.norm(artifact_name),
            cls.norm(id12),
            cls.arch(arch),
        )

    @classmethod
    def metadata_from_image(cls, path: str = "") -> dict:
        """The full stamped identity record (component, deps, sources, ...); {} when absent
        or pre-rollout (a bare-id image has no JSON to parse)."""
        try:
            with open(path or cls.JSON_FILE) as fh:
                data = json.loads(fh.read())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def from_image(cls, path: str = "") -> str:
        """The prebaked image's own artifact_id, read from inside it; '' when absent.

        Without `path`, tries the JSON record then the pre-rollout bare-id file. `path` names
        one file in either format. Only a uuid is returned, so a corrupt file yields ''.
        """
        for p in [path] if path else [cls.JSON_FILE, cls.FILE]:
            try:
                with open(p) as fh:
                    text = fh.read()
            except OSError:
                continue
            try:
                data = json.loads(text)
            except ValueError:
                data = text
            aid = cls.norm(data.get("artifact_id") if isinstance(data, dict) else data)
            try:
                uuid.UUID(aid)
            except ValueError:
                continue
            return aid
        return ""


class GhaArtifactId(ArtifactId):
    """Artifact identity for a GHA leg that installed something on a prebaked image."""

    @classmethod
    def derive(
        cls, component: str, base_artifact_id: str, installed: str, arch: str
    ) -> str:
        """The leg's artifact uuid: hashes only its delta onto the base image's id."""
        if not (cls.norm(component) and cls.arch(arch)):
            return ""
        return ArtifactId.derive(
            component,
            cls.norm(base_artifact_id),
            cls.installed_digest(installed),
            arch,
        )

    @classmethod
    def installed_digest(cls, installed: str) -> str:
        """The id12-slot digest of the installed set; '' for empty (the base image)."""
        items = sorted(
            {cls.norm(x) for x in (installed or "").replace(",", " ").split() if x}
        )
        return (
            hashlib.sha256(cls.SEP.join(items).encode()).hexdigest()[:12]
            if items
            else ""
        )


@dataclass(frozen=True)
class ArtifactIdentity:
    """The four hash inputs of an artifact, plus how to fetch it, from any way it is named.

    Every constructor reduces to `ArtifactId.derive`, so an image named by digest here and the
    same bytes recorded by another writer under the same four fields share one artifact_id.
    """

    component: str
    artifact_name: str
    id12: str
    arch: str
    kind: str = "image"
    ref: str = ""
    content_digest: str = ""
    # Hash inputs kept readable beside the opaque id (e.g. base_artifact_id, installed).
    inputs: tuple = ()

    @property
    def artifact_id(self) -> str:
        return ArtifactId.derive(
            self.component, self.artifact_name, self.id12, self.arch
        )

    @classmethod
    def from_image(
        cls, ref: str, arch: str, component: str = "", name: str = "", id12: str = ""
    ) -> "ArtifactIdentity":
        """`<repo>[:tag]@sha256:<hex>` of ONE arch's image, the per-arch leaf digest.

        By default the digest is the identity: right for producers that record an image by
        digest. A producer that names it otherwise (the Jenkins orchestrator hashes its own
        inputs into id12 and uses its config name) is matched by passing those three fields.
        """
        repo, _, digest = (ref or "").strip().partition("@")
        if not _SHA256.fullmatch(
            digest.removeprefix("sha256:")
        ) or not digest.startswith("sha256:"):
            raise ValueError(f"image ref needs an @sha256:<64 hex> digest: {ref!r}")
        if id12 and not _HEX12.fullmatch(id12):
            raise ValueError(f"id12 must be 12 hex characters: {id12!r}")
        repo_name = repo.rsplit("/", 1)[-1].split(":", 1)[0]
        return cls(
            component=component or re.sub(r"-(devel|dev)$", "", repo_name),
            artifact_name=name or repo_name,
            id12=id12 or digest[7:19],
            arch=DerivedId.arch(arch),
            kind="image",
            ref=f"{repo}@{digest}",
            content_digest=digest,
        )

    @classmethod
    def from_generic(
        cls, url: str, sha256: str, component: str, arch: str, name: str = ""
    ) -> "ArtifactIdentity":
        """A downloadable file or folder; id12 is its content sha256, never its address."""
        digest = DerivedId.norm(sha256).removeprefix("sha256:")
        if not url or not _SHA256.fullmatch(digest):
            raise ValueError(
                f"generic artifact needs <url>#<64-hex sha256>: {url!r}#{sha256!r}"
            )
        return cls(
            component=component,
            artifact_name=name or url.rstrip("/").rsplit("/", 1)[-1],
            id12=digest[:12],
            arch=DerivedId.arch(arch),
            kind="generic",
            ref=url,
            content_digest=f"sha256:{digest}",
        )

    @classmethod
    def from_gha(
        cls, component: str, base_artifact_id: str, installed: str, arch: str
    ) -> "ArtifactIdentity":
        """A GHA leg: its delta installed onto a prebaked image (see GhaArtifactId)."""
        base = DerivedId.norm(base_artifact_id)
        return cls(
            component=component,
            artifact_name=base,
            id12=GhaArtifactId.installed_digest(installed),
            arch=DerivedId.arch(arch),
            kind="image",
            inputs=(
                ("base_artifact_id", base),
                ("installed", (installed or "").strip()),
            ),
        )

    @classmethod
    def parse(cls, spec: str, arch: str, component: str) -> "ArtifactIdentity | None":
        """`image:<ref@digest>`, `generic:<url>#<sha256>`, or the GHA record
        `<artifact_id>|<base_artifact_id>|<installed>`; None for a bare id with no inputs.

        image/generic take `;component=`, `;name=` and (image) `;id12=` overrides. Without
        one, an image names its own component: `component` is the caller's, the suite's
        owner, and one component's suite routinely runs in another component's image.
        """
        spec = (spec or "").strip()
        kind, _, rest = spec.partition(":")
        if kind in ("image", "generic"):
            ref, *opts = rest.split(";")
            over = dict(o.split("=", 1) for o in opts if "=" in o)
            unknown = set(over) - (
                {"component", "name", "id12"}
                if kind == "image"
                else {"component", "name"}
            )
            if unknown or len(over) != len(opts):
                raise ValueError(f"unknown or malformed {kind} option(s) in {spec!r}")
            if kind == "image":
                return cls.from_image(
                    ref,
                    arch,
                    over.get("component", ""),
                    over.get("name", ""),
                    over.get("id12", ""),
                )
            url, sep, sha = ref.rpartition("#")
            if not sep:
                raise ValueError(f"generic artifact needs <url>#<sha256>: {spec!r}")
            return cls.from_generic(
                url, sha, over.get("component") or component, arch, over.get("name", "")
            )
        parts = [p.strip() for p in spec.split(ID_SEP)] + ["", ""]
        if not parts[1]:
            return None
        return cls.from_gha(component, parts[1], parts[2], arch)


class CapabilityId(DerivedId):
    """Content identity of one (subject, capability) pair; `backend` is not hashed."""

    @classmethod
    def derive(
        cls,
        component: str,
        test_type: str,
        subject: str,
        name: str,
        disc=None,
        disc_keys=(),
    ) -> str:
        """The capability's uuid, or '' without a component, a test_type and a name."""
        if not cls.complete(component, test_type, name):
            return ""
        return cls.hash(
            cls.norm(component),
            cls.norm(test_type),
            cls.norm(subject),
            cls.norm(name),
            cls.disc_part(disc, disc_keys),
        )


class BenchmarkId(DerivedId):
    """Content identity of a benchmark; `backend` is unhashed -- the comparison axis."""

    @classmethod
    def derive(cls, component: str, name: str, tags, disc=None, disc_keys=()) -> str:
        """The benchmark's uuid, or '' without a component and a name."""
        if not cls.complete(component, name):
            return ""
        return cls.hash(
            cls.norm(component),
            cls.norm(name),
            cls.tag_part(tags),
            cls.disc_part(disc, disc_keys),
        )


class Component:
    """The component stamped on v2 rows, which every id above hashes."""

    DEFAULT = COMPONENT_DEFAULT

    @staticmethod
    def of(args, default: str = COMPONENT_DEFAULT) -> str:
        """--component when given, else `default` (each repo has its own)."""
        return (getattr(args, "component", "") or "").strip() or default


# Function API, kept so installed consumers import one definition, not a copy.
_norm = DerivedId.norm
canonical_arch = DerivedId.arch
run_id_of = RunId.derive
run_id_for = RunId.for_args
case_id_for = CaseId.derive
tags_for_case = CaseId.tags_for
artifact_id_for = ArtifactId.derive
base_artifact_id = ArtifactId.from_image
gha_artifact_id = GhaArtifactId.derive
artifact_identity = ArtifactIdentity.parse
installed_digest = GhaArtifactId.installed_digest
capability_id_for = CapabilityId.derive
benchmark_id_for = BenchmarkId.derive
component_of = Component.of
