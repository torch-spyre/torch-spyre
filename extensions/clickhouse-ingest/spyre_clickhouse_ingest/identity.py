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
import uuid

from .junit import RunCoordinates

ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com")

ID_SEP = "|"

# The component stamped on rows when the caller names none. A DEFAULT, not a constant: a
# test cell may run another component's suite, and component is a hash input.
COMPONENT_DEFAULT = "torch-spyre"

# Where the image build writes its own artifact_id, beside installed_rpms.txt.
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
    def from_image(cls, path: str = "") -> str:
        """The prebaked image's own artifact_id, read from inside it; '' when absent."""
        try:
            with open(path or cls.FILE) as fh:
                return cls.norm(fh.read())
        except OSError:
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
installed_digest = GhaArtifactId.installed_digest
capability_id_for = CapabilityId.derive
benchmark_id_for = BenchmarkId.derive
component_of = Component.of
