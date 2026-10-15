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

from .junit import RunCoordinates

ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com")

ID_SEP = "|"


# Stdlib only: derive_artifact_id.py imports this module on the runner's bare python3.
def _is_hex(s: str, n: int) -> bool:
    return len(s) == n and all(c in "0123456789abcdef" for c in s)


def _hex_token(text: str) -> str:
    """The last 12-hex token of `text` delimited by `.`, `-`, `_` or `+`; '' when none."""
    # str ops, not regex: derive-artifact-id imports this module with no third-party packages.
    parts = (text or "").translate(str.maketrans("-_+", "...")).split(".")
    for part in reversed(parts):
        if _is_hex(part, 12):
            return part
    return ""


def _strip_dev_suffix(name: str) -> str:
    for suffix in ("-devel", "-dev"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# The component stamped on rows when the caller names none. A DEFAULT, not a constant: a
# test cell may run another component's suite, and component is a hash input.
COMPONENT_DEFAULT = "torch-spyre"

# Tag namespaces that say where/when a test ran (arch, test type, cadence), not what it is.
# Never hashed, so one test keeps one id across arches and test types; stored per run. A
# deny-list: an unclassified namespace can only leave an id split, never merge two tests.
RUN_CONTEXT_TAG_NAMESPACES = frozenset({"platform", "testtype", "cadence"})

# Tag namespaces that carry a measured value (`refcoverage__48/48`): not membership at all, so
# neither hashed nor tagged -- stored as props['result.<ns>'].
RESULT_TAG_NAMESPACES = frozenset({"refcoverage"})

# Bare tags older emitters wrote, mapped to their namespaced form; migrations/006 applies the
# same map to history.
LEGACY_TAG_ALIASES = {
    "nightly": "cadence__nightly",
    "weekly": "cadence__weekly",
    "fvt": "testtype__fvt",
    "svt": "testtype__svt",
    "spyre-inference": "domain__spyre-inference",
    "spyre-backend": "domain__spyre-backend",
    "torch-spyre": "domain__torch-spyre",
}

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

    @staticmethod
    def canon_tag(tag) -> str:
        """A legacy bare tag in its namespaced form (lowercase); any other tag unchanged."""
        return LEGACY_TAG_ALIASES.get(DerivedId.norm(tag), tag)

    @staticmethod
    def namespace(tag) -> str:
        return DerivedId.norm(DerivedId.canon_tag(tag)).split("__", 1)[0]

    @classmethod
    def split_tags(cls, tags) -> tuple:
        """(identity tags, run-context tags, result props) -- what a test is, where it ran,
        and values it measured."""
        ident, ctx, results = set(), set(), {}
        for t in (cls.canon_tag(x) for x in (tags or []) if cls.norm(x)):
            ns = cls.namespace(t)
            if ns in RESULT_TAG_NAMESPACES:
                results[f"result.{ns}"] = t.split("__", 1)[1] if "__" in t else ""
            elif ns in RUN_CONTEXT_TAG_NAMESPACES:
                ctx.add(t)
            else:
                ident.add(t)
        return sorted(ident), sorted(ctx), results

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
    def tag_part(cls, tags) -> str:
        """Only the identity tags: run-context and result tags never reach the hash."""
        return super().tag_part(cls.split_tags(tags)[0])

    @classmethod
    def derive(cls, component: str, classname: str, name: str, tags) -> str:
        """The test's uuid, or '' with no component/name; classname may be blank."""
        if not cls.complete(component, name):
            return ""
        # name keeps its case: sibling tests can differ only by case (upstream test_T / test_t).
        return cls.hash(
            cls.norm(component),
            cls.norm(classname),
            str(name).strip(),
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
        if not _is_hex(digest.removeprefix("sha256:"), 64) or not digest.startswith(
            "sha256:"
        ):
            raise ValueError(f"image ref needs an @sha256:<64 hex> digest: {ref!r}")
        if id12 and not _is_hex(id12, 12):
            raise ValueError(f"id12 must be 12 hex characters: {id12!r}")
        repo_name = repo.rsplit("/", 1)[-1].split(":", 1)[0]
        return cls(
            component=component or _strip_dev_suffix(repo_name),
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
        if not url or not _is_hex(digest, 64):
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
    def from_rpm(
        cls, ref: str, arch: str, component: str = "", name: str = "", id12: str = ""
    ) -> "ArtifactIdentity":
        """An RPM by file name, URL, NEVRA or dnf glob; id12 is the 12-hex identity token the
        producer puts in its release (`<name>-*.<id12>.*.<arch>`)."""
        base = (ref or "").strip().rsplit("/", 1)[-1].removesuffix(".rpm")
        rpm_name = name or (
            base.split("-*", 1)[0] if "-*" in base else base.rsplit("-", 2)[0]
        )
        token = id12 or _hex_token(base)
        if not (rpm_name and _is_hex(token, 12)):
            raise ValueError(f"rpm needs a name and a 12-hex identity token: {ref!r}")
        return cls(
            component=component or rpm_name,
            artifact_name=rpm_name,
            id12=token,
            arch=DerivedId.arch(arch),
            kind="rpm",
            ref=(ref or "").strip(),
        )

    @classmethod
    def from_wheel(
        cls, ref: str, arch: str, component: str = "", name: str = "", id12: str = ""
    ) -> "ArtifactIdentity":
        """A wheel by `name==version`, file name or URL; id12 is the 12-hex identity token
        ending its local version (`+<id12>`, `+cpu.<id12>`)."""
        raw = (ref or "").strip()
        if "==" in raw:
            dist, _, version = raw.partition("==")
        else:
            # PEP 427: `{distribution}-{version}-...whl`, `-` in the name escaped as `_`.
            dist, version = (raw.rsplit("/", 1)[-1].split("-") + [""])[:2]
        token = id12 or _hex_token(version.rpartition("+")[2])
        dist = name or dist
        if not (dist and version and _is_hex(token, 12)):
            raise ValueError(f"wheel needs name==version+<12-hex id>: {ref!r}")
        return cls(
            component=component or dist,
            artifact_name=dist,
            id12=token,
            arch=DerivedId.arch(arch),
            kind="wheel",
            ref=f"{dist}=={version}",
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

    # A compiled kernel's name ends in a per-compile token, `_` + 16 of [a-z0-9], before an
    # optional `#<n>`. migrations/012 matches the same names in SQL; a test pins both.
    KERNEL_PREFIX = "spyre_kernel_"
    KERNEL_TOKEN = 16

    @classmethod
    def kernel_stem(cls, kernel_name) -> str:
        """A compiled kernel's name without its per-compile token; '' for any other name."""
        name = "" if kernel_name is None else str(kernel_name)
        head, sep, n = name.rpartition("#")
        if not (sep and n and all(c in "0123456789" for c in n)):
            head, sep, n = name, "", ""
        token = head[-cls.KERNEL_TOKEN - 1 :]
        if not (
            name.startswith(cls.KERNEL_PREFIX)
            and len(token) == cls.KERNEL_TOKEN + 1
            and token[0] == "_"
            and all(c in "abcdefghijklmnopqrstuvwxyz0123456789" for c in token[1:])
        ):
            return ""
        return head[: -len(token)] + sep + n

    @classmethod
    def rank_kernels(cls, component: str, entries: list) -> list:
        """`entries` with each compiled kernel's hashed kernel_name as `<stem>@<rank>`.

        One op compiles several kernels with the same stem (two `fused_add`s at 0.27 and
        0.003 ms), so the stem alone would merge them: rank 1 is the slowest by duration_ms
        among the kernels sharing every other identity input, ties broken by raw name. The
        key is also kept as props['kernel_key'], the stable label; the raw name, which changes
        on every compile, goes to run_props.
        """
        keyed = []
        samples: dict[str, dict[str, dict[str, list]]] = {}
        for e in entries:
            disc = e.get("disc") or {}
            keys = e.get("disc_keys") or ()
            stem = cls.kernel_stem(disc.get("kernel_name"))
            if not (stem and e.get("measurements") and "kernel_name" in keys):
                keyed.append(None)
                continue
            raw = str(disc["kernel_name"])
            group = cls.derive(
                component,
                e.get("name", ""),
                e.get("tags"),
                {**disc, "kernel_name": stem},
                keys,
            )
            by_backend = samples.setdefault(group, {}).setdefault(raw, {})
            by_backend.setdefault(e.get("backend", ""), []).extend(
                e["measurements"].get("duration_ms") or []
            )
            keyed.append((group, stem, raw))
        # A kernel's duration is its slowest backend's mean, as one benchmark_runs row holds it.
        dur = {
            (g, raw): max(sum(d) / len(d) if d else 0.0 for d in by_backend.values())
            for g, kernels in samples.items()
            for raw, by_backend in kernels.items()
        }
        rank = {
            (g, raw): n
            for g, kernels in samples.items()
            for n, raw in enumerate(sorted(kernels, key=lambda r: (-dur[g, r], r)), 1)
        }
        out = []
        for e, k in zip(entries, keyed):
            if k is None:
                out.append(e)
                continue
            key = f"{k[1]}@{rank[k[0], k[2]]}"
            out.append(
                {
                    **e,
                    "disc": {**e["disc"], "kernel_name": key},
                    "props": {**(e.get("props") or {}), "kernel_key": key},
                    "run_props": {**(e.get("run_props") or {}), "kernel_name": k[2]},
                }
            )
        return out

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
split_case_tags = CaseId.split_tags
artifact_id_for = ArtifactId.derive
base_artifact_id = ArtifactId.from_image
gha_artifact_id = GhaArtifactId.derive
artifact_identity = ArtifactIdentity.parse
installed_digest = GhaArtifactId.installed_digest
capability_id_for = CapabilityId.derive
benchmark_id_for = BenchmarkId.derive
component_of = Component.of
