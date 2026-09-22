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

"""Derived identity for schema-v2 rows: run_id, test_case_id, artifact_id, component, arch.

Every id is DERIVED, never minted: independent writers must reach the same uuid for the same run
without coordinating. Changing any normalisation step here invalidates every id already written.
"""

import hashlib
import uuid

from .junit import _threaded_run_id, source_and_external_run_id

ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com")


ID_SEP = "|"


# The product this script ingests for by DEFAULT. Replaces v1's hf_/si_ table-name prefixes:
# one v2 table pair serves all three products, discriminated by this column. It is also a
# test_case_id hash input, so it cannot drift from the identity it is stamped on.
#
# A default, not a constant: a test cell may run ANOTHER component's suite through this script
# (hf-adapters' perf cell already does -- `ingest_script: ../torch-spyre/.github/scripts/
# ingest_xml.py` in its config.yaml), and hardcoding the owner stamped those rows
# 'torch-spyre'. Because component is a test_case_id hash input, that does not merely
# mislabel: the same test reconciles to a DIFFERENT identity depending on whose script ran it,
# and the docstring's own rule (group trends on (component, classname, name)) then splits one
# suite across two components. --component lets the caller name the component whose suite this
# actually is; product-test already knows it (config.yaml's `PRODUCT`).
COMPONENT_DEFAULT = "torch-spyre"


def _norm(value) -> str:
    """Canonical scalar form. Lowercasing is not cosmetic: the same tier arrives as
    'Regression' from a Jenkins parameter and 'regression' from a GHA input."""
    return ("" if value is None else str(value)).strip().lower()


def canonical_arch(arch) -> str:
    """amd64/x86/x86-64 all mean x86_64 -- a leg labelled 'amd64' by Jenkins and
    'x86_64' by GHA is ONE leg, and must hash as one."""
    a = _norm(arch)
    return "x86_64" if a in ("amd64", "x86", "x86-64", "x86_64") else a


def run_id_of(source: str, external_run_id: str, arch: str, test_type: str) -> str:
    """Identity of one TEST-EXECUTION LEG: (source, external_run_id, arch, test_type).

    arch and test_type are IN the key because the real execution grain measured
    (run, arch, tier) at 19,867 legs under 16,381 CI runs. external_run_id is a
    STRING: typed numerically, every Jenkins leg would be 0 and collide into one id.
    Returns '' when a field is missing -- an all-defaults hash is a real uuid that
    every incomplete leg would share, which is worse than a blank.
    """
    fields = (source, external_run_id, arch, test_type)
    if not all(_norm(f) for f in fields):
        return ""
    return str(
        uuid.uuid5(
            ID_NAMESPACE,
            ID_SEP.join(
                (
                    _norm(source),
                    _norm(external_run_id),
                    canonical_arch(arch),
                    _norm(test_type),
                )
            ),
        )
    )


def case_id_for(component: str, classname: str, name: str, tags) -> str:
    """Content identity of a TEST, so the same test reconciles across runs. v1 minted
    uuid4 per row: 37,322,701 identities for 58,711 distinct (classname, name) pairs.

    `tags` are deduped and SORTED -- they are a set and source order is incidental,
    so an unsorted join makes two writers disagree about the same test. They are
    INSIDE the hash, so re-tagging mints a new identity; trend queries must
    therefore group on (component, classname, name), never on test_case_id.
    """
    if not (_norm(component) and _norm(name)):
        # Same collision hazard as run_id_of: an empty field still hashes to a real,
        # stable uuid that every other such case shares. The v2 table's CONSTRAINTs
        # reject component='' / name='' anyway. classname is legitimately empty for a
        # module-level test, so it is NOT required.
        return ""
    norm = sorted({t for t in (_norm(x) for x in (tags or [])) if t})
    return str(
        uuid.uuid5(
            ID_NAMESPACE,
            ID_SEP.join(
                (
                    _norm(component),
                    _norm(classname),
                    _norm(name),
                    ",".join(norm),
                )
            ),
        )
    )


def artifact_id_for(component: str, artifact_name: str, id12: str, arch: str) -> str:
    """Content identity of one built artifact: (component, artifact_name, id12, arch).

    DERIVED, never minted. A consumer that knows only those four fields computes the same id
    the producer did, with nothing threaded to it -- which is the whole reason it is a hash and
    not a random uuid. v1's newRunId minted uuid4 and only 0.31% of artifact_results ever
    resolved against a run; an artifact id has the same exposure.

    The four fields also stay in `props` on the row, because the id is opaque once hashed and
    nothing downstream can parse a component or an arch back out of it.

    THE FOURTH COPY IS THE HAZARD THIS CLOSES. The Jenkins-side writer computes the same id by
    shelling out to an INLINED python3 script (twice), and a product ingest carried its own def.
    Three implementations of one hash, agreeing only by inspection: a drift of one normalisation
    step mints ids that silently never join -- no error, just rows that reference nothing.
    Verified byte-identical against live prod rows before being centralised here, across a
    folded arch (amd64 -> x86_64) and an unfolded one (ppc64le).
    """
    if not (_norm(component) and canonical_arch(arch)):
        # An all-blank hash is a real uuid that every incomplete artifact would share.
        return ""
    return str(
        uuid.uuid5(
            ID_NAMESPACE,
            ID_SEP.join(
                (
                    _norm(component),
                    _norm(artifact_name),
                    _norm(id12),
                    canonical_arch(arch),
                )
            ),
        )
    )


def gha_artifact_id(
    component: str, base_artifact_id: str, installed: str, arch: str
) -> str:
    """Artifact identity for a GHA leg that installed something on top of a prebaked image.

    Such a leg IS a different artifact from the image it started on, so it gets its own
    artifacts row (origin='gha', identity_deps=[base_artifact_id]) rather than borrowing the
    base image's id -- otherwise artifact_results.artifact_id would reference a row whose
    contents were never what ran.

    `base_artifact_id` is the base image's ALREADY-MINTED v2 artifact_id, read back from the
    image (OCI label / in-image file), not a name or a digest. That is what keeps this
    non-circular: the identity is a build INPUT the builder recorded, never a hash of the
    finished image -- baking a digest into the image it describes cannot converge.

    Only the GHA-side delta needs hashing, since everything in the base is recoverable from
    base_artifact_id. `installed` is normalised to a SORTED, deduped set: install order is
    incidental, and an order-sensitive hash would mint a fresh identity for a re-run of the
    same environment.

    A Jenkins-initiated run never reaches here: it runs the prebaked image UNCHANGED, so the
    embedded artifact_id is already correct and is used verbatim.
    """
    if not (_norm(component) and canonical_arch(arch)):
        return ""
    items = sorted({_norm(x) for x in (installed or "").replace(",", " ").split() if x})
    digest = (
        hashlib.sha256(ID_SEP.join(items).encode()).hexdigest()[:12] if items else ""
    )
    return artifact_id_for(component, _norm(base_artifact_id), digest, arch)


# Where the image build writes its own artifact_id, beside installed_rpms.txt. Set by
# spyre-frameworks' _package-image (--build-arg SPYRE_ARTIFACT_ID), which also stamps the
# same value as the `spyre.artifact.id` OCI label.
BASE_ARTIFACT_ID_FILE = "/home/senuser/spyre_artifact_id.txt"


def base_artifact_id(path: str = BASE_ARTIFACT_ID_FILE) -> str:
    """The prebaked image's own artifact_id, read from inside the image.

    The in-image FILE rather than the OCI label: a test leg runs INSIDE the container and has
    no registry credentials or skopeo there, so reading its own label would mean an outbound
    inspect of the image it is already running. The builder writes both from one value.

    Returns '' when absent or blank -- an image built before this existed, or a standalone
    build with no orchestrator node. Callers must treat that as "no base identity" and fall
    back to their own coordinate, never to a defaulted hash.
    """
    try:
        with open(path) as fh:
            return _norm(fh.read())
    except OSError:
        return ""


def component_of(args, default: str = COMPONENT_DEFAULT) -> str:
    """The component to stamp on v2 rows: --component when given, else `default`.

    The default is a parameter, not the module constant: each consuming repo has its own, and
    baking one in would make an importing repo stamp another product's name -- which, since
    component is a test_case_id hash input, silently mints a different identity.
    """
    return (getattr(args, "component", "") or "").strip() or default


def tags_for_case(case: dict) -> list:
    """The case's tags as an ARRAY of `namespace__value` strings.

    Array, not Map: `testtype` carries up to 5 values on 91.7% of cases, so a Map
    would silently keep one and drop the rest. The v1 shape is a (prop_name,
    prop_value) list where the only prop_name is literally 'tag' and the real
    key is encoded inside the value -- so the VALUE is the tag.
    """
    tags = set()
    for pname, pvalue in case.get("properties", []) or []:
        if pname == "tag":
            if pvalue:
                tags.add(pvalue)
        elif "__" in pname:
            # Some emitters put the namespace__value in the property NAME instead.
            tags.add(pname)
    return sorted(tags)


def run_id_for(args, run_id: str, arch: str, tier: str) -> str:
    """The v2 run_id for this leg: the THREADED uuid when there is one, else a derived hash.

    A uuid minted above the CI split (the orchestrator's newRunId(), arriving as --run-id) is
    the same value the Jenkins-side artifact_results writer records, so honouring it verbatim
    makes the two tables join on one identity -- with no agreement needed on a coordinate
    string's format, case, or arch folding.

    Not folded with arch/tier: one ingest invocation carries exactly one --trigger-type, so a
    multi-tier leg ingests once per tier and no row stands for two. artifact_results is ordered
    by (artifact_id, result_kind, test_type, ts), so rows stay distinct without run_id being
    unique per row.

    Falls back to the coordinate hash for a leg dispatched with no uuid (a standalone
    component-build, or a GHA-only run), which is the only case that still needs one.
    """
    threaded = _threaded_run_id(args)
    if threaded:
        return threaded
    source, external = source_and_external_run_id(args, run_id)
    return run_id_of(source, external, arch, tier)


def capability_id_for(
    component: str, test_type: str, subject: str, name: str, disc=None, disc_keys=()
) -> str:
    """Content identity of one (subject, capability) pair, so the same pair reconciles across
    runs and across the two producers.

    v1 minted a per-row surrogate instead -- model_ops_variants held 51,356 distinct
    variant_ids for 51,356 rows -- so it identified nothing and no two runs of one operation
    ever reconciled. This is the same fix case_id_for applied to v1's uuid4 per row.

    `test_type` is IN the hash, not just a column: model_ops' `aten::conv2d` and a hypothetical
    model_support adapter of the same name are different questions about the same subject, and
    an id that ignored it would merge them.

    `disc_keys` is positional and per-producer, exactly as benchmark_id_for: model_ops
    discriminates on input shapes/dtypes (2,788 (operation, test) pairs expand to 3,707 once
    counted), while model_support needs none. Reordering the keys mints new ids.

    `backend` is deliberately NOT hashed -- the same capability measured on cpu and on spyre is
    ONE capability with two verdicts, and it is the axis a support comparison pivots on.
    """
    if not (_norm(component) and _norm(test_type) and _norm(name)):
        # An empty field still hashes to a real uuid that every unidentifiable row would
        # share, which is worse than being orphaned. subject is NOT required: a capability
        # can be asked of the component as a whole rather than of one model.
        return ""
    disc = disc or {}
    # Absent keys are still emitted, so gaining a discriminator changes only that value.
    disc_part = ",".join(f"{k}={_norm(disc.get(k))}" for k in disc_keys)
    return str(
        uuid.uuid5(
            ID_NAMESPACE,
            ID_SEP.join(
                (
                    _norm(component),
                    _norm(test_type),
                    _norm(subject),
                    _norm(name),
                    disc_part,
                )
            ),
        )
    )


def benchmark_id_for(component: str, name: str, tags, disc=None, disc_keys=()) -> str:
    """Content identity of a benchmark, so the same benchmark reconciles across runs.

    component leads the hash, which is what lets each producer own its own `disc_keys` set
    without colliding with another's. `disc_keys` is positional: reordering it mints new ids.
    `backend` is deliberately not hashed -- it is the axis cross-backend comparison pivots on.
    """
    if not (_norm(component) and _norm(name)):
        # An empty field hashes to a real uuid, so every unidentifiable benchmark would
        # collide on one id rather than merely being orphaned.
        return ""
    tag_part = ",".join(sorted({t for t in (_norm(x) for x in (tags or [])) if t}))
    disc = disc or {}
    # Absent keys are still emitted, so gaining a discriminator value changes only that value.
    disc_part = ",".join(f"{k}={_norm(disc.get(k))}" for k in disc_keys)
    return str(
        uuid.uuid5(
            ID_NAMESPACE,
            ID_SEP.join((_norm(component), _norm(name), tag_part, disc_part)),
        )
    )
