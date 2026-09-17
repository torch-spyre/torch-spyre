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

"""Derived identity for schema-v2 rows: run_id, test_case_id, component, arch.

Every id is DERIVED, never minted: independent writers must reach the same uuid for the same run
without coordinating. Changing any normalisation step here invalidates every id already written.
"""

import uuid

from .junit import _threaded_run_id, v2_source_and_external_run_id

V2_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com")


V2_SEP = "|"


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
V2_COMPONENT_DEFAULT = "torch-spyre"


def _v2_norm(value) -> str:
    """Canonical scalar form. Lowercasing is not cosmetic: the same tier arrives as
    'Regression' from a Jenkins parameter and 'regression' from a GHA input."""
    return ("" if value is None else str(value)).strip().lower()


def v2_canonical_arch(arch) -> str:
    """amd64/x86/x86-64 all mean x86_64 -- a leg labelled 'amd64' by Jenkins and
    'x86_64' by GHA is ONE leg, and must hash as one."""
    a = _v2_norm(arch)
    return "x86_64" if a in ("amd64", "x86", "x86-64", "x86_64") else a


def v2_run_id(source: str, external_run_id: str, arch: str, test_type: str) -> str:
    """Identity of one TEST-EXECUTION LEG: (source, external_run_id, arch, test_type).

    arch and test_type are IN the key because the real execution grain measured
    (run, arch, tier) at 19,867 legs under 16,381 CI runs. external_run_id is a
    STRING: typed numerically, every Jenkins leg would be 0 and collide into one id.
    Returns '' when a field is missing -- an all-defaults hash is a real uuid that
    every incomplete leg would share, which is worse than a blank.
    """
    fields = (source, external_run_id, arch, test_type)
    if not all(_v2_norm(f) for f in fields):
        return ""
    return str(
        uuid.uuid5(
            V2_NAMESPACE,
            V2_SEP.join(
                (
                    _v2_norm(source),
                    _v2_norm(external_run_id),
                    v2_canonical_arch(arch),
                    _v2_norm(test_type),
                )
            ),
        )
    )


def v2_test_case_id(component: str, classname: str, name: str, tags) -> str:
    """Content identity of a TEST, so the same test reconciles across runs. v1 minted
    uuid4 per row: 37,322,701 identities for 58,711 distinct (classname, name) pairs.

    `tags` are deduped and SORTED -- they are a set and source order is incidental,
    so an unsorted join makes two writers disagree about the same test. They are
    INSIDE the hash, so re-tagging mints a new identity; trend queries must
    therefore group on (component, classname, name), never on test_case_id.
    """
    if not (_v2_norm(component) and _v2_norm(name)):
        # Same collision hazard as v2_run_id: an empty field still hashes to a real,
        # stable uuid that every other such case shares. The v2 table's CONSTRAINTs
        # reject component='' / name='' anyway. classname is legitimately empty for a
        # module-level test, so it is NOT required.
        return ""
    norm = sorted({t for t in (_v2_norm(x) for x in (tags or [])) if t})
    return str(
        uuid.uuid5(
            V2_NAMESPACE,
            V2_SEP.join(
                (
                    _v2_norm(component),
                    _v2_norm(classname),
                    _v2_norm(name),
                    ",".join(norm),
                )
            ),
        )
    )


def v2_component(args, default: str = V2_COMPONENT_DEFAULT) -> str:
    """The component to stamp on v2 rows: --component when given, else `default`.

    The default is a parameter, not the module constant: each consuming repo has its own, and
    baking one in would make an importing repo stamp another product's name -- which, since
    component is a test_case_id hash input, silently mints a different identity.
    """
    return (getattr(args, "component", "") or "").strip() or default


def v2_tags_for_case(case: dict) -> list:
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


def v2_run_id_for(args, run_id: str, arch: str, tier: str) -> str:
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
    source, external = v2_source_and_external_run_id(args, run_id)
    return v2_run_id(source, external, arch, tier)
