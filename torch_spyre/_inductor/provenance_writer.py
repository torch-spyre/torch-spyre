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

"""Deterministic validation, merge, and atomic Spyre sidecar publication."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import regex as re
import torch
from torch._inductor import debug as inductor_debug
from torch._logging._internal import trace_log, trace_structured_artifact

from torch_spyre._inductor.kernel_provenance import (
    KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
    KERNEL_PROVENANCE_KEY_VERSION,
    KernelProvenanceDescriptor,
)
from torch_spyre._inductor.profiler_event import (
    format_kernel_provenance_event_name,
)
from torch_spyre._inductor.provenance_artifact import CollectedProvenance
from torch_spyre.version import __version__ as torch_spyre_version


_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None


PublicationResult = Literal["disabled", "unchanged", "written"]
UpstreamJoin = Literal[
    "ok",
    "partial",
    "unavailable-cache-replay",
    "unavailable-provenance-level-0",
]

_SCHEMA_VERSION = 1
_UPSTREAM_SOURCE = "inductor_provenance_tracking_node_mappings"
_UPSTREAM_VERSION = 2.0
_COMPILE_ID_DOMAIN = "torch-spyre-compile-v1"
_OCCURRENCE_ID_DOMAIN = "torch-spyre-occurrence-v1"
_EVENT_KEY_PATTERN = re.compile(rf"^[a-z2-7]{{{KERNEL_PROVENANCE_KEY_BASE32_WIDTH}}}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_ID_PATTERN = re.compile(r"^(0|[1-9][0-9]*)$")
_ALIAS_PATTERN = re.compile(r"^.+:[1-9][0-9]*$")
_EVENT_NAME_PATTERN = re.compile(
    rf"^spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_"
    rf"[A-Za-z0-9_]+_[a-z2-7]{{{KERNEL_PROVENANCE_KEY_BASE32_WIDTH}}}$"
)
_TOP_LEVEL_KEYS = {
    "schemaVersion",
    "eventKey",
    "mergeGeneration",
    "status",
    "diagnostics",
    "handles",
    "kernelIdentities",
    "kernelOccurrences",
    "upstreamProjections",
}
_PROJECTION_KEYS = {
    "source",
    "version",
    "producer",
    "settings",
    "kernels",
    "uncollectedKernels",
    "upstreamProjectionFailed",
    "upstreamJoin",
    "preToPost",
    "postToPre",
    "cppCodeToPost",
    "postToCppCode",
    "kernelStackTraces",
}
# Prefer the most actionable evidence when v1's scalar join status must summarize
# several equivalent contributions. A level-one cache replay outranks a level-zero
# fresh compile because obtaining the missing mapping then requires a fresh
# level-one compile; v2 should preserve every observed reason instead.
_JOIN_RANK = {
    "unavailable-provenance-level-0": 0,
    "unavailable-cache-replay": 1,
    "partial": 2,
    "ok": 3,
}
_TRANSFORM_KINDS = {"rewrite", "fusion", "decomposition", "clone", "remap"}
_DIAGNOSTIC_CODES = {"collection-failure", "upstream-projection-failure"}
_PUBLICATION_LOCK = threading.Lock()


class ProvenanceArtifactError(RuntimeError):
    """Safe publication or validation failure with no configured path leak."""


class UnsupportedProvenanceVersion(ProvenanceArtifactError):
    """Existing sidecar uses a schema version this writer cannot merge."""


@dataclasses.dataclass(frozen=True)
class CapturedUpstreamProjection:
    """Filtered live Inductor relationships for one wrapper contribution."""

    upstream_join: UpstreamJoin
    pre_to_post: Mapping[str, tuple[str, ...]]
    post_to_pre: Mapping[str, tuple[str, ...]]
    cpp_code_to_post: Mapping[str, tuple[str, ...]]
    post_to_cpp_code: Mapping[str, tuple[str, ...]]
    kernel_stack_traces: Mapping[str, Mapping[str, tuple[str, ...]]]
    failed: bool

    def to_projection_fields(self) -> dict[str, object]:
        return {
            "preToPost": _lists_from_tuples(self.pre_to_post),
            "postToPre": _lists_from_tuples(self.post_to_pre),
            "cppCodeToPost": _lists_from_tuples(self.cpp_code_to_post),
            "postToCppCode": _lists_from_tuples(self.post_to_cpp_code),
            "kernelStackTraces": {
                alias: _lists_from_tuples(context)
                for alias, context in self.kernel_stack_traces.items()
            },
        }


def resolve_provenance_artifact_path(
    configured_path: str | None,
) -> Path | None:
    """Resolve one exact configured filename at publication time."""
    if not configured_path:
        return None
    path = Path(configured_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.absolute()


def capture_upstream_projection(
    collection: CollectedProvenance,
) -> CapturedUpstreamProjection | None:
    """Capture and filter live upstream v2 state for one fresh contribution."""
    level = int(torch._inductor.config.trace.provenance_tracking_level)
    if level < 1 or not collection.has_graph_lowering:
        return None

    node_mapping = inductor_debug.dump_inductor_provenance_info()
    kernel_information = inductor_debug.create_kernel_information_json()
    mapping = _require_mapping(node_mapping, "upstream node mapping")
    if mapping.get("version") != _UPSTREAM_VERSION:
        raise ProvenanceArtifactError("unsupported upstream provenance version")

    aliases = sorted(
        {
            registration.alias
            for occurrence in collection.kernel_occurrences.values()
            for registration in occurrence.registrations
        }
    )
    upstream_cpp_to_post = _capture_name_map(
        mapping.get("cppCodeToPost"), "upstream cppCodeToPost"
    )
    upstream_post_to_pre = _capture_name_map(
        mapping.get("postToPre"), "upstream postToPre"
    )
    cpp_to_post = {
        alias: upstream_cpp_to_post[alias]
        for alias in aliases
        if alias in upstream_cpp_to_post
    }
    reachable_post_nodes = {
        post_node for values in cpp_to_post.values() for post_node in values
    }
    filtered_post_to_pre = {
        post_node: upstream_post_to_pre[post_node]
        for post_node in sorted(reachable_post_nodes)
        if post_node in upstream_post_to_pre
    }
    pre_to_post = _tuple_name_map(_invert_name_map(filtered_post_to_pre))
    post_to_pre = _tuple_name_map(_invert_name_map(pre_to_post))
    post_to_cpp = _tuple_name_map(_invert_name_map(cpp_to_post))

    upstream_stacks = _require_mapping(
        kernel_information, "upstream kernel information"
    )
    stack_contexts: dict[str, Mapping[str, tuple[str, ...]]] = {}
    stack_contexts_match = True
    for alias in aliases:
        value = upstream_stacks.get(alias)
        if value is None:
            continue
        context = _require_mapping(value, "upstream kernel stack context")
        stack_traces = _capture_string_tuple(
            context.get("stack_traces"), "upstream stack traces"
        )
        post_grad_nodes = _capture_string_tuple(
            context.get("post_grad_nodes"), "upstream stack post-grad nodes"
        )
        pre_grad_nodes = _capture_string_tuple(
            context.get("pre_grad_nodes"), "upstream stack pre-grad nodes"
        )
        expected_post_nodes = cpp_to_post.get(alias, ())
        expected_pre_nodes = tuple(
            dict.fromkeys(
                pre_node
                for post_node in expected_post_nodes
                for pre_node in filtered_post_to_pre.get(post_node, ())
            )
        )
        stack_contexts_match &= (
            post_grad_nodes == expected_post_nodes
            and pre_grad_nodes == expected_pre_nodes
        )
        stack_contexts[alias] = {
            "stackTraces": stack_traces,
            "postGradNodes": post_grad_nodes,
            "preGradNodes": pre_grad_nodes,
        }

    complete = (
        bool(aliases)
        and set(cpp_to_post) == set(aliases)
        and all(cpp_to_post.values())
        and set(post_to_pre) == reachable_post_nodes
        and set(stack_contexts) == set(aliases)
        and stack_contexts_match
    )
    return CapturedUpstreamProjection(
        upstream_join="ok" if complete else "partial",
        pre_to_post=pre_to_post,
        post_to_pre=post_to_pre,
        cpp_code_to_post=cpp_to_post,
        post_to_cpp_code=post_to_cpp,
        kernel_stack_traces=stack_contexts,
        failed=not complete,
    )


def publish_provenance_collection(
    collection: CollectedProvenance,
    configured_path: str | None,
    *,
    upstream_projection: CapturedUpstreamProjection | None = None,
    upstream_projection_failed: bool = False,
) -> PublicationResult:
    """Merge and atomically publish one wrapper's collection outcome."""
    path = resolve_provenance_artifact_path(configured_path)
    if path is None:
        return "disabled"

    contribution = _build_contribution(
        collection, upstream_projection_failed, upstream_projection
    )
    result: PublicationResult = "written"
    with _PUBLICATION_LOCK:
        with _interprocess_publication_lock(path):
            existing = _read_existing_document(path)
            if existing is None:
                if not contribution["kernelIdentities"]:
                    raise ProvenanceArtifactError(
                        f"cannot create {path.name} without a collected kernel identity"
                    )
                document = _new_document(contribution)
            else:
                validate_provenance_document(existing)
                document = _merge_document(existing, contribution)
                if _without_generation(document) == _without_generation(existing):
                    result = "unchanged"

            document = _canonicalize_document(document)
            validate_provenance_document(document)
            payload = _serialize_document(document)
            if result == "written":
                _atomic_write(path, payload)

    # Always submit enabled publications to PyTorch logging. Its handlers own
    # durable trace emission; structuredTracing only records destination state.
    trace_structured_artifact(
        "spyre_provenance",
        "json",
        payload_fn=lambda: payload,
    )
    return result


def validate_provenance_document(document: object) -> None:
    """Validate the closed v1 schema plus referential and derivation rules."""
    if not isinstance(document, dict):
        raise ProvenanceArtifactError("sidecar root must be an object")
    schema_version = document.get("schemaVersion")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _SCHEMA_VERSION
    ):
        raise UnsupportedProvenanceVersion("unsupported provenance schema version")
    _require_exact_keys(document, _TOP_LEVEL_KEYS, "sidecar")

    _validate_event_key(document["eventKey"])
    _require_int(document["mergeGeneration"], "mergeGeneration", minimum=1)
    if document["status"] not in {"complete", "partial"}:
        raise ProvenanceArtifactError("invalid document status")
    _validate_diagnostics(document["diagnostics"])

    handles = _require_mapping(document["handles"], "handles")
    identities = _require_mapping(document["kernelIdentities"], "kernelIdentities")
    occurrences = _require_mapping(document["kernelOccurrences"], "kernelOccurrences")
    projections = _require_mapping(
        document["upstreamProjections"], "upstreamProjections"
    )
    if not identities or not occurrences or not projections:
        raise ProvenanceArtifactError(
            "identities, occurrences, and projections must be non-empty"
        )
    for name, value in (
        ("handles", handles),
        ("kernelIdentities", identities),
        ("kernelOccurrences", occurrences),
        ("upstreamProjections", projections),
    ):
        _require_sorted_keys(value, name)

    for handle_id, handle in handles.items():
        if not isinstance(handle_id, str) or not _DECIMAL_ID_PATTERN.fullmatch(
            handle_id
        ):
            raise ProvenanceArtifactError("invalid debug handle key")
        _validate_handle(handle_id, handle, handles)

    for identity_key, identity in identities.items():
        if not isinstance(identity_key, str) or not _EVENT_KEY_PATTERN.fullmatch(
            identity_key
        ):
            raise ProvenanceArtifactError("invalid kernel identity key")
        _validate_identity(identity_key, identity, handles)

    occurrences_by_compile: dict[str, list[Mapping[str, object]]] = {}
    for occurrence_id, occurrence in occurrences.items():
        if not isinstance(occurrence_id, str) or not _SHA256_PATTERN.fullmatch(
            occurrence_id
        ):
            raise ProvenanceArtifactError("invalid kernel occurrence key")
        validated = _validate_occurrence(
            occurrence_id,
            occurrence,
            identities,
            projections,
        )
        compile_id = validated["compileId"]
        assert isinstance(compile_id, str)
        occurrences_by_compile.setdefault(compile_id, []).append(validated)

    for compile_id, projection in projections.items():
        if not isinstance(compile_id, str) or not _SHA256_PATTERN.fullmatch(compile_id):
            raise ProvenanceArtifactError("invalid upstream projection key")
        _validate_projection(
            compile_id,
            projection,
            identities,
            occurrences_by_compile.get(compile_id, []),
        )
    if set(occurrences_by_compile) != set(projections):
        raise ProvenanceArtifactError("occurrence and projection compile IDs disagree")
    if document["diagnostics"] != _diagnostics_from_projections(projections):
        raise ProvenanceArtifactError(
            "writer diagnostics disagree with compile projections"
        )

    expected_status = (
        "complete"
        if not document["diagnostics"]
        and all(
            projection["upstreamJoin"] == "ok" for projection in projections.values()
        )
        else "partial"
    )
    if document["status"] != expected_status:
        raise ProvenanceArtifactError("document status disagrees with its content")


def _build_contribution(
    collection: CollectedProvenance,
    upstream_projection_failed: bool,
    upstream_projection: CapturedUpstreamProjection | None,
) -> dict[str, Any]:
    handles = {
        handle_id: handle.to_dict() for handle_id, handle in collection.handles.items()
    }
    identities = {
        identity_key: identity.to_dict()
        for identity_key, identity in collection.kernel_identities.items()
    }
    occurrences = {
        occurrence_id: occurrence.to_dict()
        for occurrence_id, occurrence in collection.kernel_occurrences.items()
    }
    projections: dict[str, object] = {}
    if collection.kernels:
        level = int(torch._inductor.config.trace.provenance_tracking_level)
        if level < 1:
            upstream_join = "unavailable-provenance-level-0"
        elif not collection.has_graph_lowering:
            upstream_join = "unavailable-cache-replay"
        elif collection.registration_capture_failed or upstream_projection_failed:
            upstream_join = "partial"
        else:
            upstream_join = (
                upstream_projection.upstream_join
                if upstream_projection is not None
                else "partial"
            )
        projections[collection.compile_id] = {
            "source": _UPSTREAM_SOURCE,
            "version": _UPSTREAM_VERSION,
            "producer": {
                "torchSpyreVersion": torch_spyre_version,
                "torchVersion": torch.__version__,
            },
            "settings": {
                "provenanceTrackingLevel": level,
                "structuredTracing": _structured_tracing_enabled(),
            },
            "kernels": [
                {
                    "compilerKernelName": kernel_name,
                    "identityKey": identity_key,
                }
                for kernel_name, identity_key in collection.kernels
            ],
            "uncollectedKernels": sorted(
                kernel.compiler_kernel_name for kernel in collection.uncollected_kernels
            ),
            "upstreamProjectionFailed": (
                collection.registration_capture_failed
                or upstream_projection_failed
                or (upstream_projection.failed if upstream_projection else False)
            ),
            "upstreamJoin": upstream_join,
            "preToPost": {},
            "postToPre": {},
            "cppCodeToPost": {},
            "postToCppCode": {},
            "kernelStackTraces": {},
        }
        if upstream_projection is not None:
            projections[collection.compile_id].update(
                upstream_projection.to_projection_fields()
            )
    return {
        "diagnostics": _diagnostics_from_projections(projections),
        "handles": handles,
        "kernelIdentities": identities,
        "kernelOccurrences": occurrences,
        "upstreamProjections": projections,
    }


def _new_document(contribution: Mapping[str, object]) -> dict[str, Any]:
    document = {
        "schemaVersion": _SCHEMA_VERSION,
        "eventKey": {
            "version": KERNEL_PROVENANCE_KEY_VERSION,
            "algorithm": "sha256-prefix-80-base32-lower",
            "width": KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
            "stepAttribution": "bundle",
        },
        "mergeGeneration": 1,
        "status": "partial",
        **copy.deepcopy(contribution),
    }
    _derive_status(document)
    return document


def _merge_document(
    existing: Mapping[str, object],
    contribution: Mapping[str, object],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(existing))
    _merge_equal_records(merged["handles"], contribution["handles"], "debug handle")
    _merge_equal_records(
        merged["kernelIdentities"],
        contribution["kernelIdentities"],
        "kernel identity",
    )
    _merge_occurrences(merged["kernelOccurrences"], contribution["kernelOccurrences"])
    _merge_projections(
        merged["upstreamProjections"], contribution["upstreamProjections"]
    )

    merged["diagnostics"] = _diagnostics_from_projections(
        _require_mapping(merged["upstreamProjections"], "upstream projections")
    )
    _derive_status(merged)
    if _without_generation(merged) != _without_generation(existing):
        generation = existing["mergeGeneration"]
        assert isinstance(generation, int)
        merged["mergeGeneration"] = generation + 1
    return merged


def _merge_equal_records(
    destination_value: object,
    incoming_value: object,
    record_name: str,
) -> None:
    destination = _require_mapping(destination_value, record_name)
    incoming = _require_mapping(incoming_value, f"incoming {record_name}")
    for key, value in incoming.items():
        existing = destination.get(key)
        if existing is not None and existing != value:
            raise ProvenanceArtifactError(
                f"conflicting {record_name} content for key {key}"
            )
        destination[key] = copy.deepcopy(value)


def _merge_occurrences(destination_value: object, incoming_value: object) -> None:
    destination = _require_mapping(destination_value, "kernel occurrences")
    incoming = _require_mapping(incoming_value, "incoming kernel occurrences")
    for occurrence_id, occurrence_value in incoming.items():
        occurrence = _require_mapping(occurrence_value, "kernel occurrence")
        existing_value = destination.get(occurrence_id)
        if existing_value is None:
            destination[occurrence_id] = copy.deepcopy(occurrence)
            continue
        existing = _require_mapping(existing_value, "existing kernel occurrence")
        for field in ("identityKey", "compileId", "compilerKernelName", "selector"):
            if existing.get(field) != occurrence.get(field):
                raise ProvenanceArtifactError(
                    f"conflicting kernel occurrence content for key {occurrence_id}"
                )
        registrations = {
            (item["ordinal"], item["alias"]): copy.deepcopy(item)
            for item in existing["registrations"]
        }
        for item in occurrence["registrations"]:
            ordinal = item["ordinal"]
            if any(
                key[0] == ordinal and key[1] != item["alias"] for key in registrations
            ):
                raise ProvenanceArtifactError(
                    f"conflicting registration ordinal {ordinal}"
                )
            registrations[(ordinal, item["alias"])] = copy.deepcopy(item)
        existing["registrations"] = [
            registrations[key] for key in sorted(registrations)
        ]


def _merge_projections(destination_value: object, incoming_value: object) -> None:
    destination = _require_mapping(destination_value, "upstream projections")
    incoming = _require_mapping(incoming_value, "incoming upstream projections")
    for compile_id, projection_value in incoming.items():
        projection = _require_mapping(projection_value, "upstream projection")
        existing_value = destination.get(compile_id)
        if existing_value is None:
            destination[compile_id] = copy.deepcopy(projection)
            continue
        existing = _require_mapping(existing_value, "existing upstream projection")
        for field in ("source", "version", "producer", "kernels"):
            if existing[field] != projection[field]:
                raise ProvenanceArtifactError(
                    f"conflicting upstream projection metadata for key {compile_id}"
                )
        settings = _merge_projection_settings(
            existing["settings"], projection["settings"]
        )
        uncollected_kernels = sorted(
            set(existing["uncollectedKernels"]) | set(projection["uncollectedKernels"])
        )
        upstream_projection_failed = bool(
            existing["upstreamProjectionFailed"]
            or projection["upstreamProjectionFailed"]
        )
        existing_rank = _JOIN_RANK[existing["upstreamJoin"]]
        incoming_rank = _JOIN_RANK[projection["upstreamJoin"]]
        if existing_rank > incoming_rank:
            _require_projection_subset(projection, existing, compile_id)
            existing["settings"] = settings
            existing["uncollectedKernels"] = uncollected_kernels
            existing["upstreamProjectionFailed"] = upstream_projection_failed
            continue
        if incoming_rank > existing_rank:
            _require_projection_subset(existing, projection, compile_id)
            replacement = copy.deepcopy(projection)
            replacement["settings"] = settings
            replacement["uncollectedKernels"] = uncollected_kernels
            replacement["upstreamProjectionFailed"] = upstream_projection_failed
            destination[compile_id] = replacement
            continue
        # Defensive drift guard: unreachable while _JOIN_RANK values are unique,
        # but intentionally loud if a future status shares an existing rank
        # without defining an equivalence contract.
        if existing["upstreamJoin"] != projection["upstreamJoin"]:
            raise ProvenanceArtifactError(
                f"conflicting upstream join status for key {compile_id}"
            )
        # Equivalent wrappers can be compiled from different source call sites.
        # Their graph relations and stack contexts are additive even when both
        # captures are complete, so merge them instead of treating inequality as
        # a content conflict.
        _merge_projection_content(existing, projection)
        existing["settings"] = settings
        existing["uncollectedKernels"] = uncollected_kernels
        existing["upstreamProjectionFailed"] = upstream_projection_failed


def _merge_projection_settings(
    first_value: object, second_value: object
) -> dict[str, object]:
    """Retain the strongest settings observed for one merged compile ID."""
    first = _require_mapping(first_value, "projection settings")
    second = _require_mapping(second_value, "incoming projection settings")
    expected_keys = {"provenanceTrackingLevel", "structuredTracing"}
    _require_exact_keys(first, expected_keys, "projection settings")
    _require_exact_keys(second, expected_keys, "incoming projection settings")
    first_level = _require_int(
        first["provenanceTrackingLevel"],
        "provenance tracking level",
        minimum=0,
    )
    second_level = _require_int(
        second["provenanceTrackingLevel"],
        "incoming provenance tracking level",
        minimum=0,
    )
    first_tracing = first["structuredTracing"]
    second_tracing = second["structuredTracing"]
    if not isinstance(first_tracing, bool) or not isinstance(second_tracing, bool):
        raise ProvenanceArtifactError("structured tracing setting must be boolean")
    return {
        "provenanceTrackingLevel": max(first_level, second_level),
        "structuredTracing": first_tracing or second_tracing,
    }


def _merge_projection_content(
    destination: dict[str, Any], incoming: Mapping[str, object]
) -> None:
    for field in ("preToPost", "cppCodeToPost"):
        destination[field] = _merge_name_maps(destination[field], incoming[field])
    destination["postToPre"] = _invert_name_map(destination["preToPost"])
    destination["postToCppCode"] = _invert_name_map(destination["cppCodeToPost"])

    stacks = _require_mapping(destination["kernelStackTraces"], "stack context")
    incoming_stacks = _require_mapping(
        incoming["kernelStackTraces"], "incoming stack context"
    )
    for alias, context_value in incoming_stacks.items():
        context = _require_mapping(context_value, "kernel stack context")
        if alias not in stacks:
            stacks[alias] = copy.deepcopy(context)
            continue
        existing_context = _require_mapping(stacks[alias], "kernel stack context")
        for field in ("stackTraces", "postGradNodes", "preGradNodes"):
            existing_context[field] = sorted(
                set(existing_context[field]) | set(context[field])
            )


def _require_projection_subset(
    weaker: Mapping[str, object],
    richer: Mapping[str, object],
    compile_id: str,
) -> None:
    for field in ("preToPost", "postToPre", "cppCodeToPost", "postToCppCode"):
        if not _name_map_is_subset(weaker[field], richer[field]):
            raise ProvenanceArtifactError(
                f"weaker projection conflicts with richer content for key {compile_id}"
            )
    weaker_stacks = _require_mapping(weaker["kernelStackTraces"], "stack context")
    richer_stacks = _require_mapping(richer["kernelStackTraces"], "stack context")
    for alias, context_value in weaker_stacks.items():
        if alias not in richer_stacks:
            raise ProvenanceArtifactError(
                f"weaker projection conflicts with richer content for key {compile_id}"
            )
        context = _require_mapping(context_value, "kernel stack context")
        richer_context = _require_mapping(
            richer_stacks[alias], "richer kernel stack context"
        )
        for field in ("stackTraces", "postGradNodes", "preGradNodes"):
            if not set(context[field]) <= set(richer_context[field]):
                raise ProvenanceArtifactError(
                    "weaker projection conflicts with richer content for key "
                    f"{compile_id}"
                )


def _name_map_is_subset(weaker_value: object, richer_value: object) -> bool:
    weaker = _require_mapping(weaker_value, "name map")
    richer = _require_mapping(richer_value, "name map")
    return all(
        key in richer and set(values) <= set(richer[key])
        for key, values in weaker.items()
    )


def _merge_name_maps(first_value: object, second_value: object) -> dict[str, list[str]]:
    first = _require_mapping(first_value, "name map")
    second = _require_mapping(second_value, "name map")
    keys = sorted(set(first) | set(second))
    return {
        key: sorted(set(first.get(key, [])) | set(second.get(key, []))) for key in keys
    }


def _invert_name_map(mapping_value: object) -> dict[str, list[str]]:
    mapping = _require_mapping(mapping_value, "name map")
    inverse: dict[str, list[str]] = {}
    for source in sorted(mapping):
        for destination in mapping[source]:
            inverse.setdefault(destination, []).append(source)
    return inverse


def _capture_name_map(value: object, name: str) -> dict[str, tuple[str, ...]]:
    mapping = _require_mapping(value, name)
    return {key: _capture_string_tuple(values, name) for key, values in mapping.items()}


def _capture_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProvenanceArtifactError(f"{name} must be an array")
    if not all(isinstance(item, str) for item in value):
        raise ProvenanceArtifactError(f"{name} entries must be strings")
    if len(value) != len(set(value)):
        raise ProvenanceArtifactError(f"{name} entries must be unique")
    return tuple(value)


def _tuple_name_map(
    mapping: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    return {key: tuple(values) for key, values in mapping.items()}


def _lists_from_tuples(
    mapping: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    return {key: list(values) for key, values in mapping.items()}


def _structured_tracing_enabled() -> bool:
    return any(
        getattr(handler, "root_dir", None) is not None for handler in trace_log.handlers
    )


def _derive_status(document: dict[str, Any]) -> None:
    projections = _require_mapping(document["upstreamProjections"], "projections")
    document["status"] = (
        "complete"
        if not document["diagnostics"]
        and projections
        and all(item["upstreamJoin"] == "ok" for item in projections.values())
        else "partial"
    )


def _diagnostics_from_projections(
    projections: Mapping[str, object],
) -> dict[str, int]:
    collection_failure_count = 0
    upstream_projection_failure_count = 0
    for projection_value in projections.values():
        projection = _require_mapping(projection_value, "upstream projection")
        collection_failure_count += len(projection["uncollectedKernels"])
        upstream_projection_failure_count += int(projection["upstreamProjectionFailed"])

    diagnostics: dict[str, int] = {}
    if collection_failure_count:
        diagnostics["collection-failure"] = collection_failure_count
    if upstream_projection_failure_count:
        diagnostics["upstream-projection-failure"] = upstream_projection_failure_count
    return diagnostics


def _validate_event_key(value: object) -> None:
    event_key = _require_mapping(value, "eventKey")
    _require_exact_keys(
        event_key,
        {"version", "algorithm", "width", "stepAttribution"},
        "eventKey",
    )
    _require_int(event_key["version"], "event key version", minimum=1)
    _require_int(event_key["width"], "event key width", minimum=1)
    expected = {
        "version": KERNEL_PROVENANCE_KEY_VERSION,
        "algorithm": "sha256-prefix-80-base32-lower",
        "width": KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
        "stepAttribution": "bundle",
    }
    if event_key != expected:
        raise ProvenanceArtifactError("event key contract mismatch")


def _validate_diagnostics(value: object) -> None:
    diagnostics = _require_mapping(value, "diagnostics")
    _require_sorted_keys(diagnostics, "diagnostics")
    if not set(diagnostics) <= _DIAGNOSTIC_CODES:
        raise ProvenanceArtifactError("invalid writer diagnostic code")
    for code, count in diagnostics.items():
        _require_int(count, f"diagnostic {code}", minimum=1)


def _validate_handle(
    handle_id: str,
    value: object,
    handles: Mapping[str, object],
) -> None:
    handle = _require_mapping(value, "debug handle")
    _require_exact_keys(
        handle,
        {"id", "source", "aten_op", "ir_chain", "fused_from", "transform_history"},
        "debug handle",
    )
    if handle["id"] != handle_id:
        raise ProvenanceArtifactError("debug handle key and ID disagree")
    source = handle["source"]
    if source is not None:
        source_map = _require_mapping(source, "source location")
        _require_exact_keys(
            source_map,
            {"file", "start_line", "start_col", "end_line", "end_col"},
            "source location",
        )
        _require_nonempty_string(source_map["file"], "source file")
        _require_int(source_map["start_line"], "source start line", minimum=1)
        _require_int(source_map["start_col"], "source start column", minimum=0)
        for field, minimum in (("end_line", 1), ("end_col", 0)):
            if source_map[field] is not None:
                _require_int(source_map[field], field, minimum=minimum)
    if handle["aten_op"] is not None:
        _require_nonempty_string(handle["aten_op"], "aten_op")
    _validate_string_list(handle["ir_chain"], "ir_chain", nonempty_items=True)
    fused_from = _validate_string_list(handle["fused_from"], "fused_from", unique=True)
    if any(not _DECIMAL_ID_PATTERN.fullmatch(item) for item in fused_from):
        raise ProvenanceArtifactError("invalid fused handle ID")
    if any(item not in handles for item in fused_from):
        raise ProvenanceArtifactError("dangling fused handle reference")
    transforms = _require_list(handle["transform_history"], "transform history")
    for transform_value in transforms:
        transform = _require_mapping(transform_value, "provenance transform")
        _require_exact_keys(transform, {"kind", "pass_name", "reason"}, "transform")
        if transform["kind"] not in _TRANSFORM_KINDS:
            raise ProvenanceArtifactError("invalid transform kind")
        _require_nonempty_string(transform["pass_name"], "transform pass name")
        if transform["reason"] is not None and not isinstance(transform["reason"], str):
            raise ProvenanceArtifactError("invalid transform reason")


def _validate_identity(
    identity_key: str,
    value: object,
    handles: Mapping[str, object],
) -> None:
    identity = _require_mapping(value, "kernel identity")
    _require_exact_keys(
        identity,
        {"directHandleIds", "specHandleBindings", "atenOps", "eventNameBase"},
        "kernel identity",
    )
    direct_handle_ids = _validate_string_list(
        identity["directHandleIds"],
        "direct handle IDs",
        unique=True,
    )
    if any(not _DECIMAL_ID_PATTERN.fullmatch(item) for item in direct_handle_ids):
        raise ProvenanceArtifactError("invalid direct handle ID")
    if any(item not in handles for item in direct_handle_ids):
        raise ProvenanceArtifactError("dangling direct handle reference")

    binding_ids: list[str] = []
    bindings = _require_list(identity["specHandleBindings"], "spec bindings")
    for binding_value in bindings:
        binding = _require_mapping(binding_value, "spec handle binding")
        _require_exact_keys(binding, {"specPath", "handleId"}, "spec binding")
        spec_path = _require_list(binding["specPath"], "spec path")
        if not spec_path:
            raise ProvenanceArtifactError("spec path must not be empty")
        for index in spec_path:
            _require_int(index, "spec path index", minimum=0)
        handle_id = binding["handleId"]
        if (
            not isinstance(handle_id, str)
            or not _DECIMAL_ID_PATTERN.fullmatch(handle_id)
            or handle_id not in handles
        ):
            raise ProvenanceArtifactError("invalid spec binding handle reference")
        binding_ids.append(handle_id)
    if direct_handle_ids != list(dict.fromkeys(binding_ids)):
        raise ProvenanceArtifactError("direct handle order disagrees with bindings")

    aten_ops = _validate_string_list(
        identity["atenOps"],
        "ATen operations",
        unique=True,
        nonempty_items=True,
    )
    if aten_ops != sorted(aten_ops):
        raise ProvenanceArtifactError("ATen operations must be sorted")
    if aten_ops != _recursive_aten_ops(direct_handle_ids, handles):
        raise ProvenanceArtifactError("ATen operations disagree with handles")

    event_name_base = identity["eventNameBase"]
    if not isinstance(event_name_base, str) or not _EVENT_NAME_PATTERN.fullmatch(
        event_name_base
    ):
        raise ProvenanceArtifactError("invalid profiler event name base")
    descriptor = KernelProvenanceDescriptor(
        key=identity_key,
        debug_handle_ids=tuple(direct_handle_ids),
        aten_ops=tuple(aten_ops),
    )
    if event_name_base != format_kernel_provenance_event_name(descriptor):
        raise ProvenanceArtifactError("profiler event name disagrees with identity")


def _validate_occurrence(
    occurrence_id: str,
    value: object,
    identities: Mapping[str, object],
    projections: Mapping[str, object],
) -> dict[str, Any]:
    occurrence = _require_mapping(value, "kernel occurrence")
    required = {"identityKey", "compileId", "compilerKernelName", "registrations"}
    if set(occurrence) not in (required, required | {"selector"}):
        raise ProvenanceArtifactError("kernel occurrence has invalid fields")

    identity_key = occurrence["identityKey"]
    if (
        not isinstance(identity_key, str)
        or not _EVENT_KEY_PATTERN.fullmatch(identity_key)
        or identity_key not in identities
    ):
        raise ProvenanceArtifactError("invalid occurrence identity reference")
    compile_id = occurrence["compileId"]
    if (
        not isinstance(compile_id, str)
        or not _SHA256_PATTERN.fullmatch(compile_id)
        or compile_id not in projections
    ):
        raise ProvenanceArtifactError("invalid occurrence compile reference")
    kernel_name = _require_nonempty_string(
        occurrence["compilerKernelName"], "compiler kernel name"
    )
    if "selector" in occurrence:
        _require_nonempty_string(occurrence["selector"], "occurrence selector")

    registrations = _require_list(occurrence["registrations"], "registrations")
    ordinals: list[int] = []
    aliases: set[str] = set()
    for registration_value in registrations:
        registration = _require_mapping(registration_value, "registration")
        _require_exact_keys(registration, {"ordinal", "alias"}, "registration")
        ordinal = _require_int(
            registration["ordinal"], "registration ordinal", minimum=1
        )
        alias = registration["alias"]
        if (
            not isinstance(alias, str)
            or not _ALIAS_PATTERN.fullmatch(alias)
            or alias != f"{kernel_name}:{ordinal}"
        ):
            raise ProvenanceArtifactError("registration alias disagrees with ordinal")
        if alias in aliases:
            raise ProvenanceArtifactError("duplicate registration alias")
        ordinals.append(ordinal)
        aliases.add(alias)
    if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
        raise ProvenanceArtifactError("registration ordinals must be sorted and unique")

    expected_id = _canonical_digest(
        {
            "domain": _OCCURRENCE_ID_DOMAIN,
            "compileId": compile_id,
            "compilerKernelName": kernel_name,
            "identityKey": identity_key,
        }
    )
    if occurrence_id != expected_id:
        raise ProvenanceArtifactError("kernel occurrence ID derivation mismatch")
    return occurrence


def _validate_projection(
    compile_id: str,
    value: object,
    identities: Mapping[str, object],
    occurrences: Sequence[Mapping[str, object]],
) -> None:
    projection = _require_mapping(value, "upstream projection")
    _require_exact_keys(projection, _PROJECTION_KEYS, "upstream projection")
    if projection["source"] != _UPSTREAM_SOURCE:
        raise ProvenanceArtifactError("invalid upstream projection source")
    if projection["version"] != _UPSTREAM_VERSION:
        raise ProvenanceArtifactError("invalid upstream projection version")

    producer = _require_mapping(projection["producer"], "producer")
    _require_exact_keys(producer, {"torchSpyreVersion", "torchVersion"}, "producer")
    _require_nonempty_string(producer["torchSpyreVersion"], "torch-spyre version")
    _require_nonempty_string(producer["torchVersion"], "PyTorch version")

    settings = _require_mapping(projection["settings"], "settings")
    _require_exact_keys(
        settings,
        {"provenanceTrackingLevel", "structuredTracing"},
        "settings",
    )
    _require_int(
        settings["provenanceTrackingLevel"],
        "provenance tracking level",
        minimum=0,
    )
    if not isinstance(settings["structuredTracing"], bool):
        raise ProvenanceArtifactError("structured tracing setting must be boolean")

    kernels_value = _require_list(projection["kernels"], "compile kernels")
    if not kernels_value:
        raise ProvenanceArtifactError("compile kernels must not be empty")
    kernels: list[tuple[str, str]] = []
    for kernel_value in kernels_value:
        kernel = _require_mapping(kernel_value, "compile kernel")
        _require_exact_keys(
            kernel, {"compilerKernelName", "identityKey"}, "compile kernel"
        )
        kernel_name = _require_nonempty_string(
            kernel["compilerKernelName"], "compiler kernel name"
        )
        identity_key = kernel["identityKey"]
        if (
            not isinstance(identity_key, str)
            or not _EVENT_KEY_PATTERN.fullmatch(identity_key)
            or identity_key not in identities
        ):
            raise ProvenanceArtifactError("invalid compile kernel identity")
        kernels.append((kernel_name, identity_key))
    if len(kernels) != len(set(kernels)):
        raise ProvenanceArtifactError("duplicate compile kernel")

    uncollected_kernels = _validate_string_list(
        projection["uncollectedKernels"],
        "uncollected kernel names",
        unique=True,
        nonempty_items=True,
    )
    if uncollected_kernels != sorted(uncollected_kernels):
        raise ProvenanceArtifactError("uncollected kernel names must be sorted")
    if set(uncollected_kernels) & {kernel_name for kernel_name, _ in kernels}:
        raise ProvenanceArtifactError("collected and uncollected kernels overlap")
    if not isinstance(projection["upstreamProjectionFailed"], bool):
        raise ProvenanceArtifactError(
            "upstream projection failure marker must be boolean"
        )

    expected_compile_id = _canonical_digest(
        {
            "domain": _COMPILE_ID_DOMAIN,
            "kernels": [list(kernel) for kernel in kernels],
        }
    )
    if compile_id != expected_compile_id:
        raise ProvenanceArtifactError("compile ID derivation mismatch")

    occurrence_kernels = {
        (item["compilerKernelName"], item["identityKey"]) for item in occurrences
    }
    if occurrence_kernels != set(kernels):
        raise ProvenanceArtifactError("projection kernels disagree with occurrences")

    registration_aliases: set[str] = set()
    registration_ordinals: set[int] = set()
    for occurrence in occurrences:
        for registration in occurrence["registrations"]:
            ordinal = registration["ordinal"]
            if ordinal in registration_ordinals:
                raise ProvenanceArtifactError(
                    "registration ordinal repeated across one compile"
                )
            registration_ordinals.add(ordinal)
            registration_aliases.add(registration["alias"])

    join = projection["upstreamJoin"]
    if join not in _JOIN_RANK:
        raise ProvenanceArtifactError("invalid upstream join status")
    pre_to_post = _validate_name_map(projection["preToPost"], "preToPost")
    post_to_pre = _validate_name_map(projection["postToPre"], "postToPre")
    cpp_to_post = _validate_name_map(projection["cppCodeToPost"], "cppCodeToPost")
    post_to_cpp = _validate_name_map(projection["postToCppCode"], "postToCppCode")
    if not set(cpp_to_post) <= registration_aliases:
        raise ProvenanceArtifactError("upstream aliases lack exact registrations")
    if join == "ok" and set(cpp_to_post) != registration_aliases:
        raise ProvenanceArtifactError("complete projection lacks registered aliases")
    if post_to_pre != _invert_name_map(pre_to_post):
        raise ProvenanceArtifactError("postToPre is not the exact inverse")
    if post_to_cpp != _invert_name_map(cpp_to_post):
        raise ProvenanceArtifactError("postToCppCode is not the exact inverse")

    stacks = _require_mapping(projection["kernelStackTraces"], "stack contexts")
    _require_sorted_keys(stacks, "kernelStackTraces")
    for alias, context_value in stacks.items():
        if not isinstance(alias, str):
            raise ProvenanceArtifactError("stack context alias must be a string")
        context = _require_mapping(context_value, "kernel stack context")
        _require_exact_keys(
            context,
            {"stackTraces", "postGradNodes", "preGradNodes"},
            "kernel stack context",
        )
        for field in ("stackTraces", "postGradNodes", "preGradNodes"):
            _validate_string_list(context[field], field, unique=True)
    if join == "ok" and set(stacks) != registration_aliases:
        raise ProvenanceArtifactError("complete projection lacks stack contexts")


def _validate_name_map(value: object, name: str) -> dict[str, list[str]]:
    mapping = _require_mapping(value, name)
    _require_sorted_keys(mapping, name)
    result: dict[str, list[str]] = {}
    for key, values in mapping.items():
        if not isinstance(key, str):
            raise ProvenanceArtifactError(f"{name} key must be a string")
        result[key] = _validate_string_list(values, name, unique=True)
    return result


def _recursive_aten_ops(
    direct_handle_ids: Sequence[str], handles: Mapping[str, object]
) -> list[str]:
    names: set[str] = set()
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(handle_id: str) -> None:
        if handle_id in visiting:
            raise ProvenanceArtifactError("cyclic fused handle references")
        if handle_id in visited:
            return
        visiting.add(handle_id)
        handle = _require_mapping(handles[handle_id], "debug handle")
        aten_op = handle["aten_op"]
        if aten_op is not None:
            assert isinstance(aten_op, str)
            names.add(aten_op)
        for constituent_id in handle["fused_from"]:
            visit(constituent_id)
        visiting.remove(handle_id)
        visited.add(handle_id)

    for direct_handle_id in direct_handle_ids:
        visit(direct_handle_id)
    return sorted(names)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _canonicalize_document(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize_document(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_document(item) for item in value]
    return value


def _without_generation(document: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(document))
    result.pop("mergeGeneration", None)
    return result


def _serialize_document(document: Mapping[str, object]) -> str:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _read_existing_document(path: Path) -> dict[str, Any] | None:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ProvenanceArtifactError(f"failed to read {path.name}") from None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        raise ProvenanceArtifactError(f"invalid JSON in {path.name}") from None
    if not isinstance(value, dict):
        raise ProvenanceArtifactError(f"invalid document in {path.name}")
    return value


@contextlib.contextmanager
def _interprocess_publication_lock(path: Path) -> Iterator[None]:
    """Serialize cooperating publishers without creating a stale lock file.

    This advisory lock does not protect against non-cooperating writers.
    """
    if _fcntl is None:
        raise ProvenanceArtifactError(
            f"interprocess locking unavailable for {path.name}"
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            _fcntl.flock(directory_fd, _fcntl.LOCK_EX)
        except OSError:
            os.close(directory_fd)
            raise
    except OSError:
        raise ProvenanceArtifactError(f"failed to lock {path.name}") from None

    try:
        yield
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _atomic_write(path: Path, payload: str) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError:
        raise ProvenanceArtifactError(f"failed to write {path.name}") from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceArtifactError(f"{name} must be an object")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProvenanceArtifactError(f"{name} must be an array")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ProvenanceArtifactError(f"{name} has invalid fields")


def _require_sorted_keys(value: Mapping[str, object], name: str) -> None:
    if list(value) != sorted(value):
        raise ProvenanceArtifactError(f"{name} keys must be sorted")


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProvenanceArtifactError(f"{name} must be an integer >= {minimum}")
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProvenanceArtifactError(f"{name} must be a non-empty string")
    return value


def _validate_string_list(
    value: object,
    name: str,
    *,
    unique: bool = False,
    nonempty_items: bool = False,
) -> list[str]:
    items = _require_list(value, name)
    if not all(
        isinstance(item, str) and (not nonempty_items or bool(item)) for item in items
    ):
        raise ProvenanceArtifactError(f"{name} must contain valid strings")
    result = list(items)
    if unique and len(result) != len(set(result)):
        raise ProvenanceArtifactError(f"{name} must not contain duplicates")
    return result
