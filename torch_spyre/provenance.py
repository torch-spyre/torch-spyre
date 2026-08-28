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

"""Offline resolution of Spyre profiler events against a saved sidecar.

The supported module invocation disables package backend autoload::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torch_spyre.provenance \
        '<event-name>' <artifact-path>

Without that setting, Python executes ``torch_spyre.__init__`` before this
module and can re-enter the partially initialized package while PyTorch loads
PrivateUse1 backends.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import regex

from torch_spyre.provenance_codec import (
    KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
    KERNEL_PROVENANCE_KEY_VERSION,
    ParsedKernelProvenanceEvent,
    parse_kernel_provenance_event_name,
)


_RESOLUTION_VERSION = 1
_SCHEMA_VERSION = 1
_COMPILE_ID_DOMAIN = "torch-spyre-compile-v1"
_OCCURRENCE_ID_DOMAIN = "torch-spyre-occurrence-v1"
_OCCURRENCE_SUMMARY_FIELDS = (
    "identityKey",
    "compileId",
    "compilerKernelName",
    "registrations",
    "selector",
    "upstreamJoin",
)


class _ReaderError(Exception):
    """Expected reader rejection with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _SemanticError(ValueError):
    """Artifact content violates a cross-record or derivation invariant."""


def resolve_provenance_event(
    event_name: str,
    artifact_path: str | Path,
    *,
    occurrence_selector: str | None = None,
) -> dict[str, Any]:
    """Resolve one saved event name against one saved provenance sidecar."""
    path = Path(artifact_path)
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return _failure_result(
            event_name,
            "artifact-read-failure",
            f"failed to read {path.name}",
        )
    try:
        document = json.loads(payload)
    except (RecursionError, ValueError):
        return _failure_result(
            event_name,
            "schema-validation-failure",
            f"invalid JSON in {path.name}",
        )
    return resolve_provenance_document(
        event_name,
        document,
        occurrence_selector=occurrence_selector,
    )


def resolve_provenance_document(
    event_name: str,
    document: object,
    *,
    occurrence_selector: str | None = None,
) -> dict[str, Any]:
    """Resolve one event against an already decoded sidecar document."""
    parsed = parse_kernel_provenance_event_name(event_name)
    if parsed is None:
        return _failure_result(
            event_name,
            "invalid-event-name",
            "event name does not contain a supported Spyre provenance key",
        )
    if occurrence_selector is not None and not occurrence_selector:
        return _failure_result(
            event_name,
            "invalid-selector",
            "occurrence selector must not be empty",
            parsed=parsed,
        )

    try:
        validated = _validate_document(document)
    except _ReaderError as error:
        return _failure_result(
            event_name,
            error.code,
            str(error),
            parsed=parsed,
        )

    identities = validated["kernelIdentities"]
    identity = identities.get(parsed.key)
    if identity is None:
        return _failure_result(
            event_name,
            "missing-key",
            "event key is absent from the provenance artifact",
            parsed=parsed,
        )
    if parsed.base_name != identity["eventNameBase"]:
        return _failure_result(
            event_name,
            "collision",
            "event key resolves to a different persisted event-name base",
            parsed=parsed,
            details={
                "eventNameBase": parsed.base_name,
                "persistedEventNameBase": identity["eventNameBase"],
            },
        )

    candidates = [
        (occurrence_id, occurrence)
        for occurrence_id, occurrence in validated["kernelOccurrences"].items()
        if occurrence["identityKey"] == parsed.key
    ]
    if occurrence_selector is not None:
        candidates = [
            item
            for item in candidates
            if item[1].get("selector") == occurrence_selector
        ]
    if not candidates:
        details = {"identityKey": parsed.key}
        if occurrence_selector is not None:
            details["occurrenceSelector"] = occurrence_selector
        return _failure_result(
            event_name,
            "missing-key",
            "no matching kernel occurrence exists",
            parsed=parsed,
            details=details,
        )

    handles = validated["handles"]
    direct_handle_ids = identity["directHandleIds"]
    reachable_ids = _reachable_handle_ids(direct_handle_ids, handles)
    direct_set = set(direct_handle_ids)
    fused_constituent_ids = [
        handle_id for handle_id in reachable_ids if handle_id not in direct_set
    ]

    projections = validated["upstreamProjections"]
    resolved_occurrences = []
    compile_ids: set[str] = set()
    for occurrence_id, occurrence in candidates:
        compile_id = occurrence["compileId"]
        compile_ids.add(compile_id)
        resolved_occurrences.append(
            {
                "occurrenceId": occurrence_id,
                **copy.deepcopy(occurrence),
                "upstreamJoin": projections[compile_id]["upstreamJoin"],
            }
        )
    summary = _summarize_occurrences(resolved_occurrences)

    diagnostics: list[dict[str, Any]] = []
    if validated["status"] != "complete" or validated["diagnostics"]:
        diagnostics.append(
            _diagnostic(
                "incomplete-artifact",
                "warning",
                "artifact records an incomplete collection or upstream join",
                {
                    "artifactStatus": validated["status"],
                    "writerDiagnostics": copy.deepcopy(validated["diagnostics"]),
                },
            )
        )
    joins: dict[str, list[str]] = {}
    for compile_id in sorted(compile_ids):
        join = projections[compile_id]["upstreamJoin"]
        joins.setdefault(join, []).append(compile_id)
    for join, join_compile_ids in sorted(joins.items()):
        diagnostics.append(
            _diagnostic(
                "upstream-join-status",
                "info" if join == "ok" else "warning",
                f"resolved occurrence upstream join is {join}",
                {"compileIds": join_compile_ids, "upstreamJoin": join},
            )
        )
    if summary["ambiguous"]:
        diagnostics.append(
            _diagnostic(
                "ambiguity",
                "warning",
                "candidate occurrences disagree on one or more context fields",
                {
                    "fields": sorted(
                        field
                        for field, value in summary["fields"].items()
                        if value["ambiguous"]
                    )
                },
            )
        )

    return {
        "resolutionVersion": _RESOLUTION_VERSION,
        "status": _status_from_diagnostics(diagnostics),
        "event": _event_dict(parsed, occurrence_selector),
        "identityKey": parsed.key,
        "identity": copy.deepcopy(identity),
        "directHandleIds": list(direct_handle_ids),
        "fusedConstituentIds": fused_constituent_ids,
        "handles": {
            handle_id: copy.deepcopy(handles[handle_id])
            for handle_id in sorted(reachable_ids)
        },
        "occurrences": resolved_occurrences,
        "occurrenceSummary": summary,
        "upstreamProjections": {
            compile_id: copy.deepcopy(projections[compile_id])
            for compile_id in sorted(compile_ids)
        },
        "diagnostics": diagnostics,
    }


def _failure_result(
    event_name: str,
    code: str,
    message: str,
    *,
    parsed: ParsedKernelProvenanceEvent | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    event: dict[str, object] = {"name": event_name}
    if parsed is not None:
        event = _event_dict(parsed, None)
    return {
        "resolutionVersion": _RESOLUTION_VERSION,
        "status": "error",
        "event": event,
        "identityKey": parsed.key if parsed is not None else None,
        "identity": None,
        "directHandleIds": [],
        "fusedConstituentIds": [],
        "handles": {},
        "occurrences": [],
        "occurrenceSummary": {"ambiguous": False, "fields": {}},
        "upstreamProjections": {},
        "diagnostics": [_diagnostic(code, "error", message, details)],
    }


def _event_dict(
    parsed: ParsedKernelProvenanceEvent,
    occurrence_selector: str | None,
) -> dict[str, object]:
    return {
        "name": parsed.name,
        "baseName": parsed.base_name,
        "identityKey": parsed.key,
        "commandStep": parsed.step,
        "commandStepSuffix": parsed.step_suffix,
        "stepAttribution": "bundle",
        "occurrenceSelector": occurrence_selector,
    }


def _diagnostic(
    code: str,
    severity: str,
    message: str,
    details: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if details:
        result["details"] = copy.deepcopy(dict(details))
    return result


def _status_from_diagnostics(diagnostics: Sequence[Mapping[str, object]]) -> str:
    severities = {diagnostic["severity"] for diagnostic in diagnostics}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "partial"
    return "complete"


def _reachable_handle_ids(
    direct_handle_ids: Sequence[str], handles: Mapping[str, object]
) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()
    pending = list(reversed(direct_handle_ids))
    while pending:
        handle_id = pending.pop()
        if handle_id in visited:
            continue
        visited.add(handle_id)
        ordered.append(handle_id)
        handle = handles[handle_id]
        assert isinstance(handle, Mapping)
        constituents = cast(Sequence[str], handle["fused_from"])
        pending.extend(reversed(constituents))
    return ordered


def _summarize_occurrences(
    occurrences: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for field in _OCCURRENCE_SUMMARY_FIELDS:
        values = [occurrence.get(field) for occurrence in occurrences]
        distinct = _distinct_values(values)
        if len(distinct) == 1:
            fields[field] = {
                "ambiguous": False,
                "value": copy.deepcopy(distinct[0]),
            }
        else:
            fields[field] = {
                "ambiguous": True,
                "values": copy.deepcopy(distinct),
            }
    return {
        "ambiguous": any(value["ambiguous"] for value in fields.values()),
        "fields": fields,
    }


def _distinct_values(values: Sequence[object]) -> list[object]:
    keyed = {
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ): value
        for value in values
    }
    return [keyed[key] for key in sorted(keyed)]


def _validate_document(document: object) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise _ReaderError("schema-validation-failure", "artifact must be an object")
    if document.get("schemaVersion") != _SCHEMA_VERSION:
        raise _ReaderError(
            "unsupported-schema-version",
            "artifact schema version is not supported",
        )
    try:
        schema = _schema()
        _validate_schema(document, schema, schema, "$")
    except (OSError, UnicodeError, RecursionError, ValueError) as error:
        raise _ReaderError("schema-validation-failure", str(error)) from None
    try:
        _validate_semantics(document)
    except _SemanticError as error:
        raise _ReaderError("semantic-validation-failure", str(error)) from None
    return document


@functools.cache
def _schema() -> dict[str, Any]:
    path = (
        Path(__file__).parent
        / "_inductor"
        / "schemas"
        / "spyre_provenance_v1.schema.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("packaged provenance schema must be an object")
    return value


# Keep this dependency-free JSON Schema subset in parity with the packaged v1
# schema. The fixture suite also runs the real jsonschema validator as an
# independent oracle; schema changes must update both implementations and tests.
def _validate_schema(
    value: object,
    schema: Mapping[str, object],
    root: Mapping[str, object],
    path: str,
) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ValueError(f"{path}: unsupported schema reference")
        target: object = root
        for token in reference[2:].split("/"):
            if not isinstance(target, Mapping):
                raise ValueError(f"{path}: invalid schema reference")
            decoded_token = token.replace("~1", "/").replace("~0", "~")
            if decoded_token not in target:
                raise ValueError(f"{path}: invalid schema reference")
            target = target[decoded_token]
        if not isinstance(target, Mapping):
            raise ValueError(f"{path}: invalid schema reference")
        _validate_schema(value, target, root, path)
        return
    if "oneOf" in schema:
        matches = 0
        choices = cast(Sequence[Mapping[str, object]], schema["oneOf"])
        for choice in choices:
            try:
                _validate_schema(value, choice, root, path)
            except ValueError:
                continue
            matches += 1
        if matches != 1:
            raise ValueError(f"{path}: value must match exactly one schema")
        return
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value disagrees with schema constant")
    allowed = cast(Sequence[object] | None, schema.get("enum"))
    if allowed is not None and value not in allowed:
        raise ValueError(f"{path}: value is outside the allowed enum")
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        raise ValueError(f"{path}: value has the wrong type")

    if isinstance(value, dict):
        properties = cast(
            Mapping[str, Mapping[str, object]], schema.get("properties", {})
        )
        required = cast(Sequence[str], schema.get("required", []))
        if not all(key in value for key in required):
            raise ValueError(f"{path}: required field is missing")
        minimum_properties = cast(int | None, schema.get("minProperties"))
        if minimum_properties is not None and len(value) < minimum_properties:
            raise ValueError(f"{path}: object has too few fields")
        property_names = cast(Mapping[str, object] | None, schema.get("propertyNames"))
        if property_names is not None:
            for key in value:
                _validate_schema(key, property_names, root, f"{path}.<key>")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                item_schema = properties[key]
            elif additional is False:
                raise ValueError(f"{path}: unknown field {key}")
            elif isinstance(additional, Mapping):
                item_schema = additional
            else:
                continue
            _validate_schema(item, item_schema, root, f"{path}.{key}")
    elif isinstance(value, list):
        minimum_items = cast(int | None, schema.get("minItems"))
        if minimum_items is not None and len(value) < minimum_items:
            raise ValueError(f"{path}: array has too few items")
        if schema.get("uniqueItems"):
            canonical = [_canonical_json(item) for item in value]
            if len(canonical) != len(set(canonical)):
                raise ValueError(f"{path}: array items must be unique")
        list_item_schema = cast(Mapping[str, object] | None, schema.get("items"))
        if list_item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema(item, list_item_schema, root, f"{path}[{index}]")
    elif isinstance(value, str):
        minimum_length = cast(int | None, schema.get("minLength"))
        if minimum_length is not None and len(value) < minimum_length:
            raise ValueError(f"{path}: string is too short")
        pattern = cast(str | None, schema.get("pattern"))
        if pattern is not None and regex.fullmatch(pattern, value) is None:
            raise ValueError(f"{path}: string does not match its pattern")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = cast(int | float | None, schema.get("minimum"))
        if minimum is not None and value < minimum:
            raise ValueError(f"{path}: number is below its minimum")


def _matches_type(value: object, expected: object) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported JSON Schema type {expected}")


def _validate_semantics(document: Mapping[str, Any]) -> None:
    expected_event_key = {
        "version": KERNEL_PROVENANCE_KEY_VERSION,
        "algorithm": "sha256-prefix-80-base32-lower",
        "width": KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
        "stepAttribution": "bundle",
    }
    if document["eventKey"] != expected_event_key:
        raise _SemanticError("event key contract mismatch")

    handles = document["handles"]
    identities = document["kernelIdentities"]
    occurrences = document["kernelOccurrences"]
    projections = document["upstreamProjections"]
    _require_sorted(handles, "handles")
    _require_sorted(identities, "kernelIdentities")
    _require_sorted(occurrences, "kernelOccurrences")
    _require_sorted(projections, "upstreamProjections")
    _require_sorted(document["diagnostics"], "diagnostics")

    for handle_id, handle in handles.items():
        if handle["id"] != handle_id:
            raise _SemanticError("debug handle key and ID disagree")
        if any(item not in handles for item in handle["fused_from"]):
            raise _SemanticError("dangling fused handle reference")
    _validate_handle_cycles(handles)

    for identity_key, identity in identities.items():
        direct_ids = identity["directHandleIds"]
        if any(handle_id not in handles for handle_id in direct_ids):
            raise _SemanticError("dangling direct handle reference")
        binding_ids = [
            binding["handleId"] for binding in identity["specHandleBindings"]
        ]
        if any(handle_id not in handles for handle_id in binding_ids):
            raise _SemanticError("dangling spec binding handle reference")
        if direct_ids != list(dict.fromkeys(binding_ids)):
            raise _SemanticError("direct handle order disagrees with bindings")
        if identity["atenOps"] != _recursive_aten_ops(direct_ids, handles):
            raise _SemanticError("ATen operations disagree with handles")
        parsed = parse_kernel_provenance_event_name(identity["eventNameBase"])
        if parsed is None or parsed.step is not None or parsed.key != identity_key:
            raise _SemanticError("profiler event name disagrees with identity")

    occurrences_by_compile: dict[str, list[Mapping[str, Any]]] = {}
    for occurrence_id, occurrence in occurrences.items():
        identity_key = occurrence["identityKey"]
        compile_id = occurrence["compileId"]
        if identity_key not in identities:
            raise _SemanticError("invalid occurrence identity reference")
        if compile_id not in projections:
            raise _SemanticError("invalid occurrence compile reference")
        kernel_name = occurrence["compilerKernelName"]
        ordinals = []
        for registration in occurrence["registrations"]:
            ordinal = registration["ordinal"]
            if registration["alias"] != f"{kernel_name}:{ordinal}":
                raise _SemanticError("registration alias disagrees with ordinal")
            ordinals.append(ordinal)
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise _SemanticError("registration ordinals must be sorted and unique")
        expected_occurrence_id = _canonical_digest(
            {
                "domain": _OCCURRENCE_ID_DOMAIN,
                "compileId": compile_id,
                "compilerKernelName": kernel_name,
                "identityKey": identity_key,
            }
        )
        if occurrence_id != expected_occurrence_id:
            raise _SemanticError("kernel occurrence ID derivation mismatch")
        occurrences_by_compile.setdefault(compile_id, []).append(occurrence)

    for compile_id, projection in projections.items():
        kernels = [
            (kernel["compilerKernelName"], kernel["identityKey"])
            for kernel in projection["kernels"]
        ]
        if len(kernels) != len(set(kernels)):
            raise _SemanticError("duplicate compile kernel")
        uncollected_kernels = projection["uncollectedKernels"]
        if uncollected_kernels != sorted(uncollected_kernels):
            raise _SemanticError("uncollected kernel names must be sorted")
        collected_kernel_names = {kernel_name for kernel_name, _ in kernels}
        if collected_kernel_names & set(uncollected_kernels):
            raise _SemanticError("collected and uncollected kernels overlap")

        if any(identity_key not in identities for _, identity_key in kernels):
            raise _SemanticError("invalid compile kernel identity")
        expected_compile_id = _canonical_digest(
            {
                "domain": _COMPILE_ID_DOMAIN,
                "kernels": [list(kernel) for kernel in kernels],
            }
        )
        if compile_id != expected_compile_id:
            raise _SemanticError("compile ID derivation mismatch")
        occurrence_kernels = {
            (item["compilerKernelName"], item["identityKey"])
            for item in occurrences_by_compile.get(compile_id, [])
        }
        if occurrence_kernels != set(kernels):
            raise _SemanticError("projection kernels disagree with occurrences")
        _validate_projection_relations(projection, occurrences_by_compile[compile_id])
    if set(occurrences_by_compile) != set(projections):
        raise _SemanticError("occurrence and projection compile IDs disagree")

    expected_diagnostics: dict[str, int] = {}
    collection_failures = sum(
        len(projection["uncollectedKernels"]) for projection in projections.values()
    )
    upstream_failures = sum(
        int(projection["upstreamProjectionFailed"])
        for projection in projections.values()
    )
    if collection_failures:
        expected_diagnostics["collection-failure"] = collection_failures
    if upstream_failures:
        expected_diagnostics["upstream-projection-failure"] = upstream_failures
    if document["diagnostics"] != expected_diagnostics:
        raise _SemanticError("writer diagnostics disagree with projections")
    expected_status = (
        "complete"
        if not expected_diagnostics
        and all(
            projection["upstreamJoin"] == "ok" for projection in projections.values()
        )
        else "partial"
    )
    if document["status"] != expected_status:
        raise _SemanticError("document status disagrees with its content")


def _validate_handle_cycles(handles: Mapping[str, Mapping[str, Any]]) -> None:
    visited: set[str] = set()
    visiting: set[str] = set()

    for handle_id in handles:
        pending = [(handle_id, False)]
        while pending:
            current_id, exiting = pending.pop()
            if exiting:
                visiting.remove(current_id)
                visited.add(current_id)
                continue
            if current_id in visiting:
                raise _SemanticError("cyclic fused handle references")
            if current_id in visited:
                continue
            visiting.add(current_id)
            pending.append((current_id, True))
            pending.extend(
                (constituent_id, False)
                for constituent_id in reversed(handles[current_id]["fused_from"])
            )


def _recursive_aten_ops(
    direct_ids: Sequence[str], handles: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    names: set[str] = set()
    visited: set[str] = set()
    pending = list(reversed(direct_ids))
    while pending:
        handle_id = pending.pop()
        if handle_id in visited:
            continue
        visited.add(handle_id)
        handle = handles[handle_id]
        if handle["aten_op"] is not None:
            names.add(handle["aten_op"])
        pending.extend(reversed(handle["fused_from"]))
    return sorted(names)


def _validate_projection_relations(
    projection: Mapping[str, Any],
    occurrences: Sequence[Mapping[str, Any]],
) -> None:
    for name in (
        "preToPost",
        "postToPre",
        "cppCodeToPost",
        "postToCppCode",
        "kernelStackTraces",
    ):
        _require_sorted(projection[name], name)
    registration_aliases: set[str] = set()
    registration_ordinals: set[int] = set()
    for occurrence in occurrences:
        for registration in occurrence["registrations"]:
            ordinal = registration["ordinal"]
            if ordinal in registration_ordinals:
                raise _SemanticError("registration ordinal repeated across one compile")
            registration_ordinals.add(ordinal)
            registration_aliases.add(registration["alias"])
    cpp_to_post = projection["cppCodeToPost"]
    if not set(cpp_to_post) <= registration_aliases:
        raise _SemanticError("upstream aliases lack exact registrations")
    if projection["upstreamJoin"] == "ok" and set(cpp_to_post) != registration_aliases:
        raise _SemanticError("complete projection lacks registered aliases")
    if projection["postToPre"] != _invert_name_map(projection["preToPost"]):
        raise _SemanticError("postToPre is not the exact inverse")
    if projection["postToCppCode"] != _invert_name_map(cpp_to_post):
        raise _SemanticError("postToCppCode is not the exact inverse")
    if (
        projection["upstreamJoin"] == "ok"
        and set(projection["kernelStackTraces"]) != registration_aliases
    ):
        raise _SemanticError("complete projection lacks stack contexts")


def _invert_name_map(mapping: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    inverse: dict[str, list[str]] = {}
    for source in sorted(mapping):
        for destination in mapping[source]:
            inverse.setdefault(destination, []).append(source)
    return inverse


def _require_sorted(mapping: Mapping[str, object], name: str) -> None:
    if list(mapping) != sorted(mapping):
        raise _SemanticError(f"{name} keys must be sorted")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve one event from the command line and print canonical JSON."""
    parser = argparse.ArgumentParser(
        description="Resolve a Spyre profiler event against spyre_provenance.json",
        epilog=(
            "Run this offline reader with backend autoload disabled:\n"
            "  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m "
            "torch_spyre.provenance '<event-name>' <artifact-path> "
            "[--occurrence-selector TOKEN]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("event_name")
    parser.add_argument("artifact_path")
    parser.add_argument("--occurrence-selector")
    args = parser.parse_args(argv)
    result = resolve_provenance_event(
        args.event_name,
        args.artifact_path,
        occurrence_selector=args.occurrence_selector,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
