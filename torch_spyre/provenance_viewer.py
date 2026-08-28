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

"""Generate a self-contained Spyre provenance viewer from saved artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, cast

import regex

from torch_spyre.provenance import (
    load_provenance_document,
    ProvenanceReaderError,
)
from torch_spyre.provenance_codec import (
    KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
    parse_kernel_provenance_event_name,
)


_PRESENTATION_VERSION = 1
_MAX_SIDECAR_BYTES = 256 * 1024 * 1024
_MAX_KINETO_TRACE_BYTES = 256 * 1024 * 1024
_MAX_PANEL_ROWS = 10_000
_MAX_RUNTIME_OCCURRENCES = 10_000
_HASH_CHUNK_BYTES = 1024 * 1024
_NATIVE_KEY_RE = regex.compile(rf"\A[a-z2-7]{{{KERNEL_PROVENANCE_KEY_BASE32_WIDTH}}}\Z")
_RELATION_FIELDS = ("compileAliases", "handleIds", "postNodes", "preNodes")


class _InputError(ValueError):
    """An optional or bounded viewer input could not be accepted."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def build_provenance_presentation(
    sidecar_path: str | Path,
    *,
    kineto_trace: str | Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic presentation consumed by the offline page."""
    sidecar = Path(sidecar_path)
    _check_size(sidecar, _MAX_SIDECAR_BYTES, "sidecar")
    document = load_provenance_document(sidecar)
    diagnostics: list[dict[str, Any]] = []

    if document["status"] != "complete" or document["diagnostics"]:
        diagnostics.append(
            _diagnostic(
                "incomplete-artifact",
                "warning",
                "sidecar records incomplete collection or upstream evidence",
                {
                    "artifactStatus": document["status"],
                    "writerDiagnostics": copy.deepcopy(document["diagnostics"]),
                },
            )
        )

    occurrences_by_identity: dict[str, list[dict[str, Any]]] = {
        identity_key: [] for identity_key in document["kernelIdentities"]
    }
    for occurrence_id, occurrence in document["kernelOccurrences"].items():
        normalized = {
            "occurrenceId": occurrence_id,
            **copy.deepcopy(occurrence),
            "upstreamJoin": document["upstreamProjections"][occurrence["compileId"]][
                "upstreamJoin"
            ],
        }
        occurrences_by_identity[occurrence["identityKey"]].append(normalized)
    for occurrences in occurrences_by_identity.values():
        occurrences.sort(key=lambda item: item["occurrenceId"])

    identities: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    events_by_key: dict[str, dict[str, Any]] = {}
    for identity_key, identity in document["kernelIdentities"].items():
        occurrences = occurrences_by_identity[identity_key]
        normalized = _identity_presentation(
            identity_key,
            identity,
            occurrences,
            document["handles"],
            document["upstreamProjections"],
            diagnostics,
        )
        identities[identity_key] = normalized
        event = {
            "baseName": identity["eventNameBase"],
            "identityKey": identity_key,
            "observations": [],
        }
        events.append(event)
        events_by_key[identity_key] = event
    events.sort(key=lambda item: item["baseName"])

    trace_input, unresolved = _load_kineto_trace(
        kineto_trace,
        document["kernelIdentities"],
        events_by_key,
        diagnostics,
    )
    observation_summary = _bound_runtime_occurrences(events, diagnostics)

    diagnostics = _sorted_diagnostics(diagnostics)
    presentation = {
        "presentationVersion": _PRESENTATION_VERSION,
        "status": _status_from_diagnostics(diagnostics),
        "runSummary": {
            "resolvedObservations": observation_summary["total"],
            "displayedObservations": observation_summary["displayed"],
            "omittedResolvedObservations": observation_summary["omitted"],
            "unresolvedObservations": len(unresolved),
        },
        "events": events,
        "identities": identities,
        "unresolvedObservations": unresolved,
        "inputs": {
            "sidecar": {"status": "available", **_file_facts(sidecar)},
            "kinetoTrace": trace_input,
        },
        "diagnostics": diagnostics,
    }
    return cast(dict[str, Any], _sorted_json_value(presentation))


def _identity_presentation(
    identity_key: str,
    identity: Mapping[str, Any],
    occurrences: Sequence[Mapping[str, Any]],
    all_handles: Mapping[str, Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_ids = list(identity["directHandleIds"])
    handle_ids = _reachable_handle_ids(direct_ids, all_handles)
    handles = {handle_id: all_handles[handle_id] for handle_id in handle_ids}
    closure = {
        handle_id: _reachable_handle_ids([handle_id], all_handles)
        for handle_id in handle_ids
    }
    candidate_state = "ambiguous" if len(occurrences) > 1 else "unique"

    source_rows = _source_rows(handles)
    aten_rows = _aten_rows(handles)
    lower_rows = _lower_ir_rows(handles, closure)
    source_handle_count = sum(
        handle["source"] is not None for handle in handles.values()
    )
    aten_handle_count = sum(
        handle["aten_op"] is not None for handle in handles.values()
    )
    post_rows, pre_rows = _fx_rows(
        occurrences,
        projections,
        handles,
        closure,
        candidate_state,
    )
    binding_rows = _binding_rows(identity["specHandleBindings"], closure)

    aliases = sorted(
        {
            registration["alias"]
            for occurrence in occurrences
            for registration in occurrence["registrations"]
        }
    )
    compile_ids = sorted({occurrence["compileId"] for occurrence in occurrences})

    panels = [
        _panel(
            "source",
            "Python source locations",
            (
                "One row per unique Python source range recorded for this "
                "bundle. The count shows how many handles contribute to it."
            ),
            f"{len(source_rows)} structured locations",
            source_rows,
            "No structured source locations were recorded.",
            [f"Source coverage: {source_handle_count} of {len(handles)} handles"],
        ),
        _panel(
            "aten",
            "Recorded ATen identities",
            (
                "One row per unique ATen operation identity in this bundle. "
                "Equal names are grouped without losing their handle count."
            ),
            f"{len(aten_rows)} unique identities",
            aten_rows,
            "No ATen identities were recorded.",
            [f"ATen coverage: {aten_handle_count} of {len(handles)} handles"],
        ),
        _panel(
            "pre-grad",
            "FX pre-grad nodes",
            (
                "One row per pre-grad FX node and compile context, reached "
                "through the recorded post-grad-to-pre-grad mapping."
            ),
            _fx_panel_summary(pre_rows, compile_ids),
            pre_rows,
            "No pre-grad FX relationship was recorded.",
            ["PyTorch Inductor mapping version 2.0"],
        ),
        _panel(
            "post-grad",
            "FX post-grad nodes",
            (
                "One row per post-grad FX node and compile context, reached "
                "from a compiler alias. Badges show exact or derived attribution."
            ),
            _fx_panel_summary(post_rows, compile_ids),
            post_rows,
            "No post-grad FX relationship was recorded.",
            [f"{len(aliases)} registered aliases"],
        ),
        _panel(
            "lower-ir",
            "Recorded lower-IR lineage by handle",
            (
                "One row per debug handle, grouping its ordered IR names and "
                "recorded transformation steps."
            ),
            f"{len(lower_rows)} handles with recorded lineage",
            lower_rows,
            "No lower-IR lineage was recorded.",
            [f"Lineage coverage: {len(lower_rows)} of {len(handles)} handles"],
        ),
        _panel(
            "opspec",
            "Direct OpSpec bindings",
            (
                "One row per ordered direct-handle attachment in the finalized "
                "OpSpec tree. SpecPath is the attachment's structural position."
            ),
            (
                f"{len(binding_rows)} bindings | {len(direct_ids)} direct / "
                f"{len(handle_ids)} recursive handles"
            ),
            binding_rows,
            "No direct OpSpec bindings were recorded.",
        ),
    ]
    for panel in panels:
        if panel["omittedRowCount"]:
            diagnostics.append(
                _diagnostic(
                    "panel-rows-truncated",
                    "warning",
                    "panel rows exceed the offline viewer limit",
                    {
                        "identityKey": identity_key,
                        "panelId": panel["id"],
                        "rowLimit": _MAX_PANEL_ROWS,
                        "totalRows": panel["totalRowCount"],
                        "displayedRows": panel["displayedRowCount"],
                        "omittedRows": panel["omittedRowCount"],
                    },
                )
            )
    return {
        "identityKey": identity_key,
        "eventNameBase": identity["eventNameBase"],
        "compileCandidateCount": len(occurrences),
        "compileIds": compile_ids,
        "directHandleIds": direct_ids,
        "recursiveHandleIds": handle_ids,
        "panels": panels,
    }


def _source_rows(
    handles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for handle_id, handle in handles.items():
        source = handle["source"]
        if source is None:
            continue
        key = _canonical_json(source)
        group = grouped.setdefault(
            key,
            {"source": copy.deepcopy(source), "handleIds": []},
        )
        group["handleIds"].append(handle_id)

    rows = []
    for key, group in sorted(grouped.items()):
        source = group["source"]
        handle_ids = sorted(group["handleIds"])
        rows.append(
            _evidence_row(
                "source",
                key,
                _source_label(source),
                _count_summary(len(handle_ids), "contributing handle"),
                "exact",
                refs={"handleIds": handle_ids},
            )
        )
    return rows


def _aten_rows(
    handles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for handle_id, handle in handles.items():
        aten_op = handle["aten_op"]
        if aten_op is not None:
            grouped.setdefault(aten_op, []).append(handle_id)
    return [
        _evidence_row(
            "aten",
            aten_op,
            aten_op,
            _count_summary(len(handle_ids), "contributing handle"),
            "exact",
            refs={"handleIds": sorted(handle_ids)},
        )
        for aten_op, handle_ids in sorted(grouped.items())
    ]


def _lower_ir_rows(
    handles: Mapping[str, Mapping[str, Any]],
    closure: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows = []
    for handle_id, handle in handles.items():
        chain = list(handle["ir_chain"])
        transforms = list(handle["transform_history"])
        if not chain and not transforms:
            continue
        summary = []
        if chain:
            summary.append("IR: " + " -> ".join(chain))
        if transforms:
            summary.append(
                "Transforms: "
                + ", ".join(
                    item["kind"] + " by " + item["pass_name"] for item in transforms
                )
            )
        rows.append(
            _evidence_row(
                "lower-ir",
                handle_id,
                "Handle " + handle_id,
                " | ".join(summary),
                "exact",
                refs={"handleIds": list(closure[handle_id])},
            )
        )
    return rows


def _fx_rows(
    occurrences: Sequence[Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
    handles: Mapping[str, Mapping[str, Any]],
    closure: Mapping[str, Sequence[str]],
    candidate_state: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    post_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for occurrence in occurrences:
        compile_id = occurrence["compileId"]
        projection = projections[compile_id]
        for registration in occurrence["registrations"]:
            alias = registration["alias"]
            for post_name in projection["cppCodeToPost"].get(alias, []):
                group = post_groups.setdefault(
                    (compile_id, post_name),
                    {"aliases": set(), "occurrenceIds": set()},
                )
                group["aliases"].add(alias)
                group["occurrenceIds"].add(occurrence["occurrenceId"])

    post_rows = []
    post_row_data: dict[tuple[str, str], dict[str, Any]] = {}
    all_handle_ids = sorted(handles)
    for (compile_id, post_name), group in sorted(post_groups.items()):
        exact_handles = sorted(
            handle_id
            for handle_id, handle in handles.items()
            if post_name in handle["ir_chain"]
        )
        if exact_handles:
            related_handles = sorted(
                {
                    related
                    for handle_id in exact_handles
                    for related in closure[handle_id]
                }
            )
            strength = "exact"
        else:
            related_handles = all_handle_ids
            strength = "derived"
        aliases = sorted(group["aliases"])
        scoped_post = _compile_scoped_ref(compile_id, post_name)
        row = _evidence_row(
            "post-grad",
            {"compileId": compile_id, "name": post_name},
            post_name,
            (
                f"Compile {compile_id[:12]} | "
                f"{_count_summary(len(aliases), 'alias', 'aliases')}"
            ),
            strength,
            candidate_state=candidate_state,
            refs={
                "compileAliases": [
                    _compile_scoped_ref(compile_id, alias) for alias in aliases
                ],
                "handleIds": related_handles,
                "postNodes": [scoped_post],
            },
        )
        post_rows.append(row)
        post_row_data[(compile_id, post_name)] = row

    pre_groups: dict[tuple[str, str], dict[str, set[str]]] = {}
    for (compile_id, post_name), post_row in post_row_data.items():
        projection = projections[compile_id]
        for pre_name in projection["postToPre"].get(post_name, []):
            group = pre_groups.setdefault(
                (compile_id, pre_name),
                {"handleIds": set(), "postNodes": set()},
            )
            group["handleIds"].update(post_row["refs"]["handleIds"])
            group["postNodes"].add(_compile_scoped_ref(compile_id, post_name))

    pre_rows = []
    for (compile_id, pre_name), group in sorted(pre_groups.items()):
        pre_rows.append(
            _evidence_row(
                "pre-grad",
                {"compileId": compile_id, "name": pre_name},
                pre_name,
                (
                    f"Compile {compile_id[:12]} | "
                    f"{_count_summary(len(group['postNodes']), 'post-grad node')}"
                ),
                "derived",
                candidate_state=candidate_state,
                refs={
                    "handleIds": sorted(group["handleIds"]),
                    "postNodes": sorted(group["postNodes"]),
                    "preNodes": [_compile_scoped_ref(compile_id, pre_name)],
                },
            )
        )
    return post_rows, pre_rows


def _binding_rows(
    bindings: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows = []
    for position, binding in enumerate(bindings):
        handle_id = binding["handleId"]
        spec_path = list(binding["specPath"])
        rows.append(
            _evidence_row(
                "opspec",
                {"position": position, **copy.deepcopy(binding)},
                "SpecPath [" + ", ".join(map(str, spec_path)) + "]",
                f"Binding {position} | Handle {handle_id}",
                "exact",
                refs={"handleIds": list(closure[handle_id])},
            )
        )
    return rows


def _evidence_row(
    prefix: str,
    identity: object,
    label: str,
    summary: str,
    strength: str,
    *,
    candidate_state: str = "unique",
    refs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    relation = {
        field: sorted(set((refs or {}).get(field, []))) for field in _RELATION_FIELDS
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "id": prefix + ":" + digest[:16],
        "label": label,
        "summary": summary,
        "evidenceStrength": strength,
        "candidateState": candidate_state,
        "refs": relation,
    }


def _panel(
    panel_id: str,
    title: str,
    description: str,
    summary: str,
    rows: Sequence[Mapping[str, Any]],
    empty_message: str,
    scope_details: Sequence[str] = (),
) -> dict[str, Any]:
    all_rows = list(rows)
    total_row_count = len(all_rows)
    displayed_rows = all_rows[:_MAX_PANEL_ROWS]
    displayed_row_count = len(displayed_rows)
    omitted_row_count = total_row_count - displayed_row_count
    if omitted_row_count:
        summary += f" | showing {displayed_row_count} of {total_row_count} rows"
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "summary": summary,
        "scopeDetails": list(scope_details),
        "emptyMessage": empty_message,
        "totalRowCount": total_row_count,
        "displayedRowCount": displayed_row_count,
        "omittedRowCount": omitted_row_count,
        "rows": copy.deepcopy(displayed_rows),
    }


def _fx_panel_summary(
    rows: Sequence[Mapping[str, Any]],
    compile_ids: Sequence[str],
) -> str:
    return f"{len(rows)} nodes across {len(compile_ids)} compile candidates"


def _source_label(source: Mapping[str, Any]) -> str:
    label = f"{source['file']}:{source['start_line']}:{source['start_col']}"
    if source["end_line"] is not None:
        label += f"-{source['end_line']}:{source['end_col']}"
    return label


def _count_summary(count: int, unit: str, plural: str | None = None) -> str:
    return f"{count} {unit if count == 1 else plural or unit + 's'}"


def _compile_scoped_ref(compile_id: str, name: str) -> str:
    return _canonical_json({"compileId": compile_id, "name": name})


def _reachable_handle_ids(
    direct_ids: Sequence[str],
    handles: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()
    pending = list(reversed(direct_ids))
    while pending:
        handle_id = pending.pop()
        if handle_id in visited:
            continue
        visited.add(handle_id)
        ordered.append(handle_id)
        pending.extend(reversed(handles[handle_id]["fused_from"]))
    return ordered


def _bound_runtime_occurrences(
    events: Sequence[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, int]:
    ordered: list[tuple[int, dict[str, Any]]] = []
    for event in events:
        event["observations"].sort(key=lambda item: item["traceEventIndex"])
        event["observationCount"] = len(event["observations"])
        ordered.extend(
            (observation["traceEventIndex"], observation)
            for observation in event["observations"]
        )
    ordered.sort(key=lambda item: item[0])
    retained_indices = {
        trace_index for trace_index, _ in ordered[:_MAX_RUNTIME_OCCURRENCES]
    }

    displayed_count = 0
    for event in events:
        event["observations"] = [
            observation
            for observation in event["observations"]
            if observation["traceEventIndex"] in retained_indices
        ]
        event["displayedObservationCount"] = len(event["observations"])
        event["omittedObservationCount"] = (
            event["observationCount"] - event["displayedObservationCount"]
        )
        displayed_count += event["displayedObservationCount"]

    total_count = len(ordered)
    omitted_count = total_count - displayed_count
    if omitted_count:
        diagnostics.append(
            _diagnostic(
                "runtime-occurrences-truncated",
                "warning",
                "runtime occurrences exceed the offline viewer limit",
                {
                    "occurrenceLimit": _MAX_RUNTIME_OCCURRENCES,
                    "totalOccurrences": total_count,
                    "displayedOccurrences": displayed_count,
                    "omittedOccurrences": omitted_count,
                    "retentionPolicy": "earliest-trace-event-index",
                },
            )
        )
    return {
        "total": total_count,
        "displayed": displayed_count,
        "omitted": omitted_count,
    }


def _load_kineto_trace(
    trace_path: str | Path | None,
    identities: Mapping[str, Mapping[str, Any]],
    events_by_key: Mapping[str, dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if trace_path is None:
        return _omitted_input(), []
    path = Path(trace_path)
    try:
        value = _read_json_with_limit(path, _MAX_KINETO_TRACE_BYTES, "Kineto trace")
    except _InputError as error:
        diagnostics.append(_diagnostic(error.code, "warning", str(error)))
        return {
            "status": "unavailable",
            "path": str(path.resolve()),
            "sizeBytes": _safe_size(path),
            "sha256": None,
            "pairing": None,
        }, []

    trace_events = value.get("traceEvents") if isinstance(value, dict) else None
    if not isinstance(trace_events, list):
        diagnostics.append(
            _diagnostic(
                "kineto-trace-schema-failure",
                "warning",
                "Kineto trace must contain a traceEvents array",
            )
        )
        return {
            "status": "unavailable",
            **_file_facts(path),
            "pairing": None,
        }, []

    candidate_count = 0
    resolved_count = 0
    unresolved: list[dict[str, Any]] = []
    for trace_index, raw_event in enumerate(trace_events):
        if not isinstance(raw_event, dict):
            continue
        name = raw_event.get("name")
        name = name if isinstance(name, str) else None
        args = raw_event.get("args")
        args = args if isinstance(args, dict) else {}
        parsed = parse_kernel_provenance_event_name(name) if name else None
        native_value = args.get("provenance_key")
        native_key = (
            native_value
            if isinstance(native_value, str)
            and _NATIVE_KEY_RE.fullmatch(native_value) is not None
            else None
        )
        if native_value is not None and native_key is None:
            diagnostics.append(
                _trace_diagnostic(
                    "malformed-native-provenance-key",
                    "warning",
                    "trace activity has a malformed native provenance key",
                    trace_index,
                    name,
                )
            )
        if parsed is None and native_key is None:
            continue

        candidate_count += 1
        name_key = parsed.key if parsed is not None else None
        if native_key is not None and name_key is not None and native_key != name_key:
            unresolved.append(
                _unresolved_observation(
                    trace_index,
                    name,
                    "carrier-conflict",
                    "native and event-name provenance keys disagree",
                )
            )
            diagnostics.append(
                _trace_diagnostic(
                    "carrier-conflict",
                    "error",
                    "native and event-name provenance keys disagree",
                    trace_index,
                    name,
                )
            )
            continue

        key = native_key or name_key
        assert key is not None
        identity = identities.get(key)
        if identity is None:
            unresolved.append(
                _unresolved_observation(
                    trace_index,
                    name,
                    "missing-key",
                    "trace provenance key is absent from the sidecar",
                )
            )
            diagnostics.append(
                _trace_diagnostic(
                    "kineto-sidecar-pairing-mismatch",
                    "error",
                    "trace provenance key is absent from the sidecar",
                    trace_index,
                    name,
                )
            )
            continue
        if parsed is not None and parsed.base_name != identity["eventNameBase"]:
            unresolved.append(
                _unresolved_observation(
                    trace_index,
                    name,
                    "collision",
                    "event key resolves to a different persisted event base",
                )
            )
            diagnostics.append(
                _trace_diagnostic(
                    "collision",
                    "error",
                    "event key resolves to a different persisted event base",
                    trace_index,
                    name,
                )
            )
            continue

        native_handles = _native_handle_ids(
            args.get("debug_handles"),
            trace_index,
            name,
            diagnostics,
        )
        if native_handles is not None and native_handles != identity["directHandleIds"]:
            diagnostics.append(
                _trace_diagnostic(
                    "native-handle-disagreement",
                    "warning",
                    "native direct handles disagree with the sidecar",
                    trace_index,
                    name,
                )
            )

        observation = {
            "traceEventIndex": trace_index,
            "name": name,
            "timestampUs": _finite_number(raw_event.get("ts")),
            "durationUs": _nonnegative_number(raw_event.get("dur")),
            "commandStep": parsed.step if parsed is not None else None,
            "keySource": (
                "both"
                if native_key is not None and name_key is not None
                else "native"
                if native_key is not None
                else "event-name"
            ),
            "nativeDirectHandleIds": native_handles,
            "correlation": args.get("correlation"),
            "processId": raw_event.get("pid"),
            "threadId": raw_event.get("tid"),
            "stream": args.get("stream"),
        }
        events_by_key[key]["observations"].append(observation)
        resolved_count += 1

    if candidate_count == 0:
        diagnostics.append(
            _diagnostic(
                "no-supported-kineto-events",
                "warning",
                "Kineto trace contains no supported Spyre provenance activities",
            )
        )
    pairing_status = "complete" if not unresolved else "error"
    return {
        "status": "available",
        **_file_facts(path),
        "pairing": {
            "status": pairing_status,
            "candidateObservations": candidate_count,
            "resolvedObservations": resolved_count,
            "unresolvedObservations": len(unresolved),
            "resolutionRate": (
                resolved_count / candidate_count if candidate_count else None
            ),
        },
    }, unresolved


def _native_handle_ids(
    value: object,
    trace_index: int,
    name: str | None,
    diagnostics: list[dict[str, Any]],
) -> list[str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) for item in value)
        or len(value) != len(set(value))
    ):
        diagnostics.append(
            _trace_diagnostic(
                "malformed-native-debug-handles",
                "warning",
                "native debug_handles must be a unique string array",
                trace_index,
                name,
            )
        )
        return None
    return list(value)


def _trace_diagnostic(
    code: str,
    severity: str,
    message: str,
    trace_index: int,
    name: str | None,
) -> dict[str, Any]:
    return _diagnostic(
        code,
        severity,
        message,
        {"traceEventIndex": trace_index, "eventName": name},
    )


def _unresolved_observation(
    trace_index: int,
    name: str | None,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "traceEventIndex": trace_index,
        "name": name,
        "diagnostic": {"code": code, "message": message},
    }


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _nonnegative_number(value: object) -> int | float | None:
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def _read_json_with_limit(path: Path, limit: int, label: str) -> object:
    _check_size(path, limit, label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise _InputError("input-read-failure", f"{label} could not be read") from error
    except (RecursionError, UnicodeError, ValueError) as error:
        raise _InputError(
            "input-json-failure",
            f"{label} is not valid UTF-8 JSON",
        ) from error


def _check_size(path: Path, limit: int, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise _InputError("input-read-failure", f"{label} could not be read") from error
    if size > limit:
        raise _InputError(
            "input-size-limit",
            f"{label} exceeds the {limit}-byte input limit",
        )


def _safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _file_facts(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sizeBytes": size,
        "sha256": digest.hexdigest(),
    }


def _omitted_input() -> dict[str, Any]:
    return {
        "status": "omitted",
        "path": None,
        "sizeBytes": None,
        "sha256": None,
        "pairing": None,
    }


def _diagnostic(
    code: str,
    severity: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if details is not None:
        result["details"] = copy.deepcopy(dict(details))
    return result


def _status_from_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> str:
    severities = {item["severity"] for item in diagnostics}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "partial"
    return "complete"


def _sorted_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (copy.deepcopy(dict(item)) for item in diagnostics),
        key=lambda item: (
            item["code"],
            _canonical_json(item.get("details", {})),
            item["message"],
        ),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sorted_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json_value(item) for item in value]
    return value


def render_provenance_html(presentation: Mapping[str, Any]) -> str:
    """Render one deterministic self-contained HTML document."""
    encoded = _canonical_json(presentation)
    encoded = (
        encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Spyre provenance viewer</title>\n"
        "<style>\n" + _CSS + "\n</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "<h1>Spyre provenance viewer</h1>\n"
        '<p id="run-summary"></p>\n'
        "</header>\n"
        "<main>\n"
        '<section class="controls" aria-label="Provenance selection">\n'
        '<label>Profiler event<select id="event-select"></select></label>\n'
        '<label>Runtime occurrence<select id="observation-select"></select></label>\n'
        "</section>\n"
        '<section class="facts" aria-label="Selected profiler event facts">\n'
        '<div><span>Exact observed event name</span><strong id="fact-name"></strong></div>\n'
        '<div><span>Timestamp (us)</span><strong id="fact-ts"></strong></div>\n'
        '<div><span>Duration (us)</span><strong id="fact-duration"></strong></div>\n'
        "<div><span>JobPlan step (static command index)</span>"
        '<strong id="fact-step"></strong></div>\n'
        "<div><span>Compile candidates</span>"
        '<strong id="fact-candidates"></strong></div>\n'
        "</section>\n"
        '<section id="panels" class="panels" aria-label="Provenance evidence"></section>\n'
        "</main>\n"
        '<script id="spyre-provenance-data" type="application/json">'
        + encoded
        + "</script>\n"
        "<script>\n" + _SCRIPT + "\n</script>\n"
        "</body>\n"
        "</html>\n"
    )


def write_provenance_html(
    presentation: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """Atomically write one offline viewer."""
    output = Path(output_path)
    if not output.parent.is_dir():
        raise OSError("output parent does not exist")
    payload = render_provenance_html(presentation)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=output.parent,
            prefix="." + output.name + ".",
            suffix=".tmp",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_CSS = r"""
:root {
  color-scheme: light dark;
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  --background: #f4f4f4;
  --layer: #ffffff;
  --layer-alt: #e8e8e8;
  --text: #161616;
  --muted: #525252;
  --border: #c6c6c6;
  --focus: #198038;
  --selection: #defbe6;
  --selection-border: #198038;
  --exact: #198038;
  --derived: #8e6a00;
  --danger: #da1e28;
}
@media (prefers-color-scheme: dark) {
  :root {
    --background: #161616;
    --layer: #262626;
    --layer-alt: #393939;
    --text: #f4f4f4;
    --muted: #c6c6c6;
    --border: #525252;
    --focus: #42be65;
    --selection: #103b20;
    --selection-border: #42be65;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--background);
  color: var(--text);
  margin: 0;
}
header {
  background: var(--layer);
  border-bottom: 1px solid var(--border);
  padding: 1rem 1.5rem;
}
h1 { font-size: 1.75rem; margin: 0 0 .3rem; }
header p { color: var(--muted); margin: 0; }
main { padding: 1rem 1.5rem 2rem; }
.controls {
  background: var(--layer);
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  padding: 1rem;
}
.controls label {
  color: var(--muted);
  display: grid;
  font-size: .95rem;
  font-weight: 600;
  gap: .35rem;
}
select {
  background: var(--layer);
  border: 1px solid var(--border);
  color: var(--text);
  font: inherit;
  min-height: 2.5rem;
  min-width: 0;
  padding: .5rem;
  width: 100%;
}
select:focus-visible, .evidence-row:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
}
.facts {
  background: var(--layer);
  display: grid;
  gap: 1px;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  margin-top: 1px;
}
.facts div {
  background: var(--layer);
  min-width: 0;
  padding: .75rem 1rem;
}
.facts span {
  color: var(--muted);
  display: block;
  font-size: .82rem;
  font-weight: 600;
}
.facts strong {
  display: block;
  font-size: 1rem;
  margin-top: .25rem;
  overflow-wrap: anywhere;
}
.panels {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: 1rem;
}
.panel {
  background: var(--layer);
  border: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  height: 33rem;
  min-width: 0;
}
.panel-header {
  border-bottom: 1px solid var(--border);
  min-height: 8.75rem;
  padding: .85rem;
}
.panel-header h2 { font-size: 1.15rem; margin: 0 0 .35rem; }
.panel-description {
  font-size: .9rem;
  line-height: 1.4;
  margin: .3rem 0 .55rem;
}
.panel-summary, .scope-detail {
  color: var(--muted);
  font-size: .86rem;
  line-height: 1.35;
  margin: .25rem 0;
  overflow-wrap: anywhere;
}
.rows {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: .5rem;
  min-height: 0;
  overflow: auto;
  padding: .75rem;
}
.evidence-row {
  background: var(--layer-alt);
  border: 1px solid transparent;
  border-left: .25rem solid var(--border);
  color: var(--text);
  cursor: pointer;
  display: block;
  font: inherit;
  padding: .65rem;
  text-align: left;
  width: 100%;
}
.evidence-row:hover { border-color: var(--focus); }
.evidence-row.is-focused {
  background: var(--selection);
  border-color: var(--selection-border);
  border-left-color: var(--selection-border);
  box-shadow: inset 0 0 0 1px var(--selection-border);
}
.evidence-row.is-related {
  background: var(--selection);
  border-left-color: var(--selection-border);
}
.evidence-row.is-dimmed { opacity: .38; }
.row-badges { display: flex; flex-wrap: wrap; gap: .3rem; }
.badge {
  border: 1px solid var(--border);
  border-radius: 1rem;
  font-size: .75rem;
  padding: .1rem .45rem;
  text-transform: capitalize;
}
.badge-exact { border-color: var(--exact); }
.badge-derived { border-color: var(--derived); }
.badge-ambiguous { border-color: var(--danger); }
.row-label {
  display: block;
  font-size: 1rem;
  font-weight: 600;
  margin-top: .4rem;
  overflow-wrap: anywhere;
}
.row-summary {
  display: block;
  font-size: .86rem;
  margin-top: .3rem;
  overflow-wrap: anywhere;
}
.row-summary { color: var(--muted); }
.empty-state {
  color: var(--muted);
  font-size: .9rem;
  font-style: italic;
  padding: .75rem;
}
@media (max-width: 78rem) {
  .panels { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .facts { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 62rem) {
  main { padding: .75rem; }
  .controls, .panels, .facts { grid-template-columns: 1fr; }
  .panel { height: 24rem; }
  .panel-header { min-height: auto; }
}
""".strip()


_SCRIPT = r"""
"use strict";
const data = JSON.parse(
  document.getElementById("spyre-provenance-data").textContent
);
const elements = {
  eventSelect: document.getElementById("event-select"),
  observationSelect: document.getElementById("observation-select"),
  panels: document.getElementById("panels"),
  runSummary: document.getElementById("run-summary"),
  factName: document.getElementById("fact-name"),
  factTimestamp: document.getElementById("fact-ts"),
  factDuration: document.getElementById("fact-duration"),
  factStep: document.getElementById("fact-step"),
  factCandidates: document.getElementById("fact-candidates"),
};
const relationFields = [
  "compileAliases",
  "handleIds",
  "postNodes",
  "preNodes",
];
const state = {
  eventIndex: 0,
  observationIndex: 0,
  focusedPanelId: null,
  focusedRowId: null,
  rowById: new Map(),
};

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) {
    result.className = className;
  }
  if (text !== undefined) {
    result.textContent = String(text);
  }
  return result;
}

function display(value) {
  return value === null || value === undefined ? "Unavailable" : String(value);
}

function currentEvent() {
  return data.events[state.eventIndex] || null;
}

function currentIdentity() {
  const event = currentEvent();
  return event ? data.identities[event.identityKey] : null;
}

function currentObservation() {
  const event = currentEvent();
  return event ? event.observations[state.observationIndex] || null : null;
}

function eventLabel(event) {
  const total = event.observationCount;
  const displayed = event.displayedObservationCount;
  const count = total === displayed
    ? String(total)
    : "showing " + displayed + " of " + total;
  return event.baseName + " | " + count + " runtime occurrence" +
    (total === 1 ? "" : "s");
}

function observationLabel(observation) {
  return "trace[" + observation.traceEventIndex + "] | " +
    display(observation.name) + " | " +
    (observation.durationUs === null
      ? "duration unavailable"
      : observation.durationUs + " us");
}

function populateEvents() {
  elements.eventSelect.replaceChildren();
  data.events.forEach((event, index) => {
    const option = node("option", "", eventLabel(event));
    option.value = String(index);
    elements.eventSelect.append(option);
  });
  elements.eventSelect.disabled = data.events.length === 0;
}

function populateObservations() {
  const event = currentEvent();
  elements.observationSelect.replaceChildren();
  if (!event || !event.observations.length) {
    const option = node("option", "", "No runtime occurrence");
    option.value = "0";
    elements.observationSelect.append(option);
    elements.observationSelect.disabled = true;
    state.observationIndex = 0;
    return;
  }
  event.observations.forEach((observation, index) => {
    const option = node("option", "", observationLabel(observation));
    option.value = String(index);
    elements.observationSelect.append(option);
  });
  elements.observationSelect.disabled = false;
  elements.observationSelect.value = String(state.observationIndex);
}

function renderFacts() {
  const event = currentEvent();
  const identity = currentIdentity();
  const observation = currentObservation();
  elements.factName.textContent = display(
    observation ? observation.name : event ? event.baseName : null
  );
  elements.factTimestamp.textContent = display(
    observation ? observation.timestampUs : null
  );
  elements.factDuration.textContent = display(
    observation ? observation.durationUs : null
  );
  elements.factStep.textContent = display(
    observation ? observation.commandStep : null
  );
  elements.factCandidates.textContent = display(
    identity ? identity.compileCandidateCount : null
  );
}

function badge(value, className) {
  return node("span", "badge " + className, value);
}

function renderRow(row, panelId) {
  const button = node("button", "evidence-row");
  button.type = "button";
  button.dataset.rowId = row.id;
  button.dataset.panelId = panelId;
  const badges = node("span", "row-badges");
  badges.append(
    badge(row.evidenceStrength, "badge-" + row.evidenceStrength)
  );
  if (row.candidateState === "ambiguous") {
    badges.append(badge("multiple candidates", "badge-ambiguous"));
  }
  button.append(
    badges,
    node("span", "row-label", row.label),
    node("span", "row-summary", row.summary)
  );
  button.addEventListener("click", () => focusRow(button, row));
  state.rowById.set(row.id, row);
  return button;
}

function renderPanels() {
  const identity = currentIdentity();
  state.rowById = new Map();
  elements.panels.replaceChildren();
  if (!identity) {
    elements.panels.append(node("p", "empty-state", "No persisted events."));
    return;
  }
  identity.panels.forEach((panel) => {
    const section = node("section", "panel");
    section.dataset.panelId = panel.id;
    const header = node("div", "panel-header");
    header.append(
      node("h2", "", panel.title),
      node("p", "panel-description", panel.description),
      node("p", "panel-summary", panel.summary)
    );
    panel.scopeDetails.forEach((detail) => {
      header.append(node("p", "scope-detail", detail));
    });
    const rows = node("div", "rows");
    if (!panel.rows.length) {
      rows.append(node("p", "empty-state", panel.emptyMessage));
    } else {
      panel.rows.forEach((row) => rows.append(renderRow(row, panel.id)));
    }
    section.append(header, rows);
    elements.panels.append(section);
  });
}

function intersects(left, right) {
  const values = new Set(left);
  return right.some((value) => values.has(value));
}

function rowsRelated(left, right) {
  if (left.id === right.id) {
    return true;
  }
  return relationFields.some(
    (field) => intersects(left.refs[field], right.refs[field])
  );
}

function updateFocusClasses() {
  const focused = state.focusedRowId
    ? state.rowById.get(state.focusedRowId)
    : null;
  document.querySelectorAll(".evidence-row").forEach((button) => {
    const row = state.rowById.get(button.dataset.rowId);
    const samePanel = button.dataset.panelId === state.focusedPanelId;
    const related = focused &&
      (!samePanel || row.id === focused.id) &&
      rowsRelated(focused, row);
    button.classList.toggle("is-focused", Boolean(focused && row.id === focused.id));
    button.classList.toggle(
      "is-related",
      Boolean(focused && related && row.id !== focused.id)
    );
    button.classList.toggle("is-dimmed", Boolean(focused && !related));
    button.setAttribute(
      "aria-pressed",
      focused && row.id === focused.id ? "true" : "false"
    );
  });
}

function centerRelatedRows(clickedPanelId) {
  document.querySelectorAll(".panel").forEach((panel) => {
    if (panel.dataset.panelId === clickedPanelId) {
      return;
    }
    const body = panel.querySelector(".rows");
    const target = body.querySelector(".evidence-row.is-related");
    if (!target) {
      return;
    }
    const bodyBox = body.getBoundingClientRect();
    const targetBox = target.getBoundingClientRect();
    const targetTop = targetBox.top - bodyBox.top + body.scrollTop;
    body.scrollTop = Math.max(
      0,
      targetTop - body.clientHeight / 2 + targetBox.height / 2
    );
  });
}

function focusRow(button, row) {
  const pageX = window.scrollX || 0;
  const pageY = window.scrollY || 0;
  const clickedPanel = button.closest(".panel");
  const clickedBody = clickedPanel.querySelector(".rows");
  const clickedTop = clickedBody.scrollTop;
  const clickedLeft = clickedBody.scrollLeft;
  const clearing = state.focusedRowId === row.id;
  state.focusedRowId = clearing ? null : row.id;
  state.focusedPanelId = clearing ? null : clickedPanel.dataset.panelId;
  updateFocusClasses();
  if (!clearing) {
    centerRelatedRows(clickedPanel.dataset.panelId);
  }
  clickedBody.scrollTop = clickedTop;
  clickedBody.scrollLeft = clickedLeft;
  if (typeof window.scrollTo === "function") {
    window.scrollTo(pageX, pageY);
  }
}

function selectEvent(index) {
  state.eventIndex = index;
  state.observationIndex = 0;
  state.focusedPanelId = null;
  state.focusedRowId = null;
  populateObservations();
  renderFacts();
  renderPanels();
  updateFocusClasses();
}

elements.eventSelect.addEventListener("change", () => {
  selectEvent(Number(elements.eventSelect.value));
});
elements.observationSelect.addEventListener("change", () => {
  state.observationIndex = Number(elements.observationSelect.value);
  renderFacts();
});

const pairing = data.inputs.kinetoTrace.pairing;
const resolved = data.runSummary.resolvedObservations;
const displayed = data.runSummary.displayedObservations;
const summaryParts = [
  (resolved === displayed
    ? resolved
    : displayed + " of " + resolved + " shown") +
    " resolved runtime occurrences",
];
if (data.runSummary.unresolvedObservations) {
  summaryParts.push(data.runSummary.unresolvedObservations + " unresolved");
}
summaryParts.push("status " + data.status);
if (pairing && pairing.resolutionRate !== null) {
  summaryParts.push("pairing " + Math.round(pairing.resolutionRate * 100) + "%");
}
elements.runSummary.textContent = summaryParts.join(" | ");
populateEvents();
selectEvent(0);
window.__spyreProvenanceViewer = {
  data,
  state,
  rowsRelated,
  selectEvent,
};
""".strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Build one offline viewer from a saved sidecar and optional trace."""
    parser = argparse.ArgumentParser(
        description="Build a self-contained Spyre provenance viewer",
        epilog=(
            "Run with backend autoload disabled:\n"
            "  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m "
            "torch_spyre.provenance_viewer SIDECAR --output VIEWER.html"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sidecar", help="validated spyre_provenance.json")
    parser.add_argument(
        "--kineto-trace",
        help="optional Kineto Chrome trace containing runtime occurrences",
    )
    parser.add_argument("--output", required=True, help="output HTML path")
    args = parser.parse_args(argv)

    try:
        presentation = build_provenance_presentation(
            args.sidecar,
            kineto_trace=args.kineto_trace,
        )
    except (ProvenanceReaderError, _InputError) as error:
        code = getattr(error, "code", "viewer-input-failure")
        print(
            _canonical_json(_diagnostic(code, "error", str(error))),
            file=sys.stderr,
        )
        return 1
    try:
        write_provenance_html(presentation, args.output)
    except OSError:
        print(
            _canonical_json(
                _diagnostic(
                    "output-write-failure",
                    "error",
                    "viewer output could not be written",
                )
            ),
            file=sys.stderr,
        )
        return 1
    return 1 if presentation["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
