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

"""Compile-scoped collection for the durable Spyre provenance artifact.

This module stops at an immutable in-memory contribution. Physical publication,
upstream projection, and merge policy are separate lifecycle steps so finalized
OpSpec collection remains reusable by a future KTIR serializer while wire
serialization stays independently testable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from torch._inductor.graph import GraphLowering
from torch_spyre._inductor.kernel_provenance import KernelProvenanceDescriptor
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    LoopSpec,
    OpSpec,
    ProvenanceTransform,
    SourceLoc,
)
from torch_spyre._inductor.profiler_event import (
    format_kernel_provenance_event_name,
)


_GRAPH_REGISTRATIONS_ATTRIBUTE = "_spyre_provenance_registrations"
_GRAPH_REGISTRATION_FAILURE_ATTRIBUTE = "_spyre_provenance_registration_failed"
_COMPILE_ID_DOMAIN = "torch-spyre-compile-v1"
_OCCURRENCE_ID_DOMAIN = "torch-spyre-occurrence-v1"


@dataclasses.dataclass(frozen=True)
class KernelRegistration:
    """One exact upstream alias returned during scheduler code generation."""

    ordinal: int
    alias: str

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or self.ordinal < 1:
            raise ValueError("kernel registration ordinal must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {"ordinal": self.ordinal, "alias": self.alias}


@dataclasses.dataclass(frozen=True)
class KernelRegistrationState:
    """Consumed graph-scoped registrations and their availability state."""

    has_graph_lowering: bool
    registrations: Mapping[str, tuple[KernelRegistration, ...]]
    capture_failed: bool


@dataclasses.dataclass(frozen=True)
class NormalizedDebugHandle:
    """Wire-ready DebugHandle with constituent handles normalized by ID."""

    id: str
    source: SourceLoc | None
    aten_op: str | None
    ir_chain: tuple[str, ...]
    fused_from: tuple[str, ...]
    transform_history: tuple[ProvenanceTransform, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source.to_dict() if self.source is not None else None,
            "aten_op": self.aten_op,
            "ir_chain": list(self.ir_chain),
            "fused_from": list(self.fused_from),
            "transform_history": [
                transform.to_dict() for transform in self.transform_history
            ],
        }


@dataclasses.dataclass(frozen=True)
class SpecHandleBinding:
    """Direct handle attachment at one finalized OpSpec tree position."""

    spec_path: tuple[int, ...]
    handle_id: str

    def __post_init__(self) -> None:
        if not self.spec_path:
            raise ValueError("spec handle binding path must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {"specPath": list(self.spec_path), "handleId": self.handle_id}


@dataclasses.dataclass(frozen=True)
class CollectedKernelIdentity:
    """Cache-stable identity and full direct-handle layout for one bundle."""

    key: str
    direct_handle_ids: tuple[str, ...]
    spec_handle_bindings: tuple[SpecHandleBinding, ...]
    aten_ops: tuple[str, ...]
    event_name_base: str

    def to_dict(self) -> dict[str, object]:
        return {
            "directHandleIds": list(self.direct_handle_ids),
            "specHandleBindings": [
                binding.to_dict() for binding in self.spec_handle_bindings
            ],
            "atenOps": list(self.aten_ops),
            "eventNameBase": self.event_name_base,
        }


@dataclasses.dataclass(frozen=True)
class CollectedKernel:
    """One finalized generated-wrapper kernel before compile ID assignment."""

    compiler_kernel_name: str
    identity: CollectedKernelIdentity
    handles: Mapping[str, NormalizedDebugHandle]


@dataclasses.dataclass(frozen=True)
class UncollectedKernel:
    """Kernel lacking a safe identity, retained with any exact registrations."""

    compiler_kernel_name: str
    registrations: tuple[KernelRegistration, ...]


@dataclasses.dataclass(frozen=True)
class CollectedKernelOccurrence:
    """Compile-scoped occurrence of one cache-stable kernel identity."""

    occurrence_id: str
    identity_key: str
    compile_id: str
    compiler_kernel_name: str
    registrations: tuple[KernelRegistration, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "identityKey": self.identity_key,
            "compileId": self.compile_id,
            "compilerKernelName": self.compiler_kernel_name,
            "registrations": [
                registration.to_dict() for registration in self.registrations
            ],
        }


@dataclasses.dataclass(frozen=True)
class CollectedProvenance:
    """In-memory Spyre collection outcome for one generated wrapper."""

    compile_id: str
    kernels: tuple[tuple[str, str], ...]
    handles: Mapping[str, NormalizedDebugHandle]
    kernel_identities: Mapping[str, CollectedKernelIdentity]
    kernel_occurrences: Mapping[str, CollectedKernelOccurrence]
    uncollected_kernels: tuple[UncollectedKernel, ...]
    has_graph_lowering: bool
    registration_capture_failed: bool

    @property
    def collection_failure_count(self) -> int:
        return len(self.uncollected_kernels)


def record_kernel_registration(
    graph: object,
    kernel_name: str,
    ordinal: int | None,
) -> None:
    """Append an exact registration to the current GraphLowering side table."""
    if ordinal is None:
        return
    registration = KernelRegistration(
        ordinal=ordinal,
        alias=f"{kernel_name}:{ordinal}",
    )
    namespace = _graph_namespace(graph)
    registrations = namespace.setdefault(_GRAPH_REGISTRATIONS_ATTRIBUTE, {})
    if not isinstance(registrations, dict):
        raise TypeError("Spyre provenance registration side table is not a dict")
    kernel_registrations = registrations.setdefault(kernel_name, [])
    if not isinstance(kernel_registrations, list):
        raise TypeError("Spyre kernel registration entry is not a list")
    if kernel_registrations and kernel_registrations[-1].ordinal >= ordinal:
        raise ValueError(
            "kernel registration ordinals must increase within one compilation"
        )
    kernel_registrations.append(registration)


def consume_kernel_registration_state(
    graph: object,
) -> KernelRegistrationState:
    """Consume graph state or explicitly report generated-wrapper replay."""
    if not isinstance(graph, GraphLowering):
        return KernelRegistrationState(
            has_graph_lowering=False,
            registrations=MappingProxyType({}),
            capture_failed=False,
        )

    namespace = _graph_namespace(graph)
    registrations = namespace.pop(_GRAPH_REGISTRATIONS_ATTRIBUTE, {})
    if not isinstance(registrations, dict):
        raise TypeError("Spyre provenance registration side table is not a dict")
    result: dict[str, tuple[KernelRegistration, ...]] = {}
    for kernel_name, values in registrations.items():
        if not isinstance(kernel_name, str) or not isinstance(values, list):
            raise TypeError("invalid Spyre provenance registration entry")
        if not all(isinstance(value, KernelRegistration) for value in values):
            raise TypeError("invalid Spyre kernel registration value")
        result[kernel_name] = tuple(values)
    capture_failed = namespace.pop(
        _GRAPH_REGISTRATION_FAILURE_ATTRIBUTE,
        False,
    )
    if not isinstance(capture_failed, bool):
        raise TypeError("Spyre provenance registration failure marker is not bool")
    return KernelRegistrationState(
        has_graph_lowering=True,
        registrations=MappingProxyType(dict(result)),
        capture_failed=capture_failed,
    )


def mark_kernel_registration_failure(graph: object) -> None:
    """Mark alias capture incomplete without failing compiler code generation."""
    _graph_namespace(graph)[_GRAPH_REGISTRATION_FAILURE_ATTRIBUTE] = True


def collect_kernel_provenance(
    kernel_name: str,
    specs: Sequence[OpSpec | LoopSpec],
    descriptor: KernelProvenanceDescriptor,
) -> CollectedKernel:
    """Normalize every handle reachable from one finalized OpSpec tree."""
    handles: dict[str, NormalizedDebugHandle] = {}
    bindings: list[SpecHandleBinding] = []
    normalized_objects: set[int] = set()

    def walk(entries: Sequence[object], prefix: tuple[int, ...] = ()) -> None:
        for index, spec in enumerate(entries):
            spec_path = (*prefix, index)
            if isinstance(spec, OpSpec):
                if spec.debug_handle is not None:
                    handle_id = str(spec.debug_handle.id)
                    bindings.append(SpecHandleBinding(spec_path, handle_id))
                    _normalize_handle(
                        spec.debug_handle,
                        handles,
                        set(),
                        normalized_objects,
                    )
            elif isinstance(spec, LoopSpec):
                walk(spec.body, spec_path)
            else:
                raise TypeError(
                    f"Unsupported finalized kernel spec: {type(spec).__qualname__}"
                )

    walk(specs)
    direct_handle_ids = tuple(dict.fromkeys(binding.handle_id for binding in bindings))
    if direct_handle_ids != descriptor.debug_handle_ids:
        raise ValueError("descriptor handle order disagrees with finalized OpSpecs")

    identity = CollectedKernelIdentity(
        key=descriptor.key,
        direct_handle_ids=descriptor.debug_handle_ids,
        spec_handle_bindings=tuple(bindings),
        aten_ops=descriptor.aten_ops,
        event_name_base=format_kernel_provenance_event_name(descriptor),
    )
    return CollectedKernel(
        compiler_kernel_name=kernel_name,
        identity=identity,
        handles=MappingProxyType(dict(sorted(handles.items()))),
    )


class ProvenanceCollectionBuilder:
    """Ordered per-wrapper accumulator for finalized Spyre kernels."""

    def __init__(self) -> None:
        self._kernels: list[CollectedKernel] = []
        self._uncollected_kernel_names: list[str] = []
        self._all_kernel_names: set[str] = set()

    def add_kernel(self, kernel: CollectedKernel) -> None:
        self._reserve_kernel_name(kernel.compiler_kernel_name)
        self._kernels.append(kernel)

    def add_uncollected_kernel(self, kernel_name: str) -> None:
        """Retain a kernel that could not safely produce a bundle identity."""
        self._reserve_kernel_name(kernel_name)
        self._uncollected_kernel_names.append(kernel_name)

    def _reserve_kernel_name(self, kernel_name: str) -> None:
        if kernel_name in self._all_kernel_names:
            raise ValueError(
                "duplicate compiler kernel name in one generated wrapper: "
                f"{kernel_name}"
            )
        self._all_kernel_names.add(kernel_name)

    def finish(
        self,
        registration_state: KernelRegistrationState,
    ) -> CollectedProvenance | None:
        """Assign compile/occurrence IDs and preserve partial collection state."""
        registrations = registration_state.registrations
        if not self._all_kernel_names:
            if registrations:
                raise ValueError("registrations exist without collected Spyre kernels")
            return None

        unknown_names = set(registrations) - self._all_kernel_names
        if unknown_names:
            raise ValueError(
                "registrations reference unknown Spyre kernels: "
                f"{sorted(unknown_names)}"
            )
        _validate_registrations(registrations)

        kernels = tuple(
            (kernel.compiler_kernel_name, kernel.identity.key)
            for kernel in self._kernels
        )
        compile_id = _canonical_digest(
            {
                "domain": _COMPILE_ID_DOMAIN,
                "kernels": [list(kernel) for kernel in kernels],
            }
        )

        handles: dict[str, NormalizedDebugHandle] = {}
        identities: dict[str, CollectedKernelIdentity] = {}
        occurrences: dict[str, CollectedKernelOccurrence] = {}
        for kernel in self._kernels:
            _merge_equal_records(handles, kernel.handles, "debug handle")
            _merge_equal_records(
                identities,
                {kernel.identity.key: kernel.identity},
                "kernel identity",
            )
            occurrence_payload = {
                "domain": _OCCURRENCE_ID_DOMAIN,
                "compileId": compile_id,
                "compilerKernelName": kernel.compiler_kernel_name,
                "identityKey": kernel.identity.key,
            }
            occurrence_id = _canonical_digest(occurrence_payload)
            occurrence = CollectedKernelOccurrence(
                occurrence_id=occurrence_id,
                identity_key=kernel.identity.key,
                compile_id=compile_id,
                compiler_kernel_name=kernel.compiler_kernel_name,
                registrations=tuple(registrations.get(kernel.compiler_kernel_name, ())),
            )
            _merge_equal_records(
                occurrences,
                {occurrence_id: occurrence},
                "kernel occurrence",
            )

        uncollected_kernels = tuple(
            UncollectedKernel(
                compiler_kernel_name=kernel_name,
                registrations=tuple(registrations.get(kernel_name, ())),
            )
            for kernel_name in self._uncollected_kernel_names
        )
        return CollectedProvenance(
            compile_id=compile_id,
            kernels=kernels,
            handles=MappingProxyType(dict(sorted(handles.items()))),
            kernel_identities=MappingProxyType(dict(sorted(identities.items()))),
            kernel_occurrences=MappingProxyType(dict(sorted(occurrences.items()))),
            uncollected_kernels=uncollected_kernels,
            has_graph_lowering=registration_state.has_graph_lowering,
            registration_capture_failed=registration_state.capture_failed,
        )


def _normalize_handle(
    handle: DebugHandle,
    handles: dict[str, NormalizedDebugHandle],
    visiting: set[str],
    normalized_objects: set[int],
) -> None:
    handle_id = str(handle.id)
    if handle_id in visiting:
        raise ValueError(f"cyclic fused_from lineage at debug handle {handle_id}")
    object_id = id(handle)
    if object_id in normalized_objects:
        normalized = _normalized_handle_record(handle)
        existing = handles.get(handle_id)
        if existing != normalized:
            raise ValueError(f"conflicting content for debug handle ID {handle_id}")
        return

    visiting.add(handle_id)
    for constituent in handle.fused_from:
        _normalize_handle(
            constituent,
            handles,
            visiting,
            normalized_objects,
        )
    visiting.remove(handle_id)
    constituent_ids = tuple(str(child.id) for child in handle.fused_from)
    if len(constituent_ids) != len(set(constituent_ids)):
        raise ValueError(
            f"duplicate fused_from constituent on debug handle {handle_id}"
        )

    normalized = _normalized_handle_record(handle)
    existing = handles.get(handle_id)
    if existing is not None and existing != normalized:
        raise ValueError(f"conflicting content for debug handle ID {handle_id}")
    handles[handle_id] = normalized
    normalized_objects.add(object_id)


def _normalized_handle_record(handle: DebugHandle) -> NormalizedDebugHandle:
    return NormalizedDebugHandle(
        id=str(handle.id),
        source=handle.source,
        aten_op=handle.aten_op,
        ir_chain=handle.ir_chain,
        fused_from=tuple(str(child.id) for child in handle.fused_from),
        transform_history=handle.transform_history,
    )


def _validate_registrations(
    registrations: Mapping[str, Sequence[KernelRegistration]],
) -> None:
    seen_ordinals: set[int] = set()
    for kernel_name, kernel_registrations in registrations.items():
        ordinals = [registration.ordinal for registration in kernel_registrations]
        if ordinals != sorted(set(ordinals)):
            raise ValueError(
                "kernel registrations must have unique increasing ordinals"
            )
        if seen_ordinals.intersection(ordinals):
            raise ValueError(
                "kernel registration ordinals must be unique within a compilation"
            )
        if any(
            registration.alias != f"{kernel_name}:{registration.ordinal}"
            for registration in kernel_registrations
        ):
            raise ValueError(
                "kernel registration alias disagrees with its name and ordinal"
            )
        seen_ordinals.update(ordinals)


def _merge_equal_records(
    destination: dict[str, Any],
    incoming: Mapping[str, Any],
    record_name: str,
) -> None:
    for key, value in incoming.items():
        existing = destination.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting {record_name} content for key {key}")
        destination[key] = value


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _graph_namespace(graph: object) -> dict[str, Any]:
    if not isinstance(graph, GraphLowering):
        raise TypeError("Spyre provenance state requires a real GraphLowering")
    namespace = getattr(graph, "__dict__", None)
    if not isinstance(namespace, dict):
        raise TypeError("GraphLowering does not expose a mutable namespace")
    return namespace
