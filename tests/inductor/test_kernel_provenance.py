# Copyright 2025 The Torch-Spyre Authors.
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

"""Device-free tests for kernel provenance identity and event transport."""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import ctypes
import dataclasses
import hashlib
from importlib.resources import files
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

from jsonschema import ValidationError
from jsonschema.validators import validator_for
import pytest
import regex as re

import torch  # noqa: F401
from sympy import Integer, Symbol, sympify
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._logging._internal import LazyTraceHandler

from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    extract_kernel_provenance_key as extract_kernel_provenance_key_cpp,
)
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    IndirectAccess,
    LoopSpec,
    OpSpec,
    ProvenanceTransform,
    SourceLoc,
    TensorArg,
    TensorWorkDivision,
    UnimplementedOp,
)
from torch_spyre._inductor.kernel_provenance import (
    build_kernel_provenance_descriptor,
    KernelProvenanceDescriptor,
    KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
    KERNEL_PROVENANCE_KEY_VERSION,
)
from torch_spyre._inductor.profiler_event import (
    extract_kernel_provenance_key,
    format_kernel_provenance_event_name,
    parse_kernel_provenance_event_name,
)
from torch_spyre._inductor.provenance_artifact import (
    collect_kernel_provenance,
    consume_kernel_registration_state,
    KernelRegistrationState,
    ProvenanceCollectionBuilder,
    record_kernel_registration,
)
from torch_spyre._inductor.provenance_writer import (
    CapturedUpstreamProjection,
    capture_upstream_projection,
    ProvenanceArtifactError,
    publish_provenance_collection,
    resolve_provenance_artifact_path,
    validate_provenance_document,
)
from torch_spyre._inductor.scheduler import SuperDSCScheduling
from torch_spyre._inductor.spyre_kernel import _codegen_op_spec_list
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner
from torch_spyre.provenance import (
    resolve_provenance_document,
    resolve_provenance_event,
)


_DEFAULT_PROVENANCE_ARTIFACT_PATH = spyre_config.provenance_artifact_path


def _handle(
    handle_id: int,
    *,
    aten_op: str | None = "aten.mm.default",
    source: SourceLoc | None = SourceLoc("/workspace/model.py", 117),
    fused_from: tuple[DebugHandle, ...] = (),
) -> DebugHandle:
    return DebugHandle(
        id=handle_id,
        source=source,
        aten_op=aten_op,
        ir_chain=(f"op{handle_id}",),
        fused_from=fused_from,
    )


def _op(
    handle: DebugHandle | None,
    *,
    op: str = "identity",
    iteration_space=None,
    args=(),
    op_info=None,
    tiled_symbols=None,
    tiled_symbol_trip_counts=None,
    symbolic_dim_bounds=None,
    node_output_ranges=None,
) -> OpSpec:
    return OpSpec(
        op=op,
        is_reduction=False,
        iteration_space={} if iteration_space is None else iteration_space,
        args=list(args),
        op_info={} if op_info is None else op_info,
        tiled_symbols=[] if tiled_symbols is None else tiled_symbols,
        tiled_symbol_trip_counts=(
            {} if tiled_symbol_trip_counts is None else tiled_symbol_trip_counts
        ),
        symbolic_dim_bounds=(
            {} if symbolic_dim_bounds is None else symbolic_dim_bounds
        ),
        node_output_ranges=node_output_ranges,
        debug_handle=handle,
    )


def _event_name(
    descriptor: KernelProvenanceDescriptor,
) -> str:
    return format_kernel_provenance_event_name(descriptor)


def _graph_lowering(**attributes):
    graph = object.__new__(GraphLowering)
    graph.__dict__.update(attributes)
    return graph


def _registration_state(
    *,
    has_graph_lowering: bool = True,
) -> KernelRegistrationState:
    return KernelRegistrationState(
        has_graph_lowering=has_graph_lowering,
        registrations={},
        capture_failed=False,
    )


def _publication_collection(
    *,
    kernel_name: str = "sdsc_fused_mm_0",
    handle_id: int = 9,
    has_graph_lowering: bool = True,
    registration_ordinal: int | None = None,
    include_uncollected: bool = False,
    upstream_projection_failed: bool = False,
):
    specs = [_op(_handle(handle_id))]
    descriptor = build_kernel_provenance_descriptor(specs)
    kernel = collect_kernel_provenance(kernel_name, specs, descriptor)
    builder = ProvenanceCollectionBuilder()
    builder.add_kernel(kernel)
    if include_uncollected:
        builder.add_uncollected_kernel(f"{kernel_name}_uncollected")

    if has_graph_lowering:
        graph = _graph_lowering()
        if registration_ordinal is not None:
            record_kernel_registration(graph, kernel_name, registration_ordinal)
        registration_state = consume_kernel_registration_state(graph)
    else:
        assert registration_ordinal is None
        registration_state = _registration_state(has_graph_lowering=False)
    if upstream_projection_failed:
        registration_state = dataclasses.replace(
            registration_state, capture_failed=True
        )
    collection = builder.finish(registration_state)
    assert collection is not None
    return collection


def _publish_collection_in_process(
    path: str,
    kernel_name: str,
    handle_id: int,
    registration_ordinal: int,
    start_event,
) -> None:
    """Publish after a delayed read to expose cross-process lost updates."""
    from torch_spyre._inductor import provenance_writer

    collection = _publication_collection(
        kernel_name=kernel_name,
        handle_id=handle_id,
        registration_ordinal=registration_ordinal,
    )
    read_existing_document = provenance_writer._read_existing_document

    def delayed_read(candidate_path):
        document = read_existing_document(candidate_path)
        time.sleep(0.25)
        return document

    start_event.wait()
    with (
        torch._inductor.config.patch("trace.provenance_tracking_level", 1),
        patch.object(
            provenance_writer,
            "_read_existing_document",
            side_effect=delayed_read,
        ),
        patch.object(provenance_writer, "trace_structured_artifact"),
    ):
        publish_provenance_collection(collection, path)


def _generated_wrapper_roundtrip(specs):
    """Serialize and reconstruct specs through the generated-wrapper seam."""

    def sympy_str(value):
        if isinstance(value, IndirectAccess):
            name_sym = value.args[0]
            return f"IndirectAccess('{name_sym}')"
        return "sympify('" + str(value) + "')"

    buf = IndentedBuffer()
    buf.writeline("[")
    with buf.indent():
        _codegen_op_spec_list(specs, buf, sympy_str)
    buf.writeline("]")
    namespace = {
        "DataFormats": DataFormats,
        "DebugHandle": DebugHandle,
        "ElementArrangement": ElementArrangement,
        "IndirectAccess": IndirectAccess,
        "LoopSpec": LoopSpec,
        "OpSpec": OpSpec,
        "SourceLoc": SourceLoc,
        "TensorArg": TensorArg,
        "sympify": sympify,
    }
    return eval(buf.getvalue(), namespace)  # noqa: S307


class TestKernelProvenanceDescriptor:
    def test_builds_bundle_identity_without_handles(self):
        specs = [
            _op(None),
            LoopSpec(count=Integer(2), body=[_op(None)]),
        ]

        descriptor = build_kernel_provenance_descriptor(specs)

        assert descriptor.debug_handle_ids == ()
        assert descriptor.aten_ops == ()
        assert _event_name(descriptor) == (
            f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_unknown_{descriptor.key}"
        )

    def test_collects_nested_handles_in_order_and_deduplicates_ids(self):
        first = _handle(9)
        second = _handle(12, aten_op="aten.add.Tensor")
        specs = [
            _op(first),
            LoopSpec(
                count=Integer(4),
                body=[
                    _op(second),
                    LoopSpec(count=Integer(2), body=[_op(first)]),
                ],
            ),
        ]

        descriptor = build_kernel_provenance_descriptor(specs)

        assert descriptor is not None
        assert descriptor.debug_handle_ids == ("9", "12")
        assert descriptor.aten_ops == ("aten.add.Tensor", "aten.mm.default")
        assert extract_kernel_provenance_key(_event_name(descriptor)) == descriptor.key
        assert not hasattr(descriptor, "fusion_context")

    def test_is_deterministic_and_order_sensitive(self):
        first = _handle(9)
        second = _handle(12)
        forward = build_kernel_provenance_descriptor([_op(first), _op(second)])
        independently_reconstructed = build_kernel_provenance_descriptor(
            [_op(_handle(9)), _op(_handle(12))]
        )
        reverse = build_kernel_provenance_descriptor([_op(second), _op(first)])

        assert independently_reconstructed == forward
        assert forward.key == "vsancadvtjfcq6cv"
        assert reverse.key != forward.key
        assert len(forward.key) == KERNEL_PROVENANCE_KEY_BASE32_WIDTH
        assert set(forward.key) <= set("abcdefghijklmnopqrstuvwxyz234567")

    def test_distinguishes_structures_with_the_same_handle_set(self):
        handle = _handle(9)
        flat = build_kernel_provenance_descriptor([_op(handle)])
        different_op = build_kernel_provenance_descriptor([_op(handle, op="add")])
        looped = build_kernel_provenance_descriptor(
            [LoopSpec(count=Integer(2), body=[_op(handle)])]
        )
        repeated_op = build_kernel_provenance_descriptor([_op(handle), _op(handle)])

        descriptors = (flat, different_op, looped, repeated_op)
        assert all(descriptor is not None for descriptor in descriptors)
        assert len({descriptor.key for descriptor in descriptors if descriptor}) == 4
        assert all(
            descriptor.debug_handle_ids == ("9",)
            for descriptor in descriptors
            if descriptor
        )

    def test_canonicalizes_real_tensor_args_and_unordered_metadata(self):
        handle = _handle(9)
        c0 = Symbol("c0")
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64],
            device_coordinates=[Integer(0), c0],
            allocation={"hbm": 0},
            name="arg0",
        )
        first = _op(
            handle,
            iteration_space={c0: (Integer(128), 1)},
            args=(arg,),
            op_info={"constants": {"alpha": 1.0, "beta": Integer(2)}},
        )
        reordered_metadata = _op(
            handle,
            iteration_space={c0: (Integer(128), 1)},
            args=(arg,),
            op_info={"constants": {"beta": Integer(2), "alpha": 1.0}},
        )
        changed_shape = dataclasses.replace(
            first,
            args=[dataclasses.replace(arg, device_size=[4, 64])],
        )
        changed_arrangement = dataclasses.replace(
            first,
            args=[
                dataclasses.replace(arg, element_arrangement=ElementArrangement.EXX2)
            ],
        )
        owned_arg = dataclasses.replace(
            arg,
            work_division=TensorWorkDivision({c0: 2}, {c0: Symbol("core_id")}),
        )
        changed_owner = dataclasses.replace(first, args=[owned_arg])
        changed_owner_cores = dataclasses.replace(
            first,
            args=[
                dataclasses.replace(
                    arg,
                    work_division=TensorWorkDivision(
                        {c0: 2}, {c0: Symbol("core_id")}, num_cores=4
                    ),
                )
            ],
        )
        changed_core_mapping = dataclasses.replace(
            first, core_id_to_work_slice={c0: Integer(1)}
        )
        canonical_core_mapping = dataclasses.replace(
            first, core_id_to_work_slice={c0: Integer(0)}
        )

        first_descriptor = build_kernel_provenance_descriptor([first])
        reordered_descriptor = build_kernel_provenance_descriptor([reordered_metadata])
        changed_descriptor = build_kernel_provenance_descriptor([changed_shape])
        changed_arrangement_descriptor = build_kernel_provenance_descriptor(
            [changed_arrangement]
        )
        changed_owner_descriptor = build_kernel_provenance_descriptor([changed_owner])
        changed_owner_cores_descriptor = build_kernel_provenance_descriptor(
            [changed_owner_cores]
        )
        changed_core_mapping_descriptor = build_kernel_provenance_descriptor(
            [changed_core_mapping]
        )
        canonical_core_mapping_descriptor = build_kernel_provenance_descriptor(
            [canonical_core_mapping]
        )

        assert first_descriptor is not None
        assert reordered_descriptor is not None
        assert changed_descriptor is not None
        assert changed_arrangement_descriptor is not None
        assert changed_owner_descriptor is not None
        assert reordered_descriptor.key == first_descriptor.key
        assert changed_descriptor.key != first_descriptor.key
        assert changed_arrangement_descriptor.key != first_descriptor.key
        assert changed_owner_descriptor.key != first_descriptor.key
        assert changed_owner_cores_descriptor.key != changed_owner_descriptor.key
        assert changed_core_mapping_descriptor.key != first_descriptor.key
        assert canonical_core_mapping_descriptor.key == first_descriptor.key

    def test_pins_rich_canonical_bundle_key(self):
        c0 = Symbol("c0")
        index = Symbol("index")
        constituent = _handle(8, aten_op="aten.permute.default")
        handle = _handle(9, fused_from=(constituent,))
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64],
            device_coordinates=[IndirectAccess(index), c0],
            allocation={"hbm_pool": {"offset": 4096}},
            element_arrangement=ElementArrangement.EXX2,
            name="arg0",
            device_tile_advance_expr=64 * c0,
        )
        spec = _op(
            handle,
            op="add",
            iteration_space={c0: (Integer(128), 2)},
            args=(arg,),
            op_info={
                "alpha": 1.5,
                "mask": b"\x00\xff",
                "expression": c0 + 1,
            },
            tiled_symbols=[[c0]],
            tiled_symbol_trip_counts={c0: 4},
            symbolic_dim_bounds={"s0": (128, 64)},
            node_output_ranges=(Integer(1), Integer(2), c0, Integer(64)),
        )

        descriptor = build_kernel_provenance_descriptor(
            [LoopSpec(count=Integer(4), body=[spec])]
        )

        assert descriptor.debug_handle_ids == ("9",)
        assert descriptor.aten_ops == (
            "aten.mm.default",
            "aten.permute.default",
        )
        assert descriptor.key == "2y3kqixggcejkcas"

    def test_generated_wrapper_roundtrip_reproduces_descriptor(self):
        constituent = _handle(8, aten_op="aten.permute.default")
        handle = _handle(9, fused_from=(constituent,))
        c0 = Symbol("c0")
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64],
            device_coordinates=[Integer(0), c0],
            allocation={"hbm": 0},
            element_arrangement=ElementArrangement.DL16_TO_FP32,
            name="arg0",
        )
        specs = [
            LoopSpec(
                count=Integer(2),
                body=[
                    _op(
                        handle,
                        op="add",
                        iteration_space={c0: (Integer(128), 1)},
                        args=(arg,),
                        op_info={"constants": {"alpha": 1.0}},
                        tiled_symbols=[[c0]],
                        tiled_symbol_trip_counts={c0: 2},
                        symbolic_dim_bounds={"s0": (128, 64)},
                        node_output_ranges=(Integer(1), Integer(2), c0),
                    )
                ],
            )
        ]

        original = build_kernel_provenance_descriptor(specs)
        reconstructed = build_kernel_provenance_descriptor(
            _generated_wrapper_roundtrip(specs)
        )

        assert original is not None
        assert reconstructed == original

    @pytest.mark.parametrize(
        "changed_schema", [OpSpec, TensorArg, TensorWorkDivision, LoopSpec]
    )
    def test_rejects_finalized_schema_drift(self, changed_schema):
        real_fields = dataclasses.fields

        def fields_with_future_field(schema):
            fields = real_fields(schema)
            if schema is changed_schema:
                return (
                    *fields,
                    types.SimpleNamespace(name="future_field", type="object"),
                )
            return fields

        with (
            patch(
                "torch_spyre._inductor.kernel_provenance.dataclasses.fields",
                side_effect=fields_with_future_field,
            ),
            pytest.raises(
                TypeError,
                match=rf"{changed_schema.__name__} schema changed.*future_field",
            ),
        ):
            build_kernel_provenance_descriptor([_op(_handle(1))])

    @pytest.mark.parametrize(
        ("changed_schema", "field_name"),
        [
            (OpSpec, "iteration_space"),
            (TensorArg, "device_coordinates"),
            (TensorWorkDivision, "work_slices"),
            (LoopSpec, "body"),
        ],
    )
    def test_rejects_finalized_schema_type_drift(self, changed_schema, field_name):
        real_fields = dataclasses.fields

        def fields_with_changed_type(schema):
            fields = real_fields(schema)
            if schema is not changed_schema:
                return fields
            return tuple(
                types.SimpleNamespace(
                    name=field.name,
                    type="future_type" if field.name == field_name else field.type,
                )
                for field in fields
            )

        with (
            patch(
                "torch_spyre._inductor.kernel_provenance.dataclasses.fields",
                side_effect=fields_with_changed_type,
            ),
            pytest.raises(
                TypeError,
                match=rf"{changed_schema.__name__} schema changed.*{field_name}",
            ),
        ):
            build_kernel_provenance_descriptor([_op(_handle(1))])

    def test_descriptor_is_frozen(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(1))])

        assert descriptor is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            descriptor.key = "a" * KERNEL_PROVENANCE_KEY_BASE32_WIDTH  # type: ignore[misc]


class TestKernelProvenanceEventName:
    def test_uses_aten_only_display_without_source_location(self):
        handle = _handle(
            42,
            source=SourceLoc("/private/workspace/model.py", 117),
        )

        descriptor = build_kernel_provenance_descriptor([_op(handle), _op(handle)])

        assert descriptor is not None
        assert (
            _event_name(descriptor)
            == f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_mm_{descriptor.key}"
        )
        assert "model" not in _event_name(descriptor)
        assert "117" not in _event_name(descriptor)

    def test_summarizes_conflicting_handles_without_choosing_a_primary(self):
        specs = [
            _op(_handle(1, source=SourceLoc("first.py", 10))),
            _op(
                _handle(
                    2,
                    aten_op="aten.add.Tensor",
                    source=SourceLoc("second.py", 20),
                )
            ),
        ]

        descriptor = build_kernel_provenance_descriptor(specs)

        assert descriptor is not None
        assert _event_name(descriptor) == (
            f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_add_mm_{descriptor.key}"
        )

    def test_uses_recursive_fused_constituents_for_display_only(self):
        constituent = _handle(1, source=SourceLoc("first.py", 10))
        fused = _handle(
            2,
            aten_op=None,
            source=None,
            fused_from=(constituent,),
        )

        descriptor = build_kernel_provenance_descriptor([_op(fused)])

        assert descriptor is not None
        assert descriptor.debug_handle_ids == ("2",)
        assert descriptor.aten_ops == ("aten.mm.default",)
        assert (
            _event_name(descriptor)
            == f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_mm_{descriptor.key}"
        )

    def test_uses_unknown_label_when_no_aten_name_exists(self):
        descriptor = build_kernel_provenance_descriptor(
            [_op(_handle(1, aten_op=None, source=None))]
        )

        assert descriptor is not None
        assert _event_name(descriptor) == (
            f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_unknown_{descriptor.key}"
        )

    def test_sanitizes_and_bounds_name_with_step_suffix_reservation(self):
        long_component = "α/" + "very-long-name." * 20
        handle = _handle(
            7,
            aten_op=f"aten.{long_component}.default",
            source=SourceLoc(f"/tmp/{long_component}.py", 123456),
        )

        descriptor = build_kernel_provenance_descriptor([_op(handle)])

        assert descriptor is not None
        assert _event_name(descriptor).isascii()
        # PR #2930 uses size_t for this JobPlan command index. Match the local
        # extension ABI instead of assuming that every target uses 64-bit size_t.
        size_t_bits = ctypes.sizeof(ctypes.c_size_t) * 8
        largest_step_suffix = f"#{(1 << size_t_bits) - 1}"
        final_name = f"{_event_name(descriptor)}{largest_step_suffix}"
        assert len(final_name.encode("ascii")) <= 127
        assert descriptor.key in final_name

    @pytest.mark.parametrize(
        "key",
        [
            "a" * (KERNEL_PROVENANCE_KEY_BASE32_WIDTH - 1),
            "a" * (KERNEL_PROVENANCE_KEY_BASE32_WIDTH + 1),
            "A" * KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
            "1" * KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
        ],
    )
    def test_rejects_noncanonical_descriptor_key(self, key):
        with pytest.raises(ValueError, match="not canonical lowercase base32"):
            KernelProvenanceDescriptor(key, (), ())

    def test_cpp_parser_literals_match_python_constants(self):
        """The C++ parser hardcodes these; see kernel_provenance_registry.cpp.

        A constant bump must update both sides in the same change.
        """
        assert KERNEL_PROVENANCE_KEY_VERSION == 1
        assert KERNEL_PROVENANCE_KEY_BASE32_WIDTH == 16

    @pytest.mark.parametrize(
        ("event_name", "expected"),
        [
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa", "a" * 16),
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa#17", "a" * 16),
            ("sdsc_mm_0", None),
            ("spyre_kernel_fused_mm_aaaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v2_fused_mm_aaaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v1_fused_mm_short", None),
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v1_fused_mmaaaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa#step", None),
            ("xspyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa", None),
        ],
    )
    def test_python_and_cpp_key_parsers_share_contract(self, event_name, expected):
        assert extract_kernel_provenance_key(event_name) == expected
        assert extract_kernel_provenance_key_cpp(event_name) == expected


class TestProvenanceArtifactCollection:
    def test_collects_valid_bundle_without_handles(self):
        specs = [
            _op(None),
            LoopSpec(count=Integer(2), body=[_op(None)]),
        ]
        descriptor = build_kernel_provenance_descriptor(specs)
        kernel = collect_kernel_provenance(
            "sdsc_fused_unknown_0",
            specs,
            descriptor,
        )
        builder = ProvenanceCollectionBuilder()
        builder.add_kernel(kernel)

        collection = builder.finish(_registration_state())

        assert collection is not None
        assert collection.handles == {}
        assert kernel.identity.direct_handle_ids == ()
        assert kernel.identity.spec_handle_bindings == ()
        assert kernel.identity.aten_ops == ()
        assert kernel.identity.event_name_base == (
            f"spyre_kernel_v1_fused_unknown_{descriptor.key}"
        )
        occurrence = next(iter(collection.kernel_occurrences.values()))
        assert occurrence.identity_key == descriptor.key
        assert occurrence.registrations == ()

    def test_normalizes_nested_handles_and_preserves_compiler_order(self):
        linear = _handle(101, aten_op="aten.linear.default")
        relu = _handle(102, aten_op="aten.relu.default")
        fused = DebugHandle(
            id=200,
            source=None,
            aten_op=None,
            ir_chain=("linear", "relu", "buf0"),
            fused_from=(linear, relu),
            transform_history=(
                ProvenanceTransform(
                    kind="fusion",
                    pass_name="fuse_linear_relu",
                    reason="same iteration space",
                ),
            ),
        )
        specs = [
            _op(fused),
            LoopSpec(
                count=Integer(2),
                body=[
                    _op(fused),
                    LoopSpec(count=Integer(3), body=[_op(None)]),
                ],
            ),
        ]
        descriptor = build_kernel_provenance_descriptor(specs)

        kernel = collect_kernel_provenance(
            "sdsc_fused_linear_relu_0",
            specs,
            descriptor,
        )

        assert kernel.identity.direct_handle_ids == ("200",)
        assert [
            binding.spec_path for binding in kernel.identity.spec_handle_bindings
        ] == [(0,), (1, 0)]
        assert list(kernel.handles) == ["101", "102", "200"]
        assert kernel.handles["200"].to_dict() == {
            "id": "200",
            "source": None,
            "aten_op": None,
            "ir_chain": ["linear", "relu", "buf0"],
            "fused_from": ["101", "102"],
            "transform_history": [
                {
                    "kind": "fusion",
                    "pass_name": "fuse_linear_relu",
                    "reason": "same iteration space",
                }
            ],
        }
        assert kernel.identity.to_dict()["eventNameBase"] == (
            f"spyre_kernel_v1_fused_linear_relu_{descriptor.key}"
        )

    def test_assigns_real_compile_and_occurrence_ids_and_deduplicates_identity(self):
        specs = [_op(_handle(9))]
        descriptor = build_kernel_provenance_descriptor(specs)
        first = collect_kernel_provenance("sdsc_fused_mm_0", specs, descriptor)
        second = collect_kernel_provenance("sdsc_fused_mm_1", specs, descriptor)
        builder = ProvenanceCollectionBuilder()
        builder.add_kernel(first)
        builder.add_kernel(second)
        graph = _graph_lowering()
        record_kernel_registration(graph, "sdsc_fused_mm_0", 3)
        record_kernel_registration(graph, "sdsc_fused_mm_0", 7)

        collection = builder.finish(consume_kernel_registration_state(graph))

        assert collection is not None
        expected_kernels = [
            ["sdsc_fused_mm_0", descriptor.key],
            ["sdsc_fused_mm_1", descriptor.key],
        ]
        expected_compile_id = _canonical_digest(
            {
                "domain": "torch-spyre-compile-v1",
                "kernels": expected_kernels,
            }
        )
        assert collection.compile_id == expected_compile_id
        assert collection.kernels == tuple(tuple(pair) for pair in expected_kernels)
        assert list(collection.kernel_identities) == [descriptor.key]
        assert len(collection.kernel_occurrences) == 2
        first_occurrence_id = _canonical_digest(
            {
                "domain": "torch-spyre-occurrence-v1",
                "compileId": expected_compile_id,
                "compilerKernelName": "sdsc_fused_mm_0",
                "identityKey": descriptor.key,
            }
        )
        first_occurrence = collection.kernel_occurrences[first_occurrence_id]
        assert [
            registration.to_dict() for registration in first_occurrence.registrations
        ] == [
            {"ordinal": 3, "alias": "sdsc_fused_mm_0:3"},
            {"ordinal": 7, "alias": "sdsc_fused_mm_0:7"},
        ]

    def test_cache_replay_keeps_identity_with_empty_registrations(self):
        specs = [_op(_handle(9))]
        descriptor = build_kernel_provenance_descriptor(specs)
        kernel = collect_kernel_provenance("sdsc_fused_mm_0", specs, descriptor)
        fresh_builder = ProvenanceCollectionBuilder()
        fresh_builder.add_kernel(kernel)
        replay_builder = ProvenanceCollectionBuilder()
        replay_builder.add_kernel(kernel)
        graph = _graph_lowering()
        record_kernel_registration(graph, "sdsc_fused_mm_0", 1)

        fresh = fresh_builder.finish(consume_kernel_registration_state(graph))
        replay = replay_builder.finish(_registration_state(has_graph_lowering=False))

        assert fresh is not None
        assert replay is not None
        assert fresh.has_graph_lowering
        assert not replay.has_graph_lowering
        assert replay.compile_id == fresh.compile_id
        assert replay.handles == fresh.handles
        assert replay.kernel_identities == fresh.kernel_identities
        assert replay.kernel_occurrences.keys() == fresh.kernel_occurrences.keys()
        replay_occurrence = next(iter(replay.kernel_occurrences.values()))
        assert replay_occurrence.registrations == ()

    def test_no_graph_state_is_explicit_and_rejects_registration(self):
        registration_state = consume_kernel_registration_state(V.graph)

        assert not registration_state.has_graph_lowering
        assert registration_state.registrations == {}
        with pytest.raises(
            TypeError,
            match="requires a real GraphLowering",
        ):
            record_kernel_registration(V.graph, "sdsc_fused_mm_0", 1)

    def test_rejects_conflicting_content_for_one_handle_id(self):
        first = _handle(101, source=SourceLoc("first.py", 10))
        conflicting = _handle(101, source=SourceLoc("second.py", 20))
        fused = _handle(200, fused_from=(first, conflicting))
        specs = [_op(fused)]
        descriptor = build_kernel_provenance_descriptor(specs)

        with pytest.raises(
            ValueError,
            match="conflicting content for debug handle ID 101",
        ):
            collect_kernel_provenance("sdsc_fused_mm_0", specs, descriptor)

    def test_rejects_unsupported_finalized_spec(self):
        descriptor = build_kernel_provenance_descriptor([])

        with pytest.raises(TypeError, match="Unsupported finalized kernel spec"):
            collect_kernel_provenance(
                "sdsc_unknown_0",
                [object()],  # type: ignore[list-item]
                descriptor,
            )

    def test_scheduler_registers_once_and_retains_repeated_exact_aliases(self):
        comments = []
        definitions = []
        wrapper = types.SimpleNamespace(
            src_to_kernel={},
            next_kernel_suffix=lambda: "0",
            define_kernel=lambda *args: definitions.append(args),
            write_provenance_debug_handle=lambda name, handle: comments.append(
                (name, handle)
            ),
        )
        graph = _graph_lowering(wrapper_code=wrapper)
        scheduling = object.__new__(SuperDSCScheduling)
        node_schedule = [object()]
        kernel = types.SimpleNamespace(
            _kernel_uses_hbm_pool=lambda: False,
            pool_size=0,
        )

        with (
            V.set_graph_handler(graph),
            patch(
                "torch_spyre._inductor.scheduler.get_fused_kernel_name",
                return_value="fused_mm",
            ),
            patch(
                "torch_spyre._inductor.scheduler.get_kernel_metadata",
                return_value=("origins", "detailed origins"),
            ),
            patch(
                "torch._inductor.debug.set_kernel_post_grad_provenance_tracing",
                side_effect=(4, 9),
            ) as register,
        ):
            first_name = scheduling.define_kernel("[]", node_schedule, kernel)
            scheduling.codegen_comment(node_schedule, first_name)
            second_name = scheduling.define_kernel("[]", node_schedule, kernel)
            scheduling.codegen_comment(node_schedule, second_name)

        registrations = consume_kernel_registration_state(graph).registrations
        assert first_name == second_name == "sdsc_fused_mm_0"
        assert len(definitions) == 1
        assert register.call_count == 2
        assert comments == [
            ("sdsc_fused_mm_0", 4),
            ("sdsc_fused_mm_0", 9),
        ]
        assert [
            registration.to_dict() for registration in registrations["sdsc_fused_mm_0"]
        ] == [
            {"ordinal": 4, "alias": "sdsc_fused_mm_0:4"},
            {"ordinal": 9, "alias": "sdsc_fused_mm_0:9"},
        ]


class TestKernelProvenancePropagation:
    def test_async_compile_builds_descriptor_from_finalized_specs(self):
        specs = [
            _op(_handle(9)),
            LoopSpec(count=Integer(2), body=[_op(_handle(12))]),
        ]
        runner = object()

        with (
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ) as runner_type,
        ):
            result = SpyreAsyncCompile().sdsc("sdsc_fused_mm_0", specs)

        assert result is runner
        descriptor = runner_type.call_args.kwargs["kernel_provenance"]
        assert descriptor.debug_handle_ids == ("9", "12")
        assert extract_kernel_provenance_key(_event_name(descriptor)) == descriptor.key

    def test_async_compile_finalizes_graph_scoped_collection_at_wait(self):
        specs = [
            _op(_handle(9)),
            LoopSpec(count=Integer(2), body=[_op(_handle(12))]),
        ]
        runner = object()
        compiler = SpyreAsyncCompile()
        graph = _graph_lowering()

        with (
            V.set_graph_handler(graph),
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ),
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait") as base_wait,
        ):
            result = compiler.sdsc("sdsc_fused_mm_0", specs)
            record_kernel_registration(graph, "sdsc_fused_mm_0", 6)
            compiler.wait({})

        assert result is runner
        base_wait.assert_called_once_with({})
        collection = compiler._last_provenance_collection
        assert collection is not None
        assert list(collection.handles) == ["12", "9"]
        assert len(collection.kernel_identities) == 1
        occurrence = next(iter(collection.kernel_occurrences.values()))
        assert occurrence.compiler_kernel_name == "sdsc_fused_mm_0"
        assert [
            registration.to_dict() for registration in occurrence.registrations
        ] == [{"ordinal": 6, "alias": "sdsc_fused_mm_0:6"}]
        assert compiler._artifact_collection_failure_count == 0

    def test_artifact_collection_failure_preserves_phase3a_descriptor(self):
        specs = [_op(_handle(9))]
        runner = object()
        compiler = SpyreAsyncCompile()
        graph = _graph_lowering()

        with (
            V.set_graph_handler(graph),
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.collect_kernel_provenance",
                side_effect=RuntimeError("artifact-only failure"),
            ),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ) as runner_type,
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait"),
            patch("torch_spyre.execution.async_compile.logger.warning") as warning,
        ):
            result = compiler.sdsc("sdsc_fused_mm_0", specs)
            compiler.wait({})

        descriptor = runner_type.call_args.kwargs["kernel_provenance"]
        assert result is runner
        assert descriptor is not None
        assert descriptor.debug_handle_ids == ("9",)
        assert extract_kernel_provenance_key(_event_name(descriptor)) == descriptor.key
        collection = compiler._last_provenance_collection
        assert collection is not None
        assert collection.kernel_identities == {}
        assert collection.collection_failure_count == 1
        assert collection.uncollected_kernels[0].compiler_kernel_name == (
            "sdsc_fused_mm_0"
        )
        assert warning.call_count == 2
        assert warning.call_args_list[0].kwargs["exc_info"] is True
        assert warning.call_args_list[1].args == (
            "provenance artifact collection incomplete after %d failure(s) "
            "across %d compiled Spyre kernels",
            1,
            1,
        )

    def test_unimplemented_kernel_does_not_discard_successful_collection(self):
        valid_specs = [_op(_handle(9))]
        unimplemented_specs = [UnimplementedOp("future_op")]
        valid_runner = object()
        unimplemented_runner = object()
        compiler = SpyreAsyncCompile()
        graph = _graph_lowering()

        with (
            V.set_graph_handler(graph),
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=valid_runner,
            ),
            patch(
                "torch_spyre.execution.async_compile.SpyreUnimplementedRunner",
                return_value=unimplemented_runner,
            ),
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait"),
        ):
            assert compiler.sdsc("sdsc_fused_mm_0", valid_specs) is valid_runner
            assert (
                compiler.sdsc("sdsc_future_op_1", unimplemented_specs)
                is unimplemented_runner
            )
            record_kernel_registration(graph, "sdsc_fused_mm_0", 1)
            record_kernel_registration(graph, "sdsc_future_op_1", 2)
            compiler.wait({})

        collection = compiler._last_provenance_collection
        assert collection is not None
        assert len(collection.kernel_identities) == 1
        assert collection.collection_failure_count == 1
        assert [
            registration.alias
            for registration in collection.uncollected_kernels[0].registrations
        ] == ["sdsc_future_op_1:2"]
        occurrence = next(iter(collection.kernel_occurrences.values()))
        assert [registration.alias for registration in occurrence.registrations] == [
            "sdsc_fused_mm_0:1"
        ]

    def test_descriptor_failure_does_not_discard_successful_collection(self):
        valid_specs = [_op(_handle(9))]
        invalid_specs = [_op(_handle(12), op_info={"future_value": object()})]
        runner = object()
        compiler = SpyreAsyncCompile()
        graph = _graph_lowering()

        with (
            V.set_graph_handler(graph),
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ),
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait"),
        ):
            assert compiler.sdsc("sdsc_fused_mm_0", valid_specs) is runner
            assert compiler.sdsc("sdsc_invalid_1", invalid_specs) is runner
            record_kernel_registration(graph, "sdsc_fused_mm_0", 1)
            record_kernel_registration(graph, "sdsc_invalid_1", 2)
            compiler.wait({})

        collection = compiler._last_provenance_collection
        assert collection is not None
        assert len(collection.kernel_identities) == 1
        assert collection.collection_failure_count == 1
        assert collection.uncollected_kernels[0].compiler_kernel_name == (
            "sdsc_invalid_1"
        )
        assert [
            registration.alias
            for registration in collection.uncollected_kernels[0].registrations
        ] == ["sdsc_invalid_1:2"]
        occurrence = next(iter(collection.kernel_occurrences.values()))
        assert occurrence.compiler_kernel_name == "sdsc_fused_mm_0"

    def test_async_compile_keeps_execution_on_unknown_bundle_value(self):
        specs = [_op(_handle(9), op_info={"future_value": object()})]
        runner = object()

        with (
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ) as runner_type,
            patch("torch_spyre.execution.async_compile.logger.warning") as warning,
        ):
            result = SpyreAsyncCompile().sdsc("sdsc_fused_mm_0", specs)

        assert result is runner
        assert runner_type.call_args.kwargs["kernel_provenance"] is None
        warning.assert_called_once()
        assert "continuing without kernel provenance" in warning.call_args.args[0]

    def test_async_compile_aggregates_provenance_failures(self):
        specs = [_op(_handle(9), op_info={"future_value": object()})]
        runner = object()
        compiler = SpyreAsyncCompile()

        with (
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ),
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait") as base_wait,
            patch("torch_spyre.execution.async_compile.logger.warning") as warning,
        ):
            assert compiler.sdsc("sdsc_fused_mm_0", specs) is runner
            assert compiler.sdsc("sdsc_fused_mm_1", specs) is runner
            compiler.wait({})

        base_wait.assert_called_once_with({})
        assert warning.call_count == 2
        assert warning.call_args_list[0].kwargs["exc_info"] is True
        assert warning.call_args_list[1].args == (
            "kernel provenance disabled for %d/%d compiled Spyre kernels",
            2,
            2,
        )
        assert compiler._provenance_attempt_count == 0
        assert compiler._provenance_failure_count == 0

    def test_runner_retains_descriptor_for_runtime_forwarding(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(9))])
        assert descriptor is not None

        with (
            patch(
                "torch_spyre.execution.kernel_runner.register_kernel_provenance",
                return_value=True,
            ) as register_kernel_provenance,
            patch(
                "torch_spyre.execution.kernel_runner.prepare_kernel",
                return_value="jobplan",
            ) as prepare_kernel,
        ):
            runner = SpyreSDSCKernelRunner(
                "sdsc_fused_mm_0",
                "/tmp/kernel",
                kernel_provenance=descriptor,
            )

        assert runner.kernel_provenance is descriptor
        assert runner.profiler_event_name == _event_name(descriptor)
        assert runner.jobplan == "jobplan"
        register_kernel_provenance.assert_called_once_with(
            _event_name(descriptor), list(descriptor.debug_handle_ids)
        )
        prepare_kernel.assert_called_once_with(
            "/tmp/kernel/spyreCodeDir",
            profiler_name=_event_name(descriptor),
        )

    def test_runner_keeps_event_name_when_native_registration_fails(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(9))])
        assert descriptor is not None

        with (
            patch(
                "torch_spyre.execution.kernel_runner.register_kernel_provenance",
                side_effect=RuntimeError("registry failed"),
            ) as register_kernel_provenance,
            patch(
                "torch_spyre.execution.kernel_runner.prepare_kernel",
                return_value="jobplan",
            ) as prepare_kernel,
            patch("torch_spyre.execution.kernel_runner.logger.warning") as warning,
        ):
            runner = SpyreSDSCKernelRunner(
                "sdsc_fused_mm_0",
                "/tmp/kernel",
                kernel_provenance=descriptor,
            )

        assert runner.kernel_provenance is descriptor
        assert runner.profiler_event_name == _event_name(descriptor)
        assert runner.jobplan == "jobplan"
        register_kernel_provenance.assert_called_once_with(
            _event_name(descriptor), list(descriptor.debug_handle_ids)
        )
        prepare_kernel.assert_called_once_with(
            "/tmp/kernel/spyreCodeDir",
            profiler_name=_event_name(descriptor),
        )
        warning.assert_called_once()
        assert warning.call_args.args[1] == "sdsc_fused_mm_0"
        assert warning.call_args.kwargs["exc_info"] is True

    def test_runner_uses_legacy_prepare_when_event_formatting_fails(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(9))])
        assert descriptor is not None

        with (
            patch(
                "torch_spyre.execution.kernel_runner."
                "format_kernel_provenance_event_name",
                side_effect=RuntimeError("formatting failed"),
            ),
            patch(
                "torch_spyre.execution.kernel_runner.register_kernel_provenance"
            ) as register_kernel_provenance,
            patch(
                "torch_spyre.execution.kernel_runner.prepare_kernel",
                return_value="jobplan",
            ) as prepare_kernel,
            patch("torch_spyre.execution.kernel_runner.logger.warning") as warning,
        ):
            runner = SpyreSDSCKernelRunner(
                "sdsc_fused_mm_0",
                "/tmp/kernel",
                kernel_provenance=descriptor,
            )

        assert runner.kernel_provenance is descriptor
        assert runner.profiler_event_name is None
        assert runner.jobplan == "jobplan"
        register_kernel_provenance.assert_not_called()
        prepare_kernel.assert_called_once_with("/tmp/kernel/spyreCodeDir")
        warning.assert_called_once()
        assert warning.call_args.args[1] == "sdsc_fused_mm_0"
        assert warning.call_args.kwargs["exc_info"] is True

    def test_runner_preserves_legacy_prepare_call_without_descriptor(self):
        with (
            patch(
                "torch_spyre.execution.kernel_runner.register_kernel_provenance"
            ) as register_kernel_provenance,
            patch(
                "torch_spyre.execution.kernel_runner.prepare_kernel",
                return_value="jobplan",
            ) as prepare_kernel,
        ):
            runner = SpyreSDSCKernelRunner("sdsc_fused_mm_0", "/tmp/kernel")

        assert runner.kernel_provenance is None
        assert runner.profiler_event_name is None
        prepare_kernel.assert_called_once_with("/tmp/kernel/spyreCodeDir")
        register_kernel_provenance.assert_not_called()


_SCHEMA = json.loads(
    files("torch_spyre._inductor.schemas")
    .joinpath("spyre_provenance_v1.schema.json")
    .read_text(encoding="utf-8")
)
_SCHEMA_VALIDATOR_TYPE = validator_for(_SCHEMA)
_SCHEMA_VALIDATOR_TYPE.check_schema(_SCHEMA)
_SCHEMA_VALIDATOR = _SCHEMA_VALIDATOR_TYPE(_SCHEMA)
_PROVENANCE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "provenance"


def _load_provenance_fixture(name: str) -> dict:
    return json.loads((_PROVENANCE_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _fixture_compile_id(projection: dict) -> str:
    kernels = [
        [kernel["compilerKernelName"], kernel["identityKey"]]
        for kernel in projection["kernels"]
    ]
    return _canonical_digest(
        {
            "domain": "torch-spyre-compile-v1",
            "kernels": kernels,
        }
    )


def _fixture_occurrence_id(occurrence: dict) -> str:
    return _canonical_digest(
        {
            "domain": "torch-spyre-occurrence-v1",
            "compileId": occurrence["compileId"],
            "compilerKernelName": occurrence["compilerKernelName"],
            "identityKey": occurrence["identityKey"],
        }
    )


def _recursive_fixture_atens(handle_ids: list[str], handles: dict) -> list[str]:
    names: set[str] = set()
    visited: set[str] = set()

    def visit(handle_id: str) -> None:
        assert handle_id in handles
        if handle_id in visited:
            return
        visited.add(handle_id)
        handle = handles[handle_id]
        if handle["aten_op"] is not None:
            names.add(handle["aten_op"])
        for constituent_id in handle["fused_from"]:
            visit(constituent_id)

    for handle_id in handle_ids:
        visit(handle_id)
    return sorted(names)


def _deduplicate_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _assert_sorted_keys(mapping: dict) -> None:
    assert list(mapping) == sorted(mapping)


def _validate_fixture_semantics(document: dict) -> None:
    handles = document["handles"]
    identities = document["kernelIdentities"]
    occurrences = document["kernelOccurrences"]
    projections = document["upstreamProjections"]

    for mapping in (
        document["diagnostics"],
        handles,
        identities,
        occurrences,
        projections,
    ):
        _assert_sorted_keys(mapping)

    for handle_id, handle in handles.items():
        assert handle["id"] == handle_id
        assert all(child_id in handles for child_id in handle["fused_from"])

    for identity_key, identity in identities.items():
        binding_ids = [
            binding["handleId"] for binding in identity["specHandleBindings"]
        ]
        assert all(handle_id in handles for handle_id in binding_ids)
        assert identity["directHandleIds"] == _deduplicate_in_order(binding_ids)
        assert identity["atenOps"] == _recursive_fixture_atens(
            identity["directHandleIds"], handles
        )
        descriptor = KernelProvenanceDescriptor(
            key=identity_key,
            debug_handle_ids=tuple(identity["directHandleIds"]),
            aten_ops=tuple(identity["atenOps"]),
        )
        assert (
            format_kernel_provenance_event_name(descriptor) == identity["eventNameBase"]
        )

    occurrences_by_compile: dict[str, list[tuple[str, dict]]] = {}
    for occurrence_id, occurrence in occurrences.items():
        assert occurrence_id == _fixture_occurrence_id(occurrence)
        assert occurrence["identityKey"] in identities
        assert occurrence["compileId"] in projections
        ordinals = [
            registration["ordinal"] for registration in occurrence["registrations"]
        ]
        assert ordinals == sorted(ordinals)
        assert len(ordinals) == len(set(ordinals))
        for registration in occurrence["registrations"]:
            assert registration["alias"] == (
                f"{occurrence['compilerKernelName']}:{registration['ordinal']}"
            )
        occurrences_by_compile.setdefault(occurrence["compileId"], []).append(
            (occurrence_id, occurrence)
        )

    for compile_id, projection in projections.items():
        assert compile_id == _fixture_compile_id(projection)
        kernels = [
            (kernel["compilerKernelName"], kernel["identityKey"])
            for kernel in projection["kernels"]
        ]
        assert len(kernels) == len(set(kernels))
        assert all(identity_key in identities for _, identity_key in kernels)
        uncollected_kernels = projection["uncollectedKernels"]
        assert uncollected_kernels == sorted(set(uncollected_kernels))
        assert all(uncollected_kernels)
        assert not set(uncollected_kernels) & {name for name, _ in kernels}
        assert isinstance(projection["upstreamProjectionFailed"], bool)

        compile_occurrences = occurrences_by_compile.get(compile_id, [])
        occurrence_kernels = {
            (
                occurrence["compilerKernelName"],
                occurrence["identityKey"],
            )
            for _, occurrence in compile_occurrences
        }
        assert occurrence_kernels == set(kernels)

        registrations = [
            registration
            for _, occurrence in compile_occurrences
            for registration in occurrence["registrations"]
        ]
        ordinals = [registration["ordinal"] for registration in registrations]
        assert len(ordinals) == len(set(ordinals))
        registration_aliases = {registration["alias"] for registration in registrations}

        for relation_name in (
            "preToPost",
            "postToPre",
            "cppCodeToPost",
            "postToCppCode",
            "kernelStackTraces",
        ):
            _assert_sorted_keys(projection[relation_name])

        cpp_to_post = projection["cppCodeToPost"]
        assert set(cpp_to_post) <= registration_aliases
        if projection["upstreamJoin"] == "ok":
            assert set(cpp_to_post) == registration_aliases
            assert set(projection["kernelStackTraces"]) == registration_aliases

        expected_post_to_cpp: dict[str, list[str]] = {}
        for alias in sorted(cpp_to_post):
            for post_node in cpp_to_post[alias]:
                expected_post_to_cpp.setdefault(post_node, []).append(alias)
        assert projection["postToCppCode"] == expected_post_to_cpp

        expected_post_to_pre: dict[str, list[str]] = {}
        for pre_node, post_nodes in projection["preToPost"].items():
            for post_node in post_nodes:
                expected_post_to_pre.setdefault(post_node, []).append(pre_node)
        assert projection["postToPre"] == expected_post_to_pre

    assert set(occurrences_by_compile) == set(projections)
    expected_diagnostics = {}
    collection_failure_count = sum(
        len(projection["uncollectedKernels"]) for projection in projections.values()
    )
    upstream_projection_failure_count = sum(
        projection["upstreamProjectionFailed"] for projection in projections.values()
    )
    if collection_failure_count:
        expected_diagnostics["collection-failure"] = collection_failure_count
    if upstream_projection_failure_count:
        expected_diagnostics["upstream-projection-failure"] = (
            upstream_projection_failure_count
        )
    assert document["diagnostics"] == expected_diagnostics
    expected_status = (
        "complete"
        if not document["diagnostics"]
        and all(
            projection["upstreamJoin"] == "ok" for projection in projections.values()
        )
        else "partial"
    )
    assert document["status"] == expected_status


def _apply_fixture_mutation(document: dict, mutation: dict) -> dict:
    mutated = copy.deepcopy(document)
    tokens = [
        token.replace("~1", "/").replace("~0", "~")
        for token in mutation["path"].lstrip("/").split("/")
    ]
    parent: object = mutated
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    final = tokens[-1]
    if isinstance(parent, list):
        parent[int(final)] = mutation["value"]
    else:
        assert isinstance(parent, dict)
        parent[final] = mutation["value"]
    return mutated


def _fixture_reader_diagnostic(document: dict) -> str | None:
    if document.get("schemaVersion") != _SCHEMA["properties"]["schemaVersion"]["const"]:
        return "unsupported-schema-version"
    try:
        _SCHEMA_VALIDATOR.validate(document)
    except ValidationError:
        return "schema-validation-failure"
    try:
        _validate_fixture_semantics(document)
    except AssertionError:
        return "semantic-validation-failure"
    return None


class TestProvenanceArtifactSchema:
    def test_valid_fixtures_pass_schema_and_semantic_validation(self):
        manifest = _load_provenance_fixture("fixture_manifest.json")

        for fixture in manifest["valid"]:
            document = _load_provenance_fixture(fixture["path"])
            _SCHEMA_VALIDATOR.validate(document)
            _validate_fixture_semantics(document)
            assert _fixture_reader_diagnostic(document) is None

    def test_invalid_fixture_mutations_report_declared_reader_diagnostic(self):
        manifest = _load_provenance_fixture("fixture_manifest.json")

        for fixture in manifest["invalid"]:
            document = _load_provenance_fixture(fixture["base"])
            mutated = _apply_fixture_mutation(document, fixture["mutation"])
            assert (
                _fixture_reader_diagnostic(mutated)
                == fixture["expectedReaderDiagnostic"]
            ), fixture["name"]

    def test_schema_event_contract_matches_production_codec(self):
        event_key_schema = _SCHEMA["properties"]["eventKey"]["$ref"]
        assert event_key_schema == "#/$defs/eventKey"
        assert (
            _SCHEMA["$defs"]["eventKey"]["properties"]["version"]["const"]
            == KERNEL_PROVENANCE_KEY_VERSION
        )
        assert (
            _SCHEMA["$defs"]["eventKey"]["properties"]["width"]["const"]
            == KERNEL_PROVENANCE_KEY_BASE32_WIDTH
        )
        assert _SCHEMA["$defs"]["eventKeyValue"]["pattern"] == (
            rf"^[a-z2-7]{{{KERNEL_PROVENANCE_KEY_BASE32_WIDTH}}}$"
        )

        document = _load_provenance_fixture("valid_v1.json")
        event_pattern = _SCHEMA["$defs"]["kernelIdentity"]["properties"][
            "eventNameBase"
        ]["pattern"]
        assert all(
            re.fullmatch(event_pattern, identity["eventNameBase"])
            for identity in document["kernelIdentities"].values()
        )

    def test_cache_replay_reuses_compile_and_occurrence_ids(self):
        fresh = _load_provenance_fixture("valid_v1.json")
        cached = _load_provenance_fixture("valid_cache_replay_v1.json")
        compile_id = "3acd1120e2323fa1fec28f1ff89a59da68cda82541b0a19815ec47a2c922ac0e"

        assert compile_id in fresh["upstreamProjections"]
        assert compile_id in cached["upstreamProjections"]
        cached_occurrence_ids = set(cached["kernelOccurrences"])
        assert cached_occurrence_ids <= set(fresh["kernelOccurrences"])
        for occurrence_id in cached_occurrence_ids:
            cached_occurrence = cached["kernelOccurrences"][occurrence_id]
            fresh_occurrence = fresh["kernelOccurrences"][occurrence_id]
            assert cached_occurrence["compileId"] == fresh_occurrence["compileId"]
            assert cached_occurrence["identityKey"] == fresh_occurrence["identityKey"]
            assert (
                cached_occurrence["compilerKernelName"]
                == (fresh_occurrence["compilerKernelName"])
            )
            assert cached_occurrence["registrations"] == []

        assert fresh["upstreamProjections"][compile_id]["upstreamJoin"] == "ok"
        assert cached["upstreamProjections"][compile_id]["upstreamJoin"] == (
            "unavailable-cache-replay"
        )

    def test_reserved_occurrence_selector_is_an_opaque_exact_match_token(self):
        document = _load_provenance_fixture("valid_cache_replay_v1.json")
        occurrence = next(iter(document["kernelOccurrences"].values()))
        occurrence["selector"] = "future-event-occurrence-token"
        _SCHEMA_VALIDATOR.validate(document)
        _validate_fixture_semantics(document)

    def test_shipped_package_avoids_internal_project_labels(self):
        package_root = Path(__file__).parents[2] / "torch_spyre"
        internal_label = re.compile(
            r"\b(?:phase\s+(?:[0-9]+[a-z]|x)|milestone\s+[0-9]+|internship)\b",
            re.IGNORECASE,
        )
        source_suffixes = {".cpp", ".h", ".json", ".py", ".pyi"}
        offenders = {}

        for path in package_root.rglob("*"):
            if path.suffix not in source_suffixes:
                continue
            matches = internal_label.findall(path.read_text(encoding="utf-8"))
            if matches:
                offenders[str(path.relative_to(package_root))] = matches

        assert offenders == {}


class TestProvenanceArtifactResolver:
    _EVENT_BASE = "spyre_kernel_v1_fused_linear_relu_atqydvnuutl766na"

    def test_production_codec_preserves_command_step_suffix(self):
        parsed = parse_kernel_provenance_event_name(f"{self._EVENT_BASE}#007")

        assert parsed is not None
        assert parsed.base_name == self._EVENT_BASE
        assert parsed.key == "atqydvnuutl766na"
        assert parsed.step == 7
        assert parsed.step_suffix == "#007"
        assert extract_kernel_provenance_key(parsed.name) == parsed.key

    def test_resolves_lineage_and_reports_occurrence_ambiguity(self):
        document = _load_provenance_fixture("valid_v1.json")

        result = resolve_provenance_document(f"{self._EVENT_BASE}#3", document)

        assert result["status"] == "partial"
        assert result["event"]["commandStep"] == 3
        assert result["event"]["commandStepSuffix"] == "#3"
        assert result["directHandleIds"] == ["200"]
        assert result["fusedConstituentIds"] == ["101", "102"]
        assert set(result["handles"]) == {"101", "102", "200"}
        assert result["handles"]["101"]["source"]["start_line"] == 17
        assert result["handles"]["102"]["aten_op"] == "aten.relu.default"
        assert result["handles"]["200"]["transform_history"][0]["kind"] == ("fusion")
        assert len(result["occurrences"]) == 2
        assert result["occurrenceSummary"]["ambiguous"] is True
        fields = result["occurrenceSummary"]["fields"]
        assert fields["identityKey"] == {
            "ambiguous": False,
            "value": "atqydvnuutl766na",
        }
        assert fields["compileId"]["ambiguous"] is True
        assert {diagnostic["code"] for diagnostic in result["diagnostics"]} == {
            "ambiguity",
            "incomplete-artifact",
            "upstream-join-status",
        }

    def test_occurrence_selector_is_an_exact_filter(self):
        document = _load_provenance_fixture("valid_v1.json")
        for occurrence in document["kernelOccurrences"].values():
            if occurrence["identityKey"] != "atqydvnuutl766na":
                continue
            occurrence["selector"] = (
                "fresh"
                if occurrence["compilerKernelName"].endswith("_0")
                else "level-zero"
            )

        result = resolve_provenance_document(
            self._EVENT_BASE,
            document,
            occurrence_selector="fresh",
        )

        assert result["event"]["occurrenceSelector"] == "fresh"
        assert len(result["occurrences"]) == 1
        assert result["occurrences"][0]["selector"] == "fresh"
        assert result["occurrenceSummary"]["ambiguous"] is False

    def test_empty_handle_identity_resolves_without_fallback(self):
        document = _load_provenance_fixture("valid_v1.json")
        event_name = "spyre_kernel_v1_fused_unknown_vsancadvtjfcq6cv#0"

        result = resolve_provenance_document(event_name, document)

        assert result["identityKey"] == "vsancadvtjfcq6cv"
        assert result["directHandleIds"] == []
        assert result["fusedConstituentIds"] == []
        assert result["handles"] == {}
        assert len(result["occurrences"]) == 1

    def test_empty_occurrence_selector_has_distinct_diagnostic(self):
        result = resolve_provenance_document(
            self._EVENT_BASE,
            _load_provenance_fixture("valid_v1.json"),
            occurrence_selector="",
        )

        assert result["status"] == "error"
        assert result["diagnostics"][0]["code"] == "invalid-selector"

    def test_deep_fused_lineage_resolves_without_python_recursion(self):
        document = _load_provenance_fixture("valid_v1.json")
        document["handles"]["200"]["fused_from"] = ["201"]
        for handle_id in range(201, 2201):
            constituents = [str(handle_id + 1)] if handle_id < 2200 else ["101", "102"]
            identifier = str(handle_id)
            document["handles"][identifier] = {
                "id": identifier,
                "source": None,
                "aten_op": None,
                "ir_chain": [],
                "fused_from": constituents,
                "transform_history": [],
            }
        document["handles"] = dict(sorted(document["handles"].items()))

        result = resolve_provenance_document(self._EVENT_BASE, document)

        assert result["status"] == "partial"
        assert len(result["handles"]) == 2003
        assert result["fusedConstituentIds"][-2:] == ["101", "102"]

    def test_reader_diagnostics_follow_frozen_fixture_manifest(self):
        manifest = _load_provenance_fixture("fixture_manifest.json")

        for fixture in manifest["invalid"]:
            document = _load_provenance_fixture(fixture["base"])
            mutated = _apply_fixture_mutation(document, fixture["mutation"])
            result = resolve_provenance_document(self._EVENT_BASE, mutated)
            assert result["status"] == "error", fixture["name"]
            assert (
                result["diagnostics"][0]["code"]
                == (fixture["expectedReaderDiagnostic"])
            ), fixture["name"]

    def test_deeply_nested_json_reports_schema_failure(self, tmp_path):
        path = tmp_path / "deep.json"
        path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")

        result = resolve_provenance_event(self._EVENT_BASE, path)

        assert result["status"] == "error"
        assert result["diagnostics"][0]["code"] == "schema-validation-failure"

    def test_recursive_runtime_schema_validation_reports_schema_failure(self):
        document = _load_provenance_fixture("valid_v1.json")

        with patch(
            "torch_spyre.provenance._validate_schema",
            side_effect=RecursionError,
        ):
            result = resolve_provenance_document(self._EVENT_BASE, document)

        assert result["status"] == "error"
        assert result["diagnostics"][0]["code"] == "schema-validation-failure"

    def test_malformed_packaged_schema_reference_reports_schema_failure(self):
        document = _load_provenance_fixture("valid_v1.json")
        malformed_schema = {"$ref": "#/target", "target": []}

        with patch("torch_spyre.provenance._schema", return_value=malformed_schema):
            result = resolve_provenance_document(self._EVENT_BASE, document)

        assert result["status"] == "error"
        assert result["diagnostics"][0]["code"] == "schema-validation-failure"

    @pytest.mark.parametrize(
        "error",
        [
            OSError("missing"),
            UnicodeError("invalid encoding"),
            ValueError("invalid JSON"),
        ],
    )
    def test_packaged_schema_load_failure_is_structured(self, error):
        document = _load_provenance_fixture("valid_v1.json")

        with patch("torch_spyre.provenance._schema", side_effect=error):
            result = resolve_provenance_document(self._EVENT_BASE, document)

        assert result["status"] == "error"
        assert result["diagnostics"][0]["code"] == "schema-validation-failure"

    def test_missing_key_and_collision_are_distinct(self):
        document = _load_provenance_fixture("valid_v1.json")
        missing = resolve_provenance_document(
            "spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa", document
        )
        collision = resolve_provenance_document(
            "spyre_kernel_v1_fused_relu_atqydvnuutl766na", document
        )

        assert missing["diagnostics"][0]["code"] == "missing-key"
        assert collision["diagnostics"][0]["code"] == "collision"

    def test_fresh_process_cli_uses_only_saved_files(self):
        fixture_path = _PROVENANCE_FIXTURE_DIR / "valid_v1.json"
        script = (
            "import sys; import torch; "
            "before = set(sys.modules); "
            "from torch_spyre.provenance import main; "
            "loaded = set(sys.modules) - before; "
            "assert 'torch_spyre._inductor.provenance_writer' not in loaded; "
            "assert 'torch_spyre._inductor.kernel_provenance' not in loaded; "
            "raise SystemExit(main())"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, f"{self._EVENT_BASE}#9", str(fixture_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["event"]["commandStep"] == 9
        assert result["identityKey"] == "atqydvnuutl766na"
        assert len(result["occurrences"]) == 2

    def test_documented_module_cli_runs_with_backend_autoload_disabled(self):
        fixture_path = _PROVENANCE_FIXTURE_DIR / "valid_v1.json"
        env = dict(os.environ)
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch_spyre.provenance",
                f"{self._EVENT_BASE}#9",
                str(fixture_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["event"]["commandStep"] == 9
        assert result["identityKey"] == "atqydvnuutl766na"

        help_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch_spyre.provenance",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert help_result.returncode == 0, help_result.stderr
        assert (
            "TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torch_spyre.provenance"
            in help_result.stdout
        )


class TestProvenanceArtifactPublication:
    def test_config_default_and_destination_resolution(self, monkeypatch, tmp_path):
        assert _DEFAULT_PROVENANCE_ARTIFACT_PATH is not None
        default_path = Path(_DEFAULT_PROVENANCE_ARTIFACT_PATH)
        assert default_path.is_absolute()
        assert default_path.name == "spyre_provenance.json"
        assert default_path.parent.name == "torchinductor"
        assert "torch_compile_debug" in default_path.parts

        monkeypatch.chdir(tmp_path)
        assert resolve_provenance_artifact_path("nested.json") == (
            tmp_path / "nested.json"
        )
        assert resolve_provenance_artifact_path(None) is None
        assert resolve_provenance_artifact_path("") is None

    def test_publication_creates_missing_parent_directory(self, tmp_path):
        collection = _publication_collection(registration_ordinal=1)
        path = tmp_path / "run_0" / "torchinductor" / "spyre_provenance.json"
        assert publish_provenance_collection(collection, str(path)) == "written"
        assert path.exists()

    def test_live_upstream_projection_is_exactly_filtered_and_published(self, tmp_path):
        from torch_spyre._inductor import provenance_writer

        collection = _publication_collection(registration_ordinal=1)
        alias = "sdsc_fused_mm_0:1"
        node_mapping = {
            "version": 2.0,
            "preToPost": {
                "pre_a": ["post_a"],
                "pre_b": ["post_a"],
                "pre_unrelated": ["post_unrelated"],
            },
            "postToPre": {
                "post_a": ["pre_a", "pre_b"],
                "post_unrelated": ["pre_unrelated"],
            },
            "cppCodeToPost": {
                alias: ["post_a"],
                "sdsc_unrelated_1:2": ["post_unrelated"],
            },
            "postToCppCode": {
                "post_a": [alias],
                "post_unrelated": ["sdsc_unrelated_1:2"],
            },
        }
        kernel_information = {
            alias: {
                "stack_traces": ["model.py:10"],
                "post_grad_nodes": ["post_a"],
                "pre_grad_nodes": ["pre_a", "pre_b"],
            },
            "sdsc_unrelated_1:2": {
                "stack_traces": ["other.py:20"],
                "post_grad_nodes": ["post_unrelated"],
                "pre_grad_nodes": ["pre_unrelated"],
            },
        }
        path = tmp_path / "filtered.json"
        inactive_handler = LazyTraceHandler(root_dir=None)
        active_handler = LazyTraceHandler(root_dir=str(tmp_path))

        with patch.object(
            provenance_writer.trace_log,
            "handlers",
            [inactive_handler],
        ):
            assert not provenance_writer._structured_tracing_enabled()

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch(
                "torch_spyre._inductor.provenance_writer.inductor_debug."
                "dump_inductor_provenance_info",
                return_value=node_mapping,
            ),
            patch(
                "torch_spyre._inductor.provenance_writer.inductor_debug."
                "create_kernel_information_json",
                return_value=kernel_information,
            ),
            patch(
                "torch_spyre._inductor.provenance_writer.trace_structured_artifact"
            ) as structured_artifact,
            patch.object(
                provenance_writer.trace_log,
                "handlers",
                [active_handler],
            ),
        ):
            assert provenance_writer._structured_tracing_enabled()
            projection = capture_upstream_projection(collection)
            assert projection is not None
            assert not projection.failed
            assert projection.upstream_join == "ok"
            assert (
                publish_provenance_collection(
                    collection,
                    str(path),
                    upstream_projection=projection,
                )
                == "written"
            )

        document = json.loads(path.read_text(encoding="utf-8"))
        validate_provenance_document(document)
        _SCHEMA_VALIDATOR.validate(document)
        assert document["status"] == "complete"
        assert document["diagnostics"] == {}
        persisted = document["upstreamProjections"][collection.compile_id]
        assert persisted["cppCodeToPost"] == {alias: ["post_a"]}
        assert persisted["settings"]["structuredTracing"] is True
        assert persisted["postToCppCode"] == {"post_a": [alias]}
        assert persisted["postToPre"] == {"post_a": ["pre_a", "pre_b"]}
        assert persisted["preToPost"] == {
            "pre_a": ["post_a"],
            "pre_b": ["post_a"],
        }
        assert persisted["kernelStackTraces"] == {
            alias: {
                "postGradNodes": ["post_a"],
                "preGradNodes": ["pre_a", "pre_b"],
                "stackTraces": ["model.py:10"],
            }
        }
        structured_artifact.assert_called_once()
        assert structured_artifact.call_args.args == ("spyre_provenance", "json")
        assert structured_artifact.call_args.kwargs["payload_fn"]() == (
            path.read_text(encoding="utf-8")
        )

    def test_projection_canonicalizes_numeric_suffix_inverse_order(self, tmp_path):
        collection = _publication_collection(registration_ordinal=1)
        alias = "sdsc_fused_mm_0:1"
        node_mapping = {
            "version": 2.0,
            "preToPost": {
                "attn_9": ["permute_53"],
                "attn_10": ["permute_53"],
                "attn_99": ["permute_503"],
                "attn_100": ["permute_503"],
            },
            "postToPre": {
                "permute_53": ["attn_9", "attn_10"],
                "permute_503": ["attn_99", "attn_100"],
            },
            "cppCodeToPost": {alias: ["permute_53", "permute_503"]},
            "postToCppCode": {
                "permute_53": [alias],
                "permute_503": [alias],
            },
        }
        kernel_information = {
            alias: {
                "stack_traces": ["model.py:10"],
                "post_grad_nodes": ["permute_53", "permute_503"],
                "pre_grad_nodes": [
                    "attn_9",
                    "attn_10",
                    "attn_99",
                    "attn_100",
                ],
            }
        }

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch(
                "torch_spyre._inductor.provenance_writer.inductor_debug."
                "dump_inductor_provenance_info",
                return_value=node_mapping,
            ),
            patch(
                "torch_spyre._inductor.provenance_writer.inductor_debug."
                "create_kernel_information_json",
                return_value=kernel_information,
            ),
        ):
            projection = capture_upstream_projection(collection)

        assert projection is not None
        assert not projection.failed
        assert projection.upstream_join == "ok"
        assert projection.post_to_pre == {
            "permute_503": ("attn_100", "attn_99"),
            "permute_53": ("attn_10", "attn_9"),
        }

        path = tmp_path / "numeric-suffix-order.json"
        assert (
            publish_provenance_collection(
                collection,
                str(path),
                upstream_projection=projection,
            )
            == "written"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_provenance_document(document)
        _SCHEMA_VALIDATOR.validate(document)

        persisted = document["upstreamProjections"][collection.compile_id]
        assert persisted["postToPre"] == {
            "permute_503": ["attn_100", "attn_99"],
            "permute_53": ["attn_10", "attn_9"],
        }
        expected_pairs = {
            ("attn_9", "permute_53"),
            ("attn_10", "permute_53"),
            ("attn_99", "permute_503"),
            ("attn_100", "permute_503"),
        }
        assert {
            (pre_node, post_node)
            for pre_node, post_nodes in persisted["preToPost"].items()
            for post_node in post_nodes
        } == expected_pairs
        assert {
            (pre_node, post_node)
            for post_node, pre_nodes in persisted["postToPre"].items()
            for pre_node in pre_nodes
        } == expected_pairs

    def test_incomplete_upstream_projection_is_partial_and_diagnostic(self, tmp_path):
        collection = _publication_collection(registration_ordinal=1)
        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch(
                "torch_spyre._inductor.provenance_writer.inductor_debug."
                "dump_inductor_provenance_info",
                return_value={
                    "version": 2.0,
                    "cppCodeToPost": {},
                    "postToPre": {},
                },
            ),
            patch(
                "torch_spyre._inductor.provenance_writer.inductor_debug."
                "create_kernel_information_json",
                return_value={},
            ),
        ):
            projection = capture_upstream_projection(collection)
            assert projection is not None
            assert projection.failed
            path = tmp_path / "partial.json"
            publish_provenance_collection(
                collection,
                str(path),
                upstream_projection=projection,
            )

        document = json.loads(path.read_text(encoding="utf-8"))
        persisted = document["upstreamProjections"][collection.compile_id]
        assert persisted["upstreamJoin"] == "partial"
        assert persisted["upstreamProjectionFailed"] is True
        assert document["diagnostics"] == {"upstream-projection-failure": 1}
        assert document["status"] == "partial"

    def test_first_write_is_deterministic_and_schema_valid(self, tmp_path):
        collection = _publication_collection(registration_ordinal=1)
        first_path = tmp_path / "first.json"
        second_path = tmp_path / "second.json"

        with torch._inductor.config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "trace.enabled": False,
            }
        ):
            assert publish_provenance_collection(collection, str(first_path)) == (
                "written"
            )
            assert publish_provenance_collection(collection, str(second_path)) == (
                "written"
            )

        assert first_path.read_bytes() == second_path.read_bytes()
        document = json.loads(first_path.read_text(encoding="utf-8"))
        validate_provenance_document(document)
        _SCHEMA_VALIDATOR.validate(document)
        assert document["mergeGeneration"] == 1
        assert document["status"] == "partial"
        projection = document["upstreamProjections"][collection.compile_id]
        assert projection["upstreamJoin"] == "partial"
        assert not projection["settings"]["structuredTracing"]

        original = first_path.read_bytes()
        with torch._inductor.config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "trace.enabled": False,
            }
        ):
            assert publish_provenance_collection(collection, str(first_path)) == (
                "unchanged"
            )
        assert first_path.read_bytes() == original

    def test_cache_then_fresh_enriches_and_replay_cannot_erase(self, tmp_path):
        path = tmp_path / "cache-order.json"
        replay = _publication_collection(has_graph_lowering=False)
        fresh = _publication_collection(registration_ordinal=1)

        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            assert publish_provenance_collection(replay, str(path)) == "written"
            cache_document = json.loads(path.read_text(encoding="utf-8"))
            cache_projection = cache_document["upstreamProjections"][replay.compile_id]
            assert cache_projection["upstreamJoin"] == "unavailable-cache-replay"

            assert publish_provenance_collection(fresh, str(path)) == "written"
            fresh_bytes = path.read_bytes()
            document = json.loads(fresh_bytes)
            occurrence = next(iter(document["kernelOccurrences"].values()))
            assert occurrence["registrations"] == [
                {"alias": "sdsc_fused_mm_0:1", "ordinal": 1}
            ]
            assert (
                document["upstreamProjections"][fresh.compile_id]["upstreamJoin"]
                == "partial"
            )
            assert document["mergeGeneration"] == 2

            assert publish_provenance_collection(replay, str(path)) == "unchanged"

        assert path.read_bytes() == fresh_bytes

    def test_fresh_then_cache_is_a_deterministic_no_op(self, tmp_path):
        path = tmp_path / "fresh-first.json"
        fresh = _publication_collection(registration_ordinal=1)
        replay = _publication_collection(has_graph_lowering=False)

        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            assert publish_provenance_collection(fresh, str(path)) == "written"
            fresh_bytes = path.read_bytes()
            assert publish_provenance_collection(replay, str(path)) == "unchanged"

        assert path.read_bytes() == fresh_bytes

    def test_projection_settings_enrich_monotonically_across_levels(self, tmp_path):
        from torch_spyre._inductor import provenance_writer

        path = tmp_path / "settings-levels.json"
        collection = _publication_collection(registration_ordinal=1)
        inactive_handler = LazyTraceHandler(root_dir=None)
        active_handler = LazyTraceHandler(root_dir=str(tmp_path))

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 0),
            patch.object(provenance_writer.trace_log, "handlers", [inactive_handler]),
        ):
            assert publish_provenance_collection(collection, str(path)) == "written"

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch.object(provenance_writer.trace_log, "handlers", [active_handler]),
        ):
            assert publish_provenance_collection(collection, str(path)) == "written"

        document = json.loads(path.read_text(encoding="utf-8"))
        projection = document["upstreamProjections"][collection.compile_id]
        assert projection["settings"] == {
            "provenanceTrackingLevel": 1,
            "structuredTracing": True,
        }
        assert document["mergeGeneration"] == 2
        richer_bytes = path.read_bytes()

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 0),
            patch.object(provenance_writer.trace_log, "handlers", [inactive_handler]),
        ):
            assert publish_provenance_collection(collection, str(path)) == "unchanged"
        assert path.read_bytes() == richer_bytes

    def test_equal_rank_projection_retains_structured_tracing_evidence(self, tmp_path):
        from torch_spyre._inductor import provenance_writer

        path = tmp_path / "settings-tracing.json"
        collection = _publication_collection(registration_ordinal=1)
        inactive_handler = LazyTraceHandler(root_dir=None)
        active_handler = LazyTraceHandler(root_dir=str(tmp_path))

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch.object(provenance_writer.trace_log, "handlers", [inactive_handler]),
        ):
            assert publish_provenance_collection(collection, str(path)) == "written"
        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch.object(provenance_writer.trace_log, "handlers", [active_handler]),
        ):
            assert publish_provenance_collection(collection, str(path)) == "written"

        document = json.loads(path.read_text(encoding="utf-8"))
        projection = document["upstreamProjections"][collection.compile_id]
        assert projection["settings"] == {
            "provenanceTrackingLevel": 1,
            "structuredTracing": True,
        }
        assert document["mergeGeneration"] == 2

    def test_unavailable_join_reasons_prefer_enabled_cache_replay(self, tmp_path):
        fresh_level_zero = _publication_collection()
        cache_level_one = _publication_collection(has_graph_lowering=False)
        level_zero_first = tmp_path / "level-zero-first.json"
        cache_first = tmp_path / "cache-first.json"

        with torch._inductor.config.patch("trace.provenance_tracking_level", 0):
            assert (
                publish_provenance_collection(fresh_level_zero, str(level_zero_first))
                == "written"
            )
        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            assert (
                publish_provenance_collection(cache_level_one, str(level_zero_first))
                == "written"
            )
            assert (
                publish_provenance_collection(cache_level_one, str(cache_first))
                == "written"
            )
        with torch._inductor.config.patch("trace.provenance_tracking_level", 0):
            assert (
                publish_provenance_collection(fresh_level_zero, str(cache_first))
                == "unchanged"
            )

        first_document = json.loads(level_zero_first.read_text(encoding="utf-8"))
        second_document = json.loads(cache_first.read_text(encoding="utf-8"))
        for document in (first_document, second_document):
            projection = document["upstreamProjections"][cache_level_one.compile_id]
            assert projection["upstreamJoin"] == "unavailable-cache-replay"
            assert projection["settings"]["provenanceTrackingLevel"] == 1
        first_document.pop("mergeGeneration")
        second_document.pop("mergeGeneration")
        assert first_document == second_document

    def test_multiple_wrappers_coexist_and_diagnostics_are_idempotent(self, tmp_path):
        path = tmp_path / "multi-wrapper.json"
        first = _publication_collection(
            kernel_name="sdsc_fused_mm_0",
            handle_id=9,
            registration_ordinal=1,
            include_uncollected=True,
            upstream_projection_failed=True,
        )
        second = _publication_collection(
            kernel_name="sdsc_fused_relu_1",
            handle_id=10,
            registration_ordinal=2,
        )

        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            assert publish_provenance_collection(first, str(path)) == "written"
            assert publish_provenance_collection(second, str(path)) == "written"
            before_replay = path.read_bytes()
            assert publish_provenance_collection(first, str(path)) == "unchanged"

        assert path.read_bytes() == before_replay

        document = json.loads(path.read_text(encoding="utf-8"))
        validate_provenance_document(document)
        _SCHEMA_VALIDATOR.validate(document)
        assert set(document["upstreamProjections"]) == {
            first.compile_id,
            second.compile_id,
        }
        assert document["diagnostics"] == {
            "collection-failure": 1,
            "upstream-projection-failure": 1,
        }
        first_projection = document["upstreamProjections"][first.compile_id]
        assert first_projection["uncollectedKernels"] == ["sdsc_fused_mm_0_uncollected"]
        assert first_projection["upstreamProjectionFailed"] is True
        assert document["mergeGeneration"] == 2

    def test_concurrent_publications_merge_without_lost_updates(self, tmp_path):
        collections = [
            _publication_collection(
                kernel_name=f"sdsc_fused_mm_{index}",
                handle_id=100 + index,
                registration_ordinal=index + 1,
            )
            for index in range(8)
        ]

        def publish_all(path):
            with ThreadPoolExecutor(max_workers=len(collections)) as executor:
                return list(
                    executor.map(
                        lambda collection: publish_provenance_collection(
                            collection, str(path)
                        ),
                        collections,
                    )
                )

        first_path = tmp_path / "concurrent-first.json"
        second_path = tmp_path / "concurrent-second.json"
        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch("torch_spyre._inductor.provenance_writer.trace_structured_artifact"),
        ):
            assert publish_all(first_path) == ["written"] * len(collections)
            assert publish_all(second_path) == ["written"] * len(collections)

        assert first_path.read_bytes() == second_path.read_bytes()
        document = json.loads(first_path.read_text(encoding="utf-8"))
        validate_provenance_document(document)
        _SCHEMA_VALIDATOR.validate(document)
        assert document["mergeGeneration"] == len(collections)
        assert set(document["upstreamProjections"]) == {
            collection.compile_id for collection in collections
        }
        assert len(document["kernelOccurrences"]) == len(collections)

    def test_cross_process_publications_merge_without_lost_updates(self, tmp_path):
        path = tmp_path / "multiprocess.json"
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        processes = [
            context.Process(
                target=_publish_collection_in_process,
                args=(
                    str(path),
                    f"sdsc_fused_mm_{index}",
                    200 + index,
                    index + 1,
                    start_event,
                ),
            )
            for index in range(2)
        ]

        for process in processes:
            process.start()
        start_event.set()
        for process in processes:
            process.join(timeout=10)

        assert [process.exitcode for process in processes] == [0, 0]
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_provenance_document(document)
        assert document["mergeGeneration"] == len(processes)
        assert len(document["kernelIdentities"]) == len(processes)
        assert len(document["kernelOccurrences"]) == len(processes)

    def test_complete_projection_survives_later_partial_contribution(self, tmp_path):
        path = tmp_path / "rich-then-partial.json"
        collection = _publication_collection(registration_ordinal=1)
        alias = "sdsc_fused_mm_0:1"
        complete = CapturedUpstreamProjection(
            upstream_join="ok",
            pre_to_post={"pre": ("post",)},
            post_to_pre={"post": ("pre",)},
            cpp_code_to_post={alias: ("post",)},
            post_to_cpp_code={"post": (alias,)},
            kernel_stack_traces={
                alias: {
                    "stackTraces": ("model.py:10",),
                    "postGradNodes": ("post",),
                    "preGradNodes": ("pre",),
                }
            },
            failed=False,
        )

        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            assert (
                publish_provenance_collection(
                    collection,
                    str(path),
                    upstream_projection=complete,
                )
                == "written"
            )
            rich_document = json.loads(path.read_text(encoding="utf-8"))
            rich_projection = copy.deepcopy(
                rich_document["upstreamProjections"][collection.compile_id]
            )
            assert (
                publish_provenance_collection(
                    collection,
                    str(path),
                    upstream_projection_failed=True,
                )
                == "written"
            )
            partial_bytes = path.read_bytes()
            assert (
                publish_provenance_collection(
                    collection,
                    str(path),
                    upstream_projection_failed=True,
                )
                == "unchanged"
            )

        assert path.read_bytes() == partial_bytes
        document = json.loads(partial_bytes)
        projection = document["upstreamProjections"][collection.compile_id]
        for field in (
            "preToPost",
            "postToPre",
            "cppCodeToPost",
            "postToCppCode",
            "kernelStackTraces",
        ):
            assert projection[field] == rich_projection[field]
        assert projection["upstreamJoin"] == "ok"
        assert projection["upstreamProjectionFailed"] is True
        assert document["diagnostics"] == {"upstream-projection-failure": 1}
        assert document["status"] == "partial"
        assert document["mergeGeneration"] == 2

    def test_complete_projections_union_additive_stack_context(self, tmp_path):
        path = tmp_path / "complete-context-union.json"
        collection = _publication_collection(registration_ordinal=1)
        alias = "sdsc_fused_mm_0:1"
        first = CapturedUpstreamProjection(
            upstream_join="ok",
            pre_to_post={"pre": ("post",)},
            post_to_pre={"post": ("pre",)},
            cpp_code_to_post={alias: ("post",)},
            post_to_cpp_code={"post": (alias,)},
            kernel_stack_traces={
                alias: {
                    "stackTraces": ("model.py:10",),
                    "postGradNodes": ("post",),
                    "preGradNodes": ("pre",),
                }
            },
            failed=False,
        )
        second = dataclasses.replace(
            first,
            kernel_stack_traces={
                alias: {
                    "stackTraces": ("model.py:20",),
                    "postGradNodes": ("post",),
                    "preGradNodes": ("pre",),
                }
            },
        )

        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            assert (
                publish_provenance_collection(
                    collection, str(path), upstream_projection=first
                )
                == "written"
            )
            assert (
                publish_provenance_collection(
                    collection, str(path), upstream_projection=second
                )
                == "written"
            )

        document = json.loads(path.read_text(encoding="utf-8"))
        validate_provenance_document(document)
        _SCHEMA_VALIDATOR.validate(document)
        projection = document["upstreamProjections"][collection.compile_id]
        assert projection["kernelStackTraces"][alias]["stackTraces"] == [
            "model.py:10",
            "model.py:20",
        ]
        assert projection["upstreamJoin"] == "ok"
        assert projection["upstreamProjectionFailed"] is False
        assert document["diagnostics"] == {}
        assert document["status"] == "complete"
        assert document["mergeGeneration"] == 2

    @pytest.mark.parametrize(
        "case",
        [
            "invalid-json",
            "invalid-utf8",
            "unsupported-version",
            "content-conflict",
        ],
    )
    def test_invalid_or_conflicting_existing_file_is_untouched(self, case, tmp_path):
        path = tmp_path / "existing.json"
        collection = _publication_collection(registration_ordinal=1)
        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            publish_provenance_collection(collection, str(path))

        if case == "invalid-json":
            path.write_bytes(b"{not-json")
        elif case == "invalid-utf8":
            path.write_bytes(b"\xff")
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
            if case == "unsupported-version":
                document["schemaVersion"] = 2
            else:
                handle = next(iter(document["handles"].values()))
                handle["source"]["file"] = "/conflicting/model.py"
            path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        before = path.read_bytes()

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            pytest.raises(ProvenanceArtifactError),
        ):
            publish_provenance_collection(collection, str(path))

        assert path.read_bytes() == before

    def test_failed_replace_preserves_existing_and_removes_temporary_file(
        self, tmp_path
    ):
        path = tmp_path / "atomic.json"
        first = _publication_collection(registration_ordinal=1)
        second = _publication_collection(
            kernel_name="sdsc_fused_relu_1",
            handle_id=10,
            registration_ordinal=2,
        )
        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            publish_provenance_collection(first, str(path))
            before = path.read_bytes()
            with (
                patch(
                    "torch_spyre._inductor.provenance_writer.os.replace",
                    side_effect=OSError("interrupted"),
                ),
                pytest.raises(ProvenanceArtifactError, match="atomic.json"),
            ):
                publish_provenance_collection(second, str(path))

        assert path.read_bytes() == before
        assert list(tmp_path.glob(".atomic.json.*.tmp")) == []

    def test_failed_only_and_disabled_contributions_write_nothing(self, tmp_path):
        builder = ProvenanceCollectionBuilder()
        builder.add_uncollected_kernel("sdsc_uncollected_0")
        collection = builder.finish(_registration_state())
        assert collection is not None
        path = tmp_path / "absent.json"

        with pytest.raises(ProvenanceArtifactError, match="without a collected"):
            publish_provenance_collection(collection, str(path))
        assert not path.exists()
        with patch(
            "torch_spyre._inductor.provenance_writer.trace_structured_artifact"
        ) as structured_artifact:
            assert publish_provenance_collection(collection, None) == "disabled"
            assert publish_provenance_collection(collection, "") == "disabled"
        structured_artifact.assert_not_called()
        assert not path.exists()

    @pytest.mark.parametrize("failure_mode", ["corrupt-sidecar", "missing-lock"])
    def test_wait_publication_failure_preserves_runtime_descriptor(
        self, tmp_path, failure_mode
    ):
        specs = [_op(_handle(9))]
        runner = object()
        compiler = SpyreAsyncCompile()
        graph = _graph_lowering()
        configured_path = tmp_path / "private" / "sidecar.json"
        if failure_mode == "corrupt-sidecar":
            configured_path.parent.mkdir()
            configured_path.write_text("{not-json", encoding="utf-8")
            original_bytes = configured_path.read_bytes()
            lock_context = contextlib.nullcontext()
        else:
            original_bytes = None
            lock_context = patch("torch_spyre._inductor.provenance_writer._fcntl", None)

        with (
            lock_context,
            V.set_graph_handler(graph),
            spyre_config.patch({"provenance_artifact_path": str(configured_path)}),
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ) as runner_type,
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait"),
            patch("torch_spyre.execution.async_compile.logger.warning") as warning,
        ):
            result = compiler.sdsc("sdsc_fused_mm_0", specs)
            compiler.wait({})

        descriptor = runner_type.call_args.kwargs["kernel_provenance"]
        assert result is runner
        assert descriptor is not None
        assert compiler._last_provenance_collection is not None
        if original_bytes is None:
            assert not configured_path.exists()
        else:
            assert configured_path.read_bytes() == original_bytes
        assert warning.call_count == 2
        assert warning.call_args_list[0].args == (
            "provenance sidecar publication failed for %s; continuing compilation",
            "sidecar.json",
        )
        assert warning.call_args_list[0].kwargs["exc_info"] is True
        assert warning.call_args_list[1].args == (
            "provenance sidecar publication incomplete after %d failure(s) "
            "across %d compiled Spyre kernels",
            1,
            1,
        )
        assert str(tmp_path) not in repr(warning.call_args_list)

    def test_wait_upstream_capture_failure_publishes_partial_sidecar(self, tmp_path):
        specs = [_op(_handle(9))]
        runner = object()
        compiler = SpyreAsyncCompile()
        graph = _graph_lowering()
        path = tmp_path / "capture-failure.json"

        with (
            V.set_graph_handler(graph),
            spyre_config.patch({"provenance_artifact_path": str(path)}),
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ),
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait"),
            patch(
                "torch_spyre.execution.async_compile.capture_upstream_projection",
                side_effect=ProvenanceArtifactError("upstream capture failed"),
            ),
            patch("torch_spyre.execution.async_compile.logger.warning") as warning,
        ):
            assert compiler.sdsc("sdsc_fused_mm_0", specs) is runner
            record_kernel_registration(graph, "sdsc_fused_mm_0", 1)
            compiler.wait({})

        document = json.loads(path.read_text(encoding="utf-8"))
        collection = compiler._last_provenance_collection
        assert collection is not None
        projection = document["upstreamProjections"][collection.compile_id]
        assert projection["upstreamJoin"] == "partial"
        assert projection["upstreamProjectionFailed"] is True
        assert document["diagnostics"] == {"upstream-projection-failure": 1}
        assert warning.call_count == 2
        assert warning.call_args_list[0].args == (
            "upstream provenance projection capture failed; continuing "
            "with a partial sidecar",
        )
        assert warning.call_args_list[0].kwargs["exc_info"] is True
        assert warning.call_args_list[1].args == (
            "provenance upstream projection incomplete for this generated wrapper",
        )

    def test_failed_temp_write_removes_temporary_file(self, tmp_path):
        from torch_spyre._inductor import provenance_writer

        path = tmp_path / "write-failure.json"
        collection = _publication_collection(registration_ordinal=1)
        real_named_temporary_file = provenance_writer.tempfile.NamedTemporaryFile

        def failing_named_temporary_file(*args, **kwargs):
            temporary = real_named_temporary_file(*args, **kwargs)

            class FailingTemporaryFile:
                name = temporary.name

                def __enter__(self):
                    temporary.__enter__()
                    return self

                def write(self, payload):
                    raise OSError("interrupted write")

                def __exit__(self, *exc_info):
                    return temporary.__exit__(*exc_info)

            return FailingTemporaryFile()

        with (
            torch._inductor.config.patch("trace.provenance_tracking_level", 1),
            patch(
                "torch_spyre._inductor.provenance_writer.tempfile.NamedTemporaryFile",
                side_effect=failing_named_temporary_file,
            ),
            pytest.raises(ProvenanceArtifactError, match="write-failure.json"),
        ):
            publish_provenance_collection(collection, str(path))

        assert not path.exists()
        assert list(tmp_path.glob(".write-failure.json.*.tmp")) == []

    def test_disabled_publication_logs_once(self):
        from torch_spyre.execution import async_compile as async_compile_module

        with (
            patch.object(
                async_compile_module,
                "_publication_disabled_logged",
                False,
            ),
            patch.object(async_compile_module.logger, "debug") as debug,
        ):
            async_compile_module._log_publication_disabled_once()
            async_compile_module._log_publication_disabled_once()

        debug.assert_called_once_with(
            "Spyre provenance sidecar publication is disabled"
        )

    def test_level_zero_publication_warns_once_with_enabling_setting(self, tmp_path):
        from torch_spyre.execution import async_compile as async_compile_module

        path = tmp_path / "level-zero.json"
        collection = _publication_collection(registration_ordinal=1)
        compiler = SpyreAsyncCompile()
        expected_warning = (
            "Spyre provenance upstream projection is unavailable; set "
            "torch._inductor.config.trace.provenance_tracking_level=1 "
            "to enable it"
        )

        with (
            spyre_config.patch({"provenance_artifact_path": str(path)}),
            torch._inductor.config.patch("trace.provenance_tracking_level", 0),
            patch.object(
                async_compile_module,
                "_provenance_level_zero_logged",
                False,
            ),
            patch.object(
                compiler._artifact_collection_builder,
                "finish",
                return_value=collection,
            ),
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait"),
            patch.object(async_compile_module.logger, "warning") as warning,
        ):
            compiler.wait({})
            warning.assert_called_once_with(expected_warning)
            async_compile_module._log_provenance_level_zero_once()
            warning.assert_called_once_with(expected_warning)

        document = json.loads(path.read_text(encoding="utf-8"))
        projection = document["upstreamProjections"][collection.compile_id]
        assert projection["upstreamJoin"] == "unavailable-provenance-level-0"

    def test_production_validator_rejects_boolean_integer_constants(self, tmp_path):
        path = tmp_path / "strict-integers.json"
        collection = _publication_collection(registration_ordinal=1)
        with torch._inductor.config.patch("trace.provenance_tracking_level", 1):
            publish_provenance_collection(collection, str(path))
        document = json.loads(path.read_text(encoding="utf-8"))

        for field_path in (
            ("schemaVersion",),
            ("eventKey", "version"),
            ("eventKey", "width"),
        ):
            mutated = copy.deepcopy(document)
            target = mutated
            for token in field_path[:-1]:
                target = target[token]
            target[field_path[-1]] = True

            with pytest.raises(ProvenanceArtifactError):
                validate_provenance_document(mutated)
            with pytest.raises(ValidationError):
                _SCHEMA_VALIDATOR.validate(mutated)
