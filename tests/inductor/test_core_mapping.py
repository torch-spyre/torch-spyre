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

import copy
import dataclasses
import math
from types import SimpleNamespace
from unittest import mock

import pytest
import sympy
from lx_finalizer_parity import _normalize_call

import torch_spyre._inductor.codegen.superdsc as superdsc_module
import torch_spyre._inductor.pass_utils as pass_utils_module
import torch_spyre._inductor.spyre_kernel as spyre_kernel_module
from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.codegen.superdsc import parse_op_spec
from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_OP,
)
from torch_spyre._inductor.core_mapping import (
    core_mappings_equal,
    core_to_slice_mapping,
    derive_core_mapping,
    derive_operation_mapping,
    finalize_core_mapping_pure,
    finalize_tensor_work_divisions,
    partition_physical_span_bytes,
    remap_work_division,
)
from torch_spyre._inductor.op_spec import (
    OpSpec,
    TensorArg,
    TensorWorkDivision,
    LX_RELAYOUT_INFO_KEY,
    is_lx_relayout_identity,
)
from torch_spyre._inductor.pass_utils import PerCoreView, per_core_views_equal
from torch_spyre._inductor.spyre_kernel import simplify_op_spec
from torch_spyre._inductor.views import (
    align_tensors,
    align_tensors_pure,
    build_alignment_inputs,
)


def _coordinates(splits, num_cores, **kwargs):
    dims = sympy.symbols(f"dim_0:{len(splits)}")
    mapping = core_to_slice_mapping(dims, splits, num_cores, **kwargs)
    core_id = sympy.Symbol("core_id")
    return [
        tuple(int(mapping[dim].subs(core_id, core)) for dim in dims)
        for core in range(num_cores)
    ]


def test_lx_finalizer_audit_keeps_graph_node_identities_alive(lx_finalizer_parity):
    def node():
        operation = SimpleNamespace(get_operation_name=lambda: "pointwise")
        return SimpleNamespace(node=operation, get_name=lambda: "op0")

    first, second = node(), node()
    first_identity = lx_finalizer_parity._identity(first)
    second_identity = lx_finalizer_parity._identity(second)

    assert first_identity != second_identity
    assert lx_finalizer_parity._node_refs[id(first)] is first
    assert lx_finalizer_parity._node_refs[id(second)] is second


def test_default_mapping_preserves_existing_core_order():
    one_grid = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert _coordinates((2, 3), 12) == one_grid * 2


@pytest.mark.parametrize("slot", [sympy.Rational(1, 2), sympy.Symbol("unresolved")])
def test_owner_slots_must_be_concrete_integers(slot):
    dim = sympy.Symbol("dim")
    division = TensorWorkDivision({dim: 2}, {dim: slot}, num_cores=2)
    with pytest.raises(ValueError, match="non-integral"):
        division.to_core_slices(2)
    assert not core_mappings_equal({dim: slot}, {dim: slot}, 2)
    view = PerCoreView(((0, 2),), ((0, slot),), num_cores=2)
    from torch_spyre._inductor.scratchpad.lx_relayout import _core_slices

    with pytest.raises(ValueError, match="non-integral"):
        _core_slices(view, 2)


def test_codegen_rejects_lx_allocation_without_physical_ownership():
    kernel = spyre_kernel_module.SpyreKernel.__new__(spyre_kernel_module.SpyreKernel)
    kernel.current_node = SimpleNamespace(node=object())
    tensor = SimpleNamespace(layout=SimpleNamespace(allocation={"lx": 0}, lx_view=None))
    with (
        mock.patch.object(spyre_kernel_module, "iteration_space", return_value={}),
        pytest.raises(
            RuntimeError,
            match="missing_view reached codegen without physical ownership",
        ),
    ):
        kernel.create_tensor_arg(False, "missing_view", tensor)


@pytest.mark.parametrize("elems_per_stick", [32, 64, 128])
def test_partition_span_includes_padding_between_rows(elems_per_stick):
    assert (
        partition_physical_span_bytes(
            device_size=(8, 4, elems_per_stick),
            elems_per_stick=elems_per_stick,
            split_by_device_dim={0: 4, 1: 2},
        )
        == 768
    )


@pytest.mark.parametrize(
    "device_size,eps,splits,reason",
    [
        ((), 64, {}, "extents"),
        ((0, 64), 64, {}, "extents"),
        ((-1, 64), 64, {}, "extents"),
        ((8, 64), 0, {}, "elems_per_stick"),
        ((8, 64), 64, {0: 0}, "invalid split"),
        ((8, 64), 64, {-1: 2}, "invalid split"),
        ((8, 64), 64, {2: 2}, "invalid split"),
        ((8, 32), 64, {}, "complete final stick"),
        ((8, 64), 64, {1: 2}, "cannot be split"),
    ],
)
def test_partition_span_rejects_unsupported_geometry(device_size, eps, splits, reason):
    with pytest.raises(ValueError, match=reason):
        partition_physical_span_bytes(device_size, eps, splits)


@pytest.mark.parametrize("contiguous_dim", [0, 1, 2])
def test_selected_dim_varies_first(contiguous_dim):
    splits = (2, 3, 4)
    coordinates = _coordinates(
        splits,
        math.prod(splits),
        contiguous_dim=contiguous_dim,
    )
    assert [
        coordinate[contiguous_dim]
        for coordinate in coordinates[: splits[contiguous_dim]]
    ] == list(range(splits[contiguous_dim]))
    assert all(
        coordinate[dim] == 0
        for coordinate in coordinates[: splits[contiguous_dim]]
        for dim in range(len(splits))
        if dim != contiguous_dim
    )


def _mapping_coordinates(mapping, dims, num_cores):
    core_id = sympy.Symbol("core_id")
    return [
        tuple(int(mapping[dim].subs(core_id, core)) for dim in dims)
        for core in range(num_cores)
    ]


def test_late_mapping_derives_contiguous_gather_groups():
    h, lq = sympy.symbols("h lq")
    mapping = derive_core_mapping(
        (h, lq),
        (4, 8),
        32,
        grouped_splits={h: 4},
    )
    coordinates = _mapping_coordinates(mapping, (h, lq), 32)
    assert coordinates == [(core // 8, core % 8) for core in range(32)]


def test_late_mapping_preserves_selected_contiguous_dimension():
    batch, output, reduction = sympy.symbols("batch output reduction")
    mapping = derive_core_mapping(
        (batch, output, reduction),
        (2, 4, 4),
        32,
        contiguous_dim=reduction,
    )
    coordinates = _mapping_coordinates(mapping, (batch, output, reduction), 32)
    assert [coordinate[2] for coordinate in coordinates[:4]] == [0, 1, 2, 3]
    assert all(coordinate[:2] == (0, 0) for coordinate in coordinates[:4])


def test_late_mapping_derives_contiguous_broadcast_groups():
    h, query = sympy.symbols("h query")
    mapping = derive_core_mapping(
        (query, h),
        (16, 2),
        32,
        grouped_splits={h: 2},
    )
    coordinates = _mapping_coordinates(mapping, (query, h), 32)
    assert coordinates == [(core % 16, core // 16) for core in range(32)]


def test_late_mapping_keeps_shared_destination_after_one_consumer_factors():
    h, query, inner = sympy.symbols("h query inner")
    original = derive_core_mapping(
        (h, query),
        (4, 8),
        32,
        grouped_splits={h: 4},
    )
    factored = derive_core_mapping(
        (h, inner, query),
        (4, 2, 4),
        32,
        grouped_splits={h: 4},
    )
    assert core_mappings_equal({h: original[h]}, {h: factored[h]}, 32)


def test_group_topology_does_not_follow_final_loop_reordering():
    head, kv, query = sympy.symbols("head kv query")
    grouped_splits = {head: 2, kv: 2}
    original = derive_core_mapping(
        (head, kv, query),
        (2, 2, 8),
        32,
        grouped_splits=grouped_splits,
    )
    reordered = derive_core_mapping(
        (query, kv, head),
        (8, 2, 2),
        32,
        grouped_splits=grouped_splits,
    )
    assert _mapping_coordinates(original, (head, kv), 32) == _mapping_coordinates(
        reordered, (head, kv), 32
    )


def test_tensor_work_division_compares_physical_owners_not_sympy_spelling():
    head = sympy.Symbol("head")
    core_id = sympy.Symbol("core_id")
    left = TensorWorkDivision(
        {head: 4},
        {head: sympy.Mod(core_id, 4)},
        num_cores=8,
    )
    right = TensorWorkDivision(
        {head: 4},
        {head: core_id - 4 * sympy.floor(core_id / 4)},
        num_cores=8,
    )

    assert left != right
    assert left.same_ownership(right)


def test_tensor_work_division_rejects_different_core_order():
    head = sympy.Symbol("head")
    core_id = sympy.Symbol("core_id")
    left = TensorWorkDivision(
        {head: 4},
        {head: sympy.Mod(core_id, 4)},
        num_cores=8,
    )
    right = TensorWorkDivision(
        {head: 4},
        {head: sympy.floor(core_id / 2)},
        num_cores=8,
    )

    assert not left.same_ownership(right)


def test_tensor_work_division_ignores_unsplit_dimensions_and_infers_core_count():
    head, local = sympy.symbols("head local")
    core_id = sympy.Symbol("core_id")
    left = TensorWorkDivision(
        {head: 4, local: 1},
        {head: sympy.Mod(core_id, 4), local: sympy.Integer(0)},
    )
    right = TensorWorkDivision(
        {head: 4},
        {head: sympy.Mod(core_id, 4)},
        num_cores=4,
    )

    assert left.physical_core_count == 4
    assert left.same_ownership(right)


def test_per_core_view_compares_physical_owners_not_sympy_spelling():
    core_id = sympy.Symbol("core_id")
    left = PerCoreView(
        ((0, 4),),
        ((0, sympy.Mod(core_id, 4)),),
        num_cores=8,
    )
    right = PerCoreView(
        ((0, 4), (1, 1)),
        (
            (0, core_id - 4 * sympy.floor(core_id / 4)),
            (1, sympy.Integer(0)),
        ),
        num_cores=8,
    )
    reordered = PerCoreView(
        ((0, 4),),
        ((0, sympy.floor(core_id / 2)),),
        num_cores=8,
    )

    assert left != right
    assert left.same_partition(right)
    assert per_core_views_equal(left, right)
    assert not left.same_partition(reordered)
    assert per_core_views_equal(None, None)


def test_remap_work_division_accepts_equivalent_merged_owner_slots():
    first, second, merged = sympy.symbols("first second merged")
    core_id = sympy.Symbol("core_id")
    division = TensorWorkDivision(
        {first: 4, second: 4},
        {
            first: sympy.Mod(core_id, 4),
            second: core_id - 4 * sympy.floor(core_id / 4),
        },
        num_cores=8,
    )

    remapped = remap_work_division(
        division,
        {first: ((merged, 4),), second: ((merged, 4),)},
    )

    assert remapped.physical_core_count == 8
    assert remapped.work_slices == {merged: 4}
    assert core_mappings_equal(
        {merged: remapped.core_id_to_work_slice[merged]},
        {merged: sympy.Mod(core_id, 4)},
        8,
    )


def test_remap_work_division_rejects_conflicting_merged_owner_slots():
    first, second, merged = sympy.symbols("first second merged")
    core_id = sympy.Symbol("core_id")
    division = TensorWorkDivision(
        {first: 4, second: 4},
        {
            first: sympy.Mod(core_id, 4),
            second: sympy.floor(core_id / 2),
        },
        num_cores=8,
    )

    with pytest.raises(ValueError, match="conflicting normalized ownership"):
        remap_work_division(
            division,
            {first: ((merged, 4),), second: ((merged, 4),)},
        )


def test_remap_work_division_reports_missing_alignment():
    old, other = sympy.symbols("old other")
    division = TensorWorkDivision(
        {old: 2},
        {old: sympy.Mod(sympy.Symbol("core_id"), 2)},
        num_cores=2,
    )

    with pytest.raises(ValueError, match="old has no alignment"):
        remap_work_division(division, {other: ((other, 2),)})


def test_commit_tensor_work_division_completes_unsplit_loops(monkeypatch):
    split, local = sympy.symbols("split local")
    core_id = sympy.Symbol("core_id")
    op = SimpleNamespace()
    monkeypatch.setattr(
        pass_utils_module,
        "iteration_space_from_op",
        lambda _: {split: sympy.Integer(8), local: sympy.Integer(64)},
    )

    pass_utils_module.commit_tensor_work_division(
        op,
        TensorWorkDivision(
            {split: 4},
            {split: sympy.Mod(core_id, 4)},
            num_cores=8,
        ),
    )

    assert op.iteration_space_ownership == TensorWorkDivision(
        {split: 4, local: 1},
        {split: sympy.Mod(core_id, 4), local: sympy.S.Zero},
        num_cores=8,
    )


def test_certified_identity_fails_if_equivalent_owner_spellings_collapse():
    head = sympy.Symbol("head")
    core_id = sympy.Symbol("core_id")
    base = TensorArg(
        True,
        0,
        DataFormats.SEN169_FP16,
        [4, 64],
        [head, head],
        {"lx": 0},
        work_division=TensorWorkDivision(
            {head: 4},
            {head: sympy.Mod(core_id, 4)},
            num_cores=4,
        ),
    )
    destination = dataclasses.replace(
        base,
        is_input=False,
        work_division=TensorWorkDivision(
            {head: 4},
            {head: core_id - 4 * sympy.floor(core_id / 4)},
            num_cores=4,
        ),
    )

    with pytest.raises(ValueError, match="ownership collapsed"):
        is_lx_relayout_identity(
            "identity", (base, destination), {LX_RELAYOUT_INFO_KEY: "shuffle"}
        )
    different_destination = dataclasses.replace(
        destination,
        work_division=TensorWorkDivision(
            {head: 4},
            {head: sympy.floor(core_id / 2)},
            num_cores=4,
        ),
    )
    assert not is_lx_relayout_identity("identity", (base, different_destination), {})
    assert is_lx_relayout_identity(
        "identity",
        (base, different_destination),
        {LX_RELAYOUT_INFO_KEY: "shuffle"},
    )
    with pytest.raises(ValueError, match="registered plan kind"):
        is_lx_relayout_identity(
            "identity", (base, different_destination), {LX_RELAYOUT_INFO_KEY: True}
        )


def test_late_mapping_rejects_geometry_that_does_not_fill_groups():
    h, query = sympy.symbols("h query")
    with pytest.raises(ValueError, match="does not match operation split"):
        derive_core_mapping(
            (h, query),
            (4, 8),
            32,
            grouped_splits={h: 2},
        )


def test_final_tensor_ownership_preserves_committed_core_order():
    extra, shared = sympy.symbols("extra shared")
    core_id = sympy.Symbol("core_id")
    division = TensorWorkDivision(
        {shared: 2},
        # Planning-time placement is working data, not the final assignment.
        {shared: sympy.Mod(core_id, 2)},
        num_cores=4,
    )

    (finalized,) = finalize_tensor_work_divisions(
        {extra: (8, 2), shared: (8, 2)},
        [division],
    )

    assert finalized == TensorWorkDivision(
        {shared: 2},
        {shared: sympy.Mod(core_id, 2)},
        num_cores=4,
    )


def test_shared_lx_buffer_keeps_owners_across_different_operation_dims():
    producer_extra, producer_shared = sympy.symbols("producer_extra producer_shared")
    consumer_shared, consumer_extra = sympy.symbols("consumer_shared consumer_extra")
    core_id = sympy.Symbol("core_id")
    # The shared tensor owns contiguous two-core groups. That one physical
    # order remains valid when producer and consumer spell their loops in a
    # different order.
    owners = sympy.floor(core_id / 2)
    producer_division = finalize_tensor_work_divisions(
        {producer_extra: (8, 2), producer_shared: (8, 2)},
        [
            TensorWorkDivision(
                {producer_shared: 2},
                {producer_shared: owners},
                num_cores=4,
            )
        ],
    )[0]
    consumer_division = finalize_tensor_work_divisions(
        {consumer_shared: (8, 2), consumer_extra: (8, 2)},
        [
            TensorWorkDivision(
                {consumer_shared: 2},
                {consumer_shared: owners},
                num_cores=4,
            )
        ],
    )[0]
    assert producer_division is not None
    assert consumer_division is not None

    producer = derive_operation_mapping(
        {producer_extra: (8, 2), producer_shared: (8, 2)},
        [producer_division],
    )
    consumer = derive_operation_mapping(
        {consumer_shared: (8, 2), consumer_extra: (8, 2)},
        [consumer_division],
    )

    assert core_mappings_equal(
        {producer_shared: producer[producer_shared]},
        {producer_shared: consumer[consumer_shared]},
        4,
    )


def test_final_tensor_ownership_makes_the_inferred_core_domain_explicit():
    shared = sympy.Symbol("shared")
    core_id = sympy.Symbol("core_id")

    (finalized,) = finalize_tensor_work_divisions(
        {shared: (8, 2)},
        [
            TensorWorkDivision(
                {shared: 2},
                {shared: sympy.Mod(core_id, 2)},
            )
        ],
    )

    assert finalized is not None
    assert finalized.num_cores == 2
    assert finalized.core_id_to_work_slice == {shared: sympy.Mod(core_id, 2)}


def test_operation_mapping_preserves_a_satisfying_default_map():
    batch, head = sympy.symbols("batch head")
    core_id = sympy.Symbol("core_id")
    iteration_space = {batch: (8, 2), head: (16, 4)}
    default = derive_core_mapping((batch, head), (2, 4), 8)
    division = TensorWorkDivision(
        {head: 4},
        {head: sympy.Mod(sympy.floor(core_id / 2), 4)},
        num_cores=8,
    )

    assert derive_operation_mapping(iteration_space, [division]) == default


def test_operation_mapping_rejects_conflicting_lx_tensor_owners():
    shared, extra = sympy.symbols("shared extra")
    core_id = sympy.Symbol("core_id")
    divisions = [
        TensorWorkDivision(
            {shared: 2},
            {shared: sympy.Mod(core_id, 2)},
            num_cores=4,
        ),
        TensorWorkDivision(
            {shared: 2},
            {shared: sympy.floor(core_id / 2)},
            num_cores=4,
        ),
    ]

    with pytest.raises(ValueError, match="disagree on core ownership"):
        derive_operation_mapping(
            {shared: (8, 2), extra: (8, 2)},
            divisions,
        )


def test_operation_mapping_bounds_aligned_owner_dimension_permutations():
    dims = sympy.symbols("d0:6")
    core_id = sympy.Symbol("core_id")
    division = TensorWorkDivision(
        {dim: 2 for dim in dims},
        {
            dim: sympy.Mod(sympy.floor(core_id / (2 ** (len(dims) - index - 1))), 2)
            for index, dim in enumerate(dims)
        },
        num_cores=64,
    )

    with pytest.raises(
        ValueError,
        match="too many aligned tensor-owned dimensions.*6 > 5",
    ):
        derive_operation_mapping(
            {dim: (sympy.Integer(2), 2) for dim in dims},
            [division],
        )


def test_scalar_op_has_a_complete_empty_mapping():
    op_spec = OpSpec("identity", False, {}, [], {})
    simplify_op_spec(op_spec)
    assert op_spec.core_id_to_work_slice == {}


def test_alignment_preview_is_repeatable_and_does_not_consume_repeat_info():
    dim = sympy.Symbol("dim")
    repeat_info = {
        dim: {
            "modulus": sympy.Integer(2),
            "node": sympy.Mod(dim, 2),
            "kind": "mod",
        }
    }
    original = {symbol: dict(info) for symbol, info in repeat_info.items()}
    args = (
        {dim: (sympy.Integer(4), 2)},
        [{"size": [2, 64], "coordinates": [sympy.floor(dim / 2), dim]}],
    )

    preview = align_tensors(*args, repeat_info=repeat_info)
    codegen = align_tensors(*args, repeat_info=repeat_info)

    assert repeat_info == original
    assert preview == codegen


def test_captured_alignment_inputs_leave_codegen_unchanged(monkeypatch):
    dim = sympy.Symbol("dim")
    op_spec = OpSpec(
        "identity",
        False,
        {dim: (sympy.Integer(4), 2)},
        [
            TensorArg(
                True,
                0,
                DataFormats.SEN169_FP16,
                [2, 64],
                [sympy.floor(dim / 2), dim],
                {"hbm": 0},
            )
        ],
        {},
    )
    monkeypatch.setattr(
        pass_utils_module,
        "alignment_coordinates",
        lambda *args, **kwargs: [sympy.floor(dim / 2), dim],
    )
    captured = pass_utils_module.build_operation_alignment_inputs(
        {dim: sympy.Integer(4)},
        [pass_utils_module.AlignmentAccess(SimpleNamespace(device_size=[2, 64]), dim)],
        indirect_sizes={sympy.Symbol("unused_prior_index"): 67},
        aligned_iteration_space=op_spec.iteration_space,
    )
    assert captured.indirect_sizes == {}
    # A preceding validation preview must neither consume nor change the input
    # subsequently used by codegen.
    align_tensors_pure(captured)

    captured_path = copy.deepcopy(op_spec)
    ordinary_path = copy.deepcopy(op_spec)
    simplify_op_spec(captured_path, alignment_inputs=captured)
    simplify_op_spec(ordinary_path)

    assert captured_path == ordinary_path


@pytest.mark.parametrize("is_relayout", [False, True])
def test_preflight_and_codegen_finalize_from_identical_inputs(is_relayout):
    """The recoverable preflight and codegen must run the same finalization.

    This is the permanent caller-boundary proof: preflight asks the shared input
    builder to attach the operation's committed splits, while codegen supplies
    that already-finalized split space. Neither caller may add another ownership
    derivation.
    """

    row, column, element = sympy.symbols("row column element")
    raw_iteration_space = {
        row: sympy.Integer(4),
        column: sympy.Integer(4),
        element: sympy.Integer(64),
    }
    index = 256 * row + 64 * column + element
    read = pass_utils_module.MemoryDep(
        "source",
        index,
        tuple(raw_iteration_space),
        tuple(raw_iteration_space.values()),
    )
    write = pass_utils_module.MemoryDep(
        "destination",
        index,
        tuple(raw_iteration_space),
        tuple(raw_iteration_space.values()),
    )
    read_writes = SimpleNamespace(reads={read}, writes={write})
    op = SimpleNamespace(
        op_it_space_splits=({sympy.Integer(256): 2, sympy.Integer(64): 2}, {}),
        get_name=lambda: "alignment_contract",
    )
    device_layout = pass_utils_module.SpyreTensorLayout(
        [4, 4, 64],
        [256, 64, 1],
        DataFormats.SEN169_FP16,
        ElementArrangement.STANDARD,
    )
    accesses = [
        pass_utils_module.AlignmentAccess(device_layout, read.index),
        pass_utils_module.AlignmentAccess(device_layout, write.index),
    ]
    aligned_iteration_space = pass_utils_module.iteration_space_with_splits(
        op, read_writes, raw_iteration_space
    )
    preflight_inputs = pass_utils_module.build_operation_alignment_inputs(
        raw_iteration_space,
        accesses,
        indirect_sizes={},
        op=op,
        read_writes=read_writes,
    )
    codegen_inputs = pass_utils_module.build_operation_alignment_inputs(
        raw_iteration_space,
        accesses,
        indirect_sizes={},
        aligned_iteration_space=aligned_iteration_space,
    )

    assert preflight_inputs == codegen_inputs

    source_mapping = core_to_slice_mapping((row, column), (2, 2), 4)
    destination_mapping = core_to_slice_mapping((column, row), (2, 2), 4)
    source_division = TensorWorkDivision(
        {row: 2, column: 2},
        source_mapping,
        num_cores=4,
    )
    destination_division = TensorWorkDivision(
        {row: 2, column: 2},
        destination_mapping,
        num_cores=4,
    )
    divisions = (
        (source_division, destination_division)
        if is_relayout
        else (destination_division, destination_division)
    )
    kwargs = {
        "is_matmul": False,
        "core_id_k_fast": False,
        "is_relayout": is_relayout,
    }

    assert finalize_core_mapping_pure(
        preflight_inputs, divisions, **kwargs
    ) == finalize_core_mapping_pure(codegen_inputs, divisions, **kwargs)
    if is_relayout:
        with pytest.raises(ValueError, match="ownership collapse after alignment"):
            finalize_core_mapping_pure(
                preflight_inputs,
                (destination_division, destination_division),
                **kwargs,
            )


def test_finalizer_treats_repeated_identical_constraints_idempotently():
    """ReadWrites may collapse ``x + x`` while codegen keeps both operands."""

    row, column = sympy.symbols("row column")
    iteration_space = {
        row: (sympy.Integer(4), 2),
        column: (sympy.Integer(4), 2),
    }
    tensor = {"size": [4, 4, 64], "coordinates": [row, column, sympy.S.Zero]}
    output = {"size": [4, 4, 64], "coordinates": [row, column, sympy.S.Zero]}
    one_read = build_alignment_inputs(iteration_space, [tensor, output])
    repeated_read = build_alignment_inputs(iteration_space, [tensor, tensor, output])
    mapping = core_to_slice_mapping((row, column), (2, 2), 4)
    division = TensorWorkDivision(
        {row: 2, column: 2},
        mapping,
        num_cores=4,
    )
    equivalent = TensorWorkDivision(
        division.work_slices,
        {
            dim: sympy.Mod(expression + 2, 2, evaluate=False)
            for dim, expression in mapping.items()
        },
        num_cores=4,
    )
    kwargs = {
        "is_matmul": False,
        "core_id_k_fast": False,
        "is_relayout": False,
    }

    unique_result = finalize_core_mapping_pure(
        one_read,
        (division, division),
        **kwargs,
    )
    repeated_result = finalize_core_mapping_pure(
        repeated_read,
        (division, equivalent, division),
        **kwargs,
    )

    assert unique_result[0] == repeated_result[0]
    assert unique_result[3:] == repeated_result[3:]
    assert repeated_result[1][0] == repeated_result[1][1] == unique_result[1][0]
    assert repeated_result[2][0].same_ownership(repeated_result[2][1])
    assert repeated_result[2][0].same_ownership(unique_result[2][0])
    assert _normalize_call(one_read, (division, division), kwargs) == _normalize_call(
        repeated_read, (division, equivalent, division), kwargs
    )
    unsplit = sympy.Symbol("unsplit")
    explicit_unsplit = TensorWorkDivision(
        {**division.work_slices, unsplit: 1},
        {**mapping, unsplit: sympy.S.Zero},
        num_cores=4,
    )
    assert _normalize_call(one_read, (division, division), kwargs) == _normalize_call(
        one_read, (explicit_unsplit, division), kwargs
    )


def test_relayout_supports_distinct_physical_core_domains():
    """A two-core source can broadcast into a 32-core destination view."""

    head = sympy.Symbol("head")
    core_id = sympy.Symbol("core_id")
    inputs = build_alignment_inputs(
        {head: (sympy.Integer(2), 2)},
        [
            {"size": [2, 64], "coordinates": [head, sympy.S.Zero]},
            {"size": [2, 64], "coordinates": [head, sympy.S.Zero]},
        ],
    )
    source = TensorWorkDivision(
        {head: 2},
        {head: sympy.Mod(core_id, 2)},
        num_cores=2,
    )
    destination = TensorWorkDivision(
        {head: 2},
        {head: sympy.floor(core_id / 16)},
        num_cores=32,
    )

    aligned_space, _, _, mapping, _ = finalize_core_mapping_pure(
        inputs,
        (source, destination),
        is_matmul=False,
        core_id_k_fast=False,
        is_relayout=True,
    )

    assert set(mapping) == set(aligned_space) == {head}
    assert core_mappings_equal(
        {head: mapping[head]},
        {head: sympy.floor(core_id / 16)},
        32,
    )

    invalid_source = TensorWorkDivision(
        {head: 3},
        {head: sympy.Mod(core_id, 3)},
        num_cores=3,
    )
    with pytest.raises(ValueError, match="core domains must divide"):
        finalize_core_mapping_pure(
            inputs,
            (invalid_source, destination),
            is_matmul=False,
            core_id_k_fast=False,
            is_relayout=True,
        )

    mismatched_destination = TensorWorkDivision(
        {head: 4},
        {head: sympy.Mod(core_id, 4)},
        num_cores=32,
    )
    with pytest.raises(ValueError, match="split exceeds its aligned extent"):
        finalize_core_mapping_pure(
            inputs,
            (source, mismatched_destination),
            is_matmul=False,
            core_id_k_fast=False,
            is_relayout=True,
        )


def _bmm_op_spec(op: str) -> OpSpec:
    mb, out, reduction = sympy.symbols("mb out reduction")
    args = [
        TensorArg(
            True,
            0,
            DataFormats.SEN169_FP16,
            [512, 64, 1, 64],
            [
                mb,
                sympy.floor(reduction / 64),
                sympy.Integer(0),
                sympy.Mod(reduction, 64),
            ],
            {"hbm": 0},
        ),
        TensorArg(
            True,
            1,
            DataFormats.SEN169_FP16,
            [200, 4096, 64],
            [sympy.floor(out / 64), reduction, sympy.Mod(out, 64)],
            {"hbm": 0x400000000},
        ),
        TensorArg(
            False,
            2,
            DataFormats.SEN169_FP16,
            [512, 200, 1, 64],
            [
                mb,
                sympy.floor(out / 64),
                sympy.Integer(0),
                sympy.Mod(out, 64),
            ],
            {"hbm": 0x800000000},
        ),
    ]
    return OpSpec(
        op,
        True,
        {mb: (512, 2), out: (12800, 4), reduction: (4096, 4)},
        args,
        {},
    )


@pytest.mark.parametrize("op", [BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP])
@pytest.mark.parametrize("reduction_contiguous", [False, True])
def test_planner_and_sdsc_use_the_same_mapping(monkeypatch, op, reduction_contiguous):
    class FakeReduction:
        def __init__(self, reduction_type):
            self.reduction_type = reduction_type

    class FakeComputedBuffer:
        def __init__(self, reduction_type):
            self.data = FakeReduction(reduction_type)

    monkeypatch.setattr(pass_utils_module, "Reduction", FakeReduction)
    monkeypatch.setattr(pass_utils_module, "ComputedBuffer", FakeComputedBuffer)
    monkeypatch.setattr(
        pass_utils_module.config,
        "core_id_k_fast_emission",
        reduction_contiguous,
    )
    monkeypatch.setattr(
        superdsc_module._spyre_config,
        "core_id_k_fast_emission",
        reduction_contiguous,
    )

    op_spec = _bmm_op_spec(op)
    dims = tuple(op_spec.iteration_space)
    splits = dict(zip(dims, (2, 4, 4)))
    prep = pass_utils_module._ViewPrep(
        iter_space=op_spec.iteration_space,
        write_index=dims[0],
        read_index=dims[-1],
        dep_coeff={dims[0]: 1, dims[1]: 2, dims[2]: 0},
        dep_device_coordinates=(dims[0], dims[1]),
        device_size=[2, 4],
        stride_map=[1, 2],
        elems_per_stick=64,
        device_stride_to_dim={1: 0, 2: 1},
        stick_host_stride=None,
        num_stick_dim=None,
        num_stick=0,
        num_stick_stride=0,
        is_matmul=pass_utils_module._is_matmul_op(FakeComputedBuffer(op)),
    )
    planner_view, _, representable = pass_utils_module._per_core_view_from_prep(
        prep, splits, {dims[2]: 4}
    )

    op_spec.core_id_to_work_slice = derive_operation_mapping(
        op_spec.iteration_space,
        contiguous_dim=dims[-1] if reduction_contiguous else None,
    )
    sdsc_spec, renamed = parse_op_spec(op_spec)
    sdsc_output_mapping = {
        device_dim: sdsc_spec.core_id_to_work_slice[renamed[dim]]
        for device_dim, dim in enumerate(dims[:2])
    }
    assert representable
    assert dict(planner_view.core_to_slot) == sdsc_output_mapping


def test_flattened_iteration_span_is_not_a_single_axis_view():
    heads, flat = sympy.symbols("heads flat")
    prep = pass_utils_module._ViewPrep(
        iter_space={heads: 16, flat: 512},
        write_index=512 * heads + flat,
        read_index=512 * heads + flat,
        dep_coeff={heads: 512, flat: 1},
        dep_device_coordinates=(
            sympy.floor(flat / 256),
            sympy.S.Zero,
            sympy.S.Zero,
            sympy.S.Zero,
            sympy.floor(sympy.Mod(flat, 256) / 64),
            heads,
            sympy.Mod(flat, 64),
        ),
        device_size=[2, 1, 1, 1, 4, 16, 64],
        stride_map=[256, -1, -1, -1, 64, 512, 1],
        elems_per_stick=64,
        device_stride_to_dim={256: 0, 64: 4, 512: 5, 1: 6},
        stick_host_stride=1,
        num_stick_dim=4,
        num_stick=4,
        num_stick_stride=64,
        is_matmul=False,
    )

    view, partial, representable = pass_utils_module._per_core_view_from_prep(
        prep, {heads: 16, flat: 2}
    )

    assert not representable
    assert not partial
    assert not view.work_slice_dims


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("captured_k", [False, True])
def test_stride_selected_compound_view_matches_actual_owned_values(reverse, captured_k):
    """A successful stride lookup must not hide a fused batch/head split."""
    flat = sympy.Symbol("flat", integer=True, nonnegative=True)
    core = sympy.Symbol("core_id")
    if captured_k:
        # The saved K access fuses four batches of eight heads. Its stride
        # selects the head axis, but each of eight partitions owns four heads
        # of one batch, not one head across all four batches.
        extent, split = 32, 8
        coordinates = (sympy.Mod(flat, 8), sympy.floor(flat / 8))
        device_size, stride_map = [8, 4], [1, 8]
        stick_host_stride, num_stick_dim, num_stick = None, None, 0
    else:
        extent, split = 128, 2
        coordinates = (sympy.floor(flat / 64), sympy.Mod(flat, 64))
        device_size, stride_map = [2, 64], [64, 1]
        stick_host_stride, num_stick_dim, num_stick = 1, 0, 2
    owner = split - 1 - core if reverse else core
    prep = pass_utils_module._ViewPrep(
        iter_space={flat: extent},
        write_index=flat,
        read_index=flat,
        dep_coeff={flat: 1},
        dep_device_coordinates=coordinates,
        device_size=device_size,
        stride_map=stride_map,
        elems_per_stick=64,
        device_stride_to_dim={stride: axis for axis, stride in enumerate(stride_map)},
        stick_host_stride=stick_host_stride,
        num_stick_dim=num_stick_dim,
        num_stick=num_stick,
        num_stick_stride=64 if num_stick else 0,
        is_matmul=False,
    )
    view, partial, representable = pass_utils_module._per_core_view_from_prep(
        prep,
        {flat: split},
        ownership=TensorWorkDivision({flat: split}, {flat: owner}, num_cores=split),
    )
    assert not partial
    if not representable:
        # The ownership foundation rejects K; the later exact-decomposition
        # extension may accept it, but must satisfy the same element proof.
        assert captured_k
        return
    physical_splits = dict(view.work_slice_dims)
    physical_slots = dict(view.core_to_slot)
    for c in range(split):
        logical_slot = int(owner.subs(core, c))
        expected = set(
            range(
                logical_slot * (extent // split), (logical_slot + 1) * (extent // split)
            )
        )
        actual = {
            point
            for point in range(extent)
            if all(
                int(coordinates[axis].subs(flat, point))
                // (device_size[axis] // factor)
                == int(physical_slots[axis].subs(core, c))
                for axis, factor in physical_splits.items()
            )
        }
        assert actual == expected, (c, actual, expected)


@pytest.mark.parametrize(
    "extent, split, representable",
    [(129, 3, True), (192, 3, True), (129, 2, False), (100, 3, False)],
)
def test_stick_axis_split_is_proved_in_whole_sticks(extent, split, representable):
    """A loop over the stickified axis owns whole sticks, padding included.

    129 fp16 values occupy three sticks of 64 slots. The exact proof must
    partition those three sticks, not 43 host elements at a time; a
    43-element piece crosses a stick boundary and would reject a layout the
    device expresses exactly. Two cores cannot own three sticks evenly, and a
    loop that stops short of the last stick keeps the element model.
    """
    flat = sympy.Symbol("flat", integer=True, nonnegative=True)
    core = sympy.Symbol("core_id")
    coordinates = (sympy.floor(flat / 64), sympy.Mod(flat, 64))
    prep = pass_utils_module._ViewPrep(
        iter_space={flat: extent},
        write_index=flat,
        read_index=flat,
        dep_coeff={flat: 1},
        dep_device_coordinates=coordinates,
        device_size=[3, 64],
        stride_map=[64, 1],
        elems_per_stick=64,
        device_stride_to_dim={64: 0, 1: 1},
        stick_host_stride=1,
        num_stick_dim=0,
        num_stick=3,
        num_stick_stride=64,
        is_matmul=False,
    )
    view, partial, is_representable = pass_utils_module._per_core_view_from_prep(
        prep,
        {flat: split},
        ownership=TensorWorkDivision({flat: split}, {flat: core}, num_cores=split),
    )
    assert not partial
    assert is_representable == representable
    if not representable:
        return
    assert dict(view.work_slice_dims) == {0: split}
    slot = dict(view.core_to_slot)[0]
    assert [int(slot.subs(core, c)) for c in range(split)] == list(range(split))


def _prepare_compound_axis_view(iter_space, index, repeat_info=None):
    device_layout = pass_utils_module.SpyreTensorLayout(
        [1, 1, 8, 16, 64],
        [-1, -1, 64, 512, 1],
        DataFormats.SEN169_FP16,
        ElementArrangement.STANDARD,
    )
    layout = pass_utils_module.FixedTiledLayout(
        "spyre:0",
        pass_utils_module.torch.float16,
        [16, 512],
        [512, 1],
        device_layout,
    )
    dep = pass_utils_module.MemoryDep(
        "buf",
        index,
        tuple(iter_space),
        tuple(iter_space.values()),
    )
    graph = SimpleNamespace(
        _repeat_info={} if repeat_info is None else repeat_info,
        get_buffer=lambda name: SimpleNamespace(layout=layout),
    )
    rw = SimpleNamespace(writes={dep}, reads={dep})
    with (
        pass_utils_module.V.set_graph_handler(graph),
        mock.patch.object(pass_utils_module, "op_read_writes", return_value=rw),
        mock.patch.object(
            pass_utils_module,
            "iteration_space_from_op",
            return_value=iter_space,
        ),
    ):
        prep = pass_utils_module._prepare_per_core_view(
            object(),
            dep,
            "buf",
        )
    assert prep is not None
    return prep, graph


def test_prepare_per_core_view_does_not_record_repeat_info():
    head, flat = sympy.symbols("head flat", integer=True, nonnegative=True)
    prior = sympy.Symbol("prior")
    existing = {prior: {"kind": "mod", "modulus": 2}}
    before = dict(existing)

    _, graph = _prepare_compound_axis_view(
        {head: 16, flat: 512},
        512 * head + sympy.Mod(flat, 256),
        existing,
    )

    assert graph._repeat_info is existing
    assert graph._repeat_info == before


def test_reshape_changes_per_core_ownership_within_one_device_axis():
    """A split of an inner term is not a contiguous split of the containing axis.

    This is the Gemma 4 decode geometry: the producer splits a flattened
    512-element dimension into two contiguous 256-element halves, while its
    consumer views that dimension as ``[2, 256]`` and splits the inner 256.
    The latter owns alternating pairs of sticks, not contiguous groups of four.
    """
    producer_head, producer_flat = sympy.symbols(
        "producer_head producer_flat", integer=True, nonnegative=True
    )
    producer_prep, _ = _prepare_compound_axis_view(
        {producer_head: 16, producer_flat: 512},
        512 * producer_head + producer_flat,
    )
    producer_view, _, producer_representable = (
        pass_utils_module._per_core_view_from_prep(
            producer_prep,
            {producer_head: 16, producer_flat: 2},
        )
    )

    consumer_head, consumer_outer, consumer_inner = sympy.symbols(
        "consumer_head consumer_outer consumer_inner",
        integer=True,
        nonnegative=True,
    )
    consumer_index = 512 * consumer_head + 256 * consumer_outer + consumer_inner
    consumer_prep, _ = _prepare_compound_axis_view(
        {consumer_head: 16, consumer_outer: 2, consumer_inner: 256},
        consumer_index,
    )
    consumer_view, _, consumer_representable = (
        pass_utils_module._per_core_view_from_prep(
            consumer_prep,
            {consumer_head: 16, consumer_outer: 1, consumer_inner: 2},
        )
    )

    assert producer_representable
    assert not consumer_representable
    assert producer_view != consumer_view

    # Splitting the outer term of the same compound coordinate is contiguous:
    # each core group owns one four-stick half of the physical axis.
    _, _, outer_split_representable = pass_utils_module._per_core_view_from_prep(
        consumer_prep,
        {consumer_head: 16, consumer_outer: 2, consumer_inner: 1},
    )
    assert outer_split_representable
