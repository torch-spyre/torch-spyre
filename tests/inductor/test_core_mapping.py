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
import math
from types import SimpleNamespace

import pytest
import sympy

import torch_spyre._inductor.codegen.superdsc as superdsc_module
import torch_spyre._inductor.pass_utils as pass_utils_module
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
    derive_partition_mapping,
    finalize_tensor_work_divisions,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg, TensorWorkDivision
from torch_spyre._inductor.spyre_kernel import simplify_op_spec
from torch_spyre._inductor.views import (
    align_tensors,
    align_tensors_pure,
)


def _coordinates(splits, num_cores, **kwargs):
    dims = sympy.symbols(f"dim_0:{len(splits)}")
    mapping = core_to_slice_mapping(dims, splits, num_cores, **kwargs)
    core_id = sympy.Symbol("core_id")
    return [
        tuple(int(mapping[dim].subs(core_id, core)) for dim in dims)
        for core in range(num_cores)
    ]


def test_default_mapping_preserves_existing_core_order():
    one_grid = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert _coordinates((2, 3), 12) == one_grid * 2


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


def test_late_partition_mapping_repeats_contiguous_owners():
    head = sympy.Symbol("head")
    mapping = derive_partition_mapping((head,), (4,), 32)
    assert _mapping_coordinates(mapping, (head,), 32) == [
        (core // 8,) for core in range(32)
    ]


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


def test_late_mapping_rejects_geometry_that_does_not_fill_groups():
    h, query = sympy.symbols("h query")
    with pytest.raises(ValueError, match="does not match operation split"):
        derive_core_mapping(
            (h, query),
            (4, 8),
            32,
            grouped_splits={h: 2},
        )


def test_final_tensor_ownership_is_derived_from_aligned_buffer_geometry():
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
        {shared: sympy.Mod(sympy.floor(core_id / 2), 2)},
        num_cores=4,
    )


def test_shared_lx_buffer_keeps_owners_across_different_operation_dims():
    producer_extra, producer_shared = sympy.symbols("producer_extra producer_shared")
    consumer_shared, consumer_extra = sympy.symbols("consumer_shared consumer_extra")
    core_id = sympy.Symbol("core_id")
    owners = sympy.Mod(core_id, 2)
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


def test_final_tensor_ownership_requires_a_buffer_core_domain():
    shared = sympy.Symbol("shared")
    core_id = sympy.Symbol("core_id")

    with pytest.raises(ValueError, match="physical core domain"):
        finalize_tensor_work_divisions(
            {shared: (8, 2)},
            [
                TensorWorkDivision(
                    {shared: 2},
                    {shared: sympy.Mod(core_id, 2)},
                )
            ],
        )


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
        aligned_iteration_space=op_spec.iteration_space,
    )
    # A preceding validation preview must neither consume nor change the input
    # subsequently used by codegen.
    align_tensors_pure(captured)

    captured_path = copy.deepcopy(op_spec)
    ordinary_path = copy.deepcopy(op_spec)
    simplify_op_spec(captured_path, alignment_inputs=captured)
    simplify_op_spec(ordinary_path)

    assert captured_path == ordinary_path


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
        prep, ({512: 16, 1: 2}, {})
    )

    assert not representable
    assert not partial
    assert not view.work_slice_dims


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
    with pass_utils_module.V.set_graph_handler(graph):
        prep = pass_utils_module._prepare_per_core_view(
            object(),
            dep,
            "buf",
            parts=(iter_space, index, index),
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
