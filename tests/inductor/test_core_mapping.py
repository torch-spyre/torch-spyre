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

import dataclasses
from functools import partial
import math
from types import SimpleNamespace
from unittest import mock

import pytest
import sympy

import torch_spyre._inductor.codegen.superdsc as superdsc_module
import torch_spyre._inductor.core_mapping as core_mapping_module
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
    derive_partition_mapping,
    derive_operation_mapping,
    finalize_tensor_work_divisions,
    select_unique_partition_division,
)
from torch_spyre._inductor.op_spec import (
    OpSpec,
    TensorArg,
    TensorWorkDivision,
    LX_RELAYOUT_INFO_KEY,
)
from torch_spyre._inductor.pass_utils import PerCoreView, per_core_views_equal
from torch_spyre._inductor.spyre_kernel import simplify_op_spec
from torch_spyre._inductor.views import (
    align_tensors,
)


_CORE_ID = sympy.Symbol("core_id")
_FUSED = sympy.Symbol("fused")

# Literal defaults only: each case supplies its coordinates, strides and owners.
_view_prep = partial(
    pass_utils_module._ViewPrep,
    elems_per_stick=64,
    stick_host_stride=None,
    num_stick_dim=None,
    num_stick=0,
    num_stick_stride=0,
    is_matmul=False,
)


def _coordinates(splits, num_cores, **kwargs):
    dims = sympy.symbols(f"dim_0:{len(splits)}")
    mapping = core_to_slice_mapping(dims, splits, num_cores, **kwargs)
    return _mapping_coordinates(mapping, dims, num_cores)


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
    with pytest.raises(ValueError, match=f"non-integral owner slot {slot} on core 0"):
        core_mapping_module.owner_slots({dim: slot}, {dim: 2}, 2)
    view = PerCoreView(((0, 2),), ((0, slot),), num_cores=2)
    from torch_spyre._inductor.scratchpad.lx_relayout import _core_slices

    with pytest.raises(ValueError, match="non-integral"):
        _core_slices(view, 2)


def test_owner_evaluation_reuse_keeps_domain_and_range_checks():
    dim = sympy.Symbol("dim")
    direct = {dim: _CORE_ID}
    wrapped = {dim: sympy.Mod(_CORE_ID, 4)}
    evaluate = core_mapping_module._owner_at_core
    evaluate.cache_clear()
    assert core_mappings_equal(direct, wrapped, 4)
    misses = evaluate.cache_info().misses
    assert core_mapping_module.owner_slots(direct, {dim: 4}, 4) == tuple(
        {dim: core} for core in range(4)
    )
    assert evaluate.cache_info().misses == misses
    # Reusing earlier points must not hide a difference on a larger domain.
    assert not core_mappings_equal(direct, wrapped, 8)
    with pytest.raises(ValueError, match="outside split 2 on core 2"):
        core_mapping_module.owner_slots(direct, {dim: 2}, 4)
    with pytest.raises(ValueError, match="owner slot -1 outside split 2 on core 0"):
        core_mapping_module.owner_slots({dim: sympy.S.NegativeOne}, {dim: 2}, 4)
    assert not core_mappings_equal(direct, direct, 0)
    evaluate.cache_clear()


def test_owner_evaluation_keeps_short_circuit_order():
    class FailsOnLaterCore(sympy.Function):
        @classmethod
        def eval(cls, value):
            if value == 1:
                raise NotImplementedError("later core must not be evaluated")
            if value == 0:
                return sympy.Integer(0)

    dim = sympy.Symbol("dim")
    assert not core_mappings_equal(
        {dim: FailsOnLaterCore(_CORE_ID)}, {dim: sympy.Integer(1)}, 2
    )


def test_kernel_rejects_lx_allocation_without_physical_ownership():
    """The fault is a placement rejection: preparation demotes and retries."""

    kernel = spyre_kernel_module.SpyreKernel.__new__(spyre_kernel_module.SpyreKernel)
    kernel.current_node = SimpleNamespace(node=object())
    tensor = SimpleNamespace(layout=SimpleNamespace(allocation={"lx": 0}, lx_view=None))
    with (
        mock.patch.object(spyre_kernel_module, "iteration_space", return_value={}),
        pytest.raises(ValueError, match="missing_view has no physical ownership"),
    ):
        kernel.create_tensor_arg(False, "missing_view", tensor)


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


def test_ambiguous_canonical_owner_orders_are_rejected():
    first, second = sympy.symbols("first second")
    reasons = []
    assert (
        select_unique_partition_division(
            (first, second),
            {first: 2, second: 2},
            4,
            lambda _: True,
            rejection_reasons=reasons,
        )
        is None
    )
    assert reasons == ["ambiguous ownership: multiple canonical maps matched"]


def test_late_partition_mapping_repeats_contiguous_owners():
    head = sympy.Symbol("head")
    mapping = derive_partition_mapping((head,), (4,), 32)
    assert _mapping_coordinates(mapping, (head,), 32) == [
        (core // 8,) for core in range(32)
    ]


@pytest.mark.parametrize(
    ("coords", "extent", "split", "sizes"),
    [
        ((_FUSED, _FUSED), 4, 2, (4, 4)),
        ((3 - _FUSED,), 4, 2, (4,)),
        ((sympy.floor((_FUSED + 1) / 4),), 6, 2, (2,)),
        ((2 * _FUSED,), 4, 2, (8,)),
        ((sympy.Mod(_FUSED + 6, 8),), 4, 2, (8,)),
        ((sympy.floor(_FUSED / 98), sympy.Mod(_FUSED, 98)), 196, 2, (2, 98)),
    ],
)
def test_index_regions_match_exact_points(coords, extent, split, sizes):
    coordinates = tuple(
        c.xreplace({_FUSED: core_mapping_module._LOOP_POINT}) for c in coords
    )
    bounds, full = [], True
    width = extent // split
    for part in range(split):
        points = {
            tuple(c.subs(_FUSED, p) for c in coords)
            for p in range(part * width, (part + 1) * width)
        }
        region = tuple(zip(map(min, zip(*points)), map(max, zip(*points))))
        bounds.append(region)
        full &= len(points) == width == math.prod(hi - lo + 1 for lo, hi in region)
    for rectangles in (False, True):
        if rectangles and not full:
            with pytest.raises(ValueError):
                core_mapping_module._loop_regions(
                    extent, coordinates, sizes, split, True
                )
        else:
            assert core_mapping_module._loop_regions(
                extent, coordinates, sizes, split, rectangles
            ) == tuple(bounds)


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


def test_owner_maps_compare_physical_owners_not_sympy_spelling():
    """Equivalent spellings compare equal; unsplit dimensions describe nothing."""

    head, local = sympy.symbols("head local")
    same = _CORE_ID - 4 * sympy.floor(_CORE_ID / 4)
    reordered = sympy.floor(_CORE_ID / 2)
    left = TensorWorkDivision(
        {head: 4, local: 1}, {head: sympy.Mod(_CORE_ID, 4), local: sympy.S.Zero}
    )
    equivalent = TensorWorkDivision({head: 4}, {head: same}, num_cores=4)

    assert left.physical_core_count == 4
    assert left != equivalent and left.same_ownership(equivalent)
    assert not left.same_ownership(
        TensorWorkDivision({head: 4}, {head: reordered}, num_cores=4)
    )

    view = PerCoreView(((0, 4),), ((0, sympy.Mod(_CORE_ID, 4)),), num_cores=8)
    equivalent_view = PerCoreView(
        ((0, 4), (1, 1)), ((0, same), (1, sympy.S.Zero)), num_cores=8
    )
    assert view != equivalent_view and view.same_partition(equivalent_view)
    assert per_core_views_equal(view, equivalent_view)
    assert per_core_views_equal(None, None)
    assert not view.same_partition(
        PerCoreView(((0, 4),), ((0, reordered),), num_cores=8)
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


def _lx_op_spec(op, iteration_space, tensors, divisions, *, certified=False):
    """An operation on LX-resident operands, before alignment.

    ``certified`` marks it as a planner-certified relayout identity.
    """

    args = [
        TensorArg(
            index + 1 < len(tensors),
            index,
            DataFormats.SEN169_FP16,
            list(tensor["size"]),
            list(tensor["coordinates"]),
            {"lx": 0},
            work_division=division,
        )
        for index, (tensor, division) in enumerate(zip(tensors, divisions))
    ]
    op_info = {LX_RELAYOUT_INFO_KEY: True} if certified else {}
    return OpSpec(op, False, dict(iteration_space), args, op_info)


def test_relayouts_are_finished_on_the_operation_split_space():
    """Ownership is settled on the operation's committed split space.

    An ordinary operation keeps its owners; a certified relayout keeps the
    destination's owners, also across core domains; coinciding divisions,
    domains that do not divide the execution domain and splits beyond the
    aligned extent are rejected.
    """

    head = sympy.Symbol("head")
    space = {head: (sympy.Integer(2), 2)}
    tensor = {"size": [2, 64], "coordinates": [head, sympy.S.Zero]}

    def division(split, slot, num_cores):
        return TensorWorkDivision({head: split}, {head: slot}, num_cores=num_cores)

    source = division(2, sympy.Mod(_CORE_ID, 2), 2)
    wide = division(2, sympy.floor(_CORE_ID / 16), 32)

    ordinary = _lx_op_spec("add", space, [tensor, tensor], (source, source))
    simplify_op_spec(ordinary)
    assert core_mappings_equal(
        {head: ordinary.core_id_to_work_slice[head]}, source.core_id_to_work_slice, 2
    )

    def finish(source, destination):
        op_spec = _lx_op_spec(
            "identity", space, [tensor, tensor], (source, destination), certified=True
        )
        simplify_op_spec(op_spec)
        return op_spec

    relayout = finish(source, wide)
    assert set(relayout.core_id_to_work_slice) == set(relayout.iteration_space)
    assert core_mappings_equal(
        relayout.core_id_to_work_slice, wide.core_id_to_work_slice, 32
    )
    with pytest.raises(ValueError, match="ownership collapsed"):
        finish(source, division(2, _CORE_ID - 2 * sympy.floor(_CORE_ID / 2), 2))
    with pytest.raises(ValueError, match="core domains must divide"):
        finish(division(3, sympy.Mod(_CORE_ID, 3), 3), wide)
    with pytest.raises(ValueError, match="split exceeds its aligned extent"):
        finish(source, division(4, sympy.Mod(_CORE_ID, 4), 32))


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
@pytest.mark.parametrize("dim_splits", [(2, 4, 4), (1, 1, 4)])
def test_planner_and_sdsc_use_the_same_mapping(
    monkeypatch, op, reduction_contiguous, dim_splits
):
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
    splits = dict(zip(dims, dim_splits))
    op_spec.iteration_space = {
        dim: (extent, splits[dim])
        for dim, (extent, _) in op_spec.iteration_space.items()
    }
    monkeypatch.setattr(
        pass_utils_module,
        "iteration_space_from_op",
        lambda _: {dim: extent for dim, (extent, _) in op_spec.iteration_space.items()},
    )
    ownership = pass_utils_module.make_iteration_space_ownership(
        FakeComputedBuffer(op), splits
    )
    assert ownership.num_cores == math.prod(dim_splits)
    assert dataclasses.replace(
        ownership, num_cores=None
    ).physical_core_count == math.prod(dim_splits)
    prep = _view_prep(
        iter_space={
            dim: extent for dim, (extent, _) in op_spec.iteration_space.items()
        },
        write_index=dims[0],
        dep_coeff={dims[0]: 1, dims[1]: 2, dims[2]: 0},
        dep_device_coordinates=(dims[0], dims[1]),
        device_size=[2, 4],
        stride_map=[1, 2],
        device_stride_to_dim={1: 0, 2: 1},
        is_matmul=pass_utils_module._is_matmul_op(FakeComputedBuffer(op)),
    )
    planner_view, partial, representable = pass_utils_module._per_core_view_from_prep(
        prep, ownership.work_slices, {dims[2]: dim_splits[2]}, ownership=ownership
    )

    op_spec.core_id_to_work_slice = derive_operation_mapping(
        op_spec.iteration_space,
        contiguous_dim=dims[-1] if reduction_contiguous else None,
    )
    sdsc_spec, renamed = parse_op_spec(op_spec)
    sdsc_output_mapping = {
        device_dim: sdsc_spec.core_id_to_work_slice[renamed[dim]]
        for device_dim, dim in enumerate(dims[:2])
        if splits[dim] > 1
    }
    assert representable
    assert partial
    assert planner_view.num_cores == ownership.num_cores
    assert dict(planner_view.core_to_slot) == sdsc_output_mapping
    if dim_splits[:2] == (1, 1):
        assert planner_view.work_slice_dims == ()
        assert not planner_view.same_partition(PerCoreView((), (), num_cores=1))


def test_flattened_iteration_span_is_not_a_single_axis_view():
    heads, flat = sympy.symbols("heads flat")
    prep = _view_prep(
        iter_space={heads: 16, flat: 512},
        write_index=512 * heads + flat,
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
        device_stride_to_dim={256: 0, 64: 4, 512: 5, 1: 6},
        stick_host_stride=1,
        num_stick_dim=4,
        num_stick=4,
        num_stick_stride=64,
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
    prep = _view_prep(
        iter_space={flat: extent},
        write_index=flat,
        dep_coeff={flat: 1},
        dep_device_coordinates=coordinates,
        device_size=device_size,
        stride_map=stride_map,
        device_stride_to_dim={stride: axis for axis, stride in enumerate(stride_map)},
        stick_host_stride=stick_host_stride,
        num_stick_dim=num_stick_dim,
        num_stick=num_stick,
        num_stick_stride=64 if num_stick else 0,
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


def test_direct_axis_proof_budget_boundary():
    prove = core_mapping_module.direct_axis_ownership_failure
    point = core_mapping_module._LOOP_POINT
    assert prove(65536, 1, point, 65536, 1) is None
    assert prove(65537, 1, point, 65537, 1).startswith("proof limit:")
    assert prove(8, 2, sympy.Mod(point, 4), 8, 2).startswith("ownership mismatch:")


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


@pytest.mark.parametrize("relayout_enabled", [False, True])
def test_reshape_changes_per_core_ownership_within_one_device_axis(
    monkeypatch, relayout_enabled
):
    """A split of an inner term is not a contiguous split of the containing axis.

    This is the Gemma 4 decode geometry: the producer splits a flattened
    512-element dimension into two contiguous 256-element halves, while its
    consumer views that dimension as ``[2, 256]`` and splits the inner 256.
    The latter owns alternating pairs of sticks, not contiguous groups of four.
    """
    # Turning off movement must not turn off the physical-ownership guard.
    monkeypatch.setattr(
        pass_utils_module.config, "lx_planner_relayout", relayout_enabled
    )
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
