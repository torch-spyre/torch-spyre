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
import types

import pytest
import sympy
import torch
from torch._inductor.dependencies import (
    MemoryDep,
    StarDep,
    WeakDep,
    extract_read_writes,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise, Scatter
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch_spyre._C import ElementArrangement, SpyreTensorLayout, get_device_dtype

import torch_spyre._inductor.pass_utils as pass_utils
from torch_spyre._inductor.patches import _unique_operation_names
from torch_spyre._inductor.pass_utils import (
    collapse_valid_elements,
    forward_valid_elements,
    prepend_dim_valid_elements,
    reorder_valid_elements,
    restickify_valid_elements,
)
from torch_spyre._inductor.propagate_layouts import _compact_broadcast_device_dims
from torch_spyre._inductor.wrapper import noop_simplify_loops_impl
from utils_inductor import DEVICE


@pytest.mark.parametrize("folded_axis", ["new", "old", None])
def test_restickify_does_not_discard_folded_cache_pages(folded_axis):
    from torch_spyre._C import DataFormats, SpyreTensorLayout
    from torch_spyre._inductor.propagate_layouts import PropArg

    page, token, head, feature = sympy.symbols(
        "page token head feature", integer=True, nonnegative=True
    )
    if folded_axis == "old":
        size = [64, 2, 20, 128]
        stride = [5120, 2560, 128, 1]
        stl = SpyreTensorLayout(
            [64, 2, 40, 64],
            [5120, 2560, 64, 1],
            DataFormats.SEN169_FP16,
            {(0,): 64, (1,): 2, (2, 3): 40},
        )
        host_coords = [token, head, page, feature]
        device_coords = [
            token,
            head,
            2 * page + sympy.floor(feature / 64),
            sympy.Mod(feature, 64),
        ]
        dep = MemoryDep(
            "x",
            5120 * token + 2560 * head + 128 * page + feature,
            (token, head, page, feature),
            (64, 2, 20, 128),
        )
    else:
        size = [20, 64, 2, 64]
        stride = [8192, 128, 64, 1]
        host_coords = [page, token, head, feature]
        dep = MemoryDep(
            "x",
            8192 * page + 128 * token + 64 * head + feature,
            (page, token, head, feature),
            (20, 64, 2, 64),
        )
        if folded_axis == "new":
            stl = SpyreTensorLayout(
                [1280, 2, 1, 64],
                [128, 64, 64, 1],
                DataFormats.SEN169_FP16,
                {(0,): 1280, (1,): 2, (2, 3): 1},
            )
            device_coords = [64 * page + token, head, sympy.S.Zero, feature]
        else:
            stl = SpyreTensorLayout(
                [20, 64, 2, 1, 64],
                [8192, 128, 64, 64, 1],
                DataFormats.SEN169_FP16,
                {(0,): 20, (1,): 64, (2,): 2, (3, 4): 64},
            )
            device_coords = [page, token, head, sympy.S.Zero, feature]
    host = FixedLayout(torch.device("cpu"), torch.float16, size, stride)
    arg = PropArg(dep, host, [stl])
    target = pass_utils.compute_restickify_target_layout(
        stl,
        host,
        sympy.Mod(token, 64),
        host_coords,
        device_coords,
        arg=arg,
    )
    if folded_axis is not None:
        assert target is None
    else:
        assert target is not None
        assert list(target.device_size) == [20, 64, 2, 1, 64]
        assert list(target.stride_map) == [8192, 1, 64, 8192, 128]


def test_scatter_index_discovery_preserves_load_order_after_wrapping():
    from torch._inductor.ir import Scatter

    def indexer(index):
        row, col = index
        row_index = V.ops.indirect_indexing(V.ops.load("z_rows", row), 192, check=False)
        col_index = V.ops.indirect_indexing(
            V.ops.load("a_columns", col), 128, check=False
        )
        return [row_index, col_index]

    scatter = ComputedBuffer(
        name="scatter",
        layout=FixedLayout(torch.device("cpu"), torch.float32, [192, 128], [128, 1]),
        data=Scatter(
            device=torch.device("cpu"),
            dtype=torch.float32,
            ranges=[32, 64],
            inner_fn=lambda index: V.ops.load("values", index[0] * 64 + index[1]),
            output_indexer=indexer,
        ),
    )
    scatter.operation_name = "scatter_op"
    graph = types.SimpleNamespace(
        sizevars=SizeVarAllocator(),
        name_to_buffer={"scatter": scatter},
        name_to_op={"scatter_op": scatter},
    )
    with V.set_graph_handler(graph):
        assert pass_utils._scatter_index_buf_names_ordered(scatter) == [
            "z_rows",
            "a_columns",
        ]
        scatter = pass_utils.redirect_computed_buffer_reads(
            scatter,
            {"z_rows": "new_rows", "a_columns": "new_columns"},
            [scatter],
            pass_name="test",
        )
        assert pass_utils._scatter_index_buf_names_ordered(scatter) == [
            "new_rows",
            "new_columns",
        ]
        subs, _ = pass_utils._build_indirect_store_subs(scatter)
    assert [value.base.name for value in subs.values()] == ["new_rows", "new_columns"]


@pytest.mark.parametrize("indexed_axis", [0, 1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_scatter_iteration_space_preserves_indirect_axes(indexed_axis, normalize):
    """A scatter must visit all source rows even when its write hides an axis."""
    sizes = [4, 8, 16]

    def indexer(index):
        result = list(index)
        result[indexed_axis] = V.ops.indirect_indexing(
            V.ops.load("indices", index[indexed_axis]), 32, check=False
        )
        return result

    scatter = ComputedBuffer(
        name="scatter",
        layout=FixedLayout(torch.device("cpu"), torch.float32, [32, 32, 32]),
        data=Scatter(
            device=torch.device("cpu"),
            dtype=torch.float32,
            ranges=sizes,
            inner_fn=lambda index: V.ops.load(
                "values", index[0] * 128 + index[1] * 16 + index[2]
            ),
            output_indexer=indexer,
        ),
    )
    sizevars = SizeVarAllocator()
    sizevars._simplify_loops_impl = types.MethodType(noop_simplify_loops_impl, sizevars)
    with V.set_graph_handler(types.SimpleNamespace(sizevars=sizevars)):
        pre_schedule = pass_utils.iteration_space_from_op(scatter)
        rw = extract_read_writes(
            scatter.get_store_function(), sizes, normalize=normalize
        )
    rw.reads = OrderedSet(
        [StarDep("destination"), WeakDep("mutation", "buffer"), *rw.reads]
    )
    node = types.SimpleNamespace(node=scatter, read_writes=rw)
    scheduled = pass_utils.iteration_space(node)

    assert [(str(sym), size) for sym, size in pre_schedule.items()] == [
        (f"d{i}", size) for i, size in enumerate(sizes)
    ]
    prefix = "c" if normalize else "d"
    assert [(str(sym), size) for sym, size in scheduled.items()] == [
        (f"{prefix}{i}", size) for i, size in enumerate(sizes)
    ]
    if indexed_axis == 2:
        assert len(next(iter(rw.writes)).ranges) == 2


def test_scatter_iteration_space_without_memory_deps():
    from torch._inductor.dependencies import ReadWrites

    rw = ReadWrites(
        reads=OrderedSet([StarDep("destination")]),
        writes=OrderedSet([WeakDep("mutation", "buffer")]),
        index_exprs=OrderedSet(),
    )
    assert pass_utils._scatter_iteration_space(rw) == {}


def test_reduction_iteration_space_ignores_weak_dependencies(monkeypatch):
    """Mutation ordering edges describe no loop dimensions."""

    class FakeReduction:
        pass

    output_dim = sympy.Symbol("d0")
    reduction_dim = sympy.Symbol("r0")
    write = types.SimpleNamespace(ranges={output_dim: 4})
    read = MemoryDep(
        "input",
        output_dim * 8 + reduction_dim,
        (output_dim, reduction_dim),
        (4, 8),
    )
    node = types.SimpleNamespace(
        node=types.SimpleNamespace(data=FakeReduction()),
        read_writes=types.SimpleNamespace(
            writes=[write],
            reads=[WeakDep("mutation", "buffer"), read],
        ),
    )
    monkeypatch.setattr(pass_utils, "Reduction", FakeReduction)

    assert pass_utils.iteration_space(node) == {output_dim: 4, reduction_dim: 8}


def test_late_operation_registration_does_not_reuse_a_removed_slot():
    """Late graph edits must not derive names from the shortened op list."""

    old_ops = [types.SimpleNamespace(operation_name=f"op{i}") for i in range(4)]
    graph = types.SimpleNamespace(
        # Model a pass that removed op1 but retained its name registry entry.
        operations=[old_ops[0], old_ops[2], old_ops[3]],
        name_to_op={op.operation_name: op for op in old_ops},
        qualify_name=lambda name: name,
    )
    inserted = types.SimpleNamespace(operation_name=None)

    name = pass_utils.register_operation_after_graph_edit(graph, inserted)

    assert name == "op4"
    assert graph.operations[-1] is inserted
    assert graph.name_to_op["op3"] is old_ops[3]
    assert graph.name_to_op["op4"] is inserted


def test_lowering_during_a_spyre_compile_does_not_reuse_a_live_name():
    """A pass that lowers through ``run_node`` registers via upstream
    ``register_operation``, which would name the op ``op3`` here."""
    old_ops = [types.SimpleNamespace(operation_name=f"op{i}") for i in range(4)]
    graph = types.SimpleNamespace(
        operations=[old_ops[0], old_ops[2], old_ops[3]],
        name_to_op={op.operation_name: op for op in old_ops},
        qualify_name=lambda name: name,
    )
    upstream = GraphLowering.register_operation

    with _unique_operation_names():
        name = GraphLowering.register_operation(
            graph, types.SimpleNamespace(operation_name=None)
        )

    assert name == "op4"
    assert graph.name_to_op["op3"] is old_ops[3]
    assert GraphLowering.register_operation is upstream


def test_dedup_freed_name_is_not_reissued_during_spyre_compile():
    """A name freed by dedup_and_promote_constants must not be reissued.

    The sequence that triggers the bug:
    1. splice_while_loops shortens graph.operations (5 ops -> 3 ops).
       op3 and op4 are still live and still in name_to_op.
    2. _unique_operation_names is active for the whole compile.
    3. split_multi_ops registers a new op (gets op5); operations is now len 4.
    4. dedup_and_promote_constants removes one op from operations (len->3) and
       pops op3 from name_to_op entirely.
    5. _maybe_scratchpad_planning registers another op: len(operations)==3 so
       it tries op3 first, finds it not in name_to_op, and reissues it.
    """
    old_ops = [types.SimpleNamespace(operation_name=f"op{i}") for i in range(5)]
    graph = types.SimpleNamespace(
        # operations was shortened to 3 entries by splice_while_loops;
        # op3 and op4 are still live and in name_to_op.
        operations=[old_ops[0], old_ops[1], old_ops[2]],
        name_to_op={op.operation_name: op for op in old_ops},
        qualify_name=lambda name: name,
    )

    with _unique_operation_names():
        # Step 3: split_multi_ops registers a new op (op5); operations grows to 4.
        first = types.SimpleNamespace(operation_name=None)
        GraphLowering.register_operation(graph, first)
        assert first.operation_name == "op5"

        # Step 4: dedup removes one op from operations (back to 3) and pops op3.
        graph.operations.remove(old_ops[2])
        graph.name_to_op.pop("op3")

        # Step 5: scratchpad registers another op. len(operations)==3, so it
        # tries op3 first — which is no longer in name_to_op. Bug: reissues op3.
        inserted = types.SimpleNamespace(operation_name=None)
        name = GraphLowering.register_operation(graph, inserted)

    assert name == "op6", (
        f"expected op6 but got {name!r} (freed name 'op3' must not be reissued)"
    )


def test_replace_computed_buffer_body_preserves_body_origins():
    """A dataclass body rewrite must retain the FX provenance used by layouts."""

    def inner_fn(index):
        return index[0]

    old_data = Pointwise(
        device=torch.device("cpu"),
        dtype=torch.float32,
        inner_fn=inner_fn,
        ranges=[4],
    )
    old_origin = object()
    replacement_origin = object()
    old_data.origins.add(old_origin)
    new_data = dataclasses.replace(old_data, inner_fn=inner_fn)
    new_data.origins.add(replacement_origin)
    op = ComputedBuffer(
        name="buf0",
        layout=FixedLayout(torch.device("cpu"), torch.float32, [4], [1]),
        data=old_data,
    )
    op.operation_name = "op0"
    graph = types.SimpleNamespace(
        name_to_buffer={"buf0": op},
        name_to_op={"op0": op},
    )
    operations = [op]

    with V.set_graph_handler(graph):
        replacement = pass_utils.replace_computed_buffer_body(
            op,
            new_data,
            operations,
            pass_name="test",
        )

    assert set(replacement.data.origins) == {old_origin, replacement_origin}


def test_compute_restickify_target_layout_moves_to_existing_size1_dim():
    """Case (a): an existing size-1 host dim hosts the stick, no synthesis needed."""
    d1 = sympy.Symbol("d1")
    host_layout = FixedLayout(torch.device("spyre"), torch.float16, [1, 64], [64, 1])
    # Host coords: dim0 is the constant size-1 dim, dim1 varies with d1.
    ic = [sympy.S.Zero, d1]
    # Device layout with the stick currently on dim1 (idc[-1] == d1); the
    # outer dim's coordinate is the same constant as the host's size-1 dim.
    stl = SpyreTensorLayout([1, 64], [64, 1], torch.float16, [0, 1])
    idc = [sympy.S.Zero, d1]

    result = pass_utils.compute_restickify_target_layout(
        stl, host_layout, sympy.S.Zero, ic, idc
    )

    assert result is not None
    # The stick actually moved: the leading device dim goes from a size-1
    # placeholder (stl's old outer slot for the size-1 host dim) to the
    # elems-per-stick extent, reflecting the restickify onto that dim.
    assert stl.device_size[0] == 1
    assert result.device_size[0] == 64


def test_compute_restickify_target_layout_synthesizes_when_no_size1_dim_exists():
    """Case (b): no size-1 host dim exists, so the size-1 stick dim is synthetic."""
    d0, d1 = sympy.symbols("d0 d1", integer=True, nonnegative=True)
    dep = MemoryDep("x", d0 * 64 + d1, (d0, d1), (4, 64))
    host_layout = FixedLayout(torch.device("spyre"), torch.float16, [4, 64], [64, 1])
    stl = SpyreTensorLayout([4, 64], [64, 1], torch.float16, [0, 1])
    ic = pass_utils.host_coordinates(host_layout, dep, None)
    idc = pass_utils.device_coordinates(stl, dep, None)

    result = pass_utils.compute_restickify_target_layout(
        stl, host_layout, sympy.S.Zero, ic, idc
    )

    assert result is not None
    # One element per stick: the old stick's 64 elements become an outer dim
    # and the stick maps to no host stride.
    assert list(result.device_size) == [64, 4, 64]
    assert list(result.stride_map) == [1, 64, -1]


_FP16 = get_device_dtype(torch.float16)


def _stl(device_size, stride_map, valid_elements, ea=ElementArrangement.STANDARD):
    return SpyreTensorLayout(device_size, stride_map, _FP16, valid_elements, ea)


def test_forward_valid_elements_returns_copy():
    """Returns a copy; mutating the result does not affect the original."""
    _ds = [4, 3, 64]
    stl = _stl(_ds, [192, 64, 1], {(0,): 4, (1, 2): 3})
    result = forward_valid_elements(stl)
    assert result == {(0,): 4, (1, 2): 3}
    assert stl.valid_elements == {(0,): 4, (1, 2): 3}


def test_forward_valid_elements_independent_groups():
    _ds = [8, 4, 64]
    stl = _stl(_ds, [256, 64, 1], {(0,): 8, (1, 2): 4})
    assert forward_valid_elements(stl) == {(0,): 8, (1, 2): 4}


def test_forward_valid_elements_padded_valid_count_unchanged():
    """Padded allocation (device_size[1]=512) does not affect stored valid count."""
    _ds = [1, 512, 64]
    stl = _stl(_ds, [2400, 1, 60], {(0,): 1, (1, 2): 40})
    assert forward_valid_elements(stl) == {(0,): 1, (1, 2): 40}


def test_reorder_valid_elements_swap_two_outer_dims():
    """new_order=[1,0,2]: group (1,2) moves to (0,2); group (0,) to (1,)."""
    _ds = [4, 8, 64]
    stl = _stl(_ds, [512, 64, 1], {(0,): 4, (1, 2): 8})
    assert reorder_valid_elements(stl, new_order=[1, 0, 2]) == {(1,): 4, (0, 2): 8}


def test_reorder_valid_elements_identity_permutation():
    _ds = [4, 8, 64]
    stl = _stl(_ds, [512, 64, 1], {(0,): 4, (1, 2): 8})
    assert reorder_valid_elements(stl, new_order=[0, 1, 2]) == {(0,): 4, (1, 2): 8}


def test_reorder_valid_elements_cyclic_rotation():
    """new_order=[1,2,0,3]: old dim 0→new 2, old 1→new 0, old dim (2,3)→new (1,3)."""
    _ds = [2, 3, 5, 64]
    stl = _stl(_ds, [960, 320, 64, 1], {(0,): 2, (1,): 3, (2, 3): 5})
    assert reorder_valid_elements(stl, new_order=[1, 2, 0, 3]) == {
        (2,): 2,
        (0,): 3,
        (1, 3): 5,
    }


def test_reorder_valid_elements_grouped_dims_remapped():
    """Multi-dim group (2,3): swap dims 0 and 1 → group stays (2,3), outer dims swap."""
    _ds = [2, 4, 3, 64]
    stl = _stl(_ds, [768, 192, 64, 1], {(0,): 2, (1,): 4, (2, 3): 3})
    assert reorder_valid_elements(stl, new_order=[1, 0, 2, 3]) == {
        (1,): 2,
        (0,): 4,
        (2, 3): 3,
    }


def test_prepend_dim_valid_elements_shifts_existing_indices():
    _ds = [4, 64]
    stl = _stl(_ds, [64, 1], {(0, 1): 4})
    assert prepend_dim_valid_elements(stl, new_dim_size=1) == {(0,): 1, (1, 2): 4}


def test_prepend_dim_valid_elements_grouped_dims_shift():
    """Multi-dim group (0,1) shifts to (1,2)."""
    _ds = [3, 64]
    stl = _stl(_ds, [64, 1], {(0, 1): 3})
    assert prepend_dim_valid_elements(stl, new_dim_size=1) == {(0,): 1, (1, 2): 3}


def test_prepend_dim_valid_elements_new_dim_size_eps():
    """Sparse expand: new outermost dim carries eps valid elements."""
    _ds = [4, 8, 64]
    stl = _stl(_ds, [512, 64, 1], {(0,): 4, (1, 2): 8})
    assert prepend_dim_valid_elements(stl, new_dim_size=2) == {
        (0,): 2,
        (1,): 4,
        (2, 3): 8,
    }


def test_prepend_dim_valid_elements_single_dim():
    _ds = [64]
    stl = _stl(_ds, [1], {(0,): 64})
    assert prepend_dim_valid_elements(stl, new_dim_size=1) == {(0,): 1, (1,): 64}


def test_restickify_valid_elements_basic_swap():
    """Stick moves from group (1,2) to group (0,): valid counts exchange."""
    _ds = [4, 8, 64]
    stl = _stl(_ds, [512, 64, 1], {(0,): 4, (1, 2): 8})
    result = restickify_valid_elements(
        stl,
        old_sd_outer_dim=1,
        old_sd_host_size=8,
        new_sd_outer_dim=0,
        new_sd_host_size=4,
    )
    assert result == {(0,): 8, (1, 2): 4}


def test_restickify_valid_elements_unrelated_dims_unchanged():
    """Dims not touching old/new stick groups are forwarded unchanged."""
    _ds = [2, 4, 8, 64]
    stl = _stl(_ds, [2048, 512, 64, 1], {(0,): 2, (1,): 4, (2, 3): 8})
    result = restickify_valid_elements(
        stl,
        old_sd_outer_dim=2,
        old_sd_host_size=8,
        new_sd_outer_dim=1,
        new_sd_host_size=4,
    )
    assert result == {(0,): 2, (1,): 8, (2, 3): 4}


def test_restickify_valid_elements_outermost_becomes_new_stick():
    """Stick moves to the outermost dim; the unrelated middle dim is unchanged."""
    _ds = [2, 4, 8, 64]
    stl = _stl(_ds, [2048, 512, 64, 1], {(0,): 2, (1,): 4, (2, 3): 8})
    result = restickify_valid_elements(
        stl,
        old_sd_outer_dim=2,
        old_sd_host_size=8,
        new_sd_outer_dim=0,
        new_sd_host_size=2,
    )
    assert result == {(0,): 8, (1,): 4, (2, 3): 2}


def test_restickify_valid_elements_preserves_all_keys():
    _ds = [3, 5, 7, 64]
    stl = _stl(_ds, [2240, 448, 64, 1], {(0,): 3, (1,): 5, (2, 3): 7})
    result = restickify_valid_elements(
        stl,
        old_sd_outer_dim=2,
        old_sd_host_size=7,
        new_sd_outer_dim=1,
        new_sd_host_size=5,
    )
    assert result == {(0,): 3, (1,): 7, (2, 3): 5}


def test_restickify_valid_elements_single_stick_fallback():
    """Single-stick fallback: size-1 outer dim must not steal the new stick count.

    src = SpyreTensorLayout([1, 32, 64], [2048, 64, 1], fp16, dim_order=[1, 0, 2])
    tgt = same with dim_order=[2, 0, 1]
    restickify_valid_elements(src, old_sd_outer_dim=0, old_sd_host_size=64,
                              new_sd_outer_dim=2, new_sd_host_size=32)
    must equal tgt.valid_elements, not give dim 0 a count of 32.
    """
    src = SpyreTensorLayout([1, 32, 64], [2048, 64, 1], torch.float16, [1, 0, 2])
    tgt = SpyreTensorLayout([1, 32, 64], [2048, 64, 1], torch.float16, [2, 0, 1])
    result = restickify_valid_elements(
        src,
        old_sd_outer_dim=0,
        old_sd_host_size=64,
        new_sd_outer_dim=2,
        new_sd_host_size=32,
    )
    assert result == tgt.valid_elements


def test_restickify_valid_elements_same_dim_returns_empty():
    """When old and new stick outer dims coincide, return an empty map."""
    stl = _stl([4, 8, 64], [512, 64, 1], {(0,): 4, (1, 2): 8})
    result = restickify_valid_elements(
        stl,
        old_sd_outer_dim=1,
        old_sd_host_size=8,
        new_sd_outer_dim=1,
        new_sd_host_size=8,
    )
    assert result == {}


def test_restickify_valid_elements_oracle_transpose():
    """Ordinary transpose matches postmath's oracle check.

    src = SpyreTensorLayout([2, 256, 128], [32768, 128, 1], fp16, dim_order=[0, 1, 2])
    tgt = same with dim_order=[0, 2, 1]
    restickify_valid_elements(src, 1, 128, 0, 256) == tgt.valid_elements
    """
    src = SpyreTensorLayout([2, 256, 128], [32768, 128, 1], torch.float16, [0, 1, 2])
    tgt = SpyreTensorLayout([2, 256, 128], [32768, 128, 1], torch.float16, [0, 2, 1])
    result = restickify_valid_elements(
        src,
        old_sd_outer_dim=1,
        old_sd_host_size=128,
        new_sd_outer_dim=0,
        new_sd_host_size=256,
    )
    assert result == tgt.valid_elements


def test_compact_broadcast_device_dims_caps_valid_elements():
    """Broadcast dim shrunk to size 1: its valid_elements count must be capped to 1.

    SpyreTensorLayout([8, 100], [0, 1], fp16, [0, 1]) has device_size=[2, 8, 64],
    stride_map=[64, 0, 1], valid_elements={(0, 2): 100, (1,): 8}.
    After compaction dim 1 is size 1, so its count must become 1, not 8.
    """
    stl = SpyreTensorLayout([8, 100], [0, 1], torch.float16, [0, 1])
    result = _compact_broadcast_device_dims(stl)
    assert result.device_size[1] == 1, "broadcast dim should be collapsed to 1"
    assert result.valid_elements[(1,)] == 1, (
        "collapsed broadcast dim count must be 1, not the original 8"
    )


def test_compact_broadcast_device_dims_no_broadcast_unchanged():
    """Non-broadcast STL: _compact_broadcast_device_dims returns it unchanged."""
    stl = SpyreTensorLayout([4, 8, 64], [512, 64, 1], torch.float16, [0, 1, 2])
    result = _compact_broadcast_device_dims(stl)
    assert result is stl, "no-broadcast STL should be returned as-is"


def test_collapse_valid_elements_basic():
    """Two dims collapse into one; renumbering shifts the stick group."""
    _ds = [2, 4, 8, 64]
    stl = _stl(_ds, [2048, 512, 64, 1], {(0,): 2, (1,): 4, (2, 3): 8})
    result = collapse_valid_elements(stl, outer_dims=[0, 1], keep=0)
    # old dims 0 and 1 merge into new dim 0 (size 2*4=8); old dims 2,3 → 1,2.
    assert result == {(0,): 2 * 4, (1, 2): 8}


def test_collapse_valid_elements_renumbers_stick_group():
    """Stick group indices shift down when intermediate dims are removed."""
    _ds = [2, 3, 5, 64]
    stl = _stl(_ds, [960, 320, 64, 1], {(0,): 2, (1,): 3, (2, 3): 5})
    result = collapse_valid_elements(stl, outer_dims=[0, 1], keep=0)
    # old dim 0 and 1 → new dim 0; old dim 2 → new dim 1; old stick 3 → new dim 2.
    assert result == {(0,): 2 * 3, (1, 2): 5}


def test_collapse_valid_elements_unrelated_dim_preserved():
    """A dim not in outer_dims but between them gets renumbered correctly."""
    _ds = [2, 5, 3, 8, 64]
    stl = _stl(
        _ds,
        [5 * 3 * 8 * 64, 3 * 8 * 64, 8 * 64, 64, 1],
        {(0,): 2, (1,): 5, (2,): 3, (3, 4): 8},
    )
    # Collapse dims 0 and 2 into dim 0 (keep=0); dim 1 and dims 3,4 renumber.
    result = collapse_valid_elements(stl, outer_dims=[0, 2], keep=0)
    # remove={2}; dims 0,2 merge → new (0,) val=6; (1,)→(1,); (3,4)→(2,3).
    assert result == {(0,): 2 * 3, (1,): 5, (2, 3): 8}


# Propagation through compiled ops: send tensors to device, compile the op,
# check that the output tensor's valid_elements match the input's.


def _to_spyre(t: torch.Tensor) -> torch.Tensor:
    return t.to(DEVICE)


def test_compiled_relu_output_valid_elements_match_input():
    """Compiled relu (stick-aligned input): output valid_elements == input."""
    x_dev = _to_spyre(torch.randn(40, 128, dtype=torch.float16))
    inp_ve = x_dev.device_tensor_layout().valid_elements

    out = torch.compile(torch.relu)(x_dev)

    assert out.device_tensor_layout().valid_elements == inp_ve


def test_compiled_relu_padded_output_valid_elements_match_input():
    """Compiled relu (non-stick-aligned: 60 < 64): output valid_elements == input."""
    # [40, 60]: last dim 60 is not a multiple of 64, so the stick is partially filled.
    x_dev = _to_spyre(torch.randn(40, 60, dtype=torch.float16))
    inp_ve = x_dev.device_tensor_layout().valid_elements

    out = torch.compile(torch.relu)(x_dev)

    assert out.device_tensor_layout().valid_elements == inp_ve


def test_compiled_add_output_valid_elements_match_input():
    """Compiled add (stick-aligned inputs): output valid_elements == input."""
    a_dev = _to_spyre(torch.randn(40, 128, dtype=torch.float16))
    b_dev = _to_spyre(torch.randn(40, 128, dtype=torch.float16))
    inp_ve = a_dev.device_tensor_layout().valid_elements

    out = torch.compile(torch.add)(a_dev, b_dev)

    assert out.device_tensor_layout().valid_elements == inp_ve


def test_compiled_add_padded_output_valid_elements_match_input():
    """Compiled add (non-stick-aligned: 60 < 64): output valid_elements == input."""
    a_dev = _to_spyre(torch.randn(40, 60, dtype=torch.float16))
    b_dev = _to_spyre(torch.randn(40, 60, dtype=torch.float16))
    inp_ve = a_dev.device_tensor_layout().valid_elements

    out = torch.compile(torch.add)(a_dev, b_dev)

    assert out.device_tensor_layout().valid_elements == inp_ve


def test_compiled_transpose_restickify_valid_elements():
    """Transpose moving the stick dim triggers restickify; output valid_elements
    must reflect the new stick host size.

    Input:  [2, 256, 128] fp16 — stick on H=128 (last host dim).
    After transpose(1, 2): [2, 128, 256] — stick now covers S=256.
    Output stick group valid count must be S=256, not H=128.
    """
    B, S, H = 2, 256, 128
    x_dev = _to_spyre(torch.randn(B, S, H, dtype=torch.float16))

    out = torch.compile(lambda x: x.transpose(1, 2).clone())(x_dev)

    out_stl = out.device_tensor_layout()
    stick_dim = len(out_stl.device_size) - 1
    stick_key = next(k for k in out_stl.valid_elements if stick_dim in k)
    assert out_stl.valid_elements[stick_key] == S


def test_compiled_transpose_restickify_padded_valid_elements():
    """Same as above with a non-stick-aligned S=60; stick group valid count must
    be 60, not the padded device size of 64.

    Input:  [2, 60, 128] fp16 — stick on H=128.
    After transpose(1, 2): [2, 128, 60] — stick now covers S=60.
    """
    B, S, H = 2, 60, 128
    x_dev = _to_spyre(torch.randn(B, S, H, dtype=torch.float16))

    out = torch.compile(lambda x: x.transpose(1, 2).clone())(x_dev)

    out_stl = out.device_tensor_layout()
    stick_dim = len(out_stl.device_size) - 1
    stick_key = next(k for k in out_stl.valid_elements if stick_dim in k)
    assert out_stl.valid_elements[stick_key] == S
