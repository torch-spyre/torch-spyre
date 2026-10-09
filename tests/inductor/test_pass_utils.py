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
from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise, Scatter
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor import pass_utils
from torch_spyre._inductor.wrapper import noop_simplify_loops_impl


@pytest.mark.parametrize("folded_axis", ["new", "old", None])
def test_restickify_does_not_discard_folded_cache_pages(folded_axis):
    from torch_spyre._C import DataFormats, SpyreTensorLayout

    page, token, head, feature = sympy.symbols(
        "page token head feature", integer=True, nonnegative=True
    )
    if folded_axis == "old":
        size = [64, 2, 20, 128]
        stride = [5120, 2560, 128, 1]
        stl = SpyreTensorLayout(
            [64, 2, 40, 64], [5120, 2560, 64, 1], DataFormats.SEN169_FP16
        )
        host_coords = [token, head, page, feature]
        device_coords = [
            token,
            head,
            2 * page + sympy.floor(feature / 64),
            sympy.Mod(feature, 64),
        ]
    else:
        size = [20, 64, 2, 64]
        stride = [8192, 128, 64, 1]
        host_coords = [page, token, head, feature]
        if folded_axis == "new":
            stl = SpyreTensorLayout(
                [1280, 2, 1, 64], [128, 64, 64, 1], DataFormats.SEN169_FP16
            )
            device_coords = [64 * page + token, head, sympy.S.Zero, feature]
        else:
            stl = SpyreTensorLayout(
                [20, 64, 2, 1, 64],
                [8192, 128, 64, 64, 1],
                DataFormats.SEN169_FP16,
            )
            device_coords = [page, token, head, sympy.S.Zero, feature]
    target = pass_utils.compute_restickify_target_layout(
        stl,
        FixedLayout(torch.device("cpu"), torch.float16, size, stride),
        sympy.Mod(token, 64),
        host_coords,
        device_coords,
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
