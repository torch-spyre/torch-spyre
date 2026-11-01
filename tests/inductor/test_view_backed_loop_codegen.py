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

from types import SimpleNamespace
from unittest import mock

import sympy
import torch
from torch._inductor.ir import FixedLayout

import torch_spyre._inductor.spyre_kernel as spyre_kernel_module
from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.propagate_layouts import _real_layout_matches_op_size
from torch_spyre._inductor.spyre_kernel import SpyreKernel, TensorAccess


def test_rank_changing_mutation_view_does_not_reuse_backing_layout():
    logical_size = [4, 8, 4, 256, 128]
    logical_layout = FixedLayout(
        torch.device("spyre"),
        torch.float16,
        logical_size,
        [1048576, 131072, 32768, 128, 1],
    )
    mutation_layout = SimpleNamespace(
        size=logical_size, target=SimpleNamespace(layout=logical_layout)
    )
    node = SimpleNamespace(
        data=SimpleNamespace(get_size=lambda: logical_size),
        get_layout=lambda: mutation_layout,
    )
    backing_layout = FixedLayout(
        torch.device("spyre"),
        torch.float16,
        [4, 8, 1024, 128],
        [1048576, 131072, 128, 1],
    )

    assert not _real_layout_matches_op_size(node, backing_layout)


def test_rank_preserving_static_mutation_slice_can_reuse_backing_layout():
    logical_size = [2, 128]
    logical_layout = FixedLayout(
        torch.device("spyre"), torch.float16, logical_size, [128, 1], offset=256
    )
    mutation_layout = SimpleNamespace(
        size=logical_size, target=SimpleNamespace(layout=logical_layout)
    )
    node = SimpleNamespace(
        data=SimpleNamespace(get_size=lambda: logical_size),
        get_layout=lambda: mutation_layout,
    )
    backing_layout = FixedLayout(
        torch.device("spyre"), torch.float16, [8, 128], [128, 1]
    )

    assert _real_layout_matches_op_size(node, backing_layout)


def test_rank_preserving_different_size_static_offset_can_reuse_backing_layout():
    """Same rank, different size, but a concrete (symbol-free) offset.

    The op writes a fixed sub-region of the backing buffer (e.g. the copy
    interior of a constant_pad_nd output).  real_layout() is safe to reuse.
    """
    logical_size = [2, 128]
    logical_layout = FixedLayout(
        torch.device("spyre"),
        torch.float16,
        logical_size,
        [128, 1],
        offset=sympy.Integer(512),  # concrete, no free symbols
    )
    mutation_layout = SimpleNamespace(
        size=logical_size, target=SimpleNamespace(layout=logical_layout)
    )
    node = SimpleNamespace(
        data=SimpleNamespace(get_size=lambda: logical_size),
        get_layout=lambda: mutation_layout,
    )
    backing_layout = FixedLayout(
        torch.device("spyre"), torch.float16, [8, 128], [128, 1]
    )

    assert _real_layout_matches_op_size(node, backing_layout)


def test_rank_preserving_different_size_symbolic_offset_cannot_reuse_backing_layout():
    """Same rank, different size, offset contains a loop symbol.

    The op writes a per-iteration tile (e.g. a while_loop body mutation at
    offset 12*u0).  real_layout() covers the whole underlying buffer, not the
    tile this iteration writes, so it must NOT be reused.
    """
    loop_var = sympy.Symbol("u0", integer=True)
    logical_size = [2, 6]
    logical_layout = FixedLayout(
        torch.device("spyre"),
        torch.float16,
        logical_size,
        [6, 1],
        offset=12 * loop_var,  # free symbol → per-iteration placement
    )
    mutation_layout = SimpleNamespace(
        size=logical_size, target=SimpleNamespace(layout=logical_layout)
    )
    node = SimpleNamespace(
        data=SimpleNamespace(get_size=lambda: logical_size),
        get_layout=lambda: mutation_layout,
    )
    backing_layout = FixedLayout(
        torch.device("spyre"), torch.float16, [4, 2, 6], [12, 6, 1]
    )

    assert not _real_layout_matches_op_size(node, backing_layout)


def test_mutation_view_with_no_target_layout_cannot_reuse_backing_layout():
    """Target has no layout attribute (e.g. a SpyreEmptyFallback-like stub).

    getattr(target, 'layout', None) returns None → must return False rather
    than crashing.
    """
    logical_size = [4, 128]
    mutation_layout = SimpleNamespace(
        size=logical_size,
        target=SimpleNamespace(),  # no .layout attribute
    )
    node = SimpleNamespace(
        data=SimpleNamespace(get_size=lambda: logical_size),
        get_layout=lambda: mutation_layout,
    )
    backing_layout = FixedLayout(
        torch.device("spyre"), torch.float16, [8, 128], [128, 1]
    )

    assert not _real_layout_matches_op_size(node, backing_layout)


def test_spliced_loop_symbol_is_removed_from_tensor_base_coordinates():
    loop_var, inner_var = sympy.symbols("u0 d0", integer=True)
    operation = SimpleNamespace(
        dim_hints=[
            DimHint(
                dim_names=["_while_loop"],
                split_count=1,
                loop_var=loop_var,
                is_reduction=False,
                loop_var_range=8,
            )
        ]
    )
    scheduler_node = SimpleNamespace(node=operation)
    device_layout = SimpleNamespace(
        device_size=[64],
        stride_map=[1],
        device_dtype=DataFormats.SEN169_FP16,
        element_arrangement=ElementArrangement.STANDARD,
    )
    layout = SimpleNamespace(
        allocation={"hbm": 0}, lx_view=None, device_layout=device_layout
    )
    tensor = TensorAccess("input", 64 * loop_var + inner_var, layout)
    seen = {}

    def capture_coordinates(_layout, index, iteration_space, indirect_sizes, **_):
        seen["index"] = index
        seen["iteration_space"] = iteration_space
        seen["indirect_sizes"] = indirect_sizes
        return [index]

    kernel = SpyreKernel()
    kernel.current_node = scheduler_node
    with (
        mock.patch.object(spyre_kernel_module._spyre_config, "ktir_emitter", False),
        mock.patch.object(
            spyre_kernel_module,
            "iteration_space",
            return_value={inner_var: (sympy.Integer(64), 1)},
        ),
        mock.patch.object(
            spyre_kernel_module,
            "alignment_coordinates",
            side_effect=capture_coordinates,
        ),
        mock.patch.object(
            spyre_kernel_module, "work_division_from_view", return_value=None
        ),
        mock.patch.object(
            kernel, "_general_tile_advance", return_value=sympy.Integer(64)
        ),
    ):
        arg = kernel.create_tensor_arg(False, "output", tensor)

    assert seen["index"] == inner_var
    assert seen["indirect_sizes"] == {}
    assert arg.device_coordinates == [inner_var]
    assert arg.device_tile_advance_expr == 64
