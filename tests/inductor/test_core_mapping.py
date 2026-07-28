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

import math

import pytest
import sympy

import torch_spyre._inductor.codegen.superdsc as superdsc_module
import torch_spyre._inductor.pass_utils as pass_utils_module
from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import parse_op_spec
from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_OP,
)
from torch_spyre._inductor.core_mapping import core_to_slice_mapping
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _coordinates(splits, num_cores, **kwargs):
    dims = sympy.symbols(f"dim_0:{len(splits)}")
    mapping = core_to_slice_mapping(dims, splits, num_cores, **kwargs)
    core_id = sympy.Symbol("core_id")
    return [
        tuple(int(mapping[str(dim)].subs(core_id, core)) for dim in dims)
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
    monkeypatch.setattr(
        pass_utils_module, "apply_splits_from_index_coeff", lambda *_: splits
    )
    prep = pass_utils_module._ViewPrep(
        iter_space=op_spec.iteration_space,
        write_index=dims[0],
        read_index=dims[-1],
        dep_coeff={dims[0]: 1, dims[1]: 2, dims[2]: 0},
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
        prep, ({1: 2, 2: 4}, {3: 4})
    )

    sdsc_spec, renamed = parse_op_spec(op_spec)
    sdsc_output_mapping = {
        device_dim: sdsc_spec.core_id_to_work_slice[str(renamed[dim])]
        for device_dim, dim in enumerate(dims[:2])
    }
    assert representable
    assert dict(planner_view.core_to_slot) == sdsc_output_mapping
