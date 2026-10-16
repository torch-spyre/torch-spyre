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

import sympy
import torch
from torch._inductor.dependencies import MemoryDep, WeakDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise
from torch._inductor.virtualized import V
from torch_spyre._C import ElementArrangement, SpyreTensorLayout, get_device_dtype

import torch_spyre._inductor.pass_utils as pass_utils
from torch_spyre._inductor.patches import _unique_operation_names
from torch_spyre._inductor.pass_utils import (
    forward_valid_elements,
    prepend_dim_valid_elements,
    reorder_valid_elements,
    restickify_valid_elements,
)
from utils_inductor import DEVICE


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
