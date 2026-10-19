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

import torch_spyre._inductor.pass_utils as pass_utils
from torch_spyre._inductor.patches import _unique_operation_names


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
