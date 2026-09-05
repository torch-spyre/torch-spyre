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


"""GraphLowering invariant validation.

Call ``validate_graph(graph, pass_name=...)`` after a compiler pass to
catch consistency violations early. Gated by
``config.validate_graph_invariants``.

Invariants checked:

- INV-1: Buffer names in ``graph.buffers`` are unique.
- INV-3: ``name_to_buffer`` is consistent with ``buffers`` and
  ``removed_buffers``.
- INV-4: ``name_to_op`` is consistent with ``operations``.
- INV-5: Operations only read from defined buffers (in ``name_to_buffer``
  or ``graph_inputs``).
- INV-6: Graph outputs reference valid buffers.
- INV-8: No live operation reads from a removed buffer.
"""

from __future__ import annotations

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    NoneAsConstantBuffer,
    ShapeAsConstantBuffer,
)


class GraphValidationError(ValueError):
    """Raised when a GraphLowering invariant is violated."""

    def __init__(self, invariant: str, detail: str, pass_name: str = "") -> None:
        prefix = f"[after {pass_name}] " if pass_name else ""
        msg = f"{prefix}GraphLowering validation failed: {invariant}. {detail}"
        super().__init__(msg)
        self.invariant = invariant
        self.pass_name = pass_name


def validate_graph(
    graph: GraphLowering,
    *,
    pass_name: str = "",
) -> None:
    """Validate GraphLowering invariants.

    Args:
        graph: The GraphLowering instance to validate.
        pass_name: Name of the pass that just ran (for error messages).

    Raises:
        GraphValidationError: If any invariant is violated.
    """
    _check_buffer_name_uniqueness(graph, pass_name)
    _check_name_to_buffer_consistency(graph, pass_name)
    _check_name_to_op_consistency(graph, pass_name)
    _check_reads_from_defined_buffers(graph, pass_name)
    _check_graph_outputs_valid(graph, pass_name)


# ------------------------------------------------------------------
# INV-1: Buffer name uniqueness
# ------------------------------------------------------------------


def _check_buffer_name_uniqueness(graph: GraphLowering, pass_name: str) -> None:
    seen: dict[str, int] = {}
    for i, buf in enumerate(graph.buffers):
        name = buf.get_name()
        if name in seen:
            raise GraphValidationError(
                "INV-1: buffer names must be unique",
                f"Buffer name {name!r} appears at indices "
                f"{seen[name]} and {i} in graph.buffers",
                pass_name,
            )
        seen[name] = i


# ------------------------------------------------------------------
# INV-3: name_to_buffer consistency
# ------------------------------------------------------------------


def _check_name_to_buffer_consistency(graph: GraphLowering, pass_name: str) -> None:
    removed = graph.removed_buffers

    for buf in graph.buffers:
        name = buf.get_name()
        if name in removed:
            if name in graph.name_to_buffer:
                raise GraphValidationError(
                    "INV-3: name_to_buffer stale entry for removed buffer",
                    f"Buffer {name!r} is in removed_buffers but still "
                    f"present in name_to_buffer",
                    pass_name,
                )
            continue

        if name not in graph.name_to_buffer:
            raise GraphValidationError(
                "INV-3: name_to_buffer missing entry for live buffer",
                f"Buffer {name!r} is in graph.buffers (not removed) "
                f"but has no entry in name_to_buffer",
                pass_name,
            )

        registered = graph.name_to_buffer[name]
        if registered is not buf:
            raise GraphValidationError(
                "INV-3: name_to_buffer points to wrong buffer object",
                f"name_to_buffer[{name!r}] points to a different "
                f"buffer object than graph.buffers[{name!r}]",
                pass_name,
            )


# ------------------------------------------------------------------
# INV-4: name_to_op consistency
# ------------------------------------------------------------------


def _check_name_to_op_consistency(graph: GraphLowering, pass_name: str) -> None:
    for op in graph.operations:
        op_name = op.get_operation_name()
        if op_name is None:
            raise GraphValidationError(
                "INV-4: operation has None operation_name",
                f"Operation {op!r} in graph.operations has operation_name=None",
                pass_name,
            )

        if op_name not in graph.name_to_op:
            raise GraphValidationError(
                "INV-4: name_to_op missing entry for live operation",
                f"Operation {op_name!r} is in graph.operations but "
                f"has no entry in name_to_op",
                pass_name,
            )


# ------------------------------------------------------------------
# INV-5 + INV-8: reads from defined / non-removed buffers
# ------------------------------------------------------------------


def _defined_names(graph: GraphLowering) -> set[str]:
    """Build the set of buffer names that an operation may legally read."""
    names = set(graph.name_to_buffer.keys())
    names |= set(graph.graph_inputs.keys())
    return names


def _check_reads_from_defined_buffers(graph: GraphLowering, pass_name: str) -> None:
    defined = _defined_names(graph)
    removed = graph.removed_buffers

    for op in graph.operations:
        try:
            rw = op.get_read_writes()
        except Exception:
            continue
        for dep in rw.reads:
            if not isinstance(dep, MemoryDep):
                continue
            name = dep.name
            if name in removed:
                raise GraphValidationError(
                    "INV-8: live operation reads from removed buffer",
                    f"Operation {op.get_name()!r} reads buffer "
                    f"{name!r} which is in removed_buffers",
                    pass_name,
                )
            if name not in defined:
                raise GraphValidationError(
                    "INV-5: operation reads from undefined buffer",
                    f"Operation {op.get_name()!r} reads buffer "
                    f"{name!r} which is not in name_to_buffer or "
                    f"graph_inputs",
                    pass_name,
                )


# ------------------------------------------------------------------
# INV-6: graph outputs reference valid buffers
# ------------------------------------------------------------------


def _check_graph_outputs_valid(graph: GraphLowering, pass_name: str) -> None:
    if not hasattr(graph, "graph_outputs") or graph.graph_outputs is None:
        return

    defined = _defined_names(graph)

    for i, node in enumerate(graph.graph_outputs):
        if isinstance(node, (NoneAsConstantBuffer, ShapeAsConstantBuffer)):
            continue
        name = node.get_name()
        if name not in defined:
            raise GraphValidationError(
                "INV-6: graph output references undefined buffer",
                f"graph_outputs[{i}] has name {name!r} which is "
                f"not in name_to_buffer or graph_inputs",
                pass_name,
            )
