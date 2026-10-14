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
- INV-2: Buffers are never removed from ``graph.buffers`` (append-only).
- INV-3: ``name_to_buffer`` is consistent with ``buffers`` and
  ``removed_buffers``.
- INV-4: ``name_to_op`` is consistent with ``operations``.
- INV-5: Operations only read from defined buffers (in ``name_to_buffer``,
  ``graph_inputs``, or ``constants``).
- INV-6: Graph outputs reference valid buffers.
- INV-7: ``name_to_users`` keys reference defined or removed names.
- INV-8: No live operation reads from a removed buffer.

Planned future work:

- INV-9: Synthetic index bounds checking (validate that ops do not read
  past the end of a tensor).
"""

from __future__ import annotations

import logging

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    NoneAsConstantBuffer,
    ShapeAsConstantBuffer,
)

logger = logging.getLogger(__name__)


class GraphValidationError(ValueError):
    """Raised when a GraphLowering invariant is violated."""

    def __init__(
        self,
        invariant: str,
        detail: str,
        pass_name: str = "",
        *,
        violations: list[GraphValidationError] | None = None,
    ) -> None:
        prefix = f"[after {pass_name}] " if pass_name else ""
        msg = f"{prefix}GraphLowering validation failed: {invariant}. {detail}"
        super().__init__(msg)
        self.invariant = invariant
        self.detail = detail
        self.pass_name = pass_name
        self.violations: list[GraphValidationError] = violations or []


def validate_graph(
    graph: GraphLowering,
    *,
    pass_name: str = "",
    prev_buffer_count: int | None = None,
) -> None:
    """Validate GraphLowering invariants.

    Args:
        graph: The GraphLowering instance to validate.
        pass_name: Name of the pass that just ran (for error messages).
        prev_buffer_count: If provided, the length of ``graph.buffers``
            before the pass ran.  Used by INV-2 to verify that the buffer
            list is append-only.

    Raises:
        GraphValidationError: If any invariant is violated.
    """
    if not isinstance(graph, GraphLowering):
        return

    violations: list[GraphValidationError] = []
    violations.extend(_check_buffer_name_uniqueness(graph, pass_name))
    if prev_buffer_count is not None:
        violations.extend(
            _check_buffer_list_immutability(graph, prev_buffer_count, pass_name)
        )
    violations.extend(_check_name_to_buffer_consistency(graph, pass_name))
    violations.extend(_check_name_to_op_consistency(graph, pass_name))
    violations.extend(_check_reads_from_defined_buffers(graph, pass_name))
    violations.extend(_check_graph_outputs_valid(graph, pass_name))
    violations.extend(_check_name_to_users_consistency(graph, pass_name))

    if not violations:
        return

    if len(violations) == 1:
        raise violations[0]

    suffix = f" after {pass_name}" if pass_name else ""
    n = len(violations)
    lines = [f"{n} invariant violations detected{suffix}:"]
    for i, v in enumerate(violations, 1):
        lines.append(f"  {i}. {v.invariant}: {v.detail}")
    raise GraphValidationError(
        invariant="multiple violations",
        detail="\n".join(lines),
        pass_name=pass_name,
        violations=violations,
    )


# ------------------------------------------------------------------
# INV-1: Buffer name uniqueness
# ------------------------------------------------------------------


def _check_buffer_name_uniqueness(
    graph: GraphLowering, pass_name: str
) -> list[GraphValidationError]:
    violations: list[GraphValidationError] = []
    seen: dict[str, int] = {}
    for i, buf in enumerate(graph.buffers):
        name = buf.get_name()
        if name in seen:
            violations.append(
                GraphValidationError(
                    "INV-1: buffer names must be unique",
                    f"Buffer name {name!r} appears at indices "
                    f"{seen[name]} and {i} in graph.buffers",
                    pass_name,
                )
            )
        seen[name] = i
    return violations


# ------------------------------------------------------------------
# INV-2: Buffer list immutability (append-only)
# ------------------------------------------------------------------


def _check_buffer_list_immutability(
    graph: GraphLowering, prev_count: int, pass_name: str
) -> list[GraphValidationError]:
    violations: list[GraphValidationError] = []
    current_count = len(graph.buffers)
    if current_count < prev_count:
        violations.append(
            GraphValidationError(
                "INV-2: buffers removed from graph.buffers",
                f"graph.buffers had {prev_count} entries before "
                f"{pass_name} but now has {current_count}",
                pass_name,
            )
        )
    return violations


# ------------------------------------------------------------------
# INV-3: name_to_buffer consistency
# ------------------------------------------------------------------


def _check_name_to_buffer_consistency(
    graph: GraphLowering, pass_name: str
) -> list[GraphValidationError]:
    violations: list[GraphValidationError] = []
    removed = graph.removed_buffers

    # Forward check: every name_to_buffer entry must hold a buffer whose
    # get_name() matches the dict key.
    for name, buf in graph.name_to_buffer.items():
        buf_name = buf.get_name()
        if buf_name != name:
            violations.append(
                GraphValidationError(
                    "INV-3: name_to_buffer key does not match buffer name",
                    f"name_to_buffer[{name!r}] holds a buffer whose "
                    f"get_name() returns {buf_name!r}",
                    pass_name,
                )
            )
        if name in removed:
            logger.debug(
                "INV-3: name_to_buffer retains entry for removed buffer %r "
                "(after %s) — pass should clean up name_to_buffer",
                name,
                pass_name,
            )

    # Reverse check: every live buffer in graph.buffers must have an entry
    # in name_to_buffer.
    for buf in graph.buffers:
        name = buf.get_name()
        if name not in removed and name not in graph.name_to_buffer:
            violations.append(
                GraphValidationError(
                    "INV-3: name_to_buffer missing entry for live buffer",
                    f"Buffer {name!r} is in graph.buffers (not removed) "
                    f"but has no entry in name_to_buffer",
                    pass_name,
                )
            )

    return violations


# ------------------------------------------------------------------
# INV-4: name_to_op consistency
# ------------------------------------------------------------------


def _check_name_to_op_consistency(
    graph: GraphLowering, pass_name: str
) -> list[GraphValidationError]:
    violations: list[GraphValidationError] = []
    for op in graph.operations:
        op_name = op.get_operation_name()
        if op_name is None:
            violations.append(
                GraphValidationError(
                    "INV-4: operation has None operation_name",
                    f"Operation {op!r} in graph.operations has operation_name=None",
                    pass_name,
                )
            )
            continue

        if op_name not in graph.name_to_op:
            violations.append(
                GraphValidationError(
                    "INV-4: name_to_op missing entry for live operation",
                    f"Operation {op_name!r} is in graph.operations but "
                    f"has no entry in name_to_op",
                    pass_name,
                )
            )
        elif graph.name_to_op[op_name] is not op:
            violations.append(
                GraphValidationError(
                    "INV-4: name_to_op points to wrong object",
                    f"name_to_op[{op_name!r}] points to a different "
                    f"object than the operation in graph.operations",
                    pass_name,
                )
            )

    return violations


# ------------------------------------------------------------------
# INV-5 + INV-8: reads from defined / non-removed buffers
# ------------------------------------------------------------------


def _defined_names(graph: GraphLowering) -> set[str]:
    """Build the set of buffer names that an operation may legally read."""
    names = set(graph.name_to_buffer.keys())
    names |= set(graph.graph_inputs.keys())
    names |= set(graph.constants.keys())
    if hasattr(graph, "torchbind_constants"):
        names |= set(graph.torchbind_constants.keys())
    return names


def _check_reads_from_defined_buffers(
    graph: GraphLowering, pass_name: str
) -> list[GraphValidationError]:
    violations: list[GraphValidationError] = []
    defined = _defined_names(graph)
    removed = graph.removed_buffers

    for op in graph.operations:
        try:
            rw = op.get_read_writes()
        except Exception as e:
            logger.debug(
                "validate_graph: skipping read/write check for %r: %s",
                op.get_name(),
                e,
            )
            continue
        for dep in rw.reads:
            if not isinstance(dep, MemoryDep):
                continue
            name = dep.name
            if name in removed:
                violations.append(
                    GraphValidationError(
                        "INV-8: live operation reads from removed buffer",
                        f"Operation {op.get_name()!r} reads buffer "
                        f"{name!r} which is in removed_buffers",
                        pass_name,
                    )
                )
            elif name not in defined:
                violations.append(
                    GraphValidationError(
                        "INV-5: operation reads from undefined buffer",
                        f"Operation {op.get_name()!r} reads buffer "
                        f"{name!r} which is not in name_to_buffer, "
                        f"graph_inputs, or constants",
                        pass_name,
                    )
                )

    return violations


# ------------------------------------------------------------------
# INV-6: graph outputs reference valid buffers
# ------------------------------------------------------------------


def _check_graph_outputs_valid(
    graph: GraphLowering, pass_name: str
) -> list[GraphValidationError]:
    if not hasattr(graph, "graph_outputs") or graph.graph_outputs is None:
        return []

    violations: list[GraphValidationError] = []
    defined = _defined_names(graph)

    for i, node in enumerate(graph.graph_outputs):
        if isinstance(node, (NoneAsConstantBuffer, ShapeAsConstantBuffer)):
            continue
        name = node.get_name()
        if name not in defined:
            violations.append(
                GraphValidationError(
                    "INV-6: graph output references undefined buffer",
                    f"graph_outputs[{i}] has name {name!r} which is "
                    f"not in name_to_buffer, graph_inputs, or constants",
                    pass_name,
                )
            )

    return violations


# ------------------------------------------------------------------
# INV-7: name_to_users consistency
# ------------------------------------------------------------------


def _check_name_to_users_consistency(
    graph: GraphLowering, pass_name: str
) -> list[GraphValidationError]:
    if not hasattr(graph, "name_to_users"):
        return []
    violations: list[GraphValidationError] = []
    defined = _defined_names(graph) | graph.removed_buffers
    for name in graph.name_to_users:
        if name not in defined:
            violations.append(
                GraphValidationError(
                    "INV-7: name_to_users references undefined name",
                    f"name_to_users contains {name!r} which is not in "
                    f"name_to_buffer, graph_inputs, constants, "
                    f"or removed_buffers",
                    pass_name,
                )
            )
    return violations
