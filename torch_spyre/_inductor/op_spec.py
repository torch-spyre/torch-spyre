# Copyright 2025 The Torch-Spyre Authors.
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


from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from sympy import Symbol, Expr, Function
from torch_spyre._C import DataFormats
import torch


class IndirectAccess(Function):
    """Sympy function: IndirectAccess(tensor_name) — runtime index read from that tensor at the current iteration point.

    Used in TensorArg.device_coordinates to encode indirect access: as a coordinate of an
    input arg (gather) or an output arg (scatter).
    IndexedBase was not used because sympify('arg1_1[i]') fails: the parser reconstructs
    arg1_1 as a Symbol, and Symbol.__getitem__ raises TypeError.
    """

    @classmethod
    def eval(cls, name):  # noqa: ARG003
        return None  # keep unevaluated


# --- Source-to-kernel provenance schema -------------------------------------
# These dataclasses live here with the other IR-op schema types; the logic that
# builds them from Inductor IR lives in ``provenance.py``. They serialize into
# the current OpSpec/SuperDSC JSON path, with field names lined up to MLIR
# location attributes so the future KTIR (MLIR) migration is a low-friction
# serializer change rather than a redesign.


@dataclasses.dataclass(frozen=True)
class SourceLoc:
    """Structured source location attached to provenance handles.

    Serialized into the current OpSpec/SuperDSC JSON path; the field names
    mirror MLIR ``FileLineColRange`` (start/end line and column) so a future
    KTIR (MLIR) migration maps 1:1 rather than requiring a reshape.
    """

    file: str
    start_line: int
    start_col: int = 0
    end_line: int | None = None
    end_col: int | None = None

    def to_str(self) -> str:
        return f"{self.file}:{self.start_line}:{self.start_col}"

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DebugHandle:
    """Source-to-kernel provenance handle.

    Nestable to map onto MLIR locations: ``NameLoc(aten_op) -> SourceLoc``,
    ``fused_from -> FusedLoc``, ``ir_chain -> CallSiteLoc`` lineage.

    A ``None`` ``source`` or ``aten_op`` is a *normal, expected* value, not a
    missing-data error: when an op fuses origins from several distinct source
    lines there is no single honest headline, so both are set to ``None`` and the
    full set is preserved in ``fused_from``. Consumers should fall back to
    ``fused_from`` rather than treating a null headline as an error.
    """

    id: int
    source: SourceLoc | None
    aten_op: str | None
    ir_chain: tuple[str, ...]
    fused_from: tuple["DebugHandle", ...] = ()
    fusion_context: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            # id is serialized as a string: a 63-bit value exceeds JS
            # Number.MAX_SAFE_INTEGER (2**53-1), and JSON.parse would round it to
            # float64 before a consumer could act. The dataclass field stays int
            # for the MLIR/protobuf mapping (a separate serializer).
            "id": str(self.id),
            "source": self.source.to_dict() if self.source is not None else None,
            "aten_op": self.aten_op,
            "ir_chain": list(self.ir_chain),
            "fused_from": [h.to_dict() for h in self.fused_from],
            "fusion_context": self.fusion_context,
        }


@dataclasses.dataclass
class TensorArg:
    """
    A class representing a Tensor argument to an OpSpec

    Attributes:
        is_input: Is the Tensor used as an input to the operation?
        arg_index: The index of the Tensor in the argument array of the Kernel.
        device_dtype: The device dtype of the tensor elements.
        device_size: The device size (as per SpyreTensorLayout) of the Tensor
        device_coordinates: The sympy Exprs that describe how elements in the Tensor are accessed.
                Free variables in device_coordinates refer to entries in the OpSpec's iteration_space.
        allocation: If present, the offset in scratchpad memory assigned to the Tensor.
    """

    is_input: bool
    arg_index: int
    device_dtype: DataFormats
    device_size: list[int]
    device_coordinates: list[Expr]
    allocation: Any
    per_tile_fixed: bool = False
    name: str | None = None


@dataclasses.dataclass
class OpSpec:
    """
    A class representing a single operation to perform on the device

    Attributes:
        op: The name of the operation.
        is_reduction: Is the operation a reduction?
        iteration_space: The iteration space of the operation. The values are tuples of (range, work_division).
        args: The input and output arguments to the operation.
        op_info: A dictionary of auxiliary information whose content is operation-specific.
        tiled_symbols: Per-loop-level iteration-space symbols, innermost first.
            ``tiled_symbols[0]`` lists the symbols tiled by the innermost enclosing
            loop; ``tiled_symbols[1]`` lists those tiled by the next-outer loop; etc.
            Empty for ops not inside any loop.
            **Invariant:** every enclosing loop level must have an entry, even if
            empty (``[]``).  An empty entry means the op is loop-invariant at that
            level.  This keeps level indices aligned with nesting depth so that
            ``compile_op_spec``'s reversal maps each level to the correct
            ``loop_var_depth`` index in ``_collect_affine_maps``.
            The unroller reads ``tiled_symbols[0]`` at each level and removes it
            from the list after processing, leaving outer-level entries intact.
            The bundle path (compile_op_spec / generate_sdsc) reverses this list to
            outermost-first and builds per-level affine.apply stride maps, mapping
            each level's strides to the correct loop variable by explicit index.
    """

    op: str
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]
    tiled_symbols: list[list[Symbol]] = dataclasses.field(default_factory=list)
    # Maps PyTorch symbol name (e.g. 's97') -> (max, granularity) bounds.
    # Populated by compute_symbolic_bounds during
    # create_op_spec; empty for concrete dims.
    symbolic_dim_bounds: dict[str, tuple[int, int]] = dataclasses.field(
        default_factory=dict
    )
    debug_handle: DebugHandle | None = None


@dataclasses.dataclass
class UnimplementedOp:
    op: str


@dataclasses.dataclass
class LoopSpec:
    """A counted loop whose body is a sequence of ops, possibly nested.

    Attributes:
        count: Trip count of the loop. May be a symbolic shape expression.
        body: The operations to execute each iteration. Each element may be
            an OpSpec, UnimplementedOp, or a nested LoopSpec.

    Each OpSpec in the body carries its own ``tiled_symbols`` list identifying
    which of its iteration-space symbols are tiled by the loop that directly
    contains it.  The unroller reads these per-op symbols rather than a shared
    list, so ops with different iteration-space layouts in the same loop are
    each advanced by the correct stride.
    """

    count: Expr
    # list[OpSpec | UnimplementedOp | LoopSpec], typed as Any to accommodate
    # the two distinct UnimplementedOp types (op_spec vs spyre_kernel).
    body: list[Any]


def spyre_constant_tensor(const_val, device, dtype=torch.float16):
    return torch.tensor(const_val, dtype=dtype).to(device)


def find_unimplemented(specs: list) -> UnimplementedOp | None:
    """Return the first UnimplementedOp in specs (recursing into LoopSpec), or None."""
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            return entry
        if isinstance(entry, LoopSpec):
            found = find_unimplemented(entry.body)
            if found is not None:
                return found
    return None


def format_op_spec_list(specs: list, indent: int = 0) -> str:
    """Format an op spec list for structured logging output.

    Uses an explicit stack to avoid recursion-depth issues with deeply
    nested LoopSpecs.
    """
    lines: list[str] = []
    stack: list[tuple[list, int, int]] = [(specs, indent, 0)]
    while stack:
        current_specs, cur_indent, idx = stack.pop()
        if idx >= len(current_specs):
            continue
        # Push remainder back for later processing.
        stack.append((current_specs, cur_indent, idx + 1))
        item = current_specs[idx]
        prefix = "  " * cur_indent
        if isinstance(item, LoopSpec):
            lines.append(f"{prefix}LoopSpec(count={item.count})")
            lines.append(f"{prefix}  body=[")
            # Push a sentinel to close the body bracket after children.
            stack.append(([_LoopClose(prefix)], cur_indent, 0))
            # Push the body for processing at deeper indent.
            stack.append((item.body, cur_indent + 2, 0))
        elif isinstance(item, OpSpec):
            it_space_str = ", ".join(
                f"{k}: ({v[0]}, {v[1]})" for k, v in item.iteration_space.items()
            )
            lines.append(
                f"{prefix}OpSpec(op={item.op!r}, "
                f"is_reduction={item.is_reduction}, "
                f"iteration_space={{{it_space_str}}})"
            )
            for arg in item.args:
                lines.append(
                    f"{prefix}  TensorArg("
                    f"{'input' if arg.is_input else 'output'}, "
                    f"arg_index={arg.arg_index}, "
                    f"device_size={arg.device_size}, "
                    f"device_coordinates={arg.device_coordinates}, "
                    f"allocation={arg.allocation})"
                )
            if item.tiled_symbols:
                lines.append(f"{prefix}  tiled_symbols={item.tiled_symbols}")
            if item.symbolic_dim_bounds:
                lines.append(
                    f"{prefix}  symbolic_dim_bounds={item.symbolic_dim_bounds}"
                )
        elif isinstance(item, UnimplementedOp):
            lines.append(f"{prefix}UnimplementedOp(op={item.op!r})")
        elif isinstance(item, _LoopClose):
            lines.append(f"{item.prefix}  ]")
        else:
            lines.append(f"{prefix}{item!r}")
    return "\n".join(lines)


class _LoopClose:
    """Sentinel used by format_op_spec_list to emit closing brackets."""

    __slots__ = ("prefix",)

    def __init__(self, prefix: str):
        self.prefix = prefix
