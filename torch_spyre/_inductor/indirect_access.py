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

from typing import Callable

from sympy import Symbol

from .op_spec import IndirectAccess, OpSpec, TensorArg


def has_index_load(arg: TensorArg) -> bool:
    """Return True if any of arg's device_coordinates contains an IndirectAccess
    node."""
    return any(
        coord.has(IndirectAccess)
        for coord in arg.device_coordinates
        if hasattr(coord, "has")
    )


def get_index_load_names(arg: TensorArg) -> set[str]:
    """Return the set of tensor names referenced by IndirectAccess nodes in
    arg's coordinates."""
    names: set[str] = set()
    for coord in arg.device_coordinates:
        if not hasattr(coord, "atoms"):
            continue
        for node in coord.atoms(IndirectAccess):
            names.add(str(node.args[0]))
    return names


def is_indirect_value_tensor(arg: TensorArg) -> bool:
    """Return True if this tensor is accessed via indirect indexing (has
    IndirectAccess in coords)."""
    return has_index_load(arg)


def is_index_tensor(arg: TensorArg, op_spec: OpSpec) -> bool:
    """Return True if this tensor's name is referenced by an IndirectAccess in
    any other arg."""
    name = getattr(arg, "name", "")
    if not name:
        return False
    for other in op_spec.args:
        if other is arg:
            continue
        if name in get_index_load_names(other):
            return True
    return False


def get_index_tensor_for_value(
    op_spec: OpSpec, value_arg: TensorArg
) -> TensorArg | None:
    """Find the index TensorArg referenced by an IndirectAccess in
    value_arg's coordinates."""
    names = get_index_load_names(value_arg)
    for arg in op_spec.args:
        if getattr(arg, "name", "") in names:
            return arg
    return None


def get_indirect_stride_idx(arg: TensorArg) -> int | None:
    """Find the stride_idx (from right, 0-indexed) of the IndirectAccess in
    this arg's coordinates.

    Returns the position counting from the right (where 0 is the rightmost element).
    """
    for idx, coord in enumerate(reversed(arg.device_coordinates)):
        if isinstance(coord, IndirectAccess):
            return idx
    return None


def get_indirect_dim_symbols(
    value_arg: TensorArg, index_arg: TensorArg, symbol_mapping: dict
) -> list[Symbol]:
    """Extract free symbols from the index tensor's device coordinates.

    Iterates right-to-left through coordinates (excluding stick), collecting
    free symbols in the order they appear.
    """
    seen: set[Symbol] = set()
    ordered: list[Symbol] = []
    for i in range(len(index_arg.device_coordinates) - 2, -1, -1):
        coord = index_arg.device_coordinates[i]
        if hasattr(coord, "free_symbols"):
            expr = coord.subs(symbol_mapping)
            for sym in expr.free_symbols:
                if sym not in seen:
                    seen.add(sym)
                    ordered.append(sym)
    return ordered


def get_value_tensor_idx_for_index(op_spec: OpSpec, index_arg_idx: int) -> int:
    """Return the position of the value tensor accessed by the given index tensor.

    Returns -1 if no corresponding value tensor is found.
    """
    index_arg = op_spec.args[index_arg_idx]
    index_name = getattr(index_arg, "name", "")
    if not index_name:
        return -1
    for j, arg in enumerate(op_spec.args):
        if index_name in get_index_load_names(arg):
            return j
    return -1


def _get_index_tensor_device_size_at(
    index_arg: TensorArg,
    stride_idx: int,
) -> int | None:
    """Return index_arg.device_size at the same stride position, or None if
    out of range."""
    pos = -stride_idx - 2
    if abs(pos) <= len(index_arg.device_size):
        return index_arg.device_size[pos]
    return None


def _get_index_active_dims_for_value(
    index_arg: "TensorArg",
    symbol_mapping: dict,
) -> list[Symbol]:
    """Return the ordered list of free symbols (active dims) in index_arg's coords."""
    seen: set[Symbol] = set()
    active: list[Symbol] = []
    for coord in index_arg.device_coordinates:
        if not hasattr(coord, "free_symbols"):
            continue
        for sym in coord.subs(symbol_mapping).free_symbols:
            if sym not in seen:
                seen.add(sym)
                active.append(sym)
    return active


def compute_indirect_max_dim_sizes(
    tensor_idx: int,
    dim: Symbol,
    stick_dim: Symbol | None,
    stride_idx: int,
    original_dev_dim_size: int,
    op_spec: OpSpec,
    symbol_mapping: dict,
    index_tensor_indices: set[int],
    index_active_dims: dict,
    logger: object,
) -> int:
    """Compute max_dim_size for a tensor dimension in an indirect access operation.

    Value tensor:
      - stick dim → -1 (always dynamic)
      - dimension not in index tensor → -1 (data dimension, full range)
      - dimension in index tensor, multi-dim index, not the last active dim →
        idx_size (bounded by the index tensor's size along that axis)
      - dimension in index tensor, last (or only) active dim → 1 (indirect)

    Index tensor:
      - always -1

    Output tensor:
      - always -1
    """
    arg = op_spec.args[tensor_idx]

    if is_indirect_value_tensor(arg):
        if dim == stick_dim:
            return -1
        index_arg = get_index_tensor_for_value(op_spec, arg)
        if index_arg is None:
            return -1
        indirect_dims = get_indirect_dim_symbols(arg, index_arg, symbol_mapping)
        if dim not in indirect_dims:
            return -1
        idx_size = _get_index_tensor_device_size_at(index_arg, stride_idx)
        if idx_size is None:
            return -1
        active_dims = _get_index_active_dims_for_value(index_arg, symbol_mapping)
        if len(active_dims) > 1 and dim != active_dims[-1]:
            return idx_size
        return 1

    elif tensor_idx in index_tensor_indices:
        return -1

    # Output tensor
    return -1


def get_indirect_layout_label(
    tensor_idx: int,
    index_tensor_indices: set[int],
    layouts: dict,
    dim_order: list,
    effective_stick: object,
    stick_size: int,
    layout_labels: list[str],
    get_layout_label_func: Callable,
    logger: object,
) -> str:
    """Get layout label for a tensor in an indirect access operation.

    Index tensors → KERNEL_IDX
    Value/Output tensors → OUTPUT
    """
    if tensor_idx in index_tensor_indices:
        label = "KERNEL_IDX"
        if label not in layouts:
            layouts[label] = {
                "dim_order": dim_order,
                "stick_dim_order": effective_stick,
                "stick_size": stick_size,
            }
        logger.debug(f"Tensor {tensor_idx}: KERNEL_IDX layout (index tensor)")
        return label

    # For Value/Output tensors
    label = "OUTPUT"
    if label not in layouts:
        layouts[label] = {
            "dim_order": dim_order,
            "stick_dim_order": effective_stick,
            "stick_size": stick_size,
        }
    logger.debug(f"Tensor {tensor_idx}: OUTPUT layout (indirect access)")
    return label
