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

import dataclasses
import math
from typing import Any

from sympy import Integer, Symbol, Expr, Mod, floor

from torch_spyre._C import DataFormats
from torch_spyre._inductor.constants import (
    IDENTITY_OP,
    INPUT_DIM_LABELS,
    OUTPUT_DIM_LABELS,
    LAYOUT_LABELS,
    MATMUL_DIM_LABELS,
    MATMUL_LAYOUT_LABELS,
    SEGMENT_OFFSETS,
)
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.op_spec import TensorArg

from .compute_ops import generate_sdsc

logger = get_inductor_logger("codegen.superdsc")


@dataclasses.dataclass
class SDSCArgs:
    layout: str
    data_format: DataFormats
    scales: dict[Symbol, Any]
    strides: dict[Symbol, Any]
    offsets: dict[Symbol, Any]
    max_dim_sizes: dict[Symbol, Any]
    allocation: dict[str, Any]
    start_address: int | Symbol
    backGap: dict[Symbol, int]

    def __str__(self) -> str:
        scales = ", ".join(f"{k}={v}" for k, v in self.scales.items())
        strides = ", ".join(f"{k}={v}" for k, v in self.strides.items())
        offsets = ", ".join(f"{k}={v}" for k, v in self.offsets.items())
        max_dim_sizes = ", ".join(f"{k}={v}" for k, v in self.max_dim_sizes.items())
        allocation = ", ".join(f"{k}={v}" for k, v in self.allocation.items())
        return (
            f"SDSCArgs(\n"
            f"  layout={self.layout},\n"
            f"  data_format={self.data_format.name},\n"
            f"  scales=[{scales}],\n"
            f"  strides=[{strides}],\n"
            f"  offsets=[{offsets}],\n"
            f"  max_dim_sizes=[{max_dim_sizes}],\n"
            f"  allocation=[{allocation}],\n"
            f"  start_address={self.start_address}\n"
            f"  backGap={self.backGap}\n"
            f")"
        )


@dataclasses.dataclass
class SDSCSpec:
    opfunc: str
    execution_unit: str
    data_format: DataFormats
    num_inputs: int
    iteration_space: dict[Symbol, Any]
    num_cores: int
    work_slices: dict[Symbol, Any]
    core_id_to_work_slice: dict[Symbol, Any]
    padding: dict[Symbol, Any]
    layouts: dict[int, Any]
    args: list[SDSCArgs]
    constants: dict[str, Any]
    coordinate_masking: dict[Symbol, Any]

    def __str__(self) -> str:
        iter_space = ", ".join(f"{k}={v}" for k, v in self.iteration_space.items())
        slices = ", ".join(f"{k}={v}" for k, v in self.work_slices.items())
        layouts = "\n".join(
            f"    {label}: dim_order=[{', '.join(str(d) for d in info['dim_order'])}],"
            f" stick_dim_order={info['stick_dim_order']},"
            f" stick_size={info['stick_size']}"
            for label, info in self.layouts.items()
        )
        core_slice_map = ", ".join(
            f"{k}={v}" for k, v in self.core_id_to_work_slice.items()
        )
        args = "\n".join("  " + line for a in self.args for line in str(a).splitlines())
        parts = [
            f"  opfunc={self.opfunc}",
            f"  exec_unit={self.execution_unit}",
            f"  data_format={self.data_format.name}",
            f"  num_inputs={self.num_inputs}",
            f"  iteration_space=[{iter_space}]",
            f"  work_slices=[{slices}]",
            f"  core_id_to_work_slice=[{core_slice_map}]",
            f"  layouts=[\n{layouts}\n  ]",
            f"  args=[\n{args}\n  ]",
        ]
        if self.padding:
            parts.append(
                f"  padding=[{', '.join(f'{k}={v}' for k, v in self.padding.items())}]"
            )
        if self.coordinate_masking:
            parts.append(
                "  coordinate_masking=["
                + ", ".join(f"{k}={v}" for k, v in self.coordinate_masking.items())
                + "]"
            )
        if self.constants:
            parts.append(
                f"  constants=[{', '.join(f'{k}={v}' for k, v in self.constants.items())}]"
            )
        return "SDSCSpec(\n" + "\n".join(parts) + "\n)"


def _get_core_to_slice_mapping(
    iteration_space, dim_splits: dict[Symbol, int], num_cores: int
) -> dict[Symbol, Expr]:
    core_id_sym = Symbol("core_id")

    dim_to_expr: dict[str, object] = {}
    inner_product = Integer(1)

    for dim in iteration_space:
        if dim_splits[dim] == 1:
            expr = Integer(0)
        elif inner_product == Integer(1):
            expr = Mod(core_id_sym, Integer(dim_splits[dim]))
        else:
            expr = Mod(floor(core_id_sym / inner_product), Integer(dim_splits[dim]))
        dim_to_expr[str(dim)] = expr
        inner_product = inner_product * Integer(dim_splits[dim])

    return dim_to_expr


def _get_mask_value(op: str) -> float:
    return float("-inf") if op == "max" else float("inf") if op == "min" else 0


def _get_coordinate_mask(
    iteration_space: dict, arg: SDSCArgs, dim_padding: dict
) -> dict:
    return {
        dim: [[iteration_space[dim] - padding, padding]]
        for dim, padding in dim_padding.items()
        if padding > 0 and dim in arg.scales and arg.scales[dim] == -2
    }


def _calculate_device_stride(dev_dim_idx: int, device_size: list) -> int:
    return math.prod(device_size[-dev_dim_idx - 2 :])


def _get_device_dim_order(
    arg: TensorArg, symbol_mapping: dict
) -> tuple[list[Symbol], Symbol | None]:
    """Return (dim_order, stick_dim) for the arg's device layout after symbol substitution."""
    last_coord = arg.device_coordinates[-1].subs(symbol_mapping)
    free = sorted(last_coord.free_symbols, key=str)
    stick_dim = free[0] if free else None

    dim_order: list[Symbol] = []
    for i in range(len(arg.device_coordinates) - 2, -1, -1):
        expr = arg.device_coordinates[i].subs(symbol_mapping)
        if expr == 0 and stick_dim is not None and stick_dim not in dim_order:
            dim_order.append(stick_dim)
        for sym in expr.free_symbols:
            if sym not in dim_order:
                dim_order.append(sym)
    return dim_order, stick_dim


def _get_layout_label(
    layouts: dict,
    dim_order: list,
    stick_dim_order: Symbol | None,
    stick_size: int,
    layout_labels: list[str],
) -> str:
    for label, layout in layouts.items():
        if (
            layout["stick_dim_order"] == stick_dim_order
            and layout["dim_order"] == dim_order
            and layout["stick_size"] == stick_size
        ):
            return label
    label = layout_labels[len(layouts)]
    layouts[label] = {
        "dim_order": dim_order,
        "stick_dim_order": stick_dim_order,
        "stick_size": stick_size,
    }
    return label


def _get_padded_iteration_space(
    op_spec_args: list[TensorArg],
    sdsc_args: list[SDSCArgs],
    sdsc_iteration_space: dict,
    layouts: dict,
) -> dict:
    """
    Compute padding per dim when device size exceeds iteration space.

    Update sdsc_iteration_space when padding is needed.
    Returns a mapping of dim -> padding amount
    """
    padding: dict = {}
    for sdsc_arg, op_spec_arg in zip(sdsc_args, op_spec_args):
        layout = layouts[sdsc_arg.layout]
        stick_dim = layout["stick_dim_order"]
        dev_size = op_spec_arg.device_size[-2::-1]
        for idx, dim in enumerate(layout["dim_order"]):
            if idx >= len(dev_size) or dim != stick_dim:
                continue
            dim_size = dev_size[idx] * layout["stick_size"]
            if sdsc_iteration_space[dim] < dim_size:
                padding[dim] = dim_size - sdsc_iteration_space[dim]
                sdsc_iteration_space[dim] = dim_size
    return padding


def _is_matmul(op: str) -> bool:
    return op in ("matmul", "batchmatmul")


def _get_op_dim_labels(ndim: int, is_matmul: bool) -> list[str]:
    if is_matmul:
        return MATMUL_DIM_LABELS[5 - ndim :]
    return INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]


def _create_sdsc_tensors(
    op_spec: OpSpec,
    symbol_mapping: dict,
    iteration_space: dict,
    op_dim_order: list[Symbol],
    op_stick_dim: Symbol | None,
) -> tuple[list[SDSCArgs], dict, Symbol | None]:
    dims = list(iteration_space.keys())
    layouts: dict = {}
    use_op_dims = not _is_matmul(op_spec.op)

    missing_dim = None
    adjusted_output_size = op_spec.args[-1].device_size.copy()
    sdsc_args: list[SDSCArgs] = []
    for arg in op_spec.args:
        addr = None if arg.arg_index < 0 else SEGMENT_OFFSETS[arg.arg_index]
        dim_order, stick_dim = _get_device_dim_order(arg, symbol_mapping)
        scales: dict = {}
        strides: dict = {}
        offsets: dict = {}
        backGap: dict[Symbol, int] = {}
        max_dim_sizes: dict = {}
        reduced_dims: list = []
        use_adjusted_size = op_spec.op == "overwrite" and not arg.is_input
        if use_op_dims and dim_order != dims:
            reduced_dims = [d for d in op_dim_order if d not in dim_order]
            dim_order = dim_order + reduced_dims

        if op_stick_dim is None:
            # No stick dim found in op - add one
            stick_dim = next(d for d in dims if d not in op_dim_order)
            dim_order = dim_order + [stick_dim]
        if op_spec.op == "layernormscale" and len(sdsc_args) == 0:
            reduced_dims = [stick_dim]
        stride_dim_order = [
            d for d in dim_order if d not in reduced_dims
        ] + reduced_dims
        for dim in dim_order:
            stride_idx = stride_dim_order.index(dim)
            if dim in reduced_dims and op_spec.op != "layernormscale":
                scales[dim] = -2 if (stick_dim is None and dim is op_stick_dim) else -1
            elif dim in reduced_dims and op_spec.op == "layernormscale":
                scales[dim] = -2 if (dim is stick_dim) else -1
            else:
                scales[dim] = 1
            strides[dim] = _calculate_device_stride(
                stride_idx,
                arg.device_size if not use_adjusted_size else adjusted_output_size,
            )
            offsets[dim] = 0
            dim_device_stride = math.prod(arg.device_size[-stride_idx - 1 :])

            dev_dim_size = arg.device_size[-stride_idx - 2]
            it_dim_size = iteration_space[dim]
            if dim == stick_dim:
                stick_size = arg.device_dtype.elems_per_stick()
                dev_dim_size *= stick_size
                it_dim_size = ((it_dim_size - 1) // stick_size + 1) * stick_size

            if dev_dim_size > it_dim_size:
                dim_coord = arg.device_coordinates[-stride_idx - 2]
                dim_offset = int(dim_coord.as_coeff_Add()[0])
                offsets[dim] = dim_offset * dim_device_stride
                backGap[dim] = dev_dim_size - it_dim_size
                strides[dim] = strides[dim] / dev_dim_size * it_dim_size

            max_dim_sizes[dim] = -1

        effective_stick = op_stick_dim if stick_dim is None else stick_dim
        label = _get_layout_label(
            layouts,
            dim_order,
            effective_stick,
            arg.device_dtype.elems_per_stick(),
            MATMUL_LAYOUT_LABELS if not use_op_dims else LAYOUT_LABELS,
        )
        sdsc_args.append(
            SDSCArgs(
                layout=label,
                data_format=arg.device_dtype,
                scales=scales,
                strides=strides,
                offsets=offsets,
                max_dim_sizes=max_dim_sizes,
                allocation=arg.allocation,
                start_address=addr,
                backGap=backGap,
            )
        )

    return sdsc_args, layouts, missing_dim


def _get_op_func(op: str, is_reduction: bool, output_scales: dict) -> str:
    if op == "to_dtype" or op == "overwrite":
        return IDENTITY_OP
    if is_reduction and not _is_matmul(op) and -2 not in output_scales.values():
        return op + "nonstick"
    return op


def _ref_arg(op_spec):
    if op_spec.is_reduction:
        return op_spec.args[0]

    return op_spec.args[-1]


def parse_op_spec(op_spec: OpSpec) -> SDSCSpec:
    is_matmul = _is_matmul(op_spec.op)
    ndim = len(op_spec.iteration_space)
    dim_labels = _get_op_dim_labels(ndim, is_matmul)

    symbol_mapping = {
        sym: Symbol(dim_labels[i]) for i, sym in enumerate(op_spec.iteration_space)
    }
    logger.debug(
        "symbol mapping: %s",
        ", ".join(f"{k} -> {v}" for k, v in symbol_mapping.items()),
    )

    sdsc_iteration_space = {
        symbol_mapping[sym]: (size.p if isinstance(size, Integer) else size)
        for sym, (size, _) in op_spec.iteration_space.items()
    }

    dim_splits = {
        symbol_mapping[dim]: value[-1] for dim, value in op_spec.iteration_space.items()
    }
    num_cores = math.prod(dim_splits.values())

    work_slices = {
        symbol_mapping[sym]: wk_slice
        for sym, (_, wk_slice) in op_spec.iteration_space.items()
    }

    ref_arg = _ref_arg(op_spec)
    op_dim_order, op_stick_dim = _get_device_dim_order(ref_arg, symbol_mapping)

    if op_stick_dim is None:
        stick_sym = Symbol(INPUT_DIM_LABELS[ndim])
        sdsc_iteration_space[stick_sym] = op_spec.args[0].device_dtype.elems_per_stick()
        work_slices[stick_sym] = 1
        dim_splits[stick_sym] = 1

    args, layouts, missing_dim = _create_sdsc_tensors(
        op_spec,
        symbol_mapping,
        sdsc_iteration_space,
        op_dim_order,
        op_stick_dim,
    )
    if missing_dim is not None:
        # A dimension was added to the iteration space, update splits and work slices
        dim_splits[missing_dim] = 1
        work_slices[missing_dim] = 1

    if is_matmul:
        pad_args, pad_sdsc_args = list(op_spec.args), args
    elif op_spec.is_reduction or op_spec.op == "overwrite":
        pad_args, pad_sdsc_args = [op_spec.args[0]], [args[0]]
    else:
        pad_args, pad_sdsc_args = [op_spec.args[-1]], [args[-1]]
    padding = _get_padded_iteration_space(
        pad_args, pad_sdsc_args, sdsc_iteration_space, layouts
    )
    constants = dict(op_spec.op_info.get("constants", {})) if op_spec.op_info else {}
    coordinate_masking = _get_coordinate_mask(sdsc_iteration_space, args[-1], padding)
    if coordinate_masking:
        constants["samv-maskvalue"] = _get_mask_value(op_spec.op)

    num_inputs = len(args[:-1]) if is_matmul or not op_spec.is_reduction else len(args)

    return SDSCSpec(
        opfunc=_get_op_func(op_spec.op, op_spec.is_reduction, args[-1].scales),
        execution_unit="pt" if is_matmul else "sfp",
        data_format=op_spec.args[
            0
        ].device_dtype,  # TODO: op_spec needs operation data format
        num_inputs=num_inputs,
        iteration_space=sdsc_iteration_space,
        num_cores=num_cores,
        work_slices=work_slices,
        core_id_to_work_slice=_get_core_to_slice_mapping(
            sdsc_iteration_space, dim_splits, num_cores
        ),
        padding=padding,
        layouts=layouts,
        args=args,
        constants=constants,
        coordinate_masking=coordinate_masking,
    )


def compile_op_spec(kernel_name: str, op_spec: OpSpec) -> Any:
    sdsc_spec = parse_op_spec(op_spec)
    logger.debug("%s", sdsc_spec)
    return generate_sdsc(sdsc_spec)
