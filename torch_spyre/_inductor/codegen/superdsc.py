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
from collections import Counter
from sympy import Integer, Symbol, Expr, Mod, floor

from torch._inductor.virtualized import V
from torch_spyre._C import DataFormats
from torch_spyre._inductor.constants import (
    IDENTITY_OP,
    INPUT_DIM_LABELS,
    OUTPUT_DIM_LABELS,
    LAYOUT_LABELS,
    MATMUL_DIM_LABELS,
    MATMUL_LAYOUT_LABELS,
    RESTICKIFY_OP,
    TOPK_OPS,
)
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.indirect_access import (
    compute_indirect_max_dim_sizes,
    get_index_tensor_for_value,
    get_indirect_dim_symbols,
    get_indirect_layout_label,
    get_value_tensor_idx_for_index,
    is_index_tensor,
    is_indirect_value_tensor,
)
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import IndirectAccess, OpSpec, TensorArg
from torch_spyre._inductor.dtype_ops import DtypeOpTable

from .compute_ops import SymbolKind, generate_sdsc

logger = get_inductor_logger("codegen.superdsc")


@dataclasses.dataclass
class SDSCArgs:
    layout: str
    dim_order: list[Symbol]
    data_format: DataFormats
    scales: dict[Symbol, Any]
    strides: dict[Symbol, Any]
    offsets: dict[Symbol, Any]
    max_dim_sizes: dict[Symbol, Any]
    allocation: dict[str, Any]
    start_address: int | Symbol
    backGap: dict[Symbol, int]
    arg_index: int = -1
    is_index_tensor: bool = False
    related_value_tensor_idx: int = -1

    def __str__(self) -> str:
        scales = ", ".join(f"{k}={v}" for k, v in self.scales.items())
        strides = ", ".join(f"{k}={v}" for k, v in self.strides.items())
        offsets = ", ".join(f"{k}={v}" for k, v in self.offsets.items())
        max_dim_sizes = ", ".join(f"{k}={v}" for k, v in self.max_dim_sizes.items())
        allocation = ", ".join(f"{k}={v}" for k, v in self.allocation.items())
        return (
            f"SDSCArgs(\n"
            f"  layout={self.layout},\n"
            f"  dim_order={self.dim_order}, \n"
            f"  data_format={self.data_format.name},\n"
            f"  scales=[{scales}],\n"
            f"  strides=[{strides}],\n"
            f"  offsets=[{offsets}],\n"
            f"  max_dim_sizes=[{max_dim_sizes}],\n"
            f"  allocation=[{allocation}],\n"
            f"  start_address={self.start_address}\n"
            f"  backGap={self.backGap}\n"
            f"  is_index_tensor={self.is_index_tensor}\n"
            f"  related_value_tensor_idx={self.related_value_tensor_idx}\n"
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
    indirect_access_indices: list[int] = dataclasses.field(default_factory=list)

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


def _k_fast_core_to_slice_mapping(
    iteration_space, dim_splits: dict[Symbol, int], num_cores: int
) -> dict[Symbol, Expr]:
    """K-cohort-adjacent core-to-slice mapping for matmul.

    Computed directly from the same `(iteration_space, dim_splits, num_cores)`
    inputs as `_get_core_to_slice_mapping`, by treating the K (reduction) dim
    as the innermost/fastest-varying axis along `core_id`. K-cohort members
    (varying `i_k`, fixed `i_m, i_n`) then sit at adjacent physical core IDs,
    so the PSUM ring reduction traverses 1 hop per output tile instead of
    `m * n`.

    Caller is responsible for the gating decision (matmul + k_fast flag + k>1).
    """
    dim_list = list(iteration_space.keys())
    k_dim = dim_list[-1]
    reordered = {k_dim: iteration_space[k_dim]}
    for d in dim_list[:-1]:
        reordered[d] = iteration_space[d]
    return _get_core_to_slice_mapping(reordered, dim_splits, num_cores)


def _should_use_k_fast_mapping(
    is_matmul: bool, iteration_space, dim_splits: dict[Symbol, int]
) -> bool:
    """Decide whether the k_fast mapping should be used for this op.

    Fires only when all three hold: this op is a matmul, the feature flag is
    on, and the planner has chosen a K-split (k > 1). When k == 1 the k_fast
    mapping is identical to the default, so we just use the default to keep
    the code path explicit.
    """
    if not is_matmul:
        return False
    if not _spyre_config.core_id_k_fast_emission:
        return False
    dim_list = list(iteration_space.keys())
    if len(dim_list) < 3:
        return False
    return dim_splits[dim_list[-1]] > 1


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
    arg: TensorArg, symbol_mapping: dict, op_spec: OpSpec | None = None
) -> tuple[list[Symbol], Symbol | None]:
    """Return (dim_order, stick_dim) for the arg's device layout after symbol substitution."""
    last_coord = arg.device_coordinates[-1].subs(symbol_mapping)
    free = sorted(last_coord.free_symbols, key=str)
    stick_dim = free[0] if free else None

    dim_order: list[Symbol] = []
    for i in range(len(arg.device_coordinates) - 2, -1, -1):
        coord = arg.device_coordinates[i]
        # Handle coordinates containing IndirectAccess — extract symbols from index tensor.
        if hasattr(coord, "has") and coord.has(IndirectAccess):
            if op_spec is not None and is_indirect_value_tensor(arg):
                index_arg = get_index_tensor_for_value(op_spec, arg)
                if index_arg is not None:
                    indirect_dims = get_indirect_dim_symbols(
                        arg, index_arg, symbol_mapping
                    )
                    for sym in indirect_dims:
                        if sym not in dim_order:
                            dim_order.append(sym)
            continue
        expr = coord.subs(symbol_mapping)
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
            and Counter(layout["dim_order"]) == Counter(dim_order)
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
    dim_order,
) -> dict:
    """
    Compute padding per dim when device size exceeds iteration space.

    Update sdsc_iteration_space when padding is needed.
    Returns a mapping of dim -> padding amount
    """
    padding: dict = {}
    for sdsc_arg, op_spec_arg, dim_order in zip(sdsc_args, op_spec_args, dim_order):
        layout = layouts[sdsc_arg.layout]
        stick_dim = layout["stick_dim_order"]
        dev_size = op_spec_arg.device_size[-2::-1]
        for idx, dim in enumerate(dim_order):
            if idx >= len(dev_size) or dim != stick_dim:
                continue
            unaligned = sdsc_iteration_space[dim] % layout["stick_size"]
            if unaligned > 0:
                padding[dim] = layout["stick_size"] - unaligned
                sdsc_iteration_space[dim] += padding[dim]
    return padding


def _is_matmul(op: str) -> bool:
    return op in ("matmul", "batchmatmul", "batchmatmulfp8")


def _is_topk(op: str) -> bool:
    return op in TOPK_OPS


def _get_op_dim_labels(ndim: int, is_matmul: bool) -> list[str]:
    if is_matmul:
        return MATMUL_DIM_LABELS[len(MATMUL_DIM_LABELS) - ndim :]
    else:
        return INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]


def _get_data_format(op, device_dtype):
    """
    NOTE: This is NOT a data conversion.
    This is only a temporary re-labeling of the same 32 bit data.
    The underlying data remains unchanged.

    In the long term, SDSC should accept int32 as the data format.
    Such re-labeling will become unnecessary.
    """
    data_format = {
        (
            IDENTITY_OP,
            DataFormats.IEEE_INT32,
        ): DataFormats.IEEE_FP32,  # Identity op: int32 -> fp32
    }
    return data_format.get((op, device_dtype), device_dtype)


def _collect_index_tensor_layouts(
    op_spec: OpSpec,
    symbol_mapping: dict,
    index_tensor_indices: set[int],
    logger: object,
) -> tuple[dict, dict]:
    """First pass: compute (dim_order, stick_dim) for each index tensor.

    Returns:
        index_tensor_layouts: dict mapping tensor_idx -> (dim_order, stick_dim)
        index_active_dims: dict mapping tensor_idx -> set of active (non-stick) dims
    """
    index_tensor_layouts: dict[int, tuple[list, object]] = {}
    index_active_dims: dict[int, set] = {}

    for i in index_tensor_indices:
        arg = op_spec.args[i]
        dim_order, stick_dim = _get_device_dim_order(arg, symbol_mapping)
        index_tensor_layouts[i] = (dim_order, stick_dim)
        active_dims = {d for d in dim_order if d is not stick_dim}
        index_active_dims[i] = active_dims
        logger.debug(
            f"Index tensor {i}: dim_order={dim_order}, stick_dim={stick_dim}, "
            f"active_dims={sorted(map(str, active_dims))}"
        )

    return index_tensor_layouts, index_active_dims


def _create_sdsc_tensors(
    op_spec: OpSpec,
    symbol_mapping: dict,
    iteration_space: dict,
    op_dim_order: list[Symbol],
    op_stick_dim: Symbol | None,
    mb_sym: Symbol | None = None,
) -> tuple[list[SDSCArgs], dict, Symbol | None]:
    dims = list(iteration_space.keys())
    layouts: dict = {}
    use_op_dims = not _is_matmul(op_spec.op)

    # Detect indirect access from device_coordinates: index tensors are those
    # whose name is referenced by an IndirectAccess in another tensor's coordinates,
    # and value tensors are those that contain IndirectAccess in their coordinates.
    index_tensor_indices = {
        i for i, arg in enumerate(op_spec.args) if is_index_tensor(arg, op_spec)
    }
    has_indirect_access = bool(index_tensor_indices)

    # For indirect access: pre-compute index tensor layouts (first pass)
    index_tensor_layouts: dict[int, tuple[list, Any]] = {}
    index_active_dims: dict[int, set] = {}
    if has_indirect_access:
        index_tensor_layouts, index_active_dims = _collect_index_tensor_layouts(
            op_spec, symbol_mapping, index_tensor_indices, logger
        )

    missing_dim = None
    sdsc_args: list[SDSCArgs] = []

    for i, arg in enumerate(op_spec.args):
        # Step 1: Determine dimension order and stick dimension.
        # Index tensors use their pre-computed layout (their coords have no IndirectAccess).
        if has_indirect_access and i in index_tensor_layouts:
            dim_order, stick_dim = index_tensor_layouts[i]
        else:
            dim_order, stick_dim = _get_device_dim_order(arg, symbol_mapping, op_spec)

        scales: dict = {}
        strides: dict = {}
        offsets: dict = {}
        backGap: dict[Symbol, int] = {}
        max_dim_sizes: dict = {}
        reduced_dims: list = []

        # Step 2: Handle reduced dimensions — skip for index tensors.
        if use_op_dims and dim_order != dims and not _is_topk(op_spec.op):
            if not (has_indirect_access and i in index_tensor_indices):
                reduced_dims = [
                    d for d in op_dim_order if d not in dim_order and d is not mb_sym
                ]
                dim_order = dim_order + reduced_dims

        # Step 3: Handle missing stick dimension — skip for index tensors.
        if op_stick_dim is None:
            if not (has_indirect_access and i in index_tensor_indices):
                stick_dim = next(d for d in dims if d not in op_dim_order)
                dim_order = dim_order + [stick_dim]

        if op_spec.op == "layernormscale" and len(sdsc_args) == 0:
            reduced_dims = [stick_dim]
        stride_dim_order = [
            d for d in dim_order if d not in reduced_dims
        ] + reduced_dims

        for dim in dim_order:
            stride_idx = stride_dim_order.index(dim)

            if has_indirect_access and (
                i in index_tensor_indices or is_indirect_value_tensor(arg)
            ):
                scales[dim] = 1
            elif dim in reduced_dims and op_spec.op != "layernormscale":
                scales[dim] = -2 if (stick_dim is None and dim is op_stick_dim) else -1
            elif dim in reduced_dims and op_spec.op == "layernormscale":
                scales[dim] = -2 if (dim is stick_dim) else -1
            else:
                scales[dim] = 1

            strides[dim] = _calculate_device_stride(stride_idx, arg.device_size)
            offsets[dim] = 0
            dim_device_stride = math.prod(arg.device_size[-stride_idx - 1 :])

            dev_dim_size = arg.device_size[-stride_idx - 2]
            it_dim_size = iteration_space[dim]
            if dim == stick_dim:
                stick_size = arg.device_dtype.elems_per_stick()
                dev_dim_size *= stick_size
                it_dim_size = ((it_dim_size - 1) // stick_size + 1) * stick_size

            if has_indirect_access:
                max_dim_sizes[dim] = compute_indirect_max_dim_sizes(
                    i,
                    dim,
                    stick_dim,
                    stride_idx,
                    dev_dim_size,
                    op_spec,
                    symbol_mapping,
                    index_tensor_indices,
                    index_active_dims,
                    logger,
                )
            else:
                max_dim_sizes[dim] = -1

            dim_coord = arg.device_coordinates[-stride_idx - 2]
            if not isinstance(dim_coord, IndirectAccess) and dev_dim_size > it_dim_size:
                dim_offset = int(dim_coord.as_coeff_Add()[0])
                offsets[dim] = dim_offset * dim_device_stride
                backGap[dim] = dev_dim_size - it_dim_size
                strides[dim] = strides[dim] // dev_dim_size * it_dim_size

        if mb_sym is not None:
            dim_order = [mb_sym] + dim_order
            scales[mb_sym] = 1
            strides[mb_sym] = _calculate_device_stride(0, arg.device_size)
            offsets[mb_sym] = 0
            max_dim_sizes[mb_sym] = -1

        effective_stick = op_stick_dim if stick_dim is None else stick_dim
        layout_labels = MATMUL_LAYOUT_LABELS if not use_op_dims else LAYOUT_LABELS

        if has_indirect_access:
            label = get_indirect_layout_label(
                i,
                index_tensor_indices,
                layouts,
                dim_order,
                effective_stick,
                arg.device_dtype.elems_per_stick(),
                layout_labels,
                _get_layout_label,
                logger,
            )
        else:
            label = _get_layout_label(
                layouts,
                dim_order,
                effective_stick,
                arg.device_dtype.elems_per_stick(),
                layout_labels,
            )

        # Index tensors carry 32-bit integer indices; re-label as SENUINT32 since
        # the backend doesn't yet accept IEEE_INT32 in SDSC (deeptools #4307).
        arg_data_format = (
            DataFormats.SENUINT32
            if (has_indirect_access and i in index_tensor_indices)
            else _get_data_format(op_spec.op, arg.device_dtype)
        )

        start_addr = (
            arg.allocation.get("pool")
            if "pool" in arg.allocation
            else arg.allocation.get("lx")
            if "lx" in arg.allocation
            else arg.allocation.get("hbm")
        )

        is_idx_tensor = has_indirect_access and i in index_tensor_indices
        related_val_idx = (
            get_value_tensor_idx_for_index(op_spec, i) if is_idx_tensor else -1
        )

        sdsc_args.append(
            SDSCArgs(
                layout=label,
                dim_order=dim_order,
                data_format=arg_data_format,
                scales=scales,
                strides=strides,
                offsets=offsets,
                max_dim_sizes=max_dim_sizes,
                allocation=arg.allocation,
                start_address=start_addr,
                backGap=backGap,
                arg_index=arg.arg_index,
                is_index_tensor=is_idx_tensor,
                related_value_tensor_idx=related_val_idx,
            )
        )

    return sdsc_args, layouts, missing_dim


def _get_op_func(op: str, is_reduction: bool, output_scales: dict) -> str:
    if (
        is_reduction
        and not _is_matmul(op)
        and not _is_topk(op)
        and -2 not in output_scales.values()
    ):
        return op + "nonstick"
    return op


def _concretize_for_sdsc(expr: Expr) -> int:
    """Concretize a symbolic expression at the SDSC generation boundary.

    SDSC generation (and the downstream DeepTools backend compiler) currently
    requires all iteration-space sizes to be concrete integers.  This is the
    final concretization point in the pipeline: everything upstream may be
    symbolic, but the SDSC JSON emitted here is fully concrete.

    TODO(issue#220): once SDSC generation emits ``symbolDefinitions_`` and
    ``symbolicDimInfo_`` for the DeepTools VariableDefinition DAG, this
    function can be replaced with symbolic expression serialisation and
    iteration-space sizes can remain symbolic all the way through.
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, Integer):
        return int(expr)
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        return V.graph.sizevars.size_hint(expr)
    return int(expr)


def _ref_arg(op_spec):
    if op_spec.is_reduction:
        return op_spec.args[0]

    return op_spec.args[-1]


def _extend_matmul_k_to_padded(
    op_spec: OpSpec,
    sdsc_iteration_space: dict,
    symbol_mapping: dict,
) -> None:
    """Extend sdsc_iteration_space[K] to K_padded for matmul ops.

    The IR-level padding pass pads y's K dimension to K_padded rows but keeps
    the host iteration space (and op_spec.iteration_space) at K.  This function
    computes K_padded = round_up(K, stick_size) and updates
    sdsc_iteration_space[K_sym] before _create_sdsc_tensors runs.

    With sdsc_iteration_space[K_sym] = K_padded:
    - y's dev_dim_size for K == it_dim_size → backGap branch never fires for y.
    - Strides are computed against K_padded → correct for K_padded-extended iteration.
    - _get_padded_iteration_space becomes a no-op for K (already aligned).

    K is identified as the symbol that appears in y's (non-stick) device_coordinates
    but NOT in the output's device_coordinates.  This is the reduction symbol and is
    layout-position agnostic: it works regardless of how MATMUL_DIM_LABELS maps the
    iteration symbols for this particular ndim.
    """
    # y is always args[1]; output is always args[-1] for matmul.
    y_arg = op_spec.args[1]
    out_arg = op_spec.args[-1]

    # Collect non-stick symbols in y's device_coordinates (after symbol_mapping).
    y_dim_order, y_stick_dim = _get_device_dim_order(y_arg, symbol_mapping)
    # y_stick_dim is the within-stick symbol; the remaining dims include K.
    y_non_stick_syms: set = set(y_dim_order) - ({y_stick_dim} if y_stick_dim else set())

    # Collect all symbols in the output's device_coordinates.
    out_dim_order, _ = _get_device_dim_order(out_arg, symbol_mapping)
    out_syms: set = set(out_dim_order)

    # K is in y but not in the output (it's reduced).
    k_candidates = y_non_stick_syms - out_syms
    if not k_candidates:
        logger.warning(
            "_extend_matmul_k_to_padded: could not identify K symbol "
            "(y_non_stick=%s, out_syms=%s), skipping",
            y_non_stick_syms,
            out_syms,
        )
        return
    k_sym = next(iter(k_candidates))

    if k_sym not in sdsc_iteration_space:
        logger.warning(
            "_extend_matmul_k_to_padded: K symbol %s not in sdsc_iteration_space %s, skipping",
            k_sym,
            list(sdsc_iteration_space.keys()),
        )
        return

    # Compute K_padded by rounding K up to the next stick boundary.
    # Reading K_padded from y_arg.device_size would be wrong when y is a view
    # (e.g. a slice) of a larger buffer: device_size reflects the underlying
    # allocation's K extent, not the slice's logical K, so it can be larger
    # than the matmul's actual K and would over-extend the iteration space.
    stick_size = y_arg.device_dtype.elems_per_stick()
    k_current = sdsc_iteration_space[k_sym]
    k_padded = ((k_current + stick_size - 1) // stick_size) * stick_size

    if k_padded > k_current:
        logger.debug(
            "_extend_matmul_k_to_padded: extending K %d -> %d (sym=%s)",
            k_current,
            k_padded,
            k_sym,
        )
        sdsc_iteration_space[k_sym] = k_padded


def parse_op_spec(op_spec: OpSpec) -> tuple["SDSCSpec", "dict"]:
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
        symbol_mapping[sym]: _concretize_for_sdsc(size)
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

    # On-device type-conversion ops (DL16TOFP32/FP32TODL16, not identity)
    # require at least one outer spatial dim beyond the stick; inject a
    # virtual mb=1 row when the op's tensor has only the stick dim.
    mb_sym: Symbol | None = None
    if (
        DtypeOpTable.is_dtype_op(op_spec.op)
        and op_spec.op != IDENTITY_OP
        and op_stick_dim is not None
        and all(d is op_stick_dim for d in op_dim_order)
    ):
        mb_sym = Symbol(INPUT_DIM_LABELS[0])
        sdsc_iteration_space = {mb_sym: 1, **sdsc_iteration_space}
        dim_splits = {mb_sym: 1, **dim_splits}
        work_slices = {mb_sym: 1, **work_slices}
        op_dim_order = [mb_sym] + op_dim_order

    if op_stick_dim is None:
        stick_sym = Symbol(INPUT_DIM_LABELS[ndim])
        sdsc_iteration_space[stick_sym] = op_spec.args[0].device_dtype.elems_per_stick()
        work_slices[stick_sym] = 1
        dim_splits[stick_sym] = 1

    if is_matmul:
        _extend_matmul_k_to_padded(op_spec, sdsc_iteration_space, symbol_mapping)

    args, layouts, missing_dim = _create_sdsc_tensors(
        op_spec,
        symbol_mapping,
        sdsc_iteration_space,
        op_dim_order,
        op_stick_dim,
        mb_sym,
    )
    if missing_dim is not None:
        # A dimension was added to the iteration space, update splits and work slices
        dim_splits[missing_dim] = 1
        work_slices[missing_dim] = 1

    # In case of same type conversion (identity op) user gets compile time error & avoid
    # changing the padding logic here to fix errors with torch.split() for 3d shapes.
    is_dtype_op = DtypeOpTable.is_dtype_op(op_spec.op) and op_spec.op != IDENTITY_OP
    if is_matmul or is_dtype_op:
        pad_args, pad_sdsc_args, dim_order = (
            list(op_spec.args),
            args,
            [arg.dim_order for arg in args],
        )
    elif op_spec.is_reduction:
        pad_args, pad_sdsc_args, dim_order = (
            [op_spec.args[0]],
            [args[0]],
            [args[0].dim_order],
        )
    elif op_spec.op == RESTICKIFY_OP:
        # Pad iteration space using all args so both the old stick (input) and
        # new stick (output) are rounded up to the nearest stick boundary.
        pad_args, pad_sdsc_args, dim_order = (
            list(op_spec.args),
            args,
            [arg.dim_order for arg in args],
        )
    else:
        pad_args, pad_sdsc_args, dim_order = (
            [op_spec.args[-1]],
            [args[-1]],
            [args[-1].dim_order],
        )
    padding = _get_padded_iteration_space(
        pad_args, pad_sdsc_args, sdsc_iteration_space, layouts, dim_order
    )

    # For restickify, update backGaps based on the padded iteration space,
    # since non-stick dimensions may now have it_dim_size > dev_dim_size.
    if op_spec.op == RESTICKIFY_OP:
        for sdsc_arg, op_spec_arg in zip(args, op_spec.args):
            layout = layouts[sdsc_arg.layout]
            stick_dim = layout["stick_dim_order"]
            for coord_idx, coord in enumerate(op_spec_arg.device_coordinates):
                mapped_coord = coord.subs(symbol_mapping)
                dim_sym = next(
                    (
                        s
                        for s in symbol_mapping.values()
                        if s in mapped_coord.free_symbols
                    ),
                    None,
                )
                if dim_sym is None or dim_sym == stick_dim:
                    continue
                padded_it_size = sdsc_iteration_space[dim_sym]
                dev_dim_size = op_spec_arg.device_size[coord_idx]
                if dev_dim_size < padded_it_size:
                    sdsc_arg.backGap[dim_sym] = padded_it_size - dev_dim_size
        for dim in padding:
            dim_splits[dim] = 1
            work_slices[dim] = 1
        num_cores = math.prod(dim_splits.values())

    constants = dict(op_spec.op_info.get("constants", {})) if op_spec.op_info else {}
    coordinate_masking = _get_coordinate_mask(sdsc_iteration_space, args[-1], padding)
    if coordinate_masking:
        constants["samv-maskvalue"] = _get_mask_value(op_spec.op)

    num_inputs = len(args[:-1]) if is_matmul or not op_spec.is_reduction else len(args)

    if _is_topk(op_spec.op):
        num_inputs = 1  # topk has exactly 1 input tensor and 1 output tensor

    if _should_use_k_fast_mapping(is_matmul, sdsc_iteration_space, dim_splits):
        core_id_to_work_slice = _k_fast_core_to_slice_mapping(
            sdsc_iteration_space, dim_splits, num_cores
        )
    else:
        core_id_to_work_slice = _get_core_to_slice_mapping(
            sdsc_iteration_space, dim_splits, num_cores
        )

    # Collect index tensor indices for indirect access
    indirect_access_indices = [
        i for i, arg in enumerate(op_spec.args) if is_index_tensor(arg, op_spec)
    ]

    return (
        SDSCSpec(
            opfunc=_get_op_func(op_spec.op, op_spec.is_reduction, args[-1].scales),
            execution_unit="pt" if is_matmul else "sfp",
            data_format=args[
                1 if indirect_access_indices else 0
            ].data_format,  # TODO: op_spec needs operation data format. Use value tensor (args[1]) for indirect access ops
            num_inputs=num_inputs,
            iteration_space=sdsc_iteration_space,
            num_cores=num_cores,
            work_slices=work_slices,
            core_id_to_work_slice=core_id_to_work_slice,
            padding=padding,
            layouts=layouts,
            args=args,
            constants=constants,
            coordinate_masking=coordinate_masking,
            indirect_access_indices=indirect_access_indices,
        ),
        symbol_mapping,
    )


def compile_op_spec(
    idx: int,
    op_spec: OpSpec,
    symbols: list[int],
    symbol_id_offset: int = 0,
    use_symbols: bool = False,
) -> tuple[Any, list[int], list[dict], list[SymbolKind]]:
    sdsc_spec, symbol_mapping = parse_op_spec(op_spec)
    logger.debug("%s", sdsc_spec)
    # Translate tiled_symbols from OpSpec's inductor symbols to the renamed
    # SDSC symbols via the same mapping used to build sdsc_spec.
    tiled_symbols = [
        symbol_mapping[s] for s in op_spec.tiled_symbols if s in symbol_mapping
    ]
    return generate_sdsc(
        idx,
        sdsc_spec,
        symbols,
        symbol_id_offset,
        tiled_symbols=tiled_symbols,
        use_symbols=use_symbols,
    )
