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
from collections import Counter
from typing import Any

from sympy import Expr, Integer, Symbol
from torch._inductor.virtualized import V

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._C import ElementArrangement
from torch_spyre._inductor.constants import (
    CONV2D_DIM_LABELS,
    CONV2D_FWD_OP,
    CONV2D_LAYOUT_LABELS,
    CONV_DIM_LABELS,
    CONV_OPS,
    DEPTHWISE_CONV2D_OP,
    FP32TOINT32_OP,
    IDENTITY_OP,
    INPUT_DIM_LABELS,
    INT32TOFP32_OP,
    LAYOUT_LABELS,
    MATMUL_DIM_LABELS,
    MATMUL_LAYOUT_LABELS,
    MATMUL_REDUCTION_OPS,
    OUTPUT_DIM_LABELS,
    POOL_DIM_LABELS,
    POOL_OPS,
    RESTICKIFY_OP,
    TOPK_OPS,
    KEEP_BY_INDEX_OP,
)
from torch_spyre._inductor.dtype_ops import DtypeOpTable
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
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    IndirectAccess,
    OpSpec,
    TensorArg,
    TensorWorkDivision,
    is_lx_relayout_identity,
)
from torch_spyre._inductor.pass_utils import coeff_through_floor

from .compute_ops import SymbolKind, generate_sdsc, num_bytes

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
    work_division: TensorWorkDivision | None = None
    arg_index: int = -1
    is_index_tensor: bool = False
    related_value_tensor_idx: int = -1
    device_tile_advance_expr: Expr | None = None

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
            f"  work_division={self.work_division}\n"
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
    conv_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    # maps SDSC dim name -> (pytorch_sym_name, granularity, max_val)
    symbolic_dims: dict[str, tuple[str, int, int]] = dataclasses.field(
        default_factory=dict
    )
    indirect_access_indices: list[int] = dataclasses.field(default_factory=list)
    debug_handle: DebugHandle | None = None
    # Generic pool/window fields.  Neutral defaults mean generate_sdsc treats a
    # non-pool op exactly as before; parse_op_spec fills these for pool ops via
    # _avgpool_sdsc_fields, so compute_ops.py stays free of op-specific logic.
    padding_sizes: dict = dataclasses.field(default_factory=dict)
    padding_sizes_per_core: dict = dataclasses.field(default_factory=dict)
    pds_reuse: bool = False
    stick_replication: bool = False
    window_dims: frozenset = dataclasses.field(default_factory=frozenset)
    input_coord_padding: dict = dataclasses.field(default_factory=dict)
    input_coord_sizes: dict = dataclasses.field(default_factory=dict)
    emit_memorg_padding: bool = False

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
        psizes = ", ".join(f"{k}={v}" for k, v in self.padding_sizes.items())
        psizes_per_core = ", ".join(
            f"{k}={v}" for k, v in self.padding_sizes_per_core.items()
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
            f"  padding_sizes=[{psizes}]",
            f"  padding_sizes_per_core=[{psizes_per_core}]",
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


# Pointwise ops whose *output* padding lanes are seeded to a deterministic value
# rather than left as allocator garbage. The mask covers the out-of-logical-range
# padding lanes of every padded output dim of the op (see _get_coordinate_mask:
# for an allowlisted op it masks each dim with padding > 0, not only the stick
# dim), so seeding them is safe for ANY consumer:
#   - a downstream contraction (matmul) reads them as an operand → the value is
#     chosen contraction-neutral so they add nothing;
#   - a downstream reduction masks its own padding anyway;
#   - a direct host read-out never includes padding lanes.
#
# The motivating case is the flash-attention numerator matmul (exp_scores @
# value): with an unpadded kv sequence (seqlen_kv % stick_size != 0) the final
# kv-stick's padding lanes are uninitialized, exp() of that garbage overflows
# fp16, and the overflow poisons the matmul. Value: SAMV substitutes it at the
# masked input coordinate before the op runs (same semantics as the reduction
# path, where "max" uses -inf), so exp(-inf) = 0 → the padded lanes contribute
# nothing.
#
# BANDAGE — scope is deliberately narrow, do not read this as general support:
#   - Only "exp" is covered: it is the one pointwise op on the SDPA kv axis that
#     turns garbage into a non-finite value. Other overflow-prone ops
#     (reciprocal, rsqrt, ...) are NOT handled and CANNOT be by this mechanism —
#     SAMV masks the op's INPUT, and for those ops no finite input maps to a
#     neutral output (there is no x with 1/x == 0). See #3290.
#   - Multi-dim masking is UNTESTED. SDPA only pads the stick dim, so in practice
#     _get_coordinate_mask emits a single-dim mask here. The comprehension will
#     emit a mask per padded dim if an op ever has more than one, but that path
#     has no test coverage — treat multi-padded-dim pointwise ops as unverified.
#   - Masking is unconditional by op-name, not gated on whether the output
#     actually feeds a contraction (that consumer analysis is not available at
#     this point in codegen). Safe (padding lanes are never valid data), but
#     broader than necessary. TODO(consumer-gating).
#
# STOPGAP: this op allowlist bakes a consumer-specific neutral value at
# production time because SpyreTensorLayout carries no record of the padded-stick
# state. The principled replacement is a padded-stick-state enum on the layout
# (set at DMA-in and at buffer allocation), which would let the compiler pick the
# right neutral value per consumer and elide pad/zero copies — tracked in #3290.
# Retire this dict once that lands.
#
# NOTE: the mask value is a large finite negative, not -inf. encode_constant()
# (module.cpp -> deeptools::FloatToFp16Bin) mis-encodes IEEE +-inf as a NaN bit
# pattern instead of the fp16 infinity encoding (confirmed empirically: -inf
# round-trips to NaN, not 0xFC00), so exp(-inf) never reaches the runtime as
# exp(-inf) -- it reaches it as exp(NaN), which poisons the output instead of
# masking it to 0. -1e4 is finite (encodes correctly) and still underflows
# exp() to exactly 0 in fp16 (fp16's smallest positive value is ~6e-8; anything
# below about -12 already underflows), so it produces the same masking effect
# without going through the broken infinity path.
#
# The max/min reduction identities below have the same encode_constant bug:
# _get_mask_value("max") fed float("-inf") through the same broken path, so a
# max-reduction's padded lanes were seeded with NaN instead of -inf. Unlike
# exp's poisoned-but-locally-contained NaN, max(x, NaN) == NaN -- the identity
# failure propagates through the whole reduction result, not just the padded
# lanes, which would poison any block_max = torch.amax(scores, dim=-1) whose
# padded lanes get reduced over. _FP16_MAX/_FP16_MIN are the most extreme
# finite fp16 values, so they lose (max)/win (min) against any real fp16 score
# while still encoding correctly.
_FP16_MAX = 65504.0
_FP16_MIN = -65504.0

_POINTWISE_PADDING_MASK_VALUE: dict[str, float] = {
    "exp": -1e4,  # exp(-1e4) underflows to 0 in fp16; see NOTE above.
}


def _get_mask_value(op: str) -> float:
    if op == "max":
        return _FP16_MIN
    if op == "min":
        return _FP16_MAX
    if op in _POINTWISE_PADDING_MASK_VALUE:
        return _POINTWISE_PADDING_MASK_VALUE[op]
    return 0


def _get_coordinate_mask(
    iteration_space: dict, arg: SDSCArgs, dim_padding: dict, op: str = ""
) -> dict:
    # Reduction path: mask the stick dim being reduced (scale == -2), so the
    # padding lanes take the reduction identity.
    # Pointwise path: for allowlisted ops (e.g. exp feeding a matmul), also mask
    # EVERY padded output dim so its lanes are contraction-neutral. In practice
    # SDPA pads only the stick dim, so this emits a single-dim mask; the multi-dim
    # case is unexercised (see the BANDAGE note on _POINTWISE_PADDING_MASK_VALUE).
    mask_pointwise = op in _POINTWISE_PADDING_MASK_VALUE
    return {
        dim: [[iteration_space[dim] - padding, padding]]
        for dim, padding in dim_padding.items()
        if padding > 0
        and dim in arg.scales
        and (arg.scales[dim] == -2 or mask_pointwise)
    }


def _calculate_device_stride(dev_dim_idx: int, device_size: list) -> int:
    return math.prod(device_size[-dev_dim_idx - 2 :])


# SDSC dim labels for the conv2d padding (output-spatial) and window (kernel)
# axes. These labels are owned by codegen -- see the note on CONV2D_DIM_LABELS in
# constants.py -- so they are defined here rather than plumbed down from
# ``lower_convolution`` through ``op_info``.
_CONV2D_PAD_DIM_I = CONV2D_DIM_LABELS[2]
_CONV2D_PAD_DIM_J = CONV2D_DIM_LABELS[3]
_CONV2D_WINDOW_DIM_I = CONV2D_DIM_LABELS[-2]
_CONV2D_WINDOW_DIM_J = CONV2D_DIM_LABELS[-1]


def _is_conv2d_kernel_tensor(arg: TensorArg, tensor_position: int | None) -> bool:
    """Check if a tensor is a kernel tensor for conv2d ops.

    A conv2d kernel tensor is identified by its position in op_spec.args.
    This function centralizes the kernel identification logic.
    Called only for tensors in op_spec.args (via enumerate), so tensor_position is never -1.
    """
    return tensor_position == 1


def _get_device_dim_order(
    arg: TensorArg,
    symbol_mapping: dict,
    op_spec: OpSpec | None = None,
    tensor_position: int | None = None,
) -> tuple[list[Symbol], Symbol | None]:
    """Return (dim_order, stick_dim) for the arg's device layout after symbol substitution.

    For kernel tensors in conv ops (tensor_position==1):
    - Excludes size-1 output-spatial dimensions (i, j) since kernels have no spatial-output dependence.
    - Explicitly includes kernel dimensions (ki, kj) even if they don't appear in device_coordinates
      (e.g., when kernel_size=1, they don't iterate naturally but are still structural dimensions).
    """
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
        for sym in sorted(expr.free_symbols, key=str):
            # For kernel tensors in conv ops, exclude size-1 output-spatial dimensions.
            # Kernels don't depend on output spatial position, so i and j (when size-1)
            # are synthetic placeholders that shouldn't affect kernel layout.
            skip_sym = False
            if (
                op_spec is not None
                and _is_depthwise_conv(op_spec.op)
                and _is_conv2d_kernel_tensor(arg, tensor_position)
                and str(sym) in ("i", "j")
            ):
                # sym is already mapped to "i" or "j", but iteration_space has the original c2/c3/z0/z1 names.
                # Find which original symbol maps to this i/j by looking up the reverse mapping.
                orig_sym = None
                for orig, mapped in symbol_mapping.items():
                    if str(mapped) == str(sym):
                        orig_sym = orig
                        break

                if orig_sym is not None and orig_sym in op_spec.iteration_space:
                    size_expr, _ = op_spec.iteration_space[orig_sym]
                    sym_size = _concretize_for_sdsc(size_expr)
                    if sym_size == 1:
                        # Skip this synthetic placeholder
                        skip_sym = True

            if not skip_sym and sym not in dim_order:
                dim_order.append(sym)

    # For kernel tensors in conv ops, always explicitly add ki/kj dimensions.
    # These are spatial dimensions of the kernel tensor and should be part of its layout.
    # They may not appear in device_coordinates when kernel_size=1, but they should still
    # be in layoutDimOrder and coordinates_.
    if (
        op_spec is not None
        and _is_depthwise_conv(op_spec.op)
        and _is_conv2d_kernel_tensor(arg, tensor_position)
        and op_spec.op_info
        and "conv_params" in op_spec.op_info
    ):
        conv_params = op_spec.op_info["conv_params"]
        # Add ki and kj to dim_order if they have non-zero kernel sizes
        for window_sym_name, kernel_key in [("ki", "kernel_h"), ("kj", "kernel_w")]:
            kernel_size = conv_params.get(kernel_key, 1)
            if kernel_size > 0:
                window_sym = Symbol(window_sym_name)
                if window_sym not in dim_order:
                    dim_order.insert(0, window_sym)

    return dim_order, stick_dim


def _get_layout_label(
    layouts: dict,
    dim_order: list,
    stick_dim_order: list,
    stick_size: list,
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
        stick_dim_order = layout["stick_dim_order"]
        stick_size = layout["stick_size"]
        dev_size = op_spec_arg.device_size[-2::-1]
        for idx, dim in enumerate(dim_order):
            if idx >= len(dev_size) or dim not in stick_dim_order:
                continue
            effective_stick_size = (
                stick_size[0] if len(stick_size) == 1 else stick_size[0] * stick_size[1]
            )
            unaligned = sdsc_iteration_space[dim] % effective_stick_size
            if unaligned > 0:
                padding[dim] = effective_stick_size - unaligned
                sdsc_iteration_space[dim] += padding[dim]
    return padding


def _is_matmul(op: str) -> bool:
    return op in MATMUL_REDUCTION_OPS


def _is_topk(op: str) -> bool:
    return op in TOPK_OPS


def _is_pool(op: str) -> bool:
    return op in POOL_OPS


def _is_conv(op: str) -> bool:
    # CONV_OPS covers both the forward conv2d (aten.convolution direct lowering,
    # PR #3284) and depthwise conv2d (spyre.conv2d, PR #3510) op strings.
    return op in CONV_OPS


def _is_depthwise_conv(op: str) -> bool:
    return op == DEPTHWISE_CONV2D_OP


def _is_keep_by_index(op: str) -> bool:
    return op == KEEP_BY_INDEX_OP


# Canonical avgpool iteration-space order (NHWC) -> SDSC labels.  Codegen owns
# these label strings; survival of each role is read from the node's live output
# ranges (see _align_pool_dim_labels), so no size info leaks above codegen.
# Order matches POOL_DIM_LABELS and the emitted (NHWC) iteration space.
_POOL_ROLE_LABELS = list(
    zip(["batch", "out_h", "out_w", "channel", "win_h", "win_w"], POOL_DIM_LABELS)
)

# Canonical conv2d iteration-space order -> SDSC labels, mirroring
# _POOL_ROLE_LABELS.  Conv adds the ``in_channel`` contraction role between the
# output channel and the kernel taps.  Codegen owns these strings; each role is
# recovered structurally from the args' access expressions (see
# _match_labels_by_structure), never from sizes.  Order matches CONV_DIM_LABELS.
_CONV_ROLE_LABELS = list(
    zip(
        ["batch", "out_h", "out_w", "channel", "in_channel", "win_h", "win_w"],
        CONV_DIM_LABELS,
    )
)


def _is_static_one(sz) -> bool:
    try:
        return int(sz) == 1
    except (TypeError, ValueError):
        return False  # symbolic/dynamic dim: never dropped


def _try_static_int(sz) -> int | None:
    """Return sz as a concrete int, or None if it is symbolic/dynamic."""
    try:
        return int(sz)
    except (TypeError, ValueError):
        return None


def _align_pool_dim_labels(node_output_ranges, ndim: int) -> list[str]:
    """Return the pool dim labels aligned to the (possibly squeezed) iteration space.

    ``node_output_ranges`` is the reduction node's full logical output ranges in
    **NCHW** order ``[N, C, H_out, W_out]`` (live IR, incl. unit dims) — see
    ``OpSpec.node_output_ranges``.  Codegen owns the SDSC label for each role
    (``_POOL_ROLE_LABELS``, in **NHWC** order).  The compilation pipeline drops
    statically size-1 output dims (e.g. batch N=1) before parse_op_spec runs, so
    a role whose live range is 1 has no surviving iteration-space dim and its
    label is filtered out.  Survival is keyed by role name and emitted in NHWC
    order; the window dims (win_h/win_w) always survive because the lowering
    delegates to the in-tree path when kH==1 or kW==1, so a SpyreReduction always
    has kH>1 and kW>1.  This keeps labels aligned to the real iteration space
    using live node ranges rather than a lowering-time size snapshot.
    """
    if node_output_ranges is None or len(node_output_ranges) != 4:
        raise ValueError(
            "pool node_output_ranges must be NCHW [N, C, H_out, W_out]; got "
            f"{node_output_ranges!r}"
        )
    # NCHW positions: 0=batch, 1=channel, 2=out_h, 3=out_w.
    survives = {
        "batch": not _is_static_one(node_output_ranges[0]),
        "channel": not _is_static_one(node_output_ranges[1]),
        "out_h": not _is_static_one(node_output_ranges[2]),
        "out_w": not _is_static_one(node_output_ranges[3]),
        "win_h": True,  # kH>1 guaranteed by the lowering delegation guard
        "win_w": True,  # kW>1 guaranteed by the lowering delegation guard
    }
    labels = [label for role, label in _POOL_ROLE_LABELS if survives[role]]
    if len(labels) != ndim:
        raise ValueError(
            f"pool dim label count {len(labels)} ({labels}) does not match "
            f"iteration-space rank {ndim}; node_output_ranges {node_output_ranges!r} "
            "are out of sync with the emitted iteration space"
        )
    return labels


def _arg_stick_symbol(arg) -> Symbol | None:
    """The iteration symbol on ``arg``'s stick (innermost) device coordinate.

    Spyre lays the stick dim last, so ``device_coordinates[-1]`` is the stick
    lane (e.g. ``Mod(c3, 64)``).  Returns its single free symbol, or ``None``
    when the arg has no coordinates or the stick coordinate is not driven by
    exactly one symbol (so callers can bail to a positional mapping).  Mirrors
    the stick-var read in ``spyre_kernel.create_op_spec``.
    """
    if not arg.device_coordinates:
        return None
    free = arg.device_coordinates[-1].free_symbols
    return next(iter(free)) if len(free) == 1 else None


def _ordered_arg_symbols(arg, exclude: tuple = ()) -> list[Symbol]:
    """Free symbols across ``arg``'s device coordinates, first-appearance order.

    Deduplicates and skips ``exclude``; within a single coordinate the symbols
    are ordered by name so the walk is deterministic (conv coordinates carry at
    most one free symbol each, so this only matters defensively).
    """
    seen, ordered = set(exclude), []
    for coord in arg.device_coordinates:
        for sym in sorted(coord.free_symbols, key=str):
            if sym not in seen:
                seen.add(sym)
                ordered.append(sym)
    return ordered


def _match_labels_by_structure(op_spec: "OpSpec") -> dict | None:
    """Map each conv2d iteration symbol to its SDSC dim label, structurally.

    A Reduction's iteration space appends its reduction axes in data-dependent
    read-dep access order (see ``iteration_space`` in pass_utils), so the
    contraction dim (``in``) and the kernel taps (``ki``/``kj``) cannot be told
    apart positionally.  Rather than reconstruct the canonical ordering from a
    carried size snapshot, recover each role from what the args' access
    expressions already encode -- set membership and co-occurrence, never sizes
    or positions:

      * ``out`` (C_out): the *output* arg's stick symbol -- also the weight's
        stick symbol, which is how the weight input is told from the activation.
      * ``in`` (C_in, contraction): the *activation's* stick symbol.  Reading
        only the two inputs' stick symbols (never the weight's permuted
        non-stick dim order) keeps this robust to the weight's channel-last
        layout.
      * ``i``/``j``/``mb`` (out_h/out_w/batch): the output arg's non-stick
        symbols in device-coordinate order -- batch trails the spatial dims.
      * ``ki``/``kj`` (win_h/win_w): the weight's symbols other than its stick
        and the contraction, paired to a spatial axis by adjacency in the
        activation window (a tap immediately follows its output-spatial dim).

    Squeezed size-1 dims (batch N=1, a 1xN / Nx1 kernel's collapsed tap) simply
    never appear as a symbol, so they drop out for free -- no per-role size-1
    filtering and no ``C_in`` vs tap size-collision guard are needed.

    Returns ``None`` (caller falls back to a positional mapping) whenever the
    arg structure is not the expected two-input reduction, so nothing else is
    affected.  Unlike the former size-matching this has no static-shape
    requirement, so dynamic spatial dims are handled too.
    """
    role_label = dict(_CONV_ROLE_LABELS)
    inputs = [a for a in op_spec.args if a.is_input]
    outputs = [a for a in op_spec.args if not a.is_input]
    if len(inputs) != 2 or len(outputs) != 1:
        return None
    out_arg = outputs[0]
    channel = _arg_stick_symbol(out_arg)
    if channel is None:
        return None
    weight = next((a for a in inputs if _arg_stick_symbol(a) == channel), None)
    activation = next((a for a in inputs if a is not weight), None)
    if weight is None or activation is None:
        return None
    contraction = _arg_stick_symbol(activation)
    if contraction is None or contraction == channel:
        return None

    # Output roles: non-stick output symbols in coordinate order are
    # [out_h, out_w, batch] (batch trails the spatial dims in the output layout).
    out_spatial = _ordered_arg_symbols(out_arg, exclude=(channel,))
    spatial_roles = ["out_h", "out_w", "batch"]
    if len(out_spatial) > len(spatial_roles):
        return None
    role_of = dict(zip(out_spatial, spatial_roles))

    # Taps are the weight's symbols other than its stick (channel) and the
    # contraction; pair each to a spatial axis by activation-window adjacency.
    act_order = _ordered_arg_symbols(activation)
    tap_role = {}
    for tap in _ordered_arg_symbols(weight, exclude=(channel, contraction)):
        if tap not in act_order:
            return None
        i = act_order.index(tap)
        prev_spatial = next(
            (
                act_order[k]
                for k in range(i - 1, -1, -1)
                if role_of.get(act_order[k]) in ("out_h", "out_w")
            ),
            None,
        )
        if prev_spatial is None:
            return None
        tap_role[tap] = "win_h" if role_of[prev_spatial] == "out_h" else "win_w"

    mapping = {
        channel: Symbol(role_label["channel"]),
        contraction: Symbol(role_label["in_channel"]),
    }
    for sym, role in role_of.items():
        mapping[sym] = Symbol(role_label[role])
    for sym, role in tap_role.items():
        mapping[sym] = Symbol(role_label[role])
    # Every iteration symbol must have received a label; otherwise fall back so
    # the positional path assigns a complete mapping.
    if any(sym not in mapping for sym in op_spec.iteration_space):
        return None
    return mapping


_CONV2D_ROLE_LABELS = list(
    zip(
        ["batch", "channel", "out_h", "out_w"],
        [
            CONV2D_DIM_LABELS[0],
            CONV2D_DIM_LABELS[1],
            _CONV2D_PAD_DIM_I,
            _CONV2D_PAD_DIM_J,
        ],
    )
)


def _align_conv2d_dim_labels(
    node_output_ranges,
    ndim: int,
    kernel_h,
    kernel_w,
) -> list[str]:
    """Return conv2d dim labels aligned to the (possibly squeezed) iteration space.

    ``node_output_ranges`` is the reduction node's full logical output ranges in
    **NCHW** order ``[N, C, H_out, W_out]`` (live IR, incl. unit dims) -- see
    ``OpSpec.node_output_ranges``.  Codegen owns the SDSC label for each role
    (``_CONV2D_ROLE_LABELS``, emitted in canonical ``mb, out, i, j`` order).

    A role whose live range is 1 is a *candidate* for having been squeezed out of
    the iteration space, but the squeeze is **not unconditional**, so candidacy
    alone cannot decide survival.  Two behaviours have to be reconciled:

    - ``N == 1``  and ``kernel_size == 1`` really are
      dropped, and the squeeze can land at the front, at the back, or in the
      **middle** (``H_out == 1`` with ``mb`` surviving, dw-15).  No suffix slice
      or count-from-one-end scheme can express a middle squeeze, which is why
      survival is keyed per role rather than positionally.
    - A **fully collapsed 1x1 output** *keeps*
      its unit output-spatial dims: the iteration space stays rank 6 with the
      kernel extents live and ``i``/``j`` present as size-1 dims.

    So unit-range roles are dropped only in *reverse canonical order*, and only
    as many as the iteration-space rank requires.  ``ndim`` is the ground truth;
    the ranges only say which roles are *eligible* to be dropped.  Unlike
    ``_align_pool_dim_labels`` this must also consider the **window** roles:
    ``lower_avg_pool2d`` delegates to the in-tree lowering when
    ``kH == 1 or kW == 1``, so a pool ``SpyreReduction`` always has both window
    dims, but conv2d explicitly supports 1x1 / 1xN / Nx1 kernels.
    """
    if node_output_ranges is None or len(node_output_ranges) != 4:
        raise ValueError(
            "conv2d node_output_ranges must be NCHW [N, C, H_out, W_out]; got "
            f"{node_output_ranges!r}"
        )
    # Full canonical label list with each entry's unit-ness, in iteration-space
    # order: output roles (mb, out, i, j) then window roles (ki, kj).
    candidates: list[tuple[str, bool]] = [
        (label, _is_static_one(node_output_ranges[pos]))
        for pos, (_role, label) in enumerate(_CONV2D_ROLE_LABELS)
    ]
    candidates.append((_CONV2D_WINDOW_DIM_I, _is_static_one(kernel_h)))
    candidates.append((_CONV2D_WINDOW_DIM_J, _is_static_one(kernel_w)))

    n_to_drop = len(candidates) - ndim
    if n_to_drop < 0:
        raise ValueError(
            f"conv2d iteration-space rank {ndim} exceeds the {len(candidates)} "
            f"canonical dim labels {[label for label, _ in candidates]}; "
            f"node_output_ranges {node_output_ranges!r}, kernel "
            f"{kernel_h}x{kernel_w}"
        )
    # Drop unit-range roles from the back: the pipeline squeezes the innermost
    # eligible dims first, and a 1x1 output keeps i/j rather than dropping them.
    dropped: set[int] = set()
    for idx in range(len(candidates) - 1, -1, -1):
        if len(dropped) == n_to_drop:
            break
        if candidates[idx][1]:
            dropped.add(idx)
    if len(dropped) != n_to_drop:
        unit_labels = [label for label, is_unit in candidates if is_unit]
        raise ValueError(
            f"conv2d needs to drop {n_to_drop} dim label(s) to reach "
            f"iteration-space rank {ndim}, but only {len(unit_labels)} "
            f"unit-range role(s) {unit_labels} are eligible; node_output_ranges "
            f"{node_output_ranges!r} and kernel {kernel_h}x{kernel_w} are out of "
            "sync with the emitted iteration space"
        )
    return [label for idx, (label, _) in enumerate(candidates) if idx not in dropped]


def _avgpool_sdsc_fields(iteration_space: dict, pool_params: dict) -> dict:
    """Compute the pool-specific SDSC field values for an avgpool op.

    Returns plain data that is threaded onto ``SDSCSpec`` and consumed
    generically by ``generate_sdsc`` in compute_ops.py, which keeps no
    pool-specific knowledge (see the generic ``padding``/``num_inputs``
    fields for the established pattern).  ``iteration_space`` is the renamed
    SDSC iteration space, so the spatial dims are keyed by ``Symbol("i")`` and
    ``Symbol("j")``.
    """
    kH = int(pool_params["kernel_h"])
    kW = int(pool_params["kernel_w"])
    sH = int(pool_params.get("stride_h", 1))
    sW = int(pool_params.get("stride_w", 1))
    pH = int(pool_params.get("pad_h", 0))
    pW = int(pool_params.get("pad_w", 0))
    fullspan = "padded_fullspan_wunneeded"

    # One entry per spatial axis whose pooling window actually survives in the
    # iteration space.  kernel_size==1 makes that axis' reduction dim size-1,
    # which the pipeline squeezes out (so its label was already dropped by
    # _align_pool_dim_labels).  Such an axis is a plain pass-through: emitting a
    # paddingSizes_/windowDim_ entry for it would reference a dim the SDSC no
    # longer has, and dxp_standalone aborts with "Missing window size for padded
    # size calculation".  So skip any axis whose window dim is absent.
    axes = [
        ("i", "ki", kH, sH, pH),
        ("j", "kj", kW, sW, pW),
    ]
    padding_sizes: dict = {}
    window_dims: set = set()
    input_coord_padding: dict = {}
    input_coord_sizes: dict = {}
    for spatial, window, k, s, p in axes:
        if Symbol(window) not in iteration_space:
            continue
        out = int(iteration_space.get(Symbol(spatial), 1))
        in_size = (out - 1) * s + k
        padding_sizes[spatial] = {
            "padFront_": p,
            "padBack_": p,
            "totalSize_": in_size,
            "stride_": s,
            "dilation_": 1,
            "windowDim_": window,
        }
        window_dims.add(window)
        input_coord_padding[spatial] = fullspan
        input_coord_sizes[spatial] = in_size

    return {
        "padding_sizes": padding_sizes,
        "pds_reuse": True,
        "stick_replication": True,
        "window_dims": frozenset(window_dims),
        "input_coord_padding": input_coord_padding,
        "input_coord_sizes": input_coord_sizes,
        "emit_memorg_padding": True,
    }


def _conv2d_sdsc_fields(
    iteration_space: dict, conv_params: dict, dim_splits: dict
) -> dict:
    """Compute conv2d-specific SDSC field values for depthwise conv2d.

    Computes paddingSizes_ for both top-level (full-size) and per-core (split-size)
    variants, storing them in the returned dict as padding_sizes and
    padding_sizes_per_core respectively. Both variants use the same inner structure
    (flat dict keyed by spatial dim label), matching the shape already used by avgpool.
    """
    if not conv_params:
        return {}

    def compute_padding_for_dim(
        suffix,
        pad_dim,
        kernel_key,
        stride_key,
        window_dim,
        total_size_key,
        dim_sizes=None,
    ):
        """Compute padding fields for a single dimension (i or j).

        ``pad_dim`` and ``window_dim`` are SDSC dim labels supplied by codegen,
        not keys to look up in ``conv_params``.
        """
        stride = conv_params[stride_key]
        kernel_size = conv_params[kernel_key]
        pad_amount = conv_params.get(f"pad_{suffix}", 0)

        if dim_sizes is None:
            total_size = conv_params[total_size_key]
            full_output = (total_size - kernel_size) // stride + 1
            per_core_output = full_output
        else:
            per_core_output = dim_sizes[Symbol(pad_dim)] // dim_splits[Symbol(pad_dim)]
            num_splits = dim_splits[Symbol(pad_dim)]
            if num_splits == 1:
                total_size = conv_params[total_size_key]
            else:
                total_size = (per_core_output) * stride + kernel_size - 1

        min_required_input = (per_core_output - 1) * stride + kernel_size
        unneeded_pad = total_size - min_required_input

        padFront = pad_amount
        padBack = pad_amount
        valid_size = total_size - padFront - padBack

        unneeded_remaining = unneeded_pad
        unneeded_pad_front = 0
        unneeded_pad_back = 0

        if padBack > 0 and unneeded_remaining > 0:
            reduce_amount = min(padBack, unneeded_remaining)
            unneeded_pad_back += reduce_amount
            padBack -= reduce_amount
            unneeded_remaining -= reduce_amount

        if valid_size > 0 and unneeded_remaining > 0:
            reduce_amount = min(valid_size, unneeded_remaining)
            unneeded_remaining -= reduce_amount

        if padFront > 0 and unneeded_remaining > 0:
            reduce_amount = min(padFront, unneeded_remaining)
            unneeded_pad_front += reduce_amount
            padFront -= reduce_amount
            unneeded_remaining -= reduce_amount

        return {
            "padFront_": padFront,
            "padBack_": padBack,
            "unneededPad_": unneeded_pad,
            "unneededPadFront_": unneeded_pad_front,
            "unneededPadBack_": unneeded_pad_back,
            "totalSize_": total_size,
            "stride_": stride,
            "dilation_": conv_params.get(f"dilation_{suffix}", 1),
            "windowDim_": window_dim,
        }

    def build_padding_sizes_variant(dim_sizes=None):
        """Build paddingSizes_ for one variant (top-level or per-core).

        Emits one entry per spatial axis whose output dim actually survives in
        the iteration space, mirroring ``_avgpool_sdsc_fields``.  A collapsed
        window (kernel extent 1) leaves its output-spatial label unassigned --
        e.g. K=1x1 with N=1 yields no ``j`` -- and subscripting ``dim_sizes``
        for the absent label would raise ``KeyError``.  Such an axis has no
        pooling/conv window to describe, so it is correctly omitted rather than
        defaulted.
        """
        variant = {}
        for suffix, pad_dim, kernel_key, stride_key, window_dim, total_key in (
            (
                "i",
                _CONV2D_PAD_DIM_I,
                "kernel_h",
                "stride_i",
                _CONV2D_WINDOW_DIM_I,
                "total_size_i",
            ),
            (
                "j",
                _CONV2D_PAD_DIM_J,
                "kernel_w",
                "stride_j",
                _CONV2D_WINDOW_DIM_J,
                "total_size_j",
            ),
        ):
            if dim_sizes is not None and Symbol(pad_dim) not in dim_sizes:
                continue
            variant[str(pad_dim)] = compute_padding_for_dim(
                suffix,
                pad_dim,
                kernel_key,
                stride_key,
                window_dim,
                total_key,
                dim_sizes,
            )
        return variant

    return {
        "padding_sizes": build_padding_sizes_variant(dim_sizes=None),
        "padding_sizes_per_core": build_padding_sizes_variant(
            dim_sizes=iteration_space
        ),
        "emit_memorg_padding": True,
    }


def _build_conv2d_symbol_mapping(
    op_spec: OpSpec,
    dim_labels: list[str],
) -> dict[Any, Symbol]:
    """Build the symbol mapping for depthwise conv2d from live dim roles.

    Labels come from ``_align_conv2d_dim_labels``, which derives which roles
    survived the pipeline's size-1 squeeze from the node's live NCHW output
    ranges plus the kernel extents.  Those labels are already in
    iteration-space order (output roles in canonical ``mb, out, i, j`` order,
    then the surviving window dims), so symbols map onto them positionally.

    This deliberately does **not** identify kernel dims by matching their extent
    against ``kernel_h``/``kernel_w``.  An extent of 1 both collapses the dim
    away and collides with any other unit dim, so extent matching cannot
    distinguish a kernel dim from a unit spatial dim -- and a squeeze that lands
    in the middle of the canonical order (``H_out == 1`` dropping ``i`` while
    ``mb`` survives) is invisible to any positional or count-based scheme.

    Falls back to the caller's positional labels only when the kernel extents
    or live ranges are unavailable, which is the pre-existing behaviour for
    op_specs that carry no ``conv_params``.
    """
    conv_params = op_spec.op_info.get("conv_params", {})
    kernel_h = conv_params.get("kernel_h")
    kernel_w = conv_params.get("kernel_w")
    sym_list = list(op_spec.iteration_space.keys())

    if kernel_h is None or kernel_w is None or op_spec.node_output_ranges is None:
        # No kernel sizes or no live output ranges: keep the caller's positional
        # mapping rather than guessing.
        return {sym: Symbol(dim_labels[i]) for i, sym in enumerate(sym_list)}

    labels = _align_conv2d_dim_labels(
        op_spec.node_output_ranges, len(sym_list), kernel_h, kernel_w
    )

    # The label list is in canonical order (mb, out, i, j, ki, kj minus the
    # squeezed roles), but the *iteration space* is not guaranteed to be.  It is
    # built write-dep-first and then extended with read-only symbols, so when the
    # output spatial dims collapse to 1x1 they drop out of the write dep and are
    # re-appended AFTER the kernel dims.
    # Assigning canonical labels positionally would swap i/j with ki/kj.
    #
    # Only a fully collapsed 1x1 output can reorder this way (a surviving spatial
    # dim stays in the write dep and keeps its slot), so handle exactly that case:
    # the unit-extent symbols are the output spatial dims and the kernel-extent
    # symbols are the window.
    window_labels = (_CONV2D_WINDOW_DIM_I, _CONV2D_WINDOW_DIM_J)
    spatial_labels = (_CONV2D_PAD_DIM_I, _CONV2D_PAD_DIM_J)
    needs_reorder = all(lbl in labels for lbl in window_labels) and all(
        lbl in labels for lbl in spatial_labels
    )
    if needs_reorder and _is_static_one(op_spec.node_output_ranges[2]):
        sym_to_size = {
            sym: _concretize_for_sdsc(size)
            for sym, (size, _) in op_spec.iteration_space.items()
        }
        # Symbols whose extent is 1 are the collapsed spatial dims; the rest, in
        # order, take the remaining canonical labels.
        unit_syms = [sym for sym in sym_list if sym_to_size.get(sym) == 1]
        if len(unit_syms) == 2:
            non_spatial = [lbl for lbl in labels if lbl not in spatial_labels]
            non_spatial_iter = iter(non_spatial)
            spatial_iter = iter(spatial_labels)
            return {
                sym: Symbol(
                    next(spatial_iter) if sym in unit_syms else next(non_spatial_iter)
                )
                for sym in sym_list
            }

    return {sym: Symbol(labels[i]) for i, sym in enumerate(sym_list)}


def _get_op_dim_labels(ndim: int, is_matmul: bool, is_conv2d: bool) -> list[str]:
    if is_matmul:
        return MATMUL_DIM_LABELS[len(MATMUL_DIM_LABELS) - ndim :]
    elif is_conv2d:
        return CONV2D_DIM_LABELS[len(CONV2D_DIM_LABELS) - ndim :]
    else:
        return INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]


def _get_tensor_layout_labels(use_op_dims: bool, op_name: str) -> list[str]:
    # use_op_dims is False only for matmul and conv (see the caller's
    # `not (_is_matmul or _is_conv)`), so the non-use_op_dims branch is reached
    # by those two families alone. CONV2D_LAYOUT_LABELS is the depthwise-conv
    # (#3510) layout; forward conv2d (#3284) uses MATMUL_LAYOUT_LABELS, the same
    # INPUT/KERNEL/OUTPUT tensor-role order it shipped with. Gating on
    # _is_depthwise_conv keeps forward conv (and matmul) on that order.
    if use_op_dims:
        return LAYOUT_LABELS
    if _is_depthwise_conv(op_name):
        return CONV2D_LAYOUT_LABELS
    return MATMUL_LAYOUT_LABELS


def _get_data_format(op, device_dtype):
    """Re-label int32 tensor data formats to fp32 for SDSC compatibility.

    NOTE: This is NOT a data conversion.
    This is only a temporary re-labeling of the same 32 bit data.
    The underlying data remains unchanged.

    In the long term, SDSC should accept int32 as the data format.
    Such re-labeling will become unnecessary.
    See backend issue deeptools#4307.
    """
    if device_dtype == DataFormats.IEEE_INT32 and op == IDENTITY_OP:
        return DataFormats.IEEE_FP32
    return device_dtype


def _get_sdsc_spec_data_format(op, arg_data_format):
    """Re-label int32 ops' SDSC spec data_format to fp32 for backend compatibility.

    For fp32<->int32 dtype-conversion ops, the SDSC spec must report fp32 as
    the op's data format, but unlike `_get_data_format`'s IDENTITY_OP case,
    the int32 tensor descriptor itself stays int32.
    See backend issue deeptools#4307.
    """
    if op in (FP32TOINT32_OP, INT32TOFP32_OP):
        return DataFormats.IEEE_FP32
    return arg_data_format


def _collect_index_tensor_layouts(
    op_spec: OpSpec,
    symbol_mapping: dict,
    index_tensor_indices: set[int],
    logger: object,
    mb_sym: Symbol | None = None,
    index_stick_syms: dict[int, Symbol] | None = None,
) -> tuple[dict, dict]:
    """First pass: compute (dim_order, stick_dim) for each index tensor.

    For P=1 gathers (mb_sym present, dim_order empty), override to ([mb_sym], mb_sym).
    For absent stick coordinate, append the pre-registered placeholder symbol.

    Returns:
        index_tensor_layouts: dict mapping tensor_idx -> (dim_order, stick_dim)
        index_active_dims: dict mapping tensor_idx -> set of active (non-stick) dims
    """
    index_tensor_layouts: dict[int, tuple[list, object]] = {}
    index_active_dims: dict[int, set] = {}
    index_stick_syms = index_stick_syms or {}

    for i in index_tensor_indices:
        arg = op_spec.args[i]
        dim_order, stick_dim = _get_device_dim_order(arg, symbol_mapping)
        if mb_sym is not None and not dim_order:
            # P=1: all-constant coords; use injected mb_sym.
            dim_order = [mb_sym]
            stick_dim = mb_sym
        elif dim_order and stick_dim is None and i in index_stick_syms:
            # Absent stick coordinate; append pre-registered placeholder.
            stick_dim = index_stick_syms[i]
            dim_order = dim_order + [stick_dim]
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
    injected_dims: dict[str, Any] | None = None,
) -> tuple[list[SDSCArgs], dict, Symbol | None]:
    dims = list(iteration_space.keys())
    if injected_dims is None:
        injected_dims = {}
    mb_sym = injected_dims.get("mb_sym")
    index_stick_syms = injected_dims.get("index_stick_syms")
    layouts: dict = {}
    # matmul and conv share the two-input tensor treatment: each arg keeps its
    # own natural (per-tensor) dim order and the weight gets the KERNEL layout
    # label. Reduced-dim appending (below) is a single-input-reduction concern.
    is_matmul = _is_matmul(op_spec.op)
    use_op_dims = not (is_matmul or _is_conv(op_spec.op))
    matmul_x_reuse_dims: list[Symbol] = []
    # Detect indirect access from device_coordinates: index tensors are those
    # whose name is referenced by an IndirectAccess in another tensor's coordinates,
    # and value tensors are those that contain IndirectAccess in their coordinates.
    index_tensor_indices = {
        i for i, arg in enumerate(op_spec.args) if is_index_tensor(arg, op_spec)
    }
    has_indirect_access = bool(index_tensor_indices)

    # For indirect access: pre-compute index tensor layouts (first pass).
    # mb_sym injects a dimension for P=1 gathers (all-constant coords).
    index_tensor_layouts: dict[int, tuple[list, Any]] = {}
    index_active_dims: dict[int, set] = {}
    if has_indirect_access:
        index_tensor_layouts, index_active_dims = _collect_index_tensor_layouts(
            op_spec,
            symbol_mapping,
            index_tensor_indices,
            logger,
            mb_sym=mb_sym,
            index_stick_syms=index_stick_syms,
        )

    missing_dim = None
    sdsc_args: list[SDSCArgs] = []
    matmul_n_dim = injected_dims.get("matmul_n_dim")

    for i, arg in enumerate(op_spec.args):
        is_fp8_mm_kernel_arg = arg.element_arrangement == ElementArrangement.QFP8WT

        # Step 1: Determine dimension order and stick dimension.
        # Index tensors use their pre-computed layout (their coords have no IndirectAccess).
        if has_indirect_access and i in index_tensor_layouts:
            dim_order, stick_dim = index_tensor_layouts[i]
        else:
            dim_order, stick_dim = _get_device_dim_order(
                arg, symbol_mapping, op_spec, tensor_position=i
            )

        if is_matmul and i == 0 and len(op_spec.args) >= 3 and matmul_n_dim is None:
            matmul_x_reuse_dims = _matmul_reuse_dims(op_spec, symbol_mapping, dim_order)

        # Case 2 (MutationLayoutSHOULDREMOVE) ops carry an authoritative
        # device-stride sympy.Expr for each coarse-tiled dim's per-iteration
        # advance, stamped by coarse_tile._propagate_tiled_op (host-stride
        # terms) and substituted to device-stride terms, per-arg, by
        # spyre_kernel.create_tensor_arg. The per-iteration *advance* across
        # levels is handled later, in compute_ops.generate_sdsc's
        # affine_strides construction (which is structured per level). Here
        # we only need the **iteration-0 base** fact -- the actual
        # (innermost) tile extent this arg is written/read at per
        # iteration, and the full extent it sits within -- to compute a
        # correct base offset/backGap, since device_coordinates cannot
        # represent "which supertile" for these ops (see
        # coarse_tiling_loops.md's IR-rewiring appendix). The innermost
        # level that tiles a given dim owns its true per-iteration
        # tile_size; the full extent is that tile_size times every level's
        # supertile_count for that dim.
        sdsc_dim_advance: dict[Symbol, tuple[int, int]] = {}
        if arg.device_tile_advance_expr is not None:
            arg_elem_bytes = num_bytes(arg.device_dtype)
            for level_syms in op_spec.tiled_symbols:
                for sym in level_syms:
                    if sym not in symbol_mapping:
                        continue
                    coeff = coeff_through_floor(arg.device_tile_advance_expr, sym)
                    if not coeff:
                        continue
                    tile_size = int(coeff) * arg_elem_bytes
                    trip_count = op_spec.tiled_symbol_trip_counts.get(sym, 1)
                    sdsc_sym = symbol_mapping[sym]
                    sdsc_dim_advance[sdsc_sym] = (tile_size, trip_count)

        scales: dict = {}
        strides: dict = {}
        offsets: dict = {}
        backGap: dict[Symbol, int] = {}
        max_dim_sizes: dict = {}
        reduced_dims: list = []

        # Step 2: Handle reduced dimensions — skip for index tensors.
        if (
            use_op_dims
            and dim_order != dims
            and not _is_topk(op_spec.op)
            and not _is_keep_by_index(op_spec.op)
        ):
            if not (has_indirect_access and i in index_tensor_indices):
                reduced_dims = [
                    d for d in op_dim_order if d not in dim_order and d is not mb_sym
                ]
                dim_order = dim_order + reduced_dims

        if is_matmul and i == 0 and matmul_x_reuse_dims:
            reduced_dims = reduced_dims + matmul_x_reuse_dims
            dim_order = dim_order + matmul_x_reuse_dims

        # Step 3: Handle missing stick dimension — skip for index tensors.
        if op_stick_dim is None:
            if not (has_indirect_access and i in index_tensor_indices):
                stick_dim = next(d for d in dims if d not in op_dim_order)
                # The chosen dim is absent from the *op*'s dim_order, but an
                # individual arg may already carry it: a conv2d kernel tensor
                # gets ki/kj added explicitly by _get_device_dim_order (they are
                # structural for the weight even when they do not appear in its
                # device_coordinates). Appending unconditionally would repeat the
                # dim -- e.g. a single-channel depthwise weight [1, 1, 3, 3] has
                # dim_order [kj, ki] and became [kj, ki, ki], which the scheduler
                # rejects with "external allocations with repeated dimensions".
                if stick_dim not in dim_order:
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

            # Injected stick dims don't exist in device_size; use stride=1.
            if (
                has_indirect_access
                and i in index_tensor_indices
                and dim in (index_stick_syms.values() if index_stick_syms else [])
            ):
                strides[dim] = 1
            else:
                strides[dim] = _calculate_device_stride(stride_idx, arg.device_size)
            offsets[dim] = 0
            dim_device_stride = math.prod(arg.device_size[-stride_idx - 1 :])

            if dim is stick_dim and dim in sdsc_dim_advance:
                # Authoritative fact from coarse_tile.py: the stick dim's
                # iteration-0 tile is tile_size elements out of
                # supertile_count tiles total (supertile_count already folds
                # in every nesting level that tiles this dim, when there is
                # more than one -- see the accumulation above).
                # _get_device_dim_order's dim_order walk can place the stick
                # dim at a different position for this (Case 2 / mutated) arg
                # than for its sibling args, which makes the stride_idx-based
                # arg.device_size[-stride_idx-2] lookup below read the wrong
                # slot for this arg specifically (see
                # coarse_tiling_loops.md's IR-rewiring appendix). Use the
                # authoritative supertile count for dev_dim_size instead of
                # trusting that slot. Scoped to the stick dim only: other
                # coarse-tiled dims (e.g. mb) already read the correct slot
                # via the existing device_size lookup for every arg in this
                # op, and overriding them too double-applies the tile split
                # baked into arg.device_size, corrupting an already-correct
                # stride (see the input mb regression this scoping fixes).
                # This establishes only the iteration-0 base offset/backGap;
                # the per-iteration advance across nesting levels is applied
                # separately in compute_ops.generate_sdsc's affine_strides.
                tile_size, supertile_count = sdsc_dim_advance[dim]
                dev_dim_size = tile_size * supertile_count
                it_dim_size = tile_size
            else:
                # A reduced dim (e.g. a conv's ki/kj on the output arg) has no
                # physical axis here, so stride_idx can run past device_size once
                # enough unit dims are squeezed out; fall back to the iteration
                # extent, which skips the padding corrections below.
                size_idx = -stride_idx - 2
                if -size_idx > len(arg.device_size):
                    dev_dim_size = iteration_space[dim]
                else:
                    dev_dim_size = arg.device_size[size_idx]
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

            # Same out-of-range case as the device_size lookup above: such a dim
            # has no device coordinate either, and this subscript would raise
            # before the size comparison below could skip it.
            coord_idx = -stride_idx - 2
            dim_coord = (
                arg.device_coordinates[coord_idx]
                if -coord_idx <= len(arg.device_coordinates)
                else None
            )
            if (
                dim_coord is not None
                and not isinstance(dim_coord, IndirectAccess)
                and dev_dim_size > it_dim_size
            ):
                dim_offset = int(dim_coord.as_coeff_Add()[0])
                offsets[dim] = dim_offset * dim_device_stride
                # conv2d addresses the difference between device and iteration space sizes
                # through the window/padding machinery in _conv2d_sdsc_fields, which already
                # accounts for the gap between the device extent and the
                # iteration extent. Emitting a backGap for a conv op double-counts
                # that gap and corrupts the generated addressing.
                #
                if not _is_conv(op_spec.op):
                    backGap[dim] = dev_dim_size - it_dim_size
                strides[dim] = strides[dim] // dev_dim_size * it_dim_size

        # Injected dimensions (mb_sym for P=1, stick symbols for absent coords)
        # require explicit max_dim_size: 1 for value/output, -1 for others.
        injected_dim_sizes: dict[Symbol, int] = {}

        # P=1 gather: mb_sym prepended to value/output tensors only (index
        # tensors already have it from _collect_index_tensor_layouts).
        is_index_p1 = (
            mb_sym is not None and has_indirect_access and i in index_tensor_indices
        )
        if mb_sym is not None and not is_index_p1:
            dim_order = [mb_sym] + dim_order
            scales[mb_sym] = 1
            strides[mb_sym] = _calculate_device_stride(0, arg.device_size)
            offsets[mb_sym] = 0
            injected_dim_sizes[mb_sym] = 1 if is_indirect_value_tensor(arg) else -1

        # Injected stick dims: set max_dim_size=1 for all tensors
        # that carry them in dim_order.
        if index_stick_syms:
            for idx, stick_sym in index_stick_syms.items():
                if stick_sym in dim_order:
                    injected_dim_sizes[stick_sym] = 1

        for dim, size in injected_dim_sizes.items():
            max_dim_sizes[dim] = size

        # For topk: inject topk_missing_dim only into output tensor's dim_order.
        topk_missing_dim = injected_dims.get("topk_missing_dim")
        if topk_missing_dim is not None and i == len(op_spec.args) - 1:
            dim_order = dim_order + [topk_missing_dim]
            scales[topk_missing_dim] = 1
            strides[topk_missing_dim] = _calculate_device_stride(
                len(dim_order) - 1, arg.device_size
            )
            offsets[topk_missing_dim] = 0
            max_dim_sizes[topk_missing_dim] = -1

        # For matmul N=1: inject matmul_n_dim into y (i==1) and output (i==-1) tensors.
        if (
            matmul_n_dim is not None
            and is_matmul
            and (i == 1 or i == len(op_spec.args) - 1)
        ):
            dim_order = dim_order + [matmul_n_dim]
            stick_dim = (
                matmul_n_dim  # Override stick_dim to be the injected N dimension
            )
            scales[matmul_n_dim] = 1
            strides[matmul_n_dim] = _calculate_device_stride(
                len(dim_order) - 1, arg.device_size
            )
            offsets[matmul_n_dim] = 0
            max_dim_sizes[matmul_n_dim] = -1

        effective_stick = [op_stick_dim if stick_dim is None else stick_dim]
        layout_labels = _get_tensor_layout_labels(use_op_dims, op_spec.op)

        # Special handling for FP8 matmul KERNEL tensor
        dtype_stick_size = arg.device_dtype.elems_per_stick()
        layout_stick_size = [dtype_stick_size]
        if is_fp8_mm_kernel_arg:
            # FP8 KERNEL needs 2D stick: [2, stick_size/2]
            layout_stick_size = [2, dtype_stick_size // 2]
            # Use the last two dimensions from dim_order for 2D stick
            effective_stick = dim_order[-2:]

        if has_indirect_access:
            label = get_indirect_layout_label(
                i,
                index_tensor_indices,
                layouts,
                dim_order,
                effective_stick,
                layout_stick_size,
                layout_labels,
                _get_layout_label,
                logger,
            )
        else:
            label = _get_layout_label(
                layouts,
                dim_order,
                effective_stick,
                layout_stick_size,
                layout_labels,
            )

        # Index tensors carry 32-bit integer indices; re-label as SENUINT32 since
        # the backend doesn't yet accept IEEE_INT32 in SDSC (deeptools #4307).
        arg_data_format = (
            DataFormats.SENUINT32
            if (has_indirect_access and i in index_tensor_indices)
            else _get_data_format(op_spec.op, arg.device_dtype)
        )

        # allocation keys are mutually exclusive (see TensorArg.allocation
        # docstring in op_spec.py); this chain just reads whichever one is
        # present. Priority order here is cosmetic, not semantic.
        start_addr = (
            arg.allocation.get("hbm_pool")
            if "hbm_pool" in arg.allocation
            else arg.allocation.get("lx")
            if "lx" in arg.allocation
            else arg.allocation.get("hbm")
        )

        is_idx_tensor = has_indirect_access and i in index_tensor_indices
        related_val_idx = (
            get_value_tensor_idx_for_index(op_spec, i) if is_idx_tensor else -1
        )

        sdsc_arg = SDSCArgs(
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
            device_tile_advance_expr=arg.device_tile_advance_expr,
        )
        if arg.work_division is not None:
            sdsc_arg.work_division = arg.work_division.remap_symbols(symbol_mapping)
        sdsc_args.append(sdsc_arg)

    return sdsc_args, layouts, missing_dim


def _get_op_func(op: str, is_reduction: bool, output_scales: dict) -> str:
    if _is_pool(op) or _is_conv(op):
        return op
    if (
        is_reduction
        and not _is_matmul(op)
        and not _is_topk(op)
        and not _is_conv(op)
        and -2 not in output_scales.values()
        and not _is_keep_by_index(op)
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
        # This is a correctness-critical boundary: the SDSC JSON / DeepTools
        # backend needs the *true* concrete size, not an optimization heuristic.
        # guarding_hint_or_throw resolves backed symbols and raises on unbacked
        # ones, rather than silently emitting a fallback (e.g. sys.maxsize) size.
        return V.graph.sizevars.guarding_hint_or_throw(expr)
    return int(expr)


def _resolve_sdsc_size(expr: Expr, symbolic_dim_bounds: dict) -> int:
    """Resolve an iteration-space size for SDSC generation.

    For symbolic dims, reads the max from symbolic_dim_bounds (computed at
    codegen time from ShapeEnv, serialized as plain ints into the generated
    file) so this works during the reload phase when ShapeEnv is gone.
    Falls back to _concretize_for_sdsc for concrete expressions.
    """
    # ``symbolic_dim_bounds`` is keyed by a single symbol name, so it can only
    # answer for a single-symbol expression; a multi-symbol one falls through to
    # ``_concretize_for_sdsc``, which resolves the whole expression. Reading one
    # arbitrary symbol's bound out of the ``free_symbols`` set both answered the
    # wrong question and made the answer depend on PYTHONHASHSEED.
    syms = getattr(expr, "free_symbols", None)
    if syms and len(syms) == 1:
        sym_name = str(next(iter(syms)))
        if sym_name in symbolic_dim_bounds:
            return symbolic_dim_bounds[sym_name][0]  # max
    return _concretize_for_sdsc(expr)


def _ref_arg(op_spec):
    if op_spec.is_reduction:
        return op_spec.args[0]

    return op_spec.args[-1]


def _round_up_to_stick(
    sdsc_iteration_space: dict,
    sym,
    stick_size: int,
    caller: str,
) -> None:
    """Round ``sdsc_iteration_space[sym]`` up to the next stick boundary."""
    cur = sdsc_iteration_space[sym]
    padded = ((cur + stick_size - 1) // stick_size) * stick_size
    if padded > cur:
        logger.debug("%s: extending %s %d -> %d", caller, sym, cur, padded)
        sdsc_iteration_space[sym] = padded


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
    k_sym = min(k_candidates, key=str)

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
    _round_up_to_stick(
        sdsc_iteration_space, k_sym, stick_size, "_extend_matmul_k_to_padded"
    )


def _matmul_reuse_dims(
    op_spec: OpSpec,
    symbol_mapping: dict,
    x_dim_order: list[Symbol],
) -> list[Symbol]:
    """Dims x is broadcast over (in y/output, not in x, excluding N).

    matmul broadcasts x over batch dims it doesn't index (issue #3888/#3927).
    These are indistinguishable from the true generated dim N by set membership
    alone. Use layout policy to disambiguate: y and output always stick on N,
    broadcast-batch dims never do. N = y's stick dim; others are reuse dims.
    """
    y_arg = op_spec.args[1]
    out_arg = op_spec.args[-1]
    y_dim_order, y_stick = _get_device_dim_order(y_arg, symbol_mapping)
    out_dim_order, out_stick = _get_device_dim_order(out_arg, symbol_mapping)
    if y_stick is None:
        return []
    y_syms = set(y_dim_order) | {y_stick}
    out_syms = set(out_dim_order) | ({out_stick} if out_stick is not None else set())
    reuse_syms = (y_syms & out_syms) - set(x_dim_order) - {y_stick}
    return [d for d in y_dim_order if d in reuse_syms]


def _extend_restickify_to_padded(
    op_spec: OpSpec,
    sdsc_iteration_space: dict,
    symbol_mapping: dict,
) -> None:
    """Round sdsc_iteration_space[stick_sym] up to a stick boundary for each
    restickify arg.  Both input (old stick) and output (new stick) may carry
    the unaligned iter, so we extend per-arg.

    Running before ``_create_sdsc_tensors`` keeps backGap correct: the
    unaligned-stick arg gets dev_dim_size==it_dim_size on the within-stick
    axis (no backGap), and the other arg's outer-split strides are computed
    against the padded extent (no stale-stride mismatch with the later
    widening done by ``_get_padded_iteration_space``).
    """
    for arg in op_spec.args:
        _, stick_sym = _get_device_dim_order(arg, symbol_mapping)
        if stick_sym is None or stick_sym not in sdsc_iteration_space:
            continue
        stick_size = arg.device_dtype.elems_per_stick()
        _round_up_to_stick(
            sdsc_iteration_space, stick_sym, stick_size, "_extend_restickify_to_padded"
        )


def _inject_implicit_conv_kernel_dims(
    is_conv2d: bool,
    op_spec: OpSpec,
    sdsc_iteration_space: dict,
    dim_splits: dict,
    work_slices: dict,
) -> None:
    """Inject implicit kernel dimensions (ki, kj) for conv2d when kernel_size=1.

    When kernel_size=1, the kernel dimensions don't iterate naturally (they'd be
    0..0), so they don't appear in op_spec.iteration_space. We inject them here
    with their actual kernel sizes so they appear in sdsc_iteration_space and
    can be included in the kernel tensor's layoutDimOrder and coordinates_.
    """
    if not is_conv2d:
        return

    conv_params = op_spec.op_info.get("conv_params", {})
    kernel_h = conv_params.get("kernel_h", 1)
    kernel_w = conv_params.get("kernel_w", 1)

    ki_sym = Symbol("ki")
    kj_sym = Symbol("kj")

    # Inject missing kernel dimensions. For 1xN or Nx1 kernels, one dimension iterates
    # naturally (already in iteration_space) and the other is implicit (needs injection).
    if ki_sym not in sdsc_iteration_space:
        sdsc_iteration_space[ki_sym] = kernel_h
        dim_splits[ki_sym] = 1
        work_slices[ki_sym] = 1

    if kj_sym not in sdsc_iteration_space:
        sdsc_iteration_space[kj_sym] = kernel_w
        dim_splits[kj_sym] = 1
        work_slices[kj_sym] = 1


def _finalize_tensor_work_divisions(
    args: list[SDSCArgs],
    mapping_dims: tuple[Symbol, ...],
    work_slices: dict[Symbol, Any],
    core_map: dict[Symbol, Expr],
    num_cores: int,
    is_lx_relayout: bool,
) -> None:
    """Give every tensor one effective ownership after SDSC normalization."""

    operation_work_division = TensorWorkDivision(
        {dim: work_slices[dim] for dim in mapping_dims},
        {dim: core_map[dim] for dim in mapping_dims},
        num_cores=num_cores,
    )
    assert is_lx_relayout or all(arg.work_division is None for arg in args), (
        "per-tensor ownership is supported only for LX relayout identities"
    )
    for arg in args:
        override = arg.work_division
        # A relayout tensor can override the operation-wide split on selected
        # dimensions; unsplit dimensions inherit one slice owned by core zero.
        effective = (
            operation_work_division
            if override is None
            else TensorWorkDivision(
                {dim: int(override.work_slices.get(dim, 1)) for dim in mapping_dims},
                {
                    dim: override.core_id_to_work_slice.get(dim, Integer(0))
                    for dim in mapping_dims
                },
                num_cores=override.num_cores or num_cores,
            )
        )
        tensor_cores = effective.num_cores or num_cores
        tensor_owners = math.prod(effective.work_slices.values())
        valid = (
            tensor_cores == num_cores == tensor_owners
            if not is_lx_relayout
            else num_cores % tensor_cores == 0 and tensor_cores % tensor_owners == 0
        )
        if not valid:
            raise ValueError(
                f"tensor ownership uses {tensor_owners} slices across "
                f"{tensor_cores} of {num_cores} operation cores"
            )
        arg.work_division = effective


def parse_op_spec(op_spec: OpSpec) -> tuple["SDSCSpec", "dict"]:
    is_matmul = _is_matmul(op_spec.op)
    is_conv2d = _is_conv(op_spec.op)
    is_relayout = is_lx_relayout_identity(op_spec.op, op_spec.args)
    is_restickify = op_spec.op == RESTICKIFY_OP
    is_pool = _is_pool(op_spec.op)
    is_conv = _is_conv(op_spec.op)
    ndim = len(op_spec.iteration_space)
    # Detect indirect access from device_coordinates: index tensors are those
    # whose name is referenced by an IndirectAccess in another tensor's coordinates,
    # and value tensors are those that contain IndirectAccess in their coordinates.
    index_tensor_indices = {
        i for i, arg in enumerate(op_spec.args) if is_index_tensor(arg, op_spec)
    }
    has_indirect_access = bool(index_tensor_indices)

    symbol_mapping = None
    if op_spec.op == CONV2D_FWD_OP:
        # Forward conv2d (#3284) is a Reduction whose iteration space appends its
        # reduction axes (in/ki/kj) in data-dependent read-dep access order, so
        # the contraction dim and the kernel taps cannot be told apart
        # positionally.  Recover each dim's role from what the args' access
        # expressions already carry -- set membership and co-occurrence
        # (_match_labels_by_structure) -- rather than from a size snapshot.  This
        # needs no live IR ranges, drops squeezed size-1 dims for free, and has
        # no C_in-vs-tap size-collision constraint.  Falls back to the positional
        # mapping below only if the arg structure is unexpected.
        dim_labels = [label for _role, label in _CONV_ROLE_LABELS][-ndim:]
        symbol_mapping = _match_labels_by_structure(op_spec)
    elif is_pool:
        # Pool survival is read from the node's live output ranges (NCHW); the
        # lowering supplies no size snapshot.  Positional mapping (below) is
        # correct because pool is a single-input reduction with a fixed
        # iteration-space order.
        dim_labels = _align_pool_dim_labels(op_spec.node_output_ranges, ndim)
    else:
        dim_labels = _get_op_dim_labels(ndim, is_matmul, is_conv2d)
        # Depthwise conv2d (#3510): size-based label matching. Forward conv2d is
        # handled by the CONV2D_FWD_OP branch above; matmul/elementwise stay
        # positional (falls through to the None-guard below).
        if (
            symbol_mapping is None
            and is_conv2d
            and op_spec.op_info
            and "conv_params" in op_spec.op_info
        ):
            symbol_mapping = _build_conv2d_symbol_mapping(op_spec, dim_labels)
    # Forward conv (structural recovery) and depthwise (above) may already have
    # set symbol_mapping; everything else gets a positional mapping.
    if symbol_mapping is None:
        symbol_mapping = {
            sym: Symbol(dim_labels[i]) for i, sym in enumerate(op_spec.iteration_space)
        }
    logger.debug(
        "symbol mapping: %s",
        ", ".join(f"{k} -> {v}" for k, v in symbol_mapping.items()),
    )
    # Minted per-(op, level) tile-advance symbols (see spyre_kernel.py's
    # _get_or_mint_level_symbol) are not iteration-space dimensions -- they are
    # loop-nesting-level markers -- so they have no dim label to rename to.
    # Register each as an identity mapping instead, so compile_op_spec's
    # `symbol_mapping[s]` lookup for op_spec.tiled_symbols does not silently
    # drop them. setdefault never overwrites a real-symbol entry above, and
    # collides with none: minted names (`_tile_adv_{op_name}_lvl{n}`) can
    # never equal a dim label or a real Inductor `d{i}` symbol name.
    for level in op_spec.tiled_symbols:
        for sym in level:
            symbol_mapping.setdefault(sym, sym)
    logger.debug(
        "symbol mapping: %s",
        ", ".join(f"{k} -> {v}" for k, v in symbol_mapping.items()),
    )

    # For symbolic dims, use the max from symbolic_dim_bounds as the iteration-space size
    # so the emitted SDSC JSON is generated max sizes baked in, not symbols.
    sdsc_iteration_space = {
        symbol_mapping[sym]: _resolve_sdsc_size(size, op_spec.symbolic_dim_bounds)
        for sym, (size, _) in op_spec.iteration_space.items()
    }

    # Build the SDSC dim name -> (pytorch_sym_name, granularity, max_val) map
    # for any iteration-space dims.
    # This drives symbolicDimInfo_ and dimToSymbolMapping_ in the generated JSON.
    symbolic_dims: dict[str, tuple[str, int, int]] = {}
    for sym, (size_expr, _) in op_spec.iteration_space.items():
        sdsc_dim_name = str(symbol_mapping[sym])
        sym_str = str(size_expr)
        if sym_str in op_spec.symbolic_dim_bounds:
            max_val, granularity = op_spec.symbolic_dim_bounds[sym_str]
            symbolic_dims[sdsc_dim_name] = (sym_str, granularity, max_val)

    dim_splits = {
        symbol_mapping[dim]: value[-1] for dim, value in op_spec.iteration_space.items()
    }
    num_cores = math.prod(dim_splits.values())

    work_slices = {
        symbol_mapping[sym]: wk_slice
        for sym, (_, wk_slice) in op_spec.iteration_space.items()
    }
    if op_spec.core_id_to_work_slice is None:
        raise ValueError("OpSpec is missing its finalized core mapping")
    core_id_to_work_slice = {
        symbol_mapping[sym]: expression
        for sym, expression in op_spec.core_id_to_work_slice.items()
    }

    # Inject implicit kernel dimensions for conv2d when kernel_size=1. This is
    # the depthwise-conv (#3510) path only: forward conv2d (#3284) sources its
    # kernel dims from the node's live ranges via the CONV2D_FWD_OP branch and
    # must not have ki/kj injected here (that would add dims it never emitted).
    _inject_implicit_conv_kernel_dims(
        _is_depthwise_conv(op_spec.op),
        op_spec,
        sdsc_iteration_space,
        dim_splits,
        work_slices,
    )

    ref_arg = _ref_arg(op_spec)
    op_dim_order, op_stick_dim = _get_device_dim_order(ref_arg, symbol_mapping)

    # On-device type-conversion ops (DL16TOFP32/FP32TODL16, not identity)
    # require at least one outer spatial dim beyond the stick; inject a
    # virtual mb=1 row when the op's tensor has only the stick dim.
    mb_sym: Symbol | None = None
    if (
        (DtypeOpTable.is_dtype_op(op_spec.op) or op_spec.op == "qfp8ch")
        and op_spec.op != IDENTITY_OP
        and op_stick_dim is not None
        and all(d is op_stick_dim for d in op_dim_order)
    ):
        mb_sym = Symbol(INPUT_DIM_LABELS[0])
        sdsc_iteration_space = {mb_sym: 1, **sdsc_iteration_space}
        dim_splits = {mb_sym: 1, **dim_splits}
        work_slices = {mb_sym: 1, **work_slices}
        op_dim_order = [mb_sym] + op_dim_order

    # Inject missing dimensions into index tensors: P=1 (no loops) or
    # absent stick coordinate (size-1 logical dim).
    def _inject_index_dim(sym: Symbol, prepend: bool = False) -> None:
        """Register injected dimension into iteration space and dim_order."""
        sdsc_iteration_space[sym] = 1
        dim_splits[sym] = 1
        work_slices[sym] = 1
        nonlocal op_dim_order
        if prepend:
            op_dim_order = [sym] + op_dim_order
        elif sym not in op_dim_order:
            op_dim_order = op_dim_order + [sym]

    if has_indirect_access and mb_sym is None:
        for idx in index_tensor_indices:
            idx_arg = op_spec.args[idx]
            idx_dim_order, _ = _get_device_dim_order(idx_arg, symbol_mapping)
            if not idx_dim_order:
                # P=1: all-constant coords, no loop variable.
                _existing_names = {s.name for s in op_dim_order}
                _p1_label = next(
                    lbl for lbl in INPUT_DIM_LABELS if lbl not in _existing_names
                )
                mb_sym = Symbol(_p1_label)
                _inject_index_dim(mb_sym, prepend=True)
                logger.debug(
                    "P=1 gather detected (index tensor %d): injecting virtual %s=1",
                    idx,
                    _p1_label,
                )
                break

    # Absent stick coordinate: batch dim present but stick collapsed to 0.
    index_stick_syms: dict[int, Symbol] = {}
    if has_indirect_access:
        for idx in index_tensor_indices:
            idx_arg = op_spec.args[idx]
            idx_dim_order, idx_stick_dim = _get_device_dim_order(
                idx_arg, symbol_mapping
            )
            if idx_dim_order and idx_stick_dim is None:
                _existing_names = {s.name for s in op_dim_order} | {
                    s.name for s in idx_dim_order
                }
                _stick_label = next(
                    lbl for lbl in INPUT_DIM_LABELS if lbl not in _existing_names
                )
                stick_sym = Symbol(_stick_label)
                _inject_index_dim(stick_sym, prepend=False)
                index_stick_syms[idx] = stick_sym
                logger.debug(
                    "Index tensor %d: stick coordinate absent; injecting %s",
                    idx,
                    _stick_label,
                )

    if op_stick_dim is None:
        if is_pool or _is_depthwise_conv(op_spec.op):
            # Pool/depthwise-conv op where C fits in one stick (e.g. C=1): the
            # "out" (channel) dimension was dropped from the iteration space
            # because its size is 1, but the SDSC still needs it.  Take the
            # channel count from the node's live NCHW output ranges (position 1)
            # rather than the physical device layout, which rounds channel up to
            # a full stick and so cannot recover C when C < elems_per_stick.
            # (Using INPUT_DIM_LABELS[ndim] would collide with the dim labels
            # "i", "j", "ki", "kj".)  Forward conv2d (#3284) is not diverted here:
            # its C_in stick-alignment gate guarantees a real stick dim, so it
            # keeps the original INPUT_DIM_LABELS[ndim] else-branch below.
            stick_sym = Symbol("out")
            # _align_pool_dim_labels / _align_conv2d_dim_labels already rejected a
            # None here; restate the invariant so the index is well-typed.
            assert op_spec.node_output_ranges is not None
            if _is_depthwise_conv(op_spec.op):
                # Front-insert for conv2d, append for pool.  ``_create_sdsc_tensors``
                # step 3 takes the stick as the FIRST iteration-space dim missing
                # from ``op_dim_order``.  Both ops have ki/kj in the iteration
                # space, but they differ in the reference arg (args[0], the input):
                #   pool  op_dim_order = [kj, j, ki, i]  -- window dims present
                #   conv  op_dim_order = [j, i]          -- window dims absent
                # So pool skips ki/kj and reaches "out" wherever it sits, while for
                # conv an appended "out" loses to ki -- whose cardinality is the
                # kernel extent (e.g. 3), and the scheduler then aborts with
                # "[distributeElemArrToTemporalLoops] Not enough elements to
                # distribute ... requires 64 elements".  Front-inserting makes the
                # channel dim the first candidate, so the input and the weight both
                # get a full-width channel stick.
                sdsc_iteration_space = {
                    stick_sym: int(op_spec.node_output_ranges[1]),
                    **sdsc_iteration_space,
                }
            else:
                sdsc_iteration_space[stick_sym] = int(op_spec.node_output_ranges[1])
        else:
            stick_sym = Symbol(INPUT_DIM_LABELS[ndim])
            sdsc_iteration_space[stick_sym] = op_spec.args[
                0
            ].device_dtype.elems_per_stick()
        work_slices[stick_sym] = 1
        dim_splits[stick_sym] = 1

    if is_matmul:
        _extend_matmul_k_to_padded(op_spec, sdsc_iteration_space, symbol_mapping)
    elif is_restickify:
        _extend_restickify_to_padded(op_spec, sdsc_iteration_space, symbol_mapping)

    # Grow the index-entry iteration to the padded output device_size so a
    # partial-last-stick gather splits stick-aligned across cores. The output's
    # entry-dim device_size was rounded up to the index stick multiple at layout
    # time (enforce_indirect_access_layout); match the SDSC iteration to it BEFORE
    # _create_sdsc_tensors so the output's per-core base stride is computed from
    # the padded (stick-aligned) size rather than the shorter logical count.
    # Otherwise the per-core base lands element-aligned (mid-stick) and the split
    # miscompiles. No-op unless the output was actually padded (device_size >
    # iteration), i.e. only for the multi-core partial-stick case.
    if has_indirect_access and _spyre_config.sencores > 1:
        idx_arg = op_spec.args[next(iter(index_tensor_indices))]
        idx_stick = idx_arg.device_coordinates[-1]
        if len(idx_stick.free_symbols) == 1:
            entry_c = next(iter(idx_stick.free_symbols))
            out_arg = op_spec.args[-1]
            for pos, coord in enumerate(out_arg.device_coordinates[:-1]):
                if coord.free_symbols == {entry_c}:
                    entry_mb = symbol_mapping.get(entry_c)
                    dev = int(out_arg.device_size[pos])
                    if (
                        entry_mb in sdsc_iteration_space
                        and dev > sdsc_iteration_space[entry_mb]
                    ):
                        sdsc_iteration_space[entry_mb] = dev
                    break

    # For topk: if all output dims are in the input, add a missing dimension.
    injected_dims = {"mb_sym": mb_sym} if mb_sym else {}
    if index_stick_syms:
        injected_dims["index_stick_syms"] = index_stick_syms
    if _is_topk(op_spec.op) and len(op_spec.args) >= 2:
        input_arg = op_spec.args[0]
        output_arg = op_spec.args[-1]
        input_dim_order, _ = _get_device_dim_order(input_arg, symbol_mapping, op_spec)
        output_dim_order, _ = _get_device_dim_order(output_arg, symbol_mapping, op_spec)
        output_only_dims = [d for d in output_dim_order if d not in input_dim_order]
        if not output_only_dims:
            # No new output dimension; add one to the iteration space with size 1.
            idx = len(sdsc_iteration_space)
            if idx < len(INPUT_DIM_LABELS):
                missing_dim_label = INPUT_DIM_LABELS[idx]
                topk_missing_dim = Symbol(missing_dim_label)
                sdsc_iteration_space[topk_missing_dim] = 1
                dim_splits[topk_missing_dim] = 1
                work_slices[topk_missing_dim] = 1
                injected_dims["topk_missing_dim"] = topk_missing_dim

    # For matmul N=1: inject N dimension into iteration space so y and output have it.
    if is_matmul:
        y_arg = op_spec.args[1]
        out_arg = op_spec.args[-1]
        y_stick_coord = (
            y_arg.device_coordinates[-1] if len(y_arg.device_coordinates) > 0 else None
        )
        out_stick_coord = (
            out_arg.device_coordinates[-1]
            if len(out_arg.device_coordinates) > 0
            else None
        )
        if y_stick_coord == 0 and out_stick_coord == 0:
            idx = len(sdsc_iteration_space)
            if idx < len(INPUT_DIM_LABELS):
                n_dim_label = INPUT_DIM_LABELS[idx]
                n_dim = Symbol(n_dim_label)
                sdsc_iteration_space[n_dim] = 1
                dim_splits[n_dim] = 1
                work_slices[n_dim] = 1
                injected_dims["matmul_n_dim"] = n_dim

    args, layouts, missing_dim = _create_sdsc_tensors(
        op_spec,
        symbol_mapping,
        sdsc_iteration_space,
        op_dim_order,
        op_stick_dim,
        injected_dims=injected_dims,
    )
    if missing_dim is not None:
        # A dimension was added to the iteration space, update splits and work slices
        dim_splits[missing_dim] = 1
        work_slices[missing_dim] = 1

    # In case of same type conversion (identity op) user gets compile time error & avoid
    # changing the padding logic here to fix errors with torch.split() for 3d shapes.
    is_dtype_op = DtypeOpTable.is_dtype_op(op_spec.op) and op_spec.op != IDENTITY_OP
    if is_matmul or is_conv or is_dtype_op:
        # Two-input reductions pad every arg so the activation (INPUT), weight
        # (KERNEL) and output stick dims are each rounded to the stick boundary.
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
    elif is_restickify:
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
    if is_restickify:
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
                if dim_sym is None or dim_sym in stick_dim:
                    continue
                padded_it_size = sdsc_iteration_space[dim_sym]
                dev_dim_size = op_spec_arg.device_size[coord_idx]
                if dev_dim_size < padded_it_size:
                    sdsc_arg.backGap[dim_sym] = padded_it_size - dev_dim_size
        for dim in padding:
            if dim_splits[dim] != 1:
                raise ValueError(
                    f"restickify padding dimension {dim} must be unsplit before codegen"
                )

    conv_params = (
        dict(op_spec.op_info.get("conv_params", {})) if op_spec.op_info else {}
    )
    pool_params_out: dict = {}
    if is_pool and op_spec.op_info:
        pool_params_out = dict(op_spec.op_info.get("constants", {}))
        scaling_factor = pool_params_out.get("scaling_factor", 1.0)
        constants = {"nmap": scaling_factor}
    else:
        constants = (
            dict(op_spec.op_info.get("constants", {})) if op_spec.op_info else {}
        )
    coordinate_masking = _get_coordinate_mask(
        sdsc_iteration_space, args[-1], padding, op_spec.op
    )
    if coordinate_masking:
        constants["samv-maskvalue"] = _get_mask_value(op_spec.op)

    # Forward conv2d (#3284), like matmul, counts only the non-output args as
    # inputs. Depthwise conv2d (#3510) is a reduction that feeds its output
    # tensor back as an accumulator input, so it must fall through to the
    # reduction branch (len(args)); using `is_conv` here would wrongly drop that
    # third labeled input for depthwise.
    num_inputs = (
        len(args[:-1])
        if is_matmul or op_spec.op == CONV2D_FWD_OP or not op_spec.is_reduction
        else len(args)
    )

    if _is_topk(op_spec.op):
        num_inputs = 1  # topk has exactly 1 input tensor and 1 output tensor

    if _is_keep_by_index(op_spec.op):
        num_inputs = 2  # keep_by_index has exactly 2 input tensors (values, indices)

    if is_pool:
        num_inputs = 1  # avgpool has exactly 1 input tensor and 1 output tensor
        # The pool hardware accumulates the full kernel window on each core.
        # Splitting ki/kj across cores produces partial sums, giving wrong results.
        for _k_sym in (Symbol("ki"), Symbol("kj")):
            if dim_splits.get(_k_sym, 1) != 1:
                raise ValueError(
                    f"pool kernel dimension {_k_sym} must be unsplit before codegen"
                )

    if is_conv:
        # Both conv paths accumulate the full kernel window per core; splitting
        # ki/kj gives partial sums (same constraint as pool). The in-channel
        # reduction MAY be core-split (matmul K via psum), so it is left to the
        # default work division.
        for _k_sym in (Symbol("ki"), Symbol("kj")):
            if dim_splits.get(_k_sym, 1) != 1:
                raise ValueError(
                    f"convolution kernel dimension {_k_sym} must be unsplit before codegen"
                )

        if _spyre_config.disable_conv2d_spatial_split:
            # Strided convs cannot split the output spatial dims (i/j): a strided
            # window's per-core output coordinates do not map to a contiguous
            # input span, so a spatial split shuffles the result. That constraint
            # is enforced during the work-split pass -- conv_spatial_blocked_vars
            # in work_division_constraints.py, for both the forward (#3284) and
            # depthwise (#3510) paths -- so it is NOT overridden here. Span
            # reduction outranks the flag: if the hardware span limit forced a
            # spatial split, work division kept it -- warn if so.
            _sI = _try_static_int(
                conv_params.get("stride_i", conv_params.get("stride_h", 1))
            )
            _sJ = _try_static_int(
                conv_params.get("stride_j", conv_params.get("stride_w", 1))
            )
            if (_sI or 1) > 1 or (_sJ or 1) > 1:
                for _spatial_sym in (Symbol("i"), Symbol("j")):
                    if dim_splits.get(_spatial_sym, 1) > 1:
                        logger.warning(
                            f"strided conv2d {op_spec.op}: spatial dim "
                            f"{_spatial_sym} is split {dim_splits[_spatial_sym]} "
                            f"ways; expected work division to block it unless the "
                            f"memory-span limit required the split."
                        )
    # Pool-specific SDSC field values (#3510).  Empty for non-pool ops.
    pool_sdsc_fields = (
        _avgpool_sdsc_fields(sdsc_iteration_space, pool_params_out) if is_pool else {}
    )
    # Forward-conv2d windowed SDSC fields (#3284): the windowed-input half is the
    # same sliding-window pattern as pool (dilation==1 guaranteed by the
    # direct-lowering gate), so it reuses the avgpool builder.  Depthwise conv2d
    # uses _conv2d_sdsc_fields (below) instead, so this stays empty for it and
    # the two field dicts never both populate the same keys.
    if op_spec.op == CONV2D_FWD_OP and op_spec.op_info:
        window_sdsc_fields = _avgpool_sdsc_fields(
            sdsc_iteration_space, op_spec.op_info.get("conv_params", {})
        )
    else:
        window_sdsc_fields = {}

    # Unit dimensions may be injected while translating the completed OpSpec
    # into backend labels. They are unsplit and therefore owned at slice zero.
    # Every non-unit dimension must already have a final assignment.
    mapping_dims = tuple(sdsc_iteration_space)
    if is_relayout:
        num_cores = max(
            num_cores,
            *(arg.work_division.num_cores or 1 for arg in args if arg.work_division),
        )
    for dim in mapping_dims:
        if dim not in core_id_to_work_slice:
            if int(dim_splits[dim]) != 1:
                raise ValueError(f"final core mapping is missing split dimension {dim}")
            core_id_to_work_slice[dim] = Integer(0)
    _finalize_tensor_work_divisions(
        args,
        mapping_dims,
        work_slices,
        core_id_to_work_slice,
        num_cores,
        is_relayout,
    )
    # Collect index tensor indices for indirect access
    indirect_access_indices = [
        i for i, arg in enumerate(op_spec.args) if is_index_tensor(arg, op_spec)
    ]

    # Depthwise-conv2d-specific SDSC fields (#3510): top-level and per-core
    # padding sizes.  Gated to depthwise only -- forward conv2d uses
    # window_sdsc_fields instead, so the two never populate overlapping keys.
    conv2d_sdsc_fields = (
        _conv2d_sdsc_fields(sdsc_iteration_space, conv_params, work_slices)
        if _is_depthwise_conv(op_spec.op) and conv_params
        else {}
    )

    return (
        SDSCSpec(
            opfunc=(
                "shuffle"
                if is_relayout
                else _get_op_func(op_spec.op, op_spec.is_reduction, args[-1].scales)
            ),
            # Forward conv2d (#3284) is a native "pt" (processing-tile) op like
            # matmul; depthwise conv2d (#3510) runs on the "sfp" unit. `is_conv`
            # matches both, so dispatch forward explicitly and leave depthwise
            # (and every non-matmul op) on "sfp".
            execution_unit="pt"
            if (is_matmul or op_spec.op == CONV2D_FWD_OP)
            else "sfp",
            data_format=_get_sdsc_spec_data_format(
                op_spec.op,
                args[1 if indirect_access_indices else 0].data_format,
            ),  # TODO: op_spec needs operation data format. Use value tensor (args[1]) for indirect access ops
            num_inputs=num_inputs,
            iteration_space=sdsc_iteration_space,
            num_cores=num_cores,
            work_slices=work_slices,
            core_id_to_work_slice=core_id_to_work_slice,
            padding=padding,
            layouts=layouts,
            args=args,
            constants=constants,
            conv_params=conv_params,
            coordinate_masking=coordinate_masking,
            symbolic_dims=symbolic_dims,
            indirect_access_indices=indirect_access_indices,
            debug_handle=op_spec.debug_handle,
            # At most one of these is non-empty for a given op (pool / depthwise
            # / forward-conv are mutually exclusive), so the keys never collide.
            **pool_sdsc_fields,
            **conv2d_sdsc_fields,
            **window_sdsc_fields,
        ),
        symbol_mapping,
    )


def compile_op_spec(
    idx: int,
    op_spec: OpSpec,
    symbols: list[int],
    symbol_id_offset: int = 0,
) -> tuple[Any, list[int], list[list[dict]], list[SymbolKind]]:
    sdsc_spec, symbol_mapping = parse_op_spec(op_spec)
    logger.debug("%s", sdsc_spec)
    # Translate tiled_symbols from OpSpec's per-level inductor symbols (innermost-
    # first) to the renamed SDSC symbols via the same mapping used to build
    # sdsc_spec.  generate_sdsc expects outermost-first, so reverse.
    tiled_symbols_per_level = [
        [symbol_mapping[s] for s in level if s in symbol_mapping]
        for level in reversed(op_spec.tiled_symbols)
    ]
    result = generate_sdsc(
        idx,
        sdsc_spec,
        symbols,
        symbol_id_offset,
        tiled_symbols=tiled_symbols_per_level,
    )
    return result
