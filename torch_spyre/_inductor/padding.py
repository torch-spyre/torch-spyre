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

"""IR-level pass to pad the K (reduction) dimension to a stick boundary for
BATCH_MATMUL_OP operations.  Runs in CustomPreSchedulingPasses immediately
after insert_restickify, when every ComputedBuffer has a FixedTiledLayout.

Only y is padded via a new allocation.  x uses its original buffer without
any allocation or copy — the pt hardware unit masks within-stick K reads
beyond the true K size.  reduction_ranges is set to K_padded so the hardware
iterates r_K = 0..K_padded-1.

x's inner_fn computes the flat index into x's original buffer using K_padded
as the M-row stride (c_M*K_padded + c_K), then calls ops.load directly.
SpyreKernel.load() detects a per-op layout override stored in
op_info["x_layout_override"] and substitutes a FixedTiledLayout whose
stride_map uses K_padded instead of K.  This makes the index expression and
stride_map consistent, preventing the spurious r_K//K overflow term in the M
coordinate that compute_coordinates would otherwise emit when r_K's range
(K_padded) exceeds x's original M-row stride (K).  x's original buffer is
untouched; no other consumer sees the padded strides.

For y, the K (row/mb) dimension is not within-stick, so the hardware has no
implicit masking.  y is padded to K_padded = next stick boundary:
  spyre.empty(padded_size)                         — uninitialised allocation
  spyre.constant(0.0)                              — scalar zero, generated on-device (cached)
  aten.expand(constant, pad_size)                  — broadcast to pad-region shape; free
  aten.clone(expand)                               — on-device broadcast copy → fill buffer
  overwrite(fill_buf, empty, [dim], [fill_offset]) — write zeros into pad region
  overwrite(orig,     empty, [dim], [0])           — copy original data at offset 0

fill_offset is original_size[dim] rounded down to the nearest stick boundary.
This ensures the fill overwrite is stick-aligned; any elements between
fill_offset and original_size[dim] that are over-zeroed are restored by the
data overwrite, which always runs after the fill overwrite.

spyre.constant is cached across all matmuls with the same (fill_value, device,
dtype) key so it is lowered at most once per unique fill value and dtype,
regardless of tensor shape or which dimension is padded.

x and y are identified via device_coordinates: x is the input sticked on the
reduction coord, y is the other.  This avoids positional assumptions and handles
square matrices (M==K==N) correctly.

x's effective size is derived from the output ranges ([batch..., M, K]) rather
than from the underlying buffer's shape.  This handles cases where mm_to_bmm_pass
adds a batch dimension to the output ranges while leaving the input buffers 2D
(e.g. torch.einsum("mk,kn->mn")).

reduction_ranges is updated to K_padded so the hardware iterates r=0..K_padded-1.
"""

import sympy
import torch
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    Operation,
    Reduction,
    TensorBox,
)
from torch._inductor.virtualized import V, ops

from .constants import BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import (
    _build_padded_stl,
    concretize_expr,
    device_coordinates,
    host_coordinates,
    lower_pad_sequence,
    replace_computed_buffer_body,
)
from .views import matching_dim
from torch_spyre._C import get_elem_in_stick

logger = get_inductor_logger("padding")


def compute_padding(cur_size: int, dtype: torch.dtype) -> int:
    stick_size = get_elem_in_stick(dtype)
    pad = (stick_size - (cur_size % stick_size)) % stick_size
    return pad


def _patch_env(graph_lowering) -> None:
    """Add view nodes (ReinterpretView) to env from name_to_users."""
    env: dict = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            if not tb.data.origins:
                continue
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)


def _find_arg_fx_node(
    arg_name: str, expected_size: list[int] | None = None
) -> torch.fx.Node:
    """Return the FX node whose lowered TensorBox has the given buffer name.

    Buffer names are unique, but a single buffer can be reached through
    multiple FX nodes that present it at different sizes.  For example,
    mm_to_bmm_pass inserts an unsqueeze/reshape so the matmul inner_fn
    indexes x as 3D [1, M, K] even though the underlying buffer is 2D
    [M, K].  Both FX nodes lower to a TensorBox whose get_name() returns
    the same buffer name, but with different get_size() results.

    ``expected_size`` selects the FX node whose TensorBox size matches the
    dimensionality that the matmul inner_fn actually uses.  This ensures the
    padded clone gets the right shape and _rebuild_matmul's loaders index it
    with the correct number of dimensions.

    Raises RuntimeError if no candidate matches the expected size, or if no
    candidate exists at all.  When ``expected_size`` is None, returns the first
    candidate (the base buffer, with no view applied).
    """
    graph_lowering = V.graph
    _patch_env(graph_lowering)
    candidates = [
        (fx_node, tb)
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    ]
    if not candidates:
        raise RuntimeError(f"no FX node found for buffer {arg_name!r}")
    if expected_size is not None:
        for fx_node, tb in candidates:
            if [int(s) for s in tb.get_size()] == expected_size:
                return fx_node
        raise RuntimeError(
            f"no FX node for buffer {arg_name!r} with size {expected_size}; "
            f"found sizes {[[int(s) for s in tb.get_size()] for _, tb in candidates]}"
        )
    return candidates[0][0]


def _rebuild_matmul(
    op: ComputedBuffer,
    x_name: str,
    x_padded_host_stride: list[int],
    y_padded_buf: Buffer,
    k_padded: int,
    operations: list[Operation],
) -> ComputedBuffer:
    """Rebuild the matmul ComputedBuffer with padded loaders and updated reduction_ranges.

    x is NOT padded — its original buffer is used directly.  The flat index for the
    x load is computed using x_padded_host_stride (K_padded as the M-row stride) so
    that SpyreKernel.load() receives a K_padded-based index.  SpyreKernel.load() reads
    op_info["x_layout_override"] and uses the padded FixedTiledLayout for the
    TensorAccess, ensuring compute_coordinates sees stride_map[M_dim]=K_padded —
    consistent with the index — and produces clean device coordinates.

    reduction_ranges is set to K_padded so the hardware iterates r_K = 0..K_padded-1.
    x's within-stick K tail (uninitialised storage past element K-1) is masked by
    the pt hardware unit.

    y_padded_buf is fully zero-filled in the K pad region.
    """
    reduction = op.data
    assert isinstance(reduction, Reduction)

    y_padded_loader = y_padded_buf.make_loader()
    # y's batch dims: y_ndim - 2 batch dims come first in y_index.
    y_ndim = len(y_padded_buf.get_size())
    y_batch_ndim = y_ndim - 2  # number of batch dims in y (0 for non-batched y)

    def new_inner_fn(
        index,
        reduction_index,
        _x_name=x_name,
        _x_stride=x_padded_host_stride,
        _y_loader=y_padded_loader,
        _y_batch_ndim=y_batch_ndim,
    ):
        # x: all output dims except the last (N), plus the reduction dim.
        # y: first y_batch_ndim batch dims, then reduction dim, then N (index[-1]).
        # Matches the lowering pattern for all mm/bmm variants:
        #   mm (2D×2D):   x_load([i_M, r_K]),       y_loader([r_K, i_N])
        #   bmm (3D×3D):  x_load([i_B, i_M, r_K]),  y_loader([i_B, r_K, i_N])
        #   bmm (4D×4D):  x_load([i_B,i_H,i_M,r_K]),y_loader([i_B,i_H,r_K,i_N])
        #   bmm (3D×2D):  x_load([i_B, i_M, r_K]),  y_loader([r_K, i_N])
        #   einsum→bmm:   x_load([i_B, i_M, r_K]),  y_loader([r_K, i_N])  (y 2D)
        x_index = list(index[:-1]) + list(reduction_index)
        # Compute flat index using K_padded as M-row stride so the index is consistent
        # with the op_info x_layout_override (stride_map[M_dim]=K_padded).
        x_flat = sum(i * s for i, s in zip(x_index, _x_stride))
        y_index = list(index[:_y_batch_ndim]) + list(reduction_index) + [index[-1]]
        return (ops.load(_x_name, x_flat), _y_loader(y_index))

    object.__setattr__(reduction, "inner_fn", new_inner_fn)
    object.__setattr__(reduction, "reduction_ranges", [sympy.Integer(k_padded)])

    return replace_computed_buffer_body(op, reduction, operations)


def insert_padding_ir(operations: list[Operation]) -> None:
    """
    Pad y's K dimension for each BATCH_MATMUL_OP to a stick boundary.

    x is not padded — the pt hardware masks x's within-stick K tail.
    y's K is a row dimension with no hardware masking, so it is explicitly
    zero-filled.  reduction_ranges is updated to K_padded so the hardware
    iterates r_K = 0..K_padded-1.

    Mutates ``operations`` in place.  New y-padding ops are inserted immediately
    before the matmul that consumes them to preserve topological order.

    x and y are identified via device_coordinates: x is the input sticked on
    the reduction coord, y is the other.  This avoids positional assumptions
    and handles square matrices (M==K==N) correctly.

    A fill_cache is shared across all matmuls so that spyre.constant is lowered
    only once per unique (fill_value, device, dtype) combination.
    """
    fill_cache: dict[tuple, torch.fx.Node] = {}
    for op in list(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        reduction = op.data
        if not isinstance(reduction, Reduction):
            continue
        if reduction.reduction_type != BATCH_MATMUL_OP:
            continue

        rw = op.get_read_writes()
        reads = [r for r in rw.reads if hasattr(r, "name")]
        if len(reads) != 2:  # noqa: PLR2004
            continue

        # Identify x and y via device_coordinates.
        # x is the input sticked on the reduction coord (hardware masks within-stick
        # padding for x).  y is the other input; its K host dim is derived from the
        # same reduction coord.  This avoids positional assumptions and handles
        # square matrices (M==K==N) correctly.
        # See propagate_layouts._topk_layouts for the same reduction-coord derivation.
        write_dep = next(iter(rw.writes))
        out_coords = host_coordinates(op.get_layout(), write_dep)

        x_dep = None
        y_dep = None
        y_host_k_dim: int | None = None
        for dep in reads:
            buf = V.graph.get_buffer(dep.name)
            if buf is None:
                continue
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                continue
            h_coords = host_coordinates(layout, dep)
            d_coords = device_coordinates(layout.device_layout, dep)
            stick_expr = d_coords[-1]
            reduction_coord = next(
                (
                    c
                    for c in h_coords
                    if len(c.free_symbols) > 0 and matching_dim(out_coords, c) is None
                ),
                None,
            )
            if reduction_coord is None:
                continue
            stick_dim = matching_dim(h_coords, stick_expr)
            reduction_dim_host = matching_dim(h_coords, reduction_coord)
            if stick_dim == reduction_dim_host:
                x_dep = dep
            else:
                y_dep = dep
                y_host_k_dim = reduction_dim_host

        if x_dep is None or y_dep is None:
            logger.warning(
                "insert_padding_ir: could not identify x/y for %s, skipping",
                op.get_name(),
            )
            continue

        x_name = x_dep.name
        y_name = y_dep.name
        x_buf = V.graph.get_buffer(x_name)
        y_buf = V.graph.get_buffer(y_name)
        if x_buf is None or y_buf is None:
            continue

        # x's effective size for the inner_fn is derived from the output ranges:
        # all output dims except N, plus K.  This correctly handles cases where
        # the inner_fn accesses x through a view with more dims than x_buf
        # (e.g. when mm_to_bmm_pass wraps a 2D mm into a 3D bmm).
        output_ranges = [concretize_expr(s) for s in reduction.ranges]
        k_val = concretize_expr(reduction.reduction_ranges[0])
        x_size = output_ranges[:-1] + [k_val]  # [batch..., M, K]
        dtype = x_buf.get_dtype()
        device = x_buf.get_device()

        pad = compute_padding(k_val, dtype)
        if pad == 0:
            continue

        k_padded = k_val + pad

        logger.debug(
            "insert_padding_ir: padding %s K=%d -> K=%d (pad=%d)",
            op.get_name(),
            k_val,
            k_padded,
            pad,
        )

        # The FX node for the matmul is used as the insertion anchor so padding
        # nodes are placed immediately before the matmul in the FX graph,
        # minimising their live range.
        matmul_fx_node = next(iter(op.origins))

        # --- x: no allocation, no copy — layout override via op_info ---
        # The pt hardware unit masks within-stick K reads beyond the true K size,
        # so x's underlying storage is used as-is.  A FixedTiledLayout with
        # K_padded as the M-row host stride is built and stored on the matmul's
        # op_info under "x_layout_override".  SpyreKernel.load() reads this
        # override so that (a) the index expression uses K_padded as the M-row
        # stride, and (b) compute_coordinates sees stride_map[M_dim]=K_padded.
        # Both are consistent, preventing the spurious r_K//K overflow term that
        # would appear when r_K's range (K_padded) exceeds x's original M-row
        # stride (K).
        x_fx_node = _find_arg_fx_node(x_name, expected_size=x_size)

        x_orig_stl = x_buf.get_layout().device_layout
        x_overlay_host_size = list(x_size)
        x_overlay_host_size[-1] = k_padded  # K is the last dim of x_size
        x_overlay_stl = _build_padded_stl(
            x_fx_node,
            padded_size=x_overlay_host_size,
            orig_stl=x_orig_stl,
            dtype=dtype,
        )

        # Row-major host strides for the padded x shape.
        n = len(x_overlay_host_size)
        x_padded_host_stride = [1] * n
        for i in range(n - 2, -1, -1):
            x_padded_host_stride[i] = (
                x_padded_host_stride[i + 1] * x_overlay_host_size[i + 1]
            )

        x_padded_layout = FixedTiledLayout(
            device,
            dtype,
            [sympy.Integer(s) for s in x_overlay_host_size],
            [sympy.Integer(s) for s in x_padded_host_stride],
            x_overlay_stl,
        )
        op_info = getattr(reduction, "op_info", None)
        if op_info is None:
            op_info = {}
            object.__setattr__(reduction, "op_info", op_info)
        op_info["x_layout_override"] = (x_name, x_padded_layout)

        # --- Pad y: size=[batch..., K_padded, N] ---
        # y's K is a row (mb) dimension, not the within-stick dim, so the hardware
        # does not mask it.  Explicitly zero-fill rows K..K_padded-1 so the
        # reduction over r_K = 0..K_padded-1 reads zero in the pad region.
        y_size = [concretize_expr(s) for s in y_buf.get_size()]
        if y_host_k_dim is None:
            y_k_dim = len(y_size) - 2
        else:
            y_k_dim = y_host_k_dim
        y_padded_size = list(y_size)
        y_padded_size[y_k_dim] = k_padded
        y_fx_node = _find_arg_fx_node(y_name)

        y_orig_stl = y_buf.get_layout().device_layout
        y_padded_buf, y_new_ops = lower_pad_sequence(
            y_fx_node,
            padded_size=y_padded_size,
            device=device,
            dtype=dtype,
            dim=y_k_dim,
            insert_before=matmul_fx_node,
            orig_stl=y_orig_stl,
            fill_cache=fill_cache,
        )

        # --- Relocate new ops before the matmul ---
        # run_node appended them at the end of operations; move before op.
        for new_op in y_new_ops:
            operations.remove(new_op)
        op_idx = operations.index(op)
        for i, new_op in enumerate(y_new_ops):
            operations.insert(op_idx + i, new_op)

        # --- Rebuild matmul inner_fn and reduction_ranges ---
        _rebuild_matmul(
            op,
            x_name,
            x_padded_host_stride,
            y_padded_buf,
            k_padded,
            operations,
        )
