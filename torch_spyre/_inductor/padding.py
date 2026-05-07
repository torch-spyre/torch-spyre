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

"""IR-level pass to pad the reduction dimension K to a stick boundary for
BATCH_MATMUL_OP operations.  Runs in CustomPreSchedulingPasses immediately
after insert_restickify, when every ComputedBuffer has a FixedTiledLayout.

Both x and y are padded along their K dimension.  For each argument:
  spyre.empty(padded_size)                         — uninitialised allocation
  spyre.full([1]*(ndim-1) + [pad_extent], 0.0)     — one stick of zeros (128 bytes DMA)
  aten.expand(full, pad_size)                      — broadcast to pad-region shape; free
  aten.clone(expand)                               — on-device broadcast copy → fill buffer
  overwrite(fill_buf, empty, [dim], [fill_offset]) — write zeros into pad region
  overwrite(orig,     empty, [dim], [0])           — copy original data at offset 0

Only spyre.full crosses the host→device DMA bus; its last dimension equals
pad_extent (the actual pad amount) to ensure the DMA covers exactly one full
stick.  aten.expand broadcasts the one-stick source to the full pad-region
shape and aten.clone materialises it on-device.

fill_offset = original_size[dim] so the pad region starts right after the
original data.  pad_size equals padded_size with pad_size[dim] = pad_extent.

spyre.full is cached across all matmuls with the same (one_stick_size, device,
dtype) key so the DMA is issued at most once per unique pad extent and dtype.

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
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_OP
from .logging_utils import get_inductor_logger
from .pass_utils import concretize_expr, lower_pad_sequence, rebuild_computed_buffer
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
    x_padded_buf: Buffer,
    y_padded_buf: Buffer,
    k_padded: int,
    operations: list[Operation],
) -> ComputedBuffer:
    """Rebuild the matmul ComputedBuffer with padded loaders and updated reduction_ranges.

    Patches the Reduction's inner_fn to load from the padded buffers and updates
    reduction_ranges to k_padded, then delegates the ComputedBuffer reconstruction
    and operations-list swap to rebuild_computed_buffer.

    x_padded_buf must have ndim matching len(output_ranges) (all output dims
    except N plus K_padded) so indices [batch..., M, r_K] are valid.
    y_padded_buf must have the same ndim as the original y buffer.
    """
    reduction = op.data
    assert isinstance(reduction, Reduction)

    x_padded_loader = x_padded_buf.make_loader()
    y_padded_loader = y_padded_buf.make_loader()
    # y's batch dims: y_ndim - 2 batch dims come first in y_index.
    y_ndim = len(y_padded_buf.get_size())
    y_batch_ndim = y_ndim - 2  # number of batch dims in y (0 for non-batched y)

    def new_inner_fn(
        index,
        reduction_index,
        _x_loader=x_padded_loader,
        _y_loader=y_padded_loader,
        _y_batch_ndim=y_batch_ndim,
    ):
        # x: all output dims except the last (N), plus the reduction dim.
        # y: first y_batch_ndim batch dims, then reduction dim, then N (index[-1]).
        # Matches the lowering pattern for all mm/bmm variants:
        #   mm (2D×2D):   x_loader([i_M, r_K]),       y_loader([r_K, i_N])
        #   bmm (3D×3D):  x_loader([i_B, i_M, r_K]),  y_loader([i_B, r_K, i_N])
        #   bmm (4D×4D):  x_loader([i_B,i_H,i_M,r_K]),y_loader([i_B,i_H,r_K,i_N])
        #   bmm (3D×2D):  x_loader([i_B, i_M, r_K]),  y_loader([r_K, i_N])
        #   einsum→bmm:   x_loader([i_B, i_M, r_K]),  y_loader([r_K, i_N])  (y 2D)
        x_index = list(index[:-1]) + list(reduction_index)
        y_index = list(index[:_y_batch_ndim]) + list(reduction_index) + [index[-1]]
        return (_x_loader(x_index), _y_loader(y_index))

    object.__setattr__(reduction, "inner_fn", new_inner_fn)
    object.__setattr__(reduction, "reduction_ranges", [sympy.Integer(k_padded)])

    return rebuild_computed_buffer(op, reduction, operations)


def insert_padding_ir(operations: list[Operation]) -> None:
    """
    Pad the K (reduction) dimension of each BATCH_MATMUL_OP to a stick boundary.

    Mutates ``operations`` in place.  All new buffers are inserted immediately
    before the matmul that consumes them to preserve topological order.

    A fill_cache is shared across all matmuls so that spyre.full is lowered
    only once per unique (one_stick_size, device, dtype) combination.  All pad
    operations with the same N and dtype reuse the same host-allocated fill row.
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

        # Identify x and y: x has K as its last dimension (reduction dim),
        # y has K as its second-to-last dimension (N is y's last dim).
        # For square matrices (M==K==N) both buffers look identical; in that
        # case assign the first dep to x and the second to y.
        k_val = concretize_expr(reduction.reduction_ranges[0])
        x_dep, y_dep = None, None
        for dep in reads:
            buf = V.graph.get_buffer(dep.name)
            if buf is None:
                continue
            buf_last_dim = concretize_expr(buf.get_size()[-1])
            if buf_last_dim == k_val and x_dep is None:
                x_dep = dep
            elif y_dep is None:
                y_dep = dep

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

        # --- Pad x: size=[batch..., M, K_padded] ---
        # Look for the FX node with the expected pre-padded x size so that the
        # padded clone has the right dimensionality for the inner_fn's loaders.
        x_padded_size = x_size[:-1] + [k_padded]
        try:
            x_fx_node = _find_arg_fx_node(x_name, expected_size=x_size)
        except RuntimeError:
            logger.warning(
                "insert_padding_ir: FX node not found for x=%s, skipping", x_name
            )
            continue

        x_padded_buf, x_new_ops = lower_pad_sequence(
            x_fx_node,
            padded_size=x_padded_size,
            device=device,
            dtype=dtype,
            dim=len(x_padded_size) - 1,
            insert_before=matmul_fx_node,
            fill_cache=fill_cache,
        )

        # --- Pad y: size=[batch..., K_padded, N] ---
        # y's K dimension is y's row (mb) dimension.  Padding it to K_padded
        # ensures the matmul reduction does not read uninitialised rows of y
        # when reduction_ranges is extended to k_padded.
        y_size = [concretize_expr(s) for s in y_buf.get_size()]
        y_k_dim = len(y_size) - 2  # K is second-to-last in y for all mm/bmm variants
        y_padded_size = list(y_size)
        y_padded_size[y_k_dim] = k_padded
        try:
            y_fx_node = _find_arg_fx_node(y_name)
        except RuntimeError:
            logger.warning(
                "insert_padding_ir: FX node not found for y=%s, skipping", y_name
            )
            continue

        y_padded_buf, y_new_ops = lower_pad_sequence(
            y_fx_node,
            padded_size=y_padded_size,
            device=device,
            dtype=dtype,
            dim=y_k_dim,
            insert_before=matmul_fx_node,
            fill_cache=fill_cache,
        )

        # --- Relocate new ops before the matmul ---
        # run_node appended them at the end of operations; move before op.
        all_new_ops = x_new_ops + y_new_ops
        for new_op in all_new_ops:
            operations.remove(new_op)
        op_idx = operations.index(op)
        for i, new_op in enumerate(all_new_ops):
            operations.insert(op_idx + i, new_op)

        # --- Rebuild matmul inner_fn and reduction_ranges ---
        _rebuild_matmul(
            op,
            x_padded_buf,
            y_padded_buf,
            k_padded,
            operations,
        )
