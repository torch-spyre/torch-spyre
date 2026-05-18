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

Both x and y are padded along their K dimension to K_padded = next stick
boundary.  For each argument:
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
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import (
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
    x_padded_buf: Buffer,
    y_padded_buf: Buffer,
    k_padded: int,
    operations: list[Operation],
) -> ComputedBuffer:
    """Rebuild the matmul ComputedBuffer with padded loaders and updated reduction_ranges.

    Patches the Reduction's inner_fn to load from the padded buffers and updates
    reduction_ranges to k_padded, then delegates the ComputedBuffer reconstruction
    and operations-list swap to replace_computed_buffer_body.

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

    return replace_computed_buffer_body(op, reduction, operations)


def insert_padding_ir(operations: list[Operation]) -> None:
    """
    Pad the K (reduction) dimension of each BATCH_MATMUL_OP to a stick boundary.

    Mutates ``operations`` in place.  All new buffers are inserted immediately
    before the matmul that consumes them to preserve topological order.

    x and y are identified via device_coordinates: x is the input sticked on
    the reduction coord, y is the other.  This avoids positional assumptions
    and handles square matrices (M==K==N) correctly.

    Deduplication of identical constants across multiple pad calls happens later
    at the IR level via dedup_and_promote_constants.
    """
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

        # --- Pad x: size=[batch..., M, K_padded] ---
        # Look for the FX node with the expected pre-padded x size so that the
        # padded clone has the right dimensionality for the inner_fn's loaders.
        x_padded_size = x_size[:-1] + [k_padded]
        x_fx_node = _find_arg_fx_node(x_name, expected_size=x_size)

        x_orig_stl = x_buf.get_layout().device_layout
        x_padded_buf, x_new_ops = lower_pad_sequence(
            x_fx_node,
            padded_size=x_padded_size,
            device=device,
            dtype=dtype,
            dim=len(x_padded_size) - 1,
            insert_before=matmul_fx_node,
            orig_stl=x_orig_stl,
        )

        # --- Pad y: size=[batch..., K_padded, N] ---
        # y's K dimension is y's row (mb) dimension.  Padding it to K_padded
        # ensures the matmul reduction does not read uninitialised rows of y
        # when reduction_ranges is extended to k_padded.
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
