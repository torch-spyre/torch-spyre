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

"""IR-level pass to pad y's K (row) dimension to a stick boundary for
BATCH_MATMUL_OP operations.  Runs in CustomPreSchedulingPasses immediately
after insert_restickify, when every ComputedBuffer has a FixedTiledLayout.

Only y is padded; x is left untouched.

For y, the following IR sequence is emitted:
  1. ComputedBuffer - output buffer allocation (FixedLayout)
  2. SpyreConstantFallback - fill constant (FixedLayout)
  3. ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
  4. ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)

y's padded buffer is built at the full K_padded host size by lower_pad_sequence.
reduction_ranges stays at K; the K→K_padded extension happens at SDSC codegen
time: _extend_matmul_k_to_padded in superdsc.py reads K_padded from y's
device_size and widens sdsc_iteration_space[K] to K_padded before
_create_sdsc_tensors runs.

x is left physically untouched.  The hardware masks within-stick elements of x
beyond the true K to zero, so extending the SDSC iteration to K_padded does not
introduce numerical error from x.

Deduplication of identical constants across multiple pad calls happens later
at the IR level via dedup_and_promote_constants.

x and y are identified via identify_matmul_inputs() using the BatchMatmul
generated_dim definition: y is the input whose index contains a symbol
present in the output but absent from x (N).  This handles M==K==N and
M=1 (decode phase) correctly.
"""

import torch
from torch._inductor.graph import GraphLowering
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
    find_reduction_var,
    identify_matmul_inputs,
    host_coordinates,
    lower_pad_sequence,
    replace_computed_buffer_body,
)
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


def _find_arg_fx_node(arg_name: str) -> torch.fx.Node:
    """Return the FX node whose lowered TensorBox has the given buffer name.

    Buffer names are unique, but a single buffer can be reached through
    multiple FX nodes that present it at different sizes.  For example,
    mm_to_bmm_pass inserts an unsqueeze/reshape so the matmul inner_fn
    indexes x as 3D [1, M, K] even though the underlying buffer is 2D
    [M, K].  Both FX nodes lower to a TensorBox whose get_name() returns
    the same buffer name, but with different get_size() results.

    Returns the first candidate (the base buffer, with no view applied).
    Raises RuntimeError if no candidate exists.
    """
    graph_lowering = V.graph
    _patch_env(graph_lowering)
    candidates = [
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    ]
    if not candidates:
        raise RuntimeError(f"no FX node found for buffer {arg_name!r}")
    return candidates[0]


def _rebuild_matmul(
    op: ComputedBuffer,
    y_padded_buf: Buffer,
    operations: list[Operation],
) -> ComputedBuffer:
    """Rebuild the matmul ComputedBuffer so y's loader reads from the padded buffer.

    Preserves the original inner_fn's x loading unchanged; only replaces y's
    loader with one that reads from the padded buffer.  reduction_ranges stays
    at K; the K→K_padded extension happens at SDSC codegen time via
    _extend_matmul_k_to_padded in superdsc.py.
    """
    reduction = op.data
    assert isinstance(reduction, Reduction)

    orig_inner_fn = reduction.inner_fn
    y_padded_loader = y_padded_buf.make_loader()
    y_ndim = len(y_padded_buf.get_size())
    y_batch_ndim = y_ndim - 2

    def new_inner_fn(
        index,
        reduction_index,
        _orig_inner_fn=orig_inner_fn,
        _y_loader=y_padded_loader,
        _y_batch_ndim=y_batch_ndim,
    ):
        # x_val comes from the original inner_fn; discard its y and replace below.
        x_val, _ = _orig_inner_fn(index, reduction_index)
        y_index = list(index[:_y_batch_ndim]) + list(reduction_index) + [index[-1]]
        y_val = _y_loader(y_index)
        return (x_val, y_val)

    object.__setattr__(reduction, "inner_fn", new_inner_fn)
    # reduction_ranges stays at K; no extension here.

    return replace_computed_buffer_body(op, reduction, operations)


def insert_bmm_padding(graph: GraphLowering) -> None:
    """
    Pad y's K (row) dimension for each BATCH_MATMUL_OP to a stick boundary.

    Mutates ``operations`` in place.  New buffers for y are inserted immediately
    before the matmul that consumes them to preserve topological order.

    x is left entirely untouched.  y's padded buffer is built at K_padded host
    size by lower_pad_sequence; reduction_ranges stays at K so the IR iteration
    space is unchanged.  The K→K_padded widening happens at SDSC codegen time.

    x and y are identified via identify_matmul_inputs() using the BatchMatmul
    generated_dim definition: y is the input whose index contains a symbol
    present in the output but absent from x (N).  This handles M==K==N and
    M=1 (decode phase) correctly.

    Deduplication of identical constants across multiple pad calls happens later
    at the IR level via dedup_and_promote_constants.
    """
    operations = graph.operations
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

        # Skip aligned-K matmuls early before any x/y identification.
        # Aligned-K matmuls need no padding regardless of input layout, and
        # skipping here avoids a spurious warning for e.g. decode-phase SDPA
        # attention where constant-folded dimensions cause identify_matmul_inputs
        # to fail.
        k_val = concretize_expr(reduction.reduction_ranges[0])
        first_buf = next(
            (graph.get_buffer(d.name) for d in reads if graph.get_buffer(d.name)),
            None,
        )
        assert first_buf is not None, (
            f"insert_bmm_padding: no input buffer found for matmul {op.get_name()}"
        )
        dtype = first_buf.get_dtype()
        if compute_padding(k_val, dtype) == 0:
            continue

        write_dep = next(iter(rw.writes))
        x_dep, y_dep = identify_matmul_inputs(reads, write_dep)
        if x_dep is None or y_dep is None:
            logger.warning(
                "insert_bmm_padding: could not identify x/y for %s, skipping",
                op.get_name(),
            )
            continue

        reduction_var = find_reduction_var(x_dep, write_dep)

        # y's K host dim: the dim whose host coordinate contains reduction_var.
        y_buf_tmp = graph.get_buffer(y_dep.name)
        y_host_k_dim: int | None = None
        if y_buf_tmp is not None and isinstance(
            y_buf_tmp.get_layout(), FixedTiledLayout
        ):
            y_h_coords = host_coordinates(y_buf_tmp.get_layout(), y_dep, None)
            y_host_k_dim = next(
                (
                    i
                    for i, c in enumerate(y_h_coords)
                    if reduction_var in c.free_symbols
                ),
                None,
            )

        x_name = x_dep.name
        y_name = y_dep.name
        x_buf = graph.get_buffer(x_name)
        y_buf = graph.get_buffer(y_name)
        if x_buf is None or y_buf is None:
            continue

        device = x_buf.get_device()
        pad = compute_padding(k_val, dtype)

        k_padded = k_val + pad

        logger.debug(
            "insert_bmm_padding: padding %s K=%d -> K=%d (pad=%d)",
            op.get_name(),
            k_val,
            k_padded,
            pad,
        )

        # The FX node for the matmul is used as the insertion anchor so padding
        # nodes are placed immediately before the matmul in the FX graph,
        # minimising their live range.
        matmul_fx_node = next(iter(op.origins))

        # --- Pad y only ---
        # y's K dimension is y's row (mb) dimension.  Padding it to K_padded
        # ensures rows K..K_padded-1 of y are zero-filled so the hardware
        # accumulates no contribution from those rows.
        # lower_pad_sequence builds the padded buffer at K_padded host size;
        # reduction_ranges is NOT changed.  superdsc._extend_matmul_k_to_padded
        # widens sdsc_iteration_space[K] to K_padded at SDSC codegen time,
        # reading K_padded from y's device_layout.device_size.
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
        for new_op in y_new_ops:
            operations.remove(new_op)
        op_idx = operations.index(op)
        for i, new_op in enumerate(y_new_ops):
            operations.insert(op_idx + i, new_op)

        # --- Rebuild matmul inner_fn to load y from the padded buffer ---
        # x is left entirely untouched: the original inner_fn's x loader is
        # preserved as-is.  Only y's loader is replaced with the padded buffer.
        _rebuild_matmul(op, y_padded_buf, operations)
