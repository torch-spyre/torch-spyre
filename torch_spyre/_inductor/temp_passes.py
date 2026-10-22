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

# This file contains inductor passes that are only needed as temp fixes

import contextlib
from math import prod

import torch
from torch._guards import detect_fake_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from .logging_utils import get_inductor_logger
from .pass_utils import copy_fx_custom_meta

aten = torch.ops.aten

logger = get_inductor_logger("work_division")

_RESHAPE_OPS = (
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
)

mm_to_bmm_pass = PatternMatcherPass(pass_name="unflatten_mm_to_bmm")
bmm_unflatten_pass = PatternMatcherPass(pass_name="unflatten_bmm_batch_dims")
broadcast_gather_pass = PatternMatcherPass(pass_name="broadcast_gather_to_row_gather")


def _node_shape(node: torch.fx.Node) -> tuple | None:
    """Return an FX node's fake/meta shape, if one is available."""
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    shape = getattr(val, "shape", None)
    return tuple(shape) if shape is not None else None


def _shapes_statically_equal(lhs, rhs) -> bool:
    """Whether two shape sequences are provably equal without adding guards."""
    return len(lhs) == len(rhs) and statically_known_true(
        sym_eq(tuple(lhs), tuple(rhs))
    )


@register_graph_pattern(
    CallFunction(aten.mm.default, Arg(), Arg()),
    pass_dict=mm_to_bmm_pass,
)
def _unflatten_mm_to_bmm(
    match: Match, mat1_node: torch.fx.Node, mat2_node: torch.fx.Node
) -> None:
    """
    Convert view(3D→2D) → mm(2D, 2D) → view(2D→3D) into bmm(3D, unsqueeze(2D)).

    When torch.matmul is called with a batched input and a 2D weight, the
    decomposition flattens the batch dimensions:
      1. view(input, [B*M, K])
      2. mm(flattened, weight) -> [B*M, N]
      3. view(mm_result, [B, M, N])

    The Spyre backend handles bmm better. This pass converts the pattern
    into a semantically correct bmm by unsqueezeing and expanding the 2D
    weight to match the batch dimension of the input.
    """
    node = match.nodes[-1]
    graph = node.graph
    lhs, rhs = mat1_node, mat2_node

    # LHS must be a reshape that flattens a higher-dim tensor to 2D
    if not (
        isinstance(lhs, torch.fx.Node)
        and lhs.op == "call_function"
        and lhs.target in _RESHAPE_OPS
    ):
        return
    lhs_input = lhs.args[0]
    if not (isinstance(lhs_input, torch.fx.Node) and "val" in lhs_input.meta):
        return
    lhs_orig_shape = list(lhs_input.meta["val"].shape)

    # RHS must be a plain 2D tensor (not a reshaped one)
    if not (isinstance(rhs, torch.fx.Node) and "val" in rhs.meta):
        return
    rhs_shape = list(rhs.meta["val"].shape)
    if len(rhs_shape) != 2:
        return

    # The mm result must feed into exactly one view that restores batch dims
    mm_users = list(node.users.keys())
    if len(mm_users) != 1:
        return
    output_view = mm_users[0]
    if not (output_view.op == "call_function" and output_view.target in _RESHAPE_OPS):
        return
    output_shape = output_view.args[1]
    if not isinstance(output_shape, (list, tuple)):
        return
    if len(output_shape) <= 2:
        return

    # Verify the output shape's batch dims match the original input's
    if list(output_shape[:-1]) != lhs_orig_shape[:-1]:
        return

    # Build the bmm: bmm(lhs_orig, unsqueeze(rhs, 0).expand(B, K, N))
    batch_dims = lhs_orig_shape[:-2]  # e.g. [2] from [2, 64, 128]
    K, N = rhs_shape

    with graph.inserting_before(node):
        # unsqueeze weight to 3D+: [K, N] → [1, ..., 1, K, N]
        unsqueezed = rhs
        rhs_dtype = rhs.meta["val"].dtype
        unsqueezed_shape = list(rhs_shape)
        for i in range(len(batch_dims)):
            unsqueezed = graph.call_function(
                aten.unsqueeze.default,
                args=(unsqueezed, 0),
            )
            unsqueezed_shape = [1] + unsqueezed_shape
            unsqueezed.meta["val"] = torch.empty(
                unsqueezed_shape, dtype=rhs_dtype, device="meta"
            )

        # expand to match batch dims: [1, ..., 1, K, N] → [B, ..., K, N]
        expanded_shape = batch_dims + [K, N]
        expanded = graph.call_function(
            aten.expand.default,
            args=(unsqueezed, expanded_shape),
        )
        expanded.meta["val"] = torch.empty(
            expanded_shape, dtype=rhs_dtype, device="meta"
        )

        # Use spyre.batched_matmul for >3D to avoid FakeTensorUpdater crash
        # (aten.bmm requires exactly 3D inputs)
        target = (
            torch.ops.spyre.batched_matmul.default
            if len(output_shape) > 3
            else aten.bmm.default
        )
        bmm_node = graph.call_function(
            target,
            args=(lhs_input, expanded),
        )
        bmm_node.meta["val"] = torch.empty(output_shape, dtype=rhs_dtype, device="meta")
        copy_fx_custom_meta(node, bmm_node)

    # Replace all uses of mm and output view with the bmm
    node.replace_all_uses_with(bmm_node)
    output_view.replace_all_uses_with(bmm_node)

    # Clean up dead nodes
    graph.erase_node(output_view)
    graph.erase_node(node)
    if not lhs.users:
        graph.erase_node(lhs)


def _is_row_broadcast_gather(x, dim: int, index) -> bool:
    """Whether ``gather(x, dim, index)`` selects whole trailing slabs of ``x``.

    True when ``index`` matches ``x`` on every dim but ``dim`` and is broadcast
    (stride 0) along the dims after ``dim``, e.g. ``pos.view(-1, 1, 1).expand(-1,
    1, V)``. Every trailing entry then repeats one index, so each output slab is
    one whole slab of ``x.reshape(-1, *x.shape[dim+1:])``. Requires static
    shapes and at least one real broadcast dim; anything else keeps the generic
    gather lowering.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(index, torch.Tensor):
        return False
    ndim = x.dim()
    if ndim < 2 or index.dim() != ndim or index.dtype not in (torch.int32, torch.int64):
        return False
    dim = dim % ndim
    if dim == ndim - 1:
        return False
    if not all(isinstance(s, int) for s in (*x.shape, *index.shape, *index.stride())):
        return False
    if any(index.shape[k] != x.shape[k] for k in range(ndim) if k != dim):
        return False
    trailing = range(dim + 1, ndim)
    if any(index.shape[k] != 1 and index.stride(k) != 0 for k in trailing):
        return False
    return any(index.shape[k] > 1 for k in trailing)


def _emit_row_gather(call, x, index, dim, x_shape, n, index_dtype, index_device):
    """Emit ``gather(x, dim, index)`` as a plain row gather, one aten op at a time.

    With P = prod(x_shape[:dim]) and T = x_shape[dim], output slab (p, j) is slab
    ``p * T + index[p, j]`` of ``x.reshape(P * T, *x_shape[dim+1:])``. Only the
    dims up to ``dim`` are merged; the trailing dims, which hold the stick, keep
    their shape, since merging into the stick dim is not a view on Spyre.

    ``call(op, *args, **kwargs)`` applies one op: the pass passes a builder that
    adds a graph node, and the unit test passes one that runs the op on tensors,
    so both exercise this same sequence.
    """
    prefix, trailing = list(x_shape[:dim]), list(x_shape[dim + 1 :])
    T, P = x_shape[dim], prod(prefix)
    # The index repeats along the trailing dims, so keep one entry per slab.
    idx = index
    for _ in trailing:
        idx = call(aten.select.int, idx, dim + 1, 0)
    rows = call(aten.reshape.default, idx, [P, n])
    if P > 1:
        # Row offset of each prefix position. A stepped arange plus an int add
        # keeps the index exact and avoids an int mul on the device.
        base = call(
            aten.arange.start_step, 0, P * T, T, dtype=index_dtype, device=index_device
        )
        rows = call(aten.add.Tensor, rows, call(aten.unsqueeze.default, base, 1))
    rows = call(aten.reshape.default, rows, [P * n])
    table = call(aten.reshape.default, x, [P * T, *trailing])
    gathered = call(aten.index.Tensor, table, [rows])
    return call(aten.reshape.default, gathered, [*prefix, n, *trailing])


@register_graph_pattern(
    CallFunction(aten.gather.default, Arg(), Arg(), Arg()),
    pass_dict=broadcast_gather_pass,
)
def _broadcast_gather_to_row_gather(
    match: Match, x_node: torch.fx.Node, dim: int, index_node: torch.fx.Node
) -> None:
    """Rewrite a gather with a trailing-broadcast index into a row gather (#4840).

    ``logits.gather(1, pos.view(-1, 1, 1).expand(-1, 1, V))`` reads
    ``x[b, IndirectAccess(pos[b]), v]``: the batch dim b drives the index load
    and also addresses the value directly. The indirect-access SDSC encoding has
    no slot for a dim in both roles, so the wrong rows come back without an
    error. Computing the full row index first leaves an ``x2[rows]`` gather
    whose index and value dims are disjoint, the form the embedding and
    advanced-indexing gathers already use.
    """
    if not isinstance(dim, int):
        return
    x_val = x_node.meta.get("val") if isinstance(x_node, torch.fx.Node) else None
    index_val = (
        index_node.meta.get("val") if isinstance(index_node, torch.fx.Node) else None
    )
    if not _is_row_broadcast_gather(x_val, dim, index_val):
        return
    dim = dim % x_val.dim()

    node = match.nodes[-1]
    graph = node.graph
    fake_mode = detect_fake_mode([x_val, index_val]) or contextlib.nullcontext()

    def add_node(op, *args, **kwargs):
        new_node = graph.call_function(op, args=args, kwargs=kwargs)
        arg_vals, kwarg_vals = torch.fx.node.map_arg(
            (args, kwargs), lambda a: a.meta["val"]
        )
        # Run the op on the inputs' fake values so the node's meta carries the
        # real shape, stride, dtype and device.
        with fake_mode:
            new_node.meta["val"] = op(*arg_vals, **kwarg_vals)
        copy_fx_custom_meta(node, new_node)
        return new_node

    with graph.inserting_before(node):
        out = _emit_row_gather(
            add_node,
            x_node,
            index_node,
            dim,
            tuple(x_val.shape),
            index_val.shape[dim],
            index_val.dtype,
            index_val.device,
        )
    node.replace_all_uses_with(out)
    graph.erase_node(node)


def _is_batch_collapsing_reshape(node: torch.fx.Node) -> bool:
    """Check if a node is a reshape that collapses batch dims into a single dim."""
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op != "call_function":
        return False
    if node.target not in _RESHAPE_OPS:
        return False
    # The reshape output should be 3D (batch_product, M, K)
    output_shape = node.args[1]
    if not isinstance(output_shape, (list, tuple)) or len(output_shape) != 3:
        return False
    # The input should be higher dimensional
    input_node = node.args[0]
    if isinstance(input_node, torch.fx.Node) and "val" in input_node.meta:
        input_ndim = input_node.meta["val"].dim()
        return input_ndim > 3
    return False


@register_graph_pattern(
    CallFunction(aten.bmm.default, Arg(), Arg()),
    pass_dict=bmm_unflatten_pass,
)
def _unflatten_bmm_batch_dims(
    match: Match, mat1_node: torch.fx.Node, mat2_node: torch.fx.Node
) -> None:
    """
    Undo the matmul decomposition's flattening of batch dimensions into 3D bmm.

    The matmul decomposition in torch/_decomp/decompositions.py converts N-D
    matmuls (e.g. 4D SDPA attention) into 3D by:
      1. expand(input, [B, H, M, K]) -> reshape([B*H, M, K])
      2. expand(input, [B, H, K, N]) -> reshape([B*H, K, N])
      3. bmm(reshaped1, reshaped2) -> [B*H, M, N]
      4. view(bmm_result, [B, H, M, N]) -> back to original dims

    This pass removes the reshape/view wrapper so the bmm operates on the
    original higher-dimensional tensors, which the Spyre backend can handle
    natively via its 4D batch matmul lowering.

    This is needed as the flattened views are not supported by the current
    backend. When KTIR is implemented this pass can be dropped.
    """
    node = match.nodes[-1]
    graph = node.graph
    lhs_reshape, rhs_reshape = mat1_node, mat2_node

    # Both inputs must be reshape/view that collapse batch dims to 3D
    if not _is_batch_collapsing_reshape(lhs_reshape):
        return
    if not _is_batch_collapsing_reshape(rhs_reshape):
        return

    # The bmm result must feed into exactly one view that restores the batch dims
    bmm_users = list(node.users.keys())
    if len(bmm_users) != 1:
        return
    output_view = bmm_users[0]
    if not (output_view.op == "call_function" and output_view.target in _RESHAPE_OPS):
        return

    # Get the original (pre-reshape) tensors
    lhs_orig = lhs_reshape.args[0]  # the expand or original tensor
    rhs_orig = rhs_reshape.args[0]

    # Prove the entire reshape sandwich before removing it.  Equal element
    # counts are not enough: a reshape can collapse different logical batch
    # prefixes, interchange M/K/N, or restore the result in a different order.
    # Reusing those operands directly would then make lower_bmm index one
    # producer with another operand's matrix domain.
    lhs_orig_shape = _node_shape(lhs_orig)
    rhs_orig_shape = _node_shape(rhs_orig)
    lhs_flat_shape = _node_shape(lhs_reshape)
    rhs_flat_shape = _node_shape(rhs_reshape)
    bmm_shape = _node_shape(node)
    output_shape = _node_shape(output_view)
    if (
        lhs_orig_shape is None
        or rhs_orig_shape is None
        or lhs_flat_shape is None
        or rhs_flat_shape is None
        or bmm_shape is None
        or output_shape is None
    ):
        return

    # lower_bmm currently has a native contract for exactly two batch axes.
    # Leave other ranks as the original, semantically valid flattened bmm.
    if len(lhs_orig_shape) != 4 or len(rhs_orig_shape) != 4:
        return

    lhs_batch = lhs_orig_shape[:-2]
    rhs_batch = rhs_orig_shape[:-2]
    lhs_rows, lhs_contraction = lhs_orig_shape[-2:]
    rhs_contraction, rhs_columns = rhs_orig_shape[-2:]
    flat_batch = prod(lhs_batch)

    if not _shapes_statically_equal(lhs_batch, rhs_batch):
        return
    if not statically_known_true(sym_eq(lhs_contraction, rhs_contraction)):
        return
    if not _shapes_statically_equal(
        lhs_flat_shape, (flat_batch, lhs_rows, lhs_contraction)
    ):
        return
    if not _shapes_statically_equal(
        rhs_flat_shape, (flat_batch, rhs_contraction, rhs_columns)
    ):
        return
    if not _shapes_statically_equal(bmm_shape, (flat_batch, lhs_rows, rhs_columns)):
        return
    if not _shapes_statically_equal(output_shape, (*lhs_batch, lhs_rows, rhs_columns)):
        return

    # Replace the 3D bmm with a spyre.batched_matmul that accepts N-D inputs.
    # Using aten.bmm.default with >3D args would crash FakeTensorUpdater.
    with graph.inserting_before(node):
        matmul_node = graph.call_function(
            torch.ops.spyre.batched_matmul.default,
            args=(lhs_orig, rhs_orig),
        )
        matmul_node.meta["val"] = output_view.meta["val"]
        copy_fx_custom_meta(node, matmul_node)

    # Replace all uses of the output view with the new matmul
    output_view.replace_all_uses_with(matmul_node)
    node.replace_all_uses_with(matmul_node)
    graph.erase_node(output_view)
    graph.erase_node(node)

    # Clean up dead reshape nodes
    for reshape_node in (lhs_reshape, rhs_reshape):
        if not reshape_node.users:
            expand_node = reshape_node.args[0]
            graph.erase_node(reshape_node)
            # Also remove the expand if it's now unused
            if (
                isinstance(expand_node, torch.fx.Node)
                and expand_node.op == "call_function"
                and expand_node.target == aten.expand.default
                and not expand_node.users
            ):
                graph.erase_node(expand_node)


def decompose_addmm(graph: torch.fx.Graph) -> None:
    """Decompose ``aten.addmm.default`` into ``add(scaled_input, alpha*mm)``.

    Inductor's post-grad pattern matcher re-fuses ``add(input, mm(a, b))`` back
    into ``aten.addmm.default`` after AOTAutograd, defeating the upstream
    decomposition. With no Spyre lowering for ``addmm``, the op then falls
    back to ``extern_kernels.addmm`` which produces an ``ExternKernelOut``
    without a ``FixedTiledLayout`` and breaks subsequent Spyre passes.

    This pass undoes the re-fusion at FX time so the resulting ``mm``,
    ``mul`` and ``add`` nodes flow through the existing Spyre lowerings.
    Any ``alpha`` / ``beta`` scalars become ``aten.mul.Scalar`` nodes whose
    scalar constants are later materialized into ``spyre.constant`` tensors by
    the LoopLevel IR multi-ops pass (``split_multi_ops``).
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.addmm.default:
            continue
        input_node, mat1, mat2 = node.args[0], node.args[1], node.args[2]
        beta = node.kwargs.get("beta", 1)
        alpha = node.kwargs.get("alpha", 1)

        out_meta = node.meta.get("val", None)

        with graph.inserting_before(node):
            mm_node = graph.call_function(aten.mm.default, args=(mat1, mat2))
            if out_meta is not None:
                mm_node.meta["val"] = torch.empty_like(out_meta, device="meta")
            copy_fx_custom_meta(node, mm_node)

            scaled_mm = mm_node
            if alpha != 1:
                scaled_mm = graph.call_function(aten.mul.Scalar, args=(mm_node, alpha))
                if out_meta is not None:
                    scaled_mm.meta["val"] = torch.empty_like(out_meta, device="meta")
                copy_fx_custom_meta(node, scaled_mm)

            if beta == 0:
                replacement = scaled_mm
            else:
                scaled_input = input_node
                if beta != 1:
                    scaled_input = graph.call_function(
                        aten.mul.Scalar, args=(input_node, beta)
                    )
                    in_meta = (
                        input_node.meta.get("val", None)
                        if isinstance(input_node, torch.fx.Node)
                        else None
                    )
                    if in_meta is not None:
                        scaled_input.meta["val"] = torch.empty_like(
                            in_meta, device="meta"
                        )
                    copy_fx_custom_meta(node, scaled_input)

                replacement = graph.call_function(
                    aten.add.Tensor, args=(scaled_input, scaled_mm)
                )
                if out_meta is not None:
                    replacement.meta["val"] = torch.empty_like(out_meta, device="meta")
                copy_fx_custom_meta(node, replacement)

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)

    graph.lint()
