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

from math import log, prod

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from .dtype_ops import DtypeOpTable
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


# Marks both the exp a guard was built around and the exp that measures the floor, so
# a second run over the same graph is a no-op rather than a second guard.
_EXP_UNDERFLOW_DONE = "_spyre_exp_underflow_done"


def _exp_underflow_threshold(dtype: torch.dtype) -> float | None:
    """Largest input whose ``exp()`` rounds to zero in ``dtype``, else None.

    ``exp(x)`` rounds to zero once it drops below half the smallest subnormal, which
    is ``finfo.tiny * finfo.eps`` in every IEEE binary format: -17.33 at fp16,
    -92.88 at bf16.
    """
    if not dtype.is_floating_point:
        return None
    finfo = torch.finfo(dtype)
    return log(finfo.tiny * finfo.eps / 2.0)


def guard_exp_underflow(graph: torch.fx.Graph) -> None:
    """Give ``aten.exp`` the underflow to zero that the device op does not perform.

    The device's ``exp`` saturates instead of underflowing. Every fp16 input below
    about -17 returns the smallest subnormal, 2**-24, and so does ``-inf``, where
    IEEE gives an exact zero. Two callers read that as a zero and are quietly wrong
    without this:

      * a masked softmax -- attention adds a large negative sentinel to the scores
        of the positions it must ignore, and at weight 2**-24 rather than 0 every
        ignored position still contributes its value through ``matmul(probs, v)``.
        Paged attention gathers whole pages, so the slots past a sequence's end
        reach the output, and because vLLM hands the same block to later requests,
        one forward pass comes to depend on the requests that held those pages
        before it: a fresh process answers the same greedy request differently
        until block reuse stops changing those bytes;
      * ``_POINTWISE_PADDING_MASK_VALUE`` (codegen/superdsc.py), which seeds an
        exp's padding lanes with -1e4 to make them contraction-neutral.

    Rewritten as ``exp(x) - exp(clamp_max(x, threshold))``. Below the threshold both
    terms are whatever value the device saturates at, so they cancel to an exact
    zero **without this pass having to know that value** -- the second exp measures
    the floor on the device, in device precision. Above it the second term is still
    that floor, which is the smallest representable magnitude and therefore far
    below the first term's ulp, so the result is unchanged.

    Not knowing the floor is the point. The obvious cheaper shape, subtracting the
    host's idea of the smallest subnormal and clamping at zero with ``relu``,
    compiles everywhere and **does not fix the bug**: a host read-back of the probs
    then shows exact zeros while the leak persists at half amplitude, because device
    fp16 is DL16 (1-6-9) and the subnormal range does not survive the D2H rounding,
    so the host constant is not the device's floor. That also rules out judging any
    fix here by reading an intermediate back.

    Selecting the zero instead -- ``where(x < threshold, 0, exp(x))`` -- is exact and
    was the first shape tried. It does not lower: the select gives the softmax
    numerator a third operand and a full-extent constant, and inside the SDPA
    decomposition the stick-layout solver then has no mechanism to reconcile them
    with the layout the matmul needs, so 14 of torch-spyre's own attention tests
    fail to compile with ``no mechanism to resolve stick incompatibility``. A
    multiplicative mask and ``masked_fill`` fail the same check on more tests.
    Staying with unary ops plus one same-shape subtract is what makes this shape
    lower: every operand carries exactly the layout the ``exp`` already had.

    Rewriting at FX rather than in a lowering is also forced: ``split_multi_ops``
    materializes each op of a multi-op pointwise body by finding the FX node that
    produced it, so ops conjured inside an ``inner_fn`` have nothing to resolve
    against and codegen fails with ``No FX node for buf<n>``.

    fp16 family only. The device's fp32 exp saturates at 2**-17 rather than near
    fp32's smallest subnormal, which is a much larger error of its own, and fp32 has
    no masked-softmax caller.

    Two costs. A second transcendental on the same tensor. And because the
    subtracted term is the device's floor rather than a mathematical threshold, this
    zeroes every input for which the device has already saturated -- measured,
    x <= -16, where IEEE would zero only x < -17.33 -- so inputs in that gap return
    0 instead of 2**-24 or 2**-23, the two smallest representable magnitudes, in a
    range where the device op is already returning about half the true value.
        """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.exp.default:
            continue
        if node.meta.get(_EXP_UNDERFLOW_DONE):
            continue
        out_meta = node.meta.get("val", None)
        if out_meta is None or out_meta.dtype not in DtypeOpTable.fp16_types():
            continue
        threshold = _exp_underflow_threshold(out_meta.dtype)
        if threshold is None:
            continue

        with graph.inserting_before(node):
            clamped = graph.call_function(
                aten.clamp_max.default, args=(node.args[0], threshold)
            )
            clamped.meta["val"] = torch.empty_like(out_meta, device="meta")
            copy_fx_custom_meta(node, clamped)
            floor = graph.call_function(aten.exp.default, args=(clamped,))
            floor.meta["val"] = torch.empty_like(out_meta, device="meta")
            copy_fx_custom_meta(node, floor)
            # This exp *is* the floor measurement; rewriting it would not terminate.
            floor.meta[_EXP_UNDERFLOW_DONE] = True

        with graph.inserting_after(node):
            guarded = graph.call_function(aten.sub.Tensor, args=(node, floor))
            guarded.meta["val"] = torch.empty_like(out_meta, device="meta")
            copy_fx_custom_meta(node, guarded)

        # The exp keeps feeding the subtract it now sits behind; every other reader
        # takes the subtract. Without the callback this would rewrite the subtract's
        # own operand into a reference to itself.
        node.replace_all_uses_with(
            guarded, delete_user_cb=lambda user: user is not guarded
        )
        node.meta[_EXP_UNDERFLOW_DONE] = True

    graph.lint()


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
