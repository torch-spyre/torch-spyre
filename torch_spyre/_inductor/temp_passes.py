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

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from .logging_utils import get_inductor_logger

aten = torch.ops.aten

logger = get_inductor_logger("core_division")

_RESHAPE_OPS = (
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
)

mm_to_bmm_pass = PatternMatcherPass(pass_name="unflatten_mm_to_bmm")
bmm_unflatten_pass = PatternMatcherPass(pass_name="unflatten_bmm_batch_dims")


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

        # Replace mm with bmm
        bmm_node = graph.call_function(
            aten.bmm.default,
            args=(lhs_input, expanded),
        )
        bmm_node.meta["val"] = torch.empty(output_shape, dtype=rhs_dtype, device="meta")

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

    output_shape = output_view.args[1]
    if len(output_shape) <= 3:
        return

    # Get the original (pre-reshape) tensors
    lhs_orig = lhs_reshape.args[0]  # the expand or original tensor
    rhs_orig = rhs_reshape.args[0]

    # Update bmm to take the higher-dimensional inputs directly
    node.args = (lhs_orig, rhs_orig)

    # Update bmm output shape metadata
    node.meta["val"] = output_view.meta["val"]

    # Replace all uses of the output view with the bmm itself
    output_view.replace_all_uses_with(node)
    graph.erase_node(output_view)

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


def convert_constant_with_graph_node(graph: torch.fx.Graph) -> None:
    """
    Replace constant arguments to any operation with spyre.constant node.
    Scalar constants are converted to size=1 tensor and passed to the corresponding
    operations which was consuming the scalar value at lowering.
    """

    ops_support_list = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.true_divide.Tensor,
        torch.ops.aten.div.Tensor,
    ]

    # Created node cache for scalar values, and reuse the node when
    # the scalar found again.
    const_node_map: dict[int | float, torch.fx.node.Node] = {}

    for node in graph.nodes:
        if node.target not in ops_support_list:
            continue
        scalar_indexes = []
        for i in range(len(node.args)):
            in_arg = node.args[i]
            if not isinstance(in_arg, torch.fx.node.Node):
                if isinstance(in_arg, (int, float)):
                    scalar_indexes.append(i)
                else:
                    logger.warning(f"Warning: unhandled node type {type(in_arg)}")

        if len(scalar_indexes) > 0:
            for idx in scalar_indexes:
                scalar_val = node.args[idx]

                with graph.inserting_before(node):
                    if scalar_val in const_node_map:
                        const_node = const_node_map[scalar_val]
                    else:
                        # Currently the dtype of the scalar tensor is set as same as the output dtype.
                        # TODO: Set the scalar tensor type same as scalar type after to_dtype supported
                        # (open issue: https://github.com/torch-spyre/torch-spyre/issues/41)
                        dtype = torch.float16
                        meta = node.meta.get("tensor_meta", None)
                        if meta:
                            dtype = meta.dtype
                        node_name = "py_const"
                        node_args = [scalar_val, dtype, torch.device("spyre")]
                        const_node = graph.create_node(
                            "call_function",
                            torch.ops.spyre.constant.default,
                            tuple(node_args),
                            {},
                            node_name,
                            node.type,
                        )
                        const_node_map[scalar_val] = const_node
                    node.update_arg(idx, const_node)

    graph.lint()
