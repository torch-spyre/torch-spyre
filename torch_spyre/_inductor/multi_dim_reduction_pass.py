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

"""
FX Graph pass to decompose multi-dimensional reductions into sequences of
single-dimension reductions.

The transformation is applied to reduction operations that reduce along multiple
dimensions, rewriting them as a sequence of single-dimension reductions.
"""

from typing import List, Optional, Union
import torch
from torch import fx

# Reduction operations that support multi-dimensional reduction
# TODO: Only sum is tested for multi-dimensional reduction.
# Example implementation for now
MULTI_DIM_REDUCTION_OPS = {
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.dim,
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
    torch.ops.aten.prod.dim_int,
}


def _normalize_dims(dims: Union[int, List[int]], ndim: int) -> List[int]:
    """
    Normalize dimension indices to positive values and return as sorted list.
    """
    if isinstance(dims, int):
        dims = [dims]

    normalized = []
    for d in dims:
        if d < 0:
            d = ndim + d
        if d < 0 or d >= ndim:
            raise ValueError(
                f"Dimension {d} out of range for tensor with {ndim} dimensions"
            )
        normalized.append(d)

    return sorted(set(normalized), reverse=True)


def _get_reduction_dims(node: fx.Node) -> Optional[Union[int, List[int]]]:
    """
    Extract the dimension(s) being reduced from a reduction node.
    """
    if "dim" in node.kwargs:
        return node.kwargs["dim"]

    # Sanity check for dim - sum, mean, amax, amin, prod

    if node.target in {torch.ops.aten.sum.dim_IntList, torch.ops.aten.mean.dim}:
        if len(node.args) >= 2:
            return node.args[1]
    elif node.target in {torch.ops.aten.amax.default, torch.ops.aten.amin.default}:
        if len(node.args) >= 2:
            return node.args[1]
    elif node.target == torch.ops.aten.prod.dim_int:
        if len(node.args) >= 2:
            return node.args[1]

    return None


def _get_keepdim(node: fx.Node) -> bool:
    """
    Extract the keepdim parameter from a reduction node.
    """
    if "keepdim" in node.kwargs:
        return node.kwargs["keepdim"]

    if node.target in {
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.mean.dim,
        torch.ops.aten.amax.default,
        torch.ops.aten.amin.default,
        torch.ops.aten.prod.dim_int,
    }:
        # most of cases keepdim is there  the 3rd argument
        if len(node.args) >= 3:
            return node.args[2]

    return False


def _get_dtype(node: fx.Node) -> Optional[torch.dtype]:
    """
    Extract the dtype parameter from a reduction node.
    """
    if "dtype" in node.kwargs:
        return node.kwargs["dtype"]

    if node.target in {
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.mean.dim,
        torch.ops.aten.prod.dim_int,
    }:
        # dtype is typically the 4th argument
        if len(node.args) >= 4:
            return node.args[3]

    return None


def _decompose_multi_dim_reduction(
    graph: fx.Graph,
    node: fx.Node,
    dims: List[int],
    keepdim: bool,
    dtype: Optional[torch.dtype],
) -> fx.Node:
    """
    Decompose a multi-dimensional reduction into a sequence of single-dim reductions.
    """
    input_node = node.args[0]
    current = input_node

    # Process each dimension, reducing one at a time
    # Strategy: Always use the user's keepdim setting for ALL reductions
    # When keepdim=False, we need to adjust dimension indices after each reduction
    for i, dim in enumerate(dims):
        is_last = i == len(dims) - 1

        # Since dims is sorted in descending order, we reduce from highest to lowest
        adjusted_dim = dim

        # Create the single-dimension reduction node
        with graph.inserting_before(node):
            # Build positional args for the reduction
            args = (current, adjusted_dim, keepdim)
            # Build kwargs for the reduction
            # only dtype as kwargs only for the last reduction as needed basis.
            kwargs = {}
            if dtype is not None and is_last:
                kwargs["dtype"] = dtype

            current = graph.call_function(
                node.target,
                args=args,
                kwargs=kwargs,
            )

    return current


def decompose_multi_dim_reductions(graph: fx.Graph) -> None:
    """
    FX Graph pass to decompose multi-dimensional reductions into sequences of
    single-dimension reductions.
    This pass iterates through the graph and transforms any reduction operation
    that reduces along multiple dimensions into a sequence of single-dimension
    reductions.
    """
    replacements = {}

    for node in graph.nodes:
        if node.op != "call_function" or node.target not in MULTI_DIM_REDUCTION_OPS:
            continue

        # helper function to get reduction parameter
        dims = _get_reduction_dims(node)
        if dims is None:
            continue

        # reducing along a single dimension, then no need for decomposition
        if isinstance(dims, int) or len(dims) <= 1:
            continue

        keepdim = _get_keepdim(node)
        dtype = _get_dtype(node)

        input_node = node.args[0]
        if not hasattr(input_node, "meta") or "val" not in input_node.meta:
            continue

        input_val = input_node.meta["val"]
        if not isinstance(input_val, torch.Tensor):
            continue

        ndim = input_val.ndim

        try:
            normalized_dims = _normalize_dims(dims, ndim)
        except ValueError:
            continue

        # Decompose the multi-dimensional reduction
        replacement = _decompose_multi_dim_reduction(
            graph, node, normalized_dims, keepdim, dtype
        )

        replacements[node] = replacement

    # Replace all uses of original nodes with their decomposed versions
    for old_node, new_node in replacements.items():
        old_node.replace_all_uses_with(new_node)
        graph.erase_node(old_node)

    graph.lint()
