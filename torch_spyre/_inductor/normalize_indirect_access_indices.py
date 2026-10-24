# Copyright 2026 The Torch-Spyre Authors.
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

"""Wrap negative advanced-indexing indices before lowering.

PyTorch leaves the wrap to the backend: it emits
``indirect_indexing(var, size, wrap_neg=True)`` and expects the ops handler to
honour the flag.  Spyre's handler stands in for upstream's ``CSEProxy`` without
its bounds field, so the flag is dropped and an unwrapped ``-1`` reaches the
device as 4294967295, faulting the compute block.

Codegen is too late to repair: indirect access reads the index out of memory
and needs a named buffer, so the wrap has to produce a real tensor.

Each index becomes ``idx - size * floor(idx / size)`` in fp32, since the device
has no integer compare or divide and fp32 is exact below 2**24.  An index
outside ``[-size, size)`` then reads a defined but wrong row rather than raising
``IndexError``; nothing checks bounds on device.
"""

import torch
from torch.fx import Graph, Node

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("normalize_indirect_access_indices")

aten = torch.ops.aten
prims = torch.ops.prims

# Advanced indexing only: gather and embedding reject negative indices in
# eager, so wrapping them would diverge from CPU.
_INDEX_OPS = (
    aten.index.Tensor,
    aten._unsafe_index.Tensor,
    aten.index_put.default,
)

# bool/uint8 index entries are masks, not positions, and are left alone.
_INDEX_DTYPES = (torch.int64, torch.int32)


def _is_non_negative(index: Node) -> bool:
    """Whether ``index`` provably yields values >= 0, so the wrap can be skipped.

    Conservative: an unproven source is wrapped, which costs ops but never a
    wrong result.
    """
    if index.op != "call_function":
        return False
    if index.target is aten.arange.default:  # arange(end) starts at 0
        return True
    if index.target is aten.arange.start:  # step is 1, so start bounds it
        return isinstance(index.args[0], int) and index.args[0] >= 0
    if index.target is aten.arange.start_step:  # a negative step descends
        return all(isinstance(a, int) and a >= 0 for a in index.args[:3:2])
    return False


def _emit(graph: Graph, target, args, dtype, shape) -> Node:
    """Insert a call_function node carrying a fake tensor as ``meta['val']``."""
    node = graph.call_function(target, args=args)
    node.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
    return node


def _convert(graph: Graph, src: Node, dtype, shape) -> Node:
    return _emit(graph, prims.convert_element_type.default, (src, dtype), dtype, shape)


def _wrap(graph: Graph, before: Node, idx: Node, size: int) -> Node:
    """Insert ``idx % size`` before ``before`` and return the node holding it."""
    shape = tuple(idx.meta["val"].shape)

    with graph.inserting_before(before):
        as_f32 = _convert(graph, idx, torch.float32, shape)

        # Floor-modulo, not a select: every value lands in [0, size).
        quot = _emit(
            graph, aten.div.Scalar, (as_f32, float(size)), torch.float32, shape
        )
        whole = _emit(graph, aten.floor.default, (quot,), torch.float32, shape)
        scaled = _emit(
            graph, aten.mul.Scalar, (whole, float(size)), torch.float32, shape
        )
        wrapped = _emit(graph, aten.sub.Tensor, (as_f32, scaled), torch.float32, shape)

        # Stop at int32: index ops accept it, and int64 is IEEE_INT32 anyway.
        cur = _convert(graph, wrapped, torch.int32, shape)

    logger.debug("wrapped %s (size=%d) -> %s", idx.name, size, cur.name)
    return cur


def normalize_indirect_access_indices(graph: Graph) -> None:
    """Wrap every possibly-negative index of an advanced-indexing op."""
    dirty = False
    for node in graph.nodes:
        if node.op != "call_function" or node.target not in _INDEX_OPS:
            continue
        self_val = getattr(node.args[0], "meta", {}).get("val")
        indices = node.args[1]
        if self_val is None or not isinstance(indices, (list, tuple)):
            continue

        rewritten = list(indices)
        changed = False
        for dim, index in enumerate(indices):  # indices[dim] selects along dim
            if not isinstance(index, Node):
                continue
            val = index.meta.get("val")
            if val is None or val.dtype not in _INDEX_DTYPES:
                continue
            if _is_non_negative(index):
                continue
            rewritten[dim] = _wrap(graph, node, index, int(self_val.shape[dim]))
            changed = True

        if changed:
            node.args = (node.args[0], type(indices)(rewritten), *node.args[2:])
            dirty = True

    if dirty:
        graph.lint()
