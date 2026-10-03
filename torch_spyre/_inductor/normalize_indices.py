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

"""Wrap negative advanced-indexing indices before lowering.

PyTorch does not wrap negative indices itself.  It emits
``indirect_indexing(var, size, wrap_neg=True)`` and expects the backend's ops
handler to honour the flag.  Spyre's handler cannot: it stands in for
upstream's ``CSEProxy``, which reads a bounds field ``TensorAccess`` does not
have, and the flag is dropped.  An unwrapped ``-1`` then reaches the device as
an unsigned 4294967295, far outside the tensor, and faults the compute block.

Codegen is too late to repair that, because indirect access reads the index
out of memory and so needs a named buffer.  Wrapping here, on the graph, is
what makes the wrapped index a real tensor by the time codegen asks for one.

Each index becomes ``idx - size * floor(idx / size)`` in fp32.  fp32 because
the device has no integer compare or divide, and it is exact for integers
below 2**24.  int64 reaches fp32 through int32, which costs nothing since both
share a device format; converting int64 directly is unsupported and would fall
back to the host.

Indices outside ``[-size, size)`` read a defined but wrong row instead of
raising ``IndexError``.  Nothing checks bounds on device, and a wrong read is
preferable to an address that takes the card down.
"""

import torch
from torch.fx import Graph, Node

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("normalize_indices")

aten = torch.ops.aten
prims = torch.ops.prims

# post_grad_custom_pre_pass fires more than once; a later run skips marked nodes.
_WRAPPED = "_spyre_index_wrapped"

# Ops upstream tags wrap_neg=True.  gather/embedding/scatter pass False and
# reject negatives on CPU too, so they are deliberately absent.
_INDEX_OPS = (
    aten.index.Tensor,
    aten._unsafe_index.Tensor,
    aten.index_put.default,
    aten.index_put_.default,
)

# bool/uint8 index entries are masks, not positions, and are left alone.
_INDEX_DTYPES = (torch.int64, torch.int32)

_NON_NEGATIVE_OPS = (
    aten.argmax.default,
    aten.argmin.default,
    aten.nonzero.default,
    aten.abs.default,
)


def _is_non_negative(node: object) -> bool:
    """Whether ``node`` provably yields values >= 0, so the wrap can be skipped.

    Conservative: an unproven node is treated as possibly negative, which costs
    a redundant wrap but never a wrong result.
    """
    if not isinstance(node, Node):
        return False
    if node.op == "get_attr":  # lifted constant: read the values
        gm = node.graph.owning_module
        val = getattr(gm, node.target, None) if gm is not None else None
        return isinstance(val, torch.Tensor) and bool((val >= 0).all())
    if node.op != "call_function":
        return False
    if node.target in _NON_NEGATIVE_OPS:
        return True
    # arange(end) and randint(high) start at 0; the low-bound overloads need it checked.
    if node.target in (aten.arange.default, aten.randint.default):
        return True
    if node.target in (aten.arange.start, aten.arange.start_step, aten.randint.low):
        return isinstance(node.args[0], int) and node.args[0] >= 0
    return False


def _emit(graph: Graph, target, args, dtype, shape, kwargs=None) -> Node:
    """Insert a call_function node carrying a fake tensor as ``meta['val']``."""
    node = graph.call_function(target, args=args, kwargs=kwargs or {})
    node.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
    return node


def _convert(graph: Graph, src: Node, dtype, shape) -> Node:
    return _emit(graph, prims.convert_element_type.default, (src, dtype), dtype, shape)


def _wrap(graph: Graph, before: Node, idx: Node, size: int) -> Node:
    """Insert ``idx % size`` before ``before`` and return the node holding it."""
    from torch_spyre._C import get_elem_in_stick

    val = idx.meta["val"]
    stick = int(get_elem_in_stick(torch.int32))
    dims = val.shape
    numel = None
    if all(isinstance(d, int) for d in dims):
        numel = 1
        for d in dims:
            numel *= d

    # WORKAROUND: int32tofp32 does not lower on a partial stick. An index
    # buffer is allocated rounded up to whole sticks, so this wider view reads
    # only slots that already exist and a slice restores the logical length.
    # REMOVE when partial sticks lower: drop this block and the slice below.
    widen = numel is not None and numel % stick != 0
    padded = ((numel + stick - 1) // stick) * stick if widen and numel else 0

    with graph.inserting_before(before):
        cur = idx
        if widen:
            if not val.is_contiguous():
                # as_strided pins stride 1, so a strided index (x[big[::2]])
                # would be read as consecutive elements: wrong rows, no error.
                cur = _emit(
                    graph,
                    aten.clone.default,
                    (cur,),
                    val.dtype,
                    tuple(dims),
                    kwargs={"memory_format": torch.contiguous_format},
                )
            if val.dim() != 1:  # as_strided describes one contiguous run
                cur = _emit(
                    graph, aten.reshape.default, (cur, [numel]), val.dtype, (numel,)
                )
            cur = _emit(
                graph,
                aten.as_strided.default,
                (cur, [padded], [1]),
                val.dtype,
                (padded,),
            )

        shape = (padded,) if widen else tuple(dims)

        if val.dtype != torch.int32:
            cur = _convert(graph, cur, torch.int32, shape)
        as_f32 = _convert(graph, cur, torch.float32, shape)

        # Floor-modulo rather than upstream's select: it maps *every* value into
        # [0, size), including the junk in the widened padding lanes above.
        quot = _emit(
            graph, aten.div.Scalar, (as_f32, float(size)), torch.float32, shape
        )
        whole = _emit(graph, aten.floor.default, (quot,), torch.float32, shape)
        scaled = _emit(
            graph, aten.mul.Scalar, (whole, float(size)), torch.float32, shape
        )
        wrapped = _emit(graph, aten.sub.Tensor, (as_f32, scaled), torch.float32, shape)

        # Stop at int32: index ops accept it and the device format is
        # IEEE_INT32 either way, so converting back would be a dead byte copy.
        cur = _convert(graph, wrapped, torch.int32, shape)

        if widen:
            cur = _emit(
                graph, aten.slice.Tensor, (cur, 0, 0, numel), torch.int32, (numel,)
            )
            if val.dim() != 1:
                cur = _emit(
                    graph,
                    aten.reshape.default,
                    (cur, list(dims)),
                    torch.int32,
                    tuple(dims),
                )
        cur.meta[_WRAPPED] = True

    logger.debug(
        "wrapped %s (size=%d, widened=%s) -> %s", idx.name, size, widen, cur.name
    )
    return cur


def normalize_negative_indices(graph: Graph) -> None:
    """Wrap every possibly-negative index of an advanced-indexing op."""
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
            if not isinstance(index, Node) or index.meta.get(_WRAPPED):
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

    graph.lint()
