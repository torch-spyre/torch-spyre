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

"""Turn a broadcast scalar that feeds an indexed write into a real tensor.

Upstream builds the value of a scalar-valued write as a stride-0 view::

    scalar_tensor(7.0).expand(3, 8, 128)      # 3072 positions, ONE number

That is a good optimization nearly everywhere, and fatal here.  Spyre's
indirect-access engine has to physically walk memory, and asks which dimension
of the value advances as the write advances across rows.  For a stride-0 view
the answer is "none": every position reads the same byte.  The value never
gets a buffer at all, so codegen rejects it ("store value of unexpected type
Constant"), and when the value is forced to look shaped without being shaped,
the backend's per-dim lookup returns -1 and ``dbo-opt`` aborts.

This pass replaces the view with a real tensor::

    full([3, 8, 128], 7.0)

Same values, now actually in memory, with strides that advance -- so dim 0 of
the value lines up with the indexed dim of the write, which is exactly the
correspondence the backend looks for.

Why here.  It runs on the post-grad FX graph, where the ``expand`` node still
carries its target shape, so the rewrite is local: read the size off the node,
emit a ``full``, repoint the users.  Nothing is reconstructed.  After lowering
the ``expand`` has been folded into the consumer's index expression as a
constant 0, and recovering the mapping then means rebuilding symbolic index
expressions by hand -- the silent-wrong-code hazard of issue #2797.

Because the pattern is matched on the VALUE, not on the consuming operator,
one rewrite covers ``index_fill``, ``scatter.value``, ``scatter.value_reduce``
and any future op that writes a broadcast scalar.  Related: torch-spyre#4446,
#4472.
"""

import torch

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("materialize_scalar_fill")

aten = torch.ops.aten

# Ops whose value operand must be real memory: the indexed writes that run on
# the indirect-access engine.  The rewrite itself is value-shaped and would be
# correct for any consumer, but materializing a broadcast scalar costs memory,
# so it is limited to the writes that cannot work without it.
_INDEXED_WRITES = frozenset(
    {
        aten.index_put.default,
        aten.index_put_.default,
        aten._unsafe_index_put.default,
        aten.index_copy.default,
        aten.index_copy_.default,
        aten.scatter.src,
        aten.scatter_.src,
        aten.scatter.reduce,
        aten.scatter_.reduce,
        aten.scatter_add.default,
        aten.slice_scatter.default,
        aten.select_scatter.default,
    }
)

# Transparent for the purposes of "is this ultimately a scalar?".
_PASSTHROUGH = frozenset(
    {
        aten.to.dtype,
        aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
        aten.clone.default,
        aten.detach.default,
    }
)

# Nodes that produce a single value with no real extent.
_SCALAR_SOURCES = frozenset({aten.scalar_tensor.default, aten.lift_fresh_copy.default})


def _scalar_source(node) -> torch.fx.Node | None:
    """The ``scalar_tensor``-like node behind ``node``, through dtype casts."""
    seen = 0
    while isinstance(node, torch.fx.Node) and node.op == "call_function":
        if node.target in _SCALAR_SOURCES:
            return node
        if node.target not in _PASSTHROUGH or not node.args:
            return None
        node = node.args[0]
        seen += 1
        if seen > 8:  # defensive: never loop on a malformed chain
            return None
    return None


def _feeds_indexed_write(node) -> bool:
    """True when any user of ``node`` is an indexed write."""
    return any(
        user.op == "call_function" and user.target in _INDEXED_WRITES
        for user in node.users
    )


def materialize_scalar_fill(graph: torch.fx.Graph) -> None:
    """Rewrite ``expand(scalar_tensor(v), size)`` feeding a write into ``full``."""
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.expand.default:
            continue
        if len(node.args) < 2:
            continue
        src = _scalar_source(node.args[0])
        if src is None:
            logger.debug("%s: not a broadcast scalar, skipped", node.name)
            continue
        if not _feeds_indexed_write(node):
            logger.debug(
                "%s: broadcast scalar but no indexed-write user (%s), left alone",
                node.name,
                [u.target for u in node.users],
            )
            continue

        size = list(node.args[1])
        value = src.args[0] if src.args else None
        if value is None:
            continue
        example = node.meta.get("val")
        if example is None:
            logger.debug("%s: no meta['val'], cannot type the full", node.name)
            continue

        with graph.inserting_before(node):
            full = graph.call_function(
                aten.full.default,
                (size, value),
                {"dtype": example.dtype, "device": example.device},
            )
        full.meta["val"] = example
        node.replace_all_uses_with(full)
        graph.erase_node(node)
        logger.debug(
            "materialized %s -> full(%s, %s) for %s",
            node.name,
            size,
            value,
            full.name,
        )
