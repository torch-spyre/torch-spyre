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

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.virtualized import V

from .ir import SpyreConstantFallback
from .insert_restickify import NameSwapHandler
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("dedup_constants")


def _constant_key(op: SpyreConstantFallback) -> tuple:
    """Normalised (value, dtype, device) identity key for a SpyreConstantFallback."""
    layout = op.layout
    dev = layout.device
    norm_dev = (
        torch.device(dev.type, dev.index)
        if dev.index is not None
        else torch.device(dev.type)
    )
    return (op.constant_args[0], layout.dtype, norm_dev)


def _patch_inner_fn(consumer: ComputedBuffer, name_map: dict[str, str]) -> None:
    """Wrap consumer's inner_fn to redirect duplicate constant reads to the canonical name."""
    orig_inner = consumer.data.inner_fn

    def _new_inner(*args, _map=name_map, _orig=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig(*args)

    object.__setattr__(consumer.data, "inner_fn", _new_inner)
    ComputedBuffer.get_default_sizes_body.clear_cache(consumer)


def _redirect_consumers(
    operations: list[Operation],
    dup: SpyreConstantFallback,
    canonical: SpyreConstantFallback,
) -> None:
    """Rewrite every ComputedBuffer consumer of dup to read canonical instead."""
    D = dup.get_name()
    C = canonical.get_name()
    name_map = {D: C}

    # Do not dedup a constant that is itself a graph output.
    if D in V.graph.get_output_names():
        logger.debug("dedup_and_promote_constants: skipping output constant %s", D)
        return

    for op in operations:
        if op is dup or op is canonical:
            continue
        rw = op.get_read_writes()
        if not any(dep.name == D for dep in rw.reads):
            continue
        if isinstance(op, ComputedBuffer):
            _patch_inner_fn(op, name_map)
        else:
            raise AssertionError(
                f"dedup_and_promote_constants: unsupported consumer type "
                f"{type(op).__name__} reads constant {D!r} — cannot rewrite"
            )


def _drop_constant(
    operations: list[Operation],
    dup: SpyreConstantFallback,
    canonical: SpyreConstantFallback,
) -> None:
    """Remove a duplicate constant from the graph and update bookkeeping."""
    D = dup.get_name()
    C = canonical.get_name()
    op_name = dup.get_operation_name()
    operations.remove(dup)
    V.graph.removed_buffers.add(D)
    V.graph.name_to_buffer.pop(D, None)
    V.graph.name_to_op.pop(op_name, None)
    # Merge the duplicate's users into the canonical's user list so that passes
    # which iterate name_to_users (e.g. scratchpad planning) see the full set.
    extra_users = V.graph.name_to_users.pop(D, [])
    if extra_users:
        V.graph.name_to_users.setdefault(C, []).extend(extra_users)
    logger.debug("dedup_and_promote_constants: merged %s into canonical %s", D, C)


def dedup_and_promote_constants(graph: GraphLowering) -> None:
    """Deduplicate SpyreConstantFallback ops and move them to the head of operations.

    Steps:
      1. Group SpyreConstantFallback ops by (value, dtype, device).
      2. For each group with >1 instance, keep the first (canonical); rewrite
         ComputedBuffer consumers of each duplicate to read from canonical, then
         drop the duplicate using the removed_buffers convention.
      3. Move all surviving SpyreConstantFallback ops to the front of
         operations, preserving relative order.

    Mutates operations in place.
    """
    operations = graph.operations
    # --- Step 1: group by identity key ---
    groups: dict[tuple, list[SpyreConstantFallback]] = {}
    for op in operations:
        if not isinstance(op, SpyreConstantFallback):
            continue
        key = _constant_key(op)
        groups.setdefault(key, []).append(op)

    # --- Step 2: dedup ---
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        canonical = group[0]
        for dup in group[1:]:
            _redirect_consumers(operations, dup, canonical)
            _drop_constant(operations, dup, canonical)

    # --- Step 3: front-load surviving constants ---
    constants = [op for op in operations if isinstance(op, SpyreConstantFallback)]
    if not constants:
        return
    non_constants = [
        op for op in operations if not isinstance(op, SpyreConstantFallback)
    ]
    operations[:] = constants + non_constants
    logger.debug(
        "dedup_and_promote_constants: %d constant(s) promoted to front of operations",
        len(constants),
    )
