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

"""Loop unrolling for coarse-tiling LoopSpec trees.

When ``bundle_hbm_symbols=False`` the backend compiler does not support
``scf.for`` loops in ``bundle.mlir``.  This module provides
``unroll_loop_specs``, which fully unrolls a ``list[OpSpec | LoopSpec]`` tree
into a flat list of ``OpSpec`` entries with concrete per-iteration HBM
addresses baked into each ``TensorArg.allocation['hbm']``.

Each iteration produces an independent copy of the inner ``OpSpec`` objects
with the tiled-dimension HBM addresses advanced by the per-iteration byte
offset.  LX / pool (scratchpad / temporary) tensors are left unchanged — they
hold the same fixed address every iteration.  After unrolling,
``tiled_symbols`` is cleared on every copy so ``generate_bundle`` treats the
op as a plain non-tiled entry.

Nested ``LoopSpec`` nodes (e.g. outer K=2 / inner M=4) are unrolled
innermost-first, yielding K×M flat copies with correct combined addresses.
"""

from __future__ import annotations

import copy
import sympy
from sympy import Symbol

from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
from torch_spyre._inductor.codegen.superdsc import parse_op_spec
from torch_spyre._inductor.codegen.compute_ops import _tiled_byte_stride
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("codegen.unroll")


def _compute_tiled_byte_strides(op_spec: OpSpec) -> dict[Symbol, int]:
    """Return {tiled_sym: byte_stride} for each tiled HBM dimension.

    For every symbol in ``op_spec.tiled_symbols`` that maps to at least one
    HBM ``SDSCArgs`` (non-lx, non-pool) with a matching stride entry, compute
    the total byte advance per loop step using ``_tiled_byte_stride``.

    Returns an empty dict when ``op_spec.tiled_symbols`` is empty.
    """
    if not op_spec.tiled_symbols:
        return {}

    sdsc_spec, symbol_mapping = parse_op_spec(op_spec)

    byte_strides: dict[Symbol, int] = {}
    for tiled_sym in op_spec.tiled_symbols:
        sdsc_sym = symbol_mapping.get(tiled_sym)
        if sdsc_sym is None:
            continue
        # Use the first HBM arg that carries this symbol as its stride
        # representative.  All HBM tensors tiled on the same dimension share
        # the same per-iteration byte advance (they are slices of tensors with
        # the same shape/stride structure).
        for sdsc_arg in sdsc_spec.args:
            if "lx" in sdsc_arg.allocation or "pool" in sdsc_arg.allocation:
                continue
            if sdsc_sym not in sdsc_arg.strides:
                continue
            byte_strides[tiled_sym] = _tiled_byte_stride(
                sdsc_arg, sdsc_sym, sdsc_spec.iteration_space
            )
            break

    return byte_strides


def _unroll_one(
    loop: LoopSpec,
    accumulated_offset: int,
) -> list:
    """Unroll a single LoopSpec node, returning flat OpSpec copies.

    ``accumulated_offset`` is the HBM byte offset already applied by outer
    loop iterations; it is added to the per-iteration offset computed here
    before patching each copy's ``TensorArg.allocation['hbm']``.
    """
    # --- Recursively unroll any nested LoopSpecs in body first. ----------
    flat_body: list[OpSpec] = []
    for entry in loop.body:
        if isinstance(entry, LoopSpec):
            flat_body.extend(_unroll_one(entry, accumulated_offset=0))
        else:
            flat_body.append(entry)

    # --- Evaluate trip count. --------------------------------------------
    count_expr = sympy.sympify(loop.count)
    if count_expr.free_symbols:
        raise ValueError(
            f"unroll_loop_specs: LoopSpec count {loop.count!r} contains free "
            f"symbols {count_expr.free_symbols} and cannot be statically unrolled."
        )
    count = int(count_expr)

    # --- Pre-compute byte strides for each OpSpec in body once. ----------
    strides_per_op: list[dict[Symbol, int]] = []
    for entry in flat_body:
        if isinstance(entry, OpSpec):
            strides_per_op.append(_compute_tiled_byte_strides(entry))
        else:
            strides_per_op.append({})

    # --- Emit count copies, advancing HBM addresses per iteration. -------
    result: list = []
    for i in range(count):
        for entry, byte_strides in zip(flat_body, strides_per_op):
            if not isinstance(entry, OpSpec):
                result.append(copy.deepcopy(entry))
                continue

            op_copy = copy.deepcopy(entry)

            # Advance each HBM TensorArg by accumulated_offset + i * stride.
            iter_offset = sum(
                i * byte_strides[s] for s in entry.tiled_symbols if s in byte_strides
            )
            total_offset = accumulated_offset + iter_offset
            if total_offset:
                for arg in op_copy.args:
                    if "lx" in arg.allocation or "pool" in arg.allocation:
                        continue
                    if "hbm" in arg.allocation:
                        arg.allocation = dict(arg.allocation)
                        arg.allocation["hbm"] += total_offset

            # Clear tiled_symbols: addresses are now concrete.
            op_copy.tiled_symbols = []
            result.append(op_copy)

    logger.debug(
        "unrolled LoopSpec(count=%s) → %d flat copies", loop.count, len(result)
    )
    return result


def unroll_loop_specs(specs: list) -> list:
    """Fully unroll all LoopSpec nodes in specs, returning a flat spec list.

    Each ``LoopSpec(count=K, body=[...])`` is replaced by K copies of its
    body, with tiled HBM addresses in each ``TensorArg.allocation['hbm']``
    advanced by the per-iteration byte offset for that loop index.

    LX / pool (scratchpad / temporary) tensors are left unchanged — they hold
    the same fixed address every iteration.  Non-tiled HBM tensors are also
    unchanged.  ``tiled_symbols`` is cleared on every copy so
    ``generate_bundle`` treats the ops as plain non-tiled entries.

    ``count`` must be a concrete integer expression; symbolic counts raise
    ``ValueError``.  Nested ``LoopSpec`` nodes are unrolled innermost-first.
    """
    result: list = []
    for entry in specs:
        if isinstance(entry, LoopSpec):
            result.extend(_unroll_one(entry, accumulated_offset=0))
        else:
            result.append(entry)
    return result
