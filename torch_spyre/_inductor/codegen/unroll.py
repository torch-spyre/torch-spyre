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
with the tiled-dimension HBM addresses advanced by the per-arg, per-iteration
byte offset computed from each ``TensorArg``'s ``device_coordinates`` and
``device_size``.  LX / pool tensors are left unchanged — they hold the same
fixed address every iteration.  After unrolling, ``tiled_symbols`` is cleared
on every copy so ``generate_bundle`` treats the op as a plain non-tiled entry.

Nested ``LoopSpec`` nodes (e.g. outer K=2 / inner M=4) are unrolled
innermost-first, yielding K×M flat copies with correct combined addresses.
"""

from __future__ import annotations

import copy
import math

import sympy
from sympy import Symbol

from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("codegen.unroll")


def _hbm_byte_stride_for_arg(arg: TensorArg, tiled_sym: Symbol, tile_range: int) -> int:
    """Byte advance in HBM per loop iteration for a single TensorArg.

    Uses ``device_coordinates`` and ``device_size`` to compute how far the HBM
    base address must advance when the loop index advances by one iteration
    (i.e., when ``tiled_sym`` shifts by ``tile_range`` elements).

    For each device dimension ``d``, the element-stride (elements per unit of
    that dimension) is ``prod(device_size[d+1:])``.  The contribution of
    ``tiled_sym`` to dimension ``d`` is the sympy coefficient of ``tiled_sym``
    in ``device_coordinates[d]``.  Summing over all dimensions gives the total
    element advance per unit of ``tiled_sym``; multiplying by ``tile_range``
    and ``bytes_per_element`` gives the byte advance per loop iteration.
    """
    total_elem_stride = 0
    ndim = len(arg.device_size)
    for d, coord_expr in enumerate(arg.device_coordinates):
        coeff = coord_expr.coeff(tiled_sym)
        if coeff == 0:
            continue
        # Element-stride for device dimension d: product of all trailing sizes.
        elem_stride = math.prod(arg.device_size[d + 1 :]) if d + 1 < ndim else 1
        total_elem_stride += int(coeff) * elem_stride
    return total_elem_stride * tile_range * num_bytes(arg.device_dtype)


def _arg_byte_strides(
    op_spec: OpSpec,
) -> list[dict[Symbol, int]]:
    """Return per-arg, per-tiled-sym byte strides for all args in op_spec.

    Returns a list parallel to ``op_spec.args``.  Each entry is a
    ``{tiled_sym: byte_stride}`` dict for that arg (empty for pool/lx args
    and args with no tiled-symbol contribution).
    """
    if not op_spec.tiled_symbols:
        return [{} for _ in op_spec.args]

    result: list[dict[Symbol, int]] = []
    for arg in op_spec.args:
        if "lx" in arg.allocation or "pool" in arg.allocation:
            result.append({})
            continue
        if "hbm" not in arg.allocation:
            result.append({})
            continue

        strides: dict[Symbol, int] = {}
        for tiled_sym in op_spec.tiled_symbols:
            if tiled_sym not in op_spec.iteration_space:
                continue
            tile_range = int(op_spec.iteration_space[tiled_sym][0])
            stride = _hbm_byte_stride_for_arg(arg, tiled_sym, tile_range)
            if stride != 0:
                strides[tiled_sym] = stride
        result.append(strides)
    return result


def _unroll_one(
    loop: LoopSpec,
    accumulated_offsets: list[int],
) -> list:
    """Unroll a single LoopSpec node, returning flat OpSpec copies.

    ``accumulated_offsets`` is a list of per-arg HBM byte offsets already
    applied by outer loop iterations.  It is parallel to the args of each
    OpSpec in the flattened body; outer loops pass their offsets inward so
    nested strides accumulate correctly.

    For a top-level call pass an empty list (``[]``); offsets are computed
    lazily per op.
    """
    # --- Recursively unroll any nested LoopSpecs in body first. ----------
    flat_body: list[OpSpec] = []
    for entry in loop.body:
        if isinstance(entry, LoopSpec):
            flat_body.extend(_unroll_one(entry, accumulated_offsets=[]))
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

    # --- Pre-compute per-arg byte strides for each OpSpec in body once. --
    strides_per_op: list[list[dict[Symbol, int]]] = []
    for entry in flat_body:
        if isinstance(entry, OpSpec):
            strides_per_op.append(_arg_byte_strides(entry))
        else:
            strides_per_op.append([])

    # --- Emit count copies, advancing HBM addresses per iteration. -------
    result: list = []
    for i in range(count):
        for entry, arg_strides in zip(flat_body, strides_per_op):
            if not isinstance(entry, OpSpec):
                result.append(copy.deepcopy(entry))
                continue

            op_copy = copy.deepcopy(entry)

            for arg_idx, (arg, strides) in enumerate(zip(op_copy.args, arg_strides)):
                if not strides:
                    continue
                iter_offset = sum(
                    i * strides[s] for s in entry.tiled_symbols if s in strides
                )
                if iter_offset:
                    arg.allocation = dict(arg.allocation)
                    arg.allocation["hbm"] += iter_offset

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
    body.  For each HBM ``TensorArg`` the base address is advanced by the
    per-arg, per-iteration byte offset derived from ``device_coordinates`` and
    ``device_size`` — so args with different tile sizes or layouts each get
    the correct independent advance.

    Pool and LX tensors are left unchanged.  ``tiled_symbols`` is cleared on
    every copy so ``generate_bundle`` treats the ops as plain non-tiled entries.

    ``count`` must be a concrete integer expression; symbolic counts raise
    ``ValueError``.  Nested ``LoopSpec`` nodes are unrolled innermost-first.
    """
    result: list = []
    for entry in specs:
        if isinstance(entry, LoopSpec):
            result.extend(_unroll_one(entry, accumulated_offsets=[]))
        else:
            result.append(entry)
    return result
