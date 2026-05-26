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
into a flat list of ``OpSpec`` entries with concrete per-iteration addresses
baked into each ``TensorArg.allocation``.

Whether a tensor's address advances per iteration is determined solely by
``TensorArg.per_tile_fixed`` (set by ``insert_tiling_propagation``'s use-def
analysis):

- ``per_tile_fixed=True``: tile-local scratch reused every iteration — fixed.
- ``per_tile_fixed=False``: address advances per iteration regardless of
  allocation type (hbm, pool, or lx).

After unrolling, ``tiled_symbols`` is cleared on every copy so
``generate_bundle`` treats the ops as plain non-tiled entries.

Nested ``LoopSpec`` nodes (e.g. outer K=2 / inner M=4) are unrolled
innermost-first, yielding K×M flat copies with correct combined addresses.
"""

from __future__ import annotations

import copy

import sympy
from sympy import Symbol

from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("codegen.unroll")


def _byte_stride_for_arg(arg: TensorArg, tiled_sym: Symbol, tile_range: int) -> int:
    """Byte advance per loop iteration for a single TensorArg.

    Computes the byte advance using:
        delta[d] = coord_d(sym=tile_range, others=0) - coord_d(sym=0, others=0)
        byte_stride = dot(delta, stride_map) * bytes_per_elem

    This correctly handles non-linear device coordinates such as the stick
    layout's ``floor(c/64)`` and ``Mod(c, 64)`` expressions.
    """
    assert arg.stride_map is not None, (
        "_byte_stride_for_arg: TensorArg has no stride_map — "
        "all TensorArgs reaching the unroller must be constructed with stride_map"
    )
    all_syms: set = set()
    for expr in arg.device_coordinates:
        all_syms |= expr.free_symbols
    sub_zero = {s: 0 for s in all_syms}
    sub_range = {**sub_zero, tiled_sym: tile_range}
    total_elem_stride = 0
    for d, coord_expr in enumerate(arg.device_coordinates):
        at_range = int(coord_expr.subs(sub_range))
        at_zero = int(coord_expr.subs(sub_zero))
        total_elem_stride += (at_range - at_zero) * arg.stride_map[d]
    return total_elem_stride * num_bytes(arg.device_dtype)


def _tile_device_size(
    arg: TensorArg, tiled_syms: list[Symbol], iteration_space: dict
) -> list[int]:
    """Compute the device_size that describes the tile, not the full tensor.

    After unrolling, each copy carries an absolute start address for its tile.
    The SDSC must describe the tile geometry so the hardware does not apply a
    backGap for the row dimension (mb) — that gap is already encoded in the
    start address.  The sticks-per-row dimension (the first device dimension in
    the stick layout, whose coordinate is ``floor(c_col / elems_per_stick)``)
    must keep the full-tensor size so the hardware uses the correct row stride
    when it steps from one row of the tile to the next.

    Concretely: only update ``device_size[d]`` for dimensions whose delta is
    driven solely by row-like tiled symbols — symbols that do NOT appear in
    ``device_coordinates[-1]`` (the within-stick coordinate).  The stick
    column symbol appears in both the sticks-per-row coordinate and the
    within-stick coordinate, so it is excluded from this update.
    """
    assert arg.stride_map is not None, (
        "_tile_device_size: TensorArg has no stride_map — "
        "all TensorArgs reaching the unroller must be constructed with stride_map"
    )
    all_syms: set = set()
    for expr in arg.device_coordinates:
        all_syms |= expr.free_symbols
    sub_zero = {s: 0 for s in all_syms}

    # Symbols that appear in the within-stick coordinate are column/stick
    # symbols; their sticks-per-row dimension encodes the row stride and must
    # not be shrunk to tile size.
    stick_syms: set = arg.device_coordinates[-1].free_symbols

    result = list(arg.device_size)
    for tiled_sym in tiled_syms:
        if tiled_sym not in iteration_space or tiled_sym in stick_syms:
            continue
        tile_range = int(iteration_space[tiled_sym][0])
        sub_range = {**sub_zero, tiled_sym: tile_range}
        for d, coord_expr in enumerate(arg.device_coordinates):
            at_range = int(coord_expr.subs(sub_range))
            at_zero = int(coord_expr.subs(sub_zero))
            delta = at_range - at_zero
            if delta > 0:
                result[d] = delta
    return result


def _arg_byte_strides_for_syms(
    op_spec: OpSpec,
    tiled_syms: list[Symbol],
) -> list[dict[Symbol, int]]:
    """Return per-arg byte strides for the given tiled symbols.

    ``tiled_syms`` is the list of symbols tiled by the enclosing LoopSpec being
    unrolled.  This is passed explicitly rather than read from
    ``op_spec.tiled_symbols`` so that outer-loop unrolling still works after
    inner-loop unrolling has already cleared ``tiled_symbols`` on each copy.
    """
    result: list[dict[Symbol, int]] = []
    for arg in op_spec.args:
        # per_tile_fixed: buffer is a tile-local scratch reused every iteration
        # (determined by insert_tiling_propagation use-def analysis) — fixed.
        # All other args advance per iteration regardless of allocation type
        # (hbm, pool, or lx).
        if arg.per_tile_fixed:
            result.append({})
            continue

        strides: dict[Symbol, int] = {}
        for tiled_sym in tiled_syms:
            if tiled_sym not in op_spec.iteration_space:
                continue
            tile_range = int(op_spec.iteration_space[tiled_sym][0])
            stride = _byte_stride_for_arg(arg, tiled_sym, tile_range)
            if stride != 0:
                strides[tiled_sym] = stride
        result.append(strides)
    return result


def _unroll_one(loop: LoopSpec) -> list:
    """Unroll a single LoopSpec node, returning flat OpSpec copies.

    Nested ``LoopSpec`` nodes are unrolled innermost-first: inner loops are
    fully flattened before the outer loop iterates over them.  Each level
    independently computes per-iteration byte strides from its own
    ``tiled_symbols`` and ``iteration_space``, so strides accumulate
    correctly across nesting depths without explicit offset propagation.
    """
    # --- Recursively unroll any nested LoopSpecs in body first. ----------
    flat_body: list[OpSpec] = []
    for entry in loop.body:
        if isinstance(entry, LoopSpec):
            flat_body.extend(_unroll_one(entry))
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

    # --- Determine which symbols this loop level tiles. ------------------
    # If loop.tiled_symbols is populated (new path), use it directly.
    # Otherwise fall back to reading tiled_symbols from the first OpSpec
    # in the body (works for the single-level non-nested case).
    if loop.tiled_symbols:
        this_level_syms: list[Symbol] = loop.tiled_symbols
    else:
        this_level_syms = next(
            (list(e.tiled_symbols) for e in flat_body if isinstance(e, OpSpec)),
            [],
        )

    # --- Pre-compute per-arg byte strides and tile device_sizes once. ----
    strides_per_op: list[list[dict[Symbol, int]]] = []
    tile_sizes_per_op: list[list[list[int] | None]] = []
    for entry in flat_body:
        if isinstance(entry, OpSpec):
            arg_strides = _arg_byte_strides_for_syms(entry, this_level_syms)
            strides_per_op.append(arg_strides)
            tile_sizes: list[list[int] | None] = []
            for arg, strides in zip(entry.args, arg_strides):
                if strides:
                    tile_sizes.append(
                        _tile_device_size(arg, this_level_syms, entry.iteration_space)
                    )
                else:
                    tile_sizes.append(None)
            tile_sizes_per_op.append(tile_sizes)
        else:
            strides_per_op.append([])
            tile_sizes_per_op.append([])

    # --- Emit count copies, advancing addresses per iteration. -----------
    result: list = []
    for i in range(count):
        for entry, arg_strides, tile_sizes in zip(
            flat_body, strides_per_op, tile_sizes_per_op
        ):
            if not isinstance(entry, OpSpec):
                result.append(copy.deepcopy(entry))
                continue

            op_copy = copy.deepcopy(entry)

            for arg, strides, tile_size in zip(op_copy.args, arg_strides, tile_sizes):
                if not strides:
                    continue
                iter_offset = sum(
                    i * strides[s] for s in this_level_syms if s in strides
                )
                if iter_offset:
                    arg.allocation = dict(arg.allocation)
                    # Advance whichever allocation key is present (hbm, pool, lx).
                    for alloc_key in ("pool", "hbm", "lx"):
                        if alloc_key in arg.allocation:
                            arg.allocation[alloc_key] += iter_offset
                            break
                # Replace device_size with the tile dimensions so the SDSC
                # does not generate a backGap relative to the full tensor.
                if tile_size is not None:
                    arg.device_size = tile_size

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
    body.  For each ``TensorArg`` with ``per_tile_fixed=False`` the base
    address is advanced by the per-arg, per-iteration byte offset derived from
    ``device_coordinates`` and ``stride_map`` — so args with different tile
    sizes or layouts each get the correct independent advance regardless of
    allocation type (hbm, pool, or lx).

    ``per_tile_fixed=True`` args are left unchanged.  ``tiled_symbols`` is
    cleared on every copy so ``generate_bundle`` treats the ops as plain
    non-tiled entries.

    ``count`` must be a concrete integer expression; symbolic counts raise
    ``ValueError``.  Nested ``LoopSpec`` nodes are unrolled innermost-first.
    """
    result: list = []
    for entry in specs:
        if isinstance(entry, LoopSpec):
            result.extend(_unroll_one(entry))
        else:
            result.append(entry)
    return result
