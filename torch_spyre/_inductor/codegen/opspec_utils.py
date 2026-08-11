# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""OpSpec-reading helpers.

The KTIR emitter consumes the *finished* ``OpSpec`` list and must agree on the
"decision / arithmetic" it implies: buffer identity, row-major strides,
iteration-space grouping, and reshape / broadcast alignment of pointwise
operands.  Those computations are **pure** (sympy / int over the ``op_specs``)
-- no backend emission primitives, no MLIR builder, no live Inductor kernel
state -- so they live here as plain functions rather than as base-class methods.

They are therefore **emitter-agnostic**: they read the ``OpSpec`` contract
without emitting any target IR, so one set can serve every OpSpec consumer.  The
KTIR emitter is simply the first such consumer (hence the KTIR references
below); future emitters are intended to share them, which is why the module is
named for the ``OpSpec`` it reads rather than for KTIR.
"""

from __future__ import annotations

import sympy
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _row_major_strides(device_size: list[int]) -> list[int]:
    """Row-major (C-contiguous) strides for a device-size list."""
    n = len(device_size)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * int(device_size[i + 1])
    return strides


def _buf_id(arg: TensorArg) -> str:
    """Stable identity of the buffer an op arg refers to, for register threading.

    Keys on the op-spec ``name`` (the buffer name): unique per buffer and
    identical whether the buffer appears as an input or an output, so a
    fused-away intermediate threads its register value without aliasing.
    ``arg_index`` cannot serve as the identity -- distinct fused-away
    intermediates all carry the unassigned sentinel ``-1``.

    ``name`` must therefore be populated on every projected op arg (see
    ``create_tensor_arg``).  A ``None`` name means an unnamed arg reached
    projection, which would silently alias on ``-1``; raise loudly instead of
    falling back.
    """
    if arg.name is None:
        # ValueError (not NotImplementedError): under TORCH_SPYRE_KTIR=1
        # create_tensor_arg always populates the name, so a None here is a
        # broken internal invariant, not an unsupported-capability case.
        raise ValueError(
            "_buf_id: TensorArg.name is None -- every projected op arg must "
            "carry a buffer name for register-threading identity (arg_index is "
            "-1 for fused intermediates and cannot disambiguate them)"
        )
    return arg.name


def _iteration_space_key(spec: OpSpec) -> tuple:
    """Hashable canonical form of ``spec.iteration_space`` for grouping.

    Two ops fuse iff this key matches: same symbols, same ranges, and same
    work divisions.  Symbols/ranges are compared by their string form so the
    key is order-independent and hashable.
    """
    return tuple(
        sorted(
            (str(sym), str(rng), int(div))
            for sym, (rng, div) in spec.iteration_space.items()
        )
    )


# Device-dim coordinate kinds, used to align pointwise operands whose device
# axes are ordered differently from the op's output tile (broadcast alignment).
_DIM_CONST = "const"  # no iteration-space symbol (an inserted / broadcast dim)
_DIM_BARE = "bare"  # coord == sym (a non-stick dim)
_DIM_WITHIN_STICK = "within_stick"  # coord == sym % stick  (within-stick lanes)
_DIM_OUTER_STICK = "outer_stick"  # coord == sym // stick (outer-stick chunks)


def _dim_info(coord: sympy.Expr) -> tuple[str, sympy.Symbol | None]:
    """Classify a device-dim coordinate as ``(kind, sym)``.

    Two device axes are the "same" logical dim iff they share a ``(kind, sym)``
    -- e.g. a weight's ``sym // stick`` outer-stick axis matches an output's
    ``sym // stick`` axis even when they sit at different positions.  A ``const``
    dim (extent 1, no symbol) carries no data and is dropped / broadcast.

    A coordinate carrying more than one iteration symbol (a single physical axis
    that folds two logical dims, e.g. ``a*8 + b``) is not a simple device axis.
    No legal reshape produces one today -- a plain reshape is a pure view whose
    strides stay stick-aligned (single symbol per axis), and a within-stick fold
    is rejected earlier at layout selection (the within-stick dim must be a full
    64-element stick).  Raise loudly rather than carrying a dead classification:
    if this fires, a new frontend construct reached alignment and both this
    helper and ``_align_reshape_plan`` need real multi-symbol support.
    """
    syms = coord.free_symbols
    if not syms:
        return (_DIM_CONST, None)
    if len(syms) > 1:
        raise NotImplementedError(
            f"OpSpec alignment: device coordinate {coord!r} folds multiple "
            f"iteration symbols {syms} into one physical axis; no supported "
            "reshape produces this, so multi-symbol alignment is not implemented"
        )
    sym = next(iter(syms))
    if isinstance(coord, sympy.Symbol):
        return (_DIM_BARE, sym)
    if isinstance(coord, (sympy.Mod, ModularIndexing)):
        return (_DIM_WITHIN_STICK, sym)
    if isinstance(coord, FloorDiv):
        return (_DIM_OUTER_STICK, sym)
    # A single-symbol coordinate that is none of the known device-axis forms
    # (e.g. ``2*d0 + 1``) is not a plain outer-stick chunk index.  Raise loudly
    # rather than silently classifying it as outer-stick -- same policy as the
    # multi-symbol case above; a later increment that legitimately produces such
    # a coordinate must extend this classifier explicitly.
    raise NotImplementedError(
        f"OpSpec alignment: single-symbol device coordinate {coord!r} is not a "
        "bare symbol, within-stick (Mod/ModularIndexing), or outer-stick "
        "(FloorDiv) form; classification is not implemented"
    )


def _align_reshape_plan(
    in_coords: list[sympy.Expr],
    in_block: list[int],
    out_coords: list[sympy.Expr],
    out_block: list[int],
) -> tuple[list[int], list[int] | None] | None:
    """Plan to reshape + broadcast a pointwise operand tile into the op's output
    device-axis order.

    A pointwise op's operands may carry their device axes in a different order
    (and rank) from the output tile: a per-row reduction result broadcasts over
    the outer-stick dim, a channel weight broadcasts over the row dim, etc.
    Elementwise broadcast typically only auto-aligns *leading* unit dims, so a
    misaligned operand (outer-stick where the output has rows) must be reshaped
    to the output order first.

    Each output device axis is matched to an input axis by ``(kind, sym)`` (see
    ``_dim_info``); the within-stick axis maps last -> last.  Unmatched output
    axes get extent 1 (to be broadcast).  Returns ``(reshape_to, broadcast_to)``
    -- reshape the operand to ``reshape_to`` (skip if it already equals
    ``in_block``) then broadcast to ``broadcast_to`` (``None`` to skip).
    Returns ``None`` when the operand already matches the output (fast path:
    no reshape / broadcast emitted, keeping simple kernels byte-identical).

    Raises ``NotImplementedError`` for a cross-stick transpose (an input axis
    with extent > 1 that no output axis matches, or matched axes that would need
    a permute) -- that needs a restickify, not a reshape.
    """
    in_block = [int(b) for b in in_block]
    out_block = [int(b) for b in out_block]
    if list(in_coords) == list(out_coords) and in_block == out_block:
        return None

    in_info = [_dim_info(c) for c in in_coords]
    in_rank = len(in_coords)
    out_rank = len(out_coords)

    reshape_to = [1] * out_rank
    used: set[int] = set()
    # Within-stick lanes: the last input axis always maps to the last output axis.
    reshape_to[out_rank - 1] = in_block[in_rank - 1]
    used.add(in_rank - 1)
    matched_seq: list[int] = []  # input axes matched to output axes, in out order
    for o in range(out_rank - 1):
        okind, osym = _dim_info(out_coords[o])
        if okind == _DIM_CONST:
            continue
        for a in range(in_rank - 1):
            if a in used:
                continue
            if in_info[a] == (okind, osym):
                used.add(a)
                reshape_to[o] = in_block[a]
                matched_seq.append(a)
                break

    # A pure reshape preserves the row-major element order: the matched input
    # axes must appear in increasing order.  If not, the operand would need a
    # transpose (restickify) to align -- not supported on this path.
    if any(matched_seq[i] >= matched_seq[i + 1] for i in range(len(matched_seq) - 1)):
        raise NotImplementedError(
            "OpSpec alignment: pointwise operand needs a transpose (restickify) "
            "to align device axes; not supported yet"
        )
    prod_in = 1
    for b in in_block:
        prod_in *= b
    prod_reshape = 1
    for b in reshape_to:
        prod_reshape *= b
    if prod_in != prod_reshape:
        # An input axis with extent > 1 was dropped -> real data would be lost;
        # aligning it needs a cross-stick transpose (restickify).
        raise NotImplementedError(
            "OpSpec alignment: pointwise operand needs a cross-stick transpose "
            "(restickify) to align device axes; not supported yet"
        )
    broadcast_to = out_block if reshape_to != out_block else None
    return (reshape_to, broadcast_to)
