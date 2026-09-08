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

"""Spyre-specific decompositions and PrivateUse1 dispatch-key kernels.

There is exactly one public entry point: ``register_spyre_decompositions``.
It records a decomposition for ``torch.compile`` (consumed via Inductor's
``get_decomp_fn``); for aten ops, a PrivateUse1 kernel reaching the same
implementation is auto-installed at runtime init so eager-mode dispatch
reaches it too. Eager-only registration is intentionally not exposed.

The Spyre decomposition table built by ``get_spyre_decomp_table`` is
independent from PyTorch's global ``torch._inductor.decomposition.decompositions``
registry; Spyre never mutates the global table.
"""

import math
import threading
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch._decomp as decomp

from .constants import DEVICE_NAME, FP8_E4M3FN_MAX, FP8_E4M3FN_MIN
from .errors import Unsupported
from .sliding_window_plan import (
    SlidingWindowPlan,
    band_valid_start,
    check_valid_start,
    check_window_read,
    plan_sliding_window,
    query_blocking,
    rejection_reason,
)
from . import config
from .logging_utils import get_inductor_logger

from . import customops  # noqa: F401
from . import spyre_hint
from torch_spyre._C import DataFormats, get_device_dtype, get_elem_in_stick
import torch_spyre._inductor.customops  # noqa: F401

logger = get_inductor_logger("decompositions")


_SDPA_MAX_SEQUENCE_TILE_SIZE = 512
_SDPA_MAX_TILE_PAIRS_PER_LOOP_GROUP = 16
_SDPA_PREFERRED_HEADS_PER_TILE = (4, 2, 1)


def _sdpa_num_head_tiles(num_heads: int) -> int:
    """Use at most four heads per tile and return the required tile count."""
    for heads_per_tile in _SDPA_PREFERRED_HEADS_PER_TILE:
        if num_heads % heads_per_tile == 0:
            return num_heads // heads_per_tile
    return 1


def _num_tiles_for_max_extent(sequence_length: int, max_extent: int) -> int:
    """Return an exact split count whose tile extent is at most ``max_extent``.

    Coarse tiling currently requires equal-sized tiles, so a simple ceiling is
    insufficient when it does not divide ``sequence_length``. Start with the
    minimum count that satisfies the extent cap and advance to the next exact
    divisor. Sequence lengths used by the adapters are stick-padded, so this
    normally resolves after only a few candidates.
    """
    num_tiles = max(1, (sequence_length + max_extent - 1) // max_extent)
    while sequence_length % num_tiles != 0:
        num_tiles += 1
    return num_tiles


def _kv_blocks_per_loop_group(num_q_tiles: int, num_kv_blocks: int) -> int:
    """Keep each SDPA backend bundle near the proven 4-by-4 size.

    DXP specializes a counted Lq loop across every unrolled Lk block.  Bundle
    code size therefore scales with their product, not with the number of Lk
    blocks alone.  Cap that product at sixteen while retaining at least one Lk
    block per group.  Thus 8K remains one 4-by-4 group, while 32K becomes
    sixteen 16-by-1 groups.
    """
    return min(
        num_kv_blocks,
        max(1, _SDPA_MAX_TILE_PAIRS_PER_LOOP_GROUP // num_q_tiles),
    )


# Determine the float dtype for bool at module load time (not during tracing)
_BOOL_FLOAT_DTYPE = None


def _get_float_dtype_for_bool() -> torch.dtype:
    """
    Get the appropriate float dtype to convert boolean tensors on Spyre.
    Boolean tensors are stored as either FP16 or FP32 on the device.
    This is determined once at module load time to avoid tracing issues.
    """
    global _BOOL_FLOAT_DTYPE
    if _BOOL_FLOAT_DTYPE is None:
        device_dtype = get_device_dtype(torch.bool)
        # Map DataFormats to torch.dtype, defaulting to float16
        if device_dtype == DataFormats.IEEE_FP32:
            _BOOL_FLOAT_DTYPE = torch.float32
        else:
            _BOOL_FLOAT_DTYPE = torch.float16
    return _BOOL_FLOAT_DTYPE


# A module-level lock to make the CM thread-safe
_decompositions_lock = threading.RLock()

# Spyre-specific decompositions, populated by ``@register_spyre_decompositions``.
spyre_decompositions: dict = {}

# Inductor default decompositions to drop on Spyre. They produce code the
# backend cannot lower today; falling through to the CPU fallback is preferable
# until those issues are fixed.
spyre_decompositions_to_exclude = [
    torch.ops.aten.triu,
    torch.ops.aten.tril,
    # PT 2.12 broadened torch._inductor.decomposition.mm/bmm to decompose the
    # K==1 (unit-contraction) case into a broadcast ``self * other`` on all
    # non-cpu/mps devices. That AOT decomposition runs before Spyre's
    # ``mm_to_bmm_pass`` and produces a flatten-mul-unflatten shape whose
    # trailing view yields an unsupported ``floor(d0/N)`` stick expression.
    # Spyre has its own aten.mm / aten.bmm lowerings, so drop the upstream
    # decomps and let the mm/bmm survive to mm_to_bmm_pass (2.11 behavior).
    torch.ops.aten.mm,
    torch.ops.aten.bmm,
]

OpOrOps = Union[torch._ops.OperatorBase, Sequence[torch._ops.OperatorBase]]

# Module-level Library handles, kept alive for the lifetime of the process.
# ``torch.library.Library`` uses ``weakref.finalize`` to call ``m.reset()`` on
# GC, which would silently unregister every kernel from the C++ dispatcher.
_spyre_autograd_lib = None
_spyre_lib = None
_dispatchkey_kernels_registered = False


def register_spyre_decompositions(ops: OpOrOps):
    """Register a Spyre-specific decomposition for one or more operators.

    The function is added to the Spyre decomposition table; Inductor reads it
    via ``get_decomp_fn`` during ``torch.compile`` / ``make_fx``. For aten ops,
    ``_register_spyre_dispatchkey_kernels_permanently`` additionally installs a
    PrivateUse1 kernel pointing at the same function at runtime init, so
    eager-mode dispatch reaches it too. This is required for
    ``CompositeImplicitAutograd`` ops (``rms_norm``, ``layer_norm``, ...); it
    is harmless for the rest.
    """
    return decomp.register_decomposition(ops, spyre_decompositions)


def get_spyre_decomp_table() -> dict[Any, Callable[..., Any]]:
    """Return the decomposition table Inductor sees when compiling for Spyre.

    Builds a fresh dict on each call from ``select_decomp_table()`` (Inductor's
    default, itself cached upstream) plus Spyre additions and exclusions.
    Independent from ``torch._inductor.decomposition.decompositions`` — Spyre
    never mutates the global registry.
    """
    from torch._inductor.decomposition import select_decomp_table
    from torch._ops import OpOverload, OpOverloadPacket
    from torch_spyre.ops.fallbacks import fallback_ops

    table = dict(select_decomp_table())

    def _drop(op):
        if isinstance(op, OpOverloadPacket):
            for overload_name in op.overloads():
                table.pop(getattr(op, overload_name), None)
        elif isinstance(op, OpOverload):
            table.pop(op, None)

    for op in spyre_decompositions_to_exclude:
        _drop(op)
    for op in fallback_ops:
        _drop(op)
    table.update(spyre_decompositions)
    return table


class _OPWrapper:
    """PrivateUse1 kernel that lazily ``torch.compile``-s a Spyre decomposition.

    The first eager call compiles the decomposition (with ``dynamic=False``);
    subsequent eager calls reuse the compiled entry point. When invoked from
    inside an active ``torch.compile`` context, the wrapped function is called
    directly — re-entering ``torch.compile`` would be wrong.
    """

    def __init__(self, fn):
        self._fn = fn
        self._compiled_fn = None

    def __call__(self, *args, **kwargs):
        from torch.utils import _pytree as pytree

        leaves = pytree.tree_leaves(args) + pytree.tree_leaves(kwargs)
        # ``!=`` (not ``is not``) is deliberate: this compares the device *type*
        # string against the ``DEVICE_NAME`` constant, and string equality is a
        # value comparison. ``getattr(..., None)`` yields ``None`` for
        # non-tensors, but the ``isinstance`` guard short-circuits those first.
        if any(
            isinstance(x, torch.Tensor)
            and getattr(x.device, "type", None) != DEVICE_NAME
            for x in leaves
        ):
            devs = [x.device if isinstance(x, torch.Tensor) else None for x in leaves]
            raise RuntimeError(
                f"Spyre decomposition function called with inputs on a different "
                f"device! Args devices: {devs=}"
            )
        if torch.compiler.is_compiling():
            return self._fn(*args, **kwargs)
        if self._compiled_fn is None:
            self._compiled_fn = torch.compile(self._fn, dynamic=False)
        return self._compiled_fn(*args, **kwargs)


def _register_spyre_dispatchkey_kernels_permanently():
    """Install PrivateUse1 / AutogradPrivateUse1 kernels for every aten op
    that has a Spyre decomposition and no pre-existing PrivateUse1 kernel.

    Idempotent; called from ``_SpyreImpl._lazy_init`` after eager ops and
    custom ops have been imported, so the existing-kernel check sees the final
    set of registered backends.
    """
    global _spyre_autograd_lib, _spyre_lib, _dispatchkey_kernels_registered

    if _dispatchkey_kernels_registered:
        return

    from torch.library import Library, fallthrough_kernel

    _spyre_autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
    _spyre_lib = Library("aten", "IMPL", "PrivateUse1")
    has_pu1 = torch._C._dispatch_has_kernel_for_dispatch_key

    for op, fn in spyre_decompositions.items():
        if op.namespace != "aten" or has_pu1(op._name, "PrivateUse1"):
            continue
        # Autograd key: fall through so PrivateUse1 is reached.
        _spyre_autograd_lib.impl(op._name, fallthrough_kernel)
        # PrivateUse1 key: dispatch into a lazy-compile wrapper.
        _spyre_lib.impl(op._name, _OPWrapper(fn))

    _dispatchkey_kernels_registered = True


###############################################################################
##                       Spyre decompositions                                ##
###############################################################################


@register_spyre_decompositions([torch.ops.aten.ones.default])
def ones_decomp(
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.aten.full(size, 1, dtype=dtype, layout=layout, device=device)


@register_spyre_decompositions([torch.ops.aten.new_ones.default])
def new_ones_decomp(
    self: torch.Tensor,
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.aten.full(
        size,
        1,
        dtype=dtype if dtype is not None else self.dtype,
        layout=layout,
        device=device if device is not None else self.device,
    )


@register_spyre_decompositions([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


@register_spyre_decompositions([torch.ops.aten.sign.default])
def spyre_sign(input: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(input)
    return torch.where(
        torch.gt(input, zero),
        torch.ones_like(input),
        torch.where(torch.lt(input, zero), -torch.ones_like(input), zero),
    )


###############################################################################
##                    Spyre decompositions for aten ops                      ##
###############################################################################
# For aten ops, ``register_spyre_decompositions`` automatically installs a
# PrivateUse1 dispatch kernel as well (essential for CIA ops like rms_norm,
# layer_norm; harmless for the rest).
@register_spyre_decompositions([torch.ops.aten.rms_norm.default])
def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_rms_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )

    mean = torch.mean(input * input, dim=-1, keepdim=True)
    rsqrt_inp = torch.rsqrt(mean + eps)
    output = input * rsqrt_inp
    if weight is not None:
        output = output * weight
    return output


@register_spyre_decompositions([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_layer_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )
    # F.layer_norm treats weight=None as identity and bias=None as zero;
    # spyre.layernormnorm doesn't handle missing args, so substitute defaults.
    if weight is None:
        weight = input.new_ones(normalized_shape)
    if bias is None:
        bias = input.new_zeros(normalized_shape)
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decompositions([torch.ops.aten.silu.default])
def silu(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.silu(input)


@register_spyre_decompositions([torch.ops.aten.topk])
def spyre_topk(
    input: torch.Tensor,
    k: int,
    dim: Optional[int] = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k > 128:
        raise Unsupported(f"topk with k={k} is not supported (max k=128)")
    if not largest:
        raise Unsupported("topk with largest=False")
    # sorted=False is a no-op: our reduction always returns sorted output.
    # Index stays in the input dtype (not int64) all the way out; topkindex's
    # fake reports it so Dynamo traces it with no meta conflict.
    return torch.ops.spyre.topkvalue(input, k, dim), torch.ops.spyre.topkindex(
        input, k, dim
    )


@register_spyre_decompositions([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    return torch.ops.spyre.gelu(input, approximate)


@register_spyre_decompositions([torch.ops.aten.softplus.default])
def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    if beta == 1.0:
        return torch.ops.spyre.softplus(input, beta, threshold)
    # The runtime primitive drops the outer 1/beta factor, so beta == 1 is its
    # only exact path. Scale into it and back out; the threshold branch stays
    # exact because 1 * (beta * x) > threshold is PyTorch's beta * x > threshold.
    # aten accepts beta == 0 and saturates every element to +-inf, so take the
    # reciprocal under IEEE rules rather than letting Python raise here.
    inv_beta = math.copysign(math.inf, beta) if beta == 0.0 else 1.0 / beta
    return torch.ops.spyre.softplus(input * beta, 1.0, threshold) * inv_beta


@register_spyre_decompositions([torch.ops.aten.linear.default])
def spyre_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    weight = weight.transpose(-1, -2)
    while weight.dim() < input.dim():
        weight = torch.unsqueeze(weight, 0)
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out


@register_spyre_decompositions(
    [torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default]
)
def spyre__sdpa_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = query.size(0)
    num_heads = query.size(1)
    num_kvheads = key.size(1)
    max_seqlen_q = query.size(2)
    max_seqlen_kv = key.size(2)
    head_dim = query.size(3)

    scaling_factor = scale
    if scaling_factor is None:
        scaling_factor = 1.0 / math.sqrt(math.sqrt(head_dim))
    else:
        scaling_factor = math.sqrt(scaling_factor)

    if dropout_p > 0.0:
        raise Unsupported("Attention dropout not implemented for Spyre")

    # The named_dims seeds below zip names positionally to each seeded op's
    # PHYSICAL output layout. host_coordinates derives the coord expressions
    # from the op's physical STRIDES, so the tiler's loop-var -> logical-dim
    # mapping follows physical stride order, not logical dim order. SDPA is
    # routinely called with q/k/v as transpose(1, 2) views of a [B, S, H, D]
    # tensor, so query's physical layout is [B, S, H, D] while its logical shape
    # is [B, H, S, D]. Seeding ["_b","num_heads","max_seqlen_q",...] on such
    # a buffer then maps the tile onto the head axis instead of the query
    # axis and the result is wrong.
    #
    # Normalize the QUERY wholesale: query.contiguous() is bounded (same
    # footprint as the `output` buffer we allocate below) and makes every
    # query-derived seed (q_scaled, zeros_like(query), the final permute) land
    # on logical [B, H, S, D] order. We do NOT contiguify key/value wholesale --
    # that materializes a full [B, H, S_kv, D] copy that OOMs on long KV. K/V are
    # instead normalized PER BLOCK inside the loop (keys_T's .contiguous() and
    # the per-block v_blk.contiguous()), each a bounded [B, H, kv_block_size, D]
    # copy (kv_block_size <= 2048, see below), never the full [B, H, S_kv, D].
    query = query.contiguous()

    expansion = num_heads // num_kvheads
    if expansion != 1:
        key = key.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        value = value.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)

    # Keep the original approximately four-way KV split for short sequences,
    # but cap each explicit online-softmax block at the max tile size. Round
    # the short-case target up to a 64-element fp16 stick as before.
    quarter_kv_stick_aligned = max(64, ((max_seqlen_kv + 3) // 4 + 63) // 64 * 64)
    kv_block_size = min(_SDPA_MAX_SEQUENCE_TILE_SIZE, quarter_kv_stick_aligned)
    num_kv_blocks = (max_seqlen_kv + kv_block_size - 1) // kv_block_size

    # Lq uses equal-sized WSR coarse tiles. Select the smallest exact split
    # count whose per-tile extent is at most _SDPA_MAX_SEQUENCE_TILE_SIZE.
    num_q_tiles = _num_tiles_for_max_extent(max_seqlen_q, _SDPA_MAX_SEQUENCE_TILE_SIZE)
    q_tile_size = max_seqlen_q // num_q_tiles
    kv_blocks_per_loop_group = _kv_blocks_per_loop_group(num_q_tiles, num_kv_blocks)
    logger.debug(
        "SDPA sequence tiling: Lq=%s q_tiles=%s q_tile_size=%s "
        "Lk=%s kv_blocks=%s kv_block_size=%s kv_blocks_per_loop_group=%s",
        max_seqlen_q,
        num_q_tiles,
        q_tile_size,
        max_seqlen_kv,
        num_kv_blocks,
        kv_block_size,
        kv_blocks_per_loop_group,
    )

    with spyre_hint(named_dims=["_b", "num_heads", "max_seqlen_q", "head_dim"]):
        output = torch.zeros_like(query)

    # FIXME: create a sparse M tensor via reduction
    M_reduced = torch.full(
        (batch_size, num_heads, max_seqlen_q, 64),
        float("-inf"),
        device=query.device,
        dtype=query.dtype,
    )
    with spyre_hint(named_dims=["_b", "num_heads", "max_seqlen_q"]):
        M = M_reduced.amax(dim=-1)  # batch_size, num_heads, max_seqlen_q sparse

    # FIXME: create a sparse denominator tensor via reduction
    denominator_reduced = torch.zeros(
        (batch_size, num_heads, max_seqlen_q, 64),
        device=query.device,
        dtype=query.dtype,
    )
    with spyre_hint(named_dims=["_b", "num_heads", "max_seqlen_q"]):
        denominator = denominator_reduced.amax(
            dim=-1
        )  # batch_size, num_heads, max_seqlen_q sparse

    # Precompute the causal additive mask once before entering the tiled loops.
    # Shape [1, 1, max_seqlen_q, max_seqlen_kv]: 0.0 = keep, -inf = masked.
    #
    # spyre::causal_mask builds the mask on CPU (tril + masked_fill_) and
    # transfers it to the query device. Wrapping this in a custom op makes the
    # CPU-side in-place ops opaque to torch.compile, so assert_functional_graph
    # is satisfied and the compiled graph sees only the resulting Spyre tensor.
    if is_causal:
        causal_mask = torch.ops.spyre.causal_mask(
            max_seqlen_q, max_seqlen_kv, query.dtype, query.device
        )

    # Seed named dimensions on the accumulators (above) and the scaled query so
    # the max_seqlen_q tiling hint below has a propagated named dim to bind to.
    # Without a seed the spyre_hint(tiles={"max_seqlen_q": ...}) scope is a
    # no-op: assign_dim_hints drops any tile whose named dim never propagated.
    # The names zip positionally to the query layout
    # [batch_size, num_heads, max_seqlen_q, head_dim]; from this producer the
    # names flow automatically to every downstream pointwise/reduction op.
    #
    # The head dim is named "num_heads" so it matches the
    # tiles={"num_heads": ...} scope below and the head tile activates. The
    # batch dim is still the placeholder "_b" -- it does NOT match
    # tiles={"batch_size": ...}, so the batch tile stays inactive for now.
    # Renaming "_b" -> "batch_size" is all it takes to light up the batch tile
    # in a follow-up.
    with spyre_hint(named_dims=["_b", "num_heads", "max_seqlen_q", "head_dim"]):
        q_scaled = query * scaling_factor

    # Bound each loop group's Lq-tile x unrolled-Lk-block product.  Keeping all
    # sixteen 32K blocks together creates a 322-SDSC bundle that crashes DXP;
    # grouping four at a time still produces ~48 MB binaries that crash the
    # runtime H2D launch because the Lq loop itself has sixteen trips.  A
    # sixteen-pair budget preserves the proven 8K 4x4 bundle and makes 32K use
    # sixteen 16x1 bundles of approximately the same code size.  The functional
    # M/denominator/output SSA carries are materialized between groups and then
    # resume the exact same online-softmax recurrence.
    for block_group_start in range(0, num_kv_blocks, kv_blocks_per_loop_group):
        block_group_end = min(
            block_group_start + kv_blocks_per_loop_group, num_kv_blocks
        )
        with spyre_hint(tiles={"batch_size": max(1, batch_size // 2)}):
            with spyre_hint(tiles={"num_heads": _sdpa_num_head_tiles(num_heads)}):
                with spyre_hint(num_tiles_per_dim={"max_seqlen_q": num_q_tiles}):
                    for blk in range(block_group_start, block_group_end):
                        start = blk * kv_block_size
                        end = min(start + kv_block_size, max_seqlen_kv)

                        k_blk = key[
                            ..., start:end, :
                        ]  # batch_size, num_heads, blk_len, head_dim
                        v_blk = value[
                            ..., start:end, :
                        ]  # batch_size, num_heads, blk_len, head_dim

                        # The K/V slices are produced in-graph, so they carry no
                        # named dims and stay untiled -- forming a restickify
                        # boundary against the tiled matmuls that consume them.
                        # Name each slice's first consuming op so the per-chunk
                        # key extent ("blk_len") propagates and the producers tile
                        # with their consumers.
                        with spyre_hint(
                            named_dims=["_b", "num_heads", "blk_len", "head_dim"]
                        ):
                            scaled_keys = k_blk * scaling_factor
                        # v_blk feeds the second matmul directly (line below), so
                        # unlike scaled_keys (re-normalized by keys_T.contiguous())
                        # it has no downstream .contiguous() to fix its layout. For
                        # a transposed value view the slice's physical strides stay
                        # transposed (a pointwise op preserves its input's stride
                        # order via pick_loop_order), which scrambles the tiled
                        # matmul. .contiguous() lowers to aten.clone(contiguous),
                        # which the Spyre clone override freezes to contiguous
                        # strides -- normalizing this [B, H, blk_len, D] slice per
                        # block without a full-tensor value.contiguous() (an OOM on
                        # long KV). The named_dims seed lands on the resulting clone, so
                        # blk_len still propagates.
                        with spyre_hint(
                            named_dims=["_b", "num_heads", "blk_len", "head_dim"]
                        ):
                            v_blk = v_blk.contiguous()
                        # .contiguous() materializes the transposed keys so the
                        # scores matmul sees a clean single-contraction-dim input.
                        # Without it the backend scheduler aborts with
                        # out_reuse_dim.size() == 1 (L3DlOpsScheduler): the
                        # transposed view's loop-dim-order leaves the matmul with
                        # an ambiguous contraction dim under Lq tiling.
                        keys_T = scaled_keys.transpose(
                            -1, -2
                        ).contiguous()  # batch_size, num_heads, head_dim, blk_len
                        # A matmul output inherits no named dims from its inputs,
                        # so naming it explicitly is what keeps the op inside the
                        # max_seqlen_q tile region (otherwise it becomes an
                        # untiled restickify boundary). "blk_len" is the per-chunk
                        # key extent -- the scores' last axis.
                        with spyre_hint(
                            named_dims=["_b", "num_heads", "max_seqlen_q", "blk_len"]
                        ):
                            scores = torch.matmul(
                                q_scaled, keys_T
                            )  # batch_size, num_heads, max_seqlen_q, blk_len

                        if is_causal:
                            scores = scores + causal_mask[..., :, start:end]

                        if attn_bias is not None:
                            scores = scores + attn_bias[..., :, start:end]

                        block_max = torch.amax(
                            scores, dim=-1
                        )  # batch_size, num_heads, max_seqlen_q sparse
                        new_max = torch.maximum(
                            M, block_max
                        )  # batch_size, num_heads, max_seqlen_q sparse

                        exp_scores = torch.exp(
                            scores - new_max.unsqueeze(-1)
                        )  # batch_size, num_heads, max_seqlen_q, blk_len
                        correction = torch.exp(
                            M - new_max
                        )  # batch_size, num_heads, max_seqlen_q sparse

                        # Online-softmax recurrence as FUNCTIONAL SSA -- no
                        # in-place copy_f writeback. copy_f mutates the whole
                        # accumulator buffer and is NOT tile-aware: under the
                        # max_seqlen_q tile it left the second query-tile's
                        # accumulators un-updated (0/0 -> nan, exp(-inf) -> inf).
                        # Threading new values forward (as in the coarse-tile
                        # flash e2e test, PR #3674) keeps each Lq tile's carry
                        # correct. The names flow from q_scaled/the matmuls.
                        new_denom = denominator * correction + exp_scores.sum(
                            dim=-1
                        )  # batch_size, num_heads, max_seqlen_q sparse
                        # Materialize exp_scores before the second matmul for the
                        # same reason as keys_T above -- a clean contiguous input
                        # keeps the matmul's contraction dim unambiguous.
                        exp_scores_c = exp_scores.contiguous()
                        with spyre_hint(
                            named_dims=["_b", "num_heads", "max_seqlen_q", "head_dim"]
                        ):
                            weighted = torch.matmul(exp_scores_c, v_blk)
                        new_output = (
                            output * correction.unsqueeze(-1) + weighted
                        )  # batch_size, num_heads, max_seqlen_q, head_dim

                        if blk == num_kv_blocks - 1:
                            # The final divide must live INSIDE the innermost tile
                            # scope (#3674 point #4): read past the loop group it
                            # becomes a full untiled buffer whose input
                            # (new_output, written in the tiled region) is
                            # split-layout, so finalize_layouts hits
                            # restickify-infeasible. Fold it into the last KV
                            # block so it inherits the max_seqlen_q tile.
                            with spyre_hint(
                                named_dims=[
                                    "_b",
                                    "num_heads",
                                    "max_seqlen_q",
                                    "head_dim",
                                ]
                            ):
                                output = new_output / new_denom.unsqueeze(-1)
                        else:
                            M, denominator, output = new_max, new_denom, new_output
    # The reference meta kernel for this op
    # (torch._meta_registrations.meta__scaled_dot_product_fused_attention_
    # overrideable -> alloc_with_matching_layout) declares the output layout to
    # MATCH THE QUERY's layout, not a fixed [B, S, H, D]-contiguous physical
    # layout. Inductor emits assert_size_stride against those meta strides, so the
    # decomp output must carry the same strides as ``query``. For a contiguous
    # query this is plain [B, H, S, D]-contiguous; for a transposed query
    # (physical [B, S, H, D]) it is the swapped-dim layout. Reproduce the meta's
    # dim-order permutation so both cases match exactly.
    dim_order = sorted(
        range(query.dim()), key=lambda i: query.stride()[i], reverse=True
    )
    permuted = output.permute(dim_order).contiguous()
    inverse_permute = [dim_order.index(i) for i in range(len(dim_order))]
    output = permuted.permute(inverse_permute)
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device="spyre"
    )
    philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
    philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

    return (
        output,
        logsumexp,
        None,
        None,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        None,
    )


@register_spyre_decompositions([torch.ops.spyre.kv_window.default])
def spyre_kv_window(
    key: torch.Tensor,
    value: torch.Tensor,
    read_start: int,
    buffer_width: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Q block's KV window: k_win [B, Hq, E, W] transposed, v_win [B, Hq, W, E]."""
    reason = check_window_read(
        read_start=read_start,
        buffer_width=buffer_width,
        cache_capacity=key.size(2),
        num_heads=num_heads,
        num_kv_heads=key.size(1),
        key_shape=tuple(key.shape),
        value_shape=tuple(value.shape),
    )
    if reason is not None:
        raise Unsupported(f"kv_window: {reason}")

    # Slice before expanding: expanding first and slicing after breaks the
    # stick-padding pass ("lower_pad_sequence: pad_extent=-129").
    k_win = key[:, :, read_start : read_start + buffer_width, :].transpose(-1, -2)
    v_win = value[:, :, read_start : read_start + buffer_width, :]

    expansion = num_heads // key.size(1)
    if expansion != 1:
        k_win = k_win.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        v_win = v_win.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
    return k_win, v_win


def _windowed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SlidingWindowPlan,
    scaling_factor: float,
    num_heads: int,
    valid_start: list[int] | None = None,
) -> torch.Tensor:
    """spyre__sdpa_overrideable's body, tiled over buffer_width not max_seqlen_kv.

    The Python loop is unrolled at trace time, so every read offset is a
    constant. Blocks share no softmax state; the accumulators are per
    iteration and exist because the window_size hint may tile buffer_width
    into partial softmaxes.
    """
    batch_size = query.size(0)
    head_dim = query.size(3)
    q_block = plan.q_block
    buffer_width = plan.buffer_width

    out_blocks = []
    for block_index in range(plan.num_q_blocks):
        q_start, q_end = plan.block_q_range(block_index)
        assert q_end - q_start == q_block, (
            f"sliding_window_attention: Q block {block_index} is "
            f"{q_end - q_start} rows, expected {q_block} -- plan_sliding_window "
            "should have rejected a query length that does not divide"
        )

        read_start = plan.read_start(block_index)
        k_win, v_win = torch.ops.spyre.kv_window(
            key, value, read_start, buffer_width, num_heads
        )
        # A valid_start that masks anything makes the band load-bearing even for a
        # block the window alone fully covers.
        fully_attended = (
            plan.block_is_fully_attended(block_index)
            and band_valid_start(valid_start) is None
        )
        band = (
            None
            if fully_attended
            else torch.ops.spyre.window_band_mask(
                # Logical, not read_start: the row side of this op
                # (q_row_origin) is a logical coordinate, and delta = row -
                # column only means anything if both sides agree. Identical
                # to read_start while buffer_origin is 0 -- a still-filling
                # or exactly-full cache -- and diverges for a rolled buffer.
                plan.read_start_logical(block_index),
                q_block,
                buffer_width,
                plan.q_kv_offset + q_start,
                plan.window_size,
                plan.is_causal,
                query.dtype,
                query.device,
                valid_start,
            )
        )
        q_rows = query[:, :, q_start:q_end, :]

        # SDPA's "sparse via reduction" construction. On a single pass M is
        # -inf, so correction is exp(-inf) == 0 and the running terms drop out.
        m_reduced = torch.full(
            (batch_size, num_heads, q_block, 64),
            float("-inf"),
            device=query.device,
            dtype=query.dtype,
        )
        running_max = m_reduced.amax(dim=-1)

        denominator_reduced = torch.zeros(
            (batch_size, num_heads, q_block, 64),
            device=query.device,
            dtype=query.dtype,
        )
        denominator = denominator_reduced.amax(dim=-1)

        output = torch.zeros(
            (batch_size, num_heads, q_block, head_dim),
            device=query.device,
            dtype=query.dtype,
        )

        with spyre_hint(tiles={"batch_size": max(1, batch_size // 2)}):
            with spyre_hint(tiles={"num_heads": max(1, num_heads // 4)}):
                with spyre_hint(tiles={"window_size": max(1, buffer_width // 64)}):
                    with spyre_hint(work_div={"num_heads": 4, "window_size": 8}):
                        # k_win arrives transposed.
                        scores = torch.matmul(
                            q_rows * scaling_factor, k_win * scaling_factor
                        )  # batch, num_heads, q_block, buffer_width

                        if band is not None:
                            scores = scores + band

                        block_max = torch.amax(scores, dim=-1)
                        max_running = torch.maximum(running_max, block_max)

                        exp_scores = torch.exp(scores - max_running.unsqueeze(-1))
                        correction = torch.exp(running_max - max_running)

                        denominator = torch.ops.spyre.copy_forced(
                            denominator * correction + exp_scores.sum(dim=-1),
                            denominator,
                        )
                        output = torch.ops.spyre.copy_forced(
                            output * correction.unsqueeze(-1)
                            + torch.matmul(exp_scores, v_win),
                            output,
                        )
                        running_max = torch.ops.spyre.copy_forced(
                            max_running, running_max
                        )

        out_blocks.append(
            torch.ops.spyre.copy_forced(output / denominator.unsqueeze(-1), output)
        )

    return torch.cat(out_blocks, dim=2)


@register_spyre_decompositions([torch.ops.spyre.sliding_window_attention.default])
def spyre_sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    is_causal: bool = True,
    scale: float | None = None,
    cache_seqlen: int | None = None,
    buffer_origin: int | None = None,
    valid_start: list[int] | None = None,
) -> torch.Tensor:
    """Sliding-window attention: each Q block attends only its own KV slice.

    Rather than scoring the full cache behind a band mask, each block of
    q_block query rows reads buffer_width rows -- a constant that does not
    grow with the cache -- and attends against those. See
    ``plan_sliding_window`` for the placement.

    A ragged query length is padded up rather than refused. Shapes the
    placement cannot express raise: there is no slower-but-correct fallback,
    since a band mask over full attention is itself wrong for an unaligned KV
    length.
    """
    num_heads = query.size(1)
    head_dim = query.size(3)
    batch_size = query.size(0)
    seqlen_q = query.size(2)
    cache_capacity = key.size(2)

    # The cache's position, not its allocation. None means "exactly full",
    # which is what reading key.size(2) as a position silently assumed.
    # Degenerate values (<= 0, either of them) need no check here:
    # rejection_reason below rejects them, and it is the single source of
    # truth for which geometries can be planned.
    if cache_seqlen is None:
        cache_seqlen = cache_capacity

    reason = check_valid_start(valid_start, batch_size, cache_seqlen)
    if reason is not None:
        raise Unsupported(f"sliding_window_attention: {reason}")

    if scale is not None and scale < 0:
        # math.sqrt would otherwise raise a bare ValueError.
        raise Unsupported(
            f"sliding_window_attention: scale={scale} must be non-negative"
        )

    # Split across query and key, not applied once to their product: keeps the
    # intermediate in float16 range.
    if scale is None:
        scaling_factor = 1.0 / math.sqrt(math.sqrt(head_dim))
    else:
        scaling_factor = math.sqrt(scale)

    # Pad at the FRONT: row i sits at coordinate seqlen_kv - seqlen_q + i, so
    # once seqlen_q is the padded length the two shifts cancel and every real
    # row keeps its coordinate. Back-padding would move all of them.
    q_block, padded_seqlen_q = query_blocking(seqlen_q)
    pad_rows = padded_seqlen_q - seqlen_q
    plan = plan_sliding_window(
        padded_seqlen_q,
        cache_seqlen,
        window_size,
        is_causal=is_causal,
        q_block=q_block,
        cache_capacity=cache_capacity,
        buffer_origin=buffer_origin,
    )
    if plan is None:
        # Never None when the plan is, but the type says otherwise.
        reason = (
            rejection_reason(
                padded_seqlen_q,
                cache_seqlen,
                window_size,
                is_causal,
                q_block,
                cache_capacity,
                buffer_origin,
            )
            or "the window placement cannot express this shape"
        )
        raise Unsupported(f"sliding_window_attention: {reason}")

    if pad_rows:
        query = torch.cat(
            [
                torch.zeros(
                    (batch_size, num_heads, pad_rows, head_dim),
                    device=query.device,
                    dtype=query.dtype,
                ),
                query,
            ],
            dim=2,
        )

    output = _windowed_attention(
        query, key, value, plan, scaling_factor, num_heads, valid_start
    )
    return output[:, :, pad_rows:, :] if pad_rows else output


@register_spyre_decompositions([torch.ops.aten.max.default])
def spyre_max_default_decomp(input):
    """
    Decompose torch.max(input) with conditional CPU fallback for int64.

    For int64 tensors, use custom op spyre::max_default_int64_fallback which has
    a CPU fallback registered in fallbacks.py.
    For other dtypes (float16, float32, etc.), use amax.
    """
    if input.dtype == torch.int64:
        # Use custom op with CPU fallback to avoid recursive decomposition
        # Returns a scalar (0D) tensor
        return torch.ops.spyre.max_default_int64_fallback(input)
    else:
        # Use amax for supported dtypes (can run on Spyre)
        # Returns a scalar (0D) tensor
        return torch.ops.aten.amax(input)


@register_spyre_decompositions([torch.ops.aten.max.dim])
def spyre_max_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.max(input, dim) with conditional handling for bool and int64.
    For bool: convert to float16, perform max, convert back (bool stored as fp16 on Spyre).
    For int64: use CPU fallback custom op (not supported on Spyre).
    For other dtypes: use default PyTorch decomposition (amax + argmax).
    """
    if input.dtype == torch.bool:
        # Reinterpret bool as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
        float_dtype = _get_float_dtype_for_bool()
        input_float = torch.ops.prims.convert_element_type(input, float_dtype)
        values_float = torch.ops.aten.amax(input_float, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input_float, dim=dim, keepdim=keepdim)
        values = torch.ops.prims.convert_element_type(values_float, torch.bool)
        return torch.return_types.max((values, indices))
    elif input.dtype == torch.int64:
        # Use CPU fallback custom op for int64
        return torch.ops.spyre.max_dim_int64_fallback(input, dim=dim, keepdim=keepdim)
    else:
        # Use amax and argmax for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amax(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmax(input, dim=dim, keepdim=keepdim)
        return torch.return_types.max((values, indices))


@register_spyre_decompositions([torch.ops.aten.min.dim])
def spyre_min_dim_decomp(input, dim, keepdim=False):
    """
    Decompose torch.min(input, dim) with conditional handling for bool and int64.
    For bool: convert to float16, perform min, convert back (bool stored as fp16 on Spyre).
    For int64: use CPU fallback custom op (not supported on Spyre).
    For other dtypes: use default PyTorch decomposition (amin + argmin).
    """
    if input.dtype == torch.bool:
        # Reinterpret bool as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
        float_dtype = _get_float_dtype_for_bool()
        input_float = torch.ops.prims.convert_element_type(input, float_dtype)
        values_float = torch.ops.aten.amin(input_float, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmin(input_float, dim=dim, keepdim=keepdim)
        values = torch.ops.prims.convert_element_type(values_float, torch.bool)
        return torch.return_types.min((values, indices))
    elif input.dtype == torch.int64:
        # Use CPU fallback custom op for int64
        return torch.ops.spyre.min_dim_int64_fallback(input, dim=dim, keepdim=keepdim)
    else:
        # Use amin and argmin for supported dtypes (can run on Spyre)
        values = torch.ops.aten.amin(input, dim=dim, keepdim=keepdim)
        indices = torch.ops.aten.argmin(input, dim=dim, keepdim=keepdim)
        return torch.return_types.min((values, indices))


@register_spyre_decompositions([torch.ops.aten.amax.default])
def spyre_amax_decomp(
    input: torch.Tensor, dim=None, keepdim: bool = False
) -> torch.Tensor:
    """
    Decompose torch.amax for boolean tensors.
    For bool tensors: convert to float16, perform amax, convert back (bool stored as fp16 on Spyre).
    For other dtypes: return NotImplemented to use default behavior.
    """
    if input.dtype != torch.bool:
        # For non-bool types, don't decompose - use default lowering
        return NotImplemented

    # For bool tensors: reinterpret as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
    float_dtype = _get_float_dtype_for_bool()
    input_float = torch.ops.prims.convert_element_type(input, float_dtype)
    if dim is None:
        result_float = torch.ops.aten.amax(input_float, keepdim=keepdim)
    else:
        result_float = torch.ops.aten.amax(input_float, dim=dim, keepdim=keepdim)
    return torch.ops.prims.convert_element_type(result_float, torch.bool)


@register_spyre_decompositions([torch.ops.aten.amin.default])
def spyre_amin_decomp(
    input: torch.Tensor, dim=None, keepdim: bool = False
) -> torch.Tensor:
    """
    Decompose torch.amin for boolean tensors.
    For bool tensors: convert to float16, perform amin, convert back (bool stored as fp16 on Spyre).
    For other dtypes: return NotImplemented to use default behavior.
    """
    if input.dtype != torch.bool:
        # For non-bool types, don't decompose - use default lowering
        return NotImplemented

    # For bool tensors: reinterpret as float (fp16 or fp32) using prims.convert_element_type (zero-copy identity op)
    float_dtype = _get_float_dtype_for_bool()
    input_float = torch.ops.prims.convert_element_type(input, float_dtype)
    if dim is None:
        result_float = torch.ops.aten.amin(input_float, keepdim=keepdim)
    else:
        result_float = torch.ops.aten.amin(input_float, dim=dim, keepdim=keepdim)
    return torch.ops.prims.convert_element_type(result_float, torch.bool)


@register_spyre_decompositions([torch.ops.aten.ceil.default])
def spyre_ceil(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.neg.default(
        torch.ops.aten.floor.default(torch.ops.aten.neg.default(input))
    )


@register_spyre_decompositions([torch.ops.aten.bitwise_not])
def bitwise_not(input: torch.Tensor) -> torch.Tensor:
    if input.dtype is torch.bool:
        return torch.logical_not(input)
    else:
        neg_one = torch.ops.aten.full_like(input, -1)
        return torch.ops.aten.bitwise_xor(input, neg_one)


@register_spyre_decompositions([torch.ops.aten.bitwise_and])
def bitwise_and(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    if input1.dtype is torch.bool and input2.dtype is torch.bool:
        return torch.ops.aten.logical_and(input1, input2)
    else:
        return torch.ops.aten.bitwise_not(
            torch.ops.aten.bitwise_or(
                torch.ops.aten.bitwise_not(input1), torch.ops.aten.bitwise_not(input2)
            )
        )


#: Largest kernel tap (per spatial axis) the direct conv2d path accepts. A
#: dense (groups==1) conv contracts over C_in*kH*kW; that per-output-channel
#: weight working set grows with k**2 and, at k>3 with a stick-aligned C_in>=64,
#: exceeds the initial-chunk LX budget -- the backend aborts (the initial
#: chunk must fit in LX) because the kernel taps are pinned no-split
#: (ki/kj=1) and C_out=64 is a single stick, so
#: there is nothing left to tile. Depthwise conv escapes this (its contraction
#: is kH*kW only, no C_in), which is why depthwise supports k up to 9 and dense
#: does not. Until the backend can tile the C_in*kH*kW contraction for dense
#: conv, k>3 stays on the im2col+matmul decomposition.
_CONV_MAX_KERNEL = 3


def _is_direct_conv_supported(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    transposed: bool,
    output_padding: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> bool:
    """Cases the native conv2d direct lowering (lower_convolution) handles.

    Keep this in lock-step with the guards in lower_convolution so that whenever
    the decomposition defers here, the lowering is guaranteed to accept the op.
    Everything else stays on the im2col+matmul decomposition.  Excluded:
    - 1x1 kernel: only the 1x1 case has size-1 kernel taps on *both* axes
      (ki and kj), which the pipeline squeezes out so the emitted SDSC carries
      no window dims at all -- and the backend's conv path expects at least one
      windowed spatial dim, aborting in dimension-mapping (ddl_conversion.cpp
      "Unknown primary dimension kind for a window dimension").  A 1x1 conv is
      just a channel matmul, so it stays on the im2col+matmul path, which handles
      it exactly.  A 1xN / Nx1 kernel squeezes only one tap and keeps the other
      window dim, which the backend does accept (a 1-D conv) -- so those
      direct-lower and are covered by the test_conv2d_direct k1x3 / k3x1 cases;
    - non-zero padding: the backend zero-fill for a padded conv input is not wired
      for regular conv2d, so pad>0 stays on the im2col+matmul path (which pads
      correctly);
    - dilated conv: the windowed-input SDSC fields reuse the avgpool builder,
      which assumes dilation==1, so d>1 stays on the im2col+matmul path;
    - C_in not stick-aligned: Spyre stores C as the innermost (stick) dim, and
      the conv SDSC contracts over C_in with no partial-stick handling. A C_in
      that is not a whole multiple of the fp16 stick width (get_elem_in_stick,
      = 64) would need contraction-dim padding the direct path does not emit
      (known-broken), so it stays on the im2col+matmul path;
    - kernel tap > _CONV_MAX_KERNEL (3): the dense C_in*kH*kW contraction working
      set overflows the LX budget in the backend for k>3 and cannot be tiled
      (see _CONV_MAX_KERNEL), so it stays on the im2col+matmul path;
    - ragged input width under stride: when the strided windows do not exactly
      cover the input width -- (W_in - kW) % sW != 0 -- the fp16 conv opfunc's
      width tiling mis-accumulates the dangling partial column, so such convs
      stay on the im2col+matmul path. A ragged *height* is harmless (height is
      untiled) and stride==1 is never ragged, so this only excludes strided
      convs whose width does not divide evenly (all HW-verified).

    Why these gates run here (decomposition/routing time) rather than in layout
    propagation, where the device stick dim is actually assigned:

    - Declining here is what preserves the fallback. conv2d_via_bmm_decomp either
      defers (returns NotImplemented, leaving aten.convolution for
      lower_convolution to direct-lower) or expands into im2col+matmul -- and once
      it expands, the conv node is gone, replaced by a reshape+bmm subgraph.
      Inductor lowering is a single forward pass with no backtracking, so there is
      no way to un-decompose and re-route afterwards. By the time layout
      propagation runs, the graph is already committed to the direct path; a
      stick-alignment failure discovered there is a hard compile error, not a
      graceful fallback. So the decision has to be made before the branch, i.e.
      here.
    - Making it this early is correct because the stick-alignment gate is
      layout-invariant. C_in is logical dim 1 by the aten.convolution NCHW
      contract (guarded by input.dim() == 4), and ``C_in % stick == 0`` is a
      property of the channel *count*, which no layout choice changes -- layout
      propagation picks stick *placement*, not size. We are not assuming which
      host dim becomes the stick: the direct path itself forces channel-last (C on
      the stick) to feed the PE-array contraction, so this validates a
      precondition of the layout the path *will request*, not a guess about an
      assignment the solver is free to make differently. The stick width is
      get_elem_in_stick(torch.float16) == 64, derived from the dtype the fp16 gate
      above already pins -- not a hardcoded dim assumption.

    Assumption this routing decision rests on (documented, not enforced here):
    the gate reads C_in from logical dim 1 (guaranteed by the aten.convolution
    NCHW contract) and assumes the direct path will place C_in on the device
    stick. That stick placement is NOT decided here -- it is requested by the
    direct path and enforced downstream in propagate_layouts (_conv_layouts /
    find_stick_compatible_input_layout), which restickify the activation onto
    C_in or raise Unsupported if they cannot. So the ``C_in % stick == 0`` check
    below is a precondition of the channel-last layout the path *will request*,
    validated against the channel *count* (which no layout choice changes). The
    assumption is only that C_in-on-stick keeps being the layout a direct-conv
    node lands in; if that ever stops holding (e.g. a solver change assigns a
    different stick dim to a direct-conv node), this gate would be checking the
    wrong dimension and could route wrongly. It is documented here as an
    assumption rather than re-checked after layout assignment because by then
    the im2col+matmul fallback branch is gone (see above) -- a mismatch surfaces
    downstream as a hard Unsupported, not silent wrong numerics.
    """
    kH, kW = weight.shape[-2], weight.shape[-1]
    C_in = input.shape[1]
    eps = get_elem_in_stick(torch.float16)
    supported = (
        not transposed
        and all(op == 0 for op in output_padding)
        and all(p == 0 for p in padding)
        and all(d == 1 for d in dilation)
        and groups == 1
        and input.dim() == 4
        and input.dtype == torch.float16
        and not (kH == 1 and kW == 1)
        # Dense conv k>3 overflows the LX contraction budget in the backend.
        and kH <= _CONV_MAX_KERNEL
        and kW <= _CONV_MAX_KERNEL
        # isinstance guard: a dynamic-shape C_in (SymInt) is not statically known
        # to be stick-aligned, so fall back to the decomposition rather than
        # branching on a symbolic divisibility (which would add a shape guard).
        and isinstance(C_in, int)
        # Assumes the direct path lands C_in (logical dim 1) on the stick; that
        # is requested by the path and enforced in propagate_layouts, not here.
        # See the "Assumption this routing decision rests on" note above.
        and C_in % eps == 0
    )
    if not supported:
        return False
    # Ragged input width (see docstring): the fp16 opfunc tiles the output width
    # and mis-accumulates the dangling column when (W_in - kW) % sW != 0. Only
    # decidable for a static width; a dynamic (SymInt) width stays on the direct
    # path rather than adding a symbolic-remainder shape guard.
    W_in = input.shape[-1]
    sW = stride[-1]
    if isinstance(W_in, int) and (W_in - kW) % sW != 0:
        return False
    return True


@register_spyre_decompositions([torch.ops.aten.convolution.default])
def conv2d_via_bmm_decomp(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    """
    Decompose 2D convolution into batch matrix multiplication using torch.nn.unfold.
    torch.nn.unfold directly returns (N, C_in * K_h * K_w, H_out * W_out), avoiding
    intermediate reshape/view/unsqueeze operations.
    For depthwise convolutions (C_in = groups = C_out), invoke torch.spyre.conv2d directly.
    """
    # When the direct-lowering flag is on and the case is supported, decline the
    # decomposition (return NotImplemented) so aten.convolution.default survives
    # in the FX/AOT graph and reaches the Spyre lowering (lower_convolution),
    # which emits a native conv2d SDSC. Unsupported cases (grouped/transposed/
    # non-fp16) fall through and decompose to im2col+matmul as before. This is
    # the compile-path target; the flag defaults off so eager and default
    # compile behavior are unchanged.
    if config.conv2d_direct_lowering and _is_direct_conv_supported(
        input, weight, stride, transposed, output_padding, padding, dilation, groups
    ):
        return NotImplemented

    if transposed:
        raise Unsupported("conv2d_via_bmm: transposed convolution not supported")

    if any(op != 0 for op in output_padding):
        raise Unsupported("conv2d_via_bmm: output_padding not supported")

    if input.dim() != 4:
        raise Unsupported(f"conv2d_via_bmm: expected 4D input, got {input.dim()}D")

    N, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, K_h, K_w = weight.shape

    # For depthwise convolutions (C_in = groups = C_out), use torch.spyre.conv2d_with_bias
    if C_in == groups == C_out:
        return torch.ops.spyre.conv2d_with_bias(
            input, weight, bias, stride, padding, dilation, groups
        )

    stride_h, stride_w = stride[0], stride[1]
    pad_h, pad_w = padding[0], padding[1]
    dil_h, dil_w = dilation[0], dilation[1]

    if C_in != groups * C_in_per_group:
        raise Unsupported(
            f"conv2d_via_bmm: expect C_in == groups * C_in_per_group, got C_in: {C_in}, groups: {groups} C_in_per_group: {C_in_per_group}"
        )

    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    patches = torch.ops.spyre.unfold(
        input,
        kernel_size=(K_h, K_w),
        dilation=(dil_h, dil_w),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
    )

    if groups == 1:
        # weight_2d = weight.reshape(C_out, C_in_per_group * K_h * K_w)
        weight_2d = torch.ops.spyre.reshape_via_cpu(
            weight, (C_out, C_in_per_group * K_h * K_w)
        )
        weight_2d_exp = weight_2d.unsqueeze(0).expand(N, -1, -1)
        weight_2d_exp_cln = weight_2d_exp.clone()
        # output = torch.matmul(weight_2d, patches)
        output = torch.matmul(weight_2d_exp_cln, patches)
    else:
        C_out_per_group = C_out // groups
        # patches = patches.reshape(N, groups, C_in_per_group * K_h * K_w, H_out * W_out)
        patches = torch.ops.spyre.reshape_via_cpu(
            patches, (N, groups, C_in_per_group * K_h * K_w, H_out * W_out)
        )
        # weight_grouped = weight.reshape(groups, C_out_per_group, C_in_per_group * K_h * K_w)
        weight_grouped = torch.ops.spyre.reshape_via_cpu(
            weight, (groups, C_out_per_group, C_in_per_group * K_h * K_w)
        )

        output = torch.matmul(
            weight_grouped.unsqueeze(0),
            patches,
        )
        output = output.reshape(N, C_out, H_out * W_out)

    if bias is not None:
        # Add bias while the output is still (N, C_out, H_out * W_out): the
        # matmul lays the trailing H_out*W_out dim on the stick, and for a
        # patch-embed conv (e.g. Prithvi's stride-16 kernel) that flat length
        # is a multiple of 64 even when H_out/W_out individually are not. If we
        # instead reshaped to (N, C_out, H_out, W_out) first and broadcast the
        # bias over a sub-stick spatial width (e.g. W_out == 32), the layout
        # solver rejects the resulting `w + 32*Mod(row, 2)` stick expression.
        bias_shaped = torch.ops.spyre.reshape_via_cpu(bias, (1, C_out, 1))
        output = output + bias_shaped

    output = output.reshape(N, C_out, H_out, W_out)

    return output


@register_spyre_decompositions([torch.ops.spyre.conv2d_with_bias.default])
def spyre_conv2d_with_bias_decomp(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """
    Decompose torch.ops.spyre.conv2d_with_bias into:
      1. torch.ops.spyre.conv2d without bias
      2. Expand bias to (N, C_out, H_out, W_out) for broadcasting
      3. Add the bias to the output

    This keeps the spyre.conv2d lowering simple while supporting conv2d with bias.
    """
    # Call spyre.conv2d without bias
    output = torch.ops.spyre.conv2d(input, weight, stride, padding, dilation, groups)

    # If bias is present, add it
    if bias is not None:
        # Get output shape
        N, C_out, H_out, W_out = output.shape
        # Reshape bias to (1, C_out, 1, 1) then expand to (N, C_out, H_out, W_out)
        # This avoids stick layout issues by expanding before adding
        bias_expanded = bias.reshape(1, C_out, 1, 1).expand(N, C_out, H_out, W_out)
        output = output + bias_expanded

    return output


# Register decomposition for custom spyre op (not aten, so use decomp.register_decomposition directly)
@register_spyre_decompositions([torch.ops.spyre.dequantize_fp8_with_scale])
def dequantize_fp8_with_scale_decomp(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Decompose dequantize_fp8_with_scale into:
    1. FP8→FP16 conversion using .to() (triggers fp8todl16 via dtype_ops)
    2. Multiply by scale

    This decomposition is executed during compilation and removes the custom op
    from the graph before lowering.
    """
    x_fp16 = input.to(torch.float16)
    return x_fp16 * scale


@register_spyre_decompositions([torch.ops.aten._scaled_mm.default])
def scaled_mm_decomp(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
    bias: torch.Tensor = None,
    scale_result: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    Decompose _scaled_mm into:
    1. Raw FP8 matmul via spyre.scaled_mm (no scale/bias applied)
    2. Multiply by scale_a, if present
    3. Multiply by scale_b, if present
    4. Add bias, if present

    This decomposition is executed during compilation and keeps scale/bias
    arithmetic out of lower_scaled_mm's matmul lowering - the same
    separation dequantize_fp8_with_scale_decomp uses for its FP8->FP16
    conversion.
    """
    result = torch.ops.spyre.scaled_mm(mat1, mat2, out_dtype=out_dtype)

    if scale_a is not None:
        result = result * scale_a
    if scale_b is not None:
        result = result * scale_b
    if bias is not None:
        result = result + bias

    if scale_result is not None:
        logger.warning("scale_result parameter in _scaled_mm is not yet supported")
    if use_fast_accum:
        logger.warning("use_fast_accum parameter in _scaled_mm is not yet supported")

    return result


@register_spyre_decompositions([torch.ops.aten.where.ScalarOther])
def where_scalar_other_decomp(condition, self, other):
    other_t = torch.full_like(self, other)
    return torch.ops.aten.where.self(condition, self, other_t)


@register_spyre_decompositions([torch.ops.aten.where.ScalarSelf])
def where_scalar_self_decomp(condition, self, other):
    self_t = torch.full_like(other, self)
    return torch.ops.aten.where.self(condition, self_t, other)


@register_spyre_decompositions([torch.ops.aten.where.Scalar])
def where_scalar_decomp(condition, self, other):
    # Must use dtype float16 for spyre backend where3
    dtype = torch.float16

    # Use full.default instead of full_like to explicitly control dtype
    self_t = torch.ops.aten.full.default(
        list(condition.shape),
        self,
        dtype=dtype,
        device=condition.device,
    )
    other_t = torch.ops.aten.full.default(
        list(condition.shape),
        other,
        dtype=dtype,
        device=condition.device,
    )

    return torch.ops.aten.where.self(condition, self_t, other_t)


@register_spyre_decompositions([torch.ops.spyre.quantize_fp8_with_scale])
def spyre_quantize_fp8_with_scale(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    inv_scale = torch.reciprocal(scale)
    x_scaled = input * inv_scale
    x_clamped = torch.ops.spyre.clamp(x_scaled, FP8_E4M3FN_MIN, FP8_E4M3FN_MAX)
    return torch.ops.spyre.qfp8ch(x_clamped)


@register_spyre_decompositions([torch.ops.spyre.quantize_weight_fp8_with_scale])
def spyre_quantize_weight_fp8_with_scale(
    input: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    inv_scale = torch.reciprocal(scale)
    x_scaled = input * inv_scale
    x_clamped = torch.ops.spyre.clamp(x_scaled, FP8_E4M3FN_MIN, FP8_E4M3FN_MAX)
    return torch.ops.spyre.qfp8wt(x_clamped)


@register_spyre_decompositions([torch.ops.aten.flip.default])
def spyre_flip(input: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """Reverse ``input`` along each dim in ``dims`` using gathers.

    Inductor's default decomposition of ``aten.flip`` is ``prims.rev``, whose
    index expression walks the reversed dim backwards (``N - 1 - i``). Device
    coordinates can only ascend, so that form is not lowerable on Spyre —
    ``compute_coordinates`` rejects it. The same reversal expressed as an
    ``index_select`` with a descending index tensor is an ordinary gather,
    which the backend does support: the descending order lives in the index
    *values* rather than in the access pattern.

    Registering the aten op also installs the PrivateUse1 kernel, so this is
    what eager ``Tensor.flip`` dispatches to as well (there is no
    ``aten::flip`` kernel for Spyre otherwise).
    """
    # Replacing the op means aten.flip's own argument validation no longer
    # runs, so repeat it here rather than silently accepting a program aten
    # rejects. Out-of-range dims still raise from ``size()`` below.
    seen: set[int] = set()
    for dim in dims:
        normalized = dim + input.dim() if dim < 0 else dim
        if normalized in seen:
            raise RuntimeError(
                f"dim {normalized} appears multiple times in the list of dims"
            )
        seen.add(normalized)

    out = input
    reversed_any = False
    for dim in dims:
        # A 0-d tensor accepts flip(0) and is its own reversal; ``size(0)``
        # would raise on it, so skip before asking.
        size = 1 if input.dim() == 0 else out.size(dim)
        if size <= 1:
            # A dim of size 0 or 1 is its own reversal; index_select would
            # still work, but skipping avoids an empty/degenerate gather.
            continue
        index = torch.arange(size - 1, -1, -1, device=out.device, dtype=torch.int32)
        out = torch.index_select(out, dim, index)
        reversed_any = True
    # aten.flip always returns a fresh tensor; clone so the no-op case does
    # not alias its input.
    return out if reversed_any else out.clone()


@register_spyre_decompositions([torch.ops.aten.prod.dim_int])
def spyre_prod_dim_int(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    # Currently, restickify does not support fp32 (int64 is also converted to fp32
    # for now, so it is unsupported as well).
    # Use decomposition in these cases as a safe fallback, even if restickify
    # might not be needed in the end.
    if input.dtype != torch.float32 and input.dtype != torch.int64:
        return torch.ops.spyre.prod_dim_int(input, dim, keepdim)

    if dim < 0:
        dim += input.ndim
    out_shape = list(input.shape)
    reduce_size = out_shape.pop(dim)
    acc = torch.ones(out_shape, dtype=input.dtype, device=input.device)
    for i in range(reduce_size):
        acc = acc * input.select(dim, i)

    if keepdim:
        acc = acc.unsqueeze(dim)

    return acc


@register_spyre_decompositions(
    [torch.ops.aten.all.default, torch.ops.aten.all.dim, torch.ops.aten.all.dims]
)
def spyre_all(
    input: torch.Tensor,
    dim=None,
    keepdim: bool = False,
) -> torch.Tensor:
    # Convert bool to float16 if needed
    if input.dtype is torch.bool:
        tmp = input.to(torch.float16)
    else:
        tmp = input

    tmp = torch.abs(tmp)
    result = torch.amin(tmp, dim=dim, keepdim=keepdim)

    return result.to(torch.bool)


def _masked_scatter_reject_reason(
    self: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> Optional[str]:
    """Why ``masked_scatter`` cannot use the row-level path here, or ``None`` if it can.

    The mask must select whole rows: constant across the last (column) dim, with
    every other dim matching ``self`` so there is exactly one mask bool per row.
    That covers both spellings PyTorch may hand us for a row-broadcast mask, which
    the body collapses identically via ``mask[..., 0]``:
      * un-expanded -- the last dim is literally ``1`` (e.g. ``[B, S, 1]``); or
      * expanded    -- the last dim is ``cols`` but broadcast (``stride(-1) == 0``).

    Checks are ordered cheapest/most-general to most-specific:
      1. Rank guards (no device_layout access needed).
      2. Leading-dim equality (one mask entry per row).
      3. Degenerate column guard (cols <= 1 is not a meaningful row).
      4. Source column alignment.
      5. Per-row last-dim check (the structural row-level requirement).
    """
    if self.dim() < 2 or source.dim() < 2 or mask.dim() != self.dim():
        return f"rank: self={self.dim()} mask={mask.dim()} source={source.dim()}"
    # One mask entry per row: every dim but the last must match self exactly, so
    # mask[..., 0] has exactly `rows` elements. (The last dim may differ: it is
    # either a literal 1 or a broadcast of cols -- checked below.)
    if tuple(mask.shape[:-1]) != tuple(self.shape[:-1]):
        return (
            f"mask leading dims {tuple(mask.shape[:-1])} != self "
            f"{tuple(self.shape[:-1])}"
        )
    cols = self.shape[-1]
    # cols <= 1 is a degenerate row that offers no block-per-row equivalence.
    if cols <= 1:
        return f"degenerate last dim: cols={cols}"
    if source.shape[-1] != cols:
        return f"source last dim {source.shape[-1]} != self last dim {cols}"
    # Per-row means the mask is constant across the last dim. Accept a literal
    # size-1 last dim (un-expanded) or a broadcast last dim (stride 0); reject a
    # genuinely per-element mask (last dim cols with a non-zero stride).
    if mask.shape[-1] != 1 and mask.stride(-1) != 0:
        return (
            f"mask is not per-row in its last dim (shape {tuple(mask.shape)}, "
            f"stride {tuple(mask.stride())})"
        )
    return None


@register_spyre_decompositions([torch.ops.aten.masked_scatter.default])
def spyre_masked_scatter(
    self: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """`masked_scatter` for a mask that is broadcast along the last dim.

    Such a mask (`stride(-1) == 0` -- e.g. an attention mask `[B, S]`
    expanded to `[B, S, C]`) selects *whole rows*: row `i`, if selected,
    consumes exactly one contiguous `C`-element block of `source`, i.e. one
    whole row of `source.reshape(-1, C)`. The gather is then
    `source_2d[row_idx]` -- a 1D index into the row dim of a 2D source, which
    is a plain stick gather.

    Any other mask makes this an element-level op, which Spyre cannot express:
    the index would have to address an element inside a *packed* 1D source, and
    a lane within a stick is not addressable. Exploding the source to one
    element per stick does not help -- that splits every stick 64 ways, an
    element scatter the backend cannot lower. So the generic form is rejected
    here rather than emitting a gather that fails deeper in layout propagation.
    """
    reason = _masked_scatter_reject_reason(self, mask, source)
    if reason is not None:
        raise Unsupported(
            f"masked_scatter needs a mask broadcast along the last dim so it "
            f"selects whole rows ({reason}). An element-level masked_scatter "
            f"would gather individual elements from a packed 1D source, and a "
            f"lane within a stick is not addressable on Spyre."
        )

    cols = self.shape[-1]
    rows = self.numel() // cols
    # Collapse the broadcast dim: one bool per row.
    mask_row = mask[..., 0].reshape(rows)
    source_2d = source.reshape(-1, cols)
    # masked_scatter requires source.numel() >= mask.sum(). For a whole-row
    # mask, source_2d needs at least as many rows as there are selected rows.
    torch._assert_async(
        mask_row.sum() <= source_2d.shape[0],
        "masked_scatter: source is too short -- it has fewer rows than the "
        "number of selected (True) mask rows.",
    )
    # Row i reads source row (prefix-count of selected rows - 1). Unselected
    # rows would get -1, so multiply by the mask to send them to row 0 instead
    # (in bounds, and discarded by the where). Done in fp32: Spyre has no usable
    # int `clip`, int32 sub/mul are unsupported, and fp16 loses large indices.
    pos = mask_row.cumsum(0).to(torch.float32) - 1.0
    row_idx = (pos * mask_row.to(torch.float32)).to(torch.int64)
    # The gather stays 2D (its args are indirect, so the pointwise dim_order
    # projection skips them), but the `where` must stay at `self`'s rank:
    # that projection computes `rank_diff = len(output) - len(arg)` and only
    # handles inputs of *lower* rank. A lower-rank output against the full-rank
    # ND mask gives rank_diff < 0, which shifts dims the wrong way and builds a
    # layout whose dim_order rank no longer matches host_size ("Incompatible
    # host_size and dim_order"). Reshaping back to ND keeps every `where`
    # input rank-aligned with the output.
    gathered = source_2d[row_idx].reshape(self.shape)
    return torch.where(mask, gathered, self)


@register_spyre_decompositions([torch.ops.aten.index_add.default])
def spyre_index_add(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
    *,
    alpha: Union[int, float] = 1,
) -> torch.Tensor:
    """`index_add` as gather + add + overwrite-scatter, fully on device.

    `out.index_add_(dim, index, source * alpha)` is a read-modify-write:
    read the current values at the target slots, add the (scaled) source, and
    write them back, using primitives Spyre runs on the indirect-access engine:

      * `index_select`  -> on-device indirect gather
      * `index_put`     -> on-device indirect overwrite store

    PRECONDITION -- `index` must contain NO DUPLICATE values. A read-modify-
    write cannot sum colliding writes: every duplicate reads the same old value
    and the overwrite store keeps only the last writer, so duplicate indices are
    SILENTLY WRONG.
    """
    dim = dim % self.dim()
    if alpha != 1:
        source = source * alpha
    # Read current destination values, then add the source onto them.
    gathered = torch.index_select(self, dim, index)
    updated = gathered + source
    indices: list[Optional[torch.Tensor]] = [None] * dim + [index]
    return torch.index_put(self, indices, updated, accumulate=False)
