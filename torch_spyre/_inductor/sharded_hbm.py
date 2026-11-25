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

"""Non-unified kernels over sharded storage.

A sharded tensor (``torch.spyre.to_sharded``) keeps shard *c* in its own
device chunk, which belongs to core *c*. An op whose operands are sharded
compiles to a non-unified kernel: core *c* reads and writes only chunk *c*,
each through its own start address (``nonUnifiedAllocInHBM_``).

Two op kinds are admitted:

- same-shape pointwise ops: the result has the sharded operands' placement.
  Such an op may also read *unsharded* operands, which act as read-only
  replicas: a scalar constant, or a plain unified graph input (a bias, or a
  full-size tensor). Every core reads the replica through the ordinary
  unified addressing, so no shard table is involved. Sharded operands must
  still agree on one placement; two different placements need data movement
  between chunks and are rejected;
- shard-local reductions: a reduction over dims that do not include the
  shard dim. Each core reduces its own shard and writes its own output
  shard, so no data crosses shards. The result is sharded along the
  remapped shard dim (``s`` with ``keepdim``, else ``s`` minus the number of
  reduced dims before ``s``) and gets its own shard size.

The shard plan, not the work-division planner, sets the work split: this
pass pins it to ``num_shards`` on the shard dim before work division runs.
"""

from __future__ import annotations

import dataclasses
import math

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
    ReinterpretView,
)
from torch._inductor.virtualized import V

from . import config
from .errors import Unsupported
from .ir import SpyreConstantFallback
from .pass_utils import op_read_writes
from torch_spyre.sharding import _require_exact_int, _require_int_list


@dataclasses.dataclass(frozen=True)
class ShardedPlacement:
    """Address-free sharding of one buffer: shard *c* lives in chunk *c*."""

    shard_dim: int
    num_shards: int
    shard_device_bytes: int


def placement_from_tensor(tensor: torch.Tensor) -> ShardedPlacement | None:
    """The sharding of a real (non-fake) Spyre tensor, or None if unsharded."""
    from torch_spyre import _C

    if not _C._spyre_is_sharded(tensor):
        return None
    info = _C._spyre_sharding_info(tensor)
    return ShardedPlacement(
        shard_dim=_require_exact_int("shard_dim", info["shard_dim"]),
        num_shards=_require_exact_int("num_shards", info["num_shards"]),
        shard_device_bytes=_require_exact_int(
            "shard_device_bytes", info["shard_device_bytes"]
        ),
    )


def graph_placements(graph: GraphLowering | None = None) -> dict[str, ShardedPlacement]:
    """Buffer name -> placement for the graph being compiled."""
    graph = graph if graph is not None else V.graph
    placements = getattr(graph, "_spyre_sharded_placements", None)
    if placements is None:
        placements = {}
        graph._spyre_sharded_placements = placements
    return placements


def placement_of(name: str) -> ShardedPlacement | None:
    return graph_placements().get(name)


def _local_layout(size: list[int], dtype: torch.dtype, placement: ShardedPlacement):
    from torch_spyre._C import SpyreTensorLayout

    local = list(size)
    local[placement.shard_dim] //= placement.num_shards
    return SpyreTensorLayout(local, dtype)


def check_sharded_input(
    name: str, tensor: torch.Tensor, placement: ShardedPlacement, stl
) -> None:
    """A sharded graph input must be a base tensor in its own shard layout."""
    from torch_spyre import _C

    size = list(tensor.shape)
    ok = (
        tensor._base is None
        # A detached block view (x[:8].detach()) has no _base; only the
        # whole tensor has the shape the storage was allocated for.
        and size
        == _require_int_list(
            "global_shape", _C._spyre_sharding_info(tensor)["global_shape"]
        )
        and tensor.is_contiguous()
        and tensor.storage_offset() == 0
        and 0 <= placement.shard_dim < len(size)
        and size[placement.shard_dim] % placement.num_shards == 0
        and stl == _local_layout(size, tensor.dtype, placement)
    )
    if not ok:
        raise Unsupported(
            f"{name}: non-unified kernels need sharded inputs that are base "
            "tensors from to_sharded (no views, slices or reshapes)"
        )


def _shard_device_bytes(
    size: list[int], dtype: torch.dtype, shard_dim: int, num_shards: int
) -> int:
    """Device bytes of one shard of ``size``, as ``_spyre_empty_sharded``."""
    from torch_spyre._C import get_device_size_in_bytes

    return get_device_size_in_bytes(
        _local_layout(size, dtype, ShardedPlacement(shard_dim, num_shards, 0))
    )


def _check_core_match(placement: ShardedPlacement) -> None:
    """Non-unified kernels need one core per shard.

    The storage side is validated where the tensor is created:
    ``to_sharded`` checks the placement receipt in Python. This checks
    the compiler-side core count so the pinned work split has a core to pin
    to.
    """

    if placement.num_shards != config.sencores:
        raise Unsupported(
            "non-unified kernels need one core per shard: sharded operands have "
            f"{placement.num_shards} shards but "
            f"SENCORES={config.sencores}"
        )


# TODO(#6175): share one symbol finder with spyre_kernel._non_unified_shard_dim
# (and fold _reduced_input_dims in) when the matmul work shapes it.
def _shard_symbol(write: MemoryDep, size: list[int], shard_dim: int) -> sympy.Symbol:
    """The iteration symbol that walks ``shard_dim`` of a contiguous output."""
    stride = math.prod(size[shard_dim + 1 :])
    matches = [
        sym
        for sym, extent in write.ranges.items()
        if write.index.coeff(sym) == stride and int(extent) == size[shard_dim]
    ]
    if len(matches) != 1:
        raise Unsupported(
            f"cannot find the iteration dim for shard dim {shard_dim} in "
            f"{write.index} (ranges {write.ranges})"
        )
    return matches[0]


def _check_shard_layout(
    name: str, size: list[int], dtype: torch.dtype, placement: ShardedPlacement
) -> None:
    """Reject shardings whose shards are not whole device dims of whole sticks.

    Shard-local device strides may differ from the global ones: DeepTools
    gives each core its own start address and walks the core's tile with the
    shard-local layout (verified end-to-end for pointwise ops and for
    shard-local and cross-shard reductions, bit-exact on integer data). An
    earlier version of this check also required equal strides. That rule
    guarded against a DeepTools import bug, not against stride handling:
    before DeepTools e1733ab8ad ("Read all SDSC contract flags through one
    shared bool importer"), ``nonUnifiedAllocInHBM_`` emitted as a JSON
    bool was silently parsed as disabled (json11's int_value() of ``true``
    is 0), so non-unified HBM allocations were sized and addressed as
    unified -- the then-measured counterexamples (a second stick left
    unwritten) came from that. With the flag honored, and af3e583a4d
    sizing non-unified allocations from the per-core datastage, the
    counterexamples do not reproduce; the codegen guard's unified-model
    stride arithmetic (the other half of the old rule) was replaced by
    the shard-dim identity check in ``verify_non_unified_split``. What
    must still hold:

    - the shard keeps the global device layout's dim structure (a restickified
      or otherwise relaid shard is not chunk-addressable);
    - the shard boundary falls on whole sticks of the stick dim (a stick
      cannot be split between chunks -- a permanent limit);
    - exactly one device dim is divided, exactly num_shards ways (each core's
      chunk is the tensor cut on one dim).
    """
    from torch_spyre._C import SpyreTensorLayout

    local_size = list(size)
    local_size[placement.shard_dim] //= placement.num_shards
    global_dev = list(SpyreTensorLayout(size, dtype).device_size)
    local_dev = list(SpyreTensorLayout(local_size, dtype).device_size)
    elems_per_stick = global_dev[-1]

    split = [i for i, (g, lo) in enumerate(zip(global_dev, local_dev)) if g != lo]
    ok = (
        len(global_dev) == len(local_dev)
        # The shard boundary falls on whole sticks of the stick dim.
        and (
            placement.shard_dim != len(size) - 1
            or local_size[-1] % elems_per_stick == 0
        )
        # Exactly one device dim is divided, exactly num_shards ways ...
        and len(split) == 1
        and global_dev[split[0]] == placement.num_shards * local_dev[split[0]]
    )
    if not ok:
        raise Unsupported(
            f"{name}: non-unified kernels need shards that are whole device "
            f"dims of whole sticks; got shape {size}, shard dim "
            f"{placement.shard_dim}, {placement.num_shards} shards (device "
            f"sizes {global_dev} vs shard {local_dev})"
        )


def _mutation_target(op: ComputedBuffer) -> str | None:
    layout = op.layout
    if not isinstance(layout, MutationLayoutSHOULDREMOVE):
        return None
    target = layout.target
    while isinstance(target, ReinterpretView):
        target = target.data
    return target.get_name() if hasattr(target, "get_name") else None


def verify_mixed_args(sdsc_spec, shard_kernel_args) -> None:
    """Codegen guard for a kernel that mixes sharded and ordinary tensors.

    Two shapes are admitted:

    - a cross-shard (ring) reduction reads sharded storage (non-unified, one
      start per core) and writes an ordinary tensor that the ring chain's end
      core fills: the one ordinary argument is the output;
    - a pointwise op writes a sharded output and may also read ordinary
      inputs (a scalar constant or a unified tensor, see ``plan_sharded_hbm``).
      Those are addressed as in any unified kernel, so only the output is
      constrained.

    This is only a backstop: any ordinary input is accepted when the output is
    sharded. The planner is the real gate; see ``_require_replica_source`` and
    ``_require_replica_read``.
    """
    # SDSC args list inputs first and the output last. (For a reduction,
    # ``num_inputs`` counts the output too, because it is also the
    # accumulator, so position is the reliable test.)
    output = len(sdsc_spec.args) - 1
    plain = [i for i, t in enumerate(sdsc_spec.args) if not t.non_unified]
    if not shard_kernel_args or (output in plain and plain != [output]):
        raise Unsupported(
            "non-unified kernels need a sharded output, or sharded inputs and "
            "one ordinary output (a cross-shard ring reduction); other "
            "ordinary arguments may only be inputs"
        )


def verify_non_unified_split(sdsc_spec, shard_kernel_args) -> None:
    """Codegen guard: core c's work slice is shard c, and nothing else splits.

    The pass pins the split; this re-checks the committed division the SDSC
    is emitted from, so a later pass cannot silently re-divide a sharded op.
    The split dim may be a reduction dim (a ring reduction over the shard
    dim): the check is the same, core c's slice of it is shard c.
    """
    n = sdsc_spec.num_cores
    try:
        split_dims = {d: int(v) for d, v in sdsc_spec.work_slices.items() if int(v) > 1}
        core_slices = {
            (c, d): int(expr.xreplace({sympy.Symbol("core_id"): c}))
            for c in range(n)
            for d, expr in sdsc_spec.core_id_to_work_slice.items()
        }
    except (TypeError, ValueError) as error:
        raise Unsupported(
            f"non-unified kernel work split is not statically decidable: "
            f"{sdsc_spec.work_slices} on {n} cores"
        ) from error
    if len(split_dims) != 1 or next(iter(split_dims.values())) != n:
        raise Unsupported(
            "non-unified kernel work split must be exactly num_shards on the "
            f"shard dim; got {sdsc_spec.work_slices} on {n} cores"
        )
    (dim,) = split_dims
    for (c, d), got in core_slices.items():
        want = c if d == dim else 0
        if got != want:
            raise Unsupported(
                f"non-unified kernel maps core {c} to slice {got} of dim "
                f"{d}; it must own shard {c}"
            )
    # The split dim must be the shard dim. Non-unified args are NOT addressed
    # by the unified model (compute_ops.core_idx_to_slice_offset never runs for
    # them -- each core starts from its own symbolic shard address), so the
    # shard binding cannot be re-derived from the emitted strides: the earlier
    # version computed the slice-to-slice distance as strides[dim]/n and
    # required it to equal one shard, which holds only when the shard dim is
    # the innermost non-stick device dim (measured: (32,128) fp16 Shard(0)
    # splits mb 8 ways with strides[mb]=2048, one slice 512 bytes, one shard
    # 1024). Instead each sharded arg carries the shard dim's symbol
    # (TensorArg.non_unified_shard_dim, derived from the tensor's placement
    # against the iteration space the SDSC is emitted from), and the
    # committed split is checked against it directly. With exactly one dim
    # split n ways, core c on slice c, and that dim the shard dim, core c's
    # work slice is shard c's element set by construction -- the shard is the
    # tensor divided n ways on its shard dim, which the planner already
    # checked divides evenly.
    for tensor in shard_kernel_args:
        # Defence in depth: the arg must actually index
        # the split dim in the emitted SDSC, so a later tiling pass that
        # splits the shard dim into sub-dims fails closed here even if
        # the dim label survives the symbol comparison below.
        if tensor.scales.get(dim, 0) <= 0:
            raise Unsupported(
                f"non-unified kernel splits {dim}, which tensor arg "
                f"{tensor.arg_index} does not index"
            )
        shard_dim = tensor.non_unified_shard_dim
        if shard_dim is None:
            raise Unsupported(
                f"non-unified kernel splits {dim}, but tensor arg "
                f"{tensor.arg_index} records no shard dim (its producer was "
                "not admitted by the sharded planner)"
            )
        if shard_dim != dim:
            raise Unsupported(
                f"non-unified kernel splits {dim}, but tensor arg "
                f"{tensor.arg_index} is sharded on {shard_dim}"
            )


# Reductions whose value over one shard needs nothing from other shards when
# the shard dim is not reduced. ``mean`` divides by the reduced extent, which
# is the same on every core because the reduced dims are not split.
_SHARD_LOCAL_REDUCTIONS = ("sum", "mean", "max", "min")


def _reduced_input_dims(
    name: str, read: MemoryDep, write: MemoryDep, in_size: list[int]
) -> tuple[list[int], dict[int, sympy.Symbol], dict[int, sympy.Symbol]]:
    """Which input dims a reduction reduces, the kept dims' symbols, and
    every dim's symbol.

    The read index walks a contiguous input: input dim ``d`` is the symbol
    whose coefficient is ``stride(d)`` and whose range is ``in_size[d]``. A
    dim is reduced when its symbol does not appear in the write index.
    Returns the reduced dims, for each kept dim its symbol, and the full
    dim -> symbol map (the cross-shard planner needs the reduced dims'
    symbols too).
    """
    strides = [math.prod(in_size[d + 1 :]) for d in range(len(in_size))]
    dim_symbol: dict[int, sympy.Symbol] = {}
    for d, extent in enumerate(in_size):
        if extent == 1:
            continue
        matches = [
            sym
            for sym, rng in read.ranges.items()
            if read.index.coeff(sym) == strides[d] and int(rng) == extent
        ]
        if len(matches) != 1:
            raise Unsupported(
                f"{name}: non-unified reductions need a plain read of a "
                f"contiguous sharded input; cannot find input dim {d} in "
                f"{read.index} (ranges {read.ranges})"
            )
        dim_symbol[d] = matches[0]
    if len(set(dim_symbol.values())) != len(dim_symbol) or set(read.ranges) != set(
        dim_symbol.values()
    ):
        raise Unsupported(
            f"{name}: non-unified reductions need one iteration symbol per input "
            f"dim; got {read.index} (ranges {read.ranges})"
        )
    write_symbols = write.index.free_symbols
    reduced = [d for d, sym in dim_symbol.items() if sym not in write_symbols]
    kept = {d: sym for d, sym in dim_symbol.items() if sym in write_symbols}
    return reduced, kept, dim_symbol


def _plan_cross_shard_reduction(
    op: ComputedBuffer,
    name: str,
    in_size: list[int],
    reduced: list[int],
    dim_symbol: dict[int, sympy.Symbol],
    placement: ShardedPlacement,
) -> None:
    """Admit a reduction over the shard dim as one ring-combined program.

    The reduced shard dim is split across cores along the shard boundaries,
    so core c's work slice is input shard c. DeepTools' cross-core reduction
    then passes each core's partial over the ring, and the chain's end core
    writes the result, which is an ordinary (unsharded) tensor. Nothing is
    recorded for the output in the shard plan.
    """
    s = placement.shard_dim
    # mean is admitted: the ring templates divide once by the global reduced
    # extent (measured: the result matches CPU within fp16 rounding, closer
    # than the unsharded device mean). Shards are equal, so this is the
    # global mean.
    if reduced != [s]:
        raise Unsupported(
            f"{name}: cross-shard reductions must reduce the shard dim only "
            f"(DeepTools splits at most one reduced dim across cores); "
            f"reduced dims {reduced}"
        )
    if s == len(in_size) - 1:
        raise Unsupported(
            f"{name}: cross-shard reductions over the stick dim are not supported"
        )
    _check_shard_layout(name, in_size, op.get_dtype(), placement)
    op.shard_pinned_splits = {dim_symbol[s]: placement.num_shards}
    op.shard_ring_reduction = True


def _plan_shard_local_reduction(
    graph: GraphLowering,
    op: ComputedBuffer,
    read: MemoryDep,
    write: MemoryDep,
    placement: ShardedPlacement,
) -> ShardedPlacement | None:
    """Admit a reduction of one sharded operand; return the result placement.

    If the shard dim is kept, every core reduces only its own shard, so the
    result is sharded along the shard dim's position in the output, and each
    output shard is written by the core that read the matching input shard.

    If the shard dim is reduced, the reduction is cross-shard and is planned
    for the ring (``_plan_cross_shard_reduction``); the result is unsharded
    and None is returned.
    """
    name = op.get_name()
    if (
        not isinstance(op.data, Reduction)
        or op.data.reduction_type not in _SHARD_LOCAL_REDUCTIONS
    ):
        kind = getattr(op.data, "reduction_type", type(op.data).__name__)
        raise Unsupported(
            f"{name}: non-unified kernels support shard-local "
            f"{', '.join(_SHARD_LOCAL_REDUCTIONS)} reductions only; got {kind}"
        )
    in_buf = graph.get_buffer(read.name)
    in_size = [int(s) for s in in_buf.get_size()]
    out_size = [int(s) for s in op.get_size()]
    if in_buf.get_dtype() != op.get_dtype():
        raise Unsupported(
            f"{name}: non-unified kernels need operands and result of one "
            f"dtype; got {in_buf.get_dtype()} -> {op.get_dtype()}"
        )
    reduced, kept, dim_symbol = _reduced_input_dims(name, read, write, in_size)
    s = placement.shard_dim
    if in_size[s] % placement.num_shards:
        raise Unsupported(
            f"{name}: non-unified kernels need the shard dim to divide evenly"
        )
    if s not in kept:
        _plan_cross_shard_reduction(op, name, in_size, reduced, dim_symbol, placement)
        return None
    stick_dim = len(in_size) - 1
    if stick_dim in reduced:
        raise Unsupported(
            f"{name}: non-unified reductions over the stick dim are not "
            "supported (they change the device layout of each shard)"
        )
    # Output position of the shard dim: kept dims appear in the write index
    # in input order, and keepdim inserts size-1 dims for reduced ones.
    out_strides = [math.prod(out_size[d + 1 :]) for d in range(len(out_size))]
    shard_sym = kept[s]
    coeff = write.index.coeff(shard_sym)
    out_dims = [
        d
        for d, extent in enumerate(out_size)
        if out_strides[d] == coeff and extent == in_size[s]
    ]
    if len(out_dims) != 1:
        raise Unsupported(
            f"{name}: cannot find the shard dim of the reduction result in "
            f"{write.index} (output shape {out_size})"
        )
    out_shard_dim = out_dims[0]
    expected = s if len(out_size) == len(in_size) else s - sum(r < s for r in reduced)
    if out_shard_dim != expected:
        raise Unsupported(
            f"{name}: reduction result puts shard dim {s} at {out_shard_dim}, "
            f"expected {expected}"
        )
    out_placement = ShardedPlacement(
        shard_dim=out_shard_dim,
        num_shards=placement.num_shards,
        shard_device_bytes=_shard_device_bytes(
            out_size, op.get_dtype(), out_shard_dim, placement.num_shards
        ),
    )
    _check_shard_layout(name, in_size, op.get_dtype(), placement)
    _check_shard_layout(name, out_size, op.get_dtype(), out_placement)
    op.shard_pinned_splits = {shard_sym: placement.num_shards}
    return out_placement


def _describe_placements(placements: set[ShardedPlacement]) -> str:
    return " and ".join(
        f"Shard({p.shard_dim}) over {p.num_shards} shards "
        f"({p.shard_device_bytes} B each)"
        for p in sorted(
            placements,
            key=lambda p: (p.shard_dim, p.num_shards, p.shard_device_bytes),
        )
    )


def _require_replica_source(graph: GraphLowering, name: str, dep: MemoryDep) -> None:
    """An unsharded operand of a sharded pointwise op must be a read-only
    replica that exists before the kernel runs: a scalar constant or a graph
    input. Anything computed in the graph would be placed (LX, HBM pool) by
    planning passes that know nothing about the shard plan, so it fails closed.
    """
    if dep.name in graph.graph_inputs or isinstance(
        graph.try_get_buffer(dep.name), SpyreConstantFallback
    ):
        return
    raise Unsupported(
        f"{name}: non-unified kernels admit an unsharded operand only when it "
        f"is a scalar constant or a graph input; {dep.name} is computed in the "
        "graph (compute it outside the compiled function and pass it in)"
    )


def _require_replica_read(
    graph: GraphLowering,
    name: str,
    dep: MemoryDep,
    write: MemoryDep,
    size: list[int],
) -> None:
    """The unsharded operand must be the op's shape, or broadcast to it.

    The operand keeps ordinary unified addressing: with the split pinned to
    the shard dim, core c reads the part of it that matches its shard (the
    whole operand when it does not span the shard dim: a scalar, or a bias
    broadcast along it). That is the addressing every unified multi-core
    kernel already uses, and the pinned split is checked against the same
    work-division constraints as any other split.

    What is admitted: right-aligned dims equal to the op's or 1, and an index
    built from the write's own iteration variables. Indirect reads (an index
    with a variable the write does not have) fail closed; affine reads over
    the write's own variables, including a permuted or flipped graph input of
    the op's shape, use ordinary unified addressing.
    """
    buf = graph.get_buffer(dep.name)
    shape = [int(s) for s in buf.get_size()] if buf is not None else None
    if (
        shape is None
        or len(shape) > len(size)
        or any(
            got not in (1, want)
            for got, want in zip(shape, size[len(size) - len(shape) :])
        )
        or not dep.index.free_symbols <= write.index.free_symbols
    ):
        raise Unsupported(
            f"{name}: non-unified kernels read an unsharded operand ({dep.name}, "
            f"shape {shape}) only as the op's shape {size} or broadcast to it "
            "(no indirect reads)"
        )


def plan_sharded_hbm(graph: GraphLowering) -> None:
    """Pin the work split of ops on sharded operands to their shard plan.

    Admits two kinds of op whose sharded reads all have one placement (no
    redistribution between placements):

    - same-shape pointwise ops that read each sharded operand with the write's
      index; the output gets the same placement. They may also read unsharded
      operands as read-only replicas (see ``_require_replica_source`` and
      ``_require_replica_read``): a scalar constant, or a graph input of the
      op's shape or broadcast to it;
    - shard-local reductions (see ``_plan_shard_local_reduction``) of one
      sharded operand; the output gets the remapped placement.

    The output is allocated sharded. The split is ``{shard_dim: num_shards}``;
    ``work_distribution`` commits it instead of choosing one, and LX planning
    keeps it fixed.
    """
    placements = graph_placements(graph)
    if not placements:
        return
    outputs = set(graph.get_output_names())
    # Cross-shard reductions planned for the ring: they read sharded storage
    # but produce an ordinary (unsharded) result.
    ring_reductions: set[str] = set()
    for op in graph.operations:
        if not isinstance(op, ComputedBuffer):
            continue
        target = _mutation_target(op)
        if target is not None and target in placements:
            raise Unsupported(
                f"{op.get_name()}: in-place writes into sharded tensors are not "
                "supported by non-unified kernels"
            )
        rw = op_read_writes(op)
        reads = [dep for dep in rw.reads if isinstance(dep, MemoryDep)]
        # An unsharded operand mutated in place by a sharded value is rejected
        # downstream by the sharded-intermediate guard (the mutation's
        # functional intermediate is never a graph output). A change that
        # admits sharded intermediates inherits the obligation to keep that
        # case failing closed.
        sharded = [placements[d.name] for d in reads if d.name in placements]
        if not sharded:
            continue
        name = op.get_name()
        is_reduction = isinstance(op.data, Reduction)
        if not isinstance(op.data, Pointwise) and not is_reduction:
            raise Unsupported(
                f"{name}: non-unified kernels support pointwise ops and "
                "shard-local reductions only"
            )
        if not all(isinstance(s, (int, sympy.Integer)) for s in op.get_size()):
            raise Unsupported(
                f"{name}: non-unified kernels need static shapes; compile with "
                "dynamic=False"
            )
        placement = sharded[0]
        if len(set(sharded)) > 1:
            raise Unsupported(
                f"{name}: non-unified kernels cannot combine operands sharded "
                f"differently ({_describe_placements(set(sharded))}); moving "
                "data between placements is not supported"
            )
        unsharded = [dep for dep in reads if dep.name not in placements]
        if unsharded and is_reduction:
            raise Unsupported(
                f"{name}: non-unified kernels admit an unsharded operand only "
                f"in pointwise ops; this reduction also reads {unsharded[0].name}"
            )
        for dep in unsharded:
            _require_replica_source(graph, name, dep)
        if any(not isinstance(dep, MemoryDep) for dep in rw.reads):
            raise Unsupported(
                f"{name}: non-unified kernels need plain pointwise reads "
                "(no aliasing or external indexing)"
            )
        if not config.bundle_symbolic_args:
            # The compute step must bind no I/O addresses: every core's start
            # address is a bundle input_arg, so a launch-time-bound kernel
            # cannot express the per-core shard addresses.
            raise Unsupported(
                f"{name}: non-unified kernels need BUNDLE_SYMBOLIC_ARGS=1"
            )
        _check_core_match(placement)
        if is_reduction:
            write = next(iter(rw.writes))
            if len(reads) != 1 or not isinstance(write, MemoryDep):
                raise Unsupported(
                    f"{name}: non-unified reductions need exactly one sharded operand"
                )
            out_placement = _plan_shard_local_reduction(
                graph, op, reads[0], write, placement
            )
            if out_placement is not None:
                placements[name] = out_placement
            else:
                ring_reductions.add(name)
            continue
        sharded_reads = [dep for dep in reads if dep.name in placements]
        for dep in sharded_reads:
            buf = graph.get_buffer(dep.name)
            if buf is None or list(buf.get_size()) != list(op.get_size()):
                raise Unsupported(
                    f"{name}: non-unified kernels need every sharded operand "
                    "to have the op's shape (no views of sharded tensors)"
                )
        write = next(iter(rw.writes))
        if not isinstance(write, MemoryDep) or any(
            d.index != write.index for d in sharded_reads
        ):
            raise Unsupported(
                f"{name}: non-unified kernels need sharded operands read element "
                "for element (no broadcast or reindexing)"
            )
        size = [int(s) for s in op.get_size()]
        if len(size) == 0 or size[placement.shard_dim] % placement.num_shards:
            raise Unsupported(
                f"{name}: non-unified kernels need the shard dim to divide evenly"
            )
        dtypes = {graph.get_buffer(d.name).get_dtype() for d in reads}
        if dtypes != {op.get_dtype()}:
            raise Unsupported(
                f"{name}: non-unified kernels need operands and result of one "
                f"dtype; got {sorted(map(str, dtypes))} -> {op.get_dtype()}"
            )
        _check_shard_layout(name, size, op.get_dtype(), placement)
        sym = _shard_symbol(write, size, placement.shard_dim)
        for dep in unsharded:
            _require_replica_read(graph, name, dep, write, size)
        op.shard_pinned_splits = {sym: placement.num_shards}
        placements[name] = placement
    # A sharded result is allocated as sharded storage only at the graph
    # boundary; an intermediate would be placed in LX or the HBM pool as
    # unified storage. Not supported yet: fail closed.
    for op in graph.operations:
        name = op.get_name() if isinstance(op, ComputedBuffer) else None
        if (
            name in placements
            and name not in outputs
            and name not in graph.graph_inputs
        ):
            raise Unsupported(
                f"{name}: non-unified kernels need every sharded result to be a "
                "graph output (sharded intermediates are not supported)"
            )
    # Only pinned non-unified ops may read sharded storage. Any other
    # consumer (a fallback op, an unshaped buffer) would compile as a unified
    # kernel over sharded storage and fail only at launch, with an error that
    # names neither the op nor the sharding: reject it here instead.
    for op in graph.operations:
        if op.get_name() in placements or op.get_name() in ring_reductions:
            continue
        rw = op_read_writes(op)
        if any(dep.name in placements for dep in rw.reads):
            raise Unsupported(
                f"{op.get_name()}: non-unified kernels support pointwise ops and "
                "shard-local reductions on sharded operands only (this consumer "
                "of sharded storage is not one)"
            )
