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

- same-shape pointwise ops: the result has the operands' placement;
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
from .pass_utils import op_read_writes
from torch_spyre.sharding import _require_exact_int


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
    size = list(tensor.shape)
    ok = (
        tensor._base is None
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


def _check_local_strides(
    name: str, size: list[int], dtype: torch.dtype, placement: ShardedPlacement
) -> None:
    """Reject shardings whose shard-local strides differ from the global ones.

    DeepTools gives each core its own start address, but derives HBM strides
    inside the core's tile from the global shape (N_). The kernel is correct
    only if every device dim has the same stride in the shard's own layout as
    in the global layout, and the shard dim is the outermost device dim (so a
    shard is one contiguous block). Measured counterexamples: 32x128 fp16
    Shard(0) leaves the second stick of every row unwritten; 4x8x512
    Shard(2) would write past the end of its chunk. Fail closed.
    """
    from torch_spyre._C import SpyreTensorLayout

    local_size = list(size)
    local_size[placement.shard_dim] //= placement.num_shards
    global_dev = list(SpyreTensorLayout(size, dtype).device_size)
    local_dev = list(SpyreTensorLayout(local_size, dtype).device_size)
    elems_per_stick = global_dev[-1]

    def strides(dev: list[int]) -> list[int]:
        return [math.prod(dev[i + 1 :]) for i in range(len(dev))]

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
        # ... and every dim a core walks has the same stride in the shard's own
        # layout as in the global one (DeepTools uses the global strides).
        and all(
            gs == ls
            for gs, ls, lo in zip(strides(global_dev), strides(local_dev), local_dev)
            if lo > 1
        )
    )
    if not ok:
        raise Unsupported(
            f"{name}: non-unified kernels need shard-local strides equal to the "
            f"global ones; got shape {size}, shard dim {placement.shard_dim}, "
            f"{placement.num_shards} shards (device sizes {global_dev} vs shard "
            f"{local_dev})"
        )


def _mutation_target(op: ComputedBuffer) -> str | None:
    layout = op.layout
    if not isinstance(layout, MutationLayoutSHOULDREMOVE):
        return None
    target = layout.target
    while isinstance(target, ReinterpretView):
        target = target.data
    return target.get_name() if hasattr(target, "get_name") else None


def verify_non_unified_split(sdsc_spec, shard_kernel_args) -> None:
    """Codegen guard: core c's work slice is shard c, and nothing else splits.

    The pass pins the split; this re-checks the committed division the SDSC
    is emitted from, so a later pass cannot silently re-divide a sharded op.
    """
    n = sdsc_spec.num_cores
    try:
        split_dims = {d: int(v) for d, v in sdsc_spec.work_slices.items() if int(v) > 1}
        core_slices = {
            (c, d): int(expr.subs({sympy.Symbol("core_id"): c}))
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
    # The split dim must be the shard dim: in the unified address model
    # (compute_ops.core_idx_to_slice_offset), core c's slice starts
    # c * strides[dim] / work_slices[dim] elements in; that must be exactly c
    # shards, so each core's tile is one chunk.
    # Local import: compute_ops imports this module, so a module-level
    # import would be circular.
    from .codegen.compute_ops import num_bytes

    for tensor in shard_kernel_args:
        stride = tensor.strides.get(dim)
        if stride is None or tensor.scales.get(dim, 0) <= 0:
            raise Unsupported(
                f"non-unified kernel splits {dim}, which tensor arg "
                f"{tensor.arg_index} does not index"
            )
        nbytes = num_bytes(tensor.data_format)
        slice_bytes = int(stride) // n * nbytes
        if slice_bytes != tensor.non_unified_shard_bytes:
            raise Unsupported(
                f"non-unified kernel splits {dim}: one slice is {slice_bytes} "
                f"bytes of tensor arg {tensor.arg_index}, but one shard is "
                f"{tensor.non_unified_shard_bytes}"
            )


# Reductions whose value over one shard needs nothing from other shards when
# the shard dim is not reduced. ``mean`` divides by the reduced extent, which
# is the same on every core because the reduced dims are not split.
_SHARD_LOCAL_REDUCTIONS = ("sum", "mean", "max", "min")


def _reduced_input_dims(
    name: str, read: MemoryDep, write: MemoryDep, in_size: list[int]
) -> tuple[list[int], dict[int, sympy.Symbol]]:
    """Which input dims a shard-local reduction reduces, and the kept dims' symbols.

    The read index walks a contiguous input: input dim ``d`` is the symbol
    whose coefficient is ``stride(d)`` and whose range is ``in_size[d]``. A
    dim is reduced when its symbol does not appear in the write index.
    Returns the reduced dims and, for each kept dim, its symbol.
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
    return reduced, kept


def _plan_shard_local_reduction(
    graph: GraphLowering,
    op: ComputedBuffer,
    read: MemoryDep,
    write: MemoryDep,
    placement: ShardedPlacement,
) -> ShardedPlacement:
    """Admit a reduction that leaves the shard dim intact; return its placement.

    Every core reduces only its own shard, so the result is sharded along
    the shard dim's position in the output, and each output shard is
    written by the core that read the matching input shard.
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
    reduced, kept = _reduced_input_dims(name, read, write, in_size)
    s = placement.shard_dim
    if in_size[s] % placement.num_shards or s not in kept:
        raise Unsupported(
            f"{name}: non-unified kernels support reductions that keep the "
            "shard dim only (a cross-shard reduction needs a combine across "
            "cores, which is not supported yet)"
        )
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
    _check_local_strides(name, in_size, op.get_dtype(), placement)
    _check_local_strides(name, out_size, op.get_dtype(), out_placement)
    op.shard_pinned_splits = {shard_sym: placement.num_shards}
    return out_placement


def plan_sharded_hbm(graph: GraphLowering) -> None:
    """Pin the work split of ops on sharded operands to their shard plan.

    Admits two kinds of op whose memory reads are all sharded with one
    placement (no broadcast, no redistribution):

    - same-shape pointwise ops that read each operand with the write's index;
      the output gets the same placement;
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
        if any(p != placement for p in sharded) or len(sharded) != len(reads):
            raise Unsupported(
                f"{name}: non-unified kernels need every operand sharded with the "
                "same placement"
            )
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
            placements[name] = _plan_shard_local_reduction(
                graph, op, reads[0], write, placement
            )
            continue
        for dep in reads:
            buf = graph.get_buffer(dep.name)
            if buf is None or list(buf.get_size()) != list(op.get_size()):
                raise Unsupported(
                    f"{name}: non-unified kernels need every sharded operand "
                    "to have the op's shape (no views of sharded tensors)"
                )
        write = next(iter(rw.writes))
        if not isinstance(write, MemoryDep) or any(
            d.index != write.index for d in reads
        ):
            raise Unsupported(
                f"{name}: non-unified kernels need operands read element for "
                "element (no broadcast or reindexing)"
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
        _check_local_strides(name, size, op.get_dtype(), placement)
        sym = _shard_symbol(write, size, placement.shard_dim)
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
        if op.get_name() in placements:
            continue
        rw = op_read_writes(op)
        if any(dep.name in placements for dep in rw.reads):
            raise Unsupported(
                f"{op.get_name()}: non-unified kernels support pointwise ops and "
                "shard-local reductions on sharded operands only (this consumer "
                "of sharded storage is not one)"
            )
