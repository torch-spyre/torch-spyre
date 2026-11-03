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

# mypy: allow-untyped-defs
"""Prototype frontend for `for_each_tile` (torch-spyre#3965, working-set reduction).

`for_each_tile` is `scan` with the tiling made explicit: ONE co-indexed loop level
that reduces every operand to a per-step tile -- a `narrow` view, a whole
invariant, or a gathered pool row -- threads an optional carry, and optionally
lays each step's result tile back into a full-size output along one axis.
"""

import contextlib
import enum
import platform
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from collections.abc import Sequence

import torch
import torch._prims_common as utils
from torch._higher_order_ops.scan import scan, scan_op
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import (
    keystr,
    tree_flatten,
    tree_leaves,
    tree_map_with_path,
    tree_unflatten,
)

from ..errors import Unsupported
from ..logging_utils import get_inductor_logger

__all__ = ["Gather", "for_each_tile"]

logger = get_inductor_logger("wsr.for_each_tile")


@dataclass(frozen=True)
class Gather:
    """Tile spec for a POOLED operand: step `i` sees `pool.index_select(axis, index[i])`.

    `index` is a contiguous 1-D integer tensor and its length corresponds to the
    loop's trip count.
    """

    axis: int
    index: torch.Tensor


class Kind(enum.Enum):
    SLICE = "slice"
    GATHER = "gather"
    INVARIANT = "invariant"


@dataclass(frozen=True)
class TileSpec:
    """How one operand is reduced to its per-step tile.

    `kind` picks the reduction: SLICE narrows the operand, GATHER indexes a pool with
    `index`, INVARIANT passes it whole.

    `dim` is the tiled axis for SLICE and the pool axis for GATHER.

    `tile_size` is the tile's shape: a per-dimension vector of the operand's own rank,
    equal to the operand's shape except on the tiled axis. The notation is aimed to be
    similar to the one used in the sibling `tile.py`.
    Note: tiles are RANK-PRESERVING (`narrow`, not `select`).

    `num_tiles` is the operand's own tile count: `shape[dim] // tile_size[dim]` for
    SLICE, `len(index)` for GATHER, 1 for an INVARIANT operand, which is not tiled at
    all.

    `index` is the GATHER index table, `None` for the other kinds.
    """

    kind: Kind
    dim: int = 0
    tile_size: tuple[int, ...] = ()
    num_tiles: int = 1
    index: torch.Tensor | None = None

    @property
    def extent(self) -> int:
        """The tile's size along the tiled axis."""
        return self.tile_size[self.dim] if self.tile_size else 1


DimSpec = Union[int, None, Gather]  # noqa: UP007  # `int | None | Gather` needs Gather at runtime


def _check_combine_graph(gm: torch.fx.GraphModule, name: str) -> None:
    """Validate the graph already traced by scan, without executing it again.

    Read-only operands may share storage. Writes through views, keyword ``out=``
    arguments and Tensor[] arguments must all be checked; the upstream helper
    only examines positional Tensor arguments. Check nested HOP bodies too.
    """

    def values(arg):
        return tree_leaves(torch.fx.node.map_arg(arg, lambda n: n.meta.get("val")))

    def storage(t):
        return StorageWeakRef(t._typed_storage())

    inputs = {
        storage(t)
        for n in gm.graph.find_nodes(op="placeholder")
        for t in values(n)
        if isinstance(t, torch.Tensor)
    }
    outputs = set()
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            attr = gm
            for part in node.target.split("."):
                attr = getattr(attr, part)
            if isinstance(attr, torch.Tensor):
                # Real-mode captures are just as unsafe as FakeTensor captures.
                # Also reject literal tensor constants: the graph cannot tell
                # them apart from external tensors. Factories such as full/ones
                # emit operators instead and remain supported.
                raise Unsupported(
                    "for_each_tile() body closes over a tensor or creates a "
                    f"literal tensor constant ({name}.{node.target}); pass it "
                    "as an operand (dims=None), or use a tensor factory such as "
                    "torch.full instead of torch.tensor"
                )
            if isinstance(attr, torch.fx.GraphModule):
                _check_combine_graph(attr, f"{name}.{node.target}")
        elif node.op == "output":
            for t in values(node.args[0]):
                if not isinstance(t, torch.Tensor):
                    continue
                st = storage(t)
                if st in inputs or st in outputs:
                    raise Unsupported(
                        f"for_each_tile() body in {name} returns an input, a view "
                        "of one, or outputs sharing storage; return a clone instead"
                    )
                outputs.add(st)
        elif node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            for i, arg in enumerate(node.target._schema.arguments):
                if not arg.is_write:
                    continue
                value = (
                    node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
                )
                if any(
                    isinstance(t, torch.Tensor) and storage(t) in inputs
                    for t in values(value)
                ):
                    raise Unsupported(
                        f"for_each_tile() body in {name} writes to an input in place "
                        f"through {node.target}; write to a new tensor instead"
                    )


def _scan_in_make_fx(combine_fn, init, xs, lifted, reverse):
    """Emit the ``scan`` HOP straight into the active make_fx trace.

    ``scan()`` called outside Dynamo re-enters ``torch.compile``. A
    ``for_each_tile`` reached from an AOT decomposition is exactly that case, and
    every nested level then costs a full Dynamo inline stack (~180 C-recursion
    levels each on CPython 3.12), on top of the outer compile's stack. CPython
    caps the C budget at 800 on s390x, so a GQA SDPA nest raises
    ``RecursionError`` there. Under make_fx, ``scan_op``'s proxy-mode rule traces
    ``combine_fn`` itself, at ~16 levels per nest level, so skip Dynamo.

    Unlike Dynamo, make_fx does not lift closures: a tensor ``combine_fn`` reads
    from an enclosing trace would be baked into the subgraph as a constant. Every
    operand read whole each step is therefore passed in ``lifted`` as an
    ``additional_input``, and ``_check_combine_graph`` rejects a body that still
    closes over a traced tensor.
    """
    leaves_init, spec_init = tree_flatten(init)
    leaves_xs, spec_xs = tree_flatten(xs)
    if reverse:
        leaves_xs = [x.flip(0) for x in leaves_xs]
    num_init, num_xs = len(leaves_init), len(leaves_xs)
    y_spec = []

    def flat_combine_fn(*args):
        carry = tree_unflatten(list(args[:num_init]), spec_init)
        sliced = tree_unflatten(list(args[num_init : num_init + num_xs]), spec_xs)
        next_carry, y = combine_fn(carry, sliced, args[num_init + num_xs :])
        carry_leaves, carry_spec = tree_flatten(next_carry)
        if carry_spec != spec_init:
            raise Unsupported(
                "for_each_tile() next_carry must have the same pytree structure "
                f"as init; got {carry_spec}, expected {spec_init}"
            )
        y_leaves, spec = tree_flatten(y)
        if any(not isinstance(t, torch.Tensor) for t in (*carry_leaves, *y_leaves)):
            raise Unsupported("for_each_tile() carry and output leaves must be Tensors")
        y_spec[:] = [spec]
        return [*carry_leaves, *y_leaves]

    tracer = get_proxy_mode().tracer
    flat_out = scan_op(flat_combine_fn, leaves_init, leaves_xs, tuple(lifted))
    # trace_scan has just appended this node; its first argument is the body graph.
    scan_node = next(n for n in reversed(tracer.graph.nodes) if n.target is scan_op)
    name = scan_node.args[0].target
    _check_combine_graph(getattr(tracer.root, name), name)

    final_carry = tree_unflatten(list(flat_out[:num_init]), spec_init)
    ys = list(flat_out[num_init:])
    if reverse:
        ys = [y.flip(0) for y in ys]
    return final_carry, tree_unflatten(ys, y_spec[0])


# CPython 3.12/3.13 C recursion cap on s390x (10000 on x86).
_S390X_CPYTHON_C_RECURSION_LIMIT = 800

# CPython 3.12/3.13 cap C recursion at 800 on s390x (10000 on x86). Outside Dynamo
# (eager, or an AOT decomposition inside an outer compile), `scan` re-enters
# `torch.compile` once, which inlines each nested level at ~180 C levels, so a deep nest
# exceeds 800 on s390x. Dynamo's C-recursion RAII resets the remaining budget to
# this value at each frame it compiles. Must be > 800. See torch-spyre#4973.
# Measured on s390x (py3.12): worst nest needs 1076, 1200 is the smallest passing
# value; 1800 = 1.5 x 1200.
_S390X_SCAN_C_RECURSION_LIMIT = 1800


def _scan_c_recursion_target(machine: str, py: tuple[int, int]) -> int | None:
    """Return the C-recursion budget for the nested `scan` compile.

    Args:
        machine: `platform.machine()` of the host.
        py: `(major, minor)` of the running interpreter.

    Returns:
        The budget to install, or None where it must stay untouched: off s390x it
        would lower the platform default (and warn), and outside 3.12/3.13 Dynamo's
        C-recursion RAII does not exist.
    """
    if machine != "s390x" or not (3, 12) <= py < (3, 14):
        return None
    return _S390X_SCAN_C_RECURSION_LIMIT


@contextlib.contextmanager
def _fresh_c_recursion_budget():
    """Give the nested `scan` compile its own C-recursion budget on s390x."""
    target = _scan_c_recursion_target(platform.machine(), sys.version_info[:2])
    if target is None:
        yield
        return
    previous = torch._dynamo.get_recursion_limit()  # -1 when unset
    if previous >= target:
        yield
        return
    # `c_recursion_limit` is a process-global C static: compiles running on other
    # threads see this value until it is restored.
    torch._dynamo.set_recursion_limit(target)
    try:
        yield
    finally:
        torch._dynamo.set_recursion_limit(previous)  # -1 restores "unset"


def _tile_size_vector(shape, dim: int, extent: int) -> tuple[int, ...]:
    """The operand's shape with the tiled axis narrowed to `extent`."""
    tile_size = list(shape)
    tile_size[dim] = extent
    return tuple(tile_size)


def _normalize_in_specs(operands, dims, tile_size: int) -> tuple[list[TileSpec], int]:
    """Resolve the per-operand specs and the loop's trip count."""
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError(
            f"for_each_tile() tile_size must be a positive int, got {tile_size!r}"
        )
    if isinstance(dims, list):
        raise TypeError(
            f"for_each_tile() dims must be a tuple, got a list: dims={tuple(dims)!r}"
        )
    per_operand: Sequence[DimSpec] = (
        dims if isinstance(dims, tuple) else (dims,) * len(operands)
    )
    if len(per_operand) != len(operands):
        raise ValueError(
            f"for_each_tile() got {len(per_operand)} dims entries for "
            f"{len(operands)} operands"
        )

    specs: list[TileSpec] = []
    # Every tiled operand is walked by ONE counter, so they must all yield the same
    # number of tiles: x[3, 4] tiled by 2 on dim 1 goes with y[4, 5] tiled by 2 on
    # dim 0, and with a 2-page gather, but not with y[6, 5].
    num_tiles: int | None = None
    num_tiles_of: int | None = None

    for k, (o, d) in enumerate(zip(operands, per_operand)):
        if not isinstance(o, torch.Tensor):
            raise ValueError(f"for_each_tile() operand {k} is not a Tensor: {type(o)}")
        if d is None:
            specs.append(TileSpec(Kind.INVARIANT, tile_size=tuple(o.shape)))
            continue
        if isinstance(d, Gather):
            idx = d.index
            if not isinstance(idx, torch.Tensor) or idx.ndim != 1:
                raise ValueError(
                    f"for_each_tile() Gather index for operand {k} must be a 1-D "
                    f"tensor, got {tuple(idx.shape) if isinstance(idx, torch.Tensor) else type(idx)}"
                )
            if idx.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"for_each_tile() Gather index for operand {k} must be int32 or "
                    f"int64, got {idx.dtype}"
                )
            # The frontend will not make the index tensor, e.g., a page table,
            # contiguous itself - it is the responsibility of the user.
            if not idx.is_contiguous():
                raise ValueError(
                    f"for_each_tile() Gather index for operand {k} must be contiguous, "
                    f"got strides {tuple(idx.stride())}; pass index.contiguous()"
                )
            axis = utils.canonicalize_dim(o.ndim, d.axis)
            spec = TileSpec(
                Kind.GATHER,
                axis,
                _tile_size_vector(o.shape, axis, 1),
                idx.shape[0],
                idx,
            )
        elif isinstance(d, int) and not isinstance(d, bool):
            axis = utils.canonicalize_dim(o.ndim, d)
            length = o.shape[axis]
            if length % tile_size != 0:
                raise ValueError(
                    f"for_each_tile() operand {k} has size {length} along dim {d}, "
                    f"which is not a multiple of tile_size={tile_size} (ragged tiles "
                    f"are not supported)"
                )
            # Also state it as a fact. The check above guards it, but recording
            # Mod(length, tile_size) in the shape env is what lets the
            # simplifier fold tile_size * (length // tile_size) back to length
            # when the length is symbolic. A SymInt passes isinstance(length,
            # int), so this must not be guarded by it.
            torch._check(length % tile_size == 0)
            spec = TileSpec(
                Kind.SLICE,
                axis,
                _tile_size_vector(o.shape, axis, tile_size),
                length // tile_size,
            )
        else:
            raise ValueError(
                f"for_each_tile() dims entry {k} must be an int, None or Gather, "
                f"got {d!r}"
            )

        if num_tiles is None:
            num_tiles, num_tiles_of = spec.num_tiles, k
        elif spec.num_tiles != num_tiles:
            raise ValueError(
                f"for_each_tile() operands must yield the same number of tiles: "
                f"operand {num_tiles_of} yields {num_tiles}, operand {k} yields "
                f"{spec.num_tiles}"
            )
        specs.append(spec)

    if num_tiles is None:
        raise ValueError(
            "for_each_tile() needs at least one sliced or gathered operand; a "
            "counter-only loop cannot be traced (see torch-spyre#3965, I1)"
        )
    # With no sliced operand nothing reads tile_size: a gathered step takes one pool
    # row, and the trip count comes from the index table. Silently ignoring the
    # argument would let a wrong one look right.
    if tile_size != 1 and all(s.kind is not Kind.SLICE for s in specs):
        raise ValueError(
            f"for_each_tile() got tile_size={tile_size} with no sliced operand; a "
            f"gathered step takes one pool row, so tile_size must be 1"
        )
    return specs, num_tiles


def _movedim(t: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    """Mode the tile dimension to the right place.

    Moving the tiled axis is a pure stride permutation, i.e. a view, so it should
    never materialize any copies.
    """
    return t if src == dst else torch.movedim(t, src, dst)


def _xs_leaf(operand: torch.Tensor, spec: TileSpec) -> torch.Tensor:
    """The scan `xs` leaf for one operand: [num_tiles, ...], scanned along dim 0."""
    if spec.kind is Kind.GATHER:
        if spec.index is None:
            raise AssertionError("GATHER spec without an index table")
        return spec.index
    moved = _movedim(operand, spec.dim, 0)
    # Trim to a whole number of tiles, then split. Both are views, so nothing
    # copies, and for a concrete length the trim drops nothing.
    #
    # The trim is what keeps the extent a literal when the length is symbolic:
    # given S a view cannot prove (S // G) * G == S and re-derives the extent as
    # S // (S // G), which is 100 at S=100. Given G * (S // G) the trailing
    # factor cancels by gcd to the literal G.
    #
    # `torch.unflatten`, not `Tensor.unflatten`: the method has a Python body
    # (`return super().unflatten(...)`, torch/_tensor.py) that dynamo normally
    # never reaches. Under an active `torch.device` mode -- vLLM runs its model
    # inside one -- DeviceContext.__torch_function__ re-dispatches into that body
    # and dynamo cannot trace the `super()` call. The free function has no body.
    tiles = spec.num_tiles
    trimmed = moved.narrow(0, 0, tiles * spec.extent)
    return torch.unflatten(trimmed, 0, (tiles, spec.extent))


def _tile(operand: torch.Tensor, spec: TileSpec, sliced: torch.Tensor) -> torch.Tensor:
    """Turn scan's step slice of the xs leaf back into the operand's own layout."""
    if spec.kind is Kind.GATHER:
        tile = operand.index_select(spec.dim, sliced.reshape(1))
    else:
        tile = _movedim(sliced, 0, spec.dim)
    return torch.ops.spyre.tile_dim_marker(tile, spec.dim)


def _step_counter(like: torch.Tensor) -> torch.Tensor:
    """The int64 scalar `scan` carry both modes use when the caller supplies none."""
    return torch.zeros((), dtype=torch.int64, device=like.device)


def _stacked_to_full(ys: torch.Tensor, dim: int) -> torch.Tensor:
    """Fold scan's leading step axis into the tiled output axis.

    `ys` is [num_tiles, *tile_size]; the result is `tile_size` with `dim` grown to
    `num_tiles * extent`. For dim == 0 this is a view, (4, 2, 6) -> (8, 6) on the same
    storage, since the flatten merges two contiguous leading axes. For dim != 0 the
    flatten crosses the moved axis, which strides cannot express, so it copies the whole
    output ((3, 8, 2) -> (8, 6)). Avoiding this copy requires destination-aware
    lowering that writes tile i into its final slice, separately for each output leaf.

    For dim == 0 this folds with `as_strided` rather than `flatten`. Both give
    the identical tensor, but on a symbolic leading extent `flatten` (and
    `reshape`, and `view`) walk the dims and add `Ne(count, 1)`, which splits the
    compiled artifact at a single tile. `as_strided` does not.

    That removes one source of the split, not the split itself: torch's own
    tensor-metadata extraction evaluates the same predicate eagerly further down,
    so a single-tile call still compiles separately.
    """
    # as_strided takes the strides on trust, so only take it when the leading
    # axes are PROVABLY contiguous. statically_known_true so the test itself
    # installs no guard. Otherwise flatten: slower to specialise, never wrong.
    # as_strided keeps the storage offset, and a fold that only merges the two
    # leading axes is an ordinary view, so eager and autograd see the same
    # tensor either way.
    if dim == 0:
        if statically_known_true(ys.stride(0) == ys.size(1) * ys.stride(1)):
            folded = (ys.size(0) * ys.size(1), *ys.shape[2:])
            return ys.as_strided(folded, ys.stride()[1:])
        # Eager only: a logger call while Dynamo traces is a graph break, which a
        # fullgraph compile turns into an error. A traced fold of scan's stacked
        # output always has provably contiguous leading axes, so the traced path
        # does not reach this branch for an output for_each_tile builds itself.
        if not torch.compiler.is_dynamo_compiling():
            logger.debug(
                "for_each_tile output fold: leading axes of shape %s, strides %s "
                "are not provably contiguous, folding with flatten",
                tuple(ys.shape),
                tuple(ys.stride()),
            )
    return _movedim(ys, 0, dim).flatten(dim, dim + 1)


def for_each_tile(
    body,
    operands,
    *,
    dims,
    tile_size: int,
    init=None,
    out_dim=None,
    reverse: bool = False,
):
    """Run `body` once per tile over a co-indexed tiling of `operands`.

    Args:
        body: ``(carry, tiles) -> (next_carry, out_tile)``. ``tiles`` arrives in
            operand order, each operand already reduced to its per-step tile.
            ``next_carry`` is ignored in map mode (``init=None``) and ``out_tile``
            in reduction mode (``out_dim=None``). Same restriction as ``scan``:
            the body may not alias input to output or output to output.
        operands: flat sequence of every input -- sliced, gathered and invariant.
        dims: per-operand tile spec as a tuple, or a single spec broadcast to all of
            them. ``int d`` slices the operand into ``tile_size``-wide contiguous views
            along ``d``; ``None`` passes it whole every step; ``Gather(axis, index)``
            takes one pool row per step.
        tile_size: the tile's size along each tiled axis. The loop's trip count is
            derived: ``shape[dim] // tile_size`` per sliced operand, ``len(index)`` per
            gathered one. All of them must agree.
        init: carry init (tensor or pytree of tensors); ``None`` means no carry.
        out_dim: ``int d`` lays step ``i``'s tile at ``narrow(d, i*extent, extent)``
            of each returned output leaf. Output tiles may be a tensor or a
            pytree of tensors. ``None`` means the body emits no tile.
        reverse: visit tiles high to low. The output still lands in natural order.

    Returns:
        ``(final_carry, out)``, either of which is ``None`` for the unused mode.

    Tracing contract:
        In a decomposition (under make_fx, outside Dynamo), tensor dependencies
        must be explicit operands, using ``dims=None`` for invariants. Captured
        tensors, including module parameters and literal ``torch.tensor`` constants,
        are rejected; construct constants before the loop and pass them in, or use
        factories such as ``torch.full`` inside the body. Python scalar constants
        are specialized at trace time. Derive symbolic sizes from the body inputs.

        The carry must retain init's pytree structure and tensor metadata. Inputs
        to the scan may share storage but may not be written; outputs must not
        alias them or each other. Checks include views, keyword ``out=`` writes,
        tensor-list writes, and nested body graphs. Fresh local temporaries may be
        mutated. All used carry/output leaves must be tensors.

        The body must be a pure tensor computation. Arbitrary Python side effects
        (e.g. changing a counter, list or object attribute) are NOT checked or
        replayed per iteration by make_fx, and can silently change results. Use a
        tensor carry for evolving state, and tensor operations for runtime control
        flow. These graph checks also rely on operators declaring accurate schemas.
    """
    operands = tuple(operands)
    # A for_each_tile introduced by an AOT decomposition inherits the existing
    # FakeTensorMode instead of passing through Dynamo's scalar-capture setup.
    # Inductor's scan-to-while-loop pass calls item() on its scalar induction
    # variable while tracing, so permit scalar outputs on that mode. The mode is
    # local to the active compilation; direct user HOPs already get this setting
    # from fullgraph capture.
    if not torch.compiler.is_dynamo_compiling():
        fake_mode = torch._guards.detect_fake_mode(operands)
        if fake_mode is not None:
            fake_mode.allow_scalar_outputs = True
    specs, num_tiles = _normalize_in_specs(operands, dims, tile_size)
    if num_tiles == 0:
        raise ValueError("for_each_tile() requires at least one tile")
    if not isinstance(reverse, bool):
        raise TypeError("for_each_tile() reverse must be a bool")
    if init is not None and (
        not tree_leaves(init)
        or any(not isinstance(t, torch.Tensor) for t in tree_leaves(init))
    ):
        raise ValueError("for_each_tile() init must be a non-empty pytree of Tensors")

    if isinstance(out_dim, Gather):
        # A scattered write is only meaningful with a destination to scatter into.
        # There is a plan to implement an `output` parameter that the user can
        # provide for an in-place update. Not yet implemented.
        raise NotImplementedError(
            "for_each_tile() out_dim=Gather is not supported yet."
        )
    if init is None and out_dim is None:
        raise ValueError("for_each_tile() needs init= (reduction) or out_dim= (map)")

    map_mode = init is None
    if map_mode:
        # scan requires a carry, so map mode (aka using `for_each_tile` similar to `map`)
        # carries the tile counter instead.
        ref = next(o for o, s in zip(operands, specs) if s.kind is not Kind.INVARIANT)
        scan_init = _step_counter(ref)
    else:
        scan_init = init

    # WORKAROUND: A `scan` whose TOTAL flat output is a single tensor does not lower: Inductor's
    # pattern matcher cannot tell a packed one-element return from a 1:1 replacement and
    # leaves the node's `getitem` users on the replacement value, so lowering hits
    # `<built-in function getitem> is not an OpOverload`.
    # A reduction level with one carry leaf and no per-step output is exactly that
    # shape, so it carries a step counter beside its accumulator to make the count two.
    # An extra CARRY leaf, not an extra `ys` leaf: the counter is a scalar the body
    # increments, whereas a padded per-step output costs a stacked buffer for the whole
    # trip count and an `aten.copy_` in the innermost body graph.
    init_leaves = [] if map_mode else tree_leaves(init)
    count_mode = out_dim is None and len(init_leaves) == 1
    if count_mode:
        scan_init = (_step_counter(init_leaves[0]), init)

    xs = tuple(
        _xs_leaf(o, s) for o, s in zip(operands, specs) if s.kind is not Kind.INVARIANT
    )

    # Operands every step reads whole: invariants, and the pools Gather rows index.
    lifted_idx = [i for i, s in enumerate(specs) if s.kind is not Kind.SLICE]

    def combine_fn(carry, sliced, lifted):
        # In map mode, the step counter is the whole carry.
        if count_mode:
            step, carry = carry
        elif map_mode:
            step, carry = carry, None
        whole = dict(zip(lifted_idx, lifted))
        it = iter(sliced)
        tiles = tuple(
            whole[i]
            if s.kind is Kind.INVARIANT
            else _tile(whole.get(i, o), s, next(it))
            for i, (o, s) in enumerate(zip(operands, specs))
        )
        next_carry, y = body(carry, tiles)
        if map_mode:
            next_carry = step + 1
        elif count_mode:
            next_carry = (step + 1, next_carry)
        return next_carry, (() if out_dim is None else y)

    lifted = [operands[i] for i in lifted_idx]
    if not torch.compiler.is_dynamo_compiling() and get_proxy_mode() is not None:
        # Reached from an AOT decomposition: trace the HOP without Dynamo.
        final_carry, ys = _scan_in_make_fx(combine_fn, scan_init, xs, lifted, reverse)
    else:
        # specialize_float: a Python float closed over by the body
        # is generalized to a SymFloat and scan_op's validate_subgraph_args_types
        # currently only accepts Tensor, int, SymInt.
        # Note: This is currently a shortcoming of `scan`,
        # but shouldn't be any concern for Spyre.
        with contextlib.ExitStack() as ctx:
            if not torch.compiler.is_dynamo_compiling():
                ctx.enter_context(torch._dynamo.config.patch(specialize_float=True))
                ctx.enter_context(_fresh_c_recursion_budget())
            final_carry, ys = scan(
                lambda carry, sliced: combine_fn(carry, sliced, lifted),
                scan_init,
                xs,
                dim=0,
                reverse=reverse,
            )

    if count_mode:
        _, final_carry = final_carry

    if out_dim is None:
        return (None if map_mode else final_carry), None

    def stack_output(path, y):
        rank = y.ndim - 1  # scan adds the leading step axis.
        try:
            dim = utils.canonicalize_dim(rank, out_dim, wrap_scalar=False)
        except IndexError as exc:
            raise IndexError(
                f"for_each_tile() out_dim={out_dim} is invalid for output"
                f"{keystr(path)} with tile rank {rank}: {exc}"
            ) from exc
        return _stacked_to_full(y, dim)

    out = tree_map_with_path(stack_output, ys)
    return (None if map_mode else final_carry), out
