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

from typing import Callable, NamedTuple, TypeVar, Union

import torch
import sympy
from sympy import Expr
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    FixedLayout,
    Loops,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
)
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick
from torch_spyre._inductor.errors import Unsupported

from .ir import FixedTiledLayout, SpyreConstantFallback
from .views import compute_coordinates, matching_dim


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: "FixedTiledLayout"


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")
            res.append(SchedNodeArg(arg, layout))
    return res


def concretize_expr(expr: Union[Expr, int]) -> int:
    """Concretize a sympy expression to a Python int.

    Used at boundaries where concrete values are required (e.g. C++
    constructors that only accept ``int``, comparison operators inside
    algorithms such as work-division and coordinate computation).

    Key invariant: only structural parameters (sizes, strides, split
    counts) are concretized.  Symbolic loop variables inside coordinate
    output expressions are never touched, so the generated coordinate
    expressions remain symbolic and will carry through to the SDSC when
    symbolic SDSC generation is implemented.
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        return V.graph.sizevars.size_hint(expr)
    return int(expr)


def concretize_index(index: sympy.Expr, loop_vars: set) -> sympy.Expr:
    """Replace non-loop symbolic variables in an index expression with concrete values.

    With ``dynamic=True``, the host index may contain symbolic strides. When
    ``normalize_coordinates`` isolates each loop variable's contribution
    by substituting 0 for all other free symbols, the size symbol ``s1``
    is also zeroed.  This function replaces size symbols with their concrete
    hints so that coordinate expressions are structurally identical to static-shape
    compilation while loop variable symbols are preserved.
    """
    size_syms = index.free_symbols - loop_vars
    if not size_syms:
        return index
    subs = {s: V.graph.sizevars.size_hint(s) for s in size_syms}
    result = index.subs(subs)
    return result


def get_mem_deps_from_rw(read_writes: ReadWrites) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")
            res.append(SchedNodeArg(arg, layout))
    return res


def host_coordinates(layout: FixedLayout, dep: MemoryDep) -> list[sympy.Expr]:
    # Concretize size/stride so compute_coordinates can use plain ``<``/``>``
    # comparisons.  var_ranges and index stay symbolic so the *output*
    # coordinate expressions remain symbolic.
    # TODO(issue#1373): remove concretization once compute_coordinates handles
    #              symbolic comparisons natively.
    concrete_size = [concretize_expr(s) for s in layout.size]
    concrete_stride = [concretize_expr(s) for s in layout.stride]
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(concrete_size, concrete_stride, dep.ranges, index)


def _check_stick_expr_supported(stick_expr: sympy.Expr, elems_per_stick: int) -> None:
    """Raise Unsupported for stick expressions may be valid but are not yet supported."""
    is_supported_mod = (
        isinstance(stick_expr, sympy.Mod)
        and len(stick_expr.args[0].free_symbols) == 1
        and stick_expr.args[1] == elems_per_stick
    )
    is_bare_var = stick_expr.is_symbol
    is_zero = stick_expr == sympy.S.Zero
    if not (is_supported_mod or is_bare_var or is_zero):
        raise Unsupported(
            f"Unexpected stick expression {stick_expr!r}: expected "
            f"Mod(var, {elems_per_stick}), a bare variable, or 0"
        )


def device_coordinates(stl: SpyreTensorLayout, dep: MemoryDep) -> list[sympy.Expr]:
    # device_size and stride_map come from the C++ SpyreTensorLayout and are
    # already concrete, so no concretization is needed here.
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    coords = compute_coordinates(
        stl.device_size,
        stl.stride_map,
        dep.ranges,
        index,
    )
    _check_stick_expr_supported(coords[-1], stl.elems_per_stick())
    return coords


def iter_var_id(stick_expr) -> int:
    """Iteration variable index from a stick expr: Mod(d2,64) -> 2, d2 -> 2.
    Returns -1 for constant-zero (scalar/broadcast, no real stick).
    NOTE: this is the loop variable index (suffix of dN), NOT a tensor dimension index."""
    if stick_expr == sympy.S.Zero or not stick_expr.free_symbols:
        return -1
    sym = next(iter(stick_expr.free_symbols))
    name = str(sym)
    i = len(name) - 1
    while i >= 0 and name[i].isdigit():
        i -= 1
    return int(name[i + 1 :])


def iteration_space(n: SchedulerNode) -> dict[sympy.Symbol, sympy.Expr]:
    if isinstance(n.node.data, Pointwise):
        # The iteration space of a Pointwise is that of its output
        return next(iter(n.read_writes.writes)).ranges.copy()
    elif isinstance(n.node.data, Reduction):
        # The iteration space of a Reduction is that of its input
        return next(iter(n.read_writes.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")


def iteration_space_from_op(op: ComputedBuffer) -> dict[sympy.Symbol, sympy.Expr]:
    """Pre-scheduler version of iteration_space: uses op.get_read_writes() instead
    of SchedulerNode.read_writes."""
    rw = op.get_read_writes()
    if isinstance(op.data, Pointwise):
        return next(iter(rw.writes)).ranges.copy()
    elif isinstance(op.data, Reduction):
        return next(iter(rw.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")


_V = TypeVar("_V")

# Type alias for the two-namespace split storage: (output_splits, reduction_splits).
# output_splits is keyed by the symbol's coefficient in the write dep's index.
# reduction_splits is keyed by the symbol's coefficient in the first read dep's index.
# The two dicts use different reference indices so their keys never collide.
ItSpaceSplits = tuple[dict[sympy.Expr, int], dict[sympy.Expr, int]]


def _coeff_splits_from_index(
    splits: dict[sympy.Symbol, _V],
    index: sympy.Expr,
    *,
    skip: "Callable[[_V], bool] | None" = None,
) -> dict[sympy.Expr, _V]:
    """Return a coeff→value dict for symbols with a non-zero coefficient in index.

    The coefficient of a symbol in a flat tensor index expression is stable
    across the pre-scheduling / codegen boundary (same layout strides on both
    sides), so it serves as a symbol-identity key that survives the scheduler's
    renaming.  Symbols absent from index (coeff=0) are not included.

    Entries for which ``skip(value)`` returns True are omitted.
    """
    result: dict[sympy.Expr, _V] = {}
    for sym, value in splits.items():
        if skip is not None and skip(value):
            continue
        coeff = index.coeff(sym)
        if coeff != 0:
            result[coeff] = value
    return result


def splits_by_index_coeff(
    splits: dict[sympy.Symbol, int],
    write_index: sympy.Expr,
    read_index: sympy.Expr,
) -> ItSpaceSplits:
    """Encode a symbol→split dict as a pair of coeff-keyed dicts.

    Output dims (those present in write_index) are encoded using their
    coefficient in write_index.  Reduction dims (absent from write_index) are
    encoded using their coefficient in read_index.  The two dicts form separate
    namespaces so their keys never collide, even when output and reduction dims
    happen to share the same stride value in different tensors.

    Only non-unity splits are stored; 1 is the default on the apply side.
    """
    skip = lambda v: v <= 1  # noqa: E731
    output_splits = _coeff_splits_from_index(splits, write_index, skip=skip)
    # Reduction splits: symbols with coeff==0 in write_index but coeff!=0 in read_index
    reduction_only = {
        sym: val for sym, val in splits.items() if write_index.coeff(sym) == 0
    }
    reduction_splits = _coeff_splits_from_index(reduction_only, read_index, skip=skip)
    return output_splits, reduction_splits


def apply_splits_from_index_coeff(
    coeff_splits: ItSpaceSplits,
    write_index: sympy.Expr,
    read_index: sympy.Expr,
    sched_it_space: dict[sympy.Symbol, sympy.Expr],
) -> dict[sympy.Symbol, int]:
    """Reconstruct a scheduler-symbol→split dict from an ItSpaceSplits pair.

    Output dims (non-zero coeff in write_index) are looked up in
    coeff_splits[0]; reduction dims (zero coeff in write_index) are looked up
    in coeff_splits[1] via their coefficient in read_index.  Symbols not found
    in either dict default to 1.
    """
    output_coeff_splits, reduction_coeff_splits = coeff_splits
    result: dict[sympy.Symbol, int] = {sym: 1 for sym in sched_it_space}
    for sym, size in sched_it_space.items():
        # Skip iteration vars with trivial range.  For symbolic ranges we
        # cannot statically determine triviality (and a symbolic size
        # carries no compile-time guarantee that it is 1), so we assume
        # they are non-trivial — consistent with views.compute_coordinates.
        # TODO(issue#1373): replace with a sympy-aware predicate.
        if isinstance(size, (int, sympy.Integer)) and int(size) <= 1:
            continue
        wc = write_index.coeff(sym)
        if wc != 0:
            if wc in output_coeff_splits:
                result[sym] = output_coeff_splits[wc]
        else:
            rc = read_index.coeff(sym)
            if rc != 0 and rc in reduction_coeff_splits:
                result[sym] = reduction_coeff_splits[rc]
    return result


# The following restickify helpers are used only by the restickify
# but are here to avoid circular dependences in those files


def restickify_device_size(
    old_device_size: list,
    old_sd_outer_dim: int,
    old_sd_host_size: int,
    new_sd_outer_dim: int,
    new_sd_host_size: int,
    stick_size: int,
) -> list:
    """Computes the new device size after a restickify is performed
    moving the stick from old_sd to new_sd."""
    assert new_sd_host_size % stick_size == 0, (
        f"Cannot move stick to dimension with size {new_sd_host_size}: "
        f"without padding since not a multiple of stick_size={stick_size}"
    )
    new_device_size = list(old_device_size)
    new_device_size[-1] = stick_size
    new_device_size[old_sd_outer_dim] = new_sd_host_size // stick_size
    new_device_size[new_sd_outer_dim] = old_sd_host_size
    return new_device_size


def restickify_stride_map(
    old_stride_map: list,
    old_sd_outer_dim: int,
    old_sd_host_stride: int,
    new_sd_outer_dim: int,
    new_sd_host_stride: int,
    stick_size: int,
) -> list:
    """Computes the new stride_map after a restickify is performed moving the stick from old_sd to new_sd."""
    new_stride_map = list(old_stride_map)
    new_stride_map[-1] = new_sd_host_stride
    new_stride_map[old_sd_outer_dim] = new_sd_host_stride * stick_size
    new_stride_map[new_sd_outer_dim] = old_sd_host_stride
    return new_stride_map


def compute_restickify_target_layout(
    stl: SpyreTensorLayout,
    host_layout: FixedLayout,
    target_stick_expr,
    ic: list,
    idc: list,
) -> "SpyreTensorLayout | None":
    """Compute the target STL that results from moving stl's stick to target_stick_expr.
    Returns None if the restickify is infeasible.
    """
    new_sd = matching_dim(ic, target_stick_expr)
    if new_sd is None:
        return None
    host_size = [concretize_expr(s) for s in host_layout.size]
    host_stride = [concretize_expr(s) for s in host_layout.stride]
    old_sd = matching_dim(ic, idc[-1])
    if old_sd is None:
        return None
    old_stick_expr = idc[-1]
    old_stride_map = list(stl.stride_map)
    old_var = next(iter(old_stick_expr.free_symbols))
    new_var = next(iter(target_stick_expr.free_symbols))
    stick_size = get_elem_in_stick(host_layout.dtype)
    old_sd_outer_dim = next(
        (j for j in range(len(idc) - 1) if old_var in idc[j].free_symbols),
        next((j for j in range(len(idc) - 1) if idc[j] == sympy.S.Zero), None),
    )
    if old_sd_outer_dim is None:
        return None
    candidates = [j for j in range(len(idc) - 1) if new_var in idc[j].free_symbols]
    if not candidates:
        return None
    new_sd_outer_dim = candidates[0]
    if host_size[new_sd] % stick_size != 0:
        return None
    device_size = restickify_device_size(
        list(stl.device_size),
        old_sd_outer_dim,
        host_size[old_sd],
        new_sd_outer_dim,
        host_size[new_sd],
        stick_size,
    )
    stride_map = restickify_stride_map(
        old_stride_map,
        old_sd_outer_dim,
        host_stride[old_sd],
        new_sd_outer_dim,
        host_stride[new_sd],
        stick_size,
    )
    return SpyreTensorLayout(device_size, stride_map, stl.device_dtype)


def stick_compatible(coords: "list[list[sympy.Expr]]") -> bool:
    """Return True if all tensors are stick-compatible.

    coords: list of device_coordinates() results, one per tensor.

    Compatible means: the union of stick variables (free symbols in the last
    device coordinate) across all tensors has at most one element, and is
    disjoint from the union of nonstick variables (free symbols in all other
    device coordinates, excluding each tensor's own stick variable).
    """
    stick_vars: set[sympy.Symbol] = set()
    nonstick_vars: set[sympy.Symbol] = set()
    for dc in coords:
        tensor_stick_vars = dc[-1].free_symbols
        stick_vars |= tensor_stick_vars
        for coord in dc[:-1]:
            nonstick_vars |= coord.free_symbols - tensor_stick_vars
    return len(stick_vars) <= 1 and stick_vars.isdisjoint(nonstick_vars)


def compute_restickify_needed(
    in_stl: SpyreTensorLayout,
    in_host: FixedLayout,
    in_dep: MemoryDep,
    out_stl: SpyreTensorLayout,
    out_dep: MemoryDep,
) -> "tuple[bool, SpyreTensorLayout | None]":
    """Determine whether a restickify is needed for one (in_stl, out_stl) pair.

    in_dep and out_dep may differ when the output buffer is accessed with a
    different index than the input (e.g. a transposed read).

    Returns:
      (False, None)   — stick-compatible: no restickify needed
      (True, stl)     — restickify needed, stl is the target STL for the restickified input
      (True, None)    — restickify needed but infeasible
    """
    idc = device_coordinates(in_stl, in_dep)
    out_idc = device_coordinates(out_stl, out_dep)
    assert idc, "device_coordinates returned empty list for input"
    assert out_idc, "device_coordinates returned empty list for output"
    if stick_compatible([idc, out_idc]):
        return False, None
    ic = host_coordinates(in_host, in_dep)
    return True, compute_restickify_target_layout(in_stl, in_host, out_idc[-1], ic, idc)


def replace_computed_buffer_body(
    op: ComputedBuffer,
    new_data: Loops,
    operations: list[Operation],
) -> ComputedBuffer:
    """Replace the body (``data``) of a ``ComputedBuffer`` with ``new_data``.

    ``ComputedBuffer`` is a frozen dataclass, so its ``data`` field cannot be
    mutated in place.  This function constructs a new ``ComputedBuffer`` with
    the updated body and swaps it into ``operations``, copying all metadata
    fields that downstream passes depend on: ``operation_name``, ``origins``,
    ``origin_node``, and the ``_split_size`` / ``_original_*`` fields used by
    ``get_default_sizes_body``.  The ``get_default_sizes_body`` cache is
    cleared on the new buffer so stale size results from the old body are not
    reused.

    Returns the replacement ComputedBuffer.
    """
    new_buf = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=new_data,
        _split_size=op._split_size,
        _original_inner_fn=op._original_inner_fn,
        _original_ranges=op._original_ranges,
        _original_reduction_ranges=op._original_reduction_ranges,
    )
    new_buf.operation_name = op.operation_name
    new_buf.origins = op.origins
    new_buf.origin_node = op.origin_node
    ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)

    op_idx = operations.index(op)
    operations[op_idx] = new_buf
    return new_buf


def lower_pad_sequence(
    arg_fx_node: torch.fx.Node,
    padded_size: list[int],
    device: torch.device,
    dtype: torch.dtype,
    dim: int,
    insert_before: torch.fx.Node,
    orig_stl: SpyreTensorLayout,
    fill_value: float = 0.0,
) -> tuple[Buffer, list[Operation]]:
    """Lower an IR-level pad sequence that extends a buffer along one dimension.

    Allocates a padded buffer of ``padded_size``, fills the pad region with
    ``fill_value``, then copies the original data into offset 0 along ``dim``.
    Only one dimension may differ between ``padded_size`` and the original shape.

    Uses torch.ops.aten.constant_pad_nd which lowers to a 4-op IR sequence:
      1. ComputedBuffer - output buffer allocation (FixedLayout)
      2. SpyreConstantFallback - fill constant (FixedLayout)
      3. ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
      4. ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)

    constant_pad_nd is called with align_to_stick=True to ensure the padding region
    is filled with stick-aligned offsets. This is required because the dim is
    ensured to be a stick dimension here.

    ``orig_stl`` is the ``SpyreTensorLayout`` of the unpadded buffer and is used
    to derive the padded buffer's device layout, preserving the within-stick host
    dimension.  Raises ``RuntimeError`` if the within-stick dimension cannot be
    determined from ``orig_stl``.

    Deduplication of identical constants across multiple pad calls happens later
    at the IR level via dedup_and_promote_constants.

    Returns ``(padded_buf, new_ops)`` where ``padded_buf`` is the allocated buffer
    and ``new_ops`` is the list of new IR operations in topological order.
    """

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    # Count operations before lowering so we can identify newly added ones.
    ops_before = len(graph_lowering.operations)

    original_shape = list(arg_fx_node.meta["val"].shape)
    assert len(padded_size) == len(original_shape), (
        f"lower_pad_sequence: padded_size rank {len(padded_size)} != "
        f"original rank {len(original_shape)}"
    )
    padded_dims = [
        i for i in range(len(padded_size)) if padded_size[i] != original_shape[i]
    ]
    assert padded_dims == [dim], (
        f"lower_pad_sequence: expected exactly dim={dim} to be padded, "
        f"but padded_size={padded_size} differs from original={original_shape} at dims={padded_dims}"
    )
    original_size_dim: int = original_shape[dim]
    pad_extent = padded_size[dim] - original_size_dim
    assert pad_extent > 0, (
        f"lower_pad_sequence: pad_extent={pad_extent} for dim={dim}; "
        f"padded_size={padded_size}, original_size_dim={original_size_dim}"
    )

    # Build pad tuple for constant_pad_nd: (left, right) pairs in reverse dimension order
    # We're padding only one dimension, so most pairs are (0, 0)
    pad_tuple = []
    for i in range(len(original_shape) - 1, -1, -1):
        if i == dim:
            # Pad at the end of this dimension
            pad_tuple.extend([0, pad_extent])
        else:
            pad_tuple.extend([0, 0])

    with fx_graph.inserting_before(insert_before):
        # Single constant_pad_nd call (lowers to 4 IR operations)
        pad_fx = fx_graph.create_node(
            "call_function",
            torch.ops.aten.constant_pad_nd.default,
            args=(arg_fx_node, pad_tuple, fill_value),
            kwargs={"align_to_stick": True},
        )
        pad_fx.meta["val"] = torch.empty(padded_size, dtype=dtype, device=device)

    # Lower the constant_pad_nd node, assigning FixedTiledLayouts immediately.
    # propagate_spyre_tensor_layouts already ran, so the new op keep FlexibleLayout
    # unless we assign here.
    pad_tb = graph_lowering.run_node(pad_fx)
    graph_lowering.env[pad_fx] = pad_tb
    padded_buf = pad_tb.data.data  # TensorBox -> StorageBox -> Buffer

    # Collect all newly added operations (appended at the end of graph.operations).
    new_ops = graph_lowering.operations[ops_before:]

    assert new_ops[0] == padded_buf

    # Verify structure: constant_pad_nd lowers to 4 operations
    #   op0: ComputedBuffer - output buffer allocation (FixedLayout)
    #   op1: SpyreConstantFallback - fill constant (FixedLayout)
    #   op2: ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
    #   op3: ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)
    assert (
        len(new_ops) == 4
        and isinstance(new_ops[0], ComputedBuffer)
        and isinstance(new_ops[0].get_layout(), FixedLayout)
        and isinstance(new_ops[1], SpyreConstantFallback)
        and isinstance(new_ops[1].get_layout(), FixedLayout)
        and isinstance(new_ops[2], ComputedBuffer)
        and isinstance(new_ops[2].get_layout(), MutationLayoutSHOULDREMOVE)
        and isinstance(new_ops[3], ComputedBuffer)
        and isinstance(new_ops[3].get_layout(), MutationLayoutSHOULDREMOVE)
    )

    # --- Build the device layout (SpyreTensorLayout) for the padded buffer. ---
    #
    # We need to know two things to construct the padded STL:
    #   1. The "core" host shape — the dimensions that orig_stl was actually
    #      built from.  mm_to_bmm_pass sometimes adds a leading batch=1 dim to
    #      padded_size (the view the matmul inner_fn uses) while leaving the
    #      underlying buffer 2D.  Passing that phantom dim to SpyreTensorLayout
    #      would produce a degenerate 4D device layout with a -1 sentinel stride
    #      for the size-1 device dim, which causes compute_coordinates to emit a
    #      constant nonzero stick offset and normalize_coordinates to assert.
    #      We strip phantom dims by comparing padded_size rank against the host
    #      rank implied by orig_stl: stride_map has one entry per device dim, and
    #      device dims = host dims + 1 (the extra entry is the within-stick dim),
    #      so orig_host_ndim = len(stride_map) - 1.
    #   2. Which host dimension is the within-stick dimension.  SpyreTensorLayout
    #      takes an explicit dim_order whose last element names the within-stick
    #      host dim; we must carry this over from the original buffer so that the
    #      padded buffer's device coordinates use the same stick dimension.  We
    #      identify it by matching orig_stl.stride_map[-1] (the within-stick
    #      element stride, always 1 for contiguous layouts) against the original
    #      buffer's host strides.

    # Step 1 — strip phantom batch dims to get the core host shape.
    orig_host_ndim = len(list(orig_stl.stride_map)) - 1
    n_phantom = len(padded_size) - orig_host_ndim
    padded_core = padded_size[n_phantom:]

    # Step 2 — identify the within-stick host dim in the view (which may include
    # phantom leading dims) by matching the within-stick element stride.
    sm_last = int(list(orig_stl.stride_map)[-1])
    orig_host_stride = list(arg_fx_node.meta["val"].stride())
    within_stick_dim_view = next(
        (i for i, s in enumerate(orig_host_stride) if int(s) == sm_last), None
    )
    if within_stick_dim_view is None:
        raise RuntimeError(
            f"lower_pad_sequence: cannot determine within-stick host dimension for "
            f"buffer {arg_fx_node.name!r}: orig_stl.stride_map[-1]={sm_last} not found "
            f"in view strides {orig_host_stride}.  orig_stl={list(orig_stl.device_size)} "
            f"stride_map={list(orig_stl.stride_map)}, padded_size={padded_size}"
        )

    # Step 3 — translate the within-stick dim index from view space to core space
    # (subtract the number of phantom dims that were stripped in step 1).
    within_stick_dim_core = within_stick_dim_view - n_phantom

    # Step 4 — build dim_order for SpyreTensorLayout: all non-stick dims in their
    # natural order, followed by the within-stick dim last.  This tells the STL
    # constructor which host dim maps to the innermost device (within-stick) axis.
    dim_order_core = [
        i for i in range(len(padded_core)) if i != within_stick_dim_core
    ] + [within_stick_dim_core]

    # Step 5 — compute row-major strides for the padded core shape.  These are
    # host strides, not device strides; SpyreTensorLayout derives the device
    # layout (sticks, rows, …) from the host shape + dim_order.
    core_stride = [1] * len(padded_core)
    for i in range(len(padded_core) - 2, -1, -1):
        core_stride[i] = core_stride[i + 1] * padded_core[i + 1]

    padded_stl = SpyreTensorLayout(padded_core, core_stride, dtype, dim_order_core)
    host_layout = padded_buf.layout
    padded_buf.layout = FixedTiledLayout(
        host_layout.device,
        host_layout.dtype,
        host_layout.size,
        host_layout.stride,
        padded_stl,
    )

    # LX planning (scratchpad.py) accesses op.origin_node directly on the ComputedBuffer,
    # so we set it here explicitly.
    object.__setattr__(padded_buf, "origin_node", pad_fx)

    # propagate_spyre_tensor_layouts already ran before this pass, so any op
    # lowered here keeps FlexibleLayout unless we assign a FixedTiledLayout
    # immediately. The constant buffer (new_ops[1]) is a scalar tensor (size=[]).
    const_buf = new_ops[1]
    const_layout = const_buf.get_layout()
    const_stl = SpyreTensorLayout(const_layout.size, const_layout.dtype)
    const_buf.layout = FixedTiledLayout(
        const_layout.device,
        const_layout.dtype,
        const_layout.size,
        const_layout.stride,
        const_stl,
    )

    # Mutation ops are intentionally left untouched

    assert (
        len(new_ops) == 4
        and isinstance(new_ops[0].get_layout(), FixedTiledLayout)
        and isinstance(new_ops[1].get_layout(), FixedTiledLayout)
        and isinstance(new_ops[2].get_layout(), MutationLayoutSHOULDREMOVE)
        and isinstance(new_ops[3].get_layout(), MutationLayoutSHOULDREMOVE)
    )

    return padded_buf, list(new_ops)
