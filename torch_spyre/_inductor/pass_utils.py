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
    MultiOutput,
    Operation,
    Pointwise,
    Reduction,
)
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick
from torch_spyre._inductor.errors import Unsupported

from .ir import FixedTiledLayout
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


def device_coordinates(stl: SpyreTensorLayout, dep: MemoryDep) -> list[sympy.Expr]:
    # device_size and stride_map come from the C++ SpyreTensorLayout and are
    # already concrete, so no concretization is needed here.
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(
        stl.device_size,
        stl.stride_map,
        dep.ranges,
        index,
    )


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


def rebuild_computed_buffer(
    op: ComputedBuffer,
    new_data,
    operations: list[Operation],
) -> ComputedBuffer:
    """Replace ``op`` in ``operations`` with a new ComputedBuffer sharing its layout.

    Preserves all metadata fields required by downstream passes: ``operation_name``,
    ``origins``, ``origin_node``, and the ``_split_size`` / ``_original_*`` fields
    used by ``get_default_sizes_body``.  Clears the ``get_default_sizes_body`` cache
    on the new buffer so stale size results from the old ``data`` are not reused.

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
    fill_cache: dict | None = None,
) -> tuple[Buffer, list[Operation]]:
    """Lower an IR-level pad sequence that extends a buffer along one dimension.

    Allocates a padded buffer of ``padded_size``, fills the pad region with
    ``fill_value``, then copies the original data into offset 0 along ``dim``.

    The pad region extent is ``padded_size[dim] - original_size[dim]`` where
    ``original_size`` is read from ``arg_fx_node.meta["val"].shape``.  This
    works for any pad amount, not only one stick.

    FX nodes created (in order):
      1. spyre.empty(padded_size)                        — uninitialised allocation
      2. spyre.constant(fill_value)                      — scalar constant, on-device (cached)
      3. aten.expand(constant, pad_size)                 — broadcast to fill-region shape; free
      4. aten.clone(expand)                              — on-device broadcast → fill buffer
      5. overwrite(fill_buf, empty, [dim], [fill_offset]) — write pad region
      6. overwrite(orig,     empty, [dim], [0])           — copy original data

    ``pad_size`` equals ``padded_size`` with ``pad_size[dim] = pad_extent``
    where ``pad_extent = padded_size[dim] - original_size[dim]``.
    ``fill_offset = original_size[dim]``.

    ``orig_stl`` is the ``SpyreTensorLayout`` of the unpadded buffer.  The
    padded allocation's ``SpyreTensorLayout`` is derived from it so that the
    within-stick host dimension is preserved: the last entry of
    ``device_coordinates`` for both the original and padded buffers will be
    identical.  This is achieved by recovering the within-stick host dimension
    from ``orig_stl.stride_map[-1]`` (the host stride of the within-stick dim)
    and constructing the padded STL via ``SpyreTensorLayout(padded_size,
    padded_host_stride, dtype, dim_order)`` with the same ``dim_order``.  Falls
    back to ``SpyreTensorLayout(padded_size, dtype)`` when ``orig_stl`` has a
    different number of dimensions than ``padded_size`` (e.g. when
    mm_to_bmm_pass adds a batch dimension).

    ``fill_cache`` maps ``(fill_value, device, dtype)`` to an existing
    ``spyre.constant`` FX node.  On a cache hit that node is reused and not
    re-lowered.  All padding with the same fill value, device, and dtype shares
    one constant node regardless of tensor shape or padded dimension.

    ``insert_before`` is the FX node before which new nodes are inserted.

    Returns ``(padded_buf, new_ops)`` where ``padded_buf`` is the allocated buffer
    and ``new_ops`` is the list of new IR operations in topological order.
    """
    from .propagate_layouts import generic_layout  # deferred to avoid circular import
    from .ir import SpyreConstantFallback  # deferred to avoid circular import

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    # Count operations before lowering so we can identify newly added ones.
    ops_before = len(graph_lowering.operations)

    original_size_dim: int = arg_fx_node.meta["val"].shape[dim]
    pad_extent = padded_size[dim] - original_size_dim
    assert pad_extent > 0, (
        f"lower_pad_sequence: pad_extent={pad_extent} for dim={dim}; "
        f"padded_size={padded_size}, original_size_dim={original_size_dim}"
    )
    fill_offset = original_size_dim

    # Fill-region shape: padded_size with dim replaced by pad_extent.
    pad_size = list(padded_size)
    pad_size[dim] = pad_extent

    cache_key = (fill_value, device, dtype)
    const_is_new = fill_cache is None or cache_key not in fill_cache

    with fx_graph.inserting_before(insert_before):
        # 1. Uninitialised padded buffer.
        empty_fx = fx_graph.create_node(
            "call_function",
            torch.ops.spyre.empty.default,
            args=(padded_size, device, dtype),
        )
        empty_fx.meta["val"] = torch.empty(padded_size, dtype=dtype, device=device)

        # 2. Scalar constant — generated on-device, no DMA (reused if cached).
        if fill_cache is not None and cache_key in fill_cache:
            const_fx = fill_cache[cache_key]
        else:
            const_fx = fx_graph.create_node(
                "call_function",
                torch.ops.spyre.constant.default,
                args=(fill_value, dtype, device),
            )
            const_fx.meta["val"] = fill_value
            if fill_cache is not None:
                fill_cache[cache_key] = const_fx

        # 3. Broadcast to fill-region shape (ExpandView — no allocation).
        expand_fx = fx_graph.create_node(
            "call_function",
            torch.ops.aten.expand.default,
            args=(const_fx, pad_size),
        )
        expand_fx.meta["val"] = torch.empty(pad_size, dtype=dtype, device=device)

        # 4. On-device broadcast copy: clone materialises the fill buffer.
        clone_fx = fx_graph.create_node(
            "call_function",
            torch.ops.aten.clone.default,
            args=(expand_fx,),
        )
        clone_fx.meta["val"] = torch.empty(pad_size, dtype=dtype, device=device)

        # 5. Write fill values into the pad region of empty.
        overwrite_fill_fx = fx_graph.create_node(
            "call_function",
            torch.ops.spyre.overwrite.default,
            args=(clone_fx, empty_fx, [dim], [fill_offset]),
        )
        overwrite_fill_fx.meta["val"] = None

        # 6. Copy original data into offset 0 along dim.
        overwrite_data_fx = fx_graph.create_node(
            "call_function",
            torch.ops.spyre.overwrite.default,
            args=(arg_fx_node, empty_fx, [dim], [0]),
        )
        overwrite_data_fx.meta["val"] = None

    # Lower each node in dependency order, assigning FixedTiledLayouts immediately.
    # propagate_spyre_tensor_layouts already ran, so new ops keep FlexibleLayout
    # unless we assign here.
    #
    # spyre.empty lowers to FallbackKernel + MultiOutput; the MultiOutput is
    # unwrapped from the returned TensorBox to set its layout.
    # spyre.constant lowers to SpyreConstantFallback (single op, ExternKernel subclass).
    # aten.expand lowers to an ExpandView (no Buffer produced, no layout needed).
    # aten.clone lowers to a ComputedBuffer with FlexibleLayout → FixedTiledLayout.
    # overwrite lowers to a ComputedBuffer with MutationLayoutSHOULDREMOVE — left unchanged.
    #
    # Important: layouts must be FixedTiledLayout (an Inductor Layout subclass), NOT
    # bare SpyreTensorLayout.  Inductor's get_layout() raises NotImplementedError on
    # SpyreTensorLayout; aten.expand's lowering calls get_layout() on its input.

    def _assign_layout(buf: Buffer) -> None:
        """Wrap the buffer's current FixedLayout in a FixedTiledLayout."""
        host_layout = buf.layout
        buf.layout = FixedTiledLayout(
            host_layout.device,
            host_layout.dtype,
            host_layout.size,
            host_layout.stride,
            generic_layout(buf),
        )

    empty_tb = graph_lowering.run_node(empty_fx)
    graph_lowering.env[empty_fx] = empty_tb
    padded_buf = empty_tb.data.data  # TensorBox -> StorageBox -> MultiOutput
    assert isinstance(padded_buf, MultiOutput)
    # Build the padded STL preserving the within-stick host dimension of orig_stl.
    # stride_map[-1] is the host stride of the within-stick dimension; find the
    # corresponding dim in the (possibly larger) view's stride list and use it as
    # the last entry of dim_order so that device_coordinates[-1] (the stick
    # coordinate expression) is identical for the original and padded buffers.
    # arg_fx_node.meta["val"].stride() gives the strides of the view that the
    # matmul inner_fn actually accesses (e.g. [1,M,K] when mm_to_bmm_pass added
    # a batch dim), so the lookup works even when ndim(orig_stl) < ndim(padded_size).
    # Falls back to generic only when no stride matches (should not occur in practice).
    orig_host_stride = list(arg_fx_node.meta["val"].stride())
    sm_last = int(list(orig_stl.stride_map)[-1])
    within_stick_dim = next(
        (i for i, s in enumerate(orig_host_stride) if int(s) == sm_last), None
    )
    if within_stick_dim is not None:
        dim_order = [i for i in range(len(padded_size)) if i != within_stick_dim] + [
            within_stick_dim
        ]
        padded_host_stride = [1] * len(padded_size)
        for i in range(len(padded_size) - 2, -1, -1):
            padded_host_stride[i] = padded_host_stride[i + 1] * padded_size[i + 1]
        padded_stl = SpyreTensorLayout(
            padded_size, padded_host_stride, dtype, dim_order
        )
    else:
        padded_stl = SpyreTensorLayout([concretize_expr(s) for s in padded_size], dtype)
    host_layout = padded_buf.layout
    padded_buf.layout = FixedTiledLayout(
        host_layout.device,
        host_layout.dtype,
        host_layout.size,
        host_layout.stride,
        padded_stl,
    )

    if const_is_new:
        const_tb = graph_lowering.run_node(const_fx)
        graph_lowering.env[const_fx] = const_tb
        const_buf = (
            const_tb.data.data
        )  # TensorBox -> StorageBox -> SpyreConstantFallback
        assert isinstance(const_buf, SpyreConstantFallback)
        _assign_layout(const_buf)

    expand_tb = graph_lowering.run_node(expand_fx)
    graph_lowering.env[expand_fx] = expand_tb
    # aten.expand lowers to an ExpandView — no Buffer, no layout assignment needed.

    clone_tb = graph_lowering.run_node(clone_fx)
    graph_lowering.env[clone_fx] = clone_tb
    clone_buf = clone_tb.data.data  # TensorBox -> StorageBox -> ComputedBuffer
    assert isinstance(clone_buf, ComputedBuffer)
    _assign_layout(clone_buf)
    # assign_origin_node sets origin_node on the inner Pointwise, not the ComputedBuffer.
    # LX planning (scratchpad.py) accesses op.origin_node directly on the ComputedBuffer.
    object.__setattr__(clone_buf, "origin_node", clone_fx)

    graph_lowering.run_node(overwrite_fill_fx)
    graph_lowering.env[overwrite_fill_fx] = empty_tb
    # overwrite lowers to ComputedBuffer with MutationLayoutSHOULDREMOVE — left unchanged.
    # run_node returns empty_tb (not the new overwrite buffer), so origin_node is not set.
    object.__setattr__(graph_lowering.operations[-1], "origin_node", overwrite_fill_fx)

    graph_lowering.run_node(overwrite_data_fx)
    graph_lowering.env[overwrite_data_fx] = empty_tb
    # overwrite lowers to ComputedBuffer with MutationLayoutSHOULDREMOVE — left unchanged.
    object.__setattr__(graph_lowering.operations[-1], "origin_node", overwrite_data_fx)

    # Collect all newly added operations (appended at the end of graph.operations).
    # Fresh path: spyre.empty(FK+MO=2) + spyre.constant(1) + clone(1) + overwrite×2(2) = 6.
    # Cache-hit path: spyre.constant is reused, so spyre.empty(2) + clone(1) + overwrite×2(2) = 5.
    new_ops = graph_lowering.operations[ops_before:]
    expected = 5 if not const_is_new else 6
    assert len(new_ops) >= expected, (
        f"Expected at least {expected} new ops, got {len(new_ops)}"
    )

    return padded_buf, list(new_ops)
