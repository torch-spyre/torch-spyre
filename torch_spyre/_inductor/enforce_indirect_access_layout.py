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

"""Enforce non-stick dimension ordering required by indirect-access ops.

The three-pass restickify pipeline (propagate_layouts -> optimize_restickify ->
insert_restickify) resolves stick-dimension layout constraints. Indirect-access
ops (gather/scatter) impose an additional requirement on non-stick dimension
ordering: the indexed dimension must be outermost in the value tensor's device
layout, based on their coordinate access patterns.

This pass runs after insert_restickify, once every op has a committed
FixedTiledLayout. For indirect-access ops, it checks whether the value tensor's
current dim_order matches this requirement; if not, either rewrites the
producer's output layout in place (if the producer is a ComputedBuffer and not a
graph output) or inserts a spyre.restickify copy in the required layout.
"""

import sympy
import torch

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    ReinterpretView,
    Scatter,
)
from torch_spyre._C import DataFormats, SpyreTensorLayout

from .constants import ELIDED_COPY_BACK_ATTR
from .errors import Unsupported
from .insert_restickify import (
    _fixed_tiled,
    insert_restickify_on_node_inputs,
    RestickifyArgInfo,
)
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import (
    AlignmentAccess,
    _build_indirect_store_subs,
    _find_scatter_index_buf_names,
    build_operation_alignment_inputs,
    concretize_expr,
    indirect_info_from_op,
    iteration_space_from_op,
    iteration_space_with_splits,
    padded_entry_output_stl,
    per_trip_index,
)
from .views import AlignmentInputs, UnalignedStickSplit, align_tensors_pure
from . import config

logger = get_inductor_logger("enforce_indirect_access_layout")


def _pad_output_for_stick_aligned_split(op: ComputedBuffer) -> bool:
    """Grow a gather output's index-entry dim to the index stick multiple.

    Multi-core work division splits the index-entry dim in whole index sticks.
    When the entry count is a partial last stick (e.g. 40 over a 32-int32 index
    stick), the per-core base is stick-aligned for the index tensor but
    element-aligned for the shorter output, so the two disagree and the split
    miscompiles. ``padded_entry_output_stl`` returns the output layout grown so
    that dim spans whole sticks (or None when there is nothing to pad); applying
    it aligns the output base and gives the later cores an in-bounds place to
    write. The logical size is unchanged: the D2H copy extracts the logical view
    from the (larger) physical allocation.

    No-op on a single core, on an already stick-aligned count, or on an in-place
    (mutation) destination this pass cannot safely resize.
    """
    if config.sencores <= 1:
        return False
    if isinstance(op.get_layout(), MutationLayoutSHOULDREMOVE):
        return False
    padded_stl = padded_entry_output_stl(op)
    if padded_stl is None:
        return False
    op.layout = _fixed_tiled(_real_layout(op), padded_stl)
    return True


def _scatter_access_subs_and_sizes(
    scatter_op: ComputedBuffer, coord_layout, write_dep: MemoryDep
) -> tuple[dict, dict]:
    """Build access substitutions and sizes for scatter op index symbols.

    _build_indirect_store_subs keys its result by the scatter index symbol
    itself (e.g. tmp0 in `d1 + 1024*tmp0`), mapping it to the index buffer's
    IndexedBase access. Extract the actual scatter index symbols from
    write_dep.index (symbols that appear in the write but NOT in write_dep's
    loop ranges) and build IndirectAccess mappings for each one.

    ``coord_layout`` must be the layout that write_dep's coordinates will
    actually be computed against (the scatter's output when checking the
    op's own device layout; the mutation target's layout when checking
    destination compliance against a possibly-different target layout) --
    sizing against a different buffer's strides can silently mismatch or
    misresolve a symbol's size.

    For each scatter symbol, compute its device dimension via device_coordinates
    to find the corresponding size in coord_layout.device_size.
    Returns ({sym: IndirectAccess(...)}, {sym: size}).
    """

    store_subs, _ = _build_indirect_store_subs(scatter_op)

    # Extract the actual scatter index symbols from write_dep.index.
    # These are symbols that appear in the write but NOT in write_dep.ranges.
    all_syms = write_dep.index.free_symbols
    loop_syms = set(write_dep.ranges.keys())
    scatter_syms = all_syms - loop_syms

    access_subs: dict = {
        sym: IndirectAccess(sympy.Symbol(store_subs[sym].base.name))
        for sym in scatter_syms
        if sym in store_subs and hasattr(store_subs[sym], "base")
    }

    # Compute device coordinates to find the actual device dimension for each
    # scatter symbol, then look up its size from coord_layout.device_size.
    sizes: dict[sympy.Symbol, int] = {}
    if coord_layout.size and coord_layout.stride:
        # For each scatter symbol, find which host dimension it multiplies
        # (by matching the stride of that dimension in coord_layout.stride).
        for sym in access_subs:
            # Extract the coefficient of this symbol in write_dep.index
            coeff = write_dep.index.coeff(sym)
            if coeff is not None:
                # Find which host dimension has this stride
                for dim_idx, stride in enumerate(coord_layout.stride):
                    if stride == coeff:
                        if 0 <= dim_idx < len(coord_layout.size):
                            sizes[sym] = coord_layout.size[dim_idx]
                        break

    return access_subs, sizes


def _real_layout(buf) -> FixedTiledLayout:
    layout = buf.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        assert getattr(buf, ELIDED_COPY_BACK_ATTR, False), (
            f"unexpected mutation layout on {buf.get_name()!r}"
        )
        layout = layout.real_layout()
    return layout


def _insert_relayout_copy(
    graph: GraphLowering,
    consumer_op: ComputedBuffer,
    value_buf,
    required_layout: FixedTiledLayout,
) -> ComputedBuffer:
    """Insert a spyre.restickify copy of value_buf in required_layout ahead of
    consumer_op, and patch consumer_op's inner_fn to read the new buffer.

    Returns the reconstructed ComputedBuffer that replaced consumer_op in
    graph.operations (insert_restickify_on_node_inputs invalidates the
    original instance).
    """
    operations = graph.operations
    arg_name = value_buf.get_name()
    consumer_name = consumer_op.get_name()
    insert_restickify_on_node_inputs(
        consumer_op,
        [
            RestickifyArgInfo(
                arg_name=arg_name,
                dep_index=None,
                occurrence=0,
                target_layout=required_layout,
            )
        ],
        operations,
    )
    logger.info(
        "enforce_indirect_access_layout: inserted relayout copy of %s before %s",
        arg_name,
        consumer_name,
    )
    return next(
        o
        for o in operations
        if isinstance(o, ComputedBuffer) and o.get_name() == consumer_name
    )


def _dense_scatter_source_stl(value_layout: FixedTiledLayout) -> SpyreTensorLayout:
    """Build a dense layout for a direct scatter source.

    ReStickifyOpHBM operates on the physical DL16 format, shared by logical
    float16 and bfloat16 tensors.  Keep the source's logical dtype and element
    arrangement while returning its host dimensions to their canonical order.
    """
    source_stl = value_layout.device_layout
    if source_stl.device_dtype != DataFormats.SEN169_FP16:
        raise Unsupported(
            "scatter source alignment requires materialization, but "
            f"ReStickifyOpHBM does not support {source_stl.device_dtype}"
        )
    size = [concretize_expr(value) for value in value_layout.size]
    stride = [concretize_expr(value) for value in value_layout.stride]
    return SpyreTensorLayout(
        size,
        stride,
        value_layout.dtype,
        list(range(len(size))),
        source_stl.element_arrangement,
    )


def _is_synthetic_restickify(buf) -> bool:
    """Return whether ``buf`` was produced by the restickify insertion pass."""
    if not isinstance(buf, ComputedBuffer) or len(buf.origins) != 1:
        return False
    return next(iter(buf.origins)).target is torch.ops.spyre.restickify.default


def _scatter_alignment_inputs(
    graph: GraphLowering,
    op: ComputedBuffer,
    *,
    layout_overrides: dict[str, SpyreTensorLayout] | None = None,
) -> AlignmentInputs | None:
    """Reconstruct codegen's alignment input for a committed scatter op."""
    overrides = layout_overrides or {}
    read_writes = op.get_read_writes()
    write_dep = next(
        (dep for dep in read_writes.writes if isinstance(dep, MemoryDep)), None
    )
    if write_dep is None:
        return None

    output_layout = _output_real_layout(op)
    if not isinstance(output_layout, FixedTiledLayout):
        return None
    _, indirect_sizes = _scatter_access_subs_and_sizes(op, output_layout, write_dep)

    accesses: list[AlignmentAccess] = []
    for dep in read_writes.reads:
        if not isinstance(dep, MemoryDep):
            continue
        buf = graph.get_buffer(dep.name)
        layout = _real_layout(buf)
        if not isinstance(layout, FixedTiledLayout):
            return None
        accesses.append(
            AlignmentAccess(
                overrides.get(dep.name, layout.device_layout),
                per_trip_index(op, dep.index),
            )
        )
    accesses.append(
        AlignmentAccess(
            output_layout.device_layout, per_trip_index(op, write_dep.index)
        )
    )
    space = iteration_space_from_op(op)
    return build_operation_alignment_inputs(
        space,
        accesses,
        iteration_space_with_splits(op, read_writes, space),
        indirect_sizes=indirect_sizes,
    )


def _materialize_unaligned_scatter_source(
    graph: GraphLowering,
    op: ComputedBuffer,
    index_names: set[str],
) -> ComputedBuffer:
    """Densify the direct source when it cuts through an index stick.

    Tensor alignment merges factorization boundaries from every operand.  A
    transposed direct source can contribute an internal boundary that falls
    inside the int32 index tensor's 32-element physical stick.  Such a split is
    not representable and used to truncate the index's outer-stick extent to
    zero.  Materialize just the direct source in a canonical dense DL16 layout;
    the destination and index layouts remain unchanged.

    PyTorch lowering represents every ``Scatter`` as a ``ComputedBuffer`` with
    a ``MutationLayoutSHOULDREMOVE`` target, which is the invariant required by
    ``_resolve_mutation_target`` below.
    """
    alignment_inputs = _scatter_alignment_inputs(graph, op)
    if alignment_inputs is None:
        return op
    try:
        align_tensors_pure(alignment_inputs)
        return op
    except UnalignedStickSplit as error:
        alignment_error = error

    target_name, _ = _resolve_mutation_target(op)
    # ``indirect_info_from_op`` can expose placeholder names when it cannot
    # pair a scatter symbol with its real index buffer.  Also exclude index
    # tensors recovered directly from the Scatter closure so they are never
    # mistaken for a source candidate.
    index_names = index_names | _find_scatter_index_buf_names(op)
    # A fused Scatter retains no distinguished source edge.  Its remaining
    # direct reads are therefore candidates, not assumptions: accept one only
    # when substituting its dense layout makes the complete alignment valid.
    direct_deps = [
        dep
        for dep in op.get_read_writes().reads
        if isinstance(dep, MemoryDep)
        and dep.name not in index_names
        and dep.name != target_name
    ]
    for dep in direct_deps:
        source_buf = graph.get_buffer(dep.name)
        source_layout = _real_layout(source_buf)
        if not isinstance(source_layout, FixedTiledLayout):
            logger.debug(
                "enforce_indirect_access_layout: scatter source candidate %s "
                "has non-tiled layout %s",
                dep.name,
                type(source_layout).__name__,
            )
            continue
        # ReStickify cannot materialize a non-DL16 source.  It may be an
        # unrelated direct operand, so keep looking for a feasible candidate;
        # if none resolves the split, preserve the original alignment error.
        if source_layout.device_layout.device_dtype != DataFormats.SEN169_FP16:
            logger.debug(
                "enforce_indirect_access_layout: scatter source candidate %s "
                "uses unsupported ReStickify format %s",
                dep.name,
                source_layout.device_layout.device_dtype,
            )
            continue
        dense_stl = _dense_scatter_source_stl(source_layout)
        if dense_stl == source_layout.device_layout:
            logger.debug(
                "enforce_indirect_access_layout: scatter source candidate %s "
                "is already dense",
                dep.name,
            )
            continue
        candidate_inputs = _scatter_alignment_inputs(
            graph, op, layout_overrides={dep.name: dense_stl}
        )
        if candidate_inputs is None:
            logger.debug(
                "enforce_indirect_access_layout: could not reconstruct alignment "
                "inputs for scatter source candidate %s",
                dep.name,
            )
            continue
        try:
            align_tensors_pure(candidate_inputs)
        except UnalignedStickSplit as candidate_error:
            logger.debug(
                "enforce_indirect_access_layout: dense scatter source candidate "
                "%s leaves alignment invalid: %s",
                dep.name,
                candidate_error,
            )
            continue

        logger.info(
            "enforce_indirect_access_layout: materializing scatter source %s "
            "in a dense DL16 layout before %s",
            dep.name,
            op.get_name(),
        )
        if _is_synthetic_restickify(source_buf):
            # Layout propagation may already have inserted a private restickify
            # for this scatter edge.  Retarget that copy directly: inserting a
            # second copy here would read an intermediate whose physical
            # geometry was already chosen from the scatter destination and can
            # therefore be too large for the logical source (for example a
            # 64-token K tensor feeding a 384-token KV cache).
            source_buf.layout = _fixed_tiled(source_layout, dense_stl)
            return op
        return _insert_relayout_copy(
            graph,
            op,
            source_buf,
            _fixed_tiled(source_layout, dense_stl),
        )

    raise alignment_error


def _output_real_layout(op: ComputedBuffer) -> FixedTiledLayout:
    """Resolve an op's committed output layout, unwrapping a genuine mutation
    target (unlike _real_layout, which asserts mutation layouts only appear on
    elided copy-backs — an op's own MutationLayoutSHOULDREMOVE is expected)."""
    layout = op.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    return layout


def _resolve_mutation_target(op: ComputedBuffer) -> tuple[str, object]:
    """Unwrap a MutationLayoutSHOULDREMOVE op's layout to (target_name, target_buf).

    Mirrors propagate_layouts.py's MutationLayoutSHOULDREMOVE branch, which
    unwraps ReinterpretView chains the same way to resolve the mutation target.
    """
    assert isinstance(op.layout, MutationLayoutSHOULDREMOVE)
    target = op.layout.target
    while isinstance(target, ReinterpretView):
        target = target.data
    return target.get_name(), target


def _get_indirect_access_dim_order_requirements(
    op: ComputedBuffer,
) -> tuple[set[str], dict, dict[sympy.Symbol, int] | None] | None:
    """Extract non-stick dimension ordering requirements from an indirect-access op.

    Returns (dep_names, access_subs, sizes) if the op has requirements, else None.
    """
    dep_names, access_subs, sizes = indirect_info_from_op(op)
    if dep_names:
        logger.debug(
            "enforce_indirect_access_layout: op %s has dim-order requirements "
            "from %d deps",
            op.get_name(),
            len(dep_names),
        )
        return dep_names, access_subs, sizes
    return None


def enforce_indirect_access_layout(graph: GraphLowering) -> None:
    """Post-insert_restickify fixups for indirect-access ops.

    Handles IA-specific fixups that require committed FixedTiledLayout:
      - pad gather output's index-entry dim for stick-aligned multi-core split
      - materialize unaligned scatter source when it cuts through an index stick

    Gather value tensor dim reordering is handled by reorder_nonstick_dims
    (ComputedBuffer sources) and reorder_nonstick_dims_mutation (graph inputs).
    Scatter destination dim ordering for mutation targets is handled by
    reorder_nonstick_dims_mutation (after insert_restickify).
    """
    for original_op in list(graph.operations):
        if not isinstance(original_op, ComputedBuffer):
            continue
        is_scatter = isinstance(original_op.data, Scatter)

        requirement = _get_indirect_access_dim_order_requirements(original_op)
        if not requirement:
            continue
        dep_names = requirement[0]

        _pad_output_for_stick_aligned_split(original_op)

        op = original_op
        if is_scatter:
            op = _materialize_unaligned_scatter_source(graph, op, dep_names)
            if op is not original_op:
                requirement = _get_indirect_access_dim_order_requirements(op)
                assert requirement is not None
                dep_names = requirement[0]
