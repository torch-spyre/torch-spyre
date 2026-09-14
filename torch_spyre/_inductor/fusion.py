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

from collections.abc import Callable, Sequence

from torch._inductor.ir import ComputedBuffer
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
)
from . import config
from .constants import DEVICE_NAME
from .scheduler import (
    CountedLoopSchedulerNode,
    SuperDSCBundle,
    is_allocation_only_node,
)


def _make_fused(
    nodes: list[SchedulerNode | CountedLoopSchedulerNode],
) -> BaseSchedulerNode | None:
    if len(nodes) > 1:
        return SuperDSCBundle(nodes[0].scheduler, nodes)
    elif len(nodes) == 1:
        return nodes[0]
    return None


def _is_spyre_node(node: BaseSchedulerNode) -> bool:
    """True if the node computes on the Spyre device."""
    device = node.get_device()
    return device is not None and device.type == DEVICE_NAME


def _is_compute_node(node: BaseSchedulerNode) -> bool:
    """True if ``node`` carries a loop nest, so it can describe a bundle.

    A bundle needs one member's ``group`` -- the ``(device, iteration_space)``
    pair -- to emit a kernel from, and only these node types have one.
    """
    return _is_spyre_node(node) and isinstance(
        node, (SchedulerNode, CountedLoopSchedulerNode)
    )


def _is_fusable_node(node: BaseSchedulerNode) -> bool:
    """True if ``node`` may join the run being fused: a Spyre compute node, or
    an allocation feeding one.

    Everything else -- fallback nodes, CPU nodes, extern kernels that compute
    something -- forces a bundle boundary. An allocation admitted here is only
    kept where it is interior to the run; see :func:`_compute_span`.
    """
    if not _is_spyre_node(node):
        return False
    return _is_compute_node(node) or is_allocation_only_node(node)


def _compute_span(run: list[BaseSchedulerNode]) -> tuple[int, int]:
    """The half-open index range of ``run`` between its first and last compute node.

    Folding an allocation into a bundle pays off only where it would otherwise
    split a run of compute, which is to say where it is *interior*. A ``cat`` or
    ``stack`` destination is interior by construction: the compute that fills it
    follows it. An allocation at either end is a different animal -- coarse
    tiling hoists its cross-tile reduction accumulators ahead of a loop group
    precisely so they do not split it -- and absorbing one buys nothing while
    costing an unread buffer.

    Leading and trailing allocations are therefore handed back as their own
    single-element runs, which is what they were before fusion considered them.
    A run holding no compute at all yields no bundle to describe, so it splits
    entirely.
    """
    compute = [i for i, node in enumerate(run) if _is_compute_node(node)]
    if not compute:
        return (0, 0)
    return (compute[0], compute[-1] + 1)


def group_contiguous_fusable(items: list, is_fusable: Callable) -> list[list]:
    """Split ``items`` into maximal contiguous runs of fusable entries, with each
    non-fusable entry its own single-element run.

    This is the *policy* behind :func:`spyre_fuse_nodes` -- one SuperDSC bundle
    per maximal run of fusable ops, order preserved -- factored out so that
    callers which must estimate the same grouping earlier in the pipeline (from
    IR operations, before any scheduler exists) share it rather than carrying a
    copy that can drift. Callers differ only in ``is_fusable``, which is the
    irreducible difference: one sees scheduler nodes, the other IR operations.

    See :func:`estimate_bundles` for the early-stage caller.
    """
    groups: list[list] = []
    run: list = []
    for item in items:
        if is_fusable(item):
            run.append(item)
            continue
        if run:
            groups.append(run)
            run = []
        groups.append([item])  # a boundary is its own bundle
    if run:
        groups.append(run)
    return groups


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Fuse nodes together to form kernels without changing their order.
    Each kernel will be compiled into a single SuperDSC Bundle.
    """
    if len(nodes) == 0:
        return nodes
    if not config.bundle_symbolic_args:
        # Without symbolic args, tensor addresses are baked-in constants from
        # SEGMENT_OFFSETS, which has a fixed number of slots.  Fusing ops could
        # exceed that slot count, so disable fusion when symbolic args are off.
        return nodes

    # One bundle per maximal contiguous run of fusable nodes, trimmed to the span
    # its compute members cover; a boundary node comes back as a single-element
    # run, and ``_make_fused`` returns it unchanged.
    fused_nodes: list[BaseSchedulerNode] = []
    for group in group_contiguous_fusable(nodes, _is_fusable_node):
        start, stop = _compute_span(group)
        fused_nodes.extend(group[:start])
        if start < stop and (fused := _make_fused(group[start:stop])):
            fused_nodes.append(fused)
        fused_nodes.extend(group[stop:] if start < stop else group)
    return fused_nodes


def _is_fusable_operation(op) -> bool:
    """The IR-operation analogue of :func:`_is_fusable_node`.

    A ``ComputedBuffer`` on the Spyre device is what becomes a fusable
    ``SchedulerNode`` later; anything else -- an extern kernel, a fallback, a CPU
    op -- forces a bundle boundary. This is an *estimate*: whether a given
    operation ends up fused or extern is a scheduling decision that has not been
    made yet at this point in the pipeline.
    """
    if not isinstance(op, ComputedBuffer):
        return False
    device = op.get_device()
    return device is not None and device.type == DEVICE_NAME


def estimate_bundles(operations: Sequence) -> list[list]:
    """Estimate the SuperDSC bundles (fused kernels) ``operations`` will become.

    Callers that score a graph -- a cost model, for instance -- need the grouping
    because bundle membership changes the result: external inputs are
    deduplicated across a bundle, the pointwise arity derate counts its ops, and
    the underfill derate takes its worst tile.

    Such a caller cannot ask for the real grouping if it runs as a
    *pre-scheduling* pass, where ``V.graph.scheduler`` is still ``None``; fusion
    is decided two stages later by :func:`spyre_fuse_nodes`. What makes an
    estimate viable is that the real rule is order-preserving and structural, so
    it is reproduced here by sharing :func:`group_contiguous_fusable` and
    supplying the IR-level predicate.

    Returns groups of the input operations, in order, so ``[op.get_name() ...]``
    per group gives the buffer names in each bundle.

    Fusion can be off entirely (``config.bundle_symbolic_args``), in which case
    the real pass leaves every node alone and this returns one bundle per
    operation to match.

    Expect the shape to be right and the membership to under-count. Checked
    against the real grouping on a softmax graph, the bundle count, run structure
    and boundary placement all matched (the extern kernel was correctly a
    boundary); what the estimate missed was one ``SchedulerNode`` absent from
    ``graph.operations`` because scheduling creates it later.
    """
    if not config.bundle_symbolic_args:
        return [[op] for op in operations]
    return group_contiguous_fusable(list(operations), _is_fusable_operation)
