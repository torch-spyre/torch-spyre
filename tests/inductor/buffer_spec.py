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

"""The buffer form: hand-written buffers for solver tests and capture replay.

The compiler builds the solve's graph straight from the IR
(``GraphBuilder.prepare_graph``). Tests describe a graph as buffers
instead; this module turns them into the same graph, candidates on its nodes
and relations on its edges, and a solved plan back into the buffers' terms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import sympy

from torch_spyre._inductor.scratchpad.allocator import SolvedPlan
from torch_spyre._inductor.scratchpad.buffer_graph import (
    ResourceNode,
    SolveSettings,
    SpyreOptimizationGraph,
    WorkSplit,
    empty_graph,
)
from torch_spyre._inductor.scratchpad.encoding_strategy import EnumeratedStrategy
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    ceil_div,
    division_symbol,
    sym_core_divs_for,
    sym_tile_counts_for,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.pass_utils import PerCoreView
    from torch_spyre._inductor.scratchpad.lx_relayout import ChosenRelayout
    from torch_spyre._inductor.work_division import ResidencyEdge


@dataclass
class LifetimeBoundBuffer:
    """
    A buffer with a lifetime: one node of a hand-written solve graph.

    ``uses`` is the strictly increasing list of operation indices at which the
    buffer is accessed (as returned by ``calculate_liveness``).  It is normally
    non-empty, and callers that read ``start_time``/``end_time`` require that,
    since those properties index into it.  Emptiness is nevertheless allowed at
    construction -- see :meth:`__post_init__` for the registration state that
    needs it.  ``first_use_is_read`` is True for graph inputs (all accesses are
    reads) and False for computed buffers (first access is a write, all
    subsequent accesses are reads).

    Both properties of ``uses`` are asserted in ``__post_init__``.  Strictness
    is what makes ``read_count`` trustworthy: one entry per accessing op means
    that for a computed buffer ``read_count == 0`` is exactly "written, never
    read", which the in-place invariants rely on (see
    :func:`check_in_place_parent_is_read`).  A repeated index would describe a
    buffer written and read by the same op, i.e. with a single live tick, and
    would let such a buffer pass as an in-place parent.

    ``start_time`` and ``end_time`` are convenience properties derived from
    ``uses`` and widened by optional counted-loop lifetime overrides.
    """

    name: str
    size: int
    uses: list[int]
    first_use_is_read: bool = False
    address: Optional[int] = None
    in_place_parents: list[str] = field(default_factory=list)
    # define the reason for excluding the buffer based on allocator
    # or solver logic paths.
    residency_reason: Optional[str] = None
    # Optional lifetime bounds for storage reused by a counted loop. Keep these
    # separate from ``uses``: they change address overlap, but must not
    # manufacture reads or inflate residency/spill benefit.
    lifetime_start_override: Optional[int] = None
    lifetime_end_override: Optional[int] = None
    # The physical per-core ownership accepted by the residency judge. Placement
    # writes this beside the LX address; it must never derive another view.
    lx_view: Optional["PerCoreView"] = field(default=None, repr=False, compare=False)
    # Written by the solver: why it left the buffer out of LX, when it says.
    spill_reason: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Not also asserted non-empty: buffers are sometimes registered before
        # their uses are known and filled in afterwards (see
        # ``make_buffer_registry`` in tests/inductor/test_scratchpad_patterns.py),
        # and an empty list is vacuously strictly increasing. Callers that read
        # ``start_time``/``end_time`` still require a non-empty list.
        #
        # This runs at construction only, so it does not see later mutation of
        # ``uses``; the in-place invariants therefore test the property they need
        # directly rather than inferring it from ``read_count``.
        assert all(a < b for a, b in zip(self.uses, self.uses[1:])), (
            f"buffer {self.name} has uses={self.uses}, which is not strictly "
            "increasing; uses carries one distinct index per accessing operation"
        )
        if self.lifetime_start_override is not None and self.uses:
            assert self.lifetime_start_override <= self.uses[0], (
                f"buffer {self.name} has lifetime_start_override="
                f"{self.lifetime_start_override} after nominal start {self.uses[0]}"
            )
        if self.lifetime_end_override is not None and self.uses:
            assert self.lifetime_end_override >= self.uses[-1] + 1, (
                f"buffer {self.name} has lifetime_end_override="
                f"{self.lifetime_end_override} before nominal exclusive end "
                f"{self.uses[-1] + 1}"
            )

    @property
    def read_count(self) -> int:
        """Number of reads.  For a computed buffer the first use is the producing
        write, so every use but that one is a read; when ``first_use_is_read``
        (a graph input) every use is a read.  Exact because ``uses`` holds one
        distinct index per accessing op.

        This counts the buffer's reads, not the reads residency would save: an
        input's first read is the clone-in that pinning cannot avoid, so a cost
        model has to discount it separately (see
        :func:`buffer_graph.spill_cost`).  The ``max`` only guards
        the transient empty-``uses`` state described in :meth:`__post_init__`.
        """
        return max(0, len(self.uses) - (0 if self.first_use_is_read else 1))

    @property
    def start_time(self) -> int:
        nominal = self.uses[0]
        override = self.lifetime_start_override
        return min(nominal, override if override is not None else nominal)

    @property
    def end_time(self) -> int:
        nominal = self.uses[-1] + 1
        return max(nominal, self.lifetime_end_override or nominal)

    @property
    def min_footprint(self) -> int:
        """Smallest LX footprint the buffer can take, for the capacity check"""
        return self.size

    def overlaps_in_time(self, other: "LifetimeBoundBuffer") -> bool:
        """Returns true iff self and other overlap in time."""
        return self.start_time < other.end_time and other.start_time < self.end_time

    @property
    def sym_is_lx(self) -> sympy.Symbol:
        return sympy.Symbol(f"is_lx_{self.name}", integer=True, nonnegative=True)


@dataclass
class CoreDivisionBuffer(LifetimeBoundBuffer):
    """A :class:`LifetimeBoundBuffer` carrying the joint core-division metadata
    (the buffer form of an open node)."""

    core_divisions: list[CoreDivision] = field(default_factory=list)
    # Producer buffer names; defines the producer->consumer edges for matching.
    parents: list[str] = field(default_factory=list[str])
    # parent_buf_name -> (parent_div_idx, this_div_idx) pairs that induce the
    # *same per-core slicing of the parent*, precomputed by the allocator via
    # ``_per_core_view_on_buf`` (physical device-dim view equality, correct
    # across reductions/reshapes). These are the sole slicing-match predicate;
    # an absent/empty entry means no compatible division, so the gate forbids
    # the merge/residency across that edge.
    cd_parent_matches: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # The same relation per candidate rather than per pair: one edge per divided
    # producer this buffer reads, keyed as ``cd_parent_matches`` is. A solver
    # that generates divisions asks these instead of indexing the table, and
    # constructs the division on the other end of an edge by inverting the view.
    # Empty where the allocator has not built them (they need the live ops).
    residency_edges: dict[str, "ResidencyEdge"] = field(default_factory=dict)
    chosen_division: Optional[int] = None
    # Solver-chosen relayouts feeding this consumer: parent_buf_name -> the
    # fired candidate with the destination address (bytes) of the group's copy.
    # Written back by the solver when this
    # consumer reads the parent through a resident copy. Every consumer served
    # by one copy carries the same address; the commit path materializes one
    # plan per fired group (``FiredRelayoutGroup.from_chosen``).
    chosen_relayouts: dict[str, "ChosenRelayout"] = field(default_factory=dict)
    boundary: BufferType = BufferType.Intermediate

    def per_core_size(self, division: int) -> int:
        """Bytes per core after output partitioning and output tiling.

        Reduction splits retain a full output part.
        """
        cd = self.core_divisions[division]
        return ceil_div(self.size, cd.output_partition * cd.tiling.output_tile_count)

    @property
    def min_footprint(self) -> int:
        """Smallest :meth:`per_core_size` any candidate division allows. With no
        candidates there is nothing to divide by, so it falls back to ``size``
        (the placement-only case ``SpyreOptimizationGraph.add_node`` commits)."""
        if not self.core_divisions:
            return self.size
        return min(self.per_core_size(i) for i in range(len(self.core_divisions)))

    @property
    def sym_cores(self) -> sympy.Symbol:
        return math.prod(self.sym_core_divs.values())

    @property
    def sym_division(self) -> sympy.Symbol:
        """The chosen index into ``core_divisions`` as an objective unknown.

        The per-axis split symbols (:attr:`sym_core_divs`) carry the division's
        *shape* into the cost model; this carries its *identity*, for terms that
        are tables over candidates rather than functions of the splits. Engines
        bind it to their division variable (CP-SAT) or to the chosen index (the
        annealer's ``chosen``)."""
        return division_symbol(self.name)

    @property
    def sym_core_divs(self) -> dict[sympy.Symbol, sympy.Symbol]:
        """See :func:`sym_core_divs_for`."""
        return sym_core_divs_for(self.name, self.core_divisions)

    @property
    def sym_tile_counts(self) -> dict[tuple[int, bool], sympy.Symbol]:
        """See :func:`sym_tile_counts_for`."""
        return sym_tile_counts_for(self.name, self.core_divisions)


def check_in_place_parent_is_read(
    parent: "LifetimeBoundBuffer", child_name: str
) -> None:
    """Reject an in-place parent whose storage is never read before handover.

    The child takes the parent's storage over at the parent's last use, so that
    use has to be a read. For a computed buffer the first use is the write, so a
    single use means the buffer is written and never read -- handing it to a
    child would overwrite data nothing ever consumed, and it would make parent
    and child come alive on the same tick while sharing storage. Graph inputs are
    exempt: all their uses are reads, so one use is enough.
    """
    # Tested as "a use strictly after the first" rather than via ``read_count``:
    # the two agree whenever ``uses`` is strictly increasing, but ``uses`` is
    # validated at construction and can be mutated afterwards, and this way a
    # repeated index cannot pass as a read.
    has_read_after_write = len(parent.uses) > 1 and parent.uses[-1] > parent.uses[0]
    if not (parent.first_use_is_read or has_read_after_write):
        raise ValueError(
            f"In-place parent {parent.name} is a computed buffer that is never "
            f"read (uses={parent.uses}), so it cannot hand its storage to child "
            f"{child_name}"
        )


def check_in_place_relationships(
    buffers: Sequence["LifetimeBoundBuffer"],
) -> None:
    """Reject any declared in-place pair that violates a required invariant."""
    buf_by_name = {b.name: b for b in buffers}
    for child in buffers:
        for parent_name in child.in_place_parents:
            parent = buf_by_name.get(parent_name)
            if parent:
                if parent.end_time != child.start_time + 1:
                    raise ValueError(
                        f"In-place parent {parent_name}.end_time={parent.end_time} "
                        f"must equal child {child.name}.start_time+1="
                        f"{child.start_time + 1}"
                    )
                check_in_place_parent_is_read(parent, child.name)
                # With core_divisions ``size`` is the *total* footprint, so a static
                # size check doesn't apply; the per-core match is enforced against the
                # chosen division in ``EnumeratedStrategy.encode_merge``. Only
                # the division-fixed case (plain ``LifetimeBoundBuffer``, no
                # ``core_divisions``) keeps the static check.
                if not (
                    getattr(parent, "core_divisions", None)
                    or getattr(child, "core_divisions", None)
                ):
                    if child.size > parent.size:
                        raise ValueError(
                            f"In-place child {child.name}.size={child.size} "
                            f"must be <= parent {parent_name}.size={parent.size}"
                        )


def add_node(
    graph: SpyreOptimizationGraph,
    b: LifetimeBoundBuffer,
    residency_reason: Optional[str] = None,
) -> ResourceNode:
    """A node holding ``b``'s data. A buffer with no candidate divisions is
    committed upstream."""
    divided = isinstance(b, CoreDivisionBuffer)
    return graph.add_ir_node(
        b.name,
        nbytes=b.size,
        uses=b.uses,
        first_use_is_read=b.first_use_is_read,
        boundary=b.boundary if divided else None,
        residency_reason=residency_reason,
        start_override=b.lifetime_start_override,
        end_override=b.lifetime_end_override,
        lx_view=b.lx_view,
        menu=b.core_divisions if divided else (),
    )


def build_graph(
    buffers: Sequence[LifetimeBoundBuffer],
    alignment: int,
    capacity: int,
    settings: SolveSettings,
    reasons: Mapping[str, str] = {},
) -> SpyreOptimizationGraph:
    """The graph of ``buffers``, sized in ``alignment`` units. ``reasons`` is
    each barred buffer's residency reason. Raises ``ValueError`` on an in-place
    pair no solver may realize.

    One edge per (producer, consumer) pair the consumer reads (``parents``),
    ``in_place`` when the producer is one of its ``in_place_parents``; in-place
    inputs come first, in that order, since the merge literals are made in it.
    An edge carries the consumer's pair table for the producer
    (``cd_parent_matches``) and its residency relation (``residency_edges``)."""
    check_in_place_relationships(buffers)
    graph = SpyreOptimizationGraph(
        alignment=alignment, capacity=capacity, settings=settings
    )
    for b in buffers:
        add_node(graph, b, reasons.get(b.name))
    nodes = graph.nodes
    for b in buffers:
        node = nodes[b.name]
        graph.add_reads(node, getattr(b, "parents", ()), b.in_place_parents)
        if isinstance(b, CoreDivisionBuffer):
            for edge in node.inputs:
                edge.pairs = b.cd_parent_matches.get(edge.src.name)
                edge.relation = b.residency_edges.get(edge.src.name)
    graph.link_execution_order()
    return graph


def strategy_of(buffers: Sequence[LifetimeBoundBuffer]) -> EnumeratedStrategy:
    """The table strategy; the buffers' candidates are on their graph."""
    return EnumeratedStrategy()


def plan_of(buffers: Sequence[LifetimeBoundBuffer]) -> SolvedPlan:
    """A solved plan given as buffers: the nodes of ``buffers`` in byte units,
    each decision read off its buffer (``address``, ``chosen_division``,
    ``spill_reason``, ``residency_reason``)."""
    reasons = {b.name: b.residency_reason for b in buffers if b.residency_reason}
    graph = build_graph(buffers, 1, 1 << 62, empty_graph(0).settings, reasons)
    by_name = {b.name: b for b in buffers}
    for node in graph.get_nodes():
        b = by_name[node.name]
        node.resident = b.address is not None
        node.offset = b.address
        node.spill_reason = b.spill_reason
        chosen = getattr(b, "chosen_division", None)
        if chosen is not None and node.menu:
            node.work_split = WorkSplit.of(node.menu[chosen])
    return SolvedPlan(list(graph.get_nodes()), 1)
