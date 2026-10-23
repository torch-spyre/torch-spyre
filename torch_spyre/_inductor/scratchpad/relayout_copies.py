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

"""Relayouts as constraints (Phase 2d, experimental).

Every edge whose read may go through a relayout copy gets one optional copy,
a :class:`Transform` whose view is the consumer's. A later consumer of the
same producer may read through an earlier edge's copy instead when their
destination views are equal (reified on view classes), so a copy is shared
and priced once; its lifetime ends after its last user (a variable, bounded
in ``encode_lifetime``). Legality and price are one table per edge over the
two ends' view ids: every per-pair gate of the table encoding is a function
of the two views and their core counts (``RelayoutEdge.price``), so nothing
enumerates division pairs, and no candidate cap applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from torch_spyre._inductor.scratchpad.buffer_graph import (
    DependencyEdge,
    ResourceNode,
    SpyreOptimizationGraph,
    Transform,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision, ceil_div

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.encoding_strategy import (
        Model,
        NodeVars,
        EncodingStrategy,
    )
    from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate
    from torch_spyre._inductor.scratchpad.graph_builder import RelayoutEdge

Var = Any


def _solved_entry(node: ResourceNode) -> tuple[int, CoreDivision]:
    """The node's solved work split as an entry of its menu. A strategy
    chooses among its own candidates (Lazy's point grid, or a point menu
    filtered to its legal divisions), so only the menu is indexed alike by
    the relayout's views and the commit path's checks."""
    index = next(
        (i for i, cd in enumerate(node.menu) if WorkSplit.of(cd) == node.work_split),
        None,
    )
    assert index is not None, (
        f"{node.name}: solved work split {node.work_split} is not in its menu"
    )
    return index, node.menu[index]


@dataclass(eq=False)
class _Copy:
    """One edge's optional copy and the vars that relate it."""

    edge: DependencyEdge
    data: RelayoutEdge
    transform: Transform
    tick: int  # the consumer's read
    dst_id: Var | int = -1  # the consumer's read view (structural id)
    view_class: Var | int = -1  # its ownership class: what copies share on
    price: Var | int = 0
    legal: list[tuple[int, int, int]] = field(default_factory=list)
    fire: Optional[Var] = None  # the read goes through some copy
    uses: list[tuple[_Copy, Var]] = field(default_factory=list)  # owner, lit
    users: list[tuple[_Copy, Var]] = field(default_factory=list)  # user, lit


class RelayoutCopies:
    """Every strategy's relayout copies, one optional copy per edge that may
    read through one: built by ``EncodingStrategy.prepare`` from each
    consumer's ``RelayoutEdge``s."""

    def __init__(self, strategy: EncodingStrategy, edges: dict) -> None:
        self.strategy = strategy
        self.edges = edges  # (producer, consumer) -> RelayoutEdge
        self.copies: dict[Transform, _Copy] = {}
        self.by_edge: dict[DependencyEdge, _Copy] = {}
        self.charges: list[Var] = []
        self._src: dict[ResourceNode, tuple[Var | int, list]] = {}
        self._views: dict[tuple[str, str], list] = {}

    def prepare(self, graph: SpyreOptimizationGraph) -> list[Transform]:
        """One candidate copy per eligible edge, in copy order: by producer,
        then by the consumer's tick."""
        s = self.strategy
        found = []
        for node in graph.get_nodes():
            for edge in node.outputs:
                data = self.edges.get((node.name, edge.dst.name))
                if (
                    data is None
                    or not edge.reads
                    or edge.dst.produced_at is None
                    or not s.chooses_split(node)
                    or not s.chooses_split(edge.dst)
                ):
                    continue
                # The consumer's read tick, as a table copy's lifetime counts it.
                found.append((node.name, edge.dst.start_time, edge.dst.name, edge))
        transforms: list[Transform] = []
        last_tick = max((n.end_time for n in graph.get_nodes()), default=0)
        for parent, tick, consumer, edge in sorted(found, key=lambda f: f[:3]):
            transform = Transform(
                producer=edge.src,
                name=f"{parent}__relayout_for__{consumer}",
                group=len(transforms),
                size=edge.src.size,
                read_count=edge.reads,
                division=CoreDivision(),
                start_time=tick,
                end_time=max(tick + 1, last_tick),
            )
            copy = _Copy(edge, self.edges[(parent, consumer)], transform, tick)
            self.copies[transform] = copy
            self.by_edge[edge] = copy
            transforms.append(transform)
        return transforms

    # ---- views ----

    def _key_ids(self, buffer: str, side: str, keys: list) -> list[int]:
        """A structural id per (view, cores) key, shared per buffer and side;
        -1 for no view."""
        table = self._views.setdefault((buffer, side), [])
        ids = []
        for key in keys:
            if key is None:
                ids.append(-1)
                continue
            if key not in table:
                table.append(key)
            ids.append(table.index(key))
        return ids

    def _source(self, copy: _Copy, model: Model):
        """The producer's source view id, once per producer."""
        s = self.strategy
        node = copy.edge.src
        if node not in self._src:
            cands = s._candidates(node)
            views = copy.data.source_views(cands, s._prep_cache)
            keys = [
                None if v is None else (v, cd.cores_used) for v, cd in zip(views, cands)
            ]
            ids = self._key_ids(node.name, "src", keys)
            self._src[node] = (
                s._id_var(node, ids, f"rsrc_{node.name}", model),
                sorted(set(ids) - {-1}),
            )
        return self._src[node]

    # ---- 1. vars ----

    def encode_transform(self, transform: Transform, v: NodeVars, model: Model) -> None:
        """The copy's destination view, its legal (source, destination) pairs
        with their prices, its footprint and its class; the lifetime end is a
        var for ``encode_lifetime`` to bound."""
        s = self.strategy
        copy = self.copies[transform]
        node, consumer = copy.edge.src, copy.edge.dst
        _, src_ids = self._source(copy, model)
        cands = s._candidates(consumer)
        views = copy.data.consumer_views(cands, s._prep_cache)
        keys = [
            None if w is None else (w, cd.cores_used) for w, cd in zip(views, cands)
        ]
        dst_ids = self._key_ids(node.name, "dst", keys)
        src_table = self._views[(node.name, "src")]
        dst_table = self._views.get((node.name, "dst"), [])
        alignment = s._alignment
        footprint: dict[int, int] = {}
        for b in sorted(set(dst_ids) - {-1}):
            cv, cores = dst_table[b]
            for a in src_ids:
                pv, ncores = src_table[a]
                if pv.same_partition(cv):
                    continue  # a match, not a shuffle
                priced = copy.data.price(pv, cv, ncores, cores)
                if priced is None:
                    continue
                cost, _, span = priced
                copy.legal.append((a, b, round(cost)))
                size = ceil_div(span * cores, alignment)
                footprint[b] = ceil_div(size, cores)
        v.start = copy.tick
        if not copy.legal:
            model.Add(v.in_buffer == 0)
            v.eff_size = 0
            return
        name = transform.name
        copy.dst_id = s._id_var(consumer, dst_ids, f"rdst_{name}", model)
        top = max(footprint.values())
        v.eff_size = model.IntVar(0, top, f"eff_size_{name}")
        rows = [(b, footprint.get(b, 0)) for b in sorted(set(dst_ids))]
        model.AddAllowedAssignments([copy.dst_id, v.eff_size], rows)
        classes = {
            b: s._intern(node.name, "relayout_dst", dst_table[b][0])
            for b in sorted(set(dst_ids) - {-1})
        }
        copy.view_class = model.IntVar(-1, max(classes.values()), f"rclass_{name}")
        model.AddAllowedAssignments(
            [copy.dst_id, copy.view_class],
            [(b, classes.get(b, -1)) for b in sorted(set(dst_ids))],
        )
        v.end = model.IntVar(copy.tick + 1, transform.end_time, f"end_{name}")
        copy.price = model.IntVar(0, max(p for *_, p in copy.legal), f"rprice_{name}")

    # ---- 5. reads ----

    def encode_read(self, edge: DependencyEdge, model: Model) -> None:
        """Match, or read through a copy: this edge's own or an earlier
        edge's of the same producer with the same destination view."""
        s = self.strategy
        copy = self.by_edge[edge]
        src = s.vars[edge.src]
        if not copy.legal:
            for _ in range(edge.reads):
                s._relate(edge, src.in_buffer, model)
            return
        src_id, _ = self._src[edge.src]
        fire = model.BoolVar(f"relayout_{edge.src.name}__{edge.dst.name}")
        copy.fire = fire
        model.AddAllowedAssignments(
            [src_id, copy.dst_id, copy.price], copy.legal
        ).OnlyEnforceIf(fire)
        model.AddImplication(fire, src.in_buffer)
        owners = [
            other
            for other in self.copies.values()
            if other.edge.src is edge.src
            and other.legal
            and (other.tick, other.edge.dst.name) <= (copy.tick, edge.dst.name)
        ]
        lits = []
        for owner in owners:
            lit = model.BoolVar(f"uses_{edge.dst.name}__{owner.transform.name}")
            model.AddImplication(lit, s.vars[owner.transform].in_buffer)
            if owner is not copy:
                model.Add(copy.view_class == owner.view_class).OnlyEnforceIf(lit)
            copy.uses.append((owner, lit))
            owner.users.append((copy, lit))
            lits.append(lit)
        model.Add(sum(lits) == fire)
        s.served.setdefault(edge, []).extend(
            (owner.transform, lit) for owner, lit in copy.uses
        )
        for _ in range(edge.reads):
            match = model.BoolVar(f"match_{edge.src.name}__{edge.dst.name}")
            s._relate(edge, match, model)
            model.AddBoolOr([match, fire]).OnlyEnforceIf(src.in_buffer)

    # ---- 6. transforms ----

    def encode_use(self, transform: Transform, model: Model) -> None:
        """A resident copy serves its own read first, one copy per view of a
        producer, and its price is charged once."""
        s = self.strategy
        copy = self.copies[transform]
        if not copy.legal:
            return
        own = s.vars[transform].in_buffer
        mine = [lit for owner, lit in copy.uses if owner is copy]
        model.AddImplication(own, mine[0])
        for other in self.copies.values():
            if (
                other is not copy
                and other.edge.src is copy.edge.src
                and other.legal
                and (other.tick, other.edge.dst.name) > (copy.tick, copy.edge.dst.name)
            ):
                model.Add(copy.view_class != other.view_class).OnlyEnforceIf(
                    [own, s.vars[other.transform].in_buffer]
                )
        top = max(p for *_, p in copy.legal)
        if top <= 0:
            return
        charge = model.IntVar(0, top, f"rcharge_{transform.name}")
        model.Add(charge == copy.price).OnlyEnforceIf(own)
        model.Add(charge == 0).OnlyEnforceIf(own.Not())
        self.charges.append(charge)

    # ---- 9. lifetimes ----

    def encode_lifetime(self, transform: Transform, model: Model) -> None:
        """The copy lives until its last user's read."""
        copy = self.copies[transform]
        if not copy.legal:
            return
        v = self.strategy.vars[transform]
        for user, lit in copy.users:
            model.Add(v.end >= user.tick + 1).OnlyEnforceIf(lit)

    # ---- extraction ----

    def footprint(self, transform: Transform, solver) -> int:
        """The solved per-core footprint; also tightens the copy's lifetime to
        its last user, which is what extraction places."""
        copy = self.copies[transform]
        v = self.strategy.vars[transform]
        ticks = [u.tick for u, lit in copy.users if solver.BooleanValue(lit)]
        if ticks:
            transform.end_time = max(ticks) + 1
        return v.eff_size if isinstance(v.eff_size, int) else solver.Value(v.eff_size)

    def fired(self, edge: DependencyEdge, solver) -> RelayoutCandidate:
        """The candidate the read fired through, at the solved divisions."""
        from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate

        s = self.strategy
        copy = self.by_edge[edge]
        (owner,) = [o for o, lit in copy.uses if solver.BooleanValue(lit)]
        i, src_cd = _solved_entry(edge.src)
        j, dst_cd = _solved_entry(edge.dst)
        (pv,) = copy.data.source_views([src_cd], s._prep_cache)
        (cv,) = copy.data.consumer_views([dst_cd], s._prep_cache)
        assert pv is not None and cv is not None
        priced = copy.data.price(pv, cv, src_cd.cores_used, dst_cd.cores_used)
        assert priced is not None, f"{edge} fired an illegal relayout"
        cost, source_span, destination_span = priced
        # Owner and users of one copy share its view class: the group.
        return RelayoutCandidate(
            parent=edge.src.name,
            consumer=edge.dst.name,
            source_division=i,
            consumer_division=j,
            group=solver.Value(owner.view_class),
            source_view=pv,
            destination_view=cv,
            cost_ns=cost,
            source_footprint_bytes=source_span,
            destination_footprint_bytes=destination_span,
        )
