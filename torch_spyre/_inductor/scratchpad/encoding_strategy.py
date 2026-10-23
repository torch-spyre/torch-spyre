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

"""How a :class:`SpyreOptimizationGraph` becomes a CP model.

An :class:`EncodingStrategy` encodes the graph one node, edge or transform
at a time, in the order ``SpyreOptimizationGraph._finalize`` walks
them. It owns everything about *how* decisions are searched: how each node's
work split is enumerated (a pinned value, a table of candidates), the
relations over each edge's two ends, the candidate transforms, and every
CP variable (:class:`NodeVars`). Nodes and edges stay pure data.

:class:`EncodingStrategy` holds the encoding every strategy shares, with each
open work split a ``Table`` over its menu and each edge's relation from its
clone read, its residency edge or its pair table. Three strategies refine it:
:class:`EnumeratedStrategy` keeps the menu index everywhere,
:class:`PairTableStrategy` also relates every edge by a pair table, and
``LazyStrategy`` (``lazy_strategy.py``, the default) encodes a node's work
split as constraints over its space where it can. ``config.lx_division_strategy``
picks one of the three.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import sympy

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.buffer_graph import (
    DependencyEdge,
    ExecutionEdge,
    ResourceNode,
    SpyreOptimizationGraph,
    Transform,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    ceil_div,
    division_symbol,
)

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate

Var = Any  # a backend variable or literal
Model = Any  # cp_model.CpModel until 3a, a Backend after
Item = ResourceNode | Transform


def _owner_key(view) -> Optional[tuple]:
    """A hashable form of ``view``'s owner map: equal exactly when
    ``same_partition`` holds (``same_owner_maps``), or ``None`` when an owner
    formula does not evaluate."""
    from sympy.utilities.misc import as_int

    from torch_spyre._inductor.core_mapping import _owner_at_core

    cores = view.num_cores if view.num_cores is not None else view.split_product
    if cores <= 0:
        return None
    slots = dict(view.core_to_slot)
    try:
        dims = tuple(
            (
                dim,
                split,
                tuple(as_int(_owner_at_core(slots[dim], c)) for c in range(cores)),
            )
            for dim, split in sorted(view.work_slice_dims)
            if split > 1
        )
    except (KeyError, TypeError, ValueError):
        return None
    return cores, dims


def _gate_divisions(model, compatible, src_div, dst_div, enforce_lit) -> None:
    """Enforce, when ``enforce_lit`` is true, that ``(src_div, dst_div)`` is
    one of the ``compatible`` (i, j) pairs. With no compatible pairs the
    relation is unsatisfiable, so ``enforce_lit`` is forced false."""
    if not compatible:
        model.Add(enforce_lit == 0)
        return
    # A table, instead of a Boolean and two implications per compatible pair.
    model.AddAllowedAssignments([src_div, dst_div], compatible).OnlyEnforceIf(
        enforce_lit
    )


def _sym_core_divs_of(name: str, divisions: Sequence[CoreDivision]) -> dict:
    """The per-axis split symbols of a buffer named ``name``."""
    keys = dict.fromkeys(k for cd in divisions for k in cd.splits)
    return {
        key: sympy.Symbol(f"split_{name}_{key}", integer=True, positive=True)
        for key in keys
    }


@dataclass(frozen=True)
class PrinterView:
    """What ``cpsat_expr_printer`` reads about a ``Table`` node: its
    candidates and the cost model's symbols."""

    core_divisions: tuple[CoreDivision, ...]
    sym_core_divs: Mapping[Any, sympy.Symbol]
    sym_division: sympy.Symbol


@dataclass(eq=False)
class NodeVars:
    """The variables of one node or transform. ``start`` / ``end`` are its
    lifetime, the time side of its rectangle (constants while lifetimes are
    the allocator's). ``division``, ``buffer``
    and the ``*_raw`` columns (the per-candidate values the cost lowering
    tabulates over) exist only for a ``Table``."""

    model: Model
    in_buffer: Var
    offset: Var
    eff_size: Var | int
    start: Var | int = 0
    end: Var | int = 0
    merges: dict[str, Var] = field(default_factory=dict)
    splits: dict[Any, Var | int] = field(default_factory=dict)
    tile_counts: dict[Any, Var | int] = field(default_factory=dict)
    cores: Optional[Var | int] = None
    core_cost: Optional[Var | int] = None
    division: Optional[Var] = None
    buffer: Optional[PrinterView] = None
    splits_raw: dict[Any, list[int]] = field(default_factory=dict)
    tile_counts_raw: dict[Any, list[int]] = field(default_factory=dict)
    cores_used: list[int] = field(default_factory=list)
    per_core: list[int] = field(default_factory=list)
    name: str = ""
    fixed_index: Optional[int] = None  # the index, when nothing is left to choose
    _is: dict[int, Var] = field(default_factory=dict)
    _consts: Optional[tuple[Var, Var]] = None
    # A Lazy node's per-axis encoding (no index); ties go through it.
    lazy: Optional[Any] = None

    def tie(self, values: Sequence[int], var: Var) -> None:
        """``var`` is ``values[i]`` at the chosen candidate ``i``: an element on
        the index, or for a Lazy node a table over the axes ``values`` vary
        with."""
        if self.lazy is not None:
            self.lazy.tie(values, var)
        else:
            self.model.AddElement(self.division, values, var)

    def division_is(self, i: int) -> Var:
        """Table only: the literal ``division == i``, made on first use."""
        if self._consts is not None:
            true, false = self._consts
            return true if i == self.fixed_index else false
        if i not in self._is:
            lit = self.model.BoolVar(f"div_{self.name}_is_{i}")
            self.model.Add(self.division == i).OnlyEnforceIf(lit)
            self.model.Add(self.division != i).OnlyEnforceIf(lit.Not())
            self._is[i] = lit
        return self._is[i]


@dataclass(frozen=True)
class Provenance:
    """Which call posted a range of the model (debugging side table)."""

    source: str
    element: str
    variables: range
    constraints: range

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "element": self.element,
            "variables": [self.variables.start, self.variables.stop],
            "constraints": [self.constraints.start, self.constraints.stop],
        }


# ---- enumerations of the work-split axis, internal to the strategies -------


@dataclass(frozen=True)
class _Fixed:
    """Known before the solve: ``index`` into the menu for a pinned value,
    ``None`` for COMMITTED, which posts nothing."""

    value: WorkSplit
    index: Optional[int] = None


@dataclass(frozen=True)
class _Table:
    """Explicit candidates: an index var plus element ties."""

    candidates: tuple[CoreDivision, ...]


def is_clone_read(relation: Any) -> bool:
    """Whether an edge's relation is an input clone's read, not a residency
    edge."""
    from torch_spyre._inductor.scratchpad.graph_builder import CloneRead

    return isinstance(relation, CloneRead)


@dataclass(eq=False)
class EncodingStrategy:
    """Encodes a graph into a CP model, one element at a time. Created from
    configuration alone; ``prepare(graph)`` resets the per-solve state, and
    the ``encode_*`` methods add to it.

    The encoding every strategy shares: placement, merges, relayout copies,
    coarse-tiling nests and cuts, and the menu-index work split every
    strategy falls back to (a ``Table`` over the node's menu, a pinned one
    ``Fixed``). A subclass changes how a node's work split is encoded
    (:meth:`_split_vars`, :meth:`keeps_points`) or how an edge's relation is
    (:meth:`_relation`, :meth:`_relate`).

    Its candidates are the graph's (:meth:`_collect`): each node's menu,
    filtered to its legal divisions where it is a space's points, and per edge
    its clone relation, else the pair table the builder fixed, else its
    residency relation."""

    # per-solve state
    transforms: tuple[Transform, ...] = ()
    vars: dict[Item, NodeVars] = field(default_factory=dict)
    served: dict[DependencyEdge, list[tuple[Transform, Var]]] = field(
        default_factory=dict
    )
    provenance: list[Provenance] = field(default_factory=list)
    cut_terms: list[Var] = field(default_factory=list)
    tile_terms: list[Var] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.menus: dict[str, tuple[CoreDivision, ...]] = {}
        self.matches: dict[tuple[str, str], list[tuple[int, int]]] = {}
        # Edges whose relation is derived from their ResidencyEdge (equality of
        # per-node view ids, tile-view ids and core counts) instead of a table,
        # and input-clone edges derived from the consumer's CloneRead.
        self.residency: dict[tuple[str, str], Any] = {}
        self.clone_reads: dict[tuple[str, str], Any] = {}
        # Edges whose relayouts are constraints over view ids (relayout_copies).
        self.relayout_edges: dict[tuple[str, str], Any] = {}
        self.solved: dict[Item, int] = {}

    def keeps_points(self, node: ResourceNode) -> bool:
        """Whether the node chooses among its space's points as they are,
        illegal ones included, because legality is posted another way."""
        return False

    def _collect(self, graph: SpyreOptimizationGraph) -> None:
        """The menus and edge relations this strategy encodes, from the graph.
        A point menu is filtered to its legal points (the space lists them in
        menu order), and a pair table on it renumbered to match."""
        self.menus, renumber = {}, {}
        for node in graph.get_nodes():
            if not node.menu:
                continue
            if node.point_menu and not self.keeps_points(node):
                space = node.space
                legal = [
                    i
                    for i, (k, splits) in enumerate(space.points())
                    if space.options[k].context.is_legal(splits)
                ]
                self.menus[node.name] = tuple(node.menu[i] for i in legal)
                renumber[node.name] = {old: new for new, old in enumerate(legal)}
            else:
                self.menus[node.name] = node.menu
        self.matches, self.residency, self.clone_reads = {}, {}, {}
        self.relayout_edges = {}
        for node in graph.get_nodes():
            for edge in node.inputs:
                key = (edge.src.name, node.name)
                pairs = edge.pairs
                if pairs is not None:
                    src, dst = renumber.get(key[0]), renumber.get(key[1])
                    pairs = [
                        (i if src is None else src[i], j if dst is None else dst[j])
                        for i, j in pairs
                        if (src is None or i in src) and (dst is None or j in dst)
                    ]
                self._relation(edge, key, pairs)
                if edge.relayout_edge is not None:
                    self.relayout_edges[key] = edge.relayout_edge

    def _relation(
        self,
        edge: DependencyEdge,
        key: tuple[str, str],
        pairs: Optional[list[tuple[int, int]]],
    ) -> None:
        if is_clone_read(edge.relation):
            self.clone_reads[key] = edge.relation
        elif pairs is not None:
            self.matches[key] = pairs
        elif edge.relation is not None:
            self.residency[key] = edge.relation
        else:
            # An empty table forbids the relation.
            self.matches[key] = []

    # ---- whole-graph hooks ----

    def prepare(self, graph: SpyreOptimizationGraph) -> None:
        """Reset the per-solve state, decide how each node's axes are
        enumerated and generate the candidates of table-based axes."""
        self._collect(graph)
        self.vars = {}
        self.served = {}
        self.provenance = []
        self.cut_terms = []
        self.tile_terms = []
        self.solved = {}
        self._capacity = graph.capacity
        self._alignment = graph.alignment
        self._prep_cache: dict = {}
        self._reps: dict[tuple[str, str], list] = {}
        self._rep_keys: dict[tuple[str, str], dict] = {}
        self._src_ids: dict = {}
        self._dst_ids: dict = {}
        import os

        self._check: Optional[list] = (
            [] if os.environ.get("LX_RELATION_CHECK") else None
        )
        self._incoming: dict[Item, list[Var]] = {}
        self._enum: dict[Item, Any] = {}
        self.transforms = ()
        for node in graph.get_nodes():
            self._enum[node] = self._enumeration(node)
        self._relayout_copies = None
        if self.relayout_edges:
            from torch_spyre._inductor.scratchpad.relayout_copies import RelayoutCopies

            self._relayout_copies = RelayoutCopies(self, self.relayout_edges)
            self.transforms += tuple(self._relayout_copies.prepare(graph))
        # Coarse-tiling nests over every item with a work split: their depth
        # sets the level vars, and any nest besides the untiled one turns
        # tiling on.
        self._nest_ids: dict[tuple[int, ...], int] = {(): 0}
        for node in graph.get_nodes():
            if self.chooses_split(node):
                for cd in self._candidates(node):
                    self._nest_ids.setdefault(
                        cd.tiling.level_counts, len(self._nest_ids)
                    )
        self._tiling_on = len(self._nest_ids) > 1
        self._levels: dict[Item, list[Var]] = {}
        self._tiled: dict[Item, Var] = {}
        self._group: dict[Item, Var] = {}
        self._join_prev: dict[ResourceNode, Optional[ResourceNode]] = {}
        self._chain_len = 0

    def _enumeration(self, node: ResourceNode) -> _Fixed | _Table:
        if node.committed:
            return _Fixed(WorkSplit.COMMITTED)
        menu = self.menus.get(node.name)
        assert menu, f"{node.name} has an open work split and no candidates"
        if node.work_split is not None:
            return _Fixed(node.work_split, self.menu_index(node.name, node.work_split))
        return _Table(menu)

    def menu_index(self, name: str, split: WorkSplit) -> int:
        """The first candidate of ``name``'s menu equal to ``split``."""
        for i, cd in enumerate(self.menus[name]):
            if WorkSplit.of(cd) == split:
                return i
        raise ValueError(f"{name}: work split {split} is not a candidate")

    def _candidates(self, item: Item) -> tuple[CoreDivision, ...]:
        if isinstance(item, Transform):
            return (item.division,)
        return self.menus.get(item.name, ())

    def chooses_split(self, item: Item) -> bool:
        """Whether ``item``'s work split has vars (it is not COMMITTED)."""
        enum = self._enum.get(item)
        return not (isinstance(enum, _Fixed) and enum.index is None)

    def open_choices(self, item: Item) -> int:
        """How many values the item's work split may take in the solve."""
        enum = self._enum.get(item)
        return len(enum.candidates) if isinstance(enum, _Table) else 1

    # ---- 1. vars ----

    def encode_node(self, node: ResourceNode, model: Model) -> None:
        v = self._base_vars(node, model)
        for e in node.inputs:
            if e.in_place:
                v.merges[e.src.name] = model.BoolVar(f"merge_{e.src.name}_{node.name}")
        self.vars[node] = v
        if self.chooses_split(node):
            self._split_vars(node, v, model)
        # A caller-pinned decision fixes its var; a pinned work split is
        # already a Fixed enumeration.
        if node.resident is not None:
            model.Add(v.in_buffer == int(node.resident))
        if node.offset is not None:
            model.Add(v.offset == node.offset)

    def encode_transform(self, transform: Transform, model: Model) -> None:
        v = self._base_vars(transform, model)
        self.vars[transform] = v
        self._relayouts.encode_transform(transform, v, model)

    @property
    def _relayouts(self) -> Any:
        """The lazy relayouts: every transform is one of their copies."""
        assert self._relayout_copies is not None
        return self._relayout_copies

    def _base_vars(self, item: Item, model: Model) -> NodeVars:
        return NodeVars(
            model=model,
            in_buffer=model.BoolVar(f"in_buffer_{item.name}"),
            offset=model.IntVar(0, max(0, self._capacity - 1), f"off_{item.name}"),
            eff_size=item.size,
            start=item.start_time,
            end=item.end_time,
            name=item.name,
        )

    def _syms(self, item: Item) -> tuple[Mapping, Mapping]:
        if isinstance(item, Transform):
            return _sym_core_divs_of(item.name, (item.division,)), {}
        return item.sym_core_divs, item.sym_tile_counts

    def _split_vars(self, item: Item, v: NodeVars, m: Model) -> None:
        """The candidate index and the per-candidate quantities tied to it:
        per-core footprint (the output partition and coarse tiling both
        shrink it), cores used, the balance cost (summed squared split
        factors), and the split and tile-count vars of the cost model."""
        menu = self._candidates(item)
        sym_core_divs, sym_tile_counts = self._syms(item)
        v.per_core = [
            ceil_div(item.size, cd.output_partition * cd.tiling.output_tile_count)
            for cd in menu
        ]
        cores_used = [cd.cores_used for cd in menu]
        core_cost = [sum(s**2 for s in cd.splits.values()) for cd in menu]
        v.cores_used = cores_used
        v.buffer = PrinterView(menu, sym_core_divs, division_symbol(item.name))
        enum = self._enum[item]
        fixed = (
            enum.index if isinstance(enum, _Fixed) else (0 if len(menu) == 1 else None)
        )
        if fixed is not None:
            # Nothing to choose: constants instead of element lookups on a
            # fixed index, which keeps presolve linear in the copy count.
            v.fixed_index = fixed
            v.division = m.Constant(fixed)
            v.eff_size = v.per_core[fixed]
            v.core_cost = core_cost[fixed]
            v.cores = cores_used[fixed]
            only = menu[fixed]
            v.splits = {key: only.splits.get(key, 1) for key in sym_core_divs}
            v.splits_raw = {key: [val] for key, val in v.splits.items()}
            v.tile_counts = {key: only.tiling.count_for(key) for key in sym_tile_counts}
            v.tile_counts_raw = {key: [val] for key, val in v.tile_counts.items()}
            v._consts = (m.Constant(1), m.Constant(0))
            return
        name = item.name
        v.division = m.IntVar(0, len(menu) - 1, f"div_{name}")
        v.eff_size = m.IntVar(0, max(v.per_core), f"eff_size_{name}")
        v.core_cost = m.IntVar(0, max(core_cost), f"core_cost_{name}")
        v.cores = m.IntVar(min(cores_used), max(cores_used), f"occ_{name}")
        for key, symbol in sym_core_divs.items():
            assert isinstance(symbol, sympy.Symbol)
            raw = [cd.splits.get(key, 1) for cd in menu]
            var = m.IntVar(1, config.sencores, symbol.name)
            v.tie(raw, var)
            v.splits[key] = var
            v.splits_raw[key] = raw
        for key, symbol in sym_tile_counts.items():
            raw = [cd.tiling.count_for(key) for cd in menu]
            var = m.IntVar(min(raw), max(raw), symbol.name)
            v.tie(raw, var)
            v.tile_counts[key] = var
            v.tile_counts_raw[key] = raw
        v.tie(v.per_core, v.eff_size)
        v.tie(cores_used, v.cores)
        v.tie(core_cost, v.core_cost)

    # ---- 2. merges ----

    def encode_merge(self, edge: DependencyEdge, model: Model) -> None:
        """An active merge pins parent and child to one base, both resident;
        a child that chooses a work split must also reuse the parent's exact
        per-core storage: equal footprints under divisions that slice it
        alike."""
        src, dst = self.vars[edge.src], self.vars[edge.dst]
        lit = dst.merges[edge.src.name]
        model.Add(src.offset == dst.offset).OnlyEnforceIf(lit)
        model.AddImplication(lit, src.in_buffer)
        model.AddImplication(lit, dst.in_buffer)
        if self.chooses_split(edge.dst):
            model.Add(dst.eff_size == src.eff_size).OnlyEnforceIf(lit)
            self._relate(edge, lit, model)
        self._incoming.setdefault(edge.dst, []).append(lit)

    def limit_merges(self, node: ResourceNode, model: Model, incoming: bool) -> None:
        """A storage slot is handed off linearly: at most one merge in, and at
        most one out."""
        if incoming:
            lits = list(self.vars[node].merges.values())
        else:
            lits = [
                self.vars[e.dst].merges[node.name] for e in node.outputs if e.in_place
            ]
        if len(lits) > 1:
            model.AddAtMostOne(lits)

    def _pairs(self, edge: DependencyEdge) -> list[tuple[int, int]]:
        return self.matches.get((edge.src.name, edge.dst.name), [])

    def _relate(self, edge: DependencyEdge, lit: Var, model: Model) -> None:
        """When ``lit`` holds, the two ends' divisions slice the shared buffer
        alike: a pair table, or for a derived edge equal view ids, tile-view
        ids and core counts."""
        key = (edge.src.name, edge.dst.name)
        re = self.residency.get(key)
        src, dst = self.vars[edge.src], self.vars[edge.dst]
        clone = self.clone_reads.get(key)
        if clone is not None:
            model.Add(
                src.division == self._clone_ids(edge, clone, model)
            ).OnlyEnforceIf(lit)
            return
        if re is None:
            _gate_divisions(model, self._pairs(edge), src.division, dst.division, lit)
            return
        sv, st = self._source_ids(edge.src, re, model)
        dv, dt = self._consumer_ids(edge, re, model)
        model.Add(sv == dv).OnlyEnforceIf(lit)
        model.Add(st == dt).OnlyEnforceIf(lit)
        model.Add(src.cores == dst.cores).OnlyEnforceIf(lit)

    # ---- derived relation: per-node view ids ----

    def _intern(self, buffer: str, kind: str, view) -> int:
        """A small id per distinct view of ``buffer`` (``same_partition``
        classes), shared by the producer and every consumer."""
        if view is None:
            return -1
        reps = self._reps.setdefault((buffer, kind), [])
        keyed = self._rep_keys.setdefault((buffer, kind), {})
        key = _owner_key(view)
        if key is not None and key in keyed:
            return keyed[key]
        for i, rep in enumerate(reps):
            if rep.same_partition(view):
                if key is not None:
                    keyed[key] = i
                return i
        reps.append(view)
        if key is not None:
            keyed[key] = len(reps) - 1
        return len(reps) - 1

    def _id_var(self, item: Item, ids: list[int], name: str, model: Model):
        v = self.vars[item]
        if v.fixed_index is not None:
            return ids[v.fixed_index]
        if min(ids) == max(ids):
            return ids[0]
        var = model.IntVar(min(ids), max(ids), name)
        v.tie(ids, var)
        return var

    def _source_ids(self, node: ResourceNode, re, model: Model):
        """The producer's write-view id and tile-view id, once per producer."""
        if node not in self._src_ids:
            from torch_spyre._inductor.work_division import _tile_view_for_div

            menu = self._candidates(node)
            core = [self._intern(node.name, "core", re.parent_view(cd)) for cd in menu]
            tile = [
                self._intern(
                    node.name,
                    "tile",
                    _tile_view_for_div(
                        re.parent_op, re.write_dep, re.buf_name, cd, re.prep_cache
                    ),
                )
                for cd in menu
            ]
            self._src_ids[node] = (
                self._id_var(node, core, f"wview_{node.name}", model),
                self._id_var(node, tile, f"wtile_{node.name}", model),
                core,
                tile,
            )
        return self._src_ids[node][:2]

    def _consumer_ids(self, edge: DependencyEdge, re, model: Model):
        """The consumer's read-view id and tile-view id of the edge's buffer;
        an unrepresentable read gets an id no write view has."""
        if edge not in self._dst_ids:
            menu = self._candidates(edge.dst)
            core = [
                self._intern(edge.src.name, "core", re.consumer_view(cd)) for cd in menu
            ]
            tile = [
                self._intern(edge.src.name, "tile", re.consumer_tile_view(cd))
                for cd in menu
            ]
            core = [-2 if i == -1 else i for i in core]
            tile = [-2 if i == -1 else i for i in tile]
            name = f"{edge.src.name}__{edge.dst.name}"
            self._dst_ids[edge] = (
                self._id_var(edge.dst, core, f"rview_{name}", model),
                self._id_var(edge.dst, tile, f"rtile_{name}", model),
                core,
                tile,
            )
            if self._check is not None:
                self._check_edge(edge, re)
        return self._dst_ids[edge][:2]

    def _clone_ids(self, edge: DependencyEdge, clone, model: Model):
        """The input-clone view the consumer reads under its division: an
        index into the clone's menu, -2 where no clone view serves it."""
        if edge not in self._dst_ids:
            ids = clone.clone_index(self._candidates(edge.dst))
            self._dst_ids[edge] = (
                self._id_var(
                    edge.dst, ids, f"cview_{edge.src.name}__{edge.dst.name}", model
                ),
                ids,
            )
        return self._dst_ids[edge][0]

    def _check_edge(self, edge: DependencyEdge, re) -> None:
        """Agreement test: the pairs the derived relation admits must be the
        pair table ``match_pairs`` would build."""
        src_menu, dst_menu = self._candidates(edge.src), self._candidates(edge.dst)
        _, _, sc, stile = self._src_ids[edge.src]
        _, _, dc, dtile = self._dst_ids[edge]
        derived = {
            (i, j)
            for i in range(len(src_menu))
            for j in range(len(dst_menu))
            if sc[i] >= 0
            and sc[i] == dc[j]
            and stile[i] >= 0
            and stile[i] == dtile[j]
            and src_menu[i].cores_used == dst_menu[j].cores_used
        }
        table = set(re.match_pairs(list(src_menu), list(dst_menu)))
        import json
        import os

        record = {
            "edge": repr(edge),
            "pairs": len(table),
            "equal": derived == table,
            "missing": len(table - derived),
            "extra": len(derived - table),
        }
        assert self._check is not None
        self._check.append(record)
        with open(os.environ["LX_RELATION_CHECK"], "a") as f:
            f.write(json.dumps(record) + "\n")

    # ---- 3. placement ----

    def encode_capacity(self, item: Item, model: Model) -> None:
        v = self.vars[item]
        model.Add(v.offset + v.eff_size <= self._capacity).OnlyEnforceIf(v.in_buffer)

    def encode_no_overlap(self, graph: SpyreOptimizationGraph, model: Model) -> None:
        """Every item's optional rectangle ``[start, end) x [offset, offset +
        eff_size)``. An active incoming merge shortens the child (never the
        parent, whose bytes above a smaller child are still read on that tick)
        by the one handoff tick, so the two abut and may share an offset."""
        x_intervals, y_intervals = [], []
        items: list[Item] = [*graph.get_nodes(), *self.transforms]
        for item in items:
            v = self.vars[item]
            ins = self._incoming.get(item, [])
            if ins:
                start = model.IntVar(v.start, v.end, f"start_{item.name}")
                model.Add(start == v.start + sum(ins))
                x_start: Any = start
                x_size: Any = v.end - start
            else:
                x_start = v.start
                x_size = v.end - v.start
            x_intervals.append(
                model.NewOptionalIntervalVar(
                    x_start, x_size, v.end, v.in_buffer, f"x_{item.name}"
                )
            )
            # An interval end must be affine, so the top of a division-dependent
            # footprint gets its own var; its range covers buffers larger
            # than LX itself.
            y_end = model.IntVar(
                0, max(0, self._capacity - 1) + item.size, f"top_{item.name}"
            )
            model.Add(y_end == v.offset + v.eff_size)
            y_intervals.append(
                model.NewOptionalIntervalVar(
                    v.offset, v.eff_size, y_end, v.in_buffer, f"y_{item.name}"
                )
            )
        model.AddNoOverlap2D(x_intervals, y_intervals)

    # ---- 4. op constraints ----

    def encode_op_constraints(self, item: Item, model: Model) -> None:
        if item.residency_reason is not None:
            model.Add(self.vars[item].in_buffer == 0)

    # ---- 5. reads ----

    def encode_read(self, edge: DependencyEdge, model: Model) -> None:
        """A resident producer's division must slice it as the consumer's
        does, or the consumer reads it through a resident relayout copy
        ("match or served"). Gated only when both ends choose a work split,
        and once per read the consumer lists."""
        if not (self.chooses_split(edge.src) and self.chooses_split(edge.dst)):
            return
        if (edge.src.name, edge.dst.name) in self.clone_reads:
            return  # encode_clone
        if self._relayout_copies is not None and edge in self._relayout_copies.by_edge:
            self._relayout_copies.encode_read(edge, model)
            return
        for _ in range(edge.reads):
            self._relate(edge, self.vars[edge.src].in_buffer, model)

    def encode_transform_use(self, transform: Transform, model: Model) -> None:
        """A resident relayout copy needs its producer resident and serves
        some read (``relayout_copies``)."""
        self._relayouts.encode_use(transform, model)

    def encode_tiling_vars(self, item: Item, model: Model) -> None:
        """The item's tile count per nest level (0 past its depth; tied to its
        division like ``eff_size``) and whether it is tiled."""
        if not self._tiling_on or isinstance(item, Transform):
            return
        v = self.vars[item]
        cands = self._candidates(item)
        depth = max(len(k) for k in self._nest_ids)
        levels = []
        for k in range(depth):
            vals = [
                cd.tiling.level_counts[k] if k < len(cd.tiling.level_counts) else 0
                for cd in cands
            ]
            lk = model.IntVar(min(vals), max(vals), f"L{k}_{item.name}")
            v.tie(vals, lk)
            levels.append(lk)
        tiled = model.BoolVar(f"tiled_{item.name}")
        model.Add(levels[0] != 0).OnlyEnforceIf(tiled)
        model.Add(levels[0] == 0).OnlyEnforceIf(tiled.Not())
        self._levels[item] = levels
        self._tiled[item] = tiled

    def tileable(self, node: ResourceNode) -> bool:
        return self.chooses_split(node) and any(
            not cd.tiling.is_untiled for cd in self._candidates(node)
        )

    def tiling_segments(self, graph: SpyreOptimizationGraph) -> dict[ResourceNode, int]:
        """Walk the execution chain over the tileable ops. Consecutive ops
        share a loop nest only on consecutive ticks, so an op one tick after
        the previous tileable op continues its segment and any other starts
        a new one: an untileable op in between always breaks the run, one on
        the same tick (another output of a multi-output op) does not."""
        segments: dict[ResourceNode, int] = {}
        if not self._tiling_on:
            return segments
        prev: Optional[ResourceNode] = None
        count = 0
        node = graph.first
        while node is not None:
            if self.tileable(node):
                assert node.produced_at is not None
                if prev is not None and node.produced_at - 1 == prev.produced_at:
                    segments[node] = segments[prev]
                    self._join_prev[node] = prev
                else:
                    segments[node] = count
                    count += 1
                    self._join_prev[node] = None
                prev = node
            node = node.next.after if node.next else None
        self._chain_len = len(segments)
        return segments

    def encode_chain_head(self, node: ResourceNode, model: Model) -> None:
        """The first op of the chain has no incoming execution edge."""
        self._encode_group(node, model)

    def encode_nest(self, edge: ExecutionEdge, model: Model) -> None:
        """The loop group of ``edge.after``: a fresh 0 at a segment start,
        else its predecessor's group stepped by ``1 - joined``, where
        ``joined`` means both run the same non-empty nest."""
        self._encode_group(edge.after, model)

    def _encode_group(self, node: ResourceNode, model: Model) -> None:
        if node not in self._join_prev:
            return
        prev = self._join_prev[node]
        if prev is None:
            self._group[node] = model.Constant(0)
            return
        a, b = prev.name, node.name
        same = model.BoolVar(f"samenest_{a}_{b}")
        # Same nest: every level's tile count equal (0 past a nest's depth).
        eqs = []
        for k, (la, lb) in enumerate(zip(self._levels[prev], self._levels[node])):
            e = model.BoolVar(f"eq{k}_{a}_{b}")
            model.Add(la == lb).OnlyEnforceIf(e)
            model.Add(la != lb).OnlyEnforceIf(e.Not())
            eqs.append(e)
        model.AddBoolAnd(eqs).OnlyEnforceIf(same)
        model.AddBoolOr([e.Not() for e in eqs]).OnlyEnforceIf(same.Not())
        joined = model.BoolVar(f"joined_{a}_{b}")
        model.AddBoolAnd([same, self._tiled[node]]).OnlyEnforceIf(joined)
        model.AddBoolOr([same.Not(), self._tiled[node].Not()]).OnlyEnforceIf(
            joined.Not()
        )
        var = model.IntVar(0, self._chain_len, f"tile_group_{b}")
        model.Add(var == self._group[prev] + 1 - joined)
        self._group[node] = var

    def encode_cuts(
        self, item: Item, segments: Mapping[ResourceNode, int], model: Model
    ) -> None:
        """The item's cut literal: tiled, and its value must be published to
        HBM because a consumer sits outside its loop nest. A consumer outside
        its segment, or a graph output, makes it unshareable; a consumer in
        its segment gets an ``apart`` literal, and sharing the nest requires
        their divisions to slice it alike."""
        if item not in self._tiled:
            return
        is_tiled = self._tiled[item]
        diffs: list[Var] = []
        unshareable = getattr(item, "boundary", None) == BufferType.Output
        seg = segments.get(item) if isinstance(item, ResourceNode) else None
        outputs = item.outputs if isinstance(item, ResourceNode) else []
        for edge, child in ((e, e.dst) for e in outputs for _ in range(e.reads)):
            if not self.chooses_split(child):
                continue
            if seg is None or segments.get(child) != seg:
                unshareable = True
                continue
            d = model.BoolVar(f"apart_{item.name}_{child.name}")
            model.Add(self._group[item] != self._group[child]).OnlyEnforceIf(d)
            model.Add(self._group[item] == self._group[child]).OnlyEnforceIf(d.Not())
            self._relate(edge, d.Not(), model)
            diffs.append(d)
        cut = model.BoolVar(f"cut_{item.name}")
        if unshareable or not diffs:
            model.Add(cut == is_tiled)
        else:
            any_diff = model.BoolVar(f"anydiff_{item.name}")
            model.AddMaxEquality(any_diff, diffs)
            model.AddBoolAnd([is_tiled, any_diff]).OnlyEnforceIf(cut)
            model.AddBoolOr([is_tiled.Not(), any_diff.Not()]).OnlyEnforceIf(cut.Not())
        self.cut_terms.append(cut)

    def encode_tile_counts(self, item: Item, model: Model) -> None:
        """The tile count of the chosen tiling, when the candidates differ."""
        if self.open_choices(item) < 2:
            return
        counts = [cd.tiling.tile_count for cd in self._candidates(item)]
        if min(counts) == max(counts):
            return
        var = model.IntVar(min(counts), max(counts), f"tile_count_{item.name}")
        self.vars[item].tie(counts, var)
        self.tile_terms.append(var)

    # ---- 8. clones ----

    def encode_clone(self, node: ResourceNode, model: Model) -> None:
        """A resident input clone is read alike by every consumer that
        chooses a work split: each one's read view is the clone's (derived
        clone edges; a table edge was posted as an ordinary read)."""
        if not self.chooses_split(node):
            return
        in_buffer = self.vars[node].in_buffer
        for edge in node.outputs:
            if (node.name, edge.dst.name) not in self.clone_reads:
                continue
            if self.chooses_split(edge.dst):
                for _ in range(edge.reads):
                    self._relate(edge, in_buffer, model)

    # ---- 9. lifetimes ----

    def encode_lifetime(self, item: Item, model: Model) -> None:
        """Lifetimes are the allocator's constants, except a lazy relayout
        copy's, which lives until its last user."""
        if isinstance(item, Transform):
            self._relayouts.encode_lifetime(item, model)

    def decision_variables(self) -> list[Var]:
        """The vars that define the plan: residency, merges, served relayouts
        and each work split (a menu index, or a Lazy node's option and
        splits). Offsets are left out: no objective prices an address, so every
        legal shift of a resident buffer would read as a tie."""
        out: list = []
        for v in self.vars.values():
            out += [v.in_buffer, *v.merges.values()]
            if v.lazy is not None:
                out += v.lazy.columns
            elif v.division is not None:
                out.append(v.division)
        out += [lit for reads in self.served.values() for _, lit in reads]
        return [x for x in out if not isinstance(x, int)]

    def extra_cost_terms(self) -> list[Var]:
        """Objective terms the strategy prices itself (lazy relayout
        charges), in the cost expression's unit."""
        return [] if self._relayout_copies is None else self._relayout_copies.charges

    # ---- objective inputs and extraction ----

    def objective_terms(self, item: Item) -> tuple[Optional[Var], Optional[Var]]:
        """``(cores, core_cost)`` for the parallelism and shape stages, from an
        open work split only: a fixed one is a constant that cannot move the
        optimum."""
        if self.open_choices(item) < 2:
            return None, None
        v = self.vars[item]
        return v.cores, v.core_cost

    def cost_bindings(self) -> tuple[dict, dict]:
        """``sym_map`` / ``buffer_map`` for lowering the cost expression: each
        cost-model symbol to its var, and the ``_division_of_*`` entries that
        let a table over candidates select by identity."""
        sym_map: dict = {}
        buffer_map: dict = {}
        for item, v in self.vars.items():
            sym_map[item.sym_is_lx.name] = v.in_buffer
            if v.buffer is None:
                continue
            sym_map[v.buffer.sym_division.name] = v.division
            sym_map[f"_division_of_{v.buffer.sym_division.name}"] = v
            product = []
            for key, symbol in v.buffer.sym_core_divs.items():
                sym_map[symbol.name] = v.splits[key]
                buffer_map[symbol.name] = (v, v.splits_raw[key])
                product.append(symbol.name)
            product.sort()
            symbol = sympy.Symbol("_product_" + "_".join(product))
            sym_map[symbol.name] = v.cores
            buffer_map[symbol.name] = (v, v.cores_used)
            _, sym_tile_counts = self._syms(item)
            for key, symbol in sym_tile_counts.items():
                sym_map[symbol.name] = v.tile_counts[key]
                buffer_map[symbol.name] = (v, v.tile_counts_raw[key])
        return sym_map, buffer_map

    def chosen_split(self, item: Item, solver) -> WorkSplit:
        v = self.vars[item]
        if v.division is None:
            return WorkSplit.COMMITTED
        i = self.solved[item] = solver.Value(v.division)
        return WorkSplit.of(self._candidates(item)[i])

    def footprint(self, item: Item, solver) -> int:
        if isinstance(item, Transform):
            return self._relayouts.footprint(item, solver)
        v = self.vars[item]
        return item.size if v.division is None else v.per_core[solver.Value(v.division)]

    def fired(self, edge: DependencyEdge, solver) -> RelayoutCandidate:
        """The relayout candidate the read on ``edge`` fired through."""
        assert edge.transform is not None
        return self._relayouts.fired(edge, solver)


class EnumeratedStrategy(EncodingStrategy):
    """Every open work split is an index into its enumerated menu: the base
    encoding, with no space standing in for any node's menu."""


class PairTableStrategy(EnumeratedStrategy):
    """The table model in full: every edge's relation is a pair table over
    its two ends' menu indices, never a derived clone or residency relation:
    the builder's table where it fixed one, else the residency relation's
    pairs; an edge with neither forbids residency across it."""

    def _relation(
        self,
        edge: DependencyEdge,
        key: tuple[str, str],
        pairs: Optional[list[tuple[int, int]]],
    ) -> None:
        if pairs is None and edge.relation is not None:
            pairs = edge.relation.match_pairs(
                list(self.menus[key[0]]), list(self.menus[key[1]])
            )
        self.matches[key] = pairs or []
