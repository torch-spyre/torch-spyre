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

"""The ``Lazy`` work-split encoding (Phase 2f + 2a, experimental).

A Lazy node has no candidate index. Its work split is one var per loop axis
plus a tiling choice, constrained by :func:`post_division`, and
every per-division quantity (footprint, core cost, tile counts, view ids, cost
terms) is a table over only the axes it varies with, tabulated over the
space's points rather than the menu. Nodes that cannot be Lazy stay ``Table``:
a pinned or committed split, an op with no space, and an end of an edge whose
relation is still a pair table over menu indices (loop carries, input clones).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import sympy

from torch_spyre._inductor import config
from torch_spyre._inductor import work_division as wd

from torch_spyre._inductor.scratchpad.buffer_graph import (
    DependencyEdge,
    ResourceNode,
    SpyreOptimizationGraph,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.encoding_strategy import (
    Item,
    Model,
    NodeVars,
    PrinterView,
    EncodingStrategy,
    _Fixed,
    _Table,
    is_clone_read,
)
from torch_spyre._inductor.scratchpad.division_space import DivisionSpace
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    ceil_div,
    division_symbol,
)

logger = logging.getLogger(__name__)

# Spaces with more points than this stay Table: every quantity of a Lazy node
# is tabulated over its points.
MAX_POINTS = 4096

Var = Any


def _span_terms(ctx: wd.WorkDivisionContext, td, domain: dict) -> tuple[list, int]:
    """Per outer device dim with a loop symbol: ``(stride_bytes, {axis: {split:
    term}})``; that dim's per-core extent is ``1 + sum(terms)``. ``domain`` is
    each axis's posted domain, so every term table is total."""
    device_size = td.layout.device_layout.device_size
    itemsize = td.layout.dtype.itemsize
    dims = []
    for d, coord in enumerate(td.device_coords[:-1]):
        if not coord.free_symbols:
            continue
        per_axis = {}
        for v in coord.free_symbols:
            term = coord.subs(dict.fromkeys(coord.free_symbols - {v}, 0))
            base = int(term.subs(v, 0))
            size = wd._effective_size(v, ctx.it_space, ctx.symbol_meta)
            values = domain.get(v, [1]) if v in ctx.it_space_adjusted else [1]
            per_axis[v] = {s: int(term.subs(v, size // s - 1)) - base for s in values}
        stride = math.prod(device_size[d + 1 :]) * itemsize
        dims.append((stride, per_axis))
    return dims, itemsize


def post_division(
    model, space: DivisionSpace, name: str
) -> tuple[Var | int, dict[sympy.Symbol, Var], list[Var | int], Var | int]:
    """Post ``space``'s legal divisions: ``(option, splits, option literals,
    cores)``. ``option`` and ``cores`` are constants when nothing is left to
    choose."""
    axes = space.axes
    domain = {a: space.domain(a) for a in axes}
    splits: dict[sympy.Symbol, Var] = {}
    for a in axes:
        var = model.IntVar(min(domain[a]), max(domain[a]), f"split_{name}_{a}")
        if len(domain[a]) < max(domain[a]) - min(domain[a]) + 1:
            model.AddAllowedAssignments([var], [(d,) for d in domain[a]])
        splits[a] = var
    k = len(space.options)
    option: Var | int
    if k == 1:
        option, lits = 0, [1]
    else:
        option = model.IntVar(0, k - 1, f"tiling_{name}")
        lits = []
        for i in range(k):
            lit = model.BoolVar(f"tiling_{name}_is_{i}")
            model.Add(option == i).OnlyEnforceIf(lit)
            model.Add(option != i).OnlyEnforceIf(lit.Not())
            lits.append(lit)
    cores: Var | int = 1
    for a in axes:
        if isinstance(cores, int):
            cores = splits[a]
            continue
        product = model.IntVar(1, space.max_cores, f"cores_{name}_{a}")
        model.AddMultiplicationEquality(product, [cores, splits[a]])
        cores = product
    if not isinstance(cores, int):
        model.Add(cores <= space.max_cores)
    for i, o in enumerate(space.options):
        _post_option(model, o.context, splits, domain, lits[i], f"{name}_{i}")
    return option, splits, lits, cores


def _enforce(constraint, lit) -> None:
    if lit is not True:
        constraint.OnlyEnforceIf(lit)


def _post_option(model, ctx, splits, domain, lit, name: str) -> None:
    """Under ``lit``, :meth:`WorkDivisionContext.is_legal` for ``ctx``."""
    if isinstance(lit, int):
        lit = True

    def infeasible() -> None:
        if lit is True:
            raise ValueError(f"{name}: the only tiling option has no legal division")
        model.Add(lit == 0)

    for a, var in splits.items():
        allowed = set(ctx.factor_domain(a)) if a in ctx.axes else {1}
        if not allowed:
            infeasible()
            return
        for value in domain[a]:
            if value not in allowed:
                _enforce(model.Add(var != value), lit)
    split_reductions = [v for v in ctx.reduction_vars if v in splits]
    if len(split_reductions) > 1:
        reductions = []
        for v in split_reductions:
            b = model.BoolVar(f"red_{name}_{v}")
            model.Add(splits[v] > 1).OnlyEnforceIf(b)
            model.Add(splits[v] == 1).OnlyEnforceIf(b.Not())
            reductions.append(b)
        _enforce(model.Add(sum(reductions) <= 1), lit)
    for v in ctx.constraints.blocked:
        if v in splits:
            _enforce(model.Add(splits[v] == 1), lit)
    for v, allowed in ctx.constraints.allowed_splits.items():
        if v in splits:
            for value in domain[v]:
                if value not in allowed:
                    _enforce(model.Add(splits[v] != value), lit)
    for v, minimum in ctx.span_min_splits.items():
        if v in splits:
            _enforce(model.Add(splits[v] >= minimum), lit)
        elif minimum > 1:
            infeasible()
            return
    for ti, td in enumerate(ctx.tensor_deps):
        dims, itemsize = _span_terms(ctx, td, domain)
        if itemsize > wd.MAX_SPAN_BYTES:
            infeasible()
            return
        earlier: list = []
        for di, (stride, per_axis) in enumerate(dims):
            parts: list = []
            for v, table in per_axis.items():
                # A split this frame does not adjust leaves the term at its
                # value for split 1.
                if v in splits and v in ctx.axes and len(table) > 1:
                    lo, hi = min(table.values()), max(table.values())
                    if lo == hi:
                        parts.append(lo)
                        continue
                    t = model.IntVar(lo, hi, f"span_{name}_{ti}_{di}_{v}")
                    model.AddAllowedAssignments([splits[v], t], list(table.items()))
                    parts.append(t)
                else:
                    parts.append(table.get(1, 0))
            cap = wd.MAX_SPAN_BYTES // stride
            if all(isinstance(p, int) for p in parts):
                size = 1 + sum(parts)
                if size == 1:
                    continue
                # A fixed extent above one decides the span unless an earlier
                # dim already did.
                if size > cap:
                    if not earlier:
                        infeasible()
                        return
                    _enforce(model.AddBoolOr(earlier), lit)
                break
            # Unbounded below: the term tables are posted for every option, and
            # a split legal elsewhere can make this frame's extent negative.
            size_var = model.IntVar(-(1 << 40), 1 << 40, f"extent_{name}_{ti}_{di}")
            model.Add(size_var == 1 + sum(parts))
            gt1 = model.BoolVar(f"gt1_{name}_{ti}_{di}")
            model.Add(size_var > 1).OnlyEnforceIf(gt1)
            model.Add(size_var <= 1).OnlyEnforceIf(gt1.Not())
            model.Add(size_var <= cap).OnlyEnforceIf(
                [gt1, *[e.Not() for e in earlier], *_lits(lit)]
            )
            earlier.append(gt1)


def _lits(lit) -> list:
    return [] if lit is True else [lit]


def legal_points(space: DivisionSpace) -> set[tuple[int, tuple]]:
    """Every ``(option, splits over space.axes)`` the posted constraints admit,
    by enumerating the solutions of the division model alone: the agreement
    check against the menu."""
    from ortools.sat.python import cp_model

    from torch_spyre._inductor.scratchpad.backend import CpSat

    backend = CpSat()
    option, splits, _, _ = post_division(backend, space, "check")
    axes = space.axes
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    found: set = set()

    class Collect(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            k = option if isinstance(option, int) else self.Value(option)
            found.add((k, tuple(self.Value(splits[a]) for a in axes)))

    solver.Solve(backend.model, Collect())
    return found


def menu_points(
    space: DivisionSpace, menu: Sequence[CoreDivision]
) -> set[tuple[int, tuple]]:
    """The menu in :func:`legal_points`' terms."""
    by_tiling = {o.tiling: k for k, o in enumerate(space.options)}
    axes = space.axes
    return {
        (by_tiling.get(cd.tiling, -1), tuple(cd.splits.get(a, 1) for a in axes))
        for cd in menu
    }


@dataclass(frozen=True)
class _Digits:
    """A per-core view of one buffer in digit form (2c): each device dim the
    view can split is owned by one loop symbol, whose split is that dim's
    factor and whose core stride is the product of the splits before it in
    core order. Derived from the candidate-invariant view prep, for the cases
    ``_per_core_view_from_prep`` places by a direct stride lookup; anything
    else (multi-stick rescue, fused decompositions, two symbols on one dim)
    has no digit form."""

    owners: tuple[tuple[int, Any], ...]  # (device dim, owning loop symbol)
    # Per owning symbol, the split values whose view is representable.
    allowed: tuple[tuple[Any, frozenset], ...]
    # (a, b): a split of a > 1 is representable only with a split of b > 1
    # (an outer symbol on the same device dim may not be left unsplit).
    interleave: tuple[tuple[Any, Any], ...]
    order: tuple[Any, ...]  # the op's iteration order
    contiguous: Any  # the symbol core order moves first when split, or None

    def stride(self, sym, splits: dict) -> int:
        before = self.order[: self.order.index(sym)]
        out = math.prod(splits.get(s, 1) for s in before)
        if self.contiguous is not None and sym != self.contiguous:
            if self.contiguous not in before:
                out *= splits.get(self.contiguous, 1)
        elif self.contiguous is not None and sym == self.contiguous:
            out = 1 if splits.get(sym, 1) > 1 else out
        return out

    def key(self, splits: dict) -> Optional[tuple]:
        """The view under ``splits`` as a comparable key, or ``None`` when it
        is unrepresentable: what ``_owner_key`` gives for the real view."""
        allowed = dict(self.allowed)
        for sym, ok in allowed.items():
            if splits.get(sym, 1) not in ok:
                return None
        for a, b in self.interleave:
            if splits.get(a, 1) > 1 and splits.get(b, 1) <= 1:
                return None
        dims = tuple(
            (d, splits.get(sym, 1), self.stride(sym, splits))
            for d, sym in sorted(self.owners, key=lambda o: o[0])
            if splits.get(sym, 1) > 1
        )
        return math.prod(splits.values()), dims


def _digits(prep, domain: dict) -> Optional[_Digits]:
    """``prep``'s view in digit form over split values ``domain``, or ``None``
    where some split in the domain leaves the direct stride lookup."""
    from torch_spyre._inductor.core_mapping import (
        _LOOP_POINT,
        direct_axis_ownership_failure,
    )
    from torch_spyre._inductor.pass_utils import concretize_expr

    if prep is None:
        return None
    owners: dict[int, Any] = {}
    allowed: dict[Any, frozenset] = {}
    interleave: list[tuple[Any, Any]] = []
    seen_strides: set[int] = set()
    for sym in prep.iter_space:
        h = prep.dep_coeff.get(sym, 0)
        values = [v for v in domain.get(sym, [1]) if v > 1]
        if h == 0 or not values:
            continue
        if h in seen_strides:
            return None
        seen_strides.add(h)
        dev = prep.device_stride_to_dim.get(h)
        extent = prep.iter_space[sym]
        extent = concretize_expr(extent[0] if isinstance(extent, tuple) else extent)
        if h == prep.stick_host_stride:
            dev = prep.num_stick_dim
            if dev is not None and extent > prep.num_stick * prep.elems_per_stick:
                allowed[sym] = frozenset({1})
                continue
        if dev is None or dev in owners:
            return None
        coords = prep.dep_device_coordinates
        multi = sum(sym in c.free_symbols for c in coords) > 1
        for split in values:
            if prep.device_size[dev] % split:
                return None
            if multi:
                proof_extent = extent
                padded = prep.num_stick * prep.elems_per_stick
                if (
                    h == prep.stick_host_stride
                    and padded - prep.elems_per_stick < extent <= padded
                ):
                    proof_extent = padded
                if direct_axis_ownership_failure(
                    proof_extent,
                    split,
                    coords[dev].xreplace({sym: _LOOP_POINT}),
                    prep.device_size[dev],
                    split,
                ):
                    return None
        for other in coords[dev].free_symbols:
            if other != sym and prep.dep_coeff.get(other, 0) > h:
                interleave.append((sym, other))
        owners[dev] = sym
        allowed[sym] = frozenset({1, *values})
    order = tuple(prep.iter_space)
    contiguous = (
        order[-1]
        if prep.is_matmul and config.core_id_k_fast_emission and order
        else None
    )
    return _Digits(
        tuple(owners.items()),
        tuple(allowed.items()),
        tuple(interleave),
        order,
        contiguous,
    )


def _why_illegal(space: DivisionSpace, k: int, splits: tuple) -> list[str]:
    """Which ``is_legal`` checks a point fails (agreement-check diagnostics)."""
    ctx = space.options[k].context
    s = dict(zip(space.axes, splits))
    s = {a: v for a, v in s.items() if a in ctx.axes}
    checks = (
        "_factors_in_domain",
        "_within_core_budget",
        "_one_reduction_split_at_most",
        "_spans_within_cap",
        "_in_split_domains",
        "meets_span_floors",
    )
    return [c for c in checks if not getattr(ctx, c)(s)]


def _not(lit):
    return 1 - lit if isinstance(lit, int) else lit.Not()


def _post_eq(model, a, b, lits: list) -> None:
    """``a == b`` under ``lits``; either side may be a constant."""
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            model.AddBoolOr([x.Not() for x in lits])
        return
    model.Add(a == b).OnlyEnforceIf(lits)


class _LazyVars:
    """A Lazy node's split encoding and its points, which ``tie`` tabulates
    over."""

    def __init__(self, model: Model, space: DivisionSpace, option, splits) -> None:
        self.model = model
        self.space = space
        self.option = option
        self.splits = splits
        axes = space.axes
        points = space.points()
        self.coords = [(k, *(s.get(a, 1) for a in axes)) for k, s in points]
        self.columns = [option, *(splits[a] for a in axes)]
        self._ties: dict[tuple, Any] = {}

    def tie(self, values: Sequence[int], var) -> None:
        values = list(values)
        key = tuple(values)
        if key in self._ties:
            self.model.Add(var == self._ties[key])
            return
        self._ties[key] = var
        cols = [
            c
            for c in range(len(self.columns))
            if not isinstance(self.columns[c], int)
            and len({p[c] for p in self.coords}) > 1
        ]
        for c in list(cols):
            trial = [d for d in cols if d != c]
            if self._determines(trial, values):
                cols = trial
        if not cols:
            self.model.Add(var == values[0])
            return
        rows = sorted(
            {tuple(p[c] for c in cols) + (v,) for p, v in zip(self.coords, values)}
        )
        self.model.AddAllowedAssignments([*(self.columns[c] for c in cols), var], rows)

    def _determines(self, cols: list[int], values: list) -> bool:
        seen: dict[tuple, Any] = {}
        for p, v in zip(self.coords, values):
            k = tuple(p[c] for c in cols)
            if seen.setdefault(k, v) != v:
                return False
        return True


class LazyStrategy(EncodingStrategy):
    """A ``Lazy`` work split wherever a node can take one: its legality posted
    as constraints over the node's space, never enumerated. Any other node
    keeps the base menu index."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._lazy: dict[Item, DivisionSpace] = {}
        self._grid: dict[Item, tuple[CoreDivision, ...]] = {}
        self._digit_cache: dict = {}

    def keeps_points(self, node: ResourceNode) -> bool:
        return node in self._lazy

    def prepare(self, graph: SpyreOptimizationGraph) -> None:
        # A node a pair table indexes chooses from a menu. An empty table
        # forbids the relation whatever the indices are, so it does not count.
        tabled = {
            name
            for node in graph.get_nodes()
            for edge in node.inputs
            if not is_clone_read(edge.relation) and edge.pairs
            for name in (edge.src.name, node.name)
        }
        self._lazy, self._grid = {}, {}
        for node in graph.get_nodes():
            space = node.space
            if (
                space is None
                or node.committed
                or node.work_split is not None
                or node.name in tabled
                or len(space.points()) > MAX_POINTS
            ):
                continue
            self._lazy[node] = space
            self._grid[node] = tuple(space.division(k, s) for k, s in space.points())
        logger.debug(
            "lazy work splits: %d of %d nodes", len(self._lazy), len(graph.nodes)
        )
        super().prepare(graph)
        check = os.environ.get("LX_LAZY_CHECK")
        if check:
            self._check_spaces(check)

    def _check_spaces(self, path: str) -> None:
        """Agreement: each Lazy node's posted space admits exactly its menu."""
        with open(path, "a") as f:
            for node, space in self._lazy.items():
                got = legal_points(space)
                assert isinstance(node, ResourceNode)
                menu = self.menus[node.name]
                if node.point_menu:
                    menu = tuple(
                        space.division(k, s)
                        for k, s in space.points()
                        if space.options[k].context.is_legal(s)
                    )
                want = menu_points(space, menu)
                f.write(
                    json.dumps(
                        {
                            "node": node.name,
                            "menu": len(want),
                            "points": len(space.points()),
                            "equal": got == want,
                            "missing": len(want - got),
                            "extra": len(got - want),
                            "axes": [str(a) for a in space.axes],
                            "options": [str(o.tiling.label) for o in space.options],
                            "missing_sample": sorted(want - got)[:6],
                            "extra_why": [
                                (k, splits, _why_illegal(space, k, splits))
                                for k, splits in sorted(got - want)[:4]
                            ],
                            "got_sample": sorted(got)[:6],
                        }
                    )
                    + "\n"
                )

    def _enumeration(self, node: ResourceNode) -> _Fixed | _Table:
        if node in self._grid:
            return _Table(self._grid[node])
        return super()._enumeration(node)

    def _candidates(self, item: Item) -> tuple[CoreDivision, ...]:
        if item in self._grid:
            return self._grid[item]
        return super()._candidates(item)

    def _split_vars(self, item: Item, v: NodeVars, m: Model) -> None:
        space = self._lazy.get(item)
        if space is None:
            super()._split_vars(item, v, m)
            return
        option, splits, _, cores = post_division(m, space, item.name)
        lazy = _LazyVars(m, space, option, splits)
        v.lazy = lazy
        grid = self._grid[item]
        sym_core_divs, sym_tile_counts = self._syms(item)
        name = item.name
        v.per_core = [
            ceil_div(item.size, cd.output_partition * cd.tiling.output_tile_count)
            for cd in grid
        ]
        v.cores_used = [cd.cores_used for cd in grid]
        core_cost = [sum(s**2 for s in cd.splits.values()) for cd in grid]
        v.buffer = PrinterView(grid, sym_core_divs, division_symbol(name))
        v.eff_size = m.IntVar(0, max(v.per_core), f"eff_size_{name}")
        v.tie(v.per_core, v.eff_size)
        v.cores = cores
        v.core_cost = m.IntVar(0, max(core_cost), f"core_cost_{name}")
        v.tie(core_cost, v.core_cost)
        for key in sym_core_divs:
            raw = [cd.splits.get(key, 1) for cd in grid]
            v.splits[key] = splits[key] if key in splits else 1
            v.splits_raw[key] = raw
        for key, symbol in sym_tile_counts.items():
            raw = [cd.tiling.count_for(key) for cd in grid]
            var = m.IntVar(min(raw), max(raw), symbol.name)
            v.tie(raw, var)
            v.tile_counts[key] = var
            v.tile_counts_raw[key] = raw

    # ---- views in digit form (2c) ----

    def _relate(self, edge: DependencyEdge, lit, model: Model) -> None:
        re = self.residency.get((edge.src.name, edge.dst.name))
        digits = None
        if re is not None and edge.src in self._lazy and edge.dst in self._lazy:
            digits = self._edge_digits(edge, re)
        if digits is None:
            super()._relate(edge, lit, model)
            return
        assert re is not None
        src_d, dst_d = digits
        self._post_representable(edge.src, src_d, lit, model, write=re.parent_op)
        self._post_representable(edge.dst, dst_d, lit, model)
        src_owner, dst_owner = dict(src_d.owners), dict(dst_d.owners)
        for d in sorted(set(src_owner) | set(dst_owner)):
            a, b = src_owner.get(d), dst_owner.get(d)
            _post_eq(model, self._split(edge.src, a), self._split(edge.dst, b), [lit])
            if a is not None and b is not None:
                gt = self._gt1(edge.src, a, model)
                if not isinstance(gt, int) or gt:
                    _post_eq(
                        model,
                        self._stride(edge.src, src_d, a, model),
                        self._stride(edge.dst, dst_d, b, model),
                        [lit] if isinstance(gt, int) else [lit, gt],
                    )
        src, dst = self.vars[edge.src], self.vars[edge.dst]
        _post_eq(model, src.cores, dst.cores, [lit])
        _post_eq(
            model,
            self._tile_id(edge, re, model, write=True),
            self._tile_id(edge, re, model, write=False),
            [lit],
        )
        if self._check is not None:
            self._check_digits(edge, re, src_d, dst_d)

    def _edge_digits(self, edge: DependencyEdge, re):
        from torch_spyre._inductor.work_division import _prep_for

        key = ("edge", edge)
        if key not in self._digit_cache:
            out = None
            if len(re.read_deps) == 1:
                src = _digits(
                    _prep_for(re.parent_op, re.write_dep, re.buf_name, re.prep_cache),
                    self._domains(edge.src),
                )
                dst = _digits(
                    _prep_for(
                        re.consumer_op, re.read_deps[0], re.buf_name, re.prep_cache
                    ),
                    self._domains(edge.dst),
                )
                if src is not None and dst is not None:
                    out = (src, dst)
            self._digit_cache[key] = out
        return self._digit_cache[key]

    def _domains(self, node: Item) -> dict:
        space = self._lazy[node]
        return {a: space.domain(a) for a in space.axes}

    def _split(self, node: Item, sym) -> Any:
        if sym is None:
            return 1
        lazy = self.vars[node].lazy
        assert lazy is not None
        return lazy.splits.get(sym, 1)

    def _gt1(self, node: Item, sym, model: Model):
        key = ("gt1", node, sym)
        if key not in self._digit_cache:
            split = self._split(node, sym)
            if isinstance(split, int):
                self._digit_cache[key] = int(split > 1)
            else:
                b = model.BoolVar(f"split_gt1_{node.name}_{sym}")
                model.Add(split > 1).OnlyEnforceIf(b)
                model.Add(split == 1).OnlyEnforceIf(b.Not())
                self._digit_cache[key] = b
        return self._digit_cache[key]

    def _stride(self, node: Item, digits: _Digits, sym, model: Model):
        """``sym``'s core stride: the product of the splits before it in core
        order."""
        if sym == digits.contiguous:
            return 1  # first in core order whenever it is split
        before = list(digits.order[: digits.order.index(sym)])
        if digits.contiguous is not None:
            before.append(digits.contiguous)
        factors = [self._split(node, s) for s in before]
        factors = [f for f in factors if not isinstance(f, int) or f != 1]
        key = ("prod", node, tuple(str(s) for s in before))
        if key not in self._digit_cache:
            out: Any = 1
            for i, f in enumerate(factors):
                if isinstance(out, int) and out == 1:
                    out = f
                    continue
                product = model.IntVar(
                    1, self._lazy[node].max_cores, f"stride_{node.name}_{sym}_{i}"
                )
                model.AddMultiplicationEquality(product, [out, f])
                out = product
            self._digit_cache[key] = out
        return self._digit_cache[key]

    def _post_representable(
        self, node: Item, digits: _Digits, lit, model: Model, write=None
    ) -> None:
        """Under ``lit``: the view is representable, and a write view is final
        (no reduction split)."""
        domains = self._domains(node)
        for sym, ok in digits.allowed:
            split = self._split(node, sym)
            if isinstance(split, int):
                if split not in ok:
                    model.AddBoolOr([lit.Not()])
                continue
            for value in domains.get(sym, [1]):
                if value not in ok:
                    model.Add(split != value).OnlyEnforceIf(lit)
        for a, b in digits.interleave:
            lits = [_not(self._gt1(node, a, model)), self._gt1(node, b, model)]
            if any(isinstance(x, int) and x == 1 for x in lits):
                continue
            model.AddBoolOr([x for x in lits if not isinstance(x, int)]).OnlyEnforceIf(
                lit
            )
        if write is not None:
            from torch_spyre._inductor.work_division import _reduction_syms

            space = self._lazy[node]
            for sym in _reduction_syms(write, {a: 2 for a in space.axes}):
                split = self._split(node, sym)
                if not isinstance(split, int):
                    model.Add(split == 1).OnlyEnforceIf(lit)

    def _tile_id(self, edge: DependencyEdge, re, model: Model, write: bool):
        """The tile-view id of the edge's buffer on one end: a function of the
        tiling option alone, so its table is over that var."""
        from torch_spyre._inductor.work_division import _tile_view_for_div

        node = edge.src if write else edge.dst
        key = ("tile", node, edge.src.name) if write else ("tile", edge)
        if key not in self._digit_cache:
            grid = self._grid[node]
            if write:
                ids = [
                    self._intern(
                        edge.src.name,
                        "tile",
                        _tile_view_for_div(
                            re.parent_op, re.write_dep, re.buf_name, cd, re.prep_cache
                        ),
                    )
                    for cd in grid
                ]
            else:
                ids = [
                    self._intern(edge.src.name, "tile", re.consumer_tile_view(cd))
                    for cd in grid
                ]
                ids = [-2 if i == -1 else i for i in ids]
            side = "w" if write else "r"
            self._digit_cache[key] = self._id_var(
                node, ids, f"{side}tile_{edge.src.name}__{node.name}", model
            )
        return self._digit_cache[key]

    def _check_digits(self, edge: DependencyEdge, re, src_d, dst_d) -> None:
        """Agreement: over the two menus, the digit relation admits exactly the
        pairs ``match_pairs`` builds."""
        from torch_spyre._inductor.work_division import (
            _reduction_syms,
            _tile_view_for_div,
        )

        src_menu = list(self.menus[edge.src.name])
        dst_menu = list(self.menus[edge.dst.name])

        def full(cd, digits):
            return {s: cd.splits.get(s, 1) for s in digits.order}

        src_keys = []
        for cd in src_menu:
            partial = any(
                cd.splits.get(s, 1) > 1
                for s in _reduction_syms(re.parent_op, cd.splits)
            )
            k = None if partial else src_d.key(full(cd, src_d))
            t = _tile_view_for_div(
                re.parent_op, re.write_dep, re.buf_name, cd, re.prep_cache
            )
            src_keys.append(
                None
                if k is None or t is None
                else (k, self._intern(edge.src.name, "tile", t))
            )
        dst_keys = []
        for cd in dst_menu:
            k = dst_d.key(full(cd, dst_d))
            t = re.consumer_tile_view(cd)
            dst_keys.append(
                None
                if k is None or t is None
                else (k, self._intern(edge.src.name, "tile", t))
            )
        derived = {
            (i, j)
            for i, a in enumerate(src_keys)
            if a is not None
            for j, b in enumerate(dst_keys)
            if b is not None and a == b
        }
        table = set(re.match_pairs(src_menu, dst_menu))
        record = {
            "edge": repr(edge),
            "digits": True,
            "pairs": len(table),
            "equal": derived == table,
            "missing": len(table - derived),
            "extra": len(derived - table),
        }
        with open(os.environ["LX_RELATION_CHECK"], "a") as f:
            f.write(json.dumps(record) + "\n")

    def cost_bindings(self) -> tuple[dict, dict]:
        sym_map, buffer_map = super().cost_bindings()
        for item in self._lazy:
            v = self.vars[item]
            assert v.buffer is not None
            # No index: a lookup on the division symbol has nothing to read.
            sym_map.pop(v.buffer.sym_division.name, None)
        return sym_map, buffer_map

    def _chosen(self, item: Item, solver) -> CoreDivision:
        lazy = self.vars[item].lazy
        assert lazy is not None
        k = lazy.option if isinstance(lazy.option, int) else solver.Value(lazy.option)
        axes = lazy.space.options[k].context.axes
        return lazy.space.division(k, {a: solver.Value(lazy.splits[a]) for a in axes})

    def chosen_split(self, item: Item, solver) -> WorkSplit:
        if item not in self._lazy:
            return super().chosen_split(item, solver)
        split = WorkSplit.of(self._chosen(item, solver))
        # The caller writes the choice back as an index into the buffer's menu.
        self.solved[item] = self.menu_index(item.name, split)
        return split

    def footprint(self, item: Item, solver) -> int:
        if item not in self._lazy:
            return super().footprint(item, solver)
        cd = self._chosen(item, solver)
        return ceil_div(item.size, cd.output_partition * cd.tiling.output_tile_count)
