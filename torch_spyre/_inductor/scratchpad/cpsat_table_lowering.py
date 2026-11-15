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

"""Table lowering of a sympy cost expression into a CP-SAT objective.

The objective uses three CP-SAT forms only: ``element`` lookups whose entries
are the expression's exact values rounded once onto ``COST_SCALE``, linear sums,
and ``lin_min``/``lin_max``. It has no product of variables and no integer
division.

A node is *indexed* when its value is ``values[index]`` for a short list and an
index CP-SAT can compute. Sums are never indexed in the objective; they stay
linear. Four rules keep every list short:

1. a Min/Max is indexed by the rung of its operands on a ladder without the
   values it can never return (``clamped_ladder``, ``rungs_on``);
2. a product of Min/Max factors over shared inputs is indexed by the factor
   tuples that can occur together (``fold_reachable``);
3. a node whose inputs fit the cap takes the cheaper of one table over those
   inputs and the form built from its parts;
4. a short factor times a node too wide to index is pushed into the node's
   sums and Min/Max operands, a negative factor turning a Min into a Max.

The whole objective is planned before anything is added to the model, so
``lower_cost_with_tables`` returns None with the model untouched when a node
fits none of these, and ``cost_expr_to_cpsat`` then hands the expression to
the printer (``cpsat_expr_printer``).
"""

import logging
import math
import os
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import sympy
from sympy.core.function import AppliedUndef

from torch_spyre._inductor.scratchpad import cpsat_expr_printer
from torch_spyre._inductor.scratchpad.plan_solver import RelayoutCharge

if TYPE_CHECKING:
    from ortools.sat.python import cp_model
else:
    try:
        from ortools.sat.python import cp_model
    except ImportError:  # pragma: no cover - only when ortools is absent
        cp_model = None

__all__ = [
    "COST_SCALE",
    "FoldFactor",
    "FoldStep",
    "clamped_ladder",
    "cost_expr_to_cpsat",
    "fold_reachable",
    "lower_cost_with_tables",
    "rungs_on",
]

logger = logging.getLogger(__name__)

# Objective units per cost unit: every entry is rounded once onto this grid.
COST_SCALE = 1024
# Most entries one table may have; CPSAT_TABLE_CAP overrides it.
TABLE_CAP = int(os.environ.get("CPSAT_TABLE_CAP", "8192"))
# Largest scaled entry, so that sums of thousands of entries stay in int64.
_MAX_ENTRY = 2**50
_MAX_BOUND = 2**62
# The linear form of a node replaces its table only when the table is this many
# times larger: a rank's small domain searches faster than a lin_max over wide
# values, so a near tie keeps the table (2.5x slower solve on a GQA objective).
_LINEAR_MARGIN = 2
# Values equal to this many significant digits are one value.
_SIGNIFICANT_DIGITS = 12


def _same_value_key(value: float) -> float:
    value = float(value)
    return 0.0 if value == 0 else float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def clamped_ladder(
    operands: Sequence[Sequence[float]], is_min: bool
) -> tuple[list[float], float]:
    """The values a Min (``is_min``) or Max of ``operands`` can return, ascending,
    and the bound past which an operand value is clamped.

    A Min never exceeds the smallest operand maximum, so values above it are
    clamped to it; a Max mirrors this with the largest operand minimum."""
    if is_min:
        bound = min(max(o) for o in operands)
        clamp = lambda v: min(v, bound)  # noqa: E731
    else:
        bound = max(min(o) for o in operands)
        clamp = lambda v: max(v, bound)  # noqa: E731
    keys = {_same_value_key(clamp(v)): clamp(v) for o in operands for v in o}
    return [keys[k] for k in sorted(keys)], bound


def rungs_on(
    ladder: Sequence[float], bound: float, values: Sequence[float], is_min: bool
) -> Optional[list[int]]:
    """Each of ``values``, clamped at ``bound``, as its position on ``ladder``;
    None when a value is not on it. Rungs preserve order, so the Min (Max) of
    rungs is the rung of the Min (Max)."""
    clamped = np.minimum(values, bound) if is_min else np.maximum(values, bound)
    lad = np.asarray(ladder, dtype=float)
    pos = np.clip(np.searchsorted(lad, clamped), 0, len(lad) - 1)
    lower = np.clip(pos - 1, 0, len(lad) - 1)
    pos = np.where(
        np.abs(lad[lower] - clamped) < np.abs(lad[pos] - clamped), lower, pos
    )
    scale = np.maximum(np.abs(clamped), 1.0)
    if np.any(np.abs(lad[pos] - clamped) > 1e-9 * scale):
        return None
    return [int(p) for p in pos]


@dataclass(frozen=True)
class FoldFactor:
    """A Min (``is_min``) or Max whose operands are each a tuple of inputs and
    the operand's rung at every joint assignment of them, row-major (no inputs:
    a constant operand, one rung). ``rung_values`` is the factor's value at
    each rung."""

    is_min: bool
    operands: tuple[tuple[tuple[Hashable, ...], tuple[int, ...]], ...]
    rung_values: tuple[float, ...]


@dataclass(frozen=True)
class FoldStep:
    """One step of a fold: the inputs it reads, and the next state for each
    ``previous state * joint size + joint assignment of the inputs``."""

    inputs: tuple[Hashable, ...]
    next_state: tuple[int, ...]


def fold_reachable(
    factors: Sequence[FoldFactor], sizes: dict, cap: int
) -> Optional[tuple[list[FoldStep], list[tuple[int, ...]]]]:
    """The tuples of factor rungs that can occur together, and the steps that
    compute which one occurs.

    Operands sharing an input join one group, and a step reads one group: its
    state is the tuple of each factor's Min/Max over the groups read so far,
    so only tuples some assignment reaches are ever numbered. Returns the steps
    and the final states, or None when a step's table or the number of states
    passes ``cap``."""
    pick = [min if f.is_min else max for f in factors]
    start: list = [None] * len(factors)
    groups: list[tuple[frozenset, list]] = []
    for f, factor in enumerate(factors):
        for inputs, rungs in factor.operands:
            if not inputs:
                start[f] = rungs[0] if start[f] is None else pick[f](start[f], rungs[0])
                continue
            hit = [g for g in groups if g[0] & set(inputs)]
            merged = frozenset(inputs).union(*(g[0] for g in hit))
            members = [(f, inputs, rungs)] + [m for g in hit for m in g[1]]
            groups = [g for g in groups if g not in hit] + [(merged, members)]

    def combine(a, b, f):
        return b if a is None else a if b is None else pick[f](a, b)

    states = [tuple(start)]
    steps = []
    for inputs_set, members in groups:
        inputs = tuple(sorted(inputs_set))
        shape = [sizes[k] for k in inputs]
        joint = math.prod(shape)
        if len(states) * joint > cap:
            return None
        grid = np.indices(shape).reshape(len(inputs), -1)
        contribution = [[None] * len(factors) for _ in range(joint)]
        for f, operand_inputs, rungs in members:
            row = np.zeros(joint, dtype=np.int64)
            for k in operand_inputs:
                row = row * sizes[k] + grid[inputs.index(k)]
            picked = np.asarray(rungs)[row]
            for j in range(joint):
                contribution[j][f] = combine(contribution[j][f], int(picked[j]), f)
        numbered: dict = {}
        next_state = []
        for state in states:
            for j in range(joint):
                reached = tuple(
                    combine(state[f], contribution[j][f], f)
                    for f in range(len(factors))
                )
                next_state.append(numbered.setdefault(reached, len(numbered)))
        if len(numbered) > cap:
            return None
        steps.append(FoldStep(inputs, tuple(next_state)))
        states = list(numbered)
    return steps, states


class _Index:
    """A CP-SAT index, built on first use. ``entries`` counts the table entries
    behind it (rule 3 compares them); ``size`` is its number of values."""

    __slots__ = ("_build", "_expr", "entries", "size")

    def __init__(self, build: Callable, entries: int, size: int) -> None:
        self._build = build
        self._expr = None
        self.entries = entries
        self.size = size

    def expr(self):
        if self._expr is None:
            self._expr = self._build()
        return self._expr


@dataclass(eq=False)
class _Indexed:
    """``values[index]``; a constant has no index and one value."""

    values: np.ndarray
    index: Optional[_Index]
    ranked: bool = False
    fold: Optional[FoldFactor] = None

    @property
    def entries(self) -> int:
        """Entries to compute this value: its index's tables and its own."""
        return 0 if self.index is None else self.index.entries + len(self.values)


@dataclass(eq=False)
class _Linear:
    """``constant + sum(values[index]) + sum(Min/Max of _Linear operands)``, with
    terms over one index summed into one table."""

    constant: float = 0.0
    tables: dict = field(default_factory=dict)
    extrema: list = field(default_factory=list)

    def add(self, term: _Indexed, coeff: float) -> None:
        if term.index is None:
            self.constant += coeff * float(term.values[0])
            return
        values = coeff * term.values
        acc = self.tables.get(term.index)
        self.tables[term.index] = values if acc is None else acc + values

    def extend(self, other: "_Linear") -> None:
        self.constant += other.constant
        for index, values in other.tables.items():
            acc = self.tables.get(index)
            self.tables[index] = values if acc is None else acc + values
        self.extrema.extend(other.extrema)


def _relayout_charge(lit, division, *prices):
    table = np.asarray([*prices, 0.0], dtype=float)
    position = np.minimum(np.asarray(division).astype(np.int64), len(prices))
    return lit * table[position]


_NUMPY_FUNCTIONS = {
    "KroneckerDelta": lambda a, b: np.equal(a, b).astype(float),
    "RelayoutCharge_": _relayout_charge,
}


class _TableLowering:
    def __init__(self, model, sym_map: dict, buffer_map: dict, cap: int) -> None:
        self._model = model
        self._cap = cap
        self._inputs: dict = {}  # key -> _Index of the input's choice
        self._symbol_input: dict = {}  # symbol name -> key
        self._symbol_values: dict = {}  # symbol name -> value at each choice
        self._constants: dict = {}
        self._division_of: set = set()
        self.failure: Optional[str] = None
        self._indexed: dict = {}
        self._evaluated: dict = {}
        self._compiled: dict = {}
        self._ranks: dict = {}
        self._pairs: dict = {}
        self._joint: dict = {}
        self._linear_cost: dict = {}
        self._count = 0
        self._map_inputs(sym_map, buffer_map)

    # ---- inputs ---------------------------------------------------------------

    def _map_inputs(self, sym_map, buffer_map) -> None:
        """A division-backed symbol takes its value at each of its buffer's
        divisions; any other CP-SAT variable takes its domain's values."""
        for name, (wrapper, raw) in buffer_map.items():
            if wrapper is None:
                continue
            division = wrapper.division
            key = division.index
            if key in self._inputs and self._inputs[key].size != len(raw):
                self.failure = f"{name}: menus of two lengths on one division"
                return
            if key not in self._inputs:
                self._inputs[key] = _Index(lambda d=division: d, 0, len(raw))
            self._symbol_input[name] = key
            self._symbol_values[name] = np.asarray(raw, dtype=float)
        for name, value in sym_map.items():
            if name.startswith("_division_of_"):
                self._division_of.add(name[len("_division_of_") :])
                continue
            if name in self._symbol_input:
                continue
            if isinstance(value, (bool, int, float)):
                self._constants[name] = value
            elif isinstance(value, cp_model.IntVar):
                self._map_variable(name, value)

    def _map_variable(self, name, var) -> None:
        key = var.index
        if key in self._inputs:
            values = np.arange(self._inputs[key].size, dtype=float)
        else:
            bounds = list(self._model.proto.variables[key].domain)
            values = np.concatenate(
                [np.arange(lo, hi + 1) for lo, hi in zip(bounds[::2], bounds[1::2])]
            ).astype(float)
            if len(values) > self._cap:
                return
            lo = int(values[0])
            if len(values) == int(values[-1]) - lo + 1:
                build = (lambda v=var: v) if lo == 0 else (lambda v=var, lo=lo: v - lo)
            else:
                build = lambda v=var, vals=values: self._position_of(v, vals)  # noqa: E731
            self._inputs[key] = _Index(build, 0, len(values))
        self._symbol_input[name] = key
        self._symbol_values[name] = values

    def _position_of(self, var, values):
        position = self._model.new_int_var(0, len(values) - 1, f"pos_{var.name}")
        self._model.add_element(position, [int(v) for v in values], var)
        return position

    def _keys_of(self, node) -> Optional[tuple]:
        keys = set()
        for s in node.free_symbols:
            key = self._symbol_input.get(s.name)
            if key is None:
                self.failure = f"no input for symbol {s.name}"
                return None
            keys.add(key)
        return tuple(sorted(keys))

    def _joint_size(self, keys) -> int:
        return math.prod(self._inputs[k].size for k in keys)

    def _joint_index(self, keys) -> _Index:
        if keys not in self._joint:
            if len(keys) == 1:
                self._joint[keys] = self._inputs[keys[0]]
            else:
                size = self._joint_size(keys)

                def build(keys=keys, size=size):
                    strides = np.cumprod(
                        [1] + [self._inputs[k].size for k in keys[:0:-1]]
                    )
                    combined = sum(
                        int(s) * self._inputs[k].expr()
                        for s, k in zip(strides[::-1], keys)
                    )
                    return self._aux(combined, size, "joint")

                entries = sum(self._inputs[k].entries for k in keys)
                self._joint[keys] = _Index(build, entries, size)
        return self._joint[keys]

    # ---- building -------------------------------------------------------------

    def _aux(self, linear, size: int, name: str):
        self._count += 1
        var = self._model.new_int_var(0, size - 1, f"{name}_{self._count}")
        self._model.add(var == linear)
        return var

    def _element(self, index, table, name: str):
        table = [int(t) for t in table]
        if min(table) == max(table):
            return table[0]
        self._count += 1
        domain = cp_model.Domain.from_values(sorted(set(table)))
        target = self._model.new_int_var_from_domain(domain, f"{name}_{self._count}")
        self._model.add_element(index, table, target)
        return target

    # ---- evaluation -----------------------------------------------------------

    def _evaluate(self, node, keys) -> Optional[np.ndarray]:
        """``node`` at every joint assignment of ``keys``, row-major; None when a
        value is not finite."""
        if (node, keys) in self._evaluated:
            return self._evaluated[(node, keys)]
        symbols = sorted(node.free_symbols, key=lambda s: s.name)
        if (node, tuple(symbols)) not in self._compiled:
            self._compiled[(node, tuple(symbols))] = self._compile(node, symbols)
        if self._compiled[(node, tuple(symbols))] is None:
            self.failure = f"a function with no implementation in {str(node)[:120]}"
            return None
        shape = [self._inputs[k].size for k in keys]
        grid = np.indices(shape).reshape(len(keys), -1)
        args = [
            self._symbol_values[s.name][grid[keys.index(self._symbol_input[s.name])]]
            for s in symbols
        ]
        with np.errstate(all="ignore"):
            out = np.asarray(self._compiled[(node, tuple(symbols))](*args), dtype=float)
        out = np.array(np.broadcast_to(out, (grid.shape[1],)))
        result = out if np.all(np.isfinite(out)) else None
        if result is None:
            self.failure = f"a value that is not finite in {str(node)[:120]}"
        self._evaluated[(node, keys)] = result
        return result

    @staticmethod
    def _compile(node, symbols):
        """``node`` as a numpy function of ``symbols``; None when it calls an
        undefined function, which has no values to table."""
        namespace = dict(_NUMPY_FUNCTIONS)
        replaced = {}
        for call in node.atoms(AppliedUndef):
            imp = getattr(call, "_imp_", None)
            if imp is None:
                return None
            name = f"imp_{len(namespace)}"
            namespace[name] = np.vectorize(
                lambda *a, imp=imp: float(imp(*(int(x) for x in a))), otypes=[float]
            )
            replaced[call] = sympy.Function(name)(*call.args)
        node = node.xreplace(replaced) if replaced else node
        node = node.replace(
            lambda e: isinstance(e, RelayoutCharge),
            lambda e: sympy.Function("RelayoutCharge_")(*e.args),
        )
        return sympy.lambdify(symbols, node, modules=[namespace, "numpy"])

    # ---- indexed values (rules 1-3) ---------------------------------------------

    def indexed(self, node) -> Optional[_Indexed]:
        """``node`` as an indexed value, or None when it is too wide."""
        if node not in self._indexed:
            self._indexed[node] = self._index_node(node)
        return self._indexed[node]

    def _index_node(self, node) -> Optional[_Indexed]:
        if not node.free_symbols:
            value = float(node)
            if not math.isfinite(value):
                self.failure = f"constant {node} is not finite"
                return None
            return _Indexed(np.array([value]), None)
        keys = self._keys_of(node)
        if keys is None:
            return None
        if node.is_Symbol:
            return _Indexed(self._symbol_values[node.name], self._inputs[keys[0]])
        tabled = None
        if self._joint_size(keys) <= self._cap:
            values = self._evaluate(node, keys)
            if values is not None:
                tabled = _Indexed(values, self._joint_index(keys))
        built = self._from_parts(node)
        if built is None:
            return tabled
        if tabled is not None and tabled.entries <= built.entries:
            tabled.fold = built.fold
            return tabled
        return built

    def _from_parts(self, node) -> Optional[_Indexed]:
        if isinstance(node, (sympy.Min, sympy.Max)):
            return self._extremum(node)
        if node.is_Mul:
            factors = [self.indexed(a) for a in node.args]
            return None if None in factors else self._product(factors)
        if node.is_Add:
            return self._same_index_sum(node)
        variable = [a for a in node.args if a.free_symbols]
        if len(variable) == 1 and (node.is_Pow or isinstance(node, sympy.Function)):
            if not isinstance(node, sympy.Piecewise):
                return self._function_of(node, variable[0])
        return None

    def _same_index_sum(self, node) -> Optional[_Indexed]:
        terms = [self.indexed(a) for a in node.args]
        if None in terms:
            return None
        indexes = {t.index for t in terms if t.index is not None}
        if len(indexes) != 1:
            return None
        (index,) = indexes
        size = index.size
        values = sum(np.broadcast_to(t.values, (size,)) for t in terms)
        return _Indexed(np.asarray(values, dtype=float), index)

    def _function_of(self, node, arg) -> Optional[_Indexed]:
        """A function of a short list is its values at the list's values, over
        the same index."""
        inner = self.indexed(arg)
        if inner is None or inner.index is None:
            return None
        dummy = sympy.Dummy("x")
        f = self._compile(node.xreplace({arg: dummy}), [dummy])
        if f is None:
            return None
        with np.errstate(all="ignore"):
            values = np.asarray(f(inner.values), dtype=float)
        values = np.array(np.broadcast_to(values, inner.values.shape))
        if not np.all(np.isfinite(values)):
            self.failure = f"a value that is not finite in {str(node)[:120]}"
            return None
        fold = inner.fold
        if fold is not None:
            with np.errstate(all="ignore"):
                rung_values = np.asarray(f(np.asarray(fold.rung_values)), dtype=float)
            rung_values = np.broadcast_to(rung_values, (len(fold.rung_values),))
            fold = (
                FoldFactor(fold.is_min, fold.operands, tuple(map(float, rung_values)))
                if np.all(np.isfinite(rung_values))
                else None
            )
        return _Indexed(values, inner.index, ranked=False, fold=fold)

    def _extremum(self, node) -> Optional[_Indexed]:
        """Rule 1: a Min/Max of indexed operands is the rung of their clamped
        values on its ladder."""
        operands = [self.indexed(a) for a in node.args]
        if None in operands:
            return None
        is_min = isinstance(node, sympy.Min)
        ladder, bound = clamped_ladder([o.values for o in operands], is_min)
        if len(ladder) > self._cap:
            self.failure = f"a Min/Max ladder of {len(ladder)} values"
            return None
        rungs = [rungs_on(ladder, bound, o.values, is_min) for o in operands]
        if None in rungs:
            return None

        def build():
            exprs = [
                r[0] if o.index is None else self._element(o.index.expr(), r, "rung")
                for o, r in zip(operands, rungs)
            ]
            self._count += 1
            target = self._model.new_int_var(0, len(ladder) - 1, f"rank_{self._count}")
            if is_min:
                self._model.add_min_equality(target, exprs)
            else:
                self._model.add_max_equality(target, exprs)
            return target

        # An operand clamped to one rung needs no table.
        entries = sum(o.entries for o, r in zip(operands, rungs) if min(r) != max(r))
        index = _Index(build, entries, len(ladder))
        fold = self._fold_factor(node, is_min, ladder, bound)
        return _Indexed(np.asarray(ladder), index, ranked=True, fold=fold)

    def _fold_factor(self, node, is_min, ladder, bound) -> Optional[FoldFactor]:
        operands = []
        for arg in node.args:
            if not arg.free_symbols:
                rungs = rungs_on(ladder, bound, [float(arg)], is_min)
                keys = ()
            else:
                keys = self._keys_of(arg)
                if keys is None or self._joint_size(keys) > self._cap:
                    return None
                values = self._evaluate(arg, keys)
                rungs = (
                    None if values is None else rungs_on(ladder, bound, values, is_min)
                )
            if rungs is None:
                return None
            operands.append((keys, tuple(rungs)))
        return FoldFactor(is_min, tuple(operands), tuple(map(float, ladder)))

    def _rank(self, ix: _Indexed) -> _Indexed:
        """``ix`` over its distinct values, so equal values share one index."""
        if ix.ranked or ix.index is None:
            return ix
        cache_key = (ix.index, tuple(ix.values))
        if cache_key not in self._ranks:
            keys = [_same_value_key(v) for v in ix.values]
            distinct = sorted(set(keys))
            if len(distinct) == len(keys):
                self._ranks[cache_key] = ix
            else:
                position = {k: i for i, k in enumerate(distinct)}
                table = [position[k] for k in keys]
                index = _Index(
                    lambda i=ix.index, t=table: self._element(i.expr(), t, "rank_of"),
                    ix.entries,
                    len(distinct),
                )
                self._ranks[cache_key] = _Indexed(
                    np.asarray(distinct), index, ranked=True, fold=ix.fold
                )
        return self._ranks[cache_key]

    def _pair(self, a: _Indexed, b: _Indexed) -> Optional[_Indexed]:
        """``a * b`` over the pair of their ranks; None past the cap."""
        ra, rb = self._rank(a), self._rank(b)
        if ra.index is None or rb.index is None:
            const, other = (ra, rb) if ra.index is None else (rb, ra)
            scale = float(const.values[0])
            return _Indexed(
                scale * other.values, other.index, other.ranked and scale != 0
            )
        size = len(ra.values) * len(rb.values)
        if size > self._cap:
            return None
        cache_key = (ra.index, rb.index)
        if cache_key not in self._pairs:
            n = len(rb.values)
            self._pairs[cache_key] = _Index(
                lambda: self._aux(ra.index.expr() * n + rb.index.expr(), size, "pair"),
                ra.entries + rb.entries,
                size,
            )
        return _Indexed(
            np.outer(ra.values, rb.values).reshape(-1), self._pairs[cache_key]
        )

    def _product(self, factors: list) -> Optional[_Indexed]:
        acc = None
        for ix in sorted(self._fold_too_long(factors), key=lambda i: len(i.values)):
            acc = ix if acc is None else self._pair(acc, ix)
            if acc is None:
                return None
        return acc

    def _fold_too_long(self, factors: list) -> list:
        """Rule 2, where it is needed: the Min/Max factors folded into one when
        crossing their lists would pass the cap."""
        foldable = [f for f in factors if f.fold is not None]
        if len(foldable) < 2:
            return factors
        if math.prod(len(self._rank(f).values) for f in foldable) <= self._cap:
            return factors
        folded = self._fold(foldable)
        if folded is None:
            return factors
        return [f for f in factors if f.fold is None] + [folded]

    def _fold(self, factors: list) -> Optional[_Indexed]:
        sizes = {k: v.size for k, v in self._inputs.items()}
        found = fold_reachable([f.fold for f in factors], sizes, self._cap)
        if found is None:
            return None
        steps, states = found
        values = np.array(
            [
                math.prod(f.fold.rung_values[s[i]] for i, f in enumerate(factors))
                for s in states
            ]
        )
        if len(states) == 1:
            return _Indexed(values, None)

        def build():
            state, n_states = 0, 1
            for step in steps:
                group = self._joint_index(step.inputs)
                if n_states == 1:
                    position = group.expr()
                else:
                    position = self._aux(
                        state * group.size + group.expr(), n_states * group.size, "fold"
                    )
                state = self._element(position, step.next_state, "fold_state")
                n_states = max(step.next_state) + 1
            return state

        entries = sum(len(s.next_state) for s in steps)
        return _Indexed(values, _Index(build, entries, len(states)))

    # ---- the objective (rule 4) -------------------------------------------------

    def emit(self, node, factor: Optional[_Indexed] = None, coeff: float = 1.0):
        """``coeff * factor * node`` as a _Linear, or None. A sum is never one
        table in the objective, and a factor goes into it term by term unless
        one table over the whole sum and the factor is smaller. A node is one
        table only when that is smaller than its linear form (rule 3)."""
        ix = None if node.is_Add and factor is None else self.indexed(node)
        if ix is not None and factor is None and self._prefers_linear(node, ix):
            ix = None
        if ix is not None:
            term = ix if factor is None else self._pair(ix, factor)
            if term is not None and not (
                node.is_Add and self._distributed_size(node, factor) < len(term.values)
            ):
                out = _Linear()
                out.add(term, coeff)
                return out
        if node.is_Add:
            return self._emit_sum(node, factor, coeff)
        if node.is_Mul:
            return self._emit_product(node, factor, coeff)
        if isinstance(node, (sympy.Min, sympy.Max)):
            return self._emit_extremum(node, factor, coeff)
        if self.failure is None:
            self.failure = f"cannot lower {type(node).__name__}: {str(node)[:120]}"
        return None

    def _linear_entries(self, node) -> float:
        """Entries ``node`` needs in the objective, a sum and a Min/Max kept
        linear where that is smaller than their table; inf when it cannot be
        lowered."""
        if node in self._linear_cost:
            return self._linear_cost[node]
        if not node.free_symbols:
            cost = 0.0
        elif node.is_Add:
            cost = sum(self._linear_entries(a) for a in node.args)
        else:
            ix = self.indexed(node)
            tabled = math.inf if ix is None else self._table_entries(ix)
            if isinstance(node, (sympy.Min, sympy.Max)):
                linear = sum(self._linear_entries(a) for a in node.args)
            elif node.is_Mul and sum(not a.is_Number for a in node.args) == 1:
                (rest,) = [a for a in node.args if not a.is_Number]
                linear = self._linear_entries(rest)
            else:
                linear = math.inf
            cost = linear if _LINEAR_MARGIN * linear < tabled else tabled
        self._linear_cost[node] = cost
        return cost

    def _prefers_linear(self, node, ix: _Indexed) -> bool:
        return _LINEAR_MARGIN * self._linear_entries(node) < self._table_entries(ix)

    @staticmethod
    def _table_entries(ix: _Indexed) -> int:
        """Entries ``ix`` costs as an objective term: none for its own list
        when that list is affine in its index (``_build`` makes it linear)."""
        if ix.index is None:
            return 0
        steps = np.diff(ix.values)
        affine = len(steps) == 0 or np.allclose(steps, steps[0], rtol=0, atol=1e-9)
        return ix.index.entries + (0 if affine else len(ix.values))

    def _distributed_size(self, node, factor) -> int:
        """Entries ``factor`` pushed into the terms of the sum ``node`` needs."""
        per_index: dict = {}
        for term in node.args:
            ix = self.indexed(term)
            if ix is not None and ix.index is not None:
                acc = per_index.get(ix.index)
                per_index[ix.index] = ix.values if acc is None else acc + ix.values
        width = len(self._rank(factor).values)
        return sum(
            len(self._rank(_Indexed(values, index)).values) * width
            for index, values in per_index.items()
        )

    def _emit_sum(self, node, factor, coeff):
        """A sum stays linear. Terms over one index are summed into one list
        before the factor is pushed in."""
        out = _Linear()
        same_index: dict = {}
        for term in node.args:
            ix = self.indexed(term)
            if ix is not None and factor is None and self._prefers_linear(term, ix):
                ix = None
            if ix is None or ix.index is None:
                part = self.emit(term, factor, coeff)
                if part is None:
                    return None
                out.extend(part)
            else:
                acc = same_index.get(ix.index)
                same_index[ix.index] = ix.values if acc is None else acc + ix.values
        for index, values in same_index.items():
            summed = _Indexed(values, index)
            term = summed if factor is None else self._pair(summed, factor)
            if term is None:
                self.failure = "a factor too long to push into a sum's terms"
                return None
            out.add(term, coeff)
        return out

    def _emit_product(self, node, factor, coeff):
        """Indexed factors multiply into one; a single factor too wide to index
        takes that product pushed into it."""
        variable = [a for a in node.args if not a.is_Number]
        if factor is None and len(variable) == 1:
            numbers = math.prod(float(a) for a in node.args if a.is_Number)
            return self.emit(variable[0], None, coeff * numbers)
        wide = []
        items = []
        for a in node.args:
            if a.is_Number:
                coeff *= float(a)
                continue
            ix = self.indexed(a)
            if ix is None:
                wide.append(a)
            else:
                items.append((a, ix))
        folded = self._fold_too_long([ix for _, ix in items])
        kept = {id(ix) for ix in folded}
        nodes = {id(ix): a for a, ix in items if id(ix) in kept}
        acc = factor
        for ix in sorted(folded, key=lambda i: len(self._rank(i).values)):
            joined = ix if acc is None else self._pair(acc, ix)
            if joined is not None:
                acc = joined
            elif id(ix) in nodes:
                wide.append(nodes[id(ix)])
            else:
                self.failure = "a folded product too long to join its factors"
                return None
        if not wide:
            out = _Linear()
            out.add(acc, coeff)
            return out
        if len(wide) > 1:
            self.failure = f"a product of {len(wide)} wide factors: {str(node)[:120]}"
            return None
        return self.emit(wide[0], acc, coeff)

    def _emit_extremum(self, node, factor, coeff):
        """A Min/Max too wide to index stays a lin_min/lin_max; a factor of one
        sign moves into its operands."""
        sign = 1.0
        if factor is not None:
            if np.all(factor.values >= 0):
                sign = 1.0
            elif np.all(factor.values <= 0):
                sign = -1.0
            else:
                self.failure = "a factor of both signs times a wide Min/Max"
                return None
        operands = [self.emit(a, factor, coeff) for a in node.args]
        if None in operands:
            return None
        is_min = isinstance(node, sympy.Min) == (sign * coeff >= 0)
        out = _Linear()
        out.extrema.append((is_min, operands))
        return out

    # ---- building the objective ---------------------------------------------------

    @staticmethod
    def _scaled(values) -> list[int]:
        return [int(v) for v in np.rint(COST_SCALE * np.asarray(values, dtype=float))]

    def _bounds(self, lin: _Linear) -> Optional[tuple[int, int]]:
        lo = hi = round(COST_SCALE * lin.constant)
        for values in lin.tables.values():
            table = self._scaled(values)
            if max(abs(min(table)), abs(max(table))) > _MAX_ENTRY:
                self.failure = "a table entry past the scaled bound"
                return None
            lo, hi = lo + min(table), hi + max(table)
        for is_min, operands in lin.extrema:
            bounds = [self._bounds(o) for o in operands]
            if None in bounds:
                return None
            pick = min if is_min else max
            lo += pick(b[0] for b in bounds)
            hi += pick(b[1] for b in bounds)
        if max(abs(lo), abs(hi)) > _MAX_BOUND:
            self.failure = "an objective bound past int64"
            return None
        return lo, hi

    def _build(self, lin: _Linear):
        total = round(COST_SCALE * lin.constant)
        for index, values in lin.tables.items():
            table = self._scaled(values)
            step = table[1] - table[0] if len(table) > 1 else 0
            if all(t == table[0] + i * step for i, t in enumerate(table)):
                total += table[0] + step * index.expr() if step else table[0]
            else:
                total += self._element(index.expr(), table, "cost")
        for is_min, operands in lin.extrema:
            # Each operand behind a variable of its own: a long sum of literals
            # inside lin_max makes CP-SAT's presolve run out of memory.
            exprs = [self._operand(o) for o in operands]
            lo, hi = self._bounds(_Linear(extrema=[(is_min, operands)]))
            self._count += 1
            target = self._model.new_int_var(lo, hi, f"extremum_{self._count}")
            if is_min:
                self._model.add_min_equality(target, exprs)
            else:
                self._model.add_max_equality(target, exprs)
            total += target
        return total

    def _operand(self, lin: _Linear):
        expr = self._build(lin)
        if isinstance(expr, (int, cp_model.IntVar)):
            return expr
        lo, hi = self._bounds(lin)
        self._count += 1
        var = self._model.new_int_var(lo, hi, f"operand_{self._count}")
        self._model.add(var == expr)
        return var

    def lower(self, cost_expr):
        if self.failure is not None:
            return None
        expr = sympy.sympify(cost_expr)
        expr = expr.xreplace(
            {
                s: sympy.sympify(self._constants[s.name])
                for s in expr.free_symbols
                if s.name in self._constants
            }
        )
        expr = self._drop_unpriced(expr)
        lin = self.emit(expr)
        if lin is None or self._bounds(lin) is None:
            return None
        return self._build(lin)

    def _drop_unpriced(self, expr):
        """Price 0 for a relayout of a buffer this solve does not divide, and for
        a functor of no input, as the printer does."""

        def unpriced(e):
            if isinstance(e, RelayoutCharge):
                division = e.args[1]
                return division.is_Symbol and division.name not in self._division_of
            if isinstance(e, AppliedUndef):
                return any(s.name not in self._symbol_input for s in e.free_symbols)
            return False

        return expr.replace(unpriced, lambda e: sympy.Integer(0))


def lower_cost_with_tables(
    model: "cp_model.CpModel",
    sym_map: dict,
    buffer_map: dict,
    cost_expr: sympy.Expr,
    cap: int,
):
    """``COST_SCALE * cost_expr`` as an integer CP-SAT expression over ``model``
    built from tables of at most ``cap`` entries, or None, with ``model``
    unchanged, when some node cannot be lowered that way.

    ``sym_map`` and ``buffer_map`` are the printer's (``cost_expr_to_cpsat``):
    a symbol in ``buffer_map`` takes its listed value at each division of its
    buffer; any other CP-SAT variable in ``sym_map`` its domain's values."""
    lowering = _TableLowering(model, sym_map, buffer_map, cap)
    lowered = lowering.lower(cost_expr)
    if lowered is None:
        logger.debug("[CP-SAT table lowering] not lowered: %s", lowering.failure)
    return lowered


def cost_expr_to_cpsat(
    model: "cp_model.CpModel", sym_map: dict, buffer_map: dict, cost_expr: sympy.Expr
):
    """``cost_expr`` as a CP-SAT objective: the table lowering
    (``lower_cost_with_tables``), or the printer's when a node cannot be
    tabled."""
    lowered = lower_cost_with_tables(model, sym_map, buffer_map, cost_expr, TABLE_CAP)
    if lowered is None:
        return cpsat_expr_printer.cost_expr_to_cpsat(
            model, sym_map, buffer_map, cost_expr
        )
    return lowered
