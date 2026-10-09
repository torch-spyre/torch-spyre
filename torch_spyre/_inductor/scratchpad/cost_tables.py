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

"""Redundant local cost tables for the joint CP-SAT objective.

A cost's arithmetic helpers can have a weak relaxation even when each is an
exact function of one division and a few residency decisions. Tabulate an
integer projection of their *combined* cost and link it to the same decisions.
The original objective and constraints are retained. Rounding selects the
auxiliary projection's weights; it never changes the optimized cost.
"""

import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import cache
from typing import NamedTuple

from ortools.sat.python import cp_model

_Get = Callable[[int], int]
# These are bounds on extra work, not restrictions on the allocator's choices.
_MAX_RESIDENCY_BITS = 2
_MAX_GROUP_STATES = 2048
_MAX_TOTAL_STATES = 16384
_MAX_LINEAR_ACTIVITY = 1 << 61


class _Expression(NamedTuple):
    offset: int
    vars: tuple[int, ...]
    coeffs: tuple[int, ...]

    @classmethod
    def snapshot(cls, expr):
        return cls(expr.offset, tuple(expr.vars), tuple(expr.coeffs))


class _LinearRow(NamedTuple):
    domain: tuple[int, ...]
    vars: tuple[int, ...]
    coeffs: tuple[int, ...]

    @classmethod
    def snapshot(cls, row):
        return cls(tuple(row.domain), tuple(row.vars), tuple(row.coeffs))


class _CostFunctions:
    """Recover functional definitions, without relying on helper names.

    Division and ordinary-residency variables are supplied explicitly. Other
    decisions, incomplete reifications, and unknown constraints are barriers:
    a term depending on any of them is left in its existing representation.
    Only printer-created equalities are interpreted as affine definitions;
    placement/compatibility equalities must not be oriented as cost functions.
    """

    def __init__(
        self, proto, inputs: set[int], first_variable: int, first_constraint: int
    ):
        self.inputs = set(inputs)
        # Snapshot native protobuf fields once: enumeration repeatedly visits
        # the same helpers, and accessing the Python bindings dominates its cost.
        self.domains = [tuple(var.domain) for var in proto.variables]
        self.nodes: dict[int, tuple[tuple[int, ...], Callable[..., int]]] = {}
        self._support: dict[int, frozenset[int] | None] = {}
        for i, domain in enumerate(self.domains):
            if domain[0] == domain[-1]:
                self.inputs.discard(i)
                self._record(i, (), lambda get, value=domain[0]: value)

        bool_or = {
            (tuple(c.enforcement_literal), tuple(sorted(c.bool_or.literals)))
            for c in proto.constraints
            if c.has_bool_or()
        }
        comparisons = {}
        conditional: dict[int, dict[int, _LinearRow]] = defaultdict(dict)
        for position, c in enumerate(proto.constraints):
            if c.has_element() and not c.enforcement_literal:
                e = c.element
                target = self._target(e.linear_target)
                if target is not None:
                    i, k = target
                    index = _Expression.snapshot(e.linear_index)
                    entries = tuple(_Expression.snapshot(x) for x in e.exprs)
                    offset = e.linear_target.offset
                    self._record(
                        i,
                        self._deps([index, *entries]),
                        lambda get, index=index, entries=entries, offset=offset, k=k: (
                            (self._expr(entries[self._expr(index, get)], get) - offset)
                            // k
                        ),
                    )
            elif not c.enforcement_literal and any(
                (c.has_int_prod(), c.has_int_div(), c.has_lin_max())
            ):
                kind = (
                    "int_prod"
                    if c.has_int_prod()
                    else "int_div"
                    if c.has_int_div()
                    else "lin_max"
                )
                e = getattr(c, kind)
                target = self._target(e.target)
                if target is None:
                    continue
                i, k = target
                exprs = tuple(_Expression.snapshot(x) for x in e.exprs)
                offset = e.target.offset

                def arithmetic(get, exprs=exprs, offset=offset, k=k, kind=kind):
                    values = [self._expr(x, get) for x in exprs]
                    if kind == "int_prod":
                        value = math.prod(values)
                    elif kind == "lin_max":
                        value = max(values)
                    else:
                        a, b = values
                        value = (abs(a) // abs(b)) * (1 if a * b >= 0 else -1)
                    return (value - offset) // k

                self._record(i, self._deps(exprs), arithmetic)
            elif c.has_bool_and() and len(c.enforcement_literal) == 1:
                lit = c.enforcement_literal[0]
                opposite = (
                    (-lit - 1,),
                    tuple(sorted(-x - 1 for x in c.bool_and.literals)),
                )
                if opposite in bool_or:
                    lits = tuple(c.bool_and.literals)
                    self._record(
                        self._index(lit),
                        [self._index(x) for x in lits],
                        lambda get, lits=lits, lit=lit: (
                            int(all(self._literal(x, get) for x in lits))
                            if lit >= 0
                            else int(not all(self._literal(x, get) for x in lits))
                        ),
                    )
            elif c.has_linear() and position >= first_constraint:
                row = _LinearRow.snapshot(c.linear)
                if len(c.enforcement_literal) == 1:
                    key, domain = self._normalized(row)
                    comparisons[c.enforcement_literal[0], key, domain] = row
                if (
                    len(row.domain) != 2
                    or row.domain[0] != row.domain[1]
                    or not row.vars
                ):
                    continue
                target = max(row.vars)
                if target < first_variable or target in self.inputs:
                    continue
                if not c.enforcement_literal:
                    self._record(
                        target,
                        set(row.vars) - {target},
                        lambda get, target=target, row=row: self._solve_linear(
                            target, row, get
                        ),
                    )
                elif len(c.enforcement_literal) == 1:
                    conditional[target][c.enforcement_literal[0]] = row
        for (lit, key, domain), row in comparisons.items():
            if (-lit - 1, key, self._complement(domain)) in comparisons:
                self._record(
                    self._index(lit),
                    row.vars,
                    lambda get, row=row, lit=lit: (
                        int(
                            self._contains(
                                row.domain,
                                sum(k * get(i) for i, k in zip(row.vars, row.coeffs)),
                            )
                        )
                        if lit >= 0
                        else int(
                            not self._contains(
                                row.domain,
                                sum(k * get(i) for i, k in zip(row.vars, row.coeffs)),
                            )
                        )
                    ),
                )
        for target, rows in conditional.items():
            for lit, row in rows.items():
                if lit >= 0 and -lit - 1 in rows:
                    other = rows[-lit - 1]
                    self._record(
                        target,
                        (set(row.vars) | set(other.vars) | {lit}) - {target},
                        lambda get, target=target, lit=lit, row=row, other=other: (
                            self._solve_linear(target, row if get(lit) else other, get)
                        ),
                    )
                    break

    @staticmethod
    def _index(lit: int) -> int:
        return lit if lit >= 0 else -lit - 1

    @classmethod
    def _literal(cls, lit: int, get: _Get) -> bool:
        return bool(get(cls._index(lit))) == (lit >= 0)

    @staticmethod
    def _target(expr):
        if len(expr.vars) == 1 and abs(expr.coeffs[0]) == 1:
            return expr.vars[0], expr.coeffs[0]
        return None

    @staticmethod
    def _expr(expr, get: _Get) -> int:
        return expr.offset + sum(k * get(i) for i, k in zip(expr.vars, expr.coeffs))

    @staticmethod
    def _deps(exprs) -> set[int]:
        return {i for e in exprs for i in e.vars}

    @staticmethod
    def _contains(domain, value: int) -> bool:
        domain = tuple(domain)
        return any(lo <= value <= hi for lo, hi in zip(domain[::2], domain[1::2]))

    @staticmethod
    def _normalized(row):
        terms = sorted(zip(row.vars, row.coeffs))
        domain = tuple(
            -math.inf if x == -(2**63) else math.inf if x == 2**63 - 1 else x
            for x in row.domain
        )
        if terms and terms[0][1] < 0:
            terms = [(i, -k) for i, k in terms]
            domain = tuple(-x for x in reversed(domain))
        return tuple(terms), domain

    @staticmethod
    def _complement(domain):
        result = []
        last = -math.inf
        for lo, hi in zip(domain[::2], domain[1::2]):
            if lo > last:
                result.extend([last, lo - 1])
            last = hi + 1
        if last < math.inf:
            result.extend([last, math.inf])
        return tuple(result)

    def _record(self, index: int, deps: Iterable[int], fn: Callable[..., int]) -> None:
        if index not in self.inputs and index not in self.nodes:
            self.nodes[index] = (tuple(deps), fn)

    @staticmethod
    def _solve_linear(target: int, row, get: _Get) -> int:
        rhs = row.domain[0]
        coefficient = 0
        for i, k in zip(row.vars, row.coeffs):
            if i == target:
                coefficient += k
            else:
                rhs -= k * get(i)
        if not coefficient or rhs % coefficient:
            raise ValueError("nonintegral cost helper")
        return rhs // coefficient

    def support(self, index: int) -> frozenset[int] | None:
        if index in self.inputs:
            return frozenset([index])
        if index not in self._support:
            # A cycle or an unmodelled decision cannot be tabulated.
            self._support[index] = None
            if index in self.nodes:
                deps = [self.support(i) for i in self.nodes[index][0]]
                if all(s is not None for s in deps):
                    self._support[index] = frozenset().union(
                        *(s for s in deps if s is not None)
                    )
        return self._support[index]

    def evaluator(self, inputs: dict[int, int]) -> _Get:
        @cache
        def get(index: int) -> int:
            if index in self.inputs:
                return inputs[index]
            return self.nodes[index][1](get)

        return get


def add_cost_tables(
    model: cp_model.CpModel,
    *,
    divisions: Iterable[cp_model.IntVar],
    residency: Iterable[cp_model.IntVar],
    first_variable: int,
    first_constraint: int,
) -> dict[str, int]:
    """Add exact auxiliary cost projections; keep the objective untouched.

    The projection scale tracks objective magnitude, but does not need to match
    CP-SAT's private objective scaling algorithm for correctness. Both sides of
    every added equality use the *same* integer weights. Unsupported functions
    and groups exceeding the work/integer limits are simply left alone.
    """
    stats = {"groups": 0, "states": 0, "terms": 0}
    if not model.proto.has_floating_point_objective():
        return stats
    source = model.clone()
    proto = source.proto
    objective = proto.floating_point_objective
    owners = {v.index for v in divisions}
    bits = {v.index for v in residency}
    functions = _CostFunctions(proto, owners | bits, first_variable, first_constraint)
    domains = functions.domains
    activity = sum(
        abs(c) * max(abs(domains[i][0]), abs(domains[i][-1]))
        for i, c in zip(objective.vars, objective.coeffs)
    )
    if not math.isfinite(activity) or activity == 0:
        return stats
    # Keep roughly 53 bits of weighted activity, as in integer objective
    # normalization, with at most a double mantissa's worth of scaling.
    exponent = min(52, math.floor(53 - math.log2(activity)))
    scale = math.ldexp(1.0, exponent)
    groups: dict[int, dict[int, float]] = defaultdict(dict)
    for i, c in zip(objective.vars, objective.coeffs):
        support = functions.support(i)
        if support is None:
            continue
        division = support & owners
        if len(division) == 1 and support - division <= bits:
            groups[next(iter(division))][i] = c

    for owner, terms in sorted(groups.items()):
        flags = sorted(
            frozenset().union(*(functions.support(i) or frozenset() for i in terms))
            - {owner}
        )
        domain = domains[owner]
        if (
            len(terms) < 2
            or len(flags) > _MAX_RESIDENCY_BITS
            or len(domain) != 2
            or domain[0] != 0
        ):
            continue
        count = domain[-1] + 1
        states = count * (1 << len(flags))
        if states > _MAX_GROUP_STATES or stats["states"] + states > _MAX_TOTAL_STATES:
            continue
        coefficients = {
            i: round(c * scale) for i, c in terms.items() if round(c * scale)
        }
        if not coefficients:
            continue
        divisor = math.gcd(*coefficients.values())
        coefficients = {i: c // divisor for i, c in coefficients.items()}
        masks = list(itertools.product((0, 1), repeat=len(flags)))
        table = []
        try:
            for mask in masks:
                for d in range(count):
                    get = functions.evaluator({owner: d, **dict(zip(flags, mask))})
                    table.append(sum(c * get(i) for i, c in coefficients.items()))
        except (ValueError, ZeroDivisionError, IndexError, KeyError, OverflowError):
            continue
        lo, hi = min(table), max(table)
        bound = max(abs(lo), abs(hi))
        link_activity = sum(
            abs(c) * max(abs(domains[i][0]), abs(domains[i][-1]))
            for i, c in coefficients.items()
        )
        if (
            bound + max(link_activity, sum(abs(v) for v in table))
            > _MAX_LINEAR_ACTIVITY
        ):
            continue

        target = model.new_int_var(lo, hi, f"local_cost_{owner}")
        model.add(
            target
            == sum(
                c * model.get_int_var_from_proto_index(i)
                for i, c in coefficients.items()
            )
        )
        rows = [
            model.new_bool_var(f"local_cost_{owner}_state_{j}") for j in range(states)
        ]
        model.add_exactly_one(rows)
        division_var = model.get_int_var_from_proto_index(owner)
        for d in range(count):
            selected = model.new_bool_var(f"local_cost_{owner}_division_{d}")
            model.add(division_var == d).only_enforce_if(selected)
            model.add(selected == sum(rows[j * count + d] for j in range(len(masks))))
        for bit_position, bit in enumerate(flags):
            model.add(
                model.get_int_var_from_proto_index(bit)
                == sum(
                    rows[j * count + d]
                    for j, mask in enumerate(masks)
                    if mask[bit_position]
                    for d in range(count)
                )
            )
        model.add(target == sum(row * value for row, value in zip(rows, table)))
        stats["groups"] += 1
        stats["states"] += states
        stats["terms"] += len(coefficients)
    return stats
