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

"""The Lazy work-split encoding's tables: a quantity is tabulated over only the
axes it varies with, and reads its value at every point. Device-free."""

from __future__ import annotations

import itertools
import math

import pytest

pytest.importorskip("ortools")

import sympy  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402

from torch_spyre._inductor.scratchpad.backend import CpSat  # noqa: E402
from torch_spyre._inductor.scratchpad.lazy_strategy import _LazyVars  # noqa: E402

X, Y, Z = sympy.symbols("x y z")


class _Space:
    """Two options over three axes; the second option has no ``z`` axis."""

    axes = [X, Y, Z]

    def points(self):
        out = []
        for k, axes in enumerate(([X, Y, Z], [X, Y])):
            for combo in itertools.product([1, 2, 4], repeat=len(axes)):
                if math.prod(combo) <= 8:
                    out.append((k, dict(zip(axes, combo))))
        return out


def _tie_and_enumerate(value_of):
    backend = CpSat()
    option = backend.IntVar(0, 1, "option")
    splits = {a: backend.IntVar(1, 4, str(a)) for a in _Space.axes}
    lazy = _LazyVars(backend, _Space(), option, splits)
    values = [value_of(k, s) for k, s in _Space().points()]
    var = backend.IntVar(min(values), max(values), "value")
    lazy.tie(values, var)
    table = backend.model.proto.constraints[-1]
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    seen = {}

    class Collect(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            key = (self.Value(option), *(self.Value(splits[a]) for a in _Space.axes))
            seen.setdefault(key, set()).add(self.Value(var))

    solver.Solve(backend.model, Collect())
    return lazy, table, seen


def test_tie_reads_the_value_at_every_point():
    def value_of(k, s):
        return 10 * k + s[X] * s[Y]

    lazy, _, seen = _tie_and_enumerate(value_of)
    for k, s in _Space().points():
        key = (k, *(s.get(a, 1) for a in _Space.axes))
        assert seen[key] == {value_of(k, s)}


def test_tie_drops_the_axes_a_value_does_not_vary_with():
    _, table, _ = _tie_and_enumerate(lambda k, s: s[X] * 3)
    # One table over (x, value): the option, y and z are dropped.
    assert len(table.table.exprs) == 2
