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

"""CPU contract tests for ``per_trip_index`` (loop-trip base-coordinate pin).

A spliced ``for_each_tile`` loop writes its trip counter ``u0`` into addresses
(e.g. ``d0 + 32*u0``). It describes the address advance from one trip to the
next, not an in-tile iteration axis; codegen already applies the advance once
and pins ``u0`` to zero in the base coordinates. ``per_trip_index`` is that pin,
shared with the layout passes.

The device-level regression (a real multi-trip vector page gather) lives in
``test_for_each_tile_e2e.py::TestForEachTileTripRangesE2E`` with the other
device e2e tests; the failing pre-scheduling pass runs on the device lowering,
so it is device-essential, not CPU.
"""

from __future__ import annotations

import sympy

from torch_spyre._inductor.pass_utils import per_trip_index


class _Hint:
    def __init__(self, loop_var, loop_var_range):
        self.loop_var = loop_var
        self.loop_var_range = loop_var_range


class _FakeOp:
    def __init__(self, hints):
        self.dim_hints = hints


def _u0() -> sympy.Symbol:
    return sympy.Symbol("u0", integer=True)


def test_no_hints_returns_index_unchanged():
    d0 = sympy.Symbol("d0")
    assert per_trip_index(_FakeOp([]), d0) == d0
    assert per_trip_index(None, d0) == d0


def test_pins_splice_var_to_zero():
    u0, d0 = _u0(), sympy.Symbol("d0")
    out = per_trip_index(_FakeOp([_Hint(u0, 4)]), d0 + 32 * u0)
    assert out == d0
    assert out.free_symbols == {d0}


def test_input_expression_not_mutated():
    u0, d0 = _u0(), sympy.Symbol("d0")
    idx = d0 + 32 * u0
    per_trip_index(_FakeOp([_Hint(u0, 4)]), idx)
    assert idx == d0 + 32 * u0


def test_hint_without_range_is_ignored():
    u0, d0 = _u0(), sympy.Symbol("d0")
    idx = d0 + 32 * u0
    assert per_trip_index(_FakeOp([_Hint(u0, None)]), idx) == idx
