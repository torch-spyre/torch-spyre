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

"""Helpers in ``scratchpad/utils.py`` that decide LX residency.

``_would_produce_lx_back_gap`` asks whether a device dimension is *fully
covered* by the iteration symbols that walk it: a backGap fires when
``device_size[d]`` exceeds the extent the coordinate actually reaches, and the
backend supports that for HBM but not for LX.

The covered extent is a property of the whole coordinate expression, not of any
one symbol in it. These tests pin that, because the two obvious cheaper
approximations are both wrong and were both live:

* Substituting *one* symbol's iteration range (what this did before) breaks as
  soon as a coordinate folds two symbols into one axis, in both directions --
  see ``test_multi_symbol_*`` for a spurious gap and ``test_stickified_*`` for a
  missed one. It was also read out of the unordered ``free_symbols`` set, so
  which symbol you got moved with ``PYTHONHASHSEED``; that made LX plans, and
  hence core-division plans, differ between processes on identical input.
* Substituting *every* symbol's maximum is right only for a monotone
  coordinate. ``test_mod_coordinate_uses_a_real_bound`` is the counter-example.
"""

import unittest
from types import SimpleNamespace
from unittest import TestCase, mock

import sympy
from torch._inductor.dependencies import MemoryDep

from torch_spyre._inductor.scratchpad.utils import _would_produce_lx_back_gap

_COORDS = "torch_spyre._inductor.scratchpad.utils.device_coordinates"

_BUF = "buf0"

# Iteration symbols, named as the pre-scheduler names them.
d0, d1, d2, d3 = sympy.symbols("d0 d1 d2 d3", integer=True, nonnegative=True)

# The trailing device coordinate is the within-stick lane, which the check skips
# (a stick is atomic, so it cannot carry a gap). Every fixture here has exactly
# one non-stick device dim, so ``device_size[0]`` is the dim under test.
_STICK_COORD = sympy.Mod(d0, 64)
_STICK_SIZE = 64


def _graph(extent, ranges):
    """A one-op graph whose only read of ``_BUF`` carries ``ranges``.

    ``extent`` is ``device_size[0]``, the non-stick device dim under test.
    ``ranges`` maps each iteration symbol to its size, which is what
    ``MemoryDep`` exposes as ``dep.ranges``.
    """
    symbols = tuple(ranges)
    dep = MemoryDep(
        _BUF,
        # The index is unused here: the coordinate under test is supplied by the
        # patched ``device_coordinates``. Kept plausible rather than empty.
        sum(symbols, sympy.Integer(0)),
        symbols,
        tuple(ranges[sym] for sym in symbols),
    )
    op = SimpleNamespace(
        get_read_writes=lambda: SimpleNamespace(reads={dep}, writes=set())
    )
    buf = SimpleNamespace(
        layout=SimpleNamespace(
            device_layout=SimpleNamespace(device_size=[extent, _STICK_SIZE])
        )
    )
    return SimpleNamespace(operations=[op], get_buffer=lambda _name: buf)


def _back_gap(coord, extent, ranges):
    graph = _graph(extent, ranges)
    with mock.patch(_COORDS, return_value=[coord, _STICK_COORD]):
        return _would_produce_lx_back_gap(graph, _BUF, [0])


class BackGapTest(TestCase):
    def test_multi_symbol_coordinate_that_covers_its_dim_has_no_gap(self):
        """``2*d1 + d3`` over ``d1 in [0,40)``, ``d3 in [0,2)`` reaches 0..79.

        The dim is 80 wide, so it is exactly covered and there is no gap. Note
        that *neither* single symbol's range answers this: 80 > 40 and 80 > 2 are
        both true, so picking either one reports a gap that is not there. This
        case fails on any hash seed, which is what makes it a regression test.
        """
        self.assertFalse(_back_gap(2 * d1 + d3, 80, {d1: 40, d3: 2}))

    def test_multi_symbol_stickified_coordinate_has_no_gap(self):
        """``2*d0 + floor(d2/64)`` over ``d0 in [0,8)``, ``d2 in [0,128)``.

        The shape observed on a granite-4.0-micro decoder block, where this
        decided one buffer's LX residency and, through it, the whole
        core-division plan. Reaches 0..15 on a dim 16 wide: no gap. Here the two
        symbols disagree (``d0`` says gap, ``d2`` says none), which is how the
        verdict came to depend on the hash seed.
        """
        self.assertFalse(_back_gap(2 * d0 + sympy.floor(d2 / 64), 16, {d0: 8, d2: 128}))

    def test_uncovered_dim_has_a_gap(self):
        """``d0`` over ``[0,8)`` reaches 0..7, which leaves a dim 16 wide short.

        The positive case: without it the tests above would pass against a
        function that always returned False.
        """
        self.assertTrue(_back_gap(d0, 16, {d0: 8}))

    def test_stickified_coordinate_gap_is_not_hidden_by_the_element_range(self):
        """``floor(d1/64)`` over ``d1 in [0,1024)`` reaches 0..15, not 0..1023.

        A stick-count coordinate covers ``range/64``, so a dim 32 wide really
        does have a gap. Reading ``d1``'s raw *element* range instead (1024)
        overstates coverage by 64x and hides it -- the dangerous direction, since
        LX has no backGap support, so a missed gap is a codegen hazard rather
        than a lost optimization.
        """
        self.assertTrue(_back_gap(sympy.floor(d1 / 64), 32, {d1: 1024}))

    def test_mod_coordinate_uses_a_real_bound(self):
        """``Mod(d0, 5)`` over ``d0 in [0,8)`` reaches 0..4, so a dim 4 wide is
        covered.

        Non-monotone, so substituting the symbol's maximum is not its bound:
        ``Mod(7, 5)`` is 2, which would understate the extent as 3 and report a
        gap. This is why the check needs real range analysis and not ``subs``.
        """
        self.assertFalse(_back_gap(sympy.Mod(d0, 5), 4, {d0: 8}))


if __name__ == "__main__":
    unittest.main()
