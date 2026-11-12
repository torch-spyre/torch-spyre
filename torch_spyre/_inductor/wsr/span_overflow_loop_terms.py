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

"""Put a span-overflow group's loop in the index of its accesses (stage 7a.2).

The legacy lowering (``coarse_tile_post_stickify``) records each access's per-trip
advance in ``loop_info`` and leaves the index without a loop term. This module adds
the term, ``step * u``, to every advancing access of a group whose ops all
``copy_out``, and gives every op of the group the loop variable in its
``dim_hints``. The kernel then reads the advance from the index and checks it
against the record (``LoopAdvanceMismatch``), which stays until stage 7.4.

Everything here edits by wrapping: an op's ``inner_fn`` and ``store_output`` are
wrapped, never rebuilt, because the index expressions go stale as soon as a pass
runs (issue #2797).
"""

from collections.abc import Iterable

import regex
import sympy

from ..errors import Unsupported

_INDUCTION_NAME = regex.compile(r"u(\d+)")


def mint_induction(taken: Iterable[sympy.Basic]) -> sympy.Symbol:
    """A loop variable named ``u<n>``, with the smallest ``n`` no symbol in
    ``taken`` already uses.

    ``taken`` holds symbols or expressions; every free symbol counts. The name
    family is the one Inductor gives an unbacked scalar (``for_each_tile``'s loop
    variable is one), so a graph holding both kinds of loop needs distinct names.
    The variable is an integer, so ``floor(8*u)`` folds to ``8*u``.
    """
    used: set[int] = set()
    for item in taken:
        for symbol in getattr(item, "free_symbols", ()):
            match = _INDUCTION_NAME.fullmatch(symbol.name)
            if match:
                used.add(int(match.group(1)))
    n = 0
    while n in used:
        n += 1
    return sympy.Symbol(f"u{n}", integer=True, nonnegative=True)


def access_step(
    index: sympy.Expr,
    dim_symbol: sympy.Symbol | None,
    extent: sympy.Expr,
    buffer_stride: sympy.Expr,
) -> sympy.Expr:
    """The host-element advance of one access per trip, along one tiled dim.

    ``index`` is the access's own index. ``dim_symbol`` is the index symbol of the
    tiled dim, or ``None``; ``extent`` is the tile's extent along that dim.
    When the symbol is in the index the step is its coefficient times the extent.
    A dim tiled down to extent one is squeezed out of the op, so no symbol
    survives; the step is then the buffer's own stride times the extent. It is
    never the product of the sizes to the right of the dim, which is the stride
    only for a contiguous buffer (#6093).

    Raises ``Unsupported`` for an extent that is not an integer.
    """
    if not isinstance(extent, sympy.Integer):
        raise Unsupported(
            f"span-overflow loop term: the tile extent {extent} is not an integer "
            "(symbolic span-overflow tiling is not supported)."
        )
    if dim_symbol is not None and dim_symbol in index.free_symbols:
        coefficient = index.coeff(dim_symbol)
        if coefficient != 0:
            return coefficient * extent
    return sympy.sympify(buffer_stride) * extent
