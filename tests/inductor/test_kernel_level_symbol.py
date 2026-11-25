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

"""The symbol the kernel names a loop level with (stage 7.3).

An op inside a counted loop lists the loop's variable in its ``dim_hints`` (level
``i`` is the ``i``-th loop-variable hint, outermost first), and the advance is
written in that variable. An op that lists none, or a count that differs from the
loop depth, gets a minted per-(op, level) symbol.
"""

from types import SimpleNamespace
from unittest import mock

import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor import spyre_kernel


def _symbol_kernel(hinted, loops):
    """A kernel whose current op lists ``hinted`` loop variables inside ``loops``
    enclosing loops, as ``_induction_symbol`` sees them."""
    kernel = spyre_kernel.SpyreKernel.__new__(spyre_kernel.SpyreKernel)
    kernel._tile_advance_symbols = {}
    kernel.current_node = SimpleNamespace(node=object())
    patches = (
        mock.patch.object(
            spyre_kernel,
            "loop_var_ranges_from_dim_hints",
            lambda node: dict.fromkeys(hinted, 4),
        ),
        mock.patch.object(spyre_kernel, "enclosing_loops", lambda node: [0] * loops),
    )
    return kernel, patches


def test_a_hinted_level_uses_the_loops_own_induction_variable():
    """Stage 7.3: the advance is written in the loop's variable, outermost first."""
    outer, inner = sympy.Symbol("u0"), sympy.Symbol("u1")
    kernel, patches = _symbol_kernel([outer, inner], loops=2)
    with patches[0], patches[1]:
        assert kernel._get_or_mint_level_symbol(0, "op3") is outer
        assert kernel._get_or_mint_level_symbol(1, "op3") is inner


def test_an_op_without_a_loop_variable_hint_still_mints_a_symbol():
    """Span-overflow and the allocator's tiling have no loop variable."""
    kernel, patches = _symbol_kernel([], loops=1)
    with patches[0], patches[1]:
        assert kernel._get_or_mint_level_symbol(0, "op3") == sympy.Symbol(
            "_tile_adv_op3_lvl0"
        )


def test_a_hint_count_that_differs_from_the_loop_depth_mints_a_symbol():
    """The pairing of hint to level is then unknown, as in ``_index_tile_advance``."""
    kernel, patches = _symbol_kernel([sympy.Symbol("u0")], loops=2)
    with patches[0], patches[1]:
        assert kernel._get_or_mint_level_symbol(0, "op3") == sympy.Symbol(
            "_tile_adv_op3_lvl0"
        )
