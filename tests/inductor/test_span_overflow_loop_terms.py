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

"""Loop terms in the index of span-overflow groups (stage 7a.2).

The unit tests cover the step rule and the loop-variable minting. The device tests
add the term to the legacy lowering's result and check that the kernel reads the
same advance from the index as the legacy record holds (``LoopAdvanceMismatch``
otherwise), with the same numerics and the same kernel text.
"""

import pytest
import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.wsr.span_overflow_loop_terms import (
    access_step,
    mint_induction,
)

d0, d1 = sympy.symbols("d0 d1")
u0, u1, u2, u3 = (sympy.Symbol(f"u{n}") for n in range(4))


def test_mint_skips_taken_u_names():
    assert mint_induction([u0, u2]).name == "u1"
    assert mint_induction([u0, u1]).name == "u2"
    assert mint_induction([]).name == "u0"


def test_mint_counts_a_symbol_inside_an_expression():
    assert mint_induction([3 * u1 + d0, 7 * u0]).name == "u2"


def test_mint_ignores_names_that_are_not_loop_variables():
    assert mint_induction([sympy.Symbol("u"), sympy.Symbol("us0"), d0]).name == "u0"


def test_mint_is_integer_nonnegative():
    symbol = mint_induction([])
    assert symbol.is_integer and symbol.is_nonnegative
    assert sympy.floor(8 * symbol) == 8 * symbol


def test_step_uses_the_index_coefficient():
    assert access_step(1024 * d0 + d1, d1, sympy.Integer(512), 99) == 512
    assert access_step(1024 * d0 + d1, d0, sympy.Integer(2048), 99) == 2048 * 1024


def test_step_uses_the_stride_when_the_dim_is_squeezed():
    assert access_step(512 * d1, None, sympy.Integer(1), 32768) == 32768
    # a symbol that is not in the index is the same case
    assert access_step(512 * d1, d0, sympy.Integer(1), 32768) == 32768


def test_step_is_not_the_product_of_the_sizes_to_the_right():
    # (8, 8, 64, 512) permuted (1, 0, 2, 3): strides (32768, 262144, 512, 1).
    # Dim 0 tiled 8 ways is squeezed to extent 1; the product of the sizes to its
    # right is 262144 (dim 1's stride), the buffer's stride for dim 0 is 32768.
    strides = (32768, 262144, 512, 1)
    assert access_step(sympy.S.Zero, None, sympy.Integer(1), strides[0]) == 32768


def test_a_symbolic_extent_is_unsupported():
    with pytest.raises(Unsupported, match="not an integer"):
        access_step(d0, d0, sympy.Symbol("s0"), 1)
