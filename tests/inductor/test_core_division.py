# Copyright 2025 The Torch-Spyre Authors.
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

import math

import sympy

from torch_spyre._inductor.work_division_utils import (
    multi_dim_iteration_space_split,
    multi_dim_core_split,
    extend_iteration_space_with_core_division,
)


def test_extend_iteration_space_uses_symbol_keys_for_core_division() -> None:
    """Regression test for #1244: lookup must use Symbol key, not size value."""
    k = sympy.Symbol("k")
    it_space = {k: 1152}
    core_division = {k: 8}

    it_space_extended = extend_iteration_space_with_core_division(
        it_space, core_division
    )

    assert it_space_extended[k] == (1152, 8)


def test_extend_iteration_space_defaults_to_single_core() -> None:
    m = sympy.Symbol("m")
    it_space = {m: 256}

    it_space_extended = extend_iteration_space_with_core_division(it_space, {})

    assert it_space_extended[m] == (256, 1)


def test_multi_dim_iteration_space_split_creates_real_split() -> None:
    """Guard against accidental control-flow changes that disable splitting."""
    mb, k, n = sympy.symbols("mb k n")
    it_space = {mb: 256, k: 1152, n: 256}

    splits = multi_dim_iteration_space_split(
        it_space,
        max_cores=32,
        priorities=[mb, n, k],
    )

    assert math.prod(splits.values()) > 1
    assert splits[mb] == 32


def test_multi_dim_iteration_space_split_respects_core_budget() -> None:
    """Split product must never exceed max_cores."""
    a, b, c = sympy.symbols("a b c")
    it_space = {a: 4096, b: 2048, c: 1024}

    splits = multi_dim_iteration_space_split(
        it_space,
        max_cores=16,
        priorities=[a, b, c],
    )

    assert math.prod(splits.values()) <= 16
    assert math.prod(splits.values()) > 1


def test_multi_dim_core_split_uses_all_divisible_cores() -> None:
    sizes = [256, 256, 64]

    splits = multi_dim_core_split(sizes, max_cores=32, priorities=[3, 2, 1])

    assert math.prod(splits) == 32
    assert splits[0] > 1


def test_multi_dim_core_split_respects_disabled_dimension_priority() -> None:
    """Negative priority excludes a dimension from splitting."""
    sizes = [256, 1152, 256]
    priorities = [3, -1, 2]

    splits = multi_dim_core_split(sizes, max_cores=32, priorities=priorities)

    assert splits[1] == 1
    assert math.prod(splits) > 1
