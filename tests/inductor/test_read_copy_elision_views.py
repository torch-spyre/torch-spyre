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

"""Read-copy ownership must be compared in a proven common coordinate basis."""

import dataclasses

import pytest
import sympy

from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.read_copy_elision import (
    _views_match_through_read_coordinates,
)


H, F = sympy.symbols("h f", integer=True)
# Match the compiler's physical-core symbol, including its assumptions.
C = sympy.Symbol("core_id")
DIRECT = PerCoreView(((1, 4),), ((1, sympy.Mod(C, 4)),), 32)
STAGED = PerCoreView(((2, 4),), ((2, sympy.Mod(C, 4)),), 32)
DIRECT_SIZE = [128, 2816, 11, 64]
STAGED_SIZE = [1, 11, 2816, 64]
DIRECT_COORDS = [0, H, sympy.floor(F / 64), sympy.Mod(F, 64)]
STAGED_COORDS = [0, sympy.floor(F / 64), H, sympy.Mod(F, 64)]


def check(
    staged=STAGED,
    direct=DIRECT,
    staged_size=STAGED_SIZE,
    direct_size=DIRECT_SIZE,
    staged_coords=STAGED_COORDS,
    direct_coords=DIRECT_COORDS,
):
    return _views_match_through_read_coordinates(
        staged, staged_size, staged_coords, direct, direct_size, direct_coords
    )


def test_permuted_layout_preserves_same_logical_weight_slice():
    assert check()


def test_same_layout_preserves_same_core_owners():
    assert check(staged=DIRECT, staged_size=DIRECT_SIZE, staged_coords=DIRECT_COORDS)


def test_equal_splits_different_core_owners_declines():
    assert not check(
        direct=dataclasses.replace(
            DIRECT, core_to_slot=((1, sympy.Mod(sympy.floor(C / 8), 4)),)
        )
    )


def test_different_physical_core_count_declines():
    assert not check(direct=dataclasses.replace(DIRECT, num_cores=16))


@pytest.mark.parametrize(
    "coordinates",
    [
        [0, H + 1, sympy.floor(F / 64), sympy.Mod(F, 64)],
        [0, F, sympy.floor(H / 64), sympy.Mod(H, 64)],
        [0, H, H, sympy.Mod(F, 64)],
    ],
)
def test_shifted_wrong_or_ambiguous_coordinates_decline(coordinates):
    assert not check(direct_size=[128, 2816, 2816, 64], direct_coords=coordinates)


def test_different_extent_declines():
    assert not check(direct_size=[128, 5632, 11, 64])


def test_missing_coordinate_declines():
    assert not check(direct_coords=DIRECT_COORDS[:-1])


def test_missing_owner_declines():
    assert not check(staged=dataclasses.replace(STAGED, core_to_slot=()))
