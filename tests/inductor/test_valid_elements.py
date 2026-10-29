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

"""Groups of a SpyreTensorLayout and the device coordinates derived through them."""

import pytest
import sympy
import torch
from torch_spyre._C import SpyreTensorLayout, get_device_dtype

from torch_spyre._inductor import config, valid_elements
from torch_spyre._inductor.views import compute_coordinates

d0, d1 = sympy.symbols("d0 d1")
_FP16 = get_device_dtype(torch.float16)
_FP32 = get_device_dtype(torch.float32)


def _contiguous_strides(size):
    strides, acc = [], 1
    for s in reversed(size):
        strides.append(acc)
        acc *= s
    return strides[::-1]


@pytest.mark.parametrize(
    "size,dtype",
    [
        ([7, 100], torch.float16),
        ([4, 20], torch.float32),
        ([64, 1], torch.float16),
        ([3, 5, 130], torch.float16),
        ([2, 64], torch.float32),
        ([1, 64], torch.float16),
    ],
)
def test_the_map_init_records_is_used(size, dtype):
    stl = SpyreTensorLayout(size, dtype)
    info = valid_elements.valid_elements_of(stl, size, _contiguous_strides(size))
    assert info.groups == dict(stl.valid_elements)
    assert not info.derived


@pytest.mark.parametrize(
    "size,dtype",
    [([7, 100], torch.float16), ([4, 20], torch.float32), ([1, 64], torch.float16)],
)
def test_derived_groups_match_what_init_records(size, dtype):
    stl = SpyreTensorLayout(size, dtype)
    derived = valid_elements.derived_from_host(
        stl.device_size, stl.stride_map, size, _contiguous_strides(size)
    )
    assert derived == dict(stl.valid_elements)


def test_a_single_stick_strided_by_a_whole_stick_joins_the_stick_group():
    # A restickified [64, 35] buffer: its num-sticks dim steps a whole stick, where
    # init() records the 35 elements.
    derived = valid_elements.derived_from_host(
        [1, 64, 64], [64, 35, 1], [64, 35], [35, 1]
    )
    assert derived == {(0, 2): 35, (1,): 64}


def test_padding_without_a_map_is_not_derived():
    # [4, 20] fp32 with its single-stick num-sticks dim grown to two sticks.
    stl = SpyreTensorLayout([2, 4, 32], [32, 20, 1], _FP32, {})
    with pytest.raises(valid_elements.Underivable):
        valid_elements.valid_elements_of(stl, [4, 20], [20, 1])


@pytest.mark.parametrize("cols", [20, 32])
def test_coordinates_of_a_padded_stick_dim(cols):
    # Stride matching cannot place these: for 20 columns the row term becomes
    # 20*d0/32, for 32 the rows and the num-sticks dim share a stride.
    stl = SpyreTensorLayout([2, 4, 32], [32, cols, 1], _FP32, {(0, 2): cols, (1,): 4})
    coords = valid_elements.grouped_coordinates(
        stl, [4, cols], [cols, 1], {d0: 4, d1: cols}, cols * d0 + d1
    )
    assert coords == [0, d0, d1]


@pytest.mark.parametrize(
    "index,it_space",
    [
        (128 * d0 + d1, {d0: 4, d1: 128}),  # the whole tensor
        (128 * d0 + d1 + 64, {d0: 4, d1: 64}),  # the second stick of each row
        (128 * d0 + d1 + 3, {d0: 4, d1: 120}),  # a slice starting inside a stick
    ],
)
def test_coordinates_agree_with_stride_matching_on_full_tiles(index, it_space):
    stl = SpyreTensorLayout([4, 128], torch.float16)
    grouped = valid_elements.grouped_coordinates(
        stl, [4, 128], [128, 1], it_space, index
    )
    matched = compute_coordinates(
        list(stl.device_size), list(stl.stride_map), it_space, index, None
    )
    assert valid_elements.coordinates_agree(grouped, matched, it_space)


def test_a_sparse_stick_is_one_element_per_stick():
    stl = SpyreTensorLayout([64, 1], torch.float16)
    coords = valid_elements.grouped_coordinates(stl, [64, 1], [1, 1], {d0: 64}, d0)
    assert coords == [0, d0, 0]


def test_a_stored_map_that_contradicts_the_host_layout_is_not_used():
    # Scales widened on the host to one stick each, recorded as one element.
    stl = SpyreTensorLayout([3, 1, 64], [64, 64, 1], _FP16, {(0,): 3, (1, 2): 1})
    info = valid_elements.valid_elements_of(stl, [3, 64], [64, 1])
    assert info.groups == {(0,): 3, (1, 2): 64}
    assert info.derived


def test_members_that_do_not_nest_are_left_to_stride_matching():
    # 128 lanes per stick, but the num-sticks dim steps 64 elements at a time.
    stl = SpyreTensorLayout(
        [2, 128, 128], [8192, 1, 128], _FP16, {(0, 2): 128, (1,): 128}
    )
    with pytest.raises(valid_elements.Underivable, match="nest"):
        valid_elements.grouped_coordinates(
            stl, [128, 128], [128, 1], {d0: 128, d1: 128}, 128 * d0 + d1
        )


def test_an_overlapping_view_is_left_to_stride_matching():
    # An unfold-like read: two loop vars step the same host dim.
    stl = SpyreTensorLayout([128], torch.float16)
    with pytest.raises(valid_elements.Underivable):
        valid_elements.grouped_coordinates(
            stl, [128], [1], {d0: 4, d1: 64}, 16 * d0 + d1
        )


def test_an_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("SPYRE_VALID_ELEMENTS_COORDS", "yes")
    with pytest.raises(ValueError, match="SPYRE_VALID_ELEMENTS_COORDS"):
        config._valid_elements_coords_from_env()
