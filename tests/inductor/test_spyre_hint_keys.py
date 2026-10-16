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

"""spyre_hint accepts only the keys that still drive a pass.

Tiling is expressed with for_each_tile; the old tiling keys were removed and
must fail loudly rather than silently produce an untiled graph.
"""

import pytest
import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor.propagate_hints import spyre_hint


@pytest.mark.parametrize("key", ["num_tiles_per_dim", "tiles", "slices"])
def test_removed_tiling_keys_point_to_for_each_tile(key):
    with pytest.raises(ValueError, match=rf"{key}.*for_each_tile"):
        spyre_hint(**{key: {"M": 2}})


def test_unknown_key_lists_the_accepted_keys():
    with pytest.raises(ValueError, match="bogus") as excinfo:
        spyre_hint(bogus=1)
    for accepted in (
        "work_div",
        "named_dims",
        "expected_named_dims",
        "expected_reduction_dims",
    ):
        assert accepted in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"work_div": {"M": 2}},
        {"named_dims": ["M", "K"]},
        {"expected_named_dims": ["M", "K"]},
        {"expected_reduction_dims": ["K"]},
        {"named_dims": ["M", "K"], "work_div": {"M": 2}},
    ],
)
def test_valid_keys_are_accepted(kwargs):
    spyre_hint(**kwargs)


def test_removed_key_raises_inside_torch_compile():
    def fn(x):
        with spyre_hint(num_tiles_per_dim={"M": 2}):
            return x + 1

    with pytest.raises(Exception, match="num_tiles_per_dim"):
        torch.compile(fn, backend="eager")(torch.zeros(4))
