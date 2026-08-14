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
"""Unit tests for tensor_layout_utils.apply_layout (device-free, CPU only)."""

import torch

from oot_framework.tensor_layout_utils import apply_layout


def _seeded(shape, seed, dtype=torch.bfloat16):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.rand(shape, dtype=dtype)


def test_none_stride_zero_offset_returns_same_object():
    t = _seeded([2, 3], 0)
    assert apply_layout(t, None, 0) is t


def test_contiguous_stride_is_noop():
    t = _seeded([1, 4, 39, 128], 1)
    # Requesting t's own contiguous stride must not rebuild/redraw.
    assert apply_layout(t, list(t.stride()), 0) is t


def test_non_contiguous_transpose_preserves_values():
    # index_copy source layout: transpose(1, 2) of a contiguous [1, 39, 4, 128].
    t = _seeded([1, 4, 39, 128], 3123)
    out = apply_layout(t, [19968, 128, 512, 1], 0)
    assert out.stride() == (19968, 128, 512, 1)
    assert not out.is_contiguous()
    assert torch.equal(out, t)  # logical values unchanged


def test_storage_offset_preserves_values():
    t = _seeded([4], 7)
    out = apply_layout(t, [1], 64)
    assert out.storage_offset() == 64
    assert torch.equal(out, t)


def test_reproducible_from_same_seed():
    a = apply_layout(_seeded([1, 4, 39, 128], 5), [19968, 128, 512, 1], 0)
    b = apply_layout(_seeded([1, 4, 39, 128], 5), [19968, 128, 512, 1], 0)
    assert torch.equal(a, b)


def test_broadcast_stride_zero_reconstructs_expand():
    # GQA repeat_kv: a stride-0 (broadcast) dim.
    t = _seeded([1, 4, 7, 2048, 128], 11)
    out = apply_layout(t, [1048576, 262144, 0, 128, 1], 0)
    assert out.shape == (1, 4, 7, 2048, 128)
    assert out.stride()[2] == 0
    # Every slice along the broadcast dim aliases the same storage.
    assert torch.equal(out[:, :, 0], out[:, :, 5])
