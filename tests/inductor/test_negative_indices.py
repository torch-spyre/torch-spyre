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

"""Negative advanced indexing (``x[idx]`` and ``x.index_put``) on the compiled path.

Indices stay inside ``[-size, size)``. PyTorch raises IndexError outside it and
Spyre has no device-side bounds check, so exercising that range faults the card
instead of reporting a failure.
"""

import unittest

import torch

from utils_inductor import compare_with_cpu

INDEX_DTYPES = (torch.int64, torch.int32)


def gather(x, idx):
    return x[idx]


def scatter(x, idx, src):
    return x.index_put([idx], src)


def values(*shape):
    """Random fp16 values that survive a device round trip exactly.

    The device fp16 format holds one mantissa bit less than IEEE fp16, so a
    full-mantissa value changes on transfer. Integers are exact in both, which
    lets these tests assert equality rather than pick a tolerance.
    """
    return torch.randint(-100, 100, shape).to(torch.float16)


def compare(fn, *args):
    """Compare against CPU. Eager has no aten::index kernel for spyre.

    Exact: indexing selects values, it does not compute them.
    """
    compare_with_cpu(fn, *args, atol=0, rtol=0, run_eager=False)


class TestNegativeIndexing(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_gather_negative_indices(self):
        x = values(16, 64)
        for dtype in INDEX_DTYPES:
            with self.subTest(dtype=dtype):
                compare(gather, x, torch.tensor([-1, -2, 0, 5, -1], dtype=dtype))

    def test_gather_every_index_in_range(self):
        """Full [-size, size) sweep over partial and whole stick lengths."""
        for size in (5, 16, 31, 32, 64, 127):
            for dtype in INDEX_DTYPES:
                with self.subTest(size=size, dtype=dtype):
                    x = values(size, 64)
                    compare(gather, x, torch.arange(-size, size, dtype=dtype))

    def test_gather_multi_dim_index(self):
        """2-D index, as used for expert-id lookups.

        Element counts are stick-aligned. An unaligned multi-dim index fails in
        the conversion's layout search, independently of index sign.
        """
        x = values(32, 64)
        for dtype in INDEX_DTYPES:
            for shape in ((4, 8), (2, 16)):
                with self.subTest(dtype=dtype, shape=shape):
                    n = shape[0] * shape[1]
                    compare(
                        gather,
                        x,
                        (torch.arange(n, dtype=dtype) - n // 2).reshape(shape),
                    )

    def test_gather_strided_index(self):
        """A non-contiguous index must read the elements it names."""
        x = values(64, 64)
        base = torch.arange(-32, 32, dtype=torch.int64)
        for step, count in ((2, 10), (3, 7)):
            with self.subTest(step=step, count=count):
                compare(gather, x, base[::step][:count])

    def test_gather_positive_indices(self):
        x = values(64, 64)
        for dtype in INDEX_DTYPES:
            with self.subTest(dtype=dtype):
                compare(gather, x, torch.arange(64, dtype=dtype))

    def test_gather_fused_consumer(self):
        """The index must stay correct once the gather fuses with its consumer."""
        x = values(32, 64)
        idx = torch.tensor([-1, -2, 3], dtype=torch.int64)
        compare(lambda t, i: t[i] * 2 + 1, x, idx)

    def test_gather_repeated_calls(self):
        """A failed launch leaves the stream unusable, so check a second call."""
        x = values(16, 64)
        idx = torch.tensor([-1, -2, 0, 5], dtype=torch.int64)
        for _ in range(2):
            compare(gather, x, idx)

    # Scatter indices are distinct: index_put without accumulate is
    # order-dependent for duplicates, which would make the comparison unstable.

    def test_scatter_partial_stick_length(self):
        x = torch.zeros(16, 64, dtype=torch.float16)
        src = values(4, 64)
        for dtype in INDEX_DTYPES:
            with self.subTest(dtype=dtype):
                compare(scatter, x, torch.tensor([-1, -2, 0, 5], dtype=dtype), src)

    def test_scatter_whole_stick_length(self):
        x = torch.zeros(64, 64, dtype=torch.float16)
        src = values(32, 64)
        for dtype in INDEX_DTYPES:
            with self.subTest(dtype=dtype):
                compare(scatter, x, torch.arange(-32, 0, dtype=dtype), src)

    def test_scatter_positive_indices(self):
        x = torch.zeros(16, 64, dtype=torch.float16)
        src = values(4, 64)
        compare(scatter, x, torch.tensor([1, 2, 0, 5], dtype=torch.int64), src)


if __name__ == "__main__":
    unittest.main()
