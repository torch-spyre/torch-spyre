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

"""A dynamic dim's (min, max, granularity) across separately compiled kernels.

A dim is declared once, on the input, with ``.to(DEVICE_NAME, dynamic=...)``.
Each ``torch.compile`` region has its own ShapeEnv, so a second kernel only learns
the range if the first kernel's output carries it. These tests run a real chain:

    mid = tiled_sum(x)        # x declared dynamic
    out = tiled_triple(mid)   # compiled separately, from tiled_sum's output

Neither kernel states the range with ``torch._check``: any range a kernel compiles
with has to come from the declared contract, or from the stamp on ``mid``. The
kernels use ``for_each_tile``, the route that can carry a dynamic dim today. Each
kernel is its own function: two closures of one function that differ only in a
Python int share a Dynamo cache entry, and the second compile turns that int into
a symbol, which has nothing to do with what is tested here.
"""

import unittest
from unittest import mock

import pytest
import regex as re
import torch
from torch._inductor import config as inductor_config
from torch._inductor.exc import InductorError
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401  registers the "spyre" device
import torch_spyre._inductor.wrapper as wrapper
from torch_spyre._C import (
    SpyreTensorLayout,
    get_device_size_in_bytes,
    get_reserved_dims,
    get_spyre_tensor_layout,
)
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from utils_inductor import expects_raise

COLS = 64
TILE = 64
MIN_ROWS = 128
MAX_ROWS = 512
# The declared step is two tiles, so that the contract downstream has a granularity
# above 1 and a stamp that always said 1 would be caught.
GRANULARITY = 2 * TILE
DECLARED = {0: {"min": MIN_ROWS, "max": MAX_ROWS, "granularity": GRANULARITY}}
LEGAL_ROWS = list(range(MIN_ROWS, MAX_ROWS + 1, GRANULARITY))

# A sum over 64-row tiles leaves one row per tile. The input steps by two tiles, so
# the output has 2, 4, 6 or 8 rows: granularity 2.
SUM_MAX_ROWS = MAX_ROWS // TILE
SUM_CONTRACT = {
    0: {
        "min": MIN_ROWS // TILE,
        "max": SUM_MAX_ROWS,
        "granularity": GRANULARITY // TILE,
    }
}

# Sums of 64 fp16 values near 32 were measured to be off by about 0.15 in absolute
# terms (0.5%), and about 0.45 after the x3 in the second kernel. A relative
# tolerance of 2% leaves several times that headroom, and still fails on a missing
# or mis-sized tile, which is wrong by far more.
ATOL = 1e-1
RTOL = 2e-2


def tiled_sum(x):
    """Sum of each 64-row tile, as one row. [s, COLS] -> [s // TILE, COLS]."""
    torch._check(x.shape[0] % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, tile.sum(dim=0, keepdim=True)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


def tiled_triple(x):
    """x * 3 + 1, one row at a time. Takes the output of tiled_sum."""

    def body(_carry, tiles):
        (tile,) = tiles
        return None, tile * 3 + 1

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=1, out_dim=0)
    return out


def declared_input(rows):
    """A fresh input of ``rows`` rows with the contract declared, and its host copy."""
    x_cpu = torch.rand(rows, COLS, dtype=torch.float16)
    return x_cpu.to(DEVICE_NAME, dynamic=DECLARED), x_cpu


def compiled(fn):
    return torch.compile(fn, backend="inductor", fullgraph=True)


class TestCrossKernelContract(unittest.TestCase):
    def setUp(self):
        # A cache hit would skip the compile, and with it the code under test.
        torch._dynamo.reset()
        patch = inductor_config.patch(force_disable_caches=True)
        patch.__enter__()
        self.addCleanup(patch.__exit__, None, None, None)

    def test_output_contract_follows_the_output_size(self):
        # The output has s // TILE rows, so its contract is not the input's: the
        # range scales down and the granularity follows from the input's step.
        x, _ = declared_input(MIN_ROWS)

        out = compiled(tiled_sum)(x)

        self.assertEqual(get_reserved_dims(out), SUM_CONTRACT)

    def test_stamped_output_is_allocated_at_its_maximum(self):
        # The stamp promises SUM_MAX_ROWS rows, so the buffer must hold them at
        # every size, in one layout.
        expected = SpyreTensorLayout([SUM_MAX_ROWS, COLS], torch.float16)
        kernel = compiled(tiled_sum)
        for rows in LEGAL_ROWS:
            with self.subTest(rows=rows):
                x, _ = declared_input(rows)
                out = kernel(x)
                layout = get_spyre_tensor_layout(out)
                self.assertEqual(list(layout.device_size), list(expected.device_size))
                self.assertGreaterEqual(
                    out.untyped_storage().nbytes(), get_device_size_in_bytes(expected)
                )

    def test_second_kernel_is_built_for_the_maximum_the_first_one_stamped(self):
        """Not for the size it happened to be compiled at.

        At MIN_ROWS the intermediate has 2 rows, so a kernel that did not read the
        stamp would be built for 2 (the optimization hint) and be wrong for every
        larger size. Asserting the carried maximum is the discriminating check.
        """
        x, _ = declared_input(MIN_ROWS)
        mid = compiled(tiled_sum)(x)

        _out, code = run_and_get_code(compiled(tiled_triple), mid)

        carried = re.search(
            r"count_symbol_bounds=\{'(s\d+)': \((\d+), (\d+)\)\}", "\n".join(code)
        )
        self.assertIsNotNone(
            carried, "the second kernel's loop count reached the bundle as a constant"
        )
        self.assertEqual(int(carried.group(2)), SUM_MAX_ROWS)

    @expects_raise
    def test_second_kernel_is_refused_without_the_stamp(self):
        # The counterpart of the test above: with the stamp switched off, the
        # second kernel's loop count has no upper bound and cannot be built.
        x, _ = declared_input(MIN_ROWS)
        with mock.patch.object(wrapper, "output_reserved_dims", return_value=None):
            mid = compiled(tiled_sum)(x)
        self.assertIsNone(get_reserved_dims(mid))

        with pytest.raises(InductorError, match="no finite upper bound"):
            compiled(tiled_triple)(mid)

    def test_chain_is_correct_at_every_legal_size(self):
        # The failure this guards against is silent: a kernel built for the first
        # call's size returns correct rows below it and wrong rows above it.
        kernel_1, kernel_2 = compiled(tiled_sum), compiled(tiled_triple)
        for rows in LEGAL_ROWS:
            with self.subTest(rows=rows):
                x, x_cpu = declared_input(rows)
                out = kernel_2(kernel_1(x))
                sums = x_cpu.float().reshape(-1, TILE, COLS).sum(dim=1)
                torch.testing.assert_close(
                    out.cpu().float(), sums * 3 + 1, atol=ATOL, rtol=RTOL
                )
                self.assertEqual(get_reserved_dims(out), SUM_CONTRACT)

    def test_a_static_output_is_not_stamped(self):
        # Ordinary outputs are unchanged: only a symbolic size gets a contract.
        x = torch.rand(MIN_ROWS, COLS, dtype=torch.float16).to(DEVICE_NAME)

        out = compiled(lambda x: x * 2 + 1)(x)

        self.assertIsNone(get_reserved_dims(out))

    @expects_raise
    def test_output_with_two_symbolic_dims_is_refused(self):
        # Reservations cover one dim today, so a second symbolic dim has no home.
        a, _ = declared_input(MIN_ROWS)
        b, _ = declared_input(MIN_ROWS)

        with pytest.raises(InductorError, match="more than one symbolic dim"):
            compiled(lambda a, b: a.unsqueeze(1) + b.unsqueeze(0))(a, b)


if __name__ == "__main__":
    unittest.main()
