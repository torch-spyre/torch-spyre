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

"""Tests for the module-level cache backing ``arange`` and mask coordinates.

The cache keeps one entry per (device, dtype, form), holding the longest ramp or
row column built so far up to that form's bound, and serves a shorter request as
a slice of it.  These tests cover the properties that follow: an entry grows
monotonically to a power-of-two multiple of a stick, a slice out of a grown
buffer still holds the values the caller asked for, the two forms are built
independently, and a request past the bound is served without being kept.

Every value checked here stays at or below 1024, the largest integer DLFloat16
represents exactly, so ``torch.equal`` reports only real cache errors.
"""

import threading
import unittest

import torch

from torch_spyre._inductor.op_spec import (
    _ARANGE_MAX_LEN,
    _ARANGE_SEED_LEN,
    _ARANGE_TENSOR_CACHE,
    clear_arange_tensor_cache,
    spyre_arange_tensor,
)

SPYRE = torch.device("spyre")


class TestArangeTensorCache(unittest.TestCase):
    """Test module-level caching for the ramp and row-column coordinates."""

    def setUp(self):
        clear_arange_tensor_cache()

    def tearDown(self):
        clear_arange_tensor_cache()

    def test_single_entry_per_dtype_and_device(self):
        """Ramp requests of many lengths share one entry; a second dtype adds one."""
        for length in (64, 100, 512, 1024):
            spyre_arange_tensor(length, SPYRE, torch.float16)

        self.assertEqual(len(_ARANGE_TENSOR_CACHE), 1)

        spyre_arange_tensor(64, SPYRE, torch.float32)
        self.assertEqual(len(_ARANGE_TENSOR_CACHE), 2)

    def test_growth_doubles_and_never_shrinks(self):
        """The cached length reaches a power-of-two multiple of the seed and stays."""
        spyre_arange_tensor(_ARANGE_SEED_LEN + 1, SPYRE, torch.float16)
        (grown_first,) = {entry[0] for entry in _ARANGE_TENSOR_CACHE.values()}
        self.assertEqual(grown_first, 2 * _ARANGE_SEED_LEN)

        spyre_arange_tensor(4 * _ARANGE_SEED_LEN, SPYRE, torch.float16)
        (grown_second,) = {entry[0] for entry in _ARANGE_TENSOR_CACHE.values()}
        self.assertEqual(grown_second, 4 * _ARANGE_SEED_LEN)

        # A shorter request is served from the grown buffer rather than replacing it.
        spyre_arange_tensor(_ARANGE_SEED_LEN, SPYRE, torch.float16)
        (grown_third,) = {entry[0] for entry in _ARANGE_TENSOR_CACHE.values()}
        self.assertEqual(grown_third, 4 * _ARANGE_SEED_LEN)

    def test_ramp_holds_requested_values_after_growth(self):
        """A ramp sliced out of a longer buffer holds exactly ``[0, length)``."""
        spyre_arange_tensor(1024, SPYRE, torch.float16)

        for length in (_ARANGE_SEED_LEN, 100, 1024):
            ramp = spyre_arange_tensor(length, SPYRE, torch.float16)
            self.assertEqual(list(ramp.shape), [length])
            expect = torch.arange(length, dtype=torch.float16)
            self.assertTrue(
                torch.equal(ramp.cpu(), expect), f"ramp values wrong at {length}"
            )

    def test_column_holds_requested_values_after_growth(self):
        """A row column sliced out of a longer buffer keeps ``out[i][0] == i``."""
        spyre_arange_tensor(1024, SPYRE, torch.float16, column=True)

        for length in (_ARANGE_SEED_LEN, 100, 1024):
            column = spyre_arange_tensor(length, SPYRE, torch.float16, column=True)
            self.assertEqual(list(column.shape), [length, 1])
            expect = torch.arange(length, dtype=torch.float16).unsqueeze(-1)
            self.assertTrue(
                torch.equal(column.cpu(), expect), f"column values wrong at {length}"
            )

    def test_ramp_does_not_build_column(self):
        """Each form has its own entry, so a ramp request builds no column."""
        spyre_arange_tensor(1024, SPYRE, torch.float16)
        self.assertEqual([key[-1] for key in _ARANGE_TENSOR_CACHE], [False])

        column = spyre_arange_tensor(1024, SPYRE, torch.float16, column=True)
        self.assertEqual(list(column.shape), [1024, 1])
        self.assertEqual(sorted(key[-1] for key in _ARANGE_TENSOR_CACHE), [False, True])

    def test_request_past_bound_is_not_kept(self):
        """A request longer than its form's bound is served exactly, uncached.

        A cached entry lives for the process, so keeping such a request would
        pin its memory for good.  The values are checked only at positions at
        or below 1024, where DLFloat16 is exact.
        """
        spyre_arange_tensor(1024, SPYRE, torch.float16)
        for column in (False, True):
            length = _ARANGE_MAX_LEN[column] + 64
            coord = spyre_arange_tensor(length, SPYRE, torch.float16, column=column)
            expect_shape = [length, 1] if column else [length]
            self.assertEqual(list(coord.shape), expect_shape)
            head = coord.cpu().flatten()[:1024]
            self.assertTrue(
                torch.equal(head, torch.arange(1024, dtype=torch.float16)),
                f"values wrong past the bound, column={column}",
            )

        # Only the in-bound ramp is kept, at the length it grew to.
        self.assertEqual(
            {key[-1]: entry[0] for key, entry in _ARANGE_TENSOR_CACHE.items()},
            {False: 1024},
        )

    def test_growth_stops_at_bound(self):
        """A request at the bound is kept, at exactly the bound."""
        length = _ARANGE_MAX_LEN[True]
        spyre_arange_tensor(length, SPYRE, torch.float16, column=True)
        (grown,) = {entry[0] for entry in _ARANGE_TENSOR_CACHE.values()}
        self.assertEqual(grown, length)

    def test_non_spyre_device_rejected(self):
        """Only a spyre device is served.

        The sole caller is generated wrapper code, which emits the device the
        node was laid out for, so a request for anything else is a bug in the
        caller rather than a case to serve on the host.
        """
        with self.assertRaises(ValueError):
            spyre_arange_tensor(64, torch.device("cpu"), torch.float16)

        self.assertEqual(len(_ARANGE_TENSOR_CACHE), 0)

    def test_clear_releases_every_entry(self):
        """Clearing empties the cache so a test starts from a known state."""
        spyre_arange_tensor(64, SPYRE, torch.float16)
        spyre_arange_tensor(64, SPYRE, torch.float32)
        self.assertEqual(len(_ARANGE_TENSOR_CACHE), 2)

        clear_arange_tensor_cache()
        self.assertEqual(len(_ARANGE_TENSOR_CACHE), 0)

    def test_concurrent_growth_keeps_one_entry(self):
        """Racing threads that each grow the cache still leave a single entry."""
        lengths = [64, 128, 256, 512, 1024]
        results: list[torch.Tensor] = []
        results_lock = threading.Lock()

        def build(length):
            ramp = spyre_arange_tensor(length, SPYRE, torch.float16)
            with results_lock:
                results.append(ramp)

        threads = [threading.Thread(target=build, args=(n,)) for n in lengths]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(_ARANGE_TENSOR_CACHE), 1)
        self.assertEqual(len(results), len(lengths))
        for ramp in results:
            expect = torch.arange(ramp.shape[0], dtype=torch.float16)
            self.assertTrue(torch.equal(ramp.cpu(), expect))


if __name__ == "__main__":
    unittest.main()
