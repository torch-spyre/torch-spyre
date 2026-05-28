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

"""Tests for layout solvers"""

from unittest import TestCase
from torch_spyre._inductor.scratchpad.plan_solver import (
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
)

LARGE_SIZE = 512
SMALL_SIZE = 10
ALIGNMENT = 128


class TestGreedySolver(TestCase):
    def verify_layout(
        self,
        buffers: list[LifetimeBoundBuffer],
        expected_addresses: list[int | None],
        size=SMALL_SIZE,
        alignment=1,
    ):
        result = GreedyLayoutSolver(size, alignment).plan_layout(buffers)
        for planned, expected_addr in zip(result, expected_addresses, strict=True):
            self.assertEqual(planned.address, expected_addr)

    def test_simple_layout(self):
        # Three non-overlapping buffers fill memory sequentially.
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 0, 2),
            LifetimeBoundBuffer("buffer2", 4, 0, 2),
        ]
        self.verify_layout(buffers, [0, 3, 6])

    def test_simple_layout_below_alignment(self):
        # Buffers smaller than the alignment boundary are evicted (address=None).
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 0, 2),
            LifetimeBoundBuffer("buffer2", 4, 0, 2),
        ]
        self.verify_layout(buffers, [0, None, None], alignment=ALIGNMENT)

    def test_alignment_enforced(self):
        # Each buffer is placed at the next alignment boundary.
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 0, 2),
            LifetimeBoundBuffer("buffer2", 4, 0, 2),
        ]
        self.verify_layout(buffers, [0, 128, 256], LARGE_SIZE, ALIGNMENT)

    def test_simple_eviction_layout(self):
        # buffer1 is evicted because it won't fit; buffer2 reuses buffer0's space.
        buffers = [
            LifetimeBoundBuffer("buffer0", 7, 0, 2),
            LifetimeBoundBuffer("buffer1", 4, 0, 2),
            LifetimeBoundBuffer("buffer2", 3, 0, 2),
        ]
        self.verify_layout(buffers, [0, None, 7])

    def test_realloc(self):
        # buffer1's lifetime starts after buffer0 ends, so it reuses address 0.
        buffers = [
            LifetimeBoundBuffer("buffer0", 10, 0, 2),
            LifetimeBoundBuffer("buffer1", 3, 2, 3),
        ]
        self.verify_layout(buffers, [0, 0])

    def test_realloc_between(self):
        # buffer3's lifetime begins after buffer1 ends, so it reclaims buffer1's slot.
        buffers = [
            LifetimeBoundBuffer("buffer0", 3, 0, 4),
            LifetimeBoundBuffer("buffer1", 3, 1, 3),
            LifetimeBoundBuffer("buffer2", 3, 2, 4),
            LifetimeBoundBuffer("buffer3", 3, 3, 4),
        ]
        self.verify_layout(buffers, [0, 3, 6, 3])

    def test_realloc_between_with_alignment(self):
        # Same reuse pattern as test_realloc_between, but with alignment padding applied.
        buffers = [
            LifetimeBoundBuffer("buffer0", 200, 0, 4),
            LifetimeBoundBuffer("buffer1", 100, 1, 3),
            LifetimeBoundBuffer("buffer2", 100, 2, 4),
            LifetimeBoundBuffer("buffer3", 100, 3, 4),
        ]
        self.verify_layout(buffers, [0, 256, 384, 256], LARGE_SIZE, ALIGNMENT)

    def test_inplace_allocation(self):
        # Test that adding inplace options allows for more efficient peak usage
        buffers = [
            LifetimeBoundBuffer("buffer0", LARGE_SIZE, 0, 4),
            LifetimeBoundBuffer(
                "buffer1", LARGE_SIZE, 3, 4, in_place_parents=["buffer0"]
            ),
        ]
        self.verify_layout(buffers, [0, 0], LARGE_SIZE, ALIGNMENT)

    def test_without_inplace_allocation(self):
        # Test that buffer gets evicted without in_place
        buffers = [
            LifetimeBoundBuffer("buffer0", LARGE_SIZE, 0, 4),
            LifetimeBoundBuffer("buffer1", LARGE_SIZE, 3, 4),
        ]
        self.verify_layout(buffers, [0, None], LARGE_SIZE, ALIGNMENT)

    def test_multiple_evictions_do_not_corrupt_allocation(self):
        # buffer0 fills the entire scratchpad; buffer1 and buffer2 are evicted.
        # buffer3 starts after buffer0 ends and should reclaim address 0.
        buffers = [
            LifetimeBoundBuffer("buffer0", SMALL_SIZE, 0, 2),
            LifetimeBoundBuffer("buffer1", SMALL_SIZE, 0, 2),
            LifetimeBoundBuffer("buffer2", SMALL_SIZE, 0, 2),
            LifetimeBoundBuffer("buffer3", SMALL_SIZE, 2, 3),
        ]
        self.verify_layout(buffers, [0, None, None, 0])

    def test_first_buffer_exceeds_limit_is_evicted(self):
        # A buffer whose size exceeds the scratchpad limit must be evicted even
        # when no other allocation is live (usage is empty, so address 0 would
        # otherwise be returned without the limit guard).
        buffers = [
            LifetimeBoundBuffer("buffer0", SMALL_SIZE + 1, 0, 2),
        ]
        self.verify_layout(buffers, [None], size=SMALL_SIZE)


if __name__ == "__main__":
    import unittest

    unittest.main()
