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

# Owner(s): ["module: stream"]

import contextlib
import ctypes
import gc
import os
import time

import pytest
import torch
import torch_spyre
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_spyre._C import copy_tensor


class TestSpyreStream(TestCase):
    """Test cases for Spyre stream functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        torch.manual_seed(0xAFFE)
        self.device = torch.device("spyre")

    def test_legacy_stream_creation_with_device(self):
        """Test creating stream using legacy torch.spyre.Stream API with device."""
        stream = torch.spyre.Stream(self.device)
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream, torch.spyre.Stream)

    def test_legacy_stream_creation_with_priority(self):
        """Test creating stream using legacy API with priority only."""
        stream = torch.spyre.Stream(priority=-1)
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream, torch.spyre.Stream)

    def test_legacy_stream_creation_with_device_and_priority(self):
        """Test creating stream using legacy API with both device and priority."""
        stream = torch.spyre.Stream(self.device, priority=-1)
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream, torch.spyre.Stream)

    def test_legacy_current_stream(self):
        """Test querying current stream using legacy API."""
        current_stream = torch.spyre.current_stream()
        self.assertIsNotNone(current_stream)
        self.assertIsInstance(current_stream, torch.spyre.Stream)

    def test_modern_stream_creation(self):
        """Test creating stream using modern torch.Stream API."""
        stream = torch.Stream(self.device)
        self.assertIsNotNone(stream)

    def test_modern_stream_creation_with_priority(self):
        """Test creating stream using modern API with high priority."""
        stream = torch.Stream(self.device, priority=-1)
        self.assertIsNotNone(stream)

    def test_modern_stream_context_manager(self):
        """Test using stream as context manager."""
        with torch.Stream(self.device):
            # Create tensor within stream context
            a = torch.randn(1, 32, device="spyre")
            self.assertEqual(a.device.type, "spyre")
            self.assertEqual(a.shape, (1, 32))

    def test_modern_current_stream(self):
        """Test querying current stream using modern accelerator API."""
        current_stream = torch.accelerator.current_stream()
        self.assertIsNotNone(current_stream)

    def test_stream_synchronize(self):
        """Test stream synchronization."""
        stream = torch.Stream(self.device)
        # Perform operation on stream
        with stream:
            _ = torch.randn(10, 10, device="spyre")
        # Synchronize the stream
        stream.synchronize()
        self.assertTrue(stream.query())

    def test_device_synchronize(self):
        """Test device-level synchronization."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device)

        with stream1:
            a = torch.randn(10, 10, device="spyre")  # noqa: F841

        with stream2:
            b = torch.randn(10, 10, device="spyre")  # noqa: F841

        # Synchronize all streams on the device
        torch.accelerator.synchronize(self.device)
        self.assertTrue(stream1.query())
        self.assertTrue(stream2.query())

    def test_global_synchronize(self):
        """Test global synchronization across all devices."""
        stream = torch.Stream(self.device)
        with stream:
            a = torch.randn(10, 10, device="spyre")  # noqa: F841

        # Synchronize all streams on all devices
        torch.accelerator.synchronize()
        self.assertTrue(stream.query())

    def test_multiple_streams(self):
        """Test creating and using multiple streams."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device, priority=-1)
        stream3 = torch.Stream(self.device, priority=0)

        self.assertIsNotNone(stream1)
        self.assertIsNotNone(stream2)
        self.assertIsNotNone(stream3)

    def test_stream_operations_isolation(self):
        """Test that operations on different streams are isolated."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device)

        with stream1:
            a = torch.randn(5, 5, device="spyre")
            result1 = a + a

        with stream2:
            b = torch.randn(5, 5, device="spyre")
            result2 = b + b

        # Synchronize both streams
        stream1.synchronize()
        stream2.synchronize()
        self.assertTrue(stream1.query())
        self.assertTrue(stream2.query())

        self.assertEqual(result1.shape, (5, 5))
        self.assertEqual(result2.shape, (5, 5))

    def test_nested_stream_contexts(self):
        """Test nested stream context managers."""
        stream1 = torch.Stream(self.device)
        stream2 = torch.Stream(self.device)

        with stream1:
            a = torch.randn(3, 3, device="spyre")
            with stream2:
                b = torch.randn(3, 3, device="spyre")
                c = a + b

        stream1.synchronize()
        stream2.synchronize()
        self.assertTrue(stream1.query())
        self.assertTrue(stream2.query())
        self.assertEqual(c.shape, (3, 3))

    def test_reentrant_same_stream_context(self):
        """Re-entering the *same* Stream instance must not clobber the
        state needed to restore the stream that was active before the
        outermost `with` block.

        Regression test: __enter__ used to save the previous stream into a
        single `_prev_stream` attribute, so nesting the same instance (e.g.
        via `with s: with s: ...`, or the aliasing `torch_spyre.stream(s)`
        helper) overwrote the outer value and left the wrong stream current
        after the outer block exited.
        """
        outer_stream = torch.spyre.current_stream()
        # s = torch.Stream(self.device)
        s = torch.spyre.Stream(self.device)

        with s:
            with s:
                torch.randn(3, 3, device="spyre")

            # The inner __exit__ must restore `s`, not the stream that was
            # current before the outer __enter__.
            self.assertEqual(torch.spyre.current_stream(), s)

        # use proper comparisonn
        self.assertEqual(torch.spyre.current_stream(), outer_stream)

    def test_stream_context_restores_on_exception(self):
        """The displaced stream must be restored even if the `with` body
        raises."""
        outer_stream = torch.spyre.current_stream()
        # s = torch.Stream(self.device)
        s = torch.spyre.Stream(self.device)

        with self.assertRaises(ValueError):
            with s:
                raise ValueError("boom")

        self.assertEqual(torch.spyre.current_stream(), outer_stream)

    def test_stream_priority_levels(self):
        """Test creating streams with different priority levels."""
        high_priority = torch.Stream(self.device, priority=-1)
        normal_priority = torch.Stream(self.device, priority=0)
        low_priority = torch.Stream(self.device, priority=1)

        self.assertIsNotNone(high_priority)
        self.assertIsNotNone(normal_priority)
        self.assertIsNotNone(low_priority)

    def test_stream_with_tensor_operations(self):
        """Test various tensor operations within stream context."""
        stream = torch.Stream(self.device)

        with stream:
            # Create tensors
            a = torch.randn(10, 64, dtype=torch.float16, device="spyre")
            b = torch.randn(10, 64, dtype=torch.float16, device="spyre")

            # Perform operations
            c = a + b
            d = torch.add(a, b)

            self.assertEqual(c.shape, (10, 64))
            self.assertEqual(d.shape, (10, 64))

        stream.synchronize()
        self.assertTrue(stream.query())

    def test_default_stream(self):
        """Test that default stream exists and is accessible."""
        default_stream = torch.spyre.current_stream()
        self.assertIsNotNone(default_stream)

        # Operations without explicit stream should use default stream
        a = torch.randn(5, 5, device="spyre")
        self.assertEqual(a.device.type, "spyre")

    def test_host_compute_stream_round_robin(self):
        """Host compute streams are handed out round-robin with no duplicates
        within a cycle, whatever the configured count is."""
        first = torch_spyre._C.host_compute_stream(self.device).id()
        seen = [first]
        # Walk one full cycle: keep pulling streams until we wrap back to the
        # first one. Every stream seen before wrapping must be distinct.
        while True:
            sid = torch_spyre._C.host_compute_stream(self.device).id()
            if sid == first:
                break
            self.assertNotIn(sid, seen)
            seen.append(sid)

    def test_host_compute_stream_synchronize(self):
        """A host compute stream can be synchronized without errors."""
        stream = torch_spyre._C.host_compute_stream(self.device)
        stream.synchronize()

    def test_host_compute_stream_by_id(self):
        """host_compute_stream_by_id returns the stream with the requested ID."""
        expected_id = torch_spyre._C.host_compute_stream(self.device).id()
        stream = torch_spyre._C.host_compute_stream_by_id(expected_id, self.device)
        self.assertEqual(stream.id(), expected_id)

    def test_stream_query_after_context(self):
        """Test querying stream after exiting context."""
        stream = torch.Stream(self.device)

        with stream:
            current = torch.accelerator.current_stream()
            # Current stream should be our stream within context
            self.assertIsNotNone(current)

        # After exiting context, should return to default stream
        current_after = torch.accelerator.current_stream()
        self.assertIsNotNone(current_after)

    def test_stream_reuse(self):
        """Test reusing the same stream multiple times."""
        stream = torch.Stream(self.device)

        # First use
        with stream:
            a = torch.randn(5, 5, device="spyre")
        stream.synchronize()
        self.assertTrue(stream.query())

        # Second use
        with stream:
            b = torch.randn(5, 5, device="spyre")
        stream.synchronize()
        self.assertTrue(stream.query())

        self.assertEqual(a.shape, (5, 5))
        self.assertEqual(b.shape, (5, 5))


class TestStreamCompatibility(TestCase):
    """Test compatibility between legacy and modern stream APIs."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        torch.manual_seed(0xAFFE)
        self.device = torch.device("spyre")

    def test_legacy_and_modern_interop(self):
        """Test that legacy and modern APIs can work together."""
        # Create stream using legacy API
        legacy_stream = torch.spyre.Stream(self.device)

        # Create stream using modern API
        modern_stream = torch.Stream(self.device)

        # Both should be valid
        self.assertIsNotNone(legacy_stream)
        self.assertIsNotNone(modern_stream)

    def test_current_stream_consistency(self):
        """Test that current_stream returns consistent results."""
        # Query using legacy API
        legacy_current = torch.spyre.current_stream()

        # Query using modern API
        modern_current = torch.accelerator.current_stream()

        # Both should return valid streams
        self.assertIsNotNone(legacy_current)
        self.assertIsNotNone(modern_current)


_RECYCLE_ENV = "TORCH_SPYRE_STREAM_RECYCLE_OPS"

# Retained host bytes per op above which a stream counts as leaking. Flex keeps
# ~1.4 KB per non-blocking H2D today.
_MAX_RETAINED_BYTES_PER_OP = 64

_libc = ctypes.CDLL("libc.so.6")


def _rss_bytes():
    gc.collect()
    _libc.malloc_trim(0)
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _wait_idle(stream):
    """Waits for stream without synchronizing it (no recycle point)."""
    while not stream.query():
        time.sleep(0.0005)


@contextlib.contextmanager
def _recycle_ops(n):
    """Sets the recycle threshold; a synchronize refreshes torch-spyre's copy."""
    saved = os.environ.get(_RECYCLE_ENV)
    os.environ[_RECYCLE_ENV] = str(n)
    torch.spyre.synchronize()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_RECYCLE_ENV, None)
        else:
            os.environ[_RECYCLE_ENV] = saved
        torch.spyre.synchronize()


class TestFlexStreamRecycle(TestCase):
    """Recycling of the flex stream behind each c10 stream id (#6569).

    Flex keeps per-op host state for a stream until the stream is destroyed,
    so torch-spyre swaps a fresh flex stream in every
    TORCH_SPYRE_STREAM_RECYCLE_OPS submitted ops.
    """

    def setUp(self):
        super().setUp()
        self.device = torch.device("spyre")

    def _copies(self, n):
        srcs = [torch.arange(64, dtype=torch.int32) + 1000 * i for i in range(n)]
        dsts = [torch.empty(64, dtype=torch.int32, device=self.device) for _ in srcs]
        return srcs, dsts

    def _assert_copied(self, srcs, dsts):
        for src, dst in zip(srcs, dsts):
            self.assertEqual(src.tolist(), dst.cpu().tolist())

    def _retained_bytes_per_unsynchronized_copy(self, threshold, ops):
        """Host RSS growth per non-blocking H2D on a never-synchronized stream.

        The stream is only polled until idle every 1024 copies, as a
        non-driver tensor-parallel rank never synchronizes its streams.
        """
        src = torch.arange(64, dtype=torch.int32)
        dst = src.to(self.device)
        stream = torch.Stream(self.device)

        def run(n):
            with stream:
                for i in range(n):
                    copy_tensor(src, dst, non_blocking=True)
                    if i % 1024 == 1023:
                        _wait_idle(stream)
            _wait_idle(stream)

        with _recycle_ops(threshold):
            run(ops // 4)
            before = _rss_bytes()
            run(ops)
            per_op = (_rss_bytes() - before) / ops
        # Free what an unrecycled stream retained before the next test.
        with _recycle_ops(1):
            stream.synchronize()
        return per_op

    def test_recycle_preserves_ordering(self):
        """Copies that straddle a recycle observe every earlier copy.

        A broken swap (wrong handle, dropped ops, or a destroyed handle still
        resolved) shows up as wrong values.
        """
        stream = torch.Stream(self.device)
        with _recycle_ops(4), stream:
            for i in range(64):
                expect = torch.arange(128, dtype=torch.int32) + i
                dst = expect.to(self.device)  # blocking H2D, syncs per copy
                self.assertTrue(torch.equal(expect, dst.cpu()))

            # Non-blocking bursts: many ops per synchronize, with the
            # recycle threshold crossed mid-burst.
            src = torch.arange(64, dtype=torch.int32)
            dst = src.to(self.device)
            for burst in range(8):
                for j in range(12):
                    copy_tensor(src + burst * 100 + j, dst, non_blocking=True)
                stream.synchronize()
                self.assertEqual((src + burst * 100 + 11).tolist(), dst.cpu().tolist())

    def test_idle_pool_recycles_without_synchronize(self):
        """A stream nobody synchronizes is recycled once the pool goes idle.

        S_prep under SPYRE_HAZARD_TRACKER, and every stream on a non-driver
        tensor-parallel rank, never reach a synchronization point.
        """
        srcs, dsts = self._copies(48)
        stream = torch.Stream(self.device)
        with _recycle_ops(8):
            before = torch_spyre._C._stream_recycle_stats()
            with stream:
                for i, (src, dst) in enumerate(zip(srcs, dsts)):
                    copy_tensor(src, dst, non_blocking=True)
                    if i % 4 == 3:
                        _wait_idle(stream)
                        time.sleep(0.002)  # past the idle-probe interval
            after = torch_spyre._C._stream_recycle_stats()
        self.assertGreater(after["when_idle"], before["when_idle"])
        self.assertEqual(after["at_sync"], before["at_sync"])
        self._assert_copied(srcs, dsts)

    def test_busy_pool_recycle_is_forced_without_synchronize(self):
        """Back-to-back submissions hit the forced limit and still all land.

        With threshold 2 a stream reaches the forced limit (twice the
        threshold) within a few submissions, well inside one idle-probe
        interval, so the pool is drained and swapped from the submission path.
        """
        srcs, dsts = self._copies(64)
        stream = torch.Stream(self.device)
        with _recycle_ops(2):
            before = torch_spyre._C._stream_recycle_stats()
            with stream:
                for src, dst in zip(srcs, dsts):
                    copy_tensor(src, dst, non_blocking=True)
            after = torch_spyre._C._stream_recycle_stats()
        self.assertGreater(after["forced"], before["forced"])
        self._assert_copied(srcs, dsts)

    def test_recycling_bounds_host_memory_of_unsynchronized_stream(self):
        """A never-synchronized stream stops accumulating flex host state.

        A low threshold keeps the saw-tooth between recycles (at most twice
        the threshold in retained ops) well under the per-op bound.
        """
        per_op = self._retained_bytes_per_unsynchronized_copy(256, 80_000)
        self.assertLess(per_op, _MAX_RETAINED_BYTES_PER_OP)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="flex keeps per-op host state for every DMA until the stream is "
        "destroyed (torch-spyre#6569). An XPASS means flex frees it: drop "
        "TORCH_SPYRE_STREAM_RECYCLE_OPS and this xfail.",
    )
    def test_flex_frees_dma_state_without_recycling(self):
        per_op = self._retained_bytes_per_unsynchronized_copy(0, 80_000)
        self.assertLess(per_op, _MAX_RETAINED_BYTES_PER_OP)


if __name__ == "__main__":
    run_tests()
