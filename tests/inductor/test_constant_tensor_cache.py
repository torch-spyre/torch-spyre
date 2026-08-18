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

"""Tests for spyre_constant_tensor module-level caching.

Tests verify:
1. Cache hit: Same (value, device, dtype) returns same tensor object
2. Cache miss: Different values/devices/dtypes create new tensors
3. Bounded size: Cache stays bounded at max size, regardless of underlying implementation
4. Cache clearing: Manual clear_constant_tensor_cache() for test isolation
5. CPU device: Not cached (uses standard torch.tensor path)
6. Thread safety: Concurrent access works correctly
"""

import threading
import unittest

import torch

from torch_spyre._inductor.op_spec import (
    _CONSTANT_TENSOR_CACHE,
    clear_constant_tensor_cache,
    spyre_constant_tensor,
)


class TestConstantTensorCache(unittest.TestCase):
    """Test module-level caching for spyre_constant_tensor."""

    def setUp(self):
        """Clear cache and reset state before each test."""
        clear_constant_tensor_cache()
        torch.compiler.reset()
        torch.manual_seed(0xBEEF)

    def tearDown(self):
        """Clean up cache and reset state after each test."""
        clear_constant_tensor_cache()
        torch.compiler.reset()

    def test_cache_hit_returns_same_object(self):
        """Two calls with same (value, device, dtype) return the same tensor object."""
        # Create two tensors with identical parameters
        t1 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)
        t2 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)

        # Should return the exact same object (cached)
        self.assertIs(t1, t2, "Cache hit should return same tensor object")
        self.assertEqual(t1.device.type, "spyre")
        self.assertEqual(t1.dtype, torch.float16)

    def test_cache_miss_creates_new_tensor(self):
        """Different values produce distinct tensors with separate cache entries."""
        t1 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)
        t2 = spyre_constant_tensor(2.0, torch.device("spyre"), torch.float16)

        # Should create different tensors
        self.assertIsNot(t1, t2, "Different values should create different tensors")
        self.assertEqual(t1.item(), 1.0)
        self.assertEqual(t2.item(), 2.0)

    def test_different_dtype_creates_new_tensor(self):
        """Different dtypes create separate cache entries."""
        t_fp16 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)
        t_fp32 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float32)

        # Different dtypes should not share cache entry
        self.assertIsNot(
            t_fp16, t_fp32, "Different dtypes should create different tensors"
        )
        self.assertEqual(t_fp16.dtype, torch.float16)
        self.assertEqual(t_fp32.dtype, torch.float32)

    def test_different_device_creates_new_tensor(self):
        """Different devices create separate cache entries."""
        t_spyre0 = spyre_constant_tensor(1.0, torch.device("spyre", 0), torch.float16)
        t_spyre = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)

        # Different device indices should not share cache entry
        self.assertIsNot(
            t_spyre0, t_spyre, "Different devices should create different tensors"
        )

    def test_cpu_device_not_cached(self):
        """CPU devices use standard torch.tensor path without caching."""
        t1 = spyre_constant_tensor(1.0, torch.device("cpu"), torch.float32)
        t2 = spyre_constant_tensor(1.0, torch.device("cpu"), torch.float32)

        # CPU tensors should NOT be cached (different objects)
        self.assertIsNot(t1, t2, "CPU tensors should not be cached")
        self.assertEqual(t1.device.type, "cpu")
        self.assertEqual(t2.device.type, "cpu")

    def test_dma_fill_numerical_correctness(self):
        """Verify DMA fill produces numerically correct values."""
        test_values = [0.0, 1.0, -1.0, 0.5, 3.14159, -2.71828]

        for val in test_values:
            t = spyre_constant_tensor(val, torch.device("spyre"), torch.float16)
            # fp16 has limited precision (~3 decimal digits)
            # Use relative tolerance instead of absolute places
            if val == 0.0:
                self.assertEqual(t.item(), 0.0, msg=f"DMA fill for {val} incorrect")
            else:
                relative_error = abs(t.item() - float(val)) / abs(float(val))
                self.assertLess(
                    relative_error,
                    0.001,  # 0.1% relative error
                    msg=f"DMA fill for {val} incorrect: got {t.item()}, expected {val}",
                )

    def test_manual_cache_clear(self):
        """Manual clear_constant_tensor_cache() works."""
        # Create cached tensors and HOLD references
        t1 = spyre_constant_tensor(1.0, torch.device("spyre"), torch.float16)
        t2 = spyre_constant_tensor(2.0, torch.device("spyre"), torch.float16)

        # Use values to ensure they're not optimized away
        _ = t1.item() + t2.item()

        # Cache should have entries
        self.assertGreater(len(_CONSTANT_TENSOR_CACHE), 0, "Cache should have entries")

        # Manual clear
        clear_constant_tensor_cache()
        self.assertEqual(
            len(_CONSTANT_TENSOR_CACHE), 0, "Cache should be empty after manual clear"
        )

    def test_cache_stays_bounded(self):
        """Cache stays bounded at max size when entries exceed the limit.

        This tests the desired property that the cache never grows beyond
        _CACHE_MAX_SIZE, regardless of the underlying implementation.
        """
        from torch_spyre._inductor.op_spec import _CACHE_MAX_SIZE

        # Clear cache to start fresh
        clear_constant_tensor_cache()

        # Create more entries than the max size
        num_entries = _CACHE_MAX_SIZE + 100
        for i in range(num_entries):
            spyre_constant_tensor(float(i), torch.device("spyre"), torch.float16)

        # Cache should stay bounded at or below max size
        self.assertLessEqual(
            len(_CONSTANT_TENSOR_CACHE),
            _CACHE_MAX_SIZE,
            f"Cache should stay bounded at {_CACHE_MAX_SIZE}, "
            f"got {len(_CONSTANT_TENSOR_CACHE)}",
        )

        # Cache should still function correctly after eviction
        test_value = 100.0  # Exactly representable in float16
        t = spyre_constant_tensor(test_value, torch.device("spyre"), torch.float16)
        self.assertEqual(t.item(), test_value)

        # Creating an already-cached value should still work
        t2 = spyre_constant_tensor(test_value, torch.device("spyre"), torch.float16)
        self.assertEqual(t2.item(), test_value)
        recreated = spyre_constant_tensor(0.0, torch.device("spyre"), torch.float16)
        self.assertEqual(recreated.item(), 0.0)

    def test_thread_safety(self):
        """Concurrent access from multiple threads works correctly."""
        errors = []
        results = []
        lock = threading.Lock()

        def create_constant(value):
            try:
                t = spyre_constant_tensor(value, torch.device("spyre"), torch.float16)
                with lock:
                    results.append((value, t.item()))
            except Exception as e:
                with lock:
                    errors.append(e)

        # Create multiple threads accessing cache concurrently
        threads = [
            threading.Thread(target=create_constant, args=(float(i),))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check no errors occurred
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

        # Check all values are correct
        for val, retrieved in results:
            self.assertAlmostEqual(val, retrieved, places=3)


if __name__ == "__main__":
    unittest.main()
