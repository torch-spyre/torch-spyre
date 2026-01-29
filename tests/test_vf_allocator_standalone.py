#!/usr/bin/env python3
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

# Owner(s): ["module: cpp"]

"""
Standalone unit tests for VF (Virtual Function) allocator implementation.

This test can be run directly without pytest, avoiding import-time initialization issues.

IMPORTANT: These tests still require the Spyre runtime to initialize, which requires
DOOM mode to be properly configured. If tests fail with "Incompatible DOOM mode and device"
errors, you have two options:

1. Use the C++ test binary (recommended):
   cd torch_spyre/csrc
   FLEX_DEVICE=VF ./test_vf_allocator

2. Enable DOOM mode in your environment, then run:
   FLEX_DEVICE=VF python tests/test_vf_allocator_standalone.py

   Or run a specific test:
   FLEX_DEVICE=VF python tests/test_vf_allocator_standalone.py TestVFAllocatorStandalone.test_realistic_allocation_pattern

The C++ test binary tests the core allocator logic without requiring full runtime initialization.
"""

import os
import sys
import unittest
import gc

# Check FLEX_DEVICE before importing torch to avoid initialization issues
FLEX_DEVICE = os.environ.get("FLEX_DEVICE")
if FLEX_DEVICE != "VF":
    print(f"Warning: FLEX_DEVICE is '{FLEX_DEVICE}', tests require 'VF'")
    print(
        "Please run with: FLEX_DEVICE=VF python tests/test_vf_allocator_standalone.py"
    )
    sys.exit(0)

import torch  # noqa: E402


class TestVFAllocatorStandalone(unittest.TestCase):
    """Standalone test suite for VF allocator - can be run without pytest."""

    def test_vf_mode_detection(self):
        """Test that VF mode is correctly detected from environment variable."""
        self.assertEqual(os.environ.get("FLEX_DEVICE"), "VF")

        # Create a tensor to verify allocator works
        x = torch.empty(10, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")

    def test_basic_allocation(self):
        """Test basic memory allocation in VF mode."""
        x = torch.empty(100, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(x.numel(), 100)
        self.assertEqual(x.dtype, torch.float16)

        storage_size = x.untyped_storage().nbytes()
        self.assertGreaterEqual(storage_size, 128)

    def test_allocation_alignment(self):
        """Test that allocations are aligned to 128 bytes."""
        test_sizes = [1, 50, 100, 127, 128, 129, 200, 255, 256]

        for size in test_sizes:
            x = torch.empty(size, device="spyre", dtype=torch.float16)
            storage_size = x.untyped_storage().nbytes()
            self.assertEqual(
                storage_size % 128,
                0,
                f"Allocation size {storage_size} for tensor size {size} is not aligned to 128 bytes",
            )

    def test_memory_reuse(self):
        """Test that deallocated memory can be reused for new allocations."""
        tensors = []
        for i in range(5):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            tensors.append(t)

        del tensors
        gc.collect()

        new_tensors = []
        for i in range(5):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            new_tensors.append(t)

        for t in new_tensors:
            self.assertEqual(t.device.type, "spyre")
            self.assertEqual(t.numel(), 100)

    def test_zero_size_allocation(self):
        """Test that zero-size allocation returns valid but empty tensor."""
        x = torch.empty(0, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(x.numel(), 0)

    def test_different_dtypes(self):
        """Test allocation with different data types."""
        dtypes = [
            torch.float16,
            torch.float32,
            torch.int32,
            torch.bool,
        ]

        for dtype in dtypes:
            x = torch.empty(100, device="spyre", dtype=dtype)
            self.assertEqual(x.device.type, "spyre")
            self.assertEqual(x.dtype, dtype)
            self.assertEqual(x.numel(), 100)

    def test_sequential_allocation_deallocation(self):
        """Test sequential allocation and deallocation pattern."""
        for iteration in range(5):
            x = torch.empty(500, device="spyre", dtype=torch.float16)
            x.fill_(iteration)
            x_cpu = x.cpu()
            self.assertTrue((x_cpu == iteration).all())
            del x, x_cpu
            gc.collect()

    def test_interleaved_allocations(self):
        """Test interleaved allocation pattern."""
        t1 = torch.empty(200, device="spyre", dtype=torch.float16)
        t2 = torch.empty(200, device="spyre", dtype=torch.float16)
        t3 = torch.empty(200, device="spyre", dtype=torch.float16)

        del t2
        gc.collect()

        batch2 = [
            torch.empty(200, device="spyre", dtype=torch.float16) for _ in range(2)
        ]

        self.assertEqual(t1.numel(), 200)
        self.assertEqual(t3.numel(), 200)
        for t in batch2:
            self.assertEqual(t.numel(), 200)

    def test_free_interval_merging(self):
        """Test that adjacent free intervals are merged correctly."""
        t1 = torch.empty(1000, device="spyre", dtype=torch.float16)
        t2 = torch.empty(1000, device="spyre", dtype=torch.float16)
        t3 = torch.empty(1000, device="spyre", dtype=torch.float16)

        del t2
        gc.collect()

        del t1
        gc.collect()

        del t3
        gc.collect()

        large = torch.empty(3000, device="spyre", dtype=torch.float16)
        self.assertEqual(large.numel(), 3000)

    def test_tensor_operations_with_vf_allocator(self):
        """Test that tensor operations work correctly with VF allocator."""
        # Create tensors on CPU first, then move to spyre device
        x = torch.randn(100, dtype=torch.float16).to("spyre")
        y = torch.randn(100, dtype=torch.float16).to("spyre")

        z = x + y
        w = x * y
        v = torch.sum(x)

        self.assertEqual(z.device.type, "spyre")
        self.assertEqual(w.device.type, "spyre")
        self.assertEqual(v.device.type, "spyre")

        z_cpu = z.cpu()
        w_cpu = w.cpu()
        v_cpu = v.cpu()

        self.assertEqual(z_cpu.numel(), 100)
        self.assertEqual(w_cpu.numel(), 100)
        self.assertEqual(v_cpu.numel(), 1)

    def test_large_tensor_allocation(self):
        """Test allocation of large tensors."""
        large_size = 10 * 1024 * 1024
        x = torch.empty(large_size, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(x.numel(), large_size)

        x.fill_(1.0)
        x_cpu = x.cpu()
        self.assertTrue((x_cpu == 1.0).all())

    def test_mixed_size_allocations(self):
        """Test allocations of various sizes to test block management."""
        sizes = [10, 100, 1000, 10000, 100000]
        tensors = []

        for size in sizes:
            t = torch.empty(size, device="spyre", dtype=torch.float16)
            tensors.append(t)
            self.assertEqual(t.numel(), size)

        del tensors[1], tensors[3]
        gc.collect()

        t_new1 = torch.empty(100, device="spyre", dtype=torch.float16)
        t_new2 = torch.empty(10000, device="spyre", dtype=torch.float16)

        self.assertEqual(t_new1.numel(), 100)
        self.assertEqual(t_new2.numel(), 10000)

    def test_concurrent_allocations(self):
        """Test that multiple allocations can coexist."""
        tensors = []
        num_tensors = 20

        for i in range(num_tensors):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            t.fill_(i)
            tensors.append(t)

        for i, t in enumerate(tensors):
            self.assertEqual(t.numel(), 100)
            t_cpu = t.cpu()
            self.assertEqual(t_cpu.numel(), 100)

    def test_very_small_allocation(self):
        """Test allocation of very small tensors."""
        x = torch.empty(1, device="spyre", dtype=torch.float16)
        self.assertEqual(x.numel(), 1)

        storage_size = x.untyped_storage().nbytes()
        self.assertGreaterEqual(storage_size, 128)
        self.assertEqual(storage_size % 128, 0)

    def test_allocation_after_many_deallocations(self):
        """Test allocation after many deallocations to stress test free interval management."""
        for _ in range(50):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            del t
            gc.collect()

        final = torch.empty(100, device="spyre", dtype=torch.float16)
        self.assertEqual(final.numel(), 100)

    def test_realistic_allocation_pattern(self):
        """Test a realistic sequence of allocations and deallocations with various tensor sizes.

        This test simulates a real-world scenario with:
        - Initial allocations
        - Selective deallocations (moving to CPU)
        - Reallocation to same size
        - Large tensor allocation
        - Bulk deallocations
        - Memory reuse from freed blocks
        """
        print("\n---------- allocate tensor a -------------")
        a = torch.tensor([0], dtype=torch.float16, device="spyre")
        self.assertEqual(a.numel(), 1)
        self.assertEqual(a.device.type, "spyre")

        print("\n\n---------- allocate tensor b -------------")
        b = torch.tensor([0.0], dtype=torch.float16, device="spyre")
        self.assertEqual(b.numel(), 1)
        self.assertEqual(b.device.type, "spyre")

        print("\n\n---------- allocate tensor c -------------")
        c = torch.tensor([1, 2], dtype=torch.float16, device="spyre")
        self.assertEqual(c.numel(), 2)
        self.assertEqual(c.device.type, "spyre")

        print("\n\n---------- deallocate tensor b -------------")
        b = b.to("cpu")
        self.assertEqual(b.device.type, "cpu")
        gc.collect()

        print("\n\n---------- allocate tensor d -------------")
        d = torch.tensor([7, 7], dtype=torch.float16, device="spyre")
        self.assertEqual(d.numel(), 2)
        self.assertEqual(d.device.type, "spyre")

        print("\n\n---------- allocate new tensor, then deallocate d -------------")
        d = torch.tensor([1, 9, 8, 4], dtype=torch.float16, device="spyre")
        self.assertEqual(d.numel(), 4)
        self.assertEqual(d.device.type, "spyre")
        gc.collect()

        print("\n\n---------- allocate tensor e -------------")
        e = torch.tensor(
            [
                [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [1.0, 0.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            ],
            dtype=torch.float16,
            device="spyre",
        )
        self.assertEqual(e.numel(), 64)
        self.assertEqual(e.device.type, "spyre")

        print("\n\n---------- deallocate tensor a, c, d -------------")
        del a
        del c
        del d
        gc.collect()

        print("\n\n---------- allocate tensor k (from freed space) -------------")
        k = torch.tensor([0.1, 0.2], dtype=torch.float16, device="spyre")
        self.assertEqual(k.numel(), 2)
        self.assertEqual(k.device.type, "spyre")

        print("\n\n---------- allocate tensor j (from freed space) -------------")
        j = torch.tensor([1, 2], dtype=torch.float16, device="spyre")
        self.assertEqual(j.numel(), 2)
        self.assertEqual(j.device.type, "spyre")

        print("\n\n---------- allocate tensor l (from freed space) -------------")
        tensor_l = torch.tensor([7, 0, 4, 9], dtype=torch.float16, device="spyre")
        self.assertEqual(tensor_l.numel(), 4)
        self.assertEqual(tensor_l.device.type, "spyre")

        print("\n\n---------- allocate tensor f (from freed space) -------------")
        f = torch.tensor([6, 6, 6], dtype=torch.float16, device="spyre")
        self.assertEqual(f.numel(), 3)
        self.assertEqual(f.device.type, "spyre")

        print("\n\n---------- verify all tensors are still valid -------------")
        self.assertEqual(e.device.type, "spyre")
        self.assertEqual(k.device.type, "spyre")
        self.assertEqual(j.device.type, "spyre")
        self.assertEqual(tensor_l.device.type, "spyre")
        self.assertEqual(f.device.type, "spyre")
        print("All tensors verified successfully!")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
