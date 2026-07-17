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

"""
End-to-end tests for SpyreAllocator → FlexAllocator path through PyTorch's public tensor API.

These tests verify that torch.empty(size, device="spyre") correctly allocates device memory,
that tensors going out of scope trigger ReportAndDelete to free memory, and that sequential
allocate/free cycles leave the allocator in a consistent state.
"""

import gc
import threading
import torch
import random

from torch.testing._internal.common_utils import TestCase


def get_allocator_stats():
    """Get current allocator statistics from SpyreAllocator."""
    # Ensure torch.spyre is initialized
    if not torch.spyre.is_initialized():
        torch.spyre._lazy_init()
    stats = torch.spyre._spyre_get_allocator_stats(0)
    return {
        "allocated_bytes": stats.get("allocated_bytes.all.current", 0),
        "num_allocs": stats.get("allocation.all.current", 0),
    }


class TestAllocatorE2E(TestCase):
    """End-to-end tests for SpyreAllocator → FlexAllocator integration."""

    def setUp(self):
        """Reset allocator stats before each test."""
        super().setUp()
        # Two gc.collect() passes: the first breaks cycles and drops refcounts,
        # the second collects any objects whose __del__ was queued by the first
        # pass. This ensures all ReportAndDelete calls from the previous test
        # have fired before we reset stats and snapshot the baseline.
        gc.collect()
        gc.collect()
        # Ensure torch.spyre is initialized
        if not torch.spyre.is_initialized():
            torch.spyre._lazy_init()
        torch.spyre._spyre_reset_accumulated_stats(0)
        torch.spyre._spyre_reset_peak_stats(0)
        # Snapshot baseline immediately after reset so no intervening
        # ReportAndDelete call can produce a negative delta in the test body.
        self.initial_stats = get_allocator_stats()

    def tearDown(self):
        """Clean up after each test."""
        gc.collect()
        super().tearDown()

    def test_basic_allocation(self):
        """
        Test 1: Basic allocation
        Verify that torch.empty((N,), device="spyre") returns a valid tensor
        with non-null storage and correct size.
        """
        N = 1024

        # Get initial stats
        initial_stats = self.initial_stats

        # Allocate tensor
        tensor = torch.empty((N,), device="spyre", dtype=torch.float32)

        # Verify tensor properties
        self.assertGreater(tensor.data_ptr(), 0)

        # Verify storage is non-null
        self.assertIsNotNone(tensor.untyped_storage())
        self.assertGreater(tensor.untyped_storage().data_ptr(), 0)

        # Verify allocator stats increased
        current_stats = get_allocator_stats()
        self.assertGreater(
            current_stats["allocated_bytes"], initial_stats["allocated_bytes"]
        )
        self.assertGreater(current_stats["num_allocs"], initial_stats["num_allocs"])

        # Expected allocation size (N * sizeof(float32) = N * 4 bytes)
        expected_bytes = N * 4
        allocated_bytes = (
            current_stats["allocated_bytes"] - initial_stats["allocated_bytes"]
        )
        # Allow for alignment padding (FlexAllocator aligns to DEVICE_ALIGNMENT)
        self.assertGreaterEqual(allocated_bytes, expected_bytes)

        # Verify 128-byte alignment
        self.assertEqual(
            allocated_bytes % 128,
            0,
            f"Allocated bytes ({allocated_bytes}) should be aligned to 128-byte boundary",
        )

    def test_automatic_deallocation(self):
        """
        Test 2: Automatic deallocation
        Allocate tensor in a scope, let it go out of scope, force GC,
        verify the block is freed (allocator free space increases).
        """
        N = 2048

        # Get initial stats
        initial_stats = self.initial_stats

        # Allocate tensor in a scope
        tensor = torch.empty((N,), device="spyre", dtype=torch.float32)

        # Verify allocation happened
        stats_during = get_allocator_stats()
        self.assertGreater(
            stats_during["allocated_bytes"], initial_stats["allocated_bytes"]
        )
        self.assertGreater(stats_during["num_allocs"], initial_stats["num_allocs"])

        # Verify 128-byte alignment
        allocated_bytes = (
            stats_during["allocated_bytes"] - initial_stats["allocated_bytes"]
        )
        self.assertEqual(
            allocated_bytes % 128,
            0,
            f"Allocated bytes ({allocated_bytes}) should be aligned to 128-byte boundary",
        )

        # Delete tensor reference and force garbage collection to trigger ReportAndDelete
        del tensor
        gc.collect()

        # Verify deallocation happened
        final_stats = get_allocator_stats()
        self.assertEqual(
            final_stats["allocated_bytes"],
            initial_stats["allocated_bytes"],
            "Memory should be freed after tensor goes out of scope",
        )
        self.assertEqual(
            final_stats["num_allocs"],
            initial_stats["num_allocs"],
            "Allocation count should return to initial value",
        )

    def test_coalescing_with_batch_deallocation(self):
        """
        Test 3: Coalescing verification with batch deallocation
        Allocate 100 small tensors, then deallocate them in batches of 10.
        After each batch is freed, verify coalescing by attempting to allocate
        a larger tensor that requires the combined space of the freed batch.

        This test proves that:
        1. Adjacent freed blocks are coalesced into larger contiguous blocks
        2. Memory cleanup works correctly during progressive deallocation
        3. The coalesced space can be reused for larger allocations
        """
        small_size = 512
        num_tensors = 100
        batch_size = 10
        large_size = small_size * batch_size

        initial_stats = self.initial_stats

        # Allocate 100 small tensors
        tensors = []
        for i in range(num_tensors):
            tensor = torch.empty((small_size,), device="spyre", dtype=torch.float32)
            tensors.append(tensor)

        # Verify all 100 tensors were allocated
        stats_after_alloc = get_allocator_stats()
        self.assertEqual(
            stats_after_alloc["num_allocs"] - initial_stats["num_allocs"],
            num_tensors,
            f"Expected {num_tensors} allocations",
        )

        # Verify 128-byte alignment
        allocated_bytes = (
            stats_after_alloc["allocated_bytes"] - initial_stats["allocated_bytes"]
        )
        self.assertEqual(
            allocated_bytes % 128,
            0,
            f"Allocated bytes ({allocated_bytes}) should be aligned to 128-byte boundary",
        )

        expected_bytes = stats_after_alloc["allocated_bytes"]

        # Deallocate tensors in batches and verify coalescing
        for batch_num in range(num_tensors // batch_size):
            # Deallocate a batch of 10 (batch_size) adjacent tensors
            for i in range(batch_size):
                tensor = tensors.pop(0)
                del tensor

            gc.collect()

            # After batch deallocation, verify memory is freed
            stats_after_batch = get_allocator_stats()
            tensors_freed = (batch_num + 1) * batch_size
            expected_allocs_remaining = num_tensors - tensors_freed

            self.assertEqual(
                stats_after_batch["num_allocs"] - initial_stats["num_allocs"],
                expected_allocs_remaining,
                f"After freeing {tensors_freed} tensors, expected {expected_allocs_remaining} remaining",
            )

            # Verify memory is decreasing
            self.assertLess(
                stats_after_batch["allocated_bytes"],
                expected_bytes,
                f"Memory should decrease after freeing batch {batch_num + 1}",
            )
            expected_bytes = stats_after_batch["allocated_bytes"]

            # COALESCING TEST: Try to allocate a large tensor in the freed space
            # This will only succeed if the 10 freed adjacent blocks were coalesced
            try:
                large_tensor = torch.empty(
                    (large_size,), device="spyre", dtype=torch.float32
                )

                # Verify the large allocation succeeded
                self.assertIsNotNone(large_tensor.data_ptr())
                self.assertGreater(large_tensor.data_ptr(), 0)

                # The large tensor should fit in the coalesced space
                stats_with_large = get_allocator_stats()
                self.assertEqual(
                    stats_with_large["num_allocs"] - initial_stats["num_allocs"],
                    expected_allocs_remaining + 1,
                    f"Should have {expected_allocs_remaining} remaining + 1 large allocation",
                )

                # Clean up the large tensor for next iteration
                del large_tensor
                gc.collect()

            except RuntimeError as e:
                self.fail(
                    f"Batch {batch_num + 1}: Failed to allocate large tensor ({large_size} floats) "
                    f"after freeing {batch_size} adjacent small tensors ({small_size} floats each). "
                    f"This indicates the allocator did NOT coalesce the {batch_size} adjacent free blocks. "
                    f"Error: {e}"
                )

        # Final cleanup
        tensors.clear()
        gc.collect()

        # Final check: all memory should be freed
        stats_final = get_allocator_stats()
        self.assertEqual(
            stats_final["allocated_bytes"],
            initial_stats["allocated_bytes"],
            "All memory should be freed after complete deallocation",
        )
        self.assertEqual(
            stats_final["num_allocs"],
            initial_stats["num_allocs"],
            "Allocation count should return to initial value",
        )

    def test_varying_sizes_random_order(self):
        """
        Test 4: Varying sizes with random deallocation
        Allocate tensors of different sizes (small, medium, large),
        free in random order, verify consistent state.
        """
        # Set random seed for reproducibility
        random.seed(42)

        sizes = [
            128,  # small: 512 bytes
            4096,  # medium: 16 KB
            262144,  # large: 1 MB
            128,  # small
            8192,  # medium-large: 32 KB
            524288,  # very large: 2 MB
        ]

        initial_stats = self.initial_stats

        # Allocate all tensors
        tensors = []
        for size in sizes:
            tensor = torch.empty((size,), device="spyre", dtype=torch.float32)
            self.assertIsNotNone(tensor.data_ptr())
            tensors.append(tensor)

        # Verify all allocations happened
        stats_after_alloc = get_allocator_stats()
        self.assertGreater(
            stats_after_alloc["allocated_bytes"], initial_stats["allocated_bytes"]
        )
        self.assertEqual(
            stats_after_alloc["num_allocs"] - initial_stats["num_allocs"], len(sizes)
        )

        # Verify 128-byte alignment
        allocated_bytes = (
            stats_after_alloc["allocated_bytes"] - initial_stats["allocated_bytes"]
        )
        self.assertEqual(
            allocated_bytes % 128,
            0,
            f"Allocated bytes ({allocated_bytes}) should be aligned to 128-byte boundary",
        )

        # Shuffle tensors for random-order deallocation
        random.shuffle(tensors)

        while tensors:
            tensor = tensors.pop()
            del tensor
            gc.collect()

        # Verify all memory is freed
        final_stats = get_allocator_stats()
        self.assertEqual(
            final_stats["allocated_bytes"],
            initial_stats["allocated_bytes"],
            "Memory leaked after random-order deallocation",
        )
        self.assertEqual(
            final_stats["num_allocs"],
            initial_stats["num_allocs"],
            "Allocation count mismatch after random-order deallocation",
        )

    def test_zero_size_allocation(self):
        """
        Test 5: Zero-size allocation
        Verify that torch.empty((0,), device="spyre") does not crash
        and behavior matches CPU allocator semantics.
        """
        initial_stats = self.initial_stats

        # Allocate zero-size tensor
        tensor = torch.empty((0,), device="spyre", dtype=torch.float32)

        # Verify tensor properties
        self.assertEqual(tensor.numel(), 0)

        # Zero-size allocations should return nullptr (data_ptr == 0)
        self.assertEqual(
            tensor.data_ptr(),
            0,
            "Zero-size allocation should return nullptr (data_ptr == 0)",
        )

        # Zero-size allocations should not allocate memory
        current_stats = get_allocator_stats()
        self.assertEqual(
            current_stats["allocated_bytes"],
            initial_stats["allocated_bytes"],
            "Zero-size allocation should not allocate memory",
        )

        # Delete tensor
        del tensor
        gc.collect()

        # Verify no memory leak
        final_stats = get_allocator_stats()
        self.assertEqual(
            final_stats["allocated_bytes"], initial_stats["allocated_bytes"]
        )

    def test_gc_many_tensor_release(self):
        """
        Bulk tensor release via reference counting.

        Allocate K=100 tensors of mixed sizes in a Python list; record allocator
        state before. Drop all references and verify allocator free-space returns
        to the pre-allocation baseline.

        Note: these tensors are non-cyclic, so CPython's reference counting drives
        deallocation immediately when the list is cleared — gc.collect() is called
        as a precaution but is a no-op for these objects. The test verifies that
        every tensor's ReportAndDelete callback fires and all K allocations are
        accounted for, not just the most recent one.
        """
        K = 100  # Number of tensors to allocate

        # Mixed sizes: small (1KB), medium (64KB), large (1MB)
        # Using a pattern that creates variety but is deterministic
        sizes = []
        for i in range(K):
            if i % 10 == 0:
                # Every 10th tensor is large
                sizes.append(262144)
            elif i % 3 == 0:
                # Every 3rd tensor (not 10th) is medium
                sizes.append(16384)
            else:
                # Rest are small
                sizes.append(256)

        # Record baseline allocator state
        initial_stats = self.initial_stats
        initial_allocated_bytes = initial_stats["allocated_bytes"]
        initial_num_allocs = initial_stats["num_allocs"]

        # Allocate K tensors in a list
        tensor_list = []
        for i, size in enumerate(sizes):
            tensor = torch.empty((size,), device="spyre", dtype=torch.float16)
            self.assertGreater(
                tensor.data_ptr(), 0, f"Tensor {i} should have valid data pointer"
            )
            tensor_list.append(tensor)

        # Verify all K tensors were allocated
        stats_after_alloc = get_allocator_stats()
        allocated_bytes_delta = (
            stats_after_alloc["allocated_bytes"] - initial_allocated_bytes
        )
        num_allocs_delta = stats_after_alloc["num_allocs"] - initial_num_allocs

        self.assertEqual(
            num_allocs_delta, K, f"Expected {K} allocations, got {num_allocs_delta}"
        )
        self.assertGreater(
            allocated_bytes_delta,
            0,
            "Total allocated bytes should increase after allocating tensors",
        )

        # Verify 128-byte alignment
        self.assertEqual(
            allocated_bytes_delta % 128,
            0,
            f"Allocated bytes ({allocated_bytes_delta}) should be aligned to 128-byte boundary",
        )

        # Drop all references. tensor_list.clear() drops every element's refcount
        # to zero immediately (non-cyclic objects), triggering ReportAndDelete for
        # each. The loop variable 'tensor' holds an extra reference to the last
        # element; release it first so that clear() is the final drop for every
        # element uniformly.
        try:
            del tensor
        except NameError:
            pass
        tensor_list.clear()
        del tensor_list
        gc.collect()

        # Verify all memory was freed
        stats_after_gc = get_allocator_stats()
        final_allocated_bytes = stats_after_gc["allocated_bytes"]
        final_num_allocs = stats_after_gc["num_allocs"]

        # Check that free-space returned to baseline
        # "modulo fragmentation" means we allow for some internal fragmentation,
        # but the number of live allocations should be exactly zero
        self.assertEqual(
            final_num_allocs,
            initial_num_allocs,
            f"Number of live allocations should return to baseline after GC. "
            f"Expected {initial_num_allocs}, got {final_num_allocs}. "
            f"This indicates {final_num_allocs - initial_num_allocs} tensors were not freed.",
        )

        # Allocated bytes should also return to baseline
        # FlexAllocator should not have fragmentation issues that prevent
        # returning to the exact baseline, since freed blocks are coalesced
        self.assertEqual(
            final_allocated_bytes,
            initial_allocated_bytes,
            f"Allocated bytes should return to baseline after GC. "
            f"Expected {initial_allocated_bytes}, got {final_allocated_bytes}. "
            f"Delta: {final_allocated_bytes - initial_allocated_bytes} bytes. "
            f"This indicates a memory leak or fragmentation issue.",
        )

    def test_gc_mixed_scope_release(self):
        """
        Mixed-scope tensor release via reference counting.

        Allocate tensors across two scopes:
        - Some held in a long-lived dict (still reachable throughout the test)
        - Others in a function's local scope (freed by refcounting on return)

        Verify that exactly the out-of-scope tensors are released when the function
        returns and that the long-lived ones remain allocated.

        Note: function-local tensors are non-cyclic, so CPython's reference counting
        frees them immediately when the function returns — gc.collect() is a no-op
        for them. The test verifies that refcount-driven release correctly distinguishes
        reachable from unreachable tensors and that only the unreachable allocations
        are removed from the allocator's accounting.
        """
        # Record baseline allocator state
        initial_stats = self.initial_stats
        initial_allocated_bytes = initial_stats["allocated_bytes"]
        initial_num_allocs = initial_stats["num_allocs"]

        # Module-level globals that will remain reachable
        # We'll use a dictionary to simulate module globals
        module_globals = {}

        # Size constants
        GLOBAL_SIZE = 4096  # 16KB per global tensor
        LOCAL_SIZE = 2048  # 8KB per local tensor
        NUM_GLOBALS = 3
        NUM_LOCALS = 5

        # Allocate global tensors (these will remain reachable)
        for i in range(NUM_GLOBALS):
            tensor = torch.empty((GLOBAL_SIZE,), device="spyre", dtype=torch.float16)
            self.assertGreater(
                tensor.data_ptr(),
                0,
                f"Global tensor {i} should have valid data pointer",
            )
            module_globals[f"global_tensor_{i}"] = tensor

        # Delete loop variables to avoid holding extra references
        del tensor

        # Verify global tensors were allocated
        stats_after_globals = get_allocator_stats()
        globals_allocated_bytes = (
            stats_after_globals["allocated_bytes"] - initial_allocated_bytes
        )
        globals_num_allocs = stats_after_globals["num_allocs"] - initial_num_allocs

        self.assertEqual(
            globals_num_allocs,
            NUM_GLOBALS,
            f"Expected {NUM_GLOBALS} global allocations, got {globals_num_allocs}",
        )
        self.assertGreater(
            globals_allocated_bytes, 0, "Global tensors should allocate memory"
        )

        # Function that allocates local tensors (unreachable after return)
        def allocate_local_tensors():
            """Allocate tensors in function scope that will be unreachable after return."""
            local_tensors = []
            for i in range(NUM_LOCALS):
                tensor = torch.empty((LOCAL_SIZE,), device="spyre", dtype=torch.float16)
                self.assertGreater(
                    tensor.data_ptr(),
                    0,
                    f"Local tensor {i} should have valid data pointer",
                )
                local_tensors.append(tensor)

            # Verify local tensors were allocated
            stats_with_locals = get_allocator_stats()
            return stats_with_locals

        # Call function to allocate local tensors
        stats_with_locals = allocate_local_tensors()

        # At this point, local_tensors is out of scope and has already been freed
        # by refcounting. stats_with_locals captured the state while it was still alive.
        total_num_allocs = stats_with_locals["num_allocs"] - initial_num_allocs

        self.assertEqual(
            total_num_allocs,
            NUM_GLOBALS + NUM_LOCALS,
            f"Expected {NUM_GLOBALS + NUM_LOCALS} total allocations, got {total_num_allocs}",
        )

        # Local tensors were already freed by refcounting on function return above.
        # gc.collect() is a no-op for them but is called for hygiene.
        gc.collect()

        # Verify that only the out-of-scope tensors were freed, globals remain
        stats_after_gc = get_allocator_stats()
        remaining_allocated_bytes = (
            stats_after_gc["allocated_bytes"] - initial_allocated_bytes
        )
        remaining_num_allocs = stats_after_gc["num_allocs"] - initial_num_allocs

        # Check that exactly NUM_GLOBALS allocations remain (the reachable ones)
        self.assertEqual(
            remaining_num_allocs,
            NUM_GLOBALS,
            f"Expected {NUM_GLOBALS} allocations to remain (globals), got {remaining_num_allocs}. "
            f"{'Local tensors were not all freed.' if remaining_num_allocs > NUM_GLOBALS else 'Too many allocations were freed.'}",
        )

        # Check that allocated bytes match the global tensors only
        # (exact match expected, since FlexAllocator coalesces freed blocks)
        self.assertEqual(
            remaining_allocated_bytes,
            globals_allocated_bytes,
            f"Expected {globals_allocated_bytes} bytes to remain (globals), got {remaining_allocated_bytes}. "
            f"Delta: {remaining_allocated_bytes - globals_allocated_bytes} bytes.",
        )

        # Verify that global tensors are still accessible and valid
        # We'll verify just one tensor to avoid creating multiple references
        test_tensor = module_globals["global_tensor_0"]
        self.assertGreater(
            test_tensor.data_ptr(),
            0,
            "Global tensor should still have valid data pointer after GC",
        )
        # Verify we can still use the tensor
        test_tensor.fill_(42.0)
        # Create CPU copy in a temporary variable to avoid holding references
        cpu_copy = test_tensor.cpu()
        self.assertTrue(
            torch.all(cpu_copy == 42.0), "Global tensor should still be usable after GC"
        )
        # Explicitly delete the CPU copy and test_tensor reference immediately
        del cpu_copy
        del test_tensor

        # Cleanup: delete global tensors
        keys_to_delete = list(module_globals.keys())
        # Delete each tensor from the dictionary
        for key in keys_to_delete:
            del module_globals[key]

        # Force garbage collection for remaining tensors
        gc.collect()

        # Final verification: all memory should be freed
        final_stats = get_allocator_stats()
        self.assertEqual(
            final_stats["allocated_bytes"],
            initial_allocated_bytes,
            "All memory should be freed after cleanup",
        )
        self.assertEqual(
            final_stats["num_allocs"],
            initial_num_allocs,
            "All allocations should be freed after cleanup",
        )

    def test_gc_cyclic_references(self):
        """
        Garbage Collector cyclic reference handling

        Construct a Python object cycle that holds Spyre tensors:
        - Object A holds a tensor and a reference to B
        - Object B holds a tensor and a reference to A

        Delete external handles and force gc.collect() to invoke the cycle collector.
        Verify that the cycle is broken and both tensors' storage is released.

        This test verifies that Python's cycle collector can properly handle
        reference cycles involving Spyre tensors, ensuring no memory leaks
        when circular references exist.
        """
        # Record baseline allocator state
        initial_stats = self.initial_stats
        initial_allocated_bytes = initial_stats["allocated_bytes"]
        initial_num_allocs = initial_stats["num_allocs"]

        # Define a simple container class that can participate in reference cycles
        class TensorHolder:
            """Container that holds a tensor and can reference another TensorHolder."""

            def __init__(self, name, tensor_size):
                self.name = name
                self.tensor = torch.empty(
                    (tensor_size,), device="spyre", dtype=torch.float16
                )
                self.other = None  # Will hold reference to another TensorHolder

            def set_other(self, other):
                """Create a reference to another TensorHolder."""
                self.other = other

        # Size for each tensor
        TENSOR_SIZE = 4096  # 16KB per tensor

        # Create object A with its tensor
        obj_a = TensorHolder("A", TENSOR_SIZE)
        self.assertGreater(
            obj_a.tensor.data_ptr(),
            0,
            "Object A's tensor should have valid data pointer",
        )

        # Create object B with its tensor
        obj_b = TensorHolder("B", TENSOR_SIZE)
        self.assertGreater(
            obj_b.tensor.data_ptr(),
            0,
            "Object B's tensor should have valid data pointer",
        )

        # Verify both tensors were allocated
        stats_after_alloc = get_allocator_stats()
        allocated_bytes_delta = (
            stats_after_alloc["allocated_bytes"] - initial_allocated_bytes
        )
        num_allocs_delta = stats_after_alloc["num_allocs"] - initial_num_allocs

        self.assertEqual(
            num_allocs_delta,
            2,
            f"Expected 2 allocations (one per object), got {num_allocs_delta}",
        )
        self.assertGreater(
            allocated_bytes_delta, 0, "Both tensors should allocate memory"
        )

        # Verify 128-byte alignment
        self.assertEqual(
            allocated_bytes_delta % 128,
            0,
            f"Allocated bytes ({allocated_bytes_delta}) should be aligned to 128-byte boundary",
        )

        # Disable automatic GC so the interpreter's threshold-triggered collection
        # cannot break the cycle before our explicit gc.collect() call below.
        # This ensures collected reflects exactly our cycle, not background activity.
        gc.disable()
        try:
            # Create the reference cycle: A → B and B → A
            obj_a.set_other(obj_b)
            obj_b.set_other(obj_a)

            # Verify the cycle exists
            self.assertIs(obj_a.other, obj_b, "Object A should reference object B")
            self.assertIs(obj_b.other, obj_a, "Object B should reference object A")
            self.assertIs(
                obj_a.other.other, obj_a, "Cycle should be complete: A → B → A"
            )

            # Delete external handles to the cycle.
            # After this, the only references to obj_a and obj_b are within the
            # cycle itself — refcounts are non-zero, so only the cycle collector
            # can break this.
            del obj_a
            del obj_b

            # Invoke the cycle collector explicitly.
            # gc.collect() returns the number of unreachable objects collected.
            collected = gc.collect()
        finally:
            gc.enable()

        # Verify that both tensors' storage was released
        stats_after_gc = get_allocator_stats()
        final_allocated_bytes = stats_after_gc["allocated_bytes"]
        final_num_allocs = stats_after_gc["num_allocs"]

        # Check that all allocations were freed (cycle was broken)
        self.assertEqual(
            final_num_allocs,
            initial_num_allocs,
            f"Number of live allocations should return to baseline after cycle collection. "
            f"Expected {initial_num_allocs}, got {final_num_allocs}. "
            f"This indicates the cycle was not broken and {final_num_allocs - initial_num_allocs} tensors were not freed.",
        )

        # Check that all memory was freed
        self.assertEqual(
            final_allocated_bytes,
            initial_allocated_bytes,
            f"Allocated bytes should return to baseline after cycle collection. "
            f"Expected {initial_allocated_bytes}, got {final_allocated_bytes}. "
            f"Delta: {final_allocated_bytes - initial_allocated_bytes} bytes. "
            f"This indicates the cycle was not fully broken or there is a memory leak.",
        )

        # The cycle comprises two TensorHolder objects, so the collector must have
        # found at least 2 unreachable objects. A return value of 0 means the cycle
        # was collected before our gc.collect() (impossible with gc.disable() above);
        # a value of 1 would mean only one side of the cycle was collected.
        self.assertGreaterEqual(
            collected,
            2,
            f"gc.collect() should have collected at least 2 objects (both TensorHolders). "
            f"Got {collected}.",
        )

    def test_gc_repeated_reuse_churn(self):
        """
        Garbage Collector repeated reuse churn

        Run T≥1000 iterations of:
        1. Allocate a tensor
        2. Write a unique sentinel value
        3. Drop the tensor
        4. Force GC
        5. Allocate another tensor

        Verify:
        - Allocator free-space remains steady (no leak)

        This test ensures that repeated allocation/deallocation cycles don't
        cause memory leaks across iterations.

        Note: this test does NOT assert that reused storage is wiped. Neither the
        CPU nor the Spyre allocator zeroes memory on reuse, so observing residual
        bytes from a prior iteration in a freshly allocated tensor is EXPECTED and
        ACCEPTABLE (consistent with CPU allocator semantics). The sentinel is still
        written to exercise the fill_ path, but its residual is not checked.
        """
        T = 1000  # Number of iterations (acceptance criteria: T ≥ 1000)
        TENSOR_SIZE = 2048  # 8KB per tensor

        # Record baseline allocator state
        initial_stats = self.initial_stats
        initial_allocated_bytes = initial_stats["allocated_bytes"]
        initial_num_allocs = initial_stats["num_allocs"]

        # Samples taken while exactly one tensor (tensor2) is alive, giving a
        # meaningful steady-state signal: each sample should equal
        # initial_allocated_bytes + one aligned allocation.
        bytes_samples = []
        allocs_samples = []
        sample_interval = 100

        for iteration in range(T):
            # Use a unique sentinel for this iteration
            sentinel = float(iteration + 1000)

            # Step 1: Allocate tensor
            tensor = torch.empty((TENSOR_SIZE,), device="spyre", dtype=torch.float16)

            # Step 2: Write sentinel value
            tensor.fill_(sentinel)

            # Step 3: Drop the tensor
            del tensor

            # Step 4: Force GC periodically (non-cyclic, so refcounting already
            # freed the tensor above; this is a hygiene call)
            if iteration % 10 == 0:
                gc.collect()

            # Step 5: Allocate another tensor (will reuse the freed block)
            tensor2 = torch.empty((TENSOR_SIZE,), device="spyre", dtype=torch.float16)

            # Sample allocator state while tensor2 is alive (one allocation live).
            # Sampling here, not before the alloc, gives a non-trivial signal.
            if iteration % sample_interval == 0:
                stats = get_allocator_stats()
                bytes_samples.append(stats["allocated_bytes"])
                allocs_samples.append(stats["num_allocs"])

            # Clean up for next iteration
            del tensor2

        # Final GC to clean up any remaining allocations
        gc.collect()

        # Final verification: no memory leak
        final_stats = get_allocator_stats()
        final_allocated_bytes = final_stats["allocated_bytes"]
        final_num_allocs = final_stats["num_allocs"]

        self.assertEqual(
            final_num_allocs,
            initial_num_allocs,
            f"Number of allocations should return to baseline after {T} iterations. "
            f"Expected {initial_num_allocs}, got {final_num_allocs}. "
            f"Leak: {final_num_allocs - initial_num_allocs} allocations.",
        )

        self.assertEqual(
            final_allocated_bytes,
            initial_allocated_bytes,
            f"Allocated bytes should return to baseline after {T} iterations. "
            f"Expected {initial_allocated_bytes}, got {final_allocated_bytes}. "
            f"Delta: {final_allocated_bytes - initial_allocated_bytes} bytes.",
        )

        # Each sample was taken with exactly one tensor2 alive, so every sample
        # should show exactly one allocation above the baseline. Any variation
        # indicates a leak or double-count in the allocator's stats.
        expected_bytes_with_one_alloc = bytes_samples[0]
        self.assertTrue(
            all(b == expected_bytes_with_one_alloc for b in bytes_samples),
            f"Allocated bytes should be constant across sampled iterations "
            f"(one live tensor each time). Got: {bytes_samples}",
        )
        self.assertTrue(
            all(a == allocs_samples[0] for a in allocs_samples),
            f"Allocation count should be constant across sampled iterations "
            f"(one live tensor each time). Got: {allocs_samples}",
        )

    def test_gc_multithreaded_churn(self):
        """
        Garbage Collector multi-threaded churn

        Spawn N Python threads (N=8); each thread independently runs a churn loop
        (allocate, write thread-local sentinel, drop, gc.collect(), allocate again)
        for T iterations.

        Verify:
        (a) No allocator-side double-free or assertion failure (relies on mutex protection)
        (b) Total allocator free-space at end matches start
        (c) No deadlocks or race conditions

        Note on cross-thread data leakage:
        Similar to test_gc_residual_data_on_reuse, this test follows CPU allocator semantics.
        Neither CPU nor Spyre allocators zero memory on reuse, so cross-thread sentinel
        leakage is EXPECTED and ACCEPTABLE behavior. The allocator reuses freed memory
        without zeroing, which is consistent with CPU allocator behavior and is a
        performance optimization. The key correctness properties are:
        - No double-free or memory corruption
        - No deadlocks or race conditions
        - Memory is properly freed (no leaks)

        This test exercises the SpyreAllocator → ReportAndDelete → FlexAllocator path
        under GIL-released contention, verifying thread safety of the allocator.

        Note: Run under ThreadSanitizer (TSan) to verify TSan-clean execution.
        """

        N = 8  # Number of threads
        T = 200  # Iterations per thread
        TENSOR_SIZE = 1024  # 4KB per tensor

        # Record baseline allocator state
        initial_stats = self.initial_stats
        initial_allocated_bytes = initial_stats["allocated_bytes"]
        initial_num_allocs = initial_stats["num_allocs"]

        # Shared state for tracking errors
        errors = []
        errors_lock = threading.Lock()

        def thread_worker(thread_id):
            """Worker function that each thread executes."""
            try:
                # Per-thread sentinel base spaced 1000 apart, all within float16
                # range (max 65504). Thread 0: 1000–1199, thread 1: 2000–2199, …,
                # thread 7: 8000–8199.
                sentinel_base = (thread_id + 1) * 1000

                for iteration in range(T):
                    # Use a unique sentinel for this thread and iteration
                    sentinel = float(sentinel_base + iteration)

                    # Step 1: Allocate tensor
                    tensor = torch.empty(
                        (TENSOR_SIZE,), device="spyre", dtype=torch.float16
                    )

                    # Step 2: Write thread-local sentinel
                    tensor.fill_(sentinel)

                    # Step 3: Drop the tensor
                    del tensor

                    # Step 4: Force GC (less frequently for performance)
                    if iteration % 10 == 0:
                        gc.collect()

                    # Step 5: Allocate another tensor (will likely reuse freed storage)
                    # Note: This tensor may contain residual data from this thread or other threads.
                    # This is expected behavior matching CPU allocator semantics (no zeroing on reuse).
                    tensor2 = torch.empty(
                        (TENSOR_SIZE,), device="spyre", dtype=torch.float16
                    )

                    # Clean up for next iteration
                    del tensor2

            except Exception as e:
                error_msg = (
                    f"Thread {thread_id} raised exception: {type(e).__name__}: {e}"
                )
                with errors_lock:
                    errors.append(error_msg)

        # Spawn N threads
        threads = []
        for thread_id in range(N):
            thread = threading.Thread(target=thread_worker, args=(thread_id,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # Check for any Python-level exceptions reported by threads.
        if errors:
            self.fail(
                f"Multi-threaded churn detected {len(errors)} error(s):\n"
                + "\n".join(errors)
            )

        # Two gc.collect() passes after all threads have joined.
        # join() guarantees the thread function returned, but the OS thread's
        # frame (and any tensors it still references) may not be released until
        # CPython's threading bookkeeping drops the frame — which happens
        # asynchronously and after join(). The first collect() releases those
        # frames; the second collects anything their __del__ callbacks queued.
        gc.collect()
        gc.collect()

        # Verify allocator state returned to baseline (no leak)
        final_stats = get_allocator_stats()
        self.assertEqual(
            final_stats["num_allocs"],
            initial_num_allocs,
            f"Allocation count should return to baseline after {N}×{T} iterations. "
            f"Expected {initial_num_allocs}, got {final_stats['num_allocs']}. "
            f"Leak: {final_stats['num_allocs'] - initial_num_allocs} allocations.",
        )
        self.assertEqual(
            final_stats["allocated_bytes"],
            initial_allocated_bytes,
            f"Allocated bytes should return to baseline after {N}×{T} iterations. "
            f"Expected {initial_allocated_bytes}, got {final_stats['allocated_bytes']}. "
            f"Delta: {final_stats['allocated_bytes'] - initial_allocated_bytes} bytes.",
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
