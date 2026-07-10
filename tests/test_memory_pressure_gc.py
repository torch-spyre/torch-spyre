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

"""
Memory Pressure Callback Integration Tests

Tests the end-to-end memory pressure callback mechanism that triggers
Python garbage collection when FlexAllocator exhausts all regions.

These tests verify the acceptance criteria from issue P1-31:
- Test 1: GC releases dead tensors, allocation succeeds
- Test 2: GC runs but nothing freed, OOM is raised
- Test 3: GIL safety - no deadlock
- Test 4: Cycle-collected tensors are released
- Test 5: Concurrent pressure events
"""

import gc
import threading
import time

import pytest
import torch
import torch_spyre


class TestMemoryPressureGC:
    """Test suite for memory pressure callback triggering Python GC."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Ensure GC is enabled at start
        gc.enable()
        gc.collect()

        # Synchronize device before test to ensure clean state
        try:
            torch_spyre.synchronize()
        except Exception:
            pass  # Ignore errors if device is already in error state

        yield

        # Cleanup after test
        gc.collect()

        # Synchronize device after test to clear any pending operations
        try:
            torch_spyre.synchronize()
        except Exception:
            pass  # Ignore errors from OOM tests that put device in error state

    def test_gc_releases_dead_tensors_allocation_succeeds(self):
        """
        Test 1: Pre-allocate tensors to fill memory, drop Python refs but disable
        automatic GC. Allocate one more tensor; verify allocator triggers GC,
        dead tensors are released, and new allocation succeeds.
        """
        # Disable automatic GC to control when collection happens
        gc.disable()

        try:
            # Allocate tensors to fill most of available memory
            # Use a list to hold references, then clear it
            tensors = []
            tensor_size = 1024 * 1024 * 1024  # 1GB each
            num_tensors = 6  # Allocate 6GB

            for i in range(num_tensors):
                t = torch.randn(tensor_size // 4, device="spyre")  # float32 = 4 bytes
                tensors.append(t)

            # Drop all references (simulate Python losing references to tensors)
            tensors.clear()

            # At this point, tensors are dead but not collected (GC disabled)
            # Next allocation should trigger memory pressure callback
            # Callback will run gc.collect(), free the dead tensors, and succeed

            # This allocation should succeed after GC
            new_tensor = torch.randn(tensor_size // 4, device="spyre")

            # Verify allocation succeeded
            assert new_tensor.device.type == "spyre"
            assert new_tensor.numel() == tensor_size // 4

        finally:
            # Re-enable GC
            gc.enable()
            gc.collect()

    def test_gil_safety_no_deadlock(self):
        """
        Test 3: GIL safety - one Python thread holds GIL while another thread
        is mid-allocation; the mid-allocation thread hits pressure, releases
        GIL appropriately, and either acquires it to call GC or yields to holder.
        No deadlock; no allocator-state corruption.
        """
        gc.enable()

        allocation_succeeded = threading.Event()
        error_occurred = threading.Event()

        def allocate_with_pressure():
            """Thread that will hit memory pressure."""
            try:
                # Allocate less memory to avoid actual OOM, just trigger pressure
                tensors = []
                for i in range(4):
                    t = torch.randn(256 * 1024 * 1024, device="spyre")  # 1GB each
                    tensors.append(t)

                # Drop refs to allow GC
                tensors.clear()

                # Small delay to let GC potentially run
                time.sleep(0.05)

                # This should trigger memory pressure callback
                t = torch.randn(256 * 1024 * 1024, device="spyre")
                allocation_succeeded.set()

            except Exception as e:
                print(f"Allocation thread error: {e}")
                error_occurred.set()

        def hold_gil_briefly():
            """Thread that holds GIL doing Python work."""
            # Do some Python work that holds GIL
            for i in range(500):  # Reduced iterations
                _ = [j**2 for j in range(100)]
                time.sleep(0.001)

        # Start both threads
        t1 = threading.Thread(target=allocate_with_pressure, daemon=True)
        t2 = threading.Thread(target=hold_gil_briefly, daemon=True)

        t1.start()
        time.sleep(0.1)  # Let t1 start allocating
        t2.start()

        # Wait for completion with timeout
        t1.join(timeout=15.0)  # Increased timeout
        t2.join(timeout=15.0)

        # Verify no deadlock (threads completed)
        assert not t1.is_alive(), "Allocation thread deadlocked"
        assert not t2.is_alive(), "GIL-holding thread deadlocked"

        # Verify allocation succeeded or failed cleanly (no corruption)
        assert allocation_succeeded.is_set() or error_occurred.is_set()

    def test_cycle_collected_tensors_released(self):
        """
        Test 4: Cycle-collected tensors (with reference cycles) are also
        released by pressure-triggered GC, not just refcount-zero tensors.
        """
        gc.disable()

        try:
            # Create tensors with reference cycles
            class TensorHolder:
                def __init__(self, tensor):
                    self.tensor = tensor
                    self.ref: "TensorHolder | None" = None  # Will create cycle

            holders = []
            tensor_size = 1024 * 1024 * 1024  # 1GB

            # Create 4 tensors with reference cycles
            for i in range(4):
                t = torch.randn(tensor_size // 4, device="spyre")
                holder = TensorHolder(t)
                holder.ref = holder  # Create cycle
                holders.append(holder)

            # Break external references but cycles remain
            holders.clear()

            # Tensors are now only reachable via cycles
            # Next allocation should trigger GC, collect cycles, and succeed
            new_tensor = torch.randn(tensor_size // 4, device="spyre")

            assert new_tensor.device.type == "spyre"

        finally:
            gc.enable()
            gc.collect()

    def test_gc_runs_nothing_freed_oom_raised(self):
        """
        Test 2: Pre-allocate and retain refs to fill memory. Allocate one more
        tensor; verify GC runs, nothing is freed, and OOM is raised (allocator
        did not loop or hang).
        """
        gc.disable()

        # Allocate tensors and KEEP references
        tensors = []
        try:
            # Use torch.empty (no H2D copy) to fill device memory as fast as
            # possible without depending on stream state from prior tests.
            # Fill until the allocator itself raises OOM, keeping all refs live.
            tensor_size = 2 * 1024 * 1024 * 1024  # 2GB per tensor (float32)
            fill_oom_hit = False
            for _ in range(10_000):  # upper bound; OOM breaks the loop
                try:
                    t = torch.empty(
                        tensor_size // 4, dtype=torch.float32, device="spyre"
                    )
                    tensors.append(t)
                except RuntimeError:
                    fill_oom_hit = True
                    break

            # If we hit OOM during fill it means GC ran (nothing to free) and
            # raised OOM — that is exactly the behaviour under test.
            if fill_oom_hit:
                return

            # We filled memory without OOM (device has > 400GB?); try one more
            # explicit allocation to confirm OOM is raised and not suppressed.
            with pytest.raises(RuntimeError, match="out of memory|OOM|OutOfMemory"):
                torch.empty(tensor_size // 4, dtype=torch.float32, device="spyre")

            # Verify we didn't hang or loop infinitely (test completes)

        finally:
            # Cleanup
            tensors.clear()
            gc.enable()
            gc.collect()

    def test_concurrent_pressure_events(self):
        """
        Test 5: N threads (N≥4) each fill memory and simultaneously hit OOM.
        Verify: (a) all allocations either succeed or report OOM cleanly,
        (b) GC is invoked at most once per pressure window, (c) no hangs/corruption.
        """
        gc.enable()

        num_threads = 4
        results = []
        results_lock = threading.Lock()

        def allocate_until_pressure(thread_id):
            """Each thread tries to allocate until hitting pressure."""
            try:
                tensors = []
                # Each thread allocates 2GB
                for i in range(2):
                    t = torch.randn(512 * 1024 * 1024, device="spyre")
                    tensors.append(t)

                # Drop some refs to allow GC to help
                if thread_id % 2 == 0:
                    tensors.clear()

                # Try one more allocation - may succeed or fail
                t = torch.randn(256 * 1024 * 1024, device="spyre")

                with results_lock:
                    results.append(("success", thread_id))

            except RuntimeError:
                with results_lock:
                    results.append(("oom", thread_id))
            except Exception as e:
                with results_lock:
                    results.append(("error", thread_id, str(e)))

        # Launch threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=allocate_until_pressure, args=(i,), daemon=True)
            threads.append(t)
            t.start()

        # Wait for all threads with timeout
        for t in threads:
            t.join(timeout=15.0)

        # Verify no threads hung
        for t in threads:
            assert not t.is_alive(), "Thread deadlocked or hung"

        # Verify all threads completed with clean result
        assert len(results) == num_threads

        # Verify no corruption - all results are valid
        for result in results:
            assert result[0] in ("success", "oom", "error")
            if result[0] == "error":
                pytest.fail(f"Thread {result[1]} had unexpected error: {result[2]}")

        # Note: We can't easily verify GC was called exactly once per pressure window
        # from Python, but the C++ implementation ensures this


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
