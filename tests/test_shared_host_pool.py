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


import gc
import pickle
import uuid
from multiprocessing import shared_memory

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torch_spyre._C import (
    PoolConfig,
    PoolDescriptor,
    PoolState,
    PoolStatus,
    attach_pool,
    create_pool,
    destroy_pool,
    detach_pool,
    open_pool,
)

# Fixed slot sizes keep this test offline and focused on pool creation and
# attachment. 128 B is one stick. 256 KiB is 262144 B, and 8 of those slots
# make a 2 MiB pool.
SMALL_SLOT_BYTES = 128
LARGE_SLOT_BYTES = 256 * 1024


class TestSharedHostPool(TestCase):
    """
    Tests for the SharedHostPool functionality in the torch_spyre module.
    """

    def setUp(self):
        super().setUp()
        # Pools outlive the process that made them, so a killed run leaves its
        # pool behind. A per-run name keeps that leftover from colliding with
        # this run.
        self.pool_name = f"{self.id()}.{uuid.uuid4().hex}"

    def _destroy_checked(self, descriptor):
        # NotFound means the test destroyed the pool itself. Anything else,
        # such as PoolBusy, means the test leaked an attachment.
        self.assertIn(destroy_pool(descriptor), (PoolStatus.Ok, PoolStatus.NotFound))

    def _create(self, num_slots=5, slot_bytes=5):
        config = PoolConfig(self.pool_name, num_slots, slot_bytes)
        status, descriptor = create_pool(config)
        self.assertEqual(status, PoolStatus.Ok)
        # Pools outlive every attachment until destroyed. Cleanups run last-in
        # first-out, so this runs after any detach registered by _attach().
        self.addCleanup(self._destroy_checked, descriptor)
        return descriptor

    def _attach(self, descriptor):
        status, pool = attach_pool(descriptor)
        self.assertEqual(status, PoolStatus.Ok)
        self.addCleanup(detach_pool, pool)
        return pool

    def test_create_and_attach(self):
        # Create a shared pool; it is attachable as soon as create returns
        descriptor = self._create()
        self.assertEqual(descriptor.state, PoolState.Ready)

        shared_pool = self._attach(descriptor)

        # Check if slot count is as expected
        self.assertEqual(shared_pool.slot_count(), 5)

        # Check if greater than or equal because slot bytes are rounded up to
        # the host page size
        self.assertGreaterEqual(shared_pool.slot_bytes(), 5)

    def test_multiple_attachments(self):
        descriptor = self._create()

        # Attach to the same pool twice
        first = self._attach(descriptor)
        second = self._attach(descriptor)

        self.assertEqual(second.slot_count(), 5)
        self.assertGreaterEqual(second.slot_bytes(), 5)
        # Both handles read the same rounded size from the pool
        self.assertEqual(second.slot_bytes(), first.slot_bytes())

    def test_create_existing_pool(self):
        self._create(5, 5)

        # The name is taken, so a second create fails even with a different
        # geometry; it never resizes or replaces the existing pool
        status, descriptor = create_pool(PoolConfig(self.pool_name, 10, 10))
        self.assertEqual(status, PoolStatus.AlreadyExists)
        self.assertIsNone(descriptor)

    def test_no_host_pointer(self):
        # Create and attach a shared pool
        shared_pool = self._attach(self._create())

        # Confirm that the shared pool does not have a host pointer attribute
        self.assertFalse(hasattr(shared_pool, "slot_ptr"))

    def test_pool_small_slot(self):
        """Many small slots, one stick each."""
        slot_count = 512
        shared_pool = self._attach(self._create(slot_count, SMALL_SLOT_BYTES))

        self.assertEqual(shared_pool.slot_count(), slot_count)
        self.assertGreaterEqual(shared_pool.slot_bytes(), SMALL_SLOT_BYTES)

    def test_pool_large_slot(self):
        """A multi MB pool, 8 slots of 256 KiB."""
        slot_count = 8
        shared_pool = self._attach(self._create(slot_count, LARGE_SLOT_BYTES))

        self.assertEqual(shared_pool.slot_count(), slot_count)
        self.assertGreaterEqual(shared_pool.slot_bytes(), LARGE_SLOT_BYTES)

    def test_name(self):
        # Create and attach a shared pool
        shared_pool = self._attach(self._create())

        # Check if name is as expected
        self.assertEqual(shared_pool.name(), self.pool_name)

    def test_total_bytes(self):
        # Create and attach a shared pool
        shared_pool = self._attach(self._create())

        # Check if total bytes are as expected
        self.assertEqual(
            shared_pool.total_bytes(),
            shared_pool.slot_count() * shared_pool.slot_bytes(),
        )

    def test_open_pool(self):
        descriptor = self._create()

        # Opening by name finds the same pool, not just one with that name
        status, opened = open_pool(self.pool_name)
        self.assertEqual(status, PoolStatus.Ok)
        self.assertEqual(opened.generation, descriptor.generation)

    def test_open_missing_pool(self):
        status, descriptor = open_pool(self.pool_name)
        self.assertEqual(status, PoolStatus.NotFound)
        self.assertIsNone(descriptor)

    def test_destroy_while_attached(self):
        descriptor = self._create()
        self._attach(descriptor)

        # An attached process blocks destroy, which leaves the pool as is
        self.assertEqual(destroy_pool(descriptor), PoolStatus.PoolBusy)

    def test_detach_then_destroy(self):
        descriptor = self._create()

        # Attach without _attach(), since this test detaches by hand
        status, pool = attach_pool(descriptor)
        self.assertEqual(status, PoolStatus.Ok)
        detach_pool(pool)

        # With no one attached, destroy succeeds and the pool is gone
        self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)
        status, _ = open_pool(self.pool_name)
        self.assertEqual(status, PoolStatus.NotFound)

    def test_stale_generation(self):
        old = self._create()
        self.assertEqual(destroy_pool(old), PoolStatus.Ok)

        # Re-creating the name gives the new pool a new generation
        self._create()

        # The old descriptor can neither attach to nor destroy the new pool
        status, pool = attach_pool(old)
        self.assertEqual(status, PoolStatus.StaleGeneration)
        self.assertIsNone(pool)
        self.assertEqual(destroy_pool(old), PoolStatus.StaleGeneration)

    def test_use_after_detach_raises(self):
        descriptor = self._create()

        # Attach without _attach(), since this test detaches by hand
        status, pool = attach_pool(descriptor)
        self.assertEqual(status, PoolStatus.Ok)
        detach_pool(pool)

        # detach_pool() took the pool out of the handle, so it can't be used
        with self.assertRaises(ValueError):
            pool.slot_count()

    def test_compatibility_round_trip(self):
        # Opaque bytes, including ones that are not valid text
        tag = b"granite-8b;fp16;block_size=16;\x00\xff"
        config = PoolConfig(self.pool_name, 5, 5, compatibility=tag)
        status, descriptor = create_pool(config)
        self.assertEqual(status, PoolStatus.Ok)
        self.addCleanup(self._destroy_checked, descriptor)

        # The tag comes back unchanged from create and from open
        self.assertEqual(descriptor.config.compatibility, tag)
        status, opened = open_pool(self.pool_name)
        self.assertEqual(status, PoolStatus.Ok)
        self.assertEqual(opened.config.compatibility, tag)

        # A descriptor carrying the tag attaches
        self._attach(descriptor)

    def test_invalid_config_rejected(self):
        name = self.pool_name
        bad_configs = {
            "empty name": PoolConfig("", 5, 5),
            "slash in name": PoolConfig(f"{name}/x", 5, 5),
            "temp file name": PoolConfig(f"{name}.tmp.1", 5, 5),
            "data file name": PoolConfig(f"{name}.data.1", 5, 5),
            "zero slots": PoolConfig(name, 0, 5),
            "zero slot bytes": PoolConfig(name, 5, 0),
        }
        # Caller bugs raise instead of returning a status
        for case, config in bad_configs.items():
            with self.subTest(case), self.assertRaises(ValueError):
                create_pool(config)

        # Every entry point checks the name, not just create
        with self.assertRaises(ValueError):
            open_pool(f"{name}.tmp.1")

    def test_negative_sizes_rejected(self):
        # pybind can't convert a negative size to size_t
        with self.assertRaises(TypeError):
            PoolConfig(self.pool_name, -1, 5)
        with self.assertRaises(TypeError):
            PoolConfig(self.pool_name, 5, -1)

    def test_attach_after_destroy(self):
        descriptor = self._create()
        self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)

        # A worker attaching after the pool is gone gets NotFound, not a pool
        status, pool = attach_pool(descriptor)
        self.assertEqual(status, PoolStatus.NotFound)
        self.assertIsNone(pool)

    def test_non_pool_file_at_name(self):
        descriptor = self._create()
        self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)

        # Put a zero-filled shared memory file, not a pool, at the pool's name
        shm = shared_memory.SharedMemory(name=self.pool_name, create=True, size=4096)
        self.addCleanup(shm.unlink)
        self.addCleanup(shm.close)

        status, _ = open_pool(self.pool_name)
        self.assertEqual(status, PoolStatus.ResourceError)
        status, pool = attach_pool(descriptor)
        self.assertEqual(status, PoolStatus.ResourceError)
        self.assertIsNone(pool)

    def test_detach_twice_is_noop(self):
        descriptor = self._create()
        pool = self._attach(descriptor)

        # The second detach (and _attach()'s cleanup) must not raise; an
        # exception here fails the test
        detach_pool(pool)
        detach_pool(pool)

        # The pool really is detached: nothing blocks destroy anymore
        self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)

    def test_garbage_collection_detaches(self):
        descriptor = self._create()
        status, pool = attach_pool(descriptor)
        self.assertEqual(status, PoolStatus.Ok)

        # Freeing an attached pool's handle detaches it, so destroy succeeds
        del pool
        gc.collect()
        self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)

    def test_destroy_twice(self):
        descriptor = self._create()
        self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)

        # A second destroy reports the pool is gone instead of raising
        self.assertEqual(destroy_pool(descriptor), PoolStatus.NotFound)

    def test_descriptor_survives_pickling(self):
        descriptor = self._create()
        received = pickle.loads(pickle.dumps(descriptor))

        # Every field survives, and the copy attaches like the original
        self.assertEqual(received.name, descriptor.name)
        self.assertEqual(received.generation, descriptor.generation)
        self.assertEqual(received.state, descriptor.state)
        self.assertEqual(received.config.num_slots, descriptor.config.num_slots)
        self.assertEqual(received.config.slot_bytes, descriptor.config.slot_bytes)
        self.assertEqual(received.config.compatibility, descriptor.config.compatibility)
        self._attach(received)

    def test_descriptor_pickle_version_mismatch(self):
        state = self._create().__getstate__()

        # A descriptor pickled by a build with another format is rejected
        received = PoolDescriptor.__new__(PoolDescriptor)
        with self.assertRaises(ValueError):
            received.__setstate__((state[0] + 1,) + state[1:])

    def test_compatibility_mismatch(self):
        state = self._create().__getstate__()

        # A descriptor whose compatibility changed in transit can't attach
        received = PoolDescriptor.__new__(PoolDescriptor)
        received.__setstate__(state[:-1] + (b"other-model",))
        status, pool = attach_pool(received)
        self.assertEqual(status, PoolStatus.ConfigurationMismatch)
        self.assertIsNone(pool)


if __name__ == "__main__":
    run_tests()
