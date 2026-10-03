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


from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torch_spyre._C import SharedHostPool  # type: ignore[attr-defined]

# Fixed slot sizes keep this test offline and focused on pool creation and
# attachment. 128 B is one stick. 256 KiB is 262144 B, and 8 of those slots
# make a 2 MiB pool.
SMALL_SLOT_BYTES = 128
LARGE_SLOT_BYTES = 256 * 1024


class TestSharedHostPool(TestCase):
    """
    Tests for the SharedHostPool functionality in the torch_spyre module.
    """

    def test_create_or_attach(self):
        # Create a shared pool
        shared_pool = SharedHostPool.create_or_attach(self.id(), 5, 5)

        # Check if slot count is as expected
        self.assertEqual(shared_pool.slot_count(), 5)

        # Check if greater than or equal because the actual slot bytes may be
        # larger due to alignment of size/stride of the pool
        self.assertGreaterEqual(shared_pool.slot_bytes(), 5)

    def test_attach_existing_pool(self):
        # Create a shared pool and assign to _ to keep it alive
        _ = SharedHostPool.create_or_attach(self.id(), 5, 5)

        # Attach to the existing shared pool
        shared_pool_compare = SharedHostPool.create_or_attach(self.id(), 5, 5)

        self.assertEqual(shared_pool_compare.slot_count(), 5)
        self.assertGreaterEqual(shared_pool_compare.slot_bytes(), 5)

    def test_geometry_mismatch(self):
        # Create a shared pool and assign to _ to keep it alive
        _ = SharedHostPool.create_or_attach(self.id(), 5, 5)

        # Attempt to attach to the existing shared pool with different geometry
        with self.assertRaises(RuntimeError):
            SharedHostPool.create_or_attach(self.id(), 10, 10)

    def test_no_host_pointer(self):
        # Create a shared pool
        shared_pool = SharedHostPool.create_or_attach(self.id(), 5, 5)

        # Confirm that the shared pool does not have a host pointer attribute
        self.assertFalse(hasattr(shared_pool, "slot_ptr"))

    def test_pool_small_slot(self):
        """Many small slots, one stick each."""
        slot_count = 512
        shared_pool = SharedHostPool.create_or_attach(
            self.id(), slot_count, SMALL_SLOT_BYTES
        )

        self.assertEqual(shared_pool.slot_count(), slot_count)
        self.assertGreaterEqual(shared_pool.slot_bytes(), SMALL_SLOT_BYTES)

    def test_pool_large_slot(self):
        """A multi MB pool, 8 slots of 256 KiB."""
        slot_count = 8
        shared_pool = SharedHostPool.create_or_attach(
            self.id(), slot_count, LARGE_SLOT_BYTES
        )

        self.assertEqual(shared_pool.slot_count(), slot_count)
        self.assertGreaterEqual(shared_pool.slot_bytes(), LARGE_SLOT_BYTES)

    def test_name(self):
        # Create a shared pool
        shared_pool = SharedHostPool.create_or_attach(self.id(), 5, 5)

        # Check if name is as expected
        self.assertEqual(shared_pool.name(), self.id())

    def test_total_bytes(self):
        # Create a shared pool
        shared_pool = SharedHostPool.create_or_attach(self.id(), 5, 5)

        # Check if total bytes are as expected
        self.assertEqual(
            shared_pool.total_bytes(),
            shared_pool.slot_count() * shared_pool.slot_bytes(),
        )


if __name__ == "__main__":
    run_tests()
