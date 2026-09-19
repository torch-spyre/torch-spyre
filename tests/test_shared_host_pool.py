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


import torch

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
from transformers import AutoConfig

from torch_spyre._C import (  # type: ignore[attr-defined]
    SharedHostPool,
    get_composite_address,
)


class TestSharedHostPool(TestCase):
    """
    Tests for the SharedHostPool functionality in the torch_spyre module.
    """

    def setUp(self):
        super().setUp()

        # Load the model configuration for ibm-ai-platform/micro-g3.3-8b-instruct-1b
        self.cfg = AutoConfig.from_pretrained(
            "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
        )
        self.head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads

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

    def test_pool_real_model_small_slot(self):
        """
        Test SharedHostPool creation with real model ibm-ai-platform/micro-g3.3-8b-instruct-1b
        with a small slot for K/V for 16 tokens 64 (KiB).
        """
        # Shape: num_hidden_layers x 2 (K & V) x 16 (block size) x num_key_value_heads x head_dim
        block_size = 16
        kv_page_shape = (
            self.cfg.num_hidden_layers,
            2,
            block_size,
            self.cfg.num_key_value_heads,
            self.head_dim,
        )

        # Create a tensor for KV Cache page with the shape needed for the model
        kv_page_tensor = torch.randn(kv_page_shape, device="spyre", dtype=torch.float16)

        # Padded/tiled byte count of the page is the slot size
        slot_bytes = get_composite_address(kv_page_tensor).total_size

        # Choosing common prompt length of 8192 (tokens) for testing
        slot_count = 8192 // block_size

        SharedHostPool.create_or_attach(self.id(), int(slot_count), int(slot_bytes))

    def test_pool_real_model_large_slot(self):
        """
        Test SharedHostPool creation with real model ibm-ai-platform/micro-g3.3-8b-instruct-1b
        with a large slot for K/V for 1024 tokens 4 (MiB) to test a multi-MB page.
        """
        # Shape: num_hidden_layers x 2 (K & V) x 1024 (block size) x num_key_value_heads x head_dim
        block_size = 1024
        kv_page_shape = (
            self.cfg.num_hidden_layers,
            2,
            block_size,
            self.cfg.num_key_value_heads,
            self.head_dim,
        )

        # Create a tensor for KV Cache page with the shape needed for the model
        kv_page_tensor = torch.randn(kv_page_shape, device="spyre", dtype=torch.float16)

        # Padded/tiled byte count of the page is the slot size
        slot_bytes = get_composite_address(kv_page_tensor).total_size

        # Choosing common prompt length of 8192 (tokens) for testing
        slot_count = 8192 // block_size

        SharedHostPool.create_or_attach(self.id(), int(slot_count), int(slot_bytes))

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
