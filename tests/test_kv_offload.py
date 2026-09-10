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
    copy_tensor_raw,
    get_composite_address,
)


class TestSpyre(TestCase):
    def setUp(self):
        super().setUp()

        # Load the model configuration for ibm-ai-platform/micro-g3.3-8b-instruct-1b
        self.cfg = AutoConfig.from_pretrained(
            "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
        )
        self.head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads

    def _kv_offload_reload(self, kv_page_tensor, kv_page_tensor_reload):
        """
        Test if the bytes survived a round trip from spyre to host memory pool and back to spyre.
        """
        # Composite address handle for the tensor to determine the size of the slot needed in the shared host pool
        slot_bytes = get_composite_address(kv_page_tensor).total_size

        # Create the shared pool with a single slot of the required size
        pool = SharedHostPool.create_or_attach(
            self.id(), num_slots=1, slot_bytes=slot_bytes
        )

        # Use the first slot in the pool
        slot_id = 0

        # D2H: Move tensor from spyre to host memory pool or fill kv_page_data into the shared host pool
        copy_tensor_raw(kv_page_tensor, pool, slot_id, to_device=False)

        # H2D: Move tensor back from host memory pool to spyre
        copy_tensor_raw(kv_page_tensor_reload, pool, slot_id, to_device=True)

        self.assertEqual(kv_page_tensor.to("cpu"), kv_page_tensor_reload.to("cpu"))

    def test_kv_offload_reload_zeroed(self):
        tensor_on_spyre = torch.randn(10, device="spyre", dtype=torch.float16)
        self._kv_offload_reload(tensor_on_spyre, torch.zeros_like(tensor_on_spyre))

    def test_kv_offload_reload_diff_tensor(self):
        tensor_on_spyre = torch.randn(10, device="spyre", dtype=torch.float16)
        self._kv_offload_reload(tensor_on_spyre, torch.empty_like(tensor_on_spyre))

    def test_normal_copy_tensor_unaffected(self):
        """
        Ensure the normal copy_tensor (.to()) function is unaffected.
        """
        tensor = torch.randn(10, dtype=torch.float16)
        tensor_on_spyre = tensor.to("spyre")
        tensor_back = tensor_on_spyre.to("cpu")
        self.assertEqual(tensor, tensor_back)

    def test_real_model_small_slot(self):
        """
        Test copy_tensor_raw functionality with real model ibm-ai-platform/micro-g3.3-8b-instruct-1b
        with a small slot for K/V for 16 tokens 64 (KiB).
        """
        # Shape: 2 (K & V) x 16 (block size) x num_key_value_heads x head_dim
        kv_page_shape = (2, 16, self.cfg.num_key_value_heads, self.head_dim)

        # Create a tensor for KV Cache page with the shape needed for the model
        kv_page_tensor = torch.randn(kv_page_shape, device="spyre", dtype=torch.float16)
        self._kv_offload_reload(kv_page_tensor, torch.zeros_like(kv_page_tensor))

    def test_real_model_large_slot(self):
        """
        Test copy_tensor_raw functionality with real model ibm-ai-platform/micro-g3.3-8b-instruct-1b
        with a large slot for K/V for 1024 tokens 4 (MiB) to test a multi-MB page.
        """
        # Shape: 2 (K & V) x 1024 (block size) x num_key_value_heads x head_dim
        kv_page_shape = (2, 1024, self.cfg.num_key_value_heads, self.head_dim)

        # Create a tensor for KV Cache page with the shape needed for the model
        kv_page_tensor = torch.randn(kv_page_shape, device="spyre", dtype=torch.float16)
        self._kv_offload_reload(kv_page_tensor, torch.zeros_like(kv_page_tensor))


if __name__ == "__main__":
    run_tests()
