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

from torch_spyre._C import (
    PoolConfig,
    PoolStatus,
    SpyreTensorLayout,
    attach_pool,
    create_pool,
    destroy_pool,
    detach_pool,
    copy_kv_page_raw,
    get_composite_address,
    get_device_dtype,
    get_elem_in_stick,
)

# Geometry from ibm-ai-platform/micro-g3.3-8b-instruct-1b config.json.
# head_dim is hidden_size 4096 / num_attention_heads 32.
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_BLOCKS = 4


def slot_major_layout(num_slots, num_kv_heads, head_size, dtype=torch.float16):
    """Mirror of spyre-inference slot_major_kv_layout, which is how the real
    decoder cache is allocated. The slot axis is outermost, so each page is one
    contiguous physical range. stride_map is row major over device_size, which
    is what copy_kv_page_raw requires.
    """
    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def make_cache(kv_cache_shape, fill_random):
    """Allocate a KV cache with the production layout. Host allocated then
    transferred, since only .to() accepts a device_layout.
    """
    num_blocks, block_size, num_kv_heads, head_dim = kv_cache_shape
    layout = slot_major_layout(num_blocks * block_size, num_kv_heads, head_dim)
    host_cache = (
        torch.randn(kv_cache_shape, dtype=torch.float16)
        if fill_random
        else torch.zeros(kv_cache_shape, dtype=torch.float16)
    )
    return host_cache.to("spyre", device_layout=layout)


class TestSpyre(TestCase):
    def setUp(self):
        super().setUp()
        self._num_pools = 0

    def _pool(self, cache, num_blocks):
        """Create and attach a one-slot pool sized for one page of cache."""
        page_bytes = get_composite_address(cache).total_size // num_blocks
        # A test may build more than one pool, so give each its own name
        name = f"{self.id()}.{self._num_pools}"
        self._num_pools += 1

        config = PoolConfig(name, num_slots=1, slot_bytes=page_bytes)
        status, descriptor = create_pool(config)
        self.assertEqual(status, PoolStatus.Ok)
        # Pools outlive every attachment until destroyed. Cleanups run
        # last-in first-out, so the detach below runs before this destroy.
        self.addCleanup(destroy_pool, descriptor)

        status, pool = attach_pool(descriptor)
        self.assertEqual(status, PoolStatus.Ok)
        self.addCleanup(detach_pool, pool)
        return pool

    def _kv_page_round_trip(self, kv_cache_shape, block_id=1):
        """Offload one page to the host pool and reload it into a zeroed cache.
        The expected tensor is zero everywhere except the page we moved, so this
        checks the page survived and that no sibling page was written.
        """
        source_cache = make_cache(kv_cache_shape, fill_random=True)
        dest_cache = make_cache(kv_cache_shape, fill_random=False)
        pool = self._pool(source_cache, kv_cache_shape[0])

        # D2H: only block_id should land in slot 0
        copy_kv_page_raw(source_cache, block_id, pool, 0, to_device=False)

        # H2D: read it back into the zeroed cache
        copy_kv_page_raw(dest_cache, block_id, pool, 0, to_device=True)

        expected_cache = torch.zeros(kv_cache_shape, dtype=torch.float16)
        expected_cache[block_id] = source_cache.to("cpu")[block_id]

        # Compare raw bytes with zero tolerance: the copy must be bit for bit
        self.assertEqual(
            dest_cache.to("cpu").contiguous().view(torch.uint8),
            expected_cache.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )

    def test_small_page(self):
        """One page of 16 tokens, 32 KiB."""
        self._kv_page_round_trip((NUM_BLOCKS, 16, NUM_KV_HEADS, HEAD_DIM))

    def test_large_page(self):
        """One page of 1024 tokens, 2 MiB, to cover a multi MB transfer."""
        self._kv_page_round_trip((NUM_BLOCKS, 1024, NUM_KV_HEADS, HEAD_DIM))

    def test_first_and_last_page(self):
        """Offset 0 and the final page, the two ends of the range arithmetic."""
        shape = (NUM_BLOCKS, 16, NUM_KV_HEADS, HEAD_DIM)
        self._kv_page_round_trip(shape, block_id=0)
        self._kv_page_round_trip(shape, block_id=NUM_BLOCKS - 1)

    def test_rejects_default_layout(self):
        """A cache allocated without a device_layout gets the generated tiled
        layout, which reorders dimensions so a page is not contiguous.
        """
        cache = torch.randn(
            (NUM_BLOCKS, 16, NUM_KV_HEADS, HEAD_DIM),
            device="spyre",
            dtype=torch.float16,
        )
        pool = self._pool(cache, NUM_BLOCKS)
        # Rejected by one of the layout checks, depending on the generated
        # layout's rank
        with self.assertRaisesRegex(RuntimeError, "rank 4 device layout|not row major"):
            copy_kv_page_raw(cache, 1, pool, 0, to_device=False)

    def test_rejects_page_view(self):
        """The whole cache and a block_id, not a pre sliced page."""
        shape = (NUM_BLOCKS, 16, NUM_KV_HEADS, HEAD_DIM)
        cache = make_cache(shape, fill_random=True)
        pool = self._pool(cache, NUM_BLOCKS)
        # A page view is rank 3
        with self.assertRaisesRegex(RuntimeError, "expected a rank 4 KV cache"):
            copy_kv_page_raw(cache[1], 0, pool, 0, to_device=False)

    def test_rejects_block_id_out_of_range(self):
        shape = (NUM_BLOCKS, 16, NUM_KV_HEADS, HEAD_DIM)
        cache = make_cache(shape, fill_random=True)
        pool = self._pool(cache, NUM_BLOCKS)
        with self.assertRaisesRegex(RuntimeError, r"block_id \d+ out of range"):
            copy_kv_page_raw(cache, NUM_BLOCKS, pool, 0, to_device=False)


if __name__ == "__main__":
    run_tests()
