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

import contextlib
import os
import uuid
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests

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


# Skip all tests if RANK is not defined, or WORLD_SIZE is not set or less than 2
if "RANK" not in os.environ:
    pytest.skip(
        "RANK environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

if "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "WORLD_SIZE environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

try:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2:
        pytest.skip(
            f"WORLD_SIZE is {world_size}, need at least 2 for distributed tests",
            allow_module_level=True,
        )
except ValueError:
    pytest.skip(
        "WORLD_SIZE environment variable is not a valid integer, skipping distributed tests",
        allow_module_level=True,
    )

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"

# Geometry from ibm-ai-platform/micro-g3.3-8b-instruct-1b config.json.
# head_dim is hidden_size 4096 / num_attention_heads 32.
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_BLOCKS = 4
BLOCK_SIZE = 16
BLOCK_ID = 1


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


def make_cache(kv_cache_shape, fill_data):
    """Allocate a KV cache with the production layout. Host allocated then
    transferred, since only .to() accepts a device_layout.
    """
    num_blocks, block_size, num_kv_heads, head_dim = kv_cache_shape
    layout = slot_major_layout(num_blocks * block_size, num_kv_heads, head_dim)
    host_cache = torch.zeros(kv_cache_shape, dtype=torch.float16)
    if fill_data:
        # Each page holds its own index so both ranks build the same data.
        for block in range(num_blocks):
            host_cache[block] = block + 1
    return host_cache.to(DEVICE, device_layout=layout)


class TestKVOffloadCrossProcess(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed environment once for all tests."""
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            # A rank that dies outright (not just raises) would otherwise leave
            # its peer waiting the default 30 minutes in the next collective
            dist.init_process_group(
                f"cpu:gloo,spyre:{C10D_BACKEND}", timeout=timedelta(minutes=5)
            )

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        # Pools outlive the processes that made them, so a fixed name would
        # collide with a pool a killed run left behind. Rank 0 picks a per-run
        # name and broadcasts it so every rank uses the same one.
        name = f"{self.id()}.{uuid.uuid4().hex}" if self.comm_rank == 0 else None
        self.pool_name = self._broadcast(name)

    def _broadcast(self, obj):
        """Send obj from rank 0 to every rank; it is pickled on the way."""
        objs = [obj]
        dist.broadcast_object_list(objs, src=0, device=torch.device("cpu"))
        return objs[0]

    @contextlib.contextmanager
    def _step(self):
        """
        Run one step of a test on this rank, then sync with the other ranks.
        If any rank's step fails, every rank raises here, instead of the other
        ranks waiting forever at a later barrier.
        """
        error = None
        try:
            yield
        except Exception as e:
            error = e
        failed = torch.tensor([0 if error is None else 1])
        dist.all_reduce(failed)
        if error is not None:
            raise error
        if failed.item():
            raise RuntimeError("Another rank failed this step")

    def _cleanup(self, descriptor, pool):
        """
        Detach and destroy, also after a failed step. All ranks get here
        together, since _step() makes them fail at the same step.
        """
        if pool is not None:
            detach_pool(pool)
        dist.barrier()  # Ensure every rank has detached
        if self.comm_rank == 0 and descriptor is not None:
            status = destroy_pool(descriptor)
            self.assertIn(status, (PoolStatus.Ok, PoolStatus.NotFound))

    def test_cross_process_reload(self):
        """
        Test that a KV page offloaded in one process can be reloaded in another.
        """
        kv_cache_shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        cache = make_cache(kv_cache_shape, fill_data=True)
        descriptor = pool = None
        try:
            with self._step():
                if self.comm_rank == 0:
                    # Process 0: create the pool as the delegator would,
                    # offload one page into slot 0, then detach. The pool keeps
                    # the page.
                    page_bytes = get_composite_address(cache).total_size // NUM_BLOCKS
                    config = PoolConfig(
                        self.pool_name, num_slots=1, slot_bytes=page_bytes
                    )
                    status, descriptor = create_pool(config)
                    self.assertEqual(status, PoolStatus.Ok)
                    status, pool = attach_pool(descriptor)
                    self.assertEqual(status, PoolStatus.Ok)
                    copy_kv_page_raw(cache, BLOCK_ID, pool, 0, to_device=False)
                    detach_pool(pool)

            # The delegator hands the descriptor to the other ranks
            received = self._broadcast(descriptor)

            with self._step():
                if self.comm_rank == 1:
                    # Process 1: attach with the descriptor it received and
                    # reload the page from the pool
                    status, pool = attach_pool(received)
                    self.assertEqual(status, PoolStatus.Ok)
                    reloaded_cache = make_cache(kv_cache_shape, fill_data=False)
                    copy_kv_page_raw(reloaded_cache, BLOCK_ID, pool, 0, to_device=True)
                    detach_pool(pool)

                    # The whole reloaded cache must match byte for byte: the
                    # source page at BLOCK_ID and zeros everywhere else, so a
                    # write into a neighboring page is caught too
                    expected = torch.zeros(kv_cache_shape, dtype=torch.float16)
                    expected[BLOCK_ID] = cache.to("cpu")[BLOCK_ID]
                    self.assertEqual(
                        reloaded_cache.to("cpu").contiguous().view(torch.uint8),
                        expected.contiguous().view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
        finally:
            # Detach anything still attached, even after a failed step, then
            # the delegator destroys the pool
            self._cleanup(descriptor, pool)


if __name__ == "__main__":
    run_tests()
