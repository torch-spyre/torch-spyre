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

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import (  # type: ignore[attr-defined]
    PoolConfig,
    PoolStatus,
    attach_pool,
    create_pool,
    destroy_pool,
    detach_pool,
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

C10D_BACKEND = "spyreccl"


class TestSharedHostPoolCrossProcess(TestCase):
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
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

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

    def test_cross_process_shared_pool(self):
        """
        A pool created by one process (the delegator) can be attached in
        another, using the descriptor the delegator sent it.
        """
        num_slots = 1
        slot_bytes = 10
        descriptor = pool = None
        try:
            with self._step():
                if self.comm_rank == 0:
                    # Process 0 acts as the delegator: create, never attach
                    config = PoolConfig(self.pool_name, num_slots, slot_bytes)
                    status, descriptor = create_pool(config)
                    self.assertEqual(status, PoolStatus.Ok)

            # The delegator hands the descriptor to the workers
            received = self._broadcast(descriptor)

            with self._step():
                if self.comm_rank == 1:
                    # Process 1 only attaches, using the descriptor it received
                    status, pool = attach_pool(received)
                    self.assertEqual(status, PoolStatus.Ok)
                    self.assertEqual(pool.slot_count(), num_slots)
                    self.assertEqual(pool.name(), self.pool_name)
                    self.assertEqual(
                        pool.total_bytes(), pool.slot_count() * pool.slot_bytes()
                    )
        finally:
            # Process 1 detaches, then the delegator destroys the pool
            self._cleanup(descriptor, pool)

    def test_cross_process_destroy_blocked_while_attached(self):
        """
        An attachment in one process blocks destroy in another until it
        detaches.
        """
        descriptor = pool = None
        try:
            with self._step():
                if self.comm_rank == 0:
                    config = PoolConfig(self.pool_name, 1, 10)
                    status, descriptor = create_pool(config)
                    self.assertEqual(status, PoolStatus.Ok)

            received = self._broadcast(descriptor)

            with self._step():
                if self.comm_rank == 1:
                    status, pool = attach_pool(received)
                    self.assertEqual(status, PoolStatus.Ok)

            with self._step():
                if self.comm_rank == 0:
                    # Process 1's attachment blocks destroy
                    self.assertEqual(destroy_pool(descriptor), PoolStatus.PoolBusy)

            with self._step():
                if self.comm_rank == 1:
                    detach_pool(pool)

            with self._step():
                if self.comm_rank == 0:
                    # With process 1 detached, destroy succeeds
                    self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)
        finally:
            self._cleanup(descriptor, pool)

    def test_cross_process_stale_descriptor(self):
        """
        A descriptor held by one process is rejected after another process
        destroys and re-creates the pool under the same name.
        """
        config = PoolConfig(self.pool_name, 1, 10)
        descriptor = pool = None
        try:
            with self._step():
                if self.comm_rank == 0:
                    status, descriptor = create_pool(config)
                    self.assertEqual(status, PoolStatus.Ok)

            # Process 1 receives the first descriptor but doesn't attach yet
            old = self._broadcast(descriptor)

            with self._step():
                if self.comm_rank == 0:
                    # Same name, new generation
                    self.assertEqual(destroy_pool(descriptor), PoolStatus.Ok)
                    status, descriptor = create_pool(config)
                    self.assertEqual(status, PoolStatus.Ok)

            with self._step():
                if self.comm_rank == 1:
                    status, pool = attach_pool(old)
                    self.assertEqual(status, PoolStatus.StaleGeneration)
                    self.assertIsNone(pool)
        finally:
            self._cleanup(descriptor, pool)


if __name__ == "__main__":
    run_tests()
