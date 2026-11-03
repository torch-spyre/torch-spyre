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

"""Tests for the get_composite_address accessor (M1-T1).

get_composite_address returns a read-only CompositeAddressHandle over the
flex::CompositeAddress backing a device("spyre") tensor's storage. The handle
holds the tensor's storage alive (keepalive), so the exposed chunk geometry
remains valid even after the caller drops its own reference to the tensor.
"""

import gc

import pytest
from torch.testing._internal.common_utils import TestCase
import torch
import torch_spyre


def _handle(tensor):
    return torch_spyre._C.get_composite_address(tensor)


class TestCompositeAddress(TestCase):
    """Chunk-geometry accessor and its keepalive contract."""

    def test_chunk_geometry_matches_allocation(self):
        """A single device tensor yields at least one chunk whose sizes are
        consistent and sum to the handle's total_size."""
        tensor = torch.empty((512,), device="spyre", dtype=torch.float16)
        handle = _handle(tensor)

        self.assertGreaterEqual(handle.num_chunks, 1)
        chunks = handle.chunks()
        self.assertEqual(len(chunks), handle.num_chunks)

        # total_size is the physical (padded/tiled) byte size; it must cover
        # the logical byte count and equal the sum of the chunk sizes.
        logical_bytes = tensor.numel() * tensor.element_size()
        self.assertGreaterEqual(handle.total_size, logical_bytes)
        self.assertEqual(handle.total_size, sum(c.size for c in chunks))

        for c in chunks:
            self.assertGreater(c.size, 0)

    def test_keepalive_after_caller_drops_tensor(self):
        """The handle keeps the allocation alive: chunk geometry stays valid
        and unchanged after the caller drops its own tensor reference."""
        tensor = torch.empty((512,), device="spyre", dtype=torch.float16)
        handle = _handle(tensor)

        before_total = handle.total_size
        before_chunks = [
            (c.region_id, c.offset, c.size, c.domain_id) for c in handle.chunks()
        ]

        # Drop the caller's reference; only the handle's keepalive remains.
        del tensor
        gc.collect()
        gc.collect()

        after_total = handle.total_size
        after_chunks = [
            (c.region_id, c.offset, c.size, c.domain_id) for c in handle.chunks()
        ]

        self.assertEqual(before_total, after_total)
        self.assertEqual(before_chunks, after_chunks)

    def test_rejects_cpu_tensor(self):
        """A non-Spyre tensor is rejected rather than producing a bogus
        handle."""
        cpu_tensor = torch.empty((16,), dtype=torch.float16)
        with self.assertRaises(RuntimeError):
            _handle(cpu_tensor)

    def test_rejects_non_contiguous_tensor(self):
        """A non-contiguous device tensor is rejected."""
        base = torch.empty((8, 8), device="spyre", dtype=torch.float16)
        view = base.t()
        if view.is_contiguous():
            pytest.skip("transpose is contiguous for this layout")
        with self.assertRaises(RuntimeError):
            _handle(view)

    def test_copy_path_unaffected(self):
        """Regression: obtaining a handle does not perturb the H2D/D2H copy
        path — a round-trip through the device still preserves values."""
        src = torch.arange(64, dtype=torch.float16).reshape(64)
        dev = src.to("spyre")

        # Take a handle (and hold it) while copying back.
        handle = _handle(dev)
        self.assertGreaterEqual(handle.total_size, src.numel() * src.element_size())

        back = dev.to("cpu")
        self.assertTrue(torch.equal(src, back))
