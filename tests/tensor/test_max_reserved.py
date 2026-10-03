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


"""Tests for `tensor.to("spyre", max=...)`: max-strided HBM reservation.

This backs the runtime dependency for dynamic shaped tensors: a dynamic-batch tensor
is allocated once at its declared ceiling so a compiled graph's static
addresses/strides stay valid for every concrete size in range, and a later
in-place `resize_` up to that ceiling never reallocates or changes the
SpyreTensorLayout a recompile guard would compare against.
"""

import unittest

import torch

import torch_spyre  # noqa: F401

DTYPE = torch.float16


def _identifiable_tensor(shape):
    """Sequential integers 0..numel-1 reshaped to `shape`.

    Unique values detect corruption without assuming element ordering. All
    values are exact in float16 (< 2048).
    """
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.arange(numel, dtype=DTYPE).reshape(shape)


class TestMaxReserved(unittest.TestCase):
    def test_logical_shape_matches_real_size(self):
        """The returned tensor's PyTorch-visible shape is the real shape, not max."""
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)
        self.assertEqual(tuple(x_dev.shape), (56, 32))

    def test_data_correct_after_transfer(self):
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)
        torch.testing.assert_close(x_dev.cpu(), x)

    def test_resize_within_reservation_no_realloc(self):
        """Growing within [*, max] must reuse the same storage (no realloc)."""
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)
        storage_ptr = x_dev.untyped_storage().data_ptr()
        storage_nbytes = x_dev.untyped_storage().nbytes()

        x_dev.resize_(600, 32)

        self.assertEqual(x_dev.untyped_storage().data_ptr(), storage_ptr)
        self.assertEqual(x_dev.untyped_storage().nbytes(), storage_nbytes)
        self.assertEqual(tuple(x_dev.shape), (600, 32))

    def test_resize_shrink_within_reservation_no_realloc(self):
        x = _identifiable_tensor([560, 32])
        x_dev = x.to("spyre", max=616)
        storage_ptr = x_dev.untyped_storage().data_ptr()
        storage_nbytes = x_dev.untyped_storage().nbytes()

        x_dev.resize_(56, 32)

        self.assertEqual(x_dev.untyped_storage().data_ptr(), storage_ptr)
        self.assertEqual(x_dev.untyped_storage().nbytes(), storage_nbytes)
        self.assertEqual(tuple(x_dev.shape), (56, 32))

    def test_storage_nbytes_pinned_across_resizes(self):
        """The allocation is sized once for `max` and never changes size again,
        no matter how many times the tensor is resized within the reservation.

        This is the property that actually matters for the design doc's static
        per-tile addresses (Section 6.2): they're only valid as long as the
        buffer's size, not just its start address, never moves under them.
        A data_ptr()-only check could pass while the storage was silently
        replaced by a same-sized-or-smaller reallocation at the same address;
        checking nbytes() against the size right after the initial reservation
        pins down the actual invariant instead of inferring it indirectly.
        """
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)
        reserved_nbytes = x_dev.untyped_storage().nbytes()

        for shape in [(600, 32), (1, 32), (616, 32), (56, 32)]:
            x_dev.resize_(*shape)
            self.assertEqual(
                x_dev.untyped_storage().nbytes(),
                reserved_nbytes,
                f"storage size changed after resize_{shape}",
            )

    def test_resize_beyond_max_raises(self):
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)
        with self.assertRaises(RuntimeError):
            x_dev.resize_(617, 32)

    def test_resize_dropping_reserved_dim_raises(self):
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)
        with self.assertRaises(RuntimeError):
            x_dev.resize_(32)

    def test_data_correct_after_grow_and_shrink(self):
        """Round-trip through a grow then shrink preserves the front rows."""
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)

        x_dev.resize_(600, 32)
        x_dev.resize_(56, 32)

        torch.testing.assert_close(x_dev.cpu(), x)

    def test_max_must_be_at_least_current_size(self):
        x = _identifiable_tensor([56, 32])
        with self.assertRaises(RuntimeError):
            torch_spyre._C.spyre_empty_reserved(x.size(), x.stride(), x.dtype, 0, 32)

    def test_reserving_innermost_dim_unsupported(self):
        x = _identifiable_tensor([56, 32])
        with self.assertRaises(RuntimeError):
            torch_spyre._C.spyre_empty_reserved(x.size(), x.stride(), x.dtype, 1, 64)

    def test_max_requires_spyre_device(self):
        x = _identifiable_tensor([56, 32])
        with self.assertRaises(ValueError):
            x.to("cpu", max=616)

    def test_max_requires_cpu_source(self):
        x = _identifiable_tensor([56, 32]).to("spyre")
        with self.assertRaises(ValueError):
            x.to("spyre", max=616)

    def test_shallow_copy_preserves_reservation(self):
        """Reservation survives shallow_copy_and_detach (e.g. autograd/Dynamo's
        tensor-guard path calls this via .detach())."""
        x = _identifiable_tensor([56, 32])
        x_dev = x.to("spyre", max=616)
        detached = x_dev.detach()

        detached.resize_(600, 32)
        self.assertEqual(
            detached.untyped_storage().data_ptr(), x_dev.untyped_storage().data_ptr()
        )
        self.assertEqual(
            detached.untyped_storage().nbytes(), x_dev.untyped_storage().nbytes()
        )


if __name__ == "__main__":
    unittest.main()
