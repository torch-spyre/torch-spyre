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

"""Tests for the reserved_dims map and the resize_ pinning guard.

Covers the on-tensor reservation (SpyreTensorImpl::reserved_dims) from the
runtime side: the buffer is actually sized for `max`, a resize within
[min, max] on a granularity multiple never reallocates (whether shrinking
or growing), the device layout stays identical across every legal size so
the recompile guard never fires, and a resize outside any of the declared
bounds -- or that would change rank, or targets the reserved innermost
dimension -- refuses loudly rather than silently reallocating or running
with a stale tail.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401
from torch_spyre._C import get_reserved_dims, get_spyre_tensor_layout

DEVICE = torch.device("spyre")

# 9 buckets (630 / 70), well under the configured bucket cap of 32.
DIM_MIN = 70
DIM_MAX = 630
GRANULARITY = 70


class TestReservationResizeGuard(TestCase):
    def setUp(self) -> None:
        self.x = torch.rand(560, 1024, dtype=torch.float16)
        self.x_dev = self.x.to(
            DEVICE,
            dynamic={0: {"min": DIM_MIN, "max": DIM_MAX, "granularity": GRANULARITY}},
        )

    def test_storage_padded_to_max_not_real_size(self) -> None:
        # device_size[0] at allocation time is DIM_MAX, so storage bytes
        # reflect the ceiling, not the 560 rows actually copied in.
        expected_min_bytes = DIM_MAX * 1024 * self.x_dev.element_size()
        self.assertGreaterEqual(
            self.x_dev.untyped_storage().nbytes(), expected_min_bytes
        )

    def test_logical_size_stays_real(self) -> None:
        # The split this whole feature depends on: the buffer is padded,
        # but what the tensor reports as its own size is not.
        self.assertEqual(self.x_dev.size(0), 560)

    def test_resize_within_bounds_does_not_reallocate(self) -> None:
        before = self.x_dev.untyped_storage().data_ptr()
        nbytes_before = self.x_dev.untyped_storage().nbytes()
        self.x_dev.resize_((140, 1024))  # in [70, 630], multiple of 70
        self.assertEqual(self.x_dev.untyped_storage().data_ptr(), before)
        self.assertEqual(self.x_dev.untyped_storage().nbytes(), nbytes_before)

    def test_resize_growing_within_bounds_does_not_reallocate(self) -> None:
        # A real grow, not a shrink -- the no-realloc guarantee has to
        # hold in both directions, and growing exercises a different
        # branch than shrinking does.
        before = self.x_dev.untyped_storage().data_ptr()
        nbytes_before = self.x_dev.untyped_storage().nbytes()
        self.x_dev.resize_((DIM_MAX, 1024))  # 560 -> 630
        self.assertEqual(self.x_dev.untyped_storage().data_ptr(), before)
        self.assertEqual(self.x_dev.untyped_storage().nbytes(), nbytes_before)

    def test_resize_below_min_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the reservation"):
            self.x_dev.resize_((40, 1024))  # < DIM_MIN

    def test_resize_above_max_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the reservation"):
            self.x_dev.resize_((700, 1024))  # > DIM_MAX

    def test_resize_off_granularity_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "multiple of"):
            self.x_dev.resize_((141, 1024))  # not a multiple of 70

    def test_resize_changing_a_non_reserved_dim_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "may only change the reserved dim"):
            self.x_dev.resize_((140, 512))

    def test_resize_rank_change_rejected(self) -> None:
        # Collapsing [N, 1024] to [1024] would keep the reserved dim's
        # index in bounds of the new 1-D shape while actually dropping the
        # reserved axis entirely, so rank must be checked explicitly
        # rather than inferred from the resized index alone.
        with self.assertRaisesRegex(RuntimeError, "must keep the same rank"):
            self.x_dev.resize_((1024,))

    def test_resize_after_in_place_transpose_rejected(self) -> None:
        # t_() swaps the axes but leaves the reservation's dim index alone,
        # so resize_ would validate the wrong axis. It must refuse.
        d = self.x_dev.detach()
        d.t_()
        with self.assertRaisesRegex(RuntimeError, "changed in place"):
            d.resize_((1024, 140))

    def test_resize_after_in_place_rank_change_rejected(self) -> None:
        d = self.x_dev.detach()
        d.unsqueeze_(0)
        with self.assertRaisesRegex(RuntimeError, "changed in place"):
            d.resize_((1, DIM_MAX, 1024))

    def test_resize_after_valid_resize_still_allowed(self) -> None:
        # resize_ itself rewrites the shape; that must not look like an
        # in-place reshape to the check above.
        self.x_dev.resize_((140, 1024))
        self.x_dev.resize_((DIM_MAX, 1024))
        self.assertEqual(self.x_dev.size(0), DIM_MAX)

    def test_innermost_dim_reservation_rejected(self) -> None:
        # Reserving the innermost (stick) dimension is out of scope today:
        # padding it changes the device stride for every outer dimension,
        # unlike a batch/outer dimension.
        x = torch.rand(560, 1024, dtype=torch.float16)
        with self.assertRaisesRegex(
            RuntimeError, "does not yet support reserving the innermost"
        ):
            x.to(DEVICE, dynamic={1: {"min": 64, "max": 1024, "granularity": 64}})

    def test_layout_is_identical_across_every_legal_size(self) -> None:
        # The property that actually matters: the device layout stays
        # pinned at the declared max across every resize_, which is what
        # keeps the recompile guard from firing when the real size
        # changes. data_ptr/nbytes staying stable is necessary but isn't
        # by itself proof of this.
        before = str(get_spyre_tensor_layout(self.x_dev))
        self.x_dev.resize_((140, 1024))
        self.assertEqual(str(get_spyre_tensor_layout(self.x_dev)), before)
        self.x_dev.resize_((DIM_MAX, 1024))
        self.assertEqual(str(get_spyre_tensor_layout(self.x_dev)), before)

    def test_data_integrity_across_shrink_and_regrow(self) -> None:
        # Shrinking must preserve the kept rows (up to the backend's own
        # H2D/D2H fp16 rounding, which is present on every transfer
        # regardless of reservation -- not a correctness bug in resize_
        # itself, so this intentionally uses a tolerant comparison rather
        # than exact equality). Growing back does not restore or re-zero
        # the rows that fell outside the shrunk view -- that part of the
        # buffer holds whatever was already there. Documented here as a
        # known property of the reservation, not something a caller
        # should have to discover on their own.
        original = self.x.clone()
        self.x_dev.resize_((140, 1024))
        self.assertTrue(torch.allclose(self.x_dev.cpu(), original[:140], atol=1e-3))

        self.x_dev.resize_((560, 1024))
        self.assertTrue(
            torch.allclose(self.x_dev.cpu()[:140], original[:140], atol=1e-3)
        )


class TestReservationBoundaries(TestCase):
    """Edges of the declared range, away from the 70..630 / step-70 case."""

    def test_min_equals_max_allows_only_that_size(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        # granularity must be explicit: the default of 1 would mean 560
        # buckets, over the cap of 32.
        spec = {"min": 560, "max": 560, "granularity": 560}
        x_dev = x.to(DEVICE, dynamic={0: spec})
        self.assertEqual(get_reserved_dims(x_dev), {0: spec})
        x_dev.resize_((560, 1024))  # the one legal size: a no-op
        self.assertEqual(x_dev.size(0), 560)
        with self.assertRaisesRegex(RuntimeError, "outside the reservation"):
            x_dev.resize_((280, 1024))

    def test_granularity_one_accepts_odd_sizes(self) -> None:
        # max / granularity must stay within the bucket cap, so with
        # granularity=1 the ceiling is small.
        x = torch.rand(16, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 8, "max": 32, "granularity": 1}})
        before = x_dev.untyped_storage().data_ptr()
        x_dev.resize_((9, 1024))  # shrink to an odd size
        self.assertEqual(x_dev.size(0), 9)
        x_dev.resize_((31, 1024))  # grow to another odd size
        self.assertEqual(x_dev.size(0), 31)
        self.assertEqual(x_dev.untyped_storage().data_ptr(), before)
        with self.assertRaisesRegex(RuntimeError, "outside the reservation"):
            x_dev.resize_((7, 1024))  # below min
        with self.assertRaisesRegex(RuntimeError, "outside the reservation"):
            x_dev.resize_((33, 1024))  # above max

    def test_omitted_granularity_equals_one(self) -> None:
        x = torch.rand(16, 1024, dtype=torch.float16)
        omitted = x.to(DEVICE, dynamic={0: {"min": 8, "max": 32}})
        explicit = x.to(DEVICE, dynamic={0: {"min": 8, "max": 32, "granularity": 1}})
        self.assertEqual(get_reserved_dims(omitted), get_reserved_dims(explicit))

    def test_non_leading_dim_reservation_rejected(self) -> None:
        # Padding a middle dim changes the host stride of every dim outside
        # it (here dim 0: 560*1024 -> 630*1024), so the layout's stride_map
        # would disagree with the real data's strides and the H2D copy could
        # write past the buffer. Must be refused up front, not attempted.
        x = torch.rand(4, 560, 1024, dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "only dim 0"):
            x.to(DEVICE, dynamic={1: {"min": 70, "max": 630, "granularity": 70}})

    def test_leading_dim_of_3d_tensor_stays_in_place(self) -> None:
        # The supported case on a rank-3 tensor: reserve dim 0. Inner strides
        # do not depend on it, so the buffer and layout stay put.
        x = torch.rand(560, 4, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        ptr = x_dev.untyped_storage().data_ptr()
        layout = str(get_spyre_tensor_layout(x_dev))

        x_dev.resize_((140, 4, 1024))
        self.assertEqual(x_dev.stride(), (4 * 1024, 1024, 1))
        self.assertEqual(x_dev.untyped_storage().data_ptr(), ptr)
        self.assertEqual(str(get_spyre_tensor_layout(x_dev)), layout)

        x_dev.resize_((DIM_MAX, 4, 1024))
        self.assertEqual(x_dev.untyped_storage().data_ptr(), ptr)
        self.assertEqual(str(get_spyre_tensor_layout(x_dev)), layout)


if __name__ == "__main__":
    run_tests()
