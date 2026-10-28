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

"""Hardware-free unit tests for ``_merge_device_dims_for_view``.

These tests exercise the view-dim-at-a-time merge algorithm in
``torch_spyre._inductor.nonstick_dim_order._merge_device_dims_for_view``
without requiring a Spyre device.  They validate the layout arithmetic
(device_size, stride_map) directly against expected values.

Key invariants:
  * Dims that belong to *different* view dims are never merged.
  * The merged dim inherits the innermost (smallest) stride of the group.
  * Views with no flattening leave the STL unchanged (identity return).
"""

import unittest

import torch
from torch._inductor.ir import FixedLayout

from torch_spyre._C import DataFormats, SpyreTensorLayout
from torch_spyre._inductor.nonstick_dim_order import _merge_device_dims_for_view

# Short aliases for the two device_dtype values used in these tests.
_BF16 = DataFormats.SEN169_FP16  # elems_per_stick = 64
_F32 = DataFormats.IEEE_FP32  # elems_per_stick = 32


def _stl(device_size, stride_map, dtype=_BF16) -> SpyreTensorLayout:
    """Convenience constructor for a SpyreTensorLayout."""
    return SpyreTensorLayout(
        device_size=device_size,
        stride_map=stride_map,
        device_dtype=dtype,
    )


def _view(host_size, host_stride, dtype=torch.bfloat16) -> FixedLayout:
    """Convenience constructor for a FixedLayout (view descriptor)."""
    return FixedLayout("cpu", dtype, host_size, host_stride)


class TestMergeHardwareLayout(unittest.TestCase):
    """Hardware-reported non-default layout: device_size=[16, 8, 2, 512, 64].

    Real hardware reorders dims vs. the default ``propagate_layouts`` order.
    BLOCK_SIZE=16 ends up at pos 0 and NUM_BLOCKS=512 at pos 3; both must be
    merged for the view ``[-1, 8, 128]`` (8192 slots × 8 heads × 128 head_dim).
    """

    # Hardware-reported layout for [NUM_BLOCKS=512, BLOCK_SIZE=16, H=8, D=128] bfloat16
    # device_size=[16, 8, 2, 512, 64]  stride_map=[1024, 128, 64, 16384, 1]
    STL = _stl([16, 8, 2, 512, 64], [1024, 128, 64, 16384, 1])

    def test_blocks_merged_hardware_layout(self):
        """Merging pos-0 (BLOCK_SIZE=16) and pos-3 (NUM_BLOCKS=512) -> 8192."""
        view = _view([8192, 8, 128], [1024, 128, 1])
        result = _merge_device_dims_for_view(self.STL, view)
        self.assertEqual(list(result.device_size), [8192, 8, 2, 64])

    def test_stride_map_after_hardware_merge(self):
        """Merged stride inherits the inner (smallest-stride) dim's stride."""
        view = _view([8192, 8, 128], [1024, 128, 1])
        result = _merge_device_dims_for_view(self.STL, view)
        # Innermost of the pair is pos-0 (stride=1024); that becomes the rep.
        self.assertEqual(result.stride_map[0], 1024)

    def test_heads_and_stick_split_not_merged(self):
        """Heads (8, pos-1) and stick-split (2, pos-2) are separate view dims.

        They must NOT be merged even though 8×2=16 divides the slot dim size
        8192.  The divisibility-only algorithm would merge them; the
        view-dim-at-a-time algorithm must not.
        """
        view = _view([8192, 8, 128], [1024, 128, 1])
        result = _merge_device_dims_for_view(self.STL, view)
        # After merge: [8192-merged, 8-heads, 2-stick-split, 64-stick]
        self.assertEqual(len(result.device_size), 4)
        # The heads dim (8) must still be present.
        self.assertIn(8, list(result.device_size))
        # The stick-split dim (2) must still be present.
        self.assertIn(2, list(result.device_size))


class TestMergeDefaultLayout(unittest.TestCase):
    """Default propagate_layouts order: device_size=[512, 16, 8, 2, 64].

    Here NUM_BLOCKS=512 is at pos-0 and BLOCK_SIZE=16 at pos-1; the merge
    for view ``[-1, 8, 128]`` must collapse those two contiguous dims.
    """

    # Default layout for [512, 16, 8, 128] bfloat16
    STL = _stl([512, 16, 8, 2, 64], [16384, 1024, 128, 64, 1])

    def test_blocks_merged_default_layout(self):
        """Merging pos-0 (NUM_BLOCKS=512) and pos-1 (BLOCK_SIZE=16) -> 8192."""
        view = _view([8192, 8, 128], [1024, 128, 1])
        result = _merge_device_dims_for_view(self.STL, view)
        self.assertEqual(list(result.device_size), [8192, 8, 2, 64])

    def test_rank_reduced_to_4(self):
        """Result must be 4-D (one merge drops one dim)."""
        view = _view([8192, 8, 128], [1024, 128, 1])
        result = _merge_device_dims_for_view(self.STL, view)
        self.assertEqual(len(result.device_size), 4)


class TestMergeFloat32(unittest.TestCase):
    """Float32 sticks hold 32 elements (not 64).

    Layout for [512, 16, 8, 64] float32:
      device_size=[512, 16, 8, 2, 32]  stride_map=[8192, 512, 64, 32, 1]
    View: [-1, 8, 64] => [8192, 8, 64]
    Expected merged: [8192, 8, 2, 32]
    """

    STL = _stl([512, 16, 8, 2, 32], [8192, 512, 64, 32, 1], dtype=_F32)

    def test_float32_blocks_merged(self):
        view = _view([8192, 8, 64], [512, 64, 1], dtype=torch.float32)
        result = _merge_device_dims_for_view(self.STL, view)
        self.assertEqual(list(result.device_size), [8192, 8, 2, 32])

    def test_float32_stick_split_preserved(self):
        """Stick-split dim (2) must survive — it belongs to the head_dim view dim."""
        view = _view([8192, 8, 64], [512, 64, 1], dtype=torch.float32)
        result = _merge_device_dims_for_view(self.STL, view)
        self.assertIn(2, list(result.device_size))


class TestMergeIdentityCases(unittest.TestCase):
    """Cases where no merge is performed — function returns ``stl`` itself."""

    def test_same_rank_view_no_merge(self):
        """View same rank as base: no dims to flatten, stl unchanged."""
        # [8, 128] -> view [8, 128] (identity)
        stl = _stl([8, 2, 64], [128, 64, 1])
        view = _view([8, 128], [128, 1])
        result = _merge_device_dims_for_view(stl, view)
        self.assertIs(result, stl)

    def test_already_merged_layout(self):
        """If the device_size already matches the view dims, no merge needed."""
        # device_size=[8192, 8, 2, 64] — already merged
        stl = _stl([8192, 8, 2, 64], [1024, 128, 64, 1])
        view = _view([8192, 8, 128], [1024, 128, 1])
        result = _merge_device_dims_for_view(stl, view)
        self.assertIs(result, stl)

    def test_non_contiguous_dims_not_merged(self):
        """Dims with a stride gap between them must not be merged.

        If stride_map[outer] != device_size[inner] * stride_map[inner], the
        chain walk stops; no merge happens.
        """
        # Introduce a gap: outer stride is 3000, but inner_size * inner_stride = 2048
        stl = _stl([16, 8, 2, 64], [128, 3000, 64, 1])
        view = _view([128, 128], [128, 1])
        result = _merge_device_dims_for_view(stl, view)
        # No valid chain; stl returned unchanged.
        self.assertIs(result, stl)

    def test_size_1_view_dim_skipped(self):
        """Size-1 view dims are skipped — nothing to merge."""
        stl = _stl([512, 16, 8, 2, 64], [16384, 1024, 128, 64, 1])
        # Leading size-1 dim
        view = _view([1, 8192, 8, 128], [8388608, 1024, 128, 1])
        result = _merge_device_dims_for_view(stl, view)
        # The blocks dims (512, 16) should still merge for the 8192 view dim.
        self.assertEqual(list(result.device_size), [8192, 8, 2, 64])


class TestMergeEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_3_dim_leading_merge(self):
        """Three non-stick dims in a contiguous chain collapse into one.

        Host shape [16, 8, 4, 128] → device_size=[16, 8, 4, 2, 64]
        stride_map=[4096, 512, 128, 64, 1]  (stick-split stride=64, not 16).

        Chain for want=16*8*4=512, sv=128:
          next_stride=128:  hit pos-2 (size=4)  -> product=4,   next=4*128=512
          next_stride=512:  hit pos-1 (size=8)  -> product=32,  next=32*128=4096
          next_stride=4096: hit pos-0 (size=16) -> product=512 == want ✓
          members=[2,1,0] sorted by stride -> rep=pos-2 (smallest), merged_sz=4*8*16=512
        Result: [512, 2, 64]  stride_map=[128, 64, 1]
        """
        stl = _stl([16, 8, 4, 2, 64], [4096, 512, 128, 64, 1])
        view = _view([512, 128], [128, 1])
        result = _merge_device_dims_for_view(stl, view)
        self.assertEqual(list(result.device_size), [512, 2, 64])
        self.assertEqual(result.stride_map[-1], 1)  # stick stride unchanged
        self.assertEqual(result.device_size[-1], 64)  # stick size unchanged

    def test_fully_flat_1d_view(self):
        """A 1-D view collapses all non-stick dims; device size = total_elems / stick_size.

        device_size=[16, 8, 2, 64]  stride_map=[1024, 128, 64, 1]
        Total elements: 16*8*2*64 = 16384.
        View: [16384] with stride=[1] (sv=1).

        Chain for want=16384, sv=1 (starts at product=stick_size=64):
          next_stride=64:  hit pos-2 (size=2)  -> product=128,  next=128
          next_stride=128: hit pos-1 (size=8)  -> product=1024, next=1024
          next_stride=1024: hit pos-0 (size=16) -> product=16384 == want ✓
          members=[pos-2, pos-1, pos-0], merged_sz = 2*8*16 = 256

        The resulting device dim has size 256 (= 16384 / 64 sticks), not 16384.
        Device dims count sticks, not elements; merged_sz is the product of the
        non-stick member sizes, so the stick contribution (64) is not included.
        """
        stl = _stl([16, 8, 2, 64], [1024, 128, 64, 1])
        view = _view([16 * 8 * 2 * 64], [1])
        result = _merge_device_dims_for_view(stl, view)
        # merged non-stick size = 16*8*2 = 256; stick stays as 64
        self.assertEqual(list(result.device_size), [256, 64])
        self.assertEqual(len(result.device_size), 2)

    def test_second_view_dim_has_nothing_to_merge(self):
        """Only the first view dim merges; the second finds no chain to merge.

        (Two view dims that *both* merge are covered by ``TestMergeMultipleGroups``.)

        device_size=[4, 8, 2, 64]  stride_map=[512, 64, 32, 1]
        View: [32, 128]  strides=[64, 1]

        want=32, sv=64:
          next_stride=64: hit pos-1 (size=8) -> product=8, next=8*64=512
          next_stride=512: hit pos-0 (size=4) -> product=32 == want ✓
          members=[pos-1, pos-0]  (sorted by stride: pos-1 first)
          merged_sz = 8*4 = 32

        want=128, sv=1 (after claiming pos-0, pos-1):
          only pos-2 unclaimed (stride=32)
          next_stride=64 (stick_size): no unclaimed dim at stride=64 -> stop
          no merge for this view dim

        Result: [32, 2, 64] (only the block pair merged; stick-split dim stays).
        """
        stl = _stl([4, 8, 2, 64], [512, 64, 32, 1])
        view = _view([32, 128], [64, 1])
        result = _merge_device_dims_for_view(stl, view)
        # pos-0(4) and pos-1(8) merged into 32; pos-2(2) and stick(64) unchanged.
        self.assertEqual(list(result.device_size), [32, 2, 64])


class TestMergeMultipleGroups(unittest.TestCase):
    """A view that flattens more than one group of dims at once.

    Every merge group is computed against the original layout, so applying one
    group must not shift the dims another group refers to.
    """

    def test_slot_and_trailing_dims_both_flatten(self):
        """``[512, 16, 8, 128].view(8192, 1024)``: slots *and* heads x head_dim.

        The slot dim merges the two block dims (512, 16) → 8192.
        The trailing view dim (want=1024, sv=1) seeds from stick_size=64:
          pos-3 (stick-split=2, stride=64) → product=128, next=128
          pos-2 (heads=8, stride=128) → product=1024 == want ✓  → merge [3,2]
        merged_sz = 2*8 = 16.  The stick stays at 64.
        Result: device_size=[8192, 16, 64]  stride_map=[1024, 64, 1]
        """
        view = _view([8192, 1024], [1024, 1])
        for name, stl in (
            ("default", _stl([512, 16, 8, 2, 64], [16384, 1024, 128, 64, 1])),
            ("hardware", _stl([16, 8, 2, 512, 64], [1024, 128, 64, 16384, 1])),
        ):
            with self.subTest(layout=name):
                result = _merge_device_dims_for_view(stl, view)
                self.assertEqual(list(result.device_size), [8192, 16, 64])
                self.assertEqual(list(result.stride_map), [1024, 64, 1])

    def test_two_leading_flattenings(self):
        """``[4, 8, 2, 16, 128].view(32, 32, 128)``: (4, 8) and (2, 16) each merge."""
        stl = _stl([4, 8, 2, 16, 2, 64], [32768, 4096, 2048, 128, 64, 1])
        view = _view([32, 32, 128], [4096, 128, 1])
        result = _merge_device_dims_for_view(stl, view)
        self.assertEqual(list(result.device_size), [32, 32, 2, 64])
        self.assertEqual(list(result.stride_map), [4096, 128, 64, 1])

    def test_stick_size_is_never_changed(self):
        """Whatever merges, the last (stick) dim keeps its size and stride 1."""
        stl = _stl([512, 16, 8, 2, 64], [16384, 1024, 128, 64, 1])
        result = _merge_device_dims_for_view(stl, _view([8192, 1024], [1024, 1]))
        self.assertEqual(result.device_size[-1], 64)
        self.assertEqual(result.stride_map[-1], 1)


class TestMergeSizeOneDims(unittest.TestCase):
    """A head_dim that fits one stick leaves a size-1 stick-split device dim."""

    def test_size_1_stick_split_dim_does_not_block_the_merge(self):
        """``[32, 16, 64].view(-1, 64)`` with device layout ``[32, 16, 1, 64]``.

        The size-1 dim has the same stride (64) as the 16-dim, which would make
        the stride lookup return two hits and stop the chain.  Size-1 dims are
        excluded from the stride lookup to avoid this.
        """
        stl = _stl([32, 16, 1, 64], [1024, 64, 64, 1])
        view = _view([512, 64], [64, 1])
        result = _merge_device_dims_for_view(stl, view)
        self.assertEqual(list(result.device_size), [512, 1, 64])
        self.assertEqual(list(result.stride_map), [64, 64, 1])


if __name__ == "__main__":
    unittest.main()
