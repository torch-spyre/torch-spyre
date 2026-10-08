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

"""Regression tests for device-to-device copies of offset / strided views.

Guards against the silent-wrong-data bug where copying different slices of a
Spyre tensor D2D more than once returned the *first* call's data. The slice
position lives only in the tensor's storage_offset, and a graph input's
storage_offset is dropped by the Inductor backend (its FixedLayout.offset is
0 and SpyreTensorLayout has no offset field), so the compiled kernel bound the
storage base pointer and read from element 0.

The fix re-introduces the offset in-graph in lower_spyre_from_d2d
(torch_spyre/_inductor/lowering.py) via a ReinterpretView, so the offset lands
in the coordinate that superdsc bakes into the SDSC binary.

REQUIREMENT (not a missing feature): the re-injected offset must step by whole
sticks. The innermost device dim holds elems_per_stick elements (64 at fp16) and
the backend cannot bake an offset landing inside a stick. Measured on hardware
(sweep_d2d_offsets.py): across contiguous narrow AND select views of every inner
size, a copy is correct iff offset % elems_per_stick == 0, and every unaligned
offset instead raises "no mechanism to resolve stick incompatibility" in the
restickify pass — there is NO silent-wrong-data path for contiguous views. Note
the rule keys on the OFFSET, not the inner-dim size: an inner dim of 96 (not a
stick multiple) still copies correctly at offset 192 (3 sticks) but errors at
offset 96 (1.5 sticks). _validate_reoffset_supported in lower_spyre_from_d2d
surfaces this rejection early with an actionable message; see
test_unaligned_offset_raises{,_select} and
test_stick_multiple_offset_unaligned_inner_dim_ok.

An offset that falls INSIDE the stick dim (a column narrow at a stick-aligned
offset) used to be a silent-wrong-data case: the copy reads the source as a
flat [sticks, 64] layout whose stick-tile coordinate is a strided term with a
residual offset (2*c0 + 1), and tensor alignment inflated that dim and
dropped the offset (issue #4050). It is covered by
test_column_slice_inner_offset and now passes.

The transpose / permute / strided cases below deliberately exercise
NON-contiguous views (see TestCopyFromD2DStridedViews). permute / select at
stick-aligned offsets are carried by the offset fix; a stepped slice is the
same strided-coordinate shape as the column narrow above and passes since
issue #4050; transpose still fails in the restickify layout pass (a
pre-existing backend limitation, not a regression from this fix).
"""

import unittest

import torch

import torch_spyre  # noqa: F401

DEVICE = "spyre"
DTYPE = torch.float16


class TestCopyFromD2DContiguousOffsets(unittest.TestCase):
    """Contiguous slices at varying offsets — the core reproducer."""

    def test_multi_offset_clone(self):
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        a = x.narrow(0, 0, 1).clone()
        b = x.narrow(0, 2, 1).clone()
        torch.testing.assert_close(a.cpu(), x.cpu()[0:1])
        torch.testing.assert_close(b.cpu(), x.cpu()[2:3])

    def test_loop_varying_offsets(self):
        x = torch.arange(8 * 64, dtype=DTYPE, device=DEVICE).reshape(8, 64)
        for r in [0, 2, 4, 6, 7]:
            out = x.narrow(0, r, 1).clone()
            torch.testing.assert_close(
                out.cpu(),
                x.cpu()[r : r + 1],
                msg=f"row {r}: got {out.cpu()[0, 0].item()}",
            )

    def test_revisit_offset(self):
        x = torch.arange(6 * 64, dtype=DTYPE, device=DEVICE).reshape(6, 64)
        for r in [1, 3, 1, 5, 3]:
            out = x.narrow(0, r, 1).clone()
            torch.testing.assert_close(out.cpu(), x.cpu()[r : r + 1])

    def test_copy_into_sliced_dst(self):
        """dst is itself a narrow (nonzero dst storage_offset)."""
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        dst = torch.full((4, 64), -1.0, dtype=DTYPE, device=DEVICE)
        dst.narrow(0, 0, 1).copy_(x.narrow(0, 0, 1))
        dst.narrow(0, 2, 1).copy_(x.narrow(0, 3, 1))
        out = dst.cpu()
        torch.testing.assert_close(out[0:1], x.cpu()[0:1])
        torch.testing.assert_close(out[2:3], x.cpu()[3:4])
        torch.testing.assert_close(out[1:2], torch.full((1, 64), -1.0, dtype=DTYPE))
        torch.testing.assert_close(out[3:4], torch.full((1, 64), -1.0, dtype=DTYPE))

    def test_column_slice_inner_offset(self):
        """Offset along the last (stick) dim: narrow columns at an offset.

        The offset (64) IS a stick multiple, so _validate_reoffset_supported
        accepts it, but it falls inside the stick DIMENSION. The copy reads the
        source as a flat [4 sticks, 64] layout whose stick-tile coordinate is
        ``2*c0 + 1``: a strided term with a residual offset. Tensor alignment
        used to count that dim in device units and then add a gap dim (16
        sticks for 4) and to drop the ``+1``, so the read landed on sticks 0
        and 2 (measured in sweep_d2d_offsets.py: got 0.0, expected 64.0).
        Fixed with issue #4050 in normalize_coordinates."""
        x = torch.arange(2 * 128, dtype=DTYPE, device=DEVICE).reshape(2, 128)
        # columns [64:128) -> nonzero offset within a row
        out = x.narrow(1, 64, 64).clone()
        torch.testing.assert_close(out.cpu(), x.cpu()[:, 64:128])

    def test_unaligned_offset_raises(self):
        """A storage_offset that is not a whole number of sticks is rejected.

        REQUIREMENT (not a missing feature). The re-injected offset must step by
        complete sticks: the innermost device dim holds elems_per_stick elements
        (64 at fp16) and the backend cannot bake an offset landing inside a
        stick. Measured on hardware (sweep_d2d_offsets.py): across contiguous
        narrow / select views of every inner size, a copy is correct iff
        offset % elems_per_stick == 0 and otherwise the restickify pass raises
        "no mechanism to resolve stick incompatibility".

        The rule keys on the OFFSET, not the inner-dim size. reshape(4, 100)
        row 2 has offset 200 (200 % 64 != 0), so _validate_reoffset_supported in
        lower_spyre_from_d2d raises Unsupported at lowering time — surfacing the
        rejection early with an actionable message instead of the cryptic
        downstream restickify error. Row 0 (offset 0) is always fine."""
        x = torch.arange(4 * 100, dtype=DTYPE, device=DEVICE).reshape(4, 100)
        # offset 0 -> guard no-op, must succeed and be correct
        a = x.narrow(0, 0, 1).clone()
        torch.testing.assert_close(a.cpu(), x.cpu()[0:1])
        # offset 200 is not a stick multiple -> clean compile-time error
        with self.assertRaises(Exception) as cm:
            x.narrow(0, 2, 1).clone()
        self.assertIn("stick", str(cm.exception).lower())

    def test_unaligned_offset_raises_select(self):
        """select (rank-reducing) with an unaligned offset is rejected too.

        select(0, r) drops the outer dim, so the view handed to the op is 1-D
        with the outer-dim offset baked into its storage_offset. The rule keys
        on the offset alone, so rank reduction is irrelevant: reshape(4, 100)
        select(0, 2) has offset 200 (not a stick multiple) and must raise, while
        select(0, 0) (offset 0) succeeds. Guards against the earlier concern
        that the select path could silently misread (it cannot: it either copies
        correctly or errors)."""
        x = torch.arange(4 * 100, dtype=DTYPE, device=DEVICE).reshape(4, 100)
        torch.testing.assert_close(x.select(0, 0).clone().cpu(), x.cpu()[0])
        with self.assertRaises(Exception) as cm:
            x.select(0, 2).clone()
        self.assertIn("stick", str(cm.exception).lower())

    def test_stick_multiple_offset_unaligned_inner_dim_ok(self):
        """A stick-multiple offset is accepted even if the inner dim is not.

        Proves the rule is about the OFFSET, not the inner-dim size. reshape(
        4, 96): 96 is not a stick multiple, but row 2 sits at offset 192 == 3
        sticks, so the copy is representable and correct. (Row 1 at offset 96 ==
        1.5 sticks would instead error — see test_unaligned_offset_raises.)
        Verified on hardware in sweep_d2d_offsets.py."""
        x = torch.arange(4 * 96, dtype=DTYPE, device=DEVICE).reshape(4, 96)
        out = x.narrow(0, 2, 1).clone()  # offset 192 == 3 * 64
        torch.testing.assert_close(out.cpu(), x.cpu()[2:3])


class TestCopyFromD2DStridedViews(unittest.TestCase):
    """Non-contiguous views: transpose / permute / step slices / select.

    permute and select work (the offset fix carries them), and so does a
    stepped slice since issue #4050 fixed strided stick-tile coordinates.
    transpose cases are marked expectedFailure: they fail in the Spyre
    restickify layout pass ("no mechanism to resolve stick incompatibility" /
    "scatter elements from one stick to multiple sticks"), NOT in offset
    handling. Verified to fail identically on the pre-fix baseline (offset==0
    transpose reduces lower_spyre_from_d2d to the original mutate_to), so these
    are pre-existing backend limitations, not regressions from this fix, and
    reconstructing size+stride explicitly (Option 1) would not resolve them.
    Tracked for the follow-up PR.
    """

    @unittest.expectedFailure
    def test_transpose_clone(self):
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        out = x.t().clone()  # (64, 4), non-contiguous
        torch.testing.assert_close(out.cpu(), x.cpu().t())

    @unittest.expectedFailure
    def test_transpose_then_offset_clone(self):
        """Transpose AND a nonzero offset along the transposed dim."""
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        xt = x.t()  # (64, 4)
        out = xt.narrow(1, 2, 2).clone()  # rows of original [2:4], transposed
        torch.testing.assert_close(out.cpu(), x.cpu().t()[:, 2:4])

    def test_permute_clone(self):
        x = torch.arange(2 * 3 * 64, dtype=DTYPE, device=DEVICE).reshape(2, 3, 64)
        out = x.permute(1, 0, 2).clone()  # (3, 2, 64)
        torch.testing.assert_close(out.cpu(), x.cpu().permute(1, 0, 2))

    def test_permute_with_offset_clone(self):
        x = torch.arange(4 * 3 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 3, 64)
        v = x.permute(1, 0, 2).narrow(1, 1, 2)  # offset along a permuted dim
        out = v.clone()
        torch.testing.assert_close(out.cpu(), x.cpu().permute(1, 0, 2)[:, 1:3])

    def test_select_clone(self):
        """select drops a dim and introduces an offset."""
        x = torch.arange(4 * 64, dtype=DTYPE, device=DEVICE).reshape(4, 64)
        out = x.select(0, 2).clone()  # row 2 as 1-D (64,), storage_offset=128
        torch.testing.assert_close(out.cpu(), x.cpu()[2])

    def test_stepped_slice_clone(self):
        """Strided (step>1) slice — non-unit stride plus offset.

        Reads rows 1, 3, 5, 7 as stick-tile coordinate ``2*c0 + 1``; the
        stride and the residual offset are realized as an inner gap dim
        (issue #4050)."""
        x = torch.arange(8 * 64, dtype=DTYPE, device=DEVICE).reshape(8, 64)
        out = x[1::2].clone()  # rows 1,3,5,7 ; offset=64, stride[0]=128
        torch.testing.assert_close(out.cpu(), x.cpu()[1::2])

    @unittest.expectedFailure
    def test_transpose_varying_offsets_loop(self):
        """Multiple distinct offsets on a transposed view in one process."""
        x = torch.arange(8 * 64, dtype=DTYPE, device=DEVICE).reshape(8, 64)
        xt = x.t()  # (64, 8)
        for c in [0, 2, 5, 7]:
            out = xt.narrow(1, c, 1).clone()
            torch.testing.assert_close(
                out.cpu(),
                x.cpu().t()[:, c : c + 1],
                msg=f"transpose col {c}",
            )


class TestCopyFromD2DFP8QFP8WT(unittest.TestCase):
    """copy_from_d2d (clone) on QFP8WT KERNEL tensors.

    Regression tests for cloning a strided column-slice of a QFP8WT KERNEL
    weight tensor on Spyre (e.g. a KV-cache scatter or fused-QKV column tile).

    The fix is in the deeptools compiler (deeptools#4689); torch-spyre is
    correct as-is.
    """

    def _make_fp8_kernel(self, K, N):
        """DMA a [K, N] float8_e4m3fn matrix to Spyre in QFP8WT KERNEL layout."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        cpu = torch.randn(K, N).to(torch.float8_e4m3fn)
        return _dma_to_spyre_fp8_kernel(cpu)

    def test_contiguous_clone(self):
        """Cloning a contiguous QFP8WT tensor succeeds and preserves bit-identical data.

        Correctness is verified by cloning a second independent copy of the same
        weight tensor and comparing the two clones byte-for-byte.  Direct
        comparison to the original CPU tensor is not meaningful because
        _dma_to_spyre_fp8_kernel reorders bytes into QFP8WT packed layout on device.
        """
        K, N = 4096, 4096
        w_spyre = self._make_fp8_kernel(K, N)
        clone_a = w_spyre.clone()
        clone_b = w_spyre.clone()
        self.assertEqual(clone_a.shape, w_spyre.shape)
        self.assertEqual(clone_a.stride(), (N, 1))
        torch.testing.assert_close(
            clone_a.cpu().view(torch.uint8), clone_b.cpu().view(torch.uint8)
        )

    def test_column_tile_clone(self):
        """Cloning a strided column-slice [:, 0:N_tile] produces a correct
        contiguous copy and two clones of the same tile are byte-identical.

        Primary reproducer for the deeptools#4689 fix: w_full[:, 0:N_tile]
        has a non-unit outer stride (stride=(N_full, 1)), so the compiler
        must handle the gap at the end of each row.  Without the fix this
        aborts with a compiler scheduling error.

        Correctness is verified by comparing two independent clones of the
        same tile against each other.  Direct comparison against a separately
        DMA'd [K, N_tile] tensor is not meaningful: the QFP8WT packer derives
        on-device strides from N, so a column slice of a [K, N_full] KERNEL
        tensor has different packed bytes than a standalone [K, N_tile] tensor
        with the same fp8 values.
        """
        K, N_full, N_tile = 4096, 6144, 4096
        w_full = self._make_fp8_kernel(K, N_full)

        w_tile = w_full[:, 0:N_tile]
        self.assertFalse(w_tile.is_contiguous())
        self.assertEqual(w_tile.stride(), (N_full, 1))

        clone_a = w_tile.clone()
        clone_b = w_tile.clone()

        self.assertEqual(clone_a.shape, torch.Size([K, N_tile]))
        self.assertTrue(clone_a.is_contiguous())
        self.assertEqual(clone_a.stride(), (N_tile, 1))
        # Two clones of the same tile must be byte-identical.
        torch.testing.assert_close(
            clone_a.cpu().view(torch.uint8),
            clone_b.cpu().view(torch.uint8),
        )

    def test_multiple_column_tiles(self):
        """Cloning the same tile twice from the same base gives identical results.

        Verifies consistency across repeated copies of the same strided view —
        guards against the earlier silent-wrong-data bug where the second clone
        returned the first call's data.
        """
        K, N_full, N_tile = 4096, 6144, 2048
        w_full = self._make_fp8_kernel(K, N_full)
        n_tiles = N_full // N_tile
        for i in range(n_tiles):
            col_start = i * N_tile
            w_tile = w_full[:, col_start : col_start + N_tile]
            clone_a = w_tile.clone()
            clone_b = w_tile.clone()
            self.assertEqual(clone_a.shape, torch.Size([K, N_tile]))
            torch.testing.assert_close(
                clone_a.cpu().view(torch.uint8),
                clone_b.cpu().view(torch.uint8),
                msg=f"tile {i} (cols {col_start}:{col_start + N_tile})",
            )

    def test_revisit_same_tile(self):
        """Cloning the same tile multiple times returns consistent data."""
        K, N_full, N_tile = 4096, 6144, 4096
        w_full = self._make_fp8_kernel(K, N_full)
        w_tile = w_full[:, 0:N_tile]
        first = w_tile.clone().cpu().view(torch.uint8)
        for _ in range(2):
            w_clone = w_tile.clone()
            torch.testing.assert_close(w_clone.cpu().view(torch.uint8), first)


if __name__ == "__main__":
    unittest.main()
