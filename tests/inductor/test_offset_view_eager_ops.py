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

"""Regression tests: nonzero-offset views through the compiled eager-op path.

Background. A graph input's ``storage_offset`` is dropped by the Inductor
backend (FixedLayout.offset is 0 and SpyreTensorLayout has no offset field), so a
standalone-compiled kernel binds the storage BASE pointer and touches element 0
regardless of the view's true offset — for BOTH reads and writes. Only
``spyre::copy_from_d2d`` re-injects the offset in-graph. The eager-op dispatcher
therefore makes nonzero-offset args safe by routing them through that same path:
read args are cloned to a fresh offset-0 buffer before the kernel; write args
(from ``alias_info.is_write``) get a clone -> run -> ``copy_``-back
read-modify-write so the mutation lands at the destination's real offset.

Real-world trigger. Splitting a fused QKV projection with
``qkv[..., a:b]`` is a no-op at decode (leading dim 1): the column
slice is already row-contiguous, so it returns a VIEW with ``storage_offset==a``
and any on-device consumer (RoPE, an in-place norm, ...) would otherwise read /
write element 0.

Offsets here are stick-aligned (Granite-3.3 QKV: k.off=4096=64 sticks,
v.off=5120=80 sticks at fp16, elems_per_stick=64); unaligned offsets are a
separate concern covered by ``test_copy_from_d2d_offsets.py``.
"""

import pytest
import torch

import torch_spyre  # noqa: F401


DEVICE = "spyre"
DTYPE = torch.float16

# Granite-3.3-8b QKV geometry: 32 q heads + 8 kv + 8 kv, head_size 128.
HEAD = 128
Q = 32 * HEAD  # 4096, the k-slice offset
KV = 8 * HEAD  # 1024
TOTAL = Q + KV + KV  # 6144
ATOL = 0.5


@pytest.mark.parametrize("T", [1, 2, 32])
class TestOffsetViewReads:
    """A nonzero-offset view fed to a compiled eager op must read the right
    slice, not the storage base (element 0)."""

    def _k_slice(self, T):
        """Return (device k-view, cpu qkv). At T=1 the column slice is
        row-contiguous so this view carries storage_offset==Q; the shape
        (32, T, TOTAL) keeps the inner slice strided at all T."""
        torch.manual_seed(0)
        qkv = torch.randn(32, T, TOTAL, dtype=DTYPE)
        return qkv.to(DEVICE)[..., Q : Q + KV], qkv

    def test_reduction_reads_correct_slice(self, T):
        """aten.sum over the head dim of the offset k-slice."""
        k, qkv = self._k_slice(T)
        got = k.sum(dim=-1).to("cpu")
        ref = qkv[..., Q : Q + KV].sum(dim=-1)
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)

    def test_elementwise_reads_correct_slice(self, T):
        """aten.mul on the offset k-slice."""
        k, qkv = self._k_slice(T)
        got = (k * 2.0).to("cpu")
        ref = qkv[..., Q : Q + KV] * 2.0
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)


@pytest.mark.parametrize("T", [1, 2, 32])
class TestOffsetViewInPlaceWrites:
    """An in-place op whose destination is a nonzero-offset view must mutate
    that view at its real offset and leave the rest of the base tensor
    untouched. All comparisons are over the WHOLE base tensor so a write that
    spills to element 0 is caught."""

    def test_relu_writes_at_offset(self, T):
        """relu_ directly on the offset k-slice."""
        torch.manual_seed(1)
        qkv = torch.randn(32, T, TOTAL, dtype=DTYPE)
        dev = qkv.to(DEVICE)
        dev[..., Q : Q + KV].relu_()

        ref = qkv.clone()
        ref[..., Q : Q + KV] = torch.relu(qkv[..., Q : Q + KV])
        torch.testing.assert_close(
            dev.to("cpu").float(), ref.float(), atol=ATOL, rtol=0
        )

    def test_add_accumulates_at_offset(self, T):
        """add_ into the offset k-slice: read-modify-write must read the
        destination's CURRENT value first (an accumulating op)."""
        torch.manual_seed(2)
        qkv = torch.randn(32, T, TOTAL, dtype=DTYPE)
        addend = torch.randn(32, T, KV, dtype=DTYPE)
        dev = qkv.to(DEVICE)
        dev[..., Q : Q + KV].add_(addend.to(DEVICE))

        ref = qkv.clone()
        ref[..., Q : Q + KV] = qkv[..., Q : Q + KV] + addend
        torch.testing.assert_close(
            dev.to("cpu").float(), ref.float(), atol=ATOL, rtol=0
        )


@pytest.mark.parametrize("T", [1, 2, 32])
class TestOffsetViewListArgs:
    """Ops that take their tensors via a ``List[Tensor]`` schema arg
    (``aten.cat``/``aten.stack``) must materialize every offset view nested in
    the list, not just bare-tensor args. Otherwise each nested view is read
    from element 0 of its storage instead of its real ``storage_offset``."""

    def _kv_slices(self, T):
        """Return (device k-view, device v-view, cpu qkv). Both slices carry a
        nonzero storage_offset (Q and Q+KV) at T=1 and stay strided at T>1."""
        torch.manual_seed(3)
        qkv = torch.randn(32, T, TOTAL, dtype=DTYPE)
        dev = qkv.to(DEVICE)
        return dev[..., Q : Q + KV], dev[..., Q + KV : TOTAL], qkv

    def test_cat_reads_correct_slices(self, T):
        """aten.cat over two offset views along the last dim."""
        k, v, qkv = self._kv_slices(T)
        got = torch.cat([k, v], dim=-1).to("cpu")
        ref = torch.cat([qkv[..., Q : Q + KV], qkv[..., Q + KV : TOTAL]], dim=-1)
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)

    def test_stack_reads_correct_slices(self, T):
        """aten.stack over two offset views along a new dim."""
        k, v, qkv = self._kv_slices(T)
        got = torch.stack([k, v], dim=0).to("cpu")
        ref = torch.stack([qkv[..., Q : Q + KV], qkv[..., Q + KV : TOTAL]], dim=0)
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)


class TestPrefixViewSliceWrite:
    """Eager d2d slice writes from a PREFIX VIEW must honor the view's size.

    Regression test for #3826. ``copy_from_d2d`` is compiled through
    ``compile_once``; without ``dynamic=False`` there, dynamo's auto-dynamic
    promoted the source's row dim to a symbol after repeated calls at distinct
    lengths, and the Spyre lowering silently baked ONE concrete extent into the
    reused graph -- so a later write copied the frozen extent (the base's 64
    rows on the reporting stack), overrunning the destination window. Found in
    the wild as attention write-back corrupting other sequences in a batch.

    The ladder below uses several distinct lengths on purpose: the first call
    is always correct (fresh static trace) and the corruption only appears once
    auto-dynamic would have kicked in, so a single-length test cannot regress
    this.
    """

    def test_prefix_view_write_honors_view_extent_across_lengths(self):
        torch.manual_seed(0)
        for qlen in (16, 32, 48, 24, 40):
            src_cpu = torch.randn(64, 32, 128, dtype=torch.float16)
            dst = torch.zeros(96, 32, 128, dtype=torch.float16).to("spyre")
            dst[32 : 32 + qlen] = src_cpu.to("spyre")[:qlen]
            got = dst.to("cpu")
            outside = torch.ones(96, dtype=torch.bool)
            outside[32 : 32 + qlen] = False
            assert torch.all(got[outside] == 0), (
                f"qlen={qlen}: write escaped the destination window "
                f"(view extent ignored)"
            )
            torch.testing.assert_close(
                got[32 : 32 + qlen].float(),
                src_cpu[:qlen].float(),
                atol=1e-2,
                rtol=0,
            )
