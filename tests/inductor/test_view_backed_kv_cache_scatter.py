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

"""Regression tests for issue #5125: compiled indexed-write ops on a flattened
view of a paged KV-cache silently drop all updates.

The three write forms (index_put_, index_copy_, advanced indexing) and the
full rank-change matrix across 2-D through 5-D base tensors are covered by
test_scatter_flattened_view.py.  This file keeps only the two cases that are
unique to the paged-cache usage pattern and not exercised elsewhere:

  * test_cloned_view_index_put_passes  — clone-before-write must still work
    after the fix; guards the relayout path that materialises a contiguous
    buffer before the scatter.

  * test_view_backed_index_put_p1      — P=1 single-slot decode-step write
    through a view of a 4-D cache; end-to-end correctness check for the
    single-slot case.
"""

import torch
import torch._dynamo

import torch_spyre  # noqa: F401  -- registers the spyre backend

# ---------------------------------------------------------------------------
# KV-cache geometry matching the issue report exactly
# ---------------------------------------------------------------------------
NUM_BLOCKS = 512
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16

DEVICE = "spyre:0"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run(kernel, cache, values, slots):
    """Compile *kernel* on Spyre and assert it matches the CPU eager reference.

    Inputs are generated on CPU and transferred to device so that the reference
    and compiled paths start from identical bit-patterns.  bfloat16 copies are
    lossless (no rounding), so exact equality is required (atol=0, rtol=0).
    """
    reference = kernel(
        cache.cpu().clone(),
        values.cpu().clone(),
        slots.cpu().clone(),
    )
    torch._dynamo.reset()
    result = torch.compile(kernel, dynamic=False)(cache, values, slots).cpu()
    torch.testing.assert_close(result.float(), reference.float(), atol=0, rtol=0)
    return result, reference


# ===========================================================================
# Tests
# ===========================================================================


class TestViewBackedKVCacheScatter:
    """Issue #5125: unique cases not covered by test_scatter_flattened_view.py."""

    def test_cloned_view_index_put_passes(self):
        """4-D cache -> view(-1, H, D) -> clone() -> index_put_ must pass.

        The reporter confirmed this already works before the fix.  If it
        regresses, the clone is not re-materialising a contiguous layout and
        the relayout path introduced by the fix broke something.
        Note: the returned tensor is the cloned flat buffer, not the 4-D cache.
        """
        rng = torch.Generator()
        rng.manual_seed(42)
        cache = torch.zeros(
            (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE
        ).to(DEVICE)
        values = torch.randn(
            (4, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE, generator=rng
        ).to(DEVICE)
        slots = torch.tensor([47, 63, 80, 95], dtype=torch.int64, device=DEVICE)

        def kernel(cache, values, slots):
            flat = cache.view(-1, NUM_KV_HEADS, HEAD_DIM).clone()
            flat.index_put_((slots,), values)
            return flat

        _run(kernel, cache, values, slots)

    def test_view_backed_index_put_p1(self):
        """P=1 single-slot write through a view of a 4-D cache (decode step).

        Exercises the single-slot index_put_ path end-to-end: a scalar slot
        index writes one row into the flattened view of a small 4-D cache and
        the written slot must contain the expected values after compilation.
        """
        nb, bs = 4, 16
        rng = torch.Generator()
        rng.manual_seed(42)
        cache = torch.zeros((nb, bs, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE).to(DEVICE)
        values = torch.randn(
            (1, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE, generator=rng
        ).to(DEVICE)
        slots = torch.tensor([7], dtype=torch.int64, device=DEVICE)

        def kernel(cache, values, slots):
            cache.view(-1, NUM_KV_HEADS, HEAD_DIM).index_put_((slots,), values)
            return cache

        result, _ = _run(kernel, cache, values, slots)
        flat = result.view(-1, NUM_KV_HEADS, HEAD_DIM)
        assert flat[7].any(), (
            "P=1 view-backed index_put_: slot 7 is all-zero after compiled write"
        )
