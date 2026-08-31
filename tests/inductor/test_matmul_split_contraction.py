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

"""A matmul whose activation is a view splitting the contraction across two dims.

``lower_mm``/``lower_bmm`` give the reduction a single K range, but the read goes
through the activation's loader. When that activation is a view over a differently
shaped buffer -- a multi-head attention output arriving as ``[.., heads, head_dim]``
-- the emitted index spans two dims of the base, the kernel's iteration space
inherits two contraction symbols, and the backend scheduler rejects it
(``getMinParamBmm`` DT_CHECKs exactly one; see ``check_bmm_dim_roles``).

The shape that bites is ordinary decode-shaped attention: at ``seqlen_q == 1`` the
``transpose(1, 2).reshape(1, 1, heads * head_dim)`` is layout-free, so nothing
materialises it and the split survives. At ``seqlen_q > 1`` the same reshape must
materialise, which collapses K and the check passes -- which is why this is
seqlen-specific and easy to miss.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_matmul_split_contraction.py -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.test_case import TestCase as InductorTestCase

HEAD_DIM = 256
CACHE = 1088


class _HeadRMSNorm(nn.Module):
    """Per-head RMSNorm, the shape that re-exposes [heads, head_dim] around a matmul."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def _attention_then_projection(heads: int, seqlen_q: int, dtype, device):
    """Build (fn, args) for attention -> per-head norm -> output projection."""
    torch.manual_seed(0)
    features = heads * HEAD_DIM
    o_proj = nn.Linear(features, features, bias=False).to(dtype).to(device)
    norm = _HeadRMSNorm(HEAD_DIM).to(dtype).to(device)

    query = torch.randn(1, heads, seqlen_q, HEAD_DIM, dtype=dtype).to(device)
    key = torch.randn(1, 1, CACHE, HEAD_DIM, dtype=dtype).to(device)
    value = torch.randn(1, 1, CACHE, HEAD_DIM, dtype=dtype).to(device)
    mask = torch.zeros(1, 1, seqlen_q, CACHE, dtype=dtype).to(device)

    def fn(query, key, value, mask):
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=1.0, enable_gqa=True
        )
        flat = attended.transpose(1, 2).reshape(1, seqlen_q, -1)
        normed = norm(flat.view(1, seqlen_q, heads, HEAD_DIM)).view(1, seqlen_q, -1)
        return o_proj(normed)

    return fn, (query, key, value, mask)


class TestMatmulSplitContraction(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _run_on_spyre(self, heads, seqlen_q):
        fn, args = _attention_then_projection(heads, seqlen_q, torch.float16, "spyre")
        with torch.no_grad():
            return torch.compile(fn, dynamic=False)(*args).to("cpu").float()

    def test_single_query_row_multi_head_compiles(self):
        """The case that failed: 4 heads, seqlen_q=1, K split as 4 x 256."""
        out = self._run_on_spyre(heads=4, seqlen_q=1)
        self.assertEqual(tuple(out.shape), (1, 1, 4 * HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all())

    def test_single_query_row_multi_head_is_numerically_right(self):
        """Compiling is not enough -- a copy inserted on the wrong axis would still
        compile and would silently permute the contraction."""
        heads, seqlen_q = 4, 1
        fn32, args32 = _attention_then_projection(heads, seqlen_q, torch.float32, "cpu")
        with torch.no_grad():
            expected = fn32(*args32)
        actual = self._run_on_spyre(heads=heads, seqlen_q=seqlen_q)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)

    def test_multi_row_query_still_compiles(self):
        """The path that already worked must keep working (no copy needed there)."""
        out = self._run_on_spyre(heads=4, seqlen_q=64)
        self.assertEqual(tuple(out.shape), (1, 64, 4 * HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all())

    def test_single_head_single_row_still_compiles(self):
        """One head leaves nothing to split; it compiled before this fix too."""
        out = self._run_on_spyre(heads=1, seqlen_q=1)
        self.assertEqual(tuple(out.shape), (1, 1, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
