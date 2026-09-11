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

"""A staggered dtype conversion whose input is a *slice* of a wider buffer.

Gemma's per-head ``q_norm``/``k_norm`` apply a native FP32 RMSNorm to a slice of
the fused QKV projection output, so the fp16->fp32 upcast at the head of the norm
reads ``W_qkv``-wide rows and writes a narrower dense buffer.  Inheriting the
input buffer's ``device_size``/``stride_map`` across such a conversion makes the
token coordinate ``Mod(a*c0, b)`` with ``a/b = W_out/W_in`` in lowest terms,
which either escapes the coordinate-normalization grammar (``a != 1``) or, worse,
stays inside it while addressing the wrong sticks (``a == 1``).

The ``k`` cases are the regression guard for the silent variant, so they must
assert values -- a compile-only check passes even when every access is misplaced.
"""

import pytest
import torch

from torch_spyre._inductor.constants import DEVICE_NAME

EPS = 1e-6

# (num_q_heads, num_kv_heads, head_dim, hidden), yielding W_q/W_qkv of:
#   gemma-3-1b     1024/1536  = 2/3   (q, hard error)  and 256/1536 = 1/6 (k, silent)
#   gemma-4 global 16384/20480 = 4/5  (q, hard error)  and 2048/20480 = 1/10 (k, silent)
_SHAPES = {
    "gemma3-1b": (4, 1, 256, 1152),
    "gemma4-global": (32, 4, 512, 5376),
}

_TOKENS = 8


def _gemma_rms_norm(x, weight):
    """vLLM ``GemmaRMSNorm.forward_native``: normalize and scale in fp32."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return (x32 * (1.0 + weight.float())).to(x.dtype)


def _build(num_q_heads, num_kv_heads, head_dim, hidden, part):
    w_q = num_q_heads * head_dim
    w_kv = num_kv_heads * head_dim
    heads = num_q_heads if part == "q" else num_kv_heads
    start = 0 if part == "q" else w_q
    width = w_q if part == "q" else w_kv

    def fn(x, w_qkv, weight):
        # One fused fp16 buffer; the norm sees only `width` of its columns.
        qkv = x @ w_qkv
        part_view = qkv[:, start : start + width].reshape(_TOKENS, heads, head_dim)
        return _gemma_rms_norm(part_view, weight)

    args = (
        torch.randn(_TOKENS, hidden, dtype=torch.float16) / 8,
        torch.randn(hidden, w_q + 2 * w_kv, dtype=torch.float16) / 8,
        torch.randn(head_dim, dtype=torch.float16) / 8,
    )
    return fn, args


@pytest.mark.parametrize("shape", list(_SHAPES), ids=list(_SHAPES))
@pytest.mark.parametrize("part", ["q", "k"])
def test_native_rmsnorm_on_qkv_slice(shape, part):
    fn, args = _build(*_SHAPES[shape], part)
    expected = fn(*args)
    got = torch.compile(fn, dynamic=False)(*(a.to(DEVICE_NAME) for a in args))
    torch.testing.assert_close(got.cpu(), expected, rtol=0.05, atol=0.05)
