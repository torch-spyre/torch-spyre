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

# Owner(s): ["module: cpp"]

"""Regression guard: pointwise dim_order projection when a gather's value
operand OUTRANKS its output (rank_diff < 0), torch-spyre#3732.

``aten.index`` is a 2-arg pointwise (index tensor + value tensor), so
``compute_layouts`` routes it to ``_multi_arg_pointwise_layouts``, whose
``_is_supported_layout`` projects the output dim_order onto each input with::

    rank_diff = len(output.size) - len(arg.layout.size)   # propagate_layouts.py
    projected_dim_order = [d - rank_diff for d in dim_order if d >= rank_diff]

That was written for BROADCAST (arg rank <= output rank). When the arg
*outranks* the output (rank_diff < 0) the filter admits every entry and shifts
them the wrong way, and the arg's extra leading dims are never added -- so the
projected dim_order has fewer entries than the arg's host rank and the backend
raises "Incompatible host_size and dim_order" at compile time.

The minimal trigger is a rank-6 pointwise producer (a RoPE matmul over
``[B, L, H, 2, 2, hd/2]``) feeding an ``aten.index`` gather whose rank-4 output
is lower-rank than the producer. This is the shape hf-adapters' former Qwen3
"fill" step built, before it dropped the ``source_index`` gather in favour of a
single-token decode (see hf_adapters ``hf_common.kv_cache_update``). The
gather-*before*-RoPE ordering (rank-4 value operand, rank_diff >= 0) is a
bit-exact-equivalent workaround and compiles cleanly -- it is the passing
control here.

``test_gather_from_high_rank_producer`` is xfail until #3732 teaches the
projection to handle rank_diff < 0; it flips to xpass the day the bug is fixed.
The failure today is a compile-time ``InductorError`` (a Python exception, not a
backend abort), so it is safe to keep in the suite.
"""

import pytest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

DEVICE = "spyre"

# Shapes from the Qwen3-0.6B fill step that first surfaced this (B=1, nkv=8,
# head_dim=128, BLOCK_SIZE=64). tok is an arbitrary in-block position to gather.
_B, _NKV, _HD, _SEQ, _TOK = 1, 8, 128, 64, 37
_DT = torch.float16
# fp16 device storage rounds to ~4.9e-4; 2e-3 sits above that without masking a
# wrong-value regression (matches tests/tensor/test_tensor_layout.py).
_ATOL = _RTOL = 2e-3


def _apply_rope_matmul(x, selected_freqs):
    """RoPE-as-matmul (verbatim from hf_adapters.hf_common, inlined to keep this
    test self-contained). The ``sf.mul(...)`` intermediate is rank 6:
    ``[B, L, H, 2, 2, hd/2]``.
    """
    bsz, heads, length, dim = x.shape
    half = dim // 2
    x_ = x.transpose(1, 2).reshape(bsz, length, heads, 2, half)
    sf = selected_freqs[:, :, None, :, :, :]
    out = sf.mul(x_.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


def _gather_after_rope(k_full, freqs, sidx):
    """Minimal #3732 trigger: rank-6 RoPE producer -> rank-4 gather. The gather's
    value operand outranks its output (rank_diff < 0)."""
    return _apply_rope_matmul(k_full, freqs).index_select(2, sidx)


def _gather_before_rope(k_full, freqs, sidx):
    """Workaround: gather first (rank-4 value operand), then RoPE. RoPE is
    per-position, so gathering before or after it is bit-exact equivalent."""
    k1 = k_full.index_select(2, sidx)
    f1 = freqs.index_select(1, sidx)
    return _apply_rope_matmul(k1, f1)


def _inputs():
    torch.manual_seed(0xAFFE)
    k_full = torch.rand(_B, _NKV, _SEQ, _HD, dtype=_DT)
    freqs = torch.rand(_B, _SEQ, 2, 2, _HD // 2, dtype=_DT)
    sidx = torch.tensor([_TOK], dtype=torch.int64)
    return k_full, freqs, sidx


class TestGatherRankProjection(TestCase):
    """Pin the rank_diff < 0 pointwise projection contract (torch-spyre#3732)."""

    def setUp(self):
        # Lazy device init (mirrors test_tensor_layout.py's `x.to("spyre")`).
        torch.zeros(1, dtype=torch.float16).to(DEVICE)

    def test_gather_before_rope_workaround_compiles(self):
        """Control: the gather-before-RoPE ordering (rank-4 value operand,
        rank_diff >= 0) compiles and matches the CPU reference. Proves the
        gather itself is fine and isolates the defect to rank_diff < 0."""
        k_full, freqs, sidx = _inputs()
        want = _gather_before_rope(k_full, freqs, sidx)
        got = torch.compile(_gather_before_rope, dynamic=False)(
            k_full.to(DEVICE), freqs.to(DEVICE), sidx.to(DEVICE)
        ).cpu()
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.float(), want.float(), atol=_ATOL, rtol=_RTOL)

    @pytest.mark.xfail(
        reason=(
            "torch-spyre#3732: _multi_arg_pointwise_layouts._is_supported_layout "
            "projects the output dim_order onto inputs assuming arg rank <= output "
            "rank; when the gather's value operand outranks its output "
            "(rank_diff < 0) it builds a dim_order whose rank no longer matches "
            "host_size and raises 'Incompatible host_size and dim_order' at "
            "compile. Remove this xfail when #3732 handles rank_diff < 0."
        ),
        strict=False,
    )
    def test_gather_from_high_rank_producer(self):
        """The SAME gather fed by a rank-6 RoPE producer (rank_diff < 0) must
        compile and match the CPU reference. Today it raises 'Incompatible
        host_size and dim_order' at compile, so this is xfail."""
        k_full, freqs, sidx = _inputs()
        want = _gather_after_rope(k_full, freqs, sidx)
        got = torch.compile(_gather_after_rope, dynamic=False)(
            k_full.to(DEVICE), freqs.to(DEVICE), sidx.to(DEVICE)
        ).cpu()
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.float(), want.float(), atol=_ATOL, rtol=_RTOL)


if __name__ == "__main__":
    run_tests()
