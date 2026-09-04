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

"""Regression guard: single-token (seq_len=1) SDPA feeding a projection matmul
aborts the DDL scheduler with a degenerate ``M=1`` bmm geometry.

A single-token decode step runs ``scaled_dot_product_attention`` on a
``q=[B, H, 1, D]`` query and feeds the ``[B, H, 1, D]`` result -- reshaped to
``[B, 1, H*D]`` -- into the output projection ``@ wo``. That trailing matmul has
``M = 1`` (one row), and Inductor fuses it with the SDPA into a single bundle.
The backend then aborts::

    DtException: out_reuse_dim.size() == 1
      deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 960  (getMinParamBmm)
    -> dxp_standalone died with SIGABRT

``out_reuse_dim`` is the set of dims present in both bmm inputs but absent from
the output -- the contracted dims. ``getMinParamBmm`` assumes exactly one. When
``M`` degenerates to 1 the projection bmm's role labels come out as
``inp0=[out, in]``, ``inp1=[out, in, mb]``, ``out=[mb]`` -- it contracts *both*
``in`` and ``out``, so the reuse set has two members and the
single-contraction-dim precondition fails. At ``seq_len=64`` the same op is
generated with a healthy ``out_reuse={in}`` (size 1) and compiles cleanly, so
the defect needs *both* ``M=1`` *and* the SDPA->projection fusion.

The abort is raised inside the ``dxp_standalone`` child process and surfaces to
the caller as a catchable ``InductorError`` (``CalledProcessError`` /
``SIGABRT``), not a crash of the test process -- so this is a safe in-process
``xfail``. It flips to xpass the day the scheduler handles the ``M=1``
projection bmm geometry.

See also torch-spyre#2527, a sibling ``M=1`` single-row decode abort in the same
scheduler (``L3DlOpsScheduler.cpp:1041``, "Expect valid lower and upper bound
parameters") -- same M=1-decode family, a different scheduler precondition.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

DEVICE = "spyre"

# Granite/Qwen-class decode shapes (B=1, H=8 heads, head_dim=128, cache L=576).
_B, _H, _L, _D = 1, 8, 576, 128
_HID = _H * _D
_DT = torch.float16


def _sdpa(q, k, v, m):
    """Bare single-/multi-token attention over the full cache."""
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=m, scale=_D**-0.5, enable_gqa=True
    )


def _sdpa_oproj(q, k, v, m, wo):
    """SDPA followed by the output projection -- the minimal fusion that aborts
    at seq_len=1 (the ``M=1`` degenerate projection bmm)."""
    a = _sdpa(q, k, v, m)
    b, hh, s, dd = a.shape
    return a.transpose(1, 2).reshape(b, s, hh * dd) @ wo


def _sdpa_inputs(seq):
    """CPU inputs for a decode (seq=1) or block (seq=64) step; move to device in
    the test so the CPU tensors double as the golden reference."""
    torch.manual_seed(0xAFFE)
    q = torch.rand(_B, _H, seq, _D, dtype=_DT)
    k = torch.rand(_B, _H, _L, _D, dtype=_DT)
    v = torch.rand(_B, _H, _L, _D, dtype=_DT)
    m = torch.zeros(_B, 1, seq, _L, dtype=_DT)
    return q, k, v, m


class TestSeq1DecodeProjection(TestCase):
    """Pin the single-token SDPA->projection compile contract (M=1 bmm)."""

    def setUp(self):
        # Lazy device init (mirrors test_tensor_layout.py's `x.to("spyre")`).
        torch.zeros(1, dtype=torch.float16).to(DEVICE)

    def test_seq1_sdpa_alone_compiles(self):
        """Control: bare SDPA at seq_len=1 compiles and matches CPU. Proves the
        M=1 attention itself is fine and isolates the defect to the fusion with
        the projection matmul."""
        q, k, v, m = _sdpa_inputs(seq=1)
        want = _sdpa(q, k, v, m)
        got = torch.compile(_sdpa, dynamic=False)(
            q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), m.to(DEVICE)
        ).cpu()
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.float(), want.float(), atol=2e-3, rtol=2e-3)

    def test_seq64_sdpa_oproj_compiles(self):
        """Control: SDPA + output projection at seq_len=64 compiles and matches
        CPU. Proves the fused SDPA->projection is fine when M > 1 and isolates
        the defect to M=1."""
        q, k, v, m = _sdpa_inputs(seq=64)
        wo = torch.rand(_HID, _HID, dtype=_DT)
        want = _sdpa_oproj(q, k, v, m, wo)
        got = torch.compile(_sdpa_oproj, dynamic=False)(
            q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), m.to(DEVICE), wo.to(DEVICE)
        ).cpu()
        self.assertEqual(got.shape, want.shape)
        # o_proj reduces over H*D=1024 fp16 terms (magnitude ~O(256)); a loose
        # rtol distinguishes a correct result from garbage without flaking.
        self.assertEqual(got.float(), want.float(), atol=2e-1, rtol=2e-2)

    @pytest.mark.xfail(
        reason=(
            "torch-spyre: single-token (seq_len=1) SDPA fused with the output "
            "projection makes an M=1 bmm whose output drops both contraction "
            "dims, so getMinParamBmm's out_reuse_dim.size()==1 precondition fails "
            "and dxp_standalone aborts (DtException, L3DlOpsScheduler.cpp:960). "
            "Remove this xfail when the DDL scheduler handles the M=1 projection "
            "bmm geometry. See also torch-spyre#2527 (sibling M=1-decode abort)."
        ),
        strict=False,
    )
    def test_seq1_sdpa_oproj_degenerate_bmm(self):
        """The SAME SDPA->projection fusion at seq_len=1 must compile and match
        CPU. Today the M=1 projection bmm aborts the scheduler
        (out_reuse_dim.size()==1), so this is xfail."""
        q, k, v, m = _sdpa_inputs(seq=1)
        wo = torch.rand(_HID, _HID, dtype=_DT)
        want = _sdpa_oproj(q, k, v, m, wo)
        got = torch.compile(_sdpa_oproj, dynamic=False)(
            q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), m.to(DEVICE), wo.to(DEVICE)
        ).cpu()
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.float(), want.float(), atol=2e-1, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
