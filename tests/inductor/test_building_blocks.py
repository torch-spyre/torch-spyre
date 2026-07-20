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

import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch._inductor.utils import run_and_get_code
from torch_spyre._inductor import spyre_hint  # noqa: F401

from utils_inductor import (
    DEVICE,
    _compile_and_run,
    compare_with_cpu,
    compare_with_pytorch,
)


class TestBuildingBlocks(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_softplus(self):
        # beta * x >= threshold ? x : (log(1 + exp(-abs(beta * x)) + relu(beta * x)
        # Reference: https://github.com/onnx/onnx-mlir/pull/2792
        #
        # TODO: "one" and "minus" should be created inside the function, not passed via parameter
        def softplus(x, beta, threshold, one, minus):
            bx = beta * x
            return torch.where(
                bx >= threshold,
                x,
                torch.log(one + torch.exp(minus * abs(bx))) + F.relu(bx),
            )

        T, D = 128, 64
        beta = 1.0
        threshold = 20.0
        activation = torch.randn(D, T, dtype=torch.float16)

        compare_with_cpu(
            lambda x, beta, threshold, one, minus: softplus(
                x, beta, threshold, one, minus
            ),
            activation,
            torch.full([D, T], beta, dtype=torch.float16),
            torch.full([D, T], threshold, dtype=torch.float16),
            torch.full([D, T], 1.0, dtype=torch.float16),
            torch.full([D, T], -1.0, dtype=torch.float16),
            # aten::where.self is not registered for the Spyre eager dispatch
            run_eager=False,
        )

    def test__simple_attn(self):
        H = 4  # heads per group
        Q = 64  # Q len
        L = 256  # KV len
        D = 128  # head dim
        q = torch.randn(H * Q, D, dtype=torch.float16)
        k = torch.randn(L, D, dtype=torch.float16)
        v = torch.randn(L, D, dtype=torch.float16)
        sm_scale = torch.tensor(1 / (D**0.5), dtype=torch.float16)

        def attn(q, k, v, sm_scale):
            qk = q @ k.transpose(-1, -2).contiguous()
            qk = qk * sm_scale
            p = qk.softmax(dim=-1)
            return p @ v

        compare_with_cpu(
            lambda q, k, v, sm_scale: attn(q, k, v, sm_scale),
            q,
            k,
            v,
            sm_scale.repeat(k.shape[0]),
            # mm on Spyre tensors segfaults in libsenlib without the torch.compile
            # execution context that normally initialises the hardware session
            run_eager=False,
        )

    def test_mlp(self):
        seq_len = 256
        emb_dim = 1024
        x = torch.randn(seq_len, emb_dim, dtype=torch.float16)
        gate_proj_weight = torch.empty(emb_dim, 4 * emb_dim, dtype=torch.float16)
        up_proj_weight = torch.empty(emb_dim, 4 * emb_dim, dtype=torch.float16)
        down_proj_weight = torch.empty(4 * emb_dim, emb_dim, dtype=torch.float16)
        nn.init.kaiming_uniform_(gate_proj_weight)
        nn.init.kaiming_uniform_(up_proj_weight)
        nn.init.kaiming_uniform_(down_proj_weight)

        def mlp(x, gate, up, down):
            gate_out = x @ gate
            up_out = x @ up
            swiglu_out = up_out * F.silu(gate_out)
            out = swiglu_out @ down
            return out

        compare_with_cpu(
            lambda x, g, u, d: mlp(x, g, u, d),
            x,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            cpu_compile=True,
        )

    def test_rms_norm(self):
        F16_EPS = 1e-6
        T = 128
        D = 256

        activation = torch.randn(D, T, dtype=torch.float16)
        weight = torch.randn(D, dtype=torch.float16)

        args = [
            activation,  # [D, T]
            weight.reshape(D, 1)
            .expand(D, T)
            .contiguous(),  # [D, T] # work around on device broadcast limitation
            torch.full([T], F16_EPS, dtype=torch.float16),  # [T,] # broadcasted scalar
            torch.full([T], D, dtype=torch.float16),  # [T,] # broadcasted scalar
        ]

        # NOTE: To work around reduction dimension restriction,
        #       this version performs rms_norm along dim 0
        #       The inputs and the output should be transposed on the host
        def rms_norm(x, weight, eps, d):
            x_sq = x * x
            x_mean_sq = x_sq.mean(dim=0)
            return (
                x  # [D, T]
                * torch.rsqrt(x_mean_sq + eps)[None, :]  # [D, T]
                * weight
            )  # [D, T]

        # Compare with pytorch native implementation
        def pytorch_fn(x, w, eps, d):
            return F.rms_norm(
                x.mT,
                normalized_shape=[
                    D,
                ],
                weight=weight,
                eps=F16_EPS,
            ).mT

        compare_with_pytorch(rms_norm, pytorch_fn, *args)

        # Compare with cpu implementation
        compare_with_cpu(rms_norm, *args, cpu_compile=True)

    def test_flash_attention(self):
        B, H, L, D = 1, 8, 256, 64
        block_size = 128

        Q = torch.randn(B, H, L, D, dtype=torch.float16)
        K = torch.randn(B, H, L, D, dtype=torch.float16)
        V = torch.randn(B, H, L, D, dtype=torch.float16)

        def flash(Q, K, V, block_size):
            output = torch.zeros_like(Q)
            M = torch.full(
                (B, H, L), float("-inf"), device=Q.device, dtype=torch.float16
            )
            denominator = torch.zeros((B, H, L), device=Q.device, dtype=torch.float16)
            scale = 1.0 / math.sqrt(D)

            for start in range(0, L, block_size):
                end = start + block_size
                K_block = K[:, :, start:end, :]
                V_block = V[:, :, start:end, :]
                K_block_T = K_block.transpose(-1, -2).contiguous()

                scores = torch.matmul(Q, K_block_T) * scale  # B, H, L, Block
                scores = scores.transpose(-1, -2).contiguous()  # avoid stick reduction
                block_max = torch.amax(scores, dim=-2)
                max_running = torch.maximum(M, block_max)

                exp_scores = torch.exp(
                    scores - max_running.unsqueeze(-2)
                )  # B, H, Block, L
                correction = torch.exp(M - max_running)

                denominator = denominator * correction + exp_scores.sum(dim=-2)
                output = output * correction.unsqueeze(-1) + torch.bmm(
                    exp_scores.transpose(-1, -2).flatten(0, 1), V_block.flatten(0, 1)
                ).unflatten(0, (B, H))

                M = max_running

            return output / denominator.unsqueeze(-1)

        def sdpa_ref(Q, K, V, block_size):
            return F.scaled_dot_product_attention(Q, K, V)

        compare_with_pytorch(
            flash,
            sdpa_ref,
            Q,
            K,
            V,
            block_size,
            atol=0.1,
            rtol=0.1,
        )

    def test_causal_sdpa_unpadded_kv_no_inf(self):
        """Regression: causal SDPA must not produce inf when seqlen_kv % 64 != 0.

        The flash-attention decomposition tiles the kv dimension into 64-wide
        sticks. When seqlen_kv is not a multiple of 64 the final stick's padding
        lanes are uninitialized; the elementwise exp() over those garbage lanes
        overflows fp16 and poisons the numerator matmul, corrupting the output to
        inf. The fix seeds those lanes to exp(-inf)=0 via SAMV coordinate masking
        (see _POINTWISE_PADDING_MASK_VALUE in
        torch_spyre/_inductor/codegen/superdsc.py).

        Checks (per sequence length): (1) FINITENESS — no inf/nan, the property
        the bug directly violated; (2) ACCURACY — closeness to the CPU reference.
        The chosen tolerance (0.3) is deliberately loose: it comfortably passes
        the correct fp16 result (measured max abs diff <= ~0.11 for these sizes)
        while still catching a corrupted output, which manifests as inf or an
        error of order ~5. A tighter fp16-precision bound would be flaky for a
        reason unrelated to this fix.

        Scope note: only single-stick / multiple-of-64 lengths are checked for
        accuracy. Partial-multi-stick lengths (e.g. S=65) have a separate,
        pre-existing accuracy issue that also affects the dense (non-causal)
        path and is unrelated to this inf fix, so they are excluded from the
        accuracy assertion. S=13 and S=63 read back all-inf before the fix; S=64
        (one full stick) was already correct.
        """
        B, H, D = 1, 8, 128

        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)

        for S in (13, 63, 64):
            q = torch.randn(B, H, S, D, dtype=torch.float16)
            k = torch.randn(B, H, S, D, dtype=torch.float16)
            v = torch.randn(B, H, S, D, dtype=torch.float16)
            out = _compile_and_run(sdpa, (q, k, v), DEVICE)
            # (1) Finiteness — the direct signature of the bug.
            self.assertTrue(
                torch.isfinite(out).all(),
                msg=f"causal SDPA produced non-finite output at seqlen_kv={S} "
                f"(inf={int(torch.isinf(out).sum())}, "
                f"nan={int(torch.isnan(out).sum())})",
            )
            # (2) Accuracy — catches "finite but wrong" regressions. Reuses the
            # already-computed `out` via target= (no recompile). Loose tolerance
            # separates the correct result (<=~0.11) from corruption (~5); see
            # docstring.
            compare_with_pytorch(sdpa, sdpa, q, k, v, atol=0.3, rtol=0.3, target=out)

    def test_refactored_plain_bundle_codegen(self):
        """Pointwise ops fuse into one bundle via the refactored codegen path."""

        def fn(x, y, z):
            # Three separate pointwise ops — the scheduler should fuse them
            # into one FusedSchedulerNode, exercising _codegen_into_kernel.
            a = x + y
            b = a * z
            return b - x

        T, D = 128, 64
        x = torch.randn(T, D, dtype=torch.float16)
        y = torch.randn(T, D, dtype=torch.float16)
        z = torch.randn(T, D, dtype=torch.float16)

        compare_with_cpu(fn, x, y, z, run_eager=False)

    def test_mixed_plain_and_loop_bundle_codegen(self):
        """Plain op + hint-tiled op fuse into one bundle; LoopSpec must appear."""
        from torch_spyre._inductor import spyre_hint as sh

        T, D = 128, 64
        x_cpu = torch.randn(T, D, dtype=torch.float16)

        # Named dims must be set on the device tensor so propagation can map
        # the hint's "T" name to the loop variable at compile time.
        x_dev = x_cpu.to("spyre")
        _pnd.declare_tensor_dim("T", T)
        _pnd.declare_tensor_dim("D", D)
        _pnd.name_tensor_dims(x_dev, ["T", "D"])

        def fn(x):
            # abs is a plain SchedulerNode; neg inside the hint becomes a
            # CountedLoopSchedulerNode.  The two must fuse into one bundle.
            y = torch.abs(x)
            with sh(num_tiles_per_dim={"T": 2}):
                return torch.neg(y)

        cfn = torch.compile(fn)
        spyre_result, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        self.assertIn(
            "LoopSpec(",
            source_codes[0],
            "CountedLoopSchedulerNode must produce a LoopSpec in the bundle",
        )

        cpu_result = fn(x_cpu)
        torch.testing.assert_close(spyre_result.cpu(), cpu_result, atol=1e-3, rtol=1e-3)
