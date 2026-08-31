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

import dataclasses
import math
import unittest
from unittest import mock
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch._inductor.utils import run_and_get_code
from torch_spyre._inductor import spyre_hint  # noqa: F401
from torch_spyre._inductor import config

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

    def test_residual_rms_norm_fp32_upcast(self):
        # Regression for the mixed-EA layout gate in
        # _multi_arg_pointwise_layouts. A residual add feeding an fp32-upcast
        # RMSNorm: the add is a computed buffer that gives the upcast -- and
        # thus the reduction-broadcast operand (rsqrt) -- multiple layout
        # candidates, one with its device stick on a non-broadcast axis. The old
        # gate rejected the broadcast mul if ANY candidate was non-broadcast; it
        # now prunes to the broadcast candidate (case 3.1). WITHOUT the residual
        # add the operand got only the broadcast candidate and compiled, so the
        # residual add is essential to the repro.
        B, S, H = 1, 64, 1024
        eps = 1e-6
        hidden = torch.randn(B, S, H, dtype=torch.float16)
        residual = torch.randn(B, S, H, dtype=torch.float16)
        weight = torch.randn(H, dtype=torch.float16)

        def residual_rms_norm(hidden, residual, weight):
            x = (hidden + residual).to(torch.float32)
            var = x.pow(2).mean(-1, keepdim=True)
            normed = x * torch.rsqrt(var + eps)
            return weight * normed.to(torch.float16)

        # Spyre eager mishandles the fp32-upcast staggered layout (a separate,
        # pre-existing issue), so validate the compiled path only.
        compare_with_cpu(
            residual_rms_norm,
            hidden,
            residual,
            weight,
            cpu_compile=False,
            run_eager=False,
        )

    def test_chained_rms_norm_fp32_upcast(self):
        B, S, H = 1, 64, 2816
        eps = 1e-6
        residual = torch.randn(B, S, H, dtype=torch.float16) / 4
        dense = torch.randn(B, S, H, dtype=torch.float16) / 4
        moe = torch.randn(S, H, dtype=torch.float16) / 4
        weight1 = torch.randn(H, dtype=torch.float16) / 4
        weight2 = torch.randn(H, dtype=torch.float16) / 4
        scale = torch.tensor([0.5], dtype=torch.float16)

        def rms_norm(x, weight):
            x32 = x.to(torch.float32)
            var = x32.pow(2).mean(-1, keepdim=True)
            return (x32 * torch.rsqrt(var + eps)).to(x.dtype) * weight

        def chained_rms_norm(residual, dense, moe, weight1, weight2, scale):
            moe_out = rms_norm(moe, weight1).reshape_as(dense)
            ffn_out = rms_norm(dense + moe_out, weight2)
            return (residual + ffn_out) * scale

        compare_with_cpu(
            chained_rms_norm,
            residual,
            dense,
            moe,
            weight1,
            weight2,
            scale,
            cpu_compile=False,
            run_eager=False,
        )

    def test_mixed_ea_staggered_broadcaster_fp16(self):
        # Case 3.2 of the mixed-EA rule: the *staggered* operand is the
        # size-1-stick broadcaster (fp16 produced by an fp32->fp16 downcast,
        # FP32_TO_DL16) combined with a STANDARD full operand. A broadcastable
        # staggered operand is physically identical to STANDARD of that shape, so
        # the op is allowed (STANDARD output).
        x = torch.randn(4, 1, dtype=torch.float32)  # -> .to(f16): staggered bcast
        w = torch.randn(4, 64, dtype=torch.float16)  # STANDARD full

        def fn(x, w):
            return torch.add(x.to(torch.float16), w)

        compare_with_cpu(fn, x, w, cpu_compile=False, run_eager=False)

    def test_mixed_ea_staggered_broadcaster_fp32(self):
        # Case 3.2 with an fp32-physical staggered broadcaster (DL16_TO_FP32).
        # The mixed-EA gate ALLOWS it (physically the equivalent all-STANDARD fp32
        # broadcast), but the codegen doesn't yet emit an fp32 broadcast along
        # the stick axis. The same crash hits a pure-STANDARD fp32
        # [4,1]+[4,64] broadcast, so it is a separate, pre-existing codegen gap
        # tracked in https://github.com/torch-spyre/torch-spyre/issues/4132.
        #
        # We assert the failure originates in *codegen*, not the mixed-EA layout
        # gate: a plain @unittest.expectedFailure would also stay green if a future
        # change re-tightened the gate and raised `Unsupported` before codegen,
        # masking a regression of the path this test guards. So we require the
        # error to be a codegen failure and NOT the gate's "mixed EA"
        # Unsupported. Flip this to a compare_with_cpu once codegen lands.
        x = torch.randn(4, 1, dtype=torch.float16)  # -> .to(f32): staggered bcast
        w = torch.randn(4, 64, dtype=torch.float32)  # STANDARD full

        def fn(x, w):
            return torch.add(x.to(torch.float32), w)

        with self.assertRaises(Exception) as ctx:
            compare_with_cpu(fn, x, w, cpu_compile=False, run_eager=False)
        msg = str(ctx.exception)
        self.assertNotIn(
            "Multi-arg pointwise with mixed EA",
            msg,
            f"expected a codegen failure, but the mixed-EA gate rejected it: {msg}",
        )
        self.assertTrue(
            any(k in msg for k in ("dxp_standalone", "ddc", "sbf-")),
            f"expected a ddc/dxp codegen-stage failure, got: {msg[:300]}",
        )

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

    def test_tiled_symbol_trip_counts_populated_via_compile(self):
        """OpSpec.tiled_symbol_trip_counts reflects each tiled symbol's own
        loop trip count after a real coarse-tiled compilation.
        """
        from torch_spyre._inductor import spyre_kernel, spyre_hint as sh

        T, D = 128, 64
        x_cpu = torch.randn(T, D, dtype=torch.float16)

        # Named dims must be set on the device tensor so propagation can map
        # the hint's "T" name to the loop variable at compile time.
        x_dev = x_cpu.to("spyre")
        _pnd.declare_tensor_dim("T", T)
        _pnd.declare_tensor_dim("D", D)
        _pnd.name_tensor_dims(x_dev, ["T", "D"])

        def fn(x):
            with sh(num_tiles_per_dim={"T": 2}):
                return x + x

        captured_op_specs = []
        original_create_op_spec = spyre_kernel.SpyreKernel.create_op_spec

        def _capturing_create_op_spec(self, *args, **kwargs):
            op_spec = original_create_op_spec(self, *args, **kwargs)
            captured_op_specs.append(op_spec)
            return op_spec

        with mock.patch.object(
            spyre_kernel.SpyreKernel,
            "create_op_spec",
            _capturing_create_op_spec,
        ):
            compiled = torch.compile(fn)
            compiled(x_dev)

        tiled_specs = [s for s in captured_op_specs if s.tiled_symbols]
        self.assertTrue(tiled_specs, "expected at least one tiled OpSpec")
        op_spec = tiled_specs[0]
        tiled_syms = {s for level in op_spec.tiled_symbols for s in level}
        self.assertTrue(tiled_syms.issubset(op_spec.tiled_symbol_trip_counts.keys()))
        for sym in tiled_syms:
            self.assertEqual(op_spec.tiled_symbol_trip_counts[sym], 2)


FrontendPoolAllocationTestBuildingBlocks = config.patch(
    {"frontend_pool_allocation": True}
)(type("FrontendPoolAllocationTestBuildingBlocks", (TestBuildingBlocks,), {}))


def test_tensor_arg_has_no_tile_advance_fields():
    from torch_spyre._inductor.op_spec import TensorArg

    field_names = {f.name for f in dataclasses.fields(TensorArg)}
    assert "tile_advance_expr" not in field_names
    assert "full_tiled_extent" not in field_names


def test_op_spec_has_no_tile_advance_fields():
    from torch_spyre._inductor.op_spec import OpSpec

    field_names = {f.name for f in dataclasses.fields(OpSpec)}
    assert "tile_advance_expr" not in field_names
    assert "full_tiled_extent" not in field_names
