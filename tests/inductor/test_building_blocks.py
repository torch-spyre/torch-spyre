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

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_inductor import compare_with_cpu, compare_with_pytorch


class TestBuildingBlocks(unittest.TestCase):
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
        compare_with_cpu(rms_norm, *args)
