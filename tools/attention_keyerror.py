#!/usr/bin/env python3
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

"""Minimal repro: compiling two or more attention layers in one graph fails.

    python3 tools/repro_attention_keyerror.py --layers 1   # compiles and runs
    python3 tools/repro_attention_keyerror.py --layers 2   # KeyError: 'bufN'

Observed on the dev box (fp16, batch 1, seq 400): 1 layer works, 2 layers raise
KeyError in Scheduler.compute_ancestors. Seq 128 with 6 layers fails the same
way. Stream environment variables are not involved.
"""

import argparse

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401
from torch import nn

D_MODEL = 256
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS  # 32
D_HEAD_PAD = 64  # padded to a full fp16 stick
FFN = 2048


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        embed = N_HEADS * D_HEAD_PAD
        self.q_proj = nn.Linear(D_MODEL, embed)
        self.k_proj = nn.Linear(D_MODEL, embed)
        self.v_proj = nn.Linear(D_MODEL, embed)
        self.out_proj = nn.Linear(embed, D_MODEL)
        self.linear1 = nn.Linear(D_MODEL, FFN)
        self.linear2 = nn.Linear(FFN, D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.scaling = float(D_HEAD) ** -0.5

    def forward(self, x):
        T, B, _ = x.shape
        H, Dh = N_HEADS, D_HEAD_PAD
        q = (self.q_proj(x) * self.scaling).view(T, B * H, Dh).transpose(0, 1)
        k = self.k_proj(x).view(T, B * H, Dh).transpose(0, 1)
        v = self.v_proj(x).view(T, B * H, Dh).transpose(0, 1)
        attn = torch.bmm(q, k.transpose(1, 2)).softmax(-1)
        out = torch.bmm(attn, v).transpose(0, 1).contiguous().view(T, B, H * Dh)
        x = self.norm1(x + self.out_proj(out))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seq", type=int, default=400)
    args = p.parse_args()

    model = nn.Sequential(*(Block() for _ in range(args.layers))).eval()
    model = model.to("spyre", dtype=torch.float16)
    x = torch.randn(args.seq, 1, D_MODEL).to("spyre", dtype=torch.float16)

    with torch.no_grad():
        torch.compile(model, dynamic=False)(x).cpu()
    print(f"OK: {args.layers} layer(s), seq {args.seq} compiled and ran")


if __name__ == "__main__":
    main()
