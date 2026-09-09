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

"""OpSpec tiling tests"""

import math
import torch
import unittest


class TestOpSpecTiling(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_flash(self):
        B = 1
        H = 8
        D = 128
        Lq = 512
        Lk = 1024

        b_block_size = 1
        h_block_size = 4
        q_block_size = 256
        kv_block_size = 512

        def flash(queries, keys, values, mask):
            scale = 1.0 / math.sqrt(math.sqrt(D))

            output = torch.zeros_like(queries)

            # indirectly create sparse tensor using stick reduction
            real_max = torch.full(
                (B, H, Lq, 64),
                float("-inf"),
                device=queries.device,
                dtype=torch.float16,
            ).amax(-1)

            # indirectly create sparse tensor using stick reduction
            denominator = torch.zeros(
                (B, H, Lq, 64), device=queries.device, dtype=torch.float16
            ).amax(-1)

            for b_start in range(0, B, b_block_size):
                b_end = b_start + b_block_size
                for h_start in range(0, H, h_block_size):
                    h_end = h_start + h_block_size

                    for lq_start in range(0, Lq, q_block_size):
                        lq_end = lq_start + q_block_size
                        queries_tile = queries[
                            b_start:b_end, h_start:h_end, lq_start:lq_end
                        ]
                        real_max_tile = real_max[
                            b_start:b_end, h_start:h_end, lq_start:lq_end
                        ]
                        denominator_tile = denominator[
                            b_start:b_end, h_start:h_end, lq_start:lq_end
                        ]
                        output_tile = output[
                            b_start:b_end, h_start:h_end, lq_start:lq_end
                        ]

                        for lk_start in range(0, Lk, kv_block_size):
                            lk_end = lk_start + kv_block_size
                            mask_tile = mask[:, :, lq_start:lq_end, lk_start:lk_end]
                            keys_tile = keys[
                                b_start:b_end, h_start:h_end, lk_start:lk_end
                            ]
                            values_tile = values[
                                b_start:b_end, h_start:h_end, lk_start:lk_end
                            ]
                            keys_tile_T = keys_tile.transpose(-1, -2).contiguous()

                            scores = torch.matmul(
                                queries_tile * scale, keys_tile_T * scale
                            )  # tile_b, tile_h, tile_lq, tile_lk
                            scores = (
                                scores + mask_tile
                            )  # additive mask in [.., tile_lq, tile_lk]
                            block_max = torch.amax(
                                scores, dim=-1
                            )  # tile_b, tile_h, tile_lq
                            running_max = torch.maximum(
                                real_max_tile, block_max
                            )  # tile_b, tile_h, tile_lq

                            exp_scores = torch.exp(
                                scores - running_max.unsqueeze(-1)
                            )  # tile_b, tile_h, tile_lq, tile_lk
                            correction = torch.exp(
                                real_max_tile - running_max
                            )  # tile_b, tile_h, tile_lq

                            denominator_tile.copy_(
                                denominator_tile * correction + exp_scores.sum(dim=-1)
                            )  # tile_b, tile_h, tile_lq
                            output_tile.copy_(
                                output_tile * correction.unsqueeze(-1)
                                + torch.matmul(exp_scores, values_tile)
                            )  # tile_b, tile_h, tile_lq, D

                            real_max_tile.copy_(running_max)

            return output / denominator.unsqueeze(-1)

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)

        # Causal additive mask in natural [1, 1, Lq, Lk] orientation: query i
        # attends to keys 0..i.  0.0 = keep, -inf = masked.  The kept diagonal
        # guarantees no fully-masked row (no 0/0 NaN denominator).
        causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
        mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        mask_t.masked_fill_(~causal, float("-inf"))

        queries_t_spyre = queries_t.to("spyre")
        keys_t_spyre = keys_t.to("spyre")
        values_t_spyre = values_t.to("spyre")
        mask_t_spyre = mask_t.to(device="spyre")

        attn_t = flash(queries_t, keys_t, values_t, mask_t)

        flash_spyre = torch.compile(flash)
        attn_t_spyre = flash_spyre(
            queries_t_spyre,
            keys_t_spyre,
            values_t_spyre,
            mask_t_spyre,
        )

        torch.testing.assert_close(attn_t, attn_t_spyre.cpu(), atol=0.1, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
