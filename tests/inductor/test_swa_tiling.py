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

"""CPU-only tests for sliding-window attention tiling selection."""

import unittest

from torch_spyre._inductor.decompositions import _select_swa_tiling


class TestSWATiling(unittest.TestCase):
    _LX_BUDGET = 1_625_344

    def _select(
        self,
        *,
        batch_size=1,
        num_heads=16,
        num_kvheads=8,
        q_block=64,
        buffer_width=1088,
        head_dim=256,
        element_size=2,
        num_cores=32,
        lx_budget_bytes=_LX_BUDGET,
    ):
        return _select_swa_tiling(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kvheads=num_kvheads,
            q_block=q_block,
            buffer_width=buffer_width,
            head_dim=head_dim,
            element_size=element_size,
            num_cores=num_cores,
            lx_budget_bytes=lx_budget_bytes,
        )

    def test_gemma4_prefill_uses_swept_policy(self):
        config = self._select()

        self.assertEqual(config.strategy, "calibrated_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 3)
        self.assertEqual(config.num_head_tiles, 1)

    def test_gemma4_decode_uses_two_full_blocks_and_a_tail(self):
        config = self._select(q_block=1)

        self.assertEqual(config.strategy, "calibrated_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 3)
        self.assertEqual(config.num_head_tiles, 1)

    def test_gemma3_uses_one_compact_cache_block(self):
        config = self._select(
            num_heads=8,
            num_kvheads=4,
            q_block=64,
            buffer_width=576,
        )

        self.assertEqual(config.strategy, "calibrated")
        self.assertEqual(config.kv_block_size, 576)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_head_tiles, 1)

    def test_unknown_decode_geometry_keeps_conservative_blocks(self):
        config = self._select(
            num_heads=12,
            num_kvheads=12,
            q_block=1,
            buffer_width=1088,
            head_dim=64,
        )

        self.assertEqual(config.strategy, "fallback_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 3)
        self.assertEqual(config.num_head_tiles, 3)

    def test_unknown_gqa_geometry_preserves_native_head_axes(self):
        config = self._select(
            num_heads=12,
            num_kvheads=3,
            q_block=64,
            buffer_width=2048,
            head_dim=128,
        )

        self.assertEqual(config.strategy, "fallback_tiled")
        self.assertEqual(config.num_head_tiles, 1)

    def test_low_lx_budget_preserves_native_gqa_head_axes(self):
        config = self._select(q_block=1, lx_budget_bytes=64 * 1024)

        self.assertEqual(config.strategy, "fallback_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 3)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(
            config.reason,
            "shape or hardware is outside the calibrated SWA policies",
        )

    def test_non_swept_core_count_keeps_conservative_blocks(self):
        config = self._select(num_cores=16)

        self.assertEqual(config.strategy, "fallback_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_head_tiles, 1)

    def test_wider_generic_cache_keeps_conservative_blocks(self):
        config = self._select(buffer_width=4096)

        self.assertEqual(config.strategy, "fallback_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_head_tiles, 1)


if __name__ == "__main__":
    unittest.main()
