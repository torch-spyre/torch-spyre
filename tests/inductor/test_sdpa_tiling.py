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

"""CPU-only tests for the SDPA decomposition tiling cost model."""

import unittest

from torch_spyre._inductor.decompositions import _select_sdpa_tiling


class TestSDPATiling(unittest.TestCase):
    _LX_BUDGET = 1_625_344

    def _select(
        self,
        *,
        batch_size=1,
        num_heads=12,
        num_kvheads=12,
        max_seqlen_q=512,
        max_seqlen_kv=512,
        head_dim=128,
        element_size=2,
        num_cores=32,
        lx_budget_bytes=_LX_BUDGET,
    ):
        return _select_sdpa_tiling(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kvheads=num_kvheads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            head_dim=head_dim,
            element_size=element_size,
            num_cores=num_cores,
            lx_budget_bytes=lx_budget_bytes,
        )

    def test_issue_4339_uses_one_block_and_work_division(self):
        config = self._select(head_dim=64)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_q_tiles, 1)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(
            config.work_div,
            {"num_heads": 4, "max_seqlen_q": 8, "max_seqlen_kv": 8},
        )
        self.assertEqual(config.score_bytes_per_core, 192 * 1024)
        self.assertEqual(config.estimated_live_bytes_per_core, 443136)

    def test_granite_gqa_uses_calibrated_work_divided_tile(self):
        config = self._select(num_heads=32, num_kvheads=8)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(
            config.work_div,
            {"num_heads": 4, "max_seqlen_q": 8, "max_seqlen_kv": 4},
        )

    def test_granite_long_kv_keeps_512_block_and_avoids_head_tiling(self):
        config = self._select(
            num_heads=32,
            num_kvheads=8,
            max_seqlen_kv=8192,
        )

        self.assertEqual(config.strategy, "work_divided_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 16)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(
            config.work_div,
            {"num_heads": 4, "max_seqlen_q": 8, "max_seqlen_kv": 4},
        )

    def test_gemma_local_uses_smaller_kv_block_for_wide_heads(self):
        config = self._select(
            num_heads=16,
            num_kvheads=8,
            head_dim=256,
            max_seqlen_kv=8192,
        )

        self.assertEqual(config.strategy, "work_divided_tiled")
        self.assertEqual(config.kv_block_size, 256)
        self.assertEqual(config.num_kv_blocks, 32)
        self.assertEqual(
            config.work_div,
            {"num_heads": 2, "max_seqlen_q": 16, "max_seqlen_kv": 8},
        )

    def test_gemma_global_uses_larger_head_split_and_512_kv_block(self):
        for num_kvheads in (1, 2):
            with self.subTest(num_kvheads=num_kvheads):
                config = self._select(
                    num_heads=16,
                    num_kvheads=num_kvheads,
                    head_dim=512,
                    max_seqlen_kv=8192,
                )

                self.assertEqual(config.strategy, "work_divided_tiled")
                self.assertEqual(config.kv_block_size, 512)
                self.assertEqual(config.num_kv_blocks, 16)
                self.assertEqual(
                    config.work_div,
                    {"num_heads": 8, "max_seqlen_q": 4, "max_seqlen_kv": 4},
                )

    def test_decode_uses_geometry_calibrated_policy(self):
        cases = (
            ("granite", 32, 8, 128, 4096, None, 540800),
            ("gemma-local", 16, 8, 256, 2048, None, 147520),
            (
                "gemma-global-12b",
                16,
                1,
                512,
                512,
                {"num_heads": 16, "max_seqlen_q": 1, "max_seqlen_kv": 1},
                4100,
            ),
            (
                "gemma-global-26b",
                16,
                2,
                512,
                256,
                {"num_heads": 8, "max_seqlen_q": 1, "max_seqlen_kv": 1},
                6152,
            ),
        )
        for (
            model,
            num_heads,
            num_kvheads,
            head_dim,
            kv_block_size,
            work_div,
            expected_live_bytes,
        ) in cases:
            with self.subTest(model=model):
                config = self._select(
                    num_heads=num_heads,
                    num_kvheads=num_kvheads,
                    max_seqlen_q=1,
                    max_seqlen_kv=8192,
                    head_dim=head_dim,
                )

                expected_strategy = (
                    "decode_work_divided_tiled" if work_div else "decode_tiled"
                )
                self.assertEqual(config.strategy, expected_strategy)
                self.assertEqual(
                    config.reason, "single-query decode; geometry-calibrated"
                )
                self.assertEqual(config.kv_block_size, kv_block_size)
                self.assertEqual(config.num_kv_blocks, 8192 // kv_block_size)
                self.assertEqual(config.num_head_tiles, 1)
                self.assertEqual(config.work_div, work_div)
                self.assertEqual(
                    config.estimated_live_bytes_per_core, expected_live_bytes
                )

    def test_unknown_decode_geometry_keeps_conservative_policy(self):
        config = self._select(
            num_heads=12,
            num_kvheads=12,
            max_seqlen_q=1,
            max_seqlen_kv=8192,
            head_dim=64,
        )

        self.assertEqual(config.strategy, "decode_tiled")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 16)
        self.assertIsNone(config.work_div)

    def test_decode_block_caps_cover_calibrated_lengths(self):
        cases = (
            ("granite", 32, 8, 128, (512, 1024, 4096)),
            ("gemma-local", 16, 8, 256, (512, 1024, 2048)),
            ("gemma-global-12b", 16, 1, 512, (512, 512, 512)),
            ("gemma-global-26b", 16, 2, 512, (256, 256, 256)),
        )
        for model, num_heads, num_kvheads, head_dim, expected_blocks in cases:
            for sequence_length, expected_block in zip(
                (512, 1024, 8192), expected_blocks
            ):
                with self.subTest(model=model, sequence_length=sequence_length):
                    config = self._select(
                        num_heads=num_heads,
                        num_kvheads=num_kvheads,
                        max_seqlen_q=1,
                        max_seqlen_kv=sequence_length,
                        head_dim=head_dim,
                    )

                    self.assertEqual(config.kv_block_size, expected_block)
                    self.assertEqual(
                        config.num_kv_blocks, sequence_length // expected_block
                    )

    def test_long_contexts_keep_blocking_and_loop_grouping(self):
        for sequence_length in (8 * 1024, 32 * 1024):
            with self.subTest(sequence_length=sequence_length):
                config = self._select(
                    max_seqlen_q=sequence_length,
                    max_seqlen_kv=sequence_length,
                )

                self.assertEqual(config.strategy, "coarse_tiled")
                self.assertEqual(
                    config.reason,
                    "query extent exceeds the calibrated work-divided limit",
                )
                self.assertEqual(config.kv_block_size, 512)
                self.assertGreater(config.num_kv_blocks, 1)
                self.assertGreater(config.num_q_tiles, 1)
                self.assertEqual(
                    config.kv_blocks_per_loop_group,
                    min(config.num_kv_blocks, max(1, 16 // config.num_q_tiles)),
                )

    def test_non_divisible_sequence_keeps_coarse_tiling(self):
        config = self._select(max_seqlen_q=500, max_seqlen_kv=500)

        self.assertEqual(config.strategy, "coarse_tiled")
        self.assertEqual(
            config.reason, "no exact full-core head/sequence work division"
        )
        self.assertIsNone(config.work_div)

    def test_lx_budget_reduces_kv_block_until_live_values_fit(self):
        config = self._select(batch_size=2, lx_budget_bytes=300 * 1024)

        self.assertEqual(config.strategy, "work_divided_tiled")
        self.assertEqual(config.kv_block_size, 64)
        self.assertEqual(config.num_kv_blocks, 8)
        self.assertEqual(config.score_bytes_per_core, 48 * 1024)
        self.assertEqual(config.estimated_live_bytes_per_core, 296448)

    def test_live_footprint_over_budget_keeps_coarse_tiling(self):
        config = self._select(batch_size=2, lx_budget_bytes=250 * 1024)

        self.assertEqual(config.strategy, "coarse_tiled")
        self.assertEqual(config.score_bytes_per_core, 48 * 1024)
        self.assertEqual(config.estimated_live_bytes_per_core, 296448)
        self.assertEqual(
            config.reason,
            "estimated per-core live footprint exceeds the LX budget",
        )

    def test_work_division_scales_with_available_cores(self):
        config = self._select(num_cores=16)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(
            config.work_div,
            {"num_heads": 4, "max_seqlen_q": 4, "max_seqlen_kv": 4},
        )


if __name__ == "__main__":
    unittest.main()
