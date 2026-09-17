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

import sys
import unittest

import torch  # noqa: F401 - loads the registered Spyre backend entry point

_select_sdpa_tiling = sys.modules[
    "torch_spyre._inductor.decompositions"
]._select_sdpa_tiling
_num_tiles_for_max_extent = sys.modules[
    "torch_spyre._inductor.decompositions"
]._num_tiles_for_max_extent


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

    def test_exact_tile_search_is_bounded_and_returns_a_valid_split(self):
        sequence_lengths = (*range(1, 258), 509, 512, 9973)
        max_extents = (1, 2, 3, 7, 31, 63, 64, 65, 127, 128, 511, 512)
        alignments = (1, 2, 3, 32, 64, 128, 1024)

        for sequence_length in sequence_lengths:
            for max_extent in max_extents:
                for alignment in alignments:
                    num_tiles = _num_tiles_for_max_extent(
                        sequence_length,
                        max_extent,
                        tile_alignment=alignment,
                    )
                    tile_size = sequence_length // num_tiles
                    alignment_is_possible = (
                        sequence_length % alignment == 0 and max_extent >= alignment
                    )

                    self.assertLessEqual(num_tiles, sequence_length)
                    self.assertEqual(sequence_length % num_tiles, 0)
                    self.assertLessEqual(tile_size, max_extent)
                    if alignment_is_possible:
                        self.assertEqual(tile_size % alignment, 0)

    def test_exact_tile_search_rejects_nonpositive_inputs(self):
        for sequence_length, max_extent, alignment in (
            (0, 64, 64),
            (-1, 64, 64),
            (64, 0, 64),
            (64, -1, 64),
            (64, 64, 0),
            (64, 64, -1),
        ):
            with self.assertRaises(ValueError):
                _num_tiles_for_max_extent(
                    sequence_length,
                    max_extent,
                    tile_alignment=alignment,
                )

    def test_mha_uses_one_block_and_full_core_work_division(self):
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

    def test_low_head_long_mha_uses_query_only_work_division(self):
        for num_heads in (2, 4, 8):
            with self.subTest(num_heads=num_heads):
                config = self._select(
                    num_heads=num_heads,
                    num_kvheads=num_heads,
                    max_seqlen_q=64,
                    max_seqlen_kv=8192,
                )

                self.assertEqual(
                    config.work_div,
                    {"max_seqlen_q": 32, "max_seqlen_kv": 32},
                )

    def test_low_head_short_mha_keeps_head_query_work_division(self):
        config = self._select(
            num_heads=2,
            num_kvheads=2,
            max_seqlen_q=64,
            max_seqlen_kv=512,
        )

        self.assertEqual(
            config.work_div,
            {"num_heads": 2, "max_seqlen_q": 16, "max_seqlen_kv": 16},
        )

    def test_wide_head_long_mha_keeps_head_query_work_division(self):
        config = self._select(
            num_heads=16,
            num_kvheads=16,
            max_seqlen_q=64,
            max_seqlen_kv=8192,
        )

        self.assertEqual(
            config.work_div,
            {"num_heads": 4, "max_seqlen_q": 8, "max_seqlen_kv": 8},
        )

    def test_prefill_block_tracks_physical_kv_footprint_and_gqa_reuse(self):
        cases = (
            # Hq, Hkv, D, expected K block
            (32, 8, 128, 512),
            (16, 8, 256, 256),
            (16, 2, 512, 1024),
            (16, 1, 512, 1024),
        )
        for num_heads, num_kvheads, head_dim, expected_block in cases:
            with self.subTest(
                num_heads=num_heads, num_kvheads=num_kvheads, head_dim=head_dim
            ):
                config = self._select(
                    num_heads=num_heads,
                    num_kvheads=num_kvheads,
                    head_dim=head_dim,
                    max_seqlen_kv=8192,
                )

                self.assertEqual(config.strategy, "work_divided_tiled")
                self.assertEqual(config.kv_block_size, expected_block)
                self.assertEqual(config.num_kv_blocks, 8192 // expected_block)
                self.assertEqual(config.work_div, {"max_seqlen_q": 32})

    def test_short_gqa_chunks_use_available_query_parallelism(self):
        for query_length, expected_split in ((2, 2), (8, 8), (16, 16), (64, 32)):
            with self.subTest(query_length=query_length):
                config = self._select(
                    num_heads=32,
                    num_kvheads=8,
                    max_seqlen_q=query_length,
                    max_seqlen_kv=8192,
                )

                self.assertEqual(config.strategy, "work_divided_tiled")
                self.assertEqual(config.kv_block_size, 512)
                self.assertEqual(config.num_head_tiles, 1)
                self.assertEqual(config.work_div, {"max_seqlen_q": expected_split})

    def test_gqa_work_division_never_splits_a_head_axis(self):
        config = self._select(
            num_heads=24,
            num_kvheads=3,
            max_seqlen_q=96,
            max_seqlen_kv=2048,
            head_dim=128,
        )

        self.assertEqual(config.work_div, {"max_seqlen_q": 32})

    def test_short_chunk_does_not_overweight_gqa_reuse(self):
        for num_kvheads in (1, 2):
            with self.subTest(num_kvheads=num_kvheads):
                config = self._select(
                    num_heads=16,
                    num_kvheads=num_kvheads,
                    max_seqlen_q=16,
                    max_seqlen_kv=8192,
                    head_dim=512,
                )

                self.assertEqual(config.kv_block_size, 512)
                self.assertEqual(config.work_div, {"max_seqlen_q": 16})

    def test_decode_selects_lx_resident_k_when_block_count_is_small(self):
        cases = (
            # Hq, Hkv, D, expected K block
            (16, 2, 512, 256),
            (8, 1, 512, 256),
            (16, 2, 256, 512),
            (16, 4, 256, 1024),
            (16, 4, 512, 1024),
            (32, 4, 512, 1024),
            (16, 1, 512, 1024),
        )
        for num_heads, num_kvheads, head_dim, expected_block in cases:
            with self.subTest(
                num_heads=num_heads, num_kvheads=num_kvheads, head_dim=head_dim
            ):
                config = self._select(
                    num_heads=num_heads,
                    num_kvheads=num_kvheads,
                    max_seqlen_q=1,
                    max_seqlen_kv=1024,
                    head_dim=head_dim,
                )

                self.assertEqual(config.kv_block_size, expected_block)
                self.assertEqual(config.num_kv_blocks, 1024 // expected_block)
                self.assertEqual(config.num_head_tiles, 1)
                self.assertIsNone(config.work_div)

    def test_decode_uses_full_feasible_block_when_execution_count_dominates(self):
        for sequence_length in (2048, 4096, 8192):
            with self.subTest(sequence_length=sequence_length):
                config = self._select(
                    num_heads=16,
                    num_kvheads=2,
                    max_seqlen_q=1,
                    max_seqlen_kv=sequence_length,
                    head_dim=512,
                )

                self.assertEqual(config.strategy, "decode")
                self.assertEqual(config.kv_block_size, sequence_length)
                self.assertEqual(config.num_kv_blocks, 1)
                self.assertIn("fewest DSC executes", config.reason)

        # Four K1024 blocks keep restickified K in LX for this geometry, but
        # their BMM and online-softmax overhead exceeds one long K4096 block.
        config = self._select(
            num_heads=16,
            num_kvheads=1,
            max_seqlen_q=1,
            max_seqlen_kv=4096,
            head_dim=512,
        )
        self.assertEqual(config.kv_block_size, 4096)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertIn("fewest DSC executes", config.reason)

    def test_unknown_decode_geometry_is_not_forced_to_a_model_policy(self):
        config = self._select(
            num_heads=12,
            num_kvheads=12,
            max_seqlen_q=1,
            max_seqlen_kv=8192,
            head_dim=64,
        )

        self.assertEqual(config.strategy, "decode")
        self.assertEqual(config.kv_block_size, 8192)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertIsNone(config.work_div)

    def test_non_power_of_two_query_uses_partial_core_work_division(self):
        config = self._select(max_seqlen_q=500, max_seqlen_kv=500)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 500)
        self.assertEqual(
            config.work_div,
            {"num_heads": 3, "max_seqlen_q": 10, "max_seqlen_kv": 10},
        )

    def test_long_queries_keep_coarse_tiling_and_loop_grouping(self):
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

    def test_coarse_kv_tiles_are_exact_and_stick_aligned(self):
        config = self._select(
            num_heads=16,
            num_kvheads=16,
            max_seqlen_q=3520,
            max_seqlen_kv=3520,
        )

        self.assertEqual(config.strategy, "coarse_tiled")
        self.assertEqual(config.kv_block_size, 320)
        self.assertEqual(config.num_kv_blocks, 11)
        self.assertEqual(config.kv_block_size % 64, 0)
        self.assertEqual(
            config.num_kv_blocks * config.kv_block_size,
            3520,
        )

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

    def test_geometry_grid_preserves_tiling_invariants(self):
        geometries = (
            (8, 1, 512),
            (12, 12, 64),
            (16, 2, 256),
            (16, 4, 512),
            (24, 3, 128),
            (32, 8, 128),
        )
        for num_heads, num_kvheads, head_dim in geometries:
            for query_length in (1, 2, 16, 96, 500, 1024):
                for kv_length in (512, 1024, 8192):
                    with self.subTest(
                        num_heads=num_heads,
                        num_kvheads=num_kvheads,
                        head_dim=head_dim,
                        query_length=query_length,
                        kv_length=kv_length,
                    ):
                        config = self._select(
                            num_heads=num_heads,
                            num_kvheads=num_kvheads,
                            head_dim=head_dim,
                            max_seqlen_q=query_length,
                            max_seqlen_kv=kv_length,
                        )

                        self.assertGreaterEqual(config.kv_block_size, 64)
                        self.assertEqual(config.kv_block_size % 64, 0)
                        self.assertEqual(
                            config.num_kv_blocks,
                            (kv_length + config.kv_block_size - 1)
                            // config.kv_block_size,
                        )
                        self.assertEqual(
                            config.num_q_tiles * config.q_tile_size,
                            query_length,
                        )
                        self.assertEqual(num_heads % config.num_head_tiles, 0)
                        if config.work_div is not None:
                            self.assertEqual(
                                query_length % config.work_div["max_seqlen_q"], 0
                            )
                            if num_heads != num_kvheads:
                                self.assertEqual(set(config.work_div), {"max_seqlen_q"})
                        if config.strategy != "coarse_tiled":
                            self.assertIsNotNone(config.estimated_live_bytes_per_core)
                            assert config.estimated_live_bytes_per_core is not None
                            self.assertLessEqual(
                                config.estimated_live_bytes_per_core,
                                config.lx_budget_bytes,
                            )


if __name__ == "__main__":
    unittest.main()
