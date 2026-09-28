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

"""Tests for the SDPA decomposition and its tiling cost model."""

import dataclasses
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

from torch_spyre._C import SpyreTensorLayout, get_device_dtype

_decompositions = sys.modules["torch_spyre._inductor.decompositions"]
_select_sdpa_tiling = _decompositions._select_sdpa_tiling
_axis_slice_is_dense = _decompositions._axis_slice_is_dense
_sdpa_kv_candidates = _decompositions._sdpa_kv_candidates
_sdpa_mask_hbm_bytes = _decompositions._sdpa_mask_hbm_bytes
_sdpa_estimated_live_bytes_per_core = (
    _decompositions._sdpa_estimated_live_bytes_per_core
)
_sdpa_has_loop_boundary = _decompositions._sdpa_has_loop_boundary
_num_tiles_for_max_extent = _decompositions._num_tiles_for_max_extent
_sdpa_num_batch_tiles = _decompositions._sdpa_num_batch_tiles


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
        mask_shapes=(),
        head_tile_staging_bytes=0,
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
            mask_shapes=mask_shapes,
            head_tile_staging_bytes=head_tile_staging_bytes,
        )

    def _select_production_gqa(
        self,
        *,
        num_heads,
        num_kvheads,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim,
        batch_size=1,
        element_size=2,
    ):
        """Model the adapter's broadcast mask and interleaved B/S/H/D inputs."""
        mask_shape = (
            batch_size,
            1,
            1,
            max_seqlen_q,
            max_seqlen_kv,
        )
        query_bytes = batch_size * num_heads * max_seqlen_q * head_dim * element_size
        one_kv_bytes = (
            batch_size * num_kvheads * max_seqlen_kv * head_dim * element_size
        )
        # A head tile of an interleaved input needs a read and a staging write
        # for K and V; restoring the output query layout has the same two sides.
        head_tile_staging_bytes = 2 * (2 * one_kv_bytes + query_bytes)
        return self._select(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kvheads=num_kvheads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            head_dim=head_dim,
            element_size=element_size,
            mask_shapes=(mask_shape,),
            head_tile_staging_bytes=head_tile_staging_bytes,
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

    def test_mha_uses_one_block_and_all_available_cores(self):
        config = self._select(head_dim=64)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_q_tiles, 1)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(config.num_group_tiles, 1)
        self.assertEqual(config.estimated_active_cores, 32)
        self.assertEqual(config.estimated_load_bursts, 2)
        self.assertEqual(config.score_bytes_per_core, 192 * 1024)
        self.assertEqual(config.estimated_live_bytes_per_core, 336640)

    def test_encoder_pooling_shape_avoids_serial_hop_tiling(self):
        # Issue #4784: granite-embedding-278m's pooling path supplies
        # interleaved B/L/H/D inputs and a broadcast key-padding mask.
        tensor_bytes = 4 * 12 * 512 * 64 * 2
        config = self._select(
            batch_size=4,
            head_dim=64,
            mask_shapes=((4, 1, 1, 512),),
            # Read+write staging for Q, K, and V when H is tiled.
            head_tile_staging_bytes=6 * tensor_bytes,
        )

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 512)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_batch_tiles, 1)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(config.num_group_tiles, 1)
        self.assertEqual(config.num_q_tiles, 1)
        self.assertEqual(config.estimated_active_cores, 32)
        self.assertEqual(config.score_bytes_per_core, 768 * 1024)
        self.assertEqual(config.estimated_live_bytes_per_core, 1182720)
        self.assertEqual(config.estimated_spill_buffers, 0)

    def test_loop_free_live_set_is_independent_of_attention_geometry(self):
        kwargs = dict(
            batch_size=1,
            heads_per_core=2,
            query_rows_per_core=1,
            kv_block_size=512,
            head_dim=256,
            element_size=2,
            has_loop_boundary=False,
        )
        score_bytes, direct_live_bytes = _sdpa_estimated_live_bytes_per_core(
            **kwargs,
            full_sdpa_prefill=False,
        )
        _, prefill_live_bytes = _sdpa_estimated_live_bytes_per_core(
            **kwargs,
            full_sdpa_prefill=True,
        )
        query_bytes = 2 * 1 * 256 * 2
        accumulator_bytes = 2 * 1 * 2

        self.assertEqual(
            direct_live_bytes,
            score_bytes + 3 * query_bytes + 2 * accumulator_bytes,
        )
        self.assertEqual(direct_live_bytes, prefill_live_bytes)

    def test_grouped_decode_uses_effective_loop_counts(self):
        for num_heads, num_kvheads, kv_length, head_dim in (
            (32, 8, 128, 128),
            (16, 8, 512, 256),
        ):
            with self.subTest(
                num_heads=num_heads,
                num_kvheads=num_kvheads,
                kv_length=kv_length,
            ):
                config = self._select(
                    num_heads=num_heads,
                    num_kvheads=num_kvheads,
                    max_seqlen_q=1,
                    max_seqlen_kv=kv_length,
                    head_dim=head_dim,
                )

                self.assertGreater(config.num_group_tiles, 1)
                self.assertEqual(config.num_kv_blocks, 1)
                self.assertFalse(
                    _sdpa_has_loop_boundary(
                        num_non_group_outer_tiles=(
                            config.num_batch_tiles
                            * config.num_head_tiles
                            * config.num_q_tiles
                        ),
                        num_group_tiles=config.num_group_tiles,
                        num_q_tiles=config.num_q_tiles,
                        num_kv_blocks=config.num_kv_blocks,
                    )
                )
                _, direct_live_bytes = _sdpa_estimated_live_bytes_per_core(
                    batch_size=1,
                    heads_per_core=num_heads,
                    query_rows_per_core=1,
                    kv_block_size=config.kv_block_size,
                    head_dim=head_dim,
                    element_size=2,
                    has_loop_boundary=False,
                )
                self.assertEqual(
                    config.estimated_live_bytes_per_core, direct_live_bytes
                )
                self.assertEqual(config.estimated_load_bursts, 2)

        tiled = self._select(
            num_heads=4,
            num_kvheads=1,
            max_seqlen_q=1,
            max_seqlen_kv=512,
            head_dim=128,
        )
        self.assertEqual(tiled.num_kv_blocks, 2)
        self.assertTrue(
            _sdpa_has_loop_boundary(
                num_non_group_outer_tiles=(
                    tiled.num_batch_tiles * tiled.num_head_tiles * tiled.num_q_tiles
                ),
                num_group_tiles=tiled.num_group_tiles,
                num_q_tiles=tiled.num_q_tiles,
                num_kv_blocks=tiled.num_kv_blocks,
            )
        )

    def test_batch_tiles_are_exact_for_odd_extents(self):
        for batch_size, expected_tiles in ((1, 1), (2, 1), (3, 3), (4, 2), (7, 7)):
            with self.subTest(batch_size=batch_size):
                num_tiles = _sdpa_num_batch_tiles(batch_size)
                self.assertEqual(num_tiles, expected_tiles)
                self.assertEqual(batch_size % num_tiles, 0)

    def test_low_head_long_mha_uses_all_available_cores(self):
        for num_heads in (2, 4, 8):
            with self.subTest(num_heads=num_heads):
                config = self._select(
                    num_heads=num_heads,
                    num_kvheads=num_heads,
                    max_seqlen_q=64,
                    max_seqlen_kv=8192,
                )

                self.assertEqual(config.estimated_active_cores, 32)

    def test_low_head_short_mha_uses_all_available_cores(self):
        config = self._select(
            num_heads=2,
            num_kvheads=2,
            max_seqlen_q=64,
            max_seqlen_kv=512,
        )

        self.assertEqual(config.estimated_active_cores, 32)

    def test_wide_head_long_mha_uses_all_available_cores(self):
        config = self._select(
            num_heads=16,
            num_kvheads=16,
            max_seqlen_q=64,
            max_seqlen_kv=8192,
        )

        self.assertEqual(config.estimated_active_cores, 32)

    def test_prefill_search_can_tile_every_outer_axis(self):
        cases = (
            # Hq, Hkv, D, K block, H tiles, G tiles
            (32, 8, 128, 512, 2, 1),
            (16, 8, 256, 256, 1, 1),
            (16, 2, 512, 256, 2, 1),
            (16, 1, 512, 256, 1, 2),
        )
        for (
            num_heads,
            num_kvheads,
            head_dim,
            expected_block,
            expected_head_tiles,
            expected_group_tiles,
        ) in cases:
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
                self.assertEqual(config.num_head_tiles, expected_head_tiles)
                self.assertEqual(config.num_group_tiles, expected_group_tiles)
                self.assertEqual(config.estimated_active_cores, 32)

    def test_short_gqa_chunks_use_available_head_query_parallelism(self):
        for query_length in (2, 8, 16, 64):
            with self.subTest(query_length=query_length):
                config = self._select(
                    num_heads=32,
                    num_kvheads=8,
                    max_seqlen_q=query_length,
                    max_seqlen_kv=8192,
                )

                self.assertEqual(config.strategy, "work_divided_tiled")
                self.assertEqual(
                    config.kv_block_size,
                    4096 if query_length <= 16 else 1024,
                )
                self.assertEqual(config.num_head_tiles, 1)
                self.assertEqual(config.num_group_tiles, 1)
                self.assertEqual(config.estimated_active_cores, 32)

    def test_gqa_core_estimate_uses_the_inner_hkv_and_query_axes(self):
        config = self._select(
            num_heads=24,
            num_kvheads=3,
            max_seqlen_q=96,
            max_seqlen_kv=2048,
            head_dim=128,
        )

        self.assertEqual(config.estimated_active_cores, 32)

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

                self.assertEqual(config.kv_block_size, 1024)
                self.assertEqual(config.estimated_active_cores, 32)

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
                self.assertEqual(config.num_group_tiles, num_heads // num_kvheads)
                self.assertIsNone(config.estimated_active_cores)

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
        self.assertIsNone(config.estimated_active_cores)

    def test_non_power_of_two_query_estimates_exact_active_cores(self):
        config = self._select(max_seqlen_q=500, max_seqlen_kv=500)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 500)
        self.assertEqual(config.estimated_active_cores, 30)

    def test_long_queries_are_tiled_by_lx_and_burst_costs(self):
        for sequence_length in (8 * 1024, 32 * 1024):
            with self.subTest(sequence_length=sequence_length):
                config = self._select(
                    max_seqlen_q=sequence_length,
                    max_seqlen_kv=sequence_length,
                )

                self.assertEqual(config.strategy, "work_divided_tiled")
                self.assertEqual(
                    config.reason,
                    "lowest loop, HBM burst, and bounded-spill transfer cost",
                )
                self.assertGreater(config.num_kv_blocks, 1)
                self.assertGreater(config.num_head_tiles * config.num_q_tiles, 1)
                self.assertLessEqual(config.estimated_spill_buffers, 1)
                self.assertEqual(
                    config.kv_blocks_per_loop_group,
                    min(config.num_kv_blocks, max(1, 16 // config.num_q_tiles)),
                )

    def test_production_chunks_account_for_mask_and_interleaved_layout(self):
        cases = (
            # Hq, Hkv, Lq, Lk, D, Q tiles, expected K block, spills
            (32, 8, 512, 8192, 128, 2, 512, 1),
            (32, 8, 512, 32768, 128, 4, 1024, 0),
            (16, 8, 1024, 8192, 256, 2, 256, 0),
            (16, 8, 1024, 32768, 256, 2, 256, 0),
        )
        for (
            num_heads,
            num_kvheads,
            query_length,
            kv_length,
            head_dim,
            query_tiles,
            kv_block,
            spill_buffers,
        ) in cases:
            with self.subTest(
                num_heads=num_heads,
                num_kvheads=num_kvheads,
                query_length=query_length,
                head_dim=head_dim,
            ):
                config = self._select_production_gqa(
                    num_heads=num_heads,
                    num_kvheads=num_kvheads,
                    max_seqlen_q=query_length,
                    max_seqlen_kv=kv_length,
                    head_dim=head_dim,
                )

                self.assertEqual(config.strategy, "work_divided_tiled")
                self.assertEqual(config.num_q_tiles, query_tiles)
                self.assertEqual(config.q_tile_size, query_length // query_tiles)
                self.assertEqual(config.num_batch_tiles, 1)
                self.assertEqual(config.num_head_tiles, 1)
                self.assertEqual(config.num_group_tiles, 1)
                self.assertEqual(config.kv_block_size, kv_block)
                self.assertEqual(config.estimated_spill_buffers, spill_buffers)
                self.assertEqual(config.estimated_active_cores, 32)

    def test_selected_kv_tiles_are_exact_and_stick_aligned(self):
        config = self._select(
            num_heads=16,
            num_kvheads=16,
            max_seqlen_q=3520,
            max_seqlen_kv=3520,
        )

        self.assertEqual(config.strategy, "work_divided_tiled")
        self.assertEqual(config.kv_block_size, 704)
        self.assertEqual(config.num_kv_blocks, 5)
        self.assertEqual(config.kv_block_size % 64, 0)
        self.assertEqual(
            config.num_kv_blocks * config.kv_block_size,
            3520,
        )

    def test_lx_budget_allows_one_priced_carry_spill(self):
        config = self._select(batch_size=2, lx_budget_bytes=300 * 1024)

        self.assertEqual(config.strategy, "work_divided_tiled")
        self.assertEqual(config.kv_block_size, 256)
        self.assertEqual(config.num_kv_blocks, 2)
        self.assertEqual(config.num_q_tiles, 1)
        self.assertEqual(config.num_head_tiles, 6)
        self.assertEqual(config.score_bytes_per_core, 32 * 1024)
        self.assertEqual(config.estimated_live_bytes_per_core, 311552)
        self.assertEqual(config.estimated_spill_buffers, 1)
        self.assertGreater(config.estimated_spill_bytes, 0)

    def test_batch_axis_is_enumerated_and_exact(self):
        config = self._select(batch_size=6, lx_budget_bytes=600 * 1024)

        self.assertEqual(config.num_batch_tiles, 2)
        self.assertEqual(6 % config.num_batch_tiles, 0)
        self.assertEqual(config.num_head_tiles, 4)
        self.assertGreater(config.estimated_live_bytes_per_core, config.lx_budget_bytes)
        self.assertEqual(config.estimated_spill_buffers, 1)

    def test_broadcast_mask_replay_is_charged_on_outer_map_axes(self):
        one_mask_pass = 512 * 8192 * 2
        estimated = _sdpa_mask_hbm_bytes(
            mask_shapes=((1, 1, 1, 512, 8192),),
            axis_extents=(1, 8, 4, 512, 8192),
            axis_tile_counts=(1, 2, 4, 2, 16),
            element_size=2,
        )

        # Q and K slice the mask, while its broadcast H/G axes replay it.
        self.assertEqual(estimated, one_mask_pass * 2 * 4)

    def test_interleaved_head_staging_favors_query_tiling(self):
        kwargs = dict(
            num_heads=32,
            num_kvheads=8,
            max_seqlen_q=512,
            max_seqlen_kv=8192,
            head_dim=128,
            mask_shapes=((1, 1, 1, 512, 8192),),
        )
        without_staging = self._select(**kwargs)
        with_staging = self._select(
            **kwargs,
            head_tile_staging_bytes=75_497_472,
        )

        self.assertEqual(without_staging.num_head_tiles, 2)
        self.assertEqual(without_staging.num_q_tiles, 1)
        self.assertEqual(with_staging.num_head_tiles, 1)
        self.assertEqual(with_staging.num_q_tiles, 2)

    def test_interleaved_head_axis_is_not_dense(self):
        shape = (1, 8, 8192, 128)
        self.assertTrue(
            _axis_slice_is_dense(shape, (8 * 8192 * 128, 8192 * 128, 128, 1), 1)
        )
        self.assertFalse(
            _axis_slice_is_dense(shape, (8192 * 8 * 128, 128, 8 * 128, 1), 1)
        )

    def test_conservative_restick_estimate_rejects_near_budget_plan(self):
        candidates = _sdpa_kv_candidates(
            batch_size=1,
            num_heads=32,
            num_kvheads=8,
            max_seqlen_q=512,
            max_seqlen_kv=8192,
            head_dim=128,
            element_size=2,
            num_cores=32,
            query_tile_size=512,
            group_tile_size=4,
            num_non_group_outer_tiles=1,
            full_sdpa_prefill=True,
        )
        k256 = next(
            candidate for candidate in candidates if candidate.block_size == 256
        )

        self.assertEqual(k256.estimated_restick_active_cores, 8)
        self.assertEqual(k256.estimated_live_bytes_per_core, 2_033_664)
        self.assertGreater(k256.estimated_live_bytes_per_core, self._LX_BUDGET)

    def test_map_carry_residency_preserves_the_right_production_split(self):
        def candidate(*, query_tile_size, group_tile_size, block_size, head_dim):
            candidates = _sdpa_kv_candidates(
                batch_size=1,
                num_heads=8 * group_tile_size,
                num_kvheads=8,
                max_seqlen_q=query_tile_size,
                max_seqlen_kv=8192,
                head_dim=head_dim,
                element_size=2,
                num_cores=32,
                query_tile_size=query_tile_size,
                group_tile_size=group_tile_size,
                num_non_group_outer_tiles=1,
                full_sdpa_prefill=True,
            )
            return next(item for item in candidates if item.block_size == block_size)

        granite_q2_k512 = candidate(
            query_tile_size=256,
            group_tile_size=4,
            block_size=512,
            head_dim=128,
        )
        granite_q4_k1024 = candidate(
            query_tile_size=128,
            group_tile_size=4,
            block_size=1024,
            head_dim=128,
        )
        gemma_q2_k256 = candidate(
            query_tile_size=512,
            group_tile_size=2,
            block_size=256,
            head_dim=256,
        )

        self.assertGreater(
            granite_q2_k512.estimated_live_bytes_per_core,
            self._LX_BUDGET,
        )
        self.assertLessEqual(
            granite_q4_k1024.estimated_live_bytes_per_core,
            self._LX_BUDGET,
        )
        self.assertLessEqual(
            gemma_q2_k256.estimated_live_bytes_per_core,
            self._LX_BUDGET,
        )

    def test_live_footprint_over_budget_keeps_coarse_tiling(self):
        config = self._select(batch_size=2, lx_budget_bytes=16 * 1024)

        self.assertEqual(config.strategy, "coarse_tiled")
        self.assertGreater(config.estimated_live_bytes_per_core, 16 * 1024)
        self.assertEqual(
            config.reason,
            "estimated per-core live footprint exceeds the LX budget",
        )

    def test_estimated_active_cores_scales_with_available_cores(self):
        config = self._select(num_cores=16)

        self.assertEqual(config.estimated_active_cores, 16)

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
                        self.assertEqual(1 % config.num_batch_tiles, 0)
                        group_size = num_heads // num_kvheads
                        self.assertEqual(group_size % config.num_group_tiles, 0)
                        if config.estimated_active_cores is not None:
                            self.assertGreaterEqual(config.estimated_active_cores, 1)
                            self.assertLessEqual(config.estimated_active_cores, 32)
                        if config.strategy != "coarse_tiled":
                            self.assertIsNotNone(config.estimated_live_bytes_per_core)
                            assert config.estimated_live_bytes_per_core is not None
                            self.assertIsNotNone(config.estimated_spill_buffers)
                            self.assertLessEqual(config.estimated_spill_buffers, 1)
                            self.assertEqual(
                                config.estimated_spill_buffers,
                                int(
                                    config.estimated_live_bytes_per_core
                                    > config.lx_budget_bytes
                                ),
                            )
                            self.assertGreaterEqual(config.estimated_load_bursts, 2)


class TestSDPAForEachTileIntegration(unittest.TestCase):
    def test_gqa_decode_with_interleaved_kv_cache(self):
        """A tiled-away group must preserve query strides across KV heads."""
        generator = torch.Generator().manual_seed(123)
        dtype = torch.bfloat16
        query = torch.randn(1, 16, 1, 128, dtype=dtype, generator=generator)
        key = torch.randn(1, 4, 512, 128, dtype=dtype, generator=generator)
        value = torch.randn(1, 4, 512, 128, dtype=dtype, generator=generator)
        mask = torch.full((1, 16, 1, 512), float("-inf"), dtype=dtype)
        # Include values on both sides of the 256-token KV tile boundary.
        mask[..., 246:257] = 0

        def sdpa(q, k, v, mask):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=1 / 128, enable_gqa=True
            )

        query_device = query.to("spyre")
        # Position-first cache layout used by decode: [L, Hkv, D/64, B, D%64].
        cache_layout = SpyreTensorLayout(
            device_size=[512, 4, 2, 1, 64],
            stride_map=[128, 512 * 128, 64, 4 * 512 * 128, 1],
            device_dtype=get_device_dtype(dtype),
        )
        caches = [
            torch.empty(
                tensor.shape,
                dtype=dtype,
                device=torch.device("spyre"),
                device_layout=cache_layout,
            ).copy_(tensor)
            for tensor in (key, value)
        ]
        expected = sdpa(query.float(), key.float(), value.float(), mask.float())
        actual = torch.compile(sdpa, fullgraph=True, dynamic=False)(
            query_device, *caches, mask.to("spyre")
        )
        torch.testing.assert_close(actual.cpu().float(), expected, atol=0.03, rtol=0.03)

    def test_gqa_decode_direct_body_tracks_effective_group_loop(self):
        """A nominal G split is a real loop only when a sequence axis is tiled."""

        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                scale=128**-0.5,
                enable_gqa=True,
            )

        generator = torch.Generator().manual_seed(0)
        for kv_length, expect_loop in ((128, False), (512, True)):
            with self.subTest(kv_length=kv_length):
                query = torch.randn(
                    1, 4, 1, 128, dtype=torch.float16, generator=generator
                )
                key = torch.randn(
                    1, 1, kv_length, 128, dtype=torch.float16, generator=generator
                )
                value = torch.randn(
                    1, 1, kv_length, 128, dtype=torch.float16, generator=generator
                )
                expected = sdpa(query, key, value)
                actual, sources = run_and_get_code(
                    torch.compile(
                        sdpa, backend="inductor", fullgraph=True, dynamic=False
                    ),
                    query.to("spyre"),
                    key.to("spyre"),
                    value.to("spyre"),
                )
                source = "\n".join(sources)

                torch.testing.assert_close(
                    actual.cpu().float(), expected.float(), atol=0.1, rtol=0.1
                )
                self.assertEqual("LoopSpec(" in source, expect_loop)
                self.assertEqual("op='maximum'" in source, expect_loop)

    def test_complete_gqa_tile_nest(self):
        """The actual decomposition lowers B/Hkv/G/Lq maps around an Lk scan."""
        batch = 2
        query_heads = 4
        kv_heads = 2
        query_length = 32
        kv_length = 256
        head_dim = 128
        generator = torch.Generator().manual_seed(0)
        query = torch.randn(
            batch,
            query_heads,
            query_length,
            head_dim,
            dtype=torch.float16,
            generator=generator,
        )
        key = torch.randn(
            batch,
            kv_heads,
            kv_length,
            head_dim,
            dtype=torch.float16,
            generator=generator,
        )
        value = torch.randn(
            batch,
            kv_heads,
            kv_length,
            head_dim,
            dtype=torch.float16,
            generator=generator,
        )
        bias = (
            torch.randn(
                batch,
                query_heads,
                query_length,
                kv_length,
                dtype=torch.float16,
                generator=generator,
            )
            * 0.01
        )

        def sdpa(q, k, v, b):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=b,
                dropout_p=0.0,
                scale=head_dim**-0.5,
                enable_gqa=True,
            )

        def force_every_loop(**kwargs):
            selected = _select_sdpa_tiling(**kwargs)
            return dataclasses.replace(
                selected,
                strategy="work_divided_tiled",
                kv_block_size=128,
                num_kv_blocks=2,
                num_q_tiles=2,
                q_tile_size=16,
                num_batch_tiles=2,
                num_head_tiles=2,
                num_group_tiles=2,
                kv_blocks_per_loop_group=2,
            )

        expected = sdpa(query, key, value, bias)
        with mock.patch.object(
            _decompositions,
            "_select_sdpa_tiling",
            side_effect=force_every_loop,
        ):
            actual, sources = run_and_get_code(
                torch.compile(sdpa, backend="inductor", fullgraph=True, dynamic=False),
                query.to("spyre"),
                key.to("spyre"),
                value.to("spyre"),
                bias.to("spyre"),
            )

        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), atol=0.1, rtol=0.1
        )
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 5)


if __name__ == "__main__":
    unittest.main()
