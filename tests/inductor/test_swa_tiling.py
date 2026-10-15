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

import inspect
import unittest

from torch_spyre._inductor.decompositions import _select_swa_tiling, _windowed_attention


class TestSWATiling(unittest.TestCase):
    _LX_BUDGET = 1_625_344

    def test_windowed_attention_uses_no_spyre_hints(self):
        self.assertNotIn("spyre_hint(", inspect.getsource(_windowed_attention))

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

    def test_gemma4_prefill_scans_the_window_in_one_block(self):
        # Gemma 4 sliding layers at S=2048: one 1088-row block measured
        # 16.1 ms, against 49.2 ms for the former five 256-row blocks.
        config = self._select()

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 1088)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(config.work_div, {"q_block": 16})
        self.assertEqual(config.kv_bytes_per_core, 1088 * 4096)
        self.assertEqual(
            config.reason, "fewest K/V blocks whose resident floor fits LX"
        )

    def test_long_query_block_takes_fewest_blocks_whose_floor_fits(self):
        # A 512-row query block cannot hold a 4096-row score pair in LX.
        # Four 1024-row blocks measured 1.80 ms; 2048- and 4096-row blocks,
        # which spill scores, measured 4.0 and 12.6 ms.
        config = self._select(q_block=512, buffer_width=4096)

        self.assertEqual(config.strategy, "work_divided_tiled")
        self.assertEqual(config.kv_block_size, 1024)
        self.assertEqual(config.num_kv_blocks, 4)

    def test_granite_window_scans_in_one_block(self):
        # Granite SWA (H32/Hkv8/D128, window 2048) at S=4096: one 2112-row
        # block measured 63.8 ms; 256-row blocks 97.5 ms and five 448-row
        # blocks 125.7 ms.
        config = self._select(num_heads=32, head_dim=128, buffer_width=2112)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 2112)
        self.assertEqual(config.num_kv_blocks, 1)

    def test_unpadded_odd_block_count_is_not_selected(self):
        # Three 704-row blocks cover 2112 exactly, so the K/V loop would slice
        # the cache view, whose strides an odd tile count does not divide.
        config = self._select(
            num_heads=16,
            num_kvheads=2,
            q_block=512,
            buffer_width=2112,
            head_dim=512,
        )

        self.assertEqual(config.kv_block_size, 448)
        self.assertEqual(config.num_kv_blocks, 5)

    def test_repeated_bmm_tiles_are_stick_aligned(self):
        config = self._select(head_dim=128, num_heads=32, q_block=512)

        self.assertEqual(config.kv_block_size, 576)
        self.assertEqual(config.num_kv_blocks, 2)
        self.assertEqual(config.kv_block_size % 64, 0)
        self.assertEqual(config.kv_block_size * config.num_kv_blocks, 1152)

    def test_gemma4_decode_uses_fewest_dsc_executions(self):
        config = self._select(q_block=1)

        self.assertEqual(config.strategy, "decode")
        self.assertEqual(config.kv_block_size, 1088)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertIsNone(config.work_div)
        self.assertIn("fewest DSC executes", config.reason)

    def test_gemma3_prefill_keeps_one_block(self):
        config = self._select(
            num_heads=8,
            num_kvheads=4,
            q_block=64,
            buffer_width=576,
        )

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 576)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(config.work_div, {"q_block": 16})

    def test_decode_is_not_tied_to_known_model_geometry(self):
        config = self._select(
            num_heads=12,
            num_kvheads=12,
            q_block=1,
            buffer_width=1088,
            head_dim=64,
        )

        self.assertEqual(config.strategy, "decode")
        self.assertEqual(config.kv_block_size, 1088)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_head_tiles, 1)

    def test_gqa_preserves_native_head_axes(self):
        config = self._select(
            num_heads=12,
            num_kvheads=3,
            q_block=64,
            buffer_width=2048,
            head_dim=128,
        )

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.work_div, {"q_block": 16})
        self.assertEqual(config.num_head_tiles, 1)

    def test_mha_work_division_uses_swa_dimension_names(self):
        config = self._select(
            num_heads=12,
            num_kvheads=12,
            head_dim=64,
        )

        self.assertEqual(
            config.work_div,
            {"num_heads": 4, "q_block": 8},
        )

    def test_low_lx_budget_uses_fallback(self):
        config = self._select(lx_budget_bytes=32 * 1024)

        self.assertEqual(config.strategy, "fallback_tiled")
        self.assertEqual(config.kv_block_size, 384)
        self.assertEqual(config.num_kv_blocks, 3)
        self.assertEqual(config.num_head_tiles, 1)
        self.assertEqual(
            config.reason,
            "estimated per-core live footprint exceeds the LX budget",
        )
        self.assertGreater(config.estimated_live_bytes_per_core, 32 * 1024)

    def test_work_division_scales_with_available_cores(self):
        config = self._select(num_cores=16)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 1088)
        self.assertEqual(config.work_div, {"q_block": 16})
        self.assertEqual(config.num_head_tiles, 1)

    def test_wider_cache_uses_the_same_geometry_model(self):
        config = self._select(buffer_width=4096)

        self.assertEqual(config.strategy, "work_divided")
        self.assertEqual(config.kv_block_size, 4096)
        self.assertEqual(config.num_kv_blocks, 1)
        self.assertEqual(config.num_head_tiles, 1)

    def test_long_query_block_keeps_coarse_fallback(self):
        config = self._select(q_block=1024)

        self.assertEqual(config.strategy, "fallback_tiled")
        self.assertIsNone(config.work_div)
        self.assertEqual(
            config.reason,
            "query block exceeds the calibrated work-divided limit",
        )

    def test_geometry_grid_preserves_tiling_invariants(self):
        geometries = (
            (8, 1, 512),
            (12, 12, 64),
            (16, 2, 256),
            (24, 3, 128),
            (32, 8, 128),
        )
        for num_heads, num_kvheads, head_dim in geometries:
            for query_block in (1, 2, 16, 64):
                for buffer_width in (64, 576, 1088, 4096):
                    with self.subTest(
                        num_heads=num_heads,
                        num_kvheads=num_kvheads,
                        head_dim=head_dim,
                        query_block=query_block,
                        buffer_width=buffer_width,
                    ):
                        config = self._select(
                            num_heads=num_heads,
                            num_kvheads=num_kvheads,
                            head_dim=head_dim,
                            q_block=query_block,
                            buffer_width=buffer_width,
                        )

                        self.assertGreaterEqual(config.kv_block_size, 64)
                        self.assertEqual(config.kv_block_size % 64, 0)
                        physical_width = config.num_kv_blocks * config.kv_block_size
                        self.assertGreaterEqual(physical_width, buffer_width)
                        self.assertLess(
                            physical_width - buffer_width,
                            config.num_kv_blocks * 64,
                        )
                        self.assertEqual(num_heads % config.num_head_tiles, 0)
                        if config.work_div is not None:
                            self.assertEqual(
                                query_block % config.work_div["q_block"], 0
                            )
                            if num_heads != num_kvheads:
                                self.assertEqual(set(config.work_div), {"q_block"})
                        if not config.strategy.startswith("fallback"):
                            self.assertIsNotNone(config.estimated_live_bytes_per_core)
                            self.assertLessEqual(
                                config.estimated_live_bytes_per_core,
                                config.lx_budget_bytes,
                            )


if __name__ == "__main__":
    unittest.main()
