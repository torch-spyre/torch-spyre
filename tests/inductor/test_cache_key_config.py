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

"""The compile cache key must ignore settings that only control logs."""

import unittest

import torch_spyre._inductor.config as spyre_config


def _cache_key_config():
    # Inductor hashes exactly this dict into its FX graph cache key
    # (see FxGraphHashDetails in torch/_inductor/codecache.py).
    return spyre_config.save_config_portable(ignore_private_configs=False)


class TestCacheKeyConfig(unittest.TestCase):
    def test_log_only_settings_are_not_in_the_key(self):
        key = _cache_key_config()
        for name in ("timing", "timing_out", "dump_cost_expr_file"):
            self.assertNotIn(name, key)
        self.assertNotIn("spyre_kernel_cache", key)

    def test_log_only_settings_do_not_change_the_key(self):
        with spyre_config.patch(
            {
                "timing": False,
                "timing_out": "",
                "dump_cost_expr_file": "",
                "spyre_kernel_cache": False,
            }
        ):
            before = _cache_key_config()
        with spyre_config.patch(
            {
                "timing": True,
                "timing_out": "/tmp/other_timing.json",
                "dump_cost_expr_file": "/tmp/other_cost.jsonl",
                "spyre_kernel_cache": True,
            }
        ):
            after = _cache_key_config()
        self.assertEqual(before, after)

    def test_code_settings_still_change_the_key(self):
        with spyre_config.patch({"lx_planning": True}):
            on = _cache_key_config()
        with spyre_config.patch({"lx_planning": False}):
            off = _cache_key_config()
        self.assertIn("lx_planning", on)
        self.assertNotEqual(on, off)


if __name__ == "__main__":
    unittest.main()
