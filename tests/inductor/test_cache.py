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

import unittest
import torch
from torch._inductor.utils import fresh_cache
from torch._dynamo.utils import counters


class TestCache(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_cache(self):
        counters.clear()
        a = torch.randn((64, 64)).to("spyre")
        fn = torch.compile(torch.abs, dynamic=False)
        with fresh_cache():
            fn(a)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())
        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        torch._dynamo.reset()

        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes)
            fn(a)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())

    def test_cache_key_includes_spyre_layout(self):
        """
        Verify that FxGraphHashDetails includes SpyreTensorLayout in the cache key.
        Different layouts should produce different cache keys, preventing incorrect
        cache hits across layout changes.
        """
        from torch.spyre import SpyreTensorLayout

        x = torch.rand([64, 64], dtype=torch.float16)
        stl_a = SpyreTensorLayout(
            list(x.size()), list(x.stride()), torch.float16, [0, 1]
        )
        stl_b = SpyreTensorLayout(
            list(x.size()), list(x.stride()), torch.float16, [1, 0]
        )

        _ = x.to("spyre")  # wake up spyre
        tensor_a = x.to(device_layout=stl_a)
        tensor_b = x.to(device_layout=stl_b)

        fn = torch.compile(lambda a: a + a, dynamic=False)

        # ── Layout A — first compile, cache miss expected ─────────────────────────
        counters.clear()
        with fresh_cache():
            fn(tensor_a)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            artifacts_a = torch.compiler.save_cache_artifacts()
            self.assertIsNotNone(artifacts_a)
            artifact_bytes_a, _ = artifacts_a

        # ── Layout B — different layout, should NOT hit Layout A cache ────────────
        torch._dynamo.reset()
        counters.clear()
        with fresh_cache():
            # load Layout A artifact into cache
            torch.compiler.load_cache_artifacts(artifact_bytes_a)

            # compile with Layout B — should miss (different layout → different key)
            fn(tensor_b)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"],
                1,
                "Layout B should miss Layout A cache — different SpyreTensorLayout",
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"],
                0,
                "Layout B should not hit Layout A cache — SpyreTensorLayout differs",
            )

        # ── Layout A again — should hit its own cache ─────────────────────────────
        torch._dynamo.reset()
        counters.clear()
        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes_a)
            fn(tensor_a)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"],
                1,
                "Layout A should hit its own cache",
            )
