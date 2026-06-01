#!/usr/bin/env python3
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

"""
Tests for SpyreAsyncCompile persistent kernel caching.

Each test runs inside ``torch._inductor.utils.fresh_cache()``, which redirects
all Inductor / Spyre cache I/O to a fresh temporary directory.  The developer's
real cache is never read or modified by these tests.

Run with:
    python -m pytest tests/test_kernel_caching.py -v
    python -m pytest tests/test_kernel_caching.py -v -k test_cache_hit
"""

import os
import unittest
import torch
import torch_spyre  # noqa: F401 — side-effects: registers Spyre backend

import torch_spyre._inductor.config as spyre_config
from torch._inductor.utils import fresh_cache

from torch_spyre.execution.kernel_cache import (
    allocate_compile_dir,
    commit_compile_dir,
    get_cache_root_dir,
    get_cache_stats,
    get_cached_kernel_dir,
)

DEVICE = torch.device("spyre")


def _simple_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _make_input(shape=(64, 512), dtype=torch.float16):
    return torch.rand(*shape, dtype=dtype).to(DEVICE)


class TestCacheMissOnColdStart(unittest.TestCase):
    def test_cache_is_empty_before_first_compile(self):
        """Cache must be empty before any torch.compile() is called."""
        with fresh_cache():
            torch._dynamo.reset()
            stats = get_cache_stats()
            self.assertEqual(stats["total_cached_kernels"], 0)

    def test_first_compile_populates_cache(self):
        """After one torch.compile() run, at least one kernel should be cached."""
        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
            torch._dynamo.reset()
            compiled = torch.compile(_simple_fn)
            compiled(_make_input())

            stats = get_cache_stats()
            self.assertGreater(
                stats["total_cached_kernels"],
                0,
                "Expected at least one kernel in cache after first compile",
            )


class TestCacheArtifactCompleteness(unittest.TestCase):
    REQUIRED = [
        "bundle.mlir",
        os.path.join("spyreCodeDir", "init_binary.bin"),
        os.path.join("spyreCodeDir", "spyrecode.json"),
    ]

    def test_all_required_artifacts_present(self):
        """Every cached kernel directory must contain all required artifacts."""
        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
            torch._dynamo.reset()
            torch.compile(_simple_fn)(_make_input())

            cache_root = get_cache_root_dir()
            cached_entries = [
                d
                for d in os.listdir(cache_root)
                if os.path.isdir(os.path.join(cache_root, d))
            ]
            self.assertGreater(len(cached_entries), 0, "No cached entries found")

            for entry in cached_entries:
                entry_dir = os.path.join(cache_root, entry)
                for artifact in self.REQUIRED:
                    self.assertTrue(
                        os.path.isfile(os.path.join(entry_dir, artifact)),
                        f"Missing artifact '{artifact}' in cache entry '{entry}'",
                    )

                has_sdsc = any(
                    f.startswith("sdsc_") and f.endswith(".json")
                    for f in os.listdir(entry_dir)
                )
                self.assertTrue(
                    has_sdsc,
                    f"No sdsc_N.json files found in cache entry '{entry}'",
                )


class TestPartialCacheEntryTreatedAsMiss(unittest.TestCase):
    def test_partial_write_does_not_produce_cache_hit(self):
        """A directory missing spyreCodeDir/init_binary.bin must be a cache miss."""
        with fresh_cache():
            cache_root = get_cache_root_dir()
            fake_key = "c" + "a" * 63
            fake_dir = os.path.join(cache_root, fake_key)
            os.makedirs(os.path.join(fake_dir, "spyreCodeDir"), exist_ok=True)

            with open(os.path.join(fake_dir, "bundle.mlir"), "w") as f:
                f.write("fake bundle")
            with open(os.path.join(fake_dir, "sdsc_0.json"), "w") as f:
                f.write("{}")
            with open(
                os.path.join(fake_dir, "spyreCodeDir", "spyrecode.json"), "w"
            ) as f:
                f.write("{}")
            # init_binary.bin intentionally missing

            result = get_cached_kernel_dir(fake_key)
            self.assertIsNone(
                result,
                "Expected cache miss for partial entry missing init_binary.bin",
            )


class TestCacheDisabledViaConfig(unittest.TestCase):
    def test_cache_disabled_leaves_cache_empty(self):
        """With spyre_kernel_cache=False, the kernel cache must remain empty."""
        import torch_spyre._inductor.config as spyre_config

        with fresh_cache():
            torch._dynamo.reset()
            original = spyre_config.spyre_kernel_cache
            spyre_config.spyre_kernel_cache = False
            try:
                torch.compile(_simple_fn)(_make_input())
                self.assertEqual(
                    get_cache_stats()["total_cached_kernels"],
                    0,
                    "Expected empty cache when spyre_kernel_cache=False",
                )
            finally:
                spyre_config.spyre_kernel_cache = original


class TestForceDisableCaches(unittest.TestCase):
    def test_force_disable_caches_leaves_cache_empty(self):
        """torch._inductor.config.force_disable_caches must bypass the Spyre cache."""
        with fresh_cache():
            torch._dynamo.reset()
            with torch._inductor.config.patch({"force_disable_caches": True}):
                torch.compile(_simple_fn)(_make_input())

            self.assertEqual(
                get_cache_stats()["total_cached_kernels"],
                0,
                "Expected empty cache when force_disable_caches=True",
            )


class TestDifferentOpsProduceDifferentKeys(unittest.TestCase):
    def test_softmax_and_relu_have_different_cache_entries(self):
        """Two different ops must not share a cache entry."""
        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
            torch._dynamo.reset()
            x = _make_input()

            torch.compile(lambda a: torch.softmax(a, dim=-1))(x)
            count_after_softmax = get_cache_stats()["total_cached_kernels"]

            torch._dynamo.reset()
            torch.compile(lambda a: torch.relu(a))(x)
            count_after_relu = get_cache_stats()["total_cached_kernels"]

            self.assertGreater(
                count_after_relu,
                count_after_softmax,
                "Expected a new cache entry for relu vs softmax",
            )


class TestSameOpReusesCacheEntry(unittest.TestCase):
    def test_same_op_compiled_twice_uses_same_cache_entry(self):
        """Compiling the same op twice must not create duplicate cache entries."""
        with fresh_cache():
            torch._dynamo.reset()
            x = _make_input()

            torch.compile(lambda a: torch.softmax(a, dim=-1))(x)
            count_first = get_cache_stats()["total_cached_kernels"]

            torch._dynamo.reset()
            torch.compile(lambda a: torch.softmax(a, dim=-1))(x)
            count_second = get_cache_stats()["total_cached_kernels"]

            self.assertEqual(
                count_first,
                count_second,
                "Expected no new cache entries when compiling the same op twice",
            )


class TestClearCache(unittest.TestCase):
    def test_clear_cache_removes_all_entries(self):
        """clear_cache() must leave total_cached_kernels == 0."""
        from torch_spyre.execution.kernel_cache import clear_cache

        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
            torch._dynamo.reset()
            torch.compile(_simple_fn)(_make_input())

            self.assertGreater(get_cache_stats()["total_cached_kernels"], 0)

            clear_cache()
            stats = get_cache_stats()
            self.assertEqual(stats["total_cached_kernels"], 0)
            self.assertAlmostEqual(stats["cache_size_mb"], 0.0, places=1)


class TestAtomicCommit(unittest.TestCase):
    def test_concurrent_commit_same_key_does_not_corrupt(self):
        """Four threads compiling the same key concurrently must leave exactly one valid entry."""
        import threading

        fake_key = "c" + "b" * 63

        with fresh_cache():
            errors = []

            def do_compile():
                try:
                    # Each thread gets its own allocated tmp dir with the same key.
                    tmp_dir = allocate_compile_dir(fake_key)
                    # Populate it with the minimal required artifacts.
                    os.makedirs(os.path.join(tmp_dir, "spyreCodeDir"), exist_ok=True)
                    for name in ["bundle.mlir", "sdsc_0.json"]:
                        with open(os.path.join(tmp_dir, name), "w") as f:
                            f.write("content")
                    for name in ["init_binary.bin", "spyrecode.json"]:
                        with open(
                            os.path.join(tmp_dir, "spyreCodeDir", name), "wb"
                        ) as f:
                            f.write(b"content")
                    commit_compile_dir(tmp_dir, fake_key)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=do_compile) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"Concurrent commit raised errors: {errors}")

            result = get_cached_kernel_dir(fake_key)
            self.assertIsNotNone(
                result, "Expected a valid cache entry after concurrent commit"
            )


class TestNoDiskIOOnCacheHit(unittest.TestCase):
    def test_generate_bundle_skipped_on_cache_hit(self):
        """On a kernel cache hit generate_bundle must not be called."""
        from unittest.mock import patch

        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
            torch._dynamo.reset()
            # Populate the cache on first run.
            torch.compile(_simple_fn)(_make_input())

            # Second compile (after dynamo reset) must hit the cache.
            torch._dynamo.reset()
            with patch(
                "torch_spyre.execution.async_compile.generate_bundle"
            ) as mock_gen:
                torch.compile(_simple_fn)(_make_input())

            mock_gen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
