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

# Owner(s): ["module: cpp"]

import os

import torch
import torch._dynamo
from torch.testing._internal.common_utils import TestCase

import torch_spyre

_FLAGS = ("TORCH_SPYRE_CORRECTION_CACHE", "TORCH_SPYRE_CORRECTION_CACHE_VERIFY")


def _fn(x, y):
    return torch.softmax(x * y + x, dim=-1)


def _pair():
    return (
        torch.randn(64, 128, dtype=torch.float16),
        torch.randn(64, 128, dtype=torch.float16),
    )


class TestCorrectionCache(TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _FLAGS}
        torch.manual_seed(0)
        torch._dynamo.reset()
        self.compiled = torch.compile(_fn, backend="inductor")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _check(self, x, y, repeats):
        xd, yd = x.to("spyre"), y.to("spyre")
        for _ in range(repeats):
            out = self.compiled(xd, yd).cpu()
            torch.testing.assert_close(out, _fn(x, y), atol=0.1, rtol=0.1)

    def _workload(self):
        self._check(*_pair(), repeats=4)
        for _ in range(3):
            self._check(*_pair(), repeats=1)

    def test_reused_corrections_match_fresh_ones(self):
        os.environ["TORCH_SPYRE_CORRECTION_CACHE"] = "1"
        os.environ["TORCH_SPYRE_CORRECTION_CACHE_VERIFY"] = "1"
        before = torch_spyre._C._correction_cache_stats()
        self._workload()
        after = torch_spyre._C._correction_cache_stats()
        self.assertGreater(after["misses"], before["misses"])
        self.assertGreater(after["hits"], before["hits"])

    def test_cached_corrections_give_correct_results(self):
        os.environ["TORCH_SPYRE_CORRECTION_CACHE"] = "1"
        os.environ.pop("TORCH_SPYRE_CORRECTION_CACHE_VERIFY", None)
        self._workload()

    def test_disabled_by_default(self):
        os.environ.pop("TORCH_SPYRE_CORRECTION_CACHE", None)
        before = torch_spyre._C._correction_cache_stats()
        self._workload()
        self.assertEqual(torch_spyre._C._correction_cache_stats(), before)
