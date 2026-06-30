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

import copy

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOps(TestCase):
    def __init__(self, method_name="runTest", methodName="runTest"):
        super().__init__(method_name, methodName)
        self.rtol = 1e-2
        self.atol = 1e-3
        self.dtype = torch.float16

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_linear(self):
        with torch.no_grad():
            m = nn.Linear(64, 128, bias=False, dtype=self.dtype)
            m.weight.normal_(0, 0.01)
            m_spyre = copy.deepcopy(m)
            m_spyre.to("spyre")

            x = torch.randn((2, 64), dtype=self.dtype)
            x_spyre = x.to("spyre")

            y = m_spyre(x_spyre).to("cpu")

        torch.testing.assert_close(y, m(x), rtol=self.rtol, atol=self.atol)

    def test_softmax(self):
        m = nn.Softmax(dim=1)
        m_spyre = m.to("spyre")

        x = torch.randn(1024, 256, dtype=self.dtype)
        x_spyre = x.to("spyre")

        with torch.no_grad():
            y = m_spyre(x_spyre).to("cpu")

        torch.testing.assert_close(y, m(x), rtol=self.rtol, atol=self.atol)


if __name__ == "__main__":
    run_tests()
