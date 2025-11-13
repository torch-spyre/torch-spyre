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

# Owner(s): ["module: cpp"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_spyre._C import SpyreDCI


class TestSpyreDCI(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    # Test generic stick shorthands
    def test_generic_stick(self):
        dci = SpyreDCI(1)
        self.assertEqual(dci.dim_order, [0])
        self.assertEqual(dci.format, SpyreDCI.StickFormat.Dense)
        self.assertEqual(dci.num_stick_dims, 1)

        dci = SpyreDCI(3)
        self.assertEqual(dci.dim_order, [0, 1, 2])
        self.assertEqual(dci.format, SpyreDCI.StickFormat.Dense)
        self.assertEqual(dci.num_stick_dims, 1)

    def test_explicit_dci_constructor(self):
        dci_x = SpyreDCI(3)
        dci_y = SpyreDCI([0, 1, 2], 1, SpyreDCI.StickFormat.Dense)
        self.assertEqual(dci_x.format, dci_y.format)
        self.assertEqual(dci_x.num_stick_dims, dci_y.num_stick_dims)
        self.assertEqual(dci_x.dim_order, dci_y.dim_order)

    def test_sparse_dci_constructor(self):
        dci = SpyreDCI([0, 1, 2], 1, SpyreDCI.StickFormat.Sparse)
        self.assertEqual(dci.format, SpyreDCI.StickFormat.Sparse)


if __name__ == "__main__":
    run_tests()
