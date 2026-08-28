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

"""Unit tests for torch_spyre._inductor.config flag defaults."""

import unittest

from torch_spyre._inductor import config


class TestFrontendPoolAllocationConfig(unittest.TestCase):
    def test_default_is_false(self):
        self.assertFalse(config.frontend_pool_allocation)

    def test_patchable(self):
        with config.patch({"frontend_pool_allocation": True}):
            self.assertTrue(config.frontend_pool_allocation)
        self.assertFalse(config.frontend_pool_allocation)


if __name__ == "__main__":
    unittest.main()
