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

import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSpyreCCLBackend(TestCase):
    def test_spyreccl_device_to_backend(self) -> None:
        # Make sure the module has been loaded
        assert dist.distributed_c10d.is_backend_available("spyreccl")
        # Make sure the module is the default for the spyre device
        assert "spyreccl" == dist.get_default_backend_for_device("spyre")


if __name__ == "__main__":
    run_tests()
