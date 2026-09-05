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

"""Tests for logging infrastructure."""

import torch  # noqa: F401
from torch_spyre._inductor.logging_utils import (
    get_inductor_logger,
)


class TestLoggingOperations:
    def setup_method(self, method):
        torch.manual_seed(0xAFFE)

    def test_create_logger(self):
        logger = get_inductor_logger("test_module")
        assert logger is not None
        assert logger.name.endswith("test_module")

    def test_logging_does_not_crash(self):
        logger = get_inductor_logger("test")
        logger.debug("test message")
        logger.info("test message")
        logger.warning("test message")
        logger.debug("test message with data: shape=[2, 3], device_size=[1, 2, 3]")
