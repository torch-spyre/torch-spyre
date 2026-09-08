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

import logging

import torch  # noqa: F401
import torch_spyre._inductor.logging_utils as logging_utils
from torch_spyre._inductor.logging_utils import (
    get_inductor_logger,
    warn_once,
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


class TestWarnOnce:
    def setup_method(self, method):
        logging_utils._warned_once.clear()

    def test_same_key_suppresses_repeat(self, caplog):
        logger = get_inductor_logger("test_warn_once_repeat")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            warn_once(logger, "opX", "skipping %s", "opX")
            warn_once(logger, "opX", "skipping %s", "opX")
        assert [r.getMessage() for r in caplog.records] == ["skipping opX"]

    def test_different_key_still_fires(self, caplog):
        logger = get_inductor_logger("test_warn_once_distinct_keys")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            warn_once(logger, "opX", "skipping %s", "opX")
            warn_once(logger, "opY", "skipping %s", "opY")
        assert len(caplog.records) == 2

    def test_message_may_vary_without_defeating_dedup(self, caplog):
        # The dedup key is caller-supplied, not the formatted message, so a
        # message that legitimately differs per call (e.g. a shape folded
        # into the text) still dedupes correctly on the shared key.
        logger = get_inductor_logger("test_warn_once_varying_message")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            warn_once(logger, "opX", "skipping opX, shape=%s", [1, 2])
            warn_once(logger, "opX", "skipping opX, shape=%s", [3, 4])
        assert [r.getMessage() for r in caplog.records] == [
            "skipping opX, shape=[1, 2]"
        ]

    def test_same_key_different_logger_not_cross_suppressed(self, caplog):
        logger_a = get_inductor_logger("test_warn_once_logger_a")
        logger_b = get_inductor_logger("test_warn_once_logger_b")
        with caplog.at_level(logging.WARNING, logger=logger_a.name):
            with caplog.at_level(logging.WARNING, logger=logger_b.name):
                warn_once(logger_a, "opX", "skipping %s", "opX")
                warn_once(logger_b, "opX", "skipping %s", "opX")
        assert len(caplog.records) == 2
