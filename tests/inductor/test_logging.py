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

import os
import logging
from unittest.mock import patch
import torch  # noqa: F401
import torch_spyre._inductor.logging_utils as logging_utils
from torch_spyre._inductor.logging_utils import (
    get_inductor_logger,
    is_inductor_logging_enabled,
    warn_once,
)


class TestLoggingConfiguration:
    def setup_method(self, method):
        torch.manual_seed(0xAFFE)

    def test_default_is_disabled(self):
        with patch.object(logging_utils, "_needs_reinit", True):
            with patch.dict(os.environ, {}, clear=True):
                assert not is_inductor_logging_enabled()
                logger = get_inductor_logger("test_disabled")
                assert logger.level == logging.WARNING

    def test_enabled_defaults_to_info_level(self):
        with patch.object(logging_utils, "_needs_reinit", True):
            with patch.dict(os.environ, {"SPYRE_INDUCTOR_LOG": "1"}, clear=True):
                assert is_inductor_logging_enabled()
                logger = get_inductor_logger("test_enabled")
                assert logger.level == logging.INFO


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

    def test_same_key_suppresses_repeat(self):
        logger = get_inductor_logger("test_warn_once_repeat")
        with patch.object(logger, "warning") as mock_warning:
            warn_once(logger, "opX", "skipping %s", "opX")
            warn_once(logger, "opX", "skipping %s", "opX")
        mock_warning.assert_called_once_with("skipping %s", "opX")

    def test_different_key_still_fires(self):
        logger = get_inductor_logger("test_warn_once_distinct_keys")
        with patch.object(logger, "warning") as mock_warning:
            warn_once(logger, "opX", "skipping %s", "opX")
            warn_once(logger, "opY", "skipping %s", "opY")
        assert mock_warning.call_count == 2

    def test_message_may_vary_without_defeating_dedup(self):
        # The dedup key is caller-supplied, not the formatted message, so a
        # message that legitimately differs per call (e.g. a shape folded
        # into the text) still dedupes correctly on the shared key.
        logger = get_inductor_logger("test_warn_once_varying_message")
        with patch.object(logger, "warning") as mock_warning:
            warn_once(logger, "opX", "skipping opX, shape=%s", [1, 2])
            warn_once(logger, "opX", "skipping opX, shape=%s", [3, 4])
        mock_warning.assert_called_once_with("skipping opX, shape=%s", [1, 2])

    def test_same_key_different_logger_not_cross_suppressed(self):
        logger_a = get_inductor_logger("test_warn_once_logger_a")
        logger_b = get_inductor_logger("test_warn_once_logger_b")
        with patch.object(logger_a, "warning") as mock_a:
            with patch.object(logger_b, "warning") as mock_b:
                warn_once(logger_a, "opX", "skipping %s", "opX")
                warn_once(logger_b, "opX", "skipping %s", "opX")
        mock_a.assert_called_once()
        mock_b.assert_called_once()
