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


"""Tests for per-pass operation logging in CustomPreSchedulingPasses."""

import logging
import os
import functools
import subprocess
import sys
from unittest.mock import MagicMock, patch

import torch  # noqa: F401

from torch_spyre._inductor import config
from torch_spyre._inductor.passes import _get_pass_name, _should_log_pass


class TestGetPassName:
    """Tests for _get_pass_name helper."""

    def test_various_callable_types(self):
        def my_pass(graph):
            pass

        assert _get_pass_name(my_pass) == "my_pass"

        fn = lambda graph: None  # noqa: E731
        assert _get_pass_name(fn) == "<lambda>"

        class MyPass:
            def run(self, graph):
                pass

        obj = MyPass()
        assert _get_pass_name(obj.run) == "run"

        class MyCallablePass:
            def __call__(self, graph):
                pass

        callable_obj = MyCallablePass()
        assert _get_pass_name(callable_obj) == "MyCallablePass"

    def test_decorated_function_preserves_name(self):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        @decorator
        def insert_restickify(graph):
            pass

        assert _get_pass_name(insert_restickify) == "insert_restickify"


class TestShouldLogPass:
    """Tests for _should_log_pass helper."""

    def test_disabled_when_empty(self):
        with config.patch({"log_passes": ""}):
            assert _should_log_pass("split_multi_ops") is False

    def test_all_and_one_enable_everything(self):
        with config.patch({"log_passes": "all"}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is True

        with config.patch({"log_passes": "1"}):
            assert _should_log_pass("any_pass_name") is True

    def test_selective_name_matching(self):
        with config.patch({"log_passes": "split_multi_ops"}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is False

        with config.patch({"log_passes": "split_multi_ops,insert_restickify"}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is True
            assert _should_log_pass("deadcode_elimination") is False

        with config.patch({"log_passes": " split_multi_ops , insert_restickify "}):
            assert _should_log_pass("split_multi_ops") is True
            assert _should_log_pass("insert_restickify") is True

        # No partial matching
        with config.patch({"log_passes": "split_multi"}):
            assert _should_log_pass("split_multi_ops") is False


class TestLogPassesConfig:
    """Tests for the log_passes configuration knob."""

    def test_default_when_env_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "SPYRE_LOG_PASSES"}
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from torch_spyre._inductor import config; print(config.log_passes)",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ""

    def test_env_var_populates_config(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from torch_spyre._inductor import config; print(config.log_passes)",
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "SPYRE_LOG_PASSES": "split_multi_ops,deadcode_elimination",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            },
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "split_multi_ops,deadcode_elimination"


def _make_spyre_graph():
    """Create a minimal mock GraphLowering with a Spyre-device operation."""
    op = MagicMock()
    op.get_device.return_value = torch.device("spyre")
    op.get_name.return_value = "buf0"
    graph = MagicMock()
    graph.operations = [op]
    return graph


class TestPerPassLoggingIntegration:
    """Integration tests for per-pass logging in CustomPreSchedulingPasses."""

    def test_logs_after_pass_when_enabled(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph()

        def my_pass(g):
            pass

        passes_obj = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        passes_obj.passes = [my_pass]

        with config.patch({"log_passes": "all"}):
            with patch("torch_spyre._inductor.passes.logger") as mock_logger:
                mock_logger.isEnabledFor.return_value = True
                passes_obj(graph)

                debug_calls = [
                    c
                    for c in mock_logger.debug.call_args_list
                    if len(c.args) >= 2
                    and "AFTER" in c.args[0]
                    and c.args[1] == "my_pass"
                ]
                assert len(debug_calls) == 1

    def test_no_log_when_disabled(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph()

        def my_pass(g):
            pass

        passes_obj = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        passes_obj.passes = [my_pass]

        # Config empty — no per-pass logging
        with config.patch({"log_passes": ""}):
            with patch("torch_spyre._inductor.passes.logger") as mock_logger:
                mock_logger.isEnabledFor.return_value = True
                passes_obj(graph)

                debug_calls = [
                    c
                    for c in mock_logger.debug.call_args_list
                    if len(c.args) >= 2
                    and "AFTER" in c.args[0]
                    and c.args[1] == "my_pass"
                ]
                assert len(debug_calls) == 0

        # Logger level too high — DEBUG guard blocks output
        with config.patch({"log_passes": "all"}):
            with patch("torch_spyre._inductor.passes.logger") as mock_logger:
                mock_logger.isEnabledFor.side_effect = (
                    lambda level: level != logging.DEBUG
                )
                passes_obj(graph)

                debug_calls = [
                    c
                    for c in mock_logger.debug.call_args_list
                    if len(c.args) >= 2
                    and "AFTER" in c.args[0]
                    and c.args[1] == "my_pass"
                ]
                assert len(debug_calls) == 0

    def test_selective_pass_logging(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        graph = _make_spyre_graph()

        def pass_a(g):
            pass

        def pass_b(g):
            pass

        passes_obj = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        passes_obj.passes = [pass_a, pass_b]

        with config.patch({"log_passes": "pass_b"}):
            with patch("torch_spyre._inductor.passes.logger") as mock_logger:
                mock_logger.isEnabledFor.return_value = True
                passes_obj(graph)

                debug_calls = mock_logger.debug.call_args_list
                after_calls = [
                    c for c in debug_calls if len(c.args) >= 2 and "AFTER" in c.args[0]
                ]
                assert any(c.args[1] == "pass_b" for c in after_calls)
                assert not any(c.args[1] == "pass_a" for c in after_calls)

    def test_early_return_when_no_spyre_device(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses

        op = MagicMock()
        op.get_device.return_value = torch.device("cpu")
        graph = MagicMock()
        graph.operations = [op]

        def my_pass(g):
            raise AssertionError("pass should not be called")

        passes_obj = CustomPreSchedulingPasses.__new__(CustomPreSchedulingPasses)
        passes_obj.passes = [my_pass]

        with config.patch({"log_passes": "all"}):
            with patch("torch_spyre._inductor.passes.logger") as mock_logger:
                mock_logger.isEnabledFor.return_value = True
                passes_obj(graph)
                assert mock_logger.debug.call_count == 0
