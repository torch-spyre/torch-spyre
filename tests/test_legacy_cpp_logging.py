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

"""Tests for legacy C++ logging behavior under the unified logging system.

Each test spawns a fresh subprocess to avoid process-global state
(LoggingConfig singleton, g_debug_info_enabled) leaking between cases.
"""

import regex as re

from cpp_logging_test_utils import run_subprocess as _run_subprocess


class TestLegacyCppLoggingSilent:
    """Verify C++ debug output is suppressed when no debug env vars are set."""

    def test_legacy_debug_silent_by_default(self):
        """No legacy or TORCH_LOGS vars: stderr must not contain [DEBUG] lines."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "[DEBUG]" not in result.stderr
        assert "[DEBUG]" not in result.stdout


class TestLegacyEnvVarEnablesDebug:
    """Verify TORCH_SPYRE_DEBUG=1 enables C++ debug via the legacy shim."""

    def test_legacy_env_var_enables_debug(self):
        """TORCH_SPYRE_DEBUG=1 must configure spyre.runtime at DEBUG level."""
        script = """
            import warnings
            warnings.simplefilter("always")
            import torch  # noqa: F401
            import torch_spyre
            from torch_spyre import logging_config

            level = logging_config.get_log_level("spyre.runtime")
            source = logging_config.get_config_source("spyre.runtime")
            print(f"LEVEL={level.name}")
            print(f"SOURCE={source}")
        """
        result = _run_subprocess(script, {"TORCH_SPYRE_DEBUG": "1"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=DEBUG" in result.stdout
        assert "SOURCE=legacy:TORCH_SPYRE_DEBUG" in result.stdout

    def test_legacy_env_var_emits_deprecation_warning(self):
        """TORCH_SPYRE_DEBUG=1 must emit a deprecation warning."""
        script = """
            import warnings
            warnings.simplefilter("always")
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
        """
        result = _run_subprocess(script, {"TORCH_SPYRE_DEBUG": "1"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "deprecated" in combined.lower() or "TORCH_SPYRE_DEBUG" in combined


class TestTorchLogsEnablesRuntimeDebug:
    """Verify TORCH_LOGS=spyre.runtime:DEBUG enables C++ debug output."""

    def test_torch_logs_enables_runtime_debug(self):
        """TORCH_LOGS configures spyre.runtime at DEBUG without deprecation."""
        script = """
            import os
            import warnings
            warnings.simplefilter("always")
            _saved = os.environ.pop("TORCH_LOGS", None)
            import torch  # noqa: F401
            if _saved is not None:
                os.environ["TORCH_LOGS"] = _saved
            import torch_spyre
            from torch_spyre import logging_config
            logging_config.reset()

            level = logging_config.get_log_level("spyre.runtime")
            source = logging_config.get_config_source("spyre.runtime")
            print(f"LEVEL={level.name}")
            print(f"SOURCE={source}")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.runtime:DEBUG"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=DEBUG" in result.stdout
        assert "SOURCE=TORCH_LOGS" in result.stdout
        combined = result.stdout + result.stderr
        assert "TORCH_SPYRE_DEBUG" not in combined
        assert "SPYRE_INDUCTOR_LOG" not in combined

    def test_cpp_logging_config_reflects_torch_logs(self):
        """C++ LoggingConfig singleton sees the level set by TORCH_LOGS."""
        script = """
            import os
            _saved = os.environ.pop("TORCH_LOGS", None)
            import torch  # noqa: F401
            if _saved is not None:
                os.environ["TORCH_LOGS"] = _saved
            import torch_spyre  # noqa: F401
            from torch_spyre import logging_config
            logging_config.reset()
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            enabled = config.is_enabled(
                "spyre.runtime", cpp_logging.LogLevel.DEBUG
            )
            print(f"CPP_ENABLED={enabled}")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.runtime:DEBUG"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "CPP_ENABLED=True" in result.stdout


class TestTorchLogsPriorityOverLegacy:
    """Verify TORCH_LOGS takes precedence over legacy env vars."""

    def test_torch_logs_takes_priority_over_legacy(self):
        """TORCH_LOGS=WARNING overrides TORCH_SPYRE_DEBUG=1 (would be DEBUG)."""
        script = """
            import os
            import warnings
            warnings.simplefilter("always")
            _saved = os.environ.pop("TORCH_LOGS", None)
            import torch  # noqa: F401
            if _saved is not None:
                os.environ["TORCH_LOGS"] = _saved
            import torch_spyre
            from torch_spyre import logging_config
            logging_config.reset()

            level = logging_config.get_log_level("spyre.runtime")
            source = logging_config.get_config_source("spyre.runtime")
            print(f"LEVEL={level.name}")
            print(f"SOURCE={source}")
        """
        result = _run_subprocess(
            script,
            {
                "TORCH_SPYRE_DEBUG": "1",
                "TORCH_LOGS": "spyre.runtime:WARNING",
            },
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=WARNING" in result.stdout
        assert "SOURCE=TORCH_LOGS" in result.stdout


class TestOutputFormatMatchesSpec:
    """Verify C++ log output format conforms to the unified spec."""

    def test_output_format_matches_spec(self):
        """C++ Logger output must match [LEVEL] [component] timestamp message."""
        script = """
            import os
            _saved = os.environ.pop("TORCH_LOGS", None)
            import torch  # noqa: F401
            if _saved is not None:
                os.environ["TORCH_LOGS"] = _saved
            import torch_spyre  # noqa: F401
            from torch_spyre import logging_config
            logging_config.reset()
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.runtime", cpp_logging.LogLevel.DEBUG)
            enabled = config.is_enabled(
                "spyre.runtime", cpp_logging.LogLevel.DEBUG
            )
            print(f"ENABLED={enabled}")
            level = config.get_log_level("spyre.runtime")
            level_str = cpp_logging.log_level_to_string(level)
            print(f"LEVEL_STR={level_str}")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.runtime:DEBUG"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "ENABLED=True" in result.stdout
        assert "LEVEL_STR=DEBUG" in result.stdout

    def test_debug_format_regex_when_output_present(self):
        """If any [DEBUG] line appears on stderr, it must match the format spec."""
        script = """
            import os
            _saved = os.environ.pop("TORCH_LOGS", None)
            import torch  # noqa: F401
            if _saved is not None:
                os.environ["TORCH_LOGS"] = _saved
            import torch_spyre  # noqa: F401
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.runtime:DEBUG"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        pattern = re.compile(
            r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\] "
            r"\[spyre\.[a-z._]+\] "
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
            r".+"
        )
        for line in result.stderr.splitlines():
            if "[DEBUG]" in line or "[INFO]" in line:
                assert pattern.match(line), (
                    f"Log line does not match expected format: {line!r}"
                )
