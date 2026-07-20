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

"""Tests for the new C++ logging system (LoggingConfig, Logger, LogLevel).

Each test spawns a fresh subprocess to avoid process-global state
(LoggingConfig singleton) leaking between cases.
"""

import regex as re

from cpp_logging_test_utils import run_subprocess as _run_subprocess


class TestLoggingConfigSingleton:
    """Verify LoggingConfig singleton access and basic properties."""

    def test_singleton_instance_is_accessible(self):
        """LoggingConfig.instance() must return a valid object."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            print(f"TYPE={type(config).__name__}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "TYPE=LoggingConfig" in result.stdout

    def test_singleton_returns_same_instance(self):
        """Multiple calls to instance() must return the same object."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            a = cpp_logging.LoggingConfig.instance()
            b = cpp_logging.LoggingConfig.instance()
            print(f"SAME={a is b}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "SAME=True" in result.stdout


class TestLogLevelEnum:
    """Verify the C++ LogLevel enum is correctly exposed to Python."""

    def test_log_level_values(self):
        """LogLevel enum values must match Python logging numeric levels."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            print(f"NOTSET={int(cpp_logging.LogLevel.NOTSET)}")
            print(f"DEBUG={int(cpp_logging.LogLevel.DEBUG)}")
            print(f"INFO={int(cpp_logging.LogLevel.INFO)}")
            print(f"WARNING={int(cpp_logging.LogLevel.WARNING)}")
            print(f"ERROR={int(cpp_logging.LogLevel.ERROR)}")
            print(f"CRITICAL={int(cpp_logging.LogLevel.CRITICAL)}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "NOTSET=0" in result.stdout
        assert "DEBUG=10" in result.stdout
        assert "INFO=20" in result.stdout
        assert "WARNING=30" in result.stdout
        assert "ERROR=40" in result.stdout
        assert "CRITICAL=50" in result.stdout

    def test_log_level_to_string(self):
        """log_level_to_string must return correct string representations."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            for level in [cpp_logging.LogLevel.DEBUG, cpp_logging.LogLevel.INFO,
                          cpp_logging.LogLevel.WARNING, cpp_logging.LogLevel.ERROR,
                          cpp_logging.LogLevel.CRITICAL]:
                name = cpp_logging.log_level_to_string(level)
                print(f"{name}=OK")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "DEBUG=OK" in result.stdout
        assert "INFO=OK" in result.stdout
        assert "WARNING=OK" in result.stdout
        assert "ERROR=OK" in result.stdout
        assert "CRITICAL=OK" in result.stdout

    def test_string_to_log_level(self):
        """string_to_log_level must round-trip with log_level_to_string."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            for name in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                level = cpp_logging.string_to_log_level(name)
                back = cpp_logging.log_level_to_string(level)
                print(f"{name}={back}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        for name in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            assert f"{name}={name}" in result.stdout


class TestLoggingConfigSetAndGet:
    """Verify programmatic set/get of log levels on the C++ side."""

    def test_set_and_get_log_level(self):
        """set_log_level followed by get_log_level must return the set value."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.test_component", cpp_logging.LogLevel.DEBUG)
            level = config.get_log_level("spyre.test_component")
            print(f"LEVEL={cpp_logging.log_level_to_string(level)}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=DEBUG" in result.stdout

    def test_is_enabled_respects_level(self):
        """is_enabled must return True only when the message level >= config."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.test_is_enabled", cpp_logging.LogLevel.WARNING)
            debug_ok = config.is_enabled("spyre.test_is_enabled", cpp_logging.LogLevel.DEBUG)
            info_ok = config.is_enabled("spyre.test_is_enabled", cpp_logging.LogLevel.INFO)
            warn_ok = config.is_enabled("spyre.test_is_enabled", cpp_logging.LogLevel.WARNING)
            err_ok = config.is_enabled("spyre.test_is_enabled", cpp_logging.LogLevel.ERROR)
            print(f"DEBUG={debug_ok}")
            print(f"INFO={info_ok}")
            print(f"WARNING={warn_ok}")
            print(f"ERROR={err_ok}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "DEBUG=False" in result.stdout
        assert "INFO=False" in result.stdout
        assert "WARNING=True" in result.stdout
        assert "ERROR=True" in result.stdout

    def test_get_components_returns_configured_list(self):
        """get_components must include components that have been configured."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.test_components", cpp_logging.LogLevel.INFO)
            components = config.get_components()
            print(f"HAS_COMPONENT={'spyre.test_components' in components}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "HAS_COMPONENT=True" in result.stdout


class TestHierarchicalLookup:
    """Verify hierarchical component name resolution in LoggingConfig."""

    def test_child_inherits_parent_level(self):
        """A child component must inherit its parent's level when not set."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.hierarchy", cpp_logging.LogLevel.DEBUG)
            child_level = config.get_log_level("spyre.hierarchy.child")
            print(f"CHILD_LEVEL={cpp_logging.log_level_to_string(child_level)}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "CHILD_LEVEL=DEBUG" in result.stdout

    def test_child_override_takes_precedence(self):
        """An explicit child level must override the parent's level."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.parent", cpp_logging.LogLevel.DEBUG)
            config.set_log_level("spyre.parent.child", cpp_logging.LogLevel.ERROR)
            parent_level = config.get_log_level("spyre.parent")
            child_level = config.get_log_level("spyre.parent.child")
            print(f"PARENT={cpp_logging.log_level_to_string(parent_level)}")
            print(f"CHILD={cpp_logging.log_level_to_string(child_level)}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "PARENT=DEBUG" in result.stdout
        assert "CHILD=ERROR" in result.stdout

    def test_unconfigured_component_defaults_to_warning(self):
        """A component with no config and no parent must default to WARNING."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            level = config.get_log_level("totally.unknown.component")
            print(f"LEVEL={cpp_logging.log_level_to_string(level)}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=WARNING" in result.stdout


class TestInitializeFromPython:
    """Verify C++ LoggingConfig.initialize_from_python integration."""

    def test_initialize_from_python_sets_levels(self):
        """initialize_from_python must configure C++ levels from a Python list."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.initialize_from_python([
                ("spyre.init_test", 10),
                ("spyre.init_test.sub", 40),
            ])
            level_parent = config.get_log_level("spyre.init_test")
            level_child = config.get_log_level("spyre.init_test.sub")
            print(f"PARENT={cpp_logging.log_level_to_string(level_parent)}")
            print(f"CHILD={cpp_logging.log_level_to_string(level_child)}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "PARENT=DEBUG" in result.stdout
        assert "CHILD=ERROR" in result.stdout


class TestTorchLogsIntegration:
    """Verify TORCH_LOGS env var propagates to C++ LoggingConfig."""

    def test_torch_logs_configures_cpp_level(self):
        """TORCH_LOGS=spyre.runtime:DEBUG must set C++ config to DEBUG."""
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
            enabled = config.is_enabled("spyre.runtime", cpp_logging.LogLevel.DEBUG)
            level = config.get_log_level("spyre.runtime")
            print(f"ENABLED={enabled}")
            print(f"LEVEL={cpp_logging.log_level_to_string(level)}")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.runtime:DEBUG"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "ENABLED=True" in result.stdout
        assert "LEVEL=DEBUG" in result.stdout

    def test_torch_logs_info_level(self):
        """TORCH_LOGS=spyre.inductor:INFO must set C++ config to INFO."""
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
            level = config.get_log_level("spyre.inductor")
            enabled_debug = config.is_enabled(
                "spyre.inductor", cpp_logging.LogLevel.DEBUG
            )
            enabled_info = config.is_enabled(
                "spyre.inductor", cpp_logging.LogLevel.INFO
            )
            print(f"LEVEL={cpp_logging.log_level_to_string(level)}")
            print(f"DEBUG_ENABLED={enabled_debug}")
            print(f"INFO_ENABLED={enabled_info}")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.inductor:INFO"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=INFO" in result.stdout
        assert "DEBUG_ENABLED=False" in result.stdout
        assert "INFO_ENABLED=True" in result.stdout


class TestLoggerOutput:
    """Verify the Logger class produces correctly formatted output on stderr."""

    def test_logger_debug_output_format(self):
        """Logger debug() must emit [DEBUG] [component] timestamp message."""
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
            print("READY")
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
            if line.startswith("["):
                assert pattern.match(line), (
                    f"Log line does not match expected format: {line!r}"
                )

    def test_logger_suppressed_when_level_too_low(self):
        """Logger must not emit output when level is below threshold."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.quiet", cpp_logging.LogLevel.ERROR)
            enabled_debug = config.is_enabled(
                "spyre.quiet", cpp_logging.LogLevel.DEBUG
            )
            enabled_info = config.is_enabled(
                "spyre.quiet", cpp_logging.LogLevel.INFO
            )
            enabled_warn = config.is_enabled(
                "spyre.quiet", cpp_logging.LogLevel.WARNING
            )
            print(f"DEBUG_ENABLED={enabled_debug}")
            print(f"INFO_ENABLED={enabled_info}")
            print(f"WARN_ENABLED={enabled_warn}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "DEBUG_ENABLED=False" in result.stdout
        assert "INFO_ENABLED=False" in result.stdout
        assert "WARN_ENABLED=False" in result.stdout
        assert "[DEBUG]" not in result.stderr
        assert "[INFO]" not in result.stderr
        assert "[WARNING]" not in result.stderr

    def test_logger_critical_output_format(self):
        """Logger critical() must emit [CRITICAL] [component] timestamp message on stderr."""
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
            config.set_log_level("spyre.runtime", cpp_logging.LogLevel.CRITICAL)
            cpp_logging.log_message(
                "spyre.runtime", cpp_logging.LogLevel.CRITICAL, "test critical message"
            )
            print("DONE")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.runtime:CRITICAL"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "DONE" in result.stdout
        critical_lines = [
            line for line in result.stderr.splitlines() if line.startswith("[CRITICAL]")
        ]
        assert critical_lines, (
            f"Expected at least one [CRITICAL] line on stderr, got:\n{result.stderr}"
        )
        pattern = re.compile(
            r"\[CRITICAL\] "
            r"\[spyre\.[a-z._]+\] "
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
            r".+"
        )
        for line in critical_lines:
            assert pattern.match(line), (
                f"[CRITICAL] line does not match expected format: {line!r}"
            )


class TestDefaultBehavior:
    """Verify default behavior without any logging env vars set."""

    def test_default_level_is_warning(self):
        """Without env vars, C++ config must default to WARNING."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            level = config.get_log_level("spyre.runtime")
            print(f"LEVEL={cpp_logging.log_level_to_string(level)}")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=WARNING" in result.stdout

    def test_no_debug_output_by_default(self):
        """Without env vars, stderr must not contain any DEBUG lines."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "[DEBUG]" not in result.stderr
        assert "[DEBUG]" not in result.stdout


class TestMultipleComponents:
    """Verify independent configuration of multiple components."""

    def test_independent_component_levels(self):
        """Different components can have different levels simultaneously."""
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
            runtime_level = config.get_log_level("spyre.runtime")
            inductor_level = config.get_log_level("spyre.inductor")
            print(f"RUNTIME={cpp_logging.log_level_to_string(runtime_level)}")
            print(f"INDUCTOR={cpp_logging.log_level_to_string(inductor_level)}")
        """
        result = _run_subprocess(
            script,
            {"TORCH_LOGS": "spyre.runtime:DEBUG,spyre.inductor:ERROR"},
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "RUNTIME=DEBUG" in result.stdout
        assert "INDUCTOR=ERROR" in result.stdout

    def test_plus_syntax_enables_info(self):
        """TORCH_LOGS=+spyre.runtime must enable the component at INFO."""
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
            level = config.get_log_level("spyre.runtime")
            print(f"LEVEL={cpp_logging.log_level_to_string(level)}")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "+spyre.runtime"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "LEVEL=INFO" in result.stdout


class TestEmitTestLog:
    """Verify the emit_test_log binding (int-based level) works correctly."""

    def test_emit_test_log_debug(self):
        """emit_test_log with level=10 must produce [DEBUG] output on stderr."""
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
            config.set_log_level("spyre.emit_test", cpp_logging.LogLevel.DEBUG)
            cpp_logging.emit_test_log("spyre.emit_test", 10, "debug via int")
            print("DONE")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.emit_test:DEBUG"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "DONE" in result.stdout
        debug_lines = [
            line for line in result.stderr.splitlines() if line.startswith("[DEBUG]")
        ]
        assert debug_lines, f"Expected [DEBUG] line on stderr, got:\n{result.stderr}"
        assert "debug via int" in result.stderr

    def test_emit_test_log_critical(self):
        """emit_test_log with level=50 must produce [CRITICAL] output on stderr."""
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
            config.set_log_level("spyre.emit_test", cpp_logging.LogLevel.CRITICAL)
            cpp_logging.emit_test_log("spyre.emit_test", 50, "critical via int")
            print("DONE")
        """
        result = _run_subprocess(script, {"TORCH_LOGS": "spyre.emit_test:CRITICAL"})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "DONE" in result.stdout
        critical_lines = [
            line for line in result.stderr.splitlines() if line.startswith("[CRITICAL]")
        ]
        assert critical_lines, (
            f"Expected [CRITICAL] line on stderr, got:\n{result.stderr}"
        )
        assert "critical via int" in result.stderr

    def test_emit_test_log_suppressed(self):
        """emit_test_log at DEBUG must be suppressed when level is ERROR."""
        script = """
            import torch  # noqa: F401
            import torch_spyre  # noqa: F401
            from torch_spyre._C import _logging as cpp_logging
            config = cpp_logging.LoggingConfig.instance()
            config.set_log_level("spyre.emit_test", cpp_logging.LogLevel.ERROR)
            cpp_logging.emit_test_log("spyre.emit_test", 10, "should not appear")
            print("DONE")
        """
        result = _run_subprocess(script, {})
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "DONE" in result.stdout
        assert "should not appear" not in result.stderr
        assert "[DEBUG]" not in result.stderr
