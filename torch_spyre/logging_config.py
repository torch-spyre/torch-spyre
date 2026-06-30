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

"""Unified logging configuration for torch-spyre.

This module provides a centralized logging configuration system that:
1. Parses TORCH_LOGS environment variable for spyre.* namespaces
2. Maintains backward compatibility with legacy environment variables
3. Exposes configuration to C++ via pybind11
4. Provides programmatic API for runtime configuration
5. Configures hierarchical Python logging handlers for the spyre namespace
"""

import logging
import os
import warnings
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class LogLevel(IntEnum):
    """Standard Python logging levels."""

    NOTSET = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    DISABLED = 60


DEFAULT_LOG_LEVELS = {
    "spyre": LogLevel.WARNING,
    "spyre.inductor": LogLevel.WARNING,
    "spyre.inductor.lowering": LogLevel.WARNING,
    "spyre.inductor.stickify": LogLevel.WARNING,
    "spyre.inductor.codegen": LogLevel.WARNING,
    "spyre.inductor.passes": LogLevel.WARNING,
    "spyre.runtime": LogLevel.WARNING,
    "spyre.execution": LogLevel.WARNING,
    "spyre.device": LogLevel.WARNING,
}

_config: Dict[str, LogLevel] = {}
_config_source: Dict[str, str] = {}
_log_file_path: Optional[str] = None
_log_file_source: str = "default"
_initialized = False
_python_logging_configured = False
_lock = None  # Will be threading.Lock() after import


def _get_lock():
    """Lazy initialization of lock to avoid import issues."""
    global _lock
    if _lock is None:
        import threading

        _lock = threading.RLock()
    return _lock


def _parse_torch_logs() -> Dict[str, LogLevel]:
    """Parse TORCH_LOGS environment variable for spyre namespaces.

    Supported formats:
    - TORCH_LOGS="spyre.inductor:DEBUG"
    - TORCH_LOGS="+spyre.inductor"  (enables at INFO)
    - TORCH_LOGS="-spyre.inductor"  (disables)
    - TORCH_LOGS="spyre:INFO,spyre.inductor:DEBUG"

    Returns:
        Dictionary mapping component names to log levels
    """
    config: Dict[str, LogLevel] = {}
    torch_logs = os.environ.get("TORCH_LOGS", "")

    if not torch_logs:
        return config

    for entry in torch_logs.split(","):
        entry = entry.strip()
        if not entry:
            continue

        if entry.startswith("+"):
            component = entry[1:]
            if component.startswith("spyre"):
                config[component] = LogLevel.INFO
                _config_source[component] = "TORCH_LOGS"
        elif entry.startswith("-"):
            component = entry[1:]
            if component.startswith("spyre"):
                config[component] = LogLevel.DISABLED
                _config_source[component] = "TORCH_LOGS"
        elif ":" in entry:
            component, level_str = entry.split(":", 1)
            component = component.strip()
            level_str = level_str.strip()
            if component.startswith("spyre"):
                try:
                    level = getattr(LogLevel, level_str.upper())
                    config[component] = level
                    _config_source[component] = "TORCH_LOGS"
                except AttributeError:
                    warnings.warn(
                        f"Invalid log level '{level_str}' for {component}",
                        stacklevel=3,
                    )

    return config


def _parse_legacy_vars() -> Dict[str, LogLevel]:
    """Parse legacy environment variables with deprecation warnings.

    Legacy variables:
    - SPYRE_INDUCTOR_LOG=1
    - SPYRE_INDUCTOR_LOG_LEVEL=DEBUG
    - TORCH_SPYRE_DEBUG=1
    - SPYRE_LOG_FILE=/path/to/file.log

    Returns:
        Dictionary mapping component names to log levels
    """
    global _log_file_path, _log_file_source

    config: Dict[str, LogLevel] = {}

    if os.environ.get("SPYRE_INDUCTOR_LOG") == "1":
        warnings.warn(
            "SPYRE_INDUCTOR_LOG is deprecated. Use TORCH_LOGS='spyre.inductor:INFO' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        level_str = os.environ.get("SPYRE_INDUCTOR_LOG_LEVEL", "INFO")
        try:
            level = getattr(LogLevel, level_str.upper())
            config["spyre.inductor"] = level
            _config_source["spyre.inductor"] = "legacy:SPYRE_INDUCTOR_LOG"
        except AttributeError:
            config["spyre.inductor"] = LogLevel.INFO
            _config_source["spyre.inductor"] = "legacy:SPYRE_INDUCTOR_LOG"

    if os.environ.get("TORCH_SPYRE_DEBUG") == "1":
        warnings.warn(
            "TORCH_SPYRE_DEBUG is deprecated. Use TORCH_LOGS='spyre:DEBUG' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        for component in DEFAULT_LOG_LEVELS:
            if component not in config:
                config[component] = LogLevel.DEBUG
                _config_source[component] = "legacy:TORCH_SPYRE_DEBUG"

    legacy_log_file = os.environ.get("SPYRE_LOG_FILE")
    if legacy_log_file:
        warnings.warn(
            "SPYRE_LOG_FILE is deprecated. It is mapped to the top-level "
            "'spyre' logger file handler for backward compatibility.",
            DeprecationWarning,
            stacklevel=3,
        )
        _log_file_path = legacy_log_file
        _log_file_source = "legacy:SPYRE_LOG_FILE"

    return config


def _resolve_config() -> Dict[str, LogLevel]:
    """Resolve final configuration from all sources.

    Priority order:
    1. TORCH_LOGS
    2. Legacy environment variables
    3. Programmatic API (applied later)
    4. Defaults

    Returns:
        Resolved configuration dictionary
    """
    config = DEFAULT_LOG_LEVELS.copy()

    legacy_config = _parse_legacy_vars()
    config.update(legacy_config)

    torch_logs_config = _parse_torch_logs()
    config.update(torch_logs_config)

    # When a user explicitly configures a parent component, propagate that
    # level to any more-specific defaults that would otherwise shadow it.
    # For example, TORCH_LOGS='+spyre.inductor' should override the default
    # WARNING entry for 'spyre.inductor.codegen' so that child loggers like
    # 'spyre.inductor.codegen.superdsc' resolve to the user-specified level.
    explicit_sources = {
        "TORCH_LOGS",
        "legacy:SPYRE_INDUCTOR_LOG",
        "legacy:TORCH_SPYRE_DEBUG",
    }
    for component in list(config):
        if _config_source.get(component, "default") not in explicit_sources:
            # Check if a less-specific ancestor was explicitly configured
            parts = component.split(".")
            for i in range(len(parts) - 1, 0, -1):
                parent = ".".join(parts[:i])
                if _config_source.get(parent, "default") in explicit_sources:
                    config[component] = config[parent]
                    _config_source[component] = _config_source[parent]
                    break

    for component in config:
        if component not in _config_source:
            _config_source[component] = "default"

    return config


def _make_formatter() -> logging.Formatter:
    """Create the default formatter for spyre loggers."""
    return logging.Formatter("[%(levelname)s] [%(name)s] %(message)s")


def configure_python_logging():
    """Configure the top-level hierarchical Python logger for spyre.

    This is idempotent and safe to call multiple times.
    """
    global _python_logging_configured

    if _python_logging_configured:
        return

    if not _initialized:
        initialize()

    with _get_lock():
        spyre_logger = logging.getLogger("spyre")
        spyre_logger.setLevel(int(get_log_level("spyre")))

        desired_file = _log_file_path
        formatter = _make_formatter()

        existing_file_handlers = [
            handler
            for handler in spyre_logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        existing_stream_handlers = [
            handler
            for handler in spyre_logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ]

        if desired_file:
            file_handler_present = any(
                getattr(handler, "baseFilename", None) == os.path.abspath(desired_file)
                for handler in existing_file_handlers
            )
            if not file_handler_present:
                handler = logging.FileHandler(desired_file)
                handler.setFormatter(formatter)
                spyre_logger.addHandler(handler)

        if not existing_stream_handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            spyre_logger.addHandler(handler)

        _python_logging_configured = True


def initialize():
    """Initialize logging configuration from environment variables.

    This should be called once during module initialization.
    Thread-safe and idempotent.
    """
    global _config, _initialized

    with _get_lock():
        if _initialized:
            return

        _config = _resolve_config()
        _initialized = True

    configure_python_logging()


def get_log_level(component: str) -> LogLevel:
    """Get effective log level for a component.

    Args:
        component: Component name (e.g., "spyre.inductor")

    Returns:
        Effective log level for the component
    """
    if not _initialized:
        initialize()

    if component in _config:
        return _config[component]

    parts = component.split(".")
    for i in range(len(parts), 0, -1):
        parent = ".".join(parts[:i])
        if parent in _config:
            return _config[parent]

    return LogLevel.WARNING


def set_log_level(component: str, level: str):
    """Set log level for a component programmatically.

    Args:
        component: Component name (e.g., "spyre.inductor")
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL, DISABLED)
    """
    if not _initialized:
        initialize()

    try:
        level_enum = getattr(LogLevel, level.upper())
    except AttributeError as exc:
        raise ValueError(f"Invalid log level: {level}") from exc

    with _get_lock():
        _config[component] = level_enum
        _config_source[component] = "programmatic"

        logger = logging.getLogger(component)
        logger.setLevel(int(level_enum))

        if component == "spyre":
            root_logger = logging.getLogger("spyre")
            root_logger.setLevel(int(level_enum))


def enable(component: str):
    """Enable logging for a component at INFO level.

    Args:
        component: Component name (e.g., "spyre.inductor")
    """
    set_log_level(component, "INFO")


def disable(component: str):
    """Disable logging for a component.

    Args:
        component: Component name (e.g., "spyre.inductor")
    """
    set_log_level(component, "DISABLED")


def get_log_file() -> Optional[str]:
    """Get the configured log file path, if any."""
    if not _initialized:
        initialize()
    return _log_file_path


def set_log_file(path: Optional[str]):
    """Set the log file path programmatically."""
    global _log_file_path, _log_file_source, _python_logging_configured

    if not _initialized:
        initialize()

    with _get_lock():
        _log_file_path = path
        _log_file_source = "programmatic" if path else "default"
        _python_logging_configured = False
        configure_python_logging()


def get_effective_config() -> Dict[str, str]:
    """Get effective configuration for all components.

    Returns:
        Dictionary mapping component names to level names
    """
    if not _initialized:
        initialize()

    return {component: level.name for component, level in _config.items()}


def get_output_config() -> Dict[str, Optional[str]]:
    """Get effective output configuration."""
    if not _initialized:
        initialize()

    return {
        "log_file": _log_file_path,
        "log_file_source": _log_file_source,
    }


def get_config_source(component: str) -> str:
    """Get configuration source for a component.

    Args:
        component: Component name

    Returns:
        Source name: "TORCH_LOGS", "legacy", "programmatic", or "default"
    """
    if not _initialized:
        initialize()

    return _config_source.get(component, "default")


def list_components() -> List[str]:
    """List all available logging components.

    Returns:
        List of component names
    """
    return list(DEFAULT_LOG_LEVELS.keys())


def get_config_for_cpp() -> List[Tuple[str, int]]:
    """Get configuration in format suitable for C++.

    Returns:
        List of (component, level) tuples with integer levels
    """
    if not _initialized:
        initialize()

    return [(comp, int(level)) for comp, level in _config.items()]


initialize()
