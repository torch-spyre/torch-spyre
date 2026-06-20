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


"""Inductor logging utilities using unified hierarchical configuration.

This module provides the new hierarchical logger API and preserves
backward-compatible helper functions used elsewhere in the codebase.
"""

import logging
import os

from torch_spyre import logging_config

# Cache for logger instances
_loggers: dict[str, logging.Logger] = {}

# Flag set by tests to trigger re-initialization on next get_logger() call
_needs_reinit: bool = False


def _reinitialize():
    """Force re-initialization of logging config from current environment.

    Called when _needs_reinit is True (set by tests to pick up changed
    environment variables between test cases).
    """
    global _INDUCTOR_LOGGING_ENABLED, _needs_reinit
    _loggers.clear()
    logging_config._initialized = False
    logging_config._config.clear()
    logging_config._config_source.clear()
    logging_config._python_logging_configured = False
    logging_config.initialize()
    logging_config.configure_python_logging()
    _INDUCTOR_LOGGING_ENABLED = _get_env_bool("SPYRE_INDUCTOR_LOG", False)
    _needs_reinit = False


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger for the given name.

    Args:
        name: Logger name (e.g., "lowering", "codegen")
            Will be prefixed with "spyre.inductor."

    Returns:
        Configured logger instance
    """
    full_name = f"spyre.inductor.{name}"

    if _needs_reinit:
        _reinitialize()

    if full_name in _loggers:
        return _loggers[full_name]

    logger = logging.getLogger(full_name)
    logger.setLevel(int(logging_config.get_log_level(full_name)))

    _loggers[full_name] = logger
    return logger


def get_inductor_logger(name: str) -> logging.Logger:
    """Backward-compatible alias for inductor logger creation."""
    return get_logger(name)


def update_log_level(name: str, level: str):
    """Update log level for a logger.

    Args:
        name: Logger name (without "spyre.inductor." prefix)
        level: New log level (DEBUG, INFO, etc.)
    """
    full_name = f"spyre.inductor.{name}"
    logging_config.set_log_level(full_name, level)

    if full_name in _loggers:
        _loggers[full_name].setLevel(int(logging_config.get_log_level(full_name)))


def _get_env_bool(var: str, default: bool) -> bool:
    """Backward-compatible helper to parse boolean environment variables."""
    return os.getenv(var, str(int(default))).lower() in ("1", "true", "yes")


# Module-level cache for inductor logging enabled state
_INDUCTOR_LOGGING_ENABLED: bool = _get_env_bool("SPYRE_INDUCTOR_LOG", False)


def is_inductor_logging_enabled() -> bool:
    """Check if inductor logging is enabled via environment variable.

    This is a backward-compatible function that checks the SPYRE_INDUCTOR_LOG
    environment variable. Returns True if logging is enabled, False otherwise.

    Returns:
        True if SPYRE_INDUCTOR_LOG is set to a truthy value, False otherwise
    """
    if _needs_reinit:
        _reinitialize()
    return _INDUCTOR_LOGGING_ENABLED


# Convenience loggers for common components
lowering_log = get_logger("lowering")
codegen_log = get_logger("codegen")
stickify_log = get_logger("stickify")
passes_log = get_logger("passes")
