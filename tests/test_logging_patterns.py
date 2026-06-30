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


"""Pytest/unittest-compatible logging tests for torch-spyre."""

import importlib.machinery
import importlib.util
import logging
import os
import sys
import tempfile
import types
import unittest
import warnings
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import TypedDict

import pytest

TEST_FILE = Path(__file__).resolve()


def _candidate_package_roots() -> Generator[Path, None, None]:
    """Yield likely package-root locations for the torch-spyre sources."""
    explicit_root = os.environ.get("TORCH_SPYRE_PACKAGE_ROOT")
    if explicit_root:
        yield Path(explicit_root).resolve()

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    for entry in env_pythonpath.split(os.pathsep):
        if entry:
            path_entry = Path(entry).resolve()
            yield path_entry
            yield path_entry / "torch_spyre"

    script_dir = TEST_FILE.parent
    yield script_dir
    yield script_dir / "torch_spyre"
    yield script_dir / "torch-spyre" / "torch_spyre"
    yield script_dir / "torch-spyre" / "torch-spyre" / "torch_spyre"

    seen = set()
    anchors = [
        script_dir,
        *script_dir.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]
    for anchor in anchors:
        if anchor in seen:
            continue
        seen.add(anchor)
        yield anchor
        yield anchor / "torch_spyre"
        yield anchor / "torch-spyre" / "torch_spyre"
        yield anchor / "torch-spyre" / "torch-spyre" / "torch_spyre"


def _is_package_root(candidate: Path) -> bool:
    has_logging_config = (candidate / "logging_config.py").is_file()
    has_init = (candidate / "__init__.py").is_file()

    return has_logging_config and has_init


def _find_package_root() -> Path | None:
    """Return the resolved torch-spyre package root when discoverable."""
    candidates: list[Path] = []
    for candidate in _candidate_package_roots():
        if candidate in candidates:
            continue
        candidates.append(candidate)
        if _is_package_root(candidate):
            return candidate

    for candidate in candidates:
        try:
            for child in candidate.iterdir():
                if child.is_dir() and _is_package_root(child):
                    return child
        except OSError:
            continue

    return None


PACKAGE_ROOT = _find_package_root()
if PACKAGE_ROOT is not None:
    BASE_DIR = PACKAGE_ROOT.parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))


class _LoggerState(TypedDict):
    level: int
    handlers: list[logging.Handler]
    propagate: bool
    disabled: bool


class LoggingIsolationMixin:
    """Shared helpers for isolating logging state across tests."""

    def setUp(self) -> None:  # pylint: disable=invalid-name
        """Save process environment, modules, and logger state before each test."""
        self._original_env = os.environ.copy()
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "torch_spyre.logging_config",
                "torch_spyre._inductor.logging_utils",
            )
        }
        self._saved_loggers = {
            name: logging.getLogger(name)
            for name in (
                "spyre",
                "spyre.inductor",
                "spyre.inductor.lowering",
                "spyre.inductor.codegen",
                "spyre.inductor.stickify",
                "spyre.inductor.passes",
                "spyre.inductor.sdsc_compile",
                "spyre.inductor.work_division",
                "spyre.inductor.propagate_layouts",
                "spyre.inductor.test_component",
                "spyre.inductor.legacy_test",
                "spyre.runtime",
            )
        }
        self._saved_logger_state: dict[str, _LoggerState] = {}
        for name, logger in self._saved_loggers.items():
            self._saved_logger_state[name] = {
                "level": logger.level,
                "handlers": list(logger.handlers),
                "propagate": logger.propagate,
                "disabled": logger.disabled,
            }

    def tearDown(self) -> None:  # pylint: disable=invalid-name
        """Restore environment, modules, and loggers after each test."""
        os.environ.clear()
        os.environ.update(self._original_env)

        for module_name in (
            "torch_spyre.logging_config",
            "torch_spyre._inductor.logging_utils",
        ):
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Also clean up submodule attributes on the torch_spyre package,
        # which persist even after sys.modules entries are deleted.
        ts_mod = sys.modules.get("torch_spyre")
        if ts_mod:
            for attr in ("logging_config", "_inductor"):
                if hasattr(ts_mod, attr):
                    try:
                        delattr(ts_mod, attr)
                    except AttributeError:
                        pass

        for name, module in self._saved_modules.items():
            if module is not None:
                sys.modules[name] = module

        for name, logger in self._saved_loggers.items():
            state = self._saved_logger_state[name]
            logger.handlers = state["handlers"]
            logger.setLevel(state["level"])
            logger.propagate = state["propagate"]
            logger.disabled = state["disabled"]

    def _load_module(self, module_name: str, relative_path: str) -> ModuleType:
        """Load a module directly from a file beneath the package root."""
        package_root = PACKAGE_ROOT
        assert package_root is not None
        full_path = package_root / relative_path
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _ensure_package_module(package_name: str, package_path: Path) -> ModuleType:
        """Create a lightweight package module for direct submodule loading."""
        package = sys.modules.get(package_name)
        if isinstance(package, ModuleType):
            return package

        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            name=package_name,
            loader=None,
            is_package=True,
        )
        sys.modules[package_name] = package
        return package

    def _reload_logging_modules(
        self,
    ) -> tuple[ModuleType, ModuleType]:
        """Reload the logging modules under the current environment settings."""
        if PACKAGE_ROOT is None:
            pytest.skip(
                "Could not locate torch_spyre package root for logging tests. "
                "Set TORCH_SPYRE_PACKAGE_ROOT to the directory containing "
                "logging_config.py if this checkout stores sources elsewhere."
            )

        file_path = Path(str(PACKAGE_ROOT) + "/logging_config.py")
        if not file_path.is_file():
            pytest.skip(f"Invalid torch_spyre package root: {PACKAGE_ROOT}")

        # Load logging_config first — it does NOT import torch, so TORCH_LOGS
        # can safely remain in the environment for _parse_torch_logs() to read
        # during initialize() (called at module load time).
        logging_config = self._load_module("logging_config", "logging_config.py")

        # NOW save and clear TORCH_LOGS before loading modules that import
        # torch, since PyTorch's logging system rejects unregistered spyre.*
        # namespaces.
        saved_torch_logs = os.environ.get("TORCH_LOGS")
        if saved_torch_logs and "spyre" in saved_torch_logs:
            os.environ.pop("TORCH_LOGS", None)

        inductor_package_name = "_inductor"
        inductor_package = sys.modules.get(inductor_package_name)
        if inductor_package is None:
            assert PACKAGE_ROOT is not None
            inductor_package = self._ensure_package_module(
                inductor_package_name,
                PACKAGE_ROOT / "_inductor",
            )

        # Clean up stale torch_spyre.logging_config references that persist
        # as attributes on the torch_spyre package from prior test runs.
        if "torch_spyre.logging_config" in sys.modules:
            del sys.modules["torch_spyre.logging_config"]
        ts_mod = sys.modules.get("torch_spyre")
        if ts_mod and hasattr(ts_mod, "logging_config"):
            delattr(ts_mod, "logging_config")

        # logging_utils imports torch_spyre which triggers torch import
        logging_utils = self._load_module(
            "_inductor.logging_utils",
            "_inductor/logging_utils.py",
        )

        # Restore TORCH_LOGS after torch has been imported
        if saved_torch_logs:
            os.environ["TORCH_LOGS"] = saved_torch_logs

        return logging_config, logging_utils


class UnifiedLoggingPatternTests(LoggingIsolationMixin, unittest.TestCase):
    """Tests for the unified logging configuration flow."""

    # Need to figure out why this one fails.
    def test_unified_torch_logs_controls_new_patterns(self) -> None:
        """Verify TORCH_LOGS enables the new unified warning patterns."""
        os.environ["TORCH_LOGS"] = "spyre.inductor:DEBUG"
        logging_config, logging_utils = self._reload_logging_modules()

        compile_logger = logging_utils.get_logger("sdsc_compile")
        wd_logger = logging_utils.get_logger("work_division")

        self.assertEqual(
            compile_logger.name,
            "spyre.inductor.sdsc_compile",
        )
        self.assertEqual(
            compile_logger.level,
            int(logging_config.LogLevel.DEBUG),
        )
        self.assertEqual(
            wd_logger.level,
            int(logging_config.LogLevel.DEBUG),
        )
        self.assertEqual(
            logging_config.get_config_source("spyre.inductor"),
            "TORCH_LOGS",
        )

        with self.assertLogs("spyre", level="DEBUG") as captured:
            compile_logger.warning(
                "WARNING: Compiling unimplemented aten.custom_op to runtime exception"
            )
            wd_logger.warning(
                "No valid split combo found for tensor buf0 coord=x "
                "under accumulated_splits={'x': 4, 'y': 1}. Skipping."
            )
            wd_logger.critical(
                "Cannot satisfy minimum split requirement for x: need 8 splits "
                "but only 4 cores remaining. Skipping this constraint - "
                "hardware span limit may be violated."
            )

        output = "\n".join(captured.output)
        self.assertIn(
            "WARNING: Compiling unimplemented aten.custom_op to runtime exception",
            output,
        )
        self.assertIn(
            "No valid split combo found for tensor buf0 coord=x",
            output,
        )
        self.assertIn(
            "Cannot satisfy minimum split requirement for x",
            output,
        )

    def test_programmatic_override_enables_component_specific_messages(self) -> None:
        """Verify programmatic overrides affect a specific component logger."""
        logging_config, logging_utils = self._reload_logging_modules()

        test_logger = logging_utils.get_logger("test_component")
        self.assertEqual(test_logger.level, int(logging_config.LogLevel.DEBUG))

        logging_config.set_log_level("spyre.inductor.test_component", "DEBUG")
        refreshed_logger = logging_utils.get_logger("test_component")

        self.assertIs(test_logger, refreshed_logger)
        self.assertEqual(
            refreshed_logger.level,
            int(logging_config.LogLevel.DEBUG),
        )
        self.assertEqual(
            logging_config.get_config_source("spyre.inductor.test_component"),
            "programmatic",
        )

        with self.assertLogs(
            "spyre.inductor.test_component", level="DEBUG"
        ) as captured:
            refreshed_logger.debug("This DEBUG message should now be visible")
            refreshed_logger.info("This INFO message should be visible")
            refreshed_logger.warning("This WARNING message should be visible")

        self.assertEqual(len(captured.output), 3)


class LegacyCompatibilityTests(LoggingIsolationMixin, unittest.TestCase):
    """Tests for the legacy-to-unified logging compatibility layer."""

    def test_legacy_environment_variables_map_to_unified_config(self) -> None:
        """Verify legacy env vars map into unified config with warnings."""
        os.environ.pop("TORCH_LOGS", None)
        os.environ["SPYRE_INDUCTOR_LOG"] = "1"
        os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "DEBUG"
        os.environ["TORCH_SPYRE_DEBUG"] = "1"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logging_config, logging_utils = self._reload_logging_modules()

        messages = [str(w.message) for w in caught]
        self.assertEqual(
            logging_config.get_effective_config()["spyre.inductor"],
            "DEBUG",
        )
        self.assertEqual(
            logging_config.get_config_source("spyre.inductor"),
            "legacy:SPYRE_INDUCTOR_LOG",
        )
        self.assertEqual(
            logging_config.get_effective_config()["spyre.runtime"],
            "DEBUG",
        )
        self.assertEqual(
            logging_config.get_config_source("spyre.runtime"),
            "legacy:TORCH_SPYRE_DEBUG",
        )
        self.assertTrue(
            any("SPYRE_INDUCTOR_LOG is deprecated" in message for message in messages)
        )
        self.assertTrue(
            any("TORCH_SPYRE_DEBUG is deprecated" in message for message in messages)
        )

        legacy_logger = logging_utils.get_logger("legacy_test")
        with self.assertLogs("spyre.inductor.legacy_test", level="DEBUG") as captured:
            legacy_logger.debug(
                "DEBUG message (enabled via SPYRE_INDUCTOR_LOG_LEVEL=DEBUG)"
            )
            legacy_logger.info("INFO message")
            legacy_logger.warning("WARNING message")

        self.assertEqual(len(captured.output), 3)

    def test_legacy_log_file_env_var_maps_to_unified_config(self) -> None:
        """Verify SPYRE_LOG_FILE maps into unified output config with warning."""
        os.environ["SPYRE_LOG_FILE"] = "/tmp/spyre-legacy.log"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logging_config, _ = self._reload_logging_modules()

        output_config = logging_config.get_output_config()
        self.assertEqual(output_config["log_file"], "/tmp/spyre-legacy.log")
        self.assertEqual(output_config["log_file_source"], "legacy:SPYRE_LOG_FILE")
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("SPYRE_LOG_FILE is deprecated" in message for message in messages)
        )


class CompleteIntegrationTests(LoggingIsolationMixin, unittest.TestCase):
    """End-to-end tests for integrated logging configuration behavior."""

    def test_integration_flow_covers_factory_loggers_components_and_output_config(
        self,
    ) -> None:
        """Verify integration flow, convenience loggers, and file output."""
        os.environ["TORCH_LOGS"] = "spyre.inductor:DEBUG"

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "spyre.log")
            logging_config, logging_utils = self._reload_logging_modules()

            self.assertNotIn(
                "legacy:", logging_config.get_config_source("spyre.inductor")
            )
            components = logging_config.list_components()
            self.assertIn("spyre.inductor", components)
            self.assertIn("spyre.execution", components)

            self.assertEqual(logging_utils.lowering_log.name, "spyre.inductor.lowering")
            self.assertEqual(logging_utils.codegen_log.name, "spyre.inductor.codegen")
            self.assertEqual(logging_utils.stickify_log.name, "spyre.inductor.stickify")
            self.assertEqual(logging_utils.passes_log.name, "spyre.inductor.passes")

            logging_config.set_log_file(log_path)
            output_config = logging_config.get_output_config()
            self.assertEqual(output_config["log_file"], log_path)
            self.assertEqual(output_config["log_file_source"], "programmatic")

            compile_logger = logging_utils.get_logger("sdsc_compile")
            wd_logger = logging_utils.get_logger("work_division")
            layout_logger = logging_utils.get_logger("propagate_layouts")

            compile_logger.warning(
                "WARNING: Compiling unimplemented aten.test_op to runtime exception"
            )
            wd_logger.warning(
                "No valid split combo found for tensor buf0 coord=x "
                "under accumulated_splits={'x': 4}. Skipping."
            )
            layout_logger.warning("Warning: unhandled node type <class 'TestNode'>")

            spyre_logger = logging.getLogger("spyre")
            for handler in spyre_logger.handlers:
                flush = getattr(handler, "flush", None)
                if flush is not None:
                    flush()

            with open(log_path, encoding="utf-8") as handle:
                contents = handle.read()

        self.assertIn(
            "[WARNING] [spyre.inductor.sdsc_compile] WARNING: "
            "Compiling unimplemented aten.test_op to runtime exception",
            contents,
        )
        self.assertIn(
            "[WARNING] [spyre.inductor.work_division] No valid split combo "
            "found for tensor buf0 coord=x under accumulated_splits"
            "={'x': 4}. Skipping.",
            contents,
        )
        self.assertIn(
            "[WARNING] [spyre.inductor.propagate_layouts] Warning: "
            "unhandled node type <class 'TestNode'>",
            contents,
        )


if __name__ == "__main__":
    unittest.main()
