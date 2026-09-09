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


"""Tests for SDSC IR artifact logging via spyre.inductor.sdsc logger.

Verifies that the spyre.inductor.sdsc logger is correctly registered and
that SDSC JSON and bundle.mlir content is emitted when logging is enabled
via TORCH_LOGS="+spyre.inductor.sdsc".
"""

import logging
import tempfile
from unittest.mock import MagicMock, patch

import torch  # noqa: F401

from torch_spyre._inductor.logging_utils import get_inductor_logger


class TestSdscLoggerConfiguration:
    """Tests for the spyre.inductor.sdsc logger registration."""

    def test_logger_exists_with_correct_name(self):
        sdsc_log = get_inductor_logger("sdsc")
        assert sdsc_log.name == "spyre.inductor.sdsc"

    def test_registered_in_default_log_levels(self):
        from torch_spyre.logging_config import DEFAULT_LOG_LEVELS, LogLevel

        assert "spyre.inductor.sdsc" in DEFAULT_LOG_LEVELS
        assert DEFAULT_LOG_LEVELS["spyre.inductor.sdsc"] == LogLevel.WARNING

    def test_logger_created_dynamically(self):
        from torch_spyre._inductor.codegen import bundle

        assert bundle.sdsc_log.name == "spyre.inductor.sdsc"


class TestSdscJsonLogging:
    """Tests for SDSC JSON content logging in _compile_specs."""

    def test_sdsc_json_logged_when_enabled(self):
        """SDSC JSON content is logged at INFO when logger is enabled."""
        from torch_spyre._inductor.codegen import bundle

        sdsc_log = logging.getLogger("spyre.inductor.sdsc")
        with patch.object(sdsc_log, "isEnabledFor", return_value=True):
            with patch.object(sdsc_log, "info") as mock_info:
                fake_json = {"0_add": {"dscs_": []}}
                with patch(
                    "torch_spyre._inductor.codegen.bundle.compile_op_spec",
                    return_value=(fake_json, [], [], []),
                ):
                    specs = [MagicMock()]
                    specs[0].__class__ = bundle.OpSpec
                    with tempfile.TemporaryDirectory() as tmpdir:
                        bundle._compile_specs(
                            specs,
                            symbols=[],
                            compiled=[],
                            sdsc_counter=[0],
                            symbol_id_offset_counter=[0],
                            output_dir=tmpdir,
                        )

                info_calls = mock_info.call_args_list
                sdsc_json_calls = [
                    c
                    for c in info_calls
                    if len(c.args) >= 1 and "SDSC JSON" in c.args[0]
                ]
                assert len(sdsc_json_calls) == 1
                assert "sdsc_0.json" in sdsc_json_calls[0].args[1]

    def test_no_json_formatting_when_disabled(self):
        """json.dumps is not called when the sdsc logger is disabled."""
        from torch_spyre._inductor.codegen import bundle

        sdsc_log = logging.getLogger("spyre.inductor.sdsc")
        with patch.object(sdsc_log, "isEnabledFor", return_value=False):
            with patch.object(sdsc_log, "info") as mock_info:
                fake_json = {"0_add": {"dscs_": []}}
                with patch(
                    "torch_spyre._inductor.codegen.bundle.compile_op_spec",
                    return_value=(fake_json, [], [], []),
                ):
                    specs = [MagicMock()]
                    specs[0].__class__ = bundle.OpSpec
                    with tempfile.TemporaryDirectory() as tmpdir:
                        bundle._compile_specs(
                            specs,
                            symbols=[],
                            compiled=[],
                            sdsc_counter=[0],
                            symbol_id_offset_counter=[0],
                            output_dir=tmpdir,
                        )

                sdsc_json_calls = [
                    c
                    for c in mock_info.call_args_list
                    if len(c.args) >= 1 and "SDSC JSON" in c.args[0]
                ]
                assert len(sdsc_json_calls) == 0


class TestBundleMlirLogging:
    """Tests for bundle.mlir content logging in generate_bundle."""

    def test_bundle_mlir_logged_when_enabled(self):
        """bundle.mlir content is logged at INFO when logger is enabled."""
        from torch_spyre._inductor.codegen import bundle

        sdsc_log = logging.getLogger("spyre.inductor.sdsc")
        with patch.object(sdsc_log, "isEnabledFor", return_value=True):
            with patch.object(sdsc_log, "info") as mock_info:
                with (
                    patch.object(bundle.logger, "isEnabledFor", return_value=False),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._compile_specs",
                    ),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._collect_loop_bounds",
                    ),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._collect_affine_maps",
                    ),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._emit_specs",
                    ),
                ):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        bundle.generate_bundle(
                            kernel_name="test_kernel",
                            output_dir=tmpdir,
                            specs=[],
                        )

                mlir_calls = [
                    c
                    for c in mock_info.call_args_list
                    if len(c.args) >= 1 and "BUNDLE MLIR" in c.args[0]
                ]
                assert len(mlir_calls) == 1
                assert "func.func @sdsc_bundle()" in mlir_calls[0].args[1]

    def test_bundle_mlir_not_logged_when_disabled(self):
        """bundle.mlir is not read back when the sdsc logger is disabled."""
        from torch_spyre._inductor.codegen import bundle

        sdsc_log = logging.getLogger("spyre.inductor.sdsc")
        with patch.object(sdsc_log, "isEnabledFor", return_value=False):
            with patch.object(sdsc_log, "info") as mock_info:
                with (
                    patch.object(bundle.logger, "isEnabledFor", return_value=False),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._compile_specs",
                    ),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._collect_loop_bounds",
                    ),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._collect_affine_maps",
                    ),
                    patch(
                        "torch_spyre._inductor.codegen.bundle._emit_specs",
                    ),
                ):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        bundle.generate_bundle(
                            kernel_name="test_kernel",
                            output_dir=tmpdir,
                            specs=[],
                        )

                mlir_calls = [
                    c
                    for c in mock_info.call_args_list
                    if len(c.args) >= 1 and "BUNDLE MLIR" in c.args[0]
                ]
                assert len(mlir_calls) == 0
