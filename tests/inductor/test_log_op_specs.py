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


"""Tests for format_op_spec_list and per-pass op-spec logging.

Tests the formatting utility added to op_spec.py and the logger.info calls
added to spyre_kernel.py and codegen/bundle.py.  No Spyre device or backend
compiler is needed.
"""

import logging
from unittest.mock import MagicMock, patch

from sympy import Symbol, sympify

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    TensorArg,
    UnimplementedOp,
    format_op_spec_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

m = Symbol("m")
n = Symbol("n")
k = Symbol("k")


def _make_tensor_arg(is_input, arg_index, size, coords, allocation=None):
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.IEEE_FP16,
        device_size=size,
        device_coordinates=coords,
        allocation=allocation or {"hbm": 0},
    )


def _make_matmul_op():
    return OpSpec(
        op="matmul",
        is_reduction=True,
        iteration_space={
            m: (sympify(128), 1),
            n: (sympify(64), 1),
            k: (sympify(64), 4),
        },
        args=[
            _make_tensor_arg(True, 0, [128, 256], [m, k], {"hbm": 0}),
            _make_tensor_arg(True, 1, [256, 64], [k, n], {"hbm": 32768}),
            _make_tensor_arg(False, 2, [128, 64], [m, n], {"hbm": 65536}),
        ],
        op_info={},
        tiled_symbols=[[k]],
    )


def _make_add_op():
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={m: (sympify(128), 1), n: (sympify(64), 1)},
        args=[
            _make_tensor_arg(True, 0, [128, 64], [m, n]),
            _make_tensor_arg(True, 1, [128, 64], [m, n], {"hbm": 8192}),
            _make_tensor_arg(False, 2, [128, 64], [m, n], {"hbm": 16384}),
        ],
        op_info={},
    )


# ---------------------------------------------------------------------------
# Tests: format_op_spec_list
# ---------------------------------------------------------------------------


class TestFormatOpSpecList:
    """Unit tests for the format_op_spec_list formatter."""

    def test_empty_list(self):
        assert format_op_spec_list([]) == ""

    def test_single_op_spec(self):
        op = _make_matmul_op()
        result = format_op_spec_list([op])

        assert "OpSpec(op='matmul'" in result
        assert "is_reduction=True" in result
        assert "iteration_space={" in result
        assert "TensorArg(input" in result
        assert "TensorArg(output" in result
        assert "arg_index=0" in result
        assert "arg_index=2" in result
        assert "device_size=[128, 256]" in result
        assert "tiled_symbols=[[k]]" in result

    def test_op_spec_without_optional_fields(self):
        op = _make_add_op()
        result = format_op_spec_list([op])

        assert "OpSpec(op='add'" in result
        assert "is_reduction=False" in result
        assert "tiled_symbols" not in result
        assert "symbolic_dim_bounds" not in result

    def test_op_spec_with_symbolic_dim_bounds(self):
        op = OpSpec(
            op="conv2d",
            is_reduction=False,
            iteration_space={m: (sympify(128), 1)},
            args=[_make_tensor_arg(True, 0, [128, 64], [m, n])],
            op_info={},
            symbolic_dim_bounds={"s97": (256, 64)},
        )
        result = format_op_spec_list([op])

        assert "symbolic_dim_bounds=" in result
        assert "'s97': (256, 64)" in result

    def test_single_loop_spec(self):
        matmul = _make_matmul_op()
        loop = LoopSpec(count=sympify(4), body=[matmul])
        result = format_op_spec_list([loop])

        assert "LoopSpec(count=4)" in result
        assert "body=[" in result
        assert "OpSpec(op='matmul'" in result
        lines = result.split("\n")
        op_line = [line for line in lines if "OpSpec(op='matmul'" in line][0]
        assert op_line.startswith("    ")

    def test_nested_loop_specs(self):
        add_op = _make_add_op()
        inner = LoopSpec(count=sympify(2), body=[add_op])
        outer = LoopSpec(count=sympify(8), body=[inner])
        result = format_op_spec_list([outer])

        assert result.count("LoopSpec(count=") == 2
        assert result.count("body=[") == 2
        assert "OpSpec(op='add'" in result
        lines = result.split("\n")
        inner_op = [line for line in lines if "OpSpec(op='add'" in line][0]
        assert inner_op.startswith("        ")

    def test_unimplemented_op(self):
        uop = UnimplementedOp(op="custom_op")
        result = format_op_spec_list([uop])

        assert result == "UnimplementedOp(op='custom_op')"

    def test_mixed_list(self):
        add_op = _make_add_op()
        loop = LoopSpec(count=sympify(4), body=[_make_matmul_op()])
        uop = UnimplementedOp(op="scatter_nd")
        result = format_op_spec_list([add_op, loop, uop])

        assert "OpSpec(op='add'" in result
        assert "LoopSpec(count=4)" in result
        assert "OpSpec(op='matmul'" in result
        assert "UnimplementedOp(op='scatter_nd')" in result

    def test_unknown_item_type_uses_repr(self):
        result = format_op_spec_list(["unexpected_item"])
        assert "'unexpected_item'" in result

    def test_indent_parameter(self):
        op = _make_add_op()
        result = format_op_spec_list([op], indent=2)

        lines = result.split("\n")
        assert all(line.startswith("    ") for line in lines if line.strip())


# ---------------------------------------------------------------------------
# Tests: spyre_kernel.py logging integration
# ---------------------------------------------------------------------------


class TestSpyreKernelLogging:
    """Tests for logger.info calls in SpyreKernel.codegen_kernel."""

    def test_logs_at_both_stages(self):
        logger = logging.getLogger("spyre.inductor.spyre_kernel")
        with patch.object(logger, "isEnabledFor", return_value=True):
            with patch.object(logger, "info") as mock_info:
                with patch(
                    "torch_spyre._inductor.spyre_kernel.format_op_spec_list",
                    return_value="<formatted>",
                ) as mock_fmt:
                    from torch_spyre._inductor.spyre_kernel import SpyreKernel

                    kernel = SpyreKernel.__new__(SpyreKernel)
                    kernel.op_specs = [_make_add_op()]
                    kernel.indirect_vars = None
                    kernel.indirect_sizes = {}
                    kernel.args = MagicMock()
                    kernel.args.python_argdefs.return_value = (None, [])
                    kernel.spyre_kernel_args = []

                    with patch("torch_spyre._inductor.spyre_kernel.simplify_op_spec"):
                        kernel.codegen_kernel()

                    info_calls = mock_info.call_args_list
                    labels = [c.args[0] for c in info_calls if "OP SPECS" in c.args[0]]
                    assert any("CREATION/LOOP-WRAPPING" in lbl for lbl in labels)
                    assert any("SIMPLIFICATION" in lbl for lbl in labels)
                    assert mock_fmt.call_count >= 2

    def test_no_formatting_when_logging_disabled(self):
        logger = logging.getLogger("spyre.inductor.spyre_kernel")
        with patch.object(logger, "isEnabledFor", return_value=False):
            with patch(
                "torch_spyre._inductor.spyre_kernel.format_op_spec_list"
            ) as mock_fmt:
                from torch_spyre._inductor.spyre_kernel import SpyreKernel

                kernel = SpyreKernel.__new__(SpyreKernel)
                kernel.op_specs = [_make_add_op()]
                kernel.indirect_vars = None
                kernel.indirect_sizes = {}
                kernel.args = MagicMock()
                kernel.args.python_argdefs.return_value = (None, [])
                kernel.spyre_kernel_args = []

                with patch("torch_spyre._inductor.spyre_kernel.simplify_op_spec"):
                    kernel.codegen_kernel()

                mock_fmt.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: bundle.py logging integration
# ---------------------------------------------------------------------------


class TestBundleLogging:
    """Tests for logger.info call in generate_bundle."""

    def test_logs_bundle_generation_label(self):
        from torch_spyre._inductor.codegen.bundle import generate_bundle

        logger = logging.getLogger("spyre.inductor.sdsc_compile")
        with patch.object(logger, "isEnabledFor", return_value=True):
            with patch.object(logger, "info") as mock_info:
                with (
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
                    patch("builtins.open", MagicMock()),
                ):
                    op = _make_add_op()
                    generate_bundle(
                        kernel_name="test_kernel",
                        output_dir="/tmp/test",
                        specs=[op],
                        use_symbols=False,
                    )

                info_calls = mock_info.call_args_list
                op_spec_calls = [
                    c
                    for c in info_calls
                    if len(c.args) >= 1 and "OP SPECS" in c.args[0]
                ]
                assert len(op_spec_calls) >= 1
                assert "BUNDLE GENERATION" in op_spec_calls[0].args[0]
