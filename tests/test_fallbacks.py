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


import torch
from torch.testing import assert_close
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
)
import ast
import os
import subprocess
import sys
import warnings
from pathlib import Path

import torch_spyre
from torch_spyre.ops.eager import RetileWarning
from torch_spyre.ops.fallbacks import FallbackWarning


class Op(subtest):
    def __init__(self, name, fn, rtol=None, atol=None):
        super().__init__(self, name)
        self.fn = fn
        self.rtol = rtol
        self.atol = atol


class UnaryOp(Op):
    inputs = [torch.rand(64, dtype=torch.float16)]

    def __init__(self, name, fn, inputs=None, rtol=None, atol=None):
        super().__init__(name, fn, rtol, atol)
        self.inputs = inputs or type(self).inputs


class BinaryOp(Op):
    inputs = [
        (torch.rand(64, dtype=torch.float16), torch.rand(64, dtype=torch.float16))
    ]

    def __init__(self, name, fn, inputs=None, rtol=None, atol=None):
        super().__init__(name, fn, rtol, atol)
        self.inputs = inputs or type(self).inputs


# Each stands in for one fallback shape and must stay on the host-fallback list
# (torch_spyre/ops/fallbacks.py): an op that gains a Spyre kernel or
# decomposition no longer exercises the fallback, and a replacement is needed.
# The condition-only where returns a tuple of index tensors and has no out=
# overload, so out= is exercised through bitwise_xor.
_unary_ops = [
    UnaryOp(
        "where",
        torch.where,
        [(torch.arange(64 * 64).reshape(64, 64) % 3 == 0).to(torch.float16)],
    ),
]

_binary_out_ops = [
    BinaryOp(
        "bitwise_xor",
        torch.bitwise_xor,
        [
            (
                torch.randint(0, 256, (64,), dtype=torch.int32),
                torch.randint(0, 256, (64,), dtype=torch.int32),
            )
        ],
    ),
]


@instantiate_parametrized_tests
class TestFallbacks(TestCase):
    def setUp(self):
        self.rtol = 1e-2
        self.atol = 1e-3
        self.dtype = torch.float16

        torch.manual_seed(0xAFFE)

        warnings.simplefilter("ignore", FallbackWarning)

    def _assert_close(self, op, output_spyre, output_cpu):
        rtol = op.rtol or self.rtol
        atol = op.atol or self.atol
        if isinstance(output_cpu, tuple):
            output_spyre = tuple(t.cpu() for t in output_spyre)
        else:
            output_spyre = output_spyre.cpu()
        assert_close(output_spyre, output_cpu, rtol=rtol, atol=atol)

    @parametrize("op", _unary_ops)
    def test_unary_op(self, op):
        for input in op.inputs:
            output_cpu = op.fn(input)
            output_spyre = op.fn(input.to("spyre"))

            self._assert_close(op, output_spyre, output_cpu)

    @parametrize("op", _binary_out_ops)
    def test_binary_op_out(self, op):
        for x, y in op.inputs:
            buffer_cpu = torch.empty_like(x)
            output_cpu = op.fn(x, y, out=buffer_cpu)

            buffer_spyre = torch.empty_like(x, device="spyre")
            output_spyre = op.fn(x.to("spyre"), y.to("spyre"), out=buffer_spyre)

            self._assert_close(op, output_spyre, output_cpu)

    @parametrize("op", _binary_out_ops)
    def test_binary_op_out_alias(self, op):
        for x, y in op.inputs:
            buffer_cpu = torch.clone(x)
            output_cpu = op.fn(buffer_cpu, y, out=buffer_cpu)

            buffer_spyre = torch.clone(x).to(device="spyre")
            output_spyre = op.fn(buffer_spyre, y.to("spyre"), out=buffer_spyre)

            self._assert_close(op, output_spyre, output_cpu)
            self.assertEqual(id(buffer_spyre), id(output_spyre))


_WARNING_TYPES = (FallbackWarning, RetileWarning)
_IMPORT_TIMEOUT_S = 130  # Hang guard for the subprocess's torch import.


def _base_name(base):
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


class TestWarningTypes(TestCase):
    def test_importable_without_backend_autoload(self):
        # xdist's controller rebuilds each worker warning by importing its class's module.
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
        for module in sorted({c.__module__ for c in _WARNING_TYPES}):
            names = [c.__name__ for c in _WARNING_TYPES if c.__module__ == module]
            script = (
                "import importlib; "
                f"mod = importlib.import_module({module!r}); "
                f"[getattr(mod, name) for name in {names!r}]"
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                env=env,
                timeout=_IMPORT_TIMEOUT_S,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"{module}:\n{proc.stderr}")

    def test_warning_classes_live_in_warning_types(self):
        # A warning class in a module with import side effects crashes xdist's controller.
        package = Path(torch_spyre.__file__).parent
        home = package / "ops" / "warning_types.py"
        misplaced = []
        for path in sorted(package.rglob("*.py")):
            if path == home:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            misplaced += [
                f"{path.relative_to(package.parent)}:{node.lineno} {node.name}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and any(_base_name(b).endswith("Warning") for b in node.bases)
            ]
        self.assertEqual(
            misplaced,
            [],
            f"Define warning classes in {home.relative_to(package.parent)}",
        )


if __name__ == "__main__":
    run_tests()
