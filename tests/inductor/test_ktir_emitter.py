# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Golden-text snapshot test for the OpSpec->KTIR emitter (``generate_ktir``).

Self-contained: it builds the ``add`` OpSpec directly (no live Inductor graph,
no compiler run) and asserts the emitter's canonical MLIR text.
Skipped only where ``mlir_ktdp`` is not installed.
"""

import unittest
from unittest import mock

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _mlir_ktdp_available() -> bool:
    """True when mlir_ktdp is built with the func/arith dialect Python bindings."""
    try:
        from mlir_ktdp import ir  # noqa: F401
        from mlir_ktdp.dialects import arith, func, ktdp  # noqa: F401
    except ImportError:
        return False
    return True


# The canonical KTIR text ``generate_ktir`` emits for a single pointwise ``add``
# over a [512, 1024] fp16 tensor stickified to device shape [16, 512, 64].
_EXPECTED_ADD_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_0(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %4 = ktdp.load %3 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %5 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %6 = ktdp.load %5 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %7 = arith.addf %4, %6 : tensor<16x512x64xf16>
    %8 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %7, %8 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""


def _add_op_specs() -> list:
    """Finished OpSpec list for ``a + b`` at device shape [16, 512, 64] fp16.

    This mirrors what the SuperDSC frontend produces for a pointwise ``a + b``:
    two HBM inputs and one HBM output, each addressed at the identity
    coordinates ``(d0, d1, d2)`` over the stickified device shape.
    """
    d0, d1, d2 = sympy.symbols("d0 d1 d2")
    coords = [d0, d1, d2]
    size = [16, 512, 64]

    def arg(is_input: bool, index: int, name: str) -> TensorArg:
        return TensorArg(
            is_input=is_input,
            arg_index=index,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(size),
            device_coordinates=list(coords),
            allocation={"hbm": None},
            name=name,
        )

    return [
        OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={d0: (16, 1), d1: (512, 1), d2: (64, 1)},
            args=[
                arg(True, 0, "arg0"),
                arg(True, 1, "arg1"),
                arg(False, 2, "buf0"),
            ],
            op_info={},
        )
    ]


# The emitter only supports the single-core (SENCORES=1) grid so far; pin it so
# these tests exercise their intended guards rather than the multi-core guard,
# which would otherwise fire first on the default SENCORES=32.
@mock.patch("torch_spyre._inductor.config.sencores", 1)
@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with func/arith dialect bindings is not installed",
)
class TestKtirEmitter(unittest.TestCase):
    def test_pointwise_add_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", _add_op_specs())
        self.assertEqual(emitted, _EXPECTED_ADD_KTIR)

    def test_reduction_unsupported(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        specs[0].is_reduction = True
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_add_0", specs)

    def test_non_add_unsupported(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        specs[0].op = "mul"
        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_mul_0", specs)


class TestKtirCapabilityGuards(unittest.TestCase):
    """Guards that fire before the mlir_ktdp import, so they need no dialect."""

    @mock.patch("torch_spyre._inductor.config.sencores", 2)
    def test_multicore_unsupported(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        with self.assertRaises(NotImplementedError):
            generate_ktir("ktir_fused_add_0", _add_op_specs())


if __name__ == "__main__":
    unittest.main()
