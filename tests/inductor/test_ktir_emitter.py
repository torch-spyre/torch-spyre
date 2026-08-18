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
#
# ``bundle_symbolic_args`` is pinned True for the same reason
# ``TestKtirBakedAddresses`` pins it False: _EXPECTED_ADD_KTIR is the symbolic
# form, so leaving it to ambient BUNDLE_SYMBOLIC_ARGS makes the golden fail under
# BUNDLE_SYMBOLIC_ARGS=0 -- which is exactly how the device path is run.
@mock.patch("torch_spyre._inductor.config.bundle_symbolic_args", True)
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


def _mlir_ktdp_linalg_available() -> bool:
    # The baked form additionally needs the linalg/tensor bindings.
    try:
        from mlir_ktdp.dialects import linalg, tensor  # noqa: F401
    except ImportError:
        return False
    return _mlir_ktdp_available()


@mock.patch("torch_spyre._inductor.config.bundle_symbolic_args", False)
@mock.patch("torch_spyre._inductor.config.sencores", 1)
class TestKtirBakedAddresses(unittest.TestCase):
    @staticmethod
    def _arg(allocation):
        arg = _add_op_specs()[0].args[1]
        arg.allocation = allocation
        return arg

    @unittest.skipUnless(_mlir_ktdp_linalg_available(), "no mlir_ktdp linalg")
    def test_baked_form_deltas(self):
        """The baked form (dataflow-scheduler#65) vs ``_EXPECTED_ADD_KTIR``.

        Asserted as deltas rather than a second golden: the two texts differ in
        5 of 24 lines, so a full copy would be 19 lines of duplication that churn
        together, and this form is deleted outright when #65 is fixed.  The
        loads / tiles / views the two share are already pinned by
        ``_EXPECTED_ADD_KTIR``.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        specs = _add_op_specs()
        for arg in specs[0].args:
            arg.allocation = {"hbm": arg.arg_index << 34}
        emitted = generate_ktir("ktir_fused_add_0", specs)

        # 1. No address is a runtime value: zero-arg func, no %arg anywhere.
        self.assertIn("func.func @ktir_fused_add_0() attributes {grid = [1]}", emitted)
        self.assertNotIn("%arg", emitted)
        # 2. Each base is a constant, in ELEMENTS (the byte slot >> 1 for fp16).
        for arg_index in range(3):
            with self.subTest(arg_index=arg_index):
                base = (arg_index << 34) // 2
                self.assertIn(f"arith.constant {base} : index", emitted)
        # 3. linalg over tensor.empty, never arith on tensors -- required for the
        #    memref offset to fold to static, which ktdp.load's verifier needs.
        self.assertIn("tensor.empty()", emitted)
        self.assertIn("linalg.add ins(", emitted)
        self.assertNotIn("arith.addf", emitted)

    def test_addresses_resolved_without_the_dialect(self):
        from torch_spyre._inductor.codegen.ktir import _base_address_elements

        # fp16: 2 bytes per element.  Zero is a real address, not "unset".
        self.assertEqual(_base_address_elements(self._arg({"hbm": 1 << 34})), 1 << 33)
        self.assertEqual(_base_address_elements(self._arg({"hbm": 0})), 0)
        # Unassigned, and outside HBM (every memory view hardcodes HBM):
        for allocation in ({"hbm": None}, {"lx": 0x1000}, {"hbm_pool": 0x1000}, {}):
            with self.subTest(alloc=allocation), self.assertRaises(NotImplementedError):
                _base_address_elements(self._arg(allocation))


if __name__ == "__main__":
    unittest.main()
