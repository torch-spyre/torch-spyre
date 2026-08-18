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

"""Golden-text snapshot tests for the OpSpec->KTIR emitter (``generate_ktir``).

**Everything in this file needs the ``mlir_ktdp`` dialect build and is skipped
without it.**  The emitter's *rejections* need no dialect -- the plan walk
raises them all before the lazy import -- so they live in
``test_ktir_validate.py``, which is never skipped and which owns the shared spec
builders.

Self-contained otherwise: no live Inductor graph, no compiler run.
"""

import unittest

from test_ktir_validate import make_chained_op_specs, make_nested_op_spec, make_op_spec


def _mlir_ktdp_available() -> bool:
    """Whether this build can emit, asked of the emitter rather than guessed.

    The import list belongs to ``KtirBuilder.create``; duplicating it here is how
    the two drift, and a build missing one binding would then error instead of
    skipping.  ``ktir`` imports without a dialect build, so this is safe at
    module scope.
    """
    from torch_spyre._inductor.codegen.ktir import dialect_available

    return dialect_available()


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestKtirEmitter(unittest.TestCase):
    """The flat, untiled form: one pointwise add over a whole [16, 512, 64]
    device tile, which is what the frontend produces today."""

    # The canonical KTIR text ``generate_ktir`` emits for a single pointwise
    # ``add`` over a [512, 1024] fp16 tensor stickified to device shape
    # [16, 512, 64].
    EXPECTED_ADD_KTIR = """\
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
    %7 = tensor.empty() : tensor<16x512x64xf16>
    %8 = linalg.add ins(%4, %6 : tensor<16x512x64xf16>, tensor<16x512x64xf16>) outs(%7 : tensor<16x512x64xf16>) -> tensor<16x512x64xf16>
    %9 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %8, %9 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""

    def test_pointwise_add_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", [make_op_spec()])
        self.assertEqual(emitted, self.EXPECTED_ADD_KTIR)

    def test_registered_ops_reach_their_own_binding(self):
        """A second op costs one recipe: same shape, different linalg builder.

        Asserted as a delta against the golden rather than a second copy of it --
        only the compute line differs.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_mul_0", [make_op_spec("mul")])
        self.assertIn("linalg.mul ins(", emitted)
        self.assertNotIn("linalg.add", emitted)
        # Everything either side of the compute op is unchanged by the op name.
        self.assertEqual(
            emitted.replace("linalg.mul", "linalg.add").replace(
                "@ktir_fused_mul_0", "@ktir_fused_add_0"
            ),
            self.EXPECTED_ADD_KTIR,
        )


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestKtirBakedAddresses(unittest.TestCase):
    def test_baked_form_deltas(self):
        """The baked form (#65) vs ``TestKtirEmitter.EXPECTED_ADD_KTIR``.

        Asserted as deltas rather than a second golden: the two texts differ
        only in how base addresses are spelled, so a full copy would duplicate
        every line that churns together.  Reverting #65 deletes the baked arm of
        the two address helpers; the compute form does not move.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir(
            "ktir_fused_add_0", [make_op_spec(baked=True)], bake_addresses=True
        )

        # 1. No address is a runtime value: zero-arg func, no %arg anywhere.
        self.assertIn("func.func @ktir_fused_add_0() attributes {grid = [1]}", emitted)
        self.assertNotIn("%arg", emitted)
        # 2. Each base is a constant, in ELEMENTS (the byte slot >> 1 for fp16).
        for arg_index in range(3):
            with self.subTest(arg_index=arg_index):
                base = (arg_index << 34) // 2
                self.assertIn(f"arith.constant {base} : index", emitted)
        # Compute is deliberately NOT asserted here: both forms emit the same
        # linalg.add over a tensor.empty, so it is pinned by the symbolic golden
        # and is not a delta.  The two texts now differ only in addressing.


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestInternalBufferIsThreaded(unittest.TestCase):
    """``(a + b) * c`` in one kernel, with the intermediate threaded.

    ``buf0`` is allocated in LX, which is the contract saying the kernel owns it,
    so it gets no func parameter, no memory view, no store and no load: the
    ``linalg.mul`` consumes the ``linalg.add``'s result directly.

    A live kernel does not reach this yet -- the frontend puts the two ops in two
    kernels, with buf0's LX allocation crossing the boundary between them, and the
    emitter refuses that (``verify.py``'s ``chain`` case is the standing check).
    What this pins is the emission, so that fusing the two ops upstream is the
    only thing that has to change.
    """

    EXPECTED_CHAIN_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_mul_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.construct_memory_view %arg3, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %4 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %5 = ktdp.load %4 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %6 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %7 = ktdp.load %6 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %8 = tensor.empty() : tensor<16x512x64xf16>
    %9 = linalg.add ins(%5, %7 : tensor<16x512x64xf16>, tensor<16x512x64xf16>) outs(%8 : tensor<16x512x64xf16>) -> tensor<16x512x64xf16>
    %10 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    %11 = ktdp.load %10 : <16x512x64xindex> -> tensor<16x512x64xf16>
    %12 = tensor.empty() : tensor<16x512x64xf16>
    %13 = linalg.mul ins(%9, %11 : tensor<16x512x64xf16>, tensor<16x512x64xf16>) outs(%12 : tensor<16x512x64xf16>) -> tensor<16x512x64xf16>
    %14 = ktdp.construct_access_tile %3[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<16x512x64xf16> -> !ktdp.access_tile<16x512x64xindex>
    ktdp.store %13, %14 : tensor<16x512x64xf16>, <16x512x64xindex>
    return
  }
}
"""

    @staticmethod
    def _chain():
        """``(a + b) * c`` in one kernel: the add's result is the mul's operand."""
        return make_chained_op_specs(("add", "mul"))

    def test_chain_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_mul_0", self._chain())
        self.assertEqual(emitted, self.EXPECTED_CHAIN_KTIR)

    def test_the_intermediate_leaves_no_trace_in_memory(self):
        """The golden's point, asserted as counts so it cannot be read past.

        Three loads and one store for four buffers: buf0 is a value, and the
        second op's first operand is the first op's result.
        """
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_mul_0", self._chain())
        self.assertEqual(emitted.count("ktdp.load"), 3)  # a, b, c -- not buf0
        self.assertEqual(emitted.count("ktdp.store"), 1)  # buf1 only
        self.assertEqual(emitted.count("ktdp.construct_memory_view"), 4)
        [add] = [ln for ln in emitted.splitlines() if "linalg.add ins(" in ln]
        [mul] = [ln for ln in emitted.splitlines() if "linalg.mul ins(" in ln]
        self.assertIn(f"ins({add.split('=')[0].strip()},", mul)


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestWorkDividedEmission(unittest.TestCase):
    """The same add over 32 cores: the grid, one tile id, and a smaller tile.

    Everything that changes against ``EXPECTED_ADD_KTIR`` is a consequence of the
    iteration space's work division -- ``grid = [32]``, the per-core index, and
    tiles of [16, 16, 64] instead of the whole [16, 512, 64].  The *views* do not
    change: every core addresses the same buffer.
    """

    EXPECTED_DIVIDED_ADD_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_fused_add_0(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [32]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.get_compute_tile_id : index
    %1 = ktdp.construct_memory_view %arg0, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %2 = ktdp.construct_memory_view %arg1, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %3 = ktdp.construct_memory_view %arg2, sizes: [16, 512, 64], strides: [32768, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<16x512x64xf16>
    %c16 = arith.constant 16 : index
    %4 = arith.muli %0, %c16 : index
    %5 = ktdp.construct_access_tile %1[%c0, %4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<16x512x64xf16> -> !ktdp.access_tile<16x16x64xindex>
    %6 = ktdp.load %5 : <16x16x64xindex> -> tensor<16x16x64xf16>
    %c16_0 = arith.constant 16 : index
    %7 = arith.muli %0, %c16_0 : index
    %8 = ktdp.construct_access_tile %2[%c0, %7, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<16x512x64xf16> -> !ktdp.access_tile<16x16x64xindex>
    %9 = ktdp.load %8 : <16x16x64xindex> -> tensor<16x16x64xf16>
    %10 = tensor.empty() : tensor<16x16x64xf16>
    %11 = linalg.add ins(%6, %9 : tensor<16x16x64xf16>, tensor<16x16x64xf16>) outs(%10 : tensor<16x16x64xf16>) -> tensor<16x16x64xf16>
    %c16_1 = arith.constant 16 : index
    %12 = arith.muli %0, %c16_1 : index
    %13 = ktdp.construct_access_tile %3[%c0, %12, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<16x512x64xf16> -> !ktdp.access_tile<16x16x64xindex>
    ktdp.store %11, %13 : tensor<16x16x64xf16>, <16x16x64xindex>
    return
  }
}
"""

    def test_divided_add_golden(self):
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir(
            "ktir_fused_add_0", [make_op_spec(divisions={"d1": 32})]
        )
        self.assertEqual(emitted, self.EXPECTED_DIVIDED_ADD_KTIR)

    def test_one_core_emits_no_tile_id(self):
        """An undivided space costs nothing: the single-core text is unchanged."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir("ktir_fused_add_0", [make_op_spec()])
        self.assertNotIn("get_compute_tile_id", emitted)
        self.assertEqual(emitted, TestKtirEmitter.EXPECTED_ADD_KTIR)

    def test_two_divided_symbols_read_the_id_as_mixed_radix(self):
        """``d0`` takes ``id // 4`` and ``d1`` takes ``id % 4`` of an 8-core grid,
        from the one tile id -- the plan's ``inner`` and ``div`` spelled out."""
        from torch_spyre._inductor.codegen.ktir import generate_ktir

        emitted = generate_ktir(
            "ktir_add_8", [make_op_spec(divisions={"d0": 2, "d1": 4})]
        )
        self.assertIn("attributes {grid = [8]}", emitted)
        self.assertEqual(emitted.count("get_compute_tile_id"), 1)
        self.assertIn("arith.divui", emitted)
        self.assertIn("arith.remui", emitted)


@unittest.skipUnless(
    _mlir_ktdp_available(),
    "mlir_ktdp with the func/arith/linalg/scf/tensor dialect bindings is not installed",
)
class TestTiledLoopEmission(unittest.TestCase):
    """A two-level nest, planned and emitted through the ordinary path.

    Nothing special is asked for: the plan walk descends the nest because a
    ``LoopSpec`` is a loop.  The subscripts and view extents are those of a hand-written
    1-core KTIR ``sum`` kernel (``[2, 256, 64]`` strides ``[16384, 64, 1]``, tiles
    indexed ``[%n_stick, %m, %c0]``), so what comes out is a form a consumer
    already reads.
    """

    EXPECTED_TILED_ADD_KTIR = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  func.func @ktir_tiled_add_0(%arg0: index, %arg1: index, %arg2: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %1 = ktdp.construct_memory_view %arg1, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %2 = ktdp.construct_memory_view %arg2, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0_0 to %c2 step %c1 {
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      scf.for %arg4 = %c0_1 to %c256 step %c1_2 {
        %3 = ktdp.construct_access_tile %0[%arg3, %arg4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %4 = ktdp.load %3 : <1x1x64xindex> -> tensor<1x1x64xf16>
        %5 = ktdp.construct_access_tile %1[%arg3, %arg4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %6 = ktdp.load %5 : <1x1x64xindex> -> tensor<1x1x64xf16>
        %7 = tensor.empty() : tensor<1x1x64xf16>
        %8 = linalg.add ins(%4, %6 : tensor<1x1x64xf16>, tensor<1x1x64xf16>) outs(%7 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
        %9 = ktdp.construct_access_tile %2[%arg3, %arg4, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x256x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        ktdp.store %8, %9 : tensor<1x1x64xf16>, <1x1x64xindex>
      }
    }
    return
  }
}
"""

    @staticmethod
    def _tiled_nest():
        """``a + b`` over one row per iteration of a two-level nest.

        The nest is the whole kernel contract: the op sits in the inner body, so
        it is reached by walking, not by being handed out separately.
        """
        import sympy

        n_stick, m = sympy.symbols("n_stick m")
        advance = 16384 * n_stick + 64 * m
        nest, _spec, _loops = make_nested_op_spec(
            levels=[(n_stick, 2), (m, 256)],  # outermost-first
            size=[1, 1, 64],  # one row per iteration, for every arg
            advances=[advance] * 3,
        )
        return nest

    def test_two_level_nest_golden(self):
        from torch_spyre._inductor.codegen import ktir

        nest = self._tiled_nest()
        # The plan walk descends the nest, planning each buffer at the depth its
        # op sits at and turning the nest into LoopSteps: the extents below are
        # what the two levels walk over.
        plan = ktir.build_kernel_plan([nest])
        b = ktir.KtirBuilder.create(plan)
        # The builder already has the plan; opening the kernel needs only a name,
        # and the body is the plan's own steps -- the nest is not walked again.
        with b.open_kernel("ktir_tiled_add_0"):
            b.emit(plan.steps)
        # Pretty (non-generic) MLIR: the module verifies, terminators included.
        self.assertEqual(b.finish(), self.EXPECTED_TILED_ADD_KTIR)

    def test_plan_walk_grows_the_views_out_of_the_tile(self):
        """The buffer extents in the golden, read off the plan the walk built."""
        from torch_spyre._inductor.codegen import ktir

        plan = ktir.build_kernel_plan([self._tiled_nest()])
        self.assertEqual([b.buf_id for b in plan.parameters], ["arg0", "arg1", "buf0"])
        for buffer in plan.parameters:
            with self.subTest(buf_id=buffer.buf_id):
                self.assertEqual(buffer.layout.extent, (2, 256, 64))
                self.assertEqual(buffer.layout.strides, (16384, 64, 1))

    def test_generate_ktir_emits_the_nest(self):
        """No option involved: a ``LoopSpec`` is a loop, so the entry point emits
        the same text the plan-and-emit pair above does."""
        from torch_spyre._inductor.codegen import ktir

        emitted = ktir.generate_ktir("ktir_tiled_add_0", [self._tiled_nest()])
        self.assertEqual(emitted, self.EXPECTED_TILED_ADD_KTIR)


if __name__ == "__main__":
    unittest.main()
