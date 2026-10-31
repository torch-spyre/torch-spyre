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

"""generate_bundle with a symbolic loop count, driven end to end.

Every other test of this path exercises the helpers directly. That is how the
original version of this feature shipped broken: it collected loop dimensions
by scanning the SDSC symbol table, where a loop dimension can never appear, so
it found nothing on every kernel while 22 helper tests stayed green. These
tests go through ``generate_bundle`` and read the file it writes, which is the
only level at which that mistake is visible.

``compile_op_spec`` is mocked, the same way test_symbolic_dim_bundle.py mocks
it, because the SDSC content is irrelevant here: the whole point of this route
is that nothing symbolic reaches the SDSC.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import sympy
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.utils._sympy.functions import FloorDiv

from utils_inductor import expects_raise

from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec

SYM = "s0"
TILE = 64
MAX = 512
# The tile stride from the worked example in the design: one row of 1024 fp16
# is 2048 bytes, so a 64-row tile advances 131072. The loop variable counts
# ELEMENTS, so the emitted stride must be the row stride, 2048.
TILE_STRIDE = 131072
ROW_STRIDE = TILE_STRIDE // TILE


def _sym():
    return sympy.Symbol(SYM, integer=True, positive=True)


def _sdsc_json(sdsc_idx: int = 0) -> dict:
    """Minimal SDSC JSON. Deliberately carries no dimension symbol."""
    return {
        f"{sdsc_idx}_fused_test": {
            "numCoresUsed_": 1,
            "dscs_": [{"op": {"dimToSymbolMapping_": {}, "scheduleTree_": []}}],
        }
    }


def _op_spec() -> OpSpec:
    """A stub body op. compile_op_spec is mocked, so its content is unused."""
    return OpSpec(
        op="gelu", is_reduction=False, iteration_space={}, args=[], op_info={}
    )


class _BundleHarness(InductorTestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()
        super().tearDown()

    def _run(self, specs, compiled_entries, pool_size=0):
        """Returns (bundle.mlir text, the SymbolKind list generate_bundle returns).

        compile_op_spec is called twice per OpSpec, once to derive the cache key
        and once for real, so each entry is yielded twice.
        """
        side_effects = [e for entry in compiled_entries for e in (entry, entry)]
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=side_effects,
        ):
            kinds = generate_bundle("test", self.output_dir, specs, pool_size=pool_size)
        with open(os.path.join(self.output_dir, "bundle.mlir")) as f:
            return f.read(), kinds

    @staticmethod
    def _entry(strides=None):
        """One compiled OpSpec: json, base symbol values, affine strides, kinds."""
        return (
            _sdsc_json(),
            [0],
            [[{_sym(): strides}]] if strides else [],
            [SymbolKind.kernel(0)],
        )

    @staticmethod
    def _symbolic_loop(tile=TILE, bounds=None, sources=None, body=None):
        """A symbolic loop. `body` must be non-empty: op_spec_validation
        refuses an empty one, and one compiled entry is consumed per OpSpec."""
        return LoopSpec(
            count=FloorDiv(_sym(), tile),
            body=body if body is not None else [_op_spec()],
            count_symbol_bounds=({SYM: (MAX, tile)} if bounds is None else bounds),
            count_symbol_sources={SYM: (0, 0)} if sources is None else sources,
        )


class TestTheDimensionReachesTheBundle(_BundleHarness):
    """The claim the helper tests could not make."""

    def test_the_parameter_is_declared_with_its_contract(self):
        bundle, _kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])

        self.assertIn(
            f"%dim_{SYM}_base: !sdscbundle.input_arg"
            f"<index, granularity={TILE}, max_value={MAX}>",
            bundle,
        )

    def test_the_runtime_is_told_where_to_read_it_from(self):
        """The returned SymbolKind list is what the launch path reads.

        This is the assertion the original defect would have failed: it
        collected loop dimensions from the SDSC symbol table, so this list came
        back with no loop dimension in it at all and the runtime had nothing
        telling it to bind a size rather than an address.
        """
        _bundle, kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])

        loop_dims = [k for k in kinds if k.is_loop_dimension]
        self.assertEqual(len(loop_dims), 1, f"got {[k.kind for k in kinds]}")
        self.assertEqual(loop_dims[0].pytorch_sym, SYM)
        self.assertEqual((loop_dims[0].arg_index, loop_dims[0].dim_index), (0, 0))
        self.assertEqual(loop_dims[0].granularity, TILE)
        self.assertEqual(loop_dims[0].max_value, MAX)

    def test_it_is_not_an_sdsc_dimension_symbol(self):
        """So the compile-boundary gate keeps refusing the old route only."""
        _bundle, kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])

        self.assertFalse(
            any(k.is_dimension for k in kinds),
            "a loop dimension was classified as an SDSC dimension symbol, which "
            "async_compile refuses at the compile boundary",
        )

    def test_the_parameter_comes_last(self):
        """Adding one must never shift an existing parameter's position.

        The runtime fills these slots by order, so a loop dimension inserted
        before the address parameters would rebind every one of them.
        """
        bundle, _kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])
        signature = bundle.split("func.func @sdsc_bundle(")[1].split(")")[0]

        self.assertLess(signature.index("%arg_0"), signature.index(f"%dim_{SYM}"))

    def test_the_extract_precedes_the_loop_constants(self):
        """A symbolic bound IS that SSA value, so it must already be in scope."""
        bundle, _kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])

        self.assertLess(
            bundle.index(f"%dim_{SYM} = sdscbundle.input_arg_extract"),
            bundle.index("%c0 = arith.constant 0 : index"),
        )


class TestTheLoopForm(_BundleHarness):
    def test_the_bound_is_the_dimension_and_the_step_is_the_tile(self):
        bundle, _kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])

        self.assertIn(f"%step_0 = arith.constant {TILE} : index", bundle)
        self.assertIn(f"scf.for %i_0 = %c0 to %dim_{SYM} step %step_0", bundle)

    def test_no_division_is_authored(self):
        """The device derives (ub - lb) / step itself, and we never write one."""
        bundle, _kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])

        for forbidden in ("divsi", "divui", "ceildiv", "floordiv"):
            self.assertNotIn(forbidden, bundle, f"authored a {forbidden}")

    def test_the_stride_is_divided_by_the_step(self):
        """Because the loop variable counts elements, not tiles.

        The tile stride against an element-stepping loop would advance a whole
        tile too far per trip, which is a wrong answer rather than a crash.
        """
        bundle, _kinds = self._run([self._symbolic_loop()], [self._entry(TILE_STRIDE)])

        self.assertIn(f"{ROW_STRIDE}*d0", bundle)
        self.assertNotIn(f"{TILE_STRIDE}*d0", bundle)

    def test_a_concrete_loop_still_emits_the_old_form(self):
        """The regression half: nothing changes for a loop that has no symbol."""
        spec = LoopSpec(count=sympy.Integer(4), body=[_op_spec()])

        bundle, kinds = self._run([spec], [self._entry()])

        self.assertIn("%loop_bound_0 = arith.constant 4 : index", bundle)
        self.assertIn("scf.for %i_0 = %c0 to %loop_bound_0 step %c1", bundle)
        self.assertFalse(any(k.is_loop_dimension for k in kinds))


class TestItRefusesRatherThanGuesses(_BundleHarness):
    @expects_raise
    def test_bounds_with_no_source_refuse_and_name_the_symbol(self):
        """A wrong (arg_index, dim_index) is silently wrong on every launch."""
        spec = self._symbolic_loop(sources={})

        with self.assertRaises(NotImplementedError) as caught:
            self._run([spec], [self._entry(TILE_STRIDE)])

        self.assertIn(SYM, str(caught.exception))
        self.assertIn("no source", str(caught.exception))

    @expects_raise
    def test_two_loops_disagreeing_about_one_dimension_refuse(self):
        """One parameter cannot serve two different tile sizes.

        Stricter than strictly necessary, and deliberately so until the
        granularity chooser guarantees every tile size divides the declared one.
        """
        # Every LoopSpec needs a body: op_spec_validation refuses an empty one
        # before generate_bundle gets as far as merging the maps.
        inner = self._symbolic_loop(
            tile=128, bounds={SYM: (MAX, 128)}, body=[_op_spec()]
        )
        outer = self._symbolic_loop(
            tile=64, bounds={SYM: (MAX, 64)}, body=[_op_spec(), inner]
        )

        with self.assertRaises(NotImplementedError) as caught:
            self._run([outer], [self._entry(TILE_STRIDE)] * 2)

        self.assertIn(SYM, str(caught.exception))
        self.assertIn("disagree", str(caught.exception))


class TestEveryPassScalesANestedLevelTheSameWay(_BundleHarness):
    """Concrete and symbolic levels nested both ways, through the emitter.

    _collect_affine_maps registers each map under its scaled stride key and
    _emit_specs looks it up by the key it builds itself, both through
    _scaled_strides. If the two scaled differently, the lookup would raise
    KeyError. The SDSC here carries an HBM base address, so the emitter really
    issues the affine.apply and does that lookup.
    """

    OUTER_STRIDE = 1048576
    INNER_STRIDE = 4096

    @staticmethod
    def _addressed_entry(per_level_strides):
        sdsc = {
            "0_fused_test": {
                "numCoresUsed_": 1,
                "dscs_": [
                    {
                        "op": {
                            "dimToSymbolMapping_": {},
                            "scheduleTree_": [
                                {
                                    "component_": "hbm",
                                    "startAddressCoreCorelet_": {
                                        "data_": {"[0, 0, 0]": "-1"}
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        }
        return (sdsc, [0], [per_level_strides], [SymbolKind.kernel(0)])

    def test_a_symbolic_level_inside_a_concrete_one(self):
        inner = self._symbolic_loop(body=[_op_spec()])
        outer = LoopSpec(count=sympy.Integer(4), body=[inner])
        entry = self._addressed_entry(
            [{sympy.Symbol("a"): self.OUTER_STRIDE}, {_sym(): TILE_STRIDE}]
        )

        bundle, _kinds = self._run([outer], [entry])

        # The concrete level keeps its stride, the symbolic one is divided.
        self.assertIn(
            f"(s0 + {self.OUTER_STRIDE}*d0 + {ROW_STRIDE}*d1)", bundle, bundle
        )
        self.assertIn("affine.apply #map_0(%i_0, %i_1)", bundle)

    def test_a_concrete_level_inside_a_symbolic_one(self):
        inner = LoopSpec(count=sympy.Integer(2), body=[_op_spec()])
        outer = self._symbolic_loop(body=[inner])
        entry = self._addressed_entry(
            [{_sym(): TILE_STRIDE}, {sympy.Symbol("b"): self.INNER_STRIDE}]
        )

        bundle, _kinds = self._run([outer], [entry])

        self.assertIn(
            f"(s0 + {ROW_STRIDE}*d0 + {self.INNER_STRIDE}*d1)", bundle, bundle
        )
        self.assertIn("affine.apply #map_0(%i_0, %i_1)", bundle)


class TestOneParameterServesEveryLoopOnThatDimension(_BundleHarness):
    def test_two_loops_sharing_a_dimension_share_one_parameter(self):
        """The same shape variable reaches every loop that tiles on it."""
        inner = self._symbolic_loop(body=[_op_spec()])
        outer = self._symbolic_loop(body=[_op_spec(), inner])

        bundle, kinds = self._run([outer], [self._entry(TILE_STRIDE)] * 2)

        self.assertEqual(bundle.count(f"%dim_{SYM}_base:"), 1)
        self.assertEqual(len([k for k in kinds if k.is_loop_dimension]), 1)


if __name__ == "__main__":
    unittest.main()
