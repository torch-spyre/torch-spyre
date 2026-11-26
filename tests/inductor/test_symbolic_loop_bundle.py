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

import dataclasses
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
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import CountSource, LoopSpec, OpSpec

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


class TestATiedPairReachesTheBundleAsOneSymbol(_BundleHarness):
    """From ``counted_loop`` to ``generate_bundle`` with a count spelled in the
    replaced symbol. ``torch._check(a == b)`` lands in the ShapeEnv's
    replacements; ``counted_loop`` applies them once, so the LoopSpec's count,
    its bounds and the bundle's parameter all name the surviving symbol, and
    ``_loop_level`` finds it in ``loop_dim_ssa``. ``symbolic_count_bounds`` alone
    being right is not this test: the stored count is what the bundle decomposes.
    """

    def test_the_bundle_declares_the_surviving_symbol(self):
        from types import SimpleNamespace

        from torch._inductor.virtualized import V

        from torch_spyre._inductor.pass_utils import declared_dims
        from torch_spyre._inductor.spyre_kernel import SpyreKernel

        other = sympy.Symbol("s1", integer=True, positive=True)
        env = SimpleNamespace(
            bound_sympy=lambda _e: SimpleNamespace(lower=TILE, upper=sympy.oo),
            replacements={other: _sym()},
        )
        graph = SimpleNamespace(sizevars=SimpleNamespace(shape_env=env))

        with declared_dims({SYM: (TILE, MAX, TILE)}), V.set_graph_handler(graph):
            spec = SpyreKernel.counted_loop(FloorDiv(other, TILE), [_op_spec()])

        self.assertEqual(str(spec.count), str(FloorDiv(_sym(), TILE)))
        self.assertEqual(spec.count_symbol_bounds, {SYM: (MAX, TILE)})
        spec = dataclasses.replace(spec, count_symbol_sources={SYM: (0, 0)})

        bundle, kinds = self._run([spec], [self._entry(TILE_STRIDE)])

        self.assertIn(f"%dim_{SYM}_base: !sdscbundle.input_arg", bundle)
        self.assertNotIn("s1", bundle)
        self.assertEqual([k.pytorch_sym for k in kinds if k.is_loop_dimension], [SYM])


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

        with self.assertRaises(Unsupported) as caught:
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

        with self.assertRaises(Unsupported) as caught:
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


class _DeviceReadHarness(_BundleHarness):
    """A loop whose trip count an earlier op in the bundle writes to the pool."""

    COUNT_OFFSET = 4096
    MAX_TILES = 16
    POOL_SIZE = 65536

    @staticmethod
    def _pool_entry(sdsc_idx):
        """One compiled OpSpec that uses the pool, so ``%pool`` is defined."""
        return (
            _sdsc_json(sdsc_idx),
            [0, 0],
            [],
            [SymbolKind.kernel(0), SymbolKind.pool()],
        )

    def _source(self, **overrides):
        fields = {"pool_offset": self.COUNT_OFFSET, "max_value": self.MAX_TILES}
        fields.update(overrides)
        return CountSource(**fields)

    def _device_read_loop(self, count=None, source=None):
        return LoopSpec(
            count=sympy.Integer(self.MAX_TILES if count is None else count),
            body=[_op_spec()],
            count_from_buffer=self._source() if source is None else source,
        )

    def _run_with_producer(self, loop):
        """The count's producer op, then ``loop`` holding one op."""
        entries = [self._pool_entry(i) for i in range(2)]
        bundle, _kinds = self._run([_op_spec(), loop], entries, self.POOL_SIZE)
        return bundle


class TestADeviceReadCountIsReadRightBeforeItsLoop(_DeviceReadHarness):
    """The read has to come after the op that writes the count and before the
    loop that uses it, so it cannot sit with the other loop constants at the
    top of the bundle.
    """

    def test_producer_then_read_then_loop(self):
        bundle = self._run_with_producer(self._device_read_loop())

        producer = bundle.index('sdsc_filename="sdsc_0.json"')
        read = bundle.index("sdscbundle.load_value_at_address")
        loop = bundle.index("scf.for %i_0")
        self.assertLess(producer, read, bundle)
        self.assertLess(read, loop, bundle)

    def test_the_read_and_the_loop_form(self):
        bundle = self._run_with_producer(self._device_read_loop())

        self.assertIn(
            "%n_0 = sdscbundle.load_value_at_address %cnt_addr_0 "
            f"{{granularity = 1, max_value = {self.MAX_TILES}}} : index",
            bundle,
        )
        self.assertIn("scf.for %i_0 = %c0 to %n_0 step %c1", bundle)
        self.assertNotIn("%loop_bound_0", bundle)

    def test_the_address_is_set_up_at_the_top(self):
        """The address is a fixed offset into the pool, so it is a constant."""
        bundle = self._run_with_producer(self._device_read_loop())

        self.assertIn(
            f"%cnt_off_0 = arith.constant {self.COUNT_OFFSET} : index", bundle
        )
        self.assertIn("%cnt_addr_0 = arith.addi %pool, %cnt_off_0 : index", bundle)
        self.assertLess(
            bundle.index("%pool = sdscbundle.device_mem_allocate"),
            bundle.index("%cnt_addr_0 ="),
        )
        self.assertLess(
            bundle.index("%cnt_addr_0 ="), bundle.index("sdscbundle.sdsc_execute")
        )

    def test_no_division_is_authored(self):
        bundle = self._run_with_producer(self._device_read_loop())

        for forbidden in ("divsi", "divui", "ceildiv", "floordiv"):
            self.assertNotIn(forbidden, bundle, f"authored a {forbidden}")

    def test_inside_a_constant_loop_the_read_is_inside_too(self):
        outer = LoopSpec(count=sympy.Integer(4), body=[self._device_read_loop()])

        bundle = self._run_with_producer(outer)

        outer_loop = bundle.index("scf.for %i_0 = %c0 to %loop_bound_0 step %c1")
        read = bundle.index("%n_1 = sdscbundle.load_value_at_address %cnt_addr_1")
        inner_loop = bundle.index("scf.for %i_1 = %c0 to %n_1 step %c1")
        self.assertLess(outer_loop, read, bundle)
        self.assertLess(read, inner_loop, bundle)
        # Same depth as the inner loop it bounds, one deeper than the outer one.
        self.assertIn("\n\t\t\t%n_1 = sdscbundle.load_value_at_address", bundle)
        self.assertIn("\n\t\t\tscf.for %i_1", bundle)

    def test_a_count_written_inside_the_outer_loop_is_read_after_it(self):
        """The count is rewritten on every outer trip, so the read has to follow
        that write inside the outer body, not sit above the outer loop where it
        would see the count only once, before anything wrote it."""
        outer = LoopSpec(
            count=sympy.Integer(4), body=[_op_spec(), self._device_read_loop()]
        )
        entries = [self._pool_entry(i) for i in range(3)]

        bundle, _kinds = self._run([outer, _op_spec()], entries, self.POOL_SIZE)

        def depth(marker):
            line = next(line for line in bundle.splitlines() if marker in line)
            return len(line) - len(line.lstrip("\t"))

        outer_loop = "scf.for %i_0 = %c0 to %loop_bound_0 step %c1"
        producer = 'sdsc_filename="sdsc_0.json"'
        read = "%n_1 = sdscbundle.load_value_at_address %cnt_addr_1"
        inner_loop = "scf.for %i_1 = %c0 to %n_1 step %c1"
        after = 'sdsc_filename="sdsc_2.json"'
        order = [outer_loop, producer, read, inner_loop, after]
        positions = [bundle.index(marker) for marker in order]
        self.assertEqual(positions, sorted(positions), bundle)
        # The producer, the read and the inner loop all sit in the outer body,
        # and the op after the outer loop is back at its level.
        self.assertEqual(depth(producer), depth(outer_loop) + 1, bundle)
        self.assertEqual(depth(read), depth(outer_loop) + 1, bundle)
        self.assertEqual(depth(inner_loop), depth(outer_loop) + 1, bundle)
        self.assertEqual(depth(after), depth(outer_loop), bundle)


class TestADeviceReadCountRefusesRatherThanGuesses(_DeviceReadHarness):
    """The backend plans every such loop for ``max_value`` trips, so a loop and
    a source that disagree are refused rather than emitted."""

    def _refuses(self, loop, fragment):
        with self.assertRaises(Unsupported) as caught:
            self._run_with_producer(loop)
        self.assertIn(fragment, str(caught.exception))

    @expects_raise
    def test_a_non_positive_max_value(self):
        self._refuses(
            self._device_read_loop(source=self._source(max_value=0)),
            "positive max_value and granularity",
        )

    @expects_raise
    def test_a_non_positive_granularity(self):
        self._refuses(
            self._device_read_loop(source=self._source(granularity=0)),
            "positive max_value and granularity",
        )

    @expects_raise
    def test_a_max_value_other_than_the_static_count(self):
        self._refuses(
            self._device_read_loop(count=8, source=self._source(max_value=16)),
            "planned for 8 trips",
        )

    @expects_raise
    def test_a_max_value_not_a_multiple_of_the_granularity(self):
        self._refuses(
            self._device_read_loop(source=self._source(granularity=3)),
            "not a multiple of granularity=3",
        )

    @expects_raise
    def test_a_symbolic_count_too(self):
        loop = dataclasses.replace(
            self._symbolic_loop(), count_from_buffer=self._source()
        )
        self._refuses(loop, "one kind of bound")

    @expects_raise
    def test_no_pool_to_read_from(self):
        """No op uses the pool, so ``%pool`` is never defined."""
        loop = self._device_read_loop()

        with self.assertRaises(Unsupported) as caught:
            self._run([_op_spec(), loop], [self._entry(), self._entry()])

        self.assertIn("only supported from the bundle's pool", str(caught.exception))


class TestADeviceReadCountSurvivesTheRealSerializer(unittest.TestCase):
    """The generated wrapper rebuilds the LoopSpecs from source before the
    bundle is written, so a count source the serializer drops never reaches it."""

    @staticmethod
    def _round_trip(spec):
        from torch._inductor.utils import IndentedBuffer

        from torch_spyre._inductor.spyre_kernel import (
            _codegen_op_spec_list,
            _sympy_literal,
        )

        buf = IndentedBuffer()
        _codegen_op_spec_list([spec], buf, _sympy_literal)
        source = buf.getvalue()
        namespace = {
            "LoopSpec": LoopSpec,
            "CountSource": CountSource,
            "sympify": sympy.sympify,
        }
        return eval(f"[{source}]", namespace), source  # noqa: S307

    def test_the_count_source_comes_back_intact(self):
        source = CountSource(pool_offset=4096, max_value=16)
        spec = LoopSpec(count=sympy.Integer(16), body=[], count_from_buffer=source)

        (reloaded,), text = self._round_trip(spec)

        self.assertIn("count_from_buffer", text)
        self.assertEqual(reloaded.count_from_buffer, source)

    def test_a_loop_without_one_writes_nothing_new(self):
        """Every existing generated wrapper stays byte-for-byte the same."""
        spec = LoopSpec(count=sympy.Integer(16), body=[])

        (reloaded,), text = self._round_trip(spec)

        self.assertNotIn("count_from_buffer", text)
        self.assertIsNone(reloaded.count_from_buffer)


class TestTheCacheKeySeparatesADeviceReadCount(unittest.TestCase):
    """The read and its address live in the bundle, not in any SDSC JSON, so
    without an entry of its own a device-read loop and a constant loop with the
    same body and maximum share one binary."""

    @staticmethod
    def _key(source):
        from torch_spyre.execution.kernel_cache import compute_specs_hash

        return compute_specs_hash(
            [LoopSpec(count=sympy.Integer(16), body=[], count_from_buffer=source)]
        )

    def test_a_constant_loop_and_a_device_read_one_do_not_share_an_entry(self):
        self.assertNotEqual(
            self._key(None), self._key(CountSource(pool_offset=0, max_value=16))
        )

    def test_two_pool_offsets_do_not_share_an_entry(self):
        self.assertNotEqual(
            self._key(CountSource(pool_offset=0, max_value=16)),
            self._key(CountSource(pool_offset=4096, max_value=16)),
        )

    def test_two_maxima_do_not_share_an_entry(self):
        self.assertNotEqual(
            self._key(CountSource(pool_offset=0, max_value=16)),
            self._key(CountSource(pool_offset=0, max_value=8)),
        )

    def test_the_same_source_does_share_one(self):
        """The count read at run time is not part of the key, so every call of
        the same kernel, whatever count it reads, reuses one binary."""
        self.assertEqual(
            self._key(CountSource(pool_offset=0, max_value=16)),
            self._key(CountSource(pool_offset=0, max_value=16)),
        )


if __name__ == "__main__":
    unittest.main()
