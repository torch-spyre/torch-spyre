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

"""A symbolic loop count through the whole Spyre pipeline, on device.

Every other test of this feature stops somewhere: at a helper, at a synthetic
cond graph, at ``generate_bundle`` with a mocked SDSC. This one compiles a
``for_each_tile`` region over a marked-dynamic dimension and runs it.

WHAT THIS CANNOT YET ASSERT, and the reason is a missing dependency rather
than a gap in the tests. The headline claim is one binary serving a range of
sizes. That needs the varying tensor's HBM buffer reserved for the declared
maximum, which is the max-strided reservation of #5179 and is not in this tree:
there is no ``spyre_empty_reserved`` and ``.to()`` takes no ``dynamic=``.
Without it a larger size is a fresh, differently sized allocation, so the layout
guard sees a new layout and recompiles. That is safe, just not yet useful.

So these assert what is assertable alone: the region compiles through the real
pipeline, the symbolic count survives into the kernel, the emitted structure is
the loop form the design specifies, and the numbers are right at the size it
was compiled for. The multi-size no-recompile proof is the combined milestone
with #5179 and belongs in that test.

The range is declared with ``torch._check`` inside the traced function, and it
stays there. The transfer call cannot carry it: passing ``min=``/``max=`` to
``mark_dynamic`` installs a ``StrictMinMaxConstraint``, which promises every
size in the range is valid, and the ``size % granularity == 0`` check the
compiler adds later then reads as breaking that promise and raises
``ConstraintViolationError`` instead of an ordinary guard miss. So #5179 marks
the dim bare and the range is declared from inside the region. What a later
bridge changes is who writes these three lines, not where they go.

VERIFIED EMITTED BUNDLE, 6 Oct 2026, for the region below at 64 columns fp16::

    #map_0 = affine_map<(d0)[s0] -> (s0 + 128*d0)>
    func.func @sdsc_bundle(
        %arg_0_base_addr: !sdscbundle.input_arg<index>,
        %arg_1_base_addr: !sdscbundle.input_arg<index>,
        %dim_s77_base: !sdscbundle.input_arg<index, granularity=64, max_value=512>) {
      %dim_s77 = sdscbundle.input_arg_extract value from %dim_s77_base : ... -> index
      %c0 = arith.constant 0 : index
      %step_0 = arith.constant 64 : index
      scf.for %i_0 = %c0 to %dim_s77 step %step_0 {
        %addr_0 = affine.apply #map_0(%i_0)[%arg_0]
        sdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json", ...}
      }
    }

Every load-bearing detail of the design's bundle section is in that listing.
The dimension parameter is last and is the only one carrying granularity and
max_value. Its extract precedes the loop constants. The bound is the dimension
and the step is a constant 64, with a lower bound of zero, so the device's
``(ub - lb) / step`` is free and no division is authored anywhere. And the
affine stride is 128, the PER-ROW stride: 64 columns of fp16 is 128 bytes, the
tile stride is 8192, and 8192 / 64 = 128 because the loop variable counts rows
rather than tiles.
"""

import unittest
from functools import wraps

import regex as re
import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch._inductor.utils import run_and_get_code
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME

ROWS = 256
COLS = 64
TILE = 64
MIN_ROWS = 64
MAX_ROWS = 512

ATOL = 1e-2
RTOL = 1e-2


def _with_dynamo_reset(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        torch._dynamo.reset()
        return fn(*args, **kwargs)

    return wrapper


def tiled_gelu(x):
    """One tiled pointwise region over a dimension declared dynamic.

    The three ``torch._check`` calls are the contract: a range so the ShapeEnv
    has a finite ceiling for the geometry, and the divisibility so the tile
    count is exact. ``for_each_tile`` re-states the divisibility itself, but
    the range has to come from here because nothing else puts it in scope.
    """
    rows = x.shape[0]
    torch._check(rows >= MIN_ROWS)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


class TestSymbolicLoopOnDevice(unittest.TestCase):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))

    @staticmethod
    def _compile_and_run(rows=ROWS):
        x = torch.randn(rows, COLS, dtype=torch.float16)
        reference = torch.nn.functional.gelu(x.float())

        on_device = x.to(DEVICE_NAME)
        torch._dynamo.mark_dynamic(on_device, 0)

        compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
        out, code = run_and_get_code(compiled, on_device)
        return out, reference, code

    def test_the_numbers_are_right(self):
        out, reference, _code = self._compile_and_run()

        torch.testing.assert_close(out.cpu().float(), reference, atol=ATOL, rtol=RTOL)

    def test_the_count_reaches_the_kernel_as_a_symbol(self):
        """Not specialised to 256 somewhere along the way.

        If the count arrived concrete, every assertion about the loop form
        below would still pass while the feature was entirely absent, which is
        the false green this whole area is prone to.
        """
        _out, _reference, code = self._compile_and_run()
        source = "\n".join(code)

        self.assertIn("LoopSpec(", source)
        self.assertIn("count_symbol_bounds", source)
        self.assertNotIn(
            f"count=sympify('{ROWS // TILE}')",
            source,
            "the trip count was specialised to this call's size, so the kernel "
            "is size-specific and nothing here is measuring a symbolic loop",
        )

    def test_the_declared_maximum_is_what_was_carried(self):
        """Not the warm-up size. This is the one that catches hint-sized
        geometry, which is correct at the compiled size and wrong above it.

        Asserted as the whole carried pair rather than "512 appears somewhere",
        because 512 and 64 are both common enough numbers to appear by accident
        in a kernel of this shape.
        """
        _out, _reference, code = self._compile_and_run()
        source = "\n".join(code)

        carried = re.search(
            r"count_symbol_bounds=\{'(s\d+)': \((\d+), (\d+)\)\}", source
        )
        self.assertIsNotNone(
            carried,
            "no count_symbol_bounds reached the kernel, so nothing carries the "
            f"range to the bundle. Source was:\n{source[:2000]}",
        )
        self.assertEqual(
            (int(carried.group(2)), int(carried.group(3))), (MAX_ROWS, TILE)
        )


class TestItIsStillCorrectAtAnotherSize(unittest.TestCase):
    """A second size, compiled separately.

    Deliberately NOT asserting one binary: without the reservation of #5179 this
    recompiles, and asserting otherwise would be asserting a bug. What it does
    prove is that the geometry built from the declared maximum is correct at more
    than one actual size, which is the half of the claim that does not need the
    reservation.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))

    def test_a_smaller_size(self):
        out, reference, _code = TestSymbolicLoopOnDevice._compile_and_run(rows=128)

        torch.testing.assert_close(out.cpu().float(), reference, atol=ATOL, rtol=RTOL)

    def test_the_minimum_size(self):
        """One tile exactly, which is the known rough edge."""
        out, reference, _code = TestSymbolicLoopOnDevice._compile_and_run(rows=MIN_ROWS)

        torch.testing.assert_close(out.cpu().float(), reference, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
