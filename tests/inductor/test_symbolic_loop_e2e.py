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
``for_each_tile`` region over a declared dimension and runs it.

WHAT THIS ASSERTS. The headline claim is one binary serving a range of sizes.
The tensor is declared with ``tensor.to("spyre", dynamic={0: {min, max,
granularity}})`` (#5179), which reserves its HBM buffer at the declared maximum
and keeps its ``SpyreTensorLayout`` fixed across resizes, so Dynamo sees one
layout and does not recompile. The compiler builds the region's geometry at that
maximum and the launch binds the logical size into the loop's dimension slot, so
the trip count follows the size. ``TestOneBinaryServesTheRange`` resizes one
declared tensor through the minimum, the maximum, a size above the one it was
compiled at and a size below it, and asserts one graph, one kernel, right values
at every size, and that the reserved rows above the logical size are never
read.

The range is declared with ``torch._check`` inside the traced function, and it
stays there. The transfer call cannot carry it: passing ``min=``/``max=`` to
``mark_dynamic`` installs a ``StrictMinMaxConstraint``, which promises every
size in the range is valid, and the ``size % granularity == 0`` check the
compiler adds later then reads as breaking that promise and raises
``ConstraintViolationError`` instead of an ordinary guard miss. So #5179 marks
the dim bare and the range is declared from inside the region. What a later
bridge changes is who writes these three lines, not where they go.

THE MINIMUM IS TWO TILES. ``for_each_tile`` views the operand as ``(count, G,
...)``, so the tile count is a dimension of a traced tensor, and PyTorch's
contiguity check asks ``count == 1`` before it looks at the stride. The answer
becomes a guard, ``Ne(rows // G, 1)``, and a one-tile call then recompiles
(#6018, upstream pytorch/pytorch#203781). Until #6060 installs the stride-first
check, a declared range starts at ``2 * G`` and this file declares it that way.
With ``min == G`` the headline test below counted two graphs in CI.

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
and the step is a constant 64, with a lower bound of zero, so the trip count is
derived at dispatch and no division is authored anywhere. And the affine stride
is 128, the PER-ROW stride: 64 columns of fp16 is 128 bytes, the tile stride is
8192, and 8192 / 64 = 128 because the loop variable counts rows rather than
tiles.
"""

import os
import unittest

import pytest
import regex as re
import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.utils import counters
from torch._inductor.exc import InductorError
from torch._inductor.utils import fresh_cache, run_and_get_code

from utils_inductor import expects_raise

import torch_spyre  # noqa: F401  registers the "spyre" device
import torch_spyre._inductor.config as spyre_config
from torch_spyre._C import get_spyre_tensor_layout
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.execution.kernel_cache import get_cache_root_dir

ROWS = 256
COLS = 64
TILE = 64
MIN_ROWS = 2 * TILE  # two tiles, see the module docstring
MAX_ROWS = 512
SPEC = {"min": MIN_ROWS, "max": MAX_ROWS, "granularity": TILE}
# The compile size first, then every kind of other size: the minimum (two
# tiles), the maximum, above the compile size (where hint-sized geometry was
# measured wrong), and below it.
SIZES = [ROWS, MIN_ROWS, MAX_ROWS, 448, 192]

ATOL = 1e-2
RTOL = 1e-2


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


def _declare(x):
    """Move a MAX_ROWS-row host tensor to the device with the declared range.

    The transfer marks dim 0 dynamic itself (#5179), so no ``mark_dynamic`` here.
    """
    return x.to(DEVICE_NAME, dynamic={0: SPEC})


def _kernel_cache_entries():
    root = get_cache_root_dir()
    return sum(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root))


class TestSymbolicLoopOnDevice(unittest.TestCase):
    def setUp(self):
        # Every test here compiles ``tiled_gelu`` and counts what happens, so
        # none may inherit another's Dynamo cache entry, in any run order.
        torch._dynamo.reset()
        torch.manual_seed(0)

    @staticmethod
    def _compile_and_run(rows=ROWS):
        x = torch.randn(MAX_ROWS, COLS, dtype=torch.float16)
        reference = torch.nn.functional.gelu(x[:rows].float())

        on_device = _declare(x)
        on_device.resize_((rows, COLS))

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


class TestOneBinaryServesTheRange(unittest.TestCase):
    """The headline: one compiled graph and one kernel, right at every size.

    One declared tensor, one compiled function, the sizes in SIZES in that
    order. The compile happens at the first size. Every later size must be
    served by the same graph (Dynamo counts graphs) and the same kernel (the
    kernel cache holds one entry), with the right values, and the tensor's
    device layout must not move, since a moved layout is what would make Dynamo
    recompile and hide a size-specific kernel behind a green run.
    """

    def setUp(self):
        # A warm cache entry for ``tiled_gelu`` from an earlier test would make
        # the graph count below read 0 or 1 by accident. Cold every time.
        torch._dynamo.reset()
        torch.manual_seed(0)

    def test_one_graph_one_kernel_right_at_every_size(self):
        counters.clear()
        x = torch.randn(MAX_ROWS, COLS, dtype=torch.float16)
        y = _declare(x)
        layout = str(get_spyre_tensor_layout(y))

        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
            compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
            for rows in SIZES:
                if y.size(0) != rows:
                    y.resize_((rows, COLS))
                out = compiled(y).cpu().float()
                self.assertEqual(
                    str(get_spyre_tensor_layout(y)),
                    layout,
                    f"the device layout changed at {rows} rows",
                )
                torch.testing.assert_close(
                    out,
                    torch.nn.functional.gelu(x[:rows].float()),
                    atol=ATOL,
                    rtol=RTOL,
                    msg=f"at {rows} rows",
                )
            self.assertEqual(counters["stats"]["unique_graphs"], 1)
            self.assertEqual(_kernel_cache_entries(), 1)

    @expects_raise
    def test_without_a_declaration_the_loop_is_refused(self):
        """The control. A bare ``mark_dynamic`` reserves nothing: a new size is
        a new allocation with its own device layout, the layout guard carries
        the exact row count and every size recompiles (measured: four sizes,
        four graphs). A symbolic loop over such a dimension would be a per-size
        kernel whose extent is sized from a ShapeEnv bound no reservation backs,
        so it is refused at compile with the declaration named as the fix. The
        declaration is what buys the one binary.
        """
        x = torch.randn(ROWS, COLS, dtype=torch.float16)
        y = x.to(DEVICE_NAME)
        torch._dynamo.mark_dynamic(y, 0)
        compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)

        with pytest.raises(InductorError, match="no tensor.to.*declaration covers"):
            compiled(y)

    @expects_raise
    def test_an_undeclared_operand_riding_the_declared_symbol_is_refused(self):
        """Two loop operands tiled on the same axis must carry one symbol, and
        ``torch._check(a.shape[0] == b.shape[0])`` (or Dynamo's duck shaping of
        two equal dynamic sizes) gives them one. If only one of them is
        declared, the other's buffer was never reserved at the maximum the
        declared symbol builds for, so the compile refuses and names the symbol.
        Raised in the compile_fx wrapper, before Inductor, so Dynamo reports it
        as a backend failure. Without the check this would compile: the loop's
        count is over a declared symbol.
        """

        def both(a, b):
            rows = a.shape[0]
            torch._check(rows == b.shape[0])
            torch._check(rows >= MIN_ROWS)
            torch._check(rows <= MAX_ROWS)
            torch._check(rows % TILE == 0)

            def body(_carry, tiles):
                ta, tb = tiles
                return None, torch.nn.functional.gelu(ta) + tb

            _carry, out = for_each_tile(
                body, (a, b), dims=(0, 0), tile_size=TILE, out_dim=0
            )
            return out

        x = torch.randn(MAX_ROWS, COLS, dtype=torch.float16)
        y = _declare(x)
        y.resize_((ROWS, COLS))
        z = torch.randn(ROWS, COLS, dtype=torch.float16).to(DEVICE_NAME)
        torch._dynamo.mark_dynamic(z, 0)

        compiled = torch.compile(both, backend="inductor", fullgraph=True)
        with pytest.raises(
            (InductorError, BackendCompilerFailed), match="undeclared on another"
        ):
            compiled(y, z)

    def test_the_reserved_rows_above_the_size_are_never_read(self):
        """NaN in the reserved rows, moved at the maximum then shrunk, so the
        device really holds NaN above the logical size. Geometry built at the
        maximum that read past the logical size would surface it. A trip count
        bound at launch stops before it.
        """
        x = torch.randn(MAX_ROWS, COLS, dtype=torch.float16)
        x[ROWS:] = float("nan")
        y = _declare(x)
        y.resize_((ROWS, COLS))

        compiled = torch.compile(tiled_gelu, backend="inductor", fullgraph=True)
        out = compiled(y).cpu().float()

        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(
            out, torch.nn.functional.gelu(x[:ROWS].float()), atol=ATOL, rtol=RTOL
        )


if __name__ == "__main__":
    unittest.main()
