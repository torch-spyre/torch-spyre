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

"""End-to-end Spyre-device tests for WhileLoop -> OpSpec/LoopSpec lowering.

Compiles each fixture, runs it on the Spyre device, and compares against a
CPU reference -- for IR-level / mocked-IR unit tests of the lowering
machinery itself, see test_for_each_tile_lowering.py.

Minimum coverage per docs/superpowers/specs/2026-09-09-while-loop-lowering-design.md:
1. Single carry (this file: test_carry_mode_split_k) -- currently XFAIL on a
   read-copy/stick-layout gap; see that test's own docstring.
2 (carry + Kind.SLICE tile-advancing input): covered implicitly by
   test_carry_mode_split_k, whose X/Y operands are both Kind.SLICE.
4. Multiple independent carries: covered by test_carry_mode_online_softmax
   (carry = (m, denom, acc), an online-softmax flash-attention inner loop).
Cases 3, 5, 6 (Kind.GATHER, nested for_each_tile, and the deliberate-decline
case) are follow-on work -- tracked as open items rather than duplicated
here, since each needs its own fixture beyond what's vendored so far.

test_map_mode_split_m (map mode: Kind.SLICE + Kind.INVARIANT operands, a
stacking carry, no user carry) passes end to end with verified numerics and
is the case that exercises the full splice -> DimHint synthesis ->
coarse-tile -> single scf.for pipeline.
"""

import unittest

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME

from tests.inductor.for_each_tile_fixtures import (
    attention_inputs,
    matmul_inputs,
    online_softmax_fn,
    online_softmax_reference,
    split_k_fn,
    split_m_fn,
)


class TestForEachTileE2E(unittest.TestCase):
    # Spyre's matmul runs in fp16, so the reference has to be an fp16-faithful
    # one: cast the operands first, then accumulate in fp32 on CPU. Comparing
    # against the fp32 product of fp32 operands would fail on rounding alone,
    # independently of anything this test is meant to check.
    #
    # Operands must also be cast to fp16 BEFORE the host->device transfer, not
    # after: `t.to(DEVICE_NAME).half()` (transfer fp32, cast on device)
    # currently produces garbage on this backend for reasons unrelated to
    # while_loop lowering -- a plain `torch.compile`d `a @ b` reproduces it
    # without any for_each_tile involved. `t.half().to(DEVICE_NAME)` is the
    # idiom the rest of the compiled-op suite uses (see
    # tests/inductor/test_inductor_matmul.py, whose inputs are constructed
    # `dtype=torch.float16` up front).
    #
    # rtol is the binding constraint here: operand/output magnitudes are
    # O(1)-O(10), so rtol * |expected| dominates atol (which only matters
    # near zero).
    ATOL = 0.1
    RTOL = 0.1

    @staticmethod
    def _operands():
        (X, Y), _ = matmul_inputs()
        ref = (X.half().float()) @ (Y.half().float())
        return X.half().to(DEVICE_NAME), Y.half().to(DEVICE_NAME), ref

    def test_map_mode_split_m(self):
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_m_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    @unittest.expectedFailure
    def test_carry_mode_split_k(self):
        """Carry mode: accumulate a split-K matmul across tiles.

        XFAIL on a gap in read-copy layout reconciliation, downstream of and
        distinct from everything this test's own lowering path needs -- the
        accumulator carry itself is wired correctly and WSR's own tiled-
        reduction accumulator (coarse_tile_fill/combine on the K level) picks
        the K accumulation up as intended. Tracked as issue #4460.

        The gap: ``for_each_tile``'s ``xs`` leaves for ``dims=(-1, 0)`` are
        3-D, transposed, ``movedim``-derived views of the operands
        (``[4, 3, 8]`` stride ``[3, 1, 12]`` for X, ``[4, 3, 6]`` stride
        ``[18, 6, 1]`` for Y). The K-advancing reads of those leaves route
        through ``coarse_tile.py``'s read-copy machinery, which builds tile
        buffers whose own layouts (e.g. ``[8, 6, 3]`` stride ``[0, 1, 6]`` --
        a broadcast leading dim over transposed inner dims) then fail stick
        reconciliation in ``optimize_restickify.py``/``propagate_layouts.py``
        ("No mechanism to scatter elements from one stick to multiple
        sticks"). The equivalent HINT-driven K-tiled matmul (same M/K/N,
        ``spyre_hint(num_tiles_per_dim={"K": 4})``) compiles and is
        numerically correct, and needs no read copies at all -- it reads the
        2-D operands directly. So this is a read-copy/stick-layout gap
        surfaced by the 3-D stacked-leaf shape, not a while_loop-lowering
        one, and it needs the same kind of layout work that the map-mode
        carry's own ``[4, 2, 6] -> [8, 6]`` fold needed (see
        ``while_loop_bridge.fold_stacked_carry_layout``) applied to the
        read side.
        """
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_k_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_carry_mode_online_softmax(self):
        """Carry mode: 3-leaf carry (m, denom, acc), online-softmax over K/V tiles.

        Case 4 (multiple independent carries) from the design spec's minimum
        coverage list. carry_bindings_for/splice_while_loop's per-binding loop
        is already generic over an arbitrary-length carry list; this is the
        first fixture that actually drives a 3-leaf init= end to end, both to
        confirm the pytree carry survives decompose_scan_to_while_loop's
        scan -> while_loop decomposition intact, and to confirm
        _extra_readers_of_placeholder/_snapshot_carry_placeholder correctly
        handle the write-after-read hazard this body's own m carry hits:
        `correction = exp(m - m_new)` reads m's OLD value a second time,
        after m_new (m's per-iteration output) has already been computed --
        the exact case an in-place-only rewrite would silently corrupt.
        """
        Q, K, V = attention_inputs()
        ref = online_softmax_reference(Q, K, V)

        Q_spyre = Q.to(DEVICE_NAME)
        K_spyre = K.to(DEVICE_NAME)
        V_spyre = V.to(DEVICE_NAME)

        compiled = torch.compile(online_softmax_fn, backend="inductor", fullgraph=True)
        out = compiled(Q_spyre, K_spyre, V_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )


if __name__ == "__main__":
    unittest.main()
