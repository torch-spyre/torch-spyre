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

"""The per-launch symbolic argument payload, built from the bundle's symbols.

``generate_bundle`` returns the canonical symbol order and the runner turns it
into one ``SymbolicArg`` per MLIR parameter slot. The payload is built once and
is invariant across launches: a dimension slot names a tensor and a dim rather
than carrying a size, because the size changes on every call.

Order and kind both matter and they fail differently. A wrong count is a loud
check failure on the C++ side. A wrong order with the right count is silently
wrong numerics, so these tests assert positions and not just membership.

No device: the runner's constructor touches no hardware when it is given no
provenance descriptor, and the job plan is built lazily on first use.
"""

import unittest

from torch_spyre._C import SymbolicArgKind
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

SYM = "s0"
TILE = 64
MAX = 512


def _loop_dim(arg_index=0, dim_index=0):
    return SymbolKind.loop_dimension(
        granularity=TILE,
        max_value=MAX,
        pytorch_sym=SYM,
        arg_index=arg_index,
        dim_index=dim_index,
    )


def _payload(symbol_kinds):
    runner = SpyreSDSCKernelRunner("test", "/nonexistent", symbol_kinds=symbol_kinds)
    return runner._symbolic_args


def _described(payload):
    """(kind name, tensor_id, dim_index) per slot, for readable assertions."""
    return [(a.kind.name, a.tensor_id, a.dim_index) for a in payload]


class TestTheAddressOnlyPayloadIsUnchanged(unittest.TestCase):
    """The regression half. Every kernel today has only address symbols."""

    def test_without_a_pool_arg_index_maps_straight_through(self):
        payload = _payload([SymbolKind.kernel(0), SymbolKind.kernel(1)])

        self.assertEqual(
            _described(payload), [("kAddress", 0, -1), ("kAddress", 1, -1)]
        )

    def test_with_a_pool_every_kernel_arg_shifts_by_one(self):
        """call_kernel prepends the pool tensor, so it occupies args[0]."""
        payload = _payload(
            [SymbolKind.pool(), SymbolKind.kernel(0), SymbolKind.kernel(1)]
        )

        self.assertEqual(
            _described(payload),
            [("kAddress", 0, -1), ("kAddress", 1, -1), ("kAddress", 2, -1)],
        )

    def test_no_symbols_means_no_payload(self):
        self.assertIsNone(_payload([]))
        self.assertIsNone(_payload(None))


class TestADimensionSlotNamesATensorAndADim(unittest.TestCase):
    def test_a_loop_dimension_becomes_a_dimension_slot(self):
        payload = _payload([SymbolKind.kernel(0), _loop_dim(arg_index=0)])

        self.assertEqual(
            _described(payload), [("kAddress", 0, -1), ("kDimension", 0, 0)]
        )

    def test_it_carries_no_size(self):
        """The size is read at launch, so the descriptor must not hold one.

        A value baked in here would be one call's size on every call, which is
        the failure this whole design exists to avoid.
        """
        (_addr, dim) = _payload([SymbolKind.kernel(0), _loop_dim()])

        self.assertEqual(dim.value, -1)

    def test_the_dim_index_is_carried_through(self):
        payload = _payload([SymbolKind.kernel(0), _loop_dim(arg_index=0, dim_index=2)])

        self.assertEqual(payload[1].dim_index, 2)

    def test_a_pool_shifts_the_dimension_slot_too(self):
        """The offset applies to every kernel-indexed slot, not only addresses.

        Getting this wrong would read the size off the wrong tensor, which is a
        plausible number and therefore silently wrong on every launch.
        """
        payload = _payload(
            [SymbolKind.pool(), SymbolKind.kernel(0), _loop_dim(arg_index=0)]
        )

        self.assertEqual(
            _described(payload),
            [("kAddress", 0, -1), ("kAddress", 1, -1), ("kDimension", 1, 0)],
        )

    def test_the_dimension_keeps_its_position_in_the_slot_order(self):
        """The payload is consumed positionally against the MLIR parameters,
        and the bundle emits dimension parameters last."""
        payload = _payload(
            [SymbolKind.kernel(0), SymbolKind.kernel(1), _loop_dim(arg_index=1)]
        )

        self.assertEqual(
            [a.kind.name for a in payload], ["kAddress"] * 2 + ["kDimension"]
        )

    def test_two_dimensions_each_get_their_own_slot(self):
        payload = _payload(
            [
                SymbolKind.kernel(0),
                _loop_dim(arg_index=0, dim_index=0),
                _loop_dim(arg_index=0, dim_index=1),
            ]
        )

        self.assertEqual(
            _described(payload),
            [("kAddress", 0, -1), ("kDimension", 0, 0), ("kDimension", 0, 1)],
        )


class TestTheKindsAreDistinguishable(unittest.TestCase):
    def test_an_sdsc_dimension_symbol_is_not_a_dimension_slot(self):
        """It never reaches here: async_compile refuses that route earlier.

        If it ever did, treating it as an address is the conservative outcome,
        because the old route's runtime payload was never built. This pins that
        the two kinds stay distinguishable at this layer.
        """
        payload = _payload([SymbolKind.dimension(TILE, MAX, SYM)])

        self.assertEqual(payload[0].kind, SymbolicArgKind.kAddress)


if __name__ == "__main__":
    unittest.main()
