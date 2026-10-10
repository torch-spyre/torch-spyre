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

"""Helpers in ``scratchpad/utils.py`` that decide LX residency.

``_would_produce_lx_back_gap`` asks whether a device dimension is *fully
covered* by the iteration symbols that walk it: a backGap fires when
``device_size[d]`` exceeds the extent the coordinate actually reaches, and the
backend supports that for HBM but not for LX.

The covered extent is a property of the whole coordinate expression, not of any
one symbol in it. These tests pin that, because the two obvious cheaper
approximations are both wrong and were both live:

* Substituting *one* symbol's iteration range (what this did before) breaks as
  soon as a coordinate folds two symbols into one axis, in both directions --
  see ``test_multi_symbol_*`` for a spurious gap and ``test_stickified_*`` for a
  missed one. It was also read out of the unordered ``free_symbols`` set, so
  which symbol you got moved with ``PYTHONHASHSEED``; that made LX plans, and
  hence core-division plans, differ between processes on identical input.
* Substituting *every* symbol's maximum is right only for a monotone
  coordinate. ``test_mod_coordinate_uses_a_real_bound`` is the counter-example.
"""

import operator
import unittest
from types import SimpleNamespace
from unittest import TestCase, mock

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    ExternKernel,
    FallbackKernel,
    FixedLayout,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
)
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V, ops

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.utils import (
    _would_produce_lx_back_gap,
    get_ncores_for_buffers,
    ops_in_offset_mutation_component,
)

_COORDS = "torch_spyre._inductor.scratchpad.utils.device_coordinates"

_BUF = "buf0"

# Iteration symbols, named as the pre-scheduler names them.
d0, d1, d2, d3 = sympy.symbols("d0 d1 d2 d3", integer=True, nonnegative=True)

# The trailing device coordinate is the within-stick lane, which the check skips
# (a stick is atomic, so it cannot carry a gap). Every fixture here has exactly
# one non-stick device dim, so ``device_size[0]`` is the dim under test.
_STICK_COORD = sympy.Mod(d0, 64)
_STICK_SIZE = 64


class PointwiseInPlaceInputsTest(TestCase):
    def test_lowered_indices_determine_reuse(self):
        from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator

        def same(i, j):
            return 64 * i + j

        def transpose(i, j):
            return i + 64 * j

        cases = [
            ("identity", [("a", same)], ["a"]),
            ("transpose", [("a", transpose)], []),
            ("broadcast", [("a", lambda i, j: j)], []),
            ("offset", [("a", lambda i, j: same(i, j) + 1)], []),
            ("per input", [("a", same), ("b", transpose)], ["a"]),
            ("mixed reads", [("a", same), ("a", transpose)], []),
            ("mixed reads reversed", [("a", transpose), ("a", same)], []),
        ]
        allocator = object.__new__(ScratchpadAllocator)
        for label, reads, expected in cases:
            with self.subTest(label=label):

                def inner_fn(index):
                    values = [
                        ops.load(name, index_fn(*index)) for name, index_fn in reads
                    ]
                    result = values[0]
                    for value in values[1:]:
                        result = ops.add(result, value)
                    return result

                op = ComputedBuffer(
                    name="output",
                    layout=FixedLayout(
                        torch.device("cpu"), torch.float16, [64, 64], [64, 1]
                    ),
                    data=Pointwise(
                        device=torch.device("cpu"),
                        dtype=torch.float16,
                        ranges=[64, 64],
                        inner_fn=inner_fn,
                    ),
                )
                # Both tagged and tag-less origins need real load/store checks.
                for target in (torch.ops.aten.add.Tensor, operator.add):
                    with self.subTest(target=target):
                        op.origin_node = torch.fx.Graph().call_function(target)
                        with V.set_graph_handler(
                            SimpleNamespace(sizevars=SizeVarAllocator())
                        ):
                            self.assertEqual(
                                allocator._op_inputs_good_for_lx_inplace(op), expected
                            )


def _graph(extent, ranges):
    """A one-op graph whose only read of ``_BUF`` carries ``ranges``.

    ``extent`` is ``device_size[0]``, the non-stick device dim under test.
    ``ranges`` maps each iteration symbol to its size, which is what
    ``MemoryDep`` exposes as ``dep.ranges``.
    """
    symbols = tuple(ranges)
    dep = MemoryDep(
        _BUF,
        # The index is unused here: the coordinate under test is supplied by the
        # patched ``device_coordinates``. Kept plausible rather than empty.
        sum(symbols, sympy.Integer(0)),
        symbols,
        tuple(ranges[sym] for sym in symbols),
    )
    op = SimpleNamespace(
        get_read_writes=lambda: SimpleNamespace(reads={dep}, writes=set())
    )
    buf = SimpleNamespace(
        layout=SimpleNamespace(
            device_layout=SimpleNamespace(device_size=[extent, _STICK_SIZE])
        )
    )
    return SimpleNamespace(operations=[op], get_buffer=lambda _name: buf)


def _back_gap(coord, extent, ranges):
    graph = _graph(extent, ranges)
    with mock.patch(_COORDS, return_value=[coord, _STICK_COORD]):
        return _would_produce_lx_back_gap(graph, _BUF, [0])


class BackGapTest(TestCase):
    def test_multi_symbol_coordinate_that_covers_its_dim_has_no_gap(self):
        """``2*d1 + d3`` over ``d1 in [0,40)``, ``d3 in [0,2)`` reaches 0..79.

        The dim is 80 wide, so it is exactly covered and there is no gap. Note
        that *neither* single symbol's range answers this: 80 > 40 and 80 > 2 are
        both true, so picking either one reports a gap that is not there. This
        case fails on any hash seed, which is what makes it a regression test.
        """
        self.assertFalse(_back_gap(2 * d1 + d3, 80, {d1: 40, d3: 2}))

    def test_multi_symbol_stickified_coordinate_has_no_gap(self):
        """``2*d0 + floor(d2/64)`` over ``d0 in [0,8)``, ``d2 in [0,128)``.

        The shape observed on a granite-4.0-micro decoder block, where this
        decided one buffer's LX residency and, through it, the whole
        core-division plan. Reaches 0..15 on a dim 16 wide: no gap. Here the two
        symbols disagree (``d0`` says gap, ``d2`` says none), which is how the
        verdict came to depend on the hash seed.
        """
        self.assertFalse(_back_gap(2 * d0 + sympy.floor(d2 / 64), 16, {d0: 8, d2: 128}))

    def test_uncovered_dim_has_a_gap(self):
        """``d0`` over ``[0,8)`` reaches 0..7, which leaves a dim 16 wide short.

        The positive case: without it the tests above would pass against a
        function that always returned False.
        """
        self.assertTrue(_back_gap(d0, 16, {d0: 8}))

    def test_stickified_coordinate_gap_is_not_hidden_by_the_element_range(self):
        """``floor(d1/64)`` over ``d1 in [0,1024)`` reaches 0..15, not 0..1023.

        A stick-count coordinate covers ``range/64``, so a dim 32 wide really
        does have a gap. Reading ``d1``'s raw *element* range instead (1024)
        overstates coverage by 64x and hides it -- the dangerous direction, since
        LX has no backGap support, so a missed gap is a codegen hazard rather
        than a lost optimization.
        """
        self.assertTrue(_back_gap(sympy.floor(d1 / 64), 32, {d1: 1024}))

    def test_mod_coordinate_uses_a_real_bound(self):
        """``Mod(d0, 5)`` over ``d0 in [0,8)`` reaches 0..4, so a dim 4 wide is
        covered.

        Non-monotone, so substituting the symbol's maximum is not its bound:
        ``Mod(7, 5)`` is 2, which would understate the extent as 3 and report a
        gap. This is why the check needs real range analysis and not ``subs``.
        """
        self.assertFalse(_back_gap(sympy.Mod(d0, 5), 4, {d0: 8}))


class ExternalOperandTest(TestCase):
    def test_external_operands_never_publish_physical_ownership(self):
        graph = SimpleNamespace(try_get_buffer=lambda name: None)
        for op_type in (ExternKernel, FallbackKernel):
            for cores in (1, 32):
                with self.subTest(op=op_type.__name__, cores=cores):
                    op = mock.Mock(spec=op_type)
                    op.get_name.return_value = "opaque"
                    with (
                        config.patch({"sencores": cores}),
                        mock.patch(
                            "torch_spyre._inductor.scratchpad.utils._get_buffer_user_deps",
                            return_value={_BUF: [(op, None)]},
                        ),
                        mock.patch(
                            "torch_spyre._inductor.scratchpad.utils._per_core_view_on_buf"
                        ) as project,
                    ):
                        counts, reasons, views = get_ncores_for_buffers(graph)
                    self.assertEqual(counts, {_BUF: -1})
                    self.assertIn("FallbackKernel/ExternKernel", reasons[_BUF])
                    self.assertEqual(views, {})
                    project.assert_not_called()

    def test_a_drained_collective_is_not_a_user_of_the_carry(self):
        """A drain plan repoints this collective at a post-loop copy, so the
        carry it still reads before the push is judged by its other users."""
        graph = SimpleNamespace(try_get_buffer=lambda name: None)
        writer = mock.Mock(iteration_space_ownership=None)
        writer.get_name.return_value = "writer"
        collective = mock.Mock(spec=ExternKernel)
        collective.get_name.return_value = "all_reduce"
        write = mock.sentinel.write
        view = SimpleNamespace(work_slice_dims=[], same_partition=lambda _: True)
        utils = "torch_spyre._inductor.scratchpad.utils"
        for drained, cores in ((None, -1), ({_BUF: "all_reduce"}, 1)):
            with (
                self.subTest(drained=drained),
                mock.patch(
                    f"{utils}._get_buffer_user_deps",
                    return_value={_BUF: [(writer, write), (collective, None)]},
                ),
                mock.patch(
                    f"{utils}._per_core_view_on_buf", return_value=(view, False, True)
                ),
                mock.patch(
                    f"{utils}.op_read_writes",
                    return_value=SimpleNamespace(writes={write}),
                ),
            ):
                counts, _reasons, views = get_ncores_for_buffers(
                    graph, drained_readers=drained
                )
                self.assertEqual(counts, {_BUF: cores})
                self.assertEqual(_BUF in views, drained is not None)


def _pin_dep(name, offset=0):
    """A (4, 64) access of ``name``, starting ``offset`` elements in."""
    return MemoryDep(name, 64 * d0 + d1 + offset, (d0, d1), (4, 64))


def _pin_op(name, reads=(), target=None, write_offset=0):
    """An op writing ``name`` (or, through a MutationLayout, ``target``)."""
    layout = None
    if target is not None:
        layout = mock.MagicMock(spec=MutationLayoutSHOULDREMOVE)
        layout.target = mock.MagicMock()
        layout.target.get_name.return_value = target
    rw = SimpleNamespace(
        reads=[_pin_dep(r) for r in reads], writes=[_pin_dep(name, write_offset)]
    )
    return SimpleNamespace(name=name, layout=layout, rw=rw)


def _pinned(ops):
    graph = SimpleNamespace(operations=ops)
    with mock.patch(
        "torch_spyre._inductor.scratchpad.utils.op_read_writes",
        side_effect=lambda op: op.rw,
    ):
        return ops_in_offset_mutation_component(graph)


class OffsetMutationPinTest(TestCase):
    """Which ops ``ops_in_offset_mutation_component`` pins around an offset write."""

    def test_reader_chain_has_no_hop_limit(self):
        for depth in (3, 32):
            with self.subTest(depth=depth):
                ops = [
                    _pin_op("producer", reads=["arg0"]),
                    _pin_op("clone", reads=["arg1"]),
                    _pin_op(
                        "write", reads=["producer"], target="clone", write_offset=32
                    ),
                ]
                parent = "clone"
                for i in range(depth):
                    name = f"hop{i}"
                    ops.append(_pin_op(name, reads=[parent]))
                    parent = name
                expected = {"clone", "write"} | {f"hop{i}" for i in range(depth)}
                self.assertEqual(_pinned(ops), expected)
                self.assertEqual(_pinned(list(reversed(ops))), expected)

    def test_transitive_zero_offset_writer_reaches_target_and_readers(self):
        ops = [
            _pin_op("clone", reads=["src"]),
            _pin_op("write", reads=["value"], target="clone", write_offset=32),
            _pin_op("hop1", reads=["clone"]),
            _pin_op("hop2", reads=["hop1"]),
            _pin_op("hop3", reads=["hop2"]),
            _pin_op("copy_back", reads=["hop3"], target="arg0"),
            _pin_op("reader", reads=["arg0"]),
            _pin_op("second_copy", reads=["reader"], target="arg1"),
            _pin_op("tail", reads=["arg1"]),
            _pin_op("unrelated", reads=["value"]),
        ]
        expected = {op.name for op in ops} - {"unrelated"}
        self.assertEqual(_pinned(ops), expected)
        self.assertEqual(_pinned(list(reversed(ops))), expected)

    def test_alias_cycle_terminates(self):
        ops = [
            _pin_op("write", reads=["arg0"], target="clone", write_offset=32),
            _pin_op("clone", reads=["write"]),
            _pin_op("copy_back", reads=["clone"], target="arg0"),
            _pin_op("reader", reads=["arg0"]),
        ]
        self.assertEqual(_pinned(ops), {op.name for op in ops})

    def test_symbolic_tile_offset_does_not_seed_pin(self):
        tile = sympy.Symbol("tile", integer=True, nonnegative=True)
        ops = [
            _pin_op("storage"),
            _pin_op("write", target="storage", write_offset=64 * tile),
            _pin_op("reader", reads=["storage"]),
        ]
        self.assertEqual(_pinned(ops), set())

    def test_no_offset_write_pins_nothing(self):
        ops = [_pin_op("a", reads=["arg0"]), _pin_op("b", reads=["a"])]
        self.assertEqual(_pinned(ops), set())

    def test_pad_before_ffn_does_not_pin_upstream_attention(self):
        # attention -> o_proj -> norm -> pad (fill + two mutations, one at an
        # offset) -> gate_up -> silu -> down -> residual: the shape of issue #4990.
        ops = [
            _pin_op("attn", reads=["q", "k_pages", "v_pages"]),
            _pin_op("o_proj", reads=["attn", "w_o"]),
            _pin_op("norm", reads=["o_proj"]),
            _pin_op("pad_fill"),
            _pin_op("pad_rows", reads=["norm"], target="pad_fill"),
            _pin_op("pad_zeros", target="pad_fill", write_offset=256),
            _pin_op("gate_up", reads=["pad_fill", "w_gu"]),
            _pin_op("silu", reads=["gate_up"]),
            _pin_op("down", reads=["silu", "w_d"]),
            _pin_op("residual", reads=["o_proj", "down"]),
        ]
        self.assertEqual(
            _pinned(ops),
            {
                "pad_fill",
                "pad_rows",
                "pad_zeros",
                "gate_up",
                "silu",
                "down",
                "residual",
            },
        )

    def test_copy_back_into_graph_input_pins_its_readers(self):
        # z = x.copy_(y)._base; (z + 1) + amax(z), with z a mutated graph input:
        # the offset write lands in a clone that is then copied back into arg0.
        ops = [
            _pin_op("y_slice", reads=["arg1"]),
            _pin_op("y_restick", reads=["y_slice"]),
            _pin_op("clone", reads=["arg0"]),
            _pin_op(
                "slice_write", reads=["y_restick"], target="clone", write_offset=32
            ),
            _pin_op("copy_back", reads=["clone"], target="arg0"),
            _pin_op("add_one", reads=["arg0", "one"]),
            _pin_op("amax", reads=["arg0"]),
            _pin_op("add", reads=["add_one", "amax"]),
        ]
        self.assertEqual(
            _pinned(ops),
            {"clone", "slice_write", "copy_back", "add_one", "amax", "add"},
        )


if __name__ == "__main__":
    unittest.main()
