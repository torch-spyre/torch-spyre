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

"""Device-free unit tests for the indexed-selection consumer layout candidate.

These exercise ``propagate_layouts._offer_indexed_selection_layouts`` and its
helpers on hand-built IR facts (real Inductor ``MemoryDep``/``FixedLayout``,
real ``SpyreTensorLayout``), not on a compiled program. They prove the
decision logic and the candidate geometry; they do not prove the backend
accepts the layout or that an emitted program reads it -- that is the device
harness's job (``perf-gemma-bench/inspect_decode_program.py``).

The shapes are the decode reproducer: bank ``[E, K, N]`` selected into
``[R, K, N]`` and consumed by ``torch.bmm(activation[R, 1, K], selected)``.
"""

import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, FixedLayout, Reduction
from torch._inductor.virtualized import V

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor import propagate_layouts as pl
from torch_spyre._inductor import optimize_restickify as opt
from torch_spyre._inductor.constants import BATCH_MATMUL_FP8_OP, BATCH_MATMUL_OP

d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True)
tmp0 = sympy.Symbol("tmp0", integer=True)

R, K, N, E = 8, 2816, 704, 128
DEVICE = torch.device("cpu")


class _FakeReduction(Reduction):
    """A Reduction-typed stand-in carrying only the attributes the helpers read."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _FakeOp(ComputedBuffer):
    """A ComputedBuffer-typed stand-in exposing name, reads/writes and data."""

    def __init__(self, name, reads, writes, data=None):
        self._name = name
        self._reads = reads
        self._writes = writes
        self.data = data if data is not None else SimpleNamespace()

    def get_name(self):
        return self._name

    def get_read_writes(self):
        return SimpleNamespace(reads=self._reads, writes=self._writes)


def _selection(size=(R, K, N)):
    r, k, n = size
    layout = FixedLayout(DEVICE, torch.float16, [r, k, n], [k * n, n, 1])
    out_dep = MemoryDep("sel", d0 * k * n + d1 * n + d2, (d0, d1, d2), (r, k, n))
    bank_dep = MemoryDep("bank", tmp0 * k * n + d1 * n + d2, (d0, d1, d2), (r, k, n))
    ids_dep = MemoryDep("ids", d0, (d0,), (r,))
    op = _FakeOp("sel", [bank_dep, ids_dep], [out_dep])
    return op, layout, out_dep


def _bmm_consumer(m_size=1):
    """``torch.bmm(activation[R, m, K], selected[R, K, N])`` as IR deps.

    Output deps carry only output iteration variables with N last (Inductor's
    matmul lowering); the reduction variable appears on the inputs only.
    """
    if m_size == 1:
        out_dep = MemoryDep("out", d0 * N + d1, (d0, d1), (R, N))
        x_dep = MemoryDep("act", d0 * K + d2, (d0, d1, d2), (R, N, K))
        y_dep = MemoryDep("sel", d0 * K * N + d2 * N + d1, (d0, d1, d2), (R, N, K))
        ranges = [R, 1, N]
    else:
        d3 = sympy.Symbol("d3", integer=True)
        out_dep = MemoryDep(
            "out", d0 * m_size * N + d1 * N + d2, (d0, d1, d2), (R, m_size, N)
        )
        x_dep = MemoryDep(
            "act", d0 * m_size * K + d1 * K + d3, (d0, d1, d2, d3), (R, m_size, N, K)
        )
        y_dep = MemoryDep(
            "sel", d0 * K * N + d3 * N + d2, (d0, d1, d2, d3), (R, m_size, N, K)
        )
        ranges = [R, m_size, N]
    data = _FakeReduction(reduction_type=BATCH_MATMUL_OP, ranges=ranges)
    return _FakeOp("out", [x_dep, y_dep], [out_dep], data), x_dep, y_dep


def _graph(operations, layout, outputs=("out",), mutated=()):
    return SimpleNamespace(
        operations=list(operations),
        get_output_names=lambda: list(outputs),
        mutated_buffers=set(mutated),
        get_buffer=lambda name: SimpleNamespace(get_layout=lambda: layout),
    )


def _generic(size=(R, K, N)):
    r, k, n = size
    return SpyreTensorLayout([r, k, n], [k * n, n, 1], torch.float16, [0, 1, 2])


def _key(stl):
    return (tuple(stl.device_size), tuple(stl.stride_map))


class TestEntryContiguousGeometry(unittest.TestCase):
    def test_decode_repro_arrangement(self):
        """[R,K,N] with stick N: entry outermost, stick kept as coarse/fine."""
        stl = pl._entry_contiguous_stl([R, K, N], [K * N, N, 1], 0, 2, torch.float16)
        self.assertEqual(list(stl.device_size), [R, K, N // 64, 64])
        self.assertEqual(list(stl.stride_map), [K * N, N, 64, 1])
        self.assertEqual(stl.element_arrangement, ElementArrangement.STANDARD)

    def test_default_constructor_interleaves_entries(self):
        """The default layout rotates the outermost dim to just before the stick.

        This is the interleaved arrangement the candidate exists to avoid; the
        two layouts hold identical values and differ only in dim order.
        """
        generic = _generic()
        self.assertEqual(list(generic.device_size), [K, N // 64, R, 64])
        self.assertEqual(list(generic.stride_map), [N, 64, K * N, 1])
        candidate = pl._entry_contiguous_stl(
            [R, K, N], [K * N, N, 1], 0, 2, torch.float16
        )
        self.assertNotEqual(candidate, generic)
        self.assertEqual(sorted(candidate.device_size), sorted(generic.device_size))

    def test_unaligned_stick_and_size_one_dims(self):
        """A partial stick rounds up; size-one / broadcast dims collapse to -1."""
        stl = pl._entry_contiguous_stl([4, 1, 100], [100, 0, 1], 0, 2, torch.float16)
        self.assertEqual(list(stl.device_size), [4, 1, 2, 64])
        self.assertEqual(list(stl.stride_map), [100, -1, 64, 1])

    def test_entry_dim_not_first_is_moved_outermost(self):
        stl = pl._entry_contiguous_stl([K, R, N], [R * N, N, 1], 1, 2, torch.float16)
        self.assertEqual(list(stl.device_size), [R, K, N // 64, 64])
        self.assertEqual(list(stl.stride_map), [N, R * N, 64, 1])


class TestEntryDimDerivation(unittest.TestCase):
    def test_same_free_symbol_does_not_prove_coordinate_identity(self):
        op, layout, out_dep = _selection()
        for coordinate in (2 * d0, d0 + 3, sympy.floor(d0 / 2)):
            with (
                self.subTest(coordinate=coordinate),
                patch.object(pl, "host_coordinates", return_value=[coordinate, d1, d2]),
            ):
                self.assertIsNone(
                    pl._indexed_selection_entry_dim(
                        op, layout, out_dep, {"ids"}, {tmp0: E}
                    )
                )

    def test_entry_dim_is_the_index_read_variable(self):
        op, layout, out_dep = _selection()
        self.assertEqual(
            pl._indexed_selection_entry_dim(op, layout, out_dep, {"ids"}, {tmp0: E}),
            0,
        )

    def test_index_read_with_two_variables_declines(self):
        op, layout, out_dep = _selection()
        op._reads[1] = MemoryDep("ids", d0 * K + d1, (d0, d1), (R, K))
        self.assertIsNone(
            pl._indexed_selection_entry_dim(op, layout, out_dep, {"ids"}, {tmp0: E})
        )

    def test_entry_on_a_middle_dim(self):
        layout = FixedLayout(DEVICE, torch.float16, [K, R, N], [R * N, N, 1])
        out_dep = MemoryDep("sel", d0 * R * N + d1 * N + d2, (d0, d1, d2), (K, R, N))
        ids_dep = MemoryDep("ids", d1, (d1,), (R,))
        op = _FakeOp("sel", [ids_dep], [out_dep])
        self.assertEqual(
            pl._indexed_selection_entry_dim(op, layout, out_dep, {"ids"}, {tmp0: E}),
            1,
        )


class TestMatmulRoleRule(unittest.TestCase):
    def test_default_matmul_branches_request_the_same_input_roles(self):
        """Drive both existing layout branches up to the required y layout."""

        class ReachedY(Exception):
            pass

        for m_size in (1, 4):
            bmm, x_dep, y_dep = _bmm_consumer(m_size=m_size)
            out_dep = bmm.get_read_writes().writes[0]
            output = FixedLayout(
                DEVICE, torch.float16, [R, m_size, N], [m_size * N, N, 1]
            )
            args = [
                pl.PropArg(x_dep, output, []),
                pl.PropArg(y_dep, output, []),
            ]
            requested = []

            def required_layout(arg, variable, op_type, role):
                requested.append((role, variable))
                if role == "y":
                    raise ReachedY
                return None

            with (
                self.subTest(m_size=m_size),
                config.patch({"indexed_selection_consumer_layout": False}),
                patch.object(pl, "_check_supported_input_sticks"),
                patch.object(pl.logger, "isEnabledFor", return_value=False),
                patch.object(pl, "host_coordinates", return_value=[d0, d1, d2]),
                patch.object(
                    pl,
                    "find_stick_compatible_input_layout",
                    side_effect=required_layout,
                ),
                self.assertRaises(ReachedY),
            ):
                pl._matmul_layouts(bmm, output, out_dep, args)
            reduction = d2 if m_size == 1 else sympy.Symbol("d3", integer=True)
            generated = d1 if m_size == 1 else d2
            self.assertEqual(requested, [("x", reduction), ("y", generated)])

    def test_m_equals_one_uses_shared_matmul_rule(self):
        """The one-row decode BMM: y needs N (output dep's last var), x needs K."""
        bmm, x_dep, y_dep = _bmm_consumer(m_size=1)
        self.assertEqual(pl._batch_matmul_required_stick_var(bmm, y_dep), d1)
        self.assertEqual(pl._batch_matmul_required_stick_var(bmm, x_dep), d2)

    def test_m_greater_than_one(self):
        bmm, x_dep, y_dep = _bmm_consumer(m_size=4)
        d3 = sympy.Symbol("d3", integer=True)
        self.assertEqual(pl._batch_matmul_required_stick_var(bmm, y_dep), d2)
        self.assertEqual(pl._batch_matmul_required_stick_var(bmm, x_dep), d3)

    def test_n_equals_one_declines(self):
        """``_matmul_layouts`` gives y a sparse stick when N == 1; no role to prove."""
        bmm, _x_dep, y_dep = _bmm_consumer(m_size=1)
        narrow = _FakeOp(
            "out",
            bmm.get_read_writes().reads,
            bmm.get_read_writes().writes,
            _FakeReduction(reduction_type=BATCH_MATMUL_OP, ranges=[R, 1, 1]),
        )
        self.assertIsNone(pl._batch_matmul_required_stick_var(narrow, y_dep))

    def test_matmul_layouts_and_consumer_proof_share_the_rule(self):
        """``_matmul_generated_var`` is the single source for both callers."""
        bmm, x_dep, y_dep = _bmm_consumer(m_size=1)
        out_dep = bmm.get_read_writes().writes[0]
        self.assertEqual(pl._matmul_generated_var(bmm, x_dep, y_dep, out_dep, 1), d1)
        empty = MemoryDep("out", sympy.Integer(0), (), ())
        with self.assertRaises(pl.Unsupported):
            pl._matmul_generated_var(bmm, x_dep, y_dep, empty, 1)


class TestOfferIndexedSelectionLayouts(unittest.TestCase):
    def setUp(self):
        self._saved = config.indexed_selection_consumer_layout
        self.op, self.layout, self.out_dep = _selection()
        self.bmm, self.x_dep, self.y_dep = _bmm_consumer(m_size=1)
        self.generic = _generic()

    def tearDown(self):
        config.indexed_selection_consumer_layout = self._saved

    def _offer(self, graph, flag=True, results=None):
        config.indexed_selection_consumer_layout = flag
        with V.set_graph_handler(graph):
            return pl._offer_indexed_selection_layouts(
                self.op,
                self.layout,
                self.out_dep,
                list(results if results is not None else [self.generic]),
                {"ids"},
                {tmp0: E},
                torch.float16,
                ElementArrangement.STANDARD,
            )

    def test_opt_out_leaves_candidates_untouched(self):
        graph = _graph([self.op, self.bmm], self.layout)
        self.assertEqual(self._offer(graph, flag=False), [self.generic])

    def test_m_equals_one_appends_candidate_without_replacing_ordinary_choice(self):
        graph = _graph([self.op, self.bmm], self.layout)
        results = self._offer(graph)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], self.generic)
        self.assertEqual(list(results[1].device_size), [R, K, N // 64, 64])
        self.assertEqual(list(results[1].stride_map), [K * N, N, 64, 1])

    def test_m_greater_than_one_consumer_also_accepts(self):
        self.bmm, self.x_dep, self.y_dep = _bmm_consumer(m_size=4)
        graph = _graph([self.op, self.bmm], self.layout)
        results = self._offer(graph)
        self.assertEqual(list(results[-1].device_size), [R, K, N // 64, 64])

    def test_selection_read_as_matmul_input1_declines(self):
        """Input1 (x) needs K on the stick; an N-stick candidate is incompatible."""
        bmm, _x, _y = _bmm_consumer(m_size=4)
        # Rewire: the selection is read where the activation was.
        reads = bmm.get_read_writes().reads
        x_as_sel = MemoryDep("sel", reads[0].index, reads[0].var_names, reads[0].size)
        other = MemoryDep("act", reads[1].index, reads[1].var_names, reads[1].size)
        bmm._reads = [x_as_sel, other]
        graph = _graph([self.op, bmm], self.layout)
        self.assertEqual(self._offer(graph), [self.generic])

    def test_no_consumer_declines(self):
        graph = _graph([self.op], self.layout)
        self.assertEqual(self._offer(graph), [self.generic])

    def test_non_batch_matmul_consumer_declines(self):
        pointwise = _FakeOp(
            "pw", [MemoryDep("sel", self.out_dep.index, (d0, d1, d2), (R, K, N))], []
        )
        graph = _graph([self.op, self.bmm, pointwise], self.layout)
        self.assertEqual(self._offer(graph), [self.generic])

    def test_non_computed_buffer_reader_declines(self):
        """An extern/fallback-style reader is not a ComputedBuffer and must count."""
        extern = SimpleNamespace(
            get_read_writes=lambda: SimpleNamespace(
                reads=[MemoryDep("sel", d0, (d0,), (R,))], writes=[]
            )
        )
        graph = _graph([self.op, self.bmm, extern], self.layout)
        self.assertEqual(self._offer(graph), [self.generic])

    def test_whole_buffer_star_dep_reader_declines(self):
        star = SimpleNamespace(
            get_read_writes=lambda: SimpleNamespace(
                reads=[SimpleNamespace(name="sel")], writes=[]
            )
        )
        graph = _graph([self.op, self.bmm, star], self.layout)
        self.assertEqual(self._offer(graph), [self.generic])

    def test_graph_output_escape_declines(self):
        graph = _graph([self.op, self.bmm], self.layout, outputs=("out", "sel"))
        self.assertEqual(self._offer(graph), [self.generic])

    def test_mutated_buffer_declines(self):
        graph = _graph([self.op, self.bmm], self.layout, mutated=("sel",))
        self.assertEqual(self._offer(graph), [self.generic])

    def test_missing_or_unknown_mutation_metadata_declines(self):
        for missing in (True, False):
            graph = _graph([self.op, self.bmm], self.layout)
            if missing:
                del graph.mutated_buffers
            else:
                graph.mutated_buffers = None
            with self.subTest(missing=missing):
                self.assertEqual(self._offer(graph), [self.generic])

    def test_fp8_consumer_stays_on_ordinary_candidates(self):
        # Reduction is frozen in the native-backed test environment.
        object.__setattr__(self.bmm.data, "reduction_type", BATCH_MATMUL_FP8_OP)
        graph = _graph([self.op, self.bmm], self.layout)
        self.assertEqual(self._offer(graph), [self.generic])

    def test_reader_after_the_matmul_still_counts(self):
        later = SimpleNamespace(
            get_read_writes=lambda: SimpleNamespace(
                reads=[SimpleNamespace(name="sel")], writes=[]
            )
        )
        graph = _graph([self.op, self.bmm, later], self.layout)
        self.assertEqual(self._offer(graph), [self.generic])

    def test_multiple_accepting_consumers_offer_once(self):
        bmm2, _x, _y = _bmm_consumer(m_size=1)
        bmm2._name = "out2"
        graph = _graph([self.op, self.bmm, bmm2], self.layout)
        results = self._offer(graph)
        self.assertEqual(len(results), 2)
        self.assertEqual(len({_key(r) for r in results}), 2)

    def test_non_standard_arrangement_declines(self):
        graph = _graph([self.op, self.bmm], self.layout)
        config.indexed_selection_consumer_layout = True
        with V.set_graph_handler(graph):
            results = pl._offer_indexed_selection_layouts(
                self.op,
                self.layout,
                self.out_dep,
                [self.generic],
                {"ids"},
                {tmp0: E},
                torch.float16,
                ElementArrangement.QFP8WT,
            )
        self.assertEqual(results, [self.generic])

    def test_not_an_indexed_selection_declines(self):
        graph = _graph([self.op, self.bmm], self.layout)
        config.indexed_selection_consumer_layout = True
        with V.set_graph_handler(graph):
            results = pl._offer_indexed_selection_layouts(
                self.op,
                self.layout,
                self.out_dep,
                [self.generic],
                set(),
                None,
                torch.float16,
                ElementArrangement.STANDARD,
            )
        self.assertEqual(results, [self.generic])

    def test_entry_dim_equal_to_stick_declines(self):
        """Selecting along the stick dim has no separate entry slab to make contiguous."""
        layout = FixedLayout(
            DEVICE, torch.float16, [K, N, R * 64], [N * R * 64, R * 64, 1]
        )
        out_dep = MemoryDep(
            "sel", d0 * N * R * 64 + d1 * R * 64 + d2, (d0, d1, d2), (K, N, R * 64)
        )
        ids_dep = MemoryDep("ids", d2, (d2,), (R * 64,))
        op = _FakeOp("sel", [ids_dep], [out_dep])
        self.op, self.layout, self.out_dep = op, layout, out_dep
        generic = SpyreTensorLayout(
            [K, N, R * 64], [N * R * 64, R * 64, 1], torch.float16, [0, 1, 2]
        )
        graph = _graph([op, self.bmm], layout)
        self.assertEqual(self._offer(graph, results=[generic]), [generic])


class _ChoiceLayout(NamedTuple):
    name: str
    device_size: tuple = (1,)
    stride_map: tuple = (1,)


class _TableEdge:
    """Controlled conversion costs, not a replacement physical-layout model."""

    def __init__(self, name, inputs, targets, costs):
        self.dep = SimpleNamespace(name=name)
        self._in_layouts = inputs
        self._target_layouts = targets
        self.costs = costs

    def cost(self, source, target):
        return self.costs.get((source, target), float("inf"))

    def layout(self, source, target):
        cost = self.cost(source, target)
        return (
            None
            if cost == 0
            else opt.EdgeCostMap.INFEASIBLE
            if cost == float("inf")
            else target
        )


class TestInputLayoutChoiceCosts(unittest.TestCase):
    def setUp(self):
        self.ordinary, self.extra, self.output = map(
            _ChoiceLayout, ("ordinary", "extra", "output")
        )

    def node(self, costs):
        edge = _TableEdge(
            "selection", [self.ordinary, self.extra], [self.ordinary, self.extra], costs
        )
        return opt.FixedInOutNode([edge], self.output, [self.ordinary]), edge

    def test_conversion_cost_and_final_target_use_the_same_choice(self):
        node, edge = self.node(
            {(self.extra, self.ordinary): 128, (self.extra, self.extra): 0}
        )
        self.assertEqual(node.cost([self.extra], self.output), 0)
        self.assertEqual(node.min_input_cost("selection", self.extra, self.output), 0)
        final_edge, default = node.required_input_stls(self.output)[0]
        target = node.select_input_stl(final_edge, self.extra, default)
        self.assertEqual(target, self.extra)
        self.assertIsNone(edge.layout(self.extra, target))

    def test_ordinary_target_wins_tie_independent_of_target_list_order(self):
        node, edge = self.node(
            {(self.extra, self.ordinary): 0, (self.extra, self.extra): 0}
        )
        for targets in ([self.extra, self.ordinary], [self.ordinary, self.extra]):
            edge._target_layouts = targets
            self.assertEqual(
                node.select_input_stl(edge, self.extra, self.ordinary), self.ordinary
            )

    def test_feasible_conversion_is_retained_and_missing_path_is_infeasible(self):
        node, edge = self.node({(self.extra, self.ordinary): 32})
        self.assertEqual(node.cost([self.extra], self.output), 32)
        target = node.select_input_stl(edge, self.extra, self.ordinary)
        self.assertEqual(edge.layout(self.extra, target), self.ordinary)
        self.assertEqual(node.cost([self.ordinary], self.output), float("inf"))
        self.assertEqual(
            node.min_input_cost("selection", self.ordinary, self.output), float("inf")
        )
        self.assertEqual(node.cost([self.extra], self.extra), float("inf"))

    def test_other_input_with_no_legal_layout_keeps_backward_cost_infinite(self):
        node, edge = self.node({(self.extra, self.extra): 0})
        blocked = _TableEdge("other", [self.ordinary], [self.ordinary], {})
        node = opt.FixedInOutNode(
            [edge, blocked], self.output, [self.ordinary, self.ordinary]
        )
        self.assertEqual(
            node.min_input_cost("selection", self.extra, self.output), float("inf")
        )

    def test_duplicate_reads_each_pay_their_own_conversion_cost(self):
        node, edge = self.node(
            {(self.extra, self.ordinary): 4, (self.extra, self.extra): 8}
        )
        node = opt.FixedInOutNode(
            [edge, edge], self.output, [self.ordinary, self.ordinary]
        )
        self.assertEqual(node.cost([self.extra, self.extra], self.output), 8)
        self.assertEqual(node.min_input_cost("selection", self.extra, self.output), 8)

    def test_single_target_retains_existing_requirement(self):
        node, edge = self.node(
            {(self.extra, self.ordinary): 4, (self.extra, self.extra): 0}
        )
        edge._target_layouts = [self.ordinary]
        self.assertEqual(node.cost([self.extra], self.output), 4)
        self.assertEqual(
            node.select_input_stl(edge, self.extra, self.ordinary), self.ordinary
        )

    def test_existing_beam_can_choose_either_producer_layout_and_keeps_ordinary_ties(
        self,
    ):
        for ordinary_cost, extra_cost, expected in (
            (0, 8, self.ordinary),
            (8, 0, self.extra),
            (0, 0, self.ordinary),
        ):
            with self.subTest(ordinary_cost=ordinary_cost, extra_cost=extra_cost):
                node, _ = self.node(
                    {
                        (self.ordinary, self.ordinary): ordinary_cost,
                        (self.extra, self.extra): extra_cost,
                    }
                )
                source = SimpleNamespace(
                    get_name=lambda: "selection",
                    layouts=[self.ordinary, self.extra],
                    restick_cost_fn=opt.AnyInNode.from_args(),
                )
                consumer = SimpleNamespace(
                    get_name=lambda: "matmul",
                    layouts=[self.output],
                    restick_cost_fn=node,
                )
                graph = SimpleNamespace(
                    graph_input_names=[],
                    get_buffer={"selection": source, "matmul": consumer}.__getitem__,
                )
                with (
                    V.set_graph_handler(graph),
                    patch.object(opt.logger, "isEnabledFor", return_value=False),
                ):
                    opt.beam_global_min_cost([source, consumer])
                self.assertEqual(source.committed_stl, expected)


if __name__ == "__main__":
    unittest.main()
