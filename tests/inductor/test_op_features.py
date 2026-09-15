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

"""The captured per-division ``OpFeatures`` fixture, and what it buys.

The point of the fixture is that the co-optimizer's memory-only objective scores
a division change as *exactly free* unless it moves a buffer in or out of LX --
which is why several captures sit at that objective's floor and can distinguish
no move set, schedule or capacity at all. These tests assert the fixture is
well-formed and that it actually separates those cases, so a regression in the
extractor (or a schema drift in the cost model) is caught here rather
than as a silently flat search landscape.

Regenerate with ``python3 docs/source/user_guide/examples/scratchpad/capture_op_features.py`` on a Spyre machine.
"""

import dataclasses
import json
import math
import os
import unittest
from unittest import TestCase
from unittest.mock import patch

import sympy

from torch_spyre._inductor.cost_model import (
    ArgTraffic,
    OpFeatures,
    _loop_reread_bytes,
    op_from_dict,
    predict_by_bundle,
    predict_ops,
)
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision
from torch_spyre._inductor.scratchpad.sa_cooptimizer import _work_slices

FIXTURE = os.path.join(os.path.dirname(__file__), "cooptimization_op_features.json")


def _graphs():
    with open(FIXTURE) as fh:
        return json.load(fh)["graphs"]


def _entries():
    """(graph, buffer, buffer_dict) for every captured buffer."""
    for gname, g in _graphs().items():
        for bname, b in g["buffers"].items():
            yield gname, bname, b


class CandidateDivisionTest(TestCase):
    def test_candidate_is_restored_as_complete_symbol_keyed_map(self):
        """SA feature extraction neither encodes nor mutates Scheduler transport."""
        m, n, kk = sympy.symbols("m n kk")
        op = object()
        division = CoreDivision(output_splits={m: 8}, reduction_splits={kk: 2})
        expected = {m: 8, n: 1, kk: 2}
        with patch(
            "torch_spyre._inductor.scratchpad.sa_cooptimizer.iteration_space_from_op",
            return_value={m: 1024, n: 1024, kk: 2048},
        ):
            self.assertEqual(_work_slices(op, division), expected)


class FixturePresentTest(TestCase):
    def test_fixture_covers_several_graphs(self):
        graphs = _graphs()
        self.assertGreaterEqual(len(graphs), 3, "fixture lost graphs")
        for gname, g in graphs.items():
            self.assertTrue(g["buffers"], f"{gname} captured no buffers")


class SchemaTest(TestCase):
    def test_features_are_index_aligned_with_the_division_menu(self):
        # A menu index must select its own features: the co-optimizer indexes
        # core_divisions by the same integer, so a length mismatch would silently
        # score the wrong division.
        for gname, bname, b in _entries():
            tag = f"{gname}/{bname}"
            self.assertEqual(len(b["features"]), b["menu_size"], tag)
            self.assertEqual(len(b["output_partitions"]), b["menu_size"], tag)

    def test_every_feature_round_trips_and_scores(self):
        # Guards against schema drift in the cost model: a renamed or
        # newly-required field would break op_from_dict, and the branch's own
        # committed dataset already went stale that way once.
        n = 0
        for gname, bname, b in _entries():
            for i, raw in enumerate(b["features"]):
                if raw is None:
                    continue
                op = op_from_dict(raw)
                self.assertIsInstance(op, OpFeatures, f"{gname}/{bname}[{i}]")
                cost = predict_ops([op])
                self.assertTrue(
                    math.isfinite(cost) and cost >= 0.0,
                    f"{gname}/{bname}[{i}] scored {cost}",
                )
                n += 1
        self.assertGreater(n, 100, "suspiciously few featurized divisions")

    def test_featurization_coverage_is_high(self):
        # Extraction is best-effort, but a collapse in coverage means the
        # extractor stopped understanding the IR rather than a few odd ops.
        total = sum(b["menu_size"] for _, _, b in _entries())
        got = sum(1 for _, _, b in _entries() for f in b["features"] if f is not None)
        self.assertGreater(got / total, 0.85, f"only {got}/{total} featurized")


class DiscriminationTest(TestCase):
    """The property the fixture exists for."""

    def test_some_buffers_separate_across_their_division_menu(self):
        # Under the memory-only objective every one of these is an exact tie.
        separating = []
        for gname, bname, b in _entries():
            costs = {
                round(predict_ops([op_from_dict(f)]), 6)
                for f in b["features"]
                if f is not None
            }
            if len(costs) > 1:
                separating.append(
                    (f"{gname}/{bname}", len(costs), max(costs) / min(costs))
                )
        self.assertGreaterEqual(
            len(separating), 5, "the fixture no longer separates any divisions"
        )
        # And the separation is large, not a rounding artifact.
        self.assertGreater(max(s[2] for s in separating), 2.0)

    def test_only_matmul_and_reduction_buffers_separate(self):
        # cost_model reads ``cores`` only under ``is_matmul`` or ``is_reduction``,
        # so a pointwise op's cost is division-invariant by construction.
        # Asserting the split matches that expectation keeps the fixture honest
        # about *why* it separates.
        for gname, bname, b in _entries():
            feats = [op_from_dict(f) for f in b["features"] if f is not None]
            if not feats:
                continue
            costs = {round(predict_ops([f]), 6) for f in feats}
            if len(costs) > 1:
                self.assertTrue(
                    any(f.is_matmul or f.is_reduction for f in feats),
                    f"{gname}/{bname} separates but is pointwise",
                )


class SymbolicTiledFeatureTest(TestCase):
    """Coarse-tiled features carrying the co-optimizer's UNDECIDED symbols.

    ``CoOptimizingAllocator._extract_op_features`` keys ``is_lx`` and the core
    splits on the solver's own variables, so a coarse-tiled op reaches
    ``predict_ops`` with a symbolic per-core tile height. Every tiling surface
    keyed on it is a piecewise power law, and branching on a symbol raised
    ``cannot determine truth value of Relational`` -- taking down the compile,
    not just the objective (issue #4233).

    ``explain()`` (the ``SPYRE_DUMP_COST`` report path) is deliberately NOT
    covered here: it is reached only from ``dump_cost_model``'s dump hook on the
    deterministic pre-scheduling extraction, never from
    ``CoOptimizingAllocator``'s symbolic path -- so it never sees a symbolic
    feature and there is no regression surface to test.
    """

    #: One residency symbol PER BUFFER NAME, matching ``LifetimeBoundBuffer.sym_is_lx``
    #: (plan_solver.py) and how ``dump_cost_model.extract_op_features`` looks each arg's
    #: own buffer name up in the co-optimizer's ``is_lx`` map (dump_cost_model.py:700) --
    #: two different buffers get two DIFFERENT symbols, not one shared symbol.
    @staticmethod
    def _sym_is_lx(name: str) -> sympy.Symbol:
        return sympy.Symbol(f"is_lx_{name}", integer=True, nonnegative=True)

    #: What the allocator's split symbols look like (see ``plan_solver.sym_core_divs``):
    #: positive integer, one per stride coefficient. A single stand-in is enough here --
    #: only its SHAPE (must not survive) is under test, not its identity.
    SPLIT = sympy.Symbol("output_split_buf0_d0", integer=True, positive=True)

    #: Untiled row extent. A constant, not the capture's own: some captured
    #: ``logical`` entries are NAMED dims (strings), and the tile height only has
    #: to be symbolic in the right variables, not realistic.
    ROWS = 1024

    def _symbolize(self, op: OpFeatures, loop_trip: int = 8) -> OpFeatures:
        """``op`` as the co-optimizing path presents it: each arg carrying its OWN
        per-buffer residency symbol, and output-tiled with a symbolic per-core tile
        height keyed on the output's residency symbol and a split symbol."""
        rows = self.ROWS
        args = [dataclasses.replace(a, is_lx=self._sym_is_lx(a.name)) for a in op.args]
        out_is_lx = next(a.is_lx for a in args if a.role == "output")
        return dataclasses.replace(
            op,
            args=args,
            loop_trip=loop_trip,
            tiles_output_dim=True,
            tile_rows_per_core=(rows / loop_trip * (1 - out_is_lx) + rows * out_is_lx)
            / self.SPLIT,
        )

    @staticmethod
    def _leaked_symbols(expr) -> set:
        """Free symbols other than a per-buffer residency symbol -- the SPLIT symbol,
        or anything else, must never survive into the cost expression."""
        return {s for s in expr.free_symbols if not s.name.startswith("is_lx_")}

    def test_every_captured_op_builds_a_cost_expression_when_tiled(self):
        checked = 0
        for gname, bname, b in _entries():
            for raw in b["features"][:2]:
                if raw is None:
                    continue
                op = self._symbolize(op_from_dict(raw))
                # The bug: this raised TypeError rather than returning anything.
                expr = sympy.sympify(predict_ops([op]))
                self.assertFalse(
                    self._leaked_symbols(expr),
                    f"{gname}/{bname}: a non-residency symbol leaked into the cost "
                    "expr (the tiling split, or something else)",
                )
                checked += 1
        self.assertGreater(checked, 10)

    def test_the_tiled_cost_expression_stays_linearizable(self):
        """``_SympyExprToCpSat`` handles Add/Mul/Min/Max and ``symbol**-1`` only.

        A symbolic derate would be a ``Piecewise`` over a fractional ``Pow``,
        which does not linearize -- CP-SAT then discards the WHOLE cost
        objective and falls back to its lexicographic solve, so neutralising the
        derate is what keeps the objective usable, not a shortcut around it.
        """
        for _, _, b in _entries():
            raw = next((f for f in b["features"] if f is not None), None)
            if raw is None:
                continue
            expr = sympy.sympify(predict_ops([self._symbolize(op_from_dict(raw))]))
            self.assertFalse(expr.atoms(sympy.Piecewise))
            for pow_ in expr.atoms(sympy.Pow):
                self.assertEqual(pow_.exp, -1, f"non-invertible power {pow_}")

    def test_a_loop_reread_arg_does_not_consult_symbolic_mem(self):
        """``ArgTraffic.mem`` REJECTS a symbolic ``is_lx``, and
        ``_loop_reread_bytes`` is reached unconditionally for an output-tiled
        matmul, so it used to lose the whole objective to a ValueError there --
        but only once ``_tiled_rows`` already lets execution get that far. On
        the TRUE parent commit, the underfill-eff loop's
        ``o.tile_rows_per_core > 0`` comparison (``predict_ops``, a few lines
        before ``_loop_reread_bytes`` is called) raises the SAME TypeError as
        the other two tests above, first -- so this isolates the
        ``_loop_reread_bytes``/``ArgTraffic.mem`` defect specifically, and
        checks the fixed term is actually PRESENT and correctly scaled, not
        just silently absent, against ``_loop_reread_bytes`` itself.
        """
        raw = next(
            (
                f
                for _, _, b in _entries()
                for f in b["features"]
                if f is not None and f.get("is_matmul")
            ),
            None,
        )
        self.assertIsNotNone(raw, "fixture no longer holds a matmul")
        symbolized = self._symbolize(op_from_dict(raw))

        def with_input_loop_factor(lf: int) -> OpFeatures:
            return dataclasses.replace(
                symbolized,
                args=[
                    dataclasses.replace(a, loop_factor=lf) if a.role == "input" else a
                    for a in symbolized.args
                ],
            )

        base, tiled = with_input_loop_factor(1), with_input_loop_factor(4)
        expected_extra = sum(
            a.elems * 3 * tiled.dtype_bytes * (1 - a.is_lx)
            for a in tiled.args
            if a.role == "input"
        )
        # Inert at loop_factor=1 (matches _loop_reread_bytes' own docstring), and
        # exactly the documented excess-over-first-pass term at loop_factor=4 -- not
        # zeroed, not double-counted, per-arg residency correctly weighted in.
        self.assertEqual(_loop_reread_bytes([base]), 0.0)
        self.assertEqual(sympy.expand(_loop_reread_bytes([tiled]) - expected_extra), 0)
        # And the fix is reachable end-to-end through predict_ops, not just in
        # isolation.
        sympy.sympify(predict_ops([tiled]))

    def test_a_standalone_reduction_does_not_consult_symbolic_mem(self):
        """``_reduction_rows`` filtered its HBM candidates via ``a.mem == "hbm"``,
        which -- like ``_loop_reread_bytes`` before it -- REJECTS a symbolic
        ``is_lx``. Unlike the matmul case above, this branch of ``predict_ops``
        (a standalone, UNTILED reduction: ``len(ops) == 1``, ``is_reduction``,
        not matmul, ``tiles_output_dim=False``) never touches
        ``tile_rows_per_core`` at all, so it is reachable even when tiling is
        not involved (935 of the 6453 concrete-feature predictions this commit's
        message cites are exactly this shape). Built WITHOUT ``_symbolize``,
        which forces ``tiles_output_dim=True`` and would route around this
        branch entirely.
        """
        raw = next(
            (
                f
                for _, _, b in _entries()
                for f in b["features"]
                if f is not None
                and f.get("is_reduction")
                and not f.get("is_matmul")
                and not f.get("tiles_output_dim")
            ),
            None,
        )
        self.assertIsNotNone(raw, "fixture no longer holds an untiled reduction")
        op = op_from_dict(raw)
        op = dataclasses.replace(
            op,
            args=[
                dataclasses.replace(a, is_lx=self._sym_is_lx(a.name)) for a in op.args
            ],
        )
        self.assertEqual(op.loop_trip, 1, "captured entry unexpectedly coarse-tiled")
        expr = sympy.sympify(predict_ops([op]))  # ValueError before the fix
        self.assertFalse(
            self._leaked_symbols(expr), "a non-residency symbol leaked into the cost"
        )

    def test_a_working_set_with_symbolic_cols_does_not_break_the_spill_gate(self):
        """``_lx_spill_working_set`` multiplies ``rpc * _op_cols(o)``; ``_tiled_rows``
        neutralizes a symbolic ``rpc`` but ``_op_cols`` -- an arg's own LOGICAL row
        width, e.g. a dynamic-shape symbol, NOT a co-optimizer variable -- was not
        similarly guarded. ``_lx_spill_bw_derate`` then compares the resulting
        symbolic working set against its byte cap (``if ws <= _cap``), the same
        "cannot determine truth value of Relational" failure as #4233 from an
        unrelated source. Built from scratch (independent of ``_symbolize`` and of
        the fixture's own residency) so ``rpc`` stays concrete and only ``cols`` is
        symbolic -- and only ONE arg carries a ``logical`` shape at all, since
        ``_op_cols`` takes the max over every arg that has one.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)  # a dynamic-shape symbol
        op = OpFeatures(
            name="symbolic_cols_pointwise",
            is_reduction=False,
            out_elems=1024,
            cores=1,
            dtype_bytes=2,
            tiles_output_dim=True,
            tile_rows_per_core=16.0,  # concrete: not neutralized by `_tiled_rows`
            args=[
                ArgTraffic(
                    name="out0", role="output", is_lx=False, elems=1024, logical=[]
                ),
                ArgTraffic(
                    name="in0",
                    role="input",
                    is_lx=False,
                    elems=1024,
                    logical=[16, s0],  # the ONLY arg with a logical shape
                ),
            ],
        )
        expr = sympy.sympify(predict_ops([op]))  # TypeError before the fix
        self.assertFalse(
            expr.free_symbols, f"the dynamic-shape symbol leaked into the cost: {expr}"
        )

    def test_predict_by_bundle_handles_a_mixed_symbolic_and_concrete_bundle(self):
        """``predict_by_bundle`` -- what ``CoOptimizingAllocator._solve`` actually
        calls (allocator.py:1845) -- groups ``operations`` via
        ``fusion.estimate_bundles`` before any op reaches ``predict_ops``. Real IR
        operations aren't available from this JSON fixture, so the grouping is
        stubbed to put one symbolic-tiled op and one concrete op in the SAME
        bundle -- e.g. an output-tiled matmul fused beside an untiled epilogue --
        checking the bundle-level dedup/derate logic tolerates the mix.
        """
        raw = next(
            (f for _, _, b in _entries() for f in b["features"] if f is not None), None
        )
        self.assertIsNotNone(raw)
        concrete = dataclasses.replace(op_from_dict(raw), name="concrete_op")
        symbolic = self._symbolize(op_from_dict(raw))
        symbolic = dataclasses.replace(
            symbolic,
            name="symbolic_op",
            # Distinct arg names: `raw` is reused for both ops, and
            # `_fused_hbm_bytes` dedups external "argN" inputs by name across a
            # bundle -- sharing names would run a concrete float and a symbolic
            # HBM-bytes term through the same `max()`, a collision this test is
            # not about.
            args=[dataclasses.replace(a, name=f"sym_{a.name}") for a in symbolic.args],
        )
        features_by_buffer = {concrete.name: concrete, symbolic.name: symbolic}
        fake_ops = [concrete, symbolic]  # group_features_by_bundle only reads `.name`
        with patch(
            "torch_spyre._inductor.fusion.estimate_bundles", return_value=[fake_ops]
        ):
            expr = sympy.sympify(predict_by_bundle(fake_ops, features_by_buffer))
        self.assertFalse(
            self._leaked_symbols(expr),
            "a non-residency symbol leaked into the bundle-level cost",
        )


class SymbolicMatmulSplitCostTest(TestCase):
    """A matmul whose CORE SPLITS -- not just ``is_lx`` -- are undecided symbols.

    ``CoOptimizingAllocator._extract_op_features`` builds the work slices from
    ``plan_solver.sym_core_divs``, so ``dump_cost_model._matmul_features`` sees one
    ``sympy.Symbol`` per split and a matmul reaches ``predict_ops`` with symbolic
    ``m``/``n``/``k``, symbolic ``cores`` and a symbolic ``matmul_rows_per_core``.
    That is a DIFFERENT shape from ``SymbolicTiledFeatureTest``, which symbolises
    only ``is_lx``, and it is the shape ``work_division._matmul_split_cost``'s own
    ``is_symbolic`` guards were added for (#3810) -- nothing currently pins it.

    The proxy that class uses for linearizability (no ``Piecewise``, every ``Pow``
    exponent -1) does not apply here: a symbolic split legitimately produces
    ``log(m)`` and ``1/(m*n*k)``, which ``_SympyExprToCpSat`` linearizes through its
    ``log2_``/``inv_`` aux symbols and ``AddMultiplicationEquality``. So this runs
    the real converter rather than a structural stand-in.
    """

    #: One symbol per split, exactly as ``plan_solver.sym_core_divs`` declares them.
    #: ``integer`` + ``positive`` is load-bearing twice over: it lets sympy collapse
    #: ``Max(1, split)`` to ``split`` (so ``_SympyExprToCpSat._inv_sym`` still sees a
    #: BARE symbol under the reciprocal), and it lets ``expand_log`` split
    #: ``log(a/split)``.
    M_SPLIT = sympy.Symbol("output_split_buf5_i0", integer=True, positive=True)
    N_SPLIT = sympy.Symbol("output_split_buf5_i1", integer=True, positive=True)
    K_SPLIT = sympy.Symbol("reduction_split_buf5_r0", integer=True, positive=True)
    IS_LX = sympy.Symbol("is_lx_buf5", integer=True, nonnegative=True)

    #: The candidate splits a ``CoreDivisionBuffer`` would offer, which the CP-SAT
    #: converter reads back as ``_raw_<symbol>`` to tabulate ``log2``/``inv``.
    RAW_SPLITS = (1, 2, 4, 8, 16, 32)

    M, N, K, DTYPE_BYTES = 1024, 1024, 64, 2

    def _params(self):
        from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

        return _COST_PARAMS

    def _matmul(self) -> OpFeatures:
        """The record ``_matmul_features`` emits under symbolic work slices.

        Note what is and is not symbolic: ``matmul_a_bytes``/``matmul_b_bytes`` are
        ``M*K`` / ``K*N`` and never touch a split, so they stay CONCRETE, which is
        what keeps ``_matmul_axes_for_split_cost`` able to recover ``K``.
        """
        # Aliased: the module header already binds this name, and a second
        # top-level binding here would be a redefinition.
        from torch_spyre._inductor.cost_model import ArgTraffic as _Arg

        m, n, k, db = self.M, self.N, self.K, self.DTYPE_BYTES
        args = [
            _Arg(
                "buf5",
                "output",
                self.IS_LX,
                m * n,
                dims=[m, n // 64, 64],
                logical=[m, n],
            ),
            _Arg(
                "arg0",
                "input",
                False,
                m * k,
                dims=[m, 1, 64],
                logical=[m, k],
            ),
            _Arg(
                "arg1",
                "input",
                False,
                k * n,
                dims=[k, n // 64, 64],
                logical=[k, n],
            ),
        ]
        return OpFeatures(
            name="bmm",
            is_reduction=True,
            out_elems=m * n,
            cores=self.M_SPLIT * self.N_SPLIT * self.K_SPLIT,
            dtype_bytes=db,
            args=args,
            reduction_cores=self.K_SPLIT,
            is_matmul=True,
            matmul_macs=m * n * k,
            matmul_rows_per_core=m / self.M_SPLIT,
            matmul_cols_per_core=n / self.N_SPLIT,
            matmul_a_bytes=m * k * db,
            matmul_b_bytes=k * n * db,
            matmul_m_split=self.M_SPLIT,
            matmul_n_split=self.N_SPLIT,
        )

    def test_the_extractor_max_is_symbolic_aware(self):
        """``dump_cost_model`` SHADOWS the builtin ``max`` with the sympy-aware one.

        ``_matmul_features`` reduces the splits with ``max(1, readable.get(...))``.
        With ``builtins.max`` that raises ``cannot determine truth value of
        Relational``, its best-effort ``except Exception`` zeroes
        ``matmul_a_bytes``/``matmul_b_bytes``, and every matmul bundle then loses the
        graph's whole cost objective (see the counterfactual below). The import at
        the top of ``dump_cost_model`` is the only thing preventing that, so pin it.
        """
        from torch_spyre._inductor import dump_cost_model

        self.assertEqual(dump_cost_model.max(1, self.M_SPLIT), self.M_SPLIT)
        self.assertEqual(dump_cost_model.max(1, 3), 3)

    def test_symbolic_splits_still_resolve_the_matmul_axes(self):
        """``M``/``N``/``K``/``B`` come back CONCRETE; only the splits stay symbolic.

        ``M = matmul_rows_per_core * m_split`` cancels, so nothing symbolic reaches
        the ``round()`` calls in ``_matmul_axes_for_split_cost``.
        """
        from torch_spyre._inductor.cost_model import _matmul_axes_for_split_cost

        axes = _matmul_axes_for_split_cost(self._matmul())
        self.assertIsNotNone(axes)
        (_, b), (m_size, m), (n_size, n), (k_size, k), _ = axes
        self.assertEqual((m_size, n_size, k_size), (self.M, self.N, self.K))
        self.assertEqual({m, n, k}, {self.M_SPLIT, self.N_SPLIT, self.K_SPLIT})
        self.assertFalse(getattr(b, "free_symbols", set()))

    def test_a_symbolic_split_matmul_builds_a_cost_expression(self):
        expr = sympy.sympify(predict_ops([self._matmul()], self._params()))
        self.assertEqual(
            expr.free_symbols,
            {self.IS_LX, self.M_SPLIT, self.N_SPLIT, self.K_SPLIT},
        )
        # Not merely present: the objective has to MOVE with the division, or the
        # solver is optimizing a constant.
        self.assertNotEqual(expr.diff(self.M_SPLIT), 0)

    def test_the_symbolic_split_cost_expression_linearizes(self):
        """Run the REAL CP-SAT converter, not a structural proxy.

        ``sym_map`` mirrors ``CpSatLayoutSolver._minimize_cost_expr``: the split
        variable, plus the ``_buffer_``/``_raw_`` entries ``_print_Symbol`` needs to
        tabulate the ``log2_``/``inv_`` aux variables over the candidate divisions.
        """
        import types

        try:
            from ortools.sat.python import cp_model
        except ImportError:
            self.skipTest("the cpsat converter needs ortools")
        from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
            _SympyExprToCpSat,
        )

        expr = sympy.sympify(predict_ops([self._matmul()], self._params()))
        model = cp_model.CpModel()
        sym_map: dict = {}
        for sym in (self.M_SPLIT, self.N_SPLIT, self.K_SPLIT):
            sym_map[sym.name] = model.new_int_var_from_domain(
                cp_model.Domain.FromValues(list(self.RAW_SPLITS)), sym.name
            )
            sym_map[f"_buffer_{sym.name}"] = types.SimpleNamespace(
                division=model.new_int_var(
                    0, len(self.RAW_SPLITS) - 1, f"div_{sym.name}"
                )
            )
            sym_map[f"_raw_{sym.name}"] = list(self.RAW_SPLITS)
        sym_map[self.IS_LX.name] = model.new_bool_var(self.IS_LX.name)

        cp_cost = _SympyExprToCpSat(model, sym_map).convert(expr)
        # A constant would mean the objective collapsed rather than linearized.
        self.assertNotIsInstance(cp_cost, (int, float))
        model.minimize(cp_cost)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        self.assertIn(solver.Solve(model), (cp_model.OPTIMAL, cp_model.FEASIBLE))

    def test_losing_the_matmul_bytes_costs_the_whole_objective(self):
        """The counterfactual that makes the two tests above load-bearing.

        This is the state a failed extraction leaves behind -- ``is_matmul`` with no
        operand bytes. ``_matmul_ns_upstream`` raises, and since ``predict_by_bundle``
        sums over every bundle, ONE such op drops the objective for the whole graph.
        """
        lost = dataclasses.replace(
            self._matmul(),
            matmul_a_bytes=0,
            matmul_b_bytes=0,
            matmul_rows_per_core=0.0,
            matmul_cols_per_core=0.0,
            matmul_m_split=1,
            matmul_n_split=1,
            reduction_cores=1,
        )
        with self.assertRaisesRegex(RuntimeError, "unresolvable axes"):
            predict_ops([lost], self._params())


if __name__ == "__main__":
    unittest.main()
