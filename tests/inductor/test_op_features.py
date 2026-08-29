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

Division features (``sa_cooptimizer.features_for_division``) and residency
(``cost_objective.with_residency``) are the two halves of that story -- division
choice and placement choice -- so this file tests both against one shared
fixture rather than splitting per source module.

Regenerate with ``python3 docs/source/user_guide/examples/scratchpad/capture_op_features.py`` on a Spyre machine.
"""

import json
import math
import os
import unittest
from unittest import TestCase
from unittest.mock import patch

import sympy

from torch_spyre._inductor.cost_model import OpFeatures, op_from_dict, predict_ops
from torch_spyre._inductor.scratchpad.cost_objective import with_residency
from torch_spyre._inductor.scratchpad.sa_cooptimizer import features_for_division
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

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
    def test_candidate_is_passed_as_complete_symbol_keyed_map(self):
        """SA feature extraction neither encodes nor mutates Scheduler transport."""
        m, n, kk = sympy.symbols("m n kk")
        op = object()
        division = CoreDivision(output_splits={m: 8}, reduction_splits={kk: 2})
        expected = {m: 8, n: 1, kk: 2}
        with (
            patch(
                "torch_spyre._inductor.scratchpad.sa_cooptimizer.iteration_space_from_op",
                return_value={m: 1024, n: 1024, kk: 2048},
            ),
            patch(
                "torch_spyre._inductor.dump_cost_model.extract_op_features",
                return_value="features",
            ) as extract,
        ):
            self.assertEqual(features_for_division(op, division), "features")

        extract.assert_called_once_with(op, expected)


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


class ResidencyTest(TestCase):
    def test_lx_residency_never_increases_cost(self):
        # The model charges an LX-resident arg no HBM traffic, so marking args
        # resident can only help. This is the other half of what the search
        # decides, and the half the memory-only objective already modelled.
        checked = 0
        for gname, bname, b in _entries():
            for raw in b["features"][:3]:
                if raw is None:
                    continue
                op = op_from_dict(raw)
                names = {a.name for a in op.args}
                hbm = predict_ops([with_residency(op, set())])
                lx = predict_ops([with_residency(op, names)])
                self.assertLessEqual(lx, hbm + 1e-9, f"{gname}/{bname}")
                checked += 1
        self.assertGreater(checked, 10)

    def test_with_residency_does_not_mutate_its_input(self):
        # One extracted menu is scored against many candidate placements, so the
        # helper has to be pure or the second placement scores the first's args.
        for _, _, b in _entries():
            raw = next((f for f in b["features"] if f is not None), None)
            if raw is None:
                continue
            op = op_from_dict(raw)
            before = [a.mem for a in op.args]
            with_residency(op, {a.name for a in op.args})
            self.assertEqual([a.mem for a in op.args], before)
            return


if __name__ == "__main__":
    unittest.main()
