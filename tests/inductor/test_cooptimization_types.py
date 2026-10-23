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

"""The captured co-optimization graphs against the *real* substrate types.

* the captured graphs rehydrate into real :class:`CoreDivisionBuffer` objects
  that are well-formed, self-consistent, and a valid seed state (every op at
  division index 0);
* the two capture-derived assumptions the loader and the objective rest on hold:
  the ``boundary_cost`` -> ``BufferType`` mapping is exact, and ``read_count``
  equals the consumer count the old objective scaled by.
"""

import unittest
from collections import Counter
from unittest import TestCase

from torch_spyre._inductor.scratchpad.plan_solver import BufferType
from buffer_spec import CoreDivisionBuffer

from cooptimization_capture_loader import (
    DEFAULT_CAPTURE_PATH,
    LARGE_CAPTURE_PATH,
    SEED_DIVISION_INDEX,
    load_captures,
)

_ALL_CAPTURE_PATHS = (None, LARGE_CAPTURE_PATH)


def _all_captures():
    """Every captured case from both corpora, as ``(tag, graph)``."""
    for path in _ALL_CAPTURE_PATHS:
        cases = load_captures() if path is None else load_captures(path)
        for case, graphs in cases.items():
            for gi, g in enumerate(graphs):
                yield f"{case}[{gi}]", g


class CaptureRehydrationTest(TestCase):
    """Captures load into real, well-formed ``CoreDivisionBuffer`` objects."""

    def test_captures_load_and_are_non_empty(self):
        tags = list(_all_captures())
        self.assertGreater(len(tags), 0)
        for tag, g in tags:
            self.assertGreater(len(g.buffers), 0, f"{tag} has no buffers")

    def test_buffers_are_the_real_class(self):
        for tag, g in _all_captures():
            for b in g.buffers:
                self.assertIsInstance(b, CoreDivisionBuffer, f"{tag} {b.name}")

    def test_buffers_are_well_formed(self):
        for tag, g in _all_captures():
            for b in g.buffers:
                ctx = f"{tag} {b.name}"
                self.assertGreater(len(b.uses), 0, f"{ctx}: empty uses")
                self.assertEqual(b.uses, sorted(b.uses), f"{ctx}: unsorted uses")
                self.assertLess(b.start_time, b.end_time, f"{ctx}: bad lifetime")
                self.assertGreater(
                    len(b.core_divisions), 0, f"{ctx}: no candidate divisions"
                )

    def test_edges_reference_known_buffers(self):
        for tag, g in _all_captures():
            by_name = g.by_name()
            names = set(by_name)
            for b in g.buffers:
                ctx = f"{tag} {b.name}"
                for p in b.parents:
                    self.assertIn(p, names, f"{ctx}: parent {p!r} unknown")
                for p, pairs in b.cd_parent_matches.items():
                    self.assertIn(p, b.parents, f"{ctx}: match parent {p!r}")
                    parent = by_name[p]
                    for pj, cj in pairs:
                        self.assertTrue(
                            0 <= pj < len(parent.core_divisions),
                            f"{ctx}: parent idx {pj} out of range",
                        )
                        self.assertTrue(
                            0 <= cj < len(b.core_divisions),
                            f"{ctx}: child idx {cj} out of range",
                        )

    def test_solved_reference_is_consistent(self):
        for tag, g in _all_captures():
            by_name = g.by_name()
            self.assertLessEqual(set(g.solved), set(by_name), f"{tag}: name mismatch")
            for name, sol in g.solved.items():
                ctx = f"{tag} {name}"
                b = by_name[name]
                cd_idx = sol["chosen_division"]
                if cd_idx is not None:
                    self.assertTrue(
                        0 <= cd_idx < len(b.core_divisions),
                        f"{ctx}: chosen_division {cd_idx} out of range",
                    )
                if sol["resident"]:
                    self.assertIsNotNone(sol["address"], f"{ctx}: resident, no addr")
                    self.assertIsNone(
                        b.residency_reason, f"{ctx}: resident but pinned out"
                    )

    def test_seed_is_a_valid_state(self):
        for tag, g in _all_captures():
            for b in g.buffers:
                self.assertGreater(len(b.core_divisions), SEED_DIVISION_INDEX)
                b.chosen_division = SEED_DIVISION_INDEX
                self.assertEqual(b.chosen_division, SEED_DIVISION_INDEX)
                self.assertIsNone(b.address)


class CaptureAssumptionsTest(TestCase):
    """The two capture properties the loader and the objective depend on."""

    def test_capture_boundary_mapping_is_exact(self):
        """``boundary_cost`` is either 0 or exactly ``size``, never anything else.

        The loader maps ``boundary_cost > 0`` to ``BufferType.Output`` on the
        grounds that a nonzero value is the graph output's unavoidable write-out,
        i.e. exactly one ``size``. If a capture ever carried a partial boundary
        cost that mapping would be lossy, so pin it.
        """
        import json

        for path in (DEFAULT_CAPTURE_PATH, LARGE_CAPTURE_PATH):
            with open(path) as f:
                raw = json.load(f)
            for case, graphs in raw.items():
                for gi, g in enumerate(graphs):
                    nonzero = 0
                    for b in g["inputs"]:
                        bc = b["boundary_cost"]
                        if bc:
                            nonzero += 1
                            self.assertEqual(
                                bc,
                                b["size"],
                                f"{case}[{gi}] {b['name']}: boundary_cost {bc} "
                                f"!= size {b['size']}",
                            )
                        # The producer write is always the full buffer.
                        self.assertEqual(
                            b["spill_write_cost"],
                            b["size"],
                            f"{case}[{gi}] {b['name']}: partial spill_write_cost",
                        )
                    # Exactly one graph output per captured graph.
                    self.assertEqual(
                        nonzero, 1, f"{case}[{gi}]: {nonzero} boundary buffers, want 1"
                    )

    def test_rehydrated_boundary_types(self):
        """Exactly one ``Output`` per graph; everything else ``Intermediate``."""
        for tag, g in _all_captures():
            kinds = Counter(b.boundary for b in g.buffers)
            self.assertEqual(kinds[BufferType.Output], 1, f"{tag}: {dict(kinds)}")
            self.assertEqual(
                kinds[BufferType.Intermediate], len(g.buffers) - 1, f"{tag}"
            )

    def test_read_count_matches_consumer_count(self):
        """Reads-served equals the distinct-consumer count on every capture.

        The objective used to scale the spill cost by the number of consumer
        buffers (``len(children)``); the landed formula scales by the buffer's own
        reads-served count. They agree on all 308 captured buffers, which is what
        makes adopting the landed formula score-preserving. If a future capture
        breaks this, the objective stops matching the old one and this fires.

        Reads-served is ``read_count`` minus an input's clone-in, not
        ``read_count`` itself: ``read_count`` counts every read, and for a graph
        input the first one is the clone-in that pinning cannot avoid.
        ``buffer_graph.spill_cost`` discounts it, so the equivalence this test
        protects is against the discounted count.
        """
        for tag, g in _all_captures():
            by_name = g.by_name()
            kids = Counter()
            for c in g.buffers:
                for p in c.parents:
                    if p in by_name:
                        kids[p] += 1
            for b in g.buffers:
                reads_served = b.read_count - (1 if b.first_use_is_read else 0)
                self.assertEqual(
                    reads_served,
                    kids[b.name],
                    f"{tag} {b.name}: reads served {reads_served} != "
                    f"consumers {kids[b.name]}",
                )


class SpillCostTest(TestCase):
    """The spill cost the solver charges (``buffer_graph.spill_cost``)."""

    @staticmethod
    def _buf(size, uses, boundary):
        return CoreDivisionBuffer(
            name="x",
            size=size,
            uses=list(uses),
            first_use_is_read=False,
            boundary=boundary,
        )

    def test_intermediate_pays_the_producer_write(self):
        from torch_spyre._inductor.scratchpad.buffer_graph import spill_cost

        b = self._buf(2048, [0, 1, 2, 3], BufferType.Intermediate)
        # read_count == 3, plus the producer write.
        self.assertEqual(spill_cost(b), 4 * 2048)

    def test_boundary_buffers_do_not_pay_the_write(self):
        from torch_spyre._inductor.scratchpad.buffer_graph import spill_cost

        for boundary in (BufferType.Input, BufferType.Output):
            b = self._buf(2048, [0, 1, 2, 3], boundary)
            self.assertEqual(spill_cost(b), 3 * 2048)


if __name__ == "__main__":
    unittest.main()
