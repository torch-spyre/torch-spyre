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


"""Per-operation analysis work on a real compile, bounded.

A timing threshold cannot guard the frontend: sample-to-sample spread on the
2026-10-02 sweep was 3.68% median and 12.9% at p90 for per-pass times, so any
bound tight enough to catch a regression also flakes. The work counters have a
median spread of **0.00%** across the same samples -- 84.8% of counter series
were byte-identical over three cold runs -- so a bound on them is a real guard.

The numbers below are calibrated from that sweep: 50 points, 4,443 graph
operations, with the whole pre-scheduling pipeline counted.

======================================  ==========  ==================
counter                                 per op      what a breach means
======================================  ==========  ==================
``read_writes.misses``                  1.32        the memo stopped working
``read_writes.requests``                705.9       (absorbed by the memo)
``device_coordinates``                  39.1        a new unmemoized rescan
======================================  ==========  ==================

The misses figure is the load-bearing one. A perfectly cold cache is 1.0 per
operation, so 1.32 says the memo is serving essentially every repeat ask. The
705.9 requests behind those 1.32 misses are what makes the bound meaningful: a
change that routes a pass around ``op_read_writes`` converts requests into
misses and extractions, and only this ratio notices.

Bounds carry roughly 2-3x headroom over the measured value, because the point is
to catch a pass newly re-deriving per-duplicate or per-consumer -- which moves a
count by an order of magnitude -- and not to pin the current implementation.
"""

from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch_spyre._inductor import pass_counters
from torch_spyre._inductor.pass_counters import (
    DEVICE_COORDINATES,
    READ_WRITES_EXTRACTIONS,
    READ_WRITES_MISSES,
    READ_WRITES_REQUESTS,
)


#: Measured 1.32; a pass re-deriving per duplicate would push this past 1 per
#: duplicate and well beyond the bound.
MAX_MISSES_PER_OP = 4.0
#: Measured 39.1. Unmemoized by design, so the count is the work.
MAX_DEVICE_COORDS_PER_OP = 120.0
#: Measured 534x (705.9 requests against 1.32 misses). Guards the memo itself.
MIN_REQUESTS_PER_MISS = 20.0


class TestFrontendWorkBounds(TestCase):
    """Bounds on the analysis work one compile does per graph operation."""

    def _counted_compile(self, fn, args) -> tuple[dict[str, int], int]:
        """Compile and return (counter deltas, operations at scheduling).

        The operation count comes from a pipeline subclass swapped in over the
        module attribute, which is how the rest of these tests reach the
        pre-scheduling pipeline: ``patches.enable_spyre_context`` imports the
        name inside the function, so the module attribute is what it resolves.
        """
        from torch_spyre._inductor import passes

        seen: list[int] = []

        class _Capturing(passes.CustomPreSchedulingPasses):  # type: ignore[name-defined]
            def __call__(self, graph) -> None:
                seen.append(len(graph.operations))
                super().__call__(graph)

        # A cache hit skips the pipeline entirely, so every counter reads zero
        # and a bound on them passes vacuously. Forcing a cold compile is what
        # makes this a guard rather than a no-op; the PRECONDITION below is the
        # backstop if it ever stops working.
        torch._dynamo.reset()
        with (
            patch.object(passes, "CustomPreSchedulingPasses", _Capturing),
            torch._inductor.config.patch({"force_disable_caches": True}),
        ):
            # counted_region fills the dict on exit, so read it after the block.
            with pass_counters.counted_region() as counts:
                torch.compile(fn, fullgraph=True)(*args)
        deltas = dict(counts)

        self.assertTrue(
            seen,
            "PRECONDITION: the pre-scheduling pipeline never ran, so nothing "
            "was counted. Not a work regression -- the harness missed the hook.",
        )
        return deltas, max(seen)

    @parametrize("shape", ["matmul_chain", "elementwise_chain"])
    def test_memo_serves_every_repeat_ask(self, shape: str) -> None:
        """Misses stay near one per operation however often a pass asks.

        The failure this catches: a pass calling ``op.get_read_writes()``
        directly instead of ``op_read_writes``, or mutating dependencies without
        ``invalidate_op_read_writes`` and re-deriving to compensate. Either turns
        cheap requests into expensive extractions.
        """
        deltas, ops = self._counted_compile(*_workload(shape))
        misses = deltas.get(READ_WRITES_MISSES, 0)
        requests = deltas.get(READ_WRITES_REQUESTS, 0)

        self.assertGreater(
            requests,
            0,
            "PRECONDITION: no pass asked for a read/write set on this graph, so "
            "the memo was never exercised. Not a work regression.",
        )
        self.assertLessEqual(
            misses / ops,
            MAX_MISSES_PER_OP,
            f"the read-writes memo is no longer absorbing repeats: {misses} "
            f"misses over {ops} operations ({misses / ops:.2f} per op) against a "
            f"bound of {MAX_MISSES_PER_OP}. A cold cache is 1.0 per op and the "
            "sweep measured 1.32, so a figure far above that means a pass is "
            "re-deriving rather than reusing.",
        )
        self.assertGreaterEqual(
            requests / max(misses, 1),
            MIN_REQUESTS_PER_MISS,
            f"only {requests / max(misses, 1):.1f} requests per miss, against "
            f"{MIN_REQUESTS_PER_MISS} expected. The sweep measured 534. A low "
            "ratio means callers are going around op_read_writes, so the memo "
            "has nothing to serve.",
        )

    def test_coordinate_construction_is_bounded(self) -> None:
        """Device coordinates are unmemoized, so their count is their cost."""
        deltas, ops = self._counted_compile(*_workload("matmul_chain"))
        coords = deltas.get(DEVICE_COORDINATES, 0)
        self.assertGreater(
            coords,
            0,
            "PRECONDITION: no device coordinates were constructed, so this graph "
            "does not exercise layout analysis. Not a work regression.",
        )
        self.assertLessEqual(
            coords / ops,
            MAX_DEVICE_COORDS_PER_OP,
            f"coordinate construction grew to {coords / ops:.1f} per operation "
            f"against a bound of {MAX_DEVICE_COORDS_PER_OP} (sweep: 39.1). "
            "Nothing memoizes these, so the count is the work.",
        )

    def test_extractions_do_not_outnumber_requests(self) -> None:
        """Every extraction should be a miss the memo could not serve.

        Extractions far above misses mean callers reaching past the helper to
        upstream's uncached ``ComputedBuffer.get_read_writes``, which costs
        ~143 us a call. They are counted separately for exactly this reason.
        """
        deltas, ops = self._counted_compile(*_workload("matmul_chain"))
        extractions = deltas.get(READ_WRITES_EXTRACTIONS, 0)
        misses = deltas.get(READ_WRITES_MISSES, 0)
        self.assertGreaterEqual(
            extractions,
            misses,
            f"{extractions} extractions against {misses} memo misses: every "
            "miss goes on to extract, so extractions cannot be the smaller "
            "number. The counters disagree, which is an instrumentation fault "
            "rather than a work regression.",
        )


def _workload(shape: str):
    """Two graphs that exercise the analysis differently.

    ``matmul_chain`` carries the layout and coordinate work (the sweep's model
    shapes spend 160 ms a buffer in buffer preparation); ``elementwise_chain``
    carries the op count. A bound that holds for both is not shape-specific.
    """
    dtype = torch.float16
    if shape == "matmul_chain":
        x = torch.randn(64, 256, dtype=dtype, device="spyre")
        w1 = torch.randn(256, 256, dtype=dtype, device="spyre")
        w2 = torch.randn(256, 128, dtype=dtype, device="spyre")

        def fn(x, w1, w2):
            return torch.relu(torch.relu(x @ w1) @ w2)

        return fn, (x, w1, w2)

    x = torch.randn(64, 256, dtype=dtype, device="spyre")

    def fn(x):
        out = x
        for _ in range(16):
            out = torch.relu(out) * 1.5
        return out

    return fn, (x,)


instantiate_parametrized_tests(TestFrontendWorkBounds)


if __name__ == "__main__":
    run_tests()
