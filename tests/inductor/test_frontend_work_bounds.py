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

A timing threshold cannot guard the frontend: per-pass times spread 3.68% median
and 12.9% at p90 across a 50-point sweep, so a bound tight enough to catch a
regression also flakes. The same sweep's work counters spread 0.00% median and
0.03% at p90 -- stable to within a fraction of a percent -- so a bound on them
holds without pinning the implementation.

Two regressions are worth guarding, and they move different counters:

* **The memo stops absorbing repeats.** ``read_writes.misses`` rises toward the
  number of asks. Caught by the misses bound.
* **A caller bypasses the memo**, calling ``op.get_read_writes()`` instead of
  ``op_read_writes(op)``. Requests fall by that caller's share, misses barely
  move (the memo still fills once from whoever asks first), and
  ``read_writes.extractions`` rises. Only the extractions bound sees this.

An earlier revision bounded ``requests / misses`` instead, which catches neither:
defeating the memo fails the misses bound first, and a single bypass cannot move
a 534x ratio below any useful threshold.

Bounds are calibrated on the two graphs below rather than on the sweep, because
the sweep measures the pre-scheduling pipeline over model shapes while these
tests count a whole ``torch.compile`` of a small graph. Measured per operation
(``_calibrate``, torch 2.13 on a Spyre pod):

=================  ====  ========  ============  ==================  ========
workload            ops    misses   extractions  device coordinates  requests
=================  ====  ========  ============  ==================  ========
matmul_chain          4      1.50        102.00               31.00    203.25
elementwise_chain    32      1.03        107.03               26.81    214.31
=================  ====  ========  ============  ==================  ========

The sweep's figures over 4,443 operations counted at pre-scheduling-pipeline
entry -- 1.32 misses, 90.7 extractions, 39.1 device coordinates, 705.9 requests
per operation -- agree on everything except requests, which are far lower here
because a four-operation graph gives later passes little to re-ask about.

``matmul_chain``'s 1.50 misses per operation is a small-N artifact: six misses
over four operations. The bound holds anyway, and the second workload at 32
operations is the one that pins it.
"""

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


WORKLOADS = ("matmul_chain", "elementwise_chain")

#: Measured 1.03-1.50 here, 1.32 on the sweep, against a cold-cache floor of 1.0.
MAX_MISSES_PER_OP = 4.0
#: Per workload, because this is the bound a bypass has to breach and a single
#: figure covering both has to sit above the looser one. Measured 102.00 and
#: 107.03; 1.27x headroom, which is what it takes to catch the 11-site bypass
#: in the test plan (133.25 and 165.97). A counter this reproducible -- 0.03%
#: p90 across the sweep -- can carry a tight bound; a legitimate change that
#: breaches it should re-measure and move it rather than widen it blindly.
MAX_EXTRACTIONS_PER_OP = {"matmul_chain": 130.0, "elementwise_chain": 135.0}
#: Measured 27-31 here, 39.1 on the sweep. Unmemoized, so the count is the work.
MAX_DEVICE_COORDS_PER_OP = 100.0


def _workload(shape: str):
    """Two graphs that exercise the analysis differently.

    ``matmul_chain`` carries the layout and coordinate work; ``elementwise_chain``
    carries the operation count. A bound holding for both is not shape-specific.
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


def _counted_compile(fn, args) -> tuple[dict[str, int], int]:
    """Compile once, cold, and return (counter deltas, operations).

    The operation count is the sum of ``len(graph.operations)`` at
    pre-scheduling-pipeline entry over every run of that pipeline, because the
    counters sum the same way -- a graph that lowers through two pipeline runs
    contributes to both. The pipeline is reached by swapping the module
    attribute, which is what ``patches.enable_spyre_context`` resolves: it
    imports the name inside the function and builds the instance there.

    A cache hit skips the pipeline entirely, leaving every counter at zero and
    every bound passing vacuously, so the compile is forced cold. The
    PRECONDITION in ``_counts`` is the backstop if that stops working.
    """
    from unittest.mock import patch

    from torch_spyre._inductor import passes

    seen: list[int] = []

    class _Capturing(passes.CustomPreSchedulingPasses):
        def __call__(self, graph) -> None:
            seen.append(len(graph.operations))
            super().__call__(graph)

    torch._dynamo.reset()
    with (
        patch.object(passes, "CustomPreSchedulingPasses", _Capturing),
        torch._inductor.config.patch({"force_disable_caches": True}),
    ):
        # counted_region fills the dict on exit, so read it after the block.
        with pass_counters.counted_region() as counts:
            torch.compile(fn, fullgraph=True)(*args)
    return dict(counts), sum(seen)


class TestFrontendWorkBounds(TestCase):
    """Bounds on the analysis work one compile does per graph operation.

    One bound per test method, over a compile cached for the class. Stacking
    bounds in one method is what made an earlier revision's verification hollow:
    the first assertion failed and the rest never ran.
    """

    #: shape -> (counter deltas, operations). Cold compiles are expensive.
    _measured: dict[str, tuple[dict[str, int], int]] = {}

    @classmethod
    def _counts(cls, shape: str) -> tuple[dict[str, int], int]:
        if shape not in cls._measured:
            cls._measured[shape] = _counted_compile(*_workload(shape))
        deltas, ops = cls._measured[shape]
        assert ops, (
            f"PRECONDITION: the pre-scheduling pipeline never ran for {shape}, "
            "so nothing was counted -- most likely a cache hit. Not a work "
            "regression."
        )
        return deltas, ops

    @parametrize("shape", WORKLOADS)
    def test_memo_absorbs_repeat_asks(self, shape: str) -> None:
        """Misses stay near one per operation however often a pass asks."""
        deltas, ops = self._counts(shape)
        misses = deltas.get(READ_WRITES_MISSES, 0)
        requests = deltas.get(READ_WRITES_REQUESTS, 0)
        self.assertGreater(
            requests,
            0,
            f"PRECONDITION: no pass asked for a read/write set on {shape}, so "
            "the memo was never exercised. Not a work regression.",
        )
        self.assertLessEqual(
            misses / ops,
            MAX_MISSES_PER_OP,
            f"{shape}: the read-writes memo is no longer absorbing repeats -- "
            f"{misses} misses over {ops} operations ({misses / ops:.2f} per op) "
            f"against a bound of {MAX_MISSES_PER_OP}. A cold cache is 1.0 per "
            "operation, so a figure far above that means a pass is re-deriving "
            "rather than reusing.",
        )

    @parametrize("shape", WORKLOADS)
    def test_extractions_per_operation_are_bounded(self, shape: str) -> None:
        """The counter a memo bypass moves.

        A caller switching from ``op_read_writes(op)`` to ``op.get_read_writes()``
        leaves misses alone and raises this. Covers
        ``ComputedBuffer.get_read_writes`` only: the scheduler extracts directly
        in ``SchedulerNode._compute_attrs``, and so do several ``ir.py`` classes.

        Scope, measured rather than claimed: bypassing 11 call sites in
        ``scratchpad/utils.py`` raises this 31% on ``matmul_chain`` and 55% on
        ``elementwise_chain``, which the bound catches. Bypassing a *single* site
        moves it by a few percent and no bound with usable headroom would see it.
        This guards wholesale memo failure and multi-site drift, not one call.
        """
        deltas, ops = self._counts(shape)
        extractions = deltas.get(READ_WRITES_EXTRACTIONS, 0)
        bound = MAX_EXTRACTIONS_PER_OP[shape]
        self.assertGreater(
            extractions,
            0,
            f"PRECONDITION: no read/write sets were extracted on {shape}. Not a "
            "work regression.",
        )
        self.assertLessEqual(
            extractions / ops,
            bound,
            f"{shape}: {extractions} read-writes extractions over {ops} "
            f"operations ({extractions / ops:.1f} per op) against a bound of "
            f"{bound}. Each costs a full index trace, so a jump "
            "here is a caller reaching past op_read_writes to upstream's "
            "uncached method.",
        )

    @parametrize("shape", WORKLOADS)
    def test_coordinate_construction_is_bounded(self, shape: str) -> None:
        """Device coordinates are unmemoized, so their count is their cost."""
        deltas, ops = self._counts(shape)
        coords = deltas.get(DEVICE_COORDINATES, 0)
        self.assertGreater(
            coords,
            0,
            f"PRECONDITION: no device coordinates were constructed on {shape}, "
            "so this graph does not exercise layout analysis. Not a work "
            "regression.",
        )
        self.assertLessEqual(
            coords / ops,
            MAX_DEVICE_COORDS_PER_OP,
            f"{shape}: coordinate construction reached {coords / ops:.1f} per "
            f"operation against a bound of {MAX_DEVICE_COORDS_PER_OP}. Nothing "
            "memoizes these, so the count is the work.",
        )


def _calibrate() -> None:
    """Print each bounded counter per operation, for setting the bounds above.

    Not reachable as a flag on this file: ``common_utils`` parses ``sys.argv``
    at import and rejects anything it does not know. Drive it from outside::

        python3 -c "import sys; sys.argv=[sys.argv[0]]; \
            sys.path.insert(0, 'tests/inductor'); import torch, torch_spyre; \
            from test_frontend_work_bounds import _calibrate; _calibrate()"
    """
    keys = (READ_WRITES_MISSES, READ_WRITES_EXTRACTIONS, DEVICE_COORDINATES,
            READ_WRITES_REQUESTS)
    head = " ".join(f"{k.split('.')[-1]:>14}" for k in keys)
    print(f"{'workload':<20} {'ops':>5} {head}")
    for shape in WORKLOADS:
        deltas, ops = _counted_compile(*_workload(shape))
        per_op = " ".join(f"{deltas.get(k, 0) / ops:14.2f}" for k in keys)
        print(f"{shape:<20} {ops:5d} {per_op}")


instantiate_parametrized_tests(TestFrontendWorkBounds)


if __name__ == "__main__":
    run_tests()
