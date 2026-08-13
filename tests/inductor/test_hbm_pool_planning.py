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

import unittest
from unittest.mock import MagicMock

import torch
from sympy import Integer
from torch import fx
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor.hbm_pool_planning import hbm_pool_planning
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.scheduler import CountedLoopSchedulerNode


def _make_ftl_buffer(name, host_size=(64,), dim_order=(0,)):
    """Real ComputedBuffer with a FixedTiledLayout, for pool-eligibility tests.

    Mirrors _make_ftl_op in test_coarse_tiling.py:1187, trimmed to what
    hbm_pool_planning needs (device_layout.device_size must be set for
    _compute_size_bytes to work).
    """
    strides = [int(s) for s in FlexibleLayout.contiguous_strides(list(host_size))]
    device_layout = SpyreTensorLayout(
        list(host_size),
        strides,
        torch.float16,
        list(dim_order),
        ElementArrangement.STANDARD,
    )
    layout = FixedTiledLayout(
        torch.device("cpu"),
        torch.float16,
        [Integer(s) for s in host_size],
        [Integer(s) for s in strides],
        device_layout,
    )
    pw = Pointwise(
        device=torch.device("cpu"),
        dtype=torch.float16,
        inner_fn=lambda index: Integer(1),
        ranges=[Integer(s) for s in host_size],
    )
    buf = ComputedBuffer(name=name, layout=layout, data=pw)
    V.graph.name_to_buffer[name] = buf
    return buf


def _make_snode_with_rw(name, writes, reads):
    """MagicMock SchedulerNode with real MemoryDep read_writes for the
    given buffer names, and get_nodes()/get_name() wired for
    _iter_all_nodes / bundle-name lookup.

    FusedSchedulerNode.__init__ -> init_group_node/refresh_group_node_
    dependencies walks several BaseSchedulerNode bookkeeping attributes
    (ancestors, unmet_dependencies, min/max_order, min/max_input_distance,
    outputs) that a bare ``MagicMock(spec=SchedulerNode)`` leaves
    unconfigured and which then raise AttributeError. These are normally
    populated by the real Scheduler during node construction; since these
    tests build bundles directly (bypassing the Scheduler), fill them in
    with the same empty/zero defaults BaseSchedulerNode.__init__ uses.
    """
    snode = MagicMock(spec=SchedulerNode)
    snode.get_name.return_value = name
    snode.get_nodes.return_value = [snode]
    snode.ancestors = OrderedSet()
    snode.unmet_dependencies = OrderedSet()
    snode.min_order = 0
    snode.max_order = 0
    snode.min_input_distance = 0
    snode.max_input_distance = 0
    snode.outputs = []
    snode.is_reduction.return_value = False
    snode.group = (torch.device("cpu"), ())
    snode.read_writes = MagicMock()
    snode.read_writes.writes = {MemoryDep(w, Integer(0), (), ()) for w in writes}
    snode.read_writes.reads = {MemoryDep(r, Integer(0), (), ()) for r in reads}
    return snode


class TestHbmPoolPlanningPerBundle(unittest.TestCase):
    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()
        # A bare GraphLowering() never runs lowering, so graph_outputs is
        # never set by __init__ (only assigned later, mid-lowering). Set it
        # explicitly so get_output_names()/get_output_names-based filters
        # below don't raise AttributeError.
        V.graph.graph_outputs = []

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    def test_buffer_local_to_one_bundle_is_pool_eligible(self):
        """A buffer written and read within the same bundle gets an
        hbm_pool allocation."""
        _make_ftl_buffer("buf0")
        writer = _make_snode_with_rw("writer", writes=["buf0"], reads=[])
        reader = _make_snode_with_rw("reader", writes=[], reads=["buf0"])
        bundle = FusedSchedulerNode(MagicMock(), [writer, reader])

        hbm_pool_planning([bundle])

        buf = V.graph.get_buffer("buf0")
        self.assertIn("hbm_pool", buf.get_layout().allocation)
        self.assertIn(bundle.get_name(), V.graph.hbm_pool_sizes)

    def test_buffer_crossing_bundles_is_not_pool_eligible(self):
        """A buffer written in bundle A and read in bundle B falls back
        to standalone HBM (no hbm_pool allocation)."""
        _make_ftl_buffer("buf0")
        writer = _make_snode_with_rw("writer", writes=["buf0"], reads=[])
        reader = _make_snode_with_rw("reader", writes=[], reads=["buf0"])
        bundle_a = FusedSchedulerNode(MagicMock(), [writer])
        bundle_b = FusedSchedulerNode(MagicMock(), [reader])

        hbm_pool_planning([bundle_a, bundle_b])

        buf = V.graph.get_buffer("buf0")
        self.assertNotIn("hbm_pool", buf.get_layout().allocation)

    def test_multi_bundle_graph_produces_multiple_pool_size_entries(self):
        """Two independent bundles, each with their own local-only
        intermediate, each get their own hbm_pool_sizes entry."""
        _make_ftl_buffer("buf0")
        _make_ftl_buffer("buf1")
        w0 = _make_snode_with_rw("w0", writes=["buf0"], reads=[])
        r0 = _make_snode_with_rw("r0", writes=[], reads=["buf0"])
        w1 = _make_snode_with_rw("w1", writes=["buf1"], reads=[])
        r1 = _make_snode_with_rw("r1", writes=[], reads=["buf1"])
        bundle_a = FusedSchedulerNode(MagicMock(), [w0, r0])
        bundle_b = FusedSchedulerNode(MagicMock(), [w1, r1])

        hbm_pool_planning([bundle_a, bundle_b])

        self.assertEqual(len(V.graph.hbm_pool_sizes), 2)
        self.assertIn(bundle_a.get_name(), V.graph.hbm_pool_sizes)
        self.assertIn(bundle_b.get_name(), V.graph.hbm_pool_sizes)

    def test_disabled_config_sets_empty_dict(self):
        from torch_spyre._inductor import config as cfg

        old = cfg.hbm_pool_planning
        cfg.hbm_pool_planning = False
        try:
            result = hbm_pool_planning([])
        finally:
            cfg.hbm_pool_planning = old
        self.assertEqual(V.graph.hbm_pool_sizes, {})
        self.assertEqual(result, [])

    def test_single_bundle_graph_offsets_match_pre_reorder_behavior(self):
        """A graph that fuses into exactly one bundle must get the same
        per-buffer hbm_pool offsets and total size that the old
        graph-global scheme would have produced for the same buffers --
        the per-bundle rewrite must not change single-bundle behavior.

        This pins down parity by computing the expected offsets directly
        from the same Allocator/_compute_size_bytes primitives the
        implementation itself uses, rather than hardcoding byte constants
        that would silently drift if _compute_size_bytes's stick-alignment
        changes for unrelated reasons.
        """
        from torch_spyre._inductor.constants import SEGMENT_SIZE
        from torch_spyre._inductor.hbm_pool_planning import (
            Allocator,
            _compute_size_bytes,
        )

        _make_ftl_buffer("buf0", host_size=(64,))
        _make_ftl_buffer("buf1", host_size=(128,))
        w0 = _make_snode_with_rw("w0", writes=["buf0"], reads=[])
        mid = _make_snode_with_rw("mid", writes=["buf1"], reads=["buf0"])
        r1 = _make_snode_with_rw("r1", writes=[], reads=["buf1"])
        bundle = FusedSchedulerNode(MagicMock(), [w0, mid, r1])

        hbm_pool_planning([bundle])

        expected_alloc = Allocator(SEGMENT_SIZE)
        # buf0's live range ends at "mid" (step 1), buf1's starts there --
        # sorted by (start, end, name) as in the real implementation,
        # buf0 allocates first.
        buf0_size = _compute_size_bytes("buf0")
        expected_offsets = {"buf0": expected_alloc.allocate(buf0_size)}
        expected_alloc.free(expected_offsets["buf0"], buf0_size)
        expected_offsets["buf1"] = expected_alloc.allocate(_compute_size_bytes("buf1"))
        expected_pool_size = expected_alloc.get_pool_end()

        buf0 = V.graph.get_buffer("buf0")
        buf1 = V.graph.get_buffer("buf1")
        self.assertEqual(
            buf0.get_layout().allocation["hbm_pool"], expected_offsets["buf0"]
        )
        self.assertEqual(
            buf1.get_layout().allocation["hbm_pool"], expected_offsets["buf1"]
        )
        self.assertEqual(V.graph.hbm_pool_sizes[bundle.get_name()], expected_pool_size)

    def test_counted_loop_accumulator_stays_live_across_the_loop(self):
        """A buffer that is only ever read *inside* a CountedLoopScheduler
        Node's own body (a loop-carried accumulator, e.g. a running-max/sum
        pattern where each iteration reads the value the previous iteration
        wrote) must be treated as live for the whole loop -- not as ending
        right where that internal read/write happens to sit in a fully
        flattened node list.

        CountedLoopSchedulerNode represents a loop that runs loop_count
        times at runtime as a *single* node in its containing bundle's own
        node list (see CountedLoopSchedulerNode.unpack(), which refuses to
        unpack itself). The IR only contains one copy of the loop body, so
        there is no second, later textual read for a "did this survive
        another iteration" check to find -- liveness across iterations has
        to be inferred from the fact that this is a loop, i.e. by treating
        it as one opaque step whose live buffers are live for that entire
        step, not by fully flattening its body into separate timesteps.

        Setup: "init" writes "acc" outside the loop. The loop's body reads
        and writes "acc" (the accumulate step) and separately writes
        "other" (e.g. a per-iteration scratch tile, freshly written on
        every pass). After the loop, "final" reads only "other" -- "acc"
        is never read again outside the loop.

        With the buggy fully-flattened live-range computation, "acc"'s
        only visible read is inside the loop body itself, so its computed
        live range collapses to a single point (start == end) that ends
        before "other"'s start -- the allocator frees "acc"'s block and
        hands the identical byte offset to "other", even though the loop's
        accumulator and the loop's own scratch tile are simultaneously
        live at runtime throughout the loop's execution. With the fix
        (bundle.get_nodes() -- one entry for the whole loop, not one per
        body-op -- passed to _compute_live_ranges), the loop node's merged
        read/write set (ReadWrites.merge_list) drops "acc" entirely, since
        it is both written and read within the same node; with no visible
        read, _compute_live_ranges conservatively keeps "acc" live through
        the rest of the bundle, which safely overlaps "other" and gives the
        two buffers distinct offsets.
        """
        _make_ftl_buffer("acc")
        _make_ftl_buffer("other")

        init = _make_snode_with_rw("init", writes=["acc"], reads=[])
        acc_step = _make_snode_with_rw("acc_step", writes=["acc"], reads=["acc"])
        other_step = _make_snode_with_rw("other_step", writes=["other"], reads=[])
        loop = CountedLoopSchedulerNode(MagicMock(), [acc_step, other_step], Integer(4))
        final = _make_snode_with_rw("final", writes=[], reads=["other"])
        bundle = FusedSchedulerNode(MagicMock(), [init, loop, final])

        hbm_pool_planning([bundle])

        acc_buf = V.graph.get_buffer("acc")
        other_buf = V.graph.get_buffer("other")
        self.assertIn("hbm_pool", acc_buf.get_layout().allocation)
        self.assertIn("hbm_pool", other_buf.get_layout().allocation)
        # "acc" (live throughout the loop) and "other" (written fresh on
        # every pass through the same loop) are simultaneously live, so
        # they must not share the same pool offset.
        self.assertNotEqual(
            acc_buf.get_layout().allocation["hbm_pool"],
            other_buf.get_layout().allocation["hbm_pool"],
        )

    def test_counted_loop_accumulator_stays_live_as_bare_top_level_bundle(self):
        """The opacity fix above relies on the loop appearing as one entry
        inside an outer FusedSchedulerNode's own get_nodes() list. When a
        CountedLoopSchedulerNode has no fusible neighbors, spyre_fuse_nodes's
        _make_fused returns it directly as the top-level bundle (see
        _make_fused in fusion.py: `len(nodes) == 1` returns `nodes[0]`
        unwrapped, not a FusedSchedulerNode of one). In that case `bundle`
        passed to hbm_pool_planning's per-bundle loop *is* the loop itself,
        so `bundle.get_nodes()` would return the loop's own internal snodes
        (acc_step, other_step, other_reader) instead of treating the loop as
        one opaque step.

        Both "acc" and "other" are written and read entirely within the
        loop body (a loop-carried accumulator and a same-iteration temp
        respectively), so both are pool candidates. With the bug, "acc"'s
        merged read/write set still hides its internal read (start=end=0
        collapsed away by the opacity check design), but "other" is read by
        a *distinct* body op (other_reader) at a later flattened index than
        its writer -- exposing a real, non-degenerate live range that ends
        before the loop's remaining iterations complete, so the allocator
        frees and reuses its offset. This reproduces the same class of
        offset collision as the nested-loop test above, via the
        independently-reachable bare-top-level-bundle path.
        """
        _make_ftl_buffer("acc")
        _make_ftl_buffer("other")

        acc_step = _make_snode_with_rw("acc_step", writes=["acc"], reads=["acc"])
        other_step = _make_snode_with_rw("other_step", writes=["other"], reads=[])
        other_reader = _make_snode_with_rw("other_reader", writes=[], reads=["other"])
        loop = CountedLoopSchedulerNode(
            MagicMock(), [acc_step, other_step, other_reader], Integer(4)
        )

        hbm_pool_planning([loop])

        acc_buf = V.graph.get_buffer("acc")
        other_buf = V.graph.get_buffer("other")
        self.assertIn("hbm_pool", acc_buf.get_layout().allocation)
        self.assertIn("hbm_pool", other_buf.get_layout().allocation)
        self.assertNotEqual(
            acc_buf.get_layout().allocation["hbm_pool"],
            other_buf.get_layout().allocation["hbm_pool"],
        )

    def test_buffer_written_in_two_bundles_with_inplace_rewrite_is_not_pool_eligible(
        self,
    ):
        """A buffer written once in bundle A (an initializer) and then
        read-and-rewritten in place in bundle B (an accumulate step, where
        mutation_renames makes the write dep name identical to the read dep
        name) must NOT be pool-eligible in either bundle.

        Regression test for buffer_writer_bundle[name] = bundle_name being a
        plain dict overwrite: bundle B's own read of "accum" is only visible
        within bundle B, so the old _is_cross_bundle reader check
        (readers - {writer}) sees writer == bundle_b, readers == {bundle_b},
        and wrongly treats "accum" as bundle-B-local -- even though bundle A
        also wrote it and needs a stable address for its own write.
        """
        _make_ftl_buffer("accum")
        init = _make_snode_with_rw("init", writes=["accum"], reads=[])
        accumulate = _make_snode_with_rw(
            "accumulate", writes=["accum"], reads=["accum"]
        )
        bundle_a = FusedSchedulerNode(MagicMock(), [init])
        bundle_b = FusedSchedulerNode(MagicMock(), [accumulate])

        hbm_pool_planning([bundle_a, bundle_b])

        buf = V.graph.get_buffer("accum")
        self.assertNotIn("hbm_pool", buf.get_layout().allocation)

    def test_buffer_written_in_two_bundles_with_only_same_bundle_reader_is_not_pool_eligible(
        self,
    ):
        """A buffer written in bundle A and written again in bundle B, where
        the only read anywhere is inside bundle B -- so buffer_reader_bundles
        alone would say "reader set == {writer}", i.e. not cross-bundle by
        the reader-only check -- must still be excluded because it has two
        distinct writer bundles. This isolates the multi-writer signal from
        the reader-based signal entirely: no read in bundle A at all.
        """
        _make_ftl_buffer("accum")
        write_a = _make_snode_with_rw("write_a", writes=["accum"], reads=[])
        write_b = _make_snode_with_rw("write_b", writes=["accum"], reads=[])
        read_b = _make_snode_with_rw("read_b", writes=[], reads=["accum"])
        bundle_a = FusedSchedulerNode(MagicMock(), [write_a])
        bundle_b = FusedSchedulerNode(MagicMock(), [write_b, read_b])

        hbm_pool_planning([bundle_a, bundle_b])

        buf = V.graph.get_buffer("accum")
        self.assertNotIn("hbm_pool", buf.get_layout().allocation)


if __name__ == "__main__":
    unittest.main()
