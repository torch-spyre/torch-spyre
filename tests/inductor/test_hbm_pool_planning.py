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

import os
import unittest
from unittest.mock import MagicMock, patch
from unittest.mock import patch as mock_patch

import regex as re
import torch
from sympy import Integer
from torch import fx
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
import pytest

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.hbm_pool_planning import Allocator, hbm_pool_planning
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.scheduler import CountedLoopSchedulerNode

# Paths to mock for disabling actual device kernel execution.
_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"


class TestAllocator(unittest.TestCase):
    """Unit tests for Allocator.allocate's overflow-fallback behavior."""

    def test_allocate_within_budget_succeeds(self):
        allocator = Allocator(100)
        self.assertEqual(allocator.allocate(60), 0)

    def test_allocate_past_budget_returns_none(self):
        allocator = Allocator(100)
        self.assertIsNone(allocator.allocate(101))

    def test_failed_allocate_leaves_state_untouched(self):
        """A rejected allocate() must not advance pool_end, bump
        currently_allocated/peak_usage, or consume a free block -- the next
        allocate() call sees exactly the state it would have without the
        rejected call ever happening."""
        allocator = Allocator(100)
        first = allocator.allocate(50)
        self.assertEqual(first, 0)

        rejected = allocator.allocate(60)  # 50 + 60 > 100
        self.assertIsNone(rejected)

        # State is exactly as after the first allocate() alone: the next
        # real allocation continues from pool_end=50, not from some
        # partially-applied 110.
        second = allocator.allocate(40)
        self.assertEqual(second, 50)
        self.assertEqual(allocator.get_peak_usage(), 90)
        self.assertEqual(allocator.get_pool_end(), 90)

    def test_failed_allocate_does_not_consume_free_block(self):
        """A rejected allocate() that would have extended pool_end past the
        budget must leave an existing free block untouched for a later,
        smaller request."""
        allocator = Allocator(100)
        allocator.allocate(50)
        allocator.free(0, 50)  # pool_end stays 50; free: [(0, 50)]

        # allocate(60) finds no free block big enough (50 < 60), so it would
        # need to extend pool_end to 110 -- over the 100 budget. Rejected
        # without consuming the (0, 50) free block.
        rejected = allocator.allocate(60)
        self.assertIsNone(rejected)

        # The free block is still available and now fits.
        fits = allocator.allocate(50)
        self.assertEqual(fits, 0)

    def test_exact_boundary_allocation_succeeds(self):
        """An allocation that brings pool_end exactly to the segment size
        limit must succeed, not be rejected."""
        allocator = Allocator(100)
        offset = allocator.allocate(100)
        self.assertEqual(offset, 0)
        self.assertEqual(allocator.get_pool_end(), 100)

    def test_fragmentation_rejected_even_under_concurrent_usage_budget(self):
        """A free/reallocate sequence whose peak concurrent usage never
        exceeds the budget must still be rejected once it would push
        pool_end (the bump-pointer high-water mark) past the segment size
        -- concurrent usage alone is not the quantity generate_bundle's
        assert checks."""
        allocator = Allocator(100)
        first = allocator.allocate(40)
        second = allocator.allocate(40)
        allocator.free(first, 40)
        allocator.free(second, 40)  # currently_allocated back to 0

        # No free block of exactly 50 bytes exists ((0, 40) and (40, 40) are
        # both too small), so this would extend pool_end to 130 -- over
        # budget, even though concurrent usage would only be 50.
        rejected = allocator.allocate(50)
        self.assertIsNone(rejected)
        self.assertEqual(allocator.get_pool_end(), 80)


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


def _make_ftl_buffer_aliased(name, alias_of, host_size=(64,), dim_order=(0,)):
    """Real ComputedBuffer whose FixedTiledLayout.allocation is the exact
    same dict object as `alias_of`'s -- reproduces the aliasing
    MutationLayoutSHOULDREMOVE (or copy-elision) produces when one op's
    write target is another buffer's storage under a different name.
    """
    buf = _make_ftl_buffer(name, host_size, dim_order)
    buf.get_layout().allocation = alias_of.get_layout().allocation
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

    def test_aliased_buffer_across_bundles_is_not_pool_eligible(self):
        """A buffer allocated in bundle A (e.g. a `full()` accumulator) that
        is later mutated in bundle B under a *different* name -- because
        MutationLayoutSHOULDREMOVE/copy-elision makes the mutating op's
        FixedTiledLayout.allocation the same dict object as the original
        buffer's, not merely dependency-linked to it by name -- must not be
        pool-eligible in either bundle.

        Regression test for issue #3775: bundle B's own candidacy check for
        "alias" sees a normal, single-bundle-local write+read and pool-
        allocates it, silently mutating the shared allocation dict. That
        retroactively assigns "target" (bundle A, never itself a candidate)
        a stale/foreign offset that is out of bounds against bundle A's own
        pool size, since bundle A's allocator never accounted for it. The
        old _is_cross_bundle check is purely name-keyed and cannot see this:
        "target" and "alias" share no read/write dependency edge at all.
        """
        target = _make_ftl_buffer("target")
        write_target = _make_snode_with_rw("write_target", writes=["target"], reads=[])
        bundle_a = FusedSchedulerNode(MagicMock(), [write_target])

        _make_ftl_buffer_aliased("alias", alias_of=target)
        write_alias = _make_snode_with_rw("write_alias", writes=["alias"], reads=[])
        read_alias = _make_snode_with_rw("read_alias", writes=[], reads=["alias"])
        bundle_b = FusedSchedulerNode(MagicMock(), [write_alias, read_alias])

        hbm_pool_planning([bundle_a, bundle_b])

        target_buf = V.graph.get_buffer("target")
        alias_buf = V.graph.get_buffer("alias")
        self.assertNotIn("hbm_pool", target_buf.get_layout().allocation)
        self.assertNotIn("hbm_pool", alias_buf.get_layout().allocation)

    def test_aliased_buffer_within_same_bundle_shares_one_merged_live_range(self):
        """Regression test for issue #3980: two buffer *names* that share
        the exact same FixedTiledLayout.allocation dict object -- as
        enforce_indirect_access_layout.py's _insert_mutation_relayout_copy
        produces for a non-compliant scatter destination, via
        propagate_layouts.py's propagate_mutation_layouts resolving
        MutationLayoutSHOULDREMOVE.real_layout() to the *same* layout
        instance as its copy-in buffer's -- are truly one physical storage
        location. Pool-allocating them safely (rather than excluding them
        outright, which would regress coarse_tiling's pervasive use of
        mutation buffers) requires treating their live range as the union
        of both names' individual ranges, not each in isolation.

        Before the fix, "target" and "alias" (sharing one allocation dict)
        were each given their own live range from their own reads/writes
        and separately allocated -- two distinct offsets written into the
        one shared dict, the second clobbering the first. The fix merges
        alias-group members into one (min start, max end) range and
        allocates once per group.

        This test isolates the *merge* itself (not just the single-write
        aliasing, which would trivially "pass" even unfixed since both
        names share one dict): "target" is read late (read_target, step 3)
        while "alias" is written+read early (steps 1-2). "other" is written
        and read in between (step 2's write, step 2's read) -- overlapping
        alias's naive [1,2] range but *not* target's own naive [0,0] range.
        Only the merged [0,3] range correctly keeps "other" from reusing
        target/alias's block while target is still live.
        """
        target = _make_ftl_buffer("target", host_size=(64,))
        write_target = _make_snode_with_rw("write_target", writes=["target"], reads=[])

        _make_ftl_buffer_aliased("alias", alias_of=target, host_size=(64,))
        write_alias = _make_snode_with_rw("write_alias", writes=["alias"], reads=[])
        read_alias = _make_snode_with_rw("read_alias", writes=[], reads=["alias"])

        _make_ftl_buffer("other", host_size=(64,))
        write_other = _make_snode_with_rw("write_other", writes=["other"], reads=[])
        read_other = _make_snode_with_rw("read_other", writes=[], reads=["other"])

        read_target = _make_snode_with_rw("read_target", writes=[], reads=["target"])

        # Steps: 0=write_target, 1=write_alias, 2=[read_alias, write_other,
        # read_other] (fused into one timestep), 3=read_target.
        step2 = FusedSchedulerNode(MagicMock(), [read_alias, write_other, read_other])
        bundle = FusedSchedulerNode(
            MagicMock(), [write_target, write_alias, step2, read_target]
        )

        hbm_pool_planning([bundle])

        target_buf = V.graph.get_buffer("target")
        alias_buf = V.graph.get_buffer("alias")
        other_buf = V.graph.get_buffer("other")

        # Both aliased names are pool-eligible now (same-bundle aliasing no
        # longer bars pool allocation outright)...
        self.assertIn("hbm_pool", target_buf.get_layout().allocation)
        self.assertIn("hbm_pool", alias_buf.get_layout().allocation)
        self.assertIn("hbm_pool", other_buf.get_layout().allocation)
        # ...sharing the exact same offset, since they are one physical
        # storage location (same underlying allocation dict).
        self.assertEqual(
            target_buf.get_layout().allocation["hbm_pool"],
            alias_buf.get_layout().allocation["hbm_pool"],
        )
        # "other" must not reuse target/alias's block: with the naive
        # (unmerged) per-name ranges, alias's range [1, 2] would end before
        # other's start [2], wrongly freeing the block for reuse while
        # target (whose own real read is at step 3) still needs it.
        self.assertNotEqual(
            target_buf.get_layout().allocation["hbm_pool"],
            other_buf.get_layout().allocation["hbm_pool"],
        )

    def test_buffer_that_overflows_pool_falls_back_to_standalone_hbm(self):
        """When MAX_POOL_SIZE_BYTES is too small to hold every pool-eligible
        buffer, the ones that fit get an hbm_pool allocation and the
        overflow buffer(s) fall back to standalone HBM (no hbm_pool key)
        instead of raising.

        buf0 (128 bytes) and buf1 (256 bytes) have non-overlapping live
        ranges but the allocator restarts a fresh Allocator per bundle, so
        with MAX_POOL_SIZE_BYTES patched to 200 bytes, buf0 fits (128 <= 200) but
        buf1 does not (256 > 200): the very first allocation call already
        exceeds budget for buf1 regardless of live-range-driven reuse.
        """
        _make_ftl_buffer("buf0", host_size=(64,))
        _make_ftl_buffer("buf1", host_size=(128,))
        w0 = _make_snode_with_rw("w0", writes=["buf0"], reads=[])
        r0 = _make_snode_with_rw("r0", writes=[], reads=["buf0"])
        w1 = _make_snode_with_rw("w1", writes=["buf1"], reads=[])
        r1 = _make_snode_with_rw("r1", writes=[], reads=["buf1"])
        bundle = FusedSchedulerNode(MagicMock(), [w0, r0, w1, r1])

        with patch("torch_spyre._inductor.hbm_pool_planning.MAX_POOL_SIZE_BYTES", 200):
            hbm_pool_planning([bundle])

        buf0 = V.graph.get_buffer("buf0")
        buf1 = V.graph.get_buffer("buf1")
        self.assertIn("hbm_pool", buf0.get_layout().allocation)
        self.assertNotIn("hbm_pool", buf1.get_layout().allocation)
        # The bundle still gets a pool-size entry reflecting only what was
        # actually placed in the pool (buf0), not the buffer that overflowed.
        self.assertEqual(V.graph.hbm_pool_sizes[bundle.get_name()], 128)

    def test_all_buffers_overflow_pool_produces_zero_size_pool(self):
        """If MAX_POOL_SIZE_BYTES is too small for even the first buffer, every
        pool-eligible buffer in the bundle falls back to standalone HBM and
        the bundle's recorded pool extent is 0 -- no RuntimeError."""
        _make_ftl_buffer("buf0", host_size=(64,))
        writer = _make_snode_with_rw("writer", writes=["buf0"], reads=[])
        reader = _make_snode_with_rw("reader", writes=[], reads=["buf0"])
        bundle = FusedSchedulerNode(MagicMock(), [writer, reader])

        with patch("torch_spyre._inductor.hbm_pool_planning.MAX_POOL_SIZE_BYTES", 1):
            hbm_pool_planning([bundle])

        buf0 = V.graph.get_buffer("buf0")
        self.assertNotIn("hbm_pool", buf0.get_layout().allocation)
        self.assertEqual(V.graph.hbm_pool_sizes[bundle.get_name()], 0)

    def test_freed_space_is_reused_before_hitting_pool_budget(self):
        """A buffer that fits only because an earlier buffer's live range
        already ended (and its block was freed) must still get pool
        allocated -- the fallback path must not trigger for buffers that
        fit via reuse, only for genuine budget overflow."""
        _make_ftl_buffer("buf0", host_size=(64,))  # 128 bytes
        _make_ftl_buffer("buf1", host_size=(64,))  # 128 bytes
        w0 = _make_snode_with_rw("w0", writes=["buf0"], reads=[])
        r0 = _make_snode_with_rw("r0", writes=[], reads=["buf0"])
        w1 = _make_snode_with_rw("w1", writes=["buf1"], reads=[])
        r1 = _make_snode_with_rw("r1", writes=[], reads=["buf1"])
        # buf0's live range [0, 1] ends before buf1's starts at [2, 3], so
        # buf1 can reuse buf0's freed 128-byte block instead of needing a
        # fresh 128 bytes on top -- a pool budget of 128 is only enough for
        # one live buffer at a time, not both concurrently.
        bundle = FusedSchedulerNode(MagicMock(), [w0, r0, w1, r1])

        with patch("torch_spyre._inductor.hbm_pool_planning.MAX_POOL_SIZE_BYTES", 128):
            hbm_pool_planning([bundle])

        buf0 = V.graph.get_buffer("buf0")
        buf1 = V.graph.get_buffer("buf1")
        self.assertIn("hbm_pool", buf0.get_layout().allocation)
        self.assertIn("hbm_pool", buf1.get_layout().allocation)
        self.assertEqual(
            buf0.get_layout().allocation["hbm_pool"],
            buf1.get_layout().allocation["hbm_pool"],
        )


class TestHbmPoolPlanningE2E(InductorTestCase):
    """End-to-end compiled-path tests for hbm_pool threading into codegen."""

    @config.patch({"lx_planning": False})
    def test_bundle_pool_size_threaded_from_hbm_pool_sizes(self):
        """codegen_node must look up this bundle's own pool_size from
        V.graph.hbm_pool_sizes, not a stale graph-global scalar.

        lx_planning is disabled here so the `a = x + y` intermediate isn't
        claimed by LX scratchpad planning first -- with LX planning on,
        `add`/`mul`/`sub` outputs are all LX-eligible by default (see
        OP_OUTPUT_NOT_GOOD_FOR_LX_REUSE in scratchpad/utils.py) and may win the
        scratchpad before hbm_pool_planning ever sees them, leaving every
        bundle's pool_size at 0 and proving nothing about the plumbing this
        test exists to check.
        """
        from torch_spyre._inductor.spyre_kernel import SpyreKernel

        seen_pool_sizes = []
        orig_init = SpyreKernel.__init__

        def _recording_init(self, pool_size=0, **kwargs):
            seen_pool_sizes.append(pool_size)
            orig_init(self, pool_size=pool_size, **kwargs)

        def fn(x, y):
            a = x + y
            b = a * 2
            return b - x

        x = torch.randn(64, 64, dtype=torch.float16, device="spyre")
        y = torch.randn(64, 64, dtype=torch.float16, device="spyre")

        with (
            mock_patch.object(SpyreKernel, "__init__", _recording_init),
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            torch.compile(fn)(x, y)

        self.assertTrue(seen_pool_sizes)
        self.assertTrue(
            any(seen_pool_sizes),
            f"expected at least one bundle with a nonzero pool_size, got "
            f"{seen_pool_sizes}",
        )

    @config.patch({"lx_planning": False})
    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_pool_alloc_scoped_per_bundle_across_fallback_boundary(self):
        """A CPU-fallback op (torch.sin) splits the graph into multiple
        bundles. With frontend_pool_allocation at its default (False), each
        bundle's pool (if any) is allocated inside that bundle's own
        generated MLIR via sdscbundle.device_mem_allocate -- there is no
        Python-side pool tensor, and no bundle's MLIR references another
        bundle's pool."""
        from torch_spyre.execution import async_compile as async_compile_mod

        def fn(t):
            a = torch.exp(t) * 2  # compiled bundle 1; `a` crosses the
            b = torch.sin(a)  # fallback op -- forces a bundle boundary
            c = torch.exp(b) * 2  # compiled bundle 2
            return c

        x = torch.randn(64, 64, dtype=torch.float16, device="spyre")

        # get_output_dir() mints a fresh random tempdir on every call (see
        # its uuid4()-based implementation), so it cannot be re-invoked from
        # the test to recover the directory sdsc() actually used to write
        # bundle.mlir. Wrap it to record kernel_name -> output_dir while
        # still delegating to the real implementation.
        real_get_output_dir = async_compile_mod.get_output_dir
        output_dirs_by_kernel = {}

        def _recording_get_output_dir(kernel_name):
            output_dir = real_get_output_dir(kernel_name)
            output_dirs_by_kernel[kernel_name] = output_dir
            return output_dir

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
            mock_patch.object(
                async_compile_mod, "get_output_dir", _recording_get_output_dir
            ),
            pytest.warns(UserWarning),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn), x)
        src = source_codes[0]

        # No Python-side pool tensor of any kind remains in the wrapper.
        # Ordinary output buffers legitimately use spyre_empty_with_layout,
        # so distinguish pool allocs by their telltale uint8 dtype, same as
        # test_no_python_side_pool_tensor_allocated below.
        pool_alloc_lines = [
            line
            for line in src.splitlines()
            if "spyre_empty_with_layout" in line and "uint8" in line
        ]
        self.assertEqual(pool_alloc_lines, [])
        # Match the actual pool-variable assignment pattern
        # (call_kernel()'s pool_var_name = f"_pool_{name}"), not a bare
        # substring search -- this file's own path
        # (test_hbm_pool_planning.py) contains "_pool_" and would otherwise
        # false-positive via an embedded SourceLoc debug handle.
        self.assertIsNone(re.search(r"\b_pool_\w+\s*=", src))

        # Each kernel name mentioned in an async_compile.sdsc(...) call gets
        # its own bundle.mlir; any that uses a pool must self-allocate it via
        # device_mem_allocate, never via a %pool_base_addr parameter.
        kernel_names = re.findall(r"async_compile\.sdsc\('(\w+)'", src)
        self.assertTrue(kernel_names)
        saw_pool_allocate = False
        for kernel_name in kernel_names:
            self.assertIn(kernel_name, output_dirs_by_kernel)
            bundle_path = os.path.join(
                output_dirs_by_kernel[kernel_name], "bundle.mlir"
            )
            with open(bundle_path) as f:
                bundle_text = f.read()
            self.assertNotIn("%pool_base_addr", bundle_text)
            if "device_mem_allocate" in bundle_text:
                saw_pool_allocate = True
        self.assertTrue(
            saw_pool_allocate,
            f"expected at least one bundle to use device_mem_allocate, "
            f"kernels were {kernel_names}",
        )

    @config.patch({"lx_planning": False})
    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_no_python_side_pool_tensor_allocated(self):
        """With frontend_pool_allocation at its default (False), no bundle
        allocates a Python-side _pool_<name> tensor -- pool allocation is
        entirely inside the generated MLIR."""

        def fn(t):
            a = torch.exp(t) * 2
            b = torch.sin(a)  # fallback op -- forces a bundle boundary
            c = torch.exp(b) * 2
            return c

        x = torch.randn(64, 64, dtype=torch.float16, device="spyre")

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
            pytest.warns(UserWarning),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn), x)
        src = source_codes[0]

        # No Python-side pool tensor allocation (uint8 SpyreTensorLayout) --
        # ordinary output buffers still legitimately use
        # spyre_empty_with_layout, so distinguish pool allocs by their
        # telltale uint8 dtype, same as
        # test_pool_alloc_scoped_per_bundle_across_fallback_boundary above.
        pool_alloc_lines = [
            line
            for line in src.splitlines()
            if "spyre_empty_with_layout" in line and "uint8" in line
        ]
        self.assertEqual(pool_alloc_lines, [])
        # Match the actual pool-variable assignment pattern
        # (call_kernel()'s pool_var_name = f"_pool_{name}"), not a bare
        # substring search -- this file's own path
        # (test_hbm_pool_planning.py) contains "_pool_" and would otherwise
        # false-positive via an embedded SourceLoc debug handle.
        self.assertIsNone(re.search(r"\b_pool_\w+\s*=", src))

    @config.patch({"lx_planning": False})
    def test_pool_size_kwarg_in_generated_sdsc_call(self):
        """define_kernel() must append pool_size=<N> to the generated
        async_compile.sdsc(...) call text for a kernel whose pool_size > 0,
        and pool_size=0 must never be emitted explicitly. See
        test_pool_size_kwarg_omitted_when_no_pool below for the omission
        case on a kernel with no pool usage at all."""

        def fn(x, y):
            a = x + y
            b = a * 2
            return b - x

        x = torch.randn(64, 64, dtype=torch.float16, device="spyre")
        y = torch.randn(64, 64, dtype=torch.float16, device="spyre")

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn), x, y)
        src = source_codes[0]

        self.assertIn("pool_size=", src)
        # Every async_compile.sdsc( call either has no pool_size kwarg, or a
        # positive one -- pool_size=0 must never be emitted explicitly.
        self.assertNotIn("pool_size=0", src)

    @config.patch({"lx_planning": True})
    def test_pool_size_kwarg_omitted_when_no_pool(self):
        """A kernel with no pool usage gets no pool_size kwarg at all.

        With lx_planning enabled, the `a = x + y` intermediate is claimed by
        LX scratchpad planning before hbm_pool_planning ever sees it (see
        OP_OUTPUT_NOT_GOOD_FOR_LX_REUSE in scratchpad/utils.py), so this bundle
        has no pool-eligible buffer and define_kernel() must omit the
        pool_size kwarg entirely rather than emit pool_size=0.
        """

        def fn(x, y):
            a = x + y
            b = a * 2
            return b - x

        x = torch.randn(64, 64, dtype=torch.float16, device="spyre")
        y = torch.randn(64, 64, dtype=torch.float16, device="spyre")

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn), x, y)
        src = source_codes[0]

        self.assertIn("async_compile.sdsc(", src)
        self.assertNotIn("pool_size", src)

    @config.patch({"lx_planning": False})
    def test_alias_read_in_another_bundle_blocks_pool_eligibility(self):
        """A co-allocated sibling read by another bundle must keep the whole
        allocation off the pool.

        copy_forced's lowering builds a MutationLayoutSHOULDREMOVE buffer, so
        several names share one layout.allocation dict. Assigning a pool offset
        writes into that shared dict, relocating EVERY name on it -- including
        names that were never pool candidates and so were never passed to
        _is_cross_bundle. When such a sibling is read by another bundle, that
        bundle addresses it pool-relative while owning no pool, and
        generate_bundle asserts `pool_size=0 out of range ... for a bundle with
        a pool symbol present`.

        Three names on one allocation are required, which is why `acc` is
        written by TWO copy_forced calls: the zeros buffer and the first copy's
        result are written and read entirely inside the first bundle, so both
        are candidates; the second copy's result is read by the cat bundle, so
        it is correctly rejected -- and then dragged into the pool anyway by
        the two that were accepted. This mirrors sliding-window attention,
        where `output` is the dst of both the accumulator update and the final
        divide, and the per-block results are read by torch.cat.
        """

        def fn(x, y):
            outs = []
            for i in range(2):
                acc = torch.zeros_like(x)
                acc = torch.ops.spyre.copy_forced(x * (i + 1) + y, acc)
                acc = torch.ops.spyre.copy_forced(acc * 2, acc)
                outs.append(acc)
            return torch.cat(outs, dim=0)

        x = torch.randn(64, 64, dtype=torch.float16, device="spyre")
        y = torch.randn(64, 64, dtype=torch.float16, device="spyre")

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            # Without the alias-read guard this raises InductorError from
            # generate_bundle's pool_size assertion.
            torch.compile(fn)(x, y)


if __name__ == "__main__":
    unittest.main()
