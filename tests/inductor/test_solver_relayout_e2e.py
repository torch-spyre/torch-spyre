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

"""End-to-end: a solver-decided LX relayout materializes and runs correctly.

Pointwise ops share their candidate divisions, so on an unconstrained graph a
slicing MATCH is always available for free and the solver would rationally
never shuffle. To exercise the decision deterministically, the edge relation
is held false on every edge (monkeypatch), making a relayout the only route to
residency; the real cost objective then chooses it, because the shuffle
(~1-9 us depending on the chosen geometry) beats spilling the 2.10 MB
intermediate through HBM.

Asserts the whole chain: the solver fires the edge, the commit path rebuilds
the plan (views under the chosen divisions, solved addresses), the existing
``materialize_lx_relayouts`` inserts the shuffle, and the compiled graph
computes the right numbers on device.
"""

import pytest

pytest.importorskip("ortools")

import torch

import torch_spyre  # noqa: F401
from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad import allocator as alloc_mod
from torch_spyre._inductor.scratchpad.graph_builder import GraphBuilder


_COOPT = {"co_optimizing_lx_planning": True}


def _no_relation(self, edge, lit, model):
    """No division pair slices an edge's buffer alike: the derived relation's
    form of an empty pair table."""
    model.Add(lit == 0)


class _Observed:
    """Observe the whole relayout chain of one compile: the plans handed to
    materialize_lx_relayouts, any post-planning demotion of a relayout group
    by the scheduler, and every identity op codegen classified as an LX
    relayout (both args LX-resident with distinct work divisions) together
    with the LX addresses it emitted.

    With ``force=True`` no division pair matches on any edge (the strategy's
    relation is held false), so a relayout is the only route to residency (see
    the module docstring).
    With ``force=False`` the graph is compiled as a user would see it and
    only read-only spies are installed.
    """

    def __init__(self, monkeypatch, *, force: bool):
        import torch_spyre._inductor.codegen.superdsc as sdsc_mod
        import torch_spyre._inductor.scheduler as sched_mod
        import torch_spyre._inductor.spyre_kernel as sk_mod
        from torch_spyre._inductor import op_spec as opspec_mod

        self.plans: list = []
        self.demotions: list = []
        self.emitted: set[tuple[int, int]] = set()  # (src, dst) LX addresses
        # One entry per emitted LX relayout op, recorded where spyre_kernel
        # finalizes each op spec (exactly once per emitted op; superdsc's
        # parse_op_spec classifies the same op twice, so it is not counted
        # there). A list, not an address set, so two shuffles of one view that
        # land at the same offset in different ticks are still two.
        self.emitted_ops: list[tuple[int, int]] = []
        # plan edge -> materialized copy buffer name, and each consumer's read
        # names right after materialization (V.graph is live inside the pass).
        self.copies: dict = {}
        self.consumer_reads: dict[str, set[str]] = {}
        real_materialize = alloc_mod.materialize_lx_relayouts

        def spy_materialize(graph, plans):
            from torch_spyre._inductor.pass_utils import op_read_writes
            from torch_spyre._inductor.scratchpad.lx_relayout import (
                materialized_lx_relayouts,
            )

            self.plans.extend(plans)
            result = real_materialize(graph, plans)
            self.copies = {
                edge: copy_name
                for edge, (copy_name, _plan) in materialized_lx_relayouts(graph).items()
            }
            for plan in plans:
                for name in plan.consumer_names:
                    op = graph.get_buffer(name)
                    self.consumer_reads[name] = {
                        d.name for d in op_read_writes(op).reads if hasattr(d, "name")
                    }
            return result

        real_demote = sched_mod.demote_lx_relayout_group

        def spy_demote(graph, source_name, reason):
            self.demotions.append((source_name, reason))
            return real_demote(graph, source_name, reason)

        real_identity = opspec_mod.is_lx_relayout_identity

        def _check(args):
            source, destination = args
            assert set(source.allocation) == {"lx"}, source.allocation
            assert set(destination.allocation) == {"lx"}, destination.allocation
            pair = (source.allocation["lx"], destination.allocation["lx"])
            self.emitted.add(pair)
            return pair

        def spy_identity_codegen(op, args, op_info=None):
            result = real_identity(op, args, op_info)
            if result:
                _check(args)
            return result

        def spy_identity_kernel(op, args, op_info=None):
            result = real_identity(op, args, op_info)
            if result:
                self.emitted_ops.append(_check(args))
            return result

        monkeypatch.setattr(alloc_mod, "materialize_lx_relayouts", spy_materialize)
        monkeypatch.setattr(sched_mod, "demote_lx_relayout_group", spy_demote)
        monkeypatch.setattr(sdsc_mod, "is_lx_relayout_identity", spy_identity_codegen)
        monkeypatch.setattr(sk_mod, "is_lx_relayout_identity", spy_identity_kernel)
        if force:
            from torch_spyre._inductor.scratchpad.encoding_strategy import (
                EnumeratedStrategy,
            )
            from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy

            for strategy in (EnumeratedStrategy, LazyStrategy):
                monkeypatch.setattr(strategy, "_relate", _no_relation)
        torch._inductor.codecache.FxGraphCache.clear()
        torch._dynamo.reset()

    def assert_emitted_in_lx(self, expected_plans: int) -> None:
        """Every planned shuffle survived scheduling and was emitted by codegen
        as an LX relayout at the solver's addresses."""
        assert len(self.plans) == expected_plans, (
            f"expected {expected_plans} plan(s): "
            f"{[(p.source_name, p.consumer_names) for p in self.plans]}"
        )
        assert self.demotions == [], (
            f"a relayout group was demoted after planning: {self.demotions}"
        )
        planned = {(p.source_address, p.destination_address) for p in self.plans}
        assert planned == self.emitted, (
            f"planned {sorted(planned)} but codegen emitted {sorted(self.emitted)}"
        )
        # One shuffle op per plan: a bridged segment is ONE relayout however
        # many consumers read it, and two segments are two even if they reuse
        # an offset across ticks.
        assert len(self.emitted_ops) == expected_plans, (
            f"expected {expected_plans} emitted relayout op(s), codegen emitted "
            f"{len(self.emitted_ops)}: {sorted(self.emitted_ops)}"
        )
        # Every consumer of a plan reads the plan's materialized copy: that is
        # what "the bridge is used" means at the graph level.
        for plan in self.plans:
            copy_name = self.copies[plan.edge]
            for consumer in plan.consumer_names:
                assert copy_name in self.consumer_reads[consumer], (
                    f"{consumer} does not read the relayout copy {copy_name} "
                    f"(reads {sorted(self.consumer_reads[consumer])})"
                )

    def assert_nothing_emitted(self) -> None:
        """No plan, no demotion, and no identity op codegen took for an LX
        relayout: the compile ran exactly as it would without the feature."""
        assert self.plans == [], f"no relayout may materialize: {self.plans}"
        assert self.demotions == []
        assert self.emitted == set() and not self.emitted_ops, (
            f"codegen emitted an LX relayout with the feature disarmed: "
            f"{sorted(self.emitted)}"
        )


def _arm_forced(monkeypatch) -> _Observed:
    return _Observed(monkeypatch, force=True)


def test_solver_relayout_materializes_and_runs(monkeypatch):
    observed = _arm_forced(monkeypatch)
    recorded = observed.plans

    def fn(t):
        return torch.relu(torch.neg(t))

    torch.manual_seed(0)
    host = torch.randn(8, 256, 512, dtype=torch.float16)
    x = host.to("spyre")
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False)(x)

    # The solver fired the neg -> relu edge, the commit path handed
    # materialize_lx_relayouts a complete plan, the scheduler kept the group
    # in LX, and codegen emitted the shuffle at the solver's addresses.
    observed.assert_emitted_in_lx(expected_plans=1)
    plan = recorded[0]
    assert plan.source_view != plan.destination_view
    assert plan.source_address is not None
    assert plan.destination_address is not None
    assert plan.source_address != plan.destination_address

    # And the shuffle computes the right thing on device (fp16 round trips
    # are 1-ULP by design, so compare with tolerance, never bit-exact).
    ref = torch.relu(torch.neg(host.float())).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


def test_relayouts_off_materializes_nothing(monkeypatch):
    """With ``lx_planner_relayout`` off no candidate is enumerated, and the
    co-optimizing solve is exactly what it was before relayouts."""
    observed = _arm_forced(monkeypatch)

    def fn(t):
        return torch.relu(torch.neg(t)) + 1.0

    torch.manual_seed(0)
    host = torch.randn(8, 256, 512, dtype=torch.float16)
    x = host.to("spyre")
    with config.patch({**_COOPT, "lx_planner_relayout": False}):
        out = torch.compile(fn, dynamic=False)(x)

    observed.assert_nothing_emitted()
    # The +1.0 makes this graph distinct from the first test's; it also makes
    # the device round an intermediate in SEN169 fp16 (1-6-9: 9 mantissa bits)
    # where the fp32 reference does not, so the comparison must allow 1 ULP:
    # 2**-9 relative, and 2**-7 absolute for values in [4, 8).
    ref = (torch.relu(torch.neg(host.float())) + 1.0).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)


def test_relayout_fires_naturally_on_a_hinted_graph(monkeypatch):
    """The measurement harness's own graph, un-forced: neg hinted {B:4, M:2}
    feeding relu hinted {B:2, M:4}. Co-opt candidates honor work_div hints
    (user hints take ownership of the split decision), so the two ops are
    pinned to genuinely different divisions: no match exists, the relayout
    table holds exactly the canonical measured pair, and the solver fires it
    on economics alone (~8.8 us shuffle vs ~80 us demoted round trip). The
    only patches below are read-only spies."""
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint

    observed = _Observed(monkeypatch, force=False)
    recorded = observed.plans

    def fn(t):
        with spyre_hint(work_div={"B": 4, "M": 2}):
            hidden = torch.neg(t)
        with spyre_hint(work_div={"B": 2, "M": 4}):
            return torch.relu(hidden)

    torch.manual_seed(0)
    host = torch.randn(8, 256, 512, dtype=torch.float16)
    for name, size in (("B", 8), ("M", 256), ("K", 512)):
        _pnd.declare_tensor_dim(name, size)
    x = _pnd.name_tensor_dims(host.to("spyre"), ["B", "M", "K"])
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False)(x)

    observed.assert_emitted_in_lx(expected_plans=1)
    plan = recorded[0]
    # The canonical measured pair: producer {B:4, M:2} vs consumer {B:2, M:4}
    # on device [256, 8, 8, 64] (dim 0 <- M, dim 2 <- B).
    assert dict(plan.source_view.work_slice_dims) == {0: 2, 2: 4}
    assert dict(plan.destination_view.work_slice_dims) == {0: 4, 2: 2}
    assert plan.num_cores == 8

    # The full per-core ownership mapping, via the same authority the
    # compatibility gate uses. Source: core c owns M-half floor(c/4) and
    # B-quarter c%4; destination: M-quarter floor(c/2) and B-half c%2.
    from torch_spyre._inductor.scratchpad.lx_relayout import _core_slices

    assert _core_slices(plan.source_view, 8) == {
        c: {0: c // 4, 2: c % 4} for c in range(8)
    }
    assert _core_slices(plan.destination_view, 8) == {
        c: {0: c // 2, 2: c % 2} for c in range(8)
    }
    # Addresses are solver placement choices (assert legality, not values):
    # per-core footprint is 2.10 MB / 8 = 262144 B per side, both slices
    # resident simultaneously and disjoint.
    per_core = 2097152 // 8
    lo, hi = sorted((plan.source_address, plan.destination_address))
    assert lo + per_core <= hi, "source and destination overlap in LX"
    ref = torch.relu(torch.neg(host.float())).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("fragments", [8, 10])
def test_gather_respects_shuffle_register_budget(monkeypatch, fragments):
    """Keep an eight-way LX gather; replan a ten-way gather through HBM.

    Gemma's row/column producer feeding a row-only matmul operand used to
    reach Deeptools with ten mandatory L3LU bound-register candidates. The
    hints reproduce that geometry without forcing the solver's match table.
    """
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint

    observed = _Observed(monkeypatch, force=False)
    width = fragments * 256

    def fn(x, w):
        with spyre_hint(work_div={"M": 2, "K": fragments}):
            hidden = -x
        with spyre_hint(work_div={"M": 2, "N": fragments}):
            return torch.mm(hidden, w)

    torch.manual_seed(123)
    hx = torch.randn(256, width, dtype=torch.float16) * 0.1
    hw = torch.randn(width, width, dtype=torch.float16) * 0.1
    for name, size in (("M", 256), ("K", width), ("N", width)):
        _pnd.declare_tensor_dim(name, size)
    x = _pnd.name_tensor_dims(hx.to("spyre"), ["M", "K"])
    w = _pnd.name_tensor_dims(hw.to("spyre"), ["K", "N"])
    with config.patch(_COOPT):
        out = torch.compile(fn, fullgraph=True, dynamic=False)(x, w)

    if fragments == 8:
        observed.assert_emitted_in_lx(expected_plans=1)
    else:
        observed.assert_nothing_emitted()
    torch.testing.assert_close(out.cpu(), fn(hx, hw), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("kind", ["gather", "broadcast"])
def test_grouped_relayout_fires_under_cpsat(monkeypatch, kind):
    """#3440's grouped fixtures decided by the solver instead of the greedy
    collector (see ``relayout_fixtures.grouped_relayout_graph``). Hints pin both
    divisions, so no slicing match exists and the solver either buys the
    grouped movement or demotes ``hidden`` to HBM, where every replica core
    re-reads its slice (#4454 prices that). The only patches are read-only
    spies.

    Frontend checks (no native-output validation):
    - assert_emitted_in_lx verifies that exactly one plan was produced, no
    group was demoted by the scheduler, the solver's planned (source,
    destination) LX address pair matches the pair emitted by codegen, exactly
    one shuffle op was recorded by spyre_kernel, and every plan consumer
    reads the materialized copy buffer.
    - plan.num_cores and destination_view.num_cores match the fixture's
    source_cores / destination_cores expectations, verifying that the
    solver preserves the core counts from the grouped graph.
    - _core_slices owner-set sizes match src_owners / dst_owners,
    verifying the physical partitioning at the source and destination.

    Numerical check:
    torch.testing.assert_close compares the device output, moved to CPU
    float, against the fixture reference (rtol=2e-2, atol=2e-1)."""

    from torch_spyre._inductor.scratchpad.lx_relayout import _core_slices
    from relayout_fixtures import grouped_relayout_graph

    observed = _Observed(monkeypatch, force=False)
    fn, args, reference, expect = grouped_relayout_graph(kind)
    with config.patch({**_COOPT, "sencores": 32}):
        out = torch.compile(fn, dynamic=False, options={"epilogue_fusion": False})(
            *args
        )

    observed.assert_emitted_in_lx(expected_plans=1)
    (plan,) = observed.plans
    assert plan.num_cores == expect["source_cores"], "the plan keeps the source's"
    assert plan.destination_view.num_cores == expect["destination_cores"]
    owners = lambda view, n: len(  # noqa: E731
        {tuple(sorted(r.items())) for r in _core_slices(view, n).values()}
    )
    assert owners(plan.source_view, expect["source_cores"]) == expect["src_owners"]
    assert (
        owners(plan.destination_view, expect["destination_cores"])
        == expect["dst_owners"]
    )
    torch.testing.assert_close(out.cpu().float(), reference, rtol=2e-2, atol=2e-1)


def test_relayout_into_a_matmul_x_operand(monkeypatch):
    """The edge gate admits a matmul consumer; the destination view must then
    satisfy the matmul's own division rules. Pointwise producer feeding x."""
    forced = _arm_forced(monkeypatch)
    recorded = forced.plans

    def fn(t, w):
        return torch.neg(t) @ w

    torch.manual_seed(0)
    ht = torch.randn(8, 256, 512, dtype=torch.float16)
    hw = torch.randn(512, 128, dtype=torch.float16) * 0.05
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False)(ht.to("spyre"), hw.to("spyre"))

    forced.assert_emitted_in_lx(expected_plans=1)
    plan = recorded[0]
    assert plan.source_view != plan.destination_view
    assert plan.source_address != plan.destination_address
    ref = torch.neg(ht.float()) @ hw.float()
    # fp16 K=512 accumulation on device vs fp32 reference.
    torch.testing.assert_close(out.cpu().float(), ref, rtol=2e-2, atol=2e-1)


def test_permutation_past_split_8_fires(monkeypatch):
    """The row-split normalization geometry on a hinted pointwise pair: neg
    hinted {R: 32} (16 whole rows per core, as a row reduction needs) feeds
    relu hinted {R: 4, H: 8}. Logical [512, 4096] fp16 lays out as
    [64 sticks, 512 rows, 64], so the pair's governing split is 32 (a run of
    16 * 64 elements); both views are complete partitions of the 32 cores, so
    the move is a permutation, and each destination gathers its 128 rows from
    8 sources, inside the shuffle budget. The relayout law is measured for
    permutations up to split 32, so the solver is offered the pair at the law's
    price and fires it on economics alone (a shuffle of a few microseconds
    against a 4 MB HBM write and re-read). Read-only spies."""
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.dump_cost_model import governing_run_split

    observed = _Observed(monkeypatch, force=False)
    recorded = observed.plans

    def fn(t):
        with spyre_hint(work_div={"R": 32}):
            hidden = torch.neg(t)
        with spyre_hint(work_div={"R": 4, "H": 8}):
            return torch.relu(hidden)

    torch.manual_seed(0)
    host = torch.randn(512, 4096, dtype=torch.float16)
    for name, size in (("R", 512), ("H", 4096)):
        _pnd.declare_tensor_dim(name, size)
    x = _pnd.name_tensor_dims(host.to("spyre"), ["R", "H"])
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False)(x)

    observed.assert_emitted_in_lx(expected_plans=1)
    (plan,) = recorded
    assert plan.num_cores == 32
    assert governing_run_split(
        plan.source_view, plan.destination_view, [64, 512, 64]
    ) == (1024, 32)
    ref = torch.relu(torch.neg(host.float())).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


def test_two_relayout_edges_into_one_consumer(monkeypatch):
    """add(neg(a), abs(b)) with both edges forced: two destination rectangles
    at the consumer's tick, and two clones inserted before one consumer."""
    forced = _arm_forced(monkeypatch)
    recorded = forced.plans

    def fn(a, b):
        return torch.neg(a) + torch.abs(b)

    torch.manual_seed(0)
    ha = torch.randn(8, 256, 512, dtype=torch.float16)
    hb = torch.randn(8, 256, 512, dtype=torch.float16)
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False)(ha.to("spyre"), hb.to("spyre"))

    into_add = [p for p in recorded if len(p.consumer_names) == 1]
    consumers = {p.consumer_names[0] for p in into_add}
    forced.assert_emitted_in_lx(expected_plans=2)
    assert len(consumers) == 1, (
        f"expected both producer edges into the single add: {recorded}"
    )
    # Two destinations live at the same tick: they must not overlap in LX.
    a, b = recorded
    per_core = 8 * 256 * 512 * 2 // a.num_cores
    lo, hi = sorted((a.destination_address, b.destination_address))
    assert lo + per_core <= hi, "the two destinations overlap in LX"
    ref = (torch.neg(ha.float()) + torch.abs(hb.float())).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)


def test_one_source_two_consumers_share_one_relayout(monkeypatch):
    """relu(h) and abs(h) from one h with both edges forced. Both consumers
    want the same destination view of h, so they form one relayout GROUP: one
    shuffle, one LX destination spanning both consumers' ticks, one plan read
    by both, and the objective charges it once (see the decision tests for
    the per-edge vs per-group economics). Each output is correct."""
    forced = _arm_forced(monkeypatch)
    recorded = forced.plans

    def fn(t):
        h = torch.neg(t)
        return torch.relu(h), torch.abs(h)

    torch.manual_seed(0)
    ht = torch.randn(8, 256, 512, dtype=torch.float16)
    with config.patch(_COOPT):
        o1, o2 = torch.compile(fn, dynamic=False)(ht.to("spyre"))

    forced.assert_emitted_in_lx(expected_plans=1)
    (plan,) = recorded
    assert len(plan.consumer_names) == 2, (
        f"one shared plan must serve both consumers: {plan.consumer_names}"
    )
    torch.testing.assert_close(
        o1.cpu(),
        torch.relu(torch.neg(ht.float())).to(torch.float16),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        o2.cpu(),
        torch.abs(torch.neg(ht.float())).to(torch.float16),
        rtol=1e-3,
        atol=1e-3,
    )


def test_chained_relayouts(monkeypatch):
    """neg -> relu -> abs with every edge forced: a consumer that read through a
    relayout is itself the source of the next one. Three plans, one per edge,
    and the chain computes the right numbers."""
    forced = _arm_forced(monkeypatch)
    recorded = forced.plans

    def fn(t):
        return torch.abs(torch.relu(torch.neg(t)) - 0.5)

    torch.manual_seed(0)
    ht = torch.randn(8, 256, 512, dtype=torch.float16)
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False)(ht.to("spyre"))

    edges = [(p.source_name, p.consumer_names) for p in recorded]
    forced.assert_emitted_in_lx(expected_plans=3)
    assert len(edges) == 3
    sources = [p.source_name for p in recorded]
    consumers = [p.consumer_names[0] for p in recorded]
    # Each intermediate is the consumer of one plan and the source of the next.
    assert set(sources[1:]) <= set(consumers)
    ref = torch.abs(torch.relu(torch.neg(ht.float())) - 0.5).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_relayout_on_other_dtypes(monkeypatch, dtype):
    """The fitted law was measured on fp16; fp32 (32 elems/stick) and bf16 go
    through the same enumeration and materialization and must be correct.
    Prices for these dtypes are extrapolations of the law, not measurements."""
    forced = _arm_forced(monkeypatch)

    def fn(t):
        return torch.relu(torch.neg(t))

    torch.manual_seed(0)
    ht = torch.randn(8, 256, 512, dtype=dtype)
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False)(ht.to("spyre"))

    forced.assert_emitted_in_lx(expected_plans=1)
    torch.testing.assert_close(
        out.cpu().float(), torch.relu(torch.neg(ht.float())), rtol=1e-2, atol=1e-2
    )


def test_cross_group_coarse_tiled_edges_are_never_offered(monkeypatch):
    """A coarse-tiled edge that crosses for_each_tile scopes must enumerate NO
    relayout candidates, even though a same-group coarse-tiled edge now can
    (see test_same_group_coarse_tiled_edge_is_offered_and_priced_per_trip
    below, torch-spyre#5057).

    The producer sits in one for_each_tile body and the consumer in a
    separate, sibling body with its own loop_group_id, so this is a true
    cross-group edge, not merely two ops that happen to both be tiled. A
    relayout taken at that seam would still be stale on the next outer
    iteration the same way an in-loop one would be; the solver path must
    decline at enumeration (loop_info gate), not merely decline to fire.

    The two for_each_tile calls tile different axes of the intermediate (dim
    0 then dim 1) specifically to force two genuinely distinct loop scopes.
    Verified empirically, not just by inspection: equal tile_size/dims
    siblings over the same shape fuse into one shared while_loop body (both
    intermediates named ``while_loop_body_graph_0_0_*``); giving the second
    call a different tile_size ALONE (same dim) still fuses the same way,
    byte-identical candidate list. Only a different tiled axis keeps the
    splicer from inlining the producer directly into the consumer's body --
    which is not the cross-group case this test means to cover."""
    from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

    tables = []
    orig_edges = GraphBuilder._relayout_edges

    def spy_edges(self, *args, **kwargs):
        out = orig_edges(self, *args, **kwargs)
        if out:
            tables.append(out)
        return out

    monkeypatch.setattr(GraphBuilder, "_relayout_edges", spy_edges)

    recorded = []
    real_materialize = alloc_mod.materialize_lx_relayouts

    def spy_mat(graph, plans):
        recorded.extend(plans)
        return real_materialize(graph, plans)

    monkeypatch.setattr(alloc_mod, "materialize_lx_relayouts", spy_mat)

    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()

    def fn(t, u):
        def abs_body(_, ops):
            (t_tile,) = ops
            return None, torch.abs(t_tile)

        def add_body(_, ops):
            v_tile, u_tile = ops
            return None, (v_tile + u_tile) * 2

        _, v = for_each_tile(abs_body, (t,), dims=(0,), tile_size=128, out_dim=0)
        return for_each_tile(add_body, (v, u), dims=(1, 1), tile_size=64, out_dim=1)[1]

    torch.manual_seed(0)
    ha = torch.randn(512, 256, dtype=torch.float16)
    hb = torch.randn(512, 256, dtype=torch.float16)
    a = ha.to("spyre")
    b = hb.to("spyre")
    with config.patch(_COOPT):
        out = torch.compile(fn, dynamic=False, fullgraph=True)(a, b)

    # Graph-global, same as the test this replaces: also covers each body's
    # own within-scope edges, which happen to offer nothing for this graph's
    # shapes. A future same-group change that makes one of those offer could
    # fail this assert looking like a cross-group regression when it isn't;
    # the printed table disambiguates by parent/consumer name.
    assert tables == [], (
        f"cross-group coarse-tiled edges were offered relayout: {tables}"
    )
    assert recorded == []
    ref = ((torch.abs(ha.float()) + hb.float()) * 2).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)


def test_same_group_coarse_tiled_edge_is_offered_and_priced_per_trip(monkeypatch):
    """A coarse-tiled edge within a SINGLE for_each_tile body may be offered a
    relayout (the same-group exception in ``solver_relayout_edge_context``,
    torch-spyre#5057) when ``lx_same_tile_group_relayout`` is enabled (off by
    default as of #5632, pending that PR's compile-time investigation), and
    the offered ``RelayoutEdge`` must price every pair at the loop's own
    trip count (``_coarse_tile_loop_trip_count``, see
    TestCoarseTileLoopTripCount in test_scratchpad_solver.py): a relayout
    taken once but paid on every tile iteration is not priced the same as one
    taken once, and the fitted law has no loop_trip factor of its own.

    The cross-group case is test_cross_group_coarse_tiled_edges_are_never_offered
    above."""
    from torch_spyre._inductor.loop_info import CoarseTileInfo
    from torch_spyre._inductor.scratchpad.lx_relayout import (
        solver_relayout_pair_cost,
    )
    from torch_spyre._inductor.scratchpad.graph_builder import RelayoutEdge
    from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

    tile_size = 128

    def fn(t, u):
        def body(_, ops):
            t_tile, u_tile = ops
            return None, (torch.abs(t_tile) + u_tile) * 2

        return for_each_tile(body, (t, u), dims=(0, 0), tile_size=tile_size, out_dim=0)[
            1
        ]

    torch.manual_seed(0)
    ha = torch.randn(512, 256, dtype=torch.float16)
    hb = torch.randn(512, 256, dtype=torch.float16)
    loop_trip_count = ha.shape[0] // tile_size

    edges = []
    orig_edges = GraphBuilder._relayout_edges

    def spy_edges(self, *args, **kwargs):
        out = orig_edges(self, *args, **kwargs)
        edges.extend(out.values())
        return out

    priced = []
    orig_price = RelayoutEdge.price

    def spy_price(self, pv, cv, ncores, dst_cores):
        out = orig_price(self, pv, cv, ncores, dst_cores)
        if out is not None:
            priced.append((self, (pv, cv, ncores, dst_cores), out[0]))
        return out

    monkeypatch.setattr(GraphBuilder, "_relayout_edges", spy_edges)
    monkeypatch.setattr(RelayoutEdge, "price", spy_price)
    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "lx_same_tile_group_relayout": True,
        }
    ):
        out = torch.compile(fn, dynamic=False, fullgraph=True)(
            ha.to("spyre"), hb.to("spyre")
        )

    tiled = [
        e
        for e in edges
        if isinstance(getattr(e.parent_op, "loop_info", None), CoarseTileInfo)
        and isinstance(getattr(e.consumer_op, "loop_info", None), CoarseTileInfo)
    ]
    assert tiled, f"same-group coarse-tiled edge was not offered relayout: {edges}"
    assert all(e.loop_trip_count == loop_trip_count for e in tiled), [
        (e.parent, e.loop_trip_count) for e in tiled
    ]
    tiled_priced = [p for p in priced if p[0] in tiled]
    assert tiled_priced, "no pair of the same-group edge was priced"
    for edge, (pv, cv, ncores, dst_cores), cost in tiled_priced:
        # The fitted law prices one tile's shuffle; the edge pays it per trip.
        # Same deterministic base, so the scaling is exact.
        base = solver_relayout_pair_cost(
            pv,
            cv,
            ncores,
            edge.device_dims,
            edge.out_elems,
            edge.dtype_bytes,
            destination_num_cores=dst_cores,
        )
        assert cost == base * loop_trip_count, (edge.parent, cost, base)
    ref = ((torch.abs(ha.float()) + hb.float()) * 2).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)
