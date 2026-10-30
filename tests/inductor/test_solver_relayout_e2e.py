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
never shuffle. To exercise the decision deterministically, the match table is
emptied on every edge (monkeypatch), making a relayout the only route to
residency; the real cost objective then chooses it, because the shuffle
(~1-9 us depending on the chosen geometry) beats spilling the 2.10 MB
intermediate through HBM.

Asserts the whole chain: the solver fires the edge, the commit path rebuilds
the plan (views under the chosen divisions, solved addresses), the existing
``materialize_lx_relayouts`` inserts the shuffle, and the compiled graph
computes the right numbers on device.
"""

import math

import pytest

pytest.importorskip("ortools")

import torch

import torch_spyre  # noqa: F401
from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad import allocator as alloc_mod
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver


_COOPT = {"co_optimizing_lx_planning": True, "layout_solver": "cpsat"}


def _no_matches(self, edges, *args, **kwargs):
    return {parent: [] for parent in edges}


class _Observed:
    """Observe the whole relayout chain of one compile: the plans handed to
    materialize_lx_relayouts, any post-planning demotion of a relayout group
    by the scheduler, and every identity op codegen classified as an LX
    relayout (both args LX-resident with distinct work divisions) together
    with the LX addresses it emitted.

    With ``force=True`` the match table is emptied on every edge, so a
    relayout is the only route to residency (see the module docstring).
    With ``force=False`` the graph is compiled as a user would see it and
    only read-only spies are installed.
    """

    def __init__(self, monkeypatch, *, force: bool):
        import torch_spyre._inductor.codegen.superdsc as sdsc_mod
        import torch_spyre._inductor.scheduler as sched_mod
        import torch_spyre._inductor.spyre_kernel as sk_mod
        from torch_spyre._inductor import op_spec as opspec_mod

        self.plans: list = []
        # The committed path's own plans for the SAME committed graph, taken at
        # the moment the solver's plans are materialized (see
        # assert_fired_edges_are_committed_eligible).
        self.committed: list = []
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
        self.source_layouts: dict = {}
        self.copy_layouts: dict = {}
        self.consumer_reads: dict[str, set[str]] = {}
        real_materialize = alloc_mod.materialize_lx_relayouts

        def spy_materialize(graph, plans):
            from torch_spyre._inductor.pass_utils import op_read_writes
            from torch_spyre._inductor.scratchpad.lx_relayout import (
                collect_lx_relayout_plans,
                materialized_lx_relayouts,
            )

            self.plans.extend(plans)
            # Divisions are committed by now and nothing is materialized yet:
            # exactly the state the committed collector expects.
            self.committed.extend(collect_lx_relayout_plans(graph))
            result = real_materialize(graph, plans)
            self.copies = {
                edge: copy_name
                for edge, (copy_name, _plan) in materialized_lx_relayouts(graph).items()
            }
            for plan in plans:
                layout = graph.get_buffer(plan.source_name).get_layout()
                self.source_layouts[plan.edge] = (
                    list(layout.device_layout.device_size),
                    list(layout.device_layout.stride_map),
                    list(layout.stride),
                )
                copy_layout = graph.get_buffer(self.copies[plan.edge]).get_layout()
                self.copy_layouts[plan.edge] = (
                    list(copy_layout.size),
                    list(copy_layout.stride),
                    list(copy_layout.device_layout.device_size),
                    list(copy_layout.device_layout.stride_map),
                )
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
            monkeypatch.setattr(
                alloc_mod.CoOptimizingAllocator, "_cd_parent_matches", _no_matches
            )
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
        self.assert_fired_edges_are_committed_eligible()

    def assert_fired_edges_are_committed_eligible(self) -> None:
        """Parity with the committed path.

        The solver enumerates candidates before divisions exist, so it mirrors
        the committed path's eligibility gates instead of calling them. Those
        gates are being rewritten under it (#4284, #3440, #4153). This pins the
        one direction that must always hold: every edge the solver fired is one
        the committed collector, run on the same committed graph, also
        certifies, with the same physical source and destination ownership,
        core count and per-core LX spans (the sizes both allocators reserve for
        the shuffle's endpoints, #3440). (The reverse need not hold: the solver may decline an
        eligible edge on economics or capacity.) A solver segment's consumers
        are a subset of the committed plan's, which groups every consumer of a
        destination view together.
        """
        for plan in self.plans:
            twins = [
                c
                for c in self.committed
                if c.source_name == plan.source_name
                and set(plan.consumer_names) <= set(c.consumer_names)
                and c.num_cores == plan.num_cores
                and c.source_view.same_partition(plan.source_view)
                and c.destination_view.same_partition(plan.destination_view)
                and c.source_footprint_bytes == plan.source_footprint_bytes
                and c.destination_footprint_bytes == plan.destination_footprint_bytes
            ]
            assert twins, (
                f"solver fired {plan.source_name} -> {plan.consumer_names} "
                f"(src {dict(plan.source_view.work_slice_dims)}, dst "
                f"{dict(plan.destination_view.work_slice_dims)}, {plan.num_cores} "
                f"cores, spans {plan.source_footprint_bytes}/"
                f"{plan.destination_footprint_bytes} B) but the committed collector "
                "certifies no such edge on the committed graph; committed plans: "
                f"{[(c.source_name, c.consumer_names, dict(c.source_view.work_slice_dims), dict(c.destination_view.work_slice_dims), c.source_footprint_bytes, c.destination_footprint_bytes) for c in self.committed]}"
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
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "layout_solver": "cpsat",
        }
    ):
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


def test_an_unsupporting_solver_materializes_nothing(monkeypatch):
    """The solver's ``decides_lx_relayouts()`` says whether it decides
    relayouts; when it does not, no candidate is enumerated and the
    co-optimizing solve is exactly what it was before this feature."""
    observed = _arm_forced(monkeypatch)
    monkeypatch.setattr(
        CpSatLayoutSolver, "decides_lx_relayouts", classmethod(lambda cls: False)
    )

    def fn(t):
        return torch.relu(torch.neg(t)) + 1.0

    torch.manual_seed(0)
    host = torch.randn(8, 256, 512, dtype=torch.float16)
    x = host.to("spyre")
    with config.patch(_COOPT):
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
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "layout_solver": "cpsat",
        }
    ):
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
        assert observed.committed == []
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
    - The parity sub-check runs the committed collector on the same committed
    graph and verifies the same edge with matching core count, work-slice
    partition, and per-endpoint LX footprint.
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
    from torch_spyre._inductor.cost_features import governing_run_split

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
    """Relayouts must stay within one coarse-tile scope.

    The two for_each_tile bodies tile different axes to retain distinct
    scopes. Check every offered and materialized edge by its endpoint
    metadata; valid edges inside either body may still offer relayouts.
    The candidate unit test also probes real sibling and nested group IDs
    to prove that the scope gate rejects them before layout inspection.
    """
    from torch_spyre._inductor.loop_info import CoarseTileInfo
    from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

    groups_seen = set()

    def assert_same_scope(producer, consumer):
        prod_info = getattr(producer, "loop_info", None)
        cons_info = getattr(consumer, "loop_info", None)
        if prod_info is None and cons_info is None:
            return
        assert isinstance(prod_info, CoarseTileInfo)
        assert isinstance(cons_info, CoarseTileInfo)
        assert prod_info.loop_group_id == cons_info.loop_group_id, (
            f"cross-group relayout: {producer.get_name()} "
            f"{prod_info.loop_group_id} -> {consumer.get_name()} "
            f"{cons_info.loop_group_id}"
        )

    orig_tables = alloc_mod.CoOptimizingAllocator._cd_parent_relayouts

    def spy_tables(
        self,
        graph,
        consumer_op,
        consumer_divs,
        parent_names,
        divisions,
        op_by_name,
        *args,
        **kwargs,
    ):
        out = orig_tables(
            self,
            graph,
            consumer_op,
            consumer_divs,
            parent_names,
            divisions,
            op_by_name,
            *args,
            **kwargs,
        )
        for op in [consumer_op, *(op_by_name.get(name) for name in parent_names)]:
            info = getattr(op, "loop_info", None)
            if isinstance(info, CoarseTileInfo):
                groups_seen.add(info.loop_group_id)
        for parent, candidates in out.items():
            if candidates:
                assert_same_scope(op_by_name[parent], consumer_op)
        return out

    monkeypatch.setattr(
        alloc_mod.CoOptimizingAllocator, "_cd_parent_relayouts", spy_tables
    )

    real_materialize = alloc_mod.materialize_lx_relayouts

    def spy_mat(graph, plans):
        for plan in plans:
            for name in plan.consumer_names:
                assert_same_scope(
                    graph.get_buffer(plan.source_name), graph.get_buffer(name)
                )
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
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "layout_solver": "cpsat",
        }
    ):
        out = torch.compile(fn, dynamic=False, fullgraph=True)(a, b)

    assert len(groups_seen) >= 2, "fixture did not retain distinct coarse-tile scopes"
    ref = ((torch.abs(ha.float()) + hb.float()) * 2).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)


def test_same_group_coarse_tiled_edge_is_offered_and_priced_per_trip(monkeypatch):
    """A coarse-tiled edge within a SINGLE for_each_tile body may be offered a
    relayout candidate (the same-group exception in
    ``solver_relayout_edge_context``, torch-spyre#5057), when
    ``lx_same_tile_group_relayout`` is enabled, and that candidate's
    cost must carry the loop's own trip-count multiplier
    (``_coarse_tile_loop_trip_count``, see TestCoarseTileLoopTripCount in
    test_scratchpad_solver.py) -- a relayout taken once but paid on every
    tile iteration is not priced the same as one taken once, and the fitted
    law has no loop_trip factor of its own to account for that.

    Before #5057, this exact graph was asserted to offer NO candidates
    (test_coarse_tiled_edges_are_never_offered). #5367 unskipped that test on
    main after confirming the #3810 crash it cited was gone, and the
    invariant genuinely held on main as it stood then; #5057 is what
    deliberately narrows it to the cross-group case covered by
    test_cross_group_coarse_tiled_edges_are_never_offered above."""
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

    def run_and_collect_candidates():
        tables = []
        orig_tables = alloc_mod.CoOptimizingAllocator._cd_parent_relayouts

        def spy_tables(self, *args, **kwargs):
            out = orig_tables(self, *args, **kwargs)
            if out:
                tables.append(out)
            return out

        with monkeypatch.context() as m:
            m.setattr(
                alloc_mod.CoOptimizingAllocator, "_cd_parent_relayouts", spy_tables
            )
            torch._inductor.codecache.FxGraphCache.clear()
            torch._dynamo.reset()
            a = ha.to("spyre")
            b = hb.to("spyre")
            with config.patch(
                {
                    "co_optimizing_lx_planning": True,
                    "layout_solver": "cpsat",
                    "lx_same_tile_group_relayout": True,
                }
            ):
                out = torch.compile(fn, dynamic=False, fullgraph=True)(a, b)
        candidates = [c for table in tables for cs in table.values() for c in cs]
        return out, candidates

    def _key(c):
        return (c.parent, c.consumer, c.source_division, c.consumer_division, c.group)

    out, scaled_candidates = run_and_collect_candidates()
    assert scaled_candidates, "same-group coarse-tiled edge was not offered relayout"
    scaled_by_key = {_key(c): c.cost_ns for c in scaled_candidates}
    assert len(scaled_by_key) == len(scaled_candidates), (
        "candidate key collided -- some candidates were dropped from the comparison"
    )

    with monkeypatch.context() as m:
        m.setattr(alloc_mod, "_coarse_tile_loop_trip_count", lambda op: 1)
        _, unscaled_candidates = run_and_collect_candidates()
    assert unscaled_candidates, "forcing trip count to 1 should not remove candidates"
    unscaled_by_key = {_key(c): c.cost_ns for c in unscaled_candidates}
    assert len(unscaled_by_key) == len(unscaled_candidates), (
        "candidate key collided -- some candidates were dropped from the comparison"
    )

    assert scaled_by_key.keys() == unscaled_by_key.keys()
    for key, scaled_cost in scaled_by_key.items():
        unscaled_cost = unscaled_by_key[key]
        # pair_cost itself carries no trip-count factor (that is the whole point
        # of multiplying it in locally), so both runs share the same
        # deterministic base and the scaling must be exact, not approximate.
        assert scaled_cost == unscaled_cost * loop_trip_count, (
            f"candidate {key} cost {scaled_cost} is not the per-trip cost "
            f"{unscaled_cost} scaled by the loop trip count ({loop_trip_count})"
        )

    ref = ((torch.abs(ha.float()) + hb.float()) * 2).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)


def test_sa_cooptimizer_keeps_its_objective_and_decides_no_relayout(monkeypatch):
    """The simulated-annealing engine does not decide relayouts yet, so the
    allocator must hand it neither candidates nor copies, and its objective
    must carry no relayout term: with one it cannot bind, the annealer would
    silently drop the WHOLE cost objective and fall back to memory-only
    scoring (the regression this guards). Same hinted graph as the natural
    firing test above, under the annealer: it compiles, computes the right
    numbers, fires nothing, and scores with the real cost expression."""
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.scratchpad import sa_cooptimizer as sa_mod

    observed = _Observed(monkeypatch, force=False)
    score_fns: list = []
    real_build = sa_mod.SaCoOptimizingSolver._build_score_fn

    def spy_build(self, cost_expr):
        fn = real_build(self, cost_expr)
        score_fns.append((cost_expr, fn))
        return fn

    monkeypatch.setattr(sa_mod.SaCoOptimizingSolver, "_build_score_fn", spy_build)

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
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "layout_solver": "simulated_annealing",
        }
    ):
        out = torch.compile(fn, dynamic=False)(x)

    observed.assert_nothing_emitted()
    assert score_fns, "the annealer never built a score function"
    for cost_expr, score_fn in score_fns:
        assert cost_expr is not None, "the allocator produced no cost expression"
        assert not any(
            str(sym).startswith(("is_lx___spyre_lx_relayout__", "relayout"))
            for sym in cost_expr.free_symbols
        ), (
            f"relayout term handed to an engine that cannot decide relayouts: {cost_expr}"
        )
        assert score_fn is not None, (
            "the annealer fell back to the memory-only objective: a symbol in the "
            "cost expression could not be bound"
        )
    ref = torch.relu(torch.neg(host.float())).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "rows,width,dtype,batched,strided",
    [(512, 4096, torch.float16, True, False), (64, 512, torch.float32, False, True)],
)
def test_complete_partition_expansion_fires_and_preserves_values(
    monkeypatch, rows, width, dtype, batched, strided
):
    """A real 16-to-32-core transfer with four-source destination gathers."""
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.scratchpad.lx_relayout import _core_slices

    observed = _Observed(monkeypatch, force=False)

    def fn(t):
        with spyre_hint(work_div={"R": 16}):
            hidden = torch.neg(t)
        with spyre_hint(work_div={"R": 4, "H": 8}):
            return torch.relu(hidden)

    # Give all 128 source/feature fragments distinct value ranges.
    # Keep inputs negative so relu(-t) observes every destination.
    storage_width = width * (2 if strided else 1)
    row_ids = torch.arange(rows).reshape(-1, 1)
    column_ids = torch.arange(storage_width).reshape(1, -1) // (2 if strided else 1)
    fragment = (row_ids // (rows // 16)) * 8 + column_ids // (width // 8)
    offset = (row_ids * 37 + column_ids * 17) % 8
    host = -(fragment * 8 + offset + 1).to(dtype)
    if strided:
        host = host[:, ::2]
        assert not host.is_contiguous()
    names = ["R", "H"]
    if batched:
        host = host.unsqueeze(0)
        names = ["B", *names]
        _pnd.declare_tensor_dim("B", 1)
    for name, size in (("R", rows), ("H", width)):
        _pnd.declare_tensor_dim(name, size)
    x = _pnd.name_tensor_dims(host.to("spyre"), names)
    with config.patch({**_COOPT, "sencores": 32}):
        out = torch.compile(fn, dynamic=False)(x)

    observed.assert_emitted_in_lx(expected_plans=1)
    (plan,) = observed.plans
    assert plan.num_cores == 16 and plan.destination_view.num_cores == 32
    # Physical dimension order may change when a singleton batch is present.
    # Locate row and outer-stick axes from the committed tensor's strides,
    # independently of the per-core views being checked.
    dims, strides, logical_strides = observed.source_layouts[plan.edge]
    row_axes = [
        i
        for i, (size, stride) in enumerate(zip(dims, strides))
        if size == rows and stride == logical_strides[-2]
    ]
    column_axes = [
        i
        for i, (size, stride) in enumerate(zip(dims[:-1], strides[:-1]))
        if size == width // dims[-1] and stride == logical_strides[-1] * dims[-1]
    ]
    assert len(row_axes) == len(column_axes) == 1
    row_axis, column_axis = row_axes[0], column_axes[0]
    assert row_axis != column_axis
    assert _core_slices(plan.source_view, 16) == {c: {row_axis: c} for c in range(16)}
    assert _core_slices(plan.destination_view, 32) == {
        c: {column_axis: c // 4, row_axis: c % 4} for c in range(32)
    }
    # Striding doubles host backing storage, not this logical device tensor's LX span.
    assert plan.source_footprint_bytes == rows * width * host.element_size() // 16
    assert plan.destination_footprint_bytes == rows * width * host.element_size() // 32
    assert (
        plan.source_address + plan.source_footprint_bytes <= plan.destination_address
        or plan.destination_address + plan.destination_footprint_bytes
        <= plan.source_address
    ), "live source and destination LX spans overlap"
    assert torch.equal(out.cpu(), torch.relu(-host))


@pytest.mark.parametrize(
    "rows,inner,strided",
    [(512, 4096, False), (64, 4096, False), (128, 4096, False), (64, 65, True)],
)
def test_q_projection_rope_handoff_preserves_pairs_and_other_consumer(
    monkeypatch, rows, inner, strided
):
    _check_projection_rope_handoff(monkeypatch, rows, inner, strided, heads=8)


@pytest.mark.parametrize("rows,inner,strided", [(128, 65, True)])
def test_k_projection_rope_complete_contraction(monkeypatch, rows, inner, strided):
    _check_projection_rope_handoff(
        monkeypatch, rows, inner, strided, heads=2, contract=True
    )


def _check_projection_rope_handoff(
    monkeypatch, rows, inner, strided, *, heads, contract=False
):
    """Compile producers and consumers together; never replay implicit LX inputs."""
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.scratchpad.lx_relayout import _core_slices

    observed = _Observed(monkeypatch, force=False)
    features = heads * 128
    feature_groups = 4 if contract else min(heads, 4)
    cores = 8 * feature_groups
    destination_cores = 16 if contract else cores
    # Linear merges head/pair into one output loop; name its total split once.
    producer_hint = {"R": 8, "H": feature_groups}

    def fn(t, w, frequencies):
        with spyre_hint(work_div=producer_hint):
            q = torch.nn.functional.linear(t, w)
        with spyre_hint(work_div={"R": destination_cores}):
            shaped = q.reshape(1, rows, heads, 2, 64)
            rotated = (frequencies * shaped.unsqueeze(-3)).sum(4, keepdim=True)
            # This second read must be preserved when the private copy is
            # inserted. Returning Q itself would require an HBM graph output.
            side = q + 1
            return rotated.flatten(3), side

    storage_width = inner * (2 if strided else 1)
    host = (
        (torch.arange(rows * storage_width).reshape(1, rows, storage_width) * 13) % 127
        - 63
    ).to(torch.float16)
    if strided:
        host = host[:, :, ::2]
        assert not host.is_contiguous()
    # One nonzero weight per output makes the complete GEMM reference exact,
    # including the padded K=65 case. Power-of-two scales distinguish heads
    # when that small K causes input-column reuse.
    columns = torch.arange(features) % inner
    scales = (2 ** ((torch.arange(features) // inner) % 4)).to(torch.float16)
    weight = torch.zeros(features, inner, dtype=torch.float16)
    weight[torch.arange(features), columns] = scales
    q_ref = host[:, :, columns] * scales
    # Valid 0/90-degree rotations vary with both row and feature. A swapped
    # pair, wrong sign, misplaced row fragment or wrong head changes values.
    cos = ((torch.arange(rows)[:, None] + torch.arange(64)[None, :]) % 2).to(
        torch.float16
    )
    sin = 1 - cos
    frequencies = torch.stack(
        (torch.stack((cos, -sin), dim=1), torch.stack((sin, cos), dim=1)), dim=1
    ).reshape(1, rows, 1, 2, 2, 64)
    expected = (
        (
            frequencies.float()
            * q_ref.float().reshape(1, rows, heads, 2, 64).unsqueeze(-3)
        )
        .sum(4, keepdim=True)
        .flatten(3)
        .to(torch.float16)
    )
    # Preserve head/pair/feature names through the RoPE view. Each feature
    # shard contains complete rotation pairs, including the two-head K case.
    for name, size in (
        ("B", 1),
        ("R", rows),
        ("K", inner),
        ("H", heads),
        ("P", 2),
        ("D", 64),
    ):
        _pnd.declare_tensor_dim(name, size)
    x = _pnd.name_tensor_dims(host.to("spyre"), ["B", "R", "K"])
    w = _pnd.name_tensor_dims(weight.to("spyre"), ["H", "P", "D", "K"])
    with config.patch({**_COOPT, "sencores": 32}):
        out, side = torch.compile(fn, dynamic=False, fullgraph=True)(
            x, w, frequencies.to("spyre")
        )

    # The test is a capability/correctness gate, not a performance result.
    # Both consumers use one destination view and hence one allocated copy.
    observed.assert_emitted_in_lx(expected_plans=1)
    (plan,) = observed.plans
    assert len(plan.consumer_names) == 2
    dims, strides, logical_strides = observed.source_layouts[plan.edge]
    row_axes = [
        i
        for i, (size, stride) in enumerate(zip(dims, strides))
        if size == rows and stride == logical_strides[-2]
    ]
    column_axes = [
        i
        for i, (size, stride) in enumerate(zip(dims[:-1], strides[:-1]))
        if size == features // 64 and stride == logical_strides[-1] * dims[-1]
    ]
    assert len(row_axes) == len(column_axes) == 1
    row_axis, column_axis = row_axes[0], column_axes[0]
    assert _core_slices(plan.source_view, cores) == {
        c: {row_axis: c % 8, column_axis: c // 8} for c in range(cores)
    }
    assert _core_slices(plan.destination_view, destination_cores) == {
        c: {row_axis: c} for c in range(destination_cores)
    }
    assert plan.source_footprint_bytes == rows * features * 2 // cores
    assert plan.destination_footprint_bytes == rows * features * 2 // destination_cores
    assert (
        plan.source_address + plan.source_footprint_bytes <= plan.destination_address
        or plan.destination_address + plan.destination_footprint_bytes
        <= plan.source_address
    ), "simultaneously live projection and its relayout copy overlap"
    assert torch.equal(out.cpu(), expected)
    assert torch.equal(side.cpu(), q_ref + 1)


@pytest.mark.parametrize("rows", [512, 128, 64])
def test_attention_output_repack_handoff(monkeypatch, rows):
    """Produce softmax/PV/divide before repack/O; no implicit LX live-in.

    One unmasked key per query makes exp and the denominator exactly 0/1.
    Sparse O weights select every input feature, so wrong head/row/width
    routing cannot hide in a wide GEMM tolerance. A second divide consumer
    verifies that rewriting the repack read preserves the original lifetime.
    The test ends at O's HBM graph output; distributed collectives remain a
    separate production-harness requirement.
    """
    from torch._inductor.dependencies import MemoryDep
    from torch._inductor.ir import ComputedBuffer, Pointwise

    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.pass_utils import op_read_writes

    observed = _Observed(monkeypatch, force=False)
    restickify_inputs = {}
    real_materialize = alloc_mod.materialize_lx_relayouts

    def capture_restickify_inputs(graph, plans):
        for plan in plans:
            source = graph.get_buffer(plan.source_name)
            if not (
                isinstance(source, ComputedBuffer)
                and isinstance(source.data, Pointwise)
                and {node.target for node in source.origins}
                == {torch.ops.spyre.restickify.default}
            ):
                continue
            # This layout copy can merge head/width factors without changing
            # the within-stick dimension. Require its actual lowering origin
            # and one identity read, not a general path through other ops.
            rw = op_read_writes(source)
            assert len(rw.reads) == len(rw.writes) == 1
            (read,), (write,) = list(rw.reads), list(rw.writes)
            assert isinstance(read, MemoryDep) and isinstance(write, MemoryDep)
            assert read.index == write.index and read.ranges == write.ranges
            restickify_inputs[plan.source_name] = (
                read.name,
                graph.get_buffer(read.name).get_layout(),
                source.get_layout(),
            )
        return real_materialize(graph, plans)

    monkeypatch.setattr(
        alloc_mod, "materialize_lx_relayouts", capture_restickify_inputs
    )

    def fn(scores, values, weight):
        with spyre_hint(work_div={"R": 32}):
            shifted = scores - scores.amax(-1, keepdim=True)
            probability = torch.exp(shifted)
            denominator = probability.sum(-1, keepdim=True)
            numerator = torch.ops.spyre.batched_matmul(probability, values)
            divided = numerator / denominator
            side = torch.neg(divided)
        # Reshape merges KV/G into one loop; a hint names that loop once.
        with spyre_hint(work_div={"R": 8, "G": 4}):
            packed = divided.reshape(1, 8, rows, 128).transpose(1, 2).contiguous()
        with spyre_hint(work_div={"R": 4, "O": 8}):
            result = torch.nn.functional.linear(packed.reshape(1, rows, 1024), weight)
        return result, side

    selected = (torch.arange(rows) * 13 + 17) % 512
    scores = torch.full((1, 2, 4, rows, 512), -float("inf"), dtype=torch.float16)
    scores[:, :, :, torch.arange(rows), selected] = 0
    k = torch.arange(512).reshape(1, 1, 1, 512, 1)
    h = torch.arange(2).reshape(1, 2, 1, 1, 1)
    d = torch.arange(128).reshape(1, 1, 1, 1, 128)
    values = ((k * 37 + h * 257 + d * 3) % 1021 / 1024).to(torch.float16)
    selectors = (torch.arange(4096) * 37) % 1024
    weight = torch.zeros(4096, 1024, dtype=torch.float16)
    weight[torch.arange(4096), selectors] = 1
    divided = values[:, :, :, selected, :].expand(1, 2, 4, rows, 128)
    packed = divided.reshape(1, 8, rows, 128).transpose(1, 2).reshape(1, rows, 1024)
    expected = packed[:, :, selectors]
    for name, size in (
        ("B", 1),
        ("KV", 2),
        ("G", 4),
        ("R", rows),
        ("S", 512),
        ("D", 128),
        ("O", 4096),
    ):
        _pnd.declare_tensor_dim(name, size)
    score_arg = _pnd.name_tensor_dims(scores.to("spyre"), ["KV", "G", "R", "S"])
    value_arg = _pnd.name_tensor_dims(values.to("spyre"), ["KV", "S", "D"])
    weight_arg = _pnd.name_tensor_dims(weight.to("spyre"), ["O", "KV", "G", "D"])
    with config.patch({**_COOPT, "sencores": 32}):
        result, side = torch.compile(fn, dynamic=False, fullgraph=True)(
            score_arg, value_arg, weight_arg
        )
    observed.assert_emitted_in_lx(expected_plans=2)
    first, second = observed.plans
    if second.source_name not in first.consumer_names:
        assert second.source_name in restickify_inputs
        input_name, input_layout, output_layout = restickify_inputs[second.source_name]
        assert input_name in first.consumer_names
        assert (
            set(input_layout.allocation) == set(output_layout.allocation) == {"lx"}
        ), "the repack layout copy must keep both endpoints in LX"
    assert first.source_view.split_product == first.destination_view.split_product == 32
    assert second.source_view.split_product == 32
    assert second.destination_view.split_product == 4
    assert (
        first.source_footprint_bytes == first.destination_footprint_bytes == rows * 64
    )
    assert second.source_footprint_bytes == rows * 64
    assert second.destination_footprint_bytes == rows * 512
    for plan in observed.plans:
        assert (
            plan.source_address + plan.source_footprint_bytes
            <= plan.destination_address
            or plan.destination_address + plan.destination_footprint_bytes
            <= plan.source_address
        ), "live source and copy overlap"
    assert torch.equal(result.cpu(), expected)
    assert torch.equal(side.cpu(), -divided)


def _assert_cache_update_persistence(
    specs, cache_writes, source_address, elements, *, later_write=False
):
    """Require direct persistence or a complete ordered functionalized copyback."""
    from torch_spyre._inductor.constants import RESTICKIFY_OP

    snapshots = [
        (i, spec.args[0])
        for i, spec in enumerate(specs)
        if spec.op == RESTICKIFY_OP
        and len(spec.args) == 2
        and spec.args[1].allocation == {"lx": source_address}
    ]
    assert len(snapshots) == 1, "expected one persistent-cache snapshot"
    snapshot_index, cache = snapshots[0]
    assert math.prod(cache.device_size) == elements
    assert len(cache_writes) == 1 + int(later_write)
    assert set(cache_writes).issubset(map(id, specs))
    if later_write:
        writes = [
            (i, cache_writes[id(spec)])
            for i, spec in enumerate(specs)
            if id(spec) in cache_writes
        ]
        assert len(writes) == 2, (
            "expected indexed updates before and after the snapshot"
        )
        assert writes[0][0] < snapshot_index < writes[1][0]
        assert all(set(arg.allocation) in ({"hbm"}, {"hbm_pool"}) for _, arg in writes)
        # Functionalization can snapshot the first pooled version and copy the
        # second version back later. Exact outputs and the returned alias below
        # check both versions, including untouched history in the final cache.
        return
    assert cache.arg_index >= 0 and cache.allocation == {"hbm": cache.arg_index}
    for i, spec in enumerate(specs):
        if id(spec) not in cache_writes:
            continue
        updated = cache_writes[id(spec)]
        assert i < snapshot_index, "cache snapshot precedes its indexed update"
        if updated.allocation == cache.allocation:
            assert updated.arg_index == cache.arg_index
            continue
        assert set(updated.allocation) == {"hbm_pool"}, (
            "indexed cache writes must remain in HBM"
        )

        # Returning an existing view can functionalize the mutation. Require
        # history initialization and a full copyback to the exact external
        # cache read by the snapshot, not merely some later external write.
        def copies(source, destination):
            return [
                (j, candidate)
                for j, candidate in enumerate(specs)
                if candidate.op == "identity"
                and len(candidate.args) == 2
                and candidate.args[0].is_input
                and not candidate.args[1].is_input
                and candidate.args[0].allocation == source
                and candidate.args[1].allocation == destination
            ]

        initializers = copies(cache.allocation, updated.allocation)
        copybacks = copies(updated.allocation, cache.allocation)
        assert len(initializers) == len(copybacks) == 1, (
            "pooled cache update requires history initialization and copyback"
        )
        init_index, initializer = initializers[0]
        copy_index, copyback = copybacks[0]
        assert init_index < i < copy_index < snapshot_index, (
            "cache initialization, indexed update, copyback and snapshot are unordered"
        )
        for copy in (initializer, copyback):
            assert (
                math.prod(int(size) for size, _ in copy.iteration_space.values())
                == elements
            )
            assert all(math.prod(arg.device_size) == elements for arg in copy.args)
        for original, restored in zip(initializer.args, reversed(copyback.args)):
            assert original.arg_index == restored.arg_index
            assert original.device_size == restored.device_size
            assert original.device_coordinates == restored.device_coordinates
            assert original.device_dtype == restored.device_dtype == cache.device_dtype
        assert initializer.args[1].device_size == updated.device_size
        assert initializer.args[0].arg_index == cache.arg_index
        pool_writers = [
            j
            for j, candidate in enumerate(specs)
            if init_index <= j <= copy_index
            and any(
                not arg.is_input and arg.allocation == updated.allocation
                for arg in candidate.args
            )
        ]
        assert pool_writers == [init_index, i], "cache history has an unrelated writer"


@pytest.mark.parametrize(
    "rows,later_write",
    [(512, False), (128, False), (64, False), (512, True)],
    ids=["512", "128", "64", "512-later-write"],
)
def test_attention_cache_transpose_handoff(monkeypatch, rows, later_write):
    """Update persistent K, restickify its prefix, gather, then run two QKs.

    The original cache holds history and has twice the attended capacity.
    Nonmonotonic unique updates hit both halves. Exact one-hot queries check
    every head/width/context value, and a second QK keeps the snapshot live.
    Returning a pre-existing cache view checks mutation and alias semantics.
    No test-side buffer is assumed to arrive in LX.
    """
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    import torch_spyre._inductor.codegen.superdsc as sdsc_mod
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.constants import RESTICKIFY_OP
    from torch_spyre._inductor.op_spec import IndirectAccess

    observed = _Observed(monkeypatch, force=False)
    restick_allocations, cache_writes, parsed_specs = [], {}, {}
    real_parse = sdsc_mod.parse_op_spec

    def observe_parse(spec):
        parsed_specs.setdefault(id(spec), spec)
        if spec.op == RESTICKIFY_OP:
            restick_allocations.append([dict(arg.allocation) for arg in spec.args])
        for arg in spec.args:
            if not arg.is_input and any(
                coordinate.has(IndirectAccess) for coordinate in arg.device_coordinates
            ):
                cache_writes[id(spec)] = arg
        return real_parse(spec)

    monkeypatch.setattr(sdsc_mod, "parse_op_spec", observe_parse)

    def fn(cache, indices, keys, queries, other_queries, mask):
        alias = cache.view_as(cache)
        # The int32 indices occupy 32-element sticks: split only whole sticks.
        with spyre_hint(work_div={"R": rows // 32}):
            fresh = torch.neg(keys)
            cache.index_copy_(2, indices, fresh)
        with spyre_hint(work_div={"S": 2}, named_dims=["KV", "D", "S"]):
            transposed = cache[0, :, :512, :].transpose(-1, -2).contiguous()
        if later_write:
            # The immutable snapshot must keep the first update, not these values.
            with spyre_hint(work_div={"R": rows // 32}):
                cache.index_copy_(2, indices, keys)
        transposed = transposed.unsqueeze(0).unsqueeze(2)
        with spyre_hint(work_div={"R": 32}):
            query = torch.neg(queries)
            scores = torch.ops.spyre.batched_matmul(query, transposed)
            other = torch.ops.spyre.batched_matmul(other_queries, transposed)
            masked = scores.masked_fill(mask, -float("inf"))
        return masked, other, alias

    c = torch.arange(1024).reshape(1, 1, 1024, 1)
    h = torch.arange(2).reshape(1, 2, 1, 1)
    d = torch.arange(128).reshape(1, 1, 1, 128)
    cache = ((37 * c + 251 * h + 3 * d) % 1021).to(torch.float16)
    keys = ((19 * torch.arange(rows).reshape(1, 1, rows, 1) + 131 * h + d) % 509).to(
        torch.float16
    )
    indices = (torch.arange(rows) * 13 + 17) % 1024
    expected_cache = cache.clone()
    expected_cache.index_copy_(2, indices, -keys)
    selectors = (
        torch.arange(rows).reshape(1, 1, 1, rows)
        + 17 * torch.arange(4).reshape(1, 1, 4, 1)
        + 31 * torch.arange(2).reshape(1, 2, 1, 1)
    ) % 128
    query = torch.nn.functional.one_hot(selectors, 128).to(torch.float16)
    other_selectors = (selectors * 7 + 9) % 128
    other_query = torch.nn.functional.one_hot(other_selectors, 128).to(torch.float16)
    # A causal mask with a position offset exercises both kept and discarded
    # history. The raw second QK verifies the masked-out values as well.
    mask = torch.arange(512)[None, :] > (torch.arange(rows)[:, None] + 11)
    history = expected_cache[:, :, :512, :].transpose(-1, -2).unsqueeze(2)
    expected = torch.gather(
        history.expand(1, 2, 4, 128, 512),
        3,
        selectors[..., None].expand(1, 2, 4, rows, 512),
    ).masked_fill(mask, -float("inf"))
    expected_other = torch.gather(
        history.expand(1, 2, 4, 128, 512),
        3,
        other_selectors[..., None].expand(1, 2, 4, rows, 512),
    )
    if later_write:
        expected_cache.index_copy_(2, indices, keys)
    for name, size in (
        ("KV", 2),
        ("C", 1024),
        ("S", 512),
        ("R", rows),
        ("D", 128),
        ("G", 4),
    ):
        _pnd.declare_tensor_dim(name, size)
    cache_arg = _pnd.name_tensor_dims(cache.to("spyre"), ["KV", "C", "D"])
    key_arg = _pnd.name_tensor_dims(keys.to("spyre"), ["KV", "R", "D"])
    query_arg = _pnd.name_tensor_dims((-query).to("spyre"), ["KV", "G", "R", "D"])
    other_arg = _pnd.name_tensor_dims(other_query.to("spyre"), ["KV", "G", "R", "D"])
    with config.patch({**_COOPT, "sencores": 32}):
        result, other, alias = torch.compile(fn, dynamic=False, fullgraph=True)(
            cache_arg,
            indices.to("spyre"),
            key_arg,
            query_arg,
            other_arg,
            mask.to("spyre"),
        )
    observed.assert_emitted_in_lx(expected_plans=1)
    (plan,) = observed.plans
    assert plan.source_view.num_cores == plan.source_view.split_product == 2
    assert plan.destination_view.num_cores == 32
    assert plan.destination_view.split_product == 1
    assert len(plan.consumer_names) == 2
    assert plan.source_footprint_bytes == 131072
    assert plan.destination_footprint_bytes == 262144
    assert any(
        len(args) == 2
        and "lx" not in args[0]
        and args[1] == {"lx": plan.source_address}
        for args in restick_allocations
    ), "the transpose must read HBM cache and write the planned LX source"
    _assert_cache_update_persistence(
        list(parsed_specs.values()),
        cache_writes,
        plan.source_address,
        cache.numel(),
        later_write=later_write,
    )
    assert (
        plan.source_address + plan.source_footprint_bytes <= plan.destination_address
        or plan.destination_address + plan.destination_footprint_bytes
        <= plan.source_address
    ), "live K snapshot and its gathered replica overlap"
    assert torch.equal(result.cpu(), expected)
    assert torch.equal(other.cpu(), expected_other)
    assert torch.equal(cache_arg.cpu(), expected_cache)
    assert torch.equal(alias.cpu(), expected_cache)
    assert alias.data_ptr() == cache_arg.data_ptr()


@pytest.mark.parametrize("rows,inner,strided", [(512, 65, True), (64, 65, True)])
def test_v_projection_cache_handoff_preserves_persistent_alias(
    monkeypatch, rows, inner, strided
):
    """Relayout the fresh V input while preserving persistent cache writes.

    Include the complete producer, a later reader of its original ownership,
    and unique reversed non-contiguous cache indices. Untouched rows and the
    caller's cache alias must retain exactly their original semantics.
    """
    import torch_spyre._inductor.codegen.superdsc as sdsc_mod
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    observed = _Observed(monkeypatch, force=False)
    specs = {}
    real_parse = sdsc_mod.parse_op_spec

    def capture(spec):
        specs[id(spec)] = spec
        return real_parse(spec)

    monkeypatch.setattr(sdsc_mod, "parse_op_spec", capture)

    def fn(t, w, cache, index):
        with spyre_hint(work_div={"R": 8, "H": 4}):
            v = torch.nn.functional.linear(t, w)
        with spyre_hint(work_div={"R": 16} if rows == 512 else {}):
            values = v.reshape(1, rows, 2, 128).transpose(1, 2)
            cache.index_copy_(2, index, values)
        with spyre_hint(work_div={"R": 8, "H": 4}):
            side = v + 1
        return cache, side

    storage_width = inner * (2 if strided else 1)
    host = (
        (torch.arange(rows * storage_width).reshape(1, rows, storage_width) * 13) % 127
        - 63
    ).to(torch.float16)
    if strided:
        host = host[:, :, ::2]
        assert not host.is_contiguous()
    columns = torch.arange(256) % inner
    scales = (2 ** ((torch.arange(256) // inner) % 4)).to(torch.float16)
    weight = torch.zeros(256, inner, dtype=torch.float16)
    weight[torch.arange(256), columns] = scales
    v_ref = host[:, :, columns] * scales
    index = torch.arange(2 * rows - 1, 0, -2, dtype=torch.int64)
    cache_host = torch.full((1, 2, 1024, 128), -777, dtype=torch.float16)
    expected = cache_host.clone()
    expected.index_copy_(2, index, v_ref.reshape(1, rows, 2, 128).transpose(1, 2))
    # V's cache view splits the flat output into head and 128-wide feature
    # loops. Unlike RoPE, it never exposes a separate rotation-pair loop.
    # The producer's H:4 hint still applies to the merged H*D output loop.
    for name, size in (
        ("B", 1),
        ("R", rows),
        ("K", inner),
        ("H", 2),
        ("D", 128),
    ):
        _pnd.declare_tensor_dim(name, size)
    x = _pnd.name_tensor_dims(host.to("spyre"), ["B", "R", "K"])
    w = _pnd.name_tensor_dims(weight.to("spyre"), ["H", "D", "K"])
    # Match the production cache allocator: the indexed position is the first
    # physical axis. A generic layout inserts an unrelated cache restickify
    # buffer and no longer exercises the direct external mutation contract.
    cache_layout = SpyreTensorLayout(
        device_size=[1024, 2, 2, 1, 64],
        stride_map=[128, 1024 * 128, 64, 2 * 1024 * 128, 1],
        device_dtype=get_device_dtype(torch.float16),
    )
    cache = cache_host.to("spyre", device_layout=cache_layout)
    alias = cache
    with config.patch({**_COOPT, "sencores": 32}):
        out, side = torch.compile(fn, dynamic=False, fullgraph=True)(
            x, w, cache, index.to("spyre")
        )

    assert torch.equal(alias.cpu(), expected), "persistent cache alias lost its update"
    assert torch.equal(out.cpu(), expected)
    assert torch.equal(side.cpu(), v_ref + 1), "cache write changed a live projection"
    projections = [s for s in specs.values() if s.op == "batchmatmul"]
    assert len(projections) == 1
    projection = projections[0]
    projected = projection.args[-1]
    # Match the actual indirect cache store, not another identity/host copy.
    updates = [
        s
        for s in specs.values()
        if s.op == "identity"
        and any("IndirectAccess" in str(c) for c in s.args[-1].device_coordinates)
    ]
    assert len(updates) == 1
    update = updates[0]
    if rows == 64:
        observed.assert_emitted_in_lx(expected_plans=0)
        assert set(projected.allocation) == {"hbm_pool"}
        assert set(update.args[-1].allocation) == {"hbm"}
        assert any(
            arg.allocation == projected.allocation for arg in update.args[:-1]
        ), "the short scatter must read the original HBM projection"
        assert (
            math.prod(int(split) for _, split in update.iteration_space.values())
            <= rows // 32
        ), "index sticks do not permit a larger scatter division"
        assert all("lx" not in arg.allocation for arg in update.args)
        return
    observed.assert_emitted_in_lx(expected_plans=1)
    (plan,) = observed.plans
    assert plan.source_view.num_cores == 32
    assert plan.destination_view.num_cores == 16
    assert len(plan.consumer_names) == 1, (
        "later original reader was unnecessarily rewritten"
    )
    assert set(projected.allocation) == {"lx"}, "V projection still stages in HBM"
    assert set(update.args[-1].allocation) == {"hbm"}, "cache persistence was removed"
    values = [a for a in update.args[:-1] if set(a.allocation) == {"lx"}]
    assert len(values) == 1, "cache update did not read the LX projection copy"
    assert values[0].allocation["lx"] == plan.destination_address
    assert projected.allocation["lx"] == plan.source_address
    assert sorted(
        int(split) for _, split in projection.iteration_space.values() if int(split) > 1
    ) == [4, 8]
    assert [
        int(split) for _, split in update.iteration_space.values() if int(split) > 1
    ] == [16]


def test_prefix_linear_reads_a_separate_full_backing_lx_gather(monkeypatch):
    """A 3200-wide linear reads a gathered 3328-wide physical allocation."""
    _run_prefix_linear_case(monkeypatch, mode="served")


def test_prefix_linear_without_served_copy_spills_the_source(monkeypatch):
    """Removing this edge's candidates must remove its route to LX residency."""
    _run_prefix_linear_case(monkeypatch, mode="no_copy")


def test_prefix_linear_scheduler_demotion_retains_valid_hbm_clone(monkeypatch):
    """Amended D6-2: retry through the ordinary HBM clone, not topology rollback."""
    _run_prefix_linear_case(monkeypatch, mode="demoted")


def _run_prefix_linear_case(monkeypatch, *, mode):
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    import torch_spyre._inductor.scheduler as sched_mod
    import torch_spyre._inductor.passes as passes_mod
    from torch._inductor.virtualized import V
    from torch_spyre._inductor.pass_utils import op_read_writes
    from torch_spyre._inductor.scratchpad.utils import storage_has_single_writer
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick
    from torch_spyre._inductor import spyre_hint
    from torch_spyre._inductor.scratchpad.lx_relayout import (
        prefix_lx_contraction_axis,
        materialized_lx_relayouts,
    )

    observed = _Observed(monkeypatch, force=False)
    prefix_sources, removed_candidates, source_residency = set(), [], []
    real_tables = alloc_mod.CoOptimizingAllocator._cd_parent_relayouts

    def tables(
        allocator,
        graph,
        consumer,
        consumer_divs,
        parents,
        divisions,
        op_by_name,
        *args,
        **kwargs,
    ):
        result = real_tables(
            allocator,
            graph,
            consumer,
            consumer_divs,
            parents,
            divisions,
            op_by_name,
            *args,
            **kwargs,
        )
        for parent in parents:
            if parent not in op_by_name or consumer is None:
                continue
            if prefix_lx_contraction_axis(op_by_name[parent], consumer) is None:
                continue
            prefix_sources.add(parent)
            if mode == "no_copy":
                # cap=0 means unlimited. Remove only the actual prefix edge's
                # candidates, leaving all unrelated allocator choices intact.
                removed_candidates.extend(result.pop(parent, []))
        return result

    real_post_solve = alloc_mod.CoOptimizingAllocator._post_solve

    def post_solve(allocator, graph, allocation, plans):
        source_residency.extend(
            (buffer.name, buffer.address)
            for buffer in allocation
            if buffer.name in prefix_sources
        )
        return real_post_solve(allocator, graph, allocation, plans)

    monkeypatch.setattr(alloc_mod.CoOptimizingAllocator, "_cd_parent_relayouts", tables)
    monkeypatch.setattr(alloc_mod.CoOptimizingAllocator, "_post_solve", post_solve)

    rejected, restored, final_kernels = [], [], []
    if mode == "demoted":
        real_prepare = sched_mod.SuperDSCScheduling.prepare_kernel
        real_prepare_all = sched_mod.prepare_spyre_kernels

        def layout_facts(layout):
            return (
                tuple(layout.size),
                tuple(layout.stride),
                tuple(layout.device_layout.device_size),
                tuple(layout.device_layout.stride_map),
            )

        def prepare(backend, bundle, kernel):
            graph = V.graph
            if rejected:
                plan, copy_name, copy_buffer, original_facts, unrelated, _ = rejected[0]
                assert all(
                    edge[0] != plan.source_name
                    for edge in materialized_lx_relayouts(graph)
                )
                source_layout = graph.get_buffer(plan.source_name).get_layout()
                copy_layout = copy_buffer.get_layout()
                for layout in (source_layout, copy_layout):
                    assert "lx" not in layout.allocation
                    assert layout.lx_view is None
                assert graph.get_buffer(copy_name) is copy_buffer
                assert copy_buffer in graph.operations
                assert layout_facts(copy_layout) == original_facts
                assert storage_has_single_writer(graph, copy_buffer)
                for name in plan.consumer_names:
                    reads = {
                        dep.name for dep in op_read_writes(graph.get_buffer(name)).reads
                    }
                    assert copy_name in reads and plan.source_name not in reads
                assert plan.source_name in {
                    dep.name for dep in op_read_writes(copy_buffer).reads
                }
                for name, allocation, view in unrelated:
                    layout = graph.get_buffer(name).get_layout()
                    assert layout.allocation == allocation and layout.lx_view == view
                restored.append(copy_name)
            else:
                for copy_name, plan in materialized_lx_relayouts(graph).values():
                    if plan.source_name not in prefix_sources:
                        continue
                    for node in sched_mod._all_scheduler_nodes([bundle]):
                        if not any(
                            dep.name == copy_name for dep in node.read_writes.writes
                        ):
                            continue
                        copy_buffer = graph.get_buffer(copy_name)
                        assert storage_has_single_writer(graph, copy_buffer)
                        unrelated = []
                        for op in graph.operations:
                            name = op.get_name()
                            layout = op.get_layout()
                            if (
                                name not in (plan.source_name, copy_name)
                                and hasattr(layout, "allocation")
                                and "lx" in layout.allocation
                            ):
                                unrelated.append(
                                    (name, dict(layout.allocation), layout.lx_view)
                                )
                        rejected.append(
                            (
                                plan,
                                copy_name,
                                copy_buffer,
                                layout_facts(copy_buffer.get_layout()),
                                unrelated,
                                kernel,
                            )
                        )
                        kernel.failed_node = node
                        raise ValueError("prefix gather test: reject LX finalization")
            result = real_prepare(backend, bundle, kernel)
            if rejected:
                plan, copy_name, _, _, _, _ = rejected[0]
                for name, arg in kernel.spyre_kernel_args:
                    if name in (plan.source_name, copy_name):
                        assert "lx" not in arg.allocation
                        assert arg.work_division is None
            return result

        def prepare_all(nodes):
            result = real_prepare_all(nodes)
            if rejected:
                failed_kernel = rejected[0][-1]
                for node in result:
                    prepared = getattr(node, "prepared_kernel", None)
                    if prepared is not None:
                        assert prepared is not failed_kernel
                        final_kernels.append(prepared)
            return result

        monkeypatch.setattr(sched_mod.SuperDSCScheduling, "prepare_kernel", prepare)
        monkeypatch.setattr(sched_mod, "prepare_spyre_kernels", prepare_all)
        monkeypatch.setattr(passes_mod, "prepare_spyre_kernels", prepare_all)

    rows, physical, logical, outputs = 512, 3328, 3200, 512

    def fn(value, weight):
        with spyre_hint(work_div={"M": 8, "P": 4}):
            backing = torch.neg(value)
        with spyre_hint(work_div={"M": 4, "N": 8}):
            return torch.nn.functional.linear(backing[:, :logical], weight)

    for name, size in (("M", rows), ("P", physical), ("K", logical), ("N", outputs)):
        _pnd.declare_tensor_dim(name, size)
    selected = (torch.arange(outputs) * 37) % logical
    selected[-1] = logical - 1
    weight = torch.zeros(outputs, logical, dtype=torch.float16)
    weight[torch.arange(outputs), selected] = 1
    device_weight = _pnd.name_tensor_dims(weight.to("spyre"), ["N", "K"])
    eps = get_elem_in_stick(torch.float16)
    assert eps == 64
    layout = SpyreTensorLayout(
        device_size=[physical // eps, rows, eps],
        stride_map=[eps, physical, 1],
        device_dtype=get_device_dtype(torch.float16),
    )
    compiled = torch.compile(fn, dynamic=False, fullgraph=True)
    options = {**_COOPT, "sencores": 32, "lx_same_tile_group_relayout": False}
    if mode == "no_copy":
        options["lx_solver_relayout_groups_per_edge"] = 0
    with config.patch(options):
        for tail in (512.0, float("nan")):
            # One selected element per output makes the reduction exact. Every
            # row differs; a compacted stride or a read into the tail is visible.
            host = (
                (torch.arange(rows * physical).reshape(rows, physical) * 17) % 251 - 125
            ).to(torch.float16)
            host[:, logical:] = tail
            value = _pnd.name_tensor_dims(
                host.to("spyre", device_layout=layout), ["M", "P"]
            )
            out = compiled(value, device_weight).cpu()
            expected = -host[:, selected]
            control = torch.nn.functional.linear(
                -host[:, :logical].float(), weight.float()
            )
            assert torch.equal(control, expected.float())
            assert torch.isfinite(out).all()
            assert torch.equal(out, expected)

    assert prefix_sources and source_residency, (
        "the fixture did not reach the prefix edge"
    )
    if mode == "no_copy":
        assert removed_candidates, "the negative arm removed no prefix candidates"
        assert all(address is None for _, address in source_residency), (
            f"a prefix producer resided without a served gather: {source_residency}"
        )
        observed.assert_nothing_emitted()
        return
    if mode == "demoted":
        assert rejected and restored and final_kernels
        assert observed.demotions and not observed.emitted and not observed.emitted_ops
    else:
        observed.assert_emitted_in_lx(expected_plans=1)
    (plan,) = observed.plans
    assert plan.num_cores == plan.destination_view.num_cores == 32
    assert observed.source_layouts[plan.edge] == (
        [physical // eps, rows, eps],
        [eps, physical, 1],
        [physical, 1],
    )
    assert observed.copy_layouts[plan.edge] == (
        [rows, physical],
        [physical, 1],
        [physical // eps, rows, eps],
        [eps, physical, 1],
    )
    assert dict(plan.source_view.work_slice_dims) == {0: 4, 1: 8}
    destination = dict(plan.destination_view.work_slice_dims)
    assert destination.get(0, 1) == 1, "the linear reader must not split K"
    assert destination.get(1, 1) == 4
    assert plan.source_footprint_bytes == rows * physical * 2 // 32
    assert plan.destination_footprint_bytes == rows * physical * 2 // 4
    assert (
        plan.source_address + plan.source_footprint_bytes <= plan.destination_address
        or plan.destination_address + plan.destination_footprint_bytes
        <= plan.source_address
    ), "the full-backing gather must have separate LX storage"
