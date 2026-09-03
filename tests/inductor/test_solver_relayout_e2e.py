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

import pytest

pytest.importorskip("ortools")

import torch

import torch_spyre  # noqa: F401
from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad import allocator as alloc_mod


_COOPT = {"co_optimizing_lx_planning": True, "layout_solver": "cpsat"}


def _no_matches(self, consumer_op, consumer_divs, parent_names, *args, **kwargs):
    return {parent: [] for parent in parent_names}


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

        self.plans = []
        self.demotions = []
        self.emitted = set()  # (source_lx_address, destination_lx_address)
        real_materialize = alloc_mod.materialize_lx_relayouts

        def spy_materialize(graph, plans):
            self.plans.extend(plans)
            return real_materialize(graph, plans)

        real_demote = sched_mod.demote_lx_relayout_group

        def spy_demote(graph, source_name, reason):
            self.demotions.append((source_name, reason))
            return real_demote(graph, source_name, reason)

        real_identity = opspec_mod.is_lx_relayout_identity

        def spy_identity(op, args):
            result = real_identity(op, args)
            if result:
                source, destination = args
                assert set(source.allocation) == {"lx"}, source.allocation
                assert set(destination.allocation) == {"lx"}, destination.allocation
                self.emitted.add(
                    (source.allocation["lx"], destination.allocation["lx"])
                )
            return result

        monkeypatch.setattr(alloc_mod, "materialize_lx_relayouts", spy_materialize)
        monkeypatch.setattr(sched_mod, "demote_lx_relayout_group", spy_demote)
        monkeypatch.setattr(sdsc_mod, "is_lx_relayout_identity", spy_identity)
        monkeypatch.setattr(sk_mod, "is_lx_relayout_identity", spy_identity)
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

    def assert_nothing_emitted(self) -> None:
        """No plan, no demotion, and no identity op codegen took for an LX
        relayout: the compile ran exactly as it would without the feature."""
        assert self.plans == [], f"no relayout may materialize: {self.plans}"
        assert self.demotions == []
        assert self.emitted == set(), (
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


def test_flag_off_materializes_nothing(monkeypatch):
    """The kill switch: with the feature default-on, SPYRE_LX_SOLVER_RELAYOUT=0
    (config.lx_solver_relayout=False) must disarm every relayout decision and
    leave the co-optimizing solve exactly as it was before this feature."""
    observed = _arm_forced(monkeypatch)

    def fn(t):
        return torch.relu(torch.neg(t)) + 1.0

    torch.manual_seed(0)
    host = torch.randn(8, 256, 512, dtype=torch.float16)
    x = host.to("spyre")
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "layout_solver": "cpsat",
            "lx_solver_relayout": False,
        }
    ):
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


def test_one_source_two_consumers_two_edges(monkeypatch):
    """relu(h) and abs(h) from one h with both edges forced: per-edge design,
    so each consumer gets its own shuffle (two plans from the same source),
    and each output is correct. A single group move serving both consumers
    would be cheaper; that is the deferred multi-consumer extension."""
    forced = _arm_forced(monkeypatch)
    recorded = forced.plans

    def fn(t):
        h = torch.neg(t)
        return torch.relu(h), torch.abs(h)

    torch.manual_seed(0)
    ht = torch.randn(8, 256, 512, dtype=torch.float16)
    with config.patch(_COOPT):
        o1, o2 = torch.compile(fn, dynamic=False)(ht.to("spyre"))

    forced.assert_emitted_in_lx(expected_plans=2)
    assert {p.source_name for p in recorded} == {recorded[0].source_name}
    assert len({p.consumer_names for p in recorded}) == 2
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


@pytest.mark.skip(
    reason="co-opt+cpsat crashes on any coarse-tiled graph before relayout "
    "is even considered (symbolic relationals in coarse_underfill_eff and "
    "the LX-spill derate reached by tiled ops; #3810 review finding). "
    "Unskip when the substrate handles symbolic coarse-tile features; the "
    "gate itself is covered by unit tests in "
    "test_solver_relayout_candidates.py."
)
def test_coarse_tiled_edges_are_never_offered(monkeypatch):
    """A coarse-tiled graph must enumerate NO relayout candidates.

    Both ops sit in one coarse-tile hint scope, so the producer's buffer is
    per-tile scratch and the edge lives inside the loop. The fitted law has
    no loop_trip factor and the committed-path planner forbids in-loop
    relayouts; the solver path must decline at enumeration (loop_info gate),
    not merely decline to fire."""
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint

    tables = []
    orig_tables = alloc_mod.CoOptimizingAllocator._cd_parent_relayouts

    def spy_tables(self, *args, **kwargs):
        out = orig_tables(self, *args, **kwargs)
        if out:
            tables.append(out)
        return out

    monkeypatch.setattr(
        alloc_mod.CoOptimizingAllocator, "_cd_parent_relayouts", spy_tables
    )

    recorded = []
    real_materialize = alloc_mod.materialize_lx_relayouts

    def spy_mat(graph, plans):
        recorded.extend(plans)
        return real_materialize(graph, plans)

    monkeypatch.setattr(alloc_mod, "materialize_lx_relayouts", spy_mat)

    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()

    def fn(t, u):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                z = torch.abs(t) + u
                return z * 2

    torch.manual_seed(0)
    ha = torch.randn(512, 256, dtype=torch.float16)
    hb = torch.randn(512, 256, dtype=torch.float16)
    for name, size in (("A", 512), ("B", 256)):
        _pnd.declare_tensor_dim(name, size)
    a = _pnd.name_tensor_dims(ha.to("spyre"), ["A", "B"])
    b = _pnd.name_tensor_dims(hb.to("spyre"), ["A", "B"])
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "layout_solver": "cpsat",
        }
    ):
        out = torch.compile(fn, dynamic=False)(a, b)

    assert tables == [], f"coarse-tiled edges were offered relayout: {tables}"
    assert recorded == []
    ref = ((torch.abs(ha.float()) + hb.float()) * 2).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=2**-8, atol=2**-6)
