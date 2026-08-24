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


def _no_matches(self, consumer_op, consumer_divs, parent_names, *args, **kwargs):
    return {parent: [] for parent in parent_names}


def test_solver_relayout_materializes_and_runs(monkeypatch):
    recorded = []
    real_materialize = alloc_mod.materialize_lx_relayouts

    def spy(graph, plans):
        recorded.extend(plans)
        return real_materialize(graph, plans)

    monkeypatch.setattr(alloc_mod, "materialize_lx_relayouts", spy)
    monkeypatch.setattr(
        alloc_mod.CoOptimizingAllocator, "_cd_parent_matches", _no_matches
    )

    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()

    def fn(t):
        return torch.relu(torch.neg(t))

    torch.manual_seed(0)
    host = torch.randn(8, 256, 512, dtype=torch.float16)
    x = host.to("spyre")
    with config.patch(
        {
            "co_optimizing_lx_planning": True,
            "layout_solver": "cpsat",
            "lx_solver_relayout": True,
        }
    ):
        out = torch.compile(fn, dynamic=False)(x)

    # The solver fired the neg -> relu edge and the commit path handed
    # materialize_lx_relayouts a complete plan.
    assert len(recorded) == 1, f"expected one materialized relayout: {recorded}"
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
    recorded = []
    real_materialize = alloc_mod.materialize_lx_relayouts

    def spy(graph, plans):
        recorded.extend(plans)
        return real_materialize(graph, plans)

    monkeypatch.setattr(alloc_mod, "materialize_lx_relayouts", spy)
    monkeypatch.setattr(
        alloc_mod.CoOptimizingAllocator, "_cd_parent_matches", _no_matches
    )

    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()

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

    assert recorded == [], "no relayout may materialize with the flag off"
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
    only patch below is a read-only spy."""
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint

    recorded = []
    real_materialize = alloc_mod.materialize_lx_relayouts

    def spy(graph, plans):
        recorded.extend(plans)
        return real_materialize(graph, plans)

    monkeypatch.setattr(alloc_mod, "materialize_lx_relayouts", spy)

    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()

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
            "lx_solver_relayout": True,
        }
    ):
        out = torch.compile(fn, dynamic=False)(x)

    assert len(recorded) == 1, f"expected one natural relayout: {recorded}"
    plan = recorded[0]
    # The canonical measured pair: producer {B:4, M:2} vs consumer {B:2, M:4}
    # on device [256, 8, 8, 64] (dim 0 <- M, dim 2 <- B).
    assert dict(plan.source_view.work_slice_dims) == {0: 2, 2: 4}
    assert dict(plan.destination_view.work_slice_dims) == {0: 4, 2: 2}
    assert plan.num_cores == 8
    ref = torch.relu(torch.neg(host.float())).to(torch.float16)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
