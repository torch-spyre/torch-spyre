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

"""The Annealing backend: its plans satisfy the model they solve, checked by
CP-SAT with every variable fixed to the plan."""

import copy
import sys
from pathlib import Path

import pytest

pytest.importorskip("ortools")

sys.path.insert(0, str(Path(__file__).parent))

from google.protobuf import text_format  # noqa: E402
from ortools.sat import cp_model_pb2  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402

from buffer_spec import build_graph  # noqa: E402
from cooptimization_capture_loader import load_captures  # noqa: E402
from torch_spyre._inductor.scratchpad.backend import (  # noqa: E402
    Annealing,
    _AnnealSearch,
)
from torch_spyre._inductor.scratchpad.buffer_graph import SolveSettings  # noqa: E402
from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy  # noqa: E402


class _Recorded(Annealing):
    """Annealing that keeps each ladder stage's model and plan."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.stages = []

    def Solve(self, objective: bool = False) -> int:
        status = super().Solve(objective)
        self.stages.append((self._proto(), self._values, status))
        return status


def _holds(proto, values) -> bool:
    """Whether the model accepts ``values``, by CP-SAT with each var fixed."""
    model = cp_model.CpModel()
    model.proto.parse_text_format(str(proto))
    for i, x in enumerate(values):
        model.add(model.get_int_var_from_proto_index(i) == x)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    return solver.solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def _in_place_model():
    """Four boxes in a 16-unit LX: ``a`` alive [0, 2), ``b`` [4, 6), and an
    in-place pair ``p`` [2, 4) -> ``c`` [4, 6) whose merge holds them at one
    offset. ``c`` is alive with ``b``, ``p`` with nothing."""
    model = cp_model.CpModel()
    xs, ys, vars_ = [], [], {}
    for name, (start, end, size) in {
        "a": (0, 2, 4),
        "b": (4, 6, 4),
        "p": (2, 4, 4),
        "c": (4, 6, 4),
    }.items():
        on = model.new_bool_var(f"in_buffer_{name}")
        off = model.new_int_var(0, 15, f"off_{name}")
        top = model.new_int_var(0, 15 + size, f"top_{name}")
        model.add(top == off + size)
        model.add(off + size <= 16).only_enforce_if(on)
        xs.append(model.new_optional_interval_var(start, end - start, end, on, ""))
        ys.append(model.new_optional_interval_var(off, size, top, on, ""))
        vars_[name] = (on, off)
    merge = model.new_bool_var("merge_p_c")
    model.add(vars_["p"][1] == vars_["c"][1]).only_enforce_if(merge)
    model.add_bool_and([vars_["p"][0], vars_["c"][0]]).only_enforce_if(merge)
    model.add_no_overlap_2d(xs, ys)
    model.minimize(sum(4 * (1 - on) for on, _ in vars_.values()))
    return model, vars_, merge


def _search(model) -> _AnnealSearch:
    proto = text_format.Parse(str(model.proto), cp_model_pb2.CpModelProto())
    return _AnnealSearch(proto, max_nodes=0)


def _plan(model, vars_, merge, offsets):
    """Every var of ``model``: all resident and merged, at ``offsets``."""
    values = [0] * len(model.proto.variables)
    for name, (on, off) in vars_.items():
        values[on.index] = 1
        values[off.index] = offsets[name]
    values[merge.index] = 1
    for i, var in enumerate(model.proto.variables):
        if var.name.startswith("top_"):
            name = var.name[len("top_") :]
            values[i] = offsets[name] + 4
    return values


def test_an_in_place_group_packs_clear_of_its_members_lifetimes():
    # First fit in offset order lowers p to 0 (nothing is alive with it);
    # c must follow p, and c is alive with b at 0. The group packs as one box
    # over [2, 6), so it lands above b and the hint's plan survives.
    model, vars_, merge = _in_place_model()
    hint = _plan(model, vars_, merge, {"a": 0, "b": 0, "p": 4, "c": 4})
    assert _holds(model.proto, hint)
    search = _search(model)
    assert search.anneal(seed=0, budget_s=0.0, start=hint)
    assert search.best == 0
    assert _holds(model.proto, search.best_a)
    off = {name: search.best_a[o.index] for name, (_, o) in vars_.items()}
    assert off["p"] == off["c"] and off["c"] >= off["b"] + 4


def test_a_plan_at_the_objective_bound_is_proved():
    model, vars_, merge = _in_place_model()
    hint = _plan(model, vars_, merge, {"a": 0, "b": 0, "p": 4, "c": 4})
    search = _search(model)
    assert search.anneal(seed=0, budget_s=5.0, start=hint)
    assert search.proved and search.best == 0


def _buffers(name):
    return copy.deepcopy(load_captures()[name][0].buffers)


@pytest.mark.parametrize("name", ["softmax", "mlp", "sdpa"])
def test_every_ladder_stage_plan_satisfies_its_model(name):
    buffers = _buffers(name)
    tot = sum(-(-b.size // b.core_divisions[0].output_partition) for b in buffers)
    reasons = {b.name: b.residency_reason for b in buffers if b.residency_reason}
    graph = build_graph(buffers, 128, tot // 2 // 128, SolveSettings(), reasons)
    backend = _Recorded(seed=0, budget_s=0.3)
    graph.solve(model=backend, strategy=LazyStrategy())
    assert backend.stages
    for proto, values, status in backend.stages:
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert _holds(proto, values)


def test_the_running_state_matches_a_fresh_evaluation(monkeypatch):
    monkeypatch.setenv("SPYRE_ANNEAL_VERIFY", "3")
    graph = build_graph(_buffers("sdpa"), 128, 1 << 12, SolveSettings(), {})
    graph.solve(model=Annealing(seed=1, budget_s=0.3), strategy=LazyStrategy())
