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

"""Pure input-stage proposals; native coordinate proof, no device timing."""

from types import SimpleNamespace
from dataclasses import replace

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep, StarDep
from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction
from torch._inductor.runtime.hints import ReductionHint

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor import config
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad import allocator as am
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.pass_utils import PerCoreView


T, H = sympy.symbols("token hidden", integer=True, nonnegative=True)
C = sympy.Symbol("core_id")
WANTED = PerCoreView(
    # The native layout below is [H/64, T, 64], not host [T,H].
    ((0, 4), (1, 8)),
    ((0, sympy.Mod(C, 4)), (1, sympy.Mod(sympy.floor(C / 4), 8))),
    32,
)


@pytest.fixture(params=[torch.float16, torch.bfloat16])
def case(monkeypatch, request):
    dtype = request.param
    device_layout = SpyreTensorLayout(
        [512, 2816], [2816, 1], dtype, [0, 1], ElementArrangement.STANDARD
    )

    def make(name):
        return ComputedBuffer(
            name=name,
            layout=FixedTiledLayout(
                torch.device("cpu"),
                dtype,
                [512, 2816],
                [2816, 1],
                device_layout,
            ),
            data=(
                Pointwise(
                    device=torch.device("cpu"),
                    dtype=dtype,
                    inner_fn=lambda index: index[0],
                    ranges=[512, 2816],
                )
                if name == "stage"
                else Reduction(
                    device=torch.device("cpu"),
                    dtype=dtype,
                    inner_fn=lambda index, reduction_index: index[0],
                    ranges=[512, 2816],
                    reduction_ranges=[64],
                    reduction_type=BATCH_MATMUL_OP,
                    src_dtype=dtype,
                    reduction_hint=ReductionHint.DEFAULT,
                )
            ),
        )

    def dep(name):
        return MemoryDep(name, T * 2816 + H, (T, H), (512, 2816))

    stage, left, right = [make(name) for name in ("stage", "left", "right")]
    stage.iteration_space_ownership = SimpleNamespace(physical_core_count=32)
    rws = {
        id(stage): SimpleNamespace(reads=[dep("input")], writes=[dep("stage")]),
        id(left): SimpleNamespace(reads=[dep("stage")], writes=[dep("left")]),
        id(right): SimpleNamespace(reads=[dep("stage")], writes=[dep("right")]),
    }
    graph = SimpleNamespace(
        operations=[stage, left, right],
        graph_input_names={"input"},
        get_output_names=lambda: ["left", "right"],
    )
    committed = []
    observed_overrides = []
    views = {id(op): (WANTED, False, True) for op in graph.operations}
    monkeypatch.setattr(am, "op_read_writes", lambda op: rws[id(op)])
    monkeypatch.setattr(
        am,
        "iteration_space_from_op",
        lambda op: {T: sympy.Integer(512), H: sympy.Integer(2816)},
    )
    monkeypatch.setattr(am, "_has_work_div_hint", lambda *args: False)
    monkeypatch.setattr(
        am,
        "work_division_context_for_op",
        lambda *args, **kwargs: SimpleNamespace(is_legal=lambda splits: True),
    )
    monkeypatch.setattr(am, "_is_matmul_op", lambda op: op in (left, right))

    def view(op, *args, ownership_override=None):
        observed_overrides.append((op, ownership_override))
        if op is stage and ownership_override is stage.iteration_space_ownership:
            return None, False, False
        return views[id(op)]

    monkeypatch.setattr(am, "_per_core_view_on_buf", view)
    monkeypatch.setattr(
        am,
        "commit_tensor_work_division",
        lambda op, candidate: committed.append((op, candidate)),
    )
    with config.patch(
        {
            "consumer_compatible_input_staging": True,
            "lx_planning": True,
            "lx_planner_relayout": True,
            "co_optimizing_lx_planning": False,
            "ktir_emitter": False,
            "sencores": 32,
        }
    ):
        yield SimpleNamespace(
            graph=graph,
            stage=stage,
            left=left,
            right=right,
            rws=rws,
            dep=dep,
            views=views,
            committed=committed,
            observed_overrides=observed_overrides,
        )


def test_proposes_both_readers_using_real_coordinate_projection(case):
    old = case.stage.iteration_space_ownership
    proposals = am._reader_compatible_input_stage_proposal(case.graph, {})
    assert set(proposals) == {"stage"}
    assert case.committed == []
    assert case.stage.iteration_space_ownership is old
    candidate = proposals["stage"]
    assert candidate.work_slices[T] == 8 and candidate.work_slices[H] == 4
    # Core 1 has the first token slice and second hidden slice, not vice versa.
    assert int(candidate.core_id_to_work_slice[T].subs(C, 1)) == 0
    assert int(candidate.core_id_to_work_slice[H].subs(C, 1)) == 1


def test_generated_hint_applicability_reaches_the_real_presence_check(
    case, monkeypatch
):
    from torch_spyre._inductor.propagate_hints import exclude_op_hint_keys
    from torch_spyre._inductor.work_division import _has_work_div_hint

    origin = torch.fx.Graph().placeholder("activation")
    origin.meta["custom"] = {
        "_hint_0": {"work_div": {"T": 32}},
        "_hint_1": {"work_div": {"H": 4}},
    }
    case.stage.origins = (origin,)
    case.left.origins = (origin,)
    monkeypatch.setattr(am, "_has_work_div_hint", _has_work_div_hint)
    # Real user constraints still exclude a stage, even without resolved names.
    assert am._reader_compatible_input_stage_proposal(case.graph, {}) == {}
    exclude_op_hint_keys(case.stage, "work_div")
    assert set(am._reader_compatible_input_stage_proposal(case.graph, {})) == {"stage"}
    assert _has_work_div_hint(case.left)
    assert case.committed == []


def test_trial_reader_ownership_is_used_without_committing(case):
    left, right = object(), object()
    overrides = {"left": left, "right": right}
    assert "stage" in am._reader_compatible_input_stage_proposal(case.graph, overrides)
    assert (case.left, left) in case.observed_overrides
    assert (case.right, right) in case.observed_overrides
    assert overrides == {"left": left, "right": right}
    assert case.committed == []


def test_already_matching_stage_adds_no_trial(case, monkeypatch):
    monkeypatch.setattr(
        am, "_per_core_view_on_buf", lambda *args, **kwargs: (WANTED, False, True)
    )
    assert am._reader_compatible_input_stage_proposal(case.graph, {}) == {}
    assert case.committed == []


def test_changed_trial_reader_view_changes_the_stage_candidate(case, monkeypatch):
    new_view = PerCoreView(
        ((0, 2), (1, 16)),
        ((0, sympy.Mod(C, 2)), (1, sympy.Mod(sympy.floor(C / 2), 16))),
        32,
    )
    reader_override = object()

    def view(op, *args, ownership_override=None):
        if op is case.stage:
            if ownership_override is case.stage.iteration_space_ownership:
                return None, False, False
            result = new_view if ownership_override.work_slices[T] == 16 else WANTED
        else:
            result = new_view if ownership_override is reader_override else WANTED
        return result, False, True

    monkeypatch.setattr(am, "_per_core_view_on_buf", view)
    baseline = am._reader_compatible_input_stage_proposal(case.graph, {})
    trial = am._reader_compatible_input_stage_proposal(
        case.graph, {"left": reader_override, "right": reader_override}
    )
    assert baseline["stage"].work_slices[T] == 8
    assert trial["stage"].work_slices[T] == 16
    assert trial["stage"].work_slices[H] == 2
    # Changing just one reader must not reuse the baseline's agreement.
    assert (
        am._reader_compatible_input_stage_proposal(
            case.graph, {"left": reader_override}
        )
        == {}
    )
    assert case.committed == []


def test_producer_core_domain_is_taken_from_same_trial(case):
    override = SimpleNamespace(physical_core_count=16)
    assert (
        am._reader_compatible_input_stage_proposal(case.graph, {"stage": override})
        == {}
    )
    assert case.stage.iteration_space_ownership.physical_core_count == 32
    assert case.committed == []


@pytest.mark.parametrize("failure", [ValueError, Unsupported])
def test_unrepresentable_projection_declines(case, monkeypatch, failure):
    def fail(*args):
        raise failure("unrepresentable projection")

    monkeypatch.setattr(am, "work_division_from_view", fail)
    assert am._reader_compatible_input_stage_proposal(case.graph, {}) == {}
    assert case.committed == []


def test_projection_implementation_error_is_not_hidden(case, monkeypatch):
    def fail(*args):
        raise TypeError("wrong projection call")

    monkeypatch.setattr(am, "work_division_from_view", fail)
    with pytest.raises(TypeError, match="wrong projection call"):
        am._reader_compatible_input_stage_proposal(case.graph, {})
    assert case.committed == []


def test_unavailable_legal_domain_keeps_other_proposals(case, monkeypatch):
    def fail(*args, **kwargs):
        raise Unsupported("span cannot be represented")

    compact = {"left": object()}
    monkeypatch.setattr(am, "work_division_context_for_op", fail)
    monkeypatch.setattr(am, "_compact_work_division_proposals", lambda graph: [compact])
    assert am._reader_compatible_input_stage_proposal(case.graph, {}) == {}
    assert am._placement_work_division_proposals(case.graph) == [compact]
    assert case.committed == []


def test_unpriced_reader_compute_declines(case):
    case.left.data = replace(case.left.data, reduction_type="unpriced_matmul")
    assert am._reader_compatible_input_stage_proposal(case.graph, {}) == {}
    assert case.committed == []


def test_different_device_storage_declines(case):
    case.stage.layout.device_layout = SpyreTensorLayout(
        [512, 2816], [2816, 1], torch.float32, [0, 1], ElementArrangement.STANDARD
    )
    assert am._reader_compatible_input_stage_proposal(case.graph, {}) == {}
    assert case.committed == []


def test_reduction_split_on_reader_does_not_make_its_input_partial(case):
    case.views[id(case.left)] = (WANTED, True, True)
    case.views[id(case.right)] = (WANTED, True, True)
    assert "stage" in am._reader_compatible_input_stage_proposal(case.graph, {})
    assert case.committed == []


@pytest.mark.parametrize(
    "flag,value",
    [
        ("consumer_compatible_input_staging", False),
        ("lx_planning", False),
        ("lx_planner_relayout", False),
        ("co_optimizing_lx_planning", True),
        ("ktir_emitter", True),
    ],
)
def test_disabled_or_incompatible_modes_do_not_inspect_graph(flag, value):
    options = {
        "consumer_compatible_input_staging": True,
        "lx_planning": True,
        "lx_planner_relayout": True,
        "co_optimizing_lx_planning": False,
        "ktir_emitter": False,
    }
    options[flag] = value
    with config.patch(options):
        assert am._reader_compatible_input_stage_proposal(object(), {}) == {}


@pytest.mark.parametrize(
    "reason",
    [
        "loop",
        "hint",
        "output",
        "internal",
        "mutated_input",
        "mutated_stage",
        "alias",
        "unknown_contract",
        "second_writer",
        "dtype",
        "star_reader",
        "non_matmul",
        "partial",
        "unrepresentable",
        "different_owners",
        "core_count",
        "illegal",
        "write_mismatch",
        "projection_failure",
    ],
)
def test_rejections_preserve_ordinary_choice(case, monkeypatch, reason):
    if reason == "loop":
        case.stage.loop_info = object()
    elif reason == "hint":
        monkeypatch.setattr(am, "_has_work_div_hint", lambda *args: {T: 32})
    elif reason == "output":
        case.graph.get_output_names = lambda: ["stage"]
    elif reason == "internal":
        case.graph.graph_input_names = set()
    elif reason in ("mutated_input", "mutated_stage"):
        case.right.get_mutation_names = lambda: [
            "input" if reason == "mutated_input" else "stage"
        ]
    elif reason == "alias":
        case.right.get_inputs_that_alias_output = lambda: ["stage"]
    elif reason == "unknown_contract":
        case.right.get_mutation_names = None
    elif reason == "second_writer":
        case.rws[id(case.right)].writes = [case.dep("stage")]
    elif reason == "dtype":
        case.stage.layout.dtype = torch.float32
    elif reason == "star_reader":
        case.rws[id(case.right)].reads = [StarDep("stage")]
    elif reason == "non_matmul":
        monkeypatch.setattr(am, "_is_matmul_op", lambda op: op is case.left)
    elif reason == "partial":
        case.views[id(case.stage)] = (WANTED, True, True)
    elif reason == "unrepresentable":
        case.views[id(case.right)] = (None, False, False)
    elif reason == "different_owners":
        case.views[id(case.right)] = (
            PerCoreView(
                ((0, 4), (1, 8)),
                ((0, sympy.Mod(sympy.floor(C / 8), 4)), (1, sympy.Mod(C, 8))),
                32,
            ),
            False,
            True,
        )
    elif reason == "core_count":
        case.stage.iteration_space_ownership.physical_core_count = 16
    elif reason == "illegal":
        monkeypatch.setattr(
            am,
            "work_division_context_for_op",
            lambda *args, **kwargs: SimpleNamespace(is_legal=lambda splits: False),
        )
    elif reason == "write_mismatch":
        case.views[id(case.stage)] = (None, False, False)
    elif reason == "projection_failure":
        monkeypatch.setattr(
            am,
            "device_coordinates",
            lambda *args: [T + H, sympy.floor(H / 64), sympy.Mod(H, 64)],
        )
    assert am._reader_compatible_input_stage_proposal(case.graph, {}) == {}
    assert case.committed == []


@pytest.mark.parametrize("compact_present", [False, True])
def test_stage_proposals_use_each_trial_and_do_not_require_matmul_changes(
    monkeypatch, compact_present
):
    compact = {"reader": object()} if compact_present else {}
    seen = []
    stages = [object(), object()]
    monkeypatch.setattr(
        am,
        "_compact_work_division_proposals",
        lambda graph: [compact] if compact_present else [],
    )

    def stage(graph, overrides):
        seen.append(dict(overrides))
        return {"stage": stages[len(seen) - 1]}

    monkeypatch.setattr(am, "_reader_compatible_input_stage_proposal", stage)
    proposals = am._placement_work_division_proposals(object())
    assert seen == ([{}, compact] if compact_present else [{}])
    if compact_present:
        assert proposals == [
            compact,
            {"stage": stages[0]},
            {**compact, "stage": stages[1]},
        ]
        assert set(compact) == {"reader"}
    else:
        assert proposals == [{"stage": stages[0]}]


def test_no_eligible_stage_preserves_existing_trial_count(monkeypatch):
    compact = {"reader": object()}
    monkeypatch.setattr(am, "_compact_work_division_proposals", lambda graph: [compact])
    monkeypatch.setattr(
        am, "_reader_compatible_input_stage_proposal", lambda graph, overrides: {}
    )
    assert am._placement_work_division_proposals(object()) == [compact]


def test_no_proposals_reports_no_commit(monkeypatch):
    monkeypatch.setattr(am, "_placement_work_division_proposals", lambda graph: [])
    assert am.ScratchpadAllocator._select_work_division(object(), object()) is False


@pytest.mark.parametrize(
    "costs,winner",
    [
        ((100, 120, 130), None),  # Matching readers can cost more elsewhere.
        ((100, 100, 100), None),  # The ordinary choice wins all ties.
        ((100, 80, 90), 0),
        ((100, 80, 60), 1),  # Do not commit the first improvement prematurely.
        ((100, float("inf"), 60), 1),
        ((100, float("nan"), 60), 1),
        ((100, -1, 60), 1),
        ((float("inf"), 0, 0), None),  # No comparison without a known control.
    ],
)
def test_only_placement_priced_winner_commits(monkeypatch, costs, winner):
    old = object()
    stage = SimpleNamespace(get_name=lambda: "stage", iteration_space_ownership=old)
    graph = SimpleNamespace(operations=[stage])
    candidates = [{"stage": object()}, {"stage": object()}]
    monkeypatch.setattr(
        am, "_placement_work_division_proposals", lambda graph: candidates
    )
    evaluated, committed = [], []

    def prepare(graph, *, ownership_overrides):
        assert stage.iteration_space_ownership is old
        assert committed == []
        return ownership_overrides

    def cost(graph, allocation, plans, overrides):
        assert allocation is overrides and plans is overrides
        assert stage.iteration_space_ownership is old
        assert committed == []
        evaluated.append(overrides)
        return costs[len(evaluated) - 1]

    # Fake the solver/cost boundary, not the production selection function.
    # These are policy tests, not proofs of placement or cost-model accuracy.
    allocator = SimpleNamespace(
        _prepare_fixed_buffers=prepare,
        _build_solver=lambda buffers: SimpleNamespace(plan_layout=lambda: buffers),
        _finalize_lx_relayout_allocation=lambda allocation: allocation,
        _allocation_cost=cost,
    )
    monkeypatch.setattr(
        am,
        "commit_tensor_work_division",
        lambda op, division: committed.append((op, division)),
    )
    changed = am.ScratchpadAllocator._select_work_division(allocator, graph)
    assert changed is (winner is not None)
    assert evaluated == ([{}] if costs[0] == float("inf") else [{}, *candidates])
    assert committed == (
        [] if winner is None else [(stage, candidates[winner]["stage"])]
    )
    assert stage.iteration_space_ownership is old


@pytest.mark.parametrize("failure_at", ["baseline", "trial", "cost", "assertion"])
def test_trial_recovery_preserves_the_new_parent_contract(monkeypatch, failure_at):
    old = object()
    stage = SimpleNamespace(get_name=lambda: "stage", iteration_space_ownership=old)
    graph = SimpleNamespace(operations=[stage])
    proposals = [{"stage": object()}, {"stage": object()}]
    monkeypatch.setattr(
        am, "_placement_work_division_proposals", lambda graph: proposals
    )
    committed = []

    def prepare(graph, *, ownership_overrides):
        assert stage.iteration_space_ownership is old
        if failure_at == "baseline" and not ownership_overrides:
            raise am.Unsupported("baseline unavailable")
        if ownership_overrides is proposals[0]:
            if failure_at == "trial":
                raise am.Unsupported("optional trial unavailable")
            if failure_at == "assertion":
                raise AssertionError("ownership invariant")
        return ownership_overrides

    def cost(graph, allocation, plans, overrides):
        if failure_at == "cost" and overrides is proposals[0]:
            raise ValueError("cost unavailable")
        return 100 if not overrides else 60

    allocator = SimpleNamespace(
        _prepare_fixed_buffers=prepare,
        _build_solver=lambda buffers: SimpleNamespace(plan_layout=lambda: buffers),
        _finalize_lx_relayout_allocation=lambda allocation: allocation,
        _allocation_cost=cost,
    )
    monkeypatch.setattr(
        am, "commit_tensor_work_division", lambda op, value: committed.append(value)
    )
    if failure_at == "assertion":
        with pytest.raises(AssertionError, match="ownership invariant"):
            am.ScratchpadAllocator._select_work_division(allocator, graph)
    else:
        changed = am.ScratchpadAllocator._select_work_division(allocator, graph)
        assert changed is (failure_at != "baseline")
    assert committed == (
        [] if failure_at in ("baseline", "assertion") else [proposals[1]["stage"]]
    )
    assert stage.iteration_space_ownership is old
