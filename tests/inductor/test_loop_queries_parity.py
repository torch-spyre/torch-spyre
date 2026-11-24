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

"""``LoopProgram``'s lifetimes against today's liveness, on real compiles.

Each case of ``test_loop_scheduler_review`` is compiled on the device exactly as
that file compiles it, and at the hand-off (``passes._hand_off_loop_nodes``, where
``graph.operations`` is the real nested list) two answers are computed for every
buffer: the model's ``live_interval`` and ``uses``, and today's
``calculate_liveness`` with ``counted_loop_lifetime_overrides`` applied. They must
agree, except where :func:`classified` names a difference the design expects: the
model answers *execution*, today's liveness answers *membership* (the design's
"Three relations, kept apart").
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest

from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.loop_facts import counted_loop_path
from torch_spyre._inductor.loop_node import LoopNode, OpsView
from torch_spyre._inductor.loop_queries import LoopProgram, _runs_once
from torch_spyre._inductor.pass_utils import op_read_writes
from torch_spyre._inductor.scratchpad.utils import (
    calculate_liveness,
    counted_loop_lifetime_overrides,
)

from test_loop_queries import _Alloc, _Extern, _Op
from test_loop_scheduler_review import _CASES, _compile


def legacy_interval(
    liveness: dict[str, list[int]],
    starts: dict[str, int],
    ends: dict[str, int],
    name: str,
) -> tuple[int, int] | None:
    """Today's lifetime of ``name`` as inclusive slots, comparable with
    ``LoopProgram.live_interval``; ``None`` when ``name`` has no use.

    ``liveness`` is ``calculate_liveness(graph)``; ``starts`` and ``ends`` are
    ``counted_loop_lifetime_overrides(graph)``, whose ends are exclusive. Without an
    override the interval is the uses' range."""
    uses = liveness.get(name)
    if not uses:
        return None
    start = starts.get(name, uses[0])
    end = ends.get(name, uses[-1] + 1) - 1
    return (start, end)


def _placement(program: LoopProgram, op) -> tuple:
    """The loops whose ``LoopNode`` bodies hold ``op``, outermost first; ``()`` at
    the top level. Read from the program's own walk (``LoopProgram`` keeps the
    placement it records per leaf), so it holds on any tree, hand-built or not."""
    return program._leaf(op).placement


def classified(program: LoopProgram, name: str) -> str | None:
    """Why ``program``'s answer for ``name`` may differ from today's liveness, or
    ``None`` when the two must agree.

    The model answers execution; ``calculate_liveness`` and
    ``counted_loop_lifetime_overrides`` place every op at its program position and
    take its loops from membership (``counted_loop_path``). The two differ exactly
    for an op whose execution is not its membership, so a buffer that such an op
    reads or writes is classified by the first such op, in program order:

    * ``"extern-in-body"``: an extern kernel placed in a ``LoopNode`` body. The loop
      scheduler runs it once, before its outermost loop, so the model times it
      there; today's liveness times it at its position in the body.
    * ``"non-member-in-body"``: an op placed in a body whose membership differs from
      its placement (an LX clone the allocator puts beside a body consumer, an
      allocation for a pad target). It runs every trip where it is placed, so the
      model counts it in its loops; today's liveness treats it as outside them.

    Stage A3 uses this to tell an expected difference from one it must classify."""
    for op in program.ops():
        placement = _placement(program, op)
        if not placement:
            continue
        rw = op_read_writes(op)
        if name not in {d.name for d in rw.reads | rw.writes}:
            continue
        if _runs_once(op):
            return "extern-in-body"
        if counted_loop_path(op) != tuple(loop.node.group_id for loop in placement):
            return "non-member-in-body"
    return None


def _answers(graph) -> dict:
    """Both answers for every buffer of ``graph``, at this point of the pipeline."""
    flat = SimpleNamespace(
        operations=OpsView(graph.operations),
        graph_input_names=graph.graph_input_names,
    )
    liveness = calculate_liveness(flat)
    starts, ends = counted_loop_lifetime_overrides(flat)
    program = LoopProgram(graph)
    names = set(liveness) | set(program.input_names())
    for op in program.ops():
        rw = op_read_writes(op)
        names |= {d.name for d in rw.reads | rw.writes}
    out = {}
    for name in sorted(names):
        out[name] = {
            "model": program.live_interval(name),
            "legacy": legacy_interval(liveness, starts, ends, name),
            "model_uses": program.uses(name),
            "legacy_uses": list(liveness.get(name, ())),
            "classification": classified(program, name),
        }
    return out


_RESULTS: dict[str, list[dict]] = {}


def _at_hand_off(case: str) -> list[dict]:
    """Compile ``case`` once per session; per graph handed off, both answers for
    every buffer."""
    if case in _RESULTS:
        return _RESULTS[case]
    graphs: list[dict] = []
    real = ts_passes._hand_off_loop_nodes

    def hand_off(graph):
        graphs.append(_answers(graph))
        real(graph)

    fn, make_args, extra_patch = _CASES[case]
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(ts_passes, "_hand_off_loop_nodes", hand_off)
        )
        _compile(fn, make_args(), extra_patch)
    assert graphs, "no graph reached the hand-off"
    _RESULTS[case] = graphs
    return graphs


def _diffs_at_hand_off(case: str) -> dict:
    """``{name: (model, legacy, classification)}`` for every buffer whose live
    interval differs, over the graphs ``case`` hands off."""
    diffs = {}
    for i, graph in enumerate(_at_hand_off(case)):
        for name, a in graph.items():
            if a["model"] != a["legacy"]:
                diffs[f"{i}:{name}"] = (a["model"], a["legacy"], a["classification"])
    return diffs


@pytest.mark.parametrize("case", sorted(_CASES))
def test_live_intervals_match_todays_liveness_except_classified_differences(case):
    diffs = _diffs_at_hand_off(case)
    unexplained = {n: d for n, d in diffs.items() if d[2] is None}
    assert not unexplained, unexplained


@pytest.mark.parametrize("case", sorted(_CASES))
def test_uses_match_calculate_liveness_for_unclassified_buffers(case):
    mismatched = {
        f"{i}:{name}": (a["model_uses"], a["legacy_uses"])
        for i, graph in enumerate(_at_hand_off(case))
        for name, a in graph.items()
        if a["classification"] is None and a["model_uses"] != a["legacy_uses"]
    }
    assert not mismatched, mismatched


# The helpers on hand-built trees (CPU, stub ops): the compiled cases above hold
# no extern kernel or non-member op in a body, so these are what exercise the
# classifications.


def _body(*ops):
    return LoopNode("n0", list(ops), induction=None, trip_count=4, group_id=0)


def test_legacy_interval_applies_the_overrides_with_an_exclusive_end():
    liveness = {"x": [2, 5], "in0": []}
    assert legacy_interval(liveness, {}, {}, "x") == (2, 5)
    assert legacy_interval(liveness, {"x": 1}, {"x": 8}, "x") == (1, 7)
    assert legacy_interval(liveness, {}, {}, "in0") is None
    assert legacy_interval(liveness, {}, {}, "nope") is None


def test_classified_names_an_extern_kernel_in_a_body():
    member = _Op("m", gid=(0,), counts=[4], reads=["e"])
    program = LoopProgram([_body(_Extern("e"), member)])
    assert classified(program, "e") == "extern-in-body"
    assert classified(program, "m") is None


def test_classified_names_a_non_member_op_in_a_body():
    member = _Op("m", gid=(0,), counts=[4], reads=["x"])
    clone = _Op("clone", reads=["x"])  # no loop_member: placed, not a member
    program = LoopProgram([_Op("x"), _body(member, clone)])
    assert classified(program, "x") == "non-member-in-body"
    assert classified(program, "clone") == "non-member-in-body"
    assert classified(program, "m") is None


def test_classified_names_an_allocation_in_a_body_a_non_member():
    # It runs every trip where it is placed; today's liveness, by membership,
    # puts it outside the loop.
    reader = _Op("r", gid=(0,), counts=[4], reads=["buf"])
    program = LoopProgram([_body(_Alloc("buf"), reader)])
    assert classified(program, "buf") == "non-member-in-body"
