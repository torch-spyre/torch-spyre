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

"""The LoopNode flow-through (each ``LoopNode`` reaches the scheduler as one
``LoopSchedulerNode``) on the graphs and cases the unit tests cannot reach.

Each test compiles on the device and checks the structure the scheduler was
handed (loops, once-run nodes, work division) and the answer against an fp32 CPU
reference:

* work division left to the LX solve (#5695, ``defer_work_division``) and
  committed up front, on seven for_each_tile graphs, and divided by the greedy
  fallback when the solve fails;
* a scheduler-time reader of ``graph.operations`` (the LX local-read contract
  at kernel preparation) in a graph that also holds a loop;
* a dead op inside a live loop, under numerics;
* a constant and an extern fallback written inside a loop body, an extern op
  inside a LoopNode (hoisted), and a non-member kernel op inside one
  (refused);
* a structured loop and a span-overflow loop in one compile.
"""

import contextlib
import dataclasses
import sys
import unittest

import regex as re
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import config
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor import work_division
from torch_spyre._inductor.ir import SpyreConstantFallback
from torch_spyre._inductor.loop_facts import (
    counted_loop_path,
    enclosing_loops,
    is_looped,
)
from torch_spyre._inductor.loop_node import (
    LoopNode,
    all_ops,
    insert_after,
    insert_at_start,
    record_member,
    remove_op,
)
from torch_spyre._inductor.loop_scheduler import LoopSchedulerNode
from torch_spyre._inductor.scheduler import CountedLoopSchedulerNode
from torch_spyre._inductor.scratchpad import allocator as allocator_module
from torch_spyre._inductor.scratchpad import lx_relayout
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor
from torch_spyre._inductor.wsr import for_each_tile

from for_each_tile_fixtures import (
    D,
    K,
    LK,
    LQ,
    M,
    N,
    nested_online_softmax_fn,
    nested_split_m_then_k_fn,
    nested_two_inner_loops_shared_init_fn,
    online_softmax_fn,
    online_softmax_reference,
    split_k_fn,
    triple_nested_stardep_multilevel_fn,
)
from utils_inductor import cached_xavier, dl16_round, strict_xfail

_decompositions = sys.modules["torch_spyre._inductor.decompositions"]


def _matmul_args():
    return (cached_xavier((M, K)), cached_xavier((K, N), differentiation=1))


def _nested_matmul_args():
    return (
        cached_xavier((256, 256)),
        cached_xavier((256, 64), differentiation=1),
    )


def _attention_args():
    return (
        cached_xavier((LQ, D)),
        cached_xavier((LK, D), differentiation=1),
        cached_xavier((LK, D), differentiation=2),
    )


def _sdpa_gqa_args():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 4, 32, 128, dtype=torch.float16, generator=g)
    k = torch.randn(2, 2, 256, 128, dtype=torch.float16, generator=g)
    v = torch.randn(2, 2, 256, 128, dtype=torch.float16, generator=g)
    b = torch.randn(2, 4, 32, 256, dtype=torch.float16, generator=g) * 0.01
    return (q, k, v, b)


def _sdpa_gqa(q, k, v, b):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=b, dropout_p=0.0, scale=128**-0.5, enable_gqa=True
    )


_ORIGINAL_SELECT = _decompositions._select_sdpa_tiling


def _force_every_sdpa_loop(**kwargs):
    # As test_sdpa_tiling.test_complete_gqa_tile_nest: B/Hkv/G/Lq maps around
    # an Lk scan, five loops.
    return dataclasses.replace(
        _ORIGINAL_SELECT(**kwargs),
        strategy="work_divided_tiled",
        kv_block_size=128,
        num_kv_blocks=2,
        num_q_tiles=2,
        q_tile_size=16,
        num_batch_tiles=2,
        num_head_tiles=2,
        num_group_tiles=2,
        kv_blocks_per_loop_group=2,
    )


# name -> (fn, args factory, extra patch factory or None)
_CASES = {
    "split_k": (split_k_fn, _matmul_args, None),
    "online_softmax": (online_softmax_fn, _attention_args, None),
    "nested_online_softmax": (nested_online_softmax_fn, _attention_args, None),
    "nested_split_m_then_k": (nested_split_m_then_k_fn, _nested_matmul_args, None),
    "nested_two_inner_loops": (
        nested_two_inner_loops_shared_init_fn,
        _matmul_args,
        None,
    ),
    "triple_nested_stardep_multilevel": (
        triple_nested_stardep_multilevel_fn,
        lambda: (
            cached_xavier((2, 256, 256)),
            cached_xavier((2, 256, 64), differentiation=1),
        ),
        None,
    ),
    "sdpa_gqa_tile_nest": (
        _sdpa_gqa,
        _sdpa_gqa_args,
        lambda: mock.patch.object(
            _decompositions,
            "_select_sdpa_tiling",
            side_effect=_force_every_sdpa_loop,
        ),
    ),
}

_LX = re.compile(r"('lx':\s*)\d+")
# A carry's scratch buffer is named ``while_carry_<id(while_op)>_<i>``
# (while_loop_bridge.py), so the name differs between any two compiles.
_CARRY_ID = re.compile(r"(while_carry_)\d+(_)")


def _mask(text: str) -> str:
    """Mask what differs between two compiles of the same graph: LX allocation
    offsets and the ``id()`` in a carry's scratch-buffer name."""
    return _CARRY_ID.sub(r"\g<1>N\g<2>", _LX.sub(r"\g<1>N", text))


@contextlib.contextmanager
def _deterministic_algorithms():
    """``torch.use_deterministic_algorithms(True, warn_only=True)``, restored.

    The co-optimizing LX planner's CP-SAT solve is time-bounded and runs several
    search workers, so on a larger graph two compiles of one graph can get
    different work splits.  With deterministic algorithms on it uses a single
    worker (fixed seed), so a compile here is reproducible."""
    was_on = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=was_warn_only)


def _shape(nodes) -> list:
    """The loop tree of a scheduler-node list: a leaf is ``(kind, name)``
    (kind ``"kernel"`` for a SchedulerNode, else ``"once"``), a counted loop
    is ``(trip_count, [children])``, and any other group (a fusion bundle) is
    spliced in place."""
    out: list = []
    for node in nodes:
        if isinstance(node, CountedLoopSchedulerNode):
            out.append((str(node.loop_count), _shape(node.get_nodes())))
        elif isinstance(node, FusedSchedulerNode):
            out.extend(_shape(node.get_nodes()))
        else:
            # A no-op becomes a NopKernelSchedulerNode, an extern op an
            # ExternKernelSchedulerNode: neither is a SchedulerNode.
            kind = "kernel" if isinstance(node, SchedulerNode) else "once"
            out.append((kind, _mask(node.get_name())))
    return out


def _is_loop(entry) -> bool:
    return isinstance(entry[1], list)


def _loops_in(nodes) -> list:
    """Names of the LoopSchedulerNodes in ``nodes``, nested ones included."""
    out = []
    for node in nodes:
        if isinstance(node, LoopSchedulerNode):
            out.append(node.get_name())
        if isinstance(node, FusedSchedulerNode):
            out.extend(_loops_in(node.get_nodes()))
    return out


class _Recording(ts_passes.CustomPreSchedulingPasses):
    """Records the work-division transport the scheduler is handed."""

    splits: dict = {}
    looped: set = set()

    def __call__(self, graph) -> None:
        super().__call__(graph)
        type(self).splits = {
            _mask(op.get_name()): repr(getattr(op, "op_it_space_splits", None))
            for op in all_ops(graph.operations)
        }
        type(self).looped = {
            _mask(op.get_name()) for op in all_ops(graph.operations) if is_looped(op)
        }


@contextlib.contextmanager
def _record_scheduler_nodes(seen: dict):
    """Record the scheduler nodes as ``CustomPreFusionPasses`` is handed them
    (``scheduler_nodes``: what ``create_scheduler_node`` made of the graph) and
    as it leaves them (``pre_fusion``: the loop tree)."""
    original_init = ts_passes.CustomPreFusionPasses.__init__

    def first(nodes):
        seen["scheduler_nodes"] = _loops_in(nodes)
        return nodes

    def last(nodes):
        seen["pre_fusion"] = _shape(nodes)
        return nodes

    def init(self):
        original_init(self)
        self.passes = [first, *self.passes, last]

    with mock.patch.object(ts_passes.CustomPreFusionPasses, "__init__", init):
        yield


def _compile(fn, args, extra_patch=None):
    seen: dict = {"scheduler_nodes": None, "pre_fusion": None}
    _Recording.splits = {}
    _Recording.looped = set()
    torch._dynamo.reset()
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(ts_passes, "CustomPreSchedulingPasses", _Recording)
        )
        stack.enter_context(_record_scheduler_nodes(seen))
        stack.enter_context(torch._inductor.config.patch(force_disable_caches=True))
        stack.enter_context(_deterministic_algorithms())
        if extra_patch is not None:
            stack.enter_context(extra_patch())
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        out, code = run_and_get_code(compiled, *[a.to(DEVICE_NAME) for a in args])
    return {
        "code": "\n".join(code),
        "out": out.cpu(),
        "scheduler_loops": seen["scheduler_nodes"],
        "pre_fusion": seen["pre_fusion"],
        "splits": dict(_Recording.splits),
        "looped": set(_Recording.looped),
    }


def _assert_matches_cpu(fn, args, out):
    """``out`` against ``fn`` run on the CPU in fp32 on the dl16-rounded inputs.
    The tolerance is 2% of the reference's largest value: fp16 accumulation over
    the loop's trips measured 0.3% to 0.8% on these graphs, so a wrong or missing
    tile (a whole tile's contribution) still fails."""
    ref = fn(*[dl16_round(a.float()) for a in args])
    torch.testing.assert_close(
        out.float(), ref, atol=0.02 * ref.abs().max().item(), rtol=0
    )


def _assert_compiled_with_nodes(res):
    assert res["scheduler_loops"], "no LoopSchedulerNode reached the scheduler"
    assert any(v != "None" for v in res["splits"].values())


def _matmul_inputs():
    return cached_xavier((M, K)), cached_xavier((K, N), differentiation=1)


def _split_k_reference(X, Y):
    """fp32 product of the dl16-rounded inputs.  Compared at atol = rtol =
    1e-2, the for_each_tile e2e tolerance (``TestForEachTileE2E.ATOL``):
    four fp16 partial sums of K=64 xavier products."""
    return dl16_round(X.float()) @ dl16_round(Y.float())


# ---------------------------------------------------------------------------
# 1. Work division deferred to the LX solve (#5695) and committed up front
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _recording_deferral(record: dict):
    """Record whether ``_distribute_work`` deferred, and which ops it left
    to the solve (the set ``defer_work_division`` stores on the graph)."""
    real = ts_passes.defer_work_division

    def defer(graph):
        real(graph)
        record["deferred"] = set(work_division.deferred_work_division(graph))
        record["body_ops"] = {
            op.get_name() for op in all_ops(graph.operations) if is_looped(op)
        }

    with mock.patch.object(ts_passes, "defer_work_division", defer):
        yield


@pytest.mark.parametrize("solve_decides", [True, False], ids=["deferred", "upfront"])
@pytest.mark.parametrize("case", list(_CASES))
def test_work_division_with_and_without_deferral(case, solve_decides):
    """``_distribute_work`` runs inside the pass loop against the flat view of the
    loop body, so the body's ops are recorded for the solve (or divided up front)
    and every one of them reaches the scheduler with a split, whichever way the
    work is divided."""
    fn, make_args, extra_patch = _CASES[case]
    record: dict = {}

    @contextlib.contextmanager
    def patch():
        with contextlib.ExitStack() as stack:
            if not solve_decides:
                stack.enter_context(config.patch({"co_optimizing_lx_planning": False}))
            stack.enter_context(_recording_deferral(record))
            if extra_patch is not None:
                stack.enter_context(extra_patch())
            assert allocator_module.solve_decides_work_division() is solve_decides
            yield

    args = make_args()
    nodes = _compile(fn, args, patch)
    if solve_decides:
        # The deferral branch was taken and saw the body ops: some body op
        # is in the deferred set (a loop body always holds a kernel).
        assert record["deferred"] & record["body_ops"], record
    else:
        assert "deferred" not in record
    _assert_compiled_with_nodes(nodes)
    # The loop body was divided: if the finalizer skipped the body, no body op
    # would have ``op_it_space_splits`` set.
    body = {n for n in nodes["looped"] if "while_loop_body" in n}
    assert any(nodes["splits"].get(n, "None") != "None" for n in body), {
        n: nodes["splits"].get(n) for n in body
    }
    _assert_matches_cpu(fn, args, nodes["out"])


@pytest.mark.parametrize("case", ["split_k", "nested_online_softmax"])
def test_deferred_division_when_the_solve_fails(case, request):
    """On these graphs the solve commits every deferred op, so
    ``divide_deferred_ops`` divides nothing.  When the solve fails, the
    greedy fallback calls it with every op still deferred, loop bodies
    included: each body op must be divided, as when the pass runs up
    front."""
    fn, make_args, extra_patch = _CASES[case]
    left: list = []
    real_divide_all = allocator_module.divide_deferred_ops

    def divide_all(graph):
        left.append(set(work_division.deferred_work_division(graph)))
        real_divide_all(graph)
        # Nothing the pass recorded is left undecided.
        assert not work_division.deferred_work_division(graph)

    def fail(self, solver, graph):
        raise allocator_module.SolveError("forced by the test")

    @contextlib.contextmanager
    def patch():
        with (
            mock.patch.object(allocator_module, "divide_deferred_ops", divide_all),
            mock.patch.object(allocator_module.CoOptimizingAllocator, "_solve", fail),
        ):
            yield

    def upfront():
        return config.patch({"co_optimizing_lx_planning": False})

    args = make_args()
    nodes = _compile(fn, args, patch)
    # The fallback really divided body ops (everything was still deferred).
    assert left and all(any("while_loop_body" in n for n in names) for names in left)
    _assert_compiled_with_nodes(nodes)
    # And each op got the split the pass gives it when it runs up front.
    # (The greedy fallback may add ops of its own, so compare the ops the two
    # compiles share; every body op is among them.)
    reference = _compile(fn, args, upfront)
    shared = nodes["splits"].keys() & reference["splits"].keys()
    assert {n for n in nodes["splits"] if "while_loop_body" in n} <= shared
    assert {n: nodes["splits"][n] for n in shared} == {
        n: reference["splits"][n] for n in shared
    }
    # The answer.  Only this check carries the xfail, so a failure above is not
    # taken for the known one.
    if case == "nested_online_softmax":
        strict_xfail(
            request,
            AssertionError,
            "greedy LX placement, which the fallback uses, gives a wrong answer "
            "(62% error against the CPU reference) on this graph: #6633",
        )
    _assert_matches_cpu(fn, args, nodes["out"])


# ---------------------------------------------------------------------------
# Scheduler-time readers of graph.operations
# ---------------------------------------------------------------------------


def test_lx_local_read_contract_with_a_loop_in_the_graph(monkeypatch):
    """Kernel preparation checks a partial LX read with
    ``local_lx_emission_contract``, which walks ``graph.operations``
    (``storage_has_single_writer``, ``_is_activation_source``).  With a
    LoopNode left in that list it raised ``AttributeError: 'LoopNode' object
    has no attribute 'get_mutation_names'``; the scheduler now puts the leaf
    list back once its nodes exist.  The gate/up graph of
    ``test_solver_relayout_e2e.test_gate_up_full_parent_repartition`` next to
    a split-K loop."""
    import torch_spyre._inductor.scheduler as sched_mod
    import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
    from torch_spyre._inductor import spyre_hint

    calls = []
    real = lx_relayout.local_lx_emission_contract

    def spy(graph, name, consumer):
        calls.append(name)
        return real(graph, name, consumer)

    monkeypatch.setattr(sched_mod, "local_lx_emission_contract", spy)
    rows, inner, half = 512, 128, 3200

    def fn(x, weight, a, b):
        with spyre_hint(work_div={"R": 8, "F": 4}):
            fused = F.linear(x, weight)
        with spyre_hint(work_div={"R": 32}):
            gate, up = fused.chunk(2, dim=-1)
            product = F.silu(gate) * up
        return -product, split_k_fn(a, b)

    host = ((torch.arange(rows * inner).reshape(rows, inner) % 33) - 16).to(
        torch.float16
    ) / 8
    columns = torch.arange(2 * half) % inner
    scales = torch.where(torch.arange(2 * half) < half, 1.0, 0.5).to(torch.float16)
    weight = torch.zeros(2 * half, inner, dtype=torch.float16)
    weight[torch.arange(2 * half), columns] = scales
    for name, size in (("R", rows), ("K", inner), ("F", 2 * half)):
        _pnd.declare_tensor_dim(name, size)
    x = _pnd.name_tensor_dims(host.to("spyre"), ["R", "K"])
    w = _pnd.name_tensor_dims(weight.to("spyre"), ["F", "K"])
    a, b = _matmul_inputs()
    fused_ref = host[:, columns] * scales
    gate_ref, up_ref = fused_ref.chunk(2, dim=-1)
    expected = -(F.silu(gate_ref.float()) * up_ref.float())

    torch._dynamo.reset()
    with (
        config.patch(
            {
                "co_optimizing_lx_planning": True,
                "layout_solver": "cpsat",
                "sencores": 32,
            }
        ),
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        out, acc = torch.compile(fn, dynamic=False, fullgraph=True)(
            x, w, a.to("spyre"), b.to("spyre")
        )
    # The contract really ran (two partial reads of the fused parent).
    assert calls, "no local LX read reached kernel preparation"
    torch.testing.assert_close(out.cpu().float(), expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(
        acc.cpu().float(), _split_k_reference(a, b), atol=1e-2, rtol=1e-2
    )


def test_lx_input_clone_hoisted_before_a_loop_runs_once():
    """The LX allocator hoists an input clone "before its consumers' loop
    entry" (``push_allocation_with_clone``). It used to insert it at the flat
    position of the loop's first member, which through ``OpsView`` is inside
    the LoopNode's body and ran it every trip. It is placed before the loop's
    node, so no hoisted clone is ever at the edge of a body, and each runs once,
    before the loop. The graph
    of ``test_for_each_tile_lowering.test_one_unhinted_scan_does_not_block_the_other``
    (two sibling paged-gather scans)."""
    from for_each_tile_fixtures import (
        paged_gather_kv_inputs,
        paged_gather_kv_one_hinted_fn,
    )

    # The co-optimizing solve's splits on this graph vary between compiles
    # (measured: 2 of 6 compiles differ, with deterministic algorithms on), so
    # run under placement-only LX planning, which still hoists the clones.
    seen: dict = {}
    real = ts_passes._hand_off_loop_nodes

    def hand_off(graph):
        seen["edge_non_members"] = [
            op.get_name()
            for node in graph.operations
            if isinstance(node, LoopNode) and node.body
            for op in (node.body[0], node.body[-1])
            if isinstance(op, ComputedBuffer) and not is_looped(op)
        ]
        # The clones hoisted in this compile, each in the unbroken run of hoisted
        # clones that ends right before a node (several are hoisted to one loop).
        ops = list(graph.operations)
        hoisted = set(seen["hoisted_now"])
        placed: set[str] = set()
        for i, op in enumerate(ops):
            if isinstance(op, LoopNode):
                j = i - 1
                while j >= 0 and ops[j].get_name() in hoisted:
                    placed.add(ops[j].get_name())
                    j -= 1
        seen["hoisted_total"] += len(hoisted)
        seen["misplaced"] += sorted(hoisted - placed)
        seen["hoisted_now"] = []
        real(graph)

    place_clone = GraphEditor._place_clone
    seen["hoisted_now"] = []
    seen["hoisted_total"] = 0
    seen["misplaced"] = []
    seen["hoisted_ops"] = []

    def spy_place_clone(editor, new_com_buf, buf_name, **kwargs):
        place_clone(editor, new_com_buf, buf_name, **kwargs)
        if kwargs.get("lower_before") is not None:
            seen["hoisted_now"].append(new_com_buf.get_name())
            seen["hoisted_ops"].append(new_com_buf.get_operation_name())

    @contextlib.contextmanager
    def patch():
        with (
            config.patch({"co_optimizing_lx_planning": False}),
            mock.patch.object(ts_passes, "_hand_off_loop_nodes", hand_off),
            mock.patch.object(GraphEditor, "_place_clone", spy_place_clone),
        ):
            yield

    nodes = _compile(paged_gather_kv_one_hinted_fn, paged_gather_kv_inputs(), patch)
    # The clones are placed outside the loop; none ever sat at the edge of a body.
    assert seen["edge_non_members"] == [], seen["edge_non_members"]
    # ... and the clones the allocator hoisted (it passed ``lower_before``) sit right
    # before a node, not merely some op that happens to precede one.
    assert seen["hoisted_total"], "the allocator hoisted no input clone"
    assert seen["misplaced"] == [], seen["misplaced"]
    _assert_compiled_with_nodes(nodes)
    # The clones are kernels outside every loop, so each runs once.
    top_level = {e[1] for e in nodes["pre_fusion"] if not _is_loop(e)}
    inside = {leaf[1] for e in nodes["pre_fusion"] if _is_loop(e) for leaf in e[1]}
    for name in seen["hoisted_ops"]:
        assert _mask(name) in top_level, (name, nodes["pre_fusion"])
        assert _mask(name) not in inside, (name, nodes["pre_fusion"])


# ---------------------------------------------------------------------------
# 2. A dead op inside a live loop
# ---------------------------------------------------------------------------


def _inject_dead_body_copy(record: dict):
    """A ``deadcode_elimination`` that, after running, adds an op to the loop
    body that nothing reads: a copy of the body's Pointwise op (same loop
    membership).  An unread value written in a for_each_tile body never
    reaches the IR (FX dead-code elimination drops it before lowering), and
    Spyre's own pass would remove one; this puts one where only the
    scheduler can judge it."""
    real_dce = ts_passes.deadcode_elimination

    def dce(graph):
        real_dce(graph)
        real = getattr(graph.operations, "real", graph.operations)
        loop = next(op for op in real if isinstance(op, LoopNode))
        src = next(
            op
            for op in loop.body
            if isinstance(op, ComputedBuffer) and isinstance(op.data, Pointwise)
        )
        data = Pointwise(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=list(src.data.ranges),
        )
        name = V.graph.qualify_name("dead_body_copy")
        buf = ComputedBuffer(
            name=name,
            layout=FlexibleLayout(src.get_device(), src.get_dtype(), src.get_size()),
            data=data,
        )
        buf.origins = src.origins
        buf.operation_name = name
        V.graph.name_to_buffer[name] = buf
        V.graph.name_to_op[name] = buf
        insert_after(real, src, buf)
        record_member(
            buf,
            [(x.tiled_dims, x.tiled_reduction_dims) for x in enclosing_loops(src)],
        )
        record["name"] = name

    return dce


def test_dead_op_in_a_live_loop_is_kept_and_the_answer_is_right():
    """The 5.1 rule: a loop that survives keeps every body buffer (dead
    compute, not wrong code).  The answer must match the CPU reference."""
    X, Y = _matmul_inputs()
    record: dict = {}
    real_init = ts_passes.CustomPreSchedulingPasses.__init__
    dce = _inject_dead_body_copy(record)

    def init(self):
        real_init(self)
        self.passes = [
            dce if p is ts_passes.deadcode_elimination else p for p in self.passes
        ]

    def patch():
        return mock.patch.object(ts_passes.CustomPreSchedulingPasses, "__init__", init)

    nodes = _compile(split_k_fn, (X, Y), patch)
    name = _mask(record["name"])
    leaves = [
        leaf for e in nodes["pre_fusion"] if isinstance(e[1], list) for leaf in e[1]
    ]
    # Kept inside the loop.
    assert ("kernel", name) in leaves, nodes["pre_fusion"]
    assert record["name"] in nodes["code"]
    torch.testing.assert_close(
        nodes["out"].float(), _split_k_reference(X, Y), atol=1e-2, rtol=1e-2
    )


# ---------------------------------------------------------------------------
# 3. A constant and an extern fallback written in a loop body
# ---------------------------------------------------------------------------


def _scaled_split_k_fn(X, Y):
    """A ``torch.full`` in the body: a SpyreConstantFallback scalar in the
    loop's ops at splice time."""

    def body(acc, ops):
        x_tile, y_tile = ops
        scale = torch.full((M, N), 0.5, device=X.device, dtype=X.dtype)
        return acc + (x_tile @ y_tile) * scale, None

    final, _ = for_each_tile(
        body,
        (X, Y),
        dims=(-1, 0),
        tile_size=64,
        init=torch.zeros(M, N, device=X.device, dtype=X.dtype),
    )
    return final


def _cumsum_split_k_fn(X, Y):
    """A cumsum in the body that reads the body's matmul.  cumsum decomposes
    to device ops, so the body holds only kernels."""

    def body(acc, ops):
        x_tile, y_tile = ops
        return acc + torch.cumsum(x_tile @ y_tile, dim=0), None

    final, _ = for_each_tile(
        body,
        (X, Y),
        dims=(-1, 0),
        tile_size=64,
        init=torch.zeros(M, N, device=X.device, dtype=X.dtype),
    )
    return final


def test_constant_written_in_a_body_runs_once_before_the_loop():
    """``splice_while_loops`` already places the body's SpyreConstantFallback
    before the loop's first op, so ``collapse`` leaves it outside the
    ``LoopNode`` and the scheduler has no extern op to hoist: the constant
    runs once, outside every loop."""
    X, Y = _matmul_inputs()
    seen: dict = {}
    real = ts_passes.splice_while_loops

    def splice(graph):
        real(graph)
        seen["body_externs"] = [
            op.get_name()
            for node in graph.operations
            if isinstance(node, LoopNode)
            for op in all_ops(node.body)
            if not isinstance(op, ComputedBuffer)
        ]
        seen["top_constants"] = [
            op.get_name()
            for op in graph.operations
            if isinstance(op, SpyreConstantFallback)
        ]

    real_init = ts_passes.CustomPreSchedulingPasses.__init__

    def init(self):
        real_init(self)
        self.passes = [splice if p is real else p for p in self.passes]

    def patch():
        return mock.patch.object(ts_passes.CustomPreSchedulingPasses, "__init__", init)

    nodes = _compile(_scaled_split_k_fn, (X, Y), patch)
    assert seen["body_externs"] == [], seen
    # The constant written in the body is a top-level op.
    assert any("while_loop_body" in name for name in seen["top_constants"]), seen
    _assert_compiled_with_nodes(nodes)
    torch.testing.assert_close(
        nodes["out"].float(), 0.5 * _split_k_reference(X, Y), atol=1e-2, rtol=1e-2
    )


def test_extern_inside_a_loop_node_is_hoisted_on_a_real_compile():
    """With a constant inside the LoopNode at the hand-off,
    ``create_scheduler_node`` makes it an extern scheduler node and places it
    immediately before its loop, so it runs once.  The hoisting rule assumes a
    hoisted op gives the same result every trip; that holds here (a constant) and
    is not checked."""
    X, Y = _matmul_inputs()
    seen: dict = {}
    real = ts_passes._hand_off_loop_nodes

    def hand_off(graph):
        # Put the body's constant (dedup hoisted it out of the body) at the head
        # of the LoopNode's body, after the pass loop and before the hand-off.
        # Production keeps it outside (see the test above; the passes'
        # re-collapse moves it out again), so this is what reaches the
        # scheduler's own hoisting rule on a real compile.
        node = next(op for op in graph.operations if isinstance(op, LoopNode))
        # The body's ``torch.full((M, N), 0.5)`` (``_scaled_split_k_fn``); the graph's
        # other constants are not from the loop body.
        constant = next(
            op
            for op in graph.operations
            if isinstance(op, SpyreConstantFallback) and op.constant_args[0] == 0.5
        )
        remove_op(graph.operations, constant)
        insert_at_start(node, constant)
        seen["body_externs"] = [constant.get_name()]
        real(graph)

    def patch():
        return mock.patch.object(ts_passes, "_hand_off_loop_nodes", hand_off)

    nodes = _compile(_scaled_split_k_fn, (X, Y), patch)
    assert seen["body_externs"], "no extern op was inside a LoopNode"
    shape = nodes["pre_fusion"]
    loop_at = next(i for i, e in enumerate(shape) if isinstance(e[1], list))
    # Immediately before its loop, and nothing but kernels inside it.
    assert shape[loop_at - 1][0] == "once", shape
    assert all(e[0] == "kernel" for e in shape[loop_at][1]), shape
    _assert_compiled_with_nodes(nodes)
    torch.testing.assert_close(
        nodes["out"].float(), 0.5 * _split_k_reference(X, Y), atol=1e-2, rtol=1e-2
    )


class _Stub:
    """An op with a name and, optionally, a loop group: all ``_check_members`` reads."""

    def __init__(self, name, group=None):
        self._name = name
        if group is not None:
            self.loop_member = SimpleNamespace(loop_group_id=group)

    def get_name(self):
        return self._name


def test_non_member_kernel_op_inside_a_loop_node_is_refused():
    """``collapse`` keeps an op that sits between two members of a loop in
    the node's span.  A kernel op there that is not a member (no loop_member)
    would run every trip: the accumulator's zero-fill, re-run each trip,
    returned only the last tile's product (max error 0.146 against the
    reference, 0.0007 when it ran once).
    ``_check_members`` refuses it.

    This calls the check directly. It used to compile a graph with a patch that
    made the body's constant count as a member, so the span opened early and took
    the zero-fill in; that no longer happens, because
    ``dedup_and_promote_constants`` now hoists the constant out of the body before
    the nodes are built (the graph compiles and is right, error 0.00073), so no
    compiled graph reaches this check any more. The check guards a future pass
    that places an op between two members without stamping its ``loop_member``."""
    from torch_spyre._inductor.loop_scheduler import _check_members

    node = _Stub("loopnode0")
    member_a, member_b = _Stub("buf0", (0,)), _Stub("buf2", (0,))
    outsider = _Stub("buf1")

    # members only: nothing to refuse, in one loop or in a nest
    _check_members(node, [member_a, member_b])
    _check_members(node, [])

    # a non-member at the head or the tail is refused too (the hand-off no longer
    # moves it out of the node)
    for edge in ([outsider, member_a, member_b], [member_a, member_b, outsider]):
        with pytest.raises(Exception, match="not all members of it"):
            _check_members(node, edge)

    # a non-member between two members is refused, and named
    with pytest.raises(Exception, match="not all members of it") as info:
        _check_members(node, [member_a, outsider, member_b])
    assert "are in no counted loop" in str(info.value)
    assert "buf1" in str(info.value)

    # members of two different loops in one node are refused too
    with pytest.raises(Exception, match="their loop_group_ids differ"):
        _check_members(node, [member_a, _Stub("buf3", (1,))])


def test_cumsum_reading_a_body_buffer_runs_on_device():
    """cumsum in a body reads a buffer the loop writes.  It used to be a CPU
    fallback, an extern op that cannot be hoisted before the loop and was
    refused; it now decomposes to device kernels, so it stays in the loop
    and the answer matches the reference."""
    X, Y = _matmul_inputs()
    nodes = _compile(_cumsum_split_k_fn, (X, Y))
    _assert_compiled_with_nodes(nodes)
    # The cumsum's kernels are in the loop with the matmul (a matmul-only body
    # is two kernels), and nothing but kernels is: no extern op was left in it.
    loop = next(e[1] for e in nodes["pre_fusion"] if isinstance(e[1], list))
    assert len(loop) > 2 and all(e[0] == "kernel" for e in loop), loop
    # cumsum is linear, so summing the tiles' cumsums is the cumsum of the product.
    torch.testing.assert_close(
        nodes["out"].float(),
        torch.cumsum(_split_k_reference(X, Y), dim=0),
        atol=5e-2,
        rtol=1e-2,
    )


# ---------------------------------------------------------------------------
# 4. A structured loop and a span-overflow loop in one compile
# ---------------------------------------------------------------------------


def test_structured_and_span_overflow_loops_in_one_graph():
    """span-overflow stamps its group above the structured loop's group id (its
    ``group_idx_offset`` is computed inside the pass loop, against the flat view,
    so it sees the structured loop's body ops), and the pass loop wraps that
    group in a ``LoopNode`` too: the structured loop is group 0, the
    span-overflow loop group 1, and both reach the scheduler as
    LoopSchedulerNodes and the kernel as separate counted loops."""
    torch.manual_seed(0)
    X, Y = _matmul_inputs()
    big = torch.randn(1, 8195, 256, 64, dtype=torch.float16)

    def fn(X, Y, big):
        return split_k_fn(X, Y), big + 1.0

    seen: dict = {}
    original_init = ts_passes.CustomPreFusionPasses.__init__

    def record(nodes):
        seen["loops"] = [
            (
                type(n).__name__,
                sorted(
                    {
                        counted_loop_path(leaf.node)
                        for leaf in n.get_nodes()
                        if counted_loop_path(leaf.node)
                    }
                ),
            )
            for n in nodes
            if isinstance(n, CountedLoopSchedulerNode)
        ]
        return nodes

    def init(self):
        original_init(self)
        self.passes = [*self.passes, record]

    torch._dynamo.reset()
    with (
        mock.patch.object(ts_passes.CustomPreFusionPasses, "__init__", init),
        torch._inductor.config.patch(force_disable_caches=True),
        config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
    ):
        (acc, out), _ = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=False),
            X.to("spyre"),
            Y.to("spyre"),
            big.to("spyre"),
        )
    assert seen["loops"] == [
        ("LoopSchedulerNode", [(0,)]),
        ("LoopSchedulerNode", [(1,)]),
    ], seen["loops"]
    torch.testing.assert_close(
        acc.cpu().float(), _split_k_reference(X, Y), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        out.cpu().float(), big.float() + 1.0, atol=1e-2, rtol=1e-2
    )


def test_online_softmax_carry_numerics_against_cpu():
    """The self-dependency case under numerics: online softmax's ``m`` carry is
    written by one child (``m_new``) and read again by a LATER child of the
    same trip (``correction = exp(m - m_new)`` reads the OLD ``m``).  Checked
    against the fp32 CPU reference on dl16-rounded inputs, with the
    for_each_tile e2e tolerance."""
    Q = cached_xavier((LQ, D))
    Kt = cached_xavier((LK, D), differentiation=1)
    Vt = cached_xavier((LK, D), differentiation=2)
    res = _compile(online_softmax_fn, (Q, Kt, Vt))
    assert res["scheduler_loops"]
    ref = online_softmax_reference(
        dl16_round(Q.float()), dl16_round(Kt.float()), dl16_round(Vt.float())
    )
    torch.testing.assert_close(res["out"].float(), ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
