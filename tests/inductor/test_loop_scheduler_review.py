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

"""The LoopNode flow-through on the cases the equivalence gate leaves open.

``test_loop_scheduler_equivalence.py`` compares the flow-through (each
``LoopNode`` reaches the scheduler as one ``LoopSchedulerNode``) with the flat
path it replaced on seven for_each_tile graphs, under the default config.
This file covers what that gate does not:

* work division left to the LX solve (#5695, ``defer_work_division``) and
  committed up front, on the same seven graphs, and divided by the greedy
  fallback when the solve fails;
* a scheduler-time reader of ``graph.operations`` (the LX local-read contract
  at kernel preparation) in a graph that also holds a loop;
* a dead op inside a live loop, under numerics;
* a constant and an extern fallback written inside a loop body, an extern op
  inside a LoopNode (hoisted), and a non-member kernel op inside one
  (refused);
* a structured loop and a flat span-overflow loop in one compile.

Each compiles on the device with the flow-through on and off.
"""

import contextlib
import copy
import unittest
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor import config, loop_scheduler
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor import work_division
from torch_spyre._inductor.ir import SpyreConstantFallback
from torch_spyre._inductor.loop_node import LoopNode, all_ops
from torch_spyre._inductor.scheduler import CountedLoopSchedulerNode
from torch_spyre._inductor.scratchpad import allocator as allocator_module
from torch_spyre._inductor.scratchpad import lx_relayout
from torch_spyre._inductor.wsr import for_each_tile

from for_each_tile_fixtures import K, M, N, split_k_fn
from test_loop_node_equivalence import _CASES, _mask
from test_loop_scheduler_equivalence import (
    _HBM_POOL,
    _compile,
    _pools,
    _settle_once_runs,
)
from utils_inductor import cached_xavier, dl16_round


def _compile_both(fn, args, extra_patch=None):
    return (
        _compile(fn, args, extra_patch, flow_through=False),
        _compile(fn, args, extra_patch, flow_through=True),
    )


def _assert_same_compile(flat, nodes):
    """What the gate compares: loop tree, splits, kernels (HBM-pool offsets
    masked after checking per-kernel pool size), numerics bit for bit."""
    assert nodes["scheduler_loops"], "no LoopSchedulerNode reached the scheduler"
    assert flat["scheduler_loops"] == []
    assert _settle_once_runs(nodes["pre_fusion"]) == _settle_once_runs(
        flat["pre_fusion"]
    )
    assert nodes["splits"] == flat["splits"]
    assert _pools(nodes["code"]) == _pools(flat["code"])
    assert _HBM_POOL.sub(r"\g<1>N", _mask(nodes["code"])) == _HBM_POOL.sub(
        r"\g<1>N", _mask(flat["code"])
    )
    torch.testing.assert_close(nodes["out"], flat["out"], atol=0, rtol=0)


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
            op.get_name()
            for op in all_ops(graph.operations)
            if hasattr(op, "loop_info")
        }

    with mock.patch.object(ts_passes, "defer_work_division", defer):
        yield


@pytest.mark.parametrize("solve_decides", [True, False], ids=["deferred", "upfront"])
@pytest.mark.parametrize("case", list(_CASES))
def test_work_division_with_and_without_deferral(case, solve_decides):
    """``_distribute_work`` runs inside the pass loop against the flat view,
    so the loop body's ops are recorded for the solve (or divided up front)
    whether or not the nodes later reach the scheduler; the splits the
    scheduler is handed must be the same either way."""
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

    flat, nodes = _compile_both(fn, make_args(), patch)
    if solve_decides:
        # The deferral branch was taken and saw the body ops: some body op
        # is in the deferred set (a loop body always holds a kernel).
        assert record["deferred"] & record["body_ops"], record
    else:
        assert "deferred" not in record
    _assert_same_compile(flat, nodes)


@pytest.mark.parametrize("case", ["split_k", "nested_online_softmax"])
def test_deferred_division_when_the_solve_fails(case):
    """On these graphs the solve commits every deferred op, so
    ``divide_deferred_ops`` divides nothing.  When the solve fails, the
    greedy fallback calls it with every op still deferred, loop bodies
    included: each body op must be divided, as when the pass runs up
    front, with the flow-through on and off."""
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

    flat, nodes = _compile_both(fn, make_args(), patch)
    # The fallback really divided body ops (everything was still deferred).
    assert left and all(any("while_loop_body" in n for n in names) for names in left)
    _assert_same_compile(flat, nodes)
    # And each op got the split the pass gives it when it runs up front.
    # (The greedy fallback may add ops of its own, so compare the ops the two
    # compiles share; every body op is among them.)
    reference = _compile(fn, make_args(), upfront, flow_through=True)
    shared = nodes["splits"].keys() & reference["splits"].keys()
    assert {n for n in nodes["splits"] if "while_loop_body" in n} <= shared
    assert {n: nodes["splits"][n] for n in shared} == {
        n: reference["splits"][n] for n in shared
    }


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
    entry" (``push_allocation_with_clone``) by inserting it at the flat
    position of the loop's first member; through ``OpsView`` that is inside
    the LoopNode's body, and the clone carries no ``loop_info``.  The flat
    path rebuilt loops from ``loop_info`` and ran it once, before the loop;
    the flow-through ran it every trip (and, before the member check, did so
    silently).  ``peel_non_members`` at the hand-off moves it out.  The graph
    of ``test_for_each_tile_lowering.test_one_unhinted_scan_does_not_block_the_other``
    (two sibling paged-gather scans)."""
    from for_each_tile_fixtures import (
        paged_gather_kv_inputs,
        paged_gather_kv_one_hinted_fn,
    )

    # The co-optimizing solve's splits on this graph vary between compiles of
    # the SAME pipeline (measured: 2 of 6 flat-path compiles differ, with
    # deterministic algorithms on), so compare under placement-only LX
    # planning, which still hoists the clones.
    seen: dict = {}
    real = ts_passes._hand_off_loop_nodes

    def hand_off(graph):
        seen["edge_non_members"] = [
            op.get_name()
            for node in graph.operations
            if isinstance(node, LoopNode) and node.body
            for op in (node.body[0], node.body[-1])
            if isinstance(op, ComputedBuffer) and getattr(op, "loop_info", None) is None
        ]
        real(graph)

    @contextlib.contextmanager
    def patch():
        with (
            config.patch({"co_optimizing_lx_planning": False}),
            mock.patch.object(ts_passes, "_hand_off_loop_nodes", hand_off),
        ):
            yield

    flat, nodes = _compile_both(
        paged_gather_kv_one_hinted_fn, paged_gather_kv_inputs(), patch
    )
    # The clones really were at the edge of a body before the peel.
    assert seen["edge_non_members"], "no hoisted clone landed in a LoopNode"
    _assert_same_compile(flat, nodes)
    # The clones are once-run kernels before the loop on both paths.
    assert nodes["pre_fusion"] == flat["pre_fusion"]


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
        buf.loop_info = copy.deepcopy(src.loop_info)
        V.graph.name_to_buffer[name] = buf
        V.graph.name_to_op[name] = buf
        loop.body.insert(loop.body.index(src) + 1, buf)
        record["name"] = name

    return dce


def test_dead_op_in_a_live_loop_is_kept_and_the_answer_is_right():
    """The 5.1 rule: a loop that survives keeps every body buffer (dead
    compute, not wrong code).  The flat path drops the dead op instead; both
    answers must match each other and the CPU reference."""
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

    flat, nodes = _compile_both(split_k_fn, (X, Y), patch)
    name = _mask(record["name"])
    leaves = [
        leaf for e in nodes["pre_fusion"] if isinstance(e[1], list) for leaf in e[1]
    ]
    # Kept inside the loop on the flow-through; removed on the flat path.
    assert ("kernel", name) in leaves, nodes["pre_fusion"]
    assert record["name"] in nodes["code"]
    assert record["name"] not in flat["code"]
    torch.testing.assert_close(nodes["out"], flat["out"], atol=0, rtol=0)
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
    """A fallback (aten.cumsum) in the body that reads the body's matmul."""

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
    runs once, outside every loop, on both paths, and the kernels match."""
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

    flat, nodes = _compile_both(_scaled_split_k_fn, (X, Y), patch)
    assert seen["body_externs"] == [], seen
    # The constant written in the body is a top-level op.
    assert any("while_loop_body" in name for name in seen["top_constants"]), seen
    _assert_same_compile(flat, nodes)
    torch.testing.assert_close(
        nodes["out"].float(), 0.5 * _split_k_reference(X, Y), atol=1e-2, rtol=1e-2
    )


def test_extern_inside_a_loop_node_is_hoisted_on_a_real_compile():
    """With a constant inside the LoopNode at the hand-off,
    ``create_scheduler_node`` makes it an extern scheduler node and places it
    immediately before its loop, so it runs once.  The flat path (expand,
    then ``build_loop_scheduler_nodes``) leaves it out of the loop too:
    ``_loop_group_id`` finds no SchedulerNode in it.  Same kernels, same
    answer.  The hoisting rule assumes a hoisted op gives the same result
    every trip; that holds here (a constant) and is not checked."""
    X, Y = _matmul_inputs()
    seen: dict = {}
    real = ts_passes._hand_off_loop_nodes

    def hand_off(graph):
        # Put the body's constant (it keeps the body's loop_info) at the head
        # of the LoopNode's body, after the pass loop and before the hand-off.
        # Production keeps it outside (see the test above; the passes'
        # re-collapse moves it out again), so this is what reaches the
        # scheduler's own hoisting rule on a real compile.
        node = next(op for op in graph.operations if isinstance(op, LoopNode))
        constant = next(
            op
            for op in graph.operations
            if isinstance(op, SpyreConstantFallback)
            and getattr(op, "loop_info", None) is not None
        )
        graph.operations.remove(constant)
        node.body.insert(0, constant)
        seen["body_externs"] = [constant.get_name()]
        real(graph)

    def patch():
        return mock.patch.object(ts_passes, "_hand_off_loop_nodes", hand_off)

    flat, nodes = _compile_both(_scaled_split_k_fn, (X, Y), patch)
    assert seen["body_externs"], "no extern op was inside a LoopNode"
    shape = nodes["pre_fusion"]
    loop_at = next(i for i, e in enumerate(shape) if isinstance(e[1], list))
    # Immediately before its loop, and nothing but kernels inside it.
    assert shape[loop_at - 1][0] == "once", shape
    assert all(e[0] == "kernel" for e in shape[loop_at][1]), shape
    _assert_same_compile(flat, nodes)
    torch.testing.assert_close(
        nodes["out"].float(), 0.5 * _split_k_reference(X, Y), atol=1e-2, rtol=1e-2
    )


def test_non_member_kernel_op_inside_a_loop_node_is_refused():
    """``collapse`` keeps an op that sits between two members of a loop in
    the node's span.  A kernel op there that is not a member (no loop_info)
    would run every trip on the flow-through and once on the flat path: here
    the accumulator's zero-fill, re-run each trip, so the flow-through
    returned only the last tile's product (max error 0.146 against the
    reference, 0.0007 on the flat path).  It is refused instead."""
    from torch_spyre._inductor import loop_node
    from torch_spyre._inductor.loop_facts import counted_loop_path

    X, Y = _matmul_inputs()

    def patch():
        # Count an extern op with loop_info as a member: the body's constant,
        # which precedes the zero-fill, then opens the span early.
        return mock.patch.object(
            loop_node,
            "_group_path",
            lambda op: counted_loop_path(op, include_extern=True),
        )

    torch._dynamo.reset()
    with (
        patch(),
        torch._inductor.config.patch(force_disable_caches=True),
        pytest.raises(Exception, match="not all members of it") as info,
    ):
        torch.compile(_scaled_split_k_fn, fullgraph=True, dynamic=False)(
            X.to("spyre"), Y.to("spyre")
        )
    assert "are in no counted loop" in str(info.value)


def test_fallback_reading_a_body_buffer_is_refused():
    """An extern op in a body is hoisted before its loop; one that reads a
    buffer the loop writes cannot be, and is refused naming the op.  (The
    flat path fails on the same graph too, later and obscurely: a cycle in
    Inductor's memory planning.)"""
    X, Y = _matmul_inputs()
    torch._dynamo.reset()
    with (
        torch._inductor.config.patch(force_disable_caches=True),
        pytest.raises(Exception, match="cannot be placed before the loop") as info,
    ):
        torch.compile(_cumsum_split_k_fn, fullgraph=True, dynamic=False)(
            X.to("spyre"), Y.to("spyre")
        )
    # Inductor wraps the backend's Unsupported in an InductorError.
    assert "Unsupported" in str(info.value)
    assert "FallbackKernel" in str(info.value)


# ---------------------------------------------------------------------------
# 4. A structured loop and a flat span-overflow loop in one compile
# ---------------------------------------------------------------------------


def test_structured_and_span_overflow_loops_in_one_graph():
    """span-overflow stamps flat loop ops (stage 1b is held), which
    ``build_loop_scheduler_nodes`` still groups.  Its ``group_idx_offset``
    is computed inside the pass loop, against the flat view, so it sees the
    structured loop's body ops and starts above their group id: the flat
    loop is group 1, the LoopSchedulerNode group 0, and both reach the
    kernel as separate counted loops."""
    torch.manual_seed(0)
    X, Y = _matmul_inputs()
    big = torch.randn(1, 8195, 256, 64, dtype=torch.float16)

    def fn(X, Y, big):
        return split_k_fn(X, Y), big + 1.0

    def patch():
        return config.patch({"sencores": 4, "ignore_span_overflow_hints": False})

    seen: dict = {}

    def run(flow_through):
        real_build = ts_passes.build_loop_scheduler_nodes

        def build(nodes):
            result = real_build(nodes)
            seen[flow_through] = [
                (
                    type(n).__name__,
                    sorted(
                        {
                            leaf.node.loop_info.loop_group_id
                            for leaf in n.get_nodes()
                            if getattr(leaf.node, "loop_info", None) is not None
                        }
                    ),
                )
                for n in result
                if isinstance(n, CountedLoopSchedulerNode)
            ]
            return result

        real_init = ts_passes.CustomPreFusionPasses.__init__

        def init(self):
            real_init(self)
            self.passes = [build if p is real_build else p for p in self.passes]

        torch._dynamo.reset()
        with (
            mock.patch.object(ts_passes.CustomPreFusionPasses, "__init__", init),
            mock.patch.object(loop_scheduler, "SCHEDULE_LOOP_NODES", flow_through),
            torch._inductor.config.patch(force_disable_caches=True),
            patch(),
        ):
            (acc, out), code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False),
                X.to("spyre"),
                Y.to("spyre"),
                big.to("spyre"),
            )
        return acc.cpu(), out.cpu(), "\n".join(code)

    acc_flat, out_flat, code_flat = run(False)
    acc, out, code = run(True)
    assert seen[True] == [
        ("LoopSchedulerNode", [(0,)]),
        ("CountedLoopSchedulerNode", [(1,)]),
    ], seen[True]
    assert seen[False] == [
        ("CountedLoopSchedulerNode", [(0,)]),
        ("CountedLoopSchedulerNode", [(1,)]),
    ], seen[False]
    assert _HBM_POOL.sub(r"\g<1>N", _mask(code)) == _HBM_POOL.sub(
        r"\g<1>N", _mask(code_flat)
    )
    torch.testing.assert_close(acc, acc_flat, atol=0, rtol=0)
    torch.testing.assert_close(out, out_flat, atol=0, rtol=0)
    torch.testing.assert_close(
        acc.float(), _split_k_reference(X, Y), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(out.float(), big.float() + 1.0, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
