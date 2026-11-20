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

"""Handing each LoopNode to the scheduler as one node changes no kernel.

Each graph is compiled twice: with the flow-through (the default: the
pre-scheduling pipeline leaves its ``LoopNode``s in ``graph.operations`` and
the scheduler makes ONE ``LoopSchedulerNode`` of each) and with
``loop_scheduler.SCHEDULE_LOOP_NODES`` off, the path it replaced (the nodes are
expanded after the pass loop and ``build_loop_scheduler_nodes`` rebuilds the
loops from the flat ops' ``loop_info``).

The IR at the end of pre-scheduling is not comparable (one list holds nodes),
so what is compared is what the IR is for:

* the generated code, with what differs between any two compiles of one graph
  masked (LX offsets, the AOT compile counter, the ``id()`` in a carry's
  scratch-buffer name), exactly as ``test_loop_node_equivalence.py`` masks it;
* the numerics, bit for bit;
* the scheduler's loop structure after the pre-fusion passes: the same tree
  of counted loops (trip counts and nesting) over the same leaf ops in the
  same order;
* the work division handed to the scheduler (``op_it_space_splits``) on every
  op, which is how a body op silently skipped by the finalizer would show.

The test also checks that a ``LoopSchedulerNode`` really reached the
scheduler on the flow-through compile and none on the other, so it cannot
pass by never taking the new path.
"""

import contextlib
import unittest
from unittest import mock

import pytest
import regex as re
import torch
from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import loop_scheduler
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.loop_node import all_ops
from torch_spyre._inductor.loop_scheduler import LoopSchedulerNode
from torch_spyre._inductor.scheduler import CountedLoopSchedulerNode

from for_each_tile_fixtures import (
    D,
    LK,
    LQ,
    online_softmax_fn,
    online_softmax_reference,
)
from test_loop_node_equivalence import (
    _CASES,
    _deterministic_algorithms,
    _mask,
)
from utils_inductor import cached_xavier, dl16_round


def _shape(nodes) -> list:
    """The loop tree of a scheduler-node list: a leaf is ``(kind, name)``
    (kind ``"kernel"`` for a SchedulerNode, else ``"once"``), a counted loop
    is ``(trip_count, [children])``, and any other group (a fusion bundle) is
    spliced in place.  The loop node's own name is
    left out: the flat path names a loop after its first member, the
    flow-through after the ``LoopNode``."""
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


def _settle_once_runs(shape) -> list:
    """``shape`` with each maximal run of adjacent ``"once"`` leaves sorted.

    A no-op or extern node in a loop body runs once, before the loop.  The
    flow-through hoists them in body order (outer level first); the flat path
    leaves them where ``_regroup_by_outer_loop_key``'s DFS put them, which on
    a nest is innermost first.  They are allocations or host ops that read
    nothing the loop writes (``loop_scheduler`` refuses one that does), so
    their order among themselves does not reach a kernel.  Kernel leaves and
    loops keep their exact order."""
    out: list = []
    run: list = []
    for entry in shape:
        if entry[0] == "once":
            run.append(entry)
            continue
        out.extend(sorted(run))
        run = []
        if not _is_loop(entry):
            out.append(entry)
        else:
            out.append((entry[0], _settle_once_runs(entry[1])))
    return out + sorted(run)


_HBM_POOL = re.compile(r"('hbm_pool':\s*)\d+")
_POOL_SIZE = re.compile(r"pool_size=(\d+)")


def _pools(code: str) -> list:
    """Per kernel, in order: (pool size, number of HBM-pool tensor args)."""
    kernels = re.split(r"async_compile\.\w+\(", code)[1:]
    return [(_POOL_SIZE.findall(k), len(_HBM_POOL.findall(k))) for k in kernels]


class _Recording(ts_passes.CustomPreSchedulingPasses):
    """Records the work-division transport the scheduler is handed."""

    splits: dict = {}

    def __call__(self, graph) -> None:
        super().__call__(graph)
        type(self).splits = {
            _mask(op.get_name()): repr(getattr(op, "op_it_space_splits", None))
            for op in all_ops(graph.operations)
        }


def _loops_in(nodes) -> list:
    """Names of the LoopSchedulerNodes in ``nodes``, nested ones included."""
    out = []
    for node in nodes:
        if isinstance(node, LoopSchedulerNode):
            out.append(node.get_name())
        if isinstance(node, FusedSchedulerNode):
            out.extend(_loops_in(node.get_nodes()))
    return out


def _is_loop(entry) -> bool:
    return isinstance(entry[1], list)


def _depth(shape) -> int:
    return max((1 + _depth(e[1]) for e in shape if _is_loop(e)), default=0)


def _count(shape) -> int:
    return sum(1 + _count(e[1]) for e in shape if _is_loop(e))


@contextlib.contextmanager
def _patch_pre_fusion(real, replacement):
    """Swap ``real`` for ``replacement`` in every CustomPreFusionPasses built
    while the block runs (enable_spyre_context builds one per compile, and it
    holds the function object, so the module attribute is not enough)."""
    original_init = ts_passes.CustomPreFusionPasses.__init__

    def init(self):
        original_init(self)
        self.passes = [replacement if p is real else p for p in self.passes]

    with mock.patch.object(ts_passes.CustomPreFusionPasses, "__init__", init):
        yield


def _compile(fn, args, extra_patch, *, flow_through: bool):
    seen: dict = {"scheduler_nodes": None, "pre_fusion": None}
    real_build = ts_passes.build_loop_scheduler_nodes

    def build(nodes):
        # ``nodes`` is Scheduler.nodes as Scheduler._init's front half left
        # it: what create_scheduler_node made of graph.operations.
        seen["scheduler_nodes"] = _loops_in(nodes)
        result = real_build(nodes)
        seen["pre_fusion"] = _shape(result)
        return result

    _Recording.splits = {}
    torch._dynamo.reset()
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(ts_passes, "CustomPreSchedulingPasses", _Recording)
        )
        stack.enter_context(
            mock.patch.object(loop_scheduler, "SCHEDULE_LOOP_NODES", flow_through)
        )
        stack.enter_context(_patch_pre_fusion(real_build, build))
        stack.enter_context(torch._inductor.config.patch(force_disable_caches=True))
        # See test_loop_node_equivalence._compile: CP-SAT is time-bounded and
        # multi-worker unless deterministic algorithms are on.
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
    }


@pytest.mark.parametrize("case", list(_CASES))
def test_flow_through_changes_no_kernel(case):
    fn, make_args, extra_patch = _CASES[case]
    args = make_args()
    flat = _compile(fn, args, extra_patch, flow_through=False)
    nodes = _compile(fn, args, extra_patch, flow_through=True)

    # The new path was taken: LoopSchedulerNodes reached Scheduler.nodes...
    assert nodes["scheduler_loops"], "no LoopSchedulerNode reached the scheduler"
    # ...and none on the path it replaced.
    assert flat["scheduler_loops"] == []

    # Same loop tree over the same kernel leaves in the same order (and the
    # same once-run nodes between them; see _settle_once_runs).
    assert _settle_once_runs(nodes["pre_fusion"]) == _settle_once_runs(
        flat["pre_fusion"]
    )
    # Same work division handed to the scheduler, op by op (a body op the
    # finalizer skipped would keep op_it_space_splits unset).
    assert nodes["splits"] == flat["splits"]
    assert any(v != "None" for v in nodes["splits"].values())
    # Same kernels, except where HBM-pool planning places each buffer.  On a
    # bundle that is one bare loop, hbm_pool_planning takes the loop as ONE
    # timestep and reads its read_writes: the flat CountedLoopSchedulerNode
    # keeps internal reads at another index (merge_list subtracts equal deps
    # only), a LoopSchedulerNode keeps external reads only.  So a body buffer's
    # live range ends at the loop's step on the flat path and at "after the
    # bundle" here.  With one timestep nothing is freed either way, so every
    # buffer gets its own region of the same pool; only the allocation order
    # (sorted by live range) and so the offsets differ.  Asserted per kernel:
    # the same pool size and the same number of pooled tensor args; then the
    # offsets are masked.
    assert _pools(nodes["code"]) == _pools(flat["code"])
    assert _HBM_POOL.sub(r"\g<1>N", _mask(nodes["code"])) == _HBM_POOL.sub(
        r"\g<1>N", _mask(flat["code"])
    )
    # Same answer, bit for bit.
    torch.testing.assert_close(nodes["out"], flat["out"], atol=0, rtol=0)

    shape = nodes["pre_fusion"]
    exact = nodes["pre_fusion"] == flat["pre_fusion"]
    pools = _mask(nodes["code"]) == _mask(flat["code"])
    print(
        f"\nCOVERAGE {case}: {len(nodes['scheduler_loops'])} LoopSchedulerNode(s) "
        f"reached the scheduler; {_count(shape)} counted loop(s), depth "
        f"{_depth(shape)}; once-run order identical: {exact}; HBM-pool offsets "
        f"identical: {pools}"
    )


def test_online_softmax_carry_numerics_against_cpu():
    """The self-dependency case under numerics: online softmax's ``m`` carry is
    written by one child (``m_new``) and read again by a LATER child of the
    same trip (``correction = exp(m - m_new)`` reads the OLD ``m``).  Checked
    against the fp32 CPU reference on dl16-rounded inputs, with the
    for_each_tile e2e tolerance."""
    Q = cached_xavier((LQ, D))
    K = cached_xavier((LK, D), differentiation=1)
    V = cached_xavier((LK, D), differentiation=2)
    res = _compile(online_softmax_fn, (Q, K, V), None, flow_through=True)
    assert res["scheduler_loops"]
    ref = online_softmax_reference(
        dl16_round(Q.float()), dl16_round(K.float()), dl16_round(V.float())
    )
    torch.testing.assert_close(res["out"].float(), ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
