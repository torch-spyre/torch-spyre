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


"""Each bundle's kernel is prepared once, before HBM pooling; emission consumes it.

The stub scenario drives the preparation pass with hand-built nodes; the
real compile checks one preparation per bundle, the identical operations at
emission, and the late binding of pooled intermediates.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch as mock_patch

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

import torch_spyre._inductor.scheduler as scheduler_module
import torch_spyre._inductor.scratchpad.lx_relayout as lx_relayout_module
from torch_spyre._inductor import config
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.spyre_kernel import SpyreKernel, _iter_op_specs

_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"
_CORE_ID = sympy.Symbol("core_id")
_VIEW = PerCoreView(((0, 8),), ((0, _CORE_ID),), num_cores=8)
_OTHER_VIEW = PerCoreView(
    ((0, 4), (1, 2)),
    ((0, sympy.floor(_CORE_ID / 2)), (1, sympy.Mod(_CORE_ID, 2))),
    num_cores=8,
)


class _Node:
    """A scheduler-node stand-in: one operation that is also its own bundle."""

    def __init__(self, name, reads=(), writes=()):
        self.name = name
        self.read_writes = SimpleNamespace(
            reads=[MemoryDep(read, sympy.S.Zero, (), ()) for read in reads],
            writes=[MemoryDep(write, sympy.S.Zero, (), ()) for write in writes],
        )

    def get_nodes(self):
        return [self]

    def get_name(self):
        return self.name

    def get_device(self):
        return None

    is_template = is_extern = is_foreach = staticmethod(lambda: False)


def _layout(lx, view=_VIEW):
    # Real MemoryDeps above and a spec'd layout here let the pass's own
    # isinstance checks do the filtering, with no patched module types.
    layout = Mock(spec=FixedTiledLayout)
    layout.allocation = {"lx": 0} if lx else {"hbm": 0}
    layout.lx_view = view if lx else None
    layout.device_layout = SimpleNamespace(device_size=(8, 64))
    return layout


def _graph(layouts, scheduler=None):
    buffers = {
        name: SimpleNamespace(
            layout=SimpleNamespace(), get_layout=lambda layout=layout: layout
        )
        for name, layout in layouts.items()
    }
    backend = scheduler_module.SuperDSCScheduling(scheduler)
    graph = SimpleNamespace(
        try_get_buffer=buffers.get,
        get_buffer=buffers.__getitem__,
        removed_buffers=OrderedSet(),
        scheduler=SimpleNamespace(get_backend=lambda _device: backend, name_to_buf={}),
    )
    return graph, backend


# --- the preparation pass ---------------------------------------------------


def test_a_rejected_attempt_demotes_restarts_and_keeps_the_final_kernels():
    """One failing relayout copy: its group falls back, nothing else moves.

    The copy touches both ends of the relayout, so source and destination are
    demoted together (the first name closes the connected group through the
    registry; the second finds no plan and clears its own layout again). The
    attempt's graph removals are dropped and the next attempt starts over in
    order; an unrelated LX placement survives; every bundle keeps the kernel
    of the last, complete attempt.
    """

    plan = lx_relayout_module.LXRelayoutPlan(
        "source", ("consumer",), _VIEW, _OTHER_VIEW, 8
    )
    layouts = {
        "target": _layout(False),
        "source": _layout(True),
        "destination": _layout(True, _OTHER_VIEW),
        "z": _layout(True),
    }
    writer = _Node("writer", writes=("target",))
    copy = _Node("copy", reads=("source",), writes=("destination",))
    unrelated = _Node("unrelated", writes=("z",))
    graph, backend = _graph(layouts)
    graph._spyre_lx_relayout_copies = {("source", "consumer"): ("destination", plan)}
    order, kernels = [], {}

    def prepare(node, kernel):
        order.append(node.name)
        kernels[node.name] = kernel
        if node.name == "writer":
            # Handlers add mutation aliases to the graph's removed set. A
            # restarted attempt must start from the entry state again.
            assert "alias" not in graph.removed_buffers
            graph.removed_buffers.add("alias")
        elif node.name == "copy" and "lx" in layouts["source"].allocation:
            kernel.failed_node = node
            raise ValueError("ownership mismatch: forced")
        return kernel

    with (
        mock_patch.object(scheduler_module, "SchedulerNode", _Node),
        mock_patch.object(scheduler_module, "V", SimpleNamespace(graph=graph)),
        mock_patch.object(backend, "prepare_kernel", side_effect=prepare),
    ):
        scheduler_module.prepare_spyre_kernels([writer, copy, unrelated])

    assert order == ["writer", "copy", "writer", "copy", "unrelated"]
    assert list(graph.removed_buffers) == ["alias"]
    assert graph._spyre_lx_relayout_copies == {}
    for name in ("source", "destination"):
        assert "lx" not in layouts[name].allocation and layouts[name].lx_view is None
    assert layouts["z"].allocation == {"lx": 0} and layouts["z"].lx_view is _VIEW
    for node in (writer, copy, unrelated):
        assert node.prepared_kernel is kernels[node.name]


def test_a_pooled_mutation_destination_leaves_the_external_argument_list():
    """Emission prunes a pooled destination registered through a mutation alias.

    Preparation runs before HBM pooling, so an operation that writes a padded
    intermediate registers it as an external argument. ``KernelArgs.output``
    resolves ``mutation_real_name``, so the alias lands in
    ``store_buffer_names`` while the real destination lands in
    ``output_buffers``. Emission skips a pooled tensor's descriptor argument,
    so the same tensor must leave the call argument list, or the wrapper passes
    one argument more than the bundle declares.
    """

    pooled = _layout(False)
    pooled.allocation = {"hbm_pool": 4096}  # filled in after preparation
    layouts = {"destination": pooled, "result": _layout(False), "alias": Mock()}
    emitted = []
    graph = SimpleNamespace(
        get_buffer={
            name: SimpleNamespace(get_layout=lambda layout=layout: layout)
            for name, layout in layouts.items()
        }.get,
        removed_buffers=OrderedSet(),
        scheduler=SimpleNamespace(mutation_real_name={"alias": "destination"}),
        wrapper_code=SimpleNamespace(writeline=emitted.append),
        get_dtype=lambda name: None,
    )

    with V.set_graph_handler(graph):
        kernel = SpyreKernel()
        kernel.store_buffer_names.add("alias")
        kernel.args.output("alias")  # registered as "destination"
        kernel.args.output("result")
        kernel.args.input("dead_index")  # simplified away after registration
        destination_arg = Mock(allocation=pooled.allocation)
        result_arg = Mock(allocation={"hbm": 0})
        # The destination was registered before pooling and remains in this
        # prepared list; only emission knows it no longer needs an external arg.
        kernel.spyre_kernel_args = [
            ("destination", destination_arg),
            ("result", result_arg),
        ]
        kernel.codegen_kernel()
        assert "destination" not in kernel.args.python_argdefs()[1]
        assert kernel._live_call_arg_names == ["result"]
        assert result_arg.arg_index == 0
        kernel.call_kernel("prepared")
        assert emitted == ["prepared.run(result)"]


# --- the real compile -------------------------------------------------------


@config.patch({"lx_planning": False})
def test_emission_consumes_the_kernels_prepared_before_pooling():
    """One preparation per bundle inside the pass; emission consumes the same
    operations and binds only what HBM pooling decided afterwards."""

    prepared, emitted = {}, []
    emitted_arguments = {}
    real_prepare = scheduler_module.SuperDSCScheduling.prepare_kernel
    real_codegen = SpyreKernel.codegen_kernel

    def operation_ids(kernel):
        return [id(spec) for spec in _iter_op_specs(kernel.op_specs)]

    def prepare(backend, node, kernel):
        result = real_prepare(backend, node, kernel)
        # The scheduler gains ``removed_ops`` only after the post-fusion hook,
        # so its absence marks a preparation made by the pass.
        if not hasattr(V.graph.scheduler, "removed_ops"):
            prepared[id(kernel)] = operation_ids(kernel)
        return result

    def codegen_kernel(kernel):
        emitted.append((kernel, operation_ids(kernel)))
        result = real_codegen(kernel)
        # This is the same filtered ordering used by arg_index and .run().
        assert kernel._live_call_arg_names is not None
        emitted_arguments[id(kernel)] = kernel._live_call_arg_names.copy()
        return result

    def fn(x, y):
        a = x + y
        b = a * 2
        return b - x

    x = torch.randn(64, 64, dtype=torch.float16, device="spyre")
    y = torch.randn(64, 64, dtype=torch.float16, device="spyre")
    with (
        mock_patch.object(
            scheduler_module.SuperDSCScheduling,
            "prepare_kernel",
            side_effect=prepare,
            autospec=True,
        ),
        mock_patch.object(
            SpyreKernel, "codegen_kernel", side_effect=codegen_kernel, autospec=True
        ),
        mock_patch(_LAUNCH_JOBPLAN),
        mock_patch(_PREPARE_KERNEL),
        mock_patch("subprocess.run"),
    ):
        _, code = run_and_get_code(torch.compile(fn, dynamic=False), x, y)
    generated = "\n".join(code)

    assert emitted
    assert all(prepared.get(id(kernel)) == ids for kernel, ids in emitted)
    assert len({id(kernel) for kernel, _ in emitted}) == len(emitted)
    assert "'hbm_pool'" in generated  # intermediates were pooled after preparation
    assert any(kernel.pool_size > 0 for kernel, _ in emitted)  # bound at emission
    # Pooling ran after preparation: pooled intermediates are pruned from the
    # external call arguments at emission, and the survivors keep the indices
    # of the actual argument list.
    pooled = set()
    for kernel, _ in emitted:
        call_args = emitted_arguments[id(kernel)]
        for name, arg in kernel.spyre_kernel_args:
            if "hbm_pool" in arg.allocation:
                pooled.add(name)
                assert name not in call_args
            elif "lx" not in arg.allocation:
                assert arg.arg_index == call_args.index(name)
    assert pooled
