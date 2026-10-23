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

import math
from collections.abc import Sequence
from contextlib import contextmanager
import functools
import itertools
from types import SimpleNamespace
from typing import Callable, TypeVarTuple, Unpack, Optional, override

import unittest
from unittest.mock import patch
import torch

from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    ReinterpretView,
    StorageBox,
    TensorBox,
)

from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor import passes
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor.pass_utils import op_read_writes
from torch_spyre._inductor.patches import enable_spyre_context
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor
from torch_spyre._inductor.scratchpad.utils import calculate_liveness
from torch_spyre._inductor.scratchpad.graph_builder import GraphBuilder
from torch_spyre._inductor.scratchpad import residency_rules

try:
    from ortools.sat.python import cp_model  # noqa: F401

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False


Ts = TypeVarTuple("Ts")


@contextmanager
def _recording_division_modes():
    """The division mode of every ``ScratchpadAllocator.plan_allocation`` call,
    in order: ``["open", "committed"]`` means the open solve failed and
    ``scratchpad_planning`` replanned over the committed divisions."""
    from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator

    modes: list[str] = []
    original = ScratchpadAllocator.plan_allocation

    def recording(allocator, graph):
        modes.append(allocator.division_mode)
        return original(allocator, graph)

    with patch.object(ScratchpadAllocator, "plan_allocation", recording):
        yield modes


def _graph_editor_test_buffer(name: str) -> ComputedBuffer:
    device = torch.device("spyre")
    return ComputedBuffer(
        name=name,
        layout=FixedLayout(device, torch.float16, [2, 3]),
        data=Pointwise(
            device=device,
            dtype=torch.float16,
            inner_fn=lambda _i0, _i1: 0,
            ranges=[2, 3],
        ),
    )


def test_change_graph_output_skips_unrelated_view_and_preserves_matching_view():
    old = _graph_editor_test_buffer("old")
    new = _graph_editor_test_buffer("new")
    unrelated = _graph_editor_test_buffer("unrelated")
    view_layout = FixedLayout(
        torch.device("spyre"), torch.float16, [3, 2], [1, 3], offset=1
    )
    view = ReinterpretView(data=StorageBox(old), layout=view_layout)
    output = TensorBox(StorageBox(view))
    unrelated_view = ReinterpretView(
        data=StorageBox(unrelated), layout=unrelated.layout
    )
    unrelated_output = TensorBox(StorageBox(unrelated_view))
    lowering = SimpleNamespace(graph_outputs=[unrelated_output, output])
    editor = object.__new__(GraphEditor)
    editor.lowering = lowering

    editor.change_graph_output(old, new)

    assert lowering.graph_outputs[0] is unrelated_output
    assert unrelated_view.data.data is unrelated
    assert lowering.graph_outputs[1] is output
    assert output.data.data is view
    assert view.layout is view_layout
    assert view.data.data is new


def test_nested_spyre_context_runs_pre_scheduling_once():
    calls = []

    class CountingPreSchedulingPasses:
        def __call__(self, graph):
            calls.append(graph)

    graph = SimpleNamespace(graph=SimpleNamespace(owning_module=None))
    with (
        patch.object(passes, "CustomPreSchedulingPasses", CountingPreSchedulingPasses),
        patch.object(GraphLowering, "_update_scheduler", lambda _self: None),
        enable_spyre_context([]),
        enable_spyre_context([]),
    ):
        GraphLowering._update_scheduler(graph)

    assert calls == [graph]


class CustomPreSchedulingPassesWithOurPasses(CustomPreSchedulingPasses):
    """torch_spyre._inductor.patches.enable_spyre_context sets
    torch._inductor.config._post_fusion_custom_pass to
    torch_spyre._inductor.passes.CustomPostFusionPasses(), so we have to monkey patch that class
    to add the ability to add custom passes."""

    test_instance: Optional["BaseTestScratchpadUsage"] = None

    @classmethod
    def initialize(cls, test_instance: "BaseTestScratchpadUsage"):
        cls.test_instance = test_instance

    @override
    def __call__(self, graph: GraphLowering) -> None:
        assert self.test_instance is not None, (
            "CustomPreSchedulingPassesWithOurPasses.test_instance must be set to an instance of "
            "BaseTestScratchpadUsage before get_passes is called"
        )
        super().__call__(graph)
        for f in self.test_instance.our_pre_scheduling_passes:
            f(graph)


class BaseTestScratchpadUsage(unittest.TestCase):
    our_pre_scheduling_passes: list[Callable[[GraphLowering], None]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patchers = []

    def setUp(self):
        torch.manual_seed(0xAFFE)

        self.patchers.append(t_inductor_config.patch("force_disable_caches", True))
        self.patchers.append(
            ts_inductor_config.patch("allow_all_ops_in_lx_planning", True)
        )
        CustomPreSchedulingPassesWithOurPasses.initialize(self)
        self.patchers.append(
            patch.object(
                passes,
                "CustomPreSchedulingPasses",
                CustomPreSchedulingPassesWithOurPasses,
            )
        )

        for p in self.patchers:
            p.__enter__()

        torch.compiler.reset()

    def tearDown(self):
        for p in self.patchers:
            p.__exit__(None, None, None)

        torch.compiler.reset()

    def rand_device(self, shape: Sequence[int]):
        result = torch.rand(shape, dtype=torch.float16, device="spyre")
        return result

    @contextmanager
    def pre_scheduling_iterating_pass(
        self,
        f: Callable[[GraphLowering], None],
    ):
        """Context manager to add a post fusion custom pass that processes each node independently
        using `f`."""

        def new_pass(graph: GraphLowering) -> None:
            f(graph)

        self.our_pre_scheduling_passes.append(new_pass)
        yield
        self.our_pre_scheduling_passes.remove(new_pass)

    def compile_and_collect_mem_usage(
        self, f: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor, dict[str, str]]:
        mem_usages = {}

        def visitor(graph: GraphLowering) -> None:
            nonlocal mem_usages
            operations = graph.operations
            for op in operations:
                buf_name = op.name
                buffer = graph.get_buffer(buf_name)
                layout = buffer.get_layout()
                if isinstance(layout, MutationLayoutSHOULDREMOVE):
                    layout = layout.real_layout()
                device_layout = layout.device_layout
                allocation = getattr(layout, "allocation", {})
                mem_usages[buf_name] = {
                    "location": "LX" if "lx" in allocation else "HBM",
                    "size": math.prod(device_layout.device_size[:-1]) * 128,
                }

        with self.pre_scheduling_iterating_pass(visitor):
            compiled_kernel = torch.compile(f, fullgraph=True)
            result = compiled_kernel(*args).to("cpu")

        return (result, mem_usages)

    def measure_hbm_transfers(
        self, model: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor | None, int]:
        """Compile ``model`` and return ``(result, hbm_bytes)``, where
        ``hbm_bytes`` is the total size of all HBM-resident buffers. LX-resident
        buffers are treated as free."""
        result, mem_usages = self.compile_and_collect_mem_usage(model, args)
        hbm_transfers = sum(
            mem_usage["size"]
            for mem_usage in mem_usages.values()
            if mem_usage["location"] == "HBM"
        )
        return (result, hbm_transfers)

    def assert_uses_lx(self, mem_usages: dict[str, dict]) -> None:
        """Assert the allocator placed at least one buffer in LX."""
        self.assertTrue(
            any(mem_usage["location"] == "LX" for mem_usage in mem_usages.values()),
            "Expected at least one buffer to be allocated in LX, but none were",
        )

    def run_case(self, params: dict, factory: Callable) -> None:
        """Body for one metaclass-generated parameterized case. Overridden by
        classes using ``_ParameterizedScratchpadMeta``: ``params`` is the config
        combo (empty when the class has no ``parameter_axes``) and
        ``factory(self) -> (model, args, kwargs)``."""
        raise NotImplementedError

    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        **kwargs,
    ):
        """Run the current class's test procedure on the given model and arguments. Override this
        in each subclass."""
        cpu_result = model(*(t.to("cpu") for t in args))

        with ts_inductor_config.patch(lx_planning=True):
            device_result, mem_usages = self.compile_and_collect_mem_usage(model, args)

        self.assert_uses_lx(mem_usages)

        atol = kwargs.get("atol", 1e-4)
        rtol = kwargs.get("rtol", 1e-5)
        self.assertTrue(
            torch.allclose(cpu_result, device_result, atol=atol, rtol=rtol),
            "Results do not match",
        )

    def _simple_mlp(
        self,
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        """Two-layer linear MLP matching ``SimpleMLP`` from the provenance
        example: ``nn.Linear -> silu -> nn.Linear``.
        """
        seq_len, in_dim, hidden_dim, out_dim = 128, 256, 1024, 256
        fc1 = torch.nn.Linear(in_dim, hidden_dim).half()
        fc2 = torch.nn.Linear(hidden_dim, out_dim).half()

        def mlp(x, w1, b1, w2, b2):
            return torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1)), w2, b2
            )

        x = torch.randn(seq_len, in_dim, dtype=torch.float16).to("spyre")
        args = (
            x,
            fc1.weight.to("spyre"),
            fc1.bias.to("spyre"),
            fc2.weight.to("spyre"),
            fc2.bias.to("spyre"),
        )
        return mlp, args

    def _swiglu(
        self,
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        """A single functional SwiGLU layer."""
        seq_len, in_dim, hidden_dim = 128, 256, 1024

        fc_gate = torch.nn.Linear(in_dim, hidden_dim).half()
        fc_up = torch.nn.Linear(in_dim, hidden_dim).half()

        def swiglu(x, w_gate, b_gate, w_up, b_up):
            gate = torch.nn.functional.linear(x, w_gate, b_gate)
            up = torch.nn.functional.linear(x, w_up, b_up)
            return torch.nn.functional.silu(gate) * up

        x = torch.randn(seq_len, in_dim, dtype=torch.float16).to("spyre")
        args = (
            x,
            fc_gate.weight.to("spyre"),
            fc_gate.bias.to("spyre"),
            fc_up.weight.to("spyre"),
            fc_up.bias.to("spyre"),
        )
        return swiglu, args


class _ParameterizedScratchpadMeta(type):
    """Data-driven metaclass that expands a model list (and, optionally, a
    cartesian product of config axes) into one test *method* per case on a
    single collected class.

    A class carrying ``parameter_models`` (the ``(label, factory)`` list) gets a
    ``test_<label>`` method per model. If it also carries ``parameter_axes``
    (axis name -> values), it gets a ``test_<label>__<combo>`` method per
    ``(model, config-combo)`` instead. Each generated method delegates to the
    class's ``run_case(self, params, factory)`` — so the per-case body (apply
    the combo and check correctness, compare HBM off vs on, ...) is defined by
    the class, not baked into the metaclass.

    Generating methods rather than sibling classes keeps everything in the
    ``attrs`` dict handed to ``__new__`` — no module-namespace or ``sys.modules``
    access, so it is immune to the OOT runner's out-of-``sys.modules``
    pre-import.
    """

    # How each axis renders into the test-id suffix. Axes not listed fall back to
    # "<name><value>"; this keeps the curated short labels while letting new axes
    # added to ``parameter_axes`` work without editing this method.
    _AXIS_LABELS = {
        "tiling_mode": lambda v: str(v),
        "sencores": lambda v: f"sc{v}",
        "co_optimization": lambda v: "coopt" if v else "nocoopt",
    }

    @staticmethod
    def _combo_suffix(params: dict) -> str:
        """Readable, test-id-safe suffix for one combo. Empty -> '' (bare name)."""
        if not params:
            return ""
        labels = _ParameterizedScratchpadMeta._AXIS_LABELS
        return "_".join(
            labels[name](value) if name in labels else f"{name}{value}"
            for name, value in params.items()
        )

    def __new__(mcs, name, bases, attrs):
        models = attrs.get("parameter_models")
        if models:
            axes = attrs.get("parameter_axes") or {}
            axis_names = list(axes)
            if axis_names:
                combos = [
                    dict(zip(axis_names, c))
                    for c in itertools.product(*(axes[a] for a in axis_names))
                ]
            else:
                combos = [{}]
            for params in combos:
                suffix = mcs._combo_suffix(params)
                for label, factory in models:
                    test_name = f"test_{label}__{suffix}" if suffix else f"test_{label}"
                    test_method = mcs._make_case(params, factory)
                    attrs[test_name] = test_method
        return super().__new__(mcs, name, bases, attrs)

    @staticmethod
    def _make_case(params: dict, factory: Callable):
        """Build one isolated test method bound to ``params``/``factory`` via
        arguments (not loop-variable closure), so each method keeps its own
        combo. The body is the class's ``run_case``."""

        def test(self):
            self.run_case(params, factory)

        test.__doc__ = (
            f"Parameterized case under {params}." if params else ("Parameterized case.")
        )
        return test


class ParameterizedScratchpadUsage(
    BaseTestScratchpadUsage, metaclass=_ParameterizedScratchpadMeta
):
    """Full cartesian product of the scratchpad-planning configuration knobs.

    Replaces the hand-written solver-variant classes: the metaclass injects a
    ``test_<model>__sc<n>_<coopt>_<clones>`` method for every model in
    ``parameter_models`` and every point in ``parameter_axes``. Edit
    ``parameter_axes`` to widen or narrow the sweep.
    """

    # Models swept by the parameterized suites, as ``(label, factory)`` where
    # ``factory(self) -> (model, args, kwargs)``. ``kwargs`` are forwarded to the
    # per-case body (e.g. relaxed tolerances for fp16 matmul). SDPA is intentionally
    # omitted — it is too slow under co-optimization.
    def _softmax_case(self):
        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        return f, (x,), {}

    def _mlp_case(self):
        mlp, args = self._simple_mlp()
        return mlp, args, {"atol": 0.1, "rtol": 0.1}

    parameter_axes = {
        "sencores": (1, 32),
        "co_optimization": (False, True),
    }

    parameter_models = (("softmax", _softmax_case), ("mlp", _mlp_case))

    def run_case(self, params: dict, factory: Callable) -> None:
        """Run ``factory``'s model for correctness under this combo, applying
        the combo's config at the test-case level (the inherited setUp only
        applies invariants)."""
        with ts_inductor_config.patch(
            sencores=params["sencores"],
            co_optimizing_lx_planning=params["co_optimization"],
            _cpsat_warn_on_cost_expr=False,
        ):
            model, args, kwargs = factory(self)
            torch.compiler.reset()
            with ts_inductor_config.patch(lx_planning=False):
                result_without_lx, hbm_without_lx = self.measure_hbm_transfers(
                    model, args
                )
            torch.compiler.reset()
            with ts_inductor_config.patch(lx_planning=True):
                result_with_lx, hbm_with_lx = self.measure_hbm_transfers(model, args)

        self.assertLess(
            hbm_with_lx,
            hbm_without_lx,
            f"Expected LX planning to reduce HBM transfers, but it did not "
            f"({hbm_with_lx} vs {hbm_without_lx} bytes)",
        )
        # LX placement only moves buffers, so on/off should match within fp16
        # rounding (the difference is a couple of ULP). Tolerances come from the
        # model's kwargs, matching how the correctness path compares elsewhere.
        atol = kwargs.get("atol", 1e-4)
        rtol = kwargs.get("rtol", 1e-5)
        self.assertTrue(
            torch.allclose(result_without_lx, result_with_lx, atol=atol, rtol=rtol),
            "Results do not match between LX planning on and off",
        )


class TestMeasureHBMUsageCoOptimizing(BaseTestScratchpadUsage):
    """Compares HBM transfers with co-optimization off vs on.

    Co-optimization should be ≤ default on every shape, and strictly better
    where adjacent ops disagree on which iteration-space dim to split. The
    canonical case is softmax(dim=0): work_distribution picks rows for the
    pointwise ops and cols for the reductions, forcing 3 of 4 shared buffers to
    HBM by default — co-optimization reconciles them and pins all 4.
    """

    @override
    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        strict: bool = False,
        **kwargs,
    ):
        """Compare HBM transfers with cooptimization off vs on. If
        `strict`, asserts coopt < default; otherwise coopt ≤ default."""
        # Cooptimization needs > 1 core to have anything to optimize; this class
        # applies its own config here at the test-case level.
        with ts_inductor_config.patch(sencores=4, lx_planning=True):
            with ts_inductor_config.patch(co_optimizing_lx_planning=False):
                result_default, hbm_default = self.measure_hbm_transfers(model, args)
            torch.compiler.reset()
            with ts_inductor_config.patch(co_optimizing_lx_planning=True):
                result_coopt, hbm_coopt = self.measure_hbm_transfers(model, args)

        cmp = self.assertLess if strict else self.assertLessEqual
        rel = "<" if strict else "≤"
        cmp(
            hbm_coopt,
            hbm_default,
            f"Expected cooptimization to be {rel} default HBM, got "
            f"coopt={hbm_coopt} default={hbm_default}",
        )
        self.assertTrue(
            torch.allclose(result_default, result_coopt, atol=1e-4),
            "Results do not match between cooptimization on and off",
        )

    def test_softmax_dim0_strictly_lower_hbm(self):
        """The canonical motivating case from the design doc. softmax(dim=0)
        has every adjacent op pair disagreeing on which dim to split, so
        ScratchpadAllocator only pins 1 of 4 shared buffers; co-optimization
        should flip the pointwise ops to cols and pin all 4 → strictly lower HBM."""
        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        self.run_test(f, (x,), strict=True)

    def test_softmax_dim_neg1_no_regression(self):
        """softmax(dim=-1) is the well-behaved baseline where ScratchpadAllocator
        already pins everything pinnable. Co-optimization must match (no regression)."""
        f = functools.partial(torch.softmax, dim=-1)
        x = self.rand_device((512, 1024))
        self.run_test(f, (x,))


class TestCloneAtGraphBoundaries(
    BaseTestScratchpadUsage, metaclass=_ParameterizedScratchpadMeta
):
    """End-to-end tests for clone insertion at graph input/output boundaries.

    The allocator now inserts clone ops on-demand inside _push_allocation rather than
    as a separate pre-scheduling pass.  These tests verify that:
    - graph inputs read by multiple ops get a clone that lands in LX
    - graph outputs that are also read inside the graph get a clone (for the HBM return
      value), while the original buffer is pinned to LX

    Boundary cloning (``clone_at_graph_boundaries()``) is always on, making the
    inserted clone outputs LX-eligible, so this class exercises that path
    directly.
    """

    def _input_clone_when_read_by_multiple_ops(self):
        """A graph input read by two different ops is cloned; the clone lands in LX."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # x is consumed by both exp_op and add_op → two reads → eligible for input clone
            return torch.exp(x) + x

        def assertion_fn(
            result_with_lx,
            n_ops_with_lx,
            mem_usages_with_lx,
            result_no_lx,
            n_ops_no_lx,
            mem_usages_no_lx,
        ):
            self.assertGreater(
                n_ops_with_lx,
                n_ops_no_lx,
                f"Expected the input clone to add an op: {n_ops_no_lx} ops without LX, "
                f"{n_ops_with_lx} with LX",
            )
            self.assertTrue(
                any(u["location"] == "LX" for u in mem_usages_with_lx.values()),
                "Expected at least one LX-allocated buffer after input cloning",
            )
            # Clone is an exact copy; LX planning must not change the numerical result.
            self.assertTrue(
                torch.equal(result_no_lx, result_with_lx),
                "LX input clone changed the numerical result",
            )

        return fn, (x,), {"assertion_fn": assertion_fn}

    def _output_clone_when_intermediate_is_also_graph_output(self):
        """A buffer that is both a graph output and read inside the graph is pinned to LX;
        a clone of it is inserted as the actual (HBM) graph output returned to the caller."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # After CSE, y = exp(x) is produced once.
            # y is a graph output AND is read by add_op → eligible for output clone.
            y = torch.exp(x)
            z = y + 1  # add_op reads y
            return y, z

        def assertion_fn(
            result_with_lx,
            n_ops_with_lx,
            mem_usages_with_lx,
            result_no_lx,
            n_ops_no_lx,
            mem_usages_no_lx,
        ):
            lx_y, lx_z = result_with_lx
            ref_y, ref_z = result_no_lx

            self.assertGreater(
                n_ops_with_lx,
                n_ops_no_lx,
                f"Expected the output clone to add an op: {n_ops_no_lx} ops without LX, "
                f"{n_ops_with_lx} with LX",
            )
            self.assertTrue(
                any(u["location"] == "LX" for u in mem_usages_with_lx.values()),
                "Expected at least one LX-allocated buffer after output cloning",
            )
            # Clone is an exact copy; LX planning must not change the numerical result.
            self.assertTrue(
                torch.equal(ref_y, lx_y), "LX output clone changed result y"
            )
            self.assertTrue(
                torch.equal(ref_z, lx_z), "LX output clone changed result z"
            )

        return fn, (x,), {"assertion_fn": assertion_fn}

    def _input_read_at_multiple_offsets_is_correct(self):
        """A graph input read by one op at two distinct offsets must not be
        LX-pinned.

        An LX-pinned buffer is addressed by a single base (SDSC start_address
        = allocation["lx"]); per-access slice offsets are not folded into it.
        Pinning ``x`` for ``x[:, 0:512] + x[:, 512:1024]`` made both reads
        resolve to the LX base, so the op computed ``x0 + x0`` instead of
        ``x0 + x1``. The allocator now skips such inputs (they stay in HBM,
        where multi-offset reads work)."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # The fused add reads x at offset 0 and offset 512 -> two distinct
            # offsets on the same buffer -> ineligible for LX pinning.
            return x[:, 0:512] + x[:, 512:1024]

        def assertion_fn(
            result_with_lx,
            n_ops_with_lx,
            mem_usages_with_lx,
            result_no_lx,
            n_ops_no_lx,
            mem_usages_no_lx,
        ):
            self.assertTrue(
                torch.equal(result_no_lx, result_with_lx),
                "Multi-offset input read produced wrong values under LX planning",
            )

        return fn, (x,), {"assertion_fn": assertion_fn}

    def _input_feeding_reduction_is_cloned_and_correct(self):
        """A graph input read by a reduction is LX-cloned, with the clone's
        per-core split re-keyed correctly.

        push_allocation_with_clone projects the accepted physical view through
        the clone's own coordinates and commits that complete division. Copying
        a reduction consumer's logical split verbatim could split the wrong axis
        of the full-shape clone (wrong values / SDSC abort at multi-core). The
        numerical failure only manifests when work is split across cores; here
        (sencores=1) we assert the clone is inserted and the result is correct.
        Multi-core numerical coverage lives in
        tests/inductor/test_inductor_ops.py (max_sub_broadcast, aminmax,
        softmax)."""
        x = self.rand_device((64, 256))

        def fn(x):
            # x feeds the max reduction (and the sub) -> reduction consumer.
            return x - torch.unsqueeze(torch.max(x, dim=1).values, dim=1)

        def assertion_fn(
            result_with_lx,
            n_ops_with_lx,
            mem_usages_with_lx,
            result_no_lx,
            n_ops_no_lx,
            mem_usages_no_lx,
        ):
            self.assertGreater(
                n_ops_with_lx,
                n_ops_no_lx,
                "Expected a boundary clone for the reduction-fed input, but the op "
                f"count did not grow ({n_ops_no_lx} -> {n_ops_with_lx})",
            )
            self.assertTrue(
                any(u["location"] == "LX" for u in mem_usages_with_lx.values()),
                "Expected at least one LX-allocated buffer for the reduction input",
            )
            self.assertTrue(
                torch.equal(result_no_lx, result_with_lx),
                "Reduction-fed input changed result under LX planning",
            )

        return fn, (x,), {"assertion_fn": assertion_fn}

    def _input_read_partially_is_correct(self):
        """A graph input read only over a sub-extent (a slice) must not be
        LX-pinned.

        Strided partial reads of a multi-dim LX buffer mis-address against the
        single LX base. Pinning ``x`` for ``add(x[:, :, 0:64].clone(),
        x[:, :, 0:64])`` produced wrong values; the allocator now leaves such
        inputs in HBM, where partial reads work."""
        x = self.rand_device((3, 3, 192))

        def fn(x):
            s = x[:, :, 0:64]  # partial inner-dim slice -> sub-extent read
            return torch.add(s.clone(), s)

        def assertion_fn(
            result_with_lx,
            n_ops_with_lx,
            mem_usages_with_lx,
            result_no_lx,
            n_ops_no_lx,
            mem_usages_no_lx,
        ):
            self.assertTrue(
                torch.equal(result_no_lx, result_with_lx),
                "Partial input read produced wrong values under LX planning",
            )

        return fn, (x,), {"assertion_fn": assertion_fn}

    parameter_axes = {
        "sencores": (1, 32),
        "co_optimization": (False, True),
    }

    parameter_models = (
        ("multiple_ops_read", _input_clone_when_read_by_multiple_ops),
        (
            "output_is_intermediate",
            _output_clone_when_intermediate_is_also_graph_output,
        ),
        ("multiple_offset_input_read", _input_read_at_multiple_offsets_is_correct),
        ("input_reduction", _input_feeding_reduction_is_cloned_and_correct),
        ("partial_input_read", _input_read_partially_is_correct),
    )

    def _compile_and_inspect(
        self,
        f: Callable,
        args: tuple,
    ) -> tuple:
        """Compile f, capture op count and mem_usages after the allocator runs.

        Handles both single-tensor and tuple outputs.
        Returns (result_on_cpu, n_ops, mem_usages).
        """
        n_ops_captured: list[int] = []
        mem_usages: dict[str, dict] = {}

        def visitor(graph: GraphLowering) -> None:
            n_ops_captured.append(len(graph.operations))
            for op in graph.operations:
                buf_name = op.name
                buffer = graph.get_buffer(buf_name)
                layout = buffer.get_layout()
                device_layout = layout.device_layout
                allocation = getattr(layout, "allocation", {})
                mem_usages[buf_name] = {
                    "location": "LX" if "lx" in allocation else "HBM",
                    "size": math.prod(device_layout.device_size[:-1]) * 128,
                }

        with self.pre_scheduling_iterating_pass(visitor):
            compiled_kernel = torch.compile(f, fullgraph=True)
            raw = compiled_kernel(*args)
            if isinstance(raw, tuple):
                result = tuple(r.to("cpu") for r in raw)
            else:
                result = raw.to("cpu")

        n_ops = n_ops_captured[0] if n_ops_captured else 0
        return result, n_ops, mem_usages

    def run_case(self, params: dict, factory: Callable) -> None:
        """Run ``factory``'s model for correctness under this combo, applying
        the combo's config at the test-case level (the inherited setUp only
        applies invariants)."""
        with ts_inductor_config.patch(
            sencores=params["sencores"],
            co_optimizing_lx_planning=params["co_optimization"],
            _cpsat_warn_on_cost_expr=False,
        ):
            model, args, kwargs = factory(self)
            torch.compiler.reset()
            with ts_inductor_config.patch(lx_planning=True):
                result_with_lx, n_ops_with_lx, mem_usages_with_lx = (
                    self._compile_and_inspect(model, args)
                )
            torch.compiler.reset()
            with ts_inductor_config.patch(lx_planning=False):
                result_no_lx, n_ops_no_lx, mem_usages_no_lx = self._compile_and_inspect(
                    model, args
                )

            assertion_fn = kwargs["assertion_fn"]
            assertion_fn(
                result_with_lx,
                n_ops_with_lx,
                mem_usages_with_lx,
                result_no_lx,
                n_ops_no_lx,
                mem_usages_no_lx,
            )


class TestIntermediatePartialReadNotPinned(BaseTestScratchpadUsage):
    """An *intermediate* buffer read partially (sliced) must not be LX-pinned.

    Companion to ``TestCloneAtGraphBoundaries``, which guards graph
    input/output clones. ``residency_rules._residency_reasons`` applies the
    same ``buffer_not_read_in_full`` guard to intermediate buffers: a buffer that is
    produced in full and then read over a sub-extent (an inner-dim slice that
    feeds a chained op) would be LX-pinned and mis-addressed by the single-base
    LX path. Without the intermediate guard this regresses to a large
    numerical mismatch (~94%).
    """

    def test_sliced_intermediate_is_correct(self):
        # Both leading dims large so the chained ops divide cleanly across
        # cores (no core-division mismatch) — the case that would otherwise
        # LX-pin the sliced intermediate. allow_all_ops_in_lx_planning makes
        # the intermediate LX-eligible; sencores=32 gives the multi-core split.
        x = self.rand_device((128, 192, 256))

        def fn(x):
            t = torch.exp(x)  # full intermediate, produced once
            s = t[:, :, 32:96]  # sub-stick partial read of the intermediate
            return s.clone() + s

        cpu_result = fn(x.to("cpu"))

        with ts_inductor_config.patch(
            lx_planning=True,
            allow_all_ops_in_lx_planning=True,
            sencores=32,
        ):
            result, mem_usages = self.compile_and_collect_mem_usage(fn, (x,))

        # The scenario must still exercise LX-pinning, else it would pass
        # trivially without covering the guard.
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer in this scenario",
        )
        torch.testing.assert_close(
            result,
            cpu_result,
            atol=0.1,
            rtol=0.1,
            msg="sliced intermediate miscompiled — is the partial-read guard present?",
        )


@unittest.skipUnless(_HAS_ORTOOLS, "the joint solve needs ortools")
class TestIntermediateReadTwice(BaseTestScratchpadUsage):
    """An *intermediate* buffer one consumer reads twice is LX-pinned only under
    a division that owns it the same way through both reads.

    ``a + a.permute(1, 0, 2)`` reads ``a`` once in step and once transposed. A
    split of the consumer's dim 0 slices ``a`` along dim 0 for the first read
    and along dim 1 for the second, so no residency of ``a`` serves both. The
    match table once looked at the first read alone: the solve pinned ``a``
    under such a split, and the post-solve ownership check could only raise.
    """

    def test_twice_read_intermediate_is_correct(self):
        # Small enough that a fits LX, so the solve wants it resident.
        x = self.rand_device((64, 64, 128))
        y = self.rand_device((64, 64, 128))

        def fn(x, y):
            a = x + y
            return a + a.permute(1, 0, 2)

        cpu_result = fn(x.to("cpu"), y.to("cpu"))

        with ts_inductor_config.patch(
            lx_planning=True,
            co_optimizing_lx_planning=True,
            sencores=32,
        ):
            result, _ = self.compile_and_collect_mem_usage(fn, (x, y))

        torch.testing.assert_close(result, cpu_result, atol=0.1, rtol=0.1)


class TestLivenessIndicesAreDistinct(BaseTestScratchpadUsage):
    """``calculate_liveness`` records one distinct op index per accessing op.

    ``rw.reads | rw.writes`` is a set of *dependencies*, not of names, so an op
    touching one buffer through two index expressions contributes two deps naming
    it; appending per dep would repeat that op's index. The repeat is invisible to
    ``start_time``/``end_time``, inflates ``read_count``, and would let a buffer
    written and read by the same op pass as an in-place parent -- so
    ``calculate_liveness`` collapses it and
    :class:`LifetimeBoundBuffer` asserts the result is strictly increasing.

    That assertion is exercised elsewhere over hand-built lists, which proves it
    fires but not that the producer satisfies it. Only a real lowering can show
    that, which is what this test does.
    """

    def test_fused_slice_input_has_one_use_per_op(self):
        seen: dict[str, list[int]] = {}
        multi_dep: list[str] = []

        def visitor(graph: GraphLowering) -> None:
            seen.update(calculate_liveness(graph))
            for op in graph.operations:
                rw = op_read_writes(op)
                names = [dep.name for dep in rw.reads | rw.writes]
                multi_dep.extend(n for n in set(names) if names.count(n) > 1)

        def fn(x):
            # One fused add reading x at offset 0 and at offset 512: two deps on
            # the same buffer from a single op, the shape the dedup exists for.
            return x[:, 0:512] + x[:, 512:1024]

        with self.pre_scheduling_iterating_pass(visitor):
            # (64, 1024) matches the shape ``_input_read_at_multiple_offsets_is_correct``
            # already drives this same expression with, so the two-deps lowering is
            # known to hold for it.
            torch.compile(fn, fullgraph=True)(self.rand_device((64, 1024))).to("cpu")

        self.assertTrue(seen, "liveness visitor never ran")
        # The scenario must still produce the two-deps-one-buffer shape, else the
        # check below is free for every buffer and covers nothing.
        self.assertTrue(
            multi_dep, "no op read one buffer through two deps in this scenario"
        )
        for name, uses in seen.items():
            # Distinct *and* ascending in one assertion: exactly what
            # LifetimeBoundBuffer requires of ``uses``.
            self.assertEqual(uses, sorted(set(uses)), name)


@unittest.skipUnless(
    _HAS_ORTOOLS, "forcing a CP-SAT timeout requires the real ortools solver"
)
class TestCpSatTimeoutFallback(BaseTestScratchpadUsage):
    """A CP-SAT solve that times out leaves the graph compilable: the open mode
    replans over the committed divisions, and when that solve times out too,
    nothing is placed in LX."""

    @contextmanager
    def _zero_solver_timeout(self):
        """Force every CP-SAT solve to run with a 0-second budget so it returns
        ``UNKNOWN`` and the solve raises ``SolveError``."""
        from ortools.sat.python import cp_model

        original_solve = cp_model.CpSolver.Solve

        def zero_timeout_solve(solver_self, *args, **kwargs):
            solver_self.parameters.max_time_in_seconds = 0.0
            return original_solve(solver_self, *args, **kwargs)

        with patch.object(cp_model.CpSolver, "Solve", zero_timeout_solve):
            yield

    def test_softmax_timeout_leaves_graph_unplanned(self):
        """A reduction chain (softmax(dim=0)) compiles correctly with no LX
        plan when both solves time out."""
        from torch_spyre._inductor.scratchpad import allocator as allocator_module

        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        with ts_inductor_config.patch(
            sencores=32,
            co_optimizing_lx_planning=True,
            _cpsat_warn_on_cost_expr=False,
        ):
            torch.compiler.reset()
            with ts_inductor_config.patch(lx_planning=False):
                result_without_lx, _ = self.compile_and_collect_mem_usage(f, (x,))
            torch.compiler.reset()
            with (
                self._zero_solver_timeout(),
                _recording_division_modes() as modes,
                self.assertLogs(allocator_module.logger, level="INFO") as logs,
                ts_inductor_config.patch(lx_planning=True),
            ):
                result_with_lx, mem_usages = self.compile_and_collect_mem_usage(f, (x,))

        self.assertEqual(modes, ["open", "committed"])
        self.assertIn("nothing is placed in LX", "\n".join(logs.output))
        self.assertFalse(
            [name for name, u in mem_usages.items() if u["location"] == "LX"],
            "a failed solve must leave every buffer in HBM",
        )
        # The unplanned graph is the LX-off graph, so the results agree within
        # fp16 rounding.
        self.assertTrue(
            torch.allclose(result_without_lx, result_with_lx, atol=1e-4, rtol=1e-5),
            "Results do not match between LX planning on and off",
        )


class TestSolveErrorFallback(unittest.TestCase):
    """``scratchpad_planning`` after a ``SolveError``: an open-mode allocator
    replans over the committed divisions and keeps its post-allocation passes;
    when that fails too, or the failing allocator was already committed,
    nothing is placed and the failure is logged at INFO.
    Pure dispatch, no device needed."""

    def _config(self, co_optimizing: bool):
        return ts_inductor_config.patch(
            co_optimizing_lx_planning=co_optimizing,
            enable_lx_context_switching=True,
            _cpsat_warn_on_cost_expr=False,
        )

    def _plan_with_failing(self, failing, replan_side_effect=None):
        """Run ``scratchpad_planning`` with ``failing`` raising ``SolveError``
        and every other allocator's ``plan_allocation`` mocked."""
        from torch_spyre._inductor.scratchpad import allocator as allocator_module
        from torch_spyre._inductor.scratchpad.plan_solver import SolveError

        graph = object()
        with (
            patch.object(
                failing, "plan_allocation", side_effect=SolveError("forced")
            ) as failed,
            patch.object(
                allocator_module.ScratchpadAllocator,
                "plan_allocation",
                autospec=True,
                side_effect=replan_side_effect,
            ) as replan,
            self.assertLogs(allocator_module.logger, level="INFO") as logs,
        ):
            allocator_module.scratchpad_planning(graph, failing)
        failed.assert_called_once_with(graph)
        for call in replan.call_args_list:
            self.assertIs(call.args[1], graph)
        return replan, logs

    def test_open_mode_replans_committed(self):
        from torch_spyre._inductor.scratchpad import allocator as allocator_module
        from torch_spyre._inductor.scratchpad.lx_context_switching import (
            LxContextSwitchingPass,
        )

        with self._config(co_optimizing=True):
            failing = allocator_module.select_allocator()
            post_passes = failing.post_optimization_passes
            self.assertTrue(
                any(isinstance(p, LxContextSwitchingPass) for p in post_passes)
            )
            replan, logs = self._plan_with_failing(failing)
        replan.assert_called_once()
        (committed, _) = replan.call_args.args
        self.assertIs(type(committed), allocator_module.ScratchpadAllocator)
        self.assertEqual(committed.division_mode, "committed")
        self.assertIs(committed.post_optimization_passes, post_passes)
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])
        self.assertIn(
            "replanning with the committed work divisions",
            "\n".join(logs.output),
        )

    def test_failed_committed_replan_places_nothing(self):
        from torch_spyre._inductor.scratchpad import allocator as allocator_module
        from torch_spyre._inductor.scratchpad.plan_solver import SolveError

        with self._config(co_optimizing=True):
            failing = allocator_module.select_allocator()
            replan, logs = self._plan_with_failing(
                failing, replan_side_effect=SolveError("forced again")
            )
        replan.assert_called_once()
        self.assertEqual([r.levelname for r in logs.records], ["INFO", "INFO"])
        self.assertIn("nothing is placed in LX", logs.output[-1])
        self.assertIn("forced again", logs.output[-1])

    def test_committed_mode_failure_places_nothing(self):
        from torch_spyre._inductor.scratchpad import allocator as allocator_module

        with self._config(co_optimizing=False):
            failing = allocator_module.select_allocator()
            replan, logs = self._plan_with_failing(failing)
        replan.assert_not_called()
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])
        self.assertIn("nothing is placed in LX", logs.output[0])


class TestSelectAllocator(unittest.TestCase):
    """select_allocator maps config -> allocator so the allocator never
    inspects config itself. Pure dispatch, no device needed."""

    def test_co_optimizing_opens_the_divisions(self):
        from torch_spyre._inductor.scratchpad.allocator import (
            ScratchpadAllocator,
            select_allocator,
        )

        for co_optimizing, mode in ((True, "open"), (False, "committed")):
            with self.subTest(co_optimizing_lx_planning=co_optimizing):
                with ts_inductor_config.patch(
                    co_optimizing_lx_planning=co_optimizing,
                    _cpsat_warn_on_cost_expr=False,
                ):
                    a = select_allocator()
                self.assertIs(type(a), ScratchpadAllocator)
                self.assertEqual(a.division_mode, mode)

    def test_division_strategy_picks_the_encoding(self):
        from torch_spyre._inductor.scratchpad.allocator import select_allocator
        from torch_spyre._inductor.scratchpad.encoding_strategy import (
            PairTableStrategy,
            EnumeratedStrategy,
        )
        from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy

        for strategy, cls in (
            ("lazy", LazyStrategy),
            ("enumerated", EnumeratedStrategy),
            ("table", PairTableStrategy),
        ):
            with self.subTest(lx_division_strategy=strategy):
                with ts_inductor_config.patch(lx_division_strategy=strategy):
                    a = select_allocator()
                self.assertIs(a.strategy_class, cls)


class TestInplaceEdgeGate(unittest.TestCase):
    """Unit tests for ``residency_rules._inplace_edge_ok``, the sole predicate
    defining a legal in-place edge (shared by the normal producer path and the
    graph-input clone reverse-parent path, issue #3212)."""

    def _base_kwargs(self) -> dict:
        layout = ("device-layout-sentinel",)
        return dict(
            child_pointwise_inputs=["p"],
            parent_name="p",
            child_size_per_core=128,
            parent_size_per_core=128,
            child_device_layout=layout,
            parent_device_layout=layout,
            child_start=5,
            parent_end=5,
            child_core_div_mismatch=False,
        )

    def test_all_conditions_met(self):
        self.assertTrue(residency_rules._inplace_edge_ok(**self._base_kwargs()))

    def test_each_condition_blocks_edge(self):
        # Each entry flips exactly one of the five conditions to failing.
        for label, overrides in {
            "parent not a pointwise input": {"child_pointwise_inputs": []},
            "per-core size mismatch": {"parent_size_per_core": 127},
            "device-layout mismatch": {"parent_device_layout": ("other",)},
            "not single handoff tick": {"parent_end": 4},
            "child core-division mismatch": {"child_core_div_mismatch": True},
        }.items():
            with self.subTest(label):
                kwargs = self._base_kwargs()
                kwargs.update(overrides)
                self.assertFalse(
                    residency_rules._inplace_edge_ok(**kwargs),
                    f"edge should be forbidden when: {label}",
                )

    def test_division_invariant_defers_size_and_core_div(self):
        # Under division_invariant (the co-optimizing path), the per-core size match
        # and core-division check are deferred to the solver, so a mismatch there is
        # ignored -- but the division-invariant preconditions still gate.
        self.assertTrue(
            residency_rules._inplace_edge_ok(
                **{
                    **self._base_kwargs(),
                    "parent_size_per_core": 999,
                    "child_core_div_mismatch": True,
                    "division_invariant": True,
                }
            )
        )
        for overrides in (
            {"child_pointwise_inputs": []},
            {"parent_device_layout": ("other",)},
            {"parent_end": 4},
        ):
            with self.subTest(str(overrides)):
                self.assertFalse(
                    residency_rules._inplace_edge_ok(
                        **{
                            **self._base_kwargs(),
                            **overrides,
                            "division_invariant": True,
                        }
                    )
                )

    def test_counted_loop_overrides_block_handoff(self):
        """A buffer kept live across a counted loop cannot hand off in place.

        The parent's last use and the child's first use abut at tick 5, but a
        loop-widened parent end or child start moves one of them off that tick.
        """
        from torch_spyre._inductor.scratchpad.drain_plans import (
            _handoff_child_start,
            _handoff_parent_end,
        )

        lifetimes = {"p": [2, 5], "c": [5, 7]}
        for label, starts, ends, expected in (
            ("no overrides", {}, {}, True),
            ("parent live through the loop", {}, {"p": 9}, False),
            ("child widened to the loop start", {"c": 0}, {}, False),
        ):
            with self.subTest(label):
                kwargs = {
                    **self._base_kwargs(),
                    "child_start": _handoff_child_start("c", lifetimes, starts),
                    "parent_end": _handoff_parent_end("p", lifetimes, ends),
                }
                self.assertIs(residency_rules._inplace_edge_ok(**kwargs), expected)


class TestInPlaceMutationCoOptimizing(BaseTestScratchpadUsage):
    """Plain in-place mutations compile under the co-optimizing path
    (issue #3940).

    Candidate enumeration calls ``_output_stride_to_device_size`` (through
    ``_split_fits_sticks``) on every op. An op whose buffer is mutated in place
    carries ``MutationLayoutSHOULDREMOVE``, which has no ``device_layout`` --
    the helper must unwrap it via ``real_layout()``. Sliced mutations
    (``x[:, 32:96] = ...``) never reach the helper because the
    offset-mutation component routes them to a fixed division first, which is
    exactly what masked this; hence the plain-mutation cases here."""

    def _compile_and_compare(self, fn, args, cpu_args=None):
        if cpu_args is None:
            cpu_args = tuple(t.to("cpu") for t in args)
        cpu_result = fn(*cpu_args)
        with ts_inductor_config.patch(
            lx_planning=True,
            co_optimizing_lx_planning=True,
        ):
            device_result = torch.compile(fn, fullgraph=True)(*args).to("cpu")
        torch.testing.assert_close(device_result, cpu_result, atol=1e-2, rtol=1e-3)

    def test_inplace_add(self):
        def fn(dst, a):
            dst.add_(a)
            return dst * 2.0

        self._compile_and_compare(
            fn, (self.rand_device((64, 256)), self.rand_device((64, 256)))
        )

    def test_inplace_copy(self):
        def fn(dst, a, b):
            dst.copy_(a + b)
            return dst

        self._compile_and_compare(
            fn,
            (
                torch.zeros(64, 256, dtype=torch.float16, device="spyre"),
                self.rand_device((64, 256)),
                self.rand_device((64, 256)),
            ),
        )

    def test_inplace_index_copy(self):
        def fn(cache, idx, v):
            cache.index_copy_(2, idx, v)
            return cache

        cache = torch.zeros(1, 8, 128, 64, dtype=torch.float16, device="spyre")
        idx = torch.tensor([3], dtype=torch.int32, device="spyre")
        v = self.rand_device((1, 8, 1, 64))
        # CPU index_copy_ requires an int64 index; Spyre wants int32.
        self._compile_and_compare(
            fn,
            (cache, idx, v),
            cpu_args=(cache.to("cpu"), idx.to("cpu").long(), v.to("cpu")),
        )


class TestOffsetMutationCoOptimizing(BaseTestScratchpadUsage):
    def test_deep_reader_chain_and_copy_back(self):
        """Offset writes stay correct past the old two-reader boundary."""

        def fn(x, value, dst):
            x[:, 32:96] = value
            hop1 = x + 1
            hop2 = hop1 * 0.5
            hop3 = hop2 + 2
            dst.copy_(hop3)
            return dst + torch.amax(dst, dim=-1, keepdim=True), x

        args = (
            self.rand_device((128, 256)),
            self.rand_device((128, 64)),
            self.rand_device((128, 256)),
        )
        cpu_args = tuple(t.cpu() for t in args)
        expected = fn(*cpu_args)
        with ts_inductor_config.patch(
            lx_planning=True,
            co_optimizing_lx_planning=True,
        ):
            result = torch.compile(fn, fullgraph=True)(*args)
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(actual.cpu(), reference, atol=1e-2, rtol=1e-3)
        for actual, reference in zip(args, cpu_args):
            torch.testing.assert_close(actual.cpu(), reference, atol=1e-2, rtol=1e-3)

    @unittest.skipUnless(_HAS_ORTOOLS, "co-optimizing path needs ortools")
    def test_attention_before_padded_ffn_keeps_division_candidates(self):
        """The #4990 pad must not pin the upstream attention's divisions."""
        from torch_spyre._inductor.pass_utils import _is_matmul_op
        from torch_spyre._inductor.scratchpad.utils import _writes_at_constant_offset

        def fn(q, k, v, gate_up, down):
            scores = (q @ k.transpose(-1, -2)) * 0.125
            attention = torch.softmax(scores, dim=-1) @ v
            rows = attention.reshape(4, 64)
            padded = torch.nn.functional.pad(rows, (0, 0, 0, 4))
            ffn = torch.nn.functional.silu(padded @ gate_up) @ down
            return ffn[:4] + rows

        args = tuple(
            self.rand_device(shape) * 0.125
            for shape in ((4, 1, 64), (4, 128, 64), (4, 128, 64), (64, 128), (128, 64))
        )
        expected = fn(*(t.cpu() for t in args))
        captured = []
        original = GraphBuilder._division_map

        def capture(allocator, graph, **kwargs):
            division_map = original(allocator, graph, **kwargs)
            divisions = division_map.divisions
            offset_writes = [
                i
                for i, op in enumerate(graph.operations)
                if _writes_at_constant_offset(op)
            ]
            if offset_writes:
                # The two batched matmuls precede the pad. A whole-component
                # pin would leave every one of these upstream ops a singleton.
                captured.append(
                    any(
                        len(divisions[op.name]) > 1
                        for op in graph.operations[: min(offset_writes)]
                        if _is_matmul_op(op)
                    )
                )
            return division_map

        with (
            patch.object(GraphBuilder, "_division_map", capture),
            ts_inductor_config.patch(
                lx_planning=True,
                co_optimizing_lx_planning=True,
            ),
        ):
            result = torch.compile(fn, fullgraph=True)(*args)
        self.assertTrue(captured, "the graph must contain an offset mutation")
        self.assertTrue(all(captured), "upstream attention lost its candidate menu")
        torch.testing.assert_close(result.cpu(), expected, atol=1e-2, rtol=1e-3)


class TestBoundaryCloneInPlace(BaseTestScratchpadUsage):
    """In-place reuse of boundary-clone buffers by the solve's graph (#3212),
    over the committed divisions and in the co-optimizing path.

    These are assertion-style tests (they inspect the graph the allocator builds
    and the final LX addresses), so they live in a plain non-parameterized class
    rather than the model-sweep ``TestCloneAtGraphBoundaries``."""

    @staticmethod
    def _in_place_parents(solve_graph) -> dict[str, set[str]]:
        """Each node's in-place parents in a solve graph."""
        return {
            n.name: {e.src.name for e in n.inputs if e.in_place}
            for n in solve_graph.get_nodes()
        }

    @contextmanager
    def _capture_solve_graphs(self):
        """The allocator's solve graphs, in either division mode."""
        captured: list = []
        orig = GraphBuilder._build_graph

        def spy(inner_self, *a, **k):
            result = orig(inner_self, *a, **k)
            captured.append(result)
            return result

        with patch.object(GraphBuilder, "_build_graph", spy):
            yield captured

    @unittest.skipUnless(_HAS_ORTOOLS, "co-optimizing path needs ortools")
    def test_division_invariant_edges_respect_per_input_pointwise(self):
        """``_determine_in_place_division_invariant`` routes through
        ``_inplace_edge_ok(division_invariant=True)``, so every in-place parent it
        returns is a pointwise-eligible read of the child's op -- the per-input
        check the loop previously omitted (#3212 follow-up).

        Invariant guard: an input read at a different index than the output write
        must never be offered as an in-place parent pre-solver. The assertion locks
        the shared gate into this path; index-specific eligibility is covered by
        PointwiseInPlaceInputsTest."""

        x = self.rand_device((64, 1024))

        def fn(x):
            a = x + 1.0
            b = a * 2.0
            return b + 3.0

        # The check must run inside the spy: _op_inputs_good_for_lx_inplace needs
        # the virtualized (V) compile context, which is torn down once the
        # torch.compile block exits.
        violations: list[str] = []
        edge_count = [0]
        called = [False]
        orig = GraphBuilder._determine_in_place_division_invariant

        def spy(self, graph):
            result = orig(self, graph)
            called[0] = True
            op_by_name = {op.name: op for op in graph.operations}
            for buf_name, parents in result.items():
                op = op_by_name.get(buf_name)
                if op is None:
                    continue
                eligible = residency_rules._op_inputs_good_for_lx_inplace(op)
                for parent in parents:
                    edge_count[0] += 1
                    if parent not in eligible:
                        violations.append(f"{buf_name} -> {parent}")
            return result

        with patch.object(
            GraphBuilder,
            "_determine_in_place_division_invariant",
            spy,
        ):
            with ts_inductor_config.patch(
                lx_planning=True,
                co_optimizing_lx_planning=True,
                _cpsat_warn_on_cost_expr=False,
            ):
                torch.compile(fn, fullgraph=True)(x)

        self.assertTrue(
            called[0], "_determine_in_place_division_invariant was not called"
        )
        self.assertFalse(
            violations,
            f"in-place parents that are not pointwise-eligible reads: {violations}",
        )
        self.assertTrue(edge_count[0] > 0, "no in-place edges were produced to verify")

    def test_input_clone_reused_in_place_by_last_consumer(self):
        """The input clone's last consumer names the clone as an in-place parent
        (issue #3212), so the two may share an LX slot.

        The clone (buffer named after the graph input) is pinned to LX and dies at
        its last read; when that last reader is pointwise and writes a same-shape
        buffer that is itself read again (a realized candidate), the solve's graph
        (``GraphBuilder._build_graph``) marks the clone as that consumer's
        in-place parent. We capture the graph the allocator builds and assert the
        reverse-parent edge is present. Values must be unchanged.

        ``x * 2 + x * 3`` gives x two direct pointwise readers; the second (``x*3``)
        is x's last read and its output feeds the final add, so that output is a
        candidate that names the input clone as parent. (A shape like
        ``torch.abs(x) + x`` would instead have x's last reader be the graph output
        itself -- single-use, never a candidate -- so no edge, correctly.)"""

        x = self.rand_device((64, 1024))

        def fn(x):
            return x * 2.0 + x * 3.0

        with self._capture_solve_graphs() as captured:
            # Over the committed divisions; the co-optimizing path is the
            # sibling test below.
            with ts_inductor_config.patch(
                lx_planning=True, co_optimizing_lx_planning=False
            ):
                compiled = torch.compile(fn, fullgraph=True)
                result = compiled(x).to("cpu")

        self.assertTrue(captured, "GraphBuilder._build_graph was not called")
        edge_found = False
        for solve_graph in captured:
            # Input clones are exactly the nodes whose first access is a read.
            input_clone_names = {
                n.name for n in solve_graph.get_nodes() if n.first_use_is_read
            }
            parents = self._in_place_parents(solve_graph).values()
            if any(p & input_clone_names for p in parents):
                edge_found = True
                break
        self.assertTrue(
            edge_found,
            "expected the input clone to be named as an in-place parent by its "
            "last consumer",
        )
        self.assertTrue(
            torch.allclose(fn(x.to("cpu")), result, atol=1e-2, rtol=1e-3),
            "input clone in-place reuse changed the numerical result",
        )

    @unittest.skipUnless(_HAS_ORTOOLS, "co-optimizing path needs ortools")
    def test_input_clone_reverse_parent_in_cooptimizing_path(self):
        """The co-optimizing (joint CP-SAT) path also lets an input clone's last
        consumer name it as an in-place parent, with the merge gate populated
        (issue #3212). Mirrors the committed-division test above."""
        from torch_spyre._inductor.scratchpad.plan_solver import BufferType

        x = self.rand_device((64, 1024))

        def fn(x):
            return x * 2.0 + x * 3.0

        with self._capture_solve_graphs() as captured:
            with ts_inductor_config.patch(
                lx_planning=True,
                co_optimizing_lx_planning=True,
                _cpsat_warn_on_cost_expr=False,
            ):
                result = torch.compile(fn, fullgraph=True)(x).to("cpu")

        self.assertTrue(captured, "GraphBuilder._build_graph was not called")
        edge_found = False
        for solve_graph in captured:
            input_clones = {
                n.name
                for n in solve_graph.get_nodes()
                if n.boundary == BufferType.Input
            }
            for child, parents in self._in_place_parents(solve_graph).items():
                merged = parents & input_clones
                # The CP-SAT merge also needs the division gate populated: a
                # pair table, or the clone relation the derived encoding reads.
                if any(
                    e.src.name in merged
                    and (e.pairs is not None or e.relation is not None)
                    for e in solve_graph.nodes[child].inputs
                ):
                    edge_found = True
                    break
        self.assertTrue(
            edge_found,
            "expected an input clone to be an in-place parent (with a "
            "cd_parent_matches gate) in the co-optimizing path",
        )
        self.assertTrue(
            torch.allclose(fn(x.to("cpu")), result, atol=1e-2, rtol=1e-3),
            "co-opt input clone in-place reuse changed the numerical result",
        )

    def test_output_feeding_buffer_reused_in_place(self):
        """A buffer feeding a graph output participates in in-place merge via the
        normal computed-buffer path -- issue #3212's output side needs no dedicated
        metadata (unlike the input side).

        An LX-pinned buffer that is (or is cloned into) a graph output is a normal
        op-backed ComputedBuffer, so the solve's graph already gives it in-place
        parents (as a child of its producer's input) and lets consumers name it as a
        parent. Here ``y`` is a graph output also read internally; its pointwise
        consumer ``p`` (whose result is itself read, so it is a realized candidate)
        reuses ``y``'s slot -- i.e. the output-feeding buffer is an in-place parent.
        This is a regression guard: if output in-place ever breaks, it fails here."""
        from torch_spyre._inductor.pass_utils import op_short_name

        x = self.rand_device((64, 1024))

        def fn(x):
            y = x * 2.0  # graph output, also read internally -> pinned + cloned
            p = y + 1.0  # p reads y (pointwise) at y's last use
            q = p * 3.0  # p read by q -> p is a realized candidate
            return y, q

        output_feeders: set[str] = set()

        def collect_feeders(graph: GraphLowering) -> None:
            by_name = {op.name: op for op in graph.operations}
            for name in graph.get_output_names():
                output_feeders.add(name)
                op = by_name.get(name)
                # A graph output that is a clone pins the buffer it copies.
                if op is not None and op_short_name(op) == "clone":
                    output_feeders.update(d.name for d in op.get_read_writes().reads)

        with self.pre_scheduling_iterating_pass(collect_feeders):
            with self._capture_solve_graphs() as captured:
                # Over the committed divisions (see the input-clone test above).
                with ts_inductor_config.patch(
                    lx_planning=True, co_optimizing_lx_planning=False
                ):
                    compiled = torch.compile(fn, fullgraph=True)
                    ry, rq = compiled(x)
                    ry, rq = ry.to("cpu"), rq.to("cpu")

        self.assertTrue(captured, "GraphBuilder._build_graph was not called")
        merged = False
        for solve_graph in captured:
            for parents in self._in_place_parents(solve_graph).values():
                # An output-feeding buffer is used as an in-place *parent*: some
                # consumer names it among its in-place parents (matches the
                # docstring; the sibling aliasing test uses the same tighter check).
                if parents & output_feeders:
                    merged = True
                    break
        self.assertTrue(
            merged,
            "expected a graph-output-feeding buffer to be reused in place as a "
            "parent (output-clone in-place should work via the normal path)",
        )
        ref_y, ref_q = fn(x.to("cpu"))
        self.assertTrue(
            torch.allclose(ref_y, ry, atol=1e-2, rtol=1e-3), "output y changed"
        )
        self.assertTrue(
            torch.allclose(ref_q, rq, atol=1e-2, rtol=1e-3), "output q changed"
        )

    def test_returned_buffer_reused_in_place_is_still_returned_correctly(self):
        """Aliasing guard: a returned buffer read multiple times internally may have
        its LX slot reused in-place by its last consumer, yet the value handed to the
        caller must be intact (issue #3212 aliasing risk).

        ``y`` is returned *and* read three times inside the graph, so it is pinned to
        LX and copied to HBM for the return. Its last reader ``v`` is pointwise and
        its result is read again (``u``), so ``v`` is a realized candidate that the
        allocator may let reuse ``y``'s slot in place. That reuse happens at ``y``'s
        *last* tick, while the HBM copy of ``y`` is taken at its *first* -- so the
        returned ``y`` is captured before its slot is overwritten. We assert the
        reuse edge is actually offered (the hazard is exercised, not vacuous) and
        that both returned values are correct."""
        from torch_spyre._inductor.pass_utils import op_short_name

        x = self.rand_device((64, 1024))

        def fn(x):
            y = x * 2.0  # returned AND read by z, w, v -> pinned + cloned to HBM
            z = y * 3.0
            w = y + z
            v = y + w  # y's last internal read; v is realized (read by u)
            u = v + 1.0
            return y, u

        output_feeders: set[str] = set()

        def collect_feeders(graph: GraphLowering) -> None:
            by_name = {op.name: op for op in graph.operations}
            for name in graph.get_output_names():
                output_feeders.add(name)
                op = by_name.get(name)
                if op is not None and op_short_name(op) == "clone":
                    output_feeders.update(d.name for d in op.get_read_writes().reads)

        with self.pre_scheduling_iterating_pass(collect_feeders):
            with self._capture_solve_graphs() as captured:
                # Over the committed divisions (see the input-clone test above).
                with ts_inductor_config.patch(
                    lx_planning=True, co_optimizing_lx_planning=False
                ):
                    compiled = torch.compile(fn, fullgraph=True)
                    ry, ru = compiled(x)
                    ry, ru = ry.to("cpu"), ru.to("cpu")

        # The hazard is real only if a returned (output-feeding) buffer is actually
        # named as an in-place parent by some consumer.
        reused = any(
            parents & output_feeders
            for solve_graph in captured
            for parents in self._in_place_parents(solve_graph).values()
        )
        self.assertTrue(
            reused,
            "expected the returned buffer's slot to be reused in place (hazard not "
            "exercised); adjust the graph so the aliasing case is actually tested",
        )
        ref_y, ref_u = fn(x.to("cpu"))
        self.assertTrue(
            torch.allclose(ref_y, ry, atol=1e-2, rtol=1e-3),
            "returned buffer y was corrupted by in-place reuse of its slot",
        )
        self.assertTrue(
            torch.allclose(ref_u, ru, atol=1e-2, rtol=1e-3), "output u changed"
        )

    @unittest.skipUnless(_HAS_ORTOOLS, "co-optimizing path needs ortools")
    def test_returned_buffer_reused_in_place_correct_in_cooptimizing_path(self):
        """Same aliasing hazard as the sibling test, exercised on the co-optimizing
        (joint CP-SAT) path: a returned buffer whose LX slot is reused in place must
        still be handed back to the caller intact (#3212)."""
        from torch_spyre._inductor.pass_utils import op_short_name

        x = self.rand_device((64, 1024))

        def fn(x):
            y = x * 2.0  # returned AND read by z, w, v -> pinned + cloned to HBM
            z = y * 3.0
            w = y + z
            v = y + w  # y's last internal read; v is realized (read by u)
            u = v + 1.0
            return y, u

        output_feeders: set[str] = set()

        def collect_feeders(graph: GraphLowering) -> None:
            by_name = {op.name: op for op in graph.operations}
            for name in graph.get_output_names():
                output_feeders.add(name)
                op = by_name.get(name)
                if op is not None and op_short_name(op) == "clone":
                    output_feeders.update(d.name for d in op.get_read_writes().reads)

        with self.pre_scheduling_iterating_pass(collect_feeders):
            with self._capture_solve_graphs() as captured:
                with ts_inductor_config.patch(
                    lx_planning=True,
                    co_optimizing_lx_planning=True,
                    _cpsat_warn_on_cost_expr=False,
                ):
                    compiled = torch.compile(fn, fullgraph=True)
                    ry, ru = compiled(x)
                    ry, ru = ry.to("cpu"), ru.to("cpu")

        reused = any(
            parents & output_feeders
            for solve_graph in captured
            for parents in self._in_place_parents(solve_graph).values()
        )
        self.assertTrue(
            reused,
            "expected the returned buffer's slot to be reused in place on the "
            "co-optimizing path (hazard not exercised)",
        )
        ref_y, ref_u = fn(x.to("cpu"))
        self.assertTrue(
            torch.allclose(ref_y, ry, atol=1e-2, rtol=1e-3),
            "returned buffer y was corrupted by in-place reuse (co-opt path)",
        )
        self.assertTrue(
            torch.allclose(ref_u, ru, atol=1e-2, rtol=1e-3), "output u changed"
        )

    def _assert_input_clone_shares_lx_slot(self, co_optimizing: bool) -> None:
        """Peak-LX: the input clone reuses a slot rather than adding one (#3212).

        The physical input clone shares its LX address with the consumer that
        reuses it in place, so it does not occupy a dedicated slot. A 2-slot LX
        budget cannot hold the three LX-eligible buffers unmerged, so the solve
        must take the merge. We read the final LX allocations after the
        allocator runs and assert the input clone's address is shared by
        another LX buffer (and values are correct)."""
        from torch_spyre._inductor.pass_utils import op_short_name
        from torch_spyre._inductor.scratchpad import allocator as alloc_mod

        sencores = 32
        x = self.rand_device((64, 1024))
        # Per-core footprint of each buffer at the 32-way split (fp16 -> 2 bytes),
        # already 128-byte aligned.
        per_core_bytes = 64 * 1024 * 2 // sencores
        lx_budget = 2 * per_core_bytes

        def fn(x):
            return x * 2.0 + x * 3.0

        input_names: set[str] = set()
        per_op: dict[str, dict] = {}

        def visit(graph: GraphLowering) -> None:
            input_names.update(graph.graph_input_names)
            for op in graph.operations:
                alloc = getattr(
                    graph.get_buffer(op.name).get_layout(), "allocation", {}
                )
                per_op[op.name] = {
                    "short": op_short_name(op),
                    "lx": alloc.get("lx"),
                    "reads": [d.name for d in op.get_read_writes().reads],
                }

        with self.pre_scheduling_iterating_pass(visit):
            with (
                patch.object(alloc_mod, "_lx_planning_size", lambda: lx_budget),
                _recording_division_modes() as modes,
                ts_inductor_config.patch(
                    lx_planning=True,
                    co_optimizing_lx_planning=co_optimizing,
                    sencores=sencores,
                ),
            ):
                result = torch.compile(fn, fullgraph=True)(x).to("cpu")

        # A failed open solve replans over the committed divisions, which is
        # the sibling case; without this the co-optimizing case could pass on
        # the replan.
        self.assertEqual(modes, ["open" if co_optimizing else "committed"])

        # Group LX-resident buffers by address; a shared address == in-place reuse.
        addr_to_buffers: dict[int, list[str]] = {}
        for name, info in per_op.items():
            if info["lx"] is not None:
                addr_to_buffers.setdefault(info["lx"], []).append(name)

        # The physical input clone: a clone op reading a graph input, LX-resident.
        input_clones = [
            name
            for name, info in per_op.items()
            if info["short"] == "clone"
            and info["lx"] is not None
            and any(r in input_names for r in info["reads"])
        ]
        self.assertTrue(input_clones, "expected an LX-resident clone of a graph input")
        self.assertTrue(
            any(len(addr_to_buffers[per_op[c]["lx"]]) > 1 for c in input_clones),
            "input clone occupies a dedicated LX slot -- expected the solver to "
            "reuse it in place under LX pressure (no peak-LX reduction)",
        )
        self.assertTrue(
            torch.allclose(fn(x.to("cpu")), result, atol=1e-2, rtol=1e-3),
            "input clone slot sharing changed the numerical result",
        )

    @unittest.skipUnless(_HAS_ORTOOLS, "the cpsat solver needs ortools")
    def test_input_clone_inplace_shares_lx_slot(self):
        self._assert_input_clone_shares_lx_slot(co_optimizing=False)

    @unittest.skipUnless(_HAS_ORTOOLS, "co-optimizing path needs ortools")
    def test_input_clone_inplace_shares_lx_slot_in_cooptimizing_path(self):
        """The co-optimizing path prices input cloning and in-place operations
        (PR4596)."""
        self._assert_input_clone_shares_lx_slot(co_optimizing=True)


if __name__ == "__main__":
    unittest.main()
