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
from typing import Callable, TypeVarTuple, Unpack, Optional, override

import unittest
from unittest.mock import patch
import torch

from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor import passes
from torch_spyre._inductor import config as ts_inductor_config

try:
    from ortools.sat.python import cp_model  # noqa: F401

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False


Ts = TypeVarTuple("Ts")

# One buffer's entry in an allocation fingerprint (keyed by buffer name):
#   (location, size_bytes, (output_splits, reduction_splits))
# where each split list is a sorted tuple of (iteration_space_stride, factor).
_Splits = tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
_AllocEntry = tuple[str, int, _Splits]


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

    A class may also define a static ``case_decorators(params) -> list`` hook.
    Each decorator it returns is applied to the generated method for that combo
    (e.g. mark the ``cpsat`` combos ``expectedFailure``). Absent hook -> no
    per-case decoration.

    Generating methods rather than sibling classes keeps everything in the
    ``attrs`` dict handed to ``__new__`` — no module-namespace or ``sys.modules``
    access, so it is immune to the OOT runner's out-of-``sys.modules``
    pre-import.
    """

    # How each axis renders into the test-id suffix. Axes not listed fall back to
    # "<name><value>"; this keeps the curated short labels while letting new axes
    # added to ``parameter_axes`` work without editing this method.
    _AXIS_LABELS = {
        "solver_method": lambda v: str(v),
        "sencores": lambda v: f"sc{v}",
        "co_optimization": lambda v: "coopt" if v else "nocoopt",
        "boundary_clones": lambda v: "clones" if v else "noclones",
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
            # Optional per-combo decorator hook (e.g. mark cpsat as expectedFailure).
            decorators_for = attrs.get("case_decorators")
            if isinstance(decorators_for, staticmethod):
                decorators_for = decorators_for.__func__
            for params in combos:
                suffix = mcs._combo_suffix(params)
                for label, factory in models:
                    test_name = f"test_{label}__{suffix}" if suffix else f"test_{label}"
                    test_method = mcs._make_case(params, factory)
                    if decorators_for is not None:
                        for dec in decorators_for(params):
                            test_method = dec(test_method)
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
    ``test_<model>__<solver>_sc<n>_<coopt>_<clones>`` method for every model in
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
        "solver_method": ("greedy", "bestfit", "firstfit", "cpsat"),
        "sencores": (1, 32),
        "co_optimization": (False, True)
        if ts_inductor_config.co_optimizing_lx_planning
        else (False,),
        "boundary_clones": (False, True),
    }

    parameter_models = (("softmax", _softmax_case), ("mlp", _mlp_case))

    def run_case(self, params: dict, factory: Callable) -> None:
        """Run ``factory``'s model for correctness under this combo, applying
        the combo's config at the test-case level (the inherited setUp only
        applies invariants)."""
        with ts_inductor_config.patch(
            layout_solver=params["solver_method"],
            sencores=params["sencores"],
            co_optimizing_lx_planning=params["co_optimization"],
            lx_boundary_clones=params["boundary_clones"],
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
    HBM by default — Strategy B reconciles them and pins all 4.
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
        ScratchpadAllocator only pins 1 of 4 shared buffers; Strategy B should
        flip the pointwise ops to cols and pin all 4 → strictly lower HBM."""
        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        self.run_test(f, (x,), strict=True)

    def test_softmax_dim_neg1_no_regression(self):
        """softmax(dim=-1) is the well-behaved baseline where ScratchpadAllocator
        already pins everything pinnable. Strategy B must match (no regression)."""
        f = functools.partial(torch.softmax, dim=-1)
        x = self.rand_device((512, 1024))
        self.run_test(f, (x,))


class TestCloneAtGraphBoundaries(BaseTestScratchpadUsage):
    """End-to-end tests for clone insertion at graph input/output boundaries.

    The allocator now inserts clone ops on-demand inside _push_allocation rather than
    as a separate pre-scheduling pass.  These tests verify that:
    - graph inputs read by multiple ops get a clone that lands in LX
    - graph outputs that are also read inside the graph get a clone (for the HBM return
      value), while the original buffer is pinned to LX

    Enabling ``lx_boundary_clones`` flips ``clone_at_graph_boundaries()`` on and
    makes the inserted clone outputs LX-eligible, so the boundary clone path is
    exercised. This class applies that patch itself at the test-case level (in
    ``_compile_and_inspect``), so every compile here runs with it on.
    """

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
            with ts_inductor_config.patch(lx_boundary_clones=True):
                compiled_kernel = torch.compile(f, fullgraph=True)
                raw = compiled_kernel(*args)
            if isinstance(raw, tuple):
                result = tuple(r.to("cpu") for r in raw)
            else:
                result = raw.to("cpu")

        n_ops = n_ops_captured[0] if n_ops_captured else 0
        return result, n_ops, mem_usages

    def test_input_clone_when_read_by_multiple_ops(self):
        """A graph input read by two different ops is cloned; the clone lands in LX."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # x is consumed by both exp_op and add_op → two reads → eligible for input clone
            return torch.exp(x) + x

        with ts_inductor_config.patch(lx_planning=False):
            ref_result, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, n_ops_with_lx, mem_usages = self._compile_and_inspect(fn, (x,))

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            f"Expected the input clone to add an op: {n_ops_no_lx} ops without LX, "
            f"{n_ops_with_lx} with LX",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer after input cloning",
        )
        # Clone is an exact copy; LX planning must not change the numerical result.
        self.assertTrue(
            torch.equal(ref_result, result),
            "LX input clone changed the numerical result",
        )

    # TODO: Update to expected pass on PR2907
    @unittest.skipUnless(
        _HAS_ORTOOLS, "joint CP-SAT boundary-clone xfail needs ortools"
    )
    @unittest.expectedFailure
    def test_input_clone_unsupported_under_joint_cpsat(self):
        """Boundary clone insertion is not yet supported by the joint CP-SAT
        allocator (``layout_solver="cpsat"`` with ``co_optimizing_lx_planning``).

        The joint allocator builds CoreDivisionBuffers only for graph *ops*:
        graph inputs are never candidates (so an input read by multiple ops is
        never cloned into LX) and graph outputs carry a "graph output" residency
        reason (never pinned, so no output clone). The op count therefore does
        not grow. This is an *expected failure* until the CP-SAT path grows
        boundary-clone support; the greedy and placement-only CP-SAT paths handle
        it (see ``test_input_clone_when_read_by_multiple_ops``).

        Skipped without ortools: the allocator would then fall back to greedy,
        which *does* clone, making the assertion pass (an unexpected success).
        """
        x = self.rand_device((64, 1024))

        def fn(x):
            # x read by both exp and add -> eligible for an input clone.
            return torch.exp(x) + x

        with ts_inductor_config.patch(lx_planning=False):
            _, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(
            lx_planning=True,
            layout_solver="cpsat",
            co_optimizing_lx_planning=True,
        ):
            _, n_ops_with_lx, _ = self._compile_and_inspect(fn, (x,))

        # Holds on the greedy / placement-only paths (see the sibling test); the
        # joint CP-SAT path inserts no boundary clone -> fails -> expected failure.
        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            "joint CP-SAT unexpectedly inserted a boundary clone "
            f"({n_ops_no_lx} -> {n_ops_with_lx} ops)",
        )

    # TODO: Update to expected pass on PR2907
    @unittest.skipUnless(
        _HAS_ORTOOLS, "joint CP-SAT boundary-clone xfail needs ortools"
    )
    @unittest.expectedFailure
    def test_output_clone_unsupported_under_joint_cpsat(self):
        """Output boundary clone insertion is not yet supported by the joint
        CP-SAT allocator (``layout_solver="cpsat"`` with
        ``co_optimizing_lx_planning``).

        Output-side counterpart to
        ``test_input_clone_unsupported_under_joint_cpsat``. The joint allocator
        builds CoreDivisionBuffers only for graph *ops* and tags graph outputs
        with a "graph output" residency reason, so a buffer that is both a graph
        output and read inside the graph is never pinned and no output clone is
        inserted. The op count therefore does not grow. This is an *expected
        failure* until the CP-SAT path grows boundary-clone support; the greedy
        and placement-only CP-SAT paths handle it (see
        ``test_output_clone_when_intermediate_is_also_graph_output``).

        Skipped without ortools: the allocator would then fall back to greedy,
        which *does* clone, making the assertion pass (an unexpected success).
        """
        x = self.rand_device((64, 1024))

        def fn(x):
            # y = exp(x) is both a graph output and read by add_op -> eligible
            # for an output clone on the greedy / placement-only paths.
            y = torch.exp(x)
            z = y + 1
            return y, z

        with ts_inductor_config.patch(lx_planning=False):
            _, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(
            lx_planning=True,
            layout_solver="cpsat",
            co_optimizing_lx_planning=True,
        ):
            _, n_ops_with_lx, _ = self._compile_and_inspect(fn, (x,))

        # Holds on the greedy / placement-only paths (see the sibling test); the
        # joint CP-SAT path inserts no boundary clone -> fails -> expected failure.
        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            "joint CP-SAT unexpectedly inserted a boundary clone "
            f"({n_ops_no_lx} -> {n_ops_with_lx} ops)",
        )

    def test_output_clone_when_intermediate_is_also_graph_output(self):
        """A buffer that is both a graph output and read inside the graph is pinned to LX;
        a clone of it is inserted as the actual (HBM) graph output returned to the caller."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # After CSE, y = exp(x) is produced once.
            # y is a graph output AND is read by add_op → eligible for output clone.
            y = torch.exp(x)
            z = y + 1  # add_op reads y
            return y, z

        with ts_inductor_config.patch(lx_planning=False):
            (ref_y, ref_z), n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            (result_y, result_z), n_ops_with_lx, mem_usages = self._compile_and_inspect(
                fn, (x,)
            )

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            f"Expected the output clone to add an op: {n_ops_no_lx} ops without LX, "
            f"{n_ops_with_lx} with LX",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer after output cloning",
        )
        # Clone is an exact copy; LX planning must not change the numerical result.
        self.assertTrue(
            torch.equal(ref_y, result_y), "LX output clone changed result y"
        )
        self.assertTrue(
            torch.equal(ref_z, result_z), "LX output clone changed result z"
        )

    def test_input_read_at_multiple_offsets_is_correct(self):
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

        with ts_inductor_config.patch(lx_planning=False):
            ref, _, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, _, _ = self._compile_and_inspect(fn, (x,))

        self.assertTrue(
            torch.equal(ref, result),
            "Multi-offset input read produced wrong values under LX planning",
        )

    def test_input_feeding_reduction_is_cloned_and_correct(self):
        """A graph input read by a reduction is LX-cloned, with the clone's
        per-core split re-keyed correctly.

        push_allocation_with_clone re-keys the consumer's op_it_space_splits
        through the buffer's strides before assigning them to the clone. A reduction consumer's split is keyed to its
        reduced-shape output; copied verbatim it would split the wrong axis of
        the full-shape clone (wrong values / SDSC abort at multi-core). The
        numerical failure only manifests when work is split across cores; here
        (sencores=1) we assert the clone is inserted and the result is correct.
        Multi-core numerical coverage lives in
        tests/inductor/test_inductor_ops.py (max_sub_broadcast, aminmax,
        softmax)."""
        x = self.rand_device((64, 256))

        def fn(x):
            # x feeds the max reduction (and the sub) -> reduction consumer.
            return x - torch.unsqueeze(torch.max(x, dim=1).values, dim=1)

        with ts_inductor_config.patch(lx_planning=False):
            ref, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, n_ops_with_lx, mem_usages = self._compile_and_inspect(fn, (x,))

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            "Expected a boundary clone for the reduction-fed input, but the op "
            f"count did not grow ({n_ops_no_lx} -> {n_ops_with_lx})",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer for the reduction input",
        )
        self.assertTrue(
            torch.equal(ref, result),
            "Reduction-fed input changed result under LX planning",
        )

    def test_input_read_partially_is_correct(self):
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

        with ts_inductor_config.patch(lx_planning=False):
            ref, _, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, _, _ = self._compile_and_inspect(fn, (x,))

        self.assertTrue(
            torch.equal(ref, result),
            "Partial input read produced wrong values under LX planning",
        )


# TODO: Remove hard coded core division. This test exists to check for
# regressions when operating on matmuls. There is likely a better
# approach where we use a proxy to estimate the runtime perforamance
# of given allocations.
@unittest.skipUnless(
    ts_inductor_config.co_optimizing_lx_planning, "co-optimization is not enabled"
)
class CoOptAllocatorIntegrationTests(BaseTestScratchpadUsage):
    """Generic real-graph coverage for the co-optimising allocator.

    ``StrategyBCoOptimizingAllocator`` (``co_optimizing_lx_planning=True``) seeds
    from the core-division work-distribution, commits the winning splits onto
    ``op_it_space_splits``, then places buffers. These tests put real compiled
    graphs through that path.

    The prescribed-allocation tests encode the *desired* plan, which is the one
    StrategyB produces. These plans are brittle and are not unique but are
    plans which achieve desirable performance. New plans should be profiled
    before making these test more permissive.

    NOTE: this suite is intentionally *disabled* today. Unlike
    ``ParameterizedScratchpadUsage`` / ``TestCpSatAllocatorFallback`` it does not
    set ``metaclass=_ParameterizedScratchpadMeta``, so no ``test_*`` methods are
    generated and nothing is collected -- the co-optimization compiles are too
    slow to run on every CI job. The ``parameter_axes`` / ``parameter_models`` /
    ``case_decorators`` / ``run_case`` machinery below is ready; re-enable the
    suite by attaching the metaclass once ``cpsat`` becomes the default
    ``layout_solver``. (This omission is deliberate, not a dropped metaclass.)

    When enabled: the acceptance criterion for each model (its prescribed
    fingerprint) is defined *once* in that model's factory and swept over the
    ``solver_method`` axis by ``_ParameterizedScratchpadMeta``. The prescribed
    plans are the *greedy* StrategyB plans; the joint CP-SAT allocator
    (``layout_solver="cpsat"``) optimises core division and placement jointly
    and is expected to land on a different (not yet pinned-down) plan, so the
    ``cpsat`` combos are marked ``expectedFailure`` via ``case_decorators``.
    They guard against the CP-SAT path silently regressing to the greedy plan;
    when CP-SAT's plans are profiled and stabilised, give ``cpsat`` its own
    prescribed fingerprints.

    It extends :class:`BaseTestScratchpadUsage` for the shared helpers
    (``rand_device``, ``_simple_mlp``, ``pre_scheduling_iterating_pass`` and the
    pre-scheduling-pass setup) and applies its own config patches at the
    test-case level in ``_allocation_fingerprint`` (sencores=32 /
    co-optimization / boundary clones); the solver is the swept axis.
    """

    def _allocation_fingerprint(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        layout_solver: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, _AllocEntry]]:
        """Compile ``model`` through the allocator (sencores=32, lx_planning on)
        and return ``(cpu_result, device_result, fingerprint)``.

        The fingerprint maps each op's buffer name to its
        ``(location, size_bytes, split)``. ``location`` is "LX"/"HBM"; ``split``
        is the committed core division as
        ``((output_splits...), (reduction_splits...))``, each a sorted tuple of
        ``(iteration_space_stride, factor)`` pairs. We keep the full per-axis
        split rather than the core-count product so that, e.g., a 32-way split
        of one axis (``((1024, 32),)``) is distinguished from an 8x4 split across
        two axes (``((1, 8), (1024, 4))``) even though both use 32 cores. The
        buffer names and the values are both deterministic run-to-run, so the
        per-buffer allocation can be prescribed exactly.
        """
        cpu_result = model(*(t.to("cpu") for t in args))

        fingerprint: dict[str, _AllocEntry] = {}

        def visitor(graph: GraphLowering) -> None:
            fingerprint.clear()
            for op in graph.operations:
                layout = graph.get_buffer(op.name).get_layout()
                device_layout = layout.device_layout
                allocation = getattr(layout, "allocation", {})
                out, red = getattr(op, "op_it_space_splits", ({}, {}))
                split = (tuple(sorted(out.items())), tuple(sorted(red.items())))
                fingerprint[op.name] = (
                    "LX" if "lx" in allocation else "HBM",
                    math.prod(device_layout.device_size[:-1]) * 128,
                    split,
                )

        with self.pre_scheduling_iterating_pass(visitor):
            with ts_inductor_config.patch(
                layout_solver=layout_solver,
                sencores=32,
                co_optimizing_lx_planning=True,
                lx_boundary_clones=True,
            ):
                compiled = torch.compile(model, fullgraph=True)
                device_result = compiled(*args).to("cpu")

        return cpu_result, device_result, fingerprint

    def _assert_prescribed_allocation(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        expected: dict[str, _AllocEntry],
        layout_solver: str,
        atol: float = 0.1,
        rtol: float = 0.1,
    ) -> None:
        cpu_result, device_result, fingerprint = self._allocation_fingerprint(
            model, args, layout_solver=layout_solver
        )
        self.assertEqual(
            fingerprint,
            expected,
            "allocation does not match the prescribed (desired = StrategyB) plan "
            "{buf: (location, size, split)}:\n"
            f"  expected {expected}\n  got      {fingerprint}",
        )
        torch.testing.assert_close(
            device_result,
            cpu_result,
            atol=atol,
            rtol=rtol,
            msg="prescribed-allocation result diverged from CPU",
        )

    # Model factories. Each returns ``(model, args, kwargs)`` where ``kwargs``
    # carries the acceptance criterion for that model — its prescribed
    # fingerprint (``expected``) plus optional tolerances — defined once and
    # reused across every solver in ``parameter_axes``.
    def _softmax_case(self):
        """softmax(dim=0) over (512, 1024). The desired plan keeps only the
        ``exp`` intermediate (buf1) resident in LX; the two reductions (buf0=max,
        buf3=sum) and the normalised bodies (buf2, buf4) spill to HBM. The
        reductions take a 16-way split of the stride-1 (column) axis with a 2-way
        split of the reduced axis (``((1, 16),), ((1024, 2),)``); the pointwise
        ops take a full 32-way split of the stride-1024 axis (``((1024, 32),)``).
        """
        return (
            functools.partial(torch.softmax, dim=0),
            (self.rand_device((512, 1024)),),
            {
                "expected": {
                    "buf0": ("HBM", 2048, (((1, 16),), ((1024, 2),))),
                    "buf1": ("LX", 1048576, (((1024, 32),), ())),
                    "buf2": ("HBM", 1048576, (((1024, 32),), ())),
                    "buf3": ("HBM", 2048, (((1, 16),), ((1024, 2),))),
                    "buf4": ("HBM", 1048576, (((1024, 32),), ())),
                }
            },
        )

    def _mlp_case(self):
        """Two-layer linear MLP (``nn.Linear -> silu -> nn.Linear``). With every
        op LX-eligible, the plan keeps two of the three hidden-width activations
        resident (buf0, buf1); the third hidden-width buffer (buf2), the two
        output-width buffers (buf3, buf4) and the two Linear weight buffers
        (buf5, buf6) spill to HBM. The resident hidden-width ops take an 8x4
        split across two axes (``((1, 8), (1024, 4))``).
        """
        model, args = self._simple_mlp()
        return (
            model,
            args,
            {
                "expected": {
                    "buf0": ("LX", 262144, (((1, 8), (1024, 4)), ())),
                    "buf1": ("LX", 262144, (((1, 8), (1024, 4)), ())),
                    "buf2": ("HBM", 262144, (((1, 8), (1024, 4)), ())),
                    "buf3": ("HBM", 65536, (((1, 2), (256, 8)), ((1, 2),))),
                    "buf4": ("HBM", 65536, (((256, 32),), ())),
                    "buf5": ("HBM", 524288, (((1, 2), (256, 16)), ())),
                    "buf6": ("HBM", 524288, (((1, 16), (1024, 2)), ())),
                }
            },
        )

    def _sdpa_case(self):
        """4D scaled-dot-product attention. With every op LX-eligible, the plan
        keeps most of the matmul -> softmax -> matmul chain resident (buf0,
        buf2-buf8, all 32-way split); one matmul output (buf1), a normalised
        output (buf9), the final result (buf12) and the empty constant of the
        decomposition (buf10) land in HBM. The resident ops take single-axis
        32-way splits; buf12 takes a 4x4 two-axis split
        (``((64, 4), (16384, 4))``) and the empty constant is undivided.
        """
        batch, heads, seq_len, head_dim = 1, 4, 256, 64
        return (
            torch.nn.functional.scaled_dot_product_attention,
            (
                self.rand_device((batch, heads, seq_len, head_dim)),
                self.rand_device((batch, heads, seq_len, head_dim)),
                self.rand_device((batch, heads, seq_len, head_dim)),
            ),
            {
                "expected": {
                    "buf0": ("LX", 131072, (((64, 32),), ())),
                    "buf1": ("HBM", 131072, (((64, 32),), ())),
                    "buf2": ("LX", 524288, (((256, 32),), ())),
                    "buf3": ("LX", 131072, (((1, 32),), ())),
                    "buf4": ("LX", 524288, (((256, 32),), ())),
                    "buf5": ("LX", 524288, (((256, 32),), ())),
                    "buf6": ("LX", 131072, (((1, 32),), ())),
                    "buf7": ("LX", 524288, (((256, 32),), ())),
                    "buf8": ("LX", 131072, (((64, 32),), ())),
                    "buf9": ("HBM", 131072, (((256, 32),), ())),
                    "buf10": ("HBM", 128, ((), ())),
                    # buf11 is eliminated in dedup_and_promote_constants
                    "buf12": ("HBM", 131072, (((64, 4), (16384, 4)), ())),
                }
            },
        )

    def _swiglu_case(self):
        """A single SwiGLU layer: two parallel ``nn.Linear`` projections (each a
        ``mm`` GEMM plus a bias add) feeding ``silu(gate) * up``. The lowered
        graph is eight buffers:

          - buf6, buf7 -- restickified gate/up weights
          - buf0 -- gate GEMM (``mm``, a Reduction), the input to SiLU
          - buf1 -- gate + bias
          - buf2 -- ``silu(buf1)``
          - buf3 -- up GEMM (``mm``, a Reduction)
          - buf4 -- up + bias
          - buf5 -- ``buf2 * buf4``, the layer output

        With every op LX-eligible, the plan keeps the whole hidden-width chain
        resident in LX (buf0-buf4), each taking an 8x4 split across two axes
        (``((1, 8), (1024, 4))``); the output (buf5) spills to HBM. The two
        weight buffers (buf6, buf7) spill to HBM with a 2x16 split
        (``((1, 2), (256, 16))``).

        The shared input ``x`` is *not* LX-pinned: both GEMMs split it 8-way
        along their free (N) dimension, which ``x`` does not have, so each
        consumer's per-core view of ``x`` (a 4-way split of the shared M axis)
        covers fewer cores than the GEMM runs. That is a broadcast read of a
        per-core scratchpad buffer, which the single-base LX path cannot serve,
        so the broadcast-read guard in ``get_ncores_for_buffers`` keeps ``x`` in
        HBM.
        """
        model, args = self._swiglu()
        return (
            model,
            args,
            {
                "expected": {
                    "buf6": ("HBM", 524288, (((1, 2), (256, 16)), ())),
                    "buf0": ("LX", 262144, (((1, 8), (1024, 4)), ())),
                    "buf1": ("LX", 262144, (((1, 8), (1024, 4)), ())),
                    "buf2": ("LX", 262144, (((1, 8), (1024, 4)), ())),
                    "buf7": ("HBM", 524288, (((1, 2), (256, 16)), ())),
                    "buf3": ("LX", 262144, (((1, 8), (1024, 4)), ())),
                    "buf4": ("LX", 262144, (((1, 8), (1024, 4)), ())),
                    "buf5": ("HBM", 262144, (((1, 8), (1024, 4)), ())),
                }
            },
        )

    parameter_axes = {"solver_method": ("greedy", "cpsat")}
    parameter_models = (
        ("softmax_prescribed_allocation", _softmax_case),
        ("mlp_prescribed_allocation", _mlp_case),
        ("sdpa_prescribed_allocation", _sdpa_case),
        ("swiglu_prescribed_allocation", _swiglu_case),
    )

    # TODO: Update this when we have matching alloctions with CP-SAT or equally optimal plans
    @staticmethod
    def case_decorators(params):
        """The greedy plans are prescribed exactly; CP-SAT is expected to differ,
        so mark its combos ``expectedFailure`` (and skip when ortools is absent
        since the joint CP-SAT path needs it)."""
        if params["solver_method"] == "cpsat":
            return [
                unittest.expectedFailure,
                unittest.skipUnless(
                    _HAS_ORTOOLS, "joint CP-SAT prescribed xfail needs ortools"
                ),
            ]
        return []

    def run_case(self, params: dict, factory: Callable) -> None:
        """Compile the factory's model through the co-optimising allocator on the
        combo's solver and assert it matches the model's prescribed fingerprint."""
        model, args, kwargs = factory(self)
        self._assert_prescribed_allocation(
            model,
            args,
            kwargs["expected"],
            layout_solver=params["solver_method"],
            atol=kwargs.get("atol", 0.1),
            rtol=kwargs.get("rtol", 0.1),
        )


class TestIntermediatePartialReadNotPinned(BaseTestScratchpadUsage):
    """An *intermediate* buffer read partially (sliced) must not be LX-pinned.

    Companion to ``TestCloneAtGraphBoundaries``, which guards graph
    input/output clones. ``_filter_ops`` applies the same
    ``buffer_not_read_in_full`` guard to intermediate buffers: a buffer that is
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
            msg="sliced intermediate miscompiled — is the _filter_ops guard present?",
        )


class TestCpSatAllocatorFallback(
    BaseTestScratchpadUsage, metaclass=_ParameterizedScratchpadMeta
):
    """CP-SAT gracefully degrades to greedy placement when ortools is absent.

    Forces the missing-ortools condition (``cp_model = None``) and drives the
    ``layout_solver="cpsat"`` path over the model sweep, in both the joint
    (``co_optimization=True``) and placement-only (``co_optimization=False``)
    routings. In every combination the compile must succeed and LX planning must
    still reduce HBM traffic (the greedy fallback is correct, just not
    CP-SAT-optimal). The metaclass injects one ``test_<model>__<combo>`` method
    per ``(model, config-combo)``.
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
        "solver_method": ("cpsat",),
        "sencores": (32,),
        "co_optimization": (False, True),
    }

    parameter_models = (("softmax", _softmax_case), ("mlp", _mlp_case))

    @contextmanager
    def _ortools_absent(self):
        """Force the missing-ortools condition: CpSatLayoutSolver.__init__ raises
        ImportError (so the allocator falls back) exactly when cp_model is None,
        which is how a real missing install presents."""
        from torch_spyre._inductor.scratchpad import ilp_solver_ortools

        saved = ilp_solver_ortools.cp_model
        ilp_solver_ortools.cp_model = None
        try:
            yield
        finally:
            ilp_solver_ortools.cp_model = saved

    def run_case(self, params: dict, factory: Callable) -> None:
        """Run ``factory``'s model for correctness under this combo, applying
        the combo's config at the test-case level (the inherited setUp only
        applies invariants)."""
        with self._ortools_absent():
            with ts_inductor_config.patch(
                layout_solver=params["solver_method"],
                sencores=params["sencores"],
                co_optimizing_lx_planning=params["co_optimization"],
            ):
                model, args, kwargs = factory(self)
                torch.compiler.reset()
                with ts_inductor_config.patch(lx_planning=False):
                    result_without_lx, hbm_without_lx = self.measure_hbm_transfers(
                        model, args
                    )
                torch.compiler.reset()
                with ts_inductor_config.patch(lx_planning=True):
                    result_with_lx, hbm_with_lx = self.measure_hbm_transfers(
                        model, args
                    )

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


class TestSelectAllocator(unittest.TestCase):
    """select_allocator maps config -> (allocator, solver) so the allocators
    never inspect config themselves. Pure dispatch, no device needed."""

    def test_dispatch_by_config(self):
        from torch_spyre._inductor.scratchpad.allocator import (
            CoOptimizingAllocator,
            ScratchpadAllocator,
            StrategyBCoOptimizingAllocator,
            select_allocator,
        )
        from torch_spyre._inductor.scratchpad.plan_solver import GreedyLayoutSolver
        from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
            BestFitLayoutSolver,
        )

        with ts_inductor_config.patch(
            layout_solver="greedy", co_optimizing_lx_planning=False
        ):
            a = select_allocator()
            self.assertIs(type(a), ScratchpadAllocator)
            self.assertIsInstance(a.layout_planning, GreedyLayoutSolver)

        with ts_inductor_config.patch(
            layout_solver="bestfit", co_optimizing_lx_planning=False
        ):
            a = select_allocator()
            self.assertIs(type(a), ScratchpadAllocator)
            self.assertIsInstance(a.layout_planning, BestFitLayoutSolver)

        with ts_inductor_config.patch(
            layout_solver="greedy", co_optimizing_lx_planning=True
        ):
            self.assertIsInstance(select_allocator(), StrategyBCoOptimizingAllocator)

        # cpsat + co-optimization routes to the joint allocator.
        with ts_inductor_config.patch(
            layout_solver="cpsat", co_optimizing_lx_planning=True
        ):
            self.assertIsInstance(select_allocator(), CoOptimizingAllocator)

        # cpsat without co-optimization is placement-only: a ScratchpadAllocator
        # driven by the CP-SAT solver on the pre-determined core divisions.
        with ts_inductor_config.patch(
            layout_solver="cpsat", co_optimizing_lx_planning=False
        ):
            a = select_allocator()
            self.assertIs(type(a), ScratchpadAllocator)
            if _HAS_ORTOOLS:
                from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
                    CpSatLayoutSolver,
                )

                self.assertIsInstance(a.layout_planning, CpSatLayoutSolver)
            else:
                # ortools absent: falls back to greedy placement.
                self.assertIsInstance(a.layout_planning, GreedyLayoutSolver)

        with ts_inductor_config.patch(
            layout_solver="bogus", co_optimizing_lx_planning=False
        ):
            with self.assertRaises(ValueError):
                select_allocator()


if __name__ == "__main__":
    unittest.main()
