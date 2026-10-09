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

"""Automated coarse tiling: explicit ``for_each_tile`` loops and tile discovery."""

import contextlib
import dataclasses
import functools
import pytest
import os
import sys
import torch
import unittest

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Optional

import sympy
from unittest.mock import MagicMock, patch

from torch._inductor import config as t_inductor_config
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise, Reduction
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.loop_info import CoarseTileInfo
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor.scratchpad.allocator import (
    CoOptimizingAllocator,
    _spec_within_read_distance,
    commit_lx_views,
    select_allocator,
)
from torch_spyre._inductor.scratchpad.coarse_tiling import (
    splits_on_tile,
)
from torch_spyre._inductor.scratchpad.lx_relayout import LXRelayoutPlan
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    RelayoutCopyBuffer,
    TileAxis,
    TileSpec,
    relayout_copy_name,
)
from torch_spyre._inductor.scratchpad.utils import get_ncores_for_buffers
from torch_spyre._inductor.wsr import for_each_tile

sys.path.insert(0, os.path.dirname(__file__))
from test_scratchpad_use import _ParameterizedScratchpadMeta  # noqa: E402

try:
    from ortools.sat.python import cp_model  # noqa: F401

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False


def expected_unimplemented(fn):
    """Expect a test to fail *only* by reaching an unbuilt part of the feature.

    ``unittest.expectedFailure`` absorbs any exception, so a test written
    against a gate that does not exist yet would be satisfied by the resulting
    ``AttributeError`` -- and would stay satisfied after the feature landed
    wrong.  This narrows the expectation to one declared cause and fails the
    test on anything else, including a clean pass (the signal to delete the
    marker).

    Because it is imperative rather than a pytest mark, ``-m 'not xfail'`` does
    not deselect these; they still run and still xfail at runtime.

    Nothing here is specific to coarse tiling; it belongs in
    ``utils_inductor.py`` once a second suite wants it.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as exc:
            pytest.xfail(f"not built yet: {exc}")
        else:
            self.fail(f"{fn.__name__} passed -- remove @expected_unimplemented")

    return wrapper


# The trip counts of a loop nest, outermost level first.  An op at an outer
# level of a deeper nest carries a prefix of its nest's counts -- an op only
# the outer loop stamped reads (2,) where the interior ops read (2, 4).
_Counts = tuple[int, ...]


def _counts(info: CoarseTileInfo) -> _Counts:
    return tuple(int(count) for count in info.loop_count)


def _describe(tiling: dict[str, CoarseTileInfo]) -> str:
    """One line per tiled op: its loop path, trip counts and tiled dims."""
    return "\n".join(
        f"  {name}: loop_group_id={info.loop_group_id} "
        f"loop_count={_counts(info)} loop_tiled_dims={info.loop_tiled_dims}"
        for name, info in sorted(tiling.items())
    )


def _nests(tiling: dict[str, CoarseTileInfo]) -> dict[int, dict[str, CoarseTileInfo]]:
    """The tiled ops grouped into loop nests, keyed by outermost loop id.

    ``loop_group_id[0]`` names an op's outermost loop, so ops sharing it sit in
    one nest however deep each of them goes.
    """
    nests: dict[int, dict[str, CoarseTileInfo]] = {}
    for name, info in tiling.items():
        nests.setdefault(info.loop_group_id[0], {})[name] = info
    return nests


def _nest_mismatch(nest: dict[str, CoarseTileInfo], expected: _Counts) -> Optional[str]:
    """Why ``nest`` is not a loop nest with trip counts ``expected``, or None.

    The longest ``loop_group_id`` in ``nest`` is its path.  Every op must sit
    on a prefix of that path -- an op only the outer levels cover carries only
    theirs -- with the same prefix of ``expected`` as its counts.  So a loop
    that went missing shortens the path, a level added inside the nest
    lengthens it, and a loop with the wrong count fails the counts of every op
    it covers.

    Prefix holds for a nest that tiles only output dims, which is all these
    cases write.  A nest that tiles a reduction dim can give its fill op a
    subset of levels that skips one (``_compute_fill_loop_info_planned``);
    such a nest never matches here, and no case expects one to.
    """
    path = max((info.loop_group_id for info in nest.values()), key=len)
    if len(path) != len(expected):
        return f"it is {len(path)} deep, not {len(expected)}"
    for name, info in sorted(nest.items()):
        depth = len(info.loop_group_id)
        if info.loop_group_id != path[:depth]:
            return f"{name} sits in a loop off the nest's path {path}"
        if _counts(info) != expected[:depth]:
            return f"{name} is tiled {_counts(info)}, not {expected[:depth]}"
    return None


@dataclasses.dataclass(frozen=True)
class _TilingCase:
    """One model plus the tiling contract asserted against it.

    inner:
        The part of the model the pins wrap in ``for_each_tile`` loops, untiled
        as written.  It takes the first ``len(named_dims)`` arguments.
    outer:
        The rest of the model, run on ``inner``'s result and the remaining
        arguments, outside every loop, or ``None`` when the loops cover the
        whole model.  It is what automatic tiling may add loops to: a loop the
        user wrote is never re-tiled.
    args:
        Device tensors passed to the compiled model, ``inner``'s first.
    named_dims:
        Per-argument axis labels for ``inner``'s arguments, and ``out_dims``
        the same for its result.  They are local to the test: a pin names an
        axis, and these say which axis of each operand (``None`` where it has
        none) and of the result that is.  Nothing is declared to the compiler
        -- ``for_each_tile`` states its tiling in the program itself.
    pins:
        The ``for_each_tile`` loops the *explicit* mode wraps around ``inner``,
        outermost first, and the whole of that mode's expectation: a pin is a
        ``(dim, count)`` and the nest it prescribes is those counts in that
        order, so a separate ``expected`` beside it could only restate them or
        contradict them.
    explicit_auto_pins:
        The loops for the *explicit_auto* mode, where automatic tiling is on
        as well.  What must survive is each loop exactly as written.
    """

    inner: Callable[..., torch.Tensor]
    outer: Optional[Callable[..., torch.Tensor]]
    args: tuple[torch.Tensor, ...]
    named_dims: tuple[Sequence[str], ...]
    out_dims: Sequence[str]
    pins: tuple[tuple[str, int], ...]
    explicit_auto_pins: tuple[tuple[str, int], ...]
    atol: float
    rtol: float

    @property
    def explicit_nest(self) -> _Counts:
        """The loop nest ``pins`` prescribes: their counts, outermost first."""
        return tuple(count for _, count in self.pins)

    @property
    def explicit_auto_nest(self) -> _Counts:
        """The same for ``explicit_auto_pins``."""
        return tuple(count for _, count in self.explicit_auto_pins)

    def model(self, pins: tuple[tuple[str, int], ...]) -> Callable[..., torch.Tensor]:
        """The whole model, with ``pins`` wrapped around ``inner``."""
        n_inner = len(self.named_dims)

        def run(*args: torch.Tensor) -> torch.Tensor:
            result = _apply_pins(self, pins, *args[:n_inner])
            if self.outer is None:
                return result
            return self.outer(result, *args[n_inner:])

        return run


def _apply_pins(
    case: _TilingCase, pins: tuple[tuple[str, int], ...], *args: torch.Tensor
) -> torch.Tensor:
    """Run ``case.inner`` inside one ``for_each_tile`` per pin, outermost first.

    Each pin slices every operand that has its axis into ``count`` tiles along
    it, passes the rest whole, and lays the result tiles back along the same
    axis of the output.  Tiles are rank-preserving, so an inner loop finds its
    axis at the same position in the tiles the outer one handed it, and the
    nesting order is the loop-nest order.
    """
    if not pins:
        return case.inner(*args)
    (dim, count), rest = pins[0], pins[1:]
    axes = tuple(
        list(names).index(dim) if dim in names else None for names in case.named_dims
    )
    extent = next(arg.shape[axis] for arg, axis in zip(args, axes) if axis is not None)

    def body(_, tiles):
        return None, _apply_pins(case, rest, *tiles)

    _, out = for_each_tile(
        body,
        args,
        dims=axes,
        tile_size=extent // count,
        out_dim=list(case.out_dims).index(dim),
    )
    return out


class CollectTilingPasses(CustomPreSchedulingPasses):
    """Pre-scheduling pipeline that records the applied tiling once it is done.

    ``torch_spyre._inductor.patches.enable_spyre_context`` installs
    ``CustomPreSchedulingPasses`` itself, so observing its result means
    substituting this subclass for it.  ``coarse_tile`` stamps ``loop_info``
    well before the scheduler is built, so reading it here sees the final plan.

    ``tiling`` maps every coarse-tiled op to its ``CoarseTileInfo``: the loops
    codegen will emit, whichever pass stamped them. ``core_splits`` maps each
    of them to its committed core division over its output: the split factor
    of each dim, keyed by that dim's stride in the output index.
    ``lx_resident`` names those of them whose buffer was placed in LX.
    """

    tiling: dict[str, CoarseTileInfo] = {}
    core_splits: dict[str, dict[int, int]] = {}
    lx_resident: set[str] = set()

    def __call__(self, graph: GraphLowering) -> None:
        super().__call__(graph)
        tiled = [
            op for op in graph.operations if getattr(op, "loop_info", None) is not None
        ]
        type(self).tiling = {op.get_name(): op.loop_info for op in tiled}
        type(self).core_splits = {
            op.get_name(): dict(op.op_it_space_splits[0])
            for op in tiled
            if getattr(op, "op_it_space_splits", None) is not None
        }
        type(self).lx_resident = {
            op.get_name()
            for op in tiled
            if "lx" in getattr(op.get_layout(), "allocation", {})
        }


class AutomatedCoarseTilingTests(
    unittest.TestCase, metaclass=_ParameterizedScratchpadMeta
):
    """model x tiling_mode x solver, one generated method per combination.

    The metaclass expands ``parameter_models`` against ``parameter_axes`` and
    routes each generated method through ``run_case``; ``case_decorators``
    marks the combos that cannot pass until the tile search exists.
    """

    def setUp(self):
        torch.manual_seed(0xAFFE)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    # ------------------------------------------------------------------
    # Compile and observe
    # ------------------------------------------------------------------
    def _compile_and_collect(
        self,
        case: "_TilingCase",
        pins: tuple[tuple[str, int], ...],
        *,
        layout_solver: str,
        auto_tiling: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, CoarseTileInfo]]:
        """Compile ``case`` and return (cpu_result, device_result, tiling)."""
        cpu_result = case.model(())(*(arg.to("cpu") for arg in case.args))

        CollectTilingPasses.tiling = {}
        # Auto tiling needs the joint CP-SAT co-opt path on (R8.1): the solver
        # carries and applies the tiling candidates only under
        # co_optimizing_lx_planning + layout_solver="cpsat". The two gates
        # default off, so they are patched on only for the auto modes.
        tiling_cfg = (
            dict(
                co_optimizing_lx_planning=True,
                auto_coarse_tiling=True,
            )
            if auto_tiling
            else {}
        )
        # force_disable_caches belongs to torch's inductor config, not Spyre's;
        # CustomPreSchedulingPasses is a plain module attribute that
        # enable_spyre_context re-imports per compile, so it is swapped with
        # patch.object rather than a config knob.  no_grad because the Linear
        # weights require grad, and scan cannot trace an autograd graph for a
        # nested for_each_tile.
        with (
            torch.no_grad(),
            t_inductor_config.patch(force_disable_caches=True),
            ts_inductor_config.patch(
                allow_all_ops_in_lx_planning=True,
                layout_solver=layout_solver,
                **tiling_cfg,
            ),
            patch.object(ts_passes, "CustomPreSchedulingPasses", CollectTilingPasses),
        ):
            compiled = torch.compile(case.model(pins), fullgraph=True)
            device_result = compiled(*case.args).to("cpu")

        return cpu_result, device_result, CollectTilingPasses.tiling

    def _assert_matches_cpu(self, case: "_TilingCase", device, cpu) -> None:
        torch.testing.assert_close(
            device,
            cpu,
            atol=case.atol,
            rtol=case.rtol,
            msg=lambda m: f"coarse-tiled result diverged from CPU\n\n{m}\n",
        )

    # ------------------------------------------------------------------
    # The three contracts
    # ------------------------------------------------------------------
    def _check_loops_kept(
        self,
        case: "_TilingCase",
        pins: tuple[tuple[str, int], ...],
        expected: _Counts,
        solver: str,
        *,
        auto_tiling: bool,
    ) -> None:
        """Some loop nest is exactly the one ``pins`` wrote, and the result is right.

        What else the compiler tiles, and which pass stamped the matching nest,
        does not matter -- only that codegen gets the written loops.  A loop
        removed, or a level added inside one, leaves no nest that matches.
        """
        cpu, device, tiling = self._compile_and_collect(
            case, pins, layout_solver=solver, auto_tiling=auto_tiling
        )
        mismatches = {
            outer: _nest_mismatch(nest, expected)
            for outer, nest in _nests(tiling).items()
        }
        self.assertIn(
            None,
            mismatches.values(),
            f"no loop nest is the written {list(pins)} "
            f"(per nest: {mismatches}):\n{_describe(tiling)}",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_loops_preserved(self, case: "_TilingCase", solver: str) -> None:
        """The written loops come out exactly as written."""
        self._check_loops_kept(
            case, case.pins, case.explicit_nest, solver, auto_tiling=False
        )

    def _check_tiling_discovered(self, case: "_TilingCase", solver: str) -> None:
        """With no loops at all, the compiler picks a tiling by itself."""
        if solver == "simulated_annealing":
            raise NotImplementedError
        cpu, device, tiling = self._compile_and_collect(
            case, (), layout_solver=solver, auto_tiling=True
        )
        self.assertTrue(
            tiling,
            "Auto tiling is on and no loops were written, but no op was "
            "coarse-tiled -- the tile search found nothing to do",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_loops_preserved_with_auto(
        self, case: "_TilingCase", solver: str
    ) -> None:
        """With the tile search on as well, the written loops still come out."""
        self._check_loops_kept(
            case,
            case.explicit_auto_pins,
            case.explicit_auto_nest,
            solver,
            auto_tiling=True,
        )

    # ------------------------------------------------------------------
    # Models.  Each returns the model, its axis labels and the tiling contract,
    # defined once and reused across every tiling_mode and solver.
    #
    # Each is sized so its working set overflows LX even split across 32
    # cores: a cost-driven tile search is right to leave a graph that already
    # fits untiled, so a smaller model cannot tell a working search from one
    # that never tiles.
    # ------------------------------------------------------------------
    def _softmax_case(self) -> "_TilingCase":
        """softmax(dim=0) over (512, 65536), dims R (reduced) x C.

        One level: C divided 4 ways, each tile a whole-column softmax.  The
        other axis, R, is the reduced one: a map loop over it would softmax
        each row block on its own and change the result, and a reduction loop
        is a different model, so C is the whole prescribed plan.  The loop
        covers the whole model, so the explicit_auto mode leaves the compiler
        nothing outside it to tile.
        """
        return _TilingCase(
            inner=functools.partial(torch.softmax, dim=0),
            outer=None,
            args=(torch.rand((512, 65536), dtype=torch.float16, device=DEVICE_NAME),),
            named_dims=(["R", "C"],),
            out_dims=["R", "C"],
            pins=(("C", 4),),  # Reduction axis is not tiled for now
            explicit_auto_pins=(("C", 4),),
            # A good run lands at 2e-5 on outputs of order 1/512; the
            # reduction-tiled one lands at 3e-3, and this has to separate them.
            atol=5e-4,
            rtol=0.02,
        )

    def _mlp_case(self) -> "_TilingCase":
        """Two-layer MLP (Linear -> silu -> Linear), dims S x Din x Dh x Dout.

        The loops cover the whole model: S divided 2 ways outside Dout divided
        2 ways.  Both are free (output) axes -- Din is the first GEMM's
        reduction and Dh the second's -- so both GEMMs sit inside the nest,
        the second one reading the first's in-loop result.  The Dout loop
        slices only the second Linear's weight and bias; the first Linear and
        silu have no Dout axis and are invariant at that level.  The loops
        cover the whole model, so the explicit_auto mode writes the same nest
        and leaves the compiler nothing outside it to tile.
        """
        # S is large enough that the hidden activations (S x Dh) do not fit LX
        # untiled. At a size where they do, a nest only adds the copy-out at
        # its boundary and the solve rightly leaves the model untiled.
        seq_len, in_dim, hidden_dim, out_dim = 32768, 256, 1024, 256
        fc1 = torch.nn.Linear(in_dim, hidden_dim).half()
        fc2 = torch.nn.Linear(hidden_dim, out_dim).half()

        def mlp(x, w1, b1, w2, b2):
            return torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1)), w2, b2
            )

        args = (
            torch.randn(seq_len, in_dim, dtype=torch.float16).to(DEVICE_NAME),
            fc1.weight.to(DEVICE_NAME),
            fc1.bias.to(DEVICE_NAME),
            fc2.weight.to(DEVICE_NAME),
            fc2.bias.to(DEVICE_NAME),
        )
        return _TilingCase(
            inner=mlp,
            outer=None,
            args=args,
            named_dims=(
                ["S", "Din"],
                ["Dh", "Din"],
                ["Dh"],
                ["Dout", "Dh"],
                ["Dout"],
            ),
            out_dims=["S", "Dout"],
            pins=(("S", 2), ("Dout", 2)),
            explicit_auto_pins=(("S", 2), ("Dout", 2)),
            atol=0.02,
            rtol=0.05,
        )

    def _swiglu_case(self) -> "_TilingCase":
        """SwiGLU FFN (silu(gate) * up, then a down projection), dims S x Din x Dh.

        The loops cover the gated half: S divided 2 ways outside Dh divided 4
        ways.  Dh is a free axis the whole way through it -- the N dimension of
        both GEMMs and the layout of every activation -- and both weights
        carry the ``Dh`` label, so the inner loop slices the gate and up
        branches together.  The down projection reduces over Dh and runs
        outside every loop; the explicit_auto mode writes only the S loop.
        """
        seq_len, in_dim, hidden_dim = 16384, 256, 1024
        fc_gate = torch.nn.Linear(in_dim, hidden_dim).half()
        fc_up = torch.nn.Linear(in_dim, hidden_dim).half()
        fc_down = torch.nn.Linear(hidden_dim, in_dim, bias=False).half()

        def gated(x, w_gate, b_gate, w_up, b_up):
            gate = torch.nn.functional.linear(x, w_gate, b_gate)
            up = torch.nn.functional.linear(x, w_up, b_up)
            return torch.nn.functional.silu(gate) * up

        def down_proj(h, w_down):
            return torch.nn.functional.linear(h, w_down)

        args = (
            torch.randn(seq_len, in_dim, dtype=torch.float16).to(DEVICE_NAME),
            fc_gate.weight.to(DEVICE_NAME),
            fc_gate.bias.to(DEVICE_NAME),
            fc_up.weight.to(DEVICE_NAME),
            fc_up.bias.to(DEVICE_NAME),
            fc_down.weight.to(DEVICE_NAME),
        )
        return _TilingCase(
            inner=gated,
            outer=down_proj,
            args=args,
            named_dims=(
                ["S", "Din"],
                ["Dh", "Din"],
                ["Dh"],
                ["Dh", "Din"],
                ["Dh"],
            ),
            out_dims=["S", "Dh"],
            pins=(("S", 2), ("Dh", 4)),
            explicit_auto_pins=(("S", 2),),
            atol=0.02,
            rtol=0.05,
        )

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------
    _CHECKS = {
        "explicit": _check_loops_preserved,
        "auto": _check_tiling_discovered,
        "explicit_auto": _check_loops_preserved_with_auto,
    }

    parameter_axes = {
        "tiling_mode": tuple(_CHECKS),
        "solver_method": ("cpsat", "simulated_annealing"),
    }

    # SDPA is omitted: using SDPA in this test suite requires resolution of
    # https://github.com/torch-spyre/torch-spyre/issues/3198

    parameter_models = (
        ("softmax_tiling", _softmax_case),
        ("mlp_tiling", _mlp_case),
        ("swiglu_tiling", _swiglu_case),
    )

    @staticmethod
    def case_decorators(params):
        """Per-combo decorators.

        The ``auto``/``explicit_auto`` combos were ``@expected_unimplemented``
        while the solver-driven tile search was unbuilt, and briefly
        ``@expected_lx_ownership_gap`` while commit_divisions dropped the
        division the solve had chosen for each ``coarse_tile_copy_*``. Both
        markers retired themselves the moment those modes passed (each fails a
        clean run), so only the ortools skip for the cpsat solver remains.
        """
        decorators = []
        if params["solver_method"] == "cpsat":
            decorators.append(
                unittest.skipUnless(_HAS_ORTOOLS, "the cpsat solver needs ortools")
            )
        if params["solver_method"] in ("simulated_annealing") and params[
            "tiling_mode"
        ] in ("auto"):
            decorators.append(expected_unimplemented)
        return decorators

    def run_case(self, params: dict, factory: Callable) -> None:
        """Body of one generated method: build the model, check its contract."""
        self._CHECKS[params["tiling_mode"]](
            self, factory(self), params["solver_method"]
        )


# ---------------------------------------------------------------------------
# Read-distance (span-limit) enforcement on the discovery path
# ---------------------------------------------------------------------------
# Device-free op builders, mirroring the span-overflow / enumerate-tilings
# tests: a real FixedTiledLayout ComputedBuffer over a lightweight Pointwise
# mock is all _tiling_candidates -> enumerate_tile_options -> the read-distance
# filter needs.
def _fixed_tiled_layout(shape, dtype=torch.float16):
    size = list(shape)
    stride = list(FlexibleLayout.contiguous_strides(size))
    stride_ints = [int(s) for s in stride]
    size_ints = [int(s) for s in size]
    within_stick_dim = len(size_ints) - 1
    dim_order = [i for i in range(len(size_ints)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size_ints, stride_ints, dtype, dim_order)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _pointwise_op(shape, name="buf0"):
    data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    layout = _fixed_tiled_layout(shape)
    op = ComputedBuffer(name=name, layout=layout, data=data)
    op.operation_name = name
    syms = sympy.symbols(
        " ".join(f"d{i}" for i in range(len(shape))), integer=True, nonnegative=True
    )
    if not isinstance(syms, tuple):
        syms = (syms,)
    index = sympy.Integer(0)
    for sym, stride in zip(syms, layout.stride):
        index += sym * int(stride)
    write = MemoryDep(name, index, syms, tuple(shape))
    op.get_read_writes = MagicMock(
        return_value=SimpleNamespace(reads=set(), writes={write})
    )
    return op


def _matmul_op(out_shape=(128, 256), k=64, name="mm"):
    """A minimal matmul ``ComputedBuffer`` for ``_tiling_candidates`` unit tests.

    Mirrors ``_pointwise_op`` but gives the op a ``Reduction`` ``data`` carrying
    a matmul reduction type, so ``_is_matmul_op`` recognises it. Like the
    pointwise mock it declares no reads, keeping the read-distance filter
    permissive; the enumerator's own filters (output-axis ``is_clean``, stick-dim
    exclusion) are what these tests exercise, so the reduction ranges only need
    to exist, not to enumerate.
    """
    data = MagicMock(spec=Reduction)
    data.ranges = list(out_shape)
    data.reduction_ranges = [k]
    data.reduction_type = BATCH_MATMUL_OP
    layout = _fixed_tiled_layout(out_shape)
    op = ComputedBuffer(name=name, layout=layout, data=data)
    op.operation_name = name
    syms = sympy.symbols(
        " ".join(f"d{i}" for i in range(len(out_shape))), integer=True, nonnegative=True
    )
    if not isinstance(syms, tuple):
        syms = (syms,)
    index = sympy.Integer(0)
    for sym, stride in zip(syms, layout.stride):
        index += sym * int(stride)
    write = MemoryDep(name, index, syms, tuple(out_shape))
    op.get_read_writes = MagicMock(
        return_value=SimpleNamespace(reads=set(), writes={write})
    )
    return op


# (1, 8195, 256, 64) fp16 = 268.7 MB, just over the 256 MiB read-distance limit;
# splitting dim 1 (8195 = 5*11*149) by 5 -> 53.7 MB brings it back under.
_OVERFLOW_SHAPE = (1, 8195, 256, 64)
# dim 1 is prime with no divisor <= _MAX_AUTO_TILE_SPLIT_COUNT, so it cannot be
# tiled; even the largest legal split of dim 2 leaves the span over the limit,
# so no discovered tiling (untiled included) fits.
_UNTILEABLE_OVERFLOW_SHAPE = (1, 1048583, 256, 64)


@unittest.skipUnless(_HAS_ORTOOLS, "the cpsat solver needs ortools")
class DiscoveryReadDistanceTests(unittest.TestCase):
    """The auto-tiling discovery path never offers the solve an over-span tiling.

    Integration check for the real ``_tiling_candidates`` wiring (config gates ->
    ``enumerate_tile_options`` -> the read-distance filter): under
    ``auto_coarse_tiling`` an unpinned op whose untiled read overflows
    ``MAX_SPAN_BYTES`` loses the untiled option -- it must be tiled to fit --
    while a fitting op keeps it, and an op that cannot be tiled to fit raises
    ``Unsupported``, matching the span-overflow planner's own gate. No device or
    solve is needed: ``_tiling_candidates`` is a pure function of the op and
    config.
    """

    _AUTO_CFG = dict(
        co_optimizing_lx_planning=True,
        auto_coarse_tiling=True,
        layout_solver="cpsat",
        sencores=4,
    )

    def _candidates(self, op, **overrides):
        cfg = {**self._AUTO_CFG, **overrides}
        with ts_inductor_config.patch(cfg):
            alloc = select_allocator()
            return alloc._tiling_candidates(op, cfg["sencores"])

    def test_untiled_dropped_for_overspan_unhinted_op(self):
        op = _pointwise_op(_OVERFLOW_SHAPE)
        options = self._candidates(op)

        # The untiled read overflows, so untiled is not offered -- the solve is
        # forced to tile -- but a fitting tiling remains.
        self.assertNotIn(TileSpec(), options)
        self.assertTrue(options)
        # And every option the solve is offered actually fits the limit.
        with ts_inductor_config.patch(self._AUTO_CFG):
            for spec in options:
                self.assertTrue(
                    _spec_within_read_distance(op, spec, 4),
                    f"discovery offered an over-span tiling {spec}",
                )

    def test_untiled_kept_for_fitting_unhinted_op(self):
        op = _pointwise_op((1, 2, 16, 64))
        options = self._candidates(op)

        # Nothing overflows, so untiled stays on the menu.
        self.assertIn(TileSpec(), options)

    def test_raises_when_no_discovered_tiling_fits(self):
        op = _pointwise_op(_UNTILEABLE_OVERFLOW_SHAPE)

        with self.assertRaisesRegex(Unsupported, "read-distance limit"):
            self._candidates(op)

    def test_no_filtering_without_auto_coarse_tiling(self):
        # Discovery off -> the op stays untiled and the read-distance filter never
        # runs, even when the untiled read overflows: the drop is discovery's job.
        op = _pointwise_op(_OVERFLOW_SHAPE)
        options = self._candidates(op, auto_coarse_tiling=False)

        self.assertEqual(options, [TileSpec()])

    def test_reshaping_reader_drops_the_unit_tile(self):
        # d3:64 leaves a 1-extent tile, which apply_tilings cannot retile
        # for a reader that views the output through another rank.
        op = _pointwise_op((2, 8, 5, 64, 128))
        reader = MagicMock(spec=ComputedBuffer)
        reader.data = SimpleNamespace(ranges=[2, 40, 64, 128])
        unit_tile = TileSpec((TileAxis(host_dim=3, count=64),))

        self.assertIn(unit_tile, self._candidates(op))
        with patch.object(
            CoOptimizingAllocator,
            "_readers_by_name",
            {op.get_name(): [reader]},
            create=True,
        ):
            self.assertNotIn(unit_tile, self._candidates(op))

    def test_op_inside_a_for_each_tile_region_offered_no_tiling(self):
        # A for_each_tile region also holds ops no loop level stamped. The
        # user's loop covers them all the same, so the solve may not start a
        # nest of its own there.
        op = _pointwise_op((1, 2, 16, 64))
        self.assertGreater(len(self._candidates(op)), 1)  # non-vacuity
        with patch.object(
            CoOptimizingAllocator,
            "_prescribed_ops",
            frozenset({op.get_operation_name()}),
            create=True,
        ):
            self.assertEqual(self._candidates(op), [TileSpec()])

    def test_restickify_offered_no_tiling(self):
        # No tiling of a restickify is correct once the sticks are laid out.
        op = _pointwise_op((1, 2, 16, 64))
        self.assertGreater(len(self._candidates(op)), 1)  # non-vacuity
        op.origin_node = SimpleNamespace(target=torch.ops.spyre.restickify.default)
        self.assertEqual(self._candidates(op), [TileSpec()])

    def test_matmul_offered_output_tiling(self):
        # A matmul is no longer excluded by an op-kind guard: under discovery it
        # is offered its row/M-axis (host_dim 0, non-reduction) output tilings,
        # exactly like any other op.  This locks in the removal of the former
        # `_is_matmul_op(op)` short-circuit in `_tiling_candidates`.
        op = _matmul_op(out_shape=(128, 256))
        options = self._candidates(op)

        self.assertIn(TileSpec(), options)  # untiled still offered (read fits)
        tiled = [t for t in options if not t.is_untiled]
        self.assertTrue(
            tiled,
            "matmul was offered no tiling -- the removed matmul guard has "
            f"silently returned (options were {options})",
        )
        for spec in tiled:
            self.assertEqual(
                [a.host_dim for a in spec.axes],
                [0],
                f"matmul offered a non-M-axis output tiling {spec}",
            )

    def test_matmul_never_offered_reduction_or_stick_tiling(self):
        # Removing the guard does not open the numerically-fragile forms: the
        # `is_clean` filter still drops every reduction (K) tiling, and the
        # enumerator never emits the stick (innermost / N) dim.  So no offered
        # spec tiles a reduction axis or the last host dim.
        out_shape = (128, 256)
        stick_dim = len(out_shape) - 1
        op = _matmul_op(out_shape=out_shape)
        options = self._candidates(op)

        for spec in options:
            self.assertFalse(
                any(a.is_reduction for a in spec.axes),
                f"matmul offered a reduction tiling {spec} (K-tiling is "
                "~2 orders off CPU -- see _mlp_case)",
            )
            self.assertFalse(
                any(a.host_dim == stick_dim for a in spec.axes),
                f"matmul offered a stick-dim tiling {spec}",
            )


# ---------------------------------------------------------------------------
# Loop nests follow tile ownership, not TileSpec equality
# ---------------------------------------------------------------------------
_CHOSEN_TILINGS = "torch_spyre._inductor.scratchpad.coarse_tiling.chosen_tilings"
_D0_BY_4 = TileSpec((TileAxis(host_dim=0, count=4),))
_D1_BY_4 = TileSpec((TileAxis(host_dim=1, count=4),))


def _reads_graph_inputs_only(op) -> bool:
    graph_inputs = set(V.graph.graph_inputs)
    return all(dep.name in graph_inputs for dep in op.get_read_writes().reads)


@unittest.skipUnless(_HAS_ORTOOLS, "the cpsat solver needs ortools")
class TileOwnershipGroupingTests(unittest.TestCase):
    """A consumer shares its producer's loop nest only where it reads it tile by
    tile.

    ``TileAxis.host_dim`` is positional in each op's own output, so a consumer
    that reads its producer permuted, reduced or narrowed can carry an equal
    ``TileSpec`` and still walk a different part of the producer's buffer.
    Sharing a nest there reads, on tile ``t``, what the producer has not
    written on tile ``t``. Each case compiles with automatic tiling on and is
    checked against CPU.

    ``consumer_menu`` replaces the discovered menus: the producer ``a = x + y``
    (the one op that reads only graph inputs) is offered ``d0:4`` alone, and
    every other op ``consumer_menu``. The ``_apply`` cases skip the solve's
    choice instead, so they check ``apply_tilings``'s own refusal.
    """

    def setUp(self):
        torch.manual_seed(0xAFFE)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def _compile(self, fn, args, consumer_menu=None):
        """(cpu result, device result, tiling) for ``fn`` under auto tiling."""
        cpu = fn(*args)
        CollectTilingPasses.tiling = {}
        with contextlib.ExitStack() as stack:
            if consumer_menu is not None:

                def offered(alloc, op, max_cores):
                    if _reads_graph_inputs_only(op):
                        return [_D0_BY_4]
                    return consumer_menu

                stack.enter_context(
                    patch.object(CoOptimizingAllocator, "_tiling_candidates", offered)
                )
            stack.enter_context(torch.no_grad())
            stack.enter_context(t_inductor_config.patch(force_disable_caches=True))
            stack.enter_context(
                ts_inductor_config.patch(
                    co_optimizing_lx_planning=True,
                    auto_coarse_tiling=True,
                    layout_solver="cpsat",
                    allow_all_ops_in_lx_planning=True,
                )
            )
            stack.enter_context(
                patch.object(
                    ts_passes, "CustomPreSchedulingPasses", CollectTilingPasses
                )
            )
            compiled = torch.compile(fn, fullgraph=True)
            device = compiled(*(arg.to(DEVICE_NAME) for arg in args)).to("cpu")
        return cpu, device, CollectTilingPasses.tiling

    def _assert_close(self, device, cpu):
        torch.testing.assert_close(device, cpu, atol=0.1, rtol=0.05)

    @staticmethod
    def _model_ops(nest):
        """The nest's ops other than the copies coarse tiling adds to it."""
        return sorted(name for name in nest if not name.startswith("coarse_tile_copy_"))

    def test_permuted_consumer_of_a_solver_tiled_producer(self):
        # Unforced: the solve alone once tiled both ops d0:2/d1:2 into one
        # nest, and only the diagonal tiles came out right.
        x = torch.randn(128, 128, 2048, dtype=torch.float16)
        y = torch.randn(128, 128, 2048, dtype=torch.float16)
        cpu, device, _ = self._compile(
            lambda x, y: (x + y).permute(1, 0, 2) * 2 + 1, (x, y)
        )
        self._assert_close(device, cpu)

    def _compile_with_both_tiled_d0(self, fn, args):
        """Compile ``fn`` as though the solve had tiled every op ``d0:4``.

        Bypasses the solve's own pairing, so what is left to split an
        out-of-step group is ``apply_tilings`` itself.
        """

        def chosen(graph, allocation):
            return {
                op.get_operation_name(): _D0_BY_4
                for op in graph.operations
                if isinstance(op, ComputedBuffer)
            }

        with patch(_CHOSEN_TILINGS, chosen):
            return self._compile(fn, args)

    def _assert_group_split(self, fn):
        """The consumer does not read its producer tile by tile, so the apply
        starts a new loop group at it, and it reads the producer's full buffer
        across the boundary."""
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)
        cpu, device, tiling = self._compile_with_both_tiled_d0(fn, (x, y))
        self._assert_close(device, cpu)
        self.assertGreater(len(_nests(tiling)), 1, _describe(tiling))

    def test_apply_splits_a_permuted_consumer_from_the_nest(self):
        # The consumer's d0 is the producer's dim 1.
        self._assert_group_split(lambda x, y: (x + y).permute(1, 0, 2) * 2)

    def test_apply_splits_a_reducing_consumer_from_the_nest(self):
        # sum(0) drops the producer's dim 0, so the consumer's d0 is dim 1.
        self._assert_group_split(lambda x, y: (x + y).sum(dim=0))

    def test_apply_splits_a_narrowing_consumer_from_the_nest(self):
        # Same dim, half the rows: tile t reads 8 rows where the producer
        # wrote 16.
        self._assert_group_split(lambda x, y: (x + y)[:32] * 2)

    def test_apply_refuses_a_tiling_along_a_repeated_axis(self):
        # repeat walks dim 1 of its input twice. Tiled four ways on that axis
        # it once compiled and read the wrong elements: a tile has to advance
        # and then wrap back, which the apply cannot express. The enumerator
        # does not offer the axis; forcing it checks the apply itself.
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)

        def chosen(graph, allocation):
            return {
                op.get_operation_name(): _D1_BY_4
                for op in graph.operations
                if isinstance(op, ComputedBuffer) and not _reads_graph_inputs_only(op)
            }

        with (
            patch(_CHOSEN_TILINGS, chosen),
            self.assertRaises(Exception) as refusal,
        ):
            self._compile(lambda x, y: (x + y).repeat(1, 2, 1) * 2, (x, y))
        self.assertIn("more than once along", str(refusal.exception))

    def test_apply_accepts_a_consumer_that_reads_in_step(self):
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)
        cpu, device, tiling = self._compile_with_both_tiled_d0(
            lambda x, y: (x + y) * 2, (x, y)
        )
        self._assert_close(device, cpu)
        nests = list(_nests(tiling).values())
        self.assertEqual(len(nests), 1, _describe(tiling))
        self.assertEqual(len(self._model_ops(nests[0])), 2, _describe(tiling))

    def test_apply_grows_back_a_unit_tile_beside_a_unit_dim(self):
        # d1:64 leaves the producer's tile (1, 1, 2048). Growing its copy-out's
        # full buffer back from that tile's device layout cannot tell the two
        # unit dims apart beside the unit dim 0, and would return
        # [1, 32, 1, 64] for [64, 32, 1, 64]. The apply instead gives the full
        # buffer the producer's own device layout, read before the tile is
        # divided, so the untiled consumer reads a buffer of the size it
        # expects. The enumerator does not offer this tiling; forcing it checks
        # the apply itself.
        x = torch.randn(1, 64, 2048, dtype=torch.float16)
        y = torch.randn(1, 64, 2048, dtype=torch.float16)
        unit_tile = TileSpec((TileAxis(host_dim=1, count=64),))

        def chosen(graph, allocation):
            return {
                op.get_operation_name(): unit_tile
                for op in graph.operations
                if isinstance(op, ComputedBuffer) and _reads_graph_inputs_only(op)
            }

        with patch(_CHOSEN_TILINGS, chosen):
            cpu, device, _ = self._compile(lambda x, y: (x + y) * 2, (x, y))
        self._assert_close(device, cpu)

    def test_permuted_consumer_shares_the_nest_on_the_matching_dim(self):
        # d1:4 is the consumer tiling that walks the producer's dim 0, so the
        # two share one four-trip nest although their specs differ.
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)
        cpu, device, tiling = self._compile(
            lambda x, y: (x + y).permute(1, 0, 2) * 2,
            (x, y),
            consumer_menu=[_D1_BY_4],
        )
        self._assert_close(device, cpu)
        nests = list(_nests(tiling).values())
        self.assertEqual(len(nests), 1, _describe(tiling))
        self.assertEqual(len(self._model_ops(nests[0])), 2, _describe(tiling))
        self.assertIsNone(_nest_mismatch(nests[0], (4,)), _describe(tiling))

    def test_a_tile_is_divided_across_cores_along_its_tiled_dim(self):
        # Every op runs one d0:4 nest, 16 rows to a tile. The solve owns each
        # op's division and ties the divisions of the nest together, so it may
        # give each core a slice of the tiled dim itself. The balanced division
        # it prefers does: 4 cores along those 16 rows.
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)
        cpu, device, tiling = self._compile(
            lambda x, y: (x + y) * 2 + 1, (x, y), consumer_menu=[_D0_BY_4]
        )
        self._assert_close(device, cpu)
        (nest,) = _nests(tiling).values()
        model_ops = self._model_ops(nest)
        self.assertEqual(len(model_ops), 3, _describe(tiling))
        dim_0_stride = 64 * 128
        for name in model_ops:
            splits = CollectTilingPasses.core_splits[name]
            self.assertGreater(splits.get(dim_0_stride, 1), 1, f"{name}: {splits}")

    def test_a_cut_tile_resides_on_the_tiled_graph(self):
        # The producer is tiled and its readers are not, so it is a cut: its
        # copy op drains each tile into a full HBM buffer for them and is the
        # tile's only reader. The solve splits the tiled dim across cores, and
        # the copy has to read the tile sliced the same way. Held to a whole
        # tiled dim it could not, and the tile the solve had placed in LX was
        # demoted to HBM on the tiled graph.
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)
        cpu, device, tiling = self._compile(
            lambda x, y: (x + y) * 2 + 1, (x, y), consumer_menu=[TileSpec()]
        )
        self._assert_close(device, cpu)
        (nest,) = _nests(tiling).values()
        (producer,) = self._model_ops(nest)
        self.assertIn(producer, CollectTilingPasses.lx_resident, _describe(tiling))

    def test_a_matmul_cut_tile_resides_on_the_tiled_graph(self):
        # The matmul is tiled and is the graph output, so it is a cut and its
        # copy op is the tile's only reader. The copy has no reduction axis,
        # and reads the tile from LX only where each core holds a slice of the
        # output. The solve once split the matmul's reduction axis too, for a
        # tile it had placed in LX, and the tiled graph spilled the tile.
        x = torch.rand(64, 64, 128, dtype=torch.float16)
        w = torch.rand(128, 128, dtype=torch.float16)
        cpu, device, tiling = self._compile(
            lambda x, w: x @ w, (x, w), consumer_menu=[TileSpec()]
        )
        self._assert_close(device, cpu)
        (nest,) = _nests(tiling).values()
        (producer,) = self._model_ops(nest)
        self.assertIn(producer, CollectTilingPasses.lx_resident, _describe(tiling))

    def test_a_unit_tile_keeps_the_plan_the_solve_made(self):
        # Every op is offered d0:64 alone, a tile one row long. Such a tile no
        # longer iterates dim 0, so its iteration symbols are numbered from
        # dim 1: what the solve called d1 is the tile's d0. The division the
        # solve chose is carried onto the tile under the tile's own symbols.
        # Taken under the old ones it did not fit the tile, and the whole plan
        # was given up for greedy placement, which left every tile in HBM.
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)
        unit_tile = TileSpec((TileAxis(host_dim=0, count=64),))

        def offered(alloc, op, max_cores):
            return [unit_tile]

        with patch.object(CoOptimizingAllocator, "_tiling_candidates", offered):
            cpu, device, tiling = self._compile(lambda x, y: (x + y) * 2 + 1, (x, y))
        self._assert_close(device, cpu)
        (nest,) = _nests(tiling).values()
        model_ops = self._model_ops(nest)
        self.assertEqual(len(model_ops), 3, _describe(tiling))
        for name in model_ops:
            self.assertIn(name, CollectTilingPasses.lx_resident, _describe(tiling))

    def test_untiled_op_splits_a_skip_connection_into_two_nests(self):
        # a -> b -> c with a -> c as well. a and c may only be tiled d0:4 and
        # b, between them, only untiled. Loop groups are consecutive runs, so
        # a and c cannot share a nest across b whatever their own edge allows:
        # a is copied out, and b and c both read its full buffer.
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)

        def fn(x, y):
            a = x + y
            b = a * a
            return a + b

        def offered(alloc, op, max_cores):
            # b is the one op with a single buffer to read.
            reads = {dep.name for dep in op.get_read_writes().reads}
            if len(reads) == 1:
                return [TileSpec()]
            return [_D0_BY_4]

        with patch.object(CoOptimizingAllocator, "_tiling_candidates", offered):
            cpu, device, tiling = self._compile(fn, (x, y))
        self._assert_close(device, cpu)
        nests = [self._model_ops(nest) for nest in _nests(tiling).values()]
        self.assertEqual([len(ops) for ops in nests], [1, 1], _describe(tiling))

    def test_mutation_op_compiles(self):
        # copy_forced writes through its target's layout, which has no device
        # layout for the enumerator to stick-check a tile against.
        a = torch.randn(128, 256, dtype=torch.float16) * 0.01
        b = torch.randn(128, 256, dtype=torch.float16) * 0.01
        d = torch.randn(128, 256, dtype=torch.float16) * 0.01

        def fn(a, b, d):
            return torch.ops.spyre.copy_forced(d, a + b)

        cpu, device, _ = self._compile(fn, (a, b, d))
        self._assert_close(device, cpu)

    def test_narrowing_consumer_stays_out_of_its_producers_nest(self):
        # The consumer takes half the producer's rows, and the only tiling on
        # offer for either would walk them in one nest. A buffer read in part
        # gets no pairs, so the solve may not pick that.
        x = torch.randn(64, 64, 128, dtype=torch.float16)
        y = torch.randn(64, 64, 128, dtype=torch.float16)
        cpu, device, tiling = self._compile(
            lambda x, y: (x + y)[:32] * 2, (x, y), consumer_menu=[_D0_BY_4]
        )
        self._assert_close(device, cpu)
        for nest in _nests(tiling).values():
            self.assertLess(len(self._model_ops(nest)), 2, _describe(tiling))

    def test_consumer_that_reads_its_producer_twice(self):
        # Unforced. One read of a is in step with it and the other transposed,
        # so no tiling of the consumer walks both the way a is written. The
        # pair table was once built from the first read alone: the solve put
        # the two ops in one nest and the apply refused it.
        x = torch.randn(128, 128, 2048, dtype=torch.float16)
        y = torch.randn(128, 128, 2048, dtype=torch.float16)

        def fn(x, y):
            a = x + y
            return a + a.permute(1, 0, 2)

        cpu, device, _ = self._compile(fn, (x, y))
        self._assert_close(device, cpu)

    def test_consumer_that_repeats_its_producer(self):
        # Unforced. repeat walks dim 1 of its input twice. The solve once tiled
        # the repeat on that dim, which the apply took and codegen then could
        # not express; the axis is no longer on offer.
        #
        # The repeat also takes half the producer's rows, each twice, so it
        # reads as many elements as the producer wrote. The apply tells a full
        # read by counting elements and would take the two in one nest, where
        # a tile reads half the rows its producer wrote. It is never handed
        # one: the unread rows are a back gap, which bars the producer's
        # buffer from LX, and a barred buffer gets no pairs. There is no test
        # of the apply refusing that nest for that reason.
        x = torch.randn(128, 128, 2048, dtype=torch.float16)
        y = torch.randn(128, 128, 2048, dtype=torch.float16)
        cpu, device, _ = self._compile(
            lambda x, y: (x + y)[:64].repeat(1, 2, 1) * 2, (x, y)
        )
        self._assert_close(device, cpu)

    def test_consumer_of_a_fallback_kernel(self):
        # tril lowers to a FallbackKernel, whose output is a MultiOutput with
        # no iteration space to build a view from. It cannot live in LX, so it
        # gets no pairs -- its consumer having tilings to offer changes nothing.
        x = torch.randn(128, 128, dtype=torch.float16)
        y = torch.randn(128, 128, dtype=torch.float16)
        cpu, device, _ = self._compile(lambda x, y: torch.tril(x) + y, (x, y))
        self._assert_close(device, cpu)


class SplitsOnTileTests(unittest.TestCase):
    """``splits_on_tile`` restates a core division on the symbols the tiled op
    iterates, from the ``symbols_on_tile`` stamp ``apply_tilings`` leaves on
    it."""

    _D0, _D1 = sympy.symbols("d0 d1")

    def test_an_op_the_apply_did_not_tile_keeps_its_symbols(self):
        untiled = SimpleNamespace()
        self.assertEqual(splits_on_tile(untiled, {self._D1: 2}), {self._D1: 2})

    def test_a_unit_tile_renumbers_the_dims_after_it(self):
        # A (64, 64, 128) op tiled d0:64 is a (1, 64, 128) tile: it no longer
        # iterates dim 0, and what was d1 is its d0.
        tile = SimpleNamespace(symbols_on_tile={self._D1: self._D0})
        self.assertEqual(splits_on_tile(tile, {self._D1: 2}), {self._D0: 2})

    def test_a_split_of_a_dim_the_tile_no_longer_iterates_is_refused(self):
        tile = SimpleNamespace(symbols_on_tile={self._D1: self._D0})
        self.assertIsNone(splits_on_tile(tile, {self._D0: 2}))


class CommitLxViewsTests(unittest.TestCase):
    """``commit_lx_views`` gives each buffer placed in LX the per-core view it
    is held under: the one the residency judge accepts for it on the graph,
    or, for the source of an accepted relayout, the one its plan carries."""

    def setUp(self):
        ops = {name: _pointwise_op((4, 64), name=name) for name in ("a", "b")}
        self.graph = SimpleNamespace(
            operations=list(ops.values()), try_get_buffer=ops.get
        )
        # What the judge itself accepts for each buffer of this graph.
        _, _, self.judged = get_ncores_for_buffers(self.graph)

    @staticmethod
    def _buf(name, address=None):
        return CoreDivisionBuffer(
            name, 128, [0, 1], core_divisions=[CoreDivision()], address=address
        )

    def test_a_resident_buffer_takes_the_view_the_judge_accepted(self):
        resident, spilled = self._buf("a", address=0), self._buf("b")
        commit_lx_views(self.graph, [resident, spilled], [])
        self.assertEqual(resident.lx_view, self.judged["a"])
        self.assertIsNone(spilled.lx_view)

    def test_a_relayout_source_takes_the_view_its_plan_carries(self):
        core_id = sympy.Symbol("core_id")
        plan = LXRelayoutPlan(
            "a",
            ("b",),
            PerCoreView(((0, 4),), ((0, core_id),), num_cores=4),
            PerCoreView(((0, 2),), ((0, sympy.floor(core_id / 2)),), num_cores=4),
            4,
        )
        self.assertNotEqual(plan.source_view, self.judged["a"])
        source, consumer = self._buf("a", address=0), self._buf("b", address=128)
        # The copy is in the allocation but is not a buffer of the graph:
        # materialize_lx_relayouts gives its destination the plan's view.
        copy = RelayoutCopyBuffer(
            relayout_copy_name("a", 0),
            128,
            [1, 2],
            core_divisions=[CoreDivision()],
            address=256,
            relayout_parent="a",
            group=0,
        )
        commit_lx_views(self.graph, [source, copy, consumer], [plan])
        self.assertIs(source.lx_view, plan.source_view)
        self.assertEqual(consumer.lx_view, self.judged["b"])
        self.assertIsNone(copy.lx_view)
