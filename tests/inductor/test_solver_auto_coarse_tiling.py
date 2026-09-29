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

import dataclasses
import functools
import pytest
import os
import sys
import torch
import unittest

from collections.abc import Callable, Sequence
from typing import Optional

from unittest.mock import patch

from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering

from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.loop_info import CoarseTileInfo
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
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
    codegen will emit, whichever pass stamped them.
    """

    tiling: dict[str, CoarseTileInfo] = {}

    def __call__(self, graph: GraphLowering) -> None:
        super().__call__(graph)
        type(self).tiling = {
            op.get_name(): op.loop_info
            for op in graph.operations
            if getattr(op, "loop_info", None) is not None
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
        # Raises the "gate is missing" NotImplementedError before compiling.
        if auto_tiling:
            # TODO: Implement coarse tiling configuration
            raise NotImplementedError("unified-tiling: config.auto_coarse_tiling")

        cpu_result = case.model(())(*(arg.to("cpu") for arg in case.args))

        CollectTilingPasses.tiling = {}
        # TODO: Patch coarse tiling config here
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
        seq_len, in_dim, hidden_dim, out_dim = 8192, 256, 1024, 256
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
        """Mark the combos that cannot pass until the tile search is built.

        These entries are never edited again: each combo stops xfailing on its
        own, the moment the last unbuilt piece on its path lands, because
        ``expected_unimplemented`` keys on the exception rather than on a list
        maintained by hand.  A combo turning green is the signal to delete its
        row here.
        """
        decorators = []
        if params["solver_method"] == "cpsat":
            decorators.append(
                unittest.skipUnless(_HAS_ORTOOLS, "the cpsat solver needs ortools")
            )
        if params["tiling_mode"] in ("auto", "explicit_auto"):
            decorators.append(expected_unimplemented)
        return decorators

    def run_case(self, params: dict, factory: Callable) -> None:
        """Body of one generated method: build the model, check its contract."""
        self._CHECKS[params["tiling_mode"]](
            self, factory(self), params["solver_method"]
        )
