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

"""Automated coarse tiling: hint preservation and hint-free tile discovery."""

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
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd

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


# One buffer's coarse-tile fingerprint: the trip counts of the loop nest it
# sits in, outermost level first.  An op at an outer level of a deeper nest
# carries a prefix of its group's counts -- a drain left outside a two-level
# nest reads (4,) where the interior ops read (4, 2).
_Counts = tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _Level:
    """One level of a loop nest, and the hint scope that asked for it.

    The label is what makes a pinned level distinguishable from a discovered
    one by *identity* rather than by position.  Position cannot do it: a
    level's place in the nest is decided by hint id ordering (``_hints_levels``
    in wsr/coarse_tile_hints.py sorts by it), so reading "the first k levels
    are the pinned ones" silently assumes the tile search mints ids above the
    caller's -- true of span overflow's ``_SPAN_OVERFLOW_HINT_ID = 10000``, but
    a fact about one namespace choice, not about what a pin means.  Checking
    labels lets a discovered level land *outside* a pinned one without failing
    the test, while still catching a pin that was dropped or re-tiled.

    ``hint_id`` identifies the scope; ``dim`` is the name that scope tiled --
    the caller's own named dim for a pin, ``"_span_overflow"`` for a level the
    compiler added.  ``dim`` is what turns "some level was pinned to 2" into
    "the level *on S* was pinned to 2", which is the assertion a caller
    actually wants: a hint binding to the wrong axis divides the right number
    of ways.  Both are ``None`` on a level that could not be attributed to a
    scope (see ``_label_nest``).
    """

    count: int
    hint_id: Optional[int] = None
    dim: Optional[str] = None

    def __repr__(self) -> str:
        label = self.dim or (f"hint{self.hint_id}" if self.hint_id is not None else "?")
        return f"{label}:{self.count}"


_Nest = tuple[_Level, ...]


def _trip_counts(nest: _Nest) -> _Counts:
    """Drop the labels: the plain outermost-first trip counts of ``nest``."""
    return tuple(level.count for level in nest)


def _is_subsequence(counts: _Counts, nest: _Counts) -> bool:
    """True if ``counts`` is ``nest`` with zero or more levels left out.

    Subsequence rather than prefix: an op at an outer level of a deeper nest
    drops the *inner* levels and so does read as a prefix, but a reduction's
    fill op keeps only the output levels outer to the reduction (see
    ``_compute_fill_loop_info_planned``), which can leave out a level in the
    middle.  Prefix would call that legitimate nest a violation.
    """
    remaining = iter(nest)
    return all(count in remaining for count in counts)


def _group_hints(ops: Sequence) -> tuple[DimHint, ...]:
    """One hint per level of a loop group, outermost first.

    The group, not the op, is the unit here.  ``loop_count`` is a group-level
    fact -- every member carries the whole nest, including the levels it is
    invariant at -- so a single op's ``dim_hints`` can be *shorter* than the
    nest and is not a list the counts can be zipped against.  This unions
    across the group the way ``_hints_levels`` does, keeping a scope as soon
    as *some* member is tiled by it, which is exactly the rule that decided
    the group's levels.

    Two filters mirror that function: a hint the op is broadcast against
    (``loop_var is None``) and a split of 1 both produce no loop level, so
    neither can label one.
    """
    best: dict[int, DimHint] = {}
    for op in ops:
        for h in getattr(op, "dim_hints", []):
            prev = best.get(h.hint_id)
            if (
                prev is None
                or prev.loop_var is None
                or (prev.split_count == 1 and h.split_count > 1)
            ):
                best[h.hint_id] = h
    return tuple(
        sorted(
            (h for h in best.values() if h.loop_var is not None and h.split_count != 1),
            key=lambda h: h.hint_id,
        )
    )


def _label_nest(op, group_hints: tuple[DimHint, ...]) -> _Nest:
    """Pair ``op``'s trip counts with the hints that produced them.

    The group's hints are the label list, and the pairing is positional:
    both sequences are ordered outermost-first, so equal lengths make it
    unambiguous.  The op's own hints are deliberately not consulted -- being
    a subset of the group's, they can only agree on length by being the same
    list, and where they *would* differ (below) the op has none at all.

    The lengths disagree for a *trimmed* nest: a reduction's fill op keeps
    only the output levels outer to the reduction
    (``_compute_fill_loop_info_planned``), as does the ``reduce_copy`` built
    from it.  Neither is constructed through ``copy_op_metadata``, so neither
    carries ``dim_hints`` to fall back on, and their levels come back
    unlabelled rather than guessed at from a subset that merely fits.  That
    is safe as long as nothing keys on them: a pin still shows up labelled on
    the ops that carry the untrimmed nest, and the count-only checks in
    ``_check_hints_preserved`` cover the trimmed op.  A reduction-tiled case
    is what would make a real handler for them worth writing.
    """
    counts = tuple(int(count) for count in op.loop_info.loop_count)
    if len(group_hints) != len(counts):
        return tuple(_Level(count=count) for count in counts)
    return tuple(
        _Level(
            count=count,
            hint_id=h.hint_id,
            dim=h.dim_names[0] if h.dim_names else None,
        )
        for h, count in zip(group_hints, counts)
    )


def _label_tiling(operations: Sequence) -> dict[str, _Nest]:
    """Every coarse-tiled op in ``operations``, mapped to its labelled nest."""
    tiled = [op for op in operations if getattr(op, "loop_info", None) is not None]

    def group_key(op) -> tuple[int, ...]:
        # Group index is loop_group_id[0]; the rest of the tuple is nesting
        # depth, which a trimmed nest truncates (_compute_fill_loop_info_planned
        # keeps the prefix), so keying on the whole tuple would split a group.
        return tuple(op.loop_info.loop_group_id[:1])

    by_group: dict[tuple[int, ...], list] = {}
    for op in tiled:
        by_group.setdefault(group_key(op), []).append(op)
    group_hints = {key: _group_hints(ops) for key, ops in by_group.items()}
    return {op.get_name(): _label_nest(op, group_hints[group_key(op)]) for op in tiled}


@dataclasses.dataclass(frozen=True)
class _TilingCase:
    """One model plus the tiling contract asserted against it.

    body:
        The unhinted model.  Pins are wrapped around it at compile time, so the
        same callable serves all three hint modes.
    args:
        Device tensors passed to the compiled model.
    named_dims:
        Per-argument named-dim labels, positionally aligned with ``args``.
        ``spyre_hint`` addresses dimensions by these names, so they are
        declared and attached for the hinted and partial modes and omitted
        entirely for the unhinted one -- a graph with no named dims is what
        "no hints" actually looks like to the compiler.
    pins:
        The hint scopes the *hinted* mode wraps around ``body``, outermost
        first, and the whole of that mode's expectation: a pin is a
        ``(dim, count)`` and the nest it prescribes is those counts in that
        order, so a separate ``expected`` beside it could only restate them or
        contradict them.  A model hinted by its own Spyre decomposition instead
        of by the caller pins nothing and therefore cannot state a contract
        this way; it needs an expectation field of its own, which is one more
        thing to settle when the SDPA case below is unblocked.
    partial_pins / partial_named_dims:
        The same for the *partial* mode, where the caller pins a strict subset
        of the tiling and leaves the rest to the compiler.  What must survive
        is again each pin's own ``(dim, count)``, on whatever level the
        compiler ends up giving it.  ``partial_named_dims``
        defaults to ``named_dims``, and is overridden only when withholding a
        *name* is the only way to withhold a hint -- again, the
        decomposition-hinted case, where the caller cannot delete a scope the
        compiler emitted.
    """

    body: Callable[..., torch.Tensor]
    args: tuple[torch.Tensor, ...]
    named_dims: tuple[Sequence[str], ...]
    pins: tuple[tuple[str, int], ...]
    partial_pins: tuple[tuple[str, int], ...]
    atol: float
    rtol: float
    partial_named_dims: Optional[tuple[Sequence[str], ...]] = None

    @property
    def hinted_nest(self) -> _Counts:
        """The loop nest ``pins`` prescribes: their counts, outermost first."""
        return tuple(count for _, count in self.pins)

    def dims_for(self, hint_mode: str) -> Optional[tuple[Sequence[str], ...]]:
        if hint_mode == "unhinted":
            return None
        if hint_mode == "partial" and self.partial_named_dims is not None:
            return self.partial_named_dims
        return self.named_dims


def _apply_pins(pins: tuple[tuple[str, int], ...], body: Callable, *args):
    """Run ``body`` inside one ``spyre_hint`` scope per pin, outermost first.

    One scope per dimension: ``assign_dim_hints`` raises ``NotImplementedError``
    on a ``spyre_hint`` naming more than one, and the nesting order is what
    fixes the loop-nest order (hint ids increase inwards).
    """
    if not pins:
        return body(*args)
    (dim, tiles), rest = pins[0], pins[1:]
    with spyre_hint(num_tiles_per_dim={dim: tiles}):
        return _apply_pins(rest, body, *args)


class CollectTilingPasses(CustomPreSchedulingPasses):
    """Pre-scheduling pipeline that records the applied tiling once it is done.

    ``torch_spyre._inductor.patches.enable_spyre_context`` installs
    ``CustomPreSchedulingPasses`` itself, so observing its result means
    substituting this subclass for it.  ``coarse_tile`` stamps ``loop_info``
    well before the scheduler is built, so reading it here sees the final plan.

    ``dim_hints`` is an input the tiling passes read and never clear, so it is
    still on the ops here and each level can be labelled with the scope that
    asked for it -- which is the only thing that distinguishes a caller's pin
    from a level the compiler found on its own.
    """

    tiling: dict[str, _Nest] = {}

    def __call__(self, graph: GraphLowering) -> None:
        super().__call__(graph)
        type(self).tiling = _label_tiling(graph.operations)


class AutomatedCoarseTilingTests(
    unittest.TestCase, metaclass=_ParameterizedScratchpadMeta
):
    """model x hint_mode x solver, one generated method per combination.

    The metaclass expands ``parameter_models`` against ``parameter_axes`` and
    routes each generated method through ``run_case``; ``case_decorators``
    marks the combos that cannot pass until the tile search exists.
    """

    def setUp(self):
        torch.manual_seed(0xAFFE)
        # Named dims live in module state that outlives a compile, so a stale
        # name left by another test would silently bind a hint here.
        _pnd.reset()
        self.addCleanup(_pnd.reset)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    # ------------------------------------------------------------------
    # Compile and observe
    # ------------------------------------------------------------------
    def _compile_and_collect(
        self,
        case: "_TilingCase",
        hint_mode: str,
        pins: tuple[tuple[str, int], ...],
        *,
        layout_solver: str,
        auto_tiling: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, _Nest]]:
        """Compile ``case`` and return (cpu_result, device_result, tiling)."""
        # Raises the "gate is missing" NotImplementedError before compiling.
        if auto_tiling:
            # TODO: Implement coarse tiling configuration
            raise NotImplementedError("unified-tiling: config.auto_coarse_tiling")

        # declare the tensor dimensions
        named_dims = case.dims_for(hint_mode)
        if named_dims is not None:
            for arg, dims in zip(case.args, named_dims):
                for dim, size in zip(dims, arg.shape):
                    _pnd.declare_tensor_dim(dim, int(size))
            for arg, dims in zip(case.args, named_dims):
                _pnd.name_tensor_dims(arg, list(dims))

        cpu_result = case.body(*(arg.to("cpu") for arg in case.args))

        if pins:

            def model(*args):
                return _apply_pins(pins, case.body, *args)
        else:
            model = case.body
        CollectTilingPasses.tiling = {}
        # TODO: Patch coarse tiling config here
        # force_disable_caches belongs to torch's inductor config, not Spyre's;
        # CustomPreSchedulingPasses is a plain module attribute that
        # enable_spyre_context re-imports per compile, so it is swapped with
        # patch.object rather than a config knob.
        with (
            t_inductor_config.patch(force_disable_caches=True),
            ts_inductor_config.patch(
                allow_all_ops_in_lx_planning=True,
                layout_solver=layout_solver,
            ),
            patch.object(ts_passes, "CustomPreSchedulingPasses", CollectTilingPasses),
        ):
            device_result = torch.compile(model, fullgraph=True)(*case.args).to("cpu")

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
    # Reading the labels
    # ------------------------------------------------------------------
    def _classify_levels(
        self, tiling: dict[str, _Nest], pins: tuple[tuple[str, int], ...]
    ) -> tuple[dict[int, _Level], dict[int, _Level]]:
        """Split the applied levels into the caller's pins and the rest.

        ``pins[i]`` is the ``(dim, count)`` the caller's *i*-th ``spyre_hint``
        scope asked for, outermost first.  That indexing is the
        identification: ``get_id`` is a per-compile counter starting at 0 and
        ``spyre_hint`` is its only caller, so with ``fullgraph=True`` the
        scopes wrapped around the model own hint ids ``0..len(pins)-1`` in
        nesting order, and every other id on a level was minted by the
        compiler.  (A model hinted by its own decomposition would break that
        split -- it also mints low ids -- which is one more reason the SDPA
        case is left out below.)

        Identifying by id rather than by ``dim`` is deliberate even though the
        name is right there: the compiler is free to add a *second* level on an
        axis the caller already pinned (a finer division of it), and that level
        is a discovered one, not a broken pin.

        Asserts each pinned level still divides the dim it named, by the count
        it named, wherever in the nest it ended up.  Returns the ``(pinned,
        discovered)`` levels, keyed by hint id, so the caller can say which of
        the two it expected.
        """
        pinned = dict(enumerate(pins))
        seen_pinned: dict[int, _Level] = {}
        discovered: dict[int, _Level] = {}
        for name, nest in sorted(tiling.items()):
            for level in nest:
                if level.hint_id is None:
                    continue
                if level.hint_id not in pinned:
                    discovered[level.hint_id] = level
                    continue
                dim, count = pinned[level.hint_id]
                self.assertEqual(
                    level.count,
                    count,
                    f"{name} tiles the level pinned by hint_{level.hint_id} "
                    f"{level.count} ways, not the pinned {count} "
                    f"(its nest is {list(nest)})",
                )
                # A hint that bound to the wrong axis still divides by the
                # right number, so the count alone cannot see it.
                if level.dim is not None:
                    self.assertEqual(
                        level.dim,
                        dim,
                        f"{name}: the hint_{level.hint_id} scope pinned "
                        f"'{dim}', but its level tiles '{level.dim}' "
                        f"(its nest is {list(nest)})",
                    )
                seen_pinned[level.hint_id] = level
        return seen_pinned, discovered

    # ------------------------------------------------------------------
    # The three contracts
    # ------------------------------------------------------------------
    def _check_hints_preserved(self, case: _TilingCase, solver: str) -> None:
        """Hints are applied exactly: every level asked for, no level invented."""
        cpu, device, tiling = self._compile_and_collect(
            case, "hinted", case.pins, layout_solver=solver, auto_tiling=False
        )
        expected = case.hinted_nest
        self.assertTrue(
            tiling,
            "no op was coarse-tiled: the hints were dropped before coarse_tile "
            f"(expected the nest {list(case.pins)})",
        )
        # Two count-only claims, kept beside the keyed ones below as the single
        # witness here that does not depend on the labelling: a level the
        # labeller could not attribute is invisible to every keyed check.
        nests = {name: _trip_counts(nest) for name, nest in tiling.items()}
        for name, counts in sorted(nests.items()):
            self.assertTrue(
                _is_subsequence(counts, expected),
                f"{name} is tiled {counts}, which is not the hinted nest "
                f"{expected} with levels left out",
            )
        self.assertIn(
            expected,
            set(nests.values()),
            f"no op carries the full hinted nest {expected}; "
            f"the applied tiling was {nests}",
        )
        # The rest is keyed on the hint scopes themselves: with the tile search
        # off they are the only thing that may tile anything, so every level is
        # accounted for by a pin, dividing the axis that pin named.
        seen_pinned, discovered = self._classify_levels(tiling, case.pins)
        self.assertEqual(
            sorted(seen_pinned),
            list(range(len(case.pins))),
            f"the applied tiling {tiling} does not carry one level per "
            f"hint: {len(case.pins)} scopes were opened around the model",
        )
        self.assertFalse(
            discovered,
            f"levels {list(discovered.values())} were invented by the "
            f"compiler, but only the {len(case.pins)} hinted ones were "
            f"asked for (the applied tiling was {tiling})",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_tiling_discovered(self, case: "_TilingCase", solver: str) -> None:
        """With no hints at all, the compiler picks a tiling by itself."""
        cpu, device, tiling = self._compile_and_collect(
            case, "unhinted", (), layout_solver=solver, auto_tiling=True
        )
        self.assertTrue(
            tiling,
            "Auto tiling is on and no hints were given, but no op was "
            "coarse-tiled -- the tile search found nothing to do",
        )
        self._assert_matches_cpu(case, device, cpu)

    def _check_partial_hints_preserved(self, case: "_TilingCase", solver: str) -> None:
        """Pinned levels survive verbatim; the compiler fills in the rest.

        Both halves are checked by label, not by position, so the contract is
        the one a pin actually carries -- *this dimension, divided this many
        ways* -- and not "and outside everything the tile search adds".  Where
        the compiler nests its own levels relative to a pin is its choice to
        make: it may put them inside a pin, outside one, or in a separate loop
        group over ops the pins never covered, and only the numerics
        (``_assert_matches_cpu``) can call any of those wrong.
        """
        cpu, device, tiling = self._compile_and_collect(
            case,
            "partial",
            case.partial_pins,
            layout_solver=solver,
            auto_tiling=True,
        )
        self.assertTrue(tiling, "no op was coarse-tiled: the pins were dropped")
        seen_pinned, discovered = self._classify_levels(tiling, case.partial_pins)
        self.assertEqual(
            sorted(seen_pinned),
            list(range(len(case.partial_pins))),
            f"the applied tiling {tiling} lost a pinned level: the pins "
            f"{list(case.partial_pins)} should all still be there",
        )
        self.assertTrue(
            discovered,
            f"the pins {list(case.partial_pins)} survived but nothing was "
            f"added: the tile search left every unpinned dimension untiled "
            f"({tiling})",
        )
        self._assert_matches_cpu(case, device, cpu)

    # ------------------------------------------------------------------
    # Models.  Each returns the model, its named dims and the tiling contract,
    # defined once and reused across every hint_mode and solver.
    # ------------------------------------------------------------------
    def _softmax_case(self) -> "_TilingCase":
        """softmax(dim=0) over (512, 1024), dims R (reduced) x C.

        One level: C divided 4 ways, tiling the eight ops of the lowered
        softmax into a single group and leaving the graph output untiled.  The
        other axis, R, is the reduced one; hinting it as a second level does
        compile and is *numerically wrong* today -- the tiled max and sum drain
        through coarse_tile_combine/reduce_copy and land further from CPU than
        the output magnitude itself -- so C is the whole prescribed plan.  The
        partial mode pins that same single level; what it leaves to the
        compiler is R, plus any finer division of C.
        """
        return _TilingCase(
            body=functools.partial(torch.softmax, dim=0),
            args=(torch.rand((512, 1024), dtype=torch.float16, device=DEVICE_NAME),),
            named_dims=(["R", "C"],),
            pins=(("C", 4),),  # Reduction axis is not tiled for now
            partial_pins=(("C", 4),),
            # A good run lands at 2e-5 on outputs of order 1/512; the
            # reduction-tiled one lands at 3e-3, and this has to separate them.
            atol=5e-4,
            rtol=0.02,
        )

    def _mlp_case(self) -> "_TilingCase":
        """Two-layer MLP (Linear -> silu -> Linear), dims S x Din x Dh x Dout.

        Two levels: S divided 2 ways outside Dout divided 2 ways.  Both are
        free (output) axes -- Din is the first GEMM's reduction and Dh the
        second's, and pinning Dh instead of Dout compiles into a
        coarse_tile_combine/reduce_copy pair whose result is two orders of
        magnitude off CPU.  The partial mode pins only S, leaving Dout for the
        compiler to find.
        """
        seq_len, in_dim, hidden_dim, out_dim = 128, 256, 1024, 256
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
            body=mlp,
            args=args,
            named_dims=(
                ["S", "Din"],
                ["Dh", "Din"],
                ["Dh"],
                ["Dout", "Dh"],
                ["Dout"],
            ),
            pins=(("S", 2), ("Dout", 2)),
            partial_pins=(("S", 2),),
            atol=0.02,
            rtol=0.05,
        )

    def _swiglu_case(self) -> "_TilingCase":
        """SwiGLU (two parallel Linears -> silu(gate) * up), dims S x Din x Dh.

        Two levels: S divided 2 ways outside Dh divided 4 ways.  Unlike the
        MLP's, this Dh is a free axis the whole way through -- it is the N
        dimension of both GEMMs and the layout of every activation -- so the
        entire chain, both restickified weights included, lands in one
        two-level nest.  Both weights must carry the *same* label for it: name
        them apart and the inner level binds only to the gate branch, which
        still compiles and is wrong by 300x the correctly-tiled error.  The
        partial mode pins only S.
        """
        seq_len, in_dim, hidden_dim = 128, 256, 1024
        fc_gate = torch.nn.Linear(in_dim, hidden_dim).half()
        fc_up = torch.nn.Linear(in_dim, hidden_dim).half()

        def swiglu(x, w_gate, b_gate, w_up, b_up):
            gate = torch.nn.functional.linear(x, w_gate, b_gate)
            up = torch.nn.functional.linear(x, w_up, b_up)
            return torch.nn.functional.silu(gate) * up

        args = (
            torch.randn(seq_len, in_dim, dtype=torch.float16).to(DEVICE_NAME),
            fc_gate.weight.to(DEVICE_NAME),
            fc_gate.bias.to(DEVICE_NAME),
            fc_up.weight.to(DEVICE_NAME),
            fc_up.bias.to(DEVICE_NAME),
        )
        return _TilingCase(
            body=swiglu,
            args=args,
            named_dims=(
                ["S", "Din"],
                ["Dh", "Din"],
                ["Dh"],
                ["Dh", "Din"],
                ["Dh"],
            ),
            pins=(("S", 2), ("Dh", 4)),
            partial_pins=(("S", 2),),
            atol=0.02,
            rtol=0.05,
        )

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------
    _CHECKS = {
        "hinted": _check_hints_preserved,
        "unhinted": _check_tiling_discovered,
        "partial": _check_partial_hints_preserved,
    }

    parameter_axes = {"hint_mode": tuple(_CHECKS), "solver_method": ("cpsat",)}

    # SDPA is omitted: it is the one model whose hints come from the compiler
    # using SDPA in this test suite requires resolution of
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
        if params["hint_mode"] in ("unhinted", "partial"):
            decorators.append(expected_unimplemented)
        return decorators

    def run_case(self, params: dict, factory: Callable) -> None:
        """Body of one generated method: build the model, check its contract."""
        self._CHECKS[params["hint_mode"]](self, factory(self), params["solver_method"])
