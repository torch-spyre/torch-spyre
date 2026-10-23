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

"""An op's core divisions as a space: its tiling options, each with the
:class:`WorkDivisionContext` of its per-tile frame, and per option a factor
domain per axis. The graph builder lists a small space's points as the node's
menu instead of enumerating the legal divisions; a strategy filters them, or
posts legality as constraints (``LazyStrategy``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import sympy

from torch_spyre._inductor import work_division as wd
from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision, TileSpec


@dataclass(frozen=True)
class DivisionOption:
    """One tiling the op may take, with its per-tile frame."""

    tiling: TileSpec
    tile_splits: tuple[tuple[sympy.Symbol, int], ...]
    context: wd.WorkDivisionContext


@dataclass(eq=False)
class DivisionSpace:
    """An op's divisions as per-axis choices under each tiling option: what a
    node's candidates are drawn from, whichever strategy encodes them."""

    op: Any
    options: list[DivisionOption]
    max_cores: int
    _points: Optional[list[tuple[int, dict]]] = field(default=None, repr=False)
    _divisions: dict = field(default_factory=dict, repr=False)

    @property
    def axes(self) -> list[sympy.Symbol]:
        return list(dict.fromkeys(a for o in self.options for a in o.context.axes))

    def domain(self, axis: sympy.Symbol) -> list[int]:
        """Every factor ``axis`` takes under some option (1 where it is not an
        axis of that option's frame)."""
        values = {1}
        for o in self.options:
            if axis in o.context.axes:
                values.update(o.context.factor_domain(axis))
        return sorted(values)

    def points(self) -> list[tuple[int, dict]]:
        """``(option, splits)`` over each option's factor domains within the
        core budget: a superset of the legal divisions, cheap to list, that
        the strategy tabulates per-division quantities over. Legality is
        posted separately; a quantity's value at an illegal point is never
        read."""
        if self._points is None:
            out = []
            for k, o in enumerate(self.options):
                axes = o.context.axes
                domains = [o.context.factor_domain(a) or [1] for a in axes]
                for combo in itertools.product(*domains):
                    if math.prod(combo) <= self.max_cores:
                        out.append((k, dict(zip(axes, combo))))
            self._points = out
        return self._points

    def division(self, option: int, splits: dict) -> CoreDivision:
        key = (option, tuple(sorted((str(a), v) for a, v in splits.items())))
        if key not in self._divisions:
            o = self.options[option]
            self._divisions[key] = wd._core_division(
                self.op, dict(splits), o.tiling, o.tile_splits
            )
        return self._divisions[key]
