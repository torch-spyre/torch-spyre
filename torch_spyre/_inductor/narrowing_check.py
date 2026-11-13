# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Catch guards that split a declared dynamic range, at the first compile.

A declaration ``x.to("spyre", dynamic={0: {"min", "max", "granularity"}})``
promises one program for every legal size ``min, min + granularity, ..., max``.
A guard that is False for some legal size breaks that promise: those sizes
compile a second program later, silently. This check runs right after each
Spyre compile, tests every guard on a declared size against every legal size,
and reports the ones that reject any.

``TORCH_SPYRE_NARROWING_CHECK`` sets what a report does, the same for every
guard: ``warn`` (default) warns and lets the compile through, ``error`` stops
the compile, ``off`` skips the check. An unknown value falls back to ``warn``
with a logged warning.
"""

import os
import warnings

from torch._guards import TracingContext

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre.ops.warning_types import DeclaredRangeNarrowedWarning

logger = get_inductor_logger("narrowing_check")

MODE_ENV = "TORCH_SPYRE_NARROWING_CHECK"
_MODES = ("warn", "error", "off")
_DEFAULT_MODE = "warn"

# Unknown mode values already warned about, so each is logged once.
_warned_modes: set[str] = set()


class DeclaredRangeNarrowedError(RuntimeError):
    """A guard rejects sizes in a declared range, in ``error`` mode."""


def _mode():
    """The mode from the environment; an unknown value falls back to warn.

    A bad value must not fail a compile, so it is logged once instead.
    """
    mode = os.environ.get(MODE_ENV, _DEFAULT_MODE).lower()
    if mode in _MODES:
        return mode
    if mode not in _warned_modes:
        _warned_modes.add(mode)
        logger.warning(
            "%s=%r is not one of %s; using %s",
            MODE_ENV,
            mode,
            ", ".join(_MODES),
            _DEFAULT_MODE,
        )
    return _DEFAULT_MODE


def declared_symbols(declared):
    """Map each declared symbol's name to its legal sizes.

    ``declared`` is ``{symbol name: (min, max, granularity)}``, as
    ``_declared_dims_of`` returns it for this compile.
    """
    return {
        name: list(range(lo, hi + 1, granularity))
        for name, (lo, hi, granularity) in declared.items()
    }


def rejected_sizes(expr, sym_legal):
    """Return the legal sizes ``expr`` is False for.

    Returns ``[]`` when every legal size passes, and ``None`` when the guard is
    not on exactly one declared symbol or cannot be evaluated. An unevaluable
    guard is skipped: the check must never be the reason a compile fails.
    """
    if len(expr.free_symbols) != 1:
        return None
    (sym,) = expr.free_symbols
    if str(sym) not in sym_legal:
        return None
    bad = []
    for n in sym_legal[str(sym)]:
        try:
            if not bool(expr.xreplace({sym: n})):
                bad.append(n)
        except Exception:  # noqa: BLE001
            return None
    return bad


def _where(sloc):
    """The user line behind a guard, or PyTorch's location if there is none."""
    return sloc.maybe_user_loc or str(sloc).strip("()")


def check_declared_ranges(declared):
    """Report guards that split a declared range. Call after the compile.

    ``declared`` is what ``_declared_dims_of`` returned for this compile.
    """
    mode = _mode()
    if mode == "off":
        return
    ctx = TracingContext.try_get()
    shape_env = getattr(getattr(ctx, "fake_mode", None), "shape_env", None)
    if shape_env is None:
        return
    sym_legal = declared_symbols(declared)
    if not sym_legal:
        return

    reports = []
    for guard in shape_env.guards:
        bad = rejected_sizes(guard.expr, sym_legal)
        if bad:
            reports.append(
                f"Warning! guard {guard.expr} causes a recompile for dim values "
                f"{bad} (at: {_where(guard.sloc)})"
            )
    if not reports:
        return
    if mode == "error":
        raise DeclaredRangeNarrowedError("\n".join(reports))
    for report in reports:
        warnings.warn(report, DeclaredRangeNarrowedWarning, stacklevel=2)
