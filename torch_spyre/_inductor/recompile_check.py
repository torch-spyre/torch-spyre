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

"""Report recompiles that break a declared size range, and keep a compile ledger.

A declaration ``x.to("spyre", dynamic={0: {"min", "max", "granularity"}})``
promises one program for every size ``min, min + granularity, ..., max``. A
recompile while such an input has one of those sizes breaks the promise: a size
branch, a guard the narrowing check could not see, or a guard it only warned
about. This hook wraps the callback Dynamo builds for every cache miss, so it
sees each compile, and reports those recompiles.

torch-spyre's own per-op eager programs (``torch_spyre/ops/eager.py``) and
PyTorch's internals recompile per size by design. Their code lives inside the
torch or torch-spyre package, so they are recognised by file and ignored.

``TORCH_SPYRE_RECOMPILE_CHECK`` sets the mode:

- ``warn`` (default): warn on a recompile that breaks a declared range.
- ``error``: stop such a recompile.
- ``all``: warn on every recompile of user code, declared or not. This also
  shows a declaration lost through ``clone``/``view``/a graph break.
- ``off``: report nothing.

An unknown value falls back to ``warn`` with a logged warning.

Declared inputs are found among the function's arguments and locals, including
one level inside a list or tuple. A declared tensor inside a dict, an attribute
or deeper nesting is not recognised here; ``all`` mode still reports the
recompile, and the narrowing check still sees the declaration.

The ledger records every user compile whatever the mode.
"""

import inspect
import os
import time
import warnings

import torch

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.ops.warning_types import RecompileWarning

logger = get_inductor_logger("recompile_check")

MODE_ENV = "TORCH_SPYRE_RECOMPILE_CHECK"
_MODES = ("warn", "error", "all", "off")
_DEFAULT_MODE = "warn"

# The private Dynamo names the hook depends on, and the signature it wraps.
# A PyTorch upgrade that removes or changes them leaves Dynamo alone and
# logs a warning, rather than wrapping something the hook no longer matches.
CALLBACK_FROM_STANCE_PARAMS = ("callback",)

_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__)) + os.sep
_SPYRE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

# {"name (file:line)": {"compiles": int, "seconds": float}}
_ledger: dict[str, dict[str, float]] = {}

# Unknown mode values already warned about, so each is logged once.
_warned_modes: set[str] = set()


class RecompileError(RuntimeError):
    """A recompile broke a declared size range, in ``error`` mode."""


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


def _is_internal(filename):
    path = os.path.abspath(filename)
    return path.startswith(_TORCH_DIR) or path.startswith(_SPYRE_DIR)


def _declaration_of(tensor):
    """The tensor's ``{dim: {min, max, granularity}}`` declaration, or None.

    Only a Spyre tensor can carry a declaration, so any other tensor is None
    without a call into the runtime.
    """
    if tensor.device.type != DEVICE_NAME:
        return None
    from torch_spyre import _C

    return _C.get_reserved_dims(tensor)


def _broken_promise(frame):
    """Describe the first input whose size is one it was declared to serve.

    Returns None when no input is declared, or every declared input has a size
    outside its promised list (the declaration refuses those separately).
    """
    for arg, value in frame.f_locals.items():
        tensors = value if isinstance(value, (list, tuple)) else (value,)
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                continue
            try:
                decl = _declaration_of(t)
            except Exception:  # noqa: BLE001  never let the check break a call
                return None
            for d, spec in (decl or {}).items():
                legal = range(spec["min"], spec["max"] + 1, spec["granularity"])
                if t.shape[d] in legal:
                    return (
                        f"input '{arg}' dim {d} is {t.shape[d]}, one of its "
                        f"declared sizes {spec['min']}..{spec['max']} "
                        f"step {spec['granularity']}"
                    )
    return None


def _reasons(entries, frame, callback):
    """The guards that failed, as PyTorch words them; [] if unavailable."""
    from torch._dynamo.eval_frame import innermost_fn
    from torch._dynamo.guards import get_and_maybe_log_recompilation_reasons

    backend = innermost_fn(callback)
    for arg in (entries, entries[0]):  # a list on torch 2.13, one entry before
        try:
            return get_and_maybe_log_recompilation_reasons(
                arg, frame, backend, skip_logging=True
            )
        except Exception:  # noqa: BLE001  the reason is a nicety, never fatal
            continue
    return []


def compile_ledger():
    """A copy of the ledger: compiles and seconds per user function."""
    return {name: dict(row) for name, row in _ledger.items()}


def reset_compile_ledger():
    _ledger.clear()


def _report(frame, callback, code, mode):
    """The text to report for this miss, or None when there is nothing to say."""
    from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list

    entries = _debug_get_cache_entry_list(code)
    if not entries:  # a first compile is normal
        return None
    broken = _broken_promise(frame)
    if broken is None and mode != "all":
        return None
    name = f"{code.co_name} ({code.co_filename}:{code.co_firstlineno})"
    reasons = "; ".join(_reasons(entries, frame, callback)) or "unknown"
    why = f" Declared range broken: {broken}." if broken else ""
    return (
        f"Warning! {name} recompiled (compile {len(entries) + 1}).{why} "
        f"Reason: {reasons}"
    )


def _wrap(callback):
    def wrapped(frame, *args, **kwargs):
        code = frame.f_code
        if _is_internal(code.co_filename):
            return callback(frame, *args, **kwargs)

        mode = _mode()
        report = None if mode == "off" else _report(frame, callback, code, mode)
        if report and mode == "error":
            raise RecompileError(report)
        if report:
            warnings.warn(report, RecompileWarning, stacklevel=2)

        name = f"{code.co_name} ({code.co_filename}:{code.co_firstlineno})"
        start = time.perf_counter()
        try:
            return callback(frame, *args, **kwargs)
        finally:
            row = _ledger.setdefault(name, {"compiles": 0, "seconds": 0.0})
            row["compiles"] += 1
            row["seconds"] += time.perf_counter() - start

    # Dynamo compares backends through this attribute; keep the original's.
    wrapped._torchdynamo_orig_backend = getattr(
        callback, "_torchdynamo_orig_backend", callback
    )
    return wrapped


def _missing_dynamo_hooks():
    """Why the hook cannot be installed on this torch, or None if it can."""
    import torch._dynamo.eval_frame as ef
    from torch._C._dynamo import eval_frame as c_ef

    original = getattr(ef, "_callback_from_stance", None)
    if original is None:
        return "torch._dynamo.eval_frame._callback_from_stance is absent"
    try:
        params = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        params = None
    if params != CALLBACK_FROM_STANCE_PARAMS and not getattr(
        original, "_spyre_recompile_check", False
    ):
        return (
            f"torch._dynamo.eval_frame._callback_from_stance has parameters "
            f"{params}, the hook was written for {CALLBACK_FROM_STANCE_PARAMS}"
        )
    if getattr(c_ef, "_debug_get_cache_entry_list", None) is None:
        return "torch._C._dynamo.eval_frame._debug_get_cache_entry_list is absent"
    return None


def install_recompile_check():
    """Wrap Dynamo's per-miss callback once. Safe to call more than once.

    Installed only over the private names and signature it was written
    against. On a torch where they are gone or changed, Dynamo is left alone
    and a warning says the recompile check is off.
    """
    import torch._dynamo.eval_frame as ef

    problem = _missing_dynamo_hooks()
    if problem is not None:
        logger.warning(
            "The recompile check (%s) is not installed: %s. Update or remove "
            "torch_spyre/_inductor/recompile_check.py for this torch version",
            MODE_ENV,
            problem,
        )
        return
    original = ef._callback_from_stance
    if getattr(original, "_spyre_recompile_check", False):
        return

    def callback_from_stance(callback):
        cb = original(callback)
        if cb in (False, None):
            return cb
        return _wrap(cb)

    callback_from_stance._spyre_recompile_check = True
    ef._callback_from_stance = callback_from_stance
