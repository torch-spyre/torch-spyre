# Copyright 2025 The Torch-Spyre Authors.
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

"""Keep PyTorch's compiler bisector away from Spyre's own nested compiles.

A Spyre eager op is a compiled kernel: the first call of ``aten.mul`` on a
Spyre tensor runs ``torch.compile(_op_frame(aten.mul))``. For
``torch._inductor.compiler_bisector.CompilerBisector`` those kernels are the
device's eager mode, the reference the program under test is compared against.
They are not part of that program.

The bisector does not know that. While it is active it

* overrides the backend of **every** ``torch.compile`` call, so the per-op
  kernel is itself compiled with ``backend="eager"``, runs the op eagerly, and
  dispatches back into the kernel that called it; and
* counts **every** ``disable_subsystem`` query, so a per-op kernel compiled
  while a subsystem is under test has that subsystem disabled too, and its
  applications are added to the program's count.

The first makes the bisector blame ``eager`` for a program that is always
correct. The second makes a disabled lowering fall back into a kernel whose own
lowering is disabled as well.

:func:`suspended` marks a region of the current thread as exempt. Inside it the
bisector reports no active backend, no config change, and no disabled
subsystem, and it counts nothing. Every place torch-spyre compiles and runs a
nested kernel enters it.

PyTorch has no such region today, so the three queries are wrapped here. The
shape is one upstream could adopt as ``CompilerBisector.suspended()``; if the
installed PyTorch has that attribute, it is used and nothing is patched.
"""

import functools
import threading

_state = threading.local()
_install_lock = threading.Lock()
_installed = False
_upstream_suspended = None

# The three queries through which the bisector reaches a compile, and what
# each must answer inside an exempt region.
_QUERIES = {
    "get_backend": None,  # torch.compile keeps the caller's backend
    "get_config_change": None,  # no bisector config override is applied
    "disable_subsystem": False,  # nothing is disabled, and nothing is counted
}


def is_suspended() -> bool:
    """Whether the current thread is inside a :func:`suspended` region."""
    return getattr(_state, "depth", 0) > 0


def _install() -> None:
    """Wrap the bisector's queries so they honor :func:`is_suspended`.

    Idempotent and thread-safe. Deferred to the first exempt region so that
    importing torch-spyre does not import the bisector.
    """
    global _installed, _upstream_suspended
    with _install_lock:
        if _installed:
            return
        from torch._inductor.compiler_bisector import CompilerBisector

        upstream = getattr(CompilerBisector, "suspended", None)
        if upstream is not None:
            _upstream_suspended = upstream
        else:
            for name, exempt_answer in _QUERIES.items():
                original = getattr(CompilerBisector, name).__func__

                def query(cls, *args, _original=original, _answer=exempt_answer, **kw):
                    if is_suspended():
                        return _answer
                    return _original(cls, *args, **kw)

                functools.update_wrapper(query, original)
                setattr(CompilerBisector, name, classmethod(query))
        _installed = True


class suspended:
    """Context manager: exempt the current thread's compiles from bisection.

    Re-entrant and thread-local. A class with ``__enter__``/``__exit__`` and
    not a generator-based context manager, because it sits on the eager-op hot
    path and a generator costs about a microsecond per call.
    """

    __slots__ = ("_upstream",)

    def __enter__(self) -> "suspended":
        if not _installed:
            _install()
        self._upstream = None
        if _upstream_suspended is not None:
            self._upstream = _upstream_suspended()
            self._upstream.__enter__()
        _state.depth = getattr(_state, "depth", 0) + 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _state.depth -= 1
        if self._upstream is not None:
            self._upstream.__exit__(exc_type, exc, tb)
