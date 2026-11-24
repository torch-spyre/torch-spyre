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

"""Explicit initialization of the Spyre device runtime.

Several Python allocation paths call pybind11 entry points
(``spyre_empty_with_layout``, ``spyre_empty_reserved``, ...) directly. Those
calls bypass the ATen dispatcher path whose stream pool self-initializes the
runtime, so when such a call is the first Spyre operation in a process the
runtime must be brought up explicitly first (PR #5179, issue #6129).

This module is internal; it is not the public ``torch_spyre.runtime``
logging namespace.
"""

from __future__ import annotations


def ensure_runtime_initialized() -> None:
    """Start the Spyre device runtime if it is not already running.

    Idempotent: ``_lazy_init()`` returns immediately once the runtime is up,
    and the underlying ``start_runtime()`` is ``std::call_once``.
    """

    import torch

    # torch.spyre is registered by this package's autoload before any code
    # here can run, so both attributes exist unconditionally.
    if not torch.spyre.is_initialized():
        torch.spyre._impl._lazy_init()
