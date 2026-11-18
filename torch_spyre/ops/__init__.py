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

"""Public ops for torch_spyre.

The discrete Fourier transforms (``fft`` / ``ifft`` / ``fft2`` / ``ifft2`` /
``fftn`` / ``ifftn``) live in :mod:`torch_spyre.ops._fft` and are re-exported
here lazily: importing another ``torch_spyre.ops`` submodule (e.g. the eager
path) does not load ``_fft`` or its ``torch``/``os`` imports and matrix caches.
"""

__all__ = ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"]


def __getattr__(name: str):
    if name in __all__:
        from . import _fft

        return getattr(_fft, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Keep the real submodules (eager, fallbacks, ...) visible alongside the
    # lazily re-exported FFT names, so tab-completion is not narrowed to __all__.
    return sorted(set(globals()) | set(__all__))
