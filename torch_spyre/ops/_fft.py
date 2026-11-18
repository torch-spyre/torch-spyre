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

"""Discrete Fourier transform on Spyre.

``fft`` / ``ifft`` (1-D) and ``fft2`` / ``ifft2`` / ``fftn`` / ``ifftn`` (N-D,
separable). Complex<->real packing and the fp16<->fp32 casts run on the host;
only the real interleaved matmul (``u @ W`` -> ``matmul_fp16_fwd``) runs on
device.

Imported lazily from ``torch_spyre.ops`` (see ``ops/__init__.py``) so that
importing other ``torch_spyre.ops`` submodules does not pull in ``torch``/``os``
or the module-level caches below.
"""

import os
from typing import Optional, Sequence

import torch

__all__ = ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"]

# Compiled real matmul (u @ W), one per device.
_COMPILED: dict = {}
# Interleaved DFT matrix on device, keyed by (device, length, inverse).
_W_CACHE: dict = {}


def _to_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Complex ``(..., n)`` -> real ``(..., 2n)`` as ``[re0, im0, re1, ...]``."""
    return torch.view_as_real(x).reshape(*x.shape[:-1], 2 * x.shape[-1])


def _from_interleaved(v: torch.Tensor, n: int) -> torch.Tensor:
    """Real ``(..., 2n)`` -> complex ``(..., n)`` (inverse of interleaving)."""
    return torch.view_as_complex(v.reshape(*v.shape[:-1], n, 2).contiguous())


def _dft_matrix(n: int, inverse: bool) -> torch.Tensor:
    """``(2n, 2n)`` real interleaved DFT matrix.

    Complex ``W[j, k] = exp(-2pi i jk/n)`` (``conj/n`` for the inverse) expanded
    into real ``[[Re, Im], [-Im, Re]]`` blocks so a complex DFT runs as one real
    matmul on interleaved ``[re0, im0, ...]`` data.
    """
    k = torch.arange(n, dtype=torch.float64)
    theta = 2.0 * torch.pi * torch.outer(k, k) / n
    ang = theta if inverse else -theta
    re, im = torch.cos(ang), torch.sin(ang)
    if inverse:
        re, im = re / n, im / n
    m = torch.empty(2 * n, 2 * n, dtype=torch.float32)
    m[0::2, 0::2], m[0::2, 1::2] = re, im
    m[1::2, 0::2], m[1::2, 1::2] = -im, re
    return m


def _device_dft_matrix(n: int, inverse: bool, device: str) -> torch.Tensor:
    """fp16 DFT matrix on ``device``, built once per (device, n, inverse)."""
    key = (device, n, inverse)
    w = _W_CACHE.get(key)
    if w is None:
        w = _dft_matrix(n, inverse).to(torch.float16).to(device)
        _W_CACHE[key] = w
    return w


# Largest single-transform length run as one direct (2n, 2n) matmul. The
# direct weight is 8*n^2 bytes in HBM; measured on device it is correct and
# fastest up to n=32768, above which it exceeds either the 256 MiB per-core
# access span (MAX_SPAN_BYTES, at low batch) or HBM capacity (at high batch),
# and fp16 accumulation error nears the usable limit. Longer transforms fall
# back to the four-step factorization -- slower (a serial chain of sub-DFTs)
# but it fits and is more accurate. Override with TORCH_SPYRE_FFT_DIRECT_MAX.
_DIRECT_MATMUL_MAX_DEFAULT = 32768


def _direct_matmul_max() -> int:
    raw = os.environ.get("TORCH_SPYRE_FFT_DIRECT_MAX")
    if raw is None:
        return _DIRECT_MATMUL_MAX_DEFAULT
    try:
        v = int(raw)
    except ValueError:
        return _DIRECT_MATMUL_MAX_DEFAULT
    return v if v > 0 else _DIRECT_MATMUL_MAX_DEFAULT


def _direct(x: torch.Tensor, inverse: bool, device: str) -> torch.Tensor:
    """Length-n DFT along the last axis as one real interleaved matmul."""
    n = x.shape[-1]
    u = _to_interleaved(x).to(torch.float16).to(device)  # pack + cast on host
    comp = _COMPILED.get(device)
    if comp is None:
        comp = torch.compile(lambda a, b: a @ b, dynamic=False)
        _COMPILED[device] = comp
    w = _device_dft_matrix(n, inverse, device)
    # fp16 -> fp32 on the host: a bitwidth cast on device reorders within the
    # stick and would corrupt element order.
    return _from_interleaved(comp(u, w).cpu().float(), n)


def _factor(n: int) -> tuple[int, int]:
    """Split n = n1 * n2 with factors near sqrt(n); (1, n) if n is prime."""
    for a in range(int(n**0.5), 1, -1):
        if n % a == 0:
            return a, n // a
    return 1, n


def _twiddle(n1: int, n2: int, inverse: bool) -> torch.Tensor:
    """Four-step twiddle T[k1, n2] = exp(sign * 2pi i k1 n2 / (n1 n2))."""
    k1 = torch.arange(n1)[:, None]
    n2a = torch.arange(n2)[None, :]
    sign = 1.0 if inverse else -1.0
    ang = sign * 2.0 * torch.pi * k1 * n2a / (n1 * n2)
    return torch.polar(torch.ones(n1, n2), ang)


def _four_step(x: torch.Tensor, inverse: bool, device: str) -> torch.Tensor:
    """Cooley-Tukey two-factor split for large n; recurses via _run_1d."""
    n = x.shape[-1]
    n1, n2 = _factor(n)
    batch = x.shape[:-1]
    xm = x.reshape(*batch, n1, n2)
    # Stage A: DFT along the n1 axis (-2) -> A[..., k1, n2].
    a = _run_1d(xm.movedim(-2, -1), inverse, device).movedim(-1, -2)
    # Stage B: twiddle, broadcast over any batch dims.
    a = a * _twiddle(n1, n2, inverse)
    # Stage C: DFT along the n2 axis (-1), then reorder k1 + n1 k2.
    c = _run_1d(a, inverse, device)
    return c.transpose(-2, -1).reshape(*batch, n)


def _run_1d(x: torch.Tensor, inverse: bool, device: str) -> torch.Tensor:
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    n = x.shape[-1]
    threshold = _direct_matmul_max()
    if n > threshold:
        if _factor(n)[0] == 1:
            raise NotImplementedError(
                f"FFT length {n} has a prime factor larger than {threshold}; "
                "not supported (it would need an O(n^2) "
                f"{2 * n}x{2 * n} matrix). Use a composite length."
            )
        return _four_step(x, inverse, device)
    return _direct(x, inverse, device)


def fft(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """1-D DFT along the last axis (real input promoted to complex)."""
    return _run_1d(x, inverse=False, device=device)


def ifft(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """Inverse 1-D DFT along the last axis (includes 1/n)."""
    return _run_1d(x, inverse=True, device=device)


def _run_nd(
    x: torch.Tensor,
    dim: Optional[Sequence[int]],
    inverse: bool,
    device: str,
) -> torch.Tensor:
    """Separable N-D DFT: transform each axis in turn (host movedim)."""
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    dims = tuple(range(x.ndim)) if dim is None else tuple(dim)
    y = x
    for d in dims:
        y = torch.movedim(y, d, -1)
        y = _run_1d(y, inverse, device)
        y = torch.movedim(y, -1, d)
    return y


def fftn(
    x: torch.Tensor,
    dim: Optional[Sequence[int]] = None,
    *,
    device: str = "spyre",
) -> torch.Tensor:
    """N-D DFT over ``dim`` (all axes if ``dim`` is None)."""
    return _run_nd(x, dim, inverse=False, device=device)


def ifftn(
    x: torch.Tensor,
    dim: Optional[Sequence[int]] = None,
    *,
    device: str = "spyre",
) -> torch.Tensor:
    """Inverse N-D DFT over ``dim`` (all axes if ``dim`` is None)."""
    return _run_nd(x, dim, inverse=True, device=device)


def fft2(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """2-D DFT over the last two axes."""
    return _run_nd(x, (-2, -1), inverse=False, device=device)


def ifft2(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """Inverse 2-D DFT over the last two axes."""
    return _run_nd(x, (-2, -1), inverse=True, device=device)
