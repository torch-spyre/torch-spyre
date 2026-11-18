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
import threading
import warnings
from collections import OrderedDict
from typing import Optional, Sequence

import torch

__all__ = ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"]

# Compiled real matmul (u @ W), one per device; built once under a lock.
_COMPILED: dict = {}
_COMPILED_LOCK = threading.Lock()

# Interleaved DFT matrix on device, keyed by (device, length, inverse). An LRU
# bounded by total bytes (TORCH_SPYRE_FFT_W_CACHE_BYTES); each entry is
# 8*n**2 bytes. Build + insert run under the lock so two callers never build
# the same multi-gigabyte matrix at once.
_W_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_W_CACHE_LOCK = threading.Lock()


def _to_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Complex ``(..., n)`` -> real ``(..., 2n)`` as ``[re0, im0, re1, ...]``."""
    return torch.view_as_real(x).reshape(*x.shape[:-1], 2 * x.shape[-1])


def _from_interleaved(v: torch.Tensor, n: int) -> torch.Tensor:
    """Real ``(..., 2n)`` -> complex ``(..., n)`` (inverse of interleaving)."""
    return torch.view_as_complex(v.reshape(*v.shape[:-1], n, 2).contiguous())


# Rows of the (2n, 2n) matrix assembled per block (one complex DFT row = two
# interleaved real rows), to keep the host working set at O(block*n).
_DFT_MATRIX_ROW_BLOCK = 256


def _build_device_dft_matrix(
    n: int, inverse: bool, device: str, block: int = _DFT_MATRIX_ROW_BLOCK
) -> torch.Tensor:
    """``(2n, 2n)`` fp16 interleaved DFT matrix, assembled directly on ``device``.

    Complex ``W[j, k] = exp(-2pi i jk/n)`` (``conj`` for the inverse) expanded
    into real ``[[Re, Im], [-Im, Re]]`` blocks so a complex DFT runs as one real
    matmul on interleaved ``[re0, im0, ...]`` data. The matrix is kept
    unit-magnitude: the inverse ``1/n`` is applied on the host after the matmul
    (see :func:`_direct`), because baking it in would make the fp16 entries
    ``~1/n`` -- subnormal below the fp16 normal minimum (~6.1e-5) for n>16384,
    where they lose precision or flush to zero.

    Built a row block at a time from length-n ``cos``/``sin`` tables indexed by
    the exact integer phase ``(j*k) % n``, so no ``(n, n)`` float64 outer
    product (or float32 full matrix) is ever materialized -- host peak is
    ``O(block*n)`` rather than ``O(n**2)``. The exact integer index also avoids
    the float rounding a large ``j*k`` would suffer once it exceeds the mantissa.
    """
    phase = (2.0 * torch.pi / n) * torch.arange(n, dtype=torch.float64)
    cos_t, sin_t = torch.cos(phase), torch.sin(phase)
    im_sign = 1.0 if inverse else -1.0  # exp(+) inverse, exp(-) forward
    k = torch.arange(n, dtype=torch.int64)
    out = torch.empty(2 * n, 2 * n, dtype=torch.float16, device=device)
    for j0 in range(0, n, block):
        j = torch.arange(j0, min(j0 + block, n), dtype=torch.int64).unsqueeze(1)
        idx = (j * k) % n  # (rows, n) exact integer phase index
        re = cos_t[idx].to(torch.float16)
        im = (im_sign * sin_t[idx]).to(torch.float16)
        rows = re.shape[0]
        blk = torch.empty(2 * rows, 2 * n, dtype=torch.float16)
        blk[0::2, 0::2], blk[0::2, 1::2] = re, im
        blk[1::2, 0::2], blk[1::2, 1::2] = -im, re
        out[2 * j0 : 2 * j0 + 2 * rows] = blk.to(device)
    return out


def _entry_bytes(key: tuple) -> int:
    """HBM bytes held by a cached ``(device, n, inverse)`` matrix: ``8*n**2``."""
    _, n, _ = key
    return 8 * n * n  # (2n, 2n) fp16


def _w_cache_budget_bytes() -> int:
    """LRU byte budget; default holds forward+inverse at the direct ceiling."""
    default = 2 * 8 * _DIRECT_MATMUL_MAX_DEFAULT**2
    raw = os.environ.get("TORCH_SPYRE_FFT_W_CACHE_BYTES")
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return v if v > 0 else default


def _evict_w_cache_locked(keep: tuple) -> None:
    """Evict least-recently-used entries until within budget (caller holds lock).

    ``keep`` (the just-built entry) is never evicted, so a single matrix larger
    than the whole budget is still returned -- just not retained across calls.
    """
    budget = _w_cache_budget_bytes()
    total = sum(_entry_bytes(k) for k in _W_CACHE)
    for k in list(_W_CACHE):  # OrderedDict iterates least-recently-used first
        if total <= budget:
            break
        if k == keep:
            continue
        del _W_CACHE[k]
        total -= _entry_bytes(k)


def _device_dft_matrix(n: int, inverse: bool, device: str) -> torch.Tensor:
    """fp16 DFT matrix on ``device``, built once per (device, n, inverse).

    Returned from a byte-bounded LRU cache; build and insert run under a lock so
    concurrent callers do not build the same matrix twice.
    """
    key = (device, n, inverse)
    with _W_CACHE_LOCK:
        w = _W_CACHE.get(key)
        if w is not None:
            _W_CACHE.move_to_end(key)
            return w
        w = _build_device_dft_matrix(n, inverse, device)
        _W_CACHE[key] = w
        _evict_w_cache_locked(keep=key)
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
        warnings.warn(
            f"TORCH_SPYRE_FFT_DIRECT_MAX={raw!r} is not an integer; using the "
            f"default {_DIRECT_MATMUL_MAX_DEFAULT}.",
            stacklevel=2,
        )
        return _DIRECT_MATMUL_MAX_DEFAULT
    if v <= 0:
        warnings.warn(
            f"TORCH_SPYRE_FFT_DIRECT_MAX={v} must be positive; using the "
            f"default {_DIRECT_MATMUL_MAX_DEFAULT}.",
            stacklevel=2,
        )
        return _DIRECT_MATMUL_MAX_DEFAULT
    return v


def _direct(x: torch.Tensor, inverse: bool, device: str) -> torch.Tensor:
    """Length-n DFT along the last axis as one real interleaved matmul.

    The input is prescaled by its max magnitude on the host so the fp16 cast
    and the fp16 matmul output stay in range -- the transform is linear, so the
    scale is reapplied afterward. Without it the unnormalized DC term reaches
    ``n*|x|`` and overflows fp16 at the ceiling, and an input ``|x| > 65504``
    overflows on the cast itself. The inverse ``1/n`` is applied here too, in
    fp32, since the matrix is unit-magnitude (see :func:`_build_device_dft_matrix`).
    """
    n = x.shape[-1]
    s = x.abs().amax()
    xs = x / s if s > 0 else x
    u = _to_interleaved(xs).to(torch.float16)  # (*batch, 2n), pack + cast on host
    # Flatten every leading dim into one so the matmul always sees a 2D
    # (B, 2n) @ (2n, 2n); otherwise each distinct batch rank/shape recompiles
    # the dynamic=False callable (fftn over several axes + four-step sub-DFTs
    # would hit Dynamo's recompile limit quickly).
    batch = u.shape[:-1]
    u = u.reshape(-1, u.shape[-1]).to(device)
    comp = _COMPILED.get(device)
    if comp is None:
        with _COMPILED_LOCK:
            comp = _COMPILED.get(device)
            if comp is None:
                comp = torch.compile(lambda a, b: a @ b, dynamic=False)
                _COMPILED[device] = comp
    w = _device_dft_matrix(n, inverse, device)
    # Cast fp16 -> fp32 on the host (.cpu() then .float(), never .float().cpu()).
    # On device a bitwidth cast produces a staggered element arrangement (fp16
    # packs 64 elements per 128-byte stick, fp32 packs 32) -- a normal device
    # characteristic, not a bug (docs/source/user_guide/tensors_and_layouts.md);
    # but a device->host copy reads it as standard layout and returns a
    # permutation. The result returns to the host anyway, and casting after the
    # fp16 D2H copy moves half the bytes of an on-device upcast. A device-side
    # upcast (torch-spyre #5495 insert_ea_shuffle) could revisit this.
    v = _from_interleaved(comp(u, w).cpu().float(), n)  # (B, n) complex
    post = (s / n) if inverse else s
    return v.reshape(*batch, n) * post


def _factor(n: int) -> tuple[int, int]:
    """Split n = n1 * n2 with factors near sqrt(n); (1, n) if n is prime."""
    for a in range(int(n**0.5), 1, -1):
        if n % a == 0:
            return a, n // a
    return 1, n


def _largest_prime_factor(n: int) -> int:
    """Largest prime factor of ``n`` (``n`` itself when prime)."""
    largest, m, d = 1, n, 2
    while d * d <= m:
        while m % d == 0:
            largest, m = d, m // d
        d += 1
    return max(largest, m)


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
        # Validate the full factorization of the requested length up front. The
        # four-step split only helps if every prime factor is <= threshold; a
        # length like 2*65537 would otherwise fail deep in the recursion naming
        # the sub-length (65537), not the length the caller asked for. Because
        # this runs before recursing, the top-level call names the original n,
        # and sub-DFTs (factors of an already-validated n) never re-raise.
        pf = _largest_prime_factor(n)
        if pf > threshold:
            raise NotImplementedError(
                f"FFT length {n} has prime factor {pf} larger than the direct "
                f"ceiling {threshold}; the four-step factorization cannot reduce "
                f"it below a single O(n^2) matmul. Use a length whose prime "
                f"factors are all <= {threshold}, or raise "
                f"TORCH_SPYRE_FFT_DIRECT_MAX."
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
