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

``fft`` / ``ifft`` (1-D) and ``fft2`` / ``ifft2`` / ``fftn`` / ``ifftn`` (N-D)
compute the discrete Fourier transform on Spyre; the N-D transforms are
separable and reuse the 1-D transform along each axis. The device cannot hold
complex tensors, so the complex<->real conversion and the
(bitwidth-changing) fp16<->fp32 cast are done here on the host; only the real
interleaved matmul runs on the device, via the ``spyre::dft`` custom op whose
decomposition lowers it to ``matmul_fp16_fwd`` (see
``torch_spyre/_inductor/{customops,decompositions}.py``).
"""

from typing import Optional, Sequence

import torch

__all__ = ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"]

# Compiled ``spyre::dft`` wrapper, one per device.
_COMPILED: dict = {}


def _to_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Complex ``(..., n)`` -> real ``(..., 2n)`` as ``[re0, im0, re1, ...]``."""
    return torch.view_as_real(x).reshape(*x.shape[:-1], 2 * x.shape[-1])


def _from_interleaved(v: torch.Tensor, n: int) -> torch.Tensor:
    """Real ``(..., 2n)`` -> complex ``(..., n)`` (inverse of interleaving)."""
    return torch.view_as_complex(v.reshape(*v.shape[:-1], n, 2).contiguous())


# Transform lengths above this use the four-step factorization; a single
# (2n, 2n) matmul is O(n^2) in compute and weight memory, so it is only used
# for small n. The crossover is ~4096 on current hardware.
_FOUR_STEP_THRESHOLD = 4096


def _direct(x: torch.Tensor, inverse: bool, device: str) -> torch.Tensor:
    """Length-n DFT along the last axis as one real interleaved matmul."""
    n = x.shape[-1]
    # Interleave and cast to fp16 on the host, then run the matmul on device.
    u = _to_interleaved(x).to(torch.float16).to(device)
    if device not in _COMPILED:
        _COMPILED[device] = torch.compile(
            lambda t, inv: torch.ops.spyre.dft(t, inv), dynamic=False
        )
    # Cast fp16 -> fp32 on the host (.cpu().float()); a bitwidth-changing cast on
    # the device reorders elements within the stick and would corrupt the order.
    v = _COMPILED[device](u, inverse).cpu().float()
    return _from_interleaved(v, n)


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
    """Cooley-Tukey four-step DFT for large n = n1 * n2.

    Replaces one O(n^2) matmul with two smaller DFTs plus an elementwise
    twiddle. The reshape, twiddle and transpose are host complex ops; only the
    two sub-DFTs (reused via :func:`_run_1d`) run on the device, so this also
    recurses automatically when a sub-factor is itself above the threshold.
    """
    n = x.shape[-1]
    n1, n2 = _factor(n)
    batch = x.shape[:-1]
    xm = x.reshape(*batch, n1, n2)
    # Stage A: DFT along the n1 axis (-2) -> A[..., k1, n2].
    a = _run_1d(xm.movedim(-2, -1), inverse, device).movedim(-1, -2)
    # Stage B: twiddle (host complex elementwise), broadcast over any batch dims.
    a = a * _twiddle(n1, n2, inverse)
    # Stage C: DFT along the n2 axis (-1) -> C[..., k1, k2].
    c = _run_1d(a, inverse, device)
    # Output reorder X[..., k1 + n1 k2] = C[..., k1, k2].
    return c.transpose(-2, -1).reshape(*batch, n)


def _run_1d(x: torch.Tensor, inverse: bool, device: str) -> torch.Tensor:
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    n = x.shape[-1]
    if n > _FOUR_STEP_THRESHOLD:
        if _factor(n)[0] == 1:
            raise NotImplementedError(
                f"FFT length {n} is a prime larger than {_FOUR_STEP_THRESHOLD}; "
                "not supported (a single prime-length transform would need an "
                f"O(n^2) {2 * n}x{2 * n} matrix). Use a composite length."
            )
        return _four_step(x, inverse, device)
    return _direct(x, inverse, device)


def fft(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """One-dimensional DFT along the last axis, computed on Spyre.

    Args:
        x: complex (or real, promoted to complex) tensor; the last dimension is
            transformed, leading dimensions are batched.
        device: device to run the matmul on (default ``"spyre"``).

    Returns:
        A ``torch.complex64`` tensor on the host, matching
        :func:`torch.fft.fft` within fp16 accuracy.
    """
    return _run_1d(x, inverse=False, device=device)


def ifft(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """Inverse one-dimensional DFT along the last axis (includes 1/n)."""
    return _run_1d(x, inverse=True, device=device)


def _run_nd(
    x: torch.Tensor,
    dim: Optional[Sequence[int]],
    inverse: bool,
    device: str,
) -> torch.Tensor:
    # An N-D DFT is separable: transform each axis in turn. Each axis is moved
    # to the last position, run through the 1-D transform, and moved back. The
    # movedim and complex handling stay on the host; only the matmuls hit the
    # device.
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
    """N-dimensional DFT over ``dim`` (all axes if ``dim`` is None)."""
    return _run_nd(x, dim, inverse=False, device=device)


def ifftn(
    x: torch.Tensor,
    dim: Optional[Sequence[int]] = None,
    *,
    device: str = "spyre",
) -> torch.Tensor:
    """Inverse N-dimensional DFT over ``dim`` (all axes if ``dim`` is None)."""
    return _run_nd(x, dim, inverse=True, device=device)


def fft2(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """Two-dimensional DFT over the last two axes."""
    return _run_nd(x, (-2, -1), inverse=False, device=device)


def ifft2(x: torch.Tensor, *, device: str = "spyre") -> torch.Tensor:
    """Inverse two-dimensional DFT over the last two axes."""
    return _run_nd(x, (-2, -1), inverse=True, device=device)
