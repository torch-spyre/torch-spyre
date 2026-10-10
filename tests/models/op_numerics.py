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
"""Op-specific device placement, CPU references and comparators for model ops.

An ``OpAdapter`` in ``op_registry`` may opt into three hooks defined here:

- ``arg_arrangements``: the ``ElementArrangement`` each positional tensor arg
  must have on Spyre (``to_spyre_with_arrangement``).
- ``reference``: a CPU function computing a high-precision reference instead
  of running the op itself on CPU (``scaled_mm_reference``).
- ``compare``: a comparator replacing the default atol/rtol check
  (``assert_matmul_close``).
"""

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------


def to_spyre_with_arrangement(
    t: torch.Tensor, arrangement: str, device: torch.device
) -> torch.Tensor:
    """Copy a CPU tensor to Spyre with the named ``ElementArrangement``.

    ``QFP8WT`` is the [2, 64] 2D-stick KERNEL layout ``_scaled_mm`` expects for
    its weight; it is produced by the same DMA helper the model loader uses.
    Every other arrangement keeps the host dim order and only sets the
    arrangement. The arrangement is read back so a silently canonicalized
    layout fails here rather than as a numerics mismatch later.
    """
    from torch_spyre._C import (
        ElementArrangement,
        SpyreTensorLayout,
        copy_tensor,
        spyre_empty_with_layout,
    )
    from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel, _ensure_spyre_runtime

    expected = getattr(ElementArrangement, arrangement)
    _ensure_spyre_runtime()
    if expected == ElementArrangement.QFP8WT:
        dst = _dma_to_spyre_fp8_kernel(t)
    else:
        stl = SpyreTensorLayout(
            list(t.shape),
            list(t.stride()),
            t.dtype,
            list(range(t.dim())),
            expected,
        )
        dst = spyre_empty_with_layout(t.size(), t.stride(), t.dtype, stl, device)
        copy_tensor(t, dst, non_blocking=False)

    actual = dst.device_tensor_layout().element_arrangement
    assert actual == expected, (
        f"tensor shape={list(t.shape)} dtype={t.dtype} was placed with "
        f"{actual}, expected {expected}"
    )
    return dst


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatmulReference:
    """FP32 reference for a matmul-like op.

    value:     the exact-as-possible result, computed in float32.
    magnitude: the same contraction over absolute values (|A| @ |B|, scaled
               by |scales|, plus |bias|). The rounding error of any summation
               order of a dot product is bounded by a multiple of it, so it
               is the natural per-element error scale even when the dot
               product itself cancels to ~0.
    out_dtype: the dtype the op under test must return.
    """

    value: torch.Tensor
    magnitude: torch.Tensor
    out_dtype: torch.dtype


def scaled_mm_reference(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> MatmulReference:
    """aten._scaled_mm computed as an FP32 matmul on the dequantized inputs.

    FP8 -> FP32 is exact, so the only error left in the reference is FP32
    accumulation, far below what the device produces.
    """
    assert scale_result is None, "scale_result is not supported by the reference"
    a = mat1.float()
    b = mat2.float()
    sa = scale_a.float()
    sb = scale_b.float()
    value = (a @ b) * sa * sb
    magnitude = (a.abs() @ b.abs()) * sa.abs() * sb.abs()
    if bias is not None:
        value = value + bias.float()
        magnitude = magnitude + bias.float().abs()
    return MatmulReference(
        value=value,
        magnitude=magnitude,
        out_dtype=out_dtype if out_dtype is not None else mat1.dtype,
    )


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

# Default thresholds of assert_matmul_close. Measured on Spyre for
# _scaled_mm with signed U[-1, 1) FP8 inputs (M=4, K=256..12800): the worst
# element error is 0.0035 x magnitude, the relative Frobenius error 1.1% and
# the cosine similarity 0.99995. A mis-arranged operand gives >= 0.018,
# >= 0.68 and <= 0.77 respectively.
MATMUL_ELEM_TOL = 2.0**-7
MATMUL_FRO_TOL = 0.03
MATMUL_MIN_COS = 0.999


def assert_matmul_close(
    ref: MatmulReference,
    got: torch.Tensor,
    *,
    case_name: str,
    description: Optional[str],
    elem_tol: float = MATMUL_ELEM_TOL,
    fro_tol: float = MATMUL_FRO_TOL,
    min_cos: float = MATMUL_MIN_COS,
) -> None:
    """Compare a device matmul result against an FP32 ``MatmulReference``.

    All of the following must hold:

    1. metadata: ``got`` has the reference shape and ``ref.out_dtype``.
    2. finite:   ``got`` has no NaN/Inf.
    3. element:  |got - ref| <= elem_tol * magnitude + eps(out_dtype) * |ref|
                 for every element. The first term is the accumulation error
                 bound of a dot product; the second covers the final rounding
                 to out_dtype. Unlike atol/rtol this scales with K and with
                 the input/scale magnitudes, and stays meaningful for
                 elements that cancel to ~0.
    4. global:   ||got - ref||_F / ||ref||_F <= fro_tol and
                 cos(got, ref) >= min_cos, which catch a systematic bias or a
                 permuted result even when each element is within bound.
    """
    failures = []
    if tuple(got.shape) != tuple(ref.value.shape):
        failures.append(
            f"shape: expected {tuple(ref.value.shape)}, got {tuple(got.shape)}"
        )
    if got.dtype != ref.out_dtype:
        failures.append(f"dtype: expected {ref.out_dtype}, got {got.dtype}")
    if failures:
        _raise(case_name, description, failures, [])

    g = got.float()
    r = ref.value
    tiny = torch.finfo(torch.float32).tiny
    stats = []

    n_nonfinite = int((~torch.isfinite(g)).sum())
    if n_nonfinite:
        failures.append(f"finite: {n_nonfinite} NaN/Inf elements")

    err = (g - r).abs()
    bound = elem_tol * ref.magnitude + torch.finfo(ref.out_dtype).eps * r.abs()
    ratio = err / bound.clamp_min(tiny)
    worst = int(ratio.argmax())
    worst_idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(worst), r.shape))
    n_bad = int((err > bound).sum())
    worst_line = (
        f"worst element {worst_idx}: got={g.flatten()[worst]:.6g} "
        f"ref={r.flatten()[worst]:.6g} |A||B|={ref.magnitude.flatten()[worst]:.6g} "
        f"err/bound={ratio.flatten()[worst]:.3g}"
    )
    if n_bad:
        failures.append(
            f"element: {n_bad}/{r.numel()} elements exceed "
            f"{elem_tol:.3g} x |A||B| + eps x |ref|; {worst_line}"
        )
    else:
        stats.append(worst_line)

    rel_fro = float(err.norm() / r.norm().clamp_min(tiny))
    cos = float(torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0))
    (failures if rel_fro > fro_tol else stats).append(
        f"relative Frobenius error: {rel_fro:.4g} (limit {fro_tol:.3g})"
    )
    (failures if cos < min_cos else stats).append(
        f"cosine similarity: {cos:.6f} (limit {min_cos})"
    )

    if failures:
        _raise(case_name, description, failures, stats)


def _raise(case_name, description, failures, stats) -> None:
    lines = "\n".join(f"  FAIL {f}" for f in failures)
    passed = "\n".join(f"  ok   {s}" for s in stats)
    raise AssertionError(
        f"{case_name} FAILED: output does not match the FP32 reference\n"
        f"{lines}\n{passed}\n"
        f"location: {description}\n"
    )
