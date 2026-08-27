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

"""Cost units for the joint work-division + LX-layout SA optimizer.

**Determinism**: microsecond quantities are mapped to a fixed-point integer scale
by a *single* deterministic rounding step, so an accumulated score is
bit-for-bit reproducible with no float non-determinism.
"""

from __future__ import annotations

import math

# Microseconds are the universal currency; every µs quantity is converted to an
# integer on this scale by exactly one rounding step, so accumulation is pure
# integer. 1e6 gives picosecond resolution -- ample for the smallest memory
# terms -- while Python's arbitrary-precision ints keep large sums exact.
US_FIXED_POINT_SCALE = 1_000_000


def to_fixed_us(us: float) -> int:
    """Map a non-negative microsecond quantity to the fixed-point integer scale
    with a single deterministic round-half-up step.

    Round-half-up on non-negative inputs is order-independent and platform-stable
    (no banker's rounding), which is what the determinism guarantee needs. An
    infinite cost (an infeasible split) is a caller error, flagged rather than
    silently mapped.
    """
    if not math.isfinite(us) or us < 0.0:
        raise ValueError(f"cost must be finite and non-negative, got {us!r}")
    return int(us * US_FIXED_POINT_SCALE + 0.5)


def hbm_bytes_per_us() -> float:
    """HBM bandwidth as bytes per microsecond, sourced from the native cost model
    (``_HBM_BW_GBS`` GB/s x 1000) so the memory objective and the cost model's
    own traffic term use the identical constant.

    Imported lazily so the fixed-point helper above stays importable without
    pulling in torch.
    """
    from torch_spyre._inductor import work_division  # noqa: PLC0415

    return float(work_division._HBM_BW_GBS) * 1000.0
