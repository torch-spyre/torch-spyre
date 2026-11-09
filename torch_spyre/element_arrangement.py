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

"""Element order inside a stick for the staggered ``ElementArrangement`` values.

The hardware ``dl16tofp32`` and ``fp32todl16`` conversions leave the elements of
a stick in a fixed, non-sequential order (``DL16_TO_FP32`` and ``FP32_TO_DL16``).
Every stick of a tensor is permuted the same way, so one table per arrangement
describes the whole tensor. Code that reads such a tensor on the host, or that
builds a device tensor to pair with one, should use these tables rather than
re-deriving the pattern.
"""

import torch

from torch_spyre._C import ElementArrangement

# Both staggered arrangements permute a whole 64-element (128-byte) stick.
STICK_ELEMENTS = 64

_HALF = STICK_ELEMENTS // 2


def _dl16_to_fp32_slot_to_logical() -> tuple[int, ...]:
    # fp16 -> fp32 widens a 64-lane stick into two 32-lane fp32 sticks. Slot
    # ``half*32 + k*4 + w`` (half in {0, 1}, k in 0..7, w in 0..3) holds the
    # logical element ``k*8 + half*4 + w``.
    return tuple(
        (s % _HALF // 4) * 8 + (s // _HALF) * 4 + s % 4 for s in range(STICK_ELEMENTS)
    )


def _fp32_to_dl16_slot_to_logical() -> tuple[int, ...]:
    # fp32 -> fp16 narrows two 32-lane fp32 sticks into one 64-lane stick. Slot
    # ``k*8 + j*4 + w`` (j in {0, 1}) holds the logical element ``j*32 + k*4 + w``.
    return tuple(
        (s % 8 // 4) * _HALF + (s // 8) * 4 + s % 4 for s in range(STICK_ELEMENTS)
    )


_SLOT_TO_LOGICAL = {
    ElementArrangement.DL16_TO_FP32: _dl16_to_fp32_slot_to_logical(),
    ElementArrangement.FP32_TO_DL16: _fp32_to_dl16_slot_to_logical(),
}


def stick_permutation(ea: ElementArrangement) -> tuple[int, ...]:
    """Return ``perm`` with ``perm[slot] = logical index`` for one stick of ``ea``.

    ``perm`` has ``STICK_ELEMENTS`` entries and is the same for every stick of a
    tensor. ``ea`` must be ``DL16_TO_FP32`` or ``FP32_TO_DL16``; ``STANDARD`` is
    the identity and needs no table.
    """
    try:
        return _SLOT_TO_LOGICAL[ea]
    except KeyError:
        raise ValueError(f"{ea} is not a staggered element arrangement") from None


def destagger_host(t: torch.Tensor, ea: ElementArrangement) -> torch.Tensor:
    """Reorder a host copy of a staggered device tensor into logical order, in place.

    ``t`` must hold the raw device buffer (a D2H copy that ignored ``ea``), be
    contiguous, and have a multiple of ``STICK_ELEMENTS`` elements. ``STANDARD`` is a
    no-op. Returns ``t``.
    """
    if ea == ElementArrangement.STANDARD:
        return t
    perm = torch.tensor(stick_permutation(ea), dtype=torch.long)
    sticks = t.reshape(-1, STICK_ELEMENTS)
    # slot ``s`` holds logical ``perm[s]``, so logical ``l`` sits in slot ``inv[l]``.
    t.copy_(sticks[:, perm.argsort()].reshape(t.shape))
    return t
