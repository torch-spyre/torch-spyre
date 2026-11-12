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

"""Counter-based random numbers on Spyre: Simon32/64 with 12 rounds.

Each value depends only on (key, counter), so it does not depend on the batch
layout or on how the work is split across cores: ``bits[i, t] = simon(key[i],
t)``. The cipher runs on device as six ``spyre::simon32x2`` calls of two rounds
each; the key schedule runs on the host, once per key.

Typical use, with one 64-bit key per row and the token index as the counter::

    pairs = simon32_key_pairs(keys, device="spyre")      # host, per step
    counter = torch.arange(n, dtype=torch.int32).view(1, n).to("spyre")
    u = torch.compile(simon32_uniform)(counter, pairs)    # [rows, n] in (0, 1)
"""

from typing import Sequence

import torch

_M16 = 0xFFFF
ROUNDS = 12
_Z0 = "11111010001001010110000111001101111101000100101011000011100110"


def _rotl16(x: int, r: int) -> int:
    return ((x << r) | (x >> (16 - r))) & _M16


def simon32_round_keys(key: int, rounds: int = ROUNDS) -> list[int]:
    """The round keys of Simon32/64 for a 64-bit key."""
    k = [(key >> (16 * i)) & _M16 for i in range(4)]
    rk = []
    for i in range(rounds):
        rk.append(k[0])
        tmp = _rotl16(k[3], 13) ^ k[1]
        tmp ^= _rotl16(tmp, 15)
        new = (~k[0] & _M16) ^ tmp ^ int(_Z0[i % 62]) ^ 3
        k = [k[1], k[2], k[3], new]
    return rk


def simon32_key_pairs(
    keys: Sequence[int] | torch.Tensor, device="spyre"
) -> list[torch.Tensor]:
    """The six ``keys`` inputs of ``spyre::simon32x2`` for one 64-bit key per row.

    Each is an int32 ``[rows, 1]`` tensor holding two round keys, inverted:
    ``~k_a & 0xFFFF | (~k_b & 0xFFFF) << 16``.
    """
    if isinstance(keys, torch.Tensor):
        keys = [int(k) & (2**64 - 1) for k in keys.tolist()]
    pairs: list[list[int]] = [[] for _ in range(ROUNDS // 2)]
    for key in keys:
        rk = simon32_round_keys(key)
        for j in range(ROUNDS // 2):
            v = (~rk[2 * j] & _M16) | ((~rk[2 * j + 1] & _M16) << 16)
            pairs[j].append(v - 2**32 if v >= 2**31 else v)
    return [torch.tensor(p, dtype=torch.int32).view(-1, 1).to(device) for p in pairs]


def simon32_bits(counter: torch.Tensor, key_pairs: Sequence[torch.Tensor]):
    """32 random bits per element: Simon32/64-12 of ``counter`` under each row's key."""
    state = counter
    for pair in key_pairs:
        state = torch.ops.spyre.simon32x2(state, pair)
    return state


def simon32_uniform(counter: torch.Tensor, key_pairs: Sequence[torch.Tensor]):
    """An fp32 uniform in (0, 1) per element, from :func:`simon32_bits`."""
    return torch.ops.spyre.bits_to_uniform(simon32_bits(counter, key_pairs))
