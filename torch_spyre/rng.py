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

import numpy as np
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


def _as_uint64_keys(keys: Sequence[int] | torch.Tensor) -> np.ndarray:
    """64-bit keys as a uint64 host array (an int64 tensor is read as unsigned)."""
    if isinstance(keys, torch.Tensor):
        return (
            keys.to(device="cpu", dtype=torch.int64).reshape(-1).numpy().view(np.uint64)
        )
    return np.array([int(k) & (2**64 - 1) for k in keys], dtype=np.uint64)


def simon32_key_pairs(
    keys: Sequence[int] | torch.Tensor, device="spyre"
) -> list[torch.Tensor]:
    """The six ``keys`` inputs of ``spyre::simon32x2`` for one 64-bit key per row.

    Each is an int32 ``[rows, 1]`` tensor holding two round keys, inverted:
    ``~k_a & 0xFFFF | (~k_b & 0xFFFF) << 16``. The key schedule runs on the
    host for all rows at once, and the six tensors reach ``device`` in one copy.
    """
    m = np.uint64(_M16)

    def rotl(x, r):
        return ((x << np.uint64(r)) | (x >> np.uint64(16 - r))) & m

    k64 = _as_uint64_keys(keys)
    k = [(k64 >> np.uint64(16 * i)) & m for i in range(4)]
    rk = []
    for i in range(ROUNDS):
        rk.append(k[0])
        tmp = rotl(k[3], 13) ^ k[1]
        tmp ^= rotl(tmp, 15)
        new = (~k[0] & m) ^ tmp ^ np.uint64(int(_Z0[i % 62]) ^ 3)
        k = [k[1], k[2], k[3], new]
    rk_a = np.stack(rk)  # [ROUNDS, rows]
    pairs = (~rk_a[0::2] & m) | ((~rk_a[1::2] & m) << np.uint64(16))
    t = torch.from_numpy(pairs.astype(np.uint32).view(np.int32))
    return list(t.reshape(ROUNDS // 2, -1, 1).to(device).unbind(0))


def simon32_bits(counter: torch.Tensor, key_pairs: Sequence[torch.Tensor]):
    """32 random bits per element: Simon32/64-12 of ``counter`` under each row's key."""
    state = counter
    for pair in key_pairs:
        state = torch.ops.spyre.simon32x2(state, pair)
    return state


def simon32_uniform(counter: torch.Tensor, key_pairs: Sequence[torch.Tensor]):
    """An fp32 uniform in (0, 1) per element, from :func:`simon32_bits`."""
    return torch.ops.spyre.bits_to_uniform(simon32_bits(counter, key_pairs))
