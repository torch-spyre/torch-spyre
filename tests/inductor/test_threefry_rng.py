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

"""Threefry-4x16-13 on device: spyre::threefry4x16_stage, torch_spyre.rng_threefry.

Checked exactly against a host model of the cipher written here (Random123
structure, 16-bit words, Aaron Sawdey's rotation table, parity 0x1BD1).
"""

import numpy as np
import pytest
import torch
import torch._dynamo as dynamo

from torch_spyre import rng_threefry

M16 = 0xFFFF
ROT16 = ((5, 13), (6, 10), (6, 14), (12, 2), (3, 10), (8, 6), (12, 5), (9, 10))


def _threefry(t, key):
    """x0 | x1 << 16 of Threefry-4x16-13, counter (t, t >> 16, 0, 0), key (k, k >> 16, 0, 0)."""
    k = [key & M16, (key >> 16) & M16, 0 * key, 0 * key]
    ks = k + [0x1BD1 ^ k[0] ^ k[1] ^ k[2] ^ k[3]]
    c = [t & M16, (t >> 16) & M16, 0 * t, 0 * t]
    x = [(c[i] + k[i]) & M16 for i in range(4)]
    for r in range(13):
        ra, rb = ROT16[r % 8]
        i, j, p, q = (0, 1, 2, 3) if r % 2 == 0 else (0, 3, 2, 1)
        x[i] = (x[i] + x[j]) & M16
        x[j] = (((x[j] << ra) | (x[j] >> (16 - ra))) & M16) ^ x[i]
        x[p] = (x[p] + x[q]) & M16
        x[q] = (((x[q] << rb) | (x[q] >> (16 - rb))) & M16) ^ x[p]
        if r % 4 == 3:
            s = (r + 1) // 4
            x = [
                (x[n] + ks[(s + n) % 5] + (s if n == 3 else 0)) & M16 for n in range(4)
            ]
    return x[0] | (x[1] << 16)


def _as_int32(a):
    return torch.from_numpy(
        np.asarray(a, dtype=np.int64).astype(np.uint32).view(np.int32)
    )


def _run(fn, *args):
    dynamo.reset()
    return torch.compile(fn, dynamic=False)(*[a.to("spyre") for a in args]).cpu()


@pytest.mark.parametrize(
    "key, want",
    [
        (0, [0x0C183244, 0x0926458A, 0xF3FA15A5, 0x0116B993]),
        (0x12345678, [0xEBD01FCE, 0x65E5F314, 0xFF627CAB, 0x35F12619]),
    ],
)
def test_host_model_known_values(key, want):
    # Pins the host model to a reference implementation that reproduces
    # Random123's Threefry-4x32 known-answer vectors with the same code.
    got = _threefry(np.arange(4, dtype=np.int64), np.int64(key))
    assert [int(v) for v in got] == want


@pytest.mark.parametrize("n", [256, 130])
def test_threefry4x16_bits(n):
    keys = np.array([0, 0xFFFFFFFF, 0x12345678, 0x0000FFFF], dtype=np.int64)
    t = np.arange(n, dtype=np.int64)
    want = np.stack([_threefry(t, k) for k in keys])
    counter = torch.arange(n, dtype=torch.int32).view(1, n)
    key = _as_int32(keys).view(-1, 1)
    got = _run(rng_threefry.threefry4x16_bits, counter, key)
    assert got.shape == (len(keys), n)
    assert torch.equal(got, _as_int32(want))
