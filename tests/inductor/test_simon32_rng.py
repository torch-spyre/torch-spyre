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

"""The Simon32/64 device RNG: spyre::simon32x2, spyre::bits_to_uniform, torch_spyre.rng.

Every check is exact against a host model of the cipher written here,
independently of the device code.
"""

import numpy as np
import pytest
import torch
import torch._dynamo as dynamo

from torch_spyre import rng

M16 = 0xFFFF


def _rotl(x, r):
    return ((x << r) | (x >> (16 - r))) & M16


def _rounds(state, keys):
    """Simon rounds on int64 arrays of x << 16 | y with plain (not inverted) keys."""
    x, y = (state >> 16) & M16, state & M16
    for k in keys:
        x, y = y ^ (_rotl(x, 1) & _rotl(x, 8)) ^ _rotl(x, 2) ^ k, x
    return (x << 16) | y


def _as_int32(a):
    return torch.from_numpy(
        np.asarray(a, dtype=np.int64).astype(np.uint32).view(np.int32)
    )


def _run(fn, *args):
    dynamo.reset()
    return torch.compile(fn, dynamic=False)(*[a.to("spyre") for a in args]).cpu()


def _scalar_key_pairs(key):
    """The six key pairs of one key, from the scalar key schedule."""
    rk = rng.simon32_round_keys(key)
    return [(~rk[2 * j] & M16) | ((~rk[2 * j + 1] & M16) << 16) for j in range(6)]


@pytest.mark.parametrize("rows", [1, 300])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_key_pairs_match_scalar_schedule(rows, as_tensor):
    rng_ = np.random.default_rng(3)
    keys = [int(k) for k in rng_.integers(0, 2**64, rows, dtype=np.uint64)]
    keys[: min(rows, 4)] = [0, 2**63 - 1, 2**63, 2**64 - 1][: min(rows, 4)]
    arg = (
        torch.tensor([k - 2**64 if k >= 2**63 else k for k in keys])
        if as_tensor
        else keys
    )
    got = rng.simon32_key_pairs(arg, device="cpu")
    assert len(got) == 6
    for j, pair in enumerate(got):
        assert pair.shape == (rows, 1) and pair.dtype == torch.int32
        want = [_scalar_key_pairs(k)[j] for k in keys]
        assert pair.view(-1).numpy().view(np.uint32).tolist() == want


@pytest.mark.parametrize("shape", [(3, 256), (4, 130)])
def test_simon32x2_two_rounds(shape):
    rng_ = np.random.default_rng(1)
    state = rng_.integers(0, 2**32, shape, dtype=np.uint64).astype(np.int64)
    pairs = rng_.integers(0, 2**32, (shape[0], 1), dtype=np.uint64).astype(np.int64)
    state.flat[:4] = [0, 0xFFFFFFFF, 0x80000000, 0x0000FFFF]
    pairs[0, 0] = 0xFFFFFFFF
    k_a, k_b = (~pairs) & M16, (~(pairs >> 16)) & M16
    want = np.stack(
        [_rounds(state[i], [k_a[i, 0], k_b[i, 0]]) for i in range(shape[0])]
    )
    got = _run(torch.ops.spyre.simon32x2, _as_int32(state), _as_int32(pairs))
    assert torch.equal(got, _as_int32(want))


@pytest.mark.parametrize("n", [256, 130])
def test_simon32_bits_counter_mode(n):
    keys = [0, 2**64 - 1, 0x0123456789ABCDEF]
    pairs = rng.simon32_key_pairs(keys, device="cpu")
    counter = torch.arange(n, dtype=torch.int32).view(1, n)
    t = np.arange(n, dtype=np.int64)
    want = np.stack([_rounds(t, rng.simon32_round_keys(k)) for k in keys])

    def fn(counter, *pairs):
        return rng.simon32_bits(counter, pairs)

    got = _run(fn, counter, *pairs)
    assert got.shape == (len(keys), n)
    assert torch.equal(got, _as_int32(want))


def test_bits_to_uniform():
    rng_ = np.random.default_rng(2)
    bits = rng_.integers(0, 2**32, (2, 512), dtype=np.uint64).astype(np.int64)
    bits.flat[:4] = [0, 0xFFFFFFFF, 0x000001FF, 0xFFFFFE00]
    v = (bits >> 9) & 0x7FFFFF
    want = torch.from_numpy(((v + 0.5) * 2.0**-23).astype(np.float32))
    got = _run(torch.ops.spyre.bits_to_uniform, _as_int32(bits))
    assert got.dtype == torch.float32
    assert torch.equal(got, want)
    assert got.min() > 0 and got.max() < 1


def test_simon32_uniform_end_to_end():
    keys = [7, 11]
    n = 1024
    pairs = rng.simon32_key_pairs(keys, device="cpu")
    counter = torch.arange(n, dtype=torch.int32).view(1, n)

    def fn(counter, *pairs):
        return rng.simon32_uniform(counter, pairs)

    got = _run(fn, counter, *pairs)
    t = np.arange(n, dtype=np.int64)
    bits = np.stack([_rounds(t, rng.simon32_round_keys(k)) for k in keys])
    want = (((bits >> 9) & 0x7FFFFF) + 0.5) * 2.0**-23
    assert torch.equal(got, torch.from_numpy(want.astype(np.float32)))
