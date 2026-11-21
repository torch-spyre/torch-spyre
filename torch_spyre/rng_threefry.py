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

"""Counter-based random numbers on Spyre: Threefry-4x16-13.

``bits[i, t]`` is the word ``x0 | x1 << 16`` of Threefry-4x16-13 with counter
words ``(t & 0xFFFF, t >> 16, 0, 0)`` and key words ``(key[i] & 0xFFFF,
key[i] >> 16, 0, 0)``: one 32-bit key per row.

The cipher's 64-bit state does not fit one int32 output, so it runs as 12
calls of ``spyre::threefry4x16_stage`` on its halves A = x0 | x1 << 16 and
B = x2 | x3 << 16: round stages of 4 rounds map (A, B) to one half, and key
injections add the key to one half.
"""

import torch


def _stage(a, b, name):
    return torch.ops.spyre.threefry4x16_stage(a, b, name)


def threefry4x16_bits(counter: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """32 random bits per element; ``counter`` [1, n] int32, ``key`` [rows, 1] int32."""
    a = _stage(counter, key, "inj0a")
    b = torch.zeros_like(counter)  # x2 = x3 = 0 + 0
    a, b = _stage(a, b, "r0123a"), _stage(a, b, "r0123b")
    a, b = _stage(a, key, "inj1a"), _stage(b, key, "inj1b")
    a, b = _stage(a, b, "r4567a"), _stage(a, b, "r4567b")
    b = _stage(b, key, "inj2b")  # injection 2 adds zero words to A
    a, b = _stage(a, b, "r0123a"), _stage(a, b, "r0123b")  # rounds 8-11
    a = _stage(a, key, "inj3a")
    return _stage(a, b, "r12a")  # round 12 is even: A depends on A only
