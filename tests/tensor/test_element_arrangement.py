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

"""Host-only tests for the staggered element-arrangement tables.

The tables are checked against the device in
``tests/inductor/test_ea_bidirectional_conversion.py``.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import ElementArrangement
from torch_spyre.element_arrangement import (
    STICK_ELEMENTS,
    destagger_host,
    stick_permutation,
)

STAGGERED = (ElementArrangement.DL16_TO_FP32, ElementArrangement.FP32_TO_DL16)


def _reference_destagger(buf: torch.Tensor, ea: ElementArrangement) -> torch.Tensor:
    """The reshape/stack form ``spyre__copy_from`` used before the shared table."""
    shape = buf.shape
    if ea == ElementArrangement.DL16_TO_FP32:
        flat = buf.reshape(-1, 64)
        h0 = flat[:, :32].reshape(-1, 8, 4)
        h1 = flat[:, 32:].reshape(-1, 8, 4)
        return torch.stack([h0, h1], dim=2).reshape(shape)
    flat = buf.reshape(-1, 8, 2, 4)
    s0 = flat[:, :, 0, :].reshape(-1, 32)
    s1 = flat[:, :, 1, :].reshape(-1, 32)
    return torch.cat([s0, s1], dim=1).reshape(shape)


class TestElementArrangement(TestCase):
    def test_tables_are_stick_permutations(self):
        for ea in STAGGERED:
            self.assertEqual(
                sorted(stick_permutation(ea)), list(range(STICK_ELEMENTS)), ea
            )

    def test_known_slots(self):
        dl = stick_permutation(ElementArrangement.DL16_TO_FP32)
        self.assertEqual(dl[:8], (0, 1, 2, 3, 8, 9, 10, 11))
        self.assertEqual(dl[32:36], (4, 5, 6, 7))
        fd = stick_permutation(ElementArrangement.FP32_TO_DL16)
        self.assertEqual(fd[:8], (0, 1, 2, 3, 32, 33, 34, 35))

    def test_standard_has_no_table(self):
        with self.assertRaises(ValueError):
            stick_permutation(ElementArrangement.STANDARD)

    def test_destagger_matches_reference(self):
        for ea in STAGGERED:
            for dtype in (torch.float32, torch.float16):
                x = torch.randn(3, 128).to(dtype)
                got = destagger_host(x.clone(), ea)
                self.assertTrue(
                    torch.equal(got, _reference_destagger(x, ea)), (ea, dtype)
                )

    def test_destagger_inverts_staggering(self):
        for ea in STAGGERED:
            perm = torch.tensor(stick_permutation(ea))
            logical = torch.arange(2 * STICK_ELEMENTS, dtype=torch.float32).reshape(
                2, -1
            )
            raw = logical[:, perm]  # slot s holds logical perm[s]
            self.assertTrue(torch.equal(destagger_host(raw.clone(), ea), logical), ea)

    def test_standard_is_noop(self):
        x = torch.randn(2, 64)
        self.assertTrue(
            torch.equal(destagger_host(x.clone(), ElementArrangement.STANDARD), x)
        )


if __name__ == "__main__":
    run_tests()
