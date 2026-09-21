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

"""CPU unit tests for ``is_ea_compatible`` (issue #3223).

Locks the ElementArrangement set-membership contract used by
``validate_ops``. Lives under ``tests/`` rather than ``tests/inductor/``
so the inductor session HBM-poison fixture is not collected. Needs a
built ``torch_spyre._C`` for the enum, not a Spyre card.
"""

import pytest

from torch_spyre._C import ElementArrangement as EA
from torch_spyre._inductor.constants import is_ea_compatible


@pytest.mark.parametrize(
    ("eas", "expected"),
    [
        ([], True),
        ([EA.STANDARD], True),
        ([EA.STANDARD, EA.STANDARD], True),
        ([EA.EXX2, EA.EXX2], True),
        ([EA.QFP8CH, EA.QFP8CH], True),
        ([EA.DL16_TO_FP32, EA.DL16_TO_FP32], True),
        ([EA.FP32_TO_DL16, EA.FP32_TO_DL16], True),
        ([EA.STANDARD, EA.DL16_TO_FP32], True),
        ([EA.STANDARD, EA.FP32_TO_DL16], True),
        ([EA.STANDARD, EA.QFP8CH], True),
        ([EA.DL16_TO_FP32, EA.FP32_TO_DL16], False),
        ([EA.EXX2, EA.STANDARD], False),
        ([EA.STANDARD, EA.STANDARD, EA.DL16_TO_FP32], True),
        ([EA.DL16_TO_FP32, EA.DL16_TO_FP32, EA.STANDARD], True),
    ],
    ids=[
        "empty",
        "one_standard",
        "all_standard",
        "all_exx2",
        "all_qfp8ch",
        "all_dl16_to_fp32",
        "all_fp32_to_dl16",
        "standard_plus_dl16_to_fp32",
        "standard_plus_fp32_to_dl16",
        "standard_plus_qfp8ch",
        "two_different_non_standard",
        "exx2_plus_standard",
        "duplicate_standard_broadcast",
        "duplicate_non_standard_broadcast",
    ],
)
def test_is_ea_compatible(eas, expected):
    assert is_ea_compatible(eas) is expected
