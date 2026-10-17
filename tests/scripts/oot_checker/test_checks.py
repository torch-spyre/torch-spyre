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

"""oot_checker.checks.check_invalid_platforms: typo'd exclude_platforms values."""

from pathlib import Path

import pytest

from oot_checker.checks import KNOWN_PLATFORMS, check_invalid_platforms


@pytest.mark.parametrize(
    "platforms,expected",
    [
        (["ppc64le"], 0),  # known value
        ([], 0),  # nothing declared
        (["PPC64LE"], 0),  # known value, mixed case
        (["ppc64"], 1),  # typo
        (["power"], 1),  # typo
    ],
)
def test_check_invalid_platforms(platforms, expected):
    assert check_invalid_platforms({Path("a.yaml"): platforms}) == expected


def test_known_platforms_contains_expected_values():
    # Pinned exactly: a silent edit here (e.g. dropping "s390x") would make
    # check_invalid_platforms stop flagging a real typo for that arch, and
    # the parametrized cases above wouldn't catch it since none exercise it.
    assert KNOWN_PLATFORMS == {"x86_64", "ppc64le", "s390x", "aarch64"}
