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

"""oot_checker.checks.check_invalid_platforms: typo'd exclude__plat_* tag suffixes."""

from pathlib import Path

import pytest

from oot_checker.checks import (
    KNOWN_PLATFORMS,
    check_ci_groups,
    check_invalid_platforms,
)


@pytest.mark.parametrize(
    "platforms,expected",
    [
        (["ppc64le"], 0),  # known value
        ([], 0),  # nothing declared
        (["PPC64LE"], 1),  # wrong case: runtime match is exact, so it never fires
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
    assert KNOWN_PLATFORMS == {"x86_64", "ppc64le", "s390x"}


def test_check_ci_groups_accepts_a_pair():
    d = Path("d")
    assert check_ci_groups({d / "a.yaml": ("g", {}), d / "b.yaml": ("g", {})}) == 0


def test_check_ci_groups_accepts_identical_global_blocks():
    d = Path("d")
    g = {"supported_dtypes": [{"name": "float16"}]}
    assert check_ci_groups({d / "a.yaml": ("g", g), d / "b.yaml": ("g", dict(g))}) == 0


def test_check_ci_groups_rejects_differing_global_blocks():
    d = Path("d")
    a = {"supported_dtypes": [{"name": "float16"}]}
    b = {"supported_dtypes": [{"name": "float32"}]}
    assert check_ci_groups({d / "a.yaml": ("g", a), d / "b.yaml": ("g", b)}) == 1
    # A block versus none is also a difference.
    assert check_ci_groups({d / "a.yaml": ("g", a), d / "b.yaml": ("g", {})}) == 1


def test_check_ci_groups_rejects_singleton_but_not_same_name_elsewhere():
    assert check_ci_groups({Path("d1/a.yaml"): ("g", {})}) == 1
    # Same name in two directories is two singletons, not one group.
    both = {Path("d1/a.yaml"): ("g", {}), Path("d2/b.yaml"): ("g", {})}
    assert check_ci_groups(both) == 2
