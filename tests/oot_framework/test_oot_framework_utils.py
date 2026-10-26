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

"""Unit tests for OOT framework utilities changed in PR #4588.

Covers:
  - _filter_entries_by_platform  (platform-tag override logic, #3902)
  - _extract_op_name_from_method (OpDTypes.none / no-dtype-suffix path, #3444)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import patch

from oot_framework.oot_test_utilities import (
    _extract_op_name_from_method,
    _filter_entries_by_platform,
)


def _entry(tags: List[str]) -> SimpleNamespace:
    """Minimal YAML-entry stub with a tags attribute."""
    return SimpleNamespace(tags=tags)


# ---------------------------------------------------------------------------
# _filter_entries_by_platform
# ---------------------------------------------------------------------------


class TestFilterEntriesByPlatform:
    def test_matching_arch_tagged_entry_wins(self):
        """On s390x a platform__s390x entry takes precedence over the untagged one."""
        untagged = _entry([])
        tagged = _entry(["platform__s390x", "platform__ppc64le"])
        with patch("oot_framework.oot_test_utilities._OOT_PLATFORM_ARCH", "s390x"):
            result = _filter_entries_by_platform([untagged, tagged])
        assert result == [tagged]

    def test_non_matching_arch_falls_back_to_untagged(self):
        """On x86_64 the s390x-tagged entry is dropped and the untagged one is returned."""
        untagged = _entry([])
        tagged = _entry(["platform__s390x", "platform__ppc64le"])
        with patch("oot_framework.oot_test_utilities._OOT_PLATFORM_ARCH", "x86_64"):
            result = _filter_entries_by_platform([untagged, tagged])
        assert result == [untagged]

    def test_no_platform_tags_returns_all_entries(self):
        """Entries with no platform__ tag are returned unchanged on any arch."""
        entry_a = _entry([])
        entry_b = _entry(["testtype__regression"])
        with patch("oot_framework.oot_test_utilities._OOT_PLATFORM_ARCH", "x86_64"):
            result = _filter_entries_by_platform([entry_a, entry_b])
        assert result == [entry_a, entry_b]

    def test_only_tagged_non_matching_returns_empty(self):
        """When only a non-matching platform-tagged entry exists the result is empty."""
        tagged = _entry(["platform__s390x"])
        with patch("oot_framework.oot_test_utilities._OOT_PLATFORM_ARCH", "x86_64"):
            result = _filter_entries_by_platform([tagged])
        assert result == []


# ---------------------------------------------------------------------------
# _extract_op_name_from_method
# ---------------------------------------------------------------------------


class TestExtractOpNameFromMethod:
    def test_opdtypes_none_no_dtype_suffix(self):
        """OpDTypes.none names end with _<device> only — new path added for #3444.

        Exact example from the reviewer:
          ("test_dtypes_bitwise_and_spyre", "test_dtypes", "spyre") -> "bitwise_and"
        """
        assert (
            _extract_op_name_from_method(
                "test_dtypes_bitwise_and_spyre", "test_dtypes", "spyre"
            )
            == "bitwise_and"
        )

    def test_normal_path_with_dtype_suffix(self):
        """Normal names with _<device>_<dtype> return the op name (regression guard)."""
        assert (
            _extract_op_name_from_method(
                "test_dtypes_add_spyre_float16", "test_dtypes", "spyre"
            )
            == "add"
        )

    def test_wrong_base_name_returns_none(self):
        """A method name not starting with base_test_name returns None."""
        assert (
            _extract_op_name_from_method(
                "test_other_add_spyre_float16", "test_dtypes", "spyre"
            )
            is None
        )

    def test_remainder_equals_device_type_returns_none(self):
        """remainder == oot_device_type guard prevents an empty op name being returned."""
        # "test_dtypes_spyre" -> remainder = "spyre" == oot_device_type -> None
        assert (
            _extract_op_name_from_method("test_dtypes_spyre", "test_dtypes", "spyre")
            is None
        )
