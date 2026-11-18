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


"""Tests for the ``expect_fail_unstable`` key of ParameterizedTestMeta.

``expect_fail_unstable`` marks a case as a *non-strict* xfail with a required reason, for
a case that fails but is known to pass on some runs. A strict xfail that passes fails the
run, so it would turn an unstable case into a flaky failure.
"""

import os
import sys
import unittest

import pytest

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

from inductor.utils_inductor import ParameterizedTestMeta  # noqa: E402


def _make(cases):
    def base(self, mode):
        pass

    namespace = {"PARAMS": {("test_f", "test_f_base"): cases}, "test_f_base": base}
    return ParameterizedTestMeta("Fake", (unittest.TestCase,), namespace)


def _xfail_mark(fn):
    marks = [m for m in getattr(fn, "pytestmark", []) if m.name == "xfail"]
    return marks[0] if marks else None


class TestExpectFailUnstable(unittest.TestCase):
    def test_case_is_a_non_strict_xfail_with_the_reason(self):
        cls = _make(
            {
                "param_sets": {"c": ("x",), "d": ("x",)},
                "expect_fail_unstable": {"c": "flaky, see #1"},
            }
        )
        mark = _xfail_mark(cls.test_f_c)
        self.assertIsNotNone(mark)
        self.assertFalse(mark.kwargs["strict"])
        self.assertIn("flaky, see #1", mark.kwargs["reason"])
        self.assertIsNone(_xfail_mark(cls.test_f_d))

    def test_expect_fail_stays_strict(self):
        cls = _make({"param_sets": {"c": ("x",)}, "expect_fail": ["c"]})
        self.assertTrue(_xfail_mark(cls.test_f_c).kwargs["strict"])

    def test_ops_dict_entry_can_target_one_op(self):
        def base(self, op, mode):
            pass

        namespace = {
            "PARAMS": {
                ("test_f", "test_f_base"): {
                    "ops_dict": {"a": "a", "b": "b"},
                    "param_sets": {"c": ("x",)},
                    "expect_fail_unstable": {"a_c": "only op a"},
                }
            },
            "test_f_base": base,
        }
        cls = ParameterizedTestMeta("Fake", (unittest.TestCase,), namespace)
        self.assertFalse(_xfail_mark(cls.test_f_a_c).kwargs["strict"])
        self.assertIsNone(_xfail_mark(cls.test_f_b_c))

    def test_reason_must_not_be_empty(self):
        with pytest.raises(AssertionError, match="reason"):
            _make(
                {
                    "param_sets": {"c": ("x",)},
                    "expect_fail_unstable": {"c": ""},
                }
            )

    def test_entry_also_in_expect_fail_is_rejected(self):
        with pytest.raises(AssertionError, match="expect_fail"):
            _make(
                {
                    "param_sets": {"c": ("x",)},
                    "expect_fail_unstable": {"c": "flaky"},
                    "expect_fail": ["c"],
                }
            )

    def test_entry_that_matches_no_case_is_rejected(self):
        with pytest.raises(AssertionError, match="matches no"):
            _make(
                {
                    "param_sets": {"c": ("x",)},
                    "expect_fail_unstable": {"typo": "flaky"},
                }
            )


if __name__ == "__main__":
    unittest.main()
