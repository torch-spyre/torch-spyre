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

"""Tests for the ``{case: reason}`` form of the ``expect_fail`` key of ParameterizedTestMeta.

``expect_fail`` is a list of cases, each a strict xfail whose reason names only the case.
As a mapping it also says why (for example, which issue), so the reason can be read from
the xfail report instead of from a comment next to the list.
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


class TestExpectFailReason(unittest.TestCase):
    def test_mapping_gives_each_case_its_own_reason(self):
        cls = _make(
            {
                "param_sets": {"c": ("x",), "d": ("x",), "e": ("x",)},
                "expect_fail": {"c": "wrong values, #1", "d": "crash, #2"},
            }
        )
        c, d = _xfail_mark(cls.test_f_c), _xfail_mark(cls.test_f_d)
        self.assertTrue(c.kwargs["strict"] and d.kwargs["strict"])
        self.assertIn("wrong values, #1", c.kwargs["reason"])
        self.assertIn("crash, #2", d.kwargs["reason"])
        self.assertIsNone(_xfail_mark(cls.test_f_e))

    def test_list_keeps_the_default_reason(self):
        cls = _make({"param_sets": {"c": ("x",)}, "expect_fail": ["c"]})
        mark = _xfail_mark(cls.test_f_c)
        self.assertTrue(mark.kwargs["strict"])
        self.assertEqual(mark.kwargs["reason"], "Expected fail for c")

    def test_ops_dict_entry_can_target_one_op_with_a_reason(self):
        def base(self, op, mode):
            pass

        namespace = {
            "PARAMS": {
                ("test_f", "test_f_base"): {
                    "ops_dict": {"a": "a", "b": "b"},
                    "param_sets": {"c": ("x",)},
                    "expect_fail": {"a_c": "only op a, #3"},
                }
            },
            "test_f_base": base,
        }
        cls = ParameterizedTestMeta("Fake", (unittest.TestCase,), namespace)
        self.assertIn("only op a, #3", _xfail_mark(cls.test_f_a_c).kwargs["reason"])
        self.assertIsNone(_xfail_mark(cls.test_f_b_c))

    def test_reason_must_not_be_empty(self):
        for reason in ("", "  "):
            with (
                self.subTest(reason=reason),
                pytest.raises(AssertionError, match="reason"),
            ):
                _make({"param_sets": {"c": ("x",)}, "expect_fail": {"c": reason}})

    def test_mapping_entry_also_in_expect_raise_is_rejected(self):
        with pytest.raises(AssertionError, match="expect_raise"):
            _make(
                {
                    "param_sets": {"c": ("x",)},
                    "expect_fail": {"c": "why"},
                    "expect_raise": {"c": "fragment"},
                }
            )


if __name__ == "__main__":
    unittest.main()
