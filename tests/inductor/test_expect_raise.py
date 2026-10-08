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


"""Tests for the ``expect_raise`` key of ParameterizedTestMeta.

``expect_raise`` turns a parameterized case into a negative test: the body must raise,
and the message must match a required fragment. A case that no longer raises, or raises
for a different reason, fails.
"""

import os
import sys
import unittest

import pytest

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

from inductor.utils_inductor import ParameterizedTestMeta  # noqa: E402


def _make(cases, base):
    """Build a TestCase whose tests are generated from one PARAMS entry."""
    namespace = {"PARAMS": {("test_f", "test_f_base"): cases}, "test_f_base": base}
    return ParameterizedTestMeta("Fake", (unittest.TestCase,), namespace)


def _run(cls, name):
    getattr(cls(name), name)()


def _base(self, mode):
    if mode == "reject":
        raise RuntimeError("Unsupported: mixed EA is not supported here")
    if mode == "other":
        raise RuntimeError("something unrelated broke")


class TestExpectRaise(unittest.TestCase):
    def test_case_that_raises_with_the_fragment_passes(self):
        cls = _make(
            {"param_sets": {"c": ("reject",)}, "expect_raise": {"c": "mixed EA"}}, _base
        )
        _run(cls, "test_f_c")

    def test_case_that_does_not_raise_fails(self):
        cls = _make(
            {"param_sets": {"c": ("ok",)}, "expect_raise": {"c": "mixed EA"}}, _base
        )
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            _run(cls, "test_f_c")

    def test_case_that_raises_something_else_fails(self):
        cls = _make(
            {"param_sets": {"c": ("other",)}, "expect_raise": {"c": "mixed EA"}}, _base
        )
        with pytest.raises(AssertionError, match="did not match"):
            _run(cls, "test_f_c")

    def test_ops_dict_entry_can_target_one_op(self):
        def base(self, op, mode):
            if op == "a" and mode == "reject":
                raise RuntimeError("Unsupported: only op a is rejected")

        cls = _make(
            {
                "ops_dict": {"a": "a", "b": "b"},
                "param_sets": {"c": ("reject",)},
                "expect_raise": {"a_c": "only op a"},
            },
            base,
        )
        _run(cls, "test_f_a_c")  # raises, as expected
        _run(cls, "test_f_b_c")  # an ordinary test; does not raise

    def test_generated_raise_tests_are_tagged(self):
        cls = _make(
            {
                "param_sets": {"c": ("reject",), "d": ("ok",)},
                "expect_raise": {"c": "mixed EA"},
            },
            _base,
        )
        self.assertTrue(cls.test_f_c._expects_raise)
        self.assertFalse(getattr(cls.test_f_d, "_expects_raise", False))

    def test_entry_also_in_expect_fail_is_rejected(self):
        with pytest.raises(AssertionError, match="expect_fail"):
            _make(
                {
                    "param_sets": {"c": ("reject",)},
                    "expect_raise": {"c": "mixed EA"},
                    "expect_fail": ["c"],
                },
                _base,
            )

    def test_empty_or_missing_fragment_is_rejected(self):
        # "" matches any exception and None would be a bare must-raise.
        for bad in ("", None, "  "):
            with self.subTest(fragment=bad):
                with pytest.raises(AssertionError, match="non-empty"):
                    _make(
                        {"param_sets": {"c": ("reject",)}, "expect_raise": {"c": bad}},
                        _base,
                    )

    def test_bare_expect_raise_and_op_specific_expect_fail_are_rejected(self):
        # expect_raise={"c"} also covers op "a"'s case, so expect_fail=["a_c"] would
        # be silently replaced by the raise wrapper.
        with pytest.raises(AssertionError, match="expect_fail"):
            _make(
                {
                    "ops_dict": {"a": "a", "b": "b"},
                    "param_sets": {"c": ("reject",)},
                    "expect_raise": {"c": "mixed EA"},
                    "expect_fail": ["a_c"],
                },
                lambda self, op, mode: None,
            )

    def test_entry_that_matches_no_case_is_rejected(self):
        with pytest.raises(AssertionError, match="matches no"):
            _make(
                {"param_sets": {"c": ("ok",)}, "expect_raise": {"typo": "mixed EA"}},
                _base,
            )


if __name__ == "__main__":
    unittest.main()
