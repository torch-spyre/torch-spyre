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


"""Checks how test_inductor_ops_lx_planning.py derives its suite from TestOps.

The lx classes wrap tests from TestOps to check LX planning. A TestOps test that is
marked xfail fails for reasons unrelated to LX planning, and the wrap can hide or
change that failure, so the lx classes must not contain copies of such tests.
"""

import os
import sys
import unittest
from unittest import mock

import regex as re

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops as ops  # noqa: E402
import inductor.test_inductor_ops_lx_planning as lx  # noqa: E402
from inductor.utils_inductor import expects_raise  # noqa: E402

_LX_CLASSES = (lx.LxPlanningTwoOpPointwiseAdditionTest, lx.LxPlanningTwoOpReductionTest)


def _tests(cls):
    return {n: v for n, v in vars(cls).items() if n.startswith("test_")}


def _has_xfail(fn):
    return any(m.name == "xfail" for m in getattr(fn, "pytestmark", []))


def _expects_raise(fn):
    return getattr(fn, "_expects_raise", False)


class TestLxPlanningSuiteGeneration(unittest.TestCase):
    def test_lx_classes_have_no_xfail_marked_tests(self):
        for cls in _LX_CLASSES:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(_tests(cls))
                marked = [n for n, v in _tests(cls).items() if _has_xfail(v)]
                self.assertEqual(marked, [])

    def test_xfail_marked_testops_tests_are_not_copied(self):
        xfail_names = {n for n, v in _tests(ops.TestOps).items() if _has_xfail(v)}
        self.assertTrue(xfail_names, "TestOps has no xfail-marked tests to check")
        for cls in _LX_CLASSES:
            with self.subTest(cls=cls.__name__):
                copied = {re.sub(r"_lx_planning_\w+$", "", n) for n in _tests(cls)}
                self.assertEqual(sorted(xfail_names & copied), [])

    def test_expect_raise_testops_tests_are_not_copied(self):
        raise_names = {n for n, v in _tests(ops.TestOps).items() if _expects_raise(v)}
        self.assertTrue(raise_names, "TestOps has no expect_raise tests to check")
        for cls in _LX_CLASSES:
            with self.subTest(cls=cls.__name__):
                copied = {re.sub(r"_lx_planning_\w+$", "", n) for n in _tests(cls)}
                self.assertEqual(sorted(raise_names & copied), [])

    def test_passing_testops_tests_are_still_copied(self):
        ok_names = {
            n
            for n, v in _tests(ops.TestOps).items()
            if not _has_xfail(v) and not _expects_raise(v)
        }
        if not lx.tests_lx_planning_full:
            # Only the canonical subset is copied.
            ok_names &= lx._canonical_test_names(ops.TestOps)
        for cls in _LX_CLASSES:
            with self.subTest(cls=cls.__name__):
                copied = {re.sub(r"_lx_planning_\w+$", "", n) for n in _tests(cls)}
                # _DELEGATOR_TESTS are deliberately not wrapped.
                missing = ok_names - copied - lx._DELEGATOR_TESTS
                self.assertEqual(sorted(missing), [])

    def test_expect_raise_tests_are_not_copied(self):
        # A test that asserts a rejection of the base op says nothing about LX planning.
        class Src:
            def test_ok(self):
                pass

            def test_rejected(self):
                pass

            test_rejected._expects_raise = True

        class Dst:
            pass

        with mock.patch.object(lx, "tests_lx_planning_full", True):
            lx._copy_canonical_tests(Src, Dst, "sfx", None, [])
        self.assertEqual(
            sorted(n for n in vars(Dst) if n.startswith("test_")), ["test_ok_sfx"]
        )

    def test_hand_written_expects_raise_tests_are_not_copied(self):
        # A hand-written test that wraps its body in pytest.raises is tagged with
        # the decorator and skipped the same way as a generated expect_raise case.
        class Src:
            def test_ok(self):
                pass

            @expects_raise
            def test_rejected(self):
                pass

        class Dst:
            pass

        with mock.patch.object(lx, "tests_lx_planning_full", True):
            lx._copy_canonical_tests(Src, Dst, "sfx", None, [])
        self.assertEqual(
            sorted(n for n in vars(Dst) if n.startswith("test_")), ["test_ok_sfx"]
        )

    # Hand-written tests that wrap their body in pytest.raises (or assert an error
    # some other way) without going through the LX wrap. Not found by name, so list
    # them: a new one has to be added here and tagged with expects_raise.
    _HAND_WRITTEN_REJECTION_TESTS = (
        "test_core_reduction_invalid_dims_api",
        "test_core_reduction_invalid_dims_spyre",
        "test_single_dim_reduction_invalid_dims_api",
        "test_single_dim_reduction_invalid_dims_spyre",
        "test_is_nonzero_error_cases",
        "test_restickify_fp32_unsupported",
        "test_restickify_int64_unsupported",
        "test_slice_scatter_step_raises",
        "test_slice_stick_mutation_no_alt_dim_raises",
    )

    def test_hand_written_rejection_tests_have_no_lx_copies(self):
        # The rejection is raised in the op under test (an unregistered operator, an
        # unsupported layout, an API check) before the second op is built, or the test
        # never calls the overridden compare_with_cpu at all, so a copy would only
        # repeat the base test.
        rejected = {
            n
            for n, v in _tests(ops.TestOps).items()
            if (n.endswith("_rejected") and not _has_xfail(v))
            or n in self._HAND_WRITTEN_REJECTION_TESTS
        }
        self.assertTrue(rejected, "TestOps has no rejection tests to check")
        self.assertLessEqual(set(self._HAND_WRITTEN_REJECTION_TESTS), set(rejected))
        untagged = sorted(
            n for n in rejected if not _expects_raise(ops.TestOps.__dict__[n])
        )
        self.assertEqual(untagged, [])
        for cls in _LX_CLASSES:
            with self.subTest(cls=cls.__name__):
                copied = {re.sub(r"_lx_planning_\w+$", "", n) for n in _tests(cls)}
                self.assertEqual(sorted(rejected & copied), [])

    def test_canonical_subset_skips_expect_raise_cases(self):
        class Fake:
            PARAMS = {
                ("test_a", "base"): {
                    "param_sets": {"rejected": (), "good": ()},
                    "expect_raise": {"rejected": "boom"},
                },
            }

        self.assertEqual(lx._canonical_test_names(Fake), {"test_a_good"})

    def test_canonical_subset_skips_xfail_cases(self):
        class Fake:
            PARAMS = {
                ("test_a", "base"): {
                    "ops_dict": {"op1": None, "op2": None},
                    "param_sets": {"bad": (), "good": ()},
                    "expect_fail": ["bad", "op2_good"],
                },
                ("test_b", "base"): {
                    "param_sets": {"bad": (), "good": ()},
                    "expect_fail": ["bad"],
                },
            }

        # op1: "bad" is xfail for every op, so "good" is the first usable case.
        # op2: "bad" and "good" are both xfail for op2, so it has no usable case.
        self.assertEqual(
            lx._canonical_test_names(Fake), {"test_a_op1_good", "test_b_good"}
        )


if __name__ == "__main__":
    unittest.main()
