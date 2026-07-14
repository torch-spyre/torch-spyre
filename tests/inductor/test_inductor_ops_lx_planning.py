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

import copy
import functools
import torch_spyre
import os
import sys
import torch
from torch.utils import _pytree as pytree
from torch.testing import FileCheck

from torch._dynamo.testing import make_test_cls_with_patches

import unittest
from utils_inductor import compare_with_cpu

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops  # noqa: E402

tests_lx_planning_run_skips: bool = (
    os.environ.get("TEST_LX_PLANNING_RUN_SKIPS", "1") == "1"
)

# By default, only run one representative test per (prefix, op) cell of
# TestOps.PARAMS plus all non-parameterized methods. Set this to "1" to
# wrap every generated test — useful for thorough triage, skip-list
# maintenance, and CI but slow for everyday dev workflow.
tests_lx_planning_full: bool = os.environ.get("TEST_LX_PLANNING_FULL", "1") == "1"


def make_lx_planning_class(cls):
    return make_test_cls_with_patches(
        cls,
        "LxPlanning",
        "",
        (torch_spyre._inductor.config, "lx_planning", True),
        (torch_spyre._inductor.config, "allow_all_ops_in_lx_planning", True),
        (torch_spyre._inductor.config, "sencores", 32),
    )


def _copy_inherited_methods(src, dst, attrs):
    for attr in attrs:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _canonical_test_names(test_cls):
    """Pick one representative test name per (prefix, op) cell of
    ``test_cls.PARAMS``, plus all non-parameterized test methods.

    For a PARAMS entry with an ``ops_dict``, every op gets one shape (the
    first ``param_sets`` case); without an ``ops_dict``, only the first
    case is emitted.
    """
    params = getattr(test_cls, "PARAMS", {})
    canonical = set()
    parameterized_prefixes = []
    for (prefix, _base_func_name), cases in params.items():
        param_sets = cases.get("param_sets", {})
        if not param_sets:
            continue
        parameterized_prefixes.append(prefix)
        first_case = next(iter(param_sets))
        ops_dict = cases.get("ops_dict")
        if ops_dict:
            for op_name in ops_dict:
                canonical.add(f"{prefix}_{op_name}_{first_case}")
        else:
            canonical.add(f"{prefix}_{first_case}")
    for name in vars(test_cls):
        if name.startswith("test_") and not any(
            name.startswith(p + "_") for p in parameterized_prefixes
        ):
            canonical.add(name)
    return canonical


# Delegator test methods in ``TestOps`` whose body simply calls a sibling
# ``self.test_*()`` by its bare name (e.g. ``test_eq_scalar_constant_zero``
# -> ``self.test_eq_scalar_zero()``). The LX-planning wrap renames every test
# with a ``_{suffix}`` suffix, so those bare-name calls raise ``AttributeError``
# on the wrapped class. They are redundant aliases of canonical tests that are
# already wrapped, so skip wrapping them entirely. (The skip-lists below cannot
# cover this because ``TEST_LX_PLANNING_RUN_SKIPS=1`` bypasses them.)
_DELEGATOR_TESTS = frozenset(
    {
        "test_scalar_comparison",
        "test_eq_scalar_constant_int",
        "test_eq_scalar_constant_float",
        "test_eq_scalar_constant_negative",
        "test_eq_scalar_constant_zero",
        "test_eq_scalar_constant_multidim",
        "test_eq_scalar_constant_large_tensor",
        "test_eq_scalar_vs_tensor_comparison",
    }
)


def _copy_canonical_tests(
    src_cls, dst_cls, suffix, test_failures, inherited_test_attributes
):
    """Copy test methods from ``src_cls`` into ``dst_cls`` with ``_{suffix}``
    appended to each name. Unless ``TEST_LX_PLANNING_FULL`` is set, restrict
    to the canonical subset derived from ``TestOps.PARAMS``."""
    keep = (
        None
        if tests_lx_planning_full
        else _canonical_test_names(inductor.test_inductor_ops.TestOps)
    )
    for name, value in src_cls.__dict__.items():
        if not name.startswith("test_"):
            continue
        if name in _DELEGATOR_TESTS:
            continue
        if keep is not None and name not in keep:
            continue

        @functools.wraps(value)
        def new_test(self, value=value):
            return value(self)

        new_test.__dict__ = copy.deepcopy(value.__dict__)
        if test_failures and name in test_failures:
            new_test = unittest.skip("Skipped!")(new_test)
        setattr(dst_cls, f"{name}_{suffix}", new_test)
    _copy_inherited_methods(src_cls, dst_cls, inherited_test_attributes)


INHERITED_TEST_ATTRIBUTES = [
    "is_dtype_supported",
    "_get_core_reduction_invalid_dim_cases",
    "_get_single_dim_reduction_invalid_dim_cases",
]

POINTWISE_TEST_FAILURES = []


class _LxPlanningTwoOpTestBase(unittest.TestCase):
    """Shared scaffolding for LX-planning two-op tests.

    Subclasses implement ``wrap(fn)`` to append a second op (pointwise,
    reduction, ...) onto the result of each upstream op test.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def wrap(self, fn):
        raise NotImplementedError

    def compare_with_cpu(self, fn, *args, **kwargs):
        def source_check(source):
            FileCheck().check("{lx: 0}").run(source)

        kwargs["cpu_compile"] = False
        return compare_with_cpu(
            self.wrap(fn), source_check=source_check, *args, **kwargs
        )

    def compare(
        self,
        fn,
        *args,
        atol=0.0,
        rtol=0.0,
        cpu_atol=0.1,
        cpu_rtol=0.1,
        needs_device=False,
    ):
        return compare_with_cpu(
            self.wrap(fn),
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


class LxPlanningTwoOpPointwiseAdditionTest(_LxPlanningTwoOpTestBase):
    def wrap(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: (
                    (x + x) / 2
                    if isinstance(x, torch.Tensor) and x.dtype == torch.float16
                    else x
                ),
                result,
            )

        return make_seq_of_ops


_copy_canonical_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpPointwiseAdditionTest,
    "lx_planning_pointwise",
    POINTWISE_TEST_FAILURES if not tests_lx_planning_run_skips else None,
    INHERITED_TEST_ATTRIBUTES,
)


REDUCTION_TEST_FAILURES = []


class LxPlanningTwoOpReductionTest(_LxPlanningTwoOpTestBase):
    def wrap(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: (
                    torch.sum(x, dim=0)
                    if isinstance(x, torch.Tensor) and x.dtype == torch.float16
                    else x
                ),
                result,
            )

        return make_seq_of_ops


_copy_canonical_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpReductionTest,
    "lx_planning_reduction",
    REDUCTION_TEST_FAILURES if not tests_lx_planning_run_skips else None,
    INHERITED_TEST_ATTRIBUTES,
)
