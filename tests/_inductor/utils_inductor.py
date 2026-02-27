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

import functools
import torch

DEVICE = torch.device("spyre")


# shape is a tuple of integers representing dimension of the tensor
# to avoid using the same cached tensor of the same shape, add a unique
# differentiation argument
@functools.lru_cache(maxsize=None)
def cached_randn(
    shape, differentiation=None, abs=False, dtype=torch.float16, scale=1.0
):
    out = torch.randn(shape, dtype=dtype) * scale
    return out if not abs else torch.abs(out)


# init_helper initiates tensors given a list of shape tuples
def init_helper(shapes, dtype=torch.float16, cached=True):
    randn_func = cached_randn if cached else torch.randn
    return tuple(
        randn_func(shape, differentiation=i, dtype=dtype)
        for i, shape in enumerate(shapes)
    )


# shapes2key uses the int values of shape tuples to construct
# a string as unique id for the parameterized test cases
# e.g. ((4, 8), (4, 8)) -> 4x8_4x8
# shapes: tuple of shapes
def shapes2key(shapes):
    return "_".join(["x".join(str(dim) for dim in s) for s in shapes])


# cases: Tuple of cases. Each case is defined by shapes of tensors
def make_param_dict(cases):
    return {shapes2key(shapes): init_helper(shapes) for shapes in cases}


# ParameterizedTestMeta injects parameterized test methods
# based on PARAMS of the subclass.
# The metaclass looks through the keys in the PARAMS dict,
# and use "base_func_name" to look up the base_func as the
# template for creating parameterized test methods.
#
# PARAMS is a dictionary of test parameters, where
# each key-value pair contains
# (test_name_prefix, base_func_name):
#    {
#        "ops_dict": ops_dict, # optional
#        "param_sets": param_dict,
#    }
# the number of test methods is determined by the cross-
# product of the ops_dict and param_sets.
# if ops_dict is not provided, it means the base_function
# has concrete implementation.
#
# ops_dict (optional) contains the mapping from op_name to
# op_func pointer. The op_names are used to create new test
# method names, and the op_func pointer is used to make
# specialized new test functions.
#
# param_dict contains a mapping from a user defined test case
# name to a tuple of arguments that will be passed to the
# materialized test functions when invoked.
#
# The materialized test function name is determined by:
# 1. Explicitly specified --> {test_name_prefix}_[op_name_]{test_case}
# 2. If using make_param_dict helper function,
#    the test case name is {test_name_prefix}_[op_name_]{shapes2key(shapes)}
#
# E.g. The following example has a test_name_prefix == "test_name"
#      and 3 test cases (per op if ops_dict is supplied):
#      "test_name_[op_name_]case_0", "test_name_[op_name_]case_1",
#      and "test_name_[op_name_]case_2"
# ("test_name", "base_func_name"): {
#       "case_0": (arg0, arg1, ...),
#       "case_1": (arg0, arg1, ...),
#       "case_2": (arg0, arg1, ...),
# }
#
# NOTE:
# - The base_func will be removed from the namespace if there is
#   at least one parameterized method associated with it.
# - If parameterization is not needed for a concrete test case,
#   simply implement it in TestOps without adding an item
#   to PARAMS. It will be executed by unittests.
class ParameterizedTestMeta(type):
    def __new__(mcs, name, bases, namespace):
        param_map = namespace.get("PARAMS", {})
        to_delete = set()

        for (test_name_prefix, base_func_name), cases in param_map.items():
            base_func = namespace.get(base_func_name)
            if base_func is None:
                continue

            ops_dict = cases["ops_dict"] if "ops_dict" in cases else None
            param_sets = cases["param_sets"]

            for test_case, params in param_sets.items():
                if ops_dict:
                    # ---- Cross product: ops × cases ----
                    for op_name, op in ops_dict.items():

                        def make_test(_base_func, _op, _params):
                            @functools.wraps(_base_func)
                            def test(self):
                                _base_func(self, _op, *_params)

                            # Propagate unittest.skip from base
                            if getattr(_base_func, "__unittest_skip__", False):
                                setattr(test, "__unittest_skip__", True)
                                setattr(
                                    test,
                                    "__unittest_skip_why__",
                                    getattr(_base_func, "__unittest_skip_why__", ""),
                                )
                            return test

                        test_name = f"{test_name_prefix}_{op_name}_{test_case}"
                        assert test_name not in namespace, (
                            f"Test name conflict: {test_name}"
                        )
                        namespace[test_name] = make_test(base_func, op, params)
                else:
                    # ---- Original per-case expansion ----
                    def make_test(_base_func, _params):
                        @functools.wraps(_base_func)
                        def test(self):
                            _base_func(self, *_params)

                        if getattr(_base_func, "__unittest_skip__", False):
                            setattr(test, "__unittest_skip__", True)
                            setattr(
                                test,
                                "__unittest_skip_why__",
                                getattr(_base_func, "__unittest_skip_why__", ""),
                            )
                        return test

                    test_name = f"{test_name_prefix}_{test_case}"
                    assert test_name not in namespace, (
                        f"Test name conflict: {test_name}"
                    )
                    namespace[test_name] = make_test(base_func, params)

            # Remove base function if parameterized
            to_delete.add(base_func_name)

        for key in to_delete:
            namespace.pop(key, None)

        return super().__new__(mcs, name, bases, namespace)


# Helper functions for compare operations
def _compile_and_run(fn, args, device, backend=None, needs_device=False):
    """Compile and execute function on specified device/backend, returning result on CPU."""
    torch._dynamo.reset_code_caches()
    device = torch.device(device) if isinstance(device, str) else device
    device_args = [arg.to(device) for arg in args]
    device_kwargs = {"device": device} if needs_device else {}

    if backend:
        result = torch.compile(fn, backend=backend)(*device_args, **device_kwargs)
    else:
        result = torch.compile(fn)(*device_args, **device_kwargs)

    if not isinstance(result, int):
        assert result.device.type == device.type, (
            f"Output not on expected device. Expected {device}, got {result.device}"
        )
        result = result.cpu()

    return result


def _assert_results_close(actual, expected, atol, rtol, comparison_name):
    """Assert two results are close with formatted error message."""
    torch.testing.assert_close(
        actual,
        expected,
        equal_nan=True,
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"{comparison_name} mismatch\n\n{msg}\n",
    )


# Compare functions
# The compare functions would compile the function with torch-spyre and
# assign the result of excuting the compiled function to `target` for comparison
# if `target` is not set by the caller. Otherwise, the target tensor is used
# in the comparisons.


def compare_with_cpu(
    fn,
    *args,
    atol=0.1,
    rtol=0.1,
    needs_device=False,
    cpu_compile=True,
    target=None,
):
    """Compare compiled Spyre execution against uncompiled (and optionally compiled) CPU execution."""
    cpu_result = fn(*args)

    # compiled spyre execution
    if target is None:
        target = _compile_and_run(fn, args, DEVICE, needs_device=needs_device)

    _assert_results_close(target, cpu_result, atol, rtol, "compiled spyre <-> cpu")

    # compiled cpu execution
    if cpu_compile:
        cpu_compiled_result = _compile_and_run(
            fn, args, "cpu", needs_device=needs_device
        )
        _assert_results_close(
            target,
            cpu_compiled_result,
            atol,
            rtol,
            "compiled spyre <-> compiled cpu",
        )


def compare_with_pytorch(fn, fn_pytorch, *args, atol=0.1, rtol=0.1, target=None):
    """Compare compiled Spyre function against uncompiled PyTorch reference function."""
    if target is None:
        target = _compile_and_run(fn, args, DEVICE)
    pytorch_result = fn_pytorch(*args)
    _assert_results_close(target, pytorch_result, atol, rtol, "pytorch")


def compare_with_sendnn(fn, *args, atol=0.0, rtol=0.0, needs_device=False, target=None):
    """Compare compiled Spyre execution against sendnn backend execution."""
    if target is None:
        target = _compile_and_run(fn, args, DEVICE, needs_device=needs_device)
    sendnn_result = _compile_and_run(fn, args, "cpu", backend="sendnn")
    _assert_results_close(target, sendnn_result, atol, rtol, "sendnn")


def compare(
    fn, *args, atol=0.0, rtol=0.0, cpu_atol=0.1, cpu_rtol=0.1, needs_device=False
):
    """3-way comparison: compiled Spyre vs uncompiled CPU vs sendnn backend."""
    spyre_compiled_result = _compile_and_run(
        fn, args, DEVICE, needs_device=needs_device
    )
    compare_with_cpu(
        fn,
        *args,
        atol=cpu_atol,
        rtol=cpu_rtol,
        needs_device=needs_device,
        target=spyre_compiled_result,
    )
    compare_with_sendnn(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        needs_device=needs_device,
        target=spyre_compiled_result,
    )
