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
import os

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


@functools.lru_cache(maxsize=None)
def cached_xavier(
    shape,
    differentiation=None,
    dtype=torch.float16,
):
    out = torch.empty(shape, dtype=dtype)
    torch.nn.init.xavier_uniform_(out)
    return out


@functools.lru_cache(maxsize=None)
def unique_randn_along_dim(
    shape,
    dim=-1,
    min_val=-100.0,
    max_val=100.0,
    dtype=torch.float16,
    seed=None,
    warn_precision=True,
):
    """
    Generate tensor with unique values along a specified dimension.

    This is useful for testing operations like argmax/argmin where you want
    to avoid that multiple elements in a tensor have the same maximum value,
    whcih is called "tie-breaking". For large tensors, generating globally
    unique values can cause float16 overflow. This function generates unique
    values only along the specified dimension, keeping values in a safe range.

    The function automatically checks for float16 precision issues and warns
    if the spacing between values is too small to guarantee uniqueness after
    float16 conversion.

    Args:
        shape: Tuple specifying tensor shape (e.g., (64, 128, 256))
        dim: Dimension along which to ensure uniqueness (default: -1, last dim)
             Can be negative to count from the end
        min_val: Minimum value in the range
        max_val: Maximum value in the range
        dtype: Target data type (torch.float16 or torch.float32)
        seed: Random seed for reproducibility
        warn_precision: If True, warn about potential float16 precision issues

    Returns:
        Tensor with unique values along the specified dimension

    Raises:
        ValueError: If parameters would cause float16 overflow or precision loss

    Examples:
        >>> # Unique values along last dimension (rows)
        >>> tensor = create_unique_along_dim((64, 128), dim=-1)
        >>> # Each row has 128 unique values

        >>> # Unique values along first dimension (columns)
        >>> tensor = create_unique_along_dim((64, 128), dim=0)
        >>> # Each column has 64 unique values

        >>> # 3D tensor with unique values along middle dimension
        >>> tensor = create_unique_along_dim((32, 64, 128), dim=1)
        >>> # For each (i, k), tensor[i, :, k] has 64 unique values

        >>> # Large tensor - automatically uses safe range
        >>> tensor = create_unique_along_dim((1000, 1000), dim=-1)
        >>> # Warns if range is too large for float16 precision
    """

    if seed is not None:
        torch.random.manual_seed(seed)

    # Normalize dimension to positive index
    ndim = len(shape)
    if dim < 0:
        dim = ndim + dim

    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim {dim} out of range for tensor with {ndim} dimensions")

    # Size along the dimension we want to make unique
    unique_size = shape[dim]

    # Calculate total number of "slices" along the unique dimension
    # For shape (A, B, C) with dim=1, we have A*C slices of size B each
    num_slices = 1
    for i, s in enumerate(shape):
        if i != dim:
            num_slices *= s

    # Check for float16 overflow
    if dtype == torch.float16:
        float16_max = torch.finfo(torch.float16).max  # 65504.0
        if abs(min_val) > float16_max or abs(max_val) > float16_max:
            raise ValueError(
                f"Values [{min_val}, {max_val}] exceed float16 range "
                f"[{-float16_max}, {float16_max}]. Use smaller range or float32."
            )

    # Check for float16 precision issues
    value_range = max_val - min_val
    min_spacing = value_range / unique_size

    if dtype == torch.float16 and warn_precision:
        # Estimate float16 precision at this value range
        # Float16 precision degrades with larger absolute values
        max_abs_val = max(abs(min_val), abs(max_val))

        # Float16 precision formula: eps * value
        # For values around 1000, precision is ~0.5
        # For values around 100, precision is ~0.05
        float16_eps = 0.001  # Approximate relative precision
        estimated_precision = max_abs_val * float16_eps

        if min_spacing < estimated_precision * 2:  # 2x safety margin
            import warnings

            warnings.warn(
                f"Float16 precision warning: Spacing between values ({min_spacing:.4f}) "
                f"is close to float16 precision (~{estimated_precision:.4f}) at this range. "
                f"With {unique_size} unique values in range [{min_val}, {max_val}], "
                f"some values may become equal after float16 conversion.\n"
                f"Recommendations:\n"
                f"  1. Use smaller range (e.g., [{-max_abs_val / 2:.0f}, {max_abs_val / 2:.0f}])\n"
                f"  2. Use fewer unique values (reduce size along dim {dim})\n"
                f"  3. Use dtype=torch.float32 instead",
                UserWarning,
                stacklevel=2,
            )

    # Create result tensor
    result = torch.zeros(shape, dtype=torch.float32)

    # Flatten all dimensions except the unique dimension
    # This makes it easier to iterate over slices
    if dim == 0:
        # Special case: unique along first dimension
        for slice_idx in range(num_slices):
            # Generate unique values
            unique_ints = torch.randperm(unique_size, dtype=torch.float32)
            scaled = min_val + (unique_ints / unique_size) * value_range

            # Compute multi-dimensional index for this slice
            remaining_shape = shape[1:]
            multi_idx = []
            temp_idx = slice_idx
            for s in reversed(remaining_shape):
                multi_idx.insert(0, temp_idx % s)
                temp_idx //= s

            # Assign to result
            result[(slice(None),) + tuple(multi_idx)] = scaled

    elif dim == ndim - 1:
        # Special case: unique along last dimension (most common, optimized)
        result_flat = result.view(-1, unique_size)
        for i in range(num_slices):
            unique_ints = torch.randperm(unique_size, dtype=torch.float32)
            scaled = min_val + (unique_ints / unique_size) * value_range
            result_flat[i] = scaled

    else:
        # General case: unique along middle dimension
        # Move the unique dimension to the last position for easier processing
        perm = list(range(ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]

        # Permute and flatten
        result_permuted = result.permute(perm)
        result_flat = result_permuted.reshape(-1, unique_size)
        # Generate unique values for each slice
        for i in range(num_slices):
            unique_ints = torch.randperm(unique_size, dtype=torch.float32)
            scaled = min_val + (unique_ints / unique_size) * value_range
            result_flat[i] = scaled
        # Reshape and permute back
        result_permuted = result_flat.reshape(result_permuted.shape)
        result = result_permuted.permute(perm)
    # Convert to target dtype
    result = result.to(dtype)

    # Verify uniqueness after conversion (for float16)
    if dtype == torch.float16 and warn_precision:
        # Check a sample of slices for uniqueness
        sample_size = min(10, num_slices)
        issues_found = 0

        if dim == ndim - 1:
            # Check sample rows
            result_flat = result.view(-1, unique_size)
            for i in range(sample_size):
                unique_count = len(torch.unique(result_flat[i]))
                if unique_count < unique_size:
                    issues_found += 1
        elif dim == 0:
            # Check sample columns
            for j in range(min(sample_size, shape[1] if ndim > 1 else 1)):
                if ndim == 2:
                    unique_count = len(torch.unique(result[:, j]))
                else:
                    # For higher dimensions, check first slice
                    unique_count = len(
                        torch.unique(result[:, j, 0] if ndim > 2 else result[:, j])
                    )
                if unique_count < unique_size:
                    issues_found += 1

        if issues_found > 0:
            import warnings

            warnings.warn(
                f"Float16 precision loss detected: {issues_found}/{sample_size} sampled slices "
                f"have duplicate values after float16 conversion. "
                f"Consider using a smaller range or dtype=torch.float32.",
                UserWarning,
                stacklevel=2,
            )

    return result.contiguous()


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
def _to_cpu(result, device):
    """Move a result (tensor, tuple/list of tensors, or scalar) to CPU, with device validation."""
    if isinstance(result, torch.Tensor):
        assert result.device.type == device.type, (
            f"Output not on expected device. Expected {device}, got {result.device}"
        )
        return result.cpu()
    elif isinstance(result, (tuple, list)):
        cpu_items = [_to_cpu(r, device) for r in result]
        return type(result)(cpu_items)
    else:
        # Scalars (e.g. Python int from torch.numel()) are returned as-is
        return result


def _compile_and_run(fn, args, device, backend=None, needs_device=False, compile=True):
    """Compile and execute function on specified device/backend, returning result on CPU."""
    torch._dynamo.reset_code_caches()
    device = torch.device(device) if isinstance(device, str) else device
    device_args = [
        arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    device_kwargs = {"device": device} if needs_device else {}

    if compile:
        if backend:
            result = torch.compile(fn, backend=backend)(*device_args, **device_kwargs)
        else:
            result = torch.compile(fn)(*device_args, **device_kwargs)
    else:
        result = fn(*device_args, **device_kwargs)

    if isinstance(result, (int, float)):
        return result

    return _to_cpu(result, device)


def _assert_results_close(actual, expected, atol, rtol, comparison_name):
    """Assert two results are close with formatted error message."""
    if isinstance(actual, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected), (
            f"{comparison_name} mismatch: result types or lengths differ "
            f"(actual: {type(actual).__name__}[{len(actual)}], "
            f"expected: {type(expected).__name__}[{len(expected)}])"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_results_close(a, e, atol, rtol, f"{comparison_name}[{i}]")
    else:
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
    cpu_compile=None,
    target=None,
    run_eager=True,
    run_compile=True,
):
    """Compare Spyre execution against CPU for one or both Spyre execution paths.

    ``run_compile`` and ``run_eager`` select ``torch.compile`` vs eager on ``DEVICE``
    (spyre). At least one must be ``True``:

    - **Both** (default): ``run_compile=True``, ``run_eager=True`` — run compiled
      Spyre, then eager Spyre (same order as before).
    - **Compiled only**: ``run_compile=True``, ``run_eager=False``.
    - **Eager only**: ``run_compile=False``, ``run_eager=True``.
    - **Neither**: raises ``ValueError``.

    When ``cpu_compile`` is True, each selected Spyre path is also compared to CPU
    using the same compile flag (compiled vs compiled, or eager vs eager).

    Args:
        run_compile: Run the compiled path on Spyre.
        run_eager: Run the eager (non-compiled) path on Spyre.
    """
    # if this flag is explicitly passed in by the test, use it
    if cpu_compile is None:
        # the bool function parses all non-empty strings to true
        # if this env var is set at all, it gets marked as true
        cpu_compile = bool(os.getenv("TEST_COMPARE_CPU_COMPILE"))

    cpu_result = fn(*args)

    # Order: compiled first, then eager (matches prior [True, False] when both on).
    modes = tuple(
        compiled
        for compiled, enabled in ((True, run_compile), (False, run_eager))
        if enabled
    )
    if not modes:
        raise ValueError("At least one of run_compile or run_eager must be True")

    for compiled in modes:
        mode = "compiled" if compiled else "eager"
        spyre_result = (
            target
            if target is not None
            else _compile_and_run(
                fn, args, DEVICE, needs_device=needs_device, compile=compiled
            )
        )

        _assert_results_close(
            spyre_result, cpu_result, atol, rtol, f"{mode} spyre <-> cpu"
        )

        if cpu_compile:
            cpu_other_result = _compile_and_run(
                fn, args, "cpu", needs_device=needs_device, compile=compiled
            )
            _assert_results_close(
                spyre_result,
                cpu_other_result,
                atol,
                rtol,
                f"{mode} spyre <-> {mode} cpu",
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
