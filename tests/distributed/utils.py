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

import math
import unittest

import pytest
import torch


def _assert_tensor_equal(
    result, expected, dtype, message_prefix, *, rtol=1e-5, atol=1e-5
):
    if dtype.is_floating_point:
        matches = torch.allclose(result, expected, rtol=rtol, atol=atol)
        if not matches:
            # Find first mismatch for detailed error reporting
            diff = torch.abs(result - expected)
            mismatch_mask = diff > (atol + rtol * torch.abs(expected))
            if mismatch_mask.any():
                mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
                first_mismatch = mismatch_indices[0].tolist()
                first_mismatch_idx = (
                    tuple(first_mismatch)
                    if len(first_mismatch) > 1
                    else first_mismatch[0]
                )
                error_msg = (
                    f"{message_prefix}\n"
                    f"First mismatch at index {first_mismatch_idx}:\n"
                    f"  Expected: {expected[first_mismatch_idx]}\n"
                    f"  Got:      {result[first_mismatch_idx]}\n"
                    f"First 10 expected: {expected.flatten()[:10]}\n"
                    f"First 10 result:   {result.flatten()[:10]}"
                )
            else:
                error_msg = f"{message_prefix}: tensors not close"
            pytest.fail(error_msg)
    else:
        matches = torch.equal(result, expected)
        if not matches:
            # Find first mismatch for detailed error reporting
            mismatch_mask = result != expected
            if mismatch_mask.any():
                mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
                first_mismatch = mismatch_indices[0].tolist()
                first_mismatch_idx = (
                    tuple(first_mismatch)
                    if len(first_mismatch) > 1
                    else first_mismatch[0]
                )
                error_msg = (
                    f"{message_prefix}\n"
                    f"First mismatch at index {first_mismatch_idx}:\n"
                    f"  Expected: {expected[first_mismatch_idx]}\n"
                    f"  Got:      {result[first_mismatch_idx]}\n"
                    f"First 10 expected: {expected.flatten()[:10]}\n"
                    f"First 10 result:   {result.flatten()[:10]}"
                )
            else:
                error_msg = f"{message_prefix}: tensors not equal"
            pytest.fail(error_msg)


def _collective_rank_input(rank, shape, dtype):
    # Signed powers of two are exact in BF16 and DL16; 2**20 also detects an
    # unintended conversion through IEEE FP16, whose range is smaller.
    large = 2.0**20 if dtype == torch.bfloat16 else 16.0
    pattern = torch.tensor(
        [large, -large, 2.0**-20, -(2.0**-20), 0.5, -0.5, 1.5, -1.5], dtype=dtype
    )
    return (pattern.repeat(math.prod(shape) // len(pattern)) * (rank + 1)).reshape(
        shape
    )


def _bfloat16_rank_input(rank, shape):
    return _collective_rank_input(rank, shape, torch.bfloat16)


def _collective_expected(op, shape, world_size, dtype):
    if op == "broadcast":
        return _collective_rank_input(world_size - 1, shape, dtype)
    values = [_collective_rank_input(rank, shape, dtype) for rank in range(world_size)]
    if op in ("allgather", "gather"):
        return torch.cat(values, dim=0)
    return torch.stack(values).sum(dim=0)


def _bfloat16_collective_expected(op, shape, world_size):
    return _collective_expected(op, shape, world_size, torch.bfloat16)


def _native_bfloat16_tensor(device):
    from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout

    # Explicit native BF16 storage must not be reinterpreted as DL16.
    layout = SpyreTensorLayout([2, 64], [64, 1], DataFormats.BFLOAT16, {(0, 1): 128})
    return spyre_empty_with_layout([128], [1], torch.bfloat16, layout, device=device)


def _overpadded_collective_tensor(values, padded_width, device):
    from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout

    # Match the materialized input in Granite's compiled logits allgather:
    # logical [1, 24640], device [416, 1, 64], stride map [64, -1, 1].
    # The compiler retains the projection's 26624-element physical allocation.
    layout = SpyreTensorLayout(
        [padded_width // 64, 1, 64], [64, -1, 1], DataFormats.SEN169_FP16, {}
    )
    tensor = spyre_empty_with_layout(
        values.shape, values.stride(), values.dtype, layout, device=device
    )
    tensor.copy_(values)
    return tensor


def _dlfloat16_ulp(value: float) -> float:
    """Spacing between adjacent DLFloat16 values at magnitude of value (1-6-9 format).

    DLFloat16 format: 1 sign bit, 6 exponent bits (bias 31), 9 mantissa bits.
    ULP spacing is 2^(floor(log2(magnitude)) - 9).
    Smallest non-zero magnitude is 2^-31.
    """
    magnitude = max(abs(value), 2.0**-31)
    return 2.0 ** (math.floor(math.log2(magnitude)) - 9)


def _calculate_collective_tolerance(
    comm_size: int, max_expected: float, min_floor: float = 0.1
) -> float:
    """Calculate calibrated tolerance for reductions in DLFloat16 across world size.

    Each level of the addition tree introduces at most 1 ULP of rounding error
    at the partial-sum magnitude. Tolerance scales with tree depth * DLFloat16 ULP,
    with a minimum floor for small values.
    """
    if max_expected <= 0.0:
        return min_floor
    depth = 1 if comm_size <= 1 else math.ceil(math.log2(comm_size))
    calculated_tol = depth * _dlfloat16_ulp(max_expected)
    return max(min_floor, calculated_tol)


# DLFloat16 (1-6-9) holds every integer up to 2**10 exactly.
_DLFLOAT16_EXACT_INT_LIMIT = 1024.0

# Largest tensor (float16 elements, 512 KB) verified to compile for reduce and
# allreduce. 262208 (one stick more) fails in deeptools: "sbf-ddc: There must
# be at least one valid candidate" (L3DlOpsScheduler.cpp).
NUMEL_512KB = 262144


def _exact_collective_case(pattern, numel, rank, world_size):
    """(input, expected) float16 vectors whose SUM over ranks is exact.

    Every partial sum is representable, so any reduction order and any
    algorithm must reproduce ``expected`` bit for bit (assert with atol=0).
    ``disjoint_shards`` needs ``numel`` divisible by ``world_size``.
    """
    if pattern == "zeros":
        return torch.zeros(numel, dtype=torch.float16), torch.zeros(
            numel, dtype=torch.float16
        )
    if pattern == "ones":
        return torch.ones(numel, dtype=torch.float16), torch.full(
            (numel,), float(world_size), dtype=torch.float16
        )
    if pattern == "uniform_constant":
        return torch.full((numel,), 4.0, dtype=torch.float16), torch.full(
            (numel,), 4.0 * world_size, dtype=torch.float16
        )
    if pattern == "powers_of_two":
        total = sum(float(2 ** (r % 4)) for r in range(world_size))
        return torch.full((numel,), float(2 ** (rank % 4)), dtype=torch.float16), (
            torch.full((numel,), total, dtype=torch.float16)
        )
    if pattern == "sign_cancellation":
        # Even ranks +2.0, odd ranks -2.0: zero on even world sizes, else 2.0.
        sign = 1.0 if rank % 2 == 0 else -1.0
        total = 0.0 if world_size % 2 == 0 else 2.0
        return torch.full((numel,), sign * 2.0, dtype=torch.float16), torch.full(
            (numel,), total, dtype=torch.float16
        )
    if pattern == "single_rank_impulse":
        # Values wrap in 1..1021 (prime), so they never align with
        # power-of-two chunk sizes. Rank 0 contributes, the rest send zeros.
        impulse = (torch.arange(numel) % 1021 + 1).to(torch.float16)
        own = impulse if rank == 0 else torch.zeros(numel, dtype=torch.float16)
        return own.clone(), impulse
    if pattern == "disjoint_shards":
        # Each rank owns one slice, so a wrong chunk boundary or offset shows
        # up as a wrong value.
        shard = numel // world_size
        values = (torch.arange(numel) % 251 + 1).to(torch.float16)
        own = torch.zeros(numel, dtype=torch.float16)
        own[rank * shard : (rank + 1) * shard] = values[
            rank * shard : (rank + 1) * shard
        ]
        return own, values
    raise ValueError(f"Unknown exact pattern: {pattern}")


def _integer_range_case(numel, rank, world_size):
    """Rank r sends the integers r*numel .. (r+1)*numel - 1.

    Inputs are exact, so any error comes from the reduction. ``expected`` is
    the exact float32 sum.
    """
    own = torch.arange(rank * numel, (rank + 1) * numel, dtype=torch.float16)
    expected = (
        torch.arange(numel, dtype=torch.float32) * world_size
        + numel * world_size * (world_size - 1) / 2
    )
    return own, expected


def _integer_range_inputs_exact(numel, world_size):
    """Whether every input of ``_integer_range_case`` is exact in DLFloat16."""
    return numel * world_size - 1 <= _DLFLOAT16_EXACT_INT_LIMIT


def _random_ints_case(numel, rank, world_size):
    """Deterministic per-rank integers in [0, 512): exact inputs, exact float32 sum."""

    def ints(r):
        gen = torch.Generator().manual_seed(1_234_567 + 7_919 * r + numel)
        return torch.randint(0, 512, (numel,), generator=gen, dtype=torch.int64)

    expected = sum(ints(r) for r in range(world_size)).float()
    return ints(rank).to(torch.float16), expected


def _ramp_case(numel, rank, world_size, dtype):
    """(input, expected, atol) for rank r sending ``i / 10 + (r + 1)``.

    The inputs are not exact in the device format, so this is a tolerance
    check. Expected: ``world_size * i / 10 + sum_{r=1..world_size} r``.
    """
    own = torch.arange(numel, dtype=torch.float32).div(10.0).add(rank + 1).to(dtype)
    offset = world_size * (world_size + 1) / 2
    expected = (
        torch.arange(numel, dtype=torch.float32)
        .div(10.0)
        .mul(world_size)
        .add(offset)
        .to(dtype)
    )
    atol = (
        _calculate_collective_tolerance(
            world_size, float(expected.abs().max()), min_floor=0.1
        )
        if dtype == torch.float16
        else 0.2
    )
    return own, expected, atol


def _integer_case(case, numel, rank, world_size):
    """(input, expected) for a non-negative integer reduction.

    ``case`` is "range" or "random". Skips the test when a "range" case would
    hold integers DLFloat16 cannot represent at this world size.
    """
    if case == "range":
        if not _integer_range_inputs_exact(numel, world_size):
            raise unittest.SkipTest("inputs not exact in DLFloat16 at this world size")
        return _integer_range_case(numel, rank, world_size)
    return _random_ints_case(numel, rank, world_size)


def _check_work_handle(test_case, work, async_op):
    """Assert the Work handle matches ``async_op``; wait for async collectives."""
    if async_op:
        test_case.assertIsNotNone(work, "async_op=True must return a Work handle")
        work.wait()
        test_case.assertTrue(work.is_completed())
    else:
        test_case.assertIsNone(work, "async_op=False must return None")


def _chunk_numel(kind, world_size):
    """Sizes that probe chunk handling in the reduce-scatter based algorithms."""
    sticks = world_size if kind == "one_stick_per_rank" else world_size + 1
    return 64 * sticks


def _assert_reduction_close(result, expected, world_size, message_prefix):
    """Check a SUM of non-negative, DLFloat16-exact inputs against the exact sum.

    Two regions, so a tolerance never hides a bug that must be exact:
      * ``expected <= 1024``: every partial sum is a non-negative integer no
        larger than the total, hence exactly representable in any order, so
        the result must match bit for bit (atol=0).
      * ``expected > 1024``: rounding is possible; the error is bounded by
        accumulation depth * DLFloat16 ULP at the largest sum.
    """
    result = result.float()
    exact = expected <= _DLFLOAT16_EXACT_INT_LIMIT
    if exact.any():
        _assert_tensor_equal(
            result[exact],
            expected[exact],
            torch.float32,
            f"{message_prefix} (exact region: sums <= 1024)",
            rtol=0.0,
            atol=0.0,
        )
    if (~exact).any():
        atol = _calculate_collective_tolerance(
            world_size, float(expected.max()), min_floor=0.0
        )
        _assert_tensor_equal(
            result[~exact],
            expected[~exact],
            torch.float32,
            f"{message_prefix} (tolerance region: sums > 1024, atol={atol})",
            rtol=0.0,
            atol=atol,
        )
