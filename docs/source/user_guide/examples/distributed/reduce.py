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
import argparse
from datetime import timedelta
import json
import math
import os
import tempfile
from typing import Tuple
import regex as re
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"

# spyre-comms picks the reduce algorithm (COLL_REDUCE_ALGO forces one). Most are
# trees of depth ceil(log2 N); Progressive accumulates one RECV at a time, so its
# chain of DLFloat16 additions is N - 1 deep.
SEQUENTIAL_ALGOS = frozenset({"Progressive"})
STICK_ELEMENTS = 64  # 128 B stick / 2 B per float16
# 262144 float16 = 512 KB, the largest size verified to compile. 262208 (one
# stick more) fails in deeptools ("sbf-ddc: There must be at least one valid
# candidate") and leaves the other rank waiting, so it is not swept.
SWEEP_SIZES = (4096, 65536, 262144)


def accumulation_depth(comm_size: int) -> int:
    """Longest chain of DLFloat16 additions for the selected algorithm."""
    if comm_size <= 1:
        return 1
    if os.environ.get("COLL_REDUCE_ALGO") in SEQUENTIAL_ALGOS:
        return comm_size - 1
    return math.ceil(math.log2(comm_size))


def calculate_reduce_tolerance(comm_size: int, max_expected: float) -> float:
    """Calculate self-calibrating tolerance for the reduction.

    Each level of the addition chain introduces at most 1 ULP of DLFloat16 rounding
    error at the running partial-sum magnitude. The depth d is ceil(log2(N)) for
    the tree algorithms (COMPUTE_TREE in spyre-comms) and N - 1 for Progressive.
    DLFloat16 uses a 1-6-9 format: X = (-1)^s * 2^(e - 31) * (1 + m / 512)
    with 1 sign bit, 6 exponent bits (bias 31), and 9 fraction/mantissa bits.
    ULP of DLFloat16 at value v is 2^(floor(log2(v)) - 9), unlike IEEE Float16
    which is 1-5-10 (10 mantissa bits, bias 15).

    Args:
        comm_size: Total number of ranks (world size)
        max_expected: Maximum expected value in the reduced tensor

    Returns:
        atol: Absolute tolerance scaled by accumulation depth and DLFloat16 ULP
    """
    if max_expected <= 0.0:
        return 0.0
    ulp = dlfloat16_ulp(max_expected)
    return accumulation_depth(comm_size) * ulp


def check_close(
    result: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 0.0,
    rtol: float = 0.0,
    test_id: str = "",
    comm_rank: int = 0,
    comm_size: int = 1,
):
    """Check whether result matches expected tensor within tolerance.

    Inspired by verification utilities in torch_nnpa/tests (e.g. check_close with
    explicit rtol=0.0, atol=0.0 for exact invariants vs calibrated tolerance).

    Args:
        result: Output tensor from device (on CPU)
        expected: Golden expected tensor on CPU
        atol: Absolute tolerance (0.0 for exact tests)
        rtol: Relative tolerance (0.0 for exact tests)
        test_id: Identifier of the test case (e.g. 'EXACT_ZEROS_1D')
        comm_rank: Current process rank
        comm_size: Total world size
    """
    if result.shape != expected.shape:
        raise RuntimeError(
            f"[{comm_rank} of {comm_size}] [{test_id}] Shape mismatch: "
            f"got {result.shape} vs expected {expected.shape}"
        )

    if result.dtype != expected.dtype:
        raise RuntimeError(
            f"[{comm_rank} of {comm_size}] [{test_id}] Dtype mismatch: "
            f"got {result.dtype} vs expected {expected.dtype}"
        )

    is_close = torch.allclose(result, expected, atol=atol, rtol=rtol)
    diff = (result.float() - expected.float()).abs()
    max_diff = float(diff.max())
    arg_max = int(diff.argmax())

    flat_expected = expected.flatten()
    flat_result = result.flatten()

    if is_close:
        tol_str = (
            "exact (atol=0, rtol=0)"
            if (atol == 0.0 and rtol == 0.0)
            else f"atol={atol:.4f}, rtol={rtol}"
        )
        print(
            f"[{comm_rank} of {comm_size}] [{test_id:<28}] PASS "
            f"(shape={str(list(result.shape)):<12} max_diff={max_diff:.6f}, tol={tol_str})"
        )
    else:
        raise RuntimeError(
            f"[{comm_rank} of {comm_size}] [{test_id}] FAILED:\n"
            f"  Shape:    {result.shape}, Dtype: {result.dtype}\n"
            f"  Expected: {flat_expected[:8]} .. {flat_expected[-8:]}\n"
            f"  Got:      {flat_result[:8]} .. {flat_result[-8:]}\n"
            f"  Max absolute difference: {max_diff} at flat index {arg_max} "
            f"(expected {float(flat_expected[arg_max])}, got {float(flat_result[arg_max])})\n"
            f"  Allowed tolerance: atol={atol}, rtol={rtol}"
        )


def dlfloat16_ulp(value: float) -> float:
    """Spacing between adjacent DLFloat16 values at the magnitude of ``value``.

    DLFloat16 specification (IBM Research):
    - Format: X = (-1)^s * 2^(e - 31) * (1 + m / 512)
    - 1 sign bit (s)
    - 6 exponent bits (e), exponent bias b = 31 (smallest magnitude 2^-31)
    - 9 fraction/mantissa bits (m), giving 512 steps per binade
    - ULP spacing is 2^(floor(log2(magnitude)) - 9)
    - Smallest non-zero magnitude is 2^-31 * (1 + 1/512)
    - Largest finite magnitude is 2^32 * (1 + 510/512) (~8.573e9, 0x7FFE / 0xFFFE)
    - Merged NaN-Infinity is encoded as e=63, m=511 (0x7FFF / 0xFFFF)
    """
    magnitude = max(abs(value), 2.0**-31)
    return 2.0 ** (math.floor(math.log2(magnitude)) - 9)


def verify_negative_rejection(
    result: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    test_id: str,
    comm_rank: int,
):
    """Verify that assertion logic correctly rejects corrupted outputs (negative test).

    Ensures the test suite does not produce false passes when outputs violate
    numerical bounds.

    The check stays in float16 (DLFloat16 on device, 1-6-9 format). The perturbation
    must survive DLFloat16 rounding, so it is sized from the tolerance and the
    DLFloat16 ULP at the largest expected magnitude (both grow with world size):

        perturbation = 2 * atol + 2 * ULP

    Worst case, the result already sits atol away from expected in the opposite
    direction and rounding loses up to ULP/2, leaving a realized error of at
    least atol + 1.5 * ULP, which is strictly outside tolerance.
    """
    corrupted = result.clone().flatten()
    flat_expected = expected.flatten()
    max_magnitude = max(float(flat_expected.abs().max()), 1.0)
    perturbation = 2.0 * atol + 2.0 * dlfloat16_ulp(max_magnitude)
    corrupted[0] += perturbation

    # Guard against a vacuous test: the perturbation must remain outside
    # tolerance after DLFloat16 rounding.
    realized = abs(float(corrupted[0]) - float(flat_expected[0]))
    if not realized > atol:
        raise RuntimeError(
            f"[{comm_rank}] [{test_id}] Negative validation setup invalid: "
            f"perturbation {perturbation} realized as {realized} <= atol {atol}"
        )

    is_rejected = not torch.allclose(corrupted, flat_expected, atol=atol, rtol=0.0)
    if not is_rejected:
        raise RuntimeError(
            f"[{comm_rank}] [{test_id}] Negative validation failed: corrupted output was not rejected!"
        )


def random_rank_ints(num_elements: int, rank: int) -> torch.Tensor:
    """Deterministic per-rank integers in [0, 512) as int64.

    Every rank can regenerate every other rank's data, so the root builds the
    reference from the actual inputs. Integers in [0, 512) are exact in DLFloat16
    (1-6-9 format, 9 mantissa bits), so only the reduction itself rounds, and
    the reference is the correctly rounded exact sum (no float16 overflow up to 128 ranks).
    """
    gen = torch.Generator().manual_seed(1_234_567 + 7_919 * rank + num_elements)
    return torch.randint(0, 512, (num_elements,), generator=gen, dtype=torch.int64)


def generate_test_tensors(
    test_spec: dict, comm_rank: int, comm_size: int
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Generate input and golden expected tensors for a given parametrized test specification.

    Follows the comprehensive test matrix patterns used in torch_nnpa/tests/test_matmul.py
    covering small, medium, large, multidimensional (1D, 2D, 3D, 4D), exact invariants,
    and calibrated dynamic ranges.
    """
    shape = test_spec["shape"]
    pattern = test_spec["pattern"]
    num_elements = int(torch.prod(torch.tensor(shape)).item())

    if pattern == "zeros":
        input_tensor = torch.zeros(shape, dtype=torch.float16)
        expected_tensor = (
            torch.zeros(shape, dtype=torch.float16) if comm_rank == 0 else None
        )
        atol = 0.0

    elif pattern == "ones":
        input_tensor = torch.ones(shape, dtype=torch.float16)
        expected_tensor = (
            torch.full(shape, float(comm_size), dtype=torch.float16)
            if comm_rank == 0
            else None
        )
        atol = 0.0

    elif pattern == "uniform_constant":
        const_val = float(test_spec.get("value", 4.0))
        input_tensor = torch.full(shape, const_val, dtype=torch.float16)
        expected_tensor = (
            torch.full(shape, const_val * comm_size, dtype=torch.float16)
            if comm_rank == 0
            else None
        )
        atol = 0.0

    elif pattern == "powers_of_two":
        # Rank r sends 2^(r % 4). Sum over all ranks is exact integer power representation in fp16
        rank_val = float(2 ** (comm_rank % 4))
        input_tensor = torch.full(shape, rank_val, dtype=torch.float16)
        if comm_rank == 0:
            total_sum = sum(float(2 ** (r % 4)) for r in range(comm_size))
            expected_tensor = torch.full(shape, total_sum, dtype=torch.float16)
        else:
            expected_tensor = None
        atol = 0.0

    elif pattern == "single_rank":
        # Impulse: Only rank 0 contributes non-zero values, others contribute 0.0
        # Values wrap in 1..1021 (prime): exact in fp16/DLFloat16, and never
        # aligned with power-of-two chunk sizes.
        if comm_rank == 0:
            input_tensor = (
                (torch.arange(num_elements) % 1021 + 1).to(torch.float16).reshape(shape)
            )
            expected_tensor = input_tensor.clone()
        else:
            input_tensor = torch.zeros(shape, dtype=torch.float16)
            expected_tensor = None
        atol = 0.0

    elif pattern == "cancellation":
        # Even ranks send +2.0, odd ranks send -2.0 (exact zero cancellation on even world sizes)
        sign = 1.0 if (comm_rank % 2 == 0) else -1.0
        input_tensor = torch.full(shape, sign * 2.0, dtype=torch.float16)
        if comm_rank == 0:
            expected_val = 0.0 if (comm_size % 2 == 0) else 2.0
            expected_tensor = torch.full(shape, expected_val, dtype=torch.float16)
        else:
            expected_tensor = None
        atol = 0.0

    elif pattern == "range":
        start_value = comm_rank * num_elements
        end_value = start_value + num_elements
        input_tensor = torch.arange(
            start_value, end_value, dtype=torch.float16
        ).reshape(shape)
        if comm_rank == 0:
            flat_expected = torch.zeros(num_elements, dtype=torch.float16)
            for i in range(num_elements):
                flat_expected[i] = (
                    i * comm_size + num_elements * comm_size * (comm_size - 1) / 2
                )
            expected_tensor = flat_expected.reshape(shape)
            max_expected = float(expected_tensor.max())
            atol = calculate_reduce_tolerance(comm_size, max_expected)
        else:
            expected_tensor = None
            atol = 0.0
    elif pattern == "random_ints":
        input_tensor = random_rank_ints(num_elements, comm_rank).to(torch.float16)
        input_tensor = input_tensor.reshape(shape)
        if comm_rank == 0:
            total = sum(random_rank_ints(num_elements, r) for r in range(comm_size))
            expected_tensor = total.to(torch.float16).reshape(shape)
            atol = calculate_reduce_tolerance(comm_size, float(expected_tensor.max()))
        else:
            expected_tensor = None
            atol = 0.0

    elif pattern == "disjoint_shards":
        if num_elements % comm_size != 0:
            raise ValueError(
                f"{num_elements} elements do not split over {comm_size} ranks"
            )
        shard = num_elements // comm_size
        values = (torch.arange(num_elements) % 251 + 1).to(torch.float16)
        input_tensor = torch.zeros(num_elements, dtype=torch.float16)
        lo, hi = comm_rank * shard, (comm_rank + 1) * shard
        input_tensor[lo:hi] = values[lo:hi]
        input_tensor = input_tensor.reshape(shape)
        expected_tensor = values.reshape(shape) if comm_rank == 0 else None
        atol = 0.0

    else:
        raise ValueError(f"Unknown test pattern: {pattern}")

    return input_tensor, expected_tensor, atol


def execute_with_algo(step, show_algo: bool):
    """Run ``step()``; with ``show_algo`` also return the algorithms that ran.

    The Spyre profiler tags each collective's device activities with the
    spyre-comms algorithm that executed (``coll_algo``), so this also reports
    the cost-based default choice, not just a forced COLL_*_ALGO. Returns
    ``(result, algos)``; ``algos`` is None when not requested, and an empty list
    when the profiler build does not emit coll metadata.

    Fallback: spyre-comms prefixes compute kernel names with
    ``[Coll,Algo,Bytes,CollId]``. When that prefix is not stripped into
    ``coll_algo`` (e.g. a leading space defeats flex's parser), the algorithm
    is read from the event name instead.
    """
    if not show_algo:
        return step(), None
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        result = step()
        torch.spyre.synchronize()
    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_path = os.path.join(tmp_dir, "trace.json")
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            events = json.load(f)["traceEvents"]
    algos = {e["args"]["coll_algo"] for e in events if "coll_algo" in e.get("args", {})}
    if not algos:
        for e in events:
            m = re.match(r"\s*\[[^,\]]+,([^,\]]+),\d+", e.get("name", ""))
            if m:
                algos.add(m.group(1))
    return result, sorted(algos)


def execute_reduce_step(
    input_tensor: torch.Tensor, comm_rank: int, async_op: bool = False
):
    """Execute a single reduce collective across all ranks."""
    input_device = input_tensor.to(DEVICE)
    if async_op:
        work = dist.reduce(input_device, dst=0, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
    else:
        dist.reduce(input_device, dst=0, op=dist.ReduceOp.SUM)
    return input_device.to("cpu") if comm_rank == 0 else None


def range_fits_float16(num_elements: int, comm_size: int) -> bool:
    """Whether a ``range`` case is exact on input and finite on output.

    Inputs are the integers 0 .. comm_size * num_elements - 1. DLFloat16 (9
    mantissa bits) holds integers exactly only up to 1024, so larger inputs
    would be rounded on upload, an error the depth * ULP tolerance does not
    cover. The reference tensor is built in float16, so a sum above 65504
    becomes inf.
    """
    if comm_size * num_elements - 1 > 1024:
        return False
    largest = (num_elements - 1) * comm_size + (
        num_elements * comm_size * (comm_size - 1) / 2
    )
    return largest <= 65504.0


def get_test_matrix(
    num_elements: int = 128, comm_size: int = 1, size_sweep: bool = False
) -> list:
    """Return the structured test case matrix inspired by torch_nnpa/tests."""
    # Hardware constraints: minimum DMA transfer size = 128 bytes (64 float16 elements).
    # Multi-dimensional splitting requires inner dimension aligned to stick boundary (64 elements).
    inner_dim = 64
    rows_2d = max(2, num_elements // inner_dim)
    d1_3d = 2
    d2_3d = max(2, num_elements // (d1_3d * inner_dim))
    d1_4d = 2
    d2_4d = 2
    d3_4d = max(2, num_elements // (d1_4d * d2_4d * inner_dim))
    # Shard-per-rank size: a whole number of sticks per rank.
    sticks_per_rank = max(1, num_elements // (STICK_ELEMENTS * comm_size))
    shard_len = STICK_ELEMENTS * comm_size * sticks_per_rank

    matrix = [
        # --- Exact Invariant Test Cases (atol=0.0, rtol=0.0) ---
        {
            "id": "EXACT_ZEROS_1D",
            "shape": (max(64, num_elements),),
            "pattern": "zeros",
            "category": "exact",
        },
        {
            "id": "EXACT_ZEROS_2D",
            "shape": (rows_2d, inner_dim),
            "pattern": "zeros",
            "category": "exact",
        },
        {
            "id": "EXACT_ONES_1D_SMALL",
            "shape": (64,),
            "pattern": "ones",
            "category": "exact",
        },
        {
            "id": "EXACT_ONES_1D_MEDIUM",
            "shape": (max(64, num_elements),),
            "pattern": "ones",
            "category": "exact",
        },
        {
            "id": "EXACT_ONES_2D",
            "shape": (rows_2d, inner_dim),
            "pattern": "ones",
            "category": "exact",
        },
        {
            "id": "EXACT_ONES_3D",
            "shape": (d1_3d, d2_3d, inner_dim),
            "pattern": "ones",
            "category": "exact",
        },
        {
            "id": "EXACT_ONES_4D",
            "shape": (d1_4d, d2_4d, d3_4d, inner_dim),
            "pattern": "ones",
            "category": "exact",
        },
        {
            "id": "EXACT_POWERS_OF_TWO",
            "shape": (max(64, num_elements),),
            "pattern": "powers_of_two",
            "category": "exact",
        },
        {
            "id": "EXACT_SINGLE_RANK_IMPULSE",
            "shape": (max(64, num_elements),),
            "pattern": "single_rank",
            "category": "exact",
        },
        {
            "id": "EXACT_SIGN_CANCELLATION",
            "shape": (max(64, num_elements),),
            "pattern": "cancellation",
            "category": "exact",
        },
        {
            "id": "EXACT_UNIFORM_CONSTANT",
            "shape": (max(64, num_elements),),
            "pattern": "uniform_constant",
            "value": 4.0,
            "category": "exact",
        },
        # --- Calibrated Tree Tolerance Test Cases (atol = depth * ULP) ---
        {
            "id": "TOLERANCE_RANGE_SMALL",
            "shape": (64,),
            "pattern": "range",
            "category": "tolerance",
        },
        {
            "id": "TOLERANCE_RANGE_MEDIUM",
            "shape": (max(64, num_elements),),
            "pattern": "range",
            "category": "tolerance",
        },
        {
            "id": "TOLERANCE_RANGE_2D",
            "shape": (rows_2d, inner_dim),
            "pattern": "range",
            "category": "tolerance",
        },
        {
            "id": "TOLERANCE_RANGE_3D",
            "shape": (d1_3d, d2_3d, inner_dim),
            "pattern": "range",
            "category": "tolerance",
        },
        {
            "id": "TOLERANCE_RANGE_4D",
            "shape": (d1_4d, d2_4d, d3_4d, inner_dim),
            "pattern": "range",
            "category": "tolerance",
        },
        # --- One logical tensor spread across ranks (size scales, tensor count doesn't) ---
        # Each rank owns one disjoint slice; summing with zeros is exact, so any
        # wrong chunk boundary or offset shows up as a wrong value (atol=0).
        {
            "id": "EXACT_DISJOINT_SHARDS",
            "shape": (shard_len,),
            "pattern": "disjoint_shards",
            "category": "exact",
        },
        # Inputs are random integers (exact in float16 and DLFloat16) so only the
        # accumulation rounds. Sizes probe chunk handling in the reduce-scatter
        # based algorithms: exactly one stick per rank, a non-multiple, and the
        # smallest legal tensor.
        {
            "id": "CHUNK_ONE_STICK_PER_RANK",
            "shape": (STICK_ELEMENTS * comm_size,),
            "pattern": "random_ints",
            "category": "chunking",
        },
        {
            "id": "CHUNK_NON_MULTIPLE",
            "shape": (STICK_ELEMENTS * (comm_size + 1),),
            "pattern": "random_ints",
            "category": "chunking",
        },
    ]
    if size_sweep:
        matrix += [
            {
                "id": f"SWEEP_RANDOM_{n}",
                "shape": (n,),
                "pattern": "random_ints",
                "category": "sweep",
            }
            for n in SWEEP_SIZES
        ]
    fitting = []
    for spec in matrix:
        n = math.prod(spec["shape"])
        if spec["pattern"] == "range" and not range_fits_float16(n, comm_size):
            if os.environ.get("RANK", "0") == "0":
                print(
                    f"SKIP [{spec['id']}] range inputs/sums not exact in "
                    f"DLFloat16/float16 at {comm_size} ranks"
                )
            continue
        fitting.append(spec)
    return fitting


def run_test(
    comm_rank,
    comm_size,
    async_op=False,
    num_elements=128,
    test_case="all",
    size_sweep=False,
    show_algo=False,
):
    """Run reduce test matrix with exact invariant and tolerance verification."""
    global DEVICE

    print("=" * 80)
    print(
        f"[{comm_rank} of {comm_size}] Running Reduce Parametrized Test Suite "
        f"(elements={num_elements}, async={async_op}, filter='{test_case}', "
        f"algo={os.environ.get('COLL_REDUCE_ALGO', 'default')})"
    )
    print("=" * 80)

    test_matrix = get_test_matrix(num_elements, comm_size, size_sweep)
    passed_count = 0
    total_count = 0

    for test_spec in test_matrix:
        test_id = test_spec["id"]
        category = test_spec.get("category", "")

        # Filter by test_case name or category
        if test_case != "all" and test_case not in (
            test_id,
            category,
            test_spec["pattern"],
        ):
            continue

        total_count += 1
        input_tensor, expected_tensor, atol = generate_test_tensors(
            test_spec, comm_rank, comm_size
        )

        result, algos = execute_with_algo(
            lambda: execute_reduce_step(input_tensor, comm_rank, async_op=async_op),
            show_algo,
        )
        if comm_rank == 0 and algos is not None:
            print(
                f"[{comm_rank} of {comm_size}] [{test_id:<28}] algo="
                f"{', '.join(algos) if algos else 'unknown (no coll_algo in profiler trace)'}"
            )

        failure = None
        if comm_rank == 0:
            try:
                check_close(
                    result,
                    expected_tensor,
                    atol=atol,
                    rtol=0.0,
                    test_id=test_id,
                    comm_rank=comm_rank,
                    comm_size=comm_size,
                )
                # Perform negative validation on root
                verify_negative_rejection(
                    result,
                    expected_tensor,
                    atol=atol,
                    test_id=test_id,
                    comm_rank=comm_rank,
                )
            except RuntimeError as exc:
                failure = exc
        else:
            mode = " (async)" if async_op else ""
            print(
                f"[{comm_rank} of {comm_size}] [{test_id:<28}] Non-root rank completed{mode}"
            )

        # Only the root verifies, so share its verdict (gloo, CPU) before the
        # next case. Otherwise non-root ranks enter the next collective alone
        # and hang until the launcher timeout instead of failing fast.
        verdict = torch.tensor([0 if failure is None else 1], dtype=torch.int32)
        dist.broadcast(verdict, src=0)
        if failure is not None:
            raise failure
        if verdict.item():
            raise RuntimeError(
                f"[{comm_rank} of {comm_size}] [{test_id}] FAILED: "
                "root rank reported a mismatch"
            )
        passed_count += 1

    print("=" * 80)
    print(
        f"[{comm_rank} of {comm_size}] Summary: {passed_count}/{total_count} test cases passed."
    )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed reduce example with exact and tolerance checks"
    )
    parser.add_argument(
        "--async",
        dest="async_op",
        action="store_true",
        default=False,
        help="Launch reduce asynchronously (async_op=True)",
    )
    parser.add_argument(
        "--num-elements",
        dest="num_elements",
        type=int,
        default=128,
        help="Number of elements per rank tensor (default: 128)",
    )
    parser.add_argument(
        "--test-case",
        dest="test_case",
        type=str,
        default="all",
        help="Specific test case ID, category ('exact', 'tolerance', 'chunking', 'sweep'), "
        "pattern ('zeros', 'ones', 'range', 'random_ints', ...), or 'all'",
    )
    parser.add_argument(
        "--size-sweep",
        dest="size_sweep",
        action="store_true",
        default=False,
        help="Also run large-tensor cases up to 512 KB per rank",
    )
    parser.add_argument(
        "--show-algo",
        dest="show_algo",
        action="store_true",
        default=False,
        help="Profile each collective and print the spyre-comms algorithm that "
        "ran (needs a profiler build that emits coll_algo)",
    )
    args = parser.parse_args()

    # Check that the c10d backend was loaded properly
    if dist.distributed_c10d.is_backend_available(C10D_BACKEND) is False:
        raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
    if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
        raise RuntimeError(
            f"Error: Missing a C10 Backend for {'spyre'}! Expected {C10D_BACKEND}"
        )

    # Initialize the distributed environment
    print("# Initialize Distributed Group ")
    dist.init_process_group(
        f"cpu:gloo,spyre:{C10D_BACKEND}",
        timeout=timedelta(seconds=120),
    )

    comm_size = dist.get_world_size()
    comm_rank = dist.get_rank()

    run_test(
        comm_rank,
        comm_size,
        async_op=args.async_op,
        num_elements=args.num_elements,
        test_case=args.test_case,
        size_sweep=args.size_sweep,
        show_algo=args.show_algo,
    )

    dist.destroy_process_group()
