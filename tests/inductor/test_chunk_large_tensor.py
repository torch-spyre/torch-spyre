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

import gc
import os
import pytest
import torch

import utils_inductor

# ---------------------------------------------------------------------------
# Pointwise helper functions
# ---------------------------------------------------------------------------


def _exp_sigmoid(x):
    return torch.sigmoid(torch.exp(x))


def _relu_tanh(x):
    return torch.tanh(torch.relu(x))


def _neg_abs(x):
    return torch.neg(torch.abs(x))


def _mul(x, y):
    return x * y


def _add(x, y):
    return x + y


def _sub(x, y):
    return x - y


def _relu(x):
    return torch.relu(x)


def _where(x, y):
    return torch.where(x > 0, x, y)


def _chained(x, y):
    """cos(sin(x) * silu(y)) + x  --  tests deep fusion."""
    return torch.cos(torch.sin(x) * torch.nn.functional.silu(y)) + x


def _chained_deep(x, y):
    """Five-op chain: floor(mish(softplus(x))) <= 0 ? x : y  --  codegen stress."""
    return torch.where(
        torch.floor(torch.nn.functional.mish(torch.nn.functional.softplus(x))) <= 0.0,
        x,
        y,
    )


# fp16 chunking reorders floating-point ops; 0.1 accommodates accumulation error.
_ATOL = 1e-1

# ---------------------------------------------------------------------------
# Parametrisation helpers
# ---------------------------------------------------------------------------


def _expand_dict(d):
    """Append '-cores{N}' suffix to every key; treat None cores as 1."""
    out = {}
    for k, v in d.items():
        c = v[-1]
        if c is not None:
            out[f"{k}-cores{c}"] = v
        else:
            out[f"{k}-cores1"] = (*v[:-1], 1)
    return out


def _make_params(param_sets, expect_fail=None):
    """Build pytest.param list from (fn, shape_x, shape_y, dtype, cores) tuples."""
    ef = expect_fail or {}
    result = []
    for pid, args in param_sets.items():
        marks = []
        if pid in ef:
            act, reason = ef[pid]
            marks.append(
                pytest.mark.skip(reason=reason)
                if act == "skip"
                else pytest.mark.xfail(reason=reason)
            )
        result.append(pytest.param(*args, id=pid, marks=marks))
    return result


def _make_custom_params(param_sets, expect_fail=None):
    """Build pytest.param list from (factory_fn, fn, ref_fn) tuples."""
    ef = expect_fail or {}
    result = []
    for pid, (factory_fn, fn, ref_fn) in param_sets.items():
        marks = []
        if pid in ef:
            act, reason = ef[pid]
            marks.append(
                pytest.mark.skip(reason=reason)
                if act == "skip"
                else pytest.mark.xfail(reason=reason)
            )
        result.append(pytest.param(factory_fn, fn, ref_fn, id=pid, marks=marks))
    return result


# =============================================================================
# 1-D CHUNKING PARAMETER SETS
# =============================================================================
# Tuple layout: (fn, shape_x, shape_y_or_None, dtype, cores)
#
# MAX_SPAN = 268_435_456 bytes (256 MiB)
#
# For 1-D (N,) the pass splits on N-sticks.  Three trigger conditions (strict >):
#   core_split        = largest divisor of ceil(N/64) that is <= cores
#   per_core_span     = ceil(N/64 / core_split) * 64 * itemsize
#   unsplit_span      = N * itemsize
#   total_bytes       = N * itemsize
#   Trigger A: per_core_span > 256 MB
#   Trigger B: unsplit_span  > 256 MB   (= total_bytes for 1D)
#   Trigger C: total_bytes   > 256 MB * cores
#
# In 1D, total == unsplit, so B and C both scale with N.
# Category A (per_core > 256MB, total < 256MB*cores) IS reachable in 1D
# when N is large enough that unsplit > 256MB but total < 256MB*cores.
# Category B (total > 256MB*cores, per_core <= 256MB) IS reachable in 1D
# when cores is large enough to keep per_core safe while total exceeds threshold.

CHUNKING_1D_PARAM_SETS = {
    # -- [per_core>256MB, total<8GB] -------------------------------------------
    # 134_217_729 * 2 = 268_435_458 bytes > 256 MB
    # cores=1: core_split=1, per_core=268_435_458 B = 268.4 MB > 256 MB  [Trigger A]
    # total=268_435_458 B = 0.268 GB  < 256 MB*1 = 0.268 GB  (exact equal, NOT >)
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16": (
        _exp_sigmoid,
        (134_217_729,),
        None,
        torch.float16,
        1,
    ),
    # 134_217_757 * 2 = 268_435_514 bytes > 256 MB
    # cores=32: core_split=9 (largest divisor of ceil(134_217_757/64)=2_097_152 that is <=32)
    # per_core = ceil(2_097_152/9)*64*2 = 29_826_176 B = 29.8 MB  [per_core safe]
    # unsplit = 268_435_514 B = 268.4 MB > 256 MB  [Trigger B fires]
    # total = 0.268 GB < 256 MB*32 = 8.59 GB  [Trigger C does NOT fire]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16": (_relu_tanh, (134_217_757,), None, torch.bfloat16, 32),
    # 134_217_761 * 4 = 536_871_044 bytes > 256 MB
    # cores=7: core_split=3, per_core=ceil(2_097_153/3)*64*4 = 178_957_056 B = 179 MB  [per_core safe]
    # unsplit = 536_871_044 B = 537 MB > 256 MB  [Trigger B fires]
    # total = 0.537 GB < 256 MB*7 = 1.879 GB  [Trigger C does NOT fire]
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32": (_neg_abs, (134_217_761,), None, torch.float32, 7),
    # -- [per_core>256MB AND total>256MB*cores] --------------------------------
    # 4_412_345_679 * 4 = 17_649_382_716 bytes = 17.65 GB
    # cores=7: core_split=7, per_core=ceil(68_942_902/7)*64*4 = 2_521_340_416 B = 2521 MB > 256 MB  [A]
    # total = 17.65 GB > 256 MB*7 = 1.879 GB  [C]
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32": (_relu_tanh, (4_412_345_679,), None, torch.float32, 7),
    # 5_123_456_789 * 2 = 10_246_913_578 bytes = 10.25 GB
    # cores=32: core_split=13, per_core=ceil(80_054_012/13)*64*2 = 788_224_128 B = 788 MB > 256 MB  [A]
    # total = 10.25 GB > 256 MB*32 = 8.59 GB  [C]
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_fp16": (_neg_abs, (5_123_456_789,), None, torch.float16, 32),
    # -- [EXACT boundary: NO trigger] -----------------------------------------
    # 4_294_967_296 * 2 = 8_589_934_592 bytes = exactly 256 MB * 32 = 8 GiB
    # cores=1: per_core=8_589_934_592 B >> 256 MB  [A fires]
    # total = 8.59 GB > 256 MB*1 = 0.268 GB  [C fires]
    # This is NOT a no-trigger test -- it chunks heavily. Tests extreme 1D shape.
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "exact_8gb_fp16": (_exp_sigmoid, (4_294_967_296,), None, torch.float16, 1),
    # 67_108_864 * 4 = 268_435_456 bytes = exactly 256 MB = MAX_SPAN exactly
    # cores=1: per_core=268_435_456 B -- strict > means does NOT trigger  [NONE]
    # Boundary regression guard: chunking must NOT activate.
    # TODO: Error in codegen for ComputedBuffer (issue #2612)
    "exact_span_256mb_fp32": (_relu_tanh, (67_108_864,), None, torch.float32, 1),
    # -- [per_core>256MB, total<8GB] floor-division boundary -------------------
    # 402_653_184 * 2 = 805_306_368 bytes = 768 MB = 3 * 256 MB exactly
    # cores=1: core_split=1, per_core=805_306_368 B = 805 MB > 256 MB  [A fires]
    # total = 0.805 GB > 256 MB*1 = 0.268 GB  [C fires at cores=1 threshold]
    # Verifies exactly 3 equal chunks along the single dim.
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "floor_division_boundary_fp16": (
        _exp_sigmoid,
        (402_653_184,),
        None,
        torch.float16,
        1,
    ),
    # -- [per_core>256MB, total<256MB*cores] dtype coverage -------------------
    # bool: 4_987_654_321 * 1 = 4_987_654_321 bytes = 4.99 GB
    # cores=32: core_split=7, per_core=712_522_048 B = 712 MB > 256 MB  [A fires]
    # total = 4.99 GB < 256 MB*32 = 8.59 GB  [C does NOT fire]
    # TODO: FlexAllocator OutOfMemory
    "bool_unary_prime": (torch.logical_not, (4_987_654_321,), None, torch.bool, 32),
    # int32 binary: 134_217_730 * 4 = 536_870_920 bytes = 537 MB
    # cores=7: core_split=3 (largest divisor of ceil(134_217_730/64)=2_097_153 that is <=7)
    # per_core = ceil(2_097_153/3)*64*4 = 178_957_056 B = 179 MB  [per_core safe]
    # unsplit = 537 MB > 256 MB  [B fires]; total = 0.537 GB < 1.879 GB  [C safe]
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_binary_even": (torch.add, (134_217_730,), (134_217_730,), torch.int32, 7),
    # int64 binary: 134_217_761 * 8 = 1_073_742_088 bytes = 1.07 GB
    # cores=32: core_split=9, per_core=119_304_704 B = 119 MB  [per_core safe]
    # unsplit = 1.07 GB > 256 MB  [B fires]; total = 1.07 GB < 8.59 GB  [C safe]
    # TODO: FlexAllocator OutOfMemory
    "int64_add_prime": (torch.add, (134_217_761,), (134_217_761,), torch.int64, 32),
}

EXPECTED_FAILURES_1D = {
    "span_gt256_lt8gb_even_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_odd_bf16-cores32": (
        "xfail",
        "DtException: No valid prime factor but input still too large (LoopCoalescing.cpp)",
    ),
    "span_gt256_lt8gb_prime_fp32-cores7": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    "total_gt8gb_odd_fp32-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16-cores32": (
        "xfail",
        "DtException: isValidDimParam assertion failure in L3DlOpsScheduler.cpp",
    ),
    "exact_8gb_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "exact_span_256mb_fp32-cores1": (
        "xfail",
        "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)",
    ),
    "floor_division_boundary_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "bool_unary_prime-cores32": (
        "xfail",
        "DtException: isValidDimParam assertion failure in L3DlOpsScheduler.cpp",
    ),
    "int32_binary_even-cores7": (
        "xfail",
        "Missing Backend Codegen for Dtype/Op (int32 not supported)",
    ),
    "int64_add_prime-cores32": ("xfail", "DtException: EAR overflow detected"),
}
CHUNKING_1D_PARAM_SETS = _expand_dict(CHUNKING_1D_PARAM_SETS)

# =============================================================================
# 2-D CHUNKING PARAMETER SETS
# =============================================================================
# For (M, N) tensors:
#   core_split    = largest divisor of M that is <= cores
#   per_core_span = ceil(M / core_split) * N * itemsize
#   unsplit_span  = M * N * itemsize   (= total_bytes for 2D -- no outer batch)
#   total_bytes   = M * N * itemsize
#
# Unlike 3D+, there is no outer batch dim, so per_core_span and total are
# decoupled: it is possible to have total > 256MB*cores with per_core <= 256MB
# (Category B) by choosing a large M that divides evenly into cores.
# Category A (per_core>256MB, total<256MB*cores) and B both exist in 2D.

CHUNKING_2D_PARAM_SETS = {
    # -- [per_core>256MB] cores=1 so core_split=1, per_core=unsplit ------------
    # 36 * 4_000_000 * 2 = 288_000_000 B = 288 MB
    # cores=1: core_split=1, per_core=288 MB > 256 MB  [A fires]
    # total=288 MB > 256 MB*1=256 MB  [C also fires at this core count]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16": (
        _exp_sigmoid,
        (36, 4_000_000),
        None,
        torch.float16,
        1,
    ),
    # 36 * 4_000_001 * 2 = 288_000_072 B = 288 MB
    # cores=7: core_split=6, per_core=ceil(36/6)*4_000_001*2 = 48_000_768 B = 48 MB  [per_core safe]
    # unsplit=288 MB > 256 MB  [B fires]; total=0.288 GB < 256 MB*7=1.879 GB  [C safe]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16": (_relu_tanh, (36, 4_000_001), None, torch.bfloat16, 7),
    # 36 * 4_000_037 * 4 = 576_005_328 B = 576 MB
    # cores=32: core_split=18, per_core=ceil(36/18)*4_000_037*4 = 32_000_512 B = 32 MB  [per_core safe]
    # unsplit=576 MB > 256 MB  [B fires]; total=0.576 GB < 256 MB*32=8.59 GB  [C safe]
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32": (_neg_abs, (36, 4_000_037), None, torch.float32, 32),
    # -- [per_core>256MB AND total>256MB*cores] --------------------------------
    # 63 * 70_312_501 * 4 = 17_718_750_252 B = 17.72 GB
    # cores=7: core_split=7, per_core=ceil(63/7)*70_312_501*4 = 2_531_250_432 B = 2531 MB > 256 MB  [A]
    # total=17.72 GB > 256 MB*7=1.879 GB  [C]
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32": (_relu_tanh, (63, 70_312_501), None, torch.float32, 7),
    # 37 * 121_621_621 * 2 = 8_999_999_954 B = 9.0 GB
    # cores=32: core_split=1 (37 is prime), per_core=37*121_621_621*2 = 9000 MB > 256 MB  [A]
    # total=9.0 GB > 256 MB*32=8.59 GB  [C]
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_fp16": (_neg_abs, (37, 121_621_621), None, torch.float16, 32),
    # -- [per_core>256MB AND total>256MB*cores] dim extremes ------------------
    # 1 * 4_294_967_296 * 2 = 8_589_934_592 B = 8.59 GB
    # cores=1: core_split=1, per_core=8590 MB > 256 MB  [A]; total>256MB*1  [C]
    # TODO: FlexAllocator OutOfMemory
    "dim0_1_dim1_max_fp16": (_neg_abs, (1, 4_294_967_296), None, torch.float16, 1),
    # 5_500_123_456 * 1 * 2 = 11_000_246_912 B = 11.0 GB
    # cores=7: core_split=4, per_core=ceil(5_500_123_456/4)*1*2 = 176_003_950_592 B  [A]
    # total=11.0 GB > 256 MB*7=1.879 GB  [C]
    # TODO: FlexAllocator OutOfMemory
    "dim0_max_dim1_1_fp16": (_neg_abs, (5_500_123_456, 1), None, torch.float16, 7),
    # -- [per_core>256MB AND total>256MB*cores] broadcast ---------------------
    # 42_000 * 31_231 * 4 = 5_246_808_000 B = 5.25 GB
    # cores=7: core_split=7, per_core=ceil(42_000/7)*31_231*4 = 749_568_000 B = 750 MB > 256 MB  [A]
    # total=5.25 GB > 256 MB*7=1.879 GB  [C]
    # TODO: Missing Backend Codegen for Dtype/Op
    "broadcast_2d_1d_int32": (torch.add, (42_000, 31_231), (31_231,), torch.int32, 7),
    # 21_000 * 13_000 * 4 = 1_092_000_000 B = 1.09 GB
    # cores=1: core_split=1, per_core=1_096_704_000 B = 1097 MB > 256 MB  [A]
    # total=1.09 GB > 256 MB*1=0.268 GB  [C at this core count]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "broadcast_2d_1d_fp32": (_add, (21_000, 13_000), (13_000,), torch.float32, 1),
    # -- [NONE] dtype sanity -- no chunking triggers ---------------------------
    # bool: 37 * 7_000_001 * 1 = 259_000_037 B = 259 MB
    # cores=1: core_split=1, per_core=259_002_368 B = 259 MB < 256 MB  [A does NOT fire]
    # total=0.259 GB < 256 MB*1=0.268 GB  [C does NOT fire]
    # Note: 259 MB < MAX_SPAN=268 MB -- chunking does NOT activate.
    # Tests bool codegen path without chunking.
    # TODO: Missing Backend Codegen for Dtype/Op
    "bool_logical_or_even": (
        torch.logical_or,
        (37, 7_000_001),
        (37, 7_000_001),
        torch.bool,
        1,
    ),
    # -- [per_core>256MB, total<256MB*cores] dtype coverage -------------------
    # int32: 37 * 4_000_037 * 4 = 592_005_476 B = 592 MB
    # cores=32: core_split=1 (37 is prime), per_core=592_009_472 B = 592 MB > 256 MB  [A fires]
    # total=0.592 GB < 256 MB*32=8.59 GB  [C does NOT fire]
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_sub_prime": (torch.sub, (37, 4_000_037), (37, 4_000_037), torch.int32, 32),
    # int64: 36 * 4_000_001 * 8 = 1_152_000_288 B = 1.15 GB
    # cores=7: core_split=6, per_core=ceil(36/6)*4_000_001*8 = 192_003_072 B = 192 MB  [per_core safe]
    # unsplit=1.15 GB > 256 MB  [B fires]; total=1.15 GB < 1.879 GB  [C safe]
    # TODO: Missing Backend Codegen for Dtype/Op
    "int64_add_odd": (torch.add, (36, 4_000_001), (36, 4_000_001), torch.int64, 7),
    # bf16: 36 * 4_000_000 * 2 = 288_000_000 B = 288 MB
    # cores=1: core_split=1, per_core=288 MB > 256 MB  [A fires]
    # total=0.288 GB > 256 MB*1=0.268 GB  [C also fires at cores=1]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "bf16_add_even": (_add, (36, 4_000_000), (36, 4_000_000), torch.bfloat16, 1),
    # -- [NONE] mixed precision codegen tests (tiny shapes) -------------------
    # 128 * 32 * 2 = 8_192 B -- well under any threshold. Tests mixed-dtype codegen.
    # TODO: Error in codegen for ComputedBuffer (issue #2612)
    "mixed_fp16_bf16_add": (
        torch.add,
        (128, 32),
        (128, 32),
        (torch.float16, torch.bfloat16),
        1,
    ),
    # TODO: Error in codegen for ComputedBuffer (issue #2612)
    "mixed_fp16_fp32_add": (
        torch.add,
        (128, 32),
        (128, 32),
        (torch.float16, torch.float32),
        1,
    ),
}

EXPECTED_FAILURES_2D = {
    "span_gt256_lt8gb_even_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_odd_bf16-cores7": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_prime_fp32-cores32": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    "total_gt8gb_odd_fp32-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dim0_1_dim1_max_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "dim0_max_dim1_1_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_2d_1d_int32-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "broadcast_2d_1d_fp32-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "bool_logical_or_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_sub_prime-cores32": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_add_odd-cores7": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "bf16_add_even-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "mixed_fp16_bf16_add-cores1": (
        "xfail",
        "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)",
    ),
    "mixed_fp16_fp32_add-cores1": (
        "xfail",
        "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)",
    ),
}
CHUNKING_2D_PARAM_SETS = _expand_dict(CHUNKING_2D_PARAM_SETS)

# =============================================================================
# 3-D CHUNKING PARAMETER SETS
# =============================================================================
# For (B, M, N) tensors:
#   core_split    = largest divisor of M that is <= cores
#   per_core_span = ceil(M / core_split) * N * B * itemsize
#   unsplit_span  = M * N * B * itemsize   (= total_bytes -- includes B)
#   total_bytes   = B * M * N * itemsize
#
# KEY DIFFERENCE FROM 1D/2D: selected_device_span_stride_elems includes ALL
# inner device dims, which for (B,M,N) = prod(device_size[device_dim+1:]) = N*B.
# So unsplit_span = total_bytes -- they are the same value.
# This means Category B (total>8GB, per_core<=256MB) is mathematically
# unreachable in 3D: as cores grows to bring per_core below 256MB, the
# total threshold (256MB*cores) grows at the same rate and the tensor
# drops below it before per_core ever reaches 256MB.
#
# Only two chunking categories exist for 3D:
#   [per_core>256MB]        per_core_span > 256MB (total may or may not exceed 256MB*cores)
#   [unsplit_span only]     per_core <= 256MB but total > 256MB (chunking still activates
#                           because unsplit_span = total >> 256MB)
#   [NONE]                  total <= 256MB (all conditions False)


CHUNKING_3D_PARAM_SETS = {
    # ==========================================================================
    # [NONE] No chunking -- sanity / regression guards
    # ==========================================================================
    # 32*1024*1024*2=67 MB; per_core(32)=ceil(1024/32)*1024*32*2=2 MB  [NONE]
    "small_no_chunk": (_mul, (32, 1024, 1024), (32, 1024, 1024), torch.float16, 32),
    # 32*4096*4096*2=1.07 GB; per_core(32)=ceil(4096/32)*4096*32*2=33 MB  [NONE]
    "medium_no_chunk": (_mul, (32, 4096, 4096), (32, 4096, 4096), torch.float16, 32),
    # 32*8192*1024*2=537 MB; per_core(32)=ceil(8192/32)*1024*32*2=17 MB  [NONE]
    "clearly_under_limit": (
        _mul,
        (32, 8192, 1024),
        (32, 8192, 1024),
        torch.float16,
        32,
    ),
    # ==========================================================================
    # [EXACT] Boundary shapes: exactly at threshold (strict > means no trigger)
    # ==========================================================================
    # 32*8192*16384*2=8_589_934_592 B; per_core(1)=8192*16384*32*2=8590 MB >> 256 MB
    # cores=1: core_split=1, per_core=8_589_934_592 B -- both per_core and total fire.
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "exact_8gb_span_256mb_fp16": (
        _sub,
        (32, 8192, 16384),
        (32, 8192, 16384),
        torch.float16,
        1,
    ),
    # N=16383: 32*8192*16383*2=8_589_410_304 B; per_core(7)=ceil(8192/4)*16383*32*2=2148 MB
    # per_core > 256 MB and total > 256 MB*7=1.879 GB -- both fire.
    # Hardware bug (DtException: Invalid xlat size) is unrelated to chunking.
    # TODO: DtException: Invalid xlat size (hardware bug unrelated to chunking)
    "threshold_just_under_8gb_fp16": (
        _add,
        (32, 8192, 16383),
        (32, 8192, 16383),
        torch.float16,
        7,
    ),
    # N=16385: 32*8192*16385*2=8_590_458_880 B; per_core(32)=ceil(8192/32)*16385*32*2=270 MB
    # per_core=270 MB > 256 MB  [per_core fires]; total=8.59 GB > 256 MB*32=8.59 GB  [C fires]
    # TODO: FlexAllocator OutOfMemory
    "threshold_just_over_8gb_fp16": (
        _mul,
        (32, 8192, 16385),
        (32, 8192, 16385),
        torch.float16,
        32,
    ),
    # ==========================================================================
    # [per_core>256MB] per_core fires, total may or may not exceed 256MB*cores
    # ==========================================================================
    # 36*2000*2000*2=288_000_000 B=288 MB; cores=1: core_split=1
    # per_core=ceil(2000/1)*2000*36*2=288_000_000 B=288 MB > 256 MB  [per_core fires]
    # total=0.288 GB > 256 MB*1=0.268 GB  [total also fires at cores=1]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16": (
        _exp_sigmoid,
        (36, 2000, 2000),
        None,
        torch.float16,
        1,
    ),
    # 33*2001*2001*2=264_264_066 B=264 MB; cores=7: core_split=3
    # per_core=ceil(2001/3)*2001*33*2=90_243_126 B=90 MB < 256 MB  [per_core safe]
    # unsplit=264 MB > 256 MB  [unsplit fires]; total=0.264 GB < 256 MB*7=1.879 GB
    # Note: per_core is safe at cores=7; chunking activates via unsplit only.
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16": (
        _relu_tanh,
        (33, 2001, 2001),
        None,
        torch.bfloat16,
        7,
    ),
    # 31*2003*2003*4=497_486_228 B=497 MB; cores=32: core_split=1 (31 is prime)
    # per_core=ceil(2003/1)*2003*31*4=497_486_228 B=497 MB > 256 MB  [per_core fires]
    # total=0.497 GB < 256 MB*32=8.59 GB  [total does NOT fire]
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32": (
        _neg_abs,
        (31, 2003, 2003),
        None,
        torch.float32,
        32,
    ),
    # ==========================================================================
    # [per_core>256MB AND total>256MB*cores]
    # ==========================================================================
    # 32*8192*17408*2=9_126_805_504 B=9.13 GB; cores=1: core_split=1
    # per_core=ceil(8192/1)*17408*32*2=9_126_805_504 B=9127 MB > 256 MB  [per_core fires]
    # total=9.13 GB > 256 MB*1=0.268 GB  [total fires]
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_even_fp16_c1": (_relu_tanh, (32, 8192, 17408), None, torch.float16, 1),
    # 11*8191*24001*4=8_682_695_684 B=8.08 GB; cores=7: core_split=1 (8191 is prime)
    # per_core=ceil(8191/1)*24001*11*4=8_682_695_684 B=8283 MB > 256 MB  [per_core fires]
    # total=8.08 GB > 256 MB*7=1.879 GB  [total fires]
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32_c7": (_add, (11, 8191, 24001), None, torch.float32, 7),
    # 37*8191*14401*2=8_729_302_174 B=8.13 GB; cores=32: core_split=1 (8191 is prime)
    # per_core=ceil(8191/1)*14401*37*2=8_729_302_174 B=8729 MB > 256 MB  [per_core fires]
    # total=8.13 GB < 256 MB*32=8.59 GB  [total does NOT fire]
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_bf16_c32": (_mul, (37, 8191, 14401), None, torch.bfloat16, 32),
    # -- dev: large batch + M --------------------------------------------------
    # 32*8192*17472*2=9_160_359_936 B=8.53 GB; cores=32: core_split=32
    # per_core=ceil(8192/32)*17472*32*2=286_326_784 B=286 MB > 256 MB  [per_core fires]
    # total=8.53 GB < 256 MB*32=8.59 GB  [total does NOT fire]
    # TODO: FlexAllocator OutOfMemory
    "dev_large_batch_large_m": (_mul, (32, 8192, 17472), None, torch.float16, 32),
    # 32*8192*17408*2=9_126_805_504 B=9.13 GB; cores=32: core_split=32
    # per_core=ceil(8192/32)*17408*32*2=285_212_672 B=285 MB > 256 MB  [per_core fires]
    # total=9.13 GB > 256 MB*32=8.59 GB  [total fires]
    # TODO: FlexAllocator OutOfMemory
    "dev_large_total_bytes_trigger": (_mul, (32, 8192, 17408), None, torch.float16, 32),
    # ==========================================================================
    # [SKIP] SIGABRT cases -- hardware crashes (tracked for regression only)
    # ==========================================================================
    # 32*8193*1740*2=912_372_480 B=912 MB; cores=32: core_split=3 (8193=3*2731)
    # per_core=ceil(8193/3)*1740*32*2=313_294_848 B=313 MB > 256 MB  [per_core fires]
    # total=0.912 GB < 256 MB*32=8.59 GB  [total does NOT fire]
    # SKIP: M non-mult-64 crashes dxp_standalone bundler (SIGABRT).
    "dev_prime_m_per_core_trigger": (
        _mul,
        (32, 8193, 1740),
        (32, 8193, 1740),
        torch.float16,
        32,
    ),
    # 7*8193*1740*2=199_595_880 B=200 MB; core_split=3, per_core=68 MB  [NONE]
    "dev_awkward_batch_7": (_mul, (7, 8193, 1740), (7, 8193, 1740), torch.float16, 32),
    # 32*1*524_288*2=33_554_432 B=34 MB; M=1 so _find_split_dim falls back to N  [NONE]
    "dev_case2_m1_giant_n": (
        _mul,
        (32, 1, 524_288),
        (32, 1, 524_288),
        torch.float16,
        32,
    ),
    # 1*1*8_388_608*2=16_777_216 B=17 MB  [NONE]
    "dev_case4_batch1_m1_flat_giant": (
        _mul,
        (1, 1, 8_388_608),
        (1, 1, 8_388_608),
        torch.float16,
        32,
    ),
    # 128*8193*1740*2=3_649_489_920 B=3.65 GB; core_split=3
    # per_core=ceil(8193/3)*1740*128*2=1_253_179_392 B=1253 MB > 256 MB  [per_core fires]
    # total=3.65 GB < 8.59 GB  [total does NOT fire]
    # SKIP: SIGABRT (M non-mult-64).
    "dev_case5_large_batch_prime_m": (
        _mul,
        (128, 8193, 1740),
        (128, 8193, 1740),
        torch.float16,
        32,
    ),
    # 1*8192*17408*2=285_212_672 B=285 MB; cores=32: core_split=32
    # per_core=ceil(8192/32)*17408*1*2=8_912_896 B=9 MB  [per_core safe]
    # unsplit=285 MB > 256 MB  [unsplit fires]; total=0.285 GB < 8.59 GB
    "dev_case1_batch1_large_N_total": (
        _mul,
        (1, 8192, 17408),
        (1, 8192, 17408),
        torch.float16,
        32,
    ),
    # 1*65537*4096*2=536_895_488 B=537 MB; cores=32: core_split=1 (65537 is prime)
    # per_core=ceil(65537/1)*4096*1*2=536_895_488 B=537 MB > 256 MB  [per_core fires]
    # total=0.537 GB < 8.59 GB  [total does NOT fire]
    # SKIP: SIGABRT.
    "dev_single_giant_dim": (
        _mul,
        (1, 65537, 4096),
        (1, 65537, 4096),
        torch.float16,
        32,
    ),
    # 32*8193*1740*4=1_824_744_960 B=1.82 GB; core_split=3
    # per_core=ceil(8193/3)*1740*32*4=626_589_696 B=627 MB > 256 MB  [per_core fires]
    # SKIP: SIGABRT (M non-mult-64).
    "dev_float32_prime_M": (
        _mul,
        (32, 8193, 1740),
        (32, 8193, 1740),
        torch.float32,
        32,
    ),
    # ==========================================================================
    # Broadcast shapes [per_core>256MB AND total>256MB*cores at cores=1]
    # ==========================================================================
    # (32,8192,17408) fp16 at cores=1: core_split=1; per_core=8192*17408*32*2=9127 MB > 256 MB
    # total=9.13 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "broadcast_batch_dim_fp16": (
        _mul,
        (32, 8192, 17408),
        (1, 8192, 17408),
        torch.float16,
        1,
    ),
    # (16,8192,17408) fp32 at cores=1: core_split=1; per_core=8192*17408*16*4=9127 MB > 256 MB
    # total=17.2 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "broadcast_m_dim_fp32": (_add, (16, 8192, 17408), (16, 1, 17408), torch.float32, 1),
    # 3-D + 1-D broadcast (mismatched ndim): 21_000*321*64*2=864_864_000 B=865 MB
    # cores=1: core_split=1; per_core=ceil(321/1)*64*21_000*2=864_864_000 B=865 MB > 256 MB
    # total=0.865 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "broadcast_3d_1d_fp16": (_add, (21_000, 321, 64), (64,), torch.float16, 1),
    # SKIP: SIGABRT (broadcast + non-mult-64 M).
    "dev_broadcast": (_mul, (32, 8193, 1740), (1, 8193, 1740), torch.float16, 32),
    # ==========================================================================
    # Basic binary chunking (bfloat16) [per_core>256MB AND total>256MB*cores]
    # ==========================================================================
    # 32*8192*17408*2=9_126_805_504 B=9.13 GB
    # cores=1: core_split=1; per_core=8192*17408*32*2=9127 MB > 256 MB
    # total=9.13 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "basic_chunking_bf16": (_add, (32, 8192, 17408), None, torch.bfloat16, 1),
    # ==========================================================================
    # Non-multiples of 64 (stick-alignment stress)
    # ==========================================================================
    # M=8193, N=17409: 32*8193*17409*2=9_128_697_984 B=9.13 GB
    # core_split(32)=3; per_core=ceil(8193/3)*17409*32*2=3_042_863_616 B=3043 MB > 256 MB
    # total=9.13 GB > 256 MB*32=8.59 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "both_m_n_not_multiple_64_fp16": (_sub, (32, 8193, 17409), None, torch.float16, 32),
    # M=8191 prime: 32*8191*17408*2=9_124_806_656 B=9.12 GB
    # core_split(7)=1 (8191 is prime); per_core=8191*17408*32*2=9125 MB > 256 MB
    # total=9.12 GB > 256 MB*7=1.879 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "prime_m_dim_fp16": (_add, (32, 8191, 17408), None, torch.float16, 7),
    # ==========================================================================
    # Dtype coverage (one shape per dtype)
    # ==========================================================================
    # bool unary: 22*8192*17408*1=3_138_584_576 B=3.14 GB
    # cores=1: core_split=1; per_core=8192*17408*22*1=3139 MB > 256 MB
    # total=3.14 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "bool_logical_and": (torch.logical_and, (22, 8192, 17408), None, torch.bool, 1),
    # int32 unary: 11*8192*17408*4=6_251_855_872 B=6.25 GB
    # core_split(7)=1 (8192%7!=0, largest div<=7 is 4 -- wait 8192/4=2048 -- actually core_split=4)
    # per_core=ceil(8192/4)*17408*11*4=1_562_963_968 B=1563 MB > 256 MB
    # total=6.25 GB > 256 MB*7=1.879 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "int32_add_even": (torch.add, (11, 8192, 17408), None, torch.int32, 7),
    # int64 unary: 6*8192*17408*8=6_835_449_856 B=6.84 GB
    # core_split(32)=32; per_core=ceil(8192/32)*17408*6*8=213_909_504 B=214 MB < 256 MB
    # total=6.84 GB < 256 MB*32=8.59 GB  [per_core safe, total safe -- chunking via unsplit only]
    # TODO: FlexAllocator OutOfMemory
    "int64_add_prime": (torch.add, (6, 8192, 17408), None, torch.int64, 32),
    # float32 unary: 11*8192*8192*4=3_623_878_656 B=3.44 GB
    # core_split(7)=4; per_core=ceil(8192/4)*8192*11*4=905_969_664 B=906 MB > 256 MB
    # total=3.44 GB > 256 MB*7=1.879 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "float32_large_mul": (_mul, (11, 8192, 8192), None, torch.float32, 7),
    # ==========================================================================
    # Op variety and misc
    # ==========================================================================
    # 5-op chain: 32*8192*1740*2=912_453_120 B=912 MB
    # cores=1: core_split=1; per_core=8192*1740*32*2=912_453_120 B=912 MB > 256 MB
    # total=0.91 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: Error in codegen for ComputedBuffer (issue #2612)
    "deep_five_op_chain": (
        _chained_deep,
        (32, 8192, 1740),
        (32, 8192, 1740),
        torch.float16,
        1,
    ),
    # M=N=8193 prime: 1*8193*8193*2=134_261_298 B=134 MB
    # core_split(7)=1; per_core=8193*8193*1*2=134 MB < 256 MB
    # total=0.134 GB < 256 MB*7=1.879 GB  [NONE -- sanity guard, must NOT chunk]
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "prime_m_prime_n_fp16": (_mul, (1, 8193, 8193), (1, 8193, 8193), torch.float16, 7),
    # Empty tensor: zero elements.  Must not crash the pass.
    "empty_tensor_fp16": (_add, (32, 0, 17408), (32, 0, 17408), torch.float16, 1),
    # ==========================================================================
    # Padding shapes (trailing / donut dim=1) [per_core>256MB AND total>256MB*cores]
    # ==========================================================================
    # Trailing 1: 8192*524288*1*2=8_589_934_592 B=8.59 GB
    # cores=1: core_split=1; per_core=8192*524288*1*2=8590 MB > 256 MB
    # total=8.59 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "trailing_padding": (_add, (8192, 524288, 1), None, torch.float16, 1),
    # Donut (middle dim=1): 256*1*16_777_216*2=8_589_934_592 B=8.59 GB
    # cores=1: core_split=1; per_core=1*16_777_216*256*2=8590 MB > 256 MB
    # total=8.59 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "donut_padding": (_mul, (256, 1, 16_777_216), None, torch.float16, 1),
    # ==========================================================================
    # Misc extreme shapes
    # ==========================================================================
    # All-1 except M: 1*3_000_000_000*1*2=6_000_000_000 B=6 GB; B=1, N=1 so split on M
    # cores=1: core_split=1; per_core=3_000_000_000*1*1*2=6000 MB > 256 MB
    # total=6 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "all_ones_except_m": (
        _add,
        (1, 3_000_000_000, 1),
        (1, 3_000_000_000, 1),
        torch.float16,
        1,
    ),
    # All-1 except batch: 4_800_000_000*1*1*2=9_600_000_000 B=9.6 GB; M=N=1 so split on batch
    # core_split(7)=7; per_core=ceil(4_800_000_000/7)*1*1*2=1_371_428_572 B=1371 MB > 256 MB
    # total=9.6 GB > 256 MB*7=1.879 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "all_ones_except_batch": (_add, (4_800_000_000, 1, 1), None, torch.float16, 7),
    # Perfect cube: 1408*1408*1408*2=5_583_457_280 B=5.58 GB
    # cores=1: core_split=1; per_core=1408*1408*1408*2=5584 MB > 256 MB
    # total=5.58 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "perfect_cube_fp16": (
        _mul,
        (1408, 1408, 1408),
        (1408, 1408, 1408),
        torch.float16,
        1,
    ),
    # Prime sandwich: 16384*7*16384*2=3_758_096_384 B=3.76 GB
    # split dim is M=7 (outermost non-size-1). core_split(32)=7;
    # per_core=ceil(7/7)*16384*16384*2=536 MB > 256 MB
    # total=3.76 GB < 256 MB*32=8.59 GB  [per_core>256MB only]
    # TODO: FlexAllocator OutOfMemory
    "prime_sandwich_fp16": (
        _mul,
        (16384, 7, 16384),
        (16384, 7, 16384),
        torch.float16,
        32,
    ),
    # batch=1, M=1, only N splittable: 1*1*3_000_000_000*2=6_000_000_000 B=6 GB
    # cores=1: split falls to N; per_core=3_000_000_000*1*1*2=6000 MB > 256 MB
    # total=6 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "batch1_m1_only_n_splittable": (
        _mul,
        (1, 1, 3_000_000_000),
        None,
        torch.float16,
        1,
    ),
    # where conditional: 22*8192*17408*2=6_275_006_464 B=6.28 GB
    # cores=1: core_split=1; per_core=8192*17408*22*2=6275 MB > 256 MB
    # total=6.28 GB > 256 MB*1=0.268 GB  [per_core>256MB AND total>256MB*cores]
    # TODO: FlexAllocator OutOfMemory
    "where_conditional": (
        _where,
        (22, 8192, 17408),
        (22, 8192, 17408),
        torch.float16,
        1,
    ),
    # ==========================================================================
    # Additional tests: local_* covers Groups 1-14 of TestPointwiseChunking
    # ==========================================================================
    # Group 1 -- sanity / regression guards
    # 1*128*128*2=32_768 B; per_core(32)=ceil(128/32)*128*1*2=1024 B=0.001 MB  [NONE]
    "local_1_1_small_exp_sigmoid": (
        _exp_sigmoid,
        (1, 128, 128),
        None,
        torch.float16,
        32,
    ),
    # 32*1024*1024*2=67_108_864 B; per_core(32)=ceil(1024/32)*1024*32*2=2_097_152 B=2 MB  [NONE]
    "local_1_2_medium_exp_sigmoid": (
        _exp_sigmoid,
        (32, 1024, 1024),
        None,
        torch.float16,
        32,
    ),
    # 32*8192*8192*2=4_294_967_296 B=4 GB; per_core(32)=134 MB < 256 MB, total=4 GB < 8.59 GB  [NONE]
    "local_1_3_just_under_8gb": (
        _exp_sigmoid,
        (32, 8192, 8192),
        None,
        torch.float16,
        32,
    ),
    # Same shape as local_1_3 (32,8192,8192). per_core(32)=134 MB < 256 MB, total=4 GB < 8.59 GB  [NONE]
    "local_1_4_just_under_8gb": (
        _exp_sigmoid,
        (32, 8192, 8192),
        None,
        torch.float16,
        32,
    ),
    # Group 2 -- per_core>256MB AND total>256MB*cores, chunking MUST fire
    # 32*8192*17408*2=9_126_805_504 B=9.13 GB
    # core_split(32)=32; per_core=285 MB > 256 MB; total=9.13 GB > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "local_2_1_basic_chunking": (
        _exp_sigmoid,
        (32, 8192, 17408),
        None,
        torch.float16,
        32,
    ),
    # 32*8192*16512*2=8_657_043_456 B=8.66 GB
    # core_split(32)=32; per_core=271 MB > 256 MB; total=8.66 GB > 8.59 GB
    # Numerical mismatch after chunking: max_diff=49.5 >> atol=0.1 (relu+tanh)
    "local_2_2_slightly_over_8gb": (
        _relu_tanh,
        (32, 8192, 16512),
        None,
        torch.float16,
        32,
    ),
    # 32*24576*17408*2=27_380_416_512 B=27.4 GB
    # core_split(32)=32; per_core=856 MB > 256 MB; total=27.4 GB > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "local_2_3_very_large_abs_neg": (
        _neg_abs,
        (32, 24576, 17408),
        None,
        torch.float16,
        32,
    ),
    # 35_651_584*1*128*2=9_126_805_504 B=9.13 GB
    # core_split(32)=32; per_core=9127 MB > 256 MB; total=9.13 GB > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "local_2_4_large_batch_dim": (
        _exp_sigmoid,
        (35651584, 1, 128),
        None,
        torch.float16,
        32,
    ),
    # Group 3 -- broadcast [per_core>256MB AND total>256MB*cores at cores=1]
    # (32,8192,17408) fp16 at cores=1: per_core=9127 MB > 256 MB; total=9.13 GB > 0.268 GB
    # TODO: FlexAllocator OutOfMemory
    "local_3_1_broadcast_batch": (
        _mul,
        (32, 8192, 17408),
        (1, 8192, 17408),
        torch.float16,
        1,
    ),
    # TODO: FlexAllocator OutOfMemory
    "local_3_2_broadcast_m_dim": (
        _add,
        (32, 8192, 17408),
        (32, 1, 17408),
        torch.float16,
        1,
    ),
    # TODO: FlexAllocator OutOfMemory
    "local_3_3_broadcast_n_dim": (
        _sub,
        (32, 8192, 17408),
        (32, 8192, 1),
        torch.float16,
        1,
    ),
    # Group 4 -- non-multiples of 64 (stick-alignment stress)
    # M=8193 small N: 32*8193*1740*2=912 MB; per_core=313 MB > 256 MB  SKIP: SIGABRT.
    "local_4_1_m_not_mult64_small": (
        _mul,
        (32, 8193, 1740),
        (32, 8193, 1740),
        torch.float16,
        32,
    ),
    # M=8193 large N: 32*8193*17408*2=9.13 GB; per_core=3043 MB  SKIP: SIGABRT.
    "local_4_1_m_not_mult64_large": (_mul, (32, 8193, 17408), None, torch.float16, 32),
    # N=17409 small: 8*128*17409*2=35.6 MB; per_core=1.1 MB < 256 MB  [NONE]
    "local_4_2_n_not_mult64_small": (
        _add,
        (8, 128, 17409),
        (8, 128, 17409),
        torch.float16,
        32,
    ),
    # N=17409 large: 22*8192*17409*2=6.28 GB; per_core=197 MB < 256 MB  [unsplit only]
    # TODO: FlexAllocator OutOfMemory
    "local_4_2_n_not_mult64_large": (
        _add,
        (22, 8192, 17409),
        (22, 8192, 17409),
        torch.float16,
        32,
    ),
    # Both M=8193 and N=17409: 22*8193*17409*2=6.28 GB; per_core=2100 MB > 256 MB
    # TODO: FlexAllocator OutOfMemory
    "local_4_3_both_not_mult64": (
        _sub,
        (22, 8193, 17409),
        (22, 8193, 17409),
        torch.float16,
        32,
    ),
    # Group 5 -- edge batch sizes
    # batch=1 small: 1*8192*17408*2=285 MB; per_core=285 MB > 256 MB; total > 0.268 GB
    # Chunking activates and produces correct result.
    "local_5_1_batch1_small": (
        _mul,
        (1, 8192, 17408),
        (1, 8192, 17408),
        torch.float16,
        1,
    ),
    # batch=1 large: 1*185_000*17408*2=6.44 GB; per_core=258 MB > 256 MB  [per_core only]
    # TODO: FlexAllocator OutOfMemory
    "local_5_1_batch1_large": (
        _mul,
        (1, 185_000, 17408),
        (1, 185_000, 17408),
        torch.float16,
        32,
    ),
    # batch=2 small: 2*8192*17408*2=570 MB; per_core=18 MB < 256 MB; total < 8.59 GB  [NONE]
    "local_5_2_batch2_small": (
        _add,
        (2, 8192, 17408),
        (2, 8192, 17408),
        torch.float16,
        32,
    ),
    # batch=2 large: 2*92_500*17408*2=6.44 GB; per_core=258 MB > 256 MB  [per_core only]
    # TODO: FlexAllocator OutOfMemory
    "local_5_2_batch2_large": (
        _mul,
        (2, 92_500, 17408),
        (2, 92_500, 17408),
        torch.float16,
        32,
    ),
    # batch=32 small M small N: 32*256*65536*2=1.07 GB; per_core=34 MB < 256 MB; total < 8.59 GB  [NONE]
    "local_5_3_batch32_sm_small": (
        _sub,
        (32, 256, 65536),
        (32, 256, 65536),
        torch.float16,
        32,
    ),
    # batch=32 large N: 32*256*390_000*2=6.39 GB; per_core=200 MB < 256 MB  [unsplit only]
    # TODO: FlexAllocator OutOfMemory
    "local_5_3_batch32_sm_large": (
        _sub,
        (32, 256, 390_000),
        (32, 256, 390_000),
        torch.float16,
        32,
    ),
    # Group 6 -- prime / odd dimensions
    # batch=23 (prime): 23*8192*17408*2=6.56 GB; per_core=205 MB < 256 MB  [unsplit only]
    # TODO: FlexAllocator OutOfMemory
    "local_6_1_prime_batch_dim": (
        _mul,
        (23, 8192, 17408),
        (23, 8192, 17408),
        torch.float16,
        32,
    ),
    # M=8191 prime: 22*8191*17408*2=6.27 GB; per_core=6274 MB > 256 MB  [per_core only]
    # TODO: FlexAllocator OutOfMemory
    "local_6_2_prime_m_dim": (
        _add,
        (22, 8191, 17408),
        (22, 8191, 17408),
        torch.float16,
        32,
    ),
    # All-odd B=22,M=8193,N=17409: 6.28 GB; per_core=2100 MB > 256 MB
    # TODO: FlexAllocator OutOfMemory
    "local_6_3_all_odd_dims": (
        _sub,
        (22, 8193, 17409),
        (22, 8193, 17409),
        torch.float16,
        32,
    ),
    # float32 prime M=8191: 11*8191*17408*4=6.27 GB; per_core=6274 MB > 256 MB
    # TODO: FlexAllocator OutOfMemory
    "local_6_4_prime_m_fp32": (_add, (11, 8191, 17408), None, torch.float32, 32),
    # Group 7 -- size-1 dims; _find_split_dim must pick correct non-size-1 dim
    # 1*1*3_000_000_000*2=6 GB; M=1 so split falls to N; per_core=6000 MB > 256 MB
    # TODO: FlexAllocator OutOfMemory
    "local_7_1_batch1_m1_n_only": (
        _mul,
        (1, 1, 3_000_000_000),
        (1, 1, 3_000_000_000),
        torch.float16,
        1,
    ),
    # 1*185_000*17408*2=6.44 GB; B=1 so picks M; per_core=258 MB > 256 MB
    # TODO: FlexAllocator OutOfMemory
    "local_7_2_multiple_size1_batch1": (
        _add,
        (1, 185_000, 17408),
        (1, 185_000, 17408),
        torch.float16,
        32,
    ),
    # Group 8 -- very large tensors, multiple chunks
    # 11*24576*17408*2=9.41 GB; per_core=294 MB > 256 MB; total=9.41 GB > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "local_8_1_multi_chunk_unary": (_mul, (11, 24576, 17408), None, torch.float16, 32),
    # 6*49152*17408*2=10.27 GB; per_core=321 MB > 256 MB; total=10.27 GB > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "local_8_2_large_n_unary": (_add, (6, 49152, 17408), None, torch.float16, 32),
    # Group 9 -- various pointwise op types
    # 22*8192*17408*2=6.28 GB; per_core=197 MB < 256 MB; total < 8.59 GB  [unsplit only]
    # TODO: FlexAllocator OutOfMemory
    "local_9_1_addition": (
        _add,
        (22, 8192, 17408),
        (22, 8192, 17408),
        torch.float16,
        32,
    ),
    # 32*8192*17408*2=9.13 GB; per_core=285 MB > 256 MB; total > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "local_9_2_relu_unary": (_relu, (32, 8192, 17408), None, torch.float16, 32),
    # Same shape as local_9_1; per_core=197 MB < 256 MB  [unsplit only]
    # TODO: FlexAllocator OutOfMemory
    "local_9_3_where_ternary": (
        _where,
        (22, 8192, 17408),
        (22, 8192, 17408),
        torch.float16,
        32,
    ),
    # Group 10 -- exact 8 GB boundary
    # 32*8192*16384*2=8_589_934_592 B = exactly 256 MB*32 = exactly 8 GiB
    # per_core == MAX_SPAN (strict > means NOT triggered); total == 256MB*32 (NOT triggered)
    # Chunking does NOT activate.
    "local_10_1_exactly_8gb_no_chunk": (
        _mul,
        (32, 8192, 16384),
        (32, 8192, 16384),
        torch.float16,
        32,
    ),
    # N=16448=16384+64: 22*8192*16448*2=5.93 GB; per_core=185 MB < 256 MB  [unsplit only]
    # TODO: FlexAllocator OutOfMemory
    "local_10_2_one_stick_over": (_add, (22, 8192, 16448), None, torch.float16, 32),
    # Group 11 -- 4-D pointwise mul  [per_core>256MB AND total>256MB*cores]
    # 2*32*4096*17408*2=9.13 GB; per_core=285 MB > 256 MB; total > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "local_11_1_4d_pointwise_mul": (
        _mul,
        (2, 32, 4096, 17408),
        None,
        torch.float16,
        32,
    ),
    # Group 12 -- awkward batch=7, sanity guard
    # 7*8193*1740*2=200 MB; per_core=69 MB < 256 MB; total < 8.59 GB  [NONE]
    "local_12_awkward_batch7": (_exp_sigmoid, (7, 8193, 1740), None, torch.float16, 32),
    # Group 13 -- long sequence  [per_core>256MB only]
    # 1*65537*4096*2=537 MB; per_core=537 MB > 256 MB; total < 8.59 GB  SKIP: SIGABRT.
    "local_13_long_sequence": (_exp_sigmoid, (1, 65537, 4096), None, torch.float16, 32),
    # Group 14 -- chained ops  [per_core>256MB only]
    # 32*8193*1740*2=912 MB; per_core=313 MB > 256 MB; total < 8.59 GB  SKIP: SIGABRT.
    "local_14_chained_ops": (
        _chained,
        (32, 8193, 1740),
        (32, 8193, 1740),
        torch.float16,
        32,
    ),
}

EXPECTED_FAILURES_3D = {
    # EXACT boundary
    "exact_8gb_span_256mb_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "threshold_just_under_8gb_fp16-cores7": ("xfail", "DtException: Invalid xlat size"),
    "threshold_just_over_8gb_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    # SPAN
    "span_gt256_lt8gb_even_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_odd_bf16-cores7": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_prime_fp32-cores32": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    # TOTAL
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_bf16_c32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_batch_large_m-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "dev_large_total_bytes_trigger-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    # SKIP: SIGABRT
    "dev_prime_m_per_core_trigger-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "dev_case5_large_batch_prime_m-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "dev_single_giant_dim-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "dev_float32_prime_M-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "dev_broadcast-cores32": ("skip", "Crashes with SIGABRT in dxp_standalone bundler"),
    # Broadcast
    "broadcast_batch_dim_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_m_dim_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "broadcast_3d_1d_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "basic_chunking_bf16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "both_m_n_not_multiple_64_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "prime_m_dim_fp16-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    # Dtype
    "bool_logical_and-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_even-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "int64_add_prime-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_large_mul-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "deep_five_op_chain-cores1": (
        "xfail",
        "Error in codegen for ComputedBuffer during pointwise operations (issue #2612)",
    ),
    "prime_m_prime_n_fp16-cores7": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    # Misc
    "all_ones_except_m-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "all_ones_except_batch-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "perfect_cube_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "batch1_m1_only_n_splittable-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "where_conditional-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "trailing_padding-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "donut_padding-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 2
    "local_2_1_basic_chunking-cores32": ("xfail", "DtException: EAR overflow detected"),
    "local_2_2_slightly_over_8gb-cores32": (
        "xfail",
        "Numerical mismatch after chunking: relu+tanh produces wrong results",
    ),
    "local_2_3_very_large_abs_neg-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "local_2_4_large_batch_dim-cores32": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    # Local group 3
    "local_3_1_broadcast_batch-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "local_3_2_broadcast_m_dim-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "local_3_3_broadcast_n_dim-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 4
    "local_4_1_m_not_mult64_small-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "local_4_1_m_not_mult64_large-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler (M non-mult-64 at any N)",
    ),
    "local_4_3_both_not_mult64-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 5
    "local_5_1_batch1_small-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    # Local group 6
    "local_6_2_prime_m_dim-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler (prime M=8191)",
    ),
    "local_6_3_all_odd_dims-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "local_6_4_prime_m_fp32-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 7
    "local_7_1_batch1_m1_n_only-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 8
    "local_8_1_multi_chunk_unary-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "local_8_2_large_n_unary-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 9
    "local_9_2_relu_unary-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    # Local group 10
    "local_10_2_one_stick_over-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 11
    "local_11_1_4d_pointwise_mul-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    # Local group 13, 14
    "local_13_long_sequence-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "local_14_chained_ops-cores32": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
}
CHUNKING_3D_PARAM_SETS = _expand_dict(CHUNKING_3D_PARAM_SETS)

# =============================================================================
# 4-D CHUNKING PARAMETER SETS
# =============================================================================
# For (A, B, M, N) tensors:
#   core_split    = largest divisor of M that is <= cores
#   per_core_span = ceil(M / core_split) * N * B * A * itemsize
#   unsplit_span  = M * N * B * A * itemsize   (= total_bytes)
#   total_bytes   = A * B * M * N * itemsize
#
# In 4D, selected_device_span_stride_elems = N * B * A (all inner dims).
# per_core_span includes all outer batch dims (A, B), so it is proportional
# to total_bytes. Category B (total>8GB, per_core<=256MB) is unreachable
# in 4D for the same reason as 3D -- the batch dims couple the two values.
# All 4D shapes that are large enough to trigger chunking land in:
#   [per_core>256MB]: per_core >> 256 MB (total may or may not exceed 256MB*cores)

CHUNKING_4D_PARAM_SETS = {
    # -- [per_core>256MB AND total>256MB*cores] --------------------------------
    # 4*4*8192*17408*2=4_563_402_752 B=4.56 GB; cores=1: core_split=1
    # per_core=4563 MB > 256 MB; total=4.56 GB > 0.268 GB
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16": (
        _exp_sigmoid,
        (4, 4, 8192, 17408),
        None,
        torch.float16,
        1,
    ),
    # 5*7*8191*17409*2=9_981_886_290 B=9.98 GB; cores=7: core_split=1 (8191 prime)
    # per_core=9982 MB > 256 MB; total=9.98 GB > 1.879 GB
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16": (
        _relu_tanh,
        (5, 7, 8191, 17409),
        None,
        torch.bfloat16,
        7,
    ),
    # 3*11*8191*8193*4=8_857_935_612 B=8.86 GB; cores=32: core_split=1 (8191 prime)
    # per_core=8858 MB > 256 MB; total=8.86 GB > 8.59 GB
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32": (
        _neg_abs,
        (3, 11, 8191, 8193),
        None,
        torch.float32,
        32,
    ),
    # 6*5*8192*17408*2=8_556_380_160 B=8.56 GB; cores=1: core_split=1
    # per_core=8556 MB > 256 MB; total=8.56 GB > 0.268 GB
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_even_fp16_c1": (
        _relu_tanh,
        (6, 5, 8192, 17408),
        None,
        torch.float16,
        1,
    ),
    # 3*5*8191*17409*4=8_556_469_260 B=8.56 GB; cores=7: core_split=1 (8191 prime)
    # per_core=8557 MB > 256 MB; total=8.56 GB > 1.879 GB
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32_c7": (_add, (3, 5, 8191, 17409), None, torch.float32, 7),
    # 37*1*8192*14801*2=8_972_423_168 B=8.97 GB; cores=32: core_split=32
    # per_core=281 MB > 256 MB; total=8.97 GB > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_fp16_c32": (
        _neg_abs,
        (37, 1, 8192, 14801),
        None,
        torch.float16,
        32,
    ),
    # 2*32*4096*17408*2=9_126_805_504 B=9.13 GB; cores=32: core_split=32
    # per_core=285 MB > 256 MB; total=9.13 GB > 8.59 GB
    # TODO: FlexAllocator OutOfMemory
    "dev_4D_large": (_mul, (2, 32, 4096, 17408), None, torch.float16, 32),
    # -- [per_core>256MB] dtype coverage --------------------------------------
    # bf16: 2*4*256*1_048_576*2=4_294_967_296 B=4.29 GB; cores=1: core_split=1
    # per_core=4295 MB > 256 MB; total=4.29 GB > 0.268 GB
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "bf16_add_even": (
        torch.add,
        (2, 4, 256, 1_048_576),
        (2, 4, 256, 1_048_576),
        torch.bfloat16,
        1,
    ),
    # bool: 7*5*8192*17408*1=4_990_664_704 B=4.99 GB; cores=7: core_split=4
    # per_core=1248 MB > 256 MB; total=4.99 GB > 1.879 GB
    # TODO: Missing Backend Codegen for Dtype/Op
    "bool_logical_or": (
        torch.logical_or,
        (7, 5, 8192, 17408),
        (7, 5, 8192, 17408),
        torch.bool,
        7,
    ),
    # int32: 13*1*8192*19801*4=8_434_667_008 B=8.43 GB; cores=32: core_split=32
    # per_core=264 MB < 256 MB  [per_core safe]; unsplit=8.43 GB >> 256 MB  [unsplit fires]
    # total=8.43 GB < 8.59 GB  [total safe]
    # This is the only 4D case where per_core is safe but unsplit still triggers.
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_sub_prime": (torch.sub, (13, 1, 8192, 19801), None, torch.int32, 32),
    # int64: 2*8*8192*4096*8=4_294_967_296 B=4.29 GB; cores=1: core_split=1
    # per_core=4295 MB > 256 MB; total=4.29 GB > 0.268 GB
    # TODO: Missing Backend Codegen for Dtype/Op
    "int64_mul_even": (_mul, (2, 8, 8192, 4096), (2, 8, 8192, 4096), torch.int64, 1),
    # fp32: 2*4*8192*8194*4=2_148_007_936 B=2.15 GB; cores=1: core_split=1
    # per_core=2148 MB > 256 MB; total=2.15 GB > 0.268 GB
    "float32_add": (
        torch.add,
        (2, 4, 8192, 8194),
        (2, 4, 8192, 8194),
        torch.float32,
        1,
    ),
    # -- [per_core>256MB] ops / shapes ----------------------------------------
    # 1*1*65537*4096*2=536_895_488 B=537 MB; cores=32: core_split=1 (65537 prime)
    # per_core=537 MB > 256 MB; total=0.537 GB < 8.59 GB  [per_core only]
    # TODO: FlexAllocator OutOfMemory
    "single_giant_prime_dim_fp16": (
        _exp_sigmoid,
        (1, 1, 65537, 4096),
        None,
        torch.float16,
        32,
    ),
    # 3*7*8191*8191*2=2_817_636_066 B=2.82 GB; cores=7: core_split=1 (8191 prime)
    # per_core=2818 MB > 256 MB  SKIP: SIGABRT.
    "seven_cores_unary": (_exp_sigmoid, (3, 7, 8191, 8191), None, torch.float16, 7),
    # -- [per_core>256MB] waterfall / padding shapes --------------------------
    # 2*4*256*1_048_576*2=4_294_967_296 B=4.29 GB; cores=1: core_split=1
    # per_core=4295 MB > 256 MB
    # Tests dim-scan order where M=256 is inner and N=1_048_576 is largest.
    # TODO: FlexAllocator OutOfMemory
    "waterfall_ascending_fp16": (
        _sub,
        (2, 4, 256, 1_048_576),
        (2, 4, 256, 1_048_576),
        torch.float16,
        1,
    ),
    # 1_048_576*256*4*2*2=4_294_967_296 B=4.29 GB; cores=1: core_split=1
    # Tests reversed dim order for _find_split_dim.
    # TODO: FlexAllocator OutOfMemory
    "waterfall_descending_fp16": (
        _add,
        (1_048_576, 256, 4, 2),
        (1_048_576, 256, 4, 2),
        torch.float16,
        1,
    ),
    # -- [per_core>256MB] padding shapes --------------------------------------
    # 11*8192*17408*1*4=6_275_346_432 B=6.27 GB; cores=1: core_split=1; per_core >> 256 MB
    # TODO: FlexAllocator OutOfMemory
    "trailing_padding_fp32": (_mul, (11, 8192, 17408, 1), None, torch.float32, 1),
    # 65536*1*65536*1*2=8_589_934_592 B=8.59 GB; cores=1: core_split=1; per_core >> 256 MB
    # TODO: FlexAllocator OutOfMemory
    "interleaved_padding_fp16": (_mul, (65536, 1, 65536, 1), None, torch.float16, 1),
}

EXPECTED_FAILURES_4D = {
    "span_gt256_lt8gb_even_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_odd_bf16-cores7": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_prime_fp32-cores32": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16_c32-cores32": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    "dev_4D_large-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "bf16_add_even-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "bool_logical_or-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int32_sub_prime-cores32": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_mul_even-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "float32_add-cores1": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "seven_cores_unary-cores7": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "single_giant_prime_dim_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "waterfall_ascending_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "waterfall_descending_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "trailing_padding_fp32-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "interleaved_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
}
CHUNKING_4D_PARAM_SETS = _expand_dict(CHUNKING_4D_PARAM_SETS)

# =============================================================================
# 5-D CHUNKING PARAMETER SETS
# =============================================================================
# For (A, B, C, M, N):
#   per_core_span = ceil(M / core_split) * N * itemsize
#   total_bytes   = A * B * C * M * N * itemsize
#
# Shapes use M=8191/8192 and large N so span actually exceeds threshold.

CHUNKING_5D_PARAM_SETS = {
    # -- [SPAN] span > 256 MB -------------------------------------------------
    # 2*2*2*8192*17408*2=4.57 GB unary  [SPAN + TOTAL at cores=1].
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_even_fp16": (
        _exp_sigmoid,
        (2, 2, 2, 8192, 17408),
        None,
        torch.float16,
        1,
    ),
    # 3*3*3*8191*17409*2=7.73 GB bf16 unary  [TOTAL].
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "span_gt256_lt8gb_odd_bf16": (
        _relu_tanh,
        (3, 3, 3, 8191, 17409),
        None,
        torch.bfloat16,
        7,
    ),
    # 3*3*3*8191*8193*4=7.26 GB fp32 unary  [TOTAL].
    # span=8191*8193*4=268_442_892 bytes > 256 MB  [SPAN also fires].
    # TODO: DtException: EAR overflow detected
    "span_gt256_lt8gb_prime_fp32": (
        _neg_abs,
        (3, 3, 3, 8191, 8193),
        None,
        torch.float32,
        32,
    ),
    # -- [TOTAL] total > 8 GB -------------------------------------------------
    # 2*2*8*8192*8704*2=4.57 GB unary  [SPAN at cores=1].
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_even_fp16_c1": (
        _relu_tanh,
        (2, 2, 8, 8192, 8704),
        None,
        torch.float16,
        1,
    ),
    # 1*3*3*8191*14201*4=4.2 GB fp32 unary  [SPAN].
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_odd_fp32_c7": (_add, (1, 3, 3, 8191, 14201), None, torch.float32, 7),
    # 37*1*1*8192*14801*2=9.0 GB unary  [TOTAL].
    # TODO: FlexAllocator OutOfMemory
    "total_gt8gb_prime_fp16_c32": (
        _neg_abs,
        (37, 1, 1, 8192, 14801),
        None,
        torch.float16,
        32,
    ),
    # -- [TOTAL] dtype coverage -----------------------------------------------
    # bf16 binary: 1*4*4*8192*4096*2=1.07 GB/tensor  [TOTAL cores=1].
    # TODO: C++ compiler crashes with std::out_of_range map::at (issue #2614)
    "bf16_add_even": (
        torch.add,
        (1, 4, 4, 8192, 4096),
        (1, 4, 4, 8192, 4096),
        torch.bfloat16,
        1,
    ),
    # bool unary: 3*3*3*8191*34001*1=7.56 GB  [SPAN].
    # TODO: Missing Backend Codegen for Dtype/Op
    "bool_logical_and_prime": (
        torch.logical_and,
        (3, 3, 3, 8191, 34001),
        None,
        torch.bool,
        7,
    ),
    # int64 unary "buried core": 1*1*1_100_000_000*1*1*8=8.8 GB  [TOTAL].
    # TODO: FlexAllocator OutOfMemory
    "int64_mul_buried_core": (_mul, (1, 1, 1_100_000_000, 1, 1), None, torch.int64, 32),
    # float32 binary: 3*3*3*8191*4091*4=3.63 GB  [TOTAL at cores=7].
    # TODO: Missing Backend Codegen for Dtype/Op
    "float32_add_odd": (
        torch.add,
        (3, 3, 3, 8191, 4091),
        (3, 3, 3, 8191, 4091),
        torch.float32,
        7,
    ),
    # -- [SPAN] Shapes / ops --------------------------------------------------
    # 1*1*1*65537*4096*2=537 MB > 256 MB  [SPAN].  M=65537=2^16+1.
    # TODO: FlexAllocator OutOfMemory
    "single_giant_prime_dim_fp16": (
        _exp_sigmoid,
        (1, 1, 1, 65537, 4096),
        None,
        torch.float16,
        32,
    ),
    # 5-D unary cores=1.  SKIP: SIGABRT.
    "single_core_unary": (_exp_sigmoid, (2, 4, 4, 8192, 4096), None, torch.float16, 1),
    # 5-D binary fp32 cores=7.  SKIP: SIGABRT.
    "seven_cores_binary_fp32": (
        torch.add,
        (3, 3, 3, 8191, 4091),
        (3, 3, 3, 8191, 4091),
        torch.float32,
        7,
    ),
    # -- [TOTAL] Padding shapes -----------------------------------------------
    # 5-D trailing 1 padding: 32*8192*17408*1*1*2=9.13 GB  [TOTAL].
    # TODO: FlexAllocator OutOfMemory
    "trailing_padding_fp16": (_add, (32, 8192, 17408, 1, 1), None, torch.float16, 1),
    # Interleaved 1s: 1*65536*1*65536*1*2=8.59 GB  [TOTAL].
    # TODO: FlexAllocator OutOfMemory
    "interleaved_padding_fp16": (_add, (1, 65536, 1, 65536, 1), None, torch.float16, 1),
    # Donut-padding: 32*64*1*128*17408*2=9.13 GB  [TOTAL].
    # TODO: FlexAllocator OutOfMemory
    "donut_padding_fp16": (_add, (32, 64, 1, 128, 17408), None, torch.float16, 1),
}

EXPECTED_FAILURES_5D = {
    "span_gt256_lt8gb_even_fp16-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_odd_bf16-cores7": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "span_gt256_lt8gb_prime_fp32-cores32": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    "total_gt8gb_even_fp16_c1-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_odd_fp32_c7-cores7": ("xfail", "FlexAllocator OutOfMemory"),
    "total_gt8gb_prime_fp16_c32-cores32": (
        "xfail",
        "DtException: EAR overflow detected",
    ),
    "bf16_add_even-cores1": (
        "xfail",
        "C++ compiler crashes with std::out_of_range map::at (issue #2614)",
    ),
    "bool_logical_and_prime-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "int64_mul_buried_core-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "float32_add_odd-cores7": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    "single_giant_prime_dim_fp16-cores32": ("xfail", "FlexAllocator OutOfMemory"),
    "single_core_unary-cores1": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "seven_cores_binary_fp32-cores7": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "trailing_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "interleaved_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
    "donut_padding_fp16-cores1": ("xfail", "FlexAllocator OutOfMemory"),
}
CHUNKING_5D_PARAM_SETS = _expand_dict(CHUNKING_5D_PARAM_SETS)

# =============================================================================
# CUSTOM / EDGE-CASE PARAMETER SETS
# =============================================================================
# Tuple layout: (factory_fn, spyre_fn, cpu_ref_fn)
#
#   factory_fn  -- callable() -> tuple of Spyre tensors
#   spyre_fn    -- callable(*tensors) -> tensor  (compiled on Spyre)
#   cpu_ref_fn  -- callable(*tensors.float()) -> tensor  (CPU reference)
#                  None means same as spyre_fn
#
# Size variants:
#   small            (1,128,128)         no chunking, codegen sanity
#   large            (B,8192,34816)      total trigger (large number of elements)
#   span_large       (1,1,300_000_000)   span trigger only (single big N)
#   span_total_large (batch,1,300_000_000) both triggers
#
#   float8 relu:  alloc fp16 then cast => peak = shape*2 (fp16) + shape*1 (fp8)
#   float8 add:   2x fp16 alloc + fp8 out => peak = shape*5
#   int8 unary:   shape*1 in + shape*1 out = shape*2
#   int8 binary:  shape*1 * 3 = shape*3
#   bool binary:  shape*1 * 3
#   bf16 binary:  shape*2 * 3
#   int32 binary: shape*4 * 3
#

CUSTOM_OP_PARAM_SETS = {
    # ---- float8_e4m3fn relu (unary) -----------------------------------------
    # TODO: Spyre backend does not support type conversion fp16 -> float8_e4m3fn
    "float8_e4m3fn_relu_small": (
        lambda: (
            torch.randn(1, 128, 128, dtype=torch.float16).to(torch.float8_e4m3fn),
        ),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    # TODO: FlexAllocator OutOfMemory
    "float8_e4m3fn_relu_large": (
        lambda: (
            torch.randn(22, 8192, 34816, dtype=torch.float16).to(torch.float8_e4m3fn),
        ),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    # 1*1*300_000_000*1=286 MB  [SPAN]
    # TODO: Spyre backend does not support type conversion fp16 -> float8_e4m3fn
    "float8_e4m3fn_relu_span_large": (
        lambda: (
            torch.randn(1, 1, 300_000_000, dtype=torch.float16).to(torch.float8_e4m3fn),
        ),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    # 11*1*300_000_000*1=3.3 GB  [SPAN]
    # TODO: FlexAllocator OutOfMemory
    "float8_e4m3fn_relu_span_total_large": (
        lambda: (
            torch.randn(11, 1, 300_000_000, dtype=torch.float16).to(
                torch.float8_e4m3fn
            ),
        ),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e4m3fn),
        lambda t: torch.relu(t.float()),
    ),
    # ---- float8_e5m2 relu (unary) -------------------------------------------
    # TODO: Spyre backend does not support dtype Float8_e5m2
    "float8_e5m2_relu_small": (
        lambda: (torch.randn(1, 128, 128, dtype=torch.float16).to(torch.float8_e5m2),),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    # TODO: FlexAllocator OutOfMemory
    "float8_e5m2_relu_large": (
        lambda: (
            torch.randn(22, 8192, 34816, dtype=torch.float16).to(torch.float8_e5m2),
        ),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    # TODO: Spyre backend does not support dtype Float8_e5m2
    "float8_e5m2_relu_span_large": (
        lambda: (
            torch.randn(1, 1, 300_000_000, dtype=torch.float16).to(torch.float8_e5m2),
        ),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    # TODO: FlexAllocator OutOfMemory
    "float8_e5m2_relu_span_total_large": (
        lambda: (
            torch.randn(11, 1, 300_000_000, dtype=torch.float16).to(torch.float8_e5m2),
        ),
        lambda t: torch.relu(t.to(torch.float16)).to(torch.float8_e5m2),
        lambda t: torch.relu(t.float()),
    ),
    # ---- float8_e4m3fn binary add -------------------------------------------
    # TODO: Spyre backend does not support type conversion fp16 -> float8_e4m3fn
    "float8_e4m3fn_add_small": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.float16).to(torch.float8_e4m3fn),
            torch.randn(32, 128, 128, dtype=torch.float16).to(torch.float8_e4m3fn),
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(
            torch.float8_e4m3fn
        ),
        lambda a, b: a.float() + b.float(),
    ),
    # TODO: FlexAllocator OutOfMemory
    "float8_e4m3fn_add_large": (
        lambda: (
            torch.randn(11, 8192, 34816, dtype=torch.float16).to(torch.float8_e4m3fn),
            torch.randn(11, 8192, 34816, dtype=torch.float16).to(torch.float8_e4m3fn),
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(
            torch.float8_e4m3fn
        ),
        lambda a, b: a.float() + b.float(),
    ),
    # TODO: Spyre backend does not support type conversion fp16 -> float8_e4m3fn
    "float8_e4m3fn_add_span_large": (
        lambda: (
            torch.randn(1, 1, 300_000_000, dtype=torch.float16).to(torch.float8_e4m3fn),
            torch.randn(1, 1, 300_000_000, dtype=torch.float16).to(torch.float8_e4m3fn),
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(
            torch.float8_e4m3fn
        ),
        lambda a, b: a.float() + b.float(),
    ),
    # TODO: FlexAllocator OutOfMemory
    "float8_e4m3fn_add_span_total_large": (
        lambda: (
            torch.randn(11, 1, 300_000_000, dtype=torch.float16).to(
                torch.float8_e4m3fn
            ),
            torch.randn(11, 1, 300_000_000, dtype=torch.float16).to(
                torch.float8_e4m3fn
            ),
        ),
        lambda a, b: (a.to(torch.float16) + b.to(torch.float16)).to(
            torch.float8_e4m3fn
        ),
        lambda a, b: a.float() + b.float(),
    ),
    # ---- int8 neg_abs (unary) -----------------------------------------------
    # TODO: Spyre backend does not support dtype Int8
    "int8_neg_abs_small": (
        lambda: (torch.randint(-127, 127, (1, 128, 128), dtype=torch.int8),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    # 22*8192*34816*1=6.27 GB  [TOTAL]
    # TODO: FlexAllocator OutOfMemory
    "int8_neg_abs_large": (
        lambda: (torch.randint(-127, 127, (22, 8192, 34816), dtype=torch.int8),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    # 300 MB  [SPAN]  peak tiny
    # TODO: Spyre backend does not support dtype Int8
    "int8_neg_abs_span_large": (
        lambda: (torch.randint(-127, 127, (1, 1, 300_000_000), dtype=torch.int8),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    # 21*300M*1=6.3 GB  [SPAN].
    # TODO: FlexAllocator OutOfMemory
    "int8_neg_abs_span_total_large": (
        lambda: (torch.randint(-127, 127, (21, 1, 300_000_000), dtype=torch.int8),),
        lambda t: torch.neg(torch.abs(t)),
        lambda t: torch.neg(torch.abs(t.float())),
    ),
    # ---- int8 binary add ----------------------------------------------------
    # TODO: Spyre backend does not support dtype Int8
    "int8_add_small": (
        lambda: (
            torch.randint(-127, 127, (32, 128, 128), dtype=torch.int8),
            torch.randint(-127, 127, (32, 128, 128), dtype=torch.int8),
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    # TODO: FlexAllocator OutOfMemory
    "int8_add_large": (
        lambda: (
            torch.randint(-127, 127, (22, 8192, 34816), dtype=torch.int8),
            torch.randint(-127, 127, (22, 8192, 34816), dtype=torch.int8),
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    # TODO: Spyre backend does not support dtype Int8
    "int8_add_span_large": (
        lambda: (
            torch.randint(-127, 127, (1, 1, 300_000_000), dtype=torch.int8),
            torch.randint(-127, 127, (1, 1, 300_000_000), dtype=torch.int8),
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    # TODO: FlexAllocator OutOfMemory
    "int8_add_span_total_large": (
        lambda: (
            torch.randint(-127, 127, (21, 1, 300_000_000), dtype=torch.int8),
            torch.randint(-127, 127, (21, 1, 300_000_000), dtype=torch.int8),
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    # ---- non-contiguous slice (stride != 1) ---------------------------------
    # Allocates FULL backing buffer before slicing.
    # SKIP: non-contiguous tensor crashes spyre__copy_from (arg.to(device) requires contiguous).
    "non_contiguous_slice_small": (
        lambda: (
            torch.randn(32, 128, 256, dtype=torch.float16)[:, :, ::2],
            torch.randn(32, 128, 256, dtype=torch.float16)[:, :, ::2],
        ),
        _add,
        None,
    ),
    # TODO: FlexAllocator OutOfMemory
    "non_contiguous_slice_large": (
        lambda: (
            torch.randn(11, 8192, 34816, dtype=torch.float16)[:, :, ::2],
            torch.randn(11, 8192, 34816, dtype=torch.float16)[:, :, ::2],
        ),
        _add,
        None,
    ),
    # SKIP: non-contiguous tensor crashes spyre__copy_from (arg.to(device) requires contiguous).
    "non_contiguous_slice_span_large": (
        lambda: (
            torch.randn(1, 1, 300_000_000, dtype=torch.float16)[:, :, ::2],
            torch.randn(1, 1, 300_000_000, dtype=torch.float16)[:, :, ::2],
        ),
        _add,
        None,
    ),
    # TODO: FlexAllocator OutOfMemory
    "non_contiguous_slice_span_total_large": (
        lambda: (
            torch.randn(12, 1, 300_000_000, dtype=torch.float16)[:, :, ::2],
            torch.randn(12, 1, 300_000_000, dtype=torch.float16)[:, :, ::2],
        ),
        _add,
        None,
    ),
    # ---- expand / zero-stride broadcast -------------------------------------
    # (1,1,1) tensor expanded to large shape; chunking must not slice the 0-stride dim.
    # SKIP: zero-stride expanded tensor crashes spyre__copy_from (ConvertDataNonPcaLoop5).
    "expand_zero_stride_small": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16).expand(32, 128, 128),
            torch.randn(32, 128, 128, dtype=torch.float16),
        ),
        _add,
        None,
    ),
    # TODO: FlexAllocator OutOfMemory
    "expand_zero_stride_large": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16).expand(32, 8192, 17408),
            torch.randn(32, 8192, 17408, dtype=torch.float16),
        ),
        _add,
        None,
    ),
    # SKIP: zero-stride expanded tensor segfaults spyre__copy_from (ConvertDataNonPcaLoop5).
    "expand_zero_stride_span_large": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16).expand(1, 1, 150_000_000),
            torch.randn(1, 1, 150_000_000, dtype=torch.float16),
        ),
        _add,
        None,
    ),
    # SKIP: zero-stride expanded tensor segfaults spyre__copy_from (ConvertDataNonPcaLoop5).
    "expand_zero_stride_span_total_large": (
        lambda: (
            torch.randn(1, 1, 1, dtype=torch.float16).expand(29, 1, 150_000_000),
            torch.randn(29, 1, 150_000_000, dtype=torch.float16),
        ),
        _add,
        None,
    ),
    # ---- in-place mutation (a.add_(b)) --------------------------------------
    # Chunking pass must correctly track the in-place alias.
    # TODO: Missing Backend Codegen for Dtype/Op
    "in_place_mutation_conflict_small": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.float16),
            torch.randn(1, 1, 1, dtype=torch.float16),
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    # TODO: FlexAllocator OutOfMemory
    "in_place_mutation_conflict_large": (
        lambda: (
            torch.randn(32, 8192, 17408, dtype=torch.float16),
            torch.randn(1, 1, 1, dtype=torch.float16),
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    # TODO: Missing Backend Codegen for Dtype/Op
    "in_place_mutation_conflict_span_large": (
        lambda: (
            torch.randn(1, 1, 150_000_000, dtype=torch.float16),
            torch.randn(1, 1, 1, dtype=torch.float16),
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    # TODO: FlexAllocator OutOfMemory
    "in_place_mutation_conflict_span_total_large": (
        lambda: (
            torch.randn(29, 1, 150_000_000, dtype=torch.float16),
            torch.randn(1, 1, 1, dtype=torch.float16),
        ),
        lambda a, b: a.add_(b),
        lambda a, b: a.clone().add_(b),
    ),
    # ---- mixed precision fp16 + fp32 ----------------------------------------
    # Small shape; tests that mixed-dtype graphs compile without chunking.
    # TODO: Missing Backend Codegen for Dtype/Op
    "mixed_precision_fp16_fp32_add": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.float16),
            torch.randn(32, 128, 128, dtype=torch.float32),
        ),
        lambda a, b: a + b,
        lambda a, b: a.float() + b.float(),
    ),
    # ---- bool logical_and ---------------------------------------------------
    # TODO: Missing Backend Codegen for Dtype/Op
    "bool_logical_and_small": (
        lambda: (
            torch.randint(0, 2, (32, 128, 128), dtype=torch.bool),
            torch.randint(0, 2, (32, 128, 128), dtype=torch.bool),
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    # TODO: DtException: EAR overflow detected
    "bool_logical_and_large": (
        lambda: (
            torch.randint(0, 2, (22, 8192, 34816), dtype=torch.bool),
            torch.randint(0, 2, (22, 8192, 34816), dtype=torch.bool),
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    # TODO: Missing Backend Codegen for Dtype/Op
    "bool_logical_and_span_large": (
        lambda: (
            torch.randint(0, 2, (1, 1, 300_000_000), dtype=torch.bool),
            torch.randint(0, 2, (1, 1, 300_000_000), dtype=torch.bool),
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    "bool_logical_and_span_total_large": (
        lambda: (
            torch.randint(0, 2, (21, 1, 300_000_000), dtype=torch.bool),
            torch.randint(0, 2, (21, 1, 300_000_000), dtype=torch.bool),
        ),
        lambda a, b: torch.logical_and(a, b),
        None,
    ),
    # ---- bfloat16 add -------------------------------------------------------
    "bf16_add_small": (
        lambda: (
            torch.randn(32, 128, 128, dtype=torch.bfloat16),
            torch.randn(32, 128, 128, dtype=torch.bfloat16),
        ),
        lambda a, b: a + b,
        None,
    ),
    "bf16_add_large": (
        lambda: (
            torch.randn(11, 8192, 34816, dtype=torch.bfloat16),
            torch.randn(11, 8192, 34816, dtype=torch.bfloat16),
        ),
        lambda a, b: a + b,
        None,
    ),
    "bf16_add_span_large": (
        lambda: (
            torch.randn(1, 1, 300_000_000, dtype=torch.bfloat16),
            torch.randn(1, 1, 300_000_000, dtype=torch.bfloat16),
        ),
        lambda a, b: a + b,
        None,
    ),
    "bf16_add_span_total_large": (
        lambda: (
            torch.randn(11, 1, 300_000_000, dtype=torch.bfloat16),
            torch.randn(11, 1, 300_000_000, dtype=torch.bfloat16),
        ),
        lambda a, b: a + b,
        None,
    ),
    # ---- int32 add ----------------------------------------------------------
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_add_small": (
        lambda: (
            torch.randint(-100, 100, (32, 128, 128), dtype=torch.int32),
            torch.randint(-100, 100, (32, 128, 128), dtype=torch.int32),
        ),
        lambda a, b: a + b,
        None,
    ),
    # TODO: FlexAllocator OutOfMemory
    "int32_add_large": (
        lambda: (
            torch.randint(-100, 100, (5, 8192, 34816), dtype=torch.int32),
            torch.randint(-100, 100, (5, 8192, 34816), dtype=torch.int32),
        ),
        lambda a, b: a + b,
        None,
    ),
    # TODO: Missing Backend Codegen for Dtype/Op
    "int32_add_span_large": (
        lambda: (
            torch.randint(-100, 100, (1, 1, 300_000_000), dtype=torch.int32),
            torch.randint(-100, 100, (1, 1, 300_000_000), dtype=torch.int32),
        ),
        lambda a, b: a + b,
        None,
    ),
    # TODO: FlexAllocator OutOfMemory
    "int32_add_span_total_large": (
        lambda: (
            torch.randint(-100, 100, (5, 1, 300_000_000), dtype=torch.int32),
            torch.randint(-100, 100, (5, 1, 300_000_000), dtype=torch.int32),
        ),
        lambda a, b: a + b,
        None,
    ),
}

EXPECTED_FAILURES_CUSTOM = {
    # float8 -- type conversion not supported on Spyre (bidirectional fallback to CPU)
    "float8_e4m3fn_relu_small": (
        "xfail",
        "Spyre backend does not support float8_e4m3fn: both float8_e4m3fn<->float16 conversions fall back to CPU",
    ),
    "float8_e4m3fn_relu_large": ("xfail", "DtException: EAR overflow detected"),
    "float8_e4m3fn_relu_span_large": (
        "xfail",
        "Spyre backend does not support float8_e4m3fn: both float8_e4m3fn<->float16 conversions fall back to CPU",
    ),
    "float8_e4m3fn_relu_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e5m2_relu_small": (
        "xfail",
        "Spyre backend does not support dtype Float8_e5m2",
    ),
    "float8_e5m2_relu_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e5m2_relu_span_large": (
        "xfail",
        "Spyre backend does not support dtype Float8_e5m2",
    ),
    "float8_e5m2_relu_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    "float8_e4m3fn_add_small": (
        "xfail",
        "Spyre backend does not support float8_e4m3fn: both float8_e4m3fn<->float16 conversions fall back to CPU",
    ),
    "float8_e4m3fn_add_large": ("xfail", "DtException: EAR overflow detected"),
    "float8_e4m3fn_add_span_large": (
        "xfail",
        "Spyre backend does not support float8_e4m3fn: both float8_e4m3fn<->float16 conversions fall back to CPU",
    ),
    "float8_e4m3fn_add_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    # int8 -- dtype not supported (codegen failure for SENINT8)
    "int8_neg_abs_small": (
        "xfail",
        "Spyre backend does not support dtype Int8 (Error in codegen for ComputedBuffer SENINT8)",
    ),
    "int8_neg_abs_large": (
        "xfail",
        "Error in codegen for ComputedBuffer (SENINT8 dtype not supported)",
    ),
    "int8_neg_abs_span_large": (
        "xfail",
        "Spyre backend does not support dtype Int8 (Error in codegen for ComputedBuffer SENINT8)",
    ),
    "int8_neg_abs_span_total_large": (
        "xfail",
        "Error in codegen for ComputedBuffer (SENINT8 dtype not supported)",
    ),
    "int8_add_small": (
        "xfail",
        "Spyre backend does not support dtype Int8 (Error in codegen for ComputedBuffer SENINT8)",
    ),
    "int8_add_large": (
        "xfail",
        "Error in codegen for ComputedBuffer (SENINT8 dtype not supported)",
    ),
    "int8_add_span_large": (
        "xfail",
        "Spyre backend does not support dtype Int8 (Error in codegen for ComputedBuffer SENINT8)",
    ),
    "int8_add_span_total_large": (
        "xfail",
        "Error in codegen for ComputedBuffer (SENINT8 dtype not supported)",
    ),
    # non-contiguous -- crashes spyre__copy_from; skip small/span to avoid crash
    "non_contiguous_slice_small": (
        "skip",
        "Non-contiguous tensor crashes spyre__copy_from (arg.to(device) requires contiguous)",
    ),
    "non_contiguous_slice_span_large": (
        "skip",
        "Non-contiguous tensor crashes spyre__copy_from (arg.to(device) requires contiguous)",
    ),
    "non_contiguous_slice_large": ("xfail", "FlexAllocator OutOfMemory"),
    "non_contiguous_slice_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
    # expand -- zero-stride crashes/segfaults spyre__copy_from
    "expand_zero_stride_small": (
        "skip",
        "Zero-stride expanded tensor crashes spyre__copy_from (ConvertDataNonPcaLoop5 in libutil.so)",
    ),
    "expand_zero_stride_span_large": (
        "skip",
        "Zero-stride expanded tensor segfaults spyre__copy_from (ConvertDataNonPcaLoop5 in libutil.so)",
    ),
    "expand_zero_stride_span_total_large": (
        "skip",
        "Zero-stride expanded tensor segfaults spyre__copy_from (ConvertDataNonPcaLoop5 in libutil.so)",
    ),
    "expand_zero_stride_large": ("xfail", "FlexAllocator OutOfMemory"),
    # in-place mutation
    "in_place_mutation_conflict_small": (
        "xfail",
        "Missing Backend Codegen for Dtype/Op",
    ),
    "in_place_mutation_conflict_large": ("xfail", "FlexAllocator OutOfMemory"),
    "in_place_mutation_conflict_span_large": (
        "xfail",
        "Missing Backend Codegen for Dtype/Op",
    ),
    "in_place_mutation_conflict_span_total_large": (
        "xfail",
        "FlexAllocator OutOfMemory",
    ),
    # mixed precision
    "mixed_precision_fp16_fp32_add": ("xfail", "Missing Backend Codegen for Dtype/Op"),
    # bool -- large xfails with EAR overflow; span variants crash bundler with SIGABRT
    "bool_logical_and_large": ("xfail", "DtException: EAR overflow detected"),
    "bool_logical_and_span_large": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    "bool_logical_and_span_total_large": (
        "skip",
        "Crashes with SIGABRT in dxp_standalone bundler",
    ),
    # bf16 -- all four variants PASS (no xfail entries)
    # int32
    "int32_add_small": (
        "xfail",
        "Missing Backend Codegen for Dtype/Op (int32 not supported)",
    ),
    "int32_add_large": ("xfail", "FlexAllocator OutOfMemory"),
    "int32_add_span_large": (
        "xfail",
        "Missing Backend Codegen for Dtype/Op (int32 not supported)",
    ),
    "int32_add_span_total_large": ("xfail", "FlexAllocator OutOfMemory"),
}

# =============================================================================
# TEST CLASS
# =============================================================================

torch.manual_seed(0xAFFE)


class TestOps:
    """Pytest class for all chunk_large_tensors correctness tests."""

    def setup_method(self):
        """Reset RNG before every test."""
        torch.manual_seed(0xAFFE)

    def teardown_method(self):
        """Force GC + dynamo cache clear after every test to free device memory."""
        gc.collect()
        torch._dynamo.reset()

    @pytest.fixture(autouse=True)
    def env_chunking(self):
        os.environ["CHUNK_LARGE_TENSORS"] = "1"
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        yield
        os.environ.pop("CHUNK_LARGE_TENSORS", None)
        os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
        os.environ.pop("SENCORES", None)

    def compare_with_cpu(self, *args, **kwargs):
        """Thin wrapper matching the pattern in test_inductor_ops.py.

        Passes CPU tensors + fn directly to utils_inductor.compare_with_cpu.
        compare_with_cpu internally:
          1. Calls fn(*cpu_args) for the CPU reference result.
          2. Moves args to the spyre device, compiles fn, and runs it.
          3. Asserts the two results are close within atol/rtol.
        """
        kwargs.setdefault("run_eager", False)
        kwargs.setdefault("cpu_compile", True)
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    def run_chunking_test(self, fn, shape_x, shape_y, dtype, cores):
        """Create CPU tensors, set SENCORES, call compare_with_cpu.

        Tensors are created on CPU here (not on spyre).  compare_with_cpu
        then moves them to the spyre device and compiles fn — exactly the same
        pattern used throughout test_inductor_ops.py.

        SENCORES is set here because it varies per parametrized test case.
        The env_chunking fixture tears it down after each test.
        """
        if isinstance(dtype, tuple):
            dtype_x, dtype_y = dtype
        else:
            dtype_x = dtype_y = dtype

        def _make_cpu(shape, dt):
            """Build a CPU tensor with appropriate randomisation for dtype."""
            if dt == torch.bool:
                return torch.randint(0, 2, shape, dtype=dt)
            elif dt in (torch.int8, torch.int32, torch.int64):
                return torch.randint(-100, 100, shape, dtype=dt)
            else:
                return torch.randn(*shape, dtype=dt)

        x = _make_cpu(shape_x, dtype_x)
        y = _make_cpu(shape_y, dtype_y) if shape_y is not None else None

        if cores is not None:
            os.environ["SENCORES"] = str(cores)

        if y is not None:
            self.compare_with_cpu(fn, x, y, atol=_ATOL)
        else:
            self.compare_with_cpu(fn, x, atol=_ATOL)

    # -------------------------------------------------------------------------
    # Parametrised test methods
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_1D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_1D),
    )
    def test_chunking_1d(self, fn, shape_x, shape_y, dtype, cores):
        """1-D: span, total, boundary, and dtype coverage."""
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_2D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_2D),
    )
    def test_chunking_2d(self, fn, shape_x, shape_y, dtype, cores):
        """2-D: span, total, dim extremes, broadcast, mixed precision."""
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_3D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_3D),
    )
    def test_chunking_3d(self, fn, shape_x, shape_y, dtype, cores):
        """3-D: full coverage including Groups 1-14 of TestPointwiseChunking."""
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_4D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_4D),
    )
    def test_chunking_4d(self, fn, shape_x, shape_y, dtype, cores):
        """4-D: span, total, waterfall dims, padding layouts."""
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "fn, shape_x, shape_y, dtype, cores",
        _make_params(CHUNKING_5D_PARAM_SETS, expect_fail=EXPECTED_FAILURES_5D),
    )
    def test_chunking_5d(self, fn, shape_x, shape_y, dtype, cores):
        """5-D: span, total, buried dims, interleaved padding."""
        self.run_chunking_test(fn, shape_x, shape_y, dtype, cores)

    @pytest.mark.parametrize(
        "factory_fn, fn, ref_fn",
        _make_custom_params(CUSTOM_OP_PARAM_SETS, expect_fail=EXPECTED_FAILURES_CUSTOM),
    )
    def test_custom_edge_cases(self, factory_fn, fn, ref_fn):
        """Custom dtypes and edge cases: float8, int8, non-contiguous, expand, in-place.

        factory_fn() returns CPU tensors.  If ref_fn is provided, it is used as
        the CPU reference (fn is the spyre callable after device move + compile).
        compare_with_cpu handles moving inputs to spyre and compiling fn.
        """
        cpu_inputs = factory_fn()
        if ref_fn is not None:
            # Wrap so compare_with_cpu uses ref_fn on CPU and fn on spyre.
            # Device detection relies on compare_with_cpu calling the wrapped fn
            # first with CPU tensors (ref path) then with spyre tensors (spyre path).
            def _wrapped(*args):
                if any(
                    isinstance(a, torch.Tensor) and a.device.type == "spyre"
                    for a in args
                ):
                    return fn(*args)
                return ref_fn(
                    *tuple(
                        a.float() if isinstance(a, torch.Tensor) else a for a in args
                    )
                )

            self.compare_with_cpu(_wrapped, *cpu_inputs, atol=_ATOL)
        else:
            self.compare_with_cpu(fn, *cpu_inputs, atol=_ATOL)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
