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

# Tests for torch.ops.spyre.compact — sparse-to-dense layout op.


from pathlib import Path

import pytest
import torch
import torch._dynamo as dynamo


# -------- Tests: ops in compacted tensors --------
#
# Tests that verify that operation on "compacted" tensors run
# without crashing and produce correct results
filename = Path(__file__).stem

DTYPES = [torch.float16]
DTYPE_IDS = ["fp16"]
BOTH = [False, True]

_TOLERANCES = {
    torch.float16: {"atol": 0.1, "rtol": 0.1},
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.int32: {"atol": 0, "rtol": 0},
}


def _ones(*args):
    return torch.ones(args)


PRINT_FOR_REPRO = False


def run_binary_op(func, device, dtype, dim, reduce_keep_dim, pre_op_keep_dim, a, b):
    if pre_op_keep_dim:
        # do this before sending to device to create the initial tensor layouts correctly
        b = b.unsqueeze(dim)
    a = a.to(device, dtype)
    b = b.to(device, dtype)

    if device == "cpu" and PRINT_FOR_REPRO:
        explanation = dynamo.explain(func)(dim, reduce_keep_dim, pre_op_keep_dim, a, b)
        for i, gm in enumerate(explanation.graphs):
            print(f"\nRepro {i}:\n")
            print("import torch")
            print("device='spyre'")
            print(f"a = torch.ones({tuple(a.shape)}, device=device, {dtype=})")
            print(f"b = torch.ones({tuple(b.shape)}, device=device, {dtype=})")
            print(gm.code)
            print("compiled = torch.compile(forward)")
            print("print(compiled(None, a,b))")

    return func(dim, reduce_keep_dim, pre_op_keep_dim, a, b).cpu()


def run_test(do_run):
    # run on CPU first to be sure that we didn't mess up the pytorch logic
    cpu_result = do_run("cpu")
    spyre_result = do_run("spyre")

    torch.testing.assert_close(
        cpu_result, spyre_result, equal_nan=True, **_TOLERANCES[cpu_result.dtype]
    )


@torch.compile
def mul_on_reduced(dim, reduce_keep_dim, pre_mul_keep_dim, a, b):
    reduced = a.sum(dim, keepdim=reduce_keep_dim)
    if reduce_keep_dim and not pre_mul_keep_dim:
        reduced.squeeze_(dim)
    elif not reduce_keep_dim and pre_mul_keep_dim:
        reduced.unsqueeze_(dim)
    return reduced * b


POINTWISE_CASES = {
    "scalar": (_ones(120), _ones(1), -1),
    "1d_stick": (_ones(128, 128), _ones(128), -1),
    "1d_nonstick": (_ones(128, 128), _ones(128), -2),
    "2d_stick": (_ones(2, 128, 128), _ones(2, 128), -1),
    "2d_nonstick": (_ones(2, 128, 128), _ones(2, 128), -2),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_keep_dim", BOTH)
@pytest.mark.parametrize("pre_mul_keep_dim", BOTH)
@pytest.mark.parametrize(
    "a,b,dim",
    list(POINTWISE_CASES.values()),
    ids=list(POINTWISE_CASES.keys()),
)
def test_pointwise_binary_op(
    dtype: torch.dtype,
    dim: int,
    reduce_keep_dim: bool,
    pre_mul_keep_dim: bool,
    a: torch.tensor,
    b: torch.tensor,
):
    def do_run(device):
        return run_binary_op(
            mul_on_reduced,
            device,
            dtype,
            dim,
            reduce_keep_dim,
            pre_mul_keep_dim,
            a,
            b,
        )

    run_test(do_run)


@torch.compile
def matmul_on_reduced(dim, reduce_keep_dim, pre_mul_keep_dim, a, b):
    reduced = a.sum(dim, keepdim=reduce_keep_dim)
    if reduce_keep_dim and not pre_mul_keep_dim:
        reduced.squeeze_(dim)
    elif not reduce_keep_dim and pre_mul_keep_dim:
        reduced.unsqueeze_(dim)
    return reduced @ b


MATMUL_CASES = {
    "1d_stick": (_ones(128, 128), _ones(128), -1),
    "1d_nonstick": (_ones(128, 128), _ones(128), -2),
    "2d_stick": (_ones(2, 128, 128), _ones(128, 128), -1),
    "2d_nonstick": (_ones(2, 128, 128), _ones(128, 128), -2),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    "a,b,dim",
    list(MATMUL_CASES.values()),
    ids=list(MATMUL_CASES.keys()),
)
def test_matmul_op(dtype: torch.dtype, dim: int, a: torch.tensor, b: torch.tensor):
    def do_run(device):
        return run_binary_op(matmul_on_reduced, device, dtype, dim, False, False, a, b)

    run_test(do_run)
