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
from unittest.mock import patch

import pytest
import torch
import torch._dynamo as dynamo

import torch_spyre._inductor.passes as _passes
from torch._inductor.virtualized import V
from torch_spyre._C import get_spyre_dma_sizes, get_spyre_dma_strides
from utils_inductor import _compile_and_run

DEVICE = torch.device("spyre")


# -------- Helpers --------


def _capture_plans(fn, args):
    """Run fn on DEVICE and capture restickify_plan."""
    captured = {}
    orig_finalize = _passes.finalize_layouts

    def capturing_finalize(graph):
        orig_finalize(graph)
        captured["restickify_plan"] = dict(V.graph.restickify_plan)

    with patch.object(_passes, "finalize_layouts", capturing_finalize):
        result = _compile_and_run(fn, args, DEVICE)

    return result, captured


def _assert_no_restickify(restickify_plan):
    assert not restickify_plan, (
        f"Expected no restickify, but got plan: {restickify_plan}"
    )


def test_compact_3d_layouts():
    """3D: verify sparse→dense device_size and stride_map transformation."""

    captured = {}
    orig_finalize = _passes.finalize_layouts

    def capturing_finalize(graph):
        orig_finalize(graph)
        for name, buf in graph.name_to_buffer.items():
            try:
                layout = buf.get_layout()
                if hasattr(layout, "device_layout"):
                    captured.setdefault("layouts", {})[name] = layout.device_layout
            except Exception:
                pass

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)

    with patch.object(_passes, "finalize_layouts", capturing_finalize):
        _compile_and_run(fn, [x_spyre], DEVICE)

    layouts = captured["layouts"]
    # buf0: sum output — sparse, device_rank 5
    # buf1: compact output — dense, device_rank 4 (one dim stripped)
    buf0_stl = layouts["buf0"]
    buf1_stl = layouts["buf1"]

    # Sparse: two inner size-1 dims (tile count + outer-stick) both synthetic
    assert list(buf0_stl.device_size) == [48, 1, 1, 4, 64], (
        f"Unexpected sparse device_size: {list(buf0_stl.device_size)}"
    )
    assert list(buf0_stl.stride_map) == [1, -1, -1, 48, -1], (
        f"Unexpected sparse stride_map: {list(buf0_stl.stride_map)}"
    )

    # Dense: one dim fewer, stick dim is now real (stride_map[-1] != -1 except
    # stick synthetic marker; verify it's the standard dense (4,48,1) layout)
    assert list(buf1_stl.device_size) == [48, 1, 4, 64], (
        f"Unexpected dense device_size: {list(buf1_stl.device_size)}"
    )
    assert list(buf1_stl.stride_map) == [1, -1, 48, -1], (
        f"Unexpected dense stride_map: {list(buf1_stl.stride_map)}"
    )


# -------- Tests: compact layout --------
#
# Tests that verify that the result of compact has the default
# STL for a tensor of a given torch shape

DTYPES = [torch.float16]
DTYPE_IDS = ["fp16"]

_TOLERANCES = {
    torch.float16: {"atol": 0.1, "rtol": 0.1},
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.int32: {"atol": 0, "rtol": 0},
}


@torch.compile
def sum_and_compact(x, dim, keepdim):
    reduced = x.sum(dim, keepdim)
    return torch.ops.spyre.compact(reduced)


def _make_cpu_input(shape, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    if dtype in (torch.float16, torch.float32):
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype == torch.int32:
        return torch.randint(-100, 100, shape, dtype=dtype, generator=gen)
    raise ValueError(f"Unsupported dtype: {dtype}")


def _tensor_layout_snapshot(t):
    """All comparable tensor/device-layout attributes, excluding pointers."""
    return {
        "shape": tuple(t.shape),
        "stride": t.stride(),
        "storage_offset": t.storage_offset(),
        "numel": t.numel(),
        "dtype": t.dtype,
        "element_size": t.element_size(),
        "storage_nbytes": t.untyped_storage().nbytes(),
        "contiguous": t.is_contiguous(),
        "device": t.device,
        "dev_layout": t.device_tensor_layout(),
        "dma_sizes": get_spyre_dma_sizes(t),
        "dma_strides": get_spyre_dma_strides(t),
    }


def _assert_layout_matches(actual, expected):
    actual_snapshot = _tensor_layout_snapshot(actual)
    expected_snapshot = _tensor_layout_snapshot(expected)
    for key in actual_snapshot:
        dtype_size = torch.tensor([1], dtype=actual.dtype).element_size()
        stick_size = int(128 / dtype_size)
        if key == "storage_nbytes":
            assert actual_snapshot[key] in (
                expected_snapshot[key],
                stick_size * expected_snapshot[key],
            )
        elif key == "dev_layout":
            actual_size = actual_snapshot[key].device_size
            expected_size = actual_snapshot[key].device_size
            assert actual_size in (expected_size, [stick_size] + expected_size)
        else:
            assert actual_snapshot[key] == expected_snapshot[key], (
                f"{key} mismatch: actual={actual_snapshot[key]!r} "
                f"expected={expected_snapshot[key]!r}"
            )


# (shape, dim, keepdim) cases: 1D/2D/3D inputs, reducing the last dim or
# another dim, with and without keepdim.
REDUCTION_CASES = {
    "1d_dimneg1_keepdimF": ((200,), -1, False),
    "1d_dimneg1_keepdimT": ((200,), -1, True),
    "2d_dimneg1_keepdimF": ((256, 256), -1, False),
    "2d_dimneg1_keepdimT": ((256, 256), -1, True),
    "2d_dim0_keepdimF": ((256, 256), 0, False),
    "2d_dim0_keepdimT": ((256, 256), 0, True),
    "3d_dimneg1_keepdimF": ((2, 4, 256), -1, False),
    "3d_dimneg1_keepdimT": ((2, 4, 256), -1, True),
    "3d_dim0_keepdimF": ((2, 4, 256), 0, False),
    "3d_dim0_keepdimT": ((2, 4, 256), 0, True),
    "3d_dim1_keepdimF": ((2, 4, 256), 1, False),
    "3d_dim1_keepdimT": ((2, 4, 256), 1, True),
}


@pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
@pytest.mark.parametrize(
    "dtype",
    DTYPES,
    ids=DTYPE_IDS,
)
@pytest.mark.parametrize(
    "case_name,shape,dim,keepdim",
    [(name, *params) for name, params in REDUCTION_CASES.items()],
    ids=list(REDUCTION_CASES.keys()),
)
def test_sum_and_compact(case_name, shape, dim, keepdim, dtype):
    x_cpu = _make_cpu_input(shape, dtype, seed=0xAFFE)

    actual = sum_and_compact(x_cpu.to(DEVICE), dim, keepdim)
    expected = x_cpu.sum(dim, keepdim).to(DEVICE)

    _assert_layout_matches(actual, expected)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), equal_nan=True, **_TOLERANCES[dtype]
    )


# -------- Tests: ops in compacted tensors --------
#
# Tests that verify that operation on "compacted" tensors run
# without crashing and produce correct results
filename = Path(__file__).stem

BOTH = [False, True]

_TOLERANCES = {
    torch.float16: {"atol": 0.1, "rtol": 0.1},
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.int32: {"atol": 0, "rtol": 0},
}


def _ones(*args):
    return torch.ones(args)


# Allow in graph for debugging purposes
@torch.compiler.allow_in_graph
def maybe_compact(x: torch.Tensor, compact: bool):
    if compact and x.device.type == "spyre":
        return torch.ops.spyre.compact(x)
    return x


PRINT_FOR_REPRO = False


def run_binary_op(
    func, device, dtype, dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b
):
    if pre_op_keep_dim:
        # do this before sending to device to create the initial tensor layouts correctly
        b = b.unsqueeze(dim)
    a = a.to(device, dtype)
    b = b.to(device, dtype)

    if device == "cpu" and PRINT_FOR_REPRO:
        explanation = dynamo.explain(func)(
            dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b
        )
        for i, gm in enumerate(explanation.graphs):
            print(f"\nRepro {i}:\n")
            print("import torch")
            print("device='spyre'")
            print(f"a = torch.ones({tuple(a.shape)}, device=device, {dtype=})")
            print(f"b = torch.ones({tuple(b.shape)}, device=device, {dtype=})")
            print(
                f"{filename}_maybe_compact =  lambda x, _: torch.ops.spyre.compact(x)"
            )
            print(gm.code)
            print("compiled = torch.compile(forward)")
            print("print(compiled(None, a,b))")

    return func(dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b).cpu()


def run_test(do_run, compact):
    # run on CPU first to be sure that we didn't mess up the pytorch logic
    cpu_result = do_run("cpu")

    # Now that CPU hasn't failed, set xfail if we're not compacting
    if not compact:
        pytest.xfail("Operation may fail without compacting")

    spyre_result = do_run("spyre")

    torch.testing.assert_close(
        cpu_result, spyre_result, equal_nan=True, **_TOLERANCES[cpu_result.dtype]
    )


@torch.compile
def mul_on_reduced(dim, compact, reduce_keep_dim, pre_mul_keep_dim, a, b):
    reduced = a.sum(dim, keepdim=reduce_keep_dim)
    if reduce_keep_dim and not pre_mul_keep_dim:
        reduced.squeeze_(dim)
    elif not reduce_keep_dim and pre_mul_keep_dim:
        reduced.unsqueeze_(dim)
    reduced = maybe_compact(reduced, compact)
    return reduced * b


POINTWISE_CASES = {
    "scalar": (_ones(120), _ones(1), -1),
    "1d_stick": (_ones(128, 128), _ones(128), -1),
    "1d_nonstick": (_ones(128, 128), _ones(128), -2),
    "2d_stick": (_ones(2, 128, 128), _ones(2, 128), -1),
    "2d_nonstick": (_ones(2, 128, 128), _ones(2, 128), -2),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("compact", BOTH, ids=["no_compact", "compact"])
@pytest.mark.parametrize("reduce_keep_dim", BOTH)
@pytest.mark.parametrize("pre_mul_keep_dim", BOTH)
@pytest.mark.parametrize(
    "a,b,dim",
    list(POINTWISE_CASES.values()),
    ids=list(POINTWISE_CASES.keys()),
)
def test_pointwise_binary_op(
    dtype: torch.dtype,
    compact: bool,
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
            compact,
            reduce_keep_dim,
            pre_mul_keep_dim,
            a,
            b,
        )

    run_test(do_run, compact)


@torch.compile
def matmul_on_reduced(dim, compact, reduce_keep_dim, pre_mul_keep_dim, a, b):
    reduced = a.sum(dim, keepdim=reduce_keep_dim)
    if reduce_keep_dim and not pre_mul_keep_dim:
        reduced.squeeze_(dim)
    elif not reduce_keep_dim and pre_mul_keep_dim:
        reduced.unsqueeze_(dim)
    reduced = maybe_compact(reduced, compact)
    return reduced @ b


MATMUL_CASES = {
    "1d_stick": (_ones(128, 128), _ones(128), -1),
    "1d_nonstick": (_ones(128, 128), _ones(128), -2),
    "2d_stick": (_ones(2, 128, 128), _ones(128, 128), -1),
    "2d_nonstick": (_ones(2, 128, 128), _ones(128, 128), -2),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("compact", BOTH, ids=["no_compact", "compact"])
@pytest.mark.parametrize(
    "a,b,dim",
    list(MATMUL_CASES.values()),
    ids=list(MATMUL_CASES.keys()),
)
def test_matmul_op(
    dtype: torch.dtype, compact: bool, dim: int, a: torch.tensor, b: torch.tensor
):
    def do_run(device):
        return run_binary_op(
            matmul_on_reduced, device, dtype, dim, compact, False, False, a, b
        )

    run_test(do_run, compact)
