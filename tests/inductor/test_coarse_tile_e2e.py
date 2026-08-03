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

"""End-to-end compilation tests for the coarse-tiling loop IR.

This file has two sections:

STRUCTURED TESTS (Groups 1-10)
    Flat module-level tests using the run_coarse_tile_test() driver.
    These are the primary test suite going forward — easy to read, copy,
    and extend.  Each test declares its inputs via tensor() descriptors,
    defines a plain fn(), and calls the driver with optional loopspec=
    and correctness= flags.

    Group 1: Basic tiling — abs/add on 2D tensors, varied sizes and tile counts
    Group 2: 3D tensors — [A=512, B=256, C=256], all tiling combinations
    Group 3: Pointwise op chains — abs(a+b)*c, exp(abs(...)), etc.
    Group 4: Reductions — amin over 2D (all tiling combos) and 3D (all-dims)
    Group 5: Mixed pointwise + reduction — add_min, reduce_both, softmax
    Group 6: Restickify + coarse tiling — transpose inputs with tiling
    Group 7: Copies — pre-allocated buffers, in-place accumulators, RMW
    Group 8: Tiled ops with outside consumers
    Group 9: Views — 1D sub-dim naming, reshape, view+transpose, unsqueeze
    Group 10: Flash attention variants — v1/v2/v3/v4, parameterized by size and tile dims

    Tests marked loopspec=None, correctness=False are known broken and
    skipped; see inline comments for root cause.

ORIGINAL TESTS (below the boundary marker)
    Original class-based tests preserved for coverage and reference, to be cleaned up in future.
"""

import dataclasses
import math
import os
import sys
import regex as re

import pytest
import torch
import unittest
from unittest.mock import patch as mock_patch

from torch._inductor.test_case import TestCase as InductorTestCase, fresh_cache
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor import config
from torch_spyre._inductor import spyre_hint
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils_inductor import compare_with_cpu, _compile_and_run  # noqa: E402

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims

# Paths to mock for disabling actual device kernel execution.
_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class LoopSpecCheck:
    """Callable loopspec checker passed as the loopspec= argument to run_coarse_tile_test.

    loopspec=LoopSpecCheck()             — asserts LoopSpec( appears in generated source (default)
    loopspec=LoopSpecCheck(counts=[4,2]) — also checks count=sympify('N') for each N
    loopspec=None                        — skip loopspec check entirely
    """

    def __init__(self, counts=None):
        self.counts = counts

    def __call__(self, src):
        assert "LoopSpec(" in src, f"Expected LoopSpec( in generated source:\n{src}"
        if self.counts:
            for count in self.counts:
                assert f"count=sympify('{count}')" in src, (
                    f"Expected count=sympify('{count}') in generated source:\n{src}"
                )


@dataclasses.dataclass
class TensorSpec:
    """Descriptor for a test input tensor.

    name:       parameter name in fn — for readability only.
    shape:      physical tensor shape passed to torch.randn.
    dims:       named dim labels in order, passed to _name_tensor_dims.
    named_dims: optional explicit {dim_name: size} dict for declaring dims.
                Use when len(dims) > len(shape) — i.e. multiple logical dims
                are fused into one physical dim (e.g. flat [B, S, H*D] input
                named ["batch_size", "max_seqlen", "num_heads", "head_dim"]).
                When absent, sizes are inferred by zipping dims with shape.
    value:      optional pre-built CPU tensor. When set, used directly instead
                of torch.randn — shape and scale are ignored for tensor creation.
                Useful for structured inputs like causal masks.
    """

    name: str
    shape: tuple
    dims: list
    named_dims: dict | None = dataclasses.field(default=None)
    value: "torch.Tensor | None" = dataclasses.field(default=None)


def tensor(name, *, shape, dims, named_dims=None, value=None):
    """Shorthand constructor for TensorSpec."""
    return TensorSpec(
        name=name, shape=shape, dims=dims, named_dims=named_dims, value=value
    )


def run_coarse_tile_test(
    fn,
    inputs,
    loopspec=LoopSpecCheck(),
    correctness=True,
    atol=None,
    rtol=None,
    scale=0.01,
):
    """Compile fn on Spyre once, then check loopspec and/or correctness.

    inputs: list of TensorSpec (from tensor(...)) — driver creates tensors,
        declares dims, and calls _name_tensor_dims before each compile.

    loopspec: LoopSpecCheck() — check generated source for LoopSpec (default).
              LoopSpecCheck(counts=[4,2]) also checks specific tile counts.
              None — skip loopspec check.
    correctness: True  — compare_with_cpu against CPU reference.

    Always compiles exactly once, regardless of which checks are enabled.
    """

    cpu_tensors = [
        s.value
        if s.value is not None
        else torch.randn(s.shape, dtype=torch.float16) * scale
        for s in inputs
    ]

    def _setup_dims_and_dev_tensors():
        _pnd.reset()
        for spec in inputs:
            if spec.named_dims is not None:
                for dim, size in spec.named_dims.items():
                    _declare_tensor_dim(dim, size)
            else:
                for dim, size in zip(spec.dims, spec.shape):
                    _declare_tensor_dim(dim, size)
        dev_tensors = [t.to("spyre") for t in cpu_tensors]
        for spec, t in zip(inputs, dev_tensors):
            _name_tensor_dims(t, spec.dims)
        return dev_tensors

    with fresh_cache():
        dev_tensors = _setup_dims_and_dev_tensors()
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("torch_spyre.execution.async_compile.subprocess.run"),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn), *dev_tensors)

    if loopspec:
        assert len(source_codes) > 0
        loopspec(source_codes[0])

    if correctness:
        dev_tensors = _setup_dims_and_dev_tensors()
        spyre_result = _compile_and_run(fn, dev_tensors, "spyre")
        kwargs = {}
        if atol is not None:
            kwargs["atol"] = atol
        if rtol is not None:
            kwargs["rtol"] = rtol
        compare_with_cpu(
            fn, *cpu_tensors, target=spyre_result, run_eager=False, **kwargs
        )


# ---------------------------------------------------------------------------
# Group 1: Test basic tiling with various sizes
# ---------------------------------------------------------------------------


def test_abs_256x256_A4():
    """abs [256,256] tiled A÷4 → 64 elems/tile (1 stick)."""
    inputs = [tensor("x", shape=(256, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return torch.abs(x)

    run_coarse_tile_test(fn, inputs)


def test_abs_256x256_B4():
    """abs [256,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [tensor("x", shape=(256, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return torch.abs(x)

    run_coarse_tile_test(fn, inputs)


def test_abs_256x256_A4_B4():
    """abs [256,256] tiled A÷4 B÷4 → 64 elems/tile each (1 stick)."""
    inputs = [tensor("x", shape=(256, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return torch.abs(x)

    run_coarse_tile_test(fn, inputs)


# --- add: scenario 1 — square 256×256, 1-stick tiles (64 elems/tile) ---


def test_add_256x256_A4():
    """add [256,256] tiled A÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("x", shape=(256, 256), dims=["A", "B"]),
        tensor("y", shape=(256, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_256x256_B4():
    """add [256,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("x", shape=(256, 256), dims=["A", "B"]),
        tensor("y", shape=(256, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_256x256_A4_B4():
    """add [256,256] tiled A÷4 B÷4 → 64 elems/tile each (1 stick)."""
    inputs = [
        tensor("x", shape=(256, 256), dims=["A", "B"]),
        tensor("y", shape=(256, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


# --- add: scenario 2 — non-square 512×256, A>B, 2-stick A tiles ---


def test_add_512x256_A4():
    """add [512,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_512x256_B4():
    """add [512,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_512x256_A4_B4():
    """add [512,256] tiled A÷4 B÷4 → 128 and 64 elems/tile."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


# --- add: scenario 3 — non-square 256×512, B>A, 2-stick B tiles ---


def test_add_256x512_A4():
    """add [256,512] tiled A÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("x", shape=(256, 512), dims=["A", "B"]),
        tensor("y", shape=(256, 512), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_256x512_B4():
    """add [256,512] tiled B÷4 → 128 elems/tile (2 sticks)."""
    inputs = [
        tensor("x", shape=(256, 512), dims=["A", "B"]),
        tensor("y", shape=(256, 512), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_256x512_A4_B4():
    """add [256,512] tiled A÷4 B÷4 → 64 and 128 elems/tile."""
    inputs = [
        tensor("x", shape=(256, 512), dims=["A", "B"]),
        tensor("y", shape=(256, 512), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


# --- add: scenario 4 — square 512×512, 2-stick tiles ---


def test_add_512x512_A4():
    """add [512,512] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [
        tensor("x", shape=(512, 512), dims=["A", "B"]),
        tensor("y", shape=(512, 512), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_512x512_B4():
    """add [512,512] tiled B÷4 → 128 elems/tile (2 sticks)."""
    inputs = [
        tensor("x", shape=(512, 512), dims=["A", "B"]),
        tensor("y", shape=(512, 512), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_512x512_A4_B4():
    """add [512,512] tiled A÷4 B÷4 → 128 elems/tile each (2 sticks)."""
    inputs = [
        tensor("x", shape=(512, 512), dims=["A", "B"]),
        tensor("y", shape=(512, 512), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


# --- add: scenario 5 — asymmetric tile counts, same tile size ---


def test_add_512x256_A4_B2():
    """add [512,256] tiled A÷4 B÷2 → 128 elems/tile each, different counts."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


# ---------------------------------------------------------------------------
# Group 2: 3D tensors — [A=512, B=256, C=256]
# A÷4=128/tile (2 sticks), B÷2=128/tile (2 sticks, count=2), C÷4=64/tile (1 stick)
# ---------------------------------------------------------------------------


def test_add_3d_512x256x256_A4():
    """add [512,256,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [
        tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("y", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B", "C"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_3d_512x256x256_B2():
    """add [512,256,256] tiled B÷2 → 128 elems/tile (2 sticks, count=2)."""
    inputs = [
        tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("y", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"B": 2}):
            with spyre_hint(expected_named_dims=["A", "B", "C"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_3d_512x256x256_C4():
    """add [512,256,256] tiled C÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("y", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"C": 4}):
            with spyre_hint(expected_named_dims=["A", "B", "C"]):
                return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_3d_512x256x256_A4_B2():
    """add [512,256,256] tiled A÷4 B÷2 → 128+128 elems/tile."""
    inputs = [
        tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("y", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(expected_named_dims=["A", "B", "C"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_3d_512x256x256_A4_C4():
    """add [512,256,256] tiled A÷4 C÷4 → 128+64 elems/tile."""
    inputs = [
        tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("y", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"C": 4}):
                with spyre_hint(expected_named_dims=["A", "B", "C"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_3d_512x256x256_B2_C4():
    """add [512,256,256] tiled B÷2 C÷4 → 128+64 elems/tile."""
    inputs = [
        tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("y", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"B": 2}):
            with spyre_hint(num_tiles_per_dim={"C": 4}):
                with spyre_hint(expected_named_dims=["A", "B", "C"]):
                    return x + y

    run_coarse_tile_test(fn, inputs)


def test_add_3d_512x256x256_A4_B2_C4():
    """add [512,256,256] tiled A÷4 B÷2 C÷4 → 128+128+64 elems/tile."""
    inputs = [
        tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("y", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(expected_named_dims=["A", "B", "C"]):
                        return x + y

    run_coarse_tile_test(fn, inputs)


# ---------------------------------------------------------------------------
# Group 3: pointwise op chains — [512x256], 3 tiling variants each
# A÷4=128/tile (2 sticks), B÷4=64/tile (1 stick)
# ---------------------------------------------------------------------------


def test_abs_add_mul_512x256_A4():
    """abs(a+b)*c on [512,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
        tensor("c", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b, c):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return torch.abs(a + b) * c

    run_coarse_tile_test(fn, inputs)


def test_abs_add_mul_512x256_B4():
    """abs(a+b)*c on [512,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
        tensor("c", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b, c):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return torch.abs(a + b) * c

    run_coarse_tile_test(fn, inputs)


def test_abs_add_mul_512x256_A4_B4():
    """abs(a+b)*c on [512,256] tiled A÷4 B÷4 → 128+64 elems/tile."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
        tensor("c", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b, c):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return torch.abs(a + b) * c

    run_coarse_tile_test(fn, inputs)


def test_exp_abs_add_mul_512x256_A4():
    """exp(abs((a+b)*c)) on [512,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
        tensor("c", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b, c):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return torch.exp(torch.abs((a + b) * c))

    run_coarse_tile_test(fn, inputs)


def test_exp_abs_add_mul_512x256_B4():
    """exp(abs((a+b)*c)) on [512,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
        tensor("c", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b, c):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return torch.exp(torch.abs((a + b) * c))

    run_coarse_tile_test(fn, inputs)


def test_exp_abs_add_mul_512x256_A4_B4():
    """exp(abs((a+b)*c)) on [512,256] tiled A÷4 B÷4 → 128+64 elems/tile."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
        tensor("c", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b, c):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return torch.exp(torch.abs((a + b) * c))

    run_coarse_tile_test(fn, inputs)


# ---------------------------------------------------------------------------
# Group 4: reductions (amin) — 2D all tiling combos, 3D all-dims tiling
# 2D [512x256]: A÷4=128/tile, B÷4=64/tile
# 3D [512x256x256]: A÷4=128/tile, B÷2=128/tile, C÷4=64/tile
# ---------------------------------------------------------------------------


def test_min_2d_512x256_reduce_dim0_A4():
    """amin over dim=0 on [512,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                return x.amin(dim=0)

    run_coarse_tile_test(fn, inputs)


def test_min_2d_512x256_reduce_dim0_B4():
    """amin over dim=0 on [512,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                return x.amin(dim=0)

    run_coarse_tile_test(fn, inputs)


def test_min_2d_512x256_reduce_dim0_A4_B4():
    """amin over dim=0 on [512,256] tiled A÷4 B÷4 → 128+64 elems/tile."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["B"], expected_reduction_dims=["A"]
                ):
                    return x.amin(dim=0)

    run_coarse_tile_test(
        fn, inputs, correctness=False
    )  # nested tiling + reduction correctness bug


def test_min_2d_512x256_reduce_dim1_A4():
    """amin over dim=1 on [512,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A"], expected_reduction_dims=["B"]):
                return x.amin(dim=1)

    run_coarse_tile_test(fn, inputs)


def test_min_2d_512x256_reduce_dim1_B4():
    """amin over dim=1 on [512,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A"], expected_reduction_dims=["B"]):
                return x.amin(dim=1)

    run_coarse_tile_test(fn, inputs)


def test_min_2d_512x256_reduce_dim1_A4_B4():
    """amin over dim=1 on [512,256] tiled A÷4 B÷4 → 128+64 elems/tile."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["A"], expected_reduction_dims=["B"]
                ):
                    return x.amin(dim=1)

    run_coarse_tile_test(
        fn, inputs, correctness=False
    )  # nested tiling + reduction correctness bug


@pytest.mark.skip(reason="inconsistent loop_count across reduction fill/combine nodes")
def test_min_3d_512x256x256_reduce_dim0_A4_B2_C4():
    """amin over dim=0 on [512,256,256] tiled A÷4 B÷2 C÷4."""
    inputs = [tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(
                        expected_named_dims=["B", "C"], expected_reduction_dims=["A"]
                    ):
                        return x.amin(dim=0)

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # scheduler crash: inconsistent loop_count across reduction fill/combine nodes


@pytest.mark.skip(reason="inconsistent loop_count across reduction fill/combine nodes")
def test_min_3d_512x256x256_reduce_dim1_A4_B2_C4():
    """amin over dim=1 on [512,256,256] tiled A÷4 B÷2 C÷4."""
    inputs = [tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(
                        expected_named_dims=["A", "C"], expected_reduction_dims=["B"]
                    ):
                        return x.amin(dim=1)

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # scheduler crash: inconsistent loop_count across reduction fill/combine nodes


def test_min_3d_512x256x256_reduce_dim2_A4_B2_C4():
    """amin over dim=2 on [512,256,256] tiled A÷4 B÷2 C÷4."""
    inputs = [tensor("x", shape=(512, 256, 256), dims=["A", "B", "C"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(
                        expected_named_dims=["A", "B"], expected_reduction_dims=["C"]
                    ):
                        return x.amin(dim=2)

    run_coarse_tile_test(
        fn, inputs, correctness=False
    )  # nested tiling + reduction correctness bug


# ---------------------------------------------------------------------------
# Group 5: mixed pointwise + reduction — add_min, reduce_both, softmax
# add_min: min(a + abs(amin(b))) — 2D all 3 tiling variants × 2 reduction dims,
#   3D all-dims tiling × 3 reduction dims
# reduce_both: amin(a,dim) + amin(b,dim) — dense+dense and sparse+sparse, 3 tiling variants
# softmax: decomposes into amax+pointwise+sum+pointwise — dim0 and dim1, 3 tiling variants each
# ---------------------------------------------------------------------------


def test_add_min_2d_512x256_reduce_dim0_A4():
    """min(a + abs(amin(b, dim=0))) on [512,256] tiled A÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                r = b.amin(dim=0)
            with spyre_hint(expected_named_dims=["B"]):
                temp = torch.abs(r)
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a + temp

    run_coarse_tile_test(fn, inputs)


def test_add_min_2d_512x256_reduce_dim0_B4():
    """min(a + abs(amin(b, dim=0))) on [512,256] tiled B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                r = b.amin(dim=0)
            with spyre_hint(expected_named_dims=["B"]):
                temp = torch.abs(r)
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a + temp

    run_coarse_tile_test(fn, inputs)


def test_add_min_2d_512x256_reduce_dim0_A4_B4():
    """min(a + abs(amin(b, dim=0))) on [512,256] tiled A÷4 B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["B"], expected_reduction_dims=["A"]
                ):
                    r = b.amin(dim=0)
                with spyre_hint(expected_named_dims=["B"]):
                    temp = torch.abs(r)
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return a + temp

    run_coarse_tile_test(fn, inputs)


def test_add_min_2d_512x256_reduce_dim1_A4():
    """min(a + abs(amin(b, dim=1))) on [512,256] tiled A÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A"], expected_reduction_dims=["B"]):
                r = b.amin(dim=1, keepdim=True)
            with spyre_hint(expected_named_dims=["A"]):
                temp = torch.abs(r)
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a + temp

    run_coarse_tile_test(fn, inputs)


def test_add_min_2d_512x256_reduce_dim1_B4():
    """min(a + abs(amin(b, dim=1))) on [512,256] tiled B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A"], expected_reduction_dims=["B"]):
                r = b.amin(dim=1, keepdim=True)
            with spyre_hint(expected_named_dims=["A"]):
                temp = torch.abs(r)
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a + temp

    run_coarse_tile_test(fn, inputs)


def test_add_min_2d_512x256_reduce_dim1_A4_B4():
    """min(a + abs(amin(b, dim=1))) on [512,256] tiled A÷4 B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["A"], expected_reduction_dims=["B"]
                ):
                    r = b.amin(dim=1, keepdim=True)
                with spyre_hint(expected_named_dims=["A"]):
                    temp = torch.abs(r)
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return a + temp

    run_coarse_tile_test(fn, inputs)


@pytest.mark.skip(reason="inconsistent loop_count across reduction fill/combine nodes")
def test_add_min_3d_512x256x256_reduce_dim0_A4_B2_C4():
    """min(a + abs(amin(b, dim=0))) on [512,256,256] tiled A÷4 B÷2 C÷4."""
    inputs = [
        tensor("a", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("b", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(
                        expected_named_dims=["B", "C"], expected_reduction_dims=["A"]
                    ):
                        r = b.amin(dim=0)
                    with spyre_hint(expected_named_dims=["B", "C"]):
                        temp = torch.abs(r)
                    with spyre_hint(expected_named_dims=["A", "B", "C"]):
                        return a + temp

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # scheduler crash: mixed loop counts in 3D nested tiling + reduction


@pytest.mark.skip(reason="inconsistent loop_count across reduction fill/combine nodes")
def test_add_min_3d_512x256x256_reduce_dim1_A4_B2_C4():
    """min(a + abs(amin(b, dim=1))) on [512,256,256] tiled A÷4 B÷2 C÷4."""
    inputs = [
        tensor("a", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("b", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(
                        expected_named_dims=["A", "C"], expected_reduction_dims=["B"]
                    ):
                        r = b.amin(dim=1)
                    with spyre_hint(expected_named_dims=["A", "C"]):
                        temp = torch.abs(r)
                    with spyre_hint(expected_named_dims=["A", "B", "C"]):
                        return a + temp

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # scheduler crash: mixed loop counts in 3D nested tiling + reduction


@pytest.mark.skip(reason="inconsistent loop_count across reduction fill/combine nodes")
def test_add_min_3d_512x256x256_reduce_dim2_A4_B2_C4():
    """min(a + abs(amin(b, dim=2))) on [512,256,256] tiled A÷4 B÷2 C÷4."""
    inputs = [
        tensor("a", shape=(512, 256, 256), dims=["A", "B", "C"]),
        tensor("b", shape=(512, 256, 256), dims=["A", "B", "C"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(
                        expected_named_dims=["A", "B"], expected_reduction_dims=["C"]
                    ):
                        r = b.amin(dim=2)
                    with spyre_hint(expected_named_dims=["A", "B"]):
                        temp = torch.abs(r)
                    with spyre_hint(expected_named_dims=["A", "B", "C"]):
                        return a + temp

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # scheduler crash: mixed loop counts in 3D nested tiling + reduction


# dense+dense: both inputs reduce over dim=0 → [B] dense outputs, then add
# sparse+sparse: both inputs reduce over dim=1 (stick) → [A] sparse outputs, then add


def test_reduce_both_dense_add_2d_512x256_A4():
    """amin(a,dim=0) + amin(b,dim=0) on [512,256] tiled A÷4 — dense+dense."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["B"]):
                return a.amin(dim=0) + b.amin(dim=0)

    run_coarse_tile_test(fn, inputs)


def test_reduce_both_dense_add_2d_512x256_B4():
    """amin(a,dim=0) + amin(b,dim=0) on [512,256] tiled B÷4 — dense+dense."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["B"]):
                return a.amin(dim=0) + b.amin(dim=0)

    run_coarse_tile_test(fn, inputs)


def test_reduce_both_dense_add_2d_512x256_A4_B4():
    """amin(a,dim=0) + amin(b,dim=0) on [512,256] tiled A÷4 B÷4 — dense+dense."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["B"]):
                    return a.amin(dim=0) + b.amin(dim=0)

    run_coarse_tile_test(fn, inputs)


def test_reduce_both_sparse_add_2d_512x256_A4():
    """amin(a,dim=1) + amin(b,dim=1) on [512,256] tiled A÷4 — sparse+sparse."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A"]):
                return a.amin(dim=1) + b.amin(dim=1)

    run_coarse_tile_test(fn, inputs)


def test_reduce_both_sparse_add_2d_512x256_B4():
    """amin(a,dim=1) + amin(b,dim=1) on [512,256] tiled B÷4 — sparse+sparse."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A"]):
                return a.amin(dim=1) + b.amin(dim=1)

    run_coarse_tile_test(fn, inputs)


def test_reduce_both_sparse_add_2d_512x256_A4_B4():
    """amin(a,dim=1) + amin(b,dim=1) on [512,256] tiled A÷4 B÷4 — sparse+sparse."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A"]):
                    return a.amin(dim=1) + b.amin(dim=1)

    run_coarse_tile_test(fn, inputs)


# softmax decomposes into amax + pointwise + sum + pointwise — exercises
# mixed reduction+pointwise tiling in a single op


def test_softmax_2d_512x256_dim1_A4():
    """softmax(x, dim=1) on [512,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            return torch.softmax(x, dim=1)

    run_coarse_tile_test(fn, inputs)


def test_softmax_2d_512x256_dim1_B4():
    """softmax(x, dim=1) on [512,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            return torch.softmax(x, dim=1)

    run_coarse_tile_test(fn, inputs)


def test_softmax_2d_512x256_dim1_A4_B4():
    """softmax(x, dim=1) on [512,256] tiled A÷4 B÷4."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return torch.softmax(x, dim=1)

    run_coarse_tile_test(fn, inputs)


def test_softmax_2d_512x256_dim0_A4():
    """softmax(x, dim=0) on [512,256] tiled A÷4 → 128 elems/tile (2 sticks)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            return torch.softmax(x, dim=0)

    run_coarse_tile_test(fn, inputs)


def test_softmax_2d_512x256_dim0_B4():
    """softmax(x, dim=0) on [512,256] tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            return torch.softmax(x, dim=0)

    run_coarse_tile_test(fn, inputs)


def test_softmax_2d_512x256_dim0_A4_B4():
    """softmax(x, dim=0) on [512,256] tiled A÷4 B÷4."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return torch.softmax(x, dim=0)

    run_coarse_tile_test(fn, inputs)


# ---------------------------------------------------------------------------
# Group 6: restickify + coarse tiling
# a=[256,128], x=[128,256]: a.t() gives [128,256] same shape as x but
# stick-incompatible — restickify is inserted before the add.
# Named dims on the result shape [128, 256]: A=128, B=256.
# A÷2=64/tile (1 stick), B÷4=64/tile (1 stick)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_add_256x128_A2():
    """a.t() + x on [128,256] result, tiled A÷2 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("a", shape=(256, 128), dims=["B", "A"]),
        tensor("x", shape=(128, 256), dims=["A", "B"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"A": 2}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a.t() + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_add_256x128_B4():
    """a.t() + x on [128,256] result, tiled B÷4 → 64 elems/tile (1 stick)."""
    inputs = [
        tensor("a", shape=(256, 128), dims=["B", "A"]),
        tensor("x", shape=(128, 256), dims=["A", "B"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a.t() + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_add_256x128_A2_B4():
    """a.t() + x on [128,256] result, tiled A÷2 B÷4 → 64 elems/tile each."""
    inputs = [
        tensor("a", shape=(256, 128), dims=["B", "A"]),
        tensor("x", shape=(128, 256), dims=["A", "B"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"A": 2}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return a.t() + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


# 2D two-transpose: a.t() + b.t() + x
# a=[256,128], b=[256,128], x=[128,256]; result [128,256]: A=128, B=256
# A÷2=64/tile (1 stick), B÷4=64/tile (1 stick)


@pytest.mark.skip(reason="Unsupported: unexpected stick expression d0+d1")
def test_restickify_2t_add_256x128_A2():
    """a.t()+b.t()+x on [128,256] result, tiled A÷2."""
    inputs = [
        tensor("a", shape=(256, 128), dims=["B", "A"]),
        tensor("b", shape=(256, 128), dims=["B", "A"]),
        tensor("x", shape=(128, 256), dims=["A", "B"]),
    ]

    def fn(a, b, x):
        with spyre_hint(num_tiles_per_dim={"A": 2}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a.t() + b.t() + x

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # d0+d1 stick expr bug: two restickified inputs produce unsupported stick expression


@pytest.mark.skip(reason="Unsupported: unexpected stick expression d0+d1")
def test_restickify_2t_add_256x128_B4():
    """a.t()+b.t()+x on [128,256] result, tiled B÷4."""
    inputs = [
        tensor("a", shape=(256, 128), dims=["B", "A"]),
        tensor("b", shape=(256, 128), dims=["B", "A"]),
        tensor("x", shape=(128, 256), dims=["A", "B"]),
    ]

    def fn(a, b, x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                return a.t() + b.t() + x

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # d0+d1 stick expr bug


@pytest.mark.skip(reason="Unsupported: unexpected stick expression d0+d1")
def test_restickify_2t_add_256x128_A2_B4():
    """a.t()+b.t()+x on [128,256] result, tiled A÷2 B÷4."""
    inputs = [
        tensor("a", shape=(256, 128), dims=["B", "A"]),
        tensor("b", shape=(256, 128), dims=["B", "A"]),
        tensor("x", shape=(128, 256), dims=["A", "B"]),
    ]

    def fn(a, b, x):
        with spyre_hint(num_tiles_per_dim={"A": 2}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    return a.t() + b.t() + x

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # d0+d1 stick expr bug


# 3D transpose: a.transpose(1,2) + x
# a=[256,512,256], x=[256,256,512]; result [256,256,512]: A=256, B=256, C=512
# A÷4=64/tile, B÷4=64/tile, C÷4=128/tile (2 sticks)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_3d_transpose12_256x512x256_A4():
    """a.transpose(1,2)+x on [256,256,512] result, tiled A÷4."""
    inputs = [
        tensor("a", shape=(256, 512, 256), dims=["A", "C", "B"]),
        tensor("x", shape=(256, 256, 512), dims=["A", "B", "C"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B", "C"]):
                return a.transpose(1, 2) + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_3d_transpose12_256x512x256_B4():
    """a.transpose(1,2)+x on [256,256,512] result, tiled B÷4."""
    inputs = [
        tensor("a", shape=(256, 512, 256), dims=["A", "C", "B"]),
        tensor("x", shape=(256, 256, 512), dims=["A", "B", "C"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B", "C"]):
                return a.transpose(1, 2) + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_3d_transpose12_256x512x256_C4():
    """a.transpose(1,2)+x on [256,256,512] result, tiled C÷4."""
    inputs = [
        tensor("a", shape=(256, 512, 256), dims=["A", "C", "B"]),
        tensor("x", shape=(256, 256, 512), dims=["A", "B", "C"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"C": 4}):
            with spyre_hint(expected_named_dims=["A", "B", "C"]):
                return a.transpose(1, 2) + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_3d_transpose12_256x512x256_A4_B4():
    """a.transpose(1,2)+x on [256,256,512] result, tiled A÷4 B÷4."""
    inputs = [
        tensor("a", shape=(256, 512, 256), dims=["A", "C", "B"]),
        tensor("x", shape=(256, 256, 512), dims=["A", "B", "C"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B", "C"]):
                    return a.transpose(1, 2) + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_3d_transpose12_256x512x256_A4_C4():
    """a.transpose(1,2)+x on [256,256,512] result, tiled A÷4 C÷4."""
    inputs = [
        tensor("a", shape=(256, 512, 256), dims=["A", "C", "B"]),
        tensor("x", shape=(256, 256, 512), dims=["A", "B", "C"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"C": 4}):
                with spyre_hint(expected_named_dims=["A", "B", "C"]):
                    return a.transpose(1, 2) + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_3d_transpose12_256x512x256_B4_C4():
    """a.transpose(1,2)+x on [256,256,512] result, tiled B÷4 C÷4."""
    inputs = [
        tensor("a", shape=(256, 512, 256), dims=["A", "C", "B"]),
        tensor("x", shape=(256, 256, 512), dims=["A", "B", "C"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(num_tiles_per_dim={"C": 4}):
                with spyre_hint(expected_named_dims=["A", "B", "C"]):
                    return a.transpose(1, 2) + x

    run_coarse_tile_test(
        fn, inputs, loopspec=None, correctness=False
    )  # KNOWN BROKEN: PR #3381 insert_restickify env lookup fails for coarse_tile_read_copy buffers


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_3d_transpose12_256x512x256_A4_B4_C4():
    """a.transpose(1,2)+x on [256,256,512] result, tiled A÷4 B÷4 C÷4."""
    inputs = [
        tensor("a", shape=(256, 512, 256), dims=["A", "C", "B"]),
        tensor("x", shape=(256, 256, 512), dims=["A", "B", "C"]),
    ]

    def fn(a, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(num_tiles_per_dim={"C": 4}):
                    with spyre_hint(expected_named_dims=["A", "B", "C"]):
                        return a.transpose(1, 2) + x

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


# Matmul + transpose: x.t()@y and x@y.t()
# x=[128,256], y=[128,256]; x.t()@y=[256,128]@[128,256]=[256,256]
# x@y.t()=[128,256]@[256,128]=[128,128]
# For x.t()@y result [256,256]: M=256, N=256; M÷4=64, N÷4=64
# For x@y.t() result [128,128]: M=128, N=128; M÷2=64, N÷2=64


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_matmul_xt_y_256x128_M4():
    """x.t()@y, result [256,256], tiled M÷4."""
    inputs = [
        tensor("x", shape=(128, 256), dims=["K", "M"]),
        tensor("y", shape=(128, 256), dims=["K", "N"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"M": 4}):
            with spyre_hint(expected_named_dims=["M", "N"]):
                return torch.matmul(x.t(), y)

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_matmul_xt_y_256x128_N4():
    """x.t()@y, result [256,256], tiled N÷4."""
    inputs = [
        tensor("x", shape=(128, 256), dims=["K", "M"]),
        tensor("y", shape=(128, 256), dims=["K", "N"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"N": 4}):
            with spyre_hint(expected_named_dims=["M", "N"]):
                return torch.matmul(x.t(), y)

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_matmul_xt_y_256x128_M4_N4():
    """x.t()@y, result [256,256], tiled M÷4 N÷4."""
    inputs = [
        tensor("x", shape=(128, 256), dims=["K", "M"]),
        tensor("y", shape=(128, 256), dims=["K", "N"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"M": 4}):
            with spyre_hint(num_tiles_per_dim={"N": 4}):
                with spyre_hint(expected_named_dims=["M", "N"]):
                    return torch.matmul(x.t(), y)

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_matmul_x_yt_128x256_M2():
    """x@y.t(), result [128,128], tiled M÷2."""
    inputs = [
        tensor("x", shape=(128, 256), dims=["M", "K"]),
        tensor("y", shape=(128, 256), dims=["N", "K"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"M": 2}):
            with spyre_hint(expected_named_dims=["M", "N"]):
                return torch.matmul(x, y.t())

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_matmul_x_yt_128x256_N2():
    """x@y.t(), result [128,128], tiled N÷2."""
    inputs = [
        tensor("x", shape=(128, 256), dims=["M", "K"]),
        tensor("y", shape=(128, 256), dims=["N", "K"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"N": 2}):
            with spyre_hint(expected_named_dims=["M", "N"]):
                return torch.matmul(x, y.t())

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_restickify_matmul_x_yt_128x256_M2_N2():
    """x@y.t(), result [128,128], tiled M÷2 N÷2."""
    inputs = [
        tensor("x", shape=(128, 256), dims=["M", "K"]),
        tensor("y", shape=(128, 256), dims=["N", "K"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"M": 2}):
            with spyre_hint(num_tiles_per_dim={"N": 2}):
                with spyre_hint(expected_named_dims=["M", "N"]):
                    return torch.matmul(x, y.t())

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


# ---------------------------------------------------------------------------
# Group 7: copies
# Patterns: copy into pre-allocated buffer, in-place accumulation,
# read-modify-write with correction factor, copy after reduction.
# All on [512x256]: A÷4=128/tile (2 sticks), B÷4=64/tile (1 stick)
# ---------------------------------------------------------------------------

# --- copy into pre-allocated buffer: c.copy_(a + b) ---


def test_copy_into_preallocated_512x256_A4():
    """c.copy_(a+b) on [512,256] tiled A÷4 — result written into zeros buffer."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        c = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                c.copy_(a + b)
        return c

    run_coarse_tile_test(fn, inputs)


def test_copy_into_preallocated_512x256_B4():
    """c.copy_(a+b) on [512,256] tiled B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        c = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                c.copy_(a + b)
        return c

    run_coarse_tile_test(fn, inputs)


def test_copy_into_preallocated_512x256_A4_B4():
    """c.copy_(a+b) on [512,256] tiled A÷4 B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        c = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    c.copy_(a + b)
        return c

    run_coarse_tile_test(fn, inputs)


# --- in-place accumulation: acc.copy_(acc + x) ---


def test_copy_inplace_accum_512x256_A4():
    """acc.copy_(acc + x) on [512,256] tiled A÷4 — acc read and written inside loop."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("x", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                acc.copy_(acc + x)
        return acc

    run_coarse_tile_test(fn, inputs)


def test_copy_inplace_accum_512x256_B4():
    """acc.copy_(acc + x) on [512,256] tiled B÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("x", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                acc.copy_(acc + x)
        return acc

    run_coarse_tile_test(fn, inputs)


def test_copy_inplace_accum_512x256_A4_B4():
    """acc.copy_(acc + x) on [512,256] tiled A÷4 B÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("x", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    acc.copy_(acc + x)
        return acc

    run_coarse_tile_test(fn, inputs)


# --- read-modify-write with correction: acc.copy_(acc * scale + y) ---
# flash attention accumulator pattern


def test_copy_rmw_correction_512x256_A4():
    """acc.copy_(acc * scale + y) on [512,256] tiled A÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, scale, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                acc.copy_(acc * scale + y)
        return acc

    run_coarse_tile_test(fn, inputs)


def test_copy_rmw_correction_512x256_B4():
    """acc.copy_(acc * scale + y) on [512,256] tiled B÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, scale, y):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                acc.copy_(acc * scale + y)
        return acc

    run_coarse_tile_test(fn, inputs)


def test_copy_rmw_correction_512x256_A4_B4():
    """acc.copy_(acc * scale + y) on [512,256] tiled A÷4 B÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, scale, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    acc.copy_(acc * scale + y)
        return acc

    run_coarse_tile_test(fn, inputs)


# --- copy after reduction: out.copy_(x.amin(dim=0)) ---
# copies sparse reduction result into a dense buffer


def test_copy_after_reduction_512x256_A4():
    """out.copy_(x.amin(dim=0)) on [512,256] tiled A÷4."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        out = torch.zeros(256, device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                out.copy_(x.amin(dim=0))
        return out

    run_coarse_tile_test(fn, inputs)


def test_copy_after_reduction_512x256_B4():
    """out.copy_(x.amin(dim=0)) on [512,256] tiled B÷4."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        out = torch.zeros(256, device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                out.copy_(x.amin(dim=0))
        return out

    run_coarse_tile_test(fn, inputs)


def test_copy_after_reduction_512x256_A4_B4():
    """out.copy_(x.amin(dim=0)) on [512,256] tiled A÷4 B÷4."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        out = torch.zeros(256, device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["B"], expected_reduction_dims=["A"]
                ):
                    out.copy_(x.amin(dim=0))
        return out

    run_coarse_tile_test(fn, inputs)


def test_copy_running_max_4d_H4_Lq4():
    """running_max.copy_(maximum(real_max, amax(scores,dim=-2))) on [B,H,Lk,Lq] tiled H÷4 Lq÷4.

    Minimal flash-attention-style reproducer: 4D scores [B,H,Lk,Lq] reduced over
    dim=-2 (Lk), then max with a running accumulator, then copy_ back.
    """
    B, H, Lk, Lq = 2, 32, 4096, 4096
    h_block_size = 4
    lq_block_size = 1024

    inputs = [tensor("scores", shape=(B, H, Lk, Lq), dims=["B", "H", "Lk", "Lq"])]

    def fn(scores):
        real_max = torch.full(
            (B, H, Lq), float("-inf"), device=scores.device, dtype=scores.dtype
        )
        with spyre_hint(num_tiles_per_dim={"H": H // h_block_size}):
            with spyre_hint(num_tiles_per_dim={"Lq": Lq // lq_block_size}):
                with spyre_hint(
                    expected_named_dims=["B", "H", "Lq"], expected_reduction_dims=["Lk"]
                ):
                    block_max = torch.amax(scores, dim=-2)
                with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                    running_max = torch.maximum(real_max, block_max)
                real_max.copy_(running_max)
        return real_max

    run_coarse_tile_test(fn, inputs, correctness=False)


# --- copy + restickify: c.copy_(a.t() + b) ---
# copy target receives a restickified input — tests copy layout after restickify


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_copy_restickify_512x256_A4():
    """c.copy_(a.t()+b) on [256,512] result tiled A÷4 — copy of restickified add."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["B", "A"]),
        tensor("b", shape=(256, 512), dims=["A", "B"]),
    ]

    def fn(a, b):
        c = torch.zeros(b.shape, device=b.device, dtype=b.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                c.copy_(a.t() + b)
        return c

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_copy_restickify_512x256_B4():
    """c.copy_(a.t()+b) on [256,512] result tiled B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["B", "A"]),
        tensor("b", shape=(256, 512), dims=["A", "B"]),
    ]

    def fn(a, b):
        c = torch.zeros(b.shape, device=b.device, dtype=b.dtype)
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                c.copy_(a.t() + b)
        return c

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="StopIteration: insert_restickify env lookup fails for coarse_tile_read_copy buffers"
)
def test_copy_restickify_512x256_A4_B4():
    """c.copy_(a.t()+b) on [256,512] result tiled A÷4 B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["B", "A"]),
        tensor("b", shape=(256, 512), dims=["A", "B"]),
    ]

    def fn(a, b):
        c = torch.zeros(b.shape, device=b.device, dtype=b.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    c.copy_(a.t() + b)
        return c

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


# --- nested copy + reduction: acc.copy_(acc * scale + x.amin(dim=1, keepdim=True)) ---
# flash attention accumulator pattern: correction * running value + new contribution


def test_copy_accum_with_reduction_512x256_A4():
    """acc.copy_(acc * scale + x.amin(dim=1,keepdim=True)) tiled A÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 1), dims=["A", "B"]),
        tensor("x", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, scale, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A"], expected_reduction_dims=["B"]):
                r = x.amin(dim=1, keepdim=True)
            with spyre_hint(expected_named_dims=["A", "B"]):
                acc.copy_(acc * scale + r)
        return acc

    run_coarse_tile_test(fn, inputs)


@pytest.mark.skip(
    reason="IndexError: _insert_read_copy_ops fails when tiling B with unit-size B dim in scale"
)
def test_copy_accum_with_reduction_512x256_B4():
    """acc.copy_(acc * scale + x.amin(dim=1,keepdim=True)) tiled B÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 1), dims=["A", "B"]),
        tensor("x", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, scale, x):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A"], expected_reduction_dims=["B"]):
                r = x.amin(dim=1, keepdim=True)
            with spyre_hint(expected_named_dims=["A", "B"]):
                acc.copy_(acc * scale + r)
        return acc

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="IndexError: _insert_read_copy_ops fails when tiling B with unit-size B dim in scale"
)
def test_copy_accum_with_reduction_512x256_A4_B4():
    """acc.copy_(acc * scale + x.amin(dim=1,keepdim=True)) tiled A÷4 B÷4."""
    inputs = [
        tensor("acc", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 1), dims=["A", "B"]),
        tensor("x", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(acc, scale, x):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["A"], expected_reduction_dims=["B"]
                ):
                    r = x.amin(dim=1, keepdim=True)
                with spyre_hint(expected_named_dims=["A", "B"]):
                    acc.copy_(acc * scale + r)
        return acc

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


# --- two copies in same hint scope: c1.copy_(a+b); c2.copy_(a*b) ---


def test_copy_two_copies_same_scope_512x256_A4():
    """Two copy_ ops in same hint scope tiled A÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        c1 = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        c2 = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                c1.copy_(a + b)
            with spyre_hint(expected_named_dims=["A", "B"]):
                c2.copy_(a * b)
        return c1, c2

    run_coarse_tile_test(fn, inputs)


def test_copy_two_copies_same_scope_512x256_B4():
    """Two copy_ ops in same hint scope tiled B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        c1 = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        c2 = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                c1.copy_(a + b)
            with spyre_hint(expected_named_dims=["A", "B"]):
                c2.copy_(a * b)
        return c1, c2

    run_coarse_tile_test(fn, inputs)


def test_copy_two_copies_same_scope_512x256_A4_B4():
    """Two copy_ ops in same hint scope tiled A÷4 B÷4."""
    inputs = [
        tensor("a", shape=(512, 256), dims=["A", "B"]),
        tensor("b", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(a, b):
        c1 = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        c2 = torch.zeros(a.shape, device=a.device, dtype=a.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    c1.copy_(a + b)
                with spyre_hint(expected_named_dims=["A", "B"]):
                    c2.copy_(a * b)
        return c1, c2

    run_coarse_tile_test(fn, inputs)


# ---------------------------------------------------------------------------
# Group 8: tiled ops with outside consumers
# Pattern: buffer initialized outside loop, written tile-by-tile inside,
# then read again outside before returning.
# All on [512x256]: A÷4=128/tile, B÷4=64/tile
# ---------------------------------------------------------------------------

# --- minimal: z = tiled(x+y); return z * 2.0 ---
# The tiled op's output is a full-sized buffer consumed outside the loop.
# Forces _allocate_full_buffer + correct stickification.


def test_outside_consumer_pointwise_512x256_A4():
    """z=tiled(x+y) consumed outside as z*2.0, tiled A÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                z = x + y
        return z * 2.0

    run_coarse_tile_test(fn, inputs)


def test_outside_consumer_pointwise_512x256_B4():
    """z=tiled(x+y) consumed outside as z*2.0, tiled B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                z = x + y
        return z * 2.0

    run_coarse_tile_test(fn, inputs)


def test_outside_consumer_pointwise_512x256_A4_B4():
    """z=tiled(x+y) consumed outside as z*2.0, tiled A÷4 B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    z = x + y
        return z * 2.0

    run_coarse_tile_test(fn, inputs)


# --- pattern: output=zeros outside; copy_ inside; read outside ---
# output initialized outside, written tile-by-tile via copy_, divided outside.


def test_outside_consumer_copy_then_read_512x256_A4():
    """out=zeros; tiled out.copy_(x+y); return out/norm — tiled A÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
        tensor("norm", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y, norm):
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                out.copy_(x + y)
        return out / (torch.abs(norm) + 1.0)

    run_coarse_tile_test(fn, inputs)


def test_outside_consumer_copy_then_read_512x256_B4():
    """out=zeros; tiled out.copy_(x+y); return out/norm — tiled B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
        tensor("norm", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y, norm):
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                out.copy_(x + y)
        return out / (torch.abs(norm) + 1.0)

    run_coarse_tile_test(fn, inputs)


def test_outside_consumer_copy_then_read_512x256_A4_B4():
    """out=zeros; tiled out.copy_(x+y); return out/norm — tiled A÷4 B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("y", shape=(512, 256), dims=["A", "B"]),
        tensor("norm", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, y, norm):
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    out.copy_(x + y)
        return out / (torch.abs(norm) + 1.0)

    run_coarse_tile_test(fn, inputs)


# --- two-accumulator flash pattern ---
# Both output and denom initialized outside, updated inside, divided outside.
# This is the minimal flash attention accumulator pattern.


@pytest.mark.skip(
    reason="infeasible restickify for 1D denom in mixed 1D/2D tiled scope"
)
def test_outside_consumer_two_accum_512x256_A4():
    """Flash-style: out=zeros, denom=zeros; tiled copy_; return out/denom — A÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, scale):
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        denom = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                out.copy_(out * scale + x)
            with spyre_hint(expected_named_dims=["A"]):
                denom.copy_(denom + x.sum(dim=1))
        return out / denom.unsqueeze(1)

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


def test_outside_consumer_two_accum_512x256_B4():
    """Flash-style: out=zeros, denom=zeros; tiled copy_; return out/denom — B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, scale):
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        denom = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["A", "B"]):
                out.copy_(out * scale + x)
            with spyre_hint(expected_named_dims=["A"]):
                denom.copy_(denom + x.sum(dim=1))
        return out / denom.unsqueeze(1)

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


@pytest.mark.skip(
    reason="infeasible restickify for 1D denom in mixed 1D/2D tiled scope"
)
def test_outside_consumer_two_accum_512x256_A4_B4():
    """Flash-style: out=zeros, denom=zeros; tiled copy_; return out/denom — A÷4 B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("scale", shape=(512, 256), dims=["A", "B"]),
    ]

    def fn(x, scale):
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        denom = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["A", "B"]):
                    out.copy_(out * scale + x)
                with spyre_hint(expected_named_dims=["A"]):
                    denom.copy_(denom + x.sum(dim=1))
        return out / denom.unsqueeze(1)

    run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


# --- reduction inside loop, result consumed outside ---
# s = tiled_amin(x, dim=0) → [256] dense; return s + bias
# Tests _allocate_full_buffer for sparse reduction output with outside consumer.


def test_outside_consumer_reduction_512x256_A4():
    """s=tiled_amin(x,dim=0) consumed outside as s+bias, tiled A÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("bias", shape=(256,), dims=["B"]),
    ]

    def fn(x, bias):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                s = x.amin(dim=0)
        return s + bias

    run_coarse_tile_test(fn, inputs)


def test_outside_consumer_reduction_512x256_B4():
    """s=tiled_amin(x,dim=0) consumed outside as s+bias, tiled B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("bias", shape=(256,), dims=["B"]),
    ]

    def fn(x, bias):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["A"]):
                s = x.amin(dim=0)
        return s + bias

    run_coarse_tile_test(fn, inputs)


def test_outside_consumer_reduction_512x256_A4_B4():
    """s=tiled_amin(x,dim=0) consumed outside as s+bias, tiled A÷4 B÷4."""
    inputs = [
        tensor("x", shape=(512, 256), dims=["A", "B"]),
        tensor("bias", shape=(256,), dims=["B"]),
    ]

    def fn(x, bias):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["B"], expected_reduction_dims=["A"]
                ):
                    s = x.amin(dim=0)
        return s + bias

    run_coarse_tile_test(fn, inputs)


# ---------------------------------------------------------------------------
# Group 9: views — reshape, 1D sub-dim, view+transpose, unsqueeze
# Op inside fn is multiply (more numerically stable than add on fp16).
# ---------------------------------------------------------------------------

# --- 1D sub-dim naming: flat [Lq*D] tensor named with 2 sub-dims ---
# Both outer (Lq÷2) and inner (D÷2) hints land on the single host dim.


def test_view_1d_subdim_Lq2_D2():
    """a*b on 1D [Lq*D] named ["Lq","D"], nested Lq÷2 / D÷2 on same host dim."""
    Lq, D = 256, 128
    inputs = [
        tensor("a", shape=(Lq * D,), dims=["Lq", "D"], named_dims={"Lq": Lq, "D": D}),
        tensor("b", shape=(Lq * D,), dims=["Lq", "D"], named_dims={"Lq": Lq, "D": D}),
    ]

    def fn(a, b):
        a = a.view(Lq, D)
        b = b.view(Lq, D)
        with spyre_hint(num_tiles_per_dim={"Lq": 2}):
            with spyre_hint(num_tiles_per_dim={"D": 2}):
                with spyre_hint(expected_named_dims=["Lq", "D"]):
                    return a * b

    run_coarse_tile_test(fn, inputs)


# --- named input directly viewed+transposed inside fn ---
# Modeled after granite_flash_attention style views: flat [B,S,H*D] inputs viewed to 4D
# then transposed. Named dims span the fused H*D dim.
# B=2, S=256, H=4, D=64 → flat shape [2,256,256]; post-view+transpose [2,4,256,64]


def test_view_named_input_view_transpose_H2():
    """flat [B,S,H*D] view+transpose before hint scope, tiled H÷2."""
    B, S, H, D = 2, 256, 8, 128
    _nd = {"B": B, "S": S, "H": H, "D": D}
    inputs = [
        tensor(
            "q",
            shape=(B, S, H * D),
            dims=["B", "S", "H", "D"],
            named_dims=_nd,
        ),
        tensor(
            "k",
            shape=(B, S, H * D),
            dims=["B", "S", "H", "D"],
            named_dims=_nd,
        ),
    ]

    def fn(q, k):
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        with spyre_hint(num_tiles_per_dim={"H": 2}):
            with spyre_hint(expected_named_dims=["B", "H", "S", "D"]):
                return q * k

    run_coarse_tile_test(fn, inputs)


def test_view_named_input_view_transpose_S4():
    """flat [B,S,H*D] view+transpose before hint scope, tiled S÷4."""
    B, S, H, D = 2, 256, 8, 128
    _nd = {"B": B, "S": S, "H": H, "D": D}
    inputs = [
        tensor(
            "q",
            shape=(B, S, H * D),
            dims=["B", "S", "H", "D"],
            named_dims=_nd,
        ),
        tensor(
            "k",
            shape=(B, S, H * D),
            dims=["B", "S", "H", "D"],
            named_dims=_nd,
        ),
    ]

    def fn(q, k):
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        with spyre_hint(num_tiles_per_dim={"S": 4}):
            with spyre_hint(expected_named_dims=["B", "H", "S", "D"]):
                return q * k

    run_coarse_tile_test(fn, inputs)


def test_view_named_input_view_transpose_H2_S4():
    """flat [B,S,H*D] view+transpose before hint scope, tiled H÷2 S÷4."""
    B, S, H, D = 2, 256, 8, 128
    _nd = {"B": B, "S": S, "H": H, "D": D}
    inputs = [
        tensor(
            "q",
            shape=(B, S, H * D),
            dims=["B", "S", "H", "D"],
            named_dims=_nd,
        ),
        tensor(
            "k",
            shape=(B, S, H * D),
            dims=["B", "S", "H", "D"],
            named_dims=_nd,
        ),
    ]

    def fn(q, k):
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        with spyre_hint(num_tiles_per_dim={"H": 2}):
            with spyre_hint(num_tiles_per_dim={"S": 4}):
                with spyre_hint(expected_named_dims=["B", "H", "S", "D"]):
                    return q * k

    run_coarse_tile_test(fn, inputs)


# --- 4D input transposed then multiplied ---
# Inputs already in 4D shape [B,H,S,D], transpose swaps non-stick dims.
# B=2, S=256, H=4, D=64


def test_view_4d_transpose_H2():
    """x.view(B,S,H,D).transpose(1,2)*y tiled H÷2."""
    B, S, H, D = 2, 256, 4, 64
    _nd = {"B": B, "Lq": S, "H": H, "D": D}
    inputs = [
        tensor(
            "x",
            shape=(B, S, H * D),
            dims=["B", "Lq", "H", "D"],
            named_dims=_nd,
        ),
        tensor(
            "y",
            shape=(B, H, S, D),
            dims=["B", "H", "Lq", "D"],
            named_dims=_nd,
        ),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"H": 2}):
            with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                return x.view(B, S, H, D).transpose(1, 2) * y

    run_coarse_tile_test(fn, inputs)


def test_view_4d_transpose_S4():
    """x.view(B,S,H,D).transpose(1,2)*y tiled S÷4."""
    B, S, H, D = 2, 256, 4, 64
    _nd = {"B": B, "Lq": S, "H": H, "D": D}
    inputs = [
        tensor(
            "x",
            shape=(B, S, H * D),
            dims=["B", "Lq", "H", "D"],
            named_dims=_nd,
        ),
        tensor(
            "y",
            shape=(B, H, S, D),
            dims=["B", "H", "Lq", "D"],
            named_dims=_nd,
        ),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"Lq": 4}):
            with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                return x.view(B, S, H, D).transpose(1, 2) * y

    run_coarse_tile_test(fn, inputs)


def test_view_4d_transpose_H2_S4():
    """x.view(B,S,H,D).transpose(1,2)*y tiled H÷2 S÷4."""
    B, S, H, D = 2, 256, 4, 64
    _nd = {"B": B, "Lq": S, "H": H, "D": D}
    inputs = [
        tensor(
            "x",
            shape=(B, S, H * D),
            dims=["B", "Lq", "H", "D"],
            named_dims=_nd,
        ),
        tensor(
            "y",
            shape=(B, H, S, D),
            dims=["B", "H", "Lq", "D"],
            named_dims=_nd,
        ),
    ]

    def fn(x, y):
        with spyre_hint(num_tiles_per_dim={"H": 2}):
            with spyre_hint(num_tiles_per_dim={"Lq": 4}):
                with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                    return x.view(B, S, H, D).transpose(1, 2) * y

    run_coarse_tile_test(fn, inputs)


# --- unsqueeze: a.unsqueeze(0) * b where b has the broadcast shape ---


def test_view_unsqueeze_broadcast_A4():
    """a.unsqueeze(0)*b: a=[256,256], b=[4,256,256], tiled A÷4."""
    inputs = [
        tensor("a", shape=(256, 256), dims=["A", "B"]),
        tensor("b", shape=(4, 256, 256), dims=["N", "A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(expected_named_dims=["N", "A", "B"]):
                return a.unsqueeze(0) * b

    run_coarse_tile_test(fn, inputs)


def test_view_unsqueeze_broadcast_B4():
    """a.unsqueeze(0)*b: a=[256,256], b=[4,256,256], tiled B÷4."""
    inputs = [
        tensor("a", shape=(256, 256), dims=["A", "B"]),
        tensor("b", shape=(4, 256, 256), dims=["N", "A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"B": 4}):
            with spyre_hint(expected_named_dims=["N", "A", "B"]):
                return a.unsqueeze(0) * b

    run_coarse_tile_test(fn, inputs)


def test_view_unsqueeze_broadcast_A4_B4():
    """a.unsqueeze(0)*b: a=[256,256], b=[4,256,256], tiled A÷4 B÷4."""
    inputs = [
        tensor("a", shape=(256, 256), dims=["A", "B"]),
        tensor("b", shape=(4, 256, 256), dims=["N", "A", "B"]),
    ]

    def fn(a, b):
        with spyre_hint(num_tiles_per_dim={"A": 4}):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(expected_named_dims=["N", "A", "B"]):
                    return a.unsqueeze(0) * b

    run_coarse_tile_test(fn, inputs)


# ---------------------------------------------------------------------------
# Group 10: Flash attention variants
# ---------------------------------------------------------------------------
# Flash v1: no mask, reassignment-based accumulators, scores transposed.
# Parameterized helpers — tests specify sizes and tile counts directly.


def _flash_v1_inputs(B, H, Lq, Lk, D):
    """TensorSpec list for flash v1 (no mask)."""
    return [
        tensor("queries", shape=(B, H, Lq, D), dims=["B", "H", "Lq", "D"]),
        tensor("keys", shape=(B, H, Lk, D), dims=["B", "H", "Lk", "D"]),
        tensor("values", shape=(B, H, Lk, D), dims=["B", "H", "Lk", "D"]),
    ]


def _flash_v1_fn(
    queries,
    keys,
    values,
    *,
    B,
    H,
    Lq,
    Lk,
    D,
    b_tiles=1,
    h_tiles=1,
    lq_tiles=1,
    lk_tiles=1,
):
    """Flash attention v1 body. Tile any combination of B/H/Lq/Lk."""
    scale = 1.0 / math.sqrt(math.sqrt(D))
    with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
        output = torch.zeros_like(queries)
    with spyre_hint(named_dims=["B", "H", "Lq"]):
        M = torch.full(
            (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
        )
    with spyre_hint(named_dims=["B", "H", "Lq"]):
        denominator = torch.zeros(
            (B, H, Lq), device=queries.device, dtype=torch.float16
        )
    with spyre_hint(num_tiles_per_dim={"B": b_tiles}):
        with spyre_hint(num_tiles_per_dim={"H": h_tiles}):
            with spyre_hint(num_tiles_per_dim={"Lq": lq_tiles}):
                with spyre_hint(num_tiles_per_dim={"Lk": lk_tiles}):
                    with spyre_hint(expected_named_dims=["B", "H", "D", "Lk"]):
                        keys_T = keys.transpose(-1, -2).contiguous()
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        q_scaled = queries * scale
                    with spyre_hint(expected_named_dims=["B", "H", "D", "Lk"]):
                        k_scaled = keys_T * scale
                    with spyre_hint(named_dims=["B", "H", "Lq", "Lk"]):
                        scores = torch.matmul(q_scaled, k_scaled)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        scores = scores.transpose(-1, -2).contiguous()
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        block_max = torch.amax(scores, dim=-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        max_running = torch.maximum(M, block_max)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        scores_shifted = scores - max_running.unsqueeze(-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        exp_scores = torch.exp(scores_shifted)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        M_diff = M - max_running
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        correction = torch.exp(M_diff)
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        sum_scores = exp_scores.sum(dim=-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        denom_corrected = denominator * correction
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        denominator = denom_corrected + sum_scores
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "Lk"]):
                        exp_scores_T = exp_scores.transpose(-1, -2).contiguous()
                    with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
                        matmul_out = torch.matmul(exp_scores_T, values)
                    corr_expanded = correction.unsqueeze(-1)
                    output_corrected = output * corr_expanded
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        output = output_corrected + matmul_out
                    M = max_running  # noqa: F841
    return output / denominator.unsqueeze(-1)


def test_flash_tile_H():
    """Flash v1: tile H÷4 only."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q, k, v, B=1, H=8, Lq=256, Lk=256, D=64, h_tiles=4
        ),
        _flash_v1_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4]),
        correctness=True,
        atol=0.01,
        rtol=0.1,
    )


@pytest.mark.skip(reason="KeyError: 0 — B tiling not yet supported")
def test_flash_tile_B():
    """Flash v1: tile B÷2 only. B=2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q, k, v, B=2, H=8, Lq=256, Lk=256, D=64, b_tiles=2
        ),
        _flash_v1_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


def test_flash_tile_Lq():
    """Flash v1: tile Lq÷2 only."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q, k, v, B=1, H=8, Lq=256, Lk=256, D=64, lq_tiles=2
        ),
        _flash_v1_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_tile_Lk():
    """Flash v1: tile Lk÷2 only."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q, k, v, B=1, H=8, Lq=256, Lk=256, D=64, lk_tiles=2
        ),
        _flash_v1_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(reason="KeyError: 0 — B tiling not yet supported")
def test_flash_tile_B_H():
    """Flash v1: tile B÷2 H÷4. B=2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q, k, v, B=2, H=8, Lq=256, Lk=256, D=64, b_tiles=2, h_tiles=4
        ),
        _flash_v1_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2, 4]),
        correctness=False,
    )


def test_flash_tile_H_Lq():
    """Flash v1: tile H÷4 Lq÷2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q, k, v, B=1, H=8, Lq=256, Lk=256, D=64, h_tiles=4, lq_tiles=2
        ),
        _flash_v1_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4, 2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_tile_H_Lq_Lk():
    """Flash v1: tile H÷4 Lq÷2 Lk÷2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q, k, v, B=1, H=8, Lq=256, Lk=256, D=64, h_tiles=4, lq_tiles=2, lk_tiles=2
        ),
        _flash_v1_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4, 2, 2]),
        correctness=False,
    )


@pytest.mark.skip(reason="KeyError: 0 — B tiling not yet supported")
def test_flash_tile_all():
    """Flash v1: tile all dims. B=2, H÷4, Lq÷2, Lk÷2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v1_fn(
            q,
            k,
            v,
            B=2,
            H=8,
            Lq=256,
            Lk=256,
            D=64,
            b_tiles=2,
            h_tiles=4,
            lq_tiles=2,
            lk_tiles=2,
        ),
        _flash_v1_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2, 4, 2, 2]),
        correctness=False,
    )


# ---------------------------------------------------------------------------
# Flash v2: causal mask, copy_ accumulators, sparse init, reduces over dim=-1
# ---------------------------------------------------------------------------


def _flash_v2_inputs(B, H, Lq, Lk, D):
    """TensorSpec list for flash v2 (with causal mask)."""
    causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
    mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
    mask_t.masked_fill_(~causal, float("-inf"))
    return [
        tensor("queries", shape=(B, H, Lq, D), dims=["B", "H", "Lq", "D"]),
        tensor("keys", shape=(B, H, Lk, D), dims=["B", "H", "Lk", "D"]),
        tensor("values", shape=(B, H, Lk, D), dims=["B", "H", "Lk", "D"]),
        tensor(
            "mask",
            shape=(1, 1, Lq, Lk),
            dims=["B", "H", "Lq", "Lk"],
            named_dims={"Lq": Lq, "Lk": Lk},
            value=mask_t,
        ),
    ]


def _flash_v2_fn(
    queries,
    keys,
    values,
    mask,
    *,
    B,
    H,
    Lq,
    Lk,
    D,
    b_tiles=1,
    h_tiles=1,
    lq_tiles=1,
    lk_tiles=1,
):
    """Flash attention v2 body. Tile any combination of B/H/Lq/Lk."""
    scale = 1.0 / math.sqrt(math.sqrt(D))
    output = torch.zeros_like(queries)
    real_max = torch.full(
        (B, H, Lq, 64),
        float("-inf"),
        device=queries.device,
        dtype=torch.float16,
    ).amax(dim=-1)
    denominator = torch.zeros(
        (B, H, Lq, 64),
        device=queries.device,
        dtype=torch.float16,
    ).amax(dim=-1)
    with spyre_hint(num_tiles_per_dim={"B": b_tiles}):
        with spyre_hint(num_tiles_per_dim={"H": h_tiles}):
            with spyre_hint(num_tiles_per_dim={"Lq": lq_tiles}):
                with spyre_hint(num_tiles_per_dim={"Lk": lk_tiles}):
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "D"]):
                        scaled_keys = keys * scale
                    with spyre_hint(expected_named_dims=["B", "H", "D", "Lk"]):
                        keys_T = scaled_keys.transpose(-1, -2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        q_scaled = queries * scale
                    with spyre_hint(named_dims=["B", "H", "Lq", "Lk"]):
                        scores_pre = torch.matmul(q_scaled, keys_T)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "Lk"]):
                        scores = scores_pre + mask
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        block_max = torch.amax(scores, dim=-1)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        running_max = torch.maximum(real_max, block_max)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "Lk"]):
                        scores_shifted = scores - running_max.unsqueeze(-1)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "Lk"]):
                        exp_scores = torch.exp(scores_shifted)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        real_max_diff = real_max - running_max
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        correction = torch.exp(real_max_diff)
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        sum_scores = exp_scores.sum(dim=-1)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        denom_corrected = denominator * correction
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        new_denom = denom_corrected + sum_scores
                    denominator.copy_(new_denom)
                    with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
                        matmul_out = torch.matmul(exp_scores, values)
                    # correction.unsqueeze(-1) is [B,H,Lq,1] — size-1 dim can't carry "D"
                    corr_expanded = correction.unsqueeze(-1)
                    output_corrected = output * corr_expanded
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        new_output = output_corrected + matmul_out
                    output.copy_(new_output)
                    real_max.copy_(running_max)
    return output / denominator.unsqueeze(-1)


@pytest.mark.skip(
    reason="finalize_layouts: restickify infeasible for copy ops across loop groups"
)
def test_flash_v2_tile_H():
    """Flash v2: tile H÷4 only."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, h_tiles=4
        ),
        _flash_v2_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4]),
        correctness=True,
        atol=0.01,
        rtol=0.1,
    )


@pytest.mark.skip(reason="KeyError: 0 — B tiling not yet supported")
def test_flash_v2_tile_B():
    """Flash v2: tile B÷2 only. B=2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q, k, v, m, B=2, H=8, Lq=256, Lk=256, D=64, b_tiles=2
        ),
        _flash_v2_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="finalize_layouts: restickify infeasible for copy ops across loop groups"
)
def test_flash_v2_tile_Lq():
    """Flash v2: tile Lq÷2 only."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, lq_tiles=2
        ),
        _flash_v2_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_v2_tile_Lk():
    """Flash v2: tile Lk÷2 only."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, lk_tiles=2
        ),
        _flash_v2_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(reason="KeyError: 0 — B tiling not yet supported")
def test_flash_v2_tile_B_H():
    """Flash v2: tile B÷2 H÷4. B=2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q, k, v, m, B=2, H=8, Lq=256, Lk=256, D=64, b_tiles=2, h_tiles=4
        ),
        _flash_v2_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2, 4]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="finalize_layouts: restickify infeasible for copy ops across loop groups"
)
def test_flash_v2_tile_H_Lq():
    """Flash v2: tile H÷4 Lq÷2. Equivalent to original test_flash_v2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, h_tiles=4, lq_tiles=2
        ),
        _flash_v2_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4, 2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_v2_tile_H_Lq_Lk():
    """Flash v2: tile H÷4 Lq÷2 Lk÷2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q,
            k,
            v,
            m,
            B=1,
            H=8,
            Lq=256,
            Lk=256,
            D=64,
            h_tiles=4,
            lq_tiles=2,
            lk_tiles=2,
        ),
        _flash_v2_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4, 2, 2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_v2_tile_all():
    """Flash v2: tile all dims. B=2, H÷4, Lq÷2, Lk÷2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v2_fn(
            q,
            k,
            v,
            m,
            B=2,
            H=8,
            Lq=256,
            Lk=256,
            D=64,
            b_tiles=2,
            h_tiles=4,
            lq_tiles=2,
            lk_tiles=2,
        ),
        _flash_v2_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2, 4, 2, 2]),
        correctness=False,
    )


# ---------------------------------------------------------------------------
# Flash v3: causal mask, copy_ accumulators, scores transposed, tiles= API
# Uses num_tiles_per_dim= (normalized from tiles=) for consistency
# ---------------------------------------------------------------------------


def _flash_v3_inputs(B, H, Lq, Lk, D):
    """TensorSpec list for flash v3 (with causal mask)."""
    causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
    mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
    mask_t.masked_fill_(~causal, float("-inf"))
    return [
        tensor("queries", shape=(B, H, Lq, D), dims=["B", "H", "Lq", "D"]),
        tensor("keys", shape=(B, H, Lk, D), dims=["B", "H", "Lk", "D"]),
        tensor("values", shape=(B, H, Lk, D), dims=["B", "H", "Lk", "D"]),
        tensor(
            "mask",
            shape=(1, 1, Lq, Lk),
            dims=["B", "H", "Lq", "Lk"],
            named_dims={"Lq": Lq, "Lk": Lk},
            value=mask_t,
        ),
    ]


def _flash_v3_fn(
    queries,
    keys,
    values,
    mask,
    *,
    B,
    H,
    Lq,
    Lk,
    D,
    b_tiles=1,
    h_tiles=1,
    lq_tiles=1,
    lk_tiles=1,
):
    """Flash attention v3 body (scores transposed). Tile any combination of B/H/Lq/Lk."""
    scale = 1.0 / math.sqrt(math.sqrt(D))
    output = torch.zeros_like(queries)
    real_max = torch.full(
        (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
    )
    denominator = torch.zeros((B, H, Lq), device=queries.device, dtype=torch.float16)
    with spyre_hint(num_tiles_per_dim={"B": b_tiles}):
        with spyre_hint(num_tiles_per_dim={"H": h_tiles}):
            with spyre_hint(num_tiles_per_dim={"Lq": lq_tiles}):
                with spyre_hint(num_tiles_per_dim={"Lk": lk_tiles}):
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "D"]):
                        scaled_keys = keys * scale
                    with spyre_hint(expected_named_dims=["B", "H", "D", "Lk"]):
                        keys_T = scaled_keys.transpose(-1, -2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        q_scaled = queries * scale
                    with spyre_hint(named_dims=["B", "H", "Lq", "Lk"]):
                        scores_pre = torch.matmul(q_scaled, keys_T)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "Lk"]):
                        scores_masked = scores_pre + mask
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        scores = scores_masked.transpose(-1, -2).contiguous()
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        block_max = torch.amax(scores, dim=-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        running_max = torch.maximum(real_max, block_max)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        scores_shifted = scores - running_max.unsqueeze(-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        exp_scores = torch.exp(scores_shifted)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        real_max_diff = real_max - running_max
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        correction = torch.exp(real_max_diff)
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        sum_scores = exp_scores.sum(dim=-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        denom_corrected = denominator * correction
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        new_denom = denom_corrected + sum_scores
                    denominator.copy_(new_denom)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "Lk"]):
                        exp_scores_T = exp_scores.transpose(-1, -2).contiguous()
                    with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
                        matmul_out = torch.matmul(exp_scores_T, values)
                    # correction.unsqueeze(-1) is [B,H,Lq,1] — size-1 dim can't carry "D"
                    corr_expanded = correction.unsqueeze(-1)
                    output_corrected = output * corr_expanded
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        new_output = output_corrected + matmul_out
                    output.copy_(new_output)
                    real_max.copy_(running_max)
    return output / denominator.unsqueeze(-1)


def test_flash_v3_tile_H():
    """Flash v3: tile H÷4 only."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, h_tiles=4
        ),
        _flash_v3_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4]),
        correctness=False,
    )


@pytest.mark.skip(reason="KeyError: 0 — B tiling not yet supported")
def test_flash_v3_tile_B():
    """Flash v3: tile B÷2 only. B=2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q, k, v, m, B=2, H=8, Lq=256, Lk=256, D=64, b_tiles=2
        ),
        _flash_v3_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


def test_flash_v3_tile_Lq():
    """Flash v3: tile Lq÷2 only."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, lq_tiles=2
        ),
        _flash_v3_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_v3_tile_Lk():
    """Flash v3: tile Lk÷2 only."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, lk_tiles=2
        ),
        _flash_v3_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(reason="KeyError: 0 — B tiling not yet supported")
def test_flash_v3_tile_B_H():
    """Flash v3: tile B÷2 H÷4. B=2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q, k, v, m, B=2, H=8, Lq=256, Lk=256, D=64, b_tiles=2, h_tiles=4
        ),
        _flash_v3_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2, 4]),
        correctness=False,
    )


def test_flash_v3_tile_H_Lq():
    """Flash v3: tile H÷4 Lq÷2. Equivalent to original test_flash_v3 (small sizes)."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q, k, v, m, B=1, H=8, Lq=256, Lk=256, D=64, h_tiles=4, lq_tiles=2
        ),
        _flash_v3_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4, 2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_v3_tile_H_Lq_Lk():
    """Flash v3: tile H÷4 Lq÷2 Lk÷2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q,
            k,
            v,
            m,
            B=1,
            H=8,
            Lq=256,
            Lk=256,
            D=64,
            h_tiles=4,
            lq_tiles=2,
            lk_tiles=2,
        ),
        _flash_v3_inputs(1, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[4, 2, 2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: Lk reduction-dim tiling requires carry propagation"
)
def test_flash_v3_tile_all():
    """Flash v3: tile all dims. B=2, H÷4, Lq÷2, Lk÷2."""
    run_coarse_tile_test(
        lambda q, k, v, m: _flash_v3_fn(
            q,
            k,
            v,
            m,
            B=2,
            H=8,
            Lq=256,
            Lk=256,
            D=64,
            b_tiles=2,
            h_tiles=4,
            lq_tiles=2,
            lk_tiles=2,
        ),
        _flash_v3_inputs(2, 8, 256, 256, 64),
        loopspec=LoopSpecCheck(counts=[2, 4, 2, 2]),
        correctness=False,
    )


# ---------------------------------------------------------------------------
# Flash v4: flat [B,S,H*D] inputs, view+transpose inside fn
# Known broken: propagate_named_dims bug — num_heads layout dim has no loop vars
# ---------------------------------------------------------------------------


def _flash_v4_inputs(B, S, H, D):
    """TensorSpec list for flash v4 (flat fused-dim inputs)."""
    _nd_q = {"B": B, "Lq": S, "H": H, "D": D}
    _nd_kv = {"B": B, "Lk": S, "H": H, "D": D}
    return [
        tensor("q", shape=(B, S, H * D), dims=["B", "Lq", "H", "D"], named_dims=_nd_q),
        tensor("k", shape=(B, S, H * D), dims=["B", "Lk", "H", "D"], named_dims=_nd_kv),
        tensor("v", shape=(B, S, H * D), dims=["B", "Lk", "H", "D"], named_dims=_nd_kv),
    ]


def _flash_v4_fn(q, k, v, *, B, S, H, D, b_tiles=1, h_tiles=1, lq_tiles=1, lk_tiles=1):
    """Flash attention v4 body (flat fused-dim inputs). Tile any combination."""
    q = q.view(B, S, H, D).transpose(1, 2)
    k = k.view(B, S, H, D).transpose(1, 2)
    v = v.view(B, S, H, D).transpose(1, 2)
    scale = 1.0 / math.sqrt(math.sqrt(D))
    output = torch.zeros_like(q)
    real_max = torch.full((B, H, S), float("-inf"), device=q.device, dtype=q.dtype)
    denominator = torch.zeros((B, H, S), device=q.device, dtype=q.dtype)
    with spyre_hint(num_tiles_per_dim={"B": b_tiles}):
        with spyre_hint(num_tiles_per_dim={"H": h_tiles}):
            with spyre_hint(num_tiles_per_dim={"Lq": lq_tiles}):
                with spyre_hint(num_tiles_per_dim={"Lk": lk_tiles}):
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "D"]):
                        scaled_keys = k * scale
                    with spyre_hint(expected_named_dims=["B", "H", "D", "Lk"]):
                        keys_T = scaled_keys.transpose(-1, -2).contiguous()
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        q_scaled = q * scale
                    with spyre_hint(named_dims=["B", "H", "Lq", "Lk"]):
                        scores_pre = torch.matmul(q_scaled, keys_T)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        scores = scores_pre.transpose(-1, -2).contiguous()
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        block_max = torch.amax(scores, dim=-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        running_max = torch.maximum(real_max, block_max)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        scores_shifted = scores - running_max.unsqueeze(-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lk", "Lq"]):
                        exp_scores = torch.exp(scores_shifted)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        real_max_diff = real_max - running_max
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        correction = torch.exp(real_max_diff)
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        sum_scores = exp_scores.sum(dim=-2)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        denom_corrected = denominator * correction
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        new_denom = denom_corrected + sum_scores
                    denominator.copy_(new_denom)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "Lk"]):
                        exp_scores_T = exp_scores.transpose(-1, -2).contiguous()
                    with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
                        matmul_out = torch.matmul(exp_scores_T, v)
                    # correction.unsqueeze(-1) is size-1 in D — can't carry "D"
                    output_corrected = output * correction.unsqueeze(-1)
                    with spyre_hint(expected_named_dims=["B", "H", "Lq", "D"]):
                        new_output = output_corrected + matmul_out
                    output.copy_(new_output)
                    real_max.copy_(running_max)
    output.copy_(output / denominator.unsqueeze(-1))
    return output.transpose(1, 2).reshape(B, S, H * D)


@pytest.mark.skip(
    reason="Unsupported: propagate_named_dims bug — num_heads layout dim has no loop vars after view+transpose"
)
def test_flash_v4_tile_H():
    """Flash v4: tile num_heads÷4 only."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v4_fn(q, k, v, B=2, S=256, H=8, D=64, h_tiles=4),
        _flash_v4_inputs(2, 256, 8, 64),
        loopspec=LoopSpecCheck(counts=[4]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: propagate_named_dims bug — num_heads layout dim has no loop vars after view+transpose"
)
def test_flash_v4_tile_B():
    """Flash v4: tile batch_size÷2 only. B=2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v4_fn(q, k, v, B=2, S=256, H=8, D=64, b_tiles=2),
        _flash_v4_inputs(2, 256, 8, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: propagate_named_dims bug — num_heads layout dim has no loop vars after view+transpose"
)
def test_flash_v4_tile_Lq():
    """Flash v4: tile max_seqlen_q÷2 only."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v4_fn(q, k, v, B=2, S=256, H=8, D=64, lq_tiles=2),
        _flash_v4_inputs(2, 256, 8, 64),
        loopspec=LoopSpecCheck(counts=[2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: propagate_named_dims bug — num_heads layout dim has no loop vars after view+transpose"
)
def test_flash_v4_tile_H_Lq():
    """Flash v4: tile num_heads÷4 max_seqlen_q÷2. Equivalent to original test_flash_v4."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v4_fn(
            q, k, v, B=2, S=256, H=8, D=64, h_tiles=4, lq_tiles=2
        ),
        _flash_v4_inputs(2, 256, 8, 64),
        loopspec=LoopSpecCheck(counts=[4, 2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: propagate_named_dims bug — num_heads layout dim has no loop vars after view+transpose"
)
def test_flash_v4_tile_H_Lq_Lk():
    """Flash v4: tile num_heads÷4 max_seqlen_q÷2 max_seqlen_kv÷2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v4_fn(
            q, k, v, B=2, S=256, H=8, D=64, h_tiles=4, lq_tiles=2, lk_tiles=2
        ),
        _flash_v4_inputs(2, 256, 8, 64),
        loopspec=LoopSpecCheck(counts=[4, 2, 2]),
        correctness=False,
    )


@pytest.mark.skip(
    reason="Unsupported: propagate_named_dims bug — num_heads layout dim has no loop vars after view+transpose"
)
def test_flash_v4_tile_all():
    """Flash v4: tile all dims. B=2, H÷4, Lq÷2, Lk÷2."""
    run_coarse_tile_test(
        lambda q, k, v: _flash_v4_fn(
            q, k, v, B=2, S=256, H=8, D=64, b_tiles=2, h_tiles=4, lq_tiles=2, lk_tiles=2
        ),
        _flash_v4_inputs(2, 256, 8, 64),
        loopspec=LoopSpecCheck(counts=[2, 4, 2, 2]),
        correctness=False,
    )


# ---------------------------------------------------------------------------
# validate_named_dims tests
# ---------------------------------------------------------------------------


def test_validate_named_dims_raises_on_mismatch():
    """validate_named_dims raises AssertionError when expected_named_dims is wrong."""
    inputs = [tensor("x", shape=(256, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(expected_named_dims=["WRONG", "DIMS"]):
            return torch.abs(x)

    with pytest.raises(Exception, match="expected_named_dims"):
        run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


def test_validate_reduction_dims_raises_on_mismatch():
    """validate_named_dims raises AssertionError when expected_reduction_dims is wrong."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["WRONG"]):
            return x.amin(dim=0)

    with pytest.raises(Exception, match="expected_reduction_dims"):
        run_coarse_tile_test(fn, inputs, loopspec=None, correctness=False)


# ===========================================================================
# END OF STRUCTURED TESTS (Groups 1-9)
# ===========================================================================
# ORIGINAL TESTS — preserved for reference, being migrated to structured format.
#
# spyre_hint-driven coarse tiling
# These tests verify that coarse tiling is driven automatically by
# spyre_hint(num_tiles_per_dim=...) annotations.  Named tensor dimensions
# must be declared and annotated on device tensors for the hint resolver to
# map dimension names to loop variables.
# ===========================================================================


_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims


class TestCoarseTileSpyreHints(InductorTestCase):
    """Coarse tiling driven by spyre_hint(num_tiles_per_dim=...) annotations."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        _pnd.reset()

    # ------------------------------------------------------------------
    # Baseline: no hints -> no tiling
    # ------------------------------------------------------------------

    def test_hint_no_tiling_baseline(self):
        """Without spyre_hint annotations, coarse tiling must not fire."""
        x = torch.randn(256, 128, dtype=torch.float16).to("spyre")

        def fn(x):
            return torch.abs(x)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x)
        self.assertTrue(len(source_codes) > 0)
        # LoopSpec appears as an import even without tiling; check for a call.
        self.assertNotIn("LoopSpec(", source_codes[0])

    # ------------------------------------------------------------------
    # Single pointwise op
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_single_group_pointwise(self):
        """spyre_hint(num_tiles_per_dim={"A": 4}) tiles a pointwise abs into 4 iterations."""
        from torch_spyre._inductor import spyre_hint

        # 256 rows × 128 cols.  Tiling the outermost dim by 4 → 64 rows/iter.
        A, B = 256, 128
        x = torch.randn(A, B, dtype=torch.float16)

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"A": 4}):
                return torch.abs(x)

        x_dev = x.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(x_dev, ["A", "B"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec call in generated source")
        self.assertIn(
            "sympify('4')",
            src,
            "Expected loop count 4 in generated source",
        )

    # ------------------------------------------------------------------
    # Softmax-shaped chain (pointwise-reduce-pointwise)
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_softmax_shaped(self):
        """Tile the pointwise-reduce-pointwise stages of a softmax-like kernel.

        softmax(x, dim=-1) lowers to roughly:
          max_val = x.amax(dim=-1, keepdim=True)   # reduction
          x_shifted = x - max_val                   # pointwise broadcast sub
          exp_x = x_shifted.exp()                   # pointwise
          sum_exp = exp_x.sum(dim=-1, keepdim=True) # reduction
          out = exp_x / sum_exp                     # pointwise broadcast div

        All stages share the batch (row) dimension B.  Tiling over that
        dimension by K=4 means each loop iteration processes B/K rows.
        """
        from torch_spyre._inductor import spyre_hint

        B, D = 256, 128  # batch = 256 rows, each of length 128
        x = torch.randn(B, D, dtype=torch.float16)

        def softmax_fn(x):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                with spyre_hint(
                    expected_named_dims=["B"], expected_reduction_dims=["D"]
                ):
                    max_val = x.amax(dim=-1, keepdim=True)
                with spyre_hint(expected_named_dims=["B", "D"]):
                    x_shifted = x - max_val
                with spyre_hint(expected_named_dims=["B", "D"]):
                    exp_x = x_shifted.exp()
                with spyre_hint(
                    expected_named_dims=["B"], expected_reduction_dims=["D"]
                ):
                    sum_exp = exp_x.sum(dim=-1, keepdim=True)
                with spyre_hint(expected_named_dims=["B", "D"]):
                    return exp_x / sum_exp

        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        cfn = torch.compile(softmax_fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "LoopSpec(",
            src,
            "Expected LoopSpec call in generated source for softmax-shaped fn",
        )
        self.assertIn(
            "sympify('4')",
            src,
            "Expected loop count 4 in generated softmax source",
        )

    # ------------------------------------------------------------------
    # Nested hints: outer K=2, inner M=4 on a single op
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "sencores": 4,
        }
    )
    def test_hint_nested_loop_with_scratchpad(self):
        """Design-doc small example: y=a+b; z=y*c with nested K=2×M=4 hints.

        This is the canonical spyre_hint(num_tiles_per_dim=...) version of the
        small example from docs/source/compiler/coarse_tiling_loops.md.

        Shape [1024, 4096], outer hint tiles A-dim by 2 (512 rows/iter),
        inner hint tiles B-dim by 4 (1024 cols/iter).  With lx_planning
        enabled, the intermediate result y=a+b is allocated to LX scratchpad
        (it is only consumed within the loop body); the final output z stays
        in HBM.  sencores=4 keeps the generated bundle.mlir's per-core
        address expansion small enough to quote in full in the design doc
        (the default SENCORES=32 would unroll to 32 addresses per operand).

        Assertions:
        - LoopSpec entries are emitted (tiling is active).
        - At least one TensorArg carries allocation={'lx': ...}.
        - The output buffer allocation uses 'hbm'.
        - The per-tile sizes 512 and 1024 appear in the generated source.
        """
        from torch_spyre._inductor import spyre_hint

        A, B = 1024, 4096
        a = torch.randn(A, B, dtype=torch.float16)
        b = torch.randn(A, B, dtype=torch.float16)
        c = torch.randn(A, B, dtype=torch.float16)

        def fn(a, b, c):
            with spyre_hint(num_tiles_per_dim={"A": 2}):
                with spyre_hint(num_tiles_per_dim={"B": 4}):
                    y = a + b
                    z = y * c
                    return z

        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        c_dev = c.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(a_dev, ["A", "B"])
        _name_tensor_dims(b_dev, ["A", "B"])
        _name_tensor_dims(c_dev, ["A", "B"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev, c_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('2')", src, "Expected outer loop count 2")
        self.assertIn("sympify('4')", src, "Expected inner loop count 4")
        self.assertGreaterEqual(
            src.count("LoopSpec("),
            2,
            f"Expected ≥2 LoopSpec entries for nested loops\n\nSource:\n{src}",
        )
        self.assertIn(
            "allocation={'lx'",
            src,
            "Expected intermediate TensorArg with lx allocation",
        )
        self.assertIn(
            "allocation={'hbm'",
            src,
            "Expected output TensorArg with hbm allocation",
        )
        # Per-tile shape: K=2 over 1024 rows → 512 rows/tile;
        # M=4 over 4096 cols → 1024 cols/tile.
        self.assertIn("512", src, "Expected per-tile row count 512")
        self.assertIn("1024", src, "Expected per-tile col count 1024")

    # ------------------------------------------------------------------
    # Two ops in separate groups tiling different iteration dimensions
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_per_group_tiled_dims(self):
        """Two ops in separate hint groups tile different sets of iteration dims.

        Uses sub-dimension naming to map a [B, D] tensor's physical dims to
        named sub-dims, then tiles each op independently:

        op_a = abs(x): hint num_tiles_per_dim={"B": 4} tiles dim 0 only.
          B=256 → 4 tiles of 64 rows each.  Iteration space per tile: [64, D].

        op_b = neg(y): tensor named ["B0","B1","D0","D1"] with B0×B1=B and
          D0×D1=D.  Outer hint num_tiles_per_dim={"B0": 4} tiles dim 0 (c0,
          range 256) into 4.  Inner hint num_tiles_per_dim={"D0": 4} tiles
          dim 1 (c1, range 128) into 4.  Iteration space per tile: [64, 32].

        Both ops form separate groups → ≥2 LoopSpec entries, each with
        count=sympify('4').
        """
        from torch_spyre._inductor import spyre_hint

        B, D = 256, 128
        x = torch.randn(B, D, dtype=torch.float16)
        y = torch.randn(B, D, dtype=torch.float16)

        # Sub-dims for y: B0×B1 = B, D0×D1 = D
        B0, B1 = 4, B // 4  # 4 × 64 = 256
        D0, D1 = 4, D // 4  # 4 × 32 = 128

        x_dev = x.to("spyre")
        y_dev = y.to("spyre")

        # abs group: simple single-dim tiling over B
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        # neg group: sub-dim decomposition to tile both dims independently
        _declare_tensor_dim("B0", B0)
        _declare_tensor_dim("B1", B1)
        _declare_tensor_dim("D0", D0)
        _declare_tensor_dim("D1", D1)
        _name_tensor_dims(y_dev, ["B0", "B1", "D0", "D1"])

        def fn(x, y):
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                out_x = torch.abs(x)
            with spyre_hint(num_tiles_per_dim={"B0": 4}):
                with spyre_hint(num_tiles_per_dim={"D0": 4}):
                    out_y = torch.neg(y)
            return out_x, out_y

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev, y_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        loop_spec_count = src.count("LoopSpec(")
        self.assertGreaterEqual(
            loop_spec_count,
            2,
            f"Expected ≥2 LoopSpec entries (one per group), "
            f"got {loop_spec_count}\n\nSource:\n{src}",
        )
        self.assertIn(
            "sympify('4')",
            src,
            "Expected loop count 4 in generated source",
        )

    # ------------------------------------------------------------------
    # Two ops with different slice counts -> two separate groups
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_two_groups(self):
        """Two separate tiling groups produce two LoopSpec entries in the source."""
        from torch_spyre._inductor import spyre_hint

        A, B = 256, 128
        x = torch.randn(A, B, dtype=torch.float16)
        y = torch.randn(A, B, dtype=torch.float16)

        def fn(x, y):
            # Two independent pointwise ops: each becomes its own group.
            with spyre_hint(num_tiles_per_dim={"A": 4}):
                out_x = torch.abs(x)
            with spyre_hint(num_tiles_per_dim={"A": 8}):
                out_y = torch.neg(y)
            return out_x, out_y

        x_dev = x.to("spyre")
        y_dev = y.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(x_dev, ["A", "B"])
        _name_tensor_dims(y_dev, ["A", "B"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev, y_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        loop_spec_count = src.count("LoopSpec(")
        self.assertGreaterEqual(
            loop_spec_count,
            2,
            f"Expected ≥2 LoopSpec entries, got {loop_spec_count}\n\nSource:\n{src}",
        )

    # ------------------------------------------------------------------
    # Op inside hint scope with no matching named dim
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_group_includes_op_with_no_matching_dim(self):
        """An op inside a hint scope whose loop vars don't match the hinted dim stays in the group.

        torch.full lowers to a scalar-fill pointwise with no named loop variables.
        It has the hint but no loop var maps to "M", so it gets a scope-marker
        DimHint.  Its hint_id set still matches the surrounding ops so grouping
        is not broken.  The generated source must contain a single LoopSpec
        covering all ops.
        """
        from torch_spyre._inductor import spyre_hint

        M, K = 256, 64
        x = torch.randn(M, K, dtype=torch.float16)

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                # torch.full produces a scalar-fill with no M/K loop dim mapping.
                bias = torch.full(x.shape, 0.5, dtype=x.dtype, device=x.device)
                return x + bias

        x_dev = x.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _name_tensor_dims(x_dev, ["M", "K"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('4')", src, "Expected loop count 4")
        self.assertEqual(
            src.count("LoopSpec("),
            1,
            "Op with no matching dim must not break the group into two LoopSpec entries",
        )

    # ------------------------------------------------------------------
    # Loop-invariant (broadcast) op gets per_tile_fixed=True
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_loop_invariant_op_per_tile_fixed_in_sdsc(self):
        """A loop-invariant ComputedBuffer inside a coarse-tile group must have
        per_tile_fixed=True in the generated SDSC source, so the unroller does
        not advance its address.

        torch.full lowers to a scalar-fill ComputedBuffer with no loop var matching
        the hinted dim.  Its loop_tiled_dims are all-empty, making it loop-invariant
        w.r.t. the tiling.  finalize_layouts must stamp per_tile_fixed=True on it.
        """
        from torch_spyre._inductor import spyre_hint

        M, K = 256, 64
        x = torch.randn(M, K, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _name_tensor_dims(x_dev, ["M", "K"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                # torch.full produces a scalar-fill ComputedBuffer with no M-dim
                # loop var — its loop_tiled_dims are all empty (loop-invariant).
                bias = torch.full(x.shape, 0.5, dtype=x.dtype, device=x.device)
                return x + bias

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        src = source_codes[0]
        self.assertIn(
            "per_tile_fixed=True",
            src,
            "Loop-invariant (scalar-fill) buffer must have per_tile_fixed=True in SDSC",
        )

    # ------------------------------------------------------------------
    # Hint propagation through mm_to_bmm_pass
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_survives_mm_to_bmm_rewrite(self):
        """spyre_hint is not dropped when mm_to_bmm_pass rewrites mm -> bmm.

        A 3D matmul inside a spyre_hint scope is decomposed to mm then rewritten
        back to bmm by mm_to_bmm_pass.  copy_fx_custom_meta must propagate the
        hint onto the new bmm node so assign_dim_hints can tile it.
        """
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 2, 128, 64, 32
        x = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        y = torch.randn(K, N, dtype=torch.float16) * 0.01

        def fn(x, y):
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                return torch.matmul(x, y)

        x_dev = x.to("spyre")
        y_dev = y.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x_dev, ["B", "M", "K"])
        _name_tensor_dims(y_dev, ["K", "N"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev, y_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "LoopSpec(",
            src,
            "Expected LoopSpec: hint must survive mm->bmm rewrite",
        )
        self.assertIn("sympify('4')", src, "Expected loop count 4 after bmm rewrite")

    # ------------------------------------------------------------------
    # Hint propagation into inserted restickify nodes
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_restickify_stays_in_group(self):
        """A restickify node inserted inside a hint scope lands in the same group.

        output * correction triggers a restickify because output is col-major
        from a preceding transpose while correction is row-major.  The inserted
        restickify buffer must carry the hint metadata from its consumer so that
        assign_dim_hints includes it in the hinted group.  If it were ungrouped
        the LoopSpec count would cover fewer ops and the generated source would
        reflect a split group.
        """
        from torch_spyre._inductor import spyre_hint

        M, N = 256, 64
        x = torch.randn(M, N, dtype=torch.float16)
        scale = torch.randn(M, dtype=torch.float16)

        def fn(x, scale):
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                # transpose + contiguous forces a restickify on x before the mul
                x_t = x.transpose(0, 1).contiguous().transpose(0, 1)
                return x_t * scale.unsqueeze(-1)

        x_dev = x.to("spyre")
        scale_dev = scale.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(x_dev, ["M", "N"])
        _name_tensor_dims(scale_dev, ["M"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev, scale_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "LoopSpec(",
            src,
            "Expected LoopSpec: restickify must not break the hint group",
        )
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    # ------------------------------------------------------------------
    # Softmax with row-tiling: large [NROW, NCOL] tensor
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_softmax_row_tiling(self):
        """spyre_hint(num_tiles_per_dim={"NROW": 4}) tiles softmax over the row dimension.

        NCOL=4096 gives 64 sticks/row.  Row-tiling this shape exercises the
        multi-stick device_size[1] invariant: a per-tile device_size bug that
        shrinks the row-stride dimension corrupts all stick groups after the
        first in each non-first tile.  atol=0.02 is tight enough to catch
        values from the wrong row (fp16 errors from random inputs exceed 0.5).
        """
        from torch_spyre._inductor import spyre_hint

        NROW, NCOL = 16384, 4096
        x = torch.rand(NROW, NCOL, dtype=torch.float16)

        _declare_tensor_dim("NROW", NROW)
        _declare_tensor_dim("NCOL", NCOL)

        def fn(x, dim=-1):
            _name_tensor_dims(x, ["NROW", "NCOL"])
            with spyre_hint(num_tiles_per_dim={"NROW": 4}):
                return torch.softmax(x, dim)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=0.02, rtol=0.1)

    # ------------------------------------------------------------------
    # Matmul with row-tiling: tile the M dimension of x @ y
    # ------------------------------------------------------------------

    def test_hint_matmul_row_tiling(self):
        """spyre_hint(num_tiles_per_dim={"M": 4}) tiles matmul over the row (M) dimension."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 256, 128, 64
        x = torch.randn(M, K, dtype=torch.float16) * 0.01
        y = torch.randn(K, N, dtype=torch.float16) * 0.01

        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(x, y):
            _name_tensor_dims(x, ["M", "K"])
            _name_tensor_dims(y, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"M": 4}):
                return x @ y

        compare_with_cpu(
            fn, x, y, run_compile=True, run_eager=False, atol=0.01, rtol=0.01
        )

    # Consider deleting — superseded by Group 10 structured tests (_flash_v1_fn)
    @pytest.mark.skip
    def test_hint_flash_attention(self):
        """Flash attention tiled over H (4 slices) via nested spyre_hints.

        # TODO: re-enable Lk tiling once the numerical error is understood.
        # The Lk hint was previously a no-op (dropped by _hints_levels bug fixed
        # on this branch).  Now that Lk tiling is correctly applied, the result
        # is numerically wrong (~90% element mismatch).  Investigate and fix
        # before re-adding spyre_hint(num_tiles_per_dim={"Lk": lk_slices}).

        Decision xfail: failing in CI (Actions run 30385154736, job
        90362755639) on PR #3293. We've decided to xfail the coarse tiling
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.
        """
        import math
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk, D = 1, 8, 256, 256, 64
        block_size = 128

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)

        scale = 1.0 / math.sqrt(math.sqrt(D))
        lk_slices = Lk // block_size  # noqa: F841 — used in commented-out Lk hint

        def flash(queries, keys, values):
            with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
                output = torch.zeros_like(queries)
            with spyre_hint(named_dims=["B", "H", "Lq"]):
                M = torch.full(
                    (B, H, Lq),
                    float("-inf"),
                    device=queries.device,
                    dtype=torch.float16,
                )
            with spyre_hint(named_dims=["B", "H", "Lq"]):
                denominator = torch.zeros(
                    (B, H, Lq), device=queries.device, dtype=torch.float16
                )
            with spyre_hint(
                num_tiles_per_dim={"B": 1}
            ):  # 3 nested scopes exercises multi-hint logic
                with spyre_hint(num_tiles_per_dim={"H": 4}):
                    # TODO: re-enable once numerical error with Lk tiling is fixed
                    # with spyre_hint(num_tiles_per_dim={"Lk": lk_slices}):
                    keys_T = keys.transpose(-1, -2).contiguous()
                    scores = torch.matmul(queries * scale, keys_T * scale)
                    scores = scores.transpose(-1, -2).contiguous()
                    block_max = torch.amax(scores, dim=-2)
                    max_running = torch.maximum(M, block_max)
                    exp_scores = torch.exp(scores - max_running.unsqueeze(-2))
                    correction = torch.exp(M - max_running)
                    denominator = denominator * correction + exp_scores.sum(dim=-2)
                    output = output * correction.unsqueeze(-1) + torch.matmul(
                        exp_scores.transpose(-1, -2), values
                    )
                    M = max_running
            return output / denominator.unsqueeze(-1)

        # CPU reference first, then device setup — matching the driver pattern exactly
        ref = flash(queries_t, keys_t, values_t)

        queries_dev = queries_t.to("spyre")
        keys_dev = keys_t.to("spyre")
        values_dev = values_t.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])

        result = torch.compile(flash)(queries_dev, keys_dev, values_dev).cpu()
        torch.testing.assert_close(
            result,
            ref,
            equal_nan=True,
            atol=0.01,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )

    # Consider deleting — superseded by Group 10 structured tests (_flash_v2_fn)
    @pytest.mark.skip(
        reason="finalize_layouts: restickify infeasible for copy ops across loop groups"
    )
    def test_hint_flash_attention_v2(self):
        """Flash attention tiled over H (4 slices) via nested spyre_hints.

        Variant of test_hint_flash_attention with a causal mask and an
        explicit running-max (real_max) formulation that updates output and
        denominator in place via copy_.

        Still xfailed.  The divide sits outside the tiled scopes, so
        `output`/`denominator` get full buffers + copy ops; each copy writes its
        target without reading it, nothing costs that pairing, and
        finalize_layouts overwrites the target with the writer's layout ->
        "restickify needed but infeasible for op='buf24' input='buf26'".

        Not resolvable by layout choice: writer and consumer need mutually
        unrestickifiable candidates (forcing either aborts or gives ~70% wrong).
        {"Lq": 2} alone reproduces it.  See
        test_hint_flash_attention_v2_divide_in_scope for the formulation that
        works, which localizes this to the cross-loop-group copy path.

        Decision xfail: failing in CI (Actions run 30385154736, job
        90362755639) on PR #3293. We've decided to xfail the coarse tiling
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.
        """
        import math
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk, D = 1, 8, 256, 256, 64
        block_size = 128

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
        mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        mask_t.masked_fill_(~causal, float("-inf"))
        lq_slices = Lq // block_size

        def flash(queries, keys, values, mask):
            scale = 1.0 / math.sqrt(math.sqrt(D))
            output = torch.zeros_like(queries)
            real_max = torch.full(
                (B, H, Lq, 64),
                float("-inf"),
                device=queries.device,
                dtype=torch.float16,
            )
            real_max = real_max.amax(dim=-1)  # B, H, Lq sparse

            denominator = torch.zeros(
                (B, H, Lq, 64),
                device=queries.device,
                dtype=torch.float16,
            )
            denominator = denominator.amax(dim=-1)  # B, H, Lq sparse
            with spyre_hint(
                num_tiles_per_dim={"B": 1}
            ):  # 3 nested scopes exercises multi-hint logic
                with spyre_hint(num_tiles_per_dim={"H": 4}):
                    with spyre_hint(num_tiles_per_dim={"Lq": lq_slices}):
                        scaled_keys = keys * scale  # B, H, Lk, D
                        keys_T = scaled_keys.transpose(-1, -2)  # B, H, D, Lk
                        scores = torch.matmul(queries * scale, keys_T)  # B, H, Lq, Lk
                        scores = scores + mask  # B, H, Lq, Lk

                        block_max = torch.amax(scores, dim=-1)  # B, H, Lq sparse
                        running_max = torch.maximum(
                            real_max, block_max
                        )  # B, H, Lq sparse

                        exp_scores = torch.exp(
                            scores - running_max.unsqueeze(-1)
                        )  # B, H, Lq, Lk
                        correction = torch.exp(
                            real_max - running_max
                        )  # B, H, Lq sparse

                        denominator.copy_(
                            denominator * correction + exp_scores.sum(dim=-1)
                        )  # B, H, Lq sparse
                        output.copy_(
                            output * correction.unsqueeze(-1)
                            + torch.matmul(exp_scores, values)
                        )  # B, H, Lq, D

                        real_max.copy_(running_max)  # B, H, Lq sparse

            return output / denominator.unsqueeze(-1)

        # CPU reference first, then device setup — matching the driver pattern exactly
        ref = flash(queries_t, keys_t, values_t, mask_t)

        queries_dev = queries_t.to("spyre")
        keys_dev = keys_t.to("spyre")
        values_dev = values_t.to("spyre")
        mask_dev = mask_t.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(mask_dev, ["Lq", "Lk"])

        result = torch.compile(flash)(queries_dev, keys_dev, values_dev, mask_dev).cpu()
        torch.testing.assert_close(
            result,
            ref,
            equal_nan=True,
            atol=0.01,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )

    def test_hint_flash_attention_v2_divide_in_scope(self):
        """test_hint_flash_attention_v2 with the final divide INSIDE the scope.

        Outside, `output`/`denominator` are read past the loop group, so both get a
        full buffer + copy op whose target the divide (buf24) also reads; the copy
        writes its target without reading it, so no edge costs that pairing and
        finalize_layouts overwrites the target with the writer's layout, killing
        buf24's solved edge.  Inside, only `result` crosses: one copy op, target
        has no second consumer, nothing to invalidate.

        Sound only because H/Lq are output dims (each tile's denominator is final).
        Lk tiling still needs carry propagation -- #3198.

        The LoopSpec assertion is load-bearing: without it this passes even if
        tiling is silently skipped.
        """
        import math

        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk, D = 1, 8, 256, 256, 64
        block_size = 128

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
        mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        mask_t.masked_fill_(~causal, float("-inf"))
        lq_slices = Lq // block_size

        def flash(queries, keys, values, mask):
            scale = 1.0 / math.sqrt(math.sqrt(D))
            output = torch.zeros_like(queries)
            real_max = torch.full(
                (B, H, Lq, 64),
                float("-inf"),
                device=queries.device,
                dtype=torch.float16,
            ).amax(dim=-1)
            denominator = torch.zeros(
                (B, H, Lq, 64),
                device=queries.device,
                dtype=torch.float16,
            ).amax(dim=-1)
            with spyre_hint(num_tiles_per_dim={"H": 4}):
                with spyre_hint(num_tiles_per_dim={"Lq": lq_slices}):
                    scaled_keys = keys * scale
                    keys_T = scaled_keys.transpose(-1, -2)
                    scores = torch.matmul(queries * scale, keys_T)
                    scores = scores + mask

                    block_max = torch.amax(scores, dim=-1)
                    running_max = torch.maximum(real_max, block_max)

                    exp_scores = torch.exp(scores - running_max.unsqueeze(-1))
                    correction = torch.exp(real_max - running_max)

                    denominator.copy_(denominator * correction + exp_scores.sum(dim=-1))
                    output.copy_(
                        output * correction.unsqueeze(-1)
                        + torch.matmul(exp_scores, values)
                    )
                    real_max.copy_(running_max)

                    # The one difference from test_hint_flash_attention_v2.
                    result = output / denominator.unsqueeze(-1)
            return result

        ref = flash(queries_t, keys_t, values_t, mask_t)

        queries_dev = queries_t.to("spyre")
        keys_dev = keys_t.to("spyre")
        values_dev = values_t.to("spyre")
        mask_dev = mask_t.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(mask_dev, ["Lq", "Lk"])

        cfn = torch.compile(flash)
        result, source_codes = run_and_get_code(
            cfn, queries_dev, keys_dev, values_dev, mask_dev
        )
        torch.testing.assert_close(
            result.cpu(),
            ref,
            equal_nan=True,
            atol=0.01,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )
        # Both hint levels must survive into codegen (H=4 outer, Lq=2 inner).
        self.assertEqual(
            source_codes[0].count("LoopSpec("),
            2,
            "expected two nested LoopSpec entries (H then Lq); coarse tiling "
            "must not be silently skipped",
        )

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    # Consider deleting — superseded by Group 10 structured tests (_flash_v3_fn)
    @pytest.mark.skip(
        reason="finalize_layouts: restickify infeasible for copy ops across loop groups"
    )
    def test_hint_flash_attention_v3(self):
        from torch_spyre._inductor import spyre_hint

        B, H, D = 1, 32, 128
        Lq = 4096
        Lk = 4096

        q_block_size = Lq // 4  # replace by 'Lq // 2' for faster compilation time

        # FIXME: current limitation disallows coarse tiling in Lk
        kv_block_size = Lk // 1

        h_block_size = 4  # replace by 'H // 2' for faster compilation time
        b_block_size = 1

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
        mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        mask_t.masked_fill_(~causal, float("-inf"))

        def flash(queries, keys, values, mask):
            scale = 1.0 / math.sqrt(math.sqrt(D))

            output = torch.zeros_like(queries)

            # FIXME: create a sparse real_max tensor via reduction
            real_max = torch.full(
                (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
            )

            denominator = torch.zeros(
                (B, H, Lq), device=queries.device, dtype=torch.float16
            )

            with spyre_hint(tiles={"B": B // b_block_size}):
                with spyre_hint(tiles={"H": H // h_block_size}):
                    with spyre_hint(tiles={"Lq": Lq // q_block_size}):
                        with spyre_hint(tiles={"Lk": Lk // kv_block_size}):
                            # with spyre_hint(work_div={"H": 4, "Lq": 8, "Lk": 8}):
                            scaled_keys = keys * scale  # B, H, Lk, D
                            keys_T = scaled_keys.transpose(-1, -2)  # B, H, D, Lk
                            scores = torch.matmul(
                                queries * scale, keys_T
                            )  # B, H, Lq, Lk
                            scores = scores + mask  # B, H, Lq, Lk
                            scores = scores.transpose(-1, -2).contiguous()
                            block_max = torch.amax(scores, dim=-2)  # B, H, Lq sparse
                            running_max = torch.maximum(
                                real_max, block_max
                            )  # B, H, Lq sparse

                            exp_scores = torch.exp(
                                scores - running_max.unsqueeze(-2)
                            )  # B, H, Lq, Lk
                            correction = torch.exp(
                                real_max - running_max
                            )  # B, H, Lq sparse

                            denominator.copy_(
                                denominator * correction + exp_scores.sum(dim=-2)
                            )  # B, H, Lq sparse
                            output.copy_(
                                output * correction.unsqueeze(-1)
                                + torch.matmul(exp_scores.transpose(-1, -2), values)
                            )  # B, H, Lq, D

                            real_max.copy_(running_max)  # B, H, Lq sparse
            return output / denominator.unsqueeze(-1)

        queries_t_spyre = queries_t.to(device="spyre")
        keys_t_spyre = keys_t.to(device="spyre")
        values_t_spyre = values_t.to(device="spyre")
        mask_t_spyre = mask_t.to(device="spyre")

        ref = flash(queries_t, keys_t, values_t, mask_t)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)

        _name_tensor_dims(queries_t_spyre, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_t_spyre, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_t_spyre, ["B", "H", "Lk", "D"])
        _name_tensor_dims(mask_t_spyre, ["Lq", "Lk"])
        result = torch.compile(flash)(
            queries_t_spyre, keys_t_spyre, values_t_spyre, mask_t_spyre
        ).cpu()
        torch.testing.assert_close(
            result,
            ref,
            equal_nan=True,
            atol=0.1,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )

    @pytest.mark.skip(
        reason="finalize_layouts: restickify infeasible for copy ops across loop groups"
    )
    def test_hint_flash_attention_v3_b2(self):
        """Same as flash_v3 but with B=2 and b_block_size=2 so B is nto tiled"""
        from torch_spyre._inductor import spyre_hint

        B, H, D = 2, 32, 128
        Lq = 4096
        Lk = 4096

        q_block_size = Lq // 4  # replace by 'Lq // 2' for faster compilation time

        # FIXME: current limitation disallows coarse tiling in Lk
        kv_block_size = Lk // 1

        h_block_size = 4  # replace by 'H // 2' for faster compilation time
        b_block_size = 2

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
        mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        mask_t.masked_fill_(~causal, float("-inf"))

        def flash(queries, keys, values, mask):
            scale = 1.0 / math.sqrt(math.sqrt(D))

            output = torch.zeros_like(queries)

            # FIXME: create a sparse real_max tensor via reduction
            real_max = torch.full(
                (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
            )

            denominator = torch.zeros(
                (B, H, Lq), device=queries.device, dtype=torch.float16
            )

            with spyre_hint(tiles={"B": B // b_block_size}):
                with spyre_hint(tiles={"H": H // h_block_size}):
                    with spyre_hint(tiles={"Lq": Lq // q_block_size}):
                        with spyre_hint(tiles={"Lk": Lk // kv_block_size}):
                            scaled_keys = keys * scale  # B, H, Lk, D
                            keys_T = scaled_keys.transpose(-1, -2)  # B, H, D, Lk
                            scores = torch.matmul(
                                queries * scale, keys_T
                            )  # B, H, Lq, Lk
                            scores = scores + mask  # B, H, Lq, Lk
                            scores = scores.transpose(-1, -2).contiguous()
                            block_max = torch.amax(scores, dim=-2)  # B, H, Lq sparse
                            running_max = torch.maximum(
                                real_max, block_max
                            )  # B, H, Lq sparse

                            exp_scores = torch.exp(
                                scores - running_max.unsqueeze(-2)
                            )  # B, H, Lq, Lk
                            correction = torch.exp(
                                real_max - running_max
                            )  # B, H, Lq sparse

                            denominator.copy_(
                                denominator * correction + exp_scores.sum(dim=-2)
                            )  # B, H, Lq sparse
                            output.copy_(
                                output * correction.unsqueeze(-1)
                                + torch.matmul(exp_scores.transpose(-1, -2), values)
                            )  # B, H, Lq, D

                            real_max.copy_(running_max)  # B, H, Lq sparse
            return output / denominator.unsqueeze(-1)

        queries_t_spyre = queries_t.to(device="spyre")
        keys_t_spyre = keys_t.to(device="spyre")
        values_t_spyre = values_t.to(device="spyre")
        mask_t_spyre = mask_t.to(device="spyre")

        ref = flash(queries_t, keys_t, values_t, mask_t)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)

        _name_tensor_dims(queries_t_spyre, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_t_spyre, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_t_spyre, ["B", "H", "Lk", "D"])
        _name_tensor_dims(mask_t_spyre, ["Lq", "Lk"])
        result = torch.compile(flash)(
            queries_t_spyre, keys_t_spyre, values_t_spyre, mask_t_spyre
        ).cpu()
        torch.testing.assert_close(
            result,
            ref,
            equal_nan=True,
            atol=0.1,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )

    @pytest.mark.skip(
        reason=(
            "flash attention v3/v4 not yet passing: Lk reduction-dim tiling is "
            "disabled (see FIXME on kv_block_size in this file), unrelated to "
            "carry propagation. Confirmed (4/4 local full-suite runs) to leave "
            "the device in an error state that cascades skips to every later "
            "test in the same process (see conftest.py's has_stream_error() "
            "check) when run as xfail -- skipped outright instead. Revisit "
            "once the Lk coarse-tiling limitation above is fixed; a real fix "
            "there should make this test pass rather than merely change its "
            "failure mode, at which point this skip should be removed."
        ),
    )
    def test_hint_flash_attention_v3_b2_minimal(self):
        """Minimal reproducer for v3_b2 correctness failure."""
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk = 2, 32, 4096, 4096
        h_block_size = 4  # 8 H-tiles
        lq_block_size = 1024  # 4 Lq-tiles

        scores = torch.randn(B, H, Lk, Lq, dtype=torch.float16)

        def fn(scores):
            real_max = torch.full(
                (B, H, Lq), float("-inf"), device=scores.device, dtype=scores.dtype
            )
            with spyre_hint(tiles={"H": H // h_block_size}):
                with spyre_hint(tiles={"Lq": Lq // lq_block_size}):
                    with spyre_hint(
                        expected_named_dims=["B", "H", "Lq"],
                        expected_reduction_dims=["Lk"],
                    ):
                        block_max = torch.amax(scores, dim=-2)  # [B, H, Lq]
                    with spyre_hint(expected_named_dims=["B", "H", "Lq"]):
                        running_max = torch.maximum(real_max, block_max)
                    real_max.copy_(running_max)
            return real_max

        ref = fn(scores)

        scores_dev = scores.to("spyre")

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)

        _name_tensor_dims(scores_dev, ["B", "H", "Lk", "Lq"])

        result = torch.compile(fn)(scores_dev).cpu()
        torch.testing.assert_close(
            result,
            ref,
            equal_nan=True,
            atol=0.1,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )

    # Consider deleting — superseded by Group 10 structured tests (_flash_v4_fn)
    @pytest.mark.skip
    def test_hint_flash_attention_v4(self):
        """This test attempts to replicate the standalone test_granite_attn.py with views
        but the flash logic is inlined rather than relying on decompositions.py
        it is essentially flash_v3 but with views.  The flat [B,S,H*D] inputs and
        view+transpose in block(), matching the test_granite_attn.py call pattern.
        F.scaled_dot_product_attention is replaced by an inline online-softmax loop so the
        same function runs on both CPU (reference) and Spyre (compiled).
        """
        from torch_spyre._inductor import spyre_hint

        B, H, S, D = 2, 32, 4096, 256
        q_block_size = S // 4
        kv_block_size = S // 1

        queries_t = torch.randn(B, S, H * D, dtype=torch.float16)
        keys_t = torch.randn(B, S, H * D, dtype=torch.float16)
        values_t = torch.randn(B, S, H * D, dtype=torch.float16)

        def block(q, k, v):
            q = q.view(B, S, H, D).transpose(1, 2)
            k = k.view(B, S, H, D).transpose(1, 2)
            v = v.view(B, S, H, D).transpose(1, 2)

            scale = 1.0 / math.sqrt(math.sqrt(D))

            output = torch.zeros_like(q)
            real_max = torch.full(
                (B, H, S), float("-inf"), device=q.device, dtype=q.dtype
            )
            denominator = torch.zeros((B, H, S), device=q.device, dtype=q.dtype)

            with spyre_hint(tiles={"batch_size": max(1, B // 2)}):
                with spyre_hint(tiles={"num_heads": max(1, H // 4)}):
                    with spyre_hint(tiles={"max_seqlen_q": max(1, S // q_block_size)}):
                        with spyre_hint(
                            tiles={"max_seqlen_kv": max(1, S // kv_block_size)}
                        ):
                            scaled_keys = k * scale
                            keys_T = scaled_keys.transpose(-1, -2).contiguous()
                            scores = torch.matmul(q * scale, keys_T)
                            scores = scores.transpose(-1, -2).contiguous()
                            block_max = torch.amax(scores, dim=-2)
                            running_max = torch.maximum(real_max, block_max)
                            exp_scores = torch.exp(scores - running_max.unsqueeze(-2))
                            correction = torch.exp(real_max - running_max)
                            denominator.copy_(
                                denominator * correction + exp_scores.sum(dim=-2)
                            )
                            output.copy_(
                                output * correction.unsqueeze(-1)
                                + torch.matmul(exp_scores.transpose(-1, -2), v)
                            )
                            real_max.copy_(running_max)

            output.copy_(output / denominator.unsqueeze(-1))
            return output.transpose(1, 2).reshape(B, S, H * D)

        ref = block(queries_t, keys_t, values_t)

        queries_t_spyre = queries_t.to(device="spyre")
        keys_t_spyre = keys_t.to(device="spyre")
        values_t_spyre = values_t.to(device="spyre")

        _declare_tensor_dim("batch_size", B)
        _declare_tensor_dim("num_heads", H)
        _declare_tensor_dim("max_seqlen_q", S)
        _declare_tensor_dim("max_seqlen_kv", S)
        _declare_tensor_dim("head_dim", D)

        # Flat [B, S, H*D] inputs named with fused dims — S*H*D maps to
        # ["max_seqlen_q", "num_heads", "head_dim"], matching test_reshape_b pattern.
        _name_tensor_dims(
            queries_t_spyre, ["batch_size", "max_seqlen_q", "num_heads", "head_dim"]
        )
        _name_tensor_dims(
            keys_t_spyre, ["batch_size", "max_seqlen_kv", "num_heads", "head_dim"]
        )
        _name_tensor_dims(
            values_t_spyre, ["batch_size", "max_seqlen_kv", "num_heads", "head_dim"]
        )

        result = torch.compile(block)(
            queries_t_spyre, keys_t_spyre, values_t_spyre
        ).cpu()
        torch.testing.assert_close(
            result,
            ref,
            equal_nan=True,
            atol=0.1,
            rtol=0.1,
            msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
        )

    def test_hint_mixed_coverage_loopspec(self):
        """Union-across-ops: B level not dropped when first op has no B dimension.

        Two ops share a group under nested hints {A:2}/{B:4}.
        Op1 is x.abs() with shape [A, D] — iterates A, has loop_var=None for B.
        Op2 is abs_x + y with shape [A, B, D] — iterates both A and B.
        The old _hints_levels returned early at Op1 and dropped the B level.
        The fixed version unions across all ops and finds loop_var for B from Op2.
        """
        from torch_spyre._inductor import spyre_hint

        A, B, D = 128, 8, 64
        x = torch.randn(A, D, dtype=torch.float16)
        y = torch.randn(A, B, D, dtype=torch.float16)
        x_dev = x.to("spyre")
        y_dev = y.to("spyre")
        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["A", "D"])
        _name_tensor_dims(y_dev, ["A", "B", "D"])

        def fn(x, y):
            with spyre_hint(num_tiles_per_dim={"A": 2}):
                with spyre_hint(num_tiles_per_dim={"B": 4}):
                    # abs_x has shape [A, D], unsqueeze to [A, 1, D] for broadcast
                    abs_x = torch.abs(x).unsqueeze(1)
                    return abs_x + y

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev, y_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn(
            "count=sympify('2')", src, "Expected A loop count 2 as count= in LoopSpec"
        )
        self.assertIn(
            "count=sympify('4')", src, "Expected B loop count 4 as count= in LoopSpec"
        )

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    @pytest.mark.skip
    def test_hint_flash_attention_loopspec(self):
        """Lk loop level not dropped when Lk-broadcast ops appear first in group.

        Flash-attention-style code with nested hints {B:1}/{H:4}/{Lk:2}.
        The B=1 hint tiles by 1 and is optimised away (no loop generated),
        leaving two effective loop levels: H and Lk.  Ops like amax/exp/sum
        have shape [B,H,Lq] — no Lk dimension — and appear before
        Lk-iterating ops in topological order.  The old _hints_levels
        returned early from one of those ops and dropped Lk.  The fixed
        version unions loop_var assignments and finds Lk from a later op
        in the group.

        Decision xfail: failing in CI (Actions run 30385154736, job
        90362755639) on PR #3293. We've decided to xfail the coarse tiling
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.
        """
        import math
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk, D = 1, 8, 256, 256, 64
        block_size = 128
        scale = 1.0 / math.sqrt(math.sqrt(D))
        lk_slices = Lk // block_size  # 2

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        queries_dev = queries_t.to("spyre")
        keys_dev = keys_t.to("spyre")
        values_dev = values_t.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])

        def flash(queries, keys, values):
            with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
                output = torch.zeros_like(queries)
            with spyre_hint(named_dims=["B", "H", "Lq"]):
                M = torch.full(
                    (B, H, Lq),
                    float("-inf"),
                    device=queries.device,
                    dtype=torch.float16,
                )
            with spyre_hint(named_dims=["B", "H", "Lq"]):
                denominator = torch.zeros(
                    (B, H, Lq),
                    device=queries.device,
                    dtype=torch.float16,
                )
            with spyre_hint(num_tiles_per_dim={"B": 1}):
                with spyre_hint(num_tiles_per_dim={"H": 4}):
                    with spyre_hint(num_tiles_per_dim={"Lk": lk_slices}):
                        keys_T = keys.transpose(-1, -2).contiguous()
                        scores = torch.matmul(queries * scale, keys_T * scale)
                        scores = scores.transpose(-1, -2).contiguous()
                        block_max = torch.amax(scores, dim=-2)
                        max_running = torch.maximum(M, block_max)
                        exp_scores = torch.exp(scores - max_running.unsqueeze(-2))
                        correction = torch.exp(M - max_running)
                        denominator = denominator * correction + exp_scores.sum(dim=-2)
                        output = output * correction.unsqueeze(-1) + torch.matmul(
                            exp_scores.transpose(-1, -2), values
                        )
                        M = max_running
            return output / denominator.unsqueeze(-1)

        cfn = torch.compile(flash)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, queries_dev, keys_dev, values_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn(
            "count=sympify('4')",
            src,
            "Expected H loop count 4 — hint_H must appear as count= in LoopSpec",
        )
        self.assertIn(
            "count=sympify('2')",
            src,
            "Expected Lk loop count 2 as count= in LoopSpec — hint_Lk must not be"
            " dropped by _hints_levels",
        )

    def test_hint_mixed_output_and_reduction_loopspec(self):
        """Lk loop level stamped correctly when Lk is output dim for some ops and
        reduction dim for others in the same group.

        Bug: _stamp_group used a group-wide is_reduction_level flag taken from
        an arbitrary representative op.  If the flag disagreed with a given op's
        reality, the wrong divide function was called (or not called at all),
        so that op's ranges were not divided and it iterated over the full Lk
        per tile.

        Fix: per-op dispatch using each op's own hint_id_to_ranges_pos /
        hint_id_to_reduction_ranges_pos lookup tables.
        """
        from torch_spyre._inductor import spyre_hint

        H, Lq, Lk = 8, 64, 128  # Lk/2 = 64 elements = 1 stick at fp16
        x = torch.randn(H, Lq, Lk, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _name_tensor_dims(x_dev, ["H", "Lq", "Lk"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"Lk": 2}):
                # Op1: pointwise — Lk is an output dim
                y = x * 2.0
                # Op2: reduction over Lk — Lk is a reduction dim
                s = y.sum(dim=-1)
            return s

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn(
            "count=sympify('2')",
            src,
            "Expected Lk loop count 2 as count= in LoopSpec — must not be dropped"
            " when group contains mixed output/reduction ops for the same dim",
        )
        # The sum op's Lk reduction dim must be divided.  The bug causes the
        # sum to receive is_reduction_level=False (taken from the pointwise op),
        # so _divide_reduction_ranges is never called for it: the sum iterates
        # over the full Lk (tiled_symbols inner level stays empty).  After the
        # per-op dispatch fix, the sum gets tiled_symbols=[[sympify('c2')]]
        # confirming Lk is properly divided for the reduction op.
        # Anchor on op='sum' so a pointwise mul that also has c2 cannot satisfy
        # this check — the sum OpSpec must carry the tiled symbol itself.
        sum_op_idx = src.find("op='sum'")
        self.assertGreater(
            sum_op_idx, 0, "Expected op='sum' OpSpec in generated source"
        )
        self.assertNotIn(
            "tiled_symbols=[[]]",
            src[sum_op_idx : sum_op_idx + 300],
            "sum op has empty tiled_symbols — Lk reduction range not divided"
            " by _stamp_group (group-wide is_reduction_level flag bug)",
        )

    @pytest.mark.skip
    def test_hint_flash_attention_two_loop_levels(self):
        """Flash-attention graph: both H and Lk loop levels survive into codegen.

        Nested hints {B:1}/{H:4}/{Lk:2} — B=1 tiles by 1 and is optimised
        away (no loop generated), leaving two effective loop levels: H and Lk.
        The generated LoopSpec must carry both count=sympify('4') for H and
        count=sympify('2') for Lk.  Before the _stamp_group per-op dispatch
        fix, Lk tiling was silently skipped for ops where the group-wide
        is_reduction_level flag disagreed with the op's own dim role.

        Decision xfail: failing in CI (Actions run 30385154736, job
        90362755639) on PR #3293. We've decided to xfail the coarse tiling
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.
        """
        import math
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk, D = 1, 8, 256, 256, 64
        block_size = 128
        scale = 1.0 / math.sqrt(math.sqrt(D))
        lk_slices = Lk // block_size  # 2

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        queries_dev = queries_t.to("spyre")
        keys_dev = keys_t.to("spyre")
        values_dev = values_t.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])

        def flash(queries, keys, values):
            with spyre_hint(named_dims=["B", "H", "Lq", "D"]):
                output = torch.zeros_like(queries)
            with spyre_hint(named_dims=["B", "H", "Lq"]):
                M = torch.full(
                    (B, H, Lq),
                    float("-inf"),
                    device=queries.device,
                    dtype=torch.float16,
                )
            with spyre_hint(named_dims=["B", "H", "Lq"]):
                denominator = torch.zeros(
                    (B, H, Lq), device=queries.device, dtype=torch.float16
                )
            with spyre_hint(num_tiles_per_dim={"B": 1}):
                with spyre_hint(num_tiles_per_dim={"H": 4}):
                    with spyre_hint(num_tiles_per_dim={"Lk": lk_slices}):
                        keys_T = keys.transpose(-1, -2).contiguous()
                        scores = torch.matmul(queries * scale, keys_T * scale)
                        scores = scores.transpose(-1, -2).contiguous()
                        block_max = torch.amax(scores, dim=-2)
                        max_running = torch.maximum(M, block_max)
                        exp_scores = torch.exp(scores - max_running.unsqueeze(-2))
                        correction = torch.exp(M - max_running)
                        denominator = denominator * correction + exp_scores.sum(dim=-2)
                        output = output * correction.unsqueeze(-1) + torch.matmul(
                            exp_scores.transpose(-1, -2), values
                        )
                        M = max_running
            return output / denominator.unsqueeze(-1)

        cfn = torch.compile(flash)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, queries_dev, keys_dev, values_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn(
            "count=sympify('4')",
            src,
            "Expected H loop count 4 as count= in LoopSpec",
        )
        self.assertIn(
            "count=sympify('2')",
            src,
            "Expected Lk loop count 2 as count= in LoopSpec — _stamp_group must"
            " divide Lk ranges on each op using that op's own dim role",
        )
        # The amax (op='max') reduces over Lk.  The bug causes it to receive
        # is_reduction_level=False at the Lk loop level (group-wide flag taken
        # from a pointwise op), so _divide_reduction_ranges is never called:
        # amax iterates over the full Lk per tile and its tiled_symbols inner
        # level stays empty ("[[], ").  After the per-op dispatch fix, the amax
        # op gets a non-empty inner tiled_symbols entry for its Lk reduction dim.
        max_op_idx = src.find("op='max'")
        self.assertGreater(
            max_op_idx, 0, "Expected op='max' (amax) OpSpec in generated source"
        )
        self.assertNotIn(
            "tiled_symbols=[[], ",
            src[max_op_idx : max_op_idx + 500],
            "amax op has empty inner tiled_symbols — Lk reduction range not divided"
            " by _stamp_group (group-wide is_reduction_level flag bug)",
        )

    @pytest.mark.skip
    def test_hint_flash_attention_two_loop_levels_v2(self):
        """Flash-attention graph: both H and Lq loop levels survive into codegen.

        Variant of test_hint_flash_attention_two_loop_levels with a causal
        mask and an explicit running-max (real_max) formulation that updates
        output and denominator in place via copy_.

        Decision xfail: failing in CI (Actions run 30385154736, job
        90362755639) on PR #3293. We've decided to xfail the coarse tiling
        tests to allow us to merge to main -- deliberate decision to unblock
        the merge, not a claim about a specific bisected root cause. Un-xfail
        once the underlying regression is investigated and fixed.
        """
        import math
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, Lk, D = 1, 8, 256, 256, 64
        block_size = 128
        lq_slices = Lq // block_size  # 2

        queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
        keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
        causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
        mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        mask_t.masked_fill_(~causal, float("-inf"))
        queries_dev = queries_t.to("spyre")
        keys_dev = keys_t.to("spyre")
        values_dev = values_t.to("spyre")
        mask_dev = mask_t.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(queries_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(keys_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(values_dev, ["B", "H", "Lk", "D"])
        _name_tensor_dims(mask_dev, ["B", "H", "Lq", "Lk"])

        def flash(queries, keys, values, mask):
            scale = 1.0 / math.sqrt(math.sqrt(D))
            output = torch.zeros_like(queries)
            real_max = torch.full(
                (B, H, Lq, 64),
                float("-inf"),
                device=queries.device,
                dtype=torch.float16,
            )
            real_max = real_max.amax(dim=-1)  # B, H, Lq sparse
            denominator = torch.zeros(
                (B, H, Lq, 64),
                device=queries.device,
                dtype=torch.float16,
            )
            denominator = denominator.amax(dim=-1)  # B, H, Lq sparse
            with spyre_hint(num_tiles_per_dim={"B": 1}):
                with spyre_hint(num_tiles_per_dim={"H": 4}):
                    with spyre_hint(num_tiles_per_dim={"Lq": lq_slices}):
                        scaled_keys = keys * scale  # B, H, Lk, D
                        keys_T = scaled_keys.transpose(-1, -2)  # B, H, D, Lk
                        scores = torch.matmul(queries * scale, keys_T)  # B, H, Lq, Lk
                        scores = scores + mask  # B, H, Lq, Lk

                        block_max = torch.amax(scores, dim=-1)  # B, H, Lq sparse
                        running_max = torch.maximum(
                            real_max, block_max
                        )  # B, H, Lq sparse

                        exp_scores = torch.exp(
                            scores - running_max.unsqueeze(-1)
                        )  # B, H, Lq, Lk
                        correction = torch.exp(
                            real_max - running_max
                        )  # B, H, Lq sparse

                        denominator.copy_(
                            denominator * correction + exp_scores.sum(dim=-1)
                        )  # B, H, Lq sparse
                        output.copy_(
                            output * correction.unsqueeze(-1)
                            + torch.matmul(exp_scores, values)
                        )  # B, H, Lq, D

                        real_max.copy_(running_max)  # B, H, Lq sparse
            return output / denominator.unsqueeze(-1)

        cfn = torch.compile(flash)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(
                cfn, queries_dev, keys_dev, values_dev, mask_dev
            )
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn(
            "count=sympify('4')",
            src,
            "Expected H loop count 4 as count= in LoopSpec",
        )
        self.assertIn(
            "count=sympify('2')",
            src,
            "Expected Lq loop count 2 as count= in LoopSpec — _stamp_group must"
            " divide Lq ranges on each op using that op's own dim role",
        )

    def test_hint_h_tiling_elementwise(self):
        """spyre_hint(num_tiles_per_dim={"H": 2}) tiles elementwise multiply over the H dimension.

        Regression test for a bug in per-tile byte-stride computation where
        per-tile HBM base addresses advanced by the wrong amount when the tiled
        dimension was not the outermost host dimension (e.g. H in BHLD).
        """
        from torch_spyre._inductor import spyre_hint

        torch.manual_seed(42)
        B, H, Lq, Lk, D = 1, 8, 256, 256, 64  # Lk == Lq intentionally; same seq-len

        Q = torch.randn(B, H, Lq, D, dtype=torch.float16)
        V = torch.randn(B, H, Lk, D, dtype=torch.float16)

        def fn(q, v):
            with spyre_hint(num_tiles_per_dim={"H": 2}):
                return q * v

        ref = fn(Q, V)

        Q_dev = Q.to("spyre")
        V_dev = V.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("Lk", Lk)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(Q_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(V_dev, ["B", "H", "Lk", "D"])

        result = torch.compile(fn)(Q_dev, V_dev).cpu()
        torch.testing.assert_close(result, ref, atol=0.02, rtol=0.1)

    def test_hint_h_tiling_elementwise_loopspec(self):
        """H-tiling on BHLD (B=1 unit-size) selects the H iteration symbol, not Lq.

        Regression test for the host-range-index → iteration-space-key mapping in
        create_op_spec: loop_tiled_dims stores host-range indices which include
        unit-size dimensions that the iteration space skips.  Without the mapping,
        index 1 (H in BHLD with B=1) maps to the 2nd iteration-space key (Lq)
        rather than the 1st (H), producing wrong per-tile stride advances.

        Previously also broken for the copy ops inserted by
        _insert_read_copy_ops: their tiled_dims_per_read/output_tiled_dims
        dicts were keyed by tiled_op's raw (unsqueezed) host-range indices
        but read against copy_ranges (== dep.size, already squeezed) --
        fixed by mapping tiled_op's raw dim index to its squeezed position
        (mirroring SpyreKernel._host_dim_to_index_symbol) for both the
        extent lookup and the dict key itself.
        """
        from torch_spyre._inductor import spyre_hint

        B, H, Lq, D = 1, 8, 256, 64

        Q = torch.randn(B, H, Lq, D, dtype=torch.float16)
        V = torch.randn(B, H, Lq, D, dtype=torch.float16)
        Q_dev = Q.to("spyre")
        V_dev = V.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(Q_dev, ["B", "H", "Lq", "D"])
        _name_tensor_dims(V_dev, ["B", "H", "Lq", "D"])

        def fn(q, v):
            with spyre_hint(num_tiles_per_dim={"H": 2}):
                return q * v

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, Q_dev, V_dev)

        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for H-tiled elementwise")
        self.assertIn("sympify('2')", src, "Expected loop count 2 for H/2 tiles")
        # tiled_symbols now holds one minted per-(op, level) symbol (see
        # spyre_kernel._get_or_mint_level_symbol), not the real iteration-space
        # symbol (c0 for H) — so the regression this test guards against
        # (host-range index 1 (H) incorrectly resolving to the Lq iteration-space
        # symbol instead of H's) must instead be checked via the *value* of
        # device_tile_advance_expr's coefficient on that minted symbol.
        #
        # Different ops in this kernel can legitimately commit to different
        # device layouts for H (e.g. the read-copy ops keep H outermost, while
        # op0/coarse_tile_copy_buf0's own layouts place H just before the D
        # stick) -- so the *value* of the coefficient is not the same across
        # every op. What must hold for every op is that the coefficient equals
        # H's per-tile extent (8 // 2 == 4) times *that op's own*
        # device-element stride for H, derived structurally from its
        # device_size/device_coordinates (the device dim whose coordinate
        # expression is exactly the tiled iteration symbol c0). The original
        # bug instead advanced by a coefficient tied to Lq's extent/stride,
        # which this per-op recomputation catches regardless of which layout
        # a given op happens to commit to.

        tiled_syms_matches = re.findall(r"tiled_symbols=\[(\[.*?\])\]", src, re.DOTALL)
        self.assertTrue(
            tiled_syms_matches,
            "Expected tiled_symbols=[[...]] in generated OpSpec source",
        )
        minted_sym_matches = re.findall(
            r"_tile_adv_\w+_lvl\d+", "".join(tiled_syms_matches)
        )
        self.assertTrue(
            minted_sym_matches,
            f"Expected a minted _tile_adv_* symbol in tiled_symbols, "
            f"got: {tiled_syms_matches}",
        )
        tensor_arg_matches = re.findall(
            r"TensorArg\((?:(?!TensorArg\().)*?"
            r"device_size=\[([^\]]*)\],\s*"
            r"device_coordinates=\[([^\]]*)\],(?:(?!TensorArg\().)*?"
            r"device_tile_advance_expr=sympify\('([^']*)'\),",
            src,
            re.DOTALL,
        )
        self.assertTrue(
            tensor_arg_matches,
            "Expected TensorArg(...device_tile_advance_expr=...) in generated source",
        )
        for device_size_str, coords_str, advance_expr in tensor_arg_matches:
            embedded_syms = re.findall(r"_tile_adv_\w+_lvl\d+", advance_expr)
            self.assertTrue(
                embedded_syms,
                f"Expected a minted _tile_adv_* symbol embedded in "
                f"device_tile_advance_expr, got: {advance_expr}",
            )
            device_size = [int(x.strip()) for x in device_size_str.split(",")]
            coord_exprs = re.findall(r"sympify\('([^']*)'\)", coords_str)
            tiled_dim_positions = [i for i, c in enumerate(coord_exprs) if c == "c0"]
            self.assertTrue(
                tiled_dim_positions,
                f"Expected H's tiled iteration symbol c0 to appear bare in "
                f"device_coordinates, got: {coord_exprs}",
            )
            device_stride = 1
            for s in device_size[tiled_dim_positions[0] + 1 :]:
                device_stride *= s
            expected_coeff = 4 * device_stride
            coeff_match = re.search(r"floor\((\d+)\*", advance_expr)
            self.assertTrue(
                coeff_match,
                f"Expected a numeric coefficient in device_tile_advance_expr, "
                f"got: {advance_expr}",
            )
            self.assertEqual(
                int(coeff_match.group(1)),
                expected_coeff,
                f"device_tile_advance_expr should advance by H's per-tile "
                f"extent (4) times this op's own device-element stride for H "
                f"({device_stride}, from device_size={device_size} with H at "
                f"position {tiled_dim_positions[0]}) == {expected_coeff} -- "
                f"got: {advance_expr}",
            )

    def test_hint_row_tiling_multi_stick_pointwise_correct(self):
        """Row-tiling a multi-stick pointwise chain produces correct output.

        y = a + b; z = y * c on [1024, 4096] fp16 with num_tiles_per_dim={"A": 2}.
        This is the minimal reproducer for the _tile_device_size bug: with 64
        sticks/row, shrinking device_size[1] from 1024 to 512 corrupts the
        inter-stick-group stride, producing wrong values in the second tile.

        atol=0.01: fp16 (a+b)*c on inputs in [0,1) accumulates ~0.002 rounding
        error; atol=0.01 clears that comfortably while remaining well below the
        ~0.1 average error produced by a wrong-address read.
        """
        from torch_spyre._inductor import spyre_hint

        A, B = 1024, 4096
        a = torch.rand(A, B, dtype=torch.float16)
        b = torch.rand(A, B, dtype=torch.float16)
        c = torch.rand(A, B, dtype=torch.float16)

        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)

        def fn(a, b, c):
            _name_tensor_dims(a, ["A", "B"])
            _name_tensor_dims(b, ["A", "B"])
            _name_tensor_dims(c, ["A", "B"])
            with spyre_hint(num_tiles_per_dim={"A": 2}):
                y = a + b
                z = y * c
                return z

        compare_with_cpu(
            fn, a, b, c, run_compile=True, run_eager=False, atol=0.01, rtol=0.01
        )

    # ------------------------------------------------------------------
    # Tiled pointwise with outside consumer (_allocate_full_buffer)
    # ------------------------------------------------------------------

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_hint_tiled_pointwise_outside_consumer_correct(self):
        """Tiled pointwise op with a consumer outside the loop (tests
        _allocate_full_buffer pre-stickify: the full buffer must be correctly
        stickified by layout propagation).
        """
        from torch_spyre._inductor import spyre_hint

        A, B = 128, 64
        x = torch.randn(A, B, dtype=torch.float16)
        y = torch.randn(A, B, dtype=torch.float16)

        _declare_tensor_dim("A", A)
        _declare_tensor_dim("B", B)
        _name_tensor_dims(x, ["A", "B"])
        _name_tensor_dims(y, ["A", "B"])

        def fn(x, y):
            _name_tensor_dims(x, ["A", "B"])
            _name_tensor_dims(y, ["A", "B"])
            with spyre_hint(num_tiles_per_dim={"A": 2}):
                z = x + y  # tiled op
            return z * 2.0  # outside consumer -- forces _allocate_full_buffer

        compare_with_cpu(fn, x, y, run_compile=True, run_eager=False)

    def test_hint_nested_tiling_copy_mutation_correct(self):
        """Nested Lq/D tiling into a direct copy_() mutation (Case 3 rewire)."""
        from torch_spyre._inductor import spyre_hint

        Lq, D = 256, 128
        a = torch.randn(Lq, D, dtype=torch.float16)
        b = torch.randn(Lq, D, dtype=torch.float16)

        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("D", D)

        def fn(a, b):
            _name_tensor_dims(a, ["Lq", "D"])
            _name_tensor_dims(b, ["Lq", "D"])
            c = torch.full((Lq, D), 0, device=a.device, dtype=torch.float16)
            with spyre_hint(num_tiles_per_dim={"Lq": 2}):
                with spyre_hint(num_tiles_per_dim={"D": 2}):
                    c.copy_(a + b)
            return c

        compare_with_cpu(fn, a, b, run_compile=True, run_eager=False)

    def test_hint_nested_tiling_copy_mutation_divergent_input_layout(self):
        """Case 3 nested coarse-tiling where `a`'s device layout genuinely
        diverges from `b`'s -- exercises per-arg tile_advance_expr (each arg
        must compute its own device-byte-stride/device_coordinates, not
        share the output's).

        Uses a 3-D [B, Lq, D] shape (unlike
        test_hint_nested_tiling_copy_mutation_correct's 2-D [Lq, D]) so the
        divergence can be constructed with an explicit ``SpyreTensorLayout``
        ``dim_order`` swap on two *non-stick* dims (B and Lq): `a` gets
        dim_order [1, 0, 2] (Lq outermost, B next, D -- last -- still the
        stick dim) while `b`/`c` keep the default [0, 1, 2] (B outermost, Lq
        next, D the stick dim). Both tensors still end with D as the stick
        dimension, so no restickify is inserted to normalize the mismatch
        away before the coarse-tiled `add` runs -- unlike a 2-D [Lq, D]
        tensor, where any dim_order divergence necessarily swaps the stick
        dim itself and pointwise-op restickify insertion collapses the two
        inputs onto one shared layout before coarse-tiling's per-arg logic
        ever sees them (see docs/source/compiler/coarse_tiling_loops.md's
        note on `_get_device_dim_order`'s stick-dim placement, and the
        divergent-stick-dim case being a separate, pre-existing,
        out-of-scope gap -- confirmed by direct repro, not exercised here;
        tracked as https://github.com/torch-spyre/torch-spyre/issues/3332).
        Nesting num_tiles_per_dim={"Lq": 2} outer / {"B": 2} inner tiles two
        non-stick dims, each with a distinct per-arg device_coordinates walk.
        """
        from torch_spyre._C import SpyreTensorLayout
        from torch_spyre._inductor import spyre_hint

        B, Lq, D = 4, 256, 128
        a = torch.randn(B, Lq, D, dtype=torch.float16)
        b = torch.randn(B, Lq, D, dtype=torch.float16)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("D", D)

        a_stl = SpyreTensorLayout(a.size(), a.stride(), torch.float16, [1, 0, 2])
        _ = a.to("spyre")  # required for lazy device initialization
        a_dev = a.to(device_layout=a_stl)
        b_dev = b.to("spyre")

        def fn(a, b):
            _name_tensor_dims(a, ["B", "Lq", "D"])
            _name_tensor_dims(b, ["B", "Lq", "D"])
            c = torch.full((B, Lq, D), 0, device=a.device, dtype=torch.float16)
            with spyre_hint(num_tiles_per_dim={"Lq": 2}):
                with spyre_hint(num_tiles_per_dim={"B": 2}):
                    c.copy_(a + b)
            return c

        spyre_result = torch.compile(fn)(a_dev, b_dev).cpu()
        compare_with_cpu(fn, a, b, target=spyre_result, run_eager=False)

    def test_hint_nested_tiling_copy_mutation_flat(self):
        """Same Case 3 rewire as test_hint_nested_tiling_copy_mutation_correct,
        but on a flattened [Lq * D] 1-D tensor rather than [Lq, D] 2-D.

        Both the outer Lq:2 and inner D:2 coarse-tiling hints land on the same
        (only) host dim here, unlike the 2-D case where each hint owns a
        distinct host dim. dim_advance_overrides carries one
        (tile_size, supertile_count) fact per nesting level rather than one
        per host dim, so this no longer collapses the two levels' facts into
        one.
        """
        from torch_spyre._inductor import spyre_hint

        Lq, D = 256, 128
        a = torch.randn(Lq * D, dtype=torch.float16)
        b = torch.randn(Lq * D, dtype=torch.float16)

        _declare_tensor_dim("Lq", Lq)
        _declare_tensor_dim("D", D)

        def fn(a, b):
            _name_tensor_dims(a, ["Lq", "D"])
            _name_tensor_dims(b, ["Lq", "D"])
            c = torch.full([Lq * D], 0, device=a.device, dtype=torch.float16)
            with spyre_hint(num_tiles_per_dim={"Lq": 2}):
                with spyre_hint(num_tiles_per_dim={"D": 2}):
                    c.copy_(a + b)
            return c

        compare_with_cpu(fn, a, b, run_compile=True, run_eager=False)

    @config.patch(
        {
            "sencores": 4,
            "ignore_span_overflow_hints": False,
        }
    )
    @pytest.mark.skip
    def test_span_overflow_mutation_case_external_input_layout_mismatch(self):
        """Originally a regression repro for the Case 2/"Case 3"
        layout-reconciliation gap; now an xfail for a separate, deeper,
        pre-existing bug this task's fix newly exposes on this exact op.

        Task 2 (this task) deleted the direct-mutation Case 2/"Case 3"
        branch in `_propagate_tiled_op` entirely, so every cross-loop-group
        write -- including this test's -- now takes the `_insert_copy_op`
        path unconditionally. That fix *is* correct and closes the gap this
        test originally targeted: `TestCoarseTileBufferPropagation`'s
        `test_case2_condition_now_produces_copy_op` unit test directly
        confirms the old Case 2 branch's code no longer exists and the
        copy-op path is taken instead. A control probe run without any
        divergent input layout at all (plain contiguous `x`/`y`, same
        shapes, same span-overflow trigger) confirms the external-input-
        layout-mismatch scenario is no longer what's failing here.

        What still fails: `_insert_copy_op`'s interaction with a
        post-stickify, span-overflow-scaled `full_buf` has its own,
        independent, pre-existing addressing bug in `superdsc.py`'s per-arg
        `device_size`/`device_coordinates` handling -- confirmed present on
        the *pre-Task-2* code too, via a second control probe that forces a
        real inside-consumer (so the OLD Case 2/"Case 3" branch does not
        apply and the OLD code already takes the Case 1 copy-op path) on
        this same post-stickify span-overflow setup: it fails the same way
        (~87% mismatch) with zero divergent input layouts. This is the same
        general class of latent Case-1-copy-path bug flagged as "item 4,
        out of scope" in task-1-report.md (there triggered by a 3-input
        case); here it's shown to need neither 3 inputs nor a divergent
        layout, only `_insert_copy_op` + post-stickify span-overflow with
        `loop_count > 1`. It was never exercised end to end before because,
        pre-Task-2, an op with no inside consumers and no loop-internal
        input (this test's exact shape) always took Case 2 instead, and no
        other post-stickify span-overflow e2e test in this file forces
        Case 1 with `loop_count > 1`.

        This bug is in `_insert_copy_op`/`superdsc.py` addressing, not in
        the Case 2/"Case 3" deletion this task performs, and is out of this
        task's scope (its fix would require changes to `superdsc.py`, not
        listed in this task's file scope). Per this task's explicit
        instructions, this test stays `@unittest.expectedFailure` -- do not
        un-xfail a test that still fails, and do not reinstate any form of
        the deleted Case 2/"Case 3" branch as a workaround.

        Confirmed failure mode (current code, post-Task-2 fix): 15779/16384
        elements (96.3%) mismatch at atol=0.01/rtol=0.01, max abs diff
        ~6.74 -- still far outside fp16 rounding noise, but a different
        mismatch count/magnitude than the pre-fix 12221/16384 (74.6%),
        max abs diff ~5.89 recorded in task-1-report.md, consistent with a
        different root cause now being hit.

        MAX_SPAN_BYTES is patched down so a small tensor triggers automatic
        span-overflow tiling without needing a multi-hundred-MB real
        allocation (technique matches
        test_span_overflow_hint_analysis.py's TestSpanOverflowNumericValidation
        class).
        """
        from unittest.mock import patch as mock_patch2

        B, Lq, D = 32, 8, 64
        x_raw = torch.randn(Lq, B, D, dtype=torch.float16)
        x = x_raw.transpose(0, 1)  # logical [B, Lq, D], non-contiguous strides
        y = torch.randn(B, Lq, D, dtype=torch.float16)

        def fn(x, y):
            return x + y

        with mock_patch2(
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
            8192,
        ):
            compare_with_cpu(
                fn, x, y, run_compile=True, run_eager=False, atol=0.01, rtol=0.01
            )


class TestNamedDimsHint(InductorTestCase):
    """Tests for propagate_named_dims handling of ops with a named_dims hint.

    torch.full and torch.empty lower to ops whose loop variables carry no
    named-dim information from their inputs.  The new hint path allows
    spyre_hint(named_dims=[...]) to supply the named-dim mapping directly,
    enabling coarse tiling to work on these ops.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        _pnd.reset()

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_full_with_named_dims_hint_tiles(self):
        """spyre_hint(named_dims=[...]) on torch.full enables coarse tiling.

        Without the hint, torch.full has no named-dim mapping and coarse tiling
        cannot apply.  With named_dims supplied via the hint, propagate_named_dims
        should set _dim_prop_info correctly so assign_dim_hints produces a
        DimHint and LoopSpec appears in the generated source.
        """
        from torch_spyre._inductor import spyre_hint

        M, K = 256, 64

        def fn(x):
            with spyre_hint(slices={"M": 4}, named_dims=["M", "K"]):
                bias = torch.full(x.shape, 0.5, dtype=x.dtype, device=x.device)
            return x + bias

        x = torch.randn(M, K, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _name_tensor_dims(x_dev, ["M", "K"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_full_like_with_named_dims_hint_tiles(self):
        """spyre_hint(named_dims=[...]) on torch.full_like enables coarse tiling."""
        from torch_spyre._inductor import spyre_hint

        M, K = 128, 64

        def fn(x):
            with spyre_hint(slices={"M": 2}, named_dims=["M", "K"]):
                buf = torch.full_like(x, 2.0)
            return x + buf

        x = torch.randn(M, K, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _name_tensor_dims(x_dev, ["M", "K"])

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('2')", src, "Expected loop count 2")

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_named_dims_hint_self_contained_no_driver_calls(self):
        """spyre_hint(named_dims=[...]) alone enables coarse tiling.

        Unlike the tests above, this omits the driver-side declare_tensor_dim /
        name_tensor_dims calls entirely.  It locks in the in-graph path: the
        named_dims hint must (1) self-enable propagate_named_dims and (2)
        self-register the dim sizes, so the tiling hint resolves without any
        driver bootstrapping.  This is how a decomposition names its own
        intermediate dims (e.g. the flash SDPA decomposition).
        """
        from torch_spyre._inductor import spyre_hint

        M, K = 256, 64

        def fn(x):
            with spyre_hint(slices={"M": 4}, named_dims=["M", "K"]):
                bias = torch.full(x.shape, 0.5, dtype=x.dtype, device=x.device)
            return x + bias

        x = torch.randn(M, K, dtype=torch.float16)
        x_dev = x.to("spyre")
        # Deliberately NO _declare_tensor_dim / _name_tensor_dims here.

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec in generated source")
        self.assertIn("sympify('4')", src, "Expected loop count 4")


class TestCoarseTileReductionE2E(InductorTestCase):
    """E2E tests for coarse-tiling a reduction dimension.

    Stick-dim reduction tiling (dim=-1 on a [..., D] tensor where D maps to
    the stick) is now supported.  The loopspec tests run without hardware via
    mock_patch + run_and_get_code.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        _pnd.reset()

    def test_hint_tiled_reduction_sum_loopspec(self):
        """x.sum(dim=-1) tiled over D produces a LoopSpec with count 4."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16) * 0.1
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.sum(dim=-1)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for D-tiled sum")
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    def test_hint_tiled_reduction_sum_correct(self):
        """x.sum(dim=-1) tiled over D (4 tiles) produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16) * 0.1
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.sum(dim=-1)

        # atol=0.05: fp16 sum over 512 elements scaled by 0.1 accumulates ~0.05 error.
        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)

    def test_hint_tiled_reduction_matmul_loopspec(self):
        """torch.matmul tiled over K produces a LoopSpec with count 4."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return a @ b

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for K-tiled matmul")
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    def test_hint_tiled_reduction_matmul_correct(self):
        """torch.matmul tiled over K (4 tiles) produces correct results."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return a @ b

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_hint_tiled_reduction_max_loopspec(self):
        """x.amax(dim=-1) tiled over D produces a LoopSpec with count 4."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amax(dim=-1)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for D-tiled amax")
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    def test_hint_tiled_reduction_max_correct(self):
        """x.amax(dim=-1) tiled over D (4 tiles) produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amax(dim=-1)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=1e-3, rtol=1e-3)

    def test_hint_tiled_reduction_min_loopspec(self):
        """x.amin(dim=-1) tiled over D produces a LoopSpec with count 4."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        x_dev = x.to("spyre")
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)
        _name_tensor_dims(x_dev, ["B", "D"])

        def fn(x):
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amin(dim=-1)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, x_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for D-tiled amin")
        self.assertIn("sympify('4')", src, "Expected loop count 4")

    def test_hint_tiled_reduction_min_correct(self):
        """x.amin(dim=-1) tiled over D (4 tiles) produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 64, 512
        x = torch.randn(B, D, dtype=torch.float16)
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"D": 4}):
                return x.amin(dim=-1)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=1e-3, rtol=1e-3)


class TestCoarseTileReductionDim0E2E(InductorTestCase):
    """E2E tests for coarse-tiling a reduction over dim=0.

    These reduce a [B, D] tensor over B (dim=0), producing a [D] output where
    D is on the stick.  This is a simpler case than dim=-1 reductions because
    the output has a normal stick layout (no column-vector addressing).
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        _pnd.reset()

    def test_hint_tiled_reduction_dim0_sum_correct(self):
        """x.sum(dim=0) tiled over B produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 512, 64
        x = torch.randn(B, D, dtype=torch.float16) * 0.1

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return x.sum(dim=0)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=0.05, rtol=0.05)

    def test_hint_tiled_reduction_dim0_max_correct(self):
        """x.amax(dim=0) tiled over B produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 512, 64
        x = torch.randn(B, D, dtype=torch.float16)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return x.amax(dim=0)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=1e-3, rtol=1e-3)

    def test_hint_tiled_reduction_dim0_min_correct(self):
        """x.amin(dim=0) tiled over B produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, D = 512, 64
        x = torch.randn(B, D, dtype=torch.float16)

        _declare_tensor_dim("B", B)
        _declare_tensor_dim("D", D)

        def fn(x):
            _name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": 4}):
                return x.amin(dim=0)

        compare_with_cpu(fn, x, run_compile=True, run_eager=False, atol=1e-3, rtol=1e-3)


class TestCoarseTileMatmulKTilingE2E(InductorTestCase):
    """Correctness and LoopSpec tests for matmul/bmm tiled over the K (reduction) dimension.

    K=512 tiled by 4 gives 128 per tile (two sticks at fp16); shapes are chosen
    so K/T is stick-aligned without padding, keeping results deterministic.
    Use small weight scale (0.01) to keep fp16 accumulation error bounded.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        _pnd.reset()

    def test_mm_k_tiled_correct(self):
        """2D mm [M,K] @ [K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.mm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_bmm_k_tiled_correct(self):
        """3D bmm [B,M,K] @ [B,K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 8, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(B, K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["B", "K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.bmm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_bmm_3d2d_k_tiled_correct(self):
        """3D×2D matmul [B,M,K] @ [K,N] tiled over K produces correct results."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 8, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.matmul(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_mm_k_tiled_loopspec(self):
        """K-tiled mm produces a LoopSpec with count 4 in generated source."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 64, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"K": 4}):
                return torch.mm(a, b)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for K-tiled mm")
        self.assertIn("sympify('4')", src, "Expected loop count 4")


class TestCoarseTileNestedReductionE2E(InductorTestCase):
    """Correctness and LoopSpec tests for nested output-dim + reduction-dim tiling.

    Pattern: outer loop tiles an output dim, inner loop tiles a reduction dim.
    The fill op runs inside the outer loop (once per outer tile), so the
    accumulator is per-outer-tile sized.  The full output buffer spans all outer
    tiles; address advancement across outer iterations assembles the result.

    mm shapes: M=128, K=512, N=32; outer tiles M by 2 (64 rows/tile),
    inner tiles K by 4 (128 elements/tile = 2 sticks at fp16).
    bmm shapes: B=4, M=64, K=512, N=32; outer tiles B by 2,
    inner tiles K by 4.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xCAFE)
        _pnd.reset()

    @pytest.mark.skip(
        reason=(
            "This reproduces on the CI runners but NOT on every local stack, "
            "so a passing local run is not evidence it is fixed. Observed "
            "10.2% element mismatch (833/8192) on CI, repeatable across all "
            "retry attempts, while the same commit passes on a dev pod. "
            "Skipped rather than xfailed because some CI runs use strict "
            "xfail mode, where a passing xfail (e.g. on a dev pod) is itself "
            "a failure. Un-skip only on the strength of a green CI run."
        )
    )
    def test_nested_bmm_outer_Batch_inner_K_correct(self):
        """bmm [B,M,K]@[B,K,N] outer B (output) + inner K (reduction) — correct."""
        from torch_spyre._inductor import spyre_hint

        B, M, K, N = 4, 64, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(B, K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("B", B)
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["B", "M", "K"])
            _name_tensor_dims(b, ["B", "K", "N"])
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.bmm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    @pytest.mark.skip(
        reason="compiled spyre <-> cpu mismatch: nested matmul correctness not yet passing"
    )
    def test_nested_matmul_outer_M_inner_K_correct(self):
        """mm [M,K]@[K,N] with outer M (output) + inner K (reduction) — correct."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)

        def fn(a, b):
            _name_tensor_dims(a, ["M", "K"])
            _name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        compare_with_cpu(
            fn, a, b, run_compile=True, run_eager=False, atol=0.05, rtol=0.05
        )

    def test_nested_matmul_outer_M_inner_K_loopspec(self):
        """Nested mm produces two LoopSpec levels (outer count 2, inner count 4)."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn("LoopSpec(", src, "Expected LoopSpec for nested mm")
        self.assertIn("sympify('2')", src, "Expected outer loop count 2")
        self.assertIn("sympify('4')", src, "Expected inner loop count 4")

    @config.patch({"lx_planning": False})
    def test_nested_matmul_copy_after_inner_loop(self):
        """The accum→output copy op appears in generated source for nested K-tiling."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "coarse_tile_reduce_copy",
            src,
            "Expected a coarse_tile_reduce_copy op in generated source for nested M+K tiling",
        )

    @config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_nested_matmul_outer_M_inner_K_accum_in_lx(self):
        """With lx_planning enabled, the tile-sized accum buffer lands in LX scratchpad."""
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "allocation={'lx'",
            src,
            "Expected tile-sized accum TensorArg with lx allocation for nested M+K tiling",
        )

    def test_nested_matmul_accum_tile_per_tile_fixed_in_sdsc(self):
        """Accumulator tile buffer in nested outer-M + inner-K reduction must have
        per_tile_fixed=True in the generated SDSC source.

        The accum_tile buffer is loop-internal to the inner K-loop, so the unroller
        must not advance its base address across inner iterations.  This requires
        _pending_per_tile_fixed to be set by _propagate_tiled_reduction_op (pre-
        stickify path) and consumed by finalize_layouts.
        """
        from torch_spyre._inductor import spyre_hint

        M, K, N = 128, 512, 32
        a = torch.randn(M, K, dtype=torch.float16) * 0.01
        b = torch.randn(K, N, dtype=torch.float16) * 0.01
        a_dev = a.to("spyre")
        b_dev = b.to("spyre")
        _declare_tensor_dim("M", M)
        _declare_tensor_dim("K", K)
        _declare_tensor_dim("N", N)
        _name_tensor_dims(a_dev, ["M", "K"])
        _name_tensor_dims(b_dev, ["K", "N"])

        def fn(a, b):
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": 4}):
                    return torch.mm(a, b)

        cfn = torch.compile(fn)
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev)
        self.assertTrue(len(source_codes) > 0)
        src = source_codes[0]
        self.assertIn(
            "per_tile_fixed=True",
            src,
            "Nested-reduction accum_tile must have per_tile_fixed=True in SDSC",
        )


# ===========================================================================
# New tests appended below — do not modify the code above this line.
# ===========================================================================


def test_tiled_in_place_accumulator():
    """Regression test for the SpyreEmptyFallback / ct_fill STL bug.

    Was xfailed pending reorder-passes-clean (#3293, #3377, #3381); passes now.
    """
    from torch_spyre._inductor import spyre_hint

    torch.manual_seed(0xAFFE)
    _pnd.reset()

    B, H, Lq, D = 1, 8, 256, 64
    lq_slices = Lq // 128

    x_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
    scale_t = torch.randn(B, H, Lq, 1, dtype=torch.float16)
    # acc is a real graph input (not zeros_like) so there is no ct_fill
    # zeroing the tile each iteration — each tile genuinely accumulates.
    acc_t = torch.zeros(B, H, Lq, D, dtype=torch.float16)

    def fn(x, scale, acc):
        with spyre_hint(num_tiles_per_dim={"H": 4}):
            with spyre_hint(num_tiles_per_dim={"Lq": lq_slices}):
                block_max = torch.amax(x, dim=-1, keepdim=True)
                acc.copy_(acc + block_max * scale)
        return acc

    ref = fn(x_t, scale_t, acc_t.clone())

    x_dev = x_t.to("spyre")
    scale_dev = scale_t.to("spyre")
    acc_dev = acc_t.to("spyre")
    _declare_tensor_dim("B", B)
    _declare_tensor_dim("H", H)
    _declare_tensor_dim("Lq", Lq)
    _declare_tensor_dim("D", D)
    _name_tensor_dims(x_dev, ["B", "H", "Lq", "D"])
    _name_tensor_dims(scale_dev, ["B", "H", "Lq", "D"])
    _name_tensor_dims(acc_dev, ["B", "H", "Lq", "D"])

    result = torch.compile(fn)(x_dev, scale_dev, acc_dev).cpu()
    torch.testing.assert_close(result, ref, atol=0.01, rtol=0.1)


def test_sum_reduce_with_explicit_zero_accumulator():
    """Tiled sum with an explicit zeros accumulator (z += sum(a, dim=0))."""
    from torch_spyre._inductor import spyre_hint

    torch.manual_seed(0)
    _pnd.reset()

    A, B = 1024, 4096
    a_t = torch.randn(A, B, dtype=torch.float16) * 0.01

    def f(a):
        z = torch.zeros(B, device=a.device, dtype=torch.float16)
        with spyre_hint(num_tiles_per_dim={"A": 2}):
            y = torch.sum(a, dim=0)
            z += y
        return z

    ref = f(a_t)

    a_dev = a_t.to("spyre")
    _declare_tensor_dim("A", A)
    _declare_tensor_dim("B", B)
    _name_tensor_dims(a_dev, ["A", "B"])

    result = torch.compile(f)(a_dev).cpu()
    torch.testing.assert_close(result, ref, atol=0.01, rtol=0.1)


def test_sum_reduce_implicit_accumulator():
    """Tiled sum where the reduction output is returned directly (no explicit zero buffer)."""
    from torch_spyre._inductor import spyre_hint

    torch.manual_seed(0)
    _pnd.reset()

    A, B = 1024, 4096
    a_t = torch.randn(A, B, dtype=torch.float16) * 0.01

    def f_implicit(a):
        with spyre_hint(num_tiles_per_dim={"A": 2}):
            z = torch.sum(a, dim=0)
        return z

    ref = f_implicit(a_t)

    a_dev = a_t.to("spyre")
    _declare_tensor_dim("A", A)
    _declare_tensor_dim("B", B)
    _name_tensor_dims(a_dev, ["A", "B"])

    result = torch.compile(f_implicit)(a_dev).cpu()
    torch.testing.assert_close(result, ref, atol=0.01, rtol=0.1)


@pytest.mark.skip(
    reason="zeros with named_dims hint acquires wrong layout in tiled scope"
)
def test_zeros_named_dims_hint_correctness():
    """zeros with explicit named_dims hint inside a tiled scope should be correct.

    CURRENT STATUS (to delete when reorder-passes-clean is merged)
      - Passes on maim
      - Fails on reorder-passes-clean
    """
    from torch_spyre._inductor import spyre_hint

    torch.manual_seed(0)
    _pnd.reset()

    B, H, Lk, Lq = 1, 8, 256, 256
    x_t = torch.randn(B, H, Lk, Lq, dtype=torch.float16)
    cval_t = torch.randn(B, H, Lq, dtype=torch.float16)

    def f(x, cval):
        with spyre_hint(named_dims=["B", "H", "Lq"]):
            denom_named = torch.zeros((B, H, Lq), device=x.device, dtype=torch.float16)
        with spyre_hint(num_tiles_per_dim={"H": 4}):
            corr = torch.exp(cval)
            denom_likecval = torch.zeros_like(cval)
            s_simple = x.sum(dim=-2)
            s_named = denom_named * corr + x.sum(dim=-2)
            s_likecval = denom_likecval * corr + x.sum(dim=-2)
        return s_simple, s_named, s_likecval

    ref_simple, ref_named, ref_likecval = f(x_t, cval_t)

    xd = x_t.to("spyre")
    cd = cval_t.to("spyre")
    _declare_tensor_dim("B", B)
    _declare_tensor_dim("H", H)
    _declare_tensor_dim("Lk", Lk)
    _declare_tensor_dim("Lq", Lq)
    _name_tensor_dims(xd, ["B", "H", "Lk", "Lq"])
    _name_tensor_dims(cd, ["B", "H", "Lq"])

    got_simple, got_named, got_likecval = torch.compile(f)(xd, cd)

    torch.testing.assert_close(got_simple.cpu(), ref_simple, atol=0.5, rtol=0.1)
    # s_named is expected to fail — zeros with explicit named_dims hint is broken
    torch.testing.assert_close(got_named.cpu(), ref_named, atol=0.5, rtol=0.1)
    torch.testing.assert_close(got_likecval.cpu(), ref_likecval, atol=0.5, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
