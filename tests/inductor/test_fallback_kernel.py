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

"""Regression tests for the three FallbackKernel output shapes.

`FallbackKernel.create` in upstream PyTorch produces three shapes
(see torch/_inductor/ir.py):

  shape 1 (single tensor)  -> MultiOutputLayout + 1 trailing MultiOutput
  shape 2 (tuple of N)     -> MultiOutputLayout + N trailing MultiOutputs
  shape 3 (void/in-place)  -> NoneLayout       + 0 trailing MultiOutputs

torch-spyre previously asserted "FallbackKernel must be followed by exactly
one MultiOutput" in two passes (propagate_layouts, work_division) and
unconditionally called `get_layout()` on every dependency in fusion's
non-intermediate counter. Shape 3 raised RuntimeError, shape 2 emitted
"unhandled node type MultiOutput" warnings with incomplete layout
propagation, and shape 3 separately tripped fusion via the NoneLayout
MutationOutput sentinels that void fallbacks register.

These tests exercise all three shapes end-to-end through `torch.compile(...,
backend="inductor")` on the Spyre device to guard against regressions.
"""

import unittest

import torch


DEVICE = "spyre"
DTYPE = torch.float16


# Use FRAGMENT (not DEF) and guard against re-defining schemas, so the module
# is safe to import more than once — the test harness re-imports test files
# under different module names during analysis + execution, and DEF +
# unguarded define()/impl() would trip both
#   "Only a single TORCH_LIBRARY can be used to register the namespace ..."
# and
#   "Tried to register an operator ... with the same name multiple times".
def _ns_has_op(ns: str, op: str) -> bool:
    return hasattr(getattr(torch.ops, ns, None), op)


# Shape 1: op(x) -> Tensor
_LIB_S1 = torch.library.Library("test_fk_s1", "FRAGMENT")
if not _ns_has_op("test_fk_s1", "scale_two"):
    _LIB_S1.define("scale_two(Tensor x) -> Tensor")
    _LIB_S1.impl("scale_two", lambda x: x * 2, dispatch_key="CompositeExplicitAutograd")
    _LIB_S1._register_fake("scale_two", lambda x: torch.empty_like(x))


# Shape 2: op(x) -> (Tensor, Tensor)
_LIB_S2 = torch.library.Library("test_fk_s2", "FRAGMENT")
if not _ns_has_op("test_fk_s2", "split_two"):
    _LIB_S2.define("split_two(Tensor x) -> (Tensor, Tensor)")
    _LIB_S2.impl(
        "split_two",
        lambda x: (x + 1.0, x - 1.0),
        dispatch_key="CompositeExplicitAutograd",
    )
    _LIB_S2._register_fake(
        "split_two", lambda x: (torch.empty_like(x), torch.empty_like(x))
    )


# Shape 3: op(x, out) -> ()  (void / in-place mutation)
_LIB_S3 = torch.library.Library("test_fk_s3", "FRAGMENT")
if not _ns_has_op("test_fk_s3", "inplace_add"):
    _LIB_S3.define("inplace_add(Tensor x, Tensor(a!) out) -> ()")

    def _inplace_add_impl(x, out):
        out.add_(x)

    _LIB_S3.impl(
        "inplace_add", _inplace_add_impl, dispatch_key="CompositeExplicitAutograd"
    )
    _LIB_S3._register_fake("inplace_add", lambda x, out: None)


_LIB_POOL = torch.library.Library("test_fk_pool", "FRAGMENT")
if not _ns_has_op("test_fk_pool", "norm"):
    _LIB_POOL.define("norm(Tensor x, Tensor residual) -> Tensor")
    _LIB_POOL.impl(
        "norm", lambda x, r: (x + r) * 0.5, dispatch_key="CompositeExplicitAutograd"
    )
    _LIB_POOL._register_fake("norm", lambda x, r: torch.empty_like(x))

if not _ns_has_op("test_fk_pool", "norm_inplace"):
    _LIB_POOL.define("norm_inplace(Tensor(a!) x, Tensor residual) -> ()")

    def _norm_inplace_impl(x, r):
        x.copy_((x + r) * 0.5)

    _LIB_POOL.impl(
        "norm_inplace", _norm_inplace_impl, dispatch_key="CompositeExplicitAutograd"
    )
    _LIB_POOL._register_fake("norm_inplace", lambda x, r: None)


class TestFallbackKernelShape1Single(unittest.TestCase):
    """Shape 1: op(...) -> Tensor.

    Lowered to MultiOutputLayout FallbackKernel + 1 trailing MultiOutput.
    Was the only shape the original passes handled correctly; included
    here so all three shapes are covered by one test file.
    """

    def test_single_tensor_return_compiles(self):
        def fn(x):
            return torch.ops.test_fk_s1.scale_two(x) + 1.0

        x = torch.ones(4, dtype=DTYPE, device=DEVICE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x).cpu()
        torch.testing.assert_close(out, torch.full((4,), 3.0, dtype=DTYPE))


class TestFallbackKernelShape2Tuple(unittest.TestCase):
    """Shape 2: op(...) -> (Tensor, ..., Tensor).

    Lowered to MultiOutputLayout FallbackKernel + N trailing MultiOutputs.
    The original `next(it)` pattern consumed the first MultiOutput and
    fell through `unhandled node type MultiOutput` warnings for the rest;
    layout propagation was silently incomplete.
    """

    def test_two_tensor_return_compiles(self):
        def fn(x):
            a, b = torch.ops.test_fk_s2.split_two(x)
            return a * b

        x = torch.full((4,), 2.0, dtype=DTYPE, device=DEVICE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x).cpu()
        # (x + 1) * (x - 1) = x^2 - 1 = 3.0
        torch.testing.assert_close(out, torch.full((4,), 3.0, dtype=DTYPE))


class TestFallbackKernelShape3Void(unittest.TestCase):
    """Shape 3: op(...) -> () (void / in-place mutation).

    Lowered to NoneLayout FallbackKernel + 0 MultiOutputs. Upstream
    additionally registers MutationOutput sentinel buffers (one per
    mutated arg) with NoneLayout — those slip past fusion's
    `isinstance(buf, FallbackKernel)` guard and previously crashed
    `_is_non_intermediate` via `get_layout()`.

    This op signature mirrors the vLLM
    `torch.ops.vllm.unified_attention_with_output(...)` contract that
    triggered the original bug report.
    """

    def test_void_inplace_compiles(self):
        def fn(x):
            out = torch.zeros_like(x)
            torch.ops.test_fk_s3.inplace_add(x, out)
            return out + 1.0

        x = torch.full((4,), 5.0, dtype=DTYPE, device=DEVICE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x).cpu()
        # zeros + x + 1 = 6.0
        torch.testing.assert_close(out, torch.full((4,), 6.0, dtype=DTYPE))


class TestFallbackKernelPoolResidentArg(unittest.TestCase):
    """FallbackKernel consuming an intermediate buffer keeps the correct dtype."""

    def test_fresh_output_arg_keeps_dtype(self):
        def fn(x):
            residual = x
            x = torch.ops.test_fk_pool.norm(x, residual)
            x = residual + x  # pool-eligible intermediate, read by fallback
            residual = x
            x = torch.ops.test_fk_pool.norm(x, residual)
            return residual + x

        x = torch.randn(16, 4096, dtype=DTYPE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x.to(DEVICE))
        self.assertEqual(out.dtype, DTYPE)
        torch.testing.assert_close(out.cpu(), fn(x), atol=0.1, rtol=0.1)

    def test_inplace_arg_keeps_dtype(self):
        def fn(x):
            residual = x.clone()
            torch.ops.test_fk_pool.norm_inplace(x, residual)  # x mutated
            x = residual + x  # pool-eligible intermediate, read by fallback
            residual = x.clone()
            torch.ops.test_fk_pool.norm_inplace(x, residual)  # x mutated
            return residual + x

        x = torch.randn(16, 4096, dtype=DTYPE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x.clone().to(DEVICE))
        self.assertEqual(out.dtype, DTYPE)
        torch.testing.assert_close(out.cpu(), fn(x.clone()), atol=0.1, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
