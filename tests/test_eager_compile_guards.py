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

# Owner(s): ["module: dynamo"]

"""Regression tests for dynamo cache/compile behavior around standalone-
compiled eager ops.

Two things are exercised here:

1. The dynamo cache budget for standalone-compiled eager ops. Compiled eager
   kernels used to share one dynamo cache line, and so one recompile budget:
   ``torch.compile`` on an ``OpOverload`` routes through ``wrap_inline``, whose
   ``inner`` is a single code object. Exhausting the shared budget made dynamo
   run an op eagerly, which re-dispatched into the Spyre kernel that called it
   and recursed (hf-adapters#402).

2. ``torch.compiler.nested_compile_region`` under the Spyre tensor-match guard
   patch. torch-spyre replaces ``GuardBuilder.TENSOR_MATCH`` with
   ``_spyre_TENSOR_MATCH`` (see ``torch_spyre/_monkey_patch.py``) to
   additionally guard on ``SpyreTensorLayout``. ``nested_compile_region``
   compiles a region once and reuses the same subgraph across repeated calls;
   on reuse, dynamo re-evaluates every guard on the sources touched inside the
   region by looking its type up in ``GUARD_VALUE_DISPATCH``. A guard type
   absent from that registry is a hard error:

       RuntimeError: subgraph_reuse: unsupported guard type '_spyre_TENSOR_MATCH'

   Because the patch is process-global (installed at ``import torch_spyre``),
   this broke *any* use of ``nested_compile_region`` in a torch-spyre process
   -- even on plain CPU tensors that never touch the device. The fix registers
   a layout-aware ``GuardCheckSpec`` for ``_spyre_TENSOR_MATCH`` in
   ``GUARD_VALUE_DISPATCH``.

Both are CPU-only; no Spyre device needed.
"""

import unittest

import torch
import torch._dynamo
import torch._dynamo as dynamo
import torch._ops

# Read the guard registry off the live module at call time rather than binding
# the name at import, so the assertions can never see a stale reference.
import torch._dynamo.guards as _dynamo_guards
from torch import nn
from torch._dynamo.guards import GuardBuilder
from torch.compiler import nested_compile_region

# Importing torch_spyre applies the dynamo config changes under test, and
# triggers the monkey-patch that replaces TENSOR_MATCH.
import torch_spyre  # noqa: F401
from torch_spyre.ops.eager import _guard_reentry, _op_frame

# NB: use plain unittest, NOT torch.testing._internal.common_utils.run_tests.
# run_tests() calls torch.manual_seed(), which fires torch-spyre's custom-device
# seed hook and eagerly initializes the Spyre VFIO device -- turning these
# CPU-only tests into device tests that fail when a card is busy. Plain
# unittest never seeds a device.


# Single-overload fixture op: torch.ops.torch_spyre_test.single_overload_echo
# is an OpOverloadPacket, matching the shape of every spyre::* custom op
# (e.g. quantize_weight_fp8_with_scale) -- unlike torch.ops.aten.add.Tensor,
# which is an explicit OpOverload. See test_single_overload_custom_op_gets_a_qualname.
@torch.library.custom_op("torch_spyre_test::single_overload_echo", mutates_args=())
def _single_overload_echo(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


class TestDynamoCacheLimits(unittest.TestCase):
    def test_accumulated_limit_is_raised_with_the_per_line_limit(self):
        # Checked first, so a default 256 caps the cache whatever the other says.
        config = torch._dynamo.config
        self.assertGreaterEqual(
            config.accumulated_recompile_limit, config.recompile_limit
        )
        # 1024 is the production value set in torch_spyre/__init__.py, asserted
        # here as an intentional regression anchor -- not a magic constant to
        # keep in sync by hand. If that value changes, update it here too.
        self.assertEqual(config.accumulated_recompile_limit, 1024)


class TestPerOpCacheLine(unittest.TestCase):
    def setUp(self):
        # Belt-and-suspenders: clear any cache entries left by import-time
        # side effects or earlier tests before this test contributes its own,
        # so cross-test cache pollution can't make the budget assertions flake.
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_each_op_gets_its_own_code_object(self):
        add = _op_frame(torch.ops.aten.add.Tensor)
        mul = _op_frame(torch.ops.aten.mul.Tensor)
        self.assertIsNot(add.__code__, mul.__code__)

    def test_frame_forwards_args_and_kwargs_to_the_op(self):
        add = _op_frame(torch.ops.aten.add.Tensor)
        x, y = torch.ones(3), torch.full((3,), 2.0)
        torch.testing.assert_close(add(x, y, alpha=2), x + 2 * y)

    def test_single_overload_custom_op_gets_a_qualname(self):
        # torch.ops.<ns>.<name> for a single-overload custom op (e.g. every
        # spyre::* op registered via torch.library.custom_op) is an
        # OpOverloadPacket, not an OpOverload -- unlike aten.add.Tensor above.
        # OpOverloadPacket has no .name(); attribute access falls through to
        # __getattr__, which tries to resolve "name" as an overload name and
        # raises AttributeError. Regression test for exactly that: stamping
        # __qualname__ via op.name() blew up on spyre.quantize_weight_fp8_with_scale
        # in hf-adapters CI with "has no overload name 'name'". str(op) works
        # on both OpOverload and OpOverloadPacket.
        op = torch.ops.torch_spyre_test.single_overload_echo
        self.assertIsInstance(op, torch._ops.OpOverloadPacket)
        frame = _op_frame(op)
        self.assertEqual(frame.__qualname__, f"_op_frame.<locals>.{op}")
        torch.testing.assert_close(frame(torch.ones(3)), torch.ones(3))

    def test_one_op_exhausting_its_budget_leaves_other_ops_compilable(self):
        graphs = []

        def counting_backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        limit = 4
        with torch._dynamo.config.patch(
            recompile_limit=1024, accumulated_recompile_limit=limit
        ):
            compiled_add = torch.compile(
                _op_frame(torch.ops.aten.add.Tensor),
                backend=counting_backend,
                dynamic=False,
            )
            compiled_mul = torch.compile(
                _op_frame(torch.ops.aten.mul.Tensor),
                backend=counting_backend,
                dynamic=False,
            )
            # One graph per shape, so this spends add's budget.
            for n in range(1, limit + 1):
                compiled_add(torch.randn(n), torch.randn(n))
            spent_on_add = len(graphs)
            for n in range(1, limit + 1):
                compiled_mul(torch.randn(n), torch.randn(n))
            spent_on_mul = len(graphs) - spent_on_add

        self.assertEqual(spent_on_add, limit)
        # Sharing one cache line, as before the fix, this would be 0.
        self.assertEqual(spent_on_mul, limit)


class TestReentryGuard(unittest.TestCase):
    def test_reentering_the_same_op_raises_instead_of_recursing(self):
        op = torch.ops.aten.add.Tensor
        with self.assertRaisesRegex(RuntimeError, "re-entered itself"):
            with _guard_reentry(op):
                with _guard_reentry(op):
                    pass

    def test_a_different_op_may_nest(self):
        with _guard_reentry(torch.ops.aten.add.Tensor):
            with _guard_reentry(torch.ops.aten.mul.Tensor):
                pass

    def test_the_op_is_released_after_an_exception(self):
        op = torch.ops.aten.add.Tensor
        with self.assertRaises(ValueError):
            with _guard_reentry(op):
                raise ValueError
        with _guard_reentry(op):  # must not still be marked in flight
            pass


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, h):
        return self.lin(h).relu()


def _region_block(block):
    # nested_compile_region cannot mark a bound method, so wrap it.
    def wrapper(*args, **kwargs):
        return block.forward(*args, **kwargs)

    return nested_compile_region(wrapper)


class TestNestedCompileRegionGuard(unittest.TestCase):
    def test_patch_is_installed(self):
        # Sanity: the Spyre guard patch is active in this process.
        self.assertEqual(GuardBuilder.TENSOR_MATCH.__name__, "_spyre_TENSOR_MATCH")

    def test_spyre_guard_registered_for_subgraph_reuse(self):
        # The fix: _spyre_TENSOR_MATCH must be dispatchable during subgraph
        # reuse. Guard.create_fn_name() reports create_fn.__name__, so the
        # registry key is exactly this string.
        self.assertIn("_spyre_TENSOR_MATCH", _dynamo_guards.GUARD_VALUE_DISPATCH)
        self.assertTrue(hasattr(GuardBuilder.TENSOR_MATCH, "guard_check_spec"))

    def test_region_block_reused_across_layers_cpu(self):
        # The load-bearing behavior: a region-wrapped block called N times
        # compiles the region once and reuses one subgraph, without raising
        # "subgraph_reuse: unsupported guard type '_spyre_TENSOR_MATCH'".
        dim = 16
        blocks = [_region_block(_Block(dim)) for _ in range(4)]

        def outer(h):
            for b in blocks:
                h = b(h)
            return h

        dynamo.reset()
        compiled = torch.compile(outer, dynamic=False, fullgraph=True)
        h = torch.randn(2, dim)
        out = compiled(h)  # used to raise InternalTorchDynamoError here
        self.assertEqual(out.shape, (2, dim))

        # The repeated region calls must lower to invoke_subgraph, sharing one
        # subgraph (>= 2 confirms reuse rather than inlining).
        gm, _ = dynamo.export(outer)(h)
        calls = [
            n
            for n in gm.graph.nodes
            if "invoke_subgraph" in str(getattr(n, "target", ""))
        ]
        self.assertGreaterEqual(
            len(calls),
            2,
            f"expected repeated invoke_subgraph calls, got {calls}",
        )


# NB: plain unittest, not run_tests() -- see the module docstring.
if __name__ == "__main__":
    unittest.main()
