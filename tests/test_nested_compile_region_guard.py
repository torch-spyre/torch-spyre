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

"""Regression test for torch.compiler.nested_compile_region under the Spyre
tensor-match guard patch.

torch-spyre replaces ``GuardBuilder.TENSOR_MATCH`` with ``_spyre_TENSOR_MATCH``
(see ``torch_spyre/_monkey_patch.py``) to additionally guard on
``SpyreTensorLayout``. ``torch.compiler.nested_compile_region`` compiles a
region once and reuses the same subgraph across repeated calls; on reuse,
Dynamo re-evaluates every guard on the sources touched inside the region by
looking its type up in ``GUARD_VALUE_DISPATCH``. A guard type absent from that
registry is a hard error:

    RuntimeError: subgraph_reuse: unsupported guard type '_spyre_TENSOR_MATCH'

Because the patch is process-global (installed at ``import torch_spyre``), this
broke *any* use of ``nested_compile_region`` in a torch-spyre process — even on
plain CPU tensors that never touch the device. The fix registers a
layout-aware ``GuardCheckSpec`` for ``_spyre_TENSOR_MATCH`` in
``GUARD_VALUE_DISPATCH``. This test is CPU-only and needs no Spyre device: it
exercises the exact reuse path that used to abort.
"""

import unittest

import torch
from torch import nn
from torch.compiler import nested_compile_region
import torch._dynamo as dynamo

# NB: use plain unittest, NOT torch.testing._internal.common_utils.run_tests.
# run_tests() calls torch.manual_seed(), which fires torch-spyre's custom-device
# seed hook and eagerly initializes the Spyre VFIO device — turning this
# CPU-only test into a device test that fails when a card is busy. Plain
# unittest never seeds a device.

# Read the guard registry off the live module at call time rather than binding
# the name at import, so the assertions can never see a stale reference.
import torch._dynamo.guards as _dynamo_guards
from torch._dynamo.guards import GuardBuilder

# Importing torch_spyre triggers the monkey-patch that replaces TENSOR_MATCH.
import torch_spyre  # noqa: F401


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


if __name__ == "__main__":
    unittest.main()
