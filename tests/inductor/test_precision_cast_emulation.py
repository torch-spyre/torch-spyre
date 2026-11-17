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

"""``torch._inductor.config.emulate_precision_casts`` on the Spyre device.

With the setting on, Inductor's pointwise lowering wraps each low-precision op
in ``ops.to_dtype(value, dtype, use_compute_types=False)`` followed by
``ops.to_dtype(value, dtype)``: a cast to the op's own dtype, with no
``src_dtype``. ``SpyreOpFuncs.to_dtype`` cannot lower a cast without a source
dtype, so every Spyre compile failed there.

The setting is not exotic: PyTorch's compiler bisector flips it as one of its
Inductor subsystems, so every bisection of a Spyre program hit the error.

The emulation has nothing to do on Spyre, whose kernels compute in the tensor's
own dtype, so the request is dropped before lowering
(``TestDropPrecisionCastBarriers``, ``TestEmulatePrecisionCastsOnDevice``).
Accepting the cast in ``to_dtype`` would not be enough: a marked ``div``,
``where`` or ``softmax`` still fails to lower with the cast out of the way
(``Cannot resolve target for 'truediv'``).

Only ``TestEmulatePrecisionCastsOnDevice`` needs a card.
"""

import unittest

import torch
import torch._dynamo
import torch._inductor.config as inductor_config

import torch_spyre  # noqa: F401
from torch_spyre._inductor.precision_cast_barriers import (
    BARRIER_META_KEY,
    drop_precision_cast_barriers,
)

# NB: plain unittest, not torch.testing._internal.common_utils.run_tests, which
# seeds the Spyre device and would turn the CPU-only classes into device tests.


class TestDropPrecisionCastBarriers(unittest.TestCase):
    def _graph(self) -> torch.fx.Graph:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(torch.add, (x, x))
        z = graph.call_function(torch.relu, (y,))
        graph.output(z)
        return graph

    def test_marks_are_removed_from_every_node(self):
        graph = self._graph()
        for node in graph.nodes:
            node.meta[BARRIER_META_KEY] = True
        drop_precision_cast_barriers(graph)
        self.assertFalse(any(BARRIER_META_KEY in n.meta for n in graph.nodes))

    def test_other_meta_is_kept(self):
        graph = self._graph()
        for node in graph.nodes:
            node.meta[BARRIER_META_KEY] = True
            node.meta["val"] = node.name
        drop_precision_cast_barriers(graph)
        self.assertEqual(
            [n.meta for n in graph.nodes], [{"val": n.name} for n in graph.nodes]
        )

    def test_unmarked_graph_is_left_alone(self):
        graph = self._graph()
        drop_precision_cast_barriers(graph)
        self.assertEqual([n.meta for n in graph.nodes], [{}] * 4)

    def test_key_matches_the_one_upstream_writes(self):
        # The key is upstream's, not ours. If PyTorch renames it, the pass goes
        # quiet and the device tests below fail; this names the cause.
        import inspect

        import torch.fx.experimental.proxy_tensor as proxy_tensor

        self.assertIn(f'"{BARRIER_META_KEY}"', inspect.getsource(proxy_tensor))


def _chain(x, y):
    # Fuses into one kernel: the case where a CPU kernel's rounding differs
    # from eager's and the emulation would insert casts between the ops.
    return torch.relu(x + y * 2.0).sum(dim=-1)


class TestEmulatePrecisionCastsOnDevice(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(64, 128, dtype=torch.float16)
        self.y = torch.randn(64, 128, dtype=torch.float16)

    def tearDown(self):
        torch._dynamo.reset()

    def _run(self, fn, emulate: bool) -> torch.Tensor:
        torch._dynamo.reset()
        with inductor_config.patch(emulate_precision_casts=emulate):
            return torch.compile(fn)(self.x.to("spyre"), self.y.to("spyre")).cpu()

    def test_compiled_chain_matches_cpu(self):
        out = self._run(_chain, emulate=True)
        # Spyre fp16 is not IEEE half, and this sums 128 values.
        torch.testing.assert_close(
            out.float(), _chain(self.x, self.y).float(), atol=1.0, rtol=0.1
        )

    def test_the_setting_does_not_change_the_result(self):
        # There is no wider compute dtype on Spyre, so there is nothing for the
        # emulation to change: on and off must be the same kernel.
        self.assertTrue(
            torch.equal(
                self._run(_chain, emulate=True), self._run(_chain, emulate=False)
            )
        )

    def test_ops_the_marks_broke_beyond_to_dtype(self):
        # With only ``to_dtype`` fixed these three still failed to lower while
        # their nodes carried the mark; dropping the marks is what fixes them.
        cases = {
            "div": lambda x, y: x / (y.abs() + 1.0),
            "where": lambda x, y: torch.where(x > 0, x, y),
            "softmax": lambda x, y: torch.softmax(x, -1),
        }
        for name, fn in cases.items():
            with self.subTest(op=name):
                self.assertTrue(
                    torch.equal(
                        self._run(fn, emulate=True), self._run(fn, emulate=False)
                    )
                )

    def test_bfloat16_chain_compiles(self):
        self.x, self.y = self.x.to(torch.bfloat16), self.y.to(torch.bfloat16)
        out = self._run(lambda x, y: torch.relu(x + y), emulate=True)
        expected = torch.relu(self.x + self.y)
        torch.testing.assert_close(out.float(), expected.float(), atol=0.1, rtol=0.1)

    def test_eager_op_compiles(self):
        # A Spyre eager op is a compiled kernel too, and the setting is global.
        with inductor_config.patch(emulate_precision_casts=True):
            out = (self.x.to("spyre") + self.y.to("spyre")).cpu()
        torch.testing.assert_close(
            out.float(), (self.x + self.y).float(), atol=0.1, rtol=0.1
        )


if __name__ == "__main__":
    unittest.main()
