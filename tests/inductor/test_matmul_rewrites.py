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

import unittest
from types import SimpleNamespace

import torch

from torch_spyre._inductor.temp_passes import _unflatten_bmm_batch_dims


class TestUnflattenBmmBatchDims(unittest.TestCase):
    @staticmethod
    def _make_graph(
        lhs_orig_shape,
        rhs_orig_shape,
        lhs_flat_shape,
        rhs_flat_shape,
        bmm_shape,
        output_shape,
    ):
        graph = torch.fx.Graph()

        def placeholder(name, shape):
            node = graph.placeholder(name)
            node.meta["val"] = torch.empty(shape, device="meta")
            return node

        def call(target, args, shape):
            node = graph.call_function(target, args=args)
            node.meta["val"] = torch.empty(shape, device="meta")
            return node

        lhs_orig = placeholder("lhs", lhs_orig_shape)
        rhs_orig = placeholder("rhs", rhs_orig_shape)
        lhs_reshape = call(
            torch.ops.aten.reshape.default,
            (lhs_orig, list(lhs_flat_shape)),
            lhs_flat_shape,
        )
        rhs_reshape = call(
            torch.ops.aten.reshape.default,
            (rhs_orig, list(rhs_flat_shape)),
            rhs_flat_shape,
        )
        bmm = call(
            torch.ops.aten.bmm.default,
            (lhs_reshape, rhs_reshape),
            bmm_shape,
        )
        output_view = call(
            torch.ops.aten.reshape.default,
            (bmm, list(output_shape)),
            output_shape,
        )
        graph.output(output_view)
        return graph, lhs_reshape, rhs_reshape, bmm

    @staticmethod
    def _run_rewrite(graph, lhs_reshape, rhs_reshape, bmm):
        _unflatten_bmm_batch_dims(
            SimpleNamespace(nodes=[bmm]), lhs_reshape, rhs_reshape
        )
        graph.lint()

    @staticmethod
    def _targets(graph):
        return [node.target for node in graph.nodes if node.op == "call_function"]

    def test_rewrites_only_exact_four_dimensional_matmul_sandwich(self):
        graph, lhs, rhs, bmm = self._make_graph(
            (2, 3, 5, 7),
            (2, 3, 7, 11),
            (6, 5, 7),
            (6, 7, 11),
            (6, 5, 11),
            (2, 3, 5, 11),
        )

        self._run_rewrite(graph, lhs, rhs, bmm)

        targets = self._targets(graph)
        self.assertIn(torch.ops.spyre.batched_matmul.default, targets)
        self.assertNotIn(torch.ops.aten.bmm.default, targets)

    def test_rewrites_exact_sdpa_prefill_and_unit_query_shapes(self):
        for query_length in (64, 1):
            for contraction, columns in ((128, 64), (64, 128)):
                with self.subTest(
                    query_length=query_length,
                    contraction=contraction,
                    columns=columns,
                ):
                    graph, lhs, rhs, bmm = self._make_graph(
                        (1, 32, query_length, contraction),
                        (1, 32, contraction, columns),
                        (32, query_length, contraction),
                        (32, contraction, columns),
                        (32, query_length, columns),
                        (1, 32, query_length, columns),
                    )

                    self._run_rewrite(graph, lhs, rhs, bmm)

                    targets = self._targets(graph)
                    self.assertIn(torch.ops.spyre.batched_matmul.default, targets)
                    self.assertNotIn(torch.ops.aten.bmm.default, targets)

    def test_rejects_same_numel_but_different_batch_prefixes(self):
        graph, lhs, rhs, bmm = self._make_graph(
            (2, 3, 5, 7),
            (1, 6, 7, 11),
            (6, 5, 7),
            (6, 7, 11),
            (6, 5, 11),
            (2, 3, 5, 11),
        )

        self._run_rewrite(graph, lhs, rhs, bmm)

        self.assertIn(torch.ops.aten.bmm.default, self._targets(graph))

    def test_rejects_flattened_operand_not_derived_from_restored_shape(self):
        graph, lhs, rhs, bmm = self._make_graph(
            (2, 3, 5, 7),
            (2, 3, 7, 11),
            (6, 5, 7),
            (6, 11, 7),
            (6, 5, 11),
            (2, 3, 5, 11),
        )

        self._run_rewrite(graph, lhs, rhs, bmm)

        self.assertIn(torch.ops.aten.bmm.default, self._targets(graph))

    def test_rejects_output_view_that_does_not_restore_matmul_axes(self):
        graph, lhs, rhs, bmm = self._make_graph(
            (2, 3, 5, 7),
            (2, 3, 7, 11),
            (6, 5, 7),
            (6, 7, 11),
            (6, 5, 11),
            (2, 3, 11, 5),
        )

        self._run_rewrite(graph, lhs, rhs, bmm)

        self.assertIn(torch.ops.aten.bmm.default, self._targets(graph))

    def test_does_not_create_rank_unsupported_native_bmm(self):
        graph, lhs, rhs, bmm = self._make_graph(
            (2, 3, 4, 5, 7),
            (2, 3, 4, 7, 11),
            (24, 5, 7),
            (24, 7, 11),
            (24, 5, 11),
            (2, 3, 4, 5, 11),
        )

        self._run_rewrite(graph, lhs, rhs, bmm)

        self.assertIn(torch.ops.aten.bmm.default, self._targets(graph))


if __name__ == "__main__":
    unittest.main()
