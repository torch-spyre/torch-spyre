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

"""Feasibility tests for expressing the complete SDPA tile nest as HOPs."""

import contextlib
import math
import unittest

import torch
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor.wsr import for_each_tile


@contextlib.contextmanager
def _post_grad_graphs():
    """Capture graphs after scan has been converted to while_loop."""
    import torch._inductor.fx_passes.post_grad as post_grad

    seen: list[torch.fx.GraphModule] = []
    original = post_grad.decompose_scan_to_while_loop

    def capture(graph_module):
        result = original(graph_module)
        seen.append(graph_module)
        return result

    post_grad.decompose_scan_to_while_loop = capture
    try:
        yield seen
    finally:
        post_grad.decompose_scan_to_while_loop = original


def _count_while_loops(graph_module: torch.fx.GraphModule) -> int:
    count = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if "while_loop" in str(node.target):
            count += 1
        for argument in node.args:
            subgraph = argument
            if isinstance(subgraph, torch.fx.Node) and subgraph.op == "get_attr":
                subgraph = getattr(graph_module, str(subgraph.target), None)
            if isinstance(subgraph, torch.fx.GraphModule):
                count += _count_while_loops(subgraph)
    return count


def _materialization_nodes(graph_module: torch.fx.GraphModule) -> list[str]:
    materializations = ("copy", "copies", "clone", "contiguous", "cat", "stack")
    found = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if any(name in target for name in materializations):
            found.append(target)
        for argument in node.args:
            subgraph = argument
            if isinstance(subgraph, torch.fx.Node) and subgraph.op == "get_attr":
                subgraph = getattr(graph_module, str(subgraph.target), None)
            if isinstance(subgraph, torch.fx.GraphModule):
                found.extend(_materialization_nodes(subgraph))
    return found


def _sdpa_for_each_tile(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    *,
    batch_tile_size: int,
    head_tile_size: int,
    query_tile_size: int,
    kv_tile_size: int,
    group_tile_size: int | None = None,
) -> torch.Tensor:
    """Prototype B/H[/G]/Lq maps around an Lk online-softmax reduction."""
    use_gqa = group_tile_size is not None
    scale = 1.0 / math.sqrt(query.shape[-1])

    def kv_level(q_tile, k_tile, v_tile, bias_tile):
        def body(carry, tiles):
            running_max, denominator, accumulator = carry
            q_whole, k_block, v_block, bias_block = tiles
            scores = torch.matmul(q_whole, k_block.transpose(-1, -2)) * scale
            scores = scores + bias_block
            new_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
            correction = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max)
            new_denominator = denominator * correction + probabilities.sum(
                dim=-1, keepdim=True
            )
            new_accumulator = accumulator * correction + torch.matmul(
                probabilities, v_block
            )
            return (new_max, new_denominator, new_accumulator), None

        accumulator_shape = (*q_tile.shape[:-1], 1)
        (_, denominator, accumulator), _ = for_each_tile(
            body,
            (q_tile, k_tile, v_tile, bias_tile),
            dims=(None, -2, -2, -1),
            tile_size=kv_tile_size,
            init=(
                torch.full(
                    accumulator_shape,
                    float("-inf"),
                    dtype=q_tile.dtype,
                    device=q_tile.device,
                ),
                torch.zeros(
                    accumulator_shape, dtype=q_tile.dtype, device=q_tile.device
                ),
                torch.zeros_like(q_tile),
            ),
        )
        return accumulator / denominator

    def query_level(q_tile, k_tile, v_tile, bias_tile):
        def body(_, tiles):
            q_block, k_whole, v_whole, bias_block = tiles
            return None, kv_level(q_block, k_whole, v_whole, bias_block)

        _, result = for_each_tile(
            body,
            (q_tile, k_tile, v_tile, bias_tile),
            dims=(-2, None, None, -2),
            tile_size=query_tile_size,
            out_dim=-2,
        )
        return result

    def group_level(q_tile, k_tile, v_tile, bias_tile):
        if not use_gqa:
            return query_level(q_tile, k_tile, v_tile, bias_tile)

        def body(_, tiles):
            q_group, k_whole, v_whole, bias_group = tiles
            return None, query_level(q_group, k_whole, v_whole, bias_group)

        _, result = for_each_tile(
            body,
            (q_tile, k_tile, v_tile, bias_tile),
            dims=(2, None, None, 2),
            tile_size=group_tile_size,
            out_dim=2,
        )
        return result

    def head_level(q_tile, k_tile, v_tile, bias_tile):
        def body(_, tiles):
            q_heads, k_heads, v_heads, bias_heads = tiles
            return None, group_level(q_heads, k_heads, v_heads, bias_heads)

        _, result = for_each_tile(
            body,
            (q_tile, k_tile, v_tile, bias_tile),
            dims=(1, 1, 1, 1),
            tile_size=head_tile_size,
            out_dim=1,
        )
        return result

    def batch_body(_, tiles):
        q_batch, k_batch, v_batch, bias_batch = tiles
        return None, head_level(q_batch, k_batch, v_batch, bias_batch)

    _, result = for_each_tile(
        batch_body,
        (query, key, value, bias),
        dims=(0, 0, 0, 0),
        tile_size=batch_tile_size,
        out_dim=0,
    )
    return result


def _sdpa_query_kv_for_each_tile(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_tile_size: int,
    kv_tile_size: int,
) -> torch.Tensor:
    """Two-level Lq-map/Lk-reduction isolating the first required nesting."""

    def query_body(_, query_tiles):
        (query_tile,) = query_tiles

        def kv_body(carry, kv_tiles):
            running_max, denominator, accumulator = carry
            key_tile, value_tile = kv_tiles
            scores = query_tile @ key_tile.transpose(-1, -2)
            new_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
            correction = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max)
            return (
                new_max,
                denominator * correction + probabilities.sum(dim=-1, keepdim=True),
                accumulator * correction + probabilities @ value_tile,
            ), None

        rows = (query_tile.shape[0], 1)
        (_, denominator, accumulator), _ = for_each_tile(
            kv_body,
            (key, value),
            dims=(0, 0),
            tile_size=kv_tile_size,
            init=(
                torch.full(
                    rows,
                    float("-inf"),
                    dtype=query.dtype,
                    device=query.device,
                ),
                torch.zeros(rows, dtype=query.dtype, device=query.device),
                torch.zeros_like(query_tile),
            ),
        )
        return None, accumulator / denominator

    _, result = for_each_tile(
        query_body,
        (query,),
        dims=(0,),
        tile_size=query_tile_size,
        out_dim=0,
    )
    return result


def _sdpa_kv_for_each_tile(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    *,
    kv_tile_size: int,
) -> torch.Tensor:
    """One-level Lk reduction with production-like batched operands and bias."""

    def body(carry, tiles):
        running_max, denominator, accumulator = carry
        query_whole, key_tile, value_tile, bias_tile = tiles
        scores = query_whole @ key_tile.transpose(-1, -2)
        scores = scores / math.sqrt(query.shape[-1]) + bias_tile
        new_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(running_max - new_max)
        probabilities = torch.exp(scores - new_max)
        return (
            new_max,
            denominator * correction + probabilities.sum(dim=-1, keepdim=True),
            accumulator * correction + probabilities @ value_tile,
        ), None

    carry_shape = (*query.shape[:-1], 1)
    (_, denominator, accumulator), _ = for_each_tile(
        body,
        (query, key, value, bias),
        dims=(None, -2, -2, -1),
        tile_size=kv_tile_size,
        init=(
            torch.full(
                carry_shape,
                float("-inf"),
                dtype=query.dtype,
                device=query.device,
            ),
            torch.zeros(carry_shape, dtype=query.dtype, device=query.device),
            torch.zeros_like(query),
        ),
    )
    return accumulator / denominator


class TestSDPAForEachTile(unittest.TestCase):
    def _check(self, *, use_gqa: bool) -> None:
        torch.default_generator.manual_seed(0)
        batch, heads, groups, query_length, kv_length, head_dim = 2, 4, 2, 8, 12, 6
        if use_gqa:
            query = torch.randn(batch, heads, groups, query_length, head_dim)
            key = torch.randn(batch, heads, 1, kv_length, head_dim)
            value = torch.randn(batch, heads, 1, kv_length, head_dim)
            bias = (torch.randn(batch, 1, 1, query_length, kv_length) * 0.1).expand(
                batch, heads, groups, query_length, kv_length
            )
            group_tile_size = 1
            expected_loops = 5
        else:
            query = torch.randn(batch, heads, query_length, head_dim)
            key = torch.randn(batch, heads, kv_length, head_dim)
            value = torch.randn(batch, heads, kv_length, head_dim)
            bias = (torch.randn(batch, 1, query_length, kv_length) * 0.1).expand(
                batch, heads, query_length, kv_length
            )
            group_tile_size = None
            expected_loops = 4

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        reference = torch.matmul(torch.softmax(scores + bias, dim=-1), value)

        def fn(q, k, v, b):
            return _sdpa_for_each_tile(
                q,
                k,
                v,
                b,
                batch_tile_size=1,
                head_tile_size=2 if not use_gqa else 1,
                group_tile_size=group_tile_size,
                query_tile_size=4,
                kv_tile_size=3,
            )

        torch.testing.assert_close(fn(query, key, value, bias), reference)
        # Both variants use the same nested function code objects with different
        # ranks.  Reset Dynamo so its automatic-dynamic-shape retry from the first
        # variant does not turn the Python scale into a SymFloat captured by scan.
        torch._dynamo.reset()
        with _post_grad_graphs() as graphs:
            actual = torch.compile(fn, fullgraph=True)(query, key, value, bias)
        torch.testing.assert_close(actual, reference)
        self.assertTrue(graphs)
        self.assertEqual(_count_while_loops(graphs[-1]), expected_loops)
        # Each map level currently writes a scan stack; non-leading folds can
        # add another copy.  The proposed scan(out=) extension removes these.
        expected_materializations = 4
        materializations = _materialization_nodes(graphs[-1])
        self.assertEqual(
            len(materializations), expected_materializations, materializations
        )

    def test_mha_complete_tile_nest(self):
        self._check(use_gqa=False)

    def test_gqa_complete_tile_nest(self):
        self._check(use_gqa=True)

    def test_non_divisible_kv_requires_padding(self):
        key = torch.randn(1, 2, 10, 8)
        value = torch.randn(1, 2, 10, 8)

        def body(carry, tiles):
            return carry, None

        with self.assertRaisesRegex(ValueError, "ragged tiles are not supported"):
            for_each_tile(
                body,
                (key, value),
                dims=(-2, -2),
                tile_size=4,
                init=torch.zeros(1),
            )

    def _check_kv_loop_spyre(self, *, broadcast_bias: bool):
        batch, heads, query_length, kv_length, head_dim = 1, 2, 64, 256, 128
        torch.default_generator.manual_seed(0)
        query = torch.randn(batch, heads, query_length, head_dim, dtype=torch.float16)
        key = torch.randn(batch, heads, kv_length, head_dim, dtype=torch.float16)
        value = torch.randn(batch, heads, kv_length, head_dim, dtype=torch.float16)
        bias = (
            torch.randn(
                batch,
                1 if broadcast_bias else heads,
                query_length,
                kv_length,
                dtype=torch.float16,
            )
            * 0.01
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=bias
        )

        def fn(q, k, v, b):
            if broadcast_bias:
                b = b.expand(batch, heads, query_length, kv_length)
            return _sdpa_kv_for_each_tile(
                q,
                k,
                v,
                b,
                kv_tile_size=128,
            )

        actual, sources = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False),
            query.to("spyre"),
            key.to("spyre"),
            value.to("spyre"),
            bias.to("spyre"),
        )
        torch.testing.assert_close(
            actual.cpu().float(), reference.float(), atol=0.1, rtol=0.1
        )
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 1)

    def test_kv_loop_with_dense_bias_device(self):
        """The standalone Lk HOP works with a production-like dense bias."""
        self._check_kv_loop_spyre(broadcast_bias=False)

    def test_gqa_kv_loop_with_dense_bias_device(self):
        """The Lk HOP preserves native GQA without repeating K and V."""
        batch, heads, groups, query_length, kv_length, head_dim = 1, 2, 2, 64, 256, 128
        torch.default_generator.manual_seed(0)
        query = torch.randn(
            batch, heads, groups, query_length, head_dim, dtype=torch.float16
        )
        key = torch.randn(batch, heads, 1, kv_length, head_dim, dtype=torch.float16)
        value = torch.randn(batch, heads, 1, kv_length, head_dim, dtype=torch.float16)
        bias = torch.randn(
            batch, heads, groups, query_length, kv_length, dtype=torch.float16
        )
        scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)
        reference = torch.softmax(scores + bias, dim=-1) @ value

        def fn(q, k, v, b):
            return _sdpa_kv_for_each_tile(q, k, v, b, kv_tile_size=128)

        actual, sources = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False),
            query.to("spyre"),
            key.to("spyre"),
            value.to("spyre"),
            bias.to("spyre"),
        )
        torch.testing.assert_close(
            actual.cpu().float(), reference.float(), atol=0.1, rtol=0.1
        )
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 1)

    def test_kv_loop_with_broadcast_bias_device(self):
        """The standalone Lk HOP supports a stride-zero broadcast bias."""
        self._check_kv_loop_spyre(broadcast_bias=True)

    def test_gqa_complete_tile_nest_device(self):
        """The five-level GQA nest compiles and executes correctly."""
        batch, heads, groups, query_length, kv_length, head_dim = 2, 2, 2, 32, 256, 128
        torch.default_generator.manual_seed(0)
        query = torch.randn(
            batch, heads, groups, query_length, head_dim, dtype=torch.float16
        )
        key = torch.randn(batch, heads, 1, kv_length, head_dim, dtype=torch.float16)
        value = torch.randn(batch, heads, 1, kv_length, head_dim, dtype=torch.float16)
        bias = (
            torch.randn(
                batch, heads, groups, query_length, kv_length, dtype=torch.float16
            )
            * 0.01
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        reference = torch.matmul(torch.softmax(scores + bias, dim=-1), value)

        def fn(q, k, v, b):
            return _sdpa_for_each_tile(
                q,
                k,
                v,
                b,
                batch_tile_size=1,
                head_tile_size=1,
                group_tile_size=1,
                query_tile_size=16,
                kv_tile_size=128,
            )

        device_inputs = tuple(t.to("spyre") for t in (query, key, value, bias))
        actual, sources = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False),
            *device_inputs,
        )
        torch.testing.assert_close(
            actual.cpu().float(), reference.float(), atol=0.1, rtol=0.1
        )
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 5)

    def test_two_nested_maps_device(self):
        """Two map-mode levels compile and execute after PR #4705."""
        x = torch.randn(128, 128, dtype=torch.float16)

        def fn(x):
            def outer_body(_, outer_tiles):
                (outer_tile,) = outer_tiles

                def inner_body(_, inner_tiles):
                    (inner_tile,) = inner_tiles
                    return None, inner_tile + 1

                _, inner_result = for_each_tile(
                    inner_body,
                    (outer_tile,),
                    dims=(1,),
                    tile_size=64,
                    out_dim=1,
                )
                return None, inner_result

            _, result = for_each_tile(
                outer_body, (x,), dims=(0,), tile_size=64, out_dim=0
            )
            return result

        actual, sources = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False),
            x.to("spyre"),
        )
        torch.testing.assert_close(
            actual.cpu().float(), (x + 1).float(), atol=0.1, rtol=0.1
        )
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 2)

    def test_query_kv_nested_device(self):
        """An Lq map composes correctly with an Lk carry loop."""
        query = torch.randn(128, 128, dtype=torch.float16)
        key = torch.randn(256, 128, dtype=torch.float16)
        value = torch.randn(256, 128, dtype=torch.float16)
        reference = torch.softmax(query @ key.T, dim=-1) @ value

        def fn(q, k, v):
            return _sdpa_query_kv_for_each_tile(
                q, k, v, query_tile_size=64, kv_tile_size=128
            )

        actual, sources = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False),
            query.to("spyre"),
            key.to("spyre"),
            value.to("spyre"),
        )
        torch.testing.assert_close(
            actual.cpu().float(), reference.float(), atol=0.1, rtol=0.1
        )
        self.assertEqual(sum(source.count("LoopSpec(") for source in sources), 2)


if __name__ == "__main__":
    unittest.main()
