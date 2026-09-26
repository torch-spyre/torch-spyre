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

"""End-to-end coverage for ``nested_compile_region`` bodies on Spyre.

A ``torch.compiler.nested_compile_region`` block called N times lowers to an
``invoke_subgraph`` HOP, so the region body is compiled once as a *separate*
Inductor graph and called from the parent. That split graph is what these
tests exercise:

- ``TestInvokeSubgraphSplit`` -- regression guard for cross-graph buffer
  origins in ``split_multi_ops`` (issue #3883).
- ``TestInvokeSubgraphAttention`` -- a realistic Granite-8B prefill attention
  body (SDPA + o_proj) inside a region: compiles, and matches CPU numerically.
- ``TestInvokeSubgraphEmbeddingFedOperand`` -- the same attention body, but with
  the layer-0 hidden state produced by an in-graph embedding, so the call sites
  disagree about the operand's device layout.

Both assert on the FX graph Inductor actually receives: if Dynamo inlined the
regions into the parent there is no subgraph at all, and a test that only
checked shapes or numerics would pass while guarding nothing.
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F
from torch import nn
from torch._inductor import config as t_inductor_config
from torch.compiler import nested_compile_region

import torch_spyre  # noqa: F401  registers "spyre" + installs the inductor passes
from torch_spyre.constants import DEVICE_NAME

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils_inductor import compare_with_pytorch  # noqa: E402


class _Block(nn.Module):
    def forward(self, h):
        # Fused multi-op pointwise body: mul -> add -> relu in one loop body,
        # forcing split_multi_ops to materialize an intermediate and hit the
        # FX-node insertion that used to assert.
        return torch.relu(h * 3.0 + 1.0)


def _region(block):
    # nested_compile_region cannot mark a bound method, so wrap it.
    def wrapper(*args, **kwargs):
        return block.forward(*args, **kwargs)

    return nested_compile_region(wrapper)


class _RegionTestCase(unittest.TestCase):
    """Base for region tests: disables the FX graph cache, resets Dynamo."""

    def setUp(self):
        super().setUp()
        # Load-bearing, not hygiene: a cached FX graph is replayed without
        # re-running the Spyre pre-scheduling passes, so the passes under test
        # never fire and these tests would pass against broken code.
        patcher = t_inductor_config.patch("fx_graph_cache", False)
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def _compile_counting_hops(self, fn, seen_hops):
        """Compile ``fn``, recording the parent graph's invoke_subgraph nodes.

        Appends one list of HOP nodes per backend invocation to ``seen_hops``,
        so a caller can assert the regions were not inlined away.
        """

        def backend(gm, example_inputs):
            seen_hops.append(
                [
                    n
                    for n in gm.graph.nodes
                    if "invoke_subgraph" in str(getattr(n, "target", ""))
                ]
            )
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        return torch.compile(fn, backend=backend, dynamic=False, fullgraph=True)

    def _assert_regions_not_inlined(self, seen_hops, expected=2):
        self.assertTrue(seen_hops, "compile backend never ran")
        self.assertGreaterEqual(
            len(seen_hops[0]),
            expected,
            "expected repeated invoke_subgraph calls, got "
            f"{[n.name for n in seen_hops[0]]}",
        )


class TestInvokeSubgraphSplit(_RegionTestCase):
    """Regression test for invoke_subgraph + split_multi_ops graph identity.

    A region with a *fused multi-op* pointwise body (``relu(h * 3 + 1)`` ->
    mul, add, relu in one loop body, so ``split_multi_ops`` fires and reaches
    its FX-node insertion), on Spyre-device tensors, called N times so the
    region lowers to an ``invoke_subgraph`` HOP. Before the fix this failed
    during ``torch.compile`` with::

        torch._inductor.exc.InductorError: AssertionError:
            Node to insert before is not in graph.

    Root cause: the subgraph ComputedBuffer's ``origins`` set spans TWO
    ``fx.Graph`` objects -- the parent's ``invoke_subgraph`` / ``get_attr``
    nodes AND the subgraph-local ``mul`` node. ``split_multi_ops`` picked
    ``next(iter(op.origins))``, which could return the parent's
    invoke_subgraph node; that node is not in the subgraph's ``gl.graph``, so
    ``gl.graph.inserting_before(orig_node)`` asserted. The fix
    (``pass_utils.origin_in_graph``) selects the origin whose ``.graph
    is gl.graph``.
    """

    def test_nested_region_multi_op_compiles(self):
        """A reused region with a fused multi-op body must compile and run.

        Used to raise InductorError("Node to insert before is not in graph.")
        from split_multi_ops' FX insertion, during compilation.
        """
        blocks = [_region(_Block()) for _ in range(3)]

        def outer(h):
            for b in blocks:
                h = b(h)
            return h

        # Assert on what Inductor actually receives: if the regions were
        # inlined into the parent graph there is no cross-graph origin set,
        # and the test would silently guard nothing.
        seen_hops = []
        compiled = self._compile_counting_hops(outer, seen_hops)

        h = torch.randn(2, 64, dtype=torch.float16, device=DEVICE_NAME)
        out = compiled(h)

        self.assertEqual(tuple(out.shape), (2, 64))
        self._assert_regions_not_inlined(seen_hops)


# Granite 3.3 2B prefill geometry, after hf_adapters' stick-alignment padding
# of head_dim 64 -> 128 (prepare_rope_and_heads -> pad_attention_heads).
_BATCH = 1
_SEQLEN = 512  # == _SDPA_MAX_SEQUENCE_TILE_SIZE, so SDPA picks work_divided
_HIDDEN = 2048
_NUM_HEADS = 32
_NUM_KVHEADS = 8
_HEAD_DIM = 128
# Granite scales attention by an explicit multiplier, not by head_dim**-0.5.
_ATTENTION_MULTIPLIER = 0.015625
_NUM_LAYERS = 2  # >= 2 call sites, else Dynamo inlines the region


class _AttentionTail(nn.Module):
    """Granite attention tail: SDPA -> o_proj, hidden_state in and out.

    A trimmed ``StandardGQABlock._region_attention_tail`` (hf_common.py) --
    the residual/norm/MLP stages are dropped, keeping the SDPA -> o_proj pair
    that carries the failure. Taking and returning a hidden state is what lets
    the block stack: layer N's output is layer N+1's input, at identical
    shapes, so Dynamo shares one subgraph across the call sites.

    Deliberately more than a pointwise chain -- SDPA is a *Spyre
    decomposition*, so a region containing it exercises the decomp table
    threading in torch_spyre/_monkey_patch.py::_patch_invoke_subgraph
    _decompositions. The ``transpose(1, 2).reshape(...)`` view feeding o_proj
    is kept verbatim: it is the shape the failing o_proj batchmatmul's
    provenance points at, and it gives the linear a factorized layout coming
    out of SDPA's internal head tiling.
    """

    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(_NUM_HEADS * _HEAD_DIM, _HIDDEN, bias=False)

    def forward(self, hidden_states, q, key_cache, value_cache, attn_mask):
        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=_ATTENTION_MULTIPLIER,
            enable_gqa=True,
        )
        # Collapse heads before o_proj: the combined H*D dim is the linear's
        # contraction dim, spanning both num_heads and head_dim.
        attn_out = attn_out.transpose(1, 2).reshape(_BATCH, _SEQLEN, -1)
        # hidden_states threads through so the block composes as a layer; the
        # add also keeps o_proj's result from being the only thing on the wire.
        return hidden_states + self.o_proj(attn_out)


@nested_compile_region
def _shared_attention_tail(block, hidden_states, q, key_cache, value_cache, mask):
    """Mirrors ``hf_common._shared_region_attention_tail``.

    ``block`` is passed POSITIONALLY and never closed over: a
    ``nested_compile_region`` body is traced once, so a closed-over block
    would bind every call site to the first layer's weights.
    """
    return block(hidden_states, q, key_cache, value_cache, mask)


class TestInvokeSubgraphAttention(_RegionTestCase):
    """A Granite-2B prefill attention tail in a region must compile and be accurate.

    Covers the realistic use of ``nested_compile_region``: the attention tail
    marked as a region and reused across layers, so the body is compiled once
    into an ``invoke_subgraph`` subgraph rather than inlined per layer.

    Reproduces the ``dxp_standalone`` failure from the Granite 3.3 2B
    whole-forward compile, where the o_proj batchmatmul lowered inside
    ``repeated_subgraph0`` comes out rank-4 and role-scrambled instead of a
    clean rank-3 ``[mb, out, in]``, so the backend scheduler finds more than
    one output-reuse (reduction) dimension and aborts with::

        error: sbf-ddc: DtException: out_reuse_dim.size() == 1
        error: sbf-run-scheduler-on-sdsc: failed on program 'sdsc_28'
    """

    @staticmethod
    def _inputs():
        """Build the inputs ONCE on CPU, for both paths to share.

        Generating them per-device would silently compare two different random
        problems: ``torch.randn`` does not reproduce values across devices or
        dtypes from the same seed.
        """
        torch.manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, dtype=torch.float16)

        hidden_states = randn(_BATCH, _SEQLEN, _HIDDEN)
        q = randn(_BATCH, _NUM_HEADS, _SEQLEN, _HEAD_DIM)
        key_cache = randn(_BATCH, _NUM_KVHEADS, _SEQLEN, _HEAD_DIM)
        value_cache = randn(_BATCH, _NUM_KVHEADS, _SEQLEN, _HEAD_DIM)
        # Causal additive mask, matching what generate() hands the block.
        mask = torch.full((_SEQLEN, _SEQLEN), float("-inf"), dtype=torch.float16)
        mask = torch.triu(mask, diagonal=1).view(1, 1, _SEQLEN, _SEQLEN)
        return hidden_states, q, key_cache, value_cache, mask

    def test_attention_tail_region_compiles_and_matches_cpu(self):
        torch.manual_seed(0)
        blocks = [_AttentionTail().eval() for _ in range(_NUM_LAYERS)]

        def outer(hidden_states, q, key_cache, value_cache, mask):
            h = hidden_states
            for block in blocks:
                h = _shared_attention_tail(block, h, q, key_cache, value_cache, mask)
            return h

        # Stacking is what keeps every call site's input shapes identical, so
        # Dynamo shares one subgraph instead of inlining per layer -- the HOP
        # assertion below is what catches a regression back to inlining.
        for block in blocks:
            block.to(device=DEVICE_NAME, dtype=torch.float16)

        seen_hops = []
        compiled = self._compile_counting_hops(outer, seen_hops)
        cpu_inputs = self._inputs()
        spyre_inputs = [t.to(DEVICE_NAME) for t in cpu_inputs]

        # .cpu() forces the launch, so a runtime (not just compile) failure surfaces.
        out = compiled(*spyre_inputs).cpu()

        self.assertEqual(tuple(out.shape), (_BATCH, _SEQLEN, _HIDDEN))
        self._assert_regions_not_inlined(seen_hops, expected=_NUM_LAYERS)
        self.assertTrue(
            torch.isfinite(out.to(torch.float32)).all(), "output has non-finite values"
        )

        # Same computation without the region, in eager CPU float32: an
        # inlined-but-wrong subgraph would still produce the right shape, so
        # numerics are the real assertion here.
        for block in blocks:
            block.to(device="cpu", dtype=torch.float32)

        def outer_cpu(hidden_states, q, key_cache, value_cache, mask):
            h = hidden_states
            for block in blocks:
                h = block(h, q, key_cache, value_cache, mask)
            return h

        compare_with_pytorch(
            None,
            outer_cpu,
            *[t.float() for t in cpu_inputs],
            atol=0.2,
            rtol=0.2,
            target=out.float(),
        )


_VOCAB = 128  # small: the embedding table's own size is irrelevant here


class TestInvokeSubgraphEmbeddingFedOperand(_RegionTestCase):
    """An embedding-fed region stack: every call site must get one operand layout.

    Same body as ``TestInvokeSubgraphAttention`` (SDPA -> o_proj), but the hidden
    state entering layer 0 comes from an ``nn.Embedding`` inside the graph instead
    of arriving as a graph input.

    Cost parity, not yet an optimization: the compiler inserts the SAME single
    copy the eager path open-coded -- one restickify of the embedding output, with
    every later call site already compliant. ``_subgraph_boundary_stl`` documents
    why one fixed boundary layout is the conservative first choice and how it
    could be sharpened later.

    Covers the OPERAND side of the boundary only. Subgraph RESULTS are declared to
    carry the generic layout by the ``MultiOutput`` branch of
    ``propagate_spyre_tensor_layouts``, and nothing verifies the body produces it;
    Granite 3.3 2B happens to comply, so this test passes without exercising that
    half. A body whose last op committed a different orientation would disagree
    silently -- see the comment on that branch.

    The asymmetry: layer 0's operand is the embedding output, committed as
    ``device_size=[1, 32, 512, 64]`` / ``stride_map=[-1, 64, 2048, 1]``; layers
    1..N-1 take the previous region's ``MultiOutput``, stamped with the generic
    layout ``[512, 32, 1, 64]`` / ``[2048, 64, -1, 1]``. The ``stride_map``
    contents agree and both put the same variable on the stick -- so
    ``stick_compatible`` calls them compatible and no ordinary restickify is
    planned -- but the 512 extent sits on a different device AXIS. Codegen derives
    device strides from ``device_size`` positionally
    (``_calculate_device_stride``), so the body, codegened once from the first
    call site, would address every later site's operand wrongly. Hence
    ``require_exact`` on the boundary edge in propagate_layouts.

    The body must MIX across the axis whose placement differs, or this test cannot
    see the bug: a pointwise body reads every element exactly once and writes each
    result back through the same addressing, so the permutation cancels and the
    output is bit-identical no matter which layout arrives. SDPA + o_proj contract
    over the sequence and hidden axes, so wrong addressing changes which values
    are combined -- which is exactly why the 40-layer model emitted wrong tokens
    (' Par' -> 'pec') rather than merely failing to compile.
    """

    def test_embedding_fed_attention_region_matches_cpu(self):
        torch.manual_seed(0)
        embed = nn.Embedding(_VOCAB, _HIDDEN).eval()
        blocks = [_AttentionTail().eval() for _ in range(_NUM_LAYERS)]

        def outer(ids, q, key_cache, value_cache, mask):
            # Mirrors hf_granite._run_backbone_forward's prologue MINUS the
            # transpose/contiguous round trip it used to need.
            h = embed(ids)
            for block in blocks:
                h = _shared_attention_tail(block, h, q, key_cache, value_cache, mask)
            return h

        embed.to(device=DEVICE_NAME, dtype=torch.float16)
        for block in blocks:
            block.to(device=DEVICE_NAME, dtype=torch.float16)

        seen_hops = []
        compiled = self._compile_counting_hops(outer, seen_hops)

        # Reuse the attention test's tensors, dropping its hidden_states (the
        # embedding produces the hidden state here) and adding token ids.
        _, q, key_cache, value_cache, mask = TestInvokeSubgraphAttention._inputs()
        ids = torch.randint(0, _VOCAB, (_BATCH, _SEQLEN))
        cpu_inputs = (ids, q, key_cache, value_cache, mask)
        spyre_inputs = [t.to(DEVICE_NAME) for t in cpu_inputs]

        # .cpu() forces the launch, so a runtime (not just compile) failure surfaces.
        out = compiled(*spyre_inputs).cpu()

        self.assertEqual(tuple(out.shape), (_BATCH, _SEQLEN, _HIDDEN))
        self._assert_regions_not_inlined(seen_hops, expected=_NUM_LAYERS)
        self.assertTrue(
            torch.isfinite(out.to(torch.float32)).all(), "output has non-finite values"
        )

        # The real gate: a body fed its operand through the wrong axis-order
        # addressing still produces the right shape and finite values, so only
        # value equality separates a correct boundary copy from a missing one.
        embed.to(device="cpu", dtype=torch.float32)
        for block in blocks:
            block.to(device="cpu", dtype=torch.float32)

        def outer_cpu(ids_cpu, q_, k_, v_, mask_):
            h = embed(ids_cpu)
            for block in blocks:
                h = block(h, q_, k_, v_, mask_)
            return h

        compare_with_pytorch(
            None,
            outer_cpu,
            ids,
            *[t.float() for t in cpu_inputs[1:]],
            atol=0.2,
            rtol=0.2,
            target=out.float(),
        )


if __name__ == "__main__":
    unittest.main()
