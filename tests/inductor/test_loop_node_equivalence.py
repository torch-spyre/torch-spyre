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

"""Building LoopNodes around pre-scheduling changes nothing downstream.

Each graph is compiled twice: with the nodes (the default: ``collapse`` at the
end of ``splice_while_loops``, every later pass against ``OpsView``, ``expand``
before the cost model) and with ``loop_node.BUILD_LOOP_NODES`` off, which is the
pipeline as it was before nodes existed.  The IR at the end of pre-scheduling
and the generated code must be identical, with only LX allocation offsets
masked.  The test also checks that the nodes really were in the graph while the
passes ran, so it cannot pass by never building them.
"""

import contextlib
import dataclasses
import os
import sys
import unittest
from unittest import mock

import pytest
import regex as re
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME
from torch_spyre._inductor import loop_node
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.loop_node import LoopNode
from torch_spyre._inductor.pass_utils import format_operations
from torch_spyre._inductor.passes import CustomPreSchedulingPasses

from for_each_tile_fixtures import (
    D,
    LK,
    LQ,
    M,
    K,
    N,
    nested_online_softmax_fn,
    nested_split_m_then_k_fn,
    nested_two_inner_loops_shared_init_fn,
    online_softmax_fn,
    split_k_fn,
    triple_nested_stardep_multilevel_fn,
)
from utils_inductor import cached_xavier

_decompositions = sys.modules["torch_spyre._inductor.decompositions"]


def _depth(operations) -> int:
    return max(
        (1 + _depth(op.body) for op in operations if isinstance(op, LoopNode)),
        default=0,
    )


def _count(operations) -> int:
    return sum(1 + _count(op.body) for op in operations if isinstance(op, LoopNode))


class _Recording(CustomPreSchedulingPasses):
    """Records the final IR and what the real operations list held per pass."""

    ir_text: str = ""
    per_pass: list = []
    leftover_nodes: bool = False

    def __call__(self, graph) -> None:
        super().__call__(graph)
        cls = type(self)
        cls.leftover_nodes = any(isinstance(op, LoopNode) for op in graph.operations)
        cls.ir_text = format_operations(graph.operations)


_real_flat_operations = ts_passes._flat_operations


def _recording_flat_operations(graph, pass_name):
    real = graph.operations
    _Recording.per_pass.append((pass_name, _count(real), _depth(real)))
    return _real_flat_operations(graph, pass_name)


_LX = re.compile(r"('lx':\s*)\d+")
# The AOT ID counts compiles in the process, so the second compile of the
# same graph always gets a different one.
_AOT_ID = re.compile(r"(# AOT ID: \[')\d+(_)")
# A carry's scratch buffer is named ``while_carry_<id(while_op)>_<i>``
# (while_loop_bridge.py), so the name differs between any two compiles.
_CARRY_ID = re.compile(r"(while_carry_)\d+(_)")


def _mask(text: str) -> str:
    """Mask what differs between two compiles of the same graph whatever the
    pipeline: LX allocation offsets (they vary run to run on nested graphs),
    the per-process compile counter in the AOT ID header, and the ``id()`` in
    a carry's scratch-buffer name."""
    text = _LX.sub(r"\g<1>N", text)
    text = _AOT_ID.sub(r"\g<1>N\g<2>", text)
    return _CARRY_ID.sub(r"\g<1>N\g<2>", text)


def _matmul_args():
    return (cached_xavier((M, K)), cached_xavier((K, N), differentiation=1))


def _nested_matmul_args():
    return (
        cached_xavier((256, 256)),
        cached_xavier((256, 64), differentiation=1),
    )


def _attention_args():
    return (
        cached_xavier((LQ, D)),
        cached_xavier((LK, D), differentiation=1),
        cached_xavier((LK, D), differentiation=2),
    )


def _sdpa_gqa_args():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 4, 32, 128, dtype=torch.float16, generator=g)
    k = torch.randn(2, 2, 256, 128, dtype=torch.float16, generator=g)
    v = torch.randn(2, 2, 256, 128, dtype=torch.float16, generator=g)
    b = torch.randn(2, 4, 32, 256, dtype=torch.float16, generator=g) * 0.01
    return (q, k, v, b)


def _sdpa_gqa(q, k, v, b):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=b, dropout_p=0.0, scale=128**-0.5, enable_gqa=True
    )


_ORIGINAL_SELECT = _decompositions._select_sdpa_tiling


def _force_every_sdpa_loop(**kwargs):
    # As test_sdpa_tiling.test_complete_gqa_tile_nest: B/Hkv/G/Lq maps around
    # an Lk scan, five loops.
    return dataclasses.replace(
        _ORIGINAL_SELECT(**kwargs),
        strategy="work_divided_tiled",
        kv_block_size=128,
        num_kv_blocks=2,
        num_q_tiles=2,
        q_tile_size=16,
        num_batch_tiles=2,
        num_head_tiles=2,
        num_group_tiles=2,
        kv_blocks_per_loop_group=2,
    )


# name -> (fn, args factory, extra patch factory or None)
_CASES = {
    "split_k": (split_k_fn, _matmul_args, None),
    "online_softmax": (online_softmax_fn, _attention_args, None),
    "nested_online_softmax": (nested_online_softmax_fn, _attention_args, None),
    "nested_split_m_then_k": (nested_split_m_then_k_fn, _nested_matmul_args, None),
    "nested_two_inner_loops": (
        nested_two_inner_loops_shared_init_fn,
        _matmul_args,
        None,
    ),
    "triple_nested_stardep_multilevel": (
        triple_nested_stardep_multilevel_fn,
        lambda: (
            cached_xavier((2, 256, 256)),
            cached_xavier((2, 256, 64), differentiation=1),
        ),
        None,
    ),
    "sdpa_gqa_tile_nest": (
        _sdpa_gqa,
        _sdpa_gqa_args,
        lambda: mock.patch.object(
            _decompositions,
            "_select_sdpa_tiling",
            side_effect=_force_every_sdpa_loop,
        ),
    ),
}


@contextlib.contextmanager
def _deterministic_algorithms():
    """``torch.use_deterministic_algorithms(True, warn_only=True)``, restored."""
    was_on = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=was_warn_only)


def _compile(fn, args, extra_patch, *, nodes: bool):
    _Recording.ir_text = ""
    _Recording.per_pass = []
    _Recording.leftover_nodes = False
    torch._dynamo.reset()
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(ts_passes, "CustomPreSchedulingPasses", _Recording)
        )
        stack.enter_context(
            mock.patch.object(ts_passes, "_flat_operations", _recording_flat_operations)
        )
        stack.enter_context(mock.patch.object(loop_node, "BUILD_LOOP_NODES", nodes))
        stack.enter_context(torch._inductor.config.patch(force_disable_caches=True))
        # The co-optimizing LX planner's CP-SAT solve is time-bounded
        # (cpsat_time_limit_seconds) and runs several search workers, so on a
        # larger graph it returns whichever good solution it reached first and
        # two compiles of the same graph can get different work splits. With
        # deterministic algorithms on, it uses a single worker (fixed seed), so
        # the plan is reproducible and the co-optimizer is compared under nodes.
        stack.enter_context(_deterministic_algorithms())
        if extra_patch is not None:
            stack.enter_context(extra_patch())
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        out, code = run_and_get_code(compiled, *[a.to(DEVICE_NAME) for a in args])
    return {
        "ir": _Recording.ir_text,
        "code": "\n".join(code),
        "per_pass": list(_Recording.per_pass),
        "leftover_nodes": _Recording.leftover_nodes,
        "out": out.cpu(),
    }


@pytest.mark.parametrize("case", list(_CASES))
def test_loop_nodes_change_nothing_downstream(case):
    fn, make_args, extra_patch = _CASES[case]
    args = make_args()
    flat = _compile(fn, args, extra_patch, nodes=False)
    with_nodes = _compile(fn, args, extra_patch, nodes=True)

    # The nodes really existed while the passes ran: the pass right after the
    # splice (deadcode_elimination) saw at least one node in the real list.
    names = [name for name, _, _ in with_nodes["per_pass"]]
    assert names[0] == "splice_while_loops"
    first_after_splice = with_nodes["per_pass"][1]
    assert first_after_splice[1] > 0, (
        f"no LoopNode in graph.operations at {first_after_splice[0]}"
    )
    # Every pass after the splice saw them (none removes a whole loop here).
    assert all(count > 0 for _, count, _ in with_nodes["per_pass"][1:])
    # ...and none were built with the switch off.
    assert all(count == 0 for _, count, _ in flat["per_pass"])
    # expand left none behind.
    assert not with_nodes["leftover_nodes"]
    assert "LoopNode" not in with_nodes["ir"]

    dump = os.environ.get("LOOP_NODE_EQUIV_DUMP")
    if dump:
        for tag, res in (("flat", flat), ("nodes", with_nodes)):
            for key in ("ir", "code"):
                with open(f"{dump}/{case}.{tag}.{key}.txt", "w") as f:
                    f.write(_mask(res[key]))
    assert _mask(with_nodes["ir"]) == _mask(flat["ir"])
    assert _mask(with_nodes["code"]) == _mask(flat["code"])
    # Same kernels give the same answer.
    torch.testing.assert_close(with_nodes["out"], flat["out"], atol=0, rtol=0)

    count, depth = first_after_splice[1], first_after_splice[2]
    print(f"\nCOVERAGE {case}: {count} LoopNode(s), depth {depth}")


if __name__ == "__main__":
    unittest.main()
