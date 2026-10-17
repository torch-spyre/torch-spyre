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


"""Every frontend-timing workload builds, and the cheapest ones compile.

The suite's own contract (``tools/frontend_timing/workloads.py``) is that each builder's
body is a COPY of a test that passes today, reconciled by hand when that test moves. A
copy with nothing exercising it drifts silently, and has: a builder went on citing a
fixture whose shapes had changed under it, and another passed no attention mask where
the test it copied passes one.

Building is the cheap half and covers every family: it runs the builder's shape rules
and materializes real device tensors, which is where a stale shape shows up. Compiling
is the expensive half -- tens of seconds per family -- so only the smallest few do it,
enough to keep the compile path itself honest.
"""

import os
import sys

import pytest
import torch  # noqa: F401

from torch_spyre._inductor import config

_TOOLS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools",
    "frontend_timing",
)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import workloads  # noqa: E402

#: The cheapest shape for each family: enough to exercise the builder, small enough that
#: building every one of them stays quick.
SMALLEST = {
    "granite_layer": {"S": 1, "layers": 1},
    "granite_lm_head": {"S": 1, "chunks": 2},
    "granite_embedding": {"S": 64},
    "transformer_block": {"S": 1},
    "mlp": {"seq_len": 64, "emb_dim": 256, "intermediate": 512, "layers": 1},
    "flash": {"B": 1, "H": 1, "Lq": 128, "Lk": 128, "D": 64, "block_size": 128},
    "elementwise_chain": {"ops": 2, "rows": 64, "cols": 64},
    "fanout": {"consumers": 2, "rows": 64, "cols": 64},
    "dup_constants": {"dups": 2, "B": 1, "M": 8, "N": 64},
    "control_flow": {},
}

#: Families cheap enough to compile here. The rest are covered by building alone; the
#: sweep itself is where a full compile of every family happens.
COMPILED = ("elementwise_chain", "fanout", "mlp")


def test_every_registered_family_has_a_smallest_shape():
    assert set(workloads.BUILDERS) == set(SMALLEST)


@pytest.mark.parametrize("family", sorted(SMALLEST))
def test_builds(family):
    """The builder runs, and its arguments are real tensors on the device."""
    built = workloads.build(family, **SMALLEST[family])
    assert built.name == family
    assert built.args
    assert built.params, "build() fills params in, defaults included"

    def check(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "spyre", f"{family}: {value.device}"
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)

    for arg in built.args:
        check(arg)


@pytest.mark.parametrize("family", COMPILED)
def test_compiles_frontend_only(family):
    """A real frontend compile, stopping before the backend."""
    built = workloads.build(family, **SMALLEST[family])
    with config.patch({"frontend_only": True}):
        compiled = torch.compile(built.fn, fullgraph=True)
        try:
            compiled(*built.args)
        except RuntimeError as exc:
            # Frontend-only compiles and then refuses to launch, which is the point
            # being reached. The driver treats this the same way.
            if "TORCH_SPYRE_FRONTEND_ONLY" not in str(exc):
                raise


def test_a_bad_shape_is_refused_by_the_builder():
    # The shape rules are checked in the builder so they surface here rather than as an
    # assertion deep in lowering. 128 heads of 4096 is head_dim 32, a partial stick.
    with pytest.raises(ValueError, match="multiple of 64"):
        workloads.build("granite_layer", S=1, E=4096, heads=128)
    with pytest.raises(ValueError, match="not divisible by heads"):
        workloads.build("granite_layer", S=1, E=4096, heads=30)
    with pytest.raises(ValueError, match="divisible by kv_heads"):
        workloads.build("granite_layer", S=1, heads=32, kv_heads=7)


def test_an_unknown_family_is_refused():
    with pytest.raises(SystemExit, match="unknown workload"):
        workloads.build("not_a_workload")
