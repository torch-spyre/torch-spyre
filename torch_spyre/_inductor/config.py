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

import os
import sys
from typing import Literal

from torch.utils._config_module import install_config_module

lx_planning: bool = os.environ.get("LX_PLANNING", "1") == "1"
co_optimizing_lx_planning: bool = (
    os.environ.get("CO_OPTIMIZING_LX_PLANNING", "0") == "1"
)
chunk_large_tensors: bool = os.environ.get("CHUNK_LARGE_TENSORS", "0") == "1"

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

allow_all_ops_in_lx_planning: bool = False

# Insert clone ops at graph input/output boundaries so those buffers can be
# LX-pinned (see scratchpad.utils.clone_at_graph_boundaries). This path is not
# yet correct for all op types (e.g. matmul/layernorm/split under multi-core
# K-split) and is kept off by default. Deliberately separate from
# allow_all_ops_in_lx_planning, which only widens *intermediate* output
# eligibility and must not, on its own, enable boundary clone insertion.
lx_boundary_clones: bool = os.environ.get("LX_BOUNDARY_CLONES", "0") == "1"

dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

sencores: int = int(os.getenv("SENCORES", "32"))

# Symbolic-dim knobs consumed by compute_granularity in pass_utils.py.
# The pointwise work-division PR (#2499) wires that helper into the
# compilation pipeline; until then these knobs are read only by the
# helper and its unit tests. See #2284, #2287 for the design.

# Cap on bucket count (= max_size / granularity).
# TODO: confirm the default with the Deeptools team.
max_buckets: int = int(os.getenv("MAX_BUCKETS", "32"))

# Soft floor on the auto-derived granularity when mark_dynamic(min=...)
# is not provided. Keeps the picked granularity from collapsing to a
# very small divisor when max_size has many of them.
min_default_granularity: int = int(os.getenv("MIN_DEFAULT_GRANULARITY", "4"))

ignore_work_division_hints: bool = (
    os.environ.get("SPYRE_INDUCTOR_IGNORE_HINTS", "0") == "1"
)

# For K-split matmuls, permute physical core IDs so the cores collaborating on a
# K reduction land on adjacent ring positions, cutting PSUM chain hops from m*n
# to 1. The split itself is chosen by the cost-model planner; this only reorders
# cores at SDSC emission. Set SPYRE_CORE_ID_K_FAST_EMISSION=0 to disable.
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)

# When False (default), HBM tensor addresses are baked as concrete integers
# into the SDSC JSON and bundle.mlir emits sdsc_execute with no operands.
# When True, addresses are emitted as runtime symbols with
# !sdscbundle.input_arg<index> parameters, input_arg_extract ops, and
# affine.apply indirection for tiled loops.
bundle_symbolic_args: bool = os.environ.get("BUNDLE_SYMBOLIC_ARGS", "0") == "1"

# When True (default), LoopSpec nodes are fully unrolled into flat OpSpecs
# before generate_bundle runs.  Set to False to pass LoopSpecs through intact
# for the scf.for / affine.apply path.
unroll_loops: bool = os.environ.get("UNROLL_LOOPS", "1") == "1"

# Layout solver class used by default in scratchpad.allocator.DefaultAllocator.
# Options:
#  "greedy":   GreedyLayoutSolver (default),
#  "bestfit":  BestFitLayoutSolver,
#  "firstfit": FirstFitLayoutSolver.

# TODO(isuruf): Change to firstfit when deeptools PR4298 lands
layout_solver: Literal["greedy", "bestfit", "firstfit"] = "greedy"

install_config_module(sys.modules[__name__])
