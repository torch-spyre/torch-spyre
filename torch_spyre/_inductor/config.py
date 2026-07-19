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
from .logging_utils import _get_env_bool

lx_planning: bool = os.environ.get("LX_PLANNING", "1") == "1"
co_optimizing_lx_planning: bool = (
    os.environ.get("CO_OPTIMIZING_LX_PLANNING", "0") == "1"
)
hbm_planning: bool = _get_env_bool("SPYRE_INDUCTOR_MEMORY_PLAN", True)

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

allow_all_ops_in_lx_planning: bool = False

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

ignore_wsr_hints: bool = os.environ.get("SPYRE_INDUCTOR_IGNORE_HINTS", "0") == "1"

# Per-pass operation logging for CustomPreSchedulingPasses.
# Set to "all" or "1" to log after every pass, or a comma-separated list of
# pass function names (e.g., "split_multi_ops,insert_restickify") to log only
# after specific passes. Set via SPYRE_LOG_PASSES env var or programmatically.
log_passes: str = os.environ.get("SPYRE_LOG_PASSES", "")

# Disable compiler-generated span-overflow coarse-tiling hints.  The global
# SPYRE_INDUCTOR_IGNORE_HINTS flag also disables these so one switch can still
# suppress all WSR/coarse-tiling hint paths.
#
# Defaults to disabled (opt-in): span-overflow auto-tiling can synchronize
# compatible contiguous pointwise groups, but incompatible producer/consumer
# groups and reduction-dim tiling still need broader support. Set
# SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS=0 to opt in;
# tests exercising this path directly should override via
# config.patch({"ignore_span_overflow_hints": False}).
ignore_span_overflow_hints: bool = (
    ignore_wsr_hints
    or os.environ.get("SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS", "1") == "1"
)

# For K-split matmuls, permute physical core IDs so the cores collaborating on a
# K reduction land on adjacent ring positions, cutting PSUM chain hops from m*n
# to 1. The split itself is chosen by the cost-model planner; this only reorders
# cores at SDSC emission. Set SPYRE_CORE_ID_K_FAST_EMISSION=0 to disable.
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)

# When True (default), HBM tensor addresses are emitted as runtime symbols
# with !sdscbundle.input_arg<index> parameters and input_arg_extract ops
# in the bundle.mlir.
# When False, HBM tensor addresses are baked as concrete integers
# into the SDSC JSON and bundle.mlir emits sdsc_execute with no operands.
bundle_symbolic_args: bool = os.environ.get("BUNDLE_SYMBOLIC_ARGS", "1") == "1"

# Layout solver class used by default in scratchpad.allocator.ScratchpadAllocator.
# Options:
#  "greedy":       GreedyLayoutSolver (default),
#  "bestfit":      BestFitLayoutSolver,
#  "firstfit":     FirstFitLayoutSolver,
#  "simulated_annealing":  SimulatedAnnealingLayoutSolver,
#  "cpsat":    CpSatLayoutSolver (OR-Tools CP-SAT joint core-division +
#              LX placement, minimizing HBM transfer traffic).

# TODO(isuruf): Change to firstfit when deeptools PR4298 lands
layout_solver: Literal[
    "greedy", "bestfit", "firstfit", "cpsat", "simulated_annealing"
] = os.environ.get("LAYOUT_SOLVER", "greedy")  # type: ignore[assignment]

install_config_module(sys.modules[__name__])
