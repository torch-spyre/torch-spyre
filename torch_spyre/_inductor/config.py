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
hbm_pool_planning: bool = _get_env_bool("HBM_POOL_PLANNING", True)

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

# Emit a native conv2d SDSC (opFuncName="conv2d" on the "pt" unit) instead of
# the im2col+matmul decomposition (conv2d_via_bmm_decomp). Off by default: the
# decomposition remains the default path and the fallback for cases the direct
# lowering does not yet support (grouped/transposed/non-fp16).
conv2d_direct_lowering: bool = os.environ.get("SPYRE_CONV2D_DIRECT", "0") == "1"

# For a strided (stride>1) direct-lowered conv2d, forbid splitting the output
# spatial dims (i/j) across cores. A strided conv's output coordinates do not
# map to a contiguous input span per core, so a spatial split shuffles the
# result (same failure and fix as the depthwise conv work). Forcing
# dim_splits[i]=dim_splits[j]=1 keeps each core computing whole spatial rows/
# cols. Defaults on; set SPYRE_INDUCTOR_DISABLE_CONV2D_SPATIAL_SPLIT=0 to opt
# out (e.g. to measure the shuffle or once the planner models strided spans).
disable_conv2d_spatial_split: bool = (
    os.environ.get("SPYRE_INDUCTOR_DISABLE_CONV2D_SPATIAL_SPLIT", "1") == "1"
)

# Opt-in OpSpec->KTIR emitter (experimental, #3380). When enabled the scheduler
# emits ``async_compile.ktir(...)`` instead of the SDSC bundle, and
# ``create_tensor_arg`` populates the op-spec buffer name so the emitter has a
# stable per-buffer identity. Inert by default: the SDSC/flex path is unchanged.
ktir_emitter: bool = os.environ.get("TORCH_SPYRE_KTIR", "0") == "1"

# Settings for device execution over the KTIR path. What is required is checked
# upfront by ``_check_ktir_device_prerequisites`` in ``execution/async_compile``,
# which names anything missing.

# A .mlir declaring the target device, passed to the backend compiler.
ktir_device_mlir: str = os.environ.get("KTIR_DEVICE_MLIR", "")

# Materialize compatible producer/consumer LX ownership changes as identity copies.
# Set SPYRE_LX_PLANNER_RELAYOUT=0 to disable this optimization.
lx_planner_relayout: bool = _get_env_bool("SPYRE_LX_PLANNER_RELAYOUT", True)

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

# Predicted-runtime reporting from the analytical cost model (cost_model.py,
# cost_model_pass.py).  NOT related to work_division.cost_model_matmul_division,
# which is a separate model used to choose a matmul work division.
#   ""/"0"/"false"  disabled -- the pass returns before touching the graph, so
#                   leaving it off costs one attribute read per compilation
#   "1"/"true"/"yes"/"on"  print a per-kernel breakdown and the program total after
#                   pre-scheduling, and expose them as
#                   CustomPreSchedulingPasses.last_cost_report
# Reads SPYRE_DUMP_COST so existing sweep scripts keep working.  NOTE that value is
# ALSO read directly by dump_cost_model.cost_dump_enabled(); both accept the same
# spellings, so one value drives this pass and that older per-op dump together.
# Tests override with config.patch({"cost_model": "1"}) rather than the environment.
cost_model: str = os.environ.get("SPYRE_DUMP_COST", "")

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

# Enable reduction-dim (Lk-style) coarse tiling. Defaults to enabled — this
# capability is exercised by passing tests today. Disabling it (or a future
# hardware limitation that can't support it) makes planning treat any op
# whose group requests reduction-dim tiling as unsupported, raising
# Unsupported rather than attempting to tile it.
enable_reduction_tiling: bool = (
    os.environ.get("SPYRE_INDUCTOR_ENABLE_REDUCTION_TILING", "1") == "1"
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
# When False, HBM tensor addresses are baked as concrete integers.
# (SDSC path always symbolic as of #3741; baked mode only via the KTIR
# emitter, i.e. also requires ktir_emitter=True / TORCH_SPYRE_KTIR=1.)
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

# OpSpec validation at pipeline stage boundaries. Enabled by default to catch
# invariant violations early. Set SPYRE_VALIDATE_OP_SPECS=0 to disable.
validate_op_specs: bool = os.environ.get("SPYRE_VALIDATE_OP_SPECS", "1") == "1"
# Use the C++ (native) permutation-layout packer accelerator, which the
# simulated-annealing layout solver drives. The native and Python packers are
# behaviourally identical (verified bit-for-bit); the native one is faster. Set
# False (or ``TORCH_SPYRE_NATIVE_PACKER=0``/``false``, which backs this default)
# to force the pure-Python packer. A missing native class is a stale or
# incomplete build, not a supported mode, and raises rather than falling back.
native_layout_packer: bool = _get_env_bool("TORCH_SPYRE_NATIVE_PACKER", True)

install_config_module(sys.modules[__name__])
