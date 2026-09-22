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
    os.environ.get("CO_OPTIMIZING_LX_PLANNING", "1") == "1"
)
hbm_pool_planning: bool = os.getenv("HBM_POOL_PLANNING", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Bracket risky FallbackKernel calls (opaque device dispatches with no native
# Spyre lowering, e.g. custom ops or ops/eager.py's nested-torch.compile'd
# eager kernels) with per-buffer LX dump/restore clones, so a buffer this
# graph still has LX-resident across such a call survives whatever the
# opaque call's own kernel does to LX. Defaults on; set
# ENABLE_LX_CONTEXT_SWITCHING=0 to disable for debugging/bisection.
enable_lx_context_switching: bool = os.getenv(
    "ENABLE_LX_CONTEXT_SWITCHING", "1"
).lower() in (
    "1",
    "true",
    "yes",
)

# Select who allocates the HBM pool for an SDSC bundle's intermediates:
# False (default) has the backend self-allocate via
# sdscbundle.device_mem_allocate, exactly matching pre-existing behavior.
# True has the front end allocate a real PyTorch tensor (via
# spyre_empty_with_layout) and pass its address in as %pool_base_addr.
frontend_pool_allocation: bool = os.getenv("FRONTEND_POOL_ALLOCATION", "0").lower() in (
    "1",
    "true",
    "yes",
)

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

# Enable certified LX ownership changes: movement, exact fused-axis views,
# consumer-compatible producer order, and same-core restickify residency.
# Set SPYRE_LX_PLANNER_RELAYOUT=0 to disable these optional optimizations, not
# ownership validation. This does not change the allocator or LX memory budget.
lx_planner_relayout: bool = os.getenv("SPYRE_LX_PLANNER_RELAYOUT", "1").lower() in (
    "1",
    "true",
    "yes",
)

# How many destination views the CP-SAT relayout enumeration keeps per
# (source, consumer) edge, cheapest first: a consumer with many equal-core
# divisions induces one distinct destination partition (one relayout copy the
# solver must place) per division, though it will read through at most one.
# On the spyre_attn decode graph with 16 unrolled KV blocks the unbounded
# enumeration built 5789 copies and CP-SAT's presolve outlived the time limit.
# 0 keeps every view.
lx_solver_relayout_groups_per_edge: int = int(
    os.getenv("SPYRE_LX_SOLVER_RELAYOUT_GROUPS_PER_EDGE", "4")
)

# Skip CP-SAT's presolve above this many free relayout copies in one solve;
# 0 (the default) never skips it, priced or not. Presolve once scaled
# super-linearly in the number of free copy residency literals (measured on the
# spyre_attn decode graph: 16 copies 5 s, 64 copies 13 s, 160 copies 40 s, 312
# copies past the 120 s limit). Constant-binding single-division copies and the
# cost printer's lin_max proxy variables removed that cost, and a priced model
# searched without presolve can exhaust memory in the LNS workers. Kept as an
# escape hatch for a graph where presolve still outlives the time limit.
lx_solver_relayout_presolve_max_copies: int = int(
    os.getenv("SPYRE_LX_SOLVER_RELAYOUT_PRESOLVE_MAX_COPIES", "0")
)

# Submit independent DXP kernel compilations to Inductor's subprocess pool and
# resolve them together at the generated wrapper's async_compile.wait() barrier.
# This is opt-in while the parallel path is evaluated on full model compiles.
async_dxp_compile: bool = os.getenv("SPYRE_ASYNC_DXP_COMPILE", "0").lower() in (
    "1",
    "true",
    "yes",
)

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

# Temporary kill switch for removing a proven-redundant read copy after LX
# planning.  A failed proof leaves the original graph unchanged.
read_copy_elision: bool = os.getenv("SPYRE_READ_COPY_ELISION", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Per-pass operation logging for CustomPreSchedulingPasses.
# Set to "all" or "1" to log after every pass, or a comma-separated list of
# pass function names (e.g., "split_multi_ops,insert_restickify") to log only
# after specific passes. Set via SPYRE_LOG_PASSES env var or programmatically.
log_passes: str = os.environ.get("SPYRE_LOG_PASSES", "")

# Structured per-compile timing records (timing_recorder.py).  Off by default;
# when off, a timed region records nothing and allocates no event, but the call
# site still builds its name and keyword arguments and this flag is still read,
# so it is cheap rather than free: measured at ~1.4 us per region off and ~3.9 us
# on (of which ~1.0 us is reading this flag through install_config_module, the
# same cost log_passes already pays per pass).  A compile emitting 132 regions
# pays ~0.2 ms off, ~0.5 ms on.  Records go to timing_out at process exit, or via
# timing_recorder.dump_and_finalize() for callers that want them sooner.
# Tests override with config.patch({"timing": True}) rather than the environment.
timing: bool = os.getenv("TORCH_SPYRE_TIMING", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Destination for the timing record.  The pid is inserted before the suffix, so
# one setting is safe when a run fans out into several processes.  Empty means
# keep the events in memory and write nothing, which is what a caller reading
# timing_recorder.RECORDER directly wants.
timing_out: str = os.environ.get("TORCH_SPYRE_TIMING_OUT", "")

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
# Append one JSON record per co-optimized graph to this file: the symbolic cost
# objective the solver minimized (per-bundle terms and relayout charges as sympy
# ``srepr`` strings), the symbol values the solve chose, and each term evaluated
# under them. Read by the summarize-sdsc skill. Empty = off.
dump_cost_expr_file: str = os.environ.get("SPYRE_DUMP_COST_EXPR_FILE", "")

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

# Cache and reuse sdsc.json files during codegen when two OpSpecs produce
# identical SuperDSC content, reducing bundle size for programs with loops.
# Set SPYRE_INDUCTOR_SDSC_CACHE=0 to disable.
sdsc_cache: bool = os.environ.get("SPYRE_INDUCTOR_SDSC_CACHE", "1") == "1"

# Layout solver class used by default in scratchpad.allocator.ScratchpadAllocator.
# Options:
#  "greedy":       GreedyLayoutSolver,
#  "bestfit":      BestFitLayoutSolver,
#  "firstfit":     FirstFitLayoutSolver,
#  "simulated_annealing":  SimulatedAnnealingLayoutSolver, or -- when
#              ``co_optimizing_lx_planning`` is set -- SaCoOptimizingSolver, the
#              joint work-division + LX-placement annealer. Two different
#              solvers sharing one config value, not one solver in two modes.
#  "cpsat":    CpSatLayoutSolver (OR-Tools CP-SAT joint core-division +
#              LX placement, minimizing HBM transfer traffic) (default).
#
# For "cpsat" and "simulated_annealing" the value names a solver *family* whose
# joint-ness is selected by ``co_optimizing_lx_planning``; for the gap-based
# solvers that same flag instead wraps them in ExhaustiveSearchSolver.

layout_solver: Literal[
    "greedy", "bestfit", "firstfit", "cpsat", "simulated_annealing"
] = os.environ.get("LAYOUT_SOLVER", "cpsat")  # type: ignore[assignment]

# Wall-clock budget for one CP-SAT solve, in seconds. The joint objective is
# lexicographic and re-solves the same model up to three times (residency, then
# parallelism, then division balance), so this bounds each phase, not the pass.
# It is a compile-time guard, not a correctness one: a solve that runs out of
# budget without an incumbent raises SolveError, and scratchpad_planning falls
# back to greedy placement (correct, but co-optimization is lost for that
# graph). Raise it if large graphs are falling back; 0 disables the limit.
# The default matches the budget CpSatLayoutSolver hard-coded before this knob
# existed, so exposing it does not change how long any solve is allowed to run.
cpsat_time_limit_seconds: float = float(
    os.environ.get("CPSAT_TIME_LIMIT_SECONDS", "30")
)

# OpSpec validation at pipeline stage boundaries. Enabled by default to catch
# invariant violations early. Set SPYRE_VALIDATE_OP_SPECS=0 to disable.
validate_op_specs: bool = os.environ.get("SPYRE_VALIDATE_OP_SPECS", "1") == "1"

# Use the C++ (native) permutation-layout packer accelerator, which both
# simulated-annealing solvers drive (the layout-only one and the joint
# co-optimizer). The native and Python packers are behaviourally identical
# (verified bit-for-bit); the native one is faster. Set
# False (or ``TORCH_SPYRE_NATIVE_PACKER=0``/``false``, which backs this default)
# to force the pure-Python packer. A missing native class is a stale or
# incomplete build, not a supported mode, and raises rather than falling back.
native_layout_packer: bool = os.getenv("TORCH_SPYRE_NATIVE_PACKER", "1").lower() in (
    "1",
    "true",
    "yes",
)

# When symbolic cost_expr fails, use the fallback cost instead of erroring out
_cpsat_warn_on_cost_expr: bool = True
# Enable persistent on-disk caching of compiled Spyre kernels across
# invocations.
# Set SPYRE_KERNEL_CACHE=0 to disable.
# To force recompilation (bypass lookup but still save), use the standard
# PyTorch flag: TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 / set
# torch._inductor.config.force_disable_caches = True.
spyre_kernel_cache: bool = os.environ.get("SPYRE_KERNEL_CACHE", "0") == "1"

install_config_module(sys.modules[__name__])
