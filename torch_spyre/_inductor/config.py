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
from typing import Callable, Optional

from torch.utils._config_module import install_config_module

lx_planning: bool = os.environ.get("LX_PLANNING", "0") == "1"
chunk_large_tensors: bool = os.environ.get("CHUNK_LARGE_TENSORS", "0") == "1"

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

allow_all_ops_in_lx_planning: bool = False

dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

sencores: int = int(os.getenv("SENCORES", "32"))

# k_fast: a two-layer optimisation for K-split matmul work-divisions.
#   Layer 1 (planner, core_division.py): picks (1, n, k>1) over pure-M
#     for narrow-N small-M matmul shapes that would otherwise leave the
#     PT array under-utilised.
#   Layer 2 (SDSC emitter, codegen/compute_ops.py): permutes physical
#     core IDs so K-collaborators land on adjacent ring positions,
#     reducing PSUM chain hops from m*n to 1.
# Set SPYRE_CORE_ID_K_FAST_EMISSION=0 to disable both layers.
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)

coarse_tiling: bool = os.environ.get("COARSE_TILING", "0") == "1"

# When True, HBM tensor addresses are emitted as runtime symbols (%sym_N
# constants) in bundle.mlir and resolved via affine.apply for tiled loops.
# Requires backend compiler support for the sdscbundle symbol table, which is
# still under development.
bundle_hbm_symbols: bool = os.environ.get("BUNDLE_HBM_SYMBOLS", "0") == "1"

# When True (default), LoopSpec nodes are fully unrolled into flat OpSpecs
# before generate_bundle runs.  Set to False to pass LoopSpecs through intact
# (used with bundle_hbm_symbols=True for the scf.for / affine.apply path).
unroll_loops: bool = os.environ.get("UNROLL_LOOPS", "1") == "1"

# Optional callable injected by callers to compute coarse-tiling groups.
# Signature: (list[Operation]) -> list[tuple[list[Operation], sympy.Expr[, list[int]]]]
# Each tuple is (ops, loop_count) or (ops, loop_count, tiled_dims).
# tiled_dims overrides the default per-group (None = tile outermost dim only).
# When None and coarse_tiling is True, coarse_tile() is called with groups=[]
# (a no-op useful for testing the pipeline without real group detection).
# Must be a module-level named function (not a lambda) for Inductor cache pickling.
# This is intended to be used for interim testing of the coarse-tiling transformation
# until the working set reduction annotation framework is being developed.
# It will be removed once the full-fledged annotation mechanism is available.
coarse_tiling_groups_fn: Optional[Callable] = None

install_config_module(sys.modules[__name__])
