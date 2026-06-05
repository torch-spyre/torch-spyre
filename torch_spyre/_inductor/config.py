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

dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

sencores: int = int(os.getenv("SENCORES", "32"))

# For K-split matmuls, permute physical core IDs so the cores collaborating on a
# K reduction land on adjacent ring positions, cutting PSUM chain hops from m*n
# to 1. The split itself is chosen by the cost-model planner; this only reorders
# cores at SDSC emission. Set SPYRE_CORE_ID_K_FAST_EMISSION=0 to disable.
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)

# When True, HBM tensor addresses are emitted as runtime symbols (%sym_N
# constants) in bundle.mlir and resolved via affine.apply for tiled loops.
# Requires backend compiler support for the sdscbundle symbol table, which is
# still under development.
bundle_hbm_symbols: bool = os.environ.get("BUNDLE_HBM_SYMBOLS", "0") == "1"

# When True (default), LoopSpec nodes are fully unrolled into flat OpSpecs
# before generate_bundle runs.  Set to False to pass LoopSpecs through intact
# (used with bundle_hbm_symbols=True for the scf.for / affine.apply path).
unroll_loops: bool = os.environ.get("UNROLL_LOOPS", "1") == "1"

# Layout solver class used by default in scratchpad.allocator.DefaultAllocator.
# Options:
#  "greedy":   GreedyLayoutSolver (default),
#  "bestfit":  BestFitLayoutSolver,
#  "firstfit": FirstFitLayoutSolver.

# TODO(isuruf): Change to firstfit when deeptools PR4298 lands
layout_solver: Literal["greedy", "bestfit", "firstfit"] = "greedy"

install_config_module(sys.modules[__name__])
