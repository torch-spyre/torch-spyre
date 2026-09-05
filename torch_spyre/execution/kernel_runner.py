# Copyright 2025-2026 The Torch-Spyre Authors.
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

import torch
from torch_spyre._C import (
    SymbolicArg,
    SymbolicArgKind,
    launch_jobplan,
    prepare_kernel,
    register_kernel_provenance,
)
from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.kernel_provenance import KernelProvenanceDescriptor
from torch_spyre._inductor.profiler_event import (
    format_kernel_provenance_event_name,
)
from torch_spyre.profiler._ffdc import (
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    with_ffdc,
)

logger = get_inductor_logger("kernel_runner")


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    @with_ffdc(CATEGORY_UNIMPLEMENTED, logger, code_dir_attr=None)
    def run(self, *args, **kw_args):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains"
            f" unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    def __init__(
        self,
        name: str,
        code_dir: str,
        kernel_provenance: KernelProvenanceDescriptor | None = None,
        symbol_kinds: list[SymbolKind] | None = None,
    ):
        self.kernel_name = name
        self.code_dir = code_dir
        self.kernel_provenance = kernel_provenance
        # Canonical symbol order returned by generate_bundle()
        self.symbol_kinds: list[SymbolKind] = (
            symbol_kinds if symbol_kinds is not None else []
        )
        self.profiler_event_name: str | None
        spyrecode_dir = code_dir + "/spyreCodeDir"
        if kernel_provenance is None:
            self.profiler_event_name = None
            self.jobplan = prepare_kernel(spyrecode_dir)
        else:
            self.profiler_event_name = format_kernel_provenance_event_name(
                kernel_provenance
            )
            # Rejection is intentionally fail-open: C++ warns and counts
            # conflicts while the key-bearing name remains the compatibility
            # join.
            register_kernel_provenance(
                self.profiler_event_name,
                list(kernel_provenance.debug_handle_ids),
            )
            with torch.profiler.record_function(f"prepare_kernel:{self.kernel_name}"):
                self.jobplan = prepare_kernel(
                    spyrecode_dir,
                    profiler_name=self.profiler_event_name,
                )

    @with_ffdc(CATEGORY_RUNTIME_LAUNCH, logger)
    def run(self, *args, **kw_args):
        logger.info("RUN: %s %s", self.kernel_name, self.code_dir)
        with torch.profiler.record_function(f"launch_jobplan:{self.kernel_name}"):
            if _spyre_config.bundle_symbolic_args and self.symbol_kinds:
                # Build the SymbolicArg payload from the canonical symbol order
                # that generate_bundle() returned and stored on this runner.
                # symbol_kinds matches the MLIR input_arg slot order: pool first
                # (when frontend_pool_allocation is active), then kernel tensor
                # args in arg_index order.
                if self.symbol_kinds[0].is_pool:
                    # call_kernel prepends the pool tensor to args, so it sits
                    # at args[0]. Kernel tensor arg_indices are 0-based among
                    # kernel tensors only, so add 1 to account for the pool.
                    symbolic_args = (
                        [SymbolicArg(kind=SymbolicArgKind.kAddress, tensor_id=0)]
                    ) + (
                        [
                            SymbolicArg(
                                kind=SymbolicArgKind.kAddress,
                                tensor_id=sk.arg_index + 1,
                            )
                            for sk in self.symbol_kinds[1:]
                        ]
                    )
                else:
                    # No pool param — arg_index maps directly to args position.
                    symbolic_args = [
                        SymbolicArg(
                            kind=SymbolicArgKind.kAddress, tensor_id=sk.arg_index
                        )
                        for sk in self.symbol_kinds
                    ]
                launch_jobplan(self.jobplan, args, symbolic_args)
            else:
                launch_jobplan(self.jobplan, args)
