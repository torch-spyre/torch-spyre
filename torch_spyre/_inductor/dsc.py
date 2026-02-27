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

from typing import Any

from torch._inductor.utils import IndentedBuffer
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.utils import (
    get_kernel_metadata,
    get_fused_kernel_name,
)
from torch._inductor.virtualized import V

from .spyre_kernel import SpyreKernel


class SuperDSCScheduling(SIMDScheduling):
    kernel_type: type[Any] = SpyreKernel
    dsc_type: str = "sdsc"

    def define_kernel(self, src_code, node_schedule, kernel):
        """Codegen kernel definition to go in output wrapper code"""
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = get_fused_kernel_name(node_schedule, "original_aten")
            kernel_name = "_".join(
                [self.dsc_type, fused_name, wrapper.next_kernel_suffix()]
            )
            wrapper.src_to_kernel[src_code] = kernel_name
            buf = IndentedBuffer()
            buf.writeline(f"async_compile.{self.dsc_type}('{kernel_name}',")
            with buf.indent():
                buf.splice(f"{src_code}")
            buf.writeline(")")
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(kernel_name, buf.getvalue(), metadata_comment)

        return kernel_name
