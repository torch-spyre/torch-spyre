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

"""Drop Inductor's precision-cast emulation request from a Spyre graph.

Inductor's CPU and GPU kernels compute float16 and bfloat16 pointwise ops in
float32 and round once, at the end of a fused kernel. Eager rounds after every
op, so the two can differ in the last bits.
``torch._inductor.config.emulate_precision_casts`` asks Inductor to match eager:
tracing marks each low-precision pointwise node with
``meta["low_precision_pointwise_barrier"]``, and the pointwise lowering wraps a
marked node's inputs and output in a cast down to the low-precision dtype and
back up to the compute dtype.

A Spyre kernel has no wider compute dtype. Every op, fused or not, computes in
the tensor's own dtype, which is what Spyre eager does as well, so there is no
rounding to restore and the cast pair is the identity. It is also not
expressible: the pair is a conversion from a dtype to itself, and ``DtypeOpTable``
has no such operator.

So the marks are removed before lowering, and the lowering emits the plain op.
"""

import torch

BARRIER_META_KEY = "low_precision_pointwise_barrier"


def drop_precision_cast_barriers(graph: torch.fx.Graph) -> None:
    """Remove the precision-cast emulation mark from every node of ``graph``."""
    for node in graph.nodes:
        node.meta.pop(BARRIER_META_KEY, None)
