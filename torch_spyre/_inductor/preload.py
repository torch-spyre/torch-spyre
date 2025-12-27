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

# This module is loaded synchronously when the torch module is loaded,
# while the other Spyre inductor modules are loaded lazily

import torch
from torch._inductor.decomposition import decompositions

# Exclude specific Inductor default decompositions on Spyre.
#
# Some Inductor decompositions do not work reliably on the Spyre backend yet.
# We disable them here and rely on implicit fallbacks to eager ops instead. Once
# the blocking issues are resolved, these exclusions can be removed.
decomps_to_exclude = [
    # The default decomposition for torch.arange (defined in pytorch/torch/refs/__init__.py)
    # uses prims.iota, which is non-trivial to implement on Spyre at the moment.
    # See: https://github.com/torch-spyre/torch-spyre/pull/178#issuecomment-3610709413
    torch.ops.aten.arange.default,
    torch.ops.aten.arange.out,
    torch.ops.aten.arange.start,
    torch.ops.aten.arange.start_step,
    torch.ops.aten.arange.start_out,
    # The default decomposition for torch.new_ones (defined in pytorch/torch/refs/__init__.py)
    # uses torch.full, which is not yet supported in Spyre eager mode.
    # See: https://github.com/torch-spyre/torch-spyre/issues/128#issuecomment-3576168221
    torch.ops.aten.new_ones,
]

# Remove the selected decompositions from Inductor's registry for Spyre.
torch._decomp.remove_decompositions(decompositions, decomps_to_exclude)
