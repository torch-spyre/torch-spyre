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

from typing import Optional, Any, Callable, List

import torch
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_hash_for_files,
)

from .conversions import RemoveElementTypeConversions


custom_pre_passes: List[Callable[[torch.fx.graph.Graph], None]] = [
    RemoveElementTypeConversions(),
]
custom_post_passes: List[Callable[[torch.fx.graph.Graph], None]] = []


class CustomPrePasses(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in custom_pre_passes:
            p(graph)

    def uuid(self) -> Optional[Any]:
        files = [c.file() for c in custom_pre_passes]
        return get_hash_for_files(tuple(set(files + [__file__])))


class CustomPostPasses(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in custom_post_passes:
            p(graph)

    def uuid(self) -> Optional[Any]:
        files = [c.file() for c in custom_post_passes]
        return get_hash_for_files(tuple(set(files + [__file__])))
