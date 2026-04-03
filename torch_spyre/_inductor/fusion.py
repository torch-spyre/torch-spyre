# Copyright 2026 The Torch-Spyre Authors.
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

from torch._inductor.scheduler import BaseSchedulerNode, FusedSchedulerNode

from torch_spyre._inductor.logging_utils import _get_env_bool

_FUSION_ENABLED = _get_env_bool("SPYRE_INDUCTOR_ENABLE_FUSION")


def spyre_fuse_nodes(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    If fusion is enabled, put all nodes into a single SpyreKernel
    """
    if not _FUSION_ENABLED or len(nodes) == 0:
        return nodes

    return [FusedSchedulerNode(nodes[0].scheduler, nodes)]
