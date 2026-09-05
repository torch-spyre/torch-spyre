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

"""Compiler-only tensor layout hint plumbing."""

import torch

from .errors import Unsupported

REQUIRE_LAYOUT_KEY = "require_layout"
REQUESTS_KEY = "_spyre_require_layout_requests"


_VIEW_OPS = {
    torch.ops.aten.expand.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
}


def _producer(source: torch.fx.Node) -> torch.fx.Node:
    """Walk output-only views back to producer that must emit requested layout."""
    while source.target in _VIEW_OPS:
        source = source.args[0]
        if not isinstance(source, torch.fx.Node):
            raise TypeError("require_layout expects a tensor producer")
    return source


def apply_require_layout(graph: torch.fx.Graph) -> None:
    """Move static marker request onto producer and track its consumption."""
    requests = graph.__dict__.setdefault(REQUESTS_KEY, {})
    for node in list(graph.nodes):
        if node.target != torch.ops.spyre.require_layout.default:
            continue
        source, device_size, stride_map = node.args
        if not isinstance(source, torch.fx.Node):
            raise TypeError("require_layout expects a tensor producer")
        if not all(isinstance(v, int) for v in (*device_size, *stride_map)):
            raise TypeError("require_layout layout must be static")
        if len(device_size) != len(stride_map) or not device_size:
            raise ValueError(
                "require_layout device_size and stride_map must have equal nonzero lengths"
            )
        if any(extent <= 0 for extent in device_size):
            raise ValueError("require_layout device_size extents must be positive")
        source = _producer(source)
        custom = source.meta.setdefault("custom", {})
        geometry = (list(device_size), list(stride_map))
        previous = custom.get(REQUIRE_LAYOUT_KEY)
        if previous is not None and previous["geometry"] != geometry:
            raise ValueError("conflicting require_layout requests for one producer")
        if previous is None:
            request = {"geometry": geometry, "consumed": False}
            requests[id(request)] = request
            custom[REQUIRE_LAYOUT_KEY] = request
        node.replace_all_uses_with(node.args[0])
        graph.erase_node(node)


def assert_require_layout_consumed(graph) -> None:
    """Reject hints not honored by a layout-aware producer in this graph."""
    requests = graph.graph.__dict__.get(REQUESTS_KEY, {})
    pending = [request for request in requests.values() if not request["consumed"]]
    if pending:
        raise Unsupported("require_layout target has no supported compiled producer")
