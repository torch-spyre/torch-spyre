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

"""JSON-compatible descriptions of Spyre physical tensor layouts.

No checkpoint format, file I/O or transfer scheduling is defined here.
"""

from collections.abc import Mapping
from typing import Any


def layout_to_json(layout) -> dict:
    return {
        "device_size": list(layout.device_size),
        "stride_map": list(layout.stride_map),
        "device_dtype": layout.device_dtype.name,
        "element_arrangement": layout.element_arrangement.name,
        "valid_elements": [
            [list(dims), count] for dims, count in layout.valid_elements.items()
        ],
    }


def layout_from_json(d: Mapping[str, Any]):
    from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout

    return SpyreTensorLayout(
        device_size=list(d["device_size"]),
        stride_map=list(d["stride_map"]),
        device_dtype=DataFormats.__members__[d["device_dtype"]],
        valid_elements={
            tuple(dims): count for dims, count in d.get("valid_elements", [])
        },
        element_arrangement=ElementArrangement.__members__[d["element_arrangement"]],
    )
