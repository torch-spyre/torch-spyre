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

"""Layout metadata helpers tested without loading the Spyre runtime."""

import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def layouts(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "torch_spyre" / "layout.py"
    spec = importlib.util.spec_from_file_location("layout_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DataFormats(Enum):
        FP16 = 1

    class ElementArrangement(Enum):
        CONTIGUOUS = 1

    def layout(
        device_size, stride_map, device_dtype, valid_elements, element_arrangement
    ):
        return SimpleNamespace(
            device_size=device_size,
            stride_map=stride_map,
            device_dtype=device_dtype,
            valid_elements=valid_elements,
            element_arrangement=element_arrangement,
        )

    runtime = ModuleType("torch_spyre._C")
    runtime.DataFormats = DataFormats
    runtime.ElementArrangement = ElementArrangement
    runtime.SpyreTensorLayout = layout
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    return module, layout(
        [2, 4], [4, 1], DataFormats.FP16, {}, ElementArrangement.CONTIGUOUS
    )


@pytest.mark.parametrize("valid_elements", [{}, {(0,): 2, (1, 2): 7}])
def test_layout_round_trip(layouts, valid_elements):
    module, layout = layouts
    layout.valid_elements = valid_elements
    encoded = json.loads(json.dumps(module.layout_to_json(layout)))
    assert encoded == {
        "device_size": [2, 4],
        "stride_map": [4, 1],
        "device_dtype": "FP16",
        "element_arrangement": "CONTIGUOUS",
        "valid_elements": [
            [list(dims), count] for dims, count in valid_elements.items()
        ],
    }
    assert module.layout_from_json(encoded) == layout


def test_legacy_layout_without_valid_elements(layouts):
    module, layout = layouts
    encoded = module.layout_to_json(layout)
    del encoded["valid_elements"]
    assert module.layout_from_json(encoded) == layout


@pytest.mark.parametrize("field", ["device_dtype", "element_arrangement"])
def test_unknown_layout_enum_is_rejected(layouts, field):
    module, layout = layouts
    encoded = module.layout_to_json(layout)
    encoded[field] = "unknown"
    with pytest.raises(KeyError):
        module.layout_from_json(encoded)
