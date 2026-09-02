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

from pathlib import Path
import json

from enum import Enum
from pydantic import BaseModel, ValidationError
from typing import List


class Dtype(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class TensorSpec(BaseModel):
    ndims: int
    dimensions: List[int]  # has to be opened-up for symbolic
    dtype: Dtype


class IOSpec(BaseModel):
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]


def parse_json_file(iofile):
    iospec = Path(iofile)
    if not iospec.exists():
        raise FileNotFoundError(f"No file found: {iospec}")

    if not iospec.suffix == ".json":
        raise ValueError(f"IO Spec file is not a json: {iospec}")

    try:
        with open(iospec, "r") as f:
            iospec_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in IO Spec file: {e}") from e

    try:
        spec = IOSpec(**iospec_data)
    except ValidationError as e:
        raise ValueError(f"IO Spec validation failed: {e}") from e

    if len(spec.outputs) == 0:
        raise ValueError(f"No output found in IO Spec. One or more output is expected.")

    return spec
