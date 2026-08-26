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

    return spec
