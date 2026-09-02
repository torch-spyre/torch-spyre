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

import json
import pytest
from pathlib import Path

from spyre_cli.iospec.iospec import Dtype, TensorSpec, IOSpec, parse_json_file


def test_tensor_spec_from_string_dtype():
    spec = TensorSpec(dimensions=[1, 2, 3], dtype="float32")
    assert spec.dtype == Dtype.FLOAT32


def test_parse_json_file_valid(tmp_path: Path):
    iofile = tmp_path / "io.json"
    data = {
        "inputs": [{"dimensions": [10, 1024], "dtype": "float16"}],
        "outputs": [{"dimensions": [512, 1024], "dtype": "float32"}],
    }
    iofile.write_text(json.dumps(data))

    spec = parse_json_file(str(iofile))
    assert isinstance(spec, IOSpec)
    assert spec.inputs[0].dimensions == [10, 1024]
    assert spec.outputs[0].dtype == Dtype.FLOAT32


def test_parse_json_file_missing_file(tmp_path: Path):
    iofile = tmp_path / "missing.json"
    with pytest.raises(Exception):
        parse_json_file(str(iofile))


def test_parse_json_file_not_json(tmp_path: Path):
    iofile = tmp_path / "io.txt"
    iofile.write_text("not json")
    with pytest.raises(Exception):
        parse_json_file(str(iofile))


def test_parse_json_file_invalid_data(tmp_path: Path):
    iofile = tmp_path / "io.json"
    iofile.write_text(json.dumps({"inputs": "bad", "outputs": []}))
    with pytest.raises(Exception):
        parse_json_file(str(iofile))


def test_parse_json_file_no_outputs(tmp_path: Path):
    iofile = tmp_path / "io.json"
    data = {
        "inputs": [{"dimensions": [10, 1024], "dtype": "float16"}],
        "outputs": [],
    }
    iofile.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="No output found in IO Spec"):
        parse_json_file(str(iofile))


def test_parse_json_file_e_incorrect():
    iofile = Path(__file__).parent.parent / "spyre_cli" / "iospec" / "e_incorrect.json"
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_json_file(str(iofile))
