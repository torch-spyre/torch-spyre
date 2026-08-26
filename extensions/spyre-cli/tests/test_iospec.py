import json
import pytest
from pathlib import Path

from spyre_cli.iospec.iospec import Dtype, TensorSpec, IOSpec, parse_json_file


def test_tensor_spec_from_string_dtype():
    spec = TensorSpec(ndims=3, dimensions=[1, 2, 3], dtype="float32")
    assert spec.dtype == Dtype.FLOAT32


def test_parse_json_file_valid(tmp_path: Path):
    iofile = tmp_path / "io.json"
    data = {
        "inputs": [{"ndims": 2, "dimensions": [10, 1024], "dtype": "float16"}],
        "outputs": [{"ndims": 2, "dimensions": [512, 1024], "dtype": "float32"}],
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
