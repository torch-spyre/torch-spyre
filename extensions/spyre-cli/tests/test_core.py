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

import pytest

from spyre_cli.core import create_tensor_info


def test_create_tensor_info_with_dtype():
    dims, dtype = create_tensor_info("10x1024@fp16")
    assert dims == [10, 1024]
    assert dtype == "float16"


def test_create_tensor_info_fp32():
    dims, dtype = create_tensor_info("512x1024@fp32")
    assert dims == [512, 1024]
    assert dtype == "float32"


def test_create_tensor_info_bf16():
    dims, dtype = create_tensor_info("8x8@bf16")
    assert dims == [8, 8]
    assert dtype == "bfloat16"


def test_create_tensor_info_no_dtype_defaults_to_fp16():
    dims, dtype = create_tensor_info("10x1024")
    assert dims == [10, 1024]
    assert dtype == "float16"


def test_create_tensor_info_single_dimension():
    dims, dtype = create_tensor_info("128@fp16")
    assert dims == [128]
    assert dtype == "float16"


def test_create_tensor_info_multiple_dimensions():
    dims, dtype = create_tensor_info("2x3x4x5@fp32")
    assert dims == [2, 3, 4, 5]
    assert dtype == "float32"


def test_create_tensor_info_scalar_no_dims():
    with pytest.raises(ValueError, match="Found no dimensions"):
        create_tensor_info("@fp16")


def test_create_tensor_info_too_many_at_signs():
    with pytest.raises(ValueError, match="Expected a single @"):
        create_tensor_info("10x1024@fp16@extra")


def test_create_tensor_info_unknown_dtype():
    with pytest.raises(ValueError, match="Unexpected dtype"):
        create_tensor_info("10x1024@int8")


def test_create_tensor_info_non_integer_dimension():
    with pytest.raises(ValueError, match="Found non integer dimension"):
        create_tensor_info("10xabc@fp16")
