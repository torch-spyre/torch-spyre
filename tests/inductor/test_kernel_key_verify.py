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

"""The transpose-edge tests again, with every stored kernel key checked.

With the kernel cache on and ``precomputed_kernel_key="verify"``, each sdsc()
call computes its key as well and raises PrecomputedKernelKeyMismatch if the
key stored at codegen differs. tests/test_precomputed_kernel_key.py covers a
few hand-built specs; this runs the kernel mix of a compile-heavy suite, so a
key input missing from kernel_key_fingerprint() fails here.
"""

import pytest

import torch_spyre._inductor.config as spyre_config
from test_inductor_transpose_edge import (
    TestTransposeDeviceSemantics as TransposeDeviceSemanticsBase,
    TestTransposeEdge as TransposeEdgeBase,
)


class KernelKeyVerify:
    @pytest.fixture(autouse=True)
    def verify_stored_kernel_keys(self):
        with spyre_config.patch(
            spyre_kernel_cache=True, precomputed_kernel_key="verify"
        ):
            yield


class TestTransposeEdgeKeyVerify(KernelKeyVerify, TransposeEdgeBase):
    pass


class TestTransposeDeviceSemanticsKeyVerify(
    KernelKeyVerify, TransposeDeviceSemanticsBase
):
    pass
