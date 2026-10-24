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

# Owner(s): ["module: spyre"]

"""Tests for SPYRE_FAST_WEIGHT_DMA (pre-encoded fp16 weight upload).

The fast path must leave exactly the same device bytes as the regular fp16
DMA. Device contents are compared twice: through the normal fp16 readback,
and byte for byte through an int8 alias of the device storage (the int8 DMA
is a bit-exact identity). The encoder tests at the top need no hardware.

Run with:
    python -m pytest tests/tensor/test_fast_weight_dma.py -v
"""

import os
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch_spyre import model_utils
from torch_spyre._C import fp16_to_sen169_bits


def _all_fp16_patterns() -> torch.Tensor:
    return torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)


def _table_lookup(x: torch.Tensor) -> torch.Tensor:
    """Element-wise lookup in the encoder's own output for all fp16 patterns."""
    table = fp16_to_sen169_bits(_all_fp16_patterns())
    idx = x.reshape(-1).view(torch.int16).to(torch.int64) & 0xFFFF
    return table[idx].reshape(x.shape)


def _bits(x: torch.Tensor) -> list[int]:
    return [v & 0xFFFF for v in x.view(torch.int16).reshape(-1).tolist()]


class TestFp16ToSen169Bits(TestCase):
    """Host encoder; no hardware needed."""

    def test_known_values(self):
        # fp16 bits -> DLF16 bits, from DeepTools' conversion contract; the same
        # bits the regular DMA path writes on hardware
        cases = {
            0x0000: 0x0000,  # +0
            0x8000: 0x8000,  # -0
            0x0001: 0x0E00,  # smallest subnormal, normalized
            0x03FF: 0x21FF,  # largest subnormal
            0x0400: 0x2200,  # smallest normal
            0x3C00: 0x3E00,  # 1.0
            0xBC00: 0xBE00,  # -1.0
            0x3C01: 0x3E01,  # half-way rounds up
            0x3555: 0x3AAB,
            0x7BFF: 0x5DFF,  # 65504 saturates
            0xFBFF: 0xDDFF,
            0x7C00: 0x7E00,  # inf
            0xFC00: 0xFE00,  # -inf
            0x7C01: 0x7FFF,  # nan
            0x7E00: 0x7FFF,
            0xFE00: 0xFFFF,
        }
        src = torch.tensor(list(cases), dtype=torch.int32).to(torch.int16)
        out = fp16_to_sen169_bits(src.view(torch.float16))
        self.assertEqual(out.dtype, torch.int16)
        self.assertEqual(_bits(out), list(cases.values()))

    def test_large_and_noncontiguous(self):
        # large enough to be split over several threads; strided input too
        x = (torch.randn(2048, 3001) * 0.05).half()
        x.view(-1)[:65536] = _all_fp16_patterns()
        ref = _table_lookup(x)
        self.assertTrue(torch.equal(fp16_to_sen169_bits(x), ref))
        self.assertTrue(torch.equal(fp16_to_sen169_bits(x.t()), ref.t()))

    def test_rejects_other_dtypes(self):
        with self.assertRaises(RuntimeError):
            fp16_to_sen169_bits(torch.zeros(4, dtype=torch.bfloat16))


def _spyre_available() -> bool:
    try:
        torch.zeros(1, dtype=torch.float16, device="spyre")
        return True
    except Exception:
        return False


requires_spyre = unittest.skipUnless(_spyre_available(), "requires Spyre hardware")


def _raw_bytes(dev: torch.Tensor) -> torch.Tensor:
    """Device bytes of fp16 ``dev`` (logical extent) via an int8 alias."""
    from torch_spyre._C import (
        DataFormats,
        SpyreTensorLayout,
        get_spyre_tensor_layout,
        reinterpret_tensor_with_layout,
    )

    # A 128-byte stick of 64 fp16 elements holds 128 int8 elements; every
    # other device dim keeps its size, with its host stride doubled.
    fp16 = get_spyre_tensor_layout(dev)
    assert fp16.device_size[-1] == 64 and fp16.stride_map[-1] == 1
    strides = [s if s == -1 else 2 * s for s in fp16.stride_map[:-1]] + [1]
    sizes = list(fp16.device_size[:-1]) + [128]
    stl = SpyreTensorLayout(sizes, strides, DataFormats.SENINT8)
    shape = list(dev.shape[:-1]) + [dev.shape[-1] * 2]
    stride = torch.empty(shape, dtype=torch.int8).stride()
    alias = torch.empty(0, dtype=torch.int8, device=dev.device)
    alias.set_(dev.untyped_storage(), 0, shape, stride)
    return reinterpret_tensor_with_layout(alias, shape, stride, 0, stl).cpu()


def _upload(fn, x, fast: bool, **kw):
    with mock.patch.dict(
        os.environ, {model_utils.FAST_WEIGHT_DMA_ENV: "1" if fast else "0"}
    ):
        return fn(x, **kw)


def _weights(shape) -> torch.Tensor:
    x = (torch.randn(shape) * 0.05).half()
    flat = x.view(-1)
    n = min(flat.numel(), 65536)
    flat[:n] = _all_fp16_patterns()[torch.randperm(65536)[:n]]
    return x


@instantiate_parametrized_tests
class TestFastWeightDma(TestCase):
    def setUp(self):
        torch.manual_seed(0xFA57)

    def _assert_same_device_bytes(self, old, new, check_raw=True):
        self.assertEqual(old.shape, new.shape)
        self.assertTrue(
            torch.equal(old.cpu().view(torch.int16), new.cpu().view(torch.int16))
        )
        if check_raw:
            self.assertTrue(torch.equal(_raw_bytes(old), _raw_bytes(new)))

    @requires_spyre
    def test_encoder_matches_runtime_on_all_patterns(self):
        x = _all_fp16_patterns().reshape(1024, 64)
        old = _upload(model_utils._dma_to_spyre_default, x, fast=False)
        raw = _raw_bytes(old).view(torch.int16)
        self.assertTrue(torch.equal(raw, fp16_to_sen169_bits(x)))

    @requires_spyre
    @parametrize(
        "shape",
        [
            (1024, 64),
            (4096, 4096),
            (4096, 1024),
            (1024, 4096),
            (4096, 12800),
            (12800, 4096),
            (4096, 1000),  # last dim not a multiple of 64
            (49155, 4096),
            (50, 10),
            (7, 3),  # odd last dim
            (2, 30, 100),
            (2, 30, 128),
            (4096,),
            (4100,),
            (33,),
        ],
    )
    def test_default_layout(self, shape):
        x = _weights(shape)
        old = _upload(model_utils._dma_to_spyre_default, x, fast=False)
        new = _upload(model_utils._dma_to_spyre_default, x, fast=True)
        self._assert_same_device_bytes(old, new)

    @requires_spyre
    @parametrize("shape", [(1024, 64), (2000, 4096), (49155, 4096), (1000, 192)])
    def test_indirect_access_layout(self, shape):
        x = _weights(shape)
        old = _upload(model_utils._dma_to_spyre_indirect_access, x, fast=False)
        new = _upload(model_utils._dma_to_spyre_indirect_access, x, fast=True)
        self._assert_same_device_bytes(old, new)

    @requires_spyre
    def test_moe_layouts(self):
        x = _weights((4, 256, 192))
        old = _upload(model_utils.dma_moe_expert_weight_to_spyre, x, fast=False)
        new = _upload(model_utils.dma_moe_expert_weight_to_spyre, x, fast=True)
        self._assert_same_device_bytes(old, new)
        s = _weights((8,))
        old = _upload(model_utils.dma_moe_per_expert_scale_to_spyre, s, fast=False)
        new = _upload(model_utils.dma_moe_per_expert_scale_to_spyre, s, fast=True)
        self._assert_same_device_bytes(old, new)

    @requires_spyre
    @parametrize(
        "shape,image",
        [
            ((4096, 4096), True),  # unpadded: verbatim device image
            ((2, 30, 128), True),
            ((4096,), True),
            ((4096, 1000), False),  # stick padding: regular DMA
            ((7, 3), False),  # odd last dim: regular DMA
        ],
    )
    def test_path_selection(self, shape, image):
        x = _weights(shape)
        with (
            mock.patch.object(
                model_utils,
                "copy_bytes_to_device",
                wraps=model_utils.copy_bytes_to_device,
            ) as img,
            mock.patch.object(
                model_utils, "copy_tensor", wraps=model_utils.copy_tensor
            ) as dma,
        ):
            new = _upload(model_utils._dma_to_spyre_default, x, fast=True)
        self.assertEqual(img.call_count, 1 if image else 0)
        self.assertEqual(dma.call_count, 0 if image else 1)
        old = _upload(model_utils._dma_to_spyre_default, x, fast=False)
        self._assert_same_device_bytes(old, new)

    @requires_spyre
    def test_fallbacks_unchanged(self):
        # dim_order=[1, 0] (stick across rows), bf16 and fp32->fp16 all keep
        # the regular DMA; results must not change with the switch on.
        w = _weights((256, 384))
        with mock.patch.object(model_utils, "fp16_to_sen169_bits") as enc:
            old = _upload(model_utils._dma_to_spyre_dim_order_swapped, w, fast=False)
            new = _upload(model_utils._dma_to_spyre_dim_order_swapped, w, fast=True)
            self._assert_same_device_bytes(old, new, check_raw=False)
            b = torch.randn(64, 128, dtype=torch.bfloat16)
            old = _upload(model_utils._dma_to_spyre_default, b, fast=False)
            new = _upload(model_utils._dma_to_spyre_default, b, fast=True)
            self.assertTrue(torch.equal(old.cpu(), new.cpu()))
            f = torch.randn(64, 128)
            kw = {"target_dtype": torch.float16}
            old = _upload(model_utils._dma_to_spyre_default, f, fast=False, **kw)
            new = _upload(model_utils._dma_to_spyre_default, f, fast=True, **kw)
            self.assertTrue(torch.equal(old.cpu(), new.cpu()))
            enc.assert_not_called()

    @requires_spyre
    def test_load_model_to_spyre(self):
        torch.manual_seed(1)
        model = torch.nn.Sequential(
            torch.nn.Embedding(1000, 128),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 320),
        ).half()
        params = {k: v.detach().clone() for k, v in model.named_parameters()}
        with mock.patch.dict(os.environ, {model_utils.FAST_WEIGHT_DMA_ENV: "1"}):
            model_utils.load_model_to_spyre(model)
        for k, v in model.named_parameters():
            ref = _upload(model_utils._dma_to_spyre_default, params[k], fast=False)
            self.assertTrue(
                torch.equal(v.cpu().view(torch.int16), ref.cpu().view(torch.int16)), k
            )


if __name__ == "__main__":
    run_tests()
