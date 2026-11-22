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
"""Discrete Fourier transform on Spyre via matmul.

``torch_spyre.ops.fft`` / ``fft2`` / ``fftn`` run the DFT as a real matmul on
the AIU; complex values and the dtype cast stay on the host, only the matmul
runs on device. Large composite lengths automatically use the Cooley-Tukey
four-step factorization. Requires a Spyre device.
"""

import torch

import torch_spyre.ops as ops

torch.manual_seed(0xAFFE)

# Every ops.fft* call below runs the DFT matmul on the Spyre AIU
# (device="spyre" by default). Inputs and outputs are host complex64 tensors:
# the complex packing and the fp16 cast happen on the host, only the real
# matmul crosses to the device, and the result is returned on the host (CPU).


def rel_peak(got, ref):  # error relative to the spectrum peak (fp16: ~0.5%)
    return (got - ref).abs().max().item() / ref.abs().max().item()


# 1-D FFT: a length-n complex DFT as one (2n, 2n) real matmul on device.
x = torch.randn(4, 256, dtype=torch.complex64)
print("fft    ", rel_peak(ops.fft(x), torch.fft.fft(x)))

# 2-D FFT: separable -- a 1-D FFT (device matmul) along each of the last two axes.
img = torch.randn(32, 48, dtype=torch.complex64)
print("fft2   ", rel_peak(ops.fft2(img), torch.fft.fft2(img)))

# A length stays one direct (2n, 2n) matmul up to the ceiling (default 32768,
# TORCH_SPYRE_FFT_DIRECT_MAX); above it the transform is factored automatically
# into two smaller device DFTs plus a host twiddle (Cooley-Tukey four-step) so
# it still fits. This 8192 case is still a single direct matmul.
big = torch.randn(8192, dtype=torch.complex64)
print("fft 8k ", rel_peak(ops.fft(big), torch.fft.fft(big)))
