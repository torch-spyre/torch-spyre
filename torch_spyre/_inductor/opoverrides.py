# Copyright 2025 The Torch-Spyre Authors.
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

from torch._inductor.codegen.common import OpOverrides


class SpyreKernelOverrides(OpOverrides):
    """
    Additional ops that are defined for the Spyre device.

    We don't actually use the strings returned from these methods for anything;
    the only thing that is significant is that the method is defined.

    Keep these ops sorted in alphabetical order!
    """

    @staticmethod
    def abs(x):
        return f"spyre.abs({x})"

    @staticmethod
    def clamp(input, min=None, max=None):
        return f"spyre.clamp({input} {min} {max})"

    @staticmethod
    def exp(x):
        return f"spyre.exp({x})"

    @staticmethod
    def exx2(a, b, c):
        return f"spyre.exx2({a} {b} {c})"

    @staticmethod
    def fma(x):
        return f"spyre.fma({x})"

    @staticmethod
    def gelu(x):
        return f"spyre.gelu({x})"

    @staticmethod
    def layernormnorm(a, b, c, d, e):
        return f"spyre.layernormnorm({a}, {b}, {c}, {d}, {e})"

    @staticmethod
    def layernormscale(x, y):
        return f"spyre.layernormscale({x}, {y})"

    @staticmethod
    def log(x):
        return f"spyre.log({x})"

    @staticmethod
    def reciprocal(x):
        return f"spyre.reciprocal({x})"

    @staticmethod
    def relu(x):
        return f"spyre.relu({x})"

    @staticmethod
    def rsqrt(x):
        return f"spyre.rsqrt({x})"

    @staticmethod
    def sigmoid(x):
        return f"spyre.sigmoid({x})"

    @staticmethod
    def softplus(x, y, z):
        return f"spyre.softplus({x}, {y}, {z})"

    @staticmethod
    def sqrt(x):
        return f"spyre.sqrt({x})"

    @staticmethod
    def tanh(x):
        return f"spyre.tanh({x})"

    @staticmethod
    def where(x, y, z):
        return f"spyre.where({x}, {y}, {z})"


SpyreKernelOverrides._initialize_pointwise_overrides("halide")
