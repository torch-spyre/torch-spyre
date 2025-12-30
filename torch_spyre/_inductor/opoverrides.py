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

from typing import Any
from torch._inductor.ops_handler import OpsHandler


class SpyreBasicMathMixins:
    """
    This class replaces op_handlers.BasicMathOpsMixin.
    Keep methods in the same order as the upstream class.
    """

    @staticmethod
    def floordiv(a, b):
        return f"{a} // {b}"

    @staticmethod
    def mod(a, b):
        # careful, depending on target semantics varies
        return f"{a} % {b}"

    @staticmethod
    def lshift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def rshift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def and_(a, b):
        return f"{a} & {b}"

    @staticmethod
    def or_(a, b):
        return f"{a} | {b}"

    @staticmethod
    def xor(a, b):
        return f"{a} ^ {b}"


class SpyreOpFuncs(SpyreBasicMathMixins, OpsHandler[Any]):
    """
    Torch ops that are directly supported by the Spyre device.

    Keep these ops sorted in alphabetical order!
    """

    @staticmethod
    def abs(x):
        return f"spyre.abs({x})"

    @staticmethod
    def add(a, b):
        return f"{a} + {b}"

    @staticmethod
    def clamp(input, min=None, max=None):
        return f"spyre.clamp({input} {min} {max})"

    @staticmethod
    def eq(a, b):
        return f"{a} == {b}"

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
    def ge(a, b):
        return f"{a} >= {b}"

    @staticmethod
    def gelu(x):
        return f"spyre.gelu({x})"

    @staticmethod
    def gt(a, b):
        return f"{a} > {b}"

    @staticmethod
    def layernormscale(x, y):
        return f"spyre.layernormscale({x}, {y})"

    @staticmethod
    def layernormnorm(a, b, c, d, e):
        return f"spyre.layernormnorm({a}, {b}, {c}, {d}, {e})"

    @staticmethod
    def le(a, b):
        return f"{a} <= {b}"

    @staticmethod
    def log(x):
        return f"spyre.log({x})"

    @staticmethod
    def lt(a, b):
        return f"{a} < {b}"

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"

    @staticmethod
    def ne(a, b):
        return f"{a} != {b}"

    @staticmethod
    def neg(a):
        return f"-{a}"

    @staticmethod
    def pow(a, b):
        return f"{a} ** {b}"

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
    def square(x):
        return f"{x} * {x}"

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"

    @staticmethod
    def tanh(x):
        return f"spyre.tanh({x})"

    @staticmethod
    def to_dtype(x, dtype, src_dtype):
        return f"spyre.to_dtype({x} {dtype} {src_dtype})"

    @staticmethod
    def truediv(a, b):
        return f"{a} / {b}"

    @staticmethod
    def where(x, y, z):
        return f"spyre.where({x}, {y}, {z})"
