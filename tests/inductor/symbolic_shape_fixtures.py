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

"""Fake tensors with a symbolic dimension, for tests that need no device.

Most of the symbolic-shape work can be asserted against a real ``ShapeEnv``
without compiling anything, which keeps those tests fast and runnable
anywhere torch is. This module holds the setup so each test file does not
carry its own copy.

Not collected by pytest directly: no ``test_`` prefix and no CI config entry.
"""

import unittest

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

__all__ = [
    "ROWS_HINT",
    "TILE",
    "WIDTH",
    "fake_symbolic_rows",
    "is_symbolic",
    "shape_env_attr",
]

ROWS_HINT = 320
WIDTH = 128
TILE = 64


def fake_symbolic_rows(rows_hint=ROWS_HINT, width=WIDTH):
    """A fake ``[s, width]`` whose dim 0 is a backed dynamic symbol.

    Returns ``(shape_env, fake_mode, tensor)``. Operate on the tensor inside
    ``with fake_mode:`` so derived tensors land in the same mode.

    Two construction routes, because which one works has moved between torch
    versions and these tests have to keep running on both. ``create_symbol``
    is the direct one, ``from_tensor`` is the fallback. If neither works the
    test skips with both errors quoted, rather than failing as if the thing
    under test were broken.
    """
    errors = []

    try:
        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        src = LocalSource("x")
        with mode:
            sym = shape_env.create_symbol(rows_hint, src, DimDynamic.DYNAMIC)
            rows = shape_env.create_symintnode(sym, hint=rows_hint, source=src)
            tensor = torch.empty(rows, width)
        return shape_env, mode, tensor
    except Exception as exc:  # noqa: BLE001
        errors.append(f"create_symbol: {type(exc).__name__}: {exc}")

    try:
        from torch.fx.experimental.symbolic_shapes import StatelessSymbolicContext

        shape_env = ShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        real = torch.empty(rows_hint, width)
        kwargs = {"dynamic_sizes": [DimDynamic.DYNAMIC, DimDynamic.STATIC]}
        try:
            ctx = StatelessSymbolicContext(**kwargs)
        except TypeError:
            kwargs["dynamic_strides"] = [DimDynamic.INFER_STRIDE] * 2
            ctx = StatelessSymbolicContext(**kwargs)
        with mode:
            tensor = mode.from_tensor(
                real, source=LocalSource("x"), symbolic_context=ctx
            )
        return shape_env, mode, tensor
    except Exception as exc:  # noqa: BLE001
        errors.append(f"from_tensor: {type(exc).__name__}: {exc}")

    raise unittest.SkipTest(
        "could not build a fake tensor with a symbolic dim on this torch build: "
        + " | ".join(errors)
    )


def is_symbolic(value):
    """True for a SymInt or a sympy expression, False for a plain int."""
    return bool(getattr(value, "free_symbols", None)) or hasattr(value, "node")


def shape_env_attr(shape_env, name):
    """Read a ``ShapeEnv`` internal, failing with a sentence if it has moved.

    Some claims can only be checked against ``ShapeEnv`` internals, which is
    where they live. If torch renames one, a bare ``AttributeError`` says
    nothing useful, so name it here instead.
    """
    value = getattr(shape_env, name, None)
    if value is None:
        raise AssertionError(
            f"ShapeEnv has no '{name}' on this torch build, so this test cannot "
            f"check what it claims to check. Find the new name before deleting "
            f"the test, because the behaviour it pins is load-bearing."
        )
    return value
