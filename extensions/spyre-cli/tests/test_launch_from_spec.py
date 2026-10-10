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

"""What the CLI decides when it launches from a spec.

Scoped to the consumer's own choices -- which tensors to build, in what order,
and when to refuse. The schema, shape resolution and validation all live in
``torch_spyre._inductor.codegen.kernel_launchspec`` and are covered with the
producer; what matters here is that the CLI drives them correctly.

Tensor construction runs against a stub for the device factory, so argument
order and the pool handoff are checked without a Spyre card. The refusal cases
need real tensors and skip without one.
"""

import json
from unittest import mock

import pytest

from spyre_cli.core import _tensors_from_spec, launch_from_cli

torch = pytest.importorskip("torch")


def _spec_dict(**overrides):
    """A [10, 512] fp16 add: two inputs, one output, no pool, no layout."""
    spec = {
        "version": 1,
        "kernel_name": "sdsc_fused_add_0",
        "pool_size": 0,
        "bundle_symbolic_args": True,
        "args": [
            {
                "arg_index": i,
                "role": "input" if i < 2 else "output",
                "shape": [10, 512],
                "dtype": "float16",
            }
            for i in range(3)
        ],
    }
    spec.update(overrides)
    return spec


def _spec(**overrides):
    """The same, parsed into the ``LaunchSpec`` the consumer now takes."""
    from torch_spyre._inductor.codegen.kernel_launchspec import LaunchSpec

    return LaunchSpec.from_dict(_spec_dict(**overrides))


class _FakeTensor:
    """Stands in for a device tensor: records how it was made."""

    def __init__(self, shape, dtype):
        # The pool is allocated with a bare int extent, so normalise both
        # spellings the way torch itself does.
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.filled = None

    def fill_(self, value):
        self.filled = value
        return self

    def cpu(self):
        # launch_from_cli reads its outputs back before printing them; a stub
        # has nothing to transfer, so it stands in for itself.
        return self


@pytest.fixture
def made():
    """Capture every tensor ``_tensors_from_spec`` allocates.

    It reaches for ``torch.empty`` inside the function, so patching the module
    attribute is enough -- no import-order games. ``_alloc`` also asks torch for
    host strides on the meta device, which needs no patching since nothing is
    allocated there.
    """
    allocated = []
    real_empty = torch.empty

    def _empty(shape, dtype=None, device=None, **kwargs):
        # Only stand in for device allocations. torch.ones on the explicit path
        # routes through torch.empty too and then fills for real, so handing it
        # a stub would break it; the meta stride query must stay real as well.
        if device != "spyre":
            return real_empty(shape, dtype=dtype, device=device, **kwargs)
        t = _FakeTensor(shape, dtype)
        allocated.append(t)
        return t

    with mock.patch.object(torch, "empty", _empty):
        yield allocated


# --------------------------------------------------------------------------
# Building the launch
# --------------------------------------------------------------------------


def test_builds_the_tensors_the_spec_describes(made):
    """One per arg, in arg_index order, with the spec's dtypes.

    The spec file lists its args reversed here: binding is positional, so the
    order they appear on disk must not change what gets built where.
    """
    raw = _spec_dict()
    raw["args"] = list(reversed(raw["args"]))
    from torch_spyre._inductor.codegen.kernel_launchspec import LaunchSpec

    tensors = _tensors_from_spec(LaunchSpec.from_dict(raw), {})

    assert len(tensors) == 3
    assert [t.shape for t in tensors] == [(10, 512)] * 3
    assert all(t.dtype is torch.float16 for t in tensors)
    # Inputs arrive filled, the output untouched.
    assert [t.filled for t in tensors] == [1.0, 1.0, None]


def test_pool_tensor_is_prepended_only_when_the_caller_supplies_one(made):
    """pool_size is the caller's tensor, so 0 means prepend nothing."""
    assert len(_tensors_from_spec(_spec(pool_size=0), {})) == 3

    tensors = _tensors_from_spec(_spec(pool_size=32768), {})
    assert len(tensors) == 4, "3 kernel args + 1 pool"
    assert tensors[0].shape == (32768,)
    assert tensors[0].dtype is torch.uint8


def test_symbolic_dim_needs_a_binding(made):
    """Resolved when bound, refused when not -- never guessed."""
    symbolic = _spec(symbols={"s0": {}})
    for arg in symbolic.args:
        arg.shape = ["s0", 512]

    tensors = _tensors_from_spec(symbolic, {"s0": 128})
    assert [t.shape for t in tensors] == [(128, 512)] * 3

    with pytest.raises(ValueError, match="symbolic"):
        _tensors_from_spec(symbolic, {})


# --------------------------------------------------------------------------
# Refusing a launch the spec does not accept
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs,outputs,expected",
    [
        (["512x10@fp16", "10x512@fp16"], ["10x512@fp16"], "transposed"),
        (["10x512@fp32", "10x512@fp16"], ["10x512@fp16"], "expected dtype float16"),
        (["10x512@fp16"], ["10x512@fp16"], "expected 3 tensors"),
    ],
    ids=["transposed", "wrong-dtype", "wrong-count"],
)
def test_explicit_args_that_disagree_with_the_spec_are_refused(
    tmp_path, inputs, outputs, expected
):
    """Each of these used to launch and return wrong data."""
    if not torch.accelerator.is_available():
        pytest.skip("needs a device to allocate the tensors being checked")
    (tmp_path / "spyreCodeDir").mkdir()
    (tmp_path / "spyreCodeDir" / "launch_spec.json").write_text(
        json.dumps(_spec_dict())
    )
    with pytest.raises(ValueError, match=expected):
        launch_from_cli(str(tmp_path), inputs, outputs)


def test_a_folder_without_a_spec_and_no_arguments_is_refused(tmp_path):
    """Neither source of truth: refuse rather than launch zero tensors.

    Without this the launch proceeds and dies in the runtime with an
    argument-count error, which says nothing about what the caller omitted.
    """
    with pytest.raises(ValueError, match="no launch spec"):
        launch_from_cli(str(tmp_path), [], [])


def test_a_folder_without_a_spec_is_launched_from_the_arguments(tmp_path, made):
    """Folders compiled before the spec existed keep working."""
    from spyre_cli import core

    launched = []
    with mock.patch.object(
        core, "_launch", side_effect=lambda p, t: launched.append(t)
    ):
        launch_from_cli(str(tmp_path), ["10x512@fp16"], ["10x512@fp16"])

    assert len(launched) == 1
    assert len(launched[0]) == 2, "one input, one output, unchecked"
