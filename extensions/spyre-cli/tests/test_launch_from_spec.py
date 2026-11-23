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

from spyre_cli.core import launch_from_cli
from spyre_cli.launch_spec import default_tensors, tensors_from_spec

torch = pytest.importorskip("torch")


# The layout the compiler records for a [10, 512] fp16 tensor: 8 sticks of 64
# elements, the 10 rows inside each stick.
_ADD_LAYOUT = {
    "device_size": [8, 10, 64],
    "stride_map": [64, 512, 1],
    "device_dtype": "SEN169_FP16",
    "element_arrangement": "STANDARD",
    "valid_elements": [[[0, 2], 512], [[1], 10]],
}


def _spec_dict(**overrides):
    """A [10, 512] fp16 add: two inputs, one output, no pool."""
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
                "stride": [512, 1],
                "storage_offset": 0,
                "layout": dict(_ADD_LAYOUT),
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

    def __init__(self, shape, dtype, stride=None, layout=None):
        # The pool is allocated with a bare int extent, so normalise both
        # spellings the way torch itself does.
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self._stride = stride
        self.layout = layout
        self.filled = None

    def stride(self):
        return self._stride

    def fill_(self, value):
        self.filled = value
        return self

    def copy_(self, src):
        # default_tensors writes its ones through a host copy; record the value
        # it copied, the way fill_ records the value it filled with.
        values = set(src.flatten().tolist())
        self.filled = values.pop() if len(values) == 1 else values
        return self

    def cpu(self):
        # launch_from_cli reads its outputs back before printing them; a stub
        # has nothing to transfer, so it stands in for itself.
        return self


@pytest.fixture
def made():
    """Capture every tensor ``tensors_from_spec`` allocates.

    The pool comes from ``torch.empty`` and every argument from
    ``spyre_empty_with_layout``, so both are stood in for; ``_lazy_init`` is
    too, since it brings the runtime up. The layouts themselves are real
    ``SpyreTensorLayout`` objects: building one needs no device.
    """
    from spyre_cli import launch_spec as ls

    allocated = []
    real_empty = torch.empty

    def _with_layout(shape, stride, dtype, layout, device):
        t = _FakeTensor(shape, dtype, stride=tuple(stride), layout=layout)
        allocated.append(t)
        return t

    def _empty(shape, dtype=None, device=None, **kwargs):
        # Only stand in for device allocations. torch.ones on the explicit path
        # routes through torch.empty too and then fills for real, so handing it
        # a stub would break it; the meta stride query must stay real as well.
        if device != "spyre":
            return real_empty(shape, dtype=dtype, device=device, **kwargs)
        t = _FakeTensor(shape, dtype)
        allocated.append(t)
        return t

    with (
        mock.patch.object(torch, "empty", _empty),
        mock.patch.object(ls, "spyre_empty_with_layout", _with_layout),
        mock.patch.object(torch.spyre._impl, "_lazy_init", lambda: None),
    ):
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

    spec = LaunchSpec.from_dict(raw)
    tensors = tensors_from_spec(spec, {})

    assert len(tensors) == 3
    assert [t.shape for t in tensors] == [(10, 512)] * 3
    assert all(t.dtype is torch.float16 for t in tensors)
    # Allocation does not touch the values: that is the caller's business.
    assert [t.filled for t in tensors] == [None, None, None]

    # The smoke-test fill is a separate, explicit step, and it leaves the
    # output alone.
    default_tensors(spec, tensors)
    assert [t.filled for t in tensors] == [1.0, 1.0, None]


def test_pool_tensor_is_prepended_only_when_the_caller_supplies_one(made):
    """pool_size is the caller's tensor, so 0 means prepend nothing."""
    assert len(tensors_from_spec(_spec(pool_size=0), {})) == 3

    tensors = tensors_from_spec(_spec(pool_size=32768), {})
    assert len(tensors) == 4, "3 kernel args + 1 pool"
    assert tensors[0].shape == (32768,)
    assert tensors[0].dtype is torch.uint8


def test_symbolic_dim_needs_a_binding(made):
    """Resolved when bound, refused when not -- never guessed."""
    symbolic = _spec(symbols={"s0": {}})
    for arg in symbolic.args:
        arg.shape = ["s0", 512]
        arg.stride = [512, 1]

    tensors = tensors_from_spec(symbolic, {"s0": 128})
    assert [t.shape for t in tensors] == [(128, 512)] * 3

    with pytest.raises(ValueError, match="symbolic"):
        tensors_from_spec(symbolic, {})


def test_tensors_get_the_recorded_layout_and_strides(made):
    """The recorded layout, valid_elements included, and host strides are used.

    A transposed output's strides are (1, rows), not contiguous: rebuilt with
    contiguous ones it reads back differently from the compiled graph.
    """
    spec = _spec()
    spec.args[2].shape = [512, 10]
    spec.args[2].stride = [1, 512]
    tensors = tensors_from_spec(spec, {})

    assert [t.stride() for t in tensors] == [(512, 1), (512, 1), (1, 512)]
    layout = tensors[0].layout
    assert list(layout.device_size) == [8, 10, 64]
    assert layout.valid_elements == {(0, 2): 512, (1,): 10}


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("element_arrangement", "NOT_IN_THIS_BUILD", "element arrangement"),
        ("device_dtype", "NOT_IN_THIS_BUILD", "device dtype"),
        (None, None, "no device layout is recorded"),
    ],
    ids=["unknown-arrangement", "unknown-device-dtype", "no-layout"],
)
def test_a_layout_this_build_cannot_rebuild_is_refused(made, field, value, expected):
    """Refused rather than replaced by the default arrangement.

    A tensor of the right shape and dtype in the wrong arrangement launches and
    returns wrong data, so falling back would hide exactly what the spec is for.
    """
    raw = _spec_dict()
    if field is None:
        del raw["args"][1]["layout"]
    else:
        raw["args"][1]["layout"][field] = value
    from torch_spyre._inductor.codegen.kernel_launchspec import LaunchSpec

    with pytest.raises(ValueError, match=expected):
        tensors_from_spec(LaunchSpec.from_dict(raw), {})


def test_a_nonzero_storage_offset_is_refused(made):
    """A rebuilt tensor cannot be placed at an offset into a larger storage."""
    spec = _spec()
    spec.args[0].storage_offset = 512
    with pytest.raises(ValueError, match="storage offset 512"):
        tensors_from_spec(spec, {})


def test_default_ones_cover_input_output_args(made):
    """An in-place op's target is read by the kernel, so it gets ones too."""
    spec = _spec()
    spec.args[0].role = "input_output"
    tensors = tensors_from_spec(spec, {})
    default_tensors(spec, tensors)
    assert [t.filled for t in tensors] == [1.0, 1.0, None]


def _layout(
    device_size, stride_map, device_dtype="SEN169_FP16", arrangement="STANDARD"
):
    from torch_spyre._inductor.codegen.kernel_launchspec import LaunchArgLayout
    from spyre_cli.launch_spec import spyre_layout

    return spyre_layout(
        LaunchArgLayout(device_size, stride_map, device_dtype, arrangement)
    )


def test_layout_comparison_ignores_size_one_device_dims():
    """A keepdim output recorded as [1, 1, 512, 64] matches [1, 512, 64].

    A size-1 device dimension is always indexed at 0, so the two place every
    element in the same place, and a launch with either is correct.
    """
    from spyre_cli.launch_spec import _layout_mismatch

    recorded = _layout([1, 1, 512, 64], [0, 0, 1, 0])
    plain = _layout([1, 512, 64], [0, 1, 0])
    assert _layout_mismatch(recorded, plain) is None
    other = _layout([2, 256, 64], [0, 1, 0])
    assert "packed differently" in _layout_mismatch(recorded, other)


def test_layout_comparison_includes_the_arrangement():
    """Same placement, different arrangement: different bytes, so refused."""
    from spyre_cli.launch_spec import _layout_mismatch

    want = _layout([1, 64, 128], [128, 128, 1], "SEN143_FP8", "QFP8CH")
    got = _layout([1, 64, 128], [128, 128, 1], "SEN143_FP8", "STANDARD")
    assert "element arrangement QFP8CH" in _layout_mismatch(want, got)
    assert _layout_mismatch(want, want) is None


# --------------------------------------------------------------------------
# Refusing a launch the spec does not accept
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs,outputs,expected",
    [
        (["512x10@fp16", "10x512@fp16"], ["10x512@fp16"], "transposed"),
        (["10x512@fp32", "10x512@fp16"], ["10x512@fp16"], "expected device dtype"),
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
        core, "_launch", side_effect=lambda p, t, symbol_kinds=None: launched.append(t)
    ):
        launch_from_cli(str(tmp_path), ["10x512@fp16"], ["10x512@fp16"])

    assert len(launched) == 1
    assert len(launched[0]) == 2, "one input, one output, unchecked"


def test_a_non_default_layout_is_allocated_as_recorded():
    """Allocation with a special packing gives a tensor in that packing.

    The depthwise conv2d puts its channels in one stick; the tensor built from
    its recorded layout must carry that layout, not the default for its shape.
    """
    if not torch.accelerator.is_available():
        pytest.skip("needs a device to allocate the tensor")
    from torch_spyre._C import get_spyre_tensor_layout

    raw = _spec_dict()
    raw["args"] = [
        {
            "arg_index": 0,
            "role": "input",
            "shape": [1, 64, 32, 32],
            "dtype": "float16",
            "layout": {
                "device_size": [32, 32, 1, 1, 64],
                "stride_map": [32, 1, 0, 65536, 1024],
                "device_dtype": "SEN169_FP16",
                "element_arrangement": "STANDARD",
            },
        }
    ]
    from torch_spyre._inductor.codegen.kernel_launchspec import LaunchSpec

    (tensor,) = tensors_from_spec(LaunchSpec.from_dict(raw), {})
    got = get_spyre_tensor_layout(tensor)
    assert list(got.device_size) == [32, 32, 1, 1, 64]
    assert list(got.stride_map) == [32, 1, 0, 65536, 1024]


def test_host_strides_are_checked():
    """A contiguous tensor for an argument recorded with transposed strides."""
    if not torch.accelerator.is_available():
        pytest.skip("needs a device to allocate the tensors being checked")
    from torch_spyre._C import get_spyre_tensor_layout
    from torch_spyre._inductor.codegen.kernel_launchspec import LaunchSpec
    from spyre_cli.launch_spec import check_tensors_against_spec

    transposed = torch.empty((256, 128), dtype=torch.float16, device="spyre").t()
    layout = get_spyre_tensor_layout(transposed)
    raw = _spec_dict()
    raw["args"] = [
        {
            "arg_index": 0,
            "role": "output",
            "shape": [128, 256],
            "dtype": "float16",
            "stride": [1, 128],
            "layout": {
                "device_size": list(layout.device_size),
                "stride_map": list(layout.stride_map),
                "device_dtype": "SEN169_FP16",
                "element_arrangement": "STANDARD",
            },
        }
    ]
    spec = LaunchSpec.from_dict(raw)
    assert check_tensors_against_spec(spec, [transposed], {}) == []

    contiguous = torch.empty((128, 256), dtype=torch.float16, device="spyre")
    problems = check_tensors_against_spec(spec, [contiguous], {})
    assert any("expected host strides [1, 128]" in p for p in problems), problems
