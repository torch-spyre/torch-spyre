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

"""Eager ``x[index]`` with a 1-D integer index on a Spyre tensor (#5768)."""

import pytest
import torch

import torch_spyre  # noqa: F401

DEVICE = "spyre"
DTYPE = torch.float16
ROWS, HIDDEN = 512, 4096


def _to_device(x):
    """The device copy and its exact host image.

    fp16 is stored on Spyre in a device format, so the H2D/D2H round trip is not
    bit-exact; a gather must reproduce the stored rows exactly, so compare to those.
    """
    dx = x.to(DEVICE)
    return dx.cpu(), dx


@pytest.fixture(scope="module")
def source():
    torch.manual_seed(0)
    return _to_device(torch.randn(ROWS, HIDDEN, dtype=DTYPE))


@pytest.mark.parametrize("index_device", ["cpu", DEVICE])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows", [[3, 100, 511], [0], [7, 7, 2, 0], list(range(16))], ids=str
)
def test_gathers_the_indexed_rows(source, rows, dtype, index_device):
    x, dx = source
    index = torch.tensor(rows, dtype=dtype)

    got = dx[index.to(index_device)]

    assert got.device.type == DEVICE
    torch.testing.assert_close(got.cpu(), x[index], atol=0, rtol=0)


@pytest.mark.parametrize("index_device", ["cpu", DEVICE])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length", [8, 16, 63])
def test_gathers_from_a_1d_source(length, dtype, index_device):
    """A 1-D source shorter than a 64-element stick (#4334)."""
    x, dx = _to_device(torch.arange(length, dtype=DTYPE))
    index = torch.tensor([length - 1, 0, 3, 3], dtype=dtype)

    got = dx[index.to(index_device)]

    torch.testing.assert_close(got.cpu(), x[index], atol=0, rtol=0)


@pytest.mark.parametrize("index_device", ["cpu", DEVICE])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_negative_indices_wrap(source, dtype, index_device):
    x, dx = source
    index = torch.tensor([-1, 0, -ROWS], dtype=dtype)

    got = dx[index.to(index_device)]

    torch.testing.assert_close(got.cpu(), x[index], atol=0, rtol=0)


@pytest.mark.parametrize("index_device", ["cpu", DEVICE])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [[0, ROWS], [-ROWS - 1]], ids=str)
def test_out_of_range_index_raises(source, rows, dtype, index_device):
    _, dx = source
    with pytest.raises(IndexError, match="out of range"):
        dx[torch.tensor(rows, dtype=dtype, device=index_device)]


@pytest.mark.parametrize("index_device", ["cpu", DEVICE])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("row", [1, -2])
def test_out_of_range_index_into_one_row_source_raises(row, dtype, index_device):
    _, dx = _to_device(torch.randn(1, HIDDEN, dtype=DTYPE))
    with pytest.raises(IndexError, match="out of range"):
        dx[torch.tensor([row], dtype=dtype, device=index_device)]


def test_spyre_index_into_cpu_source_raises():
    x = torch.randn(8, 64, dtype=DTYPE)
    index = torch.tensor([0, 1], device=DEVICE)
    with pytest.raises(RuntimeError, match="indices should be either on cpu"):
        torch.ops.aten.index.Tensor(x, [index])


def test_empty_index_returns_no_rows(source):
    _, dx = source
    got = dx[torch.empty(0, dtype=torch.int64)]
    assert got.shape == (0, HIDDEN)
    assert got.device.type == DEVICE


def test_nonzero_offset_source_reads_its_own_rows(source):
    """A view's offset is dropped by a compiled kernel; the gather must honor it."""
    x, dx = source
    index = torch.tensor([0, 5, 9])

    got = dx[64:][index]

    torch.testing.assert_close(got.cpu(), x[64:][index], atol=0, rtol=0)


def test_trailing_dims_are_kept():
    x, dx = _to_device(torch.randn(8, 4, 64, dtype=DTYPE))
    index = torch.tensor([6, 1, 3])

    got = dx[index]

    torch.testing.assert_close(got.cpu(), x[index], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [[0], [0, 0], [0, -1, 0]], ids=str)
def test_one_row_source(rows):
    x, dx = _to_device(torch.randn(1, HIDDEN, dtype=DTYPE))
    index = torch.tensor(rows)

    got = dx[index]

    torch.testing.assert_close(got.cpu(), x[index], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [[0], [0, 0], [0, -1, 0]], ids=str)
@pytest.mark.parametrize(
    "make_source",
    [
        pytest.param(lambda d: d[:, :1].T, id="column.T"),
        pytest.param(lambda d: d[:, :1].T.squeeze(0).unsqueeze(0), id="restrided"),
        pytest.param(lambda d: d[:, 0].unsqueeze(1).T, id="(8,1).T"),
    ],
)
def test_transposed_one_row_source(make_source, rows):
    """The stick runs across rows, so this one gathers instead of broadcasting."""
    base = torch.arange(512, dtype=DTYPE).reshape(8, 64)
    dx = make_source(base.to(DEVICE))
    index = torch.tensor(rows)

    got = dx[index]

    torch.testing.assert_close(got.cpu(), dx.cpu()[index], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [[0], [0, 0], [0, -1, 0]], ids=str)
@pytest.mark.parametrize(
    "make_source",
    [
        pytest.param(lambda d: d, id="contiguous"),
        pytest.param(lambda d: d.squeeze(0).unsqueeze(1).T, id="stride(1,1)"),
        pytest.param(lambda d: d.squeeze(0)[:, None].T, id="vector[:,None].T"),
    ],
)
def test_equal_one_row_views_with_other_singleton_strides(make_source, rows):
    """The stride of dim 0 is meaningless at size 1 and must not pick the path."""
    _, base = _to_device(torch.randn(1, HIDDEN, dtype=DTYPE))
    dx = make_source(base)
    index = torch.tensor(rows)

    got = dx[index]

    torch.testing.assert_close(got.cpu(), dx.cpu()[index], atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(1, 64), (8, 64)], ids=str)
def test_bool_source_raises(shape):
    # TODO: the device copy of a bool tensor loses its values; when it is fixed,
    # bool sources must move to the positive tests above.
    dx = (torch.arange(64 * shape[0]).reshape(shape) % 2 == 0).to(DEVICE)
    with pytest.raises(NotImplementedError, match="bool source"):
        dx[torch.tensor([0, 0])]


@pytest.mark.parametrize("dtype", [torch.int16, torch.float32])
def test_index_dtype_invalid_on_cpu_raises_as_on_cpu(source, dtype):
    _, dx = source
    with pytest.raises(IndexError, match="must be long, int, byte or bool"):
        dx[torch.tensor([0, 1], dtype=dtype)]


@pytest.mark.parametrize(
    "make_index",
    [
        pytest.param(lambda: (torch.tensor([0, 1]), torch.tensor([1, 2])), id="two"),
        pytest.param(lambda: (slice(None), torch.tensor([1, 2])), id="dim1"),
        pytest.param(lambda: torch.tensor([[0, 1]]), id="2d"),
        pytest.param(lambda: torch.zeros(ROWS, dtype=torch.bool), id="bool"),
        pytest.param(lambda: torch.zeros(ROWS, dtype=torch.uint8), id="uint8"),
    ],
)
def test_other_index_forms_name_the_supported_subset(source, make_index):
    # TODO: a deliberate scope limit today; a case that gains a kernel must move to
    # the positive tests above.
    _, dx = source
    with pytest.raises(NotImplementedError, match="single 1-D int32/int64 index"):
        dx[make_index()]
