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

"""Spyre var/std/var_mean/std_mean decompositions: numerics against torch on CPU,
and the traced graph they produce. No device and no Spyre compile."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torch_spyre._inductor.decompositions import (
    get_spyre_decomp_table,
    spyre_std_correction,
    spyre_std_mean_correction,
    spyre_var_correction,
    spyre_var_mean_correction,
)
from torch_spyre._inductor.errors import Unsupported

aten = torch.ops.aten

# (shape, dim, keepdim). dim forms: list, int, None, [] (= all dims), unsorted multi-dim.
CASES = [
    ((67, 256), [-1], False),
    ((67, 256), [0], True),
    ((5, 7), -1, False),
    ((6, 7, 12, 64), [2, 3], False),
    ((4, 6, 8, 64), [3, 0], True),
    ((32, 64), None, False),
    ((32, 64), [], True),
]
CORRECTIONS = [None, 0, 1, 2, 0.5]


def _x(shape):
    g = torch.Generator().manual_seed(0xAFFE)
    # Offset mean: a one-pass E[x^2]-E[x]^2 would lose most digits here.
    return torch.randn(shape, generator=g, dtype=torch.float32) * 3.0 + 100.0


@pytest.mark.parametrize("shape,dim,keepdim", CASES)
@pytest.mark.parametrize("correction", CORRECTIONS)
def test_var_std_match_torch_fp32(shape, dim, keepdim, correction):
    x = _x(shape)
    ref_var = torch.var(
        x.double(), dim=dim, correction=correction, keepdim=keepdim
    ).float()
    ref_mean = torch.mean(
        x.double(), dim=dim if dim not in (None, []) else None, keepdim=keepdim
    ).float()
    var = spyre_var_correction(x, dim, correction=correction, keepdim=keepdim)
    std = spyre_std_correction(x, dim, correction=correction, keepdim=keepdim)
    vm_var, vm_mean = spyre_var_mean_correction(
        x, dim, correction=correction, keepdim=keepdim
    )
    sm_std, sm_mean = spyre_std_mean_correction(
        x, dim, correction=correction, keepdim=keepdim
    )
    for got in (var, vm_var):
        assert got.shape == ref_var.shape
        torch.testing.assert_close(got, ref_var, rtol=1e-4, atol=1e-4)
    for got in (std, sm_std):
        torch.testing.assert_close(got, ref_var.sqrt(), rtol=1e-4, atol=1e-4)
    for got in (vm_mean, sm_mean):
        assert got.shape == ref_mean.shape
        torch.testing.assert_close(got, ref_mean, rtol=1e-5, atol=1e-4)


def test_var_fp16_close_to_fp32_reference():
    x = _x((67, 256)).sub(100.0).half()
    ref = torch.var(x.float(), dim=-1)
    got = spyre_var_correction(x, [-1])
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "x,dim,correction,match",
    [
        (torch.ones(4, 8, dtype=torch.int32), [-1], None, "dtype"),
        (torch.tensor(1.0), None, 0, "0-d"),
        (torch.randn(4, 1), [-1], 1, "N - correction"),
        (torch.randn(4, 2), [-1], 3, "N - correction"),
    ],
)
def test_unsupported_inputs_raise(x, dim, correction, match):
    with pytest.raises(Unsupported, match=match):
        spyre_var_correction(x, dim, correction=correction)


def test_registered_in_spyre_table():
    table = get_spyre_decomp_table()
    assert table[aten.var.correction].__name__ == "spyre_var_correction"
    assert table[aten.std.correction].__name__ == "spyre_std_correction"
    assert table[aten.var_mean.correction].__name__ == "spyre_var_mean_correction"
    assert table[aten.std_mean.correction].__name__ == "spyre_std_mean_correction"


def _targets(fn, *args):
    gm = make_fx(fn, decomposition_table=get_spyre_decomp_table())(*args)
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: torch.var(x, dim=-1),
        lambda x: torch.std(x, dim=-1),
        lambda x: torch.var_mean(x, dim=0),
        lambda x: torch.std_mean(x, dim=[0, 1], keepdim=True),
        lambda x: x.var(dim=-1, keepdim=True, unbiased=False),
    ],
)
def test_traced_graph_has_no_variance_reduction(fn):
    x = torch.randn(4, 8, 64, dtype=torch.float16)
    targets = _targets(fn, x)
    for banned in (
        aten.var.correction,
        aten.std.correction,
        aten.var_mean.correction,
        aten.std_mean.correction,
        torch.ops.prims.var.default,
    ):
        assert banned not in targets
    # Same input dtype throughout: no fp32 upcast, unlike the core std ref.
    assert torch.ops.prims.convert_element_type.default not in targets
    assert targets.count(aten.mean.dim) == 2


def test_std_graph_ends_in_sqrt():
    targets = _targets(
        lambda x: torch.std(x, dim=-1), torch.randn(4, 64, dtype=torch.float16)
    )
    assert aten.sqrt.default in targets
