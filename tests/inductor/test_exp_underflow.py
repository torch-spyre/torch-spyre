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

"""A masked softmax must ignore what it masks.

The device's fp16 exp saturates at a nonzero floor rather than underflowing, including
for -inf. A masked softmax then gives each masked position a nonzero weight, so the
values behind the mask reach the output through matmul(probs, v). For paged attention,
which gathers whole KV pages, that makes masked page contents affect the result.

The hardware test states the invariant end to end -- varying only the masked-out
inputs must leave the result bit-identical -- rather than inspecting the probs. That
is deliberate: a host read-back cannot confirm a device value is zero, because device
fp16 is DL16 (1-6-9) and the subnormal range does not survive the D2H rounding. A
candidate fix once made the read-back show exact zeros while the leak was still there.

Tests in mode: compile, eager. Eager routes aten.exp through torch.compile
(torch_spyre/ops/eager.py), so both modes exercise the same pass.
"""

import math

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch_spyre._inductor.temp_passes import (
    _exp_underflow_threshold,
    guard_exp_underflow,
)
from torch_spyre.constants import DEVICE_NAME

MODES = ["compile", "eager"]
aten = torch.ops.aten


def _run(fn, *args, mode="compile"):
    return (torch.compile(fn, dynamic=False) if mode == "compile" else fn)(*args)


def test_threshold_is_the_round_to_zero_point():
    # Use the representable value immediately below each mathematical midpoint.
    for dtype, exponent in ((torch.float16, -25), (torch.bfloat16, -134)):
        exact_midpoint = exponent * math.log(2)
        midpoint = torch.tensor(exact_midpoint, dtype=dtype)
        expected = midpoint.item()
        if expected >= exact_midpoint:
            expected = torch.nextafter(
                midpoint, torch.tensor(float("-inf"), dtype=dtype)
            ).item()
        assert _exp_underflow_threshold(dtype) == expected
    assert _exp_underflow_threshold(torch.int32) is None


def test_emitted_shape_selects_a_layout_preserving_zero():
    class M(torch.nn.Module):
        def forward(self, scores):
            return torch.exp(scores)

    gm = make_fx(M())(torch.zeros(4, 8, dtype=torch.float16))
    guard_exp_underflow(gm.graph)

    wheres = [n for n in gm.graph.nodes if n.target is aten.where.self]
    assert len(wheres) == 1, [n.name for n in gm.graph.nodes]
    condition, zero, original = wheres[0].args
    assert condition.target is aten.le.Scalar
    assert condition.args[1] == pytest.approx(_exp_underflow_threshold(torch.float16))
    assert zero.target is aten.clamp.default
    assert zero.args == (original,)
    assert zero.kwargs == {"min": 0.0, "max": 0.0}
    assert original.target is aten.exp.default

    guard_exp_underflow(gm.graph)
    assert len([n for n in gm.graph.nodes if n.target is aten.where.self]) == 1


@pytest.mark.parametrize("mode", MODES)
def test_exp_underflows_to_exact_zero(mode):
    """The whole point: a deeply negative score must carry no weight at all.

    -inf is included because it is what an online-softmax running max starts at, and
    what a fully-masked row's score becomes. -1e4 is the value superdsc.py's padding
    mask uses.
    """
    dtype = torch.float16
    xs = [-40.0, -1.0e4, torch.finfo(dtype).min, float("-inf")]
    host = torch.zeros(64, dtype=dtype)
    host[: len(xs)] = torch.tensor(xs, dtype=dtype)
    got = _run(torch.exp, host.to(DEVICE_NAME), mode=mode).cpu()
    for i, x in enumerate(xs):
        assert float(got[i]) == 0.0, (
            f"exp({x}) returned {float(got[i]):.6e}, want exact 0"
        )


@pytest.mark.parametrize("mode", MODES)
def test_exp_of_ordinary_inputs_is_unchanged(mode):
    """The guard must not zero ordinary device exp results."""
    dtype = torch.float16
    xs = [0.0, -1.0, -4.0, -8.0, -10.0]
    host = torch.zeros(64, dtype=dtype)
    host[: len(xs)] = torch.tensor(xs, dtype=dtype)
    got = _run(torch.exp, host.to(DEVICE_NAME), mode=mode).cpu()
    for i, x in enumerate(xs):
        assert float(got[i]) > 0.0, f"exp({x}) was zeroed"


def test_guard_preserves_ordinary_exp_on_cpu():
    class M(torch.nn.Module):
        def forward(self, scores):
            return torch.exp(scores)

    xs = torch.tensor([0.0, -1.0, -4.0, -8.0, -10.0, -15.0])
    gm = make_fx(M())(xs)
    expected = gm(xs)
    guard_exp_underflow(gm.graph)
    gm.recompile()
    torch.testing.assert_close(gm(xs), expected, rtol=0, atol=0)


def test_guard_preserves_smallest_subnormal_at_boundary():
    class M(torch.nn.Module):
        def forward(self, scores):
            return torch.exp(scores)

    midpoint = torch.tensor(-25.0 * math.log(2), dtype=torch.float16)
    below = torch.nextafter(midpoint, torch.tensor(float("-inf"), dtype=torch.float16))
    xs = torch.stack((midpoint, below))
    gm = make_fx(M())(xs)
    guard_exp_underflow(gm.graph)
    gm.recompile()

    got = gm(xs)
    assert got[0] == torch.finfo(torch.float16).tiny * torch.finfo(torch.float16).eps
    assert got[1] == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_non_fp16_exp_is_not_rewritten(dtype):
    class M(torch.nn.Module):
        def forward(self, scores):
            return torch.exp(scores)

    gm = make_fx(M())(torch.zeros(4, 8, dtype=dtype))
    guard_exp_underflow(gm.graph)
    assert not any(node.target is aten.where.self for node in gm.graph.nodes)


@pytest.mark.parametrize("mode", MODES)
def test_masked_softmax_ignores_the_masked_values(mode):
    """Varying only masked-out inputs must leave the result bit-identical.

    Mirrors a paged-attention block: 32 real keys and 96 masked ones sharing a 128-slot
    page. The valid quarter is byte-identical across arms, so any difference is a
    masked position carrying weight.
    """
    dtype, rows, keys, valid = torch.float16, 32, 128, 32
    sentinel = torch.finfo(dtype).min

    torch.manual_seed(7)
    query = torch.randn(rows, 256, dtype=dtype) * 0.5
    torch.manual_seed(1234)
    keys_base = torch.randn(keys, 256, dtype=dtype).clamp(-1, 1) * 1.7
    torch.manual_seed(99)
    values_base = torch.randn(keys, 256, dtype=dtype).clamp(-1, 1) * 15.0
    mask = torch.zeros(rows, keys, dtype=dtype)
    mask[:, valid:] = sentinel

    def attention(q, k, v, m):
        scores = torch.matmul(q, k.transpose(-2, -1)) + m
        probs = torch.exp(scores - torch.amax(scores, dim=-1, keepdim=True))
        return torch.matmul(probs, v) / probs.sum(dim=-1, keepdim=True)

    results = []
    for fill in (0.0, 1.0, 1000.0):
        k, v = keys_base.clone(), values_base.clone()
        k[valid:] = fill
        v[valid:] = fill
        results.append(
            _run(
                attention,
                query.to(DEVICE_NAME),
                k.to(DEVICE_NAME),
                v.to(DEVICE_NAME),
                mask.to(DEVICE_NAME),
                mode=mode,
            ).cpu()
        )

    assert not torch.isnan(results[0]).any()
    for i, other in enumerate(results[1:], start=1):
        differing = int((results[0] != other).sum())
        assert differing == 0, (
            f"masked-fill arm {i} changed {differing}/{other.numel()} outputs; "
            f"maxabsdiff={float((results[0].float() - other.float()).abs().max()):.6g}"
        )


@pytest.mark.parametrize("mode", MODES)
def test_fp32_exp_still_lowers(mode):
    """fp32 is out of scope; it must keep compiling.

    Documents the gap rather than asserting a zero: the device's fp32 exp saturates at
    2**-17, and where3 is not available on fp32 to select against it.
    """
    host = torch.zeros(32, dtype=torch.float32)
    host[0], host[1] = -1.0, -40.0
    got = _run(torch.exp, host.to(DEVICE_NAME), mode=mode).cpu()
    assert float(got[0]) > 0.0
    assert not torch.isnan(got).any()
