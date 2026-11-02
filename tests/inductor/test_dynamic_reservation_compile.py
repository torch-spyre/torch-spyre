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

"""A compiled function over a ``tensor.to("spyre", dynamic=...)`` tensor.

What the reservation guarantees today: resizing within [min, max] keeps the
SpyreTensorLayout, so one compiled graph and one kernel serve every size, and
at the size it was compiled for the result is right and the reserved space
above the logical size is never read.

What it does not guarantee yet: the kernel is built for the compile-time
size, so other sizes give wrong values without an error. That test is a strict
xfail. It XPASSes, and so fails, once the compiler half builds geometry at the
declared max and binds the size at launch, and that change removes the marker.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor.utils import fresh_cache

import torch_spyre  # noqa: F401
import torch_spyre._inductor.config as spyre_config
from torch_spyre._C import get_spyre_tensor_layout
from torch_spyre.execution.kernel_cache import get_cache_root_dir

DEVICE = torch.device("spyre")
COLS = 1024
SPEC = {"min": 140, "max": 630, "granularity": 70}
# The first entry is the compile size, the declared min. The rest grow and
# shrink but never go below it: a kernel built for more rows than its output
# holds may write past the output buffer, which is not measured yet.
SIZES = [140, 630, 280, 560, 210]
FOLLOW_UP = "#<issue>"

# fp16 results of ops on values in [0, 1), against an fp32 reference: one ulp
# is about 1e-3 near 1 and sum(dim=0) over 140 rows measured a relative error
# of 3e-3.
ATOL = 1e-2
RTOL = 2e-2


def _sum_dim0(a):
    return a.sum(dim=0)


CASES = {
    "gelu": (F.gelu, lambda a: F.gelu(a.float())),
    "sum_dim0": (_sum_dim0, lambda a: a.float().sum(dim=0)),
}


def _kernel_cache_entries():
    root = get_cache_root_dir()
    return sum(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root))


def _run_over_sizes(fn, ref_fn):
    """One compiled fn over one reserved tensor resized through SIZES.

    The host tensor holds max rows, so after any in-place resize the first n
    device rows are x[:n], which is what the reference is computed from.
    """
    x = torch.rand(SPEC["max"], COLS, dtype=torch.float16)
    y = x.to(DEVICE, dynamic={0: SPEC})
    layout = str(get_spyre_tensor_layout(y))
    compiled = torch.compile(fn)
    results = []
    for n in SIZES:
        if y.size(0) != n:
            y.resize_((n, COLS))
        out = compiled(y).cpu().float()
        assert str(get_spyre_tensor_layout(y)) == layout, f"layout changed at {n} rows"
        results.append((n, out, ref_fn(x[:n])))
    return results


@pytest.fixture(autouse=True)
def _fresh_compile_state():
    torch._dynamo.reset()
    counters.clear()
    with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
        yield


@pytest.mark.parametrize("case", sorted(CASES))
def test_one_graph_one_kernel_and_right_at_compile_size(case):
    fn, ref_fn = CASES[case]
    results = _run_over_sizes(fn, ref_fn)

    assert counters["stats"]["unique_graphs"] == 1
    assert counters["frames"]["ok"] == 1
    assert _kernel_cache_entries() == 1
    for n, out, ref in results:
        assert out.shape == ref.shape, f"shape at {n} rows"
    _, out, ref = results[0]
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "compiled kernels over a reserved dim are built for the compile-time "
        "size: at other sizes rows above it are not computed (pointwise) or "
        f"not read (sum over dim 0), with no error. {FOLLOW_UP}"
    ),
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_right_at_every_size(case):
    fn, ref_fn = CASES[case]
    for n, out, ref in _run_over_sizes(fn, ref_fn):
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL, msg=f"at {n} rows")


@pytest.mark.parametrize("case", sorted(CASES))
def test_compile_size_result_ignores_poisoned_tail(case):
    # NaN above the compile-size rows, transferred at max then shrunk, so the
    # reserved space really holds NaN. A kernel that read it would surface it.
    fn, ref_fn = CASES[case]
    rows = SPEC["min"]
    x = torch.rand(SPEC["max"], COLS, dtype=torch.float16)
    x[rows:] = float("nan")
    y = x.to(DEVICE, dynamic={0: SPEC})
    y.resize_((rows, COLS))

    out = torch.compile(fn)(y).cpu().float()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_fn(x[:rows]), atol=ATOL, rtol=RTOL)
