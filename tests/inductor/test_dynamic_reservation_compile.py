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

What the reservation guarantees: resizing within [min, max] keeps the
SpyreTensorLayout, so one compiled graph serves every size in the range.

What the compiler does with the declared dimension: a ``for_each_tile`` loop
over it is built at the declared maximum and runs for the size the launch reads,
so one kernel is right at every size. ``tests/inductor/test_symbolic_loop_e2e.py``
is the full statement of that, and the positive tests here repeat it with this
file's odd granularity.

What it refuses: a plain op whose own iteration space depends on the declared
dimension, outside any ``for_each_tile`` loop. Such a kernel has no loop to
carry the size, so it would be built for one size and silently wrong at every
other (#5937, how this file read before). It is refused at compile with a
message that names ``for_each_tile``. Tiling such an op automatically is later
work.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor.exc import InductorError
from torch._inductor.utils import fresh_cache

from utils_inductor import expects_raise

import torch_spyre  # noqa: F401
import torch_spyre._inductor.config as spyre_config
from torch_spyre._C import get_spyre_tensor_layout
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.execution.kernel_cache import get_cache_root_dir

DEVICE = torch.device("spyre")
COLS = 1024
SPEC = {"min": 140, "max": 630, "granularity": 70}
TILE = SPEC["granularity"]
# The compile size first (the minimum), then the maximum, then sizes between.
SIZES = [140, 630, 280, 560, 210]

# fp16 results of ops on values in [0, 1), against an fp32 reference: one ulp
# is about 1e-3 near 1 and sum(dim=0) over 140 rows measured a relative error
# of 3e-3.
ATOL = 1e-2
RTOL = 2e-2


def _sum_dim0(a):
    return a.sum(dim=0)


PLAIN_CASES = {
    "gelu": F.gelu,
    "sum_dim0": _sum_dim0,
}


def tiled_gelu(a):
    """The same gelu, as one tile of TILE rows per trip of a for_each_tile loop."""
    rows = a.shape[0]
    torch._check(rows >= SPEC["min"])
    torch._check(rows <= SPEC["max"])
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, F.gelu(tile)

    _carry, out = for_each_tile(body, (a,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


def _kernel_cache_entries():
    root = get_cache_root_dir()
    return sum(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root))


@pytest.fixture(autouse=True)
def _fresh_compile_state():
    torch._dynamo.reset()
    counters.clear()
    with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
        yield


@pytest.mark.parametrize("case", sorted(PLAIN_CASES))
@expects_raise
def test_a_plain_op_over_the_declared_dim_is_refused(case):
    fn = PLAIN_CASES[case]
    x = torch.rand(SPEC["max"], COLS, dtype=torch.float16)
    y = x.to(DEVICE, dynamic={0: SPEC})
    y.resize_((SPEC["min"], COLS))

    with pytest.raises(InductorError, match="outside any for_each_tile loop"):
        torch.compile(fn)(y)


def test_a_loop_over_the_declared_dim_is_right_at_every_size():
    """One compiled fn over one reserved tensor resized through SIZES.

    The host tensor holds max rows, so after any in-place resize the first n
    device rows are x[:n], which is what the reference is computed from.
    """
    x = torch.rand(SPEC["max"], COLS, dtype=torch.float16)
    y = x.to(DEVICE, dynamic={0: SPEC})
    layout = str(get_spyre_tensor_layout(y))
    compiled = torch.compile(tiled_gelu, fullgraph=True)

    for n in SIZES:
        if y.size(0) != n:
            y.resize_((n, COLS))
        out = compiled(y).cpu().float()
        assert str(get_spyre_tensor_layout(y)) == layout, f"layout changed at {n} rows"
        torch.testing.assert_close(
            out, F.gelu(x[:n].float()), atol=ATOL, rtol=RTOL, msg=f"at {n} rows"
        )

    assert counters["stats"]["unique_graphs"] == 1
    assert _kernel_cache_entries() == 1


def test_the_reserved_tail_is_never_read():
    # NaN above the logical rows, transferred at max then shrunk, so the
    # reserved space really holds NaN. A loop that ran past the logical size
    # would surface it.
    rows = SPEC["min"]
    x = torch.rand(SPEC["max"], COLS, dtype=torch.float16)
    x[rows:] = float("nan")
    y = x.to(DEVICE, dynamic={0: SPEC})
    y.resize_((rows, COLS))

    out = torch.compile(tiled_gelu, fullgraph=True)(y).cpu().float()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, F.gelu(x[:rows].float()), atol=ATOL, rtol=RTOL)
