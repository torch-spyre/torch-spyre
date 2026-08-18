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

"""Regression tests: a buffer must not stay resident in the LX scratchpad
across an opaque extern kernel call.

Background. The LX scratchpad is a fixed per-core resource shared by *every*
compiled Spyre program, and a resident buffer is handed from one kernel launch
to the next by its LX offset alone -- it is not threaded through the generated
wrapper as a tensor. An extern kernel (``FallbackKernel``, e.g. a
``torch.library.custom_op``) is opaque: its body may launch other compiled
programs, and those programs allocate the same LX offsets. So a buffer left
resident across such a call is silently overwritten and its consumer reads the
other program's data.

The graph shape that triggers it is an ordinary transformer residual:

    r = f(x)                 # written before the op, read after it,
    t = x @ w                # and never passed *to* the op
    o = opaque_custom_op(t)
    out = r + o

``r`` is fused into the producing kernel and left in LX; the consumer reads it
back after the opaque call. This is the shape behind the ``spyre-inference``
whole-model-compile corruption (``SPYRE_FORCE_COMPILE_ATTN=1`` with
``-cc.mode STOCK_TORCH_COMPILE``): attention is an opaque custom op, and
compiling the paged-attention kernel makes its body launch a nested program.

``_extern_kernel_in_live_range`` is the guard; :func:`test_extern_kernel_in_live_range`
is the deterministic signal that it still rejects the spanning case, and
:func:`test_nested_launch_does_not_clobber_residual` asserts the behaviour on
device.
"""

from unittest.mock import Mock

import pytest
import torch

import torch_spyre  # noqa: F401
from torch._inductor.ir import ExternKernel
from torch_spyre._inductor.scratchpad.allocator import _extern_kernel_in_live_range


DEVICE = "spyre"
DTYPE = torch.float16
N = 128
INNER_N = 64


class _FakeGraph:
    """Minimal stand-in exposing only what the guard reads."""

    def __init__(self, operations):
        self.operations = operations


def test_extern_kernel_in_live_range():
    """The guard must fire for a buffer merely *live across* an extern kernel.

    Being read/written by the extern kernel (index in ``uses``) was already
    rejected; spanning one without touching it is the case that silently
    corrupted results.
    """
    extern = Mock(spec=ExternKernel)
    plain = Mock()

    spanning = _FakeGraph([plain, extern, plain])
    assert _extern_kernel_in_live_range(spanning, [0, 2]), (
        "a buffer live across an extern kernel must be rejected"
    )
    assert _extern_kernel_in_live_range(spanning, [1]), "extern kernel user"

    # Live range entirely before the extern kernel: LX residency is fine.
    assert not _extern_kernel_in_live_range(spanning, [0])
    assert not _extern_kernel_in_live_range(_FakeGraph([plain, plain, plain]), [0, 2])
    assert not _extern_kernel_in_live_range(spanning, [])


@pytest.mark.parametrize("layers", [1, 4])
def test_nested_launch_does_not_clobber_residual(layers):
    """Launching an unrelated compiled program inside an opaque custom op must
    not change the outer compiled program's result.

    The nested program's result is discarded, so the two runs must agree
    bit-for-bit. Before the fix they did not: the nested program's data landed
    in the LX scratchpad slot holding the outer graph's residual.
    """
    launch = False

    inner = torch.compile(lambda t: t * 2.0 + 1.0, dynamic=False)
    inner_arg = torch.ones(INNER_N, INNER_N, dtype=DTYPE, device=DEVICE)
    inner(inner_arg)  # compile + warm before the outer program ever runs

    @torch.library.custom_op("test_lx_clobber::opaque", mutates_args=())
    def opaque(x: torch.Tensor) -> torch.Tensor:
        if launch:
            inner(inner_arg)  # discarded; only the nested launch is under test
        return x.clone()

    @opaque.register_fake
    def _(x):
        return torch.empty_like(x)

    def model(x, ws):
        h = x
        for w in ws:
            r = h * 1.5 + 0.25  # live across the opaque op, not an argument
            t = h @ w
            o = torch.ops.test_lx_clobber.opaque(t)
            h = r + o
        return h

    torch.manual_seed(0)
    x = torch.randn(N, N, dtype=DTYPE).to(DEVICE)
    ws = [(torch.randn(N, N, dtype=DTYPE) / (N**0.5)).to(DEVICE) for _ in range(layers)]

    compiled = torch.compile(model, dynamic=False)
    without = compiled(x, ws).to("cpu").float()
    launch = True
    with_launch = compiled(x, ws).to("cpu").float()
    launch = False

    torch.testing.assert_close(with_launch, without, atol=0, rtol=0)
