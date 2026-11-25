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

"""Compiled reductions on sharded tensors (non-unified kernels).

A reduction over dims that do not include the shard dim (shard-local) runs on
every shard independently: core c reads input shard c and writes output shard
c, and the result stays sharded along the remapped shard dim.

A reduction over the shard dim (cross-shard) runs as one program: core c reads
input shard c, DeepTools' cross-core reduction combines the per-core partials,
and the result is an ordinary (unsharded) tensor.

The device tests need ``SENCORES`` in (8, 16) and ``FLEX_DEVICE=EMUL1p5`` at
process start. ``test_device_suite_in_emulation_shape`` runs this file in a
subprocess for each shape, so a plain ``pytest`` run (and CI) covers them; the
per-shape tests skip in a process without that environment.

Results are compared with the reduction run on the CPU, so the reference does
not depend on an unsharded device kernel being correct. Inputs are small
integers, so ``sum``, ``amax`` and ``amin`` are exact in fp16 and those
comparisons are ``torch.equal``; ``mean`` is compared against an fp32 CPU
reference within fp16 rounding (exactly, where the input makes the mean
representable) and, additionally, with the same reduction compiled on the
unsharded device tensor — exact for shard-local ``mean`` (identical per-core
arithmetic), within rounding order for cross-shard ``mean`` (the ring sums
partials in a different order than one kernel; fp16 addition is not
associative).
"""

from __future__ import annotations


import pytest
import torch
from device_env import num_shards
from device_suite import make_device_suite_test, requires_emulation
from torch.distributed.tensor import Shard

from torch_spyre.execution import kernel_runner


def expects_raise(test):
    """Tag a rejection test (see tests/inductor/utils_inductor.py)."""
    test._expects_raise = True
    return test


# The subprocess wrapper that re-runs this file per (mode, shards)
# shape on a VF-mode host, and the gating of the device tests below,
# live in device_suite.py (shared with the other sharding device
# files); conftest.py guards that they actually executed.
test_device_suite_in_emulation_shape = make_device_suite_test(__file__)


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def launched(monkeypatch):
    """Code dirs of every SDSC kernel launched in the test."""
    runs: list[str] = []
    run = kernel_runner.SpyreSDSCKernelRunner.run

    def record(self, *args):
        runs.append(self.code_dir)
        return run(self, *args)

    monkeypatch.setattr(kernel_runner.SpyreSDSCKernelRunner, "run", record)
    return runs


def _sharded(cpu: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    return torch.spyre.to_sharded(cpu, Shard(dim), num_shards=n)


def _small_ints(shape, dtype=torch.float16, seed=0) -> torch.Tensor:
    # Integers in [-8, 8): sums over a few elements stay exact in fp16.
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 8, shape, generator=generator).to(dtype)


@requires_emulation
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x.sum(0),
        lambda x: x.amax(0),
        lambda x: x.amin(0),
    ],
    ids=["sum", "amax", "amin"],
)
def test_shard_local_reduction_matches_cpu(launched, fn):
    """Reducing a dim before the shard dim keeps the result sharded.

    Input [3, 4n, 64] is Shard(1); reducing dim 0 leaves [4n, 64] sharded on
    dim 0 (the shard dim moves down by one), one shard per core.
    """
    n = num_shards()
    src = _small_ints((3, 4 * n, 64))
    x = _sharded(src, 1, n)

    y = torch.compile(fn, dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.spyre.is_sharded(y)
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert tuple(y.shape) == (4 * n, 64)
    assert torch.equal(y.cpu(), fn(src))


@requires_emulation
def test_shard_local_reduction_of_rank2_tensor(launched):
    """[4, 64n] Shard(1).sum(0) -> [64n] Shard(0): each core sums its columns."""
    n = num_shards()
    src = _small_ints((4, 64 * n), seed=1)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)

    assert len(launched) == 1, launched
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert torch.equal(y.cpu(), src.sum(0))


@requires_emulation
def test_shard_local_reduction_after_the_shard_dim(launched):
    """Reducing a dim after the shard dim keeps the shard dim in place.

    Input [3, 4n, 5, 64] is Shard(1); reducing dim 2 leaves [3, 4n, 64]
    sharded on dim 1 (the shard dim keeps its position — the
    ``expected = s`` branch of the output-placement check).
    """
    n = num_shards()
    src = _small_ints((3, 4 * n, 5, 64), seed=6)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(2), dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.spyre.is_sharded(y)
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(1) and info.num_shards == n
    assert tuple(y.shape) == (3, 4 * n, 64)
    assert torch.equal(y.cpu(), src.sum(2))


@requires_emulation
def test_shard_local_reduction_of_multiple_dims(launched):
    """Reducing several dims at once, none of them the shard dim.

    Input [3, 4n, 5, 64] is Shard(1); reducing dims 0 and 2 leaves
    [4n, 64] sharded on dim 0 (the shard dim moves down past one reduced
    dim — the ``s - sum(r < s)`` branch), and exercises pinning more than
    one reduction var to split 1.
    """
    n = num_shards()
    src = _small_ints((3, 4 * n, 5, 64), seed=7)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum((0, 2)), dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.spyre.is_sharded(y)
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert tuple(y.shape) == (4 * n, 64)
    assert torch.equal(y.cpu(), src.sum((0, 2)))


@requires_emulation
def test_each_core_reduces_only_its_own_shard():
    """Every input shard gets a distinct value, so a core reading another
    core's shard (or writing another core's output shard) shows up."""
    n = num_shards()
    rows, cols = 3, 4 * n
    shard_of_col = torch.arange(cols) // 4
    src = shard_of_col.view(1, cols, 1) * 4 + torch.arange(rows).view(rows, 1, 1)
    src = src.expand(rows, cols, 64).to(torch.float16).contiguous()
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)

    assert torch.equal(y.cpu(), src.sum(0))


@requires_emulation
def test_shard_local_mean_close_to_cpu():
    """mean is shard-local too: every core divides by the same reduced count.

    Two references, each covering the other's blind spot:

    - The fp32 CPU mean, within tolerances. The device's division for mean is
      measurably inexact (~1% relative, ~0.01 absolute on these inputs; it
      does not match a single fp16 rounding), so exactness is not available
      here. A wrong division (by a local extent, by the shard count) is off
      by a factor, far outside these tolerances.
    - The same reduction compiled on the unsharded tensor, exactly. Both
      paths run the same device arithmetic, so any difference — a core
      reading another shard, a different divisor on the sharded path — fails
      this equality while fitting inside the CPU tolerance above. It cannot
      catch an error the unsharded kernel shares, which is what the CPU
      reference is for.
    """
    n = num_shards()
    src = _small_ints((3, 4 * n, 64), seed=2)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.mean(0), dynamic=False)(x)

    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0)
    assert torch.allclose(y.cpu().float(), src.float().mean(0), rtol=2e-2, atol=2e-2)
    device_ref = torch.compile(lambda t: t.mean(0), dynamic=False)(src.to("spyre"))
    assert torch.equal(y.cpu(), device_ref.cpu())


@requires_emulation
def test_reduction_feeds_a_sharded_pointwise_op(launched):
    """The sharded reduction result is a valid input to a later compile."""
    n = num_shards()
    src = _small_ints((3, 4 * n, 64), seed=3)
    other = _small_ints((4 * n, 64), seed=4)
    x = _sharded(src, 1, n)
    b = _sharded(other, 0, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)
    z = torch.compile(lambda u, v: u + v, dynamic=False)(y, b)

    assert len(launched) == 2, launched
    assert torch.equal(z.cpu(), src.sum(0) + other)


@requires_emulation
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x.sum(1),
        lambda x: x.amax(1),
        lambda x: x.amin(1),
        lambda x: x.sum(1, keepdim=True),
    ],
    ids=["sum", "amax", "amin", "sum_keepdim"],
)
def test_cross_shard_reduction_matches_cpu(launched, fn):
    """Reducing the shard dim combines the N partials in one kernel.

    Input [3, 4n, 64] is Shard(1); reducing dim 1 needs every shard. The
    result is an ordinary tensor and, for small integers, exact.
    """
    n = num_shards()
    src = _small_ints((3, 4 * n, 64))
    x = _sharded(src, 1, n)

    y = torch.compile(fn, dynamic=False)(x)

    assert len(launched) == 1, launched
    assert not torch.spyre.is_sharded(y)
    assert torch.equal(y.cpu(), fn(src))


@requires_emulation
@pytest.mark.parametrize(
    "dtype, width",
    # One and two sticks per row: a two-stick row (fp16 or fp32) has
    # shard-local strides different from the global ones — admitted since
    # issue #5827 was lifted.
    [
        (torch.float16, 64),
        (torch.float16, 128),
        (torch.float32, 32),
        (torch.float32, 64),
    ],
    ids=["fp16_1stick", "fp16_2stick", "fp32_1stick", "fp32_2stick"],
)
def test_cross_shard_sum_of_rank2_tensor(launched, dtype, width):
    """[4n, w] Shard(0).sum(0) -> [w]: a column sum across all shards."""
    n = num_shards()
    src = _small_ints((4 * n, width), dtype=dtype, seed=5)
    x = _sharded(src, 0, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.equal(y.cpu(), src.sum(0))


@requires_emulation
def test_cross_shard_sum_combines_every_shard():
    """Each shard holds a distinct value, so the sum is wrong if any shard's
    partial is missing, counted twice, or read from another shard."""
    n = num_shards()
    cols = 4 * n
    shard_of_col = (torch.arange(cols) // 4).view(1, cols, 1)
    src = shard_of_col.expand(3, cols, 64).to(torch.float16).contiguous()
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(1), dynamic=False)(x)

    # Every output element is 4 * (0 + 1 + ... + n-1).
    assert torch.equal(y.cpu(), src.sum(1))
    assert y.cpu()[0, 0].item() == 4 * n * (n - 1) // 2


@requires_emulation
def test_cross_shard_mean_close_to_cpu():
    """mean divides once by the global count (sum first, then divide).

    Not exact in fp16; the tolerance is 0.5% relative, and the unsharded
    device mean is measured at 0.4% on the same inputs.

    The sharded result is also compared with the same reduction compiled on
    the unsharded device tensor. That one cannot be exact: the cross-shard
    path sums per-core partials in ring order and the unsharded kernel
    reduces in its own order, and fp16 addition is not associative (measured
    at just under 0.5% on these inputs). The 1% tolerance still catches a
    sharded path that combines or divides differently by more than rounding
    order; the division count itself is pinned exactly by
    ``test_cross_shard_mean_divides_by_the_global_extent``.
    """
    n = num_shards()
    generator = torch.Generator().manual_seed(7)
    src = (torch.rand((3, 4 * n, 64), generator=generator) * 4 + 1).to(torch.float16)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.mean(1), dynamic=False)(x)

    expected = src.float().mean(1)
    assert torch.allclose(y.cpu().float(), expected, rtol=5e-3, atol=0)
    device_ref = torch.compile(lambda t: t.mean(1), dynamic=False)(src.to("spyre"))
    assert torch.allclose(y.cpu().float(), device_ref.cpu().float(), rtol=1e-2, atol=0)


@requires_emulation
def test_cross_shard_mean_divides_by_the_global_extent():
    """Pins how the backend divides for mean: once, by the global extent.

    Every element of a reduced column is the same constant c, so the true
    mean is c exactly and the comparison is ``torch.equal``. A backend that
    divided by a core's local extent and let the ring sum the per-core means
    would return c * num_shards instead; one that divided by the shard count
    only would return c * (shard extent / num_shards).
    """
    n = num_shards()
    c = 3
    src = torch.full((3, 4 * n, 64), float(c), dtype=torch.float16)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.mean(1), dynamic=False)(x)

    assert torch.equal(y.cpu(), torch.full((3, 64), float(c), dtype=torch.float16))


@requires_emulation
def test_cross_shard_result_feeds_an_unsharded_op():
    """The cross-shard result is an ordinary tensor: later ops treat it so."""
    n = num_shards()
    src = _small_ints((3, 4 * n, 64), seed=6)
    other = _small_ints((3, 64), seed=8)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(1), dynamic=False)(x)
    z = torch.compile(lambda u, v: u + v, dynamic=False)(y, other.to("spyre"))

    assert torch.equal(z.cpu(), src.sum(1) + other)


@requires_emulation
def test_cross_shard_result_is_one_ordinary_chunk():
    """The cross-shard result is allocated like any unsharded tensor.

    It has one chunk, placed by the default directive (domain 0), the same as
    a tensor copied from the CPU. If a placement policy for these outputs is
    added later, this test is where that change shows.
    """
    from torch_spyre import _C

    n = num_shards()
    src = _small_ints((3, 4 * n, 64), seed=9)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(1), dynamic=False)(x)
    plain = torch.zeros(3, 64, dtype=torch.float16).to("spyre")

    assert not torch.spyre.is_sharded(y)
    chunks = _C.get_composite_address(y).chunks()
    assert len(chunks) == 1
    assert chunks[0].domain_id == _C.get_composite_address(plain).chunks()[0].domain_id
    assert chunks[0].domain_id == 0
    assert torch.equal(y.cpu(), src.sum(1))


@requires_emulation
@expects_raise
@pytest.mark.parametrize(
    "fn, match",
    [
        # TODO: positive once a reduction can split more than one reduced dim
        # across cores (DeepTools limit) or combine in two steps.
        (lambda x: x.sum(), "must reduce the shard dim only"),
        (lambda x: x.sum((0, 1)), "must reduce the shard dim only"),
        # A real limit for now: reducing the stick dim changes each shard's
        # device layout.
        (lambda x: x.sum(-1), "reductions over the stick dim"),
        # TODO(#6170): positive once the shard-local keepdim output layout
        # matches the default shard layout (it gets a size-1 outer device dim
        # today).
        (lambda x: x.sum(0, keepdim=True), "default device layout"),
    ],
    ids=["full", "includes_shard_dim", "stick_dim", "local_keepdim"],
)
def test_unsupported_reductions_fail_at_compile(fn, match):
    n = num_shards()
    x = _sharded(_small_ints((3, 4 * n, 64)), 1, n)

    with pytest.raises(RuntimeError, match=match):
        torch.compile(fn, dynamic=False)(x)


@requires_emulation
def test_reduction_with_unequal_local_strides(launched):
    """Shard(0) of [4n, 3, 64] reducing dim 1: the input's device layout puts
    dim 1 inside the shard dim, so shard-local strides differ from the global
    ones. Admitted since the codegen guard checks the split against the shard
    dim instead of unified-model stride distances (issue #5827); the backend
    walks each core's tile with the shard-local layout."""
    n = num_shards()
    src = _small_ints((4 * n, 3, 64), seed=10)
    x = _sharded(src, 0, n)

    y = torch.compile(lambda t: t.sum(1), dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.spyre.is_sharded(y)
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert torch.equal(y.cpu(), src.sum(1))


@requires_emulation
def test_shard_local_reduction_of_shard_dim_inside_wider_rows(launched):
    """[8, 4n, 64] Shard(1).sum(0): the reduced dim sits outside the shard
    dim, whose shard-local stride differs from the global one (the shard dim
    is a middle device dim, not the innermost non-stick one)."""
    n = num_shards()
    src = _small_ints((8, 4 * n, 64), seed=11)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.spyre.is_sharded(y)
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert tuple(y.shape) == (4 * n, 64)
    assert torch.equal(y.cpu(), src.sum(0))


@requires_emulation
def test_cross_shard_reduction_with_unequal_local_strides(launched):
    """[8, 4n, 64] Shard(1).sum(1): a cross-shard ring reduction whose input
    shards have shard-local strides different from the global ones — the
    lifted stride restriction composes with the ring combine."""
    n = num_shards()
    src = _small_ints((8, 4 * n, 64), seed=12)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(1), dynamic=False)(x)

    assert len(launched) == 1, launched
    assert not torch.spyre.is_sharded(y)
    assert torch.equal(y.cpu(), src.sum(1))
