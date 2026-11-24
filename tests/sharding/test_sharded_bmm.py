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

"""Compiled torch.bmm on batch-sharded tensors (non-unified kernels).

The expert-parallel shape: both operands sharded on the batch dim, so core
c multiplies input shard c of both operands and writes output shard c. The
result stays sharded on the batch dim.

torch-spyre lowers bmm to one Reduction node with two loaders (a fused
multiply-accumulate over the contraction dim), so this is a two-read
shard-local reduction -- the same class as the single-read reductions in
test_sharded_reduction.py, admitted by the same planner.

The device tests need ``SENCORES`` in (8, 16) and a Flex device mode that
admits sharded allocations (``FLEX_DEVICE=EMUL1p5``), fixed at process
start. ``test_device_suite_in_emulation_shape`` runs this file in a
subprocess for each shape, so a plain ``pytest`` run (and CI) covers all of
them; the per-shape tests skip in a process without that environment.

Inputs are small integers, exact in fp16: products stay below 16 and sums
over the contraction dim below 2048, so comparisons against the CPU
reference are ``torch.equal``.
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
    # Tests compile the same lambda shapes; a clean Dynamo cache keeps
    # recompile limits and earlier guards from changing the path.
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


def _small_ints(shape, seed=0) -> torch.Tensor:
    # Integers in [-3, 4): bmm products stay below 16 and sums over the
    # contraction dim below 2048, exact in fp16.
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, shape, generator=generator).to(torch.float16)


def _bmm_shapes(expert_per_shard, t, d, out):
    """(x shape, w shape): batch = expert_per_shard * num_shards."""
    n = num_shards()
    return (expert_per_shard * n, t, d), (expert_per_shard * n, d, out)


@requires_emulation
@pytest.mark.parametrize(
    "expert_per_shard, t, d, out",
    [
        (2, 64, 64, 64),  # the #6174 smoke shape
        (2, 32, 128, 64),  # T != d != out, K spans two sticks
        (1, 64, 64, 32),  # one expert per shard, narrow output
    ],
    ids=["smoke_shape", "asymmetric", "one_expert"],
)
def test_batch_sharded_bmm_matches_cpu(launched, expert_per_shard, t, d, out):
    """Both operands Shard(0): core c multiplies shard c of each."""
    n = num_shards()
    x_shape, w_shape = _bmm_shapes(expert_per_shard, t, d, out)
    x_cpu = _small_ints(x_shape)
    w_cpu = _small_ints(w_shape, seed=1)
    x, w = _sharded(x_cpu, 0, n), _sharded(w_cpu, 0, n)

    y = torch.compile(lambda a, b: torch.bmm(a, b), dynamic=False)(x, w)

    assert len(launched) == 1, launched
    assert torch.spyre.is_sharded(y)
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert tuple(y.shape) == x_shape[:1] + (t, out)
    assert torch.equal(y.cpu(), torch.bmm(x_cpu, w_cpu))


@requires_emulation
def test_batch_sharded_bmm_feeds_a_sharded_pointwise_op(launched):
    """The sharded bmm result is a valid input to a later compiled call."""
    n = num_shards()
    x_shape, w_shape = _bmm_shapes(2, 64, 64, 64)
    x_cpu = _small_ints(x_shape)
    w_cpu = _small_ints(w_shape, seed=1)
    b_cpu = _small_ints(x_shape[:1] + (64, 64), seed=2)
    x, w = _sharded(x_cpu, 0, n), _sharded(w_cpu, 0, n)
    b = _sharded(b_cpu, 0, n)

    y = torch.compile(lambda a, c: torch.bmm(a, c), dynamic=False)(x, w)
    z = torch.compile(lambda u, v: u + v, dynamic=False)(y, b)

    assert len(launched) == 2, launched
    assert torch.spyre.is_sharded(z)
    assert torch.equal(z.cpu(), torch.bmm(x_cpu, w_cpu) + b_cpu)


@requires_emulation
@pytest.mark.parametrize(
    "w_shard_dim",
    [1, None],
    ids=["different_placement", "unsharded_operand"],
)
@expects_raise
def test_batch_sharded_bmm_rejects_mixed_operands(w_shard_dim):
    """Fail closed: every operand must be sharded on the same dim.

    Shard(0) x Shard(1) is a different placement (the contraction dim
    sharded on one side only); a plain unsharded operand is a mixed read.
    Both need redistribution, which non-unified kernels do not do.
    """
    n = num_shards()
    x_shape, w_shape = _bmm_shapes(2, 64, 64, 64)
    x_cpu = _small_ints(x_shape)
    w_cpu = _small_ints(w_shape, seed=1)
    x = _sharded(x_cpu, 0, n)
    w = _sharded(w_cpu, w_shard_dim, n) if w_shard_dim is not None else w_cpu.to("spyre")

    # A different placement hits the placement-agreement check; a plain
    # unsharded operand hits the reduction's no-unsharded-reads check.
    expected = (
        "cannot combine operands sharded"
        if w_shard_dim is not None
        else "unsharded operand only in pointwise"
    )
    with pytest.raises(RuntimeError, match=expected):
        torch.compile(lambda a, b: torch.bmm(a, b), dynamic=False)(x, w)
