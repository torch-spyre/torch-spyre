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

Results are compared with the same reduction on unsharded device tensors,
which run the same device arithmetic. Inputs are small integers, so ``sum``,
``max`` and ``min`` are exact in fp16 and the comparison is ``torch.equal``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from xml.etree import ElementTree

import pytest
import torch
from device_env import device_available, emulation_ready, mode_usable
from torch.distributed.tensor import Shard

from torch_spyre.execution import kernel_runner


def expects_raise(test):
    """Tag a rejection test (see tests/inductor/utils_inductor.py)."""
    test._expects_raise = True
    return test


def _num_shards() -> int:
    return int(os.environ.get("SENCORES", "0") or 0)


_requires_emulation = pytest.mark.skipif(
    not emulation_ready(),
    reason="needs SENCORES in (8, 16) and FLEX_DEVICE=EMUL1p5 at process start; "
    "test_device_suite_in_emulation_shape runs these in that shape",
)

_DEVICE_MODES = {"EMUL1p5": {"FLEX_DEVICE": "EMUL1p5"}}


@pytest.mark.skipif(
    emulation_ready() or not device_available(),
    reason="needs a Spyre device; in an emulation shape the tests below run directly",
)
@pytest.mark.parametrize(
    ("mode", "shards"),
    [(m, s) for m in sorted(_DEVICE_MODES) for s in (8, 16)],
)
def test_device_suite_in_emulation_shape(mode, shards, tmp_path):
    if torch.spyre.is_initialized():
        pytest.skip(
            "the Spyre runtime is already open in this process; run "
            "tests/sharding/test_sharded_reduction.py on its own"
        )
    env = dict(os.environ)
    env["SENCORES"] = str(shards)
    env.update(_DEVICE_MODES[mode])
    env.pop("PYTEST_ADDOPTS", None)
    usable, reason = mode_usable(_DEVICE_MODES[mode], shards)
    if not usable:
        pytest.skip(
            f"Flex device mode {mode} cannot open the card on this machine "
            f"({reason}); needs bound VFs"
        )
    junit = tmp_path / "child.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-p",
            "no:cacheprovider",
            f"--junit-xml={junit}",
            __file__,
        ],
        env=env,
        timeout=1200,
    )
    outcomes = {}
    failures = []
    if junit.exists():
        for case in ElementTree.parse(junit).getroot().iter("testcase"):
            children = [c.tag for c in case]
            outcome = next(
                (t for t in ("failure", "error", "skipped") if t in children), "passed"
            )
            outcomes[case.get("name")] = outcome
            for tag in ("failure", "error"):
                node = case.find(tag)
                if node is not None:
                    message = (node.get("message") or "").splitlines()
                    failures.append(
                        f"{case.get('name')}: {message[0] if message else tag}"
                    )
    if result.returncode != 0:
        detail = (
            "; ".join(failures)
            if failures
            else "no failing case in the JUnit report (collection or setup error)"
        )
        raise AssertionError(
            f"child pytest failed for {shards} shards in {mode} mode "
            f"(exit {result.returncode}). {detail}"
        )
    wrappers = {
        f"test_device_suite_in_emulation_shape[{m}-{s}]"
        for m in sorted(_DEVICE_MODES)
        for s in (8, 16)
    }
    assert {k for k, v in outcomes.items() if v == "skipped"} == wrappers, outcomes
    ran = {k for k, v in outcomes.items() if v == "passed"}
    assert ran and ran | wrappers == set(outcomes), outcomes


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


def _reference(fn, src: torch.Tensor) -> torch.Tensor:
    return torch.compile(fn, dynamic=False)(src.to("spyre")).cpu()


@_requires_emulation
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x.sum(0),
        lambda x: x.amax(0),
        lambda x: x.amin(0),
    ],
    ids=["sum", "amax", "amin"],
)
def test_shard_local_reduction_matches_unsharded(launched, fn):
    """Reducing a dim before the shard dim keeps the result sharded.

    Input [3, 4n, 64] is Shard(1); reducing dim 0 leaves [4n, 64] sharded on
    dim 0 (the shard dim moves down by one), one shard per core.
    """
    n = _num_shards()
    src = _small_ints((3, 4 * n, 64))
    x = _sharded(src, 1, n)

    y = torch.compile(fn, dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.spyre.is_sharded(y)
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert tuple(y.shape) == (4 * n, 64)
    assert torch.equal(y.cpu(), _reference(fn, src))
    assert torch.equal(y.cpu(), fn(src))


@_requires_emulation
def test_shard_local_reduction_of_rank2_tensor(launched):
    """[4, 64n] Shard(1).sum(0) -> [64n] Shard(0): each core sums its columns."""
    n = _num_shards()
    src = _small_ints((4, 64 * n), seed=1)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)

    assert len(launched) == 1, launched
    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0) and info.num_shards == n
    assert torch.equal(y.cpu(), src.sum(0))


@_requires_emulation
def test_each_core_reduces_only_its_own_shard():
    """Every input shard gets a distinct value, so a core reading another
    core's shard (or writing another core's output shard) shows up."""
    n = _num_shards()
    rows, cols = 3, 4 * n
    shard_of_col = torch.arange(cols) // 4
    src = shard_of_col.view(1, cols, 1) * 4 + torch.arange(rows).view(rows, 1, 1)
    src = src.expand(rows, cols, 64).to(torch.float16).contiguous()
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)

    assert torch.equal(y.cpu(), src.sum(0))


@_requires_emulation
def test_mean_matches_unsharded():
    """mean is shard-local too: every core divides by the same reduced count.

    Device mean differs from the CPU mean in the last bits (measured on
    unsharded tensors too), so the reference is the unsharded device mean.
    """
    n = _num_shards()
    src = _small_ints((3, 4 * n, 64), seed=2)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.mean(0), dynamic=False)(x)

    info = torch.spyre.sharding_info(y)
    assert info.placement == Shard(0)
    assert torch.equal(y.cpu(), _reference(lambda t: t.mean(0), src))


@_requires_emulation
def test_reduction_feeds_a_sharded_pointwise_op(launched):
    """The sharded reduction result is a valid input to a later compile."""
    n = _num_shards()
    src = _small_ints((3, 4 * n, 64), seed=3)
    other = _small_ints((4 * n, 64), seed=4)
    x = _sharded(src, 1, n)
    b = _sharded(other, 0, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)
    z = torch.compile(lambda u, v: u + v, dynamic=False)(y, b)

    assert len(launched) == 2, launched
    assert torch.equal(z.cpu(), src.sum(0) + other)


@_requires_emulation
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
    n = _num_shards()
    src = _small_ints((3, 4 * n, 64))
    x = _sharded(src, 1, n)

    y = torch.compile(fn, dynamic=False)(x)

    assert len(launched) == 1, launched
    assert not torch.spyre.is_sharded(y)
    assert torch.equal(y.cpu(), fn(src))
    assert torch.equal(y.cpu(), _reference(fn, src))


@_requires_emulation
@pytest.mark.parametrize(
    "dtype, width",
    # One stick per row in both dtypes (64 fp16 or 32 fp32 elements). A
    # two-stick fp32 row hits the shard-local stride limit (#5827).
    [(torch.float16, 64), (torch.float32, 32)],
    ids=["fp16", "fp32"],
)
def test_cross_shard_sum_of_rank2_tensor(launched, dtype, width):
    """[4n, w] Shard(0).sum(0) -> [w]: a column sum across all shards."""
    n = _num_shards()
    src = _small_ints((4 * n, width), dtype=dtype, seed=5)
    x = _sharded(src, 0, n)

    y = torch.compile(lambda t: t.sum(0), dynamic=False)(x)

    assert len(launched) == 1, launched
    assert torch.equal(y.cpu(), src.sum(0))


@_requires_emulation
def test_cross_shard_sum_combines_every_shard():
    """Each shard holds a distinct value, so the sum is wrong if any shard's
    partial is missing, counted twice, or read from another shard."""
    n = _num_shards()
    cols = 4 * n
    shard_of_col = (torch.arange(cols) // 4).view(1, cols, 1)
    src = shard_of_col.expand(3, cols, 64).to(torch.float16).contiguous()
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(1), dynamic=False)(x)

    # Every output element is 4 * (0 + 1 + ... + n-1).
    assert torch.equal(y.cpu(), src.sum(1))
    assert y.cpu()[0, 0].item() == 4 * n * (n - 1) // 2


@_requires_emulation
def test_cross_shard_mean_close_to_cpu():
    """mean divides once by the global count (sum first, then divide).

    Not exact in fp16; the tolerance is 0.5% relative, and the unsharded
    device mean is measured at 0.4% on the same inputs.
    """
    n = _num_shards()
    generator = torch.Generator().manual_seed(7)
    src = (torch.rand((3, 4 * n, 64), generator=generator) * 4 + 1).to(torch.float16)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.mean(1), dynamic=False)(x)

    expected = src.float().mean(1)
    assert torch.allclose(y.cpu().float(), expected, rtol=5e-3, atol=0)


@_requires_emulation
def test_cross_shard_result_feeds_an_unsharded_op():
    """The cross-shard result is an ordinary tensor: later ops treat it so."""
    n = _num_shards()
    src = _small_ints((3, 4 * n, 64), seed=6)
    other = _small_ints((3, 64), seed=8)
    x = _sharded(src, 1, n)

    y = torch.compile(lambda t: t.sum(1), dynamic=False)(x)
    z = torch.compile(lambda u, v: u + v, dynamic=False)(y, other.to("spyre"))

    assert torch.equal(z.cpu(), src.sum(1) + other)


@_requires_emulation
def test_cross_shard_result_is_one_ordinary_chunk():
    """The cross-shard result is allocated like any unsharded tensor.

    It has one chunk, placed by the default directive (domain 0), the same as
    a tensor copied from the CPU. If a placement policy for these outputs is
    added later, this test is where that change shows.
    """
    from torch_spyre import _C

    n = _num_shards()
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


@_requires_emulation
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
        # TODO: positive once the shard-local keepdim output layout matches
        # the default shard layout (it gets a size-1 outer device dim today).
        (lambda x: x.sum(0, keepdim=True), "default device layout"),
    ],
    ids=["full", "includes_shard_dim", "stick_dim", "local_keepdim"],
)
def test_unsupported_reductions_fail_at_compile(fn, match):
    n = _num_shards()
    x = _sharded(_small_ints((3, 4 * n, 64)), 1, n)

    with pytest.raises(RuntimeError, match=match):
        torch.compile(fn, dynamic=False)(x)


@_requires_emulation
@expects_raise
def test_reduction_with_unequal_local_strides_fails():
    """Shard(0) of [4n, 3, 64] reducing dim 1: the input's device layout puts
    dim 1 outside the shard dim, so shard-local strides differ from the
    global ones. TODO: positive once DeepTools uses per-core layout sizes
    (issue #5827), as for pointwise ops."""
    n = _num_shards()
    x = _sharded(_small_ints((4 * n, 3, 64)), 0, n)

    with pytest.raises(RuntimeError, match="shard-local strides equal to the global"):
        torch.compile(lambda t: t.sum(1), dynamic=False)(x)
