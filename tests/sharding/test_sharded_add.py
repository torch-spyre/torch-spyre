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

"""Compiled torch.add on sharded tensors (non-unified kernels).

Core c reads and writes only chunk c through its own start address.

The device tests need ``SENCORES`` in (8, 16) and a Flex device mode that
admits sharded allocations (``FLEX_DEVICE=EMUL1p5``, or ``FLEX_DEVICE=VF``
with ``FLEX_VF_ADMIT_SHARDED_ALLOCATIONS=1``), both fixed at process start.
``test_device_suite_in_emulation_shape`` runs this file in a subprocess for
each shape and mode, so a plain ``pytest`` run (and CI) covers all of them;
the per-shape tests skip in a process without that environment.

Results are compared with the same add on unsharded device tensors. A CPU
comparison is not exact: fp16 goes through the device fp16 format (L9), and
the device fp32 add differs from the CPU add by up to 1 ulp (measured on
unsharded tensors too).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from xml.etree import ElementTree

import pytest
import torch
from device_env import device_available, emulation_ready, mode_usable
from torch.distributed.tensor import Shard

from torch_spyre.execution import kernel_runner

_STRIDE_REJECTION = "shard-local strides equal to the global ones"


def expects_raise(test):
    """Tag a rejection test (see tests/inductor/utils_inductor.py)."""
    test._expects_raise = True
    return test


def _num_shards() -> int:
    return int(os.environ.get("SENCORES", "0") or 0)


_requires_emulation = pytest.mark.skipif(
    not emulation_ready(),
    reason="needs SENCORES in (8, 16) and a sharded-admitting Flex device mode "
    "(FLEX_DEVICE=EMUL1p5, or VF with FLEX_VF_ADMIT_SHARDED_ALLOCATIONS=1) at "
    "process start; test_device_suite_in_emulation_shape runs these in that shape",
)


# Both Flex device modes that admit sharded allocations (the scheduler
# factory routes them to the same multi-chunk scheduler): the EMUL1p5 device
# always admits, a plain VF device only with the opt-in flag.
_DEVICE_MODES = {
    "EMUL1p5": {"FLEX_DEVICE": "EMUL1p5"},
    "VF": {"FLEX_DEVICE": "VF", "FLEX_VF_ADMIT_SHARDED_ALLOCATIONS": "1"},
}


@pytest.mark.skipif(
    emulation_ready() or not device_available(),
    reason="needs a Spyre device; in an emulation shape the tests below run directly",
)
@pytest.mark.parametrize(
    ("mode", "shards"),
    [(m, s) for m in sorted(_DEVICE_MODES) for s in (8, 16)],
)
def test_device_suite_in_emulation_shape(mode, shards, tmp_path):
    # The child opens the card itself, so this process must not hold it yet.
    if torch.spyre.is_initialized():
        pytest.skip(
            "the Spyre runtime is already open in this process; run "
            "tests/sharding/test_sharded_add.py on its own"
        )
    env = dict(os.environ)
    env["SENCORES"] = str(shards)
    env.update(_DEVICE_MODES[mode])
    env.pop("PYTEST_ADDOPTS", None)
    # A device being present is not enough: the mode must be able to open it
    # (a PF-only host has no VFs, so EMUL1p5/VF children die opening the PCI
    # path). Probe in a throwaway process and skip rather than fail there.
    usable, reason = mode_usable(_DEVICE_MODES[mode], shards)
    if not usable:
        pytest.skip(
            f"Flex device mode {mode} cannot open the card on this machine "
            f"({reason}); needs bound VFs"
        )
    junit = tmp_path / "child.xml"
    # Output is streamed (not captured) so CI's no-output watchdog sees
    # progress; results are read from the JUnit report.
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
    # Parse the report before judging the exit code, so a nonzero exit can
    # name the failing child cases (with their messages) instead of showing
    # only "assert 1 == 0". Child stdout was streamed to this process's
    # output and is in the log above.
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
    # Tests compile the same lambda shapes; a clean Dynamo cache keeps
    # recompile limits and earlier guards from changing the path.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def launched(monkeypatch):
    """(code dir, symbolic args) of every SDSC kernel launched in the test."""
    runs: list[tuple[str, list]] = []
    run = kernel_runner.SpyreSDSCKernelRunner.run

    def record(self, *args):
        runs.append((self.code_dir, list(self._symbolic_args or [])))
        return run(self, *args)

    monkeypatch.setattr(kernel_runner.SpyreSDSCKernelRunner, "run", record)
    return runs


def _kernel_summary(code_dir: str) -> tuple[int, list, list]:
    with open(os.path.join(code_dir, "sdsc_0.json")) as f:
        sdsc = json.load(f)
    op = sdsc[next(iter(sdsc))]
    dsc = op["dscs_"][0][next(iter(op["dscs_"][0]))]
    allocs = [t for t in dsc["scheduleTree_"] if t["nodeType_"] == "allocate"]
    with open(os.path.join(code_dir, "spyreCodeDir", "spyrecode.json")) as f:
        plan = json.load(f)
    ishapes = [
        step["properties"]["ishape"]
        for step in plan["JobExecPlan"]
        if step["command"] == "ComputeOnHost"
    ]
    return op["numCoresUsed_"], allocs, ishapes


def _assert_one_non_unified_kernel(runs, n: int, num_tensors: int = 3) -> None:
    """Exactly one non-unified kernel ran, binding core c to shard c."""
    from torch_spyre._C import SymbolicArgKind

    assert len(runs) == 1, [d for d, _ in runs]
    code_dir, symbolic_args = runs[0]
    cores, allocs, ishapes = _kernel_summary(code_dir)
    assert cores == n
    assert len(allocs) == num_tensors
    for alloc in allocs:
        assert alloc.get("nonUnifiedAllocInHBM_") is True
        assert len(set(alloc["startAddressCoreCorelet_"]["data_"].values())) == n
    assert ishapes == [[str(num_tensors * n)]]
    # Core c of tensor t uses the bundle input at slot t * n + c. This holds
    # only because (a) this graph has no dimension symbol IDs (they would be
    # emitted before the address IDs by _extract_symbol_ids in bundle.py) and
    # (b) the allocs are in scheduleTree tensor order with data_ keys in core
    # order. If either assumption breaks, this false-fails (never false-passes).
    with open(os.path.join(code_dir, "bundle.mlir")) as f:
        bundle = f.read()
    symbol_ids = json.loads(bundle.split('"symbol_ids"=')[1].split("}")[0])
    for t, alloc in enumerate(allocs):
        starts = alloc["startAddressCoreCorelet_"]["data_"]
        for c in range(n):
            assert symbol_ids.index(int(starts[f"[{c}, 0, 0]"])) == t * n + c
    # The launch payload resolves input (tensor t, core c) to shard c of t.
    assert [(a.kind, a.tensor_id, a.dim_index) for a in symbolic_args] == [
        (SymbolicArgKind.kShardAddress, t, c)
        for t in range(num_tensors)
        for c in range(n)
    ]


def _sharded(cpu: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    return torch.spyre.distribute_tensor(cpu, Shard(dim), num_shards=n)


@_requires_emulation
@pytest.mark.parametrize(
    "shape_per_shard, dim, dtype",
    [
        ((4, 64), 0, torch.float16),
        ((8, 32), 0, torch.float16),
        ((32, 128), 1, torch.float16),
        ((16, 32), 1, torch.float32),
        ((64,), 0, torch.float16),
    ],
)
def test_compiled_add_runs_one_shard_per_core(launched, shape_per_shard, dim, dtype):
    n = _num_shards()
    shape = list(shape_per_shard)
    shape[dim] *= n
    torch.manual_seed(0)
    a_cpu = torch.randn(*shape, dtype=dtype)
    b_cpu = torch.randn(*shape, dtype=dtype)
    a, b = _sharded(a_cpu, dim, n), _sharded(b_cpu, dim, n)

    c = torch.compile(lambda x, y: x + y, dynamic=False)(a, b)
    _assert_one_non_unified_kernel(launched, n)

    assert torch.spyre.is_sharded(c)
    info = torch.spyre.sharding_info(c)
    assert info.placement == Shard(dim) and info.num_shards == n
    expected = (a_cpu.to("spyre") + b_cpu.to("spyre")).cpu()
    assert torch.equal(c.cpu(), expected)


@_requires_emulation
def test_chained_compiled_adds(launched):
    """A compiled sharded output feeds a second compiled call.

    The output of a non-unified kernel is allocated by the wrapper
    (_spyre_empty_sharded), not by distribute_tensor; if its storage differed in
    anything the planner or launch binding keys on, the second call is where
    it would show.
    """
    n = _num_shards()
    torch.manual_seed(2)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a, b = _sharded(a_cpu, 0, n), _sharded(b_cpu, 0, n)
    add = torch.compile(lambda x, y: x + y, dynamic=False)

    c = add(a, b)
    _assert_one_non_unified_kernel(launched, n)
    launched.clear()
    d = add(c, b)
    _assert_one_non_unified_kernel(launched, n)
    assert torch.spyre.is_sharded(d)
    expected = (a_cpu.to("spyre") + b_cpu.to("spyre")).cpu()
    expected_d = (a_cpu.to("spyre") + b_cpu.to("spyre") + b_cpu.to("spyre")).cpu()
    assert torch.equal(c.cpu(), expected)
    assert torch.equal(d.cpu(), expected_d)


@_requires_emulation
@pytest.mark.parametrize("dim", [0, 1])
def test_each_core_writes_only_its_own_shard(launched, dim):
    n = _num_shards()
    # Every shard of a and of b gets distinct values (integers < 512, so the
    # sums are exact in the device fp16 format): a core reading or writing
    # another shard's chunk, of any operand, puts values in the wrong place.
    # Two calls with different b rule out stale chunk contents matching.
    shape = (4 * n, 64) if dim == 0 else (2, 64 * n)
    pos = torch.arange(shape[0] * shape[1]).view(shape)
    shard_id = pos // (pos.numel() // n) if dim == 0 else pos % shape[1] // 64
    within = pos % 64
    a_cpu = (shard_id * 16 + within % 16).to(torch.float16)
    a = _sharded(a_cpu, dim, n)
    add = torch.compile(lambda x, y: x + y, dynamic=False)

    for bias in (0, 1):
        launched.clear()
        b_cpu = (2 * shard_id + bias).to(torch.float16)
        b = _sharded(b_cpu, dim, n)
        c = add(a, b)
        _assert_one_non_unified_kernel(launched, n)
        assert torch.equal(c.cpu(), a_cpu + b_cpu)
        assert torch.equal(a.cpu(), a_cpu)
        assert torch.equal(b.cpu(), b_cpu)


@_requires_emulation
def test_unsharded_and_sharded_inputs_never_share_a_kernel(launched):
    """Sharding is part of the guards and the FX cache key (L11).

    The unsharded inputs are given the sharded tensors' (shard-local) device
    layout, so shape, dtype, layout and offset all match and only the
    sharding tells the two calls apart. Without the sharding key the second
    call reuses the unified kernel on sharded storage.
    """
    n = _num_shards()
    torch.manual_seed(1)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a, b = _sharded(a_cpu, 0, n), _sharded(b_cpu, 0, n)
    layout = a.device_tensor_layout()
    plain_a = a_cpu.to("spyre", device_layout=layout)
    plain_b = b_cpu.to("spyre", device_layout=layout)
    assert plain_a.device_tensor_layout() == layout
    add = torch.compile(lambda x, y: x + y, dynamic=False)

    expected = (a_cpu.to("spyre") + b_cpu.to("spyre")).cpu()

    launched.clear()
    add(plain_a, plain_b)  # unified kernel; its values are not checked
    assert len(launched) == 1
    unified_dir = launched[0][0]
    assert _kernel_summary(unified_dir)[1][0].get("nonUnifiedAllocInHBM_") is None

    launched.clear()
    c = add(a, b)
    _assert_one_non_unified_kernel(launched, n)
    assert torch.equal(c.cpu(), expected)

    # And back again: after the non-unified compile, the unsharded inputs
    # must still get the unified kernel. (Values are not checked: plain_a
    # deliberately carries the shard-local layout, which does not describe
    # its (global) shape.)
    launched.clear()
    add(plain_a, plain_b)
    assert len(launched) == 1
    assert _kernel_summary(launched[0][0])[1][0].get("nonUnifiedAllocInHBM_") is None


@_requires_emulation
@pytest.mark.parametrize(
    "shape_per_shard, dim",
    [
        # Rows wider than one stick: in-tile strides would follow the global
        # shape and leave the second stick of every row unwritten.
        # TODO: positive once DeepTools uses per-core layout sizes.
        ((4, 128), 0),
        # Last-dim shards that are not whole sticks (32 of 64 fp16 elements).
        # A real limit: a stick cannot be split between chunks.
        ((32, 32), 1),
        # 3-D Shard(last): a non-shard dim sits outside the stick dim, so its
        # global stride would address past the end of the chunk.
        # TODO: positive once DeepTools uses per-core layout sizes.
        ((4, 8, 64), 2),
    ],
)
@expects_raise
def test_unsupported_shardings_fail_at_compile(shape_per_shard, dim):
    n = _num_shards()
    shape = list(shape_per_shard)
    shape[dim] *= n
    torch.manual_seed(0)
    a = _sharded(torch.randn(*shape, dtype=torch.float16), dim, n)

    with pytest.raises(RuntimeError, match=_STRIDE_REJECTION):
        torch.compile(lambda x, y: x + y, dynamic=False)(a, a)


@_requires_emulation
@pytest.mark.parametrize(
    "fn, other, match",
    [
        # TODO: mixed sharded/unsharded and mixed placements need
        # redistribution; positive once that exists.
        (lambda x, y: x + y, "unsharded", "every operand sharded with the same"),
        (lambda x, y: x + y, "shard1", "every operand sharded with the same"),
        # Non-unified reductions are not implemented.
        (lambda x, y: x.sum(0) + y.sum(0), "shard0", "pointwise ops only"),
        # TODO: in-place writes into sharded storage.
        (lambda x, y: x.add_(y), "shard0", "in-place writes into sharded"),
        # TODO: sharded intermediates.
        (
            lambda x, y: (x + y) * y,
            "shard0",
            "every sharded result to be a graph output",
        ),
    ],
)
@expects_raise
def test_unsupported_ops_fail_at_compile(fn, other, match):
    n = _num_shards()
    torch.manual_seed(0)
    src = torch.randn(8 * n, 64, dtype=torch.float16)
    a = _sharded(src, 0, n)
    if other == "unsharded":
        b = src.to("spyre")
    elif other == "shard1":
        # 8n x 64n shards evenly on either dim.
        a = _sharded(torch.randn(8 * n, 64 * n, dtype=torch.float16), 0, n)
        b = _sharded(torch.randn(8 * n, 64 * n, dtype=torch.float16), 1, n)
    else:
        b = _sharded(src.clone(), 0, n)

    with pytest.raises(RuntimeError, match=match):
        torch.compile(fn, dynamic=False)(a, b)


@_requires_emulation
@expects_raise
@pytest.mark.parametrize(
    "view",
    [
        lambda t: t.view(-1),
        lambda t: t[: t.shape[0] // 2],
        lambda t: t.t(),
        # Rank-increasing: the guard must reject on rank alone, before any
        # per-dimension comparison (which would read past the base's rank).
        lambda t: t.unsqueeze(0),
    ],
)
def test_views_of_sharded_inputs_fail(view):
    # A view keeps the base's shard layout, which no longer matches the
    # view's shape; chunk addressing would read permuted data. The native
    # view constructors reject it before the tensor can reach a kernel.
    # TODO: positive once views of sharded tensors have a shard mapping.
    n = _num_shards()
    torch.manual_seed(0)
    for cpu in (
        torch.randn(2 * n, 128, dtype=torch.float16),
        torch.randn(2 * n, 128, dtype=torch.float16),
    ):
        x = _sharded(cpu, 0, n)
        with pytest.raises(
            RuntimeError, match="views of sharded Spyre tensors are not supported"
        ):
            view(x)


@_requires_emulation
def test_compiled_add_over_detached_views(launched):
    """Detached views of sharded inputs reach the compiled path unchanged.

    A detach is an identity view (same sizes, strides and storage offset),
    so it keeps the element-to-shard mapping and must stay accepted where
    non-identity views are rejected; the compiled kernel still binds core c
    to shard c and produces the right values. (Other identity views such as
    an explicit as_strided are still held back by the compile-time
    base-tensor check in sharded_hbm.py, which cannot see that the view is
    an identity one.)
    """
    n = _num_shards()
    torch.manual_seed(0)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a, b = _sharded(a_cpu, 0, n), _sharded(b_cpu, 0, n)

    c = torch.compile(lambda x, y: x + y, dynamic=False)(a.detach(), b.detach())
    _assert_one_non_unified_kernel(launched, n)

    assert torch.spyre.is_sharded(c)
    expected = (a_cpu.to("spyre") + b_cpu.to("spyre")).cpu()
    assert torch.equal(c.cpu(), expected)


@_requires_emulation
@expects_raise
def test_contiguous_prefix_view_of_sharded_fails():
    # The hole the copy-path guards could not see: a contiguous view from
    # storage offset zero passes every is_contiguous()/storage_offset()
    # check, yet maps view elements onto the wrong shards (element N comes
    # from the start of a later shard instead of the shards that back the
    # first N elements). It must be rejected at construction, not copied.
    n = _num_shards()
    x = _sharded(torch.arange(4 * n, dtype=torch.float16), 0, n)

    with pytest.raises(
        RuntimeError, match="views of sharded Spyre tensors are not supported"
    ):
        x.narrow(0, 0, 2 * n)


@_requires_emulation
def test_identity_views_of_sharded_keep_round_trip():
    # Identity views (detach, as_strided of the base tensor) reproduce the
    # base tensor exactly, so the element-to-shard mapping still holds and
    # the whole-tensor copy path must keep working through them.
    n = _num_shards()
    torch.manual_seed(0)
    cpu = torch.randn(2 * n, 128, dtype=torch.float16)
    x = _sharded(cpu, 0, n)
    d = x.detach()
    a = torch.as_strided(x, x.size(), x.stride(), 0)

    assert torch.spyre.is_sharded(d) and torch.spyre.is_sharded(a)
    # Reference through the same device round-trip the suite uses elsewhere,
    # so the emulation's fp16 last-bit DMA difference does not mask a real
    # mismatch; the three round-trips must be identical to each other.
    expected = cpu.to("spyre").to("cpu")
    assert torch.equal(x.to("cpu"), expected)
    assert torch.equal(d.to("cpu"), expected)
    assert torch.equal(a.to("cpu"), expected)


@_requires_emulation
@expects_raise
def test_mixed_dtypes_fail_at_compile():
    n = _num_shards()
    torch.manual_seed(0)
    x = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="operands and result of one dtype"):
        torch.compile(lambda a, b: (a + b).float(), dynamic=False)(x, x)


@_requires_emulation
@expects_raise
def test_dynamic_shapes_fail_at_compile():
    n = _num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="need static shapes"):
        torch.compile(lambda x, y: x + y, dynamic=True)(a, a)


@_requires_emulation
@expects_raise
def test_fallback_consumer_fails_at_compile():
    """An op that lowers to a fallback kernel is rejected at compile, not at
    launch with an error that names neither the op nor the sharding."""
    n = _num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)
    b = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="pointwise ops on sharded operands"):
        torch.compile(lambda x, y: torch.cumsum(x, dim=1) + 0 * y, dynamic=False)(a, b)


@_requires_emulation
@expects_raise
def test_bundle_symbolic_args_disabled_fails_at_compile(monkeypatch):
    """Every core's start address is a bundle input_arg, so a sharded kernel
    cannot be compiled onto the launch-time-bound path."""
    from torch_spyre._inductor import config as spyre_config

    monkeypatch.setattr(spyre_config, "bundle_symbolic_args", False)
    n = _num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="BUNDLE_SYMBOLIC_ARGS=1"):
        torch.compile(lambda x, y: x + y, dynamic=False)(a, a)


@_requires_emulation
@expects_raise
def test_dtype_casting_copy_fails_closed():
    """Dtype-converting copies of sharded tensors are rejected, not truncated.

    The installed Flex build lacks the D2H host-capacity argument (the LOCAL
    DEMO SHIM in spyre_stream.cpp), so a converting copy cannot bound its
    host output. TODO: positive once Flex is rebased onto aaf176a7.
    """
    n = _num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="without a dtype change"):
        a.to("cpu", torch.float32)


@_requires_emulation
@expects_raise
def test_resize_and_set_of_sharded_tensor_fail_closed():
    """resize_/set_ must not silently corrupt or un-shard sharded storage.

    An in-place reinterpret would leave the sharding record describing the
    old shape (wrong rows gathered by .cpu()); a reallocating resize would
    silently convert the tensor to unified storage.
    """
    n = _num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="resize_ is not supported"):
        a.resize_(4 * n, 64)
    with pytest.raises(RuntimeError, match="set_ is not supported"):
        a.set_(a.storage(), 0, [4 * n, 64], [64, 1])
    # The no-op resize is still fine and changes nothing.
    a.resize_(a.shape)
    assert torch.spyre.is_sharded(a)


@_requires_emulation
def test_inputs_usable_after_compile_rejection():
    """A rejected compile leaves the sharded inputs intact and usable."""
    n = _num_shards()
    torch.manual_seed(0)
    src = torch.randn(8 * n, 64, dtype=torch.float16)
    a = _sharded(src, 0, n)
    b = _sharded(src.clone(), 0, n)

    with pytest.raises(RuntimeError, match="pointwise ops only"):
        torch.compile(lambda x, y: x.sum(0) + y.sum(0), dynamic=False)(a, b)

    # Same tensors still work in a supported compile afterwards.
    c = torch.compile(lambda x, y: x + y, dynamic=False)(a, b)
    roundtrip = src.to("spyre").cpu()  # fp16 goes through the device format
    expected = roundtrip + roundtrip
    assert torch.equal(c.cpu(), expected)
    assert torch.equal(a.cpu(), roundtrip)
