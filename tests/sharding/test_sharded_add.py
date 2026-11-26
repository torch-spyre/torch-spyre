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
admits sharded allocations (``FLEX_DEVICE=EMUL1p5``), fixed at process
start. ``test_device_suite_in_emulation_shape`` runs this file in a
subprocess for each shape, so a plain ``pytest`` run (and CI) covers all of
them; the per-shape tests skip in a process without that environment. It
also runs the file once more per shape with a small DMA slice size
(``FLEX_DATA_TRANSFER_BUFFER_SIZE``), so every shard copy is split into
several DMA slices; the multi-slice round-trip tests run only there.

Results are compared with the same add on unsharded device tensors. A CPU
comparison is not exact: fp16 goes through the device fp16 format (L9), and
the device fp32 add differs from the CPU add by up to 1 ulp (measured on
unsharded tensors too).
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch
from device_env import num_shards
from device_suite import (
    SMALL_DMA_SLICE_BYTES,
    make_device_suite_test,
    requires_emulation,
)
from torch.distributed.tensor import Shard

from torch_spyre._inductor import config as spyre_config

from torch_spyre.execution import kernel_runner
from torch_spyre.ops.fallbacks import register_fallback

_STRIDE_REJECTION = "whole device dims of whole sticks"


@torch.library.custom_op("spyre::test_sharding_host_copy", mutates_args=())
def _host_copy(x: torch.Tensor) -> torch.Tensor:
    """A copy that always runs on the host, as a fallback kernel in a graph.

    The fallback-consumer test needs an op that falls back to the host. A real
    op on the fallback list leaves it once it gains a Spyre implementation, so
    the test owns one instead.
    """
    return x.clone()


@_host_copy.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@register_fallback(["spyre::test_sharding_host_copy"])
def _host_copy_fallback(x, **kwargs):
    return x.clone()


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


def _assert_mixed_kernel(runs, n: int, sharded_inputs: int, plain_inputs: int) -> None:
    """Exactly one kernel ran: its sharded inputs and its output are
    non-unified, its other inputs are ordinary tensors.

    A non-unified tensor gets one start address per core, bound at launch to
    that core's shard. An unsharded operand keeps ordinary addressing and is
    never launched through a shard-address symbol. Argument order is not
    asserted: the output is last, the inputs are in the op's own order.
    """
    from torch_spyre._C import SymbolicArgKind

    assert len(runs) == 1, [d for d, _ in runs]
    code_dir, symbolic_args = runs[0]
    cores, allocs, _ = _kernel_summary(code_dir)
    assert cores == n
    assert len(allocs) == sharded_inputs + plain_inputs + 1
    non_unified = [a for a in allocs if a.get("nonUnifiedAllocInHBM_") is True]
    assert len(non_unified) == sharded_inputs + 1
    assert allocs[-1].get("nonUnifiedAllocInHBM_") is True
    for alloc in non_unified:
        assert len(set(alloc["startAddressCoreCorelet_"]["data_"].values())) == n
    shard_args = [a for a in symbolic_args if a.kind == SymbolicArgKind.kShardAddress]
    assert len(shard_args) == n * (sharded_inputs + 1)
    # Every sharded tensor is launched once per core, core c on shard c.
    assert {(a.tensor_id, a.dim_index) for a in shard_args} == {
        (t, c) for t in {a.tensor_id for a in shard_args} for c in range(n)
    }


def _sharded(cpu: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    return torch.spyre.to_sharded(cpu, Shard(dim), num_shards=n)


@requires_emulation
@pytest.mark.parametrize(
    "shape_per_shard, dim, dtype",
    [
        ((4, 64), 0, torch.float16),
        ((8, 32), 0, torch.float16),
        ((32, 128), 1, torch.float16),
        ((16, 32), 1, torch.float32),
        ((64,), 0, torch.float16),
        # Shard-local strides differ from the global ones (the shard dim is
        # not the innermost non-stick device dim): admitted since the codegen
        # guard checks the split against the shard dim instead of unified-
        # model stride distances (issue #5827).
        ((4, 128), 0, torch.float16),
        ((4, 8, 64), 2, torch.float16),
    ],
)
def test_compiled_add_runs_one_shard_per_core(launched, shape_per_shard, dim, dtype):
    n = num_shards()
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


@requires_emulation
@pytest.mark.parametrize(
    "op,num_args",
    [(lambda x, y: x * y, 2), (lambda x: torch.abs(x), 1)],
)
def test_compiled_pointwise_ops_run_one_shard_per_core(launched, op, num_args):
    """The admission check is on the op kind, not on add specifically.

    One binary and one unary representative; the doc promises any
    same-shape pointwise op.
    """

    n = num_shards()
    torch.manual_seed(0)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a, b = _sharded(a_cpu, 0, n), _sharded(b_cpu, 0, n)

    args = (a, b) if num_args == 2 else (a,)
    c = torch.compile(op, dynamic=False)(*args)
    _assert_one_non_unified_kernel(launched, n, num_tensors=num_args + 1)

    assert torch.spyre.is_sharded(c)
    ref_args = (
        (a_cpu.to("spyre"), b_cpu.to("spyre"))
        if num_args == 2
        else (a_cpu.to("spyre"),)
    )
    expected = torch.compile(op, dynamic=False)(*ref_args).cpu()
    assert torch.equal(c.cpu(), expected)


@requires_emulation
def test_eager_add_on_sharded_inputs():
    """The eager path also computes sharded add correctly (no torch.compile)."""

    n = num_shards()
    torch.manual_seed(3)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a, b = _sharded(a_cpu, 0, n), _sharded(b_cpu, 0, n)

    c = a + b

    assert torch.spyre.is_sharded(c)
    expected = (a_cpu.to("spyre") + b_cpu.to("spyre")).cpu()
    assert torch.equal(c.cpu(), expected)


@requires_emulation
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x + 1.0,
        lambda x: 2.0 * x,
        lambda x: x - 0.5,
    ],
    ids=["add", "mul_left", "sub"],
)
def test_sharded_op_with_scalar_matches_unsharded(launched, fn):
    """A Python scalar is a replicated constant: every core reads the same
    value, only the tensor is sharded."""
    n = num_shards()
    torch.manual_seed(4)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a = _sharded(a_cpu, 0, n)

    c = torch.compile(fn, dynamic=False)(a)

    assert torch.spyre.is_sharded(c)
    info = torch.spyre.sharding_info(c)
    assert info.placement == Shard(0) and info.num_shards == n
    # The scalar is a materialized constant buffer: one ordinary input.
    _assert_mixed_kernel(launched, n, sharded_inputs=1, plain_inputs=1)
    expected = torch.compile(fn, dynamic=False)(a_cpu.to("spyre")).cpu()
    assert torch.equal(c.cpu(), expected)


# name -> (shard dim, activation shape, bias shape), as functions of the shard
# count. A bias that does not span the shard dim is read whole by every core; one
# that does (a hidden-dim bias on a last-dim shard) is read by core c at its own
# slice, through ordinary unified addressing.
_BIAS_CASES = {
    "shard0_bias_row": (0, lambda n: (4 * n, 64), lambda n: (64,)),
    "shard0_bias_1xN": (0, lambda n: (4 * n, 64), lambda n: (1, 64)),
    "shard1_bias_vec": (1, lambda n: (2, 64 * n), lambda n: (64 * n,)),
    "shard1_bias_1xN": (1, lambda n: (2, 64 * n), lambda n: (1, 64 * n)),
    # A column bias: its stride along the split dim differs from the output's.
    "shard0_bias_column": (0, lambda n: (4 * n, 64), lambda n: (4 * n, 1)),
    "shard1_bias_column": (1, lambda n: (2, 64 * n), lambda n: (2, 1)),
}


@requires_emulation
@pytest.mark.parametrize("case", list(_BIAS_CASES))
def test_sharded_plus_unified_bias_matches_unsharded(launched, case):
    """A plain (unified) bias is read by every core like a replica; the result
    is sharded as the activation is."""
    dim, shape_of, bias_shape_of = _BIAS_CASES[case]
    n = num_shards()
    torch.manual_seed(5)
    a_cpu = torch.randn(*shape_of(n), dtype=torch.float16)
    bias_cpu = torch.randn(*bias_shape_of(n), dtype=torch.float16)
    a = _sharded(a_cpu, dim, n)
    bias = bias_cpu.to("spyre")
    assert not torch.spyre.is_sharded(bias)
    add = torch.compile(lambda x, b: x + b, dynamic=False)

    c = add(a, bias)

    assert torch.spyre.is_sharded(c)
    assert torch.spyre.sharding_info(c).placement == Shard(dim)
    _assert_mixed_kernel(launched, n, sharded_inputs=1, plain_inputs=1)
    expected = add(a_cpu.to("spyre"), bias).cpu()
    assert torch.equal(c.cpu(), expected)
    # The unified operand is only read.
    assert torch.equal(bias.cpu(), bias_cpu.to("spyre").cpu())


@requires_emulation
def test_sharded_plus_zero_dim_tensor_matches_unsharded(launched):
    """A 0-d device tensor is a graph input, so it is a replica like a bias."""
    n = num_shards()
    torch.manual_seed(8)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    s = torch.tensor(1.5, dtype=torch.float16).to("spyre")
    a = _sharded(a_cpu, 0, n)
    add = torch.compile(lambda x, y: x + y, dynamic=False)

    c = add(a, s)

    assert torch.spyre.is_sharded(c)
    _assert_mixed_kernel(launched, n, sharded_inputs=1, plain_inputs=1)
    assert torch.equal(c.cpu(), add(a_cpu.to("spyre"), s).cpu())


@requires_emulation
def test_unified_operand_address_is_not_part_of_the_compiled_kernel(launched):
    """One compiled kernel serves every unified operand allocation: its
    address is bound at launch, never baked into the artifact or cache key."""
    n = num_shards()
    torch.manual_seed(9)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a = _sharded(a_cpu, 0, n)
    add = torch.compile(lambda x, b: x + b, dynamic=False)
    biases = [torch.randn(64, dtype=torch.float16) for _ in range(2)]
    on_device = [b.to("spyre") for b in biases]

    results = [add(a, b).cpu() for b in on_device]

    assert len({code_dir for code_dir, _ in launched}) == 1
    for b, got in zip(on_device, results):
        assert torch.equal(got, add(a_cpu.to("spyre"), b).cpu())


@requires_emulation
def test_sharded_and_unified_second_operand_do_not_share_a_kernel(launched):
    """The sharding signature is per operand: a sharded and a unified second
    operand of the same shape must compile separately."""
    n = num_shards()
    torch.manual_seed(10)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a = _sharded(a_cpu, 0, n)
    add = torch.compile(lambda x, y: x + y, dynamic=False)

    both_sharded = add(a, _sharded(b_cpu, 0, n))
    one_sharded = add(a, b_cpu.to("spyre"))

    assert len({code_dir for code_dir, _ in launched}) == 2
    assert torch.equal(both_sharded.cpu(), one_sharded.cpu())


@requires_emulation
def test_unsharded_operand_first_matches_unsharded(launched):
    """Operand order does not matter: the sharded operand need not come first."""
    n = num_shards()
    torch.manual_seed(6)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    bias_cpu = torch.randn(64, dtype=torch.float16)
    a, bias = _sharded(a_cpu, 0, n), bias_cpu.to("spyre")
    sub = torch.compile(lambda b, x: b - x, dynamic=False)

    c = sub(bias, a)

    assert torch.spyre.is_sharded(c)
    _assert_mixed_kernel(launched, n, sharded_inputs=1, plain_inputs=1)
    assert torch.equal(c.cpu(), sub(bias, a_cpu.to("spyre")).cpu())


@requires_emulation
def test_sharded_plus_full_size_unified_tensor_matches_unsharded(launched):
    """A unified tensor of the op's own shape is a replica too: core c reads
    the rows of it that line up with its shard."""
    n = num_shards()
    torch.manual_seed(7)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a, b = _sharded(a_cpu, 0, n), b_cpu.to("spyre")
    add = torch.compile(lambda x, y: x + y, dynamic=False)

    c = add(a, b)

    assert torch.spyre.is_sharded(c)
    _assert_mixed_kernel(launched, n, sharded_inputs=1, plain_inputs=1)
    assert torch.equal(c.cpu(), add(a_cpu.to("spyre"), b).cpu())


@requires_emulation
def test_chained_compiled_adds(launched):
    """A compiled sharded output feeds a second compiled call.

    The output of a non-unified kernel is allocated by the wrapper
    (_spyre_empty_sharded), not by to_sharded; if its storage differed in
    anything the planner or launch binding keys on, the second call is where
    it would show.
    """
    n = num_shards()
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


@requires_emulation
@pytest.mark.parametrize("dim", [0, 1])
def test_each_core_writes_only_its_own_shard(launched, dim):
    n = num_shards()
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


@requires_emulation
def test_unsharded_and_sharded_inputs_never_share_a_kernel(launched):
    """Sharding is part of the guards and the FX cache key (L11).

    The unsharded inputs are given the sharded tensors' (shard-local) device
    layout, so shape, dtype, layout and offset all match and only the
    sharding tells the two calls apart. Without the sharding key the second
    call reuses the unified kernel on sharded storage.
    """
    n = num_shards()
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


@requires_emulation
@pytest.mark.parametrize(
    "shape_per_shard, dim",
    [
        # Last-dim shards that are not whole sticks (32 of 64 fp16 elements).
        # A real limit: a stick cannot be split between chunks.
        ((32, 32), 1),
    ],
)
@expects_raise
def test_unsupported_shardings_fail_at_compile(shape_per_shard, dim):
    n = num_shards()
    shape = list(shape_per_shard)
    shape[dim] *= n
    torch.manual_seed(0)
    a = _sharded(torch.randn(*shape, dtype=torch.float16), dim, n)

    with pytest.raises(RuntimeError, match=_STRIDE_REJECTION):
        torch.compile(lambda x, y: x + y, dynamic=False)(a, a)


# ---------------------------------------------------------------------------
# verify_non_unified_split: the committed work split must be the shard dim.
#
# Pure-python unit tests on the codegen guard (no device needed). The guard
# checks the committed division against each sharded arg's recorded shard
# dim (TensorArg.non_unified_shard_dim); the earlier stride-distance
# arithmetic only held when the shard dim was the innermost non-stick
# device dim, so it was replaced by the dim-identity check these tests
# pin. The e2e behavior (a re-divided op is rejected at compile) is covered
# by the tests above through the full pipeline.
# ---------------------------------------------------------------------------


def _split_spec(num_cores=8):
    """An SDSCSpec whose work division splits ``mb`` 8 ways, core c -> c."""
    import sympy

    from torch_spyre._inductor.codegen.superdsc import SDSCSpec
    from torch_spyre._C import DataFormats

    core_id = sympy.Symbol("core_id")
    return SDSCSpec(
        opfunc="add",
        execution_unit="sfp",
        data_format=DataFormats.SEN169_FP16,
        num_inputs=2,
        iteration_space={sympy.Symbol("mb"): 32, sympy.Symbol("out"): 128},
        num_cores=num_cores,
        work_slices={sympy.Symbol("mb"): num_cores, sympy.Symbol("out"): 1},
        core_id_to_work_slice={
            sympy.Symbol("mb"): sympy.Mod(core_id, num_cores),
            sympy.Symbol("out"): sympy.Integer(0),
        },
        padding={},
        layouts={},
        args=[],
        constants={},
        coordinate_masking={},
    )


def _shard_arg(shard_dim, arg_index=0, indexed=("mb",)):
    import sympy

    from torch_spyre._inductor.codegen.superdsc import SDSCArgs

    # ``scales`` says which dims the arg indexes in the emitted SDSC; a
    # tiling pass that split the shard dim into sub-dims would leave the
    # split dim out and the guard must fail closed (defence in depth).
    return SDSCArgs(
        layout="OUTPUT",
        dim_order=[],
        data_format=None,
        scales={sympy.Symbol(d): 1 for d in indexed},
        strides={},
        offsets={},
        max_dim_sizes={},
        allocation={},
        start_address=0,
        backGap={},
        arg_index=arg_index,
        non_unified_shard_bytes=1024,
        non_unified_shard_dim=shard_dim,
    )


def test_verify_non_unified_split_passes_on_the_shard_dim():
    import sympy

    from torch_spyre._inductor.sharded_hbm import verify_non_unified_split

    verify_non_unified_split(_split_spec(), [_shard_arg(sympy.Symbol("mb"))])


def test_verify_non_unified_split_rejects_a_split_on_another_dim():
    import sympy

    from torch_spyre._inductor.sharded_hbm import verify_non_unified_split
    from torch_spyre._inductor.errors import Unsupported

    # A later pass re-divided the op on ``out`` while the tensors are
    # sharded on ``mb``: each core's work slice would straddle shards.
    spec = _split_spec()
    spec.work_slices = {sympy.Symbol("mb"): 1, sympy.Symbol("out"): 8}
    spec.core_id_to_work_slice = {
        sympy.Symbol("mb"): sympy.Integer(0),
        sympy.Symbol("out"): sympy.Mod(sympy.Symbol("core_id"), 8),
    }
    with pytest.raises(Unsupported, match="sharded on"):
        verify_non_unified_split(
            spec, [_shard_arg(sympy.Symbol("mb"), indexed=("mb", "out"))]
        )


def test_verify_non_unified_split_rejects_an_arg_without_a_shard_dim():
    from torch_spyre._inductor.sharded_hbm import verify_non_unified_split
    from torch_spyre._inductor.errors import Unsupported

    # A sharded arg whose producer was not admitted by the sharded
    # planner carries no shard dim; the guard must fail closed.
    with pytest.raises(Unsupported, match="records no shard dim"):
        verify_non_unified_split(_split_spec(), [_shard_arg(None)])


def test_verify_non_unified_split_rejects_an_arg_that_does_not_index_the_split():
    from torch_spyre._inductor.sharded_hbm import verify_non_unified_split
    from torch_spyre._inductor.errors import Unsupported

    # Defence in depth: the symbol comparison alone would pass (the arg
    # records the shard dim), but the arg does not index the split dim in
    # the emitted SDSC -- e.g. a later tiling pass split the shard dim
    # into sub-dims. The guard must fail closed.
    import sympy

    with pytest.raises(Unsupported, match="does not index"):
        verify_non_unified_split(
            _split_spec(), [_shard_arg(sympy.Symbol("mb"), indexed=())]
        )


def test_non_unified_shard_dim_rejects_an_unmatched_shard_dim(monkeypatch):
    """The derivation fails closed when no iteration symbol matches the
    shard dim's stride and extent in the iteration space."""
    import sympy
    from types import SimpleNamespace

    from torch_spyre._inductor import spyre_kernel
    from torch_spyre._inductor.errors import Unsupported
    from torch_spyre._inductor.sharded_hbm import ShardedPlacement

    monkeypatch.setattr(
        spyre_kernel,
        "placement_of",
        lambda name: ShardedPlacement(
            shard_dim=1, num_shards=8, shard_device_bytes=128
        ),
    )
    # layout [4, 64]: shard dim 1 has stride 1 and extent 64, but no
    # iteration symbol has that coefficient and extent.
    with pytest.raises(Unsupported, match="cannot identify the shard dim"):
        spyre_kernel._non_unified_shard_dim(
            "arg0",
            # Any layout object carrying the tensor's logical size.
            SimpleNamespace(size=[4, 64]),
            {sympy.Symbol("mb"): 4},
            sympy.Symbol("mb") * 64 + sympy.Symbol("out"),
        )


def test_non_unified_shard_dim_rejects_an_ambiguous_shard_dim(monkeypatch):
    """Two symbols matching the shard dim's stride and extent means the
    identification is not unique; the derivation must fail closed."""
    import sympy
    from types import SimpleNamespace

    from torch_spyre._inductor import spyre_kernel
    from torch_spyre._inductor.errors import Unsupported
    from torch_spyre._inductor.sharded_hbm import ShardedPlacement

    monkeypatch.setattr(
        spyre_kernel,
        "placement_of",
        lambda name: ShardedPlacement(
            shard_dim=1, num_shards=8, shard_device_bytes=128
        ),
    )
    # Both s0 and s1 have coefficient 1 and extent 64 in the iteration
    # space, so the shard dim (stride 1, extent 64) matches twice.
    with pytest.raises(Unsupported, match="cannot identify the shard dim"):
        spyre_kernel._non_unified_shard_dim(
            "arg0",
            SimpleNamespace(size=[4, 64]),
            {sympy.Symbol("s0"): 64, sympy.Symbol("s1"): 64},
            sympy.Symbol("s0") + sympy.Symbol("s1"),
        )


@requires_emulation
@pytest.mark.parametrize(
    "fn, other, match",
    [
        # TODO: mixed placements need redistribution (#5820); positive once
        # that exists. The error names both placements.
        (
            lambda x, y: x + y,
            "shard1",
            r"sharded differently \(Shard\(0\).*and Shard\(1\)",
        ),
        # TODO: a full reduction reduces more than the shard dim, which needs
        # more than one split reduced dim. Reductions that are supported are
        # covered in test_sharded_reduction.py.
        (
            lambda x, y: x.sum() + y.sum(),
            "shard0",
            "must reduce the shard dim only",
        ),
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
    n = num_shards()
    torch.manual_seed(0)
    src = torch.randn(8 * n, 64, dtype=torch.float16)
    a = _sharded(src, 0, n)
    if other == "shard1":
        # 8n x 64n shards evenly on either dim.
        a = _sharded(torch.randn(8 * n, 64 * n, dtype=torch.float16), 0, n)
        b = _sharded(torch.randn(8 * n, 64 * n, dtype=torch.float16), 1, n)
    else:
        b = _sharded(src.clone(), 0, n)

    with pytest.raises(RuntimeError, match=match):
        torch.compile(fn, dynamic=False)(a, b)


@requires_emulation
@expects_raise
@pytest.mark.parametrize(
    "view",
    [
        lambda t: t.view(-1),
        # A strided slice: not a block of the base. (Unit-step slices are
        # box views, accepted at construction; see the box-view tests below.)
        lambda t: t[::2],
        lambda t: t.t(),
        # Rank-increasing: the guard must reject on rank alone, before any
        # per-dimension comparison (which would read past the base's rank).
        lambda t: t.unsqueeze(0),
    ],
)
def test_views_of_sharded_inputs_fail(view):
    # A view keeps the base's shard layout, which no longer matches the
    # view's shape; chunk addressing would read permuted data. The native
    # view constructors reject every view that is not a block of the base
    # before it can reach a kernel.
    # TODO: positive once views of sharded tensors have a shard mapping.
    n = num_shards()
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


@requires_emulation
@expects_raise
def test_as_strided_in_place_of_sharded_fails():
    # as_strided_ changes metadata in place, bypassing the view
    # constructors; an unguarded call leaves the shard table describing the
    # old shape and the next copy reads the wrong elements.
    n = num_shards()
    x = _sharded(torch.arange(4 * n, dtype=torch.float16), 0, n)

    with pytest.raises(
        RuntimeError, match="views of sharded Spyre tensors are not supported"
    ):
        x.as_strided_((2 * n,), (1,), 0)


@requires_emulation
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
    n = num_shards()
    torch.manual_seed(0)
    a_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    b_cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a, b = _sharded(a_cpu, 0, n), _sharded(b_cpu, 0, n)

    c = torch.compile(lambda x, y: x + y, dynamic=False)(a.detach(), b.detach())
    _assert_one_non_unified_kernel(launched, n)

    assert torch.spyre.is_sharded(c)
    expected = (a_cpu.to("spyre") + b_cpu.to("spyre")).cpu()
    assert torch.equal(c.cpu(), expected)


@requires_emulation
@expects_raise
def test_contiguous_prefix_view_of_sharded_fails_at_compile():
    # The hole the copy-path guards could not see: a contiguous view from
    # storage offset zero passes every is_contiguous()/storage_offset()
    # check, yet a kernel binding core c to shard c would map view elements
    # onto the wrong shards. It is a box view (copied to the CPU correctly,
    # see test_box_view_of_sharded_copies_to_cpu), so it is built, but the
    # compiled path must still reject it.
    # TODO: positive once compiled ops accept box views.
    n = num_shards()
    x = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)
    prefix = x.narrow(0, 0, 2 * n)

    with pytest.raises(RuntimeError, match="no views, slices or reshapes"):
        torch.compile(lambda a, b: a + b, dynamic=False)(prefix, prefix)


@requires_emulation
def test_identity_views_of_sharded_keep_round_trip():
    # Identity views (detach, as_strided of the base tensor) reproduce the
    # base tensor exactly, so the element-to-shard mapping still holds and
    # the whole-tensor copy path must keep working through them.
    n = num_shards()
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
    x.as_strided_(x.size(), x.stride(), 0)
    assert torch.equal(x.to("cpu"), expected)


@requires_emulation
@expects_raise
def test_mixed_dtypes_fail_at_compile():
    n = num_shards()
    torch.manual_seed(0)
    x = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="operands and result of one dtype"):
        torch.compile(lambda a, b: (a + b).float(), dynamic=False)(x, x)


@requires_emulation
@expects_raise
def test_multi_output_op_with_conversion_fails_at_compile():
    """argsort (through aten.sort, a multi-output op) feeding a dtype-converting
    consumer is rejected by the sharding checks, not by a crash in
    split_multi_ops ("'tuple' object has no attribute 'to'", #5831)."""
    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)
    b = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)

    # The sort result is an unsharded operand computed in the graph.
    unsharded_rejection = (
        "admit an unsharded operand only when it is a scalar constant or a graph input"
    )
    with pytest.raises(RuntimeError, match=unsharded_rejection):
        torch.compile(lambda x, y: torch.argsort(x, dim=1) + 0 * y, dynamic=False)(a, b)


@requires_emulation
@expects_raise
def test_dynamic_shapes_fail_at_compile():
    """torch.compile(dynamic=True) is refused before lowering, so a sharded
    operand never reaches the non-unified static-shape check."""
    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(
        RuntimeError, match=r"does not support: torch\.compile\(dynamic=True\)"
    ):
        torch.compile(lambda x, y: x + y, dynamic=True)(a, a)


@requires_emulation
@expects_raise
def test_unsharded_operand_read_by_reduction_fails_at_compile():
    """An unsharded operand is admitted only in a pointwise op, so a reduction
    that reads one is rejected at compile, not at launch with an error that
    names neither the op nor the sharding. cumsum decomposes to a matmul (a
    reduction) over the sharded operand and an unsharded triu mask. The
    computed-operand path is covered by
    test_scaled_unsharded_operand_fails_at_compile."""
    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)
    b = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(
        RuntimeError, match="admit an unsharded operand only in pointwise ops"
    ):
        torch.compile(lambda x, y: torch.cumsum(x, dim=1) + 0 * y, dynamic=False)(a, b)


@requires_emulation
@expects_raise
def test_scaled_unsharded_operand_fails_at_compile():
    """add(x, b, alpha=2) scales b inside the graph, so b is no longer a graph
    input: rejected, and the message says how to get past it.

    lower_add realizes y * alpha as its own buffer (the split_multi_ops "No FX
    node" workaround in lowering.py), which is why b counts as computed.
    TODO: positive once that workaround is gone and the mul fuses into the add.
    """
    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)
    bias = torch.randn(64, dtype=torch.float16).to("spyre")

    with pytest.raises(RuntimeError, match="computed in the graph.*pass it in"):
        torch.compile(lambda x, b: torch.add(x, b, alpha=2.0), dynamic=False)(a, bias)


@requires_emulation
@expects_raise
def test_fallback_consumer_fails_at_compile():
    """An op that lowers to a fallback kernel is rejected at compile, not at
    launch with an error that names neither the op nor the sharding."""
    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)

    # The fallback's result is the graph output: an op that combined it
    # with a sharded operand would be rejected first, as mixed
    # sharded/unsharded. The message names the ops non-unified kernels
    # support since the cross-shard-reduction work: pointwise and
    # shard-local reductions.
    with pytest.raises(
        RuntimeError, match="shard-local reductions on sharded operands only"
    ):
        torch.compile(
            lambda x: torch.ops.spyre.test_sharding_host_copy(x), dynamic=False
        )(a)


@requires_emulation
@expects_raise
@spyre_config.patch(bundle_symbolic_args=False)
def test_bundle_symbolic_args_disabled_fails_at_compile():
    """Every core's start address is a bundle input_arg, so a sharded kernel
    cannot be compiled onto the launch-time-bound path."""
    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="BUNDLE_SYMBOLIC_ARGS=1"):
        torch.compile(lambda x, y: x + y, dynamic=False)(a, a)


def _exact_source(shape, dtype) -> torch.Tensor:
    # Whole numbers below 1024: exact in fp32 and in the device fp16 format
    # (DL16 has one fraction bit fewer than IEEE fp16, so it rounds integers
    # above 1024), so the round trips can be compared exactly.
    return (torch.arange(torch.Size(shape).numel()) % 1024).reshape(shape).to(dtype)


@requires_emulation
@pytest.mark.parametrize(
    "shape_per_shard, dim, dtype",
    [
        # Rows of 50 elements end partway through a stick (32 fp32 or 64 fp16
        # per stick), so each shard is padded on the device.
        ((2, 50), 0, torch.float32),
        ((2, 50), 0, torch.float16),
        ((50,), 0, torch.float32),
        ((3, 7, 50), 1, torch.float32),
        # Shard(1) with 50 columns per shard: every shard row is padded.
        ((16, 50), 1, torch.float32),
    ],
)
def test_padded_shards_round_trip(shape_per_shard, dim, dtype):
    """Shards that end partway through a stick copy to and from the CPU."""
    n = num_shards()
    shape = list(shape_per_shard)
    shape[dim] *= n
    cpu = _exact_source(shape, dtype)

    x = _sharded(cpu, dim, n)

    # Padded on the device, or the test proves nothing about padding: 50
    # elements per row take 64 slots (e.g. 200 B of fp32 in 256 B).
    logical_bytes = torch.Size(shape_per_shard).numel() * cpu.element_size()
    assert torch.spyre.sharding_info(x).shard_device_bytes > logical_bytes
    # Control: if the unsharded round trip fails too, the fault is not in the
    # sharded path.
    assert torch.equal(cpu.to("spyre").cpu(), cpu)
    assert torch.equal(x.cpu(), cpu)


# The DMA slice size Flex uses: FLEX_DATA_TRANSFER_BUFFER_SIZE, else Flex's
# default of 256 MiB (RuntimeConfig::DataTransferBufferSize), mirrored here.
_DMA_SLICE_BYTES = int(
    os.environ.get("FLEX_DATA_TRANSFER_BUFFER_SIZE", 256 * 1024 * 1024)
)


@requires_emulation
@pytest.mark.skipif(
    _DMA_SLICE_BYTES > SMALL_DMA_SLICE_BYTES,
    reason="needs FLEX_DATA_TRANSFER_BUFFER_SIZE <= "
    f"{SMALL_DMA_SLICE_BYTES} at process start; "
    "test_device_suite_in_emulation_shape runs it in the EMUL1p5-small-dma mode",
)
@pytest.mark.parametrize(
    "shape_per_shard, dim, dtype",
    [
        ((8, 512), 0, torch.float32),  # 16 KiB per shard
        ((64, 128), 1, torch.float32),  # 32 KiB per shard
        ((16, 512), 0, torch.float16),  # 16 KiB per shard
        ((40, 50), 0, torch.float32),  # padded rows, several slices per shard
    ],
)
def test_shards_larger_than_one_dma_slice_round_trip(shape_per_shard, dim, dtype):
    """A shard whose copy is split into several DMA slices round-trips exactly."""
    n = num_shards()
    shape = list(shape_per_shard)
    shape[dim] *= n
    cpu = _exact_source(shape, dtype)
    x = _sharded(cpu, dim, n)

    # Several slices per shard, or the test proves nothing about splitting.
    assert torch.spyre.sharding_info(x).shard_device_bytes > 2 * _DMA_SLICE_BYTES
    assert torch.equal(cpu.to("spyre").cpu(), cpu)
    assert torch.equal(x.cpu(), cpu)


@requires_emulation
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("box", [False, True], ids=["whole", "box_view"])
def test_dtype_converting_copy_to_cpu(dtype, dim, box):
    """Gathering a sharded fp16/bf16 tensor (or a box view of it) as fp32
    matches the same copy of an unsharded tensor bit for bit (#5824).

    Compared with the unsharded device copy, not the host original: the device
    fp16 format (DL16) rounds some fp16 values.
    """
    n = num_shards()
    shape = (4 * n, 64) if dim == 0 else (8, 64 * n)
    torch.manual_seed(0)
    cpu = torch.randn(shape, dtype=dtype)

    def view(t):
        if not box:
            return t
        return t[1:5] if dim == 0 else t[:, 30:100]

    expected = view(cpu.to("spyre")).to("cpu", torch.float32)
    got = view(_sharded(cpu, dim, n)).to("cpu", torch.float32)

    assert got.dtype == torch.float32
    assert torch.equal(got, expected)


@requires_emulation
@expects_raise
def test_dtype_converting_copy_to_device_fails_closed():
    """A converting copy into a sharded tensor is rejected and leaves it as it
    was: only copies to the CPU may change the dtype so far (#5824)."""
    n = num_shards()
    torch.manual_seed(0)
    cpu = torch.randn(4 * n, 64, dtype=torch.float16)
    a = _sharded(cpu, 0, n)

    with pytest.raises(RuntimeError, match="to the device without a dtype change"):
        a.copy_(torch.zeros(4 * n, 64, dtype=torch.float32))
    assert torch.equal(a.cpu(), cpu.to("spyre").cpu())


@requires_emulation
@expects_raise
def test_dtype_view_of_sharded_fails_closed():
    """A same-size view(dtype) must not reach the copy path.

    view(dtype) is CompositeExplicitAutograd and never passes through
    check_sharded_view; the fp16 -> bf16 view keeps sizes, strides and
    offset, so it looks like an identity view, but its dtype disagrees
    with the recorded shard layout. TODO: reject at construction instead
    of at copy time.
    """

    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(4 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(
        RuntimeError, match="dtype no longer matches the sharded allocation"
    ):
        a.view(torch.bfloat16).cpu()


@requires_emulation
@expects_raise
def test_resize_and_set_of_sharded_tensor_fail_closed():
    """resize_/set_ must not silently corrupt or un-shard sharded storage.

    An in-place reinterpret would leave the sharding record describing the
    old shape (wrong rows gathered by .cpu()); a reallocating resize would
    silently convert the tensor to unified storage.
    """
    n = num_shards()
    torch.manual_seed(0)
    a = _sharded(torch.randn(8 * n, 64, dtype=torch.float16), 0, n)

    with pytest.raises(RuntimeError, match="resize_ is not supported"):
        a.resize_(4 * n, 64)
    with pytest.raises(RuntimeError, match="set_ is not supported"):
        a.set_(a.storage(), 0, [4 * n, 64], [64, 1])
    # The no-op resize is still fine and changes nothing.
    a.resize_(a.shape)
    assert torch.spyre.is_sharded(a)


@requires_emulation
def test_inputs_usable_after_compile_rejection():
    """A rejected compile leaves the sharded inputs intact and usable."""
    n = num_shards()
    torch.manual_seed(0)
    src = torch.randn(8 * n, 64, dtype=torch.float16)
    a = _sharded(src, 0, n)
    b = _sharded(src.clone(), 0, n)

    with pytest.raises(RuntimeError, match="must reduce the shard dim only"):
        torch.compile(lambda x, y: x.sum() + y.sum(), dynamic=False)(a, b)

    # Same tensors still work in a supported compile afterwards.
    c = torch.compile(lambda x, y: x + y, dynamic=False)(a, b)
    roundtrip = src.to("spyre").cpu()  # fp16 goes through the device format
    expected = roundtrip + roundtrip
    assert torch.equal(c.cpu(), expected)
    assert torch.equal(a.cpu(), roundtrip)


# ---------------------------------------------------------------------------
# Box views (#5817): a block of a sharded base in the same dimension order.
# They are built and copied to the CPU; every other consumer rejects them.
# ---------------------------------------------------------------------------


def _box_source(shape, dtype):
    # Whole numbers below 1024: exact in fp32 and in the device fp16 format
    # (DL16 has one fraction bit fewer than IEEE fp16, so it rounds integers
    # above 1024), so a wrong value shows where it came from.
    return (torch.arange(math.prod(shape)) % 1024).reshape(shape).to(dtype)


_BOX_VIEWS = {
    0: [
        ("inside_one_shard", lambda t, e: t[1 : e - 1]),
        ("one_whole_shard", lambda t, e: t[e : 2 * e]),
        ("across_shards", lambda t, e: t[e // 2 : 3 * e + 1]),
        ("one_row", lambda t, e: t[3 * e + 1 : 3 * e + 2]),
        ("prefix_narrow", lambda t, e: t.narrow(0, 0, 2 * e)),
        ("columns", lambda t, e: t[:, 16:48]),
        ("block", lambda t, e: t[e - 1 : 2 * e + 1, 5:37]),
        ("slice_of_slice", lambda t, e: t[e // 2 :][1 : 2 * e, 8:]),
    ],
    1: [
        ("inside_one_shard", lambda t, e: t[:, 1 : e - 1]),
        ("one_whole_shard", lambda t, e: t[:, e : 2 * e]),
        ("across_shards", lambda t, e: t[:, e // 2 : 3 * e + 1]),
        ("rows", lambda t, e: t[3:11]),
        ("block", lambda t, e: t[2:9, e - 3 : 2 * e + 3]),
        ("slice_of_slice", lambda t, e: t[:, e // 2 :][1:, 1 : 2 * e]),
    ],
}


@requires_emulation
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "dim, name",
    [(d, name) for d, views in _BOX_VIEWS.items() for name, _ in views],
)
def test_box_view_of_sharded_copies_to_cpu(dim, name, dtype):
    n = num_shards()
    # Shard(1): one stick (64 fp16 or 32 fp32 elements) or more per shard.
    shape = (4 * n, 64) if dim == 0 else (16, 64 * n)
    extent = shape[dim] // n
    view = dict(_BOX_VIEWS[dim])[name]
    cpu = _box_source(shape, dtype)
    x = _sharded(cpu, dim, n)

    v = view(x, extent)

    assert torch.spyre.is_sharded(v)
    assert v.shape == view(cpu, extent).shape
    assert torch.equal(v.cpu(), view(cpu, extent))
    assert torch.equal(x.cpu(), cpu)


@requires_emulation
def test_box_view_of_sharded_prints_its_values():
    n = num_shards()
    cpu = _box_source((4 * n, 64), torch.float32)
    x = _sharded(cpu, 0, n)

    assert str(x[2:7, 3:9]).replace(", device='spyre:0'", "") == str(cpu[2:7, 3:9])


@requires_emulation
@expects_raise
@pytest.mark.parametrize(
    "view",
    [
        lambda t: t[1],  # rank-reducing select
        lambda t: t[:, ::2],  # strided slice of the stick dimension
        lambda t: t[1:5].t(),  # a box, then transposed
        lambda t: t.view(-1)[3:70],  # flattened slice
        lambda t: t[:1].expand(4, 64),  # broadcast
        lambda t: t[2:4].unsqueeze(0),  # rank-increasing
    ],
)
def test_views_of_sharded_that_are_not_boxes_fail(view):
    n = num_shards()
    x = _sharded(_box_source((4 * n, 64), torch.float16), 0, n)

    with pytest.raises(
        RuntimeError, match="views of sharded Spyre tensors are not supported"
    ):
        view(x)


@requires_emulation
@expects_raise
def test_box_views_of_sharded_fail_in_other_consumers():
    """A box view is only copied to the CPU so far; everything else rejects
    it and leaves the sharded tensor untouched.
    TODO: positive one by one as #5817 follow-ups land."""
    n = num_shards()
    cpu = _box_source((4 * n, 64), torch.float16)
    x = _sharded(cpu, 0, n)
    v = x[2 : 2 * n]

    with pytest.raises(RuntimeError, match="copies into a view of a sharded"):
        v.copy_(torch.zeros(v.shape, dtype=v.dtype))
    with pytest.raises(RuntimeError, match="only support whole-tensor fills"):
        v.fill_(0)
    with pytest.raises(RuntimeError, match="only support whole-tensor fills"):
        v.zero_()
    with pytest.raises(TypeError, match="does not support views of sharded"):
        torch.spyre.sharding_info(v)
    with pytest.raises(RuntimeError, match="no views, slices or reshapes"):
        torch.compile(lambda a, b: a + b, dynamic=False)(v, v)
    # A detached block view has no _base, so the compile check must look at
    # its shape against the sharded allocation.
    d = x[: 2 * n].detach()
    with pytest.raises(RuntimeError, match="no views, slices or reshapes"):
        torch.compile(lambda a, b: a + b, dynamic=False)(d, d)
    # In-place as_strided_ rewrites the tensor itself, so it must keep the
    # whole shape even when the new geometry would be a box.
    with pytest.raises(
        RuntimeError, match="views of sharded Spyre tensors are not supported"
    ):
        x.as_strided_(v.shape, v.stride(), v.storage_offset())

    assert torch.equal(x.cpu(), cpu)
    assert torch.spyre.sharding_info(x).num_shards == n
