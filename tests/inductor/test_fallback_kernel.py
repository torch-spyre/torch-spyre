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

"""Regression tests for FallbackKernel lowering on the Spyre device.

Covers the three `FallbackKernel.create` output shapes (upstream
torch/_inductor/ir.py):

  shape 1 (single tensor)  -> MultiOutputLayout + 1 trailing MultiOutput
  shape 2 (tuple of N)     -> MultiOutputLayout + N trailing MultiOutputs
  shape 3 (void/in-place)  -> NoneLayout       + 0 trailing MultiOutputs

torch-spyre previously asserted "FallbackKernel must be followed by exactly
one MultiOutput" in two passes (propagate_layouts, work_division) and
unconditionally called `get_layout()` on every dependency in fusion's
non-intermediate counter. Shape 3 raised RuntimeError, shape 2 emitted
"unhandled node type MultiOutput" warnings with incomplete layout
propagation, and shape 3 separately tripped fusion via the NoneLayout
MutationOutput sentinels that void fallbacks register.

Plus `reinterpret_tensor` on the CPU buffers that fallbacks emit when a graph
mixes Spyre and CPU-C++ kernels (TestReinterpretTensorCpuBuffer).

Plus a traced spyre -> cpu -> spyre round-trip via plain `.to()`
(TestTracedDeviceCopy).

Plus a CPU tensor lifted as a graph input (TestLiftedCpuGraphInput) and CPU
pointwise ComputedBuffers inside the graph (TestInGraphCpuComputedBuffers) --
two ways a non-Spyre buffer enters a mixed CPU/Spyre graph. The lifted input
tripped propagate_layouts' graph-input loop ("missing device_tensor_layout on
graph input"); the CPU ComputedBuffers tripped the scratchpad planner
("'FixedLayout' object has no attribute 'device_layout'").

All tests run end-to-end through `torch.compile(..., backend="inductor")` on
the Spyre device to guard against regressions.
"""

import unittest

import torch
import torch.nn.functional as F

from torch_spyre._inductor import config


DEVICE = "spyre"
DTYPE = torch.float16


# Use FRAGMENT (not DEF) and guard against re-defining schemas, so the module
# is safe to import more than once — the test harness re-imports test files
# under different module names during analysis + execution, and DEF +
# unguarded define()/impl() would trip both
#   "Only a single TORCH_LIBRARY can be used to register the namespace ..."
# and
#   "Tried to register an operator ... with the same name multiple times".
def _ns_has_op(ns: str, op: str) -> bool:
    return hasattr(getattr(torch.ops, ns, None), op)


# Shape 1: op(x) -> Tensor
_LIB_S1 = torch.library.Library("test_fk_s1", "FRAGMENT")
if not _ns_has_op("test_fk_s1", "scale_two"):
    _LIB_S1.define("scale_two(Tensor x) -> Tensor")
    _LIB_S1.impl("scale_two", lambda x: x * 2, dispatch_key="CompositeExplicitAutograd")
    _LIB_S1._register_fake("scale_two", lambda x: torch.empty_like(x))


# Shape 2: op(x) -> (Tensor, Tensor)
_LIB_S2 = torch.library.Library("test_fk_s2", "FRAGMENT")
if not _ns_has_op("test_fk_s2", "split_two"):
    _LIB_S2.define("split_two(Tensor x) -> (Tensor, Tensor)")
    _LIB_S2.impl(
        "split_two",
        lambda x: (x + 1.0, x - 1.0),
        dispatch_key="CompositeExplicitAutograd",
    )
    _LIB_S2._register_fake(
        "split_two", lambda x: (torch.empty_like(x), torch.empty_like(x))
    )


# Shape 3: op(x, out) -> ()  (void / in-place mutation)
_LIB_S3 = torch.library.Library("test_fk_s3", "FRAGMENT")
if not _ns_has_op("test_fk_s3", "inplace_add"):
    _LIB_S3.define("inplace_add(Tensor x, Tensor(a!) out) -> ()")

    def _inplace_add_impl(x, out):
        out.add_(x)

    _LIB_S3.impl(
        "inplace_add", _inplace_add_impl, dispatch_key="CompositeExplicitAutograd"
    )
    _LIB_S3._register_fake("inplace_add", lambda x, out: None)


_LIB_CONV = torch.library.Library("test_fk_conv", "FRAGMENT")
if not _ns_has_op("test_fk_conv", "convert"):
    _LIB_CONV.define("convert(Tensor x, Device device) -> Tensor")
    _LIB_CONV.impl(
        "convert",
        lambda x, d: x.to(device=d).contiguous(),
        dispatch_key="CompositeExplicitAutograd",
    )
    _LIB_CONV._register_fake(
        "convert", lambda x, d: torch.empty(x.shape, dtype=x.dtype, device=d)
    )
_LIB_POOL = torch.library.Library("test_fk_pool", "FRAGMENT")
if not _ns_has_op("test_fk_pool", "norm"):
    _LIB_POOL.define("norm(Tensor x, Tensor residual) -> Tensor")
    _LIB_POOL.impl(
        "norm", lambda x, r: (x + r) * 0.5, dispatch_key="CompositeExplicitAutograd"
    )
    _LIB_POOL._register_fake("norm", lambda x, r: torch.empty_like(x))

if not _ns_has_op("test_fk_pool", "norm_inplace"):
    _LIB_POOL.define("norm_inplace(Tensor(a!) x, Tensor residual) -> ()")

    def _norm_inplace_impl(x, r):
        x.copy_((x + r) * 0.5)

    _LIB_POOL.impl(
        "norm_inplace", _norm_inplace_impl, dispatch_key="CompositeExplicitAutograd"
    )
    _LIB_POOL._register_fake("norm_inplace", lambda x, r: None)


# A gather whose eager kernel returns a non-canonically-tiled result when the
# index is size-1 (see TestFallbackResultNonCanonicalTiling).
_LIB_GATHER = torch.library.Library("test_fk_gather", "FRAGMENT")
if not _ns_has_op("test_fk_gather", "gather"):
    _LIB_GATHER.define("gather(Tensor cache, Tensor positions) -> Tensor")
    _LIB_GATHER.impl(
        "gather",
        lambda cache, positions: cache.index_select(0, positions.flatten()),
        dispatch_key="CompositeExplicitAutograd",
    )
    _LIB_GATHER._register_fake(
        "gather",
        lambda cache, positions: torch.empty(
            (positions.numel(), *cache.shape[1:]),
            dtype=cache.dtype,
            device=cache.device,
        ),
    )


# A concat whose eager result tiles to a different *rank* than the assumed
# layout, on a shape with no size-1 dim at all.
_LIB_CAT = torch.library.Library("test_fk_cat", "FRAGMENT")
if not _ns_has_op("test_fk_cat", "cat_two"):
    _LIB_CAT.define("cat_two(Tensor a, Tensor b) -> Tensor")
    _LIB_CAT.impl(
        "cat_two",
        lambda a, b: torch.cat([a, b], dim=0),
        dispatch_key="CompositeExplicitAutograd",
    )
    _LIB_CAT._register_fake(
        "cat_two",
        lambda a, b: torch.empty(
            (a.shape[0] + b.shape[0], *a.shape[1:]), dtype=a.dtype, device=a.device
        ),
    )


class TestFallbackKernelShape1Single(unittest.TestCase):
    """Shape 1: op(...) -> Tensor.

    Lowered to MultiOutputLayout FallbackKernel + 1 trailing MultiOutput.
    Was the only shape the original passes handled correctly; included
    here so all three shapes are covered by one test file.
    """

    def test_single_tensor_return_compiles(self):
        def fn(x):
            return torch.ops.test_fk_s1.scale_two(x) + 1.0

        x = torch.ones(4, dtype=DTYPE, device=DEVICE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x).cpu()
        torch.testing.assert_close(out, torch.full((4,), 3.0, dtype=DTYPE))


class TestFallbackKernelShape2Tuple(unittest.TestCase):
    """Shape 2: op(...) -> (Tensor, ..., Tensor).

    Lowered to MultiOutputLayout FallbackKernel + N trailing MultiOutputs.
    The original `next(it)` pattern consumed the first MultiOutput and
    fell through `unhandled node type MultiOutput` warnings for the rest;
    layout propagation was silently incomplete.
    """

    def test_two_tensor_return_compiles(self):
        def fn(x):
            a, b = torch.ops.test_fk_s2.split_two(x)
            return a * b

        x = torch.full((4,), 2.0, dtype=DTYPE, device=DEVICE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x).cpu()
        # (x + 1) * (x - 1) = x^2 - 1 = 3.0
        torch.testing.assert_close(out, torch.full((4,), 3.0, dtype=DTYPE))


class TestFallbackKernelShape3Void(unittest.TestCase):
    """Shape 3: op(...) -> () (void / in-place mutation).

    Lowered to NoneLayout FallbackKernel + 0 MultiOutputs. Upstream
    additionally registers MutationOutput sentinel buffers (one per
    mutated arg) with NoneLayout — those slip past fusion's
    `isinstance(buf, FallbackKernel)` guard and previously crashed
    `_is_non_intermediate` via `get_layout()`.

    This op signature mirrors the vLLM
    `torch.ops.vllm.unified_attention_with_output(...)` contract that
    triggered the original bug report.
    """

    def test_void_inplace_compiles(self):
        def fn(x):
            out = torch.zeros_like(x)
            torch.ops.test_fk_s3.inplace_add(x, out)
            return out + 1.0

        x = torch.full((4,), 5.0, dtype=DTYPE, device=DEVICE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x).cpu()
        # zeros + x + 1 = 6.0
        torch.testing.assert_close(out, torch.full((4,), 6.0, dtype=DTYPE))


class TestTracedDeviceCopy(unittest.TestCase):
    """A traced spyre -> cpu -> spyre round-trip must not crash.

    A plain `.to()` that Dynamo can trace lowers to an in-graph
    `DeviceCopy`, so the schedule mixes Spyre and CPU nodes -- which used
    to break the Spyre passes (propagate_layouts, work_division,
    spyre_fuse_nodes) and the `SpyreAsyncCompile` stub. Sibling to the
    custom-op test in the `reinterpret_device_fix` branch.
    """

    def test_cpu_slice_roundtrip_compiles(self):
        def fn(x):
            x_cpu = x.to("cpu")
            d = x_cpu.shape[-1] // 2
            x1 = x_cpu[..., :d].to(DEVICE)
            x2 = x_cpu[..., d:].to(DEVICE)
            return F.silu(x1) * x2

        x = torch.randn(16, 256, dtype=DTYPE, device=DEVICE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x)
        self.assertEqual(out.device.type, DEVICE)
        torch.testing.assert_close(out.cpu(), fn(x).cpu(), atol=0.01, rtol=0.01)


class TestReinterpretTensorCpuBuffer(unittest.TestCase):
    """`reinterpret_tensor` on a CPU buffer must not crash.

    The Spyre `reinterpret_tensor` binding used to `static_cast` its input to
    SpyreTensorImpl unconditionally and read `spyre_layout` — undefined
    behaviour on the CPU buffers a graph produces when it mixes Spyre and
    CPU-C++ kernels, crashing with `std::bad_array_new_length`. Here the host
    slices `x_cpu[..., :d]` / `[..., d:]` lower to `reinterpret_tensor(cpu_buf,
    ...)` views feeding the convert-back-to-Spyre fallbacks — the exact shape
    that tripped the cast. The fix guards on device type and delegates
    non-Spyre inputs to PyTorch's own `_reinterpret_tensor`.
    """

    def test_cpu_slice_roundtrip_compiles(self):
        cpu = torch.device("cpu")
        spyre = torch.device(DEVICE)

        def fn(x):
            x_cpu = torch.ops.test_fk_conv.convert(x, cpu)
            d = x_cpu.shape[-1] // 2
            x1 = torch.ops.test_fk_conv.convert(x_cpu[..., :d], spyre)
            x2 = torch.ops.test_fk_conv.convert(x_cpu[..., d:], spyre)
            return F.silu(x1) * x2

        x = torch.randn(16, 256, dtype=DTYPE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x.to(spyre))
        self.assertEqual(out.device.type, DEVICE)
        torch.testing.assert_close(out.cpu(), fn(x).cpu(), atol=0.01, rtol=0.01)


class TestFallbackKernelPoolResidentArg(unittest.TestCase):
    """FallbackKernel consuming an intermediate buffer keeps the correct dtype."""

    def test_fresh_output_arg_keeps_dtype(self):
        def fn(x):
            residual = x
            x = torch.ops.test_fk_pool.norm(x, residual)
            x = residual + x  # pool-eligible intermediate, read by fallback
            residual = x
            x = torch.ops.test_fk_pool.norm(x, residual)
            return residual + x

        x = torch.randn(16, 4096, dtype=DTYPE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x.to(DEVICE))
        self.assertEqual(out.dtype, DTYPE)
        torch.testing.assert_close(out.cpu(), fn(x), atol=0.01, rtol=0.01)

    def test_inplace_arg_keeps_dtype(self):
        def fn(x):
            residual = x.clone()
            torch.ops.test_fk_pool.norm_inplace(x, residual)  # x mutated
            x = residual + x  # pool-eligible intermediate, read by fallback
            residual = x.clone()
            torch.ops.test_fk_pool.norm_inplace(x, residual)  # x mutated
            return residual + x

        x = torch.randn(16, 4096, dtype=DTYPE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x.clone().to(DEVICE))
        self.assertEqual(out.dtype, DTYPE)
        torch.testing.assert_close(out.cpu(), fn(x.clone()), atol=0.01, rtol=0.01)


class TestLiftedCpuGraphInput(unittest.TestCase):
    """A CPU tensor lifted as a graph input must not crash layout propagation.

    When a compiled fn closes over (or Dynamo/AOTAutograd constant-folds) a CPU
    tensor that has no data-dependency on any declared input, it is lifted as an
    extra graph-input placeholder. That input is CPU-resident, so
    `device_tensor_layout()` returns None. propagate_layouts' graph-input loop
    used to treat that as fatal and raised
    `missing device_tensor_layout on graph input <name>`; it now skips the input
    (leaving its FixedLayout intact), mirroring the non-Spyre ComputedBuffer skip
    further down the same pass. The lifted CPU input's only consumer here is the
    convert fallback that moves it onto Spyre.

    """

    def test_lifted_cpu_constant_matmul_compiles(self):
        spyre = torch.device(DEVICE)
        inner, padded, num_tokens = 32, 64, 16

        # Built outside the compiled region -> closed over -> lifted as a CPU
        # graph input. A {0,1} expand matrix, like RoPE's _get_expand_matrix.
        e_cpu = torch.zeros(2 * inner, 2 * padded, dtype=DTYPE)
        idx = torch.arange(inner)
        e_cpu[idx, idx] = 1
        e_cpu[inner + idx, padded + idx] = 1

        def fn(x):
            e = torch.ops.test_fk_conv.convert(e_cpu, spyre)
            return x @ e

        x = torch.randn(num_tokens, 2 * inner, dtype=DTYPE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x.to(spyre))
        self.assertEqual(out.device.type, DEVICE)
        self.assertEqual(tuple(out.shape), (num_tokens, 2 * padded))
        torch.testing.assert_close(out.cpu(), x @ e_cpu, atol=0.01, rtol=0.01)


class TestInGraphCpuComputedBuffers(unittest.TestCase):
    """CPU pointwise ComputedBuffers in a mixed graph must not crash scratchpad planning.

    A convert fallback returns a CPU tensor; a chain of pointwise ops
    (add/sub/mul -- all in OP_OUTPUT_GOOD_FOR_LX_REUSE) then runs on it inside
    the same graph, before converting back to Spyre. propagate_layouts SKIPS
    those CPU ComputedBuffers (device != spyre), leaving them a plain
    `FixedLayout` with no `device_layout`. The scratchpad planner's op gate
    (`_op_output_good_for_lx_reuse`) whitelisted by op NAME only, so a CPU
    add/sub/mul passed the gate, entered graph_view, and reached
    `mem_usage_by_buf`, which read `layout.device_layout` and raised
    `'FixedLayout' object has no attribute 'device_layout'`. The gate now also
    requires a Spyre `FixedTiledLayout`, so CPU buffers stay out of the planner.

    Same class of bug as spyre-inference's RoPE `_get_expand_matrix` (a CPU
    constant built in-graph) and the TP>1 vocab-shard mask. Uses tensor-bound
    ops (no dtype change / no scalar-int constants) so the ONLY thing under test
    is "CPU pointwise ComputedBuffer in a Spyre graph".

    Both scratchpad allocators are covered: the default greedy allocator (which
    filters ops through `_op_output_good_for_lx_reuse` -> the gate fix) and the
    co-optimizing allocator (`co_optimizing_lx_planning`), whose `_search`
    footprint accounting reads `.device_layout` over every buffer -> the
    `buf_total_bytes` guard.
    """

    @staticmethod
    def _run_and_check(test: "unittest.TestCase") -> None:
        cpu = torch.device("cpu")
        spyre = torch.device(DEVICE)
        num_tokens, hidden = 16, 256

        def fn(x):
            # Opaque spyre -> cpu: CPU FallbackKernel output.
            x_cpu = torch.ops.test_fk_conv.convert(x, cpu)
            # CPU pointwise chain -> CPU ComputedBuffers (add/sub/mul).
            y = (x_cpu + 1.0) * (x_cpu - 1.0)
            # Opaque cpu -> spyre and an on-device consumer.
            y_s = torch.ops.test_fk_conv.convert(y, spyre)
            return y_s * 2.0

        x = torch.randn(num_tokens, hidden, dtype=DTYPE)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(x.to(spyre))
        test.assertEqual(out.device.type, DEVICE)
        # ((x + 1)(x - 1)) * 2 = (x^2 - 1) * 2
        torch.testing.assert_close(
            out.cpu(), ((x + 1.0) * (x - 1.0)) * 2.0, atol=0.05, rtol=0.05
        )

    def test_cpu_pointwise_chain_compiles_greedy(self):
        """Default greedy allocator: exercises the `_op_output_good_for_lx_reuse`
        gate (mem_usage_by_buf over the filtered graph_view)."""
        self._run_and_check(self)

    @config.patch({"co_optimizing_lx_planning": True})
    def test_cpu_pointwise_chain_compiles_co_optimizing(self):
        """Co-optimizing allocator: `mem_usage_by_buf` runs on the RAW graph in
        `_build_cd_bound_buffers` / `_determine_in_place_division_invariant`,
        bypassing the gate -> exercises the defensive sentinel."""
        self._run_and_check(self)


class TestFallbackResultNonCanonicalTiling(unittest.TestCase):
    """A fallback whose eager kernel returns a non-canonically-tiled buffer.

    `propagate_layouts` stamps every `MultiOutput` with `generic_layout(op)` --
    the size-only `SpyreTensorLayout(size, dtype)` ctor, which derives
    `dim_order` from the generic stick order and strides from a dense row-major
    assumption. Consumers then index the fallback's buffer by *that* assumed
    tiling. Because the node also gets `AnyInNode` (whose
    `required_input_stls()` is empty), no restickify is ever inserted to make
    the assumption true, so the graph's layout contract rests entirely on the
    eager op returning a canonically-tiled result.

    `aten.index_select` on a size-1 index does not: a size-1 dim has an
    arbitrary stride, so nothing pins where it sorts in `dim_order` and the real
    tiling permutes relative to the canonical one. Consumers then read the wrong
    elements -- silently, with no error and (for a pure tile permutation)
    plausible-looking magnitudes.

    Shape mirrors the vLLM RoPE cos/sin cache gather that surfaced this
    (`(T, 2, 2, head_size // 2)` gathered from a position cache, then
    broadcast-multiplied and reduced): T == 1 is the single-token decode step,
    where the bug bites. T == 8 is the control -- there the assumed and real
    tilings coincide, so it passed throughout and is what made the bug look
    shape-dependent rather than layout-dependent.
    """

    INNER = 64  # == elems_per_stick at fp16
    HEADS = 4
    MAXPOS = 16

    def _run(self, num_tokens: int):
        cache = torch.randn(self.MAXPOS, 2, 2, self.INNER, dtype=DTYPE)
        x = torch.randn(num_tokens, self.HEADS * 2 * self.INNER, dtype=DTYPE) * 0.5
        pos = torch.arange(num_tokens, dtype=torch.int64)

        def fn(x, cache, pos):
            # index_select is the fallback; its result feeds a broadcast
            # multiply + reduction that reads it by the assumed tiling.
            rot = torch.ops.test_fk_gather.gather(cache, pos)
            pairs = x.view(num_tokens, -1, 2, self.INNER)
            out = (rot.unsqueeze(1) * pairs.unsqueeze(-3)).sum(dim=-2)
            return out.flatten(-2).view(x.shape)

        expected = fn(x.float(), cache.float(), pos)

        args = [t.to(DEVICE) for t in (x, cache)] + [pos.to(DEVICE)]
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")
        out = compiled(*args).cpu().float()

        # A tile permutation preserves norms, so compare elementwise. Tolerance
        # is the fp16 accumulation floor; the bug produced errors of O(1..7).
        torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)

    def test_single_token_gather(self):
        """T == 1: assumed and real tilings disagree (the regression)."""
        self._run(1)

    def test_multi_token_gather(self):
        """T == 8: they coincide -- guards the control arm too."""
        self._run(8)

    def test_cat_disagrees_in_rank_without_size_one_dim(self):
        """`aten.cat(dim=0)` at (8, 2, 2, 64): a second, distinct instance.

        Kept separate because it refutes the obvious-looking shape heuristic
        ("flag size-1 axes outside the stick dim"): this shape has no size-1 dim
        at all, and the two layouts disagree in *rank* (the eager result tiles to
        6 dims, `generic_layout` assumes 5). Any fix must therefore compare the
        real layout against the assumed one directly rather than infer ambiguity
        from the logical shape.
        """
        half = torch.randn(4, 2, 2, self.INNER, dtype=DTYPE)
        scale = torch.randn(8, 2, 2, self.INNER, dtype=DTYPE)

        def fn(half, scale):
            joined = torch.ops.test_fk_cat.cat_two(half, half)
            return joined * scale

        expected = fn(half.float(), scale.float())
        out = torch.compile(fn, fullgraph=True, dynamic=False, backend="inductor")(
            half.to(DEVICE), scale.to(DEVICE)
        )
        torch.testing.assert_close(out.cpu().float(), expected, atol=5e-2, rtol=5e-2)


class TestPermutedEagerResultNotNormalized(unittest.TestCase):
    """A permuted eager result must be left alone, not forced to dense tiling.

    `_normalize_result_layout` compares against `SpyreTensorLayout(size, dtype)`,
    which synthesizes *dense row-major* strides. That is only the layout a
    consumer would assume for a tensor that is itself dense row-major. A permuted
    result is not: TensorIterator propagates its inputs' strides through
    elementwise ops, so adding two permuted views yields a permuted output whose
    real tiling legitimately differs from the size-only one. Normalizing it
    rewrites a buffer that was already self-consistent, corrupting it.

    These ran green before the normalization existed and regressed the moment it
    was added without a contiguity gate (six `permute` cases in
    `test_inductor_ops_lx_planning.py`); they are duplicated here, in the suite
    that owns the fix, so a future change to the predicate fails locally rather
    than only in CI.
    """

    # (base shape, permutation) -- the permuted result is non-contiguous, and its
    # real device tiling differs from the size-only layout for its logical shape.
    CASES = (
        ((2, 1024, 844), (0, 2, 1)),
        ((64, 128, 256), (2, 0, 1)),
        ((2, 7, 11, 13), (0, 2, 1, 3)),
    )

    def test_permuted_addition_is_not_corrupted(self):
        for base, perm in self.CASES:
            with self.subTest(base=base, perm=perm):
                a = torch.randn(base, dtype=DTYPE)
                b = torch.randn(base, dtype=DTYPE)
                av, bv = a.permute(*perm), b.permute(*perm)
                self.assertFalse(av.is_contiguous())

                expected = av.float() + bv.float()
                out = a.to(DEVICE).permute(*perm) + b.to(DEVICE).permute(*perm)
                torch.testing.assert_close(
                    out.cpu().float(), expected, atol=5e-2, rtol=5e-2
                )


class TestMapResultReconstruction(unittest.TestCase):
    """`_map_result` must rebuild every container an aten schema can return.

    The end-to-end tests above only reach single-tensor results: no op currently
    registered by `register_torch_compile_kernel` declares more than one return
    (116 overloads, none multi-tensor), so the tuple/list branches are reachable
    only if that list grows. They are exercised directly here rather than left to
    a future registration to discover.

    The interesting case is a tuple SUBCLASS. `structseq` -- what the multi-output
    aten schemas (`aten.max.dim`, `aten.sort`, ...) actually return -- and
    namedtuple both rebuild from an iterable via `_make`, but not by calling the
    type with a list. `_map_result` must therefore not reconstruct via
    `type(result)(mapped)`.
    """

    def _fn(self, t):
        return t + 1

    def test_tensor_and_passthrough(self):
        from torch_spyre.ops.eager import _map_result

        self.assertEqual(_map_result(torch.zeros(2), self._fn).tolist(), [1.0, 1.0])
        # Non-tensor leaves are returned untouched.
        for leaf in (None, 3, "s"):
            self.assertIs(_map_result(leaf, self._fn), leaf)

    def test_tuple_list_and_nesting(self):
        from torch_spyre.ops.eager import _map_result

        out = _map_result((torch.zeros(1), torch.zeros(1)), self._fn)
        self.assertIsInstance(out, tuple)
        self.assertEqual([t.item() for t in out], [1.0, 1.0])

        out = _map_result([torch.zeros(1), None], self._fn)
        self.assertIsInstance(out, list)
        self.assertEqual(out[0].item(), 1.0)
        self.assertIsNone(out[1])

        out = _map_result(((torch.zeros(1),), [torch.zeros(1)]), self._fn)
        self.assertEqual(out[0][0].item(), 1.0)
        self.assertEqual(out[1][0].item(), 1.0)

    def test_structseq_and_namedtuple_are_rebuilt_as_their_own_type(self):
        from collections import namedtuple

        from torch_spyre.ops.eager import _map_result

        # structseq: the real return type of multi-output aten ops.
        real = torch.max(torch.zeros(2, 2), dim=0)
        out = _map_result(real, lambda t: t)
        self.assertIsInstance(out, type(real))
        self.assertEqual(out.values.shape, real.values.shape)

        Pair = namedtuple("Pair", "first second")
        out = _map_result(Pair(torch.zeros(1), torch.zeros(1)), self._fn)
        self.assertIsInstance(out, Pair)
        self.assertEqual(out.first.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
