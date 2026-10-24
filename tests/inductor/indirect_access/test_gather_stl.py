# Copyright 2025 The Torch-Spyre Authors.
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

"""SpyreTensorLayout (STL) annotation tests.

Three groups:

  1. Implicit-layout gather (TestGatherImplicitLayout) — tensors moved to Spyre
     via plain ``.to(DEVICE)``; the compiler infers the generic-stick layout.
     Covers shape/dtype variety, downstream ops, named dims, int32 vs int64
     index dtypes, and real-model stride patterns (Granite/RoPE).

  2. Explicit 2-arg STL annotation (TestGatherExplicitStl2Arg) — tensors moved
     to Spyre with ``SpyreTensorLayout(shape, dtype)`` via
     ``.to(DEVICE, device_layout=stl)``.  Verifies the 2-arg constructor path
     round-trips correctly through compile across shapes, dtypes, and both
     int32 and int64 index dtypes.

  3. Explicit 4-arg STL annotation (TestGatherExplicitStl4Arg) — tensors
     annotated with the full ``SpyreTensorLayout(device_size, stride_map,
     device_dtype)`` 3-keyword constructor.  Exercises GQA/MQA KV cache shapes,
     downstream ops, bfloat16, and named dims.  A final section asserts that
     explicit-STL and implicit compiled outputs are bit-identical.

Index dtype policy
------------------
The Spyre hardware executes gather with int32 indices; int64 tensors are
silently downcast at the hardware boundary (see types_mapping.h).  Both dtypes
are valid at the compiler level (for_each_tile.py checks this).  Each class
therefore covers:
  * ``int64`` — the normal PyTorch default; exercises the downcast pipeline.
  * ``int32`` — arrives at hardware width directly; verifies no double-cast.

SENCORES and execution_mode parametrisation is driven by conftest
``pytest_generate_tests`` + ``patch_sencores`` fixture — the same mechanism
used by ``test_gather_attention_up.py``.  Classes carry no ``@parametrize``
decorator; do not add manual ``os.environ["SENCORES"]`` calls.

Known xfail markers
-------------------
_INDEX_EAGER : issue #1219 — aten::index.Tensor_out not registered on Spyre
               eager (and int32 type-conversion that precedes it).
_SYMBOLIC_INT_CONV : issue #4304 — dynamic=True / symbolic size Cannot convert
               symbols to int.
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn, compare_with_cpu  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

try:
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    _HAS_STL = True
except ImportError:
    _HAS_STL = False

import torch_spyre._inductor.wsr.propagate_named_dims as _pnd

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5

# ---------------------------------------------------------------------------
# Module-level xfail issue markers
# ---------------------------------------------------------------------------

# aten::index.Tensor_out is not registered on Spyre eager.  Also covers the
# int32→int64 type-conversion InductorError which fires before the gather op.
_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")

# torch.compile(dynamic=True): Cannot convert symbols to int during codegen.
_SYMBOLIC_INT_CONV = (
    4304,
    "compile_dyn / symbolic size Cannot convert symbols to int.",
)

_requires_stl = pytest.mark.skipif(
    not _HAS_STL, reason="SpyreTensorLayout C extension not available"
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _stl_2arg(shape, dtype) -> "SpyreTensorLayout":
    """2-arg constructor: ``SpyreTensorLayout(shape_list, dtype)``."""
    return SpyreTensorLayout([int(s) for s in shape], dtype)


def _stl_generic(shape, dtype) -> "SpyreTensorLayout":
    """Full 3-keyword constructor matching the canonical generic-stick layout.

    Replicates ``canonical_device_layout`` from indirect_access_common.py:
        shape (H, W)    -> device_size [H, W//eps, eps], stride_map [W, eps, 1]
        shape (A, B, W) -> device_size [A, B, W//eps, eps], stride_map [B*W, W, eps, 1]

    The stick width is read from ``SpyreTensorLayout(shape, dtype).elems_per_stick()``
    so it stays in sync with the runtime definition.
    """
    shape = [int(s) for s in shape]
    eps = SpyreTensorLayout(shape, dtype).elems_per_stick()
    *lead, last = shape
    stick_count = math.ceil(last / eps)
    device_size = [*lead, stick_count, eps]
    stride_map = [math.prod(shape[k + 1 :]) for k in range(len(lead))] + [eps, 1]
    return SpyreTensorLayout(
        device_size=device_size,
        stride_map=stride_map,
        device_dtype=get_device_dtype(dtype),
    )


def _compare_explicit_stl(
    fn, x_cpu, idx_cpu, x_dev, idx_dev, execution_mode, atol, rtol
):
    """Run *fn* with an explicitly-STL-annotated device tensor and compare to CPU.

    ``x_dev`` / ``idx_dev`` are already on DEVICE with the explicit layout.
    ``_compile_and_run`` inside ``compare_with_cpu`` calls ``.to(DEVICE)`` on
    them — a no-op for tensors already on the target device — so the explicit
    layout is preserved at graph-input time.

    The CPU reference is computed here from ``x_cpu`` / ``idx_cpu`` (plain CPU
    tensors) and passed as ``cpu_eager_result`` so ``compare_with_cpu`` does not
    re-run ``fn(x_dev, idx_dev)`` as a "CPU reference", which would silently
    execute in Spyre-eager mode instead of on CPU.
    """
    cpu_ref = fn(x_cpu, idx_cpu)
    compare_with_cpu(
        fn,
        x_dev,
        idx_dev,
        atol=atol,
        rtol=rtol,
        cpu_eager_result=cpu_ref,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


def _assert_explicit_eq_implicit(x_cpu, idx_cpu):
    """Assert explicit 4-arg STL and implicit ``.to(DEVICE)`` yield bit-identical output.

    Compiles ``x[i]`` twice — once with ``x`` annotated via ``_stl_generic``,
    once with plain ``x.to(DEVICE)`` — and checks the results are equal.
    Both tensors hold the same data; only their device-layout annotation differs.
    Because the canonical generic-stick layout ``_stl_generic`` produces is
    identical to the layout the compiler infers automatically, the outputs must
    match bit-for-bit.

    Only meaningful for compiled mode: the compile-path layout annotation is not
    exercised in eager mode.  Call this after ``pytest.skip`` for eager.
    """
    stl = _stl_generic(x_cpu.shape, x_cpu.dtype)
    x_explicit = x_cpu.to(DEVICE, device_layout=stl)
    x_implicit = x_cpu.to(DEVICE)
    idx_dev = idx_cpu.to(DEVICE)
    fn = lambda x, i: x[i]  # noqa: E731
    torch._dynamo.reset_code_caches()
    out_explicit = torch.compile(fn)(x_explicit, idx_dev).cpu()
    torch._dynamo.reset_code_caches()
    out_implicit = torch.compile(fn)(x_implicit, idx_dev).cpu()
    torch.testing.assert_close(out_explicit, out_implicit, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Group 1 — Implicit-layout gather
# ---------------------------------------------------------------------------


class TestGatherImplicitLayout:
    """Gather with plain ``.to(DEVICE)`` (implicit generic-stick layout).

    The compiler infers the layout from shape/dtype; no explicit STL object is
    constructed.  Covers shape/dtype variety, downstream ops, named dims,
    int32 and int64 index dtypes, and real-model stride patterns.

    SENCORES and execution_mode are driven by conftest pytest_generate_tests +
    patch_sencores; do not add manual os.environ["SENCORES"] calls.
    """

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    # ------------------------------------------------------------------
    # Shape / dtype variety

    def test_2d_square(self, execution_mode):
        """(128,64) gather at dim=0; narrow inner dim forces non-trivial stick packing."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((128, 64), differentiation="gstl01", dtype=torch.float16)
        idx = torch.randint(0, 128, (48,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_2d_frs_example_a(self, execution_mode):
        """FRS canonical Example A — (12,1024) gather at dim=0, P=8."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((12, 1024), differentiation="gstl02", dtype=torch.float16)
        idx = torch.randint(0, 12, (8,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_2d_16x512(self, execution_mode):
        """(16,512), 8 sticks in inner dim; stick grid."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((16, 512), differentiation="gstl03", dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_2d_int32_idx(self, execution_mode):
        """int32 index arrives at hardware width directly; no double-downcast.

        Eager fails with InductorError (int32→int64 type conversion unsupported)
        before reaching the gather op — same root cause as _INDEX_EAGER (#1219).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((48, 192), differentiation="gstl04i32", dtype=torch.float16)
        idx = torch.randint(0, 48, (24,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_2d_with_tanh(self, execution_mode):
        """(24,128) gather + tanh downstream co-scheduled."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((24, 128), differentiation="gstl05", dtype=torch.float16)
        idx = torch.randint(0, 24, (12,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.tanh(x[i]),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_2d_with_scalar_div(self, execution_mode):
        """(20,256) gather + /8.0 attention scale downstream."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((20, 256), differentiation="gstl06", dtype=torch.float16)
        idx = torch.randint(0, 20, (10,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] / 8.0,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_2d_with_sum_reduction(self, execution_mode):
        """(24,192) gather + sum(dim=1) reduction; shape collapses to (12,)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((24, 192), differentiation="gstl07", dtype=torch.float16)
        idx = torch.randint(0, 24, (12,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i].sum(dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_interior_dim_no_stl_needed(self, execution_mode):
        """x[:,idx,:] — dim=1 gather on (5,64,512); no STL injection needed."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((5, 64, 512), differentiation="gstl08", dtype=torch.float16)
        idx = torch.randint(0, 64, (8,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_3d_kv_shape(self, execution_mode):
        """3D (8,32,128) gather at dim=0, P=4."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((8, 32, 128), differentiation="gstl09", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_2d_bfloat16(self, execution_mode):
        """bfloat16 value tensor; SDSC wordLength and stride map for bf16."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((16, 128), differentiation="gstl18", dtype=torch.bfloat16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16
        )

    # ------------------------------------------------------------------
    # Compiler features

    def test_with_named_dims(self, execution_mode):
        """Gather + name_tensor_dims on value and index."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((64, 128), differentiation="gstl10", dtype=torch.float16)
        idx = torch.randint(0, 64, (8,), dtype=torch.int64)
        _pnd.name_tensor_dims(x, ["M", "N"])
        _pnd.name_tensor_dims(idx, ["P"])
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_dynamic_compile(self, execution_mode):
        """torch.compile(dynamic=True); three runtime index sizes.

        Fails in all modes with 'Cannot convert symbols to int' (issue #4304).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing(execution_mode, eager=_INDEX_EAGER, compiled=_SYMBOLIC_INT_CONV)
        x = cached_randn((40, 128), differentiation="gstl11", dtype=torch.float16)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for p in (8, 16, 24):
            idx = torch.randint(0, 40, (p,), dtype=torch.int64)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_api_equivalence(self, execution_mode):
        """index_select(x,0,idx) on Spyre matches CPU reference (float16, implicit layout)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 256), differentiation="cmp08a", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------
    # Production-scale shapes

    def test_embedding_shape(self, execution_mode):
        """Embedding (1024,128); stick-aligned inner dim at production scale."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((1024, 128), differentiation="gstl15", dtype=torch.float16)
        idx = torch.randint(0, 1024, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_kv_cache_shape(self, execution_mode):
        """KV cache 3D (256,8,64); paged attention page lookup."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((256, 8, 64), differentiation="gstl16", dtype=torch.float16)
        idx = torch.randint(0, 256, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_moe_shape(self, execution_mode):
        """MoE expert weight (8,512,64); large inner dims."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((8, 512, 64), differentiation="gstl17", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------
    # Stride-exact real-model patterns

    def test_stride_2d_standard(self, execution_mode):
        """(32,64) standard strides [64,1]; SDSC stride map = [64,1]."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 64), differentiation="str_s2d", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        assert x.stride() == (64, 1)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_granite_kv_strides(self, execution_mode):
        """Granite-3.3-8b/Ministral exact strides (1,8,2048,128).

        Gathers 41 sequence positions from the KV cache.
        squeeze(0) → (8,2048,128); index_select(dim=1) selects from seq_len=2048.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        cache = cached_randn(
            (1, 8, 2048, 128), differentiation="str_grt", dtype=torch.float16
        )
        idx = torch.randint(0, 2048, (41,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda c, i: torch.index_select(c.squeeze(0), 1, i),
            cache,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_stride_rope_cache(self, execution_mode):
        """RoPE cos/sin cache (4096,128) gather; stride [128,1]."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        cos_sin = cached_randn(
            (4096, 128), differentiation="str06", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (128,), dtype=torch.int64)
        assert cos_sin.stride() == (128, 1)
        compare_mode(
            execution_mode,
            lambda c, p: torch.index_select(c, 0, p),
            cos_sin,
            pos,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_stride_non_contiguous_t(self, execution_mode):
        """Transposed 2D (M,N).T strides — stride mismatch triggers restickify."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        base = cached_randn((64, 32), differentiation="str09", dtype=torch.float16)
        x = base.t().contiguous()
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_stride_float32_kv(self, execution_mode):
        """float32 KV (512,8,64) exact strides; wordLength=4."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        kv = cached_randn(
            (512, 8, 64), differentiation="str_f32kv", dtype=torch.float32
        )
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F32, rtol=_ATOL_F32
        )


# ---------------------------------------------------------------------------
# Group 2 — Explicit 2-arg STL annotation
# ---------------------------------------------------------------------------


@_requires_stl
class TestGatherExplicitStl2Arg:
    """Gather source pre-annotated with ``SpyreTensorLayout(shape, dtype)`` (2-arg form).

    ``_compile_and_run`` calls ``.to(DEVICE)`` on the already-on-device tensor —
    a no-op — so the explicit layout is preserved at graph-input time.
    ``_compare_explicit_stl`` computes the CPU reference from the original CPU
    tensors so the comparison is always against true CPU output.

    Both int32 and int64 index dtypes are covered: int64 exercises the
    downcast pipeline; int32 verifies the compiler does not double-cast.

    Eager fails for all tests with aten::index.Tensor_out not registered (or the
    int32→int64 type conversion that precedes it); xfail'd via _INDEX_EAGER.

    SENCORES and execution_mode come from conftest pytest_generate_tests +
    patch_sencores.
    """

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    def _to_dev(self, x_cpu):
        """Move *x_cpu* to Spyre using the 2-arg STL constructor."""
        return x_cpu.to(DEVICE, device_layout=_stl_2arg(x_cpu.shape, x_cpu.dtype))

    def _run(self, fn, x_cpu, idx_cpu, execution_mode, atol, rtol):
        """Build device tensors and delegate to ``_compare_explicit_stl``."""
        _compare_explicit_stl(
            fn,
            x_cpu,
            idx_cpu,
            self._to_dev(x_cpu),
            idx_cpu.to(DEVICE),
            execution_mode,
            atol,
            rtol,
        )

    # ------------------------------------------------------------------

    def test_2arg_2d_float16_int64(self, execution_mode):
        """(32,256) float16 + int64 index; 2-arg STL round-trips through compile."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 256), differentiation="e2arg01", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    def test_2arg_2d_float16_int32(self, execution_mode):
        """(32,256) float16 + int32 index; hardware-native width, no double-cast."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 256), differentiation="e2arg01i32", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    def test_2arg_3d_kv(self, execution_mode):
        """2-arg STL on (512,8,64) KV cache shape."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((512, 8, 64), differentiation="e2arg02", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    def test_2arg_bfloat16(self, execution_mode):
        """2-arg STL on bfloat16 (64,64)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((64, 64), differentiation="e2arg03", dtype=torch.bfloat16)
        idx = torch.randint(0, 64, (8,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_BF16, _ATOL_BF16)

    def test_2arg_embedding_shape(self, execution_mode):
        """2-arg STL on embedding (1024,128)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((1024, 128), differentiation="e2arg05", dtype=torch.float16)
        idx = torch.randint(0, 1024, (64,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)


# ---------------------------------------------------------------------------
# Group 3 — Explicit 4-arg (3-keyword) STL annotation
# ---------------------------------------------------------------------------


@_requires_stl
class TestGatherExplicitStl4Arg:
    """Gather source pre-annotated with the full 3-keyword STL constructor.

    ``SpyreTensorLayout(device_size=..., stride_map=..., device_dtype=...)``
    built by ``_stl_generic`` encodes the canonical generic-stick layout
    explicitly.  The device tensor is pre-built via
    ``.to(DEVICE, device_layout=stl)`` so the compiler observes it at
    graph-input time.  ``_compare_explicit_stl`` supplies the CPU reference.

    The final section uses ``_assert_explicit_eq_implicit`` to verify that
    explicit-STL and implicit compiled outputs are bit-identical.

    Eager fails for all tests with aten::index.Tensor_out not registered (or the
    int32→int64 type conversion that precedes it); xfail'd via _INDEX_EAGER.

    SENCORES and execution_mode come from conftest pytest_generate_tests +
    patch_sencores.
    """

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    def _to_dev(self, x_cpu):
        """Move *x_cpu* to Spyre using the full 3-keyword STL constructor."""
        return x_cpu.to(DEVICE, device_layout=_stl_generic(x_cpu.shape, x_cpu.dtype))

    def _run(self, fn, x_cpu, idx_cpu, execution_mode, atol, rtol):
        """Build device tensors and delegate to ``_compare_explicit_stl``."""
        _compare_explicit_stl(
            fn,
            x_cpu,
            idx_cpu,
            self._to_dev(x_cpu),
            idx_cpu.to(DEVICE),
            execution_mode,
            atol,
            rtol,
        )

    # ------------------------------------------------------------------
    # Shape / dtype variety

    def test_4arg_2d_float16_int64(self, execution_mode):
        """(36,256) float16 + int64 index; compiled path reads annotated layout."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((36, 256), differentiation="e4arg01", dtype=torch.float16)
        idx = torch.randint(0, 36, (18,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    def test_4arg_2d_float16_int32(self, execution_mode):
        """(48,128) float16 + int32 index; hardware-native width, no double-cast."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((48, 128), differentiation="e4arg01i32", dtype=torch.float16)
        idx = torch.randint(0, 48, (16,), dtype=torch.int32)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    def test_4arg_kv_exact_paged(self, execution_mode):
        """Canonical paged-attention STL: (512,32,128) cache, gather 12 rows."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((512, 32, 128), differentiation="e4arg02", dtype=torch.float16)
        idx = torch.randint(0, 512, (12,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    # ------------------------------------------------------------------
    # GQA / MQA KV cache shapes

    def test_4arg_gqa_h8_d128(self, execution_mode):
        """GQA KV cache (1024,8,128) with 4-arg STL."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((1024, 8, 128), differentiation="e4arg04", dtype=torch.float16)
        idx = torch.randint(0, 1024, (4 * 64,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    def test_4arg_mqa_h1(self, execution_mode):
        """MQA KV cache (1024,1,64) with 4-arg STL."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((1024, 1, 64), differentiation="e4arg05", dtype=torch.float16)
        idx = torch.randint(0, 1024, (4 * 64,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_F16, _ATOL_F16)

    def test_4arg_bfloat16(self, execution_mode):
        """4-arg STL for bfloat16 (128,4,64) KV cache."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((128, 4, 64), differentiation="e4arg06", dtype=torch.bfloat16)
        idx = torch.randint(0, 128, (32,), dtype=torch.int64)
        self._run(lambda x, i: x[i], x, idx, execution_mode, _ATOL_BF16, _ATOL_BF16)

    # ------------------------------------------------------------------
    # Downstream ops after STL-annotated gather

    def test_4arg_with_downstream_softmax(self, execution_mode):
        """4-arg STL + softmax(dim=-1) after KV gather; (512,8,64) stick-aligned shape."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((512, 8, 64), differentiation="e4arg07", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        self._run(
            lambda x, i: torch.softmax(x[i].float(), dim=-1).half(),
            x,
            idx,
            execution_mode,
            _ATOL_F16,
            _ATOL_F16,
        )

    def test_4arg_with_named_dims(self, execution_mode):
        """4-arg STL + named dims (cache,H,D) on (128,8,64) KV tensor."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x_cpu = cached_randn(
            (128, 8, 64), differentiation="e4arg08", dtype=torch.float16
        )
        idx_cpu = torch.randint(0, 128, (16,), dtype=torch.int64)
        x_dev = self._to_dev(x_cpu)
        idx_dev = idx_cpu.to(DEVICE)
        _pnd.declare_tensor_dim("cache", 128)
        _pnd.declare_tensor_dim("H", 8)
        _pnd.declare_tensor_dim("D", 64)
        _pnd.declare_tensor_dim("slots", 16)
        _pnd.name_tensor_dims(x_dev, ["cache", "H", "D"])
        _pnd.name_tensor_dims(idx_dev, ["slots"])
        _compare_explicit_stl(
            lambda x, i: x[i],
            x_cpu,
            idx_cpu,
            x_dev,
            idx_dev,
            execution_mode,
            _ATOL_F16,
            _ATOL_F16,
        )

    # ------------------------------------------------------------------
    # Numeric equivalence: explicit 4-arg STL == implicit layout (compiled only)
    #
    # ``_assert_explicit_eq_implicit`` compiles ``x[i]`` twice — once with the
    # 4-arg annotated tensor, once with plain ``.to(DEVICE)`` — and asserts
    # bit-identical output.  Compiled-only: the annotation is a compile-time
    # concept not observable in eager mode.

    def test_explicit_vs_implicit_2d(self):
        """Explicit 4-arg STL == implicit layout, 2D float16."""
        _assert_explicit_eq_implicit(
            cached_randn((32, 256), differentiation="cmp01", dtype=torch.float16),
            torch.randint(0, 32, (16,), dtype=torch.int64),
        )

    def test_explicit_vs_implicit_kv(self):
        """Explicit 4-arg STL == implicit layout, KV 3D float16."""
        _assert_explicit_eq_implicit(
            cached_randn((64, 4, 32), differentiation="cmp03", dtype=torch.float16),
            torch.randint(0, 64, (16,), dtype=torch.int64),
        )

    def test_explicit_vs_implicit_embedding(self):
        """Explicit 4-arg STL == implicit layout, embedding 2D float16."""
        _assert_explicit_eq_implicit(
            cached_randn((512, 64), differentiation="cmp04", dtype=torch.float16),
            torch.randint(0, 512, (32,), dtype=torch.int64),
        )

    def test_explicit_vs_implicit_bfloat16(self):
        """Explicit 4-arg STL == implicit layout, 2D bfloat16."""
        _assert_explicit_eq_implicit(
            cached_randn((16, 192), differentiation="cmp05", dtype=torch.bfloat16),
            torch.randint(0, 16, (8,), dtype=torch.int64),
        )

    def test_explicit_vs_implicit_moe(self):
        """Explicit 4-arg STL == implicit layout, MoE 3D float16."""
        _assert_explicit_eq_implicit(
            cached_randn((8, 512, 64), differentiation="cmp09", dtype=torch.float16),
            torch.randint(0, 8, (32,), dtype=torch.int64),
        )
