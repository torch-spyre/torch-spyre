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

"""Cross-check our SPYRE_PROFILE_SYNC min-of-N against the PyTorch profiler.

The torch.profiler ``PrivateUse1`` activity (see
docs/source/user_guide/profiling/pytorch_profiler.md) reports a **"Self SPYRE"**
column = the TRUE per-kernel device time. Two uses:

1. **Validate our timer.** Our SPYRE_PROFILE_SYNC min brackets the launch+sync, so it
   = device time + ~7 us host residue. Run the SAME op/size here and in bench_ops.py:
   ``our_min  ~=  Self_SPYRE(sdsc_fused_*)  +  ~7us`` confirms the measurement.
2. **Decompose the ~20 us fixed term.** The profiler surfaces a separate
   ``Memset (Device)`` event (~12.5 us on a tiny add). If it stays ~constant across
   sizes, a big chunk of our "fixed" is a real DEVICE memset, not host overhead.

NOTE: this is a TIME + memory-allocation profiler -- it has NO DRAM bandwidth / read-
vs-write / bus-utilization counters, so it CANNOT explain why read+write halves
bandwidth (that needs aiu-smi). It only gives cleaner device time.

Knobs: BENCH_OP, BENCH_ROWS, BENCH_COLS, BENCH_WARMUP. BENCH_EMIT_RECORDS=1 adds the
machine-readable IO/MODEL/FEATS block the sweep folds into the cost-model database;
it is off by default because it restates the cost-model dump. BENCH_OP is any of the
bench_ops/bench_bandwidth ops: neg copy gelu relu sigmoid exp | mul add | add3 add4 |
read sumrow sumall amax mean | bcast mulbcast | write.

Examples:
    # one op (prints the table + a parseable SUMMARY line with kernel_us / memset_us)
    BENCH_OP=neg BENCH_COLS=1024 \
        python docs/source/user_guide/examples/profile_ops.py
    # full golden re-sweep (rebuild the model from kernel time):
    bash docs/source/user_guide/examples/run_profile_sweep.sh
"""

import contextlib
import logging
import os
import sys
import statistics
import tempfile
import time

# Enable the cost-model dump so we read ITS device-layout I/O size (the same byte
# accounting the model uses), and force compile so the dump fires.
os.environ.setdefault("SPYRE_DUMP_COST", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from torch_spyre._inductor import cost_model, dump_cost_model  # noqa: E402

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_OP", "gelu")
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "16384"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
# Noise protocol: take BENCH_REPS back-to-back PROFILED measurements (not one) and report
# min/median/mean/std/cv of kernel_us, so a jittery point is visible (cv) instead of read
# as a model error. The profiled kernel is a small fraction of the run and the profiler
# adds at most ~3x, so extra reps are nearly free. kernel_us in the SUMMARY = the median.
REPS = max(1, int(os.environ.get("BENCH_REPS", "7")))
SENCORES = os.environ.get("SENCORES", "")  # core count (read only to tag the SUMMARY)
TILES = int(os.environ.get("BENCH_TILES", "0"))  # coarse-tile dim0 into K (>=2 on)
LX = os.environ.get("LX_PLANNING", "1")  # scratchpad planning on(1)/off(0); SUMMARY tag
NCOLS = int(os.environ.get("BENCH_N", str(COLS)))  # matmul N dim (M=ROWS, K=COLS, N)
BB = int(os.environ.get("BENCH_B", "8"))  # batch dim for bmm ops (a[B,M,K] @ b[B,K,N])
STAGES = int(
    os.environ.get("BENCH_STAGES", "0")
)  # extra LX-resident pointwise stages for `softmax_stages` (see that op)
# The IO/MODEL/FEATS blocks are the machine-readable record the sweep folds into the
# database (tools/cost_model/parse_sweep_logs.py). They restate what the cost-model dump
# already printed, so for a human running one op they are pure noise -- emitted only when
# the sweep asks for them.
EMIT_RECORDS = os.environ.get("BENCH_EMIT_RECORDS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _emit(*args, **kwargs) -> None:
    """Print a record line, but only when the sweep is collecting records."""
    if EMIT_RECORDS:
        print(*args, **kwargs)


def _banner(title: str) -> str:
    """A boxed section header, matching the cost-model dumps' own banners."""
    bar = "=" * 78
    return f"{bar}\n==== {title}\n{bar}"


#: Lines the device stack writes on every profiler session and which say nothing about
#: the measurement. ``SyncActivityProfilerHandler`` logs a start and a stop per session,
#: so a 7-rep run emits 14 of them; kineto adds one fixed complaint about a ``[memory]``
#: event it does not recognise. Both come from the out-of-tree runtime, not this repo,
#: so there is nothing to silence at the source.
_CHATTER = (
    "SyncActivityProfilerHandler.cpp",
    "is not present in the set of known events",
)


@contextlib.contextmanager
def _quiet_device_chatter():
    """Drop the runtime's per-session profiler chatter from fd 1 and 2.

    The lines are written by C++ on the real file descriptors, so a Python-level
    redirect cannot see them; this swaps the descriptors for a temporary file and
    replays the capture afterwards.

    ONLY the known-noise lines are dropped -- everything else is written straight back
    out. A filter that swallowed an unexpected error to tidy the output would cost far
    more than the noise it removed, and this wraps the part of the run where a device
    failure is most likely to be reported.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as cap:
        try:
            os.dup2(cap.fileno(), 1)
            os.dup2(cap.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)
            cap.seek(0)
            text = cap.read().decode("utf-8", "replace")
            kept = [
                ln for ln in text.splitlines() if not any(c in ln for c in _CHATTER)
            ]
            if any(ln.strip() for ln in kept):
                sys.stderr.write("\n".join(kept) + "\n")
                sys.stderr.flush()


TO_MID = int(
    os.environ.get("TO_MID", "8")
)  # transpose_outer middle (outer-swap) dim M:
# [R,M,C]->[M,R,C]. Swept to isolate whether the outer-scatter count M drives effBW (it is
# fixed at 8 in the R×C grid; the block-transpose vs strided-gather mechanism hinges on it).
WD_B = os.environ.get("WD_B")  # forced work-div split (spyre_hint work_div). cores =
WD_M = os.environ.get("WD_M")  # WD_B*WD_M*WD_N*WD_K. Unset dim -> stays 1 (the hint is
WD_N = os.environ.get("WD_N")  # FINAL, not floor-filled). WD_M/N/K used by `mmwd`; all
WD_K = os.environ.get("WD_K")  # four (incl. WD_B) used by `bmm_wd`/`bmm_wd_3d2d`.
# Device tensor-layout dim_order for the two bmm operands (op `bmm_layout`), e.g. "1,0,2".
# The LAST element is the stick dim -- keep it = the original last axis (K for arg A, N for
# arg B) so the compiler does NOT insert a restickify/clone (which would change the counted
# bytes and confound the layout signal). "1,0,2" only reorders the two OUTER axes.
WD_LAYOUT_A = os.environ.get("WD_LAYOUT_A")  # dim_order for A [B,M,K]
WD_LAYOUT_B = os.environ.get("WD_LAYOUT_B")  # dim_order for B [B,K,N]

torch.manual_seed(0xAFFE)


def _rand(*shape):
    return torch.rand(*shape, dtype=torch.float16).to(DEVICE)


def _sum_all(*ts):
    acc = ts[0]
    for t in ts[1:]:
        acc = acc + t
    return acc


# Same ops as bench_ops/bench_bandwidth so the GOLDEN kernel times map 1:1.
_UNARY = {  # 1 read + 1 write (gelu/relu/sigmoid/exp also probe arithmetic-free)
    "neg": lambda x: -x,  # cleanest balanced 1R+1W (no constant)
    "copy": lambda x: x + 1.0,  # scalar 1.0 is a cached broadcast -> still 1R+1W
    "gelu": F.gelu,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
}
_BINARY = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}  # 2R + 1W
_NARY = {
    "add3": 3,
    "add4": 4,
    "add5": 5,
    "add6": 6,
    "add8": 8,
}  # n inputs summed (dependent chain)
_REDUCE = {  # read-dominated; sumall reduces to a scalar -> ring combine
    "read": lambda x: x.sum(dim=-1),
    "sumrow": lambda x: x.sum(dim=-1),  # reduce COLS -> [ROWS] (within-stick axis)
    "sumcol": lambda x: x.sum(dim=0),  # reduce ROWS -> [COLS] (the other axis)
    "sumall": lambda x: x.sum(),  # -> scalar: reduced axis split across all cores
    "amax": lambda x: x.amax(dim=-1),
    "mean": lambda x: x.mean(dim=-1),
}
_BCAST = {"bcast": lambda a, b: a + b, "mulbcast": lambda a, b: a * b}  # b = [1, COLS]
# Data-movement ("transport") ops -- a RESTICKIFY / copy that moves the SAME bytes as
# neg/copy but reorganizes the stick layout (scattered access). Access patterns differ:
#   transpose : [R,C]->[C,R], the stick dim MOVES (heavy intra-stick reshuffle)
#   cat0/cat1 : concat 2 copies along rows (non-stick) / cols (stick dim)
# Compare vs `neg` (contiguous 1R+1W) via effective BW to read the access-pattern cost.
# (`transpose_outer` -- a 3D stick-PRESERVING outer-dim swap -- is a separate branch.)
_TRANSPORT = {
    "transpose": lambda x: x.transpose(0, 1).contiguous(),  # [R,C]->[C,R]: stick C->R
    "cat0": lambda x: torch.cat([x, x], dim=0),  # append rows (non-stick dim)
    "cat1": lambda x: torch.cat([x, x], dim=1),  # append cols (stick dim -> interleave)
}
# Coarse-tiled dim0 reductions (mirror the coarse-tiling harness{sum,amax,amin}_dim0_tiled.py):
# reduce a [B=ROWS, D=COLS] tensor over B, tiling B into BENCH_TILES chunks. With
# BENCH_TILES>=2 a spyre_hint wraps it in a K-iteration loop (fill + K x reduce/combine)
# BENCH_TILES<=1 is the plain untiled baseline. Run each under LX_PLANNING=0/1.
_CT_REDUCE = {"ctsum": "sum", "ctamax": "amax", "ctamin": "amin"}


# Per-call setup hook: the coarse-tile path sets this to re-declare its named dims
# before every traced invocation (the example declares once + compiles once; this
# profiling harness traces repeatedly). None for all non-tiled ops.
_PREPARE = None


def _ct_workload(rtype: str):
    """Coarse-tiled dim0 reduction, mirroring the coarse-tiling harness*_dim0_tiled.py + utils.py
    ``_compile_and_run``: a CPU input (sum scaled 0.1 as the example does), eager
    ``declare_tensor_dim``, ``name_tensor_dims`` + ``spyre_hint`` inside fn, an eager
    reference call of fn, then a dynamo/Fx cache reset right before compile.
    """
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    b, d = ROWS, COLS
    scale = 0.1 if rtype == "sum" else 1.0  # example scales sum to avoid fp16 growth
    x_cpu = torch.randn(b, d, dtype=torch.float16) * scale

    def _declare():
        pnd.declare_tensor_dim("B", b)
        pnd.declare_tensor_dim("D", d)

    _declare()  # eager, as the example does in main() before compiling

    def reduce_fn(x):
        return getattr(x, rtype)(dim=0)

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": TILES}):
                return reduce_fn(x)

        # Re-declare EAGERLY (not inside fn -> no trace perturbation) before each traced
        # call so propagate_named_dims always resolves the named-dim sizes.
        _PREPARE = _declare
    else:
        fn = reduce_fn

    fn(x_cpu)  # eager reference call (mirror compare_with_cpu's cpu_result = fn(...))
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _softmax_row_tiling():
    """Softmax over dim=-1, row-tiled over NROW (
    tiling.py + the new_coarse_tiling IR): the 5 softmax ops (max, sub, exp, sum, div)
    fuse into ONE NROW-tiled loop with the intermediates LX-resident. BENCH_TILES>=2
    tiles NROW into that many chunks; else untiled. Measured via the harness AIU
    profiler (torch.profiler "Self SPYRE"), NOT the SPYRE_PROFILE host-launch path."""
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    nrow, ncol = ROWS, COLS
    x_cpu = torch.randn(nrow, ncol, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("NROW", nrow)
        pnd.declare_tensor_dim("NCOL", ncol)

    _declare()

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["NROW", "NCOL"])
            with spyre_hint(num_tiles_per_dim={"NROW": TILES}):
                return torch.softmax(x, dim=-1)

        _PREPARE = _declare
    else:

        def fn(x):
            return torch.softmax(x, dim=-1)

    fn(x_cpu)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _softmax_stages():
    """Softmax with a CONTROLLED number of extra fused stages.

    The fused-kernel floor is ``elements / (cores * rate)``, keyed on the ELEMENT COUNT and
    independent of how many stages the fused chain has. An LX-bandwidth explanation fits the
    same calibration data equally well (whole-loop LX traffic is as tile-count-invariant as
    the element count, and LX is per-core so it scales with cores identically), but it
    predicts something different HERE: LX traffic is proportional to the number of stages.

    Each extra stage reads its input from LX and writes its output to LX, so across
    ``BENCH_STAGES`` the element count and the HBM traffic are unchanged while LX traffic
    grows linearly. Run at a LOW core count, where the floor actually binds:

      * time flat in BENCH_STAGES   -> the element-only form of the floor is right
      * time grows with BENCH_STAGES -> the floor is MIS-KEYED and must scale with the
        chain length. That does not by itself name the mechanism: LX traffic and
        per-element work through more stages are both proportional to
        ``elements * stages``, and this test does not separate them.

    ``sigmoid`` is used rather than a scalar multiply on purpose: repeated ``y * c`` folds
    to a single multiply, which would silently collapse every stage count to one kernel.
    CHECK THE CONTROL when reading results -- if HBM bytes move with BENCH_STAGES the
    intermediates have spilled and the comparison is confounded.
    """
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    nrow, ncol = ROWS, COLS
    x_cpu = torch.randn(nrow, ncol, dtype=torch.float16)
    extra = max(0, STAGES)

    def _declare():
        pnd.declare_tensor_dim("NROW", nrow)
        pnd.declare_tensor_dim("NCOL", ncol)

    _declare()

    def _chain(x):
        m = torch.amax(x, dim=-1, keepdim=True)
        y = x - m
        for _ in range(extra):
            y = torch.sigmoid(y)  # LX -> LX, not algebraically foldable
        e = torch.exp(y)
        s = torch.sum(e, dim=-1, keepdim=True)
        return e / s

    # COARSE TILING IS REQUIRED, not optional. Untiled, every intermediate is the FULL
    # tensor (16.8 MB at 4096x2048) and cannot be LX-resident, so each added stage does a
    # full HBM round trip instead -- which moves the HBM traffic and destroys the control.
    # Tiling is what makes the intermediates per-tile and LX-resident in the first place.
    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["NROW", "NCOL"])
            with spyre_hint(num_tiles_per_dim={"NROW": TILES}):
                return _chain(x)

        _PREPARE = _declare
    else:
        fn = _chain

    fn(x_cpu)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _mm_workload():
    """Matmul ``a @ b`` with a FORCED work-division split (spyre_hint work_div), so the
    (m, n, k) core split is controlled instead of planner-chosen -- for term isolation
    (compute / hbm / psum). M=ROWS, K=COLS, N=BENCH_N; WD_M/WD_N/WD_K give the per-dim
    split (cores = product, FINAL). Mirrors tests/inductor/test_work_division_hint.py:
    eager declare_tensor_dim, name the inputs' dims, hint inside fn; the coarse-tile
    _PREPARE hook re-declares before each traced call.
    """
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    m, k, n = ROWS, COLS, NCOLS
    wd = {
        nm: int(os.environ[ev])
        for nm, ev in (("M", "WD_M"), ("N", "WD_N"), ("K", "WD_K"))
        if os.environ.get(ev)
    }
    xa = torch.rand(m, k, dtype=torch.float16)  # CPU; timing only, values irrelevant
    yb = torch.rand(k, n, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("M", m)
        pnd.declare_tensor_dim("K", k)
        pnd.declare_tensor_dim("N", n)

    _declare()

    def fn(a, b):
        pnd.name_tensor_dims(a, ["M", "K"])
        pnd.name_tensor_dims(b, ["K", "N"])
        with spyre_hint(work_div=wd):
            return a @ b

    _PREPARE = _declare
    fn(xa, yb)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (xa.to(DEVICE), yb.to(DEVICE))


def _softmax_noexp_row_tiling():
    """MATCHED CONTROL for the softmax double-count/exp test: identical NROW-tiled fused
    structure as ``_softmax_row_tiling`` -- 2 reductions (amax, sum) + 3 pointwise (sub,
    MUL, div) -- but with the transcendental ``exp`` REPLACED by a cheap ``mul``. Same
    tiling, same fusion depth, same LX-resident intermediates; the ONLY difference is exp.
    So (softmax_time - softmax_noexp_time) at matched [ROWS,COLS,TILES] isolates the exp
    cost -- the design-review-mandated control that an untiled copy cannot provide."""
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    nrow, ncol = ROWS, COLS
    x_cpu = torch.randn(nrow, ncol, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("NROW", nrow)
        pnd.declare_tensor_dim("NCOL", ncol)

    _declare()

    def _sm_noexp(x):  # amax, sub, mul (<- exp), sum, div : softmax minus the exp
        y = x - x.amax(dim=-1, keepdim=True)
        z = y * 0.5  # cheap pointwise stand-in for exp (same pipeline stage, no exp)
        return z / z.sum(dim=-1, keepdim=True)

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["NROW", "NCOL"])
            with spyre_hint(num_tiles_per_dim={"NROW": TILES}):
                return _sm_noexp(x)

        _PREPARE = _declare
    else:

        def fn(x):
            return _sm_noexp(x)

    fn(x_cpu)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _matmul_row_tiling():
    """Matmul ``a @ b`` COARSE-TILED over the M (row) dim via
    spyre_hint(num_tiles_per_dim={"M": TILES}) DISTINCT from `mmwd`, which forces a
    work-division CORE split: this creates a sequential M-tile LOOP. M=ROWS, K=COLS,
    N=BENCH_N; BENCH_TILES>=2 tiles M, else untiled. Measured via the AIU profiler."""
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    m, k, n = ROWS, COLS, NCOLS
    xa = torch.rand(m, k, dtype=torch.float16)  # CPU; timing only, values irrelevant
    yb = torch.rand(k, n, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("M", m)
        pnd.declare_tensor_dim("K", k)
        pnd.declare_tensor_dim("N", n)

    _declare()

    if TILES >= 2:

        def fn(a, b):
            pnd.name_tensor_dims(a, ["M", "K"])
            pnd.name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"M": TILES}):
                return a @ b

        _PREPARE = _declare
    else:

        def fn(a, b):
            return a @ b

    fn(xa, yb)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (xa.to(DEVICE), yb.to(DEVICE))


def _matmul_k_tiling():
    """`a @ b` [M,K]@[K,N] COARSE-TILED over the K (reduction) dim via
    spyre_hint(num_tiles_per_dim={"K": TILES}) and run_mm_k_tiled.py. M=ROWS, K=COLS, N=BENCH_N; TILES>=2 tiles K, else untiled."""
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    m, k, n = ROWS, COLS, NCOLS
    xa = torch.rand(m, k, dtype=torch.float16)
    yb = torch.rand(k, n, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("M", m)
        pnd.declare_tensor_dim("K", k)
        pnd.declare_tensor_dim("N", n)

    _declare()

    if TILES >= 2:

        def fn(a, b):
            pnd.name_tensor_dims(a, ["M", "K"])
            pnd.name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"K": TILES}):
                return a @ b

        _PREPARE = _declare
    else:

        def fn(a, b):
            return a @ b

    fn(xa, yb)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (xa.to(DEVICE), yb.to(DEVICE))


def _mm_nested_m_k():
    """torch.mm [M,K]@[K,N] with NESTED tiling -- outer M x2, inner K x TILES M=ROWS, K=COLS, N=BENCH_N."""
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    m, k, n = ROWS, COLS, NCOLS
    xa = torch.rand(m, k, dtype=torch.float16)
    yb = torch.rand(k, n, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("M", m)
        pnd.declare_tensor_dim("K", k)
        pnd.declare_tensor_dim("N", n)

    _declare()

    if TILES >= 2:

        def fn(a, b):
            pnd.name_tensor_dims(a, ["M", "K"])
            pnd.name_tensor_dims(b, ["K", "N"])
            with spyre_hint(num_tiles_per_dim={"M": 2}):
                with spyre_hint(num_tiles_per_dim={"K": TILES}):
                    return torch.mm(a, b)

        _PREPARE = _declare
    else:

        def fn(a, b):
            return torch.mm(a, b)

    fn(xa, yb)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (xa.to(DEVICE), yb.to(DEVICE))


def _to_dev(t: torch.Tensor, layout_spec):
    """Place a CPU tensor on the device, optionally with a chosen ``dim_order`` device
    layout (op ``bmm_layout``). ``layout_spec`` is a comma-separated order like ``"1,0,2"``
    (or None for the default placement). The LAST element must be the tensor's original
    last axis so the compiler inserts NO restickify/clone (verified by the sweep's IR
    grep). Lazy-init caveat: the very first ``.to(DEVICE)`` in the process must be plain,
    so we do a tiny plain ``.to`` first to initialize before any ``device_layout`` copy."""
    if not layout_spec:
        return t.to(DEVICE)
    from torch_spyre._C import SpyreTensorLayout

    order = [int(x) for x in layout_spec.split(",")]
    torch.zeros(1, dtype=t.dtype).to(DEVICE)  # plain to() first (fragile lazy init)
    stl = SpyreTensorLayout(list(t.size()), list(t.stride()), t.dtype, order)
    return t.to(DEVICE, device_layout=stl)


def _bmm_workload(kind: str):
    """Batched matmul *.py. B=BENCH_B, M=ROWS, K=COLS,
    N=BENCH_N; TILES>=2 tiles K (else untiled). ``kind``:
      "k"      -> torch.bmm(a[B,M,K], b[B,K,N])       tiled over K
      "3d2d"   -> torch.matmul(a[B,M,K], b[K,N])      2-D weight shared over the batch
      "nested" -> torch.bmm(a[B,M,K], b[B,K,N])       outer B x2, inner K x TILES

    If any of WD_B/WD_M/WD_N/WD_K is set, a FORCED work-division split
    (spyre_hint work_div) is applied instead of coarse tiling -- the `bmm_wd` /
    `bmm_wd_3d2d` ops, mirroring `mmwd` but for a batched output.
    """
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    b_n, m, k, n = BB, ROWS, COLS, NCOLS
    xa = torch.rand(b_n, m, k, dtype=torch.float16)
    yb = (
        torch.rand(k, n, dtype=torch.float16)
        if kind == "3d2d"
        else torch.rand(b_n, k, n, dtype=torch.float16)
    )
    mm = torch.matmul if kind == "3d2d" else torch.bmm

    def _declare():
        pnd.declare_tensor_dim("B", b_n)
        pnd.declare_tensor_dim("M", m)
        pnd.declare_tensor_dim("K", k)
        pnd.declare_tensor_dim("N", n)

    _declare()

    def _name(a, b):
        pnd.name_tensor_dims(a, ["B", "M", "K"])
        pnd.name_tensor_dims(b, ["K", "N"] if kind == "3d2d" else ["B", "K", "N"])

    # FORCED-split path (bmm_wd): any WD_* set -> hint work_div instead of coarse tiling.
    # NB: must call _name(a,b) here -- the untiled `else` branch below does NOT name dims,
    # and work_div_loop_info (which the cost-model decode reads) is populated only for
    # NAMED, work_div-hinted ops.
    wd = {
        nm: int(os.environ[ev])
        for nm, ev in (("B", "WD_B"), ("M", "WD_M"), ("N", "WD_N"), ("K", "WD_K"))
        if os.environ.get(ev)
    }
    if wd:

        def fn(a, b):
            _name(a, b)
            with spyre_hint(work_div=wd):
                return mm(a, b)

        _PREPARE = _declare
    elif TILES >= 2 and kind == "nested":

        def fn(a, b):
            _name(a, b)
            with spyre_hint(num_tiles_per_dim={"B": 2}):
                with spyre_hint(num_tiles_per_dim={"K": TILES}):
                    return mm(a, b)

        _PREPARE = _declare
    elif TILES >= 2:

        def fn(a, b):
            _name(a, b)
            with spyre_hint(num_tiles_per_dim={"K": TILES}):
                return mm(a, b)

        _PREPARE = _declare
    else:

        def fn(a, b):
            return mm(a, b)

    fn(xa, yb)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    # Operand placement: default `.to(DEVICE)`, or a chosen device dim_order when
    # WD_LAYOUT_A/B is set (op `bmm_layout`) -- see `_to_dev`.
    return torch.compile(fn), (_to_dev(xa, WD_LAYOUT_A), _to_dev(yb, WD_LAYOUT_B))


def _softmax_unrolled():
    """Manual softmax chain (amax/sub/exp/sum/div) over [B,D]=[ROWS,COLS], tiled over B
    with the tile loop UNROLLED (config.unroll_loops=True, sencores=1) The unrolled IR has NO CoarseTileInfo (tiling
    shows only as dim_hints), so it exercises the extractor's non-loop coarse path."""
    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import config, spyre_hint

    global _PREPARE
    b_n, d = ROWS, COLS
    x_cpu = torch.randn(b_n, d, dtype=torch.float16)

    def _declare():
        pnd.declare_tensor_dim("B", b_n)
        pnd.declare_tensor_dim("D", d)

    _declare()

    def _softmax(x):
        mx = x.amax(dim=-1, keepdim=True)
        e = (x - mx).exp()
        return e / e.sum(dim=-1, keepdim=True)

    if TILES >= 2:

        def fn(x):
            pnd.name_tensor_dims(x, ["B", "D"])
            with spyre_hint(num_tiles_per_dim={"B": TILES}):
                return _softmax(x)

        _PREPARE = _declare
    else:  # TILES<=1: the UNTILED single-core reference (no B-tiling)
        fn = _softmax

    # Match the example's config: unroll the tile loop, single core, LX planning on.
    config.unroll_loops = True
    config.sencores = 1
    config.lx_planning = True
    config.allow_all_ops_in_lx_planning = True

    fn(x_cpu)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    return torch.compile(fn), (x_cpu.to(DEVICE),)


def _flash_attn_workload():
    """Spyre flash-attention with the coarse-tiling and work-division HINTS made
    env-configurable, so a sweep can vary the block sizes and read the measured time for
    each. A MULTI-OP coarse-tiled program (~28 ops): two batched matmuls (QK^T reduces
    over D, PV reduces over Lk) glued by the online-softmax reductions, fused under one
    tiled loop nest.
    Knobs: FA_B/FA_H/FA_LQ/FA_LK/FA_D (shape); FA_B_TILES/FA_H_TILES/FA_LQ_TILES/
    FA_LK_TILES (coarse tile COUNTS per dim -> loop nest); FA_WD ("H:4,Lq:8,Lk:8" work_div
    -> intra-tile cores). We are NOT modeling flash attn yet; this is data capture: the
    model over-predicts it by more than 10x, so it is excluded from every scored figure."""
    import math as _math

    import torch_spyre._inductor.wsr.propagate_named_dims as pnd
    from torch._inductor.codecache import FxGraphCache
    from torch_spyre._inductor import spyre_hint

    global _PREPARE
    B = int(os.environ.get("FA_B", "1"))
    H = int(os.environ.get("FA_H", "32"))
    Lq = int(os.environ.get("FA_LQ", "4096"))
    Lk = int(os.environ.get("FA_LK", "4096"))
    D = int(os.environ.get("FA_D", "128"))
    bt = int(os.environ.get("FA_B_TILES", "1"))
    ht = int(os.environ.get("FA_H_TILES", "8"))
    qt = int(os.environ.get("FA_LQ_TILES", "4"))
    kt = int(os.environ.get("FA_LK_TILES", "1"))
    wd = {}
    for part in os.environ.get("FA_WD", "H:4,Lq:8,Lk:8").split(","):
        nm, val = part.split(":")
        wd[nm.strip()] = int(val)
    scale = 1.0 / _math.sqrt(_math.sqrt(D))

    def _declare():
        pnd.declare_tensor_dim("B", B)
        pnd.declare_tensor_dim("H", H)
        pnd.declare_tensor_dim("Lq", Lq)
        pnd.declare_tensor_dim("Lk", Lk)
        pnd.declare_tensor_dim("D", D)

    _declare()

    def _name(q, k, v, m):
        pnd.name_tensor_dims(q, ["B", "H", "Lq", "D"])
        pnd.name_tensor_dims(k, ["B", "H", "Lk", "D"])
        pnd.name_tensor_dims(v, ["B", "H", "Lk", "D"])
        pnd.name_tensor_dims(m, ["B", "H", "Lq", "Lk"])

    def flash(queries, keys, values, mask):
        _name(queries, keys, values, mask)
        output = torch.zeros_like(queries)
        # sparse running max / denominator (the example's reduction hack)
        real_max = torch.full(
            (B, H, Lq, 64), float("-inf"), device=queries.device, dtype=torch.float16
        ).amax(dim=-1)
        denominator = torch.zeros(
            (B, H, Lq, 64), device=queries.device, dtype=torch.float16
        ).amax(dim=-1)
        with spyre_hint(tiles={"B": bt}):
            with spyre_hint(tiles={"H": ht}):
                with spyre_hint(tiles={"Lq": qt}):
                    with spyre_hint(tiles={"Lk": kt}):
                        with spyre_hint(work_div=wd):
                            keys_t = (keys * scale).transpose(-1, -2)
                            scores = torch.matmul(queries * scale, keys_t) + mask
                            block_max = torch.amax(scores, dim=-1)
                            running_max = torch.maximum(real_max, block_max)
                            exp_scores = torch.exp(scores - running_max.unsqueeze(-1))
                            correction = torch.exp(real_max - running_max)
                            denominator.copy_(
                                denominator * correction + exp_scores.sum(dim=-1)
                            )
                            output.copy_(
                                output * correction.unsqueeze(-1)
                                + torch.matmul(exp_scores, values)
                            )
                            real_max.copy_(running_max)
        return output / denominator.unsqueeze(-1)

    q = torch.randn(B, H, Lq, D, dtype=torch.float16)
    k = torch.randn(B, H, Lk, D, dtype=torch.float16)
    v = torch.randn(B, H, Lk, D, dtype=torch.float16)
    causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
    m = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
    m.masked_fill_(~causal, float("-inf"))
    flash(q, k, v, m)  # eager reference call
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    _PREPARE = _declare
    return (
        torch.compile(flash),
        (q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), m.to(DEVICE)),
    )


def make_workload():
    if OP in _UNARY:
        return torch.compile(_UNARY[OP]), (_rand(ROWS, COLS),)
    if OP in _BINARY:
        return torch.compile(_BINARY[OP]), (_rand(ROWS, COLS), _rand(ROWS, COLS))
    if OP in _NARY:
        xs = tuple(_rand(ROWS, COLS) for _ in range(_NARY[OP]))
        return torch.compile(_sum_all), xs
    if OP in _REDUCE:
        return torch.compile(_REDUCE[OP]), (_rand(ROWS, COLS),)
    if OP in _BCAST:  # row-vector broadcast: a[R,C] + b[1,C] (b cached across rows)
        return torch.compile(_BCAST[OP]), (_rand(ROWS, COLS), _rand(1, COLS))
    if OP in _TRANSPORT:  # data movement (restickify): same bytes as a copy, scattered
        return torch.compile(_TRANSPORT[OP]), (_rand(ROWS, COLS),)
    if OP == "transpose_outer":  # 3D [R,M,C]: swap OUTER dims, stick (last dim C) kept
        tp = lambda x: x.transpose(0, 1).contiguous()  # noqa: E731
        return torch.compile(tp), (_rand(ROWS, TO_MID, COLS),)
    if OP == "bcastcol":  # col-vector broadcast: a[R,C] + b[R,1] (b cached across cols)
        return torch.compile(lambda a, b: a + b), (_rand(ROWS, COLS), _rand(ROWS, 1))
    if OP == "write":  # write-only: both inputs broadcast -> cached
        return torch.compile(lambda b, c: b + c), (_rand(1, COLS), _rand(ROWS, 1))
    if OP == "add_indep2":  # control: two INDEPENDENT adds, same 4R:2W as add3 but NO
        # read-after-write dependency (op0=a+b, op1=c+d). add3 - add_indep2 isolates the
        # dependent round-trip cost from the byte count. (Verify in the IR whether Inductor
        # fuses the two into one kernel or emits two; both readings are informative.)
        indep = lambda a, b, c, d: (a + b, c + d)  # noqa: E731
        return torch.compile(indep), tuple(_rand(ROWS, COLS) for _ in range(4))
    import regex as _re

    _sepm = _re.fullmatch(r"add(\d+)_sep", OP)
    if (
        _sepm
    ):  # add{N}_sep -- control: the SAME dependent chain as add{N} but forced into
        # SEPARATE kernels (each add is its own torch.compile), read-after-write dependency
        # IDENTICAL (the intermediate is written to HBM by one kernel and read by the next).
        # add{N} vs add{N}_sep = the fusion cost with the dependency held fixed (same bytes).
        # NOTE: the cost-model dump captures only the LAST sub-kernel's feats; the MEASURED
        # kernel time (what we compare) covers all sub-kernels.
        f = torch.compile(lambda a, b: a + b)  # noqa: E731
        n = int(_sepm[1])

        def chain(*ts):  # ((t0+t1)+t2)+... each '+' is a distinct compiled kernel
            acc = f(ts[0], ts[1])
            for t in ts[2:]:
                acc = f(acc, t)
            return acc

        return chain, tuple(_rand(ROWS, COLS) for _ in range(n))
    if OP in _CT_REDUCE:  # coarse-tiled dim0 reduction (BENCH_TILES, LX_PLANNING)
        return _ct_workload(_CT_REDUCE[OP])
    if OP == "softmax_stages":  # discriminator: LX traffic vs element count
        return _softmax_stages()
    if OP == "softmax_row_tiling":  # softmax(dim=-1) NROW-tiled -> 5 ops fuse in LX
        return _softmax_row_tiling()
    if OP == "softmax_noexp_row_tiling":  # matched control: softmax structure, exp->mul
        return _softmax_noexp_row_tiling()
    if OP == "matmul_row_tiling":  # a@b coarse-tiled over M (num_tiles, not core split)
        return _matmul_row_tiling()
    if OP == "matmul_k_tiling":  # a@b coarse-tiled over K (reduction dim)
        return _matmul_k_tiling()
    if OP == "mm_nested_m_k":  # mm nested: outer M x2, inner K x TILES
        return _mm_nested_m_k()
    if OP == "bmm_k_tiling":  # bmm [B,M,K]@[B,K,N] tiled over K
        return _bmm_workload("k")
    if OP == "bmm_3d2d_k_tiling":  # matmul [B,M,K]@[K,N] (shared weight) tiled over K
        return _bmm_workload("3d2d")
    if OP == "bmm_nested_b_k":  # bmm nested: outer B x2, inner K x TILES
        return _bmm_workload("nested")
    if OP == "bmm_wd":  # bmm [B,M,K]@[B,K,N] with a FORCED split via WD_B/M/N/K
        return _bmm_workload("k")
    if OP == "bmm_wd_3d2d":  # matmul [B,M,K]@[K,N] shared weight, FORCED split
        return _bmm_workload("3d2d")
    if OP == "bmm_layout":  # full bmm with a chosen device dim_order (WD_LAYOUT_A/B), +
        return _bmm_workload("k")  # optional forced split (WD_B/M/N/K) -- see _to_dev
    if OP == "flash_attn":  # multi-op coarse-tiled flash attention (FA_* hint knobs)
        return _flash_attn_workload()
    if OP == "softmax_unrolled":  # unrolled softmax chain over B (no CoarseTileInfo)
        return _softmax_unrolled()
    if OP == "mm":  # matmul [M=ROWS, K=COLS] @ [K=COLS, N=BENCH_N]: a Reduction over K
        # -> work-division Pass 2 (cost_model_matmul_division) picks the (m,n,k) split.
        mm = lambda a, b: a @ b  # noqa: E731
        return torch.compile(mm), (_rand(ROWS, COLS), _rand(COLS, NCOLS))
    if OP == "mmwd":  # matmul with a FORCED (m,n,k) split via WD_M/WD_N/WD_K
        return _mm_workload()
    known = (
        list(_UNARY)
        + list(_BINARY)
        + list(_NARY)
        + list(_REDUCE)
        + list(_BCAST)
        + list(_TRANSPORT)
        + [
            "bcastcol",
            "write",
            "add_indep2",
            "add3_sep",
            "add4_sep",
            "add5_sep",
            "add6_sep",
            "softmax_row_tiling",
            "softmax_stages",
            "softmax_noexp_row_tiling",
            "softmax_unrolled",
            "matmul_row_tiling",
            "matmul_k_tiling",
            "mm_nested_m_k",
            "bmm_k_tiling",
            "bmm_3d2d_k_tiling",
            "bmm_nested_b_k",
            "bmm_wd",
            "bmm_wd_3d2d",
            "bmm_layout",
            "flash_attn",
            "mm",
            "mmwd",
            "transpose_outer",
        ]
        + list(_CT_REDUCE)
    )
    raise SystemExit(f"unknown BENCH_OP={OP!r} (use {known})")


def _print_io(io: dict) -> None:
    """Per-tensor device-layout I/O the COST MODEL counts: dims, residency, bytes.

    Lines are prefixed ``IO `` so the sweep (run_profile_sweep.sh) can grep them
    alongside the SUMMARY line instead of dropping the breakdown.
    """
    _emit("IO -- device-layout I/O (cost model, stick-padded) --")
    for o in io.get("ops", []):
        red = " [reduction]" if o.get("is_reduction") else ""
        _emit(f"IO   op {o['name']}{red}")
        for a in o["args"]:
            bc = " broadcast (loaded once)" if a["broadcast"] else ""
            lf = a.get("loop_factor", 1)
            xl = f" xL={lf}" if lf > 1 else ""
            log = f"torch {a['logical']} -> " if a.get("logical") else ""
            _emit(
                f"IO     {a['role']:<6} {a.get('name', '?'):<22} "
                f"{log}device {a['dims']} in {a['mem'].upper()} = "
                f"{a['elems']} elems x 2B = {a['bytes']} B"
                f"  (hbm counted: {a['hbm_counted']} B){xl}{bc}"
            )
    _emit(
        f"IO   => HBM I/O total = {io.get('hbm_bytes', 0)} B  "
        f"(lx {io.get('lx_bytes', 0)} B, ~free)"
    )


def _print_model(feats: list) -> float:
    """Print the cost model's ESTIMATED kernel time + rough calc, after the I/O dump.

    Lines are prefixed ``MODEL `` so the sweep greps them. Returns the predicted us so
    the SUMMARY can carry it next to the measured kernel_us.
    """
    if not feats:
        _emit("MODEL (no features extracted)")
        return 0.0
    p = cost_model.CostParams()
    # Delegate the whole step-by-step breakdown to cost_model.explain() rather than
    # duplicating its logic here -- a matmul bundle is now priced entirely by
    # work_division._matmul_split_cost (see cost_model's module docstring), which has a
    # different breakdown shape (reconstructed B/M/N/K axes, no base/turn/eff terms) than
    # the pointwise turnaround model this function used to spell out by hand.
    t = cost_model.predict_ops(feats, p)
    for ln in cost_model.explain(feats, p).splitlines():
        _emit(f"MODEL {ln.strip()}")
    # Machine-readable feature vector (the model's INPUT) so a NEW model version can be
    # scored OFFLINE against the stored measured time -- no hardware re-run. Prefixed
    # `MODEL ` so the sweeps' `^MODEL ` grep already captures it; parse_sweep_logs.py
    # pulls it into the record's `feats`. See tools/cost_model/eval_model.py. Best-effort: a
    # serialization hiccup must NEVER fail the run (the kernel_us is what matters).
    try:
        _emit(f"MODEL FEATS {cost_model.ops_to_json(feats)}")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        _emit(f"MODEL FEATS_SKIPPED {type(exc).__name__}: {str(exc)[:120]}")
    return t / 1000


class _SpanOverflowGuard(logging.Handler):
    """Abort a configuration whose per-core tensor span exceeds the hardware limit.

    WHY THIS EXISTS. On 2026-08-07 a sweep ran one configuration -- a K-tiled
    `bmm_3d2d` -- whose coarse-tile read copy was built over the full iteration space,
    giving a `[4,1024,1024,1024]` staging buffer: 256 MB per core against the 255.996 MB
    MVLOC addressing limit, and 17 GB of HBM traffic from 20 MB of inputs. It was the only
    CRITICAL in 1640 runs. The run immediately after it, and all 1389 that followed, died
    with `RAS::MCI::DdrInitRetryLimitExceeded`, and the card never recovered.

    ONE CO-OCCURRENCE IS NOT A CAUSE, and this guard does not assume otherwise. But a
    per-core span past the addressing limit means DMA descriptors that cannot address what
    they were told to, the cost of being wrong is measured in days of dead hardware, and
    the cost of being over-careful is one skipped data point. So: refuse to keep executing
    a configuration the compiler has already flagged as out of spec.

    LIMIT OF THE PROTECTION. `torch.compile` compiles lazily, so the CRITICAL is emitted
    during the *first* execution -- this cuts exposure from 12 executions (5 warmup +
    7 profiled) to 1, not to 0. Reaching zero would mean compiling the configuration
    under fake tensors first, so that nothing is ever dispatched to the device.
    """

    LOGGER = "spyre.inductor.work_division"
    NEEDLE = "exceeds hardware limit"

    def __init__(self):
        super().__init__(level=logging.CRITICAL)
        self.hits: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if self.NEEDLE in msg:
            self.hits.append(msg)

    @classmethod
    def install(cls):
        guard = cls()
        logging.getLogger(cls.LOGGER).addHandler(guard)
        return guard


def _run():
    def _sync(
        out,
    ):  # move result(s) to host; multi-output workloads return a tuple/list
        for t in out if isinstance(out, (tuple, list)) else (out,):
            t.cpu()

    guard = _SpanOverflowGuard.install()
    compiled, args = make_workload()
    for i in range(WARMUP):  # compile (-> cost-model dump fires) + warm the kernel
        if _PREPARE is not None:  # coarse-tile: re-declare named dims before each trace
            _PREPARE()
        _sync(compiled(*args))
        if i == 0 and guard.hits:
            # Compilation flagged an out-of-spec span. Stop before the remaining warmup
            # and profiled runs; report it the way the sweep parser understands.
            print(f"FAILED reason=span_overflow ({len(guard.hits)} CRITICAL)")
            for msg in guard.hits[:3]:
                print(f"   {msg}")
            print(
                "SKIPPED: this configuration asks for a per-core span past the MVLOC "
                "addressing limit. See _SpanOverflowGuard for why it is not measured."
            )
            return
    io = dict(dump_cost_model.LAST_IO)  # device-layout I/O the model computed
    feats = list(dump_cost_model.LAST_FEATS)  # raw OpFeatures for predict_ops()
    io_hbm_bytes = io.get("hbm_bytes", 0)

    def _profile_once():
        """One profiled trace -> (kernel_us, memset_us, other_us), classified by
        EXCLUSION (a Memset / Memcpy is NOT the fused compute kernel; the new image
        leaves the kernel event name BLANK, so a name match silently reports 0)."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            if (
                _PREPARE is not None
            ):  # coarse-tile: re-declare before the profiled trace
                _PREPARE()
            _sync(compiled(*args))
        kernel = memset = other = 0.0
        for ev in prof.key_averages():
            us = getattr(ev, "self_device_time_total", 0) or getattr(
                ev, "self_cuda_time_total", 0
            )
            if not us or us <= 0:
                continue
            key = ev.key or ""
            if "Memset" in key:
                memset += us
            elif "Memcpy" in key:
                other += us
            else:
                kernel += (
                    us  # fused compute kernel (sdsc_fused / inductor-spyre / BLANK)
                )
        return prof, kernel, memset, other

    # NOISE PROTOCOL: REPS back-to-back profiled measurements. The kernel time is
    # jitter-prone (esp. large-COLS memory ops), so a single measurement can read as a
    # model error; we report min/median/mean/std/cv and use the MEDIAN as kernel_us.
    kernels, memsets, others = [], [], []
    first_prof = None
    t_prof0 = time.perf_counter()
    with _quiet_device_chatter():
        for i in range(REPS):
            prof, k, ms, ot = _profile_once()
            if i == 0:
                first_prof = prof
            if k > 0:
                kernels.append(k)
                memsets.append(ms)
                others.append(ot)
    prof_wall_s = time.perf_counter() - t_prof0

    print(_banner(f"PyTorch profiler: measured device time -- {OP}[{ROWS}x{COLS}]"))
    print(
        "Self SPYRE is the measured device time. The fused compute kernel is the row "
        "this harness reports as kernel_us; Memcpy and Memset are counted separately."
    )
    if first_prof is not None:  # show the first trace's event table (diagnostic)
        print(
            first_prof.key_averages()
            .table(sort_by="cuda_time_total", row_limit=20)
            .replace("CUDA", "AIU")
        )
    _print_io(io)
    pred_us = _print_model(feats)  # cost-model estimate + rough calc (after I/O dump)

    # Aggregate the replicate kernel times. MEDIAN is the reported kernel_us (robust);
    # MIN is the cleanest true-kernel estimate; CV is the noise gate for the analysis.
    if kernels:
        k_med = statistics.median(kernels)
        k_min = min(kernels)
        k_mean = statistics.fmean(kernels)
        k_std = statistics.pstdev(kernels) if len(kernels) > 1 else 0.0
        k_cv = (k_std / k_mean * 100.0) if k_mean > 0 else 0.0
        memset = statistics.median(memsets)
        other = statistics.median(others)
    else:
        k_med = k_min = k_mean = k_std = k_cv = memset = other = 0.0
    # A profiler that produced no Spyre events yields 0.0 here, which reads as a
    # valid degenerate measurement rather than a failure. Say so loudly: the usual
    # cause is a torch_spyre build with USE_SPYRE_PROFILER=0 or a libaiupti that
    # does not match this PyTorch version, and a silent kernel_us=0 would be
    # folded into the database as real data.
    if not kernels or k_med <= 0.0:
        print(
            "WARNING: the profiler reported no Spyre device time. "
            "Check that torch_spyre was built with USE_SPYRE_PROFILER=1 and "
            "that libaiupti matches this PyTorch version "
            "(docs/source/user_guide/profiling/pytorch_profiler.md). "
            "The measurement below is NOT usable.",
            file=sys.stderr,
        )

    kernel = k_med  # SUMMARY kernel_us = median (back-compat)
    # Effective BW from the GOLDEN kernel time and the model's device-layout I/O.
    bw = io_hbm_bytes / (kernel * 1000) if kernel > 0 else 0.0
    err = (pred_us - kernel) / kernel * 100.0 if kernel > 0 else 0.0
    # ACTUAL cores from the model (split product), not the SENCORES budget tag -- a
    # forced (mmwd) split uses cores = m*n*k, which may be < SENCORES.
    mcores = max((getattr(o, "cores", 0) for o in feats), default=0)
    cores_tag = mcores if mcores > 0 else (SENCORES or "-")
    # Per-run TIMING feedback: profiled-region wall time + per-rep, so future sweeps can
    # be sized from real numbers instead of guessing. Prefixed TIMING for the sweep grep.
    print(
        f"TIMING op={OP} reps={len(kernels)}/{REPS} prof_wall_s={prof_wall_s:.2f} "
        f"per_rep_s={prof_wall_s / max(1, REPS):.3f}"
    )
    print(
        f"SUMMARY op={OP} rows={ROWS} cols={COLS} cores={cores_tag} "
        f"tiles={TILES} stages={STAGES} lx={LX} io_hbm_bytes={io_hbm_bytes} "
        f"kernel_us={kernel:.3f} pred_us={pred_us:.3f} err_pct={err:+.1f} "
        f"bw_gbps={bw:.1f} memset_us={memset:.3f} "
        f"other_dev_us={other:.3f} total_dev_us={kernel + memset + other:.3f} "
        f"kernel_us_min={k_min:.3f} kernel_us_median={k_med:.3f} "
        f"kernel_us_mean={k_mean:.3f} kernel_us_std={k_std:.3f} "
        f"kernel_us_cv={k_cv:.2f} reps={len(kernels)}"
    )


def main():
    # Self-report failures: print the full traceback (stderr) AND a parseable FAILED
    # SUMMARY (stdout) carrying the reason, so a sweep that greps SUMMARY still records
    # WHY a run died instead of a bare "FAILED".
    try:
        _run()
    except Exception as exc:  # noqa: BLE001 - diagnostic wrapper
        import traceback

        traceback.print_exc()
        print(
            f"SUMMARY op={OP} rows={ROWS} cols={COLS} tiles={TILES} lx={LX} "
            f"FAILED reason={type(exc).__name__}: {str(exc)[:140]}"
        )


if __name__ == "__main__":
    main()
