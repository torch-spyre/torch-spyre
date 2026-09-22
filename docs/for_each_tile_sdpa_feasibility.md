# SDPA `for_each_tile` feasibility report

Last updated: 2026-09-22

## Executive summary

This branch rewrites Spyre SDPA's complete `B`/`Hkv`/`G`/`Lq`/`Lk` tile nest
with `for_each_tile`. It is based on upstream main at `a2e41402`, including
the non-contiguous input streaming support from #4750 and the shared SDPA/SWA
cost-model helpers from #4610. The full-HOP path contains no named-dimension
hints.

The nested-HOP correctness blockers found during the original experiment are
fixed. The focused nine-case suite and the production SDPA tests pass on
Spyre. Fresh-cache Granite 3.3 8B and Gemma 4 26B A4B runs both complete
chunked-prefill plus decode at 8K and 32K.

The tiling selector is cost based. It does not contain model identities,
sequence-length cutoffs, or maximum query/K tile limits. It enumerates exact
`B`/`Hkv`/`G`/`Lq`/`Lk` plans and estimates live LX, active-core ownership,
logical HBM bursts, aggregate HBM traffic, non-dense head staging, and
loop/DSC overhead. A conservative one-carry live-set uncertainty is admitted
only after pricing its write/read traffic.

## SDPA structure

The decomposition expresses these loops directly:

```text
MHA:
for_each_tile(B, map)
  for_each_tile(H, map)
    for_each_tile(Lq, map)
      for_each_tile(Lk, carry M, l, O)

GQA:
for_each_tile(B, map)
  for_each_tile(Hkv, map)
    for_each_tile(G, map)
      for_each_tile(Lq, map)
        for_each_tile(Lk, carry M, l, O)
```

For GQA, Q and bias have logical shape `[B, Hkv, G, Lq, ...]`; K and V keep a
unit G dimension and are invariant in the G loop. The Lk reduction implements
stable online softmax:

```text
scores = Q @ K_tile.T * scale + bias_tile
M_next = max(M, amax(scores, dim=-1))
correction = exp(M - M_next)
P = exp(scores - M_next)
l_next = l * correction + sum(P, dim=-1)
O_next = O * correction + P @ V_tile
result = O_final / l_final
```

## Selector model

For chunked prefill, the selector enumerates every exact tile count for the
batch, physical-head, and GQA-group axes, every exact query tile generated
from the full Lq extent down to one row, and every exact, stick-aligned K tile
generated from power-of-two burst candidates plus the full K extent. For each
plan it estimates:

- CP-SAT's usable core count over every axis visible to the inner HOP;
- four simultaneously live score-shaped values;
- seven query/output-shaped values, including map staging and carry handoff;
- the two scalar online-softmax carries;
- the per-core restickified K footprint;
- mask replay on broadcast outer axes and K/V replay on G/Lq tiles;
- staging traffic when an interleaved head slice is not dense; and
- two logical K/V bursts per outer HOP trip and Lk block.

Aggregate bytes are converted to full-card transfer waves using the calibrated
1 MiB/core HBM target. The analytical liveness count is conservative by one
query/output carry at the map boundary, so the selector can admit that one
buffer of shortfall, but charges a write and read of the larger score/query
buffer on every inner trip. Plans requiring a larger shortfall are rejected.
The remaining plans are ranked by DSC executions plus logical load bursts plus
HBM transfer waves, with residency, parallelism, restick work, and tile counts
as tie breakers. Decode retains its separately calibrated policy because its
one-row execution is structurally different from chunked prefill.

Representative choices with a 1,625,344-byte per-core LX budget are:

| Geometry | Selected plan | Estimated live bytes/core | Priced shortfall |
| --- | --- | ---: | ---: |
| Granite 8K, Hq=32, Hkv=8, Lq=512, D=128 | B1/H1/G1/Q2/K512 | 1,639,424 | 1 buffer |
| Granite 32K, Hq=32, Hkv=8, Lq=512, D=128 | B1/H1/G1/Q4/K1024 | 1,540,608 | none |
| Gemma 4 8K/32K, Hq=16, Hkv=8, Lq=1024, D=256 | B1/H1/G1/Q2/K256 | 1,573,888 | none |

The Granite crossover is selected from costs rather than a sequence-length
condition. At 8K, two Q tiles avoid enough K/V replay to offset the conservatively
priced carry shortfall. At 32K, four Q tiles with K1024 eliminate that shortfall
and win as its repeated transfer cost grows. Gemma's wider D=256 geometry keeps
the resident Q2/K256 plan at both lengths.

## Correctness results

On upstream main `a2e41402` plus this branch:

```text
tests/inductor/test_sdpa_tiling.py:                 29 passed, 142 subtests
tests/inductor/test_sdpa_for_each_tile.py (OOT):     9 passed
focused production SDPA device tests:                7 passed
pre-commit on all modified files:                    passed
```

The production group covers the Lk-HOP structural check, Granite finite-mask
decode, broadcast-mask prefill, direct and fallback non-contiguous KV-prefix
paths, forced four-by-four Lq/Lk tiling, and the larger production layout. The
two transposed-input tests no longer attach named dimensions: the nested HOP
must carry all tiling structure itself.

All requested chunked-prefill plus two-token decode E2Es pass with the final
selector:

| Model | Chunk | 8K | 32K |
| --- | ---: | --- | --- |
| Granite 3.3 8B Instruct | 512 | Pass; `of France` | Pass; `France is` |
| Gemma 4 26B A4B | 1024 | Pass; `of France` | Pass; `France-` |

## Compile-time and runtime observations

A fresh-cache, same-process kernel comparison used 100 synchronized,
device-resident iterations. K/V are transposed prefix views whose backing
allocation is one prefill chunk longer, matching the static-cache layout used
by the adapter.

| Geometry | KV length | Selected plan | Median runtime | Compile + first |
| --- | ---: | --- | ---: | ---: |
| Granite | 8K | Q2/K512 | 11.273 ms | 11.247 s |
| Granite | 32K | Q4/K1024 | 38.757 ms | 11.978 s |
| Gemma 4 | 8K | Q2/K256 | 17.934 ms | 10.820 s |
| Gemma 4 | 32K | Q2/K256 | 42.081 ms | 14.665 s |

These production-shaped figures should not be compared directly with the much
smaller PR #4550 benchmark below.

PR #4550's smaller benchmark geometry is not directly comparable with the
Granite rows above: it uses MHA with `H=2` and `Lq=64`, versus Granite's
`Hq=32`, `Hkv=8`, and `Lq=512` (128x as many query-head rows). Re-running this
branch with #4550's exact geometry, default CP-SAT/optimizing LX planning, and
100 synchronized samples gives:

| Lk | PR #4550 published median | This PR median | This PR compile + first |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.734 ms | 0.466 ms | 6.20 s |
| 32,768 | 2.406 ms | 1.112 ms | 6.40 s |

Thus the full-HOP version is 36.5% faster at 8K and 53.8% faster at 32K on
the same workload. The larger absolute Granite timings reflect the larger
production workload, not a regression relative to #4550.

The end-to-end runner reports two-token generation time. The cold invocation
includes compilation of both prefill and decode graphs; two warm invocations
reuse the same-process compiled graphs.

| Case | Chunk | Cold total | Warm totals | Warm first-token median |
| --- | ---: | ---: | --- | ---: |
| Granite 8B, 8K | 512 | 75.309 s | 10.532 / 10.527 s | 10.348 s |
| Granite 8B, 32K | 512 | 177.746 s | 113.810 / 113.907 s | 112.305 s |
| Gemma 4 26B, 8K | 1024 | 165.263 s | 21.176 / 21.016 s | 20.832 s |
| Gemma 4 26B, 32K | 1024 | 271.112 s | 138.327 / 136.217 s | 136.183 s |

For comparison, the reported main compile times were approximately 20 minutes
at 8K and one hour at 32K. Fixed chunk shapes now compile once instead of
specializing on total context length. The latest Granite 32K warm run is slower
than the earlier 91.05 s sample, but its isolated padded attention kernel is
unchanged at 38.76 ms versus 38.90 ms; this does not indicate an attention
codegen regression.

Isolated Granite SDPA measurements explain the K1024 choice:

| K tile | Runtime |
| ---: | ---: |
| 256 | 27.39 ms |
| 512 | 24.10 ms |
| 1024 | 23.11 ms |
| 2048 | 80.54 ms |

Generated LoopSpecs match the selected plans: Granite emits Q2/K16 at 8K and
Q4/K32 at 32K, while Gemma emits Q2/K32 and Q2/K128. The corresponding OpSpecs
place all score-shaped and query/output-shaped inner-loop intermediates in LX.
The remaining `hbm`/`hbm_pool` values are graph inputs, restick staging, carries
at loop boundaries, and map-result materialization—not wholesale spills of the
online-softmax dataflow.

Earlier forced-plan studies explain the selections. Granite's Q2/K512 plan was
faster end-to-end at 8K, while Q4/K1024 won at 32K. For Gemma's wider D=256
geometry, K512 crossed the estimated resident live set and regressed sharply
at 32K. The final model reproduces those choices without checking model names
or sequence lengths.

## Non-contiguous KV-cache support inherited from #4750

PR #4750 teaches `for_each_tile`/`WhileLoop` splicing to contract an exact-stride
materialization of a non-contiguous graph input to one streamed tile. Its
direct-read proof now compares affine bounds with `storage_size()` rather than
logical `numel`. This deliberately broadens a general-purpose direct-read proof,
not only the KV-prefix case; both quantities include the storage offset, so the
bounds comparison remains like-for-like.

If encoding, layout, ownership, or bounds do not prove a direct read safe, the
compiler retains the contracted one-tile staging copy. This branch adds a
device test that forces that proof to decline, checks the fallback shape is
exactly `(1, 64, 1, 8, 128)`, and verifies the numerical result. The test takes
about 13 seconds locally under its 30-second CP-SAT limit. It also carries the
review-requested diagnostics: comments explain why ambiguous axis matches bail
out instead of guessing, and a debug message records when the FX graph needed
for pass-through carry detection is unavailable.

PR #4551 is rebased onto current upstream main, which contains both #4750 and
#4610; its former duplicate compiler patch has been removed.

## Remaining work

- Add destination-backed map outputs to remove scan stack/fold
  materializations.
- Continue calibrating the analytical live-set and HBM-wave estimates against
  generated OpSpecs when allocator behavior changes.

## Reproduction

```bash
python -m pytest -q tests/inductor/test_sdpa_tiling.py
python -m pytest -q tests/inductor/test_sdpa_for_each_tile.py
```

The E2E runs set `HF_HOME=/mnt/models/hf_cache` and invoke the adapter with
`prefill_chunk_size=512` for Granite or `1024` for Gemma.
