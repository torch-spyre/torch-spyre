# SDPA nested `for_each_tile` implementation report

Last updated: 2026-09-22

## Executive summary

This branch rewrites Spyre SDPA's complete `B`/`Hkv`/`G`/`Lq`/`Lk` tile nest
with `for_each_tile`. It is based on upstream main at `eaea0108`, including
the non-contiguous input streaming support from #4750 and the shared SDPA/SWA
cost-model helpers from #4610. The full-HOP path contains no named-dimension
hints.

The nested-HOP correctness blockers found during the original experiment are
fixed. The focused full-nest regression and the production SDPA tests pass on
Spyre. Fresh-cache Granite 3.3 8B and Gemma 4 26B A4B runs both complete
chunked-prefill plus decode at 8K and 32K.

The prefill selector is cost based. It contains no model identities,
sequence-length cutoffs, or fixed prefill Lq/Lk ceilings. It enumerates
shape-derived `B`/`Hkv`/`G`/`Lq`/`Lk` plans and estimates live LX, active-core
ownership, logical HBM bursts, aggregate HBM traffic, non-dense head staging,
and loop/DSC overhead. A conservative one-query-buffer live-set uncertainty is
admitted only after pricing its write/read traffic.

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

A map level is elided when its selected tile covers the complete axis. The GQA
group map is also elided when both sequence axes fit in one tile, because a
G-only map cannot reduce the sequence working set and would only serialize the
query-head groups.

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

## Cost model

### Prefill candidate space

For chunked prefill, the selector constructs candidates as follows:

- every divisor of the batch, physical-head, and GQA-group extents is an exact
  map-loop trip count;
- Lq candidates start at the complete query extent and descend through
  power-of-two extent ceilings, each normalized to an exact divisor; and
- Lk candidates start from power-of-two block ceilings plus the complete KV
  extent, then normalize to unique exact tiles. They retain 64-row alignment
  whenever Lk itself is aligned, as production inputs are.

This searches every outer-axis division and a bounded, shape-derived set of
exact sequence tilings without checking a model name or context-length range.

### Per-plan estimates

For each plan, the selector estimates:

- CP-SAT's largest exact product split over B, Hkv/H, G, and the current Lq
  tile, capped by the available cores;
- four simultaneously live score-shaped values;
- seven query/output-shaped values, including map staging and carry handoff;
- the two row-shaped scalar online-softmax carries, M and l;
- a conservative per-core restickified-K footprint divided only over B and
  physical KV heads, because K is invariant over G and Lq;
- mask replay on broadcast outer axes and K/V replay on G/Lq tiles;
- read/write staging traffic when head tiling slices an interleaved,
  non-dense query, key, or value layout; and
- two logical K/V bursts per outer HOP trip and Lk block.

The live-set estimate is `4 * score + 7 * query/output + 2 * row scalar +
restickified K`. V remains streamed and is not charged as a resident value.
The LX limit is the frontend allocator's actual per-core planning budget, not
the card's nominal physical capacity.

Aggregate traffic includes one query read/output write, K/V replay for every G
and Lq tile, broadcast-mask replay, any non-dense head staging, and a modeled
spill penalty. It is converted to full-card transfer waves with the calibrated
1 MiB/core target. Logical burst count remains separate so two equally sized
transfers with different loop fragmentation do not look equivalent.

The analytical liveness estimate is deliberately conservative around the map
and carry handoff. A plan may exceed the budget by at most one query-sized slot;
the model then charges a write and read of the larger score/query buffer on
every inner trip. This is an admission and ranking penalty, not a claim that
the final allocator will spill that value. Plans requiring two or more such
slots are rejected.

The primary score is the sum of calibrated DSC executions, logical load bursts,
and aggregate HBM transfer waves. Ties prefer, in order: fewer modeled overflow
slots, fewer bursts, fewer HBM bytes, fewer DSC executions, more active cores,
less restick work, fewer B/H tiles, a larger Lq tile, fewer G tiles, and a
larger Lk block. The DSC estimate charges eight fixed executes, roughly 17 per
Lk block, and counted-loop group boundaries for every outer tile.

### Decode and fallback

Decode keeps the separately calibrated #4549 policy because its single query
row is structurally different from prefill. It first requires its narrower
two-score/two-query live estimate to fit LX, then separately prefers eligible K
blocks of at least 256 rows whose restickified K is likely to remain in LX and
whose block count stays within the calibrated multi-block window. Otherwise it
minimizes DSC executes and prefers the larger K block.

There is no legacy non-HOP SDPA path in this branch. Decode still uses the same
`for_each_tile` decomposition; only its liveness and K-block scoring remain
separately calibrated. SWA also reuses the narrow score/query accounting helper,
but it is a distinct operator with its own selector.

If no prefill plan fits within the one-slot allowance, or no decode candidate
fits LX, the selector retains conservative fallback tiling inside the same HOP
decomposition: exact Lq tiles of at most 512 rows, exact K tiles bounded by the
smaller of 512 rows and an aligned quarter of Lk, and the established B/H/G
split heuristics.

Representative choices with a 1,625,344-byte per-core LX budget are:

| Geometry | Selected plan | Active cores | Logical K/V bursts | Estimated live bytes/core | Admission penalty |
| --- | --- | ---: | ---: | ---: | --- |
| Granite 8K, Hq=32, Hkv=8, Lq=512, D=128 | B1/H1/G1/Q2/K512 | 32 | 64 | 1,639,424 | 1 query-sized slot |
| Granite 32K, Hq=32, Hkv=8, Lq=512, D=128 | B1/H1/G1/Q4/K1024 | 32 | 256 | 1,540,608 | none |
| Gemma 4 8K, Hq=16, Hkv=8, Lq=1024, D=256 | B1/H1/G1/Q2/K256 | 32 | 128 | 1,573,888 | none |
| Gemma 4 32K, Hq=16, Hkv=8, Lq=1024, D=256 | B1/H1/G1/Q2/K256 | 32 | 512 | 1,573,888 | none |

`B1/H1/G1/Q2` denotes the number of map tiles on each outer axis; `K512`
denotes the Lk tile extent in tokens. Thus Granite 8K uses two 256-row query
tiles and sixteen 512-row KV tiles, rather than a one-row head or group tile.

The Granite crossover is selected from costs rather than a sequence-length
condition. At 8K, two Q tiles avoid enough K/V replay to offset the conservatively
priced carry shortfall. At 32K, four Q tiles with K1024 eliminate that shortfall
and win as its repeated transfer cost grows. Gemma's wider D=256 geometry keeps
the resident Q2/K256 plan at both lengths.

## Correctness results

On upstream main `eaea0108` plus this branch:

```text
tests/inductor/test_sdpa_tiling.py:                 30 passed, 142 subtests
focused production SDPA device tests:                7 passed
pre-commit on all modified files:                    passed
```

The focused integration regression forces all five B/Hkv/G/Lq/Lk loops through
the actual SDPA decomposition, checks the generated LoopSpecs, and verifies the
result against PyTorch. Generic map nesting, map-over-carry, ragged-tile
rejection, and standalone Lk scans remain covered by the existing
`for_each_tile` lowering/E2E suites and the production SDPA tests instead of
being duplicated here.

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

A fresh-cache, same-process kernel comparison used default CP-SAT,
`CO_OPTIMIZING_LX_PLANNING=1`, and 100 synchronized, device-resident iterations.
K/V are transposed prefix views whose backing allocation is one prefill chunk
longer, matching the static-cache layout used by the adapter. The later
#4753/#4759 rebases changed only CI and ClickHouse ingestion code, so these
compiler/runtime measurements remain applicable to the current tree.

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

An earlier forced-K sweep at Granite's 32K geometry with tightly allocated K/V
explains the K1024 choice:

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
online-softmax dataflow. In particular, Granite 8K's one-slot admission penalty
does not become an inner-loop spill in the generated OpSpecs.

Earlier forced-plan studies explain the selections. Granite's Q2/K512 plan was
faster end-to-end at 8K, while Q4/K1024 won at 32K. For Gemma's wider D=256
geometry, K512 crossed the estimated resident live set and regressed sharply
at 32K. The final model reproduces those choices without checking model names
or sequence lengths.

## Non-contiguous KV-cache support inherited from #4750

The decomposition still canonicalizes Q once before entering the nested maps.
This is not needed to identify the logical HOP axes: those are explicit now,
and the former named-dimension path is gone. It remains an explicit operation
because `WhileLoop.create` requires exact body-input strides and otherwise
synthesizes the same materialization for the usual logical `[B,H,S,D]` view
backed by physical `[B,S,H,D]` storage. Q is bounded by the prefill chunk, so
this is a predictable one-time copy. Applying the same policy to K/V would copy
the complete context and can exhaust HBM at long sequence lengths.

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
```

The E2E runs set `HF_HOME=/mnt/models/hf_cache` and invoke the adapter with
`prefill_chunk_size=512` for Granite or `1024` for Gemma.
