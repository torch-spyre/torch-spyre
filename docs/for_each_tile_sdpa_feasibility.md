# SDPA `for_each_tile` feasibility report

Last updated: 2026-09-21

## Executive summary

This branch rewrites Spyre SDPA's complete `B`/`Hkv`/`G`/`Lq`/`Lk` tile nest
with `for_each_tile`. It is based on upstream main at `bfcaa316` and contains
no named-dimension hints.

The nested-HOP correctness blockers found during the original experiment are
fixed. The focused nine-case suite and the production SDPA tests pass on
Spyre, and Granite 3.3 8B completes chunked-prefill plus decode at 8K and 32K.
Gemma 4 12B and 26B A4B had already passed the same 8K/32K E2Es with the
K256 plan that the new selector still chooses.

The tiling selector is cost based. It does not contain model identities,
sequence-length cutoffs, or maximum query/K tile limits. It enumerates exact
Lq/Lk tile pairs, estimates the nested loop body's live LX footprint and
active-core ownership, rejects candidates that do not fit the available LX,
and ranks the remaining candidates by K/V load bursts and loop/DSC overhead.

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

For chunked prefill, the selector enumerates every exact query tile generated
from the full Lq extent down to one row and every exact, stick-aligned K tile
generated from power-of-two burst candidates plus the full K extent. For each
pair it estimates:

- CP-SAT's usable core count over the inner physical-head and query-row axes;
- four simultaneously live score-shaped values;
- four query/output-shaped values;
- the two scalar online-softmax carries;
- the per-core restickified K footprint; and
- two K/V load bursts per outer HOP trip and Lk block.

Only plans whose estimated live set fits the actual frontend LX planning
budget are eligible. Eligible plans are ordered by estimated load bursts,
then outer-loop count, DSC executions, and block width. Decode retains its
separately calibrated policy because its one-row execution is structurally
different from chunked prefill.

Representative choices with a 1,625,344-byte per-core LX budget are:

| Geometry | Selected plan | Estimated live bytes/core |
| --- | --- | ---: |
| Granite, Hq=32, Hkv=8, Lq=512, D=128 | Lq512 / K1024 | 1,442,304 |
| Gemma 4, Hq=16, Hkv=8, Lq=1024, D=256 | Lq1024 / K256 | 1,180,672 |

The same plan is selected at 8K and 32K for each geometry because sequence
length changes the number of bursts, not whether one tile's live set fits LX.

## Correctness results

On upstream main `bfcaa316`:

```text
tests/inductor/test_sdpa_tiling.py:             22 passed
tests/inductor/test_for_each_tile_lowering.py:  41 passed, 1 expected failure
tests/inductor/test_sdpa_for_each_tile.py:        9 passed
focused production SDPA tests:                   4 passed
```

The production group consists of the Lk-HOP structural check, Granite finite
mask decode, Granite finite broadcast-mask prefill, and forced four-by-four
Lq/Lk tiling.

All requested chunked-prefill plus decode E2Es passed before the final selector
rewrite:

| Model | Chunk | 8K | 32K |
| --- | ---: | --- | --- |
| Granite 3.3 8B Instruct | 512 | Pass | Pass |
| Gemma 4 12B | 1024 | Pass | Pass |
| Gemma 4 26B A4B | 1024 | Pass | Pass |

After the selector rewrite and latest-main merge, Granite was rerun because
its selected K tile changed from K512 to K1024:

| Case | Result |
| --- | --- |
| Granite 8B, 8K, chunk 512 | Pass; output suffix `of the` |
| Granite 8B, 32K, chunk 512 | Pass; output suffix `France is` |

The latest selector still chooses K256 for both Gemma models, so their
previously passing execution path did not change.

## Compile-time and runtime observations

The end-to-end runner reports first-token latency, which includes compilation
on the first invocation and chunked-prefill execution on every invocation.

| Case | Cold first token | Warm first token | Notes |
| --- | ---: | ---: | --- |
| Granite 8B, 8K | 138.94 s | 35.30 s | Approx. 103.64 s cold-only overhead |
| Granite 8B, 32K | 1,506.36 s | >4 min | Cold run passed; the 30-minute wrapper expired during the warm run |

For comparison, the reported main compile times were approximately 20 minutes
at 8K and one hour at 32K. The 8K cold-only overhead is therefore about an
order of magnitude lower. The 32K cold total is 25.1 minutes including model
execution; it is below the old compile-only baseline, but the incomplete warm
run means compile and runtime cannot yet be separated precisely.

Isolated Granite SDPA measurements explain the K1024 choice:

| K tile | Runtime |
| ---: | ---: |
| 256 | 27.39 ms |
| 512 | 24.10 ms |
| 1024 | 23.11 ms |
| 2048 | 80.54 ms |

Generated LoopSpecs/OpSpecs show that K1024 retains the online-softmax
intermediates in LX. K2048 exceeds the estimated live set and spills nearly
all intermediates to `hbm_pool`, matching its large regression. A short
Granite chunk (`Lq=64`, `Lk=8K`) likewise measured K4096 at 3.24 ms versus
K512 at 4.22 ms, supporting selection by residency and burst count rather
than a fixed K512 ceiling.

For Gemma's wider D=256 geometry, K512 crosses the estimated resident live
set. Existing measurements were 38.5 ms (K256) versus 36.2 ms (K512) at 8K,
and 47.8 ms (K256) versus 72.7 ms (K512) at 32K. The model therefore chooses
the resident K256 plan without checking the model name or sequence length.

## Additional compiler fix exposed by the E2E

The new Granite K1024 plan exposed a generic scan-carry classification bug.
PyTorch's scan lowering carries the tiled `xs` tensors through its generated
while loop. A packed K-cache view needed an output stride-repair copy, so the
IR output buffer no longer had the same name as its input placeholder. The
bridge interpreted that copy as an accumulator update and changed its output
to a mutation of the original rank-4 cache, even though the copy iterated over
the rank-5 tile stack. Dependency extraction then failed on the rank/stride
mismatch.

The bridge now consults the original body FX graph, where a pass-through carry
is unambiguous: output position `i` is the same node as placeholder `i`. This
keeps the stride-repair copy as a copy and avoids any geometry-specific
safeguard. A focused unit test covers the distinction between a real updated
carry and a stride-repaired pass-through carry.

## Remaining work

- Complete an uninterrupted same-process Granite 32K warm run to separate
  compile time from steady-state chunked-prefill runtime.
- Re-run the Gemma 12B/26B E2Es if changes after this branch alter their K256
  plan or the shared nested-HOP lowering.
- Add destination-backed map outputs to remove scan stack/fold
  materializations.
- Compare the selector's estimates against generated LoopSpecs/OpSpecs for a
  broader geometry grid and refine buffer lifetimes if the allocator changes.

## Reproduction

```bash
python tests/inductor/test_sdpa_tiling.py -v
python -m pytest -q tests/inductor/test_sdpa_for_each_tile.py
```

The E2E runs set `HF_HOME=/mnt/models/hf_cache` and invoke the adapter with
`prefill_chunk_size=512` for Granite or `1024` for Gemma.
