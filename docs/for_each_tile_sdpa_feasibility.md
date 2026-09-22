# Nested `for_each_tile` SDPA

Spyre SDPA expresses the complete `B`/`Hkv`/`G`/`Lq`/`Lk` tile nest with
`for_each_tile`. The decomposition contains no named-dimension hints.

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

The plan notation records map-loop trip counts and the Lk tile extent. For
example, `B1/H1/G1/Q2/K512` means one batch tile, one physical-head tile, one
GQA-group tile, two query tiles, and a 512-row key tile. Exact choices are
cost-model outputs and intentionally are not documented here because they vary
with tensor geometry and allocator configuration.

## Non-contiguous KV-cache support

The decomposition still canonicalizes Q once before entering the nested maps.
This is not needed to identify the logical HOP axes: those are explicit now,
and the former named-dimension path is gone. It remains an explicit operation
because `WhileLoop.create` requires exact body-input strides and otherwise
synthesizes the same materialization for the usual logical `[B,H,S,D]` view
backed by physical `[B,S,H,D]` storage. Q is bounded by the prefill chunk, so
this is a predictable one-time copy. Applying the same policy to K/V would copy
the complete context and can exhaust HBM at long sequence lengths.

`for_each_tile`/`WhileLoop` splicing contracts an exact-stride materialization
of a non-contiguous graph input to one streamed tile. Its direct-read proof
compares affine bounds with `storage_size()` rather than logical `numel`.
Both quantities include the storage offset, so the bounds comparison remains
like-for-like.

If encoding, layout, ownership, or bounds do not prove a direct read safe, the
compiler retains the contracted one-tile staging copy. Ambiguous axis matches
bail out instead of guessing, and debug logging records when the body FX graph
needed for pass-through carry detection is unavailable.

## Testing

`tests/inductor/test_sdpa_tiling.py` covers selector behavior and drives the
production decomposition through all five nested loop levels. Generic map
nesting, map-over-carry, and ragged-tile rejection are covered by the
`for_each_tile` lowering and end-to-end tests.
