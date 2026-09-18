# SDPA `for_each_tile` feasibility report

Last updated: 2026-09-17

## Executive summary

The branch is rebased on `upstream/main` after PRs #4550 and #4559. The merged
`Lk` online-softmax HOP works on Spyre, including native GQA with dense bias,
and the full four-level MHA and five-level GQA programs are numerically correct
in CPU eager and compiled modes. PR #4559 fixes the stale nested-placeholder
lookup that originally blocked this experiment, but nested HOPs are still not
end-to-end executable on Spyre. A minimal two-map program, an `Lq` map around an
`Lk` reduction, the complete GQA nest, and the production forced-tiling case all
now reach code generation and fail on an unregistered outer-loop indirect
symbol (issue #4581).

There are four other gaps to resolve before production use:

1. Map-mode loops currently materialize scan outputs. The prototypes contain
   four copy/stack-style materialization nodes. The unmerged `scan(out=)` and
   `for_each_tile(output=)` branches are designed to eliminate these writes.
2. A stride-zero broadcast attention-bias view fails Spyre layout propagation,
   even with only the otherwise-working `Lk` loop.
3. `for_each_tile` requires exact tile divisibility. Production SDPA accepts a
   shorter final K/V block, so arbitrary `Lk` needs padding and masking, an exact
   divisor policy, or frontend support for a ragged final tile.
4. The #4549 fast path relies on named work division. The no-`named_dims`
   full-HOP version needs an axis-based equivalent to preserve that schedule.

The recommendation remains to keep the merged #4550 `Lk`-only implementation
in production and leave this PR in draft. Finish #4581 and validate
destination-backed map outputs, broadcast bias, and work-division parity before
replacing the outer hint scopes.

## Experiment baseline

- Repository: `torch-spyre/torch-spyre`
- Base: `upstream/main` at
  `6d9ab2c1f709f1d4e87a57617a73a112dac51019`
- Experiment branch: `codex/sdpa-for-each-tile-experiment`
- Worktree: `/tmp/torch-spyre-sdpa-fet.8jYqoJ`
- Prototype: `tests/inductor/test_sdpa_for_each_tile.py`

The base includes the merged `Lk` conversion (#4550), the current SDPA cost
model (#4549), nested tile-dimension provenance (#4559), completed split-matmul
LX retention (#3955), and device-aware allocation helpers (#4548). The native
extension was rebuilt at this base before retesting.

This draft branch includes the proposed production replacement so the team can
inspect and iterate on the complete shape. It is intentionally not ready to
merge: the remaining nested-lowering failure below is reproduced by that
implementation.

## Original SDPA structure

On current main, `spyre__sdpa_overrideable` chooses a tiling plan with
`_select_sdpa_tiling`, expresses outer coarse tiling with nested `spyre_hint`
scopes, and uses the merged carry-mode `for_each_tile` for `Lk`:

- a batch tile scope;
- a head tile scope for MHA;
- query-sequence tiling and, for eligible shapes, named work division;
- explicit rank-5 `[B, Hkv, G, Lq, D]` operands for native GQA; and
- a carry-mode HOP that slices K and V into `Lk` blocks.

The `Lk` HOP implements stable online softmax with three SSA values:
the running maximum `M`, denominator `l`, and unnormalized output accumulator
`O`. The loop is deliberately split into bounded groups because the generated
bundle size previously scaled with the `Lq`-tile by unrolled-`Lk`-block product;
#4550 removes that Python unrolling.

Any HOP rewrite must preserve the existing tiling cost model, per-block K/V
layout normalization, GQA semantics, causal/additive mask behavior, output
stride contract, and the safety motivation behind the loop-group limit. The
goal is to change how the chosen tiling is represented, not to discard those
policies.

## Proposed HOP structure

The natural MHA nest is:

```text
for_each_tile(B, map)
  for_each_tile(H, map)
    for_each_tile(Lq, map)
      for_each_tile(Lk, reduction carrying M, l, O)
```

Native GQA adds the query-head group axis without repeating K or V:

```text
for_each_tile(B, map)
  for_each_tile(Hkv, map)
    for_each_tile(G, map)
      for_each_tile(Lq, map)
        for_each_tile(Lk, reduction carrying M, l, O)
```

For GQA, Q and bias have logical shape `[B, Hkv, G, Lq, ...]`; K and V retain a
unit group dimension and are invariant at the `G` level. At the innermost level,
Q is invariant, K and V slice dimension `-2`, and bias slices dimension `-1`.
For each K/V tile the recurrence is:

```text
scores = Q @ K_tile.T * scale + bias_tile
M_next = max(M, amax(scores, dim=-1))
correction = exp(M - M_next)
P = exp(scores - M_next)
l_next = l * correction + sum(P, dim=-1)
O_next = O * correction + P @ V_tile
result = O_final / l_final
```

The outer levels use map mode and concatenate the returned tile along their
respective output dimension. The `Lk` level uses reduction mode (`init=(M, l,
O)`) and emits no per-iteration output.

The draft implementation in this branch follows that structure directly. It
bypasses one-trip map levels, retains the existing tiling cost model, uses an
exact divisor no larger than the selected `Lk` block size, and constructs the
sparse `M` and denominator accumulators per query tile. Masks remain separate
operands and are sliced only along axes whose extent matches the tiled axis.
The HOP dimensions fully describe tiling, so this version contains no
`named_dims` hints. This code is included to make the complete lowering shape
reviewable. On the current base it traces and splices every nested level, then
fails later during Spyre code generation as described below.

## Results

### Passing cases

| Case | Backend | Result |
| --- | --- | --- |
| Full MHA `B/H/Lq/Lk` nest | CPU eager and `torch.compile` | Numerically correct; four `while_loop`s captured |
| Full GQA `B/Hkv/G/Lq/Lk` nest | CPU eager and `torch.compile` | Numerically correct; five `while_loop`s captured |
| Standalone MHA `Lk` online-softmax loop, dense bias | Spyre | Numerically correct; one `LoopSpec` emitted |
| Standalone native-GQA `Lk` loop, dense bias | Spyre | Numerically correct without repeating K/V; one `LoopSpec` emitted |
| Existing upstream online-softmax carry test | Spyre | Passes |

The Spyre tests use FP16 operands with `D=128`, `Lq=64`, `Lk=256`, and
`Lk` tiles of 128. Results match the reference at `atol=0.1, rtol=0.1`.

The complete prototype suite reports:

```text
Ran 9 tests in 46.165s
OK (expected failures=4)
```

The four expected failures are intentional regression tests for nested maps,
the `Lq`/`Lk` nest, the complete GQA nest, and broadcast bias.

Focused production-decomposition checks on the same build show:

| Test | Result |
| --- | --- |
| `test_sdpa_lk_uses_for_each_tile` | Pass |
| Granite GQA decode, `Lq=1`, `Lk=128` | Fails on outer symbol `u5` |
| Granite GQA prefill, `Lq=128`, `Lk=128` | Fails on outer symbol `u5` |
| Forced four-by-four Granite GQA, `Lq=Lk=256` | Fails on outer symbol `u42` |

GQA always has at least the `Hkv -> G` map nesting in this rewrite, so even a
one-block `Lk` case is affected. The full decomposition cannot currently serve
as a fallback for ordinary GQA decode or prefill.

### Blocker 1: outer-loop symbols in nested code generation (#4581)

The minimal reproducer is two map-mode loops over a `[128, 128]` FP16 tensor:

```python
def outer_body(_, outer_tiles):
    (outer_tile,) = outer_tiles

    def inner_body(_, inner_tiles):
        (inner_tile,) = inner_tiles
        return None, inner_tile + 1

    _, inner_result = for_each_tile(
        inner_body,
        (outer_tile,),
        dims=(1,),
        tile_size=64,
        out_dim=1,
    )
    return None, inner_result

_, result = for_each_tile(
    outer_body,
    (x,),
    dims=(0,),
    tile_size=64,
    out_dim=0,
)
```

On the old base this failed during `splice_while_loops` with a stale inner-body
placeholder. PR #4559 fixes that failure: both loops are now spliced and their
tile dimensions are recovered from `tile_dim_marker`. The program proceeds to
Spyre kernel code generation, where the inner body still contains the outer
loop variable in an address expression such as:

```text
tmp0 = ops.load(arg0_1, i1 + 128 * i0 + 8192 * u5)

Unsupported: indirect symbol u5 not found in indirect_sizes {}
```

The same failure occurs for the first SDPA-specific nest (`Lq` map around an
`Lk` reduction), the complete five-level GQA program, and
`test_granite_gqa_prefill_four_by_four_sequence_tiling` in the production
decomposition (`u42` there). This establishes that #4559 solved dimension
provenance and splicing, while #4581 still prevents the nested result from being
code-generated.

Three source-level workarounds were tried against the minimal reproducer:

- `.contiguous()` and `.clone()` on the outer tile are optimized as aliases and
  retain the unresolved outer symbol;
- an explicit pointwise materialization (`outer_tile + 0`) reaches runtime but
  is incorrect, returning approximately all ones instead of `x + 1`; and
- `copy_forced` with a tile-local destination fails fake propagation because a
  generated body placeholder is not supplied.

None is a safe decomposition-level workaround. The compiler must register and
transport outer loop variables when generating nested bodies; materializing the
tile is not sufficient and can silently miscompile.

### Blocker 2: map output materialization

On CPU, both complete prototypes contain four copy/clone/stack-style
materialization nodes after `scan` has decomposed to `while_loop`. Current map
mode stacks every step's result and then folds the leading scan dimension into
`out_dim`. A non-leading fold cannot generally be represented as a view and
therefore copies the full output.

The unmerged PyTorch `for_each_tile-fixes` branch adds `scan(out=)`. The
Torch-Spyre `for_each_tile-additions` branch, whose tip is `f55cf788`, builds
`for_each_tile(output=...)` on top of it. This lets each map level write its tile
directly into a preallocated destination and avoids the stack/fold copy.

That API is currently inference-only and rejects grad-enabled use. It also has
not landed in the pinned PyTorch source used by this checkout; a trial
application conflicted in five PyTorch files. It should be integrated and
tested independently rather than mixed into the nested-lowering fix.

### Blocker 3: broadcast attention bias

The standalone `Lk` loop passes with a dense `[B, H, Lq, Lk]` bias. Expanding a
`[B, 1, Lq, Lk]` bias to that shape creates a stride-zero head dimension and
fails in `propagate_spyre_tensor_layouts -> _multi_arg_pointwise_layouts`:

```text
RuntimeError: Incompatible host_size and dim_order
```

This reproduces without nesting, so fixing nested loops will not fix it.
Production attention commonly uses broadcast causal/additive masks. The rewrite
must either preserve lower-rank bias and select dimensions according to its
actual shape, or materialize a dense bias only at the current tile granularity.
A full expanded-bias clone would defeat the memory objective and should be
avoided.

### Constraint 4: ragged `Lk`

`for_each_tile` currently rejects a sliced extent that is not divisible by
`tile_size`:

```text
ValueError: ragged tiles are not supported
```

The merged #4550 implementation and this branch avoid a ragged last block by
selecting an exact divisor no larger than the cost model's requested block.
Other possible policies remain:

1. Pad K and V to a tile multiple and add `-inf` to padded score columns. This
   is the most practical initial implementation and preserves a fixed HOP body
   shape.
2. Select an exact-divisor tile size. This avoids padding but can select a very
   small tile for awkward or prime lengths and regress launch count badly.
3. Extend `for_each_tile` with a ragged final tile. This is the cleanest API but
   conflicts with `scan`'s fixed per-step tensor shape and is the largest
   frontend/compiler change.

The exact-divisor policy is sufficient for correctness, but padding may be a
better follow-up if awkward lengths force a very small divisor.

### Constraint 5: #4549 work-division parity

The #4549 cost model's calibrated fast path returns `num_q_tiles=1` and
`num_head_tiles=1`, then relies on a `work_div` hint to distribute `H`, `Lq`,
and, for MHA, part of `Lk` across cores. The full-HOP branch intentionally has
no `named_dims`, so it currently consumes the selected K/V block size but does
not reproduce that named work division. Mapping those split counts to
additional sequential HOP tiles would not be equivalent. Once nested codegen
works, the compiler needs either HOP-derived work division or an axis-based
work-division interface before performance can be compared fairly with main.

## Recommended implementation sequence

### 1. Make nested lowering correct in isolation

Keep the minimal two-map test and require it to emit two nested `LoopSpec`s.
Then enable the `Lq` map around the `Lk` reduction and require two nested
`LoopSpec`s plus numerical agreement. PR #4559 already handles recursive
placeholder resolution and tile-dimension provenance. The remaining fix must:

- recognize outer counted-loop variables used by an inner body's addresses;
- carry their ranges into `indirect_sizes`/coordinate generation; and
- preserve value correctness rather than hiding the symbol behind a
  materialization.

Do not use the complete SDPA graph as the primary debugger until these two
small cases pass.

### 2. Land destination-backed map mode

Integrate the PyTorch `scan(out=)` work and the Torch-Spyre `output=` frontend.
Add one-map and nested-map tests that assert output aliasing, absence of
stack/fold materializations, correct non-leading `out_dim`, and generated
`LoopSpec` structure. Decide explicitly whether inference-only support is
sufficient for Torch-Spyre SDPA.

### 3. Add attention operand coverage

Before changing the decomposition, cover:

- MHA and native GQA;
- dense, broadcast, causal, and combined masks;
- contiguous and transposed Q/K/V inputs;
- `Lq=1` decode and multi-row prefill;
- output strides required by the fused-attention meta kernel; and
- aligned and ragged `Lk`.

For bias, prefer passing the smallest-rank representation through outer loops
and densifying only an individual score tile if layout propagation requires it.

### 4. Replace production loops without replacing policy

Reuse `_select_sdpa_tiling` to choose tile sizes. Express `B`, `Hkv`, `G`, and
`Lq` as map-mode HOPs and retain #4550's `Lk` online-softmax reduction. Preserve
the current per-block K transpose/normalization, corrected carry recurrence,
mask slicing, and output-stride contract. Add an axis-based equivalent of the
cost model's `work_div`, then use preallocated destinations at every map level.

### 5. Validate correctness and performance before switching the default

At minimum, run the existing SDPA building-block, layout, work-division,
coarse-tile, and model tests. Add generated-source assertions for the exact
four/five-level `LoopSpec` tree and the absence of Python-unrolled K/V bodies.
Benchmark representative Granite and Gemma 4 decode/prefill shapes, including
long KV, and compare:

- numerical error;
- compile time and generated bundle size;
- runtime latency and throughput;
- peak HBM/LX use and materialization count; and
- behavior around every tiling-policy boundary.

Performance runs are not meaningful yet: the full branch cannot produce an
executable, and dropping #4549's work division would compare different
schedules. Only after correctness and work-division parity should the HOP
implementation become the default.

## Reproduction commands

From the experiment worktree:

```bash
env PYTHONPATH=/tmp/torch-spyre-sdpa-fet.8jYqoJ \
  python tests/inductor/test_sdpa_for_each_tile.py -v
```

To expose the minimal nested failure rather than treating it as expected:

```bash
env PYTHONPATH=/tmp/torch-spyre-sdpa-fet.8jYqoJ python -c \
  'import importlib.util; s=importlib.util.spec_from_file_location("fet", "tests/inductor/test_sdpa_for_each_tile.py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.TestSDPAForEachTile().test_two_nested_maps_spyre()'
```

## Decision

The full rewrite remains compiler-enablement work, not a merge-ready SDPA
decomposition patch. The safe partial step—#4550's `Lk` HOP—is already merged.
#4559 removes the original nested-placeholder blocker but does not make nested
maps executable: #4581 still fails codegen, and a materialization workaround
silently produces incorrect values. Keep PR #4551 draft until #4581,
destination-backed map outputs, broadcast bias, and #4549 work-division parity
are resolved and measured.
