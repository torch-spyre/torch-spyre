# SDPA `for_each_tile` feasibility report

Date: 2026-09-15

## Executive summary

Rewriting the full Torch-Spyre SDPA tile nest in terms of `for_each_tile` is
conceptually sound, but it is not ready to replace the production
implementation. The online-softmax loop over `Lk` works on Spyre, including
native GQA with dense bias, and the full four-level MHA and five-level GQA
programs work and compile on CPU. The first nested `for_each_tile` combination
does not compile on Spyre, however. A minimal two-map program fails in the same
place as an `Lq` map containing an `Lk` reduction and the complete GQA nest.

There are three other gaps to resolve before production use:

1. Map-mode loops currently materialize scan outputs. The prototypes contain
   four copy/stack-style materialization nodes. The unmerged `scan(out=)` and
   `for_each_tile(output=)` branches are designed to eliminate these writes.
2. A stride-zero broadcast attention-bias view fails Spyre layout propagation,
   even with only the otherwise-working `Lk` loop.
3. `for_each_tile` requires exact tile divisibility. Production SDPA accepts a
   shorter final K/V block, so arbitrary `Lk` needs padding and masking, an exact
   divisor policy, or frontend support for a ragged final tile.

The recommendation is therefore to land only the standalone `Lk` replacement
for now, as proposed separately in PR #4550. Do not replace the outer hint
scopes yet. Fix and test nested HOP lowering first, then add destination-backed
map outputs and handle broadcast bias before making the complete HOP nest the
production implementation.

## Experiment baseline

- Repository: `torch-spyre/torch-spyre`
- Base: current `upstream/main` at
  `fc5b1cf79c23efa648bdcb9fe589390bdbfd8768`
- Experiment branch: `codex/sdpa-for-each-tile-experiment`
- Worktree: `/tmp/torch-spyre-sdpa-fet.8jYqoJ`
- Prototype: `tests/inductor/test_sdpa_for_each_tile.py`

PR #4518 is already in this base as commit
`3b5381f0f083a9bfaddb9319e093c01a4dab0443`. Its reduction-symbol preservation
fix is useful to the existing hint-based path, but it does not fix nested
`for_each_tile` lowering.

This draft branch includes the proposed production replacement so the team can
inspect and iterate on the complete shape. It is intentionally not ready to
merge: the nested-lowering failures below are reproduced by that implementation.
A speculative compiler modification used to continue diagnosis past the first
failure was reverted.

## Original SDPA structure

Before the standalone `Lk` refactor, `spyre__sdpa_overrideable` chooses a
tiling plan with
`_select_sdpa_tiling` and expresses outer coarse tiling with nested
`spyre_hint` scopes:

- a batch tile scope;
- a head tile scope for MHA;
- query-sequence tiling and, for eligible shapes, named work division;
- explicit rank-5 `[B, Hkv, G, Lq, D]` operands for native GQA; and
- a Python `for` loop that slices K and V into `Lk` blocks.

The Python `Lk` loop implements stable online softmax with three SSA values:
the running maximum `M`, denominator `l`, and unnormalized output accumulator
`O`. The loop is deliberately split into bounded groups because the generated
bundle size scales with the `Lq`-tile by unrolled-`Lk`-block product. The current
limit is 16 tile pairs per loop group.

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
This code is included to make the complete lowering shape reviewable; it is not
expected to compile for shapes that exercise two or more HOP levels.

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
Ran 9 tests in 46.525s
OK (expected failures=4)
```

The four expected failures are intentional regression tests for nested maps,
the `Lq`/`Lk` nest, the complete GQA nest, and broadcast bias.

### Blocker 1: nested HOP lowering

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

It fails during `splice_while_loops -> coarse_tile_pre_stickify ->
_plan_read_copies -> _full_buffer_read_deps`:

```text
RuntimeError: Failed to find buffer matching name
while_loop_body_graph_0_0_arg3_1
```

The same failure occurs for the first SDPA-specific nest (`Lq` map around an
`Lk` reduction) and for the full five-level GQA program. This establishes that
the first blocker is generic nested-HOP lowering, not the online-softmax
formula.

An experimental alias-composition change allowed lowering to pass this stale
inner-placeholder lookup, but it then exposed independent failures:

- `unexpected mutation layout` during layout finalization;
- a nested map output retaining `FixedLayout` where work division expects a
  `FixedTiledLayout`; and
- a rank-5 online-softmax layout whose selected stick is incompatible with the
  operation.

The experiment was reverted because resolving the first name lookup alone does
not produce a correct nested implementation. Nested lowering needs an
end-to-end layout and mutation solution rather than a narrow lookup workaround.

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

### Blocker 4: ragged `Lk`

`for_each_tile` currently rejects a sliced extent that is not divisible by
`tile_size`:

```text
ValueError: ragged tiles are not supported
```

The production Python loop uses `end = min(start + block_size, Lk)`, so its last
block may be shorter. Reasonable options are:

1. Pad K and V to a tile multiple and add `-inf` to padded score columns. This
   is the most practical initial implementation and preserves a fixed HOP body
   shape.
2. Select an exact-divisor tile size. This avoids padding but can select a very
   small tile for awkward or prime lengths and regress launch count badly.
3. Extend `for_each_tile` with a ragged final tile. This is the cleanest API but
   conflicts with `scan`'s fixed per-step tensor shape and is the largest
   frontend/compiler change.

Padding should be preferred for a first production implementation, while
retaining a fast no-padding path for already aligned lengths.

## Recommended implementation sequence

### 1. Make nested lowering correct in isolation

Keep the minimal two-map test and require it to emit two nested `LoopSpec`s.
Then enable the `Lq` map around the `Lk` reduction and require two nested
`LoopSpec`s plus numerical agreement. The fix must cover:

- recursive placeholder/alias resolution after inner `WhileLoop` splicing;
- output and mutation layouts across nested bodies;
- promotion of nested map destinations to `FixedTiledLayout` before work
  division consumes them; and
- rank-5 GQA layout selection.

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
`Lq` as map-mode HOPs and `Lk` as the online-softmax reduction. Preserve the
existing per-block K/V `.contiguous()` normalization and mask slicing. Use
preallocated destinations at every map level.

The original 16-pair bundle guard exists because Python-unrolled `Lk` multiplies
code size by the `Lq` trip count. A real nested `Lk` `LoopSpec` may remove that
specific source-size problem, but the guard should remain until generated code,
DXP compile time, and runtime launch behavior demonstrate that it is no longer
needed.

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

Only after those results are neutral or better should the HOP implementation
become the default; retaining the current decomposition behind a temporary
fallback would make rollout safer.

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

The full rewrite should continue as compiler-enablement work, not as a
merge-ready SDPA decomposition patch yet. A standalone `Lk` HOP is proven
viable inside the existing outer hint-generated tiling and is proposed in PR
#4550. Replacing those outer hints with map-mode HOPs creates the unsupported
nested-loop condition. The safe partial production step is therefore the `Lk`
replacement; the outer `B`, `Hkv`/`H`, `G`, and `Lq` replacement must wait for
nested lowering and the remaining map-mode issues.
