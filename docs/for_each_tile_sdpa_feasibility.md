# SDPA `for_each_tile` feasibility report

Last updated: 2026-09-21

## Executive summary

This branch contains the complete proposed rewrite of Spyre SDPA tiling as
nested `for_each_tile` operations. It has been rebased onto the exact head of
PR #4705, itself rebased onto the latest `upstream/main` used for this test.

PR #4705 is a substantial improvement over the previous baseline:

- a minimal two-level map nest now compiles and executes correctly on Spyre;
- the standalone `Lk` online-softmax loop still passes for MHA and native GQA;
- a stride-zero broadcast attention bias now works with that standalone loop;
  and
- ordinary Granite GQA decode and prefill production tests pass with the full
  decomposition.

The complete rewrite is still not safe to merge. Every remaining failure
contains an outer map around an inner carry-bearing `Lk` loop:

- a minimal `Lq -> Lk` prototype compiles and runs but returns corrupt values;
- a complete four-level MHA prototype also runs but returns wrong values;
- a complete five-level GQA prototype fails earlier with a cycle in Inductor's
  scheduler/memory-planning dependency graph; and
- the production Granite four-by-four `Lq`/`Lk` tiling test runs but returns a
  wrong answer.

The wrong-answer cases are consistent with the carry-bearing nesting class
tracked by #4701. The five-level dependency-cycle failure is a separate symptom
and may need its own compiler fix. PR #4705 deliberately does not claim that
carry-bearing nested `for_each_tile` is end-to-end correct.

The recommendation is to keep #4551 as a draft. PR #4705 makes nested map mode
usable, but production SDPA needs an outer map composed with an inner reduction
whenever both `Lq` and `Lk` require more than one tile.

## Experiment baseline

- Repository: `torch-spyre/torch-spyre`
- Upstream base: `upstream/main` at
  `fdb52a0498732b421ed43d00bdb7c617dd5ae7af`
- PR #4705 head: `0f67e59b81746282c93853e27dc187336eee5981`
- PR #4705 rebased onto that main:
  `2560bb727d62053a76c1868c7a92cc99259266f9`
- Experiment branch: `codex/sdpa-for-each-tile-experiment`
- Prototype: `tests/inductor/test_sdpa_for_each_tile.py`

The native extension was rebuilt from this checkout before testing.

## SDPA structure

The production decomposition chooses a tiling plan with
`_select_sdpa_tiling`. Main represents coarse `B`, `H`, `G`, and `Lq` tiling
with hint scopes and represents the `Lk` online-softmax reduction with
`for_each_tile`.

The full-HOP branch expresses all of those loops directly:

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

For GQA, Q and bias have logical shape `[B, Hkv, G, Lq, ...]`; K and V
retain a unit group dimension and are invariant at the `G` level. The
innermost loop implements stable online softmax:

```text
scores = Q @ K_tile.T * scale + bias_tile
M_next = max(M, amax(scores, dim=-1))
correction = exp(M - M_next)
P = exp(scores - M_next)
l_next = l * correction + sum(P, dim=-1)
O_next = O * correction + P @ V_tile
result = O_final / l_final
```

The rewrite retains the current tiling cost model, exact-divisor K/V tile
selection, packed-key rebasing for unaligned physical rows, native GQA without
repeating K/V, sparse accumulator construction, mask slicing, and the output
stride contract. One-trip map and carry levels are called directly rather than
lowered to a loop. No `named_dims` hints remain in the rewritten decomposition.

## Results

### Focused prototype suite

After updating stale expected-failure annotations, the focused suite reports:

```text
Ran 9 tests in 38.214s
OK (expected failures=2)
```

| Case | Backend | Result |
| --- | --- | --- |
| Full MHA `B/H/Lq/Lk` nest | CPU eager and `torch.compile` | Pass; four `while_loop`s captured |
| Full GQA `B/Hkv/G/Lq/Lk` nest | CPU eager and `torch.compile` | Pass; five `while_loop`s captured |
| Standalone MHA `Lk` carry loop, dense bias | Spyre | Pass; one `LoopSpec` |
| Standalone native-GQA `Lk` carry loop, dense bias | Spyre | Pass; one `LoopSpec` |
| Standalone MHA `Lk` carry loop, expanded broadcast bias | Spyre | Pass; one `LoopSpec` |
| Two nested map-mode loops | Spyre | Pass; two `LoopSpec`s |
| `Lq` map around `Lk` carry loop | Spyre | Expected failure: wrong output |
| Full five-level GQA nest | Spyre | Expected failure: scheduler dependency cycle |

The simple nested-map test's maximum absolute difference was `0.0078125`,
which is within the same `atol=0.1, rtol=0.1` FP16 tolerance used by PR #4705's
Spyre end-to-end tests. Its previous default `assert_close` tolerance was too
strict for device FP16 arithmetic.

### Production decomposition checks

| Test | Result |
| --- | --- |
| `test_sdpa_lk_uses_for_each_tile` | Pass |
| Granite GQA decode, `Lq=1`, `Lk=128` | Pass |
| Granite GQA prefill, `Lq=128`, `Lk=128` | Pass |
| Forced four-by-four Granite GQA, `Lq=Lk=256` | Wrong output |

The forced four-by-four test produced `131079 / 1048576` mismatched elements
(`12.5%`) in two consecutive runs, with maximum absolute difference `3.890625`
at the test's `atol=0.2, rtol=0.2` threshold. It proves that getting through
compilation and execution is not enough when a production SDPA graph contains
both query mapping and the K/V carry loop.

The ordinary decode and prefill cases do not contradict this result. Their
sequence extents fit one selected query and K/V tile, so those levels take the
one-trip direct-call path. They validate PR #4705's map-only nesting in the
remaining GQA head/group levels, not map-plus-carry nesting.

### Additional four-level MHA probe

A Spyre probe using `B=2`, `H=2`, `Lq=32`, `Lk=256`, `D=128`, query tiles of
16, and K/V tiles of 128 compiled and executed. It remained numerically wrong:
`5305 / 16384` elements differed by more than `0.1`, with maximum absolute
difference `0.4942207`. This shows the carry-composition problem is not specific
to GQA's extra group level.

## Remaining blocker 1: map plus carry gives wrong results (#4701)

The smallest SDPA-specific reproducer is one `Lq` map around one `Lk`
online-softmax carry loop. It now completes tracing, splicing, scheduling,
code generation, and device execution. In the observed run it mismatched
`8197 / 16384` elements (`50.0%`) and produced non-finite values, including an
infinite maximum absolute difference.

This is the same composition class as #4701: an outer map whose body contains a
carry-bearing inner loop. PR #4705's own report explicitly leaves that class as
follow-up work because its nested split-M/split-K fixture can produce
nondeterministic corruption suggestive of stale or aliased HBM storage or an
incorrect carry read/write schedule.

The full MHA and forced production GQA results confirm that SDPA reaches this
unresolved path. The next investigation should start with HBM-pool liveness and
the schedule of each carry snapshot, update, and next-iteration read, using the
minimal `Lq -> Lk` reproducer before returning to a full model.

## Remaining blocker 2: deep GQA nesting creates a dependency cycle

The five-level GQA prototype does not reach code generation. Inductor's
`reorder_for_peak_memory` validation reports a cycle between an outer mapped
operation and a synthetic `while_loop_carry_snapshot` dependency generated for
the nested `Lk` carry.

This differs from the previous #4581 failure: no unregistered outer-loop
indirect symbol is reported. It also differs from the numerical #4701 symptom
because execution never starts. The full dependency path is long, but its two
endpoints are stable and include the synthetic carry snapshot, so the first
place to inspect is dependency construction when a carry loop is nested under
several map levels.

PR #4705 also documents #4706, an OS-5 symbol-consistency gap that can affect
other carry-bearing nested shapes. The SDPA probes above did not stop at that
error: the shallow cases reached execution and the deepest case reached the
memory-planning cycle.

## Remaining design work

### Destination-backed map outputs

On CPU, both complete prototypes contain four copy/clone/stack-style
materialization nodes after `scan` decomposes to `while_loop`. Map mode stacks
each step's result and folds the leading scan dimension into `out_dim`; a
non-leading fold can require a full-output copy. Destination-backed
`scan(out=)` / `for_each_tile(output=)` is still needed to eliminate these
writes and should be integrated independently of the carry-correctness fix.

### Ragged `Lk`

`for_each_tile` requires the sliced extent to be divisible by `tile_size`.
This branch chooses an exact divisor no larger than the cost model's requested
block, which is correct but can select a small tile for awkward or prime
sequence lengths. Padding K/V plus a `-inf` mask for padded score columns may be
a better production policy; native ragged tiles would require a larger frontend
and compiler change.

### Work-division parity

The cost model's calibrated fast path can use `work_div` to distribute named
`H`, `Lq`, and MHA `Lk` dimensions across cores. The full-HOP branch
intentionally has no `named_dims`, so it needs HOP-derived work division or an
axis-based equivalent before runtime performance can be compared fairly with
main.

## Recommended sequence

1. Fix and repeatedly stress the minimal `Lq` map around the `Lk` carry,
   including clean-cache runs to detect #4701-style nondeterminism.
2. Fix the deep-nest scheduler dependency cycle using the five-level GQA
   prototype as the regression test.
3. Require both the MHA and GQA device prototypes, plus the forced four-by-four
   production test, to pass numerically.
4. Add destination-backed map outputs and an axis-based work-division
   interface.
5. Only then benchmark compile time, runtime, HBM/LX use, and generated bundle
   size against the hints-based production implementation.

## Reproduction

Build the native extension and run the focused suite:

```bash
python setup.py build_ext --inplace
python tests/inductor/test_sdpa_for_each_tile.py -v
```

Run the production checks:

```bash
python -m pytest -q \
  tests/inductor/test_building_blocks.py::TestBuildingBlocks::test_sdpa_lk_uses_for_each_tile \
  tests/inductor/test_building_blocks.py::TestBuildingBlocks::test_granite_gqa_decode_with_finite_mask \
  tests/inductor/test_building_blocks.py::TestBuildingBlocks::test_granite_gqa_prefill_with_finite_broadcast_mask \
  tests/inductor/test_building_blocks.py::TestBuildingBlocks::test_granite_gqa_prefill_four_by_four_sequence_tiling
```

## Decision

PR #4705 resolves the map-only nested-HOP blocker and the earlier broadcast-bias
failure. It does not yet make nested carry-bearing SDPA correct. Keep #4551 in
draft and retain main's hints-based outer loops with the merged `Lk`-only HOP
until the wrong-answer and dependency-cycle failures above are fixed.
