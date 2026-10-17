# CP-SAT Layout Solver: Nondeterminism

## Background

The LX scratchpad planner uses [OR-Tools CP-SAT](https://developers.google.com/optimization/reference/python/sat/python/cp_model)
to jointly solve LX buffer placement and work-division (core splits) for each compiled op.
CP-SAT runs a portfolio of parallel search workers by default. When multiple workers find
solutions with equal cost, the one that finishes first wins — and that race is timing-dependent.
This means that two runs of the same model can produce different op-spec `.py` files (keyed by
content hash), breaking cache reuse and making output non-reproducible.

## Demonstrating the Nondeterminism

We ran `test_vlm_generate_spyre[ibm-granite/granite-vision-4.1-4b]` 6 times with a cold cache:

```bash
cd $HOME/dt-inductor/hf-adapters && \
TORCHINDUCTOR_COMPILE_THREADS=1 \
  python3 -m pytest \
  tests/spyre/test_vlm_e2e_spyre.py::test_vlm_generate_spyre[ibm-granite/granite-vision-4.1-4b] \
  -s -v -v 2>&1 | tee o_run1
```

`TORCHINDUCTOR_COMPILE_THREADS=1` removes a separate source of nondeterminism from Inductor's
parallel compilation, isolating the CP-SAT variation. After each run, compare op-spec filenames:

```bash
find $TORCHINDUCTOR_CACHE_DIR -name "*.py" | xargs grep -l "AOT ID" | xargs -I{} basename {} | sort
```

Op-spec filenames are content-addressed hashes of the file contents — a different filename
means different generated code. Each run produces 22 op-spec `.py` files. Of those, 17 are
stable across all 6 runs. The remaining 5 ops vary: 24 distinct filenames appear across the 6
runs for what should be the same ops. The model passes every time. Diffing two op-specs for the same AOT ID from different runs shows
the difference is in LX buffer placement offsets — the solver chose a different-but-equally-valid
packing of buffers in LX scratchpad memory, producing different generated code and therefore a
different hash. For example:

```
< allocation={'lx': 103424}
> allocation={'lx': 98304}
```

| File (truncated) | Runs seen in |
|---|---|
| `cuh6upui...` | 5 of 6 |
| `cxdxm7wg...` | 2 of 6 |
| `cr3avxbh...` | 2 of 6 |
| `czrkr2pg...` | 1 of 6 |
| `czqefl6e...` | 1 of 6 |
| `cxpydp7v...` | 1 of 6 |
| `ctnqbrbec...` | 1 of 6 |
| *(18 more singletons)* | 1 of 6 |

## A Strawman Approach to Determinism

Setting `random_seed=0` makes each individual worker deterministic, but does not prevent the
portfolio race. Reducing to one worker (`num_search_workers=1`) would fix it but discard the
performance benefit of the portfolio.

Instead, we add a single extra solve stage at the end of `_run` in
`torch_spyre/_inductor/scratchpad/ilp_solver_ortools.py`. The existing lexicographic ladder
locks each stage's optimum as a hard constraint before moving to the next. After the final
stage, those constraints are still in place — they bound the feasible region to solutions at
least as good as the ladder's optima. We then minimize a lightweight tiebreak term over that
constrained space. Because the tiebreak has a unique minimum within it, CP-SAT converges on
the same solution every run regardless of which worker finds it first:

```
tiebreak = sum(t.offset for all buffers, sorted by name)
         + sum(rank_var for multi-candidate division buffers, sorted by name)
```

- **Offset sum** — among equally-valid placements, prefers packing buffers toward offset 0.
- **Division rank** — for buffers with multiple core-split candidates, assigns each candidate
  a rank based on its sorted `(dim_name, split_value)` tuple, then minimizes the sum of chosen
  ranks. This is expressed as a CP-SAT `add_element` constraint so the solver can reason over
  it natively.

Because all prior objectives are locked as constraints before this stage, the tiebreak can never
trade quality (spills, parallelism, tile count) for determinism — it only breaks ties.

The ladder already runs up to 5 sequential solves per op; the tiebreak adds one more, for up
to 6 total. This is a temporary solution — the right fix is to collapse the ladder into a
single lexicographic objective that CP-SAT can solve in one pass, which would also eliminate
the portfolio race without needing a tiebreak at all.

The tiebreak is gated by `CP_SAT_TIEBREAK` (default `0`); the corresponding config entry is
`cp_sat_tiebreak` in `torch_spyre/_inductor/config.py`.

## Demonstrating the Strawman

Run with `CP_SAT_TIEBREAK=1`:

```bash
cd $HOME/dt-inductor/hf-adapters && \
TORCHINDUCTOR_COMPILE_THREADS=1 CP_SAT_TIEBREAK=1 \
  python3 -m pytest \
  tests/spyre/test_vlm_e2e_spyre.py::test_vlm_generate_spyre[ibm-granite/granite-vision-4.1-4b] \
  -s -v -v 2>&1 | tee o_run1
```

Running the same 6 cold-cache runs this way, all 22 op-spec filenames are identical across
every run — zero nondeterministic files. Diffing the actual file contents (not just filenames)
confirms they are byte-for-byte identical:

```
$ diff on_run1/c4r2hs....py on_run2/c4r2hs....py  # no output — identical
$ diff on_run1/c4r2hs....py on_run6/c4r2hs....py  # no output — identical
```
