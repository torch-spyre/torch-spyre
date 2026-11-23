# Quality results

Quality benchmarks measure correctness: task accuracy, parity with a reference output, numerical error, rather than
speed. They use the same catalog, run and verdict machinery as performance benchmarks.

**Status: Planned.** The design is written; the first slice records scores that existing checks already produce. The
dashboard's quality tab shows what is ingested today.

## What you can expect to see

- A **score per benchmark** (for example an accuracy or a similarity), the number of samples, and the seed.
- A **verdict** with the same five statuses; for quality the "worse" direction is a lower score.
- **Parity against a reference**: for models where a reference output exists (computed offline on another platform), the
  result shows the distance to it. The reference is an approved target ([References](references.md)), not a baseline.
- An advisory check named like `Spyre Quality (x86_64)` next to the performance one; it never gates.

## Reading a quality verdict

1. A drop is meaningful only relative to the sampling noise: stochastic evaluations are run with several repeats or
   seeds, and the band uses their spread. One seed and one repeat gives `few_repeats` (suspected).
2. Check the dataset and its version first. A changed dataset is a new benchmark by identity rules, so a "drop" at the
   same name with a new dataset id is not a regression.
3. Check determinism settings (seed, sampling) on the run page; two runs with different settings are not comparable.

## When it fails

| Symptom | Fix |
|---|---|
| quality tab empty | quality results are not yet produced for that architecture or model |
| score changed with no code change | the dataset or seed changed; compare the run properties |
| parity missing | no approved reference for this case yet; submit one |

## Who to ask

The quality benchmark's owner; the platform maintainers for pipeline questions.
