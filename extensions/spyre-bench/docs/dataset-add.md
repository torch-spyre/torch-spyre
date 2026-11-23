# Adding a dataset

One command stages the data in the dataset store and writes its Dataset and Asset definitions. The catalog only ever
sees an opaque slug (`slot-NN`); the real name lives in the store's `dataset.json` (see "Dataset slugs" in
[README.md](README.md)).

## The recipe

```bash
export STORE_BASE_URL=... STORE_DATASETS_PATH=... STORE_TOKEN=...   # the Store's variables; values stay in your shell

spyre-bench dataset add ./my-data.jsonl --display-name "My Dataset" --license internal   # dry run: prints the plan
spyre-bench dataset add ./my-data.jsonl --display-name "My Dataset" --license internal --apply
```

`--apply` prints the lines to paste into a Benchmark or Workload:

```yaml
inputs:
  - name: data
    dataset: slot-07
```

Then `spyre-bench dataset check` re-verifies the store against the catalog, and `spyre-bench dataset list` shows what
exists and which Benchmarks use it.

## What it does

- Hashes the file, or every file of a directory in a fixed order (a directory gets a `SHA256SUMS` manifest, and the
  manifest's digest is the dataset's digest).
- Takes the next free `slot-NN` below 90 from the store listing (the store decides, never a local file), stages
  `<slug>/` and writes `<slug>/dataset.json` last. On a directory store the folder appears whole or not at all.
- Appends a Dataset to `datasets.yaml` and an Asset to `assets-datasets.yaml` in `--benchmarks-dir`, and a `Store`
  entry to `stores.yaml` if the named `--store` (default `datasets`) is missing. Then it validates the catalog.
- A single file is stored as `data.<ext>`, so the source file name never reaches a definition.

Options: `--public --license L` (the display name may be shown by tools; a private dataset's name is never printed),
`--slug slot-NN`, `--store NAME`, `--subset NAME=GLOB` (repeatable; a named subset of the files), `--role` (default
`trace`), `--central`, `--pack`, `--max-bytes`.

## Safe to repeat, hard to misuse

- Adding the same data again is a no-op. A slug already holding different data, or the same data with different
  metadata, is refused and nothing is changed.
- Symlinks, names outside `[A-Za-z0-9._-]`, a top-level `dataset.json` or `SHA256SUMS`, and archives with `..`,
  absolute or link members are refused. The default cap is 20 GiB and 10000 files.
- `--apply` needs the store credential from the environment (never from a flag), and nothing printed contains it.
- An https store has no atomic create: a claim marker narrows concurrent adds for one slug but cannot close the race.
  Run one add at a time against an https store; `dataset check` finds a damaged slug. A failed https add can leave
  uploaded files under its slot, which then counts as taken: remove them by hand.
- If the catalog write or its validation fails after the store was published, the catalog files are restored; rerun
  to finish (the store entry is reused).

Exit codes: 0 ok, 2 refused or invalid (messages name the slug only), 5 the store failed.
