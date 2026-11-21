# Benchmark definitions

Definitions for [`spyre-bench`](../extensions/spyre-bench/README.md) (`bench/v1alpha1`): what torch-spyre
benchmarks, not where or when they run. The catalog is advisory; nothing here gates a merge or a check.

| Path | Contents |
|---|---|
| `project.yaml` | Project, the `ops` Suite, the `one-device` Environment request and the named Selections |
| `systems.yaml` | the operator library under test |
| `ops/<operation>.yaml` | one Benchmark per operation, expanded over input shapes into cases (`<operation>_<shape token>`); `pointwise_add` is the sample |
| `ingest-ids.golden.tsv` | ids the ingest writes for each case's report, kernel and total rows |
| `tools/` | the case runner, the identity and coverage checks, the ingest collector and a card-free stub |
| `tests/` | pytest suite for the ingest write path and the bench-directory ingest |

## Commands

```bash
make bench-validate                              # schema, packs, cross-resource rules
make bench-explain CASE=pointwise_add_prefill    # one case with its ids
make bench-check-ids                             # identity blocks against the ingest derivation
make bench-list bench-plan bench-run bench-verdict bench-smoke bench-report
```

These targets need the `spyre-bench` package:

```bash
uv pip install 'spyre-bench @ git+https://github.com/torch-spyre/torch-spyre@main#subdirectory=extensions/spyre-bench'
```

Without it a target prints the install command and exits 2; `BENCH_OPTIONAL=1` turns that into a notice and exit 0
(the `bench-validate` pre-commit hook uses it).

## Adding an operation

Copy `ops/pointwise_add.yaml`, change the name, the `op` and the shape tokens, add the expected ids to
`ingest-ids.golden.tsv` with `make bench-check-ids` (an intended id change reruns with `--update`), and run
`make bench-validate`.
