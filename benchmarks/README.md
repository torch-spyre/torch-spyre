# Benchmark definitions

Definitions for [`spyre-bench`](../extensions/spyre-bench/README.md) (`bench/v1alpha1`): what torch-spyre
benchmarks, not where or when they run. The catalog is advisory; nothing here gates a merge or a check.

| File | Contents |
|---|---|
| `project.yaml` | Project, the `ops` Suite, the `one-device` Environment request and the named Selections |
| `systems.yaml` | the operator library under test |
| `ops/<operation>.yaml` | one Benchmark per operation name; the file name is the Benchmark name |
| `legacy-ids.golden.tsv` | ids the ingest derives from each Benchmark's `legacy.*` annotations (synthetic rows) |
| `tools/check_legacy_ids.py` | recomputes that table with `spyre_clickhouse_ingest` and compares |

## Commands

```bash
make bench-validate                      # schema, packs, cross-resource rules
make bench-explain CASE=matmul           # one case with its ids
make bench-check-ids                     # legacy id inputs against the ingest derivation
make bench-list bench-plan bench-run bench-smoke bench-report
```

These targets require the `spyre-bench` package. Until torch-spyre#6298 is merged, install it from that PR
branch; afterwards from `main`:

```bash
uv pip install 'spyre-bench @ git+https://github.com/torch-spyre/torch-spyre@main#subdirectory=extensions/spyre-bench'
```

When the engine is not installed, or lacks the verb a target needs, the target prints a notice and exits 0. Setting
`BENCH_REQUIRED` to any non-empty value (including `0`) turns that into exit 2. `BENCH_ARCH`, `BENCH_CARDS` and
`BENCH_QUEUE` are passed through to the engine unchanged; the Makefile never branches on them.

## Identity

The ingest keeps writing every benchmark id (`BenchmarkId`: component, operation name, tags and the
`record_type`, `config_name`, `input_shapes`, `run_mode`, `kernel_name`, `is_total` discriminators, with compiled
kernel names reduced to `<stem>@<rank>`). The `legacy.*` annotations record the discriminators a Benchmark
declares, so editing one is visible as a golden-table failure. Shapes and kernel names come from the runner at
run time. The golden rows are synthetic: empty tags, no `config_name`, one fixed shape and made-up kernel
names, matching the shape of today's op rows in `results.py`; model rows and real shapes are not covered. The engine's own `benchmark_id` hashes different keys and is not the ingest id.

## Cadence

`schedule.*` annotations on a Selection state the preferred cadence (`nightly`, `weekly`), the fallback, a
priority and an optional runtime estimate. They move to a `schedule` block when the engine defines one.

## Runner

Cases run through the runner the Project names, `kernel-bench`, a neutral runner name.
