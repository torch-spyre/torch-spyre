# spyre-bench

A catalog engine for benchmarks: it validates `bench/v1alpha1` YAML catalogs, expands benchmark templates into cases,
derives their identity, and plans runs with reuse, invalidation and qualification rules. It is a library plus a small
CLI; it does not execute benchmarks.

> **Concepts, kinds and naming:** [`docs/README.md`](docs/README.md).

## Why identity is derived

`benchmark_id` is a uuid5 over the same id namespace as `extensions/clickhouse-ingest`, so the engine and the ingest
agree on the id for the same benchmark without coordinating. A test cross-checks `engine.benchmark_id` against
`spyre_clickhouse_ingest.identity.BenchmarkId`.

## Install

Not published; install from the checkout.

```bash
uv pip install extensions/spyre-bench                 # PyYAML + regex
uv pip install "extensions/spyre-bench[jsonschema]"   # full Draft 2020-12 validation
```

Without `jsonschema` the engine falls back to a stdlib validator that covers the schema subset the core schemas use.

## Use

```bash
spyre-bench validate examples/llm-demo --central examples/central
spyre-bench validate examples/postgres-demo --pack examples/packs/postgres-example
spyre-bench explain serve_demoa_tp4_c5_all --catalog examples/llm-demo --central examples/central
spyre-bench gen-schemas --check
```

`validate` exits 0 when valid and 2 otherwise; `gen-schemas --check` exits 1 when the schemas are stale and `gen-schemas` writes only inside a source checkout unless you pass `--out DIR`. `explain` shows a case with every reference resolved: identity inputs,
the Environment request, series and reuse keys, and the plan record. `gen-schemas` regenerates
`spyre_bench/core/schemas/` from `spyre_bench/gen_schemas.py`; edit the generator, never the JSON.

From Python:

```python
from spyre_bench import engine

cat = engine.Catalog(["examples/llm-demo"], [], ["examples/central"])
assert not cat.errors
leg = {"arch": "x86_64", "devices": {"spyre/pf": 8}, "nodes": 1, "reference_node": True}
plan = engine.plan(cat, {"purpose": "pr", "selection": {"tier": "sanity"}}, leg)
```

## Layout

| path | contents |
|---|---|
| `spyre_bench/engine.py` | registry, catalog validation, case expansion, identity, plan, reuse, invalidation, qualification, validity |
| `spyre_bench/validate.py`, `explain.py`, `gen_schemas.py`, `cli.py` | the CLI verbs |
| `spyre_bench/core/` | the core pack and the generated JSON Schemas (one per authored kind, plus tool-output shapes) |
| `spyre_bench/packs/` | built-in packs: `llm-serving`, `spyre`, `kernel`, `comms` |
| `spyre_bench/demo.py` | builds `examples/results/` from synthetic samples (`python -m spyre_bench.demo --check`) |
| `examples/` | synthetic catalogs, a third-party pack, requests and tool-output samples |
| `tests/` | the engine test suite |

## Tests

```bash
cd extensions/spyre-bench
uv run --with pytest --with jsonschema python -m pytest -q tests      # full
uv run --with pytest python -m pytest -q tests                         # stdlib validator fallback
uv run python -m spyre_bench gen-schemas --check                       # generated schemas current
```

The tests read `examples/`, which is not part of the installed wheel: run them from a source checkout (they skip with a
reason when `examples/` is absent).

All example values are synthetic; no measurement here is real.
