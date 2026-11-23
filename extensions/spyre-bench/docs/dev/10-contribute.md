# 10. Contribute: pull requests, checks, versioning

A change to benchmark definitions is a change to the history of a number. The checks below exist so that a reviewer can
see, from the pull request alone, which series start, stop or continue.

## Where a change goes

| You change | Repository and path | Reviewed for |
|---|---|---|
| a Benchmark, System, Workload, Selection, Dataset | the repository that owns the code under test, `benchmarks/` | the identity diff, cost hints, the `smoke` Selection |
| an Environment, NodeClass, ValidityPolicy shared by several repositories | the shared (central) catalog | label and fingerprint effects, policy owners |
| the engine, a built-in pack, the core schemas | `extensions/spyre-bench/` here | tests, generated schemas, ids pinned by golden files |
| the ingest parsers, views, schema | `extensions/clickhouse-ingest/` here | real-engine tests, sandbox proof, the conformance table |
| these guides and their examples | `extensions/spyre-bench/docs/dev/`, `examples/docs-*` | the snippet tests pass |

Open an issue first for a large change, as the repository's contributing guide asks.

## Pull-request checklist for a catalog change

Copy this into the description and tick what applies.

- [ ] `spyre-bench validate <catalog>` passes with the `--central` and `--pack` the repository uses.
- [ ] `spyre-bench validate <catalog> --base-dir <merge-base tree>` was read; every "identity changed" warning is
      intended and listed below with the series it restarts.
- [ ] New or renamed Benchmarks: `previous_names` set for a rename; the `smoke` Selection still names an existing case.
- [ ] `spyre-bench explain <case>` was read for one case of each new Benchmark.
- [ ] `spyre-bench plan` plans the Selections I touched; skip reasons are the ones I expect.
- [ ] The command ran once (stub or real); a `metrics.json` was written; no secret is in `run.command`.
- [ ] Metrics are registered with unit, direction, category; the ingest path stores them or a follow-up is linked.
- [ ] Cost hints (`est_hours`, `run.est`, `run.timeout`) are set.
- [ ] No `schedule`-like field: cadence belongs to the invoker.
- [ ] Nothing public carries a real host, measured number, score, threshold, credential or internal system name.
      Real benchmark, metric and model names are fine; datasets are named by slug.
- [ ] Pre-commit passes on every changed file.
- [ ] A reviewer who did not write the change read the identity diff.

## What the checks run

| Check | What it guards | Run it |
|---|---|---|
| pre-commit | ruff and ruff-format on Python, pymarkdown on `.md`, actionlint on `.github/workflows`, mypy, DCO sign-off, no spaces in filenames, the `regex` import rule | `pre-commit run --all-files` (or `--files <paths>`) |
| the engine tests | schemas, packs, identity, planning and reuse, run, verdict, about 90 negative cases | `uv run --with pytest --with jsonschema python -m pytest -q tests` in `extensions/spyre-bench` |
| docs examples and snippets | every catalog behind these guides plans and runs; every marked block matches its example or the engine | `python -m pytest -q tests/test_docs_examples.py tests/test_docs_snippets.py` |
| generated schemas | `core/schemas/*.json` equals what `gen_schemas.py` writes | `spyre-bench gen-schemas --check` (fix by editing the generator, then `gen-schemas`) |
| engine ids | the ids of every example without an `identity` block are pinned | `tests/golden/engine-ids.tsv`, rewritten from `engine_rows()` in `tests/test_identity.py` only for a deliberate change |
| ingest conformance | the ids an `identity` block derives equal the ids the real ingest writes for the same rows | `PYTHONPATH=../clickhouse-ingest python tests/refresh_conformance.py --check` (from `extensions/spyre-bench`) |
| a repository's identity check | its own `benchmarks/` blocks against the ingest, plus its golden table | its README names the target (for example a `bench-check-ids` make target in a product repository) |

YAML in `extensions/spyre-bench` is excluded from the repository's YAML formatter on purpose: the engine reads YAML 1.1,
and a formatter that strips quotes turns `"on"` or `"1024"` into a boolean or a number.

## The conformance table, step by step

`tests/golden/ingest-ids.tsv` records the ids the **real** ingest derives, per producer, for synthetic rows. It is how a
change in the ingest's identity rule shows up as a failing test instead of a silent series break.

When an ingest producer's keys change, or you add an `identity` profile or an example case:

1. Edit the producer inputs or the `probe` rows in `tests/refresh_conformance.py`.
2. Run `PYTHONPATH=extensions/clickhouse-ingest python tests/refresh_conformance.py` (it needs the ingest and
   `clickhouse-connect` importable) to rewrite the table.
3. Read the diff. A **changed id is a changed series**: explain in the pull request which stored series move.
4. Update `ID_PROFILES` in `engine.py`, the example `identity` blocks and the table together, in one commit.
5. Run `spyre-bench ids --adapter`: it must say `match` for every profile.

## Versioning and compatibility

| What | Where it lives | Rule |
|---|---|---|
| `apiVersion: bench/v1alpha1` | every document | alpha: fields can be added; a removal or meaning change is announced in the pull request and the reference |
| engine version | `pyproject.toml`, `plan.json` `producer.engine.version` | informational; compatibility is carried by the next two |
| `plan_schema` (`bench.plan/1`), `run_schema` (`bench.run/1`) | plan and run files | a change of shape bumps the number; `run` refuses a plan with another one (`engine_skew`, exit 2) |
| `identity_version` | `plan.json` | bumped with any change to how an id is derived; a plan made under another version is refused by `run` |
| pack `version` | `pack.yaml`; the Project asks for a range | minor for additions, major for removals and renames; the Project range decides who is affected |
| `fingerprint_version` | the Project | 1 (default) or 2; switching restarts every environment-keyed baseline (guide 3) |
| core pack `version` | `core/pack.yaml` | Projects require a compatible `core`; most catalogs never name it |

Rules of thumb for compatible versus breaking:

| Change | Effect | Treat as |
|---|---|---|
| add a Benchmark, a case, a metric, an Environment, an Expectation | none on existing series | compatible |
| add a field to a Workload schema with a default that keeps existing cases unchanged | none | compatible |
| edit a System or Workload field of an existing Benchmark | new `benchmark_id` for every case it touches | breaking for the series |
| rename a Benchmark or a case | new id; `previous_names` links the history | breaking for the id, not for the story |
| change `component`, a series label or a label value | new `series_key` | breaking |
| change `fingerprint_version` or add a fingerprint attribute | new `env_id` | breaking for baselines and reuse |
| reorder `identity.discriminators` or change a profile's keys | ids nothing writes | breaking; the conformance check fails |
| re-unit or rename a metric | new metric with no history | breaking |

For a breaking change that you want: make it in its own pull request, say which series restart, keep the old definition
for the transition when a baseline matters, and tell the owners of the baselines. For one you did not intend, revert
the field.

## Writing the pull request

- Use Conventional Commits (`feat(bench): ...`, `fix(bench): ...`, `docs(bench): ...`) and sign off every commit
  (`git commit -s`).
- Keep one concern per commit: definitions, then the command, then the ingest or schema change.
- Describe what the change asks, which cases and series it touches, how you tested it (the commands above) and what you
  could not test.
- Do not paste measured numbers, host names or internal links into a public pull request. Synthetic examples are
  enough to show the shape.
- Expect a reviewer to ask for a second pair of eyes on anything that changes ids, deletes or overwrites data, or touches
  the run contract; that is routine, not a verdict on you.

## Deprecating and removing

A Benchmark that should stop: remove it in a pull request that says so; its history stays in the store and its series
simply ends (`validate --base-dir` reports series that end). Do not reuse the name for something different. A metric that
should stop: stop emitting it and leave the registry entry in place for a release so old Results still read.
