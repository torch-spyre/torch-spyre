# Developer guides: defining things for the benchmark platform

These guides are for everyone who writes definitions that `spyre-bench` reads: benchmarks, the environments they ask
for, the data they consume, the metrics they report, the analyses and expectations that judge them, and the packs that
teach the engine a new kind of system. They are end to end: each one starts from nothing and ends with a catalog that
passes `spyre-bench validate` and is planned by `spyre-bench plan`.

Guides 1 to 11 use synthetic examples; [guide 12](12-worked-example-nightly-weekly.md) declares the two real benchmarks
(`vllm bench` and `spyre-perf-suite`) with nightly and weekly invocations. Every YAML, JSON, Python and `spyre-bench` snippet is a block marked
`validated`, and `tests/test_docs_snippets.py` checks it against the real example files and the real engine on every
run, so the guides cannot drift from the code without a test failing. The catalogs behind them are
`examples/docs-*`; `tests/test_docs_examples.py` plans and runs each one with a stub command.

For the reference (the identity rules, the `run` contract, planning, exit codes), read [`../README.md`](../README.md).
These guides link to it instead of repeating it.

## Which guide do I need?

| I want to | Guide |
|---|---|
| understand the vocabulary and where each kind of file lives | [1. Concepts and the sixteen kinds](01-concepts.md) |
| add a benchmark (a template that expands into cases) | [2. Define a Benchmark end to end](02-define-a-benchmark.md) |
| ask for different hardware, pin CPUs, request a clean run, describe a class of hosts | [3. Environments and node classes](03-environments-and-node-classes.md) |
| point a benchmark at a dataset, a model or a store | [4. Datasets, assets, models and stores](04-data-assets-models-stores.md) |
| report a new number | [5. Add a metric](05-add-a-metric.md) |
| compare cases, derive a number, decide what counts as a regression | [6. Analyses, expectations and validity](06-analyses-expectations-validity.md) |
| teach the engine a new kind of system or workload | [7. Write a pack](07-write-a-pack.md) |
| decide what my benchmark's command may see and do | [8. Run commands safely](08-run-commands-safely.md) |
| try it on a laptop, read what came out, find out why a case did not run | [9. Test and debug locally](09-test-and-debug-locally.md) |
| open the pull request | [10. Contribute](10-contribute.md) |
| fix an error message | [11. Troubleshooting and FAQ](11-troubleshooting-and-faq.md) |
| see the real `vllm bench` and `spyre-perf-suite` declared, with a nightly and a weekly run | [12. Worked example: nightly and weekly](12-worked-example-nightly-weekly.md) |
| describe a design parameter, sweep it, and record the claim that an optimization helped | [13. Design parameters, sweeps, claims, experiment flow](13-design-parameters-sweeps-claims.md) |

## The shortest path

A new benchmark in an existing catalog is three steps.

1. Copy a Benchmark file from the same Suite folder and change what it asks for ([guide 2](02-define-a-benchmark.md)).
2. Run the hardware-free checks: `spyre-bench validate`, `spyre-bench explain <case>`, `spyre-bench plan`.
3. Run it once with your own command on a machine that has what the Environment asks for ([guide 9](09-test-and-debug-locally.md)).

```bash validated
spyre-bench validate examples/docs-benchmark
spyre-bench explain op_scale_n1024 --catalog examples/docs-benchmark
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --select smoke --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan.json
```

## What is not in these guides

The engine validates definitions, plans, runs commands and reads verdicts. It does not decide when a benchmark runs,
on which pool, with what budget, or who is told about a regression: those belong to whatever invokes the engine (a CI
workflow, a scheduler, a developer's shell). A Benchmark carries only a cost hint, `est_hours`. Guide 1 explains the
split; guide 10 lists what a pull request to a catalog may and may not contain.

The guide catalogs `examples/docs-*` name the engine's own packs (`kernel`, `llm-serving`, `spyre`) and use made-up
model, node-class and pool names. The `examples/real-*` catalogs of guide 12 may use the real benchmark, metric and model
names; datasets are always named by slug. In either kind, never put a real host name, measured number, score, threshold, credential or
name of an internal system in a definition or a guide.
