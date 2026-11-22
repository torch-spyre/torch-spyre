# spyre-bench user guide

This guide is for people who run benchmarks, read their results, or decide what to do about a regression. It does not
require reading the engine code. Engine concepts (kinds, identity, packs) are in [`../README.md`](../README.md); how to
add a benchmark, dataset or metric is in the contributor docs.

## What the platform is

A benchmark platform with four parts:

1. **A catalog** of benchmark definitions (YAML in each product repository). A definition says what is measured, never
   when or where it is scheduled.
2. **An engine** (`spyre-bench`) that validates the catalog, plans which cases to run (or reuse), runs them, and hands
   the raw output to the ingest.
3. **A results store** (a database) that holds every repeat of every case, with its environment, placement and
   validity, and computes **verdicts** from history.
4. **Views and checks** on top: a dashboard (drill-down from a verdict to a single repeat), and an advisory check on
   pull requests.

Two properties shape everything else:

- **Advisory.** A verdict never blocks a merge. It tells you where to look.
- **Identity is derived.** A case is identified by what it asks (its inputs), not by a name someone minted. Changing an
  input makes a new benchmark and a new history; renaming needs `previous_names` to keep history linked.

## Status legend

Features move from design to use at different speeds. Each guide marks what it describes:

| Mark | Meaning |
|---|---|
| **Available** | in this tree; the commands shown run today |
| **Pending** | implemented but not merged yet, or merged but not deployed; ask the maintainers before relying on it |
| **Planned** | designed, not built; described so you know what is coming |

## Which guide do I need?

| I want to... | Guide |
|---|---|
| understand a verdict, a reason code, "suspected" or "no_baseline" | [Reading results](reading-results.md) |
| run a benchmark on my machine | [Running locally](run-locally.md) |
| see how my pull request performs | [Launching on demand](launch-on-demand.md) |
| look at last night's run | [Checking a nightly run](check-nightly.md) |
| compare two runs, two builds, two environments | [Comparing](compare.md) |
| compare against a number that is not ours | [References](references.md) |
| test whether CPU pinning matters | [Pinning experiments](pinning.md) |
| find the slow rank or kernel of a multi-card run | [Multi-card and trace debugging](multi-card-trace.md) |
| read accuracy or parity results | [Quality results](quality.md) |
| remove bad data from the baselines | [Incidents and invalidation](incidents-invalidation.md) |
| put results from another machine or site into the store | [Uploading results](uploading.md) |
| look up a term or a common question | [FAQ](faq.md), [Glossary](glossary.md) |

## Who to ask

- A wrong or confusing verdict, a missing result: open an issue in the repository that owns the benchmark and label it
  `bench`; name the run id and the benchmark id.
- A problem with the engine or the guides themselves: open an issue in the `torch-spyre` repository, component
  `spyre-bench`.
- Access (database credentials, upload rights, approval rights): ask the maintainers of your deployment; rights are
  granted per role, see [Uploading](uploading.md) and [References](references.md).

All example values in this guide are synthetic.
