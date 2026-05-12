# Contributing to the Profiler

This page extends the general [contribution guidelines](guidelines.md) with
conventions specific to the **profiling squad**, the contributors building
out the Spyre profiling toolkit described in [RFC 0601][rfc-0601]. If this
is your first profiler PR, read the guidelines page first. Everything here
assumes you already have a working fork and `pre-commit` set up.

The user-facing docs we ship live under
[Profiling](../user_guide/profiling/index.md). When you change a feature,
walk those pages as a new user would. If the instructions no longer match
reality, fix them in the same PR.

## What makes profiler work different

Most torch-spyre PRs touch one layer. Profiler PRs almost always cross
several:

| Layer | Where it lives | Typical change |
|---|---|---|
| Python API | `torch_spyre/profiler/` | `profile_spyre()` wrapper, `torch.spyre.memory_*` |
| C++ registration | `torch_spyre/csrc/profiler/` | PrivateUse1 observer, kineto wiring |
| Build | `CMakeLists.txt`, `torch_spyre/csrc/profiler/CMakeLists.txt` | Guarded by `USE_SPYRE_PROFILER` |
| Tests | `tests/profiler/` | Skip-marked when `USE_SPYRE_PROFILER` is off |
| External | [`kineto-spyre`][kineto-spyre], [`aiu-trace-analyzer`][ata] | Versioned separately |
| Docs | `docs/source/user_guide/profiling/` | User-visible additions |

Plan PRs accordingly. See [PR scope](#pr-scope) below.

## Branch naming

Use `profiler/<area>-<short-description>` so a `git branch -r` listing
tells the reviewer what each branch is about without opening the PR:

| Prefix | Use for |
|---|---|
| `profiler/build-…` | Build system, CMake, linking, `USE_SPYRE_PROFILER` |
| `profiler/reg-…` | C++ registration, PrivateUse1 plugin loading |
| `profiler/api-…` | Python APIs in `torch_spyre/profiler/` |
| `profiler/trace-…` | Trace enrichment, post-processing, Perfetto grouping |
| `profiler/mem-…` | Memory profiling at any layer |
| `profiler/test-…` | Test additions |
| `profiler/docs-…` | Documentation, examples |
| `profiler/feat-…` | Multi-PR feature work |
| `profiler/fix-…` | Bug fixes |

Keep `<short-description>` to **3–5 hyphenated words**. `cmake-libaiupti`
is about right. `sol` is too terse to read at a glance, and
`tex-scratchpad-vram-sol-average` should be split into smaller PRs.

## PR title prefix

Prefix the **PR title** with `[profiler]` so it is easy to slice profiler
work out of `git log` after merge. Because we squash-merge, the PR title
becomes the single commit message landed on `main` — there is no need to
prefix every individual commit on the feature branch. Stack a sub-area
tag when one is obvious:

```text
[profiler] Add profile_spyre() context manager
[profiler][memory] Stub torch.spyre.memory_allocated()
[profiler][trace] Group runtime events under PerfettoSpyreRuntime track
[profiler][test] Add scaffold with USE_SPYRE_PROFILER skip markers
[profiler][docs] Document kineto-spyre wheel install
```

Tooling that slices profiler work looks for this tag in `main` history
(merged work) and in `profiler/*` branches (in-flight work), so the
branch prefix plus the PR title prefix together are enough — individual
commits on the branch can carry whatever message is most useful while
you iterate.

Sign off your commits (`git commit -s`) like every other torch-spyre
commit.

## PR scope

Profiling features tend to grow large because they touch multiple layers.
Keep each PR to **one observable change**, even when the underlying feature
spans several PRs:

* Good: "Add `profile_spyre()` wrapper (Python only, no kineto wiring yet)".
* Good: "Wire kineto observer behind `USE_SPYRE_PROFILER` (no API change)".
* Bad: "Add memory profiling APIs and Perfetto trace grouping". Split it.

If a feature genuinely cannot be split, flag that in the PR description
and in the matching sub-issue so reviewers know what they are signing up
for.

## Building with the profiler enabled

The profiler is gated by a CMake flag so torch-spyre still imports
cleanly without it. Local development usually wants it on:

```bash
USE_SPYRE_PROFILER=1 pip install -e . --no-build-isolation
```

When the flag is **off**, every profiler import path must still succeed
(the import test below covers this). When it is **on**, install the
kineto-spyre wheel:

```bash
pip install kineto-spyre
```

If you change build wiring, verify both the on and off paths build and
import.

## Testing profiler changes

The profiler test suite lives at `tests/profiler/`. The `conftest.py`
exposes a `spyre_profiler_available` fixture that skips tests when the
feature is compiled out, so the same suite works in both build modes.

Run only the profiler tests:

```bash
pytest tests/profiler/ -v
```

For focused iteration on activity / trace / memory / sync subsets:

```bash
pytest tests/profiler/test_spyre_profiler.py -k activity
pytest tests/profiler/test_spyre_profiler.py -k trace
```

Smoke test validation before you open a PR:

1. `import torch_spyre.profiler` succeeds with `USE_SPYRE_PROFILER=0`.
2. `tests/profiler/` passes with the kineto-spyre wheel installed.
3. If you touched trace emission, capture a small trace and open it in
   Perfetto. See [Trace analysis](../user_guide/profiling/trace_analysis.md).
4. If you touched device telemetry, sanity-check against `aiu-smi`. See
   [Device monitoring](../user_guide/profiling/device_monitoring.md).

## Trace and telemetry sanity checks

Profiler bugs are often invisible at the test level. The test passes, the
trace is wrong. When you change anything that emits trace events or
metrics, attach one of the following to the PR description:

* A short Perfetto screenshot of the affected region.
* The output of `aiu-trace-analyzer` summarizing the trace.
* A diff of the relevant counter values before and after.

This is the most common review request on profiler PRs. Including it up
front saves a round trip.

:::{tip}
**Cross-check with `chrome://tracing` when validating event ordering.**
Perfetto silently truncates overlapping events on the same thread — two
events that overlap (which is impossible within a single thread and
indicates a real bug) are rendered as a single clean span, hiding the
problem. `chrome://tracing` instead renders the overlap as garbled,
intermingled labels, which makes the bug obvious. Using it has two
benefits for "is the trace correct?" reviews:

1. Overlapping/interleaved events on the same thread are visible
   instead of hidden.
2. It runs locally — your trace data does not transit a third-party web
   service.

Use Perfetto for analysis and presentation, but reach for
`chrome://tracing` when you specifically need to verify that no two
events on the same thread overlap.
:::

## Coordinating with kineto-spyre

[`kineto-spyre`][kineto-spyre] is a separate repository on its own release
cadence. If your change needs a new kineto-spyre symbol or behaviour:

1. Land the change in kineto-spyre **first**, with its own PR and release.
2. Pin the new kineto-spyre version in torch-spyre's requirements.
3. Open the torch-spyre PR with a description line like
   *"Requires `kineto-spyre>=X.Y.Z`."*

Do not couple a torch-spyre PR to an unreleased kineto-spyre commit.
Reviewers cannot run it and CI cannot reproduce it.

## Documentation expectations

If you change something a user can see (a new API, a new env var, a new
trace field, a new build flag), update the docs in the same PR. The right
home is almost always under `docs/source/user_guide/profiling/`. Sphinx
runs with `-W` in CI, so warnings will fail your PR. Build locally before
pushing:

```bash
python -m sphinx docs/source docs/build/html -W --keep-going
python -m http.server 8080 --directory docs/build/html
```

## Reviewers

Profiler PRs need a review from a **profiling squad lead** plus **one
other squad member**. CODEOWNERS for `torch_spyre/profiler/`,
`torch_spyre/csrc/profiler/`, and `tests/profiler/` will request the
right people automatically. If GitHub does not auto-request a lead,
request one manually.

[rfc-0601]: https://github.com/torch-spyre/rfcs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md
[kineto-spyre]: https://github.com/IBM/kineto-spyre
[ata]: https://github.com/IBM/aiu-trace-analyzer
