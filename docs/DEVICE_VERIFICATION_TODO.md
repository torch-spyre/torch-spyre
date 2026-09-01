# Device Verification TODO

This checklist was produced by a static-analysis-only documentation audit
(branch `docs-update`) comparing `docs/source/` against the current
compiler source. It has no access to a Spyre device, so the items below
could not be confirmed and need a follow-up pass on a machine with device
access.

Static-analysis findings that did not require device access (missing RFC
index entry, `CustomPostFusionPasses`/pre-scheduling pass-order and
missing-step errors in `inductor_frontend.md`, the wrong `LAYOUT_SOLVER`
default in `torch_spyre.rst`, and a stray "Internal reference" heading in
`spyre_accelerator.md`) have already been fixed directly on this branch.
The items below are the ones that genuinely need a device.

## 1. KTIR production-path framing

**File:** `docs/source/compiler/ktir.md`

The doc frames KTIR as experimental and states production still goes
through SuperDSC (see lines referencing "production path still goes
through SuperDSC" and "The README describes it as an experimental...").
Static analysis confirms there is substantial in-tree KTIR
implementation (`torch_spyre/_inductor/codegen/ktir.py`,
`tests/inductor/test_ktir_emitter.py`, `test_ktir_compile.py`,
`test_ktir_validate.py`, plus two test config YAMLs), but static analysis
cannot tell whether KTIR is actually exercised on real hardware yet, or
is still exclusively CI/simulation-only.

**Needs device access to confirm:**

- Whether any current model-enablement or perf workflow actually runs the
  KTIR path against a physical Spyre device (as opposed to only
  compiling/validating KTIR in CI without executing it).
- Whether the `ktir.md` "experimental, not the production path" framing
  is still accurate, or should be updated to describe a partial rollout.

## 2. Runtime/perf claims that can't be confirmed from source alone

**Files:** `docs/source/runtime/*.md`, `docs/source/user_guide/profiling/*.md`

Several docs make claims about runtime behavior (DMA transfer costs,
profiler counter names/units, actual observed latencies) that are
consistent with the source code but were not independently verified
against a running device or real profiler output in this audit.

**Needs device access to confirm:**

- Profiler output examples in `docs/source/user_guide/profiling/` still
  match the real output format of current profiling infrastructure.
- Any specific performance numbers or example traces quoted in the docs
  are still representative.

## 3. Quickstart / example scripts

**Files:** `docs/source/getting_started/quickstart.md`,
`docs/source/user_guide/examples/`

Code samples were checked for API correctness against current source
(import paths, function signatures) but were not actually executed,
since running them requires a Spyre device.

**Needs device access to confirm:**

- Quickstart and example scripts run end-to-end and produce the output
  shown in the docs.

## How to close out this file

For each item above: reproduce on a device, update the referenced doc
page if reality has diverged, then delete that section from this file.
Once all sections are resolved, delete this file entirely.
