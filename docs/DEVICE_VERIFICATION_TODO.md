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

## Remaining residual gaps (not fixable from this machine)

A follow-up device-access pass confirmed items 1-3 below (see git history
on this branch for the fixes). One narrow gap remains, a hard environment
constraint rather than a doc bug:

- `docs/source/user_guide/profiling/device_monitoring.md` and the
  `aiu-smi`-specific rows in `environment_variables.md` /
  `performance_analysis_methodology.md` describe `aiu-smi` output and its
  "Known issues" (`rsvmem`/`pt_act` not captured correctly). `aiu-smi`
  requires a wheel from an internal IBM package mirror not reachable from
  this machine, so these specific claims remain unverified independently.
  Everything else in those docs that does not depend on `aiu-smi` itself
  (env var plumbing, `SENPERFORMANCE=2`/`ideal_cycles.json`, the
  provenance-naming and profiler-table format) was verified on-device.

`ProfilerActivity.PrivateUse1` profiling (`pytorch_profiler.md`) was
*also* verified on-device this pass: it needs no separate `kineto-spyre`
wheel — that dependency was obsoleted by the PyTorch 2.12 upgrade (Kineto
headers now ship with PyTorch itself, and AIUPTI tracing is built
directly into `torch_spyre`'s native extension). `pytorch_profiler.md`,
`toolkit_matrix.md`, `contributing/profiling.md`, and
`examples/profile_ops.py` had stale `kineto-spyre` install instructions
from before that change; fixed on this branch, along with the same stale
instruction in the `upgrade-pytorch-version` skill's Step 7. Ran
`tests/profiler/test_spyre_profiler.py` on-device with no `kineto-spyre`
package installed to confirm: 9 passed (4 skipped on an unrelated
`requires_spyre_profiler` build-flag marker), including
`test_basic_profile`'s `ProfilerActivity.PrivateUse1` capture.
`profiling/end_to_end_example.md`'s Granite/FMS walkthrough was not run —
it requires cloning separate FMS repos and downloading a model
checkpoint, disproportionate to a docs-verification pass (its own
kineto-spyre note is already correctly scoped to "not required for
PyTorch 2.13 and later", so left as-is).

This last `aiu-smi` gap is not a documentation defect; it's a follow-up
for whoever next has access to the internal package mirror.
