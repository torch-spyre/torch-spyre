# CI Profiler Build Integration Plan

**Issue:** [#1857](https://github.com/torch-spyre/torch-spyre/issues/1857) - Adding Tekton CI profile-enabled build entry for Spyre Profiling  
**Dependency:** [#927](https://github.com/torch-spyre/torch-spyre/issues/927) - Build system: USE_SPYRE_PROFILER flag + dual Kineto dependency strategy  
**Status:** Planning Phase  
**Owner:** CI/CD Pipeline Team (coordination required)

## Executive Summary

This document outlines the requirements and implementation plan for adding profiler-enabled builds to the Tekton CI system. The profiler build requires specific build flags and dependencies that differ from standard torch-spyre builds.

## Background

### Current State
- torch-spyre uses GitHub Actions for public CI (visible in `.github/workflows/`).
  **There is no CMake in this repo** — all C++ is compiled via PyTorch's setuptools
  `CppExtension`/`BuildExtension` in `setup.py`, and CI builds it through the
  `.github/actions/build-torch-spyre` composite action (which runs `uv sync`).
- Profiler support is gated by the `USE_SPYRE_PROFILER` build flag (issue #927). #927
  implements that flag in `setup.py` by mirroring the existing `USE_SPYRE_CCL` pattern
  (env var + conditional sources + a `define_macro` guard) — **not** via CMake, despite
  earlier wording. See issue #927 for the corrected build-system design.
- Profiler requires the kineto-spyre wheel dependency (`torch-2.9.1+aiu.kineto.1.1`).
- A **"Test Spyre Profiler" suite already exists** in the test matrix
  (`.github/workflows/_test_matrix.yaml`, running
  `tests/configs/torch_spyre_tests/test_spyre_profiler_config.yaml`). But the shared build
  step does not set `USE_SPYRE_PROFILER=1` or install the kineto wheel, so today its
  `requires_spyre_profiler` tests simply **skip** (see `tests/conftest.py`). The gap #1857
  closes is the *profiler-enabled build path*, not a new test entry.

### Target State
- Tekton CI includes a dedicated profiler-enabled build entry
- Profiler builds run with `USE_SPYRE_PROFILER=1`
- Kineto-spyre wheel is pre-installed before build
- Profiler tests execute as part of the build validation

## Build Requirements

### 1. Environment Variables

```bash
USE_SPYRE_PROFILER=1
```

This flag must be set **before** running the build to enable profiler compilation.

### 2. Pre-build Dependencies

The kineto-spyre wheel must be installed before building torch-spyre:

```bash
pip install torch-2.9.1+aiu.kineto.1.1-*.whl
```

**Note:** The exact wheel version and location will be provided by the CI/CD team. The wheel should be:
- Available in the Tekton build environment
- Compatible with the PyTorch version being used
- Installed in the Python environment before `pip install -e .`

### 3. Build Command

The flag is read by `setup.py` from the environment, so it must be set for the build
invocation. **Caution for CI:** the `build-torch-spyre` action runs `uv sync --frozen`
*without* `--reinstall-package`, and torch-spyre is an editable install — so setting the
env on top of an already-built `.venv` does **not** recompile the C++ and the flag becomes
a silent no-op. CI must force a clean rebuild with the flag set:

```bash
# CI path: force a clean rebuild so the profiler sources actually compile
python setup.py clean || true
USE_SPYRE_PROFILER=1 uv pip install -e . --no-build-isolation --force-reinstall
```

For a direct (fresh) local build:

```bash
USE_SPYRE_PROFILER=1 pip install -e . --no-build-isolation
```

### 4. Test Execution

After build, profiler tests should be executed:

```bash
pytest tests/profiler/ -v
```

Or using the test framework:

```bash
bash tests/run_test.sh tests/configs/torch_spyre_tests/test_spyre_profiler_config.yaml
```

## Tekton CI Build Entry Specification

### Build Profile Name
Suggested: `torch-spyre-profiler` or `torch-spyre-with-profiling`

### Build Steps

1. **Setup Environment** — clone torch-spyre, set up the Python venv, install base deps.
2. **Install Kineto Dependency** — the kineto-spyre wheel (see §1 for source).
3. **Build with Profiler Enabled** — force a clean rebuild so the C++ recompiles.
4. **Verify Build** — confirm the profiler imports.
5. **Run Profiler Tests** — via the existing suite config.

```bash
# 2. Install Kineto dependency (wheel source TBD — see Coordination §1)
pip install <kineto-spyre-wheel-path>

# 3. Build with profiler enabled (clean rebuild — env alone won't recompile)
python setup.py clean || true
USE_SPYRE_PROFILER=1 pip install -e . --no-build-isolation --force-reinstall

# 4. Verify the build
python -c "import torch_spyre.profiler; print('Profiler import successful')"

# 5. Run the profiler test suite
bash tests/run_test.sh tests/configs/torch_spyre_tests/test_spyre_profiler_config.yaml
```

### Hardware Requirements

- **Spyre Hardware:** Required for full profiler functionality
- **Alternative:** If Spyre hardware is unavailable, tests can run with skip markers (see issue #933)
- **Minimum:** x86_64 architecture with sufficient memory for compilation

### Expected Build Time

- Estimated: 15-30 minutes (including compilation and tests)
- May vary based on hardware and parallel build settings

## Coordination Points with CI/CD Team

### 1. Kineto Wheel Availability

**Question:** Where will the kineto-spyre wheel be stored in the Tekton environment?

**Options:**
- Artifactory repository (similar to RPM downloads in current workflows)
- Pre-installed in build image
- Downloaded from GitHub releases

**Action Required:** CI/CD team to specify wheel location and access method

### 2. Build Trigger Configuration

**Question:** When should the profiler build run?

**Recommendations:**
- On every PR that touches profiler code (`torch_spyre/profiler/`, `torch_spyre/csrc/profiler/`, `tests/profiler/`)
- On nightly builds
- On release branches
- Manual trigger option for testing

**Action Required:** CI/CD team to configure trigger rules

### 3. Build Image Requirements

**Question:** Does the Tekton build image need updates?

**Requirements:**
- C++20 compiler (profiler sources build with `-std=c++20`, like the rest of `setup.py`)
- ninja + PyTorch's `cpp_extension` toolchain (already used by the standard build; **no CMake**)
- Python development headers
- libKineto + libAIUpti — supplied by the kineto-spyre wheel (WHEEL mode) or PyTorch ≥ 2.10
  integrated kineto (UPSTREAM mode); see issue #927

**Action Required:** CI/CD team to verify build image has required tools

### 4. Test Hardware Access

**Question:** How will profiler builds access Spyre hardware?

**Options:**
- Dedicated Spyre hardware pool for CI
- Use libAIUpti stub for build-only validation
- Skip hardware-dependent tests in CI

**Action Required:** CI/CD team to provision hardware or configure test skipping

### 5. Failure Handling

**Question:** Should profiler build failures block merges?

**Recommendations:**
- Initially: Non-blocking (informational only)
- After stabilization: Blocking for profiler-related PRs
- Always blocking for release branches

**Action Required:** CI/CD team to configure failure policies

## Integration with Issue #927

This CI integration depends on issue #927 being completed first. The build system changes in
#927 (implemented in `setup.py`, mirroring `USE_SPYRE_CCL` — **no CMake**) include:

1. ⏳ `USE_SPYRE_PROFILER` env-var flag in `setup.py` (default OFF)
2. ⏳ Dual Kineto dependency strategy (`SPYRE_KINETO_MODE` = AUTO/WHEEL/UPSTREAM)
3. ⏳ Conditional profiler sources + `define_macro` guard for the profiler C++
4. ⏳ Clear, actionable error messages for missing dependencies
5. ⏳ **Profiler-enabled CI build entry (Tekton + GitHub Actions)** ← This issue (#1857)

(Status marks reset to ⏳ — #927 was not yet merged at the time of writing; see issue #927
for the corrected, setuptools-based design.)

**Status Check:** Before implementing this CI integration, verify that:
- [ ] Issue #927 is merged to main
- [ ] `USE_SPYRE_PROFILER=1` builds successfully locally
- [ ] Profiler tests pass with kineto-spyre wheel installed
- [ ] Documentation in `docs/source/contributing/profiling.md` is updated

## Optional: GitHub Actions Integration

While the primary focus is Tekton CI, we can also add profiler builds to the public GitHub Actions workflows for transparency and community testing.

### Proposed GitHub Actions Changes

A **dedicated, non-blocking** `test_profiler` workflow is the recommended shape (rather than a
new row in the shared `_test_matrix.yaml`, because that matrix builds **once per job** with no
flag, so a single matrix row cannot carry `USE_SPYRE_PROFILER=1`). The scaffold lives at
`.github/workflows/profiler_build.yaml`. It mirrors the existing matrix job's steps but:

- sets `USE_SPYRE_PROFILER: '1'` as job `env`, **and forces a clean rebuild** with the flag set.
  Setting the env alone is **not** enough: `build-torch-spyre` runs `uv sync --frozen` with no
  `--reinstall-package`, and torch-spyre is an editable install, so an env-only change does not
  recompile the C++ — the flag would be a silent no-op. The job runs `setup.py clean` +
  `USE_SPYRE_PROFILER=1 uv pip install -e . --no-build-isolation --force-reinstall`;
- adds an **Install kineto-spyre wheel** step *before* the build (source TBD — the main
  coordination item; see §1);
- reconciles the env via the existing `./.github/actions/build-torch-spyre` action, then forces
  the profiler rebuild (the action alone will not rebuild on an env-only change);
- runs the existing suite via `./.github/actions/run-test-suite` with
  `tests/configs/torch_spyre_tests/test_spyre_profiler_config.yaml`;
- runs on the standard HW labels `x86_64` + `spyre_pf_x1` + `image_spyre_backend`;
- is `continue-on-error: true` and triggered on profiler paths + nightly + `workflow_dispatch`.

**Corrections vs. the earlier draft of this section:** the image label is `image_spyre_backend`
(not `image_torch_spyre`); there is no `checkout-prebuilt-torch-spyre` action (use `checkout`
- `build-torch-spyre`); tests run through `run-test-suite` with the config yaml (not a raw
`pytest tests/profiler/`).

**Note:** keep this job non-blocking until #927 merges and the wheel source is wired; flip to
blocking (remove `continue-on-error`, add to required checks) once it is green.

## Acceptance Criteria

- [ ] Tekton CI has a dedicated profiler-enabled build entry
- [ ] Build runs with `USE_SPYRE_PROFILER=1`
- [ ] Kineto-spyre wheel is installed before build
- [ ] Profiler tests execute successfully
- [ ] Build failures are reported appropriately
- [ ] Documentation is updated with CI details

## Timeline and Dependencies

### Phase 1: Coordination (Week 1)
- [ ] Meet with CI/CD team to discuss requirements
- [ ] Resolve kineto wheel availability question
- [ ] Determine hardware access strategy
- [ ] Agree on build trigger configuration

### Phase 2: Implementation (Week 2)
- [ ] CI/CD team creates Tekton build entry
- [ ] Configure kineto wheel installation
- [ ] Set up test execution
- [ ] Configure failure handling

### Phase 3: Validation (Week 3)
- [ ] Test profiler build on Tekton
- [ ] Verify all profiler tests pass
- [ ] Document any issues or limitations
- [ ] Update this plan with final configuration

### Phase 4: Stabilization (Week 4)
- [ ] Monitor build stability
- [ ] Adjust timeouts if needed
- [ ] Enable blocking mode for profiler PRs
- [ ] Close issue #1857

## References

- [Issue #1857](https://github.com/torch-spyre/torch-spyre/issues/1857) - This issue
- [Issue #927](https://github.com/torch-spyre/torch-spyre/issues/927) - Build system implementation
- [Issue #933](https://github.com/torch-spyre/torch-spyre/issues/933) - Hardware/stub availability
- [Profiling Documentation](../docs/source/user_guide/profiling/index.md)
- [Contributing to Profiler](../docs/source/contributing/profiling.md)

## Contact Points

- **Issue Owner:** @jason-liu227
- **Assignees:** @Rafael-Sadykov, @jor2
- **CI/CD Team:** [To be specified]
- **Profiling Squad:** [To be specified]

## Next Steps

1. **Immediate:** Share this plan with CI/CD team for review
2. **Immediate:** Verify issue #927 completion status
3. **This Week:** Schedule coordination meeting with CI/CD team
4. **This Week:** Resolve kineto wheel availability question
5. **Next Week:** Begin Tekton CI implementation

---

**Document Status:** Draft for Review  
**Last Updated:** 2026-06-23 (corrected CMake→setuptools framing; noted the existing
"Test Spyre Profiler" matrix suite; aligned the GitHub Actions proposal with the real
composite actions and the forced-rebuild requirement — corrected build-system design
tracked in issue #927)  
**Next Review:** After CI/CD team feedback
