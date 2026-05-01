---
name: check-docs
description: "Check documentation consistency against code changes. Audits supported ops table, RFC links, API docs, compiler docs, runtime docs, profiling, and user/developer guides for staleness or drift."
---

# Check Documentation Consistency

You are auditing the torch-spyre documentation for consistency with the current
state of the codebase. Run through each section below systematically. Report
every issue found with the file path, what is wrong, and a suggested fix.

## Audience

Documentation serves two personas:

- **Users** — data scientists and ML engineers running models on Spyre via
  `torch.compile`. They care about: installation, quickstart, supported ops,
  profiling, debugging, and examples.
- **Developers** — engineers contributing to torch-spyre. They care about:
  compiler architecture, Inductor integration, adding operations, tensor
  layouts, work division, runtime internals, and RFCs.

Check that content is appropriate for its target audience and does not leak
internal implementation details into user-facing pages.

## 1. Supported Operations Table

**File:** `docs/source/user_guide/supported_operations.md`

The table has four key columns: **Operation**, **Eager**, **Compiled**,
**Execution**, and **Notes**. Each must be verified independently.

### 1a. Compiled support (torch.compile path)

Cross-reference the "Compiled" column against the actual op registrations in:
  - `torch_spyre/_inductor/customops.py` (custom ops)
  - `torch_spyre/_inductor/decompositions.py` (decompositions)
  - `torch_spyre/_inductor/lowering.py` (lowerings)
  - `torch_spyre/ops/fallbacks.py` (fallback registrations)

Use `tests/inductor/test_inductor_ops.py` as the primary reference for
what is truly tested in the compiled path:
  - Every op tested there should appear in the table with Compiled = Y.
  - Check for ops in test_inductor_ops.py that are missing from the table.
  - Check for ops in the table that have no corresponding test.

### 1b. Eager support (direct tensor ops without torch.compile)

Eager support comes from **multiple sources**.
Eager op sources:
  - `torch_spyre/ops/eager.py` — manually registered ops via
    `@torch.library.register_kernel` (e.g., `mm`, `silu`, `mish`).
  - `torch_spyre/_inductor/decompositions.py` — some decomposed ops
    also work in eager mode, specifically: `rms_norm`, `layer_norm`,
    `softplus`, `linear`, and `scaled_dot_product_attention`.

To verify the Eager column:
  - Read `torch_spyre/ops/eager.py` for additional manual registrations
    (e.g., `mm`, `silu`, `mish`).
  - Some decompositions in `torch_spyre/_inductor/decompositions.py`
    are also triggered in eager mode: `rms_norm`, `layer_norm`,
    `softplus`, `linear`, and `scaled_dot_product_attention`. These
    should be marked Eager = Y.
  - For compiled-only determination: check `tests/inductor/test_inductor_ops.py`
    for `run_eager=False` — any op tested with this flag is
    **compiled-only** and must NOT have Eager = Y.
  - The full eager op set is: Metadata.yaml ops + eager.py ops +
    the five decompositions listed above. Do not use `eager.py` alone.

### 1c. View ops

View ops (reshape, transpose, permute, squeeze, unsqueeze, clone,
contiguous, expand, narrow, select) require special attention:
  - View ops with partial support should be noted (e.g., squeeze/unsqueeze
    may trigger internal recompile for certain shapes).
  - View ops that are planned but not yet implemented (e.g., expand,
    narrow) should be listed with empty Eager/Compiled cells and a
    "Planned" note. Check for TODOs in `test_inductor_ops.py`.
  - Most view ops are compiled-only — verify against `run_eager=False`.

### 1d. Execution column

  - Ops in `torch_spyre/ops/fallbacks.py` (registered via
    `@register_fallback`) run on CPU — mark as "CPU fallback".
  - All other ops run on Spyre — mark as "Spyre".

### 1e. General checks

- Check for ops that exist in code but are missing from the table.
- Check for ops listed in the table that no longer exist in code.

## 2. RFC Links and References

**File:** `docs/source/rfcs/index.md`

- All RFC links must point to `https://github.com/torch-spyre/rfcs` (external
  repo), NOT to `torch-spyre/torch-spyre/RFCs/` (old location, now deleted).
- Grep the entire `docs/` tree for any remaining references to the old RFC
  paths: `RFCs/`, `torch-spyre/torch-spyre/blob/main/RFCs`.
- Check that every RFC listed in the index table actually exists at the linked
  URL (verify the path structure: `<number>-<Name>/<number>-<Name>RFC.md`).
- Check for new RFCs in `https://github.com/torch-spyre/rfcs` that are not yet
  listed in the index.

## 3. Compiler Documentation

**Files:** `docs/source/compiler/*.md`

- **architecture.md** — Verify the compilation pipeline stages match the actual
  code flow in `torch_spyre/_inductor/__init__.py` and `spyre_kernel.py`.
- **inductor_frontend.md** — Check that extension points (PrePass, PostPass,
  SchedulerPass) match what is registered in `torch_spyre/_inductor/passes.py`
  and `torch_spyre/_inductor/pass_utils.py`.
- **backend.md** — Verify DeepTools invocation paths match `torch_spyre/_inductor/dsc.py`.
- **adding_operations.md** — Confirm the three patterns (direct mapping,
  decomposition, custom op) still match the current code patterns. Check that
  example ops cited still exist.
- **work_division_planning.md** — Verify op_dim_splits representation and
  dimension labels match `torch_spyre/_inductor/core_division.py`.
- **work_division_codegen.md** — Check code generation patterns match
  `torch_spyre/_inductor/codegen/compute_ops.py` and `data_ops.py`.

## 4. Runtime Documentation

**File:** `docs/source/runtime/overview.md`

- Verify device registration flow matches `torch_spyre/__init__.py`.
- Check allocator description matches `torch_spyre/csrc/spyre_mem.cpp`.
- Verify tensor implementation details match `torch_spyre/csrc/spyre_tensor_impl.cpp`.
- Check for new runtime features (e.g., streams in `torch_spyre/streams.py`,
  `torch_spyre/csrc/spyre_stream.cpp`) that are not yet documented.

## 5. API Reference

**Files:** `docs/source/api/torch_spyre.rst`, `torch_spyre/__init__.py`,
`torch_spyre/streams.py`

The API reference is **manually maintained** because `torch_spyre` cannot
be imported during the Sphinx build (C++ extensions and Spyre hardware are
required). This means the API docs will drift unless explicitly checked.

When adding or updating API entries, follow the PyTorch `torch.cuda` API
documentation style — use `.. function::`, `.. class::`, `.. method::`,
and `.. attribute::` directives with full signatures, typed parameters,
return types, and code examples where usage is non-obvious.

### 5a. Device Management API

Cross-reference the "Device Management" table in `torch_spyre.rst` against
the functions exposed by `make_spyre_module()` in `torch_spyre/__init__.py`:

- Check every `mod.xxx = lambda: impl.xxx()` line in `make_spyre_module()`.
- If a new function was added to `_SpyreImpl` and wired into the module,
  it must be added to the table in `torch_spyre.rst`.
- If a function was removed or renamed, update or remove the corresponding
  row.

### 5b. Streams API

Cross-reference the "Streams" tables in `torch_spyre.rst` against
`torch_spyre/streams.py`:

- Check the `__all__` list in `streams.py` — every exported name must
  appear in the docs.
- Check `Stream` class methods and properties — compare against the
  Stream members table.
- If new methods or properties were added to `Stream`, add them.
- Verify the function signatures (parameter names, types, defaults)
  match the code.

### 5c. Random Number Generation

- Check `manual_seed` and `manual_seed_all` signatures in `_SpyreImpl`
  still match the docs.

### 5d. Constants and Environment Variables

- Verify `torch_spyre.constants.DEVICE_NAME` value matches the docs.
- Cross-reference the environment variables table against:
  - `torch_spyre/_inductor/logging_utils.py` (Spyre Inductor logging vars)
  - `torch_spyre/__init__.py` (runtime vars like `TORCH_SPYRE_DEBUG`)
  - `CLAUDE.md` env var table
- If new env vars were added anywhere in the codebase, add them to the
  API docs table.

### 5e. New Public Modules

Check if any new public modules have been added that are not yet in the
API docs:

- `torch_spyre/ops/` — eager ops and fallbacks
- `torch_spyre/device/` — device interface
- `torch_spyre/execution/` — async compile, kernel runner
- `torch_spyre/memory/` — memory management

For each, determine if it exposes public API that users would call
directly. If so, add a section to `torch_spyre.rst`.

## 6. User Guide

**Files:** `docs/source/user_guide/*.md`

- **running_models.md** — Verify `torch.compile` usage examples are correct.
  Check that environment variables (SENCORES, etc.) match `torch_spyre/constants`
  and actual behavior.
- **tensors_and_layouts.md** — Verify SpyreTensorLayout fields match
  `torch_spyre/csrc/spyre_tensor_impl.h`. Check RFC links point to external repo.
- **profiling.md** — Check that profiling instructions match any new profiling
  infrastructure (e.g., logging utilities in `torch_spyre/_inductor/logging_utils.py`).
- **debugging.md** — Verify environment variables and compiler artifact paths
  are still accurate.
- **examples.md** — Check that referenced example scripts exist in `examples/`.

## 7. Getting Started

**Files:** `docs/source/getting_started/*.md`

- **installation.md** — Verify Python and PyTorch version requirements match
  `pyproject.toml` and `requirements/run.txt`.
- **quickstart.md** — Verify code examples actually work with current API.

## 8. Contributing Guide

**File:** `docs/source/contributing/guidelines.md`

- Verify development workflow instructions are current.
- Check that linting tools listed match `.pre-commit-config.yaml`.
- Verify test commands match `pytest.ini` configuration.

## 9. Inductor Integration Changes

Check for drift between docs and code in these areas:

- New passes added to `torch_spyre/_inductor/passes.py` or `temp_passes.py`
  that are not documented.
- New codegen patterns in `torch_spyre/_inductor/codegen/superdsc.py`.
- Changes to `torch_spyre/_inductor/wrapper.py` (host code generation).
- New modules like `torch_spyre/_inductor/views.py`,
  `torch_spyre/_inductor/multi_dim_reduction_pass.py`,
  `torch_spyre/_inductor/op_spec.py` that may need documentation.

## 10. Sensitive Content Audit

Scan all documentation files for:

- Internal Slack channel names (e.g., `#aiu-inductor`, `#torch-spyre`).
- Internal URLs (e.g., `*.ibm.com`, internal wikis, Jira links).
- Employee names or emails that should not be public.
- Proprietary tool names or internal codenames not meant for public docs.
- References to internal build systems or infrastructure.

Flag any findings and suggest replacements.

## 11. Cross-Reference and Link Integrity

- Check all relative links between docs pages resolve correctly.
- Check all external links (GitHub, PyTorch docs, Python docs) are valid.
- Verify image references in `_static/images/` — every referenced image exists,
  every image file is referenced somewhere.
- Check `intersphinx_mapping` in `conf.py` points to valid inventory URLs.

## 12. Build Verification

- Run `python -m sphinx docs/source docs/build/html -W --keep-going` and
  report any warnings or errors.
- Verify `suppress_warnings` in `conf.py` only suppresses intentional warnings
  (e.g., mocked autodoc), not real issues.

## Output Format

For each issue found, report:

```
### [SECTION] File: path/to/file.md

**Issue:** Description of the problem.
**Evidence:** What the docs say vs what the code shows.
**Fix:** Suggested correction.
**Severity:** critical | warning | info
```

At the end, provide a summary table:

| Section | Issues | Critical | Warnings | Info |
|---------|--------|----------|----------|------|
| ...     | ...    | ...      | ...      | ...  |
