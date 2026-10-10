# torch-spyre — Claude Code Project Instructions

torch-spyre is an **out-of-tree PyTorch backend** that registers the
**IBM Spyre AI Accelerator** as a first-class PyTorch device (`"spyre"`)
via the PrivateUse1 mechanism.

## Conventions

- **License:** Apache 2.0 — every source file must carry the 14-line Python
  header (or C++ `/* */` equivalent). See any file in `torch_spyre/` for the
  template.
- **Style:** Google Python Style Guide, Google C++ Style Guide.
- **Imports:** Use `import regex` (aliased as `re` when needed), **never**
  `import re`. A pre-commit hook enforces this.
- **Commits:** Sign off every commit with `git commit -s` (DCO).
- **Linting:** Run `pre-commit run --all-files` before pushing. Hooks include
  ruff, clang-format, cpplint, mypy, pymarkdown, and yamlfmt.
- **Line length:** 88 characters (ruff).

## Build and Test

```bash
# Run all tests
python3 -m pytest tests/

# Run pre-commit checks
pre-commit run --all-files
```

Test sub-suites:

| Suite | Path |
|---|---|
| Eager ops | `tests/test_spyre.py`, `tests/test_fallbacks.py` |
| Compiled ops | `tests/inductor/test_inductor_ops.py` |
| Building blocks | `tests/inductor/test_building_blocks.py` |
| Tensor layout | `tests/tensor/` |

## Test Invariants

Rules for every test under `tests/`. A test that cannot fail, or that fails for the
wrong reason, hides regressions and lets stale expectations pile up; each rule below
exists because that happened here. The `write-spyre-op-test` skill shows how to apply
them with `PARAMS`, and `pr-review` checks them.

### A test body always runs

- **Do not use `pytest.xfail(...)` to expect a failure.** It raises before the rest of
  the body runs, so the test can never XPASS, cannot be strict, and never shows that a
  bug is gone. The same applies to a `try`/`except` that turns an error into an xfail
  or a skip. Declare the expectation instead: `@pytest.mark.xfail`, a `PARAMS` key
  (`expect_fail`, `expect_fail_unstable`, `expect_raise`), or, when it depends on a
  fixture or parameter, `strict_xfail(request, raises, reason)` from
  `tests/inductor/utils_inductor.py`, which applies the marker from inside the body.
  (`expected_unimplemented` in `tests/inductor/test_solver_auto_coarse_tiling.py` is
  the one exception: it xfails only on the expected error and fails if the body
  passes.)
- **A failure that kills the process (SIGFPE, SIGABRT, a hang) is a skip, not an
  xfail**, because the run dies with it. Cite an issue in the reason.

### Say exactly what is expected

- An xfail is `strict=True`, so an unexpected pass fails the test. Where you write the
  marker (`@pytest.mark.xfail`, `strict_xfail`) also give it `raises=<exception type>`,
  so any other failure fails the test too. The `expect_fail` key is strict but cannot
  narrow the exception; for a failure with a stable error use `expect_raise`. CI runs
  xfail-marked tests, and a strict XPASS fails the run. When a bug is fixed, the same
  PR removes its xfail.
- `expect_fail_unstable` (non-strict) is only for an outcome that really varies between
  runs or backend builds, with a reason and an issue. It is not a way to silence a
  strict XPASS you do not understand.
- A deliberate rejection, or a missing feature that fails with a stable error
  (`Unsupported`, `Could not run 'aten::X'`, a layout rejection), is an assertion, not
  an xfail: `expect_raise` with a message fragment, or, in a hand-written test,
  `pytest.raises(match=...)` with the test tagged `@expects_raise` (from
  `tests/inductor/utils_inductor.py`). It then fails if the case stops raising or raises
  something else. For a missing feature say so in a `TODO`: when it lands the test must
  become a positive one. Keep xfail for wrong values and bugs.
- Every xfail and skip says why. For a bug, the reason describes the failure **today**
  and cites an **open** issue; a reason that cites a closed issue, or names an error
  the body no longer hits, is stale. Give `expect_fail` as a `{case: reason}` mapping,
  not a list, so the reason and the issue appear in the xfail report; a list only names
  the case. (A skip for a missing hardware or configuration
  condition, such as `SENCORES=1`, needs no issue.) To check a reason, run the body
  with the expectation disabled (`--runxfail`; OOT-wrapped tests still convert the
  failure to an xfail, with the original message in `wasxfail`).

### Inputs and checks must be sound

- The test must be able to tell a right answer from a wrong one. Do not scatter or
  `index_put` with repeated indices (the winner is undefined). Justify a tolerance
  against an fp32 reference: do not widen it until the test passes, and do not leave it
  tighter than fp16 accumulation allows. Compare on the host when the device has no
  kernel for the check itself (for example `torch.allclose` on device tensors).
- Never mutate a shared input. `cached_randn` returns the same tensor for the same
  arguments; clone it before setting `requires_grad` or running in-place ops.
- Tests must pass in any order and in any grouping. CI runs one file per shard, but
  `pytest tests/inductor/` runs them in one process, so check combined runs.

### Tests must be selected and must run

- Every test must be selected by a shard config in `tests/configs/`. Run
  `make check-all-configs` after adding or renaming a test.
  `unlisted_test_mode: skip` drops an unlisted test silently.
- The LX-planning suite copies each `TestOps` test and appends a second op. A test
  that asserts a rejection of the op under test gains nothing from that, because the
  error is raised before the second op is built, so it should not be copied. The
  suite skips xfail-marked tests and tests tagged `_expects_raise`, which the
  `expect_raise` key and the `@expects_raise` decorator both set.

### Local results can differ from CI

- Backend-level outcomes (`dbo-opt` errors, masking, layout limits) can differ between
  deeptools builds, and so between a local machine and CI. Before adding or removing an
  xfail because of a local failure, check PR CI or the latest nightly JUnit, and say in
  the PR when the evidence is local only.

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `TORCH_SPYRE_DEBUG=1` | Enable C++ debug logging and `-O0` builds |
| `SENCORES` | Number of Spyre cores (1–32, default 32) |
| `LX_PLANNING=1` | Enable LX scratchpad memory planning |
| `HBM_POOL_PLANNING=1` | Enable HBM-pool planning for intermediates not in LX |
| `LAYOUT_SOLVER` | LX layout solver: `greedy`, `firstfit`, `bestfit`, `cpsat` (default), `simulated_annealing` |
| `TORCH_SPYRE_NATIVE_PACKER` | Use the C++ layout packer in the `simulated_annealing` solver (default `1`; `0` = pure Python) |
| `TORCH_LOGS="+inductor"` | Verbose Inductor logging |
| `TORCH_COMPILE_DEBUG=1` | Dump Inductor debug artifacts |
| `TORCH_SPYRE_DOWNCAST_WARN=0` | Suppress int64→int32 downcast warnings |
| `SPYRE_DUMP_COST=1` | Predicted-runtime report after pre-scheduling |
| `TORCH_SPYRE_TIMING=1` | Structured per-compile frontend timing records |
| `TORCH_SPYRE_TIMING_OUT` | Where to write them (pid inserted: `rec.json` -> `rec.<pid>.json`) |
| `TORCH_SPYRE_FRONTEND_ONLY=1` | Stop each compile before the backend compiler; no runnable kernel, measurement only |

## Spyre Hardware Basics

- Default dtype: `torch.float16`
- **Stick:** 128-byte aligned memory chunk = 64 elements at fp16
- Device name constant: `torch_spyre.constants.DEVICE_NAME` = `"spyre"`
- Up to 32 cores per accelerator, >300 TOPS at 75W

## Architecture

See `docs/source/` for detailed architecture documentation:

- `docs/source/architecture/spyre_accelerator.md` — Spyre accelerator overview
- `docs/source/compiler/architecture.md` — compilation pipeline
- `docs/source/user_guide/tensors_and_layouts.md` — tiled tensor layout specification
- `docs/source/compiler/adding_operations.md` — how to add new operations
- `docs/source/compiler/work_division_planning.md` — multi-core work division

## Compiler Pass Conventions

**Modifying `ComputedBuffer.inner_fn`: wrap, never reconstruct.**
Use a `WrapperHandler` subclass (see `WrapperHandler` in
`torch._inductor.ops_handler`) to intercept specific ops and install it
with `V.set_ops_handler(handler)` inside the original `inner_fn`. Do NOT
rebuild `inner_fn` from scratch by re-creating index expressions — those
expressions are symbolic and become stale as soon as the pass runs,
causing silent wrong-code bugs (see issue #2797). Canonical examples of the
correct pattern: `NameSwapHandler` in `insert_restickify.py`,
`_SplitOpsHandler`/`_IntermediateOpHandler` in `split_multi_ops.py`.

## Skills

Task-specific guidance is available in `.claude/skills/`. These cover:

- **project-overview** — repo layout, Spyre architecture, compilation pipeline
- **add-spyre-operation** — patterns for adding new ops
- **write-spyre-op-test** — compiled-path op test framework and patterns
- **pr-review** — PR review checklist
- **debug-compilation** — troubleshooting compilation failures
- **write-rfc** — design proposal workflow (RFCs now at https://github.com/torch-spyre/rfcs)

### Writing SKILL.md Files

When creating or editing `.claude/skills/*/SKILL.md` files:

- **YAML frontmatter `description`:** Use a quoted single-line string, not
  a multi-line `>-` block scalar. Pymarkdown does not understand YAML
  frontmatter and will mangle indented continuation lines.
  - Good: `description: "One line describing the skill."`
  - Bad: `description: >-` followed by indented lines
- **Python templates in skills:** Add `# noqa: F401` to imports that are
  only used in commented-out example code, so ruff does not remove them.
