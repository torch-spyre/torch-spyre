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
| Eager ops | `tests/test_ops.py` |
| Compiled ops | `tests/_inductor/test_inductor_ops.py` |
| Building blocks | `tests/_inductor/test_building_blocks.py` |
| Tensor layout | `tests/tensor/` |

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `TORCH_SPYRE_DEBUG=1` | Enable C++ debug logging and `-O0` builds |
| `SENCORES` | Number of Spyre cores (1–32, default 32) |
| `LX_PLANNING=1` | Enable LX scratchpad memory planning |
| `TORCH_LOGS="+inductor"` | Verbose Inductor logging |
| `TORCH_COMPILE_DEBUG=1` | Dump Inductor debug artifacts |
| `TORCH_SPYRE_DOWNCAST_WARN=0` | Suppress float32→float16 warnings |

## Spyre Hardware Basics

- Default dtype: `torch.float16`
- **Stick:** 128-byte aligned memory chunk = 64 elements at fp16
- Device name constant: `torch_spyre.constants.DEVICE_NAME` = `"spyre"`
- Up to 32 cores per accelerator, >300 TOPS at 75W

## Architecture

See `docs/` for detailed architecture documentation:

- `docs/spyre.md` — Spyre accelerator overview
- `docs/compiler_architecture.md` — compilation pipeline
- `docs/tensor_layouts.md` — tiled tensor layout specification
- `docs/adding_operations.md` — how to add new operations
- `docs/work_division_planning.md` — multi-core work division

## Skills

Task-specific guidance is available in `.claude/skills/`. These cover:

- **project-overview** — repo layout, Spyre architecture, compilation pipeline
- **add-spyre-operation** — patterns for adding new ops
- **write-spyre-op-test** — compiled-path op test framework and patterns
- **pr-review** — PR review checklist
- **debug-compilation** — troubleshooting compilation failures
- **write-rfc** — design proposal workflow

### Writing SKILL.md Files

When creating or editing `.claude/skills/*/SKILL.md` files:

- **YAML frontmatter `description`:** Use a quoted single-line string, not
  a multi-line `>-` block scalar. Pymarkdown does not understand YAML
  frontmatter and will mangle indented continuation lines.
  - Good: `description: "One line describing the skill."`
  - Bad: `description: >-` followed by indented lines
- **Python templates in skills:** Add `# noqa: F401` to imports that are
  only used in commented-out example code, so ruff does not remove them.
