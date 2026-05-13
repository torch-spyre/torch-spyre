---
name: pr-review
description: "Review pull requests for the torch-spyre project. Checks license headers, pre-commit compliance, signed commits, tensor layout correctness, test coverage, and documentation."
---

# PR Review for torch-spyre

When asked to review a PR, follow this systematic process. Use `gh pr diff`
and `gh pr view` to inspect the changes.

See `review-checklist.md` in this directory for a quick reference checklist.

---

## Review Process

### Step 1: Fetch PR Information

```bash
gh pr view <number>
gh pr diff <number>
gh pr checks <number>
```

### Step 2: Run Through Checklist

Work through each category below. Flag issues with severity:

- **BLOCKER** — Must fix before merge
- **SUGGESTION** — Improvement, not blocking
- **QUESTION** — Needs clarification from author

---

## Review Categories

### 1. License Headers

Every source file must have the Apache 2.0 license header.

**Python** (14 lines):

```python
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

**C++** (`/* */` block):

```cpp
/* Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 ...
 */
```

### 2. Pre-commit Compliance

The PR must pass all pre-commit hooks:

- **ruff** — Python linting and formatting (line length 88)
- **clang-format** — C++ formatting
- **cpplint** — C++ linting
- **mypy** — Type checking
- **pymarkdown** — Markdown linting
- **yamlfmt** — YAML formatting
- **enforce-import-regex** — Must use `import regex`, never `import re`

### 3. Signed Commits (DCO)

Every commit must have a `Signed-off-by:` line. Check with:

```bash
gh pr view <number> --json commits --jq '.commits[].messageBody'
```

### 4. Tensor Layout Correctness

For changes touching layout-related code, verify:

- `SpyreTensorLayout` `dim_map` values are correct
- Stick dimension padding is handled (64 elements for fp16)
- `FixedTiledLayout` usage includes proper `device_layout`
- No assumptions about contiguous memory where tiled layout applies

### 5. Dual Path Impact

Check whether the change affects eager mode, compiled mode, or both:

- **Eager-only** changes: `torch_spyre/ops/`
- **Compiled-only** changes: `torch_spyre/_inductor/`,
  `torch_spyre/execution/`
- **Both paths**: `torch_spyre/csrc/`, `torch_spyre/__init__.py`,
  `torch_spyre/_monkey_patch.py`

Ensure tests cover the affected path(s).

### 6. Test Coverage

- New ops must have tests in `tests/inductor/test_inductor_ops.py`
  using `compare_with_cpu()` or `compare()`
- Shape variety: 1D through 4D, stick-aligned and non-aligned sizes
- Default dtype should be `torch.float16`
- Tests use `ParameterizedTestMeta` for parameterized cases
- Building-block tests for module-level changes

### 7. Work Division

For changes to `torch_spyre/_inductor/work_division.py` or related code:

- Validate against `docs/source/compiler/work_division_planning.md`
- Check dimension label correctness
- Verify multi-core splitting logic

### 8. Error Handling

- Use `Unsupported()` from `torch_spyre/_inductor/errors.py` for
  unsupported operations (not generic `RuntimeError`)
- Error messages should be descriptive: `Unsupported("thing that failed")`
  produces `"Spyre backend does not support: thing that failed"`

### 9. Documentation

- User-facing behavior changes need `docs/` updates
- New ops should be mentioned in relevant documentation
- Complex logic should have inline comments

---

## Output Format

Structure your review as:

```markdown
## PR #<number>: <title>

### Summary
<1-2 sentence summary of what the PR does>

### Review

#### BLOCKERs
- [ ] <issue description> (file:line)

#### SUGGESTIONs
- [ ] <improvement> (file:line)

#### QUESTIONs
- [ ] <question for author>

### Verdict
<APPROVE / REQUEST_CHANGES / COMMENT>
```
