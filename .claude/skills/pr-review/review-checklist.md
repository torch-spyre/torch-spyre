# PR Review Checklist

Quick reference for reviewing torch-spyre pull requests.

## Mandatory Checks (BLOCKERs if Missing)

- [ ] **Apache 2.0 headers** on all new source files (Python: 14-line block,
  C++: `/* */` block)
- [ ] **Signed commits** — every commit has `Signed-off-by:` (DCO)
- [ ] **Pre-commit passes** — ruff, clang-format, cpplint, mypy, pymarkdown
- [ ] **`import regex`** — never `import re` in Python files
- [ ] **Tests exist** for new/changed functionality
- [ ] **`Unsupported()`** used for unsupported-op errors (not bare
  `RuntimeError`)

## Code Quality

- [ ] Follows Google Python / C++ style guide
- [ ] Line length ≤ 88 characters (Python)
- [ ] No commented-out code left behind
- [ ] Clear variable and function names

## Tensor Layout

- [ ] `dim_map` values correct for any new layout logic
- [ ] Stick padding handled (64 fp16 elements per 128-byte stick)
- [ ] `FixedTiledLayout` includes `device_layout`
- [ ] No raw stride assumptions on tiled tensors

## Test Coverage

- [ ] `compare_with_cpu()` or `compare()` used in
  `tests/inductor/test_inductor_ops.py`
- [ ] Shape variety: 1D, 2D, 3D, 4D
- [ ] Stick-aligned and non-aligned sizes tested
- [ ] `torch.float16` as default dtype
- [ ] `ParameterizedTestMeta` for parameterized tests
- [ ] `torch.manual_seed(0xAFFE)` set at class level

## Path Impact

- [ ] Eager path changes have `tests/test_spyre.py` or
  `tests/test_fallbacks.py` coverage
- [ ] Compiled path changes have `tests/inductor/` coverage
- [ ] Cross-path changes tested in both suites

## Documentation

- [ ] `docs/` updated for user-facing changes
- [ ] Complex logic has inline comments
- [ ] New ops documented in relevant places
