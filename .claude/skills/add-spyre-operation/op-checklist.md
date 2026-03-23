# Operation Implementation Checklist

Use this checklist when adding a new operation to the Spyre backend.
Check off each item as you complete it.

## 1. Determine the Pattern

- [ ] Read `decision-tree.md` to pick Pattern 1, 2, or 3
- [ ] Check if Inductor already has a default decomposition for this op
- [ ] Review existing similar ops for reference

## 2. Implement the Operation

### Pattern 1: Direct ATen → OpFunc

- [ ] Add `@staticmethod` method to `SpyreOpFuncs` in
  `torch_spyre/_inductor/spyre_kernel.py`
- [ ] Method name must match the ATen op handler name
- [ ] Return the Spyre op name string (or tuple with `op_info` dict)

### Pattern 2: Spyre-Specific Decomposition

- [ ] Add decomposition function in
  `torch_spyre/_inductor/decompositions.py`
- [ ] Use `@register_spyre_decomposition([torch.ops.aten.<op>.default])`
- [ ] Decomposition must only use ops that Spyre already supports

### Pattern 3: Custom Op + Lowering

- [ ] Define custom op in `torch_spyre/_inductor/customops.py`
  - [ ] Use `@torch.library.custom_op("spyre::<name>", mutates_args=())`
  - [ ] Implement CPU fallback behavior
  - [ ] Register fake function with `@<op>.register_fake`
- [ ] Register lowering in `torch_spyre/_inductor/lowering.py`
  - [ ] Use `@register_spyre_lowering(torch.ops.spyre.<name>)`
- [ ] Add `SpyreOpFuncs` method in
  `torch_spyre/_inductor/spyre_kernel.py`

## 3. SuperDSC Codegen (if needed)

- [ ] Add handler in `torch_spyre/_inductor/codegen/compute_ops.py`
  (pointwise/reduction) or `codegen/data_ops.py` (data movement)
- [ ] Update dispatch in `torch_spyre/_inductor/codegen/superdsc.py` if
  the op type is new
- [ ] Add op name constant to `torch_spyre/_inductor/constants.py`
  if used across files

## 4. Write Tests

- [ ] Add compiled-path test in `tests/_inductor/test_inductor_ops.py`
  - [ ] Use `compare_with_cpu()` or `compare()` from `utils_inductor`
  - [ ] Test shapes: 1D, 2D, 3D, 4D
  - [ ] Include stick-aligned (multiples of 64) and non-aligned sizes
  - [ ] Use `torch.float16` as default dtype
- [ ] Add eager-path test in `tests/test_ops.py` (if op has eager support)
- [ ] Add building-block test in `tests/_inductor/test_building_blocks.py`
  (if op is part of a larger module like LayerNorm)

## 5. Final Checks

- [ ] Run `pre-commit run --all-files`
- [ ] Run `python3 -m pytest tests/_inductor/test_inductor_ops.py` (at
  minimum)
- [ ] Verify Apache 2.0 license headers on all new/modified files
- [ ] Use `import regex` not `import re` in any new Python files
- [ ] Sign off commit: `git commit -s`

## 6. Fallback (if needed)

- [ ] Register CPU fallback in `torch_spyre/fallbacks.py` if the op
  cannot run on Spyre in some configurations
