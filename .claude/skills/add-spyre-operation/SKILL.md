---
name: add-spyre-operation
description: "Guide for adding new operations to the Spyre backend. Covers direct ATen-to-OpFunc mappings, Spyre-specific decompositions, and custom ops with lowerings. Use when asked to add, implement, or enable a new op on Spyre."
---

# Adding a New Spyre Operation

This skill walks through the three patterns for adding operations to the
torch-spyre compiled path, plus test requirements.

Read `docs/adding_operations.md` for the canonical reference. Use the
companion files in this skill directory for quick decision-making:

- `decision-tree.md` — which pattern to use
- `op-checklist.md` — step-by-step checklist of files to touch

---

## Pattern 1: Direct ATen → OpFunc Mapping

**When:** A pointwise ATen op maps directly to a single Spyre OpFunc.

**What to do:** Add a `@staticmethod` method to `SpyreOpFuncs` in
`torch_spyre/_inductor/spyre_kernel.py`.

**Examples:** `add`, `softplus`, `reciprocal`, `sigmoid`

```python
# In SpyreOpFuncs class (spyre_kernel.py):
@staticmethod
def my_op(node, kernel):
    # For simple ops, just return the op name string
    return "my_op_name"

    # For ops needing op_info (non-tensor arguments like constants):
    return ("my_op_name", {"constant_value": some_value})
```

If Inductor has a default decomposition that breaks the op into pieces you
don't want, adding the `SpyreOpFuncs` method overrides that decomposition
automatically.

---

## Pattern 2: Spyre-Specific Decomposition

**When:** The ATen op should be rewritten into other ops that Spyre already
supports, at the FX graph level (before lowering).

**What to do:** Register a decomposition in
`torch_spyre/_inductor/decompositions.py` using
`@register_spyre_decomposition`.

**Examples:** `layer_norm` (→ exx2 + layernormscale + layernormnorm), `gt`
(→ ge * ne)

```python
# In decompositions.py:
@register_spyre_decomposition([torch.ops.aten.my_op.default])
def my_op_decomposition(input, ...):
    # Rewrite using ops Spyre already supports
    return supported_op_a(supported_op_b(input))
```

---

## Pattern 3: Custom Op + Lowering

**When:** The operation has no ATen equivalent — it's a Spyre-specific fused
or specialized op.

**What to do:** Touch three files:

1. **`torch_spyre/_inductor/customops.py`** — Define the custom op:

   ```python
   @torch.library.custom_op("spyre::my_op", mutates_args=())
   def my_op(input: torch.Tensor) -> torch.Tensor:
       # CPU fallback implementation
       return ...

   @my_op.register_fake
   def _(input):
       # Shape/dtype inference for tracing
       return input.new_empty(input.size())
   ```

2. **`torch_spyre/_inductor/lowering.py`** — Register the lowering:

   ```python
   @register_spyre_lowering(torch.ops.spyre.my_op)
   def my_op_lowering(input):
       # Lower to Inductor IR (typically Pointwise or SpyreReduction)
       ...
   ```

3. **`torch_spyre/_inductor/spyre_kernel.py`** — Add to SpyreOpFuncs:

   ```python
   @staticmethod
   def my_op(node, kernel):
       return "my_op_name"
   ```

**Examples:** `spyre.clamp`, `spyre.gelu`

**Alternative:** If the custom op can be decomposed away before lowering,
register a decomposition instead of a lowering (see Pattern 2). Example:
`spyre.compact`.

---

## SuperDSC Codegen

If the op requires new SuperDSC descriptor generation, modify:

- `torch_spyre/_inductor/codegen/compute_ops.py` — for pointwise/reduction ops
- `torch_spyre/_inductor/codegen/data_ops.py` — for data movement ops
- `torch_spyre/_inductor/codegen/superdsc.py` — `generate_sdsc()` dispatch

The `generate_sdsc()` function in `superdsc.py` dispatches based on the op name
in the `KernelSpec`. Add a new branch or handler for your op.

---

## Op Name Constants

If the op needs a constant name used across multiple files, add it to
`torch_spyre/_inductor/constants.py`:

```python
MY_OP = "my_op"
```

---

## Fallback Registration

If the op should fall back to CPU when not supported in a particular mode,
register it in `torch_spyre/fallbacks.py`.

---

## Test Requirements

Every new op must have tests. See the `write-spyre-op-test` skill for details.
At minimum:

- **Compiled-path test** in `tests/_inductor/test_inductor_ops.py` using
  `compare_with_cpu()` or `compare()`
- **Shape variety:** 1D through 4D, stick-aligned (multiples of 64) and
  non-aligned sizes
- **Default dtype:** `torch.float16`

---

## Quick Reference: Files to Touch

| Pattern | Files |
|---|---|
| Direct mapping | `spyre_kernel.py` |
| Decomposition | `decompositions.py` |
| Custom op | `customops.py` + `lowering.py` + `spyre_kernel.py` |
| SuperDSC | `codegen/compute_ops.py` or `codegen/data_ops.py` |
| Constants | `constants.py` |
| Fallback | `fallbacks.py` |
| Tests | `test_inductor_ops.py` (compiled), `test_ops.py` (eager) |
