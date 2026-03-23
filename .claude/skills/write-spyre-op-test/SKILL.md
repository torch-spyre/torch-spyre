---
name: write-spyre-op-test
description: "Guide for writing compiled-path operator tests using the ParameterizedTestMeta framework and compare_with_cpu utilities. Use when adding tests for new or existing Spyre ops in tests/_inductor/test_inductor_ops.py."
---

# Writing Compiled-Path Operator Tests

This skill covers the `ParameterizedTestMeta` + `compare_with_cpu` pattern
used in `tests/_inductor/test_inductor_ops.py` — the standard way to test
individual operations on the Spyre compiled path. See `test-template.py` in
this directory for a ready-to-use skeleton.

> **Scope:** This skill is specific to parameterized op tests. Other test
> styles in the repo (module tests in `test_modules.py`, building-block
> tests in `test_building_blocks.py`, fallback tests in `test_fallbacks.py`,
> layout tests in `tests/tensor/`) follow different patterns.

---

## Where Op Tests Go

| Test type | File |
|---|---|
| Compiled-path op tests | `tests/_inductor/test_inductor_ops.py` |
| Eager-path op tests | `tests/test_ops.py` |

All test utilities live in `tests/_inductor/utils_inductor.py`.

---

## ParameterizedTestMeta

The `ParameterizedTestMeta` metaclass generates parameterized test methods
from a `PARAMS` dictionary. This is the standard pattern for compiled-path
op tests in `test_inductor_ops.py`.

### PARAMS Structure

```python
class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        # Key: (test_name_prefix, base_func_name)
        # Value: dict with optional "ops_dict" and required "param_sets"

        # With ops_dict (cross product: ops × param_sets)
        ("test_pointwise_unary_op", "test_unary_op"): {
            "ops_dict": {
                "abs": torch.abs,
                "neg": torch.neg,
                "relu": torch.relu,
            },
            "param_sets": make_param_dict([
                ((256,),),           # 1D stick-aligned
                ((67, 256),),        # 2D
                ((67, 71, 256),),    # 3D
            ]),
        },

        # Without ops_dict (base function has concrete implementation)
        ("test_layer_norm", "test_layer_norm_base"): {
            "param_sets": {
                "2d": (cached_randn((67, 256)),),
                "3d": (cached_randn((4, 67, 256)),),
            },
        },
    }
```

### Generated Test Names

- With `ops_dict`: `{test_name_prefix}_{op_name}_{test_case}`
  - Example: `test_pointwise_unary_op_abs_256`
- Without `ops_dict`: `{test_name_prefix}_{test_case}`
  - Example: `test_layer_norm_2d`

### Base Function Signatures

```python
# With ops_dict — receives (self, op, *params):
def test_unary_op(self, op, x):
    compare_with_cpu(lambda a: op(a), x)

# Without ops_dict — receives (self, *params):
def test_layer_norm_base(self, x):
    compare_with_cpu(lambda a: torch.nn.functional.layer_norm(a, [256]), x)
```

---

## Compare Functions

Import from `utils_inductor`:

### `compare_with_cpu(fn, *args, atol=0.1, rtol=0.1)`

The most common pattern. Compares:

1. Uncompiled CPU execution
2. Compiled Spyre execution
3. Compiled CPU execution (optional, `cpu_compile=True`)

```python
def test_my_op(self, op, x):
    compare_with_cpu(lambda a: op(a), x)
```

### `compare(fn, *args, atol=0.0, rtol=0.0, cpu_atol=0.1, cpu_rtol=0.1)`

3-way comparison: compiled Spyre vs uncompiled CPU vs sendnn backend.
Use when you need sendnn validation.

### `compare_with_pytorch(fn, fn_pytorch, *args, atol=0.1, rtol=0.1)`

Compare compiled Spyre function against an uncompiled PyTorch reference.
Use when the reference implementation differs from the test function.

### `compare_with_sendnn(fn, *args, atol=0.0, rtol=0.0)`

Compare compiled Spyre against sendnn backend only. Use for bit-exact
comparisons with the reference compiler.

---

## Helper Functions

### `cached_randn(shape, differentiation=None, abs=False, dtype=torch.float16, scale=1.0)`

LRU-cached random tensor generation. Use `differentiation` parameter to get
different tensors with the same shape. Use `abs=True` for ops that need
positive inputs (sqrt, log, rsqrt).

### `init_helper(shapes, dtype=torch.float16, cached=True)`

Initialize a tuple of tensors from a list of shape tuples.

### `make_param_dict(cases)`

Convert a list of shape-tuple cases into a `{key: tensors}` dict for
`param_sets`. Keys are auto-generated from shapes (e.g., `"67x256"`).

### `shapes2key(shapes)`

Convert shape tuples to a string key: `((4, 8), (4, 8))` → `"4x8_4x8"`.

---

## Shape Selection Guidelines

Include variety across these dimensions:

- **Dimensionality:** 1D, 2D, 3D, and 4D where applicable
- **Stick alignment:** Include both multiples of 64 (stick-aligned) and
  non-multiples (e.g., 67, 71) to test padding behavior
- **Common sizes:** `(256,)`, `(67, 256)`, `(67, 71, 256)`,
  `(7, 12, 32, 64)`

```python
make_param_dict([
    ((256,),),                # 1D, stick-aligned
    ((67, 256),),             # 2D, non-aligned first dim
    ((67, 71, 256),),         # 3D, non-aligned dims
    ((7, 12, 32, 64),),       # 4D
])
```

For binary ops, use matching shapes:

```python
make_param_dict([
    ((256,),) * 2,
    ((67, 256),) * 2,
])
```

---

## Conventions

- **Default dtype:** `torch.float16`
- **Random seed:** `torch.manual_seed(0xAFFE)` at class level
- **Default tolerances:** `atol=0.1, rtol=0.1`
- **License header:** Every test file needs the 14-line Apache 2.0 header
- **Imports:** Use `import regex` not `import re`

---

## Running Tests

```bash
# All compiled-path tests
python3 -m pytest tests/_inductor/test_inductor_ops.py

# Single test
python3 -m pytest tests/_inductor/test_inductor_ops.py -k "test_my_op"

# All tests
python3 -m pytest tests/
```
