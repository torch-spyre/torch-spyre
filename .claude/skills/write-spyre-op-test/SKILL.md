---
name: write-spyre-op-test
description: "Guide for writing compiled-path operator tests using the ParameterizedTestMeta framework and compare_with_cpu utilities. Use when adding tests for new or existing Spyre ops in tests/inductor/test_inductor_ops.py."
---

# Writing Compiled-Path Operator Tests

This skill covers the `ParameterizedTestMeta` + `compare_with_cpu` pattern
used in `tests/inductor/test_inductor_ops.py` — the standard way to test
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
| Compiled-path op tests | `tests/inductor/test_inductor_ops.py` |
| Eager-path op tests | `tests/test_spyre.py` |

All test utilities live in `tests/inductor/utils_inductor.py`.

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

## Recording Expected Failures

A case that does not work today is recorded as an expectation. It is never recorded by
calling `pytest.xfail` or by catching the error, because then the body does not run
and the test can never tell you the problem is gone. The rules are in the
**Test Invariants** section of `CLAUDE.md`; this section shows how to apply them.

First run the case with the expectation off (`--runxfail`; an OOT-wrapped test still
reports an xfail, with the original message in `wasxfail`) and see what it does:

| The case... | Record it as |
|---|---|
| passes | nothing: remove the entry |
| raises a stable error (a deliberate rejection, or a missing feature) | `expect_raise` |
| returns wrong values, or hits a bug | `expect_fail` (strict xfail) as `{case: reason}`, citing an open issue |
| passes on some runs or some backend builds | `expect_fail_unstable` (non-strict), citing an issue |
| kills the process or faults the device | `skip` or `device_fault`, citing an issue |

The keys sit next to `param_sets` in `PARAMS`:

```python
("test_my_op", "test_my_op_base"): {
    "ops_dict": {...},
    "param_sets": {...},
    # {case or "<op>_<case>": message fragment}. Must raise, and the message must
    # match. Fails if it stops raising ("DID NOT RAISE") or raises something else.
    "expect_raise": {"fp32_4x32": "cannot rescale device layout"},
    # {case or "<op>_<case>": reason}. Strict xfail: fails if the case starts passing.
    # The reason, with the issue, appears in the xfail report. A plain list of cases
    # also works, but its reason only names the case.
    "expect_fail": {"fp16_67x71": "wrong values, #1234"},
    # {case or "<op>_<case>": reason}. Non-strict xfail, for an outcome that varies.
    "expect_fail_unstable": {"fp16_4x63": "passes on some runs, #5285"},
},
```

- A bare case name applies to every op in `ops_dict`; `"<op>_<case>"` to one op.
- `expect_fail` is strict but cannot narrow the exception with `raises=`, so a failure
  with a stable error is better as `expect_raise`. An `expect_fail` mapping entry needs
  a non-empty reason; that is checked when the class is created.
- `expect_raise` and `expect_fail_unstable` entries that match no generated test, or
  that also appear under another key, fail when the class is created. A stale
  `expect_fail` or `skip` entry is silently ignored, so delete entries when you fix
  a case.
- When the case is fixed, remove its entry in the same PR. A strict xfail that
  starts passing fails the run.

For a hand-written test, assert the rejection yourself and tag it so the LX-planning
suite does not copy it (the rejection happens before the suite's second op is built):

```python
from utils_inductor import expects_raise

@expects_raise
def test_cumprod_dim0_rejected(self):
    # TODO: a missing feature, not a design limit. When aten::cumprod.out is
    # registered this stops raising, and the test has to become a positive one.
    x = cached_randn((67, 256), scale=0.1)
    with pytest.raises(NotImplementedError, match=r"Could not run 'aten::cumprod\.out'"):
        self.compare_with_cpu(lambda x: torch.cumprod(x, dim=0), x, run_eager=False)
```

When whether it fails depends on a parameter or a fixture, in a pytest-style test, use
`strict_xfail(request, raises=<exception type>, reason=...)` from `utils_inductor`.

Also keep the test itself sound: no repeated indices in scatter-like ops, clone
`cached_randn` results before mutating them, justify a tolerance against an fp32
reference, and run `make check-all-configs` after adding a test.

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
python3 -m pytest tests/inductor/test_inductor_ops.py

# Single test
python3 -m pytest tests/inductor/test_inductor_ops.py -k "test_my_op"

# All tests
python3 -m pytest tests/
```
