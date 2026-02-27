
# RFC-0186: Model-Centric Functional Verification for OpFunc Enablement

**Authors:**
- Tuan M. Hoang Trong (YKT)
- Kazuaki Ishizaki (TRL)
- Umamaheswari Devi (IRL)

**Tracking issue:** #241
**Target repository:** `torch-spyre/torch-spyre`
**Related issues/PRs:** #323

---

## Summary
This RFC proposes a **model-centric functional verification** framework built on **pytest** to ensure the correctness of individual **torch operations** used in target models such as GPT-OSS, Granite-4h, Granite-Vision, and Granite-Speech.
Unlike existing PyTorch tests that rely on synthetic parameters and small tensor shapes, this RFC approach uses actual tensor shapes, parameters, and data types from the target models, delivering realistic and comprehensive coverage.

This framework is essential for **OpFunc enablement**, as it verifies correctness with **model-specific parameters** rather than synthetic test cases.
By doing so, we eliminate blind spots that existing tests **may not cover**.

---

## Motivation

### Background
We have repeatedly encountered cases where inference results appear mostly
correct, yet certain **torch operations** produce incorrect outputs. These
subtle inconsistencies block feature development (e.g., fine-tuning) and
introduce fragility into the stack. To build a stable foundation and accelerate
innovation, **model-centric functional verification** using **actual model
parameters** is critical.

### Problems
In the past, we observed these issues:
- Operations with certain **dimensions** work correctly, but operations with
  other dimensions generate **incorrect results**.
- Operations with specific **values** in a tensor generate **incorrect results**.

These observations lead to the need for verification with **actual parameters**.
Existing PyTorch tests are insufficient for three critical reasons:

- **Unrealistic tensor dimensions**: Current tests use shapes that differ from
  real-world workloads, failing to reflect deployment scenarios.
- **Limited coverage of Spyre execution paths**: Small tensor sizes often avoid
  **Spyre-relevant** multi-core/tiling/scheduling paths, leaving key behaviors
  untested.
- **Synthetic data limitations**: Tests do not capture edge cases arising from
  **actual parameter distributions** in production models.

These gaps can lead to undetected correctness issues that surface only with the
target models, increasing risk and slowing development.

## Goals
- **Functional correctness**: Validate each enabled torch operation against a
  **CPU reference** using **real shapes and data** from target models.
- **Nightly monitoring**: Continuously track correctness of enabled OpFuncs
  using the industry-standard **pytest** framework.
- **Scalability**: Support multiple models (Granite-4h, GPT-OSS,
  Granite-Vision-3, Granite-Speech) with **cross-model deduplication**.
- **Maintainability**: Centralize test intent in descriptors (**shapes, dtypes,
  tolerances**) for easy updates as models evolve.

## Non-Goals (for now)
- **Performance tracking**: Out of scope initially (future metrics may be
  added).
- **Per-PR gating**: Tests run in nightly CI/CD, not on every PR.

---

## Proposed Implementation

### Design Concept
Our framework delivers:

- Leverage **pytest** as the de facto testing standard.
- **Descriptor-driven** approach (currently **YAML**) to represent actual model
  parameters for each torch operation.
- Automated test generation from descriptors, with optional **deduplication
  across models**.
- Flexible test selection via **pytest markers** for skipping, expected
  failures, and targeted runs.
- CI integration as **nightly tests**, starting with **Spyre/AIU-backed**
  models and expanding to canonical sets.

This design ensures **realistic**, **maintainable**, and **scalable**
correctness checks across diverse models—closing the gap between synthetic
testing and production reality.

### Overview
We use two cooperating frameworks:

1) **Model scan & codegen (existing, internal, modified):**  
   - Scan target models and extract the set of `torch.ops` along with input shapes and/or sample inputs.  
   - **Change:** Generate **YAML** instead of Python test files.

2) **YAML test runner (the proposed one):**  
   - Load YAML case definitions.  
   - Codegen lightweight test functions dynamically (via `pytest`).  
   - Execute ops, compare against CPU reference, and report results.
   - Skip tests if needed.

---
### Execution strategy

- Run all extracted ops for a model (subject to dedupe).
- Avoid redundant runs (global cache keyed by <op, normalized_inputs>).
- Make test parameters easy to understand for debugging (schema + description fields).

---

### Models & files
We plan per-model YAML, with option for subclasses variants:

- `tests/_inductor/models/template.yaml`
- `tests/_inductor/models/gpt-oss.yaml`
- `tests/_inductor/models/gpt-oss-spyre.yaml`  
  *Temporary, backend-specific functional correctness checks. May relax original shapes and can be deleted once features are stable.*

**Naming:** (notice to changes in internal scripts) Permanent test cases (e.g., no `_fp16`) reside in files **without** postfixes. Backend-temporary cases use the `-spyre` postfix file.

---

### Deduplication & selection
- **Cross-model deduplication (default):** If `op-A(input-B)` was already tested for model-C, skip rerunning for model-D, if that test case is part of model-D.
Deduplication is based on normalized op signature and input shapes, not model name.
- **Disable dedupe:** `--no-dedupe` (explicit).
- **Selective model runs:** Choose one or many models at runtime (e.g., run only `granite3-speech`).
- **Select cases**: integrate with pytest.ini design, register custom markers for cleanliness. **Can skip tests at each input set**.
Marker usage and registration follow pytest documentation best practices, enabling selection and skip/xfail logic

**CLI Examples**

```bash
# List available models/cases
pytest --list-models 2>&1 | tee logs_test.txt
pytest --list-cases  2>&1 | tee logs_test.txt

# Run a single case by name
pytest -q tests/_inductor/test_model_ops.py --model gpt_oss -k "mul_ok"

# Run all 'torch.add' ops for GPT-oss
pytest -q tests/_inductor/test_model_ops.py --model gpt_oss -k "torch.add"

# Run all tests ignoring default marker filters
pytest tests/_inductor/test_model_ops.py --model gpt-oss-20b -m ""

# Run for multiple models
pytest --model gpt_oss tests/
pytest --model gpt_oss --model granite4h tests/

# Show selected tests based on marks
pytest --list-cases-by-mark
pytest --list-cases-by-mark --model gpt-oss
# Show excluded tests based on marks
pytest --list-cases-by-mark --show-excluded
pytest --list-cases-by-mark --show-excluded --model gpt-oss

```

**Full argument list**

supported arguments: pytest

```bash
pytest \
  [--model <model-name>]*
  --dedupe / --no-dedupe
  --list-models, --list-cases
  --compile-backend "inductor"
    TEST_COMPILE_BACKEND: env. variable to change the default backend “inductor”
  --list-cases-by-mark [marks] [--show-excluded]
```

---

### Source Layout

Below is the proposed directory structure for the YAML-driven test framework:

```bash
tests/
├── conftest.py
└── _inductor/
    ├── init.py
    ├── test_model_ops.py          # Entry point for pytest-based tests
    ├── model_cases_loader.py      # Loads and parses YAML files
    ├── op_registry.py             # Maintains mapping of ops to test logic
    ├── runner.py                  # Executes tests dynamically from YAML
    └── models/                    # YAML manifests per model
        ├── gpt_oss.yaml
        └── granite3_speech.yaml
```

## YAML Schema

Top-level structure

```yaml

model: <model-name>
default:
  dtype: <DTYPE_MAP value>
  seed: <int>
  rtol: <float>
  atol: <float>
cases:
  - name: <case-name>
    op: <torch.op>            # as registered via OP_REGISTRY
    inputs:                   # map of arguments for the op
      tensor:                 # tensor description (see below)
      value:                  # scalar input
      py:                     # restricted Python literal (tuples, None, Ellipsis, slice, ints, floats, lists, inf, -inf, nan)
      tensor_list:            # sequences (list/tuple) of tensors
      file:                   # .pt file holding values for a tensor
    description: <text>       # e.g., trace where the op was extracted
    expects_error:            # expected error (optional) - expected error type and message substring
      type: <ExceptionType>
      match: <substring>

    # Test control
    marks: <pytest marker(s)>
    skip_if:                  # Conditional skip logic for backend/device)
      expr: <python boolean>  # if True -> skip
      reason: <text>

    # Per-case overrides of defaults
    dtype: <DTYPE_MAP value>
    seed: <int>
    rtol: <float>
    atol: <float>

```

---

### pytest.ini Configuration

The `pytest.ini` file contains default marker expressions that filter tests:

```ini
[pytest]
markers =
    paddedtensor: tests for tensors requiring padding
    largedimtensor: tests for 4D and 5D tensors
    fp32operation: tests for fp32 operations
    bf16operation: tests for bfloat16 operations
    constant: tests for operations with constants

addopts = -m "not largedimtensor and not fp32operation and not bf16operation and not constant"
```

By default, tests marked with largedimtensor, fp32operation, bf16operation, or constant are excluded. To run all tests regardless of markers, use `-m ""` option.

---

### Input to a test case

A test case is generated based on the op to test, and the input sources which is a sequence of arguments, each can be one of the following

- -tensor: randomly generated tensor(s) following shape, dtype, init, etc.
- -value: scalars, lists, tuples.
- -py: restricted Python literal expressions (tuples, None, Ellipsis, slices, ints, floats, lists, inf, -inf, nan).
- -file: load from .pt files (torch.load("path/to/file.pt")).

Rationale: Debugging error-sensitive bugs often requires specific tensor values, especially in the backward path, so YAML supports exact data (init: data) and file-based inputs (-file).

### Input tensor description

```yaml

tensor:
  shape: [dim0, dim1, ...]
  init: rand | zeros | ones | arange | randint | uniform | data   # optional
  dtype: <optional dtype>
  preset: <optional preset for non-contiguous/layout-sensitive ops>
  preset_args: <optional args for preset>

  # init_args specify bounds for random generation
  # If 'randint': allow 'low' (default 0), 'high'
  # If 'uniform': allow 'low' (default 0), 'high' (default 1)
  # If 'data': pass exact values as list(s)
```

---

### Supporting Constants

The framework uses predefined mappings for dtypes and error names to ensure consistency between YAML descriptors and Python runtime.

#### DTYPE_MAP
Maps string keys in YAML to actual `torch` dtypes:

```python
DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int64": torch.int64,
    "bool": torch.bool,
}
```

##### Error type

The value in `expected_errors` is a string which is mapped to Python exception classes based on the following map:

```python
_ERR_NAME_TO_TYPE = {
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "AssertionError": AssertionError,
}
```

---

### Example YAML File

Below is a sample `template.yaml` illustrating the proposed schema and advanced features such as dtype overrides, skip conditions, error expectations, and presets for non-contiguous layouts.

```yaml
model: template
default:
  dtype: fp16        # Default dtype for all cases unless overridden
  seed: 123          # Seed for reproducibility
  rtol: 1.0e-3       # Default relative tolerance
  atol: 1.0e-3       # Default absolute tolerance

cases:
  - name: add_basic
    op: torch.add
    inputs:
      - tensor: {shape: [4, 128, 768], init: randn}
      - tensor: {shape: [4, 128, 768], init: randn}
    description: |
      Example from GPT-oss forward path:
      File: site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:138
      Code: next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]

  - name: op_bool_tensors
    op: torch.logical_and
    inputs:
      - tensor: {shape: [11, 6], dtype: bool}
      - tensor: {shape: [11, 6], dtype: bool}
    description: |
      Ops with randomly generated bool tensors (50% chance of True)

  - name: op_different_dtypes
    op: torch.add
    inputs:
      - tensor: {shape: [11, 6], dtype: fp32, init: randn}
      - tensor: {shape: [11, 6], dtype: int64, init: randint, init_args: {high: 100}}
    description: |
      Ops with args of different dtypes

  - name: mul_tensor_scalar
    op: torch.mul
    inputs:
      - tensor: {shape: [1, 64], init: uniform, init_args: {low: -1.0, high: 1.0}}
      - value: 3.0
    description: |
      Ops with tensor and scalar as args

  - name: mul_inttensor_scalar
    op: torch.mul
    inputs:
      - tensor: {shape: [1, 64], init: randint, init_args: {low: 1, high: 100}}
      - value: 3
    description: |
      Ops with tensor and scalar as args

  - name: aten_view
    op: torch.ops.aten.view
    inputs:
      - tensor: {shape: [1, 64]}
      - value: [2, 32]
    description: |
      Value can accept scalar, list or tuple

  - name: softmax_lastdim
    op: torch.nn.functional.softmax
    attrs: {dim: -1}
    inputs:
      - tensor: {shape: [2, 4, 8], init: arange}

  - name: cat_list
    op: torch.cat
    attrs: {dim: -1}
    inputs:
      - tensor_list:
          - {shape: [2, 3, 4], init: randn}
          - {shape: [2, 3, 5], init: randn}

  - name: view_should_error_on_noncontig
    op: torch.view
    expects_error: {type: RuntimeError, match: view}
    inputs:
      - tensor:
          shape: [2, 3, 8]
          preset: noncontig_slice
          preset_args: {dim: 1, step: 2}
      - value: [2, 24]

  - name: view_should_error_on_noncontig_167
    op: torch.view
    expects_error: {type: RuntimeError, match: view}
    inputs:
      - tensor:
          shape: [2, 3, 8]
          preset: noncontig_slice
          preset_args: {dim: 1, step: 2}
      - value: 32
      - value: -1
      - value: 2880

  - name: contiguous_from_transpose_view
    op: torch.contiguous
    inputs:
      - tensor:
          shape: [2, 3, 8]
          preset: transpose_view
          preset_args: {dim0: 1, dim1: 2}

  - name: rsqrt_skip_example
    op: torch.rsqrt
    skip_if:
      - expr: cfg.test_device.type == 'spyre' and dtype_str == 'fp16'
        reason: Spyre fp16 rsqrt not supported yet
    inputs:
      - tensor: {shape: [64, 1024], init: rand}

  - name: getitem_ellipsis_none_fullslice
    op: torch.getitem
    inputs:
      - tensor:
          shape: [32, 5760]
          init: rand
      - py: (Ellipsis, None, slice(None, None, None))
```

---
## Metrics
Initial (functional) metrics:

- Pass rate per model and per-op.
- Coverage (# distinct ops and input shapes per model; % of enabled opFuncs).
- Deduplication efficiency (skipped duplicates vs unique executed cases).
- Stability trend across nightly runs (failures/new regressions).
- Runtime per case and total (optional for capacity planning).

Future (performance) metrics if enabled:

- Regression detection across releases (op latency deltas).
- Impact of backend optimizations.
- Comparison against baseline (CPU or another backend).

---
## Dependencies

- Existing internal scripts for op & input extraction.
- pytest runner (markers defined in pytest.ini). [docs.pytest.org]
- torch for CPU reference and .pt loading (baseline correctness). [docs.pytorch.org]
