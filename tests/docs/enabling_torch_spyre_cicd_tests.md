# Enabling and Adding New Tests to torch-spyre CI/CD pipeline

**Authors:** Anubhav Jana (IBM Research, India), Ashok Pon Kumar Sree Prakash (IBM Research, India)

---

## Overview

This guide explains how to add a new test file to the torch-spyre CI/CD pipeline. The process has two parts:

1. **Write a test suite config** — a YAML file that tells the test runner which test file to execute
2. **Register it in the GitHub Actions workflow** — so it runs automatically on every push, pull request, or merge

---

## Part 1: Writing a Test Suite Config

Test suite configs live under:

```
tests/configs/torch_spyre_tests/
tests/configs/torch_spyre_tests/inductor/
tests/configs/torch_spyre_tests/tensors/
```

Place your config in the appropriate subdirectory based on what the test covers.

### Config Format

```yaml
test_suite_config:
  files:
    - path: ${TORCH_DEVICE_ROOT}/tests/<your_test_file>.py
      unlisted_test_mode: mandatory_success
```

**Fields:**

- `path` — absolute path to the test file, using the `${TORCH_DEVICE_ROOT}` environment variable as the root. This variable is set automatically in the CI environment.
- `unlisted_test_mode` — controls how tests not explicitly listed are treated. Use `mandatory_success` to require all discovered tests in the file to pass.

### Examples

**Top-level test (e.g. `test_device_enum.py`):**

```yaml
test_suite_config:
  files:
    - path: ${TORCH_DEVICE_ROOT}/tests/test_device_enum.py
      unlisted_test_mode: mandatory_success
```

**Inductor test (e.g. `test_cache.py`):**

```yaml
test_suite_config:
  files:
    - path: ${TORCH_DEVICE_ROOT}/tests/inductor/test_cache.py
      unlisted_test_mode: mandatory_success
```

### Naming Convention

Name the config file after the test file it targets, with a `_config` suffix for better understanding:

| Test file | Config file |
|---|---|
| `tests/test_foo.py` | `torch_spyre_tests/test_foo_config.yaml` |
| `tests/inductor/test_bar.py` | `torch_spyre_tests/inductor/test_bar_config.yaml` |
| `tests/tensors/test_baz.py` | `torch_spyre_tests/tensors/test_baz_config.yaml` |

---

## Part 2: Registering the Test in the GitHub Actions Workflow

The CI/CD pipeline is defined in `.github/workflows/torch-spyre-tests.yml`. New test suites are added to the **matrix** under `jobs.test.strategy.matrix.suite`.

### Workflow Matrix Structure

```yaml
strategy:
  fail-fast: false
  matrix:
    suite:
      - name: <Human Readable Name>
        config: <path/to/your_config.yaml>
```

- `fail-fast: false` — by default GitHub Actions cancels all in-progress matrix jobs the moment any one job fails. Setting this to `false` disables that behaviour so **all suites always run to completion**, even if one fails. This is intentional since we want the full picture of what passed and what failed across every suite, not just the first failure.

- `name` — the label shown in the GitHub Actions UI for this job
- `config` — path to your config file, **relative to** `tests/configs/`

### How to Add Your Test

Open `.github/workflows/torch-spyre-tests.yml` and append your entry to the `matrix.suite` list.

**Example — adding a new top-level test:**

```yaml
matrix:
  suite:
    # ... existing entries ...

    - name: Test Cache          # shown in GHA UI
      config: torch_spyre_tests/test_cache_config.yaml
```

**Example — adding a new inductor test:**

```yaml
    - name: Inductor Cache
      config: torch_spyre_tests/inductor/test_cache_config.yaml
```

**Example — adding a new tensor test:**

```yaml
    - name: Tensor Shapes
      config: torch_spyre_tests/tensors/test_shapes_config.yaml
```

### How the Workflow Runs Your Config

Each matrix entry spawns an independent job. The `Run tests` step invokes the test runner as:

```bash
bash tests/run_test.sh "tests/configs/${CONFIG}" -v
```

where `${CONFIG}` is the value of `config` from your matrix entry. No other changes to the workflow are needed -- the matrix handles the rest.

---

## End-to-End Checklist

```
[ ] 1. Write your test file under tests/ (or tests/inductor/, tests/tensors/)
[ ] 2. Create a config YAML under tests/configs/torch_spyre_tests/
        - Set path using ${TORCH_DEVICE_ROOT}
        - Set unlisted_test_mode: mandatory_success
[ ] 3. Add a new entry to the matrix in .github/workflows/torch-spyre-tests.yml
        - name: <Human Readable Name>
          config: torch_spyre_tests/<your_config>.yaml
[ ] 4. Open a pull request — CI will pick up the new suite automatically
```

---

## Reference: Existing Matrix Entries

The table below shows the currently registered suites and their configs as a reference:

| Name | Config |
|---|---|
| Test Device Enum | `torch_spyre_tests/test_device_enum_config.yaml` |
| Test Fallbacks | `torch_spyre_tests/test_fallbacks_config.yaml` |
| Test Modules | `torch_spyre_tests/test_modules_config.yaml` |
| Test Regex | `torch_spyre_tests/test_regex_config.yaml` |
| Test Spyre | `torch_spyre_tests/test_spyre_config.yaml` |
| Test Spyre Lazy Silent | `torch_spyre_tests/test_spyre_lazy_silent_config.yaml` |
| Test Stream | `torch_spyre_tests/test_stream_config.yaml` |
| Inductor Building Blocks | `torch_spyre_tests/inductor/test_building_blocks_config.yaml` |
| Inductor Codegen | `torch_spyre_tests/inductor/test_codegen_config.yaml` |
| Inductor Decomp | `torch_spyre_tests/inductor/test_decomp_config.yaml` |
| Inductor FX Passes | `torch_spyre_tests/inductor/test_inductor_fx_passes_config.yaml` |
| Inductor Normalization Scalars | `torch_spyre_tests/inductor/test_normalization_scalars_config.yaml` |
| Inductor Ops | `torch_spyre_tests/inductor/test_inductor_ops_config.yaml` |
| Inductor Ops LX Planning | `torch_spyre_tests/inductor/test_ops_lx_planning_config.yaml` |
| Inductor Scalar | `torch_spyre_tests/inductor/test_inductor_scalar_config.yaml` |
| Inductor Logging | `torch_spyre_tests/inductor/test_logging_config.yaml` |
| Inductor Restickify | `torch_spyre_tests/inductor/test_restickify_config.yaml` |
| Inductor Scratchpad Patterns | `torch_spyre_tests/inductor/test_scratchpad_patterns_config.yaml` |
| Inductor Scratchpad Use | `torch_spyre_tests/inductor/test_scratchpad_use_config.yaml` |
| Tensor Coordinates | `torch_spyre_tests/tensors/test_coordinates_config.yaml` |
| Tensor It Space Splits | `torch_spyre_tests/tensors/test_it_space_splits_config.yaml` |
| Tensor Layout | `torch_spyre_tests/tensors/test_tensor_layout_config.yaml` |
