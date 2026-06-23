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

The test matrix is defined in `.github/workflows/_test_matrix.yaml`. This is a reusable workflow called by both `torch_spyre_tests.yaml` (PR/push) and `runtests_nightly.yaml` (scheduled). New test suites are added **only here**, to the **matrix** under `jobs.test.strategy.matrix.suite` — both pipelines pick up the change automatically.

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

Open `.github/workflows/_test_matrix.yaml` and append your entry to the `matrix.suite` list.

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
bash tests/run_test.sh "tests/configs/${CONFIG}" <flags>
```

where `${CONFIG}` is the value of `config` from your matrix entry, and `<flags>` are supplied by the caller workflow (`-v` for PRs, `--cpu-compile -vvv -s` for nightly). No other changes are needed — the matrix handles the rest.

---

## End-to-End Checklist

```
[ ] 1. Write your test file under tests/ (or tests/inductor/, tests/tensors/)
[ ] 2. Create a config YAML under tests/configs/torch_spyre_tests/
        - Set path using ${TORCH_DEVICE_ROOT}
        - Set unlisted_test_mode: mandatory_success
[ ] 3. Add a new entry to the matrix in .github/workflows/_test_matrix.yaml
        - name: <Human Readable Name>
          config: torch_spyre_tests/<your_config>.yaml
[ ] 4. Open a pull request — CI will pick up the new suite automatically
        in both the PR workflow (tests.yml) and the nightly (runtests_nightly.yaml)
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
| Test Execution Trace | `torch_spyre_tests/test_execution_trace_config.yaml` |
| Test Spyre Lazy Silent | `torch_spyre_tests/test_spyre_lazy_silent_config.yaml` |
| Test Stream | `torch_spyre_tests/test_stream_config.yaml` |
| Test Spyre Profiler | `torch_spyre_tests/test_spyre_profiler_config.yaml` |
| Test Prepare Kernel | `torch_spyre_tests/test_prepare_kernel_config.yaml` |
| Test Allocator E2E | `torch_spyre_tests/test_allocator_e2e_config.yaml` |
| Test Launch Jobplan | `torch_spyre_tests/test_launch_jobplan_config.yaml` |
| Inductor Building Blocks | `torch_spyre_tests/inductor/test_building_blocks_config.yaml` |
| Inductor Codegen | `torch_spyre_tests/inductor/test_codegen_config.yaml` |
| Inductor Copy Back Elision | `torch_spyre_tests/inductor/test_copy_back_elision_config.yaml` |
| Inductor Decomp | `torch_spyre_tests/inductor/test_decomp_config.yaml` |
| Inductor FX Passes | `torch_spyre_tests/inductor/test_inductor_fx_passes_config.yaml` |
| Inductor Normalization Scalars | `torch_spyre_tests/inductor/test_normalization_scalars_config.yaml` |
| Inductor Ops Matmul | `torch_spyre_tests/inductor/test_inductor_ops_matmul_config.yaml` |
| Inductor Ops Reductions Keepdim Multidim0 | `torch_spyre_tests/inductor/test_inductor_ops_reduction_keepdim_multidim0_config.yaml` |
| Inductor Ops Reductions Keepdim Multidim1 | `torch_spyre_tests/inductor/test_inductor_ops_reduction_keepdim_multidim1_config.yaml` |
| Inductor Ops Reductions Keepdim Singledim0 | `torch_spyre_tests/inductor/test_inductor_ops_reduction_keepdim_singledim0_config.yaml` |
| Inductor Ops Reductions Keepdim Singledim1 | `torch_spyre_tests/inductor/test_inductor_ops_reduction_keepdim_singledim1_config.yaml` |
| Inductor Ops Reductions Keepdim Index Norm Eager | `torch_spyre_tests/inductor/test_inductor_ops_reduction_keepdim_index_norm_eager_config.yaml` |
| Inductor Ops Reductions Scalar | `torch_spyre_tests/inductor/test_inductor_ops_reduction_scalar_config.yaml` |
| Inductor Ops Pointwise | `torch_spyre_tests/inductor/test_inductor_ops_pointwise_config.yaml` |
| Inductor Ops Misc Shape A | `torch_spyre_tests/inductor/test_inductor_ops_misc_shape_a_config.yaml` |
| Inductor Ops Misc Shape B | `torch_spyre_tests/inductor/test_inductor_ops_misc_shape_b_config.yaml` |
| Inductor Ops Misc Shape C | `torch_spyre_tests/inductor/test_inductor_ops_misc_shape_c_config.yaml` |
| Inductor Ops Misc Compute A | `torch_spyre_tests/inductor/test_inductor_ops_misc_compute_a_config.yaml` |
| Inductor Ops Misc Compute B | `torch_spyre_tests/inductor/test_inductor_ops_misc_compute_b_config.yaml` |
| Inductor Ops Misc Compute C | `torch_spyre_tests/inductor/test_inductor_ops_misc_compute_c_config.yaml` |
| Inductor Ops LX Planning | `torch_spyre_tests/inductor/test_ops_lx_planning_config.yaml` |
| Inductor Scalar | `torch_spyre_tests/inductor/test_inductor_scalar_config.yaml` |
| Inductor Logging | `torch_spyre_tests/inductor/test_logging_config.yaml` |
| Inductor Dedup Constants | `torch_spyre_tests/inductor/test_dedup_constants_config.yaml` |
| Inductor Padding | `torch_spyre_tests/inductor/test_padding_config.yaml` |
| Inductor Overwrite | `torch_spyre_tests/inductor/test_overwrite.yaml` |
| Inductor Restickify | `torch_spyre_tests/inductor/test_restickify_config.yaml` |
| Inductor Scratchpad Patterns | `torch_spyre_tests/inductor/test_scratchpad_patterns_config.yaml` |
| Inductor Scratchpad Use | `torch_spyre_tests/inductor/test_scratchpad_use_config.yaml` |
| Inductor Dtype Scalars | `torch_spyre_tests/inductor/test_dtype_scalars_config.yaml` |
| Inductor Cache | `torch_spyre_tests/inductor/test_cache_config.yaml` |
| Inductor Scratchpad Solver | `torch_spyre_tests/inductor/test_scratchpad_solver_config.yaml` |
| Inductor Coarse Tiling | `torch_spyre_tests/inductor/test_coarse_tiling_config.yaml` |
| Inductor Coarse Tile E2E | `torch_spyre_tests/inductor/test_coarse_tile_e2e_config.yaml` |
| Inductor Unroll Loop Specs | `torch_spyre_tests/inductor/test_unroll_loop_specs_config.yaml` |
| Inductor Work Division Hint | `torch_spyre_tests/inductor/test_work_division_hint_config.yaml` |
| Tensor Coordinates | `torch_spyre_tests/tensors/test_coordinates_config.yaml` |
| Tensor It Space Splits | `torch_spyre_tests/tensors/test_it_space_splits_config.yaml` |
| Tensor Layout | `torch_spyre_tests/tensors/test_tensor_layout_config.yaml` |
| Tensor Resize | `torch_spyre_tests/tensors/test_resize_config.yaml` |
| Tensor Memory Format | `torch_spyre_tests/tensors/test_memory_format_config.yaml` |
