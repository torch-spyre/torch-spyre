# Spyre Test Framework Runner
- Authors: Anubhav Jana, Ashok Pon Kumar Sree Prakash (IBM Research, India)

## Prerequisites

- Access to a Spyre-enabled pod
- `torch_spyre` and `PyTorch` installed in your active virtualenv following the installation guide from `torch-spyre-docs`

## Setup

Log in to your Spyre-enabled pod and `cd` to the `torch-spyre` directory. The path will depend on where the repo is checked out on your pod — use either a relative or absolute path accordingly.

```bash
cd /path/to/torch-spyre
```

## Running tests

### Approach 1: make (recommended)

The Makefile at the repo root is the primary entry point. By default it runs all torch spyre tests (except distributed)

```bash
make tests
```

You can override which configs to run and pass extra pytest flags via `TEST_CONFIGS` and `PYTEST_ARGS`:

```bash
# Run a specific config directory or file
make tests TEST_CONFIGS="tests/configs/torch_spyre_tests/inductor"
make tests TEST_CONFIGS="tests/configs/upstream_tests/test_view_ops_config.yaml"

# Pass extra pytest flags
make tests PYTEST_ARGS="-v -k test_add"

# Run only mandatory_success tests (excludes xfail)
make tests PYTEST_ARGS="-v -m 'not xfail'"

# Combine overrides
make tests TEST_CONFIGS="tests/configs/upstream_tests/test_view_ops_config.yaml" PYTEST_ARGS="-v -m 'not xfail'"
```

Run `make help` to see all available targets.

### Approach 2: run_test.sh directly

The orchestrator script lives at `tests/run_test.sh`. Pass it a config YAML as the only required argument — everything else (env vars, root paths, PYTHONPATH) is derived automatically. The configs reside in the `tests/configs/` directory.

```bash
bash tests/run_test.sh tests/configs/upstream_tests/test_view_ops_config.yaml
```

You can pass extra pytest flags after the config path:

```bash
bash tests/run_test.sh tests/configs/upstream_tests/test_view_ops_config.yaml -v

# Run only mandatory_success tests (excludes xfail)
bash tests/run_test.sh tests/configs/upstream_tests/test_view_ops_config.yaml -v -m 'not xfail'
```

Multiple config files or directories can be passed and are merged at runtime:

```bash
bash tests/run_test.sh tests/configs/module_tests tests/configs/torch_spyre_tests
```

If you are running from a different working directory, use absolute paths:

```bash
bash /path/to/torch-spyre/tests/run_test.sh /path/to/torch-spyre/tests/configs/torch_spyre_tests/inductor/test_inductor_ops_config.yaml
```

## Configuring which tests to run (Example)

Open `tests/configs/torch_spyre_tests/inductor/test_inductor_ops_config.yaml` and edit the `files` section. Comment out, add, or remove file entries to control which test files the runner picks up (please note that the existing configs can be used as a reference for users to create a new config specific to their use cases):

```yaml
files:
  - path: ${TORCH_ROOT}/test/test_binary_ufuncs.py
    unlisted_test_mode: skip
    tests: []

  # - path: ${TORCH_ROOT}/test/test_ops.py   # uncomment to enable
  #   unlisted_test_mode: skip
  #   tests: []
```

Glob patterns are supported:

```yaml
  - path: ${TORCH_DEVICE_ROOT}/tests/**/*.py
```

For guidance on adding new tests or enabling them in CI/CD, see the [Enabling Torch Spyre CI/CD Tests](https://github.com/torch-spyre/torch-spyre/blob/main/tests/docs/enabling_torch_spyre_cicd_tests.md) guide.

## Token reference

| Token | Resolves to |
|---|---|
| `${TORCH_ROOT}` | PyTorch source tree (auto-discovered as sibling of `torch-spyre`) |
| `${TORCH_DEVICE_ROOT}` | `torch-spyre` source tree (auto-discovered from editable install metadata) |
