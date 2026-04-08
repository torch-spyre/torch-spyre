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

The orchestrator script lives at `tests/run_test.sh`. Pass it a config YAML as the only required argument - everything else (env vars, root paths, PYTHONPATH) is derived automatically. The configs reside in `tests/configs` directory.

```bash
bash tests/run_test.sh tests/configs/test_suite_config.yaml
```

This will run all test files listed in the config. You can also pass extra pytest flags after the config path:

```bash
bash tests/run_test.sh tests/configs/test_suite_config.yaml -v
bash tests/run_test.sh tests/configs/test_suite_config.yaml -k test_add
```

If you are running from a different working directory, use absolute paths:

```bash
bash /path/to/torch-spyre/tests/run_test.sh /path/to/torch-spyre/tests/configs/test_suite_config.yaml
```

## Configuring which tests to run

Open `tests/configs/test_suite_config.yaml` and edit the `files` section. Comment out, add, or remove file entries to control which test files the runner picks up (Please note that the existing configs can be used as a reference for users to create a new config specific for their use cases):

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

## Token reference

| Token | Resolves to |
|---|---|
| `${TORCH_ROOT}` | PyTorch source tree (auto-discovered as sibling of `torch-spyre`) |
| `${TORCH_DEVICE_ROOT}` | `torch-spyre` source tree (auto-discovered from editable install metadata) |
