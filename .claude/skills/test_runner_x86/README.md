# Test Runner Skill

A skill for logging into Spyre/OpenShift pods and executing PyTorch tests using the Spyre Test Framework.

## Quick Start

```bash
# Run tests from a single config file
./run_tests.sh <namespace> <pod_name> test_configs/test_nn.yaml

# Run all tests in a directory
./run_tests.sh <namespace> <pod_name> test_configs/

# Run with streaming logs
./run_tests.sh <namespace> <pod_name> test_configs/ --stream

# Run with custom timeout (in seconds)
./run_tests.sh <namespace> <pod_name> test_configs/ --timeout 7200
```

## Files

- **SKILL.md** - Detailed skill documentation and usage guide
- **run_tests.sh** - Shell wrapper script for easy execution
- **run_tests.py** - Core Python implementation for test execution
- **README.md** - This file

## Prerequisites

1. **Running Pod**: A pod must be running in your OpenShift cluster (use pod-creator skill)
2. **OpenShift CLI**: `oc` command must be installed and configured
3. **Python 3**: Required for running the test execution script
4. **Test Configs**: YAML test configuration files (from upstream-test-discovery skill)

## Workflow Integration

This skill integrates with other Spyre skills:

```
1. pod-creator skill → Create a pod
2. upstream-test-discovery skill → Generate test configs
3. test_runner skill (this) → Execute tests on the pod
```

## Output

Test results are saved to `test_results/` directory:

```
test_results/
├── summary.json              # Overall test summary
├── test_nn_results.log       # Per-file detailed logs
├── test_inductor_results.log
└── errors.log                # Any errors encountered
```

## Example Workflow

```bash
# Step 1: Ensure you have a running pod (from pod-creator skill)
# Step 2: Generate test configs (from upstream-test-discovery skill)
# Step 3: Run tests
./run_tests.sh default spyre-build-1 test_configs/ --stream

# Step 4: Review results
cat test_results/summary.json
```

## Troubleshooting

If tests fail to execute:

1. **Check pod status**: `oc get pod <pod_name> -n <namespace>`
2. **View pod logs**: `oc logs <pod_name> -n <namespace>`
3. **Verify test framework**: `oc exec <pod_name> -n <namespace> -- ls /workspace/torch-spyre/test_framework`
4. **Check PYTORCH_ROOT**: `oc exec <pod_name> -n <namespace> -- bash -c 'echo $PYTORCH_ROOT'`

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--timeout <seconds>` | Maximum test execution time | 3600 |
| `--stream` | Show logs in real-time | false |
| `--framework-path <path>` | Test framework location in pod | /workspace/torch-spyre/test_framework |
| `--output-dir <path>` | Local results directory | ./test_results |

## Support

For detailed documentation, see [SKILL.md](SKILL.md).
