---
description: Login to a running Spyre/OpenShift pod and execute PyTorch tests using the Spyre Test Framework. Tests are run directly from the pod using test configurations already present in the pod. Use this skill when you need to run tests on a deployed pod or validate torch-spyre functionality on actual hardware.
---

# Test Runner Skill

This skill automates logging into Spyre/OpenShift pods and running PyTorch tests using the Spyre Test Framework. Tests are executed directly from the pod using test configurations that already exist in the pod.

## What this skill does

1. **Validates pod connectivity**:
   - Checks if the specified pod exists and is running
   - Verifies the pod is in the correct namespace
   - Ensures the pod is ready to accept commands

2. **Verifies test configuration**:
   - Confirms the test config file exists in the pod
   - Validates the test framework path in the pod

3. **Executes tests**:
   - Runs tests using the Spyre Test Framework's run_test.sh
   - Sets up required environment variables (PYTORCH_ROOT, PATH with Spyre toolchain)
   - Captures stdout, stderr, and test results
   - Handles test timeouts and failures gracefully

4. **Collects results**:
   - Saves test logs to local directory
   - Generates a summary report with pass/fail/xfail counts
   - Reports any errors or issues encountered

5. **Provides diagnostics**:
   - Shows pod status and resource usage
   - Displays test execution logs in real-time (optional)
   - Suggests remediation steps for common failures

## When to use this skill

- You have a running pod with torch-spyre installed
- You want to run tests from YAML configs already present in the pod
- You need to validate torch-spyre functionality on Spyre hardware
- You want to execute a specific test configuration
- You need to collect test results and logs from a pod

## Prerequisites

- A running pod in the OpenShift cluster
- `oc` CLI tool installed and configured
- Test configuration YAML files already present in the pod (typically at `/home/senuser/torch-spyre/tests/configs/`)
- Pod must have torch-spyre and PyTorch installed
- Pod must be in Running state

## How to invoke this skill

Simply tell Bob Shell:

- "Run tests on my pod using test_profiler_config.yaml"
- "Execute test_nn.yaml from pod <pod-name>"
- "Run test config /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml on the pod"
- "Run all tests from upstream_tests_beta directory on my pod"

The skill will guide you through parameter collection.

## Parameter details

| Parameter | Purpose | Example |
|-----------|---------|---------|
| **Namespace** | OpenShift namespace where pod is running | `cicd-project`, `my-namespace` |
| **Pod name** | Name of the running pod | `my-torch-pod`, `test-pod-123` |
| **Test config path** | Path to YAML config file **inside the pod** | `/home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml` |
| **Config directory** | Path to directory containing multiple YAML configs (for batch execution) | `/home/senuser/torch-spyre/tests/configs/upstream_tests_beta` |
| **Framework path** | Path to Spyre Test Framework in pod | `/home/senuser/torch-spyre/tests` |
| **Timeout** | Maximum time for test execution (seconds) | `3600`, `7200` |
| **Stream logs** | Whether to show logs in real-time | `true`, `false` |

## Test execution modes

### Single test file
Run a specific test configuration from the pod:
```bash
./run_tests.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml
```

### With custom timeout
Specify a custom timeout (default: 3600 seconds):
```bash
./run_tests.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml --timeout 7200
```

### Stream logs
Show test execution logs in real-time:
```bash
./run_tests.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml --stream
```

### Custom output directory
Save results to a specific local directory:
```bash
./run_tests.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml --output-dir ./my_results
```

### Run all configs in a directory
Run all YAML test configurations from a specific directory:
```bash
./run_all_configs.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests_beta
```
This will automatically discover and run all `.yaml` files in the specified directory.

## Output structure

The skill creates a local `./test_results/` directory with:

```
test_results/
├── summary.json                      # Overall test results summary
└── test_profiler_config_results.log  # Detailed logs for the test
```

**summary.json format:**
```json
{
  "total_tests": 150,
  "passed": 120,
  "failed": 20,
  "xfailed": 10,
  "skipped": 0,
  "errors": 0,
  "duration_seconds": 1234.56,
  "pod_name": "anjali-torch-spyre-may14",
  "namespace": "cicd-project",
  "test_config": "/home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml",
  "timestamp": "2026-05-19T04:20:00Z"
}
```

## Error handling

Common issues and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| **"Pod not found"** | Pod doesn't exist or wrong namespace | Verify pod name: `oc get pods -n <namespace>` |
| **"Pod not running"** | Pod is in Pending/Failed state | Check pod status: `oc describe pod <name> -n <namespace>` |
| **"Test config not found"** | Config path doesn't exist in pod | List configs: `oc exec <pod> -n <namespace> -- ls /home/senuser/torch-spyre/tests/configs/` |
| **"Connection timeout"** | Network issues or pod unresponsive | Check pod logs: `oc logs <pod> -n <namespace>` |
| **"Test execution failed"** | Test errors or framework issues | Review detailed logs in `test_results/` directory |
| **"Permission denied"** | Insufficient cluster permissions | Verify permissions: `oc auth can-i exec pods -n <namespace>` |
| **"dxp_standalone not found"** | Spyre toolchain not properly initialized | Script sources `/etc/profile.d/ibm-aiu-setup.sh` and `$HOME/.bashrc` to set up environment |

## Integration with other skills

This skill works seamlessly with:

1. **pod-creator**: Create a pod first, then run tests on it
2. **upstream-test-discovery**: Discover which tests to run, then execute them with this skill

**Example workflow:**
```bash
# Step 1: Create a pod (using pod-creator skill)
# Step 2: Discover tests (using upstream-test-discovery skill)
# Step 3: Run tests on the pod (using this skill)
./run_tests.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests/test_profiler_config.yaml

# Or run all configs in a directory
./run_all_configs.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests_beta
```

## Advanced usage

### Environment variables
The script automatically sets up the required environment:
- Sources `$HOME/.bashrc` for user environment
- Sources `/etc/profile.d/ibm-aiu-setup.sh` for IBM AIU/Spyre toolchain setup (prevents dxp_standalone errors)
- Activates torch-spyre virtual environment at `/home/senuser/torch-spyre/.venv`
- `TORCH_ROOT=/home/senuser/pytorch`
- `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` - Disables inductor caching for consistent test results
- `TORCH_SPYRE_DEBUG=1` - Enables Spyre debug logging
- `TORCH_COMPILE_DEBUG=1` - Enables torch.compile debug output

### Test result interpretation
- **passed**: Test executed successfully and met all assertions
- **failed**: Test executed but failed assertions or raised exceptions
- **xfailed**: Test was expected to fail (marked with `mode: xfail` in YAML)
- **skipped**: Test was skipped due to conditions or filters
- **errors**: Test encountered errors during execution

## Logs and debugging

All logs are saved to `test_results/` directory:
- Individual test logs show detailed execution traces
- Use `--stream` flag to watch logs in real-time during execution
- Check pod logs for infrastructure issues: `oc logs <pod> -n <namespace>`

## Performance considerations

- Large test suites may take hours to complete
- Use appropriate timeouts based on test complexity
- Monitor pod resource usage during execution
- Tests run with full Spyre hardware acceleration when available

## Common test configurations in pods

Typical test config locations in pods:
- `/home/senuser/torch-spyre/tests/configs/upstream_tests/` - Upstream PyTorch tests
- `/home/senuser/torch-spyre/tests/configs/upstream_tests_beta/` - Beta upstream PyTorch tests
- `/home/senuser/torch-spyre/tests/configs/model_ops_tests/` - Model operation tests
- `/home/senuser/torch-spyre/tests/configs/torch_spyre_tests/` - Spyre-specific tests

List available configs:
```bash
oc exec <pod-name> -n <namespace> -- ls -la /home/senuser/torch-spyre/tests/configs/upstream_tests/
```

Run all configs from a directory:
```bash
./run_all_configs.sh <namespace> <pod-name> /home/senuser/torch-spyre/tests/configs/upstream_tests_beta
```
