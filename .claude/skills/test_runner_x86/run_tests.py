#!/usr/bin/env python3
"""
Test runner script for executing PyTorch tests on Spyre/OpenShift pods.
Runs tests directly from pod using pod's internal test configs and run_test.sh.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any


def run_command(cmd: list[str], capture_output: bool = True, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", str(e)


def check_pod_exists(namespace: str, pod_name: str) -> Tuple[bool, str]:
    """Verify pod exists in the namespace."""
    returncode, stdout, stderr = run_command(["oc", "get", "pod", pod_name, "-n", namespace])
    if returncode != 0:
        return False, f"Pod '{pod_name}' not found in namespace '{namespace}'"
    return True, ""


def check_pod_running(namespace: str, pod_name: str) -> Tuple[bool, str]:
    """Verify pod is in Running state."""
    returncode, stdout, stderr = run_command(
        ["oc", "get", "pod", pod_name, "-n", namespace, "-o", "jsonpath={.status.phase}"]
    )
    if returncode != 0:
        return False, f"Failed to get pod status: {stderr}"
    
    phase = stdout.strip()
    if phase != "Running":
        return False, f"Pod is in '{phase}' state, not Running"
    return True, ""


def check_pod_ready(namespace: str, pod_name: str) -> Tuple[bool, str]:
    """Verify pod containers are ready."""
    returncode, stdout, stderr = run_command(
        ["oc", "get", "pod", pod_name, "-n", namespace, 
         "-o", "jsonpath={.status.conditions[?(@.type=='Ready')].status}"]
    )
    if returncode != 0:
        return False, f"Failed to check pod readiness: {stderr}"
    
    ready = stdout.strip()
    if ready != "True":
        return False, "Pod containers are not ready"
    return True, ""


def verify_test_config_in_pod(namespace: str, pod_name: str, test_config_path: str) -> Tuple[bool, str]:
    """Verify test config exists in the pod."""
    returncode, stdout, stderr = run_command(
        ["oc", "exec", pod_name, "-n", namespace, "--", "test", "-e", test_config_path]
    )
    if returncode != 0:
        return False, f"Test config not found at {test_config_path} in pod"
    return True, ""


def execute_test_in_pod(
    namespace: str,
    pod_name: str,
    test_config_path: str,
    framework_path: str,
    timeout: int,
    stream_logs: bool = False
) -> Tuple[bool, str, str]:
    """Execute test using Spyre Test Framework in the pod."""
    # Construct the test execution command using torch-spyre's run_test.sh
    # This follows the exact sequence from the user's working command to avoid dxp_standalone errors
    test_cmd = (
        f"source $HOME/.bashrc && "
        f"source /etc/profile.d/ibm-aiu-setup.sh && "
        f"source /home/senuser/torch-spyre/.venv/bin/activate && "
        f"export TORCH_ROOT=/home/senuser/pytorch && "
        f'export "TORCHINDUCTOR_FORCE_DISABLE_CACHES"=1 && '
        f'export "TORCH_SPYRE_DEBUG"=1 && '
        f'export "TORCH_COMPILE_DEBUG"=1 && '
        f"cd {framework_path} && "
        f"bash run_test.sh {test_config_path}"
    )
    
    if stream_logs:
        # Stream logs in real-time
        print(f"\n{'='*80}")
        print(f"Executing tests in pod (streaming logs)...")
        print(f"{'='*80}\n")
        
        try:
            process = subprocess.Popen(
                ["oc", "exec", pod_name, "-n", namespace, "--", "bash", "-c", test_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)
            
            process.wait(timeout=timeout)
            output = ''.join(output_lines)
            
            return process.returncode == 0, output, ""
        except subprocess.TimeoutExpired:
            process.kill()
            return False, "", f"Test execution timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Error during test execution: {str(e)}"
    else:
        # Execute without streaming
        returncode, stdout, stderr = run_command(
            ["oc", "exec", pod_name, "-n", namespace, "--", "bash", "-c", test_cmd],
            timeout=timeout
        )
        
        return returncode == 0, stdout, stderr


def parse_test_results(output: str) -> Dict[str, Any]:
    """Parse pytest output to extract test results."""
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "xfailed": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # Look for pytest summary line
    for line in output.split('\n'):
        if 'passed' in line or 'failed' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit():
                    count = int(part)
                    if i + 1 < len(parts):
                        status = parts[i + 1].rstrip(',')
                        if status == 'passed':
                            results['passed'] = count
                        elif status == 'failed':
                            results['failed'] = count
                        elif status == 'xfailed':
                            results['xfailed'] = count
                        elif status == 'skipped':
                            results['skipped'] = count
                        elif status == 'error' or status == 'errors':
                            results['errors'] = count
    
    results['total'] = (results['passed'] + results['failed'] + 
                       results['xfailed'] + results['skipped'] + results['errors'])
    
    return results


def main():
    """Main entry point for test runner."""
    if len(sys.argv) < 4:
        print("Usage: run_tests.py <namespace> <pod_name> <test_config_path_in_pod> [options]")
        print("Options:")
        print("  --timeout <seconds>     Test execution timeout (default: 3600)")
        print("  --stream                Stream logs in real-time")
        print("  --framework-path <path> Path to test framework in pod (default: /home/senuser/torch-spyre/tests)")
        print("  --output-dir <path>     Local directory for results (default: ./test_results)")
        sys.exit(1)
    
    namespace = sys.argv[1]
    pod_name = sys.argv[2]
    test_config_path = sys.argv[3]  # Path inside the pod
    
    # Parse options
    timeout = 3600
    stream_logs = False
    framework_path = "/home/senuser/torch-spyre/tests"
    output_dir = Path("./test_results")
    
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "--timeout" and i + 1 < len(sys.argv):
            timeout = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--stream":
            stream_logs = True
            i += 1
        elif sys.argv[i] == "--framework-path" and i + 1 < len(sys.argv):
            framework_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    result = {"success": False, "message": ""}
    start_time = time.time()
    
    try:
        # Step 1: Validate pod exists
        print(f"Checking if pod '{pod_name}' exists in namespace '{namespace}'...")
        valid, msg = check_pod_exists(namespace, pod_name)
        if not valid:
            result["message"] = msg
            print(json.dumps(result))
            sys.exit(1)
        
        # Step 2: Validate pod is running
        print("Checking pod status...")
        valid, msg = check_pod_running(namespace, pod_name)
        if not valid:
            result["message"] = msg
            print(json.dumps(result))
            sys.exit(1)
        
        # Step 3: Validate pod is ready
        print("Checking pod readiness...")
        valid, msg = check_pod_ready(namespace, pod_name)
        if not valid:
            result["message"] = msg
            print(json.dumps(result))
            sys.exit(1)
        
        # Step 4: Verify test config exists in pod
        print(f"Verifying test config at {test_config_path} in pod...")
        valid, msg = verify_test_config_in_pod(namespace, pod_name, test_config_path)
        if not valid:
            result["message"] = msg
            print(json.dumps(result))
            sys.exit(1)
        
        # Step 5: Execute tests
        config_name = Path(test_config_path).name
        
        print(f"\n{'='*80}")
        print(f"Executing tests from: {config_name}")
        print(f"{'='*80}")
        
        success, stdout, stderr = execute_test_in_pod(
            namespace, pod_name, test_config_path, framework_path, timeout, stream_logs
        )
        
        # Parse results
        test_result = parse_test_results(stdout)
        
        # Save test log
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"{Path(test_config_path).stem}_results.log"
        with open(log_file, 'w') as f:
            f.write(f"Test Config: {test_config_path}\n")
            f.write(f"{'='*80}\n\n")
            f.write(stdout)
            if stderr:
                f.write(f"\n\nErrors:\n{stderr}")
        
        print(f"\nResults: {test_result['passed']} passed, {test_result['failed']} failed, "
              f"{test_result['xfailed']} xfailed, {test_result['skipped']} skipped")
        print(f"Detailed log saved to: {log_file}")
        
        # Generate summary
        duration = time.time() - start_time
        summary = {
            "total_tests": test_result['total'],
            "passed": test_result['passed'],
            "failed": test_result['failed'],
            "xfailed": test_result['xfailed'],
            "skipped": test_result['skipped'],
            "errors": test_result['errors'],
            "duration_seconds": duration,
            "pod_name": pod_name,
            "namespace": namespace,
            "test_config": test_config_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, indent=2, fp=f)
        
        print(f"\nSummary report written to: {summary_file}")
        
        # Report final status
        result["success"] = test_result['failed'] == 0 and test_result['errors'] == 0
        result["message"] = f"Test execution completed. Results saved to {output_dir}"
        result["summary"] = summary
        
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)
        
    except Exception as e:
        result["message"] = f"Unexpected error: {str(e)}"
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()